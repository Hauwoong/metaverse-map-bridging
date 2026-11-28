#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""간극을 SD 로 채워 3DGS 학습 데이터셋을 만든다 — 결정적 시험용.

이것이 새 설계의 본 실험이다:

    A·B 공통 좌표계 → 간극 포즈에서 G_AB 렌더 → 빈 곳을 SD 로 채움 → 3DGS 학습
                           ↑ 포즈를 추정하지 않는다. 생성물에 SfM 을 돌리지 않는다.

지금까지 무너진 단계(생성영상 SfM → Sim(3) 정렬)가 **설계에서 사라진다.**
그 대신 새 숙제가 하나 생긴다: **가우시안 초기 점을 어디서 얻는가.**
예전에는 생성 프레임을 COLMAP 에 넣어 점을 얻었지만 이제 그러지 않는다.
→ **렌더 깊이에서 만든다.** 덮인 영역은 G_AB 의 깊이가 있고, 채운 영역은
  단안 깊이를 **같은 프레임의 덮인 영역으로 스케일 정렬**해 쓴다. 전부 GT 없이 된다.

🔴 GT1(v1791 162~286)은 **포즈만** 쓴다. 그림도, 2D-3D 관측도 쓰지 않는다.
   포즈를 쓰는 것은 '평가자가 카메라를 들고 복도를 걷는 것' 과 같다(플라이스루.py).
   관측점까지 쓰면 간극의 기하가 새어 들어오므로, 채운 프레임의 관측 줄에는
   **우리가 깊이에서 만든 초기 점만** 담는다.

🔴 마스크는 accumulation 알파로 만들지 않는다. 3DGS 가 빈 곳을 불투명 floater 로
   채우므로 알파가 1.000 이다(표 3 '비어있음 0%' 와 같은 이유).
   **A∪B 가 삼각측량한 3D 점이 보이는가** 라는 기하 기준으로 만든다.

🔴 단계를 쪼갠 이유: nerfstudio 는 conda `gs`, diffusers 는 `~/venv/ltx` 에 있고
   한 프로세스에 같이 올릴 수 없다. 환경.sh 의 경고대로 gs 환경은 건드리지 않는다.

    stage1 (gs)   G_AB 렌더 + 커버리지 마스크 → ~/data/간극렌더_*/
    stage2 (ltx)  SD 채움 + 단안깊이 정렬 + 초기점 역투영
    stage3 (ltx)  COLMAP 형식 데이터셋 조립
"""
import argparse, json, os, shutil, sys, importlib
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HOME = Path(os.path.expanduser("~"))
REG = HOME / "data/fixreg2/wp4/out"
IMGS = HOME / "data/fixreg2/wp4/images"


# ───────────────────────── 공용 ─────────────────────────
def q2R(q):
    w, x, y, z = q
    return np.array([[1-2*y*y-2*z*z, 2*x*y-2*z*w,   2*x*z+2*y*w],
                     [2*x*y+2*z*w,   1-2*x*x-2*z*z, 2*y*z-2*x*w],
                     [2*x*z-2*y*w,   2*y*z+2*x*w,   1-2*x*x-2*y*y]])


def read_full(p):
    out = {}
    lines = [l for l in open(p) if not l.startswith("#")]
    for i in range(0, len(lines), 2):
        t = lines[i].split()
        if len(t) < 10:
            continue
        pts = []
        if i+1 < len(lines):
            v = lines[i+1].split()
            for k in range(0, len(v)-2, 3):
                pts.append((float(v[k]), float(v[k+1]), int(v[k+2])))
        R = q2R(list(map(float, t[1:5])))
        tv = np.array(list(map(float, t[5:8])))
        c2w = np.eye(4); c2w[:3, :3] = R.T; c2w[:3, 3] = -R.T @ tv
        out[t[9]] = dict(id=int(t[0]), q=t[1:5], t=t[5:8], cam=int(t[8]),
                         pts=pts, c2w=c2w)
    return out


def idx_of(k):
    return int(k.split("_")[1].split(".")[0])


def project(Pw, c2w, fx, fy, W, H):
    R, t = c2w[:3, :3], c2w[:3, 3]
    pc = (Pw - t) @ R
    z = pc[:, 2]
    ok = z > 1e-6
    u = fx*pc[:, 0]/np.where(ok, z, 1) + W/2.0
    v = fy*pc[:, 1]/np.where(ok, z, 1) + H/2.0
    return u, v, ok


def split_frames(I):
    A  = sorted((k for k in I if k.startswith("v1791_") and idx_of(k) <= 143), key=idx_of)
    GT = sorted((k for k in I if k.startswith("v1791_") and 162 <= idx_of(k) <= 286), key=idx_of)
    B  = sorted((k for k in I if k.startswith("v1793_")), key=idx_of)
    return A, GT, B


def cam_params(W, H):
    t = [l for l in open(REG / "cameras.txt") if not l.startswith("#")][0].split()
    W0, H0 = int(t[2]), int(t[3])
    return W0, H0, float(t[4])*W/W0, float(t[5])*H/H0


# ───────────────────────── stage 1 : gs ─────────────────────────
def stage1(a):
    import torch
    from nerfstudio.utils.eval_utils import eval_setup
    from nerfstudio.cameras.cameras import Cameras, CameraType

    W, H = a.width, a.height
    I = read_full(REG / "images.txt")
    W0, H0, fx, fy = cam_params(W, H)
    A, GT, B = split_frames(I)
    gap = GT[::a.gap_stride]
    print("A %d, GT1 %d (포즈만 → %d장 솎음), B %d" % (len(A), len(GT), len(gap), len(B)))

    ab_pids = {pid for k in A + B for (_u, _v, pid) in I[k]["pts"] if pid != -1}
    pts3d = {}
    for l in open(REG / "points3D.txt"):
        if l.startswith("#"):
            continue
        v = l.split()
        if len(v) >= 8 and int(v[0]) in ab_pids:
            pts3d[int(v[0])] = (float(v[1]), float(v[2]), float(v[3]))
    PW = np.array([pts3d[p] for p in sorted(pts3d)])
    print("A∪B 가 보는 3D 점 %d개" % len(PW), flush=True)

    dp = json.load(open(os.path.join(a.ab_run, "dataparser_transforms.json")))
    M = np.eye(4); M[:3, :4] = np.array(dp["transform"]); s = float(dp["scale"])
    # 🔴 eval_setup 은 Path 를 요구한다 (str 을 주면 .read_text 가 없다고 죽는다)
    _, pipeline, _, _ = eval_setup(Path(a.ab_run) / "config.yml", test_mode="inference")
    model = pipeline.model
    dev = next(model.parameters()).device
    flip = np.diag([1., -1., -1., 1.])

    try:
        import cv2
    except ImportError:
        cv2 = None

    out = Path(a.render_dir); out.mkdir(parents=True, exist_ok=True)
    meta = []
    for n, k in enumerate(gap):
        c2w = I[k]["c2w"]
        T = M @ (c2w @ flip); T[:3, 3] *= s
        cam = Cameras(camera_to_worlds=torch.tensor(T[:3, :4], dtype=torch.float32)[None],
                      fx=fx, fy=fy, cx=W/2, cy=H/2, width=W, height=H,
                      camera_type=CameraType.PERSPECTIVE).to(dev)
        with torch.no_grad():
            o = model.get_outputs_for_camera(cam)
        rgb = (np.clip(o["rgb"].detach().cpu().numpy(), 0, 1)*255).astype(np.uint8)
        dep = (np.squeeze(o["depth"].detach().cpu().numpy()) / s).astype(np.float32)
        acc = np.squeeze(o["accumulation"].detach().cpu().numpy()).astype(np.float32)

        u, v, ok = project(PW, c2w, fx, fy, W, H)
        seen = np.zeros((H, W), np.uint8)
        ui = np.round(u[ok]).astype(int); vi = np.round(v[ok]).astype(int)
        g = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        seen[vi[g], ui[g]] = 1
        if cv2 is not None:
            seen = cv2.dilate(seen, np.ones((a.dilate, a.dilate), np.uint8))
        mask = seen == 0

        np.savez_compressed(out / ("%05d.npz" % idx_of(k)),
                            rgb=rgb, depth=dep, acc=acc, mask=mask,
                            c2w=c2w.astype(np.float64))
        meta.append(dict(name=k, idx=idx_of(k), mask_pct=float(100*mask.mean())))
        print("  %3d/%d %-18s 채울 영역 %5.1f%%" % (n+1, len(gap), k, 100*mask.mean()),
              flush=True)

    json.dump(dict(W=W, H=H, W0=W0, H0=H0, fx=fx, fy=fy,
                   gap=[m["name"] for m in meta], meta=meta),
              open(out / "meta.json", "w"), indent=1)
    print("저장: %s" % out)


# ───────────────────────── stage 2 : ltx ─────────────────────────
def stage2(a):
    import torch
    from diffusers import AutoPipelineForInpainting
    P = importlib.import_module("인페인트_일관성_파일럿")

    rd = Path(a.render_dir)
    meta = json.load(open(rd / "meta.json"))
    W, H, fx, fy = meta["W"], meta["H"], meta["fx"], meta["fy"]
    gap = meta["gap"]

    pipe = AutoPipelineForInpainting.from_pretrained(
        a.model, torch_dtype=torch.float16, variant="fp16").to("cuda")
    pipe.set_progress_bar_config(disable=True)

    def inpaint(img, msk):
        if msk.sum() < 50:
            return img.copy()
        g = torch.Generator("cuda").manual_seed(a.seed)
        r = pipe(prompt=a.prompt, negative_prompt=a.negative,
                 image=Image.fromarray(img),
                 mask_image=Image.fromarray((msk*255).astype(np.uint8)),
                 width=W, height=H, num_inference_steps=a.steps, strength=0.99,
                 guidance_scale=7.5, generator=g).images[0]
        arr = np.asarray(r.convert("RGB").resize((W, H), Image.LANCZOS))
        return np.where(msk[..., None], arr, img)

    try:
        import cv2
    except ImportError:
        cv2 = None

    dump = Path(a.dump); dump.mkdir(parents=True, exist_ok=True)
    pts_out = []
    prev = None
    for n, k in enumerate(gap):
        z = np.load(rd / ("%05d.npz" % idx_of(k)))
        rgb, dep, acc, mask, c2w = (z["rgb"], z["depth"].astype(np.float64),
                                    z["acc"], z["mask"], z["c2w"])

        if a.variant == "anchor" and prev is not None:
            pimg, pdep, pc2w = prev
            w, valid = P.forward_warp(pimg, pdep, pc2w, c2w, fx, fy)
            canvas = rgb.copy()
            take = valid & mask
            canvas[take] = w[take]
            resid = mask & ~valid
            if cv2 is not None and resid.sum():
                resid = cv2.dilate(resid.astype(np.uint8),
                                   np.ones((5, 5), np.uint8)).astype(bool) & mask
            img = inpaint(canvas, resid)
        else:
            img = inpaint(rgb, mask)

        # 채운 영역의 깊이: 단안 → 같은 프레임의 덮인 영역으로 스케일 정렬
        cover = (~mask) & (acc > 0.5) & np.isfinite(dep) & (dep > 1e-3)
        zz = None
        if cover.sum() > 500:
            inv = P.mono_depth(Image.fromarray(img))
            x = inv[cover].astype(np.float64)
            y = 1.0 / dep[cover]
            keep = np.ones(len(x), bool)
            for _ in range(3):                      # 잔차 임계. 비율 절삭 금지
                Mx = np.stack([x[keep], np.ones(keep.sum())], 1)
                sol, *_ = np.linalg.lstsq(Mx, y[keep], rcond=None)
                c0, c1 = sol
                r = np.abs(c0*x + c1 - y)
                med = np.median(r[keep]); mad = np.median(np.abs(r[keep]-med))
                new = r <= max(med + 3*1.4826*mad, 1e-12)
                if new.sum() < 200 or (new == keep).all():
                    break
                keep = new
            Mx = np.stack([x[keep], np.ones(keep.sum())], 1)
            sol, *_ = np.linalg.lstsq(Mx, y[keep], rcond=None)
            c0, c1 = sol
            zz = 1.0 / np.maximum(c0*inv + c1, 1e-6)

        made = 0
        if zz is not None and mask.any():
            sub = np.zeros((H, W), bool)
            sub[::a.init_stride, ::a.init_stride] = True
            hi = float(np.percentile(zz[mask], 97))
            pick = sub & mask & (zz > 1e-3) & (zz < hi)
            vv, uu = np.nonzero(pick)
            d = zz[vv, uu]
            pc = np.stack([(uu-W/2.0)*d/fx, (vv-H/2.0)*d/fy, d], -1)
            Pw = pc @ c2w[:3, :3].T + c2w[:3, 3]
            for q in range(len(Pw)):
                pts_out.append(dict(X=[float(t) for t in Pw[q]],
                                    rgb=[int(t) for t in img[vv[q], uu[q]]],
                                    frame=k, u=float(uu[q]), v=float(vv[q])))
            made = len(Pw)

        Image.fromarray(img).save(dump / ("fill_%05d.png" % idx_of(k)))
        prev = (img, zz if zz is not None else dep, c2w)
        print("  %3d/%d %-18s 채움 %5.1f%%  덮임 %5.1f%%  초기점 %4d"
              % (n+1, len(gap), k, 100*mask.mean(), 100*cover.mean(), made), flush=True)

    json.dump(pts_out, open(dump / "초기점.json", "w"))
    print("채운 프레임 %d장, 초기점 %d개 → %s" % (len(gap), len(pts_out), dump))


# ───────────────────────── stage 3 ─────────────────────────
def stage3(a):
    rd = Path(a.render_dir)
    meta = json.load(open(rd / "meta.json"))
    W, H, W0, H0 = meta["W"], meta["H"], meta["W0"], meta["H0"]
    gap = meta["gap"]
    dump = Path(a.dump)
    newpts = json.load(open(dump / "초기점.json"))

    I = read_full(REG / "images.txt")
    A, GT, B = split_frames(I)

    ab_pids = {pid for k in A + B for (_u, _v, pid) in I[k]["pts"] if pid != -1}
    pts3d, pcol = {}, {}
    for l in open(REG / "points3D.txt"):
        if l.startswith("#"):
            continue
        v = l.split()
        if len(v) >= 8 and int(v[0]) in ab_pids:
            pts3d[int(v[0])] = (float(v[1]), float(v[2]), float(v[3]))
            pcol[int(v[0])] = (int(v[4]), int(v[5]), int(v[6]))

    out = Path(a.out)
    shutil.rmtree(out, ignore_errors=True)
    (out / "images").mkdir(parents=True, exist_ok=True)
    sp = out / "colmap/sparse/0"; sp.mkdir(parents=True, exist_ok=True)

    fill_names = {k: "fill_%05d.png" % idx_of(k) for k in gap}
    for k in gap:
        Image.open(dump / fill_names[k]).convert("RGB").resize(
            (W0, H0), Image.LANCZOS).save(out / "images" / fill_names[k])
    for k in A + B:
        if (IMGS / k).exists():
            shutil.copy(IMGS / k, out / "images" / k)
    keep_real = [k for k in A + B if (out / "images" / k).exists()]
    order = keep_real + [fill_names[k] for k in gap]
    newid = {nm: i+1 for i, nm in enumerate(order)}

    pid_new = {p: i+1 for i, p in enumerate(sorted(pts3d))}
    base = len(pid_new)
    per_img = defaultdict(list)
    for k in keep_real:
        per_img[k] = [(u, v, pid_new.get(pid, -1)) for (u, v, pid) in I[k]["pts"]]
    rows = [(pid_new[p], pts3d[p], pcol[p], []) for p in sorted(pts3d)]
    for q, d in enumerate(newpts):
        pid = base + q + 1
        fn = fill_names[d["frame"]]
        j = len(per_img[fn])
        per_img[fn].append((d["u"]*W0/W, d["v"]*H0/H, pid))
        rows.append((pid, tuple(d["X"]), tuple(d["rgb"]), [(newid[fn], j)]))

    tracks = defaultdict(list)
    for k in keep_real:
        for j, (_u, _v, pid) in enumerate(I[k]["pts"]):
            if pid in pid_new:
                tracks[pid_new[pid]].append((newid[k], j))

    cams = {I[k]["cam"] for k in keep_real} | {I[gap[0]]["cam"]}
    with open(sp/"cameras.txt", "w") as f:
        f.write("# Camera list\n")
        for l in open(REG/"cameras.txt"):
            if not l.startswith("#") and int(l.split()[0]) in cams:
                f.write(l)
    with open(sp/"images.txt", "w") as f:
        f.write("# Image list\n")
        for nm in order:
            k = nm if nm in I else next(g for g in gap if fill_names[g] == nm)
            d = I[k]
            f.write("%d %s %s %d %s\n" % (newid[nm], " ".join(d["q"]),
                                          " ".join(d["t"]), d["cam"], nm))
            f.write(" ".join("%.3f %.3f %d" % (u, v, pid)
                             for (u, v, pid) in per_img[nm]) + "\n")
    with open(sp/"points3D.txt", "w") as f:
        f.write("# 3D point list\n")
        for (pid, X, c, extra) in rows:
            tr = tracks.get(pid, []) + extra
            f.write("%d %.6f %.6f %.6f %d %d %d 1.0 %s\n"
                    % (pid, X[0], X[1], X[2], c[0], c[1], c[2],
                       " ".join("%d %d" % t for t in tr)))
    print("이미지 %d장 (실촬영 %d + 채움 %d)" % (len(order), len(keep_real), len(gap)))
    print("3D 점 %d개 (A∪B %d + 새 초기점 %d)" % (len(rows), base, len(newpts)))
    print("→ %s" % out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["1", "2", "3"])
    ap.add_argument("--variant", choices=["indep", "anchor"], default="indep")
    ap.add_argument("--ab-run", default=str(HOME/"data/runs/G_AB/splatfacto/run1"))
    ap.add_argument("--gap-stride", type=int, default=2)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--dilate", type=int, default=15)
    ap.add_argument("--init-stride", type=int, default=12)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1")
    ap.add_argument("--prompt", default="empty indoor hallway, tiled floor, plain painted walls, "
                                        "ceiling lights, realistic photo, consistent perspective")
    ap.add_argument("--negative", default="people, text, watermark, blurry, distorted, duplicated")
    ap.add_argument("--render-dir", default=str(HOME/"data/간극렌더"))
    ap.add_argument("--dump", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    a.dump = a.dump or str(HOME / ("data/간극채움_%s" % a.variant))
    a.out = a.out or str(HOME / ("data/scenes/SD_%s" % a.variant))
    {"1": stage1, "2": stage2, "3": stage3}[a.stage](a)


if __name__ == "__main__":
    main()
