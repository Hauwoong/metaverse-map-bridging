#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""실사 A 끝에서 출발해 간극을 한 걸음씩 워핑+인페인팅으로 전진한다 — 감쇠 곡선.

왜 설계가 또 바뀌었는가 (2026-11-27 stage1 의 측정 결과):
  "A·B 공통 좌표계에서 G_AB 를 렌더하고 빈 곳을 SD 로 채운다" 를 하려 했는데,
  간극 63 포즈를 전부 렌더해 보니 **메울 '구멍' 이 아니라 전부가 쓰레기**였다.

    accumulation > 0.5  : 63 포즈 전부 **100%**  ← 맵에게 '어디가 비었나' 를 못 묻는다
    G_AB 렌더 vs GT1    : 7.79 ~ 14.30 dB (중앙 12.56)
    간극 길이           : 17.78 (COLMAP 단위) — 파일럿 전파 범위 1.43 의 12배

  → 인페인팅의 맥락으로 쓸 화소가 간극 안에 한 장도 없다. 그러므로 맥락은
    **실사 A(또는 B)** 에서 가져와야 하고, 포즈가 알려져 있으니 워핑으로 옮길 수 있다.

하는 일:
  A 의 마지막 실사 프레임에서 출발해, 알려진 포즈를 따라
    ① 직전 프레임을 다음 포즈로 깊이 워핑
    ② 새로 드러난 곳만 SD 로 채움
    ③ 채운 그림의 단안 깊이를 **워핑된 깊이**로 스케일 정렬해 다음 걸음에 쓴다
  를 반복하고, GT1 포즈마다 실사와 비교해 **거리 대 품질 곡선**을 얻는다.

🔴 이것은 B3 '연쇄 생성'(회전 드리프트로 천장을 보고 정지, 직진성 0.266)과
   구조가 같지만 결정적으로 다르다: **포즈가 주어져 있어 기하는 드리프트하지 않는다.**
   드리프트할 수 있는 것은 외형뿐이고, 그것을 여기서 잰다.

🔴 GT1 은 포즈와 채점에만 쓴다. 그림은 전파 쪽에 한 픽셀도 들어가지 않는다.
"""
import argparse, json, os, sys, importlib
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
P = importlib.import_module("인페인트_일관성_파일럿")
M = importlib.import_module("간극_채움_데이터셋")

HOME = Path(os.path.expanduser("~"))
REG = M.REG
IMGS = M.IMGS


def slerp(Ra, Rb, u):
    Rrel = Ra.T @ Rb
    w = np.clip((np.trace(Rrel)-1)/2, -1, 1)
    ang = np.arccos(w)
    if ang < 1e-8:
        return Ra.copy()
    ax = np.array([Rrel[2, 1]-Rrel[1, 2], Rrel[0, 2]-Rrel[2, 0],
                   Rrel[1, 0]-Rrel[0, 1]])/(2*np.sin(ang))
    aa = ang*u
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return Ra @ (np.eye(3) + np.sin(aa)*K + (1-np.cos(aa))*(K @ K))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=float, default=0.10, help="한 걸음 거리 (COLMAP 단위)")
    ap.add_argument("--max-dist", type=float, default=8.0, help="여기까지만 전진")
    # 🔴 SDXL 은 width·height 가 8의 배수여야 한다. 960x540 은 540/8=67.5 라 거부당한다.
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=576)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1")
    ap.add_argument("--prompt", default="empty indoor hallway, tiled floor, plain painted walls, "
                                        "ceiling lights, handrail, realistic photo")
    ap.add_argument("--negative", default="people, text, watermark, blurry, distorted, duplicated")
    ap.add_argument("--out", default=str(HOME/"data/연쇄전파"))
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    W, H = a.width, a.height

    I = M.read_full(REG/"images.txt")
    W0, H0, fx, fy = M.cam_params(W, H)
    A, GT, B = M.split_frames(I)
    start = A[-1]
    print("출발 %s → GT1 %s~%s" % (start, GT[0], GT[-1]))

    # 궤적: A 마지막 → GT1 첫 (보간) → GT1 따라가기. step 간격으로 리샘플.
    key = [I[start]["c2w"]] + [I[k]["c2w"] for k in GT]
    keyname = [start] + GT
    traj, tag = [], []
    for i in range(len(key)-1):
        Ca, Cb = key[i], key[i+1]
        d = float(np.linalg.norm(Cb[:3, 3]-Ca[:3, 3]))
        n = max(1, int(np.ceil(d / a.step)))
        for j in range(n):
            u = j / n
            T = np.eye(4)
            T[:3, :3] = slerp(Ca[:3, :3], Cb[:3, :3], u)
            T[:3, 3] = (1-u)*Ca[:3, 3] + u*Cb[:3, 3]
            traj.append(T)
            tag.append(keyname[i+1] if j == 0 and i > 0 else None)
    traj.append(key[-1]); tag.append(keyname[-1])
    # 각 GT1 프레임이 궤적의 어느 지점인지 (채점용)
    gt_at = {}
    for t, nm in enumerate(tag):
        if nm is not None and nm not in gt_at:
            gt_at[nm] = t
    dist = [0.0]
    for t in range(1, len(traj)):
        dist.append(dist[-1] + float(np.linalg.norm(traj[t][:3, 3]-traj[t-1][:3, 3])))
    cut = next((t for t, d in enumerate(dist) if d > a.max_dist), len(traj))
    traj, dist = traj[:cut], dist[:cut]
    print("궤적 %d 걸음, 총 %.2f 단위 (한 걸음 %.3f)" % (len(traj), dist[-1], a.step))

    # ── 출발 프레임: 실사 + COLMAP 정렬 단안 깊이 ─────────────
    img0 = Image.open(IMGS/start).convert("RGB").resize((W, H), Image.LANCZOS)
    rgb = np.asarray(img0)
    pts3d = {}
    need = {pid for (_u, _v, pid) in I[start]["pts"] if pid != -1}
    for l in open(REG/"points3D.txt"):
        if l.startswith("#"):
            continue
        v = l.split()
        if len(v) >= 8 and int(v[0]) in need:
            pts3d[int(v[0])] = np.array([float(v[1]), float(v[2]), float(v[3])])
    inv = P.mono_depth(img0)
    z, used, err = P.align_depth(inv, I[start]["pts"], pts3d, I[start]["c2w"],
                                 W, H, fx, fy, W0, H0)
    assert z is not None, "출발 프레임 스케일 정렬 실패"
    print("출발 깊이 정렬: 점 %d개, 역깊이 상대오차 %.2f%%, 깊이중앙 %.2f"
          % (used, 100*err, float(np.median(z))))

    # ── SD ───────────────────────────────────────────────
    import torch
    from diffusers import AutoPipelineForInpainting
    pipe = AutoPipelineForInpainting.from_pretrained(
        a.model, torch_dtype=torch.float16, variant="fp16").to("cuda")
    pipe.set_progress_bar_config(disable=True)
    try:
        import cv2
    except ImportError:
        cv2 = None

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

    def psnr(x, y):
        d = (x.astype(np.float64)-y.astype(np.float64))/255.0
        m = float((d**2).mean())
        return 99.0 if m < 1e-12 else float(10*np.log10(1/m))

    gt_rows = []
    cur_rgb, cur_z, cur_c2w = rgb, z, I[start]["c2w"]
    Image.fromarray(cur_rgb).save(out/"step_000.png")
    for t in range(1, len(traj)):
        w, valid = P.forward_warp(cur_rgb, cur_z, cur_c2w, traj[t], fx, fy)
        hole = ~valid
        if cv2 is not None and hole.sum():
            hole = cv2.dilate(hole.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        img = inpaint(w, hole)

        # 다음 걸음용 깊이: 단안 추정 → **워핑으로 살아남은 영역의 깊이**로 스케일 정렬.
        # 색 대신 깊이를 실어 보내 z-버퍼만 다시 만든다 (forward_warp 과 같은 규약).
        Hh, Ww = cur_z.shape
        cx, cy = Ww/2.0, Hh/2.0
        vv, uu = np.meshgrid(np.arange(Hh), np.arange(Ww), indexing="ij")
        pc = np.stack([(uu-cx)*cur_z/fx, (vv-cy)*cur_z/fy, cur_z], -1).reshape(-1, 3)
        Pw = pc @ cur_c2w[:3, :3].T + cur_c2w[:3, 3]
        R, tt = traj[t][:3, :3], traj[t][:3, 3]
        p1 = (Pw - tt) @ R
        d1 = p1[:, 2]
        ok = d1 > 1e-6
        u1 = fx*p1[:, 0]/np.where(ok, d1, 1) + cx
        v1 = fy*p1[:, 1]/np.where(ok, d1, 1) + cy
        zb = np.full((H, W), np.inf, np.float64)
        idx = np.where(ok)[0]
        order = idx[np.argsort(-d1[idx])]
        ui = np.round(u1[order]).astype(np.int64); vi = np.round(v1[order]).astype(np.int64)
        mm = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        zb[vi[mm], ui[mm]] = d1[order][mm]
        good = np.isfinite(zb)

        inv2 = P.mono_depth(Image.fromarray(img))
        if good.sum() > 500:
            x = inv2[good]; y = 1.0/zb[good]
            keep = np.ones(len(x), bool)
            for _ in range(3):
                Mx = np.stack([x[keep], np.ones(keep.sum())], 1)
                sol, *_ = np.linalg.lstsq(Mx, y[keep], rcond=None)
                c0, c1 = sol
                r = np.abs(c0*x+c1-y)
                med = np.median(r[keep]); mad = np.median(np.abs(r[keep]-med))
                new = r <= max(med+3*1.4826*mad, 1e-12)
                if new.sum() < 200 or (new == keep).all():
                    break
                keep = new
            Mx = np.stack([x[keep], np.ones(keep.sum())], 1)
            sol, *_ = np.linalg.lstsq(Mx, y[keep], rcond=None)
            c0, c1 = sol
            nz = 1.0/np.maximum(c0*inv2+c1, 1e-6)
        else:
            nz = np.where(good, zb, np.nanmedian(cur_z))

        cur_rgb, cur_z, cur_c2w = img, nz, traj[t]

        line = "  %3d/%d 거리 %5.2f  새로드러남 %5.1f%%" % (t, len(traj)-1, dist[t],
                                                       100*hole.mean())
        nm = tag[t] if t < len(tag) else None
        if nm is not None and (IMGS/nm).exists():
            g = np.asarray(Image.open(IMGS/nm).convert("RGB").resize((W, H), Image.LANCZOS))
            p = psnr(img, g)
            gt_rows.append(dict(frame=nm, step=t, dist=dist[t], psnr=p))
            line += "   vs GT1 %6.2f dB" % p
            if len(gt_rows) % 5 == 1:
                Image.fromarray(img).save(out/("step_%03d.png" % t))
        print(line, flush=True)

    print("\n" + "="*70)
    if gt_rows:
        print("  거리 대 품질 (GT1 포즈에서만)")
        for r in gt_rows[::max(1, len(gt_rows)//14)]:
            print("    거리 %5.2f  %6.2f dB   %s" % (r["dist"], r["psnr"], r["frame"]))
        ps = [r["psnr"] for r in gt_rows]
        print("  중앙 %.2f dB  처음 %.2f  마지막 %.2f" % (np.median(ps), ps[0], ps[-1]))
    print("  참고: G_AB 렌더의 같은 구간 품질은 중앙 12.56 dB 였다")
    print("="*70)
    json.dump(gt_rows, open(out/"곡선.json", "w"), indent=1)
    print("저장: %s" % out)


if __name__ == "__main__":
    main()
