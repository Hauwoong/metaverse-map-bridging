#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SD 인페인팅이 멀티뷰 일관성을 유지하는가 — 하루짜리 판정 실험.

왜 이 실험인가:
  새 설계는 **"A·B 공통 좌표계에서 비어 있는 공간을 SD 로 채운다"** 이다.
  생성물의 포즈를 추정할 필요가 없다 — 우리가 정한 값이다. 그래서 지금까지
  무너진 단계(생성영상 SfM → Sim(3) 정렬)가 **설계에서 사라진다.**

  그러나 하나가 남는다: **프레임마다 다른 것을 채우면 3DGS 가 삼각측량을 못 한다.**
  3DGS 는 여러 뷰에서 같은 점을 봐야 기하를 세운다. 뷰마다 다른 벽을 그려 넣으면
  모순된 관측이 되고, 결과는 흐려지거나 또 floater 다 — 지금 겪고 있는 그 상태다.
  이것은 미관 문제가 아니라 **성립 여부의 문제**이고, 여기서 그것만 잰다.

설계:
  정답이 있는 구간(v1792 = 숨긴 중간 복도)에서 N 장을 고른다.
  **같은 3D 영역**을 모든 뷰에서 가리고(2D 고정 사각형이 아니다), 각 뷰를
  독립적으로 채운 뒤 세 가지를 잰다.

    정확도   채운 것 vs 실사              — 그럴듯하게 채웠는가
    일관성   warp(채운 i → j) vs 채운 j   — 뷰끼리 같은 것을 그렸는가
    기준선   warp(실사 i → j) vs 실사 j   — 워핑 자체의 오차. 일관성의 상한

  판정: 일관성이 기준선에 가까우면 독립 인페인팅으로 충분하다.
        크게 나쁘면 앵커 전파나 반복 데이터셋 업데이트가 필요하다.

🔴 정답(v1792 실사)은 **채점에만** 쓴다. 채우는 쪽에는 한 픽셀도 들어가지 않는다.
🔴 깊이는 단안(Depth Anything V2)을 쓴다. 스플랫 깊이는 이 복도 바닥에서 평면
   잔차 23.75% 로 나빴고 단안은 2.89% 였다(워핑_단안깊이.py 의 기록).
🔴 스케일 정렬의 이상치 제거는 **잔차 임계**로 한다. 비율 절삭은 쓰지 않는다 —
   0.85^6 절삭이 "A 잔차 0.0041" 이라는 가짜 숫자를 만든 적이 있다(11/25 정정).
"""
import argparse, json, os, sys, collections
from pathlib import Path
import numpy as np
from PIL import Image

HOME = Path(os.path.expanduser("~"))
SFM = HOME / "data/sfm_all/sparse/1"
FRAMES = HOME / "data/frames_all"


# ───────────────────────── COLMAP 읽기 ─────────────────────────
def q2R(q):
    w, x, y, z = q
    return np.array([[1-2*y*y-2*z*z, 2*x*y-2*z*w,   2*x*z+2*y*w],
                     [2*x*y+2*z*w,   1-2*x*x-2*z*z, 2*y*z-2*x*w],
                     [2*x*z-2*y*w,   2*y*z+2*x*w,   1-2*x*x-2*y*y]])


def read_images(p, want=None):
    """이름 -> (c2w, [(u, v, point3D_id), ...])  COLMAP(OpenCV) 규약.

    🔴 images.txt 는 이 프로젝트에서 114MB 다. 관측점 줄을 전부 파싱하면 몇 분이
       걸린다. `want` 에 든 프레임의 관측점만 파싱한다 (포즈는 전부 읽는다).
    """
    poses, obs = {}, {}
    pose_line = True
    name = None
    with open(p) as f:
        for l in f:
            if l.startswith("#"):
                continue
            if pose_line:
                t = l.split()
                if len(t) < 10:
                    continue
                R = q2R(list(map(float, t[1:5])))
                tv = np.array(list(map(float, t[5:8])))
                c2w = np.eye(4)
                c2w[:3, :3] = R.T
                c2w[:3, 3] = -R.T @ tv
                name = t[9]
                poses[name] = c2w
                pose_line = False
            else:
                if want is None or name in want:
                    pts = l.split()
                    o = []
                    for k in range(0, len(pts) - 2, 3):
                        pid = int(pts[k+2])
                        if pid != -1:
                            o.append((float(pts[k]), float(pts[k+1]), pid))
                    obs[name] = o
                pose_line = True
    return poses, obs


def read_points3D(p, want=None):
    """want 가 주어지면 그 id 만 담는다 (18MB 파일이라 전부 담을 이유가 없다)."""
    out = {}
    with open(p) as f:
        for l in f:
            if l.startswith("#"):
                continue
            t = l.split(None, 4)
            if len(t) < 4:
                continue
            pid = int(t[0])
            if want is not None and pid not in want:
                continue
            out[pid] = np.array([float(t[1]), float(t[2]), float(t[3])])
    return out


def read_camera(p):
    t = [l for l in open(p) if not l.startswith("#")][0].split()
    return int(t[2]), int(t[3]), float(t[4]), float(t[5])


# ───────────────────────── 단안 깊이 + 스케일 정렬 ─────────────────────────
_PIPE = None


def mono_depth(img):
    """Depth Anything V2. 값이 클수록 가까운 '상대 역깊이' 다."""
    global _PIPE
    from transformers import pipeline
    if _PIPE is None:
        _PIPE = pipeline("depth-estimation",
                         model="depth-anything/Depth-Anything-V2-Small-hf", device=0)
    d = _PIPE(img)["predicted_depth"]
    d = d.numpy() if hasattr(d, "numpy") else np.asarray(d)
    d = np.squeeze(d).astype(np.float64)
    if d.shape != (img.size[1], img.size[0]):
        d = np.asarray(Image.fromarray(d).resize(img.size, Image.BILINEAR))
    return d


def align_depth(inv_rel, obs, pts3d, c2w, W, H, fx, fy, W0, H0):
    """상대 역깊이를 COLMAP 점으로 절대 깊이에 맞춘다.

    모델이 내놓는 값은 '가까울수록 큰' 상대 역깊이다. 역깊이 공간에서
        1/z ~= a * inv_rel + b
    를 최소제곱으로 맞춘다. 역깊이에서 맞추는 것이 표준이고(원근이 선형),
    깊이 공간에서 맞추면 먼 점이 과도하게 지배한다.

    이상치 제거는 잔차 임계(중앙 + 3*1.4826*MAD)로 한다. 비율 절삭 금지.
    """
    R, t = c2w[:3, :3], c2w[:3, 3]
    sx, sy = W / W0, H / H0
    us, vs, zs = [], [], []
    for (u0, v0, pid) in obs:
        P = pts3d.get(pid)
        if P is None:
            continue
        pc = (P - t) @ R          # 카메라 좌표 (OpenCV: z 앞)
        z = pc[2]
        if z <= 1e-6:
            continue
        u, v = u0 * sx, v0 * sy
        if not (0 <= u < W and 0 <= v < H):
            continue
        us.append(u); vs.append(v); zs.append(z)
    if len(zs) < 20:
        return None, 0, np.nan
    ui = np.clip(np.round(us).astype(int), 0, W-1)
    vi = np.clip(np.round(vs).astype(int), 0, H-1)
    x = inv_rel[vi, ui]
    y = 1.0 / np.array(zs)

    keep = np.ones(len(x), bool)
    for _ in range(3):
        A = np.stack([x[keep], np.ones(keep.sum())], 1)
        sol, *_ = np.linalg.lstsq(A, y[keep], rcond=None)
        a, b = sol
        r = np.abs(a * x + b - y)
        med = np.median(r[keep])
        mad = np.median(np.abs(r[keep] - med))
        thr = med + 3 * 1.4826 * mad
        new = r <= max(thr, 1e-12)
        if new.sum() < 20 or (new == keep).all():
            keep = new if new.sum() >= 20 else keep
            break
        keep = new
    A = np.stack([x[keep], np.ones(keep.sum())], 1)
    sol, *_ = np.linalg.lstsq(A, y[keep], rcond=None)
    a, b = sol
    inv_abs = a * inv_rel + b
    z = 1.0 / np.maximum(inv_abs, 1e-6)
    rel_err = float(np.median(np.abs(a * x[keep] + b - y[keep]) / np.maximum(y[keep], 1e-9)))
    return z, int(keep.sum()), rel_err


# ───────────────────────── 워핑 ─────────────────────────
def forward_warp(rgb, depth, C0, C1, fx, fy):
    """C0 의 픽셀을 역투영해 C1 로 재투영한다. OpenCV 규약(z 앞, y 아래).

    워핑_단안깊이.py 의 것을 OpenCV 규약으로 바꾸고 유효 마스크를 함께 돌려준다.
    """
    H, W = depth.shape
    cx, cy = W/2.0, H/2.0
    vv, uu = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    pc = np.stack([(uu-cx)*depth/fx, (vv-cy)*depth/fy, depth], -1).reshape(-1, 3)
    Pw = pc @ C0[:3, :3].T + C0[:3, 3]
    col = rgb.reshape(-1, 3).astype(np.float32)

    R, t = C1[:3, :3], C1[:3, 3]
    p1 = (Pw - t) @ R
    d1 = p1[:, 2]
    ok = d1 > 1e-6
    u1 = fx*p1[:, 0]/np.where(ok, d1, 1) + cx
    v1 = fy*p1[:, 1]/np.where(ok, d1, 1) + cy

    img = np.zeros((H, W, 3), np.float32)
    zb = np.full((H, W), np.inf, np.float32)
    idx = np.where(ok)[0]
    order = idx[np.argsort(-d1[idx])]          # 먼 것부터 — 가까운 것이 이긴다
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            ui = np.round(u1[order]).astype(np.int64) + dx
            vi = np.round(v1[order]).astype(np.int64) + dy
            m = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
            img[vi[m], ui[m]] = col[order][m]
            zb[vi[m], ui[m]] = d1[order][m]
    return img.astype(np.uint8), np.isfinite(zb)


def project(P, c2w, fx, fy, W, H):
    R, t = c2w[:3, :3], c2w[:3, 3]
    pc = (P - t) @ R
    z = pc[:, 2]
    ok = z > 1e-6
    u = fx*pc[:, 0]/np.where(ok, z, 1) + W/2.0
    v = fy*pc[:, 1]/np.where(ok, z, 1) + H/2.0
    return u, v, ok


# ───────────────────────── 지표 ─────────────────────────
def psnr(a, b, m):
    if m.sum() < 50:
        return np.nan
    d = (a[m].astype(np.float64) - b[m].astype(np.float64)) / 255.0
    mse = float((d ** 2).mean())
    return 99.0 if mse < 1e-12 else float(10 * np.log10(1.0 / mse))


# ───────────────────────── 본체 ─────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="v1792", help="정답이 있는 중간 구간")
    ap.add_argument("--n", type=int, default=6, help="고를 프레임 수")
    ap.add_argument("--start", type=int, default=90)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=576)
    ap.add_argument("--box", type=float, default=0.30, help="첫 프레임에서 가릴 중앙 사각형 비율")
    ap.add_argument("--far", type=float, default=0.5,
                    help="사각형 안에서 이 분위수 이상으로 먼 픽셀만 가린다 "
                         "(가까운 곳은 카메라가 금세 지나쳐 공시야가 사라진다)")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1")
    ap.add_argument("--prompt", default="empty indoor hallway, tiled floor, plain painted walls, "
                                        "ceiling lights, realistic photo, consistent perspective")
    ap.add_argument("--negative", default="people, text, watermark, blurry, distorted, duplicated")
    ap.add_argument("--no-sd", action="store_true", help="SD 없이 기준선만 잰다")
    ap.add_argument("--anchor", action="store_true",
                    help="① 앵커 전파 — 한 장만 채우고 워핑해 나머지로 옮긴 뒤 "
                         "새로 드러난 부분만 채운다")
    ap.add_argument("--out", default=str(HOME / "data/인페인트파일럿"))
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    W, H = a.width, a.height

    want = {("%s_%05d.png" % (a.video, a.start + k * a.stride)) for k in range(a.n)}
    print("[0] COLMAP 읽는 중 (images.txt 114MB — 고른 프레임의 관측점만 파싱)", flush=True)
    poses, obs = read_images(SFM / "images.txt", want=want)
    W0, H0, fx0, fy0 = read_camera(SFM / "cameras.txt")
    fx, fy = fx0 * W / W0, fy0 * H / H0

    names = [("%s_%05d.png" % (a.video, a.start + k * a.stride)) for k in range(a.n)]
    names = [n for n in names if n in poses and (FRAMES / n).exists()]
    assert len(names) >= 3, "쓸 수 있는 프레임이 %d 장뿐이다" % len(names)
    need = {pid for n in names for (_u, _v, pid) in obs.get(n, [])}
    pts3d = read_points3D(SFM / "points3D.txt", want=need)
    print("  포즈 %d개, 고른 프레임 %d장, 필요한 3D 점 %d개 (읽음 %d개)"
          % (len(poses), len(names), len(need), len(pts3d)), flush=True)
    print("  %s" % ", ".join(names))
    print("  해상도 %dx%d  fx=%.1f fy=%.1f" % (W, H, fx, fy), flush=True)

    # ── 1. 실사 + 단안 깊이(스케일 정렬) ─────────────────────
    real, depth = {}, {}
    print("\n[1] 단안 깊이 추정 + COLMAP 스케일 정렬")
    for n in names:
        img = Image.open(FRAMES / n).convert("RGB").resize((W, H), Image.LANCZOS)
        real[n] = np.asarray(img)
        inv = mono_depth(img)
        z, used, err = align_depth(inv, obs[n], pts3d, poses[n], W, H, fx, fy, W0, H0)
        assert z is not None, "%s 에 COLMAP 점이 모자라다" % n
        depth[n] = z
        print("  %-18s 점 %4d개  역깊이 상대오차 중앙 %.3f%%  깊이중앙 %.3f"
              % (n, used, 100*err, float(np.median(z))))

    # ── 2. 같은 3D 영역을 모든 뷰에서 가린다 ─────────────────
    print("\n[2] 3D 일관 마스크")
    # 🔴 복도를 걸어가는 카메라는 가까운 영역을 금세 지나친다. 첫 시도에서
    #    stride 6 으로 고르니 3프레임 뒤 가린 영역이 0% 가 됐다(지나가 버림).
    #    공시야를 유지하려면 **먼 쪽**(복도 끝 벽)을 골라야 한다.
    n0 = names[0]
    bw, bh = int(W*a.box/2), int(H*a.box/2)
    cx, cy = W//2, H//2
    vv, uu = np.meshgrid(np.arange(cy-bh, cy+bh), np.arange(cx-bw, cx+bw), indexing="ij")
    z0 = depth[n0][vv, uu]
    thr = np.quantile(z0, a.far)
    sel = z0 >= thr
    print("  첫 프레임 중앙 사각형에서 깊이 상위 %d%% 만 사용 (>= %.2f)"
          % (round((1-a.far)*100), thr))
    uu, vv, z0 = uu[sel], vv[sel], z0[sel]
    pc = np.stack([(uu-W/2.0)*z0/fx, (vv-H/2.0)*z0/fy, z0], -1).reshape(-1, 3)
    P = pc @ poses[n0][:3, :3].T + poses[n0][:3, 3]     # 가릴 3D 점 뭉치

    mask = {}
    for n in names:
        u, v, ok = project(P, poses[n], fx, fy, W, H)
        m = np.zeros((H, W), bool)
        ui = np.round(u[ok]).astype(int); vi = np.round(v[ok]).astype(int)
        g = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        m[vi[g], ui[g]] = True
        try:                                            # 점 사이 틈을 메운다
            import cv2
            m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE,
                                 np.ones((9, 9), np.uint8)).astype(bool)
            m = cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        except ImportError:
            pass
        mask[n] = m
        print("  %-18s 가린 면적 %5.2f%%" % (n, 100*m.mean()))
        Image.fromarray((m*255).astype(np.uint8)).save(out / ("마스크_%s" % n))

    # ── 3. 기준선: 실사끼리 워핑 ─────────────────────────────
    print("\n[3] 기준선 — 실사 i 를 j 로 워핑 (워핑 자체의 오차)")
    base = []
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            if i >= j:
                continue
            w, valid = forward_warp(real[ni], depth[ni], poses[ni], poses[nj], fx, fy)
            m = valid & mask[nj]
            p = psnr(w, real[nj], m)
            base.append(p)
            print("  %d->%d  겹친 가린영역 %5.2f%%   PSNR %6.2f dB" % (i, j, 100*m.mean(), p))
    print("  기준선 중앙값 %.2f dB" % np.nanmedian(base))

    if a.no_sd:
        json.dump(dict(baseline=[float(x) for x in base]),
                  open(out / "결과.json", "w"), indent=1)
        print("\n--no-sd 이므로 여기까지. 저장: %s" % out)
        return

    # ── 4. SD 인페인팅 (각 뷰 독립) ──────────────────────────
    print("\n[4] SDXL 인페인팅 — 각 뷰를 독립적으로 채운다")
    import torch
    from diffusers import AutoPipelineForInpainting
    pipe = AutoPipelineForInpainting.from_pretrained(
        a.model, torch_dtype=torch.float16, variant="fp16").to("cuda")
    pipe.set_progress_bar_config(disable=True)

    def inpaint(base, msk, seed):
        """base 의 msk 영역만 채운다. msk 밖은 그대로 돌려준다."""
        if msk.sum() < 50:
            return base.copy()
        g = torch.Generator("cuda").manual_seed(seed)
        r = pipe(prompt=a.prompt, negative_prompt=a.negative,
                 image=Image.fromarray(base),
                 mask_image=Image.fromarray((msk*255).astype(np.uint8)),
                 width=W, height=H, num_inference_steps=a.steps, strength=0.99,
                 guidance_scale=7.5, generator=g).images[0]
        arr = np.asarray(r.convert("RGB").resize((W, H), Image.LANCZOS))
        return np.where(msk[..., None], arr, base)

    filled = {}
    if not a.anchor:
        for n in names:
            arr = inpaint(real[n], mask[n], a.seed)
            filled[n] = arr
            Image.fromarray(arr).save(out / ("채움_%s" % n))
            print("  %-18s 정확도(실사 대비) %6.2f dB" % (n, psnr(arr, real[n], mask[n])))
    else:
        # ① 앵커 전파 — 기준 프레임 하나만 자유롭게 채우고, 나머지는 그것을 워핑해
        #    가져온 뒤 **새로 드러난 부분만** 채운다. 뷰끼리 같은 것을 그리도록 강제한다.
        n0 = names[0]
        filled[n0] = inpaint(real[n0], mask[n0], a.seed)
        Image.fromarray(filled[n0]).save(out / ("채움_%s" % n0))
        print("  %-18s [앵커] 정확도 %6.2f dB" % (n0, psnr(filled[n0], real[n0], mask[n0])))
        try:
            import cv2
        except ImportError:
            cv2 = None
        for n in names[1:]:
            w, valid = forward_warp(filled[n0], depth[n0], poses[n0], poses[n], fx, fy)
            canvas = real[n].copy()          # 🔴 `base` 로 쓰면 기준선 리스트를 덮어쓴다
            take = valid & mask[n]
            canvas[take] = w[take]
            resid = mask[n] & ~valid
            if cv2 is not None and resid.sum():
                resid = cv2.dilate(resid.astype(np.uint8),
                                   np.ones((5, 5), np.uint8)).astype(bool) & mask[n]
            arr = inpaint(canvas, resid, a.seed)
            filled[n] = arr
            Image.fromarray(arr).save(out / ("채움_%s" % n))
            print("  %-18s 앵커가 덮음 %5.1f%%  새로 채움 %5.1f%%  정확도 %6.2f dB"
                  % (n, 100*take.sum()/max(mask[n].sum(), 1),
                     100*resid.sum()/max(mask[n].sum(), 1),
                     psnr(arr, real[n], mask[n])))

    # ── 5. 일관성 ────────────────────────────────────────────
    print("\n[5] 일관성 — 채운 i 를 j 로 워핑해 채운 j 와 비교")
    cons, acc = [], []
    for n in names:
        acc.append(psnr(filled[n], real[n], mask[n]))
    # 🔴 앵커 모드에서 0->j 는 **정의상 같은 그림**이다 (j 의 내용이 0 을 워핑해 온 것).
    #    99 dB 가 나오고 그것을 중앙값에 넣으면 스스로를 채점하는 꼴이 된다.
    #    앵커가 아닌 프레임끼리(1..N-1)만 센다.
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            if i >= j:
                continue
            w, valid = forward_warp(filled[ni], depth[ni], poses[ni], poses[nj], fx, fy)
            m = valid & mask[nj]
            p = psnr(w, filled[nj], m)
            deg = a.anchor and i == 0
            if not deg:
                cons.append(p)
            print("  %d->%d  PSNR %6.2f dB%s" % (i, j, p, "   (앵커 자신 — 제외)" if deg else ""))

    mb, mc, ma = np.nanmedian(base), np.nanmedian(cons), np.nanmedian(acc)
    print("\n" + "="*64)
    print("  정확도 (채운 것 vs 실사)        중앙 %6.2f dB" % ma)
    print("  일관성 (채운 i→j vs 채운 j)     중앙 %6.2f dB" % mc)
    print("  기준선 (실사 i→j vs 실사 j)     중앙 %6.2f dB" % mb)
    print("  일관성 - 기준선 = %+.2f dB" % (mc - mb))
    print("="*64)
    if mc >= mb - 1.0:
        print("판정: 독립 인페인팅으로도 뷰끼리 거의 같은 것을 그렸다. 그대로 키운다.")
    elif mc >= mb - 4.0:
        print("판정: 어긋난다. 앵커 전파(한 장 채우고 워핑해 조건으로)가 필요하다.")
    else:
        print("판정: 크게 어긋난다. 반복 데이터셋 업데이트까지 가야 한다.")

    json.dump(dict(names=names, accuracy=[float(x) for x in acc],
                   consistency=[float(x) for x in cons],
                   baseline=[float(x) for x in base],
                   median=dict(accuracy=float(ma), consistency=float(mc), baseline=float(mb))),
              open(out / "결과.json", "w"), indent=1)
    print("저장: %s" % out)


if __name__ == "__main__":
    main()
