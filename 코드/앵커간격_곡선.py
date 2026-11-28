#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""앵커를 몇 칸마다 박아야 하는가 — 간격 대 정확도 곡선.

파일럿(인페인트_일관성_파일럿.py)에서 나온 것:
  - 독립 인페인팅은 뷰마다 다른 것을 그린다 (일관성 −3.58 dB)
  - 앵커 전파는 일관성을 되찾는다 (+2.89 dB)
  - 🔴 그러나 **앵커에서 멀어질수록 정확도가 떨어진다** (23.20 → 19.34 dB)

그래서 남은 질문은 하나다: **앵커 하나가 어디까지 유효한가.**
너무 촘촘하면 앵커끼리 어긋나고(= 독립 인페인팅에 가까워진다), 너무 성기면
멀리서 정확도가 무너진다. 그 사이를 숫자로 찾는다.

재는 방법:
  프레임 N 장을 복도를 따라 늘어놓고, 간격 K 마다 앵커를 박는다.
  앵커 프레임은 SD 로 자유롭게 채우고, 나머지는 **직전 앵커**에서 깊이 워핑으로
  가져온 뒤 새로 드러난 부분만 채운다. K 를 바꿔 가며 두 지표를 본다.

    정확도   채운 것 vs 실사          — 멀어질수록 떨어진다
    일관성   warp(채운 i → j) vs 채운 j — 앵커가 많아질수록 떨어진다

  K=1 은 모든 프레임이 앵커이므로 **독립 인페인팅과 같다**(대조군).
  K=N 은 앵커 하나(파일럿의 앵커 전파)와 같다.

🔴 거리는 COLMAP 단위다. 절대 미터가 아니다. 간극 직선거리를 함께 찍어 비교 기준으로 쓴다.
🔴 SD 호출은 (앵커, 대상) 쌍으로 캐시한다. 같은 쌍은 K 가 달라도 결과가 같다.
"""
import argparse, importlib, json, os, sys
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
P = importlib.import_module("인페인트_일관성_파일럿")

HOME = Path(os.path.expanduser("~"))
SFM = HOME / "data/sfm_all/sparse/1"
FRAMES = HOME / "data/frames_all"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="v1792")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--start", type=int, default=90)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=576)
    ap.add_argument("--box", type=float, default=0.30)
    ap.add_argument("--far", type=float, default=0.85,
                    help="더 먼 영역을 골라야 긴 구간에서 공시야가 유지된다")
    ap.add_argument("--margin", type=int, default=24,
                    help="모든 프레임에서 이 여백 안쪽에 들어오는 점만 가린다")
    ap.add_argument("--spacings", type=int, nargs="+", default=[1, 2, 3, 4, 6, 12])
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1")
    ap.add_argument("--prompt", default="empty indoor hallway, tiled floor, plain painted walls, "
                                        "ceiling lights, realistic photo, consistent perspective")
    ap.add_argument("--negative", default="people, text, watermark, blurry, distorted, duplicated")
    ap.add_argument("--resid-sd", type=float, default=0.005,
                    help="잔여 영역이 마스크의 이 비율을 넘으면 SD 로 채운다 (아니면 TELEA)")
    ap.add_argument("--out", default=str(HOME / "data/앵커간격"))
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    W, H = a.width, a.height

    want = {("%s_%05d.png" % (a.video, a.start + k * a.stride)) for k in range(a.n)}
    print("[0] COLMAP", flush=True)
    poses, obs = P.read_images(SFM / "images.txt", want=want)
    W0, H0, fx0, fy0 = P.read_camera(SFM / "cameras.txt")
    fx, fy = fx0 * W / W0, fy0 * H / H0
    names = [("%s_%05d.png" % (a.video, a.start + k * a.stride)) for k in range(a.n)]
    names = [n for n in names if n in poses and (FRAMES / n).exists()]
    need = {pid for n in names for (_u, _v, pid) in obs.get(n, [])}
    pts3d = P.read_points3D(SFM / "points3D.txt", want=need)
    N = len(names)
    print("  프레임 %d장, 3D 점 %d개" % (N, len(pts3d)), flush=True)

    # 앵커로부터의 이동거리 (COLMAP 단위)
    pos = np.array([poses[n][:3, 3] for n in names])
    step = [float(np.linalg.norm(pos[i] - pos[0])) for i in range(N)]
    print("  프레임 0 기준 이동거리: %s" % ", ".join("%.2f" % s for s in step))

    print("\n[1] 단안 깊이 + 스케일 정렬", flush=True)
    real, depth = {}, {}
    for n in names:
        img = Image.open(FRAMES / n).convert("RGB").resize((W, H), Image.LANCZOS)
        real[n] = np.asarray(img)
        inv = P.mono_depth(img)
        z, used, err = P.align_depth(inv, obs[n], pts3d, poses[n], W, H, fx, fy, W0, H0)
        assert z is not None, "%s COLMAP 점 부족" % n
        depth[n] = z
        print("  %-18s 점 %4d  상대오차 %.2f%%" % (n, used, 100*err), flush=True)

    print("\n[2] 3D 일관 마스크 (먼 쪽 %d%%)" % round((1-a.far)*100), flush=True)
    n0 = names[0]
    bw, bh = int(W*a.box/2), int(H*a.box/2)
    cx, cy = W//2, H//2
    vv, uu = np.meshgrid(np.arange(cy-bh, cy+bh), np.arange(cx-bw, cx+bw), indexing="ij")
    z0 = depth[n0][vv, uu]
    thr = np.quantile(z0, a.far)
    sel = z0 >= thr
    uu, vv, z0 = uu[sel], vv[sel], z0[sel]
    pc = np.stack([(uu-W/2.0)*z0/fx, (vv-H/2.0)*z0/fy, z0], -1).reshape(-1, 3)
    Pw = pc @ poses[n0][:3, :3].T + poses[n0][:3, 3]

    # 🔴 공시야를 **설계로 보장**한다. 첫 시도에서는 프레임 0 의 사각형을 그냥 썼는데
    #    카메라가 돌면 5프레임 만에 마스크가 0% 가 됐다. 모든 프레임 화면 안에
    #    (여백 margin 을 두고) 들어오는 점만 남긴다.
    mg = a.margin
    keep = np.ones(len(Pw), bool)
    for n in names:
        u, v, ok = P.project(Pw, poses[n], fx, fy, W, H)
        keep &= ok & (u >= mg) & (u < W-mg) & (v >= mg) & (v < H-mg)
    print("  모든 프레임에 들어오는 점 %d/%d (%.1f%%)"
          % (keep.sum(), len(Pw), 100*keep.mean()))
    assert keep.sum() >= 200, ("공시야가 남는 점이 %d개뿐이다. 구간이 너무 길거나 "
                               "카메라가 많이 돈다. --n 을 줄이거나 --far 를 키울 것" % keep.sum())
    Pw = Pw[keep]
    try:
        import cv2
    except ImportError:
        cv2 = None
    mask = {}
    for n in names:
        u, v, ok = P.project(Pw, poses[n], fx, fy, W, H)
        m = np.zeros((H, W), bool)
        ui = np.round(u[ok]).astype(int); vi = np.round(v[ok]).astype(int)
        g = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        m[vi[g], ui[g]] = True
        if cv2 is not None:
            m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE,
                                 np.ones((9, 9), np.uint8)).astype(bool)
            m = cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        mask[n] = m
    cover = [100*mask[n].mean() for n in names]
    print("  가린 면적: %s" % ", ".join("%.1f%%" % c for c in cover))
    live = [i for i in range(N) if mask[names[i]].sum() > 500]
    print("  공시야가 남은 프레임: %d/%d (끝 %d번까지)" % (len(live), N, live[-1] if live else -1))

    print("\n[3] 기준선 — 실사끼리 워핑", flush=True)
    base = []
    for i in range(N):
        for j in range(i+1, N):
            w, valid = P.forward_warp(real[names[i]], depth[names[i]],
                                      poses[names[i]], poses[names[j]], fx, fy)
            m = valid & mask[names[j]]
            p = P.psnr(w, real[names[j]], m)
            if not np.isnan(p):
                base.append(p)
    mb = float(np.nanmedian(base))
    print("  기준선 중앙값 %.2f dB (%d쌍)" % (mb, len(base)))

    # ── SD ─────────────────────────────────────────────
    print("\n[4] SDXL 준비", flush=True)
    import torch
    from diffusers import AutoPipelineForInpainting
    pipe = AutoPipelineForInpainting.from_pretrained(
        a.model, torch_dtype=torch.float16, variant="fp16").to("cuda")
    pipe.set_progress_bar_config(disable=True)

    def inpaint(img, msk, seed=a.seed):
        if msk.sum() < 50:
            return img.copy()
        g = torch.Generator("cuda").manual_seed(seed)
        r = pipe(prompt=a.prompt, negative_prompt=a.negative,
                 image=Image.fromarray(img),
                 mask_image=Image.fromarray((msk*255).astype(np.uint8)),
                 width=W, height=H, num_inference_steps=a.steps, strength=0.99,
                 guidance_scale=7.5, generator=g).images[0]
        arr = np.asarray(r.convert("RGB").resize((W, H), Image.LANCZOS))
        return np.where(msk[..., None], arr, img)

    print("  앵커 후보 %d장을 각각 채운다 (캐시)" % N, flush=True)
    anchor_fill = {}
    for i, n in enumerate(names):
        anchor_fill[i] = inpaint(real[n], mask[n])
        print("    %2d %-18s 정확도 %6.2f dB"
              % (i, n, P.psnr(anchor_fill[i], real[n], mask[n])), flush=True)

    prop_cache = {}

    def propagate(ai, j):
        """앵커 ai 의 내용을 프레임 j 로 옮긴다. 새로 드러난 곳만 채운다."""
        if ai == j:
            return anchor_fill[j], 0.0
        key = (ai, j)
        if key in prop_cache:
            return prop_cache[key]
        na, nj = names[ai], names[j]
        w, valid = P.forward_warp(anchor_fill[ai], depth[na], poses[na], poses[nj], fx, fy)
        canvas = real[nj].copy()
        take = valid & mask[nj]
        canvas[take] = w[take]
        resid = mask[nj] & ~valid
        if cv2 is not None and resid.sum():
            resid = cv2.dilate(resid.astype(np.uint8),
                               np.ones((5, 5), np.uint8)).astype(bool) & mask[nj]
        frac = resid.sum() / max(mask[nj].sum(), 1)
        if frac > a.resid_sd:
            canvas = inpaint(canvas, resid)
        elif cv2 is not None and resid.sum():
            canvas = cv2.inpaint(canvas, resid.astype(np.uint8)*255, 5, cv2.INPAINT_TELEA)
        prop_cache[key] = (canvas, float(frac))
        return prop_cache[key]

    # ── 간격별 ─────────────────────────────────────────
    print("\n[5] 간격별 측정", flush=True)
    rows = []
    percurve = {}
    for K in a.spacings:
        filled, resids = {}, []
        for j in range(N):
            ai = (j // K) * K
            img, fr = propagate(ai, j)
            filled[j] = img
            resids.append(fr)
        acc = [P.psnr(filled[j], real[names[j]], mask[names[j]]) for j in live]
        # 🔴 K 마다 "자기 앵커 쌍"을 빼면 재는 쌍 집합이 달라져 비교가 안 된다.
        #    같은 앵커에서 온 쌍(within)과 다른 앵커에서 온 쌍(across)을 나눠 센다.
        #    within 은 정의상 잘 맞고, **across 가 3DGS 가 보게 될 모순**이다.
        win, acr = [], []
        for i in live:
            for j in live:
                if i >= j:
                    continue
                if (j // K) * K == i:        # 앵커 자신 → 정의상 같은 그림
                    continue
                w, valid = P.forward_warp(filled[i], depth[names[i]],
                                          poses[names[i]], poses[names[j]], fx, fy)
                m = valid & mask[names[j]]
                p = P.psnr(w, filled[j], m)
                if np.isnan(p):
                    continue
                ((win if (i // K) == (j // K) else acr)).append(p)
        ma = float(np.nanmedian(acc))
        mw = float(np.nanmedian(win)) if win else float("nan")
        mc = float(np.nanmedian(acr)) if acr else float("nan")
        nanch = len(range(0, N, K))
        rows.append((K, nanch, ma, mw, mc, mc - mb, float(np.mean(resids)*100),
                     len(win), len(acr)))
        percurve[K] = [float(x) for x in acc]
        print("  K=%-3d 앵커 %2d개  정확도 %6.2f  일관성(같은앵커) %6.2f [%2d쌍]  "
              "일관성(다른앵커) %6.2f [%2d쌍]  기준선대비 %+5.2f  잔여 %4.1f%%"
              % (K, nanch, ma, mw, len(win), mc, len(acr), mc - mb,
                 np.mean(resids)*100), flush=True)

    print("\n" + "="*86)
    print("  기준선(실사끼리 워핑) %.2f dB — 워핑 자체의 오차, 일관성의 상한" % mb)
    print("  %-5s %-7s %-9s %-14s %-14s %s"
          % ("간격K", "앵커수", "정확도", "일관성(같은앵커)", "일관성(다른앵커)", "기준선대비"))
    for (K, na_, ma, mw, mc, d, rs, nw, nc) in rows:
        print("  %-5d %-7d %-9.2f %-14.2f %-14.2f %+.2f" % (K, na_, ma, mw, mc, d))
    print("="*86)
    print("  정확도는 앵커가 많을수록 좋아지고, 다른앵커 일관성은 나빠진다 — 이것이 맞바꿈이다.")

    print("\n  앵커로부터의 거리 대 정확도 (K=%d, 앵커 하나)" % a.spacings[-1])
    for idx, i in enumerate(live):
        print("    거리 %5.2f  정확도 %6.2f dB" % (step[i], percurve[a.spacings[-1]][idx]))

    json.dump(dict(names=names, step=step, cover=cover, live=live, baseline=mb,
                   rows=[dict(K=K, anchors=na_, accuracy=ma, within=mw, across=mc,
                              delta=d, residual_pct=rs, n_within=nw, n_across=nc)
                         for (K, na_, ma, mw, mc, d, rs, nw, nc) in rows],
                   curve=percurve),
              open(out / "결과.json", "w"), indent=1)
    print("\n저장: %s" % out)


if __name__ == "__main__":
    main()
