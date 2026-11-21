#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""단안 깊이 추정이 쓸 만한지 먼저 검증한다.

왜 검증부터인가:
  깊이 워핑으로 '가까운 기하적 목표' 를 만들려면 중간 구간의 깊이가 필요하고,
  그건 단안 추정에 의존한다. 추정이 나쁘면 목표 이미지가 틀리고, 틀린 목표를
  조건으로 걸면 모델이 그 틀린 것을 그린다(오늘 가이드 영상에서 겪었다).
  **쓸 만한지 모르면서 파이프라인을 짜지 않는다.**

검증 방법:
  I_A_out 은 G_A 커버리지 안이므로 **스플랫에서 뽑은 깊이**가 있다(출발_깊이.npy).
  단안 추정은 스케일·시프트가 임의이므로 최소제곱으로 d_splat ≈ a·d_mono + b 를
  맞춘 뒤 상대 오차를 본다. 워핑에 필요한 것은 절대값이 아니라 **상대 구조**다.
"""
import os
import numpy as np
from PIL import Image

HOME = os.path.expanduser("~")
SRC = HOME + "/data/워핑/출발_rgb.png"
GT = HOME + "/data/워핑/출발_깊이.npy"
OUT = HOME + "/data/깊이검증"


def main():
    os.makedirs(OUT, exist_ok=True)
    img = Image.open(SRC).convert("RGB")
    gt = np.load(GT)
    print("스플랫 깊이: %s, 범위 %.3f~%.3f" % (gt.shape, gt.min(), gt.max()))

    from transformers import pipeline
    import torch
    mdl = "depth-anything/Depth-Anything-V2-Small-hf"
    print("모델 적재: %s" % mdl)
    pipe = pipeline("depth-estimation", model=mdl,
                    device=0 if torch.cuda.is_available() else -1)
    res = pipe(img)
    # transformers 는 '역깊이(disparity)' 에 가까운 값을 준다. 크기를 맞춘다.
    mono = np.asarray(res["predicted_depth"], dtype=np.float64) \
        if not hasattr(res["predicted_depth"], "numpy") else res["predicted_depth"].numpy()
    mono = np.squeeze(mono)
    if mono.shape != gt.shape:
        mono = np.asarray(Image.fromarray(mono).resize((gt.shape[1], gt.shape[0]),
                                                       Image.BILINEAR))
    print("단안 출력: %s, 범위 %.3f~%.3f" % (mono.shape, mono.min(), mono.max()))

    # Depth Anything 은 가까울수록 값이 크다(역깊이). 두 가정을 다 시험한다.
    best = None
    for name, x in (("깊이로 가정", mono), ("역깊이로 가정", 1.0/np.maximum(mono, 1e-6))):
        A = np.stack([x.ravel(), np.ones(x.size)], 1)
        coef, *_ = np.linalg.lstsq(A, gt.ravel(), rcond=None)
        fit = (A @ coef).reshape(gt.shape)
        err = np.abs(fit - gt)
        rel = err / np.maximum(gt, 1e-6)
        r = np.corrcoef(fit.ravel(), gt.ravel())[0, 1]
        print("  %s : 상관 %.4f, 상대오차 중앙값 %.1f%%, 90분위 %.1f%%"
              % (name, r, 100*np.median(rel), 100*np.percentile(rel, 90)))
        if best is None or r > best[1]:
            best = (name, r, fit, rel)

    name, r, fit, rel = best
    print()
    print("채택: %s (상관 %.4f)" % (name, r))
    for arr, fn in ((gt, "정답_스플랫.png"), (fit, "추정_정렬후.png")):
        a = (arr - arr.min()) / max(1e-6, arr.max()-arr.min())
        Image.fromarray((a*255).astype(np.uint8)).save(os.path.join(OUT, fn))
    e = np.clip(rel, 0, 0.5) / 0.5
    Image.fromarray((e*255).astype(np.uint8)).save(os.path.join(OUT, "상대오차.png"))
    print("저장: %s" % OUT)
    # ── 어느 쪽이 '옳은' 깊이인가 — 기하로 판정한다 ──────────────────
    # 복도 바닥은 평면이다. 양쪽 깊이에서 바닥 영역을 3D 로 펴서 평면에 맞추고
    # 잔차를 비교한다. 잔차가 작은 쪽이 기하적으로 옳다.
    # (스플랫 깊이에는 타일 줄눈·반사가 배어 있어 '정답' 으로 쓰기 어려워 보였다.
    #  추측으로 두지 않고 재서 가린다.)
    H, W = gt.shape
    fy = 869.9 * (H / 576.0)      # 코드/깊이워핑.py 와 같은 내부 파라미터
    fx = 868.9 * (W / 1024.0)
    cx, cy = W / 2.0, H / 2.0
    vv, uu = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    # 바닥 영역: 아래쪽 35%, 좌우 가장자리 15% 제외 (기둥·벽 배제)
    m = np.zeros((H, W), bool)
    m[int(H*0.65):, int(W*0.15):int(W*0.85)] = True

    def plane_rms(d):
        x = (uu - cx) * d / fx
        y = (vv - cy) * d / fy
        P = np.stack([x[m], y[m], d[m]], 1)
        A = np.c_[P[:, :2], np.ones(len(P))]
        coef, *_ = np.linalg.lstsq(A, P[:, 2], rcond=None)
        r = A @ coef - P[:, 2]
        return float(np.sqrt((r**2).mean())), float(np.median(np.abs(P[:, 2])))

    print()
    print("기하 판정 — 복도 바닥은 평면이어야 한다 (아래 35%% 영역, 좌우 가장자리 제외)")
    for nm, d in (("스플랫 깊이", gt), ("단안 추정(정렬 후)", fit)):
        rms, med = plane_rms(d)
        print("  %-18s 평면 잔차 RMS %.4f  (그 영역 깊이 중앙값 %.3f, 비율 %.2f%%)"
              % (nm, rms, med, 100*rms/max(med, 1e-9)))
    print("  → 잔차 비율이 작은 쪽이 기하적으로 옳다.")

    print()
    print("판정 기준: 상대오차 중앙값이 10%% 아래면 짧은 전진 워핑에 쓸 만하다.")
    print("          20%% 를 넘으면 워핑 목표가 틀어져 조건으로 쓸 수 없다.")
    print("          단, '정답' 쪽이 더 나쁘면 이 기준 자체를 다시 봐야 한다.")


if __name__ == "__main__":
    main()
