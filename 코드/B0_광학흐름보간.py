#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""B0 베이스라인 — 광학 흐름 기반 프레임 보간.

왜 이 베이스라인인가:
  본 방법은 "생성 모델이 중간 공간을 만든다" 는 주장 위에 선다. 그 주장은
  **보간은 만들지 못한다**는 대조가 있어야 성립한다. 보간은 두 프레임 사이를
  워핑·블렌딩할 뿐, 가려져 있던 공간이나 없던 물체를 만들어내지 못한다.

방법 (고전적 양방향 흐름 보간):
  1) A→B, B→A 의 조밀 광학 흐름을 구한다 (Farnebäck)
  2) 시각 t 에서 A 를 t·flow_AB 만큼, B 를 (1−t)·flow_BA 만큼 워핑
  3) (1−t) : t 로 가중 합성

ffmpeg 의 minterpolate 는 2장 입력에서 동작하지 않아 직접 구현했다.
구현을 명시할 수 있다는 이점도 있다 — 보고서에 재현 가능한 형태로 적힌다.

⚠️ 이 베이스라인이 '실패' 하는 것이 정상이고 그것이 요점이다.
   A 와 B 는 시선 방향이 80.1° 다르고 실제로 15.03 단위 떨어져 있다.
   흐름 추정이 성립하는 범위를 한참 벗어나므로 유령·찢어짐이 생긴다.
   **그 실패 양상 자체를 §4 에 보고한다.**
"""
import argparse
import numpy as np
import cv2


def warp(img, flow):
    h, w = flow.shape[:2]
    gx, gy = np.meshgrid(np.arange(w), np.arange(h))
    mx = (gx + flow[..., 0]).astype(np.float32)
    my = (gy + flow[..., 1]).astype(np.float32)
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", required=True)
    ap.add_argument("--last", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=161)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=576)
    a = ap.parse_args()

    A = cv2.resize(cv2.imread(a.first), (a.width, a.height))
    B = cv2.resize(cv2.imread(a.last), (a.width, a.height))
    ga = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(B, cv2.COLOR_BGR2GRAY)

    # 큰 변위를 잡으려 피라미드를 깊게, 윈도를 크게 잡는다.
    kw = dict(pyr_scale=0.5, levels=6, winsize=35, iterations=5,
              poly_n=7, poly_sigma=1.5, flags=0)
    fab = cv2.calcOpticalFlowFarneback(ga, gb, None, **kw)
    fba = cv2.calcOpticalFlowFarneback(gb, ga, None, **kw)
    mag = np.linalg.norm(fab, axis=2)
    print("  흐름 크기: 중앙값 %.1f px, 95%% %.1f px, 최대 %.1f px"
          % (np.median(mag), np.percentile(mag, 95), mag.max()))
    print("  (화면 폭의 %.0f%% — 흐름 추정이 신뢰되는 범위를 넘으면 유령이 생긴다)"
          % (100 * np.percentile(mag, 95) / a.width))

    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"),
                         a.fps, (a.width, a.height))
    for i in range(a.frames):
        t = i / (a.frames - 1)
        wa = warp(A, fab * t)
        wb = warp(B, fba * (1 - t))
        vw.write(np.clip((1 - t) * wa + t * wb, 0, 255).astype(np.uint8))
    vw.release()
    print("  저장: %s (%d프레임)" % (a.out, a.frames))


if __name__ == "__main__":
    main()
