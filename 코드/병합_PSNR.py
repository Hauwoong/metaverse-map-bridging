#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""렌더된 프레임을 실사 GT1 과 비교한다 (표 3).

🔴 PSNR 만 보면 안 된다 — 오늘 여러 번 속았다.
   빈 맵은 균일한 배경을 그리는데, 실사도 균일한 벽이면 PSNR 이 뜻밖에 높다.
   그래서 **세 가지를 같이** 낸다:
     PSNR          밝기·구조 전체
     색차           평균 RGB 차이 (내용이 바뀌어도 변하지 않을 수 있다 — 함정 5)
     비어있음 비율   렌더가 거의 검은(아무것도 없는) 프레임의 비율
   마지막 것이 "다리가 이음새를 채웠는가" 에 가장 직접적이다.
"""
import argparse, os, glob
import numpy as np
from PIL import Image


def psnr(a, b):
    m = float(np.mean((a-b)**2))
    return 99.0 if m < 1e-12 else float(10*np.log10(1.0/m))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", required=True, help="렌더 프레임 폴더")
    ap.add_argument("--gt", required=True, help="실사 폴더 (960x540)")
    ap.add_argument("--list", required=True, help="프레임 이름 목록 (순서 일치)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    names = [l.strip() for l in open(a.list, encoding="utf-8") if l.strip()]
    rf = sorted(glob.glob(os.path.join(a.render, "*.png")) +
                glob.glob(os.path.join(a.render, "*.jpg")))
    if len(rf) != len(names):
        print("🔴 렌더 %d장 vs 목록 %d장 — 짝이 맞지 않는다" % (len(rf), len(names)))
        n = min(len(rf), len(names)); rf, names = rf[:n], names[:n]

    P, D, E = [], [], 0
    for r, n in zip(rf, names):
        g = os.path.join(a.gt, n)
        if not os.path.exists(g):
            continue
        R = np.asarray(Image.open(r).convert("RGB"), np.float32)/255.
        G = np.asarray(Image.open(g).convert("RGB").resize(
            (R.shape[1], R.shape[0])), np.float32)/255.
        P.append(psnr(R, G))
        D.append(float(np.abs(R.mean((0, 1)) - G.mean((0, 1))).mean()))
        if R.mean() < 0.02:
            E += 1
    if not P:
        print("비교할 짝이 없다"); return
    P, D = np.array(P), np.array(D)
    print("%-6s 프레임 %3d | PSNR 평균 %6.2f  중앙 %6.2f  최소 %6.2f  최대 %6.2f"
          " | 색차 %.3f | 비어있음 %d장 (%.0f%%)"
          % (a.tag, len(P), P.mean(), np.median(P), P.min(), P.max(),
             D.mean(), E, 100*E/len(P)))


if __name__ == "__main__":
    main()
