#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""B0 베이스라인과 생성 방법들을 같은 자로 비교한다.

중간 프레임의 'B 유사도' 가 핵심 지표다.
  높다 → B 를 미리 보여주고 있다 (= 두 끝을 섞고 있다)
  낮다 → 중간에 B 가 아닌 무언가가 있다
단, 낮다고 해서 '새 공간' 인지 '붕괴' 인지는 이 지표로 구분되지 않는다(§4 지표 주의).
"""
import os, sys
import numpy as np
import cv2
from PIL import Image

E = os.path.expanduser("~/data/끝점")
RUNS = [
    ("B0 광학흐름", "~/data/B0/b0_flow.mp4"),
    ("B0 교차페이드", "~/data/B0/b0_blend.mp4"),
    ("단일 클립", "~/data/생성/bridge_render.mp4"),
    ("오라클 분할", "~/data/생성/split_oracle.mp4"),
    ("6걸음×97f", "~/data/연쇄/연쇄.mp4"),
]


def L(p):
    return np.asarray(Image.open(p).convert("RGB").resize((1024, 576)), float)


def psnr(a, b):
    m = ((a - b) ** 2).mean()
    return 99.0 if m == 0 else 10 * np.log10(255.0 ** 2 / m)


def frames(path, fracs=(0.0, 0.25, 0.5, 0.75, 1.0)):
    cap = cv2.VideoCapture(os.path.expanduser(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for f in fracs:
        k = min(n - 1, max(0, int(round(f * (n - 1)))))
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        ok, im = cap.read()
        out.append(None if not ok else
                   np.asarray(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                              .resize((1024, 576)), float))
    cap.release()
    return n, out


def main():
    A, B = L(f"{E}/I_A_out.png"), L(f"{E}/I_B_in.png")
    print()
    print("%-14s %6s %8s %8s | %s" %
          ("방법", "프레임", "첫vsA", "끝vsB", "중간의 B 유사도 25/50/75% (높으면 B 를 미리 보여줌)"))
    print("-" * 104)
    for name, p in RUNS:
        if not os.path.exists(os.path.expanduser(p)):
            print("%-14s  (없음)" % name)
            continue
        n, fr = frames(p)
        if any(x is None for x in fr):
            print("%-14s  (프레임 읽기 실패)" % name)
            continue
        mid = "  ".join("%5.2f" % psnr(B, fr[i]) for i in (1, 2, 3))
        print("%-14s %6d %8.2f %8.2f |  %s" % (name, n, psnr(A, fr[0]), psnr(B, fr[4]), mid))
    print()
    print("※ 중간 B 유사도가 낮다고 '새 공간' 인 것은 아니다 — '붕괴' 도 낮게 나온다.")
    print("   둘의 구분에는 SfM 등록률·직진성과 육안 확인이 함께 필요하다(§4).")


if __name__ == "__main__":
    main()
