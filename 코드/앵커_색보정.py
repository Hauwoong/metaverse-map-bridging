#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""연쇄 생성의 앵커를 다음 걸음에 넣기 전에 색을 되돌린다.

왜 필요한가:
  연쇄 생성은 자기 출력을 다시 입력으로 먹는다. 그래서 색이 걸음마다 누적으로 밀린다 —
  6걸음에서 평균 RGB 가 (163.9,160.3,149.8) → (142.0,133.9,120.6) 로 13% 어두워지고
  파랑이 16% 빠졌다(누렇게 감). 걸음을 늘리면 더 심해진다.

무엇을 하는가:
  앵커의 채널별 평균과 표준편차를 기준 이미지(A 끝점)에 맞춘다.
  구조는 건드리지 않고 색만 되돌리므로, 생성된 '새 공간' 은 보존된다.

⚠️ 한계: 실제로 조명이 다른 구간으로 이동한 경우에도 A 의 색으로 되돌려 버린다.
   복도 내부처럼 조명이 균일한 환경을 전제한 보정이다. 보고서에 명시할 것.
"""
import argparse
import numpy as np
from PIL import Image


def match_color(src, ref, strength=1.0):
    """src 의 채널별 평균·표준편차를 ref 에 맞춘다. strength 로 세기를 조절한다."""
    s = np.asarray(src, np.float64)
    r = np.asarray(ref, np.float64)
    out = s.copy()
    for c in range(3):
        ms, ss = s[..., c].mean(), s[..., c].std()
        mr, sr = r[..., c].mean(), r[..., c].std()
        if ss < 1e-6:
            continue
        fixed = (s[..., c] - ms) * (sr / ss) + mr
        out[..., c] = s[..., c] * (1 - strength) + fixed * strength
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="보정할 앵커")
    ap.add_argument("--ref", required=True, help="기준 이미지 (A 끝점)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--strength", type=float, default=1.0)
    a = ap.parse_args()
    src = Image.open(a.src).convert("RGB")
    ref = Image.open(a.ref).convert("RGB")
    fixed = match_color(src, ref, a.strength)
    fixed.save(a.out)
    s0 = np.asarray(src, float).reshape(-1, 3).mean(0)
    s1 = np.asarray(fixed, float).reshape(-1, 3).mean(0)
    r0 = np.asarray(ref, float).reshape(-1, 3).mean(0)
    print("    색보정  평균 (%.1f,%.1f,%.1f) → (%.1f,%.1f,%.1f)  기준 (%.1f,%.1f,%.1f)"
          % (*s0, *s1, *r0))


if __name__ == "__main__":
    main()
