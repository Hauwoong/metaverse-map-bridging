#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""생성 구간의 이동 거리를 합본 좌표계의 '절대 단위' 로 환산한다.

원리:
  다리 모델(실촬영 경계 + 생성)에는 실촬영 프레임이 함께 들어 있다. 그 프레임들의
  **합본 모델에서의 위치는 이미 정확히 안다.** 두 집합의 대응으로 Sim(3) 를 구해
  다리 모델 전체를 합본 좌표계로 옮기면, 생성 카메라의 위치도 절대 단위로 읽힌다.

  이렇게 하면 오늘 내내 문제였던 '재구성마다 스케일이 달라지는' 것이 사라진다.
  스케일을 추정하는 것이 아니라 **실촬영 프레임이 스케일을 고정**하기 때문이다.

⚠️ 정렬 품질을 반드시 같이 본다. 잔차가 크면 환산값을 믿을 수 없다.
   (카메라 중심만으로 Sim(3) 를 구할 때의 퇴화는 §3.4 참조. 여기서는 실촬영이
    두 영역에 걸쳐 80장 분포하므로 조건수가 나쁘지 않다 — 잔차로 확인한다.)
"""
import os, sys, glob
import numpy as np

HOME = os.path.expanduser("~")


def q2R(q):
    w, x, y, z = q
    return np.array([[1-2*y*y-2*z*z, 2*x*y-2*z*w,   2*x*z+2*y*w],
                     [2*x*y+2*z*w,   1-2*x*x-2*z*z, 2*y*z-2*x*w],
                     [2*x*z-2*y*w,   2*y*z+2*x*w,   1-2*x*x-2*y*y]])


def centers(p):
    out = {}
    lines = [l for l in open(p) if not l.startswith("#")]
    for i in range(0, len(lines), 2):
        t = lines[i].split()
        if len(t) < 10:
            continue
        R = q2R(list(map(float, t[1:5])))
        tv = np.array(list(map(float, t[5:8])))
        out[t[9]] = -R.T @ tv
    return out


def umeyama(src, dst):
    """src → dst 로 보내는 Sim(3) (s, R, t). 반사를 막는다."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    C = D.T @ S / len(src)
    U, d, Vt = np.linalg.svd(C)
    E = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        E[2, 2] = -1            # 반사 보정 — 빼먹으면 거울상이 된다
    R = U @ E @ Vt
    var_s = (S ** 2).sum() / len(src)
    s = float(np.trace(np.diag(d) @ E) / var_s)
    t = mu_d - s * R @ mu_s
    return s, R, t


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "chain6"
    bdir = os.path.join(HOME, "data/bridge_reg", tag, "sparse")
    merged = os.path.join(HOME, "data/sfm_all/sparse/1/images.txt")
    M = centers(merged)

    best, bestn = None, 0
    for m in sorted(glob.glob(os.path.join(bdir, "*"))):
        f = os.path.join(m, "images.txt")
        if not os.path.exists(f):
            continue
        C = centers(f)
        g = sum(1 for k in C if k.startswith("gen_"))
        r = sum(1 for k in C if k.startswith("v17"))
        if g > 0 and r > 0 and len(C) > bestn:
            best, bestn = C, len(C)
    if best is None:
        print("🔴 실촬영과 생성이 함께 들어간 모델이 없다 — 다리가 이어지지 않았다.")
        return

    real = sorted(k for k in best if k.startswith("v17") and k in M)
    gen = sorted(k for k in best if k.startswith("gen_"))
    nA = sum(1 for k in real if k.startswith("v1791_"))
    nB = sum(1 for k in real if k.startswith("v1793_"))
    print("모델 구성: 실촬영 A %d장 / 생성 %d장 / 실촬영 B %d장" % (nA, len(gen), nB))
    if len(real) < 6:
        print("🔴 정렬에 쓸 실촬영이 %d장뿐 — Sim(3) 를 믿을 수 없다." % len(real))
        return

    src = np.array([best[k] for k in real])
    dst = np.array([M[k] for k in real])
    s, R, t = umeyama(src, dst)
    res = np.linalg.norm((s * (R @ src.T).T + t) - dst, axis=1)
    scale_ref = np.linalg.norm(dst.max(0) - dst.min(0))
    print()
    print("Sim(3) 정렬 : 스케일 %.4f" % s)
    print("  잔차 중앙값 %.4f / 평균 %.4f / 최대 %.4f  (실촬영 퍼짐 %.3f 대비 %.1f%%)"
          % (np.median(res), res.mean(), res.max(), scale_ref,
             100*np.median(res)/max(scale_ref, 1e-9)))
    if np.median(res) > 0.1 * scale_ref:
        print("  ⚠️ 잔차가 크다 — 아래 환산값을 신뢰하지 말 것.")

    P = np.array([s * (R @ best[k]) + t for k in gen])
    path = float(sum(np.linalg.norm(P[i+1]-P[i]) for i in range(len(P)-1)))
    span = float(np.linalg.norm(P[-1]-P[0]))
    print()
    print("생성 구간 (합본 좌표계 절대 단위)")
    print("  카메라 경로 길이 %.3f" % path)
    print("  양 끝 직선거리   %.3f" % span)
    GAP = 15.032   # A 끝점 ↔ B 끝점 직선거리 (코드/간극_궤적렌더.py 측정)
    print("  간극(15.032) 대비 : 경로 %.1f%% / 직선 %.1f%%"
          % (100*path/GAP, 100*span/GAP))

    # 같은 좌표계에서 실촬영과 직접 비교한다. 가장 반박하기 어려운 형태다.
    RP = np.array([M[k] for k in real])
    rpath = float(sum(np.linalg.norm(RP[i+1]-RP[i]) for i in range(len(RP)-1)))
    print()
    print("같은 좌표계에서 실촬영과 직접 비교")
    print("  실촬영 %3d장 (%.1f초 @2.5fps) 경로 %.3f  →  프레임당 %.4f"
          % (len(real), len(real)/2.5, rpath, rpath/max(len(real)-1, 1)))
    print("  생성   %3d장 (%.1f초 @24fps·샘플4) 경로 %.3f  →  프레임당 %.4f"
          % (len(gen), len(gen)*4/24, path, path/max(len(gen)-1, 1)))
    print("  → 생성이 실촬영의 %.1f%% 속도로 움직인다 (초당 기준)"
          % (100*(path/(len(gen)*4/24))/max(rpath/(len(real)/2.5), 1e-9)))
    print()
    print("참고) 실촬영 GT1 전체: 경로 17.775, 직선거리 12.006 (50초)")


if __name__ == "__main__":
    main()
