#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""생성 영상 SfM 결과의 카메라 궤적을 분석한다.

보는 것 두 가지:
  등록률          기하가 깨져 있으면 낮다
  직진성 = 양끝직선거리 / 경로길이
                  1 에 가까우면 쭉 전진, 0 에 가까우면 제자리에서 맴돈다(러닝머신)

오늘 끝점 거리 PSNR 은 '전진 vs 자기복제' 를 구분하지 못했다. 이 지표는 구분한다.
"""
import sys, os, glob
import numpy as np


def q2R(q):
    w, x, y, z = q
    return np.array([[1-2*y*y-2*z*z, 2*x*y-2*z*w,   2*x*z+2*y*w],
                     [2*x*y+2*z*w,   1-2*x*x-2*z*z, 2*y*z-2*x*w],
                     [2*x*z-2*y*w,   2*y*z+2*x*w,   1-2*x*x-2*y*y]])


def centers(p):
    ls = [l for l in open(p) if not l.startswith("#")]
    C = {}
    for i in range(0, len(ls), 2):
        t = ls[i].split()
        if len(t) < 10:
            continue
        R = q2R(list(map(float, t[1:5])))
        tv = np.array(list(map(float, t[5:8])))
        C[t[9]] = -R.T @ tv
    return C


def points_extent(path_dir):
    """points3D.txt 의 점 구름 지름. 스케일이 임의인 모델들을 비교하려면 필요하다."""
    f = os.path.join(path_dir, "points3D.txt")
    if not os.path.exists(f):
        return None
    P = []
    for l in open(f):
        if l.startswith("#"):
            continue
        t = l.split()
        if len(t) >= 4:
            P.append([float(t[1]), float(t[2]), float(t[3])])
    if len(P) < 10:
        return None
    P = np.asarray(P)
    lo, hi = np.percentile(P, 2, axis=0), np.percentile(P, 98, axis=0)
    return float(np.linalg.norm(hi - lo))


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/data/sfm_gen")
    names = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    print()
    print("%-9s %7s %7s %9s %9s %9s %8s %9s" %
          ("이름", "등록", "전체", "경로길이", "직선거리", "퍼짐", "직진성", "장면대비"))
    print("-" * 78)
    for name in (names or ("single", "oracle", "chain")):
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            continue
        n_all = len(glob.glob(os.path.join(d, "images", "*.png")))
        best, bestC = None, {}
        for m in sorted(glob.glob(os.path.join(d, "sparse", "*"))):
            f = os.path.join(m, "images.txt")
            if not os.path.exists(f):
                continue
            C = centers(f)
            if len(C) > len(bestC):
                best, bestC = m, C
        if not bestC:
            print("%-9s %7s %7d   재구성 실패" % (name, "-", n_all))
            continue
        ks = sorted(bestC)
        P = np.array([bestC[k] for k in ks])
        path = float(sum(np.linalg.norm(P[i+1]-P[i]) for i in range(len(P)-1)))
        span = float(np.linalg.norm(P[-1]-P[0]))
        size = float(np.linalg.norm(P.max(0)-P.min(0)))
        ext = points_extent(best)
        rel = (path / ext) if ext else float("nan")
        # 영상 길이(초)는 디렉터리 이름에서 유추할 수 없으므로 메타가 있으면 쓴다.
        print("%-9s %7d %7d %9.3f %9.3f %9.3f %8.3f %9.4f"
              % (name, len(bestC), n_all, path, span, size,
                 span/max(path, 1e-9), rel))

    # 실촬영 기준선: 합본 모델의 GT1 구간
    merged = os.path.expanduser("~/data/sfm_all/sparse/1")
    if os.path.exists(os.path.join(merged, "images.txt")):
        C = centers(os.path.join(merged, "images.txt"))
        gt = sorted(k for k in C if k.startswith("v1791_")
                    and 162 <= int(k[6:11]) <= 286)
        if len(gt) > 2:
            P = np.array([C[k] for k in gt])
            path = float(sum(np.linalg.norm(P[i+1]-P[i]) for i in range(len(P)-1)))
            span = float(np.linalg.norm(P[-1]-P[0]))
            size = float(np.linalg.norm(P.max(0)-P.min(0)))
            ext = points_extent(merged)
            rel = (path / ext) if ext else float("nan")
            print("-" * 78)
            print("%-9s %7d %7d %9.3f %9.3f %9.3f %8.3f %9.4f"
                  % ("실촬영GT1", len(gt), len(gt), path, span, size,
                     span/max(path, 1e-9), rel))

    print()
    print("직진성: 1 에 가까우면 똑바로 갔다는 뜻. '얼마나 멀리' 가 아니라 '얼마나 곧게' 다.")
    print("장면대비 이동: 경로길이 ÷ 점구름 지름. 스케일이 임의인 모델을 비교하려 장면 크기로 나눴다.")
    print("  ⚠️ 프레임당으로 나누지 말 것 — 샘플링 간격이 다르면(2.5/6/12fps) 비교가 깨진다.")
    print("  ⚠️ 영상 길이가 다르면 '초당' 으로 환산해 볼 것. 총량만으로는 빠른지 오래 갔는지 모른다.")


if __name__ == "__main__":
    main()
