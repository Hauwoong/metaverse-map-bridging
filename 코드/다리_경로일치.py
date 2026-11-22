#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""등록된 생성 프레임이 **실제 복도 위에 있는가**를 잰다.

🔴 왜 A→B 축 진행도로는 안 되는가.
   실촬영 GT1 의 진행도는 −3.93 ~ 6.92 다. 실제 복도는 A→B 직선을 따라가지 않고
   뒤로 돌아갔다가 돌아온다. 축 진행도로 "중간 구간" 을 정의하면 실제 경로를
   '밖' 으로 판정해 버린다 — 실제로 그렇게 오판했다.

   올바른 자: **실촬영 GT1 궤적과의 최단거리.** 가까우면 실제 복도 위에 있는 것이다.
   비교 기준은 실촬영 자신의 프레임 간격으로 잡는다(임의 임계값을 쓰지 않는다).
"""
import argparse, os
import numpy as np

HOME = os.path.expanduser("~")


def q2R(q):
    w, x, y, z = q
    return np.array([[1-2*y*y-2*z*z, 2*x*y-2*z*w,   2*x*z+2*y*w],
                     [2*x*y+2*z*w,   1-2*x*x-2*z*z, 2*y*z-2*x*w],
                     [2*x*z-2*y*w,   2*y*z+2*x*w,   1-2*x*x-2*y*y]])


def centers(p):
    out = {}
    L = [l for l in open(p) if not l.startswith("#")]
    for i in range(0, len(L), 2):
        t = L[i].split()
        if len(t) < 10:
            continue
        R = q2R(list(map(float, t[1:5])))
        tv = np.array(list(map(float, t[5:8])))
        out[t[9]] = -R.T @ tv
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="*", default=["chain6", "oracle", "wp4"])
    a = ap.parse_args()
    M = centers(HOME+"/data/sfm_all/sparse/1/images.txt")
    gt = sorted(k for k in M if k.startswith("v1791_") and 162 <= int(k[6:11]) <= 286)
    G = np.array([M[k] for k in gt])
    sp = np.median([np.linalg.norm(G[i+1]-G[i]) for i in range(len(G)-1)])
    print("실촬영 GT1 %d장, 인접 프레임 간격 중앙값 %.4f" % (len(gt), sp))
    print("판정 기준: 그 간격의 3배(%.3f) 안이면 '실제 복도 위'로 본다." % (3*sp))
    print()

    for tag in a.tags:
        p = HOME+"/data/fixreg2/%s/out/images.txt" % tag
        if not os.path.exists(p):
            print("%-9s (없음)" % tag); continue
        C = centers(p)
        gen = sorted(k for k in C if k.startswith("gen_"))
        if not gen:
            print("%-9s 등록된 생성 프레임 없음" % tag); continue
        P = np.array([C[k] for k in gen])
        d = np.array([np.min(np.linalg.norm(G - x, axis=1)) for x in P])
        on = d <= 3*sp
        print("%-9s 생성 %d장" % (tag, len(gen)))
        print("    GT1 궤적과의 거리: 중앙값 %.3f, 25분위 %.3f, 최소 %.3f"
              % (np.median(d), np.percentile(d, 25), d.min()))
        print("    실제 복도 위(≤%.3f) : **%d장 (%.0f%%)**"
              % (3*sp, on.sum(), 100*on.mean()))
        if on.sum():
            # 복도 위 프레임이 GT1 의 어느 구간을 덮는가
            idx = [int(np.argmin(np.linalg.norm(G - x, axis=1))) for x in P[on]]
            print("    덮는 GT1 구간: %d~%d 번째 (전체 %d 중 **%d개 위치 = %.0f%%**)"
                  % (min(idx), max(idx), len(gt), len(set(idx)),
                     100*len(set(idx))/len(gt)))
            # 덮인 위치가 한 곳에 뭉쳤는가, 여러 곳에 퍼졌는가
            u2 = sorted(set(idx))
            gaps = [u2[i+1]-u2[i] for i in range(len(u2)-1)]
            cl = 1 + sum(1 for g in gaps if g > 5)
            print("    덮인 구간 덩어리 %d개  ← 웨이포인트 수와 비교할 것" % cl)


if __name__ == "__main__":
    main()
