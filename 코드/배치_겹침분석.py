#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""서로 다른 생성 실행이 같은 복도 위치를 얼마나 중복해서 덮는가.

겹치는 구간에서 실행들이 **서로 다른 그림**을 그리면, 학습이 그것을 평균내어
오히려 흐려질 수 있다. 합집합(G_M2)을 만들기 전에 그 위험이 얼마나 되는지 센다.

결과 (2026-11-23):
  덮인 60 위치 중  한 실행 33(55%) / 두 실행 24(40%) / 세 실행 3(5%)
  새로 얻은 B 쪽(115~122)은 전부 단독 → 순이득
  위험은 52~93 구간 (wp4 와 even5/brk 가 겹친다)

📌 겹침이 반드시 나쁜 것은 아니다. 두 실행 모두 **등록에 성공**했다는 것은
   기하가 실제와 맞았다는 뜻이고, 기하가 일치하면 시점이 늘어 오히려 선명해진다.
   다른 것은 텍스처 세부뿐일 수 있다. 어느 쪽이 이기는지는 수치가 답한다.
"""
from collections import defaultdict

import numpy as np

HOME = os.path.expanduser("~")
RUNS = ["wp4", "brk", "even5"]


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


C0 = centers(HOME + "/data/fixreg2/wp4/out/images.txt")


def ix(k):
    return int(k.split("_")[1].split(".")[0])


GT = sorted((k for k in C0 if k.startswith("v1791_") and 162 <= ix(k) <= 286),
            key=ix)
A = sorted((k for k in C0 if k.startswith("v1791_") and ix(k) <= 143), key=ix)
B = sorted((k for k in C0 if k.startswith("v1793_")), key=ix)
G = np.array([C0[k] for k in GT])
real = np.array([C0[k] for k in A + B])
delta = float(np.median([np.linalg.norm(G[i+1]-G[i]) for i in range(len(G)-1)]))
THR = 3 * delta

owner = defaultdict(list)       # 복도 위치 -> [실행...]
for r in RUNS:
    C = centers(HOME + "/data/fixreg2/%s/out/images.txt" % r)
    for k in sorted(C):
        if not k.startswith("gen_"):
            continue
        dr = float(np.min(np.linalg.norm(real - C[k], axis=1)))
        j = int(np.argmin(np.linalg.norm(G - C[k], axis=1)))
        dg = float(np.linalg.norm(G[j] - C[k]))
        if min(dr, dg) <= THR and dg <= THR:
            owner[j].append(r)

n1 = sum(1 for v in owner.values() if len(set(v)) == 1)
n2 = sum(1 for v in owner.values() if len(set(v)) == 2)
n3 = sum(1 for v in owner.values() if len(set(v)) >= 3)
print("복도 %d 위치 중 %d 위치가 덮인다" % (len(GT), len(owner)))
print("  한 실행만 덮음   %3d (%.0f%%)" % (n1, 100*n1/max(len(owner), 1)))
print("  두 실행이 덮음   %3d (%.0f%%)  ← 평균화 위험" % (n2, 100*n2/max(len(owner), 1)))
print("  세 실행이 덮음   %3d (%.0f%%)  ← 위험 큼" % (n3, 100*n3/max(len(owner), 1)))
print()
print("위치별 (복도 번호: 덮는 실행들)")
ks = sorted(owner)
line = []
for j in ks:
    s = "".join(sorted({r[0] for r in owner[j]}))   # w/b/e
    line.append("%d:%s" % (j, s))
for i in range(0, len(line), 10):
    print("  " + "  ".join(line[i:i+10]))
print()
print("w=wp4  b=brk  e=even5")
