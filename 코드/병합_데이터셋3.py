#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""겹치는 구간에서는 **실행 하나만** 골라 쓰는 병합 데이터셋 (G_M4).

왜:
  G_M2(합집합)는 평균 PSNR 을 올렸지만(16.38 → 17.64) 정지 영상으로 보면
  **잃은 것도 있다** — G_M 에 있던 난간·계단 구조가 갈색 뭉개짐이 됐다.
  겹침 구간에서 실행들이 같은 자리를 다르게 그렸고 학습이 평균을 냈다.
  덮인 60 위치 중 45%가 둘 이상의 실행에 중복해서 덮인다.

방법:
  복도 위치마다 **가장 가까이 지나간 실행 하나**만 채택한다.
  (그 실행이 그 자리를 가장 정확히 본 것이므로)
  단독으로 덮인 위치는 그대로 둔다. 따라서 커버리지는 G_M2 와 같고
  (48%, 43~122) 겹침만 사라진다.

🔴 이것이 G_M2 보다 나을지는 모른다. 시점이 줄면 제약도 줄어 흐려질 수 있다.
   G_M2 와의 차이가 **"평균화로 잃은 것" 과 "시점 수로 얻은 것"** 중
   무엇이 큰지를 답한다.

🔴 oracle 은 넣지 않는다 (숨긴 정답을 웨이포인트로 쓴 실행).
"""
import argparse
import os
import shutil
from collections import defaultdict

import numpy as np

HOME = os.path.expanduser("~")
OFF = 10000000


def q2R(q):
    w, x, y, z = q
    return np.array([[1-2*y*y-2*z*z, 2*x*y-2*z*w,   2*x*z+2*y*w],
                     [2*x*y+2*z*w,   1-2*x*x-2*z*z, 2*y*z-2*x*w],
                     [2*x*z-2*y*w,   2*y*z+2*x*w,   1-2*x*x-2*y*y]])


def read_images(p):
    out = {}
    lines = [l for l in open(p) if not l.startswith("#")]
    for i in range(0, len(lines), 2):
        t = lines[i].split()
        if len(t) < 10:
            continue
        pts = []
        if i+1 < len(lines):
            v = lines[i+1].split()
            for k in range(0, len(v)-2, 3):
                pid = int(v[k+2])
                if pid != -1:
                    pts.append((float(v[k]), float(v[k+1]), pid))
        out[t[9]] = dict(q=t[1:5], t=t[5:8], cam=int(t[8]), pts=pts,
                         R=q2R(list(map(float, t[1:5]))),
                         tv=np.array(list(map(float, t[5:8]))))
    return out


def read_points(p):
    d = {}
    for l in open(p):
        if l.startswith("#"):
            continue
        v = l.split()
        if len(v) >= 8:
            d[int(v[0])] = v
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regs", default="wp4,brk,even5")
    ap.add_argument("--base", default=HOME+"/data/fixreg2")
    ap.add_argument("--out", default=HOME+"/data/scenes/M4")
    a = ap.parse_args()

    runs = [r for r in a.regs.split(",") if r]
    assert "oracle" not in runs

    base0 = os.path.join(a.base, runs[0], "out")
    I0 = read_images(os.path.join(base0, "images.txt"))
    C0 = {k: -v["R"].T @ v["tv"] for k, v in I0.items()}

    def ix(k):
        return int(k.split("_")[1].split(".")[0])

    A = sorted((k for k in I0 if k.startswith("v1791_") and ix(k) <= 143), key=ix)
    GT = sorted((k for k in I0 if k.startswith("v1791_") and 162 <= ix(k) <= 286),
                key=ix)
    B = sorted((k for k in I0 if k.startswith("v1793_")), key=ix)
    G = np.array([C0[k] for k in GT])
    delta = float(np.median([np.linalg.norm(G[i+1]-G[i]) for i in range(len(G)-1)]))
    THR = 3*delta
    real = np.array([C0[k] for k in A+B])

    # 후보 수집: (복도위치, 거리, 실행, 프레임이름)
    imgs_of, cand = {}, defaultdict(list)
    extra = []          # 복도 밖(끝점 근처)이지만 채택 기준을 통과한 것
    for ri, r in enumerate(runs):
        p = os.path.join(a.base, r, "out", "images.txt")
        if not os.path.exists(p):
            print("  %-7s 없음" % r)
            continue
        I = read_images(p)
        imgs_of[r] = I
        C = {k: -v["R"].T @ v["tv"] for k, v in I.items()}
        for k in sorted(k for k in I if k.startswith("gen_")):
            dr = float(np.min(np.linalg.norm(real - C[k], axis=1)))
            j = int(np.argmin(np.linalg.norm(G - C[k], axis=1)))
            dg = float(np.linalg.norm(G[j] - C[k]))
            if dg <= THR:
                cand[j].append((dg, r, k, ri))
            elif dr <= THR:
                extra.append((r, k, ri))

    picked, by_run = [], defaultdict(int)
    for j in sorted(cand):
        cand[j].sort()
        d, r, k, ri = cand[j][0]          # 가장 가까이 지나간 실행
        picked.append(dict(new="%s_%s" % (r, k),
                           src=os.path.join(a.base, r, "images", k),
                           run=r, old=k, ri=ri))
        by_run[r] += 1
    seen = {(q["run"], q["old"]) for q in picked}
    for r, k, ri in extra:
        if (r, k) not in seen:
            picked.append(dict(new="%s_%s" % (r, k),
                               src=os.path.join(a.base, r, "images", k),
                               run=r, old=k, ri=ri))
            by_run[r] += 1

    print("복도 커버 %d/%d = %.0f%% (%d~%d)"
          % (len(cand), len(GT), 100*len(cand)/len(GT), min(cand), max(cand)))
    print("채택 %d장 | 실행별: %s"
          % (len(picked), ", ".join("%s %d" % (r, by_run[r]) for r in runs)))

    out = a.out
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    sp = os.path.join(out, "colmap", "sparse", "0")
    os.makedirs(sp, exist_ok=True)
    src0 = os.path.join(a.base, runs[0], "images")
    for k in A+B:
        s = os.path.join(src0, k)
        if os.path.exists(s):
            shutil.copy(s, os.path.join(out, "images", k))
    for q in picked:
        if os.path.exists(q["src"]):
            shutil.copy(q["src"], os.path.join(out, "images", q["new"]))

    P3, obs = {}, defaultdict(list)
    need = set()
    for k in A+B:
        for _, _, pid in I0[k]["pts"]:
            need.add(pid)
    bp = read_points(os.path.join(base0, "points3D.txt"))
    for pid in need:
        if pid in bp:
            P3[pid] = bp[pid]
    for k in A+B:
        for j, (_, _, pid) in enumerate(I0[k]["pts"]):
            if pid in P3:
                obs[pid].append((k, j))
    cache = {}
    for q in picked:
        r = q["run"]
        if r not in cache:
            cache[r] = read_points(os.path.join(a.base, r, "out", "points3D.txt"))
        d = cache[r]
        off = (q["ri"]+1)*OFF
        for j, (_, _, pid) in enumerate(imgs_of[r][q["old"]]["pts"]):
            if pid in d:
                P3.setdefault(pid+off, d[pid])
                obs[pid+off].append((q["new"], j))

    names = A + [q["new"] for q in picked] + B
    newid = {k: i+1 for i, k in enumerate(names)}
    keepp = [p for p in sorted(P3) if len(obs[p]) >= 2]
    pidnew = {p: i+1 for i, p in enumerate(keepp)}
    print("  이미지 %d장, 3D 점 %d개" % (len(names), len(keepp)))

    with open(os.path.join(sp, "cameras.txt"), "w") as f:
        f.write("# Camera list\n")
        for l in open(os.path.join(base0, "cameras.txt")):
            if not l.startswith("#"):
                f.write(l)
    look = {k: (I0[k], 0) for k in A+B}
    for q in picked:
        look[q["new"]] = (imgs_of[q["run"]][q["old"]], (q["ri"]+1)*OFF)
    with open(os.path.join(sp, "images.txt"), "w") as f:
        f.write("# Image list\n")
        for k in names:
            d, off = look[k]
            f.write("%d %s %s %d %s\n" % (newid[k], " ".join(d["q"]),
                                          " ".join(d["t"]), d["cam"], k))
            f.write(" ".join("%.3f %.3f %d" % (x, y, pidnew.get(pid+off, -1))
                             for x, y, pid in d["pts"]) + "\n")
    with open(os.path.join(sp, "points3D.txt"), "w") as f:
        f.write("# 3D point list\n")
        for p in keepp:
            v = P3[p]
            f.write("%d %s %s %s\n"
                    % (pidnew[p], " ".join(v[1:4]), " ".join(v[4:8]),
                       " ".join("%d %d" % (newid[x], y) for x, y in obs[p])))
    print("  기록 완료 -> %s" % out)


if __name__ == "__main__":
    main()
