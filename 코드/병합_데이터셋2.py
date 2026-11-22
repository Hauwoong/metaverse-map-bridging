#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""여러 웨이포인트 배치의 생성 프레임을 **합쳐서** 병합 맵을 만든다 (G_M2).

왜:
  배치를 세 번 시험했는데 각각 **다른 구간**을 덮었다.
    wp4   (23·46·69·92)     복도  43~102
    brk   ( 5·30·51·110)    복도  52~122   ← B 쪽이 더 넓다
    even5 ( 5·30·55·80·105) 복도  63~118
  합치면 43~122 다. G_M 의 커버리지 39% 가 아쉬운 지점이었고(§4.5 평균 16.38 dB),
  **재생성도 재등록도 필요 없다** — 다섯 등록 결과가 전부 같은 좌표계다
  (실촬영 프레임 위치 차이 0.000000 으로 확인).

🔴 oracle 은 넣지 않는다. 숨긴 정답(GT1)의 렌더를 웨이포인트로 쓴 실행이라,
   넣으면 "맵이 정답을 본 적 없다" 는 전제가 깨진다.

🔴 실행마다 생성 프레임 이름이 gen_00001.png 로 겹친다. 서로 다른 그림이므로
   `<실행>_<이름>` 으로 바꿔 쓴다.

🔴 3D 점 ID 도 실행마다 독립이다. 실행별로 오프셋을 준다. 같은 실제 점이
   두 번 들어갈 수 있지만, splatfacto 는 points3D 를 **초기 점구름 씨앗**으로만
   쓰므로 중복은 해가 없다.
"""
import argparse
import os
import shutil
from collections import defaultdict

import numpy as np

HOME = os.path.expanduser("~")
OFF = 10000000          # 실행별 점 ID 오프셋


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
        out[t[9]] = dict(id=int(t[0]), q=t[1:5], t=t[5:8], cam=int(t[8]), pts=pts,
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
    ap.add_argument("--regs", default="wp4,brk,even5,chain6",
                    help="합칠 등록 실행들 (oracle 은 넣지 않는다)")
    ap.add_argument("--base", default=HOME+"/data/fixreg2")
    ap.add_argument("--out", default=HOME+"/data/scenes/M2")
    a = ap.parse_args()

    runs = [r for r in a.regs.split(",") if r]
    assert "oracle" not in runs, "oracle 은 숨긴 정답을 쓴 실행이다 — 넣으면 안 된다"

    # A·B·GT1 은 어느 실행에서 읽어도 같다 (포즈 고정 등록)
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
    print("실촬영 A %d, GT1 %d(제외), B %d | 임계 %.4f (간격 %.4f x3)"
          % (len(A), len(GT), len(B), THR, delta))

    picked = []          # dict(새이름, 원본, 실행, 원이름, 실행번호)
    imgs_of = {}         # 실행 -> read_images 결과
    covered = set()
    for ri, r in enumerate(runs):
        p = os.path.join(a.base, r, "out", "images.txt")
        if not os.path.exists(p):
            print("  %-7s 없음 - 건너뜀" % r)
            continue
        I = read_images(p)
        imgs_of[r] = I
        C = {k: -v["R"].T @ v["tv"] for k, v in I.items()}
        gen = sorted(k for k in I if k.startswith("gen_"))
        keep, cov = [], set()
        for k in gen:
            dr = float(np.min(np.linalg.norm(real - C[k], axis=1)))
            j = int(np.argmin(np.linalg.norm(G - C[k], axis=1)))
            dg = float(np.linalg.norm(G[j] - C[k]))
            if min(dr, dg) <= THR:
                keep.append(k)
                if dg <= THR:
                    cov.add(j)
        covered |= cov
        rng = "%d~%d" % (min(cov), max(cov)) if cov else "없음"
        print("  %-7s 생성 %3d -> 채택 %3d | 복도 %s (%d 위치)"
              % (r, len(gen), len(keep), rng, len(cov)))
        for k in keep:
            picked.append(dict(new="%s_%s" % (r, k),
                               src=os.path.join(a.base, r, "images", k),
                               run=r, old=k, ri=ri))

    if covered:
        print("합집합 생성 %d장 | 복도 커버 %d/%d = %.0f%% (%d~%d)"
              % (len(picked), len(covered), len(GT),
                 100*len(covered)/len(GT), min(covered), max(covered)))
    else:
        print("합집합 생성 %d장 | 복도 커버 없음" % len(picked))

    out = a.out
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    sp = os.path.join(out, "colmap", "sparse", "0")
    os.makedirs(sp, exist_ok=True)

    src0 = os.path.join(a.base, runs[0], "images")
    miss = 0
    for k in A+B:
        s = os.path.join(src0, k)
        if os.path.exists(s):
            shutil.copy(s, os.path.join(out, "images", k))
        else:
            miss += 1
    for q in picked:
        if os.path.exists(q["src"]):
            shutil.copy(q["src"], os.path.join(out, "images", q["new"]))
        else:
            miss += 1
    if miss:
        print("  이미지 %d장 없음" % miss)

    # ---- points3D: 실행별 오프셋 ----
    P3 = {}
    obs = defaultdict(list)
    need = set()
    for k in A+B:
        for _, _, pid in I0[k]["pts"]:
            need.add(pid)
    base_pts = read_points(os.path.join(base0, "points3D.txt"))
    for pid in need:
        if pid in base_pts:
            P3[pid] = base_pts[pid]
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
        I = imgs_of[r]
        off = (q["ri"]+1)*OFF
        for j, (_, _, pid) in enumerate(I[q["old"]]["pts"]):
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

    n = 0
    with open(os.path.join(sp, "points3D.txt"), "w") as f:
        f.write("# 3D point list\n")
        for p in keepp:
            v = P3[p]
            f.write("%d %s %s %s\n"
                    % (pidnew[p], " ".join(v[1:4]), " ".join(v[4:8]),
                       " ".join("%d %d" % (newid[a_], b_) for a_, b_ in obs[p])))
            n += 1
    print("  기록 완료 -> %s" % out)


if __name__ == "__main__":
    main()
