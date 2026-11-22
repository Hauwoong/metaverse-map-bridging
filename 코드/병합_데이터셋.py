#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""병합 맵 G_M 의 데이터셋을 만든다 — A + 생성다리 + B. **변환 없음.**

🔴 왜 PLY 병합이 아니라 재학습인가:
   G_A·G_B·G_C 를 각각 PLY 로 내보내 합치려면, 세 모델의 dataparser 변환이
   서로 다른 회전을 담고 있어 가우시안의 구면조화 계수(f_rest 45개)를
   회전시켜야 한다. 생략하면 시점 의존 외형이 깨지고, 구현하면 틀리기 쉽다.
   반면 A·B·생성 프레임은 **이미 하나의 합본 좌표계에 있다**
   (fix_existing_frames 로 등록했고 실촬영 포즈 변화량 0.000000 검증).
   그 전체로 스플랫 하나를 학습하면 변환도 SH 회전도 필요 없고,
   **이음새가 학습 과정에서 함께 최적화된다** — ⑦ 과 ⑧ 이 한 번에 된다.

🔴 GT1 은 넣지 않는다. 숨긴 정답이므로 평가에만 쓴다.
   G_M 이 GT1 구간을 어떻게 복원했는지가 측정 대상이다.

생성 프레임 걸러내기는 GC_데이터셋2.py 와 같은 검증된 기준을 쓴다 —
실촬영 궤적과의 거리 ≤ (실촬영 인접 프레임 간격 중앙값 × 3).
축 진행도·측면이탈로 거르면 실제 복도 위 프레임을 날린다(61장을 날린 적이 있다).
"""
import argparse, os, shutil
from collections import defaultdict
import numpy as np

HOME = os.path.expanduser("~")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reg", default=HOME+"/data/fixreg2/wp4/out")
    ap.add_argument("--imgs", default=HOME+"/data/fixreg2/wp4/images")
    ap.add_argument("--out", default=HOME+"/data/scenes/M")
    ap.add_argument("--no-gen", action="store_true",
                    help="생성 다리 없이 A+B 만 — 대조군(이음새가 비어 있는 맵)")
    a = ap.parse_args()

    I = read_images(os.path.join(a.reg, "images.txt"))
    C = {k: -v["R"].T @ v["tv"] for k, v in I.items()}

    def idx(k):
        return int(k.split("_")[1].split(".")[0])

    A  = sorted(k for k in I if k.startswith("v1791_") and idx(k) <= 143)
    GT = sorted(k for k in I if k.startswith("v1791_") and 162 <= idx(k) <= 286)
    B  = sorted(k for k in I if k.startswith("v1793_"))
    gen = sorted(k for k in I if k.startswith("gen_"))
    print("등록된 프레임: A %d, GT1 %d(제외), B %d, 생성 %d"
          % (len(A), len(GT), len(B), len(gen)))

    keep_gen = []
    if not a.no_gen:
        # 임계값은 실촬영 자신의 간격에서 뽑는다 (머리로 정하지 않는다)
        G = np.array([C[k] for k in GT])
        delta = float(np.median([np.linalg.norm(G[i+1]-G[i]) for i in range(len(G)-1)]))
        THR = 3 * delta
        real = np.array([C[k] for k in A + B])
        d = {k: float(np.min(np.linalg.norm(real - C[k], axis=1))) for k in gen}
        dg = {k: float(np.min(np.linalg.norm(G - C[k], axis=1))) for k in gen}
        keep_gen = [k for k in gen if min(d[k], dg[k]) <= THR]
        cov = sorted({int(np.argmin(np.linalg.norm(G - C[k], axis=1)))
                      for k in keep_gen if dg[k] <= THR})
        print("  생성 프레임 채택 %d/%d (임계 %.4f = 간격 %.4f × 3)"
              % (len(keep_gen), len(gen), THR, delta))
        if cov:
            print("  GT1 %d~%d 를 덮음 (%d/%d 위치 = %.0f%%)"
                  % (min(cov), max(cov), len(cov), len(GT), 100*len(cov)/len(GT)))

    keep = A + keep_gen + B
    print("병합 데이터셋: %d장 (A %d + 생성 %d + B %d)"
          % (len(keep), len(A), len(keep_gen), len(B)))

    out = a.out
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    sp = os.path.join(out, "colmap", "sparse", "0")
    os.makedirs(sp, exist_ok=True)
    miss = 0
    for k in keep:
        s = os.path.join(a.imgs, k)
        if os.path.exists(s):
            shutil.copy(s, os.path.join(out, "images", k))
        else:
            miss += 1
    if miss:
        print("  🔴 이미지 %d장 없음" % miss)

    pids = set()
    for k in keep:
        for _, _, pid in I[k]["pts"]:
            pids.add(pid)
    P3 = {}
    for l in open(os.path.join(a.reg, "points3D.txt")):
        if l.startswith("#"):
            continue
        v = l.split()
        if len(v) >= 8 and int(v[0]) in pids:
            P3[int(v[0])] = v
    print("  이 프레임들이 보는 3D 점 %d개" % len(P3))

    newid = {k: i+1 for i, k in enumerate(keep)}
    pidnew = {p: i+1 for i, p in enumerate(sorted(P3))}
    cams = {I[k]["cam"] for k in keep}
    with open(os.path.join(sp, "cameras.txt"), "w") as f:
        f.write("# Camera list\n")
        for l in open(os.path.join(a.reg, "cameras.txt")):
            if not l.startswith("#") and int(l.split()[0]) in cams:
                f.write(l)
    with open(os.path.join(sp, "images.txt"), "w") as f:
        f.write("# Image list\n")
        for k in keep:
            d = I[k]
            f.write("%d %s %s %d %s\n" % (newid[k], " ".join(d["q"]),
                                          " ".join(d["t"]), d["cam"], k))
            f.write(" ".join("%.3f %.3f %d" % (x, y, pidnew.get(pid, -1))
                             for x, y, pid in d["pts"]) + "\n")
    tracks = defaultdict(list)
    for k in keep:
        for j, (_, _, pid) in enumerate(I[k]["pts"]):
            if pid in pidnew:
                tracks[pid].append((newid[k], j))
    n = 0
    with open(os.path.join(sp, "points3D.txt"), "w") as f:
        f.write("# 3D point list\n")
        for pid in sorted(P3):
            if len(tracks[pid]) < 2:
                continue
            v = P3[pid]
            f.write("%d %s %s %s\n" % (pidnew[pid], " ".join(v[1:4]),
                                       " ".join(v[4:8]),
                                       " ".join("%d %d" % t for t in tracks[pid])))
            n += 1
    print("  기록: 이미지 %d장, 점 %d개 → %s" % (len(keep), n, out))


if __name__ == "__main__":
    main()
