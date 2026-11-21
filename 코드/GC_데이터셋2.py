#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""고정 등록 결과에서 G_C 데이터셋을 만든다. **변환 없음.**

🔴 앞선 GC_데이터셋.py 와 결정적으로 다른 점:
   생성 프레임이 이미 합본 좌표계에 있다(fix_existing_frames 로 등록했으므로).
   Sim(3) 도 스케일 추정도 하지 않는다. 포즈를 그대로 옮겨 적는다.
   앞서 좌표를 사후에 맞추려다 네 번 실패했다 — 이번엔 맞출 것이 없다.

오등록 걸러내기:
   등록된 77장 중 9걸음이 전체 경로의 74% 를 차지했다. 그 프레임들은 틀렸다.
   두 가지로 거른다 —
     (1) A→B 축 진행도가 [−0.5, 간극+0.5] 밖 (A 뒤쪽 −4.74 까지 있었다)
     (2) 직전 채택 프레임과의 거리가 중앙값의 5배 초과
"""
import argparse, os, shutil
from collections import defaultdict
import numpy as np

HOME = os.path.expanduser("~")
GAP = 15.032


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
    ap.add_argument("--reg", default=HOME+"/data/fixreg2/chain6/out")
    ap.add_argument("--imgs", default=HOME+"/data/fixreg2/chain6/images")
    ap.add_argument("--merged", default=HOME+"/data/sfm_all/sparse/1")
    ap.add_argument("--out", default=HOME+"/data/scenes/C2")
    a = ap.parse_args()

    I = read_images(os.path.join(a.reg, "images.txt"))
    M = read_images(os.path.join(a.merged, "images.txt"))
    C = {k: -v["R"].T @ v["tv"] for k, v in I.items()}
    gen = sorted(k for k in I if k.startswith("gen_"))
    print("등록된 생성 프레임 %d장" % len(gen))

    A = -M["v1791_00143.png"]["R"].T @ M["v1791_00143.png"]["tv"]
    B = -M["v1793_00001.png"]["R"].T @ M["v1793_00001.png"]["tv"]
    u = (B - A) / np.linalg.norm(B - A)

    # 오등록 걸러내기.
    #   (1) A→B 축 진행도가 통로 범위 밖 → 다른 방을 보고 있다
    #   (2) 축에서 옆으로 너무 벗어남 → 복도 밖에 떠 있다
    # 🔴 '직전 프레임과의 거리' 로 거르면 안 된다. 등록 결과는 프레임 순서대로
    #    늘어서 있지 않아서(순서-진행도 상관 0.49) 멀쩡한 프레임까지 날아간다.
    #    실제로 그 방식은 77장을 3장으로 깎았다.
    prog = {k: float(np.dot(C[k]-A, u)) for k in gen}
    lat = {k: float(np.linalg.norm((C[k]-A) - prog[k]*u)) for k in gen}
    LO, HI, LAT = -1.0, GAP + 1.5, 4.0
    keep = [k for k in gen if LO <= prog[k] <= HI and lat[k] <= LAT]
    print("  진행도 [%.1f, %.1f] & 측면이탈 ≤%.1f : %d장 (%d장 제외)"
          % (LO, HI, LAT, len(keep), len(gen)-len(keep)))
    if not keep:
        print("  🔴 남은 프레임이 없다 — 이 다리로는 G_C 를 만들 수 없다")
        return
    kp0 = np.array([prog[k] for k in keep])
    mid = ((kp0 > 2) & (kp0 < 13)).sum()
    print("  그중 중간 구간(2~13) %d장  %s" % (mid, "✅" if mid >= 5 else "⚠️ 재료 부족"))

    kp = np.array([prog[k] for k in keep])
    print("  남은 진행도 %.2f ~ %.2f,  순서-진행도 상관 %.3f"
          % (kp.min(), kp.max(), np.corrcoef(np.arange(len(kp)), kp)[0, 1]))

    # 데이터셋 쓰기 (포즈 그대로)
    out = a.out
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    sp = os.path.join(out, "colmap", "sparse", "0")
    os.makedirs(sp, exist_ok=True)
    for k in keep:
        shutil.copy(os.path.join(a.imgs, k), os.path.join(out, "images", k))

    pids = set()
    for k in keep:
        for _, _, pid in I[k]["pts"]:
            pids.add(pid)
    P3 = {}
    for l in open(os.path.join(a.reg, "points3D.txt")):
        if l.startswith("#"):
            continue
        v = l.split()
        if len(v) < 8:
            continue
        if int(v[0]) in pids:
            P3[int(v[0])] = v
    print("  생성 프레임이 보는 3D 점 %d개" % len(P3))

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
            f.write("%d %s %s %d %s\n" % (newid[k], " ".join(d["q"]), " ".join(d["t"]),
                                          d["cam"], k))
            f.write(" ".join("%.3f %.3f %d" % (x, y, pidnew.get(pid, -1))
                             for x, y, pid in d["pts"]) + "\n")
    tracks = defaultdict(list)
    for k in keep:
        for idx, (_, _, pid) in enumerate(I[k]["pts"]):
            if pid in pidnew:
                tracks[pid].append((newid[k], idx))
    n = 0
    with open(os.path.join(sp, "points3D.txt"), "w") as f:
        f.write("# 3D point list\n")
        for pid in sorted(P3):
            if len(tracks[pid]) < 2:
                continue
            v = P3[pid]
            f.write("%d %s %s %s\n" % (pidnew[pid], " ".join(v[1:4]), " ".join(v[4:8]),
                                       " ".join("%d %d" % t for t in tracks[pid])))
            n += 1
    print("  기록: 이미지 %d장, 점 %d개 → %s" % (len(keep), n, out))


if __name__ == "__main__":
    main()
