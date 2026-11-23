#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""VGGT 포즈를 합본 좌표계로 옮기고 채점한다 (표 5 4행).

🔴 **GT1 을 정렬에 쓰지 않는다.** VGGT 입력에 A 꼬리·B 머리 실촬영을 함께
   넣었으므로, 그 프레임들의 합본 좌표(COLMAP)와의 대응만으로 Sim(3) 이 구해진다.
   GT1 은 **채점에만** 쓴다.

🔴 Sim(3) 퇴화 주의: 회전은 카메라 중심이 아니라 **자세**에서 구한다
   (중심이 거의 직선이면 축 둘레 회전이 결정되지 않는다).
   그리고 거리비 CV·상관으로 기하가 맞는지 **먼저** 확인한다 —
   잔차만 보면 퇴화와 왜곡을 구분하지 못한다.

🔴 VGGT 의 extrinsic 은 OpenCV world→camera 규약으로 COLMAP 과 같다.
   카메라 중심 C = -R^T t.

입력 npz 의 이름 규칙 (VGGT_다리포즈.sh 가 붙인다):
   a_<실촬영>  A 꼬리 앵커
   m_gen_####  생성 프레임
   z_<실촬영>  B 머리 앵커
"""
import argparse
import os

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
        R = q2R(list(map(float, t[1:5])))
        tv = np.array(list(map(float, t[5:8])))
        out[t[9]] = dict(R=R, C=-R.T @ tv)
    return out


def proj_so3(M):
    U, _, Vt = np.linalg.svd(M)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    return U @ D @ Vt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="VGGT_포즈.py 가 쓴 npz")
    ap.add_argument("--merged", default=HOME+"/data/sfm_all/sparse/1")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="VGGT")
    a = ap.parse_args()

    z = np.load(a.npz, allow_pickle=True)
    names = [str(x) for x in z["names"]]
    extr = z["extrinsic"]
    Cv = np.array([-extr[i][:3, :3].T @ extr[i][:3, 3] for i in range(len(extr))])
    Rv = np.array([extr[i][:3, :3] for i in range(len(extr))])

    M = read_images(os.path.join(a.merged, "images.txt"))

    def orig(n):
        """a_v1791_00143.png -> v1791_00143.png"""
        return n[2:] if len(n) > 2 and n[1] == "_" else n

    anchors = [(i, orig(n)) for i, n in enumerate(names)
               if (n.startswith("a_") or n.startswith("z_")) and orig(n) in M]
    gen = [(i, n) for i, n in enumerate(names) if n.startswith("m_")]
    nA = sum(1 for i, n in anchors if n.startswith("v1791_"))
    nB = sum(1 for i, n in anchors if n.startswith("v1793_"))
    print("%s | 앵커 %d (A %d, B %d) | 생성 %d" % (a.tag, len(anchors), nA, nB, len(gen)))
    if nA < 6 or nB < 6:
        print("  🔴 앵커가 부족하다 — 정렬이 불안정하다"); return

    # ---- 기하가 맞는지 먼저 (잔차만으로는 퇴화·왜곡을 구분 못 한다) ----
    ds, dt = [], []
    for ai in range(len(anchors)):
        for bj in range(ai+1, len(anchors)):
            i, n = anchors[ai]; j, m = anchors[bj]
            ds.append(np.linalg.norm(Cv[i] - Cv[j]))
            dt.append(np.linalg.norm(M[n]["C"] - M[m]["C"]))
    ds, dt = np.array(ds), np.array(dt)
    ok = ds > 1e-9
    ratio = dt[ok] / ds[ok]
    cv = float(np.std(ratio) / max(np.mean(ratio), 1e-12))
    corr = float(np.corrcoef(ds[ok], dt[ok])[0, 1])
    print("  거리비 CV %.1f%%  상관 %.3f" % (100*cv, corr))
    if cv > 0.25 or corr < 0.9:
        print("  🔴 VGGT 기하가 합본과 맞지 않는다. Sim(3) 을 구해도 의미 없다 — 버린다.")
        return

    # ---- Sim(3): 회전은 자세에서, 스케일·평행이동은 중심에서 ----
    Mrot = np.zeros((3, 3))
    for i, n in anchors:
        Mrot += M[n]["R"].T @ Rv[i]
    R = proj_so3(Mrot)
    Ps = np.array([Cv[i] for i, _ in anchors])
    Pt = np.array([M[n]["C"] for _, n in anchors])
    cs, ct = Ps.mean(0), Pt.mean(0)
    Xs, Xt = Ps - cs, Pt - ct
    den = float(np.sum(Xs*Xs))
    s = float(np.sum(Xt * (Xs @ R.T))) / den if den > 1e-12 else 1.0
    t = ct - s*(R @ cs)
    res = np.array([np.linalg.norm(M[n]["C"] - (s*(R @ Cv[i]) + t)) for i, n in anchors])
    print("  Sim(3): 스케일 %.4f | 앵커 잔차 중앙 %.4f, 95%% %.4f"
          % (s, float(np.median(res)), float(np.percentile(res, 95))))

    C = {n: s*(R @ Cv[i]) + t for i, n in gen}

    # ---- 채점 (여기서만 GT1 을 쓴다) ----
    def ix(k):
        return int(k.split("_")[1].split(".")[0])

    GT = sorted((k for k in M if k.startswith("v1791_") and 162 <= ix(k) <= 286),
                key=ix)
    G = np.array([M[k]["C"] for k in GT])
    delta = float(np.median([np.linalg.norm(G[i+1]-G[i]) for i in range(len(G)-1)]))
    THR = 3*delta
    real = np.array([M[k]["C"] for k in M
                     if (k.startswith("v1791_") and ix(k) <= 143)
                     or k.startswith("v1793_")])

    on, cov, dmin = 0, set(), []
    for _, n in gen:
        j = int(np.argmin(np.linalg.norm(G - C[n], axis=1)))
        dg = float(np.linalg.norm(G[j] - C[n]))
        dr = float(np.min(np.linalg.norm(real - C[n], axis=1)))
        dmin.append(min(dg, dr))
        if min(dg, dr) <= THR:
            on += 1
        if dg <= THR:
            cov.add(j)
    dmin = np.array(dmin)
    print("  채점 (임계 %.4f = 간격 %.4f ×3):" % (THR, delta))
    print("    실제 경로 위 %d/%d (%.0f%%) | 중앙거리 %.3f"
          % (on, len(gen), 100*on/max(len(gen), 1), float(np.median(dmin))))
    if cov:
        print("    복도 커버 %d/%d (%.0f%%) 구간 %d~%d"
              % (len(cov), len(GT), 100*len(cov)/len(GT), min(cov), max(cov)))
    else:
        print("    복도 커버 없음")
    print("  📌 오라클(고정 등록) 기준: 복도 위 79%%, 커버 39%% (단일 배치 wp4)")

    if a.out:
        np.savez(a.out, names=np.array([n for _, n in gen]),
                 centers=np.array([C[n] for _, n in gen]), scale=s, R=R, t=t)
        print("  저장 → %s" % a.out)


if __name__ == "__main__":
    main()
