#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""GT 없이 얻은 모델을 합본 좌표계로 옮기고, 다리를 채점한다 (표 5).

🔴 왜 정렬이 필요한가:
   `GT없는등록.sh` 는 A+B+생성만으로 SfM 을 처음부터 돌리므로 **자기만의
   좌표계**를 갖는다. 채점(= GT1 궤적과의 거리)을 하려면 합본 좌표계로 옮겨야 한다.

🔴 **정렬에 GT1 을 쓰지 않는다.** A·B 실촬영이 양쪽 모델에 모두 있으므로
   그 대응만으로 Sim(3) 을 구한다. GT1 은 **채점에만** 쓴다.

🔴 Sim(3) 퇴화 주의 (이 프로젝트에서 한 번 당했다):
   카메라 중심이 거의 한 직선 위에 있으면 축 둘레 회전이 결정되지 않는다.
   그래서 **회전은 카메라 자세에서** 따로 구한다:
       R = proj_SO(3)( Σ_k R_k^tgt (R_k^src)^T )
   그리고 **거리비의 변동계수(CV)** 로 퇴화·왜곡을 먼저 확인한다.
   잔차만 보면 퇴화와 왜곡을 구분하지 못한다.
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
    """이름 -> (R_cw, t_cw, C=카메라 중심)"""
    out = {}
    lines = [l for l in open(p) if not l.startswith("#")]
    for i in range(0, len(lines), 2):
        t = lines[i].split()
        if len(t) < 10:
            continue
        R = q2R(list(map(float, t[1:5])))
        tv = np.array(list(map(float, t[5:8])))
        out[t[9]] = dict(R=R, t=tv, C=-R.T @ tv)
    return out


def proj_so3(M):
    U, _, Vt = np.linalg.svd(M)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    return U @ D @ Vt


def sim3_from_poses(src, tgt, names):
    """회전은 자세에서, 스케일·평행이동은 중심에서 구한다.
    반환: s, R, t  (적용: X_tgt = s R X_src + t)"""
    Ps = np.array([src[n]["C"] for n in names])
    Pt = np.array([tgt[n]["C"] for n in names])
    # 회전: 카메라 자세의 평균 상대회전 (중심이 직선이어도 결정된다)
    M = np.zeros((3, 3))
    for n in names:
        M += tgt[n]["R"].T @ src[n]["R"]       # 월드→카메라 이므로 전치 주의
    R = proj_so3(M)
    # 스케일: 중심 간 거리비
    cs, ct = Ps.mean(0), Pt.mean(0)
    Xs, Xt = Ps - cs, Pt - ct
    num = float(np.sum(Xt * (Xs @ R.T)))
    den = float(np.sum(Xs * Xs))
    s = num / den if den > 1e-12 else 1.0
    t = ct - s * (R @ cs)
    return s, R, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="GT 없이 만든 모델 폴더 (images.txt)")
    ap.add_argument("--merged", default=HOME+"/data/sfm_all/sparse/1",
                    help="합본 모델 (채점 기준 좌표계)")
    ap.add_argument("--out", default=None, help="변환된 생성 프레임 포즈를 쓸 npz")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    S = read_images(os.path.join(a.src, "images.txt"))
    M = read_images(os.path.join(a.merged, "images.txt"))

    def ix(k):
        return int(k.split("_")[1].split(".")[0])

    common = sorted(k for k in S if k in M and not k.startswith("gen_"))
    gen = sorted(k for k in S if k.startswith("gen_"))
    nA = sum(1 for k in common if k.startswith("v1791_"))
    nB = sum(1 for k in common if k.startswith("v1793_"))
    print("%s공통 실촬영 %d장 (A %d, B %d) | 생성 %d장"
          % (a.tag + " " if a.tag else "", len(common), nA, nB, len(gen)))
    if nA < 8 or nB < 8:
        print("  🔴 A 또는 B 가 너무 적다 — 두 장면이 한 모델에 들어오지 않았을 수 있다")
    if len(common) < 8:
        print("  🔴 정렬 불가"); return

    # 🔴 퇴화·왜곡을 먼저 본다 — 잔차만으로는 구분되지 않는다
    idx = np.linspace(0, len(common)-1, min(60, len(common))).astype(int)
    sub = [common[i] for i in idx]
    ds, dt = [], []
    for i in range(len(sub)):
        for j in range(i+1, len(sub)):
            ds.append(np.linalg.norm(S[sub[i]]["C"] - S[sub[j]]["C"]))
            dt.append(np.linalg.norm(M[sub[i]]["C"] - M[sub[j]]["C"]))
    ds, dt = np.array(ds), np.array(dt)
    ok = ds > 1e-9
    ratio = dt[ok] / ds[ok]
    cv = float(np.std(ratio) / max(np.mean(ratio), 1e-12))
    corr = float(np.corrcoef(ds[ok], dt[ok])[0, 1])
    print("  거리비 CV %.1f%%  상관 %.3f" % (100*cv, corr))
    if cv > 0.25 or corr < 0.9:
        print("  🔴 두 모델의 기하가 맞지 않는다 (CV>25% 또는 상관<0.9).")
        print("     Sim(3) 을 구해도 의미가 없다 — 이 결과는 쓰지 않는다.")
        return

    s, R, t = sim3_from_poses(S, M, common)
    res = np.array([np.linalg.norm(M[n]["C"] - (s*(R @ S[n]["C"]) + t)) for n in common])
    print("  Sim(3): 스케일 %.4f | 실촬영 잔차 중앙 %.4f, 95%% %.4f"
          % (s, np.median(res), np.percentile(res, 95)))

    if not gen:
        print("  생성 프레임이 없다"); return

    C = {k: s*(R @ S[k]["C"]) + t for k in gen}

    # ---- 채점: GT1 궤적과의 거리 (여기서만 GT1 을 쓴다) ----
    GT = sorted((k for k in M if k.startswith("v1791_") and 162 <= ix(k) <= 286),
                key=ix)
    G = np.array([M[k]["C"] for k in GT])
    delta = float(np.median([np.linalg.norm(G[i+1]-G[i]) for i in range(len(G)-1)]))
    THR = 3*delta
    real = np.array([M[k]["C"] for k in M
                     if (k.startswith("v1791_") and ix(k) <= 143)
                     or k.startswith("v1793_")])

    on, cov, dmin = 0, set(), []
    for k in gen:
        j = int(np.argmin(np.linalg.norm(G - C[k], axis=1)))
        dg = float(np.linalg.norm(G[j] - C[k]))
        dr = float(np.min(np.linalg.norm(real - C[k], axis=1)))
        dmin.append(min(dg, dr))
        if min(dg, dr) <= THR:
            on += 1
        if dg <= THR:
            cov.add(j)
    dmin = np.array(dmin)
    print("  채점 (임계 %.4f = 간격 %.4f ×3):" % (THR, delta))
    print("    실제 경로 위 %d/%d (%.0f%%) | 중앙거리 %.3f"
          % (on, len(gen), 100*on/len(gen), float(np.median(dmin))))
    if cov:
        print("    복도 커버 %d/%d (%.0f%%) 구간 %d~%d"
              % (len(cov), len(GT), 100*len(cov)/len(GT), min(cov), max(cov)))
    else:
        print("    복도 커버 없음")

    if a.out:
        np.savez(a.out, names=np.array(gen),
                 centers=np.array([C[k] for k in gen]),
                 scale=s, R=R, t=t)
        print("  저장 → %s" % a.out)


if __name__ == "__main__":
    main()
