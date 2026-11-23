#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""기하 불일치가 **어디서** 생기는지 분해한다.

🔴 왜 필요한가:
   GT 없는 등록이 A 와 B 를 한 모델에 넣는 데는 성공했지만, 합본과의
   거리비 CV 가 74.7%(VGGT 는 108%)로 기하가 맞지 않았다.
   그런데 "전부 틀렸다" 와 "A↔B 관계만 틀렸다" 는 전혀 다른 문제다.
     - 장면 안(A–A, B–B)이 맞고 A–B 만 틀리면 → **다리가 약해 상대 위치가
       안 잡힌 것**. 다리를 보강하면 된다.
     - 장면 안도 틀리면 → 재구성 자체가 나쁜 것.
   쌍을 세 묶음으로 나눠 따로 본다.

🔴 거리비의 변동계수(CV)로 본다. 잔차는 퇴화와 왜곡을 구분하지 못한다.
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


def read_centers(p):
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


def proj_so3(M):
    U, _, Vt = np.linalg.svd(M)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    return U @ D @ Vt


def fit_sim3(Ps, Pt):
    """중심만으로 Sim(3). 묶음 안의 적합도를 보려는 것이므로 Umeyama 로 충분하다."""
    cs, ct = Ps.mean(0), Pt.mean(0)
    Xs, Xt = Ps - cs, Pt - ct
    U, S, Vt = np.linalg.svd(Xt.T @ Xs / len(Ps))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    var = float((Xs**2).sum()/len(Ps))
    s = float((S*np.diag(D)).sum()/var) if var > 1e-12 else 1.0
    t = ct - s*(R @ cs)
    res = np.linalg.norm(Pt - (s*(Ps @ R.T) + t), axis=1)
    return s, float(np.median(res)), float(np.percentile(res, 95))


def stats(ds, dt, frac=0.2):
    """거리비 통계. 🔴 인접 프레임 거리(0.02 수준)는 두 재구성 모두에서
    잡음이 커서 비율을 폭발시킨다. **큰 거리 쌍만** 쓴다 (기본: 상위 80%)."""
    ds, dt = np.array(ds), np.array(dt)
    if len(ds) < 10:
        return None
    lo = np.quantile(ds[ds > 1e-9], frac) if (ds > 1e-9).any() else 0.0
    ok = ds > max(lo, 1e-9)
    if ok.sum() < 10:
        return None
    r = dt[ok]/ds[ok]
    return (float(np.std(r)/max(np.mean(r), 1e-12)),
            float(np.corrcoef(ds[ok], dt[ok])[0, 1]),
            float(np.median(r)), int(ok.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None, help="COLMAP 모델 폴더")
    ap.add_argument("--npz", default=None, help="VGGT npz (--src 대신)")
    ap.add_argument("--merged", default=HOME+"/data/sfm_all/sparse/1")
    ap.add_argument("--tag", default="")
    ap.add_argument("--n", type=int, default=50, help="묶음마다 뽑을 프레임 수")
    a = ap.parse_args()

    M = read_centers(os.path.join(a.merged, "images.txt"))

    if a.src:
        S = read_centers(os.path.join(a.src, "images.txt"))
    else:
        z = np.load(a.npz, allow_pickle=True)
        names = [str(x) for x in z["names"]]
        e = z["extrinsic"]
        S = {}
        for i, n in enumerate(names):
            o = n[2:] if len(n) > 2 and n[1] == "_" else n
            S[o] = -e[i][:3, :3].T @ e[i][:3, 3]

    def ix(k):
        return int(k.split("_")[1].split(".")[0])

    common = [k for k in S if k in M and not k.startswith("gen_")]
    A = sorted([k for k in common if k.startswith("v1791_") and ix(k) <= 143], key=ix)
    B = sorted([k for k in common if k.startswith("v1793_")], key=ix)
    if len(A) < 8 or len(B) < 8:
        print("🔴 A %d, B %d — 너무 적다" % (len(A), len(B))); return

    def thin(L, n):
        if len(L) <= n:
            return L
        return [L[i] for i in np.linspace(0, len(L)-1, n).astype(int)]

    A, B = thin(A, a.n), thin(B, a.n)
    print("%s | A %d장, B %d장" % (a.tag or "분해", len(A), len(B)))
    print()
    print("%-10s %8s %8s %10s %8s   %s" % ("묶음", "CV", "상관", "중앙비", "쌍수", "판정"))
    print("-"*66)

    def pairs(L1, L2, same):
        ds, dt = [], []
        for i, x in enumerate(L1):
            for j, y in enumerate(L2):
                if same and j <= i:
                    continue
                ds.append(np.linalg.norm(S[x]-S[y]))
                dt.append(np.linalg.norm(M[x]-M[y]))
        return ds, dt

    out = {}
    for lab, (L1, L2, same) in (("A 안", (A, A, True)),
                                ("B 안", (B, B, True)),
                                ("A↔B", (A, B, False))):
        r = stats(*pairs(L1, L2, same))
        if r is None:
            print("%-10s  (쌍 부족)" % lab); continue
        cv, corr, med, n = r
        ok = cv <= 0.25 and corr >= 0.9
        print("%-10s %7.1f%% %8.3f %10.4f %8d   %s"
              % (lab, 100*cv, corr, med, n, "✅ 일치" if ok else "🔴 어긋남"))
        out[lab] = (cv, corr, med, ok)

    # 🔴 묶음마다 Sim(3) 을 따로 맞춰 본다 — 거리비보다 해석이 분명하다.
    #    "그 묶음만 놓고 보면 두 재구성이 같은 모양인가" 를 직접 답한다.
    print()
    print("%-10s %10s %12s %12s   %s"
          % ("묶음", "스케일", "잔차 중앙", "잔차/크기", "판정"))
    print("-"*62)
    for lab, L in (("A 안", A), ("B 안", B), ("A+B", A+B)):
        Ps = np.array([S[k] for k in L])
        Pt = np.array([M[k] for k in L])
        sc, med, p95 = fit_sim3(Ps, Pt)
        size = float(np.linalg.norm(Pt - Pt.mean(0), axis=1).mean())
        rel = med/max(size, 1e-9)
        print("%-10s %10.4f %12.4f %11.1f%%   %s"
              % (lab, sc, med, 100*rel, "✅ 같은 모양" if rel < 0.05 else "🔴 다름"))
        out["sim3_"+lab] = rel

    print()
    sa, sb, sab = out.get("sim3_A 안"), out.get("sim3_B 안"), out.get("sim3_A+B")
    if sa is not None and sb is not None and sab is not None:
        print()
        if sa < 0.05 and sb < 0.05 and sab >= 0.05:
            print("📌 **장면 안은 같은 모양인데 A+B 를 함께 맞추면 어긋난다.**")
            print("   → 재구성 자체는 멀쩡하고, **두 장면의 상대 위치**가 틀렸다.")
            print("   다리가 그만큼 약하다는 뜻이고, 보강할 여지가 있다.")
        elif sa >= 0.05 or sb >= 0.05:
            print("🔴 **장면 안부터 어긋난다.** 다리 보강으로는 해결되지 않는다.")
        else:
            print("✅ 세 경우 모두 같은 모양이다. 기하가 맞다.")

    ia = out.get("A 안"); ib = out.get("B 안"); ic = out.get("A↔B")
    if False and ia and ib and ic:
        if ia[3] and ib[3] and not ic[3]:
            print("📌 **장면 안은 맞고 A↔B 관계만 어긋난다.**")
            print("   재구성 자체는 멀쩡하다. 다리가 두 장면의 **상대 위치를**")
            print("   잡아줄 만큼 강하지 않은 것이다.")
            print("   중앙 거리비: A 안 %.4f, B 안 %.4f, A↔B %.4f"
                  % (ia[2], ib[2], ic[2]))
            if ia[2] > 1e-9:
                print("   → A↔B 가 장면 안보다 %.2f배로 잡혔다 (1.0 이면 정확)."
                      % (ic[2]/ia[2]))
            print("   보강 방향: 다리 프레임을 늘리거나, 양 끝 겹침을 키운다.")
        elif not ia[3] or not ib[3]:
            print("🔴 **장면 안부터 어긋난다.** 재구성 자체가 나쁘다.")
            print("   다리를 보강해도 해결되지 않는다.")
        else:
            print("✅ 세 묶음 모두 일치한다. 기하가 맞다.")


if __name__ == "__main__":
    main()
