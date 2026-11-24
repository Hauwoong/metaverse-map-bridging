#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A 와 B 를 **각각 따로** 합본에 맞춰, 두 변환이 얼마나 다른지 잰다.

각각은 잘 맞는데 둘이 다르면 → 재구성은 멀쩡하고 **A↔B 상대 자세만** 틀린 것.
(실제로 1차 측정에서 A 잔차 0.0041, B 잔차 0.0027 인데 회전 차이 108.9° 가 나왔다)

🔴 절삭 기준을 **고정 비율에서 잔차 임계로** 바꿨다:
   전에는 매 회 15%씩 6번 떨어내 0.85^6 ≈ 37% 를 무조건 버렸다. 그래서
   "142장 중 52장 사용" 이 **이상치가 90장이라는 뜻이 아니었다** — 수를 해석할 수
   없었다. 이제는 잔차가 로버스트 산포(MAD) 의 k배를 넘는 것만 떨어내므로
   "몇 장이 실제로 어긋났는가" 를 그대로 읽을 수 있다.
"""
import argparse
import os

import numpy as np

H = os.path.expanduser("~")


def q2R(q):
    w, x, y, z = q
    return np.array([[1-2*y*y-2*z*z, 2*x*y-2*z*w,   2*x*z+2*y*w],
                     [2*x*y+2*z*w,   1-2*x*x-2*z*z, 2*y*z-2*x*w],
                     [2*x*z-2*y*w,   2*y*z+2*x*w,   1-2*x*x-2*y*y]])


def rd(p):
    out = {}
    L = [l for l in open(p, encoding="utf-8") if not l.startswith("#")]
    for i in range(0, len(L), 2):
        t = L[i].split()
        if len(t) < 10 or not t[9].endswith(".png"):
            continue
        R = q2R(list(map(float, t[1:5])))
        tv = np.array(list(map(float, t[5:8])))
        out[t[9]] = dict(R=R, C=-R.T @ tv)
    return out


def proj(M):
    U, _, Vt = np.linalg.svd(M)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    return U @ D @ Vt


def fit(S, M, ns):
    """🔴 회전은 카메라 **자세**에서 구한다. 중심이 거의 직선이면
    중심만으로는 축 둘레 회전이 결정되지 않는다 (이 프로젝트에서 한 번 당했다)."""
    Mr = np.zeros((3, 3))
    for n in ns:
        Mr += M[n]["R"].T @ S[n]["R"]
    R = proj(Mr)
    Ps = np.array([S[n]["C"] for n in ns])
    Pt = np.array([M[n]["C"] for n in ns])
    cs, ct = Ps.mean(0), Pt.mean(0)
    Xs, Xt = Ps - cs, Pt - ct
    den = float((Xs*Xs).sum())
    s = float((Xt * (Xs @ R.T)).sum()) / den if den > 1e-12 else 1.0
    t = ct - s*(R @ cs)
    res = np.array([np.linalg.norm(M[n]["C"] - (s*(R @ S[n]["C"]) + t)) for n in ns])
    return s, R, t, res


def trim(S, M, ns, k=3.0, rounds=8):
    """잔차 임계 절삭. 임계 = 중앙 + k·1.4826·MAD (로버스트 산포).
    더 떨어낼 것이 없으면 멈춘다 — 그래서 '사용 n/N' 이 해석 가능하다."""
    cur = list(ns)
    for _ in range(rounds):
        if len(cur) < 8:
            break
        s, R, t, res = fit(S, M, cur)
        med = np.median(res)
        mad = np.median(np.abs(res - med))
        thr = med + k * 1.4826 * mad
        keep = [cur[i] for i in range(len(cur)) if res[i] <= thr]
        if len(keep) == len(cur) or len(keep) < 8:
            break
        cur = keep
    s, R, t, res = fit(S, M, cur)
    return s, R, t, res, cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None, help="GT 없이 만든 모델 폴더")
    ap.add_argument("--tag", default="")
    ap.add_argument("--merged", default=H + "/data/sfm_all/sparse/1")
    a = ap.parse_args()

    src = a.src or open(H + "/data/nogt/wp4/best.txt").read().strip()
    S = rd(os.path.join(src, "images.txt"))
    M = rd(os.path.join(a.merged, "images.txt"))

    def ix(k):
        return int(k.split("_")[1].split(".")[0])

    A = [k for k in S if k in M and k.startswith("v1791_") and ix(k) <= 143]
    Bl = [k for k in S if k in M and k.startswith("v1793_")]
    if len(A) < 8 or len(Bl) < 8:
        print("🔴 A %d, B %d — 한 모델에 둘 다 들어오지 않았다" % (len(A), len(Bl)))
        return

    print("%s%-6s %8s %9s %10s %10s" %
          (a.tag + "\n" if a.tag else "", "장면", "사용", "스케일", "잔차중앙", "잔차95%"))
    print("-" * 48)
    R_ = {}
    for lab, L in (("A", A), ("B", Bl)):
        s, R, t, res, cur = trim(S, M, L)
        R_[lab] = (s, R, t)
        print("%-6s %4d/%-3d %9.4f %10.4f %10.4f"
              % (lab, len(cur), len(L), s, np.median(res), np.percentile(res, 95)))
    sa, Ra, ta = R_["A"]
    sb, Rb, tb = R_["B"]
    ang = float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))
    d = [np.linalg.norm(M[n]["C"] - (sa*(Ra @ S[n]["C"]) + ta)) for n in Bl]
    PB = np.array([M[n]["C"] for n in Bl])
    sz = float(np.linalg.norm(PB - PB.mean(0), axis=1).mean())
    print()
    print("두 변환의 차이")
    print("  스케일 비       %.3f  (1.0 이면 같음)" % (sb / sa))
    print("  회전 차이       %.1f°" % ang)
    print("  A 변환을 B 에 적용 시 어긋남  중앙 %.3f (B 크기 %.3f 의 %.0f%%)"
          % (np.median(d), sz, 100 * np.median(d) / sz))
    print()
    if ang < 5:
        print("  ✅ **두 장면이 같은 자세로 묶였다.** 다리가 회전을 제대로 전달했다.")
    elif ang < 20:
        print("  ⚠️ 회전이 %.1f° 어긋났다 — 이어지긴 했으나 정확하지 않다." % ang)
    else:
        print("  🔴 회전이 %.1f° 틀렸다. 다리가 상대 자세를 정하지 못했다." % ang)
        print("     → 코드/다리_연결강도.py 로 사슬이 끊겼는지 먼저 확인할 것.")


if __name__ == "__main__":
    main()
