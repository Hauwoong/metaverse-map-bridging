#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""한 장면이 **자기 영상만으로** 제대로 재구성되는가를 검사한다.

🔴 왜 이것을 재야 하는가 (2026-11-25):
   GT1 없는 등록이 계속 실패해서 다리를 의심했는데, 2단계 증분 등록에서
   **A 를 자기 143장만으로 세운 블록조차** 합본의 A 와 어긋났다
   (CV 78.6%, 상관 0.244, 잔차/크기 87%).
   다리가 문제가 아니라 **장면 자체가 혼자서는 안 서는** 것일 수 있다.

🔴 그리고 어제 낸 수치를 의심해야 한다:
   "A 잔차 0.0041 — 재구성이 거의 완벽하다" 는 **고정 비율 절삭**(0.85^6 ≈ 37% 만 남김)
   으로 142장 중 52장만 쓴 결과였다. 잔차 임계로 바꾸니 135/142 를 쓰고 잔차 0.31 이 된다.
   🔴 **절삭을 세게 하면 어떤 재구성도 '완벽' 해 보인다.** 그 수치는 철회해야 한다.

재는 것:
   ① 궤적이 매끄러운가 — 인접 프레임 간격의 분포 (촬영은 등속에 가깝다)
   ② 합본과 **모양**이 같은가 — 거리비 CV·상관, Umeyama 잔차 (절삭 없이)
   ③ 전방 이동 퇴화 징후 — 이동 방향과 광축이 이루는 각
      🔴 광축 방향으로만 움직이면 SfM 은 구조적으로 불안정하다(깊이와 전진이 섞인다).
         복도·통로 촬영의 전형적인 실패 원인이다.

사용:
   python 코드/장면_자립검사.py --src ~/data/nogt/r5/sparseA/1 --scene A
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


def rd(p):
    out = {}
    L = [l for l in open(p, encoding="utf-8") if not l.startswith("#")]
    for i in range(0, len(L), 2):
        t = L[i].split()
        if len(t) < 10:
            continue
        R = q2R(list(map(float, t[1:5])))
        tv = np.array(list(map(float, t[5:8])))
        # 🔴 월드→카메라. 중심 C = -R^T t, 광축(전방) = R^T [0,0,1]
        out[t[9]] = dict(C=-R.T @ tv, F=R.T @ np.array([0.0, 0.0, 1.0]))
    return out


def umeyama(Ps, Pt):
    cs, ct = Ps.mean(0), Pt.mean(0)
    Xs, Xt = Ps - cs, Pt - ct
    U, S, Vt = np.linalg.svd(Xt.T @ Xs / len(Ps))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    var = float((Xs**2).sum() / len(Ps))
    s = float((S * np.diag(D)).sum() / var) if var > 1e-12 else 1.0
    t = ct - s * (R @ cs)
    return s, np.linalg.norm(Pt - (s * (Ps @ R.T) + t), axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None, help="COLMAP 모델 폴더")
    ap.add_argument("--npz", default=None, help="VGGT npz (--src 대신)")
    ap.add_argument("--merged", default=HOME + "/data/sfm_all/sparse/1")
    ap.add_argument("--scene", default="A", choices=["A", "B"])
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    if a.npz:
        # 🔴 VGGT 의 extrinsic 은 OpenCV world→camera 로 COLMAP 과 같은 규약이다.
        z = np.load(a.npz, allow_pickle=True)
        e = z["extrinsic"]
        S = {}
        for i, n in enumerate([str(x) for x in z["names"]]):
            o = n[2:] if len(n) > 2 and n[1] == "_" else n   # a_/m_/z_ 접두 제거
            R = e[i][:3, :3]
            S[o] = dict(C=-R.T @ e[i][:3, 3], F=R.T @ np.array([0.0, 0.0, 1.0]))
    else:
        S = rd(os.path.join(a.src, "images.txt"))
    M = rd(os.path.join(a.merged, "images.txt"))

    def ix(k):
        return int(k.split("_")[1].split(".")[0])

    if a.scene == "A":
        sel = lambda k: k.startswith("v1791_") and ix(k) <= 143
    else:
        sel = lambda k: k.startswith("v1793_")
    L = sorted([k for k in S if sel(k)], key=ix)
    if len(L) < 20:
        print("🔴 %s 프레임이 %d장뿐이다" % (a.scene, len(L))); return
    print("== 장면 %s 자립 검사%s — %d장 ==" % (a.scene, " ("+a.tag+")" if a.tag else "", len(L)))

    # ---- ① 궤적이 매끄러운가 ----
    C = np.array([S[k]["C"] for k in L])
    d = np.linalg.norm(np.diff(C, axis=0), axis=1)
    med = float(np.median(d))
    print("\n① 궤적 매끄러움 (인접 프레임 간격)")
    print("   중앙 %.4f | 95%% %.4f | 최대 %.4f | 최대/중앙 %.0f배"
          % (med, np.percentile(d, 95), d.max(), d.max() / max(med, 1e-9)))
    jump = int((d > 5 * med).sum())
    print("   중앙의 5배를 넘는 도약 %d곳 (%.0f%%)  %s"
          % (jump, 100.0 * jump / len(d),
             "✅ 매끄럽다" if jump <= 0.02 * len(d) else "🔴 궤적이 튄다"))

    # ---- ③ 전방 이동 퇴화 ----
    # 🔴 이동 방향이 광축과 거의 같으면(각이 작으면) SfM 이 구조적으로 불안정하다.
    F = np.array([S[k]["F"] for k in L])
    v = np.diff(C, axis=0)
    nv = np.linalg.norm(v, axis=1)
    ok = nv > 1e-9
    cosang = np.abs(np.sum(v[ok] / nv[ok, None] * F[:-1][ok], axis=1))
    ang = np.degrees(np.arccos(np.clip(cosang, 0, 1)))
    print("\n③ 전방 이동 퇴화 (이동 방향과 광축이 이루는 각)")
    print("   중앙 %.1f° | 25%% %.1f°  %s"
          % (np.median(ang), np.percentile(ang, 25),
             "🔴 거의 광축 방향 — 퇴화에 가깝다" if np.median(ang) < 25
             else "✅ 측면 성분이 있다"))

    # ---- ② 합본과 모양이 같은가 (절삭 없이) ----
    LM = [k for k in L if k in M]
    if len(LM) < 20:
        print("\n🔴 합본에 같은 프레임이 %d장뿐 — 비교 불가" % len(LM)); return
    idx = np.linspace(0, len(LM) - 1, min(60, len(LM))).astype(int)
    sub = [LM[i] for i in idx]
    ds, dt = [], []
    for i in range(len(sub)):
        for j in range(i + 1, len(sub)):
            ds.append(np.linalg.norm(S[sub[i]]["C"] - S[sub[j]]["C"]))
            dt.append(np.linalg.norm(M[sub[i]]["C"] - M[sub[j]]["C"]))
    ds, dt = np.array(ds), np.array(dt)
    g = ds > np.quantile(ds, 0.2)
    r = dt[g] / ds[g]
    cv = float(np.std(r) / max(np.mean(r), 1e-12))
    corr = float(np.corrcoef(ds[g], dt[g])[0, 1])
    Ps = np.array([S[k]["C"] for k in LM])
    Pt = np.array([M[k]["C"] for k in LM])
    s, res = umeyama(Ps, Pt)
    size = float(np.linalg.norm(Pt - Pt.mean(0), axis=1).mean())
    print("\n② 합본과 같은 모양인가 (**절삭 없이** 전체 %d장)" % len(LM))
    print("   거리비 CV %.1f%% | 상관 %.3f" % (100 * cv, corr))
    print("   Umeyama 스케일 %.4f | 잔차 중앙 %.4f = 장면 크기의 %.1f%%"
          % (s, np.median(res), 100 * np.median(res) / max(size, 1e-9)))
    good = cv <= 0.25 and corr >= 0.9 and np.median(res) / max(size, 1e-9) < 0.05
    print("   %s" % ("✅ 이 장면은 자기 영상만으로 선다" if good
                     else "🔴 **이 장면은 자기 영상만으로 서지 않는다**"))

    # 🔴 절삭이 얼마나 숫자를 바꾸는지 같이 보여 준다 — 어제 이것에 속았다
    k = max(8, int(len(res) * 0.37))
    print("\n   ⚠️ 참고: 잔차 작은 순 %d장(37%%)만 남기면 잔차 중앙 %.4f 로 떨어진다."
          % (k, np.median(np.sort(res)[:k])))
    print("      **절삭을 세게 하면 어떤 재구성도 완벽해 보인다.** 비율 절삭을 쓰지 말 것.")


if __name__ == "__main__":
    main()
