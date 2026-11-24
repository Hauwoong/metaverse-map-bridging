#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""다리가 A 와 B 를 **무엇을 통해** 묶고 있는지 진단한다.

🔴 왜 이것을 먼저 재는가 (2026-11-25):
   GT1 없는 등록은 A·B 를 한 모델에 넣었지만 **상대 회전이 108.9° 틀렸다.**
   원인이 어느 쪽인지에 따라 처방이 **정반대**다.
     (가) 연결은 충분한데 전역 BA 가 국소 최소에 빠졌다 → BA 를 더 돌리면 된다
     (나) 연결이 가늘어 회전이 애초에 결정되지 않는다   → 다리를 더 만들어야 한다
     (다) **A·B 가 다리를 거치지 않고 서로 직접 매칭됐다** → 그 매칭이 거짓이면
          BA 가 두 장면을 엉뚱한 자세로 끌어다 붙인다. 걸러내야 한다.
   잔차·CV 로는 구분되지 않는다. **무엇이 두 장면 사이를 지나는지**를 세야 한다.

🔴 합계를 쌍당 값과 섞지 않는다 (한 번 당했다):
   A↔B 인라이어 **총합** 52,999 는 커 보이지만 22,294 쌍에 흩어지면 쌍당 2.4 다.
   `two_view_geometries` 에는 검증을 통과한 쌍만 남으므로, **쌍 수와 쌍당 분포**를
   같이 봐야 한다. 총합만 보면 가는 연결을 두터운 연결로 오판한다.

🔴 거짓 매칭을 어떻게 판정하는가 — **합본(정답) 모델과 교차 검증**:
   합본에서 같은 A·B 이미지 쌍이 3D 점을 공유하는지 본다. 정답에서 공유가 없는데
   GT1 없는 모델에서 많다면 그 매칭은 거짓이다(복도의 반복 구조·문·벽).
   ⚠️ 이것은 **진단**에만 GT1 을 쓴다. 방법(포즈 산출)에는 쓰지 않는다.

사용:
   python 코드/다리_연결강도.py --dir ~/data/nogt/wp4
"""
import argparse
import os
import re
import sqlite3

import numpy as np

HOME = os.path.expanduser("~")
MAX_ID = 2147483647


def ix(name):
    m = re.search(r"_(\d+)\.", name)
    return int(m.group(1)) if m else -1


def group(name):
    if name.startswith("gen_"):
        return "gen"
    if name.startswith("v1791_"):
        return "A" if ix(name) <= 143 else "GT1"
    if name.startswith("v1793_"):
        return "B"
    return "?"


def read_model(p):
    """images.txt -> {name: set(point3D_id)}"""
    obs = {}
    with open(p, encoding="utf-8") as f:
        lines = [l for l in f if not l.startswith("#")]
    for i in range(0, len(lines) - 1, 2):
        t = lines[i].split()
        if len(t) < 10:
            continue
        pts = lines[i + 1].split()
        obs[t[9]] = set(int(pts[j]) for j in range(2, len(pts), 3) if int(pts[j]) >= 0)
    return obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=HOME + "/data/nogt/wp4")
    ap.add_argument("--model", default=None)
    ap.add_argument("--merged", default=HOME + "/data/sfm_all/sparse/1")
    ap.add_argument("--strong", type=int, default=100,
                    help="상대 자세를 실제로 구속한다고 볼 인라이어 하한")
    a = ap.parse_args()

    mdir = a.model
    if mdir is None:
        bt = os.path.join(a.dir, "best.txt")
        mdir = open(bt).read().strip() if os.path.exists(bt) \
            else os.path.join(a.dir, "sparse/0")
    obs = read_model(os.path.join(mdir, "images.txt"))

    con = sqlite3.connect(os.path.join(a.dir, "db.db"))
    cur = con.cursor()
    id2name = {i: n for i, n in cur.execute("SELECT image_id, name FROM images")}
    pairs = {}
    for pid, rows in cur.execute("SELECT pair_id, rows FROM two_view_geometries"):
        if not rows:
            continue
        na, nb = id2name.get(pid // MAX_ID), id2name.get(pid % MAX_ID)
        if na and nb:
            pairs[(na, nb)] = rows
    con.close()

    reg = set(obs)
    # ---- 0) 등록 현황 ----
    print("== 등록 현황 (DB 전체 기준) ==")
    for g in ("A", "GT1", "B", "gen"):
        tot = sum(1 for n in id2name.values() if group(n) == g)
        got = sum(1 for n in reg if group(n) == g)
        if tot:
            print("   %-4s %3d/%3d 등록 (%3.0f%%)" % (g, got, tot, 100.0 * got / tot))
    miss = sorted((n for n in id2name.values()
                   if group(n) == "gen" and n not in reg), key=ix)
    if miss:
        mi = [ix(n) for n in miss]
        runs, s = [], mi[0]
        for p, q in zip(mi, mi[1:] + [None]):
            if q != p + 1:
                runs.append((s, p)); s = q
        print("   🔴 등록 실패한 생성 프레임 %d장: %s"
              % (len(miss), ", ".join("%d~%d" % r if r[0] != r[1] else str(r[0])
                                      for r in runs)))
        big = max(runs, key=lambda r: r[1] - r[0])
        if big[1] - big[0] >= 5:
            print("      📌 %d~%d 가 **연속으로** 빠졌다 — 그 구간에서 사슬이 끊겼다"
                  % big)

    # ---- 1) 묶음별 쌍 통계 (합계·쌍수·쌍당 분포) ----
    GRP = ("A", "gen", "B")
    buck = {}
    for (x, y), r in pairs.items():
        gx, gy = group(x), group(y)
        if gx not in GRP or gy not in GRP:
            continue
        k = tuple(sorted((gx, gy), key=GRP.index))
        buck.setdefault(k, []).append(r)

    print("\n== 쌍 통계 — 두 시점 기하가 검증된 쌍만 ==")
    print("%-10s %8s %12s %9s %9s %10s" %
          ("묶음", "쌍수", "인라이어합", "쌍당중앙", "쌍당최대", "강한쌍(≥%d)" % a.strong))
    print("-" * 64)
    for k in (("A", "A"), ("A", "gen"), ("gen", "gen"), ("gen", "B"),
              ("B", "B"), ("A", "B")):
        v = buck.get(k)
        lab = "%s↔%s" % k
        if not v:
            print("%-10s %8d %12d %9s %9s %10d" % (lab, 0, 0, "-", "-", 0))
            continue
        v = np.array(v)
        print("%-10s %8d %12d %9.0f %9d %10d"
              % (lab, len(v), v.sum(), np.median(v), v.max(), int((v >= a.strong).sum())))

    # ---- 2) A↔B 직접 연결이 **강한** 쌍이 있는가 ----
    ab = {k: r for k, r in pairs.items()
          if {group(k[0]), group(k[1])} == {"A", "B"}}
    strong_ab = sorted(((r, k) for k, r in ab.items() if r >= a.strong), reverse=True)
    print("\n== A↔B 직접 매칭 ==")
    print("   검증된 쌍 %d, 그 중 강한 쌍(≥%d) **%d개**" % (len(ab), a.strong, len(strong_ab)))
    if strong_ab:
        print("   상위 8쌍:")
        for r, (x, y) in strong_ab[:8]:
            sh = len(obs.get(x, set()) & obs.get(y, set()))
            print("      %-18s %-18s 인라이어 %4d  공유트랙 %4d" % (x, y, r, sh))

    # ---- 3) 합본(정답)과 교차 검증 — 그 매칭이 참인가 ----
    mp = os.path.join(a.merged, "images.txt")
    if strong_ab and os.path.exists(mp):
        MO = read_model(mp)
        print("\n== 합본(정답)에서 같은 쌍은 무엇을 공유하는가 ==")
        print("   ⚠️ 진단 전용. 포즈 산출에는 GT1 을 쓰지 않는다.")
        t_no, t_me = [], []
        for r, (x, y) in strong_ab:
            if x in MO and y in MO:
                t_no.append(len(obs[x] & obs[y]))
                t_me.append(len(MO[x] & MO[y]))
        if t_me:
            t_no, t_me = np.array(t_no), np.array(t_me)
            print("   강한 A↔B 쌍 %d개 중 합본에 둘 다 있는 것 %d개" % (len(strong_ab), len(t_me)))
            print("   공유트랙 중앙값:  GT1없는모델 %d   vs   합본(정답) %d"
                  % (np.median(t_no), np.median(t_me)))
            print("   합본에서 공유가 0 인 쌍: %d/%d (%.0f%%)"
                  % ((t_me == 0).sum(), len(t_me), 100.0 * (t_me == 0).mean()))
            if (t_me == 0).mean() > 0.7:
                print("\n   🔴 **(다) 거짓 매칭이다.** 정답에서는 A 와 B 가 같은 것을 보지")
                print("      않는데도 특징이 매칭됐다 — 복도의 반복 구조다.")
                print("      BA 는 이 거짓 대응을 만족시키려고 두 장면을 **엉뚱한 자세로**")
                print("      끌어다 붙인다. 이것이 109° 의 원인으로 가장 유력하다.")
                print("      처방: A↔B 직접 매칭을 **금지**하고 다리만 통해 잇게 한다.")
            else:
                print("\n   ⚠️ 상당수가 정답에서도 공유한다 — 거짓 매칭만으로는 설명되지 않는다.")

    # ---- 4) 다리 사슬이 실제로 이어져 있는가 ----
    G = sorted((n for n in reg if group(n) == "gen"), key=ix)
    if len(G) >= 2:
        print("\n== 다리 사슬의 이음새 (등록된 생성 프레임 사이) ==")
        gaps = []
        for x, y in zip(G, G[1:]):
            sh = len(obs[x] & obs[y])
            gaps.append((sh, x, y, ix(y) - ix(x)))
        gaps.sort()
        print("   가장 약한 이음새 5곳:")
        for sh, x, y, d in gaps[:5]:
            print("      %-13s → %-13s 번호차 %2d  공유트랙 %5d" % (x, y, d, sh))
        allsh = np.array([g[0] for g in gaps])
        print("   공유트랙 중앙 %d, 최소 %d" % (np.median(allsh), allsh.min()))
        if allsh.min() < 50:
            print("   🔴 사슬이 끊긴 곳이 있다 — 그 틈을 생성으로 메워야 한다")

    # ---- 5) 실패한 구간은 **영상이 나쁜 것인가, 겹침이 모자란 것인가** ----
    #    🔴 처방이 또 갈린다:
    #       특징점 자체가 적다 → 생성 영상이 그 구간에서 뭉개졌다 (다시 생성해야 한다)
    #       특징점은 많은데 매칭이 없다 → 시점이 너무 벌어졌다 (프레임을 촘촘히)
    con = sqlite3.connect(os.path.join(a.dir, "db.db"))
    cur = con.cursor()
    kp = {}
    for iid, rows in cur.execute("SELECT image_id, rows FROM keypoints"):
        n = id2name.get(iid)
        if n:
            kp[n] = rows
    con.close()
    gk_ok = np.array([kp[n] for n in kp if group(n) == "gen" and n in reg])
    gk_no = np.array([kp[n] for n in kp if group(n) == "gen" and n not in reg])
    rk = np.array([kp[n] for n in kp if group(n) in ("A", "B")])
    if len(gk_no) and len(gk_ok):
        print("\n== 특징점 수 (SIFT) ==")
        print("   실촬영          중앙 %5d" % np.median(rk))
        print("   생성·등록성공    중앙 %5d  (실촬영의 %.2f배)"
              % (np.median(gk_ok), np.median(gk_ok) / np.median(rk)))
        print("   생성·등록실패    중앙 %5d  (실촬영의 %.2f배)"
              % (np.median(gk_no), np.median(gk_no) / np.median(rk)))
        if np.median(gk_no) < 0.6 * np.median(gk_ok):
            print("   🔴 **실패 구간은 영상 자체가 뭉개졌다.** 프레임을 더 촘촘히 뽑아도")
            print("      소용없다 — 그 구간을 **다시 생성**해야 한다.")
        else:
            print("   📌 특징점은 비슷하다 — 영상 품질이 아니라 **겹침 부족**이다.")
            print("      같은 클립에서 프레임을 더 촘촘히 뽑으면 이어질 수 있다.")


if __name__ == "__main__":
    main()
