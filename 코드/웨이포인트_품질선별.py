#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""웨이포인트를 **그 자리의 복원 품질**로 거른 뒤 고른다.

🔴 왜 필요한가 (2026-11-25, 109° 회전 오차의 뿌리):
   wp4 의 W1 은 GT1 을 균등 분할해 고른 자리였는데, 거기서 G_GT1 을 렌더하니
   **형체가 없는 뭉갬**이 나왔다. 끝점 조건이 뭉개져 있으니 LTX 가 만든 구간도
   뭉개졌고(SIFT 특징 65개 vs 실촬영 4201), 그 20장이 등록에 실패해
   **포즈 사슬이 공유트랙 0 으로 끊겼다.** 거기서 A↔B 상대 회전이 풀려 108.9° 가 됐다.

     저텍스처 복도 입구 → 3DGS 복원 불량 → 웨이포인트 렌더 뭉갬
       → 생성 영상 뭉갬 → 등록 실패 20장 → 포즈 사슬 절단 → 회전 109° 오차

   이미 알고 있던 저텍스처 한계(GT1 0~19 관측점 667 vs 80~99 의 1896, 커버리지는
   약 1300 을 넘는 곳에서 시작)와 **같은 원인**이다. 두 문제가 하나였다.

🔴 그래서 기준을 바꾼다: 균등 간격도, 공유 가시성도 "그 자리를 **렌더할 수 있는가**"
   를 묻지 않는다. 관측 3D 점 수로 먼저 거르고, 통과한 자리들 안에서 고르게 뽑는다.

   ⚠️ 웨이포인트 자체는 GT1 을 쓰므로 **오라클(상한)** 이다. 이 선별은 그 상한을
      제대로 재기 위한 것이지, GT 없는 방법을 만드는 것이 아니다.

사용:
   python 코드/웨이포인트_품질선별.py --n 4                 # 곡선만 보고 고른다
   python 코드/웨이포인트_품질선별.py --n 4 --out ~/data/끝점/wp4q/목록.txt
"""
import argparse
import os
import re

import numpy as np

HOME = os.path.expanduser("~")


def ix(n):
    m = re.search(r"_(\d+)\.", n)
    return int(m.group(1)) if m else -1


def read_counts(p):
    """{이름: 그 프레임이 관측하는 3D 점 수}"""
    out = {}
    with open(p, encoding="utf-8") as f:
        L = [l for l in f if not l.startswith("#")]
    for i in range(0, len(L) - 1, 2):
        t = L[i].split()
        if len(t) < 10:
            continue
        v = L[i + 1].split()
        out[t[9]] = sum(1 for k in range(2, len(v), 3) if int(v[k]) >= 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=HOME + "/data/sfm_all/sparse/1")
    ap.add_argument("--lo", type=int, default=162, help="복도(GT1) 시작 프레임")
    ap.add_argument("--hi", type=int, default=286, help="복도 끝 프레임")
    ap.add_argument("--n", type=int, default=4, help="웨이포인트 수")
    ap.add_argument("--thr", type=int, default=1300,
                    help="이 수 미만이면 렌더가 뭉개진다 (저텍스처 검증에서 얻은 값)")
    ap.add_argument("--min-gap", type=int, default=12,
                    dest="min_gap", help="이보다 짧은 구간은 만들지 않는다")
    ap.add_argument("--out", default=None, help="목록.txt 로 쓸 경로")
    a = ap.parse_args()

    cnt = read_counts(os.path.join(a.model, "images.txt"))
    gt = sorted(((ix(n), n) for n in cnt
                 if n.startswith("v1791_") and a.lo <= ix(n) <= a.hi))
    if not gt:
        print("🔴 복도 프레임이 없다"); return
    c = np.array([cnt[n] for _, n in gt])

    # ---- 품질 곡선 ----
    print("== 복도 복원 품질 (관측 3D 점 수) ==")
    print("   프레임 %d~%d, %d장 | 중앙 %d, 최소 %d, 최대 %d"
          % (gt[0][0], gt[-1][0], len(gt), np.median(c), c.min(), c.max()))
    step = max(1, len(gt) // 20)
    print("   %-8s %-7s %s" % ("프레임", "점수", "막대 (임계 %d)" % a.thr))
    for k in range(0, len(gt), step):
        v = c[k]
        bar = "#" * int(30 * v / max(c.max(), 1))
        print("   %-8d %-7d %s%s" % (gt[k][0], v, bar, "" if v >= a.thr else "  🔴"))

    ok = c >= a.thr
    print("\n   임계 통과 %d/%d (%.0f%%)" % (ok.sum(), len(ok), 100 * ok.mean()))
    # 통과 구간을 이어진 덩어리로 묶는다
    runs, s = [], None
    for k, v in enumerate(ok):
        if v and s is None:
            s = k
        elif not v and s is not None:
            runs.append((s, k - 1)); s = None
    if s is not None:
        runs.append((s, len(ok) - 1))
    print("   쓸 수 있는 구간:", ", ".join("%d~%d" % (gt[x][0], gt[y][0])
                                        for x, y in runs if y - x >= 2))
    bad = [(gt[x][0], gt[y][0]) for x, y in
           [(r[1] + 1, runs[i + 1][0] - 1) for i, r in enumerate(runs[:-1])]
           if x <= y]
    head = [] if not runs or runs[0][0] == 0 else [(gt[0][0], gt[runs[0][0] - 1][0])]
    for lo, hi in head + bad:
        print("   🔴 못 쓰는 구간: %d~%d" % (lo, hi))

    # ---- 선별 ----
    # 🔴 '균등 목표에 가까운 자리' 로 고르면 안 된다. 못 쓰는 구간이 한쪽에 몰려
    #    있으면 목표들이 통과 구간 한 곳에 겹쳐 몰리고(실제로 W1·W2 가 7장 간격으로
    #    붙었다), 나머지 한 구간이 62장으로 벌어졌다. **최장 구간이 전부를 지배하므로**
    #    (= wp4 의 seg01 이 뭉갠 이유) 최장 구간을 직접 최소화한다.
    #    최대 간격 G 가 가능한지는 탐욕적으로 판정되고, G 에 대해 단조이므로 이분 탐색.
    cand = [k for k in range(len(gt)) if ok[k]]
    if len(cand) < a.n:
        print("🔴 통과한 자리가 웨이포인트 수보다 적다"); return
    F = [gt[k][0] for k in cand]          # 쓸 수 있는 실촬영 프레임 번호
    START, END = 143, a.hi

    #    탐욕적으로 '갈 수 있는 한 멀리' 뛰면 최장 구간은 최소가 되지만 **나머지가
    #    쏠린다**(마지막 구간이 4장까지 줄었다). 웨이포인트 수가 적으므로
    #    **DP 로 (최장 구간, 구간 제곱합) 을 사전식으로 최소화**한다 — 최장을 먼저
    #    줄이고, 같다면 고르게 펴는 쪽을 고른다.
    V = [START] + F + [END]
    nV = len(V)
    INF = (10**9, 10**9)
    # dp[j][i] = 웨이포인트 j개를 쓰고 i 번째 위치에 도달했을 때의 최소 비용
    dp = [[INF] * nV for _ in range(a.n + 1)]
    pre = [[-1] * nV for _ in range(a.n + 1)]
    dp[0][0] = (0, 0)
    for j in range(a.n + 1):
        for i in range(nV):
            if dp[j][i] == INF:
                continue
            for k2 in range(i + 1, nV):
                if k2 != nV - 1 and j + 1 > a.n:
                    break
                g = V[k2] - V[i]
                # 🔴 너무 짧은 구간은 웨이포인트 낭비다. 끝에 4장짜리 구간이
                #    생기는 것을 막는다 (제곱합 기준만으로는 그쪽이 유리해 보인다).
                if g < a.min_gap:
                    continue
                c2 = (max(dp[j][i][0], g), dp[j][i][1] + g * g)
                jj = j if k2 == nV - 1 else j + 1
                if jj > a.n:
                    continue
                if c2 < dp[jj][k2]:
                    dp[jj][k2] = c2
                    pre[jj][k2] = (j, i)
    bj = min((j for j in range(a.n + 1) if dp[j][nV - 1] != INF),
             key=lambda j: (dp[j][nV - 1], -j), default=None)
    if bj is None:
        print("🔴 웨이포인트 %d개로는 복도를 덮을 수 없다" % a.n); return
    # 같은 비용이면 웨이포인트를 많이 쓴 해를 고른다 (구간이 짧아진다)
    bestc = dp[bj][nV - 1]
    bj = max(j for j in range(a.n + 1) if dp[j][nV - 1] == bestc)
    G = bestc[0]
    path, j, i = [], bj, nV - 1
    while pre[j][i] != -1:
        path.append(V[i]); j, i = pre[j][i]
    chosen_f = sorted(x for x in path if x != END)
    print("\n   웨이포인트 %d개로 가능한 **최장 구간의 최소값 = %d장**" % (a.n, G))

    f2k = {gt[k][0]: k for k in range(len(gt))}
    chosen = [f2k[f] for f in chosen_f]

    print("\n== 고른 웨이포인트 ==")
    print("   %-5s %-9s %-7s %s" % ("", "프레임", "점수", "비고"))
    for i, k in enumerate(chosen, 1):
        print("   W%-4d %-9d %-7d %s" % (i, gt[k][0], c[k], gt[k][1]))

    # 구간 길이 — 한 구간만 길면 거기서 뭉갠다 (wp4 의 seg01 이 그랬다)
    pts = [START] + [gt[k][0] for k in chosen] + [END]
    gaps = [b - x for x, b in zip(pts, pts[1:])]
    print("\n   구간 길이(실촬영 프레임 기준): %s" % ", ".join(str(g) for g in gaps))
    print("   최장/최단 %.2f배 %s" % (max(gaps) / min(gaps),
                                   "✅" if max(gaps) / min(gaps) < 1.6 else
                                   "⚠️ 구간 길이가 고르지 않다"))
    # 🔴 구간마다 생성 프레임 수를 길이에 비례시켜야 **프레임당 이동량**이 일정해진다.
    #    wp4 는 길이와 무관하게 97장씩 줬고, 가장 긴 구간이 뭉갰다.
    base = 97.0 / float(np.median(gaps))
    nf = [max(25, int(round((g * base - 1) / 8)) * 8 + 1) for g in gaps]
    print("   권장 생성 프레임 수(8k+1, 길이 비례): %s" % ", ".join(str(x) for x in nf))

    if a.out:
        d = os.path.dirname(a.out)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            for k in chosen:
                f.write(gt[k][1] + "\n")
        with open(os.path.join(d, "프레임수.txt"), "w", encoding="utf-8") as f:
            f.write(" ".join(str(x) for x in nf) + "\n")
        print("\n   저장 → %s  (구간별 프레임 수: 프레임수.txt)" % a.out)


if __name__ == "__main__":
    main()
