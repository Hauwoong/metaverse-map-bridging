#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""웨이포인트 후보의 **렌더 이미지 자체**가 쓸 만한지 잰다.

🔴 왜 대리 지표를 버리는가 (2026-11-25, 두 번째 교훈):
   처음에는 '그 프레임이 SfM 에서 관측하는 3D 점 수' 로 걸렀다(임계 1300).
   W1(165, 1370점)은 그 기준으로 통과했고 실제로도 선명했다.
   그런데 **W2(205, 1326점)도 통과했는데 렌더는 완전히 뭉갰다.**
   SfM 점 수는 "그 자리에서 특징이 잡혔는가" 를 말할 뿐,
   **"그 자리를 3DGS 가 그릴 수 있는가" 를 말하지 않는다.** 둘은 다르다.
   (3DGS 렌더 품질은 그 지점을 **몇 개의 학습 시점이 둘러쌌는가** 에 달려 있다)

   📌 같은 함정을 프로젝트에서 반복하고 있다 —
      '비어있음 비율' 이 0% 로 나온 것도, 잔차만 보고 퇴화를 놓친 것도 같은 구조다.
      **재려는 것을 직접 재지 않고 대리 지표를 쓰면 통과해서는 안 될 것이 통과한다.**

🔴 그래서 렌더 이미지를 직접 본다:
   ① SIFT 특징점 수 — 등록 가능성과 직결된다(실패한 생성 프레임은 65개였다)
   ② 라플라시안 분산 — 뭉갬(저주파만 남은 상태)을 직접 잡는다
   둘 다 **실촬영 원본과의 비율**로 환산해 보고한다. 절대값은 해상도에 따라 변한다.

사용:
   python 코드/앵커_렌더품질.py --renders ~/data/끝점/G_GT1 --out ~/data/끝점/렌더품질.txt
"""
import argparse
import os
import re

import cv2
import numpy as np

HOME = os.path.expanduser("~")


def ix(n):
    m = re.search(r"_(\d+)\.", n)
    return int(m.group(1)) if m else -1


def score(path, sift):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    # 🔴 해상도를 맞춘다 — 렌더와 실촬영의 크기가 다르면 비교가 성립하지 않는다
    h = 540
    if img.shape[0] != h:
        img = cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h),
                         interpolation=cv2.INTER_AREA)
    kp = sift.detect(img, None)
    lap = float(cv2.Laplacian(img, cv2.CV_64F).var())
    return len(kp), lap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", default=HOME + "/data/끝점/G_GT1")
    ap.add_argument("--real", default=HOME + "/data/frames_all",
                    help="같은 프레임의 실촬영 — 기준값을 얻는다")
    ap.add_argument("--thr", type=float, default=0.5,
                    help="실촬영 대비 SIFT 특징 수 비율 하한")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    sift = cv2.SIFT_create()
    files = []
    for root, _, fs in os.walk(a.renders):
        for f in fs:
            if f.lower().endswith((".png", ".jpg")) and f.startswith("v1791_"):
                files.append(os.path.join(root, f))
    files.sort(key=lambda p: ix(os.path.basename(p)))
    if not files:
        print("🔴 렌더가 없다: %s" % a.renders); return

    # ---- 실촬영 기준값 ----
    base_kp, base_lap = [], []
    for p in files[::7]:
        q = os.path.join(a.real, os.path.basename(p).rsplit(".", 1)[0] + ".png")
        if os.path.exists(q):
            r = score(q, sift)
            if r:
                base_kp.append(r[0]); base_lap.append(r[1])
    bk = float(np.median(base_kp)) if base_kp else 1.0
    bl = float(np.median(base_lap)) if base_lap else 1.0
    print("== 실촬영 기준 (같은 프레임 %d장) ==" % len(base_kp))
    print("   SIFT 특징 중앙 %.0f | 라플라시안 분산 중앙 %.1f" % (bk, bl))

    rows = []
    for p in files:
        r = score(p, sift)
        if r:
            rows.append((ix(os.path.basename(p)), os.path.basename(p), r[0], r[1]))

    print("\n== 렌더 품질 (실촬영 대비 비율) ==")
    print("   %-8s %9s %9s   %s" % ("프레임", "특징비", "선명비", "막대"))
    step = max(1, len(rows) // 25)
    for k in range(0, len(rows), step):
        i, _, kpn, lap = rows[k]
        rk, rl = kpn / bk, lap / bl
        bar = "#" * int(28 * min(rk, 1.2))
        print("   %-8d %8.2f %9.2f   %s" % (i, rk, rl, bar))

    rk = np.array([r[2] / bk for r in rows])
    rl = np.array([r[3] / bl for r in rows])
    print("\n   특징비 중앙 %.2f, 최소 %.2f | 선명비 중앙 %.2f, 최소 %.2f"
          % (np.median(rk), rk.min(), np.median(rl), rl.min()))

    # 🔴 **선명비로는 거르지 않는다.** 3DGS 렌더는 멀쩡한 자리에서도 실촬영보다
    #    부드럽다 — 좋은 구간에서조차 선명비가 0.26~0.46 에 머문다(중앙 0.28).
    #    0.5 를 걸면 115장 중 2장만 통과한다. 선명비는 **보기용**으로만 둔다.
    #    특징비는 0.00 ↔ 0.65 로 깨끗하게 갈린다 — 이쪽이 실제로 쓸 수 있는 자다.
    #    (등록 실패한 생성 프레임 0.02배, 성공한 것 0.29배 — 앵커는 더 좋아야 한다)
    THR = a.thr
    ok = rk >= THR
    print("   임계(특징비 ≥%.2f) 통과 %d/%d (%.0f%%)"
          % (THR, ok.sum(), len(ok), 100 * ok.mean()))
    runs, s = [], None
    for k, v in enumerate(ok):
        if v and s is None:
            s = k
        elif not v and s is not None:
            runs.append((rows[s][0], rows[k - 1][0])); s = None
    if s is not None:
        runs.append((rows[s][0], rows[-1][0]))
    print("   쓸 수 있는 구간: %s"
          % ", ".join("%d~%d" % r for r in runs if r[1] - r[0] >= 1))

    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            for i, n, kpn, lap in rows:
                # 🔴 선명비는 쓰지 않는다 (위 주석 참조) — 특징비만 내보낸다
                f.write("%s %.4f\n" % (n, kpn / bk))
        print("\n   저장 → %s  (이름 + 실촬영 대비 비율)" % a.out)
        print("   → 코드/웨이포인트_품질선별.py --quality %s --thr %.2f" % (a.out, THR))


if __name__ == "__main__":
    main()
