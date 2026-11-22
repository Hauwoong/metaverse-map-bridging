#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""웨이포인트를 **공유 가시성**으로 고른다.

왜 균등 간격이 아닌가:
  웨이포인트 4장을 균등 간격(GT1 23/46/69/92)으로 줬더니 **A→W1 과 W4→B 만 실패**했다.
  그 둘은 넓은 로비에서 좁은 복도로 넘어가는 **전환부**다. 중간의 복도끼리는 서로
  비슷해서 잘 이어졌다. 즉 웨이포인트가 필요한 자리는 **균등한 지점이 아니라
  변화가 큰 지점**이다.

기준:
  covis(i, j) = |pts(i) ∩ pts(j)| / |pts(i)|
  직전 앵커에서 보이던 것이 임계값 아래로 떨어지는 순간이 **새 앵커가 필요한 순간**이다.
  모델이 이전 앵커에서 더는 볼 수 없게 되는 지점이기 때문이다.

  (이 지표는 프로젝트 초반 장면 분할에서 쓴 것과 같다. 그때도 '카메라 중심 거리' 가
   아니라 공유 가시성이 '같은 장소인가' 의 옳은 자였다.)
"""
import argparse, os
import numpy as np

HOME = os.path.expanduser("~")


def read_seen(p):
    """{이미지이름: 그 프레임이 보는 3D 점 번호 집합}"""
    out = {}
    L = [l for l in open(p) if not l.startswith("#")]
    for i in range(0, len(L), 2):
        t = L[i].split()
        if len(t) < 10:
            continue
        s = set()
        if i+1 < len(L):
            v = L[i+1].split()
            for k in range(2, len(v), 3):
                pid = int(v[k])
                if pid != -1:
                    s.add(pid)
        out[t[9]] = s
    return out


def covis(a, b):
    if not a:
        return 0.0
    return len(a & b) / len(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=float, default=0.30,
                    help="직전 앵커와의 공유 가시성이 이 아래로 떨어지면 새 앵커")
    ap.add_argument("--merged", default=HOME+"/data/sfm_all/sparse/1/images.txt")
    a = ap.parse_args()

    S = read_seen(a.merged)
    A, B = "v1791_00143.png", "v1793_00001.png"
    gt = sorted(k for k in S if k.startswith("v1791_") and 162 <= int(k[6:11]) <= 286)
    print("GT1 %d장, A=%s, B=%s" % (len(gt), A, B))
    print("임계값 %.2f" % a.thr)

    # 1) 곡선 — A 에서 출발해 GT1 을 따라가며 공유 가시성이 어떻게 떨어지는가
    print()
    print("A 끝점에서 본 공유 가시성 (GT1 진행에 따라)")
    cur = S[A]
    for i in range(0, len(gt), 10):
        print("  GT1 %3d (%s) : %.3f" % (i, gt[i], covis(cur, S[gt[i]])))

    # 2) 탐욕적 선별 — 직전 앵커에서 임계값 아래로 떨어지면 그 직전 프레임을 앵커로
    print()
    print("공유 가시성 기반 웨이포인트 선별")
    anchors, last_i = [], None
    cur = S[A]
    for i, k in enumerate(gt):
        c = covis(cur, S[k])
        if c < a.thr:
            pick = gt[max(i-1, 0)]
            anchors.append((max(i-1, 0), pick))
            cur = S[pick]
            print("  W%d : GT1 %3d (%s)  ← 직전 앵커 가시성 %.3f 로 하락"
                  % (len(anchors), max(i-1, 0), pick, c))
    cB = covis(cur, S[B])
    print("  마지막 앵커 → B 공유 가시성 %.3f %s"
          % (cB, "✅" if cB >= a.thr else "⚠️ 여기도 앵커가 필요할 수 있다"))
    print()
    print("선별 결과 %d장: %s" % (len(anchors), [i for i, _ in anchors]))
    print("(참고) 균등 4장: [23, 46, 69, 92]")

    # 3) 어디서 변화가 큰가 — 인접 프레임 간 가시성 낙폭
    print()
    print("인접 프레임 간 공유 가시성이 가장 크게 떨어지는 구간 (전환부 후보)")
    drops = []
    for i in range(len(gt)-1):
        drops.append((covis(S[gt[i]], S[gt[i+1]]), i))
    drops.sort()
    for c, i in drops[:8]:
        print("  GT1 %3d→%3d : %.3f" % (i, i+1, c))


if __name__ == "__main__":
    main()
