# -*- coding: utf-8 -*-
"""A / 중간 / B 경계 설계 — 각 프레임이 '상대 영상이 못 본 점'을 얼마나 보는가.

exclusive(f) = (f 가 보는 3D 점 중 상대 영상의 어느 프레임도 못 본 점의 비율)

  exclusive ~ 0  : 상대와 같은 공간 → 공유 구역 (A 또는 B) 후보
  exclusive ~ 1  : 상대가 전혀 못 간 공간 → 숨길 중간 구간(GT) 후보

이 값이 §4 '정답 누출 방지' 의 30% 기준과 같은 단위다.
"""
import collections, numpy as np

D = "/home/hauwoong/data/sfm_all/sparse/1/images.txt"
FPS = 2.5

pts, name = {}, None
for ln in open(D):
    if ln.startswith("#"):
        continue
    t = ln.split()
    if len(t) == 10 and t[9].endswith(".png"):
        name = t[9]; continue
    if name is None:
        continue
    pts[name] = {int(v) for v in t[2::3] if v != "-1"}
    name = None

vid = collections.defaultdict(dict)
for n, s in pts.items():
    vid[n[:5]][(int(n.split("_")[1].split(".")[0]) - 1) / FPS] = s

allpts = {v: set().union(*vid[v].values()) for v in vid}

def excl(va, vb):
    ts = sorted(vid[va])
    other = allpts[vb]
    e = np.array([1 - len(vid[va][t] & other) / max(1, len(vid[va][t])) for t in ts])
    return np.array(ts), e

def runs(ts, e, lo, hi, minlen=4.0):
    m = (e >= lo) & (e < hi)
    out, i, n = [], 0, len(m)
    while i < n:
        if m[i]:
            j = i
            while j + 1 < n and m[j + 1]:
                j += 1
            if ts[j] - ts[i] >= minlen:
                out.append((ts[i], ts[j], e[i:j+1].mean()))
            i = j + 1
        else:
            i += 1
    return out

for va, vb in (("v1791", "v1792"), ("v1792", "v1791")):
    ts, e = excl(va, vb)
    print("\n" + "=" * 70)
    print("%s 의 각 시각: %s 가 못 본 점의 비율" % (va, vb))
    print("=" * 70)
    for k in range(0, len(ts), 10):
        bar = "#" * int(e[k] * 40)
        print("  %6.1fs  %5.0f%%  %s" % (ts[k], e[k] * 100, bar))
    print("\n  [전용 구간 후보  exclusive >= 70%%]")
    for a, b, m in runs(ts, e, 0.70, 1.01):
        print("     %6.1fs ~ %6.1fs  (%4.1fs)  평균 %.0f%%" % (a, b, b - a, m * 100))
    print("  [공유 구간 후보  exclusive < 30%%]")
    for a, b, m in runs(ts, e, -0.01, 0.30):
        print("     %6.1fs ~ %6.1fs  (%4.1fs)  평균 %.0f%%" % (a, b, b - a, m * 100))

# 1793 은 B 보강이므로, 1791/1792 양쪽과 겹치는 구간만 B 로 본다
print("\n" + "=" * 70)
print("v1793 (B 보강) 의 각 시각: 1791·1792 와의 공유 비율")
print("=" * 70)
ts = sorted(vid["v1793"])
for k in range(0, len(ts), 10):
    s = vid["v1793"][ts[k]]
    a = len(s & allpts["v1791"]) / max(1, len(s))
    b = len(s & allpts["v1792"]) / max(1, len(s))
    print("  %6.1fs   1791 %4.0f%%   1792 %4.0f%%   %s" % (ts[k], a*100, b*100,
          "#" * int(max(a, b) * 30)))
