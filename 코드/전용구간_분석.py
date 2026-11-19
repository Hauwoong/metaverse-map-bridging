# -*- coding: utf-8 -*-
"""각 영상이 '다른 두 영상의 합집합' 대비 얼마나 전용인가 + 구역별 커버리지."""
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
allp = {v: set().union(*vid[v].values()) for v in vid}

for va in sorted(vid):
    others = set().union(*[allp[v] for v in vid if v != va])
    ts = sorted(vid[va])
    e = np.array([1 - len(vid[va][t] & others) / max(1, len(vid[va][t])) for t in ts])
    print("\n=== %s : 나머지 두 영상이 못 본 점의 비율 ===" % va)
    for k in range(0, len(ts), 5):
        print("  %6.1fs  %4.0f%%  %s" % (ts[k], e[k]*100, "#" * int(e[k]*36)))
    m = e >= 0.70
    out, i, n = [], 0, len(m)
    while i < n:
        if m[i]:
            j = i
            while j+1 < n and m[j+1]:
                j += 1
            if ts[j]-ts[i] >= 4:
                out.append((ts[i], ts[j], e[i:j+1].mean()))
            i = j+1
        else:
            i += 1
    print("  [전용 >=70%% 구간]")
    for a, b, mm in out:
        print("     %6.1fs ~ %6.1fs  (%4.1fs, %d프레임)  평균 %.0f%%"
              % (a, b, b-a, int((b-a)*FPS)+1, mm*100))
