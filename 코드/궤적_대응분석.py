# -*- coding: utf-8 -*-
"""1791 / 1792 / 1793 의 대응 구조를 시간축으로 훑는다 (스케일 가정 없이 상대값)."""
import numpy as np, collections

P = "/home/hauwoong/data/sfm_all/sparse/1/images.txt"
FPS = 2.5

def q2R(w, x, y, z):
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
                     [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

cen = collections.defaultdict(dict)
for ln in open(P):
    if ln.startswith("#"):
        continue
    t = ln.split()
    if len(t) != 10 or not t[9].endswith(".png"):
        continue
    q = list(map(float, t[1:5]))
    tv = np.array(list(map(float, t[5:8])))
    R = q2R(*q)
    cen[t[9][:5]][int(t[9].split("_")[1].split(".")[0])] = -R.T @ tv

tr, tt = {}, {}
for v in sorted(cen):
    ks = sorted(cen[v])
    tr[v] = np.array([cen[v][k] for k in ks])
    tt[v] = np.array([(k-1)/FPS for k in ks])

# 스케일 기준: 1791 의 경로 길이를 1.0 으로 두고 전부 상대 비교
U = np.linalg.norm(np.diff(tr["v1791"], axis=0), axis=1).sum()
print("기준: 1791 경로 길이 = %.2f 단위 (121.3s 동안 걸은 거리)" % U)
print("     아래 거리는 '1791 전체 경로의 몇 %%' 로 읽으면 된다\n")

def table(va, vb, step=10.0):
    A, B = tr[va], tr[vb]
    D = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
    print("--- %s 의 각 시각에서 %s 까지의 최단거리 ---" % (va, vb))
    print("  %7s | %9s | %8s" % (va+" 시각", vb+" 최근접", "거리(경로%)"))
    nxt = 0.0
    for i, ti in enumerate(tt[va]):
        if ti + 1e-9 < nxt:
            continue
        nxt = ti + step
        j = D[i].argmin()
        flag = "  <== 겹침" if D[i, j] / U < 0.02 else ""
        print("  %6.1fs | %8.1fs | %7.2f%%%s" % (ti, tt[vb][j], D[i, j]/U*100, flag))
    print()

table("v1791", "v1792")
table("v1792", "v1791")
table("v1793", "v1791", 5.0)
table("v1793", "v1792", 5.0)

print("--- 끝점 사이 거리 (경로% 기준) ---")
pairs = [("1791 시작", tr["v1791"][0], "1792 시작", tr["v1792"][0]),
         ("1791 끝",   tr["v1791"][-1], "1792 끝",   tr["v1792"][-1]),
         ("1791 시작", tr["v1791"][0], "1791 끝",   tr["v1791"][-1]),
         ("1792 시작", tr["v1792"][0], "1792 끝",   tr["v1792"][-1])]
for la, a, lb, b in pairs:
    print("  %-10s <-> %-10s : %6.2f%%" % (la, lb, np.linalg.norm(a-b)/U*100))
