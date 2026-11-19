# -*- coding: utf-8 -*-
"""영상 간 공유 3D 점(co-visibility) 분석.

"두 케이스가 같은 A / 같은 B 를 공유한다" 는 설계 전제를, 카메라 거리가 아니라
'같은 3D 점을 보는 프레임이 있는가' 로 검증한다. 이 값이 §4 의 정답 누출 방지
기준(가시 점 30%)과 같은 단위다.
"""
import collections, numpy as np

D = "/home/hauwoong/data/sfm_all/sparse/1/"
FPS, BUCKET = 2.5, 10.0          # 10초 단위로 묶어서 본다

# images.txt: 1행=포즈, 2행=POINT2D (x y point3D_id ...)
pts = {}            # image name -> set(point3D id)
name = None
for ln in open(D + "images.txt"):
    if ln.startswith("#"):
        continue
    t = ln.split()
    if len(t) == 10 and t[9].endswith(".png"):
        name = t[9]
        continue
    if name is None:
        continue
    ids = t[2::3]
    pts[name] = {int(v) for v in ids if v != "-1"}
    name = None

print("프레임 %d, 평균 가시 점 %.0f개" % (len(pts), np.mean([len(v) for v in pts.values()])))

def key(n):
    return n[:5], int((int(n.split("_")[1].split(".")[0]) - 1) / FPS // BUCKET)

buck = collections.defaultdict(set)
for n, s in pts.items():
    buck[key(n)] |= s

vids = sorted({k[0] for k in buck})

def cross(va, vb):
    ba = sorted(b for v, b in buck if v == va)
    bb = sorted(b for v, b in buck if v == vb)
    M = np.zeros((len(ba), len(bb)))
    for i, x in enumerate(ba):
        A = buck[(va, x)]
        for j, y in enumerate(bb):
            B = buck[(vb, y)]
            M[i, j] = len(A & B) / max(1, min(len(A), len(B)))
    print("\n=== %s(행) x %s(열) 공유 점 비율 (작은 쪽 기준 %%) ===" % (va, vb))
    print("        " + "".join("%5d" % (y * BUCKET) for y in bb))
    for i, x in enumerate(ba):
        row = "".join(("%5.0f" % (M[i, j] * 100)) if M[i, j] >= 0.005 else "    ." for j in range(len(bb)))
        print("%4ds  %s" % (x * BUCKET, row))
    # 최고 대응
    i, j = np.unravel_index(M.argmax(), M.shape)
    print("  최대: %s %ds ~ %s %ds = %.0f%%" % (va, ba[i]*BUCKET, vb, bb[j]*BUCKET, M[i, j]*100))
    return M, ba, bb

cross("v1791", "v1792")
cross("v1791", "v1793")
cross("v1792", "v1793")
