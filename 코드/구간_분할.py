# -*- coding: utf-8 -*-
"""합본 COLMAP 모델을 G_A / G_GT1 / G_B 세 장면으로 나눈다 (케이스 1 단독, 선택지 ⓐ).

원칙 — **어느 쪽인가는 경로가 정하고, 무엇을 버릴지는 누출 규칙이 정한다.**

  1791  t <= 64s          -> A     (중간 구간에 들어가기 전에 본 것)
        64s < t <= 114s   -> GT1   (단독 주행 = 숨길 정답)
        t > 114s          -> B     (중간 구간을 빠져나온 뒤 본 것)
  1793  전부              -> B     (B 보강 촬영)
  1792  전부 제외         -> 아래 참조

  M      = GT1 프레임이 보는 3D 점 집합
  leak(f)= |pts(f) ∩ M| / |pts(f)|
  leak(f) > 0.30 인 프레임은 A·B 에서 제외   (§4 정답 누출 방지 규칙)

🔴 **1792 를 끝점 장면에서 뺀 이유.** 1792 는 A 에서 B 까지 끊김 없이 걸어간다.
이것을 G_A·G_B 에 넣으면 두 끝점 장면의 합집합이 이미 A→B 실촬영 연결로를
포함하게 되어, "끊긴 두 장면을 생성 영상으로 잇는다" 는 전제가 무너진다.
1792 는 공통 좌표계를 세우는 데 이미 기여했으므로(track length 10.43 -> 12.19)
거기까지만 쓰고, 장면 학습에서는 제외한다.

흐린 프레임은 SfM 과 달리 3DGS 에는 해로우므로 장면별 하위 분위수로 자른다.

사용:
  python 코드/구간_분할.py            # 분석만 (아무것도 쓰지 않음)
  python 코드/구간_분할.py --write    # 장면별 nerfstudio 데이터셋 생성
"""
import os, sys, shutil, collections
import numpy as np

SRC    = os.path.expanduser("~/data/sfm_all")
MODEL  = os.path.join(SRC, "sparse", "1")
FRAMES = os.path.expanduser("~/data/frames_all")
OUT    = os.path.expanduser("~/data/scenes")
FPS    = 2.5

MID_LO, MID_HI = 64.0, 114.0     # 1791 단독 주행 구간 (전용 >=70% 측정 결과)
LEAK_MAX       = 0.30            # 정답 누출 한계
SHARP_Q        = 8               # 장면별 선명도 하위 q% 제외

# ──────────────────────────── 모델 읽기 ────────────────────────────
def read_model(d):
    cams = open(os.path.join(d, "cameras.txt")).read()
    imgs, pts, name, pose = {}, {}, None, None
    for ln in open(os.path.join(d, "images.txt")):
        if ln.startswith("#"):
            continue
        t = ln.split()
        if len(t) == 10 and t[9].endswith(".png"):
            name, pose = t[9], ln.rstrip("\n")
            continue
        if name is None:
            continue
        imgs[name] = (pose, ln.rstrip("\n"), {int(v) for v in t[2::3] if v != "-1"})
        name = None
    for ln in open(os.path.join(d, "points3D.txt")):
        if ln.startswith("#"):
            continue
        t = ln.split()
        pts[int(t[0])] = ln.rstrip("\n")
    return cams, imgs, pts

def tsec(n):
    return (int(n.split("_")[1].split(".")[0]) - 1) / FPS

# ──────────────────────────── 분할 ────────────────────────────
def side(n):
    t = tsec(n)
    if n.startswith("v1792"):
        return "-1792"
    if n.startswith("v1793"):
        return "B"
    return "A" if t <= MID_LO else ("GT1" if t <= MID_HI else "B")

def split(imgs):
    mid = [n for n in imgs if side(n) == "GT1"]
    M = set().union(*[imgs[n][2] for n in mid])

    scene, rec = {}, {}
    for n, (_, _, P) in imgs.items():
        leak = len(P & M) / max(1, len(P))
        rec[n] = dict(leak=leak)
        s = side(n)
        scene[n] = s if (s in ("GT1", "-1792") or leak <= LEAK_MAX) else "-누출"
    return scene, rec, M

# ──────────────────────────── 선명도 ────────────────────────────
def sharpness(names):
    from PIL import Image
    out = {}
    for n in names:
        im = Image.open(os.path.join(FRAMES, n)).convert("L").resize((480, 270))
        a = np.asarray(im, dtype=np.float32)
        lap = (-4*a[1:-1, 1:-1] + a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:])
        out[n] = float(lap.var())
    return out

# ──────────────────────────── 쓰기 ────────────────────────────
def write_scene(tag, names, cams, imgs, pts):
    root = os.path.join(OUT, tag)
    img_d = os.path.join(root, "images")
    col_d = os.path.join(root, "colmap", "sparse", "0")
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(img_d); os.makedirs(col_d)

    keep = set(names)
    cnt = collections.Counter()
    for n in keep:
        for p in imgs[n][2]:
            cnt[p] += 1
    good = {p for p, c in cnt.items() if c >= 2 and p in pts}

    with open(os.path.join(col_d, "cameras.txt"), "w") as f:
        f.write(cams)
    with open(os.path.join(col_d, "images.txt"), "w") as f:
        f.write("# Image list\n")
        for i, n in enumerate(sorted(keep), 1):
            pose, p2d, _ = imgs[n]
            t = pose.split()
            f.write(" ".join([str(i)] + t[1:8] + [t[8], n]) + "\n")
            q = p2d.split()
            f.write(" ".join(" ".join((q[k], q[k+1],
                    q[k+2] if (q[k+2] != "-1" and int(q[k+2]) in good) else "-1"))
                    for k in range(0, len(q), 3)) + "\n")
    # 원본 image_id -> 새 image_id (빠진 프레임은 트랙에서 제거한다)
    old_id = {}
    for n in imgs:
        old_id[int(imgs[n][0].split()[0])] = n
    new_id = {n: i for i, n in enumerate(sorted(keep), 1)}
    remap = {oid: new_id[n] for oid, n in old_id.items() if n in keep}

    with open(os.path.join(col_d, "points3D.txt"), "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        for p in sorted(good):
            t = pts[p].split()
            head, track = t[:8], t[8:]
            tr = []
            for k in range(0, len(track) - 1, 2):
                oid = int(track[k])
                if oid in remap:
                    tr += [str(remap[oid]), track[k + 1]]
            if len(tr) >= 4:                      # 2장 이상에서 보이는 점만
                f.write(" ".join(head + tr) + "\n")
    for n in keep:
        os.link(os.path.join(FRAMES, n), os.path.join(img_d, n))
    return len(keep), len(good)

# ──────────────────────────── main ────────────────────────────
if __name__ == "__main__":
    cams, imgs, pts = read_model(MODEL)
    print("모델: 프레임 %d, 3D 점 %d" % (len(imgs), len(pts)))
    scene, rec, M = split(imgs)
    print("중간 구간(숨길 정답) 점 |M| = %d" % len(M))

    KINDS = ("A", "GT1", "B", "-누출", "-1792")
    g = collections.Counter()
    per = collections.defaultdict(collections.Counter)
    for n, s in scene.items():
        g[s] += 1
        per[n[:5]][s] += 1
    print("\n=== 배정 결과 ===")
    for s in KINDS:
        print("  %-6s %4d" % (s, g[s]))
    print("\n  영상별:")
    for v in sorted(per):
        print("    %s  " % v + "  ".join("%s %d" % (s, per[v][s])
              for s in KINDS if per[v][s]))

    print("\n=== 누출로 제외된 프레임 (경계에서 중간 구간이 보이는 것) ===")
    for v in sorted(per):
        bad = sorted([n for n in scene if n.startswith(v) and scene[n] == "-누출"],
                     key=tsec)
        if bad:
            print("    %s  %s" % (v, ", ".join("%.1fs(%.0f%%)" % (tsec(n), rec[n]["leak"]*100)
                  for n in bad[:6]) + (" ... 총 %d장" % len(bad) if len(bad) > 6 else "")))

    print("\n=== 시간축 배정 (10초 간격) ===")
    for v in sorted(per):
        row = []
        for n in sorted([x for x in scene if x.startswith(v)], key=tsec):
            if abs(tsec(n) % 10.0) < 1/FPS/2:
                row.append("%3.0fs:%s" % (tsec(n), scene[n]))
        print("  %s  %s" % (v, "  ".join(row)))

    if "--write" in sys.argv:
        sh = sharpness([n for n in scene if scene[n] in ("A", "B", "GT1")])
        os.makedirs(OUT, exist_ok=True)
        print("\n=== 데이터셋 생성 ===")
        for tag in ("A", "GT1", "B"):
            ns = [n for n in scene if scene[n] == tag]
            thr = np.percentile([sh[n] for n in ns], SHARP_Q)
            keep = [n for n in ns if sh[n] >= thr]
            ni, npt = write_scene(tag, keep, cams, imgs, pts)
            print("  G_%-4s 프레임 %3d (흐림 %2d 제외, 선명도 하한 %.0f)  점 %d"
                  % (tag, ni, len(ns)-len(keep), thr, npt))
        print("\n출력: %s" % OUT)
