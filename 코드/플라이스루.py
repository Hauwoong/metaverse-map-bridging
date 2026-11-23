#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""제출용 플라이스루 경로를 만든다 — A 끝 → 복도 전체 → B 처음.

이 프로젝트의 핵심 시연이다:
  **중간 구간을 한 번도 본 적 없는 맵에서 복도 전체를 끊김 없이 통과한다.**
  G_M 의 학습에는 GT1 이 들어가지 않았다. 그런데 GT1 의 포즈를 따라
  날아가며 영상을 뽑는다. 이음새가 채워졌다면 끊김 없이 지나간다.

대조군 G_AB(다리 없음)로 같은 경로를 뽑으면 **같은 구간이 비어 있는** 영상이
나온다. 두 영상을 나란히 두는 것이 §4.7 정성 결과다.

🔴 GT1 포즈를 쓰는 것은 '정답을 쓰는 것' 이 아니다. 포즈(어디서 보는가)와
   내용(무엇이 보이는가)은 다르다. 맵은 GT1 의 **그림**을 본 적이 없다.
   이것은 평가자가 카메라를 들고 복도를 걷는 것과 같다.

🔴 A 와 B 의 꼬리/머리를 붙이는 이유: 이음새가 어디인지 보여주려면 양쪽
   실촬영 구간에서 출발해 들어가고 나와야 한다. 다리만 보면 판단이 안 된다.
"""
import argparse, json, os
from pathlib import Path
import numpy as np

HOME = Path(os.path.expanduser("~"))


def qvec2R(q):
    w, x, y, z = q
    return np.array([[1-2*y*y-2*z*z, 2*x*y-2*z*w,   2*x*z+2*y*w],
                     [2*x*y+2*z*w,   1-2*x*x-2*z*z, 2*y*z-2*x*w],
                     [2*x*z-2*y*w,   2*y*z+2*x*w,   1-2*x*x-2*y*y]])


def read_c2w(p):
    out = {}
    lines = [l for l in open(p) if not l.startswith("#")]
    for i in range(0, len(lines), 2):
        t = lines[i].split()
        if len(t) < 10:
            continue
        R = qvec2R(list(map(float, t[1:5])))
        tv = np.array(list(map(float, t[5:8])))
        c2w = np.eye(4)
        c2w[:3, :3] = R.T
        c2w[:3, 3] = -R.T @ tv
        out[t[9]] = c2w
    return out


def smooth(path, win):
    """포즈 열을 부드럽게 한다. 위치만 이동평균하고 회전은 건드리지 않는다
    (회전을 같은 방식으로 섞으면 직교성이 깨진다)."""
    if win <= 1:
        return path
    P = np.array([T[:3, 3] for T in path])
    n = len(P)
    out = []
    for i in range(n):
        lo, hi = max(0, i-win//2), min(n, i+win//2+1)
        T = path[i].copy()
        T[:3, 3] = P[lo:hi].mean(0)
        out.append(T)
    return out


def slerp_R(Ra, Rb, u):
    """두 회전 사이를 보간한다. 축-각으로 바꿔 선형 보간하는 것과 같다.
    🔴 회전 행렬을 성분별로 섞으면 직교성이 깨진다. 반드시 이렇게 한다."""
    Rrel = Ra.T @ Rb
    w = float(np.clip((np.trace(Rrel) - 1) / 2, -1, 1))
    ang = float(np.arccos(w))
    if ang < 1e-8:
        return Ra.copy()
    ax = np.array([Rrel[2, 1]-Rrel[1, 2],
                   Rrel[0, 2]-Rrel[2, 0],
                   Rrel[1, 0]-Rrel[0, 1]]) / (2*np.sin(ang))
    th = ang * u
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return Ra @ (np.eye(3) + np.sin(th)*K + (1-np.cos(th))*(K @ K))


def densify(path, k):
    """인접 자세 사이에 k-1 개를 끼워 넣어 재생 속도를 1/k 로 늦춘다.

    🔴 왜 필요한가 — 이것 때문에 영상이 '깨져' 보였다.
       촬영 영상에서 자세를 약 2.5fps 로 뽑았는데 24fps 로 재생하면
       **약 10배속**이 된다. 프레임 하나하나는 선명한데 장면이 순간이동하듯
       바뀌어서 깨진 것처럼 보인다. 실측: A 구간 40장이 실제로는 16초쯤
       걸어간 거리인데 영상에서는 1.7초에 지나간다.
       보간해 채우면 실제 걷는 속도가 된다.
    """
    if k <= 1:
        return path
    out = []
    for i in range(len(path)-1):
        Ra, Rb = path[i][:3, :3], path[i+1][:3, :3]
        pa, pb = path[i][:3, 3], path[i+1][:3, 3]
        for j in range(k):
            u = j / k
            T = np.eye(4)
            T[:3, :3] = slerp_R(Ra, Rb, u)
            T[:3, 3] = (1-u)*pa + u*pb
            out.append(T)
    out.append(path[-1].copy())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sfm", default=str(HOME/"data/sfm_all/sparse/1"))
    ap.add_argument("--models", default="M,AB")
    ap.add_argument("--runs", default=str(HOME/"data/runs"))
    ap.add_argument("--a-tail", type=int, default=40, help="A 꼬리 몇 장")
    ap.add_argument("--b-head", type=int, default=40, help="B 머리 몇 장")
    ap.add_argument("--interp", type=int, default=1,
                    help="인접 자세 사이에 끼워 넣을 배수. 촬영 자세가 약 "
                         "2.5fps 라 24fps 재생은 10배속이 된다. 4~8 을 권한다")
    ap.add_argument("--gt-from", type=int, default=None,
                    help="복도 자세를 이 번째부터만 쓴다. 다리가 덮은 구간만 "
                         "보여줄 때 쓴다 (덮이지 않은 구간은 floater 로 깨진다)")
    ap.add_argument("--gt-to", type=int, default=None, help="복도 자세 끝 번째")
    ap.add_argument("--smooth", type=int, default=5)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--out", default=str(HOME/"data/플라이스루"))
    a = ap.parse_args()

    sfm = Path(a.sfm)
    imgs = read_c2w(sfm/"images.txt")

    def ix(k):
        return int(k.split("_")[1].split(".")[0])

    A  = sorted((k for k in imgs if k.startswith("v1791_") and ix(k) <= 143),
                key=ix)[-a.a_tail:]
    GT = sorted((k for k in imgs if k.startswith("v1791_")
                 and 162 <= ix(k) <= 286), key=ix)
    B  = sorted((k for k in imgs if k.startswith("v1793_")), key=ix)[:a.b_head]

    # 🔴 복도 구간을 잘라 쓸 때는 A 꼬리·B 머리를 붙이지 않는다.
    #    A 는 복도 0번 근처, B 는 125번 너머에 있으므로, 복도 43번부터 시작하면
    #    A→복도가 공간적으로 순간이동이 되어 더 어색해진다.
    cut = a.gt_from is not None or a.gt_to is not None
    if cut:
        lo = a.gt_from if a.gt_from is not None else 0
        hi = a.gt_to if a.gt_to is not None else len(GT)-1
        GT = GT[lo:hi+1]
        seq = GT
        print("복도 %d~%d번만 %d 포즈 (A·B 꼬리 생략 — 공간적 순간이동을 피한다)"
              % (lo, hi, len(seq)))
    else:
        seq = A + GT + B
        print("플라이스루 %d 포즈 = A 꼬리 %d + 복도 %d + B 머리 %d"
              % (len(seq), len(A), len(GT), len(B)))
        print("  이음새 위치: %d번째(A→복도), %d번째(복도→B)"
              % (len(A), len(A)+len(GT)))

    # 🔴 보간을 **먼저** 하고 평활화를 나중에 한다.
    #    순서를 바꾸면 평활화가 성긴 자세 위에서 걸려 코너를 크게 자른다.
    path0 = densify([imgs[k] for k in seq], a.interp)
    if a.interp > 1:
        print("  보간 ×%d → %d 자세 (%.1f초 @%dfps)"
              % (a.interp, len(path0), len(path0)/a.fps, a.fps))
    path0 = smooth(path0, a.smooth)

    cam = [l for l in open(sfm/"cameras.txt") if not l.startswith("#")][0].split()
    H0, fy = int(cam[3]), float(cam[5])
    fov = float(np.degrees(2*np.arctan(H0/2/fy)))

    outd = Path(a.out); outd.mkdir(parents=True, exist_ok=True)
    flip = np.diag([1., -1., -1., 1.])
    for m in a.models.split(","):
        dp = Path(a.runs)/("G_%s" % m)/"splatfacto/run1/dataparser_transforms.json"
        if not dp.exists():
            print("  🔴 %s 없음 — 건너뜀" % dp); continue
        j = json.load(open(dp))
        M4 = np.eye(4); M4[:3, :4] = np.array(j["transform"])
        s = float(j["scale"])
        cp = []
        for T in path0:
            Tm = M4 @ (T @ flip)
            Tm[:3, 3] *= s
            cp.append(dict(camera_to_world=[float(x) for x in Tm.reshape(-1)],
                           fov=fov, aspect=a.width/a.height))
        js = dict(camera_type="perspective", render_height=a.height,
                  render_width=a.width, fps=a.fps, seconds=len(cp)/a.fps,
                  is_cycle=False, smoothness_value=0, camera_path=cp)
        p = outd/("경로_%s.json" % m)
        json.dump(js, open(p, "w"), indent=1)
        print("  %s  (%d 포즈, %.1f초)" % (p, len(cp), len(cp)/a.fps))


if __name__ == "__main__":
    main()
