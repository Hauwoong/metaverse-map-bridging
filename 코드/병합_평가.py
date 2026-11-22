#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""병합 맵을 **숨긴 GT1 의 실제 카메라 포즈**에서 렌더할 경로를 만든다.

표 3(이음새 전후)의 측정이다. 같은 시점에서 두 맵을 재야 비교가 된다 —
  G_M   A + 생성다리 + B   (다리 있음)
  G_AB  A + B 만           (다리 없음, 이음새가 비어 있다)

🔴 GT1 은 두 맵 어느 쪽의 학습에도 들어가지 않았다. 그래서 이 수치는
   "학습에서 본 것을 다시 그린 정도" 가 아니라 **복원 정도**다.

🔴 두 모델은 각자 다른 dataparser 정규화를 쓴다(카메라 집합이 달라
   auto-orient 와 scale 이 다르다). 공통 좌표의 포즈를 **모델별로 따로**
   옮겨야 한다. 한 번 변환해 둘에 쓰면 틀린다.

🔴 COLMAP(OpenCV) 카메라는 x오른쪽 y아래 z앞, nerfstudio 는 x오른쪽 y위 z뒤.
   c2w 에 diag(1,-1,-1) 을 오른쪽 곱한다. (깊이워핑.py 에서 자기투영
   오차 0.0000 으로 검증한 규약과 같다)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sfm", default=str(HOME/"data/sfm_all/sparse/1"))
    ap.add_argument("--models", default="M,AB")
    ap.add_argument("--runs", default=str(HOME/"data/runs"))
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--out", default=str(HOME/"data/평가"))
    a = ap.parse_args()

    sfm = Path(a.sfm)
    imgs = read_c2w(sfm/"images.txt")
    gt = sorted(k for k in imgs
                if k.startswith("v1791_") and 162 <= int(k[6:11]) <= 286)
    print("GT1 %d장의 실제 포즈를 쓴다" % len(gt))

    cam = [l for l in open(sfm/"cameras.txt") if not l.startswith("#")][0].split()
    H0, fy = int(cam[3]), float(cam[5])
    fov = float(np.degrees(2*np.arctan(H0/2/fy)))
    print("세로 화각 %.2f°" % fov)

    outd = Path(a.out); outd.mkdir(parents=True, exist_ok=True)
    (outd/"목록.txt").write_text("\n".join(gt), encoding="utf-8")

    flip = np.diag([1., -1., -1., 1.])
    for m in a.models.split(","):
        dp = Path(a.runs)/("G_%s" % m)/"splatfacto/run1/dataparser_transforms.json"
        if not dp.exists():
            print("  🔴 %s 없음 — 건너뜀" % dp)
            continue
        j = json.load(open(dp))
        M4 = np.eye(4); M4[:3, :4] = np.array(j["transform"])
        s = float(j["scale"])
        path = []
        for k in gt:
            T = M4 @ (imgs[k] @ flip)
            T[:3, 3] *= s
            path.append(dict(camera_to_world=[float(x) for x in T.reshape(-1)],
                             fov=fov, aspect=a.width/a.height))
        js = dict(camera_type="perspective", render_height=a.height,
                  render_width=a.width, fps=a.fps, seconds=len(path)/a.fps,
                  is_cycle=False, smoothness_value=0, camera_path=path)
        p = outd/("경로_%s.json" % m)
        json.dump(js, open(p, "w"), indent=1)
        print("  %s  (%d 포즈, scale %.4f)" % (p, len(path), s))


if __name__ == "__main__":
    main()
