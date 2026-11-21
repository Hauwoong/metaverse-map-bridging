#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""끝점 좌표를 이용해 간극을 가로지르는 카메라 궤적을 만들고, G_A·G_B 에서 각각 렌더한다.

왜 하는가:
  게임 엔진에 올린다면 두 끝점의 좌표는 당연히 알려져 있다. 합본 COLMAP 이 이미
  세 영상을 한 좌표계에 두었으므로 우리도 A 끝점과 B 끝점의 6-DoF 포즈를 안다.
  그런데 LTX 에게는 사진 두 장만 주고 있어서, 모델은 이 간극이 얼마나 먼지 모른다.

  먼저 확인할 것: **끝점 스플랫이 간극을 어디까지 덮는가.**
  A 의 카메라가 복도 안쪽을 보고 있었다면 G_A 는 간극의 앞부분을 이미 담고 있다.
  양쪽이 각각 일부를 덮는다면 **정답(G_GT1)을 쓰지 않고도 웨이포인트를 얻는다** —
  오늘 재귀 방식이 실패한 원인(오염된 웨이포인트)이 해결된다.

출력:
  ~/data/궤적/{A,B}/ 에 렌더 영상, 그리고 카메라 경로 JSON.

주의:
  - COLMAP(OpenCV) 카메라는 x오른쪽 y아래 z앞, nerfstudio 는 x오른쪽 y위 z뒤다.
    c2w 에 diag(1,-1,-1) 을 오른쪽 곱해 변환한다.
  - G_A 와 G_B 는 각자 다른 정규화(dataparser_transforms.json)를 쓴다.
    공통 좌표의 궤적을 **각 모델의 변환으로 따로** 옮겨야 한다.
"""
import argparse, json, os
from pathlib import Path
import numpy as np

HOME = Path(os.path.expanduser("~"))
SFM  = HOME / "data/sfm_all/sparse/1"
RUNS = HOME / "data/runs"


def qvec2R(q):
    w, x, y, z = q
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*z*w,   2*x*z+2*y*w],
        [2*x*y+2*z*w,   1-2*x*x-2*z*z, 2*y*z-2*x*w],
        [2*x*z-2*y*w,   2*y*z+2*x*w,   1-2*x*x-2*y*y]])


def read_images(p):
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


def slerp(Ra, Rb, u):
    """두 회전 사이를 보간한다. 축-각 형태로 바꿔 선형 보간하는 것과 같다."""
    Rrel = Ra.T @ Rb
    w = np.clip((np.trace(Rrel) - 1) / 2, -1, 1)
    ang = np.arccos(w)
    if ang < 1e-8:
        return Ra.copy()
    ax = np.array([Rrel[2, 1]-Rrel[1, 2], Rrel[0, 2]-Rrel[2, 0], Rrel[1, 0]-Rrel[0, 1]])
    ax = ax / (2*np.sin(ang))
    a = ang * u
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return Ra @ (np.eye(3) + np.sin(a)*K + (1-np.cos(a))*(K @ K))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", default="v1791_00143.png")
    ap.add_argument("--last",  default="v1793_00001.png")
    ap.add_argument("--frames", type=int, default=161)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=576)
    ap.add_argument("--out", default=str(HOME / "data/궤적"))
    a = ap.parse_args()

    imgs = read_images(SFM / "images.txt")
    for n in (a.first, a.last):
        assert n in imgs, "합본 모델에 %s 가 없다" % n
    Ca, Cb = imgs[a.first], imgs[a.last]
    d = np.linalg.norm(Ca[:3, 3] - Cb[:3, 3])
    print("간극 직선거리 %.3f (COLMAP 단위), 프레임 %d" % (d, a.frames))

    # 세로 화각: cameras.txt 의 fy 와 높이에서
    cam = [l for l in open(SFM / "cameras.txt") if not l.startswith("#")][0].split()
    W0, H0, fy = int(cam[2]), int(cam[3]), float(cam[5])
    fov = float(np.degrees(2*np.arctan(H0 / 2 / fy)))
    print("세로 화각 %.2f°  (fy=%.1f, H=%d)" % (fov, fy, H0))

    # 공통 좌표계에서 궤적 생성 (위치는 직선 보간, 회전은 slerp)
    traj = []
    for i in range(a.frames):
        u = i / (a.frames - 1)
        T = np.eye(4)
        T[:3, :3] = slerp(Ca[:3, :3], Cb[:3, :3], u)
        T[:3, 3] = (1-u)*Ca[:3, 3] + u*Cb[:3, 3]
        traj.append(T)

    # OpenCV -> nerfstudio 축 변환
    flip = np.diag([1., -1., -1., 1.])

    outd = Path(a.out); outd.mkdir(parents=True, exist_ok=True)
    for scene in ("A", "B"):
        dp = RUNS / ("G_%s" % scene) / "splatfacto/run1/dataparser_transforms.json"
        j = json.load(open(dp))
        M = np.array(j["transform"])          # 3x4
        s = float(j["scale"])
        M4 = np.eye(4); M4[:3, :4] = M

        path = []
        for T in traj:
            Tn = T @ flip                      # nerfstudio 카메라 규약
            Tm = M4 @ Tn                       # 모델 정규화 공간으로
            Tm[:3, 3] *= s
            path.append(dict(camera_to_world=[float(x) for x in Tm.reshape(-1)],
                             fov=fov, aspect=a.width/a.height))
        js = dict(camera_type="perspective", render_height=a.height,
                  render_width=a.width, fps=a.fps,
                  seconds=a.frames/a.fps, is_cycle=False, smoothness_value=0,
                  camera_path=path)
        p = outd / ("경로_G_%s.json" % scene)
        json.dump(js, open(p, "w"), indent=1)
        print("  %s  (%d 포즈)" % (p, len(path)))

    print()
    print("다음: ns-render camera-path --load-config <G_X config> \\")
    print("        --camera-path-filename %s/경로_G_X.json \\" % outd)
    print("        --output-path %s/G_X.mp4" % outd)


if __name__ == "__main__":
    main()
