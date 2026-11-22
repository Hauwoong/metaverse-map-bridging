#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""플라이스루가 흐릿한 원인을 가린다 — 병합 탓인가, 그 구간 탓인가.

A 꼬리 자세에서 $G_A$(단독)와 $G_M$(병합)을 각각 렌더해 **같은 실사**와 비교한다.
  G_A 가 선명하고 G_M 이 흐리면  → 병합 비용 (장당 학습 갱신이 줄었다)
  둘 다 흐리면                   → 그 구간 자체가 A 커버리지의 경계라서

🔴 두 모델은 dataparser 정규화가 다르므로 자세를 **모델별로 따로** 옮긴다.
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
    ap.add_argument("--zone", default="atail",
                    choices=["atail", "ahead", "bhead"],
                    help="atail=A 꼬리 40장(복도 쪽 경계), ahead=A 머리 40장(안정 구간)")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--models", default="A,M,AB")
    ap.add_argument("--runs", default=str(HOME/"data/runs"))
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--out", default=str(HOME/"data/구역평가"))
    a = ap.parse_args()

    sfm = Path(a.sfm)
    imgs = read_c2w(sfm/"images.txt")

    def ix(k):
        return int(k.split("_")[1].split(".")[0])

    A = sorted((k for k in imgs if k.startswith("v1791_") and ix(k) <= 143), key=ix)
    B = sorted((k for k in imgs if k.startswith("v1793_")), key=ix)
    sel = {"atail": A[-a.n:], "ahead": A[:a.n], "bhead": B[:a.n]}[a.zone]
    print("%s 구역 %d장: %s ~ %s" % (a.zone, len(sel), sel[0], sel[-1]))

    cam = [l for l in open(sfm/"cameras.txt") if not l.startswith("#")][0].split()
    H0, fy = int(cam[3]), float(cam[5])
    fov = float(np.degrees(2*np.arctan(H0/2/fy)))

    outd = Path(a.out); outd.mkdir(parents=True, exist_ok=True)
    (outd/("목록_%s.txt" % a.zone)).write_text("\n".join(sel), encoding="utf-8")

    flip = np.diag([1., -1., -1., 1.])
    for m in a.models.split(","):
        dp = Path(a.runs)/("G_%s" % m)/"splatfacto/run1/dataparser_transforms.json"
        if not dp.exists():
            print("  🔴 G_%s 없음" % m); continue
        j = json.load(open(dp))
        M4 = np.eye(4); M4[:3, :4] = np.array(j["transform"])
        s = float(j["scale"])
        cp = []
        for k in sel:
            T = M4 @ (imgs[k] @ flip)
            T[:3, 3] *= s
            cp.append(dict(camera_to_world=[float(x) for x in T.reshape(-1)],
                           fov=fov, aspect=a.width/a.height))
        js = dict(camera_type="perspective", render_height=a.height,
                  render_width=a.width, fps=24, seconds=len(cp)/24,
                  is_cycle=False, smoothness_value=0, camera_path=cp)
        p = outd/("경로_%s_%s.json" % (a.zone, m))
        json.dump(js, open(p, "w"), indent=1)
        print("  %s" % p)


if __name__ == "__main__":
    main()
