#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A·B 맵 자체를 도는 플라이스루 — "맵이 돈다" 를 보여주는 제출 영상.

기존 `플라이스루.py` 는 **A → 간극 → B** 를 날아 다리의 유무를 비교한다.
그것은 §4.7 의 정성 결과이고, 간극이 깨진 모습을 일부러 보여 준다.

이 스크립트는 다른 것을 보여 준다: **촬영이 덮은 구간에서 맵이 멀쩡하다.**
A 를 한 번 통과하고 B 를 한 번 통과한다. 간극은 들어가지 않는다.
과제의 "구동 장면을 담은 유튜브 동영상" 요건에 바로 쓰인다.

🔴 모델은 `fixreg2/wp4/out` 등록본으로 학습되었으므로 포즈도 거기서 가져온다.
🔴 보간을 **먼저** 하고 평활화를 나중에 한다 (플라이스루.py 의 교훈).
   순서를 바꾸면 평활화가 성긴 자세 위에서 걸려 코너를 크게 자른다.
"""
import argparse, importlib, json, os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
M = importlib.import_module("간극_채움_데이터셋")

HOME = Path(os.path.expanduser("~"))
REG = M.REG


def densify(path, k):
    """인접 자세 사이에 k-1 개를 끼워 넣는다 (위치 선형, 회전 slerp)."""
    if k <= 1:
        return list(path)
    import importlib as _il
    C = _il.import_module("간극_연쇄전파")
    out = []
    for i in range(len(path)-1):
        A, B = path[i], path[i+1]
        for j in range(k):
            u = j/k
            T = np.eye(4)
            T[:3, :3] = C.slerp(A[:3, :3], B[:3, :3], u)
            T[:3, 3] = (1-u)*A[:3, 3] + u*B[:3, 3]
            out.append(T)
    out.append(path[-1])
    return out


def smooth(path, win):
    """위치만 이동평균한다. 회전을 같은 방식으로 섞으면 직교성이 깨진다."""
    if win <= 1:
        return path
    P = np.array([T[:3, 3] for T in path])
    out = []
    for i in range(len(path)):
        lo, hi = max(0, i-win//2), min(len(path), i+win//2+1)
        T = path[i].copy()
        T[:3, 3] = P[lo:hi].mean(0)
        out.append(T)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="AB_cull1")
    ap.add_argument("--runs", default=str(HOME/"data/runs"))
    ap.add_argument("--interp", type=int, default=6,
                    help="촬영이 약 2.5fps 라 24fps 재생은 10배속이 된다")
    ap.add_argument("--smooth", type=int, default=7)
    ap.add_argument("--stride", type=int, default=1, help="원본 자세 솎기")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--out", default=str(HOME/"data/맵플라이스루"))
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    I = M.read_full(REG/"images.txt")
    A, GT, B = M.split_frames(I)
    W0, H0, fx, fy = M.cam_params(a.width, a.height)
    fov = float(np.degrees(2*np.arctan(a.height/2/fy)))

    run = Path(a.runs)/a.model/"splatfacto/run1"
    dp = json.load(open(run/"dataparser_transforms.json"))
    M4 = np.eye(4); M4[:3, :4] = np.array(dp["transform"]); s = float(dp["scale"])
    flip = np.diag([1., -1., -1., 1.])

    for tag, names in (("A", A[::a.stride]), ("B", B[::a.stride])):
        path = [I[k]["c2w"] for k in names]
        path = densify(path, a.interp)
        path = smooth(path, a.smooth)
        cp = []
        for T in path:
            Tn = M4 @ (T @ flip)
            Tn[:3, 3] *= s
            cp.append(dict(camera_to_world=[float(x) for x in Tn.reshape(-1)],
                           fov=fov, aspect=a.width/a.height))
        js = dict(camera_type="perspective", render_height=a.height,
                  render_width=a.width, fps=a.fps, seconds=len(cp)/a.fps,
                  is_cycle=False, smoothness_value=0, camera_path=cp)
        p = out/("경로_%s.json" % tag)
        json.dump(js, open(p, "w"))
        print("%s 구간: 원본 %d 자세 → 보간·평활 %d 자세 (%.1f초 @%dfps)"
              % (tag, len(names), len(cp), len(cp)/a.fps, a.fps))
        print("   %s" % p)

    print("\n세로 화각 %.2f°, %dx%d" % (fov, a.width, a.height))
    print("\n다음:")
    for tag in ("A", "B"):
        print("  ns-render camera-path --load-config %s/config.yml \\" % run)
        print("     --camera-path-filename %s/경로_%s.json \\" % (out, tag))
        print("     --output-path %s/맵_%s.mp4 --output-format video" % (out, tag))


if __name__ == "__main__":
    main()
