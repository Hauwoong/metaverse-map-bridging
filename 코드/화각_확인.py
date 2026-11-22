#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""학습된 모델이 실제로 쓰는 카메라 내부 파라미터를 읽는다.

🔴 왜 필요한가 — 이것 때문에 렌더가 깨졌다.
   합본 COLMAP 카메라는 OPENCV 모델이고 왜곡이 작지 않다 (k1=0.154, k2=-0.291).
   nerfstudio 는 학습 시 **왜곡을 보정하면서 getOptimalNewCameraMatrix 로
   새 초점거리를 계산하고 이미지를 잘라낸다.** 따라서 스플랫이 담고 있는
   화각은 cameras.txt 의 fy 로 계산한 값과 다르다.
   cameras.txt 의 값으로 렌더하면 **스플랫 범위보다 넓게** 잡혀 가장자리에
   빈 영역이 들어온다 — '깨져 보인다' 의 원인이다.

camera_path JSON 은 fov 와 aspect 만 받으므로, 보정 후 K 에서
  fov_y = 2 atan(H/2/fy)
를 다시 계산해야 한다.
"""
import argparse, os, json
from pathlib import Path
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--runs", default=os.path.expanduser("~/data/runs"))
    a = ap.parse_args()
    from nerfstudio.utils.eval_utils import eval_setup

    out = {}
    for m in a.models:
        cfg = Path(a.runs)/("G_%s" % m)/"splatfacto/run1/config.yml"
        if not cfg.exists():
            print("G_%s 없음" % m); continue
        _, pipe, _, _ = eval_setup(cfg, test_mode="inference")
        dm = pipe.datamanager
        cams = getattr(dm, "train_dataset", None)
        cams = cams.cameras if cams is not None else None
        if cams is None:
            print("G_%s 카메라 못 읽음" % m); continue
        fx = float(cams.fx[0]); fy = float(cams.fy[0])
        W = int(cams.width[0]); H = int(cams.height[0])
        fovy = float(np.degrees(2*np.arctan(H/2/fy)))
        fovx = float(np.degrees(2*np.arctan(W/2/fx)))
        out[m] = dict(fx=fx, fy=fy, W=W, H=H, fovy=fovy, fovx=fovx)
        print("G_%-3s  W×H %d×%d  fx %.1f fy %.1f  세로화각 %.2f°  가로화각 %.2f°"
              % (m, W, H, fx, fy, fovy, fovx))
        del pipe
        torch.cuda.empty_cache()

    p = os.path.expanduser("~/data/화각.json")
    json.dump(out, open(p, "w"), indent=1)
    print("\n저장: %s" % p)
    # cameras.txt 기준값과 비교
    c = [l for l in open(os.path.expanduser(
        "~/data/sfm_all/sparse/1/cameras.txt")) if not l.startswith("#")][0].split()
    H0, fy0 = int(c[3]), float(c[5])
    print("cameras.txt 기준: H %d, fy %.1f → 세로화각 %.2f°  (이것을 쓰면 틀린다)"
          % (H0, fy0, np.degrees(2*np.arctan(H0/2/fy0))))


if __name__ == "__main__":
    main()
