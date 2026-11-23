#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""VGGT 로 생성 프레임의 포즈를 얻는다 — **GT1 을 전혀 쓰지 않는다.**

🔴 왜 (2026-11-23 맥에서 제기):
   지금의 고정 등록은 생성 프레임을 합본 DB(숨긴 GT1 실촬영 포함)에 SIFT 매칭해
   포즈를 얻는다. 즉 **포즈를 정답에서 얻고 있다.** VGGT 는 이미지만 보고
   포즈를 내므로 그 누출이 없다.

입력 구성 (맥 설계 ①):
   구간마다  [앞 앵커(실촬영 또는 웨이포인트)] + 생성 프레임 + [뒤 앵커]
   → 구간 내 생성 프레임 포즈와, 앵커 사이의 **상대 포즈(엣지)** 를 얻는다.
   A·B 실촬영은 COLMAP 포즈로 고정 노드가 되고, 엣지를 Sim(3) 포즈 그래프로
   묶는 것은 다음 단계(GT없는정렬.py / 포즈그래프)가 한다.

🔴 규약: VGGT 의 extrinsic 은 **OpenCV, world→camera (camera from world)** 다.
   COLMAP 과 같은 규약이므로 변환이 필요 없다. 카메라 중심은 C = -R^T t.

🔴 메모리: 한 번에 넣는 장수에 따라 VRAM 이 가파르게 는다. --max-frames 로
   제한하고, 넘으면 균등 간격으로 솎는다. 16GB 에서 몇 장까지 되는지는
   --probe 로 먼저 재 본다.
"""
import argparse
import glob
import os
import time

import numpy as np

HOME = os.path.expanduser("~")


def load_model(device, dtype):
    from vggt.models.vggt import VGGT
    m = VGGT.from_pretrained("facebook/VGGT-1B")
    return m.to(device).eval()


def run(model, paths, device, dtype, resolution=518):
    import torch
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    imgs = load_and_preprocess_images(paths).to(device)
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=dtype):
            tok, ps_idx = model.aggregator(imgs[None])
        pose_enc = model.camera_head(tok)[-1]
        extr, intr = pose_encoding_to_extri_intri(pose_enc, imgs.shape[-2:])
    return (extr.squeeze(0).float().cpu().numpy(),
            intr.squeeze(0).float().cpu().numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True,
                    help="이미지 경로 또는 glob. 순서가 곧 프레임 순서다")
    ap.add_argument("--out", required=True, help="결과 npz")
    ap.add_argument("--max-frames", type=int, default=120)
    ap.add_argument("--probe", action="store_true",
                    help="장수를 늘려가며 VRAM 한계를 잰다 (추론은 하지 않는다)")
    a = ap.parse_args()

    import torch
    paths = []
    for p in a.images:
        g = sorted(glob.glob(p))
        paths += g if g else ([p] if os.path.exists(p) else [])
    if not paths:
        print("🔴 이미지가 없다"); return
    print("입력 %d장" % len(paths))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cap = torch.cuda.get_device_capability()[0] if dev == "cuda" else 0
    dtype = torch.bfloat16 if cap >= 8 else torch.float16
    print("장치 %s, dtype %s" % (dev, str(dtype).split(".")[-1]))

    t0 = time.time()
    model = load_model(dev, dtype)
    print("모델 적재 %.0f초" % (time.time()-t0))

    if a.probe:
        # 🔴 추측하지 않고 실제로 늘려가며 잰다
        for n in (8, 16, 24, 32, 48, 64, 96, 128):
            if n > len(paths):
                break
            torch.cuda.reset_peak_memory_stats()
            try:
                t1 = time.time()
                run(model, paths[:n], dev, dtype)
                pk = torch.cuda.max_memory_allocated()/1024**3
                print("  %3d장  %5.1f초  VRAM 최대 %.1f GB" % (n, time.time()-t1, pk))
            except RuntimeError as e:
                print("  %3d장  🔴 %s" % (n, str(e).split("\n")[0][:70]))
                break
            finally:
                torch.cuda.empty_cache()
        return

    if len(paths) > a.max_frames:
        idx = np.linspace(0, len(paths)-1, a.max_frames).astype(int)
        paths = [paths[i] for i in idx]
        print("  %d장으로 솎음 (균등 간격)" % len(paths))

    t1 = time.time()
    extr, intr = run(model, paths, dev, dtype)
    print("추론 %.1f초, extrinsic %s" % (time.time()-t1, extr.shape))

    # C = -R^T t  (OpenCV world→camera 규약)
    C = np.array([-extr[i][:3, :3].T @ extr[i][:3, 3] for i in range(len(extr))])
    d = np.linalg.norm(np.diff(C, axis=0), axis=1)
    print("  인접 프레임 간격: 중앙 %.4f, 최대 %.4f (VGGT 자체 스케일)"
          % (float(np.median(d)), float(d.max())))
    np.savez(a.out, names=np.array([os.path.basename(p) for p in paths]),
             extrinsic=extr, intrinsic=intr, centers=C)
    print("저장 → %s" % a.out)


if __name__ == "__main__":
    main()
