#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""이음새를 잰다 — G_A 와 G_C 가 같은 포즈에서 얼마나 다른가.

정의:
  G_A 와 G_C 는 같은 좌표계에 있다(코드/GC_데이터셋.py). 겹치는 구간의 같은 카메라
  포즈에서 양쪽을 렌더하면, 같은 공간을 본다면 비슷해야 한다. **그 차이가 이음새다.**

세 가지를 나눠 잰다 — 원인이 다르면 처방도 다르기 때문이다:
  색   채널별 평균 차이        → 톤 보정으로 고칠 수 있다
  구조 정규화 후 PSNR·상관     → 기하가 어긋난 것. 보정으로 못 고친다
  불투명 누적 알파(accumulation) → 한쪽이 비어 있으면 병합 시 구멍이 보인다

⚠️ 이 측정은 '얼마나 다른가' 만 말한다. 다름의 일부는 정상이다 —
   G_C 는 생성물이라 질감이 다를 수밖에 없다. **전후 비교**로만 의미를 가진다.
"""
import argparse, os, json, glob
from pathlib import Path
import numpy as np

HOME = os.path.expanduser("~")


def q2R(q):
    w, x, y, z = q
    return np.array([[1-2*y*y-2*z*z, 2*x*y-2*z*w,   2*x*z+2*y*w],
                     [2*x*y+2*z*w,   1-2*x*x-2*z*z, 2*y*z-2*x*w],
                     [2*x*z-2*y*w,   2*y*z+2*x*w,   1-2*x*x-2*y*y]])


def read_poses(p):
    out = {}
    lines = [l for l in open(p) if not l.startswith("#")]
    for i in range(0, len(lines), 2):
        t = lines[i].split()
        if len(t) < 10:
            continue
        R = q2R(list(map(float, t[1:5])))
        tv = np.array(list(map(float, t[5:8])))
        c2w = np.eye(4); c2w[:3, :3] = R.T; c2w[:3, 3] = -R.T @ tv
        out[t[9]] = c2w
    return out


def load_model(scene):
    import torch
    from nerfstudio.utils.eval_utils import eval_setup
    run = os.path.join(HOME, "data/runs", "G_"+scene, "splatfacto/run1")
    dp = json.load(open(os.path.join(run, "dataparser_transforms.json")))
    M = np.eye(4); M[:3, :4] = np.array(dp["transform"])
    s = float(dp["scale"])
    # eval_setup 은 Path 를 요구한다 (str 을 넘기면 read_text 가 없다고 터진다)
    _, pipeline, _, _ = eval_setup(Path(run) / "config.yml", test_mode="inference")
    return pipeline.model, M, s


def render(model, M, s, c2w_colmap, W, H, fx, fy):
    import torch
    from nerfstudio.cameras.cameras import Cameras, CameraType
    flip = np.diag([1., -1., -1., 1.])
    T = M @ (c2w_colmap @ flip); T[:3, 3] *= s
    dev = next(model.parameters()).device
    cam = Cameras(camera_to_worlds=torch.tensor(T[:3, :4], dtype=torch.float32)[None],
                  fx=fx, fy=fy, cx=W/2, cy=H/2, width=W, height=H,
                  camera_type=CameraType.PERSPECTIVE).to(dev)
    with torch.no_grad():
        o = model.get_outputs_for_camera(cam)
    rgb = np.clip(o["rgb"].detach().cpu().numpy(), 0, 1)
    acc = o["accumulation"].detach().cpu().numpy().squeeze() if "accumulation" in o else None
    return rgb, acc


def psnr(a, b):
    m = ((a-b)**2).mean()
    return 99.0 if m == 0 else 10*np.log10(1.0/m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12, help="겹치는 구간에서 몇 개 포즈를 볼까")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=576)
    ap.add_argument("--out", default=str(HOME)+"/data/이음새")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # 겹치는 구간의 포즈 = G_C 데이터셋의 앞부분 (A 쪽 경계에서 출발한다)
    cpose = read_poses(os.path.join(HOME, "data/scenes/C/colmap/sparse/0/images.txt"))
    names = sorted(cpose)[:a.n]
    print("겹침 구간 포즈 %d개: %s … %s" % (len(names), names[0], names[-1]))

    cam = [l for l in open(os.path.join(HOME, "data/sfm_all/sparse/1/cameras.txt"))
           if not l.startswith("#")][0].split()
    fx = float(cam[4]) * a.width / int(cam[2])
    fy = float(cam[5]) * a.height / int(cam[3])

    from PIL import Image
    rows = []
    for scene in ("A", "C"):
        model, M, s = load_model(scene)
        outs = []
        for n in names:
            rgb, acc = render(model, M, s, cpose[n], a.width, a.height, fx, fy)
            outs.append((rgb, acc))
            Image.fromarray((rgb*255).astype(np.uint8)).save(
                os.path.join(a.out, "%s_%s" % (scene, n)))
        rows.append(outs)
        del model
        import torch, gc
        gc.collect(); torch.cuda.empty_cache()

    print()
    print("%-18s %8s %8s %8s %8s" % ("포즈", "색차", "PSNR", "정규화PSNR", "A알파"))
    print("-"*58)
    dc, dp_, dn, da = [], [], [], []
    for i, n in enumerate(names):
        A_rgb, A_acc = rows[0][i]
        C_rgb, _ = rows[1][i]
        col = float(np.abs(A_rgb.reshape(-1, 3).mean(0) - C_rgb.reshape(-1, 3).mean(0)).mean())
        p = psnr(A_rgb, C_rgb)
        # 색을 맞춘 뒤의 PSNR = '구조만' 의 차이
        Cn = C_rgb.copy()
        for c in range(3):
            sa, sc = A_rgb[..., c].std(), C_rgb[..., c].std()
            Cn[..., c] = (C_rgb[..., c]-C_rgb[..., c].mean())*(sa/max(sc, 1e-6))+A_rgb[..., c].mean()
        pn = psnr(A_rgb, np.clip(Cn, 0, 1))
        al = float(A_acc.mean()) if A_acc is not None else float("nan")
        print("%-18s %8.4f %8.2f %8.2f %8.3f" % (n, col, p, pn, al))
        dc.append(col); dp_.append(p); dn.append(pn); da.append(al)
    print("-"*58)
    print("%-18s %8.4f %8.2f %8.2f %8.3f" % ("평균", np.mean(dc), np.mean(dp_),
                                             np.mean(dn), np.nanmean(da)))
    print()
    print("읽는 법")
    print("  색차 큼          → 톤 보정으로 고칠 수 있다")
    print("  정규화PSNR 낮음  → 기하가 어긋난 것. 색 보정으로 못 고친다")
    print("  A알파 낮음       → G_A 가 그 시점에서 비어 있다 (병합 시 구멍)")
    print("저장: %s" % a.out)


if __name__ == "__main__":
    main()
