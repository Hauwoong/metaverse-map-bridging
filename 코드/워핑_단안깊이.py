#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""단안 깊이로 전방 워핑해 '가까운 기하적 목표' 를 만든다.

배경:
  - 끝점이 멀면 모델이 보간한다. 끝점이 없으면 방향을 잃는다(12걸음: 딴 건물 / 161f: 천장).
    → 매 걸음에 **가까운** 기하적 목표를 줘야 한다.
  - 경로 10% 까지는 G_A 스플랫 렌더로 목표를 만들 수 있고 실제로 작동했다.
    그 밖에는 재료가 없어 깊이 워핑이 필요하다.
  - 앞선 워핑은 **스플랫 깊이**를 썼는데, 그 깊이는 복도 바닥 평면 잔차가 23.75% 로
    나쁘다(단안은 2.89%). 바닥 타일 줄눈과 반사를 깊이로 착각한다.
    → 단안 깊이로 다시 한다.

개선점 (앞선 구현 대비):
  - 점 하나씩 찍지 않고 **반경을 준다**. 전진하면 점 간격이 벌어져 얼룩이 생겼다.
  - 구멍을 **가까운 값으로 메운다**. 조건 이미지는 '그럴듯한 사진' 이어야 한다 —
    LTX 는 조건을 내용으로 그리므로 구멍을 그대로 두면 구멍을 그린다.
"""
import argparse, os, json
import numpy as np
from PIL import Image

HOME = os.path.expanduser("~")


def q2R(q):
    w, x, y, z = q
    return np.array([[1-2*y*y-2*z*z, 2*x*y-2*z*w,   2*x*z+2*y*w],
                     [2*x*y+2*z*w,   1-2*x*x-2*z*z, 2*y*z-2*x*w],
                     [2*x*z-2*y*w,   2*y*z+2*x*w,   1-2*x*x-2*y*y]])


def read_images(p):
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


def slerp_R(Ra, Rb, u):
    Rrel = Ra.T @ Rb
    w = np.clip((np.trace(Rrel)-1)/2, -1, 1)
    ang = np.arccos(w)
    if ang < 1e-8:
        return Ra.copy()
    ax = np.array([Rrel[2,1]-Rrel[1,2], Rrel[0,2]-Rrel[2,0], Rrel[1,0]-Rrel[0,1]])/(2*np.sin(ang))
    a = ang*u
    K = np.array([[0,-ax[2],ax[1]],[ax[2],0,-ax[0]],[-ax[1],ax[0],0]])
    return Ra @ (np.eye(3)+np.sin(a)*K+(1-np.cos(a))*(K@K))


def mono_depth(img, device=0):
    from transformers import pipeline
    pipe = pipeline("depth-estimation",
                    model="depth-anything/Depth-Anything-V2-Small-hf", device=device)
    d = pipe(img)["predicted_depth"]
    d = d.numpy() if hasattr(d, "numpy") else np.asarray(d)
    d = np.squeeze(d)
    if d.shape != (img.size[1], img.size[0]):
        d = np.asarray(Image.fromarray(d).resize(img.size, Image.BILINEAR))
    return d.astype(np.float64)


def splat_at(scene, c2w_colmap, W, H, fx, fy):
    """같은 포즈를 G_scene 에서 렌더한다 (비교 기준용)."""
    import torch
    from nerfstudio.utils.eval_utils import eval_setup
    from nerfstudio.cameras.cameras import Cameras, CameraType
    run = os.path.join(HOME, "data/runs", "G_"+scene, "splatfacto/run1")
    dp = json.load(open(os.path.join(run, "dataparser_transforms.json")))
    M = np.eye(4); M[:3, :4] = np.array(dp["transform"]); s = float(dp["scale"])
    flip = np.diag([1., -1., -1., 1.])
    T = M @ (c2w_colmap @ flip); T[:3, 3] *= s
    _, pipeline, _, _ = eval_setup(os.path.join(run, "config.yml"), test_mode="inference")
    model = pipeline.model
    dev = next(model.parameters()).device
    cam = Cameras(camera_to_worlds=torch.tensor(T[:3, :4], dtype=torch.float32)[None],
                  fx=fx, fy=fy, cx=W/2, cy=H/2, width=W, height=H,
                  camera_type=CameraType.PERSPECTIVE).to(dev)
    with torch.no_grad():
        o = model.get_outputs_for_camera(cam)
    return (np.clip(o["rgb"].detach().cpu().numpy(), 0, 1) * 255).astype(np.uint8)


def forward_warp(rgb, depth, C0, C1, fx, fy, radius=1):
    """픽셀을 역투영해 목표 포즈로 재투영한다. 반경을 줘 얼룩을 줄인다."""
    H, W = depth.shape
    cx, cy = W/2.0, H/2.0
    vv, uu = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    pc = np.stack([(uu-cx)*depth/fx, -(vv-cy)*depth/fy, -depth], -1).reshape(-1, 3)
    Pw = pc @ C0[:3, :3].T + C0[:3, 3]
    col = rgb.reshape(-1, 3).astype(np.float32)

    R, t = C1[:3, :3], C1[:3, 3]
    p1 = (Pw - t) @ R
    d1 = -p1[:, 2]
    ok = d1 > 1e-4
    u1 = fx*p1[:, 0]/np.where(ok, d1, 1) + cx
    v1 = cy - fy*p1[:, 1]/np.where(ok, d1, 1)

    img = np.zeros((H, W, 3), np.float32)
    zb = np.full((H, W), np.inf, np.float32)
    idx = np.where(ok)[0]
    order = idx[np.argsort(-d1[idx])]     # 먼 것부터 칠한다 (가까운 것이 이김)
    for dy in range(-radius, radius+1):
        for dx in range(-radius, radius+1):
            ui = np.round(u1[order]).astype(np.int64) + dx
            vi = np.round(v1[order]).astype(np.int64) + dy
            m = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
            img[vi[m], ui[m]] = col[order][m]
            zb[vi[m], ui[m]] = d1[order][m]
    hole = ~np.isfinite(zb)
    return img.astype(np.uint8), hole


def fill_holes(img, hole):
    """구멍을 주변 값으로 메운다. 조건 이미지는 '그럴듯한 사진' 이어야 한다 —
    LTX 는 조건을 내용으로 그리므로 구멍을 두면 구멍을 그린다."""
    import cv2
    return cv2.inpaint(img, hole.astype(np.uint8)*255, 5, cv2.INPAINT_TELEA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", default="v1791_00143.png")
    ap.add_argument("--to", default="v1793_00001.png")
    ap.add_argument("--steps", type=float, nargs="+", default=[0.05, 0.10])
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=576)
    ap.add_argument("--src", default=str(HOME)+"/data/끝점/I_A_out.png")
    ap.add_argument("--out", default=str(HOME)+"/data/워핑단안")
    ap.add_argument("--splat", default="A", help="비교 기준으로 렌더할 장면 (빈 문자열이면 생략)")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    imgs = read_images(os.path.join(HOME, "data/sfm_all/sparse/1/images.txt"))
    C0, C1 = imgs[a.frame], imgs[a.to]

    cam = [l for l in open(os.path.join(HOME, "data/sfm_all/sparse/1/cameras.txt"))
           if not l.startswith("#")][0].split()
    W0, H0 = int(cam[2]), int(cam[3])
    fx = float(cam[4]) * a.width / W0
    fy = float(cam[5]) * a.height / H0

    src = Image.open(a.src).convert("RGB").resize((a.width, a.height), Image.LANCZOS)
    rgb = np.asarray(src)
    print("단안 깊이 추정…")
    d = mono_depth(src)
    # 단안은 스케일이 임의다. 간극 기하에 맞추려면 절대 스케일이 필요하다.
    # G_A 의 깊이 중앙값으로 맞춘다 (A 끝점은 커버리지 안이라 중앙값은 신뢰할 만하다).
    gt_path = os.path.join(HOME, "data/워핑/출발_깊이.npy")
    if os.path.exists(gt_path):
        g = np.load(gt_path)
        d = d * (np.median(g) / max(np.median(d), 1e-6))
        print("  스케일 정렬: 중앙값 %.3f (스플랫 %.3f)" % (np.median(d), np.median(g)))

    for u in a.steps:
        T = np.eye(4)
        T[:3, :3] = slerp_R(C0[:3, :3], C1[:3, :3], u)
        T[:3, 3] = (1-u)*C0[:3, 3] + u*C1[:3, 3]
        w, hole = forward_warp(rgb, d, C0, T, fx, fy, radius=1)
        f = fill_holes(w, hole)
        tag = "%03d" % round(u*100)
        Image.fromarray(w).save(os.path.join(a.out, "워핑_%s.png" % tag))
        Image.fromarray(f).save(os.path.join(a.out, "워핑메움_%s.png" % tag))
        print("  %4.0f%% 전진 → 구멍 %5.2f%% (메움 전)" % (u*100, 100*hole.mean()))
        if a.splat:
            Image.fromarray(splat_at(a.splat, T, a.width, a.height, fx, fy)).save(
                os.path.join(a.out, "스플랫_%s.png" % tag))
    print("저장: %s" % a.out)


if __name__ == "__main__":
    main()
