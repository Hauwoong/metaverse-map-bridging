#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""깊이 기반 전방 워핑으로 '앵커를 만들어낸다'.

배경 (10주차 Point-NeRF 강의):
  이미지에 깊이가 있으면 포인트 클라우드를 만들 수 있고, 그러면 임의의 새 시점으로
  투영할 수 있다. 앵커는 어딘가에서 '찾는' 것이 아니라 기하로 '만드는' 것이다.

왜 이것이 앞선 실패들과 다른가:
  - 스플랫 궤적 렌더는 커버리지 밖의 가우시안이 floater 를 만들어 무너졌다.
    워핑은 **이미지에 실제로 있는 픽셀만** 옮기므로 floater 가 낄 자리가 없다.
  - 가이드 영상 조건은 LTX 가 조건을 '내용'으로 그려서 실패했다.
    워핑 결과는 실제로 그려야 할 내용이 맞으므로 이미지 조건으로 바로 쓸 수 있다.
  - A 끝점은 G_A 의 커버리지 '안'이므로 그 깊이는 믿을 수 있다.

단계:
  1) G_A 에서 A 끝점 포즈의 rgb 와 **원시 깊이**를 꺼낸다 (ns-render 는 깊이에
     컬러맵을 씌우므로 쓸 수 없다. 모델을 직접 불러 get_outputs_for_camera 를 쓴다).
  2) 픽셀을 역투영해 3D 점으로 만들고, 조금 전진한 포즈로 재투영한다.
  3) 구멍(가려짐이 풀린 영역)이 남는다. 그 구멍만 생성 모델이 채우면 앵커가 된다.

전부 **모델 정규화 공간**에서 계산한다. 깊이도 그 공간의 단위로 나오므로
좌표계를 섞지 않는 것이 가장 안전하다.
"""
import argparse, json, os
from pathlib import Path
import numpy as np
import torch

HOME = Path(os.path.expanduser("~"))


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
        c2w = np.eye(4); c2w[:3, :3] = R.T; c2w[:3, 3] = -R.T @ tv
        out[t[9]] = c2w
    return out


def to_model_space(c2w_colmap, transform, scale):
    """COLMAP(OpenCV) c2w → nerfstudio 규약 → 모델 정규화 공간."""
    flip = np.diag([1., -1., -1., 1.])      # x우 y하 z전  →  x우 y상 z후
    M = np.eye(4); M[:3, :4] = transform
    T = M @ (c2w_colmap @ flip)
    T[:3, 3] *= scale
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="A")
    ap.add_argument("--frame", default="v1791_00143.png", help="출발 프레임")
    ap.add_argument("--to",    default="v1793_00001.png", help="목표 프레임(방향용)")
    ap.add_argument("--steps", type=float, nargs="+", default=[0.0, 0.05, 0.10, 0.20],
                    help="출발→목표 구간의 몇 % 만큼 전진해 볼 것인가")
    ap.add_argument("--width",  type=int, default=1024)
    ap.add_argument("--height", type=int, default=576)
    ap.add_argument("--out", default=str(HOME / "data/워핑"))
    a = ap.parse_args()

    from nerfstudio.utils.eval_utils import eval_setup
    from nerfstudio.cameras.cameras import Cameras, CameraType

    run = HOME / "data/runs" / ("G_%s" % a.scene) / "splatfacto/run1"
    dp = json.load(open(run / "dataparser_transforms.json"))
    imgs = read_images(HOME / "data/sfm_all/sparse/1/images.txt")
    assert a.frame in imgs and a.to in imgs

    C0 = to_model_space(imgs[a.frame], np.array(dp["transform"]), float(dp["scale"]))
    C1 = to_model_space(imgs[a.to],    np.array(dp["transform"]), float(dp["scale"]))

    cam = [l for l in open(HOME / "data/sfm_all/sparse/1/cameras.txt") if not l.startswith("#")][0].split()
    W0, H0 = int(cam[2]), int(cam[3])
    fx, fy = float(cam[4]), float(cam[5])
    s = a.width / W0
    fx, fy = fx*s, fy*s
    cx, cy = a.width/2, a.height/2
    print("렌더 %dx%d, fx=%.1f fy=%.1f" % (a.width, a.height, fx, fy))

    print("모델 적재…")
    cfg, pipeline, _, _ = eval_setup(run / "config.yml", test_mode="inference")
    model = pipeline.model
    dev = next(model.parameters()).device

    outd = Path(a.out); outd.mkdir(parents=True, exist_ok=True)

    def render(T):
        c = Cameras(camera_to_worlds=torch.tensor(T[:3, :4], dtype=torch.float32)[None],
                    fx=fx, fy=fy, cx=cx, cy=cy, width=a.width, height=a.height,
                    camera_type=CameraType.PERSPECTIVE).to(dev)
        with torch.no_grad():
            o = model.get_outputs_for_camera(c)
        return o

    o0 = render(C0)
    print("출력 키:", list(o0.keys()))
    rgb0 = o0["rgb"].detach().cpu().numpy()
    d0 = o0["depth"].detach().cpu().numpy().squeeze()
    print("깊이 shape %s, 범위 %.3f ~ %.3f (중앙값 %.3f)"
          % (d0.shape, float(d0.min()), float(d0.max()), float(np.median(d0))))

    from PIL import Image
    Image.fromarray((np.clip(rgb0, 0, 1)*255).astype(np.uint8)).save(outd / "출발_rgb.png")
    np.save(outd / "출발_깊이.npy", d0)
    dn = (d0 - d0.min()) / max(1e-6, d0.max()-d0.min())
    Image.fromarray((dn*255).astype(np.uint8)).save(outd / "출발_깊이_미리보기.png")

    # ── 역투영: 픽셀 + 깊이 → 월드 3D 점 ────────────────────────────────
    # nerfstudio 카메라 규약: x 오른쪽, y 위, z 뒤 (즉 보는 방향은 -z).
    # 행 v 는 아래로 증가하므로 y 는 부호를 뒤집는다.
    H, W = d0.shape
    vv, uu = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    x_c =  (uu - cx) * d0 / fx
    y_c = -(vv - cy) * d0 / fy
    z_c = -d0
    pc = np.stack([x_c, y_c, z_c], -1).reshape(-1, 3)
    Pw = pc @ C0[:3, :3].T + C0[:3, 3]
    col = np.clip(rgb0, 0, 1).reshape(-1, 3)

    def project_into(T):
        """월드 점들을 포즈 T 의 카메라로 투영한다. z-버퍼로 앞의 점만 남긴다."""
        R, t = T[:3, :3], T[:3, 3]
        pc1 = (Pw - t) @ R                      # R^T (P-t)  == (P-t) @ R
        d1 = -pc1[:, 2]
        ok = d1 > 1e-4
        u1 = (fx * pc1[:, 0] / np.where(ok, d1, 1) + cx)
        v1 = (cy - fy * pc1[:, 1] / np.where(ok, d1, 1))
        ui = np.round(u1).astype(np.int64); vi = np.round(v1).astype(np.int64)
        ok &= (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        img = np.zeros((H, W, 3), np.float32)
        zbuf = np.full((H, W), np.inf, np.float32)
        # 가까운 것이 이기도록 깊이 내림차순으로 칠한다 (단순 페인터 알고리즘)
        idx = np.where(ok)[0]
        order = idx[np.argsort(-d1[idx])]
        img[vi[order], ui[order]] = col[order]
        zbuf[vi[order], ui[order]] = d1[order]
        hole = ~np.isfinite(zbuf)
        return img, hole

    # 자기 자신으로 투영해 규약이 맞는지 먼저 확인한다 (맞으면 원본이 거의 그대로 나온다)
    self_img, self_hole = project_into(C0)
    err = np.abs(self_img - np.clip(rgb0, 0, 1)).mean()
    print("자기 투영 검증: 평균 오차 %.4f, 구멍 %.2f%%  (오차가 작아야 규약이 맞다)"
          % (err, 100*self_hole.mean()))
    Image.fromarray((self_img*255).astype(np.uint8)).save(outd / "검증_자기투영.png")

    # ── 전진 워핑 ───────────────────────────────────────────────────────
    print()
    print("전진 워핑 (출발→목표 구간의 비율별 구멍 비율)")
    for u in a.steps:
        if u == 0.0:
            continue
        T = np.eye(4)
        # 위치는 선형 보간, 회전도 같은 비율로 (짧은 구간이라 단순 보간으로 충분)
        T[:3, 3] = (1-u)*C0[:3, 3] + u*C1[:3, 3]
        Rrel = C0[:3, :3].T @ C1[:3, :3]
        w = np.clip((np.trace(Rrel)-1)/2, -1, 1); ang = np.arccos(w)
        if ang < 1e-8:
            T[:3, :3] = C0[:3, :3]
        else:
            ax = np.array([Rrel[2,1]-Rrel[1,2], Rrel[0,2]-Rrel[2,0], Rrel[1,0]-Rrel[0,1]])/(2*np.sin(ang))
            th = ang*u
            K = np.array([[0,-ax[2],ax[1]],[ax[2],0,-ax[0]],[-ax[1],ax[0],0]])
            T[:3, :3] = C0[:3, :3] @ (np.eye(3)+np.sin(th)*K+(1-np.cos(th))*(K@K))
        img, hole = project_into(T)
        pct = 100*hole.mean()
        tag = "%03d" % round(u*100)
        Image.fromarray((img*255).astype(np.uint8)).save(outd / ("워핑_%s.png" % tag))
        Image.fromarray((hole*255).astype(np.uint8)).save(outd / ("구멍_%s.png" % tag))
        # 참고: 같은 포즈를 스플랫으로 직접 렌더한 것과 비교 (floater 유무 대조)
        sp = render(T)["rgb"].detach().cpu().numpy()
        Image.fromarray((np.clip(sp,0,1)*255).astype(np.uint8)).save(outd / ("스플랫_%s.png" % tag))
        print("  %4.0f%% 전진 → 구멍 %5.2f%%" % (u*100, pct))

    print()
    print("저장: %s" % outd)


if __name__ == "__main__":
    main()
