#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A·B 가 간극으로 새는가 — culling 설정별로 잰다.

왜 이것을 재는가 (2026-11-28):
  어제 간극 63 포즈를 렌더하니 `accumulation > 0.5` 인 화소가 **전부 100%** 였다.
  맵에게 "어디가 비었나" 를 물어볼 수 없다는 뜻이고, 그래서 "빈 곳을 채운다" 가
  성립하지 않았다.

  그 가우시안들은 **A 와 B 자신의 것**이다. A·B 시점에서만 구속되었으므로
  간극 시점에서는 아무 제약이 없어 늘어난 반투명 덩어리로 퍼진다.
  → A·B 가 새지 않게 만들면 간극이 **진짜로 비고**, accumulation 이 다시
    커버리지 신호가 된다. 그러면 어제 막힌 실험이 다시 가능해진다.

  이것은 Point-NeRF(7·10주차)의 **point pruning & growing** 과 같은 발상이다.
  🔴 사후 가지치기는 이미 실패했다(−0.06 dB). 그래서 **학습 중** culling 을 본다.

두 가지를 동시에 봐야 한다. 하나만 보면 속는다:
  ① 간극에서  accumulation 이 내려가는가        (새지 않는가)
  ② A·B 자기 뷰에서 품질이 유지되는가            (맵을 망친 게 아닌가)
"""
import argparse, json, os
from pathlib import Path
import numpy as np
from PIL import Image

HOME = Path(os.path.expanduser("~"))
REG = HOME / "data/fixreg2/wp4/out"
IMGS = HOME / "data/fixreg2/wp4/images"


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


def idx_of(k):
    return int(k.split("_")[1].split(".")[0])


def psnr(a, b):
    d = (a.astype(np.float64) - b.astype(np.float64)) / 255.0
    m = float((d ** 2).mean())
    return 99.0 if m < 1e-12 else float(10 * np.log10(1.0 / m))


class Model:
    def __init__(self, run):
        import torch
        from nerfstudio.utils.eval_utils import eval_setup
        dp = json.load(open(Path(run) / "dataparser_transforms.json"))
        self.M = np.eye(4); self.M[:3, :4] = np.array(dp["transform"])
        self.s = float(dp["scale"])
        _, pipeline, _, _ = eval_setup(Path(run) / "config.yml", test_mode="inference")
        self.model = pipeline.model
        self.torch = torch
        self.dev = next(self.model.parameters()).device
        self.n = int(self.model.means.shape[0])

    def render(self, c2w, W, H, fx, fy):
        from nerfstudio.cameras.cameras import Cameras, CameraType
        flip = np.diag([1., -1., -1., 1.])
        T = self.M @ (c2w @ flip); T[:3, 3] *= self.s
        cam = Cameras(camera_to_worlds=self.torch.tensor(T[:3, :4],
                                                         dtype=self.torch.float32)[None],
                      fx=fx, fy=fy, cx=W/2, cy=H/2, width=W, height=H,
                      camera_type=CameraType.PERSPECTIVE).to(self.dev)
        with self.torch.no_grad():
            o = self.model.get_outputs_for_camera(cam)
        rgb = (np.clip(o["rgb"].detach().cpu().numpy(), 0, 1)*255).astype(np.uint8)
        acc = np.squeeze(o["accumulation"].detach().cpu().numpy())
        return rgb, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="G_AB,AB_cull1,AB_cull2")
    ap.add_argument("--runs", default=str(HOME/"data/runs"))
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--gap-stride", type=int, default=8)
    ap.add_argument("--real-n", type=int, default=12)
    ap.add_argument("--out", default=str(HOME/"data/커버리지비교.json"))
    a = ap.parse_args()

    W, H = a.width, a.height
    poses = read_poses(REG/"images.txt")
    t = [l for l in open(REG/"cameras.txt") if not l.startswith("#")][0].split()
    W0, H0 = int(t[2]), int(t[3])
    fx, fy = float(t[4])*W/W0, float(t[5])*H/H0

    A  = sorted((k for k in poses if k.startswith("v1791_") and idx_of(k) <= 143), key=idx_of)
    GT = sorted((k for k in poses if k.startswith("v1791_") and 162 <= idx_of(k) <= 286), key=idx_of)
    B  = sorted((k for k in poses if k.startswith("v1793_")), key=idx_of)
    gap = GT[::a.gap_stride]
    realA = A[::max(1, len(A)//a.real_n)][:a.real_n]
    realB = B[::max(1, len(B)//a.real_n)][:a.real_n]
    print("간극 %d 포즈, A %d, B %d 로 잰다\n" % (len(gap), len(realA), len(realB)))

    res = {}
    for name in a.models.split(","):
        run = Path(a.runs)/name/"splatfacto/run1"
        if not (run/"config.yml").exists():
            print("%-10s 없음 — 건너뜀" % name)
            continue
        m = Model(str(run))

        accs, acc50, gp = [], [], []
        for k in gap:
            rgb, acc = m.render(poses[k], W, H, fx, fy)
            accs.append(float(acc.mean()))
            acc50.append(float((acc > 0.5).mean()))
            g = IMGS/k
            if g.exists():
                gt = np.asarray(Image.open(g).convert("RGB").resize((W, H), Image.LANCZOS))
                gp.append(psnr(rgb, gt))

        qa = []
        for k in realA + realB:
            rgb, _ = m.render(poses[k], W, H, fx, fy)
            g = IMGS/k
            if g.exists():
                gt = np.asarray(Image.open(g).convert("RGB").resize((W, H), Image.LANCZOS))
                qa.append(psnr(rgb, gt))

        res[name] = dict(gaussians=m.n,
                         gap_acc_mean=float(np.mean(accs)),
                         gap_acc50=float(np.mean(acc50)),
                         gap_psnr=float(np.median(gp)) if gp else float("nan"),
                         real_psnr=float(np.median(qa)) if qa else float("nan"))
        r = res[name]
        print("%-10s 가우시안 %7d | 간극 acc평균 %.3f  acc>0.5 %5.1f%%  vsGT1 %5.2f dB "
              "| A·B 자기뷰 %5.2f dB"
              % (name, r["gaussians"], r["gap_acc_mean"], 100*r["gap_acc50"],
                 r["gap_psnr"], r["real_psnr"]), flush=True)
        del m

    print("\n" + "="*84)
    print("  %-10s %9s %10s %10s %10s %10s" %
          ("모델", "가우시안", "간극acc", "acc>0.5", "간극PSNR", "A·B뷰PSNR"))
    for k, r in res.items():
        print("  %-10s %9d %10.3f %9.1f%% %9.2f %10.2f"
              % (k, r["gaussians"], r["gap_acc_mean"], 100*r["gap_acc50"],
                 r["gap_psnr"], r["real_psnr"]))
    print("="*84)
    print("  판정: 간극 acc 가 내려가면서 A·B뷰 PSNR 이 유지되면 성공.")
    print("        A·B뷰가 같이 떨어지면 맵을 망친 것이다.")
    json.dump(res, open(a.out, "w"), indent=1)
    print("  저장: %s" % a.out)


if __name__ == "__main__":
    main()
