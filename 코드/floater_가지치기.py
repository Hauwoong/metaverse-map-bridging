#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""학습 카메라가 아무도 보지 않는 가우시안을 제거한다.

왜 필요한가:
  3DGS 는 관측이 없는 공간을 비워두지 않고 **바늘 모양 가우시안으로 채운다.**
  그 바늘들은 학습 카메라에서 옆으로 스쳐 보이므로 렌더를 거의 망치지 않고,
  따라서 손실이 벌점을 주지 않아 계속 살아남는다.
  G_AB 의 가우시안이 G_M 보다 58,681개 많았던 것이 이 쓰레기다.
  G_M 에도 다리가 덮지 못한 구간(복도 0~43, 122~125)에 같은 것이 있다.

판정 기준 — **학습 카메라 몇 대가 보는가**:
  각 가우시안 중심을 모든 학습 카메라에 투영해, 화면 안이고 앞쪽(z>0)이며
  너무 멀지 않은 경우를 '본다' 로 센다. 0대면 제거한다.
  🔴 불투명도나 크기로 거르지 않는다 — 그러면 정상적인 작은 가우시안도
     날아가고, 바늘인지 아닌지를 모양만으로는 안전하게 판정할 수 없다.
     **'아무도 안 본다' 가 가장 방어 가능한 기준이다.**

🔴 이것은 되돌릴 수 없으므로 원본 체크포인트를 건드리지 않고 **새 파일**로 쓴다.
"""
import argparse
import json
import os
import shutil

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="예: M2")
    ap.add_argument("--runs", default=os.path.expanduser("~/data/runs"))
    ap.add_argument("--min-views", type=int, default=1,
                    help="이 수보다 적게 보이면 제거 (기본 1 = 아무도 안 보면)")
    ap.add_argument("--max-depth", type=float, default=0.0,
                    help="이 거리보다 멀면 '본다'로 세지 않는다. 0이면 자동"
                         " (학습 카메라 사이 최대 거리의 1.5배)")
    ap.add_argument("--out-tag", default="pruned")
    a = ap.parse_args()

    from nerfstudio.utils.eval_utils import eval_setup
    from pathlib import Path

    run = Path(a.runs)/("G_%s" % a.model)/"splatfacto/run1"
    cfg = run/"config.yml"
    assert cfg.exists(), "없다: %s" % cfg
    _, pipe, _, _ = eval_setup(cfg, test_mode="inference")
    model = pipe.model
    cams = pipe.datamanager.train_dataset.cameras

    means = model.means.detach()                     # (N,3)
    N = means.shape[0]
    dev = means.device
    c2w = cams.camera_to_worlds.to(dev).float()      # (M,3,4)
    fx = cams.fx.to(dev).float().view(-1)
    fy = cams.fy.to(dev).float().view(-1)
    cx = cams.cx.to(dev).float().view(-1)
    cy = cams.cy.to(dev).float().view(-1)
    W = cams.width.to(dev).view(-1).float()
    H = cams.height.to(dev).view(-1).float()
    M = c2w.shape[0]

    centers = c2w[:, :3, 3]
    if a.max_depth > 0:
        maxd = a.max_depth
    else:
        d = torch.cdist(centers, centers)
        maxd = float(d.max()) * 1.5
    print("가우시안 %d개, 학습 카메라 %d대, 최대 거리 %.3f" % (N, M, maxd))

    seen = torch.zeros(N, dtype=torch.int32, device=dev)
    # nerfstudio c2w 는 x오른쪽 y위 z뒤. 카메라 좌표로 보내고 z 를 뒤집는다.
    for i in range(M):
        R = c2w[i, :3, :3]
        t = c2w[i, :3, 3]
        p = (means - t) @ R                       # world -> camera
        z = -p[:, 2]                              # z뒤 규약이므로 앞쪽은 -z
        ok = (z > 1e-3) & (z < maxd)
        if not bool(ok.any()):
            continue
        u = fx[i] * (p[:, 0] / z.clamp(min=1e-6)) + cx[i]
        v = fy[i] * (-p[:, 1] / z.clamp(min=1e-6)) + cy[i]
        ok = ok & (u >= 0) & (u < W[i]) & (v >= 0) & (v < H[i])
        seen += ok.to(torch.int32)

    keep = seen >= a.min_views
    nk = int(keep.sum())
    print("  보는 카메라 0대: %d개 (%.1f%%) 제거 → %d개 남음"
          % (N-nk, 100*(N-nk)/N, nk))
    print("  보는 카메라 수 분포: 중앙 %d, 평균 %.1f, 최대 %d"
          % (int(seen.median()), float(seen.float().mean()), int(seen.max())))

    if nk == N:
        print("  제거할 것이 없다"); return

    # 체크포인트를 직접 고쳐 새 run 으로 쓴다 (원본은 건드리지 않는다)
    ck = sorted((run/"nerfstudio_models").glob("*.ckpt"))[-1]
    d = torch.load(ck, map_location="cpu", weights_only=False)
    st = d["pipeline"]
    km = keep.cpu()
    n_changed = 0
    for k in list(st.keys()):
        v = st[k]
        if torch.is_tensor(v) and v.ndim >= 1 and v.shape[0] == N:
            st[k] = v[km]
            n_changed += 1
    print("  잘라낸 텐서 %d개" % n_changed)

    newrun = Path(a.runs)/("G_%s_%s" % (a.model, a.out_tag))/"splatfacto/run1"
    (newrun/"nerfstudio_models").mkdir(parents=True, exist_ok=True)
    torch.save(d, newrun/"nerfstudio_models"/ck.name)
    for f in ("config.yml", "dataparser_transforms.json"):
        if (run/f).exists():
            shutil.copy(run/f, newrun/f)
    # 🔴 config 의 experiment_name 을 바꿔야 한다.
    #    nerfstudio 는 output_dir/experiment_name/method/timestamp 로
    #    체크포인트 폴더를 **조립**하므로, 경로 문자열을 치환하는 것으로는
    #    안 된다 (그 형태로 파일에 들어 있지 않다).
    #    한 번 이것으로 원본 체크포인트를 불러와 가지치기 효과가 0으로
    #    측정됐다 — PSNR 이 소수점까지 같아서 알아챘다.
    cfgp = newrun/"config.yml"
    lines = cfgp.read_text(encoding="utf-8").splitlines()
    hit = False
    for i, l in enumerate(lines):
        if l.startswith("experiment_name:"):
            lines[i] = "experiment_name: G_%s_%s" % (a.model, a.out_tag)
            hit = True
            break
    assert hit, "config.yml 에 experiment_name 이 없다"
    cfgp.write_text(chr(10).join(lines) + chr(10), encoding="utf-8")
    print("  저장 → %s" % newrun)


if __name__ == "__main__":
    main()
