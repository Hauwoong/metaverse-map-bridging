#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A·B 맵을 Unity 에 올릴 수 있는 **단일 PLY** 로 내보낸다.

왜 단일 PLY 인가 (계획서 §7 에 이미 적어 둔 경고):
  🔴 A / 연결 / B 를 세 에셋으로 import 하면 에셋 간 깊이 정렬이 안 돼서
     경계에서 popping 이 보인다. Sim(3) 적용 후 **하나의 PLY 로 병합**해
     단일 에셋으로 넣어야 한다.
  다행히 이 프로젝트는 처음부터 **재학습 방식**을 썼다(병합_데이터셋.py).
  G_AB 는 A 와 B 를 한 좌표계에서 함께 학습한 **하나의 모델**이므로
  그대로 내보내면 단일 에셋이다. 변환도 SH 회전도 필요 없다.

왜 OBB 크롭인가:
  `ns-export gaussian-splat` 에는 `--obb-center/-rotation/-scale` 이 있다.
  학습을 건드리지 않고 **촬영이 실제로 덮은 범위 밖의 floater 를 잘라낸다.**
  culling(학습 중)과 상보적이다 — culling 은 '작고 투명한 것', OBB 는 '멀리 뜬 것'.

상자는 **A∪B 의 SfM 점**에서 분위수로 잡는다. 카메라 위치만 쓰면 복도 끝 벽이
잘려 나가고, 전체 점을 다 쓰면 이상치 하나 때문에 상자가 터진다.

🔴 OBB 는 **모델 공간**(dataparser 변환이 적용된 공간) 기준이다.
   COLMAP 점에 transform 을 곱하고 scale 을 곱해서 넘겨야 한다.
"""
import argparse, json, os, subprocess, sys
from pathlib import Path
import numpy as np

HOME = Path(os.path.expanduser("~"))
REG = HOME / "data/fixreg2/wp4/out"


def idx_of(k):
    return int(k.split("_")[1].split(".")[0])


def read_obs(p):
    """이름 -> 그 이미지가 본 point3D id 들 (포즈는 필요 없다)."""
    out = {}
    lines = [l for l in open(p) if not l.startswith("#")]
    for i in range(0, len(lines), 2):
        t = lines[i].split()
        if len(t) < 10:
            continue
        pids = set()
        if i+1 < len(lines):
            v = lines[i+1].split()
            for k in range(0, len(v)-2, 3):
                pid = int(v[k+2])
                if pid != -1:
                    pids.add(pid)
        out[t[9]] = pids
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="G_AB")
    ap.add_argument("--runs", default=str(HOME/"data/runs"))
    ap.add_argument("--out", default=str(HOME/"data/맵"))
    ap.add_argument("--q", type=float, default=0.005,
                    help="분위수. 양끝 이만큼을 이상치로 버린다")
    ap.add_argument("--pad", type=float, default=1.15, help="상자 여유")
    ap.add_argument("--color-mode", default="sh_coeffs", choices=["sh_coeffs", "rgb"])
    ap.add_argument("--no-crop", action="store_true")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    run = Path(a.runs)/a.model/"splatfacto/run1"
    cfg = run/"config.yml"
    assert cfg.exists(), "없다: %s" % cfg

    dp = json.load(open(run/"dataparser_transforms.json"))
    M = np.array(dp["transform"])          # 3x4
    s = float(dp["scale"])

    obs = read_obs(REG/"images.txt")
    keep = {pid for k, pids in obs.items()
            for pid in pids
            if (k.startswith("v1791_") and idx_of(k) <= 143) or k.startswith("v1793_")}
    pts = []
    for l in open(REG/"points3D.txt"):
        if l.startswith("#"):
            continue
        v = l.split()
        if len(v) >= 8 and int(v[0]) in keep:
            pts.append((float(v[1]), float(v[2]), float(v[3])))
    P = np.array(pts)
    print("A∪B 가 본 3D 점 %d개" % len(P))

    # COLMAP -> 모델 공간
    Pm = (np.c_[P, np.ones(len(P))] @ M.T) * s
    lo = np.quantile(Pm, a.q, axis=0)
    hi = np.quantile(Pm, 1-a.q, axis=0)
    center = (lo+hi)/2
    scale = (hi-lo)*a.pad
    print("모델 공간 범위  lo %s" % np.array2string(lo, precision=3))
    print("              hi %s" % np.array2string(hi, precision=3))
    print("OBB center %s" % np.array2string(center, precision=4))
    print("OBB scale  %s  (여유 %.2f배)" % (np.array2string(scale, precision=4), a.pad))

    outd = Path(a.out); outd.mkdir(parents=True, exist_ok=True)
    cmd = ["ns-export", "gaussian-splat",
           "--load-config", str(cfg),
           "--output-dir", str(outd),
           "--output-filename", "%s.ply" % a.model,
           "--ply-color-mode", a.color_mode]
    if not a.no_crop:
        cmd += ["--obb-center", *["%.6f" % x for x in center],
                "--obb-rotation", "0", "0", "0",
                "--obb-scale", *["%.6f" % x for x in scale]]
    print("\n실행:\n  %s\n" % " ".join(cmd))
    if a.dry:
        return
    r = subprocess.run(cmd)
    p = outd/("%s.ply" % a.model)
    if p.exists():
        print("\n내보냄: %s  (%.1f MB)" % (p, p.stat().st_size/1e6))
        print("🔴 Unity 로 가져갈 때: 단일 에셋으로 넣을 것. 스케일이 어긋나면 "
              "모델 공간이라 그렇다. 색공간(Gamma/Linear)도 확인.")
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
