#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""촬영선에서 얼마나 벗어나면 맵이 깨지는가 — 자유시점 내성.

왜 이것을 재는가 (2026-11-28):
  A·B 맵 플라이스루는 깨끗했다. 그런데 뷰어에서 직접 돌아다녀 보면 엉망이다.
  🔴 **플라이스루는 촬영 포즈를 그대로 따라간 것**이었다. 카메라가 실제로 지나간
     자리에서만 멀쩡하다는 뜻이고, 그것은 '맵' 이 아니라 '재생' 에 가깝다.

  게임 맵이 되려면 **촬영선을 벗어나도** 버텨야 한다. 그 한계를 잰다.

재는 법:
  촬영 포즈를 기준으로 **옆으로 / 위로 / 뒤로** 조금씩 밀어 가며 렌더하고,
  각 거리에서 '깨진 정도' 를 센다.

  깨짐의 지표는 **깊이 온전성**이다 (어제 찾은 신호):
    렌더 깊이가 비정상적으로 가까우면 그것은 표면이 아니라 floater 다.
    복도 깊이는 약 5 (COLMAP 단위)인데 floater 속에 들어가면 0.1~0.3 이 된다.
  🔴 `accumulation` 은 쓰지 않는다 — 간극에서 전 화면 1.000 으로 아무것도
     구분하지 못했다.

출력: 이탈 거리 대 깨진 화소 비율. "플레이 가능한 반경" 을 숫자로 준다.
"""
import argparse, json, os, sys, importlib
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
M = importlib.import_module("간극_채움_데이터셋")
V = importlib.import_module("간극_커버리지_비교")

HOME = Path(os.path.expanduser("~"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="AB_cull1")
    ap.add_argument("--runs", default=str(HOME/"data/runs"))
    ap.add_argument("--n", type=int, default=10, help="기준 포즈 수 (A·B 각각)")
    ap.add_argument("--offsets", type=float, nargs="+",
                    default=[0.0, 0.1, 0.2, 0.4, 0.8, 1.6])
    ap.add_argument("--near", type=float, default=1.0,
                    help="이보다 가까운 깊이는 표면이 아니라 floater 로 본다")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--out", default=str(HOME/"data/이탈내성.json"))
    a = ap.parse_args()

    W, H = a.width, a.height
    I = M.read_full(M.REG/"images.txt")
    W0, H0, fx, fy = M.cam_params(W, H)
    A, GT, B = M.split_frames(I)
    base = A[::max(1, len(A)//a.n)][:a.n] + B[::max(1, len(B)//a.n)][:a.n]
    print("기준 포즈 %d개 (A·B), 모델 %s" % (len(base), a.model))

    run = Path(a.runs)/a.model/"splatfacto/run1"
    mdl = V.Model(str(run))
    print("가우시안 %d\n" % mdl.n)

    # 기준 포즈에서의 '정상 깊이' 를 먼저 잡는다
    med0 = []
    for k in base:
        _rgb, _acc = mdl.render(I[k]["c2w"], W, H, fx, fy)
        med0.append(None)

    def broken(c2w):
        """깨진 화소 비율 + 깊이 중앙값."""
        from nerfstudio.cameras.cameras import Cameras, CameraType
        flip = np.diag([1., -1., -1., 1.])
        T = mdl.M @ (c2w @ flip); T[:3, 3] *= mdl.s
        cam = Cameras(camera_to_worlds=mdl.torch.tensor(T[:3, :4],
                      dtype=mdl.torch.float32)[None],
                      fx=fx, fy=fy, cx=W/2, cy=H/2, width=W, height=H,
                      camera_type=CameraType.PERSPECTIVE).to(mdl.dev)
        with mdl.torch.no_grad():
            o = mdl.model.get_outputs_for_camera(cam)
        d = np.squeeze(o["depth"].detach().cpu().numpy()) / mdl.s
        d = d[np.isfinite(d)]
        if d.size == 0:
            return 1.0, float("nan")
        return float((d < a.near).mean()), float(np.median(d))

    dirs = dict(옆=lambda R: R[:, 0], 위=lambda R: -R[:, 1], 뒤=lambda R: -R[:, 2])
    res = {}
    print("  %-5s %8s %10s %10s" % ("방향", "이탈거리", "깨진화소", "깊이중앙"))
    for dn, pick in dirs.items():
        row = []
        for off in a.offsets:
            br, md = [], []
            for k in base:
                c2w = I[k]["c2w"].copy()
                if off > 0:
                    c2w[:3, 3] = c2w[:3, 3] + pick(c2w[:3, :3]) * off
                b, m = broken(c2w)
                br.append(b); md.append(m)
            row.append(dict(offset=off, broken=float(np.mean(br)),
                            depth=float(np.nanmedian(md))))
            print("  %-5s %8.2f %9.1f%% %10.2f"
                  % (dn, off, 100*np.mean(br), np.nanmedian(md)), flush=True)
        res[dn] = row
        print()

    print("="*62)
    print("  이탈 거리별 '깨진 화소' 비율 (깊이 < %.1f 을 floater 로 봄)" % a.near)
    print("  %-8s %s" % ("거리", "  ".join("%6.2f" % o for o in a.offsets)))
    for dn in dirs:
        print("  %-8s %s" % (dn, "  ".join("%5.1f%%" % (100*r["broken"])
                                            for r in res[dn])))
    print("="*62)
    print("  참고: 복도 정상 깊이는 약 5 (COLMAP 단위). 간극 한가운데는 0.16 이었다.")
    json.dump(res, open(a.out, "w"), indent=1)
    print("  저장: %s" % a.out)


if __name__ == "__main__":
    main()
