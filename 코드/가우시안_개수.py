#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""체크포인트에서 가우시안 개수를 센다 (표 4).

🔴 torch.load 에 weights_only=False 가 필요하다. nerfstudio 체크포인트에는
   텐서 외의 객체가 들어 있어 기본값(True)으로는 못 읽는다.
   코드/nerfstudio_패치.sh 가 nerfstudio 쪽에 같은 패치를 한다.
"""
import argparse, os, glob
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="예: A B GT1 C M AB")
    ap.add_argument("--runs", default=os.path.expanduser("~/data/runs"))
    a = ap.parse_args()
    print("%-6s %12s %10s" % ("맵", "가우시안", "ckpt"))
    print("-" * 32)
    for m in a.models:
        pats = sorted(glob.glob(os.path.join(
            a.runs, "G_%s" % m, "splatfacto", "run1",
            "nerfstudio_models", "*.ckpt")))
        if not pats:
            print("%-6s %12s" % (m, "없음")); continue
        p = pats[-1]
        d = torch.load(p, map_location="cpu", weights_only=False)
        st = d.get("pipeline", d)
        n = None
        for k, v in st.items():
            if k.endswith("means") and hasattr(v, "shape") and v.ndim == 2:
                n = v.shape[0]; break
        sz = os.path.getsize(p) / 1e6
        print("%-6s %12s %9.0fMB" % (m, "%d" % n if n else "?", sz))

if __name__ == "__main__":
    main()
