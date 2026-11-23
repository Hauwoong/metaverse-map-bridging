#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""COLMAP 이 만든 부분 모델들의 구성을 본다.

🔴 왜 중요한가:
   GT 없는 등록에서 가장 중요한 질문은 PSNR 이 아니라
   **"A 와 B 가 한 모델 안에 들어왔는가"** 다.
   갈라졌다면 생성 다리가 GT1 없이는 두 장면을 잇지 못한 것이고,
   그 자체가 결과다. 가장 큰 모델만 보면 이 사실을 놓친다.

출력: 모델마다 A / GT1 / B / 생성 프레임이 몇 장씩 들어갔는지.
"""
import argparse
import glob
import os
import re

HOME = os.path.expanduser("~")
PAT = re.compile(r"(v1791_\d+|v1793_\d+|gen_\d+)\.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="sparse 폴더 (하위에 0,1,2...)")
    a = ap.parse_args()

    models = sorted(glob.glob(os.path.join(a.dir, "*")))
    if not models:
        print("부분 모델이 없다"); return

    print("%-6s %7s %7s %7s %7s   %s" % ("모델", "A", "GT1", "B", "생성", "판정"))
    print("-" * 62)
    rows = []
    for m in models:
        f = os.path.join(m, "images.txt")
        if not os.path.exists(f):
            continue
        names = set()
        for l in open(f):
            if l.startswith("#"):
                continue
            g = PAT.search(l)
            if g:
                names.add(g.group(0))
        A = sum(1 for n in names if n.startswith("v1791_")
                and int(n[6:11]) <= 143)
        GT = sum(1 for n in names if n.startswith("v1791_")
                 and 162 <= int(n[6:11]) <= 286)
        B = sum(1 for n in names if n.startswith("v1793_"))
        G = sum(1 for n in names if n.startswith("gen_"))
        if A >= 10 and B >= 10:
            verdict = "✅ A·B 가 한 모델에 있다"
        elif A >= 10:
            verdict = "A 쪽만"
        elif B >= 10:
            verdict = "B 쪽만"
        else:
            verdict = "조각"
        rows.append((os.path.basename(m), A, GT, B, G, verdict))
        print("%-6s %7d %7d %7d %7d   %s"
              % (os.path.basename(m), A, GT, B, G, verdict))

    print()
    joined = [r for r in rows if r[1] >= 10 and r[3] >= 10]
    if joined:
        r = max(joined, key=lambda x: x[4])
        print("📌 **A 와 B 가 한 모델(%s)에 들어왔다.** 생성 %d장이 함께 있다."
              % (r[0], r[4]))
        print("   → 생성 다리가 GT1 없이 두 장면을 이었다는 뜻이다.")
    else:
        tot = sum(r[4] for r in rows)
        print("🔴 **A 와 B 가 한 모델에 들어오지 못했다.**")
        print("   생성 프레임은 총 %d장 등록됐지만 두 장면을 잇지는 못했다." % tot)
        print("   → GT1 없이는 다리가 성립하지 않는다는 결과다. 그대로 보고한다.")
        print("   (지금까지의 '복도 위 79%' 는 GT1 과의 매칭이 있어서 가능했다)")
    if any(r[2] > 0 for r in rows):
        print("   ⚠️ GT1 이 들어간 모델이 있다 — 입력 구성이 잘못됐다. 확인할 것.")


if __name__ == "__main__":
    main()
