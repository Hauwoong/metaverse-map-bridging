#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""고정 등록된 생성 프레임의 이동 거리를 잰다.

🔴 앞선 측정과 결정적으로 다른 점:
   실촬영을 **고정**한 채 등록했으므로 생성 프레임이 **합본 좌표계에 직접** 앉아 있다.
   Sim(3) 도 스케일 추정도 필요 없다 — 좌표를 그냥 읽으면 된다.
   앞서 모든 오류의 근원(두 재구성을 사후에 맞추려 한 것)이 사라졌다.

같이 확인하는 것:
   실촬영 포즈가 정말 안 변했는가 (fix_existing_frames 가 동작했는가)
"""
import os, sys
import numpy as np

HOME = os.path.expanduser("~")
GAP = 15.032      # A 끝점 ↔ B 끝점 직선거리


def q2R(q):
    w, x, y, z = q
    return np.array([[1-2*y*y-2*z*z, 2*x*y-2*z*w,   2*x*z+2*y*w],
                     [2*x*y+2*z*w,   1-2*x*x-2*z*z, 2*y*z-2*x*w],
                     [2*x*z-2*y*w,   2*y*z+2*x*w,   1-2*x*x-2*y*y]])


def centers(p):
    out = {}
    lines = [l for l in open(p) if not l.startswith("#")]
    for i in range(0, len(lines), 2):
        t = lines[i].split()
        if len(t) < 10:
            continue
        R = q2R(list(map(float, t[1:5])))
        tv = np.array(list(map(float, t[5:8])))
        out[t[9]] = -R.T @ tv
    return out


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else HOME+"/data/fixreg2/chain6/out"
    C = centers(os.path.join(out, "images.txt"))
    M = centers(HOME+"/data/sfm_all/sparse/1/images.txt")

    gen = sorted(k for k in C if k.startswith("gen_"))
    real = [k for k in C if not k.startswith("gen_")]
    print("등록 결과: 실촬영 %d장, 생성 %d장" % (len(real), len(gen)))

    # 1) 실촬영이 정말 고정됐는가
    d = [np.linalg.norm(C[k]-M[k]) for k in real if k in M]
    print("  실촬영 포즈 변화: 최대 %.6f, 중앙값 %.6f  %s"
          % (np.max(d), np.median(d), "✅ 고정됨" if np.max(d) < 1e-6 else "⚠️ 움직임"))

    if len(gen) < 2:
        print("🔴 생성 프레임이 2장 미만 — 측정 불가")
        return

    P = np.array([C[k] for k in gen])
    path = float(sum(np.linalg.norm(P[i+1]-P[i]) for i in range(len(P)-1)))
    span = float(np.linalg.norm(P[-1]-P[0]))
    print()
    print("생성 구간 (합본 좌표계, 변환 없음)")
    print("  카메라 경로 길이 %.3f" % path)
    print("  양 끝 직선거리   %.3f" % span)
    print("  간극 %.3f 대비 : 경로 %.1f%% / 직선 %.1f%%" % (GAP, 100*path/GAP, 100*span/GAP))

    # 2) 실촬영과 같은 자로 속도 비교 — GT1 구간
    gt = sorted(k for k in M if k.startswith("v1791_") and 162 <= int(k[6:11]) <= 286)
    G = np.array([M[k] for k in gt])
    gpath = float(sum(np.linalg.norm(G[i+1]-G[i]) for i in range(len(G)-1)))
    print()
    print("실촬영 GT1 과 비교 (같은 좌표계, 같은 단위)")
    print("  실촬영 %d장 (%.1f초 @2.5fps) 경로 %.3f → 초당 %.4f"
          % (len(gt), len(gt)/2.5, gpath, gpath/(len(gt)/2.5)))
    # 생성: 원본 영상 프레임 간격 6, 24fps
    secs = len(gen) * 6 / 24.0
    print("  생성   %d장 (%.1f초 @24fps·샘플6) 경로 %.3f → 초당 %.4f"
          % (len(gen), secs, path, path/max(secs, 1e-9)))
    print("  → 생성이 실촬영의 %.1f%% 속도" % (100*(path/secs)/(gpath/(len(gt)/2.5))))

    # 3) 생성 구간이 A 에서 B 쪽으로 가고 있는가
    A = M.get("v1791_00143.png"); B = M.get("v1793_00001.png")
    if A is not None and B is not None:
        u = (B-A)/np.linalg.norm(B-A)
        prog = [(float(np.dot(C[k]-A, u))) for k in gen]
        print()
        print("A→B 축 투영 (0 이 A, %.3f 가 B)" % GAP)
        print("  시작 %.3f → 끝 %.3f,  순 전진 %.3f (%.1f%%)"
              % (prog[0], prog[-1], prog[-1]-prog[0], 100*(prog[-1]-prog[0])/GAP))
        print("  최대 도달 %.3f (%.1f%%)" % (max(prog), 100*max(prog)/GAP))
        # 단조 전진 비율 — 뒤로 가는 걸음이 많으면 궤적이 일관되지 않다
        fwd = sum(1 for i in range(len(prog)-1) if prog[i+1] > prog[i])
        print("  앞으로 간 걸음 %d / %d (%.0f%%)" % (fwd, len(prog)-1, 100*fwd/(len(prog)-1)))

    # 4) 경로가 왜 긴가 — 소수 이상치인가 전체 노이즈인가
    step = np.array([np.linalg.norm(P[i+1]-P[i]) for i in range(len(P)-1)])
    med = float(np.median(step))
    print()
    print("프레임 간 이동 분포 (경로가 직선의 %.1f배인 이유)" % (path/max(span,1e-9)))
    print("  중앙값 %.4f,  평균 %.4f,  최대 %.4f" % (med, step.mean(), step.max()))
    print("  상위 5개: %s" % np.round(np.sort(step)[-5:], 3))
    big = step > 5*med
    print("  중앙값의 5배 넘는 걸음 %d개가 전체 경로의 %.0f%% 를 차지"
          % (big.sum(), 100*step[big].sum()/max(step.sum(),1e-9)))
    if big.sum() <= 0.2*len(step) and step[big].sum() > 0.5*step.sum():
        print("  → 소수 이상치가 지배한다. 그 프레임들은 등록이 틀렸을 가능성이 크다.")
    else:
        print("  → 전체가 흔들린다. 궤적 자체가 일관되지 않다.")
    # 이상치를 뺀 경로
    keep = ~big
    rpath = float(step[keep].sum())
    print("  이상치 제외 경로 %.3f (간극의 %.0f%%)" % (rpath, 100*rpath/GAP))


if __name__ == "__main__":
    main()
