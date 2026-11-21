#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""연결 구간 가우시안 G_C 를 학습할 데이터셋을 **합본 좌표계로** 만든다.

왜 합본 좌표계인가:
  G_A·G_B 와 같은 좌표계여야 ⑦ 이음새 최적화와 ⑧ 병합이 가능하다.
  다리 모델(실촬영 경계 + 생성)에는 실촬영이 함께 있고 그 합본 위치를 알고 있으므로,
  Sim(3) 를 구해 생성 프레임의 포즈와 3D 점을 합본 좌표계로 옮긴다.
  **스케일을 추정하는 것이 아니라 실촬영이 고정한다.**

출력: nerfstudio 가 읽는 형식
  <out>/images/              생성 프레임
  <out>/colmap/sparse/0/     cameras.txt, images.txt, points3D.txt  (합본 좌표계)
"""
import argparse, os, glob, shutil
from collections import defaultdict
import numpy as np

HOME = os.path.expanduser("~")


def q2R(q):
    w, x, y, z = q
    return np.array([[1-2*y*y-2*z*z, 2*x*y-2*z*w,   2*x*z+2*y*w],
                     [2*x*y+2*z*w,   1-2*x*x-2*z*z, 2*y*z-2*x*w],
                     [2*x*z-2*y*w,   2*y*z+2*x*w,   1-2*x*x-2*y*y]])


def R2q(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t+1.0)*2
        return np.array([0.25*s, (R[2,1]-R[1,2])/s, (R[0,2]-R[2,0])/s, (R[1,0]-R[0,1])/s])
    i = int(np.argmax([R[0,0], R[1,1], R[2,2]]))
    if i == 0:
        s = np.sqrt(1.0+R[0,0]-R[1,1]-R[2,2])*2
        return np.array([(R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s])
    if i == 1:
        s = np.sqrt(1.0+R[1,1]-R[0,0]-R[2,2])*2
        return np.array([(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s])
    s = np.sqrt(1.0+R[2,2]-R[0,0]-R[1,1])*2
    return np.array([(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s])


def read_images(p):
    """{name: (image_id, camera_id, R, t, [(x,y,point3D_id), ...])}"""
    out = {}
    lines = [l for l in open(p) if not l.startswith("#")]
    for i in range(0, len(lines), 2):
        t = lines[i].split()
        if len(t) < 10:
            continue
        pts = []
        if i+1 < len(lines):
            v = lines[i+1].split()
            for k in range(0, len(v)-2, 3):
                pid = int(v[k+2])
                if pid != -1:
                    pts.append((float(v[k]), float(v[k+1]), pid))
        out[t[9]] = (int(t[0]), int(t[8]),
                     q2R(list(map(float, t[1:5]))),
                     np.array(list(map(float, t[5:8]))), pts)
    return out


def centers(imgs):
    return {k: -v[2].T @ v[3] for k, v in imgs.items()}


def umeyama(src, dst):
    """카메라 중심만으로 Sim(3). ⚠️ 궤적이 직선에 가까우면 그 축 둘레 회전이 퇴화한다."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src-mu_s, dst-mu_d
    C = D.T @ S / len(src)
    U, d, Vt = np.linalg.svd(C)
    E = np.eye(3)
    if np.linalg.det(U)*np.linalg.det(Vt) < 0:
        E[2, 2] = -1
    R = U @ E @ Vt
    s = float(np.trace(np.diag(d) @ E) / ((S**2).sum()/len(src)))
    return s, R, mu_d - s*R @ mu_s


def sim3_with_rotations(Cs, Ct, Rs, Rt):
    """카메라 중심 **과 자세**로 Sim(3) 를 구한다.

    🔴 왜 자세까지 쓰는가 — 중심만 쓰면 퇴화한다 (보고서 §3.4).
       정렬에 쓸 실촬영이 복도를 **거의 직선으로** 걸어간 궤적이면, 그 직선을
       축으로 한 회전이 중심만으로는 결정되지 않는다. 중심은 잘 맞으면서
       회전이 128° 틀어지는 일이 실제로 일어났다.

       회전은 자세에서 구한다:  R = proj_SO(3)( Σ_k R_k^tgt (R_k^src)^T )
       (R_k 는 camera-to-world 회전. 각 프레임이 제약을 하나씩 준다.)
       스케일·평행이동은 중심에서 구한다 — 그쪽은 퇴화하지 않는다.
    """
    A = np.zeros((3, 3))
    for rs, rt in zip(Rs, Rt):
        A += rt @ rs.T
    U, _, Vt = np.linalg.svd(A)
    E = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        E[2, 2] = -1                      # 반사 보정
    R = U @ E @ Vt
    mu_s, mu_t = Cs.mean(0), Ct.mean(0)
    S, T = Cs - mu_s, Ct - mu_t
    num = float(np.sum(T * (R @ S.T).T))
    den = float(np.sum(S * S))
    s = num / max(den, 1e-12)
    return s, R, mu_t - s * (R @ mu_s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridge", default=str(HOME)+"/data/bridge_reg/chain6")
    ap.add_argument("--merged", default=str(HOME)+"/data/sfm_all/sparse/1")
    ap.add_argument("--out", default=str(HOME)+"/data/scenes/C")
    a = ap.parse_args()

    # 실촬영과 생성이 함께 든 가장 큰 모델을 고른다
    best, bestn = None, 0
    for m in sorted(glob.glob(os.path.join(a.bridge, "sparse", "*"))):
        f = os.path.join(m, "images.txt")
        if not os.path.exists(f):
            continue
        I = read_images(f)
        if sum(1 for k in I if k.startswith("gen_")) and \
           sum(1 for k in I if k.startswith("v17")) and len(I) > bestn:
            best, bestn = m, len(I)
    assert best, "실촬영과 생성이 함께 든 모델이 없다"
    print("모델: %s (%d장)" % (best, bestn))

    I = read_images(os.path.join(best, "images.txt"))
    M = read_images(os.path.join(a.merged, "images.txt"))
    Cb, Cm = centers(I), centers(M)
    real = sorted(k for k in I if k.startswith("v17") and k in Cm)
    gen = sorted(k for k in I if k.startswith("gen_"))
    print("실촬영 %d장으로 Sim(3), 생성 %d장 변환" % (len(real), len(gen)))

    Cs = np.array([Cb[k] for k in real])
    Ct = np.array([Cm[k] for k in real])
    # camera-to-world 회전 = (world-to-camera)^T
    Rs = [I[k][2].T for k in real]
    Rt = [M[k][2].T for k in real]
    s, R, t = sim3_with_rotations(Cs, Ct, Rs, Rt)
    res = np.linalg.norm((s*(R @ Cs.T).T + t) - Ct, axis=1)
    ang = [np.degrees(np.arccos(np.clip((np.trace((R @ rs).T @ rt)-1)/2, -1, 1)))
           for rs, rt in zip(Rs, Rt)]
    print("  스케일 %.5f" % s)
    print("  중심 잔차 중앙값 %.4f, 회전 잔차 중앙값 %.3f°" % (np.median(res), np.median(ang)))
    if np.median(ang) > 5:
        print("  🔴 회전 잔차가 크다 — 정렬이 틀렸다. 이대로 쓰면 안 된다.")

    # 3D 점도 같은 변환으로 옮긴다
    P = {}
    for l in open(os.path.join(best, "points3D.txt")):
        if l.startswith("#"):
            continue
        v = l.split()
        if len(v) < 8:
            continue
        P[int(v[0])] = (np.array(list(map(float, v[1:4]))),
                        tuple(v[4:7]), float(v[7]))
    keep = set()
    for k in gen:
        for _, _, pid in I[k][4]:
            keep.add(pid)
    keep &= set(P)
    print("  생성 프레임이 보는 3D 점 %d개" % len(keep))

    out = a.out
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    sp = os.path.join(out, "colmap", "sparse", "0")
    os.makedirs(sp, exist_ok=True)

    shutil.copy(os.path.join(a.bridge, "images", gen[0]).replace(gen[0], "")
                .rstrip("/") + "/" + gen[0], os.path.join(out, "images", gen[0]))
    for k in gen[1:]:
        shutil.copy(os.path.join(a.bridge, "images", k), os.path.join(out, "images", k))

    # cameras.txt — 생성 프레임이 쓰는 카메라만
    cam_ids = {I[k][1] for k in gen}
    with open(os.path.join(sp, "cameras.txt"), "w") as f:
        f.write("# Camera list\n")
        for l in open(os.path.join(best, "cameras.txt")):
            if l.startswith("#"):
                continue
            if int(l.split()[0]) in cam_ids:
                f.write(l)

    newid = {k: i+1 for i, k in enumerate(gen)}
    pid_new = {pid: i+1 for i, pid in enumerate(sorted(keep))}

    with open(os.path.join(sp, "images.txt"), "w") as f:
        f.write("# Image list\n")
        for k in gen:
            _, cid, Rc, tc, pts = I[k]
            # 다리 좌표계 → 합본 좌표계.  c2w 를 옮긴 뒤 w2c 로 되돌린다.
            Cc = -Rc.T @ tc
            Cn = s*(R @ Cc) + t
            Rn = (R @ Rc.T).T          # c2w 회전 R@Rc.T 의 역
            q = R2q(Rn)
            tn = -Rn @ Cn
            f.write("%d %.9f %.9f %.9f %.9f %.9f %.9f %.9f %d %s\n"
                    % (newid[k], q[0], q[1], q[2], q[3], tn[0], tn[1], tn[2], cid, k))
            f.write(" ".join("%.3f %.3f %d" % (x, y, pid_new.get(pid, -1))
                             for x, y, pid in pts) + "\n")

    tracks = defaultdict(list)
    for k in gen:
        for idx, (_, _, pid) in enumerate(I[k][4]):
            if pid in pid_new:
                tracks[pid].append((newid[k], idx))
    with open(os.path.join(sp, "points3D.txt"), "w") as f:
        f.write("# 3D point list\n")
        n = 0
        for pid in sorted(keep):
            if len(tracks[pid]) < 2:
                continue
            xyz, rgb, err = P[pid]
            X = s*(R @ xyz) + t
            f.write("%d %.9f %.9f %.9f %s %s %s %.6f %s\n"
                    % (pid_new[pid], X[0], X[1], X[2], rgb[0], rgb[1], rgb[2], err,
                       " ".join("%d %d" % tk for tk in tracks[pid])))
            n += 1
    print("  기록: 이미지 %d장, 점 %d개 → %s" % (len(gen), n, out))


if __name__ == "__main__":
    main()
