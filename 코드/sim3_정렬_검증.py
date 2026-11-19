"""Sim(3) 정렬: 중심만 쓰면 roll 1자유도가 남는다는 것을 수치로 확인."""
import numpy as np
np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(0)

def rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1.]])

def rand_R(g):
    Q, R = np.linalg.qr(g.standard_normal((3,3)))
    return Q * np.sign(np.diag(R)) @ np.diag([1,1,np.linalg.det(Q*np.sign(np.diag(R)))])

def umeyama(X, Y):
    """X,Y: (n,3). returns s,R,t minimizing |Y - (sRX+t)|."""
    n = len(X); mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X-mx, Y-my
    S = Yc.T @ Xc / n
    U, D, Vt = np.linalg.svd(S)
    Sm = np.diag([1,1,np.sign(np.linalg.det(U@Vt))])
    R = U @ Sm @ Vt
    s = np.trace(np.diag(D)@Sm) / (np.sum(Xc**2)/n)
    return s, R, my - s*R@mx

def pose_align(Rs_src, Rs_tgt, C_src, C_tgt):
    """회전까지 쓰는 정렬."""
    M = sum(Rt @ Rs.T for Rs, Rt in zip(Rs_src, Rs_tgt))
    U, _, Vt = np.linalg.svd(M)
    R = U @ np.diag([1,1,np.sign(np.linalg.det(U@Vt))]) @ Vt
    mx, my = C_src.mean(0), C_tgt.mean(0)
    Xc, Yc = C_src-mx, C_tgt-my
    s = np.sum(Yc * (Xc @ R.T)) / np.sum(Xc**2)
    return s, R, my - s*R@mx

# ── 진짜 변환 ──────────────────────────────────────────
s_gt, R_gt, t_gt = 2.5, rand_R(rng), np.array([1.,-2.,3.])

# src 포즈: 복도를 "직진"하며 찍음 → 카메라 중심이 한 직선 위 (퇴화 조건)
k = 6
C_src  = np.stack([np.array([0,0,1.])*u for u in np.linspace(0, 4, k)])
Rs_src = [rand_R(rng) for _ in range(k)]
C_tgt  = (s_gt * (R_gt @ C_src.T)).T + t_gt
Rs_tgt = [R_gt @ R for R in Rs_src]

print("=== 설정: 카메라 중심 %d개가 한 직선 위 (복도 직진) ===" % k)

# [A] 중심만 쓰는 Umeyama
s_a, R_a, t_a = umeyama(C_src, C_tgt)
err_c = np.abs((s_a*(R_a@C_src.T)).T + t_a - C_tgt).max()
err_R = np.rad2deg(np.arccos(np.clip((np.trace(R_a.T@R_gt)-1)/2, -1, 1)))
print("[A] 중심만 (Umeyama)")
print("    중심 재현 오차 : %.2e   <- 중심은 완벽히 맞는다" % err_c)
print("    회전 오차      : %7.2f deg  <- 그런데 회전이 틀렸다" % err_R)

# roll 모호성 증명: 직선축 주위로 임의 회전시켜도 중심 비용이 동일한가?
axis = R_gt @ np.array([0,0,1.])          # 전역 좌표계에서의 복도 축
def rot_about(axis, a):
    ax = axis/np.linalg.norm(axis); K = np.array(
        [[0,-ax[2],ax[1]],[ax[2],0,-ax[0]],[-ax[1],ax[0],0]])
    return np.eye(3) + np.sin(a)*K + (1-np.cos(a))*K@K
print("    축 주위로 돌려도 중심 비용이 그대로인가:")
for deg in (0, 30, 90, 180):
    Rr = rot_about(axis, np.deg2rad(deg)) @ R_a
    tr = C_tgt.mean(0) - s_a*Rr@C_src.mean(0)
    cost = np.sum(((s_a*(Rr@C_src.T)).T + tr - C_tgt)**2)
    print("        roll %3d deg -> 비용 %.3e" % (deg, cost))

# [B] 회전까지 쓰는 포즈 기반 정렬 (포즈 2개만)
s_b, R_b, t_b = pose_align(Rs_src[:2], Rs_tgt[:2], C_src[:2], C_tgt[:2])
print("[B] 포즈 기반 (중심+회전, 포즈 2개만)")
print("    회전 오차 : %.2e deg" % np.rad2deg(np.arccos(np.clip((np.trace(R_b.T@R_gt)-1)/2,-1,1))))
print("    스케일    : %.6f  (참값 %.6f)" % (s_b, s_gt))
print("    이동 오차 : %.2e" % np.abs(t_b-t_gt).max())

# [C] 참고: 중심이 직선이 아니면 중심만으로도 풀린다
C_src2 = rng.standard_normal((k,3))
C_tgt2 = (s_gt*(R_gt@C_src2.T)).T + t_gt
_, R_c, _ = umeyama(C_src2, C_tgt2)
print("[C] 대조군: 중심이 일반 배치(비동일선상)면 중심만으로도 OK")
print("    회전 오차 : %.2e deg" % np.rad2deg(np.arccos(np.clip((np.trace(R_c.T@R_gt)-1)/2,-1,1))))
