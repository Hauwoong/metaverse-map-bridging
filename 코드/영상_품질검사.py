# -*- coding: utf-8 -*-
"""촬영 영상 품질 검사 — SfM/3DGS 에 쓸 수 있는지 프레임 단위로 본다.

측정:
  sharp  라플라시안 분산 (낮으면 모션블러 / 초점 나감)
  motion 연속 샘플 프레임 간 평균 절대차 (크면 급격한 움직임 = 흔들림 위험)
  clip   과노출(>=250) / 과소노출(<=5) 픽셀 비율

사용: python 코드/영상_품질검사.py <영상...> [--fps 5]
주의: 분석은 축소 프레임(480x270)에서 한다. 클리핑 비율은 면적 평균 때문에
      원본보다 과소 추정되므로 절대값이 아니라 구간 비교용으로 본다.
"""
import subprocess, sys, os, unicodedata
import numpy as np

W, H = 480, 270
FPS = 5.0

def decode(path, fps):
    cmd = ["ffmpeg", "-v", "error", "-i", path,
           "-vf", f"fps={fps},scale={W}:{H}", "-pix_fmt", "gray", "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    n = len(raw) // (W * H)
    if n == 0:
        return None
    return np.frombuffer(raw[:n * W * H], dtype=np.uint8).reshape(n, H, W)

def analyse(path, fps):
    name = unicodedata.normalize("NFC", os.path.basename(path))
    fr8 = decode(path, fps)
    if fr8 is None:
        print(f"!! 디코드 실패: {name}"); return
    fr = fr8.astype(np.float32)
    n = len(fr)

    lap = (-4 * fr[:, 1:-1, 1:-1] + fr[:, :-2, 1:-1] + fr[:, 2:, 1:-1]
           + fr[:, 1:-1, :-2] + fr[:, 1:-1, 2:])
    sharp = lap.var(axis=(1, 2))
    motion = np.concatenate([[0.0], np.abs(np.diff(fr, axis=0)).mean(axis=(1, 2))])
    hi = (fr8 >= 250).mean(axis=(1, 2))
    lo = (fr8 <= 5).mean(axis=(1, 2))
    t = np.arange(n) / fps

    print(f"\n{'='*74}\n{name}   {n}개 샘플 @ {fps}fps  ({t[-1]:.1f}s)\n{'='*74}")
    print(f"  sharp   중앙 {np.median(sharp):7.0f}   10%분위 {np.percentile(sharp,10):7.0f}"
          f"   최소 {sharp.min():7.0f}")
    print(f"  motion  중앙 {np.median(motion):7.2f}   90%분위 {np.percentile(motion,90):7.2f}"
          f"   최대 {motion.max():7.2f}")
    print(f"  과노출  평균 {hi.mean()*100:5.2f}%   최대 {hi.max()*100:5.2f}%"
          f"      과소노출 평균 {lo.mean()*100:5.2f}%  최대 {lo.max()*100:5.2f}%")

    # 나쁜 구간: 선명도가 하위 10% 미만이거나 움직임이 상위 5% 초과
    s_thr = np.percentile(sharp, 10)
    m_thr = np.percentile(motion, 95)
    bad = (sharp < s_thr) | (motion > m_thr)

    runs, i = [], 0
    while i < n:
        if bad[i]:
            j = i
            while j + 1 < n and bad[j + 1]:
                j += 1
            if (j - i + 1) >= max(2, int(fps * 0.4)):   # 0.4초 이상 이어질 때만
                runs.append((i, j))
            i = j + 1
        else:
            i += 1

    print(f"  --- 제외 후보 구간 (선명도<{s_thr:.0f} 또는 움직임>{m_thr:.2f}, 0.4s 이상) ---")
    if not runs:
        print("      없음")
    else:
        tot = 0
        for a, b in runs:
            tot += (b - a + 1) / fps
            print(f"      {t[a]:6.1f}s ~ {t[b]:6.1f}s  ({(b-a+1)/fps:4.1f}s)"
                  f"   sharp {sharp[a:b+1].mean():6.0f}   motion {motion[a:b+1].mean():5.2f}")
        print(f"      합계 {tot:.1f}s / {t[-1]:.1f}s  ({tot/t[-1]*100:.1f}%)")

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    fps = FPS
    if "--fps" in sys.argv:
        fps = float(sys.argv[sys.argv.index("--fps") + 1])
    for p in args:
        analyse(p, fps)
