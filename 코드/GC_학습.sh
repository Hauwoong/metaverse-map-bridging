#!/usr/bin/env bash
# 연결 구간 가우시안 G_C 학습.
# 데이터셋은 코드/GC_데이터셋.py 가 **합본 좌표계로** 만들어 둔 것이다 —
# G_A·G_B 와 좌표계를 공유해야 ⑦ 이음새 최적화와 ⑧ 병합이 가능하다.
#
# 🔴 setsid 로 띄울 것 (코드/3dgs_학습.sh 주석 참조).
# 생성 프레임은 1024x576 으로 이미 작다 → downscale-factor 1.
set -u
source "$(dirname "$0")/환경.sh"
SC=~/data/scenes/C
RUN=~/data/runs
LOG=~/data/GC_학습.log
mkdir -p "$RUN"
N=$(ls "$SC/images" | wc -l)
echo "[$(date +%H:%M:%S)] G_C 학습 시작 — $N장 (1024x576)" | tee -a "$LOG"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | tee -a "$LOG"
rm -rf "$RUN/G_C"
ns-train splatfacto \
  --data "$SC" --output-dir "$RUN" --experiment-name G_C --timestamp run1 \
  --vis tensorboard --max-num-iterations 30000 --steps-per-save 5000 \
  --steps-per-eval-all-images 30000 --steps-per-eval-image 5000 \
  --logging.local-writer.enable False \
  colmap --downscale-factor 1 >> "$LOG" 2>&1 < /dev/null
RC=$?
echo "[$(date +%H:%M:%S)] 종료 코드 $RC" | tee -a "$LOG"
ls -l "$RUN/G_C/splatfacto/run1/nerfstudio_models/" 2>/dev/null | tee -a "$LOG"
