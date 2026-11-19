#!/usr/bin/env bash
# 케이스 1 (선택지 ⓐ) 의 세 장면을 splatfacto 로 학습한다.
#   G_A    끝점 장면 A (1791 @0-57s)
#   G_GT1  숨길 중간 구간 정답 (1791 @64-114s) — 평가에만 쓴다
#   G_B    끝점 장면 B (1793)
# 데이터셋은 코드/구간_분할.py --write 로 만든다.
set -u
# 환경은 코드/환경.sh 한 곳에만 둔다 (conda activate 를 쓰지 않는 이유, MAX_JOBS,
# TORCH_CUDA_ARCH_LIST, CPATH 가 왜 그 값인지는 그 파일의 주석에 적혀 있다).
source "$(dirname "$0")/환경.sh"

# 해상도는 960x540 (--downscale-factor 2).
# 1920x1080 원본으로 돌렸더니 nerfstudio 가 전 프레임을 올리면서 WSL 이 메모리
# 상한을 쳐서 VM 전체가 멎었다. 960x540 은 nerfstudio 가 1920px 입력에 대해
# 스스로 고르는 기본값이기도 하다.

SC=~/data/scenes
RUN=~/data/runs
LOG=~/data/train.log
mkdir -p "$RUN"
ts() { date +%H:%M:%S; }

echo "[$(ts)] ================ 3DGS 학습 시작 ================" | tee -a "$LOG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$LOG"

# 960x540 축소본을 미리 만든다.
# 없으면 nerfstudio 가 "지금 축소할까요?" 를 대화형으로 묻는데, 배치 실행에는
# stdin 이 없어 EOFError 로 죽는다 (rich.prompt.Confirm.ask).
for S in A GT1 B; do
  SRC="$SC/$S/images"; DST="$SC/$S/images_2"
  NEED=$(ls "$SRC" | wc -l)
  HAVE=$(ls "$DST" 2>/dev/null | wc -l)
  if [ "$HAVE" -ne "$NEED" ]; then
    echo "[$(ts)] $S 축소본 생성 ($NEED장 -> 960x540)" | tee -a "$LOG"
    mkdir -p "$DST"
    for f in "$SRC"/*.png; do
      ffmpeg -v error -y -i "$f" -vf scale=960:540 "$DST/$(basename "$f")"
    done
  fi
  echo "[$(ts)] $S 축소본 $(ls "$DST" | wc -l)장 준비됨" | tee -a "$LOG"
done

for S in A GT1 B; do
  N=$(ls "$SC/$S/images" | wc -l)
  echo "" | tee -a "$LOG"
  echo "[$(ts)] ---- G_$S  (프레임 $N, 960x540) ----" | tee -a "$LOG"
  T0=$SECONDS
  ns-train splatfacto \
    --data "$SC/$S" \
    --output-dir "$RUN" \
    --experiment-name "G_$S" \
    --timestamp run1 \
    --vis tensorboard \
    --max-num-iterations 30000 \
    --steps-per-eval-all-images 30000 \
    --steps-per-eval-image 2000 \
    colmap --downscale-factor 2 \
    >> "$LOG" 2>&1 < /dev/null
  RC=$?
  echo "[$(ts)] G_$S 종료 코드 $RC, $((SECONDS-T0))초" | tee -a "$LOG"
  if [ $RC -ne 0 ]; then
    echo "!! G_$S 실패 — 로그 끝부분:" | tee -a "$LOG"
    tail -25 "$LOG"
    exit $RC
  fi
done

echo "" | tee -a "$LOG"
echo "[$(ts)] ================ 결과 ================" | tee -a "$LOG"
for S in A GT1 B; do
  D="$RUN/G_$S/splatfacto/run1"
  CK=$(ls "$D/nerfstudio_models/"*.ckpt 2>/dev/null | tail -1)
  echo "  G_$S  config: $D/config.yml" | tee -a "$LOG"
  echo "        ckpt  : $CK" | tee -a "$LOG"
done
echo "[$(ts)] 끝" | tee -a "$LOG"
