#!/usr/bin/env bash
# 병합 맵을 학습한다 — ⑦ 이음새 최적화와 ⑧ 병합을 한 번에.
#   G_M   A + 생성다리 + B   (362장)
#   G_AB  A + B 만           (300장)  ← 대조군: 이음새가 비어 있는 맵
#
# 🔴 실행법은 3dgs_학습.sh 와 같다 (setsid nohup, /mnt/c 에서 직접 돌리지 않는다).
#    이유: 호스트 메모리가 빡빡하면 윈도 쪽 wsl.exe 가 죽임을 당하면서 WSL
#    세션 프로세스 트리가 통째로 끌려간다. Linux 쪽에 흔적이 남지 않는다.
#    이어받기(--load-dir)는 쓰지 않는다 — gsplat 밀도화 상태가 복원되지 않아
#    복구 후 첫 밀도화에서 device-side assert 로 죽는다.
set -u
source "$(dirname "$0")/환경.sh"

SC=~/data/scenes
RUN=~/data/runs
LOG=~/data/train_merge.log
TRIES=2
mkdir -p "$RUN"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "================ 병합 맵 학습 시작 ================"

for S in M AB; do
  SRC="$SC/$S/images"; DST="$SC/$S/images_2"
  NEED=$(ls "$SRC" | wc -l)
  if [ "$(ls "$DST" 2>/dev/null | wc -l)" -ne "$NEED" ]; then
    say "$S 축소본 생성 ($NEED장 -> 960x540)"
    mkdir -p "$DST"
    for f in "$SRC"/*.png; do
      ffmpeg -nostdin -v error -y -i "$f" -vf scale=960:540 "$DST/$(basename "$f")"
    done
  fi
  say "$S 축소본 $(ls "$DST" | wc -l)장 준비됨"
done

for S in M AB; do
  N=$(ls "$SC/$S/images" | wc -l)
  OUT="$RUN/G_$S/splatfacto/run1"
  echo "" | tee -a "$LOG"
  say "---- G_$S  (프레임 $N, 960x540) ----"
  T0=$SECONDS
  RC=1
  for A in $(seq 1 $TRIES); do
    rm -rf "$OUT"
    say "  시도 $A: 처음부터"
    ns-train splatfacto \
      --data "$SC/$S" \
      --output-dir "$RUN" \
      --experiment-name "G_$S" \
      --timestamp run1 \
      --vis tensorboard \
      --max-num-iterations 30000 \
      --steps-per-save 10000 \
      --steps-per-eval-all-images 30000 \
      --steps-per-eval-image 10000 \
      --logging.local-writer.enable False \
      colmap --downscale-factor 2 \
      >> "$LOG" 2>&1 < /dev/null
    RC=$?
    [ $RC -eq 0 ] && break
    say "  !! 시도 $A 종료 코드 $RC"
    sleep 5
  done
  say "G_$S 종료 코드 $RC, $((SECONDS-T0))초"
  [ $RC -ne 0 ] && { say "!! G_$S 실패"; tail -25 "$LOG"; }
done

echo "" | tee -a "$LOG"
say "================ 결과 ================"
for S in M AB; do
  D="$RUN/G_$S/splatfacto/run1"
  say "  G_$S  $(ls "$D/nerfstudio_models/"*.ckpt 2>/dev/null | sort | tail -1)"
done
say "끝"
