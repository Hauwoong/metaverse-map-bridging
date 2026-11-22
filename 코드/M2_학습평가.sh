#!/usr/bin/env bash
# G_M2 (여러 배치의 생성 프레임 합집합) 학습 + 평가.
# 커버리지 39% -> 48%, 감독 프레임 62 -> 141장.
set -u
source "$(dirname "$0")/환경.sh"
SC=~/data/scenes
RUN=~/data/runs
LOG=~/data/m2.log; : > "$LOG"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

S=M2
SRC="$SC/$S/images"; DST="$SC/$S/images_2"
NEED=$(ls "$SRC" | wc -l)
if [ "$(ls "$DST" 2>/dev/null | wc -l)" -ne "$NEED" ]; then
  say "$S 축소본 생성 ($NEED장 -> 960x540)"
  mkdir -p "$DST"
  for f in "$SRC"/*.png; do
    ffmpeg -nostdin -v error -y -i "$f" -vf scale=960:540 "$DST/$(basename "$f")"
  done
fi
say "$S 축소본 $(ls "$DST" | wc -l)장"

OUT="$RUN/G_$S/splatfacto/run1"
rm -rf "$OUT"
say "---- G_$S 학습 ($NEED장) ----"
T0=$SECONDS
ns-train splatfacto --data "$SC/$S" --output-dir "$RUN" \
  --experiment-name "G_$S" --timestamp run1 --vis tensorboard \
  --max-num-iterations 30000 --steps-per-save 10000 \
  --steps-per-eval-all-images 30000 --steps-per-eval-image 10000 \
  --logging.local-writer.enable False \
  colmap --downscale-factor 2 >> "$LOG" 2>&1 < /dev/null
say "G_$S 종료 코드 $?, $((SECONDS-T0))초"

say "---- 표 3 재측정 (M2 포함) ----"
"$HOME/venv/ltx/bin/python" ~/bin/병합_평가.py --models M2,M,AB 2>&1 | tee -a "$LOG"
EV=~/data/평가
for M in M2 M AB; do
  CFG=~/data/runs/G_$M/splatfacto/run1/config.yml
  [ -f "$CFG" ] || continue
  rm -rf "$EV/프레임_$M"
  ns-render camera-path --load-config "$CFG" \
    --camera-path-filename "$EV/경로_$M.json" \
    --output-path "$EV/프레임_$M" --output-format images \
    >> "$LOG" 2>&1 < /dev/null
done
for M in M2 M AB; do
  [ -d "$EV/프레임_$M" ] || continue
  "$HOME/venv/ltx/bin/python" ~/bin/병합_PSNR.py \
    --render "$EV/프레임_$M" --gt ~/data/scenes/GT1/images_2 \
    --list "$EV/목록.txt" --tag "G_$M" 2>&1 | tee -a "$LOG"
done
say "---- 가우시안 개수 ----"
"$HOME/venv/ltx/bin/python" ~/bin/가우시안_개수.py M2 M AB 2>&1 | tee -a "$LOG"
say "끝"
