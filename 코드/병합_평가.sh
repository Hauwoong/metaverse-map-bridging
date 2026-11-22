#!/usr/bin/env bash
# 표 3 (이음새 전후) 측정.
#   두 병합 맵을 **숨긴 GT1 의 실제 카메라 포즈**에서 렌더해 실사와 비교한다.
#     G_M   A + 생성다리 + B
#     G_AB  A + B 만 (대조군)
# GT1 은 두 맵 어느 쪽의 학습에도 들어가지 않았다 → 복원 측정이다.
set -u
source "$(dirname "$0")/환경.sh"
EV=~/data/평가
LOG=~/data/eval_merge.log; : > "$LOG"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "---- 카메라 경로 생성 ----"
"$HOME/venv/ltx/bin/python" ~/bin/병합_평가.py 2>&1 | tee -a "$LOG"

for M in M AB; do
  CFG=~/data/runs/G_$M/splatfacto/run1/config.yml
  if [ ! -f "$CFG" ]; then say "G_$M 없음 — 건너뜀"; continue; fi
  say "---- G_$M 렌더 ----"
  rm -rf "$EV/프레임_$M"
  ns-render camera-path \
    --load-config "$CFG" \
    --camera-path-filename "$EV/경로_$M.json" \
    --output-path "$EV/프레임_$M" \
    --output-format images \
    >> "$LOG" 2>&1 < /dev/null
  say "  렌더 $(ls "$EV/프레임_$M" 2>/dev/null | wc -l)장"
done

say "---- 표 3: 숨긴 GT1 대비 ----"
for M in M AB; do
  [ -d "$EV/프레임_$M" ] || continue
  "$HOME/venv/ltx/bin/python" ~/bin/병합_PSNR.py \
    --render "$EV/프레임_$M" --gt ~/data/scenes/GT1/images_2 \
    --list "$EV/목록.txt" --tag "G_$M" 2>&1 | tee -a "$LOG"
done
say "끝"
