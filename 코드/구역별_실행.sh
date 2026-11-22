#!/usr/bin/env bash
# 플라이스루가 흐릿한 원인을 가린다.
#   A 꼬리(복도 쪽 경계)와 A 머리(안정 구간)에서 G_A·G_M·G_AB 를 각각 재고
#   같은 실사와 비교한다. 선명도 차이의 출처를 분리한다.
set -u
source "$(dirname "$0")/환경.sh"
Z=~/data/구역평가
LOG=~/data/구역평가.log; : > "$LOG"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

for ZONE in ahead atail; do
  say "======== $ZONE ========"
  "$HOME/venv/ltx/bin/python" ~/bin/구역별_품질.py --zone $ZONE 2>&1 | tee -a "$LOG"
  for M in A M AB; do
    CFG=~/data/runs/G_$M/splatfacto/run1/config.yml
    P="$Z/경로_${ZONE}_$M.json"
    [ -f "$CFG" ] && [ -f "$P" ] || continue
    rm -rf "$Z/프레임_${ZONE}_$M"
    ns-render camera-path --load-config "$CFG" --camera-path-filename "$P" \
      --output-path "$Z/프레임_${ZONE}_$M" --output-format images \
      >> "$LOG" 2>&1 < /dev/null
    "$HOME/venv/ltx/bin/python" ~/bin/병합_PSNR.py \
      --render "$Z/프레임_${ZONE}_$M" --gt ~/data/scenes/A/images_2 \
      --list "$Z/목록_$ZONE.txt" --tag "G_$M" 2>&1 | tee -a "$LOG"
  done
done
say "끝"
