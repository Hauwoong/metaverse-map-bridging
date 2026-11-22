#!/usr/bin/env bash
# ⑧ 제출 영상. G_M 과 G_AB 로 같은 경로를 날아 두 영상을 만든다.
#   G_M   다리 있음 → 끊김 없이 통과해야 한다
#   G_AB  다리 없음 → 같은 구간이 비어 있어야 한다
# 두 영상을 나란히 두는 것이 §4.7 정성 결과다.
set -u
source "$(dirname "$0")/환경.sh"
FT=~/data/플라이스루
LOG=~/data/flythrough.log; : > "$LOG"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "---- 경로 생성 ----"
"$HOME/venv/ltx/bin/python" ~/bin/플라이스루.py 2>&1 | tee -a "$LOG"

for M in M AB; do
  CFG=~/data/runs/G_$M/splatfacto/run1/config.yml
  [ -f "$CFG" ] || { say "G_$M 없음 — 건너뜀"; continue; }
  say "---- G_$M 렌더 ----"
  ns-render camera-path \
    --load-config "$CFG" \
    --camera-path-filename "$FT/경로_$M.json" \
    --output-path "$FT/플라이스루_$M.mp4" \
    --output-format video \
    >> "$LOG" 2>&1 < /dev/null
  if [ -f "$FT/플라이스루_$M.mp4" ]; then
    say "  $(du -h "$FT/플라이스루_$M.mp4" | cut -f1)  $(ffprobe -v error -show_entries format=duration -of csv=p=0 "$FT/플라이스루_$M.mp4" 2>/dev/null)초"
  else
    say "  🔴 렌더 실패"; tail -15 "$LOG"
  fi
done

# 나란히 비교 영상 (왼쪽 다리 있음, 오른쪽 없음)
if [ -f "$FT/플라이스루_M.mp4" ] && [ -f "$FT/플라이스루_AB.mp4" ]; then
  say "---- 나란히 비교 ----"
  ffmpeg -nostdin -v error -y -i "$FT/플라이스루_M.mp4" -i "$FT/플라이스루_AB.mp4" \
    -filter_complex "[0:v]drawtext=text='G_M (bridge)':x=10:y=10:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.5[l];[1:v]drawtext=text='G_AB (no bridge)':x=10:y=10:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.5[r];[l][r]hstack" \
    "$FT/비교.mp4" >> "$LOG" 2>&1
  [ -f "$FT/비교.mp4" ] && say "  비교.mp4 $(du -h "$FT/비교.mp4" | cut -f1)" || say "  🔴 비교 영상 실패 (drawtext 폰트 없을 수 있다)"
  if [ ! -f "$FT/비교.mp4" ]; then
    ffmpeg -nostdin -v error -y -i "$FT/플라이스루_M.mp4" -i "$FT/플라이스루_AB.mp4" \
      -filter_complex hstack "$FT/비교.mp4" >> "$LOG" 2>&1
    [ -f "$FT/비교.mp4" ] && say "  비교.mp4 (글자 없이) $(du -h "$FT/비교.mp4" | cut -f1)"
  fi
fi
say "끝"
