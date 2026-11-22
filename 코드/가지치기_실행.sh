#!/usr/bin/env bash
# floater 가지치기 + 재평가.
# 학습 카메라가 아무도 보지 않는 가우시안을 제거하고 같은 측정을 다시 한다.
# 🔴 PSNR 이 오를지 내릴지는 모른다 — 쓰레기 대신 빈 배경이 되는데
#    실사와 어느 쪽이 가까운지는 재봐야 안다. 그래서 재는 것이다.
# 🔴 floater_가지치기.py 는 nerfstudio 가 필요하므로 gs 환경의 python 을 쓴다.
#    ltx venv 에는 nerfstudio 가 없다 (ModuleNotFoundError 로 한 번 실패했다).
set -u
M=${1:-M2}
source "$(dirname "$0")/환경.sh"
LOG=~/data/prune.log; : > "$LOG"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "---- G_$M 가지치기 ----"
python ~/bin/floater_가지치기.py --model "$M" 2>&1 \
  | grep -vE "FutureWarning|custom_|^$" | tee -a "$LOG"

P="${M}_pruned"
CFG=~/data/runs/G_$P/splatfacto/run1/config.yml
if [ ! -f "$CFG" ]; then say "🔴 가지친 모델이 없다"; exit 1; fi

say "---- 재평가 ----"
EV=~/data/평가
"$HOME/venv/ltx/bin/python" ~/bin/병합_평가.py --models "$P" 2>&1 | tail -2 | tee -a "$LOG"
rm -rf "$EV/프레임_$P"
ns-render camera-path --load-config "$CFG" \
  --camera-path-filename "$EV/경로_$P.json" \
  --output-path "$EV/프레임_$P" --output-format images >> "$LOG" 2>&1 < /dev/null
"$HOME/venv/ltx/bin/python" ~/bin/병합_PSNR.py \
  --render "$EV/프레임_$P" --gt ~/data/scenes/GT1/images_2 \
  --list "$EV/목록.txt" --tag "G_$P" 2>&1 | tee -a "$LOG"

say "---- 플라이스루 다시 ----"
FT=~/data/플라이스루
"$HOME/venv/ltx/bin/python" ~/bin/플라이스루.py --models "$P" 2>&1 | tail -2 | tee -a "$LOG"
ns-render camera-path --load-config "$CFG" \
  --camera-path-filename "$FT/경로_$P.json" \
  --output-path "$FT/플라이스루_$P.mp4" --output-format video >> "$LOG" 2>&1 < /dev/null
[ -f "$FT/플라이스루_$P.mp4" ] && say "  영상 $(du -h "$FT/플라이스루_$P.mp4" | cut -f1)"
say "끝"
