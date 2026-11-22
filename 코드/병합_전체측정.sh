#!/usr/bin/env bash
# ⑦⑧ 측정 전체를 한 번에. 학습이 끝난 뒤 돌린다.
#   1) 가우시안 개수            표 4
#   2) 표 3 — 숨긴 정답 대비     다리 있음/없음
#   3) 표 4 — FPS·VRAM
#   4) 플라이스루 영상 + 나란히 비교   §4.7
set -u
D="$(dirname "$0")"
LOG=~/data/전체측정.log; : > "$LOG"
say() { echo "" | tee -a "$LOG"; echo "######## $* ########" | tee -a "$LOG"; }

say "1. 가우시안 개수 (표 4)"
"$HOME/venv/ltx/bin/python" ~/bin/가우시안_개수.py A B GT1 C M AB 2>&1 | tee -a "$LOG"

say "2. 표 3 — 숨긴 정답 대비 (다리 있음/없음)"
bash ~/bin/병합_평가.sh 2>&1 | grep -E "^G_|^\[|🔴" | tee -a "$LOG"

say "3. 표 4 — FPS·VRAM"
bash ~/bin/병합_성능.sh M AB 2>&1 | grep -E "PSNR|🔴|^\[" | tee -a "$LOG"

say "4. 플라이스루 영상 (§4.7)"
bash ~/bin/플라이스루_실행.sh 2>&1 | grep -E "^\[|포즈|이음새" | tee -a "$LOG"

say "끝"
