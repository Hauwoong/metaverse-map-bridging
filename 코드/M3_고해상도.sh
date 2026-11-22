#!/usr/bin/env bash
# G_M3 — G_M2 와 같은 데이터로 **원본 해상도(1920×1080)** 학습.
#
# 왜: 플라이스루가 "흐리다/깨진다" 는 관찰에 가장 직접적인 대응이다.
#     지금까지는 960×540(축소 계수 2)으로 학습했다.
#
# 🔴 nerfstudio 의 축소 계수는 정수만 된다. 1440×810 은 불가능하므로
#    계수 1(원본 1920×1080)으로 간다.
#
# 🔴 441장을 1080p float32 로 캐싱하면 11GB 다. GPU 캐싱이면 VRAM 이 터진다.
#    cache-images cpu + cache-images-type uint8 로 2.7GB 로 줄인다.
#    (RAM 20GB 여유, VRAM 15GB 여유 확인함)
#
# 🔴 평가는 이전 표와 비교 가능하도록 **같은 960×540** 으로 렌더해
#    같은 실사(images_2)와 비교한다. 해상도를 바꾸면 PSNR 이 직접 비교되지 않는다.
set -u
source "$(dirname "$0")/환경.sh"

SC=~/data/scenes
RUN=~/data/runs
LOG=~/data/m3.log; : > "$LOG"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "================ G_M3 (원본 해상도) ================"
free -g | sed -n 2p | tee -a "$LOG"

OUT="$RUN/G_M3/splatfacto/run1"
rm -rf "$OUT"
T0=$SECONDS
ns-train splatfacto \
  --data "$SC/M2" \
  --output-dir "$RUN" \
  --experiment-name G_M3 \
  --timestamp run1 \
  --vis tensorboard \
  --max-num-iterations 30000 \
  --steps-per-save 10000 \
  --steps-per-eval-all-images 30000 \
  --steps-per-eval-image 10000 \
  --logging.local-writer.enable False \
  --pipeline.datamanager.cache-images cpu \
  --pipeline.datamanager.cache-images-type uint8 \
  colmap --downscale-factor 1 \
  >> "$LOG" 2>&1 < /dev/null
RC=$?
say "G_M3 종료 코드 $RC, $((SECONDS-T0))초"
if [ $RC -ne 0 ]; then
  say "🔴 실패"; tail -25 "$LOG"; exit $RC
fi

say "---- 표 3 재측정 (960×540 으로 렌더해 이전과 비교 가능하게) ----"
EV=~/data/평가
"$HOME/venv/ltx/bin/python" ~/bin/병합_평가.py --models M3 2>&1 | tail -2 | tee -a "$LOG"
rm -rf "$EV/프레임_M3"
ns-render camera-path \
  --load-config "$OUT/config.yml" \
  --camera-path-filename "$EV/경로_M3.json" \
  --output-path "$EV/프레임_M3" --output-format images \
  >> "$LOG" 2>&1 < /dev/null
for M in M3 M2 M AB; do
  [ -d "$EV/프레임_$M" ] || continue
  "$HOME/venv/ltx/bin/python" ~/bin/병합_PSNR.py \
    --render "$EV/프레임_$M" --gt ~/data/scenes/GT1/images_2 \
    --list "$EV/목록.txt" --tag "G_$M" 2>&1 | tee -a "$LOG"
done

say "---- 가우시안 개수 ----"
"$HOME/venv/ltx/bin/python" ~/bin/가우시안_개수.py M3 M2 M AB 2>&1 | tee -a "$LOG"

say "---- 플라이스루 ----"
FT=~/data/플라이스루
"$HOME/venv/ltx/bin/python" ~/bin/플라이스루.py --models M3 2>&1 | tail -2 | tee -a "$LOG"
ns-render camera-path --load-config "$OUT/config.yml" \
  --camera-path-filename "$FT/경로_M3.json" \
  --output-path "$FT/플라이스루_M3.mp4" --output-format video \
  >> "$LOG" 2>&1 < /dev/null
[ -f "$FT/플라이스루_M3.mp4" ] && say "  영상 $(du -h "$FT/플라이스루_M3.mp4" | cut -f1)"
say "끝"
