#!/usr/bin/env bash
# 다시 만든 다리로 **GT1 없이** 포즈를 구하고 109° 가 잡혔는지 확인한다.
#
# 🔴 순서가 중요하다 — **사슬이 이어졌는지 먼저 본다.**
#    지난번에는 잔차·CV 부터 봤고, 그 숫자들은 "기하가 안 맞는다" 까지만 말했지
#    **왜** 안 맞는지는 말하지 못했다. 실제 원인(공유트랙 0 인 고리)은
#    연결 강도를 따로 재고 나서야 드러났다.
#    그러니 ②연결강도 → ③정렬 → ④회전 순으로 본다. 앞이 실패하면 뒤는 의미가 없다.
#
# 사용: setsid bash 코드/GT없는재측정.sh r5 [STEP]
set -u
PROJ="${MV_PROJ:-/mnt/c/Users/a0103/metaverse-map-bridging}"
GSPY="$HOME/miniconda3/envs/gs/bin/python"
TAG="${1:-r5}"
STEP="${2:-5}"
VID="$HOME/data/생성/wp_$TAG/다리.mp4"
STAGE="$HOME/data/fixreg2/$TAG"
LOG="$HOME/data/재측정_$TAG.log"; : > "$LOG"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

[ -f "$VID" ] || { log "🔴 다리 영상 없음: $VID"; exit 1; }
log "================ GT 없는 재측정 ($TAG) ================"

# 1) 생성 프레임 추출 — 🔴 실촬영과 같은 해상도로. 카메라 모델을 공유해야 한다.
rm -rf "$STAGE/images"; mkdir -p "$STAGE/images"
ffmpeg -nostdin -v error -i "$VID" \
   -vf "select=not(mod(n\,$STEP)),scale=1920:1080" -vsync 0 \
   "$STAGE/images/gen_%04d.png"
NG=$(ls "$STAGE/images" | grep -c '^gen_')
log "생성 프레임 $NG장 추출 (step $STEP)"

# 2) GT1 없는 SfM
log "---- GT 없는 등록 (오래 걸린다) ----"
bash "$PROJ/코드/GT없는등록.sh" "$TAG" >> "$LOG" 2>&1
D="$HOME/data/nogt/$TAG"
[ -s "$D/best.txt" ] || { log "🔴 재구성 실패"; exit 1; }

{
echo "######## 부분 모델 구성 ########"
"$GSPY" "$PROJ/코드/부분모델_구성.py" --dir "$D/sparse"
echo
echo "######## 다리 연결 강도 — 사슬이 이어졌는가 ########"
"$GSPY" "$PROJ/코드/다리_연결강도.py" --dir "$D"
echo
echo "######## A·B 개별 정렬 — 회전이 잡혔는가 ########"
"$GSPY" "$PROJ/코드/AB_개별정렬.py" --src "$(cat "$D/best.txt")"
echo
echo "######## 표 5 2행 — 채점 ########"
"$GSPY" "$PROJ/코드/GT없는정렬.py" --src "$(cat "$D/best.txt")" --tag "COLMAP(GT1제외,$TAG)"
echo "######## 끝 ########"
} 2>&1 | tee -a "$LOG" "$HOME/data/표5_$TAG.log"

log "================ 끝 ================"
