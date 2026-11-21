#!/usr/bin/env bash
# ⑤′ 웨이포인트 프레임 준비. 두 종류를 만든다.
#
# 🔴 설계상 반드시 구분해야 하는 것:
#
#   (1) 오라클 웨이포인트  I_M_oracle.png   — G_GT1(숨긴 정답)에서 렌더
#       정답을 입력으로 쓰므로 "끝점만으로 잇는다" 는 전제를 위반한다.
#       **본 방법으로 쓸 수 없다.** 오직 진단용이다:
#       "분할하면 블렌딩이 사라지는가?" = "길이 제약이 원인인가?" 를 가린다.
#       보고서에서는 상한(oracle upper bound)으로만 보고한다.
#
#   (2) 재귀 웨이포인트    I_M_recursive.png — 1차 생성물의 중간 프레임에서 취함
#       정답을 전혀 쓰지 않는다. 이것이 실제로 제안할 수 있는 방법이다.
#       1차 생성이 블렌딩이더라도, 각 절반이 덮을 거리가 줄어들므로
#       두 번째 라운드에서 내용이 채워질 여지가 생긴다.
#
# ⚠️ ns-render 는 .jpg 로 쓴다. 조건 프레임은 무손실이어야 하므로 ffmpeg 로 png 변환한다.
set -u
source "$(dirname "$0")/환경.sh"
RUN=~/data/runs
SC=~/data/scenes
E=~/data/끝점
G=~/data/생성
mkdir -p "$E"

# ── (1) 오라클: GT1 의 시간상 중앙 프레임 ──────────────────────────────
MID=$(ls "$SC/GT1/images" | sort | awk -v n="$(ls "$SC/GT1/images" | wc -l)" 'NR==int((n+1)/2)')
echo "== 오라클 웨이포인트: GT1 중앙 프레임 $MID =="
CFG="$RUN/G_GT1/splatfacto/run1/config.yml"
if [ ! -d "$E/G_GT1" ]; then
  ns-render dataset --load-config "$CFG" --output-path "$E/G_GT1" \
    --split train+test --rendered-output-names rgb < /dev/null 2>&1 | tail -2
fi
SRC=$(find "$E/G_GT1" -type f -name "${MID%.png}.jpg" -o -type f -name "$MID" | head -1)
if [ -n "$SRC" ]; then
  ffmpeg -v error -y -i "$SRC" "$E/I_M_oracle.png"
  cp "$SC/GT1/images/$MID" "$E/I_M_oracle_원본.png"
  echo "  I_M_oracle.png <- $SRC"
else
  echo "  !! 못 찾음: $MID"; find "$E/G_GT1" -type f | head -3
fi

# ── (2) 재귀: 1차 생성 영상의 중간 프레임 ──────────────────────────────
V="$G/bridge_render.mp4"
if [ -f "$V" ]; then
  N=$(ffprobe -v error -select_streams v:0 -count_frames \
      -show_entries stream=nb_read_frames -of csv=p=0 "$V")
  M=$(( N / 2 ))
  echo "== 재귀 웨이포인트: 1차 생성물 $N 프레임 중 $M 번 =="
  ffmpeg -v error -y -i "$V" -vf "select=eq(n\,$M)" -frames:v 1 -update 1 "$E/I_M_recursive.png"
  echo "  I_M_recursive.png 생성"
else
  echo "  !! 1차 생성물 없음: $V"
fi

echo "== 결과 =="
ls -l "$E"/I_M_*.png 2>/dev/null
