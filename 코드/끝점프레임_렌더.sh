#!/usr/bin/env bash
# ④ 끝점 조건 프레임 렌더 — LTX-Video 에 넣을 두 장을 만든다.
#   I_A_out : G_A 에서 본 A 의 마지막 시점 (중간 구간을 향해 선 자리)
#   I_B_in  : G_B 에서 본 B 의 첫 시점
#
# 포즈를 손으로 만들지 않는다. 합본 COLMAP 이 이미 세 영상을 같은 좌표계에 두었으므로,
# 경계 프레임의 '실제 촬영 포즈' 가 그대로 끝점 포즈다. ns-render dataset 으로
# 데이터셋 시점을 전부 렌더한 뒤 경계 프레임 이름으로 골라낸다.
#   → 생성 영상의 첫/끝 프레임이 3DGS 렌더와 같은 카메라에서 나오므로,
#     나중에 재구성한 중간 구간이 양쪽과 맞물릴 근거가 된다.
#
# ⚠️ --split train+test 를 쓴다. eval_mode=interval, eval_interval=8 이라
#    인덱스 0 (= B 의 첫 프레임) 은 eval 로 빠져 train 에만 렌더하면 없다. 한 번 당했다.
# ⚠️ ns-render 는 .jpg 로 쓴다. 조건 프레임은 무손실이어야 하므로 ffmpeg 로 png 변환한다.
#    (확장자만 바꿔 복사하면 png 이름의 jpeg 파일이 된다. 이것도 한 번 당했다.)
# ⚠️ 먼저 코드/nerfstudio_패치.sh 필요 (ns-render 도 eval_utils.py 를 지나간다).
set -u
source "$(dirname "$0")/환경.sh"
RUN=~/data/runs
SC=~/data/scenes
OUTD=~/data/끝점
mkdir -p "$OUTD"

for S in A B; do
  CFG="$RUN/G_$S/splatfacto/run1/config.yml"
  [ -f "$CFG" ] || { echo "G_$S: config.yml 없음 — 학습 먼저"; exit 1; }
  echo "== G_$S 데이터셋 시점 렌더 (train+test) =="
  ns-render dataset --load-config "$CFG" \
    --output-path "$OUTD/G_$S" --split train+test \
    --rendered-output-names rgb < /dev/null 2>&1 | tail -2
done

echo
echo "== 경계 프레임 고르기 =="
A_LAST=$(ls "$SC/A/images"  | sort | tail -1)
B_FIRST=$(ls "$SC/B/images" | sort | head -1)
echo "  A 마지막 프레임 : $A_LAST"
echo "  B 첫 프레임     : $B_FIRST"

pick() {  # pick <장면> <프레임.png> <태그>
  local S=$1 F=$2 TAG=$3 B=${2%.png} SRC
  SRC=$(find "$OUTD/G_$S" -type f \( -name "$B.jpg" -o -name "$B.png" \) | head -1)
  if [ -z "$SRC" ]; then
    echo "  !! $TAG 못 찾음 ($B) — 렌더 디렉터리 내용:"
    find "$OUTD/G_$S" -type f | head -3
    return 1
  fi
  ffmpeg -v error -y -i "$SRC" "$OUTD/$TAG.png"
  # 비교용으로 같은 포즈의 원본 실촬영 프레임도 같이 둔다.
  cp "$SC/$S/images/$F" "$OUTD/${TAG}_원본.png"
  echo "  $TAG.png        <- $SRC (3DGS 렌더)"
  echo "  ${TAG}_원본.png <- $SC/$S/images/$F (실촬영)"
}
pick A "$A_LAST"  I_A_out
pick B "$B_FIRST" I_B_in

echo
echo "== 결과 =="
for F in I_A_out I_B_in; do
  for V in "$F.png" "${F}_원본.png"; do
    [ -f "$OUTD/$V" ] && printf "  %-20s %s\n" "$V" "$(identify -format '%wx%h %m' "$OUTD/$V" 2>/dev/null || file -b "$OUTD/$V" | cut -c1-40)"
  done
done
