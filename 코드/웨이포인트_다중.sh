#!/usr/bin/env bash
# 오라클 웨이포인트를 N개로 늘려 다리를 만든다.
#
# 근거: 웨이포인트 1개가 '실제 복도 위 비율' 을 0% → 18% 로 올렸고,
#       그 18% 가 웨이포인트 주변에만 몰렸다(GT1 125장 중 8개 위치).
#       → 웨이포인트를 늘리면 그 '주변' 이 여러 곳에 생긴다.
#
# 🔴 정답(G_GT1)을 쓴다. 본 방법이 아니라 **상한(oracle)** 으로만 보고한다.
set -u
source "$(dirname "$0")/환경.sh"
RUN=~/data/runs; SC=~/data/scenes; E=~/data/끝점
N="${1:-4}"                      # 웨이포인트 개수
O="$E/wp$N"; mkdir -p "$O"

echo "== GT1 에서 웨이포인트 ${N}개 고르기 =="
TOT=$(ls "$SC/GT1/images" | wc -l)
for i in $(seq 1 "$N"); do
  IDX=$(( TOT * i / (N+1) ))
  F=$(ls "$SC/GT1/images" | sort | sed -n "${IDX}p")
  echo "  W$i : $F  (GT1 $IDX/$TOT)"
  echo "$F" >> "$O/목록.txt"
done

echo "== G_GT1 데이터셋 시점 렌더 (없으면) =="
if [ ! -d "$E/G_GT1" ]; then
  ns-render dataset --load-config "$RUN/G_GT1/splatfacto/run1/config.yml" \
    --output-path "$E/G_GT1" --split train+test --rendered-output-names rgb \
    < /dev/null 2>&1 | tail -2
fi

echo "== 웨이포인트 프레임 추출 =="
i=0
while read -r F; do
  i=$((i+1))
  B="${F%.png}"
  SRC=$(find "$E/G_GT1" -type f \( -name "$B.jpg" -o -name "$B.png" \) | head -1)
  if [ -n "$SRC" ]; then
    # 🔴 -nostdin 필수. 없으면 ffmpeg 이 while 루프의 stdin(목록)을 먹어버려
    #    다음 줄이 통째로 사라지거나 깨진 이름으로 읽힌다. 실제로 W2·W4 를 잃었다.
    ffmpeg -nostdin -v error -y -i "$SRC" "$O/W$i.png"
    echo "  W$i.png <- $SRC"
  else
    echo "  !! W$i 못 찾음 ($B)"
  fi
done < "$O/목록.txt"
ls -l "$O"/W*.png
