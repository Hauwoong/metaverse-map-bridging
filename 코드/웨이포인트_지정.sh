#!/usr/bin/env bash
# 웨이포인트를 **지정한 GT1 인덱스**에 배치한다 (균등 배치와 비교하기 위해).
#
# 근거: 균등 4장(GT1 23/46/69/92)에서 A→W1 과 W4→B 만 실패했고,
#       공유 가시성을 보니 그 두 곳에 **완전 단절(0.000)** 이 있었다.
#         A 끝점 → GT1 10 : 0.000
#         GT1 4 → 5       : 0.000
#         마지막 → B      : 0.000
#       중간(W2~W4)에는 완전 단절이 없어 잘 이어졌다.
#       → 같은 장수를 **단절점에** 놓으면 커버리지가 오르는가?
set -u
source "$(dirname "$0")/환경.sh"
RUN=~/data/runs; SC=~/data/scenes; E=~/data/끝점
TAG="${1:-brk}"
shift || true
IDX=("$@")
[ ${#IDX[@]} -gt 0 ] || IDX=(5 30 51 120)
O="$E/wp_$TAG"; rm -rf "$O"; mkdir -p "$O"

echo "== 웨이포인트 ${#IDX[@]}장, GT1 인덱스: ${IDX[*]} =="
if [ ! -d "$E/G_GT1" ]; then
  ns-render dataset --load-config "$RUN/G_GT1/splatfacto/run1/config.yml" \
    --output-path "$E/G_GT1" --split train+test --rendered-output-names rgb \
    < /dev/null 2>&1 | tail -2
fi
i=0
for n in "${IDX[@]}"; do
  i=$((i+1))
  F=$(ls "$SC/GT1/images" | sort | sed -n "$((n+1))p")
  B="${F%.png}"
  SRC=$(find "$E/G_GT1" -type f \( -name "$B.jpg" -o -name "$B.png" \) | head -1)
  if [ -n "$SRC" ]; then
    # -nostdin 없으면 ffmpeg 이 루프 입력을 먹는다 (한 번 당했다)
    ffmpeg -nostdin -v error -y -i "$SRC" "$O/W$i.png"
    echo "  W$i : GT1 $n ($F)"
  else
    echo "  !! W$i 못 찾음 (GT1 $n, $B)"
  fi
done
ls -l "$O"/W*.png
