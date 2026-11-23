#!/usr/bin/env bash
# 끝점 좌표로 만든 궤적을 G_A·G_B 에서 각각 렌더한다.
# 목적: 끝점 스플랫이 간극을 어디까지 덮는지 본다 (코드/간극_궤적렌더.py 주석 참조).
set -u
source "$(dirname "$0")/환경.sh"
PROJ="${MV_PROJ:-/mnt/c/Users/a0103/metaverse-map-bridging}"   # 프로젝트 폴더 (윈도우 C:\Users\a0103\metaverse-map-bridging). 다른 위치면 MV_PROJ 로 지정
SRC="$PROJ/코드/간극_궤적렌더.py"
O=~/data/궤적
python "$SRC" --out "$O" || exit 1
for S in A B; do
  echo "== G_$S 궤적 렌더 =="
  ns-render camera-path \
    --load-config ~/data/runs/G_$S/splatfacto/run1/config.yml \
    --camera-path-filename "$O/경로_G_$S.json" \
    --output-path "$O/G_$S.mp4" \
    --rendered-output-names rgb < /dev/null 2>&1 | tail -3
done
ls -l "$O"
