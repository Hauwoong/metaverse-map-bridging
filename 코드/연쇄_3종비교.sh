#!/usr/bin/env bash
# 세 연쇄본을 같은 자로 비교한다.
#   A 6걸음 × 97f  (색보정 없음)   — 기준
#   B 12걸음 × 97f (색보정 있음)   — 걸음을 늘린 것
#   C 6걸음 × 161f (색보정 없음)   — 클립을 늘린 것
#
# 🔴 거리는 PSNR 로 재지 않는다. 어느 거리를 넘으면 포화해서 '더 멀리' 와
#    '그냥 다른 사진' 을 구분하지 못한다. SfM 으로 재구성한 뒤 초당 이동량으로 잰다.
#    (오늘 지표에 일곱 번 속았다. 재기 전에 무엇을 못 재는지 적는다.)
set -u
C="$HOME/miniconda3/envs/colmap/bin/colmap"
H="$HOME/data/연쇄"; W="$HOME/data/sfm_chain"
rm -rf "$W"; mkdir -p "$W"
LOG="$HOME/data/sfm_chain.log"; : > "$LOG"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# 이름:영상:프레임간격  (간격은 등록에 쓸 프레임이 140장 안팎이 되도록)
JOBS="h6f97:$H/연쇄.mp4:4 h12f97c:$H/h12f97c/연쇄.mp4:8 h6f161:$H/h6f161/연쇄.mp4:7"

log "================ 연쇄 3종 SfM ================"
for J in $JOBS; do
  NAME=${J%%:*}; REST=${J#*:}; VID=${REST%:*}; STEP=${REST##*:}
  [ -f "$VID" ] || { log "!! 없음: $VID"; continue; }
  D="$W/$NAME"; FR="$D/images"; mkdir -p "$FR" "$D/sparse"
  ffmpeg -v error -i "$VID" -vf "select=not(mod(n\,$STEP))" -vsync 0 "$FR/f%04d.png"
  N=$(ls "$FR" | wc -l)
  log "---- $NAME : $N 장 (매 $STEP 프레임) ----"
  "$C" feature_extractor --database_path "$D/db.db" --image_path "$FR" \
     --ImageReader.single_camera 1 --ImageReader.camera_model OPENCV \
     --FeatureExtraction.use_gpu 1 >/dev/null 2>&1
  "$C" exhaustive_matcher --database_path "$D/db.db" --FeatureMatching.use_gpu 1 >/dev/null 2>&1
  "$C" mapper --database_path "$D/db.db" --image_path "$FR" --output_path "$D/sparse" \
     --Mapper.ba_use_gpu 1 --Mapper.num_threads 8 >/dev/null 2>&1
  for M in "$D"/sparse/*/; do
    [ -d "$M" ] && "$C" model_converter --input_path "$M" --output_path "$M" --output_type TXT >/dev/null 2>&1
  done
  log "  모델 $(ls -d "$D"/sparse/*/ 2>/dev/null | wc -l)개"
done
log "================ 끝 — 분석은 코드/SfM_궤적분석.py ================"
