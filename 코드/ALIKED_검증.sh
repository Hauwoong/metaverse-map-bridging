#!/usr/bin/env bash
# ALIKED 가 저텍스처 구간에서 정말 더 나은가 — **싸게 먼저 확인한다.**
#
# 배경: GT1 0~40 구간은 3D 점이 뒤쪽의 절반 이하(667 vs 1896)고 인접 프레임
#       공유 점이 0 인 지점도 있다. 웨이포인트를 어떻게 배치해도 그 구간은
#       살아나지 않았다 — 배치가 아니라 장면이 원인이다.
#
# 🔴 ALIKED 를 본 파이프라인에 쓰려면 합본 재구성(755장, 2시간)과 스플랫 3개를
#    전부 다시 만들어야 한다. 모델의 2D 관측이 SIFT 키포인트 번호를 가리키므로
#    섞어 쓸 수 없기 때문이다(어제 확인).
#    그 비용을 치르기 전에 **GT1 115장만으로 전제를 검증한다.**
set -u
C="$HOME/miniconda3/envs/colmap/bin/colmap"
SRC=~/data/scenes/GT1/images
W=~/data/aliked_test
rm -rf "$W"; mkdir -p "$W"/{sift,aliked}
LOG=~/data/aliked_test.log; : > "$LOG"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "================ ALIKED vs SIFT (GT1 $(ls "$SRC" | wc -l)장) ================"
for T in sift aliked; do
  D="$W/$T"; mkdir -p "$D/sparse"
  # 🔴 열거형 값 이름에 주의. 'ALIKED' 는 없다 — 모델 변종까지 적어야 한다.
  #    유효: ALIKED_N16ROT, ALIKED_N32 / 매칭: ALIKED_LIGHTGLUE, ALIKED_BRUTEFORCE
  #    (오류 메시지가 유효 목록을 알려주지 않아 하나씩 시험해 찾았다)
  if [ "$T" = sift ]; then
    FT=SIFT; MT=SIFT_BRUTEFORCE
  else
    FT=ALIKED_N16ROT; MT=ALIKED_LIGHTGLUE
  fi
  # 🔴 ALIKED 는 GPU 로 못 돌린다. conda 의 onnxruntime 에 CUDA 공급자가 빠져 있어
  #    libonnxruntime_providers_shared.so 적재에서 죽는다. CPU 로 우회한다.
  GPU=1; [ "$T" = aliked ] && GPU=0
  log "---- $T (extract=$FT, match=$MT) ----"
  "$C" feature_extractor --database_path "$D/db.db" --image_path "$SRC" \
     --ImageReader.single_camera 1 --ImageReader.camera_model OPENCV \
     --FeatureExtraction.type "$FT" --FeatureExtraction.use_gpu "$GPU" \
     >> "$LOG" 2>&1 || { log "  !! 특징 추출 실패"; continue; }
  "$C" exhaustive_matcher --database_path "$D/db.db" \
     --FeatureMatching.type "$MT" --FeatureMatching.use_gpu "$GPU" \
     >> "$LOG" 2>&1 || { log "  !! 매칭 실패"; continue; }
  "$C" mapper --database_path "$D/db.db" --image_path "$SRC" \
     --output_path "$D/sparse" --Mapper.ba_use_gpu 1 --Mapper.num_threads 8 \
     >> "$LOG" 2>&1
  BEST=""; BN=0
  for M in "$D"/sparse/*/; do
    [ -d "$M" ] || continue
    "$C" model_converter --input_path "$M" --output_path "$M" --output_type TXT >/dev/null 2>&1
    K=$(( ($(grep -c "^[0-9]" "$M/images.txt" 2>/dev/null || echo 0)) / 2 ))
    [ "$K" -gt "$BN" ] && { BN=$K; BEST="$M"; }
  done
  if [ -z "$BEST" ]; then log "  🔴 재구성 실패"; continue; fi
  log "  등록 $BN / $(ls "$SRC" | wc -l)장"
  "$C" model_analyzer --path "$BEST" 2>&1 | sed 's/.*\] //' \
     | grep -E "^(Points|Mean)" | sed 's/^/    /' | tee -a "$LOG"
  echo "$BEST" > "$D/best.txt"
done
log "================ 끝 ================"
