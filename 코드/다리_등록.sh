#!/usr/bin/env bash
# ⑥-2 생성 구간을 실촬영 좌표계에 붙인다.
#
# 목적 두 가지:
#   (1) 판정  — 실촬영 A ↔ 생성 ↔ 실촬영 B 가 **한 모델**에 들어가는가.
#               들어가면 생성 구간이 두 끝을 실제로 잇는다는 뜻이다.
#   (2) 측정  — 모델 안의 실촬영 프레임은 합본 모델에서 위치를 이미 안다.
#               그 대응으로 Sim(3) 를 구해 생성 구간의 이동을 **절대 단위**로 환산한다.
#               오늘 내내 문제였던 '재구성마다 스케일이 달라지는' 것이 여기서 사라진다.
#
# 🔴 합본 모델(755장) 전체에 image_registrator 로 붙이지 않는다.
#    어휘 트리 없이 755×N 매칭은 몇 시간이다. 경계 부근 실촬영만 섞으면
#    같은 질문에 훨씬 싸게 답할 수 있다.
set -u
C="$HOME/miniconda3/envs/colmap/bin/colmap"
FR="$HOME/data/frames_all"
W="$HOME/data/bridge_reg"
LOG="$HOME/data/bridge_reg.log"; : > "$LOG"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

VID="${1:-$HOME/data/연쇄/연쇄.mp4}"   # 기본: 6걸음×97f
TAG="${2:-chain6}"
STEP="${3:-4}"
NREAL="${4:-40}"                        # 양쪽 경계에서 가져올 실촬영 장수

D="$W/$TAG"; IMG="$D/images"
rm -rf "$D"; mkdir -p "$IMG" "$D/sparse"

log "================ ⑥-2 다리 등록 ($TAG) ================"
# A 의 끝 NREAL 장 (1791 의 00143 이하)
ls "$FR" | grep '^v1791_' | sort | awk -F'[_.]' '$2<=143' | tail -n "$NREAL" > "$D/realA.txt"
# B 의 앞 NREAL 장
ls "$FR" | grep '^v1793_' | sort | head -n "$NREAL" > "$D/realB.txt"
while read -r f; do cp "$FR/$f" "$IMG/"; done < "$D/realA.txt"
while read -r f; do cp "$FR/$f" "$IMG/"; done < "$D/realB.txt"
log "실촬영 A $(wc -l < "$D/realA.txt")장, B $(wc -l < "$D/realB.txt")장"

ffmpeg -v error -i "$VID" -vf "select=not(mod(n\,$STEP))" -vsync 0 "$IMG/gen_%04d.png"
NG=$(ls "$IMG" | grep -c '^gen_')
log "생성 ${NG}장 (매 $STEP 프레임)"
log "합계 $(ls "$IMG" | wc -l)장  →  $(( $(ls "$IMG" | wc -l) * ($(ls "$IMG" | wc -l)-1) / 2 )) 쌍"

# 실촬영은 1920x1080, 생성은 1024x576 — 카메라가 다르므로 single_camera 를 쓰면 안 된다.
log "1/3 feature_extractor (해상도가 다르므로 single_camera 안 씀)"
"$C" feature_extractor --database_path "$D/db.db" --image_path "$IMG" \
   --ImageReader.camera_model OPENCV --FeatureExtraction.use_gpu 1 >/dev/null 2>&1
log "2/3 exhaustive_matcher"
"$C" exhaustive_matcher --database_path "$D/db.db" --FeatureMatching.use_gpu 1 >/dev/null 2>&1
log "3/3 mapper"
"$C" mapper --database_path "$D/db.db" --image_path "$IMG" --output_path "$D/sparse" \
   --Mapper.ba_use_gpu 1 --Mapper.num_threads 8 >/dev/null 2>&1

log "---- 모델별 구성 ----"
for M in "$D"/sparse/*/; do
  [ -d "$M" ] || continue
  "$C" model_converter --input_path "$M" --output_path "$M" --output_type TXT >/dev/null 2>&1
  A=$(grep -oE 'v1791_[0-9]+\.png' "$M/images.txt" | sort -u | wc -l)
  B=$(grep -oE 'v1793_[0-9]+\.png' "$M/images.txt" | sort -u | wc -l)
  G=$(grep -oE 'gen_[0-9]+\.png'   "$M/images.txt" | sort -u | wc -l)
  log "  모델 $(basename "$M") : 실촬영A $A / 생성 $G / 실촬영B $B"
done
log "================ 끝 ================"
