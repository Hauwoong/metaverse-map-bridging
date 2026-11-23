#!/usr/bin/env bash
# 표 5 2행 — **GT1 을 전혀 보지 않고** 생성 프레임의 포즈를 얻는다.
#
# 🔴 왜 필요한가 (2026-11-23 맥에서 제기):
#    `고정등록_원본DB.sh` 는 생성 프레임을 합본 DB **755장**에 매칭한다.
#    그 755장에 **숨긴 GT1 복도 실촬영이 들어 있다.**
#      생성 프레임 ──SIFT 매칭──▶ GT1 실촬영(정답) ──▶ 포즈
#    따라서 표 3(+5.76 dB), 20.19 dB, '복도 위 79%' 는 **포즈 오라클 위의 상한**이다.
#
# 🔴 왜 합본 모델을 재활용할 수 없는가:
#    DB 에서 GT1 이미지를 지워도 모델(sparse/1)의 2D 관측이 원래 DB 의 키포인트
#    번호를 가리키므로 섞을 수 없다 (NumPoints2D 불일치로 죽는다 — 고정등록_원본DB.sh 주석).
#    → **A + B + 생성만으로 새 DB 를 만들어 SfM 을 처음부터 돌린다.**
#
# 🔴 좌표계: 새 모델은 자기만의 좌표계를 갖는다. 채점하려면 합본 좌표계로 옮겨야 하는데,
#    **A·B 실촬영이 양쪽 모델에 모두 있으므로 그것만으로 Sim(3) 을 구할 수 있다.**
#    GT1 은 정렬에도 쓰지 않는다. (정렬은 코드/GT없는정렬.py 가 한다)
#
# 사용: bash GT없는등록.sh [태그]      태그 기본값 wp4 (그 실행의 생성 프레임을 그대로 쓴다)
set -u
C="$HOME/miniconda3/envs/colmap/bin/colmap"
SRC="$HOME/data/sfm_all"
TAG="${1:-wp4}"
D="$HOME/data/nogt/$TAG"
LOG="$HOME/data/nogt_$TAG.log"; : > "$LOG"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

rm -rf "$D"; mkdir -p "$D/images" "$D/sparse"
log "================ GT 없는 등록 ($TAG) ================"

# 1) A + B 실촬영만 복사 (GT1 = v1791_00162~00286 은 제외)
n=0
for f in "$HOME"/data/frames_all/v1791_*.png; do
  b=$(basename "$f"); i=$((10#${b:6:5}))
  if [ "$i" -le 143 ]; then cp "$f" "$D/images/"; n=$((n+1)); fi
done
for f in "$HOME"/data/frames_all/v1793_*.png; do cp "$f" "$D/images/"; n=$((n+1)); done
log "실촬영 $n장 (A + B. GT1 제외)"
EXCL=$(ls "$D/images" | grep -c '^v1791_' || true)

# 2) 생성 프레임 — 기존 실행의 것을 그대로 (같은 클립이어야 비교가 성립한다)
cp "$HOME/data/fixreg2/$TAG/images/"gen_*.png "$D/images/" 2>/dev/null
NG=$(ls "$D/images" | grep -c '^gen_' || true)
log "생성 $NG장 ($TAG 의 프레임 그대로) → 합계 $(ls "$D/images" | wc -l)장"
if [ "$NG" -eq 0 ]; then log "🔴 생성 프레임이 없다"; exit 1; fi

# 3) 전부 새로 특징 추출. 카메라는 하나로 공유한다(같은 기기, 같은 해상도).
log "feature_extractor (전체 $(ls "$D/images" | wc -l)장)"
"$C" feature_extractor --database_path "$D/db.db" --image_path "$D/images" \
   --ImageReader.single_camera 1 --ImageReader.camera_model OPENCV \
   --FeatureExtraction.use_gpu 1 >> "$LOG" 2>&1

# 4) 매칭 — 가장 오래 걸리는 단계
log "exhaustive_matcher (오래 걸린다)"
"$C" exhaustive_matcher --database_path "$D/db.db" \
   --FeatureMatching.use_gpu 1 >> "$LOG" 2>&1

# 5) SfM
log "mapper"
"$C" mapper --database_path "$D/db.db" --image_path "$D/images" \
   --output_path "$D/sparse" --Mapper.ba_use_gpu 1 --Mapper.num_threads 8 \
   >> "$LOG" 2>&1

# 6) 가장 큰 모델을 고른다
BEST=""; BN=0
for M in "$D"/sparse/*/; do
  [ -d "$M" ] || continue
  "$C" model_converter --input_path "$M" --output_path "$M" --output_type TXT >/dev/null 2>&1
  K=$(grep -cE '^[0-9]+ ' "$M/images.txt" 2>/dev/null || echo 0)
  K=$((K/1))
  if [ "$K" -gt "$BN" ]; then BN=$K; BEST="$M"; fi
done
if [ -z "$BEST" ]; then log "🔴 재구성 실패"; exit 1; fi
echo "$BEST" > "$D/best.txt"

A1=$(grep -oE 'v1791_[0-9]+\.png' "$BEST/images.txt" | sort -u | wc -l)
B1=$(grep -oE 'v1793_[0-9]+\.png' "$BEST/images.txt" | sort -u | wc -l)
G1=$(grep -oE 'gen_[0-9]+\.png' "$BEST/images.txt" | sort -u | wc -l)
log "---- 결과: A $A1, B $B1, 생성 $G1/$NG 등록 → $BEST ----"
log "🔴 A 와 B 가 **한 모델 안에** 들어왔는지가 관건이다."
log "   둘이 갈라졌다면 생성 다리가 둘을 잇지 못한 것이고, 그 자체가 결과다."
"$C" model_analyzer --path "$BEST" 2>&1 | sed 's/.*\] //' \
   | grep -E "^(Cameras|Images|Registered|Points|Mean)" | sed 's/^/    /' | tee -a "$LOG"
log "================ 끝 ================"
