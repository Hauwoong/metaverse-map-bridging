#!/usr/bin/env bash
# ⑥-2 (재정정판) **합본 데이터베이스를 그대로 써서** 생성 프레임을 고정 등록한다.
#
# 🔴 앞선 시도들이 왜 전부 실패했는가 — 한 줄로:
#    **한 데이터베이스에서 만든 모델의 포즈를 다른 데이터베이스의 특징과 섞을 수 없다.**
#    모델의 2D 관측은 원래 데이터베이스의 키포인트 번호를 가리킨다. 새로 추출하면
#    개수부터 달라진다:
#      Check failed: image.NumPoints2D() == existing_image.NumPoints2D() (1288 vs 1339)
#    그 전에 만난 오류들(카메라 폭 충돌, rig 불일치, 이미지 ID 충돌)도 모두
#    "부분 모델을 손으로 만들어 새 데이터베이스에 붙이려" 했기 때문에 생긴 것이다.
#
#    755장 매칭 비용을 아끼려던 지름길이 전부 막혔다. 비용을 치르는 쪽이 옳다.
set -u
C="$HOME/miniconda3/envs/colmap/bin/colmap"
SRC="$HOME/data/sfm_all"
W="$HOME/data/fixreg2"
LOG="$HOME/data/fixreg2.log"; : > "$LOG"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

VID="${1:-$HOME/data/연쇄/연쇄.mp4}"
TAG="${2:-chain6}"
STEP="${3:-6}"          # 755장과 매칭하므로 생성 프레임을 조금 줄인다

D="$W/$TAG"
rm -rf "$D"; mkdir -p "$D/out"
log "================ ⑥-2 고정 등록 (원본 DB, $TAG) ================"

# 1) 합본 데이터베이스와 이미지 폴더를 그대로 쓴다 (복사본에 작업)
cp "$SRC/database.db" "$D/db.db"
mkdir -p "$D/images"
cp -r "$HOME/data/frames_all/." "$D/images/"
log "실촬영 $(ls "$D/images" | wc -l)장 (합본 그대로)"

# 2) 생성 프레임을 같은 해상도로 추가
ffmpeg -v error -i "$VID" -vf "select=not(mod(n\,$STEP)),scale=1920:1080" -vsync 0 "$D/images/gen_%04d.png"
NG=$(ls "$D/images" | grep -c '^gen_')
log "생성 $NG장 추가 → 합계 $(ls "$D/images" | wc -l)장"

# 3) 생성 프레임만 특징 추출. 기존 카메라를 그대로 쓴다.
ls "$D/images" | grep '^gen_' > "$D/genlist.txt"
log "feature_extractor (생성분만, 기존 카메라 1번 사용)"
"$C" feature_extractor --database_path "$D/db.db" --image_path "$D/images" \
   --image_list_path "$D/genlist.txt" \
   --ImageReader.existing_camera_id 1 \
   --FeatureExtraction.use_gpu 1 >/dev/null 2>&1

# 4) 매칭 — 생성분과 전체를 붙인다
log "exhaustive_matcher (오래 걸린다)"
"$C" exhaustive_matcher --database_path "$D/db.db" --FeatureMatching.use_gpu 1 >/dev/null 2>&1

# 5) 고정 등록
log "image_registrator (fix_existing_frames 1)"
"$C" image_registrator --database_path "$D/db.db" --input_path "$SRC/sparse/1" \
   --output_path "$D/out" --Mapper.fix_existing_frames 1 2>&1 | tail -6 | tee -a "$LOG"

if [ -f "$D/out/images.bin" ]; then
  "$C" model_converter --input_path "$D/out" --output_path "$D/out" --output_type TXT >/dev/null 2>&1
  G=$(grep -oE 'gen_[0-9]+\.png' "$D/out/images.txt" | sort -u | wc -l)
  R=$(grep -oE 'v179[123]_[0-9]+\.png' "$D/out/images.txt" | sort -u | wc -l)
  log "---- 결과: 실촬영 $R장 유지, 생성 $G / $NG 등록 ----"
else
  log "---- 🔴 출력 없음 ----"
fi
log "================ 끝 ================"
