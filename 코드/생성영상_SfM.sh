#!/usr/bin/env bash
# ⑥-1 생성 영상의 기하 일관성 검증.
#
# 왜 이것부터 하는가:
#   오늘 끝점 거리 PSNR 이 네 번 속였다 — '생성 vs 붕괴' 도, '전진 vs 자기복제' 도
#   구분하지 못했다. SfM 은 구분한다. 기하적으로 일관되지 않은 영상은 등록에 실패하고,
#   평행이동이 없는 영상(러닝머신)은 등록되더라도 카메라가 한 점에 뭉친다.
#
#   그래서 '등록률' 과 '카메라 경로 길이' 두 가지를 본다.
#     등록률 낮음        → 기하가 깨져 있다
#     등록률 높은데 경로 짧음 → 그림은 일관되지만 움직이지 않았다 (러닝머신)
#
# 합본 모델에 고정 등록(⑥-2, image_registrator --Mapper.fix_existing_frames 1)은
# 이것이 통과한 영상에 대해서만 의미가 있다. 그래서 단독 SfM 을 먼저 돌린다.
set -u
C="$HOME/miniconda3/envs/colmap/bin/colmap"
G="$HOME/data/생성"; CH="$HOME/data/연쇄"
W="$HOME/data/sfm_gen"
rm -rf "$W"; mkdir -p "$W"
LOG="$HOME/data/sfm_gen.log"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# 이름:영상경로:프레임간격
JOBS="single:$G/bridge_render.mp4:2 oracle:$G/split_oracle.mp4:2 chain:$CH/연쇄.mp4:4"

log "================ ⑥-1 생성 영상 단독 SfM ================"
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
  "$C" exhaustive_matcher --database_path "$D/db.db" \
     --FeatureMatching.use_gpu 1 >/dev/null 2>&1
  "$C" mapper --database_path "$D/db.db" --image_path "$FR" \
     --output_path "$D/sparse" --Mapper.ba_use_gpu 1 --Mapper.num_threads 8 \
     >/dev/null 2>&1

  BEST=""; BESTN=0
  for M in "$D"/sparse/*/; do
    [ -d "$M" ] || continue
    "$C" model_converter --input_path "$M" --output_path "$M" --output_type TXT >/dev/null 2>&1
    K=$(grep -c "^[0-9]" "$M/images.txt" 2>/dev/null || echo 0)
    K=$(( K / 2 ))
    [ "$K" -gt "$BESTN" ] && { BESTN=$K; BEST="$M"; }
  done
  if [ -z "$BEST" ]; then
    log "  🔴 재구성 실패 — 모델이 하나도 안 나왔다"
    continue
  fi
  log "  모델 $(basename "$BEST") : 등록 $BESTN / $N 장 ($(echo "scale=1; 100*$BESTN/$N" | bc)%)"
  "$C" model_analyzer --path "$BEST" 2>&1 | sed 's/.*\] //' \
     | grep -E "^(Points|Observations|Mean)" | sed 's/^/    /' | tee -a "$LOG"
  python3 - "$BEST/images.txt" "$NAME" <<'PY' | tee -a "$LOG"
import sys, numpy as np
def q2R(q):
    w,x,y,z=q
    return np.array([[1-2*y*y-2*z*z,2*x*y-2*z*w,2*x*z+2*y*w],
                     [2*x*y+2*z*w,1-2*x*x-2*z*z,2*y*z-2*x*w],
                     [2*x*z-2*y*w,2*y*z+2*x*w,1-2*x*x-2*y*y]])
ls=[l for l in open(sys.argv[1]) if not l.startswith("#")]
C={}
for i in range(0,len(ls),2):
    t=ls[i].split()
    if len(t)<10: continue
    R=q2R(list(map(float,t[1:5]))); tv=np.array(list(map(float,t[5:8])))
    C[t[9]]=-R.T@tv
if len(C)<2:
    print("    카메라 2개 미만 — 분석 불가"); raise SystemExit
ks=sorted(C); P=np.array([C[k] for k in ks])
path=sum(np.linalg.norm(P[i+1]-P[i]) for i in range(len(P)-1))
span=np.linalg.norm(P[-1]-P[0])
size=np.linalg.norm(P.max(0)-P.min(0))
print("    카메라 경로 길이 %.3f / 양 끝 직선거리 %.3f / 전체 퍼짐 %.3f" % (path,span,size))
print("    직진성(직선/경로) %.3f   ← 1 에 가까우면 쭉 간 것, 0 에 가까우면 제자리" % (span/max(path,1e-9)))
PY
done
log "================ 끝 ================"
