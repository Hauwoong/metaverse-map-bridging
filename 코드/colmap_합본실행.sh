#!/bin/bash
set -u
LOG="$HOME/data/all.log"
exec > >(tee -a "$LOG") 2>&1
log(){ echo "[$(date +%H:%M:%S)] $*"; }
C="$HOME/miniconda3/envs/colmap/bin/colmap"
PROJ="${MV_PROJ:-/mnt/c/Users/a0103/metaverse-map-bridging}"   # 프로젝트 폴더 (윈도우 C:\Users\a0103\metaverse-map-bridging). 다른 위치면 MV_PROJ 로 지정
SRC="$PROJ"
W="$HOME/data"; FR="$W/frames_all"; OUT="$W/sfm_all"; FPS=2.5
DB="$OUT/database.db"

log "================ 합본 COLMAP 시작 ================"
rm -rf "$FR" "$OUT"; mkdir -p "$FR" "$OUT/sparse"

for V in 1791 1792 1793; do
  F=$(ls "$SRC"/IMG_$V.* 2>/dev/null | head -1)
  [ -z "$F" ] && { log "!! IMG_$V 없음"; exit 1; }
  log "프레임 추출 IMG_$V  ($(basename "$F"))"
  /usr/bin/ffmpeg -v error -i "$F" -vf "fps=$FPS" -pix_fmt rgb24 "$FR/v${V}_%05d.png"
  log "   $(ls "$FR"/v${V}_*.png | wc -l)장"
done
N=$(ls "$FR" | wc -l)
log "총 ${N}장  ($(du -sh "$FR" | cut -f1))   v<영상>_<i> 의 원본 시각 = (i-1)/$FPS 초"

log "1/3 feature_extractor (single_camera: 세 영상 모두 같은 폰·같은 설정)"
"$C" feature_extractor --database_path "$DB" --image_path "$FR" \
  --ImageReader.single_camera 1 --ImageReader.camera_model OPENCV \
  --FeatureExtraction.use_gpu 1 2>&1 | tail -2

log "2/3 exhaustive_matcher ($((N*(N-1)/2)) 쌍) — 가장 오래 걸림"
"$C" exhaustive_matcher --database_path "$DB" --FeatureMatching.use_gpu 1 2>&1 | tail -2

log "3/3 mapper"
"$C" mapper --database_path "$DB" --image_path "$FR" --output_path "$OUT/sparse" \
  --Mapper.ba_use_gpu 1 --Mapper.num_threads 8 2>&1 | tail -3

log "================ 분석 ================"
for M in "$OUT"/sparse/*/; do
  [ -d "$M" ] || continue
  echo ""; echo "########## 모델 $(basename "$M") ##########"
  "$C" model_analyzer --path "$M" 2>&1 | sed 's/.*\] //' | grep -E "^(Cameras|Registered images|Points|Observations|Mean)" | sed 's/^/   /'
  "$C" model_converter --input_path "$M" --output_path "$M" --output_type TXT >/dev/null 2>&1
  python3 - "$M/images.txt" "$FR" "$FPS" <<'PY'
import sys, re, os, collections
img_txt, frdir, fps = sys.argv[1], sys.argv[2], float(sys.argv[3])
reg = collections.defaultdict(set)
with open(img_txt) as f:
    for line in f:
        if line.startswith("#"): continue
        p = line.split()
        if len(p) >= 10 and p[0].isdigit():
            m = re.match(r"v(\d+)_(\d+)", os.path.basename(p[-1]))
            if m: reg[m.group(1)].add(int(m.group(2)))
tot = collections.Counter()
for f in os.listdir(frdir):
    m = re.match(r"v(\d+)_", f)
    if m: tot[m.group(1)] += 1
print("   --- 영상별 등록률 ---")
for v in sorted(tot):
    r = len(reg.get(v, ()))
    print("   IMG_%s : %4d / %4d = %5.1f%%" % (v, r, tot[v], 100.0*r/tot[v]))
    miss = sorted(set(range(1, tot[v]+1)) - reg.get(v, set()))
    if miss and len(miss) < tot[v]:
        runs=[]; s=pv=miss[0]
        for x in miss[1:]:
            if x==pv+1: pv=x
            else: runs.append((s,pv)); s=pv=x
        runs.append((s,pv))
        big = [r for r in runs if r[1]-r[0]+1 >= 2]
        for a,b in big[:6]:
            print("        미등록 %6.1fs ~ %6.1fs (%d장)" % ((a-1)/fps,(b-1)/fps,b-a+1))
nv = sum(1 for v in tot if reg.get(v))
print("   >>> 이 모델에 포함된 영상 수: %d / %d  %s" % (
    nv, len(tot), "✅ 공통 좌표계 성립" if nv == len(tot) else "⚠️ 영상이 따로 놈"))
PY
done
log "================ 끝 ================"
