#!/bin/bash
set -u
log(){ echo "[$(date +%H:%M:%S)] $*"; }
C="$HOME/miniconda3/envs/colmap/bin/colmap"
W="$HOME/data"; FR="$W/frames1793"; OUT="$W/sfm1793"; FPS=3.2
N=$(ls "$FR" | wc -l)
rm -rf "$OUT"; mkdir -p "$OUT/sparse"
DB="$OUT/database.db"

log "1/4 feature_extractor  (입력 ${N}장, OPENCV, single_camera, GPU)"
"$C" feature_extractor --database_path "$DB" --image_path "$FR" \
  --ImageReader.single_camera 1 --ImageReader.camera_model OPENCV \
  --FeatureExtraction.use_gpu 1 2>&1 | tail -3

log "2/4 exhaustive_matcher  ($((N*(N-1)/2)) 쌍)"
"$C" exhaustive_matcher --database_path "$DB" --FeatureMatching.use_gpu 1 2>&1 | tail -3

log "3/4 mapper"
"$C" mapper --database_path "$DB" --image_path "$FR" --output_path "$OUT/sparse" \
  --Mapper.ba_use_gpu 1 2>&1 | tail -4

log "4/4 분석"
echo ""
for M in "$OUT"/sparse/*/; do
  [ -d "$M" ] || continue
  echo "===== 모델 $(basename "$M") ====="
  "$C" model_analyzer --path "$M" 2>&1 | grep -vE "^\s*$" | sed 's/^/   /'
  "$C" model_converter --input_path "$M" --output_path "$M" --output_type TXT >/dev/null 2>&1
  python - "$M/images.txt" "$N" "$FPS" <<'PY'
import sys, re, os
p, n_in, fps = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
idx=[]
with open(p) as f:
    for i, line in enumerate(f):
        if line.startswith("#"): continue
        parts = line.split()
        if len(parts) >= 10 and parts[0].isdigit():
            m = re.search(r"(\d+)", os.path.basename(parts[-1]))
            if m: idx.append(int(m.group(1)))
idx = sorted(set(idx))
print("   등록 %d / %d  =  %.1f%%" % (len(idx), n_in, 100.0*len(idx)/n_in))
miss = sorted(set(range(1, n_in+1)) - set(idx))
if not miss:
    print("   미등록 없음"); raise SystemExit
runs=[]; s=pv=miss[0]
for v in miss[1:]:
    if v==pv+1: pv=v
    else: runs.append((s,pv)); s=pv=v
runs.append((s,pv))
print("   미등록 구간 (원본 시각):")
for a,b in runs:
    print("      %6.1fs ~ %6.1fs   frame %3d~%3d  (%d장)" % ((a-1)/fps,(b-1)/fps,a,b,b-a+1))
PY
  echo ""
done
