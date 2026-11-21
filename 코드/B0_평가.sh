#!/usr/bin/env bash
# B0 를 생성 방법들과 **완전히 같은 자**로 잰다.
#   (1) 끝점 충실도 + 중간 프레임의 B 유사도(블렌딩 지표)
#   (2) SfM 등록률 · 직진성 · 장면대비 이동
# 지표가 다르면 비교가 성립하지 않으므로 파이프라인을 공유한다.
set -u
C="$HOME/miniconda3/envs/colmap/bin/colmap"
E="$HOME/data/끝점"; O="$HOME/data/B0"; W="$HOME/data/sfm_B0"
rm -rf "$W"; mkdir -p "$W"
LOG="$HOME/data/B0_평가.log"; : > "$LOG"
log(){ echo "$*" | tee -a "$LOG"; }

log "================ B0 평가 ================"
# 🔴 heredoc 은 파이프보다 먼저 명령에 붙어야 한다.
#    `cmd | tee <<EOF` 로 쓰면 heredoc 이 tee 의 입력이 되어 파이썬이 실행되지 않는다.
"$HOME/venv/ltx/bin/python" - "$O" "$E" <<'PY' 2>&1 | tee -a "$LOG'
import sys, os, numpy as np, cv2
from PIL import Image
O, E = sys.argv[1], sys.argv[2]
def L(p): return np.asarray(Image.open(p).convert("RGB").resize((1024,576)), float)
def psnr(a,b):
    m=((a-b)**2).mean(); return 99.0 if m==0 else 10*np.log10(255.0**2/m)
A, B = L(f"{E}/I_A_out.png"), L(f"{E}/I_B_in.png")
print()
print("%-10s %8s %8s | %s" % ("", "첫vsA", "끝vsB", "중간 프레임의 B 유사도 (25/50/75%)"))
for name in ("b0_flow", "b0_blend"):
    p = os.path.join(O, name + ".mp4")
    if not os.path.exists(p): continue
    cap = cv2.VideoCapture(p); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fr = {}
    for k in (0, n//4, n//2, 3*n//4, n-1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, k); ok, im = cap.read()
        if ok: fr[k] = np.asarray(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).resize((1024,576)), float)
    cap.release()
    ks = sorted(fr)
    mid = " ".join("%5.2f" % psnr(B, fr[k]) for k in ks[1:4])
    print("%-10s %8.2f %8.2f | %s" % (name, psnr(A, fr[ks[0]]), psnr(B, fr[ks[-1]]), mid))
print()
print("참고) 생성 방법들:  6걸음×97f  첫 36.62 / 끝 35.84,  중간 B유사도 13.15~14.51")
print("     단일 클립     첫 34.46 / 끝 26.91,  75%% 에서 15.72 (미리 보여줌)")
PY

for NAME in b0_flow b0_blend; do
  V="$O/$NAME.mp4"; [ -f "$V" ] || continue
  D="$W/$NAME"; FR="$D/images"; mkdir -p "$FR" "$D/sparse"
  ffmpeg -v error -i "$V" -vf "select=not(mod(n\,2))" -vsync 0 "$FR/f%04d.png"
  N=$(ls "$FR" | wc -l)
  log ""
  log "---- $NAME SfM : $N 장 ----"
  "$C" feature_extractor --database_path "$D/db.db" --image_path "$FR" \
     --ImageReader.single_camera 1 --ImageReader.camera_model OPENCV \
     --FeatureExtraction.use_gpu 1 >/dev/null 2>&1
  "$C" exhaustive_matcher --database_path "$D/db.db" --FeatureMatching.use_gpu 1 >/dev/null 2>&1
  "$C" mapper --database_path "$D/db.db" --image_path "$FR" --output_path "$D/sparse" \
     --Mapper.ba_use_gpu 1 --Mapper.num_threads 8 >/dev/null 2>&1
  K=0
  for M in "$D"/sparse/*/; do
    [ -d "$M" ] || continue
    "$C" model_converter --input_path "$M" --output_path "$M" --output_type TXT >/dev/null 2>&1
    K=$((K+1))
  done
  log "  모델 ${K}개"
done
log "================ 끝 ================"
