#!/usr/bin/env bash
# ⑥-2 (정정판) 생성 프레임을 **실촬영을 고정한 채** 등록한다.
#
# 🔴 왜 다시 하는가.
#   앞서 생성 프레임과 실촬영을 **함께** SfM 에 넣었더니 재구성이 휘었다 —
#   같은 실촬영 쌍의 거리 비율이 변동계수 502%, 상관 0.31 로 흩어졌다.
#   생성 프레임의 불확실한 기하가 번들 조정에서 실촬영까지 끌어다 왜곡시킨 것이다.
#   그 위에서 구한 Sim(3) 는 존재할 수 없는 변환이었고, "간극의 4.7%" 는 무효다.
#
#   계획서의 경로 ⓐ 가 옳다: image_registrator --Mapper.fix_existing_frames 1.
#   실촬영 포즈를 고정하면 왜곡될 수 없고, 생성 프레임은 **합본 좌표계에 직접** 앉는다.
#   등록에 실패하면 그 자체가 결과다 ("생성 기하가 실촬영과 맞지 않는다").
#
# 비용: 합본 755장 전체 대신 **경계 부근 실촬영만 뽑은 부분 모델**에 등록한다.
#       포즈는 합본 그대로이므로 좌표계가 보존되고, 매칭은 25,200쌍으로 끝난다.
set -u
C="$HOME/miniconda3/envs/colmap/bin/colmap"
FR="$HOME/data/frames_all"
MERGED="$HOME/data/sfm_all/sparse/1"
W="$HOME/data/fixreg"
LOG="$HOME/data/fixreg.log"; : > "$LOG"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

VID="${1:-$HOME/data/연쇄/연쇄.mp4}"
TAG="${2:-chain6}"
STEP="${3:-4}"
NREAL="${4:-40}"

D="$W/$TAG"; IMG="$D/images"; SUB="$D/sub"
rm -rf "$D"; mkdir -p "$IMG" "$SUB" "$D/out"

log "================ ⑥-2 고정 등록 ($TAG) ================"
# 1) 경계 부근 실촬영 목록
ls "$FR" | grep '^v1791_' | sort | awk -F'[_.]' '$2<=143' | tail -n "$NREAL"  > "$D/keep.txt"
ls "$FR" | grep '^v1793_' | sort | head -n "$NREAL"                          >> "$D/keep.txt"
log "고정할 실촬영 $(wc -l < "$D/keep.txt")장"
while read -r f; do cp "$FR/$f" "$IMG/"; done < "$D/keep.txt"

# 2) 합본 모델에서 그 프레임만 남긴 부분 모델 (포즈는 그대로 — 좌표계 보존)
python3 - "$MERGED" "$D/keep.txt" "$SUB" <<'PY'
import sys, os
src, keepf, out = sys.argv[1], sys.argv[2], sys.argv[3]
keep = {l.strip() for l in open(keepf) if l.strip()}
os.makedirs(out, exist_ok=True)
lines = [l for l in open(os.path.join(src, "images.txt")) if not l.startswith("#")]
ids, cams, kept = set(), set(), []
for i in range(0, len(lines), 2):
    t = lines[i].split()
    if len(t) < 10: continue
    if t[9] in keep:
        ids.add(int(t[0])); cams.add(int(t[8]))
        kept.append((lines[i], lines[i+1] if i+1 < len(lines) else "\n"))
with open(os.path.join(out, "images.txt"), "w") as f:
    f.write("# Image list\n")
    for a, b in kept: f.write(a); f.write(b)
with open(os.path.join(out, "cameras.txt"), "w") as f:
    f.write("# Camera list\n")
    for l in open(os.path.join(src, "cameras.txt")):
        if not l.startswith("#") and int(l.split()[0]) in cams: f.write(l)
# 🔴 COLMAP 3.12+ 는 rigs.txt / frames.txt 를 함께 읽는다. 빠뜨리면
#    image_registrator 가 'existing_frame.RigId() == frame.RigId()' 로 죽는다.
import shutil as _sh
_sh.copy(os.path.join(src, "rigs.txt"), os.path.join(out, "rigs.txt"))
fr_keep = []
for l in open(os.path.join(src, "frames.txt")):
    if l.startswith("#"): continue
    v = l.split()
    if len(v) < 10: continue
    # ... NUM_DATA_IDS 뒤로 (SENSOR_TYPE, SENSOR_ID, DATA_ID) 가 반복된다
    nd = int(v[9])
    dids = [int(v[10 + 3*k + 2]) for k in range(nd)]
    if any(d in ids for d in dids):
        fr_keep.append(l)
with open(os.path.join(out, "frames.txt"), "w") as f:
    f.write("# Frame list" + chr(10))
    for l in fr_keep: f.write(l)
print("  frames %d개" % len(fr_keep))

n = 0
with open(os.path.join(out, "points3D.txt"), "w") as f:
    f.write("# 3D point list\n")
    for l in open(os.path.join(src, "points3D.txt")):
        if l.startswith("#"): continue
        v = l.split()
        if len(v) < 9: continue
        tr = [(int(v[k]), int(v[k+1])) for k in range(8, len(v)-1, 2)]
        tr = [(a, b) for a, b in tr if a in ids]
        if len(tr) < 2: continue
        f.write(" ".join(v[:8]) + " " + " ".join("%d %d" % x for x in tr) + "\n")
        n += 1
print("  부분 모델: 이미지 %d장, 점 %d개" % (len(kept), n))
PY

# 3) 생성 프레임
ffmpeg -v error -i "$VID" -vf "select=not(mod(n\,$STEP))" -vsync 0 "$IMG/gen_%04d.png"
log "생성 $(ls "$IMG" | grep -c '^gen_')장, 합계 $(ls "$IMG" | wc -l)장"

# 4) 특징 추출 — 한 번에, 카메라 1개·rig 1개로.
#    🔴 두 번에 나눠 추출하면 데이터베이스에 rig 가 2개 생기고, 부분 모델은 1개뿐이라
#       frame ID 공간이 충돌해 등록기가 죽는다:
#         Check failed: existing_frame.RigId() == frame.RigId()
#    그래서 생성 프레임을 실촬영과 같은 1920x1080 으로 올려 한 카메라로 묶는다.
#    내부 파라미터를 공유하는 셈인데, 조건 프레임을 렌더할 때 이미 쓴 가정과 같다
#    (실촬영 fx, fy 를 해상도 비율로 조정해 썼다).
log "생성 프레임 1920x1080 으로 확대"
for f in "$IMG"/gen_*.png; do
  ffmpeg -v error -y -i "$f" -vf scale=1920:1080 "$f.up.png" && mv "$f.up.png" "$f"
done
log "feature_extractor (단일 카메라·단일 rig)"
"$C" feature_extractor --database_path "$D/db.db" --image_path "$IMG" \
   --ImageReader.camera_model OPENCV --ImageReader.single_camera 1 \
   --FeatureExtraction.use_gpu 1 >/dev/null 2>&1
log "exhaustive_matcher"
"$C" exhaustive_matcher --database_path "$D/db.db" --FeatureMatching.use_gpu 1 >/dev/null 2>&1

# 5) 고정 등록
log "image_registrator (fix_existing_frames 1)"
"$C" image_registrator --database_path "$D/db.db" --input_path "$SUB" \
   --output_path "$D/out" --Mapper.fix_existing_frames 1 2>&1 | tail -5 | tee -a "$LOG"

"$C" model_converter --input_path "$D/out" --output_path "$D/out" --output_type TXT >/dev/null 2>&1
A=$(grep -oE 'v1791_[0-9]+\.png' "$D/out/images.txt" 2>/dev/null | sort -u | wc -l)
B=$(grep -oE 'v1793_[0-9]+\.png' "$D/out/images.txt" 2>/dev/null | sort -u | wc -l)
G=$(grep -oE 'gen_[0-9]+\.png'   "$D/out/images.txt" 2>/dev/null | sort -u | wc -l)
NG=$(ls "$IMG" | grep -c '^gen_')
log "---- 결과 ----"
log "  실촬영 A $A / B $B  (고정)"
log "  생성 등록 $G / $NG"
log "================ 끝 ================"
