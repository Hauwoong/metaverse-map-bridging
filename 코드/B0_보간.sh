#!/usr/bin/env bash
# B0 베이스라인 — 프레임 보간.
#
# 왜 필요한가:
#   본 방법은 "생성 모델이 중간 공간을 '만든다'" 는 주장 위에 서 있다. 그 주장은
#   **보간은 만들지 못한다**는 대조가 있어야 성립한다. 보간은 두 프레임 사이를
#   워핑·블렌딩할 뿐 없던 내용을 만들지 못한다 — 그것을 같은 자로 재서 보인다.
#
# 🔴 평가는 생성 방법들과 **완전히 같은 파이프라인**으로 한다.
#    B 착지 PSNR, 중간 프레임의 B 유사도(블렌딩 지표), SfM 등록률·직진성·이동량.
#    지표가 다르면 비교가 성립하지 않는다.
#
# 두 가지를 만든다:
#   mci  — ffmpeg minterpolate, 움직임 보상 보간 (고전적 광학흐름 기반)
#   blend— 단순 교차 페이드 (보간의 하한선. "아무것도 안 한 것" 의 기준)
set -u
E="$HOME/data/끝점"; O="$HOME/data/B0"
mkdir -p "$O"
NF=161; FPS=24
DUR=$(echo "scale=4; $NF/$FPS" | bc)        # 6.7083 초
IN_FPS=$(echo "scale=6; 1/$DUR" | bc)       # 두 장을 그 길이에 걸쳐 놓는다

echo "== 입력 =="
ls -l "$E/I_A_out.png" "$E/I_B_in.png"
rm -rf "$O/src"; mkdir -p "$O/src"
# 생성 방법들과 같은 해상도(1024x576)로 맞춘다
ffmpeg -v error -y -i "$E/I_A_out.png" -vf scale=1024:576 "$O/src/f001.png"
ffmpeg -v error -y -i "$E/I_B_in.png"  -vf scale=1024:576 "$O/src/f002.png"

echo "== (1) minterpolate (mci + aobmc + vsbmc) =="
ffmpeg -v error -y -framerate "$IN_FPS" -i "$O/src/f%03d.png" \
  -vf "minterpolate=fps=$FPS:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1,format=yuv420p" \
  -frames:v "$NF" "$O/b0_mci.mp4"

echo "== (2) 단순 교차 페이드 (보간의 하한선) =="
ffmpeg -v error -y -loop 1 -t "$DUR" -i "$O/src/f001.png" -loop 1 -t "$DUR" -i "$O/src/f002.png" \
  -filter_complex "[0:v][1:v]xfade=transition=fade:duration=$DUR:offset=0,fps=$FPS,format=yuv420p" \
  -frames:v "$NF" "$O/b0_blend.mp4"

echo "== 결과 =="
for f in "$O"/b0_*.mp4; do
  N=$(ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of csv=p=0 "$f")
  echo "  $(basename "$f")  $N 프레임"
done
