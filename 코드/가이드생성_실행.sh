#!/usr/bin/env bash
# ⑤″ 끝점 좌표 기반 가이드 조건 생성.
#
# 끝점 두 개의 포즈를 알면 그 사이 카메라 궤적을 만들 수 있다(코드/간극_궤적렌더.py).
# 그 궤적을 G_A·G_B 에서 렌더하면 '양 끝은 정확하고 가운데는 무너진' 영상이 나온다.
# 이것을 낮은 strength 로 조건에 넣어, 모델에게 카메라 운동을 기하로 알려준다.
#
# 🔴 이것은 정답(G_GT1)을 전혀 쓰지 않는다. 오라클과 달리 본 방법이 될 수 있다.
set -u
PY="$HOME/venv/ltx/bin/python"
PROJ="${MV_PROJ:-/mnt/c/Users/a0103/metaverse-map-bridging}"   # 프로젝트 폴더 (윈도우 C:\Users\a0103\metaverse-map-bridging). 다른 위치면 MV_PROJ 로 지정
SRC="$PROJ/코드/끝점조건_생성.py"
E="$HOME/data/끝점"; O="$HOME/data/생성"; T="$HOME/data/궤적"
EMB="$O/프롬프트_임베딩.pt"
mkdir -p "$O"
export HF_HOME="$HOME/.cache/huggingface"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

S="${1:-0.2}"; shift || true    # 가이드 strength

# 가이드 영상: G_A 렌더 → G_B 렌더 를 클립 전체에 걸쳐 교차 페이드.
# 양 끝은 각 스플랫의 정확한 렌더이고 가운데는 둘 다 무너진 구간이다.
G="$T/가이드.mp4"
if [ ! -f "$G" ]; then
  D=$(echo "scale=3; 160/24" | bc)
  echo "== 가이드 영상 생성 (교차 페이드 ${D}초) =="
  ffmpeg -v error -y -i "$T/G_A.mp4" -i "$T/G_B.mp4" \
    -filter_complex "[0:v][1:v]xfade=transition=fade:duration=$D:offset=0,format=yuv420p" \
    -r 24 "$G" || exit 1
fi
ffprobe -v error -select_streams v:0 -count_frames \
  -show_entries stream=nb_read_frames -of csv=p=0 "$G" | sed 's/^/  가이드 프레임 수 /'

echo "== 생성 (guide-strength=$S) =="
"$PY" "$SRC" --first "$E/I_A_out.png" --last "$E/I_B_in.png" \
  --out "$O/guided_s$S.mp4" --embeds "$EMB" \
  --guide "$G" --guide-strength "$S" "$@"
