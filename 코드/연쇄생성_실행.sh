#!/usr/bin/env bash
# ⑤‴ 짧은 보폭 연쇄 생성.
#
# 가설: 지금까지 실패한 이유는 '앵커가 없어서' 가 아니라 '보폭이 너무 커서' 다.
#   - 단일 클립: 간극 전체를 한 번에 → 보간이 제일 쉬운 답이라 블렌딩했다.
#   - 재귀 분할: 50% 를 한 번에 뛰었고, 그 웨이포인트가 이미 오염돼 있었다.
#   - 짧게 끊으면 매 걸음의 재료가 직전 걸음의 '제대로 된' 결과가 된다.
#
# 설계:
#   - 중간 걸음들은 **끝점 조건을 걸지 않는다(I2V)**. 끝이 고정되지 않으면
#     두 끝을 보간할 유인이 없어져 모델은 전진할 수밖에 없다.
#   - 마지막 걸음에만 B 를 고정해 착지시킨다.
#   - 각 걸음의 시작 프레임 = 직전 걸음 결과의 마지막 프레임.
#
# 🔴 정답(G_GT1)을 전혀 쓰지 않는다. 본 방법이 될 수 있다.
set -u
PY="$HOME/venv/ltx/bin/python"
PROJ="${MV_PROJ:-/mnt/c/Users/a0103/metaverse-map-bridging}"   # 프로젝트 폴더 (윈도우 C:\Users\a0103\metaverse-map-bridging). 다른 위치면 MV_PROJ 로 지정
SRC="$PROJ/코드/끝점조건_생성.py"
E="$HOME/data/끝점"; O="$HOME/data/연쇄"
EMB="$HOME/data/생성/프롬프트_임베딩.pt"
mkdir -p "$O"
export HF_HOME="$HOME/.cache/huggingface"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

HOPS="${1:-6}"          # 걸음 수
NF="${2:-97}"           # 걸음당 프레임 (8k+1)
CFIX="${3:-0}"          # 1 이면 매 걸음 앵커의 색을 A 끝점 기준으로 되돌린다
shift 3 2>/dev/null || shift $# 2>/dev/null || true
TAG="h${HOPS}f${NF}$([ "$CFIX" = 1 ] && echo c)"
O="$O/$TAG"; mkdir -p "$O"
echo "== 연쇄 설정: ${HOPS}걸음 × ${NF}프레임, 색보정=$CFIX  →  $O =="

CUR="$E/I_A_out.png"
cp "$CUR" "$O/앵커_00.png"
for i in $(seq 1 "$HOPS"); do
  OUT="$O/hop_$(printf %02d $i).mp4"
  if [ "$i" -eq "$HOPS" ]; then
    echo "########## 걸음 $i/$HOPS — B 에 착지 (끝점 조건 있음) ##########"
    LAST=(--last "$E/I_B_in.png")
  else
    echo "########## 걸음 $i/$HOPS — 전진 (끝점 조건 없음) ##########"
    LAST=()
  fi
  "$PY" "$SRC" --first "$CUR" "${LAST[@]}" --out "$OUT" \
      --embeds "$EMB" --num-frames "$NF" "$@" || exit 1
  # 이번 걸음의 마지막 프레임 = 다음 걸음의 시작
  N=$(ffprobe -v error -select_streams v:0 -count_frames \
      -show_entries stream=nb_read_frames -of csv=p=0 "$OUT")
  CUR="$O/앵커_$(printf %02d $i).png"
  ffmpeg -v error -y -i "$OUT" -vf "select=eq(n\,$((N-1)))" -frames:v 1 -update 1 "$CUR"
  if [ "$CFIX" = 1 ] && [ "$i" -lt "$HOPS" ]; then
    # 자기 출력을 다시 먹는 구조라 색이 누적으로 민다. 먹이기 전에 되돌린다.
    "$PY" "$PROJ/코드/앵커_색보정.py"       --src "$CUR" --ref "$E/I_A_out.png" --out "$CUR"
  fi
  echo "  앵커 갱신: $CUR"
done

echo "########## 이어붙이기 ##########"
# 각 걸음의 첫 프레임은 직전 걸음의 마지막과 같으므로 두 번째 걸음부터 1프레임 버린다.
L=""; F=""; n=0
for i in $(seq 1 "$HOPS"); do
  L="$L -i $O/hop_$(printf %02d $i).mp4"
  if [ "$n" -eq 0 ]; then F="$F[0:v]setpts=PTS-STARTPTS[c0];"
  else F="$F[$n:v]trim=start_frame=1,setpts=PTS-STARTPTS[c$n];"; fi
  n=$((n+1))
done
F="$F$(for i in $(seq 0 $((HOPS-1))); do printf "[c%d]" $i; done)concat=n=$HOPS:v=1[v]"
ffmpeg -v error -y $L -filter_complex "$F" -map "[v]" -r 24 "$O/연쇄.mp4"
NT=$(ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of csv=p=0 "$O/연쇄.mp4")
echo "  $O/연쇄.mp4  ($NT 프레임)"
