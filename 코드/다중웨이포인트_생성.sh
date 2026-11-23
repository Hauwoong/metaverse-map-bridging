#!/usr/bin/env bash
# 오라클 웨이포인트 N개로 다리를 만든다.  A → W1 → … → WN → B  (N+1 구간)
#
# 근거: 웨이포인트 1개가 '실제 복도 위 비율' 을 0% → 18% 로 올렸고 그 18% 가
#       웨이포인트 주변에만 몰렸다. 늘리면 그 '주변' 이 여러 곳에 생긴다.
#       그리고 각 구간이 짧아지므로 "가까운 끝점은 보간이 아니라 방향 지시로
#       작동한다"(스플랫 앵커 실험)는 조건에 들어간다.
#
# 🔴 정답(G_GT1)을 쓴다. 본 방법이 아니라 **상한(oracle)** 으로만 보고한다.
# 🔴 setsid 로 띄울 것.
set -u
PY="$HOME/venv/ltx/bin/python"
PROJ="${MV_PROJ:-/mnt/c/Users/a0103/metaverse-map-bridging}"   # 프로젝트 폴더 (윈도우 C:\Users\a0103\metaverse-map-bridging). 다른 위치면 MV_PROJ 로 지정
SRC="$PROJ/코드/끝점조건_생성.py"
E="$HOME/data/끝점"
EMB="$HOME/data/생성/프롬프트_임베딩.pt"
export HF_HOME="$HOME/.cache/huggingface"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

N="${1:-4}"            # 웨이포인트 개수
NF="${2:-97}"          # 구간당 프레임 (8k+1)
TAG="${3:-}"           # 비면 wp<N>, 주면 wp_<TAG> 디렉터리를 쓴다
if [ -n "$TAG" ]; then WP="$E/wp_$TAG"; O="$HOME/data/생성/wp_$TAG"
else                   WP="$E/wp$N";    O="$HOME/data/생성/wp$N"; fi
mkdir -p "$O"
[ -f "$EMB" ] || { echo "!! 프롬프트 임베딩 없음"; exit 1; }

# 구간 목록: A→W1, W1→W2, …, WN→B
PTS=("$E/I_A_out.png")
for i in $(seq 1 "$N"); do
  [ -f "$WP/W$i.png" ] || { echo "!! $WP/W$i.png 없음 — 코드/웨이포인트_다중.sh 먼저"; exit 1; }
  PTS+=("$WP/W$i.png")
done
PTS+=("$E/I_B_in.png")
NSEG=$(( ${#PTS[@]} - 1 ))
echo "== 웨이포인트 $N개, 구간 $NSEG개, 구간당 $NF 프레임 =="

for s in $(seq 1 "$NSEG"); do
  A="${PTS[$((s-1))]}"; B="${PTS[$s]}"
  OUT="$O/seg$(printf %02d $s).mp4"
  echo "########## 구간 $s/$NSEG : $(basename "$A") -> $(basename "$B") ##########"
  "$PY" "$SRC" --first "$A" --last "$B" --out "$OUT" \
      --embeds "$EMB" --num-frames "$NF" || exit 1
done

echo "########## 이어붙이기 ##########"
# 각 구간의 첫 프레임은 직전 구간의 마지막과 같은 시점이므로 1프레임씩 버린다.
L=""; F=""; n=0
for s in $(seq 1 "$NSEG"); do
  L="$L -i $O/seg$(printf %02d $s).mp4"
  if [ "$n" -eq 0 ]; then F="$F[0:v]setpts=PTS-STARTPTS[c0];"
  else F="$F[$n:v]trim=start_frame=1,setpts=PTS-STARTPTS[c$n];"; fi
  n=$((n+1))
done
F="$F$(for i in $(seq 0 $((NSEG-1))); do printf "[c%d]" $i; done)concat=n=$NSEG:v=1[v]"
ffmpeg -nostdin -v error -y $L -filter_complex "$F" -map "[v]" -r 24 "$O/다리.mp4"
NT=$(ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of csv=p=0 "$O/다리.mp4")
echo "  $O/다리.mp4  ($NT 프레임, $(echo "scale=1; $NT/24" | bc)초)"
