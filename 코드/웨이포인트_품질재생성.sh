#!/usr/bin/env bash
# 웨이포인트를 **복원 품질로 고른 자리**에 놓고 다리를 다시 만든다.
#
# 🔴 왜 다시 만드는가 (2026-11-25):
#    wp4 는 GT1 을 균등 분할해 웨이포인트를 골랐다. 그 중 W1 이 **복도 입구의
#    저텍스처 구간**에 떨어졌고, 거기서 G_GT1 을 렌더하니 형체 없는 뭉갬이 나왔다.
#    끝점 조건이 뭉개지니 LTX 가 만든 구간도 뭉개졌다:
#        생성 프레임 SIFT 특징 65개 (실촬영 4201, 성공한 생성분 1236)
#    그 20장(gen 6~25)이 등록에 실패해 **포즈 사슬이 공유트랙 0 으로 끊겼고**,
#    거기서 A↔B 상대 회전이 풀려 **108.9° 오차**가 됐다.
#    (A↔B 직접 매칭은 강한 쌍이 0개였다 — 거짓 매칭 때문이 아니라 정보가 없어서다)
#
# 🔴 두 가지를 같이 고친다:
#    ① 웨이포인트를 관측 3D 점 ≥1300 인 자리에서만 고른다 (저텍스처 구간 회피)
#    ② 구간마다 **길이에 비례해** 프레임을 준다. wp4 는 길이와 무관하게 97장씩 줬고
#       가장 긴 구간(62장 분량)이 뭉갰다.
#
# ⚠️ 웨이포인트는 GT1 에서 나오므로 이 다리는 **오라클(상한)** 이다. 본 방법이 아니다.
#    (포즈는 GT없는등록.sh 로 따로 구한다 — 그쪽이 표 5 의 본 수치다)
#
# 사용: setsid bash 코드/웨이포인트_품질재생성.sh [N] [태그]
set -u
PROJ="${MV_PROJ:-/mnt/c/Users/a0103/metaverse-map-bridging}"
PY="$HOME/venv/ltx/bin/python"
GSPY="$HOME/miniconda3/envs/gs/bin/python"
SRC="$PROJ/코드/끝점조건_생성.py"
E="$HOME/data/끝점"
EMB="$HOME/data/생성/프롬프트_임베딩.pt"
RUN="$HOME/data/runs"
export HF_HOME="$HOME/.cache/huggingface"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

N="${1:-5}"
TAG="${2:-q$N}"
WP="$E/wp_$TAG"; O="$HOME/data/생성/wp_$TAG"
LOG="$HOME/data/생성_$TAG.log"; : > "$LOG"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
mkdir -p "$WP" "$O"

[ -f "$EMB" ] || { log "🔴 프롬프트 임베딩 없음: $EMB"; exit 1; }
[ -x "$PY" ]  || { log "🔴 LTX venv 없음 — 코드/ltx_환경구축.sh 먼저"; exit 1; }

log "================ 품질 기준 웨이포인트 재생성 ($TAG) ================"

# 0) G_GT1 렌더 (없으면 한 번만)
if [ ! -d "$E/G_GT1" ]; then
  log "G_GT1 데이터셋 렌더 (한 번만)"
  ns-render dataset --load-config "$RUN/G_GT1/splatfacto/run1/config.yml" \
    --output-path "$E/G_GT1" --split train+test --rendered-output-names rgb \
    < /dev/null >> "$LOG" 2>&1
fi

# 1) 🔴 **렌더 자체의 품질**을 잰다. SfM 관측점 수로 거르면 안 된다 —
#    W2(205) 는 1326점으로 통과했는데 렌더는 완전히 뭉갰다.
#    SfM 점 수는 '거기서 특징이 잡혔는가' 일 뿐 '3DGS 가 그릴 수 있는가' 가 아니다.
Q="$E/렌더품질.txt"
[ -f "$Q" ] || "$PY" "$PROJ/코드/앵커_렌더품질.py" --out "$Q" >> "$LOG" 2>&1
[ -s "$Q" ] || { log "🔴 렌더 품질 측정 실패"; exit 1; }

# 2) 자리 고르기
rm -f "$WP/목록.txt" "$WP/프레임수.txt"
"$GSPY" "$PROJ/코드/웨이포인트_품질선별.py" --n "$N" --quality "$Q" --thr 0.5 \
   --out "$WP/목록.txt" 2>&1 | tee -a "$LOG" | sed -n '/쓸 수 있는 구간/,$p'
[ -s "$WP/목록.txt" ] || { log "🔴 선별 실패"; exit 1; }
read -r -a NF < "$WP/프레임수.txt"
log "구간별 프레임 수: ${NF[*]}"

# 3) 그 프레임의 렌더를 앵커로 꺼낸다
i=0
while read -r F; do
  i=$((i+1)); B="${F%.*}"          # .png 도 .jpg 도 벗긴다
  S=$(find "$E/G_GT1" -type f \( -name "$B.jpg" -o -name "$B.png" \) | head -1)
  # 🔴 -nostdin 필수. 없으면 ffmpeg 이 while 의 stdin(목록)을 먹어 다음 줄이 사라진다.
  [ -n "$S" ] && ffmpeg -nostdin -v error -y -i "$S" "$WP/W$i.png" \
              || { log "🔴 W$i 못 찾음 ($B)"; exit 1; }
  log "  W$i <- $B  특징비 $(awk -v b="$B" '$1 ~ b {print $2}' "$Q")"
done < "$WP/목록.txt"
NW=$i

# 3) 구간 생성 — A_out → W1 → … → WN → B_in
PTS=("$E/I_A_out.png"); for k in $(seq 1 $NW); do PTS+=("$WP/W$k.png"); done
PTS+=("$E/I_B_in.png")
NSEG=$(( ${#PTS[@]} - 1 ))
log "구간 $NSEG개 생성 시작"
for s in $(seq 1 "$NSEG"); do
  A="${PTS[$((s-1))]}"; B="${PTS[$s]}"
  FR="${NF[$((s-1))]:-97}"
  OUT="$O/seg$(printf %02d $s).mp4"
  log "---- 구간 $s/$NSEG  $(basename "$A") → $(basename "$B")  ${FR}프레임 ----"
  if ! "$PY" "$SRC" --first "$A" --last "$B" --out "$OUT" \
        --embeds "$EMB" --num-frames "$FR" >> "$LOG" 2>&1; then
    # 🔴 긴 구간은 VRAM 에서 떨어질 수 있다. 97 로 줄여 한 번 더 시도한다.
    log "   ⚠️ ${FR}프레임 실패 — 97 로 재시도"
    "$PY" "$SRC" --first "$A" --last "$B" --out "$OUT" \
        --embeds "$EMB" --num-frames 97 >> "$LOG" 2>&1 \
      || { log "🔴 구간 $s 실패"; exit 1; }
  fi
done

# 4) 이어붙이기 — 각 구간의 첫 프레임은 직전 구간의 마지막과 같은 시점이라 버린다
log "이어붙이기"
L=""; F=""; n=0
for s in $(seq 1 "$NSEG"); do
  L="$L -i $O/seg$(printf %02d $s).mp4"
  if [ "$n" -eq 0 ]; then F="$F[0:v]setpts=PTS-STARTPTS[c0];"
  else F="$F[$n:v]trim=start_frame=1,setpts=PTS-STARTPTS[c$n];"; fi
  n=$((n+1))
done
F="$F$(for i in $(seq 0 $((NSEG-1))); do printf "[c%d]" $i; done)concat=n=$NSEG:v=1[v]"
ffmpeg -nostdin -v error -y $L -filter_complex "$F" -map "[v]" -r 24 "$O/다리.mp4"
NT=$(ffprobe -v error -select_streams v:0 -count_frames \
     -show_entries stream=nb_read_frames -of csv=p=0 "$O/다리.mp4")
log "완성: $O/다리.mp4  ($NT 프레임)"
log "================ 끝 ================"
