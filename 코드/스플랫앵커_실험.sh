#!/usr/bin/env bash
# ⑤⁗ 스플랫 렌더를 '끝점 조건' 으로 써서 전진을 강제할 수 있는지 본다.
#
# 배경: 연쇄 생성의 소화기가 24초 내내 같은 자리에 머물렀다 — 모델이 평행이동을
#   거의 하지 않고 '복도스러운 것' 을 재생성하고 있었다(러닝머신). 거리를 알려줄
#   통로가 없는 것이 원인이다.
#
# 아이디어: 경로 10% 까지는 G_A 스플랫이 깨끗하다. 그 지점의 렌더를 **끝점 조건**으로
#   걸면 거리가 조건에 박힌다 — 모델은 거기로 가는 것 말고 선택지가 없다.
#
# 🔴 정답(G_GT1)을 쓰지 않는다. G_A 는 A 자신의 스캔이므로 전제를 깨지 않는다.
#
# 검증: 2.5/5/7.5% 지점의 스플랫 렌더를 '참조' 로 두고 생성 중간 프레임과 비교한다.
#   진짜 전진이면 맞아떨어지고, 러닝머신이면 어긋난다.
set -u
PY="$HOME/venv/ltx/bin/python"
SRC="/mnt/c/Users/a0103/녹음/메타버스_기말프로젝트/코드/끝점조건_생성.py"
E="$HOME/data/끝점"; T="$HOME/data/궤적"; O="$HOME/data/스플랫앵커"
EMB="$HOME/data/생성/프롬프트_임베딩.pt"
mkdir -p "$O"
export HF_HOME="$HOME/.cache/huggingface"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 궤적 영상(161프레임, A→B 전체)에서 경로 비율별 프레임을 꺼낸다.
# 프레임 f 가 경로의 f/160 에 해당한다.
grab () {  # grab <프레임번호> <출력>
  ffmpeg -v error -y -i "$T/G_A.mp4" -vf "select=eq(n\,$1)" -frames:v 1 -update 1 "$2"
}
echo "== 참조 프레임 추출 (G_A 스플랫) =="
for p in 0:000 4:025 8:050 12:075 16:100; do
  f=${p%%:*}; tag=${p##*:}
  grab "$f" "$O/참조_$tag.png"
  echo "  경로 $(echo "scale=2; $f/1.6" | bc)% → 참조_$tag.png"
done

NF=97
echo
echo "########## (1) 끝점 조건 있음 — 스플랫 10% 를 목표로 ##########"
"$PY" "$SRC" --first "$E/I_A_out.png" --last "$O/참조_100.png" \
  --out "$O/hop_조건있음.mp4" --embeds "$EMB" --num-frames "$NF" || exit 1

echo
echo "########## (2) 대조군 — 끝점 조건 없음 (I2V) ##########"
"$PY" "$SRC" --first "$E/I_A_out.png" \
  --out "$O/hop_조건없음.mp4" --embeds "$EMB" --num-frames "$NF" || exit 1

echo
echo "== 생성 중간 프레임 추출 =="
for v in 조건있음 조건없음; do
  for p in 0:000 24:025 48:050 72:075 96:100; do
    f=${p%%:*}; tag=${p##*:}
    ffmpeg -v error -y -i "$O/hop_$v.mp4" -vf "select=eq(n\,$f)" -frames:v 1 -update 1 "$O/${v}_$tag.png"
  done
done
echo "끝. 결과: $O"
