#!/usr/bin/env bash
# ⑤ 끝점 조건 영상 생성. 텍스트 인코딩과 확산을 '다른 프로세스'로 나눠 돌린다.
#
# 🔴 왜 나누는가.
#    LTX 0.9.5 = transformer 2B + T5-XXL 텍스트 인코더. bf16 합 ≈ 14GB 다.
#    enable_model_cpu_offload 는 이 전체를 시스템 RAM 에 상주시키는데 WSL 상한이
#    15.6GB(호스트 31.2GB 의 50%)라 숨 쉴 틈이 없다. 실제로 한 번 통째로 날아갔다 —
#    트레이스백도 없이 WSL VM 자체가 내려갔다.
#    프롬프트는 생성 중에 바뀌지 않으므로 먼저 인코딩해 저장하고 프로세스를 끝내면
#    그 9GB 가 통째로 반환된다. 확산 단계는 ~5GB 로 돈다.
#
# 🔴 setsid 로 띄울 것 (코드/3dgs_학습.sh 주석 참조).
set -u
PY="$HOME/venv/ltx/bin/python"
SRC="/mnt/c/Users/a0103/녹음/메타버스_기말프로젝트/코드/끝점조건_생성.py"
E="$HOME/data/끝점"
O="$HOME/data/생성"
mkdir -p "$O"

export HF_HOME="$HOME/.cache/huggingface"   # /mnt/c 는 32GB 밖에 안 남았고 9P 라 느리다
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

VARIANT="${1:-render}"      # render = 3DGS 렌더본, photo = 실촬영 원본
shift || true
if [ "$VARIANT" = "photo" ]; then
  FIRST="$E/I_A_out_원본.png"; LAST="$E/I_B_in_원본.png"; TAG="photo"
else
  FIRST="$E/I_A_out.png";      LAST="$E/I_B_in.png";      TAG="render"
fi
EMB="$O/프롬프트_임베딩.pt"

echo "== 조건 프레임 =="
ls -l "$FIRST" "$LAST"

if [ ! -f "$EMB" ]; then
  echo "== 1단계: 텍스트 인코딩 (T5-XXL 만 적재) =="
  "$PY" "$SRC" --first "$FIRST" --last "$LAST" --out "$O/unused.mp4" \
       --encode-only --embeds "$EMB" "$@" || { echo "!! 1단계 실패"; exit 1; }
  free -g | awk '/Mem:/{print "  1단계 후 RAM 사용 "$3"GB / "$2"GB"}'
else
  echo "== 1단계 건너뜀 (임베딩 있음: $EMB) =="
fi

echo "== 2단계: 확산 (텍스트 인코더 미적재) =="
"$PY" "$SRC" --first "$FIRST" --last "$LAST" \
     --out "$O/bridge_$TAG.mp4" --embeds "$EMB" "$@"
