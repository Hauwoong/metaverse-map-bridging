#!/usr/bin/env bash
# ⑤′ 웨이포인트 분할 생성. A→M, M→B 두 클립을 만들고 이어붙인다.
#
# 사용법:  bash ~/bin/분할생성_실행.sh {oracle|recursive} [추가 인자]
#
# 🔴 oracle 은 숨긴 정답(G_GT1)에서 웨이포인트를 뽑는다. 본 방법이 아니라
#    "분할하면 블렌딩이 사라지는가 = 길이 제약이 원인인가" 를 가리는 진단이다.
#    recursive 는 1차 생성물에서 뽑으므로 정답을 쓰지 않는다 — 이쪽이 제안 가능한 방법.
#
# 🔴 setsid 로 띄울 것. 호스트 메모리 압박 시 배경 wsl.exe 가 거둬지면
#    WSL 프로세스 트리가 통째로 끌려간다 (코드/3dgs_학습.sh 주석 참조).
set -u
PY="$HOME/venv/ltx/bin/python"
SRC="/mnt/c/Users/a0103/녹음/메타버스_기말프로젝트/코드/끝점조건_생성.py"
E="$HOME/data/끝점"
O="$HOME/data/생성"
EMB="$O/프롬프트_임베딩.pt"
mkdir -p "$O"
export HF_HOME="$HOME/.cache/huggingface"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

V="${1:-oracle}"; shift || true
case "$V" in
  oracle)    M="$E/I_M_oracle.png" ;;
  recursive) M="$E/I_M_recursive.png" ;;
  *) echo "oracle 또는 recursive 를 지정할 것"; exit 1 ;;
esac
[ -f "$M" ] || { echo "!! 웨이포인트 없음: $M — 코드/웨이포인트_준비.sh 먼저"; exit 1; }
[ -f "$EMB" ] || { echo "!! 프롬프트 임베딩 없음 — 코드/생성_실행.sh 먼저"; exit 1; }

A="$E/I_A_out.png"; B="$E/I_B_in.png"
C1="$O/split_${V}_1.mp4"; C2="$O/split_${V}_2.mp4"; OUT="$O/split_${V}.mp4"

echo "===== 1/2 구간: A -> M ($V) ====="
"$PY" "$SRC" --first "$A" --last "$M" --out "$C1" --embeds "$EMB" "$@" || exit 1

echo "===== 2/2 구간: M -> B ($V) ====="
"$PY" "$SRC" --first "$M" --last "$B" --out "$C2" --embeds "$EMB" "$@" || exit 1

echo "===== 이어붙이기 ====="
# 두 클립의 경계에서 웨이포인트 프레임이 두 번 나오므로 둘째 클립의 첫 프레임을 버린다.
ffmpeg -v error -y -i "$C1" -i "$C2" \
  -filter_complex "[1:v]trim=start_frame=1,setpts=PTS-STARTPTS[b];[0:v][b]concat=n=2:v=1[v]" \
  -map "[v]" -r 24 "$OUT"
N=$(ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of csv=p=0 "$OUT")
echo "  $OUT  ($N 프레임, $(echo "scale=1; $N/24" | bc)초)"
ls -l "$C1" "$C2" "$OUT"
