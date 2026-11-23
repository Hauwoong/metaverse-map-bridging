#!/usr/bin/env bash
# VGGT 로 다리 전체의 포즈를 **한 번에** 얻는다 — GT1 을 전혀 쓰지 않는다.
#
# 🔴 왜 한 번에 되는가 (실측, 코드/VGGT_포즈.py --probe):
#      8장 6.8GB / 32장 8.0GB / 64장 9.6GB / 96장 11.2GB  (프레임당 약 0.05GB)
#    16GB 안에서 130장까지 들어간다. 맥 설계는 구간별로 쪼개 포즈 그래프로
#    묶는 것이었으나, **전부 한 번에 들어가면 포즈 그래프가 필요 없다** —
#    VGGT 가 내부에서 전역으로 추론한다. 구간 분할은 안 될 때의 대안으로 남긴다.
#
# 🔴 앵커를 같이 넣는 이유:
#    A 꼬리·B 머리 실촬영이 같은 출력 안에 있으면 **그것만으로 합본 좌표계
#    Sim(3)** 이 구해진다. GT1 은 정렬에도 채점 외에는 쓰지 않는다.
#
# 🔴 순서가 곧 프레임 순서다. A 꼬리 → 생성 → B 머리 로 넣는다.
set -u
TAG="${1:-wp4}"
NA="${2:-16}"          # A 꼬리 장수
NB="${3:-16}"          # B 머리 장수
OUT=~/data/vggt/$TAG
LOG=~/data/vggt_$TAG.log; : > "$LOG"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

mkdir -p "$OUT/images"
rm -f "$OUT/images"/*.png

say "== 입력 구성 ($TAG, A 꼬리 $NA + 생성 + B 머리 $NB) =="
# A 꼬리 — v1791 의 143 이하 중 마지막 NA 장
ls ~/data/frames_all/v1791_*.png | awk -F'v1791_' '{if (substr($2,1,5)+0 <= 143) print}' \
  | sort | tail -n "$NA" | while read f; do
    cp "$f" "$OUT/images/a_$(basename "$f")"
  done
# 생성
i=0
for f in $(ls ~/data/fixreg2/$TAG/images/gen_*.png | sort); do
  cp "$f" "$OUT/images/m_$(basename "$f")"; i=$((i+1))
done
# B 머리
ls ~/data/frames_all/v1793_*.png | sort | head -n "$NB" | while read f; do
  cp "$f" "$OUT/images/z_$(basename "$f")"
done
N=$(ls "$OUT/images" | wc -l)
say "   A $(ls "$OUT/images" | grep -c '^a_') + 생성 $(ls "$OUT/images" | grep -c '^m_') + B $(ls "$OUT/images" | grep -c '^z_') = $N장"
say "   예상 VRAM 약 $(python3 -c "print('%.1f' % (6.5 + $N*0.049))" 2>/dev/null || echo '?') GB"

say "== VGGT 추론 =="
nvidia-smi --query-gpu=memory.used --format=csv,noheader | sed 's/^/   시작 VRAM /' | tee -a "$LOG"
PYTHONPATH=~/src/vggt ~/venv/vggt/bin/python ~/bin/VGGT_포즈.py \
  --images "$OUT/images/*.png" --out "$OUT/포즈.npz" --max-frames 200 \
  2>&1 | grep -vE "FutureWarning|UserWarning|warnings.warn|HF_TOKEN|^\s*$" | tee -a "$LOG"

if [ -f "$OUT/포즈.npz" ]; then
  say "   저장됨: $OUT/포즈.npz"
else
  say "   🔴 실패"
fi
say "끝"
