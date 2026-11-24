#!/usr/bin/env bash
# COLMAP 이 접히는 자리에서 **VGGT 는 서는가**.
#
# 🔴 왜 묻는가 (2026-11-25):
#    GT1 을 빼고 나니 COLMAP 은 A 를 자기 143장만으로도 세우지 못했다
#    (거리비 CV 129.7%, 상관 0.061, 궤적 도약 12%).
#    증분 SfM 은 반복 구조(비슷하게 생긴 복도·로비)에서 멀리 떨어진 곳을
#    같은 곳으로 보고 지도를 접는다. **재투영 오차는 접힌 상태에서도 작다.**
#    피드포워드 재구성은 프레임을 한꺼번에 보므로 이 접힘에 덜 취약할 수 있다.
#    → 같은 장면·같은 검사기로 직접 비교한다. 이것이 '이 점이 부족해서 저것을
#      써 본다' 의 근거가 된다.
set -u
PROJ="${MV_PROJ:-/mnt/c/Users/a0103/metaverse-map-bridging}"
GSPY="$HOME/miniconda3/envs/gs/bin/python"
O="$HOME/data/vggt/장면"; mkdir -p "$O"
LOG="$HOME/data/vggt_장면.log"; : > "$LOG"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

NA="${1:-128}"   # A 에서 쓸 장수 (VRAM: 약 6.5 + 0.049N GB)
NB="${2:-128}"

say "================ VGGT 장면 자립 검사 ================"
for S in A B; do
  if [ "$S" = A ]; then
    LIST=$(ls "$HOME"/data/frames_all/v1791_*.png | awk -F'v1791_' '{if (substr($2,1,5)+0 <= 143) print}' | sort)
    N=$NA
  else
    LIST=$(ls "$HOME"/data/frames_all/v1793_*.png | sort)
    N=$NB
  fi
  say "---- 장면 $S : $(echo "$LIST" | wc -l)장 중 $N장 ----"
  PYTHONPATH="$HOME/src/vggt" "$HOME/venv/vggt/bin/python" "$PROJ/코드/VGGT_포즈.py" \
     --images $LIST --out "$O/$S.npz" --max-frames "$N" >> "$LOG" 2>&1 \
     || { say "🔴 장면 $S 추론 실패"; continue; }
  grep -E "추론|솎음|인접" "$LOG" | tail -2 | sed 's/^/   /' | tee -a "$LOG"
done

say "---- 채점 ----"
{
for S in A B; do
  [ -f "$O/$S.npz" ] || continue
  echo "######## VGGT — 장면 $S ########"
  "$GSPY" "$PROJ/코드/장면_자립검사.py" --npz "$O/$S.npz" --scene "$S" --tag VGGT
  echo
done
} 2>&1 | tee "$HOME/data/표5_자립.log"
say "================ 끝 ================"
