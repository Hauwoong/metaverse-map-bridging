#!/usr/bin/env bash
# GT 없는 포즈 — **2단계 증분 등록.** 한 번에 다 풀지 않는다.
#
# 🔴 왜 바꾸는가 (2026-11-25):
#    다리를 다시 만들어 등록률은 100% 가 됐는데(A 143, B 157, 생성 111 전부),
#    **구조가 접혔다**:
#        A   인접 0.3790 (합본 0.0195 의 19배)  최대 18.73
#        gen 인접 0.0000  최대 145.86
#        B   157장이 크기 0.0003 — 한 점으로 붕괴
#        재투영 오차 0.734px ← COLMAP 은 '잘 됐다' 고 말한다
#    복도는 어디를 봐도 비슷하다. 생성 프레임이 많아지자 **멀리 떨어진 복도 위치끼리
#    서로 매칭**돼 지도가 접혔다. 잔차는 접힌 상태에서도 작다 —
#    🔴 **재투영 오차는 접힘을 잡지 못한다.** 또 같은 함정이다.
#
# 🔴 처방: 신뢰할 수 있는 블록부터 세우고 거기서 뻗어나간다.
#    A 실촬영만으로 먼저 재구성하면 그 안에는 접힐 거리가 없다(한 번에 찍은 연속 영상).
#    그 다음 다리와 B 를 **이어서** 등록한다. 증분 매퍼는 이미 선 구조를 기준으로
#    새 프레임을 받으므로, 엉뚱한 곳에 붙는 프레임은 받아들여지지 않는다.
#
#    ⚠️ 이것은 '**A·B 를 합본 포즈로 고정**' 과 다르다. 그건 GT1 을 거쳐 정해진
#       값이라 오라클이다. 여기서는 **A 자신의 영상만으로** A 를 세운다. GT1 을 쓰지 않는다.
#
# 사용: bash GT없는등록2.sh [태그]      (GT없는등록.sh 가 만든 DB 를 재사용한다)
set -u
C="$HOME/miniconda3/envs/colmap/bin/colmap"
TAG="${1:-r5}"
D="$HOME/data/nogt/$TAG"
LOG="$HOME/data/nogt2_$TAG.log"; : > "$LOG"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

[ -f "$D/db.db" ] || { log "🔴 DB 없음: $D/db.db — GT없는등록.sh 를 먼저"; exit 1; }
log "================ 2단계 증분 등록 ($TAG) ================"
log "DB 재사용: 특징·매칭은 그대로, 매핑만 다시 한다"

# ---- 1단계: A 실촬영만으로 기준 블록 ----
rm -rf "$D/sparseA"; mkdir -p "$D/sparseA"
ls "$D/images" | grep '^v1791_' > "$D/A목록.txt"
log "1단계: A $(wc -l < "$D/A목록.txt")장만으로 재구성"
"$C" mapper --database_path "$D/db.db" --image_path "$D/images" \
   --output_path "$D/sparseA" --Mapper.image_list_path "$D/A목록.txt" \
   --Mapper.ba_use_gpu 1 --Mapper.num_threads 8 >> "$LOG" 2>&1

BA=""; BN=0
for M in "$D"/sparseA/*/; do
  [ -d "$M" ] || continue
  "$C" model_converter --input_path "$M" --output_path "$M" --output_type TXT >/dev/null 2>&1
  K=$(grep -cE '^[0-9]+ ' "$M/images.txt" 2>/dev/null || echo 0)
  [ "$K" -gt "$BN" ] && { BN=$K; BA="$M"; }
done
[ -n "$BA" ] || { log "🔴 A 재구성 실패"; exit 1; }
log "   A 블록: $((BN/2))장 등록 → $BA"
"$C" model_analyzer --path "$BA" 2>&1 | sed 's/.*\] //' \
   | grep -E "^(Registered|Points|Mean)" | sed 's/^/     /' | tee -a "$LOG"

# ---- 2단계: 다리와 B 를 이어서 등록 ----
# 🔴 A 를 고정하지 **않는다**. 고정하면 A 의 작은 오차가 전부 다리로 떠넘겨진다.
#    대신 A 가 이미 세운 구조가 초기값이 되어 접힘을 막는다.
rm -rf "$D/sparse2"; mkdir -p "$D/sparse2"
log "2단계: A 블록에서 이어서 생성·B 등록"
"$C" mapper --database_path "$D/db.db" --image_path "$D/images" \
   --input_path "$BA" --output_path "$D/sparse2" \
   --Mapper.ba_use_gpu 1 --Mapper.num_threads 8 >> "$LOG" 2>&1

OUT="$D/sparse2"
[ -f "$OUT/images.bin" ] || OUT=$(ls -d "$D"/sparse2/*/ 2>/dev/null | head -1)
[ -n "$OUT" ] && [ -e "$OUT/images.bin" ] || { log "🔴 2단계 출력 없음"; exit 1; }
"$C" model_converter --input_path "$OUT" --output_path "$OUT" --output_type TXT >/dev/null 2>&1
echo "$OUT" > "$D/best2.txt"

A1=$(grep -oE 'v1791_[0-9]+\.png' "$OUT/images.txt" | sort -u | wc -l)
B1=$(grep -oE 'v1793_[0-9]+\.png' "$OUT/images.txt" | sort -u | wc -l)
G1=$(grep -oE 'gen_[0-9]+\.png' "$OUT/images.txt" | sort -u | wc -l)
log "---- 결과: A $A1, B $B1, 생성 $G1 → $OUT ----"
"$C" model_analyzer --path "$OUT" 2>&1 | sed 's/.*\] //' \
   | grep -E "^(Registered|Points|Mean)" | sed 's/^/    /' | tee -a "$LOG"
log "🔴 재투영 오차는 접힘을 잡지 못한다. 반드시 기하 분해로 확인할 것."
log "================ 끝 ================"
