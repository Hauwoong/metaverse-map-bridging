#!/usr/bin/env bash
# v3: DL3DV 장면 쌍(A = 49 숲 비탈 산길, B = 1 마른 평원)을 splatfacto 로 학습한다.
#
# 데이터: ~/data/dl3dv/<batch>/<hash>/  (transforms.json + images_4/, 960x540)
#   transforms.json 의 file_path 는 images/... 이지만 nerfstudio-data 파서에
#   --downscale-factor 4 를 주면 images_4/ 를 읽는다. 축소본을 따로 만들 필요 없다.
#
# 실행법은 3dgs_학습.sh 와 같다 (/mnt/c 에서 직접 돌리지 않는다, setsid 로 떼어 놓는다):
#    cp 코드/v3_3dgs_학습.sh 코드/환경.sh ~/bin/ && sed -i 's/\r$//' ~/bin/*.sh
#    wsl.exe -d Ubuntu-24.04 -- bash -lc "setsid nohup bash ~/bin/v3_3dgs_학습.sh \
#        > ~/data/v3_train_outer.log 2>&1 < /dev/null & disown"
#
# 이어받기(--load-dir)는 쓰지 않는다 — 이유는 3dgs_학습.sh 주석 참고. 중단되면 처음부터.
set -u
source "$(dirname "$0")/환경.sh"

DL=~/data/dl3dv
RUN=~/data/runs_v3
LOG=~/data/v3_train.log
TRIES=3
declare -A SCENE=(
  [A]=4K/1dd87b9e7aecf331960bf0a702a7110a3599fe8e919008f68dbb159cfdd31784   # 49 숲 비탈 산길
  [B]=3K/721fecd89f6e29527dafb19cbb05a925ce91df44c2ae4778116a08d20c6ab8b6   # 1  마른 평원
)
mkdir -p "$RUN"
ts() { date +%H:%M:%S; }
say() { echo "[$(ts)] $*" | tee -a "$LOG"; }

say "================ v3 3DGS 학습 시작 ================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$LOG"

for S in A B; do
  D="$DL/${SCENE[$S]}"
  N=$(ls "$D/images_4" | wc -l)
  OUT="$RUN/G_$S/splatfacto/run1"
  echo "" | tee -a "$LOG"
  say "---- G_$S  (프레임 $N, 960x540, $D) ----"
  T0=$SECONDS
  RC=1
  for T in $(seq 1 $TRIES); do
    rm -rf "$OUT"
    say "  시도 $T: 처음부터"
    ns-train splatfacto \
      --data "$D" \
      --output-dir "$RUN" \
      --experiment-name "G_$S" \
      --timestamp run1 \
      --vis tensorboard \
      --max-num-iterations 30000 \
      --steps-per-save 5000 \
      --steps-per-eval-all-images 30000 \
      --steps-per-eval-image 5000 \
      --logging.local-writer.enable False \
      nerfstudio-data --downscale-factor 4 \
      >> "$LOG" 2>&1 < /dev/null
    RC=$?
    [ $RC -eq 0 ] && break
    say "  !! 시도 $T 종료 코드 $RC"
    sleep 5
  done
  say "G_$S 종료 코드 $RC, $((SECONDS-T0))초"
  if [ $RC -ne 0 ]; then
    say "!! G_$S 최종 실패 ($TRIES 회 시도)"
    tail -25 "$LOG"
    exit $RC
  fi
done

echo "" | tee -a "$LOG"
say "================ 결과 ================"
for S in A B; do
  say "  G_$S  $(ls "$RUN/G_$S/splatfacto/run1/nerfstudio_models/"*.ckpt 2>/dev/null | sort | tail -1)"
done
say "끝"
