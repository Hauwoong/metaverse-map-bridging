#!/usr/bin/env bash
# 케이스 1 (선택지 ⓐ) 의 세 장면을 splatfacto 로 학습한다.
#   G_A    끝점 장면 A (1791 @0-57s)
#   G_GT1  숨길 중간 구간 정답 (1791 @64-114s) — 평가에만 쓴다
#   G_B    끝점 장면 B (1793)
# 데이터셋은 코드/구간_분할.py --write 로 만든다.
#
# 🔴 실행법 — 반드시 이렇게 띄운다.
#    cp 코드/3dgs_학습.sh 코드/환경.sh ~/bin/ && sed -i 's/\r$//' ~/bin/*.sh
#    wsl.exe -d Ubuntu-24.04 -- bash -lc "setsid nohup bash ~/bin/3dgs_학습.sh \
#        > ~/data/train_outer.log 2>&1 < /dev/null & disown"
#    이유 두 가지:
#    (1) /mnt/c 에서 직접 돌리면 bash 가 스크립트를 9P 로 조금씩 읽는 도중 핸들이 끊긴다.
#    (2) setsid 로 WSL 안에 떼어놓지 않으면, 호스트 메모리가 빡빡할 때 윈도 쪽
#        wsl.exe 가 죽임을 당하면서 WSL 세션 프로세스 트리가 통째로 끌려간다.
#        그러면 Linux 쪽에는 traceback 도 dmesg OOM 도 아무 흔적이 남지 않는다.
#        이 프로젝트에서 네 번을 이것으로 날리고 나서야 알았다.
#
# 🔴 이어받기(--load-dir)를 쓰지 않는다.
#    splatfacto 체크포인트는 가우시안 N개를 복원하지만 gsplat 밀도화 전략의
#    상태 버퍼(state["grad2d"], state["count"])는 그에 맞게 복원되지 않는다.
#    그래서 복구 후 첫 밀도화 스텝에서
#      gsplat/strategy/ops.py:104  torch.cat([p, p[sel]])
#      torch.AcceleratorError: CUDA error: device-side assert triggered
#      IndexKernel.cu:111  index out of bounds
#    로 죽는다. 4,000 스텝에서 복구를 네 번 시도해 네 번 같은 자리에서 죽었다.
#    장면당 순수 계산 시간이 20분 남짓이므로, 중단되면 그냥 처음부터 돌린다.
set -u
source "$(dirname "$0")/환경.sh"

SC=~/data/scenes
RUN=~/data/runs
LOG=~/data/train.log
TRIES=3                  # 중단 시 재시도 (이어받기가 아니라 '처음부터' 다시)
mkdir -p "$RUN"
ts() { date +%H:%M:%S; }
say() { echo "[$(ts)] $*" | tee -a "$LOG"; }

say "================ 3DGS 학습 시작 ================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$LOG"

# 960x540 축소본을 미리 만든다.
# 없으면 nerfstudio 가 "지금 축소할까요?" 를 대화형으로 묻는데, 배치 실행에는
# stdin 이 없어 EOFError 로 죽는다 (rich.prompt.Confirm.ask).
for S in A GT1 B; do
  SRC="$SC/$S/images"; DST="$SC/$S/images_2"
  NEED=$(ls "$SRC" | wc -l)
  if [ "$(ls "$DST" 2>/dev/null | wc -l)" -ne "$NEED" ]; then
    say "$S 축소본 생성 ($NEED장 -> 960x540)"
    mkdir -p "$DST"
    for f in "$SRC"/*.png; do
      ffmpeg -v error -y -i "$f" -vf scale=960:540 "$DST/$(basename "$f")"
    done
  fi
  say "$S 축소본 $(ls "$DST" | wc -l)장 준비됨"
done

for S in A GT1 B; do
  N=$(ls "$SC/$S/images" | wc -l)
  OUT="$RUN/G_$S/splatfacto/run1"
  echo "" | tee -a "$LOG"
  say "---- G_$S  (프레임 $N, 960x540) ----"
  T0=$SECONDS
  RC=1
  for A in $(seq 1 $TRIES); do
    # 이어받기가 불가능하므로(위 주석) 남아 있는 부분 결과를 지우고 처음부터.
    rm -rf "$OUT"
    say "  시도 $A: 처음부터 (이전 부분 결과 삭제)"
    ns-train splatfacto \
      --data "$SC/$S" \
      --output-dir "$RUN" \
      --experiment-name "G_$S" \
      --timestamp run1 \
      --vis tensorboard \
      --max-num-iterations 30000 \
      --steps-per-save 5000 \
      --steps-per-eval-all-images 30000 \
      --steps-per-eval-image 5000 \
      --logging.local-writer.enable False \
      colmap --downscale-factor 2 \
      >> "$LOG" 2>&1 < /dev/null
    RC=$?
    [ $RC -eq 0 ] && break
    say "  !! 시도 $A 종료 코드 $RC"
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
for S in A GT1 B; do
  D="$RUN/G_$S/splatfacto/run1"
  say "  G_$S  $(ls "$D/nerfstudio_models/"*.ckpt 2>/dev/null | sort | tail -1)"
done
say "끝"
