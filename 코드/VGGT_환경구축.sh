#!/usr/bin/env bash
# VGGT 환경 — GT 없이 포즈를 얻기 위한 피드포워드 재구성 모델.
#
# 🔴 왜 필요한가 (2026-11-23 맥에서 제기):
#    지금의 포즈는 생성 프레임을 합본 DB(숨긴 GT1 실촬영 포함)에 매칭해 얻는다.
#    즉 **포즈를 정답에서 얻고 있다.** VGGT 는 이미지만 보고 포즈를 내므로
#    GT1 을 전혀 쓰지 않는 경로가 된다.
#
# 🔴 gs·ltx 와 분리된 venv 에 깐다. gs 에는 sm_120 으로 직접 빌드한 gsplat 이 있어
#    torch 가 바뀌면 그 빌드가 무효가 된다 (ltx_환경구축.sh 와 같은 이유).
#
# 🔴 conda 를 쓰지 않는다 — miniconda 기본 채널이 ToS 동의를 요구하는데
#    그것은 사용자가 할 라이선스 결정이다. python -m venv 로 격리한다.
#
# 🔴 RTX 5060 Ti 는 sm_120 이라 cu128 계열 torch 가 필요하다.
set -u
ENV="$HOME/venv/vggt"
PY="$ENV/bin/python"
BASEPY="$HOME/miniconda3/envs/gs/bin/python"
SRC="$HOME/src/vggt"
LOG="$HOME/data/vggt_setup.log"; : > "$LOG"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "== venv =="
if [ ! -x "$PY" ]; then
  mkdir -p "$HOME/venv"
  "$BASEPY" -m venv "$ENV" >> "$LOG" 2>&1
fi
say "   python $("$PY" -V 2>&1)"

say "== torch (cu128) =="
"$PY" -m pip install -q --upgrade pip >> "$LOG" 2>&1
"$PY" -m pip install -q torch torchvision \
   --index-url https://download.pytorch.org/whl/cu128 >> "$LOG" 2>&1
say "   $("$PY" -c 'import torch;print("torch",torch.__version__,"cuda",torch.version.cuda,"사용가능",torch.cuda.is_available())' 2>&1 | tail -1)"

say "== VGGT 소스 =="
mkdir -p "$HOME/src"
if [ -d "$SRC/.git" ]; then
  git -C "$SRC" pull --ff-only >> "$LOG" 2>&1 || true
else
  git clone --depth 1 https://github.com/facebookresearch/vggt.git "$SRC" >> "$LOG" 2>&1
fi
if [ ! -d "$SRC" ]; then say "   🔴 clone 실패"; exit 1; fi
say "   $(git -C "$SRC" log --oneline -1 2>/dev/null || echo '(git 정보 없음)')"

say "== 의존성 =="
if [ -f "$SRC/requirements.txt" ]; then
  # torch 는 위에서 cu128 로 깔았으므로 다시 끌어올리지 않게 거른다
  grep -viE '^(torch|torchvision)([=<>]|$)' "$SRC/requirements.txt" > "$SRC/req_notorch.txt"
  "$PY" -m pip install -q -r "$SRC/req_notorch.txt" >> "$LOG" 2>&1
fi
"$PY" -m pip install -q "huggingface_hub>=0.23" safetensors pillow numpy opencv-python-headless >> "$LOG" 2>&1
"$PY" -m pip install -q -e "$SRC" >> "$LOG" 2>&1 || say "   (-e 설치 생략 — PYTHONPATH 로 쓴다)"

say "== 확인 =="
PYTHONPATH="$SRC" "$PY" - <<'PYX' 2>&1 | tee -a "$LOG"
import torch
print("  torch", torch.__version__, "cuda", torch.cuda.is_available())
try:
    from vggt.models.vggt import VGGT
    print("  VGGT import 성공")
except Exception as e:
    print("  🔴 VGGT import 실패:", type(e).__name__, e)
PYX
say "끝"
