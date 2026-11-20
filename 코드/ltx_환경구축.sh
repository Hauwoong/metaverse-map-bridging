#!/usr/bin/env bash
# ⑤ 끝점 조건 영상 생성용 환경. 3DGS 환경(gs)과 '분리된' conda 환경에 깐다.
#
# 🔴 gs 환경에 같이 깔지 않는 이유: torch 2.11+cu128 과 그 위에 sm_120 으로 빌드한
#    gsplat_cuda.so 를 맞추는 데 여러 번 실패했다. diffusers/transformers 가 torch 를
#    끌어올리면 그 빌드가 통째로 무효가 된다. 디스크는 911GB 남았다 — 격리가 싸다.
#
# 🔴 모델은 LTX-Video 0.9.5 (2B). LTX-2(19B)·0.9.7(13B) 이 아니다.
#    LTX-2 의 FLF2V 예제는 enable_sequential_cpu_offload 를 쓰는데, 이는 모델을
#    시스템 RAM 에 상주시키는 방식이다. bf16 19B ≈ 38GB 로 WSL 상한 15.6GB 는 물론
#    호스트 RAM 31GB 로도 안 된다. 0.9.5 는 2B 라 16GB VRAM 안에서 돈다.
#    0.9.5 부터 조건 이미지를 여러 장 받으므로 끝점(처음+끝) 조건이 가능하다.
#
# 🔴 캐시를 WSL 쪽(ext4)에 둔다. /mnt/c 는 32GB 밖에 안 남았고 9P 라 느리다.
set -eu
# 🔴 conda 로 환경을 만들지 않는다. miniconda 기본 채널(repo.anaconda.com)이
#    Terms of Service 동의를 요구하는데, 그건 사용자가 할 라이선스 결정이다.
#    python -m venv 로 만들면 pip 패키지가 완전히 격리되고 conda 를 전혀 거치지 않는다.
ENV="$HOME/venv/ltx"
PY="$ENV/bin/python"
BASEPY="$HOME/miniconda3/envs/gs/bin/python"   # 3.11 인터프리터만 빌려 쓴다

if [ ! -x "$PY" ]; then
  echo "== venv 생성: $ENV =="
  mkdir -p "$HOME/venv"
  "$BASEPY" -m venv "$ENV"
fi
echo "== python: $("$PY" -V) =="

echo "== torch (cu128) =="
"$PY" -m pip install -q --upgrade pip
"$PY" -m pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu128

echo "== diffusers 및 의존성 =="
"$PY" -m pip install -q   "diffusers>=0.33" transformers accelerate safetensors sentencepiece protobuf   imageio imageio-ffmpeg opencv-python-headless pillow

echo "== 확인 =="
"$PY" - <<'PYX'
import torch, diffusers, transformers
print("  torch       ", torch.__version__, "cuda", torch.version.cuda, "사용가능", torch.cuda.is_available())
if torch.cuda.is_available():
    print("  GPU         ", torch.cuda.get_device_name(0),
          "%.1f GB" % (torch.cuda.get_device_properties(0).total_memory / 1024**3))
print("  diffusers   ", diffusers.__version__)
print("  transformers", transformers.__version__)
from diffusers import LTXConditionPipeline
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
import inspect
f = getattr(LTXVideoCondition, "__dataclass_fields__", None)
print("  LTXVideoCondition 필드:", list(f) if f else str(inspect.signature(LTXVideoCondition.__init__)))
PYX
echo "끝"
