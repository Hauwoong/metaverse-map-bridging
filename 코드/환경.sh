#!/usr/bin/env bash
# gsplat/nerfstudio 를 돌릴 때 쓰는 환경. 반드시 `source 코드/환경.sh` 로 불러온다.
#
# 🔴 여기 있는 값은 전부 "한 번 당해서" 박힌 것이다. 하나라도 빠지면 조용히 깨진다.
#
#  conda activate 를 쓰지 않는다
#     envs/gs/etc/conda/activate.d/~cuda-nvcc_activate.sh 가 TORCH_CUDA_ARCH_LIST 가
#     비어 있을 때 "5.0;...;10.0;10.1;12.0+PTX" 를 박는데 torch 2.11 은 10.1 을 모른다:
#       ValueError: Unknown CUDA arch (10.1) or GPU not supported
#
#  TORCH_CUDA_ARCH_LIST=12.0
#     RTX 5060 Ti = sm_120. 빌드 당시 값과 같아야 torch 의 JIT 가 재컴파일을 건너뛰고
#     ~/.cache/torch_extensions/py311_cu128/gsplat_cuda/gsplat_cuda.so 를 그대로 쓴다.
#     기본값(전 아키텍처)이면 수 배 느리고, 위 10.1 문제에 걸린다.
#
#  MAX_JOBS=4
#     혹시 재컴파일이 걸리더라도 nvcc 를 4개까지만 띄운다. 기본값은 nproc(=12)이고
#     그러면 WSL 메모리 상한(호스트 31GB 의 50% = 15.6GB)을 쳐서 cudafe++ 가 OOM 으로
#     죽고 VM 전체가 멎는다. 이 프로젝트에서 이것 때문에 두 번 날렸다.
#
#  CPATH
#     conda 의 cuda-toolkit 은 cuda_runtime.h 를 include/ 가 아니라
#     targets/x86_64-linux/include/ 에 둔다. 없으면 컴파일이 헤더를 못 찾는다.

export PATH="$HOME/miniconda3/envs/gs/bin:$PATH"
export CONDA_PREFIX="$HOME/miniconda3/envs/gs"
export CUDA_HOME="$CONDA_PREFIX"
export MAX_JOBS=4
export TORCH_CUDA_ARCH_LIST="12.0"
for d in "$CUDA_HOME/include" "$CUDA_HOME/targets/x86_64-linux/include"; do
  [ -f "$d/cuda_runtime.h" ] && export CPATH="$d${CPATH:+:$CPATH}"
done
