#!/bin/bash
set -u
log(){ echo "[$(date +%H:%M:%S)] $*"; }
export PATH="$HOME/miniconda3/envs/gs/bin:$PATH"
export CONDA_PREFIX="$HOME/miniconda3/envs/gs"
export CUDA_HOME="$CONDA_PREFIX"
export MAX_JOBS=4
export TORCH_CUDA_ARCH_LIST="12.0"
log "nvcc: $($CUDA_HOME/bin/nvcc --version 2>&1 | tail -1)"
log "헤더 위치 탐색"
for d in "$CUDA_HOME/include" "$CUDA_HOME/targets/x86_64-linux/include"; do
  [ -f "$d/cuda_runtime.h" ] && { log "  found: $d/cuda_runtime.h"; export CPATH="$d${CPATH:+:$CPATH}"; }
done
log "빌드 캐시 삭제"
rm -rf "$HOME/.cache/torch_extensions"
log "MAX_JOBS=$MAX_JOBS  ARCH=$TORCH_CUDA_ARCH_LIST  — 컴파일 시작"
python -u "$1"
log "종료 코드 $?"
