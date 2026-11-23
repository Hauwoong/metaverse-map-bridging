#!/usr/bin/env bash
# VGGT-1B 가중치를 미리 받아 둔다 (약 5GB).
#
# 🔴 왜 분리하는가: 지금 COLMAP 이 GPU 를 점유하고 있다. 가중치 받기는
#    네트워크만 쓰므로 먼저 끝내 두면, COLMAP 이 끝나는 즉시 추론에 들어갈 수 있다.
#
# 🔴 캐시를 WSL 쪽(ext4, $HOME/.cache)에 둔다. /mnt/c 는 9P 라 느리고 용량도 적다.
set -u
ENV="$HOME/venv/vggt"
PY="$ENV/bin/python"
SRC="$HOME/src/vggt"
LOG="$HOME/data/vggt_weights.log"; : > "$LOG"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "== VGGT-1B 가중치 받기 =="
df -h "$HOME" | tail -1 | tee -a "$LOG"

PYTHONPATH="$SRC" "$PY" - <<'PYX' 2>&1 | tee -a "$LOG"
import os, time, torch
t0 = time.time()
from vggt.models.vggt import VGGT
try:
    m = VGGT.from_pretrained("facebook/VGGT-1B")
    how = "from_pretrained"
except Exception as e:
    print("  from_pretrained 실패:", type(e).__name__, e)
    print("  → torch.hub URL 로 대체")
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    m = VGGT()
    m.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    how = "torch.hub"
n = sum(p.numel() for p in m.parameters())
print("  적재 성공 (%s), 파라미터 %.2fB, %.0f초" % (how, n/1e9, time.time()-t0))
PYX

say "== 캐시 =="
du -sh "$HOME/.cache/huggingface" "$HOME/.cache/torch" 2>/dev/null | tee -a "$LOG"
say "끝"
