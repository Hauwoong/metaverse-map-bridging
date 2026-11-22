#!/usr/bin/env bash
# 표 4 — 병합이 3DGS 의 실시간성을 해치는가.
#   가우시안 개수와 FPS·VRAM 을 잰다. ns-eval 이 둘 다 보고한다.
# 🔴 VRAM 은 ns-eval 보고값이 아니라 렌더 중 nvidia-smi 샘플링의 최대값을 쓴다
#    (앞선 측정과 같은 방식 — 2,463 MiB 가 그렇게 나온 값이다).
set -u
source "$(dirname "$0")/환경.sh"
LOG=~/data/perf_merge.log; : > "$LOG"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

for M in "$@"; do
  CFG=~/data/runs/G_$M/splatfacto/run1/config.yml
  [ -f "$CFG" ] || { say "G_$M 없음"; continue; }
  say "---- G_$M ----"
  # VRAM 샘플러를 띄워 둔다
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 0.5; done ) > /tmp/vram_$M.txt 2>/dev/null &
  SP=$!
  ns-eval --load-config "$CFG" --output-path ~/data/runs/G_$M/eval.json >> "$LOG" 2>&1 < /dev/null
  kill $SP 2>/dev/null
  if [ -f ~/data/runs/G_$M/eval.json ]; then
    "$HOME/venv/ltx/bin/python" - "$M" <<'PY' 2>&1 | tee -a "$LOG"
import json, os, sys
m = sys.argv[1]
j = json.load(open(os.path.expanduser("~/data/runs/G_%s/eval.json" % m)))
r = j.get("results", j)
V = [int(x) for x in open("/tmp/vram_%s.txt" % m) if x.strip().isdigit()] or [0]
def g(k):
    v = r.get(k)
    return "%.3f" % v if isinstance(v, (int, float)) else "-"
print("  PSNR %s  SSIM %s  LPIPS %s  FPS %s  VRAM최대 %d MiB"
      % (g("psnr"), g("ssim"), g("lpips"), g("fps"), max(V)))
PY
  else
    say "  🔴 eval 실패"; tail -8 "$LOG"
  fi
done
say "끝"
