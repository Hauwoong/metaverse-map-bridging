#!/usr/bin/env bash
# 학습이 끝난 세 장면의 보고서 지표를 한 번에 뽑는다.
#   표 1  장면별 가우시안 개수 / PSNR / SSIM / LPIPS
#   표 4  VRAM / 렌더 FPS / 렌더 해상도(960x540 고정)
# 결과: ~/data/지표/G_*.json  과 표준출력의 마크다운 표.
#
# ⚠️ 먼저 코드/nerfstudio_패치.sh 를 돌려놔야 한다. 안 하면 ns-eval 이
#    eval_utils.py:62 의 torch.load(weights_only=True) 에서 죽는다.
set -u
source "$(dirname "$0")/환경.sh"
RUN=~/data/runs
OUTD=~/data/지표
mkdir -p "$OUTD"

for S in A GT1 B; do
  CFG="$RUN/G_$S/splatfacto/run1/config.yml"
  if [ ! -f "$CFG" ]; then echo "G_$S: config.yml 없음 — 건너뜀"; continue; fi
  echo "== G_$S 평가 중 =="
  ns-eval --load-config "$CFG" --output-path "$OUTD/G_$S.json" < /dev/null 2>&1 | tail -3
done

echo
echo "== 가우시안 개수 (체크포인트의 means 길이) =="
python - "$RUN" "$OUTD" <<'PY'
import sys, os, json, glob, torch
RUN, OUTD = sys.argv[1], sys.argv[2]
rows = []
for S in ("A", "GT1", "B"):
    d = os.path.join(RUN, "G_%s" % S, "splatfacto", "run1", "nerfstudio_models")
    cks = sorted(glob.glob(os.path.join(d, "*.ckpt")))
    if not cks:
        continue
    st = torch.load(cks[-1], map_location="cpu", weights_only=False)
    sd = st["pipeline"]
    key = next(k for k in sd if k.endswith("gauss_params.means"))
    n = sd[key].shape[0]
    jp = os.path.join(OUTD, "G_%s.json" % S)
    m = {}
    if os.path.exists(jp):
        m = json.load(open(jp)).get("results", {})
    rows.append((S, os.path.basename(cks[-1]), n, m))

print()
print("| 장면 | 가우시안 | PSNR | SSIM | LPIPS | FPS |")
print("|---|---|---|---|---|---|")
def g(m, k):
    v = m.get(k)
    return "%.3f" % v if isinstance(v, (int, float)) else "[TODO]"
for S, ck, n, m in rows:
    print("| G_%s | %s | %s | %s | %s | %s |"
          % (S, format(n, ","), g(m, "psnr"), g(m, "ssim"),
             g(m, "lpips"), g(m, "fps")))
print()
for S, ck, n, m in rows:
    print("G_%-4s %s  가우시안 %s" % (S, ck, format(n, ",")))
PY

echo
echo "== VRAM 최대 사용량 (학습 로그에서) =="
grep -oE "GPU Memory[^|]*" ~/data/train.log 2>/dev/null | tail -3 || echo "  로그에 없음 — nvidia-smi 로 렌더 중 측정할 것"
