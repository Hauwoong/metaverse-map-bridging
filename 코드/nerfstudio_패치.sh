#!/usr/bin/env bash
# nerfstudio 1.1.5 를 torch 2.6+ 에서 쓰려면 반드시 먼저 한 번 돌린다. (멱등)
#
# torch 2.6 부터 torch.load 의 기본값이 weights_only=True 로 바뀌었는데
# nerfstudio 1.1.5 는 이 인자를 넘기지 않는다. 그래서 체크포인트를 읽는 모든 경로가
#   _pickle.UnpicklingError: Weights only load failed ...
#   GLOBAL numpy.core.multiarray.scalar was not an allowed global by default
# 로 죽는다. trainer.py 둘은 학습 이어받기, eval_utils.py 하나는
# ns-eval / ns-render / ns-export 가 전부 지나가는 길이다 — 지표 표와 제출 영상이
# 여기에 걸린다. 학습이 끝난 뒤에 발견하면 늦는다.
set -eu
SP=$(python -c 'import nerfstudio,os;print(os.path.dirname(nerfstudio.__file__))')
echo "대상: $SP"
for F in engine/trainer.py utils/eval_utils.py; do
  T="$SP/$F"
  [ -f "$T.bak" ] || cp "$T" "$T.bak"
  python - "$T" <<'PY'
import sys, io
p = sys.argv[1]
s = io.open(p, encoding="utf-8").read()
n = 0
for a, b in [('torch.load(load_path, map_location="cpu")',
              'torch.load(load_path, map_location="cpu", weights_only=False)'),
             ('torch.load(load_checkpoint, map_location="cpu")',
              'torch.load(load_checkpoint, map_location="cpu", weights_only=False)')]:
    n += s.count(a); s = s.replace(a, b)
io.open(p, "w", encoding="utf-8").write(s)
print("  %s  %d곳 수정" % (p.split("/")[-1], n))
PY
  grep -c "weights_only=False" "$T" | sed 's/^/  weights_only=False 현재 /'
done
echo "끝"
