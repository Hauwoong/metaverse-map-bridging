# -*- coding: utf-8 -*-
"""보고서 3. 방법론 — 시스템 전체 파이프라인 구조도 (SVG 생성, 외부 의존성 없음).

사용: python 코드/파이프라인_그림.py
출력: 그림/파이프라인.svg

폰트가 없는 환경에서도 깨지지 않도록 수식 기호는 ASCII + 그리스 문자까지만 쓴다.
"""
import os, html

W = 1240
FONT = "'Malgun Gothic','Apple SD Gothic Neo','Noto Sans KR',sans-serif"
MONO = "'Cascadia Mono','D2Coding','Consolas',monospace"

# 색: 역할별로 구분하고, 흑백 인쇄에서도 테두리 굵기로 기여 박스가 구분되게 함
C = {
    "in":   ("#EFF1F3", "#94A3B8", "#334155"),   # 입력
    "proc": ("#E8F0FE", "#5B8DEF", "#17407E"),   # 기존 도구
    "gen":  ("#FDF0E3", "#E0953B", "#7A4A10"),   # 생성 모델
    "mine": ("#E6F6EC", "#2F9E5E", "#13532F"),   # 본 기여
    "out":  ("#F3EAFB", "#9B6BD6", "#4A2272"),   # 출력
    "eval": ("#FAFAFA", "#9CA3AF", "#374151"),   # 평가
}

parts = []
def esc(s): return html.escape(s, quote=False)
def tw(s):  # 대략적인 텍스트 폭 (한글은 넓다)
    return sum(12.4 if ord(c) > 0x2000 else 6.6 for c in s)

def box(x, y, w, h, kind, title, lines, bold=False, tag=None):
    fill, stroke, ink = C[kind]
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="{2.6 if bold else 1.4}"/>')
    parts.append(f'<text x="{x+16}" y="{y+26}" font-family="{FONT}" font-size="15.5" '
                 f'font-weight="700" fill="{ink}">{esc(title)}</text>')
    ty = y + 50
    for ln, mono in lines:
        parts.append(f'<text x="{x+16}" y="{ty}" font-family="{MONO if mono else FONT}" '
                     f'font-size="{12 if mono else 12.8}" fill="#1F2937">{esc(ln)}</text>')
        ty += 20
    if tag:
        parts.append(f'<text x="{x+w-14}" y="{y+26}" text-anchor="end" font-family="{FONT}" '
                     f'font-size="12" font-weight="700" fill="{stroke}">{esc(tag)}</text>')

def arrow(x1, y1, x2, y2, dashed=False):
    d = ' stroke-dasharray="6 5"' if dashed else ''
    parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#475569" '
                 f'stroke-width="2"{d} marker-end="url(#ah)"/>')

# ───────────────────────── 본류 (왼쪽) ─────────────────────────
MX, MW = 38, 690
CX = MX + MW / 2
S, y = [], 70

spec = [
    ("in", "① 입력 — 직접 촬영한 실공간 영상", [
        ("IMG_1791 케이스 1 (121s)  ·  IMG_1792 케이스 2 (118s)  ·  IMG_1793 B 보강 (63s)", False),
        ("같은 A(로비) · 같은 B(계단·게시판)를 서로 다른 두 경로로 연결", False),
        ("흔들림 · 인물 가림 구간은 프레임 추출 단계에서 제외", False)], 96, None),

    ("proc", "② 전처리 — 하나의 공통 좌표계 구성", [
        ("세 영상의 프레임을 한 번에 COLMAP (SfM) → 공통 좌표계 W + 희소 점군", False),
        ("정답 누출 방지: 중간 GT 의 3D 점 (M1 U M2) 가시 비율 > 30% 인 프레임은", False),
        ("                A·B 학습 집합에서 자동 제외", False)], 96, "COLMAP"),

    ("proc", "③ 3DGS 학습 — nerfstudio splatfacto", [
        ("G_A , G_B      : 양 끝 구역. 두 케이스가 공유한다", False),
        ("G_GT1 , G_GT2 : 중간 구간 정답. 숨겨 두고 평가에만 사용", False)], 78, "gsplat"),

    ("proc", "④ 끝점 조건 프레임 렌더", [
        ("케이스마다 경계 시점이 다르다 → A 출구 방향 렌더 I_A(out),", False),
        ("                                 B 입구 방향 렌더 I_B(in)", False)], 74, None),

    ("gen", "⑤ 연결 구간 영상 생성", [
        ("B3 (본 방법)    LTX-Video 끝점 조건 — 조건 프레임에 σ≈0 을 주어 양 끝을 고정", False),
        ("B0 (베이스라인) RIFE / FILM 프레임 보간 — 새 내용을 만들지 못하는 대조군", False)], 78, None),

    ("proc", "⑥ 생성 영상 → 3D 복원 · 좌표 정합", [
        ("ⓐ colmap image_registrator  --Mapper.fix_existing_frames 1", True),
        ("   실촬영 모델을 고정한 채 등록 → 처음부터 공통 좌표계, 정렬 불필요", False),
        ("ⓑ 실패 시: COLMAP ‖ VGGT 독립 복원 + 포즈 기반 Sim(3) — 중심 + 회전", False),
        ("→ 연결 구간 가우시안 G_C", False)], 118, None),

    ("mine", "⑦ ★ 이음새 최적화 — 본 프로젝트의 기여", [
        ("G_A , G_B 는 freeze. G_C 만 학습한다", False),
        ("L = L_photo  +  λc · Σ | I_A − I_C |  +  λd · Σ | D_A − D_C |  +  opacity 정규화", False),
        ("           경계 포즈에서 색과 깊이가 동시에 이어지도록 (3DGS 는 미분 가능)", False)], 100, None),

    ("out", "⑧ 출력 및 렌더링", [
        ("Sim(3) 적용 후 단일 PLY 로 병합 — 세 에셋으로 두면 경계에서 popping", False),
        ("카메라 경로 플라이스루 렌더 → 유튜브 제출 영상 / (선택) Unity", False)], 78, None),
]
for kind, title, lines, h, tag in spec:
    box(MX, y, MW, h, kind, title, lines, bold=(kind == "mine"), tag=tag)
    S.append((y, h)); y += h + 34

for i in range(len(S) - 1):
    arrow(CX, S[i][0] + S[i][1] + 3, CX, S[i + 1][0] - 5)

# ───────────────────────── 평가 (오른쪽) ─────────────────────────
EX, EW = 768, 434
E6, E8 = S[5][0], S[7][0]          # ⑥ · ⑧ 의 y

box(EX, E6, EW, 112, "eval", "평가 1층 — 정렬이 없어도 항상 측정", [
    ("등록률 — 생성 프레임 중 포즈를 찾은 비율", False),
    ("reprojection error (px) · 시간 일관성 (warp error)", False),
    ("렌더 FPS · 해상도 · 가우시안 개수", False),
    ("※ 등록률 자체가 생성 영상의 기하 일관성 지표", False)])

box(EX, E6 + 126, EW, 96, "eval", "정렬 품질 — ATE 단독 사용 금지", [
    ("ATE(위치 RMSE) 와 회전 오차를 반드시 함께 본다", False),
    ("중심만으로 정렬하면 A→B 축 주위 roll 1자유도가 남아,", False),
    ("장면이 106° 틀어져도 ATE 는 만점을 준다", False)])

box(EX, E8 - 12, EW, 96, "eval", "평가 2층 — 포즈 정렬에 성공했을 때", [
    ("생성 영상과 GT 는 카메라 궤적이 달라 직접 비교가 무의미", False),
    ("→ 추정 포즈에서 GT 3DGS 를 렌더해 픽셀 정렬 쌍을 만든다", False),
    ("PSNR ↑  ·  SSIM ↑  ·  LPIPS ↓", False)])

RE = MX + MW + 4
arrow(RE, E6 + 46,  EX - 5, E6 + 46,  dashed=True)      # ⑥ → 1층
arrow(RE, E6 + 100, EX - 5, E6 + 170, dashed=True)      # ⑥ → 정렬 품질
arrow(RE, E8 + 38,  EX - 5, E8 + 34,  dashed=True)      # ⑧ → 2층

# ───────────────────────── 제목 · 범례 ─────────────────────────
H = S[-1][0] + S[-1][1] + 96
parts.insert(0, f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>')
parts.insert(1, f'<text x="{MX}" y="34" font-family="{FONT}" font-size="21" font-weight="700" '
                f'fill="#0F172A">그림 1. 시스템 전체 파이프라인</text>')
parts.insert(2, f'<text x="{MX}" y="55" font-family="{FONT}" font-size="13" fill="#475569">'
                f'실촬영 3DGS 두 장면을 끝점 조건 생성 영상으로 잇고, 이음새를 최적화해 하나의 맵으로 병합한다</text>')

ly = H - 30
parts.append(f'<line x1="{MX}" y1="{ly-24}" x2="{W-38}" y2="{ly-24}" stroke="#E2E8F0" stroke-width="1.5"/>')
lx = MX
for kind, lab in (("in", "입력 데이터"), ("proc", "기존 도구 (COLMAP · gsplat · VGGT)"),
                  ("gen", "생성 모델 (LTX · RIFE/FILM)"), ("mine", "★ 본 프로젝트의 기여"),
                  ("out", "출력"), ("eval", "평가")):
    fill, stroke, _ = C[kind]
    parts.append(f'<rect x="{lx}" y="{ly-11}" width="16" height="14" rx="3" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="{2.6 if kind == "mine" else 1.4}"/>')
    parts.append(f'<text x="{lx+23}" y="{ly}" font-family="{FONT}" font-size="12" '
                 f'fill="#334155">{esc(lab)}</text>')
    lx += 23 + tw(lab) + 28

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">\n'
       '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
       'markerHeight="7" orient="auto-start-reverse">'
       '<path d="M 0 0 L 10 5 L 0 10 z" fill="#475569"/></marker></defs>\n'
       + "\n".join(parts) + "\n</svg>\n")

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = os.path.join(root, "그림", "파이프라인.svg")
os.makedirs(os.path.dirname(out), exist_ok=True)
open(out, "w", encoding="utf-8").write(svg)
print("wrote %s  (%d x %d, %.1f KB)" % (out, W, H, len(svg) / 1024))
