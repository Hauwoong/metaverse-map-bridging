#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""⑤ 끝점 조건 영상 생성 — LTX-Video 0.9.5 (2B).

I_A_out (A 의 마지막 시점) 을 첫 프레임, I_B_in (B 의 첫 시점) 을 마지막 프레임으로
조건을 걸어 그 사이 복도를 생성한다.

🔴 모델 선택 근거는 코드/ltx_환경구축.sh 주석 참조 (LTX-2/0.9.7 은 RAM 상한에 걸린다).

⚠️ 해상도는 32 의 배수여야 한다. 3DGS 렌더가 960x540 인데 540 은 32 의 배수가 아니다.
   16:9 를 유지하면서 둘 다 32 의 배수인 1024x576 으로 맞춘다.
⚠️ 프레임 수는 8k+1 이어야 한다 (VAE 의 시간축 압축 격자).
⚠️ 조건 프레임은 '생성 해상도와 같은 크기' 로 넣는다.
"""
import argparse, json, os, sys, time
from pathlib import Path

import torch
from PIL import Image

DEFAULT_PROMPT = (
    "Smooth steady forward camera motion walking through a bright indoor hallway of a "
    "university building. Beige painted walls, grey square floor tiles, fluorescent "
    "ceiling lights, wooden doors and bulletin boards along both sides. Realistic "
    "handheld footage, consistent lighting, no people."
)
DEFAULT_NEGATIVE = (
    "worst quality, inconsistent motion, blurry, jittery, distorted, warped geometry, "
    "flickering, text, watermark, people, duplicated walls"
)


def build_conditions(first, last, num_frames):
    """LTXVideoCondition 의 필드 이름이 버전마다 다르다. 설치된 것을 보고 맞춘다.

    last 가 None 이면 시작 프레임만 조건으로 건다(I2V).
    """
    from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition

    fields = getattr(LTXVideoCondition, "__dataclass_fields__", {})
    img_key = "image" if "image" in fields else "frames"
    idx_key = "frame_index" if "frame_index" in fields else "index"
    print("  LTXVideoCondition 필드 = %s  → %s / %s" % (list(fields), img_key, idx_key))

    def mk(img, idx):
        return LTXVideoCondition(**{img_key: img, idx_key: idx})

    # 마지막 프레임 인덱스는 음수(-1)가 아니라 명시적인 양수로 준다.
    # 음수 인덱스 지원은 LTX-Video 쪽 문서에 없다 — 되는 것을 확인하기 전에는 쓰지 않는다.
    if last is None:
        return [mk(first, 0)]
    return [mk(first, 0), mk(last, num_frames - 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", required=True, help="첫 프레임 (A 의 마지막 시점)")
    ap.add_argument("--last", default=None,
                    help="마지막 프레임 (B 의 첫 시점). 생략하면 끝점 조건 없이 "
                         "시작 프레임만으로 생성한다(I2V). 짧은 보폭 연쇄에서 쓴다 — "
                         "끝이 고정되지 않으면 두 끝을 보간할 유인이 없어진다.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Lightricks/LTX-Video-0.9.5")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=576)
    ap.add_argument("--num-frames", type=int, default=161)   # 8k+1
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--guide", default=None,
                    help="가이드 영상(mp4). 끝점 좌표로 만든 카메라 궤적을 G_A·G_B 에서 "
                         "렌더해 교차 페이드한 것. 모델에게 '카메라가 이렇게 움직인다' 를 "
                         "기하로 알려준다. 텍스트 프롬프트로는 줄 수 없는 정보다.")
    ap.add_argument("--guide-strength", type=float, default=0.2,
                    help="가이드의 구속 세기. 높이면 floater 까지 따라 그리고, "
                         "낮추면 무시한다. 0.15~0.3 에서 시작할 것.")
    ap.add_argument("--embeds", default=None,
                    help="프롬프트 임베딩 파일. --encode-only 로 먼저 만들어 두면 "
                         "확산 단계에서 T5-XXL(9GB)을 RAM 에 들고 있지 않아도 된다.")
    ap.add_argument("--encode-only", action="store_true",
                    help="텍스트 인코딩만 하고 --embeds 에 저장한 뒤 종료한다.")
    ap.add_argument("--offload", default="model", choices=["none", "model", "sequential"],
                    help="none=전부 GPU 상주(가장 빠름, VRAM 부족 위험), "
                         "model=모듈 단위 교체(권장), sequential=레이어 단위(가장 적게 쓰고 가장 느림)")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--negative", default=DEFAULT_NEGATIVE)
    a = ap.parse_args()

    assert a.width % 32 == 0 and a.height % 32 == 0, "해상도는 32 의 배수여야 한다"
    assert (a.num_frames - 1) % 8 == 0, "프레임 수는 8k+1 이어야 한다"

    from diffusers import LTXConditionPipeline
    from diffusers.utils import export_to_video

    # ── 1단계: 텍스트 인코딩만 (T5-XXL 만 올린다) ───────────────────────────
    # 🔴 왜 나누는가. enable_model_cpu_offload 는 파이프라인 '전체' 가중치를
    #    시스템 RAM 에 상주시킨다. LTX 0.9.5 는 transformer 2B + T5-XXL ≈ 14GB 인데
    #    WSL 메모리 상한이 15.6GB(호스트 31.2GB 의 50%)라 그대로는 숨 쉴 틈이 없다.
    #    프롬프트는 생성 중에 바뀌지 않으므로, 먼저 인코딩해 저장하고 프로세스를 끝내면
    #    그 9GB 가 통째로 반환된다. 확산 단계는 ~5GB 로 돈다.
    if a.encode_only:
        assert a.embeds, "--encode-only 에는 --embeds 경로가 필요하다"
        print("== 텍스트 인코더만 적재 ==")
        t0 = time.time()
        pipe = LTXConditionPipeline.from_pretrained(
            a.model, dtype=torch.bfloat16, transformer=None, vae=None)
        pipe.text_encoder.to("cuda")
        pe, pm, ne, nm = pipe.encode_prompt(
            prompt=a.prompt, negative_prompt=a.negative,
            do_classifier_free_guidance=a.guidance > 1.0, device="cuda")
        Path(a.embeds).parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(prompt_embeds=pe.cpu(), prompt_attention_mask=pm.cpu(),
                        negative_prompt_embeds=None if ne is None else ne.cpu(),
                        negative_prompt_attention_mask=None if nm is None else nm.cpu(),
                        prompt=a.prompt, negative=a.negative), a.embeds)
        print("  저장: %s  (%.1f초)" % (a.embeds, time.time() - t0))
        return

    # ── 2단계: 확산 ────────────────────────────────────────────────────────
    emb = None
    if a.embeds and os.path.exists(a.embeds):
        emb = torch.load(a.embeds, map_location="cpu", weights_only=False)
        print("== 저장된 프롬프트 임베딩 사용: %s ==" % a.embeds)

    print("== 모델 적재: %s ==" % a.model)
    t0 = time.time()
    kw = dict(dtype=torch.bfloat16)
    if emb is not None:
        kw.update(text_encoder=None, tokenizer=None)   # T5-XXL 을 아예 안 올린다
    pipe = LTXConditionPipeline.from_pretrained(a.model, **kw)
    # 🔴 .to("cuda") 로 통째로 올리지 않는다.
    #    LTX 0.9.5 는 transformer 2B 외에 T5-XXL 텍스트 인코더(bf16 ≈ 9GB)를 함께 쓴다.
    #    합 ≈ 14GB 라 16GB VRAM 에 올리면 활성값까지 더해 OOM 이 난다.
    #    model_cpu_offload 는 쓰는 모듈만 GPU 에 올렸다 내린다 — 텍스트 인코딩이 끝나면
    #    그 9GB 가 GPU 에서 빠지므로 확산 단계는 여유롭게 돈다.
    if a.offload == "model":
        pipe.enable_model_cpu_offload()
    elif a.offload == "sequential":
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to("cuda")
    pipe.vae.enable_tiling()   # VAE 디코딩이 OOM 의 단골 지점이다
    print("  적재 %.1f초, 오프로딩=%s" % (time.time() - t0, a.offload))

    def load(p):
        im = Image.open(p).convert("RGB")
        if im.size != (a.width, a.height):
            print("  %s  %s -> %s" % (Path(p).name, im.size, (a.width, a.height)))
            im = im.resize((a.width, a.height), Image.LANCZOS)
        return im

    first = load(a.first)
    last = load(a.last) if a.last else None
    if last is None:
        print("  끝점 조건 없음 (I2V) — 시작 프레임만 건다")
    conditions = build_conditions(first, last, a.num_frames)

    # 가이드 영상을 조건으로 추가한다.
    # 🔴 왜: 지금까지 모델은 사진 두 장과 텍스트만 봤다. 그래서 이 간극이 15 단위이고
    #    80° 돌아야 한다는 것을 모르고, 6.7초 안에 그만큼 전진하는 대신 두 끝을 보간했다.
    #    가이드는 그 카메라 운동을 '기하로' 알려준다. 양 끝은 정확하고 가운데는
    #    스플랫 커버리지 밖이라 무너져 있으므로, strength 를 낮게 줘 운동만 가져간다.
    if a.guide:
        from diffusers.utils import load_video
        from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
        gv = load_video(a.guide)
        gv = [im.convert("RGB").resize((a.width, a.height), Image.LANCZOS) for im in gv]
        gv = gv[:a.num_frames]
        conditions.append(LTXVideoCondition(video=gv, frame_index=0,
                                            strength=a.guide_strength))
        print("  가이드 영상 %d프레임, strength=%.2f" % (len(gv), a.guide_strength))

    print("== 생성: %dx%d, %d프레임 (%.1f초 @%dfps), %d스텝 ==" %
          (a.width, a.height, a.num_frames, a.num_frames / a.fps, a.fps, a.steps))
    t0 = time.time()
    if emb is not None:
        # 🔴 저장된 임베딩은 CPU 텐서다. 파이프라인은 이것을 알아서 옮겨주지 않는다
        #    (prompt 문자열로 줄 때만 인코더가 장치에 맞춰 만든다). 직접 옮긴다:
        #      RuntimeError: mat1 is on cpu, different from other tensors on cuda:0
        #    offload 를 쓰면 모듈이 오가므로 pipe._execution_device 를 기준으로 삼는다.
        dev = getattr(pipe, "_execution_device", None) or torch.device("cuda")
        def to_dev(x):
            return None if x is None else x.to(dev)
        text_kw = dict(prompt_embeds=to_dev(emb["prompt_embeds"]),
                       prompt_attention_mask=to_dev(emb["prompt_attention_mask"]),
                       negative_prompt_embeds=to_dev(emb["negative_prompt_embeds"]),
                       negative_prompt_attention_mask=to_dev(emb["negative_prompt_attention_mask"]))
        print("  임베딩을 %s 로 이동" % dev)
    else:
        text_kw = dict(prompt=a.prompt, negative_prompt=a.negative)

    out = pipe(
        conditions=conditions,
        **text_kw,
        width=a.width,
        height=a.height,
        num_frames=a.num_frames,
        num_inference_steps=a.steps,
        guidance_scale=a.guidance,
        decode_timestep=0.05,          # 0.9.1+ 의 timestep-aware VAE
        image_cond_noise_scale=0.025,
        generator=torch.Generator("cuda").manual_seed(a.seed),
    ).frames[0]
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1024**3

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    export_to_video(out, a.out, fps=a.fps)

    meta = dict(model=a.model, width=a.width, height=a.height, num_frames=a.num_frames,
                fps=a.fps, steps=a.steps, guidance=a.guidance, seed=a.seed,
                prompt=a.prompt, negative_prompt=a.negative,
                first=a.first, last=a.last,
                offload=a.offload, guide=a.guide, guide_strength=a.guide_strength,
                생성시간_초=round(dt, 1), VRAM최대_GB=round(peak, 2),
                프레임당_초=round(dt / a.num_frames, 2))
    mp = str(Path(a.out).with_suffix(".json"))
    json.dump(meta, open(mp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("== 완료 ==")
    print("  영상      %s  (%d프레임)" % (a.out, len(out)))
    print("  메타      %s" % mp)
    print("  생성 시간 %.1f초 (%.2f초/프레임)" % (dt, dt / a.num_frames))
    print("  VRAM 최대 %.2f GB" % peak)


if __name__ == "__main__":
    main()
