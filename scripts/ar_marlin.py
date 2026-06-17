"""Qwen3-4B GPTQ-Marlin AR runner — same prompt + decode config as ar_baseline.py.

Hypothesis being tested: GPTQ-Marlin fuses dequant + matmul into one kernel,
so the per-forward dequant tax bnb-4bit pays (~90 ms/tok on T1) goes away.

Compares against `scripts/ar_baseline.py` numbers verbatim so output slots
into the same `results/qwen3_4b_ar_fa2_sweep.json`-style table.

Checkpoint: JunHowie/Qwen3-4B-Instruct-2507-GPTQ-Int4 (2.67 GB)
  - Legacy GPTQ format (`checkpoint_format: "gptq"`, NOT compressed-tensors)
  - bits=4, group_size=128, desc_act=false, sym=true  → Marlin-compatible
  - Quantized with gptqmodel:4.0.0 (same family we have installed)

Stack: system Python311 (C:\\Python311), NOT the project venv.
  - torch 2.11.0+cu126
  - transformers 5.4.0
  - gptqmodel 7.1.0+d0bed15
  - optimum installed
  - flash_attn 2.8.3.post1
  - bitsandbytes 0.49.2

Marlin sm_86 support: confirmed via upstream Marlin README
  ("compute capability >= 8.0, Ampere or Ada") and gptqmodel docs
  ("Turing+ / sm_75+"). RTX 3060 is sm_86 — in scope.

Per CUDA_NOTES "Common pitfalls": gptqmodel logs Unicode ASCII-art on
import, so this script forces UTF-8 stdout/stderr BEFORE importing.

CLI mirrors ar_baseline.py shape for consistency:
  --attn {sdpa,flash_attention_2,eager}  (default: sdpa)
  --backend {marlin,exllama_v2,triton,auto}  (default: marlin)
  --ctx N         optional synthetic long-ctx pad
  --new-tokens N  default 128 (same as baseline)
"""
from __future__ import annotations

import argparse
import io
import os
import sys

# Force UTF-8 BEFORE importing gptqmodel/transformers — see CUDA_NOTES
# "Common pitfalls: Windows cp1252 console encoding causes silent script
# death when libraries print Unicode glyphs (gptqmodel's logo)."
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import time  # noqa: E402

import torch  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    GPTQConfig,
)

MODEL_ID = "JunHowie/Qwen3-4B-Instruct-2507-GPTQ-Int4"
PROMPT = (
    "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 "
    "kilometers per hour. How many kilometers can she run in 8 hours?"
)
NEW_TOKENS = 128


def build_prompt_ids(tok, target_ctx: int | None, device):
    """Same prompt-building path as ar_baseline.py — for apples-to-apples comparison."""
    chat = tok.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        add_generation_prompt=True,
        tokenize=False,
    )
    enc = tok(chat, add_special_tokens=False, return_tensors="pt").to(device)
    base_len = int(enc["input_ids"].shape[-1])
    if target_ctx is None or target_ctx <= base_len:
        return enc
    filler_sentence = (
        "The quick brown fox jumps over the lazy dog and then runs around the field. "
    )
    filler_ids = tok(
        filler_sentence, add_special_tokens=False, return_tensors="pt"
    )["input_ids"][0]
    need = target_ctx - base_len
    reps = (need + len(filler_ids) - 1) // len(filler_ids)
    filler_pad = filler_ids.repeat(reps)[:need].unsqueeze(0).to(device)
    new_ids = torch.cat([filler_pad, enc["input_ids"]], dim=-1)
    new_mask = torch.ones_like(new_ids)
    return {"input_ids": new_ids, "attention_mask": new_mask}


def _detect_loaded_kernel(model) -> str:
    """Best-effort: walk the model to find which GPTQ kernel actually got bound.

    gptqmodel's Linear replacements have distinctive class names like
    `MarlinQuantLinear`, `ExllamaV2QuantLinear`, etc. We sample a few Linear
    layers and report the most common backend class name. This catches the
    case where `backend="marlin"` silently fell back to exllama_v2 because
    a tensor shape failed Marlin's strict requirements.
    """
    seen: dict[str, int] = {}
    for _, module in model.named_modules():
        cn = type(module).__name__
        if "QuantLinear" in cn or "Linear4bit" in cn or "Marlin" in cn:
            seen[cn] = seen.get(cn, 0) + 1
    if not seen:
        return "unknown (no QuantLinear modules found)"
    # Report the most populous one
    top = sorted(seen.items(), key=lambda kv: -kv[1])[0]
    return f"{top[0]} (x{top[1]})"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attn",
        default="sdpa",
        choices=["sdpa", "flash_attention_2", "eager"],
        help="attention backend passed to transformers (default: sdpa)",
    )
    parser.add_argument(
        "--backend",
        default="marlin",
        choices=["marlin", "exllama_v2", "triton", "auto"],
        help="GPTQ kernel backend (default: marlin). 'auto' lets gptqmodel pick.",
    )
    parser.add_argument(
        "--ctx",
        type=int,
        default=None,
        help="target prompt length in tokens (pads with filler if > natural). Default: natural.",
    )
    parser.add_argument(
        "--new-tokens",
        type=int,
        default=NEW_TOKENS,
        help=f"tokens to generate (default {NEW_TOKENS}, matches ar_baseline.py)",
    )
    parser.add_argument(
        "--model-id",
        default=MODEL_ID,
        help=f"HF repo id (default {MODEL_ID})",
    )
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA not available — wrong Python? Use C:\\Python311."

    # Build GPTQConfig.
    #
    # NOTE: `bits` is required even when loading a pre-quantized model —
    # transformers' loader uses it to sanity-check the checkpoint's
    # quantize_config.json. JunHowie's checkpoint is bits=4.
    #
    # `backend=None` (auto) defers to gptqmodel; passing it explicitly
    # documents intent and surfaces immediate errors if the backend isn't
    # compatible with this checkpoint's group_size / desc_act / sym.
    qconfig_kwargs = {"bits": 4}
    if args.backend != "auto":
        qconfig_kwargs["backend"] = args.backend
    qconfig = GPTQConfig(**qconfig_kwargs)

    load_kwargs = dict(
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation=args.attn,
        quantization_config=qconfig,
    )

    torch.cuda.reset_peak_memory_stats()
    print(
        f"[load] {args.model_id}  (backend={args.backend}, attn={args.attn})"
    )
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(args.model_id)
    m = AutoModelForCausalLM.from_pretrained(args.model_id, **load_kwargs).eval()
    load_s = time.perf_counter() - t0
    load_vram = torch.cuda.max_memory_allocated() / 2**30
    print(f"[load] done in {load_s:.1f}s, VRAM after load: {load_vram:.2f} GB")
    print(f"[load] kernel actually bound: {_detect_loaded_kernel(m)}")

    enc = build_prompt_ids(tok, args.ctx, m.device)
    prompt_tokens = int(enc["input_ids"].shape[-1])
    print(f"[prompt] tokens={prompt_tokens} (requested ctx={args.ctx})")

    # Warmup (CUDA graph capture / kernel JIT)
    with torch.inference_mode():
        m.generate(**enc, max_new_tokens=8, do_sample=False)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = m.generate(
            **enc,
            max_new_tokens=args.new_tokens,
            do_sample=False,
            use_cache=True,
        )
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    new_tokens = int(out.shape[-1]) - prompt_tokens
    tps = new_tokens / wall
    peak = torch.cuda.max_memory_allocated() / 2**30

    text = tok.decode(out[0, prompt_tokens:], skip_special_tokens=True)
    snippet = (text[:160] + "…") if len(text) > 160 else text
    print(f"---\n{snippet}\n---")
    # SAME format string shape as ar_baseline.py for grep-friendly comparison.
    print(
        f"[result] quant=gptq-marlin backend={args.backend} attn={args.attn} "
        f"ctx={prompt_tokens} new={new_tokens} wall={wall:.2f}s "
        f"tok/s={tps:.1f} peak_vram={peak:.2f} GB ms/tok={wall/new_tokens*1e3:.1f}"
    )


if __name__ == "__main__":
    main()
