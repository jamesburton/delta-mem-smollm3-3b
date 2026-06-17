"""Autoregressive baseline on RTX 3060 — quant × attn × ctx matrix.

Originally a single-config baseline (Qwen3-4B bnb 4-bit NF4) for framing the
LLaDA-8B diffusion result. Extended to sweep:

- --quant {bnb-4bit,bf16}: isolate bnb dequant cost vs raw matmul
- --attn {sdpa,flash_attention_2,eager}: isolate attention backend
- --ctx N: synthetic long-context probe (pads prompt with filler tokens up to N)

Use case: confirm whether FA2 helps when dequant isn't the bottleneck
(bf16) and/or when attention starts to dominate (long ctx). The bnb-4bit
short-ctx result (2026-06-16) was 2.9 tok/s identical SDPA vs FA2.
"""
from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
PROMPT = (
    "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 "
    "kilometers per hour. How many kilometers can she run in 8 hours?"
)
NEW_TOKENS = 128


def build_prompt_ids(tok, target_ctx: int | None, device) -> torch.Tensor:
    """Construct the prompt as chat-template; optionally pad with filler to target_ctx tokens."""
    chat = tok.apply_chat_template(
        [{"role": "user", "content": PROMPT}], add_generation_prompt=True, tokenize=False,
    )
    enc = tok(chat, add_special_tokens=False, return_tensors="pt").to(device)
    base_len = int(enc["input_ids"].shape[-1])
    if target_ctx is None or target_ctx <= base_len:
        return enc
    # Pad with repeated filler sentence to reach target_ctx (synthetic long-ctx probe).
    filler_sentence = (
        "The quick brown fox jumps over the lazy dog and then runs around the field. "
    )
    filler_ids = tok(filler_sentence, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    need = target_ctx - base_len
    reps = (need + len(filler_ids) - 1) // len(filler_ids)
    filler_pad = filler_ids.repeat(reps)[:need].unsqueeze(0).to(device)
    new_ids = torch.cat([filler_pad, enc["input_ids"]], dim=-1)
    new_mask = torch.ones_like(new_ids)
    return {"input_ids": new_ids, "attention_mask": new_mask}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attn",
        default="sdpa",
        choices=["sdpa", "flash_attention_2", "eager"],
        help="attention backend passed to transformers (default: sdpa)",
    )
    parser.add_argument(
        "--quant",
        default="bnb-4bit",
        choices=["bnb-4bit", "bf16"],
        help="bnb-4bit NF4 (default) or bf16 native (no quant)",
    )
    parser.add_argument(
        "--ctx",
        type=int,
        default=None,
        help="target prompt length in tokens (pads with filler if > natural prompt). Default: natural prompt only.",
    )
    parser.add_argument(
        "--new-tokens",
        type=int,
        default=NEW_TOKENS,
        help=f"tokens to generate (default {NEW_TOKENS})",
    )
    args = parser.parse_args()

    assert torch.cuda.is_available()

    load_kwargs: dict = dict(
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation=args.attn,
    )
    if args.quant == "bnb-4bit":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    torch.cuda.reset_peak_memory_stats()

    print(f"[load] {MODEL_ID} (quant={args.quant}, attn={args.attn})")
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    m = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kwargs).eval()
    load_s = time.perf_counter() - t0
    print(f"[load] done in {load_s:.1f}s, VRAM: {torch.cuda.max_memory_allocated()/2**30:.2f} GB")

    enc = build_prompt_ids(tok, args.ctx, m.device)
    prompt_tokens = int(enc["input_ids"].shape[-1])
    print(f"[prompt] tokens={prompt_tokens} (requested ctx={args.ctx})")

    # warmup (Triton/sdpa JIT)
    with torch.inference_mode():
        m.generate(**enc, max_new_tokens=8, do_sample=False)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = m.generate(
            **enc, max_new_tokens=args.new_tokens, do_sample=False, use_cache=True,
        )
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    new_tokens = int(out.shape[-1]) - prompt_tokens
    tps = new_tokens / wall
    peak = torch.cuda.max_memory_allocated() / 2**30

    text = tok.decode(out[0, prompt_tokens:], skip_special_tokens=True)
    snippet = (text[:160] + "…") if len(text) > 160 else text
    print(f"---\n{snippet}\n---")
    print(
        f"[result] quant={args.quant} attn={args.attn} ctx={prompt_tokens} "
        f"new={new_tokens} wall={wall:.2f}s tok/s={tps:.1f} "
        f"peak_vram={peak:.2f} GB ms/tok={wall/new_tokens*1e3:.1f}"
    )


if __name__ == "__main__":
    main()
