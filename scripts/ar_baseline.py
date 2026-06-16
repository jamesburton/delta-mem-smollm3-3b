"""Autoregressive baseline on RTX 3060 — frames the LLaDA-8B diffusion result.

Loads Qwen3-4B-Instruct-2507 in bnb 4-bit NF4 (matching the LLaDA-8B sweep
quant), runs the same Lily prompt, and reports prefill + decode tok/s. The
goal is to know whether 28 tok/s for 8B-diffusion is competitive against
4B-AR on the same hardware in the same quant.
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attn",
        default="sdpa",
        choices=["sdpa", "flash_attention_2", "eager"],
        help="attention backend passed to transformers (default: sdpa)",
    )
    args = parser.parse_args()

    assert torch.cuda.is_available()
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    torch.cuda.reset_peak_memory_stats()

    print(f"[load] {MODEL_ID} (bnb 4-bit NF4, bf16 compute, attn={args.attn})")
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    m = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        quantization_config=bnb_cfg,
        dtype=torch.bfloat16,
        attn_implementation=args.attn,
    ).eval()
    load_s = time.perf_counter() - t0
    print(f"[load] done in {load_s:.1f}s, VRAM: {torch.cuda.max_memory_allocated()/2**30:.2f} GB")

    chat = tok.apply_chat_template(
        [{"role": "user", "content": PROMPT}], add_generation_prompt=True, tokenize=False,
    )
    enc = tok(chat, add_special_tokens=False, return_tensors="pt").to(m.device)
    prompt_tokens = int(enc["input_ids"].shape[-1])
    print(f"[prompt] tokens={prompt_tokens}")

    # warmup (Triton/sdpa JIT)
    with torch.inference_mode():
        m.generate(**enc, max_new_tokens=8, do_sample=False)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = m.generate(
            **enc, max_new_tokens=NEW_TOKENS, do_sample=False, use_cache=True,
        )
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    new_tokens = int(out.shape[-1]) - prompt_tokens
    tps = new_tokens / wall
    peak = torch.cuda.max_memory_allocated() / 2**30

    text = tok.decode(out[0, prompt_tokens:], skip_special_tokens=True)
    snippet = (text[:200] + "…") if len(text) > 200 else text
    print(f"---\n{snippet}\n---")
    print(
        f"[result] new_tokens={new_tokens} wall={wall:.2f}s "
        f"tok/s={tps:.1f} peak_vram={peak:.2f} GB ms/tok={wall/new_tokens*1e3:.1f}"
    )


if __name__ == "__main__":
    main()
