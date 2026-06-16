"""LLaDA-8B-Instruct config sweep on RTX 3060.

Goal: separate "are we running the diffusion sampler with the right knobs?" from
"are the kernels actually fast?". Fixed prompt + gen_length; vary block_length
and steps. Reports tok/s, peak VRAM, and seconds per forward pass.

Also reports the active attention implementation and bnb config so we can spot
kernel-level wins later (flash-attn, sdpa selection, etc.).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"
MASK_ID = 126336
PROMPT = (
    "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 "
    "kilometers per hour. How many kilometers can she run in 8 hours?"
)
GEN_LENGTH = 128

# (block_length, steps) configs to sweep. Tokens emitted = GEN_LENGTH;
# total forward passes per generation = steps.
CONFIGS: list[tuple[int, int]] = [
    (128, 128),  # 1 block, 128 steps — most expensive, most AR-like
    (32, 64),    # 4 blocks × 16 steps each — default from canonical script
    (64, 64),    # 2 blocks × 32 steps
    (128, 64),   # 1 block, 64 steps (full parallel denoise)
    (128, 32),   # 1 block, 32 steps
    (128, 16),   # 1 block, 16 steps — most aggressive
]


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    out = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        out[i, : remainder[i]] += 1
    return out


@torch.no_grad()
def llada_generate(model, prompt, attention_mask, steps, gen_length, block_length, mask_id=MASK_ID):
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=model.device)
    x[:, : prompt.shape[1]] = prompt.clone()
    if attention_mask is not None:
        attention_mask = torch.cat(
            [attention_mask, torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device)],
            dim=-1,
        )
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    per_block_steps = steps // num_blocks
    total_fwd = 0
    for nb in range(num_blocks):
        b_start = prompt.shape[1] + nb * block_length
        b_end = prompt.shape[1] + (nb + 1) * block_length
        block_mask_index = x[:, b_start:b_end] == mask_id
        num_transfer = get_num_transfer_tokens(block_mask_index, per_block_steps)
        for i in range(per_block_steps):
            mask_index = x == mask_id
            logits = model(x, attention_mask=attention_mask).logits
            total_fwd += 1
            x0 = torch.argmax(logits, dim=-1)
            p = F.softmax(logits, dim=-1)
            x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            x0_p[:, prompt.shape[1] + (nb + 1) * block_length :] = -np.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, -np.inf))
            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            for j in range(confidence.shape[0]):
                _, sel = torch.topk(confidence[j], k=int(num_transfer[j, i]))
                transfer_index[j, sel] = True
            x[transfer_index] = x0[transfer_index]
    return x, total_fwd


def describe_runtime(model) -> dict:
    """Report which attention impl is active + bnb meta."""
    info: dict[str, object] = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(torch.cuda.get_device_name(0)),
        "device_cc": ".".join(map(str, torch.cuda.get_device_capability(0))),
    }
    try:
        info["sdpa_flash"] = torch.backends.cuda.flash_sdp_enabled()
        info["sdpa_mem_eff"] = torch.backends.cuda.mem_efficient_sdp_enabled()
        info["sdpa_math"] = torch.backends.cuda.math_sdp_enabled()
    except Exception as e:  # noqa: BLE001
        info["sdpa_query_err"] = str(e)
    try:
        import bitsandbytes as bnb
        info["bnb"] = bnb.__version__
    except ImportError:
        pass
    info["attn_impl"] = getattr(model.config, "_attn_implementation", None) or "<unset>"
    # Sample the first attention block's q_proj class to confirm Linear4bit is active.
    for name, mod in model.named_modules():
        if name.endswith("q_proj"):
            info["q_proj_class"] = type(mod).__name__
            break
    return info


def main() -> None:
    assert torch.cuda.is_available(), "CUDA not available"

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"[load] {MODEL_ID} (bnb 4-bit NF4, bf16 compute)")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"
    model = AutoModel.from_pretrained(
        MODEL_ID, trust_remote_code=True, device_map="auto", quantization_config=bnb_cfg,
    )
    model.eval()
    load_s = time.perf_counter() - t0
    print(f"[load] done in {load_s:.1f}s, VRAM: {torch.cuda.max_memory_allocated()/2**30:.2f} GB")

    runtime = describe_runtime(model)
    print("[runtime]", json.dumps(runtime, indent=None))

    # Pre-tokenize once.
    chat = tokenizer.apply_chat_template([{"role": "user", "content": PROMPT}], add_generation_prompt=True, tokenize=False)
    enc = tokenizer(chat, add_special_tokens=False, return_tensors="pt", padding=True)
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc["attention_mask"].to(model.device)
    prompt_tokens = int(input_ids.shape[-1])
    print(f"[prompt] tokens={prompt_tokens}")

    results: list[dict] = []
    # Warmup pass — first forward is always slow (Triton kernel JIT etc).
    print("[warmup] 1 short forward")
    _ = llada_generate(model, input_ids, attention_mask, steps=8, gen_length=8, block_length=8)
    torch.cuda.synchronize()

    for block_length, steps in CONFIGS:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out, n_fwd = llada_generate(model, input_ids, attention_mask, steps=steps, gen_length=GEN_LENGTH, block_length=block_length)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        tps = GEN_LENGTH / wall
        per_fwd_ms = wall / n_fwd * 1e3
        peak_vram = torch.cuda.max_memory_allocated() / 2**30
        text = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
        snippet = (text[:80] + "…") if len(text) > 80 else text
        snippet = snippet.replace("\n", " ⏎ ")
        row = {
            "block_length": block_length, "steps": steps, "n_fwd": n_fwd,
            "wall_s": round(wall, 2), "tok_per_s": round(tps, 1),
            "ms_per_fwd": round(per_fwd_ms, 1), "peak_vram_gb": round(peak_vram, 2),
            "snippet": snippet,
        }
        results.append(row)
        print(f"[run] block={block_length:3d} steps={steps:3d} n_fwd={n_fwd:3d} "
              f"wall={wall:5.2f}s tps={tps:5.1f} ms/fwd={per_fwd_ms:5.1f} "
              f"vram={peak_vram:.2f}GB | {snippet}")

    out_path = Path(__file__).parent.parent / "results" / "llada8b_sweep.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps({"runtime": runtime, "results": results}, indent=2))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
