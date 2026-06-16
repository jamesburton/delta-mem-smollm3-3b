"""LLaDA-8B-Instruct smoke test on RTX 3060 12 GB.

Tries multiple load paths (gptqmodel > transformers GPTQConfig > bnb 4-bit) and
runs the canonical mask-denoising diffusion sampler from the upstream ML-GSAI
generate.py (reproduced inline to avoid a runtime download).

Usage:
    C:\\Python311\\python.exe scripts\\smoke_llada_8b.py [--bf16-base]

By default loads FunAGI/LLaDA-8B-Instruct-gptqmodel-4bit (5 GB). With --bf16-base
falls back to GSAI-ML/LLaDA-8B-Instruct in bnb 4-bit (downloads ~16 GB).
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

GPTQ_MODEL_ID = "FunAGI/LLaDA-8B-Instruct-gptqmodel-4bit"
BF16_MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"

MASK_ID = 126336  # LLaDA [MASK] token
PROMPT = (
    "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 "
    "kilometers per hour. How many kilometers can she run in 8 hours?"
)
STEPS = 64
GEN_LENGTH = 128
BLOCK_LENGTH = 32


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
def llada_generate(
    model,
    prompt: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    steps: int = STEPS,
    gen_length: int = GEN_LENGTH,
    block_length: int = BLOCK_LENGTH,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = MASK_ID,
) -> torch.Tensor:
    """Canonical LLaDA semi-AR mask-denoising sampler (ML-GSAI/LLaDA/generate.py)."""
    x = torch.full(
        (prompt.shape[0], prompt.shape[1] + gen_length),
        mask_id,
        dtype=torch.long,
        device=model.device,
    )
    x[:, : prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (prompt.shape[0], gen_length),
                    dtype=attention_mask.dtype,
                    device=model.device,
                ),
            ],
            dim=-1,
        )

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    per_block_steps = steps // num_blocks

    for nb in range(num_blocks):
        b_start = prompt.shape[1] + nb * block_length
        b_end = prompt.shape[1] + (nb + 1) * block_length
        block_mask_index = x[:, b_start:b_end] == mask_id
        num_transfer = get_num_transfer_tokens(block_mask_index, per_block_steps)

        for i in range(per_block_steps):
            mask_index = x == mask_id
            logits = model(x, attention_mask=attention_mask).logits
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == "low_confidence":
                p = F.softmax(logits, dim=-1)
                x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            elif remasking == "random":
                x0_p = torch.rand_like(x0, dtype=torch.float32)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (nb + 1) * block_length :] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, -np.inf))

            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            for j in range(confidence.shape[0]):
                _, sel = torch.topk(confidence[j], k=int(num_transfer[j, i]))
                transfer_index[j, sel] = True
            x[transfer_index] = x0[transfer_index]

    return x


def load_gptq(model_id: str):
    """Load via transformers AutoModel.

    gptqmodel's high-level GPTQModel.from_quantized has no LLaDA model
    definition. With optimum + gptqmodel installed, transformers'
    AutoModel.from_pretrained dispatches GPTQ kernels at the Linear layer
    level instead — which works because the custom LLaDA modeling code uses
    plain nn.Linear modules.
    """
    print(f"[load] {model_id} via transformers (optimum + gptqmodel kernels)")
    model = AutoModel.from_pretrained(
        model_id, trust_remote_code=True, device_map="auto"
    )
    return model, None


def load_bf16_bnb4(model_id: str):
    from transformers import BitsAndBytesConfig
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    print(f"[load] {model_id} via bnb 4-bit NF4")
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=bnb_cfg,
    )
    return model, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16-base", action="store_true",
                    help="Skip GPTQ and use bnb 4-bit on the bf16 GSAI base.")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA not available"
    torch.cuda.reset_peak_memory_stats()

    model_id = BF16_MODEL_ID if args.bf16_base else GPTQ_MODEL_ID

    t0 = time.perf_counter()
    if args.bf16_base:
        model, tok_from_loader = load_bf16_bnb4(model_id)
    else:
        model, tok_from_loader = load_gptq(model_id)
    model.eval()
    load_s = time.perf_counter() - t0

    tokenizer = tok_from_loader or AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"
    assert tokenizer.pad_token_id != MASK_ID

    after_load_vram_gb = torch.cuda.max_memory_allocated() / 2**30
    print(f"[load] done in {load_s:.1f}s, VRAM after load: {after_load_vram_gb:.2f} GB")

    chat_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        add_generation_prompt=True,
        tokenize=False,
    )
    enc = tokenizer(chat_prompt, add_special_tokens=False, return_tensors="pt", padding=True)
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc["attention_mask"].to(model.device)
    prompt_tokens = int(input_ids.shape[-1])

    print(f"[gen] prompt_tokens={prompt_tokens} steps={STEPS} "
          f"gen_length={GEN_LENGTH} block_length={BLOCK_LENGTH}")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = llada_generate(
        model, input_ids, attention_mask,
        steps=STEPS, gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH,
        temperature=0.0, remasking="low_confidence",
    )
    torch.cuda.synchronize()
    gen_s = time.perf_counter() - t0

    new_tokens = int(out.shape[-1]) - prompt_tokens
    peak_vram_gb = torch.cuda.max_memory_allocated() / 2**30
    tok_per_s = new_tokens / gen_s if gen_s > 0 else 0.0

    text = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
    print("---")
    print(text)
    print("---")
    print(
        f"[result] new_tokens={new_tokens} wall={gen_s:.2f}s "
        f"tok/s={tok_per_s:.1f} peak_vram={peak_vram_gb:.2f} GB"
    )


if __name__ == "__main__":
    main()
