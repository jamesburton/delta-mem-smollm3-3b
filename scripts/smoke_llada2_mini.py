"""LLaDA2.1-mini smoke test on RTX 3060 12 GB.

Loads inclusionAI/LLaDA2.1-mini in 4-bit (bitsandbytes NF4) via custom_code,
runs one chat-template generation with the README's recommended args, and
reports peak VRAM, wall-clock decode time, tokens/sec, and the answer.

Usage:
    C:\\Python311\\python.exe scripts\\smoke_llada2_mini.py
"""

from __future__ import annotations

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "inclusionAI/LLaDA2.1-mini"
PROMPT = "Calculate 1+5-28*0.5-200=?"
GEN_LENGTH = 256
BLOCK_LENGTH = 32


def main() -> None:
    assert torch.cuda.is_available(), "CUDA not available"
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"[load] {MODEL_ID} (4-bit NF4, bf16 compute)")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=bnb_cfg,
    )
    model.eval()
    load_s = time.perf_counter() - t0
    after_load_vram_gb = torch.cuda.max_memory_allocated(device) / 2**30
    print(f"[load] done in {load_s:.1f}s, VRAM after load: {after_load_vram_gb:.2f} GB")

    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    ).to(device)
    prompt_tokens = int(input_ids.shape[-1])

    print(f"[gen] prompt_tokens={prompt_tokens} gen_length={GEN_LENGTH} block_length={BLOCK_LENGTH}")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(
            inputs=input_ids,
            eos_early_stop=True,
            gen_length=GEN_LENGTH,
            block_length=BLOCK_LENGTH,
            threshold=0.5,
            editing_threshold=0.0,
            temperature=0.0,
        )
    torch.cuda.synchronize()
    gen_s = time.perf_counter() - t0
    out_tokens = int(out.shape[-1]) - prompt_tokens
    peak_vram_gb = torch.cuda.max_memory_allocated(device) / 2**30
    tok_per_s = out_tokens / gen_s if gen_s > 0 else 0.0

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    print("---")
    print(text)
    print("---")
    print(
        f"[result] new_tokens={out_tokens} wall={gen_s:.2f}s "
        f"tok/s={tok_per_s:.1f} peak_vram={peak_vram_gb:.2f} GB"
    )


if __name__ == "__main__":
    main()
