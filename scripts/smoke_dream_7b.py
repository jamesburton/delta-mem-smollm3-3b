"""Dream-v0-Instruct-7B smoke test on RTX 3060 12 GB.

Dream is a block / non-block diffusion LLM from HKU NLP. Unlike LLaDA-8B, Dream
ships its own ``diffusion_generate`` method on the model via the
``DreamGenerationMixin`` (see ``generation_utils.py`` in the HF repo). This
script loads in bnb 4-bit NF4 / bf16 compute (same recipe that worked for
LLaDA-8B) and runs a single 128-token generation as a sanity check.

Usage:
    C:\\Python311\\python.exe -X utf8 scripts\\smoke_dream_7b.py

Notes:
- Dream's mask token id is 151666 (per ``config.json``); the shipped
  ``generation_config.json`` leaves ``mask_token_id=null``, so we set it
  explicitly via kwargs to ``diffusion_generate``.
- Dream has no ``block_length`` knob — the sampler is fully-parallel
  denoising across all generation positions. The speed knob is ``steps``.
- ``alg`` switches between four algorithms:
    * ``origin``         — random-transfer (the default; non-confidence-based)
    * ``maskgit_plus``   — top-confidence transfer
    * ``topk_margin``    — top-margin (p1 - p2) transfer
    * ``entropy``        — neg-entropy confidence transfer
"""

from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Dream-org/Dream-v0-Instruct-7B"
MASK_TOKEN_ID = 151666  # from config.json
PROMPT = (
    "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 "
    "kilometers per hour. How many kilometers can she run in 8 hours?"
)
STEPS = 64
GEN_LENGTH = 128
ALG = "entropy"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16", action="store_true",
                    help="Use bf16 weights instead of bnb 4-bit (needs ~15 GB VRAM).")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--gen-length", type=int, default=GEN_LENGTH)
    ap.add_argument("--alg", type=str, default=ALG,
                    choices=["origin", "maskgit_plus", "topk_margin", "entropy"])
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA not available"
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"

    if args.bf16:
        print(f"[load] {MODEL_ID} bf16 (no quant)")
        model = AutoModel.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    else:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        print(f"[load] {MODEL_ID} via bnb 4-bit NF4 (bf16 compute)")
        model = AutoModel.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            device_map="auto",
            quantization_config=bnb_cfg,
        )
    model.eval()
    # CRITICAL for diffusion: disable KV cache. Dream's config defaults
    # use_cache=True (for AR mode), but the diffusion sampler re-evaluates the
    # full sequence every step. Carrying a KV cache across steps gives stale
    # logits and produces garbage output. We must force it off.
    model.config.use_cache = False
    # CRITICAL on bnb 4-bit + transformers 5.x: the RoPE `inv_freq` buffers
    # are non-persistent (not in safetensors), but transformers' meta-device
    # weight loader leaves the buffers as uninitialized memory after load —
    # garbage values. Symptom: model outputs `<|endoftext|>` floods or word
    # salad. Re-running the rope init function fixes both the base and
    # per-layer rotary embeddings.
    model.reset_rope_parameters()
    load_s = time.perf_counter() - t0
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

    # transformers 5.x `GenerationConfig.from_model_config` chokes on
    # DreamGenerationConfig's custom `eps` field — bypass by constructing the
    # generation config explicitly so the diffusion_generate code path doesn't
    # call `from_model_config`.
    from importlib import import_module
    gen_mod = import_module(model.__class__.__module__.rsplit(".", 1)[0] + ".generation_utils")
    # Settings per the HKUNLP/Dream README example. temperature=0.2 + top_p=0.95
    # + alg_temp=0 are the recommended diffusion sampling knobs for Dream-7B;
    # pure-greedy (temperature=0) produces degenerate output on this model.
    gen_cfg = gen_mod.DreamGenerationConfig(
        max_new_tokens=args.gen_length,
        steps=args.steps,
        alg=args.alg,
        alg_temp=0.0,
        temperature=0.2,
        top_p=0.95,
        mask_token_id=MASK_TOKEN_ID,
        eos_token_id=model.config.eos_token_id,
        pad_token_id=model.config.pad_token_id,
        bos_token_id=model.config.bos_token_id,
    )

    print(f"[gen] prompt_tokens={prompt_tokens} steps={args.steps} "
          f"gen_length={args.gen_length} alg={args.alg}")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.diffusion_generate(
        input_ids,  # positional arg, matches HKUNLP/Dream README example
        attention_mask=attention_mask,
        generation_config=gen_cfg,
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
