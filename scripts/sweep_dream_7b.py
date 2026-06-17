"""Dream-v0-Instruct-7B config sweep on RTX 3060.

Mirrors ``sweep_llada_8b.py`` so the two diffusion LLMs are directly
comparable, with one key difference: Dream's sampler has **no block_length
knob** — it denoises across the full generation window every step. The
performance knob is ``steps``; the quality knob is ``alg`` (the
confidence/transfer strategy). We sweep both.

Output JSON layout mirrors ``llada8b_sweep.json``:

    {
      "runtime": {...},
      "results": [
        {"alg": "...", "steps": int, "n_fwd": int, "wall_s": ..., "tok_per_s": ..., ...},
        ...
      ]
    }

The ``block_length`` field is set to ``null`` for every row since Dream has no
semi-AR block structure to vary. The step counts {128, 64, 32, 16} match the
LLaDA-8B sweep so per-step costs can be compared apples-to-apples.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Dream-org/Dream-v0-Instruct-7B"
MASK_TOKEN_ID = 151666
PROMPT = (
    "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 "
    "kilometers per hour. How many kilometers can she run in 8 hours?"
)
GEN_LENGTH = 128

# (alg, steps) configs. Step counts match the LLaDA-8B sweep so per-step ms
# can be compared apples-to-apples. ``origin`` is the upstream default
# (random-transfer); the other three are confidence-based variants. We sweep
# every step count under ``entropy`` (consistently the strongest in HKU's
# blog) plus a couple of cross-checks for ``origin`` and ``maskgit_plus`` at
# the recommended low-step regime.
CONFIGS: list[tuple[str, int]] = [
    ("entropy", 128),       # dense / high-quality baseline
    ("entropy", 64),
    ("entropy", 32),
    ("entropy", 16),        # most aggressive (matches LLaDA-8B best @ 16 steps)
    ("origin", 64),         # upstream default sampler
    ("maskgit_plus", 32),   # confidence-based, mid step count
    ("topk_margin", 32),    # margin-based confidence
]


def describe_runtime(model) -> dict:
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
    # CRITICAL: disable KV cache (diffusion re-evaluates whole sequence each step)
    # and re-init the rope `inv_freq` buffers, which transformers 5.x's
    # meta-loader leaves as uninitialized memory under bnb 4-bit. See
    # smoke_dream_7b.py for the full diagnostic story.
    model.config.use_cache = False
    model.reset_rope_parameters()
    load_s = time.perf_counter() - t0
    print(f"[load] done in {load_s:.1f}s, VRAM: {torch.cuda.max_memory_allocated()/2**30:.2f} GB")

    runtime = describe_runtime(model)
    print("[runtime]", json.dumps(runtime, indent=None))

    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}], add_generation_prompt=True, tokenize=False,
    )
    enc = tokenizer(chat, add_special_tokens=False, return_tensors="pt", padding=True)
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc["attention_mask"].to(model.device)
    prompt_tokens = int(input_ids.shape[-1])
    print(f"[prompt] tokens={prompt_tokens}")

    # transformers 5.x's `GenerationConfig.from_model_config` chokes on
    # DreamGenerationConfig's custom `eps` field; build the gen_cfg explicitly
    # for each run instead of leaning on the model.generation_config default.
    from importlib import import_module
    gen_mod = import_module(model.__class__.__module__.rsplit(".", 1)[0] + ".generation_utils")

    def make_gen_cfg(steps: int, alg: str, gen_length: int = GEN_LENGTH):
        # Per HKUNLP/Dream README: temperature=0.2, top_p=0.95, alg_temp=0.0.
        # Pure-greedy (temperature=0) produces degenerate output on this model.
        return gen_mod.DreamGenerationConfig(
            max_new_tokens=gen_length, steps=steps, alg=alg, alg_temp=0.0,
            temperature=0.2, top_p=0.95,
            mask_token_id=MASK_TOKEN_ID,
            eos_token_id=model.config.eos_token_id,
            pad_token_id=model.config.pad_token_id,
            bos_token_id=model.config.bos_token_id,
        )

    results: list[dict] = []
    # Warmup pass — first forward is always slow (cuBLAS / Triton kernel JIT).
    print("[warmup] 1 short diffusion_generate")
    _ = model.diffusion_generate(
        input_ids, attention_mask=attention_mask,
        generation_config=make_gen_cfg(steps=4, alg="entropy", gen_length=16),
    )
    torch.cuda.synchronize()

    for alg, steps in CONFIGS:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.diffusion_generate(
            input_ids, attention_mask=attention_mask,
            generation_config=make_gen_cfg(steps=steps, alg=alg),
        )
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0

        # Dream's sampler runs exactly `steps` forward passes (one per
        # denoising step). Unlike LLaDA's semi-AR sampler there are no
        # per-block sub-iterations, so n_fwd == steps.
        n_fwd = steps
        tps = GEN_LENGTH / wall
        per_fwd_ms = wall / n_fwd * 1e3
        peak_vram = torch.cuda.max_memory_allocated() / 2**30
        text = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
        snippet = (text[:80] + "…") if len(text) > 80 else text
        snippet = snippet.replace("\n", " ⏎ ")
        row = {
            "block_length": None, "alg": alg, "steps": steps, "n_fwd": n_fwd,
            "wall_s": round(wall, 2), "tok_per_s": round(tps, 1),
            "ms_per_fwd": round(per_fwd_ms, 1), "peak_vram_gb": round(peak_vram, 2),
            "snippet": snippet,
        }
        results.append(row)
        print(f"[run] alg={alg:14s} steps={steps:3d} n_fwd={n_fwd:3d} "
              f"wall={wall:5.2f}s tps={tps:5.1f} ms/fwd={per_fwd_ms:5.1f} "
              f"vram={peak_vram:.2f}GB | {snippet}")

    out_path = Path(__file__).parent.parent / "results" / "dream7b_sweep.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps({"runtime": runtime, "results": results}, indent=2))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
