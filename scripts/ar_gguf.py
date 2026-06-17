"""Qwen3-4B GGUF / llama.cpp AR runner — same prompt + decode config as ar_baseline.py.

Tests whether llama.cpp's fused 4-bit kernels (Q4_K_M / Q5_K_M) beat the
bnb-4bit dequant floor on T1 (RTX 3060, 3.0 tok/s short-decode baseline).

Stack: the PROJECT VENV (`E:\\Development\\llm-model-tests\\delta-mem-smollm3-3b\\.venv\\`),
NOT the system Python311. The venv has:
  - torch 2.9.0+cu128
  - transformers 5.9.0
  - flash_attn 2.8.3 (kingbri1 prebuilt wheel)
  - llama-cpp-python 0.3.28 BUILT FROM SOURCE WITH CUDA (verified: GPU offload True)
The source-build is necessary because this box has Xeon X5670 (no AVX) and
prebuilt llama-cpp-python wheels crash with WinError 0xC000001D ILLEGAL_INSTRUCTION.
See `docs/local-llama-cpp-build.md`.

The system Python311 has no llama_cpp; the venv has no bnb/gptqmodel.
That's why this exists as a separate script from `ar_marlin.py`.

Checkpoint: unsloth/Qwen3-4B-Instruct-2507-GGUF
  - Q4_K_M: 2.5 GB (primary — strong quality/size tradeoff)
  - Q5_K_M: 2.89 GB (optional secondary)
  - File naming: `Qwen3-4B-Instruct-2507-<QUANT>.gguf`

Disk note: as of 2026-06-17 we have ~14 GB free on E:. Q4_K_M (2.5 GB) +
GPTQ-Int4 (2.67 GB) = 5.2 GB; safe. Don't pull Q8_0 (4.28 GB) unless we
explicitly free space first.

Prompt + decode config mirrors `scripts/ar_baseline.py` so output slots
directly into the `results/qwen3_4b_ar_fa2_sweep.json`-shaped table.
"""
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

# Mirror ar_baseline.py exactly so the comparison is fair.
PROMPT = (
    "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 "
    "kilometers per hour. How many kilometers can she run in 8 hours?"
)
NEW_TOKENS = 128
REPO_ID = "unsloth/Qwen3-4B-Instruct-2507-GGUF"


def gpu_mem_used_gib() -> float:
    """Bytes used on GPU 0 right now, per nvidia-smi. Returns 0 on failure."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                "--id",
                "0",
            ],
            text=True,
            timeout=5,
        ).strip()
        return float(out.splitlines()[0]) / 1024.0  # MiB → GiB
    except Exception:
        return 0.0


def resolve_gguf(quant: str) -> Path:
    """Download (or reuse cached) the GGUF for the requested quant.

    Naming matches the existing harness/runners/llamacpp_runner.py resolver:
    `<base-without-GGUF>-<QUANT>.gguf`, i.e. for repo
    `unsloth/Qwen3-4B-Instruct-2507-GGUF` and quant Q4_K_M:
    `Qwen3-4B-Instruct-2507-Q4_K_M.gguf`.
    """
    from huggingface_hub import hf_hub_download
    base = REPO_ID.split("/")[-1]
    for suffix in ("-GGUF",):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    fname = f"{base}-{quant}.gguf"
    print(f"[hf] resolving {REPO_ID}/{fname}")
    p = Path(hf_hub_download(repo_id=REPO_ID, filename=fname))
    print(f"[hf] using {p} ({p.stat().st_size/1024**3:.2f} GB)")
    return p


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quant",
        default="Q4_K_M",
        choices=["Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q6_K", "Q8_0"],
        help="GGUF quant tag (default Q4_K_M)",
    )
    parser.add_argument(
        "--ctx",
        type=int,
        default=4096,
        help="n_ctx for llama.cpp KV-cache allocation (default 4096; "
        "natural prompt is ~45 tokens so this leaves room and matches "
        "the ar_baseline.py short-ctx 4-cell matrix).",
    )
    parser.add_argument(
        "--new-tokens",
        type=int,
        default=NEW_TOKENS,
        help=f"tokens to generate (default {NEW_TOKENS}, matches ar_baseline.py)",
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=-1,
        help="layers to offload to GPU (-1 = all; default -1)",
    )
    parser.add_argument(
        "--n-batch",
        type=int,
        default=512,
        help="prefill batch size (default 512)",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="skip the warmup generation (default: warm with 8 tokens)",
    )
    args = parser.parse_args()

    # Import is here (not at module top) so --help works without llama_cpp.
    try:
        from llama_cpp import Llama
    except ImportError as e:
        print(
            "[fatal] llama_cpp not importable. Are you running in the project "
            "venv? Try:\n  "
            "E:\\Development\\llm-model-tests\\delta-mem-smollm3-3b\\.venv\\Scripts\\python.exe "
            f"{__file__}",
            flush=True,
        )
        raise SystemExit(2) from e

    gguf_path = resolve_gguf(args.quant)

    mem_before = gpu_mem_used_gib()
    print(f"[gpu] before load: {mem_before:.2f} GiB")

    t0 = time.perf_counter()
    llm = Llama(
        model_path=str(gguf_path),
        n_ctx=args.ctx,
        n_gpu_layers=args.n_gpu_layers,
        n_batch=args.n_batch,
        verbose=False,
        logits_all=False,
        embedding=False,
    )
    t_load = time.perf_counter() - t0
    mem_after_load = gpu_mem_used_gib()
    print(
        f"[load] done in {t_load:.1f}s, GPU used: {mem_after_load:.2f} GiB "
        f"(+{mem_after_load - mem_before:.2f})"
    )

    # Build the same chat-template prompt the AR baseline uses, so the prompt
    # token count and content match. llama-cpp doesn't auto-apply the chat
    # template — we resolve it via the model's tokenizer.json which ships in
    # the GGUF (unsloth bakes the chat template in). For Qwen3-Instruct the
    # standard wrapper looks like:
    #   <|im_start|>user\n{PROMPT}<|im_end|>\n<|im_start|>assistant\n
    # We use the explicit template here rather than letting llama-cpp
    # autopopulate it (some unsloth builds don't expose the template via
    # `apply_chat_template`).
    prompt_text = (
        f"<|im_start|>user\n{PROMPT}<|im_end|>\n<|im_start|>assistant\n"
    )

    # Tokenize once to report the prompt token count for the comparison table.
    try:
        prompt_tokens = len(
            llm.tokenize(prompt_text.encode("utf-8"), add_bos=True, special=True)
        )
    except Exception:
        prompt_tokens = -1
    print(f"[prompt] tokens={prompt_tokens}  (n_ctx={args.ctx})")

    # Warmup with 8 tokens to JIT cuBLAS / amortize first-call overhead.
    if not args.no_warmup:
        _ = llm(prompt_text, max_tokens=8, temperature=0.0, top_p=1.0)

    # The actual measurement: time end-to-end at temperature=0 (greedy).
    # Note: llama-cpp's single timing includes prefill. At ctx≈45 tokens
    # prefill is negligible (<50 ms). Long-ctx comparisons need to split
    # this — they don't here.
    t0 = time.perf_counter()
    out = llm(
        prompt_text,
        max_tokens=args.new_tokens,
        temperature=0.0,
        top_p=1.0,
        stop=None,
    )
    wall = time.perf_counter() - t0
    mem_after_decode = gpu_mem_used_gib()

    text = out["choices"][0]["text"]
    usage = out.get("usage", {}) or {}
    completion_tokens = usage.get("completion_tokens", 0) or 0
    tps = (completion_tokens / wall) if wall > 0 else 0.0

    snippet = (text[:160] + "…") if len(text) > 160 else text
    print(f"---\n{snippet.strip()}\n---")
    # SAME format string shape as ar_baseline.py for grep-friendly comparison.
    print(
        f"[result] quant=gguf-{args.quant} attn=llamacpp-cuda "
        f"ctx={prompt_tokens} new={completion_tokens} wall={wall:.2f}s "
        f"tok/s={tps:.1f} peak_vram={mem_after_decode:.2f} GB "
        f"ms/tok={wall/max(completion_tokens,1)*1e3:.1f}"
    )

    # Close cleanly so the next quant run can re-allocate without leaks.
    try:
        if hasattr(llm, "close"):
            llm.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
