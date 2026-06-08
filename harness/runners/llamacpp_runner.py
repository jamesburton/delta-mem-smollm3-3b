"""Run one llama.cpp / GGUF cell end-to-end.

Mirrors `hf_runner.run` semantics so the same `RunConfig` and result-dict
shape land in the per-cell JSON, and `scripts/context_sweep.py`'s summary
table renders both stacks uniformly.

Memory accounting is via `nvidia-smi` deltas because llama.cpp allocates
GPU memory outside PyTorch (cuBLAS / its own allocator), so the
`torch.cuda.*` hooks the HF runner uses see nothing. Coarser than the HF
metric, but it's the cleanest cross-stack signal we have.

Cell → GGUF mapping (in `_QUANT_BY_CELL`): 9a=Q4_K_M, 9b=Q5_K_M, 9c=Q8_0.
"""

from __future__ import annotations

import datetime
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from ..cells import Cell
from ..metrics import quality


_QUANT_BY_CELL: Dict[str, str] = {
    "9a": "Q4_K_M",
    "9b": "Q5_K_M",
    "9c": "Q8_0",
}


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z")


def _gpu_mem_used_bytes(gpu_index: int = 0) -> int:
    """Bytes used on GPU 0 right now, per nvidia-smi. Returns 0 if unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
             "--id", str(gpu_index)],
            text=True, timeout=5,
        ).strip().splitlines()
        if not out:
            return 0
        return int(out[0]) * 1024 * 1024
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return 0


def _resolve_gguf_path(repo_id: str, quant: str) -> Path:
    """Download (or reuse cached) the GGUF file for this quant tag.

    File naming inside the unsloth GGUF repos drops repo qualifiers like
    "-MTP" and "-GGUF" — e.g. `unsloth/Qwen3.5-4B-MTP-GGUF` contains
    `Qwen3.5-4B-Q5_K_M.gguf`, not `Qwen3.5-4B-MTP-Q5_K_M.gguf`.
    """
    from huggingface_hub import hf_hub_download
    base = repo_id.split("/")[-1]
    for suffix in ("-MTP-GGUF", "-GGUF", "-MTP"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    fname = f"{base}-{quant}.gguf"
    return Path(hf_hub_download(repo_id=repo_id, filename=fname))


def run(cell: Cell, base_cfg) -> Dict[str, Any]:
    """RunConfig from hf_runner is reused — same dataclass, same field names.

    Behavior:
      1. Resolve which GGUF file this cell uses.
      2. Snapshot nvidia-smi baseline.
      3. Load the model with all layers on GPU (n_gpu_layers=-1) and
         n_ctx sized to fit the prompt + decode.
      4. Snapshot post-load memory.
      5. Build the multineedle task, run a single generation, time it.
      6. Snapshot post-decode memory + capture llama.cpp's timing.
      7. Score NIH against the answer.
    """
    from llama_cpp import Llama

    quant = _QUANT_BY_CELL.get(cell.id)
    if quant is None:
        return _failed_record(cell, base_cfg, f"no quant mapping for cell {cell.id}")

    print(f"  resolving GGUF: {cell.base_model} @ {quant}")
    try:
        gguf_path = _resolve_gguf_path(cell.base_model, quant)
    except Exception as e:
        return _failed_record(cell, base_cfg, f"GGUF resolve failed: {e!r}")
    print(f"  using {gguf_path.name} ({gguf_path.stat().st_size/1024**3:.2f} GB)")

    task_type = getattr(base_cfg, "task_type", "multineedle")
    if task_type == "hard_multineedle":
        n_needles = base_cfg.n_needles if base_cfg.n_needles != 3 else 10
        task = quality.make_hard_multineedle_task(
            target_tokens=base_cfg.target_tokens,
            n_needles=n_needles,
            n_distractors=getattr(base_cfg, "n_distractors", 30),
            seed=base_cfg.seed,
        )
    else:
        task = quality.make_multineedle_task(
            target_tokens=base_cfg.target_tokens,
            n_needles=base_cfg.n_needles,
            seed=base_cfg.seed,
        )
    prompt = task.context + "\n\n" + task.question

    # Size the context window. NIH-task token estimates use word-count as a
    # proxy; actual tokens come in at ~2× target. Add the decode budget.
    # Round up to the next 2K-aligned boundary, capped at 32K (model native).
    rough_prompt_tokens = base_cfg.target_tokens * 2
    n_ctx_needed = rough_prompt_tokens + base_cfg.max_new_tokens + 512
    n_ctx = min(32768, max(4096, ((n_ctx_needed + 2047) // 2048) * 2048))
    print(f"  n_ctx={n_ctx}  (prompt budget ~{rough_prompt_tokens} + decode {base_cfg.max_new_tokens})")

    mem_before_load = _gpu_mem_used_bytes()

    t_load_start = time.perf_counter()
    llm = Llama(
        model_path=str(gguf_path),
        n_ctx=n_ctx,
        n_gpu_layers=-1,            # all layers on GPU
        n_batch=512,
        verbose=False,
        logits_all=False,
        embedding=False,
    )
    t_load_end = time.perf_counter()
    mem_after_load = _gpu_mem_used_bytes()

    print(f"  loaded in {t_load_end-t_load_start:.1f}s; "
          f"GPU used: {mem_after_load/1024**3:.2f} GiB "
          f"(+{(mem_after_load-mem_before_load)/1024**3:.2f} GiB)")

    # Some llama.cpp builds tokenize lazily; force one tokenize call for an
    # actual count, which we also use to verify the prompt fits.
    try:
        prompt_tokens = len(llm.tokenize(prompt.encode("utf-8"), add_bos=True, special=False))
    except Exception:
        prompt_tokens = -1
    print(f"  prompt tokens: {prompt_tokens}  (vs n_ctx {n_ctx})")

    t_gen_start = time.perf_counter()
    try:
        out = llm(
            prompt,
            max_tokens=base_cfg.max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            stop=None,
        )
    except Exception as e:
        # Most common cause: prompt longer than n_ctx. Free model and exit.
        llm.close() if hasattr(llm, "close") else None
        return _failed_record(
            cell, base_cfg,
            f"generation failed: {e!r}; prompt_tokens={prompt_tokens}, n_ctx={n_ctx}",
            extra={"memory": {"post_load_used_vram_bytes": mem_after_load,
                              "pre_load_used_vram_bytes": mem_before_load}},
        )
    t_gen_end = time.perf_counter()
    mem_after_decode = _gpu_mem_used_bytes()

    answer = out["choices"][0]["text"]
    usage = out.get("usage", {}) or {}
    completion_tokens = usage.get("completion_tokens", 0) or 0
    decode_seconds = t_gen_end - t_gen_start  # includes prefill — see below
    # llama-cpp returns one timing for the entire call; we don't get an
    # explicit prefill-vs-decode split unless we drive eval() manually.
    # Report decode_tokens_per_second over generated tokens.
    dec_tps = (completion_tokens / decode_seconds) if decode_seconds > 0 else 0.0

    print(f"  generated {completion_tokens} tokens in {decode_seconds:.1f}s = "
          f"{dec_tps:.1f} tok/s; GPU used after: {mem_after_decode/1024**3:.2f} GiB")

    if task_type == "hard_multineedle":
        s = quality.score_hard_multineedle(task, answer)
        quality_payload = {
            "hard_multineedle": {
                "per_needle_correct": s.per_needle_correct,
                "n_needles": s.n_needles,
                "n_distractors": s.n_distractors,
                "distractors_mentioned": s.distractors_mentioned,
                "fraction_correct": s.fraction_correct,
                "recall_all": s.recall_all,
                "precision_against_distractors": s.precision_against_distractors,
            },
        }
        recall_signal = any(s.per_needle_correct)
    else:
        s = quality.score_multineedle(task, answer)
        quality_payload = {
            "multineedle": {
                "per_needle": s.per_needle,
                "recall_all": s.recall_all,
                "recall_any": s.recall_any,
                "fraction": s.fraction,
            },
        }
        recall_signal = s.recall_any

    try:
        llm.close() if hasattr(llm, "close") else None
    except Exception:
        pass

    return {
        "cell_id": cell.id,
        "title": cell.title,
        "status": "ok" if recall_signal else "partial",
        "quality": {**quality_payload, "perplexity": None},
        "memory": {
            # nvidia-smi deltas — coarser than HF runner's torch.cuda hooks
            # because llama.cpp allocates outside PyTorch. Use these as the
            # cross-stack comparison number.
            "peak_vram_bytes": max(mem_after_load, mem_after_decode),
            "kv_cache_bytes_at_target_len": 0,
            "prefill_peak_vram_bytes": None,
            "decode_peak_vram_bytes": mem_after_decode,
            "decode_resident_vram_bytes": mem_after_decode,
            "pre_load_used_vram_bytes": mem_before_load,
            "post_load_used_vram_bytes": mem_after_load,
        },
        "speed": {
            "prefill_seconds": None,    # not split by llama-cpp's single call
            "ttft_seconds": None,
            "decode_seconds": decode_seconds,
            "new_tokens": completion_tokens,
            "decode_tokens_per_second": dec_tps,
        },
        "config": {
            "cell": asdict(cell) if hasattr(cell, "__dataclass_fields__") else cell.__dict__,
            "run": {
                "target_tokens": base_cfg.target_tokens,
                "n_needles": base_cfg.n_needles,
                "max_new_tokens": base_cfg.max_new_tokens,
                "dtype": "gguf",
                "device": "cuda",
                "gguf_quant": quant,
                "n_ctx": n_ctx,
                "n_gpu_layers": -1,
            },
            "backbone": {
                "model_id": cell.base_model,
                "gguf_file": gguf_path.name,
                "stack": "llamacpp",
            },
        },
        # Capture more text for reasoning models (Qwen3.5 emits <think> tags
        # that can be hundreds of tokens long before the final answer).
        "answer_preview": answer[:4096],
        "timestamp": _utc_now_iso(),
    }


def _failed_record(cell: Cell, base_cfg, err: str,
                   extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    rec: Dict[str, Any] = {
        "cell_id": cell.id,
        "title": cell.title,
        "status": "failed",
        "error": err,
    }
    if extra:
        rec.update(extra)
    return rec
