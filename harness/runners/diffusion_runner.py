"""Diffusion-LLM runner for the multi-needle NIH harness.

This is the diffusion counterpart of ``hf_runner`` for block-diffusion LLMs
(LLaDA-8B-Instruct and, in future, Dream-7B). It is purely additive — the
existing AR runner is untouched.

Why a separate runner instead of a thin ``generate()`` shim around the LLaDA
sampler?

- The AR runner's ``_generate_with_memory_split`` uses a forward_pre_hook on
  the model and assumes call #0 = prefill, call #N>=1 = decode. Block
  diffusion has no "prefill vs decode" — every step is a single
  full-sequence forward of the same shape. Trying to bolt that onto the AR
  helper would silently produce nonsense memory numbers.

- tok/s for block diffusion is ``emitted_tokens / wall_time`` and is
  meaningful as a single number (the diffusion sampler runs a fixed
  ``steps`` count and emits ``gen_length`` tokens). There is no
  ``decode_seconds`` vs ``prefill_seconds`` split to report; we just record
  the whole-generation wall and call it ``decode_seconds`` so the existing
  ``speed.decode_tokens_per_second`` accessor works.

- The LLaDA modeling code carries its OWN ``flash_attention`` config flag
  (separate from HF's ``attn_implementation``). It does not respond to
  ``attn_implementation="flash_attention_2"``. To get FA2 you must set
  ``config.flash_attention = True`` BEFORE ``from_pretrained``. We do that
  by default when CUDA + sm_80+ + ``flash_attn`` package are present, and
  fall back to the model's own SDPA path otherwise.

Quality scoring goes through the existing
``harness.metrics.quality.score_multineedle`` /
``score_hard_multineedle`` unchanged — diffusion's answer text is the same
shape (free-form decoded text) the AR runner produces, so the needle grader
needs no changes.

Long-context warning
--------------------
LLaDA has no KV-cache. Each diffusion step is a full-sequence forward, so
per-step cost grows ``O(L^2)`` in the active-attention dimension. With the
default ``steps=16`` and ``gen_length=256`` the model runs 16 full forwards
over a sequence of length ``prompt_tokens + 256``. At the NIH harness's
``ctx=4000`` that's 16 forwards over ~4256 tokens — already ~5x the per-step
cost we measured at the ``smoke_llada_8b.py`` chat prompt of ~50 tokens.

Without FA2 the inner SDPA path will materialise an O(N^2) attention matrix
and either page through WDDM or OOM (mirroring the Qwen3-4B finding —
see ``docs/fa2-sweep-results.md``). This runner therefore enables LLaDA's
internal FA2 by default on capable hardware.
"""

from __future__ import annotations

import datetime
import time
from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..cells import Cell
from ..metrics import memory, quality

# We reuse hf_runner.RunConfig so callers can swap AR ↔ diffusion runners
# without rebuilding their config object. The runner only reads
# (target_tokens, n_needles, max_new_tokens, seed, dtype, device, task_type,
#  n_distractors) — i.e. the eval-shape fields. The AR-specific fields
# (assistant_model_id, delta_mem_adapter_id) are simply ignored here.
from .hf_runner import RunConfig  # re-exported for convenience  # noqa: F401

# The LLaDA model's [MASK] token id (constant across all LLaDA-Instruct
# checkpoints). The same value lives in ``scripts/smoke_llada_8b.py``.
MASK_ID = 126336


# ---------------------------------------------------------------------------
# Sampler (kept inline to avoid taking a runtime dep on the smoke script)
# ---------------------------------------------------------------------------


def _add_gumbel_noise(logits, temperature: float):
    import torch
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def _get_num_transfer_tokens(mask_index, steps: int):
    import torch
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    out = torch.zeros(
        mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
    ) + base
    for i in range(mask_num.size(0)):
        out[i, : remainder[i]] += 1
    return out


def llada_generate(
    model,
    prompt,
    attention_mask=None,
    *,
    steps: int = 16,
    gen_length: int = 256,
    block_length: int = 128,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = MASK_ID,
):
    """Canonical LLaDA semi-AR mask-denoising sampler.

    Returns the full sequence tensor (prompt + generation).

    Parameters mirror the LLaDA upstream ``generate.py``. The
    LLaDA-8B-Instruct best-known T1 config is
    ``block_length=128, steps=16``  (27.9 tok/s on RTX 3060,
    bnb-4bit NF4 + bf16 compute) — see ``LLMs.md``.
    """
    import torch
    import torch.nn.functional as F

    x = torch.full(
        (prompt.shape[0], prompt.shape[1] + gen_length),
        mask_id, dtype=torch.long, device=model.device,
    )
    x[:, : prompt.shape[1]] = prompt.clone()
    if attention_mask is not None:
        attention_mask = torch.cat(
            [attention_mask,
             torch.ones((prompt.shape[0], gen_length),
                        dtype=attention_mask.dtype, device=model.device)],
            dim=-1,
        )

    assert gen_length % block_length == 0, \
        f"gen_length ({gen_length}) must be divisible by block_length ({block_length})"
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0, \
        f"steps ({steps}) must be divisible by num_blocks ({num_blocks})"
    per_block_steps = steps // num_blocks

    with torch.no_grad():
        for nb in range(num_blocks):
            b_start = prompt.shape[1] + nb * block_length
            b_end = prompt.shape[1] + (nb + 1) * block_length
            block_mask_index = x[:, b_start:b_end] == mask_id
            num_transfer = _get_num_transfer_tokens(block_mask_index, per_block_steps)
            for i in range(per_block_steps):
                mask_index = x == mask_id
                logits = model(x, attention_mask=attention_mask).logits
                logits_with_noise = _add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)
                if remasking == "low_confidence":
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                elif remasking == "random":
                    x0_p = torch.rand_like(x0, dtype=torch.float32)
                else:
                    raise NotImplementedError(remasking)
                x0_p[:, prompt.shape[1] + (nb + 1) * block_length:] = -np.inf
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p,
                                         torch.full_like(x0_p, -np.inf))
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(confidence.shape[0]):
                    _, sel = torch.topk(confidence[j], k=int(num_transfer[j, i]))
                    transfer_index[j, sel] = True
                x[transfer_index] = x0[transfer_index]
    return x


# ---------------------------------------------------------------------------
# Backbone adapter
# ---------------------------------------------------------------------------


def _hardware_supports_fa2() -> bool:
    """LLaDA's FA2 path needs sm_80+ AND the flash_attn package."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        for i in range(torch.cuda.device_count()):
            cap = torch.cuda.get_device_capability(i)
            if cap[0] < 8:
                return False
        import flash_attn  # noqa: F401
        return True
    except Exception:
        return False


def load_llada(
    model_id: str = "GSAI-ML/LLaDA-8B-Instruct",
    *,
    quant: str = "bnb-4bit",
    enable_flash_attention: Optional[bool] = None,
) -> Tuple[Any, Any]:
    """Load LLaDA-8B-Instruct with the right config flags for long ctx.

    ``quant``:
      - ``"bnb-4bit"`` — bnb 4-bit NF4, bf16 compute. ~5.3 GB VRAM. Default.
      - ``"bf16"`` — native bf16. ~15 GB VRAM. Won't fit on T1 (3060 12GB).

    ``enable_flash_attention``: tri-state.
      - ``None`` (default): auto-enable when sm_80+ + flash_attn are present.
      - ``True``: force on. Fails if not supported.
      - ``False``: force off. Use torch SDPA inside LLaDA's attention.

    Returns ``(model, tokenizer)``. The tokenizer is configured with
    ``padding_side="left"``, which the LLaDA sampler requires when batched.
    """
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer, BitsAndBytesConfig

    if enable_flash_attention is None:
        enable_flash_attention = _hardware_supports_fa2()

    print(f"[load] {model_id} quant={quant} flash_attention={enable_flash_attention}")

    # Patch config.flash_attention BEFORE from_pretrained so the attention
    # modules pick it up at __init__ (post-load mutation is too late — they
    # cache `self.flash_attn_func` in __init__).
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    if hasattr(config, "flash_attention"):
        config.flash_attention = bool(enable_flash_attention)
    else:
        print("  ! config has no 'flash_attention' attribute — model may "
              "be an older LLaDA revision that doesn't expose this knob.")

    common: Dict[str, Any] = dict(
        trust_remote_code=True,
        device_map="auto",
        config=config,
    )
    if quant == "bnb-4bit":
        common["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif quant == "bf16":
        common["dtype"] = torch.bfloat16
    else:
        raise ValueError(f"unknown quant={quant!r}")

    t0 = time.perf_counter()
    model = AutoModel.from_pretrained(model_id, **common).eval()
    load_s = time.perf_counter() - t0

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.padding_side != "left":
        tok.padding_side = "left"
    assert tok.pad_token_id != MASK_ID, \
        "tokenizer pad_token_id collides with LLaDA MASK_ID — sampler will misbehave"

    vram_gb = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0
    print(f"[load] done in {load_s:.1f}s; VRAM after load: {vram_gb:.2f} GB")
    return model, tok


# ---------------------------------------------------------------------------
# Runner — same signature as hf_runner.run(cell, RunConfig) -> dict
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z")


def _build_eval_task(rc):
    if rc.task_type == "hard_multineedle":
        n = rc.n_needles if rc.n_needles != 3 else 10
        return quality.make_hard_multineedle_task(
            target_tokens=rc.target_tokens,
            n_needles=n,
            n_distractors=rc.n_distractors,
            seed=rc.seed,
        )
    return quality.make_multineedle_task(
        target_tokens=rc.target_tokens,
        n_needles=rc.n_needles,
        seed=rc.seed,
    )


def _score_eval(rc, task, answer: str):
    if rc.task_type == "hard_multineedle":
        s = quality.score_hard_multineedle(task, answer)
        payload = {
            "hard_multineedle": {
                "per_needle_correct": s.per_needle_correct,
                "n_needles": s.n_needles,
                "n_distractors": s.n_distractors,
                "distractors_mentioned": s.distractors_mentioned,
                "fraction_correct": s.fraction_correct,
                "recall_all": s.recall_all,
                "precision_against_distractors": s.precision_against_distractors,
            }
        }
        return payload, any(s.per_needle_correct)
    s = quality.score_multineedle(task, answer)
    payload = {
        "multineedle": {
            "per_needle": s.per_needle,
            "recall_all": s.recall_all,
            "recall_any": s.recall_any,
            "fraction": s.fraction,
        }
    }
    return payload, s.recall_any


def run_llada(
    cell: Cell,
    base_cfg,
    *,
    model_id: str = "GSAI-ML/LLaDA-8B-Instruct",
    steps: int = 16,
    block_length: int = 128,
    quant: str = "bnb-4bit",
    enable_flash_attention: Optional[bool] = None,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Run one NIH cell using a LLaDA-family diffusion backbone.

    Output dict shape matches ``hf_runner.run`` so the existing
    ``context_sweep.render_sweep_summary`` table renderer reads it cleanly.

    Notes on diffusion-specific fields:

    - ``speed.decode_seconds`` = total wall time of the sampler (there is no
      "prefill vs decode" — every step is one full-sequence forward).
    - ``speed.prefill_seconds`` / ``ttft_seconds`` are recorded as 0 to make
      the contract explicit: with no KV-cache, the concept doesn't apply.
    - ``speed.new_tokens`` = ``gen_length`` (the model emits exactly
      ``gen_length`` tokens, modulo trailing pad). We compute tok/s the
      diffusion way: ``new_tokens / wall``.
    - ``memory.peak_vram_bytes`` = max VRAM during the whole call. There is
      no decode/prefill split for diffusion, so the other phase-specific
      fields are left as ``None``.
    """
    import torch

    rc = base_cfg
    gen_length = max(rc.max_new_tokens, block_length)
    # Snap gen_length up to a multiple of block_length so the sampler's
    # block math holds.
    if gen_length % block_length != 0:
        gen_length = ((gen_length // block_length) + 1) * block_length

    model = None
    tok = None
    try:
        memory.reset_peak_vram()
        model, tok = load_llada(
            model_id=model_id, quant=quant,
            enable_flash_attention=enable_flash_attention,
        )

        # Build the NIH task and tokenize the chat-templated prompt.
        task = _build_eval_task(rc)
        nih_user_prompt = task.context + "\n\n" + task.question
        chat = tok.apply_chat_template(
            [{"role": "user", "content": nih_user_prompt}],
            add_generation_prompt=True, tokenize=False,
        )
        enc = tok(chat, add_special_tokens=False, return_tensors="pt", padding=True)
        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc["attention_mask"].to(model.device)
        prompt_tokens = int(input_ids.shape[-1])

        # Warmup (Triton JIT etc). Short, so we don't burn the long-ctx
        # forward budget here.
        print(f"[warmup] prompt_tokens={prompt_tokens}")
        _ = llada_generate(model, input_ids[:, :min(prompt_tokens, 256)],
                           attention_mask[:, :min(prompt_tokens, 256)],
                           steps=2, gen_length=block_length,
                           block_length=block_length, temperature=temperature)
        torch.cuda.synchronize() if torch.cuda.is_available() else None

        # Time the real generation.
        memory.reset_peak_vram()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = llada_generate(
            model, input_ids, attention_mask,
            steps=steps, gen_length=gen_length,
            block_length=block_length, temperature=temperature,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        peak_vram = memory.measure_peak_vram()

        # Decode just the new tokens (after the prompt).
        new_tokens = int(out.shape[-1]) - prompt_tokens
        answer = tok.batch_decode(
            out[:, prompt_tokens:], skip_special_tokens=True
        )[0]

        # tok/s = emitted_tokens / wall (the diffusion definition).
        tok_per_s = new_tokens / wall if wall > 0 else 0.0
        per_step_ms = (wall / steps) * 1000.0 if steps > 0 else 0.0

        quality_payload, recall_signal = _score_eval(rc, task, answer)

        # LLaDA's config uses non-canonical field names
        # (``d_model``, ``n_heads``, ``n_kv_heads``, ``n_layers``) instead of
        # the modern HF convention. ``kv_bytes_from_config`` reads the HF
        # names and crashes; for diffusion this number is also of limited
        # meaning (no KV-cache, so the "KV bytes at ctx" headline is more
        # historical than load-bearing for AR cells). Best-effort: try the
        # standard path, swallow if it fails.
        try:
            cfg_obj = getattr(model, "config", None)
            kv_bytes = memory.kv_bytes_from_config(cfg_obj, seq_len=rc.target_tokens) \
                if cfg_obj is not None else 0
        except (AttributeError, KeyError):
            kv_bytes = 0  # diffusion has no real KV cache anyway

        record = {
            "cell_id": cell.id,
            "title": cell.title,
            "status": "ok" if recall_signal else "partial",
            "quality": {
                **quality_payload,
                "perplexity": None,
            },
            "memory": {
                "peak_vram_bytes": int(peak_vram),
                "kv_cache_bytes_at_target_len": int(kv_bytes),
                # Diffusion has no decode/prefill split — leave the
                # phase-specific fields empty to be explicit.
                "prefill_peak_vram_bytes": None,
                "decode_peak_vram_bytes": int(peak_vram),
                "decode_resident_vram_bytes": None,
            },
            "speed": {
                "prefill_seconds": 0.0,
                "ttft_seconds": 0.0,
                "decode_seconds": wall,
                "new_tokens": new_tokens,
                # Diffusion tok/s = emitted_tokens / wall (no AR per-step
                # accounting). This is the same definition documented in
                # LLMs.md → "Architecture family notes → Block-diffusion
                # LLMs". Reported under the same key the AR pipeline uses
                # so context_sweep's table rendering picks it up.
                "decode_tokens_per_second": tok_per_s,
                "diffusion_steps": steps,
                "diffusion_block_length": block_length,
                "diffusion_gen_length": gen_length,
                "diffusion_ms_per_step": per_step_ms,
                "diffusion_prompt_tokens": prompt_tokens,
            },
            "config": {
                "cell": asdict(cell) if hasattr(cell, "__dataclass_fields__") else cell.__dict__,
                "run": {
                    "target_tokens": rc.target_tokens,
                    "n_needles": rc.n_needles,
                    "max_new_tokens": rc.max_new_tokens,
                    "dtype": rc.dtype, "device": rc.device,
                },
                "diffusion": {
                    "model_id": model_id,
                    "quant": quant,
                    "steps": steps,
                    "block_length": block_length,
                    "gen_length": gen_length,
                    "flash_attention": bool(
                        enable_flash_attention
                        if enable_flash_attention is not None
                        else _hardware_supports_fa2()
                    ),
                    "temperature": temperature,
                },
            },
            "answer_preview": answer[:512],
            "timestamp": _utc_now_iso(),
        }
        return record
    finally:
        import gc
        model = None
        tok = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except ImportError:
            pass
