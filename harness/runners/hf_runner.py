"""Run one HF-stack cell end-to-end and return a results record."""

from __future__ import annotations

import datetime
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from ..backbone import BackboneConfig, load_backbone
from ..cells import Cell
from ..metrics import memory, quality, speed
from ..spec_decode import generate_with_spec_decode, load_assistant


@dataclass
class RunConfig:
    target_tokens: int
    n_needles: int = 3
    max_new_tokens: int = 256
    seed: int = 0
    dtype: str = "bfloat16"
    device: str = "cuda"
    results_dir: Path = Path("results")
    ppl_text: Optional[str] = None
    assistant_model_id: Optional[str] = None     # set for spec-decode cells
    delta_mem_adapter_id: Optional[str] = None   # set for δ-Mem cells


def _resolve_cell_config(cell: Cell, base: RunConfig) -> RunConfig:
    """Apply per-cell levers on top of the base config."""
    adapter = base.delta_mem_adapter_id
    assistant = base.assistant_model_id

    if "side-state" in cell.kv_lever:
        adapter = adapter or _default_adapter_for(cell.base_model)
    if cell.speed_lever == "spec-decode":
        assistant = assistant or _default_assistant_for(cell.base_model)
    return RunConfig(
        target_tokens=base.target_tokens, n_needles=base.n_needles,
        max_new_tokens=base.max_new_tokens, seed=base.seed,
        dtype=base.dtype, device=base.device, results_dir=base.results_dir,
        ppl_text=base.ppl_text,
        assistant_model_id=assistant,
        delta_mem_adapter_id=adapter,
    )


def _default_adapter_for(base_model: str) -> Optional[str]:
    if "Qwen3-4B-Instruct" in base_model:
        return "declare-lab/delta-mem_qwen3_4b-instruct"
    return None


def _default_assistant_for(base_model: str) -> Optional[str]:
    if "Qwen3-4B-Instruct" in base_model:
        return "Qwen/Qwen3-0.6B"
    if "SmolLM3" in base_model:
        return "HuggingFaceTB/SmolLM3-135M"
    return None


def _window_size_for_lever(kv_lever: str) -> Optional[int]:
    """Return the sliding-window length for this lever, or None if no window.

    Conventions taken from the v3 test matrix:
      - "window" / "window+side-state"   → 4K context window
      - "aggressive-window+side-state"   → 2K context window
      - "sink+window+side-state"         → 4K + StreamingLLM sink tokens
    """
    if "aggressive-window" in kv_lever:
        return 2048
    if "window" in kv_lever:
        return 4096
    return None


def _apply_kv_lever(model, kv_lever: str) -> None:
    """Mutate the loaded model's config so the kv_lever is actually in effect.

    Qwen3 exposes `use_sliding_window` and `sliding_window` config flags; when
    enabled, transformers' Qwen3 attention applies a sliding-window causal
    mask and the cache it picks for generation tracks only that window.

    This is the place where cells 3/4/5/8/10/12 stop being scaffolds and
    actually exercise the SW lever. The δ-Mem side of "window+side-state" is
    already handled separately (see _resolve_cell_config).
    """
    if model is None:
        return
    W = _window_size_for_lever(kv_lever)
    if W is None:
        return
    cfg = getattr(model, "config", None)
    if cfg is None:
        return
    if not (hasattr(cfg, "sliding_window") and hasattr(cfg, "use_sliding_window")):
        print(f"  ⚠️ model config does not expose sliding_window; SW lever ignored")
        return
    cfg.use_sliding_window = True
    cfg.sliding_window = W
    # Qwen3 has a `max_window_layers` knob that gates which layers use SW;
    # set it to all layers so every layer participates.
    if hasattr(cfg, "max_window_layers") and hasattr(cfg, "num_hidden_layers"):
        cfg.max_window_layers = cfg.num_hidden_layers
    print(f"  SW lever active: sliding_window={W}, max_window_layers="
          f"{getattr(cfg, 'max_window_layers', '?')}")


def _utc_now_iso() -> str:
    """Timezone-aware UTC ISO 8601 with 'Z' suffix."""
    return datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z")


def run(cell: Cell, base_cfg: RunConfig) -> Dict[str, Any]:
    # Quiet transformers' verbose generation_config warnings during the run.
    # Hundreds of these would otherwise spam the dispatch loop for spec-decode.
    import logging
    logging.getLogger("transformers.generation").setLevel(logging.ERROR)
    logging.getLogger("transformers.generation.configuration_utils").setLevel(logging.ERROR)

    rc = _resolve_cell_config(cell, base_cfg)

    bcfg = BackboneConfig(
        model_id=cell.base_model,
        dtype=rc.dtype, device=rc.device,
        delta_mem_adapter_id=rc.delta_mem_adapter_id,
    )
    memory.reset_peak_vram()
    # Pre-load diagnostic: confirm GPUs are actually empty before we start
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(i)
                print(f"  GPU {i} pre-load: {(total-free)/1024**3:.2f} GiB used / "
                      f"{total/1024**3:.2f} GiB total")
    except Exception:
        pass
    model = None
    asst = None
    session = None
    try:
        model, tok, session = load_backbone(bcfg)
        _apply_kv_lever(model, cell.kv_lever)
        asst = load_assistant(rc.assistant_model_id, device=rc.device, dtype=rc.dtype) \
            if rc.assistant_model_id and session is None else None
        # Note: if session is set, we don't currently support spec-decode + δ-Mem in
        # one shot — the upstream runtime doesn't expose that combo. Cell 7 will run
        # δ-Mem but without spec-decode.
        if session is not None and rc.assistant_model_id:
            print(f"  ⚠️ cell {cell.id}: δ-Mem + spec-decode is not yet supported by the upstream runtime; running δ-Mem only")

        task = quality.make_multineedle_task(
            target_tokens=rc.target_tokens,
            n_needles=rc.n_needles,
            seed=rc.seed,
        )

        if session is not None:
            # δ-Mem path: feed the entire NIH prompt as a single user message.
            # The session's generate_reply tokenizes via chat template, runs the
            # write phase (context flows into the online memory state), then
            # decodes the answer with write disabled.
            reply = session.generate_reply(
                user_text=task.context + "\n\n" + task.question,
                max_new_tokens=rc.max_new_tokens,
            )
            answer = reply["assistant"]
        elif asst is not None:
            answer = generate_with_spec_decode(
                model, tok, asst,
                prompt=task.context + "\n\n" + task.question,
                max_new_tokens=rc.max_new_tokens,
            )
        else:
            import torch
            prompt = task.context + "\n\n" + task.question
            inputs = tok(prompt, return_tensors="pt").to(next(model.parameters()).device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=rc.max_new_tokens, do_sample=False)
            answer = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        nih_score = quality.score_multineedle(task, answer)

        # Speed timing uses plain generate even for δ-Mem cells — measuring the
        # session's generate_reply path would require a separate timing harness
        # (the session does write-phase ingest + decode separately, both of which
        # affect tok/s). Phase 2 follow-up.
        speed_record = speed.timed_generation(
            model, tok, prompt="The capital of France is",
            max_new_tokens=min(64, rc.max_new_tokens),
            seed=rc.seed,
            assistant_model=asst,
        )

        cfg_obj = getattr(model, "config", None)
        kv_bytes_at_ctx = memory.kv_bytes_from_config(cfg_obj, seq_len=rc.target_tokens) \
            if cfg_obj is not None else 0

        ppl = None
        if rc.ppl_text:
            ppl = quality.compute_perplexity(model, tok, rc.ppl_text)

        record = {
            "cell_id": cell.id,
            "title": cell.title,
            "status": "ok" if nih_score.recall_any else "partial",
            "quality": {
                "multineedle": {
                    "per_needle": nih_score.per_needle,
                    "recall_all": nih_score.recall_all,
                    "recall_any": nih_score.recall_any,
                    "fraction": nih_score.fraction,
                },
                "perplexity": ppl,
            },
            "memory": {
                "peak_vram_bytes": memory.measure_peak_vram(),
                "kv_cache_bytes_at_target_len": int(kv_bytes_at_ctx),
            },
            "speed": speed_record.as_dict(),
            "config": {
                "cell": asdict(cell) if hasattr(cell, "__dataclass_fields__") else cell.__dict__,
                "run": {
                    "target_tokens": rc.target_tokens,
                    "n_needles": rc.n_needles,
                    "max_new_tokens": rc.max_new_tokens,
                    "dtype": rc.dtype, "device": rc.device,
                    "delta_mem_adapter_id": rc.delta_mem_adapter_id,
                    "assistant_model_id": rc.assistant_model_id,
                },
                "backbone": bcfg.as_dict(),
            },
            "answer_preview": answer[:512],
            "timestamp": _utc_now_iso(),
        }

        return record
    finally:
        import gc
        # Aggressive cleanup so the next cell starts with clean GPUs:
        # 1) accelerate hooks (high-level API)
        # 2) manual _hf_hook / _old_forward stripping (defensive fallback)
        # 3) zero out any GPU tensors still hanging in the model
        # 4) drop references
        # 5) release_memory + sync + empty_cache + ipc_collect
        def _strip_hooks(obj):
            if obj is None:
                return
            try:
                from accelerate.hooks import remove_hook_from_module
                remove_hook_from_module(obj, recurse=True)
            except (ImportError, Exception):
                pass
            # Defensive: walk modules and remove residual hook artefacts
            try:
                for m in obj.modules():
                    for attr in ("_hf_hook", "_old_forward"):
                        if hasattr(m, attr):
                            try:
                                delattr(m, attr)
                            except Exception:
                                pass
            except Exception:
                pass

        _strip_hooks(model)
        _strip_hooks(asst)

        # Drop references (assignment to None breaks circular refs in
        # accelerate's dispatcher state)
        model = None
        tok = None
        asst = None
        session = None

        try:
            from accelerate.utils import release_memory
            release_memory()
        except ImportError:
            pass
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    torch.cuda.synchronize(i)
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except ImportError:
            pass
