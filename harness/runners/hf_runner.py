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
    # "multineedle" = original 3-needle NIH (existing behaviour, default for
    # backwards compatibility). "hard_multineedle" = RULER-style harder NIH:
    # more needles by default, code-shaped distractors in the context, scorer
    # checks key→code mapping. Use the hard variant when comparing models
    # that have already saturated the simple NIH at 1.00.
    task_type: str = "multineedle"
    n_distractors: int = 30                      # only used by hard_multineedle


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
        task_type=base.task_type,
        n_distractors=base.n_distractors,
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


# NOTE: the SW lever is wired at load time inside `backbone._load_plain`
# (see BackboneConfig.sliding_window). We can't mutate config post-load
# because Qwen3's attention modules cache their mask config at __init__
# and won't pick up a runtime flip. The δ-Mem path bypasses backbone's
# loader entirely, so cells 4 (SW-4K+δ-Mem) and 5 (SW-2K+δ-Mem) currently
# run with δ-Mem only — the SW side is logged but not effective.


def _build_eval_task(rc: "RunConfig"):
    """Dispatch on task_type. Defaults to existing 3-needle NIH for back-compat."""
    if rc.task_type == "hard_multineedle":
        # n_needles default jumps from 3 → 10 when caller didn't override.
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


def _score_eval(rc: "RunConfig", task, answer: str):
    """Score the answer for the chosen task type.

    Returns (quality_payload_dict, recall_signal_bool). The payload is
    embedded directly in result["quality"] (alongside perplexity etc.),
    so cells of different eval types remain comparable when read back.
    The recall_signal drives the top-level status flag.
    """
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
            },
        }
        # Any-recall = at least one needle correctly mapped.
        return payload, any(s.per_needle_correct)
    s = quality.score_multineedle(task, answer)
    payload = {
        "multineedle": {
            "per_needle": s.per_needle,
            "recall_all": s.recall_all,
            "recall_any": s.recall_any,
            "fraction": s.fraction,
        },
    }
    return payload, s.recall_any


def _generate_with_memory_split(model, tok, prompt: str, *,
                                max_new_tokens: int,
                                seed: int = 0,
                                assistant=None):
    """Run a single `generate()` but capture prefill and decode peaks separately.

    Returns (answer_text, prefill_peak_bytes, decode_peak_bytes).

    Why this approach: `torch.cuda.max_memory_allocated()` is process-lifetime
    peak, and FA2's prefill workspace at long context dominates. The KV-cache
    differences between vanilla / SW / δ-Mem that the v3 hypothesis cares
    about live in the DECODE phase. To isolate the decode peak without
    losing PyTorch's allocator-fusion of prefill+decode workspace (a manual
    prefill/decode split inflates prefill peak by ~1.7 GiB because the
    allocator can't reuse the workspace block), we attach a forward_pre_hook
    on the top-level model:

      - Hook fires before forward call #0 (= prefill)
      - Hook fires before forward call #1 (= first decode step):
        prefill is done. Snapshot peak. Reset stats. Continue.
      - After generate() returns, peak since reset = decode peak.

    `generate()` itself runs as one call so the allocator gets to optimise.
    For spec-decode (assistant != None) the assistant fires extra forwards
    interleaved; we still treat call #0 as prefill and everything after as
    decode for the target model. Hook is attached on the TARGET only.
    """
    import torch
    device = next(model.parameters()).device
    inputs = tok(prompt, return_tensors="pt").to(device)
    if seed is not None:
        torch.manual_seed(seed)

    call_idx = [0]
    prefill_peak_box: Dict[str, int] = {"v": 0}
    decode_resident_box: Dict[str, int] = {"v": 0}

    def _pre_hook(module, args, kwargs):
        if call_idx[0] == 1:
            # First decode step is about to begin. Snapshot prefill peak,
            # then reset so subsequent measurements isolate the decode phase.
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            prefill_peak_box["v"] = memory.measure_peak_vram()
            memory.reset_peak_vram()
        elif call_idx[0] >= 2:
            # After at least one decode forward has completed. Capture
            # steady-state resident — this is where SW / sink caches
            # actually show their cropping. Keep the running max across
            # decode steps so the metric reflects the largest steady-state
            # cache the generation saw (matters when generation extends
            # beyond the window — SW stays at window, vanilla grows).
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            r = memory.measure_resident_vram()
            if r > decode_resident_box["v"]:
                decode_resident_box["v"] = r
        call_idx[0] += 1

    handle = model.register_forward_pre_hook(_pre_hook, with_kwargs=True)
    try:
        memory.reset_peak_vram()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=False, use_cache=True,
        )
        if assistant is not None:
            gen_kwargs["assistant_model"] = assistant
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    finally:
        handle.remove()

    decode_peak = memory.measure_peak_vram()
    # If max_new_tokens == 1 the hook never fired for "decode": prefill peak
    # wasn't captured. Treat the whole run as prefill.
    prefill_peak = prefill_peak_box["v"] or decode_peak
    # Decode-phase steady-state resident: max alloc seen at the start of
    # any decode-step forward (call_idx >= 2). This is AFTER the SW layer's
    # update() has cropped the cache, so it exposes real cache savings —
    # unlike either (a) post-generate resident (cache already freed) or
    # (b) prefill-boundary resident (cache not yet cropped).
    decode_resident = decode_resident_box["v"] or 0
    answer = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return answer, prefill_peak, decode_peak, decode_resident


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
        sliding_window=_window_size_for_lever(cell.kv_lever),
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
        asst = load_assistant(rc.assistant_model_id, device=rc.device, dtype=rc.dtype) \
            if rc.assistant_model_id and session is None else None
        # Note: if session is set, we don't currently support spec-decode + δ-Mem in
        # one shot — the upstream runtime doesn't expose that combo. Cell 7 will run
        # δ-Mem but without spec-decode.
        if session is not None and rc.assistant_model_id:
            print(f"  ⚠️ cell {cell.id}: δ-Mem + spec-decode is not yet supported by the upstream runtime; running δ-Mem only")

        task = _build_eval_task(rc)
        nih_prompt = task.context + "\n\n" + task.question
        prefill_peak: Optional[int] = None
        decode_peak: Optional[int] = None
        decode_resident: Optional[int] = None
        if session is not None:
            # δ-Mem path: the upstream session runs write+decode opaquely, so
            # we can only measure overall peak. Reset before the call so we
            # exclude the model-load peak; the result lands as decode_peak.
            memory.reset_peak_vram()
            reply = session.generate_reply(
                user_text=nih_prompt,
                max_new_tokens=rc.max_new_tokens,
            )
            answer = reply["assistant"]
            decode_peak = memory.measure_peak_vram()
        else:
            # Plain + spec-decode path: prefill and decode run as separate
            # calls so memory peaks split cleanly. See helper docstring.
            answer, prefill_peak, decode_peak, decode_resident = \
                _generate_with_memory_split(
                    model, tok, nih_prompt,
                    max_new_tokens=rc.max_new_tokens,
                    seed=rc.seed,
                    assistant=asst,
                )

        quality_payload, recall_signal = _score_eval(rc, task, answer)

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
            "status": "ok" if recall_signal else "partial",
            "quality": {
                **quality_payload,
                "perplexity": ppl,
            },
            "memory": {
                # Legacy field, kept so old result JSONs stay comparable.
                # Now derived from the explicit phase peaks below — the
                # post-helper torch.cuda.max_memory_allocated() only sees
                # the most recently reset window, not the historical peak.
                "peak_vram_bytes": max(prefill_peak or 0, decode_peak or 0),
                "kv_cache_bytes_at_target_len": int(kv_bytes_at_ctx),
                "prefill_peak_vram_bytes": prefill_peak,
                "decode_peak_vram_bytes": decode_peak,
                # Steady-state alloc captured AFTER the first decode forward
                # has completed (call_idx >= 2 in the helper's hook). By that
                # point the SW layer's update() has cropped the cache, so
                # the metric reflects what's actually resident during decode
                # — exposes real SW / sink / cache savings.
                "decode_resident_vram_bytes": decode_resident,
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
