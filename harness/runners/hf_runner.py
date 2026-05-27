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


def _utc_now_iso() -> str:
    """Timezone-aware UTC ISO 8601 with 'Z' suffix."""
    return datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z")


def run(cell: Cell, base_cfg: RunConfig) -> Dict[str, Any]:
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
    try:
        model, tok = load_backbone(bcfg)
        asst = load_assistant(rc.assistant_model_id, device=rc.device, dtype=rc.dtype) \
            if rc.assistant_model_id else None
        task = quality.make_multineedle_task(
            target_tokens=rc.target_tokens,
            n_needles=rc.n_needles,
            seed=rc.seed,
        )
        prompt = task.context + "\n\n" + task.question

        if asst is not None:
            answer = generate_with_spec_decode(model, tok, asst, prompt=prompt,
                                                max_new_tokens=rc.max_new_tokens)
        else:
            import torch
            inputs = tok(prompt, return_tensors="pt").to(next(model.parameters()).device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=rc.max_new_tokens, do_sample=False)
            answer = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        nih_score = quality.score_multineedle(task, answer)

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
        # Free GPU memory before returning so the notebook loop can move on cleanly
        if model is not None:
            del model
        if asst is not None:
            del asst
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
