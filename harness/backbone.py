"""Backbone model loading, with optional δ-Mem adapter attach.

The upstream `declare-lab/delta-Mem` package isn't pip-installable: it's a
reference repo that you clone and either install its `requirements.txt` or
put on `PYTHONPATH`. `scripts/kaggle_bootstrap.sh` does that automatically
under `.deps/delta-Mem`; this module finds the clone at import time.

The public API (per upstream README) is:
    from deltamem.core import HFDeltaMemConfig, attach_delta_mem, load_delta_mem_adapter
    config = HFDeltaMemConfig.from_pretrained(adapter_dir)
    attach_delta_mem(model, config)
    load_delta_mem_adapter(model, adapter_dir)

Note that `adapter_dir` is a LOCAL directory, not an HF Hub id; we
`snapshot_download` the adapter on demand.
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class BackboneConfig:
    model_id: str
    dtype: str = "bfloat16"
    device: str = "cuda"
    delta_mem_adapter_id: Optional[str] = None
    trust_remote_code: bool = True
    attn_implementation: Optional[str] = "flash_attention_2"  # falls back via try/except if unavailable

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


_DTYPE_MAP = {"bfloat16": "bfloat16", "float16": "float16", "float32": "float32"}


def _torch_dtype(name: str):
    import torch
    return getattr(torch, _DTYPE_MAP[name])


def _candidate_deltamem_roots():
    """Common places kaggle_bootstrap.sh and local dev put the clone."""
    return [
        Path.cwd() / ".deps" / "delta-Mem",
        Path(__file__).resolve().parents[1] / ".deps" / "delta-Mem",
        Path("/kaggle/working/delta-mem-smollm3-3b/.deps/delta-Mem"),
    ]


def _ensure_deltamem_importable():
    """Import deltamem.core; on ImportError, look for a clone and retry.

    Returns (HFDeltaMemConfig, attach_delta_mem, load_delta_mem_adapter).
    """
    try:
        from deltamem.core import (  # type: ignore
            HFDeltaMemConfig, attach_delta_mem, load_delta_mem_adapter,
        )
        return HFDeltaMemConfig, attach_delta_mem, load_delta_mem_adapter
    except ImportError:
        pass
    for candidate in _candidate_deltamem_roots():
        if (candidate / "deltamem").is_dir():
            sys.path.insert(0, str(candidate))
            try:
                from deltamem.core import (  # type: ignore
                    HFDeltaMemConfig, attach_delta_mem, load_delta_mem_adapter,
                )
                return HFDeltaMemConfig, attach_delta_mem, load_delta_mem_adapter
            except ImportError:
                sys.path.pop(0)
                continue
    raise RuntimeError(
        "deltamem package not importable. The upstream repo is not on PyPI — "
        "either run scripts/kaggle_bootstrap.sh (which clones it to .deps/delta-Mem) "
        "or set PYTHONPATH to include a local clone."
    )


def _resolve_device_args(device: str) -> Dict[str, Any]:
    """Translate a high-level device strategy into kwargs for from_pretrained.

    - "cpu":        device_map=None (load to CPU)
    - "cuda":       device_map="cuda" (pin to cuda:0 — single-GPU machines)
    - "auto"/"balanced": device_map="balanced" + per-GPU max_memory cap
      that leaves headroom for activations and KV cache. Explicit cap is
      necessary because accelerate's "auto" considers only the model size,
      not the inference workspace it'll need afterwards.
    """
    if device == "cpu":
        return {"device_map": None}
    if device == "cuda":
        return {"device_map": "cuda"}
    if device in ("auto", "balanced"):
        import torch
        if not torch.cuda.is_available():
            return {"device_map": None}
        n = torch.cuda.device_count()
        if n <= 1:
            return {"device_map": "cuda"}
        # Allow the user to tune via env var; the 50% default forces a real
        # split: Qwen3-4B is ~8GB bf16 and won't fit in 7GiB on a single T4.
        # If you raise this and the model fits on one GPU, accelerate will
        # happily put it all there (per its "balanced" semantics).
        pct = float(os.environ.get("GPU_MAX_PCT", "0.50"))
        per_gpu_gib = int(torch.cuda.get_device_properties(0).total_memory * pct / (1024**3))
        max_memory = {i: f"{per_gpu_gib}GiB" for i in range(n)}
        return {"device_map": "balanced", "max_memory": max_memory}
    # Unknown — pass through verbatim (e.g. a custom dict-form device_map)
    return {"device_map": device}


def _resolve_adapter_dir(adapter_id_or_path: str) -> str:
    """Return a local directory containing the adapter.

    If `adapter_id_or_path` already points to a local directory, return it
    unchanged. Otherwise treat it as a HF Hub repo id and snapshot_download.
    """
    p = Path(adapter_id_or_path)
    if p.exists() and p.is_dir():
        return str(p)
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=adapter_id_or_path)


def _attn_impl_for_hardware(requested: Optional[str]) -> Optional[str]:
    """Filter requested attention impl by what the hardware actually supports.

    FlashAttention-2 requires sm_80+ (Ampere or newer). On Turing GPUs
    (T4 = sm_75) the kernel raises RuntimeError at first forward pass —
    not at load — so silent fallback via from_pretrained's try/except
    isn't enough. This gate prevents us from ever requesting FA2 when the
    hardware can't run it.
    """
    if requested != "flash_attention_2":
        return requested
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        for i in range(torch.cuda.device_count()):
            cap = torch.cuda.get_device_capability(i)
            if cap[0] < 8:  # sm_80 = Ampere
                print(f"  GPU {i} compute capability is sm_{cap[0]}{cap[1]} — "
                      f"FlashAttention-2 needs sm_80+; falling back to SDPA")
                return None
        return "flash_attention_2"
    except Exception as e:
        print(f"  capability check failed ({e}); falling back to SDPA")
        return None


def _print_load_diagnostics(model) -> None:
    try:
        import torch
        hf_map = getattr(model, "hf_device_map", None)
        if hf_map is None:
            print("  hf_device_map: <none — model is on a single device or not dispatched>")
        else:
            # Summarise by device: how many modules each got
            counts: Dict[str, int] = {}
            for dev in hf_map.values():
                key = str(dev)
                counts[key] = counts.get(key, 0) + 1
            summary = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
            print(f"  hf_device_map summary: {{{summary}}}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(i)
                used = total - free
                print(f"  GPU {i} after load: {used/1024**3:.2f} GiB used / "
                      f"{total/1024**3:.2f} GiB total")
    except Exception as e:
        print(f"  (diagnostics failed: {e})")


def load_backbone(cfg: BackboneConfig) -> Tuple[Any, Any]:
    """Return (model, tokenizer). Attaches δ-Mem if `cfg.delta_mem_adapter_id` is set."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=cfg.trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    device_kwargs = _resolve_device_args(cfg.device)
    common = dict(
        dtype=_torch_dtype(cfg.dtype),
        trust_remote_code=cfg.trust_remote_code,
        **device_kwargs,
    )
    effective_impl = _attn_impl_for_hardware(cfg.attn_implementation)
    model = None
    if effective_impl:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg.model_id,
                attn_implementation=effective_impl,
                **common,
            )
        except (ImportError, ValueError) as e:
            print(f"  attn_implementation={effective_impl} unavailable ({e}); "
                  f"falling back to default (SDPA)")
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **common)
    model.eval()

    # Diagnostics: did accelerate actually split the model, and what's the
    # GPU memory footprint right after load?
    _print_load_diagnostics(model)

    if cfg.delta_mem_adapter_id:
        HFDeltaMemConfig, attach_delta_mem, load_delta_mem_adapter = _ensure_deltamem_importable()
        adapter_dir = _resolve_adapter_dir(cfg.delta_mem_adapter_id)
        dm_config = HFDeltaMemConfig.from_pretrained(adapter_dir)
        attach_delta_mem(model, dm_config)
        load_delta_mem_adapter(model, adapter_dir)
    return model, tok
