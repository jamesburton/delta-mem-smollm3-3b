"""Backbone model loading, with optional δ-Mem adapter attach.

The upstream `declare-lab/delta-Mem` package exposes:
    from delta_mem import attach_delta_mem, load_delta_mem_adapter

If the package isn't installed, we fail loudly when an adapter is requested
but allow vanilla loading to proceed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class BackboneConfig:
    model_id: str
    dtype: str = "bfloat16"      # "bfloat16" | "float16" | "float32"
    device: str = "cuda"          # "cuda" | "cpu"
    delta_mem_adapter_id: Optional[str] = None
    trust_remote_code: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


_DTYPE_MAP = {"bfloat16": "bfloat16", "float16": "float16", "float32": "float32"}


def _torch_dtype(name: str):
    import torch
    return getattr(torch, _DTYPE_MAP[name])


def load_backbone(cfg: BackboneConfig) -> Tuple[Any, Any]:
    """Return (model, tokenizer). Attaches δ-Mem if `cfg.delta_mem_adapter_id` is set."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=cfg.trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=_torch_dtype(cfg.dtype),
        trust_remote_code=cfg.trust_remote_code,
        device_map=cfg.device if cfg.device != "cpu" else None,
    )
    model.eval()

    if cfg.delta_mem_adapter_id:
        try:
            from delta_mem import attach_delta_mem, load_delta_mem_adapter  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "delta-Mem adapter requested but `delta_mem` package is not installed.\n"
                "Install per upstream: pip install git+https://github.com/declare-lab/delta-Mem"
            ) from e
        model = attach_delta_mem(model)
        load_delta_mem_adapter(model, cfg.delta_mem_adapter_id)
    return model, tok
