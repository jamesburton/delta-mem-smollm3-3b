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


def load_backbone(cfg: BackboneConfig) -> Tuple[Any, Any]:
    """Return (model, tokenizer). Attaches δ-Mem if `cfg.delta_mem_adapter_id` is set."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=cfg.trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        dtype=_torch_dtype(cfg.dtype),
        trust_remote_code=cfg.trust_remote_code,
        device_map=cfg.device if cfg.device != "cpu" else None,
    )
    model.eval()

    if cfg.delta_mem_adapter_id:
        HFDeltaMemConfig, attach_delta_mem, load_delta_mem_adapter = _ensure_deltamem_importable()
        adapter_dir = _resolve_adapter_dir(cfg.delta_mem_adapter_id)
        dm_config = HFDeltaMemConfig.from_pretrained(adapter_dir)
        attach_delta_mem(model, dm_config)
        load_delta_mem_adapter(model, adapter_dir)
    return model, tok
