"""Memory accounting: peak VRAM during decode, analytic KV-cache size."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


def kv_bytes(
    *,
    num_hidden_layers: int,
    num_key_value_heads: int,
    head_dim: int,
    seq_len: int,
    dtype_bytes: int = 2,
    sliding_window: Optional[int] = None,
) -> int:
    """Bytes the KV cache occupies for one stream at this seq_len."""
    effective_len = min(seq_len, sliding_window) if sliding_window is not None else seq_len
    return num_hidden_layers * num_key_value_heads * head_dim * effective_len * dtype_bytes * 2


def kv_bytes_from_config(cfg, *, seq_len: int, dtype_bytes: int = 2,
                         sliding_window: Optional[int] = None) -> int:
    n_kv = getattr(cfg, "num_key_value_heads", None)
    if n_kv is None:
        n_kv = cfg.num_attention_heads
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    return kv_bytes(
        num_hidden_layers=cfg.num_hidden_layers,
        num_key_value_heads=n_kv,
        head_dim=head_dim,
        seq_len=seq_len,
        dtype_bytes=dtype_bytes,
        sliding_window=sliding_window,
    )


@dataclass(frozen=True)
class PeakMemorySample:
    peak_vram_bytes: int
    kv_cache_bytes: int
    device: str


def measure_peak_vram() -> int:
    """Snapshot torch.cuda peak; safe to call when no CUDA — returns 0."""
    try:
        import torch
        if not torch.cuda.is_available():
            return 0
        return int(torch.cuda.max_memory_allocated())
    except ImportError:
        return 0


def measure_resident_vram() -> int:
    """Snapshot current allocation (steady-state), not peak.

    Where `measure_peak_vram` captures the worst-case allocator high-water
    mark — which includes transient spikes during operations like KV cache
    slicing — `measure_resident_vram` reports what's currently in use.
    Calling this at end of decode shows the post-generation resident set:
    model weights + final KV cache + any other tensors still alive.

    This is the metric to compare for cache-reduction tests like SW or
    sink cache — the transient peak that contains both pre-slice and
    post-slice tensors will mask the steady-state reduction.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return 0
        return int(torch.cuda.memory_allocated())
    except ImportError:
        return 0


def reset_peak_vram() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass
