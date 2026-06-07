"""Backbone model loading, with optional δ-Mem adapter attach.

The upstream `declare-lab/delta-Mem` package isn't pip-installable: it's a
reference repo that you clone and either install its `requirements.txt` or
put on `PYTHONPATH`. `scripts/kaggle_bootstrap.sh` does that automatically
under `.deps/delta-Mem`; this module finds the clone at import time.

With a δ-Mem adapter, loading is delegated entirely to the upstream runtime:

    from deltamem.runtime.session import DeltaMemChatSession, load_delta_mem_chat_model
    model, tokenizer = load_delta_mem_chat_model(
        model_path=local_base_model_path,
        device="cuda:0",
        dtype="bfloat16",
        attn_implementation=None,
        adapter_dir=local_adapter_dir,
    )
    session = DeltaMemChatSession(model=model, tokenizer=tokenizer, device="cuda:0")
    result = session.generate_reply(user_text="...", max_new_tokens=256)
    # result["assistant"] is the reply

Note that `adapter_dir` is a LOCAL directory, not an HF Hub id; we
`snapshot_download` the adapter on demand. The upstream runtime also requires
`local_files_only=True` for the base model, so we pre-download it via
`snapshot_download` before calling `load_delta_mem_chat_model`.
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
    # Sliding-window kv_lever. When set, the loader rewrites the model
    # config's `layer_types` to ["sliding_attention"] * num_hidden_layers
    # BEFORE `from_pretrained` so the attention modules initialise with
    # SW awareness and the generation cache picks DynamicSlidingWindowLayer
    # automatically. Post-load mutation does not work — Qwen3's attention
    # mask code raises KeyError mid-decode.
    sliding_window: Optional[int] = None

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
    """Ensure the deltamem package is on sys.path; on ImportError, look for a clone and retry.

    After this call, `from deltamem.runtime.session import ...` and
    `from deltamem.core import ...` are both importable.
    """
    try:
        import deltamem  # type: ignore  # noqa: F401
        return
    except ImportError:
        pass
    for candidate in _candidate_deltamem_roots():
        if (candidate / "deltamem").is_dir():
            sys.path.insert(0, str(candidate))
            try:
                import deltamem  # type: ignore  # noqa: F401
                return
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

    - "cpu": load to CPU
    - "cuda": pin to cuda:0 only
    - "cuda:N": pin to specified device
    - "auto"/"balanced": multi-GPU balanced split, with CPU offload as
      last-resort fallback when GPU memory is tight. Tunable via:
        GPU_MAX_PCT - fraction of each GPU we'll fill (default 0.50)
        CPU_MAX_GIB - host RAM budget for spillover (default 16)

    NOTE: the δ-Mem path doesn't go through here — upstream's
    DeltaMemChatSession pins to a single device.
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
        pct = float(os.environ.get("GPU_MAX_PCT", "0.50"))
        cpu_offload_gib = int(os.environ.get("CPU_MAX_GIB", "16"))
        if n == 1:
            # Single GPU: still allow CPU spillover so 3060 12GB can hold
            # the 4B model even at tight margins. Cap GPU at GPU_MAX_PCT.
            per_gpu_gib = int(torch.cuda.get_device_properties(0).total_memory * pct / (1024**3))
            max_memory = {0: f"{per_gpu_gib}GiB", "cpu": f"{cpu_offload_gib}GiB"}
            return {"device_map": "auto", "max_memory": max_memory}
        # Multi-GPU: split across GPUs, with CPU as last-resort offload.
        per_gpu_gib = int(torch.cuda.get_device_properties(0).total_memory * pct / (1024**3))
        max_memory = {i: f"{per_gpu_gib}GiB" for i in range(n)}
        max_memory["cpu"] = f"{cpu_offload_gib}GiB"
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

    FlashAttention-2 requires sm_80+ (Ampere or newer). On Turing (T4 = sm_75)
    FA2 raises RuntimeError at first forward — not at load — so silent
    fallback via from_pretrained's try/except isn't enough.

    When FA2 isn't viable, we return "sdpa" explicitly rather than None so
    transformers definitely engages PyTorch's SDP path (and we can then
    configure PyTorch's SDP backend preferences via configure_sdp_backends).
    """
    if requested != "flash_attention_2":
        return requested
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        for i in range(torch.cuda.device_count()):
            cap = torch.cuda.get_device_capability(i)
            if cap[0] < 8:
                print(f"  GPU {i} compute capability is sm_{cap[0]}{cap[1]} — "
                      f"FlashAttention-2 needs sm_80+; using SDPA mem-efficient")
                return "sdpa"
        # GPU supports FA2 — but the package also has to be importable
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            print(f"  flash_attn package not installed; using SDPA mem-efficient")
            return "sdpa"
        return "flash_attention_2"
    except Exception as e:
        print(f"  capability check failed ({e}); falling back to SDPA")
        return "sdpa"


def configure_sdp_backends() -> None:
    """Force PyTorch's SDP selector to prefer memory-efficient over math.

    Critical for T4 (sm_75): math kernel materializes the full N×N attention
    matrix and OOMs at 4K+ context. Mem-efficient computes in O(N) memory
    tiles. We DISABLE math by default so the selector can't pick it; if
    flash/mem_efficient/cudnn can't handle the shape, you get a loud error
    instead of a silent OOM at scale.

    Override via env var (set ENABLE_MATH_SDP=1) if you specifically need
    the math fallback enabled for shape-compatibility diagnosis.
    """
    try:
        import torch
        # Preference order on Turing: cudnn (if present) > mem_efficient
        # On Ampere+: flash > mem_efficient
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        # Math: disabled by default. Math kernel = O(N^2) memory = OOM trap.
        allow_math = os.environ.get("ENABLE_MATH_SDP", "0") == "1"
        torch.backends.cuda.enable_math_sdp(allow_math)
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(True)
        print(f"  SDP backends: flash={torch.backends.cuda.flash_sdp_enabled()}, "
              f"mem_efficient={torch.backends.cuda.mem_efficient_sdp_enabled()}, "
              f"math={torch.backends.cuda.math_sdp_enabled()}"
              f"{' (math forced enabled via ENABLE_MATH_SDP)' if allow_math else ''}")
    except Exception as e:
        print(f"  configure_sdp_backends failed ({e}); using PyTorch defaults")


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


def load_backbone(cfg: BackboneConfig) -> Tuple[Any, Any, Optional[Any]]:
    """Return (model, tokenizer, delta_mem_session_or_None).

    With delta_mem_adapter_id set, this delegates to the upstream δ-Mem
    runtime which constructs a stateful DeltaMemChatSession. Without an
    adapter, this returns a plain HF model + tokenizer (session=None).
    """
    configure_sdp_backends()  # set up before any from_pretrained call
    if cfg.delta_mem_adapter_id:
        return _load_with_delta_mem(cfg)
    return _load_plain(cfg)


def _load_plain(cfg: BackboneConfig) -> Tuple[Any, Any, None]:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=cfg.trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    device_kwargs = _resolve_device_args(cfg.device)
    common = dict(
        dtype=_torch_dtype(cfg.dtype),
        trust_remote_code=cfg.trust_remote_code,
        **device_kwargs,
    )
    # If a sliding-window lever is requested, edit the config BEFORE load so
    # the attention modules pick up SW awareness at init time. See the
    # BackboneConfig.sliding_window docstring for why this is load-time, not
    # post-load.
    if cfg.sliding_window:
        model_config = AutoConfig.from_pretrained(cfg.model_id, trust_remote_code=cfg.trust_remote_code)
        if not hasattr(model_config, "sliding_window"):
            print(f"  ⚠️ {cfg.model_id} config has no sliding_window attribute; SW lever ignored")
        else:
            model_config.use_sliding_window = True
            model_config.sliding_window = cfg.sliding_window
            if hasattr(model_config, "max_window_layers") and hasattr(model_config, "num_hidden_layers"):
                model_config.max_window_layers = model_config.num_hidden_layers
            if hasattr(model_config, "layer_types") and hasattr(model_config, "num_hidden_layers"):
                model_config.layer_types = ["sliding_attention"] * model_config.num_hidden_layers
            common["config"] = model_config
            print(f"  SW lever baked into config: sliding_window={cfg.sliding_window}, "
                  f"layer_types[0]=sliding_attention")

    effective_impl = _attn_impl_for_hardware(cfg.attn_implementation)
    model = None
    if effective_impl:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg.model_id, attn_implementation=effective_impl, **common,
            )
        except (ImportError, ValueError) as e:
            print(f"  attn_implementation={effective_impl} unavailable ({e}); falling back to SDPA")
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **common)
    model.eval()
    _print_load_diagnostics(model)
    return model, tok, None


def _load_with_delta_mem(cfg: BackboneConfig) -> Tuple[Any, Any, Any]:
    """Use upstream's DeltaMemChatSession (stateful runtime)."""
    _ensure_deltamem_importable()
    from deltamem.runtime.session import DeltaMemChatSession, load_delta_mem_chat_model  # type: ignore

    adapter_dir = _resolve_adapter_dir(cfg.delta_mem_adapter_id)
    effective_impl = _attn_impl_for_hardware(cfg.attn_implementation)
    # Upstream uses single-device device_map; cap at one GPU.
    if cfg.device in ("auto", "balanced"):
        # Prefer cuda:0 for δ-Mem; the runtime keeps state on one device.
        device = "cuda:0" if _has_cuda() else "cpu"
        print(f"  δ-Mem requires single-device placement; using {device}")
    else:
        device = cfg.device

    # Workaround upstream's `local_files_only=True`: pre-download the base
    # model via snapshot_download so the call succeeds offline.
    from huggingface_hub import snapshot_download
    print(f"  resolving base model {cfg.model_id} to local cache...")
    model_local = snapshot_download(repo_id=cfg.model_id)

    model, tok = load_delta_mem_chat_model(
        model_path=model_local,
        device=device,
        dtype=cfg.dtype,
        attn_implementation=effective_impl,
        adapter_dir=adapter_dir,
    )
    session = DeltaMemChatSession(model=model, tokenizer=tok, device=device)
    _print_load_diagnostics(model)
    return model, tok, session


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
