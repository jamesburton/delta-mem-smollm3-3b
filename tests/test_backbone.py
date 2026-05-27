import pytest
import torch

from harness import backbone


def test_load_tiny_backbone_returns_model_and_tokenizer(tiny_model_id):
    cfg = backbone.BackboneConfig(model_id=tiny_model_id, dtype="float32", device="cpu",
                                   delta_mem_adapter_id=None)
    model, tok, session = backbone.load_backbone(cfg)
    assert model is not None
    assert tok is not None
    assert session is None  # no adapter requested
    # Basic forward sanity
    out = model(**tok("hello world", return_tensors="pt"))
    assert out.logits.ndim == 3


def test_backbone_config_round_trips_to_dict(tiny_model_id):
    cfg = backbone.BackboneConfig(model_id=tiny_model_id, dtype="bfloat16", device="cuda",
                                   delta_mem_adapter_id="declare-lab/delta-mem_qwen3_4b-instruct")
    d = cfg.as_dict()
    assert d["model_id"] == tiny_model_id
    assert d["delta_mem_adapter_id"].startswith("declare-lab/")


def test_ensure_deltamem_importable_finds_clone(monkeypatch, tmp_path):
    """If deltamem isn't installed globally but exists as a clone, the helper
    should find it via the candidate-roots search."""
    # Build a fake deltamem package with a core.py exposing the three names
    fake_root = tmp_path / "delta-Mem"
    pkg = fake_root / "deltamem"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    core = pkg / "core.py"
    core.write_text(
        "class HFDeltaMemConfig:\n"
        "    @classmethod\n"
        "    def from_pretrained(cls, d): return cls()\n"
        "def attach_delta_mem(model, config): return model\n"
        "def load_delta_mem_adapter(model, d): return None\n"
    )

    # Force the helper to look only at our tmp clone
    monkeypatch.setattr(backbone, "_candidate_deltamem_roots", lambda: [fake_root])
    # Clear any prior partial import state so our fake package wins
    for mod in list(__import__("sys").modules):
        if mod.startswith("deltamem"):
            del __import__("sys").modules[mod]

    # Should not raise; returns None on success
    backbone._ensure_deltamem_importable()
    import deltamem  # type: ignore  # noqa: F401
    assert deltamem is not None


def test_resolve_device_args_returns_expected_shapes(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert backbone._resolve_device_args("cpu") == {"device_map": None}
    assert backbone._resolve_device_args("cuda") == {"device_map": "cuda"}
    # auto on no-CUDA falls back to None
    assert backbone._resolve_device_args("auto") == {"device_map": None}


def test_resolve_device_args_includes_cpu_offload_on_single_gpu(monkeypatch):
    """Single-GPU auto mode should include a 'cpu' key in max_memory."""
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    # Fake a 12 GB GPU
    class _Props:
        total_memory = 12 * 1024**3
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: _Props())
    result = backbone._resolve_device_args("auto")
    assert result["device_map"] == "auto"
    assert "cpu" in result["max_memory"]
    assert 0 in result["max_memory"]


def test_load_backbone_falls_back_when_flash_attn_unavailable(tiny_model_id, monkeypatch):
    """Loading a CPU tiny model with FA2 requested should silently fall back."""
    cfg = backbone.BackboneConfig(
        model_id=tiny_model_id,
        dtype="float32",
        device="cpu",
        delta_mem_adapter_id=None,
        attn_implementation="flash_attention_2",  # likely unsupported on CPU + tiny LLaMA
    )
    # Should NOT raise; either FA2 works (unlikely on CPU) or fallback kicks in
    model, tok, session = backbone.load_backbone(cfg)
    assert model is not None
    assert tok is not None
    assert session is None


def test_attn_impl_for_hardware_filters_fa2_on_turing(monkeypatch):
    """On a Turing GPU (sm_75), FA2 should be filtered out to None."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i: (7, 5))
    assert backbone._attn_impl_for_hardware("flash_attention_2") is None


def test_attn_impl_for_hardware_passes_fa2_on_ampere(monkeypatch):
    """On an Ampere+ GPU (sm_80+), FA2 should pass through."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i: (8, 0))
    assert backbone._attn_impl_for_hardware("flash_attention_2") == "flash_attention_2"


def test_attn_impl_for_hardware_passes_through_non_fa2(monkeypatch):
    """Other impls (sdpa, eager) are not filtered."""
    assert backbone._attn_impl_for_hardware("sdpa") == "sdpa"
    assert backbone._attn_impl_for_hardware(None) is None


@pytest.mark.gpu
@pytest.mark.smoke
def test_load_qwen3_4b_with_delta_mem():
    cfg = backbone.BackboneConfig(
        model_id="Qwen/Qwen3-4B-Instruct-2507",
        dtype="bfloat16",
        device="cuda:0",  # δ-Mem path uses single device
        delta_mem_adapter_id="declare-lab/delta-mem_qwen3_4b-instruct",
    )
    model, tok, session = backbone.load_backbone(cfg)
    assert session is not None
    assert hasattr(session, "generate_reply")
