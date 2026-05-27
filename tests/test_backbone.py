import pytest
import torch

from harness import backbone


def test_load_tiny_backbone_returns_model_and_tokenizer(tiny_model_id):
    cfg = backbone.BackboneConfig(model_id=tiny_model_id, dtype="float32", device="cpu",
                                   delta_mem_adapter_id=None)
    model, tok = backbone.load_backbone(cfg)
    assert model is not None
    assert tok is not None
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
    # Make sure a real deltamem isn't preempting us
    monkeypatch.setitem(__import__("sys").modules, "deltamem", None)
    monkeypatch.setitem(__import__("sys").modules, "deltamem.core", None)
    # Clear any prior partial import state
    for mod in list(__import__("sys").modules):
        if mod.startswith("deltamem"):
            del __import__("sys").modules[mod]

    HFDeltaMemConfig, attach_fn, load_fn = backbone._ensure_deltamem_importable()
    assert HFDeltaMemConfig is not None
    assert callable(attach_fn)
    assert callable(load_fn)


@pytest.mark.gpu
@pytest.mark.smoke
def test_load_qwen3_4b_with_delta_mem():
    cfg = backbone.BackboneConfig(
        model_id="Qwen/Qwen3-4B-Instruct-2507",
        dtype="bfloat16",
        device="cuda",
        delta_mem_adapter_id="declare-lab/delta-mem_qwen3_4b-instruct",
    )
    model, tok = backbone.load_backbone(cfg)
    assert hasattr(model, "delta_mem") or "delta" in str(type(model)).lower()
