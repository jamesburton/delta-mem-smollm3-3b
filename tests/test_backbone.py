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
