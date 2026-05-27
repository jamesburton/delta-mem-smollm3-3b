"""Shared fixtures.

We avoid loading the real 4B base in unit tests — use sshleifer/tiny-gpt2
(~5 MB) as a structurally-similar stand-in. Real-model tests are gated by
the `smoke` mark and the `RUN_SMOKE` env var.
"""

import os
import pytest
import torch


@pytest.fixture(scope="session")
def tiny_model_id() -> str:
    return "sshleifer/tiny-gpt2"


@pytest.fixture(scope="session")
def tiny_lm(tiny_model_id):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tiny_model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(tiny_model_id)
    model.eval()
    return model, tok


def pytest_collection_modifyitems(config, items):
    skip_gpu = pytest.mark.skip(reason="no CUDA")
    skip_smoke = pytest.mark.skip(reason="set RUN_SMOKE=1 to enable")
    have_cuda = torch.cuda.is_available()
    run_smoke = os.environ.get("RUN_SMOKE") == "1"
    for item in items:
        if "gpu" in item.keywords and not have_cuda:
            item.add_marker(skip_gpu)
        if "smoke" in item.keywords and not run_smoke:
            item.add_marker(skip_smoke)
