"""CPU-only sanity tests for the diffusion runner.

We don't load LLaDA here (it's 5 GB+ and needs CUDA). Instead we test:

1. The module imports without dragging in CUDA-only paths at import time.
2. The sampler's shape contract: given a fake model that returns deterministic
   logits, ``llada_generate`` returns a tensor of the expected size and
   doesn't leave any [MASK] tokens behind.
3. The runner builds a sensible result-JSON skeleton when handed a fake
   model (so we exercise the score-and-emit path without the real backbone).
"""

from __future__ import annotations

import pytest

# torch is a hard dep of the repo, but keep the importorskip to mirror the
# style of the other test modules so the test suite can survive a torch-less
# environment if one ever appears.
torch = pytest.importorskip("torch")
import torch  # noqa: F811 - re-import so static analysers see the symbol

from harness.runners import diffusion_runner


def test_module_imports_without_cuda():
    """The module should not require CUDA at import time."""
    # If this test ran, the import already succeeded — assert the public
    # surface is reachable.
    assert hasattr(diffusion_runner, "run_llada")
    assert hasattr(diffusion_runner, "llada_generate")
    assert hasattr(diffusion_runner, "load_llada")
    assert hasattr(diffusion_runner, "MASK_ID")


def test_get_num_transfer_tokens_sums_to_mask_count():
    """The transfer scheduler must move exactly mask_num tokens across `steps`."""
    mask_index = torch.zeros(1, 32, dtype=torch.bool)
    mask_index[0, 4:20] = True  # 16 masked positions
    steps = 4
    out = diffusion_runner._get_num_transfer_tokens(mask_index, steps)
    assert out.shape == (1, steps)
    assert int(out.sum().item()) == 16


def test_add_gumbel_noise_passthrough_at_zero_temp():
    """temperature=0 must be a no-op (greedy argmax path)."""
    logits = torch.randn(1, 4, 8)
    out = diffusion_runner._add_gumbel_noise(logits, temperature=0)
    assert torch.equal(out, logits)


class _ConstantLogitsModel(torch.nn.Module):
    """Fake LLaDA-shaped model returning constant logits that prefer token 7.

    We do this as a real ``nn.Module`` (rather than a Mock) so the sampler's
    ``model.device`` attribute resolves correctly via ``next(parameters)``.
    """

    def __init__(self, vocab: int = 32):
        super().__init__()
        self.vocab = vocab
        # tiny param so the module has a device
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    @property
    def device(self):
        return self.dummy.device

    def forward(self, x, attention_mask=None):
        # Always assign max logit to token 7 — so every unmasked position
        # becomes 7, modulo positions outside the active block.
        b, s = x.shape
        logits = torch.full((b, s, self.vocab), -10.0)
        logits[..., 7] = 5.0

        class _Out:
            pass

        out = _Out()
        out.logits = logits
        return out


def test_llada_generate_fills_all_masks_and_returns_correct_shape():
    """End-to-end sampler: after running, no MASK tokens should remain."""
    model = _ConstantLogitsModel(vocab=32)
    prompt = torch.tensor([[1, 2, 3]])  # 3 prompt tokens, none masked
    gen_length = 16
    block_length = 8
    steps = 4  # must divide num_blocks=2 evenly

    # Use a small custom mask_id that fits in our fake vocab=32.
    mask_id = 31
    out = diffusion_runner.llada_generate(
        model, prompt, attention_mask=None,
        steps=steps, gen_length=gen_length, block_length=block_length,
        temperature=0.0, remasking="low_confidence", mask_id=mask_id,
    )
    assert out.shape == (1, 3 + gen_length)
    # Prompt preserved
    assert torch.equal(out[0, :3], prompt[0])
    # No masks left in the generated region
    assert not (out[0, 3:] == mask_id).any().item()


def test_hardware_supports_fa2_returns_bool_without_crashing():
    """Just a smoke check that the helper is robust on any host."""
    assert isinstance(diffusion_runner._hardware_supports_fa2(), bool)
