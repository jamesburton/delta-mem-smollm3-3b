"""HF assistant_model speculative decoding wrapper."""

from __future__ import annotations

from typing import Any, Optional


def load_assistant(model_id: str, *, device: str = "cuda", dtype: str = "bfloat16") -> Any:
    """Load a small draft model. Caller is responsible for vocab compatibility
    (same tokenizer family as the target)."""
    from transformers import AutoModelForCausalLM
    import torch
    dtype_obj = getattr(torch, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype_obj,
        device_map=device if device != "cpu" else None,
        trust_remote_code=True,
    )
    model.eval()
    return model


def generate_with_spec_decode(
    target,
    tokenizer,
    assistant,
    *,
    prompt: str,
    max_new_tokens: int,
) -> str:
    """One-shot generation through HF's built-in spec-decode."""
    import torch
    inputs = tokenizer(prompt, return_tensors="pt").to(next(target.parameters()).device)
    with torch.no_grad():
        out = target.generate(
            **inputs,
            assistant_model=assistant,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    return tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
