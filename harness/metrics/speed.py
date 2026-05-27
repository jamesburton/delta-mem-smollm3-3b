"""Speed measurement: prefill time, time-to-first-token, decode tok/s."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class SpeedRecord:
    prefill_seconds: float
    ttft_seconds: float
    decode_seconds: float
    new_tokens: int

    @property
    def decode_tokens_per_second(self) -> float:
        # decode_seconds excludes prefill; covers the n-1 generated tokens
        # after the first (TTFT-bounded) one
        gen_after_first = max(1, self.new_tokens - 1)
        return gen_after_first / self.decode_seconds if self.decode_seconds > 0 else 0.0

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["decode_tokens_per_second"] = self.decode_tokens_per_second
        return d


def timed_generation(
    model,
    tokenizer,
    *,
    prompt: str,
    max_new_tokens: int,
    seed: int = 0,
    assistant_model=None,
) -> SpeedRecord:
    """Run one generation pass and record fine-grained timings.

    Strategy: we drive generation in two steps to break out prefill from decode:
    1) one forward pass with use_cache=True to get the first token + KV cache
       (counts toward prefill_seconds and ttft_seconds);
    2) then `generate(...)` for the remaining tokens (decode_seconds).
    """
    import torch

    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    if seed is not None:
        torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # ---- prefill + first token ----
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(**inputs, use_cache=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_prefill_done = time.perf_counter()

    next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_first_tok = time.perf_counter()

    prefill_s = t_prefill_done - t0
    ttft_s = t_first_tok - t0

    # ---- remaining decode via generate(...) ----
    if max_new_tokens <= 1:
        return SpeedRecord(
            prefill_seconds=prefill_s,
            ttft_seconds=ttft_s,
            decode_seconds=0.0,
            new_tokens=max_new_tokens,
        )

    generate_kwargs: Dict[str, Any] = dict(
        max_new_tokens=max_new_tokens - 1,
        do_sample=False,
        use_cache=True,
        past_key_values=out.past_key_values,
    )
    if assistant_model is not None:
        generate_kwargs["assistant_model"] = assistant_model

    new_input_ids = torch.cat([inputs["input_ids"], next_tok], dim=1)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    _ = model.generate(new_input_ids, **generate_kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t2 = time.perf_counter()

    return SpeedRecord(
        prefill_seconds=prefill_s,
        ttft_seconds=ttft_s,
        decode_seconds=t2 - t1,
        new_tokens=max_new_tokens,
    )
