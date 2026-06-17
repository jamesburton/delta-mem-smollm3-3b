# NIH eval integration for block-diffusion LLMs (LLaDA-8B, Dream-7B)

**Status:** design + runner ready to launch on T1. **Not yet executed** —
the GPU is busy tonight. Run the one-liner in the "Run me" section in the
morning.

## What's new

Two purely additive artifacts; the existing AR (Qwen3-4B ± δ-Mem) NIH
pipeline is untouched.

| File | Role |
|---|---|
| `harness/runners/diffusion_runner.py` | New runner. Same call signature as `hf_runner.run(cell, RunConfig) → record`, but drives the LLaDA semi-AR mask-denoising sampler. Returns the same result-JSON shape so `scripts/context_sweep.render_sweep_summary` reads it. |
| `scripts/nih_llada_8b.py` | CLI runner that uses the existing NIH machinery (`harness.metrics.quality.make_multineedle_task` + scorer) and dispatches to the diffusion runner. Writes per-context JSON to `results/LLADA8B_NIH/ctx-<N>/cell-D1.json`. |

The needle generator, distractor generator, and grader (the entire
contents of `harness/metrics/quality.py`) are reused **unchanged** —
diffusion output is decoded to the same free-form text that the AR runner
produces, so the grader is architecture-agnostic.

## Adapter choice and why

We considered two options:

**(a) Thin `generate`-shaped shim around the LLaDA sampler** so the
existing `hf_runner.run` would drive it via `model.generate(...)`.
Rejected because:

- `hf_runner._generate_with_memory_split` attaches a `forward_pre_hook` on
  the model and assumes call #0 = prefill, call #N>=1 = decode. Block
  diffusion has **no prefill/decode split** — every step is one full-seq
  forward of identical shape. The hook would mis-classify steps and
  produce nonsense memory numbers (would always book the entire run as
  "prefill", since the AR helper resets stats after call #0 expecting
  KV-cache resident bytes).
- The AR runner builds `KVCache`-shaped `past_key_values` for the timing
  helper. LLaDA has no KV-cache. Bolting that on would be dead code that
  has to be skipped via a feature flag.
- LLaDA's tokenizer needs `padding_side="left"` and the inner sampler
  drives the model with manual `model(x, attention_mask=...)` calls (not
  `.generate`). Wrapping that under HF's `GenerationMixin` would mean
  reimplementing `prepare_inputs_for_generation` and the stopping criteria
  from scratch — exactly what we'd be trying to avoid.

**(b) New `diffusion_runner.py` module alongside `hf_runner.py`. ← chosen.**

- Mirrors the AR runner's public surface: `run_llada(cell, RunConfig) → dict`
  with the same dict shape, so the existing `context_sweep` summary
  renderer and the result-JSON consumers see no difference.
- Reuses `harness.metrics.quality` (needles, distractors, grader) — zero
  duplication of the eval definition.
- Honest about diffusion semantics: `speed.prefill_seconds` and
  `ttft_seconds` are recorded as `0.0` to make the contract explicit, and
  adds diffusion-specific fields (`diffusion_steps`,
  `diffusion_block_length`, `diffusion_ms_per_step`) so we don't fake
  AR-shaped numbers.
- One small import-coupling: re-exports `RunConfig` from `hf_runner` so
  callers can swap runners without touching their config object.

## tok/s metric

For block diffusion, `tok/s = emitted_tokens / wall_time`. This is the
definition documented in `LLMs.md → Architecture family notes →
Block-diffusion LLMs`. The diffusion runner stores the wall time in
`speed.decode_seconds` and the emitted count in `speed.new_tokens` so the
existing `speed.decode_tokens_per_second` accessor (which is just
`max(1, n-1) / decode_seconds`) lines up — with the one-off case that for
diffusion `n` is exactly `gen_length` (modulo trailing pad), not "the
number of decode steps after TTFT".

We **don't** try to compute a per-token decode tok/s the AR way (that would
divide by `steps`, which is meaningless for diffusion since each step
emits ~`gen_length/steps` tokens). The number under
`decode_tokens_per_second` is the wall-time-based tok/s; this is what
LLMs.md uses to quote "27.9 tok/s @ block_length=128, steps=16".

## NIH grader compatibility

Verified by code inspection: `quality.score_multineedle(task, answer_text)`
and `quality.score_hard_multineedle(task, answer_text)` are pure-Python
functions over the final answer string. They don't care whether that
string came from an AR `generate()` call or from `tokenizer.batch_decode`
of a diffusion sampler's output tensor. The diffusion runner produces the
same input shape (single-string answer) so no grader changes were needed.

## Long-context caveat

Three real risks, in order of likelihood:

1. **LLaDA's trained max_sequence_length is 4096.** At our default
   `ctx=2000 + gen_length=256` we're fine. At `ctx=4000 + gen_length=256`
   we're inside the trained window but very close to the ceiling. Above
   `ctx ≈ 3500` recall will likely degrade as RoPE extrapolates beyond
   training. The runner prints a warning above this threshold.

2. **No KV-cache → O(L²) per step.** Each of the 16 steps re-encodes the
   entire sequence. Per-step wall scales roughly with
   `(L_total / 50)^2 * 290ms` (using the smoke-test ~290 ms/fwd at
   ~50 token chat prompt as the constant). At ctx=2000 that predicts
   ~30 s per generation; at ctx=4000 ~120 s.

3. **Without FA2, attention OOMs at long ctx** — same finding as
   Qwen3-4B (`docs/fa2-sweep-results.md`): SDPA's math-fallback materialises
   an O(N²) attention matrix and pages through WDDM at ctx=4K or OOMs at
   ctx=8K. **LLaDA carries its own `flash_attention` config flag**
   (separate from HF's `attn_implementation`) — the diffusion runner
   enables it by default on sm_80+ hardware where the `flash_attn` package
   is importable. Verified by reading
   `models--GSAI-ML--LLaDA-8B-Instruct/.../modeling_llada.py` lines
   574–648: the model picks `flash_attn_func` only if `config.flash_attention=True`,
   otherwise falls back to torch SDPA.

   Override with `--flash-attention off` to deliberately reproduce the
   long-ctx failure mode for diagnosis.

## Expected runtime on T1

(RTX 3060 12GB, bnb-4bit NF4 + bf16 compute, FA2 on, steps=16, block=128.
Per-step time extrapolated from the LLMs.md smoke result of ~290 ms/fwd at
~50-token chat prompt.)

| ctx | gen | per-step est. | wall est. | tok/s est. | risk |
|---|---|---|---|---|---|
| 2000 | 256 | ~2 s | **~30 s** | ~8 | low — safe first target |
| 4000 | 256 | ~8 s | **~130 s** | ~2 | medium — within trained window, near ceiling |
| 6000 | 256 | ~18 s | **~290 s** | ~0.9 | high — past trained ceiling; expect quality drop even if it doesn't OOM |
| 8000 | 256 | ~32 s | **~520 s** | ~0.5 | very high — likely OOM without FA2; even with FA2, RoPE extrapolation strain |

Numbers are upper bounds on tok/s (per-step is dequant-floored at ~290 ms;
the O(L²) attention work runs on top). Treat them as "1–2× this much
better than the worst case".

## Run me

In the morning, from the repo root with the venv active:

```powershell
C:\Python311\python.exe scripts\nih_llada_8b.py --ctx 2000
```

Recommended first sweep to characterise the scaling:

```powershell
C:\Python311\python.exe scripts\nih_llada_8b.py --ctx 2000,4000 --max-new-tokens 256 --steps 16 --block-length 128
```

Hard NIH (10 needles + 30 code-shaped distractors + mapping check) once
the easy NIH passes:

```powershell
C:\Python311\python.exe scripts\nih_llada_8b.py --ctx 2000 --task-type hard_multineedle
```

Diagnose long-ctx behaviour without FA2 (do this last — expect OOM at
ctx ≥ 4000):

```powershell
C:\Python311\python.exe scripts\nih_llada_8b.py --ctx 4000 --flash-attention off
```

Results land in `results/LLADA8B_NIH/ctx-<N>/cell-D1.json` plus a roll-up
`results/LLADA8B_NIH/summary.json`. The JSON shape matches the AR cells
so `scripts/context_sweep.render_sweep_summary` will render it if you
feed it back in.

## Future: Dream-7B as second backbone

Same runner should work unchanged: Dream-7B (`Dream-org/Dream-v0-Instruct-7B`,
already in HF cache) is dense block-diffusion with the same sampler shape.
The only known difference is the `[MASK]` token id (the LLaDA constant
`MASK_ID=126336` is hardcoded in the sampler). When wiring Dream-7B, lift
that constant to a per-model lookup or pass it through `run_llada(..., mask_id=...)`.
Add a second invocation under
`scripts/nih_dream_7b.py` mirroring `nih_llada_8b.py`.

## Files touched

- **Added:** `harness/runners/diffusion_runner.py` (new module)
- **Added:** `scripts/nih_llada_8b.py` (new script)
- **Added:** `docs/nih_diffusion_integration.md` (this file)
- **Added:** `tests/test_diffusion_runner.py` (CPU-only sanity tests)
- **Not touched:** `harness/runners/hf_runner.py`, `harness/metrics/quality.py`,
  `harness/cells.py` (the diffusion cell is built inline in the runner
  script rather than added to the global `CELLS` registry — the registry
  is the locked AR test matrix and shouldn't be polluted with one-off
  exploration cells).
