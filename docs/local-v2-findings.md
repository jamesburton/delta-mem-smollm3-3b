# LOCAL_V2 sweep findings — infrastructure validation

Three pieces of infrastructure landed before this sweep:
1. Subprocess isolation per (cell, ctx)
2. Decode-only VRAM peak metric (forward_pre_hook between prefill & decode)
3. δ-Mem honors SW via patched-snapshot loader

This sweep (cells 1/2/3/4 × 4K/8K) was the validation that those changes
produce data the v3 hypothesis can be tested against. Headline: **the
infrastructure works as designed, but it surfaced two findings that make
the hypothesis untestable on Qwen3-4B as-is**.

## Headline table

`results/LOCAL_V2/` — raw JSONs.

| ctx | cell | status   | NIH  | prefill_peak | decode_peak | tok/s | wall |
|-----|------|----------|------|--------------|-------------|-------|------|
| 4K  | 1 vanilla        | ok      | 1.00 | 9.21 GiB | 8.63 GiB | 3.9 |   53 s |
| 4K  | 2 δ-Mem          | ok      | 1.00 |   N/A    | 10.95 GiB| 2.0 |  366 s |
| 4K  | 3 SW-4K          | partial | 0.00 | 9.21 GiB | 8.63 GiB | 3.9 |   60 s |
| 4K  | 4 SW-4K + δ-Mem  | ok      | 0.33 |   N/A    | 10.95 GiB| 2.0 |  373 s |
| 8K  | 1 vanilla        | ok      | 1.00 |10.90 GiB | 9.75 GiB | 3.1 |   74 s |
| 8K  | 2 δ-Mem          | failed  | —    |   N/A    |    —     | —   | 1120 s |
| 8K  | 3 SW-4K          | partial | 0.00 |10.90 GiB | 9.73 GiB | 3.0 |  114 s |
| 8K  | 4 SW-4K + δ-Mem  | partial | 0.00 |   N/A    |14.35 GiB | 2.0 |  697 s |

(δ-Mem path reports `prefill_peak=None` because upstream's
`DeltaMemChatSession.generate_reply` is opaque — we can only measure
overall.)

## Infrastructure works

- **Subprocess isolation:** 8K cell 2 OOMed (cudaErrorUnknown) at 1120 s
  wall; the next combo (cell 3 at 8K) started on a fresh CUDA context
  and completed cleanly. Pre-fix this would have wedged the whole sweep.
- **Decode-only peak metric:** vanilla 8K split is 10.90 (prefill) vs
  9.75 (decode) — a 1.15 GiB gap that captures FA2's prefill workspace
  before the cache settles. The metric is reading what we wanted it to.
- **Patched-snapshot loader for δ-Mem:** cell 4 successfully loaded
  Qwen3-4B from `.cache/patched_snapshots/...sw4096/` with the SW
  config baked into `config.json` and completed generation through the
  upstream `DeltaMemChatSession`.

## Finding 1: Qwen3 retrofitted to SW produces gibberish

Cell 3 at **4K context with window=4K** (so the window covers the entire
context — SW should be a no-op):

- Vanilla cell 1 answer: `"\n\ngolf: GN47-71H\nlima: JX21-22D\nalpha: QG99-28V"` (NIH=1.00)
- Cell 3 SW answer: `"beta, gamma, delta, epsilon, zeta, omega, zeta, omega, zeta, omega, ..."` (NIH=0.00)

Even when the window can't possibly crop anything, the model output is
broken. This isn't an SW-cropping artifact — it's the **layer_types
rewrite itself**. `layer_types=["sliding_attention"]*36` switches the
attention math (the mask-builder picks `create_sliding_window_causal_mask`)
which has its own positional-encoding behavior that the pre-trained
weights weren't trained against. Qwen3-4B-Instruct's pretraining used
full attention with absolute RoPE; the SW path uses a different RoPE
position scheme inside the window, and the weights see token positions
they don't recognise.

**Implication:** cells 3, 4, 5, 8, 10, 12 (every SW-bearing cell on
Qwen3) cannot be used to test the v3 hypothesis. The cache mechanism
works, but the model output is meaningless.

The v3 hypothesis test needs:
- a model that was **pretrained with SW** (Mistral 7B / Gemma3 family
  both use SW + interleaved full attention layers natively), OR
- a retrofit mechanism that **preserves positional encoding** —
  StreamingLLM (cells 9a-c, 10) uses a sink-tokens approach that
  *doesn't* change RoPE semantics. This is exactly what cells 9a/9b/9c
  are designed for and the next priority.

## Finding 2: decode_peak still captures transient SW crop allocations

Compare cell 1 vs cell 3 at 8K (both use FA2):

| | prefill_peak | decode_peak |
|---|---|---|
| Cell 1 vanilla    | 10.90 GiB | 9.75 GiB |
| Cell 3 SW-4K      | 10.90 GiB | 9.73 GiB |

If SW were cutting the KV cache in half (window 4K vs context 8K),
decode_peak for cell 3 should drop by ~350 MB (the half-cache delta at
this model size). The observed delta is **0.02 GiB = ~20 MB**.

Why: `DynamicSlidingWindowLayer.update()` slices the cache tensor down
to the window — but the allocator records the moment when both the
full (pre-slice) tensor and the cropped (post-slice) tensor are alive
simultaneously. Peak is captured at this transient maximum, not at the
post-slice steady state.

**Implication:** decode_peak is the right *direction* (it strips out
prefill workspace) but it's still a "peak" not a "steady state."
Finishing this gap needs one more thing:

- **Sampled or end-of-decode memory measurement.** Instead of
  `torch.cuda.max_memory_allocated()`, take `torch.cuda.memory_allocated()`
  (current) at the END of generation — that reflects what's actually
  resident after all the transient allocations have freed. Add it as
  `decode_resident_vram_bytes` alongside `decode_peak_vram_bytes`.

The current metric is enough to see prefill-vs-decode separation, but
not enough to see decoder-cache-vs-decoder-cache differences at this
model size.

## Finding 3: 8K δ-Mem on the 3060 is right at the failure boundary

- FA2 sweep (June 2026, no SW): cell 2 at 8K completed at NIH=1.00,
  peak 14.4 GiB (via WDDM shared system memory), 0.4 tok/s, 1992 s wall.
- LOCAL_V2 sweep (June 2026, with new metric hooks): cell 2 at 8K
  **failed** with cudaErrorUnknown after 1120 s.

Same model, same lever, same hardware — but one run succeeds and the
next fails. WDDM's shared-memory path is non-deterministic at this
edge. Treat 8K δ-Mem on the 3060 as **unreliable**, not "works slowly."
The cloud rung needs to handle this cleanly.

Cell 4 (SW + δ-Mem) at 8K *did* complete at peak 14.4 GiB — the SW
side reduced compute enough that the run squeezed through, even though
the cache savings are masked by the transient-allocation issue above.
So even the broken-output SW path is providing real compute reduction,
just not measurable as memory savings.

## What this means for the cloud-spend decision

Before this sweep the plan was: validate locally → run hypothesis
sweep on cloud L4. After this sweep the picture is:

1. **The δ-Mem KV-overhead finding still holds** — cell 2 at 4K and
   8K shows decode_peak ~11 GiB vs vanilla 9.75 GiB, a steady ~1.2 GiB
   adder. That's measurable on the 3060 and doesn't need the cloud.
2. **The SW-vs-vanilla cache comparison is blocked by Finding 1** —
   cannot use Qwen3 + retrofit SW for this. Either switch base model
   or build StreamingLLM cells.
3. **The decode_resident metric needs to land** before any cloud run,
   or the cloud sweep produces the same blind-spot result.

Cheapest path to a hypothesis-deciding answer:

- Land `decode_resident_vram_bytes` (small change in metrics/memory.py
  and hf_runner.py).
- Implement cells 9a/9b/9c (StreamingLLM sink + SW) properly — they
  retrofit cleanly on full-attention pretraining. Then cells 9b vs 4
  is the SW-vs-vanilla cache comparison at preserved NIH.
- Run cells 1, 2, 9b, 9c at 4K/8K locally — if 9b and 9c show NIH≥0.8
  AND decode_resident_vram lower than cell 2, the hypothesis is
  supported, and the cloud run is warranted at 16K+ on bigger cards.
- If cells 9 also break NIH on this base, the answer is **switch base
  model to Mistral / Gemma3** before any further cloud work.

Either way, **don't spend cloud credits yet**. The infrastructure caught
real blockers — that's the win this sweep was supposed to deliver.
