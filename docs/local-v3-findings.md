# LOCAL_V3 sweep findings — δ-Mem is pure overhead on Qwen3-4B at ≤8K

The `decode_resident_vram_bytes` metric (commit af9c103) finally lets us
measure the steady-state cache size during decode rather than the
transient prefill workspace peak. With that in place, the cleanest
question we can answer locally is: **what does δ-Mem actually cost in
memory, and at what context does it pay off?**

## Headline numbers

`results/LOCAL_V3/` — raw JSONs. Cells 1 (vanilla) and 2 (δ-Mem) at
4K and 8K target tokens (NIH-task actual tokens ≈ 2× target).

| ctx | cell | NIH | prefill_peak | decode_peak | decode_resident | tok/s |
|-----|------|-----|--------------|-------------|-----------------|-------|
| 4K  | 1 vanilla | 1.00 | 9.21 GiB | 8.63 GiB | **8.62 GiB** | 3.4 |
| 4K  | 2 δ-Mem   | 1.00 | (opaque) |10.95 GiB | (opaque)     | 1.8 |
| 8K  | 1 vanilla | 1.00 |10.90 GiB | 9.75 GiB | **9.72 GiB** | 3.3 |
| 8K  | 2 δ-Mem   | 1.00 | (opaque) |14.35 GiB | (opaque)     | 1.5 |

`decode_resident` is `None` for δ-Mem because the upstream
`DeltaMemChatSession` runs its write+decode opaquely; we can only
measure `decode_peak` end-to-end.

## What the data says

**Comparing peaks (the only apples-to-apples we have for δ-Mem):**

|     | decode_peak vanilla | decode_peak δ-Mem | δ-Mem overhead |
|-----|---------------------|-------------------|----------------|
| 4K  | 8.63 GiB            | 10.95 GiB         | **+2.32 GiB**  |
| 8K  | 9.75 GiB            | 14.35 GiB         | **+4.60 GiB**  |

**δ-Mem's overhead grows faster than vanilla KV:**

|              | 4K → 8K growth |
|--------------|----------------|
| Vanilla KV   | +1.12 GiB      |
| δ-Mem sidecar| +3.40 GiB (~3× faster) |

That's the inverse of what the lever was supposed to do. δ-Mem is
described as a side-state mechanism that should cap KV growth and
trade compute for memory; on this base at this scale, it's adding 2–5
GiB of sidecar that scales linearly-or-worse with context.

## Break-even arithmetic

For δ-Mem to be worth it at 8K, you'd need the cache savings to
exceed the +4.60 GiB adder. Qwen3-4B bf16 vanilla KV uses about
**144 KB per token** (36 layers × 8 KV heads × 128 head_dim × 2 bytes
× 2 K+V tensors). The break-even is:

  4.60 GiB / 144 KB ≈ **32,000 vanilla tokens**

But that's the *minimum* context where δ-Mem's adder is even *matched*
by what vanilla would have to cache. To actually *win* on memory the
adapter would need to reduce vanilla KV to near zero — and the LOCAL_V2
sweep showed δ-Mem doesn't reduce KV at all on this base; it adds
sidecar *on top of* the standard cache.

Net: on Qwen3-4B at ≤8K context on the 3060, **δ-Mem is pure overhead**.
It costs 2–5 GiB of VRAM and 2× wall time, and produces the same NIH=1.0
that vanilla produces.

## What about quality?

Both vanilla and δ-Mem hit NIH=1.00 at both 4K and 8K. δ-Mem does not
*degrade* quality. It just doesn't *add* anything at this scale, while
charging a hefty memory and time premium.

This may change at very long context (32K+) where vanilla KV alone
would exceed 4.6 GiB and δ-Mem's compression promise would have
something to bite into. Testing that is a cloud rung question — and
the 3060 ceiling is 8K, so we can't see the crossover locally.

## What this means for the path forward

**Stop spending local cycles on δ-Mem cells until we can test at 32K+.**
At ≤8K, the data says it's strictly worse than vanilla. Cell 2, 7, 8,
10, 11, 14, 15 (every δ-Mem cell) is therefore on hold for local work.

**Where the local 3060 *can* still inform decisions:**

1. **Pure-attention KV growth measurements** at 4K/8K — already done,
   gives the baseline curve for what δ-Mem would need to beat.
2. **GGUF / llama.cpp cells (9a, 9b, 9c)** — these test multi-token
   prediction (MTP) on `unsloth/Qwen3.5-4B-MTP-GGUF` at Q4/Q5/Q8
   quantization. Different stack, different cache mechanics; worth
   running to see whether quantization makes long-context vanilla
   viable on the 3060 *without* needing δ-Mem at all. If Q5_K_M fits
   16K-24K context within 12 GiB at reasonable quality, that
   sidesteps the entire δ-Mem hypothesis.
3. **Spec-decode behavior alone** — cell 6 hit 9.9 GiB peak / 2.8 tok/s
   at 2K previously, but the new metric will show whether the draft
   model's KV is the bottleneck or the cache.

**Where the 3060 *cannot* inform decisions and Kaggle 16 GiB might:**

1. The crossover context where vanilla KV exceeds δ-Mem's adder. Need
   16K and 24K context tested, which neither vanilla nor δ-Mem fit in
   12 GiB. Kaggle T4×2 = 30 GiB total / 16 GiB per — possibly enough
   for vanilla 16K, definitely for vanilla 8K.
2. δ-Mem's quality at long context. We've only shown it preserves
   quality at 4K/8K; the actual claim about δ-Mem is that it preserves
   quality at much longer contexts where vanilla would degrade.

## Recommended next step

Run cells 9a/9b/9c (GGUF MTP) **at 4K and 8K and 16K** on the local
3060 with the new metric. Three reasons:

1. GGUF is much smaller — Qwen3.5-4B at Q4_K_M is ~2.5 GiB on disk,
   ~3-4 GiB in VRAM. That makes 16K+ context viable on the 3060.
2. MTP changes the speed equation entirely (it's the "MTP gives
   speedup" half of the v3 hypothesis).
3. It tests the llama.cpp leg of the harness, which is currently
   completely uncharacterized.

If cell 9b at 16K shows NIH≥0.95 within 6-8 GiB VRAM at ≥10 tok/s,
**that's the answer the project is actually looking for** — quantized
MTP on a base model that doesn't need δ-Mem's retrofit at all.

## Methodology fixes still pending

1. **`decode_resident` for the δ-Mem session path.** The upstream
   `DeltaMemChatSession` is opaque to our hook. Either patch upstream
   to expose a generate-with-callback API, or run the session's
   write/decode in two explicit calls so we can hook between them.
   Without this, we can only compare δ-Mem to vanilla via the peak
   metric, which still mixes prefill workspace and cache.

2. **Subprocess isolation regression.** Cell 1 at 8K crashed with
   ACCESS_VIOLATION (exit 3221225477) in 25 s when run via subprocess
   isolation, but worked fine in-process. The crash is during model
   load, not generation. Could be a Windows-specific subprocess+CUDA
   init race; tracked but not blocking the current sweeps which all
   run `--no-isolate`.

3. **NIH-task token target overshoots.** The task generator at
   `target_tokens=8000` produces an actual NIH prompt of ~16,088
   tokens. The "ctx" labels in our reports are therefore optimistic
   by 2×. Reframing the existing data: what we call "8K" is really
   a ~16K-token prompt. Adjust the target_tokens accounting or
   relabel the columns.
