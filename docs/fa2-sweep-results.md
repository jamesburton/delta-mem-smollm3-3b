# FA2 context sweep — RTX 3060 12 GB

First sweep across NIH context sizes with FlashAttention-2 enabled locally.
Establishes (a) the FA2 speedup vs SDPA mem-efficient and (b) the practical
context ceiling on a 12 GiB consumer card.

**Environment:** Windows 11, Python 3.11, torch 2.9.0+cu128, flash-attn 2.8.3,
transformers 5.9.0, accelerate 1.13.0. Model: Qwen3-4B-Instruct-2507 in bf16.
δ-Mem adapter: `declare-lab/delta-mem_qwen3_4b-instruct`. Spec-decode draft:
the cell 6/7 config (see `harness/cells.py`).

Env knobs: `GPU_MAX_PCT=0.80`, `CPU_MAX_GIB=24`, `PYTHONIOENCODING=utf-8`,
`STAGE=LOCAL_FA2`. NIH multi-needle = 3, `max_new_tokens=64`.

Raw per-cell JSONs in `results/LOCAL_FA2/ctx-{N}/cell-{id}.json`.

## Headline results

### Speedup vs SDPA mem-efficient (1.5K NIH → 2K NIH, same model)

| Cell             | NIH | SDPA tok/s | FA2 tok/s | speedup |
|------------------|-----|------------|-----------|---------|
| 1 vanilla        | 1.0 | 1.0        | 4.2       | **4.2×** |
| 2 +δ-Mem         | 1.0 | 0.8        | 2.2       | **2.75×** |
| 6 +spec-decode   | 1.0 | 2.4        | 2.8       | 1.2× (wall time 7×, from 413 s → 60 s) |

FA2 gives the biggest decode-throughput lift on the dense path (cell 1). The
δ-Mem path is bounded by the adapter's write-phase, not by attention math, so
FA2 helps less there. Spec-decode tok/s gain is modest, but wall-clock collapses
because the prefill is dominated by attention and FA2 owns prefill.

### Quality is preserved

NIH = 1.00 across **every** combo that fit in VRAM (2K all four cells, 4K all
four cells, 8K cells 1 and 2). FA2 doesn't drop accuracy.

### Context scaling

| Context | Cell 1 (vanilla)            | Cell 2 (+δ-Mem)         | Cell 6 (+spec)  | Cell 7 (combo)  |
|---------|-----------------------------|-------------------------|-----------------|-----------------|
| 2 K     | 8.4 GiB / 4.2 tok/s         | 9.3 / 2.2               | 9.9 / 2.8       | 9.3 / 2.2       |
| 4 K     | 9.2 / 3.9                   | 11.0 / 2.2              | 11.2 / 3.0      | 11.0 / 2.3      |
| 8 K     | **10.9 / 3.7**              | 14.4 / 0.4 †            | OOM ‡           | OOM ‡           |
| 16 K    | OOM after 1272 s wall §     | OOM                     | OOM             | OOM             |

† Cell 2 at 8K technically completed at NIH=1.0, but its 14.4 GiB peak exceeds
the 3060's 12 GiB physical VRAM. It ran via WDDM shared-system-memory spillover
at 0.35 tok/s — usable for correctness experiments but not for throughput.

‡ Cell 6 at 8K OOMed mid-decode after 671 s wall, wedging the CUDA context.
Cell 7 (= cell 6 + δ-Mem) inherits the same problem.

§ Cell 1 at 16K OOMed mid-decode after 1272 s wall (in a fresh process with
clean CUDA state). Model weights (8 GB bf16) + base KV at 16K (1.4 GB by the
harness's metric, plus the unmeasured activation/workspace overhead) does not
fit in 12 GiB physical VRAM. 16K vanilla on the 3060 is **not viable**.

### The δ-Mem memory surprise

The KV cache size (as reported by `kv_cache_bytes_at_target_len`) is **identical**
between cells 1 and 2 at every context size: 0.17 GB @ 2K, 0.34 GB @ 4K,
0.69 GB @ 8K. But the peak VRAM diverges:

| Context | Cell 1 peak | Cell 2 peak | Δ          |
|---------|-------------|-------------|------------|
| 2 K     | 8.4 GiB     | 9.3 GiB     | +0.9 GiB   |
| 4 K     | 9.2 GiB     | 11.0 GiB    | +1.8 GiB   |
| 8 K     | 10.9 GiB    | 14.4 GiB    | +3.5 GiB   |

δ-Mem's sidecar state and write-phase activations grow with context faster than
linearly. The KV-compression promise (the original hypothesis) doesn't show up
in the harness's KV metric at all — that metric only captures the base model's
attention KV, which δ-Mem doesn't replace. δ-Mem **adds** memory on top instead
of replacing the KV cache; on the 3060, that pushes the practical context
ceiling **down** for the δ-Mem path, not up.

This is the critical finding for the v3 test-matrix hypothesis ("δ-Mem +
sparse attention + MTP could reduce KV by more than δ-Mem adds while MTP gives
speedup"). What we see on this profile is the opposite of the addition side:
δ-Mem adds 3.5 GiB at 8K context with no observed KV reduction. Either:

1. The adapter's KV-savings only manifest at much longer contexts where the
   base attention KV would dominate (16K+ on bigger cards), OR
2. The harness's KV metric is too narrow — the actual KV equivalent has been
   redistributed into the sidecar state we're now seeing as "peak VRAM."

Cell 4 (sliding-window + δ-Mem) and cell 5 (aggressive SW + δ-Mem) are the
cells designed to test option (2) — they cap the base KV explicitly so any
δ-Mem savings become visible. Those become higher priority for the next sweep.

## Practical context ceilings on the 3060

Based on observed peaks (with FA2 enabled):

| Cell                   | Highest verified working context | Notes |
|------------------------|----------------------------------|-------|
| 1 (vanilla)            | **8 K** (peak 10.9 GiB, 3.7 tok/s) | 16K OOMs mid-decode |
| 2 (+δ-Mem)             | **4 K** (peak 11.0 GiB, 2.2 tok/s) | 8K spills to shared memory |
| 6 (+spec-decode)       | **4 K** (peak 11.2 GiB, 3.0 tok/s) | 8K OOM after 671 s |
| 7 (combo δ-Mem + spec) | **4 K** (peak 11.0 GiB, 2.3 tok/s) | 8K OOM |

For real long-context work this profile needs the L4 (24 GiB) or A100 (40+
GiB) cloud rung. The 3060 is fine for fast feedback at ≤4K and reasonable
data at 8K vanilla.

## Failure modes observed

### Wedged CUDA context after OOM

When cell 6 at 8K hit OOM mid-decode (after 671 s), the entire CUDA context
became wedged: every subsequent cell in the same Python process failed at
model load with `cudaErrorUnknown`. The recovery sweep (cells 6/7 at 8K in
a fresh process) reproduced the OOM on cell 6 immediately.

**Implication for harness:** the inter-cell cleanup we added (`remove_hook_from_module` +
`release_memory()` + `ipc_collect()`) works for clean exits but cannot recover
from a CUDA exception. The sweep needs to either:

- Detect a `cudaErrorUnknown` and exit the whole process so the next ctx/cell
  combination starts on a fresh CUDA context, or
- Run each cell in a subprocess.

This is a follow-up; for now the rule is: **on OOM, restart the sweep manually
from the next viable cell.**

### `UnicodeEncodeError` rendering the summary on Windows console

The `δ` character in cell titles (`Qwen3-4B + δ-Mem adapter`) crashes the
final `print(md)` in `context_sweep.py` when stdout is the Windows `charmap`
encoding. The file write succeeds; only the stdout render fails. Fixed in
[`scripts/context_sweep.py`](../scripts/context_sweep.py): the summary file is
written first, the stdout render is wrapped in `try/except UnicodeEncodeError`,
and the file write uses explicit `encoding="utf-8"`.

## What's next

1. ✅ **16 K vanilla cell 1** — confirmed: OOMs mid-decode after 1272 s wall.
   The 3060 ceiling for vanilla full-attention bf16 on a 4B model is between
   8 K and 16 K context.
2. **Cells 3/4/5** (sliding-window variants) at 2K/4K/8K. These are the cells
   designed to test whether δ-Mem's hidden KV savings emerge once the base KV
   is capped. Cell 4 (SW-4K + δ-Mem) should fit comfortably; cell 5 (SW-2K +
   δ-Mem) even better.
3. **Cell 13/14** (MoE-style + MTP) on the cloud rung — the 3060 isn't a
   sensible platform for those cells given the memory pattern we observed.
4. **Harness improvement**: make `context_sweep.py` resilient to CUDA wedge by
   running each cell in a fresh subprocess. Without this, a single OOM
   poisons the rest of the sweep. Track as a follow-up; for now the workaround
   is to re-run only the failed (cell, ctx) combos after an OOM.

Move long-context δ-Mem characterization to L4 (24 GiB) or A100 (40 GiB) cloud
rungs. The data we have is already enough to update the test-matrix hypothesis.
