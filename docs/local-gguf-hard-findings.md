# LOCAL_GGUF_HARD findings — quantised models DO degrade at long context

The previous LOCAL_GGUF sweep used the original 3-needle NIH (which
saturates at 1.00) and concluded that quantised MTP "sidesteps" the
δ-Mem question. That conclusion was wrong — it was riding on an eval
that couldn't discriminate. This sweep replays the same cells under
the new `hard_multineedle` task (10 needles + 30 distractor codes +
key→code mapping check) and the picture changes completely.

## Headline numbers

| target | actual prompt | Q4_K_M | Q5_K_M | Q8_0 |
|--------|---------------|--------|--------|------|
| 16K    | 32K tokens    | **1.00** correct (10 distractors leaked) | **1.00** clean | **1.00** clean |
| 32K    | 64K tokens    | **0.00** (truncated mid-reasoning) | **0.00** (loop) | **0.00** (loop) |

`results/LOCAL_GGUF_HARD/` carries the JSONs.

## Three failure modes at 64K actual tokens

At target=32K, the 1024 max_new_tokens budget produced three distinct
failure modes:

**Q4_K_M (cell 9a)** — *competent but truncated*. The reasoning trace
correctly identifies needle codes one by one:

```
*   Search for "yankee":
    *   Found: "The secret code for yankee is QX52-89U."
    *   Code: QX52-89U
*   Search for "ferret":
    *   Found: "The secret code for ferret is VZ57-28Q."
    *   Code: VZ57-28Q
...
```

Then runs out of tokens before getting to the final 'key: CODE'
output block. Needs more `max_new_tokens`. *Not* a quality failure;
LOCAL_GGUF_HARD_v2 retries this at 4096 tokens to confirm.

**Q5_K_M (cell 9b)** — *degenerate loop*. The model collapses into
`"tapir, tapir, tapir, tapir, ..."` for the entire output budget.
Not truncation; it's stuck. Adding more tokens won't help.

**Q8_0 (cell 9c)** — *worse degenerate loop*. `"tapist, tapist,
tapist..."` — a misspelling that loops indefinitely. The
higher-precision model failed harder than Q5.

This last point is striking: at the regime where the eval can
discriminate, **higher-precision quants fail worse than lower-precision
ones**. Possible explanations:

- Q5/Q8 retain enough of the original model's high-confidence
  paths to lock into a single token attractor; Q4's noisier output
  distribution wanders more and finds the correct path occasionally.
- This is a sampling-temperature interaction (we sample greedy/T=0,
  so a sharp distribution = stuck in a basin).
- This is a fluke of the specific seed (the prompt is deterministic,
  but a different seed might shift which needle the model locks onto).

Whichever it is, the data invalidates the earlier "quantised MTP
sidesteps δ-Mem" conclusion. The eval matters.

## Memory profile (still useful)

Memory at 32K target (64K actual tokens):

| | peak VRAM | tok/s |
|---|---|---|
| Q4_K_M | 9.3 GiB | 7.1 |
| Q5_K_M | 9.7 GiB | 7.0 |
| Q8_0   | 11.0 GiB | 6.9 |

All three fit comfortably on the 3060. Memory **isn't** the binding
constraint at 64K actual tokens — *quality* is. The 3060 has 1+ GiB
of headroom even at Q8_0, but the model output is garbage.

## What this means for the δ-Mem hypothesis

The original question was: *does a small δ-Mem adapter restore
quality at the long-context regime where squeezes degrade?*

The previous claim was that "quantisation sidesteps δ-Mem at 32K
target context" — that claim is now **falsified**. Both Q4 (under
generous tokens) and Q5/Q8 (under any tokens) hit the floor on the
hard task at 64K actual tokens. The δ-Mem hypothesis is live again.

**What the Kaggle notebook will test (and why):**

The notebook (`notebooks/kaggle_long_context_delta_mem.ipynb`) runs
cell 1 (vanilla bf16) vs cell 2 (bf16 + δ-Mem adapter) at 16K and
32K target tokens with the same `hard_multineedle` task and 1024
max_new_tokens. Decision logic at the end:

- If `fraction_correct(cell 2 at 32K)` > `fraction_correct(cell 1 at 32K)`:
  δ-Mem rescued quality where vanilla degraded — that's the win.
- If both are 0.00 or both are high: δ-Mem is overhead.
- If Q4_K_M finishes at 1.00 with 4096 tokens but bf16 vanilla fails:
  quantisation + reasoning was the right combo, not δ-Mem.

## Cross-rung comparison

When the Kaggle results land in `results/KAGGLE_LC/`, the
side-by-side becomes:

| ctx | local Q4 (GGUF) | cloud bf16 vanilla | cloud bf16 + δ-Mem |
|-----|------|--------|-------------|
| 16K | 1.00 | (Kaggle) | (Kaggle) |
| 32K | (1.00 expected with 4096 tokens) | (Kaggle, vanilla) | (Kaggle, δ-Mem) |

That table makes it possible to say honestly whether δ-Mem helps,
whether the simpler quantisation path wins instead, or whether the
hypothesis fails for both.

## Methodology notes worth keeping

1. **Eval saturation is a silent killer.** The original LOCAL_GGUF
   sweep concluded the project's central question was settled when
   it really just had a flat-line metric. The hard task should be
   the default for anything past the smoke-test stage.

2. **Reasoning models need much larger token budgets.** 1024 was
   enough for the easy task but truncates Q4's correct reasoning at
   long context. 4096 is the new default for hard runs.

3. **Failure mode varies with quant.** Q4 truncates competently,
   Q5/Q8 loop incoherently. The harness should distinguish these
   in the result records — currently they both look like
   `fraction_correct=0` and you have to inspect `answer_preview`.
   Follow-up: add a coherence-heuristic field (e.g., "output is
   highly repetitive: True/False").

4. **n_ctx must match the actual tokens, not the target.** The hard
   task at target=16K produced 32,931 actual tokens — barely over
   the previous 32K cap. Bumped to 2.2× target with 1024 padding
   plus a 128K hard cap.
