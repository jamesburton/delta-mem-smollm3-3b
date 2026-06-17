"""Multi-needle NIH evaluation for LLaDA-8B-Instruct on RTX 3060.

Uses the existing NIH machinery from ``harness.metrics.quality`` (so the
needles, distractors, and grader are the same as the AR cells) plus the
diffusion runner in ``harness.runners.diffusion_runner`` (which knows how
to talk to LLaDA's block-diffusion sampler and reports tok/s as
``emitted_tokens / wall``).

Why we need a dedicated diffusion runner: the AR ``hf_runner`` uses a
forward_pre_hook to split call #0 (= prefill) from calls #N>=1 (= decode).
Block diffusion has no prefill/decode split — every step is a single
full-sequence forward — so the AR helper's memory-split math would silently
produce nonsense numbers. See ``harness/runners/diffusion_runner.py``
module docstring for the full rationale.

LLaDA-specific gotchas baked in here
------------------------------------

1. **Context ceiling.** LLaDA's config reports ``max_sequence_length=4096``.
   At ``ctx=4000`` plus ``gen_length=256`` we end up running 16 forwards over
   a 4256-token sequence — *already* past the trained ceiling on the inner
   block. RoPE-extrapolation may carry it but recall will likely drop;
   ``ctx=2000`` is the safer first target. Don't go above ctx=3500 without
   measuring.

2. **No KV-cache → O(L²) per step.** Each of the 16 steps is a full
   re-encode of the entire sequence. Expect per-step time of roughly
   ``ms_per_step(50tok) * (L / 50)^2`` once attention dominates.
   On the chat smoke test at ~50 tokens, per-step ≈ 290 ms (LLMs.md). At
   ctx=2000 that's ~30 s/run; at ctx=4000 ~120 s/run.

3. **FA2 is a long-ctx ENABLER for LLaDA too.** LLaDA carries its own
   ``flash_attention`` config flag (separate from HF's
   ``attn_implementation``). The diffusion runner enables it by default on
   sm_80+ hardware where the ``flash_attn`` package is importable. On the
   3060 (sm_86) with FA2 built (see CUDA_NOTES.md → "Confirmed working
   source-build recipe") this should mirror the Qwen3-4B long-ctx FA2
   finding: without FA2, ctx ≥ 4K pages catastrophically through WDDM
   shared memory or OOMs at ctx=8K.

4. **bnb-4bit by default.** ~5.3 GB VRAM at load, leaves ~6.5 GB headroom
   on a 12 GB card for the attention working set. bf16 is ~15 GB and
   doesn't fit on T1.

Runtime estimate (T1, RTX 3060, bnb-4bit NF4 + FA2, steps=16, block=128):

    ctx=2000, gen=256   ≈  90 s wall   tok/s ≈ 3
    ctx=4000, gen=256   ≈ 300 s wall   tok/s ≈ 0.85
    ctx=8000, gen=256   ≈ OOM expected unless FA2 holds; if it survives, ~20 min

Usage
-----

Defaults (single cell, ctx=2000, multi-needle NIH):

    C:\\Python311\\python.exe scripts\\nih_llada_8b.py

Pick a context size and override the gen-length / steps knobs:

    C:\\Python311\\python.exe scripts\\nih_llada_8b.py --ctx 4000 --steps 16 --max-new-tokens 256

Run the harder RULER-style task (10 needles + 30 distractors + mapping check):

    C:\\Python311\\python.exe scripts\\nih_llada_8b.py --ctx 2000 --task-type hard_multineedle

Sweep contexts (results land in ``results/LLADA8B_NIH/ctx-<N>/cell-D1.json``):

    C:\\Python311\\python.exe scripts\\nih_llada_8b.py --ctx 2000,4000

The output JSON has the same shape as ``hf_runner.run`` so
``scripts/context_sweep.render_sweep_summary`` will render it (the speed
section additionally carries ``diffusion_steps``, ``diffusion_block_length``,
``diffusion_ms_per_step`` fields).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness.cells import Cell
from harness.runners import diffusion_runner
from harness.runners.diffusion_runner import RunConfig


# A synthetic Cell for the diffusion run. We don't add it to the global
# CELLS registry (which is locked to the AR test matrix) — we just need a
# Cell-shaped object so the runner can stamp its metadata into the result
# JSON. The id "D1" (D for diffusion) keeps it from colliding with the
# numeric ids in the AR matrix.
DIFFUSION_CELL = Cell(
    id="D1",
    title="LLaDA-8B-Instruct (block-diffusion) NIH",
    base_model="GSAI-ML/LLaDA-8B-Instruct",
    stack="hf",
    stages=["S3"],
    kv_lever="none-no-kv-cache",
    speed_lever="block-diffusion",
    notes="Diffusion runner; tok/s = emitted_tokens / wall.",
)


def _parse_contexts(s: str) -> list[int]:
    return [int(c.strip()) for c in s.split(",") if c.strip()]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model-id", default="GSAI-ML/LLaDA-8B-Instruct")
    p.add_argument("--ctx", default="2000",
                   help="NIH target context size (comma-separated to sweep). "
                        "LLaDA's trained max_sequence_length is 4096 — values "
                        "above ~3500 risk RoPE extrapolation hurting recall.")
    p.add_argument("--max-new-tokens", type=int, default=256,
                   help="Target generation length (will be snapped UP to a "
                        "multiple of --block-length).")
    p.add_argument("--steps", type=int, default=16,
                   help="Diffusion steps. LLMs.md best config: steps=16.")
    p.add_argument("--block-length", type=int, default=128,
                   help="Block length for the semi-AR sampler. LLMs.md best "
                        "config: 128.")
    p.add_argument("--quant", default="bnb-4bit", choices=["bnb-4bit", "bf16"],
                   help="bnb-4bit (default, ~5.3 GB) or bf16 (~15 GB; won't "
                        "fit T1).")
    p.add_argument("--flash-attention", default="auto",
                   choices=["auto", "on", "off"],
                   help="Toggle LLaDA's internal flash_attention. 'auto' "
                        "enables on sm_80+ with flash_attn installed.")
    p.add_argument("--task-type", default="multineedle",
                   choices=("multineedle", "hard_multineedle"),
                   help="Eval task. 'hard_multineedle' = 10 needles + "
                        "distractors + mapping check.")
    p.add_argument("--n-needles", type=int, default=3,
                   help="Needle count. Auto-bumps to 10 if --task-type=hard.")
    p.add_argument("--n-distractors", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results-root", type=Path,
                   default=REPO_ROOT / "results" / "LLADA8B_NIH")
    p.add_argument("--temperature", type=float, default=0.0)
    args = p.parse_args()

    fa_map = {"auto": None, "on": True, "off": False}
    fa = fa_map[args.flash_attention]
    contexts = _parse_contexts(args.ctx)

    args.results_root.mkdir(parents=True, exist_ok=True)
    all_records = []

    for ctx in contexts:
        if ctx > 3500:
            print(f"\n!! ctx={ctx} > 3500 — LLaDA was trained at "
                  f"max_sequence_length=4096; recall may degrade.\n")
        ctx_dir = args.results_root / f"ctx-{ctx}"
        ctx_dir.mkdir(parents=True, exist_ok=True)
        out_path = ctx_dir / f"cell-{DIFFUSION_CELL.id}.json"

        # Reuse the AR RunConfig so the AR / diffusion runners are
        # call-compatible — same eval shape, same scorers.
        rc = RunConfig(
            target_tokens=ctx,
            n_needles=args.n_needles,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
            dtype="bfloat16",   # diffusion runner uses bnb compute dtype directly
            device="auto",
            results_dir=ctx_dir,
            task_type=args.task_type,
            n_distractors=args.n_distractors,
        )

        print(f"\n=== ctx={ctx}  steps={args.steps}  "
              f"block={args.block_length}  task={args.task_type} ===")
        wall_t0 = time.perf_counter()
        try:
            rec = diffusion_runner.run_llada(
                DIFFUSION_CELL, rc,
                model_id=args.model_id,
                steps=args.steps,
                block_length=args.block_length,
                quant=args.quant,
                enable_flash_attention=fa,
                temperature=args.temperature,
            )
        except Exception as e:
            traceback.print_exc()
            rec = {
                "cell_id": DIFFUSION_CELL.id,
                "title": DIFFUSION_CELL.title,
                "status": "failed",
                "error": repr(e),
            }
        rec["context_tokens"] = ctx
        rec["wall_clock_seconds"] = time.perf_counter() - wall_t0
        out_path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        all_records.append(rec)

        # Pretty one-liner summary
        q = rec.get("quality", {})
        s = rec.get("speed", {})
        m = rec.get("memory", {})
        frac = (q.get("multineedle", {}).get("fraction")
                or q.get("hard_multineedle", {}).get("fraction_correct"))
        frac_s = f"{frac:.2f}" if isinstance(frac, (int, float)) else "N/A"
        tps = s.get("decode_tokens_per_second", 0) or 0
        peak_gb = (m.get("peak_vram_bytes", 0) or 0) / (1024**3)
        wall = rec.get("wall_clock_seconds", 0)
        print(f"  → ctx={ctx} {rec.get('status', '?')}  NIH={frac_s}  "
              f"tok/s={tps:.2f}  peak={peak_gb:.2f}GiB  wall={wall:.1f}s")
        print(f"  → wrote {out_path}")

    # Aggregate summary file
    summary_path = args.results_root / "summary.json"
    summary_path.write_text(json.dumps(all_records, indent=2), encoding="utf-8")
    print(f"\nSaved aggregate summary → {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
