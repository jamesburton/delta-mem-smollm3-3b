#!/usr/bin/env python3
"""Local smoke test for Phase 1 cells (1, 2, 6, 7).

Iterate fast on Windows without the Kaggle round-trip. Designed for the
RTX 3060 12 GB; uses CPU offload via accelerate if the GPU is too small.

Usage:

    python scripts/local_smoke.py                  # all 4 Phase 1 cells
    python scripts/local_smoke.py --cell 1         # one cell only
    python scripts/local_smoke.py --cells 1,2      # selected
    python scripts/local_smoke.py --target-tokens 1000  # smaller NIH context

Env knobs:

    NIH_TARGET_TOKENS       NIH prompt size in words; default 1500
    MAX_NEW_TOKENS          generation cap; default 128
    GPU_MAX_PCT             fraction of GPU to use; default 0.50
    CPU_MAX_GIB             host RAM for accelerate spillover; default 16
    DEVICE                  device strategy ("auto", "cuda", "cpu"); default auto
    DTYPE                   bf16/fp16/fp32; default bfloat16
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from harness import cells as cell_registry
from harness.runners import hf_runner
from harness.runners.hf_runner import RunConfig


PHASE_1 = ("1", "2", "6", "7")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--cell",
        action="append",
        dest="cells",
        default=[],
        help="Cell id to run; repeatable. Defaults to all of Phase 1.",
    )
    p.add_argument(
        "--cells",
        dest="cells_csv",
        default="",
        help="Comma-separated cell ids (alternative to repeating --cell).",
    )
    p.add_argument(
        "--target-tokens",
        type=int,
        default=int(os.environ.get("NIH_TARGET_TOKENS", "1500")),
        help="NIH prompt size in words.",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=int(os.environ.get("MAX_NEW_TOKENS", "128")),
        help="Generation cap.",
    )
    p.add_argument(
        "--dtype",
        default=os.environ.get("DTYPE", "bfloat16"),
        choices=("bfloat16", "float16", "float32"),
    )
    p.add_argument(
        "--device",
        default=os.environ.get("DEVICE", "auto"),
        help='"auto", "cuda", "cuda:0", "cpu".',
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "results" / "LOCAL",
    )
    return p.parse_args()


def _selected_cells(args: argparse.Namespace) -> List[str]:
    chosen: List[str] = []
    if args.cells:
        chosen.extend(args.cells)
    if args.cells_csv:
        chosen.extend([c.strip() for c in args.cells_csv.split(",") if c.strip()])
    if not chosen:
        chosen = list(PHASE_1)
    # Filter to those that exist in the registry
    known = {c.id for c in cell_registry.CELLS}
    return [c for c in chosen if c in known]


def main() -> int:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    cfg = RunConfig(
        target_tokens=args.target_tokens,
        n_needles=3,
        max_new_tokens=args.max_new_tokens,
        seed=0,
        dtype=args.dtype,
        device=args.device,
        results_dir=args.results_dir,
    )

    cells_to_run = _selected_cells(args)
    print(f"Local smoke test — will run {len(cells_to_run)} cell(s): {cells_to_run}")
    print(f"  device={args.device}  dtype={args.dtype}  target_tokens={args.target_tokens}")
    print(f"  results -> {args.results_dir}")
    print()

    by_id = {c.id: c for c in cell_registry.CELLS}
    summary_rows = []
    for cid in cells_to_run:
        if cid not in PHASE_1:
            print(f"  skipping {cid}: not a Phase 1 cell")
            continue
        cell = by_id[cid]
        print(f"running cell {cell.id}: {cell.title}")
        t0 = time.perf_counter()
        try:
            result = hf_runner.run(cell, cfg)
        except Exception as e:
            traceback.print_exc()
            result = {
                "cell_id": cell.id,
                "title": cell.title,
                "status": "failed",
                "error": repr(e),
            }
        elapsed = time.perf_counter() - t0

        out_path = args.results_dir / f"cell-{cell.id}.json"
        out_path.write_text(json.dumps(result, indent=2))

        status = result.get("status", "?")
        q = result.get("quality", {}).get("multineedle", {})
        frac = q.get("fraction")
        s = result.get("speed", {})
        tps = s.get("decode_tokens_per_second", 0.0)
        peak = result.get("memory", {}).get("peak_vram_bytes", 0)
        peak_gb = peak / (1024**3)
        ans = result.get("answer_preview", "")[:80].replace("\n", " ")
        row = (
            f"  -> {cell.id:>3s}  {status:>10s}  "
            f"NIH={frac if isinstance(frac, (int, float)) else 'N/A':<6}  "
            f"tok/s={tps:>5.1f}  peak={peak_gb:.1f}GiB  "
            f"({elapsed:.1f}s)  ans={ans!r}"
        )
        print(row)
        summary_rows.append(row)
        print()

    print("=" * 80)
    print("Summary")
    print("=" * 80)
    for row in summary_rows:
        print(row)
    print()
    print(f"Per-cell JSONs in {args.results_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
