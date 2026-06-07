#!/usr/bin/env python3
"""Context sweep for Phase 1 cells (1, 2, 6, 7).

Runs each cell at multiple NIH context sizes to characterise scaling:
- vanilla (cell 1) — peak VRAM and KV grow linearly with context
- +δ-Mem (cell 2) — KV stays small but δ-Mem write-phase adds time
- +spec-decode (cell 6) — speed lever
- compound (cell 7) — full Phase-1 stack

Usage (CLI):

    python scripts/context_sweep.py
    python scripts/context_sweep.py --contexts 2000,4000,8000
    python scripts/context_sweep.py --cells 1,2 --contexts 1000 --max-new-tokens 64
    STAGE=S1 python scripts/context_sweep.py --device cuda

Usage (notebook):

    from scripts.context_sweep import run_sweep, render_sweep_summary
    results = run_sweep(contexts=[2000, 4000, 8000], cell_ids=["1","2"])
    print(render_sweep_summary(results))
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness import cells as cell_registry
from harness.runners import hf_runner
from harness.runners.hf_runner import RunConfig


DEFAULT_CONTEXTS = (2000, 4000, 8000, 16000)
DEFAULT_CELL_IDS = ("1", "2", "6", "7")


def _run_combo_inproc(cell, cfg, ctx: int) -> dict:
    """Run one (cell, ctx) combo in-process. Returns the result dict."""
    t0 = time.perf_counter()
    try:
        rec = hf_runner.run(cell, cfg)
    except Exception as e:
        traceback.print_exc()
        rec = {
            "cell_id": cell.id,
            "title": cell.title,
            "status": "failed",
            "error": repr(e),
        }
    rec["context_tokens"] = ctx
    rec["wall_clock_seconds"] = time.perf_counter() - t0
    out_path = cfg.results_dir / f"cell-{cell.id}.json"
    out_path.write_text(json.dumps(rec, indent=2))
    return rec


def _run_combo_subproc(
    cell, ctx: int, ctx_dir: Path, *,
    max_new_tokens: int, dtype: str, device: str, seed: int,
) -> dict:
    """Run one (cell, ctx) combo in a fresh python subprocess.

    Isolates GPU/CUDA state: an OOM or CUBLAS wedge kills only the worker, and
    the next combo starts with a clean CUDA context. The subprocess writes the
    same per-cell JSON the in-process path would; we read it back to assemble
    the summary. On non-zero exit, synthesise a `failed` record with stderr.
    """
    out_path = ctx_dir / f"cell-{cell.id}.json"
    if out_path.exists():
        out_path.unlink()  # don't read a stale record from a previous run
    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--_one_combo", f"{cell.id}:{ctx}",
        "--results-root", str(ctx_dir.parent.parent),
        "--stage", ctx_dir.parent.name,
        "--max-new-tokens", str(max_new_tokens),
        "--dtype", dtype,
    ]
    if device:
        cmd += ["--device", device]
    t0 = time.perf_counter()
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    proc = subprocess.run(cmd, env=env)
    wall = time.perf_counter() - t0
    if out_path.exists():
        rec = json.loads(out_path.read_text(encoding="utf-8"))
        # Worker writes its own wall_clock; keep the outer measurement too
        # for orchestrator-level timing including subprocess startup.
        rec.setdefault("wall_clock_seconds", wall)
        return rec
    # Subprocess died before writing — synthesise a failed record
    rec = {
        "cell_id": cell.id,
        "title": cell.title,
        "status": "failed",
        "error": f"subprocess exited {proc.returncode} without writing result",
        "context_tokens": ctx,
        "wall_clock_seconds": wall,
    }
    out_path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return rec


def run_sweep(
    *,
    contexts=DEFAULT_CONTEXTS,
    cell_ids=DEFAULT_CELL_IDS,
    max_new_tokens: int = 256,
    dtype: str = "bfloat16",
    device: str | None = None,
    results_root: Path = REPO_ROOT / "results",
    stage: str = "S3",
    seed: int = 0,
    isolate: bool = True,
) -> List[dict]:
    """Run the sweep. Returns list of result dicts (one per combo).

    With isolate=True (default), each (cell, ctx) runs in a fresh subprocess so
    a CUDA wedge or OOM in one combo can't poison the rest of the sweep.
    """
    import torch
    if device is None:
        device = "auto" if torch.cuda.is_available() else "cpu"
    by_id = {c.id: c for c in cell_registry.CELLS}
    results: List[dict] = []
    for ctx in contexts:
        ctx_dir = results_root / stage / f"ctx-{ctx}"
        ctx_dir.mkdir(parents=True, exist_ok=True)
        for cid in cell_ids:
            if cid not in by_id:
                print(f"⏸ unknown cell id: {cid}")
                continue
            cell = by_id[cid]
            print(f"▶ ctx={ctx} cell={cell.id}: {cell.title}"
                  f"{'  (subprocess)' if isolate else ''}")
            if isolate:
                rec = _run_combo_subproc(
                    cell, ctx, ctx_dir,
                    max_new_tokens=max_new_tokens, dtype=dtype,
                    device=device, seed=seed,
                )
            else:
                cfg = RunConfig(
                    target_tokens=ctx,
                    n_needles=3,
                    max_new_tokens=max_new_tokens,
                    seed=seed,
                    dtype=dtype,
                    device=device,
                    results_dir=ctx_dir,
                )
                rec = _run_combo_inproc(cell, cfg, ctx)
            status = rec.get("status", "?")
            q = rec.get("quality", {}).get("multineedle", {})
            frac = q.get("fraction")
            s = rec.get("speed", {})
            peak_gb = rec.get("memory", {}).get("peak_vram_bytes", 0) / (1024**3)
            print(
                f"  → ctx={ctx} cell={cell.id}  {status}  "
                f"NIH={frac if isinstance(frac, (int, float)) else 'N/A':<5}  "
                f"peak={peak_gb:.1f}GiB  "
                f"tok/s={s.get('decode_tokens_per_second', 0):.1f}  "
                f"TTFT={s.get('ttft_seconds', 0):.3f}s  "
                f"({rec['wall_clock_seconds']:.1f}s wall)"
            )
            results.append(rec)
    return results


def render_sweep_summary(results: List[dict]) -> str:
    """One table per cell, rows = contexts, columns = metrics."""
    # Group by cell_id
    by_cell: Dict[str, List[dict]] = {}
    for r in results:
        by_cell.setdefault(r["cell_id"], []).append(r)
    out_lines = ["# Context sweep results", ""]
    for cid in sorted(by_cell, key=_natural_cell_sort):
        rows = sorted(by_cell[cid], key=lambda r: r.get("context_tokens", 0))
        if not rows:
            continue
        title = rows[0].get("title", cid)
        out_lines.append(f"## Cell {cid} — {title}")
        out_lines.append("")
        out_lines.append("| Context | Status | NIH frac | Peak VRAM | KV @ ctx | decode tok/s | TTFT (s) | Wall (s) |")
        out_lines.append("|---|---|---|---|---|---|---|---|")
        for r in rows:
            q = r.get("quality", {}).get("multineedle", {})
            m = r.get("memory", {})
            s = r.get("speed", {})
            frac = q.get("fraction")
            frac_cell = f"{frac:.2f}" if isinstance(frac, (int, float)) else "N/A"
            out_lines.append(
                f"| {r.get('context_tokens', '?')} "
                f"| {r.get('status', '?')} "
                f"| {frac_cell} "
                f"| {_fmt_bytes(m.get('peak_vram_bytes', 0))} "
                f"| {_fmt_bytes(m.get('kv_cache_bytes_at_target_len', 0))} "
                f"| {s.get('decode_tokens_per_second', 0):.1f} "
                f"| {s.get('ttft_seconds', 0):.3f} "
                f"| {r.get('wall_clock_seconds', 0):.1f} |"
            )
        out_lines.append("")
    return "\n".join(out_lines)


def _natural_cell_sort(cid):
    import re
    m = re.match(r"^(\d+)([a-z]*)$", cid)
    if m:
        return (0, int(m.group(1)), m.group(2))
    return (1, cid)


def _fmt_bytes(n):
    val = float(n or 0)
    for unit in ("B", "KB", "MB", "GB"):
        if val < 1024:
            return f"{val:.1f} {unit}"
        val /= 1024
    return f"{val:.1f} TB"


def _worker_one_combo(spec: str, *,
                      results_root: Path, stage: str,
                      max_new_tokens: int, dtype: str,
                      device: Optional[str]) -> int:
    """Worker mode: run exactly one (cell, ctx) combo and exit.

    Used by `_run_combo_subproc` to isolate GPU state per combo. Writes the
    per-cell JSON to `{results_root}/{stage}/ctx-{ctx}/cell-{cell_id}.json`.
    """
    cell_id, ctx_str = spec.split(":", 1)
    ctx = int(ctx_str)
    by_id = {c.id: c for c in cell_registry.CELLS}
    if cell_id not in by_id:
        print(f"  unknown cell id: {cell_id}", file=sys.stderr)
        return 2
    cell = by_id[cell_id]
    ctx_dir = results_root / stage / f"ctx-{ctx}"
    ctx_dir.mkdir(parents=True, exist_ok=True)
    import torch
    eff_device = device or ("auto" if torch.cuda.is_available() else "cpu")
    cfg = RunConfig(
        target_tokens=ctx,
        n_needles=3,
        max_new_tokens=max_new_tokens,
        seed=0,
        dtype=dtype,
        device=eff_device,
        results_dir=ctx_dir,
    )
    rec = _run_combo_inproc(cell, cfg, ctx)
    return 0 if rec.get("status") == "ok" else 1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--contexts", default=",".join(str(c) for c in DEFAULT_CONTEXTS),
                   help="Comma-separated NIH context sizes")
    p.add_argument("--cells", default=",".join(DEFAULT_CELL_IDS),
                   help="Comma-separated cell ids")
    p.add_argument("--max-new-tokens", type=int, default=int(os.environ.get("MAX_NEW_TOKENS", "256")))
    p.add_argument("--dtype", default=os.environ.get("DTYPE", "bfloat16"))
    p.add_argument("--device", default=os.environ.get("DEVICE"))
    p.add_argument("--results-root", type=Path, default=REPO_ROOT / "results")
    p.add_argument("--stage", default=os.environ.get("STAGE", "S3"))
    p.add_argument("--no-isolate", action="store_true",
                   help="Run all combos in-process (faster startup, but one "
                        "OOM/CUDA wedge poisons the rest of the sweep).")
    p.add_argument("--_one_combo", default=None,
                   help=argparse.SUPPRESS)  # internal worker mode
    args = p.parse_args()

    if args._one_combo:
        return _worker_one_combo(
            args._one_combo,
            results_root=args.results_root, stage=args.stage,
            max_new_tokens=args.max_new_tokens, dtype=args.dtype,
            device=args.device,
        )

    contexts = [int(c.strip()) for c in args.contexts.split(",") if c.strip()]
    cell_ids = [c.strip() for c in args.cells.split(",") if c.strip()]
    results = run_sweep(
        contexts=contexts,
        cell_ids=cell_ids,
        max_new_tokens=args.max_new_tokens,
        dtype=args.dtype,
        device=args.device,
        results_root=args.results_root,
        stage=args.stage,
        isolate=not args.no_isolate,
    )
    md = render_sweep_summary(results)
    sweep_path = args.results_root / args.stage / "context_sweep.md"
    sweep_path.write_text(md, encoding="utf-8")
    print(f"\nSaved {sweep_path}")
    try:
        print()
        print(md)
    except UnicodeEncodeError:
        # Windows console can't always render Unicode (δ etc.); the file
        # write above already succeeded.
        print("(skipped stdout render: console encoding can't handle Unicode chars)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
