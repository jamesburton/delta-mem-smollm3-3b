"""Aggregate per-cell JSONs into a Markdown summary.

Cell ids are natural-sorted (1, 2, ..., 9a, 9b, 9c, 10, ..., 16, T1) so the
table reads in the same order as the test matrix.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable


def _cell_sort_key(path: Path):
    """Natural-sort key for `cell-<id>.json` filenames.

    Numeric cells (with optional letter suffix like 9a) sort first by
    int(prefix) then by suffix; non-numeric ids (T1) sort after, alphabetical.
    """
    cid = path.stem.removeprefix("cell-")
    m = re.match(r"^(\d+)([a-z]*)$", cid)
    if m:
        return (0, int(m.group(1)), m.group(2))
    return (1, cid)


def _fmt_bytes(n: int) -> str:
    val = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if val < 1024:
            return f"{val:.1f} {unit}"
        val /= 1024
    return f"{val:.1f} TB"


def _row(d: dict) -> str:
    q = d.get("quality", {}).get("multineedle", {})
    m = d.get("memory", {})
    s = d.get("speed", {})
    frac = q.get("fraction")
    frac_cell = f"{frac:.2f}" if isinstance(frac, (int, float)) else "N/A"
    return (
        f"| {d.get('cell_id','?')} "
        f"| {d.get('title','?')[:48]} "
        f"| {d.get('status','?')} "
        f"| {frac_cell} "
        f"| {_fmt_bytes(m.get('peak_vram_bytes',0))} "
        f"| {_fmt_bytes(m.get('kv_cache_bytes_at_target_len',0))} "
        f"| {s.get('decode_tokens_per_second', 0):.1f} "
        f"| {s.get('ttft_seconds', 0):.3f} |"
    )


def render(results_root: Path, *, stage: str) -> str:
    """Build summary.md content from `results_root/{stage}/cell-*.json`."""
    stage_dir = Path(results_root) / stage
    rows = []
    for path in sorted(stage_dir.glob("cell-*.json"), key=_cell_sort_key):
        try:
            rows.append(_row(json.loads(path.read_text())))
        except json.JSONDecodeError:
            continue
    header = (
        f"# Results — {stage}\n\n"
        "| Cell | Title | Status | NIH frac | Peak VRAM | KV bytes | "
        "decode tok/s | TTFT (s) |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    return header + "\n".join(rows) + "\n"
