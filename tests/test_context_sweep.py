"""Context-sweep helper unit tests."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts import context_sweep


def test_render_sweep_summary_groups_by_cell_then_context():
    fake_results = [
        {"cell_id": "1", "title": "vanilla", "context_tokens": 1000, "status": "ok",
         "quality": {"multineedle": {"fraction": 1.0}},
         "memory": {"peak_vram_bytes": 5*1024**3, "kv_cache_bytes_at_target_len": 100*1024*1024},
         "speed": {"decode_tokens_per_second": 10.0, "ttft_seconds": 0.1},
         "wall_clock_seconds": 5.0},
        {"cell_id": "1", "title": "vanilla", "context_tokens": 2000, "status": "ok",
         "quality": {"multineedle": {"fraction": 1.0}},
         "memory": {"peak_vram_bytes": 6*1024**3, "kv_cache_bytes_at_target_len": 200*1024*1024},
         "speed": {"decode_tokens_per_second": 9.0, "ttft_seconds": 0.2},
         "wall_clock_seconds": 8.0},
        {"cell_id": "2", "title": "delta-mem", "context_tokens": 1000, "status": "ok",
         "quality": {"multineedle": {"fraction": 0.66}},
         "memory": {"peak_vram_bytes": 8*1024**3, "kv_cache_bytes_at_target_len": 100*1024*1024},
         "speed": {"decode_tokens_per_second": 7.0, "ttft_seconds": 0.3},
         "wall_clock_seconds": 9.0},
    ]
    md = context_sweep.render_sweep_summary(fake_results)
    assert "Cell 1" in md
    assert "Cell 2" in md
    # Cell 1's two rows should both appear
    assert md.count("| 1000 |") >= 1
    assert md.count("| 2000 |") >= 1


def test_natural_sort_orders_cells_correctly():
    assert context_sweep._natural_cell_sort("1") < context_sweep._natural_cell_sort("2")
    assert context_sweep._natural_cell_sort("2") < context_sweep._natural_cell_sort("10")
    assert context_sweep._natural_cell_sort("9a") < context_sweep._natural_cell_sort("10")
    assert context_sweep._natural_cell_sort("9") < context_sweep._natural_cell_sort("9a")
    assert context_sweep._natural_cell_sort("16") < context_sweep._natural_cell_sort("T1")
