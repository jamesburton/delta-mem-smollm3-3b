"""End-to-end test of the HF runner against the tiny Llama fixture.

This exercises the full pipeline (load → NIH task → answer → score → metrics)
without any GPU or large model dependency.
"""

import json
from pathlib import Path

from harness.runners import hf_runner
from harness.cells import Cell


def test_run_one_cell_writes_complete_record(tmp_path, tiny_model_id):
    cell = Cell(
        id="test-1", title="tiny baseline", base_model=tiny_model_id,
        stack="hf", stages=["S1"], kv_lever="none", speed_lever="none",
    )
    cfg = hf_runner.RunConfig(
        target_tokens=300,
        n_needles=3,
        max_new_tokens=8,
        seed=0,
        dtype="float32",
        device="cpu",
        results_dir=tmp_path,
        ppl_text=None,  # skip ppl in this test
    )
    record = hf_runner.run(cell, cfg)
    # Required keys present
    expected = {"cell_id", "title", "status", "quality", "memory", "speed", "config", "timestamp"}
    assert expected <= record.keys()
    # Quality block has the multi-needle fields
    q = record["quality"]["multineedle"]
    assert "per_needle" in q
    assert "recall_all" in q
    assert "fraction" in q
    # Status is something we can pattern-match on
    assert record["status"] in {"ok", "partial", "failed"}
