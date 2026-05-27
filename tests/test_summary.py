import json
from pathlib import Path
from harness import summary


def test_render_summary_includes_phase1_cells(tmp_path):
    (tmp_path / "S1").mkdir()
    for cid, status, frac in [("1","ok",0.33), ("2","ok",1.0), ("6","ok",0.66), ("7","partial",0.0)]:
        (tmp_path / "S1" / f"cell-{cid}.json").write_text(json.dumps({
            "cell_id": cid, "title": f"cell {cid}", "status": status,
            "quality": {"multineedle": {"fraction": frac}},
            "memory": {"peak_vram_bytes": 1_000_000_000, "kv_cache_bytes_at_target_len": 100_000_000},
            "speed": {"decode_tokens_per_second": 25.0, "ttft_seconds": 0.2},
        }))
    md = summary.render(tmp_path, stage="S1")
    for cid in ("1","2","6","7"):
        assert f"| {cid} |" in md
    assert "decode tok/s" in md.lower()
