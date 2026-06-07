# Context sweep results

## Cell 1 — Qwen3-4B vanilla full attention

| Context | Status | NIH frac | Peak VRAM | KV @ ctx | decode tok/s | TTFT (s) | Wall (s) |
|---|---|---|---|---|---|---|---|
| 4000 | ok | 1.00 | 9.2 GB | 351.6 MB | 3.4 | 0.301 | 59.2 |
| 8000 | ok | 1.00 | 10.9 GB | 703.1 MB | 3.3 | 0.298 | 66.3 |

## Cell 2 — Qwen3-4B + δ-Mem adapter

| Context | Status | NIH frac | Peak VRAM | KV @ ctx | decode tok/s | TTFT (s) | Wall (s) |
|---|---|---|---|---|---|---|---|
| 4000 | ok | 1.00 | 11.0 GB | 351.6 MB | 1.8 | 0.763 | 392.8 |
| 8000 | ok | 1.00 | 14.4 GB | 703.1 MB | 1.5 | 0.910 | 1123.9 |
