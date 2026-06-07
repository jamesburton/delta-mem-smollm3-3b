# Context sweep results

## Cell 1 — Qwen3-4B vanilla full attention

| Context | Status | NIH frac | Peak VRAM | KV @ ctx | decode tok/s | TTFT (s) | Wall (s) |
|---|---|---|---|---|---|---|---|
| 2000 | ok | 1.00 | 8.4 GB | 175.8 MB | 4.2 | 0.279 | 115.9 |
| 4000 | ok | 1.00 | 9.2 GB | 351.6 MB | 3.9 | 0.264 | 40.6 |
| 8000 | ok | 1.00 | 10.9 GB | 703.1 MB | 3.7 | 0.252 | 60.1 |
| 16000 | failed | N/A | 0.0 B | 0.0 B | 0.0 | 0.000 | 2.7 |

## Cell 2 — Qwen3-4B + δ-Mem adapter

| Context | Status | NIH frac | Peak VRAM | KV @ ctx | decode tok/s | TTFT (s) | Wall (s) |
|---|---|---|---|---|---|---|---|
| 2000 | ok | 1.00 | 9.3 GB | 175.8 MB | 2.2 | 0.612 | 205.3 |
| 4000 | ok | 1.00 | 11.0 GB | 351.6 MB | 2.2 | 0.614 | 346.1 |
| 8000 | ok | 1.00 | 14.4 GB | 703.1 MB | 0.4 | 2.340 | 1992.1 |
| 16000 | failed | N/A | 0.0 B | 0.0 B | 0.0 | 0.000 | 2.1 |

## Cell 6 — Qwen3-4B + spec-decode

| Context | Status | NIH frac | Peak VRAM | KV @ ctx | decode tok/s | TTFT (s) | Wall (s) |
|---|---|---|---|---|---|---|---|
| 2000 | ok | 1.00 | 9.9 GB | 175.8 MB | 2.8 | 0.232 | 59.5 |
| 4000 | ok | 1.00 | 11.2 GB | 351.6 MB | 3.0 | 0.245 | 542.9 |
| 8000 | failed | N/A | 0.0 B | 0.0 B | 0.0 | 0.000 | 620.7 |
| 16000 | failed | N/A | 0.0 B | 0.0 B | 0.0 | 0.000 | 2.4 |

## Cell 7 — Qwen3-4B + δ-Mem + spec-decode

| Context | Status | NIH frac | Peak VRAM | KV @ ctx | decode tok/s | TTFT (s) | Wall (s) |
|---|---|---|---|---|---|---|---|
| 2000 | ok | 1.00 | 9.3 GB | 175.8 MB | 2.2 | 0.614 | 204.4 |
| 4000 | ok | 1.00 | 11.0 GB | 351.6 MB | 2.3 | 0.581 | 341.2 |
| 8000 | failed | N/A | 0.0 B | 0.0 B | 0.0 | 0.000 | 1.9 |
| 16000 | failed | N/A | 0.0 B | 0.0 B | 0.0 | 0.000 | 1.7 |
