# Context sweep results

## Cell 3 — Qwen3-4B + sliding-window 4K

| Context | Status | NIH frac | Peak VRAM | KV @ ctx | decode tok/s | TTFT (s) | Wall (s) |
|---|---|---|---|---|---|---|---|
| 2000 | ok | 1.00 | 8.4 GB | 175.8 MB | 4.0 | 0.249 | 48.4 |
| 4000 | ok | 1.00 | 9.2 GB | 351.6 MB | 3.8 | 0.256 | 53.2 |
| 8000 | ok | 1.00 | 10.9 GB | 703.1 MB | 3.2 | 0.503 | 74.0 |

## Cell 4 — Qwen3-4B + SW-4K + δ-Mem

| Context | Status | NIH frac | Peak VRAM | KV @ ctx | decode tok/s | TTFT (s) | Wall (s) |
|---|---|---|---|---|---|---|---|
| 2000 | ok | 1.00 | 9.3 GB | 175.8 MB | 2.1 | 1.080 | 224.8 |
| 4000 | ok | 1.00 | 11.0 GB | 351.6 MB | 2.1 | 0.661 | 367.4 |
| 8000 | ok | 1.00 | 14.4 GB | 703.1 MB | 1.9 | 1.292 | 1301.7 |

## Cell 5 — Qwen3-4B + SW-2K + δ-Mem

| Context | Status | NIH frac | Peak VRAM | KV @ ctx | decode tok/s | TTFT (s) | Wall (s) |
|---|---|---|---|---|---|---|---|
| 2000 | ok | 1.00 | 9.3 GB | 175.8 MB | 2.1 | 0.616 | 217.7 |
| 4000 | ok | 1.00 | 11.0 GB | 351.6 MB | 2.1 | 0.602 | 367.5 |
| 8000 | failed | N/A | 0.0 B | 0.0 B | 0.0 | 0.000 | 43.3 |
