# Context sweep results

## Cell 1 — Qwen3-4B vanilla full attention

| Context | Status | NIH frac | Peak VRAM | KV @ ctx | decode tok/s | TTFT (s) | Wall (s) |
|---|---|---|---|---|---|---|---|
| 4000 | ok | 1.00 | 9.2 GB | 351.6 MB | 3.9 | 0.291 | 53.4 |
| 8000 | ok | 1.00 | 10.9 GB | 703.1 MB | 3.1 | 0.255 | 74.3 |

## Cell 2 — Qwen3-4B + δ-Mem adapter

| Context | Status | NIH frac | Peak VRAM | KV @ ctx | decode tok/s | TTFT (s) | Wall (s) |
|---|---|---|---|---|---|---|---|
| 4000 | ok | 1.00 | 11.0 GB | 351.6 MB | 2.0 | 0.672 | 366.3 |
| 8000 | failed | N/A | 0.0 B | 0.0 B | 0.0 | 0.000 | 1120.4 |

## Cell 3 — Qwen3-4B + sliding-window 4K

| Context | Status | NIH frac | Peak VRAM | KV @ ctx | decode tok/s | TTFT (s) | Wall (s) |
|---|---|---|---|---|---|---|---|
| 4000 | partial | 0.00 | 9.2 GB | 351.6 MB | 3.9 | 0.264 | 60.1 |
| 8000 | partial | 0.00 | 10.9 GB | 703.1 MB | 3.0 | 0.398 | 114.3 |

## Cell 4 — Qwen3-4B + SW-4K + δ-Mem

| Context | Status | NIH frac | Peak VRAM | KV @ ctx | decode tok/s | TTFT (s) | Wall (s) |
|---|---|---|---|---|---|---|---|
| 4000 | ok | 0.33 | 11.0 GB | 351.6 MB | 2.0 | 0.676 | 372.8 |
| 8000 | partial | 0.00 | 14.4 GB | 703.1 MB | 2.0 | 1.040 | 696.6 |
