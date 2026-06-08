# Context sweep results

## Cell 9a — Qwen3.5-4B-MTP-GGUF Q4_K_M

| Context | Status | NIH frac | Peak VRAM | KV @ ctx | decode tok/s | TTFT (s) | Wall (s) |
|---|---|---|---|---|---|---|---|
| 4000 | ok | 1.00 | 4.6 GB | 0.0 B | 4.4 | n/a | 36.9 |
| 8000 | ok | 1.00 | 5.1 GB | 0.0 B | 2.6 | n/a | 43.7 |
| 16000 | ok | 1.00 | 6.2 GB | 0.0 B | 1.2 | n/a | 63.4 |

## Cell 9b — Qwen3.5-4B-MTP-GGUF Q5_K_M

| Context | Status | NIH frac | Peak VRAM | KV @ ctx | decode tok/s | TTFT (s) | Wall (s) |
|---|---|---|---|---|---|---|---|
| 4000 | partial | 0.00 | 4.9 GB | 0.0 B | 7.9 | n/a | 37.7 |
| 8000 | partial | 0.00 | 5.4 GB | 0.0 B | 4.3 | n/a | 43.8 |
| 16000 | partial | 0.00 | 6.6 GB | 0.0 B | 2.0 | n/a | 64.3 |

## Cell 9c — Qwen3.5-4B-MTP-GGUF Q8_0

| Context | Status | NIH frac | Peak VRAM | KV @ ctx | decode tok/s | TTFT (s) | Wall (s) |
|---|---|---|---|---|---|---|---|
| 4000 | partial | 0.00 | 6.2 GB | 0.0 B | 7.7 | n/a | 38.3 |
| 8000 | partial | 0.00 | 6.7 GB | 0.0 B | 4.3 | n/a | 46.8 |
| 16000 | partial | 0.00 | 7.9 GB | 0.0 B | 2.0 | n/a | 67.9 |
