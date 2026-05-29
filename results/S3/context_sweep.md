# Context sweep results

## Cell 1 — Qwen3-4B vanilla full attention

| Context | Status | NIH frac | Peak VRAM | KV @ ctx | decode tok/s | TTFT (s) | Wall (s) |
|---|---|---|---|---|---|---|---|
| 2000 | ok | 1.00 | 8.9 GB | 175.8 MB | 11.6 | 0.235 | 68.4 |
| 4000 | failed | N/A | 0.0 B | 0.0 B | 0.0 | 0.000 | 5.8 |
| 8000 | failed | N/A | 0.0 B | 0.0 B | 0.0 | 0.000 | 5.4 |
| 16000 | failed | N/A | 0.0 B | 0.0 B | 0.0 | 0.000 | 5.9 |

## Cell 2 — Qwen3-4B + δ-Mem adapter

| Context | Status | NIH frac | Peak VRAM | KV @ ctx | decode tok/s | TTFT (s) | Wall (s) |
|---|---|---|---|---|---|---|---|
| 2000 | ok | 1.00 | 13.1 GB | 175.8 MB | 8.7 | 0.379 | 41.4 |
| 4000 | failed | N/A | 0.0 B | 0.0 B | 0.0 | 0.000 | 5.0 |
| 8000 | failed | N/A | 0.0 B | 0.0 B | 0.0 | 0.000 | 5.1 |
| 16000 | failed | N/A | 0.0 B | 0.0 B | 0.0 | 0.000 | 5.5 |

## Cell 6 — Qwen3-4B + spec-decode

| Context | Status | NIH frac | Peak VRAM | KV @ ctx | decode tok/s | TTFT (s) | Wall (s) |
|---|---|---|---|---|---|---|---|
| 2000 | ok | 1.00 | 8.9 GB | 175.8 MB | 9.1 | 0.214 | 44.6 |
| 4000 | failed | N/A | 0.0 B | 0.0 B | 0.0 | 0.000 | 18.7 |
| 8000 | failed | N/A | 0.0 B | 0.0 B | 0.0 | 0.000 | 6.4 |
| 16000 | failed | N/A | 0.0 B | 0.0 B | 0.0 | 0.000 | 6.7 |

## Cell 7 — Qwen3-4B + δ-Mem + spec-decode

| Context | Status | NIH frac | Peak VRAM | KV @ ctx | decode tok/s | TTFT (s) | Wall (s) |
|---|---|---|---|---|---|---|---|
| 2000 | ok | 1.00 | 13.1 GB | 175.8 MB | 8.7 | 0.248 | 36.1 |
| 4000 | failed | N/A | 0.0 B | 0.0 B | 0.0 | 0.000 | 5.0 |
| 8000 | failed | N/A | 0.0 B | 0.0 B | 0.0 | 0.000 | 5.2 |
| 16000 | failed | N/A | 0.0 B | 0.0 B | 0.0 | 0.000 | 5.5 |
