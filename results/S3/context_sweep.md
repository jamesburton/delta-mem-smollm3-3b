# Context sweep results

## Cell 1 — Qwen3-4B vanilla full attention

| Context | Status | NIH frac | Peak VRAM | KV @ ctx | decode tok/s | TTFT (s) | Wall (s) |
|---|---|---|---|---|---|---|---|
| 2000 | ok | 1.00 | 8.9 GB | 175.8 MB | 12.7 | 0.226 | 33.2 |
| 4000 | failed | N/A | 0.0 B | 0.0 B | 0.0 | 0.000 | 5.4 |
| 8000 | failed | N/A | 0.0 B | 0.0 B | 0.0 | 0.000 | 5.0 |
| 16000 | failed | N/A | 0.0 B | 0.0 B | 0.0 | 0.000 | 6.4 |

## Cell 2 — Qwen3-4B + δ-Mem adapter

| Context | Status | NIH frac | Peak VRAM | KV @ ctx | decode tok/s | TTFT (s) | Wall (s) |
|---|---|---|---|---|---|---|---|
| 2000 | ok | 1.00 | 13.1 GB | 175.8 MB | 9.5 | 0.279 | 37.5 |
| 4000 | failed | N/A | 0.0 B | 0.0 B | 0.0 | 0.000 | 4.5 |
| 8000 | failed | N/A | 0.0 B | 0.0 B | 0.0 | 0.000 | 4.8 |
| 16000 | failed | N/A | 0.0 B | 0.0 B | 0.0 | 0.000 | 5.1 |

## Cell 6 — Qwen3-4B + spec-decode

| Context | Status | NIH frac | Peak VRAM | KV @ ctx | decode tok/s | TTFT (s) | Wall (s) |
|---|---|---|---|---|---|---|---|
| 2000 | ok | 1.00 | 8.9 GB | 175.8 MB | 9.4 | 0.224 | 38.6 |
| 4000 | failed | N/A | 0.0 B | 0.0 B | 0.0 | 0.000 | 18.6 |
| 8000 | failed | N/A | 0.0 B | 0.0 B | 0.0 | 0.000 | 5.8 |
| 16000 | failed | N/A | 0.0 B | 0.0 B | 0.0 | 0.000 | 6.1 |

## Cell 7 — Qwen3-4B + δ-Mem + spec-decode

| Context | Status | NIH frac | Peak VRAM | KV @ ctx | decode tok/s | TTFT (s) | Wall (s) |
|---|---|---|---|---|---|---|---|
| 2000 | ok | 1.00 | 13.1 GB | 175.8 MB | 9.6 | 0.278 | 37.3 |
| 4000 | failed | N/A | 0.0 B | 0.0 B | 0.0 | 0.000 | 4.7 |
| 8000 | failed | N/A | 0.0 B | 0.0 B | 0.0 | 0.000 | 4.8 |
| 16000 | failed | N/A | 0.0 B | 0.0 B | 0.0 | 0.000 | 5.2 |
