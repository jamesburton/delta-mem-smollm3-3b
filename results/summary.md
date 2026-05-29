# Results — S3

| Cell | Title | Status | NIH frac | Peak VRAM | KV bytes | decode tok/s | TTFT (s) |
|---|---|---|---|---|---|---|---|
| 1 | Qwen3-4B vanilla full attention | ok | 1.00 | 8.9 GB | 175.8 MB | 12.5 | 0.236 |
| 2 | Qwen3-4B + δ-Mem adapter | ok | 1.00 | 13.1 GB | 175.8 MB | 9.4 | 0.361 |
| 3 | Qwen3-4B + sliding-window 4K | phase-2-plus | N/A | 0.0 B | 0.0 B | 0.0 | 0.000 |
| 4 | Qwen3-4B + SW-4K + δ-Mem | phase-2-plus | N/A | 0.0 B | 0.0 B | 0.0 | 0.000 |
| 5 | Qwen3-4B + SW-2K + δ-Mem | phase-2-plus | N/A | 0.0 B | 0.0 B | 0.0 | 0.000 |
| 6 | Qwen3-4B + spec-decode | ok | 1.00 | 8.9 GB | 175.8 MB | 9.7 | 0.219 |
| 7 | Qwen3-4B + δ-Mem + spec-decode | ok | 1.00 | 13.1 GB | 175.8 MB | 9.5 | 0.265 |
| 8 | Qwen3-4B + SW-4K + δ-Mem + spec-decode | phase-2-plus | N/A | 0.0 B | 0.0 B | 0.0 | 0.000 |
| 9a | Qwen3.5-4B-MTP-GGUF Q4_K_M | phase-2-plus | N/A | 0.0 B | 0.0 B | 0.0 | 0.000 |
| 9b | Qwen3.5-4B-MTP-GGUF Q5_K_M | phase-2-plus | N/A | 0.0 B | 0.0 B | 0.0 | 0.000 |
| 9c | Qwen3.5-4B-MTP-GGUF Q8_0 | phase-2-plus | N/A | 0.0 B | 0.0 B | 0.0 | 0.000 |
| 10 | Qwen3-4B + StreamingLLM sink+SW-4K + δ-Mem | phase-2-plus | N/A | 0.0 B | 0.0 B | 0.0 | 0.000 |
| 11 | GatedDeltaNet-2 1.3B reference | phase-2-plus | N/A | 0.0 B | 0.0 B | 0.0 | 0.000 |
| 12 | SmolLM3-3B vanilla baseline | phase-2-plus | N/A | 0.0 B | 0.0 B | 0.0 | 0.000 |
| 13 | SmolLM3-3B + δ-Mem (ours) | phase-2-plus | N/A | 0.0 B | 0.0 B | 0.0 | 0.000 |
| 14 | SmolLM3-3B + SW-4K + δ-Mem | phase-2-plus | N/A | 0.0 B | 0.0 B | 0.0 | 0.000 |
| 15 | SmolLM3-3B + δ-Mem + spec-decode | phase-2-plus | N/A | 0.0 B | 0.0 B | 0.0 | 0.000 |
| 16 | SmolLM3-3B + SW-4K + δ-Mem + spec-decode (full c | phase-2-plus | N/A | 0.0 B | 0.0 B | 0.0 | 0.000 |
| T1 | Train δ-Mem adapter for SmolLM3-3B | phase-2-plus | N/A | 0.0 B | 0.0 B | 0.0 | 0.000 |
