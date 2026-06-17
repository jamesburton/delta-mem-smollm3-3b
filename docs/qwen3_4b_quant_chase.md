# Beating the bnb-4bit floor on Qwen3-4B-Instruct-2507 (T1, RTX 3060)

**Date:** 2026-06-17
**Hardware:** T1 — RTX 3060 12 GB (sm_86 Ampere consumer), Xeon X5670 (no AVX, no AVX2), Windows 11
**Baseline to beat:** `scripts/ar_baseline.py` measurement (2026-06-17, `results/qwen3_4b_ar_fa2_sweep.json`)

| config | tok/s | ms/tok | VRAM |
|---|---|---|---|
| bnb-4bit NF4 + bf16 compute, SDPA | 3.0 | 336.7 | 2.56 GB |
| bnb-4bit NF4 + bf16 compute, FA2 | 3.1 | 323.4 | 2.56 GB |
| bf16, SDPA | 4.1 | 246.9 | 7.53 GB |
| bf16, FA2 | 4.1 | 245.2 | 7.53 GB |

The ~90 ms/tok gap between bnb-4bit (336 ms) and bf16 (247 ms) is the **per-forward dequant cost** of bnb's NF4 path. We want a 4-bit format that fuses dequant into matmul, so the GPU isn't paying that cost on every token.

Two candidate paths.

---

## Path A — GPTQ-Marlin (fused dequant + matmul, GPU kernels)

**Recommendation: PRIMARY PATH. High confidence it will install and run; speedup likely but not guaranteed to fully erase the gap.**

### Verified state of the stack

Everything needed is **already installed in the system Python311** (per `C:\Users\james\.claude\CUDA_NOTES.md`):

| component | version | confirmed |
|---|---|---|
| Python | 3.11.3 (`C:\Python311`) | ✓ |
| torch | 2.11.0+cu126 | ✓ |
| transformers | 5.4.0 | ✓ |
| gptqmodel | 7.1.0+d0bed15 | ✓ (`GPTQ_MARLIN` enum present in `BACKEND`) |
| optimum | installed | ✓ |
| flash_attn | 2.8.3.post1 | ✓ |
| bitsandbytes | 0.49.2 | ✓ |

We verified `from gptqmodel import BACKEND` lists `GPTQ_MARLIN` and `MARLIN`. We verified `from transformers import GPTQConfig` exposes the `backend` parameter (default None, accepts `"marlin"`).

### Marlin sm_86 support

From the upstream Marlin README (https://github.com/IST-DASLab/marlin): "**compute capability >= 8.0** (Ampere or Ada, Marlin is not yet optimized for Hopper)." sm_86 (consumer Ampere, RTX 3060/3070/3090) is in-scope. GPTQModel's docs add: "NVIDIA `Turing+` (`sm_75+`) GPUs" supported by their Marlin port.

Reported peak speedup in the Marlin paper: **3.87× over the original CUDA GPTQ kernel** at batch sizes 16-32. Batch=1 decode is a different regime (memory-bandwidth-bound) and the win is more modest, but the kernel is specifically designed to avoid the per-step dequant tax bnb pays.

### Checkpoint choice — what's actually on HF

Search ranked by downloads (`hub_repo_search "Qwen3-4B-Instruct-2507 GPTQ"`):

| repo | downloads | format | group | desc_act | sym | Marlin-compat | size |
|---|---|---|---|---|---|---|---|
| **`JunHowie/Qwen3-4B-Instruct-2507-GPTQ-Int4`** | **249K** | legacy GPTQ (`checkpoint_format: "gptq"`) | 128 | false | true | ✅ yes | **2.67 GB** |
| `superjob/Qwen3-4B-Instruct-2507-GPTQ-Int4` | 40K | **compressed-tensors** (vLLM) | 128 | static | n/a | ⚠ needs `compressed-tensors` lib, vLLM-oriented | similar |
| `kaitchup/Qwen3-4B-Instruct-2507-gptq-w4a16-g128` | 93 | llmcompressor | 128 | n/a | n/a | similar to superjob | similar |
| `pramjana/...-4bit-GPTQ` | 10 | gptq | n/a | n/a | n/a | unknown | unknown |

**JunHowie is the pick:** it's the legacy GPTQ format that `transformers.GPTQConfig(backend="marlin")` was designed to load, with `sym=true` and `desc_act=false` and `group_size=128` — all three are Marlin prerequisites. Quantized with `gptqmodel:4.0.0` (same family as ours). 249K downloads, used in vLLM serve docs widely. Model card link in `scripts/ar_marlin.py`.

(Superjob's compressed-tensors variant would require the `compressed-tensors` Python package and is wired primarily for vLLM. We tested earlier that vLLM is not in our stack. Stay on the JunHowie legacy-GPTQ path.)

### Loading invocation

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
import torch

model = AutoModelForCausalLM.from_pretrained(
    "JunHowie/Qwen3-4B-Instruct-2507-GPTQ-Int4",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    quantization_config=GPTQConfig(bits=4, backend="marlin"),
)
```

This is the documented path on https://huggingface.co/docs/transformers/quantization/gptq#marlin .

### Risks (be honest)

1. **First-load repack.** Marlin requires a specific weight layout; on first load gptqmodel converts the saved tensor layout into Marlin's packed form. This is one-time, but adds ~30-60 s to the first load. Subsequent loads from the same cache directory are fast.
2. **`backend="marlin"` may fall back to `exllama_v2` or `triton`** if any tensor shape/property fails Marlin's strict requirements. The runner prints whichever kernel actually got dispatched (we surface it as a "loaded kernel" line) so we can tell at a glance.
3. **`gptqmodel` on Windows logs ASCII-art under default `cp1252`** — we run with `PYTHONIOENCODING=utf-8` and `-X utf8` (the script does this internally; the launcher batch sets the env var as belt-and-braces). Without this, the import-time logo crashes the script per the existing pitfall log.
4. **The `gptqmodel` install once silently downgraded torch to a CPU wheel** (see CUDA_NOTES "Pip gotcha"). Not an issue for this run because we're not installing — the package is already present and working.
5. **No upstream benchmark for Qwen3-4B on RTX 3060** — community numbers are mostly A100/H100. Realistic expectation: **6-15 tok/s decode** (2-5× over bnb's 3.0), based on the Marlin paper's batch=1 curves on consumer Ampere, but this is the experiment.

### Cost / disk

2.67 GB download (one-time). Goes into `E:\.cache\huggingface\hub` (junction with `C:\Users\james\.cache\huggingface\hub`). Currently 14 GB free on E:; comfortable.

---

## Path B — GGUF + llama.cpp

**Recommendation: SECONDARY PATH, but the cheaper-to-validate one. Already known to work on this box.**

### Why this is the safe path

- `llama-cpp-python==0.3.28` is **already built from source against CUDA 12.8** in the project venv `E:\Development\llm-model-tests\delta-mem-smollm3-3b\.venv\` (per `docs/local-llama-cpp-build.md`).
- `python -c "import llama_cpp; llama_cpp.llama_supports_gpu_offload()"` returns **True** — verified just now.
- The custom build is necessary because this box has an **Intel Xeon X5670 (Westmere, 2010, no AVX/AVX2)**. Every prebuilt llama-cpp-python wheel from PyPI assumes AVX or AVX2 and crashes with `WinError 0xC000001D ILLEGAL_INSTRUCTION` on first call.
- Existing project history: `docs/local-gguf-findings.md` measured **20-22 tok/s decode at 16K context** on Qwen3.5-4B-MTP Q5/Q8 on this exact card. That model is 3.5 (MTP, multi-token-predict on by default), so the published number isn't directly comparable to single-token decode on Qwen3-4B-Instruct-2507 — but the order of magnitude is right and the runtime is the same.

### Checkpoint choice

Top GGUF repos for Qwen3-4B-Instruct-2507, ranked by downloads:

| repo | downloads | notes |
|---|---|---|
| `MaziyarPanahi/Qwen3-4B-Instruct-2507-GGUF` | 131K | Long history, broad quant coverage |
| **`unsloth/Qwen3-4B-Instruct-2507-GGUF`** | **708K** | **Pick.** Unsloth has been the most reliable GGUF source for Qwen3 on this project — their MTP variant was used in `LOCAL_GGUF` sweeps. Active maintenance, full quant matrix, imatrix variants ("UD-Q4_K_XL" etc.) for marginal quality wins. |
| `lmstudio-community/...` | 7.7K | Fine but no advantage over unsloth |
| `bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF` | 130K | Also fine, bartowski-style naming |

### File sizes (unsloth repo, confirmed via HF model card)

| quant | size | recommended? |
|---|---|---|
| **Q4_K_M** | 2.5 GB | ✅ primary — strong quality/size tradeoff |
| Q5_K_S | 2.82 GB | optional |
| **Q5_K_M** | 2.89 GB | ✅ secondary — marginal quality bump for ~400 MB more |
| Q6_K | 3.31 GB | skip — diminishing returns over Q5_K_M |
| Q8_0 | 4.28 GB | skip — disk-tight (14 GB free), and Q5 is close enough |
| F16 | 8.05 GB | skip — we already have bf16 measured via HF |

Recommend downloading **Q4_K_M only first** (2.5 GB, smallest meaningful comparison) and **adding Q5_K_M after** if Q4 demonstrates the speedup we hope for. Disk is tight (14 GB free on E:) — Q4 + GPTQ together = 5.2 GB; comfortable.

### Loading invocation (Python, via `llama-cpp-python`)

```python
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

p = hf_hub_download(
    repo_id="unsloth/Qwen3-4B-Instruct-2507-GGUF",
    filename="Qwen3-4B-Instruct-2507-Q4_K_M.gguf",  # confirmed naming convention
)
llm = Llama(
    model_path=p,
    n_ctx=4096,
    n_gpu_layers=-1,         # all on GPU
    n_batch=512,
    verbose=False,
)
```

The expected filename pattern matches the existing `harness/runners/llamacpp_runner.py` resolver:
`<base>-<quant>.gguf` where `<base>` strips `-GGUF` from the repo name. So
`Qwen3-4B-Instruct-2507-Q4_K_M.gguf`.

### Comparable invocation: matching `scripts/ar_baseline.py`

The baseline runs the same Lily math prompt with `max_new_tokens=128`, greedy
decode, then prints `tok/s = generated_tokens / wall_time`. The GGUF runner
mirrors that:

- Same prompt string from `ar_baseline.py` (the script imports it).
- `max_tokens=128`, `temperature=0.0`, `top_p=1.0` (greedy equivalent).
- Time **decode only** (warmup with 8 tokens first to amortise the prefill cost).
- Print `tok/s` and ms/tok in the same format the baseline uses, so the
  numbers slot straight into `results/qwen3_4b_ar_fa2_sweep.json`-shaped tables.

### Risks

1. **Reasoning mode caveat** (per `docs/local-gguf-findings.md`): some Qwen3.5/Qwen3 GGUFs emit `<think>...</think>` blocks even at temperature=0. Qwen3-4B-**Instruct**-2507 should NOT do this (it's the non-thinking variant, distinct from Qwen3-4B-Thinking), but if we see partial thinking text in the first ~30 tokens of output, bump `max_tokens` to 512 and re-measure.
2. **Prefill is included in llama-cpp's single timing.** At ctx=45 tokens this is negligible (~ms). At long ctx it dominates; not a problem for the short-decode comparison.
3. **Disk: 14 GB free on E:.** Q4_K_M is 2.5 GB. After this and the GPTQ download (2.67 GB), we'd have ~8.5 GB free — still OK.
4. **The venv has no bnb / gptqmodel** — so this runner can ONLY do GGUF. The Marlin runner has to live in the system Python311 stack. Different launcher batches for each.
5. **Speed expectation: ~25-35 tok/s decode** for Q4_K_M Qwen3-4B-Instruct-2507 on RTX 3060 at short ctx. Order-of-magnitude check: the project's prior Q5/Q8 numbers on Qwen3.5-4B-MTP at 16K ctx were 20-22 tok/s (decode-only, see `docs/local-gguf-findings.md`). At ctx=45 the prefill is gone and there's no MTP overhead, so we should see better. This would be **~8-12× the bnb-4bit baseline**.

---

## Path C — Ollama (GGUF, third-party runtime)

Ollama is installed at `E:\Ollama\ollama` (v0.30.7). It's essentially a packaged llama.cpp with its own model registry and a built-in tok/s readout — useful as a third comparison point alongside our project-built `llama-cpp-python` (Path B). Existing local pulls (verified 2026-06-17): `gemma4:e4b` 9.6 GB.

**Why include it:**
- Different runtime path on the same GGUF kernels — sanity-checks whether Path B's number is the kernels themselves or includes llama-cpp-python wrapper overhead.
- Standard `ollama run --verbose` reports `eval rate: X tokens/s` directly — no instrumentation needed.
- Simpler to operate than maintaining a custom llama-cpp-python venv.

**Setup:**
```powershell
# pull a Qwen3-4B-Instruct GGUF; check ollama.com/library for an Instruct-2507 tag.
# Fallback: a community quant like bartowski's, sideloaded via Modelfile if no official tag exists.
ollama pull qwen3:4b-instruct
ollama run --verbose qwen3:4b-instruct "Lily can run 12 kilometers per hour for 4 hours..."
```

**What to capture:** `eval rate` (decode tok/s), `prompt eval rate` (prefill tok/s), `total duration`, `load duration`. Apples-to-apples comparison with Path B requires checking that ollama's quant matches Path B's (Q4_K_M ideally; if Path B used Q4_K_M, point ollama at the same).

**Expected outcome:** within ~10% of Path B. If they diverge much, the llama-cpp-python wrapper has overhead worth investigating.

---

## Recommendation

**Run Path B first, Path A second, in the morning.**

1. **GGUF first** (Path B) — disk download is ~equal, the runtime is already
   built and verified, and the project has prior numbers on similar Qwen3
   GGUFs from this box. Highest probability of producing a clear "yes the
   floor is broken" or "no it isn't" answer with the smallest setup risk.
   Expected outcome: **20-35 tok/s decode** (vs bnb's 3.0).
2. **GPTQ-Marlin second** (Path A) — the more interesting answer if it works
   (it stays inside the transformers/HF ecosystem so it'd plug straight into
   the existing harness for sweeps, NIH eval, FA2 long-ctx etc., without
   maintaining a parallel llama.cpp path). Expected outcome: **6-15 tok/s
   decode** based on Marlin community numbers extrapolated to consumer
   Ampere. If it lands above llama.cpp's GGUF number, the Marlin path wins
   for everything downstream; if below, GGUF stays as the dedicated
   short-decode runner and Marlin is documented as "viable but slower than
   GGUF on this hardware tier."

The two paths are **independent** and **non-conflicting** — they use
different Pythons (Path A: system 3.11; Path B: project `.venv`), different
quant artefacts, and different runtime stacks. So if either falls over,
the other still produces a measurement.

---

## What we explicitly did NOT do

- We did not try `compressed-tensors`/vLLM. vLLM isn't on the box and is
  more painful to install on Windows than the gains warrant for one model.
  If it turns out we want vLLM for serving, that's a separate spike.
- We did not try AWQ. AWQ-Marlin is also supported in gptqmodel (`AWQ_MARLIN`
  backend present) but the legacy GPTQ path is more mature on Windows and
  the JunHowie checkpoint is already specifically prepared for it.
- We did not try EETQ, FP-Quant, or anything Blackwell-only. sm_86 rules them
  out.
- We did not pre-quantize ourselves. Both paths use community checkpoints
  that already exist on HF and are smaller than the bf16 base we already
  cache, so the only cost is one download per path.

## Tomorrow's launch one-liners

```powershell
# Path B (recommended first) — GGUF via llama-cpp in the project venv
.\scripts\ar_gguf_qwen3_4b.bat --quant Q4_K_M

# Path C — Ollama on the same/similar GGUF
ollama pull qwen3:4b-instruct ; ollama run --verbose qwen3:4b-instruct "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

# Path A — GPTQ-Marlin via gptqmodel in the system Python311
.\scripts\ar_marlin.bat
```

Path B and Path A print results in the same format as `ar_baseline.py` for direct comparison against the bnb floor. Path C's `eval rate: X tokens/s` from `--verbose` is the analogue.

---

## References

- Transformers GPTQ docs: https://huggingface.co/docs/transformers/quantization/gptq
- Marlin upstream: https://github.com/IST-DASLab/marlin
- GPTQModel: https://github.com/ModelCloud/GPTQModel
- JunHowie GPTQ-Int4 checkpoint: https://huggingface.co/JunHowie/Qwen3-4B-Instruct-2507-GPTQ-Int4
- Unsloth GGUF: https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF
- Local stack notes: `C:\Users\james\.claude\CUDA_NOTES.md`
- Existing GGUF numbers on this box: `docs/local-gguf-findings.md`
- Existing llama.cpp build recipe: `docs/local-llama-cpp-build.md`
- Baseline data being challenged: `results/qwen3_4b_ar_fa2_sweep.json`
