# delta-mem-smollm3-3b

Cross-base retrofit study of the **δ-Mem** online-memory mechanism on a second
base model family (SmolLM3-3B), alongside the published `Qwen3-4B-Instruct-2507`
baseline. The matrix also tests sliding-window / StreamingLLM sparse attention,
speculative decoding, and native MTP (via Qwen3.5-4B-MTP-GGUF) as compound
levers over the same backbones.

The headline question: **can δ-Mem + sparse attention + spec-decode jointly
beat a vanilla full-attention baseline on (quality at fixed KV budget) AND
(decode tok/s) simultaneously?**

> 📄 Full spec: [`docs/spec.md`](docs/spec.md)

## Status

Scaffolding — design locked, implementation pending. See [`docs/spec.md`](docs/spec.md) §5 for the test cells.

## Running on Kaggle

1. On Kaggle, **File → Import Notebook → URL** and paste:
   `https://raw.githubusercontent.com/jamesburton/delta-mem-smollm3-3b/main/notebooks/run_matrix.ipynb`
2. Add a GPU accelerator (T4×2, P100, or L4 — see notebook for selection guidance).
3. **Run All**. The first cell will:
   - clone this repo into `/kaggle/working/`
   - install pinned dependencies
   - prompt for Claude oracle auth (or skip if `ANTHROPIC_API_KEY` is set as a Kaggle secret)
   - read `state.json` and resume from wherever the last session stopped
4. Each cell commits its results JSON back to this repo via the configured PAT
   (set as Kaggle secret `GH_PAT_DELTA_MEM_TESTS`). Stop a run early by pushing
   `control/STATUS=stop` from your laptop; the notebook checks at every cell
   boundary.

## Running locally

```powershell
git clone --recurse-submodules https://github.com/jamesburton/delta-mem-smollm3-3b
cd delta-mem-smollm3-3b
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter notebook notebooks/run_matrix.ipynb
```

## Layout

```
delta-mem-smollm3-3b/
├── README.md
├── requirements.txt
├── notebooks/
│   └── run_matrix.ipynb         # the runnable entry point
├── harness/                     # cell registry, state, oracle, runners
├── scripts/                     # bootstrap and auth helpers
├── docs/
│   └── spec.md                  # full design doc
├── control/
│   └── STATUS                   # graceful kill-switch (push "stop" to abort)
└── results/                     # per-cell JSON metrics + summary.md
```

## Kill-switch

Three independent termination paths:

- `control/STATUS` set to `stop` (push to GitHub from anywhere) → graceful exit at next cell boundary.
- Wall-clock cap (default **8 hours**, configurable) → forces summary + exit before Kaggle's ~9h kernel limit.
- Natural completion → final summary pushed, `os._exit(0)` called, user reminded to also stop the Kaggle session.

## Reproducibility

- `requirements.txt` pins versions; deterministic seeds where the framework allows.
- Every cell records its exact command, package versions, and GPU info in its results JSON.
- The notebook is the single source of truth; harness modules are imported, not edited inline.

## License

Code: Apache-2.0 (or MIT — TBD before first public results push).
δ-Mem weights: CC-BY-4.0 per upstream.
Gated DeltaNet-2 weights (if used): NVIDIA Source Code License-NC, non-commercial only.
