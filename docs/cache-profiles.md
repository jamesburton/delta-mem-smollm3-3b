# Cache profiles — a portable pattern for skipping cloud cold starts

This repo uses a small convention to cache **wheels** and **data** for
specific accelerator profiles (Kaggle T4×2, Colab L4, RunPod A100, the local
RTX 3060, …). Drop the script + a JSON manifest into any project and you get
the same behaviour: notebooks attach a one-time-built cache and skip 5+
minutes of redownload on every fresh kernel.

This doc explains the convention so you can lift it cleanly into the next
project.

## The two files

```
wheels/<profile>/
├── manifest.json        # describes wheels + data caches for this accelerator
└── README.md            # one-time setup instructions for this profile

scripts/
└── cache_setup.py       # generic resolver; no project-specific knowledge
```

`scripts/cache_setup.py` reads `wheels/<profile>/manifest.json`. That's it.
Anything project-specific lives in the manifest.

## Manifest schema

```jsonc
{
  // Profile identification (used in log output, paths)
  "profile": "kaggle/2xt4",
  "description": "Free-text human description shown at bootstrap.",

  // Environment expectations (informational; not enforced)
  "python":   "cp310",
  "platform": "linux_x86_64",
  "torch":    "2.10.x+cu128",
  "cuda":     "cu128",

  // ----------------------------------------------------------------------
  // WHEELS — Python wheels resolved + verified + installed.
  // ----------------------------------------------------------------------
  "wheels": [
    {
      // File name as it should land in wheels/<profile>/
      "name": "flash_attn-2.7.4+cu128torch2.10-cp310-cp310-linux_x86_64.whl",

      // Primary URL — usually a GitHub Release asset on the repo, so future
      // session count goes through your control surface (rate limits, etc.).
      "url": "https://github.com/<you>/<repo>/releases/download/cache-vN/<name>",

      // Fallback URLs — tried in order if primary fails. Community wheels,
      // upstream HF, etc. Keep these so the cache is opportunistic — the
      // notebook works on day one even without a release published.
      "fallback_urls": [
        "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.3.13/..."
      ],

      // SHA256 of the wheel. Verified after download, before install. If a
      // primary or fallback returns the wrong content, the script moves on
      // to the next source.
      "sha256":      "abc123…",
      "size_bytes":  150847232
    }
  ],

  // ----------------------------------------------------------------------
  // DATA CACHES — big read-only blobs (HF model cache, pip wheelhouse, …)
  // that ride on the platform's native attach mechanism rather than HTTP.
  // ----------------------------------------------------------------------
  "data_caches": [
    {
      // Logical name (used in log output)
      "name": "huggingface_hub",
      "description": "Pre-populated HF Hub cache for this project's models",

      // Kaggle Dataset slug. Slug format is <owner>/<dataset>.
      // Attachment path: /kaggle/input/<dataset>/  (just the slug, no owner)
      "kaggle_dataset":     "jamesburton/delta-mem-smollm3-3b-cache",

      // Optional subdirectory inside the attachment to use as the source.
      // Empty / missing means the dataset root.
      "subdir_in_dataset":  "huggingface_hub",

      // Where to symlink to (the *destination*). Will be replaced with a
      // symlink → /kaggle/input/<dataset>/<subdir>. Existing non-symlink
      // contents at this path are renamed to .pre-cache-backup before
      // replacement.
      "mount_to": "~/.cache/huggingface/hub"
    }
  ]
}
```

## How the resolver behaves

`python scripts/cache_setup.py --profile <slug>`:

For each wheel:
1. Local cache (`wheels/<profile>/<name>.whl`) → SHA256 verify → install.
2. Primary URL → download → SHA256 verify → install.
3. Each fallback URL in turn → same dance.
4. If all sources fail, prints a clear message and continues. Other wheels
   still get resolved; the caller decides what to do (the existing
   `install_flash_attn.py` falls back to a source build).

For each data cache:
1. Detect Kaggle-like environment (`/kaggle/input/` exists).
2. Detect attachment (`/kaggle/input/<dataset-slug>/`).
3. If attached, replace destination with a symlink. Existing non-symlink
   contents get backed up with `.pre-cache-backup` suffix (so a half-built
   cache doesn't poison the next run).
4. If not attached, prints how to add the dataset and continues. Notebook
   continues to download fresh.

With `KAGGLE_CACHE_BUILD=1` env or `--cache-build`:
- After install + symlink, the script copies the live HF cache (resolving
  symlinks) to `/kaggle/working/cache_export/` so the user can Save Version
  and publish as a Kaggle Dataset.

## Adapting for other platforms

The current implementation is Kaggle-aware (`/kaggle/input/` detection). To
add Colab / Drive / etc.:

- **Colab + Google Drive**: replace `kaggle_dataset` with `gdrive_path` and
  detect `/content/drive/MyDrive/`. Same symlink approach.
- **RunPod + persistent volume**: replace with `runpod_volume_path` and
  detect `/workspace/`.
- **Plain S3 / GCS**: have the script download once and store in
  `/workspace/cache/`. Add `s3_uri` / `gcs_uri` fields and a download path.

All of these are clean additions to `setup_data_cache` — the manifest schema
stays the same shape (`name`, `description`, `mount_to`, plus the
platform-specific source field).

## What's already cached this way

| Profile | manifest |
|---|---|
| `local/3060` | `wheels/local/3060/manifest.json` — Windows + torch 2.9 + FA2 |
| `kaggle/2xt4` | `wheels/kaggle/2xt4/manifest.json` — T4×2, Qwen3-4B + GGUFs |

Add a new profile by copying one of those and editing the fields. No code
changes anywhere else.

## Why not just commit the wheels to git?

- GitHub blocks single files over 100 MB. flash-attn cu128 is ~150 MB.
- GitHub Releases is the official escape hatch: per-release-asset size limit
  is 2 GB, no LFS required, available via plain HTTP.
- Large datasets (the 7.5 GB Qwen3-4B base) don't belong in any git
  history — Kaggle Datasets / HF Hub / GCS exist for that.

## "Wheel as branch" pattern (alternative to Releases)

A separate long-lived branch holding the wheels works too:

```jsonc
{
  "wheels": [{
    "name": "flash_attn-2.7.4...whl",
    "url": "https://github.com/<you>/<repo>/raw/cache-kaggle-2xt4/wheels/kaggle/2xt4/flash_attn-2.7.4...whl",
    "fallback_urls": ["…"],
    "sha256": "…"
  }]
}
```

Pros: single push to update; no manual release-asset upload; ties cache
contents to git history.

Cons: branches still hit the 100 MB single-file cap unless you use git-lfs.
With LFS, repo clones get heavier even when you don't need the cache. With
Releases, the cache is genuinely out-of-band.

For sub-100 MB wheels (most things that aren't flash-attn): branches are
fine and simpler than Releases. For larger artefacts: Releases.

The schema doesn't care which you pick — just point `url` at whatever HTTP
source you want.

## Why not just always use HF Hub fresh?

HF Hub IS already serving the model. The Kaggle Dataset adds:
- **Locality**: dataset reads come from Kaggle's storage, faster than HF for
  a fresh kernel.
- **Stability**: a pinned dataset version doesn't change under you. HF Hub
  sometimes ships new metadata that re-downloads layers.
- **Cost**: free on Kaggle. No per-pull bandwidth bill on HF.

For a single-developer prototype none of these matter much. For a project
that re-runs Kaggle notebooks dozens of times during an iteration cycle, the
5+ minutes per run adds up to hours.

## When NOT to use this

- Trivially small caches (sub-100 MB): regular network downloads are fine.
- One-off experiments: setting up the Kaggle Dataset is overhead.
- Anything that changes more often than monthly: cache invalidation
  becomes a chore.

For longer-lived multi-notebook projects (where the same model weights are
hit by every notebook in the repo), it's clearly worth it.
