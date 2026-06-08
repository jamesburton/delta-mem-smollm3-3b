#!/usr/bin/env python3
"""Generic cache setup for accelerator profiles.

Pattern: each profile (kaggle/2xt4, colab/l4, local/3060, …) gets a
`wheels/<profile>/manifest.json` that lists:

  - `wheels` — Python wheels with primary `url` + optional `fallback_urls`
    + `sha256`. Cached at `wheels/<profile>/<name>.whl`. Verified, installed
    once present.

  - `data_caches` — large read-only artefacts (HF model cache, pip wheelhouse,
    etc.) that should ride on a Kaggle Dataset / Colab Drive / etc. attachment
    rather than being re-downloaded every session.

This script:

  1. Resolves each wheel: local cache → primary URL → fallback URLs.
  2. For each data cache: detects the attachment (Kaggle / Colab convention)
     and symlinks the destination to it. Reports clear instructions when
     missing.
  3. If `KAGGLE_CACHE_BUILD=1` or `--cache-build`: copies the freshly-built
     HF cache to `/kaggle/working/cache_export/` so the user can publish it
     as a Kaggle Dataset for future runs.

Designed to be portable: drop `scripts/cache_setup.py` + a
`wheels/<profile>/manifest.json` into any repo and you get the same
behaviour. No project-specific assumptions baked in.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional, List, Dict, Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHUNK = 1 << 22  # 4 MiB


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  downloading {url}")
    t0 = time.time()
    last = 0.0
    with urllib.request.urlopen(url, timeout=600) as r, tmp.open("wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        got = 0
        while True:
            data = r.read(DEFAULT_CHUNK)
            if not data:
                break
            f.write(data)
            got += len(data)
            if time.time() - last > 5 and total:
                pct = 100 * got / total
                mbps = (got / 1024 / 1024) / max(time.time() - t0, 1e-3)
                print(f"    {got/1024/1024:.0f}/{total/1024/1024:.0f} MB "
                      f"({pct:.0f}%) {mbps:.1f} MB/s")
                last = time.time()
    tmp.replace(dest)


def ensure_wheel(wheel_dir: Path, entry: Dict[str, Any]) -> Optional[Path]:
    """Resolve one wheel entry. Returns the cached path on success, None on
    full failure. Installs via `pip install --no-deps` once cached and verified.
    """
    name = entry["name"]
    dest = wheel_dir / name

    expected = entry.get("sha256") or ""
    # Existing cache hit
    if dest.exists():
        if not expected:
            print(f"  {name}: cached ({dest.stat().st_size//1024//1024} MB), "
                  f"sha256 not pinned in manifest — trusting local file")
            _pip_install(dest)
            return dest
        try:
            actual = sha256(dest)
            if actual == expected:
                print(f"  {name}: cached ({dest.stat().st_size//1024//1024} MB), sha256 OK")
                _pip_install(dest)
                return dest
            print(f"  {name}: cached file sha256 mismatch — re-downloading")
            dest.unlink()
        except OSError as e:
            print(f"  {name}: sha256 check failed ({e}); re-downloading")
            dest.unlink(missing_ok=True)

    # Try primary then fallbacks
    urls = [entry["url"]] + list(entry.get("fallback_urls", []))
    for url in urls:
        try:
            download(url, dest)
        except (urllib.error.URLError, OSError) as e:
            print(f"  {name}: download failed from {url} ({e})")
            continue
        if not expected:
            print(f"  {name}: downloaded ({dest.stat().st_size//1024//1024} MB), "
                  f"sha256 not pinned — accepting")
            _pip_install(dest)
            return dest
        try:
            actual = sha256(dest)
        except OSError as e:
            print(f"  {name}: post-download sha256 read failed ({e})")
            dest.unlink(missing_ok=True)
            continue
        if actual == expected:
            print(f"  {name}: downloaded + sha256 OK")
            _pip_install(dest)
            return dest
        print(f"  {name}: sha256 mismatch (expected {expected[:12]}…, "
              f"got {actual[:12]}…) — trying next source")
        dest.unlink(missing_ok=True)

    print(f"  {name}: ALL SOURCES FAILED. Falling back to caller's recovery path.")
    return None


def _pip_install(wheel: Path) -> None:
    rc = subprocess.call([sys.executable, "-m", "pip", "install",
                          "--no-deps", "-q", str(wheel)])
    if rc != 0:
        # `pip install` returns non-zero if the wheel is already at the
        # same version; that's fine for our purposes.
        # Treat any non-zero as informational but continue.
        print(f"    (pip install returned {rc}; assuming already-installed)")


def setup_data_cache(entry: Dict[str, Any]) -> bool:
    """Symlink a destination directory to a Kaggle/Colab attachment.

    Returns True if the cache was successfully wired up; False otherwise
    (with diagnostic prints).
    """
    name = entry["name"]
    mount_to = Path(os.path.expanduser(entry["mount_to"]))
    dataset_slug = entry.get("kaggle_dataset")
    subdir = entry.get("subdir_in_dataset", "")

    if not dataset_slug:
        print(f"  {name}: no kaggle_dataset in manifest; skipping")
        return False

    if not Path("/kaggle/input").exists():
        print(f"  {name}: not on Kaggle (no /kaggle/input); "
              f"data caches expect a Kaggle-like environment.")
        return False

    # Kaggle dataset slugs like 'owner/dataset' mount as /kaggle/input/dataset
    slug = dataset_slug.split("/")[-1]
    attached = Path(f"/kaggle/input/{slug}")
    if not attached.exists():
        print(f"  {name}: dataset '{dataset_slug}' NOT attached.")
        print(f"      To enable: open the notebook on Kaggle → '+ Add Data' →")
        print(f"      search for '{slug}'.")
        return False

    src = attached / subdir if subdir else attached
    if not src.exists():
        print(f"  {name}: dataset attached but '{src}' missing inside it. "
              f"Subdir mismatch?")
        return False

    mount_to.parent.mkdir(parents=True, exist_ok=True)
    if mount_to.is_symlink() or mount_to.exists():
        if mount_to.is_symlink():
            mount_to.unlink()
        else:
            backup = mount_to.with_name(mount_to.name + ".pre-cache-backup")
            print(f"  {name}: {mount_to} exists and isn't a symlink; "
                  f"moving to {backup}")
            mount_to.rename(backup)

    mount_to.symlink_to(src, target_is_directory=True)
    print(f"  {name}: symlinked {mount_to} → {src}")
    return True


def export_cache_for_dataset_creation(manifest: Dict[str, Any]) -> None:
    """Copy the populated HF/pip caches to /kaggle/working/cache_export/ so the
    user can Save Version → publish as a Kaggle Dataset.

    Resolves symlinks so the output is a self-contained tree (Kaggle Datasets
    don't follow symlinks back into HF's blob storage).
    """
    export_root = Path("/kaggle/working/cache_export")
    if not Path("/kaggle/working").exists():
        print("==> KAGGLE_CACHE_BUILD set but /kaggle/working not present; skipping export")
        return

    print(f"==> KAGGLE_CACHE_BUILD: seeding {export_root} for Kaggle Dataset creation")
    export_root.mkdir(parents=True, exist_ok=True)

    for entry in manifest.get("data_caches", []):
        name = entry["name"]
        src = Path(os.path.expanduser(entry["mount_to"]))
        # Don't re-export a cache that's already a symlink — that means
        # the attachment is doing its job.
        if src.is_symlink():
            print(f"  {name}: already symlinked (attachment present); skipping export")
            continue
        if not src.exists():
            print(f"  {name}: nothing at {src} to export; skipping")
            continue
        subdir = entry.get("subdir_in_dataset", name)
        dst = export_root / subdir
        if dst.exists():
            print(f"  {name}: {dst} already exists; rsyncing on top")
        dst.mkdir(parents=True, exist_ok=True)
        print(f"  {name}: copying {src} → {dst} (resolving symlinks)…")
        if shutil.which("rsync"):
            subprocess.check_call(["rsync", "-aL", "--info=progress2",
                                   f"{src}/", str(dst) + "/"])
        else:
            # Fallback: shutil-based copy. Slower than rsync but works.
            for item in src.rglob("*"):
                if item.is_symlink() or item.is_file():
                    rel = item.relative_to(src)
                    target = dst / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, target, follow_symlinks=True)
        print(f"  {name}: export complete")

    readme = export_root / "README.md"
    readme.write_text(
        f"# Cache export for {manifest['profile']}\n\n"
        f"Generated by `scripts/cache_setup.py` with `KAGGLE_CACHE_BUILD=1`.\n\n"
        f"## To publish as a Kaggle Dataset\n\n"
        f"1. On Kaggle: **Save Version → Save & Run All (Commit)**, ensure "
        f"output is preserved.\n"
        f"2. Open the saved version's output page.\n"
        f"3. **New Dataset → From Notebook Output**, pick the slug names "
        f"the manifest expects.\n"
        f"4. Subsequent notebook runs: **+ Add Data**, attach the dataset, and "
        f"this script will detect + symlink it automatically.\n",
        encoding="utf-8",
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--profile", required=True,
                   help="Profile slug, e.g. 'kaggle/2xt4' or 'local/3060'. "
                        "Resolves to wheels/<profile>/manifest.json.")
    p.add_argument("--cache-build", action="store_true",
                   default=bool(os.environ.get("KAGGLE_CACHE_BUILD")),
                   help="After install, copy HF cache contents to "
                        "/kaggle/working/cache_export/ for Kaggle Dataset "
                        "publishing. Also via env KAGGLE_CACHE_BUILD=1.")
    args = p.parse_args()

    manifest_path = REPO_ROOT / "wheels" / args.profile / "manifest.json"
    if not manifest_path.exists():
        print(f"manifest not found: {manifest_path}", file=sys.stderr)
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    wheel_dir = manifest_path.parent

    print(f"==> profile: {manifest['profile']}")
    print(f"    {manifest.get('description', '')}")
    print(f"    python={manifest.get('python')}  torch={manifest.get('torch')}  "
          f"cuda={manifest.get('cuda')}")

    wheels = manifest.get("wheels", [])
    if wheels:
        print(f"\n==> wheels ({len(wheels)}):")
        for entry in wheels:
            ensure_wheel(wheel_dir, entry)

    caches = manifest.get("data_caches", [])
    if caches:
        print(f"\n==> data caches ({len(caches)}):")
        for entry in caches:
            setup_data_cache(entry)

    if args.cache_build:
        print()
        export_cache_for_dataset_creation(manifest)

    print("\n==> cache_setup complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
