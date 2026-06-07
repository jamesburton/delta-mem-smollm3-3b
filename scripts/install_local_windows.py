#!/usr/bin/env python3
"""Idempotent installer for the Windows local-dev wheel cache.

Reads `wheels/local/<profile>/manifest.json`, downloads any missing wheels
with a progress meter, verifies SHA256, and installs with `pip install --no-deps`.
Finishes with a tiny FA2 forward pass to prove the install actually works.

Usage:

    python scripts/install_local_windows.py                # profile 3060
    python scripts/install_local_windows.py --profile 3060
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}")
    t0 = time.time()
    last = 0.0
    with urllib.request.urlopen(url, timeout=300) as r, dest.open("wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        got = 0
        while True:
            data = r.read(1 << 22)  # 4 MiB chunks
            if not data:
                break
            f.write(data)
            got += len(data)
            if time.time() - last > 5:
                pct = (100 * got / total) if total else 0
                mbps = (got / 1024 / 1024) / max(time.time() - t0, 1e-3)
                print(f"    {got/1024/1024:.0f}/{total/1024/1024:.0f} MB "
                      f"({pct:.0f}%)  {mbps:.1f} MB/s")
                last = time.time()
    print(f"  done in {time.time()-t0:.0f}s, "
          f"{dest.stat().st_size/1024/1024:.0f} MB")


def ensure_wheel(wheel_dir: Path, entry: dict) -> Path:
    dest = wheel_dir / entry["name"]
    if dest.exists():
        size_ok = dest.stat().st_size == entry["size_bytes"]
        print(f"  {entry['name']} present ({dest.stat().st_size//1024//1024} MB)"
              f"{' — size matches' if size_ok else ' — size MISMATCH, re-download'}")
        if not size_ok:
            dest.unlink()
    if not dest.exists():
        download(entry["url"], dest)
    actual = sha256(dest)
    if actual != entry["sha256"]:
        raise RuntimeError(
            f"sha256 mismatch for {entry['name']}\n"
            f"  expected: {entry['sha256']}\n"
            f"  got:      {actual}\n"
            f"  delete {dest} and re-run."
        )
    print(f"  sha256 OK: {actual[:12]}...")
    return dest


def pip_install_no_deps(wheel: Path) -> None:
    rc = subprocess.call([sys.executable, "-m", "pip", "install",
                          "--no-deps", str(wheel)])
    if rc != 0:
        raise RuntimeError(f"pip install failed for {wheel.name}")


def verify_fa2() -> None:
    print("==> verifying FA2 with a tiny forward pass")
    import torch
    import flash_attn
    from flash_attn import flash_attn_func
    print(f"  torch {torch.__version__}  flash_attn {flash_attn.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available; FA2 needs a CUDA-capable GPU")
    cap = torch.cuda.get_device_capability(0)
    print(f"  GPU 0 compute capability: sm_{cap[0]}{cap[1]}")
    if cap[0] < 8:
        raise RuntimeError(f"FA2 needs sm_80+, this GPU is sm_{cap[0]}{cap[1]}")
    q = torch.randn(2, 4, 8, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, 4, 8, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(2, 4, 8, 64, device="cuda", dtype=torch.bfloat16)
    out = flash_attn_func(q, k, v)
    assert tuple(out.shape) == (2, 4, 8, 64), out.shape
    print(f"  FA2 forward OK, output shape {tuple(out.shape)}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="3060",
                   help="Sub-directory under wheels/local (default: 3060).")
    p.add_argument("--skip-verify", action="store_true",
                   help="Skip the FA2 smoke test at the end.")
    args = p.parse_args()

    wheel_dir = REPO_ROOT / "wheels" / "local" / args.profile
    manifest_path = wheel_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"  no manifest at {manifest_path}", file=sys.stderr)
        return 1
    manifest = json.loads(manifest_path.read_text())
    print(f"==> profile: local/{args.profile}  python={manifest['python']}  "
          f"torch={manifest['torch']}")

    for entry in manifest["wheels"]:
        print(f"\n--- {entry['name']} ---")
        wheel = ensure_wheel(wheel_dir, entry)
        pip_install_no_deps(wheel)

    if not args.skip_verify:
        print()
        verify_fa2()

    print("\n==> done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
