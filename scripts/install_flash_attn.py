#!/usr/bin/env python3
"""Cache-aware flash-attn installer.

Lookup order:
  1. Local cache (wheels/$WHEEL_PROFILE/flash_attn-*.whl)
  2. Community prebuilt at github.com/mjun0812/flash-attention-prebuild-wheels
  3. Source build (slow; ~15-30 min on T4)

On a community hit, the wheel is also copied into the local cache so a future
session installs from the cache without a network call.

Honest scope: this script only KNOWS about Linux x86_64 / aarch64. macOS, Windows,
and anything exotic are punted to the source-build fallback (which usually also
won't work, but at least fails loudly).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple


COMMUNITY_REPO = "mjun0812/flash-attention-prebuild-wheels"
PREFERRED_FA_VERSIONS = ("2.8.3", "2.7.4", "2.6.3")  # newest first


def detect_env() -> dict:
    import torch  # local import — we expect this to exist already
    py = f"cp{sys.version_info.major}{sys.version_info.minor}"
    # torch.version.cuda is like "12.8" → cu128
    cuda = (torch.version.cuda or "").replace(".", "")
    # torch.__version__ is "2.10.0+cu128" → "2.10"
    torch_ver = ".".join(torch.__version__.split("+")[0].split(".")[:2])
    arch = platform.machine().lower()
    if arch == "amd64":
        arch = "x86_64"
    return {
        "python": py,
        "cuda": f"cu{cuda}" if cuda else None,
        "torch": torch_ver,
        "arch": arch,
        "platform": f"linux_{arch}",
    }


def find_in_local_cache(cache_dir: Path) -> Optional[Path]:
    if not cache_dir.exists():
        return None
    candidates = sorted(cache_dir.glob("flash_attn-*.whl"))
    return candidates[-1] if candidates else None


def list_community_releases() -> List[dict]:
    """Hit the GitHub API (no auth needed for public repos)."""
    url = f"https://api.github.com/repos/{COMMUNITY_REPO}/releases?per_page=30"
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.load(resp)
    except (urllib.error.URLError, json.JSONDecodeError) as e:
        print(f"  community release lookup failed: {e}", file=sys.stderr)
        return []


def pick_community_wheel(env: dict, releases: List[dict]) -> Optional[Tuple[str, str, str]]:
    """Return (release_tag, asset_name, download_url) or None.

    Match strategy: same (cuda, torch, python, platform) tuple, highest available
    flash-attn version preferred. Falls through versions in PREFERRED_FA_VERSIONS.
    """
    if not env["cuda"]:
        return None

    target_suffix = f"+{env['cuda']}torch{env['torch']}-{env['python']}-{env['python']}-{env['platform']}.whl"

    for fa_ver in PREFERRED_FA_VERSIONS:
        target_name = f"flash_attn-{fa_ver}{target_suffix}"
        for rel in releases:
            for asset in rel.get("assets", []):
                if asset["name"] == target_name:
                    return rel["tag_name"], asset["name"], asset["browser_download_url"]
    return None


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as resp, dest.open("wb") as f:
        shutil.copyfileobj(resp, f, length=1 << 20)


def pip_install(wheel_path: Path) -> int:
    return subprocess.call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", str(wheel_path)])


def pip_install_source(arch_list: str) -> int:
    env = os.environ.copy()
    env["TORCH_CUDA_ARCH_LIST"] = arch_list
    return subprocess.call(
        [sys.executable, "-m", "pip", "install", "-q", "flash-attn", "--no-build-isolation"],
        env=env,
    )


def _gpu_supports_fa2() -> bool:
    """FlashAttention-2 needs sm_80+ (Ampere or newer). T4/V100/etc. (sm_75 and
    below) cannot run FA2 at all — the kernel raises RuntimeError on first
    forward. Installing it on those cards is pure overhead because the harness
    will fall back to SDPA mem-efficient anyway.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        for i in range(torch.cuda.device_count()):
            cap = torch.cuda.get_device_capability(i)
            if cap[0] < 8:
                return False
        return True
    except ImportError:
        return False


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", default=os.environ.get("WHEEL_CACHE_DIR", "wheels/kaggle/2xt4"))
    p.add_argument("--arch-list", default=os.environ.get("TORCH_CUDA_ARCH_LIST", "7.5"))
    p.add_argument("--force", action="store_true",
                   help="Install even if the GPU can't actually run FA2.")
    args = p.parse_args()

    if not args.force and not _gpu_supports_fa2():
        try:
            import torch
            caps = [str(torch.cuda.get_device_capability(i))
                    for i in range(torch.cuda.device_count())]
            cap_str = ", ".join(caps) if caps else "no CUDA"
        except Exception:
            cap_str = "unknown"
        print(f"  GPU compute capability ({cap_str}) is below sm_80; FA2 "
              f"requires Ampere or newer. Skipping install — the harness "
              f"will use SDPA mem-efficient.")
        return 0

    cache_dir = Path(args.cache_dir)
    env = detect_env()
    print(f"  detected: python={env['python']} cuda={env['cuda']} torch={env['torch']} arch={env['arch']}")

    # 1. Local cache
    cached = find_in_local_cache(cache_dir)
    if cached is not None:
        print(f"  using local cache: {cached.name}")
        if pip_install(cached) == 0:
            print("  installed from local cache.")
            return 0
        print("  local cached wheel install failed; trying community next.")

    # 2. Community prebuilt
    print(f"  checking {COMMUNITY_REPO} for matching prebuilt...")
    releases = list_community_releases()
    pick = pick_community_wheel(env, releases)
    if pick is not None:
        tag, name, url = pick
        print(f"  hit: {tag} → {name}")
        target = cache_dir / name
        try:
            download(url, target)
        except Exception as e:
            print(f"  download failed: {e}")
            target = None
        if target is not None and target.exists():
            if pip_install(target) == 0:
                size_mb = target.stat().st_size // (1024 * 1024)
                print(f"  installed from community wheel ({size_mb} MB)")
                if size_mb < 95:
                    print(f"  → commit to skip the download next time:")
                    print(f"    git add {target} && git commit -m 'cache: {name}' && git push")
                else:
                    print(f"  ⚠️  wheel is {size_mb} MB — too big to commit directly to git.")
                    print(f"     Options: (a) git-lfs, (b) GitHub Release, (c) leave it as a per-session download.")
                return 0
    else:
        print(f"  no community wheel for this env tuple — falling back to source build.")

    # 3. Source build
    print(f"  building from source with TORCH_CUDA_ARCH_LIST={args.arch_list}")
    rc = pip_install_source(args.arch_list)
    if rc != 0:
        print("  source build failed; flash-attn unavailable, harness will fall back to SDPA")
    return rc


if __name__ == "__main__":
    sys.exit(main())
