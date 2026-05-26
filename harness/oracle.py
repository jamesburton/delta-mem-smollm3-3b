"""Frontier-model oracle with auto-detect auth.

Priority:
1. `claude` CLI on PATH and `claude --version` succeeds → use subprocess.
2. `ANTHROPIC_API_KEY` env var set → use the `anthropic` Python SDK.
3. Neither → return `None` and skip all oracle scoring (record as `oracle: unavailable`
   in summary).

Responses are cached on disk under `results/_oracle_cache/<sha256>.json` so re-runs
are free.

NOTE: Stub implementation — wire-up belongs in the implementation phase.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional


CACHE_DIR = Path(__file__).resolve().parents[1] / "results" / "_oracle_cache"
AuthMode = Literal["cli", "api", "unavailable"]


@dataclass
class OracleResult:
    text: str
    cached: bool
    auth_mode: AuthMode


def detect_auth() -> AuthMode:
    if shutil.which("claude"):
        try:
            r = subprocess.run(
                ["claude", "--version"], capture_output=True, text=True, timeout=5
            )
            if r.returncode == 0:
                return "cli"
        except (subprocess.TimeoutExpired, OSError):
            pass
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "api"
    return "unavailable"


def _cache_path(prompt: str, model: str) -> Path:
    h = hashlib.sha256(f"{model}\0{prompt}".encode()).hexdigest()
    return CACHE_DIR / f"{h}.json"


def ask(
    prompt: str,
    *,
    model: str = "claude-sonnet-4-6",
    auth_mode: Optional[AuthMode] = None,
) -> Optional[OracleResult]:
    """Stub. Real implementation lands in the impl-plan phase."""
    raise NotImplementedError(
        "Oracle wire-up is part of the implementation phase, not the scaffold."
    )
