"""Frontier-model oracle with CLI + API auto-detect and on-disk caching."""

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


@dataclass(frozen=True)
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
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{h}.json"


def _read_cache(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _write_cache(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload))


def _ask_cli(prompt: str, model: str) -> str:
    r = subprocess.run(
        ["claude", "--print", "--model", model, prompt],
        capture_output=True, text=True, timeout=180,
    )
    if r.returncode != 0:
        raise RuntimeError(f"claude CLI failed: {r.stderr}")
    return r.stdout.strip()


def _ask_api(prompt: str, model: str) -> str:
    from anthropic import Anthropic  # lazy import
    client = Anthropic()
    msg = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    parts = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
    return "".join(parts).strip()


def ask(
    prompt: str,
    *,
    model: str = "claude-sonnet-4-6",
    auth_mode: Optional[AuthMode] = None,
) -> Optional[OracleResult]:
    mode = auth_mode or detect_auth()
    path = _cache_path(prompt, model)
    cached = _read_cache(path)
    if cached is not None:
        return OracleResult(text=cached["text"], cached=True, auth_mode=cached.get("auth_mode", mode))
    if mode == "unavailable":
        return None
    text = _ask_cli(prompt, model) if mode == "cli" else _ask_api(prompt, model)
    _write_cache(path, {"text": text, "auth_mode": mode})
    return OracleResult(text=text, cached=False, auth_mode=mode)
