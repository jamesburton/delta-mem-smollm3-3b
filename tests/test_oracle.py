import os
import json
from pathlib import Path

import pytest

from harness import oracle


def test_detect_auth_returns_unavailable_when_nothing_set(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(oracle.shutil, "which", lambda _: None)
    assert oracle.detect_auth() == "unavailable"


def test_detect_auth_returns_api_when_key_present(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(oracle.shutil, "which", lambda _: None)
    assert oracle.detect_auth() == "api"


def test_ask_cli_returns_cached_response(monkeypatch, tmp_path):
    monkeypatch.setattr(oracle, "CACHE_DIR", tmp_path)
    # Pre-populate cache to avoid invoking real subprocess
    fake = oracle._cache_path("hello", "claude-test")
    fake.write_text(json.dumps({"text": "world", "auth_mode": "cli"}))
    result = oracle.ask("hello", model="claude-test", auth_mode="cli")
    assert result is not None
    assert result.text == "world"
    assert result.cached is True


def test_ask_returns_none_when_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(oracle, "CACHE_DIR", tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(oracle.shutil, "which", lambda _: None)
    assert oracle.ask("prompt", model="claude-test") is None
