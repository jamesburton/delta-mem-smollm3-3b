"""Kill-switch: STATUS file + wall-clock cap.

Three termination paths:
1. `control/STATUS` contents become `stop` — graceful exit at next checkpoint.
2. Wall-clock budget exceeded — forces summary + exit.
3. All cells done — natural completion.

The notebook calls `should_stop(state)` at every cell boundary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from .state import RunState

STATUS_PATH = Path(__file__).resolve().parents[1] / "control" / "STATUS"


def read_status() -> str:
    if not STATUS_PATH.exists():
        return "run"
    return STATUS_PATH.read_text().strip().lower()


def should_stop(state: RunState) -> Tuple[bool, str]:
    """Return (stop?, reason)."""
    status = read_status()
    if status == "stop":
        return True, "STATUS file set to 'stop'"
    if state.wall_clock_exceeded:
        return True, (
            f"wall-clock budget exceeded "
            f"({state.wall_clock_elapsed_seconds:.0f}s > "
            f"{state.wall_clock_budget_seconds}s)"
        )
    return False, ""
