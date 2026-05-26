"""Resumable run state.

`state.json` lives at the repo root. Schema:

    {
      "run_id": "2026-05-26-kaggle-T4x2-xxx",
      "started_at": "<ISO8601>",
      "last_checkpoint_at": "<ISO8601>",
      "completed_cells": ["1", "2", "9a", ...],
      "current_cell": "4",
      "stage": "S3",
      "wall_clock_budget_seconds": 28800,
      "wall_clock_elapsed_seconds": 1234.5,
      "git_remote": "origin",
      "git_branch": "main"
    }

Cells append to `completed_cells` on success. On crash/interrupt, the next run
reads this and skips already-done work.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional


STATE_PATH = Path(__file__).resolve().parents[1] / "state.json"


@dataclass
class RunState:
    run_id: str
    started_at: float
    stage: str
    wall_clock_budget_seconds: int = 28_800  # 8h default
    completed_cells: List[str] = field(default_factory=list)
    current_cell: Optional[str] = None
    last_checkpoint_at: float = 0.0

    @property
    def wall_clock_elapsed_seconds(self) -> float:
        return time.time() - self.started_at

    @property
    def wall_clock_exceeded(self) -> bool:
        return self.wall_clock_elapsed_seconds > self.wall_clock_budget_seconds

    def mark_completed(self, cell_id: str) -> None:
        if cell_id not in self.completed_cells:
            self.completed_cells.append(cell_id)
        self.current_cell = None
        self.last_checkpoint_at = time.time()
        self.save()

    def save(self) -> None:
        STATE_PATH.write_text(json.dumps(asdict(self) | {
            "wall_clock_elapsed_seconds": self.wall_clock_elapsed_seconds,
        }, indent=2))


def load_or_create(run_id: str, stage: str, budget_seconds: int = 28_800) -> RunState:
    if STATE_PATH.exists():
        raw = json.loads(STATE_PATH.read_text())
        return RunState(
            run_id=raw["run_id"],
            started_at=raw["started_at"],
            stage=raw["stage"],
            wall_clock_budget_seconds=raw.get("wall_clock_budget_seconds", budget_seconds),
            completed_cells=raw.get("completed_cells", []),
            current_cell=raw.get("current_cell"),
            last_checkpoint_at=raw.get("last_checkpoint_at", 0.0),
        )
    state = RunState(
        run_id=run_id,
        started_at=time.time(),
        stage=stage,
        wall_clock_budget_seconds=budget_seconds,
    )
    state.save()
    return state
