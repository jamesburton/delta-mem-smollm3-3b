"""Cell registry — single source of truth for what's in the test matrix.

Each cell is a small dataclass describing what to run; the actual implementations
live in `runners/` and are dispatched by stage. Keep this file declarative.

This is a scaffold — concrete `run()` callables are filled in during the
implementation phase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional


Stage = Literal["S1", "S2", "S3"]
Stack = Literal["hf", "llamacpp", "nvlabs"]


@dataclass(frozen=True)
class Cell:
    id: str
    title: str
    base_model: str
    stack: Stack
    stages: List[Stage]
    kv_lever: str
    speed_lever: str
    blocked_by: Optional[str] = None  # cell id
    notes: str = ""


QWEN_BASE = "Qwen/Qwen3-4B-Instruct-2507"
SMOLLM_BASE = "HuggingFaceTB/SmolLM3-3B"
QWEN_DRAFT = "Qwen/Qwen3-0.6B"
GDN2_REF = "NVlabs/GatedDeltaNet-2"

CELLS: List[Cell] = [
    Cell("1",  "Qwen3-4B vanilla full attention", QWEN_BASE, "hf", ["S1","S2","S3"], "none", "none"),
    Cell("2",  "Qwen3-4B + δ-Mem adapter",        QWEN_BASE, "hf", ["S1","S2","S3"], "side-state", "none"),
    Cell("3",  "Qwen3-4B + sliding-window 4K",     QWEN_BASE, "hf", ["S1","S2","S3"], "window", "none"),
    Cell("4",  "Qwen3-4B + SW-4K + δ-Mem",         QWEN_BASE, "hf", ["S1","S2","S3"], "window+side-state", "none"),
    Cell("5",  "Qwen3-4B + SW-2K + δ-Mem",         QWEN_BASE, "hf", ["S1","S2","S3"], "aggressive-window+side-state", "none"),
    Cell("6",  "Qwen3-4B + spec-decode",           QWEN_BASE, "hf", ["S1","S2","S3"], "none", "spec-decode"),
    Cell("7",  "Qwen3-4B + δ-Mem + spec-decode",   QWEN_BASE, "hf", ["S1","S2","S3"], "side-state", "spec-decode"),
    Cell("8",  "Qwen3-4B + SW-4K + δ-Mem + spec-decode", QWEN_BASE, "hf", ["S1","S2","S3"], "window+side-state", "spec-decode"),
    Cell("9a", "Qwen3.5-4B-MTP-GGUF Q4_K_M",       "unsloth/Qwen3.5-4B-MTP-GGUF", "llamacpp", ["S1","S2","S3"], "none", "native-MTP"),
    Cell("9b", "Qwen3.5-4B-MTP-GGUF Q5_K_M",       "unsloth/Qwen3.5-4B-MTP-GGUF", "llamacpp", ["S1","S2","S3"], "none", "native-MTP"),
    Cell("9c", "Qwen3.5-4B-MTP-GGUF Q8_0",         "unsloth/Qwen3.5-4B-MTP-GGUF", "llamacpp", ["S1","S2","S3"], "none", "native-MTP"),
    Cell("10", "Qwen3-4B + StreamingLLM sink+SW-4K + δ-Mem", QWEN_BASE, "hf", ["S1","S2","S3"], "sink+window+side-state", "none"),
    Cell("11", "GatedDeltaNet-2 1.3B reference",   GDN2_REF, "nvlabs", ["S3"], "n/a", "n/a",
         notes="Different base, NC license — descriptive only."),

    # SmolLM3-3B leg
    Cell("T1", "Train δ-Mem adapter for SmolLM3-3B", SMOLLM_BASE, "hf", ["S3"], "n/a", "n/a",
         notes="Pre-cell. Gates 13-16. ≤8 GPU-hour cap."),
    Cell("12", "SmolLM3-3B vanilla baseline",      SMOLLM_BASE, "hf", ["S1","S3"], "none", "none"),
    Cell("13", "SmolLM3-3B + δ-Mem (ours)",        SMOLLM_BASE, "hf", ["S1","S3"], "side-state", "none", blocked_by="T1"),
    Cell("14", "SmolLM3-3B + SW-4K + δ-Mem",       SMOLLM_BASE, "hf", ["S1","S3"], "window+side-state", "none", blocked_by="T1"),
    Cell("15", "SmolLM3-3B + δ-Mem + spec-decode", SMOLLM_BASE, "hf", ["S1","S3"], "side-state", "spec-decode", blocked_by="T1"),
    Cell("16", "SmolLM3-3B + SW-4K + δ-Mem + spec-decode (full compound)",
         SMOLLM_BASE, "hf", ["S1","S3"], "window+side-state", "spec-decode", blocked_by="T1"),
]


def cells_for_stage(stage: Stage) -> List[Cell]:
    return [c for c in CELLS if stage in c.stages]
