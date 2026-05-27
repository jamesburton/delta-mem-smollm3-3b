"""Quality metrics for the test matrix.

- `make_multineedle_task`: build a long-context NIH probe with N needles.
- `score_multineedle`: grade a model's free-text answer against the needles.
- `compute_perplexity`: streaming ppl over a tokenised corpus (lazy import torch).
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class Needle:
    key: str    # e.g. "alpha"
    code: str   # e.g. "XK7-92Q"


@dataclass(frozen=True)
class MultiNeedleTask:
    context: str
    question: str
    needles: List[Needle]
    seed: int


_KEY_POOL = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
    "golf", "hotel", "india", "juliet", "kilo", "lima",
]


def _gen_code(rng: random.Random) -> str:
    letters = rng.choices("ABCDEFGHJKLMNPQRSTUVWXYZ", k=2)
    digits = rng.choices("0123456789", k=2)
    suffix = rng.choices("ABCDEFGHJKLMNPQRSTUVWXYZ", k=1)
    return f"{''.join(letters)}{''.join(digits)}-{rng.randint(10,99)}{''.join(suffix)}"


def make_multineedle_task(
    *,
    target_tokens: int,
    n_needles: int = 3,
    seed: int = 0,
    filler_text: Optional[str] = None,
) -> MultiNeedleTask:
    rng = random.Random(seed)
    if n_needles > len(_KEY_POOL):
        raise ValueError(
            f"n_needles={n_needles} exceeds _KEY_POOL size ({len(_KEY_POOL)}). "
            "Extend _KEY_POOL or reduce n_needles."
        )
    if filler_text is None:
        filler_text = ("Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 200)

    keys = rng.sample(_KEY_POOL, n_needles)
    needles = [Needle(k, _gen_code(rng)) for k in keys]

    # Tokens approximated as words; good enough for synth NIH.
    words = filler_text.split()
    if not words:
        raise ValueError("filler_text must be non-empty")
    while len(words) < target_tokens:
        words = words + words

    insertion_points = sorted(rng.sample(range(len(words)), n_needles))
    for offset, (pos, n) in enumerate(zip(insertion_points, needles)):
        sentence = f" The secret code for {n.key} is {n.code}. "
        # offset shifts the index for each insertion already done
        words.insert(pos + offset, sentence)

    context = " ".join(words)
    question = (
        "Read the document above and recall the secret codes. "
        "Answer in the form 'key: CODE' on separate lines, one per key.\n"
        "Keys to recall: " + ", ".join(n.key for n in needles)
    )
    return MultiNeedleTask(context=context, question=question, needles=needles, seed=seed)


@dataclass(frozen=True)
class MultiNeedleScore:
    per_needle: List[bool]
    recall_all: bool
    recall_any: bool

    @property
    def fraction(self) -> float:
        return sum(self.per_needle) / max(1, len(self.per_needle))


def score_multineedle(task: MultiNeedleTask, answer: str) -> MultiNeedleScore:
    hits = []
    for n in task.needles:
        # Code must appear in the answer; tolerate whitespace and case variation
        # Tolerate any whitespace (newline, tab, multiple spaces) in place
        # of the hyphen, since LLMs often line-wrap structured answers.
        pattern = re.escape(n.code).replace(r"\-", r"[-\s]?")
        hits.append(bool(re.search(pattern, answer, flags=re.IGNORECASE)))
    return MultiNeedleScore(
        per_needle=hits,
        recall_all=all(hits),
        recall_any=any(hits),
    )


def compute_perplexity(
    model,
    tokenizer,
    text: str,
    stride: int = 1024,
    max_length: int = 4096,
) -> float:
    """Stride-windowed perplexity. Lazy import torch so module-import works on
    machines without it."""
    import torch  # noqa: WPS433

    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"][0]
    device = next(model.parameters()).device
    nlls = []
    prev_end = 0
    for begin in range(0, input_ids.size(0), stride):
        end = min(begin + max_length, input_ids.size(0))
        trg_len = end - prev_end
        if trg_len <= 1:
            break
        ids = input_ids[begin:end].unsqueeze(0).to(device)
        target = ids.clone()
        target[:, :-trg_len] = -100
        with torch.no_grad():
            out = model(ids, labels=target)
        nlls.append(out.loss.float() * trg_len)
        prev_end = end
        if end == input_ids.size(0):
            break
    if not nlls:
        return float("nan")
    return float(torch.exp(torch.stack(nlls).sum() / prev_end).cpu())
