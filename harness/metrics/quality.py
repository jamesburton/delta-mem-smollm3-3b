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


# -------- Hard NIH (RULER-style multi-needle with distractors + mapping check)

# Expanded key pool: NATO + animals + minerals; ~50 unique strings, plenty for
# 10–30 needles plus distractors that don't collide.
_HARD_KEY_POOL = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
    "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
    "quebec", "romeo", "sierra", "tango", "uniform", "victor", "whiskey",
    "xray", "yankee", "zulu",
    "ferret", "otter", "lynx", "panda", "tapir", "ibis", "kestrel", "newt",
    "marmot", "skink", "vole",
    "quartz", "agate", "topaz", "jasper", "opal", "spinel", "garnet", "zircon",
    "peridot", "amber", "calcite",
]


@dataclass(frozen=True)
class HardMultiNeedleTask:
    """A harder NIH variant designed to be sensitive to long-context
    quality degradation that vanilla `MultiNeedleTask` saturates over.

    Differences:

    1.  More needles (default 10 vs 3). Recall pressure scales with this.
    2.  Distractor codes are scattered through the context using the same
        code format but introduced as 'log entries', NOT as 'secret code
        for X is …'. A model that just regurgitates every code in the
        document gets low precision.
    3.  Question asks the model to answer in `key: CODE` form. The
        scorer parses the answer for that pattern and checks the
        key→code mapping — credit only when the right key is paired
        with the right code. Spraying every code at the end won't score.
    """
    context: str
    question: str
    needles: List[Needle]
    distractor_codes: List[str]  # codes that appear in context but NOT as needles
    seed: int


def _gen_log_distractor(rng: random.Random) -> str:
    """A code-shaped distractor that looks legitimate but isn't a needle.

    Renders as e.g. ' Audit ID GN47-71H logged at offset 1024. '. Same code
    format as needles, but introduced as logs/audit IDs/ticket numbers so a
    reader paying attention to the *question* ('secret code for KEY') would
    not confuse them.
    """
    code = _gen_code(rng)
    kind = rng.choice(["Audit ID", "Ticket", "Log marker", "Trace token",
                       "Session id", "Request id", "Job ref"])
    offset = rng.randint(100, 999999)
    return f" {kind} {code} logged at offset {offset}. "


def make_hard_multineedle_task(
    *,
    target_tokens: int,
    n_needles: int = 10,
    n_distractors: int = 30,
    seed: int = 0,
    filler_text: Optional[str] = None,
) -> HardMultiNeedleTask:
    """Build a long-context probe with multiple needles AND code distractors.

    The needles use phrasing "The secret code for KEY is CODE." (matching
    `make_multineedle_task` so prompting is similar). Distractors use
    "Audit ID CODE logged at offset N." — same code shape, different role.
    """
    rng = random.Random(seed)
    if n_needles > len(_HARD_KEY_POOL):
        raise ValueError(
            f"n_needles={n_needles} exceeds _HARD_KEY_POOL ({len(_HARD_KEY_POOL)})."
        )
    if filler_text is None:
        filler_text = ("Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 200)

    keys = rng.sample(_HARD_KEY_POOL, n_needles)
    needles = [Needle(k, _gen_code(rng)) for k in keys]

    # Generate distractor codes — guaranteed not to collide with any needle code
    needle_codes = {n.code for n in needles}
    distractor_codes: List[str] = []
    tries = 0
    while len(distractor_codes) < n_distractors and tries < n_distractors * 5:
        c = _gen_code(rng)
        if c not in needle_codes and c not in distractor_codes:
            distractor_codes.append(c)
        tries += 1
    # If we ran out of unique codes, that's fine — short distractors are OK.

    words = filler_text.split()
    if not words:
        raise ValueError("filler_text must be non-empty")
    while len(words) < target_tokens:
        words = words + words

    # Interleave needles and distractors at random positions across the body
    n_inserts = n_needles + len(distractor_codes)
    insertion_points = sorted(rng.sample(range(len(words)), n_inserts))
    inserts: List[str] = []
    # randomly assign which insertion points get needles vs distractors
    needle_indices = set(rng.sample(range(n_inserts), n_needles))
    needle_iter = iter(needles)
    distractor_iter = iter(distractor_codes)
    for i in range(n_inserts):
        if i in needle_indices:
            n = next(needle_iter)
            inserts.append(f" The secret code for {n.key} is {n.code}. ")
        else:
            # distractor uses its own code; rebuild the rendered string with that code
            c = next(distractor_iter)
            kind = rng.choice(["Audit ID", "Ticket", "Log marker", "Trace token",
                               "Session id", "Request id", "Job ref"])
            offset = rng.randint(100, 999999)
            inserts.append(f" {kind} {c} logged at offset {offset}. ")

    for offset, (pos, ins) in enumerate(zip(insertion_points, inserts)):
        words.insert(pos + offset, ins)
    context = " ".join(words)

    # Question — be explicit about the format and the precision constraint
    question = (
        "Read the document above. There are several entries labelled "
        "'The secret code for KEY is CODE.' interleaved with unrelated "
        "log/audit/ticket entries that use the same CODE format. "
        "Recall ONLY the secret codes (ignore the log entries).\n"
        "Answer in the form 'key: CODE' on separate lines, one per key.\n"
        "Keys to recall: " + ", ".join(n.key for n in needles)
    )
    return HardMultiNeedleTask(
        context=context,
        question=question,
        needles=needles,
        distractor_codes=distractor_codes,
        seed=seed,
    )


@dataclass(frozen=True)
class HardMultiNeedleScore:
    """Score for a HardMultiNeedleTask. Captures three orthogonal axes:

    - per_needle_correct: was each needle's KEY correctly mapped to its
      CODE in the answer?
    - distractors_mentioned: how many distractor codes leaked into the
      answer (false positives — model spamming codes it remembered).
    - fraction_correct: fraction of needles correctly mapped.

    The headline metric is `fraction_correct` because that's what scales
    smoothly between 0 and 1; the distractor count gives precision context.
    """
    per_needle_correct: List[bool]
    distractors_mentioned: int
    n_needles: int
    n_distractors: int

    @property
    def fraction_correct(self) -> float:
        return sum(self.per_needle_correct) / max(1, self.n_needles)

    @property
    def recall_all(self) -> bool:
        return all(self.per_needle_correct) if self.per_needle_correct else False

    @property
    def precision_against_distractors(self) -> float:
        """Of the codes the model produced, what fraction was a real needle?

        1.0 = no distractor leaked. 0.5 = half the model's codes were
        distractors. Counts each distractor mention once.
        """
        hits = sum(self.per_needle_correct)
        total = hits + self.distractors_mentioned
        return hits / max(1, total)


# Pattern: "key: CODE" or "key - CODE" or "key = CODE", case-insensitive,
# tolerating whitespace and optional bullet/dash prefixes.
_ANSWER_PAIR_RE = re.compile(
    r"(?P<key>[a-zA-Z]+)\s*[:=\-]\s*(?P<code>[A-Z]{2}\d{2}[-\s]?\d{2}[A-Z])",
    re.IGNORECASE,
)


def score_hard_multineedle(task: HardMultiNeedleTask, answer: str) -> HardMultiNeedleScore:
    """Parse 'key: CODE' pairs from the answer and grade.

    A pair counts as correct iff:
      - the key string matches one of the task's needle keys (case-insensitive)
      - the code matches that specific needle's code (whitespace-tolerant)

    Any 'key: CODE' line whose code matches a *distractor* code (in any
    needle-or-non-needle key) is counted as a distractor mention — this
    flags models that recall codes correctly but pair them with wrong keys
    or that just spray codes at the end of their answer.
    """
    # Build mapping for fast lookup
    key_to_needle = {n.key.lower(): n for n in task.needles}
    needle_codes = {n.code: n.key for n in task.needles}
    distractor_codes = set(task.distractor_codes)

    # Find all "key: CODE" pairs in the answer
    pairs = []
    for m in _ANSWER_PAIR_RE.finditer(answer):
        k = m.group("key").lower()
        raw_code = m.group("code")
        # Normalise: strip whitespace from inside the code, restore canonical
        # hyphen position (XX99-99X). The needle/distractor codes always have
        # the hyphen at position 4.
        c = re.sub(r"\s+", "", raw_code)
        if "-" not in c and len(c) >= 7:
            c = c[:4] + "-" + c[4:]
        pairs.append((k, c.upper()))

    per_needle_correct: List[bool] = []
    matched_keys_correctly: set[str] = set()
    for needle in task.needles:
        # Did the model produce this key paired with its actual code?
        correct = any(
            k == needle.key.lower() and c == needle.code.upper()
            for (k, c) in pairs
        )
        per_needle_correct.append(correct)
        if correct:
            matched_keys_correctly.add(needle.key.lower())

    # Count distractor mentions: any pair whose CODE appears in distractor_codes
    # OR whose code is a needle code but paired with the wrong key.
    distractors_mentioned = 0
    for (k, c) in pairs:
        if c in distractor_codes:
            distractors_mentioned += 1
        elif c in needle_codes:
            # Code is a real needle, but paired with the wrong key?
            if needle_codes[c].lower() != k:
                distractors_mentioned += 1
        # else: code doesn't match anything — model hallucinated. Count as distractor.
        else:
            distractors_mentioned += 1

    return HardMultiNeedleScore(
        per_needle_correct=per_needle_correct,
        distractors_mentioned=distractors_mentioned,
        n_needles=len(task.needles),
        n_distractors=len(task.distractor_codes),
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
