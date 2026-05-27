import re
import random

from harness.metrics import quality


def test_make_multineedle_task_has_three_needles():
    task = quality.make_multineedle_task(
        target_tokens=2000,
        n_needles=3,
        seed=42,
        filler_text="The quick brown fox jumps over the lazy dog. " * 200,
    )
    # The constructed context should contain each needle's code
    assert len(task.needles) == 3
    for n in task.needles:
        assert n.code in task.context
    # The question prompt should mention every needle's key
    for n in task.needles:
        assert n.key in task.question


def test_score_multineedle_recovers_all_codes_from_perfect_answer():
    task = quality.make_multineedle_task(
        target_tokens=1000, n_needles=3, seed=7,
        filler_text="filler " * 500,
    )
    # Synthesize a perfect answer naming every code
    perfect = "\n".join(f"{n.key}: {n.code}" for n in task.needles)
    score = quality.score_multineedle(task, perfect)
    assert score.recall_all is True
    assert score.recall_any is True
    assert score.per_needle == [True, True, True]


def test_score_multineedle_partial_recovery():
    task = quality.make_multineedle_task(
        target_tokens=1000, n_needles=3, seed=11,
        filler_text="filler " * 500,
    )
    # Answer that gets only the first code
    partial = f"{task.needles[0].key}: {task.needles[0].code}\nothers: unknown"
    score = quality.score_multineedle(task, partial)
    assert score.recall_all is False
    assert score.recall_any is True
    assert score.per_needle == [True, False, False]
