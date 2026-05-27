import time

from harness.metrics import speed


def test_timed_generation_records_prefill_decode_ttft(tiny_lm):
    model, tok = tiny_lm
    record = speed.timed_generation(
        model, tok,
        prompt="Hello",
        max_new_tokens=4,
        seed=0,
    )
    # Sanity: every counter is positive
    assert record.prefill_seconds > 0
    assert record.decode_seconds > 0
    assert record.ttft_seconds > 0
    assert record.new_tokens == 4
    # tok/s sanity (tiny-gpt2 is fast)
    assert record.decode_tokens_per_second > 1


def test_timed_generation_serialises_to_dict(tiny_lm):
    model, tok = tiny_lm
    record = speed.timed_generation(model, tok, prompt="Hi", max_new_tokens=2, seed=0)
    d = record.as_dict()
    assert set(d.keys()) >= {
        "prefill_seconds", "decode_seconds", "ttft_seconds",
        "decode_tokens_per_second", "new_tokens",
    }
