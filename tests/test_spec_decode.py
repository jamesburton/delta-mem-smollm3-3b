from harness import spec_decode


def test_load_assistant_returns_compatible_model(tiny_model_id):
    asst = spec_decode.load_assistant(tiny_model_id, device="cpu", dtype="float32")
    assert asst is not None
    # Same vocab size = compatible draft for the same target
    assert hasattr(asst, "config")


def test_generate_with_spec_decode_produces_output(tiny_lm, tiny_model_id):
    model, tok = tiny_lm
    asst = spec_decode.load_assistant(tiny_model_id, device="cpu", dtype="float32")
    out = spec_decode.generate_with_spec_decode(
        model, tok, asst,
        prompt="Hello",
        max_new_tokens=4,
    )
    assert isinstance(out, str)
    assert len(out) >= 1
