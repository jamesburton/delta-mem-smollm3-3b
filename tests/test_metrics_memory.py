"""Memory-metric tests are mostly analytic so they work on CPU."""

from harness.metrics import memory


def test_kv_bytes_dense_transformer():
    # 36-layer model, 16 KV heads, head_dim 128, 4096 seq, bf16
    n = memory.kv_bytes(
        num_hidden_layers=36, num_key_value_heads=16,
        head_dim=128, seq_len=4096, dtype_bytes=2,
    )
    # 36 * 16 * 128 * 4096 * 2 bytes (K) * 2 (K and V) = 1_207_959_552
    assert n == 36 * 16 * 128 * 4096 * 2 * 2


def test_kv_bytes_with_sliding_window_caps_at_window():
    n_full = memory.kv_bytes(num_hidden_layers=36, num_key_value_heads=16,
                             head_dim=128, seq_len=16384, dtype_bytes=2)
    n_swa = memory.kv_bytes(num_hidden_layers=36, num_key_value_heads=16,
                            head_dim=128, seq_len=16384, dtype_bytes=2, sliding_window=4096)
    assert n_swa < n_full
    assert n_swa == 36 * 16 * 128 * 4096 * 2 * 2


def test_kv_bytes_from_config_uses_correct_attributes():
    class _CfgQwen:
        num_hidden_layers = 28
        num_key_value_heads = 4   # GQA
        hidden_size = 2560
        num_attention_heads = 20
    cfg = _CfgQwen()
    n = memory.kv_bytes_from_config(cfg, seq_len=2048, dtype_bytes=2)
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    expected = cfg.num_hidden_layers * cfg.num_key_value_heads * head_dim * 2048 * 2 * 2
    assert n == expected


def test_kv_bytes_sliding_window_zero_treated_as_window_zero_not_disabled():
    # sliding_window=0 should mean "window of zero tokens", not "no window"
    n = memory.kv_bytes(num_hidden_layers=2, num_key_value_heads=1, head_dim=8,
                        seq_len=1024, dtype_bytes=2, sliding_window=0)
    assert n == 0


def test_kv_bytes_from_config_with_num_key_value_heads_zero_falls_back():
    # Defensive: a malformed config with num_key_value_heads=0 should NOT
    # silently fall back to num_attention_heads — but our current contract
    # is to honour what's set. Document by asserting current behaviour.
    class _CfgBad:
        num_hidden_layers = 2
        num_key_value_heads = 0    # malformed
        num_attention_heads = 4
        hidden_size = 32
    n = memory.kv_bytes_from_config(_CfgBad(), seq_len=128, dtype_bytes=2)
    # 0 KV heads → 0 bytes (honour the value)
    assert n == 0
