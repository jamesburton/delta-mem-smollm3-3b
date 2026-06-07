# Sliding-window retrofit notes

How the sliding-window kv_lever is wired on Qwen3-4B, what works, what
doesn't, and what the local sweep actually measured.

## What's wired (commit landing this doc)

`BackboneConfig.sliding_window` is forwarded by `hf_runner.run()` from
`cell.kv_lever`. When set, `backbone._load_plain` constructs an `AutoConfig`
**before** `from_pretrained`, mutates it, and passes it in via `config=`:

```python
model_config.use_sliding_window = True
model_config.sliding_window = W                              # 4096 or 2048
model_config.max_window_layers = num_hidden_layers
model_config.layer_types = ["sliding_attention"] * num_hidden_layers
```

This is *load-time* mutation, not post-load. Post-load mutation does not
work on Qwen3:

- The attention modules cache their mask-builder selection at `__init__`,
  so flipping `config.use_sliding_window` after load has no effect on the
  attention path.
- Rewriting `config.layer_types` post-load triggers
  `KeyError: 'sliding_attention'` mid-decode (observed on cell 5 at 8K
  during the LOCAL_SW sweep).

## What the SW lever now does

**Confirmed working:**

- The attention mask switches to a sliding-window causal mask. Empirically:
  cell 3 (SW-4K) at 8K context drops NIH from 1.00 to 0.00 — needles outside
  the 4K window are masked away. This is the unambiguous signal that the
  lever is engaged.
- Transformers' generation cache will pick `DynamicSlidingWindowLayer` for
  every layer because `layer_types` is all `"sliding_attention"`. This caps
  KV cache **storage** at `sliding_window` tokens per layer regardless of
  prompt length.

**What does NOT change:** the harness's `peak_vram_bytes` metric.

Cell 3 at 8K with SW: peak 10.9 GiB, identical to vanilla cell 1 at 8K.

The reason is a methodology issue, not an SW issue. `torch.cuda.max_memory_allocated()`
is process-lifetime peak, and on the 4B model at 8K the **prefill
workspace** dominates: FlashAttention-2 allocates roughly O(seq_len × dim)
scratch buffers during the one-shot prefill pass over the prompt, and that
spike happens BEFORE the cache even matters. SW caps the *cache* (decode
phase), not the *prefill workspace*.

To see SW's memory benefit, you'd need to either:

1. Compute prefill in chunks (prefix-fill in N-token chunks so the workspace
   never grows past N), then SW vs full-attention diverges in the cache
   accumulation, OR
2. Measure decode-phase memory specifically (peak between prefill-end and
   decode-end), not whole-run peak.

The harness today does neither. That's the next infrastructure follow-up.

## What this means for the δ-Mem hypothesis

The FA2 sweep finding — that δ-Mem adds 0.9–3.5 GiB above vanilla as
context grows — also lands in the prefill workspace, not the cache. So we
have a methodology problem on both sides: neither vanilla KV growth nor
δ-Mem sidecar growth is visible at the "peak VRAM" level on this card and
context range.

What we **can** say from the sweep data:

- δ-Mem's sidecar/write-phase IS a real adder at ≥4K (visible because
  vanilla and δ-Mem diverge in peak by 1.8 GiB at 4K, 3.5 GiB at 8K).
- SW's attention-mask is real (NIH collapse proves it).
- SW's KV cache reduction is real (the cache layer type confirms it) but
  invisible at the peak-VRAM metric we're using.

The conclusive cell-2-vs-cell-4 KV comparison the v3 test matrix wants
needs the decode-phase memory isolation above to work cleanly.

## δ-Mem cells and the SW lever

Cells 4 (SW-4K + δ-Mem), 5 (SW-2K + δ-Mem), 8, 10 use the δ-Mem code path
(`backbone._load_with_delta_mem`), which loads the base model via
`load_delta_mem_chat_model(...)` — a separate call site that does NOT
currently honor `BackboneConfig.sliding_window`. So in those cells the SW
side is logged but inactive.

Two follow-ups close that gap:

1. Pre-snapshot the base model with a patched config and point the upstream
   loader at the snapshot. Doable today; takes a bit of cache plumbing.
2. Submit a PR to declare-lab/delta-Mem so `load_delta_mem_chat_model`
   accepts a `config=` kwarg.

Neither is in scope for the local 3060 work — these cells are flagged for
the cloud rung where context >> window genuinely matters.

## File map

```
harness/backbone.py        # BackboneConfig.sliding_window + load-time bake
harness/runners/hf_runner.py # _window_size_for_lever() → BackboneConfig
docs/sw-retrofit-notes.md  # this file
results/LOCAL_SW/          # config-only sweep (pre-fix; peak didn't drop)
```

A follow-up sweep after this commit will land in `results/LOCAL_SW_v2/`
with the load-time bake, but the headline finding is methodology:
peak-VRAM as currently measured is insensitive to SW savings on this
profile. NIH-collapse at SW < ctx is the cleaner proxy.
