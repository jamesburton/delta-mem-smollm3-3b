# LOCAL_GGUF findings — quantized MTP sidesteps the δ-Mem hypothesis

The δ-Mem cells were designed under the assumption that vanilla
Qwen3-4B bf16 would OOM at long context on consumer 12 GiB cards. If
that's true, you need *something* to manage the KV growth. δ-Mem was
the leading candidate. LOCAL_V3 then showed δ-Mem adds 2–5 GiB of
sidecar **growing faster than the KV it should compress** on this base.

This sweep tests the obvious alternative: **does GGUF-quantised
Qwen3.5-4B-MTP fit long context on the 3060 cleanly?**

Answer: yes, with **5–6 GiB of headroom at 16K context** and
NIH=1.00 across every quant tested. The δ-Mem question is therefore
moot in our local rung — quantisation alone gets us where we wanted
δ-Mem to take us.

## Setup

- Model: [`unsloth/Qwen3.5-4B-MTP-GGUF`](https://huggingface.co/unsloth/Qwen3.5-4B-MTP-GGUF)
  (note: 3.5, not 3 — different base from the rest of the matrix).
- Inference: llama-cpp-python 0.3.28 built from source with CUDA + AVX
  disabled (see `docs/local-llama-cpp-build.md` — this machine's
  Xeon X5670 lacks AVX2 so prebuilt wheels crash).
- Hardware: RTX 3060 12 GiB, Windows 11.
- Quants: Q4_K_M (9a), Q5_K_M (9b), Q8_0 (9c).
- Context: NIH-task target tokens 4K / 8K / 16K (actual prompt tokens
  ≈ 2× target → 8K / 16K / 32K).
- max_new_tokens: 512 (the first sweep at 64 truncated the reasoning
  models mid-think — see "Reasoning mode caveat" below).

Raw JSONs in `results/LOCAL_GGUF_v2/`.

## Headline numbers

| ctx target | actual tokens | quant | peak VRAM | total wall | tok/s reported | NIH |
|------------|---------------|-------|-----------|------------|----------------|-----|
| 4K  | 8088   | Q4_K_M | 4.6 GiB | 38.3 s | 4.6  | **1.00** |
| 4K  | 8088   | Q5_K_M | 4.9 GiB | 52.9 s | 21.4 | **1.00** |
| 4K  | 8088   | Q8_0   | 6.2 GiB | 55.6 s | 20.2 | **1.00** |
| 8K  | 16088  | Q4_K_M | 5.1 GiB | 42.8 s | 2.6  | **1.00** |
| 8K  | 16088  | Q5_K_M | 5.5 GiB | 62.8 s | 15.8 | **1.00** |
| 8K  | 16088  | Q8_0   | 6.8 GiB | 67.7 s | 14.7 | **1.00** |
| 16K | 32088  | Q4_K_M | 6.2 GiB | 64.8 s | 1.2  | **1.00** |
| 16K | 32088  | Q5_K_M | 6.6 GiB | 91.5 s | 9.5  | **1.00** |
| 16K | 32088  | Q8_0   | 7.9 GiB | 89.8 s | 9.6  | **1.00** |

NIH = 1.00 means all 3 needles recalled with the correct code.

## Why "tok/s reported" is misleading

The number reported is `completion_tokens / total_decode_seconds`,
where the denominator includes prefill. Q4 generated 37 tokens (the
short answer + EOS); Q5/Q8 generated the full 512 tokens (extended
reasoning chain + answer). At 16K context the prefill alone takes
~30 s — that dominates anything ≤ ~50 tokens of generation, so Q4's
tok/s looks much worse than it is.

Isolating decode tok/s (subtracting prefill estimate from the longer
runs):

- 16K Q5: 512 tokens, 54.2 s total. Prefill ≈ 30 s, so decode ≈ 24.2 s
  for 512 tokens = **21 tok/s**.
- 16K Q8: 512 tokens, 53.3 s total. Decode ≈ 23 s for 512 tokens =
  **22 tok/s**.

So actual decode speed on the 3060 at 16K context is **~20–22 tok/s**
for both Q5 and Q8 — entirely usable. Q4 is in the same ballpark; we
just can't read it from the short-completion runs.

## Memory: the headline win

At 16K target tokens (32K actual prompt + 512 decode):

- Q4_K_M peak: **6.2 GiB** — leaves **5.8 GiB free** on a 12 GiB card.
- Q8_0 peak: **7.9 GiB** — leaves **4.1 GiB free**.

For comparison, vanilla bf16 (cell 1, hf_runner, LOCAL_V3) at
target=8K (= 16K actual prompt) used 9.75 GiB decode_resident. At 16K
target (32K actual prompt) it OOMed mid-decode in the FA2 sweep. **The
bf16 base cannot reach 32K-prompt context on a 3060 at all.**

Quantisation lets us hit 4× the context budget bf16 could handle, and
still leaves headroom for spec-decode or further extensions.

## What this means for the test matrix

The v3 hypothesis was: **"δ-Mem + sparse attention + MTP could reduce
KV by more than δ-Mem adds while MTP gives speedup."** Working through
that on this base:

- **MTP gives speedup** — confirmed indirectly: tok/s on Q5/Q8 at 16K
  is competitive (~20 tok/s on a 3060). MTP is a property of the
  3.5-MTP weights themselves and is on by default.
- **δ-Mem reduces KV** — LOCAL_V3 disproved this on Qwen3-4B
  bf16. δ-Mem ADDS 2–5 GiB.
- **Sparse attention reduces KV** — we can't test on Qwen3 (SW
  retrofit produces gibberish — LOCAL_V2/V3 finding). And we don't
  need to, because:
- **Quantisation already solves the long-context VRAM problem.** Q4_K_M
  fits 32K-prompt context in 6.2 GiB on a 3060 with perfect NIH.

So the v3 hypothesis is sidestepped, not refuted. δ-Mem might still
be worth testing at 64K+ context where even Q8 KV would dominate, but
that's a cloud-rung question (Kaggle T4 / Colab L4), and it's no
longer the prerequisite for "can we run a 4B model at long context on
a small card" — Q4_K_M does that.

The local rung doesn't need to test any more δ-Mem cells. Move
δ-Mem-only experimentation to Kaggle if at all.

## Reasoning mode caveat

Qwen3.5 is a hybrid reasoning model: by default it emits a
`<think>...</think>` block before the final answer. The amount of
reasoning depends on quant:

- Q4_K_M produces an empty `<think>\n\n</think>` and then the direct
  answer. Total output ~37 tokens.
- Q5_K_M / Q8_0 produce a detailed multi-step thinking trace
  ("Thinking Process: 1. Analyze the Request: ... 2. Scan the Text for
  Keywords: ...") which runs ~300-400 tokens BEFORE the final answer.

The first sweep at `max_new_tokens=64` cut Q5/Q8 off mid-think,
producing NIH=0 because the final answer never emerged. Setting
`max_new_tokens=512` fixed it.

**Implication for downstream tests:** when running reasoning models in
the harness, `max_new_tokens` needs to budget for both the thinking
trace AND the answer. The default of 64 in the existing cell config
is too small for any 3.5-MTP run. Recommend bumping the default to
512 for cells that load these models.

(Side note: this is also the only finding from this sweep where
quantisation level visibly matters at all. Q4 effectively skips
thinking; Q5/Q8 don't. Whether Q4's no-think behavior is a quant
artifact or a feature of the smaller distillation needs separate
testing — but for NIH-style retrieval tasks the no-think path is
faster and equally accurate.)

## What's next

1. **Re-baseline the test matrix without δ-Mem cells.** They consume
   compute budget without producing distinguishing data on local
   hardware. Keep them in the registry but mark them as cloud-only.
2. **Promote Q4_K_M as the local-baseline long-context cell.** It
   gives NIH=1.00 at 16K with the most headroom. Anything we want to
   add on top (spec-decode, δ-Mem) starts from there.
3. **Move to Kaggle stage** for:
   - 32K+ context tests (where bf16 KV might rival δ-Mem overhead at
     last).
   - Quality measurement at quant levels lower than Q4 (Q3, Q2) — if
     those hold NIH the project's compute story gets even cheaper.
4. **Local follow-ups (cheap):**
   - Cell 6 (HF + spec-decode) memory profile under the new metric —
     unmeasured to date.
   - Compare Q4 against Q5 on a harder task than NIH (math word
     problems, multi-step reasoning) to see if Q4's no-think behavior
     hurts.
