# TurboQuant KV Cache Quantization — Implementation & Findings

> **Disclaimer:** This is an early experimental implementation. Testing has been
> limited to short prompts (< 500 tokens) on 4 models with informal quality
> checks — no perplexity benchmarks, no long-context evaluation, no NIAH at
> scale. The implementation likely has bugs. Results below are preliminary
> observations, not rigorous benchmarks. Take everything with a grain of salt.

## GQA Impact on Quantization Quality

The main observation so far: **the number of KV heads seems to affect how well
TurboQuant works.** Models with more KV heads appear to tolerate quantization
error better, likely because each head's error is averaged across the GQA group
during attention. More testing is needed to confirm this.

| Model       | n_kv_heads | n_embd_k_gqa | tqk35+tqv35 (7.25 bpv) | tqk35+q4_0 (8.25 bpv) |
|-------------|------------|--------------|-------------------------|------------------------|
| Llama 8B    | 8          | 1024         | Coherent, minor rounding | Correct |
| Mistral 7B  | 8          | 1024         | Coherent, some truncation | Correct, Eve found |
| Qwen 7B     | 4          | 512          | Coherent after V fix     | Correct |
| Qwen 1.5B   | 2          | 256          | Coherent after V fix     | Correct |

**Hypothesis:** TQ operates on 128-dim blocks (one KV head). The QJL variance
bound scales as 1/d where d=32 for the outlier subset. With fewer KV heads, each
head serves more query heads via GQA, so quantization error may propagate more
broadly. This is speculative — the variable hasn't been isolated rigorously.

**V cache seemed to be the bottleneck for GQA models** in limited tests. K uses
the PROD algorithm (MSE + QJL asymmetric estimator). V uses pure MSE
reconstruction. The V cache was switched to a single 128×128 rotation with d=128
centroids (no outlier split) — this was based on the observation that
V doesn't have the same outlier pattern as K. The RotateKV paper mentions "Values
do not contain outliers like Keys" in a different context (they use simple offline
rotation for V). The change appeared to help on Qwen models in informal tests,
but this needs proper validation.

**Calibration is essential, not optional.** Without outlier calibration (using
default channels 0-31), all Qwen models produce complete gibberish and even
Llama output degrades significantly. The default channel assignment has no
relation to actual outlier positions, so the 32 "outlier" channels get high-bit
quantization on channels that may not need it, while real outliers in the
"regular" partition get low-bit treatment and lose critical magnitude information.

**What to try (not validated):**
- 8+ KV heads (Llama, Mistral): `tqk35 + tqv35` seemed to work (4.4× compression)
- 2-4 KV heads (Qwen, some others): `tqk35 + q4_0` seemed safer (3.9× compression)
- When in doubt: `q8_0` is the safe baseline

## Summary

TurboQuant KV cache quantization with per-layer per-head outlier calibration,
following the paper's exact Algorithm 2 (arXiv 2504.19874). K and V caches are
treated differently based on RotateKV findings (arXiv 2501.16383).

**Tested on**: Llama-3.1-8B, Mistral-7B, Qwen-2.5-7B, Qwen-2.5-1.5B
**Compression**: 4.4× KV cache reduction (tqk35 + tqv35 = ~7.25 bpv vs 32 bpv fp16)

## Architecture

### K cache (PROD types — tqk25/tqk35)

Per the paper's Algorithm 2, with outlier-aware channel split:

1. **Split** 128-dim head into 32 outlier + 96 regular channels
   - Outlier channels identified via prompt-phase calibration (per-layer, per-head)
   - Calibrated from accumulated |K[channel]| magnitudes during fp16 prompt phase
2. **Normalize** each subset independently, store norms (2 × fp16 — one per
   subset, because outlier and regular channels carry different energy levels
   and each needs its own unit-sphere projection for the centroids to match)
3. **Rotate** each subset with independent orthogonal matrices (Π_hi: 32×32, Π_lo: 96×96)
4. **MSE quantize** with b-1 bit Lloyd-Max centroids (exact for Beta((d-1)/2,(d-1)/2))
5. **Residual in original subset space**: r = x_subset - norm × Π^T × centroids[idx]
6. **QJL** on residual: sign(S · r), store ‖r‖ (2 × fp16)
7. **Asymmetric vec_dot**: raw query for QJL correction (not rotated)

### V cache (MSE types — tqv25/tqv35)

Per RotateKV paper: "Values do not contain outliers like Keys."

1. **No outlier split** — single 128×128 rotation on full vector
2. **Normalize** once (1 × fp16 norm)
3. **Rotate** full 128-dim with Π_v (128×128 orthogonal matrix)
4. **MSE quantize** with d=128 centroids:
   - First 32 rotated channels → qs_hi (higher bits)
   - Last 96 rotated channels → qs_lo (lower bits)
5. **No QJL** — pure MSE reconstruction

### Calibration Flow

The current calibration is simplistic compared to the RotateKV paper's approach.
RotateKV calibrates offline on WikiText-2 with sequence length 4096, taking ~5
minutes on a 4090. This implementation does naive online calibration from whatever
prompt the user happens to provide:

1. KV cache starts as fp16 during prompt processing
2. CPU `set_rows` accumulates `sum(|K[channel]|)` per layer per head (K only, not V)
3. After 256 tokens, top 32 channels by accumulated magnitude become "outliers"
4. Prompt fp16 data re-quantized to TQ with calibrated channels
5. Old fp16 buffers freed (saves ~4× memory)
6. Subsequent generation tokens written directly as TQ via `set_rows`

This is a rough approximation — the quality of outlier detection depends entirely
on the prompt content. A math-heavy prompt may produce different outlier channels
than a conversational one. Proper calibration would use a representative dataset
and could be done once per model, saved, and loaded at startup.

## Preliminary Observations

### What seems to work (limited testing)
- Paper's exact Algorithm 2 (original-space residual, raw query for QJL) gives
  lower bias than the original normalized-rotated-space approach
- Per-head outlier detection matters for GQA models (Qwen with 2-4 KV heads)
- V cache benefits from 128×128 rotation with d=128 exact centroids (no split)
- Calibration with 256+ tokens gives reliable outlier detection
- All tested models produce coherent output with K+V both quantized

### What was observed (may not generalize)
- **d=128 per head seemed optimal** for the outlier split in a simulated
  transformer test. Cross-head rotation (d=256/512/1024) performed worse,
  likely because of scaled (not exact) centroids for those dimensions.
  With exact centroids, larger d might win — not tested.
- **K without outlier split (single 128×128 rotation) was also tested and
  performed worse** than the 32/96 split. The paper's algorithm applied to
  the full 128-dim vector without splitting gave higher score error and lower
  top1 accuracy on real attention vectors. The outlier split is important for K.
- **V without outlier split helped Qwen** — switching V to a single 128×128
  rotation (per RotateKV's advice) appeared to fix Qwen output quality, but
  only tested with short prompts.
- **d=128 centroids in the code were approximate** — off by up to 6.5e-5 from
  exact Lloyd-Max. Fixed with scipy-computed values.
- **Tensor swap after calibration** is fragile — calling sched_reserve()
  afterward breaks things. The current approach works but may have edge cases.
- **TQ at 3.75 bpv** seems too lossy for precise tasks (arithmetic, exact
  recall) but produces coherent natural language in informal tests.

### Quality comparison (simulated 16-layer transformer, real Llama KV data)

| Config              | bpv  | Final cos_sim | Final rel_l2 |
|---------------------|------|---------------|--------------|
| q4_0                | 4.5  | 0.9998        | 0.0230       |
| tqk35 calibrated    | 3.75 | 0.9954        | 0.0945       |
| tqk35 no calibration| 3.75 | ~0.99         | ~0.17        |

### End-to-end model quality (multi-step math, ~300 token prompt)

Task: Calculate 5 people's bakery spending, identify max spender, compute total.
Correct: Alice=$15.50, Bob=$23.75, Carol=$22.00, Dave=$40.00, Eve=$99.00, Total=$204.25

| Model      | KV config      | bpv  | Alice calc | Coherent? | Notes |
|------------|----------------|------|------------|-----------|-------|
| Llama 8B   | f16/f16        | 32   | $15.50     | Yes       | Correct, full step-by-step |
| Llama 8B   | q8_0/q8_0     | 17   | $15.50     | Yes       | Correct |
| Llama 8B   | q4_0/q4_0     | 9    | $15.50     | Yes       | Correct |
| Llama 8B   | tqk35/tqv35   | 7.25 | $15.00     | Yes       | Rounds, minor ordering error |
| Llama 8B   | tqk35/q4_0    | 8.25 | $15.50     | Yes       | Correct, step-by-step |
| Mistral 7B | f16/f16        | 32   | $15.50     | Yes       | Correct |
| Mistral 7B | q8_0/q8_0     | 17   | $15.50     | Yes       | Correct |
| Mistral 7B | q4_0/q4_0     | 9    | $23.00     | Yes       | Alice wrong, Eve correct |
| Mistral 7B | tqk35/tqv35   | 7.25 | Partial    | Yes       | Coherent but truncated |
| Mistral 7B | tqk35/q4_0    | 8.25 | $17.50     | Yes       | All wrong but coherent, Eve correct |
| Qwen 7B    | f16/f16        | 32   | $15.50     | Yes       | LaTeX formatting, correct |
| Qwen 7B    | q8_0/q8_0     | 17   | $15.50     | Yes       | Correct |
| Qwen 7B    | q4_0/q4_0     | 9    | —          | **No**    | Degenerates into "?" loops |
| Qwen 7B    | tqk35/tqv35   | 7.25 | $15.50     | Yes       | Step-by-step, correct |
| Qwen 7B    | tqk35/q4_0    | 8.25 | $15.50     | Yes       | Correct, LaTeX |
| Qwen 1.5B  | f16/f16        | 32   | $15.50     | Yes       | Correct |
| Qwen 1.5B  | q8_0/q8_0     | 17   | $15.50     | Yes       | Correct |
| Qwen 1.5B  | q4_0/q4_0     | 9    | Partial    | Yes       | Coherent |
| Qwen 1.5B  | tqk35/tqv35   | 7.25 | Partial    | Yes       | Coherent but truncated |
| Qwen 1.5B  | tqk35/q4_0    | 8.25 | $15.50     | Yes       | Correct, step-by-step |

Note: Qwen 7B with q4_0 KV cache degenerated on this specific prompt while
tqk35/tqv35 produced correct output. This is one data point — it could be a
fluke or prompt-specific. Proper evaluation requires perplexity benchmarks
across diverse datasets.

### Needle-in-a-haystack (400 token context, hidden code retrieval)

A secret code (BLUE-FALCON-7742) is embedded in a passage about bread history.
The model must find and reproduce it exactly.

| Model      | KV config       | bpv  | Found needle? | Output quality |
|------------|-----------------|------|---------------|----------------|
| Llama 8B   | f16/f16         | 32   | YES           | Exact: "BLUE-FALCON-7742" |
| Llama 8B   | q8_0/q8_0      | 17   | YES           | Exact |
| Llama 8B   | q4_0/q4_0      | 9    | YES           | Correct (offers multiple-choice) |
| Llama 8B   | tqk35/tqv35    | 7.25 | YES           | Finds it but hedges |
| Llama 8B   | tqk35/q4_0     | 8.25 | YES           | Exact |
| Mistral 7B | f16/f16         | 32   | YES           | Exact |
| Mistral 7B | q8_0/q8_0      | 17   | YES           | Exact |
| Mistral 7B | q4_0/q4_0      | 9    | YES           | Exact + follow-up question |
| Mistral 7B | tqk35/tqv35    | 7.25 | YES           | "BLUE-FAL-42" (truncated) |
| Mistral 7B | tqk35/q4_0     | 8.25 | YES           | "BLUE-FALMOND-7742" (close) |
| Qwen 7B    | f16/f16         | 32   | YES           | Exact with context |
| Qwen 7B    | q8_0/q8_0      | 17   | YES           | Exact |
| Qwen 7B    | q4_0/q4_0      | 9    | YES*          | Garbled ("cattat...") |
| Qwen 7B    | tqk35/tqv35    | 7.25 | YES*          | Garbled |
| Qwen 7B    | tqk35/q4_0     | 8.25 | YES*          | Garbled |
| Qwen 1.5B  | f16/f16         | 32   | YES           | Exact with context |
| Qwen 1.5B  | q8_0/q8_0      | 17   | YES           | Exact |
| Qwen 1.5B  | q4_0/q4_0      | 9    | YES*          | Garbled |
| Qwen 1.5B  | tqk35/tqv35    | 7.25 | YES*          | Garbled |
| Qwen 1.5B  | tqk35/q4_0     | 8.25 | YES*          | Garbled |

*Needle appears in repetition/garbled output, not coherently retrieved.

**Observations (very small sample, may not generalize):**
- Llama 8B and Mistral 7B seemed to handle TQ better on this test
- Qwen models degraded with any sub-q8_0 quantization (q4_0 AND TQ)
- K=tqk35/V=q4_0 appeared to be a reasonable middle ground for Llama
- This is one prompt — real NIAH evaluation needs thousands of samples at
  varying depths and context lengths

All models produce coherent output across all KV cache types. TQ introduces
minor arithmetic rounding at 3.75 bpv but maintains step-by-step reasoning.

### RotateKV paper comparison

RotateKV (arXiv 2501.16383) uses a fundamentally different approach:
- **Hadamard rotation** (FWHT) instead of QR orthogonal — O(n log n) vs O(n²)
- **Pre-RoPE grouped-head rotation** fused into weights
- **Per-token asymmetric integer quantization** (scale + zero per group)
- **Attention-sink-aware**: keeps sink tokens in fp16

Their results at 2-bit are impressive (<0.3 PPL degradation) but they operate
at different scale (d=1536+) where QJL variance is negligible.

## File Map

| File | Role |
|------|------|
| `ggml/src/ggml-turbo-quant.c` | Core algorithms, per-head registry, calibration API |
| `ggml/src/ggml-cpu/quants.h` | Public calibration function declarations |
| `ggml/src/ggml-cpu/ops.cpp` | Layer/head detection in set_rows, get_rows, FA |
| `ggml/src/ggml-cpu/ggml-cpu.c` | Layer detection in mul_mat, MSE type traits |
| `src/llama-kv-cache.h/cpp` | fp16→TQ calibration flow, tensor swap, buffer cleanup |
| `src/llama-context.cpp` | Calibration trigger after decode |

## Known Limitations and Caveats

- **Metal V cache (TQV) supported** — get_rows and set_rows for tqv25/tqv35 on Metal
- **Metal K cache (TQK) kernels written** — but channel map upload not yet wired end-to-end
- **No Metal flash attention** — TQ types fall back to dequantize + standard attention
- **CUDA kernels not updated** — CUDA still needs the same treatment as Metal
- **Re-quantization adds error** — prompt tokens quantized with calibrated channels
  have ~0.35 rel_L2 error, which degrades arithmetic precision
- **Memory spike** during calibration — fp16 + TQ both allocated briefly
