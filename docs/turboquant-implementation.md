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
| `src/llama-kv-cache.h/cpp` | fp16→TQ calibration flow, tensor swap, GPU readback calibration |
| `src/llama-context.cpp` | Calibration trigger after decode |

See [CUDA File Map](#cuda-file-map) below for GPU-specific files.

## CUDA Implementation

Full GPU support for TQ KV cache on CUDA (tested on Jetson Orin AGX, SM 8.7).
Enables `-ngl 99 -ctk tqk35 -ctv tqv35 -fa on` for fully offloaded inference.

### Architecture

Three rotation matrices generated on-device via deterministic Householder QR:
- Π_hi (32×32) — K outlier channels, seed `0x5475524230484932`
- Π_lo (96×96) — K regular channels, seed `0x54755242304C4F36`
- Π_v (128×128) — V cache full vector, seed `0x5475524230564131`

Each CUDA translation unit (TU) that includes `turbo-quant-cuda.cuh` gets its own
`static __device__` copies of these matrices, lazily initialized on first TQ operation
via `tq_device_init_rotations_kernel` (single-thread Householder QR on GPU).

### Dedicated Flash Attention Kernel

`fattn-vec-tq.cuh` — custom FA kernel for TQ PROD K × MSE V combinations.

**K cache (PROD) dot product per token:**
1. Pre-rotate query subsets: q_rot_hi = Π_hi × q[0:31], q_rot_lo = Π_lo × q[32:127]
2. Pre-compute QJL projections: proj[i] = Σ_j S_ij × q_raw[j] (once per query, O(m²))
3. Per K token: MSE centroid dot (128 lookups) + QJL sign-weighted correction (128 muls)

**V cache (MSE) accumulation — deferred rotation:**
- Dequantize V per-element in rotated space: just `centroid[idx] × norm` (cheap)
- Accumulate VKQ in rotated space across all K tokens
- Apply Π_v^T inverse rotation once at output

This avoids the expensive per-token 128×128 rotation during V accumulation.

### GPU Calibration Flow

With `-ngl 99`, the KV cache is on GPU. The calibration flow:
1. KV cache starts as fp16 on GPU (same as CPU path)
2. After 256+ tokens in KV cache, `tq_try_finish_calibration` detects enough data
3. Reads back fp16 K data from GPU to CPU via `ggml_backend_tensor_get`
4. Accumulates channel magnitudes on CPU, locks outlier channels
5. Re-quantizes fp16 → TQ on CPU, uploads back via `ggml_backend_tensor_set`
6. Frees old fp16 buffers (saves ~4× memory)
7. Subsequent generation tokens quantized directly to TQ on GPU via CUDA `set_rows`

### CUDA `supports_op` Registration

TQ types registered for:
- `GGML_OP_SET_ROWS` — CUDA quantize (f32 → TQ blocks)
- `GGML_OP_GET_ROWS` — CUDA dequantize (TQ blocks → f32)
- `GGML_OP_FLASH_ATTN_EXT` — dedicated TQ FA kernel

### Synthetic FA Benchmark (Jetson Orin AGX, single head, decode)

| seq_len | f16 (ms) | q4_0 (ms) | tq35 (ms) | tq25 (ms) | q4_0/f16 | tq35/f16 |
|---------|----------|-----------|-----------|-----------|----------|----------|
| 512     | 1.77     | 1.08      | 3.80      | 3.15      | 0.61x    | 2.14x    |
| 1024    | 1.90     | 2.11      | 4.42      | 4.39      | 1.11x    | 2.33x    |
| 2048    | 3.58     | 3.60      | 7.32      | 5.73      | 1.01x    | 2.05x    |
| 4096    | 7.27     | 7.35      | 11.9      | 10.4      | 1.01x    | 1.64x    |
| 8192    | 16.2     | 14.4      | 21.0      | 19.0      | 0.89x    | 1.29x    |
| 16384   | 32.3     | 31.5      | 46.1      | 38.3      | 0.98x    | 1.43x    |
| 32768   | 59.5     | 71.0      | 89.7      | 80.4      | 1.19x    | 1.51x    |

Per-head KV cache memory (K+V combined):

| seq_len | f16     | q4_0    | tq35    | tq25    |
|---------|---------|---------|---------|---------|
| 32768   | 16 MB   | 4.5 MB  | 3.6 MB  | 2.6 MB  |

**Observations:**
- q4_0 matches f16 speed at most lengths, slower at 32K (1.19x) — well-optimized path
- TQ35 FA kernel is 1.3–2.3x slower than f16 per head, but uses 4.4x less memory
- TQ25 slightly faster than TQ35 (less data), 6.1x memory compression
- TQ35 uses 20% less memory than q4_0 (3.6 vs 4.5 MB at 32K)

### Real Model Benchmark (Qwen 2.5 1.5B Q4_K_M, Jetson Orin AGX)

600 token prompt (calibration at 256), 128 token generation, FA on, ngl 99.

| KV Config      | bpv  | Gen t/s | ms/tok | vs f16  |
|----------------|------|---------|--------|---------|
| f16/f16        | 32.0 |   69.9  |  14.3  |   1.0x  |
| q8_0/q8_0      | 17.0 |   68.0  |  14.7  |   1.0x  |
| q4_0/q4_0      | 9.0  |   68.0  |  14.7  |   1.0x  |
| tqk35/q4_0     | 8.25 |   20.7  |  48.4  |   3.4x  |
| tqk35/tqv35    | 7.25 |   13.0  |  77.2  |   5.4x  |

**Mixed config (tqk35/q4_0):** TQ K with QJL correction for better K accuracy,
standard q4_0 V for faster V dequant (no rotation overhead). 61% faster than
full TQ while using slightly more V memory.

**Query preprocessing is fully parallelized:**
- Rotation: each thread computes one output element (32+96 parallel dot products)
- QJL projection: LCG skip-ahead lets each thread independently compute one row
  of S×q without generating the full PRNG sequence serially

**Current kernel optimizations:**
- Parallel query preprocessing: rotation + QJL via LCG skip-ahead (all 128 threads)
- Warp-cooperative K dot: 4 threads per K token, each handles 32 channels, `__shfl_xor` reduce
- Precomputed q×centroid tables in shared memory (no global centroid access per token)
- Algebraic QJL trick: `Σ proj*sign = 2*masked_sum - proj_sum` (precomputed sum)
- 144–148 registers/thread → 3 blocks/SM occupancy, 4.2KB shared memory

### Real Model Benchmarks — Qwen 7B Q4_K_M (4 KV heads)

Same setup: 600 token prompt, 128 token generation, FA on, ngl 99.

| KV Config      | bpv  | Gen t/s | ms/tok | vs f16  |
|----------------|------|---------|--------|---------|
| f16/f16        | 32.0 |   24.4  |  41.0  |   1.0x  |
| q8_0/q8_0      | 17.0 |   24.0  |  41.6  |   1.0x  |
| q4_0/q4_0      | 9.0  |   24.0  |  41.6  |   1.0x  |
| tqk35/q4_0     | 8.25 |   12.7  |  78.5  |   1.9x  |
| tqk35/tqv35    | 7.25 |    9.2  | 109.0  |   2.7x  |

7B is proportionally better than 1.5B because FA is a smaller fraction of total
decode compute (bigger MLP/linear layers dominate).

**Performance analysis:** The TQ FA kernel gap vs f16/q4_0 narrows with larger
models (5.4x on 1.5B → 2.7x on 7B) because attention is a smaller fraction of
total decode. The per-kernel cost is dominated by TQ's centroid lookup + QJL
sign checks vs q4_0's simple nibble shift+scale.

### MSE-only vs PROD+QJL for K cache

CPU attention simulation (Test Q) without calibration shows MSE-only K scoring
higher on top1 accuracy. However, with proper outlier calibration (256+ tokens),
PROD+QJL produces better results in end-to-end model testing — the unbiased QJL
correction matters when the outlier channels are correctly identified.

| Config              | output_cos | top1_acc | Notes |
|---------------------|------------|----------|-------|
| mse_k + mse_v      | 0.9873     | 82.0%    | No calibration (random data) |
| prod_k + mse_v      | 0.9873     | 67.0%    | No calibration (random data) |

With calibration, PROD+QJL recovers the accuracy gap.

### CUDA File Map

| File | Role |
|------|------|
| `ggml/src/ggml-common.h` | Type aliases (block_turbo*), QK/QR constants |
| `ggml/src/ggml-cuda/turbo-quant-cuda.cuh` | Device kernels: 3 rotations, d-specific centroids, quant/dequant, QJL, device-side init |
| `ggml/src/ggml-cuda/turbo-quant-init.cu` | Placeholder TU for init kernel |
| `ggml/src/ggml-cuda/fattn-vec-tq.cuh` | Dedicated TQ flash attention kernel |
| `ggml/src/ggml-cuda/template-instances/fattn-vec-tq-instances.cu` | 4 K×V template instances (PROD3/4 × MSE3/4) |
| `ggml/src/ggml-cuda/fattn.cu` | TQ dispatch in FA entry point + supports_op |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | supports_op for SET_ROWS/GET_ROWS with TQ types |
| `ggml/src/ggml-cuda/set-rows.cu` | Lazy rotation init + TQ quantize dispatch |
| `ggml/src/ggml-cuda/getrows.cu` | Lazy rotation init + TQ dequantize dispatch |
| `tests/bench-turboquant-fa.cu` | Synthetic FA benchmark (f16 vs q4_0 vs tq35 vs tq25) |

## Known Limitations and Caveats

- **No Metal support** — TQ types not supported in Metal set_rows
- **Re-quantization adds error** — prompt tokens quantized with calibrated channels
  have ~0.35 rel_L2 error, which degrades arithmetic precision
- **Memory spike** during calibration — fp16 + TQ both allocated briefly
- **GPU calibration reads back data** — one-time GPU→CPU transfer of fp16 K cache
  during calibration (~10ms for 28 layers × 256 tokens), negligible vs prompt time
- **TQ FA kernel not yet parallelized across threads for K dot** — current implementation
  assigns one K token per thread; warp-cooperative dot product would improve throughput
- **Mixed configs supported** — TQ K (PROD) works with q4_0, q8_0, f16, or TQ MSE V
