# TurboQuant Metal Implementation — Benchmarks & Findings

Hardware: Apple M4 Pro | Metal Flash Attention | wikitext-2 perplexity

## KV Cache Types Implemented

| Type | Side | bpv | Algorithm | Calibration |
|---|---|---|---|---|
| `tqk_had_mse4` | K | 4.13 | Hadamard + 4-bit MSE | No |
| `tqk_had_prod5` | K | 5.25 | Hadamard + 4-bit MSE + 1-bit QJL | No |
| `tqk_had_prod4` | K | 4.25 | Hadamard + 3-bit MSE + 1-bit QJL | No |
| `tqk_5hi_3lo_had` | K | 3.88 | 32/96 channel split + 4×Hadamard | Yes (32+ tokens) |
| `tqv_had_mse4` | V | 4.13 | Hadamard + 4-bit MSE | No |

All K types pair with `tqv_had_mse4` or `q4_0` for V.

## Results: 8+ KV Head Models

### Qwen3-8B (8 KV heads, c=512, 8 chunks)

| K | V | KV bpv | PPL | pp512 | tg128 |
|---|---|---|---|---|---|
| f16 | f16 | 16.00 | 10.50 | 459 | 46.5 |
| q8_0 | q8_0 | 8.00 | 10.49 | 453 | 45.8 |
| q4_0 | q4_0 | 4.50 | 10.89 | 453 | 45.3 |
| **had_mse4** | **tqv** | **4.13** | **10.70** | 444 | 39.3 |
| had_mse4 | q4_0 | 4.31 | 10.83 | 449 | 42.0 |
| 5hi_3lo | tqv | 4.00 | 10.46 | 428 | 33.7 |
| 5hi_3lo | q4_0 | 4.19 | 10.57 | 432 | 35.9 |
| had_prod5 | tqv | 4.69 | 10.67 | 435 | 35.0 |
| had_prod4 | tqv | 4.19 | 12.06 | 436 | 34.3 |

### Llama 3.1-8B (8 KV heads)

| K | V | KV bpv | PPL |
|---|---|---|---|
| f16 | f16 | 16.00 | 7.98 |
| q4_0 | q4_0 | 4.50 | 8.12 |
| **had_mse4** | **tqv** | **4.13** | **8.15** |
| had_mse4 | q4_0 | 4.31 | 8.07 |
| 5hi_3lo | tqv | 4.00 | 8.15 |

### Mistral 7B (32 KV heads) — best case

| K | V | KV bpv | PPL |
|---|---|---|---|
| f16 | f16 | 16.00 | 7.13 |
| q4_0 | q4_0 | 4.50 | 7.11 |
| **had_mse4** | **tqv** | **4.13** | **7.13** |
| had_mse4 | q4_0 | 4.31 | 7.11 |
| 5hi_3lo | tqv | 4.00 | 7.14 |

On Mistral (32 heads): TQ at 4.13 bpv = identical to f16. All quantizations are lossless.

## Results: 2 KV Head Models (Broken)

| Model | heads | f16 | q4_0 | had_mse4+tqv | 5hi_3lo+tqv |
|---|---|---|---|---|---|
| Qwen2.5-1.5B | 2 | 11.16 | 3644 | 7796 | 108 |
| Qwen2.5-7B | 2 | 7.13 | 4652 | 5263 | 142 |

**All KV quantization below q8_0 is catastrophic on 2-head models.** This is not TQ-specific — q4_0 is equally broken. Only q8_0 survives (PPL = f16).

5hi_3lo is the least bad (108-142 vs 3600-7800) because the channel split preserves some precision on outlier channels.

## Calibration Windows (--tq-sinks N)

The `--tq-sinks N` flag controls how many tokens are used for outlier channel detection in `5hi_3lo_had`. Calibration reads the first N tokens' K values and sorts channels by accumulated magnitude.

### 8-head models: calibration window doesn't matter

| Model | sink=32 | sink=128 | sink=256 |
|---|---|---|---|
| Qwen3-8B | 10.46 | 10.52 | 10.48 |
| Llama 3.1-8B | 8.15 | 8.10 | 8.09 |
| Mistral 7B | 7.14 | 7.16 | 7.17 |

32 tokens is sufficient. Channel statistics converge fast on 8+ head models.

### 2-head models: more calibration helps slightly

| Model | sink=32 | sink=128 | sink=256 |
|---|---|---|---|
| Qwen2.5-7B | 142 | 119 | 107 |
| Qwen2.5-1.5B | 108 | 196 | 237 |

Qwen2.5-7B improves with more data (142→107). Qwen2.5-1.5B gets worse — unstable on 2 heads.

## Long Context: Mistral 7B at 16K (single chunk)

| K | V | PPL |
|---|---|---|
| f16 | f16 | 3.594 |
| q8_0 | q8_0 | 3.593 |
| q4_0 | q4_0 | 3.628 |
| 5hi_3lo+tqv | sink=0 | 3.658 |
| 5hi_3lo+tqv | sink=128 | 3.654 |
| 5hi_3lo+tqv | sink=256 | 3.656 |

At real 16K context: TQ is +0.06 above f16. Sink window makes no difference.

## Recommendations

1. **Best overall: `had_mse4 + q4_0`** (KV 4.31 bpv) — best quality, fastest TQ, no calibration
2. **Lowest memory: `had_mse4 + tqv`** (KV 4.13 bpv) — slightly slower (FWHT on output)
3. **Minimum KV heads: 8.** Models with 2 KV heads cannot use any quantization below q8_0
4. **Skip had_prod4** — 3-bit MSE is too lossy (PPL +1.5 over f16)
5. **Skip had_prod5** — QJL adds overhead with no quality gain over had_mse4
6. **5hi_3lo_had** — marginal benefit over had_mse4 for the calibration complexity

## Implementation Notes

- **Metal FA**: Q pre-rotated with fused FWHT in shared memory. K dequantized per-element in rotated space.
- **V quantization**: Inverse FWHT applied to FA output, not per V row (FWHT is linear).
- **Calibration**: K starts as fp16. After threshold tokens, outlier channels detected, K re-quantized to TQ. fp16 buffer freed immediately (separate from V buffer).
- **QJL**: Uses Hadamard (FWHT) as projection matrix for power-of-2 dims. Per-element correction: `K_eff[j] = norm*centroid + (√(π/2)/d)*rnorm*sign[j]`.
- **Channel map**: 5hi_3lo passes per-head permutation to FA kernel via src[5]. Q permuted to [outlier, regular] order in shared memory.
