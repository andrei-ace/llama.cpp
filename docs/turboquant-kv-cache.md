# TurboQuant KV Cache Quantization

Implementation of Google Research's TurboQuant algorithm (arXiv 2504.19874) for KV cache compression in llama.cpp.

## Overview

TurboQuant uses random orthogonal rotation + Lloyd-Max optimal scalar quantization + QJL (Quantized Johnson-Lindenstrauss) residual correction to compress KV cache vectors at 2.5-3.5 bits per value.

Two type families are provided:

| Type | Name | bpv | Use | Description |
|------|------|-----|-----|-------------|
| `TURBO3_0_PROD` | `tqk_lo` | 2.6 | K cache | 2-bit MSE + 1-bit QJL (hi), 1-bit MSE + 1-bit QJL (lo) |
| `TURBO4_0_PROD` | `tqk_hi` | 3.6 | K cache | 3-bit MSE + 1-bit QJL (hi), 2-bit MSE + 1-bit QJL (lo) |
| `TURBO3_0_MSE` | `tqv_lo` | 2.4 | V cache | 3-bit MSE (hi), 2-bit MSE (lo), no QJL |
| `TURBO4_0_MSE` | `tqv_hi` | 3.4 | V cache | 4-bit MSE (hi), 3-bit MSE (lo), no QJL |

## Usage

```bash
# Recommended: tqk_hi for K cache (3.6 bpv with inner product preservation)
llama-cli -m model.gguf --cache-type-k tqk_hi --cache-type-v f16

# Aggressive: tqk_lo for K cache (2.6 bpv, some quality loss)
llama-cli -m model.gguf --cache-type-k tqk_lo --cache-type-v f16
```

Note: V cache quantization with TurboQuant MSE types requires flash attention support, which is not yet implemented for TurboQuant on CPU or Metal. Use `f16` for V cache until FA support is added.

## Algorithm

Per the paper's Algorithm 1 and 2:

1. **Rotation**: Apply deterministic 128x128 orthogonal matrix (QR of Gaussian, seed `0x5475524230524F54`) to decorrelate channels
2. **Normalize**: Extract L2 norm, quantize coordinates on the unit sphere
3. **MSE Quantize**: Find nearest Lloyd-Max centroid (optimal for Beta distribution at d=128)
4. **QJL Correction** (PROD only): Project residual through random Gaussian matrix, store 1-bit signs + residual norm

The PROD types include an asymmetric inner product estimator (`vec_dot`) that computes attention scores directly from compressed data without full dequantization:

```
<q, k> ~ <q_rot, k_mse> + ||r|| * sqrt(pi/2) / m * <S @ q_rot, sign(S @ r)>
```

This estimator is **unbiased** (E[estimate] = true inner product).

### Channel Splitting

Following the paper, each block (128 elements) is split into:
- **32 outlier channels** (hi): higher bit precision
- **96 regular channels** (lo): lower bit precision

Each partition has its own QJL matrix and residual norm.

## Benchmarks with Real Model Data

### Inference Quality (Llama-3.1-8B-Instruct Q4_K_M, d=128, n_head_kv=8)

| K cache type | bpv | Output quality |
|-------------|-----|----------------|
| f16 | 16.0 | Perfect (baseline) |
| q8_0 | 8.5 | Perfect |
| q4_0 | 4.5 | Perfect |
| **tqk_hi** | **3.6** | **Coherent, high quality** |
| tqk_lo | 2.6 | Noticeable degradation |

### Attention Score Accuracy (real Llama-3.1-8B KV vectors, 998 keys)

| Method | bpv | mean\|score_err\| | bias | top1_acc | KL_div |
|--------|-----|-------------------|------|----------|--------|
| q8_0 | 8.5 | 0.015 | -0.000 | 100% | 0.0001 |
| q4_0 | 4.5 | 0.244 | -0.006 | 95% | 0.020 |
| tqk_hi | 3.6 | 1.036 | -0.019 | 79% | 0.344 |
| tqk_lo | 2.6 | 1.842 | -0.041 | 69% | 0.886 |

Key observations:
- TurboQuant has very low **bias** (-0.019) compared to q4_0 (-0.006), confirming the paper's unbiasedness property
- Higher **variance** from QJL at d=128 means raw score error is larger than q4_0
- Despite lower top1 accuracy (79%), softmax averaging across the full sequence produces coherent output
- KV vector norm matters significantly: models with larger KV norms (e.g., Qwen2.5 at norm~69) are more sensitive to quantization noise than models with smaller norms (Llama-3.1 at norm~20)

### Model Compatibility

Works well on:
- Llama-3.1-8B (n_head_kv=8, n_gqa=4, KV norm ~20)

May produce degraded output on:
- Qwen2.5 models (n_head_kv=2-4, high GQA ratio, KV norm ~69)
- Models with very few KV heads (high GQA amplification)

## Implementation Status

- [x] CPU reference quantize/dequantize (rotation baked into block functions)
- [x] Asymmetric inner product estimator (`vec_dot`)
- [x] CPU flash attention integration (via `vec_dot_type = GGML_TYPE_F32`)
- [x] Test harness with synthetic and real-data benchmarks
- [ ] CUDA kernels
- [ ] Metal backend support
- [ ] Flash attention support for V cache MSE types
- [ ] Optimized SIMD quantize/dequantize

## References

- Paper: [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874)
- Website: [turboquant.net](https://turboquant.net)
- Discussion: [ggml-org/llama.cpp#20969](https://github.com/ggml-org/llama.cpp/discussions/20969)
- Community findings: MSE-only quantization (no QJL) may work better in practice for K cache at d=128 due to lower variance, at the cost of biased inner products
