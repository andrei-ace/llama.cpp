# TurboQuant Benchmark Results

All benchmarks from `test-turboquant -bench` on Jetson Orin AGX.
Date: 2026-03-28. Branch: `feature/turboquant-kv-cache`.

Reference: "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate" (ICLR 2026, Zandieh et al.)

---

## Type Map

| CLI name | Internal enum | Algorithm | Purpose | Block layout | Block bytes | bpv |
|----------|---------------|-----------|---------|--------------|-------------|-----|
| `tqk_lo` | `GGML_TYPE_TURBO3_0_PROD` | Algo 2 (prod) | K cache | 3x fp16 norms + hi: 2b MSE + 1b QJL (32 ch) + lo: 1b MSE + 1b QJL (96 ch) | 42 | 2.625 |
| `tqk_hi` | `GGML_TYPE_TURBO4_0_PROD` | Algo 2 (prod) | K cache | 3x fp16 norms + hi: 3b MSE + 1b QJL (32 ch) + lo: 2b MSE + 1b QJL (96 ch) | 58 | 3.625 |
| `tqv_lo` | `GGML_TYPE_TURBO3_0_MSE`  | Algo 1 (mse)  | V cache | 1x fp16 norm + hi: 3b MSE (32 ch) + lo: 2b MSE (96 ch) | 38 | 2.375 |
| `tqv_hi` | `GGML_TYPE_TURBO4_0_MSE`  | Algo 1 (mse)  | V cache | 1x fp16 norm + hi: 4b MSE (32 ch) + lo: 3b MSE (96 ch) | 54 | 3.375 |

Effective data bits per channel (excluding norm overhead): (32 * hi_bits + 96 * lo_bits) / 128.
Prod types store 3 fp16 norms (norm, rnorm_hi, rnorm_lo); MSE types store 1 fp16 norm.

---

## 1. MSE Distortion (Test R)

D_mse = E[||x - x'||^2] for unit-norm vectors, full pipeline with rotation.

Paper bounds (uniform b-bit): b=1: 0.36, b=2: 0.117, b=3: 0.03, b=4: 0.009.

| Type | Effective MSE bpc | Measured D_mse | Paper range | Status |
|------|-------------------|----------------|-------------|--------|
| tqv_lo (MSE, hi=3b lo=2b) | 2.25 | **0.095** | 0.030 - 0.117 | within range |
| tqv_hi (MSE, hi=4b lo=3b) | 3.25 | **0.028** | 0.009 - 0.030 | within range |
| tqk_lo (PROD, hi=2b lo=1b MSE) | 1.25 | 0.460 | — | higher (1 bit spent on QJL) |
| tqk_hi (PROD, hi=3b lo=2b MSE) | 2.25 | 0.146 | — | higher (1 bit spent on QJL) |

MSE variants fall cleanly within the paper's predicted bounds for their effective bpc.
PROD D_mse is 3-5x higher because 1 bit per channel goes to QJL instead of centroids.

---

## 2. MSE vs PROD Reconstruction (Test O)

500 vectors, mixed KV-cache distributions (lognormal+outlier, power-law, correlated, heavy-tail, sparse, asymmetric), with rotation.

| Pair | PROD rel_L2 | PROD cos | MSE rel_L2 | MSE cos | MSE error reduction | MSE win rate |
|------|-------------|----------|------------|---------|---------------------|--------------|
| turbo3 (lo) | 0.676 | 0.827 | **0.309** | **0.951** | **54.3%** | 100% |
| turbo4 (hi) | 0.382 | 0.934 | **0.167** | **0.986** | **56.4%** | 100% |

MSE variant wins on reconstruction quality 100% of the time, confirming the paper's design.

---

## 3. Inner Product Preservation (Test P)

2000 query-key pairs, unit-norm Gaussian vectors, with rotation.

| Pair | PROD mean_err | PROD bias | PROD corr | MSE mean_err | MSE bias | MSE corr |
|------|---------------|-----------|-----------|--------------|----------|----------|
| turbo3 (lo) | 0.0468 | -0.0015 | 0.825 | 0.0220 | -0.0014 | 0.950 |
| turbo4 (hi) | 0.0266 | -0.0009 | 0.935 | 0.0122 | +0.0004 | 0.985 |

Both variants show near-zero bias. At these bit widths (2.25-3.25 bpc), MSE's lower reconstruction error translates to lower per-pair inner product error as well. The PROD QJL correction's unbiasedness advantage is more pronounced at lower bit-widths (b=1) and when averaging over many queries.

Compare with PyTorch reference (uniform b-bit, d=128):
- PyTorch PROD: b=2 bias=+0.001 corr=0.80, b=3 bias=+0.000 corr=0.93, b=4 bias=+0.000 corr=0.98
- Our PROD correlations (0.825, 0.935) align well with the PyTorch b=2 and b=3 results.

---

## 4. Combined K/V Attention Simulation (Test Q)

64 keys, 100 queries, d=128, tqk_hi/tqv_hi bit budget, with rotation.
Full attention pipeline: quantize K and V -> compute scores -> softmax -> weighted sum of V -> compare output to FP32 reference.

| Config | Output cos | Output L2 | Score KL | Top-1 acc |
|--------|-----------|-----------|----------|-----------|
| **prod_k + mse_v (intended)** | **0.9877** | **0.157** | 0.000011 | 66% |
| prod_k + prod_v | 0.9567 | 0.319 | 0.000011 | 66% |
| mse_k + mse_v | 0.9877 | 0.157 | 0.000002 | 85% |
| mse_k + prod_v | 0.9567 | 0.319 | 0.000002 | 85% |

Key findings:
- **V type dominates output quality**: MSE_v gives 0.988 cos vs PROD_v gives 0.957 (output is a weighted sum of values)
- **K type dominates score quality**: MSE_k gives lower KL and higher top-1 than PROD_k at this bit-width
- The intended config (prod_k + mse_v) achieves 98.8% output cosine similarity

---

## 5. Attention Fidelity vs Existing Types (Test K)

64 keys, 200 queries, d=128. No rotation for standard types; with rotation for TurboQuant.

| Type | bpv | Score err | Rank corr | Top-1 acc | Softmax KL |
|------|-----|-----------|-----------|-----------|------------|
| tqk_lo | 2.6 | 0.0673 | 0.797 | 48% | 0.000032 |
| tqk_hi | 3.6 | 0.0377 | 0.916 | 71% | 0.000010 |
| q4_0 | 4.5 | 0.0106 | 0.989 | 93% | 0.000001 |
| q4_1 | 5.0 | 0.0098 | 0.991 | 95% | 0.000001 |
| q5_0 | 5.5 | 0.0054 | 0.997 | 92% | 0.000000 |
| q8_0 | 8.5 | 0.0008 | 1.000 | 99% | 0.000000 |

TurboQuant achieves useful attention fidelity at much lower bit-widths. tqk_hi at 3.6 bpv gets 91.6% rank correlation and 71% top-1, compared to q4_0 at 4.5 bpv which gets 98.9% / 93%. The trade-off is ~25% less storage.

---

## 6. Long-Context Stability (Test K2)

Softmax KL divergence across sequence lengths. Lower = better.

| Type | bpv | 64 keys | 256 keys | 1024 keys | 4096 keys |
|------|-----|---------|----------|-----------|-----------|
| tqk_lo | 2.6 | 0.000037 | 0.000039 | 0.000036 | 0.000034 |
| tqk_hi | 3.6 | 0.000012 | 0.000012 | 0.000012 | 0.000011 |
| q4_0 | 4.5 | 0.000001 | 0.000001 | 0.000001 | 0.000001 |
| q8_0 | 8.5 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

KL is stable or slightly improving with longer contexts (errors average out in softmax), matching the paper's findings.

---

## 7. Adversarial Flat-Attention Stress (Test K3)

Deliberately similar keys where softmax must distinguish small differences.

### Easy (10 random keys)

| Type | bpv | Softmax KL | Top-1 | Needle top-5 |
|------|-----|-----------|-------|--------------|
| tqk_lo | 2.6 | 0.000012 | 54% | 57% |
| tqk_hi | 3.6 | 0.000004 | 66% | 64% |
| q4_0 | 4.5 | 0.000000 | 93% | 71% |
| q8_0 | 8.5 | 0.000000 | 100% | 74% |

### Medium (100 similar keys, noise=0.1)

| Type | bpv | Softmax KL | Top-1 | Needle top-5 |
|------|-----|-----------|-------|--------------|
| tqk_lo | 2.6 | 0.000010 | 92% | 100% |
| tqk_hi | 3.6 | 0.000003 | 100% | 100% |
| q4_0 | 4.5 | 0.000000 | 100% | 100% |

### Hard (1000 similar keys, noise=0.05)

| Type | bpv | Softmax KL | Top-1 | Needle top-5 |
|------|-----|-----------|-------|--------------|
| tqk_lo | 2.6 | 0.000010 | 0% | 0% |
| tqk_hi | 3.6 | 0.000004 | 0% | 24% |
| q4_0 | 4.5 | 0.000000 | 100% | 100% |
| q8_0 | 8.5 | 0.000000 | 100% | 100% |

The hard adversarial scenario (1000 near-identical keys) breaks both TurboQuant tiers but preserves softmax KL. In practice, real attention patterns are not this adversarial.

---

## 8. Seed Sensitivity (Test K4)

Hard scenario (1000 similar keys), sweeping 20 random seeds for rotation + QJL matrices.

| Type | bpv | Mean top-1 | Mean needle-top-5 | Mean KL |
|------|-----|-----------|-------------------|---------|
| tqk_hi | 3.6 | 26% | 45% | 0.000004 |
| q4_0 | 4.5 | 99.8% | 100% | 0.000000 |

Performance varies significantly across seeds (0-100% top-1). Some seeds give perfect retrieval, others fail entirely. Softmax KL is stable across seeds. This suggests the rotation matrix quality matters for fine-grained discrimination.

---

## 9. QJL Diagnostic (Test K5)

Centroid-only vs centroid+QJL for TURBO4_0, 500 unit-norm vectors, no rotation.

| Metric | Centroid-only | Centroid+QJL |
|--------|--------------|--------------|
| Mean rel_L2 | **0.309** | 0.383 |
| Mean score_err | **0.022** | 0.028 |
| Centroid wins score | 56.6% | — |
| **Inner product bias** | **-0.000371** | **-0.000051** |
| Bias reduction | — | **7.3x** |

QJL adds reconstruction noise (higher rel_L2 and score_err per pair) but reduces systematic inner product bias by 7.3x. This is the core trade-off: individual vector fidelity vs statistical unbiasedness.

---

## 10. Paper-Exact Algorithm vs Our Block Implementation (Test L)

64 keys, 100 queries, d=128, mixed KV-cache distributions.

| Method | bpw | Score err | Rank corr | Top-1 | KL |
|--------|-----|-----------|-----------|-------|------------|
| Paper uniform 4-bit | 4.25 | 0.026 | 0.959 | 81% | 0.000005 |
| Paper uniform 2-bit | 2.25 | 0.048 | 0.887 | 63% | 0.000016 |
| Our tqk_hi | 4.25 | 0.043 | 0.907 | 59% | 0.000013 |
| Our tqk_lo | 2.50 | 0.076 | 0.788 | 39% | 0.000039 |
| q4_0 | 4.50 | 0.012 | 0.988 | 83% | 0.000001 |
| q8_0 | 8.50 | 0.001 | 1.000 | 99% | 0.000000 |

Our two-tier implementation trails the paper's uniform algorithm by ~5-10% on rank correlation. The non-uniform channel split (32 hi + 96 lo) is an engineering trade-off for non-integer bit-widths.

---

## 11. d=1536 Simulation (Test M)

Paper's validation dimension. 500 query-key pairs per bit-width.

| b | Paper D_prod | Measured D_prod | Ratio |
|---|-------------|-----------------|-------|
| 2 | 0.000365 | 0.000119 | 0.33x (well within bound) |
| 3 | 0.000117 | 0.000034 | 0.29x (well within bound) |

At d=1536 the distortion is ~12x smaller than at d=128, confirming D_prod scales as O(1/d).

---

## Summary

| Metric | tqk_lo / tqv_lo | tqk_hi / tqv_hi | q4_0 | q8_0 |
|--------|-----------------|-----------------|------|------|
| bpv (K / V) | 2.6 / 2.4 | 3.6 / 3.4 | 4.5 | 8.5 |
| Block bytes (K / V) | 42 / 38 | 58 / 54 | 18 | 34 |
| Compression vs fp16 | 5.3-6.1x | 3.8-4.3x | 3.6x | 1.9x |
| D_mse (V, unit-norm) | 0.095 | 0.028 | — | — |
| Score err (K, d=128) | 0.067 | 0.038 | 0.011 | 0.001 |
| Rank correlation (K) | 0.797 | 0.916 | 0.989 | 1.000 |
| Top-1 accuracy (K) | 48% | 71% | 93% | 99% |
| Softmax KL (K) | 3.2e-5 | 1.0e-5 | 1e-6 | 0 |
| Output cos (K+V attn) | — | 0.988 | — | — |

TurboQuant achieves useful attention fidelity at 2.4-3.6 bpv, providing 4-6x compression over fp16. The MSE variant (for V cache) consistently produces 54-56% lower reconstruction error than the PROD variant. The intended pairing (prod_k + mse_v) yields 98.8% output cosine similarity at the hi tier.
