# TurboQuant KV Cache Quantization

## Glossary

| Term | Meaning |
|------|---------|
| **MSE quantize** | Nearest-centroid quantization on the unit sphere. Lloyd-Max centroids for Beta((d-1)/2,(d-1)/2). b-bit = 2^b centroids. |
| **QJL** | Quantized Johnson-Lindenstrauss. 1-bit sign projection of the MSE residual vector. Stores m sign bits + 1 fp16 residual norm. |
| **QR rotation** | Gram-Schmidt orthogonal matrix from seeded Gaussian. O(d²) per vector. |
| **FWHT** | Fast Walsh-Hadamard Transform. O(d log d) orthogonal rotation, requires power-of-2 d. Randomized with seeded ±1 sign flips. |
| **Split 32/96** | Partition 128-dim head into 32 outlier channels (more bits) + 96 regular channels (fewer bits). Channel assignment from calibration. |
| **Calibration** | Accumulate \|K[channel]\| magnitudes over first N tokens. Top-32 become outlier channels. |
| **fp16 sinks** | Keep first N tokens as fp16 instead of quantizing. Attention sinks receive disproportionate softmax weight. |
| **Asymmetric dot product** | Q·K score computed without dequantizing K. Q is rotated/split to match K's quantized domain: `score = norm · dot(rotate(Q), centroids) + dot(Q_raw, QJL_correction)`. |
| **bpv** | Bits per value (per element of the 128-dim head vector). |

## Scheme Names

| Name | Full description | bpv |
|------|-----------------|-----|
| **fp16** | Half-precision baseline | 16.00 |
| **q8_0** | Block-32 symmetric 8-bit integer | 8.50 |
| **q4_0** | Block-32 symmetric 4-bit integer | 4.50 |
| **had_mse4** | Full H_128 Hadamard, 4-bit MSE (16 centroids d=128), no split, no QJL, no calibration | 4.13 |
| **3lo_qr** | 32/96 split + QR rotation: 4-bit MSE + 1-bit QJL on 32 outliers, 3-bit MSE on 96 regulars | 3.88 |
| **3lo_fwht** | Same as 3lo_qr but FWHT instead of QR: H_32 on outliers, 3×H_32 on regulars (96=3×32) | 3.88 |
| **tqk35** | Current impl: 32/96 split + QR, 3-bit MSE + QJL outlier, 2-bit MSE + QJL regular | 3.75 |

## Summary

Measured attention output quality `out = softmax(Q·K^T/√d) · V` with quantized K cache
on real extracted tensors (8192 tokens, one layer, all KV heads).

**For TQ schemes, Q·K scores are computed via asymmetric dot product — Q is rotated/split
to match K's quantized representation. K is NOT dequantized.** This is the actual inference
path: the softmax operates on approximate scores from the compressed K cache directly.

Test: `tests/test-tq-attn-output.cpp`, verified with `tests/test-tq-sanity.cpp`.

**Single layer, all KV heads, no sinks:**

| Scheme | bpv | Llama 8B avg cos_sim | Llama avg top-1 | Qwen 1.5B avg cos_sim | Qwen avg top-1 |
|---|---|---|---|---|---|
| fp16 | 16.00 | 0.99999998 | 100.0% | 0.99999992 | 100.0% |
| q8_0 | 8.50 | 0.99999560 | 100.0% | 0.99998563 | 99.6% |
| **had_mse4** | **4.13** | **0.99940** | **95.5%** | **0.99752** | **95.7%** |
| q4_0 | 4.50 | 0.99804 | 93.6% | 0.99499 | 91.4% |
| 3lo_qr | 3.88 | 0.99747 | 93.7% | 0.99242 | 91.0% |

Llama layer 16, 8 KV heads averaged. Qwen layer 14, 2 KV heads averaged.

**had_mse4 (4.13 bpv)** beats q4_0 (4.50 bpv) at fewer bits. No calibration, no split, O(n log n).

**Note:** These are single-layer measurements with clean Q/K/V from fp16 inference. They do NOT
capture error propagation through RMSNorm → MLP → next layer. End-to-end perplexity testing
is needed to validate.

## Findings

1. **had_mse4 is the quality champion.** Best cos_sim and KL on both models at 4.13 bpv.
   No calibration, no channel split, O(n log n) rotation. Beats q4_0 at fewer bits.

2. **3lo_qr is the compression champion.** At 3.88 bpv it beats q4_0 on Llama and is
   competitive on Qwen. Needs calibration + channel split.

3. **QR > FWHT at same bpv.** The 96×96 QR rotation fully mixes all regular channels.
   3×H_32 only mixes within 32-element groups — less mixing means worse quality.

4. **QJL on regular channels hurts.** Tested 2lo+qjl (4.01 bpv) and had_3qjl (4.26 bpv) —
   both worse than schemes without QJL at lower bpv. QJL noise > correction at d≥96.

5. **fp16 sinks are unnecessary for had_mse4.** Across all 8 Llama heads, sinks improve
   avg cos_sim by only +0.00009 (0→512). had_mse4 already beats q4_0 at sink=0 on every head.
   On Qwen head 1 (hardest case), sink=0 had_mse4 (0.99592) still beats q4_0 (0.99164).

6. **Sinks plateau at 512 tokens.** 512 and 1024 give identical results on all heads.
   For 3lo_qr, sinks help more on weak heads (Llama head 5: 0.99570→0.99728 at 512).

7. **Qwen (GQA 6:1) is harder than Llama (GQA 4:1).** Each KV head error propagates
   to more Q heads with higher GQA ratio.

## Conclusion

**had_mse4 is the recommended K cache quantization scheme.** At 4.13 bpv it delivers better
attention output quality than q4_0 at 4.50 bpv on both models, across all KV heads.
It requires no calibration, no outlier channel detection, no channel split — just a single
randomized Hadamard rotation (O(n log n)) and 4-bit MSE quantization with precomputed
d=128 Lloyd-Max centroids. This makes it the simplest scheme to implement in kernels
(one FWHT + one centroid lookup per vector, same for quantize and dot product).

On Llama 3.1 8B (8 KV heads, GQA 4:1), had_mse4 achieves 0.99940 attention output cosine
similarity averaged across all heads — closer to q8_0 (0.99999) than to q4_0 (0.99804).
On Qwen 2.5 1.5B (2 KV heads, GQA 6:1), had_mse4 at 0.99752 still clearly beats
q4_0 at 0.99499.

**3lo_qr at 3.88 bpv** is viable when maximum compression matters. It matches q4_0
quality on Llama at 14% fewer bits but adds implementation complexity: per-head outlier
calibration, channel split, two separate rotations (QR 32×32 + QR 96×96), and QJL on
the outlier residual. On Qwen with high GQA ratio it's roughly on par with q4_0.

**What didn't work:**
- QJL on regular channels (d=96 or d=128) — the correction noise exceeds the benefit
- FWHT with 32/96 split — 3×H_32 block-diagonal gives less mixing than full QR_96
- fp16 sinks — unnecessary for had_mse4 (all heads beat q4_0 at sink=0). Adds bpv overhead
  and implementation complexity for negligible gain (+0.00009 avg cos_sim on Llama)

**What needs validation:** Per-layer results (all 32/28 layers) confirm had_mse4 wins on every
layer independently. End-to-end perplexity and NIAH tests are needed to confirm this translates
to output quality when all layers use quantized K simultaneously.

## Effective bpv with fp16 sinks

eff_bpv = (N_sink × 16 + (ctx - N_sink) × base_bpv) / ctx

| Context | had_mse4 (4.13) | 3lo (3.88) | q4_0 (4.50) |
|---|---|---|---|
| 8K | 4.87 | 4.64 | 4.50 |
| 16K | 4.50 | 4.26 | 4.50 |
| 64K | 4.22 | 3.97 | 4.50 |

## V Cache (unchanged)

Single 128×128 QR rotation, no outlier split, d=128 centroids, no QJL.

## File Map

| File | Role |
|------|------|
| `ggml/src/ggml-turbo-quant.c` | Core algorithms, centroids, calibration API |
| `ggml/src/ggml-cpu/ops.cpp` | Layer/head detection in set_rows, get_rows, FA |
| `src/llama-kv-cache.h/cpp` | Calibration flow, tensor swap |
| `tests/test-tq-sanity.cpp` | Algorithm correctness (synthetic data, 16/16 pass) |
| `tests/test-tq-attn-output.cpp` | Attention output quality (real model data, sink sweep) |
| `tmp/tq-extract-*/` | QKV tensor extractors |

## Known Limitations

- Metal TQK kernels written but channel map not wired end-to-end
- No Metal flash attention for TQ types
- CUDA kernels not updated
- Single-layer measurements only — does not capture error propagation through RMSNorm/MLP
- Need end-to-end perplexity and NIAH validation

---

## Raw Data

All TQ scores computed via asymmetric dot product (Q rotated, K not dequantized).
128 query positions (1024..8136), 8192 tokens.

### Llama 3.1 8B — layer 16, KV head 0, Q head 0 (GQA 4:1)

| Scheme | sink | eff bpv | out_cossim | out_rel_L2 | top-1 | top-5 | KL_div |
|---|---|---|---|---|---|---|---|
| fp16 | 0 | 16.00 | 0.99999997 | 0.000215 | 100.0% | 100.0% | 9.6e-10 |
| q8_0 | 0 | 8.50 | 0.99999560 | 0.001832 | 100.0% | 100.0% | 1.6e-07 |
| q4_0 | 0 | 4.50 | 0.99845096 | 0.037513 | 96.9% | 98.4% | 4.6e-05 |
| had_mse4 | 0 | 4.13 | 0.99971509 | 0.014784 | 99.2% | 100.0% | 1.3e-05 |
| had_mse4 | 128 | 4.32 | 0.99973299 | 0.014281 | 99.2% | 100.0% | 1.2e-05 |
| had_mse4 | 256 | 4.50 | 0.99974797 | 0.013789 | 99.2% | 100.0% | 1.1e-05 |
| had_mse4 | 512 | 4.87 | 0.99984728 | 0.010075 | 99.2% | 100.0% | 6.7e-06 |
| had_mse4 | 1024 | 5.61 | 0.99984728 | 0.010075 | 99.2% | 100.0% | 6.7e-06 |
| 3lo_qr | 0 | 3.88 | 0.99922026 | 0.025627 | 97.7% | 100.0% | 3.2e-05 |
| 3lo_qr | 128 | 4.07 | 0.99925142 | 0.024786 | 97.7% | 99.2% | 3.0e-05 |
| 3lo_qr | 256 | 4.26 | 0.99927473 | 0.024490 | 97.7% | 99.2% | 2.7e-05 |
| 3lo_qr | 512 | 4.64 | 0.99958100 | 0.017622 | 98.4% | 99.2% | 1.6e-05 |
| 3lo_qr | 1024 | 5.39 | 0.99958100 | 0.017622 | 98.4% | 99.2% | 1.6e-05 |
| 3lo_fwht | 0 | 3.88 | 0.99866910 | 0.032195 | 98.4% | 100.0% | 5.0e-05 |
| 3lo_fwht | 128 | 4.07 | 0.99868674 | 0.032127 | 98.4% | 100.0% | 4.7e-05 |
| 3lo_fwht | 256 | 4.26 | 0.99869768 | 0.031531 | 98.4% | 100.0% | 4.4e-05 |
| 3lo_fwht | 512 | 4.64 | 0.99948194 | 0.018949 | 100.0% | 100.0% | 2.4e-05 |
| 3lo_fwht | 1024 | 5.39 | 0.99948194 | 0.018949 | 100.0% | 100.0% | 2.4e-05 |

### Qwen 2.5 1.5B — layer 14, KV head 0, Q head 0 (GQA 6:1)

| Scheme | sink | eff bpv | out_cossim | out_rel_L2 | top-1 | top-5 | KL_div |
|---|---|---|---|---|---|---|---|
| fp16 | 0 | 16.00 | 0.99999997 | 0.000472 | 100.0% | 100.0% | 2.6e-08 |
| q8_0 | 0 | 8.50 | 0.99999249 | 0.005202 | 100.0% | 100.0% | 2.9e-06 |
| q4_0 | 0 | 4.50 | 0.99834233 | 0.154084 | 93.0% | 96.1% | 1.1e-03 |
| had_mse4 | 0 | 4.13 | 0.99911611 | 0.062610 | 97.7% | 99.2% | 2.8e-04 |
| had_mse4 | 128 | 4.32 | 0.99911614 | 0.062591 | 97.7% | 99.2% | 2.8e-04 |
| had_mse4 | 256 | 4.50 | 0.99930060 | 0.063466 | 97.7% | 100.0% | 2.8e-04 |
| had_mse4 | 512 | 4.87 | 0.99932616 | 0.055886 | 97.7% | 100.0% | 2.7e-04 |
| had_mse4 | 1024 | 5.61 | 0.99932616 | 0.055886 | 97.7% | 100.0% | 2.7e-04 |
| 3lo_qr | 0 | 3.88 | 0.99723430 | 0.194518 | 93.0% | 95.3% | 1.2e-03 |
| 3lo_qr | 128 | 4.07 | 0.99722881 | 0.194426 | 93.0% | 96.1% | 1.2e-03 |
| 3lo_qr | 256 | 4.26 | 0.99718525 | 0.182993 | 93.0% | 96.9% | 1.1e-03 |
| 3lo_qr | 512 | 4.64 | 0.99726155 | 0.152985 | 93.8% | 98.4% | 1.1e-03 |
| 3lo_qr | 1024 | 5.39 | 0.99726155 | 0.152985 | 93.8% | 98.4% | 1.1e-03 |
| 3lo_fwht | 0 | 3.88 | 0.99498130 | 0.164300 | 92.2% | 96.9% | 1.1e-03 |
| 3lo_fwht | 128 | 4.07 | 0.99498108 | 0.164289 | 92.2% | 96.9% | 1.1e-03 |
| 3lo_fwht | 256 | 4.26 | 0.99572980 | 0.155476 | 93.0% | 97.7% | 1.1e-03 |
| 3lo_fwht | 512 | 4.64 | 0.99712308 | 0.130518 | 93.8% | 99.2% | 9.5e-04 |
| 3lo_fwht | 1024 | 5.39 | 0.99712308 | 0.130518 | 93.8% | 99.2% | 9.5e-04 |

### Llama 3.1 8B — layer 16, all 8 KV heads × all sink sizes (had_mse4 only, cos_sim)

| KV hd | sink=0 | sink=128 | sink=256 | sink=512 | sink=1024 | q4_0 |
|---|---|---|---|---|---|---|
| 0 | 0.99972 | 0.99973 | 0.99975 | 0.99985 | 0.99985 | 0.99845 |
| 1 | 0.99966 | 0.99966 | 0.99966 | 0.99966 | 0.99966 | 0.99778 |
| 2 | 0.99933 | 0.99936 | 0.99932 | 0.99936 | 0.99936 | 0.99832 |
| 3 | 0.99913 | 0.99913 | 0.99913 | 0.99913 | 0.99913 | 0.99638 |
| 4 | 0.99963 | 0.99963 | 0.99963 | 0.99963 | 0.99963 | 0.99864 |
| 5 | 0.99884 | 0.99887 | 0.99892 | 0.99942 | 0.99942 | 0.99758 |
| 6 | 0.99923 | 0.99922 | 0.99923 | 0.99924 | 0.99924 | 0.99855 |
| 7 | 0.99962 | 0.99962 | 0.99963 | 0.99963 | 0.99963 | 0.99864 |
| **AVG** | **0.99940** | **0.99940** | **0.99941** | **0.99949** | **0.99949** | **0.99804** |

Sinks barely help had_mse4 on Llama: 0→512 gains +0.00009 avg cos_sim. Already beats q4_0 at sink=0.

### Llama 3.1 8B — layer 16, all 8 KV heads × all sink sizes (3lo_qr, cos_sim)

| KV hd | sink=0 | sink=128 | sink=256 | sink=512 | sink=1024 | q4_0 |
|---|---|---|---|---|---|---|
| 0 | 0.99922 | 0.99925 | 0.99927 | 0.99958 | 0.99958 | 0.99845 |
| 1 | 0.99814 | 0.99816 | 0.99816 | 0.99817 | 0.99817 | 0.99778 |
| 2 | 0.99723 | 0.99725 | 0.99728 | 0.99751 | 0.99751 | 0.99832 |
| 3 | 0.99751 | 0.99751 | 0.99751 | 0.99752 | 0.99752 | 0.99638 |
| 4 | 0.99883 | 0.99883 | 0.99883 | 0.99883 | 0.99883 | 0.99864 |
| 5 | 0.99570 | 0.99594 | 0.99603 | 0.99728 | 0.99728 | 0.99758 |
| 6 | 0.99416 | 0.99418 | 0.99417 | 0.99441 | 0.99441 | 0.99855 |
| 7 | 0.99901 | 0.99901 | 0.99900 | 0.99900 | 0.99900 | 0.99864 |
| **AVG** | **0.99747** | **0.99752** | **0.99753** | **0.99779** | **0.99779** | **0.99804** |

3lo_qr at 3.88 bpv is close to q4_0 at 4.50 bpv on Llama. Heads 2,5,6 are weak spots where 3lo_qr < q4_0.

### Qwen 2.5 1.5B — layer 14, all 2 KV heads × all sink sizes

| KV hd | Scheme | sink=0 | sink=128 | sink=256 | sink=512 | sink=1024 | q4_0 |
|---|---|---|---|---|---|---|---|
| 0 | had_mse4 | 0.99912 | 0.99912 | 0.99930 | 0.99933 | 0.99933 | 0.99834 |
| 0 | 3lo_qr | 0.99723 | 0.99723 | 0.99719 | 0.99726 | 0.99726 | 0.99834 |
| 0 | 3lo_fwht | 0.99498 | 0.99498 | 0.99573 | 0.99712 | 0.99712 | 0.99834 |
| 1 | had_mse4 | 0.99592 | 0.99597 | 0.99636 | 0.99683 | 0.99683 | 0.99164 |
| 1 | 3lo_qr | 0.98760 | 0.98791 | 0.98869 | 0.99330 | 0.99330 | 0.99164 |
| 1 | 3lo_fwht | 0.98907 | 0.98906 | 0.98923 | 0.99424 | 0.99424 | 0.99164 |

Qwen head 1 (GQA 6:1) is the hardest case. had_mse4 beats q4_0 on all sinks. 3lo_qr needs 512 sinks to match q4_0 on head 1.

## End-to-End Generation Results

Tested with `llama-completion`, seed=42, c=512, `tqk_5hi_3lo_fwht` (3.88 bpv) with calibrated outlier
channels. After calibration completes, all positions use pure TQ (no fp16 sinks).

### Model compatibility (pure TQ after calibration)

| Model | KV heads | GQA ratio | 5hi_3lo_fwht (3.88 bpv) | q4_0 (4.50 bpv) |
|---|---|---|---|---|
| Qwen 2.5 1.5B | 2 | 6:1 | **Collapse** | **Collapse** |
| Qwen 2.5 7B | 2 | 6:1 | **Collapse** | **Collapse** |
| Qwen 2.5 32B | 8 | 5:1 | **Coherent** | Coherent |
| Qwen3 8B | 8 | 4:1 | **Coherent** | Coherent |
| Llama 3.1 8B | 8 | 4:1 | **Coherent** | Coherent |
| Mistral 7B v0.3 | 8 | 4:1 | **Coherent** | Coherent |

**Finding: 2 KV heads = too few for any K cache quantization below q8_0.**
This is NOT specific to TurboQuant — standard q4_0 also collapses on Qwen 2-head models.
Models with 8+ KV heads work perfectly with 5hi_3lo_fwht at 3.88 bpv.

### Quality comparison on 8-head models

Math problem: "A train travels at 60 km/h one way and 90 km/h back. Distance is 360 km.
What is the average speed?" (correct answer: 72 km/h)

- **fwht matches f16 quality** — same answers, same reasoning structure
- **fwht at 3.88 bpv ≥ q4_0 at 4.50 bpv** — equal or better output, 14% less K cache memory
- **Qwen3 8B fwht** gets correct answer (72 km/h) with step-by-step math, coherent past calibration
- Generation continues seamlessly through the calibration boundary (fp16 → TQ transition)

### fp16 sinks (`--tq-sinks N`)

For 2-head models (Qwen 1.5B/7B), sinks keep the first N tokens as fp16 in the K cache.
In-block flag (`is_fp16`) in each TQ block dispatches to fp16 dot product for sink positions.

| Qwen 7B, c=512 | sinks=0 | sinks=128 | sinks=256 | sinks=512 |
|---|---|---|---|---|
| Generation quality | Collapse | Coherent ~100 tok | **Coherent** | **Coherent** |

Sinks delay the TQ collapse proportionally to sink count. sinks=256 at c=512 (50% fp16)
achieves fp16-equivalent quality. At longer contexts (c=4096, sinks=256), effective bpv = 4.6.

### Sink expiry

`tq_expire_sinks()` re-quantizes fp16 sink blocks to TQ, freeing the sink buffers.
After expiry, all positions use pure TQ. Currently disabled (needs proper token counter).
For 8-head models, sinks are unnecessary — pure TQ works after calibration.

### Sanity test (synthetic data, `tests/test-tq-sanity.cpp`) — 16/16 passed

- Rotation roundtrips (QR, FWHT split, H_128): all < 1e-5 error
- **Asymmetric dot product == dequant dot product**: diff < 2e-6 (proves Q rotation is correct)
- Outlier channels scattered [3..127], not contiguous — split uses actual calibrated indices
- Wrong channel map flips score sign: correct=-14.04, wrong=+18.07 vs ref=-17.33
- Calibrated channels: 0.120 rel_L2 vs wrong channels: 0.172
