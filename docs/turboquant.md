# TurboQuant KV Cache Quantization

TurboQuant is a family of KV cache quantization types for llama.cpp that compress the Key and Value caches during inference. Based on the TurboQuant paper (arXiv 2504.19874) and RotateKV approach (IJCAI 2025), it uses Hadamard rotation, Lloyd-Max MSE quantization, QJL correction, and outlier-aware channel splitting to achieve 2.75–5.25 bits per value with minimal quality loss.

## Types

### Uniform types (no calibration needed)

These apply a full H_128 Hadamard rotation to all channels, then quantize uniformly.

| Type | bpv | Design |
|---|---|---|
| `tqk4_0` | 4.13 | 4-bit MSE, 16 Lloyd-Max centroids |
| `tqk5_0j` | 5.25 | 4-bit MSE + 1-bit QJL correction |
| `tqk4_1j` | 4.25 | 3-bit MSE + 1-bit QJL correction |
| `tqv4_0` | 4.13 | V cache, same as tqk4_0 |

### Split types (calibration required)

These split channels into 32 outliers and 96 regulars based on calibrated per-layer-per-head channel permutations. Each subset gets independent FWHT rotation (H_32 for outliers, 3×H_32 for regulars) and quantization.

| Type | bpv | Outliers (32 ch) | Regulars (96 ch) |
|---|---|---|---|
| `tqk4_sj` | 4.13 | 5-bit MSE + 1-bit QJL | 3-bit MSE |
| `tqk3_sj` | 3.88 | 4-bit MSE + 1-bit QJL | 3-bit MSE |
| `tqk3b_sj` | 3.75 | 3-bit MSE + 1-bit QJL | 2-bit MSE + 1-bit QJL |
| `tqk2_sj` | 2.75 | 2-bit MSE + 1-bit QJL | 1-bit MSE + 1-bit QJL |

Naming convention: `tqk{approx_bpv}_{variant}` where `s` = split, `j` = QJL correction.

### Block layout and bpv calculation

Each type stores a fixed number of bytes per block of 128 elements. `bpv = block_bytes × 8 / 128`.

**Uniform types** (128 elements per block):

```
tqk4_0 (66B = 4.13 bpv):
  norm         2B (fp16 L2 norm)
  qs         64B (128 × 4-bit MSE indices)

tqk5_0j (84B = 5.25 bpv):
  norm         2B    rnorm        2B (QJL residual norm)
  qs         64B (128 × 4-bit)
  signs      16B (128 × 1-bit QJL signs)

tqk4_1j (68B = 4.25 bpv):
  norm         2B    rnorm        2B
  qs         48B (128 × 3-bit)
  signs      16B (128 × 1-bit)
```

**Split types** (128 elements = 32 outliers + 96 regulars):

```
tqk4_sj (66B = 4.13 bpv):
  norm_hi      2B    norm_lo      2B    rnorm_hi     2B
  qs_hi      20B (32 × 5-bit)
  qs_lo      36B (96 × 3-bit)
  signs_hi    4B (32 × 1-bit)

tqk3_sj (62B = 3.88 bpv):
  norm_hi      2B    norm_lo      2B    rnorm_hi     2B
  qs_hi      16B (32 × 4-bit)
  qs_lo      36B (96 × 3-bit)
  signs_hi    4B (32 × 1-bit)

tqk3b_sj (60B = 3.75 bpv):
  norm_hi      2B    norm_lo      2B    rnorm_hi     2B    rnorm_lo     2B
  qs_hi      12B (32 × 3-bit)
  qs_lo      24B (96 × 2-bit)
  signs_hi    4B (32 × 1-bit)
  signs_lo   12B (96 × 1-bit)

tqk2_sj (44B = 2.75 bpv):
  norm_hi      2B    norm_lo      2B    rnorm_hi     2B    rnorm_lo     2B
  qs_hi       8B (32 × 2-bit)
  qs_lo      12B (96 × 1-bit)
  signs_hi    4B (32 × 1-bit)
  signs_lo   12B (96 × 1-bit)
```

**For comparison — standard types** (32 elements per block):

```
q4_0  (18B = 4.50 bpv):  scale 2B + 32×4-bit 16B
q4_1  (20B = 5.00 bpv):  scale 2B + min 2B + 32×4-bit 16B
q8_0  (34B = 8.50 bpv):  scale 2B + 32×8-bit 32B
```

The fp16 norms are overhead — at low bit rates they become significant (e.g. 8B of norms in tqk2_sj's 44B total = 18%).

## How it works

### 1. Hadamard rotation

Before quantization, K vectors are rotated using the Fast Walsh-Hadamard Transform (FWHT). This spreads channel energy more uniformly, reducing the dynamic range that the quantizer must handle. The rotation is orthogonal and self-inverse (H = H^-1 after normalization).

For uniform types: a single H_128 rotation on all 128 channels.
For split types: independent H_32 on the 32 outlier channels, and 3×H_32 (block-diagonal) on the 96 regular channels.

### 2. MSE quantization

Each rotated, normalized channel value is mapped to the nearest Lloyd-Max centroid. The centroids are precomputed optimal quantization levels for the Beta((d-1)/2, (d-1)/2) distribution on [-1, 1], which is the marginal distribution of unit-sphere coordinates.

- Outlier channels (32-dim, normalized by their own L2 norm): use d=32 centroids
- Regular channels (96-dim, normalized by their combined 96-dim L2 norm): use d=96 centroids

The number of centroids determines the MSE bit rate: 2 centroids = 1-bit, 4 = 2-bit, 8 = 3-bit, 16 = 4-bit, 32 = 5-bit.

### 3. QJL correction (Quantized Johnson-Lindenstrauss)

For types with the `j` suffix, a 1-bit QJL correction is applied to the MSE residual. QJL projects the quantization residual through a random Gaussian matrix and stores only the signs. This provides an **unbiased** estimator of the residual contribution to the dot product, at the cost of 1 additional bit per channel.

The QJL estimator satisfies: E[<q, QJL(r)>] = <q, r>, meaning the expected dot product is exact. The variance decreases with dimension.

### 4. Channel splitting (split types)

The key insight from RotateKV: not all K channels are equally important. Some channels have much higher variance across tokens and carry more distinguishing information for attention. Applying uniform quantization wastes bits on low-importance channels while under-representing critical ones.

Split types separate channels into:
- **Outliers** (top 25% by importance): get more bits and QJL correction
- **Regulars** (bottom 75%): get fewer bits

The channel assignment is determined by offline calibration (see below).

### 5. Asymmetric dot product

During Flash Attention, the Q·K dot product is computed **asymmetrically**: Q is rotated to match K's quantized domain, and the dot product is evaluated directly against centroids without dequantizing K. This avoids full dequantization and enables efficient GPU kernel implementation.

For QJL types, the correction term is added: `dot = MSE_dot + QJL_dot`.

## Calibration

Split types require offline calibration to determine which channels are outliers for each (layer, KV head) pair.

### Running calibration

```bash
# Download calibration data
bash scripts/get-ptb.sh

# Calibrate (default: variance metric, pre-RoPE)
llama-tq-calibrate -m model.gguf -f ptb/ptb.train.txt -o perms.bin \
    --pre-rope -n 32000 -c 2048
```

### What it does

1. Runs a forward pass on calibration text with f16 K cache
2. Intercepts K activations via `cb_eval` callback
3. For each (layer, KV head, channel): accumulates statistics
4. Computes channel importance and sorts descending
5. Top 25% = outlier channels, bottom 75% = regular channels
6. Saves per-layer-per-head permutations to a binary file (~14 KB for a 28-layer model)

### Metrics

- `--metric var` (default): Var(K) per channel. High-variance channels carry distinguishing information between tokens. A constant-value channel doesn't affect attention ranking regardless of magnitude.
- `--metric mag`: Mean |K| per channel. Simpler but less effective.
- `--metric both`: Geometric mean of magnitude and standard deviation.

Variance consistently outperforms magnitude across models tested.

### Pre-RoPE vs Post-RoPE

- `--pre-rope`: Captures K before RoPE is applied. Recommended default.
- `--post-rope`: Captures K after RoPE. May be better on some models (e.g. Qwen 2.5 1.5B).

The channel importance ordering is model-specific but generalizes across calibration datasets. PTB train set works well for calibrating models evaluated on WikiText-2.

### Calibration file format

Binary file with header:
```
[magic:4 "TQPE"][version:4][n_layers:4][n_kv_heads:4][head_dim:4][pre_rope:4][n_layer_model:4]
[layer_map: n_layer_model × 4 bytes]
[permutations: n_layers × n_kv_heads × head_dim bytes (uint8)]
```

Each permutation row: first `head_dim/4` entries are outlier channel indices, rest are regular.

## Usage

```bash
# Uniform types (no calibration)
llama-completion -m model.gguf -ctk tqk4_0 -ctv tqv4_0 -p "Hello" -n 32

# Split types (with calibration)
llama-completion -m model.gguf -ctk tqk4_sj -ctv tqv4_0 \
    --tq-perms perms.bin -p "Hello" -n 32

# Perplexity evaluation
llama-perplexity -m model.gguf -f wikitext-2-raw/wiki.test.raw \
    -ctk tqk4_sj -ctv tqv4_0 --tq-perms perms.bin --chunks 20

# Speed benchmark
llama-bench -m model.gguf -ctk tqk4_sj -ctv tqv4_0 -fa 1 -t 10
```

## Which type to use?

| Model sensitivity | Recommended | bpv | Notes |
|---|---|---|---|
| Low (Mistral, Llama 3.1) | `tqk4_0 + tqv4_0` | 4.13 | No calibration needed, fastest |
| Medium (Qwen3) | `tqk4_sj + tqv4_0` | 4.13 | Calibration helps slightly |
| High (Qwen 2.5) | `tqk4_sj + tqv4_0` | 4.13 | Calibration essential — 21x better than q4_0 |
| Memory-constrained | `tqk3b_sj + tqv4_0` | 3.94 | Aggressive but usable on robust models |

## Benchmark summary

Results across 4 models on Apple M4 Pro (Metal) and NVIDIA RTX A6000 (CUDA). PPL = WikiText-2 perplexity (lower is better). Calibrated on Penn Treebank train set. All TQ K types paired with tqv4_0 V; standard types use matching V.

### PPL comparison (selected types)

| Model | f16 | q8_0 | q4_0 (4.50) | q4_1 (5.00) | tqk4_0 (4.13) | tqk4_sj (4.13) |
|---|---|---|---|---|---|---|
| **Qwen 2.5 7B** | 7.78 | 7.78 | 6820 | 12256 | 5069 | **328** |
| **Qwen3 8B** | 10.92 | 10.90 | 11.32 | 10.97 | 11.43 | **11.12** |
| **Mistral 7B** | 7.32 | 7.31 | 7.29 | 7.34 | 7.36 | 7.53 |

### Key findings

1. **Outlier-sensitive models (Qwen 2.5)**: `tqk4_sj` is the only viable sub-8.5 bpv option. At PPL 328 it's 21x better than q4_0 (6820) and 37x better than q4_1 (12256) at comparable or lower bpv. Without calibration, split types degrade to uniform-type levels.

2. **Robust models (Qwen3, Mistral, Llama 3.1)**: All KV quantization types work well. `tqk4_sj` at +1.7% PPL is competitive with q4_1 at +0.5% while using 17% fewer bits. For these models, `tqk4_0` (no calibration needed) is the pragmatic choice.

3. **Calibration impact**: On Qwen 2.5 7B, variance-based calibration (Var(K)) reduces `tqk4_sj` PPL from 2804 (no calibration) to 223 (calibrated) — an 12x improvement. Calibration generalizes across datasets (train on PTB, evaluate on WikiText-2).

4. **Speed**: TQ uniform types (`tqk4_0`) are 3% slower on prompt processing and 14% slower on text generation vs f16. Split types (`tqk4_sj`) are 5% slower on pp and 27% slower on tg. Standard quantization (q4_0, q4_1) has negligible speed impact.

5. **Memory savings**: At 4.13 bpv (K+V), TurboQuant achieves 3.9x KV cache compression. For a 7B model with 128K context, KV cache drops from 16.4 GiB to 4.2 GiB.

For detailed per-model results, see [Metal benchmarks](turboquant-metal-benchmark.md) and [CUDA benchmarks](turboquant-cuda-benchmark.md).

## Backend support

| Backend | FA | get_rows | set_rows |
|---|---|---|---|
| CPU | All types | All types | All types |
| Metal | All d=128 types | All d=128 types | All d=128 types |
| CUDA | All d=128 types | All d=128 types | All d=128 types |

d=256 types have partial support (CPU full, GPU get_rows/set_rows only).

## References

- TurboQuant: arXiv 2504.19874, https://arxiv.org/pdf/2504.19874
- RotateKV: IJCAI 2025, https://www.ijcai.org/proceedings/2025/0690.pdf
- QJL (Quantized Johnson-Lindenstrauss): Definition 1 in the TurboQuant paper
- Penn Treebank calibration data: Marcus et al., Computational Linguistics 19(2), 1993
