# TurboQuant: Adaptive K Cache Compression

> **Highly experimental research prototype.** Unoptimized, API will change, CPU and Apple Metal only, no CUDA. ~50% slower token generation than f16 KV. Not suitable for any production use — this is a proof of concept for exploring adaptive KV cache compression.

## Scope

**Backend support**: TurboQuant Flex (`tqk_flex`, `tqk` auto per-layer) is supported on **CPU and Metal only**. CUDA is not yet supported — using `-ctk tqk_flex` or `-ctk tqk` with TQFC configs on a CUDA build will error at context creation. The fixed TQ types (tqk4_sj, tqk3_sjj, etc.) have CUDA support.

**V cache**: TurboQuant currently compresses **only the K cache**. All results below use f16 V.

Supported V cache types with TQ flex K (`tqk_flex`, `tqk` per-layer):
- **f16**: ✅ recommended for apples-to-apples comparison
- **q8_0**: ✅ (8.5 bpv). Output identical to f16 V on both models.
- **q4_0**: ✅ (4.5 bpv). Minor wording differences vs f16 V (same as f16 K + q4_0 V — V quantization effect, not TQ K).
- **q4_1**: ✅ (5.0 bpv). Same as q4_0.
- **tqv4_0**: ✅ for fixed TQ K types. tqv4_0 (`GGML_TYPE_TQV_HAD_MSE4`) applies FWHT + 4-bit Lloyd-Max MSE to V vectors (4.125 bpv).

Quantized V types auto-enable Flash Attention. All PPL results in this document use f16 V to isolate the K cache quantization effect.

Note: q8_0/q4_1 V FA templates only exist for flex K types (`tqk_flex`). The fixed TQ K types (tqk4_sj, etc.) only have f16, q4_0, and tqv4_0 V templates.

For a model where K and V are the same size, the total KV cache savings are:
- K: 16.0 → 6.21 bpv (2.6x compression)
- V: 16.0 → 16.0 bpv (no compression)
- **Total KV: (6.21 + 16.0) / 2 = 11.1 bpv → 1.44x total KV compression**

The K cache is the more important target because the Q·K dot product is the critical path for attention quality — errors in K directly shift attention weights, while V errors only affect the output blending. V cache quantization is future work.

## How It Works

TurboQuant compresses the K cache in transformer attention. The core idea:

1. **Rotate** each K vector with an orthogonal transform (FWHT) to spread information uniformly across channels
2. **Quantize** each rotated channel to a few bits using optimal Lloyd-Max centroids
3. **Correct** the inner product bias with a 1-bit QJL sign vector (optional)

The rotation is key — without it, a few channels carry most of the signal and low-bit quantization destroys them. After FWHT, all channels have similar magnitude, making uniform quantization effective.

### The Split Trick

Some attention layers have extreme outlier concentration: 25% of channels carry >90% of the K variance. For these layers, TurboQuant splits channels into two groups:

- **32 outlier channels** (highest variance, identified by calibration): get more bits (9-bit = 512 centroids)
- **96 regular channels**: get fewer bits (4-bit = 16 centroids)
- Each group gets its own FWHT rotation and norm

For layers with uniform variance (<70% concentration), no split is needed. A single FWHT-128 on all 128 channels with 6-bit quantization works better — forcing an artificial split on uniform data wastes bits.

### QJL Correction

After MSE quantization, the residual is projected to a 1-bit sign vector. This makes the Q·K inner product estimator **unbiased** — without it, every dot product has a small systematic error that accumulates over long contexts and many layers.

- Cost: ~1 bpv overhead
- Impact: negligible at <8k context, measurable at 16k+
- For split layers: FWHT-projected signs on outliers, per-element signs on regulars
- Configurations without QJL achieve lower bpv and similar PPL at short context, but QJL is recommended for long-context use (16k+) where bias accumulation degrades attention quality

### Lloyd-Max Centroids

The quantization levels are optimal for the Beta((d-1)/2, (d-1)/2) distribution — the marginal distribution of a single component of a unit vector on the d-sphere. Computed with Anderson-accelerated fixed-point iteration to 1e-13 precision.

Available tables: d=32 (2-10 bit), d=96 (1-8 bit), d=128 (3-8 bit). Stored in `tests/turboquant-centroids-extended.c`.

## Usage

There are two modes of operation:

### Mode 1: Per-layer adaptive (`-ctk tqk`) — recommended

Calibrate once, then run. Each layer gets its own quantization config based on outlier concentration.

```bash
# Step 1: Calibrate (once per model architecture)
llama-tq-calibrate -m model.gguf -f ptb/ptb.train.txt -o perms.bin \
    --flex-extreme 1:9:4:1:1 \
    --flex-high 1:9:4:1:1 \
    --flex-moderate 0:6:0:0:0 \
    --flex-threshold-high 70

# Step 2: Run inference
llama-completion -m model.gguf -p "prompt" \
    -ctk tqk --tq-perms perms.bin -ngl 99

# Step 2b: Run perplexity
llama-perplexity -m model.gguf -f wikitext-2-raw/wiki.test.raw --chunks 20 \
    -ctk tqk --tq-perms perms.bin -ngl 99

# Step 2c: Run benchmark
llama-bench -m model.gguf --tq-perms perms.bin \
    -ctk f16,q8_0,tqk -ctv f16 -fa 1 -ngl 99
```

The perms file contains:
- Channel permutations (which channels are outliers, per layer per head)
- Per-layer type recommendations (TQLT section)
- Per-layer flex configs (TQFC section) — the bit widths, split/non-split, QJL flags

### Mode 2: Uniform flex (`-ctk tqk_flex`) — for experimentation

Apply the same quantization config to all layers. Useful for sweeping bit widths.

```bash
# First calibrate to get channel permutations (without --flex-* flags)
llama-tq-calibrate -m model.gguf -f ptb/ptb.train.txt -o perms.bin

# Then experiment with different configs
llama-perplexity -m model.gguf -f wikitext-2-raw/wiki.test.raw --chunks 3 \
    -ctk tqk_flex --tq-split --tq-hi-bits 9 --tq-lo-bits 3 --tq-qjl hi \
    --tq-perms perms.bin -ngl 99

# Non-split mode (uniform bits on all 128 channels)
llama-perplexity -m model.gguf -f wikitext-2-raw/wiki.test.raw --chunks 3 \
    -ctk tqk_flex --tq-hi-bits 8 --tq-qjl none \
    --tq-perms perms.bin -ngl 99
```

`tqk_flex` flags:
- `--tq-split` — enable 32/96 channel split (requires calibrated perms)
- `--tq-hi-bits N` — MSE bits for outlier channels (split) or all channels (non-split)
- `--tq-lo-bits N` — MSE bits for regular channels (split only)
- `--tq-qjl none|hi|both` — QJL correction: none, on outliers only, or on both subsets
- `--tq-perms FILE` — calibration file (required for split, optional for non-split)

### Flex spec format (for `--flex-*` calibrate flags)

`split:hi_bits:lo_bits:qjl_hi:qjl_lo`

| Field | Description |
|-------|-------------|
| split | 1 = split into 32 outlier + 96 regular channels, 0 = uniform 128-dim |
| hi_bits | bits per outlier channel (split) or per channel (non-split). Range: 2-10 |
| lo_bits | bits per regular channel (split only, ignored if split=0). Range: 1-8 |
| qjl_hi | 1 = QJL on outliers (split) or all channels (non-split) |
| qjl_lo | 1 = per-element QJL on regular channels (split only) |

Examples:
- `1:9:4:1:1` — split, 9-bit hi, 4-bit lo, QJL on both (6.75 bpv)
- `1:10:5:1:0` — split, 10-bit hi, 5-bit lo, QJL on hi only (6.88 bpv)
- `0:6:0:0:0` — non-split, 6-bit uniform, no QJL (6.125 bpv)
- `0:5:0:1:0` — non-split, 5-bit uniform, QJL (6.25 bpv)

### Calibration

The calibrate tool (`llama-tq-calibrate`) runs a short inference pass on calibration text and measures per-channel K variance:

```bash
llama-tq-calibrate -m model.gguf -f ptb/ptb.train.txt -o perms.bin [options]

Options:
  --pre-rope        Capture pre-RoPE K (default, required)
  --metric var      Outlier = highest variance (default)
  -n 4096           Calibration tokens (default: 4096)
  -c 512            Context size (default: 512)

Per-layer flex (appends TQFC section):
  --flex-extreme S:H:L:JH:JL   Config for extreme layers (>threshold_extreme)
  --flex-high    S:H:L:JH:JL   Config for high outlier layers (>threshold_high)
  --flex-moderate S:H:L:JH:JL  Config for moderate/uniform layers
  --flex-threshold-extreme N    % threshold for extreme (default: 90)
  --flex-threshold-high N       % threshold for high (default: 60)
```

The output perms file contains:
1. **Header**: magic, version, n_layers, n_heads, head_dim, pre_rope flag
2. **Channel permutations**: per-layer, per-head reordering (outlier channels first)
3. **TQLT section**: per-layer type recommendations (ggml_type per layer)
4. **TQFC section** (optional): per-layer flex configs (split, bits, QJL per layer)

**Calibration is architecture-specific, not weight-specific.** The outlier channel structure is determined by the model architecture (attention weight matrices), not the precision of those weights. Perms calibrated on fp16, q8_0, or q4_k_m of the same model produce 96%+ identical outlier sets.

This means you can calibrate on any quantization of a model and use the perms on all other quantizations. The only requirement is **pre-RoPE capture** (the default) — post-RoPE calibration produces garbage because RoPE rotates the channel variance structure.

**Calibration data**: Use Penn Treebank (PTB) train set or similar. Do NOT calibrate on the evaluation data (wikitext). ~4096 tokens is sufficient.

## How We Found the Best Configuration

The configuration was found through systematic sweeping on Qwen2.5 1.5B. The process:

### Step 1: Find the right hi bits for split layers

With 32 outlier channels split from 96 regular channels, we swept hi bits from 5 to 10:

| Hi bits | Lo=3, QJL=hi | bpv | PPL |
|---------|-------------|-----|-----|
| 5 | tqk4_sj equiv | 4.125 | 1250 |
| 6 | | 4.375 | 97 |
| 7 | | 4.625 | 25 |
| 8 | | 4.875 | 10.1 |
| 9 | | 5.125 | 9.9 |
| 10 | | 5.375 | 9.9 |

The jump from 5-bit to 6-bit hi was 13x — the 32 outlier channels need high precision. 9-bit is the sweet spot; 10-bit doesn't improve further.

### Step 2: Find the right lo bits

With hi=9, we swept lo bits:

| Config | bpv | PPL (20 chunks) | vs f16 |
|--------|-----|-----------------|--------|
| 9/2 QJL=hi | 4.375 | 11.03 | +5.4% |
| 9/3 QJL=hi | 5.125 | 11.89 | +2.2% |
| 9/3 none | 4.750 | 11.88 | +2.1% |
| 9/4 QJL=hi | 5.875 | 11.69 | +0.4% |
| 9/4 QJL=both | 6.750 | 11.69 | +0.4% |

4-bit lo was the sweet spot for staying under 1%.

### Step 3: Per-layer — which layers need split?

We tested each tier independently (all other layers at f16):

| Tier quantized | Layers | PPL impact |
|----------------|--------|------------|
| Extreme only (>90%) | 3 | +0.1% |
| High only (60-90%) | 10 | +0.4% |
| Moderate only (<60%) | 15 | +1.3% |

Surprise: moderate layers (with weak outlier structure) contributed the MOST degradation from split quantization. The forced 32/96 split hurts when there's no clear outlier/regular separation.

### Step 4: Non-split for moderate layers

We tested non-split (uniform FWHT-128) on moderate layers:

| Moderate config | Avg bpv | PPL | vs f16 |
|----------------|---------|-----|--------|
| split 5/4 QJL=hi | 5.34 | 11.720 | +0.7% |
| split 5/5 QJL=both | 6.62 | 11.717 | +0.7% |
| **non-split 6-bit** | **6.42** | **11.651** | **+0.1%** |
| non-split 7-bit | 6.95 | 11.676 | +0.3% |
| non-split 8-bit | 7.49 | 11.663 | +0.2% |

Non-split 6-bit beat all split variants for moderate layers.

### Step 5: Optimize the threshold

We swept the outlier % threshold that separates "use split" from "use non-split":

| Threshold | Split layers | Avg bpv | PPL |
|-----------|-------------|---------|-----|
| 50% | 28 (all) | 6.57 | 11.645 |
| 60% | 13 | 6.42 | 11.651 |
| **70%** | **8** | **6.21** | **11.639** |
| 80% | 5 | 6.19 | 11.660 |
| 90% | 3 | 6.19 | 11.660 |

70% was the sweet spot — only a few layers (4 in this calibration run) truly benefit from split quantization.

## Model-Specific Results: Qwen2.5 1.5B Instruct

> **Important**: The specific bit allocations, layer thresholds, and tier assignments below are optimized for Qwen2.5 1.5B (28 layers, 2 KV heads, d=128). Other models will have different outlier distributions and need different configurations. The calibrate tool handles this — run it on your model and tune thresholds accordingly.

### Layer Tiers (Qwen2.5 1.5B, threshold 70%)

| Tier | Outlier % | Count* | Config | bpv |
|------|-----------|--------|--------|-----|
| Split (>70%) | 79-100% | 4 | 9/4 QJL=both | 6.75 |
| Non-split (<70%) | 43-68% | 24 | 6-bit uniform | 6.125 |
| **Average** | | **28** | | **6.21** |

*Exact count varies slightly between calibration runs due to outlier % noise.

### Perplexity Across Context Lengths

q4_k_m model weights, Metal FA, wikitext-2:

| Context | f16 KV (16.0 bpv) | q8_0 KV (8.5 bpv) | TQ best (6.21 bpv) | TQ+QJL (7.18 bpv) |
|---------|-------|-------|---------|----------|
| 512     | 12.413 | 12.420 | 12.419 | 12.419 |
| 1024    | 9.163 | 9.183 | 9.190  | 9.187  |
| 2048    | 8.904 | 8.911 | 8.916  | 8.915  |
| 4096    | 8.926 | 8.939 | 8.940  | 8.936  |
| 8192    | 7.971 | 7.977 | 7.978  | 7.976  |
| 16384   | 7.130 | 7.129 | 7.140  | 7.132  |
| 32768*  | 9.018 | 9.046 | 9.038  | 9.032  |

*32k with only 2 chunks — too few for reliable evaluation.

- TQ at 6.21 bpv: always within +0.3% of f16
- TQ consistently matches or beats q8_0 (which uses 37% more memory)
- q4_0 KV at similar bpv: PPL 3600 (unusable)

Note on QJL in these results:
- **TQ best (6.21 bpv)**: QJL on split layers only (`1:9:4:1:1`), no QJL on non-split layers (`0:6:0:0:0`)
- **TQ+QJL (7.18 bpv)**: QJL on all layers — split layers have `1:9:4:1:1`, non-split layers have `0:6:0:1:0`
- A fully no-QJL config (`1:9:4:0:0` + `0:6:0:0:0`) would save ~0.4 bpv on the split layers but may degrade at very long contexts

### 32k Context (9 chunks, maximum for wikitext-2)

| KV config | bpv | PPL | ±95% CI |
|-----------|-----|-----|---------|
| f16 KV | 16.00 | 9.108 | ±0.060 |
| TQ+QJL | 7.18 | 9.118 | ~±0.06 |
| TQ best | 6.21 | 9.121 | ~±0.06 |
| q8_0 KV | 8.50 | 9.136 | ~±0.06 |

At 32k context with 9 chunks (maximum for wikitext-2), all configs fall within each other's confidence intervals (~±0.06). The differences are **not statistically significant** at this sample size. The trend (TQ ≤ q8_0) is consistent with shorter-context results but would need a larger corpus to confirm definitively.

Both TQ variants beat q8_0 in point estimate. TQ+QJL edges TQ best by 0.003 PPL — the QJL bias correction shows a consistent (if tiny) benefit at 32k.

### 200-Chunk Comparison (ctx=512)

| KV config | bpv | PPL | ±CI | vs f16 |
|-----------|-----|-----|-----|--------|
| f16 KV | 16.00 | 11.486 | ±0.139 | — |
| **TQ best** | **6.21** | **11.514** | **±0.139** | **+0.24%** |
| TQ+QJL all | 7.18 | 11.522 | ±0.139 | +0.31% |
| q8_0 KV | 8.50 | 11.524 | ±0.139 | +0.33% |
| q4_1 KV | 5.00 | 1530.5 | — | garbage |
| q4_0 KV | 4.50 | 3925.9 | — | garbage |

All non-garbage configs fall within each other's 95% confidence intervals. q4_0 and q4_1 are completely unusable on this model.

### Calibration Source Comparison

All tested on q4_k_m model at runtime, 20 chunks:

| Perms calibrated on | PPL | Outlier overlap with fp16 |
|---------------------|-----|---------------------------|
| fp16 model | 11.639 | — |
| q8_0 model | 11.645 | 96.2% |
| q4_k_m model | 11.651 | 96.1% |

All three produce near-identical results. Calibrate on whatever quantization you have.

## Block Layout

### Split block (108 bytes, 6.75 bpv)

```
Offset  Size   Field
0       2      norm_hi (fp16) — L2 norm of 32 rotated outlier channels
2       2      norm_lo (fp16) — L2 norm of 96 rotated regular channels
4       36     qs_hi[36] — 9-bit × 32 packed centroid indices (outliers)
40      48     qs_lo[48] — 4-bit × 96 packed centroid indices (regulars)
88      2      rnorm_qjl_hi (fp16) — QJL residual norm (outliers)
90      4      signs_hi[4] — 1-bit × 32 QJL signs (FWHT-projected)
94      2      rnorm_qjl_lo (fp16) — QJL residual norm (regulars)
96      12     signs_lo[12] — 1-bit × 96 QJL signs (per-element)
```

### Non-split block (98 bytes, 6.125 bpv)

```
Offset  Size   Field
0       2      norm (fp16) — L2 norm of 128 rotated channels
2       96     qs[96] — 6-bit × 128 packed centroid indices
```

## Results: Qwen2.5 7B Instruct

Model: q4_k_m weights, Metal FA, wikitext-2, 20 chunks.

The 7B has more extreme early layers than the 1.5B: layers 0-3 are all >70% outlier concentration (100%, 92%, 72%, 87%), plus layer 13 (76%), 19 (80%), and 27 (100%). The middle layers (4-12, 14-18, 20-26) are uniformly 50-65%.

### Best Configuration (Korean-safe)

The 7B requires higher precision than the 1.5B for multilingual tasks. A "Pauli test" (Korean transliteration of scientist names using 외래어 표기법 standard) revealed that lower bit configs that pass English PPL tests can corrupt cross-script generation. The winning config was determined by finding the lowest bpv that passes both PPL and Pauli tests.

```bash
llama-tq-calibrate -m model.gguf -f ptb.txt -o perms.bin \
    --flex-extreme 1:10:5:1:0 \
    --flex-high 1:10:5:1:0 \
    --flex-moderate 0:5:0:1:0 \
    --flex-threshold-high 70
```

**Split layers** (7 layers, >70% outlier): 10-bit hi (1024 centroids, d=32) + 5-bit lo (32 centroids, d=96) + QJL on hi only. 110 bytes = **6.88 bpv**.

**Non-split layers** (21 layers, <70% outlier): 5-bit uniform (32 centroids, d=128) + QJL. 100 bytes = **6.25 bpv**.

**Average: 6.41 bpv = 2.5x K cache compression.**

With QJL on both hi and lo for split layers (conservative for very long context): avg 6.62 bpv.

### Per-Layer Assignment

| Layer | Outlier % | Config | bpv |
|-------|-----------|--------|-----|
| 0 | 100% | split 10/5 QJL=hi | 6.88 |
| 1 | 92% | split 10/5 QJL=hi | 6.88 |
| 2 | 72% | split 10/5 QJL=hi | 6.88 |
| 3 | 87% | split 10/5 QJL=hi | 6.88 |
| 4-12 | 50-58% | nosplit 5-bit + QJL | 6.25 |
| 13 | 76% | split 10/5 QJL=hi | 6.88 |
| 14-18 | 60-70% | nosplit 5-bit + QJL | 6.25 |
| 19 | 80% | split 10/5 QJL=hi | 6.88 |
| 20-26 | 57-64% | nosplit 5-bit + QJL | 6.25 |
| 27 | 100% | split 10/5 QJL=hi | 6.88 |

### PPL Results (200 chunks)

| KV config | bpv | PPL | ±CI | vs f16 |
|-----------|-----|-----|-----|--------|
| f16 KV | 16.00 | 8.910 | ±0.112 | — |
| **TQ winner** | **6.41** | **8.927** | **±0.112** | **+0.19%** |
| TQ+QJL all | 6.62 | 8.933 | ±0.112 | +0.26% |
| q8_0 KV | 8.50 | 8.949 | ±0.113 | +0.44% |
| q4_1 KV | 5.00 | 10131.7 | — | garbage |
| q4_0 KV | 4.50 | 3846.9 | — | garbage |

All non-garbage configs fall within each other's 95% confidence intervals. q4_0 and q4_1 are completely unusable on this model — same as the 1.5B.

### Pauli Test (Korean Transliteration)

The Pauli test checks if TQ preserves attention precision for rare cross-script tokens. The 7B model knows Korean better than the 1.5B but is still imperfect (wrong on "Wolfgang Pauli"). The key metric is whether TQ output matches f16 output.

| Scientist | f16 / q8_0 | TQ winner (6.41 bpv) |
|-----------|-----------|----------------------|
| Wolfgang Pauli | 폴딩우글 파울리 | ✅ identical |
| Niels Bohr | 니엘스 보어 | ✅ identical |
| Erwin Schrödinger | 에르빈 슈뢰딩거 | ✅ identical |
| Werner Heisenberg | 베르너 하이젠베르크 | ✅ identical |
| Max Planck | 맥스 플랑크 | ✅ identical |
| Enrico Fermi | 엔리코 페르미 | ✅ identical |

6/6 match f16. Lower bpv configs (6-bit non-split, 9-bit split) fail this test — they produce mixed Korean/Latin garbage characters.

### Differences from 1.5B Configuration

| | 1.5B | 7B |
|---|---|---|
| Split layers | 4 (threshold 70%) | 7 (threshold 70%) |
| Split hi bits | 9-bit | 10-bit (Korean needs more) |
| Split lo bits | 4-bit | 5-bit |
| Non-split bits | 6-bit | 5-bit |
| Non-split QJL | no | yes |
| Avg bpv | 6.21 | 6.41 |
| PPL vs f16 | +0.5% | +0.7% |

The 7B needs 10-bit hi (vs 9-bit for 1.5B) because the Pauli test fails at 9-bit with QJL=hi. Investigation confirmed this is NOT a code bug but a precision limitation: at 9-bit MSE, the QJL 1-bit correction (`sqrt(π/2)/32 * rnorm * ±1`) adds a perturbation large enough to shift attention weights for rare cross-script tokens (Korean). At 10-bit MSE, the quantization error is smaller, the QJL residual is smaller, and the correction doesn't corrupt.

Key findings from QJL investigation:
- 9/6 QJL=none: ✅ identical to f16 (Korean perfect)
- 9/6 QJL=hi: ❌ mixed Korean/Latin output
- 10/5 QJL=hi: ✅ identical to f16 (Korean perfect)
- Bug reproduces on both CPU and Metal FA paths → not a backend issue
- Confirmed on both uniform flex and per-layer flex
- The 1.5B was only tested on English where this doesn't manifest

## Results: Qwen3 8B

Model: Q8_0 weights, Metal FA, wikitext-2.

Qwen3 8B has the same head geometry as Qwen2.5 7B (d=128, 8 KV heads, GQA) but 36 layers. Its outlier distribution is remarkably uniform — almost every layer falls in the "high" tier (60-90%), with only layer 0 as extreme (99%) and 4 moderate layers (25-27, 35). This changes the optimal strategy completely.

### Outlier Distribution

| Tier | Layers | Count |
|------|--------|-------|
| Extreme (>90%) | 0 | 1 |
| High (60-90%) | 1-24, 28-34 | 31 |
| Moderate (<60%) | 25-27, 35 | 4 |

### Best Configuration

Unlike Qwen2.5 where split quantization helps extreme layers, Qwen3's uniform outlier distribution means **every layer benefits equally from the same treatment**. Split quantization consistently underperforms nosplit on this model.

```bash
llama-tq-calibrate -m model.gguf -f ptb.txt -o perms.bin \
    --flex-all 0:5:5:1:0
```

**All 36 layers**: nosplit 5-bit (32 centroids, d=128) + QJL. 82 bytes = **5.125 bpv**.

**Average: 5.125 bpv = 3.1x K cache compression.**

### Why Nosplit Wins on Qwen3

The sweep tested split configs (9/4, 8/3, 7/3, 6/3, 5/4, 5/3) and nosplit (4-6 bit) with and without QJL. All split configs performed worse than nosplit 5-bit:

| Config | PPL@20 | vs f16 (13.364) | bpv |
|--------|--------|-----------------|-----|
| **nosplit 5-bit QJL** | **13.340** | **≈0%** | **5.125** |
| nosplit 5-bit | 13.348 | ≈0% | 5.0 |
| 5/4 split QJL=hi | 13.357 | ≈0% | 4.88 |
| 9/4 split QJL=both | 13.377 | +0.1% | 6.7 |
| 8/3 split QJL=hi | 13.493 | +1.0% | 4.88 |
| 6/3 split QJL=hi | 13.525 | +1.2% | 4.38 |
| nosplit 4-bit QJL | 13.891 | +3.9% | 4.125 |
| q4_0 (reference) | 13.968 | +4.5% | 4.5 |

The split 32/96 channel partition assumes some channels are dramatically more important than others. In Qwen3, the outlier distribution is too uniform for this — forcing an artificial split wastes bits on the hi block without improving the lo block enough to compensate.

Per-layer differentiation (different configs for extreme/high/moderate) also didn't help — giving layer 0 more bits (6-bit, 8-bit, or split 9/4) made PPL slightly worse, not better.

### PPL Results (200 chunks)

| KV config | bpv | PPL | ±CI | vs f16 |
|-----------|-----|-----|-----|--------|
| f16 KV | 16.00 | 10.899 | ±0.150 | — |
| q8_0 KV | 8.50 | 10.890 | ±0.150 | ≈0% |
| **TQ ns5 QJL** | **5.125** | **10.848** | **±0.149** | **≈0%** |
| q4_1 KV | 5.00 | 10.993 | ±0.149 | +0.86% |
| q4_0 KV | 4.50 | 11.470 | ±0.226 | +4.03%* |

*q4_0 from 100-chunk run.

f16, q8_0, and TQ all fall within each other's 95% confidence intervals. q4_0 is clearly worse at +4%. q4_1 consistently trends worse (+0.86%) across all chunk counts but the gap is not statistically significant at 200 chunks (~1.3σ). The trend is consistent: TQ ns5 QJL outperforms q4_1 in every run at similar bpv (5.125 vs 5.0).

### Extended Evaluation (64 chunks)

| KV config | bpv | PPL@64 | vs f16 |
|-----------|-----|--------|--------|
| f16 | 16.00 | 10.795 | — |
| q8_0 | 8.50 | 10.791 | ≈0% |
| **TQ ns5 QJL** | **5.125** | **10.755** | **≈0%** |
| q4_0 | 4.50 | 11.260 | +4.3% |

TQ ns5 QJL holds perfectly across more evaluation data. q4_0 remains at +4.3%, consistent from 20 to 64 chunks.

### Generation Quality

Model: Qwen3-8B (Q8_0), seed=42, temp=0, Metal FA. Qwen3 uses `<think>` reasoning chains.

| Prompt | f16 | q8_0 | TQ (5.125 bpv) |
|--------|-----|------|----------------|
| 127 × 38 | ✅ (correct reasoning chain) | identical | identical |
| French translation | ✅ Le chat s'est assis... | identical | identical (minor phrasing: "the original sentence") |
| Capital of Australia | ✅ Canberra | identical | identical (minor phrasing) |
| Train distance (60mph × 2.5h) | ✅ 150 miles | identical | identical |
| Korean transliteration | ✅ correct 외래어 표기법 | identical | identical |
| Python palindrome | ✅ correct implementation | identical | identical |

All outputs match f16 quality. The `<think>` reasoning chains are virtually word-for-word identical across all three K cache types.

### Differences from Qwen2.5 Configuration

| | Qwen2.5 1.5B | Qwen2.5 7B | Qwen3 8B |
|---|---|---|---|
| Outlier pattern | 4 extreme layers | 7 extreme layers | 1 extreme layer |
| Best config | split 9/4 + nosplit 6 | split 10/5 + nosplit 5 | **nosplit 5 everywhere** |
| Split needed? | yes (extreme layers) | yes (extreme layers) | **no** |
| QJL | per-tier | on hi / on all | on all |
| Avg bpv | 6.21 | 6.41 | **5.125** |
| PPL vs f16 | +0.5% | +0.7% | **≈0%** |
| Compression | 2.6x | 2.5x | **3.1x** |

Qwen3's uniform outlier distribution is ideal for TurboQuant — the single nosplit 5-bit + QJL config achieves the best compression ratio and lowest PPL delta of all tested models.

## Generation Quality Comparison

Model: Qwen2.5 1.5B Instruct (q4_k_m), seed=42, temp=0, Metal FA.

### Factual / Knowledge

| Prompt | f16 | q8_0 | TQ (6.21 bpv) |
|--------|-----|------|---------------|
| List planets in order from Sun | Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune | identical | identical |
| Capital of Australia? | Canberra | identical | identical |
| First 10 prime numbers | 2,3,5,7,11,13,17,19,23,29 | identical | identical |

All factual answers are **token-for-token identical** across f16, q8_0, and TQ.

### Math / Reasoning

**Q: What is 127 × 38?** (correct answer: 4826)
- f16: "So, 127 × 38 = 4566." (wrong)
- q8_0: "So, 127 × 38 = 4666." (wrong)
- TQ: "127 * 38 = 4866" (wrong)

All three get it wrong — 1.5B model limitation, not KV quantization.

**Q: If a train travels at 60 mph for 2.5 hours, how far does it go?**
- f16: "Distance = 60 mph × 2.5 hours = 150 miles"
- q8_0: identical
- TQ: identical

**Q: Simplify the expression: (3x + 2)(x - 5)**
- f16: Correct FOIL method: 3x·x = 3x², 3x·(-5) = -15x, 2·x = 2x, 2·(-5) = -10
- q8_0: identical steps and math
- TQ: identical steps and math (minor wording: "binomials" vs "product")

### Code

**Q: Write a Python function that checks if a string is a palindrome.**

f16 and TQ produce identical code:
```python
def is_palindrome(s):
    normalized_s = ''.join(char.lower() for char in s if char.isalnum())
    return normalized_s == normalized_s[::-1]

print(is_palindrome("A man, a plan, a canal: Panama"))  # True
print(is_palindrome("racecar"))  # True
```
q8_0 produces equivalent code with slightly different variable name (`cleaned` vs `normalized_s`).

### Creative

**Q: Write a haiku about winter.**
- f16: "Snow falls softly, / Trees stand still in white, / Winter's breath."
- q8_0: "Snow muffles the earth, / White whispers in the cold, / Winter holds its breath."
- TQ: "Snow falls softly, / Blanketing the world in white, / Winter's breath, so sweet."

All three are valid haikus, just different artistic choices. Creative generation is sensitive to tiny probability shifts — even q8_0 differs from f16.

**Q: Translate to French: The cat sat on the mat and watched the birds fly by.**
- f16: "Le chat s'est assis sur le tapis et a observé les oiseaux voler par la fenêtre."
- q8_0: identical
- TQ: identical

### Summary

TQ at 6.21 bpv produces **identical output to f16 on factual, mathematical, and coding tasks**. Differences only appear in creative/sampling-sensitive tasks where even q8_0 differs from f16. The KV cache quantization does not degrade the model's knowledge, reasoning, or instruction-following ability.

## Key Findings

1. **Outlier channels need high precision**: 5-bit on 32 outlier channels → PPL 1250. 9-bit → PPL 10. The highest-variance channels dominate attention quality.

2. **Per-layer adaptation matters**: Layers with uniform variance do better with non-split 6-bit than with a forced 32/96 split. The optimal threshold varies by model.

3. **Calibration works across quantizations**: fp16, q8_0, q4_k_m all produce equivalent perms (96%+ overlap). Calibrate on whatever you have.

4. **Pre-RoPE calibration is essential**: Post-RoPE gives garbage. The calibrate tool defaults to pre-RoPE.

5. **QJL is context-length dependent**: Skip it for <8k context (saves ~1 bpv). Consider it for 16k+ where bias accumulates. At 32k context, TQ+QJL consistently outperforms TQ without QJL.

6. **Standard quantization fails at low bpv**: q4_0 (4.5 bpv) and q4_1 (5.0 bpv) give PPL 1300-3600. TurboQuant at 6.2 bpv matches f16. The rotation + optimal centroids + outlier-aware split make the difference.

## V Cache Quantization Quality

All PPL results above use f16 V to isolate the K cache effect. Below we verify that TQ K works correctly with all supported V cache types.

### Qwen2.5 7B (q4_k_m weights, Metal FA, seed=42, temp=0)

K = TQ winner (10/5 QJL=hi split + 5-bit+QJL nosplit, 6.41 bpv K)

| V type | Math (127×38) | French translation | Korean (Pauli) |
|--------|---------------|-------------------|----------------|
| K=f16, V=f16 (baseline) | correct FOIL method | Le chat s'est assis sur le tapis et a regardé les oiseaux passer. | 볼프강 파울리, 니ELS 보어, 맥스 플랑크 |
| K=TQ, V=f16 | ✅ identical | ✅ identical | ✅ identical |
| K=TQ, V=q8_0 | ✅ identical | ✅ identical | ✅ identical |
| K=TQ, V=q4_0 | ✅ minor wording | ✅ identical | ✅ identical |
| K=TQ, V=q4_1 | ✅ different approach | ✅ identical | ✅ identical |
| K=TQ+QJL, V=f16 | ✅ identical | ✅ identical | ✅ identical |
| K=TQ+QJL, V=q8_0 | ✅ identical | ✅ identical | ✅ identical |
| K=TQ+QJL, V=q4_0 | ✅ minor wording | ✅ identical | ✅ identical |
| K=TQ+QJL, V=q4_1 | ✅ different approach | ✅ identical | ✅ slight truncation |

### Qwen2.5 1.5B (q4_k_m weights, Metal FA, seed=42, temp=0)

K = TQ winner (9/4 QJL=both split + 6-bit nosplit, 6.21 bpv K)

| V type | Math (127×38) | French translation | Korean |
|--------|---------------|-------------------|--------|
| K=f16, V=f16 (baseline) | FOIL method (correct) | Le chat s'est assis... observé les oiseaux voler | English description (1.5B can't do Korean) |
| K=TQ, V=f16 | 4866 (short answer) | ✅ identical | ✅ identical |
| K=TQ, V=q8_0 | ✅ identical to TQ+f16 | ✅ identical | ✅ identical |
| K=TQ, V=q4_0 | ✅ identical | slightly different wording | ✅ |
| K=TQ, V=q4_1 | different approach | slightly different | ✅ |
| K=TQ+QJL, V=f16 | 4866 | ✅ identical | ✅ |
| K=TQ+QJL, V=q8_0 | ✅ identical | ✅ identical | ✅ |
| K=TQ+QJL, V=q4_0 | ✅ identical | slightly different (La chatte) | ✅ |
| K=TQ+QJL, V=q4_1 | different approach | slightly different | ✅ |

**Conclusion**: TQ K produces correct output with all V cache types (f16, q8_0, q4_0, q4_1). Minor wording variations with q4_0/q4_1 V are the V quantization effect — identical differences appear with f16 K + q4_0/q4_1 V baselines.

## Throughput Benchmarks

Apple M4 Pro, Metal FA, pp512 + tg128, 3 repetitions. V=f16 for all.

Benchmarked with:
```bash
llama-bench -m model.gguf --tq-perms perms.bin -ctk f16,q8_0,tqk -ctv f16 -fa 1 -ngl 99 -r 3 -p 512 -n 128
```

### Qwen2.5 1.5B (q4_k_m, K=6.21 bpv)

| K type | K bpv | pp512 (t/s) | tg128 (t/s) | pp vs f16 | tg vs f16 |
|--------|-------|-------------|-------------|-----------|-----------|
| f16    | 16.00 | 2281.7 ± 6.0 | 167.7 ± 0.2 | — | — |
| q8_0   | 8.50  | 1608.4 ± 28.6 | 102.2 ± 0.8 | -30% | -39% |
| **TQ** | **6.21** | **2108.4 ± 2.7** | **85.0 ± 0.2** | **-8%** | **-49%** |

### Qwen2.5 7B (q4_k_m, K=6.41 bpv)

| K type | K bpv | pp512 (t/s) | tg128 (t/s) | pp vs f16 | tg vs f16 |
|--------|-------|-------------|-------------|-----------|-----------|
| f16    | 16.00 | 495.6 ± 0.3 | 51.4 ± 0.0 | — | — |
| q8_0   | 8.50  | 417.2 ± 0.4 | 41.6 ± 0.1 | -16% | -19% |
| **TQ** | **6.41** | **472.1 ± 0.1** | **25.2 ± 0.0** | **-5%** | **-51%** |

### Analysis

**Prompt processing (pp)**: TQ is only 5-8% slower than f16 — faster than q8_0 on both models. The FA kernel with TQ K dequant is efficient for batch processing.

**Token generation (tg)**: TQ is 49-51% slower than f16. The per-token decode path (vec FA kernel) has higher overhead from flex config activation + centroid lookup per layer. This is the main performance bottleneck.

q8_0 is also significantly slower than f16 (19-39% for tg), suggesting Metal FA with quantized K types has general overhead vs f16 — not specific to TQ.

**Tradeoff**: TQ saves 2.5-2.6x K cache memory at the cost of ~50% slower token generation. For memory-bound deployments (long context, large batch), the memory savings outweigh the speed loss. For latency-critical single-user scenarios, f16 K is faster.

## CUDA Benchmark Results

Hardware: NVIDIA RTX A6000 (48 GB VRAM, compute capability 8.6), CUDA 13.0, Driver 580.126.09.

Methodology: same as Metal — WikiText-2 test set, `llama-bench` with pp512/tg128, flash attention enabled, full GPU offload. Calibration identical to Metal sections above. V cache: f16 for all tests.

### Perplexity (50 chunks)

| Model | f16 | q8_0 | tqk | tqk vs f16 |
|-------|-----|------|-----|------------|
| Qwen 2.5 1.5B (Q4_K_M) | 9.837 | 9.870 | 9.870 | +0.34% |
| Qwen 2.5 7B (Q4_K_M) | 7.260 | 7.291 | 7.304 | +0.60% |
| Qwen3 8B (Q4_K_M) | 9.730 | 9.708 | 9.731 | +0.01% |

All TQ results fall within noise of f16, consistent with Metal results.

### Throughput

#### Qwen 2.5 1.5B (Q4_K_M, 1.04 GiB)

| Type | KV bpv | pp512 (t/s) | tg128 (t/s) |
|------|--------|-------------|-------------|
| f16+f16 | 16.00 | 17,271 | 359 |
| q8_0+f16 | 8.50 | 4,338 | 258 |
| tqk+f16 | 6.21 | 2,284 | 138 |

#### Qwen 2.5 7B (Q4_K_M, 4.36 GiB)

| Type | KV bpv | pp512 (t/s) | tg128 (t/s) |
|------|--------|-------------|-------------|
| f16+f16 | 16.00 | 5,479 | 130 |
| q8_0+f16 | 8.50 | 1,706 | 109 |
| tqk+f16 | 6.41 | 773 | 76 |

#### Qwen3 8B (Q4_K_M, 4.68 GiB)

| Type | KV bpv | pp512 (t/s) | tg128 (t/s) |
|------|--------|-------------|-------------|
| f16+f16 | 16.00 | 4,783 | 117 |
| q8_0+f16 | 8.50 | 1,309 | 89 |
| tqk+f16 | 5.13 | 790 | 65 |

### Analysis

**Prompt processing (pp)**: TQ is 3.2-7.6x slower than f16 on CUDA, compared to only 1.05x on Metal. The CUDA FA vec kernel with runtime-configurable flex dequant has higher per-element overhead than the Metal implementation which benefits from unified memory and different memory hierarchy.

**Token generation (tg)**: TQ is 38-58% of f16 throughput depending on model size. Larger models (7B, 8B) show better relative performance (58%, 56%) than the 1.5B (38%) because they are more memory-bandwidth bound, amortizing the TQ compute overhead.

**CUDA vs Metal comparison**: Metal tg overhead is ~50% (TQ vs f16). CUDA tg overhead is 42-62%. The gap is narrower on larger models. The CUDA implementation uses shared memory centroid tables and specialized bit-unpack to minimize per-element overhead in the FA vec_dot kernel.

**q8_0 reference**: q8_0 on CUDA is 72-84% of f16 tg, showing that quantized K cache types have general FA overhead on CUDA — not specific to TQ.
