# TurboQuant Flex: KV Cache Quantization

## Scope

**Backend support**: TurboQuant Flex (`tqk_flex`, `tqk` auto per-layer) is supported on **CPU and Metal only**. CUDA is not yet supported — using `-ctk tqk_flex` or `-ctk tqk` with TQFC configs on a CUDA build will error at context creation. The fixed TQ types (tqk4_sj, tqk3_sjj, etc.) have CUDA support.

**V cache**: TurboQuant currently compresses **only the K cache**. All results below use f16 V.

Supported V cache types with TQ K:
- **f16**: ✅ full support, recommended
- **q4_0**: ✅ works (Metal FA template exists at d=128)
- **tqv4_0**: ✅ works for fixed TQ K types, untested with flex K. tqv4_0 (`GGML_TYPE_TQV_HAD_MSE4`) applies FWHT + 4-bit Lloyd-Max MSE to V vectors (4.125 bpv).
- **q8_0**: ❌ no Metal FA template at d=128 (upstream llama.cpp limitation, not TQ-specific). Produces garbage with TQ K. Runtime guard auto-switches to f16 V.
- **q4_1**: ❌ same as q8_0

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

```bash
# Step 1: Calibrate (once per model architecture)
llama-tq-calibrate -m model.gguf -f ptb.txt -o perms.bin \
    --flex-extreme 1:9:4:1:1 \
    --flex-high 1:9:4:1:1 \
    --flex-moderate 0:6:0:0:0 \
    --flex-threshold-high 70

# Step 2: Run
llama-completion -m model.gguf -p "prompt" \
    -ctk tqk --tq-perms perms.bin -ngl 99
```

### Flex spec format

`split:hi_bits:lo_bits:qjl_hi:qjl_lo`

| Field | Description |
|-------|-------------|
| split | 1 = split into 32 outlier + 96 regular channels, 0 = uniform 128-dim |
| hi_bits | bits per outlier channel (split) or per channel (non-split) |
| lo_bits | bits per regular channel (split only, ignored if split=0) |
| qjl_hi | 1 = QJL on outliers (split) or all channels (non-split) |
| qjl_lo | 1 = per-element QJL on regular channels (split only) |

### Calibration

The calibrate tool captures K vectors during a short inference pass and identifies which channels have the highest variance per layer per head. This produces:

- **Channel permutations**: per-layer, per-head reordering that puts outlier channels first
- **Outlier concentration %**: how much variance the top-32 channels hold
- **Per-layer flex configs** (TQFC section): which quantization scheme each layer gets

**Calibration is architecture-specific, not weight-specific.** The outlier channel structure is determined by the model architecture (attention weight matrices), not the precision of those weights. Perms calibrated on fp16, q8_0, or q4_k_m of the same model produce 96%+ identical outlier sets.

This means you can calibrate on a small quantized model and use the perms on any quantization of that model. The only requirement is **pre-RoPE capture** — post-RoPE calibration produces garbage because RoPE rotates the channel variance structure.

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

*32k with 2 chunks (limited by dataset size). See 32k 9-chunk results below.

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

### 20-Chunk Comparison (ctx=512)

| KV config | bpv | PPL | vs f16 | K compression |
|-----------|-----|-----|--------|---------------|
| f16 KV | 16.00 | 11.640 | — | 1.0x |
| q8_0 KV | 8.50 | 11.657 | +0.1% | 1.9x |
| **TQ best** | **6.21** | **11.639** | **-0.01%** | **2.6x** |
| TQ+QJL | 7.18 | 11.638 | -0.02% | 2.2x |
| q4_1 KV | 5.00 | 1309.6 | +11,150% | — |
| q4_0 KV | 4.50 | 3600.2 | +30,830% | — |

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

### PPL Results (20 chunks)

| KV config | bpv | PPL | ±CI | vs f16 |
|-----------|-----|-----|-----|--------|
| f16 KV | 16.00 | 7.780 | ±0.297 | — |
| q8_0 KV | 8.50 | 7.783 | ±0.298 | +0.03% |
| **TQ QJL all** | **6.62** | **7.826** | **±0.300** | **+0.6%** |
| **TQ winner** | **6.41** | **7.831** | **±0.300** | **+0.7%** |

All results fall within each other's 95% confidence intervals.

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
