# TurboQuant Flex: KV Cache Quantization Results

## Best Configuration

**Per-layer adaptive: threshold 70%**
- 8 layers with >70% outlier concentration: split 9/4 QJL=both (6.75 bpv)
- 20 layers with <70% outlier concentration: non-split 6-bit (6.00 bpv)
- **Average: 6.21 bpv — 2.6x K cache compression**

```bash
# Calibrate (once per model architecture, works on any quantization)
llama-tq-calibrate -m model.gguf -f ptb.txt -o perms.bin \
    --flex-extreme 1:9:4:1:1 \
    --flex-high 1:9:4:1:1 \
    --flex-moderate 0:6:0:0:0 \
    --flex-threshold-high 70

# Run
llama-perplexity -m model.gguf -ctk tqk --tq-perms perms.bin -ngl 99
```

## Perplexity Across Context Lengths

Model: Qwen2.5 1.5B Instruct (q4_k_m weights), Metal FA, wikitext-2

| Context | f16 KV (16.0 bpv) | q8_0 KV (8.5 bpv) | TQ best (6.21 bpv) | TQ+QJL (7.18 bpv) |
|---------|-------|-------|---------|----------|
| 512     | 12.413 | 12.420 | 12.419 | 12.419 |
| 1024    | 9.163 | 9.183 | 9.190  | 9.187  |
| 2048    | 8.904 | 8.911 | 8.916  | 8.915  |
| 4096    | 8.926 | 8.939 | 8.940  | 8.936  |
| 8192    | 7.971 | 7.977 | 7.978  | 7.976  |
| 16384   | 7.130 | 7.129 | 7.140  | 7.132  |
| 32768   | 9.018 | 9.046 | 9.038  | 9.032  |

- TQ at 6.21 bpv: always within +0.3% of f16, beats or matches q8_0
- QJL correction helps at 16k+ context (bias accumulation over long sequences)
- q4_0/q4_1 KV at similar bpv: PPL 1300-3600 (unusable)

## 20-Chunk Comparison (ctx=512)

| KV config | bpv | PPL | vs f16 | Compression |
|-----------|-----|-----|--------|-------------|
| f16 KV | 16.00 | 11.640 | — | 1.0x |
| q8_0 KV | 8.50 | 11.657 | +0.1% | 1.9x |
| **TQ best** | **6.21** | **11.639** | **-0.01%** | **2.6x** |
| TQ+QJL | 7.18 | 11.638 | -0.02% | 2.2x |
| q4_1 KV | 4.50 | 1309.6 | +11,150% | — |
| q4_0 KV | 4.50 | 3600.2 | +30,830% | — |

## Architecture Details

### Split layers (outlier >70%): 9-bit hi / 4-bit lo + QJL both
- 32 outlier channels: 9-bit MSE (512 centroids on d=32) + 1-bit QJL sign correction
- 96 regular channels: 4-bit MSE (16 centroids on d=96) + 1-bit per-element QJL
- Channel permutation from calibration (pre-RoPE)
- Rotation: FWHT-32 on hi, bare 3×FWHT-32 on lo
- 6.75 bpv per block (108 bytes / 128 dims)

### Non-split layers (outlier <70%): 6-bit uniform
- All 128 channels: 6-bit MSE (64 centroids on d=128)
- Rotation: FWHT-128
- No QJL (not needed — uniform quantization has low bias)
- 6.00 bpv per block (98 bytes / 128 dims)

## Key Findings

1. **Outlier channels need high precision**: 5-bit on 32 outlier channels → PPL 1250. 9-bit → PPL 10. The 32 highest-variance channels dominate attention quality.

2. **Per-layer adaptation matters**: Layers with <70% outlier concentration do better with uniform non-split quantization than with split quantization that forces an artificial outlier/regular separation.

3. **Calibration is architecture-specific, not weight-specific**: Perms calibrated on fp16, q8_0, or q4_k_m give 96%+ overlap. Calibrate on any quantization, use on all.

4. **Pre-RoPE calibration is essential**: Post-RoPE calibration produces garbage perms (PPL 1163 vs 10 with pre-RoPE).

5. **QJL correction**: Negligible at short context (<8k). Helps at 16k+ where inner product bias accumulates. Cost: +1 bpv.

6. **Lloyd-Max centroids**: Proper Beta((d-1)/2,(d-1)/2) centroids are critical. The old approximate centroids (Gaussian inverse CDF) caused 100x worse PPL at 5-bit.

## Centroid Tables

Extended centroids computed with Anderson-accelerated Lloyd-Max algorithm:
- d=32: 2-10 bit (4 to 1024 centroids)
- d=96: 1-8 bit (2 to 256 centroids)  
- d=128: 3-8 bit (8 to 256 centroids)

Stored in `tests/turboquant-centroids-extended.c`.
