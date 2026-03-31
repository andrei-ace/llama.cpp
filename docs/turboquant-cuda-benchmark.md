# TurboQuant CUDA Benchmark Results

Benchmark results for TurboQuant KV cache quantization types on CUDA, comparing perplexity and throughput across all supported types.

## Hardware

- **GPU**: NVIDIA RTX A6000 (48 GB VRAM, Ampere, compute capability 8.6)
- **CUDA**: 12.x
- **Driver**: 570.x

## Model

- **Qwen3-8B** (Q4_K_M weight quantization, 4.68 GiB)
- Head dimension: 128 (d=128)
- KV heads: 8

## Methodology

- **Perplexity**: WikiText-2 test set, context 512, 3 chunks
- **Speed**: `llama-bench` with pp512 (prompt processing) and tg128 (text generation), flash attention enabled, full GPU offload (`-ngl 99`)
- **Calibration**: Split types (tqk3_sj, tqk4_sj, tqk3b_sj, tqk2_sj) calibrated on PTB train set (`--pre-rope`, 32k tokens, context 2048). Non-split types and standard types need no calibration.
- All TQ K types paired with tqv4_0 V cache (4.13 bpv). Standard types use matching V types.

## Results

### Perplexity

| Type | bpv | PPL | vs f16 |
|------|-----|-----|--------|
| f16 | 16.00 | 9.58 | baseline |
| q4_0 | 4.50 | 9.79 | +2.2% |
| tqk4_0 | 4.13 | 9.59 | +0.1% |
| tqk5_0j | 5.25 | 9.42 | -1.7% |
| tqk4_1j | 4.25 | 11.53 | +20.4% |
| tqk4_sj | 4.13 | 9.65 | +0.7% |
| tqk3_sj | 3.88 | 10.28 | +7.3% |
| tqk3b_sj | 3.75 | 13.51 | +41.0% |
| tqk2_sj | 2.75 | 97.81 | — |

### Throughput

| Type | bpv | pp512 (t/s) | tg128 (t/s) |
|------|-----|-------------|-------------|
| f16 | 16.00 | 4889 | 119 |
| q8_0 | 8.50 | 4807 | 118 |
| q4_0 | 4.50 | 4835 | 117 |
| tqk4_0 | 4.13 | 1682 | 76 |
| tqk5_0j | 5.25 | 1660 | 81 |
| tqk4_1j | 4.25 | 1839 | 83 |
| tqk3_sj | 3.88 | 1291 | 81 |
| tqk4_sj | 4.13 | 1267 | 79 |
| tqk3b_sj | 3.75 | 1357 | 80 |
| tqk2_sj | 2.75 | 821 | 80 |

## Notes

- **tqk5_0j** (5.25 bpv) achieves slightly *better* PPL than f16 on this model/eval, likely due to regularization effects of the Hadamard transform + QJL correction.
- **tqk4_sj** (4.13 bpv, calibrated split) matches f16 PPL within 0.7% while using 4x less KV cache memory.
- **tqk3_sj** (3.88 bpv) is the best quality-per-bit among split types.
- **tqk2_sj** (2.75 bpv) degrades significantly on this short evaluation — it may perform better on longer contexts where KV cache memory savings matter more, but quality loss is substantial.
- TQ types are slower than standard quantization types due to the per-block FWHT transform in the flash attention kernel. The split types additionally require channel map permutation. This overhead is most visible in prompt processing (pp512); text generation (tg128) is less affected since it is more memory-bound.
- Split types require offline calibration (`llama-tq-calibrate`) to identify outlier channels. Without calibration they use an identity permutation, which produces poor quality.

## CUDA Implementation Coverage

### d=128 Types (fully implemented)

| Operation | tqk4_0 | tqk5_0j | tqk4_1j | tqk3_sj | tqk4_sj | tqk3b_sj | tqk2_sj | tqv4_0 |
|-----------|--------|---------|---------|---------|---------|----------|---------|--------|
| get_rows | yes | yes | yes | yes | yes | yes | yes | yes |
| set_rows | yes | yes | yes | yes | yes | yes | yes | yes |
| FA (vec) | yes | yes | yes | yes | yes | yes | yes | yes |

### d=256 Types (get_rows + set_rows only)

| Operation | tqk4_0_d256 | tqk5_0j_d256 | tqk4_1j_d256 | tqk3_0j_d256 | tqv4_0_d256 |
|-----------|-------------|--------------|--------------|--------------|-------------|
| get_rows | yes | yes | yes | yes | yes |
| set_rows | yes | yes | yes | yes | yes |
| FA (vec) | no (CPU fallback) | no | no | no | no |
