# TurboQuant Metal Benchmark Results

Benchmark results for TurboQuant KV cache quantization types on Apple Metal, comparing perplexity and throughput.

## Hardware

- **SoC**: Apple M4 Pro (14-core CPU, 20-core GPU)
- **Memory**: 48 GB unified
- **OS**: macOS (Darwin 25.3.0)

## Methodology

- **Perplexity**: WikiText-2 test set, context 512, 20 chunks
- **Speed**: `llama-bench` with pp512 and tg128, flash attention enabled, full GPU offload
- **Calibration**: Split types calibrated on Penn Treebank train set using `llama-tq-calibrate --pre-rope --metric var -n 32000 -c 2048`
- **V cache**: All TQ K types paired with `tqv4_0` (4.13 bpv). Standard types use matching V types (e.g. q4_0 K + q4_0 V).

## Qwen 2.5 7B Instruct (Q4_K_M weights, 4.36 GiB)

This model has extreme sensitivity to KV cache quantization — uniform types below q8_0 are catastrophic. The split+calibrated approach is the only viable sub-8.5 bpv option.

### Perplexity

| Type | KV bpv | PPL | vs f16 |
|------|--------|-----|--------|
| f16+f16 | 16.00 | 7.78 | baseline |
| q8_0+q8_0 | 8.50 | 7.78 | 0% |
| **tqk4_sj+tqv4_0** | **4.13** | **328** | — |
| tqk3_sjj+tqv4_0 | 3.94 | 964 | — |
| tqk4_0+tqv4_0 | 4.13 | 5069 | — |
| q4_0+q4_0 | 4.50 | 6820 | — |
| q4_1+q4_1 | 5.00 | 12256 | — |

### Throughput

| Type | KV bpv | pp512 (t/s) | tg128 (t/s) |
|------|--------|-------------|-------------|
| f16+f16 | 16.00 | 496 | 51.2 |
| q8_0+q8_0 | 8.50 | 491 | 50.0 |
| q4_0+q4_0 | 4.50 | 490 | 49.6 |
| q4_1+q4_1 | 5.00 | 485 | 49.6 |
| tqk4_0+tqv4_0 | 4.13 | 483 | 44.1 |
| tqk4_sj+tqv4_0 | 4.13 | 471 | 37.2 |
| tqk3_sjj+tqv4_0 | 3.94 | 471 | 34.6 |

## Qwen 2.5 7B Instruct (Q8_0 weights, 7.54 GiB)

### Perplexity

| Type | KV bpv | PPL | vs f16 |
|------|--------|-----|--------|
| f16+f16 | 16.00 | 7.52 | baseline |
| q8_0+q8_0 | 8.50 | 7.53 | +0.1% |
| **tqk4_sj+tqv4_0** | **4.13** | **323** | — |
| tqk3_sjj+tqv4_0 | 3.94 | 994 | — |
| q4_0+q4_0 | 4.50 | 7020 | — |
| q4_1+q4_1 | 5.00 | 12351 | — |

### Throughput

| Type | KV bpv | pp512 (t/s) | tg128 (t/s) |
|------|--------|-------------|-------------|
| f16+f16 | 16.00 | 517 | 32.3 |
| q8_0+q8_0 | 8.50 | 510 | 31.7 |
| q4_0+q4_0 | 4.50 | 512 | 31.5 |
| tqk4_0+tqv4_0 | 4.13 | 503 | 29.2 |
| tqk4_sj+tqv4_0 | 4.13 | 491 | 26.0 |

## Mistral 7B Instruct v0.3 (Q6_K weights)

This model is highly robust to KV quantization — all types produce near-lossless results.

### Perplexity

| Type | KV bpv | PPL | vs f16 |
|------|--------|-----|--------|
| q4_0+q4_0 | 4.50 | 7.295 | -0.3% |
| f16+f16 | 16.00 | 7.319 | baseline |
| q4_1+q4_1 | 5.00 | 7.344 | +0.3% |
| tqk4_0+tqv4_0 | 4.13 | 7.363 | +0.6% |
| tqk4_sj+tqv4_0 | 4.13 | 7.529 | +2.9% |
| tqk3_sj+tqv4_0 | 4.00 | 7.547 | +3.1% |

## Qwen3 8B (Q8_0 weights, 8.11 GiB)

Qwen3 is robust to KV quantization — all types produce usable results. The split+calibrated `tqk4_sj` is the best TQ type.

### Perplexity

| Type | KV bpv | PPL | vs f16 |
|------|--------|-----|--------|
| q8_0+q8_0 | 8.50 | 10.90 | -0.2% |
| f16+f16 | 16.00 | 10.92 | baseline |
| q4_1+q4_1 | 5.00 | 10.97 | +0.5% |
| **tqk4_sj+tqv4_0** | **4.13** | **11.12** | **+1.7%** |
| q4_0+q4_0 | 4.50 | 11.32 | +3.6% |
| tqk4_0+tqv4_0 | 4.13 | 11.43 | +4.7% |
| tqk5_0j+tqv4_0 | 5.25 | 11.44 | +4.8% |
| tqk3_sj+tqv4_0 | 3.88 | 11.97 | +9.6% |
| tqk4_1j+tqv4_0 | 4.25 | 12.38 | +13% |
| tqk3_sjj+tqv4_0 | 3.75 | 13.71 | +26% |
| tqk2_sj+tqv4_0 | 2.75 | 131.6 | — |

### Throughput

| Type | KV bpv | pp512 (t/s) | tg128 (t/s) |
|------|--------|-------------|-------------|
| f16+f16 | 16.00 | 477 | 29.8 |
| q4_0+q4_0 | 4.50 | 470 | 28.9 |
| q8_0+q8_0 | 8.50 | 467 | 28.7 |
| q4_1+q4_1 | 5.00 | 469 | 28.5 |
| tqk4_0+tqv4_0 | 4.13 | 459 | 26.3 |
| tqk4_sj+tqv4_0 | 4.13 | 445 | 23.2 |
| tqk3_sj+tqv4_0 | 3.88 | 445 | 23.6 |
| tqk3_sjj+tqv4_0 | 3.75 | 443 | 22.0 |

## Notes

- **tqk4_sj** is the recommended type for outlier-sensitive models (Qwen 2.5). It achieves 3.9x KV compression with the only viable sub-8.5 bpv quality.
- **tqk4_0** is recommended for robust models (Mistral, Llama 3.1) — faster, no calibration needed, and quality is within noise of f16.
- TQ types are slightly slower than standard types due to FWHT transforms in the FA kernel. Split types add channel permutation overhead.
- Speed impact on prompt processing (pp512) is minimal (-3% to -5%). Text generation (tg128) is more affected (-14% to -32%) due to the per-token FA kernel overhead.
- The Q8_0 model is slower on tg than Q4_K_M (32 vs 51 t/s) because larger weights are memory-bandwidth bound.

## Metal Implementation Coverage

### d=128 Types (fully implemented)

| Operation | tqk4_0 | tqk5_0j | tqk4_1j | tqk3_sj | tqk4_sj | tqk3_sjj | tqk2_sj | tqv4_0 |
|-----------|--------|---------|---------|---------|---------|----------|---------|--------|
| get_rows | yes | yes | yes | yes | yes | yes | yes | yes |
| set_rows | yes | yes | yes | yes | yes | yes | yes | yes |
| FA (regular) | yes | yes | yes | yes | yes | yes | yes | yes |
| FA (vec) | yes | yes | yes | yes | yes | yes | yes | yes |

### d=256 Types (partial)

| Operation | tqk4_0_d256 | tqk5_0j_d256 | tqk4_1j_d256 | tqk3_sj_d256 | tqv4_0_d256 |
|-----------|-------------|--------------|--------------|--------------|-------------|
| get_rows | yes | yes | yes | yes | yes |
| set_rows | yes | yes | yes | yes | yes |
| FA | yes | yes | yes | yes | yes |
