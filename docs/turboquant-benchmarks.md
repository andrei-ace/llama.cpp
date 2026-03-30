# TurboQuant KV Cache Benchmarks

## CUDA (NVIDIA RTX A6000, sm_86)

Model: Qwen3-8B Q4_K_M | CUDA Flash Attention (vec kernel)

### Perplexity (wikitext-2, 4 chunks, c=512)

| K type | V type | K bpv | V bpv | KV bpv | PPL |
|---|---|---|---|---|---|
| f16 | f16 | 16.00 | 16.00 | 16.00 | 10.52 |
| q8_0 | q8_0 | 8.00 | 8.00 | 8.00 | 10.51 |
| q4_0 | q4_0 | 4.50 | 4.50 | 4.50 | 11.14 |
| tqk_had_mse4 | tqv_had_mse4 | 4.13 | 4.13 | 4.13 | 10.84 |
| tqk_had_prod5 | tqv_had_mse4 | 5.25 | 4.13 | 4.69 | 10.74 |
| tqk_5hi_3lo_had | tqv_had_mse4 | 3.88 | 4.13 | 4.00 | 9.20 |
| tqk_5hi_3lo_had | q4_0 | 3.88 | 4.50 | 4.19 | 9.27 |
| tqk_5hi_3lo_had | f16 | 3.88 | 16.00 | 9.94 | 9.20 |
| tqk_had_prod4 | tqv_had_mse4 | 4.25 | 4.13 | 4.19 | 12.28 |

### Speed (tok/s)

| K type | V type | K bpv | V bpv | KV bpv | pp512 | tg128 |
|---|---|---|---|---|---|---|
| f16 | f16 | 16.00 | 16.00 | 16.00 | 4597 | 117 |
| q8_0 | q8_0 | 8.00 | 8.00 | 8.00 | 4493 | 116 |
| q4_0 | q4_0 | 4.50 | 4.50 | 4.50 | 4559 | 116 |
| tqk_had_prod4 | f16 | 4.25 | 16.00 | 10.13 | 2549 | 103 |
| tqk_had_mse4 | f16 | 4.13 | 16.00 | 10.07 | 2329 | 93 |
| tqk_had_prod5 | f16 | 5.25 | 16.00 | 10.63 | 2268 | 99 |
| tqk_5hi_3lo_had | f16 | 3.88 | 16.00 | 9.94 | 1679 | 99 |
| tqk_5hi_3lo_had | q4_0 | 3.88 | 4.50 | 4.19 | 1769 | 89 |
| tqk_had_mse4 | tqv_had_mse4 | 4.13 | 4.13 | 4.13 | 1595 | 77 |
| tqk_5hi_3lo_had | tqv_had_mse4 | 3.88 | 4.13 | 4.00 | 1327 | 66 |

### CUDA Notes

- First CUDA implementation using FA vec kernel. Not yet optimized (scalar centroid lookups, no vectorized unpacking).
- TQ K + f16 V: ~50-55% pp throughput, ~80-88% tg throughput vs f16 baseline.
- TQ K + TQV V: additional overhead from inverse FWHT on output (~35% pp, ~66% tg).
- 5hi_3lo_had: best quality (PPL 9.20) thanks to calibrated channel splitting.
- Requires `-fa 1` flag (FA auto-detection pending).

---

## Metal (Apple M4 Pro)

Model: Qwen3-8B Q4_K_M | Hardware: Apple M4 Pro | Metal Flash Attention

## Perplexity (wikitext-2, 8 chunks, c=512)

| K type | V type | K bpv | V bpv | KV bpv | PPL |
|---|---|---|---|---|---|
| f16 | f16 | 16.00 | 16.00 | 16.00 | 10.50 |
| q8_0 | q8_0 | 8.00 | 8.00 | 8.00 | 10.49 |
| q4_0 | q4_0 | 4.50 | 4.50 | 4.50 | 10.89 |
| tqk_had_mse4 | tqv_had_mse4 | 4.13 | 4.13 | 4.13 | 10.70 |
| tqk_had_prod5 | tqv_had_mse4 | 5.25 | 4.13 | 4.69 | 10.67 |
| tqk_5hi_3lo_had | tqv_had_mse4 | 3.88 | 4.13 | 4.00 | 10.78 |
| tqk_had_mse4 | q4_0 | 4.13 | 4.50 | 4.31 | 10.83 |
| tqk_5hi_3lo_had | q4_0 | 3.88 | 4.50 | 4.19 | 10.87 |
| tqk_had_prod4 | tqv_had_mse4 | 4.25 | 4.13 | 4.19 | 12.06 |

## Speed (tok/s)

| K type | V type | K bpv | V bpv | KV bpv | pp512 | tg128 |
|---|---|---|---|---|---|---|
| f16 | f16 | 16.00 | 16.00 | 16.00 | 458 | 47.1 |
| q8_0 | q8_0 | 8.00 | 8.00 | 8.00 | 452 | 45.8 |
| q4_0 | q4_0 | 4.50 | 4.50 | 4.50 | 453 | 45.7 |
| tqk_had_mse4 | tqv_had_mse4 | 4.13 | 4.13 | 4.13 | 443 | 39.8 |
| tqk_had_prod5 | tqv_had_mse4 | 5.25 | 4.13 | 4.69 | 435 | 35.2 |
| tqk_5hi_3lo_had | tqv_had_mse4 | 3.88 | 4.13 | 4.00 | 428 | 33.7 |
| tqk_had_mse4 | q4_0 | 4.13 | 4.50 | 4.31 | 448 | 42.3 |
| tqk_5hi_3lo_had | q4_0 | 3.88 | 4.50 | 4.19 | 432 | 35.8 |
| tqk_had_prod4 | tqv_had_mse4 | 4.25 | 4.13 | 4.19 | 436 | 34.5 |

## Sinks (5hi_3lo_had + tqv_had_mse4)

Sinks are not wired to Metal FA — the fp16 sink dispatch is CPU-only.
All sink values produce identical PPL on GPU.

| sink | PPL |
|---|---|
| 0 | 10.78 |
| 64 | 10.78 |
| 128 | 10.78 |
| 256 | 10.78 |

## Notes

- **had_mse4 + tqv**: Best quality-per-bit. PPL 10.70 at 4.13 bpv (vs q4_0 PPL 10.89 at 4.50 bpv).
- **had_mse4 + q4_0**: Fastest TQ combo. q4_0 V is bandwidth-efficient.
- **5hi_3lo_had**: Lowest K bpv (3.88) with calibrated channel split. Needs calibration phase (first 32 tokens as fp16).
- **had_prod5**: QJL correction adds overhead with no quality gain over had_mse4 when V is also quantized.
- **had_prod4**: 3-bit MSE is too lossy (PPL 12.06). Not recommended.
- **Calibration buffer**: Freed immediately after calibration. K(fp16) allocated in separate buffer from V.
- **tqv_had_mse4**: V quantized with FWHT rotation. Inverse FWHT applied to FA output (linear: FWHT(sum) = sum(FWHT)).
