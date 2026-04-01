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
- **Calibration**: Split types (tqk3_sj, tqk4_sj, tqk3_sjj, tqk2_sj) calibrated on Penn Treebank (PTB) train set using `llama-tq-calibrate -m model.gguf -f ptb/ptb.train.txt -o perms.bin -n 32000 -c 2048 --pre-rope`. The calibration identifies per-layer per-head outlier channels by accumulating K channel magnitudes over 32k tokens with context window 2048, using pre-RoPE activations. Non-split types and standard types need no calibration.
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
| tqk3_sjj | 3.75 | 13.51 | +41.0% |
| tqk2_sj | 2.75 | 97.81 | — |

### Throughput

After warp-shuffle FWHT optimization (see below):

| Type | bpv | pp512 (t/s) | % of q4_0 pp | tg128 (t/s) | % of q4_0 tg |
|------|-----|-------------|-------------|-------------|-------------|
| f16 | 16.00 | 4887 | 101% | 119 | 101% |
| q8_0 | 8.50 | 4807 | 99% | 118 | 100% |
| q4_0 | 4.50 | 4833 | 100% | 118 | 100% |
| tqk4_1j | 4.25 | 1879 | 39% | 84 | 71% |
| tqk5_0j | 5.25 | 1693 | 35% | 81 | 69% |
| tqk4_0 | 4.13 | 1713 | 35% | 77 | 65% |
| tqk3_sjj | 3.75 | 1488 | 31% | 82 | 69% |
| tqk3_sj | 3.88 | 1399 | 29% | 83 | 70% |
| tqk4_sj | 4.13 | 1328 | 27% | 80 | 68% |
| tqk2_sj | 2.75 | 994 | 21% | 82 | 69% |

Prefill is ~21-39% of q4_0 due to FWHT compute + warp divergence in the K dequant. Decode is ~65-71% of q4_0 since it is memory-bandwidth-bound and the FWHT cost is amortized.

### Warp-shuffle FWHT optimization

The FA Q pre-rotation was rewritten to use `__shfl_xor_sync()` warp shuffles instead of `__syncthreads()`-heavy shared memory FWHT, based on the WHT warp-shuffle approach from [@seanrasch](https://github.com/seanrasch). Thank you for the suggestion!

**Split types** (4 x FWHT-32): each of the 4 warps independently transforms one 32-element sub-block via shuffles — all 4 run in parallel with zero barriers. Previously 4 sequential shared-memory FWHTs with 24 `__syncthreads()` total.

**Non-split types** (FWHT-128): hybrid approach — warp shuffles for the 5 intra-warp butterfly stages (steps 1-16), shared memory for the 2 cross-warp stages (steps 32, 64). Reduces barriers from 7 to 3.

| Type | Before pp512 | After pp512 | Speedup |
|------|-------------|-------------|---------|
| tqk2_sj | 821 | 994 | **+21%** |
| tqk3_sjj | 1357 | 1488 | **+10%** |
| tqk3_sj | 1291 | 1399 | **+8%** |
| tqk4_sj | 1267 | 1328 | **+5%** |
| tqk4_0 | 1682 | 1713 | **+2%** |

## Notes

- **tqk5_0j** (5.25 bpv) achieves slightly *better* PPL than f16 on this model/eval, likely due to regularization effects of the Hadamard transform + QJL correction.
- **tqk4_sj** (4.13 bpv, calibrated split) matches f16 PPL within 0.7% while using 4x less KV cache memory.
- **tqk3_sj** (3.88 bpv) is the best quality-per-bit among split types.
- **tqk2_sj** (2.75 bpv) degrades significantly on this short evaluation — it may perform better on longer contexts where KV cache memory savings matter more, but quality loss is substantial.
- TQ types are slower than standard quantization types due to the per-block FWHT transform in the flash attention kernel. The split types additionally require channel map permutation. This overhead is most visible in prompt processing (pp512); text generation (tg128) is less affected since it is more memory-bound.
- The remaining prefill gap vs f16 (~3x for non-split, ~3.5x for split) is dominated by warp divergence in the FA K dequant inner loop (branch on `j < 32` for split types) and the FWHT compute cost itself, not the Q pre-rotation.
- Split types require offline calibration (`llama-tq-calibrate`) to identify outlier channels. Without calibration they use an identity permutation, which produces poor quality.

## CUDA Implementation Coverage

### d=128 Types (fully implemented)

| Operation | tqk4_0 | tqk5_0j | tqk4_1j | tqk3_sj | tqk4_sj | tqk3_sjj | tqk2_sj | tqv4_0 |
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
