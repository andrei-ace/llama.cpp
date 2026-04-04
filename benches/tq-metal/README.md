# TurboQuant Metal Benchmarks

**Model:** Gemma4 26B A4B-it Q4_K_M  
**Hardware:** Apple M4 Pro (14-core, 48GB unified memory)  
**Date:** 2026-04-04 / 2026-04-05  
**Branch:** `turboquant-gemma`

**Benchmark scripts:**
- [turboquant_quality.py](https://github.com/eullm/eullm/blob/main/bench/turboquant_quality.py)
- [turboquant_math_accuracy.py](https://github.com/eullm/eullm/blob/main/bench/turboquant_math_accuracy.py)

## Speed (llama-bench, pp512 + tg128)

| Config | pp512 (t/s) | tg128 (t/s) | vs q4_0 | KV bpv |
|:------:|:-----------:|:-----------:|:-------:|:------:|
| f16/f16 | 770 | 62.4 | +8.1% | 32.00 |
| q8_0/q8_0 | 759 | 58.2 | +0.9% | 17.00 |
| q4_0/q4_0 | 758 | 57.7 | baseline | 9.00 |
| **tq3j/q4_0** | **750** | **57.9** | **+0.3%** | **8.62** |
| **tq2j/q4_0** | **760** | **57.5** | **-0.3%** | **7.62** |
| **tq3j/tq3** | **747** | **56.2** | **-2.6%** | **7.18** |
| **tq2j/tq3** | **753** | **56.3** | **-2.4%** | **6.18** |
| tq2j/tq2 | 757 | 56.8 | -1.6% | 5.18 |

### Long context (tg128 t/s at KV depth)

| Config | d=512 | d=4k | d=16k | d=32k | d=65k | d=131k | KV bpv |
|:------:|:-----:|:----:|:-----:|:-----:|:-----:|:------:|:------:|
| f16/f16 | 60.8 | 57.0 | 51.9 | 45.8 | 36.0 | 21.1 | 32.00 |
| q8_0/q8_0 | 58.0 | 51.3 | 40.2 | 31.3 | 21.2 | 12.9 | 17.00 |
| q4_0/q4_0 | 57.3 | 50.2 | 38.3 | 29.0 | 19.4 | 11.6 | 9.00 |
| **tq3j/q4_0** | **57.6** | **51.8** | **41.1** | **31.9** | 19.0 | 13.7 | **8.62** |
| **tq2j/q4_0** | **57.2** | **52.1** | **42.5** | **34.2** | **24.5** | **15.6** | **7.62** |
| **tq3j/tq3** | **56.0** | **50.3** | **39.4** | **30.1** | 20.8 | 12.7 | **7.18** |
| **tq2j/tq3** | **56.0** | **50.3** | **40.3** | **31.6** | **22.1** | **14.0** | **6.18** |
| tq2j/tq2 | 56.4 | 51.9 | 43.2 | 35.1 | 22.4 | 16.8 | 5.18 |

### TQ vs q4_0/q4_0 speed advantage at depth

| Config | d=4k | d=16k | d=32k | d=65k | d=131k |
|:------:|:----:|:-----:|:-----:|:-----:|:------:|
| tq3j/q4_0 | +3% | +7% | +10% | -2% | +18% |
| **tq2j/q4_0** | **+4%** | **+11%** | **+18%** | **+26%** | **+34%** |
| tq3j/tq3 | +0% | +3% | +4% | +7% | +9% |
| **tq2j/tq3** | **+0%** | **+5%** | **+9%** | **+14%** | **+21%** |

## Quality (37 tests, max_tokens=4096)

Categories: math (7), matrix (5), factual (6), logic (5), code (5),
Korean transliteration (6), delayed-recall matrix (3).

| Config | Score | Pct | KV bpv | Failures |
|:------:|:-----:|:---:|:------:|:---------|
| f16/f16 | 37/37 | 100% | 32.00 | none |
| q4_0/q4_0 | 37/37 | 100% | 9.00 | none |
| **tq3j/q4_0** | **37/37** | **100%** | **8.62** | **none** |
| **tq2j/q4_0** | **36/37** | **97.3%** | **7.62** | korean03 (empty) |
| **tq3j/tq3** | **36/37** | **97.3%** | **7.18** | math06 (empty) |
| **tq2j/tq3** | **36/37** | **97.3%** | **6.18** | math06 (empty) |
| tq2j/tq2 | 27/37 | 73.0% | 5.18 | 10 failures (see below) |

### Quality test details

- **Math (7):** 17×19, 2^10, GCD(48,36), sum(1..10), log2(256), 347×283, is 997 prime?
- **Matrix (5):** det([[3,8],[4,6]]), trace, det([[2,0],[0,5]]), 2×2 multiply (×2)
- **Factual (6):** Ljubljana, W, Au, Canberra, 206 bones, Pacific ocean
- **Logic (5):** roses syllogism, Fibonacci, geometric seq, discount price, distance
- **Code (5):** FizzBuzz, HTTPS port, def keyword, SSH port, len()
- **Korean (6):** Pauli→파울리, Bohr→보어, Schrödinger→슈뢰딩거, Heisenberg→하이젠베르크, Planck→플랑크, Fermi→페르미
- **Delayed (3):** matrix A×B with 500 tokens of filler text between memorize and compute

### Failure analysis

**tq3j/q4_0 (0 failures):** perfect score.

**tq2j/q4_0 (1 failure):**
- `korean03` (Schrödinger→슈뢰딩거): empty response — token budget.

**tq3j/tq3, tq2j/tq3 (1 failure each):**
- `math06` (347×283=98201): empty response even at max_tokens=4096. V=tq3
  quantization makes the model's chain-of-thought systematically longer.

**tq2j/tq2 (10 failures):**
- `math06`, `logic04`, `korean01`–`korean06`, `delayed_D2`, `delayed_D3`:
  all empty responses. tq2 V quantization (2-bit MSE, no QJL) makes the
  model's thinking chain systematically longer, exhausting the 4096 token
  budget on more tests than other configs. Combined with NIAH garbage output,
  tq2j/tq2 is not viable for production use.

## NIAH (Needle-in-a-Haystack)

8 factual needles embedded in a detective case file with dense filler.
Short prompt (~3K tokens, ~2.6K prompt tokens) and long prompt (~12K tokens with 80 filler paragraphs).

Tested via llama-server API with `max_tokens=2048`.

| Config | Short (3K) | Long (12K) | KV bpv |
|:------:|:----------:|:----------:|:------:|
| f16/f16 | 8/8 | 8/8 | 32.00 |
| q8_0/q8_0 | 8/8 | 8/8 | 17.00 |
| q4_0/q4_0 | 8/8 | 8/8 | 9.00 |
| **tq3j/q4_0** | **8/8** | **8/8** | **8.62** |
| **tq2j/q4_0** | **8/8** | **8/8** | **7.62** |
| **tq3j/tq3** | **8/8** | **8/8** | **7.18** |
| **tq2j/tq3** | **8/8** | **8/8** | **6.18** |
| tq2j/tq2 | 0/8 | **0/8** | 5.18 |

All configs except tq2j/tq2 pass 8/8 on both short and long NIAH.
Initial batch run showed some 0/8 scores for tq3j/tq3 and tq2j/q4_0
which were verified as **token budget issues** (non-deterministic thinking
chain length hitting max_tokens=2048). Clean reruns confirmed 8/8 with
answers appearing in `content` under budget.

**tq2j/tq2 is real degradation:** model outputs garbage (`"er-er-er-er..."`)
with no reasoning. The 2-bit V quantization (tq2, MSE only, no QJL) destroys
attention at 12K context.

Needles (exact values the model must retrieve):

| # | Detail | Answer |
|:-:|:-------|:-------|
| Q1 | Gold coins in safe | 14 coins, 31.1g each |
| Q2 | Stopped clock time | 11:47 PM |
| Q3 | Toxicology substance + concentration | potassium cyanide, 0.3 mg/L |
| Q4 | Cafe name + receipt amount | Blue Parrot Cafe, $47.83 |
| Q5 | License plate prefix | XR7 |
| Q6 | Phone calls to Gerald Hoffman | 23 calls |
| Q7 | Handwritten note instructions | lighthouse at midnight, Rothschild documents |
| Q8 | Carpet pigment + manufacturer | Prussian blue, Hargrove Mills (1987) |

## Recommendations

| Use case | Config | KV bpv | Why |
|:---------|:------:|:------:|:----|
| Best quality | **tq3j/q4_0** | 8.62 | 100% quality, matches q4_0 speed, +18% at 131k |
| Best memory efficiency | **tq2j/tq3** | 6.18 | 97.3% quality, 31% less KV than q4_0, +21% at 131k |
| Best long-context speed | **tq2j/q4_0** | 7.62 | 97.3% quality, +34% over q4_0 at 131k |
| Avoid | tq2j/tq2 | 5.18 | 73% quality, Korean fails, NIAH garbage |
