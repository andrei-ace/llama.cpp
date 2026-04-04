# TurboQuant Metal Benchmarks

**Model:** Gemma4 26B A4B-it Q4_K_M  
**Hardware:** Apple M4 Pro (14-core, 36GB unified memory)  
**Date:** 2026-04-04  
**Branch:** `turboquant-gemma` @ `b949625ff`

**Benchmark scripts:**
- [turboquant_quality.py](https://github.com/eullm/eullm/blob/main/bench/turboquant_quality.py)
- [turboquant_math_accuracy.py](https://github.com/eullm/eullm/blob/main/bench/turboquant_math_accuracy.py)

## Speed (llama-bench, tg128, pp512)

| Config | pp512 (t/s) | tg128 (t/s) | vs q4_0 | KV bpv | KV MiB (262k ctx) |
|:------:|:-----------:|:-----------:|:-------:|:------:|:-----------------:|
| f16/f16 | 769 | 62.3 | +8.5% | 32.0 | ~7680 |
| q8_0/q8_0 | 758 | 58.3 | +1.4% | 17.0 | ~4080 |
| q4_0/q4_0 | 756 | 57.5 | baseline | 9.0 | ~1524 |
| **tq3j/q4_0** | **747** | **57.6** | **+0.3%** | **8.6** | **~1450** |
| **tq3j/tq3** | **745** | **56.5** | **-1.7%** | **7.1** | **~1202** |
| tq2j/tq2 | — | ~55.9 | -2.6% | 5.1 | ~860 |

### Long context (tg128 at depth)

| Config | d=128 | d=8192 | d=32768 |
|:------:|:-----:|:------:|:-------:|
| f16/f16 | 62.5 | 55.3 | 46.0 |
| q4_0/q4_0 | 57.4 | 42.8 | — |
| **tq3j/q4_0** | **57.3** | **45.6 (+7%)** | **32.1** |
| tq3j/tq3 | 55.8 | 43.4 (+1%) | — |

## Quality (31 tests: math, matrix, factual, logic, code, delayed-recall)

| Config | Score | Pct | Notes |
|:------:|:-----:|:---:|:------|
| f16/f16 | 31/31 | 100% | |
| q8_0/q8_0 | 31/31 | 100% | |
| q4_0/q4_0 | 31/31 | 100% | |
| **tq3j/q4_0** | **31/31** | **100%** | |
| **tq3j/tq3** | **30/31** | **96.8%** | 1 empty (token budget, not quality) |
| **tq2j/q4_0** | **30/31** | **96.8%** | 1 empty (token budget, not quality) |
| tq2j/tq2 | 26/31 | 83.9% | 3 empty + 2 wrong answers |

### Quality test details

28 direct tests + 3 delayed-recall matrix multiplication with ~500 tokens of filler:

- **Math (7):** 17×19, 2^10, GCD(48,36), sum(1..10), log2(256), 347×283, is 997 prime?
- **Matrix (5):** det([[3,8],[4,6]]), trace, det([[2,0],[0,5]]), 2×2 multiply (×2)
- **Factual (6):** Ljubljana, W, Au, Canberra, 206 bones, Pacific ocean
- **Logic (5):** roses syllogism, Fibonacci, geometric seq, discount price, distance
- **Code (5):** FizzBuzz, HTTPS port, def keyword, SSH port, len()
- **Delayed (3):** matrix A×B with 500 tokens of filler text between memorize and compute

### Failure analysis

**tq3j/q4_0 (0 failures):** perfect score.

**tq3j/tq3 (1 failure):**
- `math06` (347×283=98201): empty response — model's chain-of-thought for large multiplication exceeded 1024 token budget. Not a quantization error.

**tq2j/q4_0 (1 failure):**
- `delayed_D2`: empty response — token budget. K-side tq2j is fine; the tq2j/tq2 failures were V-side (tq2).

**tq2j/tq2 (5 failures):**
- `math06`: empty (token budget)
- `mat04` ([[1,2],[3,4]]×[[5,6],[7,8]]): empty
- `logic04` (discount price): empty
- `delayed_D2`: empty
- `delayed_D3` ([[3,1],[2,4]]×[[1,5],[3,2]]): truncated mid-thinking — 2-bit V quantization (tq2, no QJL) loses precision on recall-heavy tasks

## NIAH (Needle-in-a-Haystack)

8 factual needles embedded in a detective case file with dense filler.
Short prompt (~3K tokens, ~2.6K prompt tokens) and long prompt (~12K tokens with 80 filler paragraphs).

Tested via llama-server API with `max_tokens=2048`.

| Config | Short (3K) | Long (12K) |
|:------:|:----------:|:----------:|
| f16/f16 | 8/8 | 8/8 |
| q8_0/q8_0 | 8/8 | 8/8 |
| q4_0/q4_0 | 8/8 | 8/8 |
| tq3j/q4_0 | 8/8 | 8/8 |
| tq2j/q4_0 | 8/8 | 8/8 |
| tq3j/tq3 | 8/8 | 8/8 |
| tq2j/tq2 | 0/8 | **0/8** |

All configs except tq2j/tq2 pass 8/8 on both short and long NIAH.
Initial batch run showed some 0/8 scores for tq3j/tq3 and tq2j/q4_0
which were verified as **token budget issues** (non-deterministic thinking
chain length hitting max_tokens=2048). Clean reruns confirmed 8/8 with
answers appearing in `content` under budget.

**tq2j/tq2 is real degradation:** model outputs garbage (`"er-er-er-er..."`)
with no reasoning. The 2-bit V quantization (tq2, MSE only, no QJL) destroys
attention at 12K context.

Needles: gold coins (14, 31.1g), clock (11:47 PM), toxicology (potassium
cyanide 0.3 mg/L), cafe (Blue Parrot, $47.83), license plate (XR7),
phone calls (23 to Gerald Hoffman), handwritten note (lighthouse, Rothschild
documents), carpet pigment (Prussian blue, Hargrove Mills 1987).

## SET_ROWS overhead

Measured at d=0 (pure SET_ROWS + matmul, no FA):

| Config | t/s | SET_ROWS overhead vs f16 |
|:------:|:---:|:------------------------:|
| f16/f16 | 62.2 | baseline |
| tq3j/tq3 (noop SET_ROWS) | 58.2 | 0 ms (Q FWHT only) |
| tq3j/tq3 (real) | 56.3 | 0.55 ms/token |

SET_ROWS accounts for ~85% of the TQ overhead at short context.
