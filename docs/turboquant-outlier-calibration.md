# TurboQuant Outlier Calibration — Next Steps

## Problem

Outlier channels in KV cache activations need to be identified before TurboQuant
can split them into the high-precision (32 channels) and regular (96 channels)
partitions. Currently, detection uses the first vector's magnitudes, which is
unreliable. Per the QJL and RotateKV papers, outlier channels should be
identified from prompt-phase statistics.

## Proposed Approach

### Phase 1: Prompt with fp16 K cache

During prompt processing (prefill), the KV cache starts as fp16. While
processing the prompt, accumulate per-channel magnitude statistics:

```
channel_sum[layer][channel] += |K[token][channel]|
```

This runs inside `llama-kv-cache.cpp` during `ggml_set_rows` for K cache
writes. The accumulator is a `float[n_layer][128]` array — ~16KB total,
negligible overhead.

### Phase 2: Detect outliers after prompt

After the prompt batch completes (in `llama_decode` or `llama_context::decode`),
for each layer:

1. Sort `channel_sum[layer]` by magnitude
2. Top 32 = outlier channels for that layer
3. Store as `int outlier_ch[layer][32]` and `int regular_ch[layer][96]`
4. These are fixed for the rest of the session

### Phase 3: Convert prompt K cache to TurboQuant

Re-quantize the prompt's fp16 K cache entries into TurboQuant blocks using the
detected outlier channels. This is a one-time cost proportional to prompt length.

Alternatively: keep the prompt K cache as fp16 (wastes memory but avoids
re-quantization) and only use TurboQuant for generation tokens.

### Phase 4: Generation with TurboQuant

All new K vectors during generation use `quantize_row_tqk_35_ref` with the
layer-specific outlier channel assignment. The `vec_dot` function uses the same
assignment for query rotation.

## Implementation Details

### Where to accumulate

In `llama-kv-cache.cpp`, the `set_rows` operation writes K vectors to the cache.
Add a hook that, when the cache type is TQK and outliers haven't been detected
yet, accumulates magnitudes:

```cpp
// In llama_kv_cache or llama_kv_cache_unified
float * tq_channel_accum;   // [n_layer][head_dim] — accumulated |K| per channel
int     tq_accum_count;     // number of vectors accumulated
bool    tq_outliers_locked;  // set after prompt, prevents further accumulation
int   * tq_outlier_ch;      // [n_layer][32] — outlier channel indices per layer
int   * tq_regular_ch;      // [n_layer][96] — regular channel indices per layer
```

### Per-layer vs global

The papers show outlier channels differ per layer. Each layer should have its
own outlier set. This means the quantize/dequantize functions need a layer index
to look up the right channel permutation.

The current static globals in `ggml-turbo-quant.c` need to become per-layer
state, passed through the KV cache context.

### API change

The `quantize_row_tqk_35_ref` signature stays the same (it's called per-row
by ggml), but the outlier channel assignment needs to come from somewhere.
Options:

1. Thread-local storage keyed by layer index
2. A global registry indexed by layer
3. Encode the outlier indices in the block (expensive — 32 bytes per block)
4. Store in the KV cache context, pass to quantize via a callback

Option 2 is simplest for now: a global `tq_outlier_registry[MAX_LAYERS][32]`
set by the KV cache during calibration, read by the quantize functions.

### Triggering the switch

After `llama_decode` processes the prompt batch:

```cpp
if (cache.type_k is TQK && !cache.tq_outliers_locked) {
    cache.tq_detect_outliers_from_accum();
    cache.tq_outliers_locked = true;
    // Optionally re-quantize prompt K cache from fp16 to TQK
}
```

### Fallback

If no prompt is processed (e.g., empty context), fall back to the current
first-vector detection. This ensures the system always works, just with
potentially suboptimal outlier assignment.
