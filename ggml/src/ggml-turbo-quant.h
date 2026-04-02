// TurboQuant: FWHT decorrelation + MSE centroids + QJL correction for KV cache
#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// TQL: layered 32×q8 + 32×(3mse+1qjl) + 64×(2mse+1qjl)
void quantize_row_tql_ref (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void dequantize_row_tql   (const void  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void ggml_vec_dot_tql_f32 (int n, float * GGML_RESTRICT s, size_t bs,
                            const void * GGML_RESTRICT vx, size_t bx,
                            const void * GGML_RESTRICT vy, size_t by, int nrc);

// TQ3J: FWHT-128 + 3-bit MSE + 1-bit QJL
void quantize_row_tq3j_ref (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void dequantize_row_tq3j   (const void  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void ggml_vec_dot_tq3j_f32 (int n, float * GGML_RESTRICT s, size_t bs,
                              const void * GGML_RESTRICT vx, size_t bx,
                              const void * GGML_RESTRICT vy, size_t by, int nrc);

// TQ2J: FWHT-128 + 2-bit MSE + 1-bit QJL
void quantize_row_tq2j_ref (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void dequantize_row_tq2j   (const void  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void ggml_vec_dot_tq2j_f32 (int n, float * GGML_RESTRICT s, size_t bs,
                              const void * GGML_RESTRICT vx, size_t bx,
                              const void * GGML_RESTRICT vy, size_t by, int nrc);

// TQ3: FWHT-128 + 3-bit MSE only (for V cache)
void quantize_row_tq3_ref (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void dequantize_row_tq3   (const void  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void ggml_vec_dot_tq3_f32 (int n, float * GGML_RESTRICT s, size_t bs,
                             const void * GGML_RESTRICT vx, size_t bx,
                             const void * GGML_RESTRICT vy, size_t by, int nrc);

// TQ2: FWHT-128 + 2-bit MSE only (for V cache)
void quantize_row_tq2_ref (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void dequantize_row_tq2   (const void  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void ggml_vec_dot_tq2_f32 (int n, float * GGML_RESTRICT s, size_t bs,
                             const void * GGML_RESTRICT vx, size_t bx,
                             const void * GGML_RESTRICT vy, size_t by, int nrc);

// Wrapper functions for ggml_quantize_chunk
size_t quantize_tql (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                     int64_t nrows, int64_t n_per_row, const float * imatrix);
size_t quantize_tq3j(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                     int64_t nrows, int64_t n_per_row, const float * imatrix);
size_t quantize_tq2j(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                     int64_t nrows, int64_t n_per_row, const float * imatrix);
size_t quantize_tq3 (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                     int64_t nrows, int64_t n_per_row, const float * imatrix);
size_t quantize_tq2 (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                     int64_t nrows, int64_t n_per_row, const float * imatrix);

// d=256 and d=512 variants (macro-generated in ggml-turbo-quant.c)
#define TQ_DECLARE(SUFFIX)                                                     \
void quantize_row_##SUFFIX##_ref(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k); \
void dequantize_row_##SUFFIX(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k); \
void ggml_vec_dot_##SUFFIX##_f32(int n, float * GGML_RESTRICT s, size_t bs, \
    const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc); \
size_t quantize_##SUFFIX(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, \
    int64_t nrows, int64_t n_per_row, const float * imatrix);

TQ_DECLARE(tq3j_256)
TQ_DECLARE(tq2j_256)
TQ_DECLARE(tq3_256)
TQ_DECLARE(tq2_256)
TQ_DECLARE(tq3j_512)
TQ_DECLARE(tq2j_512)
TQ_DECLARE(tq3_512)
TQ_DECLARE(tq2_512)
#undef TQ_DECLARE

// (keep old-style declarations for backward compat)
// size_t quantize_tq3j_256(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
size_t quantize_tq2j_256(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
size_t quantize_tq3_256(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
size_t quantize_tq2_256(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
size_t quantize_tq3j_512(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
size_t quantize_tq2j_512(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
size_t quantize_tq3_512(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
size_t quantize_tq2_512(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
// ---------------------------------------------------------------------------
// Auto-resolve TQ type by head dimension
// tq3j → tq3j (d=128), tq3j_256 (d=256), tq3j_512 (d=512)
// ---------------------------------------------------------------------------

static inline enum ggml_type tq_resolve_type(enum ggml_type type, int head_dim) {
    switch (type) {
        case GGML_TYPE_TQ3J:
            if (head_dim == 256) return GGML_TYPE_TQ3J_256;
            if (head_dim >= 512) return GGML_TYPE_TQ3J_512;
            return GGML_TYPE_TQ3J;
        case GGML_TYPE_TQ2J:
            if (head_dim == 256) return GGML_TYPE_TQ2J_256;
            if (head_dim >= 512) return GGML_TYPE_TQ2J_512;
            return GGML_TYPE_TQ2J;
        case GGML_TYPE_TQ3:
            if (head_dim == 256) return GGML_TYPE_TQ3_256;
            if (head_dim >= 512) return GGML_TYPE_TQ3_512;
            return GGML_TYPE_TQ3;
        case GGML_TYPE_TQ2:
            if (head_dim == 256) return GGML_TYPE_TQ2_256;
            if (head_dim >= 512) return GGML_TYPE_TQ2_512;
            return GGML_TYPE_TQ2;
        default:
            return type;
    }
}

// ---------------------------------------------------------------------------
// Per-channel permutation support (loaded from calibration file)
// ---------------------------------------------------------------------------

// Initialize permutation storage. Call once before context creation.
// perms: [n_layers][n_heads][head_dim] uint8_t, channels sorted by importance (most important first).
// layer_map: [n_model_layers] int32_t, maps model layer -> calibration layer index (-1 if none).
void tq_init_perms(const uint8_t * perms, int n_layers, int n_heads, int head_dim,
                   const int32_t * layer_map, int n_model_layers);
void tq_free_perms(void);

// Thread-local layer/head context — set before quantize/dequant/vec_dot calls.
// When perms are loaded, TQL uses them to split channels by importance.
// When not loaded, TQL uses fixed sequential split (dims 0-31/32-63/64-127).
void tq_set_current_layer(int layer, int is_k);
void tq_set_current_head(int head);

#ifdef __cplusplus
}
#endif
