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
