// TurboQuant public API — rotation matrices, channel maps, QJL matrices
// Used by Metal/CUDA backends to access constant data for TQ kernels.
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

// Rotation matrix accessors (deterministic, generated from fixed seeds)
// All matrices are row-major float arrays.
const float * tq_get_rot_v_fwd(void);   // 128x128 — V cache forward rotation
const float * tq_get_rot_v_inv(void);   // 128x128 — V cache inverse rotation
const float * tq_get_rot_hi_fwd(void);  // 32x32   — K cache outlier forward rotation
const float * tq_get_rot_hi_inv(void);  // 32x32   — K cache outlier inverse rotation
const float * tq_get_rot_lo_fwd(void);  // 96x96   — K cache regular forward rotation
const float * tq_get_rot_lo_inv(void);  // 96x96   — K cache regular inverse rotation

// Rotation matrix sizes (number of floats)
int tq_get_rot_v_size(void);   // returns 128*128 = 16384
int tq_get_rot_hi_size(void);  // returns 32*32   = 1024
int tq_get_rot_lo_size(void);  // returns 96*96   = 9216

// Set/get runtime head dimension (128 or 256). Must be called before init.
void tq_set_head_dim(int dim);
int  tq_get_head_dim(void);

// Outlier mask management (loaded from GGUF calibration data)
// init allocates masks with identity default (channels 0..n_hi-1 as outliers)
void tq_init_outlier_masks(int n_layers, int n_heads, int head_dim);
// Load a permutation from GGUF: first head_dim/4 entries = outlier indices
void tq_set_outlier_mask_from_perm(int layer, int head, const uint8_t * perm, int head_dim);
void tq_free_outlier_masks(void);

// Per-layer-per-head channel maps (K cache only; V uses fixed split)
// outlier: array of head_dim/4 channel indices
// regular: array of head_dim*3/4 channel indices
void tq_get_channel_map(int layer, int head, int is_k, int * outlier, int * regular);

// Compact permutation: first n_hi entries are outlier indices, then n_lo regular indices
void tq_get_channel_perm(int layer, int head, int is_k, uint8_t * perm);

// Generate the QJL i.i.d. Gaussian matrix for a given dimension and seed.
// out must have dim*dim floats allocated. Uses same PRNG as CPU QJL forward/inverse.
// Seeds: QJL_SEED_32 = 0x514A4C20, QJL_SEED_96 = 0x514A4C60
void tq_get_qjl_matrix(float * out, int dim, uint64_t seed);

// Rebuild the global int32 channel map from current outlier masks.
// Call after tq_set_outlier_mask_from_perm() to make maps available to GPU backends.
void tq_upload_channel_maps_to_devices(void);

// Get the global channel map (built by tq_upload_channel_maps_to_devices).
// Returns NULL if not built. Layout: [n_layers][n_heads][head_dim] int32.
const int * tq_get_global_channel_map(int * out_n_layers, int * out_n_heads);

// Per-layer KV cache type recommendations (from calibration)
// type_index: 0=tqk3_sj, 1=tqk3_sjj, 2=tqk4_sj, 3=q8_0
void tq_set_layer_type_recommendations(const uint8_t * types, const float * outlier_pcts, int n_layers);
int  tq_get_layer_type_index(int layer);           // returns type_index or -1 if not set
float tq_get_layer_outlier_pct(int layer);          // returns outlier% or -1 if not set
int  tq_get_n_recommended_layers(void);             // returns n_layers or 0 if not set
void tq_free_layer_type_recommendations(void);

// Layer/head context for quantize/dequant/vec_dot (thread-local)
void tq_set_current_layer(int layer, int is_k);
void tq_set_current_head(int head);

#ifdef __cplusplus
}
#endif
