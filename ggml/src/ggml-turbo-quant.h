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

// Set/get runtime head dimension (128 or 256). Must be called before calibration.
void tq_set_head_dim(int dim);
int  tq_get_head_dim(void);

// Dynamic calibration lifecycle: init allocates, free releases
void tq_init_calibration(int n_layers, int n_heads, int head_dim);
void tq_free_accum(void);        // free accumulators after locking outliers
void tq_free_calibration(void);  // free all calibration state

// Per-layer-per-head channel maps (K cache only; V uses fixed split)
// outlier: array of head_dim/4 channel indices
// regular: array of head_dim*3/4 channel indices
void tq_get_channel_map(int layer, int head, int is_k, int * outlier, int * regular);

// Generate the QJL i.i.d. Gaussian matrix for a given dimension and seed.
// out must have dim*dim floats allocated. Uses same PRNG as CPU QJL forward/inverse.
// Seeds: QJL_SEED_32 = 0x514A4C20, QJL_SEED_96 = 0x514A4C60
void tq_get_qjl_matrix(float * out, int dim, uint64_t seed);

#ifdef __cplusplus
}
#endif
