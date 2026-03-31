// TurboQuant set_rows — custom 5hi_3lo_had kernel needing channel map.
// The standard had types use inline __device__ functions in tq-set-rows.cuh.

#include "tq-set-rows.cuh"
#include "set-rows.cuh"

// Device-side channel map pointer defined in ggml-cuda.cu

// Forward declaration of channel map accessor (defined in ggml-cuda.cu)
extern int32_t * ggml_cuda_get_tq_channel_map_device(void);
extern int       ggml_cuda_get_tq_chmap_n_heads(void);

template <typename idx_t>
static __global__ void k_set_rows_tq_5hi_3lo_had(
        const float * __restrict__ src0,
        const idx_t * __restrict__ src1,
        block_tqk_5hi_3lo * __restrict__ dst,
        const int32_t * __restrict__ chmap,
        const int32_t n_kv_heads,
        const int32_t layer_idx,
        const int64_t ne00,
        const int64_t ne01,
        const int64_t ne02,
        const int64_t ne03,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const int64_t s10,
        const int64_t s11,
        const int64_t s12,
        const int64_t nb1,
        const int64_t nb2,
        const int64_t nb3) {

    const int64_t i = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;
    const int64_t ne_total = (ne00 * ne01 * ne02 * ne03) / 128;
    if (i >= ne_total) return;

    // Compute multi-dimensional indices
    int64_t tmp = i;
    const int64_t block_in_row = tmp % (ne00 / 128); tmp /= (ne00 / 128);
    const int64_t i01 = tmp % ne01; tmp /= ne01;
    const int64_t i02 = tmp % ne02; tmp /= ne02;
    const int64_t i03 = tmp;

    const int64_t i10 = i01;
    const int64_t i11 = i02; // broadcast
    const int64_t i12 = i03;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const float * src_block = src0 + i01*s01 + i02*s02 + i03*s03 + block_in_row * 128;
    block_tqk_5hi_3lo * dst_block = (block_tqk_5hi_3lo *)((char *)dst + dst_row*nb1 + i02*nb2 + i03*nb3) + block_in_row;

    // Head index from block position
    const int head = (int)(block_in_row % n_kv_heads);
    const int32_t * perm = chmap + ((int64_t)layer_idx * n_kv_heads + head) * 128;

    // Split into outlier/regular via channel map
    float hi_raw[32], lo_raw[96];
    for (int j = 0; j < 32; j++) hi_raw[j] = src_block[perm[j]];
    for (int j = 0; j < 96; j++) lo_raw[j] = src_block[perm[32 + j]];

    // FWHT rotation
    float hi_rot[32], lo_rot[96];
    for (int j = 0; j < 32; j++) hi_rot[j] = hi_raw[j];
    tq_fwht_local<32>(hi_rot);
    for (int j = 0; j < 96; j++) lo_rot[j] = lo_raw[j];
    tq_fwht_local<32>(lo_rot);
    tq_fwht_local<32>(lo_rot + 32);
    tq_fwht_local<32>(lo_rot + 64);

    // Per-subset norms
    float sum_hi = 0.0f, sum_lo = 0.0f;
    for (int j = 0; j < 32; j++) sum_hi += hi_rot[j] * hi_rot[j];
    for (int j = 0; j < 96; j++) sum_lo += lo_rot[j] * lo_rot[j];
    float norm_hi = sqrtf(sum_hi), norm_lo = sqrtf(sum_lo);

    dst_block->norm_hi  = __float2half(norm_hi);
    dst_block->norm_lo  = __float2half(norm_lo);
    dst_block->rnorm_hi = __float2half(0.0f);
    memset(dst_block->signs_hi, 0, sizeof(dst_block->signs_hi));

    if (norm_hi == 0.0f && norm_lo == 0.0f) {
        memset(dst_block->qs_hi, 0, sizeof(dst_block->qs_hi));
        memset(dst_block->qs_lo, 0, sizeof(dst_block->qs_lo));
        return;
    }

    float inv_hi = (norm_hi > 1e-12f) ? 1.0f / norm_hi : 0.0f;
    float inv_lo = (norm_lo > 1e-12f) ? 1.0f / norm_lo : 0.0f;

    // 4-bit MSE for hi (d=32)
    memset(dst_block->qs_hi, 0, sizeof(dst_block->qs_hi));
    for (int j = 0; j < 32; j++) tq_pk4(dst_block->qs_hi, j, tq_nearest(hi_rot[j] * inv_hi, tq_c16_d32, 16));

    // 3-bit MSE for lo (shared 96-D norm → d96 centroids)
    memset(dst_block->qs_lo, 0, sizeof(dst_block->qs_lo));
    for (int j = 0; j < 96; j++) tq_pk3(dst_block->qs_lo, j, tq_nearest(lo_rot[j] * inv_lo, tq_c8_d96, 8));

    // QJL on hi residual
    float yhi[32];
    for (int j = 0; j < 32; j++) yhi[j] = tq_c16_d32[tq_up4(dst_block->qs_hi, j)];
    tq_fwht_local<32>(yhi);

    float resid_hi[32];
    float rnorm_sq = 0.0f;
    for (int j = 0; j < 32; j++) { resid_hi[j] = hi_raw[j] - norm_hi * yhi[j]; rnorm_sq += resid_hi[j] * resid_hi[j]; }
    tq_fwht_local<32>(resid_hi);
    for (int j = 0; j < 32; j++) {
        if (resid_hi[j] >= 0.0f) dst_block->signs_hi[j / 8] |= (uint8_t)(1 << (j % 8));
    }
    dst_block->rnorm_hi = __float2half(sqrtf(rnorm_sq));
}

// ---------------------------------------------------------------------------
// 6hi_3lo_had: 5-bit MSE on hi + QJL, 3-bit MSE on lo, no QJL on lo
// ---------------------------------------------------------------------------

template <typename idx_t>
static __global__ void k_set_rows_tq_6hi_3lo_had(
        const float * __restrict__ src0,
        const idx_t * __restrict__ src1,
        block_tqk_6hi_3lo * __restrict__ dst,
        const int32_t * __restrict__ chmap,
        const int32_t n_kv_heads,
        const int32_t layer_idx,
        const int64_t ne00,
        const int64_t ne01,
        const int64_t ne02,
        const int64_t ne03,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const int64_t s10,
        const int64_t s11,
        const int64_t s12,
        const int64_t nb1,
        const int64_t nb2,
        const int64_t nb3) {

    const int64_t i = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;
    const int64_t ne_total = (ne00 * ne01 * ne02 * ne03) / 128;
    if (i >= ne_total) return;

    int64_t tmp = i;
    const int64_t block_in_row = tmp % (ne00 / 128); tmp /= (ne00 / 128);
    const int64_t i01 = tmp % ne01; tmp /= ne01;
    const int64_t i02 = tmp % ne02; tmp /= ne02;
    const int64_t i03 = tmp;

    const int64_t dst_row = *(src1 + i01*s10 + i02*s11 + i03*s12);

    const float * src_block = src0 + i01*s01 + i02*s02 + i03*s03 + block_in_row * 128;
    block_tqk_6hi_3lo * dst_block = (block_tqk_6hi_3lo *)((char *)dst + dst_row*nb1 + i02*nb2 + i03*nb3) + block_in_row;

    const int head = (int)(block_in_row % n_kv_heads);
    const int32_t * perm = chmap + ((int64_t)layer_idx * n_kv_heads + head) * 128;

    // Split into outlier/regular via channel map
    float hi_raw[32], lo_raw[96];
    for (int j = 0; j < 32; j++) hi_raw[j] = src_block[perm[j]];
    for (int j = 0; j < 96; j++) lo_raw[j] = src_block[perm[32 + j]];

    // FWHT rotation
    float hi_rot[32], lo_rot[96];
    for (int j = 0; j < 32; j++) hi_rot[j] = hi_raw[j];
    tq_fwht_local<32>(hi_rot);
    for (int j = 0; j < 96; j++) lo_rot[j] = lo_raw[j];
    tq_fwht_local<32>(lo_rot);
    tq_fwht_local<32>(lo_rot + 32);
    tq_fwht_local<32>(lo_rot + 64);

    // Per-subset norms
    float sum_hi = 0.0f, sum_lo = 0.0f;
    for (int j = 0; j < 32; j++) sum_hi += hi_rot[j] * hi_rot[j];
    for (int j = 0; j < 96; j++) sum_lo += lo_rot[j] * lo_rot[j];
    float norm_hi = sqrtf(sum_hi), norm_lo = sqrtf(sum_lo);

    dst_block->norm_hi  = __float2half(norm_hi);
    dst_block->norm_lo  = __float2half(norm_lo);
    dst_block->rnorm_hi = __float2half(0.0f);
    memset(dst_block->signs_hi, 0, sizeof(dst_block->signs_hi));

    if (norm_hi == 0.0f && norm_lo == 0.0f) {
        memset(dst_block->qs_hi, 0, sizeof(dst_block->qs_hi));
        memset(dst_block->qs_lo, 0, sizeof(dst_block->qs_lo));
        return;
    }

    float inv_hi = (norm_hi > 1e-12f) ? 1.0f / norm_hi : 0.0f;
    float inv_lo = (norm_lo > 1e-12f) ? 1.0f / norm_lo : 0.0f;

    // 5-bit MSE for hi (d=32, 32 centroids)
    memset(dst_block->qs_hi, 0, sizeof(dst_block->qs_hi));
    for (int j = 0; j < 32; j++) tq_pk5(dst_block->qs_hi, j, tq_nearest(hi_rot[j] * inv_hi, tq_c32_d32, 32));

    // 3-bit MSE for lo (shared 96-D norm → d96 centroids)
    memset(dst_block->qs_lo, 0, sizeof(dst_block->qs_lo));
    for (int j = 0; j < 96; j++) tq_pk3(dst_block->qs_lo, j, tq_nearest(lo_rot[j] * inv_lo, tq_c8_d96, 8));

    // QJL on hi residual
    float yhi[32];
    for (int j = 0; j < 32; j++) yhi[j] = tq_c32_d32[tq_up5(dst_block->qs_hi, j)];
    tq_fwht_local<32>(yhi);

    float resid_hi[32];
    float rnorm_sq = 0.0f;
    for (int j = 0; j < 32; j++) { resid_hi[j] = hi_raw[j] - norm_hi * yhi[j]; rnorm_sq += resid_hi[j] * resid_hi[j]; }
    tq_fwht_local<32>(resid_hi);
    for (int j = 0; j < 32; j++) {
        if (resid_hi[j] >= 0.0f) dst_block->signs_hi[j / 8] |= (uint8_t)(1 << (j % 8));
    }
    dst_block->rnorm_hi = __float2half(sqrtf(rnorm_sq));
}

// ---------------------------------------------------------------------------
// 2hi_1lo_had: 2-bit MSE + QJL on hi, 1-bit MSE + QJL on lo
// ---------------------------------------------------------------------------

template <typename idx_t>
static __global__ void k_set_rows_tq_2hi_1lo_had(
        const float * __restrict__ src0,
        const idx_t * __restrict__ src1,
        block_tqk_2hi_1lo * __restrict__ dst,
        const int32_t * __restrict__ chmap,
        const int32_t n_kv_heads,
        const int32_t layer_idx,
        const int64_t ne00,
        const int64_t ne01,
        const int64_t ne02,
        const int64_t ne03,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const int64_t s10,
        const int64_t s11,
        const int64_t s12,
        const int64_t nb1,
        const int64_t nb2,
        const int64_t nb3) {

    const int64_t i = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;
    const int64_t ne_total = (ne00 * ne01 * ne02 * ne03) / 128;
    if (i >= ne_total) return;

    int64_t tmp = i;
    const int64_t block_in_row = tmp % (ne00 / 128); tmp /= (ne00 / 128);
    const int64_t i01 = tmp % ne01; tmp /= ne01;
    const int64_t i02 = tmp % ne02; tmp /= ne02;
    const int64_t i03 = tmp;

    const int64_t dst_row = *(src1 + i01*s10 + i02*s11 + i03*s12);

    const float * src_block = src0 + i01*s01 + i02*s02 + i03*s03 + block_in_row * 128;
    block_tqk_2hi_1lo * dst_block = (block_tqk_2hi_1lo *)((char *)dst + dst_row*nb1 + i02*nb2 + i03*nb3) + block_in_row;

    const int head = (int)(block_in_row % n_kv_heads);
    const int32_t * perm = chmap + ((int64_t)layer_idx * n_kv_heads + head) * 128;

    float hi_raw[32], lo_raw[96];
    for (int j = 0; j < 32; j++) hi_raw[j] = src_block[perm[j]];
    for (int j = 0; j < 96; j++) lo_raw[j] = src_block[perm[32 + j]];

    float hi_rot[32], lo_rot[96];
    for (int j = 0; j < 32; j++) hi_rot[j] = hi_raw[j];
    tq_fwht_local<32>(hi_rot);
    for (int j = 0; j < 96; j++) lo_rot[j] = lo_raw[j];
    tq_fwht_local<32>(lo_rot);
    tq_fwht_local<32>(lo_rot + 32);
    tq_fwht_local<32>(lo_rot + 64);

    float sum_hi = 0.0f, sum_lo = 0.0f;
    for (int j = 0; j < 32; j++) sum_hi += hi_rot[j] * hi_rot[j];
    for (int j = 0; j < 96; j++) sum_lo += lo_rot[j] * lo_rot[j];
    float norm_hi = sqrtf(sum_hi), norm_lo = sqrtf(sum_lo);

    dst_block->norm_hi  = __float2half(norm_hi);
    dst_block->norm_lo  = __float2half(norm_lo);
    dst_block->rnorm_hi = __float2half(0.0f);
    dst_block->rnorm_lo = __float2half(0.0f);
    memset(dst_block->signs_hi, 0, sizeof(dst_block->signs_hi));
    memset(dst_block->signs_lo, 0, sizeof(dst_block->signs_lo));

    if (norm_hi == 0.0f && norm_lo == 0.0f) {
        memset(dst_block->qs_hi, 0, sizeof(dst_block->qs_hi));
        memset(dst_block->qs_lo, 0, sizeof(dst_block->qs_lo));
        return;
    }

    float inv_hi = (norm_hi > 1e-12f) ? 1.0f / norm_hi : 0.0f;
    float inv_lo = (norm_lo > 1e-12f) ? 1.0f / norm_lo : 0.0f;

    // 2-bit MSE for hi (d=32, 4 centroids)
    memset(dst_block->qs_hi, 0, sizeof(dst_block->qs_hi));
    for (int j = 0; j < 32; j++) tq_pk2(dst_block->qs_hi, j, tq_nearest(hi_rot[j] * inv_hi, tq_c4_d32, 4));

    // 1-bit MSE for lo (d=96, 2 centroids)
    memset(dst_block->qs_lo, 0, sizeof(dst_block->qs_lo));
    for (int j = 0; j < 96; j++) tq_pk1(dst_block->qs_lo, j, tq_nearest(lo_rot[j] * inv_lo, tq_c2_d96, 2));

    // QJL on hi residual
    float yhi[32];
    for (int j = 0; j < 32; j++) yhi[j] = tq_c4_d32[tq_up2(dst_block->qs_hi, j)];
    tq_fwht_local<32>(yhi);
    float r_hi[32];
    float rn_hi = 0.0f;
    for (int j = 0; j < 32; j++) { r_hi[j] = hi_raw[j] - norm_hi * yhi[j]; rn_hi += r_hi[j] * r_hi[j]; }
    tq_fwht_local<32>(r_hi);
    for (int j = 0; j < 32; j++) {
        if (r_hi[j] >= 0.0f) dst_block->signs_hi[j / 8] |= (uint8_t)(1 << (j % 8));
    }
    dst_block->rnorm_hi = __float2half(sqrtf(rn_hi));

    // QJL on lo residual
    float ylo[96];
    for (int j = 0; j < 96; j++) ylo[j] = tq_c2_d96[tq_up1(dst_block->qs_lo, j)];
    tq_fwht_local<32>(ylo);
    tq_fwht_local<32>(ylo + 32);
    tq_fwht_local<32>(ylo + 64);
    float r_lo[96];
    float rn_lo = 0.0f;
    for (int j = 0; j < 96; j++) { r_lo[j] = lo_raw[j] - norm_lo * ylo[j]; rn_lo += r_lo[j] * r_lo[j]; }
    tq_fwht_local<32>(r_lo);
    tq_fwht_local<32>(r_lo + 32);
    tq_fwht_local<32>(r_lo + 64);
    for (int j = 0; j < 96; j++) {
        if (r_lo[j] >= 0.0f) dst_block->signs_lo[j / 8] |= (uint8_t)(1 << (j % 8));
    }
    dst_block->rnorm_lo = __float2half(sqrtf(rn_lo));
}

// ---------------------------------------------------------------------------
// 3hi_2lo_had: 3-bit MSE + QJL on hi, 2-bit MSE + QJL on lo
// ---------------------------------------------------------------------------

template <typename idx_t>
static __global__ void k_set_rows_tq_3hi_2lo_had(
        const float * __restrict__ src0,
        const idx_t * __restrict__ src1,
        block_tqk_3hi_2lo * __restrict__ dst,
        const int32_t * __restrict__ chmap,
        const int32_t n_kv_heads,
        const int32_t layer_idx,
        const int64_t ne00,
        const int64_t ne01,
        const int64_t ne02,
        const int64_t ne03,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const int64_t s10,
        const int64_t s11,
        const int64_t s12,
        const int64_t nb1,
        const int64_t nb2,
        const int64_t nb3) {

    const int64_t i = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;
    const int64_t ne_total = (ne00 * ne01 * ne02 * ne03) / 128;
    if (i >= ne_total) return;

    int64_t tmp = i;
    const int64_t block_in_row = tmp % (ne00 / 128); tmp /= (ne00 / 128);
    const int64_t i01 = tmp % ne01; tmp /= ne01;
    const int64_t i02 = tmp % ne02; tmp /= ne02;
    const int64_t i03 = tmp;

    const int64_t dst_row = *(src1 + i01*s10 + i02*s11 + i03*s12);

    const float * src_block = src0 + i01*s01 + i02*s02 + i03*s03 + block_in_row * 128;
    block_tqk_3hi_2lo * dst_block = (block_tqk_3hi_2lo *)((char *)dst + dst_row*nb1 + i02*nb2 + i03*nb3) + block_in_row;

    const int head = (int)(block_in_row % n_kv_heads);
    const int32_t * perm = chmap + ((int64_t)layer_idx * n_kv_heads + head) * 128;

    float hi_raw[32], lo_raw[96];
    for (int j = 0; j < 32; j++) hi_raw[j] = src_block[perm[j]];
    for (int j = 0; j < 96; j++) lo_raw[j] = src_block[perm[32 + j]];

    float hi_rot[32], lo_rot[96];
    for (int j = 0; j < 32; j++) hi_rot[j] = hi_raw[j];
    tq_fwht_local<32>(hi_rot);
    for (int j = 0; j < 96; j++) lo_rot[j] = lo_raw[j];
    tq_fwht_local<32>(lo_rot);
    tq_fwht_local<32>(lo_rot + 32);
    tq_fwht_local<32>(lo_rot + 64);

    float sum_hi = 0.0f, sum_lo = 0.0f;
    for (int j = 0; j < 32; j++) sum_hi += hi_rot[j] * hi_rot[j];
    for (int j = 0; j < 96; j++) sum_lo += lo_rot[j] * lo_rot[j];
    float norm_hi = sqrtf(sum_hi), norm_lo = sqrtf(sum_lo);

    dst_block->norm_hi  = __float2half(norm_hi);
    dst_block->norm_lo  = __float2half(norm_lo);
    dst_block->rnorm_hi = __float2half(0.0f);
    dst_block->rnorm_lo = __float2half(0.0f);
    memset(dst_block->signs_hi, 0, sizeof(dst_block->signs_hi));
    memset(dst_block->signs_lo, 0, sizeof(dst_block->signs_lo));

    if (norm_hi == 0.0f && norm_lo == 0.0f) {
        memset(dst_block->qs_hi, 0, sizeof(dst_block->qs_hi));
        memset(dst_block->qs_lo, 0, sizeof(dst_block->qs_lo));
        return;
    }

    float inv_hi = (norm_hi > 1e-12f) ? 1.0f / norm_hi : 0.0f;
    float inv_lo = (norm_lo > 1e-12f) ? 1.0f / norm_lo : 0.0f;

    // 3-bit MSE for hi (d=32, 8 centroids)
    memset(dst_block->qs_hi, 0, sizeof(dst_block->qs_hi));
    for (int j = 0; j < 32; j++) tq_pk3(dst_block->qs_hi, j, tq_nearest(hi_rot[j] * inv_hi, tq_c8_d32, 8));

    // 2-bit MSE for lo (d=96, 4 centroids)
    memset(dst_block->qs_lo, 0, sizeof(dst_block->qs_lo));
    for (int j = 0; j < 96; j++) tq_pk2(dst_block->qs_lo, j, tq_nearest(lo_rot[j] * inv_lo, tq_c4_d96, 4));

    // QJL on hi residual
    float yhi[32];
    for (int j = 0; j < 32; j++) yhi[j] = tq_c8_d32[tq_up3(dst_block->qs_hi, j)];
    tq_fwht_local<32>(yhi);
    float r_hi[32];
    float rn_hi = 0.0f;
    for (int j = 0; j < 32; j++) { r_hi[j] = hi_raw[j] - norm_hi * yhi[j]; rn_hi += r_hi[j] * r_hi[j]; }
    tq_fwht_local<32>(r_hi);
    for (int j = 0; j < 32; j++) {
        if (r_hi[j] >= 0.0f) dst_block->signs_hi[j / 8] |= (uint8_t)(1 << (j % 8));
    }
    dst_block->rnorm_hi = __float2half(sqrtf(rn_hi));

    // QJL on lo residual
    float ylo[96];
    for (int j = 0; j < 96; j++) ylo[j] = tq_c4_d96[tq_up2(dst_block->qs_lo, j)];
    tq_fwht_local<32>(ylo);
    tq_fwht_local<32>(ylo + 32);
    tq_fwht_local<32>(ylo + 64);
    float r_lo[96];
    float rn_lo = 0.0f;
    for (int j = 0; j < 96; j++) { r_lo[j] = lo_raw[j] - norm_lo * ylo[j]; rn_lo += r_lo[j] * r_lo[j]; }
    tq_fwht_local<32>(r_lo);
    tq_fwht_local<32>(r_lo + 32);
    tq_fwht_local<32>(r_lo + 64);
    for (int j = 0; j < 96; j++) {
        if (r_lo[j] >= 0.0f) dst_block->signs_lo[j / 8] |= (uint8_t)(1 << (j % 8));
    }
    dst_block->rnorm_lo = __float2half(sqrtf(rn_lo));
}

// ---------------------------------------------------------------------------
// Host dispatch — helper macro for split-type set_rows
// ---------------------------------------------------------------------------

#define DEFINE_TQ_SPLIT_SET_ROWS_DISPATCH(suffix, block_type, blk_sz) \
void ggml_cuda_op_set_rows_tq_##suffix(ggml_backend_cuda_context & ctx, ggml_tensor * dst) { \
    const ggml_tensor * src0 = dst->src[0]; \
    const ggml_tensor * src1 = dst->src[1]; \
    GGML_ASSERT(src0->type == GGML_TYPE_F32); \
    GGML_ASSERT(src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32); \
    GGML_TENSOR_BINARY_OP_LOCALS \
    cudaStream_t stream = ctx.stream(); \
    int32_t layer_idx = 0; \
    const char * lp = strstr(dst->name, "_l"); \
    if (lp) layer_idx = atoi(lp + 2); \
    int32_t * chmap = ggml_cuda_get_tq_channel_map_device(); \
    int n_kv_heads = ggml_cuda_get_tq_chmap_n_heads(); \
    if (n_kv_heads < 1) n_kv_heads = (int)(ne00 / (blk_sz)); \
    GGML_ASSERT(ne00 % (blk_sz) == 0); \
    const int64_t ne_total = (ne00 * ne01 * ne02 * ne03) / (blk_sz); \
    const int num_blocks = (int)((ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE); \
    const int64_t s01 = nb01 / sizeof(float); \
    const int64_t s02 = nb02 / sizeof(float); \
    const int64_t s03 = nb03 / sizeof(float); \
    if (src1->type == GGML_TYPE_I64) { \
        const int64_t s10 = nb10 / sizeof(int64_t); \
        const int64_t s11 = nb11 / sizeof(int64_t); \
        const int64_t s12 = nb12 / sizeof(int64_t); \
        k_set_rows_tq_##suffix<<<num_blocks, CUDA_SET_ROWS_BLOCK_SIZE, 0, stream>>>( \
            (const float *)src0->data, (const int64_t *)src1->data, (block_type *)dst->data, \
            chmap, n_kv_heads, layer_idx, \
            ne00, ne01, ne02, ne03, s01, s02, s03, s10, s11, s12, nb1, nb2, nb3); \
    } else { \
        const int64_t s10 = nb10 / sizeof(int32_t); \
        const int64_t s11 = nb11 / sizeof(int32_t); \
        const int64_t s12 = nb12 / sizeof(int32_t); \
        k_set_rows_tq_##suffix<<<num_blocks, CUDA_SET_ROWS_BLOCK_SIZE, 0, stream>>>( \
            (const float *)src0->data, (const int32_t *)src1->data, (block_type *)dst->data, \
            chmap, n_kv_heads, layer_idx, \
            ne00, ne01, ne02, ne03, s01, s02, s03, s10, s11, s12, nb1, nb2, nb3); \
    } \
}

DEFINE_TQ_SPLIT_SET_ROWS_DISPATCH(6hi_3lo_had,      block_tqk_6hi_3lo,     128)
DEFINE_TQ_SPLIT_SET_ROWS_DISPATCH(2hi_1lo_had,      block_tqk_2hi_1lo,     128)
DEFINE_TQ_SPLIT_SET_ROWS_DISPATCH(3hi_2lo_had,      block_tqk_3hi_2lo,     128)

// ---------------------------------------------------------------------------
// 5hi_3lo_had_d256: 64/192 split, 4-bit + QJL on hi, 3-bit on lo
// ---------------------------------------------------------------------------

template <typename idx_t>
static __global__ void k_set_rows_tq_5hi_3lo_had_d256(
        const float * __restrict__ src0,
        const idx_t * __restrict__ src1,
        block_tqk_5hi_3lo_d256 * __restrict__ dst,
        const int32_t * __restrict__ chmap,
        const int32_t n_kv_heads,
        const int32_t layer_idx,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s10, const int64_t s11, const int64_t s12,
        const int64_t nb1, const int64_t nb2, const int64_t nb3) {

    const int64_t i = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;
    const int64_t ne_total = (ne00 * ne01 * ne02 * ne03) / 256;
    if (i >= ne_total) return;

    int64_t tmp = i;
    const int64_t block_in_row = tmp % (ne00 / 256); tmp /= (ne00 / 256);
    const int64_t i01 = tmp % ne01; tmp /= ne01;
    const int64_t i02 = tmp % ne02; tmp /= ne02;
    const int64_t i03 = tmp;

    const int64_t dst_row = *(src1 + i01*s10 + i02*s11 + i03*s12);
    const float * src_block = src0 + i01*s01 + i02*s02 + i03*s03 + block_in_row * 256;
    block_tqk_5hi_3lo_d256 * dst_block = (block_tqk_5hi_3lo_d256 *)((char *)dst + dst_row*nb1 + i02*nb2 + i03*nb3) + block_in_row;

    const int head = (int)(block_in_row % n_kv_heads);
    const int32_t * perm = chmap + ((int64_t)layer_idx * n_kv_heads + head) * 256;

    float hi_raw[64], lo_raw[192];
    for (int j = 0; j < 64; j++)  hi_raw[j] = src_block[perm[j]];
    for (int j = 0; j < 192; j++) lo_raw[j] = src_block[perm[64 + j]];

    float hi_rot[64], lo_rot[192];
    for (int j = 0; j < 64; j++) hi_rot[j] = hi_raw[j];
    tq_fwht_local<64>(hi_rot);
    for (int j = 0; j < 192; j++) lo_rot[j] = lo_raw[j];
    for (int b = 0; b < 3; b++) tq_fwht_local<64>(lo_rot + b * 64);

    float sum_hi = 0.0f, sum_lo = 0.0f;
    for (int j = 0; j < 64; j++)  sum_hi += hi_rot[j] * hi_rot[j];
    for (int j = 0; j < 192; j++) sum_lo += lo_rot[j] * lo_rot[j];
    float norm_hi = sqrtf(sum_hi), norm_lo = sqrtf(sum_lo);

    dst_block->norm_hi  = __float2half(norm_hi);
    dst_block->norm_lo  = __float2half(norm_lo);
    dst_block->rnorm_hi = __float2half(0.0f);
    memset(dst_block->signs_hi, 0, sizeof(dst_block->signs_hi));

    if (norm_hi == 0.0f && norm_lo == 0.0f) {
        memset(dst_block->qs_hi, 0, sizeof(dst_block->qs_hi));
        memset(dst_block->qs_lo, 0, sizeof(dst_block->qs_lo));
        return;
    }

    float inv_hi = (norm_hi > 1e-12f) ? 1.0f / norm_hi : 0.0f;
    float inv_lo = (norm_lo > 1e-12f) ? 1.0f / norm_lo : 0.0f;

    memset(dst_block->qs_hi, 0, sizeof(dst_block->qs_hi));
    for (int j = 0; j < 64; j++) tq_pk4(dst_block->qs_hi, j, tq_nearest(hi_rot[j] * inv_hi, tq_c16_d64, 16));

    memset(dst_block->qs_lo, 0, sizeof(dst_block->qs_lo));
    for (int j = 0; j < 192; j++) tq_pk3(dst_block->qs_lo, j, tq_nearest(lo_rot[j] * inv_lo, tq_c8_d192, 8));

    // QJL on hi residual
    float yhi[64];
    for (int j = 0; j < 64; j++) yhi[j] = tq_c16_d64[tq_up4(dst_block->qs_hi, j)];
    tq_fwht_local<64>(yhi);
    float r_hi[64];
    float rnorm_sq = 0.0f;
    for (int j = 0; j < 64; j++) { r_hi[j] = hi_raw[j] - norm_hi * yhi[j]; rnorm_sq += r_hi[j] * r_hi[j]; }
    tq_fwht_local<64>(r_hi);
    for (int j = 0; j < 64; j++) {
        if (r_hi[j] >= 0.0f) dst_block->signs_hi[j / 8] |= (uint8_t)(1 << (j % 8));
    }
    dst_block->rnorm_hi = __float2half(sqrtf(rnorm_sq));
}

DEFINE_TQ_SPLIT_SET_ROWS_DISPATCH(5hi_3lo_had_d256, block_tqk_5hi_3lo_d256, 256)

void ggml_cuda_op_set_rows_tq_5hi_3lo_had(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32);

    GGML_TENSOR_BINARY_OP_LOCALS

    cudaStream_t stream = ctx.stream();

    int32_t layer_idx = 0;
    const char * lp = strstr(dst->name, "_l");
    if (lp) layer_idx = atoi(lp + 2);

    int32_t * chmap = ggml_cuda_get_tq_channel_map_device();
    int n_kv_heads = ggml_cuda_get_tq_chmap_n_heads();
    if (n_kv_heads < 1) n_kv_heads = (int)(ne00 / 128);

    GGML_ASSERT(ne00 % 128 == 0);
    const int64_t ne_total = (ne00 * ne01 * ne02 * ne03) / 128;
    const int num_blocks = (int)((ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE);

    const int64_t s01 = nb01 / sizeof(float);
    const int64_t s02 = nb02 / sizeof(float);
    const int64_t s03 = nb03 / sizeof(float);

    if (src1->type == GGML_TYPE_I64) {
        const int64_t s10 = nb10 / sizeof(int64_t);
        const int64_t s11 = nb11 / sizeof(int64_t);
        const int64_t s12 = nb12 / sizeof(int64_t);
        k_set_rows_tq_5hi_3lo_had<<<num_blocks, CUDA_SET_ROWS_BLOCK_SIZE, 0, stream>>>(
            (const float *)src0->data, (const int64_t *)src1->data, (block_tqk_5hi_3lo *)dst->data,
            chmap, n_kv_heads, layer_idx,
            ne00, ne01, ne02, ne03, s01, s02, s03, s10, s11, s12, nb1, nb2, nb3);
    } else {
        const int64_t s10 = nb10 / sizeof(int32_t);
        const int64_t s11 = nb11 / sizeof(int32_t);
        const int64_t s12 = nb12 / sizeof(int32_t);
        k_set_rows_tq_5hi_3lo_had<<<num_blocks, CUDA_SET_ROWS_BLOCK_SIZE, 0, stream>>>(
            (const float *)src0->data, (const int32_t *)src1->data, (block_tqk_5hi_3lo *)dst->data,
            chmap, n_kv_heads, layer_idx,
            ne00, ne01, ne02, ne03, s01, s02, s03, s10, s11, s12, nb1, nb2, nb3);
    }
}
