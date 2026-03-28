// TurboQuant CUDA device functions — exact paper algorithm (arXiv 2504.19874)
//
// MSE types (V cache): pure centroid lookup, no QJL
// PROD types (K cache): MSE centroids + QJL correction on residual

#pragma once

// In the main ggml-cuda build, common.cuh is already included by the .cu file
// before this header. For standalone test builds, callers must include
// ggml-common.h (with GGML_COMMON_DECL_CUDA) before this header.
#ifndef TURBO_QUANT_STANDALONE_TEST
#include "common.cuh"
#endif

// ---------------------------------------------------------------------------
// Rotation matrix Π (128×128 orthogonal, generated via QR of Gaussian)
// Stored as two static device arrays: forward (Π) and inverse (Π^T).
// Initialized by tq_init_rotation() called from host at KV cache creation.
// ---------------------------------------------------------------------------

#define TQ_HEAD_DIM 128

static __device__ float tq_rotation_fwd[TQ_HEAD_DIM * TQ_HEAD_DIM];
static __device__ float tq_rotation_inv[TQ_HEAD_DIM * TQ_HEAD_DIM];
static __device__ int   tq_rotation_ready = 0;

// Host-callable function to upload rotation matrices to device
// (defined at bottom of file, needs to be in a .cu, not .cuh — called from host)

// Apply forward rotation: out = Π * in (for quantize)
static __device__ __forceinline__ void tq_rotate_forward(const float * in, float * out) {
    for (int i = 0; i < TQ_HEAD_DIM; i++) {
        float sum = 0.0f;
        for (int j = 0; j < TQ_HEAD_DIM; j++) {
            sum += tq_rotation_fwd[i * TQ_HEAD_DIM + j] * in[j];
        }
        out[i] = sum;
    }
}

// Apply inverse rotation: out = Π^T * in (for dequantize)
static __device__ __forceinline__ void tq_rotate_inverse(const float * in, float * out) {
    for (int i = 0; i < TQ_HEAD_DIM; i++) {
        float sum = 0.0f;
        for (int j = 0; j < TQ_HEAD_DIM; j++) {
            sum += tq_rotation_inv[i * TQ_HEAD_DIM + j] * in[j];
        }
        out[i] = sum;
    }
}

// ---------------------------------------------------------------------------
// Centroids: exact Lloyd-Max for Beta((d-1)/2, (d-1)/2), d=128
// ---------------------------------------------------------------------------

static __device__ const float tq_centroids_2[2] = {
    -0.0707250243f, 0.0707250243f,
};

static __device__ const float tq_centroids_4[4] = {
    -0.1330458627f, -0.0399983984f, 0.0399983984f, 0.1330458627f,
};

static __device__ const float tq_centroids_8[8] = {
    -0.1883988281f, -0.1181421705f, -0.0665887043f, -0.0216082019f,
     0.0216082019f,  0.0665887043f,  0.1181421705f,  0.1883988281f,
};

static __device__ const float tq_centroids_16[16] = {
    -0.2376827302f, -0.1808574273f, -0.1418271941f, -0.1103094608f,
    -0.0828467454f, -0.0577864193f, -0.0341609484f, -0.0113059237f,
     0.0113059237f,  0.0341609484f,  0.0577864193f,  0.0828467454f,
     0.1103094608f,  0.1418271941f,  0.1808574273f,  0.2376827302f,
};

// ---------------------------------------------------------------------------
// Bit unpacking helpers
// ---------------------------------------------------------------------------

static __device__ __forceinline__ int tq_unpack_1bit(const uint8_t * qs, int j) {
    return (qs[j / 8] >> (j % 8)) & 1;
}

static __device__ __forceinline__ int tq_unpack_2bit(const uint8_t * qs, int j) {
    return (qs[j / 4] >> ((j % 4) * 2)) & 0x3;
}

static __device__ __forceinline__ int tq_unpack_3bit(const uint8_t * qs, int j) {
    const int bp = j * 3;
    const int bi = bp >> 3;
    const int sh = bp & 7;
    return (sh <= 5)
        ? (qs[bi] >> sh) & 7
        : ((qs[bi] >> sh) | (qs[bi + 1] << (8 - sh))) & 7;
}

static __device__ __forceinline__ int tq_unpack_4bit(const uint8_t * qs, int j) {
    return (qs[j / 2] >> ((j % 2) * 4)) & 0xF;
}

// ---------------------------------------------------------------------------
// Bit packing helpers (for quantize)
// ---------------------------------------------------------------------------

static __device__ __forceinline__ void tq_pack_1bit(uint8_t * qs, int j, int v) {
    qs[j / 8] |= (uint8_t)(v << (j % 8));
}

static __device__ __forceinline__ void tq_pack_2bit(uint8_t * qs, int j, int v) {
    qs[j / 4] |= (uint8_t)(v << ((j % 4) * 2));
}

static __device__ __forceinline__ void tq_pack_3bit(uint8_t * qs, int j, int v) {
    const int bp = j * 3;
    const int bi = bp >> 3;
    const int sh = bp & 7;
    qs[bi] |= (uint8_t)((v << sh) & 0xFF);
    if (sh > 5) {
        qs[bi + 1] |= (uint8_t)(v >> (8 - sh));
    }
}

static __device__ __forceinline__ void tq_pack_4bit(uint8_t * qs, int j, int v) {
    qs[j / 2] |= (uint8_t)(v << ((j % 2) * 4));
}

// ---------------------------------------------------------------------------
// Nearest centroid search
// ---------------------------------------------------------------------------

static __device__ __forceinline__ int tq_nearest(float val, const float * c, int n) {
    int best = 0;
    float bd = fabsf(val - c[0]);
    for (int i = 1; i < n; i++) {
        const float d = fabsf(val - c[i]);
        if (d < bd) { bd = d; best = i; }
    }
    return best;
}

// ---------------------------------------------------------------------------
// Generic MSE dequantize: unpack index, look up centroid, scale by norm
// ---------------------------------------------------------------------------

static __device__ __forceinline__ float tq_dequant_element_turbo3_mse(
        const block_turbo3_0_mse * blk, int j, float norm) {
    if (j < QK_TURBO3_MSE_HI) {
        // hi channel: 3-bit index -> centroids_8
        const int idx = tq_unpack_3bit(blk->qs_hi, j);
        return norm * tq_centroids_8[idx];
    } else {
        // lo channel: 2-bit index -> centroids_4
        const int idx = tq_unpack_2bit(blk->qs_lo, j - QK_TURBO3_MSE_HI);
        return norm * tq_centroids_4[idx];
    }
}

static __device__ __forceinline__ float tq_dequant_element_turbo4_mse(
        const block_turbo4_0_mse * blk, int j, float norm) {
    if (j < QK_TURBO4_MSE_HI) {
        // hi channel: 4-bit index -> centroids_16
        const int idx = tq_unpack_4bit(blk->qs_hi, j);
        return norm * tq_centroids_16[idx];
    } else {
        // lo channel: 3-bit index -> centroids_8
        const int idx = tq_unpack_3bit(blk->qs_lo, j - QK_TURBO4_MSE_HI);
        return norm * tq_centroids_8[idx];
    }
}

// ---------------------------------------------------------------------------
// MSE dequantize for get_rows (returns 2 consecutive elements)
// ---------------------------------------------------------------------------

static __device__ __forceinline__ void dequantize_turbo3_0_mse(
        const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_turbo3_0_mse * x = (const block_turbo3_0_mse *) vx;
    const float norm = __half2float(x[ib].norm);

    if (tq_rotation_ready) {
        // Full block dequant in rotated space, then inverse rotate
        float rotated[QK_TURBO3_MSE];
        for (int j = 0; j < QK_TURBO3_MSE; j++) {
            rotated[j] = tq_dequant_element_turbo3_mse(&x[ib], j, norm);
        }
        float orig[QK_TURBO3_MSE];
        tq_rotate_inverse(rotated, orig);
        v.x = orig[iqs + 0];
        v.y = orig[iqs + 1];
    } else {
        v.x = tq_dequant_element_turbo3_mse(&x[ib], iqs + 0, norm);
        v.y = tq_dequant_element_turbo3_mse(&x[ib], iqs + 1, norm);
    }
}

static __device__ __forceinline__ void dequantize_turbo4_0_mse(
        const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_turbo4_0_mse * x = (const block_turbo4_0_mse *) vx;
    const float norm = __half2float(x[ib].norm);

    if (tq_rotation_ready) {
        float rotated[QK_TURBO4_MSE];
        for (int j = 0; j < QK_TURBO4_MSE; j++) {
            rotated[j] = tq_dequant_element_turbo4_mse(&x[ib], j, norm);
        }
        float orig[QK_TURBO4_MSE];
        tq_rotate_inverse(rotated, orig);
        v.x = orig[iqs + 0];
        v.y = orig[iqs + 1];
    } else {
        v.x = tq_dequant_element_turbo4_mse(&x[ib], iqs + 0, norm);
        v.y = tq_dequant_element_turbo4_mse(&x[ib], iqs + 1, norm);
    }
}

// ---------------------------------------------------------------------------
// MSE quantize for set_rows (one thread per 128-element block)
// ---------------------------------------------------------------------------

static __device__ void quantize_f32_turbo3_0_mse_block(
        const float * __restrict__ x, block_turbo3_0_mse * __restrict__ y) {
    // Apply forward rotation: rotated = Π * x
    float rotated[QK_TURBO3_MSE];
    if (tq_rotation_ready) {
        tq_rotate_forward(x, rotated);
    } else {
        for (int j = 0; j < QK_TURBO3_MSE; j++) rotated[j] = x[j];
    }

    // Compute L2 norm (invariant under rotation)
    float sum_sq = 0.0f;
    for (int j = 0; j < QK_TURBO3_MSE; j++) {
        sum_sq += rotated[j] * rotated[j];
    }
    const float norm = sqrtf(sum_sq);
    y->norm = __float2half(norm);

    memset(y->qs_hi, 0, sizeof(y->qs_hi));
    memset(y->qs_lo, 0, sizeof(y->qs_lo));

    if (norm == 0.0f) { return; }
    const float inv = 1.0f / norm;

    for (int j = 0; j < QK_TURBO3_MSE_HI; j++) {
        const float xn = rotated[j] * inv;
        const int idx = tq_nearest(xn, tq_centroids_8, 8);
        tq_pack_3bit(y->qs_hi, j, idx);
    }

    for (int j = 0; j < QK_TURBO3_MSE_LO; j++) {
        const float xn = rotated[QK_TURBO3_MSE_HI + j] * inv;
        const int idx = tq_nearest(xn, tq_centroids_4, 4);
        tq_pack_2bit(y->qs_lo, j, idx);
    }
}

static __device__ void quantize_f32_turbo4_0_mse_block(
        const float * __restrict__ x, block_turbo4_0_mse * __restrict__ y) {
    float rotated[QK_TURBO4_MSE];
    if (tq_rotation_ready) {
        tq_rotate_forward(x, rotated);
    } else {
        for (int j = 0; j < QK_TURBO4_MSE; j++) rotated[j] = x[j];
    }

    float sum_sq = 0.0f;
    for (int j = 0; j < QK_TURBO4_MSE; j++) {
        sum_sq += rotated[j] * rotated[j];
    }
    const float norm = sqrtf(sum_sq);
    y->norm = __float2half(norm);

    memset(y->qs_hi, 0, sizeof(y->qs_hi));
    memset(y->qs_lo, 0, sizeof(y->qs_lo));

    if (norm == 0.0f) { return; }
    const float inv = 1.0f / norm;

    for (int j = 0; j < QK_TURBO4_MSE_HI; j++) {
        const float xn = rotated[j] * inv;
        const int idx = tq_nearest(xn, tq_centroids_16, 16);
        tq_pack_4bit(y->qs_hi, j, idx);
    }

    for (int j = 0; j < QK_TURBO4_MSE_LO; j++) {
        const float xn = rotated[QK_TURBO4_MSE_HI + j] * inv;
        const int idx = tq_nearest(xn, tq_centroids_8, 8);
        tq_pack_3bit(y->qs_lo, j, idx);
    }
}

// ---------------------------------------------------------------------------
// PRNG for QJL (identical to CPU reference for bit-exact match)
// ---------------------------------------------------------------------------

#define QJL_SEED_32   0x514A4C20ULL
#define QJL_SEED_96   0x514A4C60ULL

struct tq_prng_state {
    uint64_t s;
};

static __device__ __forceinline__ void tq_prng_init(tq_prng_state & st, uint64_t seed) {
    st.s = seed;
}

static __device__ __forceinline__ float tq_prng_gaussian(tq_prng_state & st) {
    st.s = st.s * 6364136223846793005ULL + 1442695040888963407ULL;
    const float u1 = ((float)(uint32_t)(st.s >> 32) + 1.0f) / ((float)0xFFFFFFFF + 2.0f);
    st.s = st.s * 6364136223846793005ULL + 1442695040888963407ULL;
    const float u2 = ((float)(uint32_t)(st.s >> 32) + 1.0f) / ((float)0xFFFFFFFF + 2.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

// ---------------------------------------------------------------------------
// QJL inverse: reconstruct correction vector from sign bits
// For get_rows (not FA hot path). Serial per-block, O(m^2).
// ---------------------------------------------------------------------------

static __device__ void tq_qjl_inverse(
        const uint8_t * signs, float rnorm, float * corr, int m, uint64_t seed) {
    // Zero correction vector
    for (int j = 0; j < m; j++) {
        corr[j] = 0.0f;
    }

    tq_prng_state st;
    tq_prng_init(st, seed);

    for (int i = 0; i < m; i++) {
        const float z = ((signs[i / 8] >> (i % 8)) & 1) ? 1.0f : -1.0f;
        for (int j = 0; j < m; j++) {
            const float g = tq_prng_gaussian(st);
            corr[j] += g * z;
        }
    }

    const float scale = 1.2533141f / (float)m * rnorm;
    for (int j = 0; j < m; j++) {
        corr[j] *= scale;
    }
}

// ---------------------------------------------------------------------------
// QJL forward: project residual through PRNG matrix, store signs + norm
// For set_rows (not the hot path). Serial per-block, O(m^2).
// ---------------------------------------------------------------------------

static __device__ float tq_qjl_forward(
        const float * residual, uint8_t * signs, int m, uint64_t seed) {
    // Compute residual norm
    float rnorm_sq = 0.0f;
    for (int j = 0; j < m; j++) {
        rnorm_sq += residual[j] * residual[j];
    }

    memset(signs, 0, (m + 7) / 8);

    tq_prng_state st;
    tq_prng_init(st, seed);

    for (int i = 0; i < m; i++) {
        float proj = 0.0f;
        for (int j = 0; j < m; j++) {
            proj += tq_prng_gaussian(st) * residual[j];
        }
        if (proj >= 0.0f) {
            signs[i / 8] |= (uint8_t)(1 << (i % 8));
        }
    }

    return sqrtf(rnorm_sq);
}

// ---------------------------------------------------------------------------
// PROD dequantize for get_rows (K cache, with QJL correction)
// ---------------------------------------------------------------------------

static __device__ __forceinline__ void dequantize_turbo3_0_prod(
        const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_turbo3_0_prod * x = (const block_turbo3_0_prod *) vx;
    const float norm = __half2float(x[ib].norm);
    const int j0 = iqs;
    const int j1 = iqs + 1;

    // Compute QJL corrections for both elements
    float val0, val1;

    if (j1 < QK_TURBO3_PROD_HI) {
        // Both in hi partition (j0,j1 < 32)
        float corr_hi[QK_TURBO3_PROD_HI];
        tq_qjl_inverse(x[ib].signs_hi, __half2float(x[ib].rnorm_hi),
                        corr_hi, QK_TURBO3_PROD_HI, QJL_SEED_32);

        const int idx0 = tq_unpack_2bit(x[ib].qs_hi, j0);
        const int idx1 = tq_unpack_2bit(x[ib].qs_hi, j1);
        val0 = norm * (tq_centroids_4[idx0] + corr_hi[j0]);
        val1 = norm * (tq_centroids_4[idx1] + corr_hi[j1]);

    } else if (j0 >= QK_TURBO3_PROD_HI) {
        // Both in lo partition (j0,j1 >= 32)
        float corr_lo[QK_TURBO3_PROD_LO];
        tq_qjl_inverse(x[ib].signs_lo, __half2float(x[ib].rnorm_lo),
                        corr_lo, QK_TURBO3_PROD_LO, QJL_SEED_96);

        const int lo0 = j0 - QK_TURBO3_PROD_HI;
        const int lo1 = j1 - QK_TURBO3_PROD_HI;
        const int idx0 = tq_unpack_1bit(x[ib].qs_lo, lo0);
        const int idx1 = tq_unpack_1bit(x[ib].qs_lo, lo1);
        val0 = norm * (tq_centroids_2[idx0] + corr_lo[lo0]);
        val1 = norm * (tq_centroids_2[idx1] + corr_lo[lo1]);

    } else {
        // Straddling boundary: j0=31 (hi), j1=32 (lo)
        float corr_hi[QK_TURBO3_PROD_HI];
        tq_qjl_inverse(x[ib].signs_hi, __half2float(x[ib].rnorm_hi),
                        corr_hi, QK_TURBO3_PROD_HI, QJL_SEED_32);
        float corr_lo[QK_TURBO3_PROD_LO];
        tq_qjl_inverse(x[ib].signs_lo, __half2float(x[ib].rnorm_lo),
                        corr_lo, QK_TURBO3_PROD_LO, QJL_SEED_96);

        const int idx0 = tq_unpack_2bit(x[ib].qs_hi, j0);
        const int idx1 = tq_unpack_1bit(x[ib].qs_lo, 0);
        val0 = norm * (tq_centroids_4[idx0] + corr_hi[j0]);
        val1 = norm * (tq_centroids_2[idx1] + corr_lo[0]);
    }

    v.x = val0;
    v.y = val1;
}

static __device__ __forceinline__ void dequantize_turbo4_0_prod(
        const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_turbo4_0_prod * x = (const block_turbo4_0_prod *) vx;
    const float norm = __half2float(x[ib].norm);
    const int j0 = iqs;
    const int j1 = iqs + 1;

    float val0, val1;

    if (j1 < QK_TURBO4_PROD_HI) {
        // Both in hi partition
        float corr_hi[QK_TURBO4_PROD_HI];
        tq_qjl_inverse(x[ib].signs_hi, __half2float(x[ib].rnorm_hi),
                        corr_hi, QK_TURBO4_PROD_HI, QJL_SEED_32);

        const int idx0 = tq_unpack_3bit(x[ib].qs_hi, j0);
        const int idx1 = tq_unpack_3bit(x[ib].qs_hi, j1);
        val0 = norm * (tq_centroids_8[idx0] + corr_hi[j0]);
        val1 = norm * (tq_centroids_8[idx1] + corr_hi[j1]);

    } else if (j0 >= QK_TURBO4_PROD_HI) {
        // Both in lo partition
        float corr_lo[QK_TURBO4_PROD_LO];
        tq_qjl_inverse(x[ib].signs_lo, __half2float(x[ib].rnorm_lo),
                        corr_lo, QK_TURBO4_PROD_LO, QJL_SEED_96);

        const int lo0 = j0 - QK_TURBO4_PROD_HI;
        const int lo1 = j1 - QK_TURBO4_PROD_HI;
        const int idx0 = tq_unpack_2bit(x[ib].qs_lo, lo0);
        const int idx1 = tq_unpack_2bit(x[ib].qs_lo, lo1);
        val0 = norm * (tq_centroids_4[idx0] + corr_lo[lo0]);
        val1 = norm * (tq_centroids_4[idx1] + corr_lo[lo1]);

    } else {
        // Straddling boundary: j0=31 (hi), j1=32 (lo)
        float corr_hi[QK_TURBO4_PROD_HI];
        tq_qjl_inverse(x[ib].signs_hi, __half2float(x[ib].rnorm_hi),
                        corr_hi, QK_TURBO4_PROD_HI, QJL_SEED_32);
        float corr_lo[QK_TURBO4_PROD_LO];
        tq_qjl_inverse(x[ib].signs_lo, __half2float(x[ib].rnorm_lo),
                        corr_lo, QK_TURBO4_PROD_LO, QJL_SEED_96);

        const int idx0 = tq_unpack_3bit(x[ib].qs_hi, j0);
        const int idx1 = tq_unpack_2bit(x[ib].qs_lo, 0);
        val0 = norm * (tq_centroids_8[idx0] + corr_hi[j0]);
        val1 = norm * (tq_centroids_4[idx1] + corr_lo[0]);
    }

    v.x = val0;
    v.y = val1;
}

// ---------------------------------------------------------------------------
// PROD quantize for set_rows (K cache, one thread per block)
// ---------------------------------------------------------------------------

static __device__ void quantize_f32_turbo3_0_prod_block(
        const float * __restrict__ x, block_turbo3_0_prod * __restrict__ y) {
    // Compute L2 norm
    float sum_sq = 0.0f;
    for (int j = 0; j < QK_TURBO3_PROD; j++) {
        sum_sq += x[j] * x[j];
    }
    const float norm = sqrtf(sum_sq);
    y->norm = __float2half(norm);
    y->rnorm_hi = __float2half(0.0f);
    y->rnorm_lo = __float2half(0.0f);
    memset(y->qs_hi,    0, sizeof(y->qs_hi));
    memset(y->qs_lo,    0, sizeof(y->qs_lo));
    memset(y->signs_hi, 0, sizeof(y->signs_hi));
    memset(y->signs_lo, 0, sizeof(y->signs_lo));

    if (norm == 0.0f) { return; }
    const float inv = 1.0f / norm;

    // Hi channels (0..31): 2-bit MSE -> centroids_4
    float res_hi[QK_TURBO3_PROD_HI];
    for (int j = 0; j < QK_TURBO3_PROD_HI; j++) {
        const float xn = x[j] * inv;
        const int idx = tq_nearest(xn, tq_centroids_4, 4);
        tq_pack_2bit(y->qs_hi, j, idx);
        res_hi[j] = xn - tq_centroids_4[idx];
    }

    // Lo channels (32..127): 1-bit MSE -> centroids_2
    float res_lo[QK_TURBO3_PROD_LO];
    for (int j = 0; j < QK_TURBO3_PROD_LO; j++) {
        const float xn = x[QK_TURBO3_PROD_HI + j] * inv;
        const int idx = (xn >= 0.0f) ? 1 : 0;
        tq_pack_1bit(y->qs_lo, j, idx);
        res_lo[j] = xn - tq_centroids_2[idx];
    }

    // QJL forward on residuals
    y->rnorm_hi = __float2half(tq_qjl_forward(res_hi, y->signs_hi, QK_TURBO3_PROD_HI, QJL_SEED_32));
    y->rnorm_lo = __float2half(tq_qjl_forward(res_lo, y->signs_lo, QK_TURBO3_PROD_LO, QJL_SEED_96));
}

static __device__ void quantize_f32_turbo4_0_prod_block(
        const float * __restrict__ x, block_turbo4_0_prod * __restrict__ y) {
    // Compute L2 norm
    float sum_sq = 0.0f;
    for (int j = 0; j < QK_TURBO4_PROD; j++) {
        sum_sq += x[j] * x[j];
    }
    const float norm = sqrtf(sum_sq);
    y->norm = __float2half(norm);
    y->rnorm_hi = __float2half(0.0f);
    y->rnorm_lo = __float2half(0.0f);
    memset(y->qs_hi,    0, sizeof(y->qs_hi));
    memset(y->qs_lo,    0, sizeof(y->qs_lo));
    memset(y->signs_hi, 0, sizeof(y->signs_hi));
    memset(y->signs_lo, 0, sizeof(y->signs_lo));

    if (norm == 0.0f) { return; }
    const float inv = 1.0f / norm;

    // Hi channels (0..31): 3-bit MSE -> centroids_8
    float res_hi[QK_TURBO4_PROD_HI];
    for (int j = 0; j < QK_TURBO4_PROD_HI; j++) {
        const float xn = x[j] * inv;
        const int idx = tq_nearest(xn, tq_centroids_8, 8);
        tq_pack_3bit(y->qs_hi, j, idx);
        res_hi[j] = xn - tq_centroids_8[idx];
    }

    // Lo channels (32..127): 2-bit MSE -> centroids_4
    float res_lo[QK_TURBO4_PROD_LO];
    for (int j = 0; j < QK_TURBO4_PROD_LO; j++) {
        const float xn = x[QK_TURBO4_PROD_HI + j] * inv;
        const int idx = tq_nearest(xn, tq_centroids_4, 4);
        tq_pack_2bit(y->qs_lo, j, idx);
        res_lo[j] = xn - tq_centroids_4[idx];
    }

    // QJL forward on residuals
    y->rnorm_hi = __float2half(tq_qjl_forward(res_hi, y->signs_hi, QK_TURBO4_PROD_HI, QJL_SEED_32));
    y->rnorm_lo = __float2half(tq_qjl_forward(res_lo, y->signs_lo, QK_TURBO4_PROD_LO, QJL_SEED_96));
}
