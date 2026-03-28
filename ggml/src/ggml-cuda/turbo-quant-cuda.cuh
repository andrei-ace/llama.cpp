// TurboQuant CUDA device functions — exact paper algorithm (arXiv 2504.19874)
//
// MSE types (V cache): 128×128 rotation, pure centroid lookup, no QJL
// PROD types (K cache): 32/96 channel split, independent rotations,
//                        MSE centroids + QJL correction on residual
//
// Three independent rotation matrices (static per-TU, device-side init):
//   Π_hi  (32×32)  — K outlier channels
//   Π_lo  (96×96)  — K regular channels
//   Π_v   (128×128) — V cache (full vector, no split)

#pragma once

#ifndef TURBO_QUANT_STANDALONE_TEST
#include "common.cuh"
#endif

// ---------------------------------------------------------------------------
// Dimension constants
// ---------------------------------------------------------------------------

#define TQ_DIM_HI   32
#define TQ_DIM_LO   96
#define TQ_DIM_V   128

// ---------------------------------------------------------------------------
// Rotation matrices — static per-TU, initialized lazily on device
// ---------------------------------------------------------------------------

static __device__ float tq_rot_hi_fwd[TQ_DIM_HI * TQ_DIM_HI];
static __device__ float tq_rot_hi_inv[TQ_DIM_HI * TQ_DIM_HI];
static __device__ float tq_rot_lo_fwd[TQ_DIM_LO * TQ_DIM_LO];
static __device__ float tq_rot_lo_inv[TQ_DIM_LO * TQ_DIM_LO];
static __device__ float tq_rot_v_fwd[TQ_DIM_V * TQ_DIM_V];
static __device__ float tq_rot_v_inv[TQ_DIM_V * TQ_DIM_V];
static __device__ int   tq_rotation_ready = 0;

// ---------------------------------------------------------------------------
// Rotation helpers: generic matvec out = M * in
// ---------------------------------------------------------------------------

static __device__ __forceinline__ void tq_matvec(
        const float * __restrict__ M, const float * __restrict__ in,
        float * __restrict__ out, int d) {
    for (int i = 0; i < d; i++) {
        float sum = 0.0f;
        for (int j = 0; j < d; j++) {
            sum += M[i * d + j] * in[j];
        }
        out[i] = sum;
    }
}

#define tq_rotate_hi_fwd(in, out) tq_matvec(tq_rot_hi_fwd, in, out, TQ_DIM_HI)
#define tq_rotate_hi_inv(in, out) tq_matvec(tq_rot_hi_inv, in, out, TQ_DIM_HI)
#define tq_rotate_lo_fwd(in, out) tq_matvec(tq_rot_lo_fwd, in, out, TQ_DIM_LO)
#define tq_rotate_lo_inv(in, out) tq_matvec(tq_rot_lo_inv, in, out, TQ_DIM_LO)
#define tq_rotate_v_fwd(in, out)  tq_matvec(tq_rot_v_fwd,  in, out, TQ_DIM_V)
#define tq_rotate_v_inv(in, out)  tq_matvec(tq_rot_v_inv,  in, out, TQ_DIM_V)

// ---------------------------------------------------------------------------
// Centroids: exact Lloyd-Max for Beta((d-1)/2, (d-1)/2)
// Three dimension families: d=32 (K outlier), d=96 (K regular), d=128 (V)
// ---------------------------------------------------------------------------

// d=32 (K outlier subset)
static __device__ const float tq_centroids_8_d32[8] = {
    -0.3662682422f, -0.2324605670f, -0.1317560968f, -0.0428515156f,
     0.0428515156f,  0.1317560968f,  0.2324605670f,  0.3662682422f,
};
static __device__ const float tq_centroids_4_d32[4] = {
    -0.2633194113f, -0.0798019295f, 0.0798019295f, 0.2633194113f,
};
static __device__ const float tq_centroids_2_d32[2] = {
    -0.1421534638f, 0.1421534638f,
};

// d=96 (K regular subset)
static __device__ const float tq_centroids_8_d96[8] = {
    -0.2168529349f, -0.1361685800f, -0.0767954958f, -0.0249236898f,
     0.0249236898f,  0.0767954958f,  0.1361685800f,  0.2168529349f,
};
static __device__ const float tq_centroids_4_d96[4] = {
    -0.1534455138f, -0.0461670286f, 0.0461670286f, 0.1534455138f,
};
static __device__ const float tq_centroids_2_d96[2] = {
    -0.0816460916f, 0.0816460916f,
};

// d=128 (V cache, full vector)
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
// ---------------------------------------------------------------------------

static __device__ void tq_qjl_inverse(
        const uint8_t * signs, float rnorm, float * corr, int m, uint64_t seed) {
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
// ---------------------------------------------------------------------------

static __device__ float tq_qjl_forward(
        const float * residual, uint8_t * signs, int m, uint64_t seed) {
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
// Device-side rotation matrix generation (Householder QR of Gaussian)
// Runs on GPU in a single thread. Called once per TU on first use.
// Matches CPU tq_gen_orthogonal() bit-exactly (same PRNG + QR algorithm).
// ---------------------------------------------------------------------------

static __device__ void tq_device_qr_orthogonal(float * fwd, float * inv, int d, uint64_t seed) {
    // Fill A with Gaussian random (using same PRNG as CPU)
    tq_prng_state st;
    tq_prng_init(st, seed);

    // A stored in-place in fwd (temporary)
    float * A = fwd;  // reuse fwd as scratch, we'll overwrite it with Q
    for (int i = 0; i < d * d; i++) {
        A[i] = tq_prng_gaussian(st);
    }

    // We need separate storage for Q. Use inv as scratch for Q during QR.
    float * Q = inv;
    for (int i = 0; i < d; i++)
        for (int j = 0; j < d; j++)
            Q[i * d + j] = (i == j) ? 1.0f : 0.0f;

    // Householder QR: A → R (upper tri), Q accumulated
    for (int k = 0; k < d; k++) {
        float norm_sq = 0.0f;
        for (int i = k; i < d; i++) {
            norm_sq += A[i * d + k] * A[i * d + k];
        }
        float norm = sqrtf(norm_sq);
        if (norm < 1e-12f) continue;

        float sign = (A[k * d + k] >= 0.0f) ? 1.0f : -1.0f;
        float alpha = -sign * norm;

        // Householder vector v (stored in-place in A column below diagonal)
        // v = A[k:d, k] - alpha * e_k
        // But we need separate v storage — use the first column of scratch
        // Actually, compute v_norm_sq and apply reflector directly

        // v[i] = A[i,k] for i >= k, v[k] -= alpha
        float v_k = A[k * d + k] - alpha;
        float v_norm_sq = v_k * v_k;
        for (int i = k + 1; i < d; i++) {
            v_norm_sq += A[i * d + k] * A[i * d + k];
        }
        if (v_norm_sq < 1e-24f) continue;
        float tau = 2.0f / v_norm_sq;

        // Apply H = I - tau*v*v^T to A from left: A = H*A (columns k..d-1)
        for (int j = k; j < d; j++) {
            float dot = v_k * A[k * d + j];
            for (int i = k + 1; i < d; i++) {
                dot += A[i * d + k] * A[i * d + j];
            }
            A[k * d + j] -= tau * v_k * dot;
            for (int i = k + 1; i < d; i++) {
                A[i * d + j] -= tau * A[i * d + k] * dot;
            }
        }

        // Apply H to Q from right: Q = Q*H (all rows)
        for (int i = 0; i < d; i++) {
            float dot = Q[i * d + k] * v_k;
            for (int j2 = k + 1; j2 < d; j2++) {
                dot += Q[i * d + j2] * A[j2 * d + k];
            }
            Q[i * d + k] -= tau * dot * v_k;
            for (int j2 = k + 1; j2 < d; j2++) {
                Q[i * d + j2] -= tau * dot * A[j2 * d + k];
            }
        }

        // Zero out the Householder vector column for next iteration
        // (A[k,k] is now alpha, A[i>k,k] was the v vector — leave as is,
        //  but we read A[i,k] as v[i] above, so we need to preserve it
        //  until after the reflector is applied. It's already consumed.)
    }

    // Q is in inv. Copy to fwd.
    for (int i = 0; i < d * d; i++) {
        fwd[i] = Q[i];
    }

    // inv = Q^T
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            inv[i * d + j] = fwd[j * d + i];
        }
    }
}

static __global__ void tq_device_init_rotations_kernel() {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Seeds matching CPU: ggml-turbo-quant.c tq_init_rotations()
    tq_device_qr_orthogonal(tq_rot_hi_fwd, tq_rot_hi_inv, TQ_DIM_HI, 0x5475524230484932ULL);
    tq_device_qr_orthogonal(tq_rot_lo_fwd, tq_rot_lo_inv, TQ_DIM_LO, 0x54755242304C4F36ULL);
    tq_device_qr_orthogonal(tq_rot_v_fwd,  tq_rot_v_inv,  TQ_DIM_V,  0x5475524230564131ULL);

    __threadfence();  // Ensure all writes visible before setting ready flag
    tq_rotation_ready = 1;
}

// ===================================================================
// MSE types (V cache) — 128×128 rotation, d=128 centroids, no QJL
// ===================================================================

static __device__ __forceinline__ float tq_dequant_element_turbo3_mse(
        const block_turbo3_0_mse * blk, int j, float norm) {
    if (j < QK_TURBO3_MSE_HI) {
        const int idx = tq_unpack_3bit(blk->qs_hi, j);
        return norm * tq_centroids_8[idx];
    } else {
        const int idx = tq_unpack_2bit(blk->qs_lo, j - QK_TURBO3_MSE_HI);
        return norm * tq_centroids_4[idx];
    }
}

static __device__ __forceinline__ float tq_dequant_element_turbo4_mse(
        const block_turbo4_0_mse * blk, int j, float norm) {
    if (j < QK_TURBO4_MSE_HI) {
        const int idx = tq_unpack_4bit(blk->qs_hi, j);
        return norm * tq_centroids_16[idx];
    } else {
        const int idx = tq_unpack_3bit(blk->qs_lo, j - QK_TURBO4_MSE_HI);
        return norm * tq_centroids_8[idx];
    }
}

// MSE dequantize for get_rows: dequant in rotated space, inverse rotate Π_v^T

static __device__ __forceinline__ void dequantize_turbo3_0_mse(
        const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_turbo3_0_mse * x = (const block_turbo3_0_mse *) vx;
    const float norm = __half2float(x[ib].norm_hi);

    if (tq_rotation_ready) {
        float rotated[TQ_DIM_V];
        for (int j = 0; j < TQ_DIM_V; j++) {
            rotated[j] = tq_dequant_element_turbo3_mse(&x[ib], j, norm);
        }
        float orig[TQ_DIM_V];
        tq_rotate_v_inv(rotated, orig);
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
    const float norm = __half2float(x[ib].norm_hi);

    if (tq_rotation_ready) {
        float rotated[TQ_DIM_V];
        for (int j = 0; j < TQ_DIM_V; j++) {
            rotated[j] = tq_dequant_element_turbo4_mse(&x[ib], j, norm);
        }
        float orig[TQ_DIM_V];
        tq_rotate_v_inv(rotated, orig);
        v.x = orig[iqs + 0];
        v.y = orig[iqs + 1];
    } else {
        v.x = tq_dequant_element_turbo4_mse(&x[ib], iqs + 0, norm);
        v.y = tq_dequant_element_turbo4_mse(&x[ib], iqs + 1, norm);
    }
}

// MSE quantize for set_rows: rotate Π_v, normalize, MSE quantize

static __device__ void quantize_f32_turbo3_0_mse_block(
        const float * __restrict__ x, block_turbo3_0_mse * __restrict__ y) {
    float rotated[TQ_DIM_V];
    if (tq_rotation_ready) {
        tq_rotate_v_fwd(x, rotated);
    } else {
        for (int j = 0; j < TQ_DIM_V; j++) rotated[j] = x[j];
    }

    float sum_sq = 0.0f;
    for (int j = 0; j < TQ_DIM_V; j++) {
        sum_sq += rotated[j] * rotated[j];
    }
    const float norm = sqrtf(sum_sq);
    y->norm_hi = __float2half(norm);
    y->norm_lo = __float2half(0.0f);

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
    float rotated[TQ_DIM_V];
    if (tq_rotation_ready) {
        tq_rotate_v_fwd(x, rotated);
    } else {
        for (int j = 0; j < TQ_DIM_V; j++) rotated[j] = x[j];
    }

    float sum_sq = 0.0f;
    for (int j = 0; j < TQ_DIM_V; j++) {
        sum_sq += rotated[j] * rotated[j];
    }
    const float norm = sqrtf(sum_sq);
    y->norm_hi = __float2half(norm);
    y->norm_lo = __float2half(0.0f);

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

// ===================================================================
// PROD types (K cache) — 32/96 split, independent rotations, QJL
// Identity channel permutation (channels 0-31 = outlier, 32-127 = regular)
// ===================================================================

// PROD dequantize for get_rows: full block with rotation + QJL

static __device__ __forceinline__ void dequantize_turbo3_0_prod(
        const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_turbo3_0_prod * x = (const block_turbo3_0_prod *) vx;
    const float norm_hi = __half2float(x[ib].norm_hi);
    const float norm_lo = __half2float(x[ib].norm_lo);

    // MSE centroids in rotated space
    float yhi_rot[TQ_DIM_HI], ylo_rot[TQ_DIM_LO];
    for (int j = 0; j < TQ_DIM_HI; j++) {
        yhi_rot[j] = tq_centroids_4_d32[tq_unpack_2bit(x[ib].qs_hi, j)];
    }
    for (int j = 0; j < TQ_DIM_LO; j++) {
        ylo_rot[j] = tq_centroids_2_d96[tq_unpack_1bit(x[ib].qs_lo, j)];
    }

    // Inverse rotate and scale
    float hi_orig[TQ_DIM_HI], lo_orig[TQ_DIM_LO];
    if (tq_rotation_ready) {
        tq_rotate_hi_inv(yhi_rot, hi_orig);
        tq_rotate_lo_inv(ylo_rot, lo_orig);
    } else {
        for (int j = 0; j < TQ_DIM_HI; j++) hi_orig[j] = yhi_rot[j];
        for (int j = 0; j < TQ_DIM_LO; j++) lo_orig[j] = ylo_rot[j];
    }
    for (int j = 0; j < TQ_DIM_HI; j++) hi_orig[j] *= norm_hi;
    for (int j = 0; j < TQ_DIM_LO; j++) lo_orig[j] *= norm_lo;

    // QJL correction (in original space)
    float corr_hi[TQ_DIM_HI], corr_lo[TQ_DIM_LO];
    tq_qjl_inverse(x[ib].signs_hi, __half2float(x[ib].rnorm_hi),
                    corr_hi, TQ_DIM_HI, QJL_SEED_32);
    tq_qjl_inverse(x[ib].signs_lo, __half2float(x[ib].rnorm_lo),
                    corr_lo, TQ_DIM_LO, QJL_SEED_96);

    for (int j = 0; j < TQ_DIM_HI; j++) hi_orig[j] += corr_hi[j];
    for (int j = 0; j < TQ_DIM_LO; j++) lo_orig[j] += corr_lo[j];

    // Identity permutation: output[0..31] = hi, output[32..127] = lo
    const int j0 = iqs, j1 = iqs + 1;
    v.x = (j0 < TQ_DIM_HI) ? hi_orig[j0] : lo_orig[j0 - TQ_DIM_HI];
    v.y = (j1 < TQ_DIM_HI) ? hi_orig[j1] : lo_orig[j1 - TQ_DIM_HI];
}

static __device__ __forceinline__ void dequantize_turbo4_0_prod(
        const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_turbo4_0_prod * x = (const block_turbo4_0_prod *) vx;
    const float norm_hi = __half2float(x[ib].norm_hi);
    const float norm_lo = __half2float(x[ib].norm_lo);

    float yhi_rot[TQ_DIM_HI], ylo_rot[TQ_DIM_LO];
    for (int j = 0; j < TQ_DIM_HI; j++) {
        yhi_rot[j] = tq_centroids_8_d32[tq_unpack_3bit(x[ib].qs_hi, j)];
    }
    for (int j = 0; j < TQ_DIM_LO; j++) {
        ylo_rot[j] = tq_centroids_4_d96[tq_unpack_2bit(x[ib].qs_lo, j)];
    }

    float hi_orig[TQ_DIM_HI], lo_orig[TQ_DIM_LO];
    if (tq_rotation_ready) {
        tq_rotate_hi_inv(yhi_rot, hi_orig);
        tq_rotate_lo_inv(ylo_rot, lo_orig);
    } else {
        for (int j = 0; j < TQ_DIM_HI; j++) hi_orig[j] = yhi_rot[j];
        for (int j = 0; j < TQ_DIM_LO; j++) lo_orig[j] = ylo_rot[j];
    }
    for (int j = 0; j < TQ_DIM_HI; j++) hi_orig[j] *= norm_hi;
    for (int j = 0; j < TQ_DIM_LO; j++) lo_orig[j] *= norm_lo;

    float corr_hi[TQ_DIM_HI], corr_lo[TQ_DIM_LO];
    tq_qjl_inverse(x[ib].signs_hi, __half2float(x[ib].rnorm_hi),
                    corr_hi, TQ_DIM_HI, QJL_SEED_32);
    tq_qjl_inverse(x[ib].signs_lo, __half2float(x[ib].rnorm_lo),
                    corr_lo, TQ_DIM_LO, QJL_SEED_96);

    for (int j = 0; j < TQ_DIM_HI; j++) hi_orig[j] += corr_hi[j];
    for (int j = 0; j < TQ_DIM_LO; j++) lo_orig[j] += corr_lo[j];

    const int j0 = iqs, j1 = iqs + 1;
    v.x = (j0 < TQ_DIM_HI) ? hi_orig[j0] : lo_orig[j0 - TQ_DIM_HI];
    v.y = (j1 < TQ_DIM_HI) ? hi_orig[j1] : lo_orig[j1 - TQ_DIM_HI];
}

// PROD quantize for set_rows: split → rotate → MSE → residual in orig → QJL

static __device__ void quantize_f32_turbo3_0_prod_block(
        const float * __restrict__ x, block_turbo3_0_prod * __restrict__ y) {
    const float * hi_raw = x;
    const float * lo_raw = x + TQ_DIM_HI;

    // Rotate each subset
    float hi_rot[TQ_DIM_HI], lo_rot[TQ_DIM_LO];
    if (tq_rotation_ready) {
        tq_rotate_hi_fwd(hi_raw, hi_rot);
        tq_rotate_lo_fwd(lo_raw, lo_rot);
    } else {
        for (int j = 0; j < TQ_DIM_HI; j++) hi_rot[j] = hi_raw[j];
        for (int j = 0; j < TQ_DIM_LO; j++) lo_rot[j] = lo_raw[j];
    }

    // Per-subset norms
    float sum_hi = 0.0f, sum_lo = 0.0f;
    for (int j = 0; j < TQ_DIM_HI; j++) sum_hi += hi_rot[j] * hi_rot[j];
    for (int j = 0; j < TQ_DIM_LO; j++) sum_lo += lo_rot[j] * lo_rot[j];
    const float norm_hi = sqrtf(sum_hi);
    const float norm_lo = sqrtf(sum_lo);

    y->norm_hi = __float2half(norm_hi);
    y->norm_lo = __float2half(norm_lo);
    y->rnorm_hi = __float2half(0.0f);
    y->rnorm_lo = __float2half(0.0f);
    memset(y->qs_hi,    0, sizeof(y->qs_hi));
    memset(y->qs_lo,    0, sizeof(y->qs_lo));
    memset(y->signs_hi, 0, sizeof(y->signs_hi));
    memset(y->signs_lo, 0, sizeof(y->signs_lo));

    const float inv_hi = (norm_hi > 1e-12f) ? 1.0f / norm_hi : 0.0f;
    const float inv_lo = (norm_lo > 1e-12f) ? 1.0f / norm_lo : 0.0f;

    // MSE quantize with d-specific centroids
    for (int j = 0; j < TQ_DIM_HI; j++) {
        const float xn = hi_rot[j] * inv_hi;
        const int idx = tq_nearest(xn, tq_centroids_4_d32, 4);
        tq_pack_2bit(y->qs_hi, j, idx);
    }
    for (int j = 0; j < TQ_DIM_LO; j++) {
        const float xn = lo_rot[j] * inv_lo;
        const int idx = (xn >= 0.0f) ? 1 : 0;
        tq_pack_1bit(y->qs_lo, j, idx);
    }

    // Residual in original space: r = raw - norm * Π^T × centroids[idx]
    float yhi_rot[TQ_DIM_HI], ylo_rot[TQ_DIM_LO];
    for (int j = 0; j < TQ_DIM_HI; j++) {
        yhi_rot[j] = tq_centroids_4_d32[tq_unpack_2bit(y->qs_hi, j)];
    }
    for (int j = 0; j < TQ_DIM_LO; j++) {
        ylo_rot[j] = tq_centroids_2_d96[tq_unpack_1bit(y->qs_lo, j)];
    }

    float hi_rec[TQ_DIM_HI], lo_rec[TQ_DIM_LO];
    if (tq_rotation_ready) {
        tq_rotate_hi_inv(yhi_rot, hi_rec);
        tq_rotate_lo_inv(ylo_rot, lo_rec);
    } else {
        for (int j = 0; j < TQ_DIM_HI; j++) hi_rec[j] = yhi_rot[j];
        for (int j = 0; j < TQ_DIM_LO; j++) lo_rec[j] = ylo_rot[j];
    }

    float res_hi[TQ_DIM_HI], res_lo[TQ_DIM_LO];
    for (int j = 0; j < TQ_DIM_HI; j++) {
        res_hi[j] = hi_raw[j] - norm_hi * hi_rec[j];
    }
    for (int j = 0; j < TQ_DIM_LO; j++) {
        res_lo[j] = lo_raw[j] - norm_lo * lo_rec[j];
    }

    y->rnorm_hi = __float2half(tq_qjl_forward(res_hi, y->signs_hi, TQ_DIM_HI, QJL_SEED_32));
    y->rnorm_lo = __float2half(tq_qjl_forward(res_lo, y->signs_lo, TQ_DIM_LO, QJL_SEED_96));
}

static __device__ void quantize_f32_turbo4_0_prod_block(
        const float * __restrict__ x, block_turbo4_0_prod * __restrict__ y) {
    const float * hi_raw = x;
    const float * lo_raw = x + TQ_DIM_HI;

    float hi_rot[TQ_DIM_HI], lo_rot[TQ_DIM_LO];
    if (tq_rotation_ready) {
        tq_rotate_hi_fwd(hi_raw, hi_rot);
        tq_rotate_lo_fwd(lo_raw, lo_rot);
    } else {
        for (int j = 0; j < TQ_DIM_HI; j++) hi_rot[j] = hi_raw[j];
        for (int j = 0; j < TQ_DIM_LO; j++) lo_rot[j] = lo_raw[j];
    }

    float sum_hi = 0.0f, sum_lo = 0.0f;
    for (int j = 0; j < TQ_DIM_HI; j++) sum_hi += hi_rot[j] * hi_rot[j];
    for (int j = 0; j < TQ_DIM_LO; j++) sum_lo += lo_rot[j] * lo_rot[j];
    const float norm_hi = sqrtf(sum_hi);
    const float norm_lo = sqrtf(sum_lo);

    y->norm_hi = __float2half(norm_hi);
    y->norm_lo = __float2half(norm_lo);
    y->rnorm_hi = __float2half(0.0f);
    y->rnorm_lo = __float2half(0.0f);
    memset(y->qs_hi,    0, sizeof(y->qs_hi));
    memset(y->qs_lo,    0, sizeof(y->qs_lo));
    memset(y->signs_hi, 0, sizeof(y->signs_hi));
    memset(y->signs_lo, 0, sizeof(y->signs_lo));

    const float inv_hi = (norm_hi > 1e-12f) ? 1.0f / norm_hi : 0.0f;
    const float inv_lo = (norm_lo > 1e-12f) ? 1.0f / norm_lo : 0.0f;

    for (int j = 0; j < TQ_DIM_HI; j++) {
        const float xn = hi_rot[j] * inv_hi;
        const int idx = tq_nearest(xn, tq_centroids_8_d32, 8);
        tq_pack_3bit(y->qs_hi, j, idx);
    }
    for (int j = 0; j < TQ_DIM_LO; j++) {
        const float xn = lo_rot[j] * inv_lo;
        const int idx = tq_nearest(xn, tq_centroids_4_d96, 4);
        tq_pack_2bit(y->qs_lo, j, idx);
    }

    float yhi_rot[TQ_DIM_HI], ylo_rot[TQ_DIM_LO];
    for (int j = 0; j < TQ_DIM_HI; j++) {
        yhi_rot[j] = tq_centroids_8_d32[tq_unpack_3bit(y->qs_hi, j)];
    }
    for (int j = 0; j < TQ_DIM_LO; j++) {
        ylo_rot[j] = tq_centroids_4_d96[tq_unpack_2bit(y->qs_lo, j)];
    }

    float hi_rec[TQ_DIM_HI], lo_rec[TQ_DIM_LO];
    if (tq_rotation_ready) {
        tq_rotate_hi_inv(yhi_rot, hi_rec);
        tq_rotate_lo_inv(ylo_rot, lo_rec);
    } else {
        for (int j = 0; j < TQ_DIM_HI; j++) hi_rec[j] = yhi_rot[j];
        for (int j = 0; j < TQ_DIM_LO; j++) lo_rec[j] = ylo_rot[j];
    }

    float res_hi[TQ_DIM_HI], res_lo[TQ_DIM_LO];
    for (int j = 0; j < TQ_DIM_HI; j++) {
        res_hi[j] = hi_raw[j] - norm_hi * hi_rec[j];
    }
    for (int j = 0; j < TQ_DIM_LO; j++) {
        res_lo[j] = lo_raw[j] - norm_lo * lo_rec[j];
    }

    y->rnorm_hi = __float2half(tq_qjl_forward(res_hi, y->signs_hi, TQ_DIM_HI, QJL_SEED_32));
    y->rnorm_lo = __float2half(tq_qjl_forward(res_lo, y->signs_lo, TQ_DIM_LO, QJL_SEED_96));
}
