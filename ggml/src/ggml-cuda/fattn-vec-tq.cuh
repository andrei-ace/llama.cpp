// TurboQuant Flash Attention — dedicated kernel for TQ K/V cache types
//
// D=128 fixed (TQ block size). Handles:
//   K: PROD types (MSE centroids + QJL correction, d=32/96 split)
//   V: MSE types (rotated-space centroid dequant, deferred Π_v^T rotation)
//
// Key optimizations:
//   - Query preprocessing: rotate subsets + precompute QJL projections (once)
//   - K dot: MSE centroid dot + sign-weighted QJL correction (per token)
//   - V accumulate: per-element centroid lookup in rotated space (cheap)
//   - Post-rotation: Π_v^T applied once to final output

#pragma once

#include "common.cuh"
#include "fattn-common.cuh"
#include "turbo-quant-cuda.cuh"

#define TQ_FA_D       128
#define TQ_FA_NTHREADS 128
#define TQ_FA_NWARPS  (TQ_FA_NTHREADS / WARP_SIZE)

// ---------------------------------------------------------------------------
// TQ-specific K dot product: MSE centroid dot + QJL correction
// Called once per K token. Uses precomputed rotated query and QJL projections.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// QJL masked sum: Σ_{bit=1} proj[i] using byte-at-a-time processing
// Full dot = 2 * masked_sum - proj_sum  (algebraic trick from paper)
// ---------------------------------------------------------------------------

static __device__ __forceinline__ float tq_qjl_masked_sum(
        const uint8_t * __restrict__ signs, const float * __restrict__ proj, int m) {
    float sum = 0.0f;
    const int n_bytes = (m + 7) / 8;
    for (int b = 0; b < n_bytes; b++) {
        uint8_t byte = signs[b];
        const int base = b * 8;
        // Unrolled 8-bit scan — compiler optimizes well
        if (byte & 0x01) sum += proj[base + 0];
        if (byte & 0x02) sum += proj[base + 1];
        if (byte & 0x04) sum += proj[base + 2];
        if (byte & 0x08) sum += proj[base + 3];
        if (byte & 0x10) sum += proj[base + 4];
        if (byte & 0x20) sum += proj[base + 5];
        if (byte & 0x40) sum += proj[base + 6];
        if (byte & 0x80) sum += proj[base + 7];
    }
    return sum;
}

// TQK 2.5 (turbo3_prod): hi=2bit/d32, lo=1bit/d96
static __device__ __forceinline__ float tq_vec_dot_k_turbo3(
        const block_turbo3_0_prod * __restrict__ K_blk,
        const float * __restrict__ q_rot_hi,
        const float * __restrict__ q_rot_lo,
        const float * __restrict__ qjl_proj_hi,
        const float * __restrict__ qjl_proj_lo,
        float qjl_proj_sum_hi_val,
        float qjl_proj_sum_lo_val
) {
    const float norm_hi = __half2float(K_blk->norm_hi);
    const float norm_lo = __half2float(K_blk->norm_lo);

    // MSE centroid dot in rotated space
    float mse_hi = 0.0f;
    for (int j = 0; j < TQ_DIM_HI; j++) {
        mse_hi += q_rot_hi[j] * tq_centroids_4_d32[tq_unpack_2bit(K_blk->qs_hi, j)];
    }

    float mse_lo = 0.0f;
    for (int j = 0; j < TQ_DIM_LO; j++) {
        mse_lo += q_rot_lo[j] * tq_centroids_2_d96[tq_unpack_1bit(K_blk->qs_lo, j)];
    }

    float dot = mse_hi * norm_hi + mse_lo * norm_lo;

    // QJL: algebraic trick — Σ proj*sign = 2*masked_sum - proj_sum
    const float rnorm_hi = __half2float(K_blk->rnorm_hi);
    const float rnorm_lo = __half2float(K_blk->rnorm_lo);

    float qjl_hi = 2.0f * tq_qjl_masked_sum(K_blk->signs_hi, qjl_proj_hi, TQ_DIM_HI) - qjl_proj_sum_hi_val;
    dot += (1.2533141f / (float)TQ_DIM_HI) * rnorm_hi * qjl_hi;

    float qjl_lo = 2.0f * tq_qjl_masked_sum(K_blk->signs_lo, qjl_proj_lo, TQ_DIM_LO) - qjl_proj_sum_lo_val;
    dot += (1.2533141f / (float)TQ_DIM_LO) * rnorm_lo * qjl_lo;

    return dot;
}

// TQK 3.5 (turbo4_prod): hi=3bit/d32, lo=2bit/d96
static __device__ __forceinline__ float tq_vec_dot_k_turbo4(
        const block_turbo4_0_prod * __restrict__ K_blk,
        const float * __restrict__ q_rot_hi,
        const float * __restrict__ q_rot_lo,
        const float * __restrict__ qjl_proj_hi,
        const float * __restrict__ qjl_proj_lo,
        float qjl_proj_sum_hi_val,
        float qjl_proj_sum_lo_val
) {
    const float norm_hi = __half2float(K_blk->norm_hi);
    const float norm_lo = __half2float(K_blk->norm_lo);

    float mse_hi = 0.0f;
    for (int j = 0; j < TQ_DIM_HI; j++) {
        mse_hi += q_rot_hi[j] * tq_centroids_8_d32[tq_unpack_3bit(K_blk->qs_hi, j)];
    }

    float mse_lo = 0.0f;
    for (int j = 0; j < TQ_DIM_LO; j++) {
        mse_lo += q_rot_lo[j] * tq_centroids_4_d96[tq_unpack_2bit(K_blk->qs_lo, j)];
    }

    float dot = mse_hi * norm_hi + mse_lo * norm_lo;

    const float rnorm_hi = __half2float(K_blk->rnorm_hi);
    const float rnorm_lo = __half2float(K_blk->rnorm_lo);

    float qjl_hi = 2.0f * tq_qjl_masked_sum(K_blk->signs_hi, qjl_proj_hi, TQ_DIM_HI) - qjl_proj_sum_hi_val;
    dot += (1.2533141f / (float)TQ_DIM_HI) * rnorm_hi * qjl_hi;

    float qjl_lo = 2.0f * tq_qjl_masked_sum(K_blk->signs_lo, qjl_proj_lo, TQ_DIM_LO) - qjl_proj_sum_lo_val;
    dot += (1.2533141f / (float)TQ_DIM_LO) * rnorm_lo * qjl_lo;

    return dot;
}

// ---------------------------------------------------------------------------
// V dequantize in ROTATED space (no inverse rotation)
// Returns centroid × norm for element j of a V block.
// The inverse rotation Π_v^T is applied once at output.
// ---------------------------------------------------------------------------

static __device__ __forceinline__ float tq_dequant_v_rotated_turbo3(
        const block_turbo3_0_mse * __restrict__ blk, int j) {
    const float norm = __half2float(blk->norm_hi);
    if (j < TQV_N_OUTLIER) {
        return norm * tq_centroids_8[tq_unpack_3bit(blk->qs_hi, j)];
    } else {
        return norm * tq_centroids_4[tq_unpack_2bit(blk->qs_lo, j - TQV_N_OUTLIER)];
    }
}

static __device__ __forceinline__ float tq_dequant_v_rotated_turbo4(
        const block_turbo4_0_mse * __restrict__ blk, int j) {
    const float norm = __half2float(blk->norm_hi);
    if (j < TQV_N_OUTLIER) {
        return norm * tq_centroids_16[tq_unpack_4bit(blk->qs_hi, j)];
    } else {
        return norm * tq_centroids_8[tq_unpack_3bit(blk->qs_lo, j - TQV_N_OUTLIER)];
    }
}

// ---------------------------------------------------------------------------
// Main TQ Flash Attention Kernel
//
// One thread handles the full 128-dim dot product (serial, but avoids
// warp sync overhead for the complex TQ algorithm).
// ncols=1 only (decode mode).
// ---------------------------------------------------------------------------

template<ggml_type type_K, ggml_type type_V, bool use_logit_softcap>
__launch_bounds__(TQ_FA_NTHREADS, 2)  // Target 2 blocks/SM → ~128 regs/thread, 2x occupancy
static __global__ void flash_attn_ext_vec_tq(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        const char * __restrict__ sinks,
        const int  * __restrict__ KV_max,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33) {
#ifdef FLASH_ATTN_AVAILABLE
    constexpr int D = TQ_FA_D;  // 128
    constexpr int ncols = 1;    // decode mode only
    constexpr int nthreads = TQ_FA_NTHREADS;
    constexpr int nwarps = TQ_FA_NWARPS;

    const int tid = WARP_SIZE * threadIdx.y + threadIdx.x;

    const int ic0 = blockIdx.x * ncols;
    const int sequence = blockIdx.z / ne02;
    const int head = blockIdx.z - sequence * ne02;
    const int gqa_ratio = ne02 / ne12;

    Q += nb03*sequence + nb02*head + nb01*ic0;
    K += nb13*sequence + nb12*(head / gqa_ratio);
    V += nb23*sequence + nb22*(head / gqa_ratio);

    const half * maskh = mask ? (const half *)(mask + nb33*(sequence % ne33) + nb31*ic0) : nullptr;
    const float slope = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);

    // -----------------------------------------------------------------------
    // Query preprocessing — fully parallel across all 128 threads
    // -----------------------------------------------------------------------

    __shared__ float q_f32[D];        // Original query
    __shared__ float q_rot_hi[TQ_DIM_HI];  // Rotated outlier query
    __shared__ float q_rot_lo[TQ_DIM_LO];  // Rotated regular query
    __shared__ float qjl_proj_hi[TQ_DIM_HI]; // QJL projection of raw hi
    __shared__ float qjl_proj_lo[TQ_DIM_LO]; // QJL projection of raw lo

    // Load query — all threads cooperate (128 threads, 128 elements)
    if (tid < D) {
        q_f32[tid] = scale * ((const float *)Q)[tid];
    }
    __syncthreads();

    // --- Parallel rotation: each thread computes one output element ---
    // q_rot_hi: 32 elements (threads 0-31), q_rot_lo: 96 elements (threads 0-95)
    if (tq_rotation_ready) {
        if (tid < TQ_DIM_HI) {
            float sum = 0.0f;
            for (int j = 0; j < TQ_DIM_HI; j++) {
                sum += tq_rot_hi_fwd[tid * TQ_DIM_HI + j] * q_f32[j];
            }
            q_rot_hi[tid] = sum;
        }
        if (tid < TQ_DIM_LO) {
            float sum = 0.0f;
            for (int j = 0; j < TQ_DIM_LO; j++) {
                sum += tq_rot_lo_fwd[tid * TQ_DIM_LO + j] * q_f32[TQ_DIM_HI + j];
            }
            q_rot_lo[tid] = sum;
        }
    } else {
        if (tid < TQ_DIM_HI) q_rot_hi[tid] = q_f32[tid];
        if (tid < TQ_DIM_LO) q_rot_lo[tid] = q_f32[TQ_DIM_HI + tid];
    }
    __syncthreads();

    // --- Parallel QJL projection using LCG skip-ahead ---
    // Each thread computes one row of the S×q dot product independently.
    // LCG skip: advance state by n steps in O(log n) using exponentiation by squaring.
    {
        // proj_hi: 32 rows, threads 0-31 each compute one row
        // proj_lo: 96 rows, threads 0-95 each compute one row (threads 96-127 idle for lo)

        // LCG skip-ahead: s_{n} = A^n * s_0 + C * (A^n - 1) / (A - 1)
        auto lcg_skip = [](uint64_t s, uint64_t n) -> uint64_t {
            const uint64_t A = 6364136223846793005ULL;
            const uint64_t C = 1442695040888963407ULL;
            uint64_t cur_m = A, cur_c = C;
            uint64_t acc_m = 1, acc_c = 0;
            while (n > 0) {
                if (n & 1) {
                    acc_m *= cur_m;
                    acc_c = acc_c * cur_m + cur_c;
                }
                cur_c = cur_c * (cur_m + 1);
                cur_m *= cur_m;
                n >>= 1;
            }
            return acc_m * s + acc_c;
        };

        // QJL hi (32×32): each gaussian uses 2 LCG steps, row i starts at step i*32*2
        if (tid < TQ_DIM_HI) {
            tq_prng_state st;
            st.s = lcg_skip(QJL_SEED_32, (uint64_t)tid * TQ_DIM_HI * 2);
            float proj = 0.0f;
            for (int j = 0; j < TQ_DIM_HI; j++) {
                proj += tq_prng_gaussian(st) * q_f32[j]; // q_hi = q_f32[0:31]
            }
            qjl_proj_hi[tid] = proj;
        }

        // QJL lo (96×96): row i starts at step i*96*2
        if (tid < TQ_DIM_LO) {
            tq_prng_state st;
            st.s = lcg_skip(QJL_SEED_96, (uint64_t)tid * TQ_DIM_LO * 2);
            float proj = 0.0f;
            for (int j = 0; j < TQ_DIM_LO; j++) {
                proj += tq_prng_gaussian(st) * q_f32[TQ_DIM_HI + j]; // q_lo = q_f32[32:127]
            }
            qjl_proj_lo[tid] = proj;
        }
    }
    __syncthreads();

    // Precompute sums for algebraic QJL trick:
    // Σ proj[i]*sign[i] = 2*Σ_{bit=1} proj[i] - Σ_all proj[i]
    // Only need masked_sum per K token; total sum is constant.
    __shared__ float qjl_proj_sum_hi;  // Σ qjl_proj_hi[i]
    __shared__ float qjl_proj_sum_lo;  // Σ qjl_proj_lo[i]
    if (tid == 0) {
        float sh = 0.0f, sl = 0.0f;
        for (int i = 0; i < TQ_DIM_HI; i++) sh += qjl_proj_hi[i];
        for (int i = 0; i < TQ_DIM_LO; i++) sl += qjl_proj_lo[i];
        qjl_proj_sum_hi = sh;
        qjl_proj_sum_lo = sl;
    }

    // Precompute q_rot × centroid tables — eliminates per-token multiply + global centroid access
    // Per K token, the MSE dot becomes pure shared memory lookup + accumulate.
    // TQK 3.5: hi=8 centroids × 32 channels = 256 floats, lo=4 centroids × 96 channels = 384 floats
    // TQK 2.5: hi=4 × 32 = 128, lo=2 × 96 = 192
    constexpr int n_cent_hi = (type_K == GGML_TYPE_TURBO4_0_PROD) ? 8 : 4;
    constexpr int n_cent_lo = (type_K == GGML_TYPE_TURBO4_0_PROD) ? 4 : 2;
    __shared__ float qc_hi[TQ_DIM_HI * n_cent_hi];  // qc_hi[j*n_cent + c] = q_rot_hi[j] * centroid[c]
    __shared__ float qc_lo[TQ_DIM_LO * n_cent_lo];

    // All threads cooperate on precomputation
    {
        const float * cent_hi = (type_K == GGML_TYPE_TURBO4_0_PROD) ? tq_centroids_8_d32 : tq_centroids_4_d32;
        const float * cent_lo = (type_K == GGML_TYPE_TURBO4_0_PROD) ? tq_centroids_4_d96 : tq_centroids_2_d96;

        // hi: 32 × n_cent_hi values, spread across 128 threads
        const int total_hi = TQ_DIM_HI * n_cent_hi;
        for (int i = tid; i < total_hi; i += nthreads) {
            const int j = i / n_cent_hi;
            const int c = i % n_cent_hi;
            qc_hi[i] = q_rot_hi[j] * cent_hi[c];
        }
        // lo: 96 × n_cent_lo values
        const int total_lo = TQ_DIM_LO * n_cent_lo;
        for (int i = tid; i < total_lo; i += nthreads) {
            const int j = i / n_cent_lo;
            const int c = i % n_cent_lo;
            qc_lo[i] = q_rot_lo[j] * cent_lo[c];
        }
    }
    __syncthreads();

    // -----------------------------------------------------------------------
    // Main attention loop — warp-cooperative K dot
    //
    // K dot: 4 threads cooperate on each K token (each handles 32 channels)
    //   → 32 K tokens per batch (128 threads / 4 per token)
    //   → 4x less work per thread, reducing register pressure
    // MSE dot uses precomputed qc_hi/qc_lo tables: just index lookup + accumulate
    // V accum: all 128 threads, each owns 1 V dimension
    // -----------------------------------------------------------------------

    constexpr int K_GROUP = 4;            // threads per K dot product
    constexpr int K_PER_BATCH = nthreads / K_GROUP;  // 32 K tokens per batch
    const int k_group = tid / K_GROUP;    // which K token this thread works on (0-31)
    const int k_lane  = tid % K_GROUP;    // which slice of the dot (0-3)
    const uint32_t k_group_mask = 0xF << (k_group % 8 * K_GROUP); // mask for 4-thread group

    const int k_VKQ_max = ne11;

    float VKQ_my = 0.0f;
    float KQ_max = -FLT_MAX / 2.0f;
    float KQ_sum = 0.0f;

    __shared__ float KQ_shared[K_PER_BATCH];

    for (int k_start = blockIdx.y * K_PER_BATCH; k_start < k_VKQ_max;
         k_start += gridDim.y * K_PER_BATCH) {

        // Phase 1: Warp-cooperative K dot product
        // Each group of 4 threads computes one K token's dot.
        // Lane 0: hi[0:7] + lo[0:23] + qjl for those ranges
        // Lane 1: hi[8:15] + lo[24:47]
        // Lane 2: hi[16:23] + lo[48:71]
        // Lane 3: hi[24:31] + lo[72:95]
        const int k_idx = k_start + k_group;
        float partial = 0.0f;

        if (k_idx < k_VKQ_max) {
            const int hi_start = k_lane * 8;          // 8 hi channels per lane
            const int hi_end   = hi_start + 8;
            const int lo_start = k_lane * 24;          // 24 lo channels per lane
            const int lo_end   = lo_start + 24;

            // Unified K dot using precomputed qc tables — no centroid global mem access
            // Just: unpack index → shared mem lookup → accumulate
            if constexpr (type_K == GGML_TYPE_TURBO4_0_PROD) {
                const block_turbo4_0_prod * K_blk = (const block_turbo4_0_prod *)(K + k_idx * nb11);
                const float norm_hi = __half2float(K_blk->norm_hi);
                const float norm_lo = __half2float(K_blk->norm_lo);

                float mse_hi = 0.0f;
                for (int j = hi_start; j < hi_end; j++) {
                    mse_hi += qc_hi[j * n_cent_hi + tq_unpack_3bit(K_blk->qs_hi, j)];
                }
                partial += mse_hi * norm_hi;

                float mse_lo = 0.0f;
                for (int j = lo_start; j < lo_end; j++) {
                    mse_lo += qc_lo[j * n_cent_lo + tq_unpack_2bit(K_blk->qs_lo, j)];
                }
                partial += mse_lo * norm_lo;

                const float rnorm_hi = __half2float(K_blk->rnorm_hi);
                float qjl_hi_partial = 0.0f;
                for (int i = hi_start; i < hi_end; i++) {
                    if ((K_blk->signs_hi[i / 8] >> (i % 8)) & 1) qjl_hi_partial += qjl_proj_hi[i];
                }
                partial += (1.2533141f / (float)TQ_DIM_HI) * rnorm_hi * 2.0f * qjl_hi_partial;
                if (k_lane == 0) partial -= (1.2533141f / (float)TQ_DIM_HI) * rnorm_hi * qjl_proj_sum_hi;

                const float rnorm_lo = __half2float(K_blk->rnorm_lo);
                float qjl_lo_partial = 0.0f;
                for (int i = lo_start; i < lo_end; i++) {
                    if ((K_blk->signs_lo[i / 8] >> (i % 8)) & 1) qjl_lo_partial += qjl_proj_lo[i];
                }
                partial += (1.2533141f / (float)TQ_DIM_LO) * rnorm_lo * 2.0f * qjl_lo_partial;
                if (k_lane == 0) partial -= (1.2533141f / (float)TQ_DIM_LO) * rnorm_lo * qjl_proj_sum_lo;
            } else {
                const block_turbo3_0_prod * K_blk = (const block_turbo3_0_prod *)(K + k_idx * nb11);
                const float norm_hi = __half2float(K_blk->norm_hi);
                const float norm_lo = __half2float(K_blk->norm_lo);

                float mse_hi = 0.0f;
                for (int j = hi_start; j < hi_end; j++) {
                    mse_hi += qc_hi[j * n_cent_hi + tq_unpack_2bit(K_blk->qs_hi, j)];
                }
                partial += mse_hi * norm_hi;

                float mse_lo = 0.0f;
                for (int j = lo_start; j < lo_end; j++) {
                    mse_lo += qc_lo[j * n_cent_lo + tq_unpack_1bit(K_blk->qs_lo, j)];
                }
                partial += mse_lo * norm_lo;

                const float rnorm_hi = __half2float(K_blk->rnorm_hi);
                float qjl_hi_partial = 0.0f;
                for (int i = hi_start; i < hi_end; i++) {
                    if ((K_blk->signs_hi[i / 8] >> (i % 8)) & 1) qjl_hi_partial += qjl_proj_hi[i];
                }
                partial += (1.2533141f / (float)TQ_DIM_HI) * rnorm_hi * 2.0f * qjl_hi_partial;
                if (k_lane == 0) partial -= (1.2533141f / (float)TQ_DIM_HI) * rnorm_hi * qjl_proj_sum_hi;

                const float rnorm_lo = __half2float(K_blk->rnorm_lo);
                float qjl_lo_partial = 0.0f;
                for (int i = lo_start; i < lo_end; i++) {
                    if ((K_blk->signs_lo[i / 8] >> (i % 8)) & 1) qjl_lo_partial += qjl_proj_lo[i];
                }
                partial += (1.2533141f / (float)TQ_DIM_LO) * rnorm_lo * 2.0f * qjl_lo_partial;
                if (k_lane == 0) partial -= (1.2533141f / (float)TQ_DIM_LO) * rnorm_lo * qjl_proj_sum_lo;
            }

            if (use_logit_softcap) {
                // softcap applied after reduce
            }
        }

        // Reduce across 4 lanes within each group
        partial += __shfl_xor_sync(0xFFFFFFFF, partial, 1);
        partial += __shfl_xor_sync(0xFFFFFFFF, partial, 2);
        // Now lane 0 of each group has the full dot product

        float kq_val = -FLT_MAX;
        if (k_idx < k_VKQ_max && k_lane == 0) {
            kq_val = partial;
            if (use_logit_softcap) {
                kq_val = logit_softcap * tanhf(kq_val);
            }
            if (maskh) {
                kq_val += slope * __half2float(maskh[k_idx]);
            }
        }

        // Store KQ scores (only lane 0 of each group writes)
        if (k_lane == 0 && k_group < K_PER_BATCH) {
            KQ_shared[k_group] = kq_val;
        }
        __syncthreads();

        // Phase 2: Update softmax max
        float batch_max = -FLT_MAX;
        for (int t = 0; t < K_PER_BATCH && (k_start + t) < k_VKQ_max; t++) {
            batch_max = fmaxf(batch_max, KQ_shared[t] + FATTN_KQ_MAX_OFFSET);
        }

        if (batch_max > KQ_max) {
            const float scale_factor = expf(KQ_max - batch_max);
            VKQ_my *= scale_factor;
            KQ_sum *= scale_factor;
            KQ_max = batch_max;
        }

        // Phase 3: V accumulation — all 128 threads, each owns 1 V dimension
        const int my_dim = tid;

        for (int t = 0; t < K_PER_BATCH && (k_start + t) < k_VKQ_max; t++) {
            const float w = expf(KQ_shared[t] - KQ_max);
            if (my_dim == 0) KQ_sum += w;

            float v_val;
            if constexpr (type_V == GGML_TYPE_TURBO3_0_MSE) {
                const block_turbo3_0_mse * V_blk =
                    (const block_turbo3_0_mse *)(V + (k_start + t) * nb21);
                v_val = tq_dequant_v_rotated_turbo3(V_blk, my_dim);
            } else if constexpr (type_V == GGML_TYPE_TURBO4_0_MSE) {
                const block_turbo4_0_mse * V_blk =
                    (const block_turbo4_0_mse *)(V + (k_start + t) * nb21);
                v_val = tq_dequant_v_rotated_turbo4(V_blk, my_dim);
            } else if constexpr (type_V == GGML_TYPE_Q4_0) {
                constexpr int QK = 32;
                const block_q4_0 * V_row = (const block_q4_0 *)(V + (k_start + t) * nb21);
                const int block_idx = my_dim / QK;
                const int in_block  = my_dim % QK;
                const float d = __half2float(V_row[block_idx].d);
                const int byte_idx = in_block / 2;
                const uint8_t byte = V_row[block_idx].qs[byte_idx];
                const int nibble = (in_block & 1) ? (byte >> 4) : (byte & 0xF);
                v_val = ((float)nibble - 8.0f) * d;
            } else if constexpr (type_V == GGML_TYPE_Q8_0) {
                constexpr int QK = 32;
                const block_q8_0 * V_row = (const block_q8_0 *)(V + (k_start + t) * nb21);
                const int block_idx = my_dim / QK;
                const int in_block  = my_dim % QK;
                const float d = __half2float(V_row[block_idx].d);
                v_val = (float)V_row[block_idx].qs[in_block] * d;
            } else if constexpr (type_V == GGML_TYPE_F16) {
                const half * V_row = (const half *)(V + (k_start + t) * nb21);
                v_val = __half2float(V_row[my_dim]);
            }
            VKQ_my += w * v_val;
        }

        __syncthreads();
    }

    // -----------------------------------------------------------------------
    // Output: normalize, apply Π_v^T post-rotation, write
    // Each thread has one dimension of VKQ in rotated space.
    // -----------------------------------------------------------------------

    // Broadcast KQ_sum from thread 0
    KQ_shared[tid] = KQ_sum;
    __syncthreads();
    const float total_KQ_sum = KQ_shared[0];

    // Normalize
    const float normalized_my = VKQ_my / total_KQ_sum;

    // Store all D normalized values in shared memory for post-rotation
    __shared__ float VKQ_normalized[D];
    VKQ_normalized[tid] = normalized_my;
    __syncthreads();

    // Apply Π_v^T post-rotation only for TQ MSE V types (accumulated in rotated space)
    // For q4_0/q8_0/f16 V, values are already in original space — no rotation needed
    constexpr bool v_needs_rotation = (type_V == GGML_TYPE_TURBO3_0_MSE || type_V == GGML_TYPE_TURBO4_0_MSE);
    float output_val;
    if constexpr (v_needs_rotation) {
        if (tq_rotation_ready) {
            float sum = 0.0f;
            for (int j = 0; j < D; j++) {
                sum += tq_rot_v_inv[tid * D + j] * VKQ_normalized[j];
            }
            output_val = sum;
        } else {
            output_val = normalized_my;
        }
    } else {
        output_val = normalized_my;
    }

    // Write to destination (all threads write their dimension in parallel)
    if (gridDim.y == 1) {
        dst[(((int64_t)sequence * int(ne01.z) + ic0) * ne02 + head) * D + tid] = output_val;
    } else {
        dst[(((int64_t)sequence * int(ne01.z) + ic0) * ne02 + head) * gridDim.y * D + blockIdx.y * D + tid] = output_val;
    }

    if (gridDim.y != 1 && tid == 0) {
        dst_meta[((sequence * int(ne01.z) + ic0) * ne02 + head) * gridDim.y + blockIdx.y] =
            make_float2(KQ_max, total_KQ_sum);
    }

#else
    GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03, nb01, nb02, nb03,
        ne10, ne11, ne12, ne13, nb11, nb12, nb13,
        nb21, nb22, nb23, ne31, ne32, ne33, nb31, nb32, nb33);
    NO_DEVICE_CODE;
#endif // FLASH_ATTN_AVAILABLE
}

// ---------------------------------------------------------------------------
// Host-side dispatch
// ---------------------------------------------------------------------------

template <ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_vec_tq_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    // Lazy init rotations for this TU
    {
        static bool initialized = false;
        if (!initialized) {
            cudaStream_t stream = ctx.stream();
            tq_device_init_rotations_kernel<<<1, 1, 0, stream>>>();
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaStreamSynchronize(stream));
            initialized = true;
        }
    }

    const ggml_tensor * KQV = dst;
    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    constexpr int D = TQ_FA_D;
    constexpr int ncols = 1;  // decode mode

    fattn_kernel_t fattn_kernel;
    if (logit_softcap == 0.0f) {
        fattn_kernel = flash_attn_ext_vec_tq<type_K, type_V, false>;
    } else {
        fattn_kernel = flash_attn_ext_vec_tq<type_K, type_V, true>;
    }

    const int nwarps = TQ_FA_NWARPS;
    constexpr size_t nbytes_shared = 0;  // Static shared memory only
    const bool need_f16_K = false;  // TQ types stay as-is
    const bool need_f16_V = false;

    launch_fattn<D, ncols, 1>(ctx, dst, fattn_kernel, nwarps, nbytes_shared, D,
                              need_f16_K, need_f16_V, false);
}

// Explicit instantiation declarations
#define DECL_FATTN_VEC_TQ_CASE(type_K, type_V)                                \
    template void ggml_cuda_flash_attn_ext_vec_tq_case                        \
    <type_K, type_V>(ggml_backend_cuda_context & ctx, ggml_tensor * dst)

// PROD K × TQ MSE V (4 combos)
extern DECL_FATTN_VEC_TQ_CASE(GGML_TYPE_TURBO3_0_PROD, GGML_TYPE_TURBO3_0_MSE);
extern DECL_FATTN_VEC_TQ_CASE(GGML_TYPE_TURBO3_0_PROD, GGML_TYPE_TURBO4_0_MSE);
extern DECL_FATTN_VEC_TQ_CASE(GGML_TYPE_TURBO4_0_PROD, GGML_TYPE_TURBO3_0_MSE);
extern DECL_FATTN_VEC_TQ_CASE(GGML_TYPE_TURBO4_0_PROD, GGML_TYPE_TURBO4_0_MSE);
// PROD K × standard V (q4_0, q8_0, f16)
extern DECL_FATTN_VEC_TQ_CASE(GGML_TYPE_TURBO3_0_PROD, GGML_TYPE_Q4_0);
extern DECL_FATTN_VEC_TQ_CASE(GGML_TYPE_TURBO4_0_PROD, GGML_TYPE_Q4_0);
extern DECL_FATTN_VEC_TQ_CASE(GGML_TYPE_TURBO3_0_PROD, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_TQ_CASE(GGML_TYPE_TURBO4_0_PROD, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_TQ_CASE(GGML_TYPE_TURBO3_0_PROD, GGML_TYPE_F16);
extern DECL_FATTN_VEC_TQ_CASE(GGML_TYPE_TURBO4_0_PROD, GGML_TYPE_F16);
