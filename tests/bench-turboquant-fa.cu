// Synthetic FA benchmark for TurboQuant vs q4_0 vs f16 KV cache
//
// Directly populates quantized K/V on GPU and times the FA kernel.
// No fp16 calibration phase — measures pure TQ FA throughput.
//
// Build:
//   nvcc -O2 -arch=sm_87 -std=c++17 \
//     -I../ggml/src -I../ggml/src/ggml-cuda -I../ggml/include \
//     -DGGML_COMMON_DECL_CUDA -DTURBO_QUANT_STANDALONE_TEST \
//     -DFLASH_ATTN_AVAILABLE \
//     bench-turboquant-fa.cu ../ggml/src/ggml-turbo-quant.c \
//     -o bench-turboquant-fa -lm

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#define GGML_COMMON_DECL_CUDA
#include "ggml-common.h"
#define TURBO_QUANT_STANDALONE_TEST

// Need ggml_type enum for template parameters
enum ggml_type {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_TURBO3_0_PROD = 41,
    GGML_TYPE_TURBO4_0_PROD = 42,
    GGML_TYPE_TURBO3_0_MSE  = 43,
    GGML_TYPE_TURBO4_0_MSE  = 44,
};

// Minimal stubs for fattn-common.cuh dependencies
#define GGML_UNUSED_VARS(...)
#define GGML_UNUSED(x) (void)(x)
#define NO_DEVICE_CODE
#define WARP_SIZE 32
#define CUDA_CHECK(err) do { cudaError_t e = (err); if (e != cudaSuccess) { fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)

static __device__ __forceinline__ float warp_reduce_sum(float x) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xFFFFFFFF, x, offset);
    }
    return x;
}

static __device__ __forceinline__ float warp_reduce_max(float x) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(0xFFFFFFFF, x, offset));
    }
    return x;
}

template <int D>
static __device__ __forceinline__ float warp_reduce_sum(float x) {
    return warp_reduce_sum(x);
}

static __device__ __forceinline__ float get_alibi_slope(
        float max_bias, int head, uint32_t n_head_log2, float m0, float m1) {
    (void)max_bias; (void)head; (void)n_head_log2; (void)m0; (void)m1;
    return 0.0f; // No ALiBi for benchmark
}

struct uint3_compat { unsigned int x, y, z; };

static __host__ __device__ __forceinline__ uint3 init_fastdiv_values(int n) {
    return make_uint3((unsigned)n, 0u, 0u);
}

#define FLASH_ATTN_AVAILABLE
#define FATTN_KQ_MAX_OFFSET (3.0f*0.6931f)

// Include the TQ kernels
#include "turbo-quant-cuda.cuh"

// CPU reference functions (actual symbol names from ggml-turbo-quant.c)
extern "C" {
    void quantize_row_tqv_35_ref(const float * x, block_tqv_35 * y, int64_t k);
    void quantize_row_tqk_35_ref(const float * x, block_tqk_35 * y, int64_t k);
    void quantize_row_tqv_25_ref(const float * x, block_tqv_25 * y, int64_t k);
    void quantize_row_tqk_25_ref(const float * x, block_tqk_25 * y, int64_t k);
    void tq_set_current_layer(int layer, int is_k);
    void tq_set_current_head(int head);
}
// Alias for convenience
#define quantize_row_turbo4_0_prod_ref quantize_row_tqk_35_ref
#define quantize_row_turbo4_0_mse_ref  quantize_row_tqv_35_ref
#define quantize_row_turbo3_0_prod_ref quantize_row_tqk_25_ref
#define quantize_row_turbo3_0_mse_ref  quantize_row_tqv_25_ref

// ---------------------------------------------------------------------------
// Minimal TQ FA kernel (copied from fattn-vec-tq.cuh, self-contained)
// ---------------------------------------------------------------------------

template<ggml_type type_K, ggml_type type_V>
__launch_bounds__(128, 1)
static __global__ void bench_tq_fa_kernel(
        const float * __restrict__ Q_data,    // [D] single query
        const char  * __restrict__ K_data,    // [n_tokens × block_size]
        const char  * __restrict__ V_data,    // [n_tokens × block_size]
        float       * __restrict__ dst,       // [D] output
        const int n_tokens,
        const float scale) {

    constexpr int D = 128;
    constexpr int nthreads = 128;
    const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;

    // Load query to shared memory
    __shared__ float q_f32[D];
    __shared__ float q_rot_hi[TQ_DIM_HI];
    __shared__ float q_rot_lo[TQ_DIM_LO];
    __shared__ float qjl_proj_hi[TQ_DIM_HI];
    __shared__ float qjl_proj_lo[TQ_DIM_LO];

    if (tid < D) {
        q_f32[tid] = scale * Q_data[tid];
    }
    __syncthreads();

    // Thread 0: rotate query + precompute QJL projections
    if (tid == 0) {
        const float * q_hi = q_f32;
        const float * q_lo = q_f32 + TQ_DIM_HI;

        if (tq_rotation_ready) {
            tq_rotate_hi_fwd(q_hi, q_rot_hi);
            tq_rotate_lo_fwd(q_lo, q_rot_lo);
        } else {
            for (int j = 0; j < TQ_DIM_HI; j++) q_rot_hi[j] = q_hi[j];
            for (int j = 0; j < TQ_DIM_LO; j++) q_rot_lo[j] = q_lo[j];
        }

        // QJL projections
        {
            tq_prng_state st;
            tq_prng_init(st, QJL_SEED_32);
            for (int i = 0; i < TQ_DIM_HI; i++) {
                float proj = 0.0f;
                for (int j = 0; j < TQ_DIM_HI; j++) {
                    proj += tq_prng_gaussian(st) * q_hi[j];
                }
                qjl_proj_hi[i] = proj;
            }
        }
        {
            tq_prng_state st;
            tq_prng_init(st, QJL_SEED_96);
            for (int i = 0; i < TQ_DIM_LO; i++) {
                float proj = 0.0f;
                for (int j = 0; j < TQ_DIM_LO; j++) {
                    proj += tq_prng_gaussian(st) * q_lo[j];
                }
                qjl_proj_lo[i] = proj;
            }
        }
    }
    __syncthreads();

    // K block sizes
    constexpr int k_block_size = (type_K == GGML_TYPE_TURBO4_0_PROD)
        ? (int)sizeof(block_turbo4_0_prod) : (int)sizeof(block_turbo3_0_prod);
    constexpr int v_block_size = (type_V == GGML_TYPE_TURBO4_0_MSE)
        ? (int)sizeof(block_turbo4_0_mse) : (int)sizeof(block_turbo3_0_mse);

    // Main attention loop — each thread handles different K/V tokens
    float VKQ_rot[D];
    for (int j = 0; j < D; j++) VKQ_rot[j] = 0.0f;
    float KQ_max = -FLT_MAX / 2.0f;
    float KQ_sum = 0.0f;

    __shared__ float KQ_shared[128];

    for (int k_start = 0; k_start < n_tokens; k_start += nthreads) {
        const int k = k_start + tid;
        float kq_val = -FLT_MAX;

        if (k < n_tokens) {
            if constexpr (type_K == GGML_TYPE_TURBO4_0_PROD) {
                const block_turbo4_0_prod * K_blk = (const block_turbo4_0_prod *)(K_data + k * k_block_size);
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
                kq_val = mse_hi * norm_hi + mse_lo * norm_lo;

                const float rnorm_hi = __half2float(K_blk->rnorm_hi);
                const float rnorm_lo = __half2float(K_blk->rnorm_lo);
                float qjl_hi = 0.0f;
                for (int i = 0; i < TQ_DIM_HI; i++) {
                    const float sign = ((K_blk->signs_hi[i/8] >> (i%8)) & 1) ? 1.0f : -1.0f;
                    qjl_hi += qjl_proj_hi[i] * sign;
                }
                kq_val += (1.2533141f / (float)TQ_DIM_HI) * rnorm_hi * qjl_hi;
                float qjl_lo = 0.0f;
                for (int i = 0; i < TQ_DIM_LO; i++) {
                    const float sign = ((K_blk->signs_lo[i/8] >> (i%8)) & 1) ? 1.0f : -1.0f;
                    qjl_lo += qjl_proj_lo[i] * sign;
                }
                kq_val += (1.2533141f / (float)TQ_DIM_LO) * rnorm_lo * qjl_lo;
            } else {
                const block_turbo3_0_prod * K_blk = (const block_turbo3_0_prod *)(K_data + k * k_block_size);
                const float norm_hi = __half2float(K_blk->norm_hi);
                const float norm_lo = __half2float(K_blk->norm_lo);

                float mse_hi = 0.0f;
                for (int j = 0; j < TQ_DIM_HI; j++) {
                    mse_hi += q_rot_hi[j] * tq_centroids_4_d32[tq_unpack_2bit(K_blk->qs_hi, j)];
                }
                float mse_lo = 0.0f;
                for (int j = 0; j < TQ_DIM_LO; j++) {
                    mse_lo += q_rot_lo[j] * tq_centroids_2_d96[tq_unpack_1bit(K_blk->qs_lo, j)];
                }
                kq_val = mse_hi * norm_hi + mse_lo * norm_lo;

                const float rnorm_hi = __half2float(K_blk->rnorm_hi);
                const float rnorm_lo = __half2float(K_blk->rnorm_lo);
                float qjl_hi = 0.0f;
                for (int i = 0; i < TQ_DIM_HI; i++) {
                    const float sign = ((K_blk->signs_hi[i/8] >> (i%8)) & 1) ? 1.0f : -1.0f;
                    qjl_hi += qjl_proj_hi[i] * sign;
                }
                kq_val += (1.2533141f / (float)TQ_DIM_HI) * rnorm_hi * qjl_hi;
                float qjl_lo = 0.0f;
                for (int i = 0; i < TQ_DIM_LO; i++) {
                    const float sign = ((K_blk->signs_lo[i/8] >> (i%8)) & 1) ? 1.0f : -1.0f;
                    qjl_lo += qjl_proj_lo[i] * sign;
                }
                kq_val += (1.2533141f / (float)TQ_DIM_LO) * rnorm_lo * qjl_lo;
            }
        }

        KQ_shared[tid] = kq_val;
        __syncthreads();

        // Update softmax max
        float batch_max = -FLT_MAX;
        for (int t = 0; t < nthreads && (k_start + t) < n_tokens; t++) {
            batch_max = fmaxf(batch_max, KQ_shared[t]);
        }

        if (batch_max > KQ_max) {
            float sf = expf(KQ_max - batch_max);
            for (int j = 0; j < D; j++) VKQ_rot[j] *= sf;
            KQ_sum *= sf;
            KQ_max = batch_max;
        }

        // Accumulate V in rotated space
        for (int t = 0; t < nthreads && (k_start + t) < n_tokens; t++) {
            float w = expf(KQ_shared[t] - KQ_max);
            KQ_sum += w;

            if constexpr (type_V == GGML_TYPE_TURBO4_0_MSE) {
                const block_turbo4_0_mse * V_blk =
                    (const block_turbo4_0_mse *)(V_data + (k_start + t) * v_block_size);
                const float norm = __half2float(V_blk->norm_hi);
                for (int j = 0; j < TQV_N_OUTLIER; j++) {
                    VKQ_rot[j] += w * norm * tq_centroids_16[tq_unpack_4bit(V_blk->qs_hi, j)];
                }
                for (int j = 0; j < TQV_N_REGULAR; j++) {
                    VKQ_rot[TQV_N_OUTLIER + j] += w * norm * tq_centroids_8[tq_unpack_3bit(V_blk->qs_lo, j)];
                }
            } else {
                const block_turbo3_0_mse * V_blk =
                    (const block_turbo3_0_mse *)(V_data + (k_start + t) * v_block_size);
                const float norm = __half2float(V_blk->norm_hi);
                for (int j = 0; j < TQV_N_OUTLIER; j++) {
                    VKQ_rot[j] += w * norm * tq_centroids_8[tq_unpack_3bit(V_blk->qs_hi, j)];
                }
                for (int j = 0; j < TQV_N_REGULAR; j++) {
                    VKQ_rot[TQV_N_OUTLIER + j] += w * norm * tq_centroids_4[tq_unpack_2bit(V_blk->qs_lo, j)];
                }
            }
        }
        __syncthreads();
    }

    // Reduce across threads
    __shared__ float all_KQ_max[128];
    __shared__ float all_KQ_sum[128];
    all_KQ_max[tid] = KQ_max;
    all_KQ_sum[tid] = KQ_sum;
    __syncthreads();

    __shared__ float global_KQ_max_s;
    if (tid == 0) {
        float gmax = -FLT_MAX / 2.0f;
        for (int t = 0; t < nthreads; t++) gmax = fmaxf(gmax, all_KQ_max[t]);
        global_KQ_max_s = gmax;
    }
    __syncthreads();

    float my_scale_f = expf(KQ_max - global_KQ_max_s);
    KQ_sum *= my_scale_f;
    for (int j = 0; j < D; j++) VKQ_rot[j] *= my_scale_f;

    all_KQ_sum[tid] = KQ_sum;
    __syncthreads();

    // Reduce VKQ and KQ_sum, apply rotation, write output (thread 0)
    for (int j = 0; j < D; j++) {
        KQ_shared[tid] = VKQ_rot[j];
        __syncthreads();
        for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
            if (tid < stride) KQ_shared[tid] += KQ_shared[tid + stride];
            __syncthreads();
        }
        if (tid == 0) {
            float total_sum = 0.0f;
            if (j == 0) { for (int t = 0; t < nthreads; t++) total_sum += all_KQ_sum[t]; all_KQ_sum[0] = total_sum; }
            VKQ_rot[j] = KQ_shared[0]; // store reduced value back (only thread 0)
        }
        __syncthreads();
    }

    if (tid == 0) {
        float total_sum = all_KQ_sum[0];
        float normalized[D];
        for (int j = 0; j < D; j++) normalized[j] = VKQ_rot[j] / total_sum;

        // Post-rotate V
        if (tq_rotation_ready) {
            float output[D];
            tq_rotate_v_inv(normalized, output);
            for (int j = 0; j < D; j++) dst[j] = output[j];
        } else {
            for (int j = 0; j < D; j++) dst[j] = normalized[j];
        }
    }
}

// ---------------------------------------------------------------------------
// f16 FA kernel for comparison (simple, single block)
// ---------------------------------------------------------------------------

__launch_bounds__(128, 1)
static __global__ void bench_f16_fa_kernel(
        const float * __restrict__ Q_data,
        const half  * __restrict__ K_data,   // [n_tokens × D]
        const half  * __restrict__ V_data,   // [n_tokens × D]
        float       * __restrict__ dst,
        const int n_tokens,
        const float scale) {

    constexpr int D = 128;
    constexpr int nthreads = 128;
    const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;

    __shared__ float q_f32[D];
    if (tid < D) q_f32[tid] = scale * Q_data[tid];
    __syncthreads();

    float VKQ[D];
    for (int j = 0; j < D; j++) VKQ[j] = 0.0f;
    float KQ_max = -FLT_MAX / 2.0f;
    float KQ_sum = 0.0f;

    __shared__ float KQ_shared[128];

    for (int k_start = 0; k_start < n_tokens; k_start += nthreads) {
        const int k = k_start + tid;
        float kq_val = -FLT_MAX;

        if (k < n_tokens) {
            kq_val = 0.0f;
            for (int j = 0; j < D; j++) {
                kq_val += q_f32[j] * __half2float(K_data[k * D + j]);
            }
        }

        KQ_shared[tid] = kq_val;
        __syncthreads();

        float batch_max = -FLT_MAX;
        for (int t = 0; t < nthreads && (k_start + t) < n_tokens; t++) {
            batch_max = fmaxf(batch_max, KQ_shared[t]);
        }
        if (batch_max > KQ_max) {
            float sf = expf(KQ_max - batch_max);
            for (int j = 0; j < D; j++) VKQ[j] *= sf;
            KQ_sum *= sf;
            KQ_max = batch_max;
        }

        for (int t = 0; t < nthreads && (k_start + t) < n_tokens; t++) {
            float w = expf(KQ_shared[t] - KQ_max);
            KQ_sum += w;
            for (int j = 0; j < D; j++) {
                VKQ[j] += w * __half2float(V_data[(k_start + t) * D + j]);
            }
        }
        __syncthreads();
    }

    // Reduce + output
    __shared__ float all_KQ_max[128];
    __shared__ float all_KQ_sum[128];
    all_KQ_max[tid] = KQ_max;
    all_KQ_sum[tid] = KQ_sum;
    __syncthreads();

    __shared__ float gmax_s;
    if (tid == 0) {
        float gmax = -FLT_MAX / 2.0f;
        for (int t = 0; t < nthreads; t++) gmax = fmaxf(gmax, all_KQ_max[t]);
        gmax_s = gmax;
    }
    __syncthreads();

    float sf = expf(KQ_max - gmax_s);
    KQ_sum *= sf;
    for (int j = 0; j < D; j++) VKQ[j] *= sf;
    all_KQ_sum[tid] = KQ_sum;
    __syncthreads();

    for (int j = 0; j < D; j++) {
        KQ_shared[tid] = VKQ[j];
        __syncthreads();
        for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
            if (tid < stride) KQ_shared[tid] += KQ_shared[tid + stride];
            __syncthreads();
        }
        if (tid == 0) VKQ[j] = KQ_shared[0];
        __syncthreads();
    }

    if (tid == 0) {
        float total_sum = 0.0f;
        for (int t = 0; t < nthreads; t++) total_sum += all_KQ_sum[t];
        for (int j = 0; j < D; j++) dst[j] = VKQ[j] / total_sum;
    }
}

// ---------------------------------------------------------------------------
// q4_0 FA kernel for comparison
// ---------------------------------------------------------------------------

__launch_bounds__(128, 1)
static __global__ void bench_q4_0_fa_kernel(
        const float     * __restrict__ Q_data,
        const block_q4_0 * __restrict__ K_data,  // [n_tokens × D/32 blocks]
        const block_q4_0 * __restrict__ V_data,
        float           * __restrict__ dst,
        const int n_tokens,
        const float scale) {

    constexpr int D = 128;
    constexpr int nthreads = 128;
    constexpr int QK = 32;  // q4_0 block size
    constexpr int n_blocks_per_row = D / QK; // 4
    const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;

    __shared__ float q_f32[D];
    if (tid < D) q_f32[tid] = scale * Q_data[tid];
    __syncthreads();

    float VKQ[D];
    for (int j = 0; j < D; j++) VKQ[j] = 0.0f;
    float KQ_max = -FLT_MAX / 2.0f;
    float KQ_sum = 0.0f;

    __shared__ float KQ_shared[128];

    for (int k_start = 0; k_start < n_tokens; k_start += nthreads) {
        const int k = k_start + tid;
        float kq_val = -FLT_MAX;

        if (k < n_tokens) {
            kq_val = 0.0f;
            const block_q4_0 * K_row = K_data + k * n_blocks_per_row;
            for (int b = 0; b < n_blocks_per_row; b++) {
                const float d = __half2float(K_row[b].d);
                for (int j = 0; j < QK/2; j++) {
                    const uint8_t byte = K_row[b].qs[j];
                    const float v0 = ((float)(byte & 0xF) - 8.0f) * d;
                    const float v1 = ((float)(byte >> 4)  - 8.0f) * d;
                    kq_val += q_f32[b*QK + j*2 + 0] * v0;
                    kq_val += q_f32[b*QK + j*2 + 1] * v1;
                }
            }
        }

        KQ_shared[tid] = kq_val;
        __syncthreads();

        float batch_max = -FLT_MAX;
        for (int t = 0; t < nthreads && (k_start + t) < n_tokens; t++) {
            batch_max = fmaxf(batch_max, KQ_shared[t]);
        }
        if (batch_max > KQ_max) {
            float sf = expf(KQ_max - batch_max);
            for (int j = 0; j < D; j++) VKQ[j] *= sf;
            KQ_sum *= sf;
            KQ_max = batch_max;
        }

        for (int t = 0; t < nthreads && (k_start + t) < n_tokens; t++) {
            float w = expf(KQ_shared[t] - KQ_max);
            KQ_sum += w;
            const block_q4_0 * V_row = V_data + (k_start + t) * n_blocks_per_row;
            for (int b = 0; b < n_blocks_per_row; b++) {
                const float d = __half2float(V_row[b].d);
                for (int j = 0; j < QK/2; j++) {
                    const uint8_t byte = V_row[b].qs[j];
                    VKQ[b*QK + j*2 + 0] += w * ((float)(byte & 0xF) - 8.0f) * d;
                    VKQ[b*QK + j*2 + 1] += w * ((float)(byte >> 4)  - 8.0f) * d;
                }
            }
        }
        __syncthreads();
    }

    // Reduce + output (same as f16 kernel)
    __shared__ float all_KQ_max[128];
    __shared__ float all_KQ_sum[128];
    all_KQ_max[tid] = KQ_max;
    all_KQ_sum[tid] = KQ_sum;
    __syncthreads();

    __shared__ float gmax_s;
    if (tid == 0) {
        float gmax = -FLT_MAX / 2.0f;
        for (int t = 0; t < nthreads; t++) gmax = fmaxf(gmax, all_KQ_max[t]);
        gmax_s = gmax;
    }
    __syncthreads();

    float sf = expf(KQ_max - gmax_s);
    KQ_sum *= sf;
    for (int j = 0; j < D; j++) VKQ[j] *= sf;
    all_KQ_sum[tid] = KQ_sum;
    __syncthreads();

    for (int j = 0; j < D; j++) {
        KQ_shared[tid] = VKQ[j];
        __syncthreads();
        for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
            if (tid < stride) KQ_shared[tid] += KQ_shared[tid + stride];
            __syncthreads();
        }
        if (tid == 0) VKQ[j] = KQ_shared[0];
        __syncthreads();
    }

    if (tid == 0) {
        float total_sum = 0.0f;
        for (int t = 0; t < nthreads; t++) total_sum += all_KQ_sum[t];
        for (int j = 0; j < D; j++) dst[j] = VKQ[j] / total_sum;
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static float randf(unsigned * seed) {
    *seed = *seed * 1103515245u + 12345u;
    return ((float)(*seed >> 16) / 32768.0f) - 1.0f;
}

static double time_kernel_ms(void (*launch)(void*), void * arg, int warmup, int iters) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < warmup; i++) launch(arg);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; i++) launch(arg);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return (double)ms / iters;
}

// ---------------------------------------------------------------------------
// Launch wrappers
// ---------------------------------------------------------------------------

struct bench_args {
    float * Q_d;
    char  * K_d;       // TQ K
    char  * V_d;       // TQ V
    float * dst_d;
    half  * K_f16_d;
    half  * V_f16_d;
    block_q4_0 * K_q4_d;
    block_q4_0 * V_q4_d;
    int n_tokens;
    float scale;
};

static void launch_tq35(void * a) {
    bench_args * args = (bench_args *)a;
    dim3 block(32, 4);
    bench_tq_fa_kernel<GGML_TYPE_TURBO4_0_PROD, GGML_TYPE_TURBO4_0_MSE>
        <<<1, block>>>(
        args->Q_d, args->K_d, args->V_d, args->dst_d, args->n_tokens, args->scale);
}

static void launch_tq25(void * a) {
    bench_args * args = (bench_args *)a;
    dim3 block(32, 4);
    bench_tq_fa_kernel<GGML_TYPE_TURBO3_0_PROD, GGML_TYPE_TURBO3_0_MSE>
        <<<1, block>>>(
        args->Q_d, args->K_d, args->V_d, args->dst_d, args->n_tokens, args->scale);
}

static void launch_f16(void * a) {
    bench_args * args = (bench_args *)a;
    dim3 block(32, 4);
    bench_f16_fa_kernel
        <<<1, block>>>(
        args->Q_d, args->K_f16_d, args->V_f16_d, args->dst_d, args->n_tokens, args->scale);
}

static void launch_q4_0(void * a) {
    bench_args * args = (bench_args *)a;
    dim3 block(32, 4);
    bench_q4_0_fa_kernel
        <<<1, block>>>(
        args->Q_d, args->K_q4_d, args->V_q4_d, args->dst_d, args->n_tokens, args->scale);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    int dev;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("TurboQuant FA Synthetic Benchmark on %s (cc %d.%d)\n\n", prop.name, prop.major, prop.minor);

    // Init rotations on device (CPU rotations auto-init on first quantize call)
    tq_device_init_rotations_kernel<<<1, 1>>>();
    CUDA_CHECK(cudaDeviceSynchronize());

    constexpr int D = 128;
    const float scale = 1.0f / sqrtf((float)D);

    const int seq_lens[] = {512, 1024, 2048, 4096, 8192, 16384, 32768};
    const int n_seq = sizeof(seq_lens) / sizeof(seq_lens[0]);

    printf("| %-8s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s | %-9s | %-9s | %-9s | %-9s |\n",
           "seq_len", "f16 (ms)", "q4_0 (ms)", "tq35 (ms)", "tq25 (ms)", "q4_0/f16", "tq35/f16", "f16 KB", "q4_0 KB", "tq35 KB", "tq25 KB");
    printf("|----------|------------|------------|------------|------------|------------|------------|-----------|-----------|-----------|----------|\n");

    for (int si = 0; si < n_seq; si++) {
        const int n_tokens = seq_lens[si];

        // Generate random f32 data on host
        unsigned seed = 42 + si;
        float * Q_h = (float *)malloc(D * sizeof(float));
        float * KV_h = (float *)malloc(n_tokens * D * sizeof(float));

        for (int j = 0; j < D; j++) Q_h[j] = randf(&seed);
        for (int i = 0; i < n_tokens * D; i++) KV_h[i] = randf(&seed) * 0.1f;

        // Quantize K/V on CPU
        // TQ 3.5
        block_turbo4_0_prod * K_tq35_h = (block_turbo4_0_prod *)calloc(n_tokens, sizeof(block_turbo4_0_prod));
        block_turbo4_0_mse  * V_tq35_h = (block_turbo4_0_mse  *)calloc(n_tokens, sizeof(block_turbo4_0_mse));
        tq_set_current_layer(0, 1);
        tq_set_current_head(0);
        quantize_row_turbo4_0_prod_ref(KV_h, K_tq35_h, (int64_t)n_tokens * D);
        tq_set_current_layer(0, 0);
        quantize_row_turbo4_0_mse_ref(KV_h, V_tq35_h, (int64_t)n_tokens * D);

        // TQ 2.5
        block_turbo3_0_prod * K_tq25_h = (block_turbo3_0_prod *)calloc(n_tokens, sizeof(block_turbo3_0_prod));
        block_turbo3_0_mse  * V_tq25_h = (block_turbo3_0_mse  *)calloc(n_tokens, sizeof(block_turbo3_0_mse));
        tq_set_current_layer(0, 1);
        tq_set_current_head(0);
        quantize_row_turbo3_0_prod_ref(KV_h, K_tq25_h, (int64_t)n_tokens * D);
        tq_set_current_layer(0, 0);
        quantize_row_turbo3_0_mse_ref(KV_h, V_tq25_h, (int64_t)n_tokens * D);

        // q4_0 K/V
        const int n_q4_blocks = n_tokens * (D / QK4_0);
        block_q4_0 * K_q4_h = (block_q4_0 *)calloc(n_q4_blocks, sizeof(block_q4_0));
        block_q4_0 * V_q4_h = (block_q4_0 *)calloc(n_q4_blocks, sizeof(block_q4_0));
        for (int t = 0; t < n_tokens; t++) {
            for (int b = 0; b < D / QK4_0; b++) {
                const float * src = KV_h + t * D + b * QK4_0;
                block_q4_0 * dst_k = K_q4_h + t * (D / QK4_0) + b;
                block_q4_0 * dst_v = V_q4_h + t * (D / QK4_0) + b;
                // Simple q4_0 quantize: find max, compute scale
                float amax = 0.0f;
                for (int j = 0; j < QK4_0; j++) amax = fmaxf(amax, fabsf(src[j]));
                const float d = amax / 7.0f;
                dst_k->d = __float2half(d);
                dst_v->d = __float2half(d);
                const float id = d > 0.0f ? 1.0f / d : 0.0f;
                for (int j = 0; j < QK4_0 / 2; j++) {
                    int x0 = (int)roundf(src[j*2 + 0] * id) + 8;
                    int x1 = (int)roundf(src[j*2 + 1] * id) + 8;
                    x0 = x0 < 0 ? 0 : (x0 > 15 ? 15 : x0);
                    x1 = x1 < 0 ? 0 : (x1 > 15 ? 15 : x1);
                    dst_k->qs[j] = (uint8_t)(x0 | (x1 << 4));
                    dst_v->qs[j] = dst_k->qs[j];
                }
            }
        }

        // f16 K/V
        half * K_f16_h = (half *)malloc(n_tokens * D * sizeof(half));
        half * V_f16_h = (half *)malloc(n_tokens * D * sizeof(half));
        for (int i = 0; i < n_tokens * D; i++) {
            K_f16_h[i] = __float2half(KV_h[i]);
            V_f16_h[i] = __float2half(KV_h[i]);
        }

        // Upload to device
        float * Q_d;
        char  * K_tq35_d, * V_tq35_d, * K_tq25_d, * V_tq25_d;
        half  * K_f16_d, * V_f16_d;
        block_q4_0 * K_q4_d, * V_q4_d;
        float * dst_d;

        CUDA_CHECK(cudaMalloc(&Q_d, D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&K_tq35_d, n_tokens * sizeof(block_turbo4_0_prod)));
        CUDA_CHECK(cudaMalloc(&V_tq35_d, n_tokens * sizeof(block_turbo4_0_mse)));
        CUDA_CHECK(cudaMalloc(&K_tq25_d, n_tokens * sizeof(block_turbo3_0_prod)));
        CUDA_CHECK(cudaMalloc(&V_tq25_d, n_tokens * sizeof(block_turbo3_0_mse)));
        CUDA_CHECK(cudaMalloc(&K_f16_d, n_tokens * D * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&V_f16_d, n_tokens * D * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&K_q4_d, n_q4_blocks * sizeof(block_q4_0)));
        CUDA_CHECK(cudaMalloc(&V_q4_d, n_q4_blocks * sizeof(block_q4_0)));
        CUDA_CHECK(cudaMalloc(&dst_d, D * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(Q_d, Q_h, D * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(K_tq35_d, K_tq35_h, n_tokens * sizeof(block_turbo4_0_prod), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(V_tq35_d, V_tq35_h, n_tokens * sizeof(block_turbo4_0_mse), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(K_tq25_d, K_tq25_h, n_tokens * sizeof(block_turbo3_0_prod), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(V_tq25_d, V_tq25_h, n_tokens * sizeof(block_turbo3_0_mse), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(K_f16_d, K_f16_h, n_tokens * D * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(V_f16_d, V_f16_h, n_tokens * D * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(K_q4_d, K_q4_h, n_q4_blocks * sizeof(block_q4_0), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(V_q4_d, V_q4_h, n_q4_blocks * sizeof(block_q4_0), cudaMemcpyHostToDevice));

        // Benchmark
        bench_args args_tq35 = {Q_d, K_tq35_d, V_tq35_d, dst_d, nullptr, nullptr, nullptr, nullptr, n_tokens, scale};
        bench_args args_tq25 = {Q_d, K_tq25_d, V_tq25_d, dst_d, nullptr, nullptr, nullptr, nullptr, n_tokens, scale};
        bench_args args_f16  = {Q_d, nullptr, nullptr, dst_d, K_f16_d, V_f16_d, nullptr, nullptr, n_tokens, scale};
        bench_args args_q4   = {Q_d, nullptr, nullptr, dst_d, nullptr, nullptr, K_q4_d, V_q4_d, n_tokens, scale};

        int warmup = 5;
        int iters = 50;
        if (n_tokens >= 2048) { warmup = 3; iters = 20; }
        if (n_tokens >= 8192) { warmup = 2; iters = 10; }
        if (n_tokens >= 16384) { warmup = 1; iters = 5; }

        double ms_f16  = time_kernel_ms(launch_f16,  &args_f16,  warmup, iters);
        double ms_q4   = time_kernel_ms(launch_q4_0, &args_q4,   warmup, iters);
        double ms_tq35 = time_kernel_ms(launch_tq35, &args_tq35, warmup, iters);
        double ms_tq25 = time_kernel_ms(launch_tq25, &args_tq25, warmup, iters);

        // Memory: KV cache size per head
        double kv_f16  = n_tokens * D * 2.0 * sizeof(half) / 1024.0;
        double kv_q4   = n_tokens * 2.0 * (D / QK4_0) * sizeof(block_q4_0) / 1024.0;
        double kv_tq35 = n_tokens * (sizeof(block_turbo4_0_prod) + sizeof(block_turbo4_0_mse)) / 1024.0;
        double kv_tq25 = n_tokens * (sizeof(block_turbo3_0_prod) + sizeof(block_turbo3_0_mse)) / 1024.0;

        printf("| %8d | %8.3f   | %8.3f   | %8.3f   | %8.3f   | %9.2fx  | %9.2fx  | %7.0f   | %7.0f   | %7.0f   | %7.0f  |\n",
               n_tokens, ms_f16, ms_q4, ms_tq35, ms_tq25,
               ms_q4 / ms_f16,
               ms_tq35 / ms_f16,
               kv_f16, kv_q4, kv_tq35, kv_tq25);

        // Cleanup
        cudaFree(Q_d); cudaFree(K_tq35_d); cudaFree(V_tq35_d);
        cudaFree(K_tq25_d); cudaFree(V_tq25_d);
        cudaFree(K_f16_d); cudaFree(V_f16_d);
        cudaFree(K_q4_d); cudaFree(V_q4_d); cudaFree(dst_d);
        free(Q_h); free(KV_h);
        free(K_tq35_h); free(V_tq35_h);
        free(K_tq25_h); free(V_tq25_h);
        free(K_q4_h); free(V_q4_h);
        free(K_f16_h); free(V_f16_h);
    }

    printf("\n");
    printf("Notes:\n");
    printf("  - Single query (decode mode), 1 attention head, head_dim=128\n");
    printf("  - TQ data pre-quantized on CPU with rotation, uploaded to GPU\n");
    printf("  - 'tq35 vs f16' = f16_time / tq35_time (>100%% = TQ faster)\n");
    printf("  - Memory ratio shows KV cache compression vs f16\n");
    printf("  - tq35 = 3.75+3.5 bpv = 7.25 bpv total\n");
    printf("  - tq25 = 2.75+2.5 bpv = 5.25 bpv total\n");

    return 0;
}
