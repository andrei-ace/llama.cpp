#pragma once

// TurboQuant CUDA device code — centroids, FWHT, bit pack/unpack helpers.
// Ported from ggml-turbo-quant.c (CPU reference) and ggml-metal.metal (Metal reference).

#include "common.cuh"

// ---------------------------------------------------------------------------
// Centroid tables — exact Lloyd-Max for Beta((d-1)/2, (d-1)/2)
// ---------------------------------------------------------------------------

// d=128, 4-bit (16 centroids)
static __device__ __constant__ float tq_c16_d128[16] = {
    -0.2376271868f, -0.1807937296f, -0.1417616544f, -0.1102470655f,
    -0.0827925668f, -0.0577445357f, -0.0341340283f, -0.0112964982f,
     0.0112964982f,  0.0341340283f,  0.0577445357f,  0.0827925668f,
     0.1102470655f,  0.1417616544f,  0.1807937296f,  0.2376271868f,
};

// d=128, 3-bit (8 centroids)
static __device__ __constant__ float tq_c8_d128[8] = {
    -0.1883971860f, -0.1181397670f, -0.0665856080f, -0.0216043106f,
     0.0216043106f,  0.0665856080f,  0.1181397670f,  0.1883971860f,
};

// d=32, 4-bit (16 centroids) — for 5hi_3lo outlier subset
static __device__ __constant__ float tq_c16_d32[16] = {
    -0.4533721997f, -0.3498559145f, -0.2764914062f, -0.2161194569f,
    -0.1628573862f, -0.1138475776f, -0.0673934339f, -0.0223187462f,
     0.0223187462f,  0.0673934339f,  0.1138475776f,  0.1628573862f,
     0.2161194569f,  0.2764914062f,  0.3498559145f,  0.4533721997f,
};

// d=32, 3-bit (8 centroids) — for 5hi_3lo regular lo sub-blocks (each 32-dim)
static __device__ __constant__ float tq_c8_d32[8] = {
    -0.3662682422f, -0.2324605670f, -0.1317560968f, -0.0428515156f,
     0.0428515156f,  0.1317560968f,  0.2324605670f,  0.3662682422f,
};

// d=96, 3-bit (8 centroids) — unused currently but kept for completeness
static __device__ __constant__ float tq_c8_d96[8] = {
    -0.2168529349f, -0.1361685800f, -0.0767954958f, -0.0249236898f,
     0.0249236898f,  0.0767954958f,  0.1361685800f,  0.2168529349f,
};

// QJL scale constant: sqrt(pi/2) / dim = 1.2533141 / dim
#define QJL_SCALE_128 (1.2533141f / 128.0f)
#define QJL_SCALE_32  (1.2533141f / 32.0f)

// Device-side pointer to channel map for 5hi_3lo FA kernels.
// Set by host before launching FA kernels that use 5hi_3lo K types.
// Layout: [n_layers][n_kv_heads][128] int32_t — first 32 are outlier indices, next 96 regular.
extern __device__ const int32_t * tq_fa_channel_map_ptr;
extern __device__ int             tq_fa_chmap_n_heads;

// ---------------------------------------------------------------------------
// Bit pack/unpack helpers
// ---------------------------------------------------------------------------

// Unpack 4-bit index from packed byte array
static __device__ __forceinline__ int tq_up4(const uint8_t * q, int j) {
    return (q[j / 2] >> ((j % 2) * 4)) & 0xF;
}

// Pack 4-bit index into packed byte array
static __device__ __forceinline__ void tq_pk4(uint8_t * q, int j, int v) {
    q[j / 2] |= (uint8_t)(v << ((j % 2) * 4));
}

// Unpack 3-bit index from packed byte array
static __device__ __forceinline__ int tq_up3(const uint8_t * q, int j) {
    int bp = j * 3, bi = bp >> 3, sh = bp & 7;
    return (sh <= 5) ? (q[bi] >> sh) & 7 : ((q[bi] >> sh) | (q[bi + 1] << (8 - sh))) & 7;
}

// Pack 3-bit index into packed byte array
static __device__ __forceinline__ void tq_pk3(uint8_t * q, int j, int v) {
    int bp = j * 3, bi = bp >> 3, sh = bp & 7;
    q[bi] |= (uint8_t)((v << sh) & 0xFF);
    if (sh > 5) q[bi + 1] |= (uint8_t)(v >> (8 - sh));
}

// Unpack 1-bit sign from packed byte array (returns +1.0f or -1.0f)
static __device__ __forceinline__ float tq_sign_bit(const uint8_t * signs, int j) {
    return ((signs[j / 8] >> (j % 8)) & 1) ? 1.0f : -1.0f;
}

// ---------------------------------------------------------------------------
// Nearest centroid search (brute-force, n <= 16)
// ---------------------------------------------------------------------------

static __device__ __forceinline__ int tq_nearest(float val, const float * c, int n) {
    int best = 0;
    float bd = fabsf(val - c[0]);
    for (int i = 1; i < n; i++) {
        float d = fabsf(val - c[i]);
        if (d < bd) { bd = d; best = i; }
    }
    return best;
}

// ---------------------------------------------------------------------------
// Fast Walsh-Hadamard Transform — thread-local (one thread owns all N elements)
// Self-inverse when normalized by 1/sqrt(N).
// Used in set_rows/get_rows where one thread handles a full 128-element block.
// ---------------------------------------------------------------------------

template <int N>
static __device__ __forceinline__ void tq_fwht_local(float * x) {
    for (int step = 1; step < N; step *= 2) {
        for (int i = 0; i < N; i += 2 * step) {
            for (int j = 0; j < step; j++) {
                float a = x[i + j];
                float b = x[i + j + step];
                x[i + j]        = a + b;
                x[i + j + step] = a - b;
            }
        }
    }
    float s = rsqrtf((float)N);
    for (int i = 0; i < N; i++) x[i] *= s;
}

// ---------------------------------------------------------------------------
// Fast Walsh-Hadamard Transform — shared memory, cooperative across threads.
// Used in FA Q pre-rotation where multiple threads share work.
// Caller must ensure shmem has N floats and __syncthreads() after return.
// ---------------------------------------------------------------------------

template <int N>
static __device__ __forceinline__ void tq_fwht_shared(float * shmem, int tid, int nthreads) {
    for (int step = 1; step < N; step *= 2) {
        for (int idx = tid; idx < N / 2; idx += nthreads) {
            int i = (idx / step) * (2 * step) + (idx % step);
            float a = shmem[i];
            float b = shmem[i + step];
            shmem[i]        = a + b;
            shmem[i + step] = a - b;
        }
        __syncthreads();
    }
    float s = rsqrtf((float)N);
    for (int i = tid; i < N; i += nthreads) {
        shmem[i] *= s;
    }
    __syncthreads();
}
