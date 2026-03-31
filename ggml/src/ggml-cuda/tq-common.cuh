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

// d=32, 5-bit (32 centroids) — for 6hi_3lo outlier subset
static __device__ __constant__ float tq_c32_d32[32] = {
    -0.5264998987f, -0.4434050674f, -0.3863517231f, -0.3408589649f,
    -0.3020293074f, -0.2675540961f, -0.2361407504f, -0.2069821496f,
    -0.1795336985f, -0.1534053336f, -0.1283036018f, -0.1039980715f,
    -0.0803005660f, -0.0570515734f, -0.0341108450f, -0.0113504874f,
     0.0113504874f,  0.0341108450f,  0.0570515734f,  0.0803005660f,
     0.1039980715f,  0.1283036018f,  0.1534053336f,  0.1795336985f,
     0.2069821496f,  0.2361407504f,  0.2675540961f,  0.3020293074f,
     0.3408589649f,  0.3863517231f,  0.4434050674f,  0.5264998987f,
};

// d=32, 2-bit (4 centroids) — for 2hi_1lo outlier subset
static __device__ __constant__ float tq_c4_d32[4] = {
    -0.2633194113f, -0.0798019295f, 0.0798019295f, 0.2633194113f,
};

// d=96, 3-bit (8 centroids)
static __device__ __constant__ float tq_c8_d96[8] = {
    -0.2168529349f, -0.1361685800f, -0.0767954958f, -0.0249236898f,
     0.0249236898f,  0.0767954958f,  0.1361685800f,  0.2168529349f,
};

// d=96, 2-bit (4 centroids) — for 3hi_2lo regular subset
static __device__ __constant__ float tq_c4_d96[4] = {
    -0.1534455138f, -0.0461670286f, 0.0461670286f, 0.1534455138f,
};

// d=96, 1-bit (2 centroids) — for 2hi_1lo regular subset
static __device__ __constant__ float tq_c2_d96[2] = {
    -0.0816460916f, 0.0816460916f,
};

// d=256, 4-bit (16 centroids)
static __device__ __constant__ float tq_c16_d256[16] = {
    -0.1694104365f, -0.1285881998f, -0.1006980067f, -0.0782493129f,
    -0.0587321071f, -0.0409491965f, -0.0242008774f, -0.0080083741f,
     0.0080083741f,  0.0242008774f,  0.0409491965f,  0.0587321071f,
     0.0782493129f,  0.1006980067f,  0.1285881998f,  0.1694104365f,
};

// d=256, 3-bit (8 centroids)
static __device__ __constant__ float tq_c8_d256[8] = {
    -0.1338542901f, -0.0837654569f, -0.0471667103f, -0.0152974877f,
     0.0152974877f,  0.0471667103f,  0.0837654569f,  0.1338542901f,
};

// d=64, 4-bit (16 centroids) — for 5hi_3lo_d256 outlier subset
static __device__ __constant__ float tq_c16_d64[16] = {
    -0.3389919281f, -0.2586479166f, -0.2033218447f, -0.1583036541f,
    -0.1191039932f, -0.0831167296f, -0.0491520456f, -0.0162627764f,
     0.0162627764f,  0.0491520456f,  0.0831167296f,  0.1191039932f,
     0.1583036541f,  0.2033218447f,  0.2586479166f,  0.3389919281f,
};

// d=192, 3-bit (8 centroids) — for 5hi_3lo_d256 regular subset
static __device__ __constant__ float tq_c8_d192[8] = {
    -0.1543156657f, -0.0966361419f, -0.0544312518f, -0.0176559645f,
     0.0176559645f,  0.0544312518f,  0.0966361419f,  0.1543156657f,
};

// QJL scale constant: sqrt(pi/2) / dim = 1.2533141 / dim
#define QJL_SCALE_128 (1.2533141f / 128.0f)
#define QJL_SCALE_256 (1.2533141f / 256.0f)
#define QJL_SCALE_32  (1.2533141f / 32.0f)
#define QJL_SCALE_64  (1.2533141f / 64.0f)
#define QJL_SCALE_96  (1.2533141f / 96.0f)

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

// Unpack 5-bit index from packed byte array
static __device__ __forceinline__ int tq_up5(const uint8_t * q, int j) {
    int bp = j * 5, bi = bp >> 3, sh = bp & 7;
    int v = q[bi] >> sh;
    if (sh > 3) v |= q[bi + 1] << (8 - sh);
    return v & 0x1F;
}

// Pack 5-bit index into packed byte array
static __device__ __forceinline__ void tq_pk5(uint8_t * q, int j, int v) {
    int bp = j * 5, bi = bp >> 3, sh = bp & 7;
    q[bi] |= (uint8_t)(((v & 0x1F) << sh) & 0xFF);
    if (sh > 3) q[bi + 1] |= (uint8_t)((v & 0x1F) >> (8 - sh));
}

// Unpack 2-bit index from packed byte array
static __device__ __forceinline__ int tq_up2(const uint8_t * q, int j) {
    return (q[j / 4] >> ((j % 4) * 2)) & 3;
}

// Pack 2-bit index into packed byte array
static __device__ __forceinline__ void tq_pk2(uint8_t * q, int j, int v) {
    q[j / 4] |= (uint8_t)(v << ((j % 4) * 2));
}

// Unpack 1-bit index from packed byte array (returns 0 or 1)
static __device__ __forceinline__ int tq_up1(const uint8_t * q, int j) {
    return (q[j / 8] >> (j % 8)) & 1;
}

// Pack 1-bit index into packed byte array
static __device__ __forceinline__ void tq_pk1(uint8_t * q, int j, int v) {
    q[j / 8] |= (uint8_t)(v << (j % 8));
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
// Fast Walsh-Hadamard Transform — warp shuffle, zero barriers.
// Each lane in the warp holds one element. N must be <= WARP_SIZE (32).
// Returns the transformed value for this lane. No shared memory needed.
// ---------------------------------------------------------------------------

template <int N>
static __device__ __forceinline__ float tq_fwht_warp(float val, unsigned lane) {
    static_assert(N <= 32 && (N & (N - 1)) == 0, "N must be power of 2, <= 32");
#pragma unroll
    for (int step = 1; step < N; step *= 2) {
        float other = __shfl_xor_sync(0xFFFFFFFF, val, step);
        val = (lane & step) ? (other - val) : (val + other);
    }
    return val * rsqrtf((float)N);
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
