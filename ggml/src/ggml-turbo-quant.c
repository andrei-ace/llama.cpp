// TurboQuant CPU reference — MSE quantizer + optional QJL on residual.
// FWHT (Hadamard) rotation applied inside quantize/dequantize.
// QJL uses FWHT as projection matrix for power-of-2 dims (32, 128).

#include "ggml-quants.h"
#include "ggml-impl.h"
#include "ggml-cpu/quants.h"
#include "ggml-turbo-quant.h"

#include <math.h>
#include <string.h>
#include <assert.h>
#include <limits.h>

// ---------------------------------------------------------------------------
// Centroids: exact Lloyd-Max for Beta((d-1)/2, (d-1)/2)
// Each independent instance uses centroids for its own dimension d.
// ---------------------------------------------------------------------------

// Precomputed Lloyd-Max centroids for various (d, b) configurations.
// Some may be unused at current operating points but kept for future use.

// d=32 (outlier instance)
static const float centroids_16_d32[16] = {
    -0.4533721997f, -0.3498559145f, -0.2764914062f, -0.2161194569f,
    -0.1628573862f, -0.1138475776f, -0.0673934339f, -0.0223187462f,
     0.0223187462f,  0.0673934339f,  0.1138475776f,  0.1628573862f,
     0.2161194569f,  0.2764914062f,  0.3498559145f,  0.4533721997f,
};
static const float centroids_8_d32[8] = {
    -0.3662682422f, -0.2324605670f, -0.1317560968f, -0.0428515156f,
     0.0428515156f,  0.1317560968f,  0.2324605670f,  0.3662682422f,
};
static const float centroids_4_d32[4] = {
    -0.2633194113f, -0.0798019295f, 0.0798019295f, 0.2633194113f,
};
static const float centroids_2_d32[2] = {
    -0.1421534638f, 0.1421534638f,
};

// d=96 (regular instance)
static const float centroids_8_d96[8] = {
    -0.2168529349f, -0.1361685800f, -0.0767954958f, -0.0249236898f,
     0.0249236898f,  0.0767954958f,  0.1361685800f,  0.2168529349f,
};
static const float centroids_4_d96[4] = {
    -0.1534455138f, -0.0461670286f, 0.0461670286f, 0.1534455138f,
};
static const float centroids_2_d96[2] = {
    -0.0816460916f, 0.0816460916f,
};

// d=128 (used by MSE types which operate on full vector)
// d=128 centroids — exact Lloyd-Max for Beta(63.5, 63.5) computed with scipy
static const float centroids_16[16] = {
    -0.2376271868f, -0.1807937296f, -0.1417616544f, -0.1102470655f,
    -0.0827925668f, -0.0577445357f, -0.0341340283f, -0.0112964982f,
     0.0112964982f,  0.0341340283f,  0.0577445357f,  0.0827925668f,
     0.1102470655f,  0.1417616544f,  0.1807937296f,  0.2376271868f,
};
static const float centroids_8[8] = {
    -0.1883971860f, -0.1181397670f, -0.0665856080f, -0.0216043106f,
     0.0216043106f,  0.0665856080f,  0.1181397670f,  0.1883971860f,
};
static const float centroids_4[4] = {
    -0.1330415202f, -0.0399915952f, 0.0399915952f, 0.1330415202f,
};
static const float centroids_2[2] = {
    -0.0706615727f, 0.0706615727f,
};

// d=256: exact Lloyd-Max for Beta(127.5, 127.5) on [-1, 1]
static const float centroids_16_d256[16] = {
    -0.1694104365f, -0.1285881998f, -0.1006980067f, -0.0782493129f,
    -0.0587321071f, -0.0409491965f, -0.0242008774f, -0.0080083741f,
     0.0080083741f,  0.0242008774f,  0.0409491965f,  0.0587321071f,
     0.0782493129f,  0.1006980067f,  0.1285881998f,  0.1694104365f,
};

static const float centroids_8_d256[8] = {
    -0.1338542901f, -0.0837654569f, -0.0471667103f, -0.0152974877f,
     0.0152974877f,  0.0471667103f,  0.0837654569f,  0.1338542901f,
};

// d=64: exact Lloyd-Max for Beta(31.5, 31.5) on [-1, 1]
// Used by 5hi_3lo d=256 outlier subset (64 channels)
static const float centroids_16_d64[16] = {
    -0.3389919281f, -0.2586479166f, -0.2033218447f, -0.1583036541f,
    -0.1191039932f, -0.0831167296f, -0.0491520456f, -0.0162627764f,
     0.0162627764f,  0.0491520456f,  0.0831167296f,  0.1191039932f,
     0.1583036541f,  0.2033218447f,  0.2586479166f,  0.3389919281f,
};

// d=192: exact Lloyd-Max for Beta(95.5, 95.5) on [-1, 1]
// Used by 5hi_3lo d=256 regular subset (192 channels)
static const float centroids_8_d192[8] = {
    -0.1543156657f, -0.0966361419f, -0.0544312518f, -0.0176559645f,
     0.0176559645f,  0.0544312518f,  0.0966361419f,  0.1543156657f,
};

// d=64, 5-bit: exact Lloyd-Max for Beta(31.5, 31.5) on [-1, 1]
static const float centroids_32_d64[32] = {
    -0.3894340320f, -0.3248614736f, -0.2815369080f, -0.2474916863f,
    -0.2187303715f, -0.1933882629f, -0.1704301210f, -0.1492144256f,
    -0.1293114668f, -0.1104157121f, -0.0922990338f, -0.0747836958f,
    -0.0577257093f, -0.0410039666f, -0.0245127359f, -0.0081561488f,
     0.0081561488f,  0.0245127359f,  0.0410039666f,  0.0577257093f,
     0.0747836958f,  0.0922990338f,  0.1104157121f,  0.1293114668f,
     0.1492144256f,  0.1704301210f,  0.1933882629f,  0.2187303715f,
     0.2474916863f,  0.2815369080f,  0.3248614736f,  0.3894340320f,
};

// d=64, 3-bit: exact Lloyd-Max for Beta(31.5, 31.5) on [-1, 1]
static const float centroids_8_d64[8] = {
    -0.2639139308f, -0.1661678589f, -0.0938322632f, -0.0304691789f,
     0.0304691789f,  0.0938322632f,  0.1661678589f,  0.2639139308f,
};

// d=64, 2-bit: exact Lloyd-Max for Beta(31.5, 31.5) on [-1, 1]
static const float centroids_4_d64[4] = {
    -0.1874968494f, -0.0565148688f,  0.0565148688f,  0.1874968494f,
};

// d=192, 2-bit: exact Lloyd-Max for Beta(95.5, 95.5) on [-1, 1]
static const float centroids_4_d192[4] = {
    -0.1087535813f, -0.0326609223f,  0.0326609223f,  0.1087535813f,
};

// d=192, 1-bit: exact Lloyd-Max for Beta(95.5, 95.5) on [-1, 1]
static const float centroids_2_d192[2] = {
    -0.0576573838f,  0.0576573838f,
};

// d=32, 5-bit: exact Lloyd-Max for Beta(15.5, 15.5) on [-1, 1]
static const float centroids_32_d32[32] = {
    -0.5264998987f, -0.4434050674f, -0.3863517231f, -0.3408589649f,
    -0.3020293074f, -0.2675540961f, -0.2361407504f, -0.2069821496f,
    -0.1795336985f, -0.1534053336f, -0.1283036018f, -0.1039980715f,
    -0.0803005660f, -0.0570515734f, -0.0341108450f, -0.0113504874f,
     0.0113504874f,  0.0341108450f,  0.0570515734f,  0.0803005660f,
     0.1039980715f,  0.1283036018f,  0.1534053336f,  0.1795336985f,
     0.2069821496f,  0.2361407504f,  0.2675540961f,  0.3020293074f,
     0.3408589649f,  0.3863517231f,  0.4434050674f,  0.5264998987f,
};

// ---------------------------------------------------------------------------
// Two independent rotation matrices: Π_hi (32×32) and Π_lo (96×96)
// Per the paper: split channels into outlier/regular sets, apply independent
// TurboQuant instances to each.
// ---------------------------------------------------------------------------

#define TQ_DIM     128
#define TQ_DIM_HI  32
#define TQ_DIM_LO  96

// Maximum sizes for d=256 support (arrays sized to largest possible dimension)
#define TQ_DIM_MAX     256
#define TQ_DIM_HI_MAX   64   // max outliers (d=256 case: 256/4 = 64)
#define TQ_DIM_LO_MAX  192   // max regular  (d=256 case: 256 - 64 = 192)

// Runtime head dimension — set during calibration init, defaults to 128
static int tq_head_dim = 128;

static float tq_rot_hi_fwd[TQ_DIM_HI * TQ_DIM_HI];  // Π_hi (row-major) — K cache outlier subset
static float tq_rot_hi_inv[TQ_DIM_HI * TQ_DIM_HI];
static float tq_rot_lo_fwd[TQ_DIM_LO * TQ_DIM_LO];  // Π_lo (row-major) — K cache regular subset
static float tq_rot_lo_inv[TQ_DIM_LO * TQ_DIM_LO];
static float tq_rot_v_fwd[TQ_DIM * TQ_DIM];          // Π_v (128×128) — V cache, no split
static float tq_rot_v_inv[TQ_DIM * TQ_DIM];

// ---------------------------------------------------------------------------
// Per-layer-per-head outlier channel registry (loaded from GGUF calibration)
// With GQA, different KV heads may have different outlier patterns.
// Dynamically allocated by tq_init_outlier_masks() based on actual model dims.
// ---------------------------------------------------------------------------

static int tq_n_layers = 0;
static int tq_n_heads  = 0;

// Per-layer-per-head outlier channel bitmask (K and V may differ)
// Bit i set = channel i is outlier. words_per_head = head_dim / 32
static uint32_t * tq_k_outlier_mask = NULL;  // [n_layers * n_heads * words_per_head]
static uint32_t * tq_v_outlier_mask = NULL;

// Accessor macros for flat dynamic arrays
#define TQ_MASK_K(l, h)     (tq_k_outlier_mask + ((l) * tq_n_heads + (h)) * (tq_head_dim / 32))
#define TQ_MASK_V(l, h)     (tq_v_outlier_mask + ((l) * tq_n_heads + (h)) * (tq_head_dim / 32))

// Helper: check if channel i is outlier
static inline int tq_is_outlier(const uint32_t * mask, int i) {
    return (mask[i / 32] >> (i % 32)) & 1;
}

// Helper: set channel i as outlier
static inline void tq_set_outlier(uint32_t * mask, int i) {
    mask[i / 32] |= (1u << (i % 32));
}

// Thread-local current context (set by CPU backend before quantize/dequantize/vec_dot)
static _Thread_local int tq_cur_layer = 0;
static _Thread_local int tq_cur_head  = 0;  // block index within row = KV head index
static _Thread_local int tq_cur_is_k  = 1;  // 1 = K cache, 0 = V cache

static int   tq_initialized = 0;

// Generate a d×d orthogonal matrix via QR of Gaussian
static void tq_gen_orthogonal(float * fwd, float * inv, int d, uint64_t seed) {
    #define TQ_ROT_LCG(s) ((s) * 6364136223846793005ULL + 1442695040888963407ULL)
    float * Q = (float *)malloc(d * d * sizeof(float));
    for (int i = 0; i < d * d; i++) {
        seed = TQ_ROT_LCG(seed);
        float u1 = ((float)(uint32_t)(seed >> 32) + 1.0f) / ((float)0xFFFFFFFF + 2.0f);
        seed = TQ_ROT_LCG(seed);
        float u2 = ((float)(uint32_t)(seed >> 32) + 1.0f) / ((float)0xFFFFFFFF + 2.0f);
        Q[i] = sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
    }
    for (int j = 0; j < d; j++) {
        for (int k = 0; k < j; k++) {
            float dot = 0.0f;
            for (int i = 0; i < d; i++) dot += Q[k*d+i] * Q[j*d+i];
            for (int i = 0; i < d; i++) Q[j*d+i] -= dot * Q[k*d+i];
        }
        float norm = 0.0f;
        for (int i = 0; i < d; i++) norm += Q[j*d+i] * Q[j*d+i];
        norm = sqrtf(norm);
        for (int i = 0; i < d; i++) Q[j*d+i] /= norm;
    }
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            fwd[i*d+j] = Q[j*d+i];
            inv[j*d+i] = Q[j*d+i];
        }
    }
    free(Q);
    #undef TQ_ROT_LCG
}

static void tq_init_rotations(void) {
    if (tq_initialized) return;
    tq_gen_orthogonal(tq_rot_hi_fwd, tq_rot_hi_inv, TQ_DIM_HI, 0x5475524230484932ULL); // "TuRB0HI2"
    tq_gen_orthogonal(tq_rot_lo_fwd, tq_rot_lo_inv, TQ_DIM_LO, 0x54755242304C4F36ULL); // "TuRB0LO6"
    tq_gen_orthogonal(tq_rot_v_fwd, tq_rot_v_inv, TQ_DIM, 0x5475524230564131ULL);      // "TuRB0VA1"
    tq_initialized = 1;
}

void tq_set_head_dim(int dim) {
    tq_head_dim = dim;
}

int tq_get_head_dim(void) {
    return tq_head_dim;
}

// Extract hi/lo channel subsets from a 128-dim vector
// K cache: uses calibrated per-layer-per-head outlier bitmask
// V cache: uses default fixed split (0-31 / 32-127) — V has no outliers (per RotateKV paper)
static void tq_split_channels(const float * x, float * hi, float * lo) {
    const uint32_t * mask = tq_cur_is_k ? TQ_MASK_K(tq_cur_layer, tq_cur_head) : TQ_MASK_V(0, 0);
    int oi = 0, ri = 0;
    for (int i = 0; i < TQ_DIM; i++) {
        if (tq_is_outlier(mask, i)) hi[oi++] = x[i];
        else                        lo[ri++] = x[i];
    }
}

// Merge hi/lo channel subsets back into 128-dim vector
static void tq_merge_channels(const float * hi, const float * lo, float * x) {
    const uint32_t * mask = tq_cur_is_k ? TQ_MASK_K(tq_cur_layer, tq_cur_head) : TQ_MASK_V(0, 0);
    int oi = 0, ri = 0;
    for (int i = 0; i < TQ_DIM; i++) {
        if (tq_is_outlier(mask, i)) x[i] = hi[oi++];
        else                        x[i] = lo[ri++];
    }
}

// Rotate hi subset: out = Π_hi * in
static void tq_rotate_hi(const float * in, float * out) {
    for (int i = 0; i < TQ_DIM_HI; i++) {
        float sum = 0.0f;
        for (int j = 0; j < TQ_DIM_HI; j++) sum += tq_rot_hi_fwd[i*TQ_DIM_HI+j] * in[j];
        out[i] = sum;
    }
}
static void tq_unrotate_hi(const float * in, float * out) {
    for (int i = 0; i < TQ_DIM_HI; i++) {
        float sum = 0.0f;
        for (int j = 0; j < TQ_DIM_HI; j++) sum += tq_rot_hi_inv[i*TQ_DIM_HI+j] * in[j];
        out[i] = sum;
    }
}

// Rotate lo subset: out = Π_lo * in
static void tq_rotate_lo(const float * in, float * out) {
    for (int i = 0; i < TQ_DIM_LO; i++) {
        float sum = 0.0f;
        for (int j = 0; j < TQ_DIM_LO; j++) sum += tq_rot_lo_fwd[i*TQ_DIM_LO+j] * in[j];
        out[i] = sum;
    }
}
static void tq_unrotate_lo(const float * in, float * out) {
    for (int i = 0; i < TQ_DIM_LO; i++) {
        float sum = 0.0f;
        for (int j = 0; j < TQ_DIM_LO; j++) sum += tq_rot_lo_inv[i*TQ_DIM_LO+j] * in[j];
        out[i] = sum;
    }
}

// Rotate full 128-dim vector: out = Π_v * in (for V cache, no split)
static void tq_rotate_v(const float * in, float * out) {
    for (int i = 0; i < TQ_DIM; i++) {
        float sum = 0.0f;
        for (int j = 0; j < TQ_DIM; j++) sum += tq_rot_v_fwd[i*TQ_DIM+j] * in[j];
        out[i] = sum;
    }
}
static void tq_unrotate_v(const float * in, float * out) {
    for (int i = 0; i < TQ_DIM; i++) {
        float sum = 0.0f;
        for (int j = 0; j < TQ_DIM; j++) sum += tq_rot_v_inv[i*TQ_DIM+j] * in[j];
        out[i] = sum;
    }
}

// Full split rotate/unrotate: split into outlier/regular, rotate each independently
static void tq_rotate_split(const float * in, float * hi_rot, float * lo_rot) {
    float hi[TQ_DIM_HI], lo[TQ_DIM_LO];
    tq_split_channels(in, hi, lo);
    tq_rotate_hi(hi, hi_rot);
    tq_rotate_lo(lo, lo_rot);
}

static void tq_unrotate_merge(const float * hi_rot, const float * lo_rot, float * out) {
    float hi[TQ_DIM_HI], lo[TQ_DIM_LO];
    tq_unrotate_hi(hi_rot, hi);
    tq_unrotate_lo(lo_rot, lo);
    tq_merge_channels(hi, lo, out);
}

// ---------------------------------------------------------------------------
// QJL (Quantized Johnson-Lindenstrauss) — FWHT-based projection
// Uses Hadamard matrix via FWHT as projection for all dims (power-of-2).
// Non-power-of-2 dims (96, 192) use block-diagonal FWHT (3 × block_dim).
// ---------------------------------------------------------------------------

#define QJL_SEED_32   0x514A4C20ULL  // unused, kept for API compat
#define QJL_SEED_64   0x514A4C40ULL
#define QJL_SEED_128  0x514A4C80ULL
#define QJL_SEED_256  0x514A4D00ULL

// Forward declaration — defined later, needed by QJL FWHT path
static void tq_fwht(float * x, int n);

// QJL forward: FWHT projection, store signs, return ||r||.
// m must be power-of-2. For non-power-of-2 (96, 192), callers use
// block-diagonal calls (3 x qjl_forward with block_dim).
static float qjl_forward(const float * r, uint8_t * signs, int m, uint64_t seed) {
    (void)seed;
    float rnorm_sq = 0.0f;
    for (int j = 0; j < m; j++) rnorm_sq += r[j] * r[j];
    memset(signs, 0, (m + 7) / 8);
    float proj[TQK_BLOCK_SIZE_D256];
    for (int j = 0; j < m; j++) proj[j] = r[j];
    tq_fwht(proj, m);
    for (int i = 0; i < m; i++) {
        if (proj[i] >= 0.0f) signs[i / 8] |= (uint8_t)(1 << (i % 8));
    }
    return sqrtf(rnorm_sq);
}

// QJL inverse: reconstruct correction from signs via FWHT.
static void qjl_inverse(const uint8_t * signs, float rnorm, float * corr, int m, uint64_t seed) {
    (void)seed;
    for (int i = 0; i < m; i++) {
        corr[i] = ((signs[i / 8] >> (i % 8)) & 1) ? 1.0f : -1.0f;
    }
    tq_fwht(corr, m);
    float scale = 1.2533141f / (float)m * rnorm;
    for (int j = 0; j < m; j++) corr[j] *= scale;
}

// QJL asymmetric dot: FWHT-rotate q, dot with signs.
static float qjl_asymmetric_dot(const float * q, int m, uint64_t seed,
                                 const uint8_t * signs, float rnorm) {
    (void)seed;
    if (rnorm == 0.0f) return 0.0f;
    float q_proj[TQK_BLOCK_SIZE_D256];
    for (int j = 0; j < m; j++) q_proj[j] = q[j];
    tq_fwht(q_proj, m);
    float sum = 0.0f;
    for (int i = 0; i < m; i++) {
        float sign = ((signs[i / 8] >> (i % 8)) & 1) ? 1.0f : -1.0f;
        sum += q_proj[i] * sign;
    }
    return 1.2533141f / (float)m * rnorm * sum;
}

// Block-diagonal QJL for non-power-of-2 dims (96 = 3×32, 192 = 3×64).
// Each block gets independent FWHT projection. Matches Metal per-element correction.
static float qjl_forward_block3(const float * r, uint8_t * signs, int block_dim) {
    int m = 3 * block_dim;
    float rnorm_sq = 0;
    for (int j = 0; j < m; j++) rnorm_sq += r[j] * r[j];
    memset(signs, 0, (m + 7) / 8);
    for (int b = 0; b < 3; b++) {
        float proj[TQK_N_OUTLIER_D256]; // max block = 64
        int off = b * block_dim;
        for (int j = 0; j < block_dim; j++) proj[j] = r[off + j];
        tq_fwht(proj, block_dim);
        for (int i = 0; i < block_dim; i++) {
            if (proj[i] >= 0.0f) signs[(off+i) / 8] |= (uint8_t)(1 << ((off+i) % 8));
        }
    }
    return sqrtf(rnorm_sq);
}

static void qjl_inverse_block3(const uint8_t * signs, float rnorm, float * corr, int block_dim) {
    int m = 3 * block_dim;
    for (int b = 0; b < 3; b++) {
        int off = b * block_dim;
        for (int i = 0; i < block_dim; i++)
            corr[off+i] = ((signs[(off+i)/8] >> ((off+i)%8)) & 1) ? 1.0f : -1.0f;
        tq_fwht(corr + off, block_dim);
    }
    float scale = 1.2533141f / (float)m * rnorm;
    for (int j = 0; j < m; j++) corr[j] *= scale;
}

static float qjl_asymmetric_dot_block3(const float * q, int block_dim,
                                        const uint8_t * signs, float rnorm) {
    if (rnorm == 0.0f) return 0.0f;
    int m = 3 * block_dim;
    float sum = 0;
    for (int b = 0; b < 3; b++) {
        float q_proj[TQK_N_OUTLIER_D256];
        int off = b * block_dim;
        for (int j = 0; j < block_dim; j++) q_proj[j] = q[off + j];
        tq_fwht(q_proj, block_dim);
        for (int i = 0; i < block_dim; i++) {
            float sign = ((signs[(off+i)/8] >> ((off+i)%8)) & 1) ? 1.0f : -1.0f;
            sum += q_proj[i] * sign;
        }
    }
    return 1.2533141f / (float)m * rnorm * sum;
}

// Fast variant when q is already FWHT-rotated — O(m) dot with signs
static float qjl_asymmetric_dot_rotated(const float * q_rot, int m,
                                         const uint8_t * signs, float rnorm) {
    if (rnorm == 0.0f) return 0.0f;
    float sum = 0.0f;
    for (int i = 0; i < m; i++) {
        float sign = ((signs[i / 8] >> (i % 8)) & 1) ? 1.0f : -1.0f;
        sum += q_rot[i] * sign;
    }
    return 1.2533141f / (float)m * rnorm * sum;
}

// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static inline int nearest(float val, const float * c, int n) {
    int best = 0; float bd = fabsf(val - c[0]);
    for (int i = 1; i < n; i++) { float d = fabsf(val - c[i]); if (d < bd) { bd = d; best = i; } }
    return best;
}

static inline void pk3(uint8_t * q, int j, int v) {
    int bp = j*3, bi = bp>>3, sh = bp&7;
    q[bi] |= (uint8_t)((v<<sh)&0xFF);
    if (sh > 5) q[bi+1] |= (uint8_t)(v>>(8-sh));
}

static inline int up3(const uint8_t * q, int j) {
    int bp = j*3, bi = bp>>3, sh = bp&7;
    return (sh <= 5) ? (q[bi]>>sh)&7 : ((q[bi]>>sh)|(q[bi+1]<<(8-sh)))&7;
}

static inline void pk4(uint8_t * q, int j, int v) {
    q[j / 2] |= (uint8_t)(v << ((j % 2) * 4));
}

static inline int up4(const uint8_t * q, int j) {
    return (q[j / 2] >> ((j % 2) * 4)) & 0xF;
}

static inline void pk5(uint8_t * q, int j, int v) {
    int bit = j * 5, byte0 = bit / 8, shift = bit % 8;
    q[byte0] |= (uint8_t)((v & 0x1F) << shift);
    if (shift > 3) q[byte0 + 1] |= (uint8_t)((v & 0x1F) >> (8 - shift));
}
static inline int up5(const uint8_t * q, int j) {
    int bit = j * 5, byte0 = bit / 8, shift = bit % 8;
    int val = q[byte0] >> shift;
    if (shift > 3) val |= q[byte0 + 1] << (8 - shift);
    return val & 0x1F;
}

static inline void pk2(uint8_t * q, int j, int v) {
    q[j / 4] |= (uint8_t)(v << ((j % 4) * 2));
}
static inline int up2(const uint8_t * q, int j) {
    return (q[j / 4] >> ((j % 4) * 2)) & 3;
}

static inline void pk1(uint8_t * q, int j, int v) {
    q[j / 8] |= (uint8_t)(v << (j % 8));
}
static inline int up1(const uint8_t * q, int j) {
    return (q[j / 8] >> (j % 8)) & 1;
}

// ---------------------------------------------------------------------------
// Generic quantize/dequantize helpers (operate in rotated space)
// ---------------------------------------------------------------------------

static void quant_hi(const float * xb, float inv_norm, uint8_t * qs,
                     const float * c, int n_c, int bits, int n_hi,
                     float * residual) {
    memset(qs, 0, n_hi * bits / 8);
    for (int j = 0; j < n_hi; j++) {
        float xn = xb[j] * inv_norm;
        int idx = nearest(xn, c, n_c);
        if (bits == 2) {
            qs[j / 4] |= (uint8_t)(idx << ((j % 4) * 2));
        } else {  // bits == 3
            pk3(qs, j, idx);
        }
        residual[j] = xn - c[idx];
    }
}

static void quant_lo(const float * xb, float inv_norm, uint8_t * qs,
                     const float * c, int n_c, int bits, int n_lo, int offset,
                     float * residual) {
    memset(qs, 0, (n_lo * bits + 7) / 8);
    for (int j = 0; j < n_lo; j++) {
        float xn = xb[offset + j] * inv_norm;
        int idx;
        if (bits == 1) {
            idx = (xn >= 0.0f) ? 1 : 0;
            qs[j / 8] |= (uint8_t)(idx << (j % 8));
        } else {  // bits == 2
            idx = nearest(xn, c, n_c);
            qs[j / 4] |= (uint8_t)(idx << ((j % 4) * 2));
        }
        residual[j] = xn - c[idx];
    }
}

static void dequant_hi(const uint8_t * qs, const float * corr,
                       const float * c, int bits, int n_hi,
                       float norm, float * out) {
    for (int j = 0; j < n_hi; j++) {
        int idx;
        if (bits == 2) {
            idx = (qs[j / 4] >> ((j % 4) * 2)) & 0x3;
        } else {  // bits == 3
            idx = up3(qs, j);
        }
        out[j] = norm * (c[idx] + corr[j]);
    }
}

static void dequant_lo(const uint8_t * qs, const float * corr,
                       const float * c, int bits, int n_lo,
                       float norm, float * out, int offset) {
    for (int j = 0; j < n_lo; j++) {
        int idx;
        if (bits == 1) {
            idx = (qs[j / 8] >> (j % 8)) & 1;
        } else {  // bits == 2
            idx = (qs[j / 4] >> ((j % 4) * 2)) & 0x3;
        }
        out[offset + j] = norm * (c[idx] + corr[j]);
    }
}

// MSE-only helpers (no QJL correction)
static void quant_hi_mse(const float * xb, float inv_norm, uint8_t * qs,
                          const float * c, int n_c, int bits, int n_hi) {
    memset(qs, 0, (n_hi * bits + 7) / 8);
    for (int j = 0; j < n_hi; j++) {
        float xn = xb[j] * inv_norm;
        int idx = nearest(xn, c, n_c);
        if (bits == 2)      { qs[j / 4] |= (uint8_t)(idx << ((j % 4) * 2)); }
        else if (bits == 3) { pk3(qs, j, idx); }
        else                { pk4(qs, j, idx); }  // bits == 4
    }
}

static void quant_lo_mse(const float * xb, float inv_norm, uint8_t * qs,
                          const float * c, int n_c, int bits, int n_lo, int offset) {
    memset(qs, 0, (n_lo * bits + 7) / 8);
    for (int j = 0; j < n_lo; j++) {
        float xn = xb[offset + j] * inv_norm;
        int idx;
        if (bits == 1) {
            idx = (xn >= 0.0f) ? 1 : 0;
            qs[j / 8] |= (uint8_t)(idx << (j % 8));
        } else if (bits == 2) {
            idx = nearest(xn, c, n_c);
            qs[j / 4] |= (uint8_t)(idx << ((j % 4) * 2));
        } else {  // bits == 3
            idx = nearest(xn, c, n_c);
            pk3(qs, j, idx);
        }
    }
}

static void dequant_hi_mse(const uint8_t * qs, const float * c, int bits, int n_hi,
                            float norm, float * out) {
    for (int j = 0; j < n_hi; j++) {
        int idx;
        if (bits == 2)      { idx = (qs[j / 4] >> ((j % 4) * 2)) & 0x3; }
        else if (bits == 3) { idx = up3(qs, j); }
        else                { idx = up4(qs, j); }  // bits == 4
        out[j] = norm * c[idx];
    }
}

static void dequant_lo_mse(const uint8_t * qs, const float * c, int bits, int n_lo,
                            float norm, float * out, int offset) {
    for (int j = 0; j < n_lo; j++) {
        int idx;
        if (bits == 1)      { idx = (qs[j / 8] >> (j % 8)) & 1; }
        else if (bits == 2) { idx = (qs[j / 4] >> ((j % 4) * 2)) & 0x3; }
        else                { idx = up3(qs, j); }  // bits == 3
        out[offset + j] = norm * c[idx];
    }
}

// ---------------------------------------------------------------------------
// Public API: per-layer outlier calibration
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------

void tq_set_current_layer(int layer, int is_k) {
    tq_cur_layer = layer;
    tq_cur_is_k  = is_k;
}

void tq_set_current_head(int head) {
    tq_cur_head = (head >= 0 && (tq_n_heads == 0 || head < tq_n_heads)) ? head : 0;
}

// ---------------------------------------------------------------------------
// Public accessors for rotation matrices and channel maps (used by Metal/CUDA)
// ---------------------------------------------------------------------------

const float * tq_get_rot_v_fwd(void)  { tq_init_rotations(); return tq_rot_v_fwd; }
const float * tq_get_rot_v_inv(void)  { tq_init_rotations(); return tq_rot_v_inv; }
const float * tq_get_rot_hi_fwd(void) { tq_init_rotations(); return tq_rot_hi_fwd; }
const float * tq_get_rot_hi_inv(void) { tq_init_rotations(); return tq_rot_hi_inv; }
const float * tq_get_rot_lo_fwd(void) { tq_init_rotations(); return tq_rot_lo_fwd; }
const float * tq_get_rot_lo_inv(void) { tq_init_rotations(); return tq_rot_lo_inv; }

int tq_get_rot_v_size(void)  { return TQ_DIM * TQ_DIM; }
int tq_get_rot_hi_size(void) { return TQ_DIM_HI * TQ_DIM_HI; }
int tq_get_rot_lo_size(void) { return TQ_DIM_LO * TQ_DIM_LO; }

void tq_get_channel_map(int layer, int head, int is_k, int * outlier, int * regular) {
    tq_init_rotations();
    if (layer < 0 || layer >= tq_n_layers || head < 0 || head >= tq_n_heads) return;
    if (!tq_k_outlier_mask) return;
    const uint32_t * mask = is_k ? TQ_MASK_K(layer, head) : TQ_MASK_V(layer, head);
    int oi = 0, ri = 0;
    for (int i = 0; i < tq_head_dim; i++) {
        if (tq_is_outlier(mask, i)) {
            outlier[oi++] = i;
        } else {
            regular[ri++] = i;
        }
    }
}

// Build compact channel permutation [outlier_ch(n_hi), regular_ch(n_lo)] as uint8_t for Metal FA
void tq_get_channel_perm(int layer, int head, int is_k, uint8_t * perm) {
    if (!tq_k_outlier_mask) {
        for (int i = 0; i < tq_head_dim; i++) perm[i] = (uint8_t)i;
        return;
    }
    const uint32_t * mask = is_k ? TQ_MASK_K(layer, head) : TQ_MASK_V(layer, head);
    int n_hi = tq_head_dim / 4;
    int oi = 0, ri = n_hi;
    for (int i = 0; i < tq_head_dim; i++) {
        if (tq_is_outlier(mask, i)) {
            perm[oi++] = (uint8_t)i;
        } else {
            perm[ri++] = (uint8_t)i;
        }
    }
}

void tq_get_qjl_matrix(float * out, int dim, uint64_t seed) {
    // Generate the same i.i.d. Gaussian matrix used by QJL forward/inverse
    uint64_t st = seed;
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            st = st * 6364136223846793005ULL + 1442695040888963407ULL;
            float u1 = ((float)(uint32_t)(st >> 32) + 1.0f) / ((float)0xFFFFFFFF + 2.0f);
            st = st * 6364136223846793005ULL + 1442695040888963407ULL;
            float u2 = ((float)(uint32_t)(st >> 32) + 1.0f) / ((float)0xFFFFFFFF + 2.0f);
            out[i * dim + j] = sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
        }
    }
}

// ---------------------------------------------------------------------------
// Outlier mask management — loaded from GGUF calibration data
// ---------------------------------------------------------------------------

void tq_init_outlier_masks(int n_layers, int n_heads, int head_dim) {
    tq_n_layers = n_layers;
    tq_n_heads  = n_heads;
    tq_head_dim = head_dim;

    const int words_per_head = head_dim / 32;
    const int total_words = n_layers * n_heads * words_per_head;

    tq_k_outlier_mask = (uint32_t *)calloc(total_words, sizeof(uint32_t));
    tq_v_outlier_mask = (uint32_t *)calloc(total_words, sizeof(uint32_t));

    // Default: identity permutation (channels 0..n_hi-1 as outliers)
    const int n_hi = head_dim / 4;
    for (int l = 0; l < n_layers; l++) {
        for (int h = 0; h < n_heads; h++) {
            uint32_t * mk = TQ_MASK_K(l, h);
            uint32_t * mv = TQ_MASK_V(l, h);
            for (int i = 0; i < n_hi; i++) {
                tq_set_outlier(mk, i);
                tq_set_outlier(mv, i);
            }
        }
    }

    tq_init_rotations();
}

void tq_set_outlier_mask_from_perm(int layer, int head, const uint8_t * perm, int head_dim) {
    if (!tq_k_outlier_mask || layer < 0 || layer >= tq_n_layers || head < 0 || head >= tq_n_heads) return;

    const int n_hi = head_dim / 4;
    const int words_per_head = head_dim / 32;
    uint32_t * mk = TQ_MASK_K(layer, head);

    // Clear existing mask
    memset(mk, 0, words_per_head * sizeof(uint32_t));

    // First n_hi entries in perm are outlier channel indices
    for (int i = 0; i < n_hi; i++) {
        tq_set_outlier(mk, perm[i]);
    }
}

void tq_free_outlier_masks(void) {
    free(tq_k_outlier_mask); tq_k_outlier_mask = NULL;
    free(tq_v_outlier_mask); tq_v_outlier_mask = NULL;
    tq_n_layers = 0;
    tq_n_heads  = 0;
}

// ---------------------------------------------------------------------------
// Global channel map for GPU backends
// Built by tq_upload_channel_maps_to_devices(), consumed by Metal/CUDA.
// Layout: [n_layers][n_heads][head_dim] int32 — first n_hi outlier indices, then n_lo regular.
// ---------------------------------------------------------------------------

static int * tq_global_chmap = NULL;
static int   tq_global_chmap_n_layers = 0;
static int   tq_global_chmap_n_heads  = 0;

const int * tq_get_global_channel_map(int * out_n_layers, int * out_n_heads) {
    if (out_n_layers) *out_n_layers = tq_global_chmap_n_layers;
    if (out_n_heads)  *out_n_heads  = tq_global_chmap_n_heads;
    return tq_global_chmap;
}

void tq_upload_channel_maps_to_devices(void) {
    if (!tq_k_outlier_mask || tq_n_layers == 0 || tq_n_heads == 0) return;

    const int dim = tq_head_dim;
    const int n_hi = dim / 4;

    // (Re)build global int32 channel map
    free(tq_global_chmap);
    tq_global_chmap = (int *)malloc((size_t)tq_n_layers * tq_n_heads * dim * sizeof(int));
    if (!tq_global_chmap) return;

    tq_global_chmap_n_layers = tq_n_layers;
    tq_global_chmap_n_heads  = tq_n_heads;

    for (int l = 0; l < tq_n_layers; l++) {
        for (int h = 0; h < tq_n_heads; h++) {
            int * row = tq_global_chmap + ((size_t)l * tq_n_heads + h) * dim;
            tq_get_channel_map(l, h, 1, row, row + n_hi);
        }
    }
}

// ---------------------------------------------------------------------------
// Fast Walsh-Hadamard Transform (FWHT) — in-place, normalized, self-inverse
// n must be a power of 2.  H_n = (1/√n) * Walsh-Hadamard matrix.
// ---------------------------------------------------------------------------

static void tq_fwht(float * x, int n) {
    for (int step = 1; step < n; step *= 2) {
        for (int i = 0; i < n; i += 2 * step) {
            for (int j = 0; j < step; j++) {
                float a = x[i + j], b = x[i + j + step];
                x[i + j]        = a + b;
                x[i + j + step] = a - b;
            }
        }
    }
    float s = 1.0f / sqrtf((float)n);
    for (int i = 0; i < n; i++) x[i] *= s;
}

// ---------------------------------------------------------------------------
// TQK had_mse4: H_128 Hadamard + 4-bit MSE, no split, no calibration
// ---------------------------------------------------------------------------

void quantize_row_tqk_had_mse4_ref(const float * GGML_RESTRICT x, block_tqk_had_mse4 * GGML_RESTRICT y, int64_t k) {
    assert(k % TQK_BLOCK_SIZE == 0);
    const int64_t nb = k / TQK_BLOCK_SIZE;

    for (int64_t i = 0; i < nb; i++) {
        const float * xb = x + i * TQK_BLOCK_SIZE;

        float sum_sq = 0.0f;
        for (int j = 0; j < TQ_DIM; j++) sum_sq += xb[j] * xb[j];
        float norm = sqrtf(sum_sq);
        y[i].norm = GGML_FP32_TO_FP16(norm);

        if (norm == 0.0f) { memset(y[i].qs, 0, sizeof(y[i].qs)); continue; }
        float inv = 1.0f / norm;

        float rot[TQ_DIM];
        for (int j = 0; j < TQ_DIM; j++) rot[j] = xb[j] * inv;
        tq_fwht(rot, TQ_DIM);

        memset(y[i].qs, 0, sizeof(y[i].qs));
        for (int j = 0; j < TQ_DIM; j++) {
            int idx = nearest(rot[j], centroids_16, 16);
            pk4(y[i].qs, j, idx);
        }
    }
}

void dequantize_row_tqk_had_mse4(const block_tqk_had_mse4 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % TQK_BLOCK_SIZE == 0);
    const int64_t nb = k / TQK_BLOCK_SIZE;

    for (int64_t i = 0; i < nb; i++) {
        float norm = GGML_FP16_TO_FP32(x[i].norm);
        float rot[TQ_DIM];
        for (int j = 0; j < TQ_DIM; j++) {
            rot[j] = centroids_16[up4(x[i].qs, j)];
        }
        tq_fwht(rot, TQ_DIM);
        for (int j = 0; j < TQ_DIM; j++) y[i * TQK_BLOCK_SIZE + j] = norm * rot[j];
    }
}

// Asymmetric vec_dot: rotate Q with H_128, dot with stored centroids
void ggml_vec_dot_tqk_had_mse4_f32(
        int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, size_t bx,
        const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % TQK_BLOCK_SIZE == 0);
    assert(nrc == 1);
    (void)bs; (void)bx; (void)by; (void)nrc;

    const block_tqk_had_mse4 * GGML_RESTRICT x = (const block_tqk_had_mse4 *)vx;
    const float * GGML_RESTRICT y = (const float *)vy;
    const int64_t nb = n / TQK_BLOCK_SIZE;

    float sumf = 0.0f;

    for (int64_t i = 0; i < nb; i++) {
        const float * q = y + i * TQK_BLOCK_SIZE;

        float norm = GGML_FP16_TO_FP32(x[i].norm);

        // Rotate Q with H_128
        float q_rot[TQ_DIM];
        for (int j = 0; j < TQ_DIM; j++) q_rot[j] = q[j];
        tq_fwht(q_rot, TQ_DIM);

        // Dot with stored centroids
        float dot = 0.0f;
        for (int j = 0; j < TQ_DIM; j++) {
            dot += q_rot[j] * centroids_16[up4(x[i].qs, j)];
        }

        sumf += norm * dot;
    }

    *s = sumf;
}

// ---------------------------------------------------------------------------
// TQV had_mse4: V cache 4-bit MSE, per-block norm (no rotation)
// ---------------------------------------------------------------------------

// Quantize: same as K (normalize, FWHT, 4-bit MSE). Stores rotated values.
void quantize_row_tqv_had_mse4_ref(const float * GGML_RESTRICT x, block_tqv_had_mse4 * GGML_RESTRICT y, int64_t k) {
    quantize_row_tqk_had_mse4_ref(x, (block_tqk_had_mse4 *)y, k);
}

// Dequantize: centroids → inverse FWHT → scale by norm (full reconstruction)
void dequantize_row_tqv_had_mse4(const block_tqv_had_mse4 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % TQK_BLOCK_SIZE == 0);
    const int64_t nb = k / TQK_BLOCK_SIZE;

    for (int64_t i = 0; i < nb; i++) {
        float norm = GGML_FP16_TO_FP32(x[i].norm);
        float rot[TQ_DIM];
        for (int j = 0; j < TQ_DIM; j++) {
            rot[j] = centroids_16[up4(x[i].qs, j)];
        }
        tq_fwht(rot, TQ_DIM); // inverse FWHT = FWHT (self-inverse)
        for (int j = 0; j < TQ_DIM; j++) y[i * TQK_BLOCK_SIZE + j] = norm * rot[j];
    }
}

// ---------------------------------------------------------------------------
// TQK had_prod5: H_128 Hadamard + 4-bit MSE + 1-bit QJL (unbiased estimator)
// No split, no calibration. QJL on residual corrects MSE bias.
// ---------------------------------------------------------------------------

void quantize_row_tqk_had_prod5_ref(const float * GGML_RESTRICT x, block_tqk_had_prod5 * GGML_RESTRICT y, int64_t k) {
    assert(k % TQK_BLOCK_SIZE == 0);
    const int64_t nb = k / TQK_BLOCK_SIZE;

    for (int64_t i = 0; i < nb; i++) {
        const float * xb = x + i * TQK_BLOCK_SIZE;

        // L2 norm
        float sum_sq = 0.0f;
        for (int j = 0; j < TQ_DIM; j++) sum_sq += xb[j] * xb[j];
        float norm = sqrtf(sum_sq);
        y[i].norm = GGML_FP32_TO_FP16(norm);
        y[i].rnorm = GGML_FP32_TO_FP16(0.0f);
        memset(y[i].signs, 0, sizeof(y[i].signs));

        if (norm == 0.0f) { memset(y[i].qs, 0, sizeof(y[i].qs)); continue; }
        float inv = 1.0f / norm;

        // Normalize and apply H_128 via FWHT
        float rot[TQ_DIM];
        for (int j = 0; j < TQ_DIM; j++) rot[j] = xb[j] * inv;
        tq_fwht(rot, TQ_DIM);

        // 4-bit MSE quantize (same as had_mse4)
        memset(y[i].qs, 0, sizeof(y[i].qs));
        for (int j = 0; j < TQ_DIM; j++) {
            int idx = nearest(rot[j], centroids_16, 16);
            pk4(y[i].qs, j, idx);
        }

        // Compute residual in ORIGINAL space: r = x - norm * H^{-1} * centroids
        float recon[TQ_DIM];
        for (int j = 0; j < TQ_DIM; j++) recon[j] = centroids_16[up4(y[i].qs, j)];
        tq_fwht(recon, TQ_DIM);  // inverse FWHT = FWHT (orthogonal, self-inverse)
        float resid[TQ_DIM];
        for (int j = 0; j < TQ_DIM; j++) resid[j] = xb[j] - norm * recon[j];

        // QJL forward on residual
        y[i].rnorm = GGML_FP32_TO_FP16(qjl_forward(resid, y[i].signs, TQ_DIM, QJL_SEED_128));
    }
}

void dequantize_row_tqk_had_prod5(const block_tqk_had_prod5 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % TQK_BLOCK_SIZE == 0);
    const int64_t nb = k / TQK_BLOCK_SIZE;

    for (int64_t i = 0; i < nb; i++) {
        float norm = GGML_FP16_TO_FP32(x[i].norm);

        // MSE reconstruction: centroids → inverse FWHT → scale by norm
        float rot[TQ_DIM];
        for (int j = 0; j < TQ_DIM; j++) {
            rot[j] = centroids_16[up4(x[i].qs, j)];
        }
        tq_fwht(rot, TQ_DIM);
        for (int j = 0; j < TQ_DIM; j++) y[i * TQK_BLOCK_SIZE + j] = norm * rot[j];

        // QJL correction on residual
        float corr[TQ_DIM];
        qjl_inverse(x[i].signs, GGML_FP16_TO_FP32(x[i].rnorm), corr, TQ_DIM, QJL_SEED_128);
        for (int j = 0; j < TQ_DIM; j++) y[i * TQK_BLOCK_SIZE + j] += corr[j];
    }
}

// Asymmetric vec_dot: rotate Q with H_128, dot with stored centroids + QJL correction
void ggml_vec_dot_tqk_had_prod5_f32(
        int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, size_t bx,
        const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % TQK_BLOCK_SIZE == 0);
    assert(nrc == 1);
    (void)bs; (void)bx; (void)by; (void)nrc;

    const block_tqk_had_prod5 * GGML_RESTRICT x = (const block_tqk_had_prod5 *)vx;
    const float * GGML_RESTRICT y = (const float *)vy;
    const int64_t nb = n / TQK_BLOCK_SIZE;

    float sumf = 0.0f;

    for (int64_t i = 0; i < nb; i++) {
        const float * q = y + i * TQK_BLOCK_SIZE;

        float norm = GGML_FP16_TO_FP32(x[i].norm);

        // Rotate Q with H_128
        float q_rot[TQ_DIM];
        for (int j = 0; j < TQ_DIM; j++) q_rot[j] = q[j];
        tq_fwht(q_rot, TQ_DIM);

        // MSE centroid dot
        float dot = 0.0f;
        for (int j = 0; j < TQ_DIM; j++) {
            dot += q_rot[j] * centroids_16[up4(x[i].qs, j)];
        }
        float mse_dot = norm * dot;

        // QJL correction: q_rot is already FWHT(q), which is H·q — exact match for Hadamard QJL
        float qjl_dot = qjl_asymmetric_dot_rotated(q_rot, TQ_DIM,
                                            x[i].signs, GGML_FP16_TO_FP32(x[i].rnorm));

        sumf += mse_dot + qjl_dot;
    }

    *s = sumf;
}

// ---------------------------------------------------------------------------
// TQK had_prod4: H_128 Hadamard + 3-bit MSE + 1-bit QJL (4.25 bpv, unbiased)
// Same structure as had_prod5 but uses 3-bit (8 centroids) instead of 4-bit.
// ---------------------------------------------------------------------------

void quantize_row_tqk_had_prod4_ref(const float * GGML_RESTRICT x, block_tqk_had_prod4 * GGML_RESTRICT y, int64_t k) {
    assert(k % TQK_BLOCK_SIZE == 0);
    const int64_t nb = k / TQK_BLOCK_SIZE;

    for (int64_t i = 0; i < nb; i++) {
        const float * xb = x + i * TQK_BLOCK_SIZE;

        // L2 norm
        float sum_sq = 0.0f;
        for (int j = 0; j < TQ_DIM; j++) sum_sq += xb[j] * xb[j];
        float norm = sqrtf(sum_sq);
        y[i].norm = GGML_FP32_TO_FP16(norm);
        y[i].rnorm = GGML_FP32_TO_FP16(0.0f);
        memset(y[i].signs, 0, sizeof(y[i].signs));

        if (norm == 0.0f) { memset(y[i].qs, 0, sizeof(y[i].qs)); continue; }
        float inv = 1.0f / norm;

        // Normalize and apply H_128 via FWHT
        float rot[TQ_DIM];
        for (int j = 0; j < TQ_DIM; j++) rot[j] = xb[j] * inv;
        tq_fwht(rot, TQ_DIM);

        // 3-bit MSE quantize (8 centroids)
        memset(y[i].qs, 0, sizeof(y[i].qs));
        for (int j = 0; j < TQ_DIM; j++) {
            int idx = nearest(rot[j], centroids_8, 8);
            pk3(y[i].qs, j, idx);
        }

        // Compute residual in ORIGINAL space: r = x - norm * H^{-1} * centroids
        float recon[TQ_DIM];
        for (int j = 0; j < TQ_DIM; j++) recon[j] = centroids_8[up3(y[i].qs, j)];
        tq_fwht(recon, TQ_DIM);  // inverse FWHT = FWHT (orthogonal, self-inverse)
        float resid[TQ_DIM];
        for (int j = 0; j < TQ_DIM; j++) resid[j] = xb[j] - norm * recon[j];

        // QJL forward on residual
        y[i].rnorm = GGML_FP32_TO_FP16(qjl_forward(resid, y[i].signs, TQ_DIM, QJL_SEED_128));
    }
}

void dequantize_row_tqk_had_prod4(const block_tqk_had_prod4 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % TQK_BLOCK_SIZE == 0);
    const int64_t nb = k / TQK_BLOCK_SIZE;

    for (int64_t i = 0; i < nb; i++) {
        float norm = GGML_FP16_TO_FP32(x[i].norm);

        // MSE reconstruction: centroids → inverse FWHT → scale by norm
        float rot[TQ_DIM];
        for (int j = 0; j < TQ_DIM; j++) {
            rot[j] = centroids_8[up3(x[i].qs, j)];
        }
        tq_fwht(rot, TQ_DIM);
        for (int j = 0; j < TQ_DIM; j++) y[i * TQK_BLOCK_SIZE + j] = norm * rot[j];

        // QJL correction on residual
        float corr[TQ_DIM];
        qjl_inverse(x[i].signs, GGML_FP16_TO_FP32(x[i].rnorm), corr, TQ_DIM, QJL_SEED_128);
        for (int j = 0; j < TQ_DIM; j++) y[i * TQK_BLOCK_SIZE + j] += corr[j];
    }
}

// Asymmetric vec_dot: rotate Q with H_128, dot with stored centroids + QJL correction
void ggml_vec_dot_tqk_had_prod4_f32(
        int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, size_t bx,
        const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % TQK_BLOCK_SIZE == 0);
    assert(nrc == 1);
    (void)bs; (void)bx; (void)by; (void)nrc;

    const block_tqk_had_prod4 * GGML_RESTRICT x = (const block_tqk_had_prod4 *)vx;
    const float * GGML_RESTRICT y = (const float *)vy;
    const int64_t nb = n / TQK_BLOCK_SIZE;

    float sumf = 0.0f;

    for (int64_t i = 0; i < nb; i++) {
        const float * q = y + i * TQK_BLOCK_SIZE;

        float norm = GGML_FP16_TO_FP32(x[i].norm);

        // Rotate Q with H_128
        float q_rot[TQ_DIM];
        for (int j = 0; j < TQ_DIM; j++) q_rot[j] = q[j];
        tq_fwht(q_rot, TQ_DIM);

        // MSE centroid dot (3-bit, 8 centroids)
        float dot = 0.0f;
        for (int j = 0; j < TQ_DIM; j++) {
            dot += q_rot[j] * centroids_8[up3(x[i].qs, j)];
        }
        float mse_dot = norm * dot;

        // QJL correction: q_rot is already FWHT(q), which is H·q — exact match for Hadamard QJL
        float qjl_dot = qjl_asymmetric_dot_rotated(q_rot, TQ_DIM,
                                            x[i].signs, GGML_FP16_TO_FP32(x[i].rnorm));

        sumf += mse_dot + qjl_dot;
    }

    *s = sumf;
}

// ---------------------------------------------------------------------------
// TQK 5hi_3lo_qr: 32/96 split, QR rotation, 4-bit MSE + QJL hi, 3-bit MSE lo
// ---------------------------------------------------------------------------

void quantize_row_tqk_5hi_3lo_qr_ref(const float * GGML_RESTRICT x, block_tqk_5hi_3lo * GGML_RESTRICT y, int64_t k) {
    assert(k % TQK_BLOCK_SIZE == 0);
    const int64_t nb = k / TQK_BLOCK_SIZE;
    tq_init_rotations();

    for (int64_t i = 0; i < nb; i++) {
        if (nb > 1) tq_cur_head = (int)(i % (tq_n_heads > 0 ? tq_n_heads : nb));
        const float * xb = x + i * TQK_BLOCK_SIZE;

        // Split into outlier/regular
        float hi_raw[TQ_DIM_HI], lo_raw[TQ_DIM_LO];
        tq_split_channels(xb, hi_raw, lo_raw);

        // QR rotation
        float hi_rot[TQ_DIM_HI], lo_rot[TQ_DIM_LO];
        tq_rotate_hi(hi_raw, hi_rot);
        tq_rotate_lo(lo_raw, lo_rot);

        // Per-subset norms
        float sum_hi = 0.0f, sum_lo = 0.0f;
        for (int j = 0; j < TQ_DIM_HI; j++) sum_hi += hi_rot[j] * hi_rot[j];
        for (int j = 0; j < TQ_DIM_LO; j++) sum_lo += lo_rot[j] * lo_rot[j];
        float norm_hi = sqrtf(sum_hi), norm_lo = sqrtf(sum_lo);

        y[i].norm_hi  = GGML_FP32_TO_FP16(norm_hi);
        y[i].norm_lo  = GGML_FP32_TO_FP16(norm_lo);
        y[i].rnorm_hi = GGML_FP32_TO_FP16(0.0f);
        memset(y[i].signs_hi, 0, sizeof(y[i].signs_hi));

        if (norm_hi == 0.0f && norm_lo == 0.0f) {
            memset(y[i].qs_hi, 0, sizeof(y[i].qs_hi));
            memset(y[i].qs_lo, 0, sizeof(y[i].qs_lo));
            continue;
        }

        float inv_hi = (norm_hi > 1e-12f) ? 1.0f / norm_hi : 0.0f;
        float inv_lo = (norm_lo > 1e-12f) ? 1.0f / norm_lo : 0.0f;

        // 4-bit MSE for hi (16 centroids, d=32)
        memset(y[i].qs_hi, 0, sizeof(y[i].qs_hi));
        for (int j = 0; j < TQ_DIM_HI; j++) {
            float xn = hi_rot[j] * inv_hi;
            int idx = nearest(xn, centroids_16_d32, 16);
            pk4(y[i].qs_hi, j, idx);
        }

        // 3-bit MSE for lo (8 centroids, d=96)
        memset(y[i].qs_lo, 0, sizeof(y[i].qs_lo));
        for (int j = 0; j < TQ_DIM_LO; j++) {
            float xn = lo_rot[j] * inv_lo;
            int idx = nearest(xn, centroids_8_d96, 8);
            pk3(y[i].qs_lo, j, idx);
        }

        // QJL on hi residual (in original subset space)
        float yhi[TQ_DIM_HI];
        for (int j = 0; j < TQ_DIM_HI; j++) yhi[j] = centroids_16_d32[up4(y[i].qs_hi, j)];
        float hi_rec[TQ_DIM_HI];
        tq_unrotate_hi(yhi, hi_rec);
        float r_hi[TQ_DIM_HI];
        for (int j = 0; j < TQ_DIM_HI; j++) r_hi[j] = hi_raw[j] - norm_hi * hi_rec[j];
        y[i].rnorm_hi = GGML_FP32_TO_FP16(qjl_forward(r_hi, y[i].signs_hi, TQ_DIM_HI, QJL_SEED_32));
    }
}

void dequantize_row_tqk_5hi_3lo_qr(const block_tqk_5hi_3lo * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % TQK_BLOCK_SIZE == 0);
    const int64_t nb = k / TQK_BLOCK_SIZE;
    tq_init_rotations();

    for (int64_t i = 0; i < nb; i++) {
        if (nb > 1) tq_cur_head = (int)(i % (tq_n_heads > 0 ? tq_n_heads : nb));
        float norm_hi = GGML_FP16_TO_FP32(x[i].norm_hi);
        float norm_lo = GGML_FP16_TO_FP32(x[i].norm_lo);

        // MSE recon: centroids → unrotate → scale
        float yhi[TQ_DIM_HI], ylo[TQ_DIM_LO];
        for (int j = 0; j < TQ_DIM_HI; j++) yhi[j] = centroids_16_d32[up4(x[i].qs_hi, j)];
        for (int j = 0; j < TQ_DIM_LO; j++) ylo[j] = centroids_8_d96[up3(x[i].qs_lo, j)];

        float hi_orig[TQ_DIM_HI], lo_orig[TQ_DIM_LO];
        tq_unrotate_hi(yhi, hi_orig);
        tq_unrotate_lo(ylo, lo_orig);
        for (int j = 0; j < TQ_DIM_HI; j++) hi_orig[j] *= norm_hi;
        for (int j = 0; j < TQ_DIM_LO; j++) lo_orig[j] *= norm_lo;

        // QJL correction on hi
        float corr_hi[TQ_DIM_HI];
        qjl_inverse(x[i].signs_hi, GGML_FP16_TO_FP32(x[i].rnorm_hi), corr_hi, TQ_DIM_HI, QJL_SEED_32);
        for (int j = 0; j < TQ_DIM_HI; j++) hi_orig[j] += corr_hi[j];

        tq_merge_channels(hi_orig, lo_orig, y + i * TQK_BLOCK_SIZE);
    }
}

void ggml_vec_dot_tqk_5hi_3lo_qr_f32(
        int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, size_t bx,
        const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % TQK_BLOCK_SIZE == 0);
    assert(nrc == 1);
    (void)bs; (void)bx; (void)by; (void)nrc;

    tq_init_rotations();

    const block_tqk_5hi_3lo * GGML_RESTRICT x = (const block_tqk_5hi_3lo *)vx;
    const float * GGML_RESTRICT y = (const float *)vy;
    const int64_t nb = n / TQK_BLOCK_SIZE;

    float sumf = 0.0f;

    for (int64_t i = 0; i < nb; i++) {
        const float * q = y + i * TQK_BLOCK_SIZE;

        if (nb > 1) tq_cur_head = (int)(i % (tq_n_heads > 0 ? tq_n_heads : nb));
        float norm_hi = GGML_FP16_TO_FP32(x[i].norm_hi);
        float norm_lo = GGML_FP16_TO_FP32(x[i].norm_lo);

        // Split query and QR-rotate
        float hi_raw[TQ_DIM_HI], lo_raw[TQ_DIM_LO];
        tq_split_channels(q, hi_raw, lo_raw);
        float q_rot_hi[TQ_DIM_HI], q_rot_lo[TQ_DIM_LO];
        tq_rotate_hi(hi_raw, q_rot_hi);
        tq_rotate_lo(lo_raw, q_rot_lo);

        // MSE centroid dot per subset
        float mse_dot_hi = 0.0f, mse_dot_lo = 0.0f;
        for (int j = 0; j < TQ_DIM_HI; j++) {
            mse_dot_hi += q_rot_hi[j] * centroids_16_d32[up4(x[i].qs_hi, j)];
        }
        for (int j = 0; j < TQ_DIM_LO; j++) {
            mse_dot_lo += q_rot_lo[j] * centroids_8_d96[up3(x[i].qs_lo, j)];
        }
        float mse_dot = mse_dot_hi * norm_hi + mse_dot_lo * norm_lo;

        // QJL correction on hi (raw query, not rotated)
        float qjl_hi = qjl_asymmetric_dot(hi_raw, TQ_DIM_HI, QJL_SEED_32,
                                           x[i].signs_hi, GGML_FP16_TO_FP32(x[i].rnorm_hi));

        sumf += mse_dot + qjl_hi;
    }

    *s = sumf;
}

// ---------------------------------------------------------------------------
// TQK 5hi_3lo_had: 32/96 split, FWHT rotation, 4-bit MSE + QJL hi, 3-bit MSE lo
// Hi uses H_32 FWHT. Lo uses 3 × H_32 FWHT (block-diagonal on 96=3×32).
// Lo centroids use d=32 (each 32-dim block is independent).
// ---------------------------------------------------------------------------

void quantize_row_tqk_5hi_3lo_had_ref(const float * GGML_RESTRICT x, block_tqk_5hi_3lo * GGML_RESTRICT y, int64_t k) {
    assert(k % TQK_BLOCK_SIZE == 0);
    const int64_t nb = k / TQK_BLOCK_SIZE;
    tq_init_rotations();

    for (int64_t i = 0; i < nb; i++) {
        if (nb > 1) tq_cur_head = (int)(i % (tq_n_heads > 0 ? tq_n_heads : nb));
        const float * xb = x + i * TQK_BLOCK_SIZE;

        // Split into outlier/regular
        float hi_raw[TQ_DIM_HI], lo_raw[TQ_DIM_LO];
        tq_split_channels(xb, hi_raw, lo_raw);

        // FWHT rotation: H_32 on hi, 3×H_32 on lo
        float hi_rot[TQ_DIM_HI], lo_rot[TQ_DIM_LO];
        memcpy(hi_rot, hi_raw, sizeof(hi_rot));
        tq_fwht(hi_rot, TQ_DIM_HI);
        memcpy(lo_rot, lo_raw, sizeof(lo_rot));
        tq_fwht(lo_rot,       TQ_DIM_HI);  // lo[0:32]
        tq_fwht(lo_rot + 32,  TQ_DIM_HI);  // lo[32:64]
        tq_fwht(lo_rot + 64,  TQ_DIM_HI);  // lo[64:96]

        // Per-subset norms
        float sum_hi = 0.0f, sum_lo = 0.0f;
        for (int j = 0; j < TQ_DIM_HI; j++) sum_hi += hi_rot[j] * hi_rot[j];
        for (int j = 0; j < TQ_DIM_LO; j++) sum_lo += lo_rot[j] * lo_rot[j];
        float norm_hi = sqrtf(sum_hi), norm_lo = sqrtf(sum_lo);

        y[i].norm_hi  = GGML_FP32_TO_FP16(norm_hi);
        y[i].norm_lo  = GGML_FP32_TO_FP16(norm_lo);
        y[i].rnorm_hi = GGML_FP32_TO_FP16(0.0f);
        memset(y[i].signs_hi, 0, sizeof(y[i].signs_hi));

        if (norm_hi == 0.0f && norm_lo == 0.0f) {
            memset(y[i].qs_hi, 0, sizeof(y[i].qs_hi));
            memset(y[i].qs_lo, 0, sizeof(y[i].qs_lo));
            continue;
        }

        float inv_hi = (norm_hi > 1e-12f) ? 1.0f / norm_hi : 0.0f;
        float inv_lo = (norm_lo > 1e-12f) ? 1.0f / norm_lo : 0.0f;

        // 4-bit MSE for hi (16 centroids, d=32)
        memset(y[i].qs_hi, 0, sizeof(y[i].qs_hi));
        for (int j = 0; j < TQ_DIM_HI; j++) {
            float xn = hi_rot[j] * inv_hi;
            int idx = nearest(xn, centroids_16_d32, 16);
            pk4(y[i].qs_hi, j, idx);
        }

        // 3-bit MSE for lo (8 centroids, d=96 — norm_lo spans full 96-dim vector)
        memset(y[i].qs_lo, 0, sizeof(y[i].qs_lo));
        for (int j = 0; j < TQ_DIM_LO; j++) {
            float xn = lo_rot[j] * inv_lo;
            int idx = nearest(xn, centroids_8_d96, 8);
            pk3(y[i].qs_lo, j, idx);
        }

        // QJL on hi residual (in original subset space)
        // Reconstruct: unrotate centroids via inverse FWHT
        float yhi[TQ_DIM_HI];
        for (int j = 0; j < TQ_DIM_HI; j++) yhi[j] = centroids_16_d32[up4(y[i].qs_hi, j)];
        float hi_rec[TQ_DIM_HI];
        memcpy(hi_rec, yhi, sizeof(hi_rec));
        tq_fwht(hi_rec, TQ_DIM_HI); // inverse = forward for normalized FWHT
        float r_hi[TQ_DIM_HI];
        for (int j = 0; j < TQ_DIM_HI; j++) r_hi[j] = hi_raw[j] - norm_hi * hi_rec[j];
        y[i].rnorm_hi = GGML_FP32_TO_FP16(qjl_forward(r_hi, y[i].signs_hi, TQ_DIM_HI, QJL_SEED_32));
    }
}

void dequantize_row_tqk_5hi_3lo_had(const block_tqk_5hi_3lo * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % TQK_BLOCK_SIZE == 0);
    const int64_t nb = k / TQK_BLOCK_SIZE;
    tq_init_rotations();

    for (int64_t i = 0; i < nb; i++) {
        if (nb > 1) tq_cur_head = (int)(i % (tq_n_heads > 0 ? tq_n_heads : nb));
        float norm_hi = GGML_FP16_TO_FP32(x[i].norm_hi);
        float norm_lo = GGML_FP16_TO_FP32(x[i].norm_lo);

        // MSE recon: centroids → inverse FWHT → scale
        float yhi[TQ_DIM_HI], ylo[TQ_DIM_LO];
        for (int j = 0; j < TQ_DIM_HI; j++) yhi[j] = centroids_16_d32[up4(x[i].qs_hi, j)];
        for (int j = 0; j < TQ_DIM_LO; j++) ylo[j] = centroids_8_d96[up3(x[i].qs_lo, j)];

        // Inverse FWHT
        float hi_orig[TQ_DIM_HI], lo_orig[TQ_DIM_LO];
        memcpy(hi_orig, yhi, sizeof(hi_orig));
        tq_fwht(hi_orig, TQ_DIM_HI);
        memcpy(lo_orig, ylo, sizeof(lo_orig));
        tq_fwht(lo_orig,       TQ_DIM_HI);
        tq_fwht(lo_orig + 32,  TQ_DIM_HI);
        tq_fwht(lo_orig + 64,  TQ_DIM_HI);

        for (int j = 0; j < TQ_DIM_HI; j++) hi_orig[j] *= norm_hi;
        for (int j = 0; j < TQ_DIM_LO; j++) lo_orig[j] *= norm_lo;

        // QJL correction on hi
        float corr_hi[TQ_DIM_HI];
        qjl_inverse(x[i].signs_hi, GGML_FP16_TO_FP32(x[i].rnorm_hi), corr_hi, TQ_DIM_HI, QJL_SEED_32);
        for (int j = 0; j < TQ_DIM_HI; j++) hi_orig[j] += corr_hi[j];

        tq_merge_channels(hi_orig, lo_orig, y + i * TQK_BLOCK_SIZE);
    }
}

// Asymmetric vec_dot for 5hi_3lo_had
void ggml_vec_dot_tqk_5hi_3lo_had_f32(
        int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, size_t bx,
        const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % TQK_BLOCK_SIZE == 0);
    assert(nrc == 1);
    (void)bs; (void)bx; (void)by; (void)nrc;

    tq_init_rotations();

    const block_tqk_5hi_3lo * GGML_RESTRICT x = (const block_tqk_5hi_3lo *)vx;
    const float * GGML_RESTRICT y = (const float *)vy;
    const int64_t nb = n / TQK_BLOCK_SIZE;

    float sumf = 0.0f;

    for (int64_t i = 0; i < nb; i++) {
        const float * q = y + i * TQK_BLOCK_SIZE;

        if (nb > 1) tq_cur_head = (int)(i % (tq_n_heads > 0 ? tq_n_heads : nb));
        float norm_hi = GGML_FP16_TO_FP32(x[i].norm_hi);
        float norm_lo = GGML_FP16_TO_FP32(x[i].norm_lo);

        // Split query and FWHT-rotate
        float hi_raw[TQ_DIM_HI], lo_raw[TQ_DIM_LO];
        tq_split_channels(q, hi_raw, lo_raw);

        float q_rot_hi[TQ_DIM_HI], q_rot_lo[TQ_DIM_LO];
        memcpy(q_rot_hi, hi_raw, sizeof(q_rot_hi));
        tq_fwht(q_rot_hi, TQ_DIM_HI);
        memcpy(q_rot_lo, lo_raw, sizeof(q_rot_lo));
        tq_fwht(q_rot_lo,       TQ_DIM_HI);
        tq_fwht(q_rot_lo + 32,  TQ_DIM_HI);
        tq_fwht(q_rot_lo + 64,  TQ_DIM_HI);

        // MSE centroid dot per subset
        float mse_dot_hi = 0.0f, mse_dot_lo = 0.0f;
        for (int j = 0; j < TQ_DIM_HI; j++) {
            mse_dot_hi += q_rot_hi[j] * centroids_16_d32[up4(x[i].qs_hi, j)];
        }
        for (int j = 0; j < TQ_DIM_LO; j++) {
            mse_dot_lo += q_rot_lo[j] * centroids_8_d96[up3(x[i].qs_lo, j)];
        }
        float mse_dot = mse_dot_hi * norm_hi + mse_dot_lo * norm_lo;

        // QJL correction on hi (raw query, not rotated)
        float qjl_hi = qjl_asymmetric_dot(hi_raw, TQ_DIM_HI, QJL_SEED_32,
                                           x[i].signs_hi, GGML_FP16_TO_FP32(x[i].rnorm_hi));

        sumf += mse_dot + qjl_hi;
    }

    *s = sumf;
}

// ---------------------------------------------------------------------------
// TQK 6hi_3lo_had: like 5hi_3lo but 5-bit MSE (32 centroids) on outliers
// ---------------------------------------------------------------------------

void quantize_row_tqk_6hi_3lo_had_ref(const float * GGML_RESTRICT x, block_tqk_6hi_3lo * GGML_RESTRICT y, int64_t k) {
    assert(k % TQK_BLOCK_SIZE == 0);
    const int64_t nb = k / TQK_BLOCK_SIZE;
    tq_init_rotations();
    for (int64_t i = 0; i < nb; i++) {
        if (nb > 1) tq_cur_head = (int)(i % (tq_n_heads > 0 ? tq_n_heads : nb));
        const float * xb = x + i * TQK_BLOCK_SIZE;
        float hi_raw[TQ_DIM_HI], lo_raw[TQ_DIM_LO];
        tq_split_channels(xb, hi_raw, lo_raw);
        float hi_rot[TQ_DIM_HI], lo_rot[TQ_DIM_LO];
        memcpy(hi_rot, hi_raw, sizeof(hi_rot));
        tq_fwht(hi_rot, TQ_DIM_HI);
        memcpy(lo_rot, lo_raw, sizeof(lo_rot));
        tq_fwht(lo_rot, TQ_DIM_HI); tq_fwht(lo_rot+32, TQ_DIM_HI); tq_fwht(lo_rot+64, TQ_DIM_HI);
        float sum_hi = 0, sum_lo = 0;
        for (int j = 0; j < TQ_DIM_HI; j++) sum_hi += hi_rot[j]*hi_rot[j];
        for (int j = 0; j < TQ_DIM_LO; j++) sum_lo += lo_rot[j]*lo_rot[j];
        float norm_hi = sqrtf(sum_hi), norm_lo = sqrtf(sum_lo);
        y[i].norm_hi = GGML_FP32_TO_FP16(norm_hi);
        y[i].norm_lo = GGML_FP32_TO_FP16(norm_lo);
        y[i].rnorm_hi = GGML_FP32_TO_FP16(0.0f);
        memset(y[i].signs_hi, 0, sizeof(y[i].signs_hi));
        if (norm_hi == 0 && norm_lo == 0) { memset(y[i].qs_hi,0,sizeof(y[i].qs_hi)); memset(y[i].qs_lo,0,sizeof(y[i].qs_lo)); continue; }
        float inv_hi = (norm_hi > 1e-12f) ? 1.0f/norm_hi : 0, inv_lo = (norm_lo > 1e-12f) ? 1.0f/norm_lo : 0;
        memset(y[i].qs_hi, 0, sizeof(y[i].qs_hi));
        for (int j = 0; j < TQ_DIM_HI; j++) pk5(y[i].qs_hi, j, nearest(hi_rot[j]*inv_hi, centroids_32_d32, 32));
        memset(y[i].qs_lo, 0, sizeof(y[i].qs_lo));
        for (int j = 0; j < TQ_DIM_LO; j++) pk3(y[i].qs_lo, j, nearest(lo_rot[j]*inv_lo, centroids_8_d96, 8));
        float yhi[TQ_DIM_HI];
        for (int j = 0; j < TQ_DIM_HI; j++) yhi[j] = centroids_32_d32[up5(y[i].qs_hi, j)];
        float hi_rec[TQ_DIM_HI]; memcpy(hi_rec, yhi, sizeof(hi_rec)); tq_fwht(hi_rec, TQ_DIM_HI);
        float r_hi[TQ_DIM_HI];
        for (int j = 0; j < TQ_DIM_HI; j++) r_hi[j] = hi_raw[j] - norm_hi*hi_rec[j];
        y[i].rnorm_hi = GGML_FP32_TO_FP16(qjl_forward(r_hi, y[i].signs_hi, TQ_DIM_HI, QJL_SEED_32));
    }
}

void dequantize_row_tqk_6hi_3lo_had(const block_tqk_6hi_3lo * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % TQK_BLOCK_SIZE == 0);
    const int64_t nb = k / TQK_BLOCK_SIZE;
    tq_init_rotations();
    for (int64_t i = 0; i < nb; i++) {
        if (nb > 1) tq_cur_head = (int)(i % (tq_n_heads > 0 ? tq_n_heads : nb));
        float norm_hi = GGML_FP16_TO_FP32(x[i].norm_hi), norm_lo = GGML_FP16_TO_FP32(x[i].norm_lo);
        float yhi[TQ_DIM_HI], ylo[TQ_DIM_LO];
        for (int j = 0; j < TQ_DIM_HI; j++) yhi[j] = centroids_32_d32[up5(x[i].qs_hi, j)];
        for (int j = 0; j < TQ_DIM_LO; j++) ylo[j] = centroids_8_d96[up3(x[i].qs_lo, j)];
        float hi_orig[TQ_DIM_HI], lo_orig[TQ_DIM_LO];
        memcpy(hi_orig, yhi, sizeof(hi_orig)); tq_fwht(hi_orig, TQ_DIM_HI);
        memcpy(lo_orig, ylo, sizeof(lo_orig));
        tq_fwht(lo_orig, TQ_DIM_HI); tq_fwht(lo_orig+32, TQ_DIM_HI); tq_fwht(lo_orig+64, TQ_DIM_HI);
        for (int j = 0; j < TQ_DIM_HI; j++) hi_orig[j] *= norm_hi;
        for (int j = 0; j < TQ_DIM_LO; j++) lo_orig[j] *= norm_lo;
        float corr_hi[TQ_DIM_HI];
        qjl_inverse(x[i].signs_hi, GGML_FP16_TO_FP32(x[i].rnorm_hi), corr_hi, TQ_DIM_HI, QJL_SEED_32);
        for (int j = 0; j < TQ_DIM_HI; j++) hi_orig[j] += corr_hi[j];
        tq_merge_channels(hi_orig, lo_orig, y + i*TQK_BLOCK_SIZE);
    }
}

void ggml_vec_dot_tqk_6hi_3lo_had_f32(int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % TQK_BLOCK_SIZE == 0); assert(nrc == 1);
    (void)bs; (void)bx; (void)by; (void)nrc;
    tq_init_rotations();
    const block_tqk_6hi_3lo * GGML_RESTRICT x = (const block_tqk_6hi_3lo *)vx;
    const float * GGML_RESTRICT y = (const float *)vy;
    const int64_t nb = n / TQK_BLOCK_SIZE;
    float sumf = 0;
    for (int64_t i = 0; i < nb; i++) {
        const float * q = y + i*TQK_BLOCK_SIZE;
        if (nb > 1) tq_cur_head = (int)(i % (tq_n_heads > 0 ? tq_n_heads : nb));
        float norm_hi = GGML_FP16_TO_FP32(x[i].norm_hi), norm_lo = GGML_FP16_TO_FP32(x[i].norm_lo);
        float hi_raw[TQ_DIM_HI], lo_raw[TQ_DIM_LO];
        tq_split_channels(q, hi_raw, lo_raw);
        float q_rot_hi[TQ_DIM_HI], q_rot_lo[TQ_DIM_LO];
        memcpy(q_rot_hi, hi_raw, sizeof(q_rot_hi)); tq_fwht(q_rot_hi, TQ_DIM_HI);
        memcpy(q_rot_lo, lo_raw, sizeof(q_rot_lo));
        tq_fwht(q_rot_lo, TQ_DIM_HI); tq_fwht(q_rot_lo+32, TQ_DIM_HI); tq_fwht(q_rot_lo+64, TQ_DIM_HI);
        float mse_dot_hi = 0, mse_dot_lo = 0;
        for (int j = 0; j < TQ_DIM_HI; j++) mse_dot_hi += q_rot_hi[j]*centroids_32_d32[up5(x[i].qs_hi, j)];
        for (int j = 0; j < TQ_DIM_LO; j++) mse_dot_lo += q_rot_lo[j]*centroids_8_d96[up3(x[i].qs_lo, j)];
        float qjl_hi = qjl_asymmetric_dot(hi_raw, TQ_DIM_HI, QJL_SEED_32, x[i].signs_hi, GGML_FP16_TO_FP32(x[i].rnorm_hi));
        sumf += mse_dot_hi*norm_hi + mse_dot_lo*norm_lo + qjl_hi;
    }
    *s = sumf;
}

// ---------------------------------------------------------------------------
// TQK 2hi_1lo_had: 2-bit MSE + QJL on outliers, 1-bit MSE + QJL on regulars (2.75 bpv)
// ---------------------------------------------------------------------------


void quantize_row_tqk_2hi_1lo_had_ref(const float * GGML_RESTRICT x, block_tqk_2hi_1lo * GGML_RESTRICT y, int64_t k) {
    assert(k % TQK_BLOCK_SIZE == 0);
    const int64_t nb = k / TQK_BLOCK_SIZE;
    tq_init_rotations();
    for (int64_t i = 0; i < nb; i++) {
        if (nb > 1) tq_cur_head = (int)(i % (tq_n_heads > 0 ? tq_n_heads : nb));
        const float * xb = x + i * TQK_BLOCK_SIZE;
        float hi_raw[TQ_DIM_HI], lo_raw[TQ_DIM_LO];
        tq_split_channels(xb, hi_raw, lo_raw);
        float hi_rot[TQ_DIM_HI], lo_rot[TQ_DIM_LO];
        memcpy(hi_rot, hi_raw, sizeof(hi_rot));
        tq_fwht(hi_rot, TQ_DIM_HI);
        memcpy(lo_rot, lo_raw, sizeof(lo_rot));
        tq_fwht(lo_rot, TQ_DIM_HI); tq_fwht(lo_rot+32, TQ_DIM_HI); tq_fwht(lo_rot+64, TQ_DIM_HI);
        float sum_hi = 0, sum_lo = 0;
        for (int j = 0; j < TQ_DIM_HI; j++) sum_hi += hi_rot[j]*hi_rot[j];
        for (int j = 0; j < TQ_DIM_LO; j++) sum_lo += lo_rot[j]*lo_rot[j];
        float norm_hi = sqrtf(sum_hi), norm_lo = sqrtf(sum_lo);
        y[i].norm_hi = GGML_FP32_TO_FP16(norm_hi); y[i].norm_lo = GGML_FP32_TO_FP16(norm_lo);
        y[i].rnorm_hi = GGML_FP32_TO_FP16(0); y[i].rnorm_lo = GGML_FP32_TO_FP16(0);
        memset(y[i].signs_hi, 0, sizeof(y[i].signs_hi));
        memset(y[i].signs_lo, 0, sizeof(y[i].signs_lo));
        if (norm_hi == 0 && norm_lo == 0) { memset(y[i].qs_hi,0,sizeof(y[i].qs_hi)); memset(y[i].qs_lo,0,sizeof(y[i].qs_lo)); continue; }
        float inv_hi = (norm_hi > 1e-12f) ? 1.0f/norm_hi : 0, inv_lo = (norm_lo > 1e-12f) ? 1.0f/norm_lo : 0;
        // 2-bit MSE on hi (4 centroids, d=32)
        memset(y[i].qs_hi, 0, sizeof(y[i].qs_hi));
        for (int j = 0; j < TQ_DIM_HI; j++) pk2(y[i].qs_hi, j, nearest(hi_rot[j]*inv_hi, centroids_4_d32, 4));
        // 1-bit MSE on lo (2 centroids, d=96)
        memset(y[i].qs_lo, 0, sizeof(y[i].qs_lo));
        for (int j = 0; j < TQ_DIM_LO; j++) pk1(y[i].qs_lo, j, nearest(lo_rot[j]*inv_lo, centroids_2_d96, 2));
        // QJL on hi residual
        float yhi[TQ_DIM_HI];
        for (int j = 0; j < TQ_DIM_HI; j++) yhi[j] = centroids_4_d32[up2(y[i].qs_hi, j)];
        float hi_rec[TQ_DIM_HI]; memcpy(hi_rec, yhi, sizeof(hi_rec)); tq_fwht(hi_rec, TQ_DIM_HI);
        float r_hi[TQ_DIM_HI]; for (int j = 0; j < TQ_DIM_HI; j++) r_hi[j] = hi_raw[j] - norm_hi*hi_rec[j];
        y[i].rnorm_hi = GGML_FP32_TO_FP16(qjl_forward(r_hi, y[i].signs_hi, TQ_DIM_HI, QJL_SEED_32));
        // QJL on lo residual
        float ylo[TQ_DIM_LO];
        for (int j = 0; j < TQ_DIM_LO; j++) ylo[j] = centroids_2_d96[up1(y[i].qs_lo, j)];
        float lo_rec[TQ_DIM_LO]; memcpy(lo_rec, ylo, sizeof(lo_rec));
        tq_fwht(lo_rec, TQ_DIM_HI); tq_fwht(lo_rec+32, TQ_DIM_HI); tq_fwht(lo_rec+64, TQ_DIM_HI);
        float r_lo[TQ_DIM_LO]; for (int j = 0; j < TQ_DIM_LO; j++) r_lo[j] = lo_raw[j] - norm_lo*lo_rec[j];
        y[i].rnorm_lo = GGML_FP32_TO_FP16(qjl_forward_block3(r_lo, y[i].signs_lo, TQ_DIM_HI));
    }
}

void dequantize_row_tqk_2hi_1lo_had(const block_tqk_2hi_1lo * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % TQK_BLOCK_SIZE == 0);
    const int64_t nb = k / TQK_BLOCK_SIZE;
    tq_init_rotations();
    for (int64_t i = 0; i < nb; i++) {
        if (nb > 1) tq_cur_head = (int)(i % (tq_n_heads > 0 ? tq_n_heads : nb));
        float norm_hi = GGML_FP16_TO_FP32(x[i].norm_hi), norm_lo = GGML_FP16_TO_FP32(x[i].norm_lo);
        float yhi[TQ_DIM_HI], ylo[TQ_DIM_LO];
        for (int j = 0; j < TQ_DIM_HI; j++) yhi[j] = centroids_4_d32[up2(x[i].qs_hi, j)];
        for (int j = 0; j < TQ_DIM_LO; j++) ylo[j] = centroids_2_d96[up1(x[i].qs_lo, j)];
        float hi_orig[TQ_DIM_HI], lo_orig[TQ_DIM_LO];
        memcpy(hi_orig, yhi, sizeof(hi_orig)); tq_fwht(hi_orig, TQ_DIM_HI);
        memcpy(lo_orig, ylo, sizeof(lo_orig));
        tq_fwht(lo_orig, TQ_DIM_HI); tq_fwht(lo_orig+32, TQ_DIM_HI); tq_fwht(lo_orig+64, TQ_DIM_HI);
        for (int j = 0; j < TQ_DIM_HI; j++) hi_orig[j] *= norm_hi;
        for (int j = 0; j < TQ_DIM_LO; j++) lo_orig[j] *= norm_lo;
        float corr_hi[TQ_DIM_HI];
        qjl_inverse(x[i].signs_hi, GGML_FP16_TO_FP32(x[i].rnorm_hi), corr_hi, TQ_DIM_HI, QJL_SEED_32);
        for (int j = 0; j < TQ_DIM_HI; j++) hi_orig[j] += corr_hi[j];
        float corr_lo[TQ_DIM_LO];
        qjl_inverse_block3(x[i].signs_lo, GGML_FP16_TO_FP32(x[i].rnorm_lo), corr_lo, TQ_DIM_HI);
        for (int j = 0; j < TQ_DIM_LO; j++) lo_orig[j] += corr_lo[j];
        tq_merge_channels(hi_orig, lo_orig, y + i*TQK_BLOCK_SIZE);
    }
}

void ggml_vec_dot_tqk_2hi_1lo_had_f32(int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % TQK_BLOCK_SIZE == 0); assert(nrc == 1);
    (void)bs; (void)bx; (void)by; (void)nrc;
    tq_init_rotations();
    const block_tqk_2hi_1lo * GGML_RESTRICT x = (const block_tqk_2hi_1lo *)vx;
    const float * GGML_RESTRICT y = (const float *)vy;
    const int64_t nb = n / TQK_BLOCK_SIZE;
    float sumf = 0;
    for (int64_t i = 0; i < nb; i++) {
        const float * q = y + i*TQK_BLOCK_SIZE;
        if (nb > 1) tq_cur_head = (int)(i % (tq_n_heads > 0 ? tq_n_heads : nb));
        float norm_hi = GGML_FP16_TO_FP32(x[i].norm_hi), norm_lo = GGML_FP16_TO_FP32(x[i].norm_lo);
        float hi_raw[TQ_DIM_HI], lo_raw[TQ_DIM_LO];
        tq_split_channels(q, hi_raw, lo_raw);
        float q_rot_hi[TQ_DIM_HI], q_rot_lo[TQ_DIM_LO];
        memcpy(q_rot_hi, hi_raw, sizeof(q_rot_hi)); tq_fwht(q_rot_hi, TQ_DIM_HI);
        memcpy(q_rot_lo, lo_raw, sizeof(q_rot_lo));
        tq_fwht(q_rot_lo, TQ_DIM_HI); tq_fwht(q_rot_lo+32, TQ_DIM_HI); tq_fwht(q_rot_lo+64, TQ_DIM_HI);
        float mse_dot_hi = 0, mse_dot_lo = 0;
        for (int j = 0; j < TQ_DIM_HI; j++) mse_dot_hi += q_rot_hi[j]*centroids_4_d32[up2(x[i].qs_hi, j)];
        for (int j = 0; j < TQ_DIM_LO; j++) mse_dot_lo += q_rot_lo[j]*centroids_2_d96[up1(x[i].qs_lo, j)];
        float qjl_hi = qjl_asymmetric_dot(hi_raw, TQ_DIM_HI, QJL_SEED_32, x[i].signs_hi, GGML_FP16_TO_FP32(x[i].rnorm_hi));
        float qjl_lo = qjl_asymmetric_dot_block3(lo_raw, TQ_DIM_HI, x[i].signs_lo, GGML_FP16_TO_FP32(x[i].rnorm_lo));
        sumf += mse_dot_hi*norm_hi + mse_dot_lo*norm_lo + qjl_hi + qjl_lo;
    }
    *s = sumf;
}

// ---------------------------------------------------------------------------
// TQK 3hi_2lo_had: 3-bit MSE + QJL on outliers, 2-bit MSE + QJL on regulars (3.75 bpv)
// ---------------------------------------------------------------------------

void quantize_row_tqk_3hi_2lo_had_ref(const float * GGML_RESTRICT x, block_tqk_3hi_2lo * GGML_RESTRICT y, int64_t k) {
    assert(k % TQK_BLOCK_SIZE == 0);
    const int64_t nb = k / TQK_BLOCK_SIZE;
    tq_init_rotations();
    for (int64_t i = 0; i < nb; i++) {
        if (nb > 1) tq_cur_head = (int)(i % (tq_n_heads > 0 ? tq_n_heads : nb));
        const float * xb = x + i * TQK_BLOCK_SIZE;
        float hi_raw[TQ_DIM_HI], lo_raw[TQ_DIM_LO];
        tq_split_channels(xb, hi_raw, lo_raw);
        float hi_rot[TQ_DIM_HI], lo_rot[TQ_DIM_LO];
        memcpy(hi_rot, hi_raw, sizeof(hi_rot));
        tq_fwht(hi_rot, TQ_DIM_HI);
        memcpy(lo_rot, lo_raw, sizeof(lo_rot));
        tq_fwht(lo_rot, TQ_DIM_HI); tq_fwht(lo_rot+32, TQ_DIM_HI); tq_fwht(lo_rot+64, TQ_DIM_HI);
        float sum_hi = 0, sum_lo = 0;
        for (int j = 0; j < TQ_DIM_HI; j++) sum_hi += hi_rot[j]*hi_rot[j];
        for (int j = 0; j < TQ_DIM_LO; j++) sum_lo += lo_rot[j]*lo_rot[j];
        float norm_hi = sqrtf(sum_hi), norm_lo = sqrtf(sum_lo);
        y[i].norm_hi = GGML_FP32_TO_FP16(norm_hi); y[i].norm_lo = GGML_FP32_TO_FP16(norm_lo);
        y[i].rnorm_hi = GGML_FP32_TO_FP16(0); y[i].rnorm_lo = GGML_FP32_TO_FP16(0);
        memset(y[i].signs_hi, 0, sizeof(y[i].signs_hi));
        memset(y[i].signs_lo, 0, sizeof(y[i].signs_lo));
        if (norm_hi == 0 && norm_lo == 0) { memset(y[i].qs_hi,0,sizeof(y[i].qs_hi)); memset(y[i].qs_lo,0,sizeof(y[i].qs_lo)); continue; }
        float inv_hi = (norm_hi > 1e-12f) ? 1.0f/norm_hi : 0, inv_lo = (norm_lo > 1e-12f) ? 1.0f/norm_lo : 0;
        // 3-bit MSE on hi (8 centroids, d=32)
        memset(y[i].qs_hi, 0, sizeof(y[i].qs_hi));
        for (int j = 0; j < TQ_DIM_HI; j++) pk3(y[i].qs_hi, j, nearest(hi_rot[j]*inv_hi, centroids_8_d32, 8));
        // 2-bit MSE on lo (4 centroids, d=96)
        memset(y[i].qs_lo, 0, sizeof(y[i].qs_lo));
        for (int j = 0; j < TQ_DIM_LO; j++) pk2(y[i].qs_lo, j, nearest(lo_rot[j]*inv_lo, centroids_4_d96, 4));
        // QJL on hi residual
        float yhi[TQ_DIM_HI];
        for (int j = 0; j < TQ_DIM_HI; j++) yhi[j] = centroids_8_d32[up3(y[i].qs_hi, j)];
        float hi_rec[TQ_DIM_HI]; memcpy(hi_rec, yhi, sizeof(hi_rec)); tq_fwht(hi_rec, TQ_DIM_HI);
        float r_hi[TQ_DIM_HI]; for (int j = 0; j < TQ_DIM_HI; j++) r_hi[j] = hi_raw[j] - norm_hi*hi_rec[j];
        y[i].rnorm_hi = GGML_FP32_TO_FP16(qjl_forward(r_hi, y[i].signs_hi, TQ_DIM_HI, QJL_SEED_32));
        // QJL on lo residual
        float ylo[TQ_DIM_LO];
        for (int j = 0; j < TQ_DIM_LO; j++) ylo[j] = centroids_4_d96[up2(y[i].qs_lo, j)];
        float lo_rec[TQ_DIM_LO]; memcpy(lo_rec, ylo, sizeof(lo_rec));
        tq_fwht(lo_rec, TQ_DIM_HI); tq_fwht(lo_rec+32, TQ_DIM_HI); tq_fwht(lo_rec+64, TQ_DIM_HI);
        float r_lo[TQ_DIM_LO]; for (int j = 0; j < TQ_DIM_LO; j++) r_lo[j] = lo_raw[j] - norm_lo*lo_rec[j];
        y[i].rnorm_lo = GGML_FP32_TO_FP16(qjl_forward_block3(r_lo, y[i].signs_lo, TQ_DIM_HI));
    }
}

void dequantize_row_tqk_3hi_2lo_had(const block_tqk_3hi_2lo * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % TQK_BLOCK_SIZE == 0);
    const int64_t nb = k / TQK_BLOCK_SIZE;
    tq_init_rotations();
    for (int64_t i = 0; i < nb; i++) {
        if (nb > 1) tq_cur_head = (int)(i % (tq_n_heads > 0 ? tq_n_heads : nb));
        float norm_hi = GGML_FP16_TO_FP32(x[i].norm_hi), norm_lo = GGML_FP16_TO_FP32(x[i].norm_lo);
        float yhi[TQ_DIM_HI], ylo[TQ_DIM_LO];
        for (int j = 0; j < TQ_DIM_HI; j++) yhi[j] = centroids_8_d32[up3(x[i].qs_hi, j)];
        for (int j = 0; j < TQ_DIM_LO; j++) ylo[j] = centroids_4_d96[up2(x[i].qs_lo, j)];
        float hi_orig[TQ_DIM_HI], lo_orig[TQ_DIM_LO];
        memcpy(hi_orig, yhi, sizeof(hi_orig)); tq_fwht(hi_orig, TQ_DIM_HI);
        memcpy(lo_orig, ylo, sizeof(lo_orig));
        tq_fwht(lo_orig, TQ_DIM_HI); tq_fwht(lo_orig+32, TQ_DIM_HI); tq_fwht(lo_orig+64, TQ_DIM_HI);
        for (int j = 0; j < TQ_DIM_HI; j++) hi_orig[j] *= norm_hi;
        for (int j = 0; j < TQ_DIM_LO; j++) lo_orig[j] *= norm_lo;
        float corr_hi[TQ_DIM_HI];
        qjl_inverse(x[i].signs_hi, GGML_FP16_TO_FP32(x[i].rnorm_hi), corr_hi, TQ_DIM_HI, QJL_SEED_32);
        for (int j = 0; j < TQ_DIM_HI; j++) hi_orig[j] += corr_hi[j];
        float corr_lo[TQ_DIM_LO];
        qjl_inverse_block3(x[i].signs_lo, GGML_FP16_TO_FP32(x[i].rnorm_lo), corr_lo, TQ_DIM_HI);
        for (int j = 0; j < TQ_DIM_LO; j++) lo_orig[j] += corr_lo[j];
        tq_merge_channels(hi_orig, lo_orig, y + i*TQK_BLOCK_SIZE);
    }
}

void ggml_vec_dot_tqk_3hi_2lo_had_f32(int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % TQK_BLOCK_SIZE == 0); assert(nrc == 1);
    (void)bs; (void)bx; (void)by; (void)nrc;
    tq_init_rotations();
    const block_tqk_3hi_2lo * GGML_RESTRICT x = (const block_tqk_3hi_2lo *)vx;
    const float * GGML_RESTRICT y = (const float *)vy;
    const int64_t nb = n / TQK_BLOCK_SIZE;
    float sumf = 0;
    for (int64_t i = 0; i < nb; i++) {
        const float * q = y + i*TQK_BLOCK_SIZE;
        if (nb > 1) tq_cur_head = (int)(i % (tq_n_heads > 0 ? tq_n_heads : nb));
        float norm_hi = GGML_FP16_TO_FP32(x[i].norm_hi), norm_lo = GGML_FP16_TO_FP32(x[i].norm_lo);
        float hi_raw[TQ_DIM_HI], lo_raw[TQ_DIM_LO];
        tq_split_channels(q, hi_raw, lo_raw);
        float q_rot_hi[TQ_DIM_HI], q_rot_lo[TQ_DIM_LO];
        memcpy(q_rot_hi, hi_raw, sizeof(q_rot_hi)); tq_fwht(q_rot_hi, TQ_DIM_HI);
        memcpy(q_rot_lo, lo_raw, sizeof(q_rot_lo));
        tq_fwht(q_rot_lo, TQ_DIM_HI); tq_fwht(q_rot_lo+32, TQ_DIM_HI); tq_fwht(q_rot_lo+64, TQ_DIM_HI);
        float mse_dot_hi = 0, mse_dot_lo = 0;
        for (int j = 0; j < TQ_DIM_HI; j++) mse_dot_hi += q_rot_hi[j]*centroids_8_d32[up3(x[i].qs_hi, j)];
        for (int j = 0; j < TQ_DIM_LO; j++) mse_dot_lo += q_rot_lo[j]*centroids_4_d96[up2(x[i].qs_lo, j)];
        float qjl_hi = qjl_asymmetric_dot(hi_raw, TQ_DIM_HI, QJL_SEED_32, x[i].signs_hi, GGML_FP16_TO_FP32(x[i].rnorm_hi));
        float qjl_lo = qjl_asymmetric_dot_block3(lo_raw, TQ_DIM_HI, x[i].signs_lo, GGML_FP16_TO_FP32(x[i].rnorm_lo));
        sumf += mse_dot_hi*norm_hi + mse_dot_lo*norm_lo + qjl_hi + qjl_lo;
    }
    *s = sumf;
}

// ===========================================================================
// d=256 TurboQuant CPU reference functions
// ===========================================================================

// ---------------------------------------------------------------------------
// TQK had_mse4_d256: H_256 Hadamard + 4-bit MSE, no split, no calibration
// ---------------------------------------------------------------------------

void quantize_row_tqk_had_mse4_d256_ref(const float * GGML_RESTRICT x, block_tqk_had_mse4_d256 * GGML_RESTRICT y, int64_t k) {
    assert(k % TQK_BLOCK_SIZE_D256 == 0);
    const int64_t nb = k / TQK_BLOCK_SIZE_D256;

    for (int64_t i = 0; i < nb; i++) {
        const float * xb = x + i * TQK_BLOCK_SIZE_D256;

        float sum_sq = 0.0f;
        for (int j = 0; j < TQK_BLOCK_SIZE_D256; j++) sum_sq += xb[j] * xb[j];
        float norm = sqrtf(sum_sq);
        y[i].norm = GGML_FP32_TO_FP16(norm);

        if (norm == 0.0f) { memset(y[i].qs, 0, sizeof(y[i].qs)); continue; }
        float inv = 1.0f / norm;

        float rot[TQK_BLOCK_SIZE_D256];
        for (int j = 0; j < TQK_BLOCK_SIZE_D256; j++) rot[j] = xb[j] * inv;
        tq_fwht(rot, TQK_BLOCK_SIZE_D256);

        memset(y[i].qs, 0, sizeof(y[i].qs));
        for (int j = 0; j < TQK_BLOCK_SIZE_D256; j++) {
            int idx = nearest(rot[j], centroids_16_d256, 16);
            pk4(y[i].qs, j, idx);
        }
    }
}

void dequantize_row_tqk_had_mse4_d256(const block_tqk_had_mse4_d256 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % TQK_BLOCK_SIZE_D256 == 0);
    const int64_t nb = k / TQK_BLOCK_SIZE_D256;

    for (int64_t i = 0; i < nb; i++) {
        float norm = GGML_FP16_TO_FP32(x[i].norm);
        float rot[TQK_BLOCK_SIZE_D256];
        for (int j = 0; j < TQK_BLOCK_SIZE_D256; j++) {
            rot[j] = centroids_16_d256[up4(x[i].qs, j)];
        }
        tq_fwht(rot, TQK_BLOCK_SIZE_D256);
        for (int j = 0; j < TQK_BLOCK_SIZE_D256; j++) y[i * TQK_BLOCK_SIZE_D256 + j] = norm * rot[j];
    }
}

// Asymmetric vec_dot: rotate Q with H_256, dot with stored centroids
void ggml_vec_dot_tqk_had_mse4_d256_f32(
        int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, size_t bx,
        const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % TQK_BLOCK_SIZE_D256 == 0);
    assert(nrc == 1);
    (void)bs; (void)bx; (void)by; (void)nrc;

    const block_tqk_had_mse4_d256 * GGML_RESTRICT x = (const block_tqk_had_mse4_d256 *)vx;
    const float * GGML_RESTRICT y = (const float *)vy;
    const int64_t nb = n / TQK_BLOCK_SIZE_D256;

    float sumf = 0.0f;

    for (int64_t i = 0; i < nb; i++) {
        const float * q = y + i * TQK_BLOCK_SIZE_D256;

        float norm = GGML_FP16_TO_FP32(x[i].norm);

        // Rotate Q with H_256
        float q_rot[TQK_BLOCK_SIZE_D256];
        for (int j = 0; j < TQK_BLOCK_SIZE_D256; j++) q_rot[j] = q[j];
        tq_fwht(q_rot, TQK_BLOCK_SIZE_D256);

        // Dot with stored centroids
        float dot = 0.0f;
        for (int j = 0; j < TQK_BLOCK_SIZE_D256; j++) {
            dot += q_rot[j] * centroids_16_d256[up4(x[i].qs, j)];
        }

        sumf += norm * dot;
    }

    *s = sumf;
}

// ---------------------------------------------------------------------------
// TQV had_mse4_d256: V cache 4-bit MSE, per-block norm, d=256
// ---------------------------------------------------------------------------

void quantize_row_tqv_had_mse4_d256_ref(const float * GGML_RESTRICT x, block_tqv_had_mse4_d256 * GGML_RESTRICT y, int64_t k) {
    quantize_row_tqk_had_mse4_d256_ref(x, (block_tqk_had_mse4_d256 *)y, k);
}

void dequantize_row_tqv_had_mse4_d256(const block_tqv_had_mse4_d256 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % TQK_BLOCK_SIZE_D256 == 0);
    const int64_t nb = k / TQK_BLOCK_SIZE_D256;

    for (int64_t i = 0; i < nb; i++) {
        float norm = GGML_FP16_TO_FP32(x[i].norm);
        float rot[TQK_BLOCK_SIZE_D256];
        for (int j = 0; j < TQK_BLOCK_SIZE_D256; j++) {
            rot[j] = centroids_16_d256[up4(x[i].qs, j)];
        }
        tq_fwht(rot, TQK_BLOCK_SIZE_D256); // inverse FWHT = FWHT (self-inverse)
        for (int j = 0; j < TQK_BLOCK_SIZE_D256; j++) y[i * TQK_BLOCK_SIZE_D256 + j] = norm * rot[j];
    }
}

// ---------------------------------------------------------------------------
// TQK had_prod5_d256: H_256 Hadamard + 4-bit MSE + 1-bit QJL (unbiased)
// ---------------------------------------------------------------------------

void quantize_row_tqk_had_prod5_d256_ref(const float * GGML_RESTRICT x, block_tqk_had_prod5_d256 * GGML_RESTRICT y, int64_t k) {
    assert(k % TQK_BLOCK_SIZE_D256 == 0);
    const int64_t nb = k / TQK_BLOCK_SIZE_D256;

    for (int64_t i = 0; i < nb; i++) {
        const float * xb = x + i * TQK_BLOCK_SIZE_D256;

        // L2 norm
        float sum_sq = 0.0f;
        for (int j = 0; j < TQK_BLOCK_SIZE_D256; j++) sum_sq += xb[j] * xb[j];
        float norm = sqrtf(sum_sq);
        y[i].norm = GGML_FP32_TO_FP16(norm);
        y[i].rnorm = GGML_FP32_TO_FP16(0.0f);
        memset(y[i].signs, 0, sizeof(y[i].signs));

        if (norm == 0.0f) { memset(y[i].qs, 0, sizeof(y[i].qs)); continue; }
        float inv = 1.0f / norm;

        // Normalize and apply H_256 via FWHT
        float rot[TQK_BLOCK_SIZE_D256];
        for (int j = 0; j < TQK_BLOCK_SIZE_D256; j++) rot[j] = xb[j] * inv;
        tq_fwht(rot, TQK_BLOCK_SIZE_D256);

        // 4-bit MSE quantize (16 centroids, d=256)
        memset(y[i].qs, 0, sizeof(y[i].qs));
        for (int j = 0; j < TQK_BLOCK_SIZE_D256; j++) {
            int idx = nearest(rot[j], centroids_16_d256, 16);
            pk4(y[i].qs, j, idx);
        }

        // Compute residual in ORIGINAL space: r = x - norm * H^{-1} * centroids
        float recon[TQK_BLOCK_SIZE_D256];
        for (int j = 0; j < TQK_BLOCK_SIZE_D256; j++) recon[j] = centroids_16_d256[up4(y[i].qs, j)];
        tq_fwht(recon, TQK_BLOCK_SIZE_D256);  // inverse FWHT = FWHT (self-inverse)
        float resid[TQK_BLOCK_SIZE_D256];
        for (int j = 0; j < TQK_BLOCK_SIZE_D256; j++) resid[j] = xb[j] - norm * recon[j];

        // QJL forward on residual
        y[i].rnorm = GGML_FP32_TO_FP16(qjl_forward(resid, y[i].signs, TQK_BLOCK_SIZE_D256, QJL_SEED_256));
    }
}

void dequantize_row_tqk_had_prod5_d256(const block_tqk_had_prod5_d256 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % TQK_BLOCK_SIZE_D256 == 0);
    const int64_t nb = k / TQK_BLOCK_SIZE_D256;

    for (int64_t i = 0; i < nb; i++) {
        float norm = GGML_FP16_TO_FP32(x[i].norm);

        // MSE reconstruction: centroids -> inverse FWHT -> scale by norm
        float rot[TQK_BLOCK_SIZE_D256];
        for (int j = 0; j < TQK_BLOCK_SIZE_D256; j++) {
            rot[j] = centroids_16_d256[up4(x[i].qs, j)];
        }
        tq_fwht(rot, TQK_BLOCK_SIZE_D256);
        for (int j = 0; j < TQK_BLOCK_SIZE_D256; j++) y[i * TQK_BLOCK_SIZE_D256 + j] = norm * rot[j];

        // QJL correction on residual
        float corr[TQK_BLOCK_SIZE_D256];
        qjl_inverse(x[i].signs, GGML_FP16_TO_FP32(x[i].rnorm), corr, TQK_BLOCK_SIZE_D256, QJL_SEED_256);
        for (int j = 0; j < TQK_BLOCK_SIZE_D256; j++) y[i * TQK_BLOCK_SIZE_D256 + j] += corr[j];
    }
}

// Asymmetric vec_dot: rotate Q with H_256, dot with stored centroids + QJL correction
void ggml_vec_dot_tqk_had_prod5_d256_f32(
        int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, size_t bx,
        const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % TQK_BLOCK_SIZE_D256 == 0);
    assert(nrc == 1);
    (void)bs; (void)bx; (void)by; (void)nrc;

    const block_tqk_had_prod5_d256 * GGML_RESTRICT x = (const block_tqk_had_prod5_d256 *)vx;
    const float * GGML_RESTRICT y = (const float *)vy;
    const int64_t nb = n / TQK_BLOCK_SIZE_D256;

    float sumf = 0.0f;

    for (int64_t i = 0; i < nb; i++) {
        const float * q = y + i * TQK_BLOCK_SIZE_D256;

        float norm = GGML_FP16_TO_FP32(x[i].norm);

        // Rotate Q with H_256
        float q_rot[TQK_BLOCK_SIZE_D256];
        for (int j = 0; j < TQK_BLOCK_SIZE_D256; j++) q_rot[j] = q[j];
        tq_fwht(q_rot, TQK_BLOCK_SIZE_D256);

        // MSE centroid dot
        float dot = 0.0f;
        for (int j = 0; j < TQK_BLOCK_SIZE_D256; j++) {
            dot += q_rot[j] * centroids_16_d256[up4(x[i].qs, j)];
        }
        float mse_dot = norm * dot;

        // QJL correction: q_rot is already FWHT(q), which is H*q
        float qjl_dot = qjl_asymmetric_dot_rotated(q_rot, TQK_BLOCK_SIZE_D256,
                                            x[i].signs, GGML_FP16_TO_FP32(x[i].rnorm));

        sumf += mse_dot + qjl_dot;
    }

    *s = sumf;
}

// ---------------------------------------------------------------------------
// TQK had_prod4_d256: H_256 Hadamard + 3-bit MSE + 1-bit QJL (4.13 bpv)
// ---------------------------------------------------------------------------

void quantize_row_tqk_had_prod4_d256_ref(const float * GGML_RESTRICT x, block_tqk_had_prod4_d256 * GGML_RESTRICT y, int64_t k) {
    assert(k % TQK_BLOCK_SIZE_D256 == 0);
    const int64_t nb = k / TQK_BLOCK_SIZE_D256;

    for (int64_t i = 0; i < nb; i++) {
        const float * xb = x + i * TQK_BLOCK_SIZE_D256;

        // L2 norm
        float sum_sq = 0.0f;
        for (int j = 0; j < TQK_BLOCK_SIZE_D256; j++) sum_sq += xb[j] * xb[j];
        float norm = sqrtf(sum_sq);
        y[i].norm = GGML_FP32_TO_FP16(norm);
        y[i].rnorm = GGML_FP32_TO_FP16(0.0f);
        memset(y[i].signs, 0, sizeof(y[i].signs));

        if (norm == 0.0f) { memset(y[i].qs, 0, sizeof(y[i].qs)); continue; }
        float inv = 1.0f / norm;

        // Normalize and apply H_256 via FWHT
        float rot[TQK_BLOCK_SIZE_D256];
        for (int j = 0; j < TQK_BLOCK_SIZE_D256; j++) rot[j] = xb[j] * inv;
        tq_fwht(rot, TQK_BLOCK_SIZE_D256);

        // 3-bit MSE quantize (8 centroids, d=256)
        memset(y[i].qs, 0, sizeof(y[i].qs));
        for (int j = 0; j < TQK_BLOCK_SIZE_D256; j++) {
            int idx = nearest(rot[j], centroids_8_d256, 8);
            pk3(y[i].qs, j, idx);
        }

        // Compute residual in ORIGINAL space: r = x - norm * H^{-1} * centroids
        float recon[TQK_BLOCK_SIZE_D256];
        for (int j = 0; j < TQK_BLOCK_SIZE_D256; j++) recon[j] = centroids_8_d256[up3(y[i].qs, j)];
        tq_fwht(recon, TQK_BLOCK_SIZE_D256);  // inverse FWHT = FWHT (self-inverse)
        float resid[TQK_BLOCK_SIZE_D256];
        for (int j = 0; j < TQK_BLOCK_SIZE_D256; j++) resid[j] = xb[j] - norm * recon[j];

        // QJL forward on residual
        y[i].rnorm = GGML_FP32_TO_FP16(qjl_forward(resid, y[i].signs, TQK_BLOCK_SIZE_D256, QJL_SEED_256));
    }
}

void dequantize_row_tqk_had_prod4_d256(const block_tqk_had_prod4_d256 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % TQK_BLOCK_SIZE_D256 == 0);
    const int64_t nb = k / TQK_BLOCK_SIZE_D256;

    for (int64_t i = 0; i < nb; i++) {
        float norm = GGML_FP16_TO_FP32(x[i].norm);

        // MSE reconstruction: centroids -> inverse FWHT -> scale by norm
        float rot[TQK_BLOCK_SIZE_D256];
        for (int j = 0; j < TQK_BLOCK_SIZE_D256; j++) {
            rot[j] = centroids_8_d256[up3(x[i].qs, j)];
        }
        tq_fwht(rot, TQK_BLOCK_SIZE_D256);
        for (int j = 0; j < TQK_BLOCK_SIZE_D256; j++) y[i * TQK_BLOCK_SIZE_D256 + j] = norm * rot[j];

        // QJL correction on residual
        float corr[TQK_BLOCK_SIZE_D256];
        qjl_inverse(x[i].signs, GGML_FP16_TO_FP32(x[i].rnorm), corr, TQK_BLOCK_SIZE_D256, QJL_SEED_256);
        for (int j = 0; j < TQK_BLOCK_SIZE_D256; j++) y[i * TQK_BLOCK_SIZE_D256 + j] += corr[j];
    }
}

// Asymmetric vec_dot: rotate Q with H_256, dot with stored centroids + QJL correction
void ggml_vec_dot_tqk_had_prod4_d256_f32(
        int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, size_t bx,
        const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % TQK_BLOCK_SIZE_D256 == 0);
    assert(nrc == 1);
    (void)bs; (void)bx; (void)by; (void)nrc;

    const block_tqk_had_prod4_d256 * GGML_RESTRICT x = (const block_tqk_had_prod4_d256 *)vx;
    const float * GGML_RESTRICT y = (const float *)vy;
    const int64_t nb = n / TQK_BLOCK_SIZE_D256;

    float sumf = 0.0f;

    for (int64_t i = 0; i < nb; i++) {
        const float * q = y + i * TQK_BLOCK_SIZE_D256;

        float norm = GGML_FP16_TO_FP32(x[i].norm);

        // Rotate Q with H_256
        float q_rot[TQK_BLOCK_SIZE_D256];
        for (int j = 0; j < TQK_BLOCK_SIZE_D256; j++) q_rot[j] = q[j];
        tq_fwht(q_rot, TQK_BLOCK_SIZE_D256);

        // MSE centroid dot (3-bit, 8 centroids)
        float dot = 0.0f;
        for (int j = 0; j < TQK_BLOCK_SIZE_D256; j++) {
            dot += q_rot[j] * centroids_8_d256[up3(x[i].qs, j)];
        }
        float mse_dot = norm * dot;

        // QJL correction: q_rot is already FWHT(q), which is H*q
        float qjl_dot = qjl_asymmetric_dot_rotated(q_rot, TQK_BLOCK_SIZE_D256,
                                            x[i].signs, GGML_FP16_TO_FP32(x[i].rnorm));

        sumf += mse_dot + qjl_dot;
    }

    *s = sumf;
}

// ---------------------------------------------------------------------------
// TQK 5hi_3lo_had_d256: 64/192 split, FWHT rotation, 4-bit MSE + QJL hi,
// 3-bit MSE lo. Hi uses H_64 FWHT. Lo uses 3 x H_64 FWHT (block-diagonal
// on 192 = 3 x 64). Lo centroids use d=64 (each 64-dim block is independent).
// ---------------------------------------------------------------------------

void quantize_row_tqk_5hi_3lo_had_d256_ref(const float * GGML_RESTRICT x, block_tqk_5hi_3lo_d256 * GGML_RESTRICT y, int64_t k) {
    assert(k % TQK_BLOCK_SIZE_D256 == 0);
    const int64_t nb = k / TQK_BLOCK_SIZE_D256;
    tq_init_rotations();

    for (int64_t i = 0; i < nb; i++) {
        if (nb > 1) tq_cur_head = (int)(i % (tq_n_heads > 0 ? tq_n_heads : nb));
        const float * xb = x + i * TQK_BLOCK_SIZE_D256;

        // Split into outlier/regular using per-head bitmask
        float hi_raw[TQK_N_OUTLIER_D256], lo_raw[TQK_N_REGULAR_D256];
        const uint32_t * mask = TQ_MASK_K(tq_cur_layer, tq_cur_head);
        {
            int oi = 0, ri = 0;
            for (int j = 0; j < TQK_BLOCK_SIZE_D256; j++) {
                if (tq_is_outlier(mask, j)) hi_raw[oi++] = xb[j];
                else                        lo_raw[ri++] = xb[j];
            }
        }

        // FWHT rotation: H_64 on hi, 3 x H_64 on lo
        float hi_rot[TQK_N_OUTLIER_D256], lo_rot[TQK_N_REGULAR_D256];
        memcpy(hi_rot, hi_raw, sizeof(hi_rot));
        tq_fwht(hi_rot, TQK_N_OUTLIER_D256);
        memcpy(lo_rot, lo_raw, sizeof(lo_rot));
        tq_fwht(lo_rot,       TQK_N_OUTLIER_D256);  // lo[0:64]
        tq_fwht(lo_rot + 64,  TQK_N_OUTLIER_D256);  // lo[64:128]
        tq_fwht(lo_rot + 128, TQK_N_OUTLIER_D256);  // lo[128:192]

        // Per-subset norms
        float sum_hi = 0.0f, sum_lo = 0.0f;
        for (int j = 0; j < TQK_N_OUTLIER_D256; j++) sum_hi += hi_rot[j] * hi_rot[j];
        for (int j = 0; j < TQK_N_REGULAR_D256; j++) sum_lo += lo_rot[j] * lo_rot[j];
        float norm_hi = sqrtf(sum_hi), norm_lo = sqrtf(sum_lo);

        y[i].norm_hi  = GGML_FP32_TO_FP16(norm_hi);
        y[i].norm_lo  = GGML_FP32_TO_FP16(norm_lo);
        y[i].rnorm_hi = GGML_FP32_TO_FP16(0.0f);
        memset(y[i].signs_hi, 0, sizeof(y[i].signs_hi));

        if (norm_hi == 0.0f && norm_lo == 0.0f) {
            memset(y[i].qs_hi, 0, sizeof(y[i].qs_hi));
            memset(y[i].qs_lo, 0, sizeof(y[i].qs_lo));
            continue;
        }

        float inv_hi = (norm_hi > 1e-12f) ? 1.0f / norm_hi : 0.0f;
        float inv_lo = (norm_lo > 1e-12f) ? 1.0f / norm_lo : 0.0f;

        // 4-bit MSE for hi (16 centroids, d=64)
        memset(y[i].qs_hi, 0, sizeof(y[i].qs_hi));
        for (int j = 0; j < TQK_N_OUTLIER_D256; j++) {
            float xn = hi_rot[j] * inv_hi;
            int idx = nearest(xn, centroids_16_d64, 16);
            pk4(y[i].qs_hi, j, idx);
        }

        // 3-bit MSE for lo (8 centroids, d=192 -- each 64-dim block uses d=64 centroids)
        memset(y[i].qs_lo, 0, sizeof(y[i].qs_lo));
        for (int j = 0; j < TQK_N_REGULAR_D256; j++) {
            float xn = lo_rot[j] * inv_lo;
            int idx = nearest(xn, centroids_8_d192, 8);
            pk3(y[i].qs_lo, j, idx);
        }

        // QJL on hi residual (in original subset space)
        // Reconstruct: unrotate centroids via inverse FWHT
        float yhi[TQK_N_OUTLIER_D256];
        for (int j = 0; j < TQK_N_OUTLIER_D256; j++) yhi[j] = centroids_16_d64[up4(y[i].qs_hi, j)];
        float hi_rec[TQK_N_OUTLIER_D256];
        memcpy(hi_rec, yhi, sizeof(hi_rec));
        tq_fwht(hi_rec, TQK_N_OUTLIER_D256); // inverse = forward for normalized FWHT
        float r_hi[TQK_N_OUTLIER_D256];
        for (int j = 0; j < TQK_N_OUTLIER_D256; j++) r_hi[j] = hi_raw[j] - norm_hi * hi_rec[j];
        y[i].rnorm_hi = GGML_FP32_TO_FP16(qjl_forward(r_hi, y[i].signs_hi, TQK_N_OUTLIER_D256, QJL_SEED_64));
    }
}

void dequantize_row_tqk_5hi_3lo_had_d256(const block_tqk_5hi_3lo_d256 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % TQK_BLOCK_SIZE_D256 == 0);
    const int64_t nb = k / TQK_BLOCK_SIZE_D256;
    tq_init_rotations();

    for (int64_t i = 0; i < nb; i++) {
        if (nb > 1) tq_cur_head = (int)(i % (tq_n_heads > 0 ? tq_n_heads : nb));
        float norm_hi = GGML_FP16_TO_FP32(x[i].norm_hi);
        float norm_lo = GGML_FP16_TO_FP32(x[i].norm_lo);

        // MSE recon: centroids -> inverse FWHT -> scale
        float yhi[TQK_N_OUTLIER_D256], ylo[TQK_N_REGULAR_D256];
        for (int j = 0; j < TQK_N_OUTLIER_D256; j++) yhi[j] = centroids_16_d64[up4(x[i].qs_hi, j)];
        for (int j = 0; j < TQK_N_REGULAR_D256; j++) ylo[j] = centroids_8_d192[up3(x[i].qs_lo, j)];

        // Inverse FWHT
        float hi_orig[TQK_N_OUTLIER_D256], lo_orig[TQK_N_REGULAR_D256];
        memcpy(hi_orig, yhi, sizeof(hi_orig));
        tq_fwht(hi_orig, TQK_N_OUTLIER_D256);
        memcpy(lo_orig, ylo, sizeof(lo_orig));
        tq_fwht(lo_orig,       TQK_N_OUTLIER_D256);
        tq_fwht(lo_orig + 64,  TQK_N_OUTLIER_D256);
        tq_fwht(lo_orig + 128, TQK_N_OUTLIER_D256);

        for (int j = 0; j < TQK_N_OUTLIER_D256; j++) hi_orig[j] *= norm_hi;
        for (int j = 0; j < TQK_N_REGULAR_D256; j++) lo_orig[j] *= norm_lo;

        // QJL correction on hi
        float corr_hi[TQK_N_OUTLIER_D256];
        qjl_inverse(x[i].signs_hi, GGML_FP16_TO_FP32(x[i].rnorm_hi), corr_hi, TQK_N_OUTLIER_D256, QJL_SEED_64);
        for (int j = 0; j < TQK_N_OUTLIER_D256; j++) hi_orig[j] += corr_hi[j];

        // Merge back using per-head bitmask
        float * out = y + i * TQK_BLOCK_SIZE_D256;
        const uint32_t * mask = TQ_MASK_K(tq_cur_layer, tq_cur_head);
        {
            int oi = 0, ri = 0;
            for (int j = 0; j < TQK_BLOCK_SIZE_D256; j++) {
                if (tq_is_outlier(mask, j)) out[j] = hi_orig[oi++];
                else                        out[j] = lo_orig[ri++];
            }
        }
    }
}

// Asymmetric vec_dot for 5hi_3lo_had_d256
void ggml_vec_dot_tqk_5hi_3lo_had_d256_f32(
        int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, size_t bx,
        const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % TQK_BLOCK_SIZE_D256 == 0);
    assert(nrc == 1);
    (void)bs; (void)bx; (void)by; (void)nrc;

    tq_init_rotations();

    const block_tqk_5hi_3lo_d256 * GGML_RESTRICT x = (const block_tqk_5hi_3lo_d256 *)vx;
    const float * GGML_RESTRICT y = (const float *)vy;
    const int64_t nb = n / TQK_BLOCK_SIZE_D256;

    float sumf = 0.0f;

    for (int64_t i = 0; i < nb; i++) {
        const float * q = y + i * TQK_BLOCK_SIZE_D256;

        if (nb > 1) tq_cur_head = (int)(i % (tq_n_heads > 0 ? tq_n_heads : nb));
        float norm_hi = GGML_FP16_TO_FP32(x[i].norm_hi);
        float norm_lo = GGML_FP16_TO_FP32(x[i].norm_lo);

        // Split query using per-head bitmask
        float hi_raw[TQK_N_OUTLIER_D256], lo_raw[TQK_N_REGULAR_D256];
        const uint32_t * mask = TQ_MASK_K(tq_cur_layer, tq_cur_head);
        {
            int oi = 0, ri = 0;
            for (int j = 0; j < TQK_BLOCK_SIZE_D256; j++) {
                if (tq_is_outlier(mask, j)) hi_raw[oi++] = q[j];
                else                        lo_raw[ri++] = q[j];
            }
        }

        // FWHT-rotate
        float q_rot_hi[TQK_N_OUTLIER_D256], q_rot_lo[TQK_N_REGULAR_D256];
        memcpy(q_rot_hi, hi_raw, sizeof(q_rot_hi));
        tq_fwht(q_rot_hi, TQK_N_OUTLIER_D256);
        memcpy(q_rot_lo, lo_raw, sizeof(q_rot_lo));
        tq_fwht(q_rot_lo,       TQK_N_OUTLIER_D256);
        tq_fwht(q_rot_lo + 64,  TQK_N_OUTLIER_D256);
        tq_fwht(q_rot_lo + 128, TQK_N_OUTLIER_D256);

        // MSE centroid dot per subset
        float mse_dot_hi = 0.0f, mse_dot_lo = 0.0f;
        for (int j = 0; j < TQK_N_OUTLIER_D256; j++) {
            mse_dot_hi += q_rot_hi[j] * centroids_16_d64[up4(x[i].qs_hi, j)];
        }
        for (int j = 0; j < TQK_N_REGULAR_D256; j++) {
            mse_dot_lo += q_rot_lo[j] * centroids_8_d192[up3(x[i].qs_lo, j)];
        }
        float mse_dot = mse_dot_hi * norm_hi + mse_dot_lo * norm_lo;

        // QJL correction on hi (raw query, not rotated)
        float qjl_hi = qjl_asymmetric_dot(hi_raw, TQK_N_OUTLIER_D256, QJL_SEED_64,
                                           x[i].signs_hi, GGML_FP16_TO_FP32(x[i].rnorm_hi));

        sumf += mse_dot + qjl_hi;
    }

    *s = sumf;
}

// ---------------------------------------------------------------------------
// TQK 6hi_3lo_had_d256: 5-bit MSE + QJL on 64 outliers, 3-bit MSE on 192 regulars
// ---------------------------------------------------------------------------

void quantize_row_tqk_6hi_3lo_had_d256_ref(const float * GGML_RESTRICT x, block_tqk_6hi_3lo_d256 * GGML_RESTRICT y, int64_t k) {
    assert(k % TQK_BLOCK_SIZE_D256 == 0);
    const int64_t nb = k / TQK_BLOCK_SIZE_D256;
    tq_init_rotations();
    for (int64_t i = 0; i < nb; i++) {
        if (nb > 1) tq_cur_head = (int)(i % (tq_n_heads > 0 ? tq_n_heads : nb));
        const float * xb = x + i * TQK_BLOCK_SIZE_D256;
        float hi_raw[TQK_N_OUTLIER_D256], lo_raw[TQK_N_REGULAR_D256];
        const uint32_t * mask = TQ_MASK_K(tq_cur_layer, tq_cur_head);
        { int oi = 0, ri = 0;
          for (int j = 0; j < TQK_BLOCK_SIZE_D256; j++) {
              if (tq_is_outlier(mask, j)) hi_raw[oi++] = xb[j]; else lo_raw[ri++] = xb[j];
          }
        }
        float hi_rot[TQK_N_OUTLIER_D256], lo_rot[TQK_N_REGULAR_D256];
        memcpy(hi_rot, hi_raw, sizeof(hi_rot));
        tq_fwht(hi_rot, TQK_N_OUTLIER_D256);
        memcpy(lo_rot, lo_raw, sizeof(lo_rot));
        tq_fwht(lo_rot, TQK_N_OUTLIER_D256); tq_fwht(lo_rot+64, TQK_N_OUTLIER_D256); tq_fwht(lo_rot+128, TQK_N_OUTLIER_D256);
        float sum_hi = 0, sum_lo = 0;
        for (int j = 0; j < TQK_N_OUTLIER_D256; j++) sum_hi += hi_rot[j]*hi_rot[j];
        for (int j = 0; j < TQK_N_REGULAR_D256; j++) sum_lo += lo_rot[j]*lo_rot[j];
        float norm_hi = sqrtf(sum_hi), norm_lo = sqrtf(sum_lo);
        y[i].norm_hi = GGML_FP32_TO_FP16(norm_hi); y[i].norm_lo = GGML_FP32_TO_FP16(norm_lo);
        y[i].rnorm_hi = GGML_FP32_TO_FP16(0);
        memset(y[i].signs_hi, 0, sizeof(y[i].signs_hi));
        if (norm_hi == 0 && norm_lo == 0) { memset(y[i].qs_hi,0,sizeof(y[i].qs_hi)); memset(y[i].qs_lo,0,sizeof(y[i].qs_lo)); continue; }
        float inv_hi = (norm_hi > 1e-12f) ? 1.0f/norm_hi : 0, inv_lo = (norm_lo > 1e-12f) ? 1.0f/norm_lo : 0;
        memset(y[i].qs_hi, 0, sizeof(y[i].qs_hi));
        for (int j = 0; j < TQK_N_OUTLIER_D256; j++) pk5(y[i].qs_hi, j, nearest(hi_rot[j]*inv_hi, centroids_32_d64, 32));
        memset(y[i].qs_lo, 0, sizeof(y[i].qs_lo));
        for (int j = 0; j < TQK_N_REGULAR_D256; j++) pk3(y[i].qs_lo, j, nearest(lo_rot[j]*inv_lo, centroids_8_d192, 8));
        float yhi[TQK_N_OUTLIER_D256];
        for (int j = 0; j < TQK_N_OUTLIER_D256; j++) yhi[j] = centroids_32_d64[up5(y[i].qs_hi, j)];
        float hi_rec[TQK_N_OUTLIER_D256]; memcpy(hi_rec, yhi, sizeof(hi_rec)); tq_fwht(hi_rec, TQK_N_OUTLIER_D256);
        float r_hi[TQK_N_OUTLIER_D256]; for (int j = 0; j < TQK_N_OUTLIER_D256; j++) r_hi[j] = hi_raw[j] - norm_hi*hi_rec[j];
        y[i].rnorm_hi = GGML_FP32_TO_FP16(qjl_forward(r_hi, y[i].signs_hi, TQK_N_OUTLIER_D256, QJL_SEED_64));
    }
}

void dequantize_row_tqk_6hi_3lo_had_d256(const block_tqk_6hi_3lo_d256 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % TQK_BLOCK_SIZE_D256 == 0);
    const int64_t nb = k / TQK_BLOCK_SIZE_D256;
    tq_init_rotations();
    for (int64_t i = 0; i < nb; i++) {
        if (nb > 1) tq_cur_head = (int)(i % (tq_n_heads > 0 ? tq_n_heads : nb));
        float norm_hi = GGML_FP16_TO_FP32(x[i].norm_hi), norm_lo = GGML_FP16_TO_FP32(x[i].norm_lo);
        float yhi[TQK_N_OUTLIER_D256], ylo[TQK_N_REGULAR_D256];
        for (int j = 0; j < TQK_N_OUTLIER_D256; j++) yhi[j] = centroids_32_d64[up5(x[i].qs_hi, j)];
        for (int j = 0; j < TQK_N_REGULAR_D256; j++) ylo[j] = centroids_8_d192[up3(x[i].qs_lo, j)];
        float hi_orig[TQK_N_OUTLIER_D256], lo_orig[TQK_N_REGULAR_D256];
        memcpy(hi_orig, yhi, sizeof(hi_orig)); tq_fwht(hi_orig, TQK_N_OUTLIER_D256);
        memcpy(lo_orig, ylo, sizeof(lo_orig));
        tq_fwht(lo_orig, TQK_N_OUTLIER_D256); tq_fwht(lo_orig+64, TQK_N_OUTLIER_D256); tq_fwht(lo_orig+128, TQK_N_OUTLIER_D256);
        for (int j = 0; j < TQK_N_OUTLIER_D256; j++) hi_orig[j] *= norm_hi;
        for (int j = 0; j < TQK_N_REGULAR_D256; j++) lo_orig[j] *= norm_lo;
        float corr_hi[TQK_N_OUTLIER_D256];
        qjl_inverse(x[i].signs_hi, GGML_FP16_TO_FP32(x[i].rnorm_hi), corr_hi, TQK_N_OUTLIER_D256, QJL_SEED_64);
        for (int j = 0; j < TQK_N_OUTLIER_D256; j++) hi_orig[j] += corr_hi[j];
        float * out = y + i * TQK_BLOCK_SIZE_D256;
        const uint32_t * mask = TQ_MASK_K(tq_cur_layer, tq_cur_head);
        { int oi = 0, ri = 0;
          for (int j = 0; j < TQK_BLOCK_SIZE_D256; j++) {
              if (tq_is_outlier(mask, j)) out[j] = hi_orig[oi++]; else out[j] = lo_orig[ri++];
          }
        }
    }
}

void ggml_vec_dot_tqk_6hi_3lo_had_d256_f32(int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % TQK_BLOCK_SIZE_D256 == 0); assert(nrc == 1);
    (void)bs; (void)bx; (void)by; (void)nrc;
    tq_init_rotations();
    const block_tqk_6hi_3lo_d256 * GGML_RESTRICT x = (const block_tqk_6hi_3lo_d256 *)vx;
    const float * GGML_RESTRICT y = (const float *)vy;
    const int64_t nb = n / TQK_BLOCK_SIZE_D256;
    float sumf = 0;
    for (int64_t i = 0; i < nb; i++) {
        const float * q = y + i*TQK_BLOCK_SIZE_D256;
        if (nb > 1) tq_cur_head = (int)(i % (tq_n_heads > 0 ? tq_n_heads : nb));
        float norm_hi = GGML_FP16_TO_FP32(x[i].norm_hi), norm_lo = GGML_FP16_TO_FP32(x[i].norm_lo);
        float hi_raw[TQK_N_OUTLIER_D256], lo_raw[TQK_N_REGULAR_D256];
        const uint32_t * mask = TQ_MASK_K(tq_cur_layer, tq_cur_head);
        { int oi = 0, ri = 0;
          for (int j = 0; j < TQK_BLOCK_SIZE_D256; j++) {
              if (tq_is_outlier(mask, j)) hi_raw[oi++] = q[j]; else lo_raw[ri++] = q[j];
          }
        }
        float q_rot_hi[TQK_N_OUTLIER_D256], q_rot_lo[TQK_N_REGULAR_D256];
        memcpy(q_rot_hi, hi_raw, sizeof(q_rot_hi)); tq_fwht(q_rot_hi, TQK_N_OUTLIER_D256);
        memcpy(q_rot_lo, lo_raw, sizeof(q_rot_lo));
        tq_fwht(q_rot_lo, TQK_N_OUTLIER_D256); tq_fwht(q_rot_lo+64, TQK_N_OUTLIER_D256); tq_fwht(q_rot_lo+128, TQK_N_OUTLIER_D256);
        float mse_dot_hi = 0, mse_dot_lo = 0;
        for (int j = 0; j < TQK_N_OUTLIER_D256; j++) mse_dot_hi += q_rot_hi[j]*centroids_32_d64[up5(x[i].qs_hi, j)];
        for (int j = 0; j < TQK_N_REGULAR_D256; j++) mse_dot_lo += q_rot_lo[j]*centroids_8_d192[up3(x[i].qs_lo, j)];
        float qjl_hi = qjl_asymmetric_dot(hi_raw, TQK_N_OUTLIER_D256, QJL_SEED_64, x[i].signs_hi, GGML_FP16_TO_FP32(x[i].rnorm_hi));
        sumf += mse_dot_hi*norm_hi + mse_dot_lo*norm_lo + qjl_hi;
    }
    *s = sumf;
}

// ---------------------------------------------------------------------------
// TQK 2hi_1lo_had_d256: 2-bit MSE + QJL on 64 outliers, 1-bit MSE + QJL on 192 regulars
// ---------------------------------------------------------------------------

void quantize_row_tqk_2hi_1lo_had_d256_ref(const float * GGML_RESTRICT x, block_tqk_2hi_1lo_d256 * GGML_RESTRICT y, int64_t k) {
    assert(k % TQK_BLOCK_SIZE_D256 == 0);
    const int64_t nb = k / TQK_BLOCK_SIZE_D256;
    tq_init_rotations();
    for (int64_t i = 0; i < nb; i++) {
        if (nb > 1) tq_cur_head = (int)(i % (tq_n_heads > 0 ? tq_n_heads : nb));
        const float * xb = x + i * TQK_BLOCK_SIZE_D256;
        float hi_raw[TQK_N_OUTLIER_D256], lo_raw[TQK_N_REGULAR_D256];
        const uint32_t * mask = TQ_MASK_K(tq_cur_layer, tq_cur_head);
        { int oi = 0, ri = 0;
          for (int j = 0; j < TQK_BLOCK_SIZE_D256; j++) {
              if (tq_is_outlier(mask, j)) hi_raw[oi++] = xb[j]; else lo_raw[ri++] = xb[j];
          }
        }
        float hi_rot[TQK_N_OUTLIER_D256], lo_rot[TQK_N_REGULAR_D256];
        memcpy(hi_rot, hi_raw, sizeof(hi_rot));
        tq_fwht(hi_rot, TQK_N_OUTLIER_D256);
        memcpy(lo_rot, lo_raw, sizeof(lo_rot));
        tq_fwht(lo_rot, TQK_N_OUTLIER_D256); tq_fwht(lo_rot+64, TQK_N_OUTLIER_D256); tq_fwht(lo_rot+128, TQK_N_OUTLIER_D256);
        float sum_hi = 0, sum_lo = 0;
        for (int j = 0; j < TQK_N_OUTLIER_D256; j++) sum_hi += hi_rot[j]*hi_rot[j];
        for (int j = 0; j < TQK_N_REGULAR_D256; j++) sum_lo += lo_rot[j]*lo_rot[j];
        float norm_hi = sqrtf(sum_hi), norm_lo = sqrtf(sum_lo);
        y[i].norm_hi = GGML_FP32_TO_FP16(norm_hi); y[i].norm_lo = GGML_FP32_TO_FP16(norm_lo);
        y[i].rnorm_hi = GGML_FP32_TO_FP16(0); y[i].rnorm_lo = GGML_FP32_TO_FP16(0);
        memset(y[i].signs_hi, 0, sizeof(y[i].signs_hi));
        memset(y[i].signs_lo, 0, sizeof(y[i].signs_lo));
        if (norm_hi == 0 && norm_lo == 0) { memset(y[i].qs_hi,0,sizeof(y[i].qs_hi)); memset(y[i].qs_lo,0,sizeof(y[i].qs_lo)); continue; }
        float inv_hi = (norm_hi > 1e-12f) ? 1.0f/norm_hi : 0, inv_lo = (norm_lo > 1e-12f) ? 1.0f/norm_lo : 0;
        memset(y[i].qs_hi, 0, sizeof(y[i].qs_hi));
        for (int j = 0; j < TQK_N_OUTLIER_D256; j++) pk2(y[i].qs_hi, j, nearest(hi_rot[j]*inv_hi, centroids_4_d64, 4));
        memset(y[i].qs_lo, 0, sizeof(y[i].qs_lo));
        for (int j = 0; j < TQK_N_REGULAR_D256; j++) pk1(y[i].qs_lo, j, nearest(lo_rot[j]*inv_lo, centroids_2_d192, 2));
        float yhi[TQK_N_OUTLIER_D256];
        for (int j = 0; j < TQK_N_OUTLIER_D256; j++) yhi[j] = centroids_4_d64[up2(y[i].qs_hi, j)];
        float hi_rec[TQK_N_OUTLIER_D256]; memcpy(hi_rec, yhi, sizeof(hi_rec)); tq_fwht(hi_rec, TQK_N_OUTLIER_D256);
        float r_hi[TQK_N_OUTLIER_D256]; for (int j = 0; j < TQK_N_OUTLIER_D256; j++) r_hi[j] = hi_raw[j] - norm_hi*hi_rec[j];
        y[i].rnorm_hi = GGML_FP32_TO_FP16(qjl_forward(r_hi, y[i].signs_hi, TQK_N_OUTLIER_D256, QJL_SEED_64));
        float ylo[TQK_N_REGULAR_D256];
        for (int j = 0; j < TQK_N_REGULAR_D256; j++) ylo[j] = centroids_2_d192[up1(y[i].qs_lo, j)];
        float lo_rec[TQK_N_REGULAR_D256]; memcpy(lo_rec, ylo, sizeof(lo_rec));
        tq_fwht(lo_rec, TQK_N_OUTLIER_D256); tq_fwht(lo_rec+64, TQK_N_OUTLIER_D256); tq_fwht(lo_rec+128, TQK_N_OUTLIER_D256);
        float r_lo[TQK_N_REGULAR_D256]; for (int j = 0; j < TQK_N_REGULAR_D256; j++) r_lo[j] = lo_raw[j] - norm_lo*lo_rec[j];
        y[i].rnorm_lo = GGML_FP32_TO_FP16(qjl_forward_block3(r_lo, y[i].signs_lo, TQK_N_OUTLIER_D256));
    }
}

void dequantize_row_tqk_2hi_1lo_had_d256(const block_tqk_2hi_1lo_d256 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % TQK_BLOCK_SIZE_D256 == 0);
    const int64_t nb = k / TQK_BLOCK_SIZE_D256;
    tq_init_rotations();
    for (int64_t i = 0; i < nb; i++) {
        if (nb > 1) tq_cur_head = (int)(i % (tq_n_heads > 0 ? tq_n_heads : nb));
        float norm_hi = GGML_FP16_TO_FP32(x[i].norm_hi), norm_lo = GGML_FP16_TO_FP32(x[i].norm_lo);
        float yhi[TQK_N_OUTLIER_D256], ylo[TQK_N_REGULAR_D256];
        for (int j = 0; j < TQK_N_OUTLIER_D256; j++) yhi[j] = centroids_4_d64[up2(x[i].qs_hi, j)];
        for (int j = 0; j < TQK_N_REGULAR_D256; j++) ylo[j] = centroids_2_d192[up1(x[i].qs_lo, j)];
        float hi_orig[TQK_N_OUTLIER_D256], lo_orig[TQK_N_REGULAR_D256];
        memcpy(hi_orig, yhi, sizeof(hi_orig)); tq_fwht(hi_orig, TQK_N_OUTLIER_D256);
        memcpy(lo_orig, ylo, sizeof(lo_orig));
        tq_fwht(lo_orig, TQK_N_OUTLIER_D256); tq_fwht(lo_orig+64, TQK_N_OUTLIER_D256); tq_fwht(lo_orig+128, TQK_N_OUTLIER_D256);
        for (int j = 0; j < TQK_N_OUTLIER_D256; j++) hi_orig[j] *= norm_hi;
        for (int j = 0; j < TQK_N_REGULAR_D256; j++) lo_orig[j] *= norm_lo;
        float corr_hi[TQK_N_OUTLIER_D256];
        qjl_inverse(x[i].signs_hi, GGML_FP16_TO_FP32(x[i].rnorm_hi), corr_hi, TQK_N_OUTLIER_D256, QJL_SEED_64);
        for (int j = 0; j < TQK_N_OUTLIER_D256; j++) hi_orig[j] += corr_hi[j];
        float corr_lo[TQK_N_REGULAR_D256];
        qjl_inverse_block3(x[i].signs_lo, GGML_FP16_TO_FP32(x[i].rnorm_lo), corr_lo, TQK_N_OUTLIER_D256);
        for (int j = 0; j < TQK_N_REGULAR_D256; j++) lo_orig[j] += corr_lo[j];
        float * out = y + i * TQK_BLOCK_SIZE_D256;
        const uint32_t * mask = TQ_MASK_K(tq_cur_layer, tq_cur_head);
        { int oi = 0, ri = 0;
          for (int j = 0; j < TQK_BLOCK_SIZE_D256; j++) {
              if (tq_is_outlier(mask, j)) out[j] = hi_orig[oi++]; else out[j] = lo_orig[ri++];
          }
        }
    }
}

void ggml_vec_dot_tqk_2hi_1lo_had_d256_f32(int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % TQK_BLOCK_SIZE_D256 == 0); assert(nrc == 1);
    (void)bs; (void)bx; (void)by; (void)nrc;
    tq_init_rotations();
    const block_tqk_2hi_1lo_d256 * GGML_RESTRICT x = (const block_tqk_2hi_1lo_d256 *)vx;
    const float * GGML_RESTRICT y = (const float *)vy;
    const int64_t nb = n / TQK_BLOCK_SIZE_D256;
    float sumf = 0;
    for (int64_t i = 0; i < nb; i++) {
        const float * q = y + i*TQK_BLOCK_SIZE_D256;
        if (nb > 1) tq_cur_head = (int)(i % (tq_n_heads > 0 ? tq_n_heads : nb));
        float norm_hi = GGML_FP16_TO_FP32(x[i].norm_hi), norm_lo = GGML_FP16_TO_FP32(x[i].norm_lo);
        float hi_raw[TQK_N_OUTLIER_D256], lo_raw[TQK_N_REGULAR_D256];
        const uint32_t * mask = TQ_MASK_K(tq_cur_layer, tq_cur_head);
        { int oi = 0, ri = 0;
          for (int j = 0; j < TQK_BLOCK_SIZE_D256; j++) {
              if (tq_is_outlier(mask, j)) hi_raw[oi++] = q[j]; else lo_raw[ri++] = q[j];
          }
        }
        float q_rot_hi[TQK_N_OUTLIER_D256], q_rot_lo[TQK_N_REGULAR_D256];
        memcpy(q_rot_hi, hi_raw, sizeof(q_rot_hi)); tq_fwht(q_rot_hi, TQK_N_OUTLIER_D256);
        memcpy(q_rot_lo, lo_raw, sizeof(q_rot_lo));
        tq_fwht(q_rot_lo, TQK_N_OUTLIER_D256); tq_fwht(q_rot_lo+64, TQK_N_OUTLIER_D256); tq_fwht(q_rot_lo+128, TQK_N_OUTLIER_D256);
        float mse_dot_hi = 0, mse_dot_lo = 0;
        for (int j = 0; j < TQK_N_OUTLIER_D256; j++) mse_dot_hi += q_rot_hi[j]*centroids_4_d64[up2(x[i].qs_hi, j)];
        for (int j = 0; j < TQK_N_REGULAR_D256; j++) mse_dot_lo += q_rot_lo[j]*centroids_2_d192[up1(x[i].qs_lo, j)];
        float qjl_hi = qjl_asymmetric_dot(hi_raw, TQK_N_OUTLIER_D256, QJL_SEED_64, x[i].signs_hi, GGML_FP16_TO_FP32(x[i].rnorm_hi));
        float qjl_lo = qjl_asymmetric_dot_block3(lo_raw, TQK_N_OUTLIER_D256, x[i].signs_lo, GGML_FP16_TO_FP32(x[i].rnorm_lo));
        sumf += mse_dot_hi*norm_hi + mse_dot_lo*norm_lo + qjl_hi + qjl_lo;
    }
    *s = sumf;
}

// ---------------------------------------------------------------------------
// TQK 3hi_2lo_had_d256: 3-bit MSE + QJL on 64 outliers, 2-bit MSE + QJL on 192 regulars
// ---------------------------------------------------------------------------

void quantize_row_tqk_3hi_2lo_had_d256_ref(const float * GGML_RESTRICT x, block_tqk_3hi_2lo_d256 * GGML_RESTRICT y, int64_t k) {
    assert(k % TQK_BLOCK_SIZE_D256 == 0);
    const int64_t nb = k / TQK_BLOCK_SIZE_D256;
    tq_init_rotations();
    for (int64_t i = 0; i < nb; i++) {
        if (nb > 1) tq_cur_head = (int)(i % (tq_n_heads > 0 ? tq_n_heads : nb));
        const float * xb = x + i * TQK_BLOCK_SIZE_D256;
        float hi_raw[TQK_N_OUTLIER_D256], lo_raw[TQK_N_REGULAR_D256];
        const uint32_t * mask = TQ_MASK_K(tq_cur_layer, tq_cur_head);
        { int oi = 0, ri = 0;
          for (int j = 0; j < TQK_BLOCK_SIZE_D256; j++) {
              if (tq_is_outlier(mask, j)) hi_raw[oi++] = xb[j]; else lo_raw[ri++] = xb[j];
          }
        }
        float hi_rot[TQK_N_OUTLIER_D256], lo_rot[TQK_N_REGULAR_D256];
        memcpy(hi_rot, hi_raw, sizeof(hi_rot));
        tq_fwht(hi_rot, TQK_N_OUTLIER_D256);
        memcpy(lo_rot, lo_raw, sizeof(lo_rot));
        tq_fwht(lo_rot, TQK_N_OUTLIER_D256); tq_fwht(lo_rot+64, TQK_N_OUTLIER_D256); tq_fwht(lo_rot+128, TQK_N_OUTLIER_D256);
        float sum_hi = 0, sum_lo = 0;
        for (int j = 0; j < TQK_N_OUTLIER_D256; j++) sum_hi += hi_rot[j]*hi_rot[j];
        for (int j = 0; j < TQK_N_REGULAR_D256; j++) sum_lo += lo_rot[j]*lo_rot[j];
        float norm_hi = sqrtf(sum_hi), norm_lo = sqrtf(sum_lo);
        y[i].norm_hi = GGML_FP32_TO_FP16(norm_hi); y[i].norm_lo = GGML_FP32_TO_FP16(norm_lo);
        y[i].rnorm_hi = GGML_FP32_TO_FP16(0); y[i].rnorm_lo = GGML_FP32_TO_FP16(0);
        memset(y[i].signs_hi, 0, sizeof(y[i].signs_hi));
        memset(y[i].signs_lo, 0, sizeof(y[i].signs_lo));
        if (norm_hi == 0 && norm_lo == 0) { memset(y[i].qs_hi,0,sizeof(y[i].qs_hi)); memset(y[i].qs_lo,0,sizeof(y[i].qs_lo)); continue; }
        float inv_hi = (norm_hi > 1e-12f) ? 1.0f/norm_hi : 0, inv_lo = (norm_lo > 1e-12f) ? 1.0f/norm_lo : 0;
        memset(y[i].qs_hi, 0, sizeof(y[i].qs_hi));
        for (int j = 0; j < TQK_N_OUTLIER_D256; j++) pk3(y[i].qs_hi, j, nearest(hi_rot[j]*inv_hi, centroids_8_d64, 8));
        memset(y[i].qs_lo, 0, sizeof(y[i].qs_lo));
        for (int j = 0; j < TQK_N_REGULAR_D256; j++) pk2(y[i].qs_lo, j, nearest(lo_rot[j]*inv_lo, centroids_4_d192, 4));
        float yhi[TQK_N_OUTLIER_D256];
        for (int j = 0; j < TQK_N_OUTLIER_D256; j++) yhi[j] = centroids_8_d64[up3(y[i].qs_hi, j)];
        float hi_rec[TQK_N_OUTLIER_D256]; memcpy(hi_rec, yhi, sizeof(hi_rec)); tq_fwht(hi_rec, TQK_N_OUTLIER_D256);
        float r_hi[TQK_N_OUTLIER_D256]; for (int j = 0; j < TQK_N_OUTLIER_D256; j++) r_hi[j] = hi_raw[j] - norm_hi*hi_rec[j];
        y[i].rnorm_hi = GGML_FP32_TO_FP16(qjl_forward(r_hi, y[i].signs_hi, TQK_N_OUTLIER_D256, QJL_SEED_64));
        float ylo[TQK_N_REGULAR_D256];
        for (int j = 0; j < TQK_N_REGULAR_D256; j++) ylo[j] = centroids_4_d192[up2(y[i].qs_lo, j)];
        float lo_rec[TQK_N_REGULAR_D256]; memcpy(lo_rec, ylo, sizeof(lo_rec));
        tq_fwht(lo_rec, TQK_N_OUTLIER_D256); tq_fwht(lo_rec+64, TQK_N_OUTLIER_D256); tq_fwht(lo_rec+128, TQK_N_OUTLIER_D256);
        float r_lo[TQK_N_REGULAR_D256]; for (int j = 0; j < TQK_N_REGULAR_D256; j++) r_lo[j] = lo_raw[j] - norm_lo*lo_rec[j];
        y[i].rnorm_lo = GGML_FP32_TO_FP16(qjl_forward_block3(r_lo, y[i].signs_lo, TQK_N_OUTLIER_D256));
    }
}

void dequantize_row_tqk_3hi_2lo_had_d256(const block_tqk_3hi_2lo_d256 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % TQK_BLOCK_SIZE_D256 == 0);
    const int64_t nb = k / TQK_BLOCK_SIZE_D256;
    tq_init_rotations();
    for (int64_t i = 0; i < nb; i++) {
        if (nb > 1) tq_cur_head = (int)(i % (tq_n_heads > 0 ? tq_n_heads : nb));
        float norm_hi = GGML_FP16_TO_FP32(x[i].norm_hi), norm_lo = GGML_FP16_TO_FP32(x[i].norm_lo);
        float yhi[TQK_N_OUTLIER_D256], ylo[TQK_N_REGULAR_D256];
        for (int j = 0; j < TQK_N_OUTLIER_D256; j++) yhi[j] = centroids_8_d64[up3(x[i].qs_hi, j)];
        for (int j = 0; j < TQK_N_REGULAR_D256; j++) ylo[j] = centroids_4_d192[up2(x[i].qs_lo, j)];
        float hi_orig[TQK_N_OUTLIER_D256], lo_orig[TQK_N_REGULAR_D256];
        memcpy(hi_orig, yhi, sizeof(hi_orig)); tq_fwht(hi_orig, TQK_N_OUTLIER_D256);
        memcpy(lo_orig, ylo, sizeof(lo_orig));
        tq_fwht(lo_orig, TQK_N_OUTLIER_D256); tq_fwht(lo_orig+64, TQK_N_OUTLIER_D256); tq_fwht(lo_orig+128, TQK_N_OUTLIER_D256);
        for (int j = 0; j < TQK_N_OUTLIER_D256; j++) hi_orig[j] *= norm_hi;
        for (int j = 0; j < TQK_N_REGULAR_D256; j++) lo_orig[j] *= norm_lo;
        float corr_hi[TQK_N_OUTLIER_D256];
        qjl_inverse(x[i].signs_hi, GGML_FP16_TO_FP32(x[i].rnorm_hi), corr_hi, TQK_N_OUTLIER_D256, QJL_SEED_64);
        for (int j = 0; j < TQK_N_OUTLIER_D256; j++) hi_orig[j] += corr_hi[j];
        float corr_lo[TQK_N_REGULAR_D256];
        qjl_inverse_block3(x[i].signs_lo, GGML_FP16_TO_FP32(x[i].rnorm_lo), corr_lo, TQK_N_OUTLIER_D256);
        for (int j = 0; j < TQK_N_REGULAR_D256; j++) lo_orig[j] += corr_lo[j];
        float * out = y + i * TQK_BLOCK_SIZE_D256;
        const uint32_t * mask = TQ_MASK_K(tq_cur_layer, tq_cur_head);
        { int oi = 0, ri = 0;
          for (int j = 0; j < TQK_BLOCK_SIZE_D256; j++) {
              if (tq_is_outlier(mask, j)) out[j] = hi_orig[oi++]; else out[j] = lo_orig[ri++];
          }
        }
    }
}

void ggml_vec_dot_tqk_3hi_2lo_had_d256_f32(int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % TQK_BLOCK_SIZE_D256 == 0); assert(nrc == 1);
    (void)bs; (void)bx; (void)by; (void)nrc;
    tq_init_rotations();
    const block_tqk_3hi_2lo_d256 * GGML_RESTRICT x = (const block_tqk_3hi_2lo_d256 *)vx;
    const float * GGML_RESTRICT y = (const float *)vy;
    const int64_t nb = n / TQK_BLOCK_SIZE_D256;
    float sumf = 0;
    for (int64_t i = 0; i < nb; i++) {
        const float * q = y + i*TQK_BLOCK_SIZE_D256;
        if (nb > 1) tq_cur_head = (int)(i % (tq_n_heads > 0 ? tq_n_heads : nb));
        float norm_hi = GGML_FP16_TO_FP32(x[i].norm_hi), norm_lo = GGML_FP16_TO_FP32(x[i].norm_lo);
        float hi_raw[TQK_N_OUTLIER_D256], lo_raw[TQK_N_REGULAR_D256];
        const uint32_t * mask = TQ_MASK_K(tq_cur_layer, tq_cur_head);
        { int oi = 0, ri = 0;
          for (int j = 0; j < TQK_BLOCK_SIZE_D256; j++) {
              if (tq_is_outlier(mask, j)) hi_raw[oi++] = q[j]; else lo_raw[ri++] = q[j];
          }
        }
        float q_rot_hi[TQK_N_OUTLIER_D256], q_rot_lo[TQK_N_REGULAR_D256];
        memcpy(q_rot_hi, hi_raw, sizeof(q_rot_hi)); tq_fwht(q_rot_hi, TQK_N_OUTLIER_D256);
        memcpy(q_rot_lo, lo_raw, sizeof(q_rot_lo));
        tq_fwht(q_rot_lo, TQK_N_OUTLIER_D256); tq_fwht(q_rot_lo+64, TQK_N_OUTLIER_D256); tq_fwht(q_rot_lo+128, TQK_N_OUTLIER_D256);
        float mse_dot_hi = 0, mse_dot_lo = 0;
        for (int j = 0; j < TQK_N_OUTLIER_D256; j++) mse_dot_hi += q_rot_hi[j]*centroids_8_d64[up3(x[i].qs_hi, j)];
        for (int j = 0; j < TQK_N_REGULAR_D256; j++) mse_dot_lo += q_rot_lo[j]*centroids_4_d192[up2(x[i].qs_lo, j)];
        float qjl_hi = qjl_asymmetric_dot(hi_raw, TQK_N_OUTLIER_D256, QJL_SEED_64, x[i].signs_hi, GGML_FP16_TO_FP32(x[i].rnorm_hi));
        float qjl_lo = qjl_asymmetric_dot_block3(lo_raw, TQK_N_OUTLIER_D256, x[i].signs_lo, GGML_FP16_TO_FP32(x[i].rnorm_lo));
        sumf += mse_dot_hi*norm_hi + mse_dot_lo*norm_lo + qjl_hi + qjl_lo;
    }
    *s = sumf;
}
