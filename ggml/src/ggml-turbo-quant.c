// TurboQuant CPU reference — exact paper algorithm (arXiv 2504.19874)
//
// TurboQuant_prod(b) = (b-1)-bit MSE quantizer + 1-bit QJL on residual.
// Two operating points, each with two independent instances (32 hi + 96 lo channels):
//   TURBO3_0: hi=b3(4 centroids) + lo=b2(2 centroids) → 2.5 bpv
//   TURBO4_0: hi=b4(8 centroids) + lo=b3(4 centroids) → 3.5 bpv
//
// Rotation (Π) is applied INSIDE quantize/dequantize (per the paper's Algorithm 1).
// QJL uses one fixed i.i.d. N(0,1) matrix per dimension (reused across all blocks).

#include "ggml-quants.h"
#include "ggml-impl.h"

#include <math.h>
#include <string.h>
#include <assert.h>

// ---------------------------------------------------------------------------
// Centroids: exact Lloyd-Max for Beta((d-1)/2, (d-1)/2)
// Each independent instance uses centroids for its own dimension d.
// ---------------------------------------------------------------------------

// d=32 (outlier instance)
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
static const float centroids_8[8] = {
    -0.1883988281f, -0.1181421705f, -0.0665887043f, -0.0216082019f,
     0.0216082019f,  0.0665887043f,  0.1181421705f,  0.1883988281f,
};
static const float centroids_16[16] = {
    -0.2376827302f, -0.1808574273f, -0.1418271941f, -0.1103094608f,
    -0.0828467454f, -0.0577864193f, -0.0341609484f, -0.0113059237f,
     0.0113059237f,  0.0341609484f,  0.0577864193f,  0.0828467454f,
     0.1103094608f,  0.1418271941f,  0.1808574273f,  0.2376827302f,
};
static const float centroids_4[4] = {
    -0.1330458627f, -0.0399983984f, 0.0399983984f, 0.1330458627f,
};
static const float centroids_2[2] = {
    -0.0707250243f, 0.0707250243f,
};

// ---------------------------------------------------------------------------
// Two independent rotation matrices: Π_hi (32×32) and Π_lo (96×96)
// Per the paper: split channels into outlier/regular sets, apply independent
// TurboQuant instances to each.
// ---------------------------------------------------------------------------

#define TQ_DIM     128
#define TQ_DIM_HI  32
#define TQ_DIM_LO  96

static float tq_rot_hi_fwd[TQ_DIM_HI * TQ_DIM_HI];  // Π_hi (row-major)
static float tq_rot_hi_inv[TQ_DIM_HI * TQ_DIM_HI];
static float tq_rot_lo_fwd[TQ_DIM_LO * TQ_DIM_LO];  // Π_lo (row-major)
static float tq_rot_lo_inv[TQ_DIM_LO * TQ_DIM_LO];

// Outlier channel indices (sorted): which 32 of 128 channels are outliers
static int   tq_outlier_ch[TQ_DIM_HI];
static int   tq_regular_ch[TQ_DIM_LO];
static int   tq_initialized = 0;
static int   tq_outliers_detected = 0;

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
    // Default outlier channels: first 32 (will be overridden by detection)
    for (int i = 0; i < TQ_DIM_HI; i++) tq_outlier_ch[i] = i;
    for (int i = 0; i < TQ_DIM_LO; i++) tq_regular_ch[i] = TQ_DIM_HI + i;
    tq_initialized = 1;
}

// Detect outlier channels from a 128-dim vector (by absolute magnitude)
// Called on the first vector; locks the channel assignment for all subsequent blocks.
static void tq_detect_outliers(const float * x) {
    if (tq_outliers_detected) return;

    // Sort channels by absolute magnitude
    int order[TQ_DIM];
    float mag[TQ_DIM];
    for (int i = 0; i < TQ_DIM; i++) { order[i] = i; mag[i] = fabsf(x[i]); }
    // Simple insertion sort (only 128 elements)
    for (int i = 1; i < TQ_DIM; i++) {
        int key_idx = order[i]; float key_mag = mag[key_idx];
        int j = i - 1;
        while (j >= 0 && mag[order[j]] < key_mag) { order[j+1] = order[j]; j--; }
        order[j+1] = key_idx;
    }
    // Top 32 by magnitude = outliers
    for (int i = 0; i < TQ_DIM_HI; i++) tq_outlier_ch[i] = order[i];
    // Sort outlier indices for consistent ordering
    for (int i = 1; i < TQ_DIM_HI; i++) {
        int key = tq_outlier_ch[i]; int j = i-1;
        while (j >= 0 && tq_outlier_ch[j] > key) { tq_outlier_ch[j+1] = tq_outlier_ch[j]; j--; }
        tq_outlier_ch[j+1] = key;
    }
    // Remaining = regular channels
    int ri = 0;
    for (int i = 0; i < TQ_DIM; i++) {
        int is_outlier = 0;
        for (int j = 0; j < TQ_DIM_HI; j++) { if (tq_outlier_ch[j] == i) { is_outlier = 1; break; } }
        if (!is_outlier) tq_regular_ch[ri++] = i;
    }
    tq_outliers_detected = 1;
}

// Extract hi/lo channel subsets from a 128-dim vector
static void tq_split_channels(const float * x, float * hi, float * lo) {
    for (int i = 0; i < TQ_DIM_HI; i++) hi[i] = x[tq_outlier_ch[i]];
    for (int i = 0; i < TQ_DIM_LO; i++) lo[i] = x[tq_regular_ch[i]];
}

// Merge hi/lo channel subsets back into 128-dim vector
static void tq_merge_channels(const float * hi, const float * lo, float * x) {
    for (int i = 0; i < TQ_DIM_HI; i++) x[tq_outlier_ch[i]] = hi[i];
    for (int i = 0; i < TQ_DIM_LO; i++) x[tq_regular_ch[i]] = lo[i];
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
// PRNG for i.i.d. Gaussian QJL matrices
// ---------------------------------------------------------------------------

static uint64_t tq_prng;
static void tq_seed(uint64_t s) { tq_prng = s; }
static float tq_gaussian(void) {
    tq_prng = tq_prng * 6364136223846793005ULL + 1442695040888963407ULL;
    float u1 = ((float)(uint32_t)(tq_prng >> 32) + 1.0f) / ((float)0xFFFFFFFF + 2.0f);
    tq_prng = tq_prng * 6364136223846793005ULL + 1442695040888963407ULL;
    float u2 = ((float)(uint32_t)(tq_prng >> 32) + 1.0f) / ((float)0xFFFFFFFF + 2.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

// ---------------------------------------------------------------------------
// QJL (paper Definition 1): one fixed i.i.d. Gaussian S per dimension
// ---------------------------------------------------------------------------

#define QJL_SEED_32   0x514A4C20ULL
#define QJL_SEED_96   0x514A4C60ULL
#define QJL_SEED_128  0x514A4C80ULL

static float qjl_forward(const float * r, uint8_t * signs, int m, uint64_t seed) {
    tq_seed(seed);
    float rnorm_sq = 0.0f;
    for (int j = 0; j < m; j++) rnorm_sq += r[j] * r[j];

    memset(signs, 0, (m + 7) / 8);
    for (int i = 0; i < m; i++) {
        float proj = 0.0f;
        for (int j = 0; j < m; j++) proj += tq_gaussian() * r[j];
        if (proj >= 0.0f) signs[i / 8] |= (uint8_t)(1 << (i % 8));
    }
    return sqrtf(rnorm_sq);
}

static void qjl_inverse(const uint8_t * signs, float rnorm, float * corr, int m, uint64_t seed) {
    memset(corr, 0, m * sizeof(float));
    tq_seed(seed);
    for (int i = 0; i < m; i++) {
        float z = ((signs[i / 8] >> (i % 8)) & 1) ? 1.0f : -1.0f;
        for (int j = 0; j < m; j++) corr[j] += tq_gaussian() * z;
    }
    float scale = 1.2533141f / (float)m * rnorm;
    for (int j = 0; j < m; j++) corr[j] *= scale;
}

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
// TQK 2.5: hi=b3(centroids_4, 2-bit), lo=b2(centroids_2, 1-bit)
// ---------------------------------------------------------------------------

void quantize_row_tqk_25_ref(const float * GGML_RESTRICT x, block_tqk_25 * GGML_RESTRICT y, int64_t k) {
    assert(k % TQK_BLOCK_SIZE == 0);
    const int64_t nb = k / TQK_BLOCK_SIZE;
    tq_init_rotations();

    for (int64_t i = 0; i < nb; i++) {
        const float * xb = x + i * TQK_BLOCK_SIZE;

        // Detect outlier channels from first vector
        tq_detect_outliers(xb);

        // Split into outlier (hi) and regular (lo) channel subsets
        float hi_raw[TQK_N_OUTLIER], lo_raw[TQK_N_REGULAR];
        tq_split_channels(xb, hi_raw, lo_raw);

        // Rotate each subset independently
        float hi_rot[TQK_N_OUTLIER], lo_rot[TQK_N_REGULAR];
        tq_rotate_hi(hi_raw, hi_rot);
        tq_rotate_lo(lo_raw, lo_rot);

        // Per-subset norms
        float sum_hi = 0.0f, sum_lo = 0.0f;
        for (int j = 0; j < TQK_N_OUTLIER; j++) sum_hi += hi_rot[j] * hi_rot[j];
        for (int j = 0; j < TQK_N_REGULAR; j++) sum_lo += lo_rot[j] * lo_rot[j];
        float norm_hi = sqrtf(sum_hi), norm_lo = sqrtf(sum_lo);

        y[i].norm_hi = GGML_FP32_TO_FP16(norm_hi);
        y[i].norm_lo = GGML_FP32_TO_FP16(norm_lo);
        y[i].rnorm_hi = GGML_FP32_TO_FP16(0.0f);
        y[i].rnorm_lo = GGML_FP32_TO_FP16(0.0f);
        memset(y[i].signs_hi, 0, sizeof(y[i].signs_hi));
        memset(y[i].signs_lo, 0, sizeof(y[i].signs_lo));

        if (norm_hi == 0.0f && norm_lo == 0.0f) { memset(y[i].qs_hi, 0, sizeof(y[i].qs_hi)); memset(y[i].qs_lo, 0, sizeof(y[i].qs_lo)); continue; }

        float inv_hi = (norm_hi > 1e-12f) ? 1.0f / norm_hi : 0.0f;
        float inv_lo = (norm_lo > 1e-12f) ? 1.0f / norm_lo : 0.0f;

        float res_hi[TQK_N_OUTLIER], res_lo[TQK_N_REGULAR];
        quant_hi(hi_rot, inv_hi, y[i].qs_hi, centroids_4_d32, 4, 2, TQK_N_OUTLIER, res_hi);
        quant_lo(lo_rot, inv_lo, y[i].qs_lo, centroids_2_d96, 2, 1, TQK_N_REGULAR, 0, res_lo);

        y[i].rnorm_hi = GGML_FP32_TO_FP16(qjl_forward(res_hi, y[i].signs_hi, TQK_N_OUTLIER, QJL_SEED_32));
        y[i].rnorm_lo = GGML_FP32_TO_FP16(qjl_forward(res_lo, y[i].signs_lo, TQK_N_REGULAR, QJL_SEED_96));
    }
}

void dequantize_row_tqk_25(const block_tqk_25 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % TQK_BLOCK_SIZE == 0);
    const int64_t nb = k / TQK_BLOCK_SIZE;
    tq_init_rotations();

    for (int64_t i = 0; i < nb; i++) {
        float norm_hi = GGML_FP16_TO_FP32(x[i].norm_hi);
        float norm_lo = GGML_FP16_TO_FP32(x[i].norm_lo);

        float corr_hi[TQK_N_OUTLIER], corr_lo[TQK_N_REGULAR];
        qjl_inverse(x[i].signs_hi, GGML_FP16_TO_FP32(x[i].rnorm_hi), corr_hi, TQK_N_OUTLIER, QJL_SEED_32);
        qjl_inverse(x[i].signs_lo, GGML_FP16_TO_FP32(x[i].rnorm_lo), corr_lo, TQK_N_REGULAR, QJL_SEED_96);

        float hi_rot[TQK_N_OUTLIER], lo_rot[TQK_N_REGULAR];
        dequant_hi(x[i].qs_hi, corr_hi, centroids_4_d32, 2, TQK_N_OUTLIER, norm_hi, hi_rot);
        dequant_lo(x[i].qs_lo, corr_lo, centroids_2_d96, 1, TQK_N_REGULAR, norm_lo, lo_rot, 0);

        // Unrotate each subset and merge back
        float hi_orig[TQK_N_OUTLIER], lo_orig[TQK_N_REGULAR];
        tq_unrotate_hi(hi_rot, hi_orig);
        tq_unrotate_lo(lo_rot, lo_orig);
        tq_merge_channels(hi_orig, lo_orig, y + i * TQK_BLOCK_SIZE);
    }
}

// ---------------------------------------------------------------------------
// TQK 3.5: independent instances — hi=3-bit MSE (d=32), lo=2-bit MSE (d=96)
// ---------------------------------------------------------------------------

void quantize_row_tqk_35_ref(const float * GGML_RESTRICT x, block_tqk_35 * GGML_RESTRICT y, int64_t k) {
    assert(k % TQK_BLOCK_SIZE == 0);
    const int64_t nb = k / TQK_BLOCK_SIZE;
    tq_init_rotations();

    for (int64_t i = 0; i < nb; i++) {
        const float * xb = x + i * TQK_BLOCK_SIZE;

        tq_detect_outliers(xb);

        float hi_raw[TQK_N_OUTLIER], lo_raw[TQK_N_REGULAR];
        tq_split_channels(xb, hi_raw, lo_raw);

        float hi_rot[TQK_N_OUTLIER], lo_rot[TQK_N_REGULAR];
        tq_rotate_hi(hi_raw, hi_rot);
        tq_rotate_lo(lo_raw, lo_rot);

        float sum_hi = 0.0f, sum_lo = 0.0f;
        for (int j = 0; j < TQK_N_OUTLIER; j++) sum_hi += hi_rot[j] * hi_rot[j];
        for (int j = 0; j < TQK_N_REGULAR; j++) sum_lo += lo_rot[j] * lo_rot[j];
        float norm_hi = sqrtf(sum_hi), norm_lo = sqrtf(sum_lo);

        y[i].norm_hi = GGML_FP32_TO_FP16(norm_hi);
        y[i].norm_lo = GGML_FP32_TO_FP16(norm_lo);
        y[i].rnorm_hi = GGML_FP32_TO_FP16(0.0f);
        y[i].rnorm_lo = GGML_FP32_TO_FP16(0.0f);
        memset(y[i].signs_hi, 0, sizeof(y[i].signs_hi));
        memset(y[i].signs_lo, 0, sizeof(y[i].signs_lo));

        if (norm_hi == 0.0f && norm_lo == 0.0f) { memset(y[i].qs_hi, 0, sizeof(y[i].qs_hi)); memset(y[i].qs_lo, 0, sizeof(y[i].qs_lo)); continue; }

        float inv_hi = (norm_hi > 1e-12f) ? 1.0f / norm_hi : 0.0f;
        float inv_lo = (norm_lo > 1e-12f) ? 1.0f / norm_lo : 0.0f;

        float res_hi[TQK_N_OUTLIER], res_lo[TQK_N_REGULAR];
        quant_hi(hi_rot, inv_hi, y[i].qs_hi, centroids_8_d32, 8, 3, TQK_N_OUTLIER, res_hi);
        quant_lo(lo_rot, inv_lo, y[i].qs_lo, centroids_4_d96, 4, 2, TQK_N_REGULAR, 0, res_lo);

        y[i].rnorm_hi = GGML_FP32_TO_FP16(qjl_forward(res_hi, y[i].signs_hi, TQK_N_OUTLIER, QJL_SEED_32));
        y[i].rnorm_lo = GGML_FP32_TO_FP16(qjl_forward(res_lo, y[i].signs_lo, TQK_N_REGULAR, QJL_SEED_96));
    }
}

void dequantize_row_tqk_35(const block_tqk_35 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % TQK_BLOCK_SIZE == 0);
    const int64_t nb = k / TQK_BLOCK_SIZE;
    tq_init_rotations();

    for (int64_t i = 0; i < nb; i++) {
        float norm_hi = GGML_FP16_TO_FP32(x[i].norm_hi);
        float norm_lo = GGML_FP16_TO_FP32(x[i].norm_lo);

        float corr_hi[TQK_N_OUTLIER], corr_lo[TQK_N_REGULAR];
        qjl_inverse(x[i].signs_hi, GGML_FP16_TO_FP32(x[i].rnorm_hi), corr_hi, TQK_N_OUTLIER, QJL_SEED_32);
        qjl_inverse(x[i].signs_lo, GGML_FP16_TO_FP32(x[i].rnorm_lo), corr_lo, TQK_N_REGULAR, QJL_SEED_96);

        float hi_rot[TQK_N_OUTLIER], lo_rot[TQK_N_REGULAR];
        dequant_hi(x[i].qs_hi, corr_hi, centroids_8_d32, 3, TQK_N_OUTLIER, norm_hi, hi_rot);
        dequant_lo(x[i].qs_lo, corr_lo, centroids_4_d96, 2, TQK_N_REGULAR, norm_lo, lo_rot, 0);

        float hi_orig[TQK_N_OUTLIER], lo_orig[TQK_N_REGULAR];
        tq_unrotate_hi(hi_rot, hi_orig);
        tq_unrotate_lo(lo_rot, lo_orig);
        tq_merge_channels(hi_orig, lo_orig, y + i * TQK_BLOCK_SIZE);
    }
}

// ---------------------------------------------------------------------------
// TQV 2.5: hi=3-bit MSE (centroids_8_d32), lo=2-bit MSE (centroids_4_d96)
// Pure MSE reconstruction — no QJL. Ideal for V cache.
// ---------------------------------------------------------------------------

void quantize_row_tqv_25_ref(const float * GGML_RESTRICT x, block_tqv_25 * GGML_RESTRICT y, int64_t k) {
    assert(k % TQV_BLOCK_SIZE == 0);
    const int64_t nb = k / TQV_BLOCK_SIZE;
    tq_init_rotations();

    for (int64_t i = 0; i < nb; i++) {
        const float * xb_orig = x + i * TQV_BLOCK_SIZE;

        // Detect outlier channels from first vector
        tq_detect_outliers(xb_orig);

        // Split into outlier (hi) and regular (lo) channel subsets
        float hi_raw[TQV_N_OUTLIER], lo_raw[TQV_N_REGULAR];
        tq_split_channels(xb_orig, hi_raw, lo_raw);

        // Rotate each subset independently
        float hi_rot[TQV_N_OUTLIER], lo_rot[TQV_N_REGULAR];
        tq_rotate_hi(hi_raw, hi_rot);
        tq_rotate_lo(lo_raw, lo_rot);

        float sum_hi = 0.0f, sum_lo = 0.0f;
        for (int j = 0; j < TQV_N_OUTLIER; j++) sum_hi += hi_rot[j] * hi_rot[j];
        for (int j = 0; j < TQV_N_REGULAR; j++) sum_lo += lo_rot[j] * lo_rot[j];
        float norm_hi = sqrtf(sum_hi), norm_lo = sqrtf(sum_lo);

        y[i].norm_hi = GGML_FP32_TO_FP16(norm_hi);
        y[i].norm_lo = GGML_FP32_TO_FP16(norm_lo);

        if (norm_hi == 0.0f && norm_lo == 0.0f) { memset(y[i].qs_hi, 0, sizeof(y[i].qs_hi)); memset(y[i].qs_lo, 0, sizeof(y[i].qs_lo)); continue; }
        float inv_hi = (norm_hi > 1e-12f) ? 1.0f / norm_hi : 0.0f;
        float inv_lo = (norm_lo > 1e-12f) ? 1.0f / norm_lo : 0.0f;

        quant_hi_mse(hi_rot, inv_hi, y[i].qs_hi, centroids_8_d32, 8, 3, TQV_N_OUTLIER);
        quant_lo_mse(lo_rot, inv_lo, y[i].qs_lo, centroids_4_d96, 4, 2, TQV_N_REGULAR, 0);
    }
}

void dequantize_row_tqv_25(const block_tqv_25 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % TQV_BLOCK_SIZE == 0);
    const int64_t nb = k / TQV_BLOCK_SIZE;
    tq_init_rotations();

    for (int64_t i = 0; i < nb; i++) {
        float norm_hi = GGML_FP16_TO_FP32(x[i].norm_hi);
        float norm_lo = GGML_FP16_TO_FP32(x[i].norm_lo);

        float hi_rot[TQV_N_OUTLIER], lo_rot[TQV_N_REGULAR];
        dequant_hi_mse(x[i].qs_hi, centroids_8_d32, 3, TQV_N_OUTLIER, norm_hi, hi_rot);
        dequant_lo_mse(x[i].qs_lo, centroids_4_d96, 2, TQV_N_REGULAR, norm_lo, lo_rot, 0);

        // Unrotate each subset and merge back
        float hi_orig[TQV_N_OUTLIER], lo_orig[TQV_N_REGULAR];
        tq_unrotate_hi(hi_rot, hi_orig);
        tq_unrotate_lo(lo_rot, lo_orig);
        tq_merge_channels(hi_orig, lo_orig, y + i * TQV_BLOCK_SIZE);
    }
}

// ---------------------------------------------------------------------------
// TQV 3.5: hi=4-bit MSE (centroids_16, d=128 approx), lo=3-bit MSE (centroids_8_d96)
// Pure MSE reconstruction — no QJL. Ideal for V cache.
// ---------------------------------------------------------------------------

void quantize_row_tqv_35_ref(const float * GGML_RESTRICT x, block_tqv_35 * GGML_RESTRICT y, int64_t k) {
    assert(k % TQV_BLOCK_SIZE == 0);
    const int64_t nb = k / TQV_BLOCK_SIZE;
    tq_init_rotations();

    for (int64_t i = 0; i < nb; i++) {
        const float * xb_orig = x + i * TQV_BLOCK_SIZE;

        // Detect outlier channels from first vector
        tq_detect_outliers(xb_orig);

        // Split into outlier (hi) and regular (lo) channel subsets
        float hi_raw[TQV_N_OUTLIER], lo_raw[TQV_N_REGULAR];
        tq_split_channels(xb_orig, hi_raw, lo_raw);

        // Rotate each subset independently
        float hi_rot[TQV_N_OUTLIER], lo_rot[TQV_N_REGULAR];
        tq_rotate_hi(hi_raw, hi_rot);
        tq_rotate_lo(lo_raw, lo_rot);

        float sum_hi = 0.0f, sum_lo = 0.0f;
        for (int j = 0; j < TQV_N_OUTLIER; j++) sum_hi += hi_rot[j] * hi_rot[j];
        for (int j = 0; j < TQV_N_REGULAR; j++) sum_lo += lo_rot[j] * lo_rot[j];
        float norm_hi = sqrtf(sum_hi), norm_lo = sqrtf(sum_lo);

        y[i].norm_hi = GGML_FP32_TO_FP16(norm_hi);
        y[i].norm_lo = GGML_FP32_TO_FP16(norm_lo);

        if (norm_hi == 0.0f && norm_lo == 0.0f) { memset(y[i].qs_hi, 0, sizeof(y[i].qs_hi)); memset(y[i].qs_lo, 0, sizeof(y[i].qs_lo)); continue; }
        float inv_hi = (norm_hi > 1e-12f) ? 1.0f / norm_hi : 0.0f;
        float inv_lo = (norm_lo > 1e-12f) ? 1.0f / norm_lo : 0.0f;

        // NOTE: centroids_16 is d=128 approximation; no d=32 table for 4-bit yet
        quant_hi_mse(hi_rot, inv_hi, y[i].qs_hi, centroids_16, 16, 4, TQV_N_OUTLIER);
        quant_lo_mse(lo_rot, inv_lo, y[i].qs_lo, centroids_8_d96, 8, 3, TQV_N_REGULAR, 0);
    }
}

void dequantize_row_tqv_35(const block_tqv_35 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % TQV_BLOCK_SIZE == 0);
    const int64_t nb = k / TQV_BLOCK_SIZE;
    tq_init_rotations();

    for (int64_t i = 0; i < nb; i++) {
        float norm_hi = GGML_FP16_TO_FP32(x[i].norm_hi);
        float norm_lo = GGML_FP16_TO_FP32(x[i].norm_lo);

        float hi_rot[TQV_N_OUTLIER], lo_rot[TQV_N_REGULAR];
        // NOTE: centroids_16 is d=128 approximation
        dequant_hi_mse(x[i].qs_hi, centroids_16, 4, TQV_N_OUTLIER, norm_hi, hi_rot);
        dequant_lo_mse(x[i].qs_lo, centroids_8_d96, 3, TQV_N_REGULAR, norm_lo, lo_rot, 0);

        // Unrotate each subset and merge back
        float hi_orig[TQV_N_OUTLIER], lo_orig[TQV_N_REGULAR];
        tq_unrotate_hi(hi_rot, hi_orig);
        tq_unrotate_lo(lo_rot, lo_orig);
        tq_merge_channels(hi_orig, lo_orig, y + i * TQV_BLOCK_SIZE);
    }
}

// ---------------------------------------------------------------------------
// Asymmetric inner product estimator (paper Algorithm 2)
//
// <q, k> ≈ <q_rot, k_mse> + rnorm * sqrt(π/2) / m * <S @ q_rot, sign(S @ r_k)>
//
// where k_mse is the centroid-only reconstruction in rotated space,
// sign(S @ r_k) are the stored QJL sign bits, and rnorm = ||r_k||.
// The query q is in original space; we rotate it to match the stored data.
//
// This estimator is UNBIASED (E[estimate] = <q, k>) unlike the
// dequantize-then-dot approach which adds QJL reconstruction noise.
// ---------------------------------------------------------------------------

// Compute <S_row_i, q_rot> for the QJL projection of the query
// Uses the same deterministic PRNG as qjl_forward/qjl_inverse
static float qjl_project_query_element(const float * q_rot, int i, int m, uint64_t seed) {
    // Fast-forward PRNG to the right position: need to skip i*m Gaussian draws
    // to get the i-th row of S, then dot with q_rot[0..m-1]
    uint64_t st = seed;
    // Skip to row i: each row generates m Gaussians, each takes 2 PRNG steps
    for (int skip = 0; skip < i * m; skip++) {
        st = st * 6364136223846793005ULL + 1442695040888963407ULL;
        st = st * 6364136223846793005ULL + 1442695040888963407ULL;
    }
    // Now generate row i and dot with q_rot
    float proj = 0.0f;
    for (int j = 0; j < m; j++) {
        st = st * 6364136223846793005ULL + 1442695040888963407ULL;
        float u1 = ((float)(uint32_t)(st >> 32) + 1.0f) / ((float)0xFFFFFFFF + 2.0f);
        st = st * 6364136223846793005ULL + 1442695040888963407ULL;
        float u2 = ((float)(uint32_t)(st >> 32) + 1.0f) / ((float)0xFFFFFFFF + 2.0f);
        float g = sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
        proj += g * q_rot[j];
    }
    return proj;
}

// QJL correction term: rnorm * sqrt(π/2) / m * <S @ q_rot, sign(S @ r_k)>
// = rnorm * sqrt(π/2) / m * sum_i (S_i @ q_rot) * sign_i
static float qjl_asymmetric_dot(const float * q_rot, int m, uint64_t seed,
                                 const uint8_t * signs, float rnorm) {
    if (rnorm == 0.0f) return 0.0f;

    // Generate S @ q_rot on-the-fly (same PRNG as qjl_forward)
    // and dot with stored signs
    tq_seed(seed);
    float sum = 0.0f;
    for (int i = 0; i < m; i++) {
        // Compute (S_row_i @ q_rot)
        float proj = 0.0f;
        for (int j = 0; j < m; j++) {
            proj += tq_gaussian() * q_rot[j];
        }
        // Multiply by stored sign bit
        float sign = ((signs[i / 8] >> (i % 8)) & 1) ? 1.0f : -1.0f;
        sum += proj * sign;
    }
    // Scale: sqrt(π/2) / m * rnorm
    return 1.2533141f / (float)m * rnorm * sum;
}

// TQK 2.5 asymmetric vec_dot: key=tqk_25, query=f32
void ggml_vec_dot_tqk_25_f32(
        int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, size_t bx,
        const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % TQK_BLOCK_SIZE == 0);
    assert(nrc == 1);
    (void)bs; (void)bx; (void)by; (void)nrc;

    tq_init_rotations();

    const block_tqk_25 * GGML_RESTRICT x = (const block_tqk_25 *)vx;
    const float * GGML_RESTRICT y = (const float *)vy;
    const int64_t nb = n / TQK_BLOCK_SIZE;

    float sumf = 0.0f;

    for (int64_t i = 0; i < nb; i++) {
        const float * q = y + i * TQK_BLOCK_SIZE;
        float norm_hi = GGML_FP16_TO_FP32(x[i].norm_hi);
        float norm_lo = GGML_FP16_TO_FP32(x[i].norm_lo);

        // Split query by outlier channels, rotate each subset
        tq_detect_outliers(q);
        float hi_raw[TQK_N_OUTLIER], lo_raw[TQK_N_REGULAR];
        tq_split_channels(q, hi_raw, lo_raw);
        float q_rot_hi[TQK_N_OUTLIER], q_rot_lo[TQK_N_REGULAR];
        tq_rotate_hi(hi_raw, q_rot_hi);
        tq_rotate_lo(lo_raw, q_rot_lo);

        // MSE centroid dot product per subset
        float mse_dot_hi = 0.0f, mse_dot_lo = 0.0f;
        for (int j = 0; j < TQK_N_OUTLIER; j++) {
            int idx = (x[i].qs_hi[j / 4] >> ((j % 4) * 2)) & 0x3;
            mse_dot_hi += q_rot_hi[j] * centroids_4_d32[idx];
        }
        for (int j = 0; j < TQK_N_REGULAR; j++) {
            int idx = (x[i].qs_lo[j / 8] >> (j % 8)) & 1;
            mse_dot_lo += q_rot_lo[j] * centroids_2_d96[idx];
        }
        float mse_dot = mse_dot_hi * norm_hi + mse_dot_lo * norm_lo;

        // Step 3: split QJL correction (independent per partition)
        float qjl_hi = qjl_asymmetric_dot(q_rot_hi, TQK_N_OUTLIER, QJL_SEED_32,
                                            x[i].signs_hi, GGML_FP16_TO_FP32(x[i].rnorm_hi));
        float qjl_lo = qjl_asymmetric_dot(q_rot_lo, TQK_N_REGULAR, QJL_SEED_96,
                                            x[i].signs_lo, GGML_FP16_TO_FP32(x[i].rnorm_lo));

        sumf += mse_dot + norm_hi * qjl_hi + norm_lo * qjl_lo;
    }

    *s = sumf;
}

// TQK 3.5 asymmetric vec_dot: key=tqk_35, query=f32
void ggml_vec_dot_tqk_35_f32(
        int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, size_t bx,
        const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % TQK_BLOCK_SIZE == 0);
    assert(nrc == 1);
    (void)bs; (void)bx; (void)by; (void)nrc;

    tq_init_rotations();

    const block_tqk_35 * GGML_RESTRICT x = (const block_tqk_35 *)vx;
    const float * GGML_RESTRICT y = (const float *)vy;
    const int64_t nb = n / TQK_BLOCK_SIZE;

    float sumf = 0.0f;

    for (int64_t i = 0; i < nb; i++) {
        const float * q = y + i * TQK_BLOCK_SIZE;
        float norm_hi = GGML_FP16_TO_FP32(x[i].norm_hi);
        float norm_lo = GGML_FP16_TO_FP32(x[i].norm_lo);

        tq_detect_outliers(q);
        float hi_raw[TQK_N_OUTLIER], lo_raw[TQK_N_REGULAR];
        tq_split_channels(q, hi_raw, lo_raw);
        float q_rot_hi[TQK_N_OUTLIER], q_rot_lo[TQK_N_REGULAR];
        tq_rotate_hi(hi_raw, q_rot_hi);
        tq_rotate_lo(lo_raw, q_rot_lo);

        float mse_dot_hi = 0.0f, mse_dot_lo = 0.0f;
        for (int j = 0; j < TQK_N_OUTLIER; j++) {
            int idx = up3(x[i].qs_hi, j);
            mse_dot_hi += q_rot_hi[j] * centroids_8_d32[idx];
        }
        for (int j = 0; j < TQK_N_REGULAR; j++) {
            int idx = (x[i].qs_lo[j / 4] >> ((j % 4) * 2)) & 0x3;
            mse_dot_lo += q_rot_lo[j] * centroids_4_d96[idx];
        }
        float mse_dot = mse_dot_hi * norm_hi + mse_dot_lo * norm_lo;

        float qjl_hi = qjl_asymmetric_dot(q_rot_hi, TQK_N_OUTLIER, QJL_SEED_32,
                                            x[i].signs_hi, GGML_FP16_TO_FP32(x[i].rnorm_hi));
        float qjl_lo = qjl_asymmetric_dot(q_rot_lo, TQK_N_REGULAR, QJL_SEED_96,
                                            x[i].signs_lo, GGML_FP16_TO_FP32(x[i].rnorm_lo));

        sumf += mse_dot + norm_hi * qjl_hi + norm_lo * qjl_lo;
    }

    *s = sumf;
}
