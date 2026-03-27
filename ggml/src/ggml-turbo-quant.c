// TurboQuant CPU reference — exact paper algorithm (arXiv 2504.19874)
//
// TurboQuant_prod(b) = (b-1)-bit MSE quantizer + 1-bit QJL on residual.
// Two operating points, each with two independent instances (32 hi + 96 lo channels):
//   TURBO3_0: hi=b3(4 centroids) + lo=b2(2 centroids) → 2.5 bpv
//   TURBO4_0: hi=b4(8 centroids) + lo=b3(4 centroids) → 3.5 bpv
//
// Rotation (Π) applied at graph level, NOT here.
// QJL uses one fixed i.i.d. N(0,1) matrix per dimension (reused across all blocks).

#include "ggml-quants.h"
#include "ggml-impl.h"

#include <math.h>
#include <string.h>
#include <assert.h>

// ---------------------------------------------------------------------------
// Centroids: exact Lloyd-Max for Beta((d-1)/2, (d-1)/2), d=128
// ---------------------------------------------------------------------------

static const float centroids_8[8] = {
    -0.1883988281f, -0.1181421705f, -0.0665887043f, -0.0216082019f,
     0.0216082019f,  0.0665887043f,  0.1181421705f,  0.1883988281f,
};

static const float centroids_4[4] = {
    -0.1330458627f, -0.0399983984f, 0.0399983984f, 0.1330458627f,
};

static const float centroids_2[2] = {
    -0.0707250243f, 0.0707250243f,
};

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
// Paper: S is generated once during setup, reused for all blocks.
// We use a fixed seed per dimension so S is deterministic.

// Seeds: one per QJL dimension used in the system
#define QJL_SEED_32   0x514A4C20ULL  // "QJL " — for 32-dim instances
#define QJL_SEED_96   0x514A4C60ULL  // "QJL`" — for 96-dim instances
#define QJL_SEED_128  0x514A4C80ULL  // for 128-dim (if needed)

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

// ---------------------------------------------------------------------------
// Generic two-instance quantize/dequantize
//
// Both TURBO3 and TURBO4 have the same block layout:
//   norm, rnorm_hi, rnorm_lo, qs_hi[], qs_lo[], signs_hi[], signs_lo[]
//
// The only difference is the centroid tables and packing widths:
//   TURBO3: hi = 2-bit MSE (centroids_4), lo = 1-bit MSE (centroids_2)
//   TURBO4: hi = 3-bit MSE (centroids_8), lo = 2-bit MSE (centroids_4)
// ---------------------------------------------------------------------------

// Quantize hi channels with n_c centroids, packing at bits_per_idx bits
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

// Quantize lo channels
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

// Dequantize hi channels
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

// Dequantize lo channels
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

// ---------------------------------------------------------------------------
// TURBO3_0: hi=b3(centroids_4, 2-bit), lo=b2(centroids_2, 1-bit)
// ---------------------------------------------------------------------------

void quantize_row_turbo3_0_ref(const float * GGML_RESTRICT x, block_turbo3_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO3 == 0);
    const int64_t nb = k / QK_TURBO3;

    for (int64_t i = 0; i < nb; i++) {
        const float * xb = x + i * QK_TURBO3;
        float sum_sq = 0.0f;
        for (int j = 0; j < QK_TURBO3; j++) sum_sq += xb[j] * xb[j];
        float norm = sqrtf(sum_sq);

        y[i].norm = GGML_FP32_TO_FP16(norm);
        y[i].rnorm_hi = GGML_FP32_TO_FP16(0.0f);
        y[i].rnorm_lo = GGML_FP32_TO_FP16(0.0f);
        memset(y[i].signs_hi, 0, sizeof(y[i].signs_hi));
        memset(y[i].signs_lo, 0, sizeof(y[i].signs_lo));

        if (norm == 0.0f) { memset(y[i].qs_hi, 0, sizeof(y[i].qs_hi)); memset(y[i].qs_lo, 0, sizeof(y[i].qs_lo)); continue; }
        float inv = 1.0f / norm;

        float res_hi[QK_TURBO3_HI], res_lo[QK_TURBO3_LO];
        quant_hi(xb, inv, y[i].qs_hi, centroids_4, 4, 2, QK_TURBO3_HI, res_hi);
        quant_lo(xb, inv, y[i].qs_lo, centroids_2, 2, 1, QK_TURBO3_LO, QK_TURBO3_HI, res_lo);

        y[i].rnorm_hi = GGML_FP32_TO_FP16(qjl_forward(res_hi, y[i].signs_hi, QK_TURBO3_HI, QJL_SEED_32));
        y[i].rnorm_lo = GGML_FP32_TO_FP16(qjl_forward(res_lo, y[i].signs_lo, QK_TURBO3_LO, QJL_SEED_96));
    }
}

void dequantize_row_turbo3_0(const block_turbo3_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO3 == 0);
    const int64_t nb = k / QK_TURBO3;

    for (int64_t i = 0; i < nb; i++) {
        float norm = GGML_FP16_TO_FP32(x[i].norm);
        float * yb = y + i * QK_TURBO3;

        float ch[QK_TURBO3_HI], cl[QK_TURBO3_LO];
        qjl_inverse(x[i].signs_hi, GGML_FP16_TO_FP32(x[i].rnorm_hi), ch, QK_TURBO3_HI, QJL_SEED_32);
        qjl_inverse(x[i].signs_lo, GGML_FP16_TO_FP32(x[i].rnorm_lo), cl, QK_TURBO3_LO, QJL_SEED_96);

        dequant_hi(x[i].qs_hi, ch, centroids_4, 2, QK_TURBO3_HI, norm, yb);
        dequant_lo(x[i].qs_lo, cl, centroids_2, 1, QK_TURBO3_LO, norm, yb, QK_TURBO3_HI);
    }
}

// ---------------------------------------------------------------------------
// TURBO4_0: hi=b4(centroids_8, 3-bit), lo=b3(centroids_4, 2-bit)
// ---------------------------------------------------------------------------

void quantize_row_turbo4_0_ref(const float * GGML_RESTRICT x, block_turbo4_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO4 == 0);
    const int64_t nb = k / QK_TURBO4;

    for (int64_t i = 0; i < nb; i++) {
        const float * xb = x + i * QK_TURBO4;
        float sum_sq = 0.0f;
        for (int j = 0; j < QK_TURBO4; j++) sum_sq += xb[j] * xb[j];
        float norm = sqrtf(sum_sq);

        y[i].norm = GGML_FP32_TO_FP16(norm);
        y[i].rnorm_hi = GGML_FP32_TO_FP16(0.0f);
        y[i].rnorm_lo = GGML_FP32_TO_FP16(0.0f);
        memset(y[i].signs_hi, 0, sizeof(y[i].signs_hi));
        memset(y[i].signs_lo, 0, sizeof(y[i].signs_lo));

        if (norm == 0.0f) { memset(y[i].qs_hi, 0, sizeof(y[i].qs_hi)); memset(y[i].qs_lo, 0, sizeof(y[i].qs_lo)); continue; }
        float inv = 1.0f / norm;

        float res_hi[QK_TURBO4_HI], res_lo[QK_TURBO4_LO];
        quant_hi(xb, inv, y[i].qs_hi, centroids_8, 8, 3, QK_TURBO4_HI, res_hi);
        quant_lo(xb, inv, y[i].qs_lo, centroids_4, 4, 2, QK_TURBO4_LO, QK_TURBO4_HI, res_lo);

        y[i].rnorm_hi = GGML_FP32_TO_FP16(qjl_forward(res_hi, y[i].signs_hi, QK_TURBO4_HI, QJL_SEED_32));
        y[i].rnorm_lo = GGML_FP32_TO_FP16(qjl_forward(res_lo, y[i].signs_lo, QK_TURBO4_LO, QJL_SEED_96));
    }
}

void dequantize_row_turbo4_0(const block_turbo4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO4 == 0);
    const int64_t nb = k / QK_TURBO4;

    for (int64_t i = 0; i < nb; i++) {
        float norm = GGML_FP16_TO_FP32(x[i].norm);
        float * yb = y + i * QK_TURBO4;

        float ch[QK_TURBO4_HI], cl[QK_TURBO4_LO];
        qjl_inverse(x[i].signs_hi, GGML_FP16_TO_FP32(x[i].rnorm_hi), ch, QK_TURBO4_HI, QJL_SEED_32);
        qjl_inverse(x[i].signs_lo, GGML_FP16_TO_FP32(x[i].rnorm_lo), cl, QK_TURBO4_LO, QJL_SEED_96);

        dequant_hi(x[i].qs_hi, ch, centroids_8, 3, QK_TURBO4_HI, norm, yb);
        dequant_lo(x[i].qs_lo, cl, centroids_4, 2, QK_TURBO4_LO, norm, yb, QK_TURBO4_HI);
    }
}
