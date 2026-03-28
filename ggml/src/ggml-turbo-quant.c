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
// Centroids: exact Lloyd-Max for Beta((d-1)/2, (d-1)/2), d=128
// ---------------------------------------------------------------------------

static const float centroids_8[8] = {
    -0.1883988281f, -0.1181421705f, -0.0665887043f, -0.0216082019f,
     0.0216082019f,  0.0665887043f,  0.1181421705f,  0.1883988281f,
};

static const float centroids_16[16] = {  // 4-bit MSE, d=128
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
// Rotation matrix Π (128×128 orthogonal via QR of Gaussian)
// Generated once from a deterministic seed, shared across all blocks.
// ---------------------------------------------------------------------------

#define TQ_DIM 128

static float tq_rot_fwd[TQ_DIM * TQ_DIM];  // Π (row-major)
static float tq_rot_inv[TQ_DIM * TQ_DIM];  // Π^T (row-major)
static int   tq_rot_initialized = 0;

static void tq_init_rotation(void) {
    if (tq_rot_initialized) return;

    // Deterministic PRNG for reproducibility
    uint64_t seed = 0x5475524230524F54ULL;  // "TuRB0ROT"
    #define TQ_ROT_LCG(s) ((s) * 6364136223846793005ULL + 1442695040888963407ULL)

    float gaussian[TQ_DIM * TQ_DIM];
    for (int i = 0; i < TQ_DIM * TQ_DIM; i++) {
        seed = TQ_ROT_LCG(seed);
        float u1 = ((float)(uint32_t)(seed >> 32) + 1.0f) / ((float)0xFFFFFFFF + 2.0f);
        seed = TQ_ROT_LCG(seed);
        float u2 = ((float)(uint32_t)(seed >> 32) + 1.0f) / ((float)0xFFFFFFFF + 2.0f);
        gaussian[i] = sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
    }

    // QR decomposition via modified Gram-Schmidt (column-major in gaussian)
    // Result stored row-major in tq_rot_fwd
    float Q[TQ_DIM * TQ_DIM];
    memcpy(Q, gaussian, sizeof(Q));

    for (int j = 0; j < TQ_DIM; j++) {
        // Orthogonalize column j against previous columns
        for (int k = 0; k < j; k++) {
            float dot = 0.0f;
            for (int i = 0; i < TQ_DIM; i++) {
                dot += Q[k * TQ_DIM + i] * Q[j * TQ_DIM + i];
            }
            for (int i = 0; i < TQ_DIM; i++) {
                Q[j * TQ_DIM + i] -= dot * Q[k * TQ_DIM + i];
            }
        }
        // Normalize column j
        float norm = 0.0f;
        for (int i = 0; i < TQ_DIM; i++) {
            norm += Q[j * TQ_DIM + i] * Q[j * TQ_DIM + i];
        }
        norm = sqrtf(norm);
        for (int i = 0; i < TQ_DIM; i++) {
            Q[j * TQ_DIM + i] /= norm;
        }
    }

    // Q is column-major: Q[col*DIM + row]. Convert to row-major for tq_rot_fwd.
    // tq_rot_fwd[i][j] = Q[j*DIM + i] (column j, row i)
    for (int i = 0; i < TQ_DIM; i++) {
        for (int j = 0; j < TQ_DIM; j++) {
            tq_rot_fwd[i * TQ_DIM + j] = Q[j * TQ_DIM + i];
            tq_rot_inv[j * TQ_DIM + i] = Q[j * TQ_DIM + i];  // transpose
        }
    }

    tq_rot_initialized = 1;
    #undef TQ_ROT_LCG
}

// out[i] = sum_j Π[i][j] * in[j]
static void tq_rotate(const float * in, float * out) {
    for (int i = 0; i < TQ_DIM; i++) {
        float sum = 0.0f;
        for (int j = 0; j < TQ_DIM; j++) {
            sum += tq_rot_fwd[i * TQ_DIM + j] * in[j];
        }
        out[i] = sum;
    }
}

// out[i] = sum_j Π^T[i][j] * in[j]
static void tq_unrotate(const float * in, float * out) {
    for (int i = 0; i < TQ_DIM; i++) {
        float sum = 0.0f;
        for (int j = 0; j < TQ_DIM; j++) {
            sum += tq_rot_inv[i * TQ_DIM + j] * in[j];
        }
        out[i] = sum;
    }
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
// TURBO3_0_PROD: hi=b3(centroids_4, 2-bit), lo=b2(centroids_2, 1-bit)
// ---------------------------------------------------------------------------

void quantize_row_turbo3_0_prod_ref(const float * GGML_RESTRICT x, block_turbo3_0_prod * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO3_PROD == 0);
    const int64_t nb = k / QK_TURBO3_PROD;
    tq_init_rotation();

    for (int64_t i = 0; i < nb; i++) {
        const float * xb_orig = x + i * QK_TURBO3_PROD;

        // Step 1: rotate
        float xb[QK_TURBO3_PROD];
        tq_rotate(xb_orig, xb);

        // Step 2: compute norm (invariant under rotation)
        float sum_sq = 0.0f;
        for (int j = 0; j < QK_TURBO3_PROD; j++) sum_sq += xb[j] * xb[j];
        float norm = sqrtf(sum_sq);

        y[i].norm = GGML_FP32_TO_FP16(norm);
        y[i].rnorm_hi = GGML_FP32_TO_FP16(0.0f);
        y[i].rnorm_lo = GGML_FP32_TO_FP16(0.0f);
        memset(y[i].signs_hi, 0, sizeof(y[i].signs_hi));
        memset(y[i].signs_lo, 0, sizeof(y[i].signs_lo));

        if (norm == 0.0f) { memset(y[i].qs_hi, 0, sizeof(y[i].qs_hi)); memset(y[i].qs_lo, 0, sizeof(y[i].qs_lo)); continue; }
        float inv = 1.0f / norm;

        // Step 3: quantize in rotated space
        float res_hi[QK_TURBO3_PROD_HI], res_lo[QK_TURBO3_PROD_LO];
        quant_hi(xb, inv, y[i].qs_hi, centroids_4, 4, 2, QK_TURBO3_PROD_HI, res_hi);
        quant_lo(xb, inv, y[i].qs_lo, centroids_2, 2, 1, QK_TURBO3_PROD_LO, QK_TURBO3_PROD_HI, res_lo);

        y[i].rnorm_hi = GGML_FP32_TO_FP16(qjl_forward(res_hi, y[i].signs_hi, QK_TURBO3_PROD_HI, QJL_SEED_32));
        y[i].rnorm_lo = GGML_FP32_TO_FP16(qjl_forward(res_lo, y[i].signs_lo, QK_TURBO3_PROD_LO, QJL_SEED_96));
    }
}

void dequantize_row_turbo3_0_prod(const block_turbo3_0_prod * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO3_PROD == 0);
    const int64_t nb = k / QK_TURBO3_PROD;
    tq_init_rotation();

    for (int64_t i = 0; i < nb; i++) {
        float norm = GGML_FP16_TO_FP32(x[i].norm);

        // Dequantize in rotated space
        float rotated[QK_TURBO3_PROD];
        float ch[QK_TURBO3_PROD_HI], cl[QK_TURBO3_PROD_LO];
        qjl_inverse(x[i].signs_hi, GGML_FP16_TO_FP32(x[i].rnorm_hi), ch, QK_TURBO3_PROD_HI, QJL_SEED_32);
        qjl_inverse(x[i].signs_lo, GGML_FP16_TO_FP32(x[i].rnorm_lo), cl, QK_TURBO3_PROD_LO, QJL_SEED_96);

        dequant_hi(x[i].qs_hi, ch, centroids_4, 2, QK_TURBO3_PROD_HI, norm, rotated);
        dequant_lo(x[i].qs_lo, cl, centroids_2, 1, QK_TURBO3_PROD_LO, norm, rotated, QK_TURBO3_PROD_HI);

        // Inverse rotate back to original space
        tq_unrotate(rotated, y + i * QK_TURBO3_PROD);
    }
}

// ---------------------------------------------------------------------------
// TURBO4_0_PROD: hi=b4(centroids_8, 3-bit), lo=b3(centroids_4, 2-bit)
// ---------------------------------------------------------------------------

void quantize_row_turbo4_0_prod_ref(const float * GGML_RESTRICT x, block_turbo4_0_prod * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO4_PROD == 0);
    const int64_t nb = k / QK_TURBO4_PROD;
    tq_init_rotation();

    for (int64_t i = 0; i < nb; i++) {
        const float * xb_orig = x + i * QK_TURBO4_PROD;

        float xb[QK_TURBO4_PROD];
        tq_rotate(xb_orig, xb);

        float sum_sq = 0.0f;
        for (int j = 0; j < QK_TURBO4_PROD; j++) sum_sq += xb[j] * xb[j];
        float norm = sqrtf(sum_sq);

        y[i].norm = GGML_FP32_TO_FP16(norm);
        y[i].rnorm_hi = GGML_FP32_TO_FP16(0.0f);
        y[i].rnorm_lo = GGML_FP32_TO_FP16(0.0f);
        memset(y[i].signs_hi, 0, sizeof(y[i].signs_hi));
        memset(y[i].signs_lo, 0, sizeof(y[i].signs_lo));

        if (norm == 0.0f) { memset(y[i].qs_hi, 0, sizeof(y[i].qs_hi)); memset(y[i].qs_lo, 0, sizeof(y[i].qs_lo)); continue; }
        float inv = 1.0f / norm;

        float res_hi[QK_TURBO4_PROD_HI], res_lo[QK_TURBO4_PROD_LO];
        quant_hi(xb, inv, y[i].qs_hi, centroids_8, 8, 3, QK_TURBO4_PROD_HI, res_hi);
        quant_lo(xb, inv, y[i].qs_lo, centroids_4, 4, 2, QK_TURBO4_PROD_LO, QK_TURBO4_PROD_HI, res_lo);

        y[i].rnorm_hi = GGML_FP32_TO_FP16(qjl_forward(res_hi, y[i].signs_hi, QK_TURBO4_PROD_HI, QJL_SEED_32));
        y[i].rnorm_lo = GGML_FP32_TO_FP16(qjl_forward(res_lo, y[i].signs_lo, QK_TURBO4_PROD_LO, QJL_SEED_96));
    }
}

void dequantize_row_turbo4_0_prod(const block_turbo4_0_prod * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO4_PROD == 0);
    const int64_t nb = k / QK_TURBO4_PROD;
    tq_init_rotation();

    for (int64_t i = 0; i < nb; i++) {
        float norm = GGML_FP16_TO_FP32(x[i].norm);

        float rotated[QK_TURBO4_PROD];
        float ch[QK_TURBO4_PROD_HI], cl[QK_TURBO4_PROD_LO];
        qjl_inverse(x[i].signs_hi, GGML_FP16_TO_FP32(x[i].rnorm_hi), ch, QK_TURBO4_PROD_HI, QJL_SEED_32);
        qjl_inverse(x[i].signs_lo, GGML_FP16_TO_FP32(x[i].rnorm_lo), cl, QK_TURBO4_PROD_LO, QJL_SEED_96);

        dequant_hi(x[i].qs_hi, ch, centroids_8, 3, QK_TURBO4_PROD_HI, norm, rotated);
        dequant_lo(x[i].qs_lo, cl, centroids_4, 2, QK_TURBO4_PROD_LO, norm, rotated, QK_TURBO4_PROD_HI);

        tq_unrotate(rotated, y + i * QK_TURBO4_PROD);
    }
}

// ---------------------------------------------------------------------------
// TURBO3_0_MSE: hi=3-bit MSE (centroids_8), lo=2-bit MSE (centroids_4)
// Pure MSE reconstruction — no QJL. Ideal for V cache.
// ---------------------------------------------------------------------------

void quantize_row_turbo3_0_mse_ref(const float * GGML_RESTRICT x, block_turbo3_0_mse * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO3_MSE == 0);
    const int64_t nb = k / QK_TURBO3_MSE;
    tq_init_rotation();

    for (int64_t i = 0; i < nb; i++) {
        const float * xb_orig = x + i * QK_TURBO3_MSE;

        float xb[QK_TURBO3_MSE];
        tq_rotate(xb_orig, xb);

        float sum_sq = 0.0f;
        for (int j = 0; j < QK_TURBO3_MSE; j++) sum_sq += xb[j] * xb[j];
        float norm = sqrtf(sum_sq);

        y[i].norm = GGML_FP32_TO_FP16(norm);

        if (norm == 0.0f) { memset(y[i].qs_hi, 0, sizeof(y[i].qs_hi)); memset(y[i].qs_lo, 0, sizeof(y[i].qs_lo)); continue; }
        float inv = 1.0f / norm;

        quant_hi_mse(xb, inv, y[i].qs_hi, centroids_8, 8, 3, QK_TURBO3_MSE_HI);
        quant_lo_mse(xb, inv, y[i].qs_lo, centroids_4, 4, 2, QK_TURBO3_MSE_LO, QK_TURBO3_MSE_HI);
    }
}

void dequantize_row_turbo3_0_mse(const block_turbo3_0_mse * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO3_MSE == 0);
    const int64_t nb = k / QK_TURBO3_MSE;
    tq_init_rotation();

    for (int64_t i = 0; i < nb; i++) {
        float norm = GGML_FP16_TO_FP32(x[i].norm);

        float rotated[QK_TURBO3_MSE];
        dequant_hi_mse(x[i].qs_hi, centroids_8, 3, QK_TURBO3_MSE_HI, norm, rotated);
        dequant_lo_mse(x[i].qs_lo, centroids_4, 2, QK_TURBO3_MSE_LO, norm, rotated, QK_TURBO3_MSE_HI);

        tq_unrotate(rotated, y + i * QK_TURBO3_MSE);
    }
}

// ---------------------------------------------------------------------------
// TURBO4_0_MSE: hi=4-bit MSE (centroids_16), lo=3-bit MSE (centroids_8)
// Pure MSE reconstruction — no QJL. Ideal for V cache.
// ---------------------------------------------------------------------------

void quantize_row_turbo4_0_mse_ref(const float * GGML_RESTRICT x, block_turbo4_0_mse * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO4_MSE == 0);
    const int64_t nb = k / QK_TURBO4_MSE;
    tq_init_rotation();

    for (int64_t i = 0; i < nb; i++) {
        const float * xb_orig = x + i * QK_TURBO4_MSE;

        float xb[QK_TURBO4_MSE];
        tq_rotate(xb_orig, xb);

        float sum_sq = 0.0f;
        for (int j = 0; j < QK_TURBO4_MSE; j++) sum_sq += xb[j] * xb[j];
        float norm = sqrtf(sum_sq);

        y[i].norm = GGML_FP32_TO_FP16(norm);

        if (norm == 0.0f) { memset(y[i].qs_hi, 0, sizeof(y[i].qs_hi)); memset(y[i].qs_lo, 0, sizeof(y[i].qs_lo)); continue; }
        float inv = 1.0f / norm;

        quant_hi_mse(xb, inv, y[i].qs_hi, centroids_16, 16, 4, QK_TURBO4_MSE_HI);
        quant_lo_mse(xb, inv, y[i].qs_lo, centroids_8, 8, 3, QK_TURBO4_MSE_LO, QK_TURBO4_MSE_HI);
    }
}

void dequantize_row_turbo4_0_mse(const block_turbo4_0_mse * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO4_MSE == 0);
    const int64_t nb = k / QK_TURBO4_MSE;
    tq_init_rotation();

    for (int64_t i = 0; i < nb; i++) {
        float norm = GGML_FP16_TO_FP32(x[i].norm);

        float rotated[QK_TURBO4_MSE];
        dequant_hi_mse(x[i].qs_hi, centroids_16, 4, QK_TURBO4_MSE_HI, norm, rotated);
        dequant_lo_mse(x[i].qs_lo, centroids_8, 3, QK_TURBO4_MSE_LO, norm, rotated, QK_TURBO4_MSE_HI);

        tq_unrotate(rotated, y + i * QK_TURBO4_MSE);
    }
}
