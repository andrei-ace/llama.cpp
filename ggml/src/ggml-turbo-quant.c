// TurboQuant CPU reference implementation
// Algorithm: PolarQuant (MSE-optimal scalar quantization) + QJL (1-bit residual correction)
// Reference: "TurboQuant: Redefining AI Efficiency with Extreme Compression" (arXiv 2504.19874)
//
// TURBO3_0: mixed-precision 2.5 bpw (32 outlier channels at 3-bit, 96 regular at 2-bit)
// TURBO4_0: uniform 3.5 bpw (128 channels at 4-bit: 3-bit PolarQuant + 1-bit QJL)
//
// Both types operate on 128-element blocks (= one attention head dimension).
// The rotation matrix (PolarQuant pre-rotation) is NOT applied here.
// It is handled at the graph/operator level before calling these functions.
//
// QJL uses a randomized Hadamard projection (not identity) so that the
// inner product estimator is unbiased: E[<q, x_hat>] = <q, x>.

#include "ggml-quants.h"
#include "ggml-impl.h"

#include <math.h>
#include <string.h>
#include <assert.h>

// ---------------------------------------------------------------------------
// Centroid tables
// ---------------------------------------------------------------------------
// Exact Lloyd-Max centroids for the Beta distribution arising after random
// orthogonal rotation of unit-norm vectors in R^d:
//   f(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^((d-3)/2)

// TURBO3_0 outlier channels: 4 signed centroids (2-bit PolarQuant, d=128)
static const float turbo3_hi_centroids[4] = {
    -0.1330458627f,
    -0.0399983984f,
     0.0399983984f,
     0.1330458627f,
};

// TURBO3_0 regular channels: 2 signed centroids (1-bit PolarQuant, d=128)
static const float turbo3_lo_centroids[2] = {
    -0.0707250243f,
     0.0707250243f,
};

// TURBO4_0: 8 signed centroids (3-bit PolarQuant, d=128)
static const float turbo4_centroids[8] = {
    -0.1883988281f,
    -0.1181421705f,
    -0.0665887043f,
    -0.0216082019f,
     0.0216082019f,
     0.0665887043f,
     0.1181421705f,
     0.1883988281f,
};

// QJL correction factor: sqrt(pi/2) / d, where d = 128
#define TURBO_QJL_ALPHA  0.00979152f

// ---------------------------------------------------------------------------
// QJL Hadamard projection
// ---------------------------------------------------------------------------
// The QJL transform uses S = D_qjl * H where:
//   H = Walsh-Hadamard matrix (orthogonal, self-inverse up to scale)
//   D_qjl = diagonal of deterministic ±1 signs
// This gives an unbiased inner product estimator per the QJL paper.

// Deterministic ±1 signs for QJL projection (golden-ratio hash, fixed for all blocks)
static void qjl_signs_128(float * signs) {
    for (int i = 0; i < 128; i++) {
        // Deterministic pseudo-random sign from golden ratio hash
        uint32_t h = (uint32_t)i * 0x9E3779B9u;
        signs[i] = (h >> 16) & 1 ? 1.0f : -1.0f;
    }
}

// Fast Walsh-Hadamard Transform (in-place, unnormalized)
static void fwht_128(float * x) {
    for (int half = 1; half < 128; half *= 2) {
        for (int i = 0; i < 128; i += half * 2) {
            for (int j = i; j < i + half; j++) {
                float a = x[j];
                float b = x[j + half];
                x[j]        = a + b;
                x[j + half]  = a - b;
            }
        }
    }
}

// QJL forward (full d=128): project residual → 128 sign bits
// Returns ||r||
static float qjl_forward_128(const float * r, uint8_t * signs_out) {
    float qjl_s[128];
    float proj[128];

    qjl_signs_128(qjl_s);

    for (int j = 0; j < 128; j++) {
        proj[j] = r[j] * qjl_s[j];
    }
    fwht_128(proj);

    float rnorm_sq = 0.0f;
    for (int j = 0; j < 128; j++) {
        rnorm_sq += r[j] * r[j];
    }

    memset(signs_out, 0, 16);
    for (int j = 0; j < 128; j++) {
        int bit = (proj[j] >= 0.0f) ? 1 : 0;
        signs_out[j / 8] |= (uint8_t)(bit << (j % 8));
    }

    return sqrtf(rnorm_sq);
}

// QJL inverse (full d=128): 128 sign bits → correction vector
static void qjl_inverse_128(const uint8_t * signs_in, float rnorm, float * correction) {
    float qjl_s[128];
    qjl_signs_128(qjl_s);

    for (int j = 0; j < 128; j++) {
        int bit = (signs_in[j / 8] >> (j % 8)) & 1;
        correction[j] = bit ? 1.0f : -1.0f;
    }

    fwht_128(correction);

    // Scale: sqrt(pi/2) / d * ||r|| where d=128
    float scale = TURBO_QJL_ALPHA * rnorm;
    for (int j = 0; j < 128; j++) {
        correction[j] *= qjl_s[j] * scale;
    }
}

// QJL forward (reduced m=32): project 128-dim residual → 32 sign bits
// Uses first 32 rows of S = D*H (first 32 outputs of Hadamard)
// Returns ||r||
static float qjl_forward_32(const float * r, uint8_t * signs_out) {
    float qjl_s[128];
    float proj[128];

    qjl_signs_128(qjl_s);

    for (int j = 0; j < 128; j++) {
        proj[j] = r[j] * qjl_s[j];
    }
    fwht_128(proj);

    float rnorm_sq = 0.0f;
    for (int j = 0; j < 128; j++) {
        rnorm_sq += r[j] * r[j];
    }

    // Store sign of first 32 projections only
    memset(signs_out, 0, 4);
    for (int j = 0; j < 32; j++) {
        int bit = (proj[j] >= 0.0f) ? 1 : 0;
        signs_out[j / 8] |= (uint8_t)(bit << (j % 8));
    }

    return sqrtf(rnorm_sq);
}

// QJL inverse (reduced m=32): 32 sign bits → 128-dim correction
// Reconstructs using S^T_{128×32} (first 32 columns of H*D)
static void qjl_inverse_32(const uint8_t * signs_in, float rnorm, float * correction) {
    float qjl_s[128];
    qjl_signs_128(qjl_s);

    // Build 128-dim vector: first 32 entries from signs, rest zero
    for (int j = 0; j < 32; j++) {
        int bit = (signs_in[j / 8] >> (j % 8)) & 1;
        correction[j] = bit ? 1.0f : -1.0f;
    }
    for (int j = 32; j < 128; j++) {
        correction[j] = 0.0f;
    }

    // Apply H (inverse Hadamard = H/d for unnormalized, but we absorb the /d into scale)
    fwht_128(correction);

    // Scale: sqrt(pi/2) / m * ||r|| where m=32 (number of projections)
    // The /m (not /d) is because we're using m projections, not d
    float scale = 1.2533141f / 32.0f * rnorm;  // sqrt(pi/2) / m * ||r||
    for (int j = 0; j < 128; j++) {
        correction[j] *= qjl_s[j] * scale;
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Find nearest signed centroid (linear scan, n entries)
static inline int best_signed_index(float val, const float * centroids, int n) {
    int best = 0;
    float best_dist = fabsf(val - centroids[0]);
    for (int i = 1; i < n; i++) {
        float dist = fabsf(val - centroids[i]);
        if (dist < best_dist) {
            best_dist = dist;
            best = i;
        }
    }
    return best;
}

// ---------------------------------------------------------------------------
// TURBO3_0: mixed-precision PolarQuant + full QJL (2.5 bpw)
//   Channels [0, 32): 2-bit PolarQuant (4 signed centroids) + QJL
//   Channels [32, 128): 1-bit PolarQuant (2 signed centroids) + QJL
//   Full 128-dim Hadamard QJL for unbiased inner product estimation
// ---------------------------------------------------------------------------

void quantize_row_turbo3_0_ref(const float * GGML_RESTRICT x, block_turbo3_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO3 == 0);
    const int64_t nb = k / QK_TURBO3;

    for (int64_t i = 0; i < nb; i++) {
        const float * xb = x + i * QK_TURBO3;

        // 1. Compute block L2 norm
        float sum_sq = 0.0f;
        for (int j = 0; j < QK_TURBO3; j++) {
            sum_sq += xb[j] * xb[j];
        }
        const float norm = sqrtf(sum_sq);

        y[i].norm  = GGML_FP32_TO_FP16(norm);
        y[i].rnorm = GGML_FP32_TO_FP16(0.0f);

        // 2. Clear packed arrays
        memset(y[i].qs_hi, 0, sizeof(y[i].qs_hi));
        memset(y[i].qs_lo, 0, sizeof(y[i].qs_lo));
        memset(y[i].signs, 0, sizeof(y[i].signs));

        if (norm == 0.0f) {
            continue;
        }

        const float inv_norm = 1.0f / norm;

        // 3. PolarQuant: quantize with mixed precision
        float centroid_vals[QK_TURBO3];

        // Outlier channels [0, 32): 2-bit PolarQuant (4 signed centroids)
        for (int j = 0; j < QK_TURBO3_HI; j++) {
            const float xn = xb[j] * inv_norm;
            const int idx = best_signed_index(xn, turbo3_hi_centroids, 4);
            centroid_vals[j] = turbo3_hi_centroids[idx];
            y[i].qs_hi[j / 4] |= (uint8_t)(idx << ((j % 4) * 2));
        }

        // Regular channels [32, 128): 1-bit PolarQuant (2 signed centroids)
        for (int j = 0; j < QK_TURBO3_LO; j++) {
            const float xn = xb[QK_TURBO3_HI + j] * inv_norm;
            const int idx = (xn >= 0.0f) ? 1 : 0;
            centroid_vals[QK_TURBO3_HI + j] = turbo3_lo_centroids[idx];
            y[i].qs_lo[j / 8] |= (uint8_t)(idx << (j % 8));
        }

        // 4. Compute residual
        float residual[128];
        for (int j = 0; j < QK_TURBO3; j++) {
            residual[j] = xb[j] * inv_norm - centroid_vals[j];
        }

        // 5. Full 128-dim QJL: Hadamard-projected sign bits + residual norm
        float rnorm = qjl_forward_128(residual, y[i].signs);
        y[i].rnorm = GGML_FP32_TO_FP16(rnorm);
    }
}

void dequantize_row_turbo3_0(const block_turbo3_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO3 == 0);
    const int64_t nb = k / QK_TURBO3;

    for (int64_t i = 0; i < nb; i++) {
        const float norm  = GGML_FP16_TO_FP32(x[i].norm);
        const float rnorm = GGML_FP16_TO_FP32(x[i].rnorm);
        float * yb = y + i * QK_TURBO3;

        // Reconstruct centroids
        float centroid_vals[128];

        for (int j = 0; j < QK_TURBO3_HI; j++) {
            const int idx = (x[i].qs_hi[j / 4] >> ((j % 4) * 2)) & 0x3;
            centroid_vals[j] = turbo3_hi_centroids[idx];
        }
        for (int j = 0; j < QK_TURBO3_LO; j++) {
            const int idx = (x[i].qs_lo[j / 8] >> (j % 8)) & 1;
            centroid_vals[QK_TURBO3_HI + j] = turbo3_lo_centroids[idx];
        }

        // Full QJL inverse: 128 sign bits → 128-dim correction
        float correction[128];
        qjl_inverse_128(x[i].signs, rnorm, correction);

        // Final reconstruction
        for (int j = 0; j < QK_TURBO3; j++) {
            yb[j] = norm * (centroid_vals[j] + correction[j]);
        }
    }
}

// ---------------------------------------------------------------------------
// TURBO4_0: PolarQuant + QJL (3-bit signed centroid + 1-bit QJL, 3.5 bpw)
// ---------------------------------------------------------------------------

// 3-bit packing helpers: sequential bitstream, 128 elements x 3 bits = 48 bytes
static inline void turbo4_pack_3bit(uint8_t * qs, int j, int val) {
    const int bit_pos = j * 3;
    const int byte_idx = bit_pos >> 3;
    const int shift = bit_pos & 7;

    qs[byte_idx] |= (uint8_t)((val << shift) & 0xFF);
    if (shift > 5) {
        qs[byte_idx + 1] |= (uint8_t)(val >> (8 - shift));
    }
}

static inline int turbo4_unpack_3bit(const uint8_t * qs, int j) {
    const int bit_pos = j * 3;
    const int byte_idx = bit_pos >> 3;
    const int shift = bit_pos & 7;

    if (shift <= 5) {
        return (qs[byte_idx] >> shift) & 0x7;
    }
    return ((qs[byte_idx] >> shift) | (qs[byte_idx + 1] << (8 - shift))) & 0x7;
}

void quantize_row_turbo4_0_ref(const float * GGML_RESTRICT x, block_turbo4_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO4 == 0);
    const int64_t nb = k / QK_TURBO4;

    for (int64_t i = 0; i < nb; i++) {
        const float * xb = x + i * QK_TURBO4;

        // 1. Compute block L2 norm
        float sum_sq = 0.0f;
        for (int j = 0; j < QK_TURBO4; j++) {
            sum_sq += xb[j] * xb[j];
        }
        const float norm = sqrtf(sum_sq);

        y[i].norm  = GGML_FP32_TO_FP16(norm);
        y[i].rnorm = GGML_FP32_TO_FP16(0.0f);

        // 2. Clear packed arrays
        memset(y[i].qs,    0, sizeof(y[i].qs));
        memset(y[i].signs, 0, sizeof(y[i].signs));

        if (norm == 0.0f) {
            continue;
        }

        const float inv_norm = 1.0f / norm;

        // 3. PolarQuant: quantize each element with 3-bit signed centroid
        int indices[QK_TURBO4];
        for (int j = 0; j < QK_TURBO4; j++) {
            const float xn = xb[j] * inv_norm;
            indices[j] = best_signed_index(xn, turbo4_centroids, 8);
            turbo4_pack_3bit(y[i].qs, j, indices[j]);
        }

        // 4. Compute residual
        float residual[128];
        for (int j = 0; j < QK_TURBO4; j++) {
            residual[j] = xb[j] * inv_norm - turbo4_centroids[indices[j]];
        }

        // 5. QJL: full 128-dim Hadamard-projected sign bits + residual norm
        float rnorm = qjl_forward_128(residual, y[i].signs);
        y[i].rnorm = GGML_FP32_TO_FP16(rnorm);
    }
}

void dequantize_row_turbo4_0(const block_turbo4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO4 == 0);
    const int64_t nb = k / QK_TURBO4;

    for (int64_t i = 0; i < nb; i++) {
        const float norm  = GGML_FP16_TO_FP32(x[i].norm);
        const float rnorm = GGML_FP16_TO_FP32(x[i].rnorm);
        float * yb = y + i * QK_TURBO4;

        // Reconstruct centroids
        float centroid_vals[128];
        for (int j = 0; j < QK_TURBO4; j++) {
            const int idx = turbo4_unpack_3bit(x[i].qs, j);
            centroid_vals[j] = turbo4_centroids[idx];
        }

        // QJL inverse: reconstruct correction from 128 sign bits
        float correction[128];
        qjl_inverse_128(x[i].signs, rnorm, correction);

        // Final reconstruction
        for (int j = 0; j < QK_TURBO4; j++) {
            yb[j] = norm * (centroid_vals[j] + correction[j]);
        }
    }
}
