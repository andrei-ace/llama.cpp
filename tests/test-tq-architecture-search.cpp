// TurboQuant Architecture Search — pure-math research tool
// Explores quantization architectures for extreme outlier KV cache layers.
// No model inference, no ggml dependencies — fully self-contained.
//
// Build: cmake --build build -t test-tq-architecture-search -j14
// Run:   ./build/bin/test-tq-architecture-search

#pragma GCC diagnostic ignored "-Wunused-const-variable"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <random>
#include <vector>
#include <cfloat>

// ---------------------------------------------------------------------------
// Lloyd-Max centroids (from ggml-turbo-quant.c calibration)
// ---------------------------------------------------------------------------

// d=32 Lloyd-Max centroids
static const float c16_d32[16] = {
    -0.4534f, -0.3499f, -0.2765f, -0.2161f, -0.1629f, -0.1138f, -0.0674f, -0.0223f,
     0.0223f,  0.0674f,  0.1138f,  0.1629f,  0.2161f,  0.2765f,  0.3499f,  0.4534f
};
static const float c8_d32[8] = {
    -0.3663f, -0.2325f, -0.1318f, -0.0429f,  0.0429f,  0.1318f,  0.2325f,  0.3663f
};
static const float c4_d32[4] = {
    -0.2633f, -0.0798f, 0.0798f, 0.2633f
};

// d=96 Lloyd-Max centroids
static const float c8_d96[8] = {
    -0.2169f, -0.1362f, -0.0768f, -0.0249f,  0.0249f,  0.0768f,  0.1362f,  0.2169f
};
static const float c4_d96[4] = {
    -0.1534f, -0.0462f, 0.0462f, 0.1534f
};
static const float c2_d96[2] = {
    -0.0816f, 0.0816f
};

// d=64 Lloyd-Max centroids (interpolated for 50/50 split experiments)
static const float c32_d64[32] = {
    -0.5271f, -0.4308f, -0.3686f, -0.3198f, -0.2784f, -0.2416f, -0.2082f, -0.1774f,
    -0.1487f, -0.1215f, -0.0958f, -0.0710f, -0.0472f, -0.0240f, -0.0080f, -0.0027f,
     0.0027f,  0.0080f,  0.0240f,  0.0472f,  0.0710f,  0.0958f,  0.1215f,  0.1487f,
     0.1774f,  0.2082f,  0.2416f,  0.2784f,  0.3198f,  0.3686f,  0.4308f,  0.5271f
};
static const float c16_d64[16] = {
    -0.4534f, -0.3499f, -0.2765f, -0.2161f, -0.1629f, -0.1138f, -0.0674f, -0.0223f,
     0.0223f,  0.0674f,  0.1138f,  0.1629f,  0.2161f,  0.2765f,  0.3499f,  0.4534f
};
static const float c4_d64[4] = {
    -0.1534f, -0.0462f, 0.0462f, 0.1534f
};
static const float c2_d64[2] = {
    -0.0816f, 0.0816f
};

// ---------------------------------------------------------------------------
// Utility: FWHT (Fast Walsh-Hadamard Transform) — in-place, normalized
// ---------------------------------------------------------------------------

static void fwht(float * x, int n) {
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
// Utility: pseudo-random sign flips for structured rotation
// ---------------------------------------------------------------------------

static float struct_sign(int i, int n) {
    uint64_t seed = (n == 96) ? 0x5452534C4F393600ULL : 0x5452534C31393200ULL;
    for (int k = 0; k <= i; k++) {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
    }
    return ((uint32_t)(seed >> 32) & 1) ? -1.0f : 1.0f;
}

// O_3 matrix for structured rotation (cross-block mix of 3 FWHT blocks)
static const float O3[9] = {
    0.5773502691896257f,  0.5773502691896257f,  0.5773502691896257f,
    0.7071067811865476f,  0.0f,                -0.7071067811865476f,
    0.4082482904638631f, -0.8164965809277261f,  0.4082482904638631f,
};
static const float O3_T[9] = {
    0.5773502691896257f,  0.7071067811865476f,  0.4082482904638631f,
    0.5773502691896257f,  0.0f,                -0.8164965809277261f,
    0.5773502691896257f, -0.7071067811865476f,  0.4082482904638631f,
};

static void structured_rotate_lo(float * x, int n) {
    int bd = n / 3;
    for (int i = 0; i < n; i++) x[i] *= struct_sign(i, n);
    for (int j = 0; j < bd; j++) {
        float a = x[j], b = x[bd + j], c = x[2*bd + j];
        x[j]        = O3[0]*a + O3[1]*b + O3[2]*c;
        x[bd + j]   = O3[3]*a + O3[4]*b + O3[5]*c;
        x[2*bd + j] = O3[6]*a + O3[7]*b + O3[8]*c;
    }
    std::vector<float> tmp(n);
    for (int i = 0; i < n; i++) tmp[(35*i + 17) % n] = x[i];
    for (int i = 0; i < n; i++) x[i] = tmp[i];
    for (int b = 0; b < 3; b++) fwht(x + b * bd, bd);
}

static void structured_unrotate_lo(float * x, int n) {
    int bd = n / 3;
    for (int b = 0; b < 3; b++) fwht(x + b * bd, bd);
    std::vector<float> tmp(n);
    for (int i = 0; i < n; i++) tmp[i] = x[(35*i + 17) % n];
    for (int i = 0; i < n; i++) x[i] = tmp[i];
    for (int j = 0; j < bd; j++) {
        float a = x[j], b = x[bd + j], c = x[2*bd + j];
        x[j]        = O3_T[0]*a + O3_T[1]*b + O3_T[2]*c;
        x[bd + j]   = O3_T[3]*a + O3_T[4]*b + O3_T[5]*c;
        x[2*bd + j] = O3_T[6]*a + O3_T[7]*b + O3_T[8]*c;
    }
    for (int i = 0; i < n; i++) x[i] *= struct_sign(i, n);
}

// ---------------------------------------------------------------------------
// Utility: MSE quantize to nearest centroid
// ---------------------------------------------------------------------------

static int nearest_centroid(float val, const float * centroids, int nc) {
    int best = 0;
    float best_d = fabsf(val - centroids[0]);
    for (int i = 1; i < nc; i++) {
        float d = fabsf(val - centroids[i]);
        if (d < best_d) { best_d = d; best = i; }
    }
    return best;
}

// Quantize a sub-vector: normalize, rotate (FWHT), quantize to centroids, dequant, unrotate, denormalize
static void mse_quant_dequant_block(
    const float * in, float * out, int n,
    const float * centroids, int nc,
    bool do_fwht
) {
    std::vector<float> tmp(n);
    memcpy(tmp.data(), in, n * sizeof(float));

    // Compute norm
    float norm = 0.0f;
    for (int i = 0; i < n; i++) norm += tmp[i] * tmp[i];
    norm = sqrtf(norm);
    if (norm < 1e-30f) { memset(out, 0, n * sizeof(float)); return; }

    // Normalize
    float inv_norm = 1.0f / norm;
    for (int i = 0; i < n; i++) tmp[i] *= inv_norm;

    // Optional rotation
    if (do_fwht) fwht(tmp.data(), n);

    // Quantize + dequantize
    for (int i = 0; i < n; i++) {
        int idx = nearest_centroid(tmp[i], centroids, nc);
        tmp[i] = centroids[idx];
    }

    // Inverse rotation
    if (do_fwht) fwht(tmp.data(), n);

    // Denormalize
    for (int i = 0; i < n; i++) out[i] = tmp[i] * norm;
}

// ---------------------------------------------------------------------------
// Utility: QJL sign quantization
// ---------------------------------------------------------------------------

// QJL: store just the sign bits after rotation. 1 bpv.
static void qjl_quant_dequant_block(
    const float * in, float * out, int n,
    bool do_fwht
) {
    std::vector<float> tmp(n);
    memcpy(tmp.data(), in, n * sizeof(float));

    float norm = 0.0f;
    for (int i = 0; i < n; i++) norm += tmp[i] * tmp[i];
    norm = sqrtf(norm);
    if (norm < 1e-30f) { memset(out, 0, n * sizeof(float)); return; }

    float inv_norm = 1.0f / norm;
    for (int i = 0; i < n; i++) tmp[i] *= inv_norm;

    if (do_fwht) fwht(tmp.data(), n);

    // Sign quantization
    float rms = 0.0f;
    for (int i = 0; i < n; i++) rms += tmp[i] * tmp[i];
    rms = sqrtf(rms / n);
    for (int i = 0; i < n; i++) {
        tmp[i] = (tmp[i] >= 0.0f) ? rms : -rms;
    }

    if (do_fwht) fwht(tmp.data(), n);

    for (int i = 0; i < n; i++) out[i] = tmp[i] * norm;
}

// ---------------------------------------------------------------------------
// Data generation
// ---------------------------------------------------------------------------

struct TestConfig {
    float outlier_ratio;      // fraction of variance in outlier channels
    float sigma_hi;           // std dev for outlier channels
    float sigma_lo;           // std dev for regular channels
    int   n_hi;               // number of outlier channels
    int   n_lo;               // number of regular channels
    const char * label;
};

static void generate_vectors(
    std::vector<std::vector<float>> & vecs,
    int count, int n_hi, int n_lo,
    float sigma_hi, float sigma_lo,
    std::mt19937 & rng
) {
    std::normal_distribution<float> dist_hi(0.0f, sigma_hi);
    std::normal_distribution<float> dist_lo(0.0f, sigma_lo);
    vecs.resize(count);
    int dim = n_hi + n_lo;
    for (int i = 0; i < count; i++) {
        vecs[i].resize(dim);
        for (int j = 0; j < n_hi; j++) vecs[i][j] = dist_hi(rng);
        for (int j = 0; j < n_lo; j++) vecs[i][n_hi + j] = dist_lo(rng);
    }
}

// ---------------------------------------------------------------------------
// Quality metrics
// ---------------------------------------------------------------------------

struct Metrics {
    double dot_rel_error;   // |q.k - q.k_hat| / |q.k| averaged
    double cosine_sim;      // cos(k, k_hat) averaged
    double rel_l2;          // ||k - k_hat|| / ||k|| averaged
};

static Metrics compute_metrics(
    const std::vector<std::vector<float>> & K,
    const std::vector<std::vector<float>> & K_hat,
    const std::vector<std::vector<float>> & Q,
    int n_dot_pairs  // random Q.K pairs to sample
) {
    int dim = (int)K[0].size();
    int nk = (int)K.size();
    int nq = (int)Q.size();

    // Cosine similarity and relative L2
    double cos_sum = 0.0, l2_sum = 0.0;
    for (int i = 0; i < nk; i++) {
        double dot_kk = 0, dot_kh = 0, dot_hh = 0, diff2 = 0, knorm2 = 0;
        for (int d = 0; d < dim; d++) {
            double kd = K[i][d], hd = K_hat[i][d];
            dot_kk += kd * kd;
            dot_kh += kd * hd;
            dot_hh += hd * hd;
            diff2 += (kd - hd) * (kd - hd);
            knorm2 += kd * kd;
        }
        double cos_denom = sqrt(dot_kk * dot_hh);
        if (cos_denom > 1e-30) cos_sum += dot_kh / cos_denom;
        if (knorm2 > 1e-30) l2_sum += sqrt(diff2) / sqrt(knorm2);
    }

    // Dot product relative error — sample pairs
    double dot_err_sum = 0.0;
    int dot_count = 0;
    std::mt19937 rng(12345);
    for (int p = 0; p < n_dot_pairs; p++) {
        int qi = rng() % nq;
        int ki = rng() % nk;
        double dot_true = 0.0, dot_quant = 0.0;
        for (int d = 0; d < dim; d++) {
            dot_true  += (double)Q[qi][d] * (double)K[ki][d];
            dot_quant += (double)Q[qi][d] * (double)K_hat[ki][d];
        }
        if (fabs(dot_true) > 1e-10) {
            dot_err_sum += fabs(dot_true - dot_quant) / fabs(dot_true);
            dot_count++;
        }
    }

    Metrics m;
    m.dot_rel_error = (dot_count > 0) ? dot_err_sum / dot_count : 0.0;
    m.cosine_sim    = cos_sum / nk;
    m.rel_l2        = l2_sum / nk;
    return m;
}

// ---------------------------------------------------------------------------
// Architecture A: tqk3_sj baseline (4-bit hi MSE + FWHT + QJL, 3-bit lo MSE + structured)
// ---------------------------------------------------------------------------

static void arch_a_quantize(
    const std::vector<std::vector<float>> & K,
    std::vector<std::vector<float>> & K_hat,
    int n_hi, int n_lo
) {
    int nk = (int)K.size();
    int dim = n_hi + n_lo;
    K_hat.resize(nk);

    for (int i = 0; i < nk; i++) {
        K_hat[i].resize(dim);

        // Hi part: 4-bit MSE + FWHT-32 rotation + QJL signs
        // Step 1: MSE quantize the hi channels
        std::vector<float> hi(n_hi), lo(n_lo);
        memcpy(hi.data(), K[i].data(), n_hi * sizeof(float));
        memcpy(lo.data(), K[i].data() + n_hi, n_lo * sizeof(float));

        // Hi: norm + FWHT + 4-bit MSE
        float norm_hi = 0.0f;
        for (int j = 0; j < n_hi; j++) norm_hi += hi[j] * hi[j];
        norm_hi = sqrtf(norm_hi);
        if (norm_hi > 1e-30f) {
            float inv = 1.0f / norm_hi;
            for (int j = 0; j < n_hi; j++) hi[j] *= inv;
        }
        fwht(hi.data(), n_hi);
        for (int j = 0; j < n_hi; j++) {
            int idx = nearest_centroid(hi[j], c16_d32, 16);
            hi[j] = c16_d32[idx];
        }
        fwht(hi.data(), n_hi);
        for (int j = 0; j < n_hi; j++) hi[j] *= norm_hi;

        // Also apply QJL signs on hi
        {
            std::vector<float> hi_qjl(n_hi);
            qjl_quant_dequant_block(K[i].data(), hi_qjl.data(), n_hi, true);
            // Blend: MSE provides magnitude structure, QJL provides sign correction
            // In practice tqk3_sj stores both MSE indices and QJL signs.
            // The reconstruction uses MSE centroids for magnitude, QJL sign for correction.
            // For this simulation, we'll just use the MSE result (QJL is additive at decode time).
            // The QJL signs cost 1 bit each but are used in the dot product path, not reconstruction.
        }

        // Lo: 3-bit MSE + structured rotation (3x FWHT-32)
        float norm_lo = 0.0f;
        for (int j = 0; j < n_lo; j++) norm_lo += lo[j] * lo[j];
        norm_lo = sqrtf(norm_lo);
        if (norm_lo > 1e-30f) {
            float inv = 1.0f / norm_lo;
            for (int j = 0; j < n_lo; j++) lo[j] *= inv;
        }

        // Structured rotation for lo
        if (n_lo == 96) {
            structured_rotate_lo(lo.data(), n_lo);
        } else {
            // Fallback: just FWHT blocks
            int bd = n_lo / 3;
            for (int b = 0; b < 3; b++) fwht(lo.data() + b * bd, bd);
        }

        for (int j = 0; j < n_lo; j++) {
            const float * cents = (n_lo == 96) ? c8_d96 : c8_d32;
            int idx = nearest_centroid(lo[j], cents, 8);
            lo[j] = cents[idx];
        }

        // Unrotate
        if (n_lo == 96) {
            structured_unrotate_lo(lo.data(), n_lo);
        } else {
            int bd = n_lo / 3;
            for (int b = 0; b < 3; b++) fwht(lo.data() + b * bd, bd);
        }
        for (int j = 0; j < n_lo; j++) lo[j] *= norm_lo;

        memcpy(K_hat[i].data(), hi.data(), n_hi * sizeof(float));
        memcpy(K_hat[i].data() + n_hi, lo.data(), n_lo * sizeof(float));
    }
}

// ---------------------------------------------------------------------------
// Architecture B: tqk3_sjj (4-bit hi + QJL, 2-bit lo + structured + QJL)
// ---------------------------------------------------------------------------

static void arch_b_quantize(
    const std::vector<std::vector<float>> & K,
    std::vector<std::vector<float>> & K_hat,
    int n_hi, int n_lo
) {
    int nk = (int)K.size();
    int dim = n_hi + n_lo;
    K_hat.resize(nk);

    for (int i = 0; i < nk; i++) {
        K_hat[i].resize(dim);

        std::vector<float> hi(n_hi), lo(n_lo);
        memcpy(hi.data(), K[i].data(), n_hi * sizeof(float));
        memcpy(lo.data(), K[i].data() + n_hi, n_lo * sizeof(float));

        // Hi: norm + FWHT + 4-bit MSE (same as A)
        float norm_hi = 0.0f;
        for (int j = 0; j < n_hi; j++) norm_hi += hi[j] * hi[j];
        norm_hi = sqrtf(norm_hi);
        if (norm_hi > 1e-30f) {
            float inv = 1.0f / norm_hi;
            for (int j = 0; j < n_hi; j++) hi[j] *= inv;
        }
        fwht(hi.data(), n_hi);
        for (int j = 0; j < n_hi; j++) {
            int idx = nearest_centroid(hi[j], c16_d32, 16);
            hi[j] = c16_d32[idx];
        }
        fwht(hi.data(), n_hi);
        for (int j = 0; j < n_hi; j++) hi[j] *= norm_hi;

        // Lo: norm + structured rotation + 2-bit MSE
        float norm_lo = 0.0f;
        for (int j = 0; j < n_lo; j++) norm_lo += lo[j] * lo[j];
        norm_lo = sqrtf(norm_lo);
        if (norm_lo > 1e-30f) {
            float inv = 1.0f / norm_lo;
            for (int j = 0; j < n_lo; j++) lo[j] *= inv;
        }

        if (n_lo == 96) {
            structured_rotate_lo(lo.data(), n_lo);
        } else if (n_lo % 32 == 0) {
            for (int b = 0; b < n_lo / 32; b++) fwht(lo.data() + b * 32, 32);
        }

        for (int j = 0; j < n_lo; j++) {
            const float * cents = (n_lo == 96) ? c2_d96 : c2_d96;
            int nc = 2;
            int idx = nearest_centroid(lo[j], cents, nc);
            lo[j] = cents[idx];
        }

        if (n_lo == 96) {
            structured_unrotate_lo(lo.data(), n_lo);
        } else if (n_lo % 32 == 0) {
            for (int b = 0; b < n_lo / 32; b++) fwht(lo.data() + b * 32, 32);
        }
        for (int j = 0; j < n_lo; j++) lo[j] *= norm_lo;

        memcpy(K_hat[i].data(), hi.data(), n_hi * sizeof(float));
        memcpy(K_hat[i].data() + n_hi, lo.data(), n_lo * sizeof(float));
    }
}

// ---------------------------------------------------------------------------
// Architecture C: Per-group scaling on hi (4 groups of 8)
// ---------------------------------------------------------------------------

static void arch_c_quantize(
    const std::vector<std::vector<float>> & K,
    std::vector<std::vector<float>> & K_hat,
    int n_hi, int n_lo
) {
    int nk = (int)K.size();
    int dim = n_hi + n_lo;
    K_hat.resize(nk);

    int n_groups = 4;
    int group_sz = n_hi / n_groups;

    for (int i = 0; i < nk; i++) {
        K_hat[i].resize(dim);

        std::vector<float> hi(n_hi), lo(n_lo);
        memcpy(hi.data(), K[i].data(), n_hi * sizeof(float));
        memcpy(lo.data(), K[i].data() + n_hi, n_lo * sizeof(float));

        // Hi: per-group norm + 4-bit MSE
        // Each group of 8 gets its own fp16 norm
        for (int g = 0; g < n_groups; g++) {
            float * gp = hi.data() + g * group_sz;
            float gnorm = 0.0f;
            for (int j = 0; j < group_sz; j++) gnorm += gp[j] * gp[j];
            gnorm = sqrtf(gnorm);
            if (gnorm > 1e-30f) {
                float inv = 1.0f / gnorm;
                for (int j = 0; j < group_sz; j++) gp[j] *= inv;
            }
            // FWHT within group (group_sz must be power of 2)
            if (group_sz >= 2 && (group_sz & (group_sz - 1)) == 0) {
                fwht(gp, group_sz);
            }
            for (int j = 0; j < group_sz; j++) {
                int idx = nearest_centroid(gp[j], c16_d32, 16);
                gp[j] = c16_d32[idx];
            }
            if (group_sz >= 2 && (group_sz & (group_sz - 1)) == 0) {
                fwht(gp, group_sz);
            }
            for (int j = 0; j < group_sz; j++) gp[j] *= gnorm;
        }

        // Lo: same as tqk3_sjj (2-bit MSE + structured rotation)
        float norm_lo = 0.0f;
        for (int j = 0; j < n_lo; j++) norm_lo += lo[j] * lo[j];
        norm_lo = sqrtf(norm_lo);
        if (norm_lo > 1e-30f) {
            float inv = 1.0f / norm_lo;
            for (int j = 0; j < n_lo; j++) lo[j] *= inv;
        }
        if (n_lo == 96) {
            structured_rotate_lo(lo.data(), n_lo);
        } else if (n_lo % 32 == 0) {
            for (int b = 0; b < n_lo / 32; b++) fwht(lo.data() + b * 32, 32);
        }
        for (int j = 0; j < n_lo; j++) {
            int idx = nearest_centroid(lo[j], c2_d96, 2);
            lo[j] = c2_d96[idx];
        }
        if (n_lo == 96) {
            structured_unrotate_lo(lo.data(), n_lo);
        } else if (n_lo % 32 == 0) {
            for (int b = 0; b < n_lo / 32; b++) fwht(lo.data() + b * 32, 32);
        }
        for (int j = 0; j < n_lo; j++) lo[j] *= norm_lo;

        memcpy(K_hat[i].data(), hi.data(), n_hi * sizeof(float));
        memcpy(K_hat[i].data() + n_hi, lo.data(), n_lo * sizeof(float));
    }
}

// ---------------------------------------------------------------------------
// Architecture D: Wider split (50/50) — 5-bit MSE hi + 2-bit lo
// ---------------------------------------------------------------------------

static void arch_d_quantize(
    const std::vector<std::vector<float>> & K,
    std::vector<std::vector<float>> & K_hat,
    int n_hi_orig, int n_lo_orig
) {
    // Override split: 50/50
    int dim = n_hi_orig + n_lo_orig;
    int n_hi = dim / 2;
    int n_lo = dim - n_hi;

    int nk = (int)K.size();
    K_hat.resize(nk);

    for (int i = 0; i < nk; i++) {
        K_hat[i].resize(dim);

        std::vector<float> hi(n_hi), lo(n_lo);
        memcpy(hi.data(), K[i].data(), n_hi * sizeof(float));
        memcpy(lo.data(), K[i].data() + n_hi, n_lo * sizeof(float));

        // Hi: norm + FWHT + 5-bit MSE (32 centroids)
        float norm_hi = 0.0f;
        for (int j = 0; j < n_hi; j++) norm_hi += hi[j] * hi[j];
        norm_hi = sqrtf(norm_hi);
        if (norm_hi > 1e-30f) {
            float inv = 1.0f / norm_hi;
            for (int j = 0; j < n_hi; j++) hi[j] *= inv;
        }
        if (n_hi >= 2 && (n_hi & (n_hi - 1)) == 0) {
            fwht(hi.data(), n_hi);
        }
        for (int j = 0; j < n_hi; j++) {
            int idx = nearest_centroid(hi[j], c32_d64, 32);
            hi[j] = c32_d64[idx];
        }
        if (n_hi >= 2 && (n_hi & (n_hi - 1)) == 0) {
            fwht(hi.data(), n_hi);
        }
        for (int j = 0; j < n_hi; j++) hi[j] *= norm_hi;

        // Lo: norm + FWHT blocks + 2-bit MSE
        float norm_lo = 0.0f;
        for (int j = 0; j < n_lo; j++) norm_lo += lo[j] * lo[j];
        norm_lo = sqrtf(norm_lo);
        if (norm_lo > 1e-30f) {
            float inv = 1.0f / norm_lo;
            for (int j = 0; j < n_lo; j++) lo[j] *= inv;
        }
        if (n_lo >= 2 && (n_lo & (n_lo - 1)) == 0) {
            fwht(lo.data(), n_lo);
        } else if (n_lo % 32 == 0) {
            for (int b = 0; b < n_lo / 32; b++) fwht(lo.data() + b * 32, 32);
        }
        for (int j = 0; j < n_lo; j++) {
            int idx = nearest_centroid(lo[j], c2_d64, 2);
            lo[j] = c2_d64[idx];
        }
        if (n_lo >= 2 && (n_lo & (n_lo - 1)) == 0) {
            fwht(lo.data(), n_lo);
        } else if (n_lo % 32 == 0) {
            for (int b = 0; b < n_lo / 32; b++) fwht(lo.data() + b * 32, 32);
        }
        for (int j = 0; j < n_lo; j++) lo[j] *= norm_lo;

        memcpy(K_hat[i].data(), hi.data(), n_hi * sizeof(float));
        memcpy(K_hat[i].data() + n_hi, lo.data(), n_lo * sizeof(float));
    }
}

// ---------------------------------------------------------------------------
// Architecture E: Residual quantization on hi (two-pass 4-bit MSE)
// ---------------------------------------------------------------------------

static void arch_e_quantize(
    const std::vector<std::vector<float>> & K,
    std::vector<std::vector<float>> & K_hat,
    int n_hi, int n_lo
) {
    int nk = (int)K.size();
    int dim = n_hi + n_lo;
    K_hat.resize(nk);

    for (int i = 0; i < nk; i++) {
        K_hat[i].resize(dim);

        std::vector<float> hi(n_hi), lo(n_lo);
        memcpy(hi.data(), K[i].data(), n_hi * sizeof(float));
        memcpy(lo.data(), K[i].data() + n_hi, n_lo * sizeof(float));

        // Hi: two-pass residual quantization
        // Pass 1: norm + FWHT + 4-bit MSE
        float norm_hi = 0.0f;
        for (int j = 0; j < n_hi; j++) norm_hi += hi[j] * hi[j];
        norm_hi = sqrtf(norm_hi);

        std::vector<float> hi_pass1(n_hi);
        if (norm_hi > 1e-30f) {
            float inv = 1.0f / norm_hi;
            for (int j = 0; j < n_hi; j++) hi_pass1[j] = hi[j] * inv;
        }
        fwht(hi_pass1.data(), n_hi);
        for (int j = 0; j < n_hi; j++) {
            int idx = nearest_centroid(hi_pass1[j], c16_d32, 16);
            hi_pass1[j] = c16_d32[idx];
        }
        fwht(hi_pass1.data(), n_hi);
        // Pass 1 reconstruction
        std::vector<float> recon1(n_hi);
        for (int j = 0; j < n_hi; j++) recon1[j] = hi_pass1[j] * norm_hi;

        // Pass 2: quantize the residual
        std::vector<float> residual(n_hi);
        for (int j = 0; j < n_hi; j++) residual[j] = hi[j] - recon1[j];

        float norm_res = 0.0f;
        for (int j = 0; j < n_hi; j++) norm_res += residual[j] * residual[j];
        norm_res = sqrtf(norm_res);

        std::vector<float> res_q(n_hi);
        if (norm_res > 1e-30f) {
            float inv = 1.0f / norm_res;
            for (int j = 0; j < n_hi; j++) res_q[j] = residual[j] * inv;
        }
        fwht(res_q.data(), n_hi);
        for (int j = 0; j < n_hi; j++) {
            int idx = nearest_centroid(res_q[j], c16_d32, 16);
            res_q[j] = c16_d32[idx];
        }
        fwht(res_q.data(), n_hi);

        // Final hi = pass1 + pass2
        for (int j = 0; j < n_hi; j++) {
            hi[j] = recon1[j] + res_q[j] * norm_res;
        }

        // Lo: 2-bit MSE + structured rotation (same as B)
        float norm_lo = 0.0f;
        for (int j = 0; j < n_lo; j++) norm_lo += lo[j] * lo[j];
        norm_lo = sqrtf(norm_lo);
        if (norm_lo > 1e-30f) {
            float inv = 1.0f / norm_lo;
            for (int j = 0; j < n_lo; j++) lo[j] *= inv;
        }
        if (n_lo == 96) {
            structured_rotate_lo(lo.data(), n_lo);
        } else if (n_lo % 32 == 0) {
            for (int b = 0; b < n_lo / 32; b++) fwht(lo.data() + b * 32, 32);
        }
        for (int j = 0; j < n_lo; j++) {
            int idx = nearest_centroid(lo[j], c2_d96, 2);
            lo[j] = c2_d96[idx];
        }
        if (n_lo == 96) {
            structured_unrotate_lo(lo.data(), n_lo);
        } else if (n_lo % 32 == 0) {
            for (int b = 0; b < n_lo / 32; b++) fwht(lo.data() + b * 32, 32);
        }
        for (int j = 0; j < n_lo; j++) lo[j] *= norm_lo;

        memcpy(K_hat[i].data(), hi.data(), n_hi * sizeof(float));
        memcpy(K_hat[i].data() + n_hi, lo.data(), n_lo * sizeof(float));
    }
}

// ---------------------------------------------------------------------------
// Architecture F: Log-scale quantization on hi
// ---------------------------------------------------------------------------

static void arch_f_quantize(
    const std::vector<std::vector<float>> & K,
    std::vector<std::vector<float>> & K_hat,
    int n_hi, int n_lo
) {
    int nk = (int)K.size();
    int dim = n_hi + n_lo;
    K_hat.resize(nk);

    for (int i = 0; i < nk; i++) {
        K_hat[i].resize(dim);

        std::vector<float> hi(n_hi), lo(n_lo);
        memcpy(hi.data(), K[i].data(), n_hi * sizeof(float));
        memcpy(lo.data(), K[i].data() + n_hi, n_lo * sizeof(float));

        // Hi: 1-bit sign + 7-bit log2(|x|) quantization
        // Find min/max log magnitude for scaling
        float log_min = FLT_MAX, log_max = -FLT_MAX;
        for (int j = 0; j < n_hi; j++) {
            float absv = fabsf(hi[j]);
            if (absv > 1e-30f) {
                float lv = log2f(absv);
                if (lv < log_min) log_min = lv;
                if (lv > log_max) log_max = lv;
            }
        }

        float log_range = log_max - log_min;
        if (log_range < 1e-10f) log_range = 1.0f;
        int n_levels = 128; // 7 bits

        for (int j = 0; j < n_hi; j++) {
            float absv = fabsf(hi[j]);
            float sign = (hi[j] >= 0.0f) ? 1.0f : -1.0f;
            if (absv < 1e-30f) {
                hi[j] = 0.0f;
                continue;
            }
            float lv = log2f(absv);
            // Quantize to 7-bit level
            float normalized = (lv - log_min) / log_range;
            int level = (int)(normalized * (n_levels - 1) + 0.5f);
            if (level < 0) level = 0;
            if (level >= n_levels) level = n_levels - 1;
            // Dequantize
            float dq_log = log_min + (float)level / (float)(n_levels - 1) * log_range;
            hi[j] = sign * exp2f(dq_log);
        }

        // Lo: 2-bit MSE + structured rotation (same as B)
        float norm_lo = 0.0f;
        for (int j = 0; j < n_lo; j++) norm_lo += lo[j] * lo[j];
        norm_lo = sqrtf(norm_lo);
        if (norm_lo > 1e-30f) {
            float inv = 1.0f / norm_lo;
            for (int j = 0; j < n_lo; j++) lo[j] *= inv;
        }
        if (n_lo == 96) {
            structured_rotate_lo(lo.data(), n_lo);
        } else if (n_lo % 32 == 0) {
            for (int b = 0; b < n_lo / 32; b++) fwht(lo.data() + b * 32, 32);
        }
        for (int j = 0; j < n_lo; j++) {
            int idx = nearest_centroid(lo[j], c2_d96, 2);
            lo[j] = c2_d96[idx];
        }
        if (n_lo == 96) {
            structured_unrotate_lo(lo.data(), n_lo);
        } else if (n_lo % 32 == 0) {
            for (int b = 0; b < n_lo / 32; b++) fwht(lo.data() + b * 32, 32);
        }
        for (int j = 0; j < n_lo; j++) lo[j] *= norm_lo;

        memcpy(K_hat[i].data(), hi.data(), n_hi * sizeof(float));
        memcpy(K_hat[i].data() + n_hi, lo.data(), n_lo * sizeof(float));
    }
}

// ---------------------------------------------------------------------------
// Architecture G: Adaptive split ratio sweep
// ---------------------------------------------------------------------------

// This architecture tests multiple splits; implemented via a wrapper that
// calls arch_b_quantize-style logic with different n_hi/n_lo.
// Each split variant is run separately in the main loop.

static void arch_g_quantize(
    const std::vector<std::vector<float>> & K,
    std::vector<std::vector<float>> & K_hat,
    int n_hi, int n_lo
) {
    // Generic: n_hi-bit hi with 4-bit MSE, n_lo with 2-bit MSE
    // n_hi can be 16, 32, 48, 64 — all powers-of-2 friendly for FWHT
    int nk = (int)K.size();
    int dim = n_hi + n_lo;
    K_hat.resize(nk);

    for (int i = 0; i < nk; i++) {
        K_hat[i].resize(dim);

        std::vector<float> hi(n_hi), lo(n_lo);
        memcpy(hi.data(), K[i].data(), n_hi * sizeof(float));
        memcpy(lo.data(), K[i].data() + n_hi, n_lo * sizeof(float));

        // Hi: norm + FWHT + 4-bit MSE
        float norm_hi = 0.0f;
        for (int j = 0; j < n_hi; j++) norm_hi += hi[j] * hi[j];
        norm_hi = sqrtf(norm_hi);
        if (norm_hi > 1e-30f) {
            float inv = 1.0f / norm_hi;
            for (int j = 0; j < n_hi; j++) hi[j] *= inv;
        }
        // FWHT if power of 2
        if (n_hi >= 2 && (n_hi & (n_hi - 1)) == 0) {
            fwht(hi.data(), n_hi);
        } else if (n_hi % 32 == 0) {
            for (int b = 0; b < n_hi / 32; b++) fwht(hi.data() + b * 32, 32);
        }
        for (int j = 0; j < n_hi; j++) {
            int idx = nearest_centroid(hi[j], c16_d32, 16);
            hi[j] = c16_d32[idx];
        }
        if (n_hi >= 2 && (n_hi & (n_hi - 1)) == 0) {
            fwht(hi.data(), n_hi);
        } else if (n_hi % 32 == 0) {
            for (int b = 0; b < n_hi / 32; b++) fwht(hi.data() + b * 32, 32);
        }
        for (int j = 0; j < n_hi; j++) hi[j] *= norm_hi;

        // Lo: norm + FWHT + 2-bit MSE
        float norm_lo = 0.0f;
        for (int j = 0; j < n_lo; j++) norm_lo += lo[j] * lo[j];
        norm_lo = sqrtf(norm_lo);
        if (norm_lo > 1e-30f) {
            float inv = 1.0f / norm_lo;
            for (int j = 0; j < n_lo; j++) lo[j] *= inv;
        }
        if (n_lo >= 2 && (n_lo & (n_lo - 1)) == 0) {
            fwht(lo.data(), n_lo);
        } else if (n_lo == 96) {
            structured_rotate_lo(lo.data(), n_lo);
        } else if (n_lo % 32 == 0) {
            for (int b = 0; b < n_lo / 32; b++) fwht(lo.data() + b * 32, 32);
        }
        for (int j = 0; j < n_lo; j++) {
            int idx = nearest_centroid(lo[j], c2_d96, 2);
            lo[j] = c2_d96[idx];
        }
        if (n_lo >= 2 && (n_lo & (n_lo - 1)) == 0) {
            fwht(lo.data(), n_lo);
        } else if (n_lo == 96) {
            structured_unrotate_lo(lo.data(), n_lo);
        } else if (n_lo % 32 == 0) {
            for (int b = 0; b < n_lo / 32; b++) fwht(lo.data() + b * 32, 32);
        }
        for (int j = 0; j < n_lo; j++) lo[j] *= norm_lo;

        memcpy(K_hat[i].data(), hi.data(), n_hi * sizeof(float));
        memcpy(K_hat[i].data() + n_hi, lo.data(), n_lo * sizeof(float));
    }
}

// ---------------------------------------------------------------------------
// Architecture H: Mixed precision — 8-bit uniform hi, 2-bit MSE lo
// ---------------------------------------------------------------------------

static void arch_h_quantize(
    const std::vector<std::vector<float>> & K,
    std::vector<std::vector<float>> & K_hat,
    int n_hi, int n_lo
) {
    int nk = (int)K.size();
    int dim = n_hi + n_lo;
    K_hat.resize(nk);

    // How many groups for hi scaling
    int group_sz = 8; // per-group scale, 8 channels per group
    int n_groups = (n_hi + group_sz - 1) / group_sz;

    for (int i = 0; i < nk; i++) {
        K_hat[i].resize(dim);

        std::vector<float> hi(n_hi), lo(n_lo);
        memcpy(hi.data(), K[i].data(), n_hi * sizeof(float));
        memcpy(lo.data(), K[i].data() + n_hi, n_lo * sizeof(float));

        // Hi: per-group 8-bit uniform quantization (256 levels)
        for (int g = 0; g < n_groups; g++) {
            int start = g * group_sz;
            int end = std::min(start + group_sz, n_hi);
            // Find min/max in group
            float gmin = hi[start], gmax = hi[start];
            for (int j = start + 1; j < end; j++) {
                if (hi[j] < gmin) gmin = hi[j];
                if (hi[j] > gmax) gmax = hi[j];
            }

            float range = gmax - gmin;
            if (range < 1e-30f) range = 1e-30f;

            // Quantize to 256 levels
            for (int j = start; j < end; j++) {
                float normalized = (hi[j] - gmin) / range;
                int level = (int)(normalized * 255.0f + 0.5f);
                if (level < 0) level = 0;
                if (level > 255) level = 255;
                hi[j] = gmin + (float)level / 255.0f * range;
            }
        }

        // Lo: norm + structured rotation + 2-bit MSE (same as B)
        float norm_lo = 0.0f;
        for (int j = 0; j < n_lo; j++) norm_lo += lo[j] * lo[j];
        norm_lo = sqrtf(norm_lo);
        if (norm_lo > 1e-30f) {
            float inv = 1.0f / norm_lo;
            for (int j = 0; j < n_lo; j++) lo[j] *= inv;
        }
        if (n_lo == 96) {
            structured_rotate_lo(lo.data(), n_lo);
        } else if (n_lo % 32 == 0) {
            for (int b = 0; b < n_lo / 32; b++) fwht(lo.data() + b * 32, 32);
        }
        for (int j = 0; j < n_lo; j++) {
            int idx = nearest_centroid(lo[j], c2_d96, 2);
            lo[j] = c2_d96[idx];
        }
        if (n_lo == 96) {
            structured_unrotate_lo(lo.data(), n_lo);
        } else if (n_lo % 32 == 0) {
            for (int b = 0; b < n_lo / 32; b++) fwht(lo.data() + b * 32, 32);
        }
        for (int j = 0; j < n_lo; j++) lo[j] *= norm_lo;

        memcpy(K_hat[i].data(), hi.data(), n_hi * sizeof(float));
        memcpy(K_hat[i].data() + n_hi, lo.data(), n_lo * sizeof(float));
    }
}

// ---------------------------------------------------------------------------
// Reference: q8_0 quantization (block-of-32, 8-bit uniform + scale)
// ---------------------------------------------------------------------------

static void ref_q8_0_quantize(
    const std::vector<std::vector<float>> & K,
    std::vector<std::vector<float>> & K_hat
) {
    int nk = (int)K.size();
    int dim = (int)K[0].size();
    K_hat.resize(nk);

    for (int i = 0; i < nk; i++) {
        K_hat[i].resize(dim);
        // q8_0: blocks of 32, per-block scale, 8-bit signed quants
        for (int b = 0; b < dim; b += 32) {
            int bsz = std::min(32, dim - b);
            float amax = 0.0f;
            for (int j = 0; j < bsz; j++) {
                float av = fabsf(K[i][b + j]);
                if (av > amax) amax = av;
            }
            float d = amax / 127.0f;
            float id = (d > 1e-30f) ? 1.0f / d : 0.0f;

            for (int j = 0; j < bsz; j++) {
                int q = (int)roundf(K[i][b + j] * id);
                if (q < -127) q = -127;
                if (q >  127) q =  127;
                K_hat[i][b + j] = (float)q * d;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Reference: q4_0 quantization (block-of-32, 4-bit unsigned + scale)
// ---------------------------------------------------------------------------

static void ref_q4_0_quantize(
    const std::vector<std::vector<float>> & K,
    std::vector<std::vector<float>> & K_hat
) {
    int nk = (int)K.size();
    int dim = (int)K[0].size();
    K_hat.resize(nk);

    for (int i = 0; i < nk; i++) {
        K_hat[i].resize(dim);
        for (int b = 0; b < dim; b += 32) {
            int bsz = std::min(32, dim - b);
            float amax = 0.0f;
            for (int j = 0; j < bsz; j++) {
                float av = fabsf(K[i][b + j]);
                if (av > amax) amax = av;
            }
            float d = amax / 7.0f; // 4-bit: -8..7 range, but q4_0 uses unsigned 0..15 with offset
            float id = (d > 1e-30f) ? 1.0f / d : 0.0f;

            for (int j = 0; j < bsz; j++) {
                // q4_0: x/d + 8, clamp to 0..15
                int q = (int)roundf(K[i][b + j] * id + 8.0f);
                if (q < 0)  q = 0;
                if (q > 15) q = 15;
                K_hat[i][b + j] = ((float)q - 8.0f) * d;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// BPV calculations
// ---------------------------------------------------------------------------

struct ArchInfo {
    const char * name;
    const char * desc;
    float bpv;
};

// BPV for 128-dim vectors with 32 hi / 96 lo split
static float bpv_arch_a(int n_hi, int n_lo) {
    // Hi: fp16 norm (2B) + 32*4bit (16B) + 32*1bit QJL (4B) = 22B
    // Lo: fp16 norm (2B) + fp16 rnorm (2B) + 96*3bit (36B) = 40B
    // Total: 62B for 128 dims
    (void)n_hi; (void)n_lo;
    return 62.0f * 8.0f / 128.0f; // 3.875 bpv
}

static float bpv_arch_b(int n_hi, int n_lo) {
    // Hi: fp16 norm (2B) + 32*4bit (16B) + 32*1bit QJL (4B) = 22B
    // Lo: fp16 norm (2B) + fp16 rnorm (2B) + 96*2bit (24B) + 96*1bit QJL (12B) = 40B
    // Total: 62B for 128 dims
    (void)n_hi; (void)n_lo;
    return 62.0f * 8.0f / 128.0f; // 3.875... wait
    // Actually tqk3_sjj:
    // Hi: fp16 norm (2B) + 32*4bit (16B) + 32*1bit QJL (4B) = 22B
    // Lo: fp16 norm (2B) + fp16 rnorm (2B) + 96*2bit (24B) + 96*1bit QJL (12B) = 40B
    // Total = 62B -> but spec says 3.750. Let me recalculate.
    // Actually: for sjj, lo uses 2-bit MSE (not 3-bit)
    // Lo: 2B norm + 2B rnorm + 24B qs + 12B signs = 40B
    // Hi: 2B norm + 16B qs + 4B signs = 22B
    // = 62B = 3.875 bpv
    // Hmm, the spec says 3.750 for sjj. Let me use the spec value.
}

static float calc_bpv(const char * arch, int n_hi, int n_lo) {
    int dim = n_hi + n_lo;
    (void)dim;

    if (strcmp(arch, "A") == 0) {
        // tqk3_sj: hi=4bit+QJL+norm, lo=3bit+structured+norm+rnorm
        float hi_bytes = 2.0f + n_hi * 4.0f / 8.0f + n_hi / 8.0f; // norm + 4bit + QJL
        float lo_bytes = 2.0f + 2.0f + n_lo * 3.0f / 8.0f;        // norm + rnorm + 3bit
        return (hi_bytes + lo_bytes) * 8.0f / dim;
    }
    if (strcmp(arch, "B") == 0) {
        // tqk3_sjj: hi=4bit+QJL+norm, lo=2bit+QJL+structured+norm+rnorm
        float hi_bytes = 2.0f + n_hi * 4.0f / 8.0f + n_hi / 8.0f;
        float lo_bytes = 2.0f + 2.0f + n_lo * 2.0f / 8.0f + n_lo / 8.0f;
        return (hi_bytes + lo_bytes) * 8.0f / dim;
    }
    if (strcmp(arch, "C") == 0) {
        // Per-group scaling: 4 groups * fp16 norm + 4bit MSE + QJL on hi
        // Lo: same as B
        int n_groups = 4;
        float hi_bytes = n_groups * 2.0f + n_hi * 4.0f / 8.0f + n_hi / 8.0f;
        float lo_bytes = 2.0f + 2.0f + n_lo * 2.0f / 8.0f + n_lo / 8.0f;
        return (hi_bytes + lo_bytes) * 8.0f / dim;
    }
    if (strcmp(arch, "D") == 0) {
        // 50/50 split: hi=5bit+QJL+norm, lo=2bit+QJL+norm
        int new_hi = dim / 2, new_lo = dim - new_hi;
        float hi_bytes = 2.0f + new_hi * 5.0f / 8.0f + new_hi / 8.0f;
        float lo_bytes = 2.0f + new_lo * 2.0f / 8.0f + new_lo / 8.0f;
        return (hi_bytes + lo_bytes) * 8.0f / dim;
    }
    if (strcmp(arch, "E") == 0) {
        // Residual: hi = 2*norm + 2*4bit MSE, lo = same as B
        float hi_bytes = 2.0f + n_hi * 4.0f / 8.0f   // pass 1: norm + 4bit
                       + 2.0f + n_hi * 4.0f / 8.0f;   // pass 2: norm + 4bit
        float lo_bytes = 2.0f + 2.0f + n_lo * 2.0f / 8.0f + n_lo / 8.0f;
        return (hi_bytes + lo_bytes) * 8.0f / dim;
    }
    if (strcmp(arch, "F") == 0) {
        // Log-scale: hi = 8bit (sign+log) + per-vec min/max (4B), lo = same as B
        float hi_bytes = 4.0f + n_hi * 1.0f; // 4B overhead + 1 byte per channel
        float lo_bytes = 2.0f + 2.0f + n_lo * 2.0f / 8.0f + n_lo / 8.0f;
        return (hi_bytes + lo_bytes) * 8.0f / dim;
    }
    if (strcmp(arch, "H") == 0) {
        // 8-bit uniform hi (per-group scale) + 2-bit lo
        int group_sz = 8;
        int n_groups = (n_hi + group_sz - 1) / group_sz;
        float hi_bytes = n_groups * 4.0f + n_hi * 1.0f; // per-group min+max (2*fp16) + 1 byte per element
        float lo_bytes = 2.0f + 2.0f + n_lo * 2.0f / 8.0f + n_lo / 8.0f;
        return (hi_bytes + lo_bytes) * 8.0f / dim;
    }
    return 0.0f;
}

// For adaptive split (arch G), compute bpv at given split
static float calc_bpv_g(int n_hi, int n_lo) {
    int dim = n_hi + n_lo;
    // Same layout as B but with variable split
    float hi_bytes = 2.0f + n_hi * 4.0f / 8.0f + n_hi / 8.0f;
    float lo_bytes = 2.0f + 2.0f + n_lo * 2.0f / 8.0f + n_lo / 8.0f;
    return (hi_bytes + lo_bytes) * 8.0f / dim;
}

// ---------------------------------------------------------------------------
// Print helpers
// ---------------------------------------------------------------------------

static void print_metrics(const char * arch, const char * desc, float bpv, Metrics m) {
    printf("  Architecture %s: %s\n", arch, desc);
    printf("    bpv:                  %.3f\n", bpv);
    printf("    dot_product_rel_err:  %.4f%%\n", m.dot_rel_error * 100.0);
    printf("    cosine_similarity:    %.6f\n", m.cosine_sim);
    printf("    rel_L2_error:         %.4f\n", m.rel_l2);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    printf("==========================================================\n");
    printf("  TurboQuant Architecture Search — Pure Math Exploration\n");
    printf("  Synthetic outlier distributions, 128-dim vectors\n");
    printf("==========================================================\n\n");

    const int N_VEC = 1000;
    const int N_DOT_PAIRS = 50000;

    // Outlier profiles to sweep
    struct Profile {
        const char * label;
        float sigma_hi;
        float sigma_lo;
        float expected_var_pct; // expected % of variance in hi channels
    };

    // With 32 hi channels and 96 lo channels:
    // var_pct = (32 * sigma_hi^2) / (32 * sigma_hi^2 + 96 * sigma_lo^2)
    Profile profiles[] = {
        { "50% var (uniform-ish)",  1.225f, 1.0f,  50.0f },
        { "70% var (moderate)",     2.871f, 1.0f,  70.0f },
        { "90% var (high)",         5.196f, 1.0f,  90.0f },
        { "99% var (extreme)",     17.146f, 1.0f,  99.0f },
        { "99.9% var (Qwen L0)",   54.77f, 1.0f,  99.9f },
    };
    int n_profiles = sizeof(profiles) / sizeof(profiles[0]);

    int n_hi = 32, n_lo = 96;

    for (int p = 0; p < n_profiles; p++) {
        Profile & prof = profiles[p];

        printf("══════════════════════════════════════════════════════════\n");
        printf("  Profile: %s\n", prof.label);
        printf("  sigma_hi=%.3f  sigma_lo=%.3f  split=%d/%d\n",
               prof.sigma_hi, prof.sigma_lo, n_hi, n_lo);
        // Verify variance fraction
        float var_hi = n_hi * prof.sigma_hi * prof.sigma_hi;
        float var_lo = n_lo * prof.sigma_lo * prof.sigma_lo;
        printf("  Actual var%%: %.2f%%\n", var_hi / (var_hi + var_lo) * 100.0f);
        printf("══════════════════════════════════════════════════════════\n\n");

        // Generate data
        std::mt19937 rng(42 + p);
        std::vector<std::vector<float>> K, Q;
        generate_vectors(K, N_VEC, n_hi, n_lo, prof.sigma_hi, prof.sigma_lo, rng);
        generate_vectors(Q, N_VEC, n_hi, n_lo, prof.sigma_hi, prof.sigma_lo, rng);

        // --- Reference: q8_0 ---
        {
            std::vector<std::vector<float>> K_hat;
            ref_q8_0_quantize(K, K_hat);
            Metrics m = compute_metrics(K, K_hat, Q, N_DOT_PAIRS);
            print_metrics("REF-q8_0", "q8_0 baseline (8.50 bpv)", 8.50f, m);
            printf("\n");
        }

        // --- Reference: q4_0 ---
        {
            std::vector<std::vector<float>> K_hat;
            ref_q4_0_quantize(K, K_hat);
            Metrics m = compute_metrics(K, K_hat, Q, N_DOT_PAIRS);
            print_metrics("REF-q4_0", "q4_0 baseline (4.50 bpv)", 4.50f, m);
            printf("\n");
        }

        // --- Architecture A: tqk3_sj ---
        {
            std::vector<std::vector<float>> K_hat;
            arch_a_quantize(K, K_hat, n_hi, n_lo);
            float bpv = calc_bpv("A", n_hi, n_lo);
            Metrics m = compute_metrics(K, K_hat, Q, N_DOT_PAIRS);
            print_metrics("A", "tqk3_sj (4bit hi MSE+FWHT+QJL, 3bit lo structured)", bpv, m);
            printf("\n");
        }

        // --- Architecture B: tqk3_sjj ---
        {
            std::vector<std::vector<float>> K_hat;
            arch_b_quantize(K, K_hat, n_hi, n_lo);
            float bpv = calc_bpv("B", n_hi, n_lo);
            Metrics m = compute_metrics(K, K_hat, Q, N_DOT_PAIRS);
            print_metrics("B", "tqk3_sjj (4bit hi, 2bit+QJL lo)", bpv, m);
            printf("\n");
        }

        // --- Architecture C: Per-group scaling on hi ---
        {
            std::vector<std::vector<float>> K_hat;
            arch_c_quantize(K, K_hat, n_hi, n_lo);
            float bpv = calc_bpv("C", n_hi, n_lo);
            Metrics m = compute_metrics(K, K_hat, Q, N_DOT_PAIRS);
            print_metrics("C", "per-group(4x8) norm on hi + 4bit MSE + 2bit lo", bpv, m);
            printf("\n");
        }

        // --- Architecture D: Wider split (50/50) ---
        {
            std::vector<std::vector<float>> K_hat;
            arch_d_quantize(K, K_hat, n_hi, n_lo);
            float bpv = calc_bpv("D", n_hi, n_lo);
            Metrics m = compute_metrics(K, K_hat, Q, N_DOT_PAIRS);
            print_metrics("D", "50/50 split — 5bit hi + 2bit lo", bpv, m);
            printf("\n");
        }

        // --- Architecture E: Residual quantization on hi ---
        {
            std::vector<std::vector<float>> K_hat;
            arch_e_quantize(K, K_hat, n_hi, n_lo);
            float bpv = calc_bpv("E", n_hi, n_lo);
            Metrics m = compute_metrics(K, K_hat, Q, N_DOT_PAIRS);
            print_metrics("E", "residual 2x4bit hi + 2bit lo", bpv, m);
            printf("\n");
        }

        // --- Architecture F: Log-scale quantization on hi ---
        {
            std::vector<std::vector<float>> K_hat;
            arch_f_quantize(K, K_hat, n_hi, n_lo);
            float bpv = calc_bpv("F", n_hi, n_lo);
            Metrics m = compute_metrics(K, K_hat, Q, N_DOT_PAIRS);
            print_metrics("F", "log-scale 8bit hi + 2bit lo", bpv, m);
            printf("\n");
        }

        // --- Architecture G: Adaptive split sweep ---
        {
            printf("  Architecture G: adaptive split ratio sweep\n");
            int splits[][2] = { {16, 112}, {32, 96}, {48, 80}, {64, 64} };
            for (int s = 0; s < 4; s++) {
                int shi = splits[s][0], slo = splits[s][1];
                // Re-generate data with the right hi/lo split arrangement
                std::mt19937 rng_g(42 + p);
                std::vector<std::vector<float>> K_g, Q_g;
                generate_vectors(K_g, N_VEC, shi, slo, prof.sigma_hi, prof.sigma_lo, rng_g);
                generate_vectors(Q_g, N_VEC, shi, slo, prof.sigma_hi, prof.sigma_lo, rng_g);

                std::vector<std::vector<float>> K_hat;
                arch_g_quantize(K_g, K_hat, shi, slo);
                float bpv = calc_bpv_g(shi, slo);
                Metrics m = compute_metrics(K_g, K_hat, Q_g, N_DOT_PAIRS);

                char label[128];
                snprintf(label, sizeof(label), "G-%d/%d", shi, slo);
                char desc[256];
                snprintf(desc, sizeof(desc), "split %d/%d — 4bit hi + 2bit lo", shi, slo);
                print_metrics(label, desc, bpv, m);
            }
            printf("\n");
        }

        // --- Architecture H: Mixed precision 8-bit hi + 2-bit lo ---
        {
            std::vector<std::vector<float>> K_hat;
            arch_h_quantize(K, K_hat, n_hi, n_lo);
            float bpv = calc_bpv("H", n_hi, n_lo);
            Metrics m = compute_metrics(K, K_hat, Q, N_DOT_PAIRS);
            print_metrics("H", "8bit uniform hi (per-group scale) + 2bit lo", bpv, m);
            printf("\n");
        }

        printf("\n");
    }

    // --- Summary table: cosine similarity (more stable than dot-rel-error for heavy tails) ---
    printf("══════════════════════════════════════════════════════════════════════════════════════════\n");
    printf("  SUMMARY: cosine similarity by architecture (higher = better)\n");
    printf("══════════════════════════════════════════════════════════════════════════════════════════\n\n");

    const char * arch_labels[] = {"q8_0", "q4_0", "A", "B", "C", "D", "E", "F", "H"};
    const float  arch_bpvs[]   = {8.50f, 4.50f,
        calc_bpv("A", n_hi, n_lo), calc_bpv("B", n_hi, n_lo),
        calc_bpv("C", n_hi, n_lo), calc_bpv("D", n_hi, n_lo),
        calc_bpv("E", n_hi, n_lo), calc_bpv("F", n_hi, n_lo),
        calc_bpv("H", n_hi, n_lo)};

    printf("%-24s", "Variance profile");
    for (int a = 0; a < 9; a++) {
        printf("  %s(%4.2f)", arch_labels[a], arch_bpvs[a]);
    }
    printf("\n");
    for (int i = 0; i < 24 + 9 * 12; i++) printf("-");
    printf("\n");

    for (int p = 0; p < n_profiles; p++) {
        Profile & prof = profiles[p];
        std::mt19937 rng(42 + p);
        std::vector<std::vector<float>> K, Q;
        generate_vectors(K, N_VEC, n_hi, n_lo, prof.sigma_hi, prof.sigma_lo, rng);
        generate_vectors(Q, N_VEC, n_hi, n_lo, prof.sigma_hi, prof.sigma_lo, rng);

        printf("%-24s", prof.label);

        // q8_0
        {
            std::vector<std::vector<float>> K_hat;
            ref_q8_0_quantize(K, K_hat);
            Metrics m = compute_metrics(K, K_hat, Q, N_DOT_PAIRS);
            printf("    %8.6f", m.cosine_sim);
        }
        // q4_0
        {
            std::vector<std::vector<float>> K_hat;
            ref_q4_0_quantize(K, K_hat);
            Metrics m = compute_metrics(K, K_hat, Q, N_DOT_PAIRS);
            printf("    %8.6f", m.cosine_sim);
        }
        // A
        {
            std::vector<std::vector<float>> K_hat;
            arch_a_quantize(K, K_hat, n_hi, n_lo);
            Metrics m = compute_metrics(K, K_hat, Q, N_DOT_PAIRS);
            printf("    %8.6f", m.cosine_sim);
        }
        // B
        {
            std::vector<std::vector<float>> K_hat;
            arch_b_quantize(K, K_hat, n_hi, n_lo);
            Metrics m = compute_metrics(K, K_hat, Q, N_DOT_PAIRS);
            printf("    %8.6f", m.cosine_sim);
        }
        // C
        {
            std::vector<std::vector<float>> K_hat;
            arch_c_quantize(K, K_hat, n_hi, n_lo);
            Metrics m = compute_metrics(K, K_hat, Q, N_DOT_PAIRS);
            printf("    %8.6f", m.cosine_sim);
        }
        // D
        {
            std::vector<std::vector<float>> K_hat;
            arch_d_quantize(K, K_hat, n_hi, n_lo);
            Metrics m = compute_metrics(K, K_hat, Q, N_DOT_PAIRS);
            printf("    %8.6f", m.cosine_sim);
        }
        // E
        {
            std::vector<std::vector<float>> K_hat;
            arch_e_quantize(K, K_hat, n_hi, n_lo);
            Metrics m = compute_metrics(K, K_hat, Q, N_DOT_PAIRS);
            printf("    %8.6f", m.cosine_sim);
        }
        // F
        {
            std::vector<std::vector<float>> K_hat;
            arch_f_quantize(K, K_hat, n_hi, n_lo);
            Metrics m = compute_metrics(K, K_hat, Q, N_DOT_PAIRS);
            printf("    %8.6f", m.cosine_sim);
        }
        // H
        {
            std::vector<std::vector<float>> K_hat;
            arch_h_quantize(K, K_hat, n_hi, n_lo);
            Metrics m = compute_metrics(K, K_hat, Q, N_DOT_PAIRS);
            printf("    %8.6f", m.cosine_sim);
        }

        printf("\n");
    }

    // --- Summary table: relative L2 error ---
    printf("\n");
    printf("══════════════════════════════════════════════════════════════════════════════════════════\n");
    printf("  SUMMARY: relative L2 error by architecture (lower = better)\n");
    printf("══════════════════════════════════════════════════════════════════════════════════════════\n\n");

    printf("%-24s", "Variance profile");
    for (int a = 0; a < 9; a++) {
        printf("  %s(%4.2f)", arch_labels[a], arch_bpvs[a]);
    }
    printf("\n");
    for (int i = 0; i < 24 + 9 * 12; i++) printf("-");
    printf("\n");

    for (int p = 0; p < n_profiles; p++) {
        Profile & prof = profiles[p];
        std::mt19937 rng(42 + p);
        std::vector<std::vector<float>> K, Q;
        generate_vectors(K, N_VEC, n_hi, n_lo, prof.sigma_hi, prof.sigma_lo, rng);
        generate_vectors(Q, N_VEC, n_hi, n_lo, prof.sigma_hi, prof.sigma_lo, rng);

        printf("%-24s", prof.label);

        // q8_0
        {
            std::vector<std::vector<float>> K_hat;
            ref_q8_0_quantize(K, K_hat);
            Metrics m = compute_metrics(K, K_hat, Q, N_DOT_PAIRS);
            printf("    %8.4f", m.rel_l2);
        }
        // q4_0
        {
            std::vector<std::vector<float>> K_hat;
            ref_q4_0_quantize(K, K_hat);
            Metrics m = compute_metrics(K, K_hat, Q, N_DOT_PAIRS);
            printf("    %8.4f", m.rel_l2);
        }
        // A
        {
            std::vector<std::vector<float>> K_hat;
            arch_a_quantize(K, K_hat, n_hi, n_lo);
            Metrics m = compute_metrics(K, K_hat, Q, N_DOT_PAIRS);
            printf("    %8.4f", m.rel_l2);
        }
        // B
        {
            std::vector<std::vector<float>> K_hat;
            arch_b_quantize(K, K_hat, n_hi, n_lo);
            Metrics m = compute_metrics(K, K_hat, Q, N_DOT_PAIRS);
            printf("    %8.4f", m.rel_l2);
        }
        // C
        {
            std::vector<std::vector<float>> K_hat;
            arch_c_quantize(K, K_hat, n_hi, n_lo);
            Metrics m = compute_metrics(K, K_hat, Q, N_DOT_PAIRS);
            printf("    %8.4f", m.rel_l2);
        }
        // D
        {
            std::vector<std::vector<float>> K_hat;
            arch_d_quantize(K, K_hat, n_hi, n_lo);
            Metrics m = compute_metrics(K, K_hat, Q, N_DOT_PAIRS);
            printf("    %8.4f", m.rel_l2);
        }
        // E
        {
            std::vector<std::vector<float>> K_hat;
            arch_e_quantize(K, K_hat, n_hi, n_lo);
            Metrics m = compute_metrics(K, K_hat, Q, N_DOT_PAIRS);
            printf("    %8.4f", m.rel_l2);
        }
        // F
        {
            std::vector<std::vector<float>> K_hat;
            arch_f_quantize(K, K_hat, n_hi, n_lo);
            Metrics m = compute_metrics(K, K_hat, Q, N_DOT_PAIRS);
            printf("    %8.4f", m.rel_l2);
        }
        // H
        {
            std::vector<std::vector<float>> K_hat;
            arch_h_quantize(K, K_hat, n_hi, n_lo);
            Metrics m = compute_metrics(K, K_hat, Q, N_DOT_PAIRS);
            printf("    %8.4f", m.rel_l2);
        }

        printf("\n");
    }

    printf("\n");

    printf("Done. Use these results to pick the best architecture for extreme outlier layers.\n");
    return 0;
}
