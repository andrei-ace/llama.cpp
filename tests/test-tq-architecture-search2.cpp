// TurboQuant Architecture Search 2 — residual MSE + QJL research
// Explores multi-pass MSE with QJL correction on hi channels.
// Key insight: MSE has attenuation bias, QJL provides unbiased correction.
// No model inference, no ggml dependencies — fully self-contained.
//
// Build: cmake --build build -t test-tq-architecture-search2 -j14
// Run:   ./build/bin/test-tq-architecture-search2

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
#include <cassert>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

static constexpr int N_HI  = 32;
static constexpr int N_LO  = 96;
static constexpr int DIM   = N_HI + N_LO;
static constexpr int N_VEC = 1000;
static constexpr int N_DOT_PAIRS = 50000;
static constexpr float SQRT_PI_OVER_2 = 1.2533141373155003f;

// ---------------------------------------------------------------------------
// Lloyd-Max centroids: d=32
// ---------------------------------------------------------------------------

// 4 centroids (2-bit)
static const float c4_d32[4] = {
    -0.2633f, -0.0798f, 0.0798f, 0.2633f
};

// 8 centroids (3-bit)
static const float c8_d32[8] = {
    -0.3663f, -0.2325f, -0.1318f, -0.0429f, 0.0429f, 0.1318f, 0.2325f, 0.3663f
};

// 16 centroids (4-bit)
static const float c16_d32[16] = {
    -0.4534f, -0.3499f, -0.2765f, -0.2161f, -0.1629f, -0.1138f, -0.0674f, -0.0223f,
     0.0223f,  0.0674f,  0.1138f,  0.1629f,  0.2161f,  0.2765f,  0.3499f,  0.4534f
};

// 32 centroids (5-bit)
static const float c32_d32[32] = {
    -0.5265f, -0.4434f, -0.3864f, -0.3409f, -0.3020f, -0.2676f, -0.2361f, -0.2070f,
    -0.1795f, -0.1534f, -0.1283f, -0.1040f, -0.0803f, -0.0571f, -0.0341f, -0.0114f,
     0.0114f,  0.0341f,  0.0571f,  0.0803f,  0.1040f,  0.1283f,  0.1534f,  0.1795f,
     0.2070f,  0.2361f,  0.2676f,  0.3020f,  0.3409f,  0.3864f,  0.4434f,  0.5265f
};

// d=96 Lloyd-Max centroids
static const float c8_d96[8] = {
    -0.2169f, -0.1362f, -0.0768f, -0.0249f, 0.0249f, 0.0768f, 0.1362f, 0.2169f
};

// ---------------------------------------------------------------------------
// Approximate inverse normal CDF (Beasley-Springer-Moro algorithm)
// ---------------------------------------------------------------------------

static float approx_inv_normal(float p) {
    // Rational approximation for the central region
    if (p <= 0.0f) return -6.0f;
    if (p >= 1.0f) return  6.0f;

    float t;
    if (p < 0.5f) {
        t = sqrtf(-2.0f * logf(p));
        // Approximation for lower tail
        float c0 = 2.515517f, c1 = 0.802853f, c2 = 0.010328f;
        float d1 = 1.432788f, d2 = 0.189269f, d3 = 0.001308f;
        return -(t - (c0 + c1*t + c2*t*t) / (1.0f + d1*t + d2*t*t + d3*t*t*t));
    } else {
        t = sqrtf(-2.0f * logf(1.0f - p));
        float c0 = 2.515517f, c1 = 0.802853f, c2 = 0.010328f;
        float d1 = 1.432788f, d2 = 0.189269f, d3 = 0.001308f;
        return t - (c0 + c1*t + c2*t*t) / (1.0f + d1*t + d2*t*t + d3*t*t*t);
    }
}

// ---------------------------------------------------------------------------
// Generate centroids for 6-bit (64) and 8-bit (256) via quantile spacing
// ---------------------------------------------------------------------------

static void generate_centroids_d32(float * out, int n_centroids) {
    float sigma = 1.0f / sqrtf(32.0f);
    for (int i = 0; i < n_centroids; i++) {
        float p = ((float)i + 0.5f) / (float)n_centroids;
        out[i] = sigma * approx_inv_normal(p);
    }
}

// Storage for generated centroids
static float c64_d32[64];   // 6-bit
static float c256_d32[256]; // 8-bit

static void init_generated_centroids() {
    generate_centroids_d32(c64_d32, 64);
    generate_centroids_d32(c256_d32, 256);
}

// ---------------------------------------------------------------------------
// Centroid lookup helper
// ---------------------------------------------------------------------------

struct CentroidSet {
    const float * data;
    int           count;
};

static CentroidSet get_centroids_d32(int bits) {
    switch (bits) {
        case 2: return { c4_d32,   4 };
        case 3: return { c8_d32,   8 };
        case 4: return { c16_d32, 16 };
        case 5: return { c32_d32, 32 };
        case 6: return { c64_d32, 64 };
        case 8: return { c256_d32, 256 };
        default: assert(false && "unsupported bit width"); return { nullptr, 0 };
    }
}

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
// Utility: nearest centroid
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

// ---------------------------------------------------------------------------
// Data generation
// ---------------------------------------------------------------------------

static void generate_vectors(
    std::vector<std::vector<float>> & vecs,
    int count, float sigma_hi, float sigma_lo,
    std::mt19937 & rng
) {
    std::normal_distribution<float> dist_hi(0.0f, sigma_hi);
    std::normal_distribution<float> dist_lo(0.0f, sigma_lo);
    vecs.resize(count);
    for (int i = 0; i < count; i++) {
        vecs[i].resize(DIM);
        for (int j = 0; j < N_HI; j++) vecs[i][j] = dist_hi(rng);
        for (int j = 0; j < N_LO; j++) vecs[i][N_HI + j] = dist_lo(rng);
    }
}

// ---------------------------------------------------------------------------
// Quality metrics
// ---------------------------------------------------------------------------

struct Metrics {
    double dot_rel_error;
    double cosine_sim;
    double rel_l2;
};

static Metrics compute_metrics(
    const std::vector<std::vector<float>> & K,
    const std::vector<std::vector<float>> & K_hat,
    const std::vector<std::vector<float>> & Q
) {
    int nk = (int)K.size();
    int nq = (int)Q.size();

    double cos_sum = 0.0, l2_sum = 0.0;
    for (int i = 0; i < nk; i++) {
        double dot_kk = 0, dot_kh = 0, dot_hh = 0, diff2 = 0, knorm2 = 0;
        for (int d = 0; d < DIM; d++) {
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

    double dot_err_sum = 0.0;
    int dot_count = 0;
    std::mt19937 rng(12345);
    for (int p = 0; p < N_DOT_PAIRS; p++) {
        int qi = rng() % nq;
        int ki = rng() % nk;
        double dot_true = 0.0, dot_quant = 0.0;
        for (int d = 0; d < DIM; d++) {
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
// Lo path: always 3-bit MSE (8 centroids d=96), structured rotation, no QJL
// ---------------------------------------------------------------------------

static void quantize_lo(const float * in, float * out) {
    float lo[N_LO];
    memcpy(lo, in, N_LO * sizeof(float));

    float norm = 0.0f;
    for (int j = 0; j < N_LO; j++) norm += lo[j] * lo[j];
    norm = sqrtf(norm);
    if (norm < 1e-30f) { memset(out, 0, N_LO * sizeof(float)); return; }

    float inv = 1.0f / norm;
    for (int j = 0; j < N_LO; j++) lo[j] *= inv;

    structured_rotate_lo(lo, N_LO);

    for (int j = 0; j < N_LO; j++) {
        int idx = nearest_centroid(lo[j], c8_d96, 8);
        lo[j] = c8_d96[idx];
    }

    structured_unrotate_lo(lo, N_LO);

    for (int j = 0; j < N_LO; j++) out[j] = lo[j] * norm;
}

// ---------------------------------------------------------------------------
// Hi path building blocks
// ---------------------------------------------------------------------------

// Single MSE pass on d=32 with FWHT: normalize, rotate, quantize, unrotate, denormalize
// Returns the reconstruction and the norm used for normalization.
static void mse_pass(const float * in, float * recon, float * out_norm, int bits) {
    CentroidSet cs = get_centroids_d32(bits);
    float hi[N_HI];
    memcpy(hi, in, N_HI * sizeof(float));

    float norm = 0.0f;
    for (int j = 0; j < N_HI; j++) norm += hi[j] * hi[j];
    norm = sqrtf(norm);
    *out_norm = norm;

    if (norm < 1e-30f) { memset(recon, 0, N_HI * sizeof(float)); return; }

    float inv = 1.0f / norm;
    for (int j = 0; j < N_HI; j++) hi[j] *= inv;

    fwht(hi, N_HI);

    for (int j = 0; j < N_HI; j++) {
        int idx = nearest_centroid(hi[j], cs.data, cs.count);
        hi[j] = cs.data[idx];
    }

    fwht(hi, N_HI);

    for (int j = 0; j < N_HI; j++) recon[j] = hi[j] * norm;
}

// QJL on residual: FWHT, store signs, reconstruct via sqrt(pi/2)/d * rnorm * sign_pattern
// The reconstruction is added to out[].
static void qjl_correct(const float * residual, float * out) {
    float r[N_HI];
    memcpy(r, residual, N_HI * sizeof(float));

    float rnorm = 0.0f;
    for (int j = 0; j < N_HI; j++) rnorm += r[j] * r[j];
    rnorm = sqrtf(rnorm);
    if (rnorm < 1e-30f) return;

    // FWHT the residual to get projected coefficients
    fwht(r, N_HI);

    // Store signs
    uint8_t signs[4] = {0, 0, 0, 0}; // 32 bits = 4 bytes
    for (int j = 0; j < N_HI; j++) {
        if (r[j] >= 0.0f) signs[j / 8] |= (1 << (j % 8));
    }

    // Reconstruct: inverse FWHT of sign pattern, scaled
    float corr[N_HI];
    for (int j = 0; j < N_HI; j++) {
        corr[j] = ((signs[j / 8] >> (j % 8)) & 1) ? 1.0f : -1.0f;
    }
    fwht(corr, N_HI);

    float scale = SQRT_PI_OVER_2 / (float)N_HI * rnorm;
    for (int j = 0; j < N_HI; j++) out[j] += scale * corr[j];
}

// ---------------------------------------------------------------------------
// Architecture implementations
// ---------------------------------------------------------------------------

// Generic multi-pass residual MSE + optional QJL on hi
// pass_bits: array of bit widths for each MSE pass
// n_passes: number of MSE passes
// use_qjl: whether to apply QJL on the final residual
static void quantize_hi_multipass(
    const float * in, float * out,
    const int * pass_bits, int n_passes, bool use_qjl
) {
    float recon_total[N_HI];
    memset(recon_total, 0, N_HI * sizeof(float));

    float residual[N_HI];
    memcpy(residual, in, N_HI * sizeof(float));

    for (int p = 0; p < n_passes; p++) {
        float pass_recon[N_HI];
        float pass_norm;
        mse_pass(residual, pass_recon, &pass_norm, pass_bits[p]);

        for (int j = 0; j < N_HI; j++) {
            recon_total[j] += pass_recon[j];
            residual[j]    -= pass_recon[j];
        }
    }

    memcpy(out, recon_total, N_HI * sizeof(float));

    if (use_qjl) {
        qjl_correct(residual, out);
    }
}

// Full vector quantization: split into hi/lo, quantize each
static void quantize_vector(
    const float * in, float * out,
    const int * hi_pass_bits, int hi_n_passes, bool hi_use_qjl
) {
    // Hi path
    quantize_hi_multipass(in, out, hi_pass_bits, hi_n_passes, hi_use_qjl);
    // Lo path
    quantize_lo(in + N_HI, out + N_HI);
}

// ---------------------------------------------------------------------------
// Reference quantizers
// ---------------------------------------------------------------------------

static void ref_q8_0_quantize(
    const std::vector<std::vector<float>> & K,
    std::vector<std::vector<float>> & K_hat
) {
    int nk = (int)K.size();
    K_hat.resize(nk);
    for (int i = 0; i < nk; i++) {
        K_hat[i].resize(DIM);
        for (int b = 0; b < DIM; b += 32) {
            int bsz = std::min(32, DIM - b);
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

static void ref_q4_0_quantize(
    const std::vector<std::vector<float>> & K,
    std::vector<std::vector<float>> & K_hat
) {
    int nk = (int)K.size();
    K_hat.resize(nk);
    for (int i = 0; i < nk; i++) {
        K_hat[i].resize(DIM);
        for (int b = 0; b < DIM; b += 32) {
            int bsz = std::min(32, DIM - b);
            float amax = 0.0f;
            for (int j = 0; j < bsz; j++) {
                float av = fabsf(K[i][b + j]);
                if (av > amax) amax = av;
            }
            float d = amax / 7.0f;
            float id = (d > 1e-30f) ? 1.0f / d : 0.0f;
            for (int j = 0; j < bsz; j++) {
                int q = (int)roundf(K[i][b + j] * id + 8.0f);
                if (q < 0)  q = 0;
                if (q > 15) q = 15;
                K_hat[i][b + j] = ((float)q - 8.0f) * d;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Architecture descriptor
// ---------------------------------------------------------------------------

struct Architecture {
    const char * name;
    const char * desc;
    float        bpv;
    int          hi_pass_bits[4]; // up to 4 passes
    int          hi_n_passes;
    bool         hi_use_qjl;
    bool         is_ref;         // uses ref quantizer instead
    int          ref_type;       // 0=q8_0, 1=q4_0
};

// ---------------------------------------------------------------------------
// Run all architectures
// ---------------------------------------------------------------------------

struct Result {
    const char * name;
    float        bpv;
    Metrics      metrics;
};

static void run_architecture(
    const Architecture & arch,
    const std::vector<std::vector<float>> & K,
    const std::vector<std::vector<float>> & /*Q*/,
    std::vector<std::vector<float>> & K_hat
) {
    int nk = (int)K.size();
    K_hat.resize(nk);

    if (arch.is_ref) {
        if (arch.ref_type == 0) {
            ref_q8_0_quantize(K, K_hat);
        } else {
            ref_q4_0_quantize(K, K_hat);
        }
        return;
    }

    for (int i = 0; i < nk; i++) {
        K_hat[i].resize(DIM);
        quantize_vector(K[i].data(), K_hat[i].data(),
                        arch.hi_pass_bits, arch.hi_n_passes, arch.hi_use_qjl);
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    init_generated_centroids();

    printf("================================================================\n");
    printf("  TurboQuant Architecture Search 2\n");
    printf("  Residual MSE + QJL — Pareto Frontier Exploration\n");
    printf("  128-dim, 32/96 split, FWHT-32 hi, bare 3xFWHT lo\n");
    printf("================================================================\n\n");

    // Define all architectures
    // The hi_pass_bits arrays hold bit widths per pass; hi_n_passes is the count.

    Architecture archs[] = {
        // --- Baselines ---
        { "q8_0",     "uniform 8-bit baseline",                           8.50f, {}, 0, false, true, 0 },
        { "q4_0",     "uniform 4-bit baseline",                           4.50f, {}, 0, false, true, 1 },
        { "tqk3_sj",  "4-bit MSE hi + QJL, 3-bit lo (62B)",              0.0f, {4}, 1, true,  false, 0 },
        { "tqk4_sj",  "5-bit MSE hi + QJL, 3-bit lo (66B)",              0.0f, {5}, 1, true,  false, 0 },

        // --- E variants: multi-pass residual MSE + QJL ---
        { "E_4_4",    "4-bit + 4-bit MSE + QJL",                         0.0f, {4, 4},    2, true,  false, 0 },
        { "E_4_3",    "4-bit + 3-bit MSE + QJL",                         0.0f, {4, 3},    2, true,  false, 0 },
        { "E_4_2",    "4-bit + 2-bit MSE + QJL",                         0.0f, {4, 2},    2, true,  false, 0 },
        { "E_3_3",    "3-bit + 3-bit MSE + QJL",                         0.0f, {3, 3},    2, true,  false, 0 },
        { "E_5_4",    "5-bit + 4-bit MSE + QJL",                         0.0f, {5, 4},    2, true,  false, 0 },
        { "E_5_3",    "5-bit + 3-bit MSE + QJL",                         0.0f, {5, 3},    2, true,  false, 0 },
        { "E_3_3_3",  "3-bit x3 passes MSE + QJL",                       0.0f, {3, 3, 3}, 3, true,  false, 0 },

        // --- E variants WITHOUT QJL (ablation) ---
        { "E_4_4_nq", "4-bit + 4-bit MSE, no QJL",                       0.0f, {4, 4},    2, false, false, 0 },
        { "E_4_3_nq", "4-bit + 3-bit MSE, no QJL",                       0.0f, {4, 3},    2, false, false, 0 },
        { "E_4_2_nq", "4-bit + 2-bit MSE, no QJL",                       0.0f, {4, 2},    2, false, false, 0 },
        { "E_3_3_nq", "3-bit + 3-bit MSE, no QJL",                       0.0f, {3, 3},    2, false, false, 0 },
        { "E_5_4_nq", "5-bit + 4-bit MSE, no QJL",                       0.0f, {5, 4},    2, false, false, 0 },
        { "E_5_3_nq", "5-bit + 3-bit MSE, no QJL",                       0.0f, {5, 3},    2, false, false, 0 },
        { "E_3_3_3_nq", "3-bit x3 passes MSE, no QJL",                   0.0f, {3, 3, 3}, 3, false, false, 0 },

        // --- Single-pass MSE + QJL at various bit widths ---
        { "S_3_QJL",  "3-bit MSE + QJL (single pass)",                   0.0f, {3}, 1, true,  false, 0 },
        { "S_4_QJL",  "4-bit MSE + QJL (single pass)",                   0.0f, {4}, 1, true,  false, 0 },
        { "S_5_QJL",  "5-bit MSE + QJL (single pass)",                   0.0f, {5}, 1, true,  false, 0 },
        { "S_6_QJL",  "6-bit MSE + QJL (single pass)",                   0.0f, {6}, 1, true,  false, 0 },
        { "S_8_QJL",  "8-bit MSE + QJL (single pass)",                   0.0f, {8}, 1, true,  false, 0 },

        // --- Single-pass MSE without QJL (for comparison) ---
        { "S_3_nq",   "3-bit MSE, no QJL",                               0.0f, {3}, 1, false, false, 0 },
        { "S_4_nq",   "4-bit MSE, no QJL",                               0.0f, {4}, 1, false, false, 0 },
        { "S_5_nq",   "5-bit MSE, no QJL",                               0.0f, {5}, 1, false, false, 0 },
        { "S_6_nq",   "6-bit MSE, no QJL",                               0.0f, {6}, 1, false, false, 0 },
        { "S_8_nq",   "8-bit MSE, no QJL",                               0.0f, {8}, 1, false, false, 0 },
    };

    int n_archs = (int)(sizeof(archs) / sizeof(archs[0]));

    // BPV formula from spec:
    //   (norm_hi(2) + norm_lo(2) + rnorm2_hi(2) + rnorm_qjl_hi(2) +
    //    qs_hi_all_passes + qs_lo(36) + signs_hi(4)) * 8 / 128
    //
    // - norm_hi: always present (pass 1 normalization)
    // - norm_lo: always present
    // - rnorm2_hi: present only if n_passes > 1 (single fp16 for residual chain)
    // - rnorm_qjl_hi: present only if QJL is used
    // - signs_hi: 32 bits = 4 bytes, present only if QJL is used
    // - qs_lo: 96 * 3-bit = 36 bytes, always present

    printf("BPV formula: (norm_hi(2) + norm_lo(2) + [rnorm2(2)] + [rnorm_qjl(2)] +\n");
    printf("              qs_hi_all_passes + qs_lo(36) + [signs(4)]) * 8 / 128\n\n");

    for (int a = 0; a < n_archs; a++) {
        if (archs[a].hi_n_passes > 0) {
            // Fixed: norm_hi(2) + norm_lo(2) + qs_lo(36)
            float bytes = 2.0f + 2.0f + 36.0f;

            // qs_hi for all passes
            for (int p = 0; p < archs[a].hi_n_passes; p++) {
                int bits = archs[a].hi_pass_bits[p];
                bytes += ceilf((float)(N_HI * bits) / 8.0f);
            }

            // rnorm2_hi: 1 fp16 if there are residual passes (n_passes > 1)
            if (archs[a].hi_n_passes > 1) {
                bytes += 2.0f;
            }

            // QJL: rnorm_qjl(2) + signs(4)
            if (archs[a].hi_use_qjl) {
                bytes += 2.0f + 4.0f;
            }

            archs[a].bpv = bytes * 8.0f / (float)DIM;
        }
    }

    // Print architecture table
    printf("%-14s  %5s  %-45s\n", "Architecture", "BPV", "Description");
    for (int i = 0; i < 70; i++) printf("-");
    printf("\n");
    for (int a = 0; a < n_archs; a++) {
        printf("%-14s  %5.3f  %-45s\n", archs[a].name, archs[a].bpv, archs[a].desc);
    }
    printf("\n");

    // Outlier profiles
    struct Profile {
        const char * label;
        float sigma_hi;
        float sigma_lo;
    };

    Profile profiles[] = {
        { "50%  (uniform-ish)",  1.225f, 1.0f },
        { "70%  (moderate)",     2.871f, 1.0f },
        { "90%  (high)",         5.196f, 1.0f },
        { "99%  (extreme)",     17.146f, 1.0f },
        { "99.9% (Qwen L0)",   54.77f,  1.0f },
    };
    int n_profiles = (int)(sizeof(profiles) / sizeof(profiles[0]));

    // Store all results for summary tables
    std::vector<std::vector<Result>> all_results(n_profiles);

    for (int p = 0; p < n_profiles; p++) {
        Profile & prof = profiles[p];

        float var_hi = N_HI * prof.sigma_hi * prof.sigma_hi;
        float var_lo = N_LO * prof.sigma_lo * prof.sigma_lo;
        float var_pct = var_hi / (var_hi + var_lo) * 100.0f;

        printf("================================================================\n");
        printf("  Profile: %s  (actual var%%: %.1f%%)\n", prof.label, var_pct);
        printf("  sigma_hi=%.3f  sigma_lo=%.3f\n", prof.sigma_hi, prof.sigma_lo);
        printf("================================================================\n\n");

        std::mt19937 rng(42 + p);
        std::vector<std::vector<float>> K, Q;
        generate_vectors(K, N_VEC, prof.sigma_hi, prof.sigma_lo, rng);
        generate_vectors(Q, N_VEC, prof.sigma_hi, prof.sigma_lo, rng);

        for (int a = 0; a < n_archs; a++) {
            std::vector<std::vector<float>> K_hat;
            run_architecture(archs[a], K, Q, K_hat);
            Metrics m = compute_metrics(K, K_hat, Q);

            printf("  %-14s (%5.3f bpv): cos=%.6f  relL2=%.4f  dotErr=%.4f%%\n",
                   archs[a].name, archs[a].bpv, m.cosine_sim, m.rel_l2, m.dot_rel_error * 100.0);

            all_results[p].push_back({ archs[a].name, archs[a].bpv, m });
        }
        printf("\n");
    }

    // =====================================================================
    // Summary 1: Cosine similarity table
    // =====================================================================
    printf("================================================================\n");
    printf("  SUMMARY: Cosine Similarity (higher = better)\n");
    printf("================================================================\n\n");

    // Print header
    printf("%-22s", "Profile");
    for (int a = 0; a < n_archs; a++) {
        printf(" %11s", archs[a].name);
    }
    printf("\n");
    printf("%-22s", "");
    for (int a = 0; a < n_archs; a++) {
        printf("  (%5.3f)", archs[a].bpv);
    }
    printf("\n");
    for (int i = 0; i < 22 + n_archs * 12; i++) printf("-");
    printf("\n");

    for (int p = 0; p < n_profiles; p++) {
        printf("%-22s", profiles[p].label);
        for (int a = 0; a < n_archs; a++) {
            printf("  %9.6f", all_results[p][a].metrics.cosine_sim);
        }
        printf("\n");
    }
    printf("\n");

    // =====================================================================
    // Summary 2: Relative L2 error table
    // =====================================================================
    printf("================================================================\n");
    printf("  SUMMARY: Relative L2 Error (lower = better)\n");
    printf("================================================================\n\n");

    printf("%-22s", "Profile");
    for (int a = 0; a < n_archs; a++) {
        printf(" %11s", archs[a].name);
    }
    printf("\n");
    for (int i = 0; i < 22 + n_archs * 12; i++) printf("-");
    printf("\n");

    for (int p = 0; p < n_profiles; p++) {
        printf("%-22s", profiles[p].label);
        for (int a = 0; a < n_archs; a++) {
            printf("  %9.4f", all_results[p][a].metrics.rel_l2);
        }
        printf("\n");
    }
    printf("\n");

    // =====================================================================
    // Analysis 1: Pareto frontier (at 99% outlier concentration)
    // =====================================================================
    printf("================================================================\n");
    printf("  PARETO FRONTIER (99%% outlier concentration)\n");
    printf("  Architectures not dominated on cosine_sim vs bpv\n");
    printf("================================================================\n\n");

    int pareto_profile = 3; // 99%
    std::vector<std::pair<float, float>> points; // (bpv, cosine)
    std::vector<int> indices;
    for (int a = 0; a < n_archs; a++) {
        points.push_back({ archs[a].bpv, (float)all_results[pareto_profile][a].metrics.cosine_sim });
        indices.push_back(a);
    }

    // Sort by bpv ascending
    std::vector<int> order(n_archs);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return points[a].first < points[b].first;
    });

    // Find Pareto-optimal points
    float best_cos = -1.0f;
    std::vector<int> pareto;
    for (int idx : order) {
        if (points[idx].second > best_cos) {
            pareto.push_back(idx);
            best_cos = points[idx].second;
        }
    }

    printf("  %-14s  %5s  %10s  %-45s\n", "Architecture", "BPV", "Cosine", "Description");
    for (int i = 0; i < 80; i++) printf("-");
    printf("\n");
    for (int idx : pareto) {
        printf("  %-14s  %5.3f  %10.6f  %-45s",
               archs[idx].name, archs[idx].bpv,
               all_results[pareto_profile][idx].metrics.cosine_sim,
               archs[idx].desc);
        printf("\n");
    }
    printf("\n");

    // =====================================================================
    // Analysis 2: Target table — min BPV for cosine > 0.9999
    // =====================================================================
    printf("================================================================\n");
    printf("  TARGET: Minimum BPV for cosine > 0.9999 (q8_0-level)\n");
    printf("================================================================\n\n");

    printf("  %-22s  %-16s  %5s  %10s\n", "Profile", "Best arch", "BPV", "Cosine");
    for (int i = 0; i < 60; i++) printf("-");
    printf("\n");

    for (int p = 0; p < n_profiles; p++) {
        float best_bpv = 999.0f;
        int best_a = -1;
        for (int a = 0; a < n_archs; a++) {
            if (all_results[p][a].metrics.cosine_sim >= 0.9999 &&
                archs[a].bpv < best_bpv) {
                best_bpv = archs[a].bpv;
                best_a = a;
            }
        }
        if (best_a >= 0) {
            printf("  %-22s  %-16s  %5.3f  %10.6f\n",
                   profiles[p].label, archs[best_a].name, archs[best_a].bpv,
                   all_results[p][best_a].metrics.cosine_sim);
        } else {
            printf("  %-22s  %-16s  %5s  %10s\n",
                   profiles[p].label, "(none)", "N/A", "N/A");
        }
    }
    printf("\n");

    // =====================================================================
    // Analysis 3: QJL ablation — how much does QJL help?
    // =====================================================================
    printf("================================================================\n");
    printf("  QJL ABLATION: cosine improvement from adding QJL\n");
    printf("  Delta = cos(with_QJL) - cos(without_QJL)\n");
    printf("================================================================\n\n");

    // Pairs: (with_qjl_index, without_qjl_index)
    struct AblationPair {
        const char * name;
        int with_qjl;
        int without_qjl;
    };

    // Find architecture indices by name
    auto find_arch = [&](const char * name) -> int {
        for (int a = 0; a < n_archs; a++) {
            if (strcmp(archs[a].name, name) == 0) return a;
        }
        return -1;
    };

    AblationPair ablation_pairs[] = {
        { "E_4_4",   find_arch("E_4_4"),   find_arch("E_4_4_nq") },
        { "E_4_3",   find_arch("E_4_3"),   find_arch("E_4_3_nq") },
        { "E_4_2",   find_arch("E_4_2"),   find_arch("E_4_2_nq") },
        { "E_3_3",   find_arch("E_3_3"),   find_arch("E_3_3_nq") },
        { "E_5_4",   find_arch("E_5_4"),   find_arch("E_5_4_nq") },
        { "E_5_3",   find_arch("E_5_3"),   find_arch("E_5_3_nq") },
        { "E_3_3_3", find_arch("E_3_3_3"), find_arch("E_3_3_3_nq") },
        { "S_3",     find_arch("S_3_QJL"), find_arch("S_3_nq") },
        { "S_4",     find_arch("S_4_QJL"), find_arch("S_4_nq") },
        { "S_5",     find_arch("S_5_QJL"), find_arch("S_5_nq") },
        { "S_6",     find_arch("S_6_QJL"), find_arch("S_6_nq") },
        { "S_8",     find_arch("S_8_QJL"), find_arch("S_8_nq") },
    };
    int n_ablation = (int)(sizeof(ablation_pairs) / sizeof(ablation_pairs[0]));

    printf("  %-12s  %5s", "Pair", "BPV+");
    for (int p = 0; p < n_profiles; p++) {
        printf("  %17s", profiles[p].label);
    }
    printf("\n");
    for (int i = 0; i < 12 + 7 + n_profiles * 19; i++) printf("-");
    printf("\n");

    for (int ab = 0; ab < n_ablation; ab++) {
        int aw = ablation_pairs[ab].with_qjl;
        int an = ablation_pairs[ab].without_qjl;
        if (aw < 0 || an < 0) continue;

        // QJL adds 6 bytes (rnorm_qjl + signs) = 0.375 bpv
        float bpv_delta = archs[aw].bpv - archs[an].bpv;

        printf("  %-12s  %+.3f", ablation_pairs[ab].name, bpv_delta);
        for (int p = 0; p < n_profiles; p++) {
            double delta = all_results[p][aw].metrics.cosine_sim
                         - all_results[p][an].metrics.cosine_sim;
            printf("  %+17.6f", delta);
        }
        printf("\n");
    }
    printf("\n");

    // =====================================================================
    // Analysis 4: Single-pass vs multi-pass at same BPV
    // =====================================================================
    printf("================================================================\n");
    printf("  SINGLE-PASS vs MULTI-PASS at similar BPV (99%% outlier)\n");
    printf("================================================================\n\n");

    struct Comparison {
        const char * label;
        const char * arch1;
        const char * arch2;
    };

    Comparison comparisons[] = {
        { "~5.0 bpv", "S_5_QJL",  "E_4_4"   },
        { "~5.0 bpv", "S_5_QJL",  "E_5_3"   },
        { "~4.5 bpv", "S_4_QJL",  "E_4_2"   },
        { "~4.5 bpv", "S_4_QJL",  "E_3_3"   },
        { "~5.25bpv", "S_6_QJL",  "E_5_4"   },
        { "~5.25bpv", "S_6_QJL",  "E_3_3_3" },
    };
    int n_cmp = (int)(sizeof(comparisons) / sizeof(comparisons[0]));

    printf("  %-10s  %-14s (%5s) vs %-14s (%5s)  cos_delta\n",
           "BPV range", "Arch A", "bpv", "Arch B", "bpv");
    for (int i = 0; i < 75; i++) printf("-");
    printf("\n");

    for (int c = 0; c < n_cmp; c++) {
        int a1 = find_arch(comparisons[c].arch1);
        int a2 = find_arch(comparisons[c].arch2);
        if (a1 < 0 || a2 < 0) continue;

        double cos1 = all_results[pareto_profile][a1].metrics.cosine_sim;
        double cos2 = all_results[pareto_profile][a2].metrics.cosine_sim;
        printf("  %-10s  %-14s (%5.3f) vs %-14s (%5.3f)  %+.6f\n",
               comparisons[c].label,
               comparisons[c].arch1, archs[a1].bpv,
               comparisons[c].arch2, archs[a2].bpv,
               cos2 - cos1);
    }
    printf("\n");

    printf("Done.\n");
    return 0;
}
