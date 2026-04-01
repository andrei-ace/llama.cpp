// TurboQuant Extreme Layer Analysis — real Qwen2.5 7B variance profiles
// Reads /tmp/qwen25_stats.csv (per-channel variance from layers 0, 1, 27),
// generates synthetic K/Q vectors matching the real distribution, and measures
// dot product accuracy for MSE, MSE+QJL, residual MSE, residual+QJL, pure QJL.
//
// Key question: does QJL's unbiased correction help MORE on extreme layers
// where MSE rounding bias × high magnitude = large systematic error?
//
// Build: cmake --build build -t test-tq-analyze-extreme -j14
// Run:   ./build/bin/test-tq-analyze-extreme
//
// Self-contained. Reads CSV. No ggml dependencies.

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
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <array>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

static constexpr int N_HI      = 32;     // top-32 channels by variance
static constexpr int N_CH      = 128;    // channels per head
static constexpr int N_K_VEC   = 500;    // K vectors per layer
static constexpr int N_Q_VEC   = 50;     // Q vectors per layer
static constexpr int N_SEEDS   = 10;     // different seeds to simulate context positions
static constexpr float SQRT_PI_OVER_2 = 1.2533141373155003f;

// Target layers
static constexpr int TARGET_LAYERS[] = { 0, 1, 27 };
static constexpr int N_TARGET_LAYERS = 3;

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

// ---------------------------------------------------------------------------
// Approximate inverse normal CDF (Beasley-Springer-Moro)
// ---------------------------------------------------------------------------

static float approx_inv_normal(float p) {
    if (p <= 0.0f) return -6.0f;
    if (p >= 1.0f) return  6.0f;
    float t;
    if (p < 0.5f) {
        t = sqrtf(-2.0f * logf(p));
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

static float c64_d32[64];   // 6-bit
static float c256_d32[256]; // 8-bit

static void init_generated_centroids() {
    generate_centroids_d32(c64_d32, 64);
    generate_centroids_d32(c256_d32, 256);
}

// ---------------------------------------------------------------------------
// Centroid lookup
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
// FWHT (Fast Walsh-Hadamard Transform) — in-place, normalized
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
// Nearest centroid
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
// Hi path: single MSE pass — normalize, FWHT, quantize, inverse FWHT, denorm
// Returns reconstruction and norm.
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// QJL correction on residual
// ---------------------------------------------------------------------------

static void qjl_correct(const float * residual, float * out) {
    float r[N_HI];
    memcpy(r, residual, N_HI * sizeof(float));

    float rnorm = 0.0f;
    for (int j = 0; j < N_HI; j++) rnorm += r[j] * r[j];
    rnorm = sqrtf(rnorm);
    if (rnorm < 1e-30f) return;

    // FWHT the residual to get projected coefficients
    fwht(r, N_HI);

    // Store signs (32 bits = 4 bytes)
    uint8_t signs[4] = {0, 0, 0, 0};
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
// Multi-pass residual MSE + optional QJL on hi channels only
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Pure QJL: no MSE at all, just norm + 32 sign bits on FWHT-projected input
// ---------------------------------------------------------------------------

static void pure_qjl(const float * in, float * out) {
    float hi[N_HI];
    memcpy(hi, in, N_HI * sizeof(float));

    float norm = 0.0f;
    for (int j = 0; j < N_HI; j++) norm += hi[j] * hi[j];
    norm = sqrtf(norm);
    if (norm < 1e-30f) { memset(out, 0, N_HI * sizeof(float)); return; }

    // FWHT the input
    fwht(hi, N_HI);

    // Store signs
    uint8_t signs[4] = {0, 0, 0, 0};
    for (int j = 0; j < N_HI; j++) {
        if (hi[j] >= 0.0f) signs[j / 8] |= (1 << (j % 8));
    }

    // Reconstruct
    float corr[N_HI];
    for (int j = 0; j < N_HI; j++) {
        corr[j] = ((signs[j / 8] >> (j % 8)) & 1) ? 1.0f : -1.0f;
    }
    fwht(corr, N_HI);

    float scale = SQRT_PI_OVER_2 / (float)N_HI * norm;
    for (int j = 0; j < N_HI; j++) out[j] = scale * corr[j];
}

// ---------------------------------------------------------------------------
// CSV parsing: extract per-channel variance for target layers
// Returns map: layer -> vector<float>(128 variances, averaged across heads)
// ---------------------------------------------------------------------------

struct LayerProfile {
    float variance[N_CH];  // per-channel variance (averaged across heads)
    int   hi_idx[N_HI];   // indices of top-32 channels by variance
};

static bool load_profiles(const char * csv_path, std::map<int, LayerProfile> & profiles) {
    std::ifstream f(csv_path);
    if (!f.is_open()) {
        fprintf(stderr, "ERROR: cannot open %s\n", csv_path);
        return false;
    }

    // Accumulate variance per (layer, channel), count heads
    struct Acc {
        double sum;
        int count;
    };
    std::map<int, std::array<Acc, N_CH>> accum;

    // Initialize for target layers
    for (int i = 0; i < N_TARGET_LAYERS; i++) {
        for (int ch = 0; ch < N_CH; ch++) {
            accum[TARGET_LAYERS[i]][ch] = {0.0, 0};
        }
    }

    std::string line;
    std::getline(f, line); // skip header

    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string token;

        // layer,head,channel,mean_abs,variance,...
        int layer, head, channel;
        float mean_abs, variance;

        std::getline(ss, token, ','); layer = std::stoi(token);
        std::getline(ss, token, ','); head = std::stoi(token);
        (void)head;
        std::getline(ss, token, ','); channel = std::stoi(token);
        std::getline(ss, token, ','); mean_abs = std::stof(token);
        (void)mean_abs;
        std::getline(ss, token, ','); variance = std::stof(token);

        auto it = accum.find(layer);
        if (it != accum.end() && channel >= 0 && channel < N_CH) {
            it->second[channel].sum += variance;
            it->second[channel].count++;
        }
    }

    // Build profiles
    for (int i = 0; i < N_TARGET_LAYERS; i++) {
        int layer = TARGET_LAYERS[i];
        LayerProfile & lp = profiles[layer];

        for (int ch = 0; ch < N_CH; ch++) {
            if (accum[layer][ch].count > 0) {
                lp.variance[ch] = (float)(accum[layer][ch].sum / accum[layer][ch].count);
            } else {
                lp.variance[ch] = 1.0f;
            }
        }

        // Find top-32 channels by variance
        std::vector<std::pair<float, int>> sorted_ch(N_CH);
        for (int ch = 0; ch < N_CH; ch++) {
            sorted_ch[ch] = { lp.variance[ch], ch };
        }
        std::sort(sorted_ch.begin(), sorted_ch.end(),
                  [](const auto & a, const auto & b) { return a.first > b.first; });

        for (int k = 0; k < N_HI; k++) {
            lp.hi_idx[k] = sorted_ch[k].second;
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// Generate vectors using real per-channel variance profile (hi channels only)
// Each channel sampled from N(0, sqrt(variance[hi_idx[ch]]))
// Uses different seeds per batch to simulate context position variation.
// ---------------------------------------------------------------------------

static void generate_real_profile_vectors(
    std::vector<std::vector<float>> & vecs,
    int count,
    const LayerProfile & lp,
    uint32_t base_seed
) {
    vecs.resize(count);
    int per_seed = (count + N_SEEDS - 1) / N_SEEDS;

    for (int s = 0; s < N_SEEDS; s++) {
        std::mt19937 rng(base_seed + (uint32_t)s * 7919u);
        int start = s * per_seed;
        int end = std::min(start + per_seed, count);
        for (int i = start; i < end; i++) {
            vecs[i].resize(N_HI);
            for (int j = 0; j < N_HI; j++) {
                float sigma = sqrtf(lp.variance[lp.hi_idx[j]]);
                std::normal_distribution<float> dist(0.0f, sigma);
                vecs[i][j] = dist(rng);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Quantization scheme descriptor
// ---------------------------------------------------------------------------

enum SchemeType {
    SCHEME_MSE_ONLY,      // single-pass MSE
    SCHEME_MSE_QJL,       // single-pass MSE + QJL
    SCHEME_RESIDUAL,      // multi-pass residual MSE (no QJL)
    SCHEME_RESIDUAL_QJL,  // multi-pass residual MSE + QJL
    SCHEME_PURE_QJL,      // no MSE, just norm + signs
};

struct Scheme {
    const char * name;
    SchemeType   type;
    int          pass_bits[4];
    int          n_passes;
    float        bpv_hi;  // bits per hi channel (computed below)
};

static float compute_bpv_hi(const Scheme & s) {
    // norm_hi: 16 bits (fp16)
    // per MSE pass: bits * 32 channels packed + 16-bit norm for residual passes
    // QJL: 16-bit rnorm + 32 sign bits
    // Total bits / 32 channels = bpv

    float total_bits = 16.0f; // norm_hi always present

    for (int p = 0; p < s.n_passes; p++) {
        total_bits += (float)(s.pass_bits[p] * N_HI);
        if (p > 0) total_bits += 16.0f; // residual norm
    }

    bool has_qjl = (s.type == SCHEME_MSE_QJL ||
                    s.type == SCHEME_RESIDUAL_QJL ||
                    s.type == SCHEME_PURE_QJL);
    if (has_qjl) {
        total_bits += 16.0f + 32.0f; // rnorm + sign bits
    }

    return total_bits / (float)N_HI;
}

// ---------------------------------------------------------------------------
// Dot product metrics (hi channels only)
// ---------------------------------------------------------------------------

struct DotMetrics {
    double dot_rel_err;    // mean |exact - approx| / |exact|
    double dot_bias;       // mean (approx - exact) / |exact| — SIGNED
    double bias_sq;        // (mean signed error)^2
    double variance;       // var(signed errors) = E[err^2] - E[err]^2
};

static DotMetrics compute_dot_metrics(
    const std::vector<std::vector<float>> & K,
    const std::vector<std::vector<float>> & K_hat,
    const std::vector<std::vector<float>> & Q
) {
    int nk = (int)K.size();
    int nq = (int)Q.size();

    double rel_err_sum  = 0.0;
    double signed_sum   = 0.0;
    double signed_sq    = 0.0;
    int count = 0;

    // Use all Q x K pairs (50 * 500 = 25000 — manageable)
    for (int qi = 0; qi < nq; qi++) {
        for (int ki = 0; ki < nk; ki++) {
            double dot_true  = 0.0;
            double dot_quant = 0.0;
            for (int d = 0; d < N_HI; d++) {
                dot_true  += (double)Q[qi][d] * (double)K[ki][d];
                dot_quant += (double)Q[qi][d] * (double)K_hat[ki][d];
            }
            double adot = fabs(dot_true);
            if (adot < 1e-10) continue;

            double rel = (dot_quant - dot_true) / adot;
            rel_err_sum += fabs(rel);
            signed_sum  += rel;
            signed_sq   += rel * rel;
            count++;
        }
    }

    DotMetrics m = {};
    if (count > 0) {
        m.dot_rel_err = rel_err_sum / count;
        m.dot_bias    = signed_sum / count;
        m.bias_sq     = m.dot_bias * m.dot_bias;
        double mean_sq = signed_sq / count;
        m.variance    = mean_sq - m.bias_sq;
        if (m.variance < 0.0) m.variance = 0.0; // numerical guard
    }
    return m;
}

// ---------------------------------------------------------------------------
// Apply quantization scheme to K vectors (hi channels only)
// ---------------------------------------------------------------------------

static void apply_scheme(
    const Scheme & s,
    const std::vector<std::vector<float>> & K,
    std::vector<std::vector<float>> & K_hat
) {
    int nk = (int)K.size();
    K_hat.resize(nk);

    for (int i = 0; i < nk; i++) {
        K_hat[i].resize(N_HI);

        if (s.type == SCHEME_PURE_QJL) {
            pure_qjl(K[i].data(), K_hat[i].data());
        } else {
            bool use_qjl = (s.type == SCHEME_MSE_QJL || s.type == SCHEME_RESIDUAL_QJL);
            quantize_hi_multipass(K[i].data(), K_hat[i].data(),
                                 s.pass_bits, s.n_passes, use_qjl);
        }
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    init_generated_centroids();

    printf("================================================================\n");
    printf("  TurboQuant Extreme Layer Analysis\n");
    printf("  Real Qwen2.5 7B variance profiles (layers 0, 1, 27)\n");
    printf("  Hi channels only (top 32 by variance), FWHT-32 rotation\n");
    printf("================================================================\n\n");

    // -----------------------------------------------------------------------
    // Load real variance profiles
    // -----------------------------------------------------------------------
    std::map<int, LayerProfile> profiles;
    if (!load_profiles("/tmp/qwen25_stats.csv", profiles)) {
        fprintf(stderr, "Failed to load CSV. Aborting.\n");
        return 1;
    }

    printf("Loaded variance profiles for %d layers\n\n", (int)profiles.size());

    // Print variance summary for each layer
    for (int i = 0; i < N_TARGET_LAYERS; i++) {
        int layer = TARGET_LAYERS[i];
        const LayerProfile & lp = profiles[layer];

        float var_min = 1e30f, var_max = -1e30f, var_sum = 0.0f;
        float hi_var_sum = 0.0f;
        for (int ch = 0; ch < N_CH; ch++) {
            var_min = std::min(var_min, lp.variance[ch]);
            var_max = std::max(var_max, lp.variance[ch]);
            var_sum += lp.variance[ch];
        }
        for (int k = 0; k < N_HI; k++) {
            hi_var_sum += lp.variance[lp.hi_idx[k]];
        }

        printf("Layer %2d: var range [%.2f, %.2f], mean=%.2f, hi32 sum=%.1f (%.1f%% of total)\n",
               layer, var_min, var_max, var_sum / N_CH,
               hi_var_sum, hi_var_sum / var_sum * 100.0f);
        printf("  Top-5 hi channels: ");
        for (int k = 0; k < 5; k++) {
            printf("ch%d(var=%.1f) ", lp.hi_idx[k], lp.variance[lp.hi_idx[k]]);
        }
        printf("\n");
    }
    printf("\n");

    // -----------------------------------------------------------------------
    // Define quantization schemes
    // -----------------------------------------------------------------------
    Scheme schemes[] = {
        // MSE only (single-pass, no QJL)
        { "MSE-3",        SCHEME_MSE_ONLY,     {3},    1, 0.0f },
        { "MSE-4",        SCHEME_MSE_ONLY,     {4},    1, 0.0f },
        { "MSE-5",        SCHEME_MSE_ONLY,     {5},    1, 0.0f },
        { "MSE-6",        SCHEME_MSE_ONLY,     {6},    1, 0.0f },
        { "MSE-8",        SCHEME_MSE_ONLY,     {8},    1, 0.0f },

        // MSE + QJL (single-pass)
        { "MSE3+QJL",     SCHEME_MSE_QJL,      {3},    1, 0.0f },
        { "MSE4+QJL",     SCHEME_MSE_QJL,      {4},    1, 0.0f },
        { "MSE5+QJL",     SCHEME_MSE_QJL,      {5},    1, 0.0f },

        // Residual MSE (no QJL)
        { "R4+3",         SCHEME_RESIDUAL,      {4,3},  2, 0.0f },
        { "R4+4",         SCHEME_RESIDUAL,      {4,4},  2, 0.0f },
        { "R5+3",         SCHEME_RESIDUAL,      {5,3},  2, 0.0f },

        // Residual MSE + QJL
        { "R4+3+QJL",     SCHEME_RESIDUAL_QJL,  {4,3},  2, 0.0f },
        { "R4+4+QJL",     SCHEME_RESIDUAL_QJL,  {4,4},  2, 0.0f },
        { "R5+3+QJL",     SCHEME_RESIDUAL_QJL,  {5,3},  2, 0.0f },

        // Pure QJL (no MSE)
        { "PureQJL",      SCHEME_PURE_QJL,      {},     0, 0.0f },
    };
    int n_schemes = (int)(sizeof(schemes) / sizeof(schemes[0]));

    // Compute bpv_hi for each scheme
    for (int s = 0; s < n_schemes; s++) {
        schemes[s].bpv_hi = compute_bpv_hi(schemes[s]);
    }

    // Print scheme table
    printf("%-14s  %6s  %s\n", "Scheme", "bpv_hi", "Type");
    for (int i = 0; i < 50; i++) printf("-");
    printf("\n");
    for (int s = 0; s < n_schemes; s++) {
        const char * type_str = "";
        switch (schemes[s].type) {
            case SCHEME_MSE_ONLY:     type_str = "MSE only"; break;
            case SCHEME_MSE_QJL:      type_str = "MSE + QJL"; break;
            case SCHEME_RESIDUAL:     type_str = "Residual MSE"; break;
            case SCHEME_RESIDUAL_QJL: type_str = "Residual + QJL"; break;
            case SCHEME_PURE_QJL:     type_str = "Pure QJL"; break;
        }
        printf("%-14s  %6.2f  %s\n", schemes[s].name, schemes[s].bpv_hi, type_str);
    }
    printf("\n");

    // -----------------------------------------------------------------------
    // Per-layer analysis
    // -----------------------------------------------------------------------

    // Store results: [layer_idx][scheme_idx]
    DotMetrics all_results[N_TARGET_LAYERS][16]; // max 16 schemes
    assert(n_schemes <= 16);

    for (int li = 0; li < N_TARGET_LAYERS; li++) {
        int layer = TARGET_LAYERS[li];
        const LayerProfile & lp = profiles[layer];

        printf("================================================================\n");
        printf("  Layer %d — %d K vectors, %d Q vectors, %d seeds\n",
               layer, N_K_VEC, N_Q_VEC, N_SEEDS);
        printf("================================================================\n\n");

        // Generate K and Q vectors from real variance profile
        std::vector<std::vector<float>> K, Q;
        generate_real_profile_vectors(K, N_K_VEC, lp, 42u + (uint32_t)layer * 1000u);
        generate_real_profile_vectors(Q, N_Q_VEC, lp, 99u + (uint32_t)layer * 1000u);

        // Print header
        printf("  %-14s  %6s  %10s  %10s  %12s  %12s\n",
               "Scheme", "bpv_hi", "dot_rel_err", "dot_bias", "bias^2", "variance");
        for (int i = 0; i < 80; i++) printf("-");
        printf("\n");

        for (int s = 0; s < n_schemes; s++) {
            std::vector<std::vector<float>> K_hat;
            apply_scheme(schemes[s], K, K_hat);
            DotMetrics m = compute_dot_metrics(K, K_hat, Q);
            all_results[li][s] = m;

            // Highlight bias: show arrow indicating direction
            const char * bias_arrow = (m.dot_bias < -0.001) ? " <<< UNDER" :
                                      (m.dot_bias >  0.001) ? " >>> OVER"  : "";

            printf("  %-14s  %6.2f  %10.4f%%  %+10.4f%%  %12.6f  %12.6f%s\n",
                   schemes[s].name, schemes[s].bpv_hi,
                   m.dot_rel_err * 100.0,
                   m.dot_bias * 100.0,
                   m.bias_sq,
                   m.variance,
                   bias_arrow);
        }
        printf("\n");
    }

    // -----------------------------------------------------------------------
    // Cross-layer summary: dot_rel_err
    // -----------------------------------------------------------------------

    printf("================================================================\n");
    printf("  SUMMARY: Dot Product Relative Error (%%) — lower is better\n");
    printf("================================================================\n\n");

    printf("  %-14s  %6s", "Scheme", "bpv_hi");
    for (int li = 0; li < N_TARGET_LAYERS; li++) {
        char buf[32];
        snprintf(buf, sizeof(buf), "L%d", TARGET_LAYERS[li]);
        printf("  %10s", buf);
    }
    printf("\n");
    for (int i = 0; i < 14 + 8 + N_TARGET_LAYERS * 12; i++) printf("-");
    printf("\n");

    for (int s = 0; s < n_schemes; s++) {
        printf("  %-14s  %6.2f", schemes[s].name, schemes[s].bpv_hi);
        for (int li = 0; li < N_TARGET_LAYERS; li++) {
            printf("  %9.4f%%", all_results[li][s].dot_rel_err * 100.0);
        }
        printf("\n");
    }
    printf("\n");

    // -----------------------------------------------------------------------
    // Cross-layer summary: dot_bias (SIGNED — the key column)
    // -----------------------------------------------------------------------

    printf("================================================================\n");
    printf("  SUMMARY: Dot Product Bias (%%) — SIGNED, closer to 0 is better\n");
    printf("  Negative = systematic underestimation (MSE attenuation)\n");
    printf("  QJL removes this bias by construction\n");
    printf("================================================================\n\n");

    printf("  %-14s  %6s", "Scheme", "bpv_hi");
    for (int li = 0; li < N_TARGET_LAYERS; li++) {
        char buf[32];
        snprintf(buf, sizeof(buf), "L%d", TARGET_LAYERS[li]);
        printf("  %10s", buf);
    }
    printf("\n");
    for (int i = 0; i < 14 + 8 + N_TARGET_LAYERS * 12; i++) printf("-");
    printf("\n");

    for (int s = 0; s < n_schemes; s++) {
        printf("  %-14s  %6.2f", schemes[s].name, schemes[s].bpv_hi);
        for (int li = 0; li < N_TARGET_LAYERS; li++) {
            printf("  %+9.4f%%", all_results[li][s].dot_bias * 100.0);
        }

        // Mark if all layers show negative bias (MSE attenuation)
        bool all_neg = true;
        for (int li = 0; li < N_TARGET_LAYERS; li++) {
            if (all_results[li][s].dot_bias >= -0.0001) all_neg = false;
        }
        if (all_neg && schemes[s].type == SCHEME_MSE_ONLY) {
            printf("  <-- MSE attenuation");
        }
        if (all_neg && schemes[s].type == SCHEME_RESIDUAL) {
            printf("  <-- residual still biased");
        }
        printf("\n");
    }
    printf("\n");

    // -----------------------------------------------------------------------
    // Cross-layer summary: bias^2 vs variance decomposition
    // -----------------------------------------------------------------------

    printf("================================================================\n");
    printf("  SUMMARY: Error Decomposition — bias^2 + variance\n");
    printf("  High bias^2 = systematic error. High variance = noise.\n");
    printf("  QJL trades bias for variance (unbiased but noisier).\n");
    printf("================================================================\n\n");

    printf("  %-14s  %6s", "Scheme", "bpv_hi");
    for (int li = 0; li < N_TARGET_LAYERS; li++) {
        char buf[32];
        snprintf(buf, sizeof(buf), "L%d bias^2", TARGET_LAYERS[li]);
        printf("  %12s", buf);
        snprintf(buf, sizeof(buf), "L%d var", TARGET_LAYERS[li]);
        printf("  %10s", buf);
    }
    printf("\n");
    for (int i = 0; i < 14 + 8 + N_TARGET_LAYERS * 24; i++) printf("-");
    printf("\n");

    for (int s = 0; s < n_schemes; s++) {
        printf("  %-14s  %6.2f", schemes[s].name, schemes[s].bpv_hi);
        for (int li = 0; li < N_TARGET_LAYERS; li++) {
            printf("  %12.8f  %10.8f",
                   all_results[li][s].bias_sq,
                   all_results[li][s].variance);
        }
        printf("\n");
    }
    printf("\n");

    // -----------------------------------------------------------------------
    // Key findings: compare MSE vs QJL bias improvement per layer
    // -----------------------------------------------------------------------

    printf("================================================================\n");
    printf("  KEY FINDINGS: QJL bias reduction vs MSE-only\n");
    printf("  Shows |bias_MSE| - |bias_QJL| for matched bit widths\n");
    printf("================================================================\n\n");

    // Pairs to compare: (MSE-only index, MSE+QJL index, label)
    struct Comparison {
        int mse_idx;
        int qjl_idx;
        const char * label;
    };

    // Find indices
    auto find_scheme = [&](const char * name) -> int {
        for (int s = 0; s < n_schemes; s++) {
            if (strcmp(schemes[s].name, name) == 0) return s;
        }
        return -1;
    };

    Comparison comparisons[] = {
        { find_scheme("MSE-3"),  find_scheme("MSE3+QJL"),  "3-bit: MSE vs MSE+QJL" },
        { find_scheme("MSE-4"),  find_scheme("MSE4+QJL"),  "4-bit: MSE vs MSE+QJL" },
        { find_scheme("MSE-5"),  find_scheme("MSE5+QJL"),  "5-bit: MSE vs MSE+QJL" },
        { find_scheme("R4+3"),   find_scheme("R4+3+QJL"),  "R4+3:  residual vs +QJL" },
        { find_scheme("R4+4"),   find_scheme("R4+4+QJL"),  "R4+4:  residual vs +QJL" },
        { find_scheme("R5+3"),   find_scheme("R5+3+QJL"),  "R5+3:  residual vs +QJL" },
    };
    int n_comp = (int)(sizeof(comparisons) / sizeof(comparisons[0]));

    printf("  %-30s", "Comparison");
    for (int li = 0; li < N_TARGET_LAYERS; li++) {
        char buf[32];
        snprintf(buf, sizeof(buf), "L%d |bias| reduction", TARGET_LAYERS[li]);
        printf("  %22s", buf);
    }
    printf("\n");
    for (int i = 0; i < 30 + 2 + N_TARGET_LAYERS * 24; i++) printf("-");
    printf("\n");

    for (int c = 0; c < n_comp; c++) {
        int mi = comparisons[c].mse_idx;
        int qi = comparisons[c].qjl_idx;
        if (mi < 0 || qi < 0) continue;

        printf("  %-30s", comparisons[c].label);
        for (int li = 0; li < N_TARGET_LAYERS; li++) {
            double mse_bias = fabs(all_results[li][mi].dot_bias);
            double qjl_bias = fabs(all_results[li][qi].dot_bias);
            double reduction = mse_bias - qjl_bias;
            double pct = (mse_bias > 1e-10) ? reduction / mse_bias * 100.0 : 0.0;
            printf("  %+8.4f%% (%5.1f%% less)", reduction * 100.0, pct);
        }
        printf("\n");
    }
    printf("\n");

    // -----------------------------------------------------------------------
    // Hypothesis test: does extreme variance amplify MSE bias?
    // -----------------------------------------------------------------------

    printf("================================================================\n");
    printf("  HYPOTHESIS: extreme channels amplify MSE bias\n");
    printf("  Compare Layer 0 (extreme outliers) vs Layer 27 (mild)\n");
    printf("================================================================\n\n");

    int l0_idx = -1, l27_idx = -1;
    for (int li = 0; li < N_TARGET_LAYERS; li++) {
        if (TARGET_LAYERS[li] == 0)  l0_idx = li;
        if (TARGET_LAYERS[li] == 27) l27_idx = li;
    }

    if (l0_idx >= 0 && l27_idx >= 0) {
        printf("  %-14s  %6s  %14s  %14s  %10s\n",
               "Scheme", "bpv_hi", "L0 |bias|", "L27 |bias|", "L0/L27 ratio");
        for (int i = 0; i < 70; i++) printf("-");
        printf("\n");

        for (int s = 0; s < n_schemes; s++) {
            double b0  = fabs(all_results[l0_idx][s].dot_bias);
            double b27 = fabs(all_results[l27_idx][s].dot_bias);
            double ratio = (b27 > 1e-10) ? b0 / b27 : 0.0;

            printf("  %-14s  %6.2f  %13.4f%%  %13.4f%%  %10.2fx\n",
                   schemes[s].name, schemes[s].bpv_hi,
                   b0 * 100.0, b27 * 100.0, ratio);
        }
        printf("\n");

        printf("  Interpretation:\n");
        printf("  - If L0/L27 ratio >> 1 for MSE schemes: extreme variance amplifies bias\n");
        printf("  - If L0/L27 ratio ~ 1 for QJL schemes:  QJL corrects regardless of magnitude\n");
        printf("  - Pure QJL is the extreme test: no MSE at all, just norm + 32 sign bits\n\n");
    }

    printf("Done.\n");
    return 0;
}
