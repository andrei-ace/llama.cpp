// TurboQuant Inner Product Analysis — bias/variance decomposition
//
// Key insight from the TurboQuant paper:
//   Stage 1 (MSE): minimizes reconstruction error but introduces BIAS in dot products
//   Stage 2 (QJL): corrects the bias — combined estimator is PROVABLY UNBIASED
//   Within 2.7x of Shannon lower bound
//
// The correct metric for KV cache quantization is inner product estimation quality.
// This test decomposes error into bias^2 + variance to determine whether QJL is
// worth the extra bits. If bias_pct (= bias^2/MSE) is high for MSE-only schemes,
// the paper's claim holds and QJL is essential.
//
// Reads per-channel variance CSV files (Qwen2.5 7B, Qwen3 8B, Qwen2.5 1.5B, Mistral 7B).
// Auto-classifies layers into extreme/moderate/uniform by outlier concentration.
// Picks 3 representative layers per tier. Generates 500 K, 50 Q vectors per layer.
//
// Self-contained: no ggml deps, reads CSV, outputs bias-variance tables.
//
// Build: cmake --build build -t test-tq-inner-product-analysis -j14
// Run:   ./build/bin/test-tq-inner-product-analysis [--csv /tmp/qwen25_stats.csv]

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
#include <string>
#include <map>
#include <fstream>
#include <sstream>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

static constexpr int N_HI    = 32;
static constexpr int N_LO    = 96;
static constexpr int DIM     = N_HI + N_LO;   // 128
static constexpr int N_K_VEC = 500;            // K vectors per layer
static constexpr int N_Q_VEC = 50;             // Q vectors per layer
static constexpr float SQRT_PI_OVER_2 = 1.2533141373155003f;

// ---------------------------------------------------------------------------
// Q generation methods — the key addition
// ---------------------------------------------------------------------------
//
// Random Q: independent of K, averages away correlation between K magnitude
//           and quantization bias. Shows bias_pct ~ 0%.
// Correlated Q: Q = K + noise, mimicking real attention where Q specifically
//               looks for similar K vectors. MSE attenuation of high-magnitude
//               K directly reduces attention to the CORRECT key.
// Attention-weighted: random Q but weight errors by softmax attention probs.
//                     Errors on high-attention keys matter more.

enum QMethod {
    QMETHOD_RANDOM     = 0,   // current: independent Q from per-channel variance
    QMETHOD_CORRELATED = 1,   // Q = K[target] + noise (high attention to target)
    QMETHOD_ATTN_WEIGHTED = 2, // random Q, weight errors by softmax(Q*K)
    QMETHOD_COUNT      = 3,
};

static const char * qmethod_names[QMETHOD_COUNT] = {
    "Random_Q",
    "Correlated_Q",
    "Attn_Weighted",
};

// ---------------------------------------------------------------------------
// Lloyd-Max centroids: d=32 (for hi path)
// ---------------------------------------------------------------------------

static const float c4_d32[4] = {
    -0.2633f, -0.0798f, 0.0798f, 0.2633f
};

static const float c8_d32[8] = {
    -0.3663f, -0.2325f, -0.1318f, -0.0429f, 0.0429f, 0.1318f, 0.2325f, 0.3663f
};

static const float c16_d32[16] = {
    -0.4534f, -0.3499f, -0.2765f, -0.2161f, -0.1629f, -0.1138f, -0.0674f, -0.0223f,
     0.0223f,  0.0674f,  0.1138f,  0.1629f,  0.2161f,  0.2765f,  0.3499f,  0.4534f
};

static const float c32_d32[32] = {
    -0.5265f, -0.4434f, -0.3864f, -0.3409f, -0.3020f, -0.2676f, -0.2361f, -0.2070f,
    -0.1795f, -0.1534f, -0.1283f, -0.1040f, -0.0803f, -0.0571f, -0.0341f, -0.0114f,
     0.0114f,  0.0341f,  0.0571f,  0.0803f,  0.1040f,  0.1283f,  0.1534f,  0.1795f,
     0.2070f,  0.2361f,  0.2676f,  0.3020f,  0.3409f,  0.3864f,  0.4434f,  0.5265f
};

// ---------------------------------------------------------------------------
// Lloyd-Max centroids: d=96 (for lo path)
// ---------------------------------------------------------------------------

static const float c8_d96[8] = {
    -0.2169f, -0.1362f, -0.0768f, -0.0249f, 0.0249f, 0.0768f, 0.1362f, 0.2169f
};

static const float c16_d96[16] = {
    -0.2909f, -0.2244f, -0.1774f, -0.1388f, -0.1047f, -0.0731f, -0.0433f, -0.0144f,
     0.0144f,  0.0433f,  0.0731f,  0.1047f,  0.1388f,  0.1774f,  0.2244f,  0.2909f
};

// Generated centroid tables for higher bit widths
static float c32_d96[32];
static float c64_d96[64];

// d=128 centroids (for full-dim schemes like tqk4_0)
static const float c16_d128[16] = {
    -0.1928f, -0.1515f, -0.1210f, -0.0951f, -0.0720f, -0.0505f, -0.0299f, -0.0100f,
     0.0100f,  0.0299f,  0.0505f,  0.0720f,  0.0951f,  0.1210f,  0.1515f,  0.1928f
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

static void generate_centroids(float * out, int n_centroids, int dim) {
    float sigma = 1.0f / sqrtf((float)dim);
    for (int i = 0; i < n_centroids; i++) {
        float p = ((float)i + 0.5f) / (float)n_centroids;
        out[i] = sigma * approx_inv_normal(p);
    }
}

static void init_generated_centroids() {
    generate_centroids(c32_d96,  32,  96);
    generate_centroids(c64_d96,  64,  96);
}

// ---------------------------------------------------------------------------
// Centroid lookup helpers
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
        default: assert(false && "unsupported bit width for d=32"); return { nullptr, 0 };
    }
}

static CentroidSet get_centroids_d96(int bits) {
    switch (bits) {
        case 3: return { c8_d96,   8 };
        case 4: return { c16_d96, 16 };
        case 5: return { c32_d96, 32 };
        case 6: return { c64_d96, 64 };
        default: assert(false && "unsupported bit width for d=96"); return { nullptr, 0 };
    }
}

// ---------------------------------------------------------------------------
// CSV data structures
// ---------------------------------------------------------------------------

struct ChannelStats {
    int   layer;
    int   head;
    int   channel;
    float mean_abs;
    float variance;
    float std_dev;
    float importance;
    int   rank;
    int   is_outlier;
};

struct HeadProfile {
    int   layer;
    int   head;
    float variance[DIM];
    int   is_outlier[DIM];
    int   hi_channels[N_HI];
    int   lo_channels[N_LO];
    float outlier_pct;
};

static bool load_csv(const char * path, std::vector<ChannelStats> & out) {
    std::ifstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "ERROR: cannot open %s\n", path);
        return false;
    }
    std::string line;
    std::getline(f, line); // skip header
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        ChannelStats cs;
        std::istringstream ss(line);
        std::string tok;
        std::getline(ss, tok, ','); cs.layer      = std::stoi(tok);
        std::getline(ss, tok, ','); cs.head       = std::stoi(tok);
        std::getline(ss, tok, ','); cs.channel    = std::stoi(tok);
        std::getline(ss, tok, ','); cs.mean_abs   = std::stof(tok);
        std::getline(ss, tok, ','); cs.variance   = std::stof(tok);
        std::getline(ss, tok, ','); cs.std_dev    = std::stof(tok);
        std::getline(ss, tok, ','); cs.importance = std::stof(tok);
        std::getline(ss, tok, ','); cs.rank       = std::stoi(tok);
        std::getline(ss, tok, ','); cs.is_outlier = std::stoi(tok);
        out.push_back(cs);
    }
    return true;
}

static bool build_head_profile(
    const std::vector<ChannelStats> & all,
    int layer, int head,
    HeadProfile & hp
) {
    hp.layer = layer;
    hp.head  = head;
    memset(hp.variance, 0, sizeof(hp.variance));
    memset(hp.is_outlier, 0, sizeof(hp.is_outlier));

    int found = 0;
    for (const auto & cs : all) {
        if (cs.layer == layer && cs.head == head && cs.channel < DIM) {
            hp.variance[cs.channel]    = cs.variance;
            hp.is_outlier[cs.channel]  = cs.is_outlier;
            found++;
        }
    }
    if (found != DIM) {
        if (found == 0) return false;
    }

    // Split into hi/lo using the CSV is_outlier column
    int n_hi = 0, n_lo = 0;
    for (int c = 0; c < DIM; c++) {
        if (hp.is_outlier[c] && n_hi < N_HI) {
            hp.hi_channels[n_hi++] = c;
        } else {
            if (n_lo < N_LO) hp.lo_channels[n_lo++] = c;
        }
    }

    // If we don't have exactly 32 outliers, fill hi from highest-variance non-outliers
    if (n_hi < N_HI) {
        std::vector<std::pair<float,int>> remaining;
        for (int c = 0; c < DIM; c++) {
            bool already_hi = false;
            for (int i = 0; i < n_hi; i++) {
                if (hp.hi_channels[i] == c) { already_hi = true; break; }
            }
            if (!already_hi) {
                remaining.push_back({hp.variance[c], c});
            }
        }
        std::sort(remaining.begin(), remaining.end(),
                  [](const auto & a, const auto & b) { return a.first > b.first; });
        for (int i = 0; n_hi < N_HI && i < (int)remaining.size(); i++) {
            hp.hi_channels[n_hi++] = remaining[i].second;
        }
    }
    // If more than 32 outliers, keep top-32 by variance
    if (n_hi > N_HI) {
        std::vector<std::pair<float,int>> candidates;
        for (int i = 0; i < n_hi; i++) {
            candidates.push_back({hp.variance[hp.hi_channels[i]], hp.hi_channels[i]});
        }
        std::sort(candidates.begin(), candidates.end(),
                  [](const auto & a, const auto & b) { return a.first > b.first; });
        n_hi = N_HI;
        for (int i = 0; i < N_HI; i++) {
            hp.hi_channels[i] = candidates[i].second;
        }
    }

    // Rebuild lo as everything not in hi
    n_lo = 0;
    for (int c = 0; c < DIM; c++) {
        bool in_hi = false;
        for (int i = 0; i < N_HI; i++) {
            if (hp.hi_channels[i] == c) { in_hi = true; break; }
        }
        if (!in_hi && n_lo < N_LO) {
            hp.lo_channels[n_lo++] = c;
        }
    }

    // Compute outlier percentage
    float var_hi = 0.0f, var_lo = 0.0f;
    for (int i = 0; i < N_HI; i++) var_hi += hp.variance[hp.hi_channels[i]];
    for (int i = 0; i < N_LO; i++) var_lo += hp.variance[hp.lo_channels[i]];
    hp.outlier_pct = var_hi / (var_hi + var_lo + 1e-30f);

    return true;
}

// ---------------------------------------------------------------------------
// Auto-classify layers into extreme/moderate/uniform by outlier concentration
// ---------------------------------------------------------------------------

struct LayerInfo {
    int   layer;
    float avg_outlier_pct;   // average outlier_pct across heads
    int   n_heads;
};

static void auto_classify_layers(
    const std::vector<ChannelStats> & all_stats,
    std::vector<int> & extreme_layers,
    std::vector<int> & moderate_layers,
    std::vector<int> & uniform_layers
) {
    // Find all unique layers
    std::map<int, bool> layer_set;
    for (const auto & cs : all_stats) layer_set[cs.layer] = true;

    std::vector<LayerInfo> infos;
    for (const auto & kv : layer_set) {
        int layer = kv.first;
        // Build profiles for up to 4 heads to get average outlier_pct
        float sum_pct = 0.0f;
        int n_heads = 0;
        for (int h = 0; h < 128; h++) {  // generous upper bound
            HeadProfile hp;
            if (build_head_profile(all_stats, layer, h, hp)) {
                sum_pct += hp.outlier_pct;
                n_heads++;
            }
        }
        if (n_heads > 0) {
            infos.push_back({layer, sum_pct / n_heads, n_heads});
        }
    }

    // Sort by outlier percentage descending
    std::sort(infos.begin(), infos.end(),
              [](const LayerInfo & a, const LayerInfo & b) {
                  return a.avg_outlier_pct > b.avg_outlier_pct;
              });

    // Classify: top third extreme, middle third moderate, bottom third uniform
    int n = (int)infos.size();
    int third = std::max(1, n / 3);

    for (int i = 0; i < n; i++) {
        if (i < third) {
            extreme_layers.push_back(infos[i].layer);
        } else if (i < 2 * third) {
            moderate_layers.push_back(infos[i].layer);
        } else {
            uniform_layers.push_back(infos[i].layer);
        }
    }
}

// ---------------------------------------------------------------------------
// FWHT and rotation utilities
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

static float struct_sign(int i, int n) {
    uint64_t seed = (n == 96) ? 0x5452534C4F393600ULL : 0x5452534C31393200ULL;
    for (int k = 0; k <= i; k++) {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
    }
    return ((uint32_t)(seed >> 32) & 1) ? -1.0f : 1.0f;
}

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
// Generate vectors from real per-channel variance
// ---------------------------------------------------------------------------

static void generate_vectors_from_profile(
    const HeadProfile & hp,
    std::vector<std::vector<float>> & vecs,
    int count, std::mt19937 & rng
) {
    vecs.resize(count);
    for (int i = 0; i < count; i++) {
        vecs[i].resize(DIM);
        for (int c = 0; c < DIM; c++) {
            float sigma = sqrtf(hp.variance[c]);
            std::normal_distribution<float> dist(0.0f, std::max(sigma, 1e-6f));
            vecs[i][c] = dist(rng);
        }
    }
}

// ---------------------------------------------------------------------------
// Generate CORRELATED Q vectors: Q = K[target] + noise
// ---------------------------------------------------------------------------
//
// For each Q vector, pick a random K vector as the "target" (the key this
// query is looking for). Then add Gaussian noise scaled so that the
// correlation between Q and K[target] is ~80% (not 100% — real attention
// isn't an exact match).
//
// noise_scale controls the noise magnitude relative to K's norm.
// With noise_scale = 0.75 and K ~ N(0, sigma_per_channel):
//   correlation = ||K|| / sqrt(||K||^2 + noise_scale^2 * ||K||^2)
//               = 1 / sqrt(1 + noise_scale^2)
//   For noise_scale = 0.75: corr ~ 0.80

static void generate_correlated_q_vectors(
    const std::vector<std::vector<float>> & K,
    const HeadProfile & hp,
    std::vector<std::vector<float>> & Q_out,
    int count, std::mt19937 & rng
) {
    Q_out.resize(count);
    std::uniform_int_distribution<int> pick_k(0, (int)K.size() - 1);
    const float noise_scale = 0.75f;  // gives ~80% correlation

    for (int i = 0; i < count; i++) {
        Q_out[i].resize(DIM);
        int target = pick_k(rng);

        for (int d = 0; d < DIM; d++) {
            float sigma = sqrtf(hp.variance[d]);
            std::normal_distribution<float> noise_dist(0.0f, std::max(sigma, 1e-6f));
            Q_out[i][d] = K[target][d] + noise_scale * noise_dist(rng);
        }
    }
}

// ---------------------------------------------------------------------------
// Inner product metrics struct (forward declaration for attn_weighted)
// ---------------------------------------------------------------------------
//
// The paper's "bias" is NOT E[error] (which cancels for zero-mean Q).
// It's the ATTENUATION bias: MSE quantization produces K_hat = alpha*K + noise
// where alpha < 1 (attenuation). This means Q*K_hat ~ alpha*(Q*K) + Q*noise.
//
// The correct decomposition uses per-K-vector norm attenuation and the
// regression slope of Q*K_hat vs Q*K:
//
//   slope = Cov(Q*K_hat, Q*K) / Var(Q*K)   — should be 1.0 for unbiased
//   attenuation_bias = (1 - slope)          — fraction of signal lost
//   MSE = attenuation_bias_contribution + noise_variance
//
// where attenuation_bias_contribution = (1-slope)^2 * Var(Q*K)
// and noise_variance = MSE - attenuation_bias_contribution
//
// bias_pct = attenuation_bias_contribution / MSE * 100

struct IPMetrics {
    double slope;        // regression slope of Q*K_hat vs Q*K (1.0 = unbiased)
    double attn_bias;    // (1 - slope) — fraction of signal attenuated
    double attn_mse;     // attenuation contribution to MSE: (1-slope)^2 * Var(Q*K)
    double noise_var;    // residual noise after removing attenuation
    double mse;          // E[(Q*K_hat - Q*K)^2] = attn_mse + noise_var
    double rel_error;    // E[|Q*K_hat - Q*K| / |Q*K|]
    double bias_pct;     // attn_mse / MSE * 100
    double norm_ratio;   // avg(||K_hat|| / ||K||) — direct attenuation measure
};

// ---------------------------------------------------------------------------
// Attention-weighted IP metrics
// ---------------------------------------------------------------------------
//
// Same as compute_ip_metrics but weights errors by softmax attention probs.
// For each Q vector, compute scores = Q*K for all K, then softmax to get
// attention weights. Errors on high-attention keys matter more.
//
// This captures: if MSE attenuates large K vectors (which get high attention),
// the error is amplified because those are the keys that matter most.

static IPMetrics compute_ip_metrics_attn_weighted(
    const std::vector<std::vector<float>> & K,
    const std::vector<std::vector<float>> & K_hat,
    const std::vector<std::vector<float>> & Q
) {
    int nk = (int)K.size();
    int nq = (int)Q.size();

    // Scale factor for attention scores (1/sqrt(d))
    double inv_sqrt_d = 1.0 / sqrt((double)DIM);

    // Accumulate weighted regression statistics in double precision
    double sum_w      = 0.0;  // total weight
    double sum_wx     = 0.0;  // weighted sum of x (true dot)
    double sum_wy     = 0.0;  // weighted sum of y (quant dot)
    double sum_wxx    = 0.0;  // weighted sum of x^2
    double sum_wxy    = 0.0;  // weighted sum of x*y
    double sum_werrsq = 0.0;  // weighted sum of (y-x)^2
    double sum_wrel   = 0.0;  // weighted sum of |y-x|/|x|
    double sum_wrel_w = 0.0;  // weight for valid relative errors

    for (int qi = 0; qi < nq; qi++) {
        // Step 1: compute TRUE attention scores for this Q
        std::vector<double> scores(nk);
        double max_score = -1e30;
        for (int ki = 0; ki < nk; ki++) {
            double dot = 0.0;
            for (int d = 0; d < DIM; d++) {
                dot += (double)Q[qi][d] * (double)K[ki][d];
            }
            scores[ki] = dot * inv_sqrt_d;
            if (scores[ki] > max_score) max_score = scores[ki];
        }

        // Step 2: softmax
        std::vector<double> weights(nk);
        double sum_exp = 0.0;
        for (int ki = 0; ki < nk; ki++) {
            weights[ki] = exp(scores[ki] - max_score);
            sum_exp += weights[ki];
        }
        for (int ki = 0; ki < nk; ki++) {
            weights[ki] /= sum_exp;
        }

        // Step 3: accumulate weighted statistics
        for (int ki = 0; ki < nk; ki++) {
            double dot_true = 0.0, dot_quant = 0.0;
            for (int d = 0; d < DIM; d++) {
                dot_true  += (double)Q[qi][d] * (double)K[ki][d];
                dot_quant += (double)Q[qi][d] * (double)K_hat[ki][d];
            }

            double w = weights[ki];
            sum_w      += w;
            sum_wx     += w * dot_true;
            sum_wy     += w * dot_quant;
            sum_wxx    += w * dot_true * dot_true;
            sum_wxy    += w * dot_true * dot_quant;

            double error = dot_quant - dot_true;
            sum_werrsq += w * error * error;

            if (fabs(dot_true) > 1e-10) {
                sum_wrel   += w * fabs(error) / fabs(dot_true);
                sum_wrel_w += w;
            }
        }
    }

    // Weighted regression: slope = Cov_w(x,y) / Var_w(x)
    double mean_x  = sum_wx / sum_w;
    double mean_y  = sum_wy / sum_w;
    double var_x   = (sum_wxx / sum_w) - mean_x * mean_x;
    double cov_xy  = (sum_wxy / sum_w) - mean_x * mean_y;

    IPMetrics m;
    m.slope     = (var_x > 1e-30) ? cov_xy / var_x : 1.0;
    m.attn_bias = 1.0 - m.slope;
    m.mse       = sum_werrsq / sum_w;
    m.rel_error = (sum_wrel_w > 1e-30) ? sum_wrel / sum_wrel_w : 0.0;

    m.attn_mse  = m.attn_bias * m.attn_bias * var_x;
    m.noise_var = m.mse - m.attn_mse;
    if (m.noise_var < 0.0) m.noise_var = 0.0;

    m.bias_pct  = (m.mse > 1e-30) ? (m.attn_mse / m.mse) * 100.0 : 0.0;

    // Norm ratio (same as unweighted — property of K_hat vs K, not Q)
    int nk_count = (int)K.size();
    double sum_norm_ratio = 0.0;
    for (int ki = 0; ki < nk_count; ki++) {
        double norm_k = 0.0, norm_khat = 0.0;
        for (int d = 0; d < DIM; d++) {
            norm_k    += (double)K[ki][d] * (double)K[ki][d];
            norm_khat += (double)K_hat[ki][d] * (double)K_hat[ki][d];
        }
        if (norm_k > 1e-30) {
            sum_norm_ratio += sqrt(norm_khat) / sqrt(norm_k);
        }
    }
    m.norm_ratio = sum_norm_ratio / nk_count;

    return m;
}

// ---------------------------------------------------------------------------
// Split / unsplit
// ---------------------------------------------------------------------------

static void split_vector(const float * full, const HeadProfile & hp,
                         float * hi, float * lo) {
    for (int i = 0; i < N_HI; i++) hi[i] = full[hp.hi_channels[i]];
    for (int i = 0; i < N_LO; i++) lo[i] = full[hp.lo_channels[i]];
}

static void unsplit_vector(const float * hi, const float * lo,
                           const HeadProfile & hp, float * full) {
    for (int i = 0; i < N_HI; i++) full[hp.hi_channels[i]] = hi[i];
    for (int i = 0; i < N_LO; i++) full[hp.lo_channels[i]] = lo[i];
}

// ---------------------------------------------------------------------------
// Hi path: MSE quantization at given bit width with FWHT-32
// ---------------------------------------------------------------------------

static void mse_pass_hi(const float * in, float * recon, float * out_norm, int bits) {
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
// Hi quantizers
// ---------------------------------------------------------------------------

// 4-bit hi MSE (no QJL) — used by tqk3_sj-style
static void quantize_hi_4bit_mse(const float * in, float * out) {
    float norm;
    mse_pass_hi(in, out, &norm, 4);
}

// 5-bit hi MSE (no QJL) — used by tqk4_sj-style
static void quantize_hi_5bit_mse(const float * in, float * out) {
    float norm;
    mse_pass_hi(in, out, &norm, 5);
}

// 5+3 residual hi MSE (no QJL) — tqk5r3_sj
static void quantize_hi_residual_53(const float * in, float * out) {
    float residual[N_HI];
    memcpy(residual, in, N_HI * sizeof(float));

    // Pass 1: 5-bit
    float pass1_recon[N_HI];
    float pass1_norm;
    mse_pass_hi(residual, pass1_recon, &pass1_norm, 5);

    float recon_total[N_HI];
    for (int j = 0; j < N_HI; j++) {
        recon_total[j] = pass1_recon[j];
        residual[j]   -= pass1_recon[j];
    }

    // Pass 2: 3-bit residual
    float pass2_recon[N_HI];
    float pass2_norm;
    mse_pass_hi(residual, pass2_recon, &pass2_norm, 3);
    for (int j = 0; j < N_HI; j++) {
        recon_total[j] += pass2_recon[j];
    }

    memcpy(out, recon_total, N_HI * sizeof(float));
}

// 4-bit hi MSE + QJL correction on residual
static void quantize_hi_4bit_qjl(const float * in, float * out) {
    float hi[N_HI];
    memcpy(hi, in, N_HI * sizeof(float));

    // 4-bit MSE
    float norm = 0.0f;
    for (int j = 0; j < N_HI; j++) norm += hi[j] * hi[j];
    norm = sqrtf(norm);

    float recon[N_HI];
    if (norm > 1e-30f) {
        float inv = 1.0f / norm;
        float tmp[N_HI];
        for (int j = 0; j < N_HI; j++) tmp[j] = hi[j] * inv;
        fwht(tmp, N_HI);
        for (int j = 0; j < N_HI; j++) {
            int idx = nearest_centroid(tmp[j], c16_d32, 16);
            tmp[j] = c16_d32[idx];
        }
        fwht(tmp, N_HI);
        for (int j = 0; j < N_HI; j++) recon[j] = tmp[j] * norm;
    } else {
        memset(recon, 0, N_HI * sizeof(float));
    }

    // QJL on residual: FWHT-32 projection, store signs, reconstruct
    float residual[N_HI];
    for (int j = 0; j < N_HI; j++) residual[j] = hi[j] - recon[j];

    float rnorm = 0.0f;
    for (int j = 0; j < N_HI; j++) rnorm += residual[j] * residual[j];
    rnorm = sqrtf(rnorm);

    if (rnorm > 1e-30f) {
        float r[N_HI];
        memcpy(r, residual, N_HI * sizeof(float));
        fwht(r, N_HI);

        float corr[N_HI];
        for (int j = 0; j < N_HI; j++) {
            corr[j] = (r[j] >= 0.0f) ? 1.0f : -1.0f;
        }
        fwht(corr, N_HI);

        float scale = SQRT_PI_OVER_2 / (float)N_HI * rnorm;
        for (int j = 0; j < N_HI; j++) recon[j] += scale * corr[j];
    }

    memcpy(out, recon, N_HI * sizeof(float));
}

// 5-bit hi MSE + QJL correction on residual
static void quantize_hi_5bit_qjl(const float * in, float * out) {
    float hi[N_HI];
    memcpy(hi, in, N_HI * sizeof(float));

    // 5-bit MSE
    float norm = 0.0f;
    for (int j = 0; j < N_HI; j++) norm += hi[j] * hi[j];
    norm = sqrtf(norm);

    float recon[N_HI];
    if (norm > 1e-30f) {
        float inv = 1.0f / norm;
        float tmp[N_HI];
        for (int j = 0; j < N_HI; j++) tmp[j] = hi[j] * inv;
        fwht(tmp, N_HI);
        for (int j = 0; j < N_HI; j++) {
            int idx = nearest_centroid(tmp[j], c32_d32, 32);
            tmp[j] = c32_d32[idx];
        }
        fwht(tmp, N_HI);
        for (int j = 0; j < N_HI; j++) recon[j] = tmp[j] * norm;
    } else {
        memset(recon, 0, N_HI * sizeof(float));
    }

    // QJL on residual
    float residual[N_HI];
    for (int j = 0; j < N_HI; j++) residual[j] = hi[j] - recon[j];

    float rnorm = 0.0f;
    for (int j = 0; j < N_HI; j++) rnorm += residual[j] * residual[j];
    rnorm = sqrtf(rnorm);

    if (rnorm > 1e-30f) {
        float r[N_HI];
        memcpy(r, residual, N_HI * sizeof(float));
        fwht(r, N_HI);

        float corr[N_HI];
        for (int j = 0; j < N_HI; j++) {
            corr[j] = (r[j] >= 0.0f) ? 1.0f : -1.0f;
        }
        fwht(corr, N_HI);

        float scale = SQRT_PI_OVER_2 / (float)N_HI * rnorm;
        for (int j = 0; j < N_HI; j++) recon[j] += scale * corr[j];
    }

    memcpy(out, recon, N_HI * sizeof(float));
}

// 5+3 residual hi MSE + QJL correction on final residual
static void quantize_hi_residual_53_qjl(const float * in, float * out) {
    float hi[N_HI];
    memcpy(hi, in, N_HI * sizeof(float));

    // Pass 1: 5-bit
    float residual[N_HI];
    memcpy(residual, hi, N_HI * sizeof(float));

    float pass1_recon[N_HI];
    float pass1_norm;
    mse_pass_hi(residual, pass1_recon, &pass1_norm, 5);

    float recon_total[N_HI];
    for (int j = 0; j < N_HI; j++) {
        recon_total[j] = pass1_recon[j];
        residual[j]   -= pass1_recon[j];
    }

    // Pass 2: 3-bit residual
    float pass2_recon[N_HI];
    float pass2_norm;
    mse_pass_hi(residual, pass2_recon, &pass2_norm, 3);
    for (int j = 0; j < N_HI; j++) {
        recon_total[j] += pass2_recon[j];
        residual[j]    -= pass2_recon[j];
    }

    // QJL on final residual
    float rnorm = 0.0f;
    for (int j = 0; j < N_HI; j++) rnorm += residual[j] * residual[j];
    rnorm = sqrtf(rnorm);

    if (rnorm > 1e-30f) {
        float r[N_HI];
        memcpy(r, residual, N_HI * sizeof(float));
        fwht(r, N_HI);

        float corr[N_HI];
        for (int j = 0; j < N_HI; j++) {
            corr[j] = (r[j] >= 0.0f) ? 1.0f : -1.0f;
        }
        fwht(corr, N_HI);

        float scale = SQRT_PI_OVER_2 / (float)N_HI * rnorm;
        for (int j = 0; j < N_HI; j++) recon_total[j] += scale * corr[j];
    }

    memcpy(out, recon_total, N_HI * sizeof(float));
}

// ---------------------------------------------------------------------------
// Lo path: MSE quantization with structured rotation
// ---------------------------------------------------------------------------

static void quantize_lo_mse(const float * in, float * out, int bits) {
    CentroidSet cs = get_centroids_d96(bits);
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
        int idx = nearest_centroid(lo[j], cs.data, cs.count);
        lo[j] = cs.data[idx];
    }

    structured_unrotate_lo(lo, N_LO);
    for (int j = 0; j < N_LO; j++) out[j] = lo[j] * norm;
}

// QJL correction on lo residual: per-element signs in rotated space
static void qjl_correct_lo(const float * residual, float * out) {
    float r[N_LO];
    memcpy(r, residual, N_LO * sizeof(float));

    float rnorm = 0.0f;
    for (int j = 0; j < N_LO; j++) rnorm += r[j] * r[j];
    rnorm = sqrtf(rnorm);
    if (rnorm < 1e-30f) return;

    // Rotate residual into the rotated space
    structured_rotate_lo(r, N_LO);

    // Store per-element signs: 96 bits = 12 bytes
    float corr[N_LO];
    for (int j = 0; j < N_LO; j++) {
        corr[j] = (r[j] >= 0.0f) ? 1.0f : -1.0f;
    }
    structured_unrotate_lo(corr, N_LO);

    float scale = SQRT_PI_OVER_2 / (float)N_LO * rnorm;
    for (int j = 0; j < N_LO; j++) out[j] += scale * corr[j];
}

// ---------------------------------------------------------------------------
// Full-dim quantizers (no hi/lo split)
// ---------------------------------------------------------------------------

// q8_0: block-of-32, 8-bit signed + fp16 scale = 8.50 bpv
static void quant_q8_0(const float * in, float * out) {
    for (int b = 0; b < DIM; b += 32) {
        int bsz = std::min(32, DIM - b);
        float amax = 0.0f;
        for (int j = 0; j < bsz; j++) {
            float av = fabsf(in[b + j]);
            if (av > amax) amax = av;
        }
        float d = amax / 127.0f;
        float id = (d > 1e-30f) ? 1.0f / d : 0.0f;
        for (int j = 0; j < bsz; j++) {
            int q = (int)roundf(in[b + j] * id);
            if (q < -127) q = -127;
            if (q >  127) q =  127;
            out[b + j] = (float)q * d;
        }
    }
}

// q4_0: block-of-32, 4-bit unsigned + fp16 scale = 4.50 bpv
static void quant_q4_0(const float * in, float * out) {
    for (int b = 0; b < DIM; b += 32) {
        int bsz = std::min(32, DIM - b);
        float amax = 0.0f;
        for (int j = 0; j < bsz; j++) {
            float av = fabsf(in[b + j]);
            if (av > amax) amax = av;
        }
        float d = amax / 7.0f;
        float id = (d > 1e-30f) ? 1.0f / d : 0.0f;
        for (int j = 0; j < bsz; j++) {
            int q = (int)roundf(in[b + j] * id + 8.0f);
            if (q < 0)  q = 0;
            if (q > 15) q = 15;
            out[b + j] = ((float)q - 8.0f) * d;
        }
    }
}

// tqk4_0: FWHT-128 + 4-bit MSE = 4.125 bpv
static void quant_tqk4_0(const float * in, float * out) {
    float tmp[DIM];
    memcpy(tmp, in, DIM * sizeof(float));

    float norm = 0.0f;
    for (int j = 0; j < DIM; j++) norm += tmp[j] * tmp[j];
    norm = sqrtf(norm);
    if (norm < 1e-30f) { memset(out, 0, DIM * sizeof(float)); return; }

    float inv = 1.0f / norm;
    for (int j = 0; j < DIM; j++) tmp[j] *= inv;
    fwht(tmp, DIM);
    for (int j = 0; j < DIM; j++) {
        int idx = nearest_centroid(tmp[j], c16_d128, 16);
        tmp[j] = c16_d128[idx];
    }
    fwht(tmp, DIM);
    for (int j = 0; j < DIM; j++) out[j] = tmp[j] * norm;
}

// ---------------------------------------------------------------------------
// Scheme descriptors — the full test matrix
// ---------------------------------------------------------------------------

enum QJLMode {
    QJL_NONE,       // MSE-only, no QJL anywhere
    QJL_HI_ONLY,    // QJL on hi, MSE on lo
    QJL_BOTH,       // QJL on hi and lo
};

enum HiQuant {
    HI_4BIT,        // 4-bit MSE (tqk3_sj)
    HI_5BIT,        // 5-bit MSE (tqk4_sj)
    HI_53RES,       // 5+3 residual (tqk5r3_sj)
    HI_NONE,        // not a split scheme
};

struct SchemeDesc {
    const char * name;
    float        bpv;
    bool         is_split;      // uses hi/lo split?
    HiQuant      hi_quant;
    int          lo_bits;       // MSE bits for lo (3, 4, 5)
    QJLMode      qjl_mode;
};

// The test matrix from the spec
static SchemeDesc g_schemes[] = {
    // MSE-only (biased)
    { "tqk3_sj (no QJL)",       3.875f, true,  HI_4BIT,  3, QJL_NONE },
    { "tqk4_sj (no QJL)",       4.125f, true,  HI_5BIT,  3, QJL_NONE },
    { "tqk5r3_sj (no QJL)",     4.625f, true,  HI_53RES, 3, QJL_NONE },
    { "tqk4_0",                  4.125f, false, HI_NONE,  0, QJL_NONE },
    { "q4_0",                    4.500f, false, HI_NONE,  0, QJL_NONE },
    { "q8_0",                    8.500f, false, HI_NONE,  0, QJL_NONE },

    // MSE + QJL on hi only (partially unbiased)
    { "4bit+QJL hi, 3bit lo",   3.875f, true,  HI_4BIT,  3, QJL_HI_ONLY },
    { "5bit+QJL hi, 3bit lo",   4.125f, true,  HI_5BIT,  3, QJL_HI_ONLY },
    { "5+3res+QJL hi, 3bit lo", 5.000f, true,  HI_53RES, 3, QJL_HI_ONLY },

    // MSE + QJL on both (fully unbiased)
    { "4bit+QJL hi, 3bit+QJL",  4.750f, true,  HI_4BIT,  3, QJL_BOTH },
    { "5bit+QJL hi, 3bit+QJL",  5.000f, true,  HI_5BIT,  3, QJL_BOTH },
    { "5+3res+QJL, 3bit+QJL",   5.875f, true,  HI_53RES, 3, QJL_BOTH },

    // Lo-improved MSE + QJL
    { "5bit+QJL hi, 4bit+QJL",  5.750f, true,  HI_5BIT,  4, QJL_BOTH },
    { "5bit+QJL hi, 5bit+QJL",  6.500f, true,  HI_5BIT,  5, QJL_BOTH },
};

static const int N_SCHEMES = (int)(sizeof(g_schemes) / sizeof(g_schemes[0]));

// ---------------------------------------------------------------------------
// Apply a scheme to a single K vector
// ---------------------------------------------------------------------------

static void quantize_scheme(
    const float * in, float * out,
    const HeadProfile & hp,
    const SchemeDesc & scheme
) {
    if (!scheme.is_split) {
        // Full-dim schemes
        if (scheme.bpv > 8.0f) {
            quant_q8_0(in, out);
        } else if (scheme.bpv > 4.2f) {
            quant_q4_0(in, out);
        } else {
            quant_tqk4_0(in, out);
        }
        return;
    }

    float hi_in[N_HI], lo_in[N_LO];
    float hi_out[N_HI], lo_out[N_LO];
    split_vector(in, hp, hi_in, lo_in);

    // Hi path
    switch (scheme.hi_quant) {
        case HI_4BIT:
            if (scheme.qjl_mode >= QJL_HI_ONLY) {
                quantize_hi_4bit_qjl(hi_in, hi_out);
            } else {
                quantize_hi_4bit_mse(hi_in, hi_out);
            }
            break;
        case HI_5BIT:
            if (scheme.qjl_mode >= QJL_HI_ONLY) {
                quantize_hi_5bit_qjl(hi_in, hi_out);
            } else {
                quantize_hi_5bit_mse(hi_in, hi_out);
            }
            break;
        case HI_53RES:
            if (scheme.qjl_mode >= QJL_HI_ONLY) {
                quantize_hi_residual_53_qjl(hi_in, hi_out);
            } else {
                quantize_hi_residual_53(hi_in, hi_out);
            }
            break;
        case HI_NONE:
            break;
    }

    // Lo path: MSE
    quantize_lo_mse(lo_in, lo_out, scheme.lo_bits);

    // Lo QJL correction if requested
    if (scheme.qjl_mode == QJL_BOTH) {
        float lo_residual[N_LO];
        for (int j = 0; j < N_LO; j++) lo_residual[j] = lo_in[j] - lo_out[j];
        qjl_correct_lo(lo_residual, lo_out);
    }

    unsplit_vector(hi_out, lo_out, hp, out);
}

// ---------------------------------------------------------------------------
// Inner product metrics — unweighted (random Q)
// ---------------------------------------------------------------------------

static IPMetrics compute_ip_metrics(
    const std::vector<std::vector<float>> & K,
    const std::vector<std::vector<float>> & K_hat,
    const std::vector<std::vector<float>> & Q
) {
    int nk = (int)K.size();
    int nq = (int)Q.size();
    int total_pairs = nk * nq;

    // Accumulate dot products for regression in double precision
    double sum_xy = 0.0;   // sum of (Q*K) * (Q*K_hat)
    double sum_xx = 0.0;   // sum of (Q*K)^2
    // sum_yy intentionally not tracked — only slope matters, not R^2
    double sum_x  = 0.0;   // sum of (Q*K)
    double sum_y  = 0.0;   // sum of (Q*K_hat)
    double sum_error_sq = 0.0;
    double sum_rel_err  = 0.0;
    int valid_count     = 0;

    for (int qi = 0; qi < nq; qi++) {
        for (int ki = 0; ki < nk; ki++) {
            double dot_true  = 0.0;
            double dot_quant = 0.0;
            for (int d = 0; d < DIM; d++) {
                dot_true  += (double)Q[qi][d] * (double)K[ki][d];
                dot_quant += (double)Q[qi][d] * (double)K_hat[ki][d];
            }

            sum_xy += dot_true * dot_quant;
            sum_xx += dot_true * dot_true;
            sum_x  += dot_true;
            sum_y  += dot_quant;

            double error = dot_quant - dot_true;
            sum_error_sq += error * error;

            if (fabs(dot_true) > 1e-10) {
                sum_rel_err += fabs(error) / fabs(dot_true);
                valid_count++;
            }
        }
    }

    double n = (double)total_pairs;

    // Regression slope: slope = Cov(x,y) / Var(x)
    // where x = Q*K (true), y = Q*K_hat (quantized)
    double var_x = (sum_xx / n) - (sum_x / n) * (sum_x / n);
    double cov_xy = (sum_xy / n) - (sum_x / n) * (sum_y / n);

    IPMetrics m;
    m.slope = (var_x > 1e-30) ? cov_xy / var_x : 1.0;
    m.attn_bias = 1.0 - m.slope;
    m.mse = sum_error_sq / n;
    m.rel_error = (valid_count > 0) ? sum_rel_err / valid_count : 0.0;

    // Attenuation contribution: MSE from systematic shrinkage
    // If y ~ slope*x + noise, then E[(y-x)^2] = (slope-1)^2*E[x^2] + noise_var
    // But we want: (slope-1)^2 * Var(x) + noise_var  (since E[x]~0)
    m.attn_mse = m.attn_bias * m.attn_bias * var_x;
    m.noise_var = m.mse - m.attn_mse;
    if (m.noise_var < 0.0) m.noise_var = 0.0;  // numerical safety

    m.bias_pct = (m.mse > 1e-30) ? (m.attn_mse / m.mse) * 100.0 : 0.0;

    // Norm ratio: direct measure of vector attenuation
    double sum_norm_ratio = 0.0;
    for (int ki = 0; ki < nk; ki++) {
        double norm_k = 0.0, norm_khat = 0.0;
        for (int d = 0; d < DIM; d++) {
            norm_k    += (double)K[ki][d] * (double)K[ki][d];
            norm_khat += (double)K_hat[ki][d] * (double)K_hat[ki][d];
        }
        if (norm_k > 1e-30) {
            sum_norm_ratio += sqrt(norm_khat) / sqrt(norm_k);
        }
    }
    m.norm_ratio = sum_norm_ratio / nk;

    return m;
}

// ---------------------------------------------------------------------------
// Pareto frontier detection
// ---------------------------------------------------------------------------

static std::vector<bool> find_pareto_frontier(
    const std::vector<float> & bpv,
    const std::vector<double> & mse
) {
    int n = (int)bpv.size();
    std::vector<bool> on_frontier(n, false);

    // Sort indices by bpv ascending
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b) { return bpv[a] < bpv[b]; });

    double best_mse = 1e30;
    for (int i : idx) {
        if (mse[i] < best_mse) {
            on_frontier[i] = true;
            best_mse = mse[i];
        }
    }
    return on_frontier;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char ** argv) {
    init_generated_centroids();

    // Parse --csv flag
    const char * csv_path = "/tmp/qwen25_stats.csv";
    bool run_all_models = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
            csv_path = argv[++i];
        } else if (strcmp(argv[i], "--all") == 0) {
            run_all_models = true;
        } else {
            csv_path = argv[i];
        }
    }

    // Model CSV files for cross-model comparison
    struct ModelCSV {
        const char * name;
        const char * path;
    };
    ModelCSV all_models[] = {
        { "Qwen2.5 7B",   "/tmp/qwen25_stats.csv" },
        { "Qwen3 8B",     "/tmp/qwen3_stats.csv"  },
        { "Qwen2.5 1.5B", "/tmp/qwen15_stats.csv" },
        { "Mistral 7B",   "/tmp/mistral_stats.csv" },
    };
    int n_all_models = 4;

    // Determine which models to run
    std::vector<ModelCSV> models_to_run;
    if (run_all_models) {
        for (int i = 0; i < n_all_models; i++) {
            models_to_run.push_back(all_models[i]);
        }
    } else {
        // Single model: figure out the name from the path
        const char * model_name = "Custom";
        for (int i = 0; i < n_all_models; i++) {
            if (strcmp(csv_path, all_models[i].path) == 0) {
                model_name = all_models[i].name;
                break;
            }
        }
        models_to_run.push_back({model_name, csv_path});
    }

    printf("================================================================\n");
    printf("  TurboQuant Inner Product Analysis\n");
    printf("  Bias-Variance Decomposition for Dot Product Estimation\n");
    printf("================================================================\n\n");

    printf("Key: slope = regression of Q*K_hat vs Q*K (1.0 = no attenuation)\n");
    printf("     attn_MSE = (1-slope)^2 * Var(Q*K) = MSE from systematic shrinkage\n");
    printf("     noise_var = MSE - attn_MSE = residual noise\n");
    printf("     bias_pct = attn_MSE / MSE * 100\n\n");
    printf("     If bias_pct is HIGH for MSE-only schemes -> QJL is essential\n");
    printf("     If bias_pct is LOW -> QJL wastes bits (noise dominates)\n\n");

    // Print scheme table
    printf("%-26s  %5s  %s\n", "Scheme", "bpv", "Type");
    for (int i = 0; i < 55; i++) printf("-");
    printf("\n");
    for (int s = 0; s < N_SCHEMES; s++) {
        const char * type_str = "MSE-only (biased)";
        if (g_schemes[s].qjl_mode == QJL_HI_ONLY) type_str = "MSE + QJL hi (partial)";
        if (g_schemes[s].qjl_mode == QJL_BOTH)     type_str = "MSE + QJL both (unbiased)";
        if (!g_schemes[s].is_split && g_schemes[s].bpv <= 4.5f) type_str = "full-dim MSE (biased)";
        if (!g_schemes[s].is_split && g_schemes[s].bpv > 8.0f)  type_str = "reference (8-bit)";
        printf("%-26s  %5.3f  %s\n", g_schemes[s].name, g_schemes[s].bpv, type_str);
    }
    printf("\n");

    // =======================================================================
    // Cross-model storage for the final comparison table
    // =======================================================================

    // [model][tier] -> bias_pct for "tqk4_sj (no QJL)" (5-bit MSE, no QJL = index 1)
    static constexpr int MAX_MODELS = 4;
    static constexpr int N_TIERS    = 3;
    double cross_model_bias_pct[MAX_MODELS][N_TIERS];
    const char * tier_names[N_TIERS] = { "EXTREME", "MODERATE", "UNIFORM" };
    memset(cross_model_bias_pct, 0, sizeof(cross_model_bias_pct));

    for (int mi = 0; mi < (int)models_to_run.size(); mi++) {
        const char * model_name = models_to_run[mi].name;
        const char * model_path = models_to_run[mi].path;

        printf("================================================================\n");
        printf("  MODEL: %s (%s)\n", model_name, model_path);
        printf("================================================================\n\n");

        // Load CSV
        std::vector<ChannelStats> all_stats;
        if (!load_csv(model_path, all_stats)) {
            fprintf(stderr, "Failed to load CSV from %s, skipping\n", model_path);
            continue;
        }
        printf("Loaded %d channel records\n", (int)all_stats.size());

        // Auto-classify layers
        std::vector<int> extreme_layers, moderate_layers, uniform_layers;
        auto_classify_layers(all_stats, extreme_layers, moderate_layers, uniform_layers);

        printf("Auto-classified layers:\n");
        printf("  Extreme  (%d layers):", (int)extreme_layers.size());
        for (int l : extreme_layers) printf(" %d", l);
        printf("\n");
        printf("  Moderate (%d layers):", (int)moderate_layers.size());
        for (int l : moderate_layers) printf(" %d", l);
        printf("\n");
        printf("  Uniform  (%d layers):", (int)uniform_layers.size());
        for (int l : uniform_layers) printf(" %d", l);
        printf("\n\n");

        // Pick 3 representative layers per tier
        struct TierDesc {
            const char * name;
            int layers[3];
            int n_layers;
        };

        auto pick_3 = [](const std::vector<int> & pool, int out[3]) -> int {
            int n = (int)pool.size();
            if (n == 0) return 0;
            if (n <= 3) {
                for (int i = 0; i < n; i++) out[i] = pool[i];
                return n;
            }
            // Pick first, middle, last
            out[0] = pool[0];
            out[1] = pool[n / 2];
            out[2] = pool[n - 1];
            return 3;
        };

        TierDesc tiers[N_TIERS];
        tiers[0].name = "EXTREME";
        tiers[0].n_layers = pick_3(extreme_layers, tiers[0].layers);
        tiers[1].name = "MODERATE";
        tiers[1].n_layers = pick_3(moderate_layers, tiers[1].layers);
        tiers[2].name = "UNIFORM";
        tiers[2].n_layers = pick_3(uniform_layers, tiers[2].layers);

        // Results: [qmethod][tier][scheme] -> IPMetrics
        // All 3 Q methods share the same K and K_hat vectors
        std::vector<std::vector<std::vector<IPMetrics>>> all_results(QMETHOD_COUNT);
        for (int qm = 0; qm < QMETHOD_COUNT; qm++) {
            all_results[qm].resize(N_TIERS);
            for (int t = 0; t < N_TIERS; t++) {
                all_results[qm][t].resize(N_SCHEMES);
                for (int s = 0; s < N_SCHEMES; s++) {
                    all_results[qm][t][s] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
                }
            }
        }

        // Backward-compat alias: tier_results = all_results[QMETHOD_RANDOM]
        auto & tier_results = all_results[QMETHOD_RANDOM];

        // Process each tier
        std::vector<int> heads_per_tier(N_TIERS, 0);
        for (int t = 0; t < N_TIERS; t++) {
            if (tiers[t].n_layers == 0) {
                printf("  No layers for %s tier, skipping\n", tiers[t].name);
                continue;
            }

            printf("=== Processing %s tier ===\n", tiers[t].name);

            for (int li = 0; li < tiers[t].n_layers; li++) {
                int layer = tiers[t].layers[li];
                printf("  Layer %d...\n", layer);

                // Build profiles for all heads in this layer
                std::vector<HeadProfile> heads;
                for (int h = 0; h < 128; h++) {
                    HeadProfile hp;
                    if (build_head_profile(all_stats, layer, h, hp)) {
                        heads.push_back(hp);
                    }
                }

                if (heads.empty()) {
                    printf("    No head data found, skipping\n");
                    continue;
                }

                for (const auto & hp : heads) {
                    printf("    Head %d: outlier_pct = %.1f%%\n",
                           hp.head, hp.outlier_pct * 100.0f);

                    // Generate K vectors (shared across all Q methods)
                    std::mt19937 rng_k(42 + layer * 100 + hp.head);
                    std::vector<std::vector<float>> K;
                    generate_vectors_from_profile(hp, K, N_K_VEC, rng_k);

                    // Generate Q vectors for each method
                    // Method 0: Random Q (original)
                    std::mt19937 rng_q_rand(9999 + layer * 100 + hp.head);
                    std::vector<std::vector<float>> Q_random;
                    generate_vectors_from_profile(hp, Q_random, N_Q_VEC, rng_q_rand);

                    // Method 1: Correlated Q (Q = K[target] + noise)
                    std::mt19937 rng_q_corr(7777 + layer * 100 + hp.head);
                    std::vector<std::vector<float>> Q_correlated;
                    generate_correlated_q_vectors(K, hp, Q_correlated, N_Q_VEC, rng_q_corr);

                    // Method 2: Attention-weighted uses Q_random but different metric fn

                    // Quantize K once, shared across Q methods
                    for (int s = 0; s < N_SCHEMES; s++) {
                        std::vector<std::vector<float>> K_hat(N_K_VEC);
                        for (int i = 0; i < N_K_VEC; i++) {
                            K_hat[i].resize(DIM, 0.0f);
                            quantize_scheme(K[i].data(), K_hat[i].data(), hp, g_schemes[s]);
                        }

                        // Method 0: Random Q
                        {
                            IPMetrics m = compute_ip_metrics(K, K_hat, Q_random);
                            auto & r = all_results[QMETHOD_RANDOM][t][s];
                            r.slope      += m.slope;
                            r.attn_bias  += m.attn_bias;
                            r.attn_mse   += m.attn_mse;
                            r.noise_var  += m.noise_var;
                            r.mse        += m.mse;
                            r.rel_error  += m.rel_error;
                            r.norm_ratio += m.norm_ratio;
                        }

                        // Method 1: Correlated Q
                        {
                            IPMetrics m = compute_ip_metrics(K, K_hat, Q_correlated);
                            auto & r = all_results[QMETHOD_CORRELATED][t][s];
                            r.slope      += m.slope;
                            r.attn_bias  += m.attn_bias;
                            r.attn_mse   += m.attn_mse;
                            r.noise_var  += m.noise_var;
                            r.mse        += m.mse;
                            r.rel_error  += m.rel_error;
                            r.norm_ratio += m.norm_ratio;
                        }

                        // Method 2: Attention-weighted (random Q, softmax-weighted errors)
                        {
                            IPMetrics m = compute_ip_metrics_attn_weighted(K, K_hat, Q_random);
                            auto & r = all_results[QMETHOD_ATTN_WEIGHTED][t][s];
                            r.slope      += m.slope;
                            r.attn_bias  += m.attn_bias;
                            r.attn_mse   += m.attn_mse;
                            r.noise_var  += m.noise_var;
                            r.mse        += m.mse;
                            r.rel_error  += m.rel_error;
                            r.norm_ratio += m.norm_ratio;
                        }
                    }

                    heads_per_tier[t]++;
                }
            }

            // Average across all heads in this tier
            if (heads_per_tier[t] > 0) {
                for (int qm = 0; qm < QMETHOD_COUNT; qm++) {
                    for (int s = 0; s < N_SCHEMES; s++) {
                        auto & r = all_results[qm][t][s];
                        r.slope      /= heads_per_tier[t];
                        r.attn_bias  /= heads_per_tier[t];
                        r.attn_mse   /= heads_per_tier[t];
                        r.noise_var  /= heads_per_tier[t];
                        r.mse        /= heads_per_tier[t];
                        r.rel_error  /= heads_per_tier[t];
                        r.norm_ratio /= heads_per_tier[t];
                        // Recompute bias_pct from averaged values
                        r.bias_pct = (r.mse > 1e-30)
                            ? (r.attn_mse / r.mse) * 100.0
                            : 0.0;
                    }
                }
            }

            printf("  Processed %d heads in %s tier\n\n", heads_per_tier[t], tiers[t].name);
        }

        // =================================================================
        // Bias-Variance decomposition tables (THE key output)
        // =================================================================

        for (int t = 0; t < N_TIERS; t++) {
            printf("=== %s LAYERS (%s) ===\n\n", tiers[t].name, model_name);

            printf("%-26s  %5s  %6s  %12s  %12s  %12s  %9s  %8s\n",
                   "Scheme", "bpv", "slope", "attn_MSE", "noise_var", "MSE",
                   "rel_err%", "bias_pct");
            for (int i = 0; i < 105; i++) printf("-");
            printf("\n");

            for (int s = 0; s < N_SCHEMES; s++) {
                const IPMetrics & m = tier_results[t][s];
                printf("%-26s  %5.3f  %5.3f  %12.6f  %12.6f  %12.6f  %8.3f%%  %7.1f%%",
                       g_schemes[s].name,
                       g_schemes[s].bpv,
                       m.slope,
                       m.attn_mse,
                       m.noise_var,
                       m.mse,
                       m.rel_error * 100.0,
                       m.bias_pct);

                // Annotate what's happening
                if (m.bias_pct > 30.0) {
                    printf("  <-- attenuation bias dominates!");
                } else if (m.bias_pct > 10.0) {
                    printf("  <-- significant attenuation");
                } else if (m.bias_pct < 3.0 && g_schemes[s].qjl_mode == QJL_NONE
                           && g_schemes[s].is_split) {
                    printf("  <-- already low attn");
                }
                printf("\n");
            }
            printf("\n");

            // Norm ratio summary for this tier
            printf("  Norm ratios (||K_hat||/||K||, 1.0 = no attenuation):\n");
            for (int s = 0; s < N_SCHEMES; s++) {
                printf("    %-26s  %.4f\n", g_schemes[s].name, tier_results[t][s].norm_ratio);
            }
            printf("\n");

            // Store for cross-model comparison (index 1 = tqk4_sj no QJL)
            if (mi < MAX_MODELS) {
                cross_model_bias_pct[mi][t] = tier_results[t][1].bias_pct;
            }
        }

        // =================================================================
        // Q method comparison table — THE KEY ADDITION
        // =================================================================
        // Shows how bias_pct changes when Q correlates with K (real attention)
        // vs random Q (which averages away the correlation).

        for (int t = 0; t < N_TIERS; t++) {
            if (heads_per_tier[t] == 0) continue;

            printf("=== %s LAYERS: Q method comparison (%s) ===\n\n", tiers[t].name, model_name);

            // Header
            printf("%-26s  %5s", "Scheme", "bpv");
            for (int qm = 0; qm < QMETHOD_COUNT; qm++) {
                printf("   %13s", qmethod_names[qm]);
                printf("         ");
            }
            printf("\n");
            printf("%-26s  %5s", "", "");
            for (int qm = 0; qm < QMETHOD_COUNT; qm++) {
                (void)qm;
                printf("   %6s %6s", "err%", "bias%");
                printf("   ");
            }
            printf("\n");
            for (int i = 0; i < 100; i++) printf("-");
            printf("\n");

            for (int s = 0; s < N_SCHEMES; s++) {
                printf("%-26s  %5.3f", g_schemes[s].name, g_schemes[s].bpv);
                for (int qm = 0; qm < QMETHOD_COUNT; qm++) {
                    const IPMetrics & m = all_results[qm][t][s];
                    printf("   %5.1f%% %5.1f%%   ", m.rel_error * 100.0, m.bias_pct);
                }

                // Annotate: does correlated/weighted show more bias than random?
                double rand_bias = all_results[QMETHOD_RANDOM][t][s].bias_pct;
                double corr_bias = all_results[QMETHOD_CORRELATED][t][s].bias_pct;
                double attn_bias = all_results[QMETHOD_ATTN_WEIGHTED][t][s].bias_pct;
                if ((corr_bias > rand_bias + 5.0 || attn_bias > rand_bias + 5.0)
                    && g_schemes[s].qjl_mode == QJL_NONE && g_schemes[s].is_split) {
                    printf(" <-- HIDDEN BIAS revealed!");
                } else if (corr_bias < 5.0 && attn_bias < 5.0
                           && g_schemes[s].qjl_mode >= QJL_HI_ONLY) {
                    printf(" <-- QJL fixes it");
                }
                printf("\n");
            }
            printf("\n");

            // Interpretation
            printf("  Interpretation:\n");
            printf("    Random_Q:      Q independent of K, averages away magnitude-bias correlation\n");
            printf("    Correlated_Q:  Q ~ K+noise (80%% corr), targets high-magnitude K directly\n");
            printf("    Attn_Weighted: random Q, but errors weighted by softmax attention probs\n");
            printf("    If Correlated/Weighted show HIGHER bias%% than Random -> the bias is real\n");
            printf("    but hidden by uniform averaging. QJL is essential for real attention.\n\n");
        }

        // =================================================================
        // QJL impact analysis: compare matched pairs
        // =================================================================

        printf("=== QJL IMPACT ANALYSIS (%s) ===\n\n", model_name);
        printf("Compare MSE-only vs +QJL at same hi bits, averaged across tiers:\n\n");

        struct MatchedPair {
            const char * label;
            int idx_no_qjl;
            int idx_qjl_hi;
            int idx_qjl_both;  // -1 if N/A
        };
        MatchedPair pairs[] = {
            { "4-bit hi", 0, 6, 9 },     // tqk3_sj vs 4bit+QJL hi vs 4bit+QJL both
            { "5-bit hi", 1, 7, 10 },    // tqk4_sj vs 5bit+QJL hi vs 5bit+QJL both
            { "5+3 res",  2, 8, 11 },    // tqk5r3_sj vs 5+3res+QJL hi vs 5+3res+QJL both
        };

        printf("%-12s  %-8s  %6s  %12s  %12s  %12s  %10s  %10s\n",
               "Hi config", "QJL", "slope", "attn_MSE", "noise_var", "MSE", "rel_err%", "bias_pct");
        for (int i = 0; i < 100; i++) printf("-");
        printf("\n");

        for (const auto & pair : pairs) {
            int indices[] = { pair.idx_no_qjl, pair.idx_qjl_hi, pair.idx_qjl_both };
            const char * qjl_labels[] = { "none", "hi", "both" };

            for (int q = 0; q < 3; q++) {
                int idx = indices[q];
                if (idx < 0) continue;

                // Average across all tiers
                IPMetrics avg = {0, 0, 0, 0, 0, 0, 0, 0};
                int n_valid_tiers = 0;
                for (int t = 0; t < N_TIERS; t++) {
                    if (tier_results[t][idx].mse > 1e-30) {
                        avg.slope     += tier_results[t][idx].slope;
                        avg.attn_mse  += tier_results[t][idx].attn_mse;
                        avg.noise_var += tier_results[t][idx].noise_var;
                        avg.mse       += tier_results[t][idx].mse;
                        avg.rel_error += tier_results[t][idx].rel_error;
                        n_valid_tiers++;
                    }
                }
                if (n_valid_tiers > 0) {
                    avg.slope     /= n_valid_tiers;
                    avg.attn_mse  /= n_valid_tiers;
                    avg.noise_var /= n_valid_tiers;
                    avg.mse       /= n_valid_tiers;
                    avg.rel_error /= n_valid_tiers;
                    avg.bias_pct  = (avg.mse > 1e-30) ? (avg.attn_mse / avg.mse) * 100.0 : 0.0;
                }

                printf("%-12s  %-8s  %5.3f  %12.6f  %12.6f  %12.6f  %9.3f%%  %9.1f%%\n",
                       pair.label, qjl_labels[q],
                       avg.slope, avg.attn_mse, avg.noise_var, avg.mse,
                       avg.rel_error * 100.0, avg.bias_pct);
            }
            printf("\n");
        }

        // =================================================================
        // Pareto frontier per tier
        // =================================================================

        for (int t = 0; t < N_TIERS; t++) {
            printf("=== PARETO FRONTIER: %s (%s) ===\n\n", tiers[t].name, model_name);

            std::vector<float> bpv_vec(N_SCHEMES);
            std::vector<double> mse_vec(N_SCHEMES);
            for (int s = 0; s < N_SCHEMES; s++) {
                bpv_vec[s] = g_schemes[s].bpv;
                mse_vec[s] = tier_results[t][s].mse;
            }

            std::vector<bool> on_frontier = find_pareto_frontier(bpv_vec, mse_vec);

            // Sort by bpv for display
            std::vector<int> sorted_idx(N_SCHEMES);
            std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
            std::sort(sorted_idx.begin(), sorted_idx.end(),
                      [](int a, int b) { return g_schemes[a].bpv < g_schemes[b].bpv; });

            printf("%-26s  %5s  %12s  %9s  %8s  %s\n",
                   "Scheme", "bpv", "MSE", "rel_err%", "bias_pct", "Pareto?");
            for (int i = 0; i < 80; i++) printf("-");
            printf("\n");

            for (int si : sorted_idx) {
                const IPMetrics & m = tier_results[t][si];
                printf("%-26s  %5.3f  %12.6f  %8.3f%%  %7.1f%%  %s\n",
                       g_schemes[si].name,
                       g_schemes[si].bpv,
                       m.mse,
                       m.rel_error * 100.0,
                       m.bias_pct,
                       on_frontier[si] ? "***" : "");
            }
            printf("\n");
        }

        // =================================================================
        // Does the analysis match PPL? — practical validation
        // =================================================================
        //
        // Known PPL results (7B GPU, Qwen2.5 7B):
        //   tqk3_sjj  (3.75 bpv, QJL on both hi+lo): PPL 241  <- best
        //   tqk3_sj   (3.88 bpv, QJL on hi only):    PPL 332
        //   tqk4_sj   (4.13 bpv, no QJL on lo):      PPL 774  <- worst at MORE bits
        //
        // If QJL matters, the analysis should rank schemes with QJL better
        // (lower MSE / rel_error) than MSE-only schemes, especially under
        // correlated Q or attention-weighted metrics.

        printf("=== DOES THE ANALYSIS MATCH PPL? (%s) ===\n\n", model_name);
        printf("PPL ranking (7B GPU):  tqk3_sjj (241) > tqk3_sj (332) > tqk4_sj (774)\n");
        printf("  tqk3_sjj = 4bit+QJL hi, 3bit+QJL lo (3.75 bpv -> QJL on both)\n");
        printf("  tqk3_sj  = 4bit+QJL hi, 3bit lo     (3.88 bpv -> QJL hi only)\n");
        printf("  tqk4_sj  = 5bit hi, 3bit lo          (4.13 bpv -> no QJL)\n\n");

        // Map to scheme indices:
        //   tqk4_sj (no QJL)          = index 1 (5-bit hi MSE, 3-bit lo MSE)
        //   tqk3_sj (QJL hi only)     = index 6 (4bit+QJL hi, 3bit lo)
        //   tqk3_sjj (QJL both)       = index 9 (4bit+QJL hi, 3bit+QJL lo)
        struct PPLEntry {
            const char * name;
            int scheme_idx;
            int ppl;
            float bpv;
        };
        PPLEntry ppl_entries[] = {
            { "tqk3_sjj (QJL both)",   9,  241, 4.750f },
            { "tqk3_sj  (QJL hi)",     6,  332, 3.875f },
            { "tqk4_sj  (no QJL)",     1,  774, 4.125f },
        };
        int n_ppl = 3;

        printf("%-26s  %5s  %4s", "Scheme", "bpv", "PPL");
        for (int qm = 0; qm < QMETHOD_COUNT; qm++) {
            printf("  %13s_err%%", qmethod_names[qm]);
        }
        printf("\n");
        for (int i = 0; i < 110; i++) printf("-");
        printf("\n");

        // For each PPL entry, show the MODERATE tier results (most representative)
        int mod_tier = 1;  // MODERATE
        if (heads_per_tier[mod_tier] == 0) mod_tier = 0;  // fallback to EXTREME

        // Collect rel_errors per Q method for ranking
        double analysis_err[QMETHOD_COUNT][3];  // [qmethod][ppl_entry]

        for (int p = 0; p < n_ppl; p++) {
            int si = ppl_entries[p].scheme_idx;
            printf("%-26s  %5.3f  %4d", ppl_entries[p].name, ppl_entries[p].bpv, ppl_entries[p].ppl);
            for (int qm = 0; qm < QMETHOD_COUNT; qm++) {
                double re = all_results[qm][mod_tier][si].rel_error * 100.0;
                analysis_err[qm][p] = re;
                printf("  %17.2f%%", re);
            }
            printf("\n");
        }
        printf("\n");

        // Check if analysis ranking matches PPL ranking for each Q method
        // PPL ranking: entry[0] < entry[1] < entry[2] (lower PPL = better)
        // Analysis ranking: lower rel_error = better
        printf("Analysis ranking (by rel_error, lower=better):\n");
        for (int qm = 0; qm < QMETHOD_COUNT; qm++) {
            // Sort indices by analysis_err
            int order[3] = {0, 1, 2};
            for (int i = 0; i < 2; i++) {
                for (int j = i + 1; j < 3; j++) {
                    if (analysis_err[qm][order[i]] > analysis_err[qm][order[j]]) {
                        int tmp = order[i]; order[i] = order[j]; order[j] = tmp;
                    }
                }
            }
            printf("  %-14s: %s (%.1f%%) > %s (%.1f%%) > %s (%.1f%%)",
                   qmethod_names[qm],
                   ppl_entries[order[0]].name, analysis_err[qm][order[0]],
                   ppl_entries[order[1]].name, analysis_err[qm][order[1]],
                   ppl_entries[order[2]].name, analysis_err[qm][order[2]]);

            // Check if matches PPL ranking (best=0, mid=1, worst=2)
            bool match = (order[0] == 0 && order[2] == 2);
            printf("  %s\n", match ? "MATCH" : "MISMATCH");
        }
        printf("\n");
    }

    // =====================================================================
    // Cross-model comparison (only if --all)
    // =====================================================================

    if (run_all_models && (int)models_to_run.size() > 1) {
        printf("================================================================\n");
        printf("  CROSS-MODEL COMPARISON\n");
        printf("  BIAS FRACTION (bias^2/MSE) for tqk4_sj MSE-only (no QJL)\n");
        printf("================================================================\n\n");

        printf("%-16s", "Model");
        for (int t = 0; t < N_TIERS; t++) {
            printf("  %10s", tier_names[t]);
        }
        printf("\n");
        for (int i = 0; i < 50; i++) printf("-");
        printf("\n");

        for (int mi = 0; mi < (int)models_to_run.size() && mi < MAX_MODELS; mi++) {
            printf("%-16s", models_to_run[mi].name);
            for (int t = 0; t < N_TIERS; t++) {
                printf("  %9.1f%%", cross_model_bias_pct[mi][t]);
            }
            printf("\n");
        }
        printf("\n");

        printf("If bias fraction is consistently high (>30%%) across models and tiers,\n");
        printf("the paper's claim holds: QJL is worth the extra bits to remove bias.\n");
        printf("If consistently low (<10%%), MSE-only suffices and QJL wastes bits.\n\n");
    }

    // =====================================================================
    // Summary
    // =====================================================================

    printf("================================================================\n");
    printf("  SUMMARY\n");
    printf("================================================================\n\n");

    printf("The attenuation-noise decomposition answers THE key question:\n");
    printf("  MSE = attn_MSE + noise_var\n\n");
    printf("  - MSE-only quantization: optimizes reconstruction but ATTENUATES vectors\n");
    printf("    (||K_hat|| < ||K||), causing Q*K_hat < Q*K systematically (slope < 1)\n");
    printf("  - QJL correction: adds unbiased noise that restores the norm, pushing slope -> 1\n");
    printf("  - Paper claims: removing attenuation bias is worth the added noise\n\n");
    printf("  bias_pct tells what fraction of total error comes from attenuation.\n");
    printf("  If bias_pct >> 0%% for MSE-only, the paper is right: QJL is essential.\n");
    printf("  If bias_pct ~ 0%%, MSE already preserves norms and QJL wastes bits.\n\n");
    printf("  slope < 1 means systematic underestimation of large attention scores.\n");
    printf("  norm_ratio shows the direct vector-level attenuation.\n\n");

    return 0;
}
