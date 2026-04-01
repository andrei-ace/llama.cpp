// TurboQuant Softmax-Aware Analysis — real Qwen2.5 7B variance profiles
//
// Key insight: attention uses softmax(Q·K_1, Q·K_2, ..., Q·K_n), which is
// shift-invariant: softmax(x + c) = softmax(x). So constant bias in dot
// products doesn't matter. What matters is:
//   1. Variance of error across K vectors for a fixed Q
//   2. Correlation of error with true score
//   3. KL/JS divergence between true and quantized attention distributions
//
// This replaces per-vector dot product analysis with the correct metric for
// attention. If MSE bias cancels under softmax, QJL wastes bits.
//
// Reads /tmp/qwen25_stats.csv for per-channel variance + outlier assignments.
// Self-contained: no ggml deps.
//
// Build: cmake --build build -t test-tq-softmax-analysis -j14
// Run:   ./build/bin/test-tq-softmax-analysis [path-to-csv]

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
static constexpr int N_K_VEC = 200;            // K vectors per layer (simultaneous attention)
static constexpr int N_Q_VEC = 20;             // Q vectors per layer
static constexpr int N_SEEDS = 10;             // different seeds for position variation
static constexpr float SQRT_PI_OVER_2 = 1.2533141373155003f;

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

// Generated centroid tables
static float c32_d96[32];
static float c64_d96[64];

// d=128 centroids (for full-dim schemes)
static const float c4_d128[4] = {
    -0.1089f, -0.0328f, 0.0328f, 0.1089f
};
static const float c8_d128[8] = {
    -0.1536f, -0.0966f, -0.0544f, -0.0177f, 0.0177f, 0.0544f, 0.0966f, 0.1536f
};
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

static CentroidSet get_centroids_d128(int bits) {
    switch (bits) {
        case 2: return { c4_d128,   4 };
        case 3: return { c8_d128,   8 };
        case 4: return { c16_d128, 16 };
        default: assert(false && "unsupported bit width for d=128"); return { nullptr, 0 };
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
            hp.variance[cs.channel]   = cs.variance;
            hp.is_outlier[cs.channel] = cs.is_outlier;
            found++;
        }
    }
    if (found != DIM) {
        fprintf(stderr, "WARNING: layer %d head %d has %d channels (expected %d)\n",
                layer, head, found, DIM);
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
// Vector generation from real per-channel variance (reservoir sampling across seeds)
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

// Hi: 5+3 residual MSE (no QJL)
static void quantize_hi_residual(const float * in, float * out) {
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

// Hi: 5-bit MSE + QJL correction (tqk4_sj style)
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

// QJL correction on lo residual
static void qjl_correct_lo(const float * residual, float * out) {
    float r[N_LO];
    memcpy(r, residual, N_LO * sizeof(float));

    float rnorm = 0.0f;
    for (int j = 0; j < N_LO; j++) rnorm += r[j] * r[j];
    rnorm = sqrtf(rnorm);
    if (rnorm < 1e-30f) return;

    structured_rotate_lo(r, N_LO);

    float corr[N_LO];
    for (int j = 0; j < N_LO; j++) {
        corr[j] = (r[j] >= 0.0f) ? 1.0f : -1.0f;
    }
    structured_unrotate_lo(corr, N_LO);

    float scale = SQRT_PI_OVER_2 / (float)N_LO * rnorm;
    for (int j = 0; j < N_LO; j++) out[j] += scale * corr[j];
}

// ---------------------------------------------------------------------------
// Full-dim quantizers (no split)
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

// tqk4_0: FWHT-128 + 4-bit MSE = 4.13 bpv
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
// Quantization scheme descriptors
// ---------------------------------------------------------------------------

enum SchemeType {
    SCHEME_Q8_0,           // q8_0 reference (full-dim uniform)
    SCHEME_TQK4_SJ,       // split: 5-bit hi + QJL, 3-bit lo
    SCHEME_TQK5R3_SJ,     // split: 5+3 residual hi, 3-bit lo
    SCHEME_TQK4_0,        // full-dim: FWHT-128 + 4-bit MSE
    SCHEME_LO_VARIANT,    // split: 5+3 residual hi, variable lo
    SCHEME_ABLATION,      // ablation studies
};

enum HiType {
    HI_RESIDUAL_53,   // 5+3 residual MSE (tqk5r3_sj default)
    HI_5BIT_QJL,      // 5-bit MSE + QJL (tqk4_sj)
};

enum LoQJL {
    LO_NO_QJL,
    LO_WITH_QJL,
};

struct SchemeDesc {
    const char * name;
    SchemeType   type;
    float        bpv;
    // For split schemes:
    HiType       hi_type;
    int          lo_bits;
    LoQJL        lo_qjl;
};

// Apply a scheme to a single K vector
static void quantize_scheme(
    const float * in, float * out,
    const HeadProfile & hp,
    const SchemeDesc & scheme
) {
    switch (scheme.type) {
        case SCHEME_Q8_0: {
            quant_q8_0(in, out);
            break;
        }
        case SCHEME_TQK4_0: {
            quant_tqk4_0(in, out);
            break;
        }
        case SCHEME_TQK4_SJ: {
            float hi_in[N_HI], lo_in[N_LO];
            float hi_out[N_HI], lo_out[N_LO];
            split_vector(in, hp, hi_in, lo_in);
            quantize_hi_5bit_qjl(hi_in, hi_out);
            quantize_lo_mse(lo_in, lo_out, 3);
            unsplit_vector(hi_out, lo_out, hp, out);
            break;
        }
        case SCHEME_TQK5R3_SJ: {
            float hi_in[N_HI], lo_in[N_LO];
            float hi_out[N_HI], lo_out[N_LO];
            split_vector(in, hp, hi_in, lo_in);
            quantize_hi_residual(hi_in, hi_out);
            quantize_lo_mse(lo_in, lo_out, 3);
            unsplit_vector(hi_out, lo_out, hp, out);
            break;
        }
        case SCHEME_LO_VARIANT:
        case SCHEME_ABLATION: {
            float hi_in[N_HI], lo_in[N_LO];
            float hi_out[N_HI], lo_out[N_LO];
            split_vector(in, hp, hi_in, lo_in);

            // Hi is always 5+3 residual for lo variants and ablations
            if (scheme.hi_type == HI_5BIT_QJL) {
                quantize_hi_5bit_qjl(hi_in, hi_out);
            } else {
                quantize_hi_residual(hi_in, hi_out);
            }

            // Lo path
            quantize_lo_mse(lo_in, lo_out, scheme.lo_bits);

            if (scheme.lo_qjl == LO_WITH_QJL) {
                float lo_residual[N_LO];
                for (int j = 0; j < N_LO; j++) lo_residual[j] = lo_in[j] - lo_out[j];
                qjl_correct_lo(lo_residual, lo_out);
            }

            unsplit_vector(hi_out, lo_out, hp, out);
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// Softmax and divergence metrics
// ---------------------------------------------------------------------------

static void softmax(const float * scores, int n, float * out) {
    float max_s = scores[0];
    for (int i = 1; i < n; i++) if (scores[i] > max_s) max_s = scores[i];
    float sum = 0;
    for (int i = 0; i < n; i++) { out[i] = expf(scores[i] - max_s); sum += out[i]; }
    for (int i = 0; i < n; i++) out[i] /= sum;
}

static float kl_divergence(const float * p, const float * q, int n) {
    float kl = 0;
    for (int i = 0; i < n; i++) {
        if (p[i] > 1e-10f && q[i] > 1e-10f)
            kl += p[i] * logf(p[i] / q[i]);
    }
    return kl;
}

static float js_divergence(const float * p, const float * q, int n) {
    std::vector<float> m(n);
    for (int i = 0; i < n; i++) m[i] = 0.5f * (p[i] + q[i]);
    return 0.5f * (kl_divergence(p, m.data(), n) + kl_divergence(q, m.data(), n));
}

// ---------------------------------------------------------------------------
// Softmax metrics structure
// ---------------------------------------------------------------------------

struct SoftmaxMetrics {
    double js_div;           // Jensen-Shannon divergence (symmetric, bounded)
    double kl_div;           // KL divergence (p_true || p_quant)
    double top1_agree;       // fraction where argmax matches
    double top5_overlap;     // average overlap in top-5 keys
    double l1_dist;          // sum |p_true - p_quant|
    double max_shift;        // max_k |p_true[k] - p_quant[k]|
};

static SoftmaxMetrics compute_softmax_metrics(
    const std::vector<std::vector<float>> & K,
    const std::vector<std::vector<float>> & K_hat,
    const std::vector<std::vector<float>> & Q,
    const HeadProfile & hp
) {
    (void)hp; // not needed for softmax metrics, kept for interface consistency

    int nk = (int)K.size();
    int nq = (int)Q.size();

    double js_sum = 0.0, kl_sum = 0.0;
    double top1_sum = 0.0, top5_sum = 0.0;
    double l1_sum = 0.0, max_shift_sum = 0.0;
    int count = 0;

    std::vector<float> scores_true(nk);
    std::vector<float> scores_quant(nk);
    std::vector<float> attn_true(nk);
    std::vector<float> attn_quant(nk);

    for (int qi = 0; qi < nq; qi++) {
        // Compute dot products for all K vectors
        for (int ki = 0; ki < nk; ki++) {
            float dt = 0.0f, dq = 0.0f;
            for (int d = 0; d < DIM; d++) {
                dt += Q[qi][d] * K[ki][d];
                dq += Q[qi][d] * K_hat[ki][d];
            }
            scores_true[ki]  = dt;
            scores_quant[ki] = dq;
        }

        // Softmax
        softmax(scores_true.data(),  nk, attn_true.data());
        softmax(scores_quant.data(), nk, attn_quant.data());

        // JS divergence
        js_sum += js_divergence(attn_true.data(), attn_quant.data(), nk);

        // KL divergence
        kl_sum += kl_divergence(attn_true.data(), attn_quant.data(), nk);

        // Top-1 agreement
        int argmax_true = 0, argmax_quant = 0;
        for (int ki = 1; ki < nk; ki++) {
            if (attn_true[ki]  > attn_true[argmax_true])   argmax_true  = ki;
            if (attn_quant[ki] > attn_quant[argmax_quant]) argmax_quant = ki;
        }
        if (argmax_true == argmax_quant) top1_sum += 1.0;

        // Top-5 overlap
        std::vector<int> idx_true(nk), idx_quant(nk);
        std::iota(idx_true.begin(),  idx_true.end(),  0);
        std::iota(idx_quant.begin(), idx_quant.end(), 0);
        std::partial_sort(idx_true.begin(),  idx_true.begin()  + 5, idx_true.end(),
            [&](int a, int b) { return attn_true[a] > attn_true[b]; });
        std::partial_sort(idx_quant.begin(), idx_quant.begin() + 5, idx_quant.end(),
            [&](int a, int b) { return attn_quant[a] > attn_quant[b]; });
        int overlap = 0;
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                if (idx_true[i] == idx_quant[j]) { overlap++; break; }
            }
        }
        top5_sum += (double)overlap / 5.0;

        // L1 distance and max shift
        double l1 = 0.0, max_s = 0.0;
        for (int ki = 0; ki < nk; ki++) {
            double diff = fabs((double)attn_true[ki] - (double)attn_quant[ki]);
            l1 += diff;
            if (diff > max_s) max_s = diff;
        }
        l1_sum += l1;
        max_shift_sum += max_s;

        count++;
    }

    SoftmaxMetrics m;
    m.js_div      = (count > 0) ? js_sum / count : 0.0;
    m.kl_div      = (count > 0) ? kl_sum / count : 0.0;
    m.top1_agree  = (count > 0) ? top1_sum / count : 0.0;
    m.top5_overlap = (count > 0) ? top5_sum / count : 0.0;
    m.l1_dist     = (count > 0) ? l1_sum / count : 0.0;
    m.max_shift   = (count > 0) ? max_shift_sum / count : 0.0;
    return m;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char ** argv) {
    init_generated_centroids();

    const char * csv_path = "/tmp/qwen25_stats.csv";
    if (argc > 1) csv_path = argv[1];

    printf("================================================================\n");
    printf("  TurboQuant Softmax-Aware Analysis\n");
    printf("  Metric: attention distribution divergence (not dot product error)\n");
    printf("  softmax(Q.K) is shift-invariant: constant bias cancels out\n");
    printf("  Key question: does QJL's unbiased correction actually help\n");
    printf("  when measured by the metric that matters for attention?\n");
    printf("================================================================\n\n");

    // Load CSV
    std::vector<ChannelStats> all_stats;
    if (!load_csv(csv_path, all_stats)) {
        fprintf(stderr, "Failed to load CSV from %s\n", csv_path);
        return 1;
    }
    printf("Loaded %d channel records from %s\n\n", (int)all_stats.size(), csv_path);

    // ---------------------------------------------------------------------------
    // Define schemes
    // ---------------------------------------------------------------------------

    // bpv calculations:
    // q8_0: (2 + 32*1) * 4 blocks / 128 = 8.50 bpv
    // tqk4_sj: hi(2+20+4+2) + lo(2+36) = 28 + 38 = 66B -> 66*8/128 = 4.125 bpv
    // tqk5r3_sj: hi(2+20+2+12) + lo(2+36) = 36 + 38 = 74B -> 74*8/128 = 4.625 bpv
    // tqk4_0: (2 + 64) = 66B -> 66*8/128 = 4.125 bpv
    //
    // Lo variants (hi fixed at 5+3 residual = 36B):
    //   3-bit lo: 36 + 2 + 36 = 74B -> 4.625 bpv
    //   4-bit lo: 36 + 2 + 48 = 86B -> 5.375 bpv
    //   5-bit lo: 36 + 2 + 60 = 98B -> 6.125 bpv
    //   6-bit lo: 36 + 2 + 72 = 110B -> 6.875 bpv
    //   3-bit lo + QJL: 36 + 2 + 36 + 12 + 2 = 88B -> 5.500 bpv
    //   4-bit lo + QJL: 36 + 2 + 48 + 12 + 2 = 100B -> 6.250 bpv

    SchemeDesc schemes[] = {
        // Current types
        { "q8_0",               SCHEME_Q8_0,        8.50f,  HI_RESIDUAL_53, 3, LO_NO_QJL },
        { "tqk4_sj",           SCHEME_TQK4_SJ,     4.13f,  HI_5BIT_QJL,    3, LO_NO_QJL },
        { "tqk5r3_sj",        SCHEME_TQK5R3_SJ,   4.63f,  HI_RESIDUAL_53, 3, LO_NO_QJL },
        { "tqk4_0",           SCHEME_TQK4_0,       4.13f,  HI_RESIDUAL_53, 3, LO_NO_QJL },

        // Lo variants (fixed hi at 5+3 residual)
        { "lo_4bit",           SCHEME_LO_VARIANT,   5.38f,  HI_RESIDUAL_53, 4, LO_NO_QJL },
        { "lo_5bit",           SCHEME_LO_VARIANT,   6.13f,  HI_RESIDUAL_53, 5, LO_NO_QJL },
        { "lo_6bit",           SCHEME_LO_VARIANT,   6.88f,  HI_RESIDUAL_53, 6, LO_NO_QJL },
        { "lo_3bit+QJL",      SCHEME_LO_VARIANT,   5.50f,  HI_RESIDUAL_53, 3, LO_WITH_QJL },
        { "lo_4bit+QJL",      SCHEME_LO_VARIANT,   6.25f,  HI_RESIDUAL_53, 4, LO_WITH_QJL },

        // Ablations: MSE only vs QJL on hi only vs QJL on both
        { "mse_only",          SCHEME_LO_VARIANT,   4.63f,  HI_RESIDUAL_53, 3, LO_NO_QJL },   // same as tqk5r3_sj
        { "qjl_hi_only",      SCHEME_ABLATION,      4.13f,  HI_5BIT_QJL,    3, LO_NO_QJL },   // same as tqk4_sj
        { "qjl_both",         SCHEME_ABLATION,      5.50f,  HI_5BIT_QJL,    3, LO_WITH_QJL },
    };
    int n_schemes = (int)(sizeof(schemes) / sizeof(schemes[0]));

    // ---------------------------------------------------------------------------
    // Layer tiers: extreme, moderate, uniform (3 representative layers each)
    // ---------------------------------------------------------------------------

    struct TierDesc {
        const char * name;
        int layers[3];
        float expected_pct[3];
    };

    TierDesc tiers[] = {
        { "EXTREME",  { 0,  1, 27}, {0.97f, 0.95f, 0.60f} },
        { "MODERATE", { 2,  3, 13}, {0.72f, 0.87f, 0.76f} },
        { "UNIFORM",  {14, 16, 22}, {0.40f, 0.35f, 0.30f} },
    };
    int n_tiers = 3;

    // Results: [tier][scheme] -> averaged SoftmaxMetrics
    std::vector<std::vector<SoftmaxMetrics>> tier_results(n_tiers);
    for (int t = 0; t < n_tiers; t++) {
        tier_results[t].resize(n_schemes);
        for (int s = 0; s < n_schemes; s++) {
            tier_results[t][s] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        }
    }

    // Process each tier
    for (int t = 0; t < n_tiers; t++) {
        printf("=== Processing %s tier ===\n", tiers[t].name);

        int heads_in_tier = 0;

        for (int li = 0; li < 3; li++) {
            int layer = tiers[t].layers[li];
            printf("  Layer %d (expected ~%.0f%% outlier)...\n",
                   layer, tiers[t].expected_pct[li] * 100.0f);

            // Build profiles for heads 0-3
            std::vector<HeadProfile> heads;
            for (int h = 0; h < 4; h++) {
                HeadProfile hp;
                if (build_head_profile(all_stats, layer, h, hp)) {
                    heads.push_back(hp);
                }
            }

            if (heads.empty()) {
                printf("    No data found, skipping\n");
                continue;
            }

            for (const auto & hp : heads) {
                printf("    Head %d: outlier%% = %.1f%%\n", hp.head, hp.outlier_pct * 100.0f);
            }

            // For each head, run N_SEEDS random seeds to simulate diverse context positions
            for (const auto & hp : heads) {
                for (int seed_idx = 0; seed_idx < N_SEEDS; seed_idx++) {
                    uint32_t base_seed = (uint32_t)(42 + layer * 1000 + hp.head * 100 + seed_idx * 7);

                    // Generate K and Q vectors
                    std::mt19937 rng_k(base_seed);
                    std::vector<std::vector<float>> K;
                    generate_vectors_from_profile(hp, K, N_K_VEC, rng_k);

                    std::mt19937 rng_q(base_seed + 9999);
                    std::vector<std::vector<float>> Q;
                    generate_vectors_from_profile(hp, Q, N_Q_VEC, rng_q);

                    // Test each scheme
                    for (int s = 0; s < n_schemes; s++) {
                        // Quantize all K vectors
                        std::vector<std::vector<float>> K_hat(N_K_VEC);
                        for (int i = 0; i < N_K_VEC; i++) {
                            K_hat[i].resize(DIM, 0.0f);
                            quantize_scheme(K[i].data(), K_hat[i].data(), hp, schemes[s]);
                        }

                        SoftmaxMetrics m = compute_softmax_metrics(K, K_hat, Q, hp);
                        tier_results[t][s].js_div       += m.js_div;
                        tier_results[t][s].kl_div       += m.kl_div;
                        tier_results[t][s].top1_agree    += m.top1_agree;
                        tier_results[t][s].top5_overlap  += m.top5_overlap;
                        tier_results[t][s].l1_dist       += m.l1_dist;
                        tier_results[t][s].max_shift     += m.max_shift;
                    }
                }
                heads_in_tier++;
            }
        }

        // Average across all (heads * seeds) in this tier
        int total_runs = heads_in_tier * N_SEEDS;
        if (total_runs > 0) {
            for (int s = 0; s < n_schemes; s++) {
                tier_results[t][s].js_div       /= total_runs;
                tier_results[t][s].kl_div       /= total_runs;
                tier_results[t][s].top1_agree   /= total_runs;
                tier_results[t][s].top5_overlap /= total_runs;
                tier_results[t][s].l1_dist      /= total_runs;
                tier_results[t][s].max_shift    /= total_runs;
            }
        }

        printf("  Processed %d head-seed combinations\n\n", total_runs);
    }

    // =====================================================================
    // Output: per-tier results
    // =====================================================================

    for (int t = 0; t < n_tiers; t++) {
        printf("=== %s LAYERS ===\n\n", tiers[t].name);
        printf("%-18s  %5s  %9s  %9s  %10s  %12s  %8s  %9s\n",
               "Scheme", "bpv", "JS_div", "KL_div", "top1_agree", "top5_overlap", "L1_dist", "max_shift");
        for (int i = 0; i < 100; i++) printf("-");
        printf("\n");

        for (int s = 0; s < n_schemes; s++) {
            const SoftmaxMetrics & m = tier_results[t][s];
            printf("%-18s  %5.2f  %9.6f  %9.6f  %9.1f%%  %11.1f%%  %8.5f  %9.6f\n",
                   schemes[s].name,
                   schemes[s].bpv,
                   m.js_div,
                   m.kl_div,
                   m.top1_agree * 100.0,
                   m.top5_overlap * 100.0,
                   m.l1_dist,
                   m.max_shift);
        }
        printf("\n");
    }

    // =====================================================================
    // Key finding: does bias matter for softmax?
    // =====================================================================

    printf("=== KEY FINDING: Does bias actually matter for softmax? ===\n\n");
    printf("Compare MSE-only vs MSE+QJL at same hi config:\n\n");

    // Find indices for the ablation comparison
    int idx_mse_only   = -1;  // mse_only (tqk5r3_sj equivalent)
    int idx_qjl_hi     = -1;  // qjl_hi_only (tqk4_sj equivalent)
    int idx_qjl_both   = -1;  // qjl on both hi and lo
    int idx_lo3qjl     = -1;  // lo_3bit+QJL
    int idx_tqk5r3     = -1;  // tqk5r3_sj

    for (int s = 0; s < n_schemes; s++) {
        if (strcmp(schemes[s].name, "mse_only") == 0)      idx_mse_only = s;
        if (strcmp(schemes[s].name, "qjl_hi_only") == 0)   idx_qjl_hi = s;
        if (strcmp(schemes[s].name, "qjl_both") == 0)      idx_qjl_both = s;
        if (strcmp(schemes[s].name, "lo_3bit+QJL") == 0)   idx_lo3qjl = s;
        if (strcmp(schemes[s].name, "tqk5r3_sj") == 0)     idx_tqk5r3 = s;
    }

    for (int t = 0; t < n_tiers; t++) {
        printf("  %s tier:\n", tiers[t].name);

        if (idx_mse_only >= 0 && idx_tqk5r3 >= 0) {
            printf("    MSE-only (5+3 res hi, 3-bit lo, %.2f bpv):  JS = %.6f\n",
                   schemes[idx_mse_only].bpv, tier_results[t][idx_mse_only].js_div);
        }
        if (idx_qjl_hi >= 0) {
            printf("    QJL hi only (5-bit+QJL hi, 3-bit lo, %.2f bpv): JS = %.6f\n",
                   schemes[idx_qjl_hi].bpv, tier_results[t][idx_qjl_hi].js_div);
        }
        if (idx_lo3qjl >= 0) {
            printf("    MSE + QJL lo (5+3 res hi, 3+QJL lo, %.2f bpv): JS = %.6f\n",
                   schemes[idx_lo3qjl].bpv, tier_results[t][idx_lo3qjl].js_div);
        }
        if (idx_qjl_both >= 0) {
            printf("    QJL both (5+QJL hi, 3+QJL lo, %.2f bpv):     JS = %.6f\n",
                   schemes[idx_qjl_both].bpv, tier_results[t][idx_qjl_both].js_div);
        }

        if (idx_mse_only >= 0 && idx_qjl_hi >= 0) {
            double mse_js = tier_results[t][idx_mse_only].js_div;
            double qjl_js = tier_results[t][idx_qjl_hi].js_div;
            if (mse_js > 1e-12 && qjl_js > 1e-12) {
                double ratio = qjl_js / mse_js;
                if (ratio < 0.9) {
                    printf("    --> QJL hi HELPS (%.0f%% lower JS at lower bpv)\n", (1.0 - ratio) * 100.0);
                } else if (ratio > 1.1) {
                    printf("    --> QJL hi HURTS (%.0f%% higher JS and costs more)\n", (ratio - 1.0) * 100.0);
                } else {
                    printf("    --> QJL hi has NEGLIGIBLE effect on softmax (ratio %.2f)\n", ratio);
                    printf("       Bias cancels under softmax -> QJL wastes bits on hi\n");
                }
            }
        }
        printf("\n");
    }

    printf("  INTERPRETATION:\n");
    printf("    If JS_div is similar for MSE-only vs MSE+QJL at comparable bpv:\n");
    printf("      -> Bias doesn't matter for softmax -> QJL wastes bits\n");
    printf("    If JS_div is much lower with QJL:\n");
    printf("      -> Bias does distort relative rankings -> QJL helps\n\n");

    // =====================================================================
    // Pareto frontier: bpv vs JS divergence per tier
    // =====================================================================

    printf("=== PARETO FRONTIER: bpv vs JS_divergence per tier ===\n\n");

    // Sort schemes by bpv
    std::vector<int> sorted_idx(n_schemes);
    std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
    std::sort(sorted_idx.begin(), sorted_idx.end(),
              [&](int a, int b) { return schemes[a].bpv < schemes[b].bpv; });

    printf("%-18s  %5s", "Scheme", "bpv");
    for (int t = 0; t < n_tiers; t++) {
        printf("  %12s_JS", tiers[t].name);
    }
    printf("  Pareto?\n");
    for (int i = 0; i < 90; i++) printf("-");
    printf("\n");

    // Track pareto frontier (minimum JS seen at each bpv level)
    for (int si : sorted_idx) {
        printf("%-18s  %5.2f", schemes[si].name, schemes[si].bpv);

        bool is_pareto = true;
        for (int t = 0; t < n_tiers; t++) {
            printf("  %15.6f", tier_results[t][si].js_div);

            // Check if dominated: another scheme with lower bpv has lower JS
            for (int sj : sorted_idx) {
                if (schemes[sj].bpv < schemes[si].bpv - 0.01f &&
                    tier_results[t][sj].js_div < tier_results[t][si].js_div) {
                    is_pareto = false;
                }
            }
        }

        printf("  %s\n", is_pareto ? " ***" : "");
    }
    printf("\n");

    // =====================================================================
    // Cross-tier summary: average JS across all tiers
    // =====================================================================

    printf("=== CROSS-TIER SUMMARY (avg JS across all tiers) ===\n\n");

    printf("%-18s  %5s  %12s  %10s  %12s\n",
           "Scheme", "bpv", "avg_JS_div", "top1_agree", "top5_overlap");
    for (int i = 0; i < 70; i++) printf("-");
    printf("\n");

    for (int si : sorted_idx) {
        double avg_js = 0.0, avg_top1 = 0.0, avg_top5 = 0.0;
        for (int t = 0; t < n_tiers; t++) {
            avg_js   += tier_results[t][si].js_div;
            avg_top1 += tier_results[t][si].top1_agree;
            avg_top5 += tier_results[t][si].top5_overlap;
        }
        avg_js   /= n_tiers;
        avg_top1 /= n_tiers;
        avg_top5 /= n_tiers;

        printf("%-18s  %5.2f  %12.6f  %9.1f%%  %11.1f%%\n",
               schemes[si].name, schemes[si].bpv, avg_js,
               avg_top1 * 100.0, avg_top5 * 100.0);
    }

    printf("\n");
    return 0;
}
