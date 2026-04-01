// TurboQuant Moderate-Layer Analysis — real Qwen2.5 7B variance profiles
// Reads /tmp/qwen25_stats.csv for per-channel variance + outlier assignments.
// Tests layers 2(72%), 3(87%), 13(76%), 16(69%), 19(80%) as representative
// moderate outlier layers (53-90% range).
//
// Key question: for moderate outliers, is split+calibration worth the
// complexity over uniform q4_0/q5_0?
//
// Self-contained: no ggml deps, reads CSV, outputs metrics tables.
//
// Build: cmake --build build -t test-tq-analyze-moderate -j14
// Run:   ./build/bin/test-tq-analyze-moderate [path-to-csv]

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

static constexpr int N_HI  = 32;
static constexpr int N_LO  = 96;
static constexpr int DIM   = N_HI + N_LO;   // 128
static constexpr int N_VEC = 500;            // K vectors per layer
static constexpr int N_DOT_PAIRS = 50000;
static constexpr float SQRT_PI_OVER_2 = 1.2533141373155003f;

// ---------------------------------------------------------------------------
// Lloyd-Max centroids: d=32
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

// d=96 Lloyd-Max centroids
static const float c8_d96[8] = {
    -0.2169f, -0.1362f, -0.0768f, -0.0249f, 0.0249f, 0.0768f, 0.1362f, 0.2169f
};

static const float c4_d96[4] = {
    -0.1534f, -0.0462f, 0.0462f, 0.1534f
};

static const float c2_d96[2] = {
    -0.0816f, 0.0816f
};

static const float c16_d96[16] = {
    -0.2909f, -0.2244f, -0.1774f, -0.1388f, -0.1047f, -0.0731f, -0.0433f, -0.0144f,
     0.0144f,  0.0433f,  0.0731f,  0.1047f,  0.1388f,  0.1774f,  0.2244f,  0.2909f
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

// Generate centroids via quantile spacing for arbitrary dimension
static void generate_centroids(float * out, int n_centroids, int dim) {
    float sigma = 1.0f / sqrtf((float)dim);
    for (int i = 0; i < n_centroids; i++) {
        float p = ((float)i + 0.5f) / (float)n_centroids;
        out[i] = sigma * approx_inv_normal(p);
    }
}

// ---------------------------------------------------------------------------
// Centroid lookup helpers
// ---------------------------------------------------------------------------

struct CentroidSet {
    const float * data;
    int           count;
};

static float c64_d32[64];
static float c256_d32[256];

static CentroidSet get_centroids_d32(int bits) {
    switch (bits) {
        case 2: return { c4_d32,   4 };
        case 3: return { c8_d32,   8 };
        case 4: return { c16_d32, 16 };
        case 5: return { c32_d32, 32 };
        case 6: return { c64_d32, 64 };
        case 8: return { c256_d32, 256 };
        default: assert(false && "unsupported bit width for d=32"); return { nullptr, 0 };
    }
}

static CentroidSet get_centroids_d96(int bits) {
    switch (bits) {
        case 1: return { c2_d96,   2 };
        case 2: return { c4_d96,   4 };
        case 3: return { c8_d96,   8 };
        case 4: return { c16_d96, 16 };
        default: assert(false && "unsupported bit width for d=96"); return { nullptr, 0 };
    }
}

static void init_generated_centroids() {
    generate_centroids(c64_d32, 64, 32);
    generate_centroids(c256_d32, 256, 32);
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
    float variance[DIM];       // per-channel variance
    int   is_outlier[DIM];     // per-channel outlier flag from CSV
    int   hi_channels[N_HI];   // indices of hi (outlier) channels
    int   lo_channels[N_LO];   // indices of lo (non-outlier) channels
    float outlier_pct;         // fraction of variance in hi channels
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
        // Sort remaining by variance descending
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
    hp.outlier_pct = var_hi / (var_hi + var_lo);

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
// Generate K vectors from real per-channel variance
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
// Split a full 128-dim vector into hi[32] + lo[96] using channel mapping
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
// Hi path: single MSE pass + optional QJL
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

static void qjl_correct(const float * residual, float * out) {
    float r[N_HI];
    memcpy(r, residual, N_HI * sizeof(float));

    float rnorm = 0.0f;
    for (int j = 0; j < N_HI; j++) rnorm += r[j] * r[j];
    rnorm = sqrtf(rnorm);
    if (rnorm < 1e-30f) return;

    fwht(r, N_HI);

    uint8_t signs[4] = {0, 0, 0, 0};
    for (int j = 0; j < N_HI; j++) {
        if (r[j] >= 0.0f) signs[j / 8] |= (1 << (j % 8));
    }

    float corr[N_HI];
    for (int j = 0; j < N_HI; j++) {
        corr[j] = ((signs[j / 8] >> (j % 8)) & 1) ? 1.0f : -1.0f;
    }
    fwht(corr, N_HI);

    float scale = SQRT_PI_OVER_2 / (float)N_HI * rnorm;
    for (int j = 0; j < N_HI; j++) out[j] += scale * corr[j];
}

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
        mse_pass_hi(residual, pass_recon, &pass_norm, pass_bits[p]);
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
// Lo path variants
// ---------------------------------------------------------------------------

// 3-bit lo: structured rotation + 8 centroids d=96
static void quantize_lo_3bit(const float * in, float * out) {
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

// 2-bit lo: structured rotation + 4 centroids d=96
static void quantize_lo_2bit(const float * in, float * out) {
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
        int idx = nearest_centroid(lo[j], c4_d96, 4);
        lo[j] = c4_d96[idx];
    }

    structured_unrotate_lo(lo, N_LO);
    for (int j = 0; j < N_LO; j++) out[j] = lo[j] * norm;
}

// 4-bit lo: structured rotation + 16 centroids d=96
static void quantize_lo_4bit(const float * in, float * out) {
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
        int idx = nearest_centroid(lo[j], c16_d96, 16);
        lo[j] = c16_d96[idx];
    }

    structured_unrotate_lo(lo, N_LO);
    for (int j = 0; j < N_LO; j++) out[j] = lo[j] * norm;
}

// QJL correction on lo residual (after MSE lo quantization)
static void qjl_correct_lo(const float * residual, float * out) {
    float r[N_LO];
    memcpy(r, residual, N_LO * sizeof(float));

    float rnorm = 0.0f;
    for (int j = 0; j < N_LO; j++) rnorm += r[j] * r[j];
    rnorm = sqrtf(rnorm);
    if (rnorm < 1e-30f) return;

    // Use 3 separate FWHT-32 blocks for QJL on lo
    int bd = N_LO / 3;
    for (int b = 0; b < 3; b++) fwht(r + b * bd, bd);

    // Store signs: 96 bits = 12 bytes
    uint8_t signs[12] = {};
    for (int j = 0; j < N_LO; j++) {
        if (r[j] >= 0.0f) signs[j / 8] |= (1 << (j % 8));
    }

    float corr[N_LO];
    for (int j = 0; j < N_LO; j++) {
        corr[j] = ((signs[j / 8] >> (j % 8)) & 1) ? 1.0f : -1.0f;
    }
    for (int b = 0; b < 3; b++) fwht(corr + b * bd, bd);

    float scale = SQRT_PI_OVER_2 / (float)bd * rnorm;
    for (int j = 0; j < N_LO; j++) out[j] += scale * corr[j];
}

// ---------------------------------------------------------------------------
// Architecture descriptors
// ---------------------------------------------------------------------------

enum LoMode {
    LO_3BIT     = 0,  // 3-bit MSE (standard)
    LO_2BIT     = 1,  // 2-bit MSE
    LO_4BIT     = 2,  // 4-bit MSE
    LO_2BIT_QJL = 3,  // 2-bit MSE + QJL
};

struct Architecture {
    const char * name;
    const char * desc;
    float        bpv;
    int          hi_pass_bits[4];
    int          hi_n_passes;
    bool         hi_use_qjl;
    LoMode       lo_mode;
    bool         is_ref;
    int          ref_type;   // 0=q8_0, 1=q4_0, 2=q5_0
};

// ---------------------------------------------------------------------------
// Reference quantizers (uniform, full 128-dim)
// ---------------------------------------------------------------------------

static void ref_q8_0_quantize(const float * in, float * out, int dim) {
    for (int b = 0; b < dim; b += 32) {
        int bsz = std::min(32, dim - b);
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

static void ref_q4_0_quantize(const float * in, float * out, int dim) {
    for (int b = 0; b < dim; b += 32) {
        int bsz = std::min(32, dim - b);
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

static void ref_q5_0_quantize(const float * in, float * out, int dim) {
    for (int b = 0; b < dim; b += 32) {
        int bsz = std::min(32, dim - b);
        float amax = 0.0f;
        for (int j = 0; j < bsz; j++) {
            float av = fabsf(in[b + j]);
            if (av > amax) amax = av;
        }
        float d = amax / 15.0f;
        float id = (d > 1e-30f) ? 1.0f / d : 0.0f;
        for (int j = 0; j < bsz; j++) {
            int q = (int)roundf(in[b + j] * id + 16.0f);
            if (q < 0)  q = 0;
            if (q > 31) q = 31;
            out[b + j] = ((float)q - 16.0f) * d;
        }
    }
}

// ---------------------------------------------------------------------------
// Full-vector quantization for TQ architectures
// ---------------------------------------------------------------------------

static void quantize_tq_vector(
    const float * full_in, float * full_out,
    const HeadProfile & hp,
    const Architecture & arch
) {
    float hi_in[N_HI], lo_in[N_LO];
    float hi_out[N_HI], lo_out[N_LO];

    split_vector(full_in, hp, hi_in, lo_in);

    // Hi path
    quantize_hi_multipass(hi_in, hi_out, arch.hi_pass_bits, arch.hi_n_passes, arch.hi_use_qjl);

    // Lo path
    switch (arch.lo_mode) {
        case LO_3BIT:
            quantize_lo_3bit(lo_in, lo_out);
            break;
        case LO_2BIT:
            quantize_lo_2bit(lo_in, lo_out);
            break;
        case LO_4BIT:
            quantize_lo_4bit(lo_in, lo_out);
            break;
        case LO_2BIT_QJL: {
            quantize_lo_2bit(lo_in, lo_out);
            float lo_residual[N_LO];
            for (int j = 0; j < N_LO; j++) lo_residual[j] = lo_in[j] - lo_out[j];
            qjl_correct_lo(lo_residual, lo_out);
            break;
        }
    }

    unsplit_vector(hi_out, lo_out, hp, full_out);
}

// ---------------------------------------------------------------------------
// Metrics: full 128-dim + hi/lo breakdown
// ---------------------------------------------------------------------------

struct DetailedMetrics {
    double dot_rel_err;     // relative error on full 128-dim dot product
    double dot_bias;        // signed bias (mean of (quant - true) / |true|)
    double dot_err_hi;      // fraction of total absolute error from hi channels
    double dot_err_lo;      // fraction of total absolute error from lo channels
    float  bpv;
};

static DetailedMetrics compute_detailed_metrics(
    const std::vector<std::vector<float>> & K,
    const std::vector<std::vector<float>> & K_hat,
    const std::vector<std::vector<float>> & Q,
    const HeadProfile & hp
) {
    int nk = (int)K.size();
    int nq = (int)Q.size();

    double err_sum = 0.0, bias_sum = 0.0;
    double abs_err_hi_sum = 0.0, abs_err_lo_sum = 0.0;
    int count = 0;

    std::mt19937 rng(12345);
    for (int p = 0; p < N_DOT_PAIRS; p++) {
        int qi = rng() % nq;
        int ki = rng() % nk;

        double dot_true = 0.0, dot_quant = 0.0;
        for (int d = 0; d < DIM; d++) {
            dot_true  += (double)Q[qi][d] * (double)K[ki][d];
            dot_quant += (double)Q[qi][d] * (double)K_hat[ki][d];
        }

        if (fabs(dot_true) < 1e-10) continue;

        double rel_err = fabs(dot_true - dot_quant) / fabs(dot_true);
        double signed_bias = (dot_quant - dot_true) / fabs(dot_true);
        err_sum  += rel_err;
        bias_sum += signed_bias;

        // Decompose absolute error by hi vs lo channels
        double err_hi = 0.0, err_lo = 0.0;
        for (int i = 0; i < N_HI; i++) {
            int c = hp.hi_channels[i];
            err_hi += fabs((double)Q[qi][c] * ((double)K_hat[ki][c] - (double)K[ki][c]));
        }
        for (int i = 0; i < N_LO; i++) {
            int c = hp.lo_channels[i];
            err_lo += fabs((double)Q[qi][c] * ((double)K_hat[ki][c] - (double)K[ki][c]));
        }
        double total_abs = err_hi + err_lo;
        if (total_abs > 1e-30) {
            abs_err_hi_sum += err_hi / total_abs;
            abs_err_lo_sum += err_lo / total_abs;
        }
        count++;
    }

    DetailedMetrics m;
    m.dot_rel_err = (count > 0) ? err_sum / count : 0.0;
    m.dot_bias    = (count > 0) ? bias_sum / count : 0.0;
    m.dot_err_hi  = (count > 0) ? abs_err_hi_sum / count : 0.0;
    m.dot_err_lo  = (count > 0) ? abs_err_lo_sum / count : 0.0;
    m.bpv         = 0.0f;
    return m;
}

// ---------------------------------------------------------------------------
// BPV calculation
// ---------------------------------------------------------------------------

static float compute_bpv(const Architecture & arch) {
    if (arch.is_ref) {
        switch (arch.ref_type) {
            case 0: return 8.50f;  // q8_0: 32*8-bit + fp16 scale = 34 bytes / 32 vals = 8.5 bpv
            case 1: return 4.50f;  // q4_0: 32*4-bit + fp16 scale = 18 bytes / 32 vals = 4.5 bpv
            case 2: return 5.50f;  // q5_0: 32*5-bit + fp16 scale + 32-bit sign mask
            default: return 0.0f;
        }
    }

    // TQ BPV: (norm_hi(2) + norm_lo(2) + [rnorm2(2)] + [rnorm_qjl(2)] +
    //          qs_hi_all_passes + qs_lo + [signs_hi(4)] + [lo_qjl_stuff]) * 8 / 128
    float bytes = 2.0f + 2.0f; // norm_hi + norm_lo

    // hi MSE passes
    for (int p = 0; p < arch.hi_n_passes; p++) {
        bytes += ceilf((float)(N_HI * arch.hi_pass_bits[p]) / 8.0f);
    }
    if (arch.hi_n_passes > 1) bytes += 2.0f; // rnorm2

    // hi QJL
    if (arch.hi_use_qjl) {
        bytes += 2.0f + 4.0f; // rnorm_qjl + 32 sign bits
    }

    // lo quantized storage
    switch (arch.lo_mode) {
        case LO_3BIT:
            bytes += ceilf((float)(N_LO * 3) / 8.0f); // 36 bytes
            break;
        case LO_2BIT:
            bytes += ceilf((float)(N_LO * 2) / 8.0f); // 24 bytes
            break;
        case LO_4BIT:
            bytes += ceilf((float)(N_LO * 4) / 8.0f); // 48 bytes
            break;
        case LO_2BIT_QJL:
            bytes += ceilf((float)(N_LO * 2) / 8.0f); // 24 bytes for 2-bit
            bytes += 2.0f + 12.0f;  // rnorm_qjl_lo + 96 sign bits
            break;
    }

    return bytes * 8.0f / (float)DIM;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char ** argv) {
    init_generated_centroids();

    const char * csv_path = "/tmp/qwen25_stats.csv";
    if (argc > 1) csv_path = argv[1];

    printf("================================================================\n");
    printf("  TurboQuant Moderate-Layer Analysis\n");
    printf("  Real Qwen2.5 7B variance profiles, 32/96 outlier split\n");
    printf("  Key question: is split+calibration worth it for moderate layers?\n");
    printf("================================================================\n\n");

    // Load CSV
    std::vector<ChannelStats> all_stats;
    if (!load_csv(csv_path, all_stats)) {
        fprintf(stderr, "Failed to load CSV from %s\n", csv_path);
        return 1;
    }
    printf("Loaded %d channel records from %s\n\n", (int)all_stats.size(), csv_path);

    // Target layers: 2(72%), 3(87%), 13(76%), 16(69%), 19(80%)
    struct LayerDesc {
        int   layer;
        float expected_pct;  // approximate outlier % from user description
    };
    LayerDesc target_layers[] = {
        { 2,  0.72f },
        { 3,  0.87f },
        { 13, 0.76f },
        { 16, 0.69f },
        { 19, 0.80f },
    };
    int n_layers = (int)(sizeof(target_layers) / sizeof(target_layers[0]));

    // Define architectures
    Architecture archs[] = {
        // --- Standard baselines ---
        { "q4_0",       "uniform 4-bit (4.50 bpv)",                0, {}, 0, false, LO_3BIT, true, 1 },
        { "q5_0",       "uniform 5-bit (5.50 bpv)",                0, {}, 0, false, LO_3BIT, true, 2 },
        { "q8_0",       "uniform 8-bit (8.50 bpv)",                0, {}, 0, false, LO_3BIT, true, 0 },

        // --- Current TQ types ---
        { "tqk4_sj",    "5-bit hi + QJL, 3-bit lo",               0, {5}, 1, true,  LO_3BIT, false, 0 },
        { "tqk5r3_sj",  "5+3 residual hi, 3-bit lo",              0, {5, 3}, 2, false, LO_3BIT, false, 0 },
        { "tqk3_sj",    "4-bit hi + QJL, 3-bit lo",               0, {4}, 1, true,  LO_3BIT, false, 0 },

        // --- New candidates ---
        { "5r3_qjl_3",  "5+3 residual hi + QJL, 3-bit lo",        0, {5, 3}, 2, true,  LO_3BIT, false, 0 },
        { "4qjl_2qjl",  "4-bit hi + QJL, 2-bit lo + QJL",         0, {4}, 1, true,  LO_2BIT_QJL, false, 0 },
        { "5hi_4lo",     "5-bit hi, 4-bit lo (no QJL)",            0, {5}, 1, false, LO_4BIT, false, 0 },
    };
    int n_archs = (int)(sizeof(archs) / sizeof(archs[0]));

    // Compute BPV for all architectures
    for (int a = 0; a < n_archs; a++) {
        archs[a].bpv = compute_bpv(archs[a]);
    }

    // Print architecture table
    printf("%-14s  %5s  %-45s\n", "Architecture", "BPV", "Description");
    for (int i = 0; i < 70; i++) printf("-");
    printf("\n");
    for (int a = 0; a < n_archs; a++) {
        printf("%-14s  %5.2f  %-45s\n", archs[a].name, archs[a].bpv, archs[a].desc);
    }
    printf("\n");

    // Collect results: [layer_idx][head][arch]
    struct Result {
        const char * name;
        float        bpv;
        DetailedMetrics metrics;
    };

    // Per-layer aggregated results (averaged across heads)
    std::vector<std::vector<Result>> layer_results(n_layers);

    for (int li = 0; li < n_layers; li++) {
        int layer = target_layers[li].layer;

        printf("================================================================\n");
        printf("  Layer %d  (expected ~%.0f%% outlier)\n", layer, target_layers[li].expected_pct * 100.0f);
        printf("================================================================\n\n");

        // Build profiles for all heads
        std::vector<HeadProfile> heads;
        for (int h = 0; h < 4; h++) {
            HeadProfile hp;
            if (build_head_profile(all_stats, layer, h, hp)) {
                heads.push_back(hp);
            }
        }

        if (heads.empty()) {
            printf("  No data found for layer %d, skipping\n\n", layer);
            continue;
        }

        // Print per-head outlier percentages
        for (const auto & hp : heads) {
            printf("  Head %d: outlier%% = %.1f%%\n", hp.head, hp.outlier_pct * 100.0f);
        }
        printf("\n");

        // For each architecture, average metrics across all heads
        layer_results[li].resize(n_archs);

        for (int a = 0; a < n_archs; a++) {
            double avg_dot_rel_err = 0.0;
            double avg_dot_bias    = 0.0;
            double avg_dot_err_hi  = 0.0;
            double avg_dot_err_lo  = 0.0;

            for (const auto & hp : heads) {
                // Generate K and Q vectors using this head's variance profile
                std::mt19937 rng(42 + layer * 100 + hp.head);
                std::vector<std::vector<float>> K, Q;
                generate_vectors_from_profile(hp, K, N_VEC, rng);
                // Q uses a different seed but same profile
                std::mt19937 rng_q(9999 + layer * 100 + hp.head);
                generate_vectors_from_profile(hp, Q, N_VEC, rng_q);

                // Quantize all K vectors
                std::vector<std::vector<float>> K_hat(N_VEC);
                for (int i = 0; i < N_VEC; i++) {
                    K_hat[i].resize(DIM, 0.0f);
                    if (archs[a].is_ref) {
                        switch (archs[a].ref_type) {
                            case 0: ref_q8_0_quantize(K[i].data(), K_hat[i].data(), DIM); break;
                            case 1: ref_q4_0_quantize(K[i].data(), K_hat[i].data(), DIM); break;
                            case 2: ref_q5_0_quantize(K[i].data(), K_hat[i].data(), DIM); break;
                        }
                    } else {
                        quantize_tq_vector(K[i].data(), K_hat[i].data(), hp, archs[a]);
                    }
                }

                DetailedMetrics m = compute_detailed_metrics(K, K_hat, Q, hp);
                avg_dot_rel_err += m.dot_rel_err;
                avg_dot_bias    += m.dot_bias;
                avg_dot_err_hi  += m.dot_err_hi;
                avg_dot_err_lo  += m.dot_err_lo;
            }

            int nh = (int)heads.size();
            layer_results[li][a].name = archs[a].name;
            layer_results[li][a].bpv  = archs[a].bpv;
            layer_results[li][a].metrics.dot_rel_err = avg_dot_rel_err / nh;
            layer_results[li][a].metrics.dot_bias    = avg_dot_bias / nh;
            layer_results[li][a].metrics.dot_err_hi  = avg_dot_err_hi / nh;
            layer_results[li][a].metrics.dot_err_lo  = avg_dot_err_lo / nh;
            layer_results[li][a].metrics.bpv         = archs[a].bpv;

            printf("  %-14s (%5.2f bpv): dot_rel=%.4f%%  bias=%+.4f%%  hi_share=%.0f%%  lo_share=%.0f%%\n",
                   archs[a].name, archs[a].bpv,
                   layer_results[li][a].metrics.dot_rel_err * 100.0,
                   layer_results[li][a].metrics.dot_bias * 100.0,
                   layer_results[li][a].metrics.dot_err_hi * 100.0,
                   layer_results[li][a].metrics.dot_err_lo * 100.0);
        }
        printf("\n");
    }

    // =====================================================================
    // Summary table: dot_rel_err across all layers
    // =====================================================================
    printf("================================================================\n");
    printf("  SUMMARY: Dot Product Relative Error (%%) — lower is better\n");
    printf("================================================================\n\n");

    printf("%-14s %5s", "Architecture", "BPV");
    for (int li = 0; li < n_layers; li++) {
        char buf[16];
        snprintf(buf, sizeof(buf), "L%d", target_layers[li].layer);
        printf("  %8s", buf);
    }
    printf("  %8s\n", "AVG");

    for (int i = 0; i < 14 + 6 + (n_layers + 1) * 10; i++) printf("-");
    printf("\n");

    for (int a = 0; a < n_archs; a++) {
        printf("%-14s %5.2f", archs[a].name, archs[a].bpv);
        double sum = 0.0;
        int cnt = 0;
        for (int li = 0; li < n_layers; li++) {
            if (layer_results[li].empty()) {
                printf("  %8s", "N/A");
            } else {
                double v = layer_results[li][a].metrics.dot_rel_err * 100.0;
                printf("  %8.4f", v);
                sum += v;
                cnt++;
            }
        }
        if (cnt > 0) {
            printf("  %8.4f", sum / cnt);
        }
        printf("\n");
    }
    printf("\n");

    // =====================================================================
    // Summary table: dot_bias across all layers
    // =====================================================================
    printf("================================================================\n");
    printf("  SUMMARY: Dot Product Signed Bias (%%) — closer to 0 is better\n");
    printf("================================================================\n\n");

    printf("%-14s %5s", "Architecture", "BPV");
    for (int li = 0; li < n_layers; li++) {
        char buf[16];
        snprintf(buf, sizeof(buf), "L%d", target_layers[li].layer);
        printf("  %8s", buf);
    }
    printf("  %8s\n", "AVG");

    for (int i = 0; i < 14 + 6 + (n_layers + 1) * 10; i++) printf("-");
    printf("\n");

    for (int a = 0; a < n_archs; a++) {
        printf("%-14s %5.2f", archs[a].name, archs[a].bpv);
        double sum = 0.0;
        int cnt = 0;
        for (int li = 0; li < n_layers; li++) {
            if (layer_results[li].empty()) {
                printf("  %8s", "N/A");
            } else {
                double v = layer_results[li][a].metrics.dot_bias * 100.0;
                printf("  %+8.4f", v);
                sum += v;
                cnt++;
            }
        }
        if (cnt > 0) {
            printf("  %+8.4f", sum / cnt);
        }
        printf("\n");
    }
    printf("\n");

    // =====================================================================
    // Summary table: hi/lo error share (averaged across layers)
    // =====================================================================
    printf("================================================================\n");
    printf("  ERROR SOURCE: %% of absolute dot error from hi vs lo channels\n");
    printf("  (averaged across layers 2, 3, 13, 16, 19)\n");
    printf("================================================================\n\n");

    printf("%-14s %5s  %8s  %8s\n", "Architecture", "BPV", "hi(%%)", "lo(%%)");
    for (int i = 0; i < 42; i++) printf("-");
    printf("\n");

    for (int a = 0; a < n_archs; a++) {
        double avg_hi = 0.0, avg_lo = 0.0;
        int cnt = 0;
        for (int li = 0; li < n_layers; li++) {
            if (!layer_results[li].empty()) {
                avg_hi += layer_results[li][a].metrics.dot_err_hi;
                avg_lo += layer_results[li][a].metrics.dot_err_lo;
                cnt++;
            }
        }
        if (cnt > 0) {
            printf("%-14s %5.2f  %7.1f%%  %7.1f%%\n",
                   archs[a].name, archs[a].bpv,
                   avg_hi / cnt * 100.0, avg_lo / cnt * 100.0);
        }
    }
    printf("\n");

    // =====================================================================
    // Efficiency analysis: dot_rel_err per bpv
    // =====================================================================
    printf("================================================================\n");
    printf("  EFFICIENCY: avg dot_rel_err / bpv (lower = more efficient)\n");
    printf("================================================================\n\n");

    printf("%-14s %5s  %8s  %10s  %12s\n", "Architecture", "BPV", "Avg Err%%", "Err/BPV", "vs q4_0");
    for (int i = 0; i < 60; i++) printf("-");
    printf("\n");

    // Find q4_0 average error for comparison
    double q4_0_avg = 0.0;
    {
        int cnt = 0;
        for (int li = 0; li < n_layers; li++) {
            if (!layer_results[li].empty()) {
                q4_0_avg += layer_results[li][0].metrics.dot_rel_err;
                cnt++;
            }
        }
        if (cnt > 0) q4_0_avg /= cnt;
    }

    for (int a = 0; a < n_archs; a++) {
        double avg_err = 0.0;
        int cnt = 0;
        for (int li = 0; li < n_layers; li++) {
            if (!layer_results[li].empty()) {
                avg_err += layer_results[li][a].metrics.dot_rel_err;
                cnt++;
            }
        }
        if (cnt > 0) {
            avg_err /= cnt;
            double efficiency = avg_err * 100.0 / archs[a].bpv;
            double vs_q4 = (q4_0_avg > 1e-30) ? avg_err / q4_0_avg : 0.0;
            printf("%-14s %5.2f  %7.4f%%  %10.5f  %11.3fx\n",
                   archs[a].name, archs[a].bpv,
                   avg_err * 100.0, efficiency, vs_q4);
        }
    }
    printf("\n");

    // =====================================================================
    // Key findings
    // =====================================================================
    printf("================================================================\n");
    printf("  KEY FINDINGS\n");
    printf("================================================================\n\n");

    // Find best non-reference architecture at or below q4_0 bpv
    double best_err_sub_q4 = 1e30;
    const char * best_name_sub_q4 = "none";
    float best_bpv_sub_q4 = 0;
    for (int a = 0; a < n_archs; a++) {
        if (archs[a].bpv <= 4.50f) {
            double avg_err = 0.0;
            int cnt = 0;
            for (int li = 0; li < n_layers; li++) {
                if (!layer_results[li].empty()) {
                    avg_err += layer_results[li][a].metrics.dot_rel_err;
                    cnt++;
                }
            }
            if (cnt > 0) {
                avg_err /= cnt;
                if (avg_err < best_err_sub_q4) {
                    best_err_sub_q4 = avg_err;
                    best_name_sub_q4 = archs[a].name;
                    best_bpv_sub_q4 = archs[a].bpv;
                }
            }
        }
    }

    printf("  Best at <= 4.50 bpv: %s (%.2f bpv, %.4f%% dot err)\n",
           best_name_sub_q4, best_bpv_sub_q4, best_err_sub_q4 * 100.0);

    // Compare tqk4_sj vs q4_0
    {
        double tq4_err = 0.0, q4_err = 0.0;
        int cnt = 0;
        for (int li = 0; li < n_layers; li++) {
            if (!layer_results[li].empty()) {
                // q4_0 is index 0, tqk4_sj is index 3
                q4_err  += layer_results[li][0].metrics.dot_rel_err;
                tq4_err += layer_results[li][3].metrics.dot_rel_err;
                cnt++;
            }
        }
        if (cnt > 0) {
            q4_err /= cnt;
            tq4_err /= cnt;
            double ratio = (q4_err > 1e-30) ? tq4_err / q4_err : 0.0;
            printf("  tqk4_sj vs q4_0: %.4f%% vs %.4f%% (%.2fx at %.2f vs %.2f bpv)\n",
                   tq4_err * 100.0, q4_err * 100.0, ratio,
                   archs[3].bpv, archs[0].bpv);
        }
    }

    // Compare tqk3_sj vs q4_0
    {
        double tq3_err = 0.0, q4_err = 0.0;
        int cnt = 0;
        for (int li = 0; li < n_layers; li++) {
            if (!layer_results[li].empty()) {
                // q4_0 is index 0, tqk3_sj is index 5
                q4_err  += layer_results[li][0].metrics.dot_rel_err;
                tq3_err += layer_results[li][5].metrics.dot_rel_err;
                cnt++;
            }
        }
        if (cnt > 0) {
            q4_err /= cnt;
            tq3_err /= cnt;
            double ratio = (q4_err > 1e-30) ? tq3_err / q4_err : 0.0;
            printf("  tqk3_sj vs q4_0: %.4f%% vs %.4f%% (%.2fx at %.2f vs %.2f bpv)\n",
                   tq3_err * 100.0, q4_err * 100.0, ratio,
                   archs[5].bpv, archs[0].bpv);
        }
    }

    printf("\n  Conclusion: see summary tables above for whether split+calibration\n");
    printf("  provides sufficient quality gains over uniform quantization for\n");
    printf("  moderate outlier layers (53-90%% range).\n\n");

    return 0;
}
