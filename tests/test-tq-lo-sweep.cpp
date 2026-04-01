// TurboQuant Lo-Channel Sweep — find minimum bpv matching q8_0 quality
// Reads /tmp/qwen25_stats.csv for per-channel variance + outlier assignments.
// Tests layers 2, 3, 13, 16, 19 (moderate, 53-90% outlier).
//
// Hi path is FIXED at 5+3 residual MSE (norm_hi + rnorm2_hi + 20B + 12B = 36 bytes
// for 32 channels). This contributes ~9% of error — already solved.
//
// Lo configs sweep different quantization schemes on the 96 regular channels
// with bare block-diagonal 3xFWHT-32 rotation.
//
// Goal: find cheapest config within 5% of q8_0's ~4.2% dot error.
//
// Self-contained: no ggml deps, reads CSV, outputs metrics tables.
//
// Build: cmake --build build -t test-tq-lo-sweep -j14
// Run:   ./build/bin/test-tq-lo-sweep [path-to-csv]

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
static constexpr int N_K_VEC = 500;          // K vectors per layer
static constexpr int N_Q_VEC = 50;           // Q vectors per layer
static constexpr int N_DOT_PAIRS = 50000;
static constexpr float SQRT_PI_OVER_2 = 1.2533141373155003f;

// Fixed hi budget: 5+3 residual MSE
// norm_hi(2) + rnorm2_hi(2) + 5-bit qs(20) + 3-bit qs(12) = 36 bytes
static constexpr int HI_FIXED_BYTES = 36;

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

static const float c2_d96[2] = {
    -0.0816f, 0.0816f
};

static const float c4_d96[4] = {
    -0.1534f, -0.0462f, 0.0462f, 0.1534f
};

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
static float c256_d96[256];

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
    generate_centroids(c256_d96, 256, 96);
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
        case 1: return { c2_d96,   2 };
        case 2: return { c4_d96,   4 };
        case 3: return { c8_d96,   8 };
        case 4: return { c16_d96, 16 };
        case 5: return { c32_d96, 32 };
        case 6: return { c64_d96, 64 };
        case 8: return { c256_d96, 256 };
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
// Hi path: FIXED 5+3 residual MSE
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

static void quantize_hi_fixed(const float * in, float * out) {
    // Fixed: 5+3 residual MSE (no QJL on hi)
    float recon_total[N_HI];
    memset(recon_total, 0, N_HI * sizeof(float));

    float residual[N_HI];
    memcpy(residual, in, N_HI * sizeof(float));

    // Pass 1: 5-bit
    float pass1_recon[N_HI];
    float pass1_norm;
    mse_pass_hi(residual, pass1_recon, &pass1_norm, 5);
    for (int j = 0; j < N_HI; j++) {
        recon_total[j] += pass1_recon[j];
        residual[j]    -= pass1_recon[j];
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

// ---------------------------------------------------------------------------
// Lo path: generic MSE quantization at any bit width
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

// ---------------------------------------------------------------------------
// QJL correction on lo residual (per-element signs in rotated space)
// ---------------------------------------------------------------------------

static void qjl_correct_lo(const float * residual, float * out) {
    float r[N_LO];
    memcpy(r, residual, N_LO * sizeof(float));

    float rnorm = 0.0f;
    for (int j = 0; j < N_LO; j++) rnorm += r[j] * r[j];
    rnorm = sqrtf(rnorm);
    if (rnorm < 1e-30f) return;

    // Rotate residual into the same rotated space
    structured_rotate_lo(r, N_LO);

    // Store per-element signs: 96 bits = 12 bytes
    uint8_t signs[12] = {};
    for (int j = 0; j < N_LO; j++) {
        if (r[j] >= 0.0f) signs[j / 8] |= (1 << (j % 8));
    }

    // Reconstruct correction in rotated space, then unrotate
    float corr[N_LO];
    for (int j = 0; j < N_LO; j++) {
        corr[j] = ((signs[j / 8] >> (j % 8)) & 1) ? 1.0f : -1.0f;
    }
    structured_unrotate_lo(corr, N_LO);

    float scale = SQRT_PI_OVER_2 / (float)N_LO * rnorm;
    for (int j = 0; j < N_LO; j++) out[j] += scale * corr[j];
}

// ---------------------------------------------------------------------------
// Lo config descriptor
// ---------------------------------------------------------------------------

enum LoType {
    LO_MSE_ONLY,       // Single MSE pass
    LO_MSE_QJL,        // MSE + QJL correction
    LO_RESIDUAL,       // Two MSE passes (residual)
    LO_RESIDUAL_QJL,   // Two MSE passes + QJL on final residual
};

struct LoConfig {
    const char * name;
    LoType       type;
    int          pass1_bits;    // First MSE pass bit width
    int          pass2_bits;    // Second MSE pass bit width (0 if single pass)
    bool         use_qjl;
    int          lo_bytes;      // Total bytes for lo (excl norm_lo)
};

// Compute lo_bytes for a config
static int compute_lo_bytes(const LoConfig & cfg) {
    int bytes = 0;

    // Pass 1 quantized indices
    bytes += (int)ceilf((float)(N_LO * cfg.pass1_bits) / 8.0f);

    if (cfg.type == LO_RESIDUAL || cfg.type == LO_RESIDUAL_QJL) {
        // Pass 2 quantized indices + rnorm2_lo
        bytes += (int)ceilf((float)(N_LO * cfg.pass2_bits) / 8.0f);
        bytes += 2; // rnorm2_lo (fp16)
    }

    if (cfg.use_qjl) {
        // 96 sign bits = 12 bytes + rnorm_qjl (fp16) = 14 bytes
        bytes += 12 + 2;
    }

    return bytes;
}

// ---------------------------------------------------------------------------
// Full-vector quantization with a given lo config
// ---------------------------------------------------------------------------

static void quantize_vector(
    const float * full_in, float * full_out,
    const HeadProfile & hp,
    const LoConfig & cfg
) {
    float hi_in[N_HI], lo_in[N_LO];
    float hi_out[N_HI], lo_out[N_LO];

    split_vector(full_in, hp, hi_in, lo_in);

    // Hi path: fixed 5+3 residual MSE
    quantize_hi_fixed(hi_in, hi_out);

    // Lo path: depends on config
    switch (cfg.type) {
        case LO_MSE_ONLY: {
            quantize_lo_mse(lo_in, lo_out, cfg.pass1_bits);
            break;
        }
        case LO_MSE_QJL: {
            quantize_lo_mse(lo_in, lo_out, cfg.pass1_bits);
            float lo_residual[N_LO];
            for (int j = 0; j < N_LO; j++) lo_residual[j] = lo_in[j] - lo_out[j];
            qjl_correct_lo(lo_residual, lo_out);
            break;
        }
        case LO_RESIDUAL: {
            // Pass 1
            quantize_lo_mse(lo_in, lo_out, cfg.pass1_bits);
            // Residual
            float lo_residual[N_LO];
            for (int j = 0; j < N_LO; j++) lo_residual[j] = lo_in[j] - lo_out[j];
            // Pass 2 on residual
            float lo_out2[N_LO];
            quantize_lo_mse(lo_residual, lo_out2, cfg.pass2_bits);
            for (int j = 0; j < N_LO; j++) lo_out[j] += lo_out2[j];
            break;
        }
        case LO_RESIDUAL_QJL: {
            // Pass 1
            quantize_lo_mse(lo_in, lo_out, cfg.pass1_bits);
            // Residual
            float lo_residual[N_LO];
            for (int j = 0; j < N_LO; j++) lo_residual[j] = lo_in[j] - lo_out[j];
            // Pass 2 on residual
            float lo_out2[N_LO];
            quantize_lo_mse(lo_residual, lo_out2, cfg.pass2_bits);
            for (int j = 0; j < N_LO; j++) {
                lo_out[j] += lo_out2[j];
                lo_residual[j] -= lo_out2[j];
            }
            // QJL on final residual
            qjl_correct_lo(lo_residual, lo_out);
            break;
        }
    }

    unsplit_vector(hi_out, lo_out, hp, full_out);
}

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

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

struct Metrics {
    double dot_rel_err;     // relative error on full 128-dim dot product
    double dot_bias;        // signed bias (mean of (quant - true) / |true|)
    double dot_err_hi;      // fraction of total absolute error from hi channels
    double dot_err_lo;      // fraction of total absolute error from lo channels
};

static Metrics compute_metrics(
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

    Metrics m;
    m.dot_rel_err = (count > 0) ? err_sum / count : 0.0;
    m.dot_bias    = (count > 0) ? bias_sum / count : 0.0;
    m.dot_err_hi  = (count > 0) ? abs_err_hi_sum / count : 0.0;
    m.dot_err_lo  = (count > 0) ? abs_err_lo_sum / count : 0.0;
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
    printf("  TurboQuant Lo-Channel Sweep\n");
    printf("  Hi FIXED: 5+3 residual MSE (36 bytes, 32 channels)\n");
    printf("  Lo SWEPT: various quantization on 96 channels\n");
    printf("  Goal: minimum bpv within 5%% of q8_0 (~4.2%% dot error)\n");
    printf("================================================================\n\n");

    // Load CSV
    std::vector<ChannelStats> all_stats;
    if (!load_csv(csv_path, all_stats)) {
        fprintf(stderr, "Failed to load CSV from %s\n", csv_path);
        return 1;
    }
    printf("Loaded %d channel records from %s\n\n", (int)all_stats.size(), csv_path);

    // Define lo configurations to sweep
    LoConfig lo_configs[] = {
        // MSE only
        { "2-bit MSE",       LO_MSE_ONLY,     2, 0, false, 0 },
        { "3-bit MSE",       LO_MSE_ONLY,     3, 0, false, 0 },  // current
        { "4-bit MSE",       LO_MSE_ONLY,     4, 0, false, 0 },
        { "5-bit MSE",       LO_MSE_ONLY,     5, 0, false, 0 },
        { "6-bit MSE",       LO_MSE_ONLY,     6, 0, false, 0 },
        { "8-bit MSE",       LO_MSE_ONLY,     8, 0, false, 0 },

        // MSE + QJL
        { "2-bit + QJL",     LO_MSE_QJL,      2, 0, true,  0 },
        { "3-bit + QJL",     LO_MSE_QJL,      3, 0, true,  0 },
        { "4-bit + QJL",     LO_MSE_QJL,      4, 0, true,  0 },

        // Residual MSE
        { "3+2 residual",    LO_RESIDUAL,      3, 2, false, 0 },
        { "3+3 residual",    LO_RESIDUAL,      3, 3, false, 0 },
        { "4+3 residual",    LO_RESIDUAL,      4, 3, false, 0 },

        // Residual MSE + QJL
        { "3+2 res + QJL",   LO_RESIDUAL_QJL,  3, 2, true,  0 },
        { "3+3 res + QJL",   LO_RESIDUAL_QJL,  3, 3, true,  0 },
    };
    int n_configs = (int)(sizeof(lo_configs) / sizeof(lo_configs[0]));

    // Compute lo_bytes for each config
    for (int i = 0; i < n_configs; i++) {
        lo_configs[i].lo_bytes = compute_lo_bytes(lo_configs[i]);
    }

    // Print config table
    printf("Lo configurations (hi fixed at 36 bytes = 5+3 residual MSE):\n\n");
    printf("  %-20s  lo_bytes  total_bytes  total_bpv\n", "Config");
    for (int i = 0; i < 60; i++) printf("-");
    printf("\n");
    for (int i = 0; i < n_configs; i++) {
        int total_bytes = HI_FIXED_BYTES + 2 + lo_configs[i].lo_bytes;  // +2 for norm_lo
        float total_bpv = (float)(total_bytes * 8) / (float)DIM;
        printf("  %-20s  %7d   %10d   %8.3f\n",
               lo_configs[i].name, lo_configs[i].lo_bytes, total_bytes, total_bpv);
    }
    printf("  %-20s                        %8.3f  (reference)\n", "q8_0", 8.500f);
    printf("\n");

    // Target layers
    struct LayerDesc {
        int   layer;
        float expected_pct;
    };
    LayerDesc target_layers[] = {
        { 2,  0.72f },
        { 3,  0.87f },
        { 13, 0.76f },
        { 16, 0.69f },
        { 19, 0.80f },
    };
    int n_layers = (int)(sizeof(target_layers) / sizeof(target_layers[0]));

    // Results: [config_idx] -> averaged metrics
    struct Result {
        const char * name;
        int          lo_bytes;
        float        total_bpv;
        Metrics      avg_metrics;
    };

    // +1 for q8_0 reference
    std::vector<Result> results(n_configs + 1);

    // Initialize accumulators
    for (int c = 0; c < n_configs; c++) {
        int total_bytes = HI_FIXED_BYTES + 2 + lo_configs[c].lo_bytes;
        results[c].name      = lo_configs[c].name;
        results[c].lo_bytes  = lo_configs[c].lo_bytes;
        results[c].total_bpv = (float)(total_bytes * 8) / (float)DIM;
        results[c].avg_metrics = { 0.0, 0.0, 0.0, 0.0 };
    }
    // q8_0 reference
    results[n_configs].name      = "q8_0 reference";
    results[n_configs].lo_bytes  = 0;
    results[n_configs].total_bpv = 8.500f;
    results[n_configs].avg_metrics = { 0.0, 0.0, 0.0, 0.0 };

    int total_heads_processed = 0;

    for (int li = 0; li < n_layers; li++) {
        int layer = target_layers[li].layer;

        printf("Processing layer %d (expected ~%.0f%% outlier)...\n",
               layer, target_layers[li].expected_pct * 100.0f);

        // Build profiles for all heads in this layer
        std::vector<HeadProfile> heads;
        for (int h = 0; h < 4; h++) {
            HeadProfile hp;
            if (build_head_profile(all_stats, layer, h, hp)) {
                heads.push_back(hp);
            }
        }

        if (heads.empty()) {
            printf("  No data found for layer %d, skipping\n", layer);
            continue;
        }

        for (const auto & hp : heads) {
            printf("  Head %d: outlier%% = %.1f%%\n", hp.head, hp.outlier_pct * 100.0f);
        }

        for (const auto & hp : heads) {
            // Generate K and Q vectors
            std::mt19937 rng_k(42 + layer * 100 + hp.head);
            std::vector<std::vector<float>> K;
            generate_vectors_from_profile(hp, K, N_K_VEC, rng_k);

            std::mt19937 rng_q(9999 + layer * 100 + hp.head);
            std::vector<std::vector<float>> Q;
            generate_vectors_from_profile(hp, Q, N_Q_VEC, rng_q);

            // Test each lo config
            for (int c = 0; c < n_configs; c++) {
                std::vector<std::vector<float>> K_hat(N_K_VEC);
                for (int i = 0; i < N_K_VEC; i++) {
                    K_hat[i].resize(DIM, 0.0f);
                    quantize_vector(K[i].data(), K_hat[i].data(), hp, lo_configs[c]);
                }

                Metrics m = compute_metrics(K, K_hat, Q, hp);
                results[c].avg_metrics.dot_rel_err += m.dot_rel_err;
                results[c].avg_metrics.dot_bias    += m.dot_bias;
                results[c].avg_metrics.dot_err_hi  += m.dot_err_hi;
                results[c].avg_metrics.dot_err_lo  += m.dot_err_lo;
            }

            // q8_0 reference
            {
                std::vector<std::vector<float>> K_hat(N_K_VEC);
                for (int i = 0; i < N_K_VEC; i++) {
                    K_hat[i].resize(DIM, 0.0f);
                    ref_q8_0_quantize(K[i].data(), K_hat[i].data(), DIM);
                }

                Metrics m = compute_metrics(K, K_hat, Q, hp);
                results[n_configs].avg_metrics.dot_rel_err += m.dot_rel_err;
                results[n_configs].avg_metrics.dot_bias    += m.dot_bias;
                results[n_configs].avg_metrics.dot_err_hi  += m.dot_err_hi;
                results[n_configs].avg_metrics.dot_err_lo  += m.dot_err_lo;
            }

            total_heads_processed++;
        }
    }

    // Average across all heads
    if (total_heads_processed > 0) {
        for (int c = 0; c <= n_configs; c++) {
            results[c].avg_metrics.dot_rel_err /= total_heads_processed;
            results[c].avg_metrics.dot_bias    /= total_heads_processed;
            results[c].avg_metrics.dot_err_hi  /= total_heads_processed;
            results[c].avg_metrics.dot_err_lo  /= total_heads_processed;
        }
    }

    printf("\nProcessed %d layer-head combinations\n", total_heads_processed);

    // =====================================================================
    // Main results table
    // =====================================================================
    printf("\n");
    printf("================================================================\n");
    printf("  RESULTS: Lo Sweep with Fixed 5+3 Residual Hi\n");
    printf("  Averaged across layers 2, 3, 13, 16, 19 (all heads)\n");
    printf("================================================================\n\n");

    double q8_err = results[n_configs].avg_metrics.dot_rel_err * 100.0;

    printf("%-20s  lo_bytes  total_bpv  dot_rel_err%%  dot_bias%%   hi_share  lo_share\n",
           "Lo config");
    for (int i = 0; i < 95; i++) printf("-");
    printf("\n");

    int cheapest_within_5pct = -1;
    float cheapest_bpv = 999.0f;

    for (int c = 0; c < n_configs; c++) {
        double err_pct  = results[c].avg_metrics.dot_rel_err * 100.0;
        double bias_pct = results[c].avg_metrics.dot_bias * 100.0;
        double hi_share = results[c].avg_metrics.dot_err_hi * 100.0;
        double lo_share = results[c].avg_metrics.dot_err_lo * 100.0;

        // Check if within 5% of q8_0 quality (i.e., err <= q8_err * 1.05)
        bool within_target = (err_pct <= q8_err * 1.05);
        const char * marker = within_target ? " <--" : "";

        if (within_target && results[c].total_bpv < cheapest_bpv) {
            cheapest_within_5pct = c;
            cheapest_bpv = results[c].total_bpv;
        }

        printf("%-20s  %7d   %8.3f   %10.4f%%  %+8.4f%%   %5.1f%%    %5.1f%%%s\n",
               results[c].name,
               results[c].lo_bytes,
               results[c].total_bpv,
               err_pct, bias_pct,
               hi_share, lo_share,
               marker);
    }

    // q8_0 reference line
    printf("%-20s  %7s   %8.3f   %10.4f%%  %+8.4f%%   %5s    %5s\n",
           "q8_0 reference", "-",
           results[n_configs].total_bpv,
           q8_err,
           results[n_configs].avg_metrics.dot_bias * 100.0,
           "-", "-");

    printf("\n");

    // =====================================================================
    // Sorted by bpv (ascending) for easy reading
    // =====================================================================
    printf("================================================================\n");
    printf("  SORTED BY BPV (ascending)\n");
    printf("================================================================\n\n");

    // Build index array and sort by bpv
    std::vector<int> sorted_idx(n_configs + 1);
    std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
    std::sort(sorted_idx.begin(), sorted_idx.end(),
              [&](int a, int b) { return results[a].total_bpv < results[b].total_bpv; });

    printf("%-20s  total_bpv  dot_rel_err%%  vs_q8_0\n", "Lo config");
    for (int i = 0; i < 60; i++) printf("-");
    printf("\n");

    for (int idx : sorted_idx) {
        double err_pct = results[idx].avg_metrics.dot_rel_err * 100.0;
        double ratio   = (q8_err > 1e-10) ? err_pct / q8_err : 0.0;
        const char * marker = "";
        if (idx == cheapest_within_5pct) marker = " *** CHEAPEST ***";
        if (idx == n_configs) marker = " (reference)";

        printf("%-20s  %8.3f   %10.4f%%   %5.2fx%s\n",
               results[idx].name,
               results[idx].total_bpv,
               err_pct, ratio, marker);
    }

    printf("\n");

    // =====================================================================
    // Key findings
    // =====================================================================
    printf("================================================================\n");
    printf("  KEY FINDINGS\n");
    printf("================================================================\n\n");

    printf("  q8_0 reference: %.3f bpv, %.4f%% dot error\n\n",
           results[n_configs].total_bpv, q8_err);

    printf("  Target: within 5%% of q8_0 quality = %.4f%% max dot error\n\n",
           q8_err * 1.05);

    if (cheapest_within_5pct >= 0) {
        printf("  >>> CHEAPEST config within target: %s\n",
               results[cheapest_within_5pct].name);
        printf("      lo_bytes = %d, total_bpv = %.3f\n",
               results[cheapest_within_5pct].lo_bytes,
               results[cheapest_within_5pct].total_bpv);
        printf("      dot_rel_err = %.4f%% (%.2fx q8_0)\n",
               results[cheapest_within_5pct].avg_metrics.dot_rel_err * 100.0,
               results[cheapest_within_5pct].avg_metrics.dot_rel_err * 100.0 / q8_err);
        printf("      savings vs q8_0 = %.1f%% fewer bits\n",
               (1.0 - results[cheapest_within_5pct].total_bpv / 8.5) * 100.0);
    } else {
        printf("  >>> No config reached within 5%% of q8_0 quality.\n");
        printf("      Closest configs (in order of error):\n");

        // Sort by error ascending and show top 3
        std::vector<int> err_sorted(n_configs);
        std::iota(err_sorted.begin(), err_sorted.end(), 0);
        std::sort(err_sorted.begin(), err_sorted.end(),
                  [&](int a, int b) {
                      return results[a].avg_metrics.dot_rel_err < results[b].avg_metrics.dot_rel_err;
                  });
        for (int i = 0; i < std::min(3, n_configs); i++) {
            int idx = err_sorted[i];
            printf("      %d. %s: %.4f%% (%.2fx q8_0) at %.3f bpv\n",
                   i + 1, results[idx].name,
                   results[idx].avg_metrics.dot_rel_err * 100.0,
                   results[idx].avg_metrics.dot_rel_err * 100.0 / q8_err,
                   results[idx].total_bpv);
        }
    }

    printf("\n");

    // Show the "current" config for comparison
    printf("  Current config (3-bit MSE lo):\n");
    printf("      lo_bytes = %d, total_bpv = %.3f\n",
           results[1].lo_bytes, results[1].total_bpv);
    printf("      dot_rel_err = %.4f%% (%.2fx q8_0)\n",
           results[1].avg_metrics.dot_rel_err * 100.0,
           results[1].avg_metrics.dot_rel_err * 100.0 / q8_err);

    printf("\n");

    return 0;
}
