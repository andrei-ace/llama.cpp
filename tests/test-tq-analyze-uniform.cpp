// TurboQuant Uniform Layer Analysis — real variance profiles from Qwen2.5 7B
// Analyzes quantization quality on layers with <53% outlier variance concentration.
// These layers have nearly uniform variance distribution across channels,
// so the hi/lo split approach may not help. Key question: minimum bpv for <1% dot error?
//
// No model inference, no ggml dependencies — fully self-contained.
// Reads per-channel variance from /tmp/qwen25_stats.csv.
//
// Build: cmake --build build -t test-tq-analyze-uniform -j14
// Run:   ./build/bin/test-tq-analyze-uniform

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

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

static constexpr int DIM         = 128;
static constexpr int N_HI        = 32;   // outlier channels for split types
static constexpr int N_LO        = 96;   // regular channels for split types
static constexpr int N_VEC       = 500;
static constexpr int N_DOT_PAIRS = 50000;
static constexpr float SQRT_PI_OVER_2 = 1.2533141373155003f;

// Uniform layers from Qwen2.5 7B: layer(outlier_var%)
// layer 5 (49%), layer 9 (50%), layer 10 (51%), layer 11 (50%), layer 8 (52%)
static constexpr int TARGET_LAYERS[] = { 5, 9, 10, 11, 8 };
static constexpr int N_TARGET_LAYERS = 5;
static constexpr const char * TARGET_LABELS[] = {
    "L5  (49%)", "L9  (50%)", "L10 (51%)", "L11 (50%)", "L8  (52%)"
};

// ---------------------------------------------------------------------------
// Lloyd-Max centroids
// ---------------------------------------------------------------------------

// d=128 (full-dim FWHT)
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

// d=32 Lloyd-Max centroids (for hi split path)
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

// d=96 Lloyd-Max centroids (for lo split path)
static const float c8_d96[8] = {
    -0.2169f, -0.1362f, -0.0768f, -0.0249f, 0.0249f, 0.0768f, 0.1362f, 0.2169f
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

// Generate centroids for arbitrary dim/n_centroids via quantile spacing
static void generate_centroids(float * out, int n_centroids, int dim) {
    float sigma = 1.0f / sqrtf((float)dim);
    for (int i = 0; i < n_centroids; i++) {
        float p = ((float)i + 0.5f) / (float)n_centroids;
        out[i] = sigma * approx_inv_normal(p);
    }
}

// Storage for generated centroids
static float c32_d128[32];   // 5-bit for d=128
static float c256_d128[256]; // 8-bit for d=128
static float c256_d32[256];  // 8-bit for d=32

static void init_generated_centroids() {
    generate_centroids(c32_d128, 32, 128);
    generate_centroids(c256_d128, 256, 128);
    generate_centroids(c256_d32, 256, 32);
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
// Structured rotation for lo path (3x FWHT-32 with cross-block mix)
// ---------------------------------------------------------------------------

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
// Nearest centroid lookup
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
// CSV parsing: read per-channel variance for a given layer
// Returns variance[head][channel] sorted by variance desc within each head
// Also returns the outlier mask (is_outlier column)
// ---------------------------------------------------------------------------

struct ChannelInfo {
    int   channel;
    float variance;
    bool  is_outlier;
};

struct HeadData {
    std::vector<ChannelInfo> channels;  // all 128 channels
    std::vector<int> outlier_indices;   // channel indices marked as outlier
    std::vector<int> regular_indices;   // channel indices not outlier
};

struct LayerData {
    int layer;
    std::vector<HeadData> heads;
    float outlier_var_pct;  // % of total variance in outlier channels
};

static bool load_layer_data(const char * csv_path, int target_layer, LayerData & out) {
    FILE * f = fopen(csv_path, "r");
    if (!f) {
        fprintf(stderr, "ERROR: cannot open %s\n", csv_path);
        return false;
    }

    out.layer = target_layer;
    out.heads.clear();

    char line[512];
    // Skip header
    if (!fgets(line, sizeof(line), f)) { fclose(f); return false; }

    double total_var = 0.0, outlier_var = 0.0;

    while (fgets(line, sizeof(line), f)) {
        int layer, head, channel, rank, is_outlier;
        float mean_abs, variance, std_val, importance;
        if (sscanf(line, "%d,%d,%d,%f,%f,%f,%f,%d,%d",
                   &layer, &head, &channel, &mean_abs, &variance,
                   &std_val, &importance, &rank, &is_outlier) != 9) {
            continue;
        }
        if (layer != target_layer) continue;

        // Ensure we have enough heads
        while ((int)out.heads.size() <= head) {
            out.heads.push_back({});
        }

        ChannelInfo ci;
        ci.channel    = channel;
        ci.variance   = variance;
        ci.is_outlier = (is_outlier != 0);
        out.heads[head].channels.push_back(ci);

        total_var += variance;
        if (ci.is_outlier) {
            outlier_var += variance;
            out.heads[head].outlier_indices.push_back(channel);
        } else {
            out.heads[head].regular_indices.push_back(channel);
        }
    }

    fclose(f);

    if (out.heads.empty()) return false;

    out.outlier_var_pct = (total_var > 0) ? (float)(outlier_var / total_var * 100.0) : 0.0f;

    // Sort outlier/regular indices within each head by variance descending
    for (auto & hd : out.heads) {
        std::sort(hd.outlier_indices.begin(), hd.outlier_indices.end(),
            [&](int a, int b) {
                float va = 0, vb = 0;
                for (auto & c : hd.channels) {
                    if (c.channel == a) va = c.variance;
                    if (c.channel == b) vb = c.variance;
                }
                return va > vb;
            });
        std::sort(hd.regular_indices.begin(), hd.regular_indices.end(),
            [&](int a, int b) {
                float va = 0, vb = 0;
                for (auto & c : hd.channels) {
                    if (c.channel == a) va = c.variance;
                    if (c.channel == b) vb = c.variance;
                }
                return va > vb;
            });
    }

    return true;
}

// ---------------------------------------------------------------------------
// Generate realistic K vectors from per-channel variance profile
// ---------------------------------------------------------------------------

static void generate_vectors_from_profile(
    std::vector<std::vector<float>> & vecs,
    const HeadData & hd, int count, std::mt19937 & rng
) {
    vecs.resize(count);
    for (int i = 0; i < count; i++) {
        vecs[i].resize(DIM);
        for (int ch = 0; ch < DIM && ch < (int)hd.channels.size(); ch++) {
            float sigma = sqrtf(hd.channels[ch].variance);
            std::normal_distribution<float> dist(0.0f, sigma);
            vecs[i][hd.channels[ch].channel] = dist(rng);
        }
    }
}

// ---------------------------------------------------------------------------
// Quality metrics
// ---------------------------------------------------------------------------

struct Metrics {
    double dot_rel_error;   // |q.k - q.k_hat| / |q.k| averaged
    double dot_bias;        // (q.k_hat - q.k) / |q.k| averaged (signed)
    double cosine_sim;
    double rel_l2;
};

static Metrics compute_metrics(
    const std::vector<std::vector<float>> & K,
    const std::vector<std::vector<float>> & K_hat,
    const std::vector<std::vector<float>> & Q
) {
    int dim = (int)K[0].size();
    int nk = (int)K.size();
    int nq = (int)Q.size();

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

    double dot_err_sum = 0.0, dot_bias_sum = 0.0;
    int dot_count = 0;
    std::mt19937 rng(12345);
    for (int p = 0; p < N_DOT_PAIRS; p++) {
        int qi = rng() % nq;
        int ki = rng() % nk;
        double dot_true = 0.0, dot_quant = 0.0;
        for (int d = 0; d < dim; d++) {
            dot_true  += (double)Q[qi][d] * (double)K[ki][d];
            dot_quant += (double)Q[qi][d] * (double)K_hat[ki][d];
        }
        if (fabs(dot_true) > 1e-10) {
            dot_err_sum  += fabs(dot_true - dot_quant) / fabs(dot_true);
            dot_bias_sum += (dot_quant - dot_true) / fabs(dot_true);
            dot_count++;
        }
    }

    Metrics m;
    m.dot_rel_error = (dot_count > 0) ? dot_err_sum / dot_count : 0.0;
    m.dot_bias      = (dot_count > 0) ? dot_bias_sum / dot_count : 0.0;
    m.cosine_sim    = cos_sum / nk;
    m.rel_l2        = l2_sum / nk;
    return m;
}

// Compute metrics for a sub-range of dimensions (for hi/lo breakdown)
static Metrics compute_metrics_range(
    const std::vector<std::vector<float>> & K,
    const std::vector<std::vector<float>> & K_hat,
    const std::vector<std::vector<float>> & Q,
    int d_start, int d_end
) {
    int nk = (int)K.size();
    int nq = (int)Q.size();

    double cos_sum = 0.0, l2_sum = 0.0;
    for (int i = 0; i < nk; i++) {
        double dot_kk = 0, dot_kh = 0, dot_hh = 0, diff2 = 0, knorm2 = 0;
        for (int d = d_start; d < d_end; d++) {
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

    double dot_err_sum = 0.0, dot_bias_sum = 0.0;
    int dot_count = 0;
    std::mt19937 rng(54321);
    int n_pairs = N_DOT_PAIRS / 2;
    for (int p = 0; p < n_pairs; p++) {
        int qi = rng() % nq;
        int ki = rng() % nk;
        double dot_true = 0.0, dot_quant = 0.0;
        for (int d = d_start; d < d_end; d++) {
            dot_true  += (double)Q[qi][d] * (double)K[ki][d];
            dot_quant += (double)Q[qi][d] * (double)K_hat[ki][d];
        }
        if (fabs(dot_true) > 1e-10) {
            dot_err_sum  += fabs(dot_true - dot_quant) / fabs(dot_true);
            dot_bias_sum += (dot_quant - dot_true) / fabs(dot_true);
            dot_count++;
        }
    }

    Metrics m;
    m.dot_rel_error = (dot_count > 0) ? dot_err_sum / dot_count : 0.0;
    m.dot_bias      = (dot_count > 0) ? dot_bias_sum / dot_count : 0.0;
    m.cosine_sim    = cos_sum / nk;
    m.rel_l2        = l2_sum / nk;
    return m;
}

// ---------------------------------------------------------------------------
// Quantizers — Uniform (no split)
// ---------------------------------------------------------------------------

// q4_0: block-of-32, 4-bit unsigned + fp16 scale = 4.50 bpv
static void quant_q4_0(
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

// q5_0: block-of-32, 5-bit unsigned + fp16 scale = 5.50 bpv
static void quant_q5_0(
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
            float d = amax / 15.0f;
            float id = (d > 1e-30f) ? 1.0f / d : 0.0f;
            for (int j = 0; j < bsz; j++) {
                int q = (int)roundf(K[i][b + j] * id + 16.0f);
                if (q < 0)  q = 0;
                if (q > 31) q = 31;
                K_hat[i][b + j] = ((float)q - 16.0f) * d;
            }
        }
    }
}

// q8_0: block-of-32, 8-bit signed + fp16 scale = 8.50 bpv
static void quant_q8_0(
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

// tqk4_0: FWHT-128 + 4-bit MSE = 4.13 bpv
// (norm(2) + 128*4bit(64)) * 8 / 128 = 66*8/128 = 4.125
static void quant_tqk4_0(
    const std::vector<std::vector<float>> & K,
    std::vector<std::vector<float>> & K_hat
) {
    int nk = (int)K.size();
    K_hat.resize(nk);
    for (int i = 0; i < nk; i++) {
        K_hat[i].resize(DIM);
        float tmp[DIM];
        memcpy(tmp, K[i].data(), DIM * sizeof(float));

        float norm = 0.0f;
        for (int j = 0; j < DIM; j++) norm += tmp[j] * tmp[j];
        norm = sqrtf(norm);
        if (norm < 1e-30f) { memset(K_hat[i].data(), 0, DIM * sizeof(float)); continue; }

        float inv = 1.0f / norm;
        for (int j = 0; j < DIM; j++) tmp[j] *= inv;

        fwht(tmp, DIM);

        for (int j = 0; j < DIM; j++) {
            int idx = nearest_centroid(tmp[j], c16_d128, 16);
            tmp[j] = c16_d128[idx];
        }

        fwht(tmp, DIM);

        for (int j = 0; j < DIM; j++) K_hat[i][j] = tmp[j] * norm;
    }
}

// ---------------------------------------------------------------------------
// Quantizers — Split types (reorder channels: outliers first)
// ---------------------------------------------------------------------------

// Reorder a vector so outlier channels come first (hi), rest follow (lo)
static void reorder_to_split(
    const float * in, float * out,
    const std::vector<int> & outlier_idx,
    const std::vector<int> & regular_idx
) {
    int pos = 0;
    for (int ch : outlier_idx) out[pos++] = in[ch];
    for (int ch : regular_idx) out[pos++] = in[ch];
}

// Undo reorder: from split back to original channel order
static void unreorder_from_split(
    const float * in, float * out,
    const std::vector<int> & outlier_idx,
    const std::vector<int> & regular_idx
) {
    int pos = 0;
    for (int ch : outlier_idx) out[ch] = in[pos++];
    for (int ch : regular_idx) out[ch] = in[pos++];
}

// tqk4_sj: split + 5-bit MSE hi + QJL, 3-bit MSE lo = 4.13 bpv
// hi: norm(2) + 32*5bit(20) + QJL signs(4) + QJL norm(2) = 28B
// lo: norm(2) + 96*3bit(36) = 38B
// total: 66B * 8 / 128 = 4.125 bpv
static void quant_tqk4_sj(
    const std::vector<std::vector<float>> & K,
    std::vector<std::vector<float>> & K_hat,
    const HeadData & hd
) {
    int nk = (int)K.size();
    int n_hi = (int)hd.outlier_indices.size();
    int n_lo = (int)hd.regular_indices.size();

    K_hat.resize(nk);
    for (int i = 0; i < nk; i++) {
        K_hat[i].resize(DIM);

        // Reorder to split layout
        float split[DIM];
        reorder_to_split(K[i].data(), split, hd.outlier_indices, hd.regular_indices);

        float * hi = split;
        float * lo = split + n_hi;

        // --- Hi path: 5-bit MSE + FWHT-32 + QJL ---
        float hi_recon[N_HI];
        {
            float tmp[N_HI];
            memcpy(tmp, hi, n_hi * sizeof(float));

            float norm = 0.0f;
            for (int j = 0; j < n_hi; j++) norm += tmp[j] * tmp[j];
            norm = sqrtf(norm);

            if (norm > 1e-30f) {
                float inv = 1.0f / norm;
                for (int j = 0; j < n_hi; j++) tmp[j] *= inv;
                fwht(tmp, n_hi);
                for (int j = 0; j < n_hi; j++) {
                    int idx = nearest_centroid(tmp[j], c32_d32, 32);
                    tmp[j] = c32_d32[idx];
                }
                fwht(tmp, n_hi);
                for (int j = 0; j < n_hi; j++) hi_recon[j] = tmp[j] * norm;
            } else {
                memset(hi_recon, 0, n_hi * sizeof(float));
            }

            // QJL correction on residual
            float residual[N_HI];
            for (int j = 0; j < n_hi; j++) residual[j] = hi[j] - hi_recon[j];

            float rnorm = 0.0f;
            for (int j = 0; j < n_hi; j++) rnorm += residual[j] * residual[j];
            rnorm = sqrtf(rnorm);

            if (rnorm > 1e-30f) {
                float r[N_HI];
                memcpy(r, residual, n_hi * sizeof(float));
                fwht(r, n_hi);

                float corr[N_HI];
                for (int j = 0; j < n_hi; j++) {
                    corr[j] = (r[j] >= 0.0f) ? 1.0f : -1.0f;
                }
                fwht(corr, n_hi);

                float scale = SQRT_PI_OVER_2 / (float)n_hi * rnorm;
                for (int j = 0; j < n_hi; j++) hi_recon[j] += scale * corr[j];
            }
        }

        // --- Lo path: 3-bit MSE + structured rotation ---
        float lo_recon[N_LO];
        {
            float tmp[N_LO];
            memcpy(tmp, lo, n_lo * sizeof(float));

            float norm = 0.0f;
            for (int j = 0; j < n_lo; j++) norm += tmp[j] * tmp[j];
            norm = sqrtf(norm);

            if (norm > 1e-30f) {
                float inv = 1.0f / norm;
                for (int j = 0; j < n_lo; j++) tmp[j] *= inv;

                structured_rotate_lo(tmp, n_lo);

                for (int j = 0; j < n_lo; j++) {
                    int idx = nearest_centroid(tmp[j], c8_d96, 8);
                    tmp[j] = c8_d96[idx];
                }

                structured_unrotate_lo(tmp, n_lo);

                for (int j = 0; j < n_lo; j++) lo_recon[j] = tmp[j] * norm;
            } else {
                memset(lo_recon, 0, n_lo * sizeof(float));
            }
        }

        // Reassemble in split order, then unreorder
        float recon_split[DIM];
        memcpy(recon_split, hi_recon, n_hi * sizeof(float));
        memcpy(recon_split + n_hi, lo_recon, n_lo * sizeof(float));
        unreorder_from_split(recon_split, K_hat[i].data(), hd.outlier_indices, hd.regular_indices);
    }
}

// tqk5r3_sj: split + residual 5-bit MSE hi + 3-bit MSE lo = 4.63 bpv
// hi: norm1(2) + 32*5bit(20) + norm2(2) + 32*3bit(12) = 36B
// lo: norm(2) + 96*3bit(36) = 38B
// total: 74B * 8 / 128 = 4.625 bpv
static void quant_tqk5r3_sj(
    const std::vector<std::vector<float>> & K,
    std::vector<std::vector<float>> & K_hat,
    const HeadData & hd
) {
    int nk = (int)K.size();
    int n_hi = (int)hd.outlier_indices.size();
    int n_lo = (int)hd.regular_indices.size();

    K_hat.resize(nk);
    for (int i = 0; i < nk; i++) {
        K_hat[i].resize(DIM);

        float split[DIM];
        reorder_to_split(K[i].data(), split, hd.outlier_indices, hd.regular_indices);

        float * hi = split;
        float * lo = split + n_hi;

        // --- Hi path: 5-bit MSE pass 1, then 3-bit MSE pass 2 on residual ---
        float hi_recon[N_HI];
        {
            float tmp[N_HI];
            memcpy(tmp, hi, n_hi * sizeof(float));

            // Pass 1: 5-bit MSE
            float norm1 = 0.0f;
            for (int j = 0; j < n_hi; j++) norm1 += tmp[j] * tmp[j];
            norm1 = sqrtf(norm1);

            float pass1[N_HI];
            if (norm1 > 1e-30f) {
                float inv = 1.0f / norm1;
                for (int j = 0; j < n_hi; j++) pass1[j] = tmp[j] * inv;
                fwht(pass1, n_hi);
                for (int j = 0; j < n_hi; j++) {
                    int idx = nearest_centroid(pass1[j], c32_d32, 32);
                    pass1[j] = c32_d32[idx];
                }
                fwht(pass1, n_hi);
                for (int j = 0; j < n_hi; j++) pass1[j] *= norm1;
            } else {
                memset(pass1, 0, n_hi * sizeof(float));
            }

            // Residual
            float residual[N_HI];
            for (int j = 0; j < n_hi; j++) residual[j] = hi[j] - pass1[j];

            // Pass 2: 3-bit MSE on residual
            float norm2 = 0.0f;
            for (int j = 0; j < n_hi; j++) norm2 += residual[j] * residual[j];
            norm2 = sqrtf(norm2);

            float pass2[N_HI];
            if (norm2 > 1e-30f) {
                float inv = 1.0f / norm2;
                for (int j = 0; j < n_hi; j++) pass2[j] = residual[j] * inv;
                fwht(pass2, n_hi);
                for (int j = 0; j < n_hi; j++) {
                    int idx = nearest_centroid(pass2[j], c8_d32, 8);
                    pass2[j] = c8_d32[idx];
                }
                fwht(pass2, n_hi);
                for (int j = 0; j < n_hi; j++) pass2[j] *= norm2;
            } else {
                memset(pass2, 0, n_hi * sizeof(float));
            }

            for (int j = 0; j < n_hi; j++) hi_recon[j] = pass1[j] + pass2[j];
        }

        // --- Lo path: 3-bit MSE + structured rotation (same as tqk4_sj) ---
        float lo_recon[N_LO];
        {
            float tmp[N_LO];
            memcpy(tmp, lo, n_lo * sizeof(float));

            float norm = 0.0f;
            for (int j = 0; j < n_lo; j++) norm += tmp[j] * tmp[j];
            norm = sqrtf(norm);

            if (norm > 1e-30f) {
                float inv = 1.0f / norm;
                for (int j = 0; j < n_lo; j++) tmp[j] *= inv;
                structured_rotate_lo(tmp, n_lo);
                for (int j = 0; j < n_lo; j++) {
                    int idx = nearest_centroid(tmp[j], c8_d96, 8);
                    tmp[j] = c8_d96[idx];
                }
                structured_unrotate_lo(tmp, n_lo);
                for (int j = 0; j < n_lo; j++) lo_recon[j] = tmp[j] * norm;
            } else {
                memset(lo_recon, 0, n_lo * sizeof(float));
            }
        }

        float recon_split[DIM];
        memcpy(recon_split, hi_recon, n_hi * sizeof(float));
        memcpy(recon_split + n_hi, lo_recon, n_lo * sizeof(float));
        unreorder_from_split(recon_split, K_hat[i].data(), hd.outlier_indices, hd.regular_indices);
    }
}

// ---------------------------------------------------------------------------
// Quantizers — Minimal (full-dim, no split)
// ---------------------------------------------------------------------------

// 2-bit MSE full: FWHT-128 + 2-bit = 2.13 bpv
// (norm(2) + 128*2bit(32)) * 8 / 128 = 34*8/128 = 2.125
static void quant_2bit_mse(
    const std::vector<std::vector<float>> & K,
    std::vector<std::vector<float>> & K_hat
) {
    int nk = (int)K.size();
    K_hat.resize(nk);
    for (int i = 0; i < nk; i++) {
        K_hat[i].resize(DIM);
        float tmp[DIM];
        memcpy(tmp, K[i].data(), DIM * sizeof(float));

        float norm = 0.0f;
        for (int j = 0; j < DIM; j++) norm += tmp[j] * tmp[j];
        norm = sqrtf(norm);
        if (norm < 1e-30f) { memset(K_hat[i].data(), 0, DIM * sizeof(float)); continue; }

        float inv = 1.0f / norm;
        for (int j = 0; j < DIM; j++) tmp[j] *= inv;
        fwht(tmp, DIM);
        for (int j = 0; j < DIM; j++) {
            int idx = nearest_centroid(tmp[j], c4_d128, 4);
            tmp[j] = c4_d128[idx];
        }
        fwht(tmp, DIM);
        for (int j = 0; j < DIM; j++) K_hat[i][j] = tmp[j] * norm;
    }
}

// 3-bit MSE full: FWHT-128 + 3-bit = 3.13 bpv
// (norm(2) + 128*3bit(48)) * 8 / 128 = 50*8/128 = 3.125
static void quant_3bit_mse(
    const std::vector<std::vector<float>> & K,
    std::vector<std::vector<float>> & K_hat
) {
    int nk = (int)K.size();
    K_hat.resize(nk);
    for (int i = 0; i < nk; i++) {
        K_hat[i].resize(DIM);
        float tmp[DIM];
        memcpy(tmp, K[i].data(), DIM * sizeof(float));

        float norm = 0.0f;
        for (int j = 0; j < DIM; j++) norm += tmp[j] * tmp[j];
        norm = sqrtf(norm);
        if (norm < 1e-30f) { memset(K_hat[i].data(), 0, DIM * sizeof(float)); continue; }

        float inv = 1.0f / norm;
        for (int j = 0; j < DIM; j++) tmp[j] *= inv;
        fwht(tmp, DIM);
        for (int j = 0; j < DIM; j++) {
            int idx = nearest_centroid(tmp[j], c8_d128, 8);
            tmp[j] = c8_d128[idx];
        }
        fwht(tmp, DIM);
        for (int j = 0; j < DIM; j++) K_hat[i][j] = tmp[j] * norm;
    }
}

// ---------------------------------------------------------------------------
// Architecture descriptor
// ---------------------------------------------------------------------------

enum ArchType {
    ARCH_Q4_0,
    ARCH_Q5_0,
    ARCH_Q8_0,
    ARCH_TQK4_0,
    ARCH_TQK4_SJ,
    ARCH_TQK5R3_SJ,
    ARCH_2BIT_MSE,
    ARCH_3BIT_MSE,
};

struct Architecture {
    const char * name;
    const char * desc;
    float        bpv;
    ArchType     type;
    bool         is_split;  // needs hi/lo breakdown
};

static Architecture archs[] = {
    // --- Uniform (no split) ---
    { "q4_0",       "uniform 4-bit blocks of 32",           4.50f, ARCH_Q4_0,      false },
    { "q5_0",       "uniform 5-bit blocks of 32",           5.50f, ARCH_Q5_0,      false },
    { "q8_0",       "uniform 8-bit blocks of 32",           8.50f, ARCH_Q8_0,      false },
    { "tqk4_0",     "FWHT-128 + 4-bit MSE",                4.13f, ARCH_TQK4_0,    false },

    // --- Split types ---
    { "tqk4_sj",    "split + 5-bit hi + QJL, 3-bit lo",    4.13f, ARCH_TQK4_SJ,   true },
    { "tqk5r3_sj",  "split + residual hi, 3-bit lo",       4.63f, ARCH_TQK5R3_SJ, true },

    // --- Minimal ---
    { "2bit_mse",   "FWHT-128 + 2-bit MSE",                2.13f, ARCH_2BIT_MSE,  false },
    { "3bit_mse",   "FWHT-128 + 3-bit MSE",                3.13f, ARCH_3BIT_MSE,  false },
};
static constexpr int N_ARCHS = sizeof(archs) / sizeof(archs[0]);

// ---------------------------------------------------------------------------
// Run a quantizer
// ---------------------------------------------------------------------------

static void run_quantizer(
    ArchType type,
    const std::vector<std::vector<float>> & K,
    std::vector<std::vector<float>> & K_hat,
    const HeadData & hd
) {
    switch (type) {
        case ARCH_Q4_0:      quant_q4_0(K, K_hat);           break;
        case ARCH_Q5_0:      quant_q5_0(K, K_hat);           break;
        case ARCH_Q8_0:      quant_q8_0(K, K_hat);           break;
        case ARCH_TQK4_0:    quant_tqk4_0(K, K_hat);         break;
        case ARCH_TQK4_SJ:   quant_tqk4_sj(K, K_hat, hd);   break;
        case ARCH_TQK5R3_SJ: quant_tqk5r3_sj(K, K_hat, hd); break;
        case ARCH_2BIT_MSE:  quant_2bit_mse(K, K_hat);       break;
        case ARCH_3BIT_MSE:  quant_3bit_mse(K, K_hat);       break;
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    init_generated_centroids();

    const char * csv_path = "/tmp/qwen25_stats.csv";

    printf("================================================================\n");
    printf("  TurboQuant Uniform Layer Analysis\n");
    printf("  Real variance profiles from Qwen2.5 7B\n");
    printf("  Layers with <53%% outlier variance (nearly uniform)\n");
    printf("  Key question: minimum bpv for <1%% dot product error?\n");
    printf("================================================================\n\n");

    printf("Architectures tested:\n");
    printf("  %-12s  %5s  %-40s  %s\n", "Name", "BPV", "Description", "Type");
    for (int i = 0; i < 75; i++) printf("-");
    printf("\n");
    for (int a = 0; a < N_ARCHS; a++) {
        printf("  %-12s  %5.2f  %-40s  %s\n",
               archs[a].name, archs[a].bpv, archs[a].desc,
               archs[a].is_split ? "split" : "uniform");
    }
    printf("\n");

    // Store results for summary: [layer_idx][arch_idx]
    struct Result {
        Metrics full;
        Metrics hi;   // only for split types
        Metrics lo;
    };
    std::vector<std::vector<Result>> all_results(N_TARGET_LAYERS);

    for (int li = 0; li < N_TARGET_LAYERS; li++) {
        int layer = TARGET_LAYERS[li];
        LayerData ld;
        if (!load_layer_data(csv_path, layer, ld)) {
            fprintf(stderr, "ERROR: failed to load layer %d from CSV\n", layer);
            continue;
        }

        printf("================================================================\n");
        printf("  Layer %d: %s   (%d heads, outlier_var=%.1f%%)\n",
               layer, TARGET_LABELS[li], (int)ld.heads.size(), ld.outlier_var_pct);
        printf("================================================================\n\n");

        // Use head 0 as representative (all heads share similar profile in uniform layers)
        HeadData & hd = ld.heads[0];

        printf("  Head 0: %d outlier channels, %d regular channels\n",
               (int)hd.outlier_indices.size(), (int)hd.regular_indices.size());

        // Show variance distribution summary
        double total_var = 0, hi_var = 0;
        for (auto & c : hd.channels) {
            total_var += c.variance;
            if (c.is_outlier) hi_var += c.variance;
        }
        printf("  Head 0 outlier var%%: %.1f%%\n\n", hi_var / total_var * 100.0);

        // Generate K and Q vectors using real per-channel variance
        std::mt19937 rng(42 + layer);
        std::vector<std::vector<float>> K, Q;
        generate_vectors_from_profile(K, hd, N_VEC, rng);
        generate_vectors_from_profile(Q, hd, N_VEC, rng);

        all_results[li].resize(N_ARCHS);

        for (int a = 0; a < N_ARCHS; a++) {
            std::vector<std::vector<float>> K_hat;
            run_quantizer(archs[a].type, K, K_hat, hd);

            Metrics mf = compute_metrics(K, K_hat, Q);
            all_results[li][a].full = mf;

            printf("  %-12s (%5.2f bpv): dotErr=%7.4f%%  bias=%+7.4f%%  cos=%.6f  relL2=%.4f",
                   archs[a].name, archs[a].bpv,
                   mf.dot_rel_error * 100.0, mf.dot_bias * 100.0,
                   mf.cosine_sim, mf.rel_l2);

            if (archs[a].is_split) {
                // For split types, also show hi/lo breakdown
                // We need to compute metrics on the reordered vectors
                // Build reordered K, K_hat, Q for range-based metrics
                int n_hi = (int)hd.outlier_indices.size();
                int n_lo = (int)hd.regular_indices.size();

                // Reorder all vectors to split layout for sub-range metrics
                std::vector<std::vector<float>> K_split(N_VEC), K_hat_split(N_VEC), Q_split(N_VEC);
                for (int v = 0; v < N_VEC; v++) {
                    K_split[v].resize(DIM);
                    K_hat_split[v].resize(DIM);
                    Q_split[v].resize(DIM);
                    reorder_to_split(K[v].data(), K_split[v].data(),
                                     hd.outlier_indices, hd.regular_indices);
                    reorder_to_split(K_hat[v].data(), K_hat_split[v].data(),
                                     hd.outlier_indices, hd.regular_indices);
                    reorder_to_split(Q[v].data(), Q_split[v].data(),
                                     hd.outlier_indices, hd.regular_indices);
                }

                Metrics mhi = compute_metrics_range(K_split, K_hat_split, Q_split, 0, n_hi);
                Metrics mlo = compute_metrics_range(K_split, K_hat_split, Q_split, n_hi, n_hi + n_lo);
                all_results[li][a].hi = mhi;
                all_results[li][a].lo = mlo;

                printf("\n%46s hi: dotErr=%7.4f%%  lo: dotErr=%7.4f%%", "",
                       mhi.dot_rel_error * 100.0, mlo.dot_rel_error * 100.0);
            }

            printf("\n");
        }
        printf("\n");
    }

    // =====================================================================
    // Summary 1: Dot product relative error across all layers
    // =====================================================================
    printf("================================================================\n");
    printf("  SUMMARY: Dot Product Relative Error %% (lower = better)\n");
    printf("================================================================\n\n");

    printf("%-12s  %5s", "Arch", "BPV");
    for (int li = 0; li < N_TARGET_LAYERS; li++) {
        printf("  %10s", TARGET_LABELS[li]);
    }
    printf("      AVG\n");
    for (int i = 0; i < 12 + 6 + N_TARGET_LAYERS * 12 + 10; i++) printf("-");
    printf("\n");

    for (int a = 0; a < N_ARCHS; a++) {
        printf("%-12s  %5.2f", archs[a].name, archs[a].bpv);
        double sum = 0.0;
        int cnt = 0;
        for (int li = 0; li < N_TARGET_LAYERS; li++) {
            double v = all_results[li][a].full.dot_rel_error * 100.0;
            printf("  %9.4f%%", v);
            sum += v;
            cnt++;
        }
        printf("  %9.4f%%\n", sum / cnt);
    }
    printf("\n");

    // =====================================================================
    // Summary 2: Dot product bias across all layers
    // =====================================================================
    printf("================================================================\n");
    printf("  SUMMARY: Dot Product Bias %% (closer to 0 = better)\n");
    printf("================================================================\n\n");

    printf("%-12s  %5s", "Arch", "BPV");
    for (int li = 0; li < N_TARGET_LAYERS; li++) {
        printf("  %10s", TARGET_LABELS[li]);
    }
    printf("      AVG\n");
    for (int i = 0; i < 12 + 6 + N_TARGET_LAYERS * 12 + 10; i++) printf("-");
    printf("\n");

    for (int a = 0; a < N_ARCHS; a++) {
        printf("%-12s  %5.2f", archs[a].name, archs[a].bpv);
        double sum = 0.0;
        int cnt = 0;
        for (int li = 0; li < N_TARGET_LAYERS; li++) {
            double v = all_results[li][a].full.dot_bias * 100.0;
            printf("  %+9.4f%%", v);
            sum += v;
            cnt++;
        }
        printf("  %+9.4f%%\n", sum / cnt);
    }
    printf("\n");

    // =====================================================================
    // Summary 3: Cosine similarity across all layers
    // =====================================================================
    printf("================================================================\n");
    printf("  SUMMARY: Cosine Similarity (higher = better)\n");
    printf("================================================================\n\n");

    printf("%-12s  %5s", "Arch", "BPV");
    for (int li = 0; li < N_TARGET_LAYERS; li++) {
        printf("  %10s", TARGET_LABELS[li]);
    }
    printf("      AVG\n");
    for (int i = 0; i < 12 + 6 + N_TARGET_LAYERS * 12 + 10; i++) printf("-");
    printf("\n");

    for (int a = 0; a < N_ARCHS; a++) {
        printf("%-12s  %5.2f", archs[a].name, archs[a].bpv);
        double sum = 0.0;
        int cnt = 0;
        for (int li = 0; li < N_TARGET_LAYERS; li++) {
            double v = all_results[li][a].full.cosine_sim;
            printf("  %10.6f", v);
            sum += v;
            cnt++;
        }
        printf("  %10.6f\n", sum / cnt);
    }
    printf("\n");

    // =====================================================================
    // Key analysis: minimum bpv for <1% dot error
    // =====================================================================
    printf("================================================================\n");
    printf("  KEY QUESTION: Minimum BPV for <1%% dot product error\n");
    printf("================================================================\n\n");

    // For each layer, find minimum bpv architecture that achieves <1% avg dot error
    printf("  %-12s  %-12s  %5s  %10s\n", "Layer", "Best arch", "BPV", "DotErr%%");
    for (int i = 0; i < 50; i++) printf("-");
    printf("\n");

    for (int li = 0; li < N_TARGET_LAYERS; li++) {
        float best_bpv = 999.0f;
        int best_a = -1;
        for (int a = 0; a < N_ARCHS; a++) {
            if (all_results[li][a].full.dot_rel_error < 0.01 &&
                archs[a].bpv < best_bpv) {
                best_bpv = archs[a].bpv;
                best_a = a;
            }
        }
        if (best_a >= 0) {
            printf("  %-12s  %-12s  %5.2f  %9.4f%%\n",
                   TARGET_LABELS[li], archs[best_a].name, archs[best_a].bpv,
                   all_results[li][best_a].full.dot_rel_error * 100.0);
        } else {
            printf("  %-12s  %-12s  %5s  %10s\n",
                   TARGET_LABELS[li], "(none <1%)", "N/A", "N/A");
        }
    }
    printf("\n");

    // =====================================================================
    // Can we go below 4 bpv?
    // =====================================================================
    printf("================================================================\n");
    printf("  CAN WE GO BELOW 4 BPV?  (Architectures < 4.0 bpv)\n");
    printf("================================================================\n\n");

    printf("  %-12s  %5s", "Arch", "BPV");
    for (int li = 0; li < N_TARGET_LAYERS; li++) {
        printf("  %10s", TARGET_LABELS[li]);
    }
    printf("      AVG\n");
    for (int i = 0; i < 12 + 6 + N_TARGET_LAYERS * 12 + 10; i++) printf("-");
    printf("\n");

    for (int a = 0; a < N_ARCHS; a++) {
        if (archs[a].bpv >= 4.0f) continue;
        printf("  %-12s  %5.2f", archs[a].name, archs[a].bpv);
        double sum = 0.0;
        int cnt = 0;
        for (int li = 0; li < N_TARGET_LAYERS; li++) {
            double v = all_results[li][a].full.dot_rel_error * 100.0;
            printf("  %9.4f%%", v);
            sum += v;
            cnt++;
        }
        printf("  %9.4f%%\n", sum / cnt);
    }
    printf("\n");

    // =====================================================================
    // Split vs Uniform comparison at similar bpv
    // =====================================================================
    printf("================================================================\n");
    printf("  SPLIT vs UNIFORM at ~4.1 bpv\n");
    printf("  tqk4_0 (4.13 bpv, uniform FWHT-128) vs tqk4_sj (4.13 bpv, split)\n");
    printf("================================================================\n\n");

    // Find indices
    int idx_tqk4_0 = -1, idx_tqk4_sj = -1;
    for (int a = 0; a < N_ARCHS; a++) {
        if (archs[a].type == ARCH_TQK4_0)  idx_tqk4_0  = a;
        if (archs[a].type == ARCH_TQK4_SJ) idx_tqk4_sj = a;
    }

    if (idx_tqk4_0 >= 0 && idx_tqk4_sj >= 0) {
        printf("  %-12s", "Layer");
        printf("  %12s  %12s  %12s\n",
               "tqk4_0 err%", "tqk4_sj err%", "Winner");
        for (int i = 0; i < 55; i++) printf("-");
        printf("\n");

        int uniform_wins = 0, split_wins = 0;
        for (int li = 0; li < N_TARGET_LAYERS; li++) {
            double e_uniform = all_results[li][idx_tqk4_0].full.dot_rel_error * 100.0;
            double e_split   = all_results[li][idx_tqk4_sj].full.dot_rel_error * 100.0;
            const char * winner = (e_uniform <= e_split) ? "UNIFORM" : "SPLIT";
            if (e_uniform <= e_split) uniform_wins++; else split_wins++;
            printf("  %-12s  %11.4f%%  %11.4f%%  %s\n",
                   TARGET_LABELS[li], e_uniform, e_split, winner);
        }
        printf("\n  Score: uniform=%d  split=%d\n", uniform_wins, split_wins);
        if (uniform_wins > split_wins) {
            printf("  --> For uniform layers, full-dim FWHT wins over split approach!\n");
        } else if (split_wins > uniform_wins) {
            printf("  --> Surprisingly, split still helps even for uniform layers.\n");
        } else {
            printf("  --> Tie: uniform and split perform comparably.\n");
        }
    }
    printf("\n");

    return 0;
}
