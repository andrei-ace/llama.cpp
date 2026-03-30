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
#include "ggml-cpu/quants.h"
#include "ggml-turbo-quant.h"

#include <math.h>
#include <string.h>
#include <assert.h>
#include <limits.h>

// ---------------------------------------------------------------------------
// Centroids: exact Lloyd-Max for Beta((d-1)/2, (d-1)/2)
// Each independent instance uses centroids for its own dimension d.
// ---------------------------------------------------------------------------

// Precomputed Lloyd-Max centroids for various (d, b) configurations.
// Some may be unused at current operating points but kept for future use.

// d=32 (outlier instance)
static const float centroids_16_d32[16] = {
    -0.4533721997f, -0.3498559145f, -0.2764914062f, -0.2161194569f,
    -0.1628573862f, -0.1138475776f, -0.0673934339f, -0.0223187462f,
     0.0223187462f,  0.0673934339f,  0.1138475776f,  0.1628573862f,
     0.2161194569f,  0.2764914062f,  0.3498559145f,  0.4533721997f,
};
static const float centroids_8_d32[8] = {
    -0.3662682422f, -0.2324605670f, -0.1317560968f, -0.0428515156f,
     0.0428515156f,  0.1317560968f,  0.2324605670f,  0.3662682422f,
};
static const float centroids_4_d32[4] = {
    -0.2633194113f, -0.0798019295f, 0.0798019295f, 0.2633194113f,
};
static const float centroids_2_d32[2] = {
    -0.1421534638f, 0.1421534638f,
};

// d=96 (regular instance)
static const float centroids_8_d96[8] = {
    -0.2168529349f, -0.1361685800f, -0.0767954958f, -0.0249236898f,
     0.0249236898f,  0.0767954958f,  0.1361685800f,  0.2168529349f,
};
static const float centroids_4_d96[4] = {
    -0.1534455138f, -0.0461670286f, 0.0461670286f, 0.1534455138f,
};
static const float centroids_2_d96[2] = {
    -0.0816460916f, 0.0816460916f,
};

// d=128 (used by MSE types which operate on full vector)
// d=128 centroids — exact Lloyd-Max for Beta(63.5, 63.5) computed with scipy
static const float centroids_16[16] = {
    -0.2376271868f, -0.1807937296f, -0.1417616544f, -0.1102470655f,
    -0.0827925668f, -0.0577445357f, -0.0341340283f, -0.0112964982f,
     0.0112964982f,  0.0341340283f,  0.0577445357f,  0.0827925668f,
     0.1102470655f,  0.1417616544f,  0.1807937296f,  0.2376271868f,
};
static const float centroids_8[8] = {
    -0.1883971860f, -0.1181397670f, -0.0665856080f, -0.0216043106f,
     0.0216043106f,  0.0665856080f,  0.1181397670f,  0.1883971860f,
};
static const float centroids_4[4] = {
    -0.1330415202f, -0.0399915952f, 0.0399915952f, 0.1330415202f,
};
static const float centroids_2[2] = {
    -0.0706615727f, 0.0706615727f,
};

// ---------------------------------------------------------------------------
// Two independent rotation matrices: Π_hi (32×32) and Π_lo (96×96)
// Per the paper: split channels into outlier/regular sets, apply independent
// TurboQuant instances to each.
// ---------------------------------------------------------------------------

#define TQ_DIM     128
#define TQ_DIM_HI  32
#define TQ_DIM_LO  96

static float tq_rot_hi_fwd[TQ_DIM_HI * TQ_DIM_HI];  // Π_hi (row-major) — K cache outlier subset
static float tq_rot_hi_inv[TQ_DIM_HI * TQ_DIM_HI];
static float tq_rot_lo_fwd[TQ_DIM_LO * TQ_DIM_LO];  // Π_lo (row-major) — K cache regular subset
static float tq_rot_lo_inv[TQ_DIM_LO * TQ_DIM_LO];
static float tq_rot_v_fwd[TQ_DIM * TQ_DIM];          // Π_v (128×128) — V cache, no split
static float tq_rot_v_inv[TQ_DIM * TQ_DIM];

// ---------------------------------------------------------------------------
// Per-layer-per-head outlier channel registry (populated during calibration)
// With GQA, different KV heads may have different outlier patterns.
// ---------------------------------------------------------------------------

#define TQ_MAX_LAYERS 256
#define TQ_MAX_HEADS   128

// Per-layer-per-head outlier/regular channel indices (K and V may differ)
static int   tq_k_outlier_reg[TQ_MAX_LAYERS][TQ_MAX_HEADS][TQ_DIM_HI];
static int   tq_k_regular_reg[TQ_MAX_LAYERS][TQ_MAX_HEADS][TQ_DIM_LO];
static int   tq_v_outlier_reg[TQ_MAX_LAYERS][TQ_MAX_HEADS][TQ_DIM_HI];
static int   tq_v_regular_reg[TQ_MAX_LAYERS][TQ_MAX_HEADS][TQ_DIM_LO];
static int   tq_layer_calibrated[TQ_MAX_LAYERS];

// Calibration accumulators — per layer, per head
static float tq_k_accum[TQ_MAX_LAYERS][TQ_MAX_HEADS][TQ_DIM];
static float tq_v_accum[TQ_MAX_LAYERS][TQ_MAX_HEADS][TQ_DIM];
static int   tq_k_accum_n[TQ_MAX_LAYERS]; // count of tokens (not blocks)
static int   tq_v_accum_n[TQ_MAX_LAYERS];
static int   tq_calibration_active = 1;

// Thread-local current context (set by CPU backend before quantize/dequantize/vec_dot)
static _Thread_local int tq_cur_layer = 0;
static _Thread_local int tq_cur_head  = 0;  // block index within row = KV head index
static _Thread_local int tq_cur_is_k  = 1;  // 1 = K cache, 0 = V cache

// Per-layer fp16 sink registry (set once after calibration, read by vec_dot)
static const void * tq_sink_k_base[TQ_MAX_LAYERS];      // start of TQ K tensor data per layer
static const void * tq_sink_fp16_base[TQ_MAX_LAYERS];    // start of fp16 sink tensor data per layer
static int64_t      tq_sink_k_stride[TQ_MAX_LAYERS];     // bytes per KV position in TQ tensor (all heads)
static int64_t      tq_sink_fp16_stride[TQ_MAX_LAYERS];  // bytes per KV position in fp16 tensor (all heads)
static int64_t      tq_sink_kv_size[TQ_MAX_LAYERS];     // kv_size per stream (for multi-stream position calc)
static int          tq_sink_n_global = 0;                 // number of sink positions (same for all layers)

// Legacy single-layer state (kept for fallback first-vector detection)
static int   tq_outlier_ch[TQ_DIM_HI];
static int   tq_regular_ch[TQ_DIM_LO];
static int   tq_initialized = 0;
static int   tq_outliers_detected = 0;

// Generate a d×d orthogonal matrix via QR of Gaussian
static void tq_gen_orthogonal(float * fwd, float * inv, int d, uint64_t seed) {
    #define TQ_ROT_LCG(s) ((s) * 6364136223846793005ULL + 1442695040888963407ULL)
    float * Q = (float *)malloc(d * d * sizeof(float));
    for (int i = 0; i < d * d; i++) {
        seed = TQ_ROT_LCG(seed);
        float u1 = ((float)(uint32_t)(seed >> 32) + 1.0f) / ((float)0xFFFFFFFF + 2.0f);
        seed = TQ_ROT_LCG(seed);
        float u2 = ((float)(uint32_t)(seed >> 32) + 1.0f) / ((float)0xFFFFFFFF + 2.0f);
        Q[i] = sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
    }
    for (int j = 0; j < d; j++) {
        for (int k = 0; k < j; k++) {
            float dot = 0.0f;
            for (int i = 0; i < d; i++) dot += Q[k*d+i] * Q[j*d+i];
            for (int i = 0; i < d; i++) Q[j*d+i] -= dot * Q[k*d+i];
        }
        float norm = 0.0f;
        for (int i = 0; i < d; i++) norm += Q[j*d+i] * Q[j*d+i];
        norm = sqrtf(norm);
        for (int i = 0; i < d; i++) Q[j*d+i] /= norm;
    }
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            fwd[i*d+j] = Q[j*d+i];
            inv[j*d+i] = Q[j*d+i];
        }
    }
    free(Q);
    #undef TQ_ROT_LCG
}

static void tq_init_rotations(void) {
    if (tq_initialized) return;
    tq_gen_orthogonal(tq_rot_hi_fwd, tq_rot_hi_inv, TQ_DIM_HI, 0x5475524230484932ULL); // "TuRB0HI2"
    tq_gen_orthogonal(tq_rot_lo_fwd, tq_rot_lo_inv, TQ_DIM_LO, 0x54755242304C4F36ULL); // "TuRB0LO6"
    tq_gen_orthogonal(tq_rot_v_fwd, tq_rot_v_inv, TQ_DIM, 0x5475524230564131ULL);      // "TuRB0VA1"
    // Default outlier channels: first 32 (will be overridden by calibration)
    for (int i = 0; i < TQ_DIM_HI; i++) tq_outlier_ch[i] = i;
    for (int i = 0; i < TQ_DIM_LO; i++) tq_regular_ch[i] = TQ_DIM_HI + i;
    // Initialize per-layer-per-head registries to default (channels 0-31 = outlier)
    for (int l = 0; l < TQ_MAX_LAYERS; l++) {
        for (int h = 0; h < TQ_MAX_HEADS; h++) {
            for (int i = 0; i < TQ_DIM_HI; i++) {
                tq_k_outlier_reg[l][h][i] = i;
                tq_v_outlier_reg[l][h][i] = i;
            }
            for (int i = 0; i < TQ_DIM_LO; i++) {
                tq_k_regular_reg[l][h][i] = TQ_DIM_HI + i;
                tq_v_regular_reg[l][h][i] = TQ_DIM_HI + i;
            }
        }
    }
    tq_initialized = 1;
}

// Detect outlier channels from a 128-dim vector (by absolute magnitude)
// Called on the first vector; locks the channel assignment for all subsequent blocks.
static void tq_detect_outliers(const float * x) {
    if (tq_outliers_detected) return;

    // Sort channels by absolute magnitude
    int order[TQ_DIM];
    float mag[TQ_DIM];
    for (int i = 0; i < TQ_DIM; i++) { order[i] = i; mag[i] = fabsf(x[i]); }
    // Simple insertion sort (only 128 elements)
    for (int i = 1; i < TQ_DIM; i++) {
        int key_idx = order[i]; float key_mag = mag[key_idx];
        int j = i - 1;
        while (j >= 0 && mag[order[j]] < key_mag) { order[j+1] = order[j]; j--; }
        order[j+1] = key_idx;
    }
    // Top 32 by magnitude = outliers
    for (int i = 0; i < TQ_DIM_HI; i++) tq_outlier_ch[i] = order[i];
    // Sort outlier indices for consistent ordering
    for (int i = 1; i < TQ_DIM_HI; i++) {
        int key = tq_outlier_ch[i]; int j = i-1;
        while (j >= 0 && tq_outlier_ch[j] > key) { tq_outlier_ch[j+1] = tq_outlier_ch[j]; j--; }
        tq_outlier_ch[j+1] = key;
    }
    // Remaining = regular channels
    int ri = 0;
    for (int i = 0; i < TQ_DIM; i++) {
        int is_outlier = 0;
        for (int j = 0; j < TQ_DIM_HI; j++) { if (tq_outlier_ch[j] == i) { is_outlier = 1; break; } }
        if (!is_outlier) tq_regular_ch[ri++] = i;
    }
    tq_outliers_detected = 1;
}

// Extract hi/lo channel subsets from a 128-dim vector
// K cache: uses calibrated per-layer-per-head outlier channels
// V cache: uses default fixed split (0-31 / 32-127) — V has no outliers (per RotateKV paper)
static void tq_split_channels(const float * x, float * hi, float * lo) {
    const int * outlier = tq_cur_is_k ? tq_k_outlier_reg[tq_cur_layer][tq_cur_head] : tq_v_outlier_reg[0][0];
    const int * regular = tq_cur_is_k ? tq_k_regular_reg[tq_cur_layer][tq_cur_head] : tq_v_regular_reg[0][0];
    for (int i = 0; i < TQ_DIM_HI; i++) hi[i] = x[outlier[i]];
    for (int i = 0; i < TQ_DIM_LO; i++) lo[i] = x[regular[i]];
}

// Merge hi/lo channel subsets back into 128-dim vector
static void tq_merge_channels(const float * hi, const float * lo, float * x) {
    const int * outlier = tq_cur_is_k ? tq_k_outlier_reg[tq_cur_layer][tq_cur_head] : tq_v_outlier_reg[0][0];
    const int * regular = tq_cur_is_k ? tq_k_regular_reg[tq_cur_layer][tq_cur_head] : tq_v_regular_reg[0][0];
    for (int i = 0; i < TQ_DIM_HI; i++) x[outlier[i]] = hi[i];
    for (int i = 0; i < TQ_DIM_LO; i++) x[regular[i]] = lo[i];
}

// Rotate hi subset: out = Π_hi * in
static void tq_rotate_hi(const float * in, float * out) {
    for (int i = 0; i < TQ_DIM_HI; i++) {
        float sum = 0.0f;
        for (int j = 0; j < TQ_DIM_HI; j++) sum += tq_rot_hi_fwd[i*TQ_DIM_HI+j] * in[j];
        out[i] = sum;
    }
}
static void tq_unrotate_hi(const float * in, float * out) {
    for (int i = 0; i < TQ_DIM_HI; i++) {
        float sum = 0.0f;
        for (int j = 0; j < TQ_DIM_HI; j++) sum += tq_rot_hi_inv[i*TQ_DIM_HI+j] * in[j];
        out[i] = sum;
    }
}

// Rotate lo subset: out = Π_lo * in
static void tq_rotate_lo(const float * in, float * out) {
    for (int i = 0; i < TQ_DIM_LO; i++) {
        float sum = 0.0f;
        for (int j = 0; j < TQ_DIM_LO; j++) sum += tq_rot_lo_fwd[i*TQ_DIM_LO+j] * in[j];
        out[i] = sum;
    }
}
static void tq_unrotate_lo(const float * in, float * out) {
    for (int i = 0; i < TQ_DIM_LO; i++) {
        float sum = 0.0f;
        for (int j = 0; j < TQ_DIM_LO; j++) sum += tq_rot_lo_inv[i*TQ_DIM_LO+j] * in[j];
        out[i] = sum;
    }
}

// Rotate full 128-dim vector: out = Π_v * in (for V cache, no split)
static void tq_rotate_v(const float * in, float * out) {
    for (int i = 0; i < TQ_DIM; i++) {
        float sum = 0.0f;
        for (int j = 0; j < TQ_DIM; j++) sum += tq_rot_v_fwd[i*TQ_DIM+j] * in[j];
        out[i] = sum;
    }
}
static void tq_unrotate_v(const float * in, float * out) {
    for (int i = 0; i < TQ_DIM; i++) {
        float sum = 0.0f;
        for (int j = 0; j < TQ_DIM; j++) sum += tq_rot_v_inv[i*TQ_DIM+j] * in[j];
        out[i] = sum;
    }
}

// Full split rotate/unrotate: split into outlier/regular, rotate each independently
static void tq_rotate_split(const float * in, float * hi_rot, float * lo_rot) {
    float hi[TQ_DIM_HI], lo[TQ_DIM_LO];
    tq_split_channels(in, hi, lo);
    tq_rotate_hi(hi, hi_rot);
    tq_rotate_lo(lo, lo_rot);
}

static void tq_unrotate_merge(const float * hi_rot, const float * lo_rot, float * out) {
    float hi[TQ_DIM_HI], lo[TQ_DIM_LO];
    tq_unrotate_hi(hi_rot, hi);
    tq_unrotate_lo(lo_rot, lo);
    tq_merge_channels(hi, lo, out);
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

// Pre-computed QJL matrix (generated once, reused everywhere)
static float tq_qjl_mat_32[TQ_DIM_HI * TQ_DIM_HI];  // 32×32 = 4KB
static int   tq_qjl_mat_initialized = 0;

static void tq_init_qjl_matrices(void) {
    if (tq_qjl_mat_initialized) return;
    // Forward declaration — tq_get_qjl_matrix is defined later
    // Use inline generation instead
    uint64_t st = QJL_SEED_32;
    for (int i = 0; i < TQ_DIM_HI; i++) {
        for (int j = 0; j < TQ_DIM_HI; j++) {
            st = st * 6364136223846793005ULL + 1442695040888963407ULL;
            float u1 = ((float)(uint32_t)(st >> 32) + 1.0f) / ((float)0xFFFFFFFF + 2.0f);
            st = st * 6364136223846793005ULL + 1442695040888963407ULL;
            float u2 = ((float)(uint32_t)(st >> 32) + 1.0f) / ((float)0xFFFFFFFF + 2.0f);
            tq_qjl_mat_32[i * TQ_DIM_HI + j] = sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
        }
    }
    tq_qjl_mat_initialized = 1;
}

static float qjl_forward(const float * r, uint8_t * signs, int m, uint64_t seed) {
    (void)seed;
    tq_init_qjl_matrices();
    const float * S = (m == TQ_DIM_HI) ? tq_qjl_mat_32 : NULL;

    float rnorm_sq = 0.0f;
    for (int j = 0; j < m; j++) rnorm_sq += r[j] * r[j];

    memset(signs, 0, (m + 7) / 8);
    if (S) {
        for (int i = 0; i < m; i++) {
            float proj = 0.0f;
            for (int j = 0; j < m; j++) proj += S[i * m + j] * r[j];
            if (proj >= 0.0f) signs[i / 8] |= (uint8_t)(1 << (i % 8));
        }
    } else {
        tq_seed(seed);
        for (int i = 0; i < m; i++) {
            float proj = 0.0f;
            for (int j = 0; j < m; j++) proj += tq_gaussian() * r[j];
            if (proj >= 0.0f) signs[i / 8] |= (uint8_t)(1 << (i % 8));
        }
    }
    return sqrtf(rnorm_sq);
}

static void qjl_inverse(const uint8_t * signs, float rnorm, float * corr, int m, uint64_t seed) {
    (void)seed;
    tq_init_qjl_matrices();
    const float * S = (m == TQ_DIM_HI) ? tq_qjl_mat_32 : NULL;

    memset(corr, 0, m * sizeof(float));
    if (S) {
        for (int i = 0; i < m; i++) {
            float z = ((signs[i / 8] >> (i % 8)) & 1) ? 1.0f : -1.0f;
            for (int j = 0; j < m; j++) corr[j] += S[i * m + j] * z;
        }
    } else {
        tq_seed(seed);
        for (int i = 0; i < m; i++) {
            float z = ((signs[i / 8] >> (i % 8)) & 1) ? 1.0f : -1.0f;
            for (int j = 0; j < m; j++) corr[j] += tq_gaussian() * z;
        }
    }
    float scale = 1.2533141f / (float)m * rnorm;
    for (int j = 0; j < m; j++) corr[j] *= scale;
}

// Write fp16 copy to k_sink tensor inside from_float (atomic with TQ write, no graph ordering issue).
// Called from each TQ quantize function after writing TQ data.
// k_sink is small: [n_embd_k_gqa, n_sinks] — only n_sinks rows, NOT kv_size.
// Only writes for positions < n_sinks; skips everything else.
void tq_sink_write_fp16(const float * x, const void * tq_dst, int64_t k, int64_t tq_block_bytes) {
    (void)tq_block_bytes;
    if (tq_sink_n_global <= 0 || tq_cur_layer < 0 || tq_cur_layer >= TQ_MAX_LAYERS) return;
    const char * k_base    = (const char *)tq_sink_k_base[tq_cur_layer];
    char       * fp16_base = (char *)tq_sink_fp16_base[tq_cur_layer];
    int64_t k_stride       = tq_sink_k_stride[tq_cur_layer];
    int64_t fp16_stride    = tq_sink_fp16_stride[tq_cur_layer];
    if (!k_base || !fp16_base || k_stride <= 0) return;

    // Compute cell index from TQ destination pointer
    int64_t byte_off = (const char *)tq_dst - k_base;
    if (byte_off < 0) return;
    int64_t kv_size    = tq_sink_kv_size[tq_cur_layer];
    int64_t stream_sz  = k_stride * kv_size;
    int64_t stream_idx = stream_sz > 0 ? byte_off / stream_sz : 0;
    int64_t within     = byte_off - stream_idx * stream_sz;
    int64_t cell       = within / k_stride;

    // Only write for sink positions
    if (cell < 0 || cell >= tq_sink_n_global) return;

    // k_sink has n_sinks rows per stream, not kv_size
    int64_t fp16_row = stream_idx * tq_sink_n_global + cell;
    ggml_half * dst = (ggml_half *)(fp16_base + fp16_row * fp16_stride);
    for (int64_t j = 0; j < k; j++) {
        dst[j] = GGML_FP32_TO_FP16(x[j]);
    }
}

// ---------------------------------------------------------------------------
// Pointer-based sink dispatch for vec_dot and dequant.
// Determines if a TQ block pointer corresponds to a sink position by comparing
// against the registered k_base. If sink, reads fp16 from k_sink tensor instead.
// ---------------------------------------------------------------------------

// Try sink dot product: if this block is a sink, compute fp16 dot and return 1.
// Otherwise return 0 (caller should do normal TQ dot).
static int tq_try_sink_dot(const void * tq_ptr, const float * q, int64_t k, float * result) {
    if (tq_sink_n_global <= 0 || tq_cur_layer < 0 || tq_cur_layer >= TQ_MAX_LAYERS) return 0;
    const char * k_base    = (const char *)tq_sink_k_base[tq_cur_layer];
    const char * fp16_base = (const char *)tq_sink_fp16_base[tq_cur_layer];
    int64_t k_stride       = tq_sink_k_stride[tq_cur_layer];
    int64_t fp16_stride    = tq_sink_fp16_stride[tq_cur_layer];
    if (!k_base || !fp16_base || k_stride <= 0) return 0;

    int64_t byte_off = (const char *)tq_ptr - k_base;
    if (byte_off < 0) return 0;
    int64_t kv_size    = tq_sink_kv_size[tq_cur_layer];
    int64_t stream_sz  = k_stride * kv_size;
    int64_t stream_idx = stream_sz > 0 ? byte_off / stream_sz : 0;
    int64_t within     = byte_off - stream_idx * stream_sz;
    int64_t cell       = within / k_stride;

    if (cell < 0 || cell >= tq_sink_n_global) return 0;

    // Read fp16 from k_sink (n_sinks rows per stream), offset by head
    int64_t head_off = within % k_stride;
    int64_t head_idx = head_off / (int64_t)ggml_type_size(tq_cur_is_k ? GGML_TYPE_TQK_HAD_MSE4 : GGML_TYPE_F16);
    // Actually: we don't know which TQ type. Use k as HEAD_DIM to compute head index.
    // head_off / (k_stride / n_heads) = head_off * n_heads / k_stride
    // Simpler: fp16 offset = head_off * (k * sizeof(fp16)) / block_size
    // But we don't have block_size here. Use: head offset in fp16 = (head_off / tq_block_size) * k * sizeof(fp16)
    // We don't know tq_block_size. Compute from k_stride and k: n_heads = k_stride / block_size, block_size = k_stride / n_heads.
    // n_heads = fp16_stride / (k * sizeof(fp16)).
    int64_t n_heads = fp16_stride / (k * (int64_t)sizeof(ggml_half));
    int64_t tq_block_size = (n_heads > 0) ? k_stride / n_heads : k_stride;
    int64_t head = (tq_block_size > 0) ? head_off / tq_block_size : 0;

    int64_t fp16_row = stream_idx * tq_sink_n_global + cell;
    const ggml_half * src = (const ggml_half *)(fp16_base + fp16_row * fp16_stride + head * k * (int64_t)sizeof(ggml_half));
    float dot = 0.0f;
    for (int64_t j = 0; j < k; j++) {
        dot += GGML_FP16_TO_FP32(src[j]) * q[j];
    }
    *result = dot;
    return 1;
}

// Try sink dequant: if this block is a sink, read fp16 from k_sink and convert to f32.
// Returns 1 if handled (caller should skip TQ dequant), 0 otherwise.
static int tq_try_sink_dequant(const void * tq_ptr, float * out, int64_t k) {
    if (tq_sink_n_global <= 0 || tq_cur_layer < 0 || tq_cur_layer >= TQ_MAX_LAYERS) return 0;
    const char * k_base    = (const char *)tq_sink_k_base[tq_cur_layer];
    const char * fp16_base = (const char *)tq_sink_fp16_base[tq_cur_layer];
    int64_t k_stride       = tq_sink_k_stride[tq_cur_layer];
    int64_t fp16_stride    = tq_sink_fp16_stride[tq_cur_layer];
    if (!k_base || !fp16_base || k_stride <= 0) return 0;

    int64_t byte_off = (const char *)tq_ptr - k_base;
    if (byte_off < 0) return 0;
    int64_t kv_size    = tq_sink_kv_size[tq_cur_layer];
    int64_t stream_sz  = k_stride * kv_size;
    int64_t stream_idx = stream_sz > 0 ? byte_off / stream_sz : 0;
    int64_t within     = byte_off - stream_idx * stream_sz;
    int64_t cell       = within / k_stride;

    if (cell < 0 || cell >= tq_sink_n_global) return 0;

    // Read fp16 from k_sink (n_sinks rows per stream), offset by head
    int64_t head_off = within % k_stride;
    int64_t n_heads = fp16_stride / (k * (int64_t)sizeof(ggml_half));
    int64_t tq_block_size = (n_heads > 0) ? k_stride / n_heads : k_stride;
    int64_t head = (tq_block_size > 0) ? head_off / tq_block_size : 0;

    int64_t fp16_row = stream_idx * tq_sink_n_global + cell;
    const ggml_half * src = (const ggml_half *)(fp16_base + fp16_row * fp16_stride + head * k * (int64_t)sizeof(ggml_half));
    for (int64_t j = 0; j < k; j++) {
        out[j] = GGML_FP16_TO_FP32(src[j]);
    }
    return 1;
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
// TQK 2.5: hi=b3(centroids_4, 2-bit), lo=b2(centroids_2, 1-bit)
// ---------------------------------------------------------------------------

void quantize_row_tqk_25_ref(const float * GGML_RESTRICT x, block_tqk_25 * GGML_RESTRICT y, int64_t k) {
    assert(k % TQK_BLOCK_SIZE == 0);
    const int64_t nb = k / TQK_BLOCK_SIZE;
    tq_init_rotations();

    for (int64_t i = 0; i < nb; i++) {
        if (nb > 1) tq_cur_head = (int)(i % TQ_MAX_HEADS);
        const float * xb = x + i * TQK_BLOCK_SIZE;

        // Split into outlier (hi) and regular (lo) channel subsets (per-layer)
        float hi_raw[TQK_N_OUTLIER], lo_raw[TQK_N_REGULAR];
        tq_split_channels(xb, hi_raw, lo_raw);

        // Rotate each subset independently
        float hi_rot[TQK_N_OUTLIER], lo_rot[TQK_N_REGULAR];
        tq_rotate_hi(hi_raw, hi_rot);
        tq_rotate_lo(lo_raw, lo_rot);

        // Per-subset norms
        float sum_hi = 0.0f, sum_lo = 0.0f;
        for (int j = 0; j < TQK_N_OUTLIER; j++) sum_hi += hi_rot[j] * hi_rot[j];
        for (int j = 0; j < TQK_N_REGULAR; j++) sum_lo += lo_rot[j] * lo_rot[j];
        float norm_hi = sqrtf(sum_hi), norm_lo = sqrtf(sum_lo);

        y[i].norm_hi = GGML_FP32_TO_FP16(norm_hi);
        y[i].norm_lo = GGML_FP32_TO_FP16(norm_lo);
        y[i].rnorm_hi = GGML_FP32_TO_FP16(0.0f);
        y[i].rnorm_lo = GGML_FP32_TO_FP16(0.0f);
        memset(y[i].signs_hi, 0, sizeof(y[i].signs_hi));
        memset(y[i].signs_lo, 0, sizeof(y[i].signs_lo));

        if (norm_hi == 0.0f && norm_lo == 0.0f) { memset(y[i].qs_hi, 0, sizeof(y[i].qs_hi)); memset(y[i].qs_lo, 0, sizeof(y[i].qs_lo)); continue; }

        float inv_hi = (norm_hi > 1e-12f) ? 1.0f / norm_hi : 0.0f;
        float inv_lo = (norm_lo > 1e-12f) ? 1.0f / norm_lo : 0.0f;

        float res_hi_dummy[TQK_N_OUTLIER], res_lo_dummy[TQK_N_REGULAR];
        quant_hi(hi_rot, inv_hi, y[i].qs_hi, centroids_4_d32, 4, 2, TQK_N_OUTLIER, res_hi_dummy);
        quant_lo(lo_rot, inv_lo, y[i].qs_lo, centroids_2_d96, 2, 1, TQK_N_REGULAR, 0, res_lo_dummy);

        // Paper Algorithm 2: residual in original subset space
        float yhi[TQK_N_OUTLIER], ylo[TQK_N_REGULAR];
        for (int j = 0; j < TQK_N_OUTLIER; j++) yhi[j] = centroids_4_d32[(y[i].qs_hi[j/4] >> ((j%4)*2)) & 0x3];
        for (int j = 0; j < TQK_N_REGULAR; j++) ylo[j] = centroids_2_d96[(y[i].qs_lo[j/8] >> (j%8)) & 1];
        float hi_rec[TQK_N_OUTLIER], lo_rec[TQK_N_REGULAR];
        tq_unrotate_hi(yhi, hi_rec);
        tq_unrotate_lo(ylo, lo_rec);
        float r_hi[TQK_N_OUTLIER], r_lo[TQK_N_REGULAR];
        for (int j = 0; j < TQK_N_OUTLIER; j++) r_hi[j] = hi_raw[j] - norm_hi * hi_rec[j];
        for (int j = 0; j < TQK_N_REGULAR; j++) r_lo[j] = lo_raw[j] - norm_lo * lo_rec[j];

        y[i].rnorm_hi = GGML_FP32_TO_FP16(qjl_forward(r_hi, y[i].signs_hi, TQK_N_OUTLIER, QJL_SEED_32));
        y[i].rnorm_lo = GGML_FP32_TO_FP16(qjl_forward(r_lo, y[i].signs_lo, TQK_N_REGULAR, QJL_SEED_96));
    }
}

void dequantize_row_tqk_25(const block_tqk_25 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % TQK_BLOCK_SIZE == 0);
    const int64_t nb = k / TQK_BLOCK_SIZE;
    tq_init_rotations();

    for (int64_t i = 0; i < nb; i++) {
        if (nb > 1) tq_cur_head = (int)(i % TQ_MAX_HEADS);
        float norm_hi = GGML_FP16_TO_FP32(x[i].norm_hi);
        float norm_lo = GGML_FP16_TO_FP32(x[i].norm_lo);

        // Paper DEQUANT_prod: MSE recon in original space + QJL correction
        float yhi[TQK_N_OUTLIER], ylo[TQK_N_REGULAR];
        for (int j = 0; j < TQK_N_OUTLIER; j++) yhi[j] = centroids_4_d32[(x[i].qs_hi[j/4] >> ((j%4)*2)) & 0x3];
        for (int j = 0; j < TQK_N_REGULAR; j++) ylo[j] = centroids_2_d96[(x[i].qs_lo[j/8] >> (j%8)) & 1];

        float hi_orig[TQK_N_OUTLIER], lo_orig[TQK_N_REGULAR];
        tq_unrotate_hi(yhi, hi_orig);
        tq_unrotate_lo(ylo, lo_orig);
        for (int j = 0; j < TQK_N_OUTLIER; j++) hi_orig[j] *= norm_hi;
        for (int j = 0; j < TQK_N_REGULAR; j++) lo_orig[j] *= norm_lo;

        float corr_hi[TQK_N_OUTLIER], corr_lo[TQK_N_REGULAR];
        qjl_inverse(x[i].signs_hi, GGML_FP16_TO_FP32(x[i].rnorm_hi), corr_hi, TQK_N_OUTLIER, QJL_SEED_32);
        qjl_inverse(x[i].signs_lo, GGML_FP16_TO_FP32(x[i].rnorm_lo), corr_lo, TQK_N_REGULAR, QJL_SEED_96);
        for (int j = 0; j < TQK_N_OUTLIER; j++) hi_orig[j] += corr_hi[j];
        for (int j = 0; j < TQK_N_REGULAR; j++) lo_orig[j] += corr_lo[j];

        tq_merge_channels(hi_orig, lo_orig, y + i * TQK_BLOCK_SIZE);
    }
}

// ---------------------------------------------------------------------------
// TQK 3.5: independent instances — hi=3-bit MSE (d=32), lo=2-bit MSE (d=96)
// ---------------------------------------------------------------------------

void quantize_row_tqk_35_ref(const float * GGML_RESTRICT x, block_tqk_35 * GGML_RESTRICT y, int64_t k) {
    assert(k % TQK_BLOCK_SIZE == 0);
    const int64_t nb = k / TQK_BLOCK_SIZE;
    tq_init_rotations();

    for (int64_t i = 0; i < nb; i++) {
        if (nb > 1) tq_cur_head = (int)(i % TQ_MAX_HEADS);
        const float * xb = x + i * TQK_BLOCK_SIZE;

        float hi_raw[TQK_N_OUTLIER], lo_raw[TQK_N_REGULAR];
        tq_split_channels(xb, hi_raw, lo_raw);

        float hi_rot[TQK_N_OUTLIER], lo_rot[TQK_N_REGULAR];
        tq_rotate_hi(hi_raw, hi_rot);
        tq_rotate_lo(lo_raw, lo_rot);

        float sum_hi = 0.0f, sum_lo = 0.0f;
        for (int j = 0; j < TQK_N_OUTLIER; j++) sum_hi += hi_rot[j] * hi_rot[j];
        for (int j = 0; j < TQK_N_REGULAR; j++) sum_lo += lo_rot[j] * lo_rot[j];
        float norm_hi = sqrtf(sum_hi), norm_lo = sqrtf(sum_lo);

        y[i].norm_hi = GGML_FP32_TO_FP16(norm_hi);
        y[i].norm_lo = GGML_FP32_TO_FP16(norm_lo);
        y[i].rnorm_hi = GGML_FP32_TO_FP16(0.0f);
        y[i].rnorm_lo = GGML_FP32_TO_FP16(0.0f);
        memset(y[i].signs_hi, 0, sizeof(y[i].signs_hi));
        memset(y[i].signs_lo, 0, sizeof(y[i].signs_lo));

        if (norm_hi == 0.0f && norm_lo == 0.0f) { memset(y[i].qs_hi, 0, sizeof(y[i].qs_hi)); memset(y[i].qs_lo, 0, sizeof(y[i].qs_lo)); continue; }

        float inv_hi = (norm_hi > 1e-12f) ? 1.0f / norm_hi : 0.0f;
        float inv_lo = (norm_lo > 1e-12f) ? 1.0f / norm_lo : 0.0f;

        // MSE quantize (in normalized-rotated space, same as before)
        float res_hi_dummy[TQK_N_OUTLIER], res_lo_dummy[TQK_N_REGULAR];
        quant_hi(hi_rot, inv_hi, y[i].qs_hi, centroids_8_d32, 8, 3, TQK_N_OUTLIER, res_hi_dummy);
        quant_lo(lo_rot, inv_lo, y[i].qs_lo, centroids_4_d96, 4, 2, TQK_N_REGULAR, 0, res_lo_dummy);

        // Paper Algorithm 2: residual in original subset space
        // DEQUANT_mse: ỹ = centroids[idx], x̂_rec = Π^T · ỹ, x_rec = norm * x̂_rec
        float yhi[TQK_N_OUTLIER], ylo[TQK_N_REGULAR];
        for (int j = 0; j < TQK_N_OUTLIER; j++) yhi[j] = centroids_8_d32[up3(y[i].qs_hi, j)];
        for (int j = 0; j < TQK_N_REGULAR; j++) ylo[j] = centroids_4_d96[(y[i].qs_lo[j/4] >> ((j%4)*2)) & 0x3];
        float hi_rec[TQK_N_OUTLIER], lo_rec[TQK_N_REGULAR];
        tq_unrotate_hi(yhi, hi_rec);
        tq_unrotate_lo(ylo, lo_rec);
        // r = x_raw - norm * x̂_rec
        float r_hi[TQK_N_OUTLIER], r_lo[TQK_N_REGULAR];
        for (int j = 0; j < TQK_N_OUTLIER; j++) r_hi[j] = hi_raw[j] - norm_hi * hi_rec[j];
        for (int j = 0; j < TQK_N_REGULAR; j++) r_lo[j] = lo_raw[j] - norm_lo * lo_rec[j];

        // QJL on residual in original subset space
        y[i].rnorm_hi = GGML_FP32_TO_FP16(qjl_forward(r_hi, y[i].signs_hi, TQK_N_OUTLIER, QJL_SEED_32));
        y[i].rnorm_lo = GGML_FP32_TO_FP16(qjl_forward(r_lo, y[i].signs_lo, TQK_N_REGULAR, QJL_SEED_96));
    }
}

void dequantize_row_tqk_35(const block_tqk_35 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % TQK_BLOCK_SIZE == 0);
    const int64_t nb = k / TQK_BLOCK_SIZE;
    tq_init_rotations();

    for (int64_t i = 0; i < nb; i++) {
        if (nb > 1) tq_cur_head = (int)(i % TQ_MAX_HEADS);
        float norm_hi = GGML_FP16_TO_FP32(x[i].norm_hi);
        float norm_lo = GGML_FP16_TO_FP32(x[i].norm_lo);

        // Paper DEQUANT_prod: x̃_mse + x̃_qjl
        // Step 1: MSE reconstruction — centroids → unrotate → scale by norm
        float yhi[TQK_N_OUTLIER], ylo[TQK_N_REGULAR];
        for (int j = 0; j < TQK_N_OUTLIER; j++) yhi[j] = centroids_8_d32[up3(x[i].qs_hi, j)];
        for (int j = 0; j < TQK_N_REGULAR; j++) ylo[j] = centroids_4_d96[(x[i].qs_lo[j/4] >> ((j%4)*2)) & 0x3];

        float hi_orig[TQK_N_OUTLIER], lo_orig[TQK_N_REGULAR];
        tq_unrotate_hi(yhi, hi_orig);
        tq_unrotate_lo(ylo, lo_orig);
        for (int j = 0; j < TQK_N_OUTLIER; j++) hi_orig[j] *= norm_hi;
        for (int j = 0; j < TQK_N_REGULAR; j++) lo_orig[j] *= norm_lo;

        // Step 2: QJL correction in original subset space
        float corr_hi[TQK_N_OUTLIER], corr_lo[TQK_N_REGULAR];
        qjl_inverse(x[i].signs_hi, GGML_FP16_TO_FP32(x[i].rnorm_hi), corr_hi, TQK_N_OUTLIER, QJL_SEED_32);
        qjl_inverse(x[i].signs_lo, GGML_FP16_TO_FP32(x[i].rnorm_lo), corr_lo, TQK_N_REGULAR, QJL_SEED_96);
        for (int j = 0; j < TQK_N_OUTLIER; j++) hi_orig[j] += corr_hi[j];
        for (int j = 0; j < TQK_N_REGULAR; j++) lo_orig[j] += corr_lo[j];

        tq_merge_channels(hi_orig, lo_orig, y + i * TQK_BLOCK_SIZE);
    }
}

// ---------------------------------------------------------------------------
// TQV 2.5: 128×128 rotation, no outlier split, d=128 centroids
// hi=3-bit MSE (first 32 rotated channels), lo=2-bit MSE (last 96)
// ---------------------------------------------------------------------------

void quantize_row_tqv_25_ref(const float * GGML_RESTRICT x, block_tqv_25 * GGML_RESTRICT y, int64_t k) {
    assert(k % TQV_BLOCK_SIZE == 0);
    const int64_t nb = k / TQV_BLOCK_SIZE;
    tq_init_rotations();

    for (int64_t i = 0; i < nb; i++) {
        const float * xb = x + i * TQV_BLOCK_SIZE;

        // Single norm for full vector
        float norm = 0.0f;
        for (int j = 0; j < TQ_DIM; j++) norm += xb[j] * xb[j];
        norm = sqrtf(norm);

        y[i].norm_hi = GGML_FP32_TO_FP16(norm);
        y[i].norm_lo = GGML_FP32_TO_FP16(0.0f); // unused

        if (norm == 0.0f) { memset(y[i].qs_hi, 0, sizeof(y[i].qs_hi)); memset(y[i].qs_lo, 0, sizeof(y[i].qs_lo)); continue; }
        float inv = 1.0f / norm;

        // Normalize and rotate full 128-dim vector
        float xhat[TQ_DIM], rot[TQ_DIM];
        for (int j = 0; j < TQ_DIM; j++) xhat[j] = xb[j] * inv;
        tq_rotate_v(xhat, rot);

        // Quantize: first 32 → qs_hi (3-bit, 8 centroids), last 96 → qs_lo (2-bit, 4 centroids)
        // Both use d=128 centroids
        quant_hi_mse(rot, 1.0f, y[i].qs_hi, centroids_8, 8, 3, TQV_N_OUTLIER);
        quant_lo_mse(rot, 1.0f, y[i].qs_lo, centroids_4, 4, 2, TQV_N_REGULAR, TQV_N_OUTLIER);
    }
}

void dequantize_row_tqv_25(const block_tqv_25 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % TQV_BLOCK_SIZE == 0);
    const int64_t nb = k / TQV_BLOCK_SIZE;
    tq_init_rotations();

    for (int64_t i = 0; i < nb; i++) {
        float norm = GGML_FP16_TO_FP32(x[i].norm_hi);

        // Dequant: centroids → full 128-dim rotated vector
        float rot[TQ_DIM];
        dequant_hi_mse(x[i].qs_hi, centroids_8, 3, TQV_N_OUTLIER, 1.0f, rot);
        dequant_lo_mse(x[i].qs_lo, centroids_4, 2, TQV_N_REGULAR, 1.0f, rot, TQV_N_OUTLIER);

        // Unrotate full vector and scale by norm
        float tmp[TQ_DIM];
        tq_unrotate_v(rot, tmp);
        for (int j = 0; j < TQ_DIM; j++) y[i * TQV_BLOCK_SIZE + j] = norm * tmp[j];
    }
}

// ---------------------------------------------------------------------------
// TQV 3.5: 128×128 rotation, no outlier split, d=128 centroids
// hi=4-bit MSE (first 32 rotated channels), lo=3-bit MSE (last 96)
// Per RotateKV paper: V has no outliers, use simple offline rotation.
// ---------------------------------------------------------------------------

void quantize_row_tqv_35_ref(const float * GGML_RESTRICT x, block_tqv_35 * GGML_RESTRICT y, int64_t k) {
    assert(k % TQV_BLOCK_SIZE == 0);
    const int64_t nb = k / TQV_BLOCK_SIZE;
    tq_init_rotations();

    for (int64_t i = 0; i < nb; i++) {
        const float * xb = x + i * TQV_BLOCK_SIZE;

        // Single norm for full vector
        float norm = 0.0f;
        for (int j = 0; j < TQ_DIM; j++) norm += xb[j] * xb[j];
        norm = sqrtf(norm);

        y[i].norm_hi = GGML_FP32_TO_FP16(norm);
        y[i].norm_lo = GGML_FP32_TO_FP16(0.0f);

        if (norm == 0.0f) { memset(y[i].qs_hi, 0, sizeof(y[i].qs_hi)); memset(y[i].qs_lo, 0, sizeof(y[i].qs_lo)); continue; }
        float inv = 1.0f / norm;

        // Normalize and rotate full 128-dim
        float xhat[TQ_DIM], rot[TQ_DIM];
        for (int j = 0; j < TQ_DIM; j++) xhat[j] = xb[j] * inv;
        tq_rotate_v(xhat, rot);

        // Quantize: first 32 → qs_hi (4-bit, 16 centroids), last 96 → qs_lo (3-bit, 8 centroids)
        // Both use d=128 centroids
        quant_hi_mse(rot, 1.0f, y[i].qs_hi, centroids_16, 16, 4, TQV_N_OUTLIER);
        quant_lo_mse(rot, 1.0f, y[i].qs_lo, centroids_8, 8, 3, TQV_N_REGULAR, TQV_N_OUTLIER);
    }
}

void dequantize_row_tqv_35(const block_tqv_35 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % TQV_BLOCK_SIZE == 0);
    const int64_t nb = k / TQV_BLOCK_SIZE;
    tq_init_rotations();

    for (int64_t i = 0; i < nb; i++) {
        float norm = GGML_FP16_TO_FP32(x[i].norm_hi);

        float rot[TQ_DIM];
        dequant_hi_mse(x[i].qs_hi, centroids_16, 4, TQV_N_OUTLIER, 1.0f, rot);
        dequant_lo_mse(x[i].qs_lo, centroids_8, 3, TQV_N_REGULAR, 1.0f, rot, TQV_N_OUTLIER);

        float tmp[TQ_DIM];
        tq_unrotate_v(rot, tmp);
        for (int j = 0; j < TQ_DIM; j++) y[i * TQV_BLOCK_SIZE + j] = norm * tmp[j];
    }
}

// ---------------------------------------------------------------------------
// Asymmetric inner product estimator (paper Algorithm 2)
//
// <q, k> ≈ <q_rot, k_mse> + rnorm * sqrt(π/2) / m * <S @ q_rot, sign(S @ r_k)>
//
// where k_mse is the centroid-only reconstruction in rotated space,
// sign(S @ r_k) are the stored QJL sign bits, and rnorm = ||r_k||.
// The query q is in original space; we rotate it to match the stored data.
//
// This estimator is UNBIASED (E[estimate] = <q, k>) unlike the
// dequantize-then-dot approach which adds QJL reconstruction noise.
// ---------------------------------------------------------------------------

// Compute <S_row_i, q_rot> for the QJL projection of the query
// Uses the same deterministic PRNG as qjl_forward/qjl_inverse
static float qjl_project_query_element(const float * q_rot, int i, int m, uint64_t seed) {
    // Fast-forward PRNG to the right position: need to skip i*m Gaussian draws
    // to get the i-th row of S, then dot with q_rot[0..m-1]
    uint64_t st = seed;
    // Skip to row i: each row generates m Gaussians, each takes 2 PRNG steps
    for (int skip = 0; skip < i * m; skip++) {
        st = st * 6364136223846793005ULL + 1442695040888963407ULL;
        st = st * 6364136223846793005ULL + 1442695040888963407ULL;
    }
    // Now generate row i and dot with q_rot
    float proj = 0.0f;
    for (int j = 0; j < m; j++) {
        st = st * 6364136223846793005ULL + 1442695040888963407ULL;
        float u1 = ((float)(uint32_t)(st >> 32) + 1.0f) / ((float)0xFFFFFFFF + 2.0f);
        st = st * 6364136223846793005ULL + 1442695040888963407ULL;
        float u2 = ((float)(uint32_t)(st >> 32) + 1.0f) / ((float)0xFFFFFFFF + 2.0f);
        float g = sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
        proj += g * q_rot[j];
    }
    return proj;
}

// QJL correction term using pre-computed matrix:
// rnorm * sqrt(π/2) / m * <S @ q, sign(S @ r_k)>
static float qjl_asymmetric_dot(const float * q_rot, int m, uint64_t seed,
                                 const uint8_t * signs, float rnorm) {
    if (rnorm == 0.0f) return 0.0f;
    (void)seed;

    tq_init_qjl_matrices();
    const float * S = (m == TQ_DIM_HI) ? tq_qjl_mat_32 : NULL;

    float sum = 0.0f;
    if (S) {
        // Fast path: pre-computed matrix
        for (int i = 0; i < m; i++) {
            float proj = 0.0f;
            for (int j = 0; j < m; j++) proj += S[i * m + j] * q_rot[j];
            float sign = ((signs[i / 8] >> (i % 8)) & 1) ? 1.0f : -1.0f;
            sum += proj * sign;
        }
    } else {
        // Fallback: regenerate on-the-fly for non-standard dimensions
        tq_seed(seed);
        for (int i = 0; i < m; i++) {
            float proj = 0.0f;
            for (int j = 0; j < m; j++) proj += tq_gaussian() * q_rot[j];
            float sign = ((signs[i / 8] >> (i % 8)) & 1) ? 1.0f : -1.0f;
            sum += proj * sign;
        }
    }
    return 1.2533141f / (float)m * rnorm * sum;
}

// TQK 2.5 asymmetric vec_dot: key=tqk_25, query=f32
void ggml_vec_dot_tqk_25_f32(
        int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, size_t bx,
        const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % TQK_BLOCK_SIZE == 0);
    assert(nrc == 1);
    (void)bs; (void)bx; (void)by; (void)nrc;

    tq_init_rotations();

    const block_tqk_25 * GGML_RESTRICT x = (const block_tqk_25 *)vx;
    const float * GGML_RESTRICT y = (const float *)vy;
    const int64_t nb = n / TQK_BLOCK_SIZE;

    float sumf = 0.0f;

    for (int64_t i = 0; i < nb; i++) {
        if (nb > 1) tq_cur_head = (int)(i % TQ_MAX_HEADS);
        const float * q = y + i * TQK_BLOCK_SIZE;
        float norm_hi = GGML_FP16_TO_FP32(x[i].norm_hi);
        float norm_lo = GGML_FP16_TO_FP32(x[i].norm_lo);

        // Split query by outlier channels, rotate each subset
        float hi_raw[TQK_N_OUTLIER], lo_raw[TQK_N_REGULAR];
        tq_split_channels(q, hi_raw, lo_raw);
        float q_rot_hi[TQK_N_OUTLIER], q_rot_lo[TQK_N_REGULAR];
        tq_rotate_hi(hi_raw, q_rot_hi);
        tq_rotate_lo(lo_raw, q_rot_lo);

        // MSE centroid dot product per subset
        float mse_dot_hi = 0.0f, mse_dot_lo = 0.0f;
        for (int j = 0; j < TQK_N_OUTLIER; j++) {
            int idx = (x[i].qs_hi[j / 4] >> ((j % 4) * 2)) & 0x3;
            mse_dot_hi += q_rot_hi[j] * centroids_4_d32[idx];
        }
        for (int j = 0; j < TQK_N_REGULAR; j++) {
            int idx = (x[i].qs_lo[j / 8] >> (j % 8)) & 1;
            mse_dot_lo += q_rot_lo[j] * centroids_2_d96[idx];
        }
        float mse_dot = mse_dot_hi * norm_hi + mse_dot_lo * norm_lo;

        // Paper: QJL correction uses raw query (not rotated), residual is in original space
        float qjl_hi = qjl_asymmetric_dot(hi_raw, TQK_N_OUTLIER, QJL_SEED_32,
                                            x[i].signs_hi, GGML_FP16_TO_FP32(x[i].rnorm_hi));
        float qjl_lo = qjl_asymmetric_dot(lo_raw, TQK_N_REGULAR, QJL_SEED_96,
                                            x[i].signs_lo, GGML_FP16_TO_FP32(x[i].rnorm_lo));

        sumf += mse_dot + qjl_hi + qjl_lo;
    }

    *s = sumf;
}

// TQK 3.5 asymmetric vec_dot: key=tqk_35, query=f32
void ggml_vec_dot_tqk_35_f32(
        int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, size_t bx,
        const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % TQK_BLOCK_SIZE == 0);
    assert(nrc == 1);
    (void)bs; (void)bx; (void)by; (void)nrc;

    tq_init_rotations();

    const block_tqk_35 * GGML_RESTRICT x = (const block_tqk_35 *)vx;
    const float * GGML_RESTRICT y = (const float *)vy;
    const int64_t nb = n / TQK_BLOCK_SIZE;

    float sumf = 0.0f;

    for (int64_t i = 0; i < nb; i++) {
        if (nb > 1) tq_cur_head = (int)(i % TQ_MAX_HEADS);
        const float * q = y + i * TQK_BLOCK_SIZE;
        float norm_hi = GGML_FP16_TO_FP32(x[i].norm_hi);
        float norm_lo = GGML_FP16_TO_FP32(x[i].norm_lo);

        float hi_raw[TQK_N_OUTLIER], lo_raw[TQK_N_REGULAR];
        tq_split_channels(q, hi_raw, lo_raw);
        float q_rot_hi[TQK_N_OUTLIER], q_rot_lo[TQK_N_REGULAR];
        tq_rotate_hi(hi_raw, q_rot_hi);
        tq_rotate_lo(lo_raw, q_rot_lo);

        float mse_dot_hi = 0.0f, mse_dot_lo = 0.0f;
        for (int j = 0; j < TQK_N_OUTLIER; j++) {
            int idx = up3(x[i].qs_hi, j);
            mse_dot_hi += q_rot_hi[j] * centroids_8_d32[idx];
        }
        for (int j = 0; j < TQK_N_REGULAR; j++) {
            int idx = (x[i].qs_lo[j / 4] >> ((j % 4) * 2)) & 0x3;
            mse_dot_lo += q_rot_lo[j] * centroids_4_d96[idx];
        }
        float mse_dot = mse_dot_hi * norm_hi + mse_dot_lo * norm_lo;

        // Paper: QJL correction uses raw query (not rotated), residual is in original space
        float qjl_hi = qjl_asymmetric_dot(hi_raw, TQK_N_OUTLIER, QJL_SEED_32,
                                            x[i].signs_hi, GGML_FP16_TO_FP32(x[i].rnorm_hi));
        float qjl_lo = qjl_asymmetric_dot(lo_raw, TQK_N_REGULAR, QJL_SEED_96,
                                            x[i].signs_lo, GGML_FP16_TO_FP32(x[i].rnorm_lo));

        sumf += mse_dot + qjl_hi + qjl_lo;
    }

    *s = sumf;
}

// ---------------------------------------------------------------------------
// Public API: per-layer outlier calibration
// ---------------------------------------------------------------------------

void tq_register_sink_layer(int layer, const void * k_base, const void * fp16_base,
                            int n_sinks, int64_t k_stride, int64_t fp16_stride, int64_t kv_size) {
    if (layer < 0 || layer >= TQ_MAX_LAYERS) return;
    tq_sink_k_base[layer]      = k_base;
    tq_sink_fp16_base[layer]   = fp16_base;
    tq_sink_k_stride[layer]    = k_stride;
    tq_sink_fp16_stride[layer] = fp16_stride;
    tq_sink_kv_size[layer]     = kv_size;
    tq_sink_n_global           = n_sinks;
}

void tq_clear_sink_state(void) {
    for (int i = 0; i < TQ_MAX_LAYERS; i++) {
        tq_sink_k_base[i]      = NULL;
        tq_sink_fp16_base[i]   = NULL;
        tq_sink_k_stride[i]    = 0;
        tq_sink_fp16_stride[i] = 0;
        tq_sink_kv_size[i]     = 0;
    }
    tq_sink_n_global = 0;
}

void tq_set_current_layer(int layer, int is_k) {
    tq_cur_layer = (layer >= 0 && layer < TQ_MAX_LAYERS) ? layer : 0;
    tq_cur_is_k  = is_k;
}

void tq_set_current_head(int head) {
    tq_cur_head = (head >= 0 && head < TQ_MAX_HEADS) ? head : 0;
}

int tq_is_calibrating(void) {
    return tq_calibration_active;
}

void tq_accumulate_channels(int layer, int is_k, const float * x, int64_t k) {
    if (layer < 0 || layer >= TQ_MAX_LAYERS) return;
    const int64_t nb = k / TQ_DIM;
    int * count = is_k ? &tq_k_accum_n[layer] : &tq_v_accum_n[layer];
    for (int64_t b = 0; b < nb && b < TQ_MAX_HEADS; b++) {
        float * accum = is_k ? tq_k_accum[layer][b] : tq_v_accum[layer][b];
        const float * xb = x + b * TQ_DIM;
        for (int i = 0; i < TQ_DIM; i++) {
            accum[i] += fabsf(xb[i]);
        }
    }
    (*count)++; // count tokens, not blocks
}

// Detect outlier channels from accumulated magnitudes for one (layer, head, is_k)
static void tq_detect_outliers_for_head(int layer, int head, int is_k) {
    float * accum   = is_k ? tq_k_accum[layer][head]       : tq_v_accum[layer][head];
    int   * outlier = is_k ? tq_k_outlier_reg[layer][head]  : tq_v_outlier_reg[layer][head];
    int   * regular = is_k ? tq_k_regular_reg[layer][head]  : tq_v_regular_reg[layer][head];

    // Check if any data accumulated for this head
    float total = 0.0f;
    for (int i = 0; i < TQ_DIM; i++) total += accum[i];
    if (total == 0.0f) return; // no data — keep defaults

    // Sort channels by accumulated magnitude (descending)
    int order[TQ_DIM];
    for (int i = 0; i < TQ_DIM; i++) order[i] = i;
    for (int i = 1; i < TQ_DIM; i++) {
        int key_idx = order[i]; float key_mag = accum[key_idx];
        int j = i - 1;
        while (j >= 0 && accum[order[j]] < key_mag) { order[j+1] = order[j]; j--; }
        order[j+1] = key_idx;
    }

    // Top 32 by magnitude = outliers
    for (int i = 0; i < TQ_DIM_HI; i++) outlier[i] = order[i];
    // Sort outlier indices ascending for consistent ordering
    for (int i = 1; i < TQ_DIM_HI; i++) {
        int key = outlier[i]; int j = i - 1;
        while (j >= 0 && outlier[j] > key) { outlier[j+1] = outlier[j]; j--; }
        outlier[j+1] = key;
    }
    // Remaining = regular channels
    int ri = 0;
    for (int i = 0; i < TQ_DIM; i++) {
        int is_outlier_ch = 0;
        for (int j = 0; j < TQ_DIM_HI; j++) { if (outlier[j] == i) { is_outlier_ch = 1; break; } }
        if (!is_outlier_ch) regular[ri++] = i;
    }
}

int tq_min_accum_count(int n_layers) {
    int min_count = 0x7FFFFFFF;
    for (int l = 0; l < n_layers && l < TQ_MAX_LAYERS; l++) {
        // Skip layers with no data (filtered/unused layers, or MLA with no V cache)
        if (tq_k_accum_n[l] == 0 && tq_v_accum_n[l] == 0) continue;
        // Only include K/V counts if that cache actually has data (MLA has no V)
        if (tq_k_accum_n[l] > 0 && tq_k_accum_n[l] < min_count) min_count = tq_k_accum_n[l];
        if (tq_v_accum_n[l] > 0 && tq_v_accum_n[l] < min_count) min_count = tq_v_accum_n[l];
    }
    return min_count == 0x7FFFFFFF ? 0 : min_count;
}

void tq_lock_outliers_from_accum(int n_layers) {
    tq_init_rotations();
    for (int l = 0; l < n_layers && l < TQ_MAX_LAYERS; l++) {
        for (int h = 0; h < TQ_MAX_HEADS; h++) {
            tq_detect_outliers_for_head(l, h, 1); // K
            tq_detect_outliers_for_head(l, h, 0); // V
        }
        tq_layer_calibrated[l] = 1;
    }
    tq_calibration_active = 0;

    // Debug: print outlier channels for first 2 layers, all active heads
    fprintf(stderr, "tq_lock: %d layers, %d tokens accumulated (K layer0)\n",
            n_layers, tq_k_accum_n[0]);
    for (int l = 0; l < 2 && l < n_layers; l++) {
        for (int h = 0; h < 8; h++) {
            // Check if this head has data
            float total = 0.0f;
            for (int i = 0; i < TQ_DIM; i++) total += tq_k_accum[l][h][i];
            if (total == 0.0f) break;
            fprintf(stderr, "  K layer=%d head=%d outliers: [", l, h);
            for (int i = 0; i < 8; i++) fprintf(stderr, "%d ", tq_k_outlier_reg[l][h][i]);
            fprintf(stderr, "...]\n");
        }
    }
}

void tq_reset_calibration(void) {
    memset(tq_k_accum, 0, sizeof(tq_k_accum));
    memset(tq_v_accum, 0, sizeof(tq_v_accum));
    memset(tq_k_accum_n, 0, sizeof(tq_k_accum_n));
    memset(tq_v_accum_n, 0, sizeof(tq_v_accum_n));
    memset(tq_layer_calibrated, 0, sizeof(tq_layer_calibrated));
    tq_calibration_active = 1;
    for (int l = 0; l < TQ_MAX_LAYERS; l++) {
        for (int h = 0; h < TQ_MAX_HEADS; h++) {
            for (int i = 0; i < TQ_DIM_HI; i++) {
                tq_k_outlier_reg[l][h][i] = i;
                tq_v_outlier_reg[l][h][i] = i;
            }
            for (int i = 0; i < TQ_DIM_LO; i++) {
                tq_k_regular_reg[l][h][i] = TQ_DIM_HI + i;
                tq_v_regular_reg[l][h][i] = TQ_DIM_HI + i;
            }
        }
    }
}

void tq_reset_accumulators(void) {
    memset(tq_k_accum, 0, sizeof(tq_k_accum));
    memset(tq_v_accum, 0, sizeof(tq_v_accum));
    memset(tq_k_accum_n, 0, sizeof(tq_k_accum_n));
    memset(tq_v_accum_n, 0, sizeof(tq_v_accum_n));
}

// ---------------------------------------------------------------------------
// Public accessors for rotation matrices and channel maps (used by Metal/CUDA)
// ---------------------------------------------------------------------------

const float * tq_get_rot_v_fwd(void)  { tq_init_rotations(); return tq_rot_v_fwd; }
const float * tq_get_rot_v_inv(void)  { tq_init_rotations(); return tq_rot_v_inv; }
const float * tq_get_rot_hi_fwd(void) { tq_init_rotations(); return tq_rot_hi_fwd; }
const float * tq_get_rot_hi_inv(void) { tq_init_rotations(); return tq_rot_hi_inv; }
const float * tq_get_rot_lo_fwd(void) { tq_init_rotations(); return tq_rot_lo_fwd; }
const float * tq_get_rot_lo_inv(void) { tq_init_rotations(); return tq_rot_lo_inv; }

int tq_get_rot_v_size(void)  { return TQ_DIM * TQ_DIM; }
int tq_get_rot_hi_size(void) { return TQ_DIM_HI * TQ_DIM_HI; }
int tq_get_rot_lo_size(void) { return TQ_DIM_LO * TQ_DIM_LO; }

void tq_get_channel_map(int layer, int head, int is_k, int * outlier, int * regular) {
    tq_init_rotations();
    if (layer < 0 || layer >= TQ_MAX_LAYERS || head < 0 || head >= TQ_MAX_HEADS) return;
    const int * src_out = is_k ? tq_k_outlier_reg[layer][head] : tq_v_outlier_reg[layer][head];
    const int * src_reg = is_k ? tq_k_regular_reg[layer][head] : tq_v_regular_reg[layer][head];
    memcpy(outlier, src_out, TQ_DIM_HI * sizeof(int));
    memcpy(regular, src_reg, TQ_DIM_LO * sizeof(int));
}

void tq_get_qjl_matrix(float * out, int dim, uint64_t seed) {
    // Generate the same i.i.d. Gaussian matrix used by QJL forward/inverse
    uint64_t st = seed;
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            st = st * 6364136223846793005ULL + 1442695040888963407ULL;
            float u1 = ((float)(uint32_t)(st >> 32) + 1.0f) / ((float)0xFFFFFFFF + 2.0f);
            st = st * 6364136223846793005ULL + 1442695040888963407ULL;
            float u2 = ((float)(uint32_t)(st >> 32) + 1.0f) / ((float)0xFFFFFFFF + 2.0f);
            out[i * dim + j] = sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
        }
    }
}

// ---------------------------------------------------------------------------
// Fast Walsh-Hadamard Transform (FWHT) — in-place, normalized, self-inverse
// n must be a power of 2.  H_n = (1/√n) * Walsh-Hadamard matrix.
// ---------------------------------------------------------------------------

static void tq_fwht(float * x, int n) {
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
// TQK had_mse4: H_128 Hadamard + 4-bit MSE, no split, no calibration
// ---------------------------------------------------------------------------

void quantize_row_tqk_had_mse4_ref(const float * GGML_RESTRICT x, block_tqk_had_mse4 * GGML_RESTRICT y, int64_t k) {
    assert(k % TQK_BLOCK_SIZE == 0);
    const int64_t nb = k / TQK_BLOCK_SIZE;

    for (int64_t i = 0; i < nb; i++) {
        const float * xb = x + i * TQK_BLOCK_SIZE;

        float sum_sq = 0.0f;
        for (int j = 0; j < TQ_DIM; j++) sum_sq += xb[j] * xb[j];
        float norm = sqrtf(sum_sq);
        y[i].norm = GGML_FP32_TO_FP16(norm);

        if (norm == 0.0f) { memset(y[i].qs, 0, sizeof(y[i].qs)); continue; }
        float inv = 1.0f / norm;

        float rot[TQ_DIM];
        for (int j = 0; j < TQ_DIM; j++) rot[j] = xb[j] * inv;
        tq_fwht(rot, TQ_DIM);

        memset(y[i].qs, 0, sizeof(y[i].qs));
        for (int j = 0; j < TQ_DIM; j++) {
            int idx = nearest(rot[j], centroids_16, 16);
            pk4(y[i].qs, j, idx);
        }
    }

    // Write fp16 copy to k_sink tensor (only for sink positions)
    tq_sink_write_fp16(x, y, k, sizeof(block_tqk_had_mse4));
}

void dequantize_row_tqk_had_mse4(const block_tqk_had_mse4 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % TQK_BLOCK_SIZE == 0);
    const int64_t nb = k / TQK_BLOCK_SIZE;

    for (int64_t i = 0; i < nb; i++) {
        // Check if this block is a sink — read fp16 from k_sink tensor instead
        if (tq_try_sink_dequant(&x[i], y + i * TQK_BLOCK_SIZE, TQK_BLOCK_SIZE)) continue;

        float norm = GGML_FP16_TO_FP32(x[i].norm);
        float rot[TQ_DIM];
        for (int j = 0; j < TQ_DIM; j++) {
            rot[j] = centroids_16[up4(x[i].qs, j)];
        }
        tq_fwht(rot, TQ_DIM);
        for (int j = 0; j < TQ_DIM; j++) y[i * TQK_BLOCK_SIZE + j] = norm * rot[j];
    }
}

// Asymmetric vec_dot: rotate Q with H_128, dot with stored centroids
void ggml_vec_dot_tqk_had_mse4_f32(
        int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, size_t bx,
        const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % TQK_BLOCK_SIZE == 0);
    assert(nrc == 1);
    (void)bs; (void)bx; (void)by; (void)nrc;

    const block_tqk_had_mse4 * GGML_RESTRICT x = (const block_tqk_had_mse4 *)vx;
    const float * GGML_RESTRICT y = (const float *)vy;
    const int64_t nb = n / TQK_BLOCK_SIZE;

    float sumf = 0.0f;

    for (int64_t i = 0; i < nb; i++) {
        const float * q = y + i * TQK_BLOCK_SIZE;

        // Check if this block is a sink — do fp16 dot from k_sink tensor
        float sink_dot;
        if (tq_try_sink_dot(&x[i], q, TQK_BLOCK_SIZE, &sink_dot)) {
            sumf += sink_dot;
            continue;
        }

        float norm = GGML_FP16_TO_FP32(x[i].norm);

        // Rotate Q with H_128
        float q_rot[TQ_DIM];
        for (int j = 0; j < TQ_DIM; j++) q_rot[j] = q[j];
        tq_fwht(q_rot, TQ_DIM);

        // Dot with stored centroids
        float dot = 0.0f;
        for (int j = 0; j < TQ_DIM; j++) {
            dot += q_rot[j] * centroids_16[up4(x[i].qs, j)];
        }

        sumf += norm * dot;
    }

    *s = sumf;
}

// ---------------------------------------------------------------------------
// TQK had_prod4: H_128 Hadamard + 4-bit MSE + 1-bit QJL (unbiased estimator)
// No split, no calibration. QJL on residual corrects MSE bias.
// ---------------------------------------------------------------------------

void quantize_row_tqk_had_prod4_ref(const float * GGML_RESTRICT x, block_tqk_had_prod4 * GGML_RESTRICT y, int64_t k) {
    assert(k % TQK_BLOCK_SIZE == 0);
    const int64_t nb = k / TQK_BLOCK_SIZE;

    for (int64_t i = 0; i < nb; i++) {
        const float * xb = x + i * TQK_BLOCK_SIZE;

        // L2 norm
        float sum_sq = 0.0f;
        for (int j = 0; j < TQ_DIM; j++) sum_sq += xb[j] * xb[j];
        float norm = sqrtf(sum_sq);
        y[i].norm = GGML_FP32_TO_FP16(norm);
        y[i].rnorm = GGML_FP32_TO_FP16(0.0f);
        memset(y[i].signs, 0, sizeof(y[i].signs));

        if (norm == 0.0f) { memset(y[i].qs, 0, sizeof(y[i].qs)); continue; }
        float inv = 1.0f / norm;

        // Normalize and apply H_128 via FWHT
        float rot[TQ_DIM];
        for (int j = 0; j < TQ_DIM; j++) rot[j] = xb[j] * inv;
        tq_fwht(rot, TQ_DIM);

        // 4-bit MSE quantize (same as had_mse4)
        memset(y[i].qs, 0, sizeof(y[i].qs));
        for (int j = 0; j < TQ_DIM; j++) {
            int idx = nearest(rot[j], centroids_16, 16);
            pk4(y[i].qs, j, idx);
        }

        // Compute residual in ORIGINAL space: r = x - norm * H^{-1} * centroids
        float recon[TQ_DIM];
        for (int j = 0; j < TQ_DIM; j++) recon[j] = centroids_16[up4(y[i].qs, j)];
        tq_fwht(recon, TQ_DIM);  // inverse FWHT = FWHT (orthogonal, self-inverse)
        float resid[TQ_DIM];
        for (int j = 0; j < TQ_DIM; j++) resid[j] = xb[j] - norm * recon[j];

        // QJL forward on residual
        y[i].rnorm = GGML_FP32_TO_FP16(qjl_forward(resid, y[i].signs, TQ_DIM, QJL_SEED_128));
    }

    // Write fp16 copy to k_sink tensor (only for sink positions)
    tq_sink_write_fp16(x, y, k, sizeof(block_tqk_had_prod4));
}

void dequantize_row_tqk_had_prod4(const block_tqk_had_prod4 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % TQK_BLOCK_SIZE == 0);
    const int64_t nb = k / TQK_BLOCK_SIZE;

    for (int64_t i = 0; i < nb; i++) {
        // Check if this block is a sink — read fp16 from k_sink tensor instead
        if (tq_try_sink_dequant(&x[i], y + i * TQK_BLOCK_SIZE, TQK_BLOCK_SIZE)) continue;

        float norm = GGML_FP16_TO_FP32(x[i].norm);

        // MSE reconstruction: centroids → inverse FWHT → scale by norm
        float rot[TQ_DIM];
        for (int j = 0; j < TQ_DIM; j++) {
            rot[j] = centroids_16[up4(x[i].qs, j)];
        }
        tq_fwht(rot, TQ_DIM);
        for (int j = 0; j < TQ_DIM; j++) y[i * TQK_BLOCK_SIZE + j] = norm * rot[j];

        // QJL correction on residual
        float corr[TQ_DIM];
        qjl_inverse(x[i].signs, GGML_FP16_TO_FP32(x[i].rnorm), corr, TQ_DIM, QJL_SEED_128);
        for (int j = 0; j < TQ_DIM; j++) y[i * TQK_BLOCK_SIZE + j] += corr[j];
    }
}

// Asymmetric vec_dot: rotate Q with H_128, dot with stored centroids + QJL correction
void ggml_vec_dot_tqk_had_prod4_f32(
        int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, size_t bx,
        const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % TQK_BLOCK_SIZE == 0);
    assert(nrc == 1);
    (void)bs; (void)bx; (void)by; (void)nrc;

    const block_tqk_had_prod4 * GGML_RESTRICT x = (const block_tqk_had_prod4 *)vx;
    const float * GGML_RESTRICT y = (const float *)vy;
    const int64_t nb = n / TQK_BLOCK_SIZE;

    float sumf = 0.0f;

    for (int64_t i = 0; i < nb; i++) {
        const float * q = y + i * TQK_BLOCK_SIZE;

        // Check if this block is a sink — do fp16 dot from k_sink tensor
        float sink_dot;
        if (tq_try_sink_dot(&x[i], q, TQK_BLOCK_SIZE, &sink_dot)) {
            sumf += sink_dot;
            continue;
        }

        float norm = GGML_FP16_TO_FP32(x[i].norm);

        // Rotate Q with H_128
        float q_rot[TQ_DIM];
        for (int j = 0; j < TQ_DIM; j++) q_rot[j] = q[j];
        tq_fwht(q_rot, TQ_DIM);

        // MSE centroid dot
        float dot = 0.0f;
        for (int j = 0; j < TQ_DIM; j++) {
            dot += q_rot[j] * centroids_16[up4(x[i].qs, j)];
        }
        float mse_dot = norm * dot;

        // QJL asymmetric dot on residual (raw query, not rotated)
        float qjl_dot = qjl_asymmetric_dot(q, TQ_DIM, QJL_SEED_128,
                                            x[i].signs, GGML_FP16_TO_FP32(x[i].rnorm));

        sumf += mse_dot + qjl_dot;
    }

    *s = sumf;
}

// ---------------------------------------------------------------------------
// TQK 5hi_3lo_qr: 32/96 split, QR rotation, 4-bit MSE + QJL hi, 3-bit MSE lo
// ---------------------------------------------------------------------------

void quantize_row_tqk_5hi_3lo_qr_ref(const float * GGML_RESTRICT x, block_tqk_5hi_3lo * GGML_RESTRICT y, int64_t k) {
    assert(k % TQK_BLOCK_SIZE == 0);
    const int64_t nb = k / TQK_BLOCK_SIZE;
    tq_init_rotations();

    for (int64_t i = 0; i < nb; i++) {
        if (nb > 1) tq_cur_head = (int)(i % TQ_MAX_HEADS);
        const float * xb = x + i * TQK_BLOCK_SIZE;

        // Split into outlier/regular
        float hi_raw[TQ_DIM_HI], lo_raw[TQ_DIM_LO];
        tq_split_channels(xb, hi_raw, lo_raw);

        // QR rotation
        float hi_rot[TQ_DIM_HI], lo_rot[TQ_DIM_LO];
        tq_rotate_hi(hi_raw, hi_rot);
        tq_rotate_lo(lo_raw, lo_rot);

        // Per-subset norms
        float sum_hi = 0.0f, sum_lo = 0.0f;
        for (int j = 0; j < TQ_DIM_HI; j++) sum_hi += hi_rot[j] * hi_rot[j];
        for (int j = 0; j < TQ_DIM_LO; j++) sum_lo += lo_rot[j] * lo_rot[j];
        float norm_hi = sqrtf(sum_hi), norm_lo = sqrtf(sum_lo);

        y[i].norm_hi  = GGML_FP32_TO_FP16(norm_hi);
        y[i].norm_lo  = GGML_FP32_TO_FP16(norm_lo);
        y[i].rnorm_hi = GGML_FP32_TO_FP16(0.0f);
        memset(y[i].signs_hi, 0, sizeof(y[i].signs_hi));

        if (norm_hi == 0.0f && norm_lo == 0.0f) {
            memset(y[i].qs_hi, 0, sizeof(y[i].qs_hi));
            memset(y[i].qs_lo, 0, sizeof(y[i].qs_lo));
            continue;
        }

        float inv_hi = (norm_hi > 1e-12f) ? 1.0f / norm_hi : 0.0f;
        float inv_lo = (norm_lo > 1e-12f) ? 1.0f / norm_lo : 0.0f;

        // 4-bit MSE for hi (16 centroids, d=32)
        memset(y[i].qs_hi, 0, sizeof(y[i].qs_hi));
        for (int j = 0; j < TQ_DIM_HI; j++) {
            float xn = hi_rot[j] * inv_hi;
            int idx = nearest(xn, centroids_16_d32, 16);
            pk4(y[i].qs_hi, j, idx);
        }

        // 3-bit MSE for lo (8 centroids, d=96)
        memset(y[i].qs_lo, 0, sizeof(y[i].qs_lo));
        for (int j = 0; j < TQ_DIM_LO; j++) {
            float xn = lo_rot[j] * inv_lo;
            int idx = nearest(xn, centroids_8_d96, 8);
            pk3(y[i].qs_lo, j, idx);
        }

        // QJL on hi residual (in original subset space)
        float yhi[TQ_DIM_HI];
        for (int j = 0; j < TQ_DIM_HI; j++) yhi[j] = centroids_16_d32[up4(y[i].qs_hi, j)];
        float hi_rec[TQ_DIM_HI];
        tq_unrotate_hi(yhi, hi_rec);
        float r_hi[TQ_DIM_HI];
        for (int j = 0; j < TQ_DIM_HI; j++) r_hi[j] = hi_raw[j] - norm_hi * hi_rec[j];
        y[i].rnorm_hi = GGML_FP32_TO_FP16(qjl_forward(r_hi, y[i].signs_hi, TQ_DIM_HI, QJL_SEED_32));
    }

    // Write fp16 copy to k_sink tensor (only for sink positions)
    tq_sink_write_fp16(x, y, k, sizeof(block_tqk_5hi_3lo));
}

void dequantize_row_tqk_5hi_3lo_qr(const block_tqk_5hi_3lo * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % TQK_BLOCK_SIZE == 0);
    const int64_t nb = k / TQK_BLOCK_SIZE;
    tq_init_rotations();

    for (int64_t i = 0; i < nb; i++) {
        // Check if this block is a sink — read fp16 from k_sink tensor instead
        if (tq_try_sink_dequant(&x[i], y + i * TQK_BLOCK_SIZE, TQK_BLOCK_SIZE)) continue;

        if (nb > 1) tq_cur_head = (int)(i % TQ_MAX_HEADS);
        float norm_hi = GGML_FP16_TO_FP32(x[i].norm_hi);
        float norm_lo = GGML_FP16_TO_FP32(x[i].norm_lo);

        // MSE recon: centroids → unrotate → scale
        float yhi[TQ_DIM_HI], ylo[TQ_DIM_LO];
        for (int j = 0; j < TQ_DIM_HI; j++) yhi[j] = centroids_16_d32[up4(x[i].qs_hi, j)];
        for (int j = 0; j < TQ_DIM_LO; j++) ylo[j] = centroids_8_d96[up3(x[i].qs_lo, j)];

        float hi_orig[TQ_DIM_HI], lo_orig[TQ_DIM_LO];
        tq_unrotate_hi(yhi, hi_orig);
        tq_unrotate_lo(ylo, lo_orig);
        for (int j = 0; j < TQ_DIM_HI; j++) hi_orig[j] *= norm_hi;
        for (int j = 0; j < TQ_DIM_LO; j++) lo_orig[j] *= norm_lo;

        // QJL correction on hi
        float corr_hi[TQ_DIM_HI];
        qjl_inverse(x[i].signs_hi, GGML_FP16_TO_FP32(x[i].rnorm_hi), corr_hi, TQ_DIM_HI, QJL_SEED_32);
        for (int j = 0; j < TQ_DIM_HI; j++) hi_orig[j] += corr_hi[j];

        tq_merge_channels(hi_orig, lo_orig, y + i * TQK_BLOCK_SIZE);
    }
}

void ggml_vec_dot_tqk_5hi_3lo_qr_f32(
        int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, size_t bx,
        const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % TQK_BLOCK_SIZE == 0);
    assert(nrc == 1);
    (void)bs; (void)bx; (void)by; (void)nrc;

    tq_init_rotations();

    const block_tqk_5hi_3lo * GGML_RESTRICT x = (const block_tqk_5hi_3lo *)vx;
    const float * GGML_RESTRICT y = (const float *)vy;
    const int64_t nb = n / TQK_BLOCK_SIZE;

    float sumf = 0.0f;

    for (int64_t i = 0; i < nb; i++) {
        const float * q = y + i * TQK_BLOCK_SIZE;

        // Check if this block is a sink — do fp16 dot from k_sink tensor
        float sink_dot;
        if (tq_try_sink_dot(&x[i], q, TQK_BLOCK_SIZE, &sink_dot)) {
            sumf += sink_dot;
            continue;
        }

        if (nb > 1) tq_cur_head = (int)(i % TQ_MAX_HEADS);
        float norm_hi = GGML_FP16_TO_FP32(x[i].norm_hi);
        float norm_lo = GGML_FP16_TO_FP32(x[i].norm_lo);

        // Split query and QR-rotate
        float hi_raw[TQ_DIM_HI], lo_raw[TQ_DIM_LO];
        tq_split_channels(q, hi_raw, lo_raw);
        float q_rot_hi[TQ_DIM_HI], q_rot_lo[TQ_DIM_LO];
        tq_rotate_hi(hi_raw, q_rot_hi);
        tq_rotate_lo(lo_raw, q_rot_lo);

        // MSE centroid dot per subset
        float mse_dot_hi = 0.0f, mse_dot_lo = 0.0f;
        for (int j = 0; j < TQ_DIM_HI; j++) {
            mse_dot_hi += q_rot_hi[j] * centroids_16_d32[up4(x[i].qs_hi, j)];
        }
        for (int j = 0; j < TQ_DIM_LO; j++) {
            mse_dot_lo += q_rot_lo[j] * centroids_8_d96[up3(x[i].qs_lo, j)];
        }
        float mse_dot = mse_dot_hi * norm_hi + mse_dot_lo * norm_lo;

        // QJL correction on hi (raw query, not rotated)
        float qjl_hi = qjl_asymmetric_dot(hi_raw, TQ_DIM_HI, QJL_SEED_32,
                                           x[i].signs_hi, GGML_FP16_TO_FP32(x[i].rnorm_hi));

        sumf += mse_dot + qjl_hi;
    }

    *s = sumf;
}

// ---------------------------------------------------------------------------
// TQK 5hi_3lo_fwht: 32/96 split, FWHT rotation, 4-bit MSE + QJL hi, 3-bit MSE lo
// Hi uses H_32 FWHT. Lo uses 3 × H_32 FWHT (block-diagonal on 96=3×32).
// Lo centroids use d=32 (each 32-dim block is independent).
// ---------------------------------------------------------------------------

void quantize_row_tqk_5hi_3lo_fwht_ref(const float * GGML_RESTRICT x, block_tqk_5hi_3lo * GGML_RESTRICT y, int64_t k) {
    assert(k % TQK_BLOCK_SIZE == 0);
    const int64_t nb = k / TQK_BLOCK_SIZE;
    tq_init_rotations();

    for (int64_t i = 0; i < nb; i++) {
        if (nb > 1) tq_cur_head = (int)(i % TQ_MAX_HEADS);
        const float * xb = x + i * TQK_BLOCK_SIZE;

        // Split into outlier/regular
        float hi_raw[TQ_DIM_HI], lo_raw[TQ_DIM_LO];
        tq_split_channels(xb, hi_raw, lo_raw);

        // FWHT rotation: H_32 on hi, 3×H_32 on lo
        float hi_rot[TQ_DIM_HI], lo_rot[TQ_DIM_LO];
        memcpy(hi_rot, hi_raw, sizeof(hi_rot));
        tq_fwht(hi_rot, TQ_DIM_HI);
        memcpy(lo_rot, lo_raw, sizeof(lo_rot));
        tq_fwht(lo_rot,       TQ_DIM_HI);  // lo[0:32]
        tq_fwht(lo_rot + 32,  TQ_DIM_HI);  // lo[32:64]
        tq_fwht(lo_rot + 64,  TQ_DIM_HI);  // lo[64:96]

        // Per-subset norms
        float sum_hi = 0.0f, sum_lo = 0.0f;
        for (int j = 0; j < TQ_DIM_HI; j++) sum_hi += hi_rot[j] * hi_rot[j];
        for (int j = 0; j < TQ_DIM_LO; j++) sum_lo += lo_rot[j] * lo_rot[j];
        float norm_hi = sqrtf(sum_hi), norm_lo = sqrtf(sum_lo);

        y[i].norm_hi  = GGML_FP32_TO_FP16(norm_hi);
        y[i].norm_lo  = GGML_FP32_TO_FP16(norm_lo);
        y[i].rnorm_hi = GGML_FP32_TO_FP16(0.0f);
        memset(y[i].signs_hi, 0, sizeof(y[i].signs_hi));

        if (norm_hi == 0.0f && norm_lo == 0.0f) {
            memset(y[i].qs_hi, 0, sizeof(y[i].qs_hi));
            memset(y[i].qs_lo, 0, sizeof(y[i].qs_lo));
            continue;
        }

        float inv_hi = (norm_hi > 1e-12f) ? 1.0f / norm_hi : 0.0f;
        float inv_lo = (norm_lo > 1e-12f) ? 1.0f / norm_lo : 0.0f;

        // 4-bit MSE for hi (16 centroids, d=32)
        memset(y[i].qs_hi, 0, sizeof(y[i].qs_hi));
        for (int j = 0; j < TQ_DIM_HI; j++) {
            float xn = hi_rot[j] * inv_hi;
            int idx = nearest(xn, centroids_16_d32, 16);
            pk4(y[i].qs_hi, j, idx);
        }

        // 3-bit MSE for lo (8 centroids, d=32 — each 32-dim block independent)
        memset(y[i].qs_lo, 0, sizeof(y[i].qs_lo));
        for (int j = 0; j < TQ_DIM_LO; j++) {
            float xn = lo_rot[j] * inv_lo;
            int idx = nearest(xn, centroids_8_d32, 8);
            pk3(y[i].qs_lo, j, idx);
        }

        // QJL on hi residual (in original subset space)
        // Reconstruct: unrotate centroids via inverse FWHT
        float yhi[TQ_DIM_HI];
        for (int j = 0; j < TQ_DIM_HI; j++) yhi[j] = centroids_16_d32[up4(y[i].qs_hi, j)];
        float hi_rec[TQ_DIM_HI];
        memcpy(hi_rec, yhi, sizeof(hi_rec));
        tq_fwht(hi_rec, TQ_DIM_HI); // inverse = forward for normalized FWHT
        float r_hi[TQ_DIM_HI];
        for (int j = 0; j < TQ_DIM_HI; j++) r_hi[j] = hi_raw[j] - norm_hi * hi_rec[j];
        y[i].rnorm_hi = GGML_FP32_TO_FP16(qjl_forward(r_hi, y[i].signs_hi, TQ_DIM_HI, QJL_SEED_32));
    }

    // Write fp16 copy to k_sink tensor (only for sink positions)
    tq_sink_write_fp16(x, y, k, sizeof(block_tqk_5hi_3lo));
}

void dequantize_row_tqk_5hi_3lo_fwht(const block_tqk_5hi_3lo * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % TQK_BLOCK_SIZE == 0);
    const int64_t nb = k / TQK_BLOCK_SIZE;
    tq_init_rotations();

    for (int64_t i = 0; i < nb; i++) {
        // Check if this block is a sink — read fp16 from k_sink tensor instead
        if (tq_try_sink_dequant(&x[i], y + i * TQK_BLOCK_SIZE, TQK_BLOCK_SIZE)) continue;

        if (nb > 1) tq_cur_head = (int)(i % TQ_MAX_HEADS);
        float norm_hi = GGML_FP16_TO_FP32(x[i].norm_hi);
        float norm_lo = GGML_FP16_TO_FP32(x[i].norm_lo);

        // MSE recon: centroids → inverse FWHT → scale
        float yhi[TQ_DIM_HI], ylo[TQ_DIM_LO];
        for (int j = 0; j < TQ_DIM_HI; j++) yhi[j] = centroids_16_d32[up4(x[i].qs_hi, j)];
        for (int j = 0; j < TQ_DIM_LO; j++) ylo[j] = centroids_8_d32[up3(x[i].qs_lo, j)];

        // Inverse FWHT
        float hi_orig[TQ_DIM_HI], lo_orig[TQ_DIM_LO];
        memcpy(hi_orig, yhi, sizeof(hi_orig));
        tq_fwht(hi_orig, TQ_DIM_HI);
        memcpy(lo_orig, ylo, sizeof(lo_orig));
        tq_fwht(lo_orig,       TQ_DIM_HI);
        tq_fwht(lo_orig + 32,  TQ_DIM_HI);
        tq_fwht(lo_orig + 64,  TQ_DIM_HI);

        for (int j = 0; j < TQ_DIM_HI; j++) hi_orig[j] *= norm_hi;
        for (int j = 0; j < TQ_DIM_LO; j++) lo_orig[j] *= norm_lo;

        // QJL correction on hi
        float corr_hi[TQ_DIM_HI];
        qjl_inverse(x[i].signs_hi, GGML_FP16_TO_FP32(x[i].rnorm_hi), corr_hi, TQ_DIM_HI, QJL_SEED_32);
        for (int j = 0; j < TQ_DIM_HI; j++) hi_orig[j] += corr_hi[j];

        tq_merge_channels(hi_orig, lo_orig, y + i * TQK_BLOCK_SIZE);
    }
}

// Asymmetric vec_dot for 5hi_3lo_fwht
void ggml_vec_dot_tqk_5hi_3lo_fwht_f32(
        int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, size_t bx,
        const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % TQK_BLOCK_SIZE == 0);
    assert(nrc == 1);
    (void)bs; (void)bx; (void)by; (void)nrc;

    tq_init_rotations();

    const block_tqk_5hi_3lo * GGML_RESTRICT x = (const block_tqk_5hi_3lo *)vx;
    const float * GGML_RESTRICT y = (const float *)vy;
    const int64_t nb = n / TQK_BLOCK_SIZE;

    float sumf = 0.0f;

    for (int64_t i = 0; i < nb; i++) {
        const float * q = y + i * TQK_BLOCK_SIZE;

        // Check if this block is a sink — do fp16 dot from k_sink tensor
        float sink_dot;
        if (tq_try_sink_dot(&x[i], q, TQK_BLOCK_SIZE, &sink_dot)) {
            sumf += sink_dot;
            continue;
        }

        if (nb > 1) tq_cur_head = (int)(i % TQ_MAX_HEADS);
        float norm_hi = GGML_FP16_TO_FP32(x[i].norm_hi);
        float norm_lo = GGML_FP16_TO_FP32(x[i].norm_lo);

        // Split query and FWHT-rotate
        float hi_raw[TQ_DIM_HI], lo_raw[TQ_DIM_LO];
        tq_split_channels(q, hi_raw, lo_raw);

        float q_rot_hi[TQ_DIM_HI], q_rot_lo[TQ_DIM_LO];
        memcpy(q_rot_hi, hi_raw, sizeof(q_rot_hi));
        tq_fwht(q_rot_hi, TQ_DIM_HI);
        memcpy(q_rot_lo, lo_raw, sizeof(q_rot_lo));
        tq_fwht(q_rot_lo,       TQ_DIM_HI);
        tq_fwht(q_rot_lo + 32,  TQ_DIM_HI);
        tq_fwht(q_rot_lo + 64,  TQ_DIM_HI);

        // MSE centroid dot per subset
        float mse_dot_hi = 0.0f, mse_dot_lo = 0.0f;
        for (int j = 0; j < TQ_DIM_HI; j++) {
            mse_dot_hi += q_rot_hi[j] * centroids_16_d32[up4(x[i].qs_hi, j)];
        }
        for (int j = 0; j < TQ_DIM_LO; j++) {
            mse_dot_lo += q_rot_lo[j] * centroids_8_d32[up3(x[i].qs_lo, j)];
        }
        float mse_dot = mse_dot_hi * norm_hi + mse_dot_lo * norm_lo;

        // QJL correction on hi (raw query, not rotated)
        float qjl_hi = qjl_asymmetric_dot(hi_raw, TQ_DIM_HI, QJL_SEED_32,
                                           x[i].signs_hi, GGML_FP16_TO_FP32(x[i].rnorm_hi));

        sumf += mse_dot + qjl_hi;
    }

    *s = sumf;
}
