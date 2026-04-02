// TurboQuant CPU reference — FWHT decorrelation + MSE centroids + QJL correction
// Three types: TQL (layered 32+32+64), TQ3J (128×3bit+QJL), TQ2J (128×2bit+QJL)

#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#include "ggml-turbo-quant.h"
#include "ggml-impl.h"

#include <string.h>
#include <math.h>
#include <assert.h>

// sqrt(pi/2) ≈ 1.2533141 — E[|X|] for standard normal
#define QJL_SCALE 1.2533141f

// ---------------------------------------------------------------------------
// Precomputed Lloyd-Max centroids for Beta(d/2-0.5, d/2-0.5) distribution
// These are the optimal quantization levels for FWHT-normalized coefficients
// ---------------------------------------------------------------------------

// d=32, 3-bit (8 centroids) — Beta(15.5, 15.5)
static const float centroids_8_d32[8] = {
    -0.3662682422f, -0.2324605670f, -0.1317560968f, -0.0428515156f,
     0.0428515156f,  0.1317560968f,  0.2324605670f,  0.3662682422f,
};

// d=64, 2-bit (4 centroids) — Beta(31.5, 31.5)
static const float centroids_4_d64[4] = {
    -0.1874968494f, -0.0565148688f,  0.0565148688f,  0.1874968494f,
};

// d=128, 3-bit (8 centroids) — Beta(63.5, 63.5)
static const float centroids_8_d128[8] = {
    -0.1883971860f, -0.1181397670f, -0.0665856080f, -0.0216043106f,
     0.0216043106f,  0.0665856080f,  0.1181397670f,  0.1883971860f,
};

// d=128, 2-bit (4 centroids) — Beta(63.5, 63.5)
static const float centroids_4_d128[4] = {
    -0.1330415202f, -0.0399915952f, 0.0399915952f, 0.1330415202f,
};

// ---------------------------------------------------------------------------
// Per-channel permutation support
// ---------------------------------------------------------------------------

static uint8_t * tq_perms = NULL;    // [n_layers][n_heads][head_dim]
static int tq_n_layers  = 0;
static int tq_n_heads   = 0;
static int tq_head_dim  = 0;
static int32_t * tq_layer_map = NULL; // [n_model_layers] -> calibration layer idx
static int tq_n_model_layers = 0;

static _Thread_local int tq_cur_layer = 0;
static _Thread_local int tq_cur_head  = 0;
static _Thread_local int tq_cur_is_k  = 1;

void tq_init_perms(const uint8_t * perms, int n_layers, int n_heads, int head_dim,
                   const int32_t * layer_map, int n_model_layers) {
    size_t sz = (size_t)n_layers * n_heads * head_dim;
    tq_perms = (uint8_t *)malloc(sz);
    memcpy(tq_perms, perms, sz);
    tq_n_layers = n_layers;
    tq_n_heads  = n_heads;
    tq_head_dim = head_dim;
    tq_layer_map = (int32_t *)malloc((size_t)n_model_layers * sizeof(int32_t));
    memcpy(tq_layer_map, layer_map, (size_t)n_model_layers * sizeof(int32_t));
    tq_n_model_layers = n_model_layers;
}

void tq_free_perms(void) {
    free(tq_perms);       tq_perms = NULL;
    free(tq_layer_map);   tq_layer_map = NULL;
    tq_n_layers = tq_n_heads = tq_head_dim = tq_n_model_layers = 0;
}

void tq_set_current_layer(int layer, int is_k) { tq_cur_layer = layer; tq_cur_is_k = is_k; }
void tq_set_current_head(int head)             { tq_cur_head = head; }

// Get the permutation for current layer/head, or NULL if no perms loaded
static const uint8_t * tq_get_perm(void) {
    if (!tq_perms) return NULL;
    int cidx = (tq_cur_layer < tq_n_model_layers && tq_layer_map)
               ? tq_layer_map[tq_cur_layer] : tq_cur_layer;
    if (cidx < 0 || cidx >= tq_n_layers) return NULL;
    int h = tq_cur_head % tq_n_heads;
    return tq_perms + ((size_t)cidx * tq_n_heads + h) * tq_head_dim;
}

// Split 128-dim vector into hi(32) + mid(32) + low(64) using permutation
static void tq_split_3way(const float * x, float * hi, float * mid, float * low) {
    const uint8_t * perm = tq_get_perm();
    if (perm) {
        for (int i = 0; i < 32; i++)  hi[i]  = x[perm[i]];
        for (int i = 0; i < 32; i++)  mid[i] = x[perm[32 + i]];
        for (int i = 0; i < 64; i++)  low[i] = x[perm[64 + i]];
    } else {
        memcpy(hi,  x,      32 * sizeof(float));
        memcpy(mid, x + 32, 32 * sizeof(float));
        memcpy(low, x + 64, 64 * sizeof(float));
    }
}

// Merge hi(32) + mid(32) + low(64) back into 128-dim vector using permutation
static void tq_merge_3way(const float * hi, const float * mid, const float * low, float * x) {
    const uint8_t * perm = tq_get_perm();
    if (perm) {
        for (int i = 0; i < 32; i++)  x[perm[i]]      = hi[i];
        for (int i = 0; i < 32; i++)  x[perm[32 + i]]  = mid[i];
        for (int i = 0; i < 64; i++)  x[perm[64 + i]]  = low[i];
    } else {
        memcpy(x,      hi,  32 * sizeof(float));
        memcpy(x + 32, mid, 32 * sizeof(float));
        memcpy(x + 64, low, 64 * sizeof(float));
    }
}

// ---------------------------------------------------------------------------
// FWHT: in-place Fast Walsh-Hadamard Transform (normalized, self-inverse)
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
// Bit packing helpers
// ---------------------------------------------------------------------------

static inline int nearest(float val, const float * c, int n) {
    int best = 0;
    float best_d = fabsf(val - c[0]);
    for (int i = 1; i < n; i++) {
        float d = fabsf(val - c[i]);
        if (d < best_d) { best_d = d; best = i; }
    }
    return best;
}

static inline void pk3(uint8_t * q, int j, int v) {
    int bp = j * 3, bi = bp / 8, sh = bp % 8;
    q[bi] |= (uint8_t)((v & 7) << sh);
    if (sh > 5) q[bi + 1] |= (uint8_t)((v & 7) >> (8 - sh));
}

static inline int up3(const uint8_t * q, int j) {
    int bp = j * 3, bi = bp / 8, sh = bp % 8;
    uint32_t val = (uint32_t)q[bi];
    if (sh > 5) val |= (uint32_t)q[bi + 1] << 8;
    return (int)((val >> sh) & 7);
}

static inline void pk2(uint8_t * q, int j, int v) {
    q[j / 4] |= (uint8_t)((v & 3) << (2 * (j % 4)));
}

static inline int up2(const uint8_t * q, int j) {
    return (q[j / 4] >> (2 * (j % 4))) & 3;
}

// ---------------------------------------------------------------------------
// Shared: MSE quantize + QJL in FWHT space
// ---------------------------------------------------------------------------

// Quantize a segment: FWHT, normalize, find nearest centroids, compute QJL residual
// Returns: norm (L2 norm of FWHT coefficients)
// Writes: qs (centroid indices), signs (QJL sign bits), rnorm (residual L2 norm)
static float tq_segment_quantize(
    const float * x, int n,
    const float * centroids, int n_c,
    void (*pk)(uint8_t *, int, int),
    uint8_t * qs, int qs_bytes,
    uint8_t * signs, int signs_bytes,
    float * rnorm_out,
    float * fwht_buf)
{
    // Copy and apply FWHT
    memcpy(fwht_buf, x, (size_t)n * sizeof(float));
    tq_fwht(fwht_buf, n);

    // Compute L2 norm
    float norm_sq = 0;
    for (int j = 0; j < n; j++) norm_sq += fwht_buf[j] * fwht_buf[j];
    float norm = sqrtf(norm_sq);
    float inv_norm = (norm > 1e-12f) ? 1.0f / norm : 0.0f;

    // MSE quantize: find nearest centroid for each normalized coefficient
    memset(qs, 0, qs_bytes);
    for (int j = 0; j < n; j++) {
        pk(qs, j, nearest(fwht_buf[j] * inv_norm, centroids, n_c));
    }

    // QJL: compute residual in FWHT space, store signs
    memset(signs, 0, signs_bytes);
    float rnorm_sq = 0;
    for (int j = 0; j < n; j++) {
        int idx;
        if (n_c == 8) idx = up3(qs, j);
        else          idx = up2(qs, j);
        float residual = fwht_buf[j] - norm * centroids[idx];
        rnorm_sq += residual * residual;
        if (residual >= 0.0f) signs[j / 8] |= (uint8_t)(1 << (j % 8));
    }
    *rnorm_out = sqrtf(rnorm_sq);
    return norm;
}

// Dequantize a segment: reconstruct FWHT coefficients + QJL correction, then inverse FWHT
static void tq_segment_dequantize(
    float * out, int n,
    const float * centroids,
    int (*up)(const uint8_t *, int),
    const uint8_t * qs,
    const uint8_t * signs,
    float norm, float rnorm)
{
    float qjl_scale = QJL_SCALE / (float)n * rnorm;
    for (int j = 0; j < n; j++) {
        float val = norm * centroids[up(qs, j)];
        float sign = ((signs[j / 8] >> (j % 8)) & 1) ? 1.0f : -1.0f;
        out[j] = val + qjl_scale * sign;
    }
    tq_fwht(out, n);  // inverse FWHT (self-inverse)
}

// Vec_dot for a segment in FWHT space: centroid dot + QJL dot
static float tq_segment_vec_dot(
    const float * q_fwht, int n,
    const float * centroids,
    int (*up)(const uint8_t *, int),
    const uint8_t * qs,
    const uint8_t * signs,
    float norm, float rnorm)
{
    float mse_dot = 0;
    for (int j = 0; j < n; j++) {
        mse_dot += q_fwht[j] * centroids[up(qs, j)];
    }
    mse_dot *= norm;

    float qjl_dot = 0;
    for (int j = 0; j < n; j++) {
        float sign = ((signs[j / 8] >> (j % 8)) & 1) ? 1.0f : -1.0f;
        qjl_dot += q_fwht[j] * sign;
    }
    qjl_dot *= QJL_SCALE / (float)n * rnorm;

    return mse_dot + qjl_dot;
}

// ===========================================================================
// TQL: Layered 32×q8 + 32×(3mse+1qjl) + 64×(2mse+1qjl)
// ===========================================================================

void quantize_row_tql_ref(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TQL == 0);
    const int64_t nb = k / QK_TQL;
    block_tql * GGML_RESTRICT blk = (block_tql *)y;

    for (int64_t i = 0; i < nb; i++) {
        const float * xb = x + i * QK_TQL;
        block_tql * b = &blk[i];
        memset(b, 0, sizeof(block_tql));

        if (nb > 1) tq_cur_head = (int)(i % (tq_n_heads > 0 ? tq_n_heads : nb));

        // Split into 3 segments using channel permutation (or sequential if no perms)
        float hi_raw[32], mid_raw[32], low_raw[64];
        tq_split_3way(xb, hi_raw, mid_raw, low_raw);

        // --- hi segment: FWHT-32 + q8 ---
        float hi_fwht[32];
        memcpy(hi_fwht, hi_raw, 32 * sizeof(float));
        tq_fwht(hi_fwht, 32);

        float amax = 0;
        for (int j = 0; j < 32; j++) {
            float a = fabsf(hi_fwht[j]);
            if (a > amax) amax = a;
        }
        float d = amax / 127.0f;
        b->d_hi = GGML_FP32_TO_FP16(d);
        float id = (d > 1e-12f) ? 127.0f / amax : 0.0f;
        for (int j = 0; j < 32; j++) {
            int v = (int)roundf(hi_fwht[j] * id);
            if (v >  127) v =  127;
            if (v < -128) v = -128;
            b->qs_hi[j] = (int8_t)v;
        }

        // --- mid segment: FWHT-32 + 3-bit centroids + QJL ---
        float mid_fwht[32];
        float rnorm_mid;
        float norm_mid = tq_segment_quantize(
            mid_raw, 32,
            centroids_8_d32, 8, pk3,
            b->qs_mid, sizeof(b->qs_mid),
            b->signs_mid, sizeof(b->signs_mid),
            &rnorm_mid, mid_fwht);
        b->norm_mid  = GGML_FP32_TO_FP16(norm_mid);
        b->rnorm_mid = GGML_FP32_TO_FP16(rnorm_mid);

        // --- low segment: FWHT-64 + 2-bit centroids + QJL ---
        float low_fwht[64];
        float rnorm_low;
        float norm_low = tq_segment_quantize(
            low_raw, 64,
            centroids_4_d64, 4, pk2,
            b->qs_low, sizeof(b->qs_low),
            b->signs_low, sizeof(b->signs_low),
            &rnorm_low, low_fwht);
        b->norm_low  = GGML_FP32_TO_FP16(norm_low);
        b->rnorm_low = GGML_FP32_TO_FP16(rnorm_low);
    }
}

void dequantize_row_tql(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TQL == 0);
    const int64_t nb = k / QK_TQL;
    const block_tql * GGML_RESTRICT blk = (const block_tql *)x;

    for (int64_t i = 0; i < nb; i++) {
        const block_tql * b = &blk[i];
        float * out = y + i * QK_TQL;

        if (nb > 1) tq_cur_head = (int)(i % (tq_n_heads > 0 ? tq_n_heads : nb));

        // hi: q8 dequant + inverse FWHT
        float hi[32];
        float d_hi = GGML_FP16_TO_FP32(b->d_hi);
        for (int j = 0; j < 32; j++) hi[j] = d_hi * (float)b->qs_hi[j];
        tq_fwht(hi, 32);

        // mid: centroid + QJL + inverse FWHT
        float mid[32];
        float norm_mid  = GGML_FP16_TO_FP32(b->norm_mid);
        float rnorm_mid = GGML_FP16_TO_FP32(b->rnorm_mid);
        tq_segment_dequantize(mid, 32, centroids_8_d32, up3,
                               b->qs_mid, b->signs_mid, norm_mid, rnorm_mid);

        // low: centroid + QJL + inverse FWHT
        float low[64];
        float norm_low  = GGML_FP16_TO_FP32(b->norm_low);
        float rnorm_low = GGML_FP16_TO_FP32(b->rnorm_low);
        tq_segment_dequantize(low, 64, centroids_4_d64, up2,
                               b->qs_low, b->signs_low, norm_low, rnorm_low);

        // Merge back using permutation
        tq_merge_3way(hi, mid, low, out);
    }
}

void ggml_vec_dot_tql_f32(int n, float * GGML_RESTRICT s, size_t bs,
                           const void * GGML_RESTRICT vx, size_t bx,
                           const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_TQL == 0);
    assert(nrc == 1);
    (void)bs; (void)bx; (void)by; (void)nrc;

    const block_tql * GGML_RESTRICT blk = (const block_tql *)vx;
    const float     * GGML_RESTRICT q   = (const float *)vy;
    const int64_t nb = n / QK_TQL;
    float sumf = 0;

    for (int64_t i = 0; i < nb; i++) {
        const block_tql * b = &blk[i];
        const float * qb = q + i * QK_TQL;

        if (nb > 1) tq_cur_head = (int)(i % (tq_n_heads > 0 ? tq_n_heads : nb));

        // Split Q using same permutation as K
        float q_hi_raw[32], q_mid_raw[32], q_low_raw[64];
        tq_split_3way(qb, q_hi_raw, q_mid_raw, q_low_raw);

        // hi: FWHT-32(Q_hi) · (d_hi * qs_hi)
        float q_hi[32];
        memcpy(q_hi, q_hi_raw, 32 * sizeof(float));
        tq_fwht(q_hi, 32);
        float d_hi = GGML_FP16_TO_FP32(b->d_hi);
        float dot_hi = 0;
        for (int j = 0; j < 32; j++) dot_hi += q_hi[j] * (float)b->qs_hi[j];
        dot_hi *= d_hi;

        // mid: centroid dot + QJL dot in FWHT space
        float q_mid[32];
        memcpy(q_mid, q_mid_raw, 32 * sizeof(float));
        tq_fwht(q_mid, 32);
        float norm_mid  = GGML_FP16_TO_FP32(b->norm_mid);
        float rnorm_mid = GGML_FP16_TO_FP32(b->rnorm_mid);
        float dot_mid = tq_segment_vec_dot(q_mid, 32, centroids_8_d32, up3,
                                            b->qs_mid, b->signs_mid, norm_mid, rnorm_mid);

        // low: centroid dot + QJL dot in FWHT space
        float q_low[64];
        memcpy(q_low, q_low_raw, 64 * sizeof(float));
        tq_fwht(q_low, 64);
        float norm_low  = GGML_FP16_TO_FP32(b->norm_low);
        float rnorm_low = GGML_FP16_TO_FP32(b->rnorm_low);
        float dot_low = tq_segment_vec_dot(q_low, 64, centroids_4_d64, up2,
                                            b->qs_low, b->signs_low, norm_low, rnorm_low);

        sumf += dot_hi + dot_mid + dot_low;
    }
    *s = sumf;
}

// ===========================================================================
// TQ3J: FWHT-128 + 3-bit MSE centroids + 1-bit QJL
// ===========================================================================

void quantize_row_tq3j_ref(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TQL == 0);
    const int64_t nb = k / QK_TQL;
    block_tq3j * GGML_RESTRICT blk = (block_tq3j *)y;

    for (int64_t i = 0; i < nb; i++) {
        const float * xb = x + i * QK_TQL;
        block_tq3j * b = &blk[i];
        memset(b, 0, sizeof(block_tq3j));

        float fwht_buf[128];
        float rnorm;
        float norm = tq_segment_quantize(
            xb, 128,
            centroids_8_d128, 8, pk3,
            b->qs, sizeof(b->qs),
            b->signs, sizeof(b->signs),
            &rnorm, fwht_buf);
        b->norm  = GGML_FP32_TO_FP16(norm);
        b->rnorm = GGML_FP32_TO_FP16(rnorm);
    }
}

void dequantize_row_tq3j(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TQL == 0);
    const int64_t nb = k / QK_TQL;
    const block_tq3j * GGML_RESTRICT blk = (const block_tq3j *)x;

    for (int64_t i = 0; i < nb; i++) {
        const block_tq3j * b = &blk[i];
        float * out = y + i * QK_TQL;

        float norm  = GGML_FP16_TO_FP32(b->norm);
        float rnorm = GGML_FP16_TO_FP32(b->rnorm);
        tq_segment_dequantize(out, 128, centroids_8_d128, up3,
                               b->qs, b->signs, norm, rnorm);
    }
}

void ggml_vec_dot_tq3j_f32(int n, float * GGML_RESTRICT s, size_t bs,
                             const void * GGML_RESTRICT vx, size_t bx,
                             const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_TQL == 0);
    assert(nrc == 1);
    (void)bs; (void)bx; (void)by; (void)nrc;

    const block_tq3j * GGML_RESTRICT blk = (const block_tq3j *)vx;
    const float      * GGML_RESTRICT q   = (const float *)vy;
    const int64_t nb = n / QK_TQL;
    float sumf = 0;

    for (int64_t i = 0; i < nb; i++) {
        const block_tq3j * b = &blk[i];
        const float * qb = q + i * QK_TQL;

        float q_fwht[128];
        memcpy(q_fwht, qb, 128 * sizeof(float));
        tq_fwht(q_fwht, 128);

        float norm  = GGML_FP16_TO_FP32(b->norm);
        float rnorm = GGML_FP16_TO_FP32(b->rnorm);
        sumf += tq_segment_vec_dot(q_fwht, 128, centroids_8_d128, up3,
                                    b->qs, b->signs, norm, rnorm);
    }
    *s = sumf;
}

// ===========================================================================
// TQ2J: FWHT-128 + 2-bit MSE centroids + 1-bit QJL
// ===========================================================================

void quantize_row_tq2j_ref(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TQL == 0);
    const int64_t nb = k / QK_TQL;
    block_tq2j * GGML_RESTRICT blk = (block_tq2j *)y;

    for (int64_t i = 0; i < nb; i++) {
        const float * xb = x + i * QK_TQL;
        block_tq2j * b = &blk[i];
        memset(b, 0, sizeof(block_tq2j));

        float fwht_buf[128];
        float rnorm;
        float norm = tq_segment_quantize(
            xb, 128,
            centroids_4_d128, 4, pk2,
            b->qs, sizeof(b->qs),
            b->signs, sizeof(b->signs),
            &rnorm, fwht_buf);
        b->norm  = GGML_FP32_TO_FP16(norm);
        b->rnorm = GGML_FP32_TO_FP16(rnorm);
    }
}

void dequantize_row_tq2j(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TQL == 0);
    const int64_t nb = k / QK_TQL;
    const block_tq2j * GGML_RESTRICT blk = (const block_tq2j *)x;

    for (int64_t i = 0; i < nb; i++) {
        const block_tq2j * b = &blk[i];
        float * out = y + i * QK_TQL;

        float norm  = GGML_FP16_TO_FP32(b->norm);
        float rnorm = GGML_FP16_TO_FP32(b->rnorm);
        tq_segment_dequantize(out, 128, centroids_4_d128, up2,
                               b->qs, b->signs, norm, rnorm);
    }
}

void ggml_vec_dot_tq2j_f32(int n, float * GGML_RESTRICT s, size_t bs,
                             const void * GGML_RESTRICT vx, size_t bx,
                             const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_TQL == 0);
    assert(nrc == 1);
    (void)bs; (void)bx; (void)by; (void)nrc;

    const block_tq2j * GGML_RESTRICT blk = (const block_tq2j *)vx;
    const float      * GGML_RESTRICT q   = (const float *)vy;
    const int64_t nb = n / QK_TQL;
    float sumf = 0;

    for (int64_t i = 0; i < nb; i++) {
        const block_tq2j * b = &blk[i];
        const float * qb = q + i * QK_TQL;

        float q_fwht[128];
        memcpy(q_fwht, qb, 128 * sizeof(float));
        tq_fwht(q_fwht, 128);

        float norm  = GGML_FP16_TO_FP32(b->norm);
        float rnorm = GGML_FP16_TO_FP32(b->rnorm);
        sumf += tq_segment_vec_dot(q_fwht, 128, centroids_4_d128, up2,
                                    b->qs, b->signs, norm, rnorm);
    }
    *s = sumf;
}

// ===========================================================================
// Wrapper functions for ggml_quantize_chunk
// ===========================================================================

size_t quantize_tql(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                    int64_t nrows, int64_t n_per_row, const float * imatrix) {
    (void)imatrix;
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_tql_ref(src + row * n_per_row,
                             (char *)dst + row * (n_per_row / QK_TQL) * sizeof(block_tql),
                             n_per_row);
    }
    return (size_t)(nrows * (n_per_row / QK_TQL) * sizeof(block_tql));
}

size_t quantize_tq3j(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                     int64_t nrows, int64_t n_per_row, const float * imatrix) {
    (void)imatrix;
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_tq3j_ref(src + row * n_per_row,
                              (char *)dst + row * (n_per_row / QK_TQL) * sizeof(block_tq3j),
                              n_per_row);
    }
    return (size_t)(nrows * (n_per_row / QK_TQL) * sizeof(block_tq3j));
}

size_t quantize_tq2j(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                     int64_t nrows, int64_t n_per_row, const float * imatrix) {
    (void)imatrix;
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_tq2j_ref(src + row * n_per_row,
                              (char *)dst + row * (n_per_row / QK_TQL) * sizeof(block_tq2j),
                              n_per_row);
    }
    return (size_t)(nrows * (n_per_row / QK_TQL) * sizeof(block_tq2j));
}
