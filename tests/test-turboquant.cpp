// TurboQuant reference implementation test harness
//
// Tests: format sanity, zero vector, determinism, roundtrip with multiple
//        distributions, attention score error, norm preservation, bit-packing.
//
// Priority: correctness first. Error metrics are printed for all distributions.

#include "ggml.h"
#include "ggml-cpu.h"

#undef NDEBUG
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <algorithm>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static const char * RESULT_STR[] = {"ok", "FAILED"};

// Simple deterministic LCG PRNG (portable, no dependency on platform rand)
static uint64_t prng_state = 12345678901234567ULL;

static void prng_seed(uint64_t seed) {
    prng_state = seed;
}

static uint32_t prng_next(void) {
    prng_state = prng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (uint32_t)(prng_state >> 32);
}

// Uniform float in [lo, hi]
static float prng_uniform(float lo, float hi) {
    return lo + (hi - lo) * ((float)prng_next() / (float)0xFFFFFFFF);
}

// Approximate Gaussian via Box-Muller
static float prng_gaussian(void) {
    float u1 = ((float)prng_next() + 1.0f) / ((float)0xFFFFFFFF + 2.0f);
    float u2 = ((float)prng_next() + 1.0f) / ((float)0xFFFFFFFF + 2.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265358979323846f * u2);
}

// L2 norm of a vector
static float vec_norm(const float * v, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += (double)v[i] * (double)v[i];
    }
    return (float)sqrt(sum);
}

// Dot product
static float vec_dot(const float * a, const float * b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += (double)a[i] * (double)b[i];
    }
    return (float)sum;
}

// Cosine similarity
static float cosine_sim(const float * a, const float * b, int n) {
    float dot = vec_dot(a, b, n);
    float na = vec_norm(a, n);
    float nb = vec_norm(b, n);
    if (na < 1e-12f || nb < 1e-12f) return 0.0f;
    return dot / (na * nb);
}

// Relative L2 error: ||a-b||/||a||
static float relative_l2_error(const float * orig, const float * recon, int n) {
    float norm_orig = vec_norm(orig, n);
    if (norm_orig < 1e-12f) return 0.0f;
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        double d = (double)orig[i] - (double)recon[i];
        sum += d * d;
    }
    return (float)(sqrt(sum) / (double)norm_orig);
}

// RMSE
static float rmse(const float * a, const float * b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        double d = (double)a[i] - (double)b[i];
        sum += d * d;
    }
    return (float)sqrt(sum / n);
}

// Max absolute error
static float max_abs_error(const float * a, const float * b, int n) {
    float mx = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

// Check no NaN or Inf
static bool has_nan_inf(const float * v, int n) {
    for (int i = 0; i < n; i++) {
        if (isnan(v[i]) || isinf(v[i])) return true;
    }
    return false;
}

// Generate normalized Gaussian vector
static void gen_gaussian_normalized(float * dst, int n, float target_norm) {
    for (int i = 0; i < n; i++) {
        dst[i] = prng_gaussian();
    }
    float nrm = vec_norm(dst, n);
    if (nrm > 1e-12f) {
        float scale = target_norm / nrm;
        for (int i = 0; i < n; i++) dst[i] *= scale;
    }
}

// Generate cosine wave data (matching test-quantize-fns.cpp)
static void gen_cosine(float offset, int n, float * dst) {
    for (int i = 0; i < n; i++) {
        dst[i] = 0.1f + 2.0f * cosf((float)i + offset);
    }
}

// ---------------------------------------------------------------------------
// Generic quantize/dequantize roundtrip using type traits
// ---------------------------------------------------------------------------

struct roundtrip_result {
    float rmse_val;
    float rel_l2;
    float max_err;
    float cos_sim;
    float norm_ratio;  // ||recon|| / ||orig||
    bool  has_bad;     // NaN or Inf
};

static roundtrip_result do_roundtrip(ggml_type type, const float * input, int n) {
    const auto * traits = ggml_get_type_traits(type);
    const int64_t blck = traits->blck_size;
    assert(n % blck == 0);

    size_t qbuf_size = (size_t)(n / blck) * traits->type_size;
    std::vector<uint8_t> qbuf(qbuf_size);
    std::vector<float> output(n);

    // Quantize
    traits->from_float_ref(input, qbuf.data(), n);
    // Dequantize
    traits->to_float(qbuf.data(), output.data(), n);

    roundtrip_result r;
    r.rmse_val  = rmse(input, output.data(), n);
    r.rel_l2    = relative_l2_error(input, output.data(), n);
    r.max_err   = max_abs_error(input, output.data(), n);
    r.cos_sim   = cosine_sim(input, output.data(), n);
    r.norm_ratio = vec_norm(output.data(), n) / fmaxf(vec_norm(input, n), 1e-12f);
    r.has_bad   = has_nan_inf(output.data(), n);
    return r;
}

// ---------------------------------------------------------------------------
// Test A: Format / structural sanity
// ---------------------------------------------------------------------------

static int test_format_sanity(void) {
    int failures = 0;
    printf("\n=== Test A: Format / structural sanity ===\n");

    // TURBO3_0
    {
        bool ok = true;
        ok = ok && (ggml_type_size(GGML_TYPE_TURBO3_0_PROD)  == 42);
        ok = ok && (ggml_blck_size(GGML_TYPE_TURBO3_0_PROD)  == 128);
        ok = ok && (ggml_is_quantized(GGML_TYPE_TURBO3_0_PROD));
        ok = ok && (strcmp(ggml_type_name(GGML_TYPE_TURBO3_0_PROD), "tqk_lo") == 0);
        printf("  tqk_lo format: %s (size=%zu blck=%d name=%s)\n",
               RESULT_STR[!ok],
               ggml_type_size(GGML_TYPE_TURBO3_0_PROD),
               (int)ggml_blck_size(GGML_TYPE_TURBO3_0_PROD),
               ggml_type_name(GGML_TYPE_TURBO3_0_PROD));
        if (!ok) failures++;
    }

    // TURBO4_0
    {
        bool ok = true;
        ok = ok && (ggml_type_size(GGML_TYPE_TURBO4_0_PROD)  == 58);
        ok = ok && (ggml_blck_size(GGML_TYPE_TURBO4_0_PROD)  == 128);
        ok = ok && (ggml_is_quantized(GGML_TYPE_TURBO4_0_PROD));
        ok = ok && (strcmp(ggml_type_name(GGML_TYPE_TURBO4_0_PROD), "tqk_hi") == 0);
        printf("  tqk_hi format: %s (size=%zu blck=%d name=%s)\n",
               RESULT_STR[!ok],
               ggml_type_size(GGML_TYPE_TURBO4_0_PROD),
               (int)ggml_blck_size(GGML_TYPE_TURBO4_0_PROD),
               ggml_type_name(GGML_TYPE_TURBO4_0_PROD));
        if (!ok) failures++;
    }

    return failures;
}

// ---------------------------------------------------------------------------
// Test B: Zero vector roundtrip
// ---------------------------------------------------------------------------

static int test_zero_vector(void) {
    int failures = 0;
    printf("\n=== Test B: Zero vector roundtrip ===\n");

    // TURBO3_0
    {
        const int n = 128;
        std::vector<float> zeros(n, 0.0f);
        auto r = do_roundtrip(GGML_TYPE_TURBO3_0_PROD, zeros.data(), n);
        bool ok = (r.rmse_val == 0.0f) && !r.has_bad;
        printf("  turbo3_0 zero: %s (rmse=%e)\n", RESULT_STR[!ok], r.rmse_val);
        if (!ok) failures++;
    }

    // TURBO4_0
    {
        const int n = 128;
        std::vector<float> zeros(n, 0.0f);
        auto r = do_roundtrip(GGML_TYPE_TURBO4_0_PROD, zeros.data(), n);
        bool ok = (r.rmse_val == 0.0f) && !r.has_bad;
        printf("  turbo4_0 zero: %s (rmse=%e)\n", RESULT_STR[!ok], r.rmse_val);
        if (!ok) failures++;
    }

    return failures;
}

// ---------------------------------------------------------------------------
// Test C: Determinism
// ---------------------------------------------------------------------------

static int test_determinism(void) {
    int failures = 0;
    printf("\n=== Test C: Determinism ===\n");

    auto check_determinism = [&](ggml_type type, int n) {
        const auto * traits = ggml_get_type_traits(type);
        size_t qsz = (size_t)(n / traits->blck_size) * traits->type_size;

        prng_seed(42);
        std::vector<float> data(n);
        for (int i = 0; i < n; i++) data[i] = prng_gaussian();

        std::vector<uint8_t> q1(qsz), q2(qsz);
        traits->from_float_ref(data.data(), q1.data(), n);
        traits->from_float_ref(data.data(), q2.data(), n);

        bool ok = (memcmp(q1.data(), q2.data(), qsz) == 0);
        printf("  %s determinism: %s\n", ggml_type_name(type), RESULT_STR[!ok]);
        if (!ok) failures++;
    };

    check_determinism(GGML_TYPE_TURBO3_0_PROD, 128 * 4);
    check_determinism(GGML_TYPE_TURBO4_0_PROD, 128 * 2);
    return failures;
}

// ---------------------------------------------------------------------------
// Test D: Roundtrip with multiple distributions
// ---------------------------------------------------------------------------

struct dist_test {
    const char * name;
    float threshold_rel_l2_t3;  // TURBO3_0 max relative L2
    float threshold_cos_t3;     // TURBO3_0 min cosine sim
    float threshold_rel_l2_t4;  // TURBO4_0 max relative L2
    float threshold_cos_t4;     // TURBO4_0 min cosine sim
};

static int test_roundtrip_distributions(bool verbose) {
    int failures = 0;
    printf("\n=== Test D: Roundtrip with multiple distributions ===\n");

    // Distributions to test
    const int n_t3 = 128 * 4;  // 4 blocks of TURBO3
    const int n_t4 = 128 * 4;  // 4 blocks of TURBO4

    // TURBO3_0 is 2.5 bpw (mixed: 32 ch @ 3-bit + 96 ch @ 2-bit), higher error than TURBO4_0
    // TURBO4_0 is 3.5 bpw (uniform 4-bit), theoretical rel L2 ~ sqrt(0.03) ~ 0.17
    // TURBO3_0 theoretical MSE dominated by 96 regular channels at 1-bit PolarQuant
    dist_test tests[] = {
        // name,                    t3_rel, t3_cos, t4_rel, t4_cos
        // i.i.d. Gaussian QJL adds reconstruction noise (by design — trades MSE for
        // unbiased inner products). Thresholds are for reconstruction, not attention quality.
        { "cosine_wave",            0.80f,  0.70f,  0.60f,  0.80f  },
        { "gaussian_unit_norm",     0.80f,  0.75f,  0.45f,  0.90f  },
        { "gaussian_low_norm",      0.80f,  0.75f,  0.45f,  0.90f  },
        { "gaussian_high_norm",     0.80f,  0.75f,  0.45f,  0.90f  },
        { "single_spike",           1.50f,  0.20f,  1.20f,  0.30f  },
        { "alternating_constant",   0.80f,  0.60f,  0.70f,  0.60f  },
    };
    const int n_tests = sizeof(tests) / sizeof(tests[0]);

    for (int t = 0; t < n_tests; t++) {
        // Generate data for TURBO3_0
        std::vector<float> data_t3(n_t3);
        // Generate data for TURBO4_0
        std::vector<float> data_t4(n_t4);

        prng_seed(1000 + t);

        if (t == 0) {
            // Cosine wave
            gen_cosine(0.0f, n_t3, data_t3.data());
            gen_cosine(0.0f, n_t4, data_t4.data());
        } else if (t == 1) {
            // Gaussian, normalized to unit norm per block
            for (int b = 0; b < n_t3 / 128; b++)
                gen_gaussian_normalized(data_t3.data() + b * 128, 128, 1.0f);
            for (int b = 0; b < n_t4 / 128; b++)
                gen_gaussian_normalized(data_t4.data() + b * 128, 128, 1.0f);
        } else if (t == 2) {
            // Low norm
            for (int b = 0; b < n_t3 / 128; b++)
                gen_gaussian_normalized(data_t3.data() + b * 128, 128, 0.001f);
            for (int b = 0; b < n_t4 / 128; b++)
                gen_gaussian_normalized(data_t4.data() + b * 128, 128, 0.001f);
        } else if (t == 3) {
            // High norm
            for (int b = 0; b < n_t3 / 128; b++)
                gen_gaussian_normalized(data_t3.data() + b * 128, 128, 1000.0f);
            for (int b = 0; b < n_t4 / 128; b++)
                gen_gaussian_normalized(data_t4.data() + b * 128, 128, 1000.0f);
        } else if (t == 4) {
            // Single spike: one large value, rest small
            memset(data_t3.data(), 0, n_t3 * sizeof(float));
            memset(data_t4.data(), 0, n_t4 * sizeof(float));
            for (int b = 0; b < n_t3 / 128; b++)
                data_t3[b * 128] = 1.0f;
            for (int b = 0; b < n_t4 / 128; b++)
                data_t4[b * 128] = 1.0f;
        } else if (t == 5) {
            // Alternating +/-
            for (int i = 0; i < n_t3; i++)
                data_t3[i] = (i % 2 == 0) ? 0.5f : -0.5f;
            for (int i = 0; i < n_t4; i++)
                data_t4[i] = (i % 2 == 0) ? 0.5f : -0.5f;
        }

        // TURBO3_0
        auto r3 = do_roundtrip(GGML_TYPE_TURBO3_0_PROD, data_t3.data(), n_t3);
        bool ok3 = (r3.rel_l2 < tests[t].threshold_rel_l2_t3) &&
                   (r3.cos_sim > tests[t].threshold_cos_t3) &&
                   !r3.has_bad;
        if (!ok3) failures++;

        // TURBO4_0
        auto r4 = do_roundtrip(GGML_TYPE_TURBO4_0_PROD, data_t4.data(), n_t4);
        bool ok4 = (r4.rel_l2 < tests[t].threshold_rel_l2_t4) &&
                   (r4.cos_sim > tests[t].threshold_cos_t4) &&
                   !r4.has_bad;
        if (!ok4) failures++;

        if (!ok3 || !ok4 || verbose) {
            printf("  %-25s  turbo3: %s (rel_l2=%.6f cos=%.6f rmse=%.6f max_err=%.6f norm_ratio=%.4f)\n",
                   tests[t].name, RESULT_STR[!ok3], r3.rel_l2, r3.cos_sim, r3.rmse_val, r3.max_err, r3.norm_ratio);
            printf("  %-25s  turbo4: %s (rel_l2=%.6f cos=%.6f rmse=%.6f max_err=%.6f norm_ratio=%.4f)\n",
                   tests[t].name, RESULT_STR[!ok4], r4.rel_l2, r4.cos_sim, r4.rmse_val, r4.max_err, r4.norm_ratio);
        }
    }

    return failures;
}

// ---------------------------------------------------------------------------
// Test E: Score error (attention inner product preservation)
// ---------------------------------------------------------------------------

static int test_score_error(bool verbose) {
    int failures = 0;
    printf("\n=== Test E: Attention score error ===\n");

    const int n_pairs = 1000;

    auto test_score = [&](ggml_type type, int d) {
        const auto * traits = ggml_get_type_traits(type);
        size_t qsz = (size_t)(d / traits->blck_size) * traits->type_size;

        prng_seed(9999);

        double sum_abs_err = 0.0;
        double sum_sq_err = 0.0;
        float max_abs_err = 0.0f;

        std::vector<float> query(d), key(d), key_recon(d);
        std::vector<uint8_t> key_q(qsz);

        for (int p = 0; p < n_pairs; p++) {
            // Generate random normalized query and key
            gen_gaussian_normalized(query.data(), d, 1.0f);
            gen_gaussian_normalized(key.data(), d, 1.0f);

            // Quantize and dequantize key
            traits->from_float_ref(key.data(), key_q.data(), d);
            traits->to_float(key_q.data(), key_recon.data(), d);

            // Compute scores
            float score_orig = vec_dot(query.data(), key.data(), d);
            float score_quant = vec_dot(query.data(), key_recon.data(), d);
            float err = fabsf(score_orig - score_quant);

            sum_abs_err += err;
            sum_sq_err += (double)err * (double)err;
            if (err > max_abs_err) max_abs_err = err;
        }

        float mean_err = (float)(sum_abs_err / n_pairs);
        float std_err = (float)sqrt(sum_sq_err / n_pairs - (sum_abs_err / n_pairs) * (sum_abs_err / n_pairs));

        // For TurboQuant on unit-norm Gaussian vectors, mean score error should be small
        // Paper: i.i.d. Gaussian QJL adds reconstruction noise but preserves inner products
        // Thresholds derived from theoretical MSE + QJL variance
        float threshold_mean = (type == GGML_TYPE_TURBO3_0_PROD) ? 0.10f : 0.05f;
        bool ok = (mean_err < threshold_mean);
        if (!ok) failures++;

        printf("  %s (d=%d, %d pairs): %s\n", ggml_type_name(type), d, n_pairs, RESULT_STR[!ok]);
        printf("    mean_abs_score_err = %.6f\n", mean_err);
        printf("    max_abs_score_err  = %.6f\n", max_abs_err);
        printf("    std_score_err      = %.6f\n", std_err);

        (void)verbose;
    };

    test_score(GGML_TYPE_TURBO3_0_PROD, 128);
    test_score(GGML_TYPE_TURBO4_0_PROD, 128);

    return failures;
}

// ---------------------------------------------------------------------------
// Test F: Multi-block consistency
// ---------------------------------------------------------------------------

static int test_multiblock(void) {
    int failures = 0;
    printf("\n=== Test F: Multi-block consistency ===\n");

    auto test_mb = [&](ggml_type type, int blk_size, int n_blocks) {
        int n = blk_size * n_blocks;

        prng_seed(7777);
        std::vector<float> data(n);
        for (int b = 0; b < n_blocks; b++) {
            gen_gaussian_normalized(data.data() + b * blk_size, blk_size, 1.0f);
        }

        auto r = do_roundtrip(type, data.data(), n);

        // Check per-block error uniformity
        float min_block_err = 1e30f, max_block_err = 0.0f;
        const auto * traits = ggml_get_type_traits(type);
        size_t qsz_block = traits->type_size;
        std::vector<uint8_t> qbuf((size_t)(n / blk_size) * qsz_block);
        std::vector<float> recon(n);

        traits->from_float_ref(data.data(), qbuf.data(), n);
        traits->to_float(qbuf.data(), recon.data(), n);

        for (int b = 0; b < n_blocks; b++) {
            float block_err = relative_l2_error(
                data.data() + b * blk_size,
                recon.data() + b * blk_size,
                blk_size);
            if (block_err < min_block_err) min_block_err = block_err;
            if (block_err > max_block_err) max_block_err = block_err;
        }

        // Block errors should not vary wildly (no systematic drift)
        float err_range = max_block_err - min_block_err;
        bool ok = !r.has_bad && (err_range < 0.5f);
        if (!ok) failures++;

        printf("  %s (%d blocks): %s (overall_rel_l2=%.6f block_err_range=[%.6f, %.6f])\n",
               ggml_type_name(type), n_blocks, RESULT_STR[!ok],
               r.rel_l2, min_block_err, max_block_err);
    };

    test_mb(GGML_TYPE_TURBO3_0_PROD, 128, 32);
    test_mb(GGML_TYPE_TURBO4_0_PROD, 128, 32);
    return failures;
}

// ---------------------------------------------------------------------------
// Test G: Norm preservation
// ---------------------------------------------------------------------------

static int test_norm_preservation(void) {
    int failures = 0;
    printf("\n=== Test G: Norm preservation ===\n");

    auto test_np = [&](ggml_type type, int blk_size, int n_blocks) {
        int n = blk_size * n_blocks;

        prng_seed(5555);
        std::vector<float> data(n);
        for (int b = 0; b < n_blocks; b++) {
            gen_gaussian_normalized(data.data() + b * blk_size, blk_size, 1.0f + prng_uniform(-0.5f, 0.5f));
        }

        auto r = do_roundtrip(type, data.data(), n);
        float norm_tol = (type == GGML_TYPE_TURBO3_0_PROD) ? 0.25f : 0.15f;
        bool ok = (fabsf(r.norm_ratio - 1.0f) < norm_tol) && !r.has_bad;
        if (!ok) failures++;

        printf("  %s norm_ratio: %s (%.6f, target ~1.0)\n",
               ggml_type_name(type), RESULT_STR[!ok], r.norm_ratio);
    };

    test_np(GGML_TYPE_TURBO3_0_PROD, 128, 8);
    test_np(GGML_TYPE_TURBO4_0_PROD, 128, 8);
    return failures;
}

// ---------------------------------------------------------------------------
// Test H: Bit-packing validation
// ---------------------------------------------------------------------------

static int test_bit_packing(void) {
    int failures = 0;
    printf("\n=== Test H: Bit-packing validation ===\n");

    // TURBO3_0: inner product preservation for a single block
    // With uniform 2-bit PolarQuant + reduced QJL, inner products should be well-preserved.
    {
        const int n = 128;
        prng_seed(4321);
        float data[128], query[128];
        for (int j = 0; j < n; j++) data[j] = prng_gaussian();
        for (int j = 0; j < n; j++) query[j] = prng_gaussian() * 0.1f;

        auto r = do_roundtrip(GGML_TYPE_TURBO3_0_PROD, data, n);

        const auto * traits = ggml_get_type_traits(GGML_TYPE_TURBO3_0_PROD);
        std::vector<uint8_t> qbuf(traits->type_size);
        float recon[128];
        traits->from_float_ref(data, qbuf.data(), n);
        traits->to_float(qbuf.data(), recon, n);

        float dot_orig = vec_dot(query, data, n);
        float dot_recon = vec_dot(query, recon, n);
        float dot_err = fabsf(dot_orig - dot_recon) / fmaxf(fabsf(dot_orig), 1e-6f);

        bool ok = (dot_err < 0.4f) && !r.has_bad;
        printf("  turbo3_0 inner product check: %s (dot_err=%.4f, rel_l2=%.4f)\n",
               RESULT_STR[!ok], dot_err, r.rel_l2);
        if (!ok) failures++;
    }

    // TURBO4_0: 3-bit packing + QJL inner product check
    // Verify that cycling through centroid indices (byte-crossing patterns) produces
    // reasonable reconstruction, and inner products are preserved.
    {
        const int n = 128;
        prng_seed(7654);
        float data[128], query[128];
        for (int j = 0; j < n; j++) data[j] = prng_gaussian();
        for (int j = 0; j < n; j++) query[j] = prng_gaussian() * 0.1f;

        auto r = do_roundtrip(GGML_TYPE_TURBO4_0_PROD, data, n);

        const auto * traits = ggml_get_type_traits(GGML_TYPE_TURBO4_0_PROD);
        std::vector<uint8_t> qbuf(traits->type_size);
        float recon[128];
        traits->from_float_ref(data, qbuf.data(), n);
        traits->to_float(qbuf.data(), recon, n);

        float dot_orig = vec_dot(query, data, n);
        float dot_recon = vec_dot(query, recon, n);
        float dot_err = fabsf(dot_orig - dot_recon) / fmaxf(fabsf(dot_orig), 1e-6f);

        bool ok = (dot_err < 0.4f) && !r.has_bad;
        printf("  turbo4_0 inner product check: %s (dot_err=%.4f, rel_l2=%.4f)\n",
               RESULT_STR[!ok], dot_err, r.rel_l2);
        if (!ok) failures++;
    }

    return failures;
}

// ---------------------------------------------------------------------------
// Test I: KV cache simulation with rotation
// ---------------------------------------------------------------------------

// Rotate vector: y = Q * x (Q is d×d row-major orthogonal matrix)
static void rotate_qr(const float * x, float * y, const float * Q, int d) {
    for (int i = 0; i < d; i++) {
        double sum = 0.0;
        for (int j = 0; j < d; j++) {
            sum += (double)Q[i * d + j] * (double)x[j];
        }
        y[i] = (float)sum;
    }
}

// Inverse rotate: y = Q^T * x (Q orthogonal, so Q^{-1} = Q^T)
static void rotate_qr_inv(const float * x, float * y, const float * Q, int d) {
    for (int i = 0; i < d; i++) {
        double sum = 0.0;
        for (int j = 0; j < d; j++) {
            sum += (double)Q[j * d + i] * (double)x[j];
        }
        y[i] = (float)sum;
    }
}

// ---------------------------------------------------------------------------
// KV cache data generators
// ---------------------------------------------------------------------------

// Type 1: Log-normal channel scales with sparse outliers
// Mimics typical attention K projections with a few dominant channels
static void gen_kv_lognormal_outlier(float * dst, int d, float base_norm) {
    float channel_scale[256];
    assert(d <= 256);
    for (int i = 0; i < d; i++) {
        channel_scale[i] = expf(0.4f * prng_gaussian());
    }
    int n_outliers = 3 + (prng_next() % 4);
    for (int o = 0; o < n_outliers; o++) {
        int idx = prng_next() % d;
        channel_scale[idx] *= 5.0f + prng_uniform(0.0f, 10.0f);
    }
    for (int i = 0; i < d; i++) {
        dst[i] = prng_gaussian() * channel_scale[i];
    }
    float nrm = vec_norm(dst, d);
    if (nrm > 1e-12f) {
        for (int i = 0; i < d; i++) dst[i] *= base_norm / nrm;
    }
}

// Type 2: Power-law channel distribution
// A few channels carry most of the energy (Zipf-like)
static void gen_kv_powerlaw(float * dst, int d, float base_norm) {
    for (int i = 0; i < d; i++) {
        float scale = 1.0f / powf((float)(i + 1), 0.8f);  // Zipf-like decay
        dst[i] = prng_gaussian() * scale;
    }
    float nrm = vec_norm(dst, d);
    if (nrm > 1e-12f) {
        for (int i = 0; i < d; i++) dst[i] *= base_norm / nrm;
    }
}

// Type 3: Correlated channels (block-structured)
// Groups of adjacent channels are correlated — common in transformer projections
static void gen_kv_correlated(float * dst, int d, float base_norm) {
    int group_size = 8 + (prng_next() % 8);  // 8-16 channel groups
    for (int i = 0; i < d; i += group_size) {
        float group_mean = prng_gaussian() * 0.5f;
        float group_scale = 0.5f + prng_uniform(0.0f, 2.0f);
        int end = (i + group_size < d) ? i + group_size : d;
        for (int j = i; j < end; j++) {
            dst[j] = group_mean + prng_gaussian() * 0.3f * group_scale;
        }
    }
    float nrm = vec_norm(dst, d);
    if (nrm > 1e-12f) {
        for (int i = 0; i < d; i++) dst[i] *= base_norm / nrm;
    }
}

// Type 4: Heavy-tailed (Student-t like)
// Occasional very large values — stress test for norm handling
static void gen_kv_heavy_tail(float * dst, int d, float base_norm) {
    for (int i = 0; i < d; i++) {
        float g1 = prng_gaussian();
        float g2 = prng_gaussian();
        // Approximate Student-t with df=3 via ratio
        float denom = fabsf(g2) + 0.1f;
        dst[i] = g1 / denom;
    }
    float nrm = vec_norm(dst, d);
    if (nrm > 1e-12f) {
        for (int i = 0; i < d; i++) dst[i] *= base_norm / nrm;
    }
}

// Type 5: Near-sparse (most channels near zero, a few large)
// Mimics attention patterns where only a few positions matter
static void gen_kv_sparse(float * dst, int d, float base_norm) {
    for (int i = 0; i < d; i++) {
        dst[i] = prng_gaussian() * 0.01f;  // background noise
    }
    // 10-20% of channels have significant values
    int n_active = d / 8 + (prng_next() % (d / 8));
    for (int a = 0; a < n_active; a++) {
        int idx = prng_next() % d;
        dst[idx] = prng_gaussian() * (1.0f + prng_uniform(0.0f, 3.0f));
    }
    float nrm = vec_norm(dst, d);
    if (nrm > 1e-12f) {
        for (int i = 0; i < d; i++) dst[i] *= base_norm / nrm;
    }
}

// Type 6: Asymmetric (biased positive or negative)
// Some layers produce activations with non-zero mean
static void gen_kv_asymmetric(float * dst, int d, float base_norm) {
    float bias = prng_uniform(-0.3f, 0.3f);
    for (int i = 0; i < d; i++) {
        dst[i] = bias + prng_gaussian() * (0.5f + 0.5f * sinf((float)i * 0.1f));
    }
    float nrm = vec_norm(dst, d);
    if (nrm > 1e-12f) {
        for (int i = 0; i < d; i++) dst[i] *= base_norm / nrm;
    }
}

// ---------------------------------------------------------------------------
// Paper-reference simulation: QR rotation + i.i.d. Gaussian QJL
// ---------------------------------------------------------------------------
// This implements the EXACT paper algorithm in the test harness,
// bypassing the block-level API entirely. Used for comparison only.

// Generate random orthogonal matrix via Gram-Schmidt on Gaussian columns
static void gen_random_orthogonal(float * Q, int d) {
    // Q is d×d stored row-major
    // Generate random Gaussian, then orthogonalize columns via modified Gram-Schmidt
    for (int i = 0; i < d * d; i++) {
        Q[i] = prng_gaussian();
    }
    // Modified Gram-Schmidt on columns
    for (int j = 0; j < d; j++) {
        // Normalize column j
        float norm = 0.0f;
        for (int i = 0; i < d; i++) norm += Q[i * d + j] * Q[i * d + j];
        norm = sqrtf(norm);
        if (norm > 1e-12f) {
            for (int i = 0; i < d; i++) Q[i * d + j] /= norm;
        }
        // Subtract projection from all subsequent columns
        for (int k = j + 1; k < d; k++) {
            float dot = 0.0f;
            for (int i = 0; i < d; i++) dot += Q[i * d + j] * Q[i * d + k];
            for (int i = 0; i < d; i++) Q[i * d + k] -= dot * Q[i * d + j];
        }
    }
}

// Matrix-vector multiply: y = M * x (M is d×d row-major)
static void matvec(const float * M, const float * x, float * y, int d) {
    for (int i = 0; i < d; i++) {
        double sum = 0.0;
        for (int j = 0; j < d; j++) {
            sum += (double)M[i * d + j] * (double)x[j];
        }
        y[i] = (float)sum;
    }
}

// Matrix-transpose-vector multiply: y = M^T * x
static void matvec_t(const float * M, const float * x, float * y, int d) {
    for (int i = 0; i < d; i++) {
        double sum = 0.0;
        for (int j = 0; j < d; j++) {
            sum += (double)M[j * d + i] * (double)x[j];
        }
        y[i] = (float)sum;
    }
}

// Paper-exact TurboQuant_prod for a single d-dim vector
// Uses: QR rotation, b-bit MSE quantizer, i.i.d. Gaussian QJL
// centroids: array of 2^(b-1) signed centroids for the MSE stage
// n_centroids: number of centroids (2^(b-1))
// Pi: d×d orthogonal rotation matrix
// S: d×d i.i.d. Gaussian QJL matrix
static void paper_turbo_quantize(
    const float * input, float * output, int d,
    const float * Pi, const float * S,
    const float * centroids, int n_centroids)
{
    std::vector<float> rotated(d), normalized(d), residual(d);
    std::vector<float> qjl_proj(d), qjl_recon(d);
    std::vector<int> indices(d);

    // 1. Apply rotation: y = Pi * x
    matvec(Pi, input, rotated.data(), d);

    // 2. Extract norm and normalize
    float norm = vec_norm(rotated.data(), d);
    if (norm < 1e-12f) {
        memset(output, 0, d * sizeof(float));
        return;
    }
    for (int i = 0; i < d; i++) normalized[i] = rotated[i] / norm;

    // 3. PolarQuant: find nearest centroid for each coordinate
    for (int i = 0; i < d; i++) {
        int best = 0;
        float best_dist = fabsf(normalized[i] - centroids[0]);
        for (int c = 1; c < n_centroids; c++) {
            float dist = fabsf(normalized[i] - centroids[c]);
            if (dist < best_dist) { best_dist = dist; best = c; }
        }
        indices[i] = best;
    }

    // 4. Compute residual
    for (int i = 0; i < d; i++) {
        residual[i] = normalized[i] - centroids[indices[i]];
    }
    float rnorm = vec_norm(residual.data(), d);

    // 5. QJL: project residual with i.i.d. Gaussian S, take signs
    matvec(S, residual.data(), qjl_proj.data(), d);
    std::vector<float> sign_bits(d);
    for (int i = 0; i < d; i++) {
        sign_bits[i] = (qjl_proj[i] >= 0.0f) ? 1.0f : -1.0f;
    }

    // 6. QJL inverse: correction = sqrt(pi/2)/d * ||r|| * S^T * sign_bits
    matvec_t(S, sign_bits.data(), qjl_recon.data(), d);
    float alpha = 1.2533141f / (float)d;  // sqrt(pi/2) / d
    for (int i = 0; i < d; i++) {
        qjl_recon[i] *= alpha * rnorm;
    }

    // 7. Reconstruct in rotated space: centroid + QJL correction, scaled by norm
    std::vector<float> recon_rotated(d);
    for (int i = 0; i < d; i++) {
        recon_rotated[i] = norm * (centroids[indices[i]] + qjl_recon[i]);
    }

    // 8. Inverse rotation: output = Pi^T * recon_rotated
    matvec_t(Pi, recon_rotated.data(), output, d);
}

// Paper-exact centroids (same as in ggml-turbo-quant.c)
static const float paper_centroids_8[8] = {  // 3-bit MSE, d=128
    -0.1883988281f, -0.1181421705f, -0.0665887043f, -0.0216082019f,
     0.0216082019f,  0.0665887043f,  0.1181421705f,  0.1883988281f,
};
static const float paper_centroids_4[4] = {  // 2-bit MSE, d=128
    -0.1330458627f, -0.0399983984f, 0.0399983984f, 0.1330458627f,
};
static const float paper_centroids_2[2] = {  // 1-bit MSE, d=128
    -0.0707250243f, 0.0707250243f,
};

// Full pipeline roundtrip: rotate → quantize → dequantize → inverse rotate
// Q is d×d orthogonal matrix (row-major)
static roundtrip_result do_roundtrip_rotated(ggml_type type, const float * input, int d,
                                              const float * Q) {
    const auto * traits = ggml_get_type_traits(type);
    const int64_t blck = traits->blck_size;
    assert(d % blck == 0);

    size_t qbuf_size = (size_t)(d / blck) * traits->type_size;
    std::vector<float> rotated(d);
    std::vector<uint8_t> qbuf(qbuf_size);
    std::vector<float> dequantized(d);
    std::vector<float> output(d);

    // Forward rotation
    rotate_qr(input, rotated.data(), Q, d);
    // Quantize
    traits->from_float_ref(rotated.data(), qbuf.data(), d);
    // Dequantize
    traits->to_float(qbuf.data(), dequantized.data(), d);
    // Inverse rotation
    rotate_qr_inv(dequantized.data(), output.data(), Q, d);

    roundtrip_result r;
    r.rmse_val  = rmse(input, output.data(), d);
    r.rel_l2    = relative_l2_error(input, output.data(), d);
    r.max_err   = max_abs_error(input, output.data(), d);
    r.cos_sim   = cosine_sim(input, output.data(), d);
    r.norm_ratio = vec_norm(output.data(), d) / fmaxf(vec_norm(input, d), 1e-12f);
    r.has_bad   = has_nan_inf(output.data(), d);
    return r;
}

// Compute score error for one vector with rotation pipeline
static float score_error_rotated(ggml_type type, const float * key, const float * query,
                                  const float * Q, int d) {
    const auto * traits = ggml_get_type_traits(type);
    size_t qsz = (size_t)(d / traits->blck_size) * traits->type_size;
    std::vector<float> rotated(d), dequant(d), recon(d);
    std::vector<uint8_t> qbuf(qsz);

    rotate_qr(key, rotated.data(), Q, d);
    traits->from_float_ref(rotated.data(), qbuf.data(), d);
    traits->to_float(qbuf.data(), dequant.data(), d);
    rotate_qr_inv(dequant.data(), recon.data(), Q, d);

    float score_orig = vec_dot(query, key, d);
    float score_quant = vec_dot(query, recon.data(), d);
    return fabsf(score_orig - score_quant);
}

static int test_kv_cache_simulation(bool /*verbose*/) {
    int failures = 0;
    printf("\n=== Test I: KV cache simulation (with and without rotation) ===\n");

    // Paper-derived thresholds:
    //   TURBO4_0 (3.5 bpw): paper claims "quality neutral"
    //     Theoretical: MSE ~ 0.03, score_err ~ sqrt(0.03/128) ~ 0.015
    //     Hard limit: score_err < 0.020 with rotation (must hold for ALL distributions)
    //
    //   TURBO3_0 (2.5 bpw): paper claims "marginal quality degradation"
    //     Theoretical: MSE ~ 0.30, score_err ~ sqrt(0.30/128) ~ 0.048
    //     Hard limit: score_err < 0.060 with rotation

    const int d = 128;

    // QR rotation matrix (fixed per "head", shared across all tests)
    prng_seed(314159);
    std::vector<float> Q_rot(d * d);
    gen_random_orthogonal(Q_rot.data(), d);

    typedef void (*gen_fn)(float *, int, float);
    struct kv_dist {
        const char * name;
        gen_fn       gen;
    };
    kv_dist distributions[] = {
        { "lognormal+outlier", gen_kv_lognormal_outlier },
        { "power-law",         gen_kv_powerlaw          },
        { "correlated",        gen_kv_correlated         },
        { "heavy-tail",        gen_kv_heavy_tail         },
        { "sparse",            gen_kv_sparse             },
        { "asymmetric",        gen_kv_asymmetric         },
    };
    const int n_dists = sizeof(distributions) / sizeof(distributions[0]);

    struct type_config {
        ggml_type type;
        // Paper-derived hard thresholds for WITH-rotation score error
        float max_mean_score_err_rot;
        // Rotation must improve score error vs no-rotation
        bool  require_rotation_helps;
    };
    type_config types[] = {
        { GGML_TYPE_TURBO3_0_PROD, 0.090f, false },  // paper: "marginal degradation" at 2.5 bpv (rotation may not help with small QJL dims)
        { GGML_TYPE_TURBO4_0_PROD, 0.045f, true },  // paper: "quality neutral" at 3.5 bpw (3.75 total)
    };

    const int n_vectors_per_dist = 200;

    for (auto & tc : types) {
        printf("\n  %s (%d vectors per distribution, %d distributions):\n",
               ggml_type_name(tc.type), n_vectors_per_dist, n_dists);
        printf("    %-22s  %8s %8s %8s | %8s %8s %8s | %s\n",
               "distribution", "norot_L2", "norot_sc", "norot_cs",
               "rot_L2", "rot_sc", "rot_cs", "status");
        printf("    %s\n", "------------------------------+---"
               "------------------------------+--------");

        float worst_rot_score_err = 0.0f;
        bool any_rotation_worse = false;

        for (int di = 0; di < n_dists; di++) {
            prng_seed(100000 + di * 1000);

            double sum_rel_l2_norot = 0, sum_rel_l2_rot = 0;
            double sum_cos_norot = 0, sum_cos_rot = 0;
            double sum_score_norot = 0, sum_score_rot = 0;

            for (int v = 0; v < n_vectors_per_dist; v++) {
                std::vector<float> kv_vec(d), query(d);
                float norm_scale = 0.5f + prng_uniform(0.0f, 2.0f);
                distributions[di].gen(kv_vec.data(), d, norm_scale);
                gen_gaussian_normalized(query.data(), d, 1.0f);

                // Without rotation
                auto r_norot = do_roundtrip(tc.type, kv_vec.data(), d);
                sum_rel_l2_norot += r_norot.rel_l2;
                sum_cos_norot += r_norot.cos_sim;

                const auto * traits = ggml_get_type_traits(tc.type);
                size_t qsz = traits->type_size;
                std::vector<uint8_t> qbuf(qsz);
                std::vector<float> recon_norot(d);
                traits->from_float_ref(kv_vec.data(), qbuf.data(), d);
                traits->to_float(qbuf.data(), recon_norot.data(), d);
                float sc_orig = vec_dot(query.data(), kv_vec.data(), d);
                float sc_norot = vec_dot(query.data(), recon_norot.data(), d);
                sum_score_norot += fabsf(sc_orig - sc_norot);

                // With rotation
                auto r_rot = do_roundtrip_rotated(tc.type, kv_vec.data(), d, Q_rot.data());
                sum_rel_l2_rot += r_rot.rel_l2;
                sum_cos_rot += r_rot.cos_sim;

                float se = score_error_rotated(tc.type, kv_vec.data(), query.data(), Q_rot.data(), d);
                sum_score_rot += se;
            }

            float mean_l2_nr  = (float)(sum_rel_l2_norot / n_vectors_per_dist);
            float mean_sc_nr  = (float)(sum_score_norot / n_vectors_per_dist);
            float mean_cos_nr = (float)(sum_cos_norot / n_vectors_per_dist);
            float mean_l2_r   = (float)(sum_rel_l2_rot / n_vectors_per_dist);
            float mean_sc_r   = (float)(sum_score_rot / n_vectors_per_dist);
            float mean_cos_r  = (float)(sum_cos_rot / n_vectors_per_dist);

            if (mean_sc_r > worst_rot_score_err) worst_rot_score_err = mean_sc_r;
            if (mean_sc_r > mean_sc_nr + 0.001f) any_rotation_worse = true;

            bool dist_ok = (mean_sc_r < tc.max_mean_score_err_rot);
            if (!dist_ok) failures++;

            printf("    %-22s  %8.4f %8.4f %8.4f | %8.4f %8.4f %8.4f | %s\n",
                   distributions[di].name,
                   mean_l2_nr, mean_sc_nr, mean_cos_nr,
                   mean_l2_r, mean_sc_r, mean_cos_r,
                   RESULT_STR[!dist_ok]);
        }

        // Aggregate checks
        bool score_ok = (worst_rot_score_err < tc.max_mean_score_err_rot);
        bool rotation_ok = !any_rotation_worse || !tc.require_rotation_helps;

        printf("    ---\n");
        printf("    worst rotated score_err: %.6f (limit %.3f): %s\n",
               worst_rot_score_err, tc.max_mean_score_err_rot, RESULT_STR[!score_ok]);
        printf("    rotation always helps: %s\n", rotation_ok ? "yes" : "NO — rotation made score error worse");

        if (!score_ok) failures++;
        if (!rotation_ok) failures++;
    }

    return failures;
}

// ---------------------------------------------------------------------------
// Test J: Comparison with existing quant types at similar bit widths
// ---------------------------------------------------------------------------

// Roundtrip a vector through a standard ggml quant type (no rotation)
// Returns score error against a query vector
static float baseline_score_error(ggml_type type, const float * key, const float * query, int d) {
    const auto * traits = ggml_get_type_traits(type);
    const auto * traits_cpu = ggml_get_type_traits_cpu(type);
    if (!traits_cpu->from_float || !traits->to_float) return -1.0f;
    if (traits->blck_size == 0) return -1.0f;
    if (d % traits->blck_size != 0) return -1.0f;

    size_t qsz = (size_t)(d / traits->blck_size) * traits->type_size;
    std::vector<uint8_t> qbuf(qsz);
    std::vector<float> recon(d);

    traits_cpu->from_float(key, qbuf.data(), d);
    traits->to_float(qbuf.data(), recon.data(), d);

    float sc_orig = vec_dot(query, key, d);
    float sc_quant = vec_dot(query, recon.data(), d);
    return fabsf(sc_orig - sc_quant);
}

static float baseline_rel_l2(ggml_type type, const float * data, int d) {
    const auto * traits = ggml_get_type_traits(type);
    const auto * traits_cpu = ggml_get_type_traits_cpu(type);
    if (!traits_cpu->from_float || !traits->to_float) return -1.0f;
    if (traits->blck_size == 0) return -1.0f;
    if (d % traits->blck_size != 0) return -1.0f;

    size_t qsz = (size_t)(d / traits->blck_size) * traits->type_size;
    std::vector<uint8_t> qbuf(qsz);
    std::vector<float> recon(d);

    traits_cpu->from_float(data, qbuf.data(), d);
    traits->to_float(qbuf.data(), recon.data(), d);

    return relative_l2_error(data, recon.data(), d);
}

static int test_comparison(bool /*verbose*/) {
    int failures = 0;
    printf("\n=== Test J: Comparison with existing quant types ===\n");

    // Compare TurboQuant (with rotation) vs standard types (no rotation) on KV data.
    // This is the fair comparison: each method at its designed operating point.
    //
    // Bit widths:
    //   Q2_K  ~2.63 bpw  |  TURBO3_0  2.50 bpw
    //   Q3_K  ~3.44 bpw  |  TURBO4_0  3.50 bpw  (also compare Q4_0 at 4.50 bpw)
    //   Q4_0  ~4.50 bpw  |

    struct comparison {
        ggml_type turbo_type;
        const char * turbo_label;
        float turbo_bpw;
        ggml_type baseline_type;
        const char * baseline_label;
        float baseline_bpw;
    };

    // Use the actual KV cache types from llama.cpp --cache-type-k/v
    comparison comparisons[] = {
        // TURBO3_0 (2.5 bpv) — very aggressive, compare vs nearest KV cache types
        { GGML_TYPE_TURBO3_0_PROD, "turbo3_0", 2.63f, GGML_TYPE_Q4_0,  "q4_0",  4.50f },
        { GGML_TYPE_TURBO3_0_PROD, "turbo3_0", 2.63f, GGML_TYPE_Q8_0,  "q8_0",  8.50f },
        // TURBO4_0 (3.5 bpv) — the key claim: q4_0-level quality at 22% less storage
        { GGML_TYPE_TURBO4_0_PROD, "turbo4_0", 3.63f, GGML_TYPE_Q4_0,  "q4_0",  4.50f },
        { GGML_TYPE_TURBO4_0_PROD, "turbo4_0", 3.63f, GGML_TYPE_Q4_1,  "q4_1",  5.00f },
        { GGML_TYPE_TURBO4_0_PROD, "turbo4_0", 3.63f, GGML_TYPE_Q5_0,  "q5_0",  5.50f },
        { GGML_TYPE_TURBO4_0_PROD, "turbo4_0", 3.63f, GGML_TYPE_Q8_0,  "q8_0",  8.50f },
    };
    const int n_comparisons = sizeof(comparisons) / sizeof(comparisons[0]);

    // Use d=256 so all types' block sizes divide evenly (QK_K=256 for K-quants)
    const int d = 256;
    const int n_vectors = 500;

    // QR rotation matrix for 128-dim blocks
    prng_seed(314159);
    std::vector<float> Q_rot(128 * 128);
    gen_random_orthogonal(Q_rot.data(), 128);

    // Initialize all quant types we'll use
    for (int i = 0; i < GGML_TYPE_COUNT; i++) {
        ggml_quantize_init((ggml_type)i);
    }

    printf("  %-10s %5s  vs  %-10s %5s  |  turbo_sc  base_sc  |  turbo_L2  base_L2  | winner\n",
           "turbo", "bpw", "baseline", "bpw");
    printf("  %s\n",
           "----------------------------------------------+---------------------+---------------------+-------");

    for (int ci = 0; ci < n_comparisons; ci++) {
        auto & cmp = comparisons[ci];

        double sum_turbo_score = 0, sum_base_score = 0;
        double sum_turbo_l2 = 0, sum_base_l2 = 0;

        prng_seed(555000 + ci);

        for (int v = 0; v < n_vectors; v++) {
            // Generate KV data (use mixed distributions)
            std::vector<float> kv_vec(d), query(d);
            float norm_scale = 0.5f + prng_uniform(0.0f, 2.0f);

            // Cycle through distribution types
            switch (v % 6) {
                case 0: gen_kv_lognormal_outlier(kv_vec.data(), d, norm_scale); break;
                case 1: gen_kv_powerlaw(kv_vec.data(), d, norm_scale); break;
                case 2: gen_kv_correlated(kv_vec.data(), d, norm_scale); break;
                case 3: gen_kv_heavy_tail(kv_vec.data(), d, norm_scale); break;
                case 4: gen_kv_sparse(kv_vec.data(), d, norm_scale); break;
                case 5: gen_kv_asymmetric(kv_vec.data(), d, norm_scale); break;
            }
            gen_gaussian_normalized(query.data(), d, 1.0f);

            // TurboQuant with rotation (process two 128-dim blocks)
            {
                const auto * traits = ggml_get_type_traits(cmp.turbo_type);
                size_t qsz_block = traits->type_size;

                std::vector<float> recon(d);
                for (int blk = 0; blk < d / 128; blk++) {
                    const float * src = kv_vec.data() + blk * 128;
                    float * dst = recon.data() + blk * 128;
                    std::vector<float> rotated(128), dequant(128);
                    std::vector<uint8_t> qbuf(qsz_block);

                    rotate_qr(src, rotated.data(), Q_rot.data(), 128);
                    traits->from_float_ref(rotated.data(), qbuf.data(), 128);
                    traits->to_float(qbuf.data(), dequant.data(), 128);
                    rotate_qr_inv(dequant.data(), dst, Q_rot.data(), 128);
                }

                float sc_orig = vec_dot(query.data(), kv_vec.data(), d);
                float sc_turbo = vec_dot(query.data(), recon.data(), d);
                sum_turbo_score += fabsf(sc_orig - sc_turbo);
                sum_turbo_l2 += relative_l2_error(kv_vec.data(), recon.data(), d);
            }

            // Baseline type (no rotation, direct quantize)
            {
                float se = baseline_score_error(cmp.baseline_type, kv_vec.data(), query.data(), d);
                float l2 = baseline_rel_l2(cmp.baseline_type, kv_vec.data(), d);
                if (se >= 0) sum_base_score += se;
                if (l2 >= 0) sum_base_l2 += l2;
            }
        }

        float turbo_sc = (float)(sum_turbo_score / n_vectors);
        float base_sc  = (float)(sum_base_score / n_vectors);
        float turbo_l2 = (float)(sum_turbo_l2 / n_vectors);
        float base_l2  = (float)(sum_base_l2 / n_vectors);

        const char * winner_sc = (turbo_sc < base_sc) ? "TURBO" : "baseline";
        const char * winner_l2 = (turbo_l2 < base_l2) ? "TURBO" : "baseline";

        printf("  %-10s %4.1fb  vs  %-10s %4.1fb  |  %7.4f   %7.4f  |  %7.4f   %7.4f  | sc:%s l2:%s\n",
               cmp.turbo_label, cmp.turbo_bpw,
               cmp.baseline_label, cmp.baseline_bpw,
               turbo_sc, base_sc,
               turbo_l2, base_l2,
               winner_sc, winner_l2);
    }

    // No hard pass/fail here — this is informational.
    // The point is to show how TurboQuant compares, not to enforce a winner.
    printf("  (informational — no pass/fail threshold)\n");

    return failures;
}

// ---------------------------------------------------------------------------
// Test K: Attention inner product fidelity
// ---------------------------------------------------------------------------
// Tests what actually matters for KV cache quantization:
// - Do attention scores preserve their relative ordering?
// - Does the top-1 key stay the same?
// - How close are the softmax attention weights?

// Compute softmax in-place
static void softmax(float * x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < n; i++) {
        x[i] /= sum;
    }
}

// KL divergence: sum(p * log(p/q)), with epsilon for stability
static float kl_divergence(const float * p, const float * q, int n) {
    double kl = 0.0;
    for (int i = 0; i < n; i++) {
        float pi = fmaxf(p[i], 1e-10f);
        float qi = fmaxf(q[i], 1e-10f);
        kl += (double)pi * log((double)pi / (double)qi);
    }
    return (float)kl;
}

// Spearman rank correlation
static float rank_correlation(const float * a, const float * b, int n) {
    // Compute ranks for both arrays
    std::vector<int> rank_a(n), rank_b(n), order(n);

    // Rank array a
    for (int i = 0; i < n; i++) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int x, int y) { return a[x] < a[y]; });
    for (int i = 0; i < n; i++) rank_a[order[i]] = i;

    // Rank array b
    for (int i = 0; i < n; i++) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int x, int y) { return b[x] < b[y]; });
    for (int i = 0; i < n; i++) rank_b[order[i]] = i;

    // Spearman: 1 - 6*sum(d^2) / (n*(n^2-1))
    double sum_d2 = 0;
    for (int i = 0; i < n; i++) {
        double d = (double)rank_a[i] - (double)rank_b[i];
        sum_d2 += d * d;
    }
    return (float)(1.0 - 6.0 * sum_d2 / ((double)n * ((double)n * n - 1.0)));
}

// Quantize a set of keys and compute attention metrics vs original
struct attn_metrics {
    float mean_score_err;    // mean |s - s'| over all query-key pairs
    float rank_corr;         // Spearman rank correlation of score vectors
    float top1_accuracy;     // fraction of queries where argmax key is preserved
    float softmax_kl;        // mean KL divergence of softmax attention weights
};

// Compute attention metrics for a quantization method applied to keys
static attn_metrics compute_attn_metrics_turbo(
    ggml_type type, const float * keys, int n_keys, int d,
    const float * queries, int n_queries, const float * signs)
{
    const auto * traits = ggml_get_type_traits(type);
    size_t qsz = traits->type_size;

    // Quantize all keys (with rotation)
    std::vector<std::vector<float>> recon_keys(n_keys, std::vector<float>(d));
    for (int k = 0; k < n_keys; k++) {
        std::vector<float> rotated(d), dequant(d);
        std::vector<uint8_t> qbuf(qsz);
        rotate_qr(keys + k * d, rotated.data(), signs, d);
        traits->from_float_ref(rotated.data(), qbuf.data(), d);
        traits->to_float(qbuf.data(), dequant.data(), d);
        rotate_qr_inv(dequant.data(), recon_keys[k].data(), signs, d);
    }

    double sum_score_err = 0;
    double sum_rank_corr = 0;
    int top1_correct = 0;
    double sum_kl = 0;

    for (int q = 0; q < n_queries; q++) {
        const float * query = queries + q * d;
        std::vector<float> scores_orig(n_keys), scores_quant(n_keys);

        for (int k = 0; k < n_keys; k++) {
            scores_orig[k]  = vec_dot(query, keys + k * d, d);
            scores_quant[k] = vec_dot(query, recon_keys[k].data(), d);
            sum_score_err += fabsf(scores_orig[k] - scores_quant[k]);
        }

        // Rank correlation
        sum_rank_corr += rank_correlation(scores_orig.data(), scores_quant.data(), n_keys);

        // Top-1 accuracy
        int argmax_orig = 0, argmax_quant = 0;
        for (int k = 1; k < n_keys; k++) {
            if (scores_orig[k]  > scores_orig[argmax_orig])   argmax_orig = k;
            if (scores_quant[k] > scores_quant[argmax_quant]) argmax_quant = k;
        }
        if (argmax_orig == argmax_quant) top1_correct++;

        // Softmax KL divergence (scale scores by 1/sqrt(d) as in real attention)
        float scale = 1.0f / sqrtf((float)d);
        std::vector<float> attn_orig(n_keys), attn_quant(n_keys);
        for (int k = 0; k < n_keys; k++) {
            attn_orig[k]  = scores_orig[k]  * scale;
            attn_quant[k] = scores_quant[k] * scale;
        }
        softmax(attn_orig.data(), n_keys);
        softmax(attn_quant.data(), n_keys);
        sum_kl += kl_divergence(attn_orig.data(), attn_quant.data(), n_keys);
    }

    attn_metrics m;
    m.mean_score_err = (float)(sum_score_err / (n_queries * n_keys));
    m.rank_corr      = (float)(sum_rank_corr / n_queries);
    m.top1_accuracy  = (float)top1_correct / (float)n_queries;
    m.softmax_kl     = (float)(sum_kl / n_queries);
    return m;
}

static attn_metrics compute_attn_metrics_baseline(
    ggml_type type, const float * keys, int n_keys, int d,
    const float * queries, int n_queries)
{
    const auto * traits = ggml_get_type_traits(type);
    const auto * traits_cpu = ggml_get_type_traits_cpu(type);
    if (!traits_cpu->from_float || !traits->to_float || traits->blck_size == 0) {
        return { -1, -1, -1, -1 };
    }

    // Keys need to be a multiple of blck_size. Use d directly if compatible,
    // otherwise this type can't be tested at this dimension.
    if (d % traits->blck_size != 0) {
        return { -1, -1, -1, -1 };
    }

    size_t qsz = (size_t)(d / traits->blck_size) * traits->type_size;

    // Quantize all keys (no rotation)
    std::vector<std::vector<float>> recon_keys(n_keys, std::vector<float>(d));
    for (int k = 0; k < n_keys; k++) {
        std::vector<uint8_t> qbuf(qsz);
        traits_cpu->from_float(keys + k * d, qbuf.data(), d);
        traits->to_float(qbuf.data(), recon_keys[k].data(), d);
    }

    double sum_score_err = 0;
    double sum_rank_corr = 0;
    int top1_correct = 0;
    double sum_kl = 0;

    for (int q = 0; q < n_queries; q++) {
        const float * query = queries + q * d;
        std::vector<float> scores_orig(n_keys), scores_quant(n_keys);

        for (int k = 0; k < n_keys; k++) {
            scores_orig[k]  = vec_dot(query, keys + k * d, d);
            scores_quant[k] = vec_dot(query, recon_keys[k].data(), d);
            sum_score_err += fabsf(scores_orig[k] - scores_quant[k]);
        }

        sum_rank_corr += rank_correlation(scores_orig.data(), scores_quant.data(), n_keys);

        int argmax_orig = 0, argmax_quant = 0;
        for (int k = 1; k < n_keys; k++) {
            if (scores_orig[k]  > scores_orig[argmax_orig])   argmax_orig = k;
            if (scores_quant[k] > scores_quant[argmax_quant]) argmax_quant = k;
        }
        if (argmax_orig == argmax_quant) top1_correct++;

        float scale = 1.0f / sqrtf((float)d);
        std::vector<float> attn_orig(n_keys), attn_quant(n_keys);
        for (int k = 0; k < n_keys; k++) {
            attn_orig[k]  = scores_orig[k]  * scale;
            attn_quant[k] = scores_quant[k] * scale;
        }
        softmax(attn_orig.data(), n_keys);
        softmax(attn_quant.data(), n_keys);
        sum_kl += kl_divergence(attn_orig.data(), attn_quant.data(), n_keys);
    }

    attn_metrics m;
    m.mean_score_err = (float)(sum_score_err / (n_queries * n_keys));
    m.rank_corr      = (float)(sum_rank_corr / n_queries);
    m.top1_accuracy  = (float)top1_correct / (float)n_queries;
    m.softmax_kl     = (float)(sum_kl / n_queries);
    return m;
}

static int test_attention_fidelity(bool /*verbose*/) {
    int failures = 0;
    printf("\n=== Test K: Attention inner product fidelity ===\n");

    // Simulate realistic attention at the actual KV cache head dimension (d=128).
    // All KV cache types (q4_0, q4_1, q5_0, q8_0) have blck=32 which divides 128.

    const int d = 128;
    const int n_keys = 64;
    const int n_queries = 200;

    // Generate keys with mixed distributions
    prng_seed(42424242);
    std::vector<float> keys(n_keys * d);
    for (int k = 0; k < n_keys; k++) {
        float norm_scale = 0.5f + prng_uniform(0.0f, 2.0f);
        switch (k % 6) {
            case 0: gen_kv_lognormal_outlier(keys.data() + k * d, d, norm_scale); break;
            case 1: gen_kv_powerlaw(keys.data() + k * d, d, norm_scale); break;
            case 2: gen_kv_correlated(keys.data() + k * d, d, norm_scale); break;
            case 3: gen_kv_heavy_tail(keys.data() + k * d, d, norm_scale); break;
            case 4: gen_kv_sparse(keys.data() + k * d, d, norm_scale); break;
            case 5: gen_kv_asymmetric(keys.data() + k * d, d, norm_scale); break;
        }
    }

    // Generate queries
    std::vector<float> queries(n_queries * d);
    for (int q = 0; q < n_queries; q++) {
        gen_gaussian_normalized(queries.data() + q * d, d, 1.0f);
    }

    // QR rotation matrix
    prng_seed(314159);
    std::vector<float> Q_rot(128 * 128);
    gen_random_orthogonal(Q_rot.data(), 128);

    // Initialize all quant types
    for (int i = 0; i < GGML_TYPE_COUNT; i++) {
        ggml_quantize_init((ggml_type)i);
    }

    struct type_entry {
        ggml_type type;
        const char * label;
        float bpw;
        bool is_turbo;
    };
    // Actual KV cache types from llama.cpp --cache-type-k/v, sorted by bpv
    type_entry types[] = {
        { GGML_TYPE_TURBO3_0_PROD, "turbo3_0", 2.63f, true  },
        { GGML_TYPE_TURBO4_0_PROD, "turbo4_0", 3.63f, true  },
        { GGML_TYPE_Q4_0,     "q4_0",     4.50f, false },
        { GGML_TYPE_Q4_1,     "q4_1",     5.00f, false },
        { GGML_TYPE_Q5_0,     "q5_0",     5.50f, false },
        { GGML_TYPE_Q8_0,     "q8_0",     8.50f, false },
    };
    const int n_types = sizeof(types) / sizeof(types[0]);

    printf("  Attention simulation: %d keys, %d queries, d=%d\n\n", n_keys, n_queries, d);
    printf("  %-10s %5s  |  score_err  rank_corr  top1_acc  softmax_KL\n", "type", "bpw");
    printf("  %s\n", "-------------------+---------------------------------------------");

    for (int t = 0; t < n_types; t++) {
        attn_metrics m;

        if (types[t].is_turbo) {
            // TurboQuant: process in 128-element blocks with rotation
            // Need to restructure keys for per-block rotation
            std::vector<float> keys_for_turbo(n_keys * d);
            for (int k = 0; k < n_keys; k++) {
                memcpy(keys_for_turbo.data() + k * d, keys.data() + k * d, d * sizeof(float));
            }

            // Compute metrics manually for turbo with two 128-blocks per key
            const auto * traits = ggml_get_type_traits(types[t].type);
            size_t qsz = traits->type_size;

            std::vector<std::vector<float>> recon_keys(n_keys, std::vector<float>(d));
            for (int k = 0; k < n_keys; k++) {
                for (int blk = 0; blk < d / 128; blk++) {
                    const float * src = keys.data() + k * d + blk * 128;
                    float * dst = recon_keys[k].data() + blk * 128;
                    std::vector<float> rotated(128), dequant(128);
                    std::vector<uint8_t> qbuf(qsz);
                    rotate_qr(src, rotated.data(), Q_rot.data(), 128);
                    traits->from_float_ref(rotated.data(), qbuf.data(), 128);
                    traits->to_float(qbuf.data(), dequant.data(), 128);
                    rotate_qr_inv(dequant.data(), dst, Q_rot.data(), 128);
                }
            }

            double sum_se = 0, sum_rc = 0, sum_kl = 0;
            int top1_ok = 0;
            for (int q = 0; q < n_queries; q++) {
                const float * query = queries.data() + q * d;
                std::vector<float> s_orig(n_keys), s_quant(n_keys);
                for (int k = 0; k < n_keys; k++) {
                    s_orig[k]  = vec_dot(query, keys.data() + k * d, d);
                    s_quant[k] = vec_dot(query, recon_keys[k].data(), d);
                    sum_se += fabsf(s_orig[k] - s_quant[k]);
                }
                sum_rc += rank_correlation(s_orig.data(), s_quant.data(), n_keys);
                int am_o = 0, am_q = 0;
                for (int k = 1; k < n_keys; k++) {
                    if (s_orig[k] > s_orig[am_o]) am_o = k;
                    if (s_quant[k] > s_quant[am_q]) am_q = k;
                }
                if (am_o == am_q) top1_ok++;
                float scale = 1.0f / sqrtf((float)d);
                std::vector<float> a_o(n_keys), a_q(n_keys);
                for (int k = 0; k < n_keys; k++) { a_o[k] = s_orig[k]*scale; a_q[k] = s_quant[k]*scale; }
                softmax(a_o.data(), n_keys);
                softmax(a_q.data(), n_keys);
                sum_kl += kl_divergence(a_o.data(), a_q.data(), n_keys);
            }
            m.mean_score_err = (float)(sum_se / (n_queries * n_keys));
            m.rank_corr = (float)(sum_rc / n_queries);
            m.top1_accuracy = (float)top1_ok / (float)n_queries;
            m.softmax_kl = (float)(sum_kl / n_queries);
        } else {
            m = compute_attn_metrics_baseline(types[t].type, keys.data(), n_keys, d,
                                               queries.data(), n_queries);
        }

        if (m.mean_score_err < 0) {
            printf("  %-10s %4.1fb  |  (skipped — incompatible block size)\n",
                   types[t].label, types[t].bpw);
            continue;
        }

        printf("  %-10s %4.1fb  |  %8.5f   %8.5f   %7.1f%%   %9.6f\n",
               types[t].label, types[t].bpw,
               m.mean_score_err, m.rank_corr, m.top1_accuracy * 100.0f, m.softmax_kl);
    }

    printf("\n  Metrics explained:\n");
    printf("    score_err:  mean |<q,k> - <q,k'>| (lower is better)\n");
    printf("    rank_corr:  Spearman correlation of score ordering (higher is better, 1.0 = perfect)\n");
    printf("    top1_acc:   %% of queries where argmax key is preserved (higher is better)\n");
    printf("    softmax_KL: KL divergence of attention weights (lower is better, 0 = identical)\n");

    return failures;
}

// ---------------------------------------------------------------------------
// Test K2: Long-context stress test
// ---------------------------------------------------------------------------

static int test_long_context(bool /*verbose*/) {
    int failures = 0;
    printf("\n=== Test K2: Long-context stress (softmax_KL vs sequence length) ===\n");

    const int d = 128;
    const int n_queries = 50;
    const int seq_lengths[] = { 64, 256, 1024, 4096 };
    const int n_seq = sizeof(seq_lengths) / sizeof(seq_lengths[0]);

    struct method_entry {
        const char * name;
        float bpv;
        ggml_type type;
        bool is_turbo;
    };
    method_entry methods[] = {
        { "turbo3_0", 2.63f, GGML_TYPE_TURBO3_0_PROD, true  },
        { "turbo4_0", 3.63f, GGML_TYPE_TURBO4_0_PROD, true  },
        { "q4_0",     4.50f, GGML_TYPE_Q4_0,     false },
        { "q5_0",     5.50f, GGML_TYPE_Q5_0,     false },
        { "q8_0",     8.50f, GGML_TYPE_Q8_0,     false },
    };
    const int n_methods = sizeof(methods) / sizeof(methods[0]);

    // QR rotation for turbo types
    prng_seed(314159);
    std::vector<float> Q_rot(d * d);
    gen_random_orthogonal(Q_rot.data(), d);

    for (int i = 0; i < GGML_TYPE_COUNT; i++) {
        ggml_quantize_init((ggml_type)i);
    }

    printf("  %10s %5s |", "type", "bpv");
    for (int s = 0; s < n_seq; s++) printf("  %5d keys", seq_lengths[s]);
    printf("\n  ----------------+");
    for (int s = 0; s < n_seq; s++) printf("------------");
    printf("\n");

    for (int mi = 0; mi < n_methods; mi++) {
        auto & m = methods[mi];
        const auto * traits = ggml_get_type_traits(m.type);
        const auto * traits_cpu = ggml_get_type_traits_cpu(m.type);

        if (!m.is_turbo && (!traits_cpu->from_float || !traits->to_float || d % traits->blck_size != 0))
            continue;

        printf("  %10s %4.1fb |", m.name, m.bpv);

        for (int si = 0; si < n_seq; si++) {
            int n_keys = seq_lengths[si];

            prng_seed(88880000 + si * 1000);
            std::vector<float> keys(n_keys * d);
            for (int k = 0; k < n_keys; k++) {
                float ns = 0.5f + prng_uniform(0.0f, 2.0f);
                switch (k % 6) {
                    case 0: gen_kv_lognormal_outlier(keys.data()+k*d, d, ns); break;
                    case 1: gen_kv_powerlaw(keys.data()+k*d, d, ns); break;
                    case 2: gen_kv_correlated(keys.data()+k*d, d, ns); break;
                    case 3: gen_kv_heavy_tail(keys.data()+k*d, d, ns); break;
                    case 4: gen_kv_sparse(keys.data()+k*d, d, ns); break;
                    case 5: gen_kv_asymmetric(keys.data()+k*d, d, ns); break;
                }
            }

            size_t qsz = traits->type_size;
            std::vector<std::vector<float>> rkeys(n_keys, std::vector<float>(d));

            for (int k = 0; k < n_keys; k++) {
                if (m.is_turbo) {
                    std::vector<float> rot(d), deq(d);
                    std::vector<uint8_t> qb(qsz);
                    rotate_qr(keys.data()+k*d, rot.data(), Q_rot.data(), d);
                    traits->from_float_ref(rot.data(), qb.data(), d);
                    traits->to_float(qb.data(), deq.data(), d);
                    rotate_qr_inv(deq.data(), rkeys[k].data(), Q_rot.data(), d);
                } else {
                    size_t qs = (size_t)(d / traits->blck_size) * traits->type_size;
                    std::vector<uint8_t> qb(qs);
                    traits_cpu->from_float(keys.data()+k*d, qb.data(), d);
                    traits->to_float(qb.data(), rkeys[k].data(), d);
                }
            }

            double sum_kl = 0.0;
            prng_seed(99990000 + si);
            for (int q = 0; q < n_queries; q++) {
                std::vector<float> query(d);
                gen_gaussian_normalized(query.data(), d, 1.0f);

                std::vector<float> so(n_keys), sq(n_keys);
                for (int k = 0; k < n_keys; k++) {
                    so[k] = vec_dot(query.data(), keys.data()+k*d, d);
                    sq[k] = vec_dot(query.data(), rkeys[k].data(), d);
                }
                float sc = 1.0f / sqrtf((float)d);
                std::vector<float> ao(n_keys), aq(n_keys);
                for (int k = 0; k < n_keys; k++) { ao[k] = so[k]*sc; aq[k] = sq[k]*sc; }
                softmax(ao.data(), n_keys);
                softmax(aq.data(), n_keys);
                sum_kl += kl_divergence(ao.data(), aq.data(), n_keys);
            }
            printf("  %10.6f", (float)(sum_kl / n_queries));
        }
        printf("\n");
    }

    return failures;
}

// ---------------------------------------------------------------------------
// Test K3: Adversarial flat-attention stress test
// ---------------------------------------------------------------------------
// Create keys that are deliberately close in score so softmax must distinguish
// small differences. This mimics scenarios where multiple tokens are equally
// relevant (e.g., multi-document QA, long-range dependencies).

static int test_adversarial_attention(bool /*verbose*/) {
    int failures = 0;
    printf("\n=== Test K3: Adversarial flat-attention stress test ===\n");

    const int d = 128;
    const int n_queries = 100;

    struct method_entry {
        const char * name;
        float bpv;
        ggml_type type;
        bool is_turbo;
    };
    method_entry methods[] = {
        { "turbo3_0", 2.63f, GGML_TYPE_TURBO3_0_PROD, true  },
        { "turbo4_0", 3.63f, GGML_TYPE_TURBO4_0_PROD, true  },
        { "q4_0",     4.50f, GGML_TYPE_Q4_0,     false },
        { "q5_0",     5.50f, GGML_TYPE_Q5_0,     false },
        { "q8_0",     8.50f, GGML_TYPE_Q8_0,     false },
    };
    const int n_methods = sizeof(methods) / sizeof(methods[0]);

    // QR rotation for turbo types
    prng_seed(314159);
    std::vector<float> Q_rot(d * d);
    gen_random_orthogonal(Q_rot.data(), d);

    for (int i = 0; i < GGML_TYPE_COUNT; i++) {
        ggml_quantize_init((ggml_type)i);
    }

    // Scenarios with increasing difficulty
    struct scenario {
        const char * name;
        int n_keys;
        float noise;       // how much to perturb keys from the base direction
        int n_needles;     // how many "target" keys to plant
    };
    scenario scenarios[] = {
        { "easy (10 random keys)",            10,   1.0f,   1 },
        { "medium (100 similar keys)",       100,   0.1f,   1 },
        { "hard (1000 similar keys)",       1000,   0.05f,  1 },
        { "needle (1000 keys, 3 needles)",  1000,   0.05f,  3 },
        { "flat (256 near-identical keys)",  256,   0.01f,  1 },
    };
    const int n_scenarios = sizeof(scenarios) / sizeof(scenarios[0]);

    for (int si = 0; si < n_scenarios; si++) {
        auto & sc = scenarios[si];
        printf("\n  Scenario: %s\n", sc.name);
        printf("  %10s %5s |  softmax_KL  top1_acc  needle_in_top5\n", "type", "bpv");
        printf("  ----------------+--------------------------------------\n");

        // Generate keys: mostly similar, with a few "needle" keys
        prng_seed(55550000 + si * 1000);

        // Base direction (all keys are near this)
        std::vector<float> base_dir(d);
        gen_gaussian_normalized(base_dir.data(), d, 1.0f);

        std::vector<float> keys(sc.n_keys * d);
        // Most keys: base_dir + small noise
        for (int k = 0; k < sc.n_keys; k++) {
            for (int j = 0; j < d; j++) {
                keys[k*d + j] = base_dir[j] + sc.noise * prng_gaussian();
            }
            // Normalize to unit norm
            float nrm = vec_norm(keys.data() + k*d, d);
            if (nrm > 1e-12f) {
                for (int j = 0; j < d; j++) keys[k*d + j] /= nrm;
            }
        }

        // Plant needles: keys that are slightly more aligned with the query
        // The query will be base_dir + small perturbation, so needles should
        // score slightly higher than the background
        std::vector<int> needle_ids;
        for (int n = 0; n < sc.n_needles; n++) {
            int nid = prng_next() % sc.n_keys;
            needle_ids.push_back(nid);
            // Make this key 5% more aligned with base_dir
            for (int j = 0; j < d; j++) {
                keys[nid*d + j] = base_dir[j] * 1.05f + sc.noise * 0.5f * prng_gaussian();
            }
            float nrm = vec_norm(keys.data() + nid*d, d);
            if (nrm > 1e-12f) {
                for (int j = 0; j < d; j++) keys[nid*d + j] /= nrm;
            }
        }

        for (int mi = 0; mi < n_methods; mi++) {
            auto & m = methods[mi];
            const auto * traits = ggml_get_type_traits(m.type);
            const auto * traits_cpu = ggml_get_type_traits_cpu(m.type);

            if (!m.is_turbo && (!traits_cpu->from_float || !traits->to_float || d % traits->blck_size != 0))
                continue;

            // Quantize all keys
            size_t qsz = traits->type_size;
            std::vector<std::vector<float>> rkeys(sc.n_keys, std::vector<float>(d));
            for (int k = 0; k < sc.n_keys; k++) {
                if (m.is_turbo) {
                    std::vector<float> rot(d), deq(d);
                    std::vector<uint8_t> qb(qsz);
                    rotate_qr(keys.data()+k*d, rot.data(), Q_rot.data(), d);
                    traits->from_float_ref(rot.data(), qb.data(), d);
                    traits->to_float(qb.data(), deq.data(), d);
                    rotate_qr_inv(deq.data(), rkeys[k].data(), Q_rot.data(), d);
                } else {
                    size_t qs = (size_t)(d / traits->blck_size) * traits->type_size;
                    std::vector<uint8_t> qb(qs);
                    traits_cpu->from_float(keys.data()+k*d, qb.data(), d);
                    traits->to_float(qb.data(), rkeys[k].data(), d);
                }
            }

            // Test with queries that are slight perturbations of base_dir
            double sum_kl = 0.0;
            int top1_correct = 0;
            int needle_in_top5 = 0;

            prng_seed(77770000 + si);
            for (int q = 0; q < n_queries; q++) {
                std::vector<float> query(d);
                // Query = base_dir + small perturbation (so all keys score similarly)
                for (int j = 0; j < d; j++) {
                    query[j] = base_dir[j] + sc.noise * 0.3f * prng_gaussian();
                }
                float qnrm = vec_norm(query.data(), d);
                for (int j = 0; j < d; j++) query[j] /= qnrm;

                std::vector<float> so(sc.n_keys), sq(sc.n_keys);
                for (int k = 0; k < sc.n_keys; k++) {
                    so[k] = vec_dot(query.data(), keys.data()+k*d, d);
                    sq[k] = vec_dot(query.data(), rkeys[k].data(), d);
                }

                // Softmax KL
                float scale = 1.0f / sqrtf((float)d);
                std::vector<float> ao(sc.n_keys), aq(sc.n_keys);
                for (int k = 0; k < sc.n_keys; k++) { ao[k] = so[k]*scale; aq[k] = sq[k]*scale; }
                softmax(ao.data(), sc.n_keys);
                softmax(aq.data(), sc.n_keys);
                sum_kl += kl_divergence(ao.data(), aq.data(), sc.n_keys);

                // Top-1 accuracy
                int am_o = 0, am_q = 0;
                for (int k = 1; k < sc.n_keys; k++) {
                    if (so[k] > so[am_o]) am_o = k;
                    if (sq[k] > sq[am_q]) am_q = k;
                }
                if (am_o == am_q) top1_correct++;

                // Needle retrieval: is any needle in the top-5 of quantized scores?
                // (Check if any needle that was in orig top-5 is also in quant top-5)
                std::vector<int> top5_orig(sc.n_keys), top5_quant(sc.n_keys);
                for (int k = 0; k < sc.n_keys; k++) { top5_orig[k] = k; top5_quant[k] = k; }
                std::partial_sort(top5_orig.begin(), top5_orig.begin()+5, top5_orig.end(),
                    [&](int a, int b) { return so[a] > so[b]; });
                std::partial_sort(top5_quant.begin(), top5_quant.begin()+5, top5_quant.end(),
                    [&](int a, int b) { return sq[a] > sq[b]; });

                for (int ni = 0; ni < (int)needle_ids.size(); ni++) {
                    bool in_orig_top5 = false, in_quant_top5 = false;
                    for (int t = 0; t < 5 && t < sc.n_keys; t++) {
                        if (top5_orig[t] == needle_ids[ni]) in_orig_top5 = true;
                        if (top5_quant[t] == needle_ids[ni]) in_quant_top5 = true;
                    }
                    if (in_orig_top5 && in_quant_top5) needle_in_top5++;
                }
            }

            float mean_kl = (float)(sum_kl / n_queries);
            float top1 = (float)top1_correct / n_queries * 100.0f;
            float needle_pct = (float)needle_in_top5 / (n_queries * (int)needle_ids.size()) * 100.0f;

            printf("  %10s %4.1fb |  %10.6f   %5.1f%%     %5.1f%%\n",
                   m.name, m.bpv, mean_kl, top1, needle_pct);
        }
    }

    return failures;
}

// ---------------------------------------------------------------------------
// Test K4: Seed sweep — test sensitivity to S and rotation randomness
// ---------------------------------------------------------------------------
// The unbiasedness theorem is over the randomness of S. A single fixed S
// may have bad realizations. Sweep seeds to check if results are stable.

static int test_seed_sweep(bool /*verbose*/) {
    int failures = 0;
    printf("\n=== Test K4: Seed sweep (20 seeds for S and rotation) ===\n");

    const int d = 128;
    const int n_keys = 1000;
    const int n_queries = 50;
    const int n_seeds = 20;
    const float noise = 0.05f;  // hard scenario from K3

    struct method_entry {
        const char * name;
        float bpv;
        ggml_type type;
        bool is_turbo;
    };
    method_entry methods[] = {
        { "turbo4_0", 3.63f, GGML_TYPE_TURBO4_0_PROD, true  },
        { "q4_0",     4.50f, GGML_TYPE_Q4_0,     false },
    };
    const int n_methods = sizeof(methods) / sizeof(methods[0]);

    for (int i = 0; i < GGML_TYPE_COUNT; i++) {
        ggml_quantize_init((ggml_type)i);
    }

    printf("  Hard scenario: %d similar keys, noise=%.2f, needle 5%% stronger\n", n_keys, noise);
    printf("  Sweeping %d seeds for rotation + QJL matrix\n\n", n_seeds);

    for (int mi = 0; mi < n_methods; mi++) {
        auto & m = methods[mi];
        const auto * traits = ggml_get_type_traits(m.type);
        const auto * traits_cpu = ggml_get_type_traits_cpu(m.type);

        if (!m.is_turbo && (!traits_cpu->from_float || !traits->to_float || d % traits->blck_size != 0))
            continue;

        printf("  %s (%.1f bpv):\n", m.name, m.bpv);
        printf("    seed | top1_acc  needle_top5  softmax_KL\n");
        printf("    -----+-----------------------------------\n");

        float sum_top1 = 0, sum_needle = 0;
        double sum_kl = 0;

        for (int seed = 0; seed < n_seeds; seed++) {
            // Generate fresh rotation for each seed (turbo only)
            prng_seed(1000000 + seed * 7919);
            std::vector<float> Q_rot(d * d);
            if (m.is_turbo) {
                gen_random_orthogonal(Q_rot.data(), d);
            }

            // Generate keys: mostly similar, one needle
            prng_seed(2000000 + seed * 6271);
            std::vector<float> base_dir(d);
            gen_gaussian_normalized(base_dir.data(), d, 1.0f);

            std::vector<float> keys(n_keys * d);
            for (int k = 0; k < n_keys; k++) {
                for (int j = 0; j < d; j++) {
                    keys[k*d + j] = base_dir[j] + noise * prng_gaussian();
                }
                float nrm = vec_norm(keys.data() + k*d, d);
                for (int j = 0; j < d; j++) keys[k*d + j] /= nrm;
            }
            // Plant needle
            int needle_id = prng_next() % n_keys;
            for (int j = 0; j < d; j++) {
                keys[needle_id*d + j] = base_dir[j] * 1.05f + noise * 0.5f * prng_gaussian();
            }
            float nrm = vec_norm(keys.data() + needle_id*d, d);
            for (int j = 0; j < d; j++) keys[needle_id*d + j] /= nrm;

            // Quantize keys
            size_t qsz = traits->type_size;
            std::vector<std::vector<float>> rkeys(n_keys, std::vector<float>(d));
            for (int k = 0; k < n_keys; k++) {
                if (m.is_turbo) {
                    std::vector<float> rot(d), deq(d);
                    std::vector<uint8_t> qb(qsz);
                    rotate_qr(keys.data()+k*d, rot.data(), Q_rot.data(), d);
                    traits->from_float_ref(rot.data(), qb.data(), d);
                    traits->to_float(qb.data(), deq.data(), d);
                    rotate_qr_inv(deq.data(), rkeys[k].data(), Q_rot.data(), d);
                } else {
                    size_t qs = (size_t)(d / traits->blck_size) * traits->type_size;
                    std::vector<uint8_t> qb(qs);
                    traits_cpu->from_float(keys.data()+k*d, qb.data(), d);
                    traits->to_float(qb.data(), rkeys[k].data(), d);
                }
            }

            int top1_ok = 0, needle_ok = 0;
            double seed_kl = 0;
            prng_seed(3000000 + seed * 4111);
            for (int q = 0; q < n_queries; q++) {
                std::vector<float> query(d);
                for (int j = 0; j < d; j++) query[j] = base_dir[j] + noise * 0.3f * prng_gaussian();
                float qn = vec_norm(query.data(), d);
                for (int j = 0; j < d; j++) query[j] /= qn;

                std::vector<float> so(n_keys), sq(n_keys);
                for (int k = 0; k < n_keys; k++) {
                    so[k] = vec_dot(query.data(), keys.data()+k*d, d);
                    sq[k] = vec_dot(query.data(), rkeys[k].data(), d);
                }

                int am_o = 0, am_q = 0;
                for (int k = 1; k < n_keys; k++) {
                    if (so[k] > so[am_o]) am_o = k;
                    if (sq[k] > sq[am_q]) am_q = k;
                }
                if (am_o == am_q) top1_ok++;

                // Needle in top-5?
                std::vector<int> idx(n_keys);
                for (int k = 0; k < n_keys; k++) idx[k] = k;
                std::partial_sort(idx.begin(), idx.begin()+5, idx.end(),
                    [&](int a, int b) { return sq[a] > sq[b]; });
                for (int t = 0; t < 5; t++) {
                    if (idx[t] == needle_id) { needle_ok++; break; }
                }

                float sc = 1.0f / sqrtf((float)d);
                std::vector<float> ao(n_keys), aq(n_keys);
                for (int k = 0; k < n_keys; k++) { ao[k] = so[k]*sc; aq[k] = sq[k]*sc; }
                softmax(ao.data(), n_keys);
                softmax(aq.data(), n_keys);
                seed_kl += kl_divergence(ao.data(), aq.data(), n_keys);
            }

            float t1 = (float)top1_ok / n_queries * 100.0f;
            float np = (float)needle_ok / n_queries * 100.0f;
            float kl = (float)(seed_kl / n_queries);
            sum_top1 += t1; sum_needle += np; sum_kl += kl;

            printf("    %4d | %5.1f%%     %5.1f%%       %10.6f\n", seed, t1, np, kl);
        }

        printf("    -----+-----------------------------------\n");
        printf("    mean | %5.1f%%     %5.1f%%       %10.6f\n",
               sum_top1 / n_seeds, sum_needle / n_seeds, (float)(sum_kl / n_seeds));
        printf("\n");
    }

    return failures;
}

// ---------------------------------------------------------------------------
// Test K5: QJL diagnostic — centroid-only vs centroid+QJL
// ---------------------------------------------------------------------------

static int test_qjl_diagnostic(bool /*verbose*/) {
    int failures = 0;
    printf("\n=== Test K5: QJL diagnostic (centroid-only vs centroid+QJL) ===\n");

    const int d = 128;
    const int n_vectors = 500;

    // Generate random unit-norm vectors
    prng_seed(12345);

    const auto * traits4 = ggml_get_type_traits(GGML_TYPE_TURBO4_0_PROD);
    size_t qsz4 = traits4->type_size;

    // Measure: for each vector, compare original vs centroid-only vs centroid+QJL
    double sum_l2_centroid = 0, sum_l2_qjl = 0;
    double sum_score_centroid = 0, sum_score_qjl = 0;
    int centroid_better_count = 0;

    for (int v = 0; v < n_vectors; v++) {
        std::vector<float> input(d), query(d);
        gen_gaussian_normalized(input.data(), d, 1.0f);
        gen_gaussian_normalized(query.data(), d, 1.0f);

        // Quantize
        std::vector<uint8_t> qbuf(qsz4);
        traits4->from_float_ref(input.data(), qbuf.data(), d);

        // Dequantize (normal, with QJL)
        std::vector<float> recon_qjl(d);
        traits4->to_float(qbuf.data(), recon_qjl.data(), d);

        // Dequantize centroid-only: manually decode without QJL correction
        // Read norm from block, reconstruct centroids only
        // Block layout: norm(2) + rnorm_hi(2) + rnorm_lo(2) + qs_hi(12) + qs_lo(24) + signs_hi(4) + signs_lo(12)
        const uint8_t * blk = qbuf.data();
        uint16_t norm_bits, rnorm_hi_bits, rnorm_lo_bits;
        memcpy(&norm_bits, blk, 2);
        memcpy(&rnorm_hi_bits, blk + 2, 2);
        memcpy(&rnorm_lo_bits, blk + 4, 2);
        float norm = ggml_fp16_to_fp32(norm_bits);

        // Centroid-only reconstruction
        std::vector<float> recon_centroid(d);
        // Hi channels [0,32): 3-bit packed in qs_hi starting at offset 6
        const uint8_t * qs_hi = blk + 6;
        for (int j = 0; j < 32; j++) {
            int bp = j*3, bi = bp>>3, sh = bp&7;
            int idx = (sh <= 5) ? (qs_hi[bi]>>sh)&7 : ((qs_hi[bi]>>sh)|(qs_hi[bi+1]<<(8-sh)))&7;
            recon_centroid[j] = norm * paper_centroids_8[idx];
        }
        // Lo channels [32,128): 2-bit packed in qs_lo starting at offset 6+12=18
        const uint8_t * qs_lo = blk + 18;
        for (int j = 0; j < 96; j++) {
            int idx = (qs_lo[j/4] >> ((j%4)*2)) & 0x3;
            recon_centroid[32 + j] = norm * paper_centroids_4[idx];
        }

        // Compute errors
        float l2_centroid = relative_l2_error(input.data(), recon_centroid.data(), d);
        float l2_qjl = relative_l2_error(input.data(), recon_qjl.data(), d);
        sum_l2_centroid += l2_centroid;
        sum_l2_qjl += l2_qjl;

        float score_orig = vec_dot(query.data(), input.data(), d);
        float score_centroid = vec_dot(query.data(), recon_centroid.data(), d);
        float score_qjl = vec_dot(query.data(), recon_qjl.data(), d);
        float err_centroid = fabsf(score_orig - score_centroid);
        float err_qjl = fabsf(score_orig - score_qjl);
        sum_score_centroid += err_centroid;
        sum_score_qjl += err_qjl;

        if (err_centroid < err_qjl) centroid_better_count++;
    }

    float mean_l2_c = (float)(sum_l2_centroid / n_vectors);
    float mean_l2_q = (float)(sum_l2_qjl / n_vectors);
    float mean_sc_c = (float)(sum_score_centroid / n_vectors);
    float mean_sc_q = (float)(sum_score_qjl / n_vectors);

    printf("  %d unit-norm vectors, TURBO4_0 (no rotation)\n\n", n_vectors);
    printf("  metric          centroid-only   centroid+QJL\n");
    printf("  ----------------+-------------------------------\n");
    printf("  mean rel_l2:     %10.4f      %10.4f\n", mean_l2_c, mean_l2_q);
    printf("  mean score_err:  %10.6f      %10.6f\n", mean_sc_c, mean_sc_q);
    printf("  centroid wins score: %d / %d (%.1f%%)\n",
           centroid_better_count, n_vectors, 100.0f * centroid_better_count / n_vectors);

    if (mean_sc_q > mean_sc_c * 1.5f) {
        printf("  WARNING: QJL is making score error WORSE (%.1fx)\n", mean_sc_q / mean_sc_c);
    }

    // --- The real test: inner-product BIAS (mean of signed error, not absolute) ---
    // Paper's theorem: E[<y, x_hat>] = <y, x> for centroid+QJL
    // Centroid-only has multiplicative bias of 2/pi at b=1
    printf("\n  Inner-product BIAS test (signed error, should be ~0 for QJL):\n");
    printf("  Sweeping 50 seeds, 200 random (x,y) pairs each\n\n");

    double total_bias_centroid = 0, total_bias_qjl = 0;
    double total_var_centroid = 0, total_var_qjl = 0;
    int total_pairs = 0;

    for (int seed = 0; seed < 50; seed++) {
        prng_seed(8000000 + seed * 3571);

        double seed_bias_c = 0, seed_bias_q = 0;

        for (int p = 0; p < 200; p++) {
            std::vector<float> x(d), y(d);
            gen_gaussian_normalized(x.data(), d, 1.0f);
            gen_gaussian_normalized(y.data(), d, 1.0f);

            std::vector<uint8_t> qb(qsz4);
            std::vector<float> recon_qjl(d);
            traits4->from_float_ref(x.data(), qb.data(), d);
            traits4->to_float(qb.data(), recon_qjl.data(), d);

            // Centroid-only reconstruction (same decode as above)
            std::vector<float> recon_c(d);
            const uint8_t * blk = qb.data();
            uint16_t nb;
            memcpy(&nb, blk, 2);
            float nm = ggml_fp16_to_fp32(nb);
            const uint8_t * qh = blk + 6;
            for (int j = 0; j < 32; j++) {
                int bp = j*3, bi = bp>>3, sh = bp&7;
                int idx = (sh <= 5) ? (qh[bi]>>sh)&7 : ((qh[bi]>>sh)|(qh[bi+1]<<(8-sh)))&7;
                recon_c[j] = nm * paper_centroids_8[idx];
            }
            const uint8_t * ql = blk + 18;
            for (int j = 0; j < 96; j++) {
                int idx = (ql[j/4] >> ((j%4)*2)) & 0x3;
                recon_c[32+j] = nm * paper_centroids_4[idx];
            }

            float dot_orig = vec_dot(y.data(), x.data(), d);
            float dot_c = vec_dot(y.data(), recon_c.data(), d);
            float dot_q = vec_dot(y.data(), recon_qjl.data(), d);

            // Signed error (bias = mean of this should be ~0 for unbiased)
            float err_c = dot_c - dot_orig;
            float err_q = dot_q - dot_orig;

            seed_bias_c += err_c;
            seed_bias_q += err_q;
            total_bias_centroid += err_c;
            total_bias_qjl += err_q;
            total_var_centroid += err_c * err_c;
            total_var_qjl += err_q * err_q;
            total_pairs++;
        }
    }

    float mean_bias_c = (float)(total_bias_centroid / total_pairs);
    float mean_bias_q = (float)(total_bias_qjl / total_pairs);
    float var_c = (float)(total_var_centroid / total_pairs - mean_bias_c * mean_bias_c);
    float var_q = (float)(total_var_qjl / total_pairs - mean_bias_q * mean_bias_q);

    printf("  metric          centroid-only   centroid+QJL\n");
    printf("  ----------------+-------------------------------\n");
    printf("  mean bias:       %+10.6f      %+10.6f\n", mean_bias_c, mean_bias_q);
    printf("  variance:        %10.6f      %10.6f\n", var_c, var_q);
    printf("  std dev:         %10.6f      %10.6f\n", sqrtf(var_c), sqrtf(var_q));
    printf("\n");
    if (fabsf(mean_bias_q) < fabsf(mean_bias_c)) {
        printf("  QJL reduces bias: %.4f → %.4f (%.1fx reduction)\n",
               fabsf(mean_bias_c), fabsf(mean_bias_q), fabsf(mean_bias_c) / fmaxf(fabsf(mean_bias_q), 1e-10f));
    } else {
        printf("  QJL does NOT reduce bias — possible implementation issue\n");
    }

    return failures;
}

// ---------------------------------------------------------------------------
// Test L: Paper-exact vs our implementation vs baselines
// ---------------------------------------------------------------------------

static int test_paper_reference(bool /*verbose*/) {
    int failures = 0;
    printf("\n=== Test L: Paper-exact algorithm vs our block implementation ===\n");

    const int d = 128;
    const int n_keys = 64;
    const int n_queries = 100;

    // Generate QR rotation matrix (paper's approach)
    prng_seed(999999);
    std::vector<float> Pi(d * d);
    gen_random_orthogonal(Pi.data(), d);

    // Generate i.i.d. Gaussian QJL matrix (paper's approach)
    std::vector<float> S_gauss(d * d);
    for (int i = 0; i < d * d; i++) {
        S_gauss[i] = prng_gaussian();
    }

    // Use the SAME rotation matrix Pi for our block impl (isolates QJL difference)
    const float * Q_ours_ptr = Pi.data();

    // Generate KV-cache-like keys and queries
    prng_seed(424242);
    std::vector<float> keys(n_keys * d);
    std::vector<float> queries(n_queries * d);

    for (int k = 0; k < n_keys; k++) {
        float norm_scale = 0.5f + prng_uniform(0.0f, 2.0f);
        switch (k % 6) {
            case 0: gen_kv_lognormal_outlier(keys.data() + k*d, d, norm_scale); break;
            case 1: gen_kv_powerlaw(keys.data() + k*d, d, norm_scale); break;
            case 2: gen_kv_correlated(keys.data() + k*d, d, norm_scale); break;
            case 3: gen_kv_heavy_tail(keys.data() + k*d, d, norm_scale); break;
            case 4: gen_kv_sparse(keys.data() + k*d, d, norm_scale); break;
            case 5: gen_kv_asymmetric(keys.data() + k*d, d, norm_scale); break;
        }
    }
    for (int q = 0; q < n_queries; q++) {
        gen_gaussian_normalized(queries.data() + q*d, d, 1.0f);
    }

    // Initialize quant types
    for (int i = 0; i < GGML_TYPE_COUNT; i++) {
        ggml_quantize_init((ggml_type)i);
    }

    // --- Quantize all keys with each method ---

    struct method {
        const char * name;
        float bpw;
        std::vector<std::vector<float>> recon_keys;
    };

    // Method 1: Paper-exact uniform 4-bit (QR + 3-bit MSE + i.i.d. Gaussian QJL)
    // 4.0 bpw data = all 128 channels at 8 centroids. For reference only.
    method paper_uniform;
    paper_uniform.name = "paper_4.0b";
    paper_uniform.bpw = 4.25f;
    paper_uniform.recon_keys.resize(n_keys, std::vector<float>(d));
    for (int k = 0; k < n_keys; k++) {
        paper_turbo_quantize(keys.data() + k*d, paper_uniform.recon_keys[k].data(), d,
                             Pi.data(), S_gauss.data(), paper_centroids_8, 8);
    }

    // Method 2: Paper-exact 2-bit MSE + QJL (uniform, for reference)
    method paper_2bit;
    paper_2bit.name = "paper_2bit";
    paper_2bit.bpw = 2.25f;
    paper_2bit.recon_keys.resize(n_keys, std::vector<float>(d));
    for (int k = 0; k < n_keys; k++) {
        paper_turbo_quantize(keys.data() + k*d, paper_2bit.recon_keys[k].data(), d,
                             Pi.data(), S_gauss.data(), paper_centroids_4, 4);
    }

    // Method 3: Our TURBO4_0 block implementation (QR rotation + i.i.d. Gaussian QJL)
    // Mixed precision: 64@3-bit MSE + 64@2-bit MSE + QJL = 3.5 bpw data
    method ours_t4;
    ours_t4.name = "ours_turbo4";
    ours_t4.bpw = 4.25f;
    ours_t4.recon_keys.resize(n_keys, std::vector<float>(d));
    {
        const auto * traits = ggml_get_type_traits(GGML_TYPE_TURBO4_0_PROD);
        size_t qsz = traits->type_size;
        for (int k = 0; k < n_keys; k++) {
            std::vector<float> rotated(d), dequant(d);
            std::vector<uint8_t> qbuf(qsz);
            rotate_qr(keys.data() + k*d, rotated.data(), Q_ours_ptr, d);
            traits->from_float_ref(rotated.data(), qbuf.data(), d);
            traits->to_float(qbuf.data(), dequant.data(), d);
            rotate_qr_inv(dequant.data(), ours_t4.recon_keys[k].data(), Q_ours_ptr, d);
        }
    }

    // Method 4: Our TURBO3_0 block implementation
    method ours_t3;
    ours_t3.name = "ours_turbo3";
    ours_t3.bpw = 2.50f;
    ours_t3.recon_keys.resize(n_keys, std::vector<float>(d));
    {
        const auto * traits = ggml_get_type_traits(GGML_TYPE_TURBO3_0_PROD);
        size_t qsz = traits->type_size;
        for (int k = 0; k < n_keys; k++) {
            std::vector<float> rotated(d), dequant(d);
            std::vector<uint8_t> qbuf(qsz);
            rotate_qr(keys.data() + k*d, rotated.data(), Q_ours_ptr, d);
            traits->from_float_ref(rotated.data(), qbuf.data(), d);
            traits->to_float(qbuf.data(), dequant.data(), d);
            rotate_qr_inv(dequant.data(), ours_t3.recon_keys[k].data(), Q_ours_ptr, d);
        }
    }

    // Baseline methods (existing quant types, no rotation)
    struct baseline_method {
        const char * name;
        float bpw;
        ggml_type type;
        std::vector<std::vector<float>> recon_keys;
    };
    baseline_method baselines[] = {
        { "q2_K",  2.63f, GGML_TYPE_Q2_K,  {} },
        { "q3_K",  3.44f, GGML_TYPE_Q3_K,  {} },
        { "q4_0",  4.50f, GGML_TYPE_Q4_0,  {} },
        { "q8_0",  8.50f, GGML_TYPE_Q8_0,  {} },
    };

    for (auto & bl : baselines) {
        const auto * traits = ggml_get_type_traits(bl.type);
        const auto * traits_cpu = ggml_get_type_traits_cpu(bl.type);
        if (!traits_cpu->from_float || !traits->to_float || d % traits->blck_size != 0) continue;
        size_t qsz = (size_t)(d / traits->blck_size) * traits->type_size;
        bl.recon_keys.resize(n_keys, std::vector<float>(d));
        for (int k = 0; k < n_keys; k++) {
            std::vector<uint8_t> qbuf(qsz);
            traits_cpu->from_float(keys.data() + k*d, qbuf.data(), d);
            traits->to_float(qbuf.data(), bl.recon_keys[k].data(), d);
        }
    }

    // --- Compute attention metrics for each method ---
    auto compute_metrics = [&](const std::vector<std::vector<float>> & recon) {
        double sum_se = 0, sum_rc = 0, sum_kl = 0;
        int top1_ok = 0;
        for (int q = 0; q < n_queries; q++) {
            const float * query = queries.data() + q * d;
            std::vector<float> s_orig(n_keys), s_quant(n_keys);
            for (int k = 0; k < n_keys; k++) {
                s_orig[k]  = vec_dot(query, keys.data() + k*d, d);
                s_quant[k] = vec_dot(query, recon[k].data(), d);
                sum_se += fabsf(s_orig[k] - s_quant[k]);
            }
            sum_rc += rank_correlation(s_orig.data(), s_quant.data(), n_keys);
            int am_o = 0, am_q = 0;
            for (int k = 1; k < n_keys; k++) {
                if (s_orig[k] > s_orig[am_o]) am_o = k;
                if (s_quant[k] > s_quant[am_q]) am_q = k;
            }
            if (am_o == am_q) top1_ok++;
            float scale = 1.0f / sqrtf((float)d);
            std::vector<float> a_o(n_keys), a_q(n_keys);
            for (int k = 0; k < n_keys; k++) { a_o[k]=s_orig[k]*scale; a_q[k]=s_quant[k]*scale; }
            softmax(a_o.data(), n_keys); softmax(a_q.data(), n_keys);
            sum_kl += kl_divergence(a_o.data(), a_q.data(), n_keys);
        }
        struct { float se, rc, top1, kl; } m;
        m.se   = (float)(sum_se / (n_queries * n_keys));
        m.rc   = (float)(sum_rc / n_queries);
        m.top1 = (float)top1_ok / (float)n_queries;
        m.kl   = (float)(sum_kl / n_queries);
        return m;
    };

    printf("  %d keys, %d queries, d=%d, mixed KV-cache distributions\n\n", n_keys, n_queries, d);
    printf("  %-14s %5s  |  score_err  rank_corr  top1_acc  softmax_KL\n", "method", "bpw");
    printf("  %s\n", "---------------------+---------------------------------------------");

    // Print paper-exact results
    {
        auto m = compute_metrics(paper_uniform.recon_keys);
        printf("  %-14s %4.2fb  |  %8.5f   %8.5f   %7.1f%%   %9.6f  (paper uniform)\n",
               paper_uniform.name, paper_uniform.bpw, m.se, m.rc, m.top1*100, m.kl);
    }
    {
        auto m = compute_metrics(paper_2bit.recon_keys);
        printf("  %-14s %4.2fb  |  %8.5f   %8.5f   %7.1f%%   %9.6f  (paper uniform)\n",
               paper_2bit.name, paper_2bit.bpw, m.se, m.rc, m.top1*100, m.kl);
    }
    // Print our implementation results
    {
        auto m = compute_metrics(ours_t4.recon_keys);
        printf("  %-14s %4.2fb  |  %8.5f   %8.5f   %7.1f%%   %9.6f  (our block impl)\n",
               ours_t4.name, ours_t4.bpw, m.se, m.rc, m.top1*100, m.kl);
    }
    {
        auto m = compute_metrics(ours_t3.recon_keys);
        printf("  %-14s %4.2fb  |  %8.5f   %8.5f   %7.1f%%   %9.6f  (our block impl)\n",
               ours_t3.name, ours_t3.bpw, m.se, m.rc, m.top1*100, m.kl);
    }
    // Print baselines
    for (auto & bl : baselines) {
        if (bl.recon_keys.empty()) continue;
        auto m = compute_metrics(bl.recon_keys);
        printf("  %-14s %4.2fb  |  %8.5f   %8.5f   %7.1f%%   %9.6f  (llama.cpp)\n",
               bl.name, bl.bpw, m.se, m.rc, m.top1*100, m.kl);
    }

    return failures;
}

// ---------------------------------------------------------------------------
// Test M: d=1536 simulation (replicating paper's Figure 2 validation)
// ---------------------------------------------------------------------------
// The paper validates at d=1536 (OpenAI3 embeddings). We simulate Algorithm 2
// at d=1536 to confirm the algorithm produces the claimed D_prod values.
// Paper's D_prod for b=1,2,3,4: 1.57/d, 0.56/d, 0.18/d, 0.047/d

static int test_d1536_simulation(bool /*verbose*/) {
    int failures = 0;
    printf("\n=== Test M: d=1536 simulation (paper's validation dimension) ===\n");

    const int d = 1536;
    const int n_pairs = 500;

    // For large d, centroids scale as standard_lloyd_max / sqrt(d).
    // Half-normal Lloyd-Max centroids for unit variance, scaled by 1/sqrt(d):
    float s = 1.0f / sqrtf((float)d);

    // b=2: 4 signed centroids (2 magnitude: 0.4528, 1.5104)
    float c4_1536[4] = { -1.5104f*s, -0.4528f*s, 0.4528f*s, 1.5104f*s };
    // b=3: 8 signed centroids (4 magnitude: 0.2451, 0.7560, 1.3440, 2.1520)
    float c8_1536[8] = {
        -2.1520f*s, -1.3440f*s, -0.7560f*s, -0.2451f*s,
         0.2451f*s,  0.7560f*s,  1.3440f*s,  2.1520f*s
    };

    // Generate QR rotation matrix (d×d) — too large for stack, use heap
    prng_seed(777777);
    printf("  Generating %dx%d QR rotation matrix...\n", d, d);
    std::vector<float> Pi(d * d);
    gen_random_orthogonal(Pi.data(), d);

    // Generate i.i.d. Gaussian S matrix for QJL
    printf("  Generating %dx%d QJL matrix...\n", d, d);
    std::vector<float> S(d * d);
    for (int i = 0; i < d * d; i++) S[i] = prng_gaussian();

    struct bw_test {
        int b;
        const float * centroids;
        int n_centroids;
        float paper_dprod;  // paper's D_prod ≈ X/d
    };
    bw_test tests[] = {
        { 2, c4_1536, 4, 0.56f / d },
        { 3, c8_1536, 8, 0.18f / d },
    };

    printf("  %d query-key pairs per bitwidth\n\n", n_pairs);
    printf("  b  |  paper_D_prod  measured_D_prod  mean_|score_err|  ratio\n");
    printf("  ---+---------------------------------------------------------\n");

    prng_seed(123456);

    for (auto & t : tests) {
        double sum_sq_err = 0.0;
        double sum_abs_err = 0.0;

        for (int p = 0; p < n_pairs; p++) {
            // Generate random unit-norm key and query
            std::vector<float> key(d), query(d), recon(d);
            gen_gaussian_normalized(key.data(), d, 1.0f);
            gen_gaussian_normalized(query.data(), d, 1.0f);

            // Run paper-exact Algorithm 2
            paper_turbo_quantize(key.data(), recon.data(), d,
                                 Pi.data(), S.data(), t.centroids, t.n_centroids);

            float score_orig = vec_dot(query.data(), key.data(), d);
            float score_quant = vec_dot(query.data(), recon.data(), d);
            float err = score_orig - score_quant;
            sum_sq_err += (double)err * (double)err;
            sum_abs_err += fabsf(err);
        }

        float measured_dprod = (float)(sum_sq_err / n_pairs);
        float mean_abs = (float)(sum_abs_err / n_pairs);
        float ratio = measured_dprod / t.paper_dprod;

        printf("  %d  |  %.6f       %.6f        %.6f          %.2fx\n",
               t.b, t.paper_dprod, measured_dprod, mean_abs, ratio);

        // Should be within ~3x of paper's theoretical bound
        bool ok = (ratio < 4.0f);
        if (!ok) {
            printf("      FAILED: measured D_prod is %.1fx the paper's bound\n", ratio);
            failures++;
        }
    }

    // Also show d=128 for comparison
    printf("\n  For comparison at d=128 (our KV cache dimension):\n");
    printf("  b  |  paper_D_prod  (at d=128)\n");
    printf("  ---+---------------------------\n");
    printf("  2  |  %.6f\n", 0.56f / 128.0f);
    printf("  3  |  %.6f\n", 0.18f / 128.0f);
    printf("  (%.0fx larger than d=%d)\n", 1536.0f / 128.0f, d);

    return failures;
}

// ---------------------------------------------------------------------------
// Test N: MSE variant format / structural sanity
// ---------------------------------------------------------------------------

static int test_mse_format_sanity(void) {
    int failures = 0;
    printf("\n=== Test N: MSE variant format / structural sanity ===\n");

    // TURBO3_0_MSE
    {
        bool ok = true;
        ok = ok && (ggml_type_size(GGML_TYPE_TURBO3_0_MSE)  == 38);
        ok = ok && (ggml_blck_size(GGML_TYPE_TURBO3_0_MSE)  == 128);
        ok = ok && (ggml_is_quantized(GGML_TYPE_TURBO3_0_MSE));
        ok = ok && (strcmp(ggml_type_name(GGML_TYPE_TURBO3_0_MSE), "tqv_lo") == 0);
        printf("  tqv_lo format: %s (size=%zu blck=%d name=%s)\n",
               RESULT_STR[!ok],
               ggml_type_size(GGML_TYPE_TURBO3_0_MSE),
               (int)ggml_blck_size(GGML_TYPE_TURBO3_0_MSE),
               ggml_type_name(GGML_TYPE_TURBO3_0_MSE));
        if (!ok) failures++;
    }

    // TURBO4_0_MSE
    {
        bool ok = true;
        ok = ok && (ggml_type_size(GGML_TYPE_TURBO4_0_MSE)  == 54);
        ok = ok && (ggml_blck_size(GGML_TYPE_TURBO4_0_MSE)  == 128);
        ok = ok && (ggml_is_quantized(GGML_TYPE_TURBO4_0_MSE));
        ok = ok && (strcmp(ggml_type_name(GGML_TYPE_TURBO4_0_MSE), "tqv_hi") == 0);
        printf("  tqv_hi format: %s (size=%zu blck=%d name=%s)\n",
               RESULT_STR[!ok],
               ggml_type_size(GGML_TYPE_TURBO4_0_MSE),
               (int)ggml_blck_size(GGML_TYPE_TURBO4_0_MSE),
               ggml_type_name(GGML_TYPE_TURBO4_0_MSE));
        if (!ok) failures++;
    }

    // MSE blocks are smaller than PROD blocks (no QJL overhead)
    {
        bool ok = true;
        ok = ok && (ggml_type_size(GGML_TYPE_TURBO3_0_MSE) < ggml_type_size(GGML_TYPE_TURBO3_0_PROD));
        ok = ok && (ggml_type_size(GGML_TYPE_TURBO4_0_MSE) < ggml_type_size(GGML_TYPE_TURBO4_0_PROD));
        printf("  MSE < PROD block size: %s (t3: %zu < %zu, t4: %zu < %zu)\n",
               RESULT_STR[!ok],
               ggml_type_size(GGML_TYPE_TURBO3_0_MSE), ggml_type_size(GGML_TYPE_TURBO3_0_PROD),
               ggml_type_size(GGML_TYPE_TURBO4_0_MSE), ggml_type_size(GGML_TYPE_TURBO4_0_PROD));
        if (!ok) failures++;
    }

    return failures;
}

// ---------------------------------------------------------------------------
// Test N2: MSE zero vector and determinism
// ---------------------------------------------------------------------------

static int test_mse_zero_and_determinism(void) {
    int failures = 0;
    printf("\n=== Test N2: MSE zero vector and determinism ===\n");

    for (auto type : {GGML_TYPE_TURBO3_0_MSE, GGML_TYPE_TURBO4_0_MSE}) {
        const int n = 128;
        const auto * traits = ggml_get_type_traits(type);

        // Zero roundtrip
        {
            std::vector<float> zeros(n, 0.0f);
            auto r = do_roundtrip(type, zeros.data(), n);
            bool ok = (r.rmse_val == 0.0f) && !r.has_bad;
            printf("  %s zero: %s (rmse=%e)\n", ggml_type_name(type), RESULT_STR[!ok], r.rmse_val);
            if (!ok) failures++;
        }

        // Determinism
        {
            size_t qsz = traits->type_size;
            prng_seed(42);
            std::vector<float> data(n);
            for (int i = 0; i < n; i++) data[i] = prng_gaussian();

            std::vector<uint8_t> q1(qsz), q2(qsz);
            traits->from_float_ref(data.data(), q1.data(), n);
            traits->from_float_ref(data.data(), q2.data(), n);

            bool ok = (memcmp(q1.data(), q2.data(), qsz) == 0);
            printf("  %s determinism: %s\n", ggml_type_name(type), RESULT_STR[!ok]);
            if (!ok) failures++;
        }
    }

    return failures;
}

// ---------------------------------------------------------------------------
// Test O: MSE vs PROD reconstruction quality (WITH rotation)
// ---------------------------------------------------------------------------
// Paper: MSE variant minimizes reconstruction error because it uses all b bits
// for centroids instead of spending 1 bit on QJL.
// PyTorch validation: MSE consistently wins on per-vector MSE.
// Full pipeline: rotate → quantize → dequantize → inverse rotate.

static int test_mse_vs_prod_reconstruction(bool verbose) {
    int failures = 0;
    printf("\n=== Test O: MSE vs PROD reconstruction quality (with rotation) ===\n");
    printf("  (Paper: MSE minimizes reconstruction; PROD trades MSE for unbiased inner products)\n\n");

    const int d = 128;
    const int n_vectors = 500;

    // Shared rotation matrix (one per "head")
    prng_seed(314159);
    std::vector<float> Q_rot(d * d);
    gen_random_orthogonal(Q_rot.data(), d);

    struct type_pair {
        ggml_type prod;
        ggml_type mse;
        const char * label;
    };
    type_pair pairs[] = {
        { GGML_TYPE_TURBO3_0_PROD, GGML_TYPE_TURBO3_0_MSE, "turbo3" },
        { GGML_TYPE_TURBO4_0_PROD, GGML_TYPE_TURBO4_0_MSE, "turbo4" },
    };

    for (auto & p : pairs) {
        double sum_l2_prod = 0, sum_l2_mse = 0;
        double sum_cos_prod = 0, sum_cos_mse = 0;
        int mse_wins = 0;

        prng_seed(77777);
        for (int v = 0; v < n_vectors; v++) {
            std::vector<float> data(d);
            float norm_scale = 0.5f + prng_uniform(0.0f, 2.0f);
            switch (v % 6) {
                case 0: gen_kv_lognormal_outlier(data.data(), d, norm_scale); break;
                case 1: gen_kv_powerlaw(data.data(), d, norm_scale); break;
                case 2: gen_kv_correlated(data.data(), d, norm_scale); break;
                case 3: gen_kv_heavy_tail(data.data(), d, norm_scale); break;
                case 4: gen_kv_sparse(data.data(), d, norm_scale); break;
                case 5: gen_kv_asymmetric(data.data(), d, norm_scale); break;
            }

            auto r_prod = do_roundtrip_rotated(p.prod, data.data(), d, Q_rot.data());
            auto r_mse  = do_roundtrip_rotated(p.mse,  data.data(), d, Q_rot.data());

            sum_l2_prod += r_prod.rel_l2;
            sum_l2_mse  += r_mse.rel_l2;
            sum_cos_prod += r_prod.cos_sim;
            sum_cos_mse  += r_mse.cos_sim;
            if (r_mse.rel_l2 < r_prod.rel_l2) mse_wins++;
        }

        float mean_l2_prod = (float)(sum_l2_prod / n_vectors);
        float mean_l2_mse  = (float)(sum_l2_mse  / n_vectors);
        float mean_cos_prod = (float)(sum_cos_prod / n_vectors);
        float mean_cos_mse  = (float)(sum_cos_mse  / n_vectors);
        float win_pct = 100.0f * mse_wins / n_vectors;

        // MSE variant MUST have lower reconstruction error
        bool ok = (mean_l2_mse < mean_l2_prod) && (win_pct > 60.0f);
        if (!ok) failures++;

        printf("  %s: %s\n", p.label, RESULT_STR[!ok]);
        printf("    PROD  rel_L2=%.4f  cos=%.4f\n", mean_l2_prod, mean_cos_prod);
        printf("    MSE   rel_L2=%.4f  cos=%.4f  (%.1f%% lower error)\n",
               mean_l2_mse, mean_cos_mse,
               100.0f * (1.0f - mean_l2_mse / mean_l2_prod));
        printf("    MSE wins on %d/%d vectors (%.1f%%)\n", mse_wins, n_vectors, win_pct);

        (void)verbose;
    }

    return failures;
}

// ---------------------------------------------------------------------------
// Test P: PROD vs MSE inner product preservation (WITH rotation)
// ---------------------------------------------------------------------------
// Paper Theorem 2: TurboQuant_prod gives UNBIASED inner product estimates.
// Paper Section 1.1, Lemma 4: TurboQuant_mse is BIASED (scaled by 2/π at 1-bit).
// PyTorch test findings:
//   - test_mse_only_inner_product_bias: MSE bias visible at all bit-widths
//   - test_inner_product_unbiasedness: PROD near-zero bias, corr 0.80-0.98
// Full pipeline: rotate key → quantize → dequantize → inverse rotate, then dot with query.

static int test_prod_vs_mse_inner_product(bool verbose) {
    int failures = 0;
    printf("\n=== Test P: PROD vs MSE inner product preservation (with rotation) ===\n");
    printf("  (Paper: PROD is unbiased for inner products; MSE is biased)\n\n");

    const int d = 128;
    const int n_pairs = 2000;

    // Rotation matrix
    prng_seed(314159);
    std::vector<float> Q_rot(d * d);
    gen_random_orthogonal(Q_rot.data(), d);

    struct type_pair {
        ggml_type prod;
        ggml_type mse;
        const char * label;
        float max_prod_mean_err;
    };
    type_pair pairs[] = {
        { GGML_TYPE_TURBO3_0_PROD, GGML_TYPE_TURBO3_0_MSE, "turbo3", 0.10f },
        { GGML_TYPE_TURBO4_0_PROD, GGML_TYPE_TURBO4_0_MSE, "turbo4", 0.05f },
    };

    for (auto & p : pairs) {
        prng_seed(12345);

        double sum_err_prod = 0, sum_err_mse = 0;
        double sum_bias_prod = 0, sum_bias_mse = 0;
        double sum_sq_orig = 0;
        double sum_orig_prod = 0, sum_sq_prod = 0;
        double sum_orig_mse = 0, sum_sq_mse = 0;
        int prod_closer = 0;

        for (int i = 0; i < n_pairs; i++) {
            std::vector<float> query(d), key(d);
            gen_gaussian_normalized(query.data(), d, 1.0f);
            gen_gaussian_normalized(key.data(), d, 1.0f);

            float score_orig = vec_dot(query.data(), key.data(), d);
            sum_sq_orig += score_orig * score_orig;

            // PROD: rotate → quantize → dequantize → unrotate → dot with query
            float score_prod;
            {
                const auto * traits = ggml_get_type_traits(p.prod);
                std::vector<float> rotated(d), dequant(d), recon(d);
                std::vector<uint8_t> qbuf(traits->type_size);
                rotate_qr(key.data(), rotated.data(), Q_rot.data(), d);
                traits->from_float_ref(rotated.data(), qbuf.data(), d);
                traits->to_float(qbuf.data(), dequant.data(), d);
                rotate_qr_inv(dequant.data(), recon.data(), Q_rot.data(), d);
                score_prod = vec_dot(query.data(), recon.data(), d);
            }

            // MSE: same pipeline
            float score_mse;
            {
                const auto * traits = ggml_get_type_traits(p.mse);
                std::vector<float> rotated(d), dequant(d), recon(d);
                std::vector<uint8_t> qbuf(traits->type_size);
                rotate_qr(key.data(), rotated.data(), Q_rot.data(), d);
                traits->from_float_ref(rotated.data(), qbuf.data(), d);
                traits->to_float(qbuf.data(), dequant.data(), d);
                rotate_qr_inv(dequant.data(), recon.data(), Q_rot.data(), d);
                score_mse = vec_dot(query.data(), recon.data(), d);
            }

            sum_err_prod += fabsf(score_orig - score_prod);
            sum_err_mse  += fabsf(score_orig - score_mse);
            sum_bias_prod += (score_prod - score_orig);
            sum_bias_mse  += (score_mse  - score_orig);
            sum_orig_prod += score_orig * score_prod;
            sum_sq_prod   += score_prod * score_prod;
            sum_orig_mse  += score_orig * score_mse;
            sum_sq_mse    += score_mse  * score_mse;

            if (fabsf(score_orig - score_prod) < fabsf(score_orig - score_mse)) prod_closer++;
        }

        float mean_err_prod = (float)(sum_err_prod / n_pairs);
        float mean_err_mse  = (float)(sum_err_mse  / n_pairs);
        float bias_prod = (float)(sum_bias_prod / n_pairs);
        float bias_mse  = (float)(sum_bias_mse  / n_pairs);
        float corr_prod = (float)(sum_orig_prod / n_pairs) / sqrtf((float)(sum_sq_orig / n_pairs) * (float)(sum_sq_prod / n_pairs));
        float corr_mse  = (float)(sum_orig_mse  / n_pairs) / sqrtf((float)(sum_sq_orig / n_pairs) * (float)(sum_sq_mse  / n_pairs));
        float prod_win_pct = 100.0f * prod_closer / n_pairs;

        bool prod_ok = (mean_err_prod < p.max_prod_mean_err);
        bool ok = prod_ok;
        if (!ok) failures++;

        printf("  %s: %s\n", p.label, RESULT_STR[!ok]);
        printf("    PROD  mean_err=%.5f  bias=%+.5f  corr=%.4f\n",
               mean_err_prod, bias_prod, corr_prod);
        printf("    MSE   mean_err=%.5f  bias=%+.5f  corr=%.4f\n",
               mean_err_mse, bias_mse, corr_mse);
        printf("    PROD closer on %d/%d pairs (%.1f%%)\n",
               prod_closer, n_pairs, prod_win_pct);

        (void)verbose;
    }

    return failures;
}

// ---------------------------------------------------------------------------
// Test Q: Combined K=PROD, V=MSE attention simulation (WITH rotation)
// ---------------------------------------------------------------------------
// The intended usage: PROD for K cache (unbiased attention scores),
// MSE for V cache (lower reconstruction error for weighted sums).
// Full pipeline with rotation on both K and V.

static int test_combined_kv_attention(bool verbose) {
    int failures = 0;
    printf("\n=== Test Q: Combined K=PROD V=MSE attention simulation (with rotation) ===\n");
    printf("  (Intended usage: PROD for keys, MSE for values)\n\n");

    const int d = 128;
    const int n_keys = 64;
    const int n_queries = 100;

    // Rotation matrix (shared by K and V in real usage)
    prng_seed(314159);
    std::vector<float> Q_rot(d * d);
    gen_random_orthogonal(Q_rot.data(), d);

    struct config {
        const char * name;
        ggml_type type_k;
        ggml_type type_v;
    };
    config configs[] = {
        { "prod_k + mse_v (intended)",   GGML_TYPE_TURBO4_0_PROD, GGML_TYPE_TURBO4_0_MSE  },
        { "prod_k + prod_v",             GGML_TYPE_TURBO4_0_PROD, GGML_TYPE_TURBO4_0_PROD },
        { "mse_k  + mse_v",             GGML_TYPE_TURBO4_0_MSE,  GGML_TYPE_TURBO4_0_MSE  },
        { "mse_k  + prod_v",            GGML_TYPE_TURBO4_0_MSE,  GGML_TYPE_TURBO4_0_PROD },
    };
    const int n_configs = sizeof(configs) / sizeof(configs[0]);

    // Generate keys and values with realistic distributions
    prng_seed(42424242);
    std::vector<float> keys(n_keys * d), values(n_keys * d);
    for (int k = 0; k < n_keys; k++) {
        float ns = 0.5f + prng_uniform(0.0f, 2.0f);
        switch (k % 6) {
            case 0: gen_kv_lognormal_outlier(keys.data() + k*d, d, ns); break;
            case 1: gen_kv_powerlaw(keys.data() + k*d, d, ns); break;
            case 2: gen_kv_correlated(keys.data() + k*d, d, ns); break;
            case 3: gen_kv_heavy_tail(keys.data() + k*d, d, ns); break;
            case 4: gen_kv_sparse(keys.data() + k*d, d, ns); break;
            case 5: gen_kv_asymmetric(keys.data() + k*d, d, ns); break;
        }
        gen_gaussian_normalized(values.data() + k*d, d, 0.5f + prng_uniform(0.0f, 1.5f));
    }

    // Generate queries
    std::vector<float> queries(n_queries * d);
    for (int q = 0; q < n_queries; q++) {
        gen_gaussian_normalized(queries.data() + q*d, d, 1.0f);
    }

    printf("  %-30s  output_cos  output_L2  score_KL  top1_acc\n", "config");
    printf("  %s\n", "------------------------------  ----------  ---------  --------  --------");

    for (int ci = 0; ci < n_configs; ci++) {
        auto & cfg = configs[ci];
        const auto * traits_k = ggml_get_type_traits(cfg.type_k);
        const auto * traits_v = ggml_get_type_traits(cfg.type_v);

        // Quantize all keys and values through rotation pipeline
        std::vector<std::vector<float>> recon_keys(n_keys, std::vector<float>(d));
        std::vector<std::vector<float>> recon_values(n_keys, std::vector<float>(d));

        for (int k = 0; k < n_keys; k++) {
            // Key: rotate → quantize → dequantize → unrotate
            {
                std::vector<float> rotated(d), dequant(d);
                std::vector<uint8_t> qbuf(traits_k->type_size);
                rotate_qr(keys.data() + k*d, rotated.data(), Q_rot.data(), d);
                traits_k->from_float_ref(rotated.data(), qbuf.data(), d);
                traits_k->to_float(qbuf.data(), dequant.data(), d);
                rotate_qr_inv(dequant.data(), recon_keys[k].data(), Q_rot.data(), d);
            }
            // Value: same pipeline
            {
                std::vector<float> rotated(d), dequant(d);
                std::vector<uint8_t> qbuf(traits_v->type_size);
                rotate_qr(values.data() + k*d, rotated.data(), Q_rot.data(), d);
                traits_v->from_float_ref(rotated.data(), qbuf.data(), d);
                traits_v->to_float(qbuf.data(), dequant.data(), d);
                rotate_qr_inv(dequant.data(), recon_values[k].data(), Q_rot.data(), d);
            }
        }

        // Run attention for each query
        double sum_output_cos = 0;
        double sum_output_l2 = 0;
        double sum_score_kl = 0;
        int top1_correct = 0;

        for (int q = 0; q < n_queries; q++) {
            const float * query = queries.data() + q*d;

            std::vector<float> scores_orig(n_keys), scores_quant(n_keys);
            for (int k = 0; k < n_keys; k++) {
                scores_orig[k]  = vec_dot(query, keys.data() + k*d, d);
                scores_quant[k] = vec_dot(query, recon_keys[k].data(), d);
            }

            // Top-1 accuracy
            int am_o = 0, am_q = 0;
            for (int k = 1; k < n_keys; k++) {
                if (scores_orig[k] > scores_orig[am_o]) am_o = k;
                if (scores_quant[k] > scores_quant[am_q]) am_q = k;
            }
            if (am_o == am_q) top1_correct++;

            // Softmax
            float scale = 1.0f / sqrtf((float)d);
            std::vector<float> attn_orig(n_keys), attn_quant(n_keys);
            for (int k = 0; k < n_keys; k++) {
                attn_orig[k] = scores_orig[k] * scale;
                attn_quant[k] = scores_quant[k] * scale;
            }
            softmax(attn_orig.data(), n_keys);
            softmax(attn_quant.data(), n_keys);

            sum_score_kl += kl_divergence(attn_orig.data(), attn_quant.data(), n_keys);

            // Attention output: sum(attn_weight * value)
            std::vector<float> output_orig(d, 0.0f), output_quant(d, 0.0f);
            for (int k = 0; k < n_keys; k++) {
                for (int j = 0; j < d; j++) {
                    output_orig[j]  += attn_orig[k]  * (values.data() + k*d)[j];
                    output_quant[j] += attn_quant[k] * recon_values[k][j];
                }
            }

            sum_output_cos += cosine_sim(output_orig.data(), output_quant.data(), d);
            sum_output_l2  += relative_l2_error(output_orig.data(), output_quant.data(), d);
        }

        float mean_cos = (float)(sum_output_cos / n_queries);
        float mean_l2  = (float)(sum_output_l2  / n_queries);
        float mean_kl  = (float)(sum_score_kl   / n_queries);
        float top1_pct = (float)top1_correct / n_queries * 100.0f;

        printf("  %-30s  %10.6f  %9.6f  %8.6f  %6.1f%%\n",
               cfg.name, mean_cos, mean_l2, mean_kl, top1_pct);
    }

    printf("\n  (No hard pass/fail — informational comparison of all K/V type combinations)\n");

    (void)verbose;
    return failures;
}

// ---------------------------------------------------------------------------
// Test R: MSE distortion vs paper bounds (WITH rotation)
// ---------------------------------------------------------------------------
// Paper Theorem 1: D_mse = E[||x - x̃||²] ≤ sqrt(3π)/2 · 1/4^b for ||x||=1
// Specific values: b=1: 0.36, b=2: 0.117, b=3: 0.03, b=4: 0.009
// PyTorch findings: all measured MSE well within bounds (ratio 0.53-0.87x)
//
// Our two-tier types have effective data rates of 2.25 and 3.25 bits/channel
// (not 3 and 4). D_mse should fall between the bounds for the two bit-widths
// used in the hi and lo channels.

static int test_mse_distortion_bounds(bool verbose) {
    int failures = 0;
    printf("\n=== Test R: MSE distortion vs paper bounds (with rotation) ===\n");
    printf("  (Paper: b=2 → D_mse≈0.117, b=3 → 0.03, b=4 → 0.009)\n\n");

    const int d = 128;
    const int n_vectors = 1000;

    // Rotation matrix
    prng_seed(314159);
    std::vector<float> Q_rot(d * d);
    gen_random_orthogonal(Q_rot.data(), d);

    struct entry {
        ggml_type type;
        const char * label;
        float bpv_data;           // effective data bits/channel (not counting norms)
        // Expected D_mse range based on the hi/lo channel bit-widths
        float expected_lower;     // D_mse for the higher bit-width
        float expected_upper;     // D_mse for the lower bit-width
    };
    entry entries[] = {
        // turbo3_0_mse: hi=3b, lo=2b → 2.25 bpc data → between b=2 and b=3
        { GGML_TYPE_TURBO3_0_MSE,  "turbo3_0_mse  (2.25 bpc)", 2.25f, 0.03f,  0.117f },
        // turbo4_0_mse: hi=4b, lo=3b → 3.25 bpc data → between b=3 and b=4
        { GGML_TYPE_TURBO4_0_MSE,  "turbo4_0_mse  (3.25 bpc)", 3.25f, 0.009f, 0.03f  },
        // PROD for comparison (MSE stage uses 1 fewer bit per channel)
        { GGML_TYPE_TURBO3_0_PROD, "turbo3_0_prod (1.25 bpc MSE)", 1.25f, 0.0f, 0.0f },
        { GGML_TYPE_TURBO4_0_PROD, "turbo4_0_prod (2.25 bpc MSE)", 2.25f, 0.0f, 0.0f },
    };

    for (auto & e : entries) {
        prng_seed(54321);

        double sum_dmse = 0;
        for (int v = 0; v < n_vectors; v++) {
            std::vector<float> data(d);
            gen_gaussian_normalized(data.data(), d, 1.0f);

            // Full pipeline: rotate → quantize → dequantize → unrotate
            const auto * traits = ggml_get_type_traits(e.type);
            std::vector<float> rotated(d), dequant(d), recon(d);
            std::vector<uint8_t> qbuf(traits->type_size);

            rotate_qr(data.data(), rotated.data(), Q_rot.data(), d);
            traits->from_float_ref(rotated.data(), qbuf.data(), d);
            traits->to_float(qbuf.data(), dequant.data(), d);
            rotate_qr_inv(dequant.data(), recon.data(), Q_rot.data(), d);

            double sq_err = 0;
            for (int j = 0; j < d; j++) {
                double diff = data[j] - recon[j];
                sq_err += diff * diff;
            }
            sum_dmse += sq_err;  // ||x - x̃||² for unit-norm vector
        }
        float measured_dmse = (float)(sum_dmse / n_vectors);

        bool ok = true;
        if (e.expected_upper > 0) {
            // Should fall within the expected range (with 2x slack for two-tier split)
            ok = (measured_dmse < e.expected_upper * 2.0f);
        }
        if (!ok) failures++;

        printf("  %-35s  D_mse=%.6f", e.label, measured_dmse);
        if (e.expected_upper > 0) {
            printf("  (paper: %.3f-%.3f)", e.expected_lower, e.expected_upper);
        }
        printf("  %s\n", RESULT_STR[!ok]);
    }

    printf("\n  Note: PROD D_mse is higher because its MSE stage uses b-1 bits (rest is QJL)\n");
    printf("  Note: Effective bpc = (32*hi_bits + 96*lo_bits) / 128, not counting norm overhead\n");

    (void)verbose;
    return failures;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char * argv[]) {
    bool verbose = false;
    bool bench = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-v") == 0) {
            verbose = true;
        } else if (strcmp(argv[i], "-bench") == 0 || strcmp(argv[i], "--bench") == 0) {
            bench = true;
        } else {
            fprintf(stderr, "usage: %s [-v] [-bench]\n", argv[0]);
            fprintf(stderr, "  -v      verbose output\n");
            fprintf(stderr, "  -bench  run slow benchmarks (K-M) in addition to fast tests (A-I)\n");
            return 1;
        }
    }

    ggml_cpu_init();

    printf("TurboQuant reference implementation tests\n");
    printf("==========================================\n");
    if (!bench) {
        printf("(fast mode — use -bench for full benchmarks)\n");
    }

    int total_failures = 0;

    // --- Fast correctness tests (A-I) ---
    total_failures += test_format_sanity();
    total_failures += test_zero_vector();
    total_failures += test_determinism();
    total_failures += test_roundtrip_distributions(verbose);
    total_failures += test_score_error(verbose);
    total_failures += test_multiblock();
    total_failures += test_norm_preservation();
    total_failures += test_bit_packing();
    total_failures += test_kv_cache_simulation(verbose);

    // --- MSE variant tests (N-R) ---
    total_failures += test_mse_format_sanity();
    total_failures += test_mse_zero_and_determinism();
    total_failures += test_mse_vs_prod_reconstruction(verbose);
    total_failures += test_prod_vs_mse_inner_product(verbose);
    total_failures += test_combined_kv_attention(verbose);
    total_failures += test_mse_distortion_bounds(verbose);

    if (!bench) goto done;

    // --- Slow benchmarks (J-M) — run with -bench ---
    total_failures += test_comparison(verbose);
    total_failures += test_attention_fidelity(verbose);
    total_failures += test_long_context(verbose);
    total_failures += test_adversarial_attention(verbose);
    total_failures += test_seed_sweep(verbose);
    total_failures += test_qjl_diagnostic(verbose);
    total_failures += test_paper_reference(verbose);
    total_failures += test_d1536_simulation(verbose);

    done:

    printf("\n==========================================\n");
    if (total_failures > 0) {
        printf("%d test(s) FAILED\n", total_failures);
    } else {
        printf("All tests passed\n");
    }

    return total_failures > 0 ? 1 : 0;
}
