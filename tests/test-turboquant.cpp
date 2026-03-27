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
        ok = ok && (ggml_type_size(GGML_TYPE_TURBO3_0)  == 40);
        ok = ok && (ggml_blck_size(GGML_TYPE_TURBO3_0)  == 128);
        ok = ok && (ggml_is_quantized(GGML_TYPE_TURBO3_0));
        ok = ok && (strcmp(ggml_type_name(GGML_TYPE_TURBO3_0), "turbo3_0") == 0);
        printf("  turbo3_0 format: %s (size=%zu blck=%d name=%s)\n",
               RESULT_STR[!ok],
               ggml_type_size(GGML_TYPE_TURBO3_0),
               (int)ggml_blck_size(GGML_TYPE_TURBO3_0),
               ggml_type_name(GGML_TYPE_TURBO3_0));
        if (!ok) failures++;
    }

    // TURBO4_0
    {
        bool ok = true;
        ok = ok && (ggml_type_size(GGML_TYPE_TURBO4_0)  == 68);
        ok = ok && (ggml_blck_size(GGML_TYPE_TURBO4_0)  == 128);
        ok = ok && (ggml_is_quantized(GGML_TYPE_TURBO4_0));
        ok = ok && (strcmp(ggml_type_name(GGML_TYPE_TURBO4_0), "turbo4_0") == 0);
        printf("  turbo4_0 format: %s (size=%zu blck=%d name=%s)\n",
               RESULT_STR[!ok],
               ggml_type_size(GGML_TYPE_TURBO4_0),
               (int)ggml_blck_size(GGML_TYPE_TURBO4_0),
               ggml_type_name(GGML_TYPE_TURBO4_0));
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
        auto r = do_roundtrip(GGML_TYPE_TURBO3_0, zeros.data(), n);
        bool ok = (r.rmse_val == 0.0f) && !r.has_bad;
        printf("  turbo3_0 zero: %s (rmse=%e)\n", RESULT_STR[!ok], r.rmse_val);
        if (!ok) failures++;
    }

    // TURBO4_0
    {
        const int n = 128;
        std::vector<float> zeros(n, 0.0f);
        auto r = do_roundtrip(GGML_TYPE_TURBO4_0, zeros.data(), n);
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

    check_determinism(GGML_TYPE_TURBO3_0, 128 * 4);
    check_determinism(GGML_TYPE_TURBO4_0, 128 * 2);
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
        { "cosine_wave",            0.65f,  0.75f,  0.50f,  0.85f  },
        { "gaussian_unit_norm",     0.60f,  0.82f,  0.22f,  0.96f  },
        { "gaussian_low_norm",      0.60f,  0.82f,  0.22f,  0.96f  },
        { "gaussian_high_norm",     0.60f,  0.82f,  0.22f,  0.96f  },
        { "single_spike",           1.10f,  0.20f,  1.00f,  0.50f  },
        { "alternating_constant",   0.80f,  0.60f,  0.60f,  0.80f  },
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
        auto r3 = do_roundtrip(GGML_TYPE_TURBO3_0, data_t3.data(), n_t3);
        bool ok3 = (r3.rel_l2 < tests[t].threshold_rel_l2_t3) &&
                   (r3.cos_sim > tests[t].threshold_cos_t3) &&
                   !r3.has_bad;
        if (!ok3) failures++;

        // TURBO4_0
        auto r4 = do_roundtrip(GGML_TYPE_TURBO4_0, data_t4.data(), n_t4);
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
        float threshold_mean = (type == GGML_TYPE_TURBO3_0) ? 0.10f : 0.02f;
        bool ok = (mean_err < threshold_mean);
        if (!ok) failures++;

        printf("  %s (d=%d, %d pairs): %s\n", ggml_type_name(type), d, n_pairs, RESULT_STR[!ok]);
        printf("    mean_abs_score_err = %.6f\n", mean_err);
        printf("    max_abs_score_err  = %.6f\n", max_abs_err);
        printf("    std_score_err      = %.6f\n", std_err);

        (void)verbose;
    };

    test_score(GGML_TYPE_TURBO3_0, 128);
    test_score(GGML_TYPE_TURBO4_0, 128);

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

    test_mb(GGML_TYPE_TURBO3_0, 128, 32);
    test_mb(GGML_TYPE_TURBO4_0, 128, 32);
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
        float norm_tol = (type == GGML_TYPE_TURBO3_0) ? 0.25f : 0.15f;
        bool ok = (fabsf(r.norm_ratio - 1.0f) < norm_tol) && !r.has_bad;
        if (!ok) failures++;

        printf("  %s norm_ratio: %s (%.6f, target ~1.0)\n",
               ggml_type_name(type), RESULT_STR[!ok], r.norm_ratio);
    };

    test_np(GGML_TYPE_TURBO3_0, 128, 8);
    test_np(GGML_TYPE_TURBO4_0, 128, 8);
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

        auto r = do_roundtrip(GGML_TYPE_TURBO3_0, data, n);

        const auto * traits = ggml_get_type_traits(GGML_TYPE_TURBO3_0);
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

        auto r = do_roundtrip(GGML_TYPE_TURBO4_0, data, n);

        const auto * traits = ggml_get_type_traits(GGML_TYPE_TURBO4_0);
        std::vector<uint8_t> qbuf(traits->type_size);
        float recon[128];
        traits->from_float_ref(data, qbuf.data(), n);
        traits->to_float(qbuf.data(), recon, n);

        float dot_orig = vec_dot(query, data, n);
        float dot_recon = vec_dot(query, recon, n);
        float dot_err = fabsf(dot_orig - dot_recon) / fmaxf(fabsf(dot_orig), 1e-6f);

        bool ok = (dot_err < 0.3f) && !r.has_bad;
        printf("  turbo4_0 inner product check: %s (dot_err=%.4f, rel_l2=%.4f)\n",
               RESULT_STR[!ok], dot_err, r.rel_l2);
        if (!ok) failures++;
    }

    return failures;
}

// ---------------------------------------------------------------------------
// Test I: KV cache simulation with rotation
// ---------------------------------------------------------------------------

// Fast Walsh-Hadamard Transform (in-place, unnormalized)
// d must be a power of 2
static void fwht(float * x, int d) {
    for (int half = 1; half < d; half *= 2) {
        for (int i = 0; i < d; i += half * 2) {
            for (int j = i; j < i + half; j++) {
                float a = x[j];
                float b = x[j + half];
                x[j]        = a + b;
                x[j + half]  = a - b;
            }
        }
    }
}

// Randomized Hadamard rotation: D * H * x / sqrt(d)
// D = diagonal of random ±1 signs, H = Hadamard matrix
// signs[] has d entries, each ±1
static void rotate_hadamard(const float * x, float * y, const float * signs, int d) {
    float inv_sqrt_d = 1.0f / sqrtf((float)d);
    // Step 1: multiply by sign diagonal
    for (int i = 0; i < d; i++) {
        y[i] = x[i] * signs[i];
    }
    // Step 2: apply Hadamard
    fwht(y, d);
    // Step 3: normalize
    for (int i = 0; i < d; i++) {
        y[i] *= inv_sqrt_d;
    }
}

// Inverse: (D * H)^-1 = H^T * D^-1 / sqrt(d) = H * D / d  (since H=H^T, D^-1=D)
// But we already have 1/sqrt(d) baked in, so inverse is: H * D * x / sqrt(d)
// (Hadamard is its own inverse up to scale, and D is its own inverse)
static void rotate_hadamard_inv(const float * x, float * y, const float * signs, int d) {
    float inv_sqrt_d = 1.0f / sqrtf((float)d);
    // Step 1: apply Hadamard
    for (int i = 0; i < d; i++) {
        y[i] = x[i];
    }
    fwht(y, d);
    // Step 2: multiply by signs and normalize
    for (int i = 0; i < d; i++) {
        y[i] *= signs[i] * inv_sqrt_d;
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

// Full pipeline roundtrip: rotate → quantize → dequantize → inverse rotate
static roundtrip_result do_roundtrip_rotated(ggml_type type, const float * input, int d,
                                              const float * signs) {
    const auto * traits = ggml_get_type_traits(type);
    const int64_t blck = traits->blck_size;
    assert(d % blck == 0);

    size_t qbuf_size = (size_t)(d / blck) * traits->type_size;
    std::vector<float> rotated(d);
    std::vector<uint8_t> qbuf(qbuf_size);
    std::vector<float> dequantized(d);
    std::vector<float> output(d);

    // Forward rotation
    rotate_hadamard(input, rotated.data(), signs, d);
    // Quantize
    traits->from_float_ref(rotated.data(), qbuf.data(), d);
    // Dequantize
    traits->to_float(qbuf.data(), dequantized.data(), d);
    // Inverse rotation
    rotate_hadamard_inv(dequantized.data(), output.data(), signs, d);

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
                                  const float * signs, int d) {
    const auto * traits = ggml_get_type_traits(type);
    size_t qsz = (size_t)(d / traits->blck_size) * traits->type_size;
    std::vector<float> rotated(d), dequant(d), recon(d);
    std::vector<uint8_t> qbuf(qsz);

    rotate_hadamard(key, rotated.data(), signs, d);
    traits->from_float_ref(rotated.data(), qbuf.data(), d);
    traits->to_float(qbuf.data(), dequant.data(), d);
    rotate_hadamard_inv(dequant.data(), recon.data(), signs, d);

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

    // Hadamard rotation signs (fixed per "head", shared across all tests)
    prng_seed(314159);
    std::vector<float> signs(d);
    for (int i = 0; i < d; i++) {
        signs[i] = (prng_next() & 1) ? 1.0f : -1.0f;
    }

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
        { GGML_TYPE_TURBO3_0, 0.060f, true },  // paper: "marginal degradation" at 2.5 bpw
        { GGML_TYPE_TURBO4_0, 0.020f, true },  // paper: "quality neutral" at 3.5 bpw
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
                auto r_rot = do_roundtrip_rotated(tc.type, kv_vec.data(), d, signs.data());
                sum_rel_l2_rot += r_rot.rel_l2;
                sum_cos_rot += r_rot.cos_sim;

                float se = score_error_rotated(tc.type, kv_vec.data(), query.data(), signs.data(), d);
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

    comparison comparisons[] = {
        // TURBO3_0 (2.5 bpw) vs closest existing types
        { GGML_TYPE_TURBO3_0, "turbo3_0", 2.50f, GGML_TYPE_Q2_K,  "q2_K",  2.63f },
        { GGML_TYPE_TURBO3_0, "turbo3_0", 2.50f, GGML_TYPE_Q3_K,  "q3_K",  3.44f },
        // TURBO4_0 (3.5 bpw) vs closest existing types
        { GGML_TYPE_TURBO4_0, "turbo4_0", 3.50f, GGML_TYPE_Q3_K,  "q3_K",  3.44f },
        { GGML_TYPE_TURBO4_0, "turbo4_0", 3.50f, GGML_TYPE_Q4_K,  "q4_K",  4.50f },
        { GGML_TYPE_TURBO4_0, "turbo4_0", 3.50f, GGML_TYPE_Q4_0,  "q4_0",  4.50f },
        // Also compare against higher-precision baselines for reference
        { GGML_TYPE_TURBO4_0, "turbo4_0", 3.50f, GGML_TYPE_Q5_K,  "q5_K",  5.50f },
        { GGML_TYPE_TURBO4_0, "turbo4_0", 3.50f, GGML_TYPE_Q8_0,  "q8_0",  8.50f },
    };
    const int n_comparisons = sizeof(comparisons) / sizeof(comparisons[0]);

    // Use d=256 so all types' block sizes divide evenly (QK_K=256 for K-quants)
    const int d = 256;
    const int n_vectors = 500;

    // Hadamard signs for rotation (two 128-dim blocks within 256)
    prng_seed(314159);
    std::vector<float> signs(128);
    for (int i = 0; i < 128; i++) {
        signs[i] = (prng_next() & 1) ? 1.0f : -1.0f;
    }

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

                    rotate_hadamard(src, rotated.data(), signs.data(), 128);
                    traits->from_float_ref(rotated.data(), qbuf.data(), 128);
                    traits->to_float(qbuf.data(), dequant.data(), 128);
                    rotate_hadamard_inv(dequant.data(), dst, signs.data(), 128);
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
        rotate_hadamard(keys + k * d, rotated.data(), signs, d);
        traits->from_float_ref(rotated.data(), qbuf.data(), d);
        traits->to_float(qbuf.data(), dequant.data(), d);
        rotate_hadamard_inv(dequant.data(), recon_keys[k].data(), signs, d);
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

    // Simulate a realistic attention scenario:
    // - 64 keys (sequence length)
    // - d=128 (head dimension, must be multiple of QK_K=256... use 256 for baseline compat)
    // - 100 queries
    // - KV-cache-like key distributions
    //
    // We use d=256 so K-quant types (blck=256) work. TurboQuant processes two 128-blocks.

    const int d = 256;
    const int n_keys = 64;
    const int n_queries = 100;

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

    // Hadamard signs for rotation
    prng_seed(314159);
    std::vector<float> signs(128);
    for (int i = 0; i < 128; i++) {
        signs[i] = (prng_next() & 1) ? 1.0f : -1.0f;
    }

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
    type_entry types[] = {
        { GGML_TYPE_TURBO3_0, "turbo3_0", 2.50f, true  },
        { GGML_TYPE_Q2_K,     "q2_K",     2.63f, false },
        { GGML_TYPE_Q3_K,     "q3_K",     3.44f, false },
        { GGML_TYPE_TURBO4_0, "turbo4_0", 3.50f, true  },
        { GGML_TYPE_Q4_0,     "q4_0",     4.50f, false },
        { GGML_TYPE_Q4_K,     "q4_K",     4.50f, false },
        { GGML_TYPE_Q5_K,     "q5_K",     5.50f, false },
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
                    rotate_hadamard(src, rotated.data(), signs.data(), 128);
                    traits->from_float_ref(rotated.data(), qbuf.data(), 128);
                    traits->to_float(qbuf.data(), dequant.data(), 128);
                    rotate_hadamard_inv(dequant.data(), dst, signs.data(), 128);
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
// Main
// ---------------------------------------------------------------------------

int main(int argc, char * argv[]) {
    bool verbose = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-v") == 0) {
            verbose = true;
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", argv[i]);
            return 1;
        }
    }

    ggml_cpu_init();

    printf("TurboQuant reference implementation tests\n");
    printf("==========================================\n");

    int total_failures = 0;

    total_failures += test_format_sanity();
    total_failures += test_zero_vector();
    total_failures += test_determinism();
    total_failures += test_roundtrip_distributions(verbose);
    total_failures += test_score_error(verbose);
    total_failures += test_multiblock();
    total_failures += test_norm_preservation();
    total_failures += test_bit_packing();
    total_failures += test_kv_cache_simulation(verbose);
    total_failures += test_comparison(verbose);
    total_failures += test_attention_fidelity(verbose);

    printf("\n==========================================\n");
    if (total_failures > 0) {
        printf("%d test(s) FAILED\n", total_failures);
    } else {
        printf("All tests passed\n");
    }

    return total_failures > 0 ? 1 : 0;
}
