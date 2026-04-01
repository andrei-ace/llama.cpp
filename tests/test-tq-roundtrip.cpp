// TurboQuant split-type roundtrip tests
// Tests: structured rotation roundtrip, quantize/dequant roundtrip,
//        vec_dot consistency, quantize determinism

#include "ggml.h"
#include "ggml-cpu.h"

#undef NDEBUG
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

// Forward declarations for TQ outlier mask init (from ggml-turbo-quant.h)
extern "C" {
    void tq_init_outlier_masks(int n_layers, int n_heads, int head_dim);
    void tq_set_current_layer(int layer, int is_k);
    void tq_set_current_head(int head);
    void tq_free_outlier_masks(void);
}

// ---------------------------------------------------------------------------
// Replicate the structured rotation logic (these are static in ggml-turbo-quant.c)
// ---------------------------------------------------------------------------

static void test_fwht(float * x, int n) {
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

static const float test_O3[9] = {
    0.5773502691896257f,  0.5773502691896257f,  0.5773502691896257f,
    0.7071067811865476f,  0.0f,                -0.7071067811865476f,
    0.4082482904638631f, -0.8164965809277261f,  0.4082482904638631f,
};

// O_3 transpose (inverse of the orthogonal O_3 matrix)
static const float test_O3_T[9] = {
    0.5773502691896257f,  0.7071067811865476f,  0.4082482904638631f,
    0.5773502691896257f,  0.0f,                -0.8164965809277261f,
    0.5773502691896257f, -0.7071067811865476f,  0.4082482904638631f,
};

static float test_struct_sign(int i, int n) {
    uint64_t seed = (n == 96) ? 0x5452534C4F393600ULL : 0x5452534C31393200ULL;
    for (int k = 0; k <= i; k++) {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
    }
    return ((uint32_t)(seed >> 32) & 1) ? -1.0f : 1.0f;
}

static void test_structured_rotate_lo(float * x, int n) {
    int block_dim = n / 3;
    // Step 1: diagonal sign flip
    for (int i = 0; i < n; i++) x[i] *= test_struct_sign(i, n);
    // Step 2: O_3 cross-block mix
    for (int j = 0; j < block_dim; j++) {
        float a = x[j], b = x[block_dim + j], c = x[2*block_dim + j];
        x[j]              = test_O3[0]*a + test_O3[1]*b + test_O3[2]*c;
        x[block_dim + j]  = test_O3[3]*a + test_O3[4]*b + test_O3[5]*c;
        x[2*block_dim + j]= test_O3[6]*a + test_O3[7]*b + test_O3[8]*c;
    }
    // Step 3: fixed permutation
    std::vector<float> tmp(n);
    for (int i = 0; i < n; i++) tmp[(35*i + 17) % n] = x[i];
    for (int i = 0; i < n; i++) x[i] = tmp[i];
    // Step 4: 3x FWHT
    test_fwht(x, block_dim);
    test_fwht(x + block_dim, block_dim);
    test_fwht(x + 2*block_dim, block_dim);
}

static void test_structured_unrotate_lo(float * x, int n) {
    int block_dim = n / 3;
    // Step 1: 3x FWHT (inverse = forward for normalized Hadamard)
    test_fwht(x, block_dim);
    test_fwht(x + block_dim, block_dim);
    test_fwht(x + 2*block_dim, block_dim);
    // Step 2: inverse permutation
    std::vector<float> tmp(n);
    for (int i = 0; i < n; i++) tmp[i] = x[(35*i + 17) % n];
    for (int i = 0; i < n; i++) x[i] = tmp[i];
    // Step 3: O_3^T (inverse) cross-block mix
    for (int j = 0; j < block_dim; j++) {
        float a = x[j], b = x[block_dim + j], c = x[2*block_dim + j];
        x[j]              = test_O3_T[0]*a + test_O3_T[1]*b + test_O3_T[2]*c;
        x[block_dim + j]  = test_O3_T[3]*a + test_O3_T[4]*b + test_O3_T[5]*c;
        x[2*block_dim + j]= test_O3_T[6]*a + test_O3_T[7]*b + test_O3_T[8]*c;
    }
    // Step 4: diagonal sign flip (signs are self-inverse)
    for (int i = 0; i < n; i++) x[i] *= test_struct_sign(i, n);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static float vec_norm(const float * x, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += (double)x[i] * x[i];
    return (float)sqrt(sum);
}

static float vec_dot_f(const float * a, const float * b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += (double)a[i] * b[i];
    return (float)sum;
}

static float vec_l2_error(const float * a, const float * b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        double d = a[i] - b[i];
        sum += d * d;
    }
    return (float)sqrt(sum);
}

// Simple LCG-based random float in [-1, 1]
static uint32_t test_rng_state;

static void test_rng_seed(uint32_t seed) {
    test_rng_state = seed;
}

static float test_rng_float(void) {
    test_rng_state = test_rng_state * 1664525u + 1013904223u;
    return (float)(int32_t)test_rng_state / (float)0x7FFFFFFF;
}

static void generate_random(float * dst, int n) {
    for (int i = 0; i < n; i++) {
        dst[i] = test_rng_float();
    }
}

// ---------------------------------------------------------------------------
// TQ split types to test (d=128)
// ---------------------------------------------------------------------------

struct tq_test_type {
    enum ggml_type type;
    const char *   name;
};

static const tq_test_type tq_split_types[] = {
    { GGML_TYPE_TQK_5HI_3LO_HAD,  "tqk3_sj"  },
    { GGML_TYPE_TQK_6HI_3LO_HAD,  "tqk4_sj"  },
    { GGML_TYPE_TQK_3HI_2LO_HAD,  "tqk3_sjj" },
    { GGML_TYPE_TQK_2HI_1LO_HAD,  "tqk2_sj"  },
    { GGML_TYPE_TQK_6HI_3LO_HAD_JJ, "tqk4_sjj" },
};

static const int n_tq_split_types = (int)(sizeof(tq_split_types) / sizeof(tq_split_types[0]));

// ---------------------------------------------------------------------------
// Test 1: Structured rotation roundtrip
// ---------------------------------------------------------------------------

static int test_rotation_roundtrip(void) {
    printf("\n=== Test 1: Structured rotation roundtrip ===\n");
    int failures = 0;

    const int dims[] = { 96, 192 };
    for (int d = 0; d < 2; d++) {
        int n = dims[d];
        printf("  dim=%d:\n", n);

        // Generate two random vectors
        test_rng_seed(42 + d);
        std::vector<float> x(n), y(n), rx(n), ry(n);
        generate_random(x.data(), n);
        generate_random(y.data(), n);

        // Save originals
        std::vector<float> x_orig(x.begin(), x.end());
        std::vector<float> y_orig(y.begin(), y.end());

        // Forward rotate
        memcpy(rx.data(), x.data(), n * sizeof(float));
        memcpy(ry.data(), y.data(), n * sizeof(float));
        test_structured_rotate_lo(rx.data(), n);
        test_structured_rotate_lo(ry.data(), n);

        // --- Norm preservation: ||x|| == ||R(x)|| ---
        float norm_x    = vec_norm(x.data(), n);
        float norm_rx   = vec_norm(rx.data(), n);
        float norm_err  = fabsf(norm_x - norm_rx) / (norm_x + 1e-10f);
        bool norm_ok    = norm_err < 1e-5f;
        printf("    norm preservation: %s (rel err = %e)\n",
               norm_ok ? "PASS" : "FAIL", norm_err);
        if (!norm_ok) failures++;

        // --- Dot product preservation: x.y == R(x).R(y) ---
        float dp_orig = vec_dot_f(x.data(), y.data(), n);
        float dp_rot  = vec_dot_f(rx.data(), ry.data(), n);
        float dp_err  = fabsf(dp_orig - dp_rot) / (fabsf(dp_orig) + 1e-10f);
        bool dp_ok    = dp_err < 1e-5f;
        printf("    dot product preservation: %s (rel err = %e)\n",
               dp_ok ? "PASS" : "FAIL", dp_err);
        if (!dp_ok) failures++;

        // --- Roundtrip: rotate then unrotate ---
        test_structured_unrotate_lo(rx.data(), n);
        float l2_err = vec_l2_error(x_orig.data(), rx.data(), n);
        bool rt_ok   = l2_err < 1e-5f;
        printf("    roundtrip (rotate->unrotate): %s (L2 err = %e)\n",
               rt_ok ? "PASS" : "FAIL", l2_err);
        if (!rt_ok) failures++;
    }
    return failures;
}

// ---------------------------------------------------------------------------
// Test 2: Quantize -> Dequantize roundtrip
// ---------------------------------------------------------------------------

static int test_quant_dequant_roundtrip(void) {
    printf("\n=== Test 2: Quantize -> Dequantize roundtrip ===\n");
    int failures = 0;

    const int n_blocks = 4;  // test with several blocks

    for (int t = 0; t < n_tq_split_types; t++) {
        ggml_type type = tq_split_types[t].type;
        const char * name = tq_split_types[t].name;

        const auto * qfns     = ggml_get_type_traits(type);
        const auto * qfns_cpu = ggml_get_type_traits_cpu(type);

        if (!qfns_cpu->from_float || !qfns->to_float) {
            printf("  %s: SKIP (no from_float or to_float)\n", name);
            continue;
        }

        const int block_size = (int)qfns->blck_size;
        const int n_elems    = block_size * n_blocks;

        ggml_quantize_init(type);

        test_rng_seed(12345 + t);
        std::vector<float> orig(n_elems);
        generate_random(orig.data(), n_elems);

        // Allocate quantized buffer
        size_t qbuf_size = qfns->type_size * n_blocks;
        std::vector<uint8_t> qbuf(qbuf_size + 256, 0);
        std::vector<float> recon(n_elems);

        // Quantize
        qfns_cpu->from_float(orig.data(), qbuf.data(), n_elems);

        // Dequantize
        qfns->to_float(qbuf.data(), recon.data(), n_elems);

        // Compute relative L2 error
        float err   = vec_l2_error(orig.data(), recon.data(), n_elems);
        float norm  = vec_norm(orig.data(), n_elems);
        float rel   = err / (norm + 1e-10f);
        bool ok     = rel < 0.5f;  // 50% threshold for aggressive quantizers
        printf("  %s: %s (rel L2 = %.4f, abs L2 = %.4f)\n",
               name, ok ? "PASS" : "FAIL", rel, err);
        if (!ok) failures++;
    }
    return failures;
}

// ---------------------------------------------------------------------------
// Test 3: vec_dot consistency
// ---------------------------------------------------------------------------

static int test_vec_dot_consistency(void) {
    printf("\n=== Test 3: vec_dot consistency ===\n");
    int failures = 0;

    const int n_blocks = 4;

    for (int t = 0; t < n_tq_split_types; t++) {
        ggml_type type = tq_split_types[t].type;
        const char * name = tq_split_types[t].name;

        const auto * qfns     = ggml_get_type_traits(type);
        const auto * qfns_cpu = ggml_get_type_traits_cpu(type);

        if (!qfns_cpu->from_float || !qfns->to_float || !qfns_cpu->vec_dot) {
            printf("  %s: SKIP (missing from_float, to_float, or vec_dot)\n", name);
            continue;
        }

        const int block_size = (int)qfns->blck_size;
        const int n_elems    = block_size * n_blocks;

        ggml_quantize_init(type);

        // Generate K and Q vectors
        test_rng_seed(99999 + t);
        std::vector<float> k_float(n_elems), q_float(n_elems);
        generate_random(k_float.data(), n_elems);
        generate_random(q_float.data(), n_elems);

        // Quantize K
        size_t qbuf_size = qfns->type_size * n_blocks;
        std::vector<uint8_t> k_quant(qbuf_size + 256, 0);
        qfns_cpu->from_float(k_float.data(), k_quant.data(), n_elems);

        // Q needs to be in vec_dot_type format
        enum ggml_type vdot_type = qfns_cpu->vec_dot_type;

        // For TQ split types, vec_dot_type is F32, so Q stays as float
        std::vector<uint8_t> q_quant;
        if (vdot_type == GGML_TYPE_F32) {
            q_quant.resize(n_elems * sizeof(float));
            memcpy(q_quant.data(), q_float.data(), n_elems * sizeof(float));
        } else {
            const auto * vdot_traits = ggml_get_type_traits_cpu(vdot_type);
            const auto * q_qfns = ggml_get_type_traits(vdot_type);
            size_t q_qbuf_size = q_qfns->type_size * (n_elems / q_qfns->blck_size);
            q_quant.resize(q_qbuf_size + 256, 0);
            vdot_traits->from_float(q_float.data(), q_quant.data(), n_elems);
        }

        // Method A: vec_dot(quantized_K, Q)
        float dot_a = 0.0f;
        qfns_cpu->vec_dot(n_elems, &dot_a, 0,
                          k_quant.data(), 0,
                          q_quant.data(), 0, 1);

        // Method B: dequantize K, manual dot product
        std::vector<float> k_dequant(n_elems);
        qfns->to_float(k_quant.data(), k_dequant.data(), n_elems);
        float dot_b = vec_dot_f(k_dequant.data(), q_float.data(), n_elems);

        // Compare: TQ vec_dot includes QJL correction terms computed in
        // the rotated domain, while dequant reconstructs and then we dot
        // in the original domain. They won't match exactly but should be
        // in the same ballpark.
        float diff = fabsf(dot_a - dot_b);
        float scale = (fabsf(dot_a) + fabsf(dot_b)) / 2.0f + 1e-10f;
        float rel_err = diff / scale;
        bool ok = rel_err < 0.5f;
        printf("  %s: %s (dot_a=%.4f, dot_b=%.4f, rel_diff=%.4f)\n",
               name, ok ? "PASS" : "FAIL", dot_a, dot_b, rel_err);
        if (!ok) failures++;
    }
    return failures;
}

// ---------------------------------------------------------------------------
// Test 4: CPU quantize determinism
// ---------------------------------------------------------------------------

static int test_quantize_determinism(void) {
    printf("\n=== Test 4: Quantize determinism ===\n");
    int failures = 0;

    const int n_blocks = 4;

    for (int t = 0; t < n_tq_split_types; t++) {
        ggml_type type = tq_split_types[t].type;
        const char * name = tq_split_types[t].name;

        const auto * qfns     = ggml_get_type_traits(type);
        const auto * qfns_cpu = ggml_get_type_traits_cpu(type);

        if (!qfns_cpu->from_float) {
            printf("  %s: SKIP (no from_float)\n", name);
            continue;
        }

        const int block_size = (int)qfns->blck_size;
        const int n_elems    = block_size * n_blocks;

        ggml_quantize_init(type);

        test_rng_seed(77777 + t);
        std::vector<float> data(n_elems);
        generate_random(data.data(), n_elems);

        size_t qbuf_size = qfns->type_size * n_blocks;
        std::vector<uint8_t> q1(qbuf_size + 256, 0);
        std::vector<uint8_t> q2(qbuf_size + 256, 0);

        // Quantize twice
        qfns_cpu->from_float(data.data(), q1.data(), n_elems);
        qfns_cpu->from_float(data.data(), q2.data(), n_elems);

        // Compare byte-by-byte
        bool ok = (memcmp(q1.data(), q2.data(), qbuf_size) == 0);
        printf("  %s: %s\n", name, ok ? "PASS" : "FAIL");
        if (!ok) failures++;
    }
    return failures;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(void) {
    printf("TurboQuant split-type roundtrip tests\n");
    printf("======================================\n");

    // Initialize CPU backend
    ggml_cpu_init();

    // Initialize TQ outlier masks with default identity split (1 layer, 1 head, d=128)
    // This sets up channels 0..31 as outliers, 32..127 as regular
    tq_init_outlier_masks(1, 1, 128);
    tq_set_current_layer(0, 1);  // layer 0, K cache
    tq_set_current_head(0);

    int total_failures = 0;

    total_failures += test_rotation_roundtrip();
    total_failures += test_quant_dequant_roundtrip();
    total_failures += test_vec_dot_consistency();
    total_failures += test_quantize_determinism();

    tq_free_outlier_masks();

    printf("\n======================================\n");
    if (total_failures == 0) {
        printf("All tests PASSED\n");
    } else {
        printf("%d test(s) FAILED\n", total_failures);
    }

    return total_failures > 0 ? 1 : 0;
}
