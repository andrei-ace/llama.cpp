// TurboQuant Metal correctness tests
//
// Tests Metal get_rows/set_rows against CPU reference for all 4 TQ types.
// Compares element-wise: max abs error, mean abs error, cosine similarity.
//
// Build: cmake -B build -DGGML_METAL=ON && cmake --build build -j14
// Run:   ./build/bin/test-turboquant-metal

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#undef NDEBUG
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static uint64_t prng_state = 0xDEADBEEF42ULL;

static void prng_seed(uint64_t seed) { prng_state = seed; }

static uint32_t prng_next(void) {
    prng_state = prng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (uint32_t)(prng_state >> 32);
}

static float prng_gaussian(void) {
    float u1 = ((float)prng_next() + 1.0f) / ((float)0xFFFFFFFF + 2.0f);
    float u2 = ((float)prng_next() + 1.0f) / ((float)0xFFFFFFFF + 2.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265358979323846f * u2);
}

static float cosine_sim(const float * a, const float * b, int n) {
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; i++) {
        dot += (double)a[i] * b[i];
        na  += (double)a[i] * a[i];
        nb  += (double)b[i] * b[i];
    }
    if (na < 1e-30 || nb < 1e-30) return 1.0f;
    return (float)(dot / sqrt(na * nb));
}

static float max_abs_err(const float * a, const float * b, int n) {
    float mx = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

static float mean_abs_err(const float * a, const float * b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += fabs((double)a[i] - (double)b[i]);
    }
    return (float)(sum / n);
}

// ---------------------------------------------------------------------------
// Test: TQV get_rows roundtrip (quantize CPU → dequantize Metal vs CPU)
// ---------------------------------------------------------------------------

static bool test_tqv_get_rows(ggml_type type, const char * label) {
    printf("  test_tqv_get_rows %-8s ... ", label);

    const int n_heads = 8;
    const int head_dim = 128;
    const int ne00 = n_heads * head_dim; // 1024 elements per row
    const int n_rows = 4; // test with multiple rows

    // Generate random float data
    prng_seed(0x12345 + (uint64_t)type);
    std::vector<float> input(ne00 * n_rows);
    for (int i = 0; i < ne00 * n_rows; i++) {
        input[i] = prng_gaussian() * 2.0f;
    }

    // Quantize on CPU
    const size_t block_size = ggml_blck_size(type);
    const size_t type_size  = ggml_type_size(type);
    const int n_blocks_per_row = ne00 / block_size;
    const size_t row_quant_size = n_blocks_per_row * type_size;

    std::vector<uint8_t> quantized(row_quant_size * n_rows);
    auto from_float = ggml_get_type_traits(type)->from_float_ref;
    assert(from_float);

    for (int r = 0; r < n_rows; r++) {
        from_float(input.data() + r * ne00, quantized.data() + r * row_quant_size, ne00);
    }

    // Dequantize on CPU (reference)
    std::vector<float> cpu_output(ne00 * n_rows);
    auto to_float = ggml_get_type_traits(type)->to_float;
    assert(to_float);

    for (int r = 0; r < n_rows; r++) {
        to_float(quantized.data() + r * row_quant_size, cpu_output.data() + r * ne00, ne00);
    }

    // Now test Metal path: create ggml graph with GET_ROWS using the Metal backend
    // We'll use the backend API to create tensors and run the computation

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, NULL);
    if (!backend) {
        printf("SKIP (no GPU backend)\n");
        return true;
    }

    // Create a compute graph
    struct ggml_init_params params = {
        /*.mem_size   =*/ 256 * 1024 * 1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);

    // src0: quantized KV cache data [ne00, n_rows]
    struct ggml_tensor * src0 = ggml_new_tensor_2d(ctx, type, ne00, n_rows);
    bool is_k = (type == GGML_TYPE_TURBO3_0_PROD || type == GGML_TYPE_TURBO4_0_PROD);
    ggml_set_name(src0, is_k ? "cache_k_l0" : "cache_v_l0");

    // src1: row indices
    struct ggml_tensor * src1 = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_rows);

    // get_rows operation
    struct ggml_tensor * result = ggml_get_rows(ctx, src0, src1);

    // Allocate buffers
    //
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        printf("FAIL (buffer alloc)\n");
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }
    //

    // Upload quantized data to src0
    ggml_backend_tensor_set(src0, quantized.data(), 0, row_quant_size * n_rows);

    // Upload row indices
    std::vector<int32_t> indices(n_rows);
    for (int i = 0; i < n_rows; i++) indices[i] = i;
    ggml_backend_tensor_set(src1, indices.data(), 0, n_rows * sizeof(int32_t));

    //
    // Build and run compute graph
    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);

    //
    bool ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    //
    if (!ok) {
        printf("FAIL (compute)\n");
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    // Read back result
    std::vector<float> metal_output(ne00 * n_rows);
    ggml_backend_tensor_get(result, metal_output.data(), 0, ne00 * n_rows * sizeof(float));

    // Compare
    float max_err = max_abs_err(cpu_output.data(), metal_output.data(), ne00 * n_rows);
    float mean_err = mean_abs_err(cpu_output.data(), metal_output.data(), ne00 * n_rows);
    float cos_sim = cosine_sim(cpu_output.data(), metal_output.data(), ne00 * n_rows);

    bool pass = max_err < 0.01f && cos_sim > 0.999f;

    printf("%s  max_err=%.6f  mean_err=%.6f  cos_sim=%.6f\n",
           pass ? "ok" : "FAIL", max_err, mean_err, cos_sim);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);

    return pass;
}

// ---------------------------------------------------------------------------
// Test: TQV set_rows roundtrip (quantize Metal → read back → compare with CPU)
// ---------------------------------------------------------------------------

static bool test_tqv_set_rows(ggml_type type, const char * label) {
    printf("  test_tqv_set_rows %-8s ... ", label);

    const int n_heads = 8;
    const int head_dim = 128;
    const int ne00 = n_heads * head_dim;
    const int n_rows = 4;

    prng_seed(0x67890 + (uint64_t)type);
    std::vector<float> input(ne00 * n_rows);
    for (int i = 0; i < ne00 * n_rows; i++) {
        input[i] = prng_gaussian() * 2.0f;
    }

    // CPU quantize + dequantize (reference)
    const size_t block_size = ggml_blck_size(type);
    const size_t type_size  = ggml_type_size(type);
    const int n_blocks_per_row = ne00 / block_size;
    const size_t row_quant_size = n_blocks_per_row * type_size;

    std::vector<uint8_t> cpu_quantized(row_quant_size * n_rows);
    auto from_float = ggml_get_type_traits(type)->from_float_ref;
    for (int r = 0; r < n_rows; r++) {
        from_float(input.data() + r * ne00, (void *)(cpu_quantized.data() + r * row_quant_size), ne00);
    }

    std::vector<float> cpu_output(ne00 * n_rows);
    auto to_float = ggml_get_type_traits(type)->to_float;
    for (int r = 0; r < n_rows; r++) {
        to_float((const void *)(cpu_quantized.data() + r * row_quant_size), cpu_output.data() + r * ne00, ne00);
    }

    // Metal: set_rows (quantize on GPU) then get_rows (dequantize on GPU)
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, NULL);
    if (!backend) {
        printf("SKIP (no GPU backend)\n");
        return true;
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ 256 * 1024 * 1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);

    // Input float tensor
    struct ggml_tensor * src0 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne00, n_rows);
    ggml_set_name(src0, "input");

    // Row indices
    struct ggml_tensor * src1 = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_rows);

    // set_rows: quantize float → TQ (args: dest=quantized, source=float, indices)
    struct ggml_tensor * dst_quant = ggml_new_tensor_2d(ctx, type, ne00, n_rows);
    ggml_set_name(dst_quant, "cache_v_l0");

    struct ggml_tensor * set_op = ggml_set_rows(ctx, dst_quant, src0, src1);

    // get_rows: dequantize TQ → float
    struct ggml_tensor * src1b = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_rows);
    struct ggml_tensor * result = ggml_get_rows(ctx, set_op, src1b);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        printf("FAIL (buffer alloc)\n");
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    ggml_backend_tensor_set(src0, input.data(), 0, ne00 * n_rows * sizeof(float));

    std::vector<int32_t> indices(n_rows);
    for (int i = 0; i < n_rows; i++) indices[i] = i;
    ggml_backend_tensor_set(src1,  indices.data(), 0, n_rows * sizeof(int32_t));
    ggml_backend_tensor_set(src1b, indices.data(), 0, n_rows * sizeof(int32_t));

    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);

    bool ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    if (!ok) {
        printf("FAIL (compute)\n");
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    std::vector<float> metal_output(ne00 * n_rows);
    ggml_backend_tensor_get(result, metal_output.data(), 0, ne00 * n_rows * sizeof(float));

    float max_err = max_abs_err(cpu_output.data(), metal_output.data(), ne00 * n_rows);
    float mean_err = mean_abs_err(cpu_output.data(), metal_output.data(), ne00 * n_rows);
    float cos_sim = cosine_sim(cpu_output.data(), metal_output.data(), ne00 * n_rows);

    bool pass = max_err < 0.05f && cos_sim > 0.99f;

    printf("%s  max_err=%.6f  mean_err=%.6f  cos_sim=%.6f\n",
           pass ? "ok" : "FAIL", max_err, mean_err, cos_sim);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);

    return pass;
}

// ---------------------------------------------------------------------------
// Benchmark: get_rows throughput for TQ types vs q4_0
// ---------------------------------------------------------------------------

static void bench_get_rows(ggml_type type, const char * label, int ctx_len) {
    const int n_heads = 8;
    const int head_dim = 128;
    const int ne00 = n_heads * head_dim;
    const int n_rows = ctx_len;

    // Generate and quantize data
    prng_seed(0xBE4C0 + (uint64_t)type);
    std::vector<float> input(ne00 * n_rows);
    for (int i = 0; i < ne00 * n_rows; i++) {
        input[i] = prng_gaussian() * 2.0f;
    }

    const size_t block_size = ggml_blck_size(type);
    const size_t type_size  = ggml_type_size(type);
    const int n_blocks_per_row = ne00 / (int)block_size;
    const size_t row_quant_size = n_blocks_per_row * type_size;
    const size_t total_quant_bytes = row_quant_size * n_rows;
    const float bpv = (float)(type_size * 8) / (float)block_size;

    std::vector<uint8_t> quantized(total_quant_bytes);
    auto from_float = ggml_get_type_traits(type)->from_float_ref;
    for (int r = 0; r < n_rows; r++) {
        from_float(input.data() + r * ne00, (void *)(quantized.data() + r * row_quant_size), ne00);
    }

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, NULL);
    if (!backend) { printf("  %-8s SKIP\n", label); return; }

    struct ggml_init_params params = { 256 * 1024 * 1024, NULL, true };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * src0 = ggml_new_tensor_2d(ctx, type, ne00, n_rows);
    bool is_k = (type == GGML_TYPE_TURBO3_0_PROD || type == GGML_TYPE_TURBO4_0_PROD);
    ggml_set_name(src0, is_k ? "cache_k_l0" : "cache_v_l0");

    struct ggml_tensor * src1 = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_rows);
    struct ggml_tensor * result = ggml_get_rows(ctx, src0, src1);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_tensor_set(src0, quantized.data(), 0, total_quant_bytes);

    std::vector<int32_t> indices(n_rows);
    for (int i = 0; i < n_rows; i++) indices[i] = i;
    ggml_backend_tensor_set(src1, indices.data(), 0, n_rows * sizeof(int32_t));

    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);

    // Warmup
    ggml_backend_graph_compute(backend, graph);

    // Benchmark
    const int n_iter = 20;
    double t0 = ggml_time_us();
    for (int i = 0; i < n_iter; i++) {
        ggml_backend_graph_compute(backend, graph);
    }
    double t1 = ggml_time_us();

    double ms_per_iter = (t1 - t0) / 1000.0 / n_iter;
    double gb_per_sec = (double)total_quant_bytes / (1024.0 * 1024.0 * 1024.0) / (ms_per_iter / 1000.0);

    printf("  %-8s  %5.1f bpv  %6.2f ms  %6.2f GB/s  (%.1f KB quantized)\n",
           label, bpv, ms_per_iter, gb_per_sec, total_quant_bytes / 1024.0);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

// ---------------------------------------------------------------------------
// Test: TQ Flash Attention (Metal TQ FA vs CPU dequant + naive attention)
// ---------------------------------------------------------------------------

static bool test_tq_flash_attn(int ctx_len, const char * label) {
    printf("  test_tq_flash_attn ctx=%-4d ... ", ctx_len);

    const int n_heads_q = 8;  // Q heads
    const int n_heads_kv = 8; // KV heads (no GQA for simplicity)
    const int head_dim = 128;
    const int n_tokens_q = 1; // single query (generation mode)

    const ggml_type k_type = GGML_TYPE_TURBO4_0_PROD; // tqk35
    const ggml_type v_type = GGML_TYPE_TURBO4_0_MSE;  // tqv35

    // Generate random Q, K, V data
    prng_seed(0xFA000 + (uint64_t)ctx_len);

    std::vector<float> q_data(n_tokens_q * n_heads_q * head_dim);
    for (size_t i = 0; i < q_data.size(); i++) q_data[i] = prng_gaussian() * 0.1f;

    // Generate float K/V data, then quantize on CPU
    // Layout: [kv_len][n_heads_kv][head_dim] for generation
    const int kv_len = ctx_len;
    const int kv_total = kv_len * n_heads_kv * head_dim;
    std::vector<float> k_float(kv_total);
    std::vector<float> v_float(kv_total);
    for (size_t i = 0; i < k_float.size(); i++) k_float[i] = prng_gaussian();
    for (size_t i = 0; i < v_float.size(); i++) v_float[i] = prng_gaussian();

    // Quantize K — shape [head_dim, kv_len, n_heads_kv]
    // Contiguous in memory: for each head h, kv_len rows of head_dim elements
    const size_t k_block_size = ggml_blck_size(k_type);
    const size_t k_type_size  = ggml_type_size(k_type);
    const size_t k_row_bytes = (head_dim / k_block_size) * k_type_size; // bytes per (head_dim) row
    const size_t k_total_bytes = k_row_bytes * kv_len * n_heads_kv;
    std::vector<uint8_t> k_quant(k_total_bytes);
    auto k_from_float = ggml_get_type_traits(k_type)->from_float_ref;
    // k_float layout: [kv_len][n_heads_kv][head_dim] → need [head_dim][kv_len][n_heads_kv]
    // For ggml tensor [head_dim, kv_len, n_heads_kv], memory is contiguous on head_dim (ne0)
    // So row = (h * kv_len + t), stride = head_dim
    for (int h = 0; h < n_heads_kv; h++) {
        for (int t = 0; t < kv_len; t++) {
            k_from_float(k_float.data() + (t * n_heads_kv + h) * head_dim,
                        (void *)(k_quant.data() + (h * kv_len + t) * k_row_bytes), head_dim);
        }
    }

    // Quantize V — same layout
    const size_t v_block_size = ggml_blck_size(v_type);
    const size_t v_type_size  = ggml_type_size(v_type);
    const size_t v_row_bytes = (head_dim / v_block_size) * v_type_size;
    const size_t v_total_bytes = v_row_bytes * kv_len * n_heads_kv;
    std::vector<uint8_t> v_quant(v_total_bytes);
    auto v_from_float = ggml_get_type_traits(v_type)->from_float_ref;
    for (int h = 0; h < n_heads_kv; h++) {
        for (int t = 0; t < kv_len; t++) {
            v_from_float(v_float.data() + (t * n_heads_kv + h) * head_dim,
                        (void *)(v_quant.data() + (h * kv_len + t) * v_row_bytes), head_dim);
        }
    }

    // CPU reference: dequantize K/V per-head, then naive softmax attention
    // k_deq/v_deq layout: [n_heads_kv][kv_len][head_dim]
    std::vector<float> k_deq(n_heads_kv * kv_len * head_dim);
    std::vector<float> v_deq(n_heads_kv * kv_len * head_dim);
    auto k_to_float = ggml_get_type_traits(k_type)->to_float;
    auto v_to_float = ggml_get_type_traits(v_type)->to_float;
    for (int h = 0; h < n_heads_kv; h++) {
        for (int t = 0; t < kv_len; t++) {
            k_to_float((const void *)(k_quant.data() + (h * kv_len + t) * k_row_bytes),
                       k_deq.data() + (h * kv_len + t) * head_dim, head_dim);
            v_to_float((const void *)(v_quant.data() + (h * kv_len + t) * v_row_bytes),
                       v_deq.data() + (h * kv_len + t) * head_dim, head_dim);
        }
    }

    float scale = 1.0f / sqrtf((float)head_dim);

    // CPU naive attention per head
    // Output: [n_heads_q, head_dim] (one query token)
    std::vector<float> cpu_output(n_heads_q * head_dim, 0.0f);
    for (int h = 0; h < n_heads_q; h++) {
        int kv_h = h; // no GQA
        std::vector<float> scores(kv_len);
        float max_score = -1e30f;
        for (int t = 0; t < kv_len; t++) {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += q_data[h * head_dim + d] * k_deq[(kv_h * kv_len + t) * head_dim + d];
            }
            scores[t] = dot * scale;
            if (scores[t] > max_score) max_score = scores[t];
        }
        float sum_exp = 0.0f;
        for (int t = 0; t < kv_len; t++) {
            scores[t] = expf(scores[t] - max_score);
            sum_exp += scores[t];
        }
        for (int t = 0; t < kv_len; t++) {
            float w = scores[t] / sum_exp;
            for (int d = 0; d < head_dim; d++) {
                cpu_output[h * head_dim + d] += w * v_deq[(kv_h * kv_len + t) * head_dim + d];
            }
        }
    }

    // Metal FA
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, NULL);
    if (!backend) { printf("SKIP\n"); return true; }

    struct ggml_init_params params = { 256 * 1024 * 1024, NULL, true };
    struct ggml_context * ctx = ggml_init(params);

    // Q: [head_dim, n_tokens_q, n_heads_q]
    struct ggml_tensor * tQ = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, n_tokens_q, n_heads_q);
    ggml_set_name(tQ, "Q");

    // K: [head_dim, kv_len, n_heads_kv]
    struct ggml_tensor * tK = ggml_new_tensor_3d(ctx, k_type, head_dim, kv_len, n_heads_kv);
    ggml_set_name(tK, "cache_k_l0");

    // V: [head_dim, kv_len, n_heads_kv]
    struct ggml_tensor * tV = ggml_new_tensor_3d(ctx, v_type, head_dim, kv_len, n_heads_kv);
    ggml_set_name(tV, "cache_v_l0");

    // Flash attention
    struct ggml_tensor * fa = ggml_flash_attn_ext(ctx, tQ, tK, tV, NULL, scale, 0.0f, 0.0f);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { printf("FAIL (alloc)\n"); ggml_free(ctx); ggml_backend_free(backend); return false; }

    ggml_backend_tensor_set(tQ, q_data.data(), 0, q_data.size() * sizeof(float));
    ggml_backend_tensor_set(tK, k_quant.data(), 0, k_total_bytes);
    ggml_backend_tensor_set(tV, v_quant.data(), 0, v_total_bytes);

    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, fa);

    bool ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    if (!ok) { printf("FAIL (compute)\n"); ggml_backend_buffer_free(buf); ggml_free(ctx); ggml_backend_free(backend); return false; }

    // Output shape: [head_dim, n_heads_q, n_tokens_q] → permuted from [n_embd_v, n_head, n_batch]
    const int out_size = head_dim * n_heads_q * n_tokens_q;
    std::vector<float> metal_output(out_size);
    ggml_backend_tensor_get(fa, metal_output.data(), 0, out_size * sizeof(float));

    float merr = max_abs_err(cpu_output.data(), metal_output.data(), out_size);
    float mae = mean_abs_err(cpu_output.data(), metal_output.data(), out_size);
    float cs = cosine_sim(cpu_output.data(), metal_output.data(), out_size);

    bool pass = cs > 0.95f; // FA accumulates errors, so be lenient

    printf("%s  max_err=%.6f  mean_err=%.6f  cos_sim=%.6f\n",
           pass ? "ok" : "FAIL", merr, mae, cs);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return pass;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char ** argv) {
    bool run_bench = false;
    int bench_ctx = 2048;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--bench") == 0) { run_bench = true; }
        if (strcmp(argv[i], "--ctx") == 0 && i + 1 < argc) { bench_ctx = atoi(argv[++i]); }
    }

    printf("TurboQuant Metal correctness tests\n");
    printf("==================================\n\n");

    int n_pass = 0, n_fail = 0, n_total = 0;

    auto run = [&](bool result) {
        n_total++;
        if (result) { n_pass++; } else { n_fail++; }
    };

    // TQV get_rows tests (quantize CPU → dequantize Metal vs CPU dequantize)
    printf("TQV get_rows (CPU quantize → Metal dequantize):\n");
    run(test_tqv_get_rows(GGML_TYPE_TURBO4_0_MSE, "tqv35"));
    run(test_tqv_get_rows(GGML_TYPE_TURBO3_0_MSE, "tqv25"));
    printf("\n");

    // TQV set_rows tests (Metal quantize+dequantize vs CPU quantize+dequantize)
    printf("TQV set_rows (Metal quantize+dequantize vs CPU):\n");
    run(test_tqv_set_rows(GGML_TYPE_TURBO3_0_MSE, "tqv25"));
    run(test_tqv_set_rows(GGML_TYPE_TURBO4_0_MSE, "tqv35"));
    printf("\n");

    // TQK get_rows tests (K cache with QJL correction)
    printf("TQK get_rows (CPU quantize → Metal dequantize):\n");
    run(test_tqv_get_rows(GGML_TYPE_TURBO4_0_PROD, "tqk35"));
    run(test_tqv_get_rows(GGML_TYPE_TURBO3_0_PROD, "tqk25"));
    printf("\n");

    // Flash attention tests
    printf("TQ Flash Attention (Metal TQ FA vs CPU naive attention):\n");
    run(test_tq_flash_attn(64,   "ctx=64"));
    run(test_tq_flash_attn(256,  "ctx=256"));
    run(test_tq_flash_attn(1024, "ctx=1024"));
    printf("\n");

    printf("==================================\n");
    printf("Results: %d/%d passed", n_pass, n_total);
    if (n_fail > 0) {
        printf(", %d FAILED", n_fail);
    }
    printf("\n");

    if (run_bench) {
        printf("\nBenchmark: get_rows throughput (ctx=%d tokens, 8 heads)\n", bench_ctx);
        printf("%-10s  %5s  %8s  %8s  %s\n", "Type", "bpv", "Time", "BW", "Data size");
        bench_get_rows(GGML_TYPE_F16,           "f16",   bench_ctx);
        bench_get_rows(GGML_TYPE_Q8_0,          "q8_0",  bench_ctx);
        bench_get_rows(GGML_TYPE_Q4_0,          "q4_0",  bench_ctx);
        bench_get_rows(GGML_TYPE_TURBO4_0_MSE,  "tqv35", bench_ctx);
        bench_get_rows(GGML_TYPE_TURBO3_0_MSE,  "tqv25", bench_ctx);
        bench_get_rows(GGML_TYPE_TURBO4_0_PROD, "tqk35", bench_ctx);
        bench_get_rows(GGML_TYPE_TURBO3_0_PROD, "tqk25", bench_ctx);
        printf("\n");

        // Flash attention benchmark: TQ vs standard types
        printf("Benchmark: flash attention (ctx=%d tokens, 8 heads, d=128, single query)\n", bench_ctx);

        auto bench_fa = [&](ggml_type kt, ggml_type vt, const char * label) {
            printf("  %-24s ", label);
            ggml_backend_t be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, NULL);
            if (!be) { printf("SKIP\n"); return; }

            const int nh = 8, hd = 128;
            prng_seed(0xFABE);
            std::vector<float> q(hd * nh);
            for (auto & v : q) v = prng_gaussian() * 0.1f;

            std::vector<float> kf((size_t)bench_ctx * nh * hd);
            std::vector<float> vf((size_t)bench_ctx * nh * hd);
            for (auto & v : kf) v = prng_gaussian();
            for (auto & v : vf) v = prng_gaussian();

            size_t kbs = (hd / ggml_blck_size(kt)) * ggml_type_size(kt);
            size_t vbs = (hd / ggml_blck_size(vt)) * ggml_type_size(vt);
            std::vector<uint8_t> kq(kbs * bench_ctx * nh), vq(vbs * bench_ctx * nh);

            auto kff = ggml_get_type_traits(kt)->from_float_ref;
            auto vff = ggml_get_type_traits(vt)->from_float_ref;
            for (int h = 0; h < nh; h++) {
                for (int t = 0; t < bench_ctx; t++) {
                    kff(kf.data() + ((size_t)t*nh+h)*hd, (void*)(kq.data() + ((size_t)h*bench_ctx+t)*kbs), hd);
                    vff(vf.data() + ((size_t)t*nh+h)*hd, (void*)(vq.data() + ((size_t)h*bench_ctx+t)*vbs), hd);
                }
            }

            struct ggml_init_params p = {256*1024*1024, NULL, true};
            struct ggml_context * c = ggml_init(p);
            auto * tQ = ggml_new_tensor_3d(c, GGML_TYPE_F32, hd, 1, nh);
            auto * tK = ggml_new_tensor_3d(c, kt, hd, bench_ctx, nh);
            ggml_set_name(tK, "cache_k_l0");
            auto * tV = ggml_new_tensor_3d(c, vt, hd, bench_ctx, nh);
            ggml_set_name(tV, "cache_v_l0");
            float sc = 1.0f / sqrtf((float)hd);
            auto * fa = ggml_flash_attn_ext(c, tQ, tK, tV, NULL, sc, 0, 0);
            auto * buf = ggml_backend_alloc_ctx_tensors(c, be);
            ggml_backend_tensor_set(tQ, q.data(), 0, q.size()*4);
            ggml_backend_tensor_set(tK, kq.data(), 0, kq.size());
            ggml_backend_tensor_set(tV, vq.data(), 0, vq.size());
            auto * g = ggml_new_graph(c);
            ggml_build_forward_expand(g, fa);
            ggml_backend_graph_compute(be, g); // warmup

            int ni = 50;
            double t0 = ggml_time_us();
            for (int i = 0; i < ni; i++) ggml_backend_graph_compute(be, g);
            double t1 = ggml_time_us();
            double ms = (t1-t0)/1000.0/ni;
            double kv_kb = (kq.size() + vq.size()) / 1024.0;
            printf("%6.3f ms  (KV=%.0f KB)\n", ms, kv_kb);

            ggml_backend_buffer_free(buf);
            ggml_free(c);
            ggml_backend_free(be);
        };

        bench_fa(GGML_TYPE_F16,  GGML_TYPE_F16,  "f16+f16");
        bench_fa(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, "q8_0+q8_0");
        bench_fa(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, "q4_0+q4_0");
        bench_fa(GGML_TYPE_TURBO4_0_PROD, GGML_TYPE_TURBO4_0_MSE, "tqk35+tqv35");
        printf("\n");
    }

    return n_fail > 0 ? 1 : 0;
}
