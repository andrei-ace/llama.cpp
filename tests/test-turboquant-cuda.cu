// TurboQuant CUDA kernel correctness test
//
// Tests GPU dequantize/quantize against CPU reference.
// For each type: quantize on CPU, dequantize on GPU, compare to CPU dequant.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// ggml-common.h provides block structure definitions (needs CUDA decl mode)
#define GGML_COMMON_DECL_CUDA
#include "ggml-common.h"

// Skip common.cuh include in turbo-quant-cuda.cuh (we provide types directly)
#define TURBO_QUANT_STANDALONE_TEST

// Forward declare CPU reference functions
extern "C" {
    void quantize_row_turbo3_0_mse_ref(const float * x, block_turbo3_0_mse * y, int64_t k);
    void dequantize_row_turbo3_0_mse(const block_turbo3_0_mse * x, float * y, int64_t k);
    void quantize_row_turbo4_0_mse_ref(const float * x, block_turbo4_0_mse * y, int64_t k);
    void dequantize_row_turbo4_0_mse(const block_turbo4_0_mse * x, float * y, int64_t k);
    void quantize_row_turbo3_0_prod_ref(const float * x, block_turbo3_0_prod * y, int64_t k);
    void dequantize_row_turbo3_0_prod(const block_turbo3_0_prod * x, float * y, int64_t k);
    void quantize_row_turbo4_0_prod_ref(const float * x, block_turbo4_0_prod * y, int64_t k);
    void dequantize_row_turbo4_0_prod(const block_turbo4_0_prod * x, float * y, int64_t k);
}

// Include the CUDA kernels directly
#include "turbo-quant-cuda.cuh"

// ---------------------------------------------------------------------------
// Test kernel: dequantize all elements of quantized blocks on GPU
// ---------------------------------------------------------------------------

template<typename block_t, void (*dequant_fn)(const void *, int64_t, int, float2 &)>
__global__ void kernel_dequant_all(const block_t * src, float * dst, int64_t n_elements, int qk) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int element_pair = idx;  // each thread handles 2 elements
    const int j = element_pair * 2;
    if (j >= n_elements) return;

    const int ib  = j / qk;
    const int iqs = j % qk;

    float2 v;
    dequant_fn((const void *)src, ib, iqs, v);

    dst[j + 0] = v.x;
    if (j + 1 < n_elements) {
        dst[j + 1] = v.y;
    }
}

// ---------------------------------------------------------------------------
// Test kernel: quantize blocks on GPU
// ---------------------------------------------------------------------------

template<typename block_t, int qk, void (*quant_fn)(const float *, block_t *)>
__global__ void kernel_quant_all(const float * src, block_t * dst, int64_t n_blocks) {
    const int ib = blockIdx.x * blockDim.x + threadIdx.x;
    if (ib >= n_blocks) return;
    quant_fn(src + ib * qk, dst + ib);
}

// ---------------------------------------------------------------------------
// PRNG for test data generation
// ---------------------------------------------------------------------------

static uint64_t test_prng = 42;
static float test_gaussian() {
    test_prng = test_prng * 6364136223846793005ULL + 1442695040888963407ULL;
    float u1 = ((float)(uint32_t)(test_prng >> 32) + 1.0f) / ((float)0xFFFFFFFF + 2.0f);
    test_prng = test_prng * 6364136223846793005ULL + 1442695040888963407ULL;
    float u2 = ((float)(uint32_t)(test_prng >> 32) + 1.0f) / ((float)0xFFFFFFFF + 2.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

// ---------------------------------------------------------------------------
// Comparison helper
// ---------------------------------------------------------------------------

struct cmp_result {
    float max_abs_err;
    float mean_abs_err;
    int n_mismatches;  // above threshold
};

static cmp_result compare(const float * a, const float * b, int n, float threshold) {
    cmp_result r = {0.0f, 0.0f, 0};
    double sum_err = 0.0;
    for (int i = 0; i < n; i++) {
        float err = fabsf(a[i] - b[i]);
        if (err > r.max_abs_err) r.max_abs_err = err;
        sum_err += err;
        if (err > threshold) r.n_mismatches++;
    }
    r.mean_abs_err = (float)(sum_err / n);
    return r;
}

// ---------------------------------------------------------------------------
// Test: MSE dequantize (GPU vs CPU)
// ---------------------------------------------------------------------------

template<typename block_t, int qk,
         void (*cpu_quant)(const float *, block_t *, int64_t),
         void (*cpu_dequant)(const block_t *, float *, int64_t),
         void (*gpu_dequant)(const void *, int64_t, int, float2 &)>
static bool test_mse_dequant(const char * name, int n_blocks) {
    const int n = n_blocks * qk;
    const size_t block_bytes = n_blocks * sizeof(block_t);

    // Generate test data
    float * h_input = (float *)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) h_input[i] = test_gaussian();

    // CPU quantize
    block_t * h_quant = (block_t *)malloc(block_bytes);
    cpu_quant(h_input, h_quant, n);

    // CPU dequantize (reference)
    float * h_cpu_deq = (float *)malloc(n * sizeof(float));
    cpu_dequant(h_quant, h_cpu_deq, n);

    // GPU dequantize
    block_t * d_quant;
    float * d_deq;
    cudaMalloc(&d_quant, block_bytes);
    cudaMalloc(&d_deq, n * sizeof(float));
    cudaMemcpy(d_quant, h_quant, block_bytes, cudaMemcpyHostToDevice);

    const int threads = 256;
    const int pairs = (n + 1) / 2;
    const int blocks = (pairs + threads - 1) / threads;
    kernel_dequant_all<block_t, gpu_dequant><<<blocks, threads>>>(d_quant, d_deq, n, qk);
    cudaDeviceSynchronize();

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("  %s dequant: CUDA error: %s\n", name, cudaGetErrorString(err));
        free(h_input); free(h_quant); free(h_cpu_deq);
        cudaFree(d_quant); cudaFree(d_deq);
        return false;
    }

    float * h_gpu_deq = (float *)malloc(n * sizeof(float));
    cudaMemcpy(h_gpu_deq, d_deq, n * sizeof(float), cudaMemcpyDeviceToHost);

    cmp_result r = compare(h_cpu_deq, h_gpu_deq, n, 1e-5f);
    bool ok = r.max_abs_err < 1e-4f;

    printf("  %s dequant (%d blocks): max_err=%.2e mean_err=%.2e mismatches=%d %s\n",
           name, n_blocks, r.max_abs_err, r.mean_abs_err, r.n_mismatches,
           ok ? "ok" : "FAILED");

    if (!ok) {
        // Print first few mismatches for debugging
        int printed = 0;
        for (int i = 0; i < n && printed < 10; i++) {
            float diff = fabsf(h_cpu_deq[i] - h_gpu_deq[i]);
            if (diff > 1e-5f) {
                printf("    [%d] cpu=%.8f gpu=%.8f diff=%.2e\n",
                       i, h_cpu_deq[i], h_gpu_deq[i], diff);
                printed++;
            }
        }
    }

    free(h_input); free(h_quant); free(h_cpu_deq); free(h_gpu_deq);
    cudaFree(d_quant); cudaFree(d_deq);
    return ok;
}

// ---------------------------------------------------------------------------
// Test: MSE quantize (GPU vs CPU)
// ---------------------------------------------------------------------------

template<typename block_t, int qk,
         void (*cpu_quant)(const float *, block_t *, int64_t),
         void (*gpu_quant)(const float *, block_t *)>
static bool test_mse_quant(const char * name, int n_blocks) {
    const int n = n_blocks * qk;
    const size_t block_bytes = n_blocks * sizeof(block_t);

    // Generate test data
    float * h_input = (float *)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) h_input[i] = test_gaussian();

    // CPU quantize (reference)
    block_t * h_cpu_quant = (block_t *)malloc(block_bytes);
    cpu_quant(h_input, h_cpu_quant, n);

    // GPU quantize
    float * d_input;
    block_t * d_quant;
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_quant, block_bytes);
    cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_quant, 0, block_bytes);

    const int threads = 64;
    const int grid = (n_blocks + threads - 1) / threads;
    kernel_quant_all<block_t, qk, gpu_quant><<<grid, threads>>>(d_input, d_quant, n_blocks);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("  %s quant: CUDA error: %s\n", name, cudaGetErrorString(err));
        free(h_input); free(h_cpu_quant);
        cudaFree(d_input); cudaFree(d_quant);
        return false;
    }

    block_t * h_gpu_quant = (block_t *)malloc(block_bytes);
    cudaMemcpy(h_gpu_quant, d_quant, block_bytes, cudaMemcpyDeviceToHost);

    // Compare block bytes
    int byte_mismatches = 0;
    const uint8_t * cpu_bytes = (const uint8_t *)h_cpu_quant;
    const uint8_t * gpu_bytes = (const uint8_t *)h_gpu_quant;
    for (size_t i = 0; i < block_bytes; i++) {
        if (cpu_bytes[i] != gpu_bytes[i]) byte_mismatches++;
    }

    bool ok = byte_mismatches == 0;
    printf("  %s quant (%d blocks): byte_mismatches=%d/%zu %s\n",
           name, n_blocks, byte_mismatches, block_bytes,
           ok ? "ok" : "FAILED");

    if (!ok) {
        // Show first block differences
        const int bs = sizeof(block_t);
        for (int b = 0; b < n_blocks && b < 3; b++) {
            bool block_ok = true;
            for (int i = 0; i < bs; i++) {
                if (cpu_bytes[b * bs + i] != gpu_bytes[b * bs + i]) {
                    block_ok = false;
                    break;
                }
            }
            if (!block_ok) {
                printf("    block %d differs:\n      cpu:", b);
                for (int i = 0; i < bs; i++) printf(" %02x", cpu_bytes[b * bs + i]);
                printf("\n      gpu:");
                for (int i = 0; i < bs; i++) printf(" %02x", gpu_bytes[b * bs + i]);
                printf("\n");
            }
        }
    }

    free(h_input); free(h_cpu_quant); free(h_gpu_quant);
    cudaFree(d_input); cudaFree(d_quant);
    return ok;
}

// ---------------------------------------------------------------------------
// Test: PRNG consistency (CPU vs GPU)
// ---------------------------------------------------------------------------

__global__ void kernel_prng_test(float * out, uint64_t seed, int n) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    tq_prng_state st;
    tq_prng_init(st, seed);
    for (int i = 0; i < n; i++) {
        out[i] = tq_prng_gaussian(st);
    }
}

static bool test_prng_consistency() {
    const int n = 1000;
    const uint64_t seed = QJL_SEED_32;

    // CPU PRNG (replicate the same LCG)
    float * h_cpu = (float *)malloc(n * sizeof(float));
    {
        uint64_t s = seed;
        for (int i = 0; i < n; i++) {
            s = s * 6364136223846793005ULL + 1442695040888963407ULL;
            float u1 = ((float)(uint32_t)(s >> 32) + 1.0f) / ((float)0xFFFFFFFF + 2.0f);
            s = s * 6364136223846793005ULL + 1442695040888963407ULL;
            float u2 = ((float)(uint32_t)(s >> 32) + 1.0f) / ((float)0xFFFFFFFF + 2.0f);
            h_cpu[i] = sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
        }
    }

    // GPU PRNG
    float * d_gpu, * h_gpu;
    h_gpu = (float *)malloc(n * sizeof(float));
    cudaMalloc(&d_gpu, n * sizeof(float));
    kernel_prng_test<<<1, 1>>>(d_gpu, seed, n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_gpu, d_gpu, n * sizeof(float), cudaMemcpyDeviceToHost);

    int mismatches = 0;
    float max_diff = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = fabsf(h_cpu[i] - h_gpu[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > 1e-6f) mismatches++;
    }

    bool ok = mismatches == 0;
    printf("  PRNG consistency (%d values, seed=0x%llx): max_diff=%.2e mismatches=%d %s\n",
           n, (unsigned long long)seed, max_diff, mismatches, ok ? "ok" : "FAILED");

    if (!ok && mismatches <= 5) {
        for (int i = 0; i < n; i++) {
            float diff = fabsf(h_cpu[i] - h_gpu[i]);
            if (diff > 1e-6f) {
                printf("    [%d] cpu=%.10f gpu=%.10f diff=%.2e\n",
                       i, h_cpu[i], h_gpu[i], diff);
            }
        }
    }

    free(h_cpu); free(h_gpu);
    cudaFree(d_gpu);
    return ok;
}

// ---------------------------------------------------------------------------
// Test: Zero vector roundtrip
// ---------------------------------------------------------------------------

template<typename block_t, int qk,
         void (*cpu_quant)(const float *, block_t *, int64_t),
         void (*gpu_dequant)(const void *, int64_t, int, float2 &)>
static bool test_zero_vector(const char * name) {
    float input[qk] = {0};
    block_t quant;
    cpu_quant(input, &quant, qk);

    // GPU dequant
    block_t * d_quant;
    float * d_deq;
    cudaMalloc(&d_quant, sizeof(block_t));
    cudaMalloc(&d_deq, qk * sizeof(float));
    cudaMemcpy(d_quant, &quant, sizeof(block_t), cudaMemcpyHostToDevice);

    kernel_dequant_all<block_t, gpu_dequant><<<1, 64>>>(d_quant, d_deq, qk, qk);
    cudaDeviceSynchronize();

    float h_deq[qk];
    cudaMemcpy(h_deq, d_deq, qk * sizeof(float), cudaMemcpyDeviceToHost);

    float max_err = 0.0f;
    for (int i = 0; i < qk; i++) {
        max_err = fmaxf(max_err, fabsf(h_deq[i]));
    }

    bool ok = max_err == 0.0f;
    printf("  %s zero vector: max=%.2e %s\n", name, max_err, ok ? "ok" : "FAILED");

    cudaFree(d_quant); cudaFree(d_deq);
    return ok;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("TurboQuant CUDA test on %s (cc %d.%d)\n\n", prop.name, prop.major, prop.minor);

    bool all_ok = true;

    // Phase 1: PRNG consistency
    printf("=== PRNG Consistency ===\n");
    all_ok &= test_prng_consistency();
    printf("\n");

    // Phase 2: MSE zero vectors
    printf("=== Zero Vector Roundtrip ===\n");
    all_ok &= test_zero_vector<block_turbo3_0_mse, QK_TURBO3_MSE,
        quantize_row_turbo3_0_mse_ref, dequantize_turbo3_0_mse>("tqv_lo");
    all_ok &= test_zero_vector<block_turbo4_0_mse, QK_TURBO4_MSE,
        quantize_row_turbo4_0_mse_ref, dequantize_turbo4_0_mse>("tqv_hi");
    all_ok &= test_zero_vector<block_turbo3_0_prod, QK_TURBO3_PROD,
        quantize_row_turbo3_0_prod_ref, dequantize_turbo3_0_prod>("tqk_lo");
    all_ok &= test_zero_vector<block_turbo4_0_prod, QK_TURBO4_PROD,
        quantize_row_turbo4_0_prod_ref, dequantize_turbo4_0_prod>("tqk_hi");
    printf("\n");

    // Phase 3: MSE dequantize correctness
    printf("=== MSE Dequantize (GPU vs CPU) ===\n");
    for (int nb : {1, 4, 16, 64}) {
        all_ok &= test_mse_dequant<block_turbo3_0_mse, QK_TURBO3_MSE,
            quantize_row_turbo3_0_mse_ref, dequantize_row_turbo3_0_mse,
            dequantize_turbo3_0_mse>("tqv_lo", nb);
        all_ok &= test_mse_dequant<block_turbo4_0_mse, QK_TURBO4_MSE,
            quantize_row_turbo4_0_mse_ref, dequantize_row_turbo4_0_mse,
            dequantize_turbo4_0_mse>("tqv_hi", nb);
    }
    printf("\n");

    // Phase 4: MSE quantize correctness
    printf("=== MSE Quantize (GPU vs CPU) ===\n");
    for (int nb : {1, 4, 16}) {
        all_ok &= test_mse_quant<block_turbo3_0_mse, QK_TURBO3_MSE,
            quantize_row_turbo3_0_mse_ref,
            quantize_f32_turbo3_0_mse_block>("tqv_lo", nb);
        all_ok &= test_mse_quant<block_turbo4_0_mse, QK_TURBO4_MSE,
            quantize_row_turbo4_0_mse_ref,
            quantize_f32_turbo4_0_mse_block>("tqv_hi", nb);
    }
    printf("\n");

    // Phase 5: PROD dequantize correctness
    printf("=== PROD Dequantize (GPU vs CPU) ===\n");
    for (int nb : {1, 4}) {
        all_ok &= test_mse_dequant<block_turbo3_0_prod, QK_TURBO3_PROD,
            quantize_row_turbo3_0_prod_ref, dequantize_row_turbo3_0_prod,
            dequantize_turbo3_0_prod>("tqk_lo", nb);
        all_ok &= test_mse_dequant<block_turbo4_0_prod, QK_TURBO4_PROD,
            quantize_row_turbo4_0_prod_ref, dequantize_row_turbo4_0_prod,
            dequantize_turbo4_0_prod>("tqk_hi", nb);
    }
    printf("\n");

    // Phase 6: PROD quantize correctness
    printf("=== PROD Quantize (GPU vs CPU) ===\n");
    for (int nb : {1, 4}) {
        all_ok &= test_mse_quant<block_turbo3_0_prod, QK_TURBO3_PROD,
            quantize_row_turbo3_0_prod_ref,
            quantize_f32_turbo3_0_prod_block>("tqk_lo", nb);
        all_ok &= test_mse_quant<block_turbo4_0_prod, QK_TURBO4_PROD,
            quantize_row_turbo4_0_prod_ref,
            quantize_f32_turbo4_0_prod_block>("tqk_hi", nb);
    }
    printf("\n");

    printf("==========================================\n");
    printf("%s\n", all_ok ? "All CUDA tests passed" : "SOME TESTS FAILED");
    return all_ok ? 0 : 1;
}
