#pragma once

#include "common.cuh"
#include "tq-common.cuh"

// TurboQuant set_rows (quantize f32 → TQ block) kernels.
// Implementations must be in the header since they're __device__ functions
// used as template parameters in k_set_rows_quant (set-rows.cu).

// ---------------------------------------------------------------------------
// TQK had_mse4: H_128 Hadamard + 4-bit MSE, no QJL
// ---------------------------------------------------------------------------
static __device__ void quantize_f32_tqk_had_mse4_block(const float * src, block_tqk_had_mse4 * dst) {
    float rot[128];
    float sum_sq = 0.0f;
    for (int j = 0; j < 128; j++) { rot[j] = src[j]; sum_sq += src[j] * src[j]; }
    float norm = sqrtf(sum_sq);
    dst->norm = __float2half(norm);
    if (norm == 0.0f) { memset(dst->qs, 0, sizeof(dst->qs)); return; }
    float inv = 1.0f / norm;
    for (int j = 0; j < 128; j++) rot[j] *= inv;
    tq_fwht_local<128>(rot);
    memset(dst->qs, 0, sizeof(dst->qs));
    for (int j = 0; j < 128; j++) tq_pk4(dst->qs, j, tq_nearest(rot[j], tq_c16_d128, 16));
}

// TQV had_mse4 uses same codec
static __device__ void quantize_f32_tqv_had_mse4_block(const float * src, block_tqv_had_mse4 * dst) {
    quantize_f32_tqk_had_mse4_block(src, dst);
}

// ---------------------------------------------------------------------------
// TQK had_prod5: H_128 Hadamard + 4-bit MSE + 1-bit QJL on residual
// ---------------------------------------------------------------------------
static __device__ void quantize_f32_tqk_had_prod5_block(const float * src, block_tqk_had_prod5 * dst) {
    float rot[128];
    float sum_sq = 0.0f;
    for (int j = 0; j < 128; j++) { rot[j] = src[j]; sum_sq += src[j] * src[j]; }
    float norm = sqrtf(sum_sq);
    dst->norm = __float2half(norm);
    dst->rnorm = __float2half(0.0f);
    memset(dst->signs, 0, sizeof(dst->signs));
    if (norm == 0.0f) { memset(dst->qs, 0, sizeof(dst->qs)); return; }

    float inv = 1.0f / norm;
    for (int j = 0; j < 128; j++) rot[j] *= inv;
    tq_fwht_local<128>(rot);
    memset(dst->qs, 0, sizeof(dst->qs));
    for (int j = 0; j < 128; j++) tq_pk4(dst->qs, j, tq_nearest(rot[j], tq_c16_d128, 16));

    // QJL: residual in original space
    float recon[128];
    for (int j = 0; j < 128; j++) recon[j] = tq_c16_d128[tq_up4(dst->qs, j)];
    tq_fwht_local<128>(recon);
    float resid[128];
    float rnorm_sq = 0.0f;
    for (int j = 0; j < 128; j++) { resid[j] = src[j] - norm * recon[j]; rnorm_sq += resid[j] * resid[j]; }
    tq_fwht_local<128>(resid);
    for (int j = 0; j < 128; j++) {
        if (resid[j] >= 0.0f) dst->signs[j / 8] |= (uint8_t)(1 << (j % 8));
    }
    dst->rnorm = __float2half(sqrtf(rnorm_sq));
}

// ---------------------------------------------------------------------------
// TQK had_prod4: H_128 Hadamard + 3-bit MSE + 1-bit QJL
// ---------------------------------------------------------------------------
static __device__ void quantize_f32_tqk_had_prod4_block(const float * src, block_tqk_had_prod4 * dst) {
    float rot[128];
    float sum_sq = 0.0f;
    for (int j = 0; j < 128; j++) { rot[j] = src[j]; sum_sq += src[j] * src[j]; }
    float norm = sqrtf(sum_sq);
    dst->norm = __float2half(norm);
    dst->rnorm = __float2half(0.0f);
    memset(dst->signs, 0, sizeof(dst->signs));
    if (norm == 0.0f) { memset(dst->qs, 0, sizeof(dst->qs)); return; }

    float inv = 1.0f / norm;
    for (int j = 0; j < 128; j++) rot[j] *= inv;
    tq_fwht_local<128>(rot);
    memset(dst->qs, 0, sizeof(dst->qs));
    for (int j = 0; j < 128; j++) tq_pk3(dst->qs, j, tq_nearest(rot[j], tq_c8_d128, 8));

    // QJL: residual in original space
    float recon[128];
    for (int j = 0; j < 128; j++) recon[j] = tq_c8_d128[tq_up3(dst->qs, j)];
    tq_fwht_local<128>(recon);
    float resid[128];
    float rnorm_sq = 0.0f;
    for (int j = 0; j < 128; j++) { resid[j] = src[j] - norm * recon[j]; rnorm_sq += resid[j] * resid[j]; }
    tq_fwht_local<128>(resid);
    for (int j = 0; j < 128; j++) {
        if (resid[j] >= 0.0f) dst->signs[j / 8] |= (uint8_t)(1 << (j % 8));
    }
    dst->rnorm = __float2half(sqrtf(rnorm_sq));
}

// Split types — custom kernels, need channel map + layer/head
void ggml_cuda_op_set_rows_tq_5hi_3lo_had(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_set_rows_tq_6hi_3lo_had(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_set_rows_tq_2hi_1lo_had(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_set_rows_tq_3hi_2lo_had(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
