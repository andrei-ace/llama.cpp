// TurboQuant CUDA device function stubs
// TODO(TurboQuant): Replace with actual CUDA dequantization/quantization kernels

#pragma once

#include "common.cuh"

// Dequantize turbo3_0_prod block element for get_rows
// TODO(TurboQuant): implement 3-bit PolarQuant + QJL dequantization
static __device__ __forceinline__ void dequantize_turbo3_0_prod(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    NO_DEVICE_CODE;
    GGML_UNUSED(vx);
    GGML_UNUSED(ib);
    GGML_UNUSED(iqs);
    GGML_UNUSED(v);
}

// Dequantize turbo4_0_prod block element for get_rows
// TODO(TurboQuant): implement 4-bit PolarQuant + QJL dequantization
static __device__ __forceinline__ void dequantize_turbo4_0_prod(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    NO_DEVICE_CODE;
    GGML_UNUSED(vx);
    GGML_UNUSED(ib);
    GGML_UNUSED(iqs);
    GGML_UNUSED(v);
}

// Quantize float32 block to turbo3_0_prod for set_rows
// TODO(TurboQuant): implement 3-bit quantization with rotation + PolarQuant
static __device__ void quantize_f32_turbo3_0_prod_block(const float * __restrict__ x, block_turbo3_0_prod * __restrict__ y) {
    NO_DEVICE_CODE;
    GGML_UNUSED(x);
    GGML_UNUSED(y);
}

// Quantize float32 block to turbo4_0_prod for set_rows
// TODO(TurboQuant): implement 4-bit quantization with rotation + PolarQuant + QJL
static __device__ void quantize_f32_turbo4_0_prod_block(const float * __restrict__ x, block_turbo4_0_prod * __restrict__ y) {
    NO_DEVICE_CODE;
    GGML_UNUSED(x);
    GGML_UNUSED(y);
}

// Dequantize turbo3_0_mse block element for get_rows
// TODO(TurboQuant): implement 3-bit MSE-only dequantization
static __device__ __forceinline__ void dequantize_turbo3_0_mse(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    NO_DEVICE_CODE;
    GGML_UNUSED(vx);
    GGML_UNUSED(ib);
    GGML_UNUSED(iqs);
    GGML_UNUSED(v);
}

// Dequantize turbo4_0_mse block element for get_rows
// TODO(TurboQuant): implement 4-bit MSE-only dequantization
static __device__ __forceinline__ void dequantize_turbo4_0_mse(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    NO_DEVICE_CODE;
    GGML_UNUSED(vx);
    GGML_UNUSED(ib);
    GGML_UNUSED(iqs);
    GGML_UNUSED(v);
}

// Quantize float32 block to turbo3_0_mse for set_rows
// TODO(TurboQuant): implement MSE-only quantization
static __device__ void quantize_f32_turbo3_0_mse_block(const float * __restrict__ x, block_turbo3_0_mse * __restrict__ y) {
    NO_DEVICE_CODE;
    GGML_UNUSED(x);
    GGML_UNUSED(y);
}

// Quantize float32 block to turbo4_0_mse for set_rows
// TODO(TurboQuant): implement MSE-only quantization
static __device__ void quantize_f32_turbo4_0_mse_block(const float * __restrict__ x, block_turbo4_0_mse * __restrict__ y) {
    NO_DEVICE_CODE;
    GGML_UNUSED(x);
    GGML_UNUSED(y);
}
