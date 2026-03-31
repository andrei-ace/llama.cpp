#pragma once

#include "common.cuh"

// TurboQuant get_rows (dequantize TQ block → f32) kernels.
// Custom kernels — cannot use the standard k_get_rows template because
// TQ requires full 128-element FWHT over the entire block.

// Host-side dispatch functions called from getrows.cu
template <typename dst_t>
void get_rows_tq_had_mse4(
    const void * src0_d, const int32_t * src1_d, dst_t * dst_d,
    int64_t ne00, size_t nb01, size_t nb02, size_t nb03,
    int64_t ne10, int64_t ne11, int64_t ne12,
    size_t nb10, size_t nb11, size_t nb12,
    size_t nb1, size_t nb2, size_t nb3,
    cudaStream_t stream);

template <typename dst_t>
void get_rows_tq_had_prod5(
    const void * src0_d, const int32_t * src1_d, dst_t * dst_d,
    int64_t ne00, size_t nb01, size_t nb02, size_t nb03,
    int64_t ne10, int64_t ne11, int64_t ne12,
    size_t nb10, size_t nb11, size_t nb12,
    size_t nb1, size_t nb2, size_t nb3,
    cudaStream_t stream);

template <typename dst_t>
void get_rows_tq_had_prod4(
    const void * src0_d, const int32_t * src1_d, dst_t * dst_d,
    int64_t ne00, size_t nb01, size_t nb02, size_t nb03,
    int64_t ne10, int64_t ne11, int64_t ne12,
    size_t nb10, size_t nb11, size_t nb12,
    size_t nb1, size_t nb2, size_t nb3,
    cudaStream_t stream);

template <typename dst_t>
void get_rows_tq_5hi_3lo_had(
    const void * src0_d, const int32_t * src1_d, dst_t * dst_d,
    int64_t ne00, size_t nb01, size_t nb02, size_t nb03,
    int64_t ne10, int64_t ne11, int64_t ne12,
    size_t nb10, size_t nb11, size_t nb12,
    size_t nb1, size_t nb2, size_t nb3,
    cudaStream_t stream);

template <typename dst_t>
void get_rows_tq_6hi_3lo_had(
    const void * src0_d, const int32_t * src1_d, dst_t * dst_d,
    int64_t ne00, size_t nb01, size_t nb02, size_t nb03,
    int64_t ne10, int64_t ne11, int64_t ne12,
    size_t nb10, size_t nb11, size_t nb12,
    size_t nb1, size_t nb2, size_t nb3,
    cudaStream_t stream);

template <typename dst_t>
void get_rows_tq_2hi_1lo_had(
    const void * src0_d, const int32_t * src1_d, dst_t * dst_d,
    int64_t ne00, size_t nb01, size_t nb02, size_t nb03,
    int64_t ne10, int64_t ne11, int64_t ne12,
    size_t nb10, size_t nb11, size_t nb12,
    size_t nb1, size_t nb2, size_t nb3,
    cudaStream_t stream);

template <typename dst_t>
void get_rows_tq_3hi_2lo_had(
    const void * src0_d, const int32_t * src1_d, dst_t * dst_d,
    int64_t ne00, size_t nb01, size_t nb02, size_t nb03,
    int64_t ne10, int64_t ne11, int64_t ne12,
    size_t nb10, size_t nb11, size_t nb12,
    size_t nb1, size_t nb2, size_t nb3,
    cudaStream_t stream);
