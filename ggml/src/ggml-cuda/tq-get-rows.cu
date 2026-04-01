// TurboQuant get_rows (dequantize TQ block → f32/f16) CUDA kernels.
// Custom kernels — cannot use standard k_get_rows template because TQ
// needs full 128-element FWHT over each block.
//
// One thread per TQ block (128 output elements).

#include "tq-get-rows.cuh"
#include "tq-common.cuh"
#include "convert.cuh"

// Forward declaration of channel map accessor
extern int32_t * ggml_cuda_get_tq_channel_map_device(void);
extern int       ggml_cuda_get_tq_chmap_n_heads(void);

// ---------------------------------------------------------------------------
// Generic TQ get_rows kernel — one thread per 128-element block
// The dequantize logic differs per type, so we use separate kernels.
// ---------------------------------------------------------------------------

template <typename dst_t>
static __global__ void k_get_rows_tq_had_mse4(
        const void * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int64_t ne00,
        const int64_t ne11, const int64_t ne12,
        const size_t s1, const size_t s2, const size_t s3,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12) {

    const int64_t n_blocks_per_row = ne00 / 128;
    const int64_t total_blocks = (int64_t)gridDim.x * blockDim.x; // max thread range
    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    // Iterate over z (ne11*ne12), then row (i10=blockIdx.x-like), then block within row
    for (int64_t z = blockIdx.z; z < ne11 * ne12; z += gridDim.z) {
        const int64_t i10 = blockIdx.x;
        const int64_t block_in_row = threadIdx.x; // which 128-element block within this row

        if (block_in_row >= n_blocks_per_row) return;

        const int i11 = z / ne12;
        const int i12 = z % ne12;

        const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

        dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
        const char * src0_row = (const char *)src0 + i01*nb01 + i11*nb02 + i12*nb03;
        const block_tqk_had_mse4 * blk = (const block_tqk_had_mse4 *)src0_row + block_in_row;

        float norm = __half2float(blk->norm);

        float rot[128];
        for (int j = 0; j < 128; j++) {
            rot[j] = tq_c16_d128[tq_up4(blk->qs, j)];
        }
        tq_fwht_local<128>(rot); // inverse FWHT

        const int64_t base = block_in_row * 128;
        for (int j = 0; j < 128; j++) {
            dst_row[base + j] = ggml_cuda_cast<dst_t>(norm * rot[j]);
        }
    }
}

template <typename dst_t>
static __global__ void k_get_rows_tq_had_prod5(
        const void * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int64_t ne00,
        const int64_t ne11, const int64_t ne12,
        const size_t s1, const size_t s2, const size_t s3,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12) {

    for (int64_t z = blockIdx.z; z < ne11 * ne12; z += gridDim.z) {
        const int64_t block_in_row = threadIdx.x;
        if (block_in_row >= ne00 / 128) return;

        const int i11 = z / ne12;
        const int i12 = z % ne12;
        const int i10 = blockIdx.x;
        const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

        dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
        const char * src0_row = (const char *)src0 + i01*nb01 + i11*nb02 + i12*nb03;
        const block_tqk_had_prod5 * blk = (const block_tqk_had_prod5 *)src0_row + block_in_row;

        float norm  = __half2float(blk->norm);
        float rnorm = __half2float(blk->rnorm);

        // MSE reconstruction
        float rot[128];
        for (int j = 0; j < 128; j++) rot[j] = tq_c16_d128[tq_up4(blk->qs, j)];
        tq_fwht_local<128>(rot);
        for (int j = 0; j < 128; j++) rot[j] *= norm;

        // QJL correction: inverse FWHT of sign bits, scale by rnorm
        float corr[128];
        for (int j = 0; j < 128; j++) corr[j] = tq_sign_bit(blk->signs, j);
        tq_fwht_local<128>(corr);
        float qjl_scale = QJL_SCALE_128 * rnorm;
        for (int j = 0; j < 128; j++) rot[j] += qjl_scale * corr[j];

        const int64_t base = block_in_row * 128;
        for (int j = 0; j < 128; j++) {
            dst_row[base + j] = ggml_cuda_cast<dst_t>(rot[j]);
        }
    }
}

template <typename dst_t>
static __global__ void k_get_rows_tq_had_prod4(
        const void * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int64_t ne00,
        const int64_t ne11, const int64_t ne12,
        const size_t s1, const size_t s2, const size_t s3,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12) {

    for (int64_t z = blockIdx.z; z < ne11 * ne12; z += gridDim.z) {
        const int64_t block_in_row = threadIdx.x;
        if (block_in_row >= ne00 / 128) return;

        const int i11 = z / ne12;
        const int i12 = z % ne12;
        const int i10 = blockIdx.x;
        const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

        dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
        const char * src0_row = (const char *)src0 + i01*nb01 + i11*nb02 + i12*nb03;
        const block_tqk_had_prod4 * blk = (const block_tqk_had_prod4 *)src0_row + block_in_row;

        float norm  = __half2float(blk->norm);
        float rnorm = __half2float(blk->rnorm);

        // MSE reconstruction (3-bit)
        float rot[128];
        for (int j = 0; j < 128; j++) rot[j] = tq_c8_d128[tq_up3(blk->qs, j)];
        tq_fwht_local<128>(rot);
        for (int j = 0; j < 128; j++) rot[j] *= norm;

        // QJL correction
        float corr[128];
        for (int j = 0; j < 128; j++) corr[j] = tq_sign_bit(blk->signs, j);
        tq_fwht_local<128>(corr);
        float qjl_scale = QJL_SCALE_128 * rnorm;
        for (int j = 0; j < 128; j++) rot[j] += qjl_scale * corr[j];

        const int64_t base = block_in_row * 128;
        for (int j = 0; j < 128; j++) {
            dst_row[base + j] = ggml_cuda_cast<dst_t>(rot[j]);
        }
    }
}

template <typename dst_t>
static __global__ void k_get_rows_tq_5hi_3lo_had(
        const void * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int32_t * __restrict__ chmap, const int32_t n_kv_heads, const int32_t layer_idx,
        const int64_t ne00,
        const int64_t ne11, const int64_t ne12,
        const size_t s1, const size_t s2, const size_t s3,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12) {

    for (int64_t z = blockIdx.z; z < ne11 * ne12; z += gridDim.z) {
        const int64_t block_in_row = threadIdx.x;
        if (block_in_row >= ne00 / 128) return;

        const int i11 = z / ne12;
        const int i12 = z % ne12;
        const int i10 = blockIdx.x;
        const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

        dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
        const char * src0_row = (const char *)src0 + i01*nb01 + i11*nb02 + i12*nb03;
        const block_tqk_5hi_3lo * blk = (const block_tqk_5hi_3lo *)src0_row + block_in_row;

        const int head = (int)(block_in_row % n_kv_heads);
        const int32_t * perm = chmap + ((int64_t)layer_idx * n_kv_heads + head) * 128;

        float norm_hi = __half2float(blk->norm_hi);
        float norm_lo = __half2float(blk->norm_lo);
        float rnorm_hi = __half2float(blk->rnorm_hi);

        // Decode hi: centroid lookup → inverse FWHT → scale
        float hi[32];
        for (int j = 0; j < 32; j++) hi[j] = tq_c16_d32[tq_up4(blk->qs_hi, j)];
        tq_fwht_local<32>(hi);
        for (int j = 0; j < 32; j++) hi[j] *= norm_hi;

        // QJL correction on hi
        float corr[32];
        for (int j = 0; j < 32; j++) corr[j] = tq_sign_bit(blk->signs_hi, j);
        tq_fwht_local<32>(corr);
        float qjl_scale = QJL_SCALE_32 * rnorm_hi;
        for (int j = 0; j < 32; j++) hi[j] += qjl_scale * corr[j];

        // Decode lo: centroid lookup → 3×inverse FWHT → scale
        // Shared 96-D norm → d96 centroids
        float lo[96];
        for (int j = 0; j < 96; j++) lo[j] = tq_c8_d96[tq_up3(blk->qs_lo, j)];
        tq_fwht_local<32>(lo);
        tq_fwht_local<32>(lo + 32);
        tq_fwht_local<32>(lo + 64);
        for (int j = 0; j < 96; j++) lo[j] *= norm_lo;

        // Inverse-permute via channel map and write
        const int64_t base = block_in_row * 128;
        for (int j = 0; j < 32; j++) dst_row[base + perm[j]] = ggml_cuda_cast<dst_t>(hi[j]);
        for (int j = 0; j < 96; j++) dst_row[base + perm[32 + j]] = ggml_cuda_cast<dst_t>(lo[j]);
    }
}

// ---------------------------------------------------------------------------
// 6hi_3lo_had: 5-bit MSE + QJL on hi, 3-bit MSE on lo (no QJL on lo)
// ---------------------------------------------------------------------------

template <typename dst_t>
static __global__ void k_get_rows_tq_6hi_3lo_had(
        const void * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int32_t * __restrict__ chmap, const int32_t n_kv_heads, const int32_t layer_idx,
        const int64_t ne00,
        const int64_t ne11, const int64_t ne12,
        const size_t s1, const size_t s2, const size_t s3,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12) {

    for (int64_t z = blockIdx.z; z < ne11 * ne12; z += gridDim.z) {
        const int64_t block_in_row = threadIdx.x;
        if (block_in_row >= ne00 / 128) return;

        const int i11 = z / ne12;
        const int i12 = z % ne12;
        const int i10 = blockIdx.x;
        const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

        dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
        const char * src0_row = (const char *)src0 + i01*nb01 + i11*nb02 + i12*nb03;
        const block_tqk_6hi_3lo * blk = (const block_tqk_6hi_3lo *)src0_row + block_in_row;

        const int head = (int)(block_in_row % n_kv_heads);
        const int32_t * perm = chmap + ((int64_t)layer_idx * n_kv_heads + head) * 128;

        float norm_hi  = __half2float(blk->norm_hi);
        float norm_lo  = __half2float(blk->norm_lo);
        float rnorm_hi = __half2float(blk->rnorm_hi);

        // Decode hi: 5-bit centroid lookup → inverse FWHT → scale
        float hi[32];
        for (int j = 0; j < 32; j++) hi[j] = tq_c32_d32[tq_up5(blk->qs_hi, j)];
        tq_fwht_local<32>(hi);
        for (int j = 0; j < 32; j++) hi[j] *= norm_hi;

        // QJL correction on hi
        float corr[32];
        for (int j = 0; j < 32; j++) corr[j] = tq_sign_bit(blk->signs_hi, j);
        tq_fwht_local<32>(corr);
        float qjl_scale = QJL_SCALE_32 * rnorm_hi;
        for (int j = 0; j < 32; j++) hi[j] += qjl_scale * corr[j];

        // Decode lo: 3-bit centroid lookup → 3×inverse FWHT → scale (no QJL)
        // Shared 96-D norm → d96 centroids
        float lo[96];
        for (int j = 0; j < 96; j++) lo[j] = tq_c8_d96[tq_up3(blk->qs_lo, j)];
        tq_fwht_local<32>(lo);
        tq_fwht_local<32>(lo + 32);
        tq_fwht_local<32>(lo + 64);
        for (int j = 0; j < 96; j++) lo[j] *= norm_lo;

        // Inverse-permute via channel map and write
        const int64_t base = block_in_row * 128;
        for (int j = 0; j < 32; j++) dst_row[base + perm[j]] = ggml_cuda_cast<dst_t>(hi[j]);
        for (int j = 0; j < 96; j++) dst_row[base + perm[32 + j]] = ggml_cuda_cast<dst_t>(lo[j]);
    }
}

// ---------------------------------------------------------------------------
// 2hi_1lo_had: 2-bit MSE + QJL on hi, 1-bit MSE + QJL on lo
// ---------------------------------------------------------------------------

template <typename dst_t>
static __global__ void k_get_rows_tq_2hi_1lo_had(
        const void * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int32_t * __restrict__ chmap, const int32_t n_kv_heads, const int32_t layer_idx,
        const int64_t ne00,
        const int64_t ne11, const int64_t ne12,
        const size_t s1, const size_t s2, const size_t s3,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12) {

    for (int64_t z = blockIdx.z; z < ne11 * ne12; z += gridDim.z) {
        const int64_t block_in_row = threadIdx.x;
        if (block_in_row >= ne00 / 128) return;

        const int i11 = z / ne12;
        const int i12 = z % ne12;
        const int i10 = blockIdx.x;
        const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

        dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
        const char * src0_row = (const char *)src0 + i01*nb01 + i11*nb02 + i12*nb03;
        const block_tqk_2hi_1lo * blk = (const block_tqk_2hi_1lo *)src0_row + block_in_row;

        const int head = (int)(block_in_row % n_kv_heads);
        const int32_t * perm = chmap + ((int64_t)layer_idx * n_kv_heads + head) * 128;

        float norm_hi  = __half2float(blk->norm_hi);
        float norm_lo  = __half2float(blk->norm_lo);
        float rnorm_hi = __half2float(blk->rnorm_hi);
        float rnorm_lo = __half2float(blk->rnorm_lo);

        // Decode hi: 2-bit centroid lookup → inverse FWHT → scale
        float hi[32];
        for (int j = 0; j < 32; j++) hi[j] = tq_c4_d32[tq_up2(blk->qs_hi, j)];
        tq_fwht_local<32>(hi);
        for (int j = 0; j < 32; j++) hi[j] *= norm_hi;

        // QJL correction on hi
        float corr_hi[32];
        for (int j = 0; j < 32; j++) corr_hi[j] = tq_sign_bit(blk->signs_hi, j);
        tq_fwht_local<32>(corr_hi);
        float qjl_s_hi = QJL_SCALE_32 * rnorm_hi;
        for (int j = 0; j < 32; j++) hi[j] += qjl_s_hi * corr_hi[j];

        // Decode lo: 1-bit centroid lookup, scale + QJL correction in rotated space, then unrotate
        float lo[96];
        for (int j = 0; j < 96; j++) lo[j] = tq_c2_d96[tq_up1(blk->qs_lo, j)] * norm_lo;
        float qjl_s_lo = QJL_SCALE_96 * rnorm_lo;
        for (int j = 0; j < 96; j++) {
            float sign = tq_sign_bit(blk->signs_lo, j);
            lo[j] += qjl_s_lo * sign;
        }
        tq_fwht_local<32>(lo);
        tq_fwht_local<32>(lo + 32);
        tq_fwht_local<32>(lo + 64);

        // Inverse-permute via channel map and write
        const int64_t base = block_in_row * 128;
        for (int j = 0; j < 32; j++) dst_row[base + perm[j]] = ggml_cuda_cast<dst_t>(hi[j]);
        for (int j = 0; j < 96; j++) dst_row[base + perm[32 + j]] = ggml_cuda_cast<dst_t>(lo[j]);
    }
}

// ---------------------------------------------------------------------------
// 3hi_2lo_had: 3-bit MSE + QJL on hi, 2-bit MSE + QJL on lo
// ---------------------------------------------------------------------------

template <typename dst_t>
static __global__ void k_get_rows_tq_3hi_2lo_had(
        const void * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int32_t * __restrict__ chmap, const int32_t n_kv_heads, const int32_t layer_idx,
        const int64_t ne00,
        const int64_t ne11, const int64_t ne12,
        const size_t s1, const size_t s2, const size_t s3,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12) {

    for (int64_t z = blockIdx.z; z < ne11 * ne12; z += gridDim.z) {
        const int64_t block_in_row = threadIdx.x;
        if (block_in_row >= ne00 / 128) return;

        const int i11 = z / ne12;
        const int i12 = z % ne12;
        const int i10 = blockIdx.x;
        const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

        dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
        const char * src0_row = (const char *)src0 + i01*nb01 + i11*nb02 + i12*nb03;
        const block_tqk_3hi_2lo * blk = (const block_tqk_3hi_2lo *)src0_row + block_in_row;

        const int head = (int)(block_in_row % n_kv_heads);
        const int32_t * perm = chmap + ((int64_t)layer_idx * n_kv_heads + head) * 128;

        float norm_hi  = __half2float(blk->norm_hi);
        float norm_lo  = __half2float(blk->norm_lo);
        float rnorm_hi = __half2float(blk->rnorm_hi);
        float rnorm_lo = __half2float(blk->rnorm_lo);

        // Decode hi: 3-bit centroid lookup → inverse FWHT → scale
        float hi[32];
        for (int j = 0; j < 32; j++) hi[j] = tq_c8_d32[tq_up3(blk->qs_hi, j)];
        tq_fwht_local<32>(hi);
        for (int j = 0; j < 32; j++) hi[j] *= norm_hi;

        // QJL correction on hi
        float corr_hi[32];
        for (int j = 0; j < 32; j++) corr_hi[j] = tq_sign_bit(blk->signs_hi, j);
        tq_fwht_local<32>(corr_hi);
        float qjl_s_hi = QJL_SCALE_32 * rnorm_hi;
        for (int j = 0; j < 32; j++) hi[j] += qjl_s_hi * corr_hi[j];

        // Decode lo: 2-bit centroid lookup, scale + QJL correction in rotated space, then unrotate
        float lo[96];
        for (int j = 0; j < 96; j++) lo[j] = tq_c4_d96[tq_up2(blk->qs_lo, j)] * norm_lo;
        float qjl_s_lo = QJL_SCALE_96 * rnorm_lo;
        for (int j = 0; j < 96; j++) {
            float sign = tq_sign_bit(blk->signs_lo, j);
            lo[j] += qjl_s_lo * sign;
        }
        tq_fwht_local<32>(lo);
        tq_fwht_local<32>(lo + 32);
        tq_fwht_local<32>(lo + 64);

        // Inverse-permute via channel map and write
        const int64_t base = block_in_row * 128;
        for (int j = 0; j < 32; j++) dst_row[base + perm[j]] = ggml_cuda_cast<dst_t>(hi[j]);
        for (int j = 0; j < 96; j++) dst_row[base + perm[32 + j]] = ggml_cuda_cast<dst_t>(lo[j]);
    }
}

// ===========================================================================
// d=256 TurboQuant get_rows kernels
// ===========================================================================

template <typename dst_t>
static __global__ void k_get_rows_tq_had_mse4_d256(
        const void * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int64_t ne00,
        const int64_t ne11, const int64_t ne12,
        const size_t s1, const size_t s2, const size_t s3,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12) {

    for (int64_t z = blockIdx.z; z < ne11 * ne12; z += gridDim.z) {
        const int64_t block_in_row = threadIdx.x;
        if (block_in_row >= ne00 / 256) return;
        const int i11 = z / ne12, i12 = z % ne12, i10 = blockIdx.x;
        const int i01 = src1[i10*s10 + i11*s11 + i12*s12];
        dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
        const block_tqk_had_mse4_d256 * blk = (const block_tqk_had_mse4_d256 *)
            ((const char *)src0 + i01*nb01 + i11*nb02 + i12*nb03) + block_in_row;
        float norm = __half2float(blk->norm);
        float rot[256];
        for (int j = 0; j < 256; j++) rot[j] = tq_c16_d256[tq_up4(blk->qs, j)];
        tq_fwht_local<256>(rot);
        const int64_t base = block_in_row * 256;
        for (int j = 0; j < 256; j++) dst_row[base + j] = ggml_cuda_cast<dst_t>(norm * rot[j]);
    }
}

template <typename dst_t>
static __global__ void k_get_rows_tq_had_prod5_d256(
        const void * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int64_t ne00,
        const int64_t ne11, const int64_t ne12,
        const size_t s1, const size_t s2, const size_t s3,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12) {

    for (int64_t z = blockIdx.z; z < ne11 * ne12; z += gridDim.z) {
        const int64_t block_in_row = threadIdx.x;
        if (block_in_row >= ne00 / 256) return;
        const int i11 = z / ne12, i12 = z % ne12, i10 = blockIdx.x;
        const int i01 = src1[i10*s10 + i11*s11 + i12*s12];
        dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
        const block_tqk_had_prod5_d256 * blk = (const block_tqk_had_prod5_d256 *)
            ((const char *)src0 + i01*nb01 + i11*nb02 + i12*nb03) + block_in_row;
        float norm = __half2float(blk->norm), rnorm = __half2float(blk->rnorm);
        float rot[256];
        for (int j = 0; j < 256; j++) rot[j] = tq_c16_d256[tq_up4(blk->qs, j)];
        tq_fwht_local<256>(rot);
        for (int j = 0; j < 256; j++) rot[j] *= norm;
        float corr[256];
        for (int j = 0; j < 256; j++) corr[j] = tq_sign_bit(blk->signs, j);
        tq_fwht_local<256>(corr);
        float qjl_scale = QJL_SCALE_256 * rnorm;
        for (int j = 0; j < 256; j++) rot[j] += qjl_scale * corr[j];
        const int64_t base = block_in_row * 256;
        for (int j = 0; j < 256; j++) dst_row[base + j] = ggml_cuda_cast<dst_t>(rot[j]);
    }
}

template <typename dst_t>
static __global__ void k_get_rows_tq_had_prod4_d256(
        const void * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int64_t ne00,
        const int64_t ne11, const int64_t ne12,
        const size_t s1, const size_t s2, const size_t s3,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12) {

    for (int64_t z = blockIdx.z; z < ne11 * ne12; z += gridDim.z) {
        const int64_t block_in_row = threadIdx.x;
        if (block_in_row >= ne00 / 256) return;
        const int i11 = z / ne12, i12 = z % ne12, i10 = blockIdx.x;
        const int i01 = src1[i10*s10 + i11*s11 + i12*s12];
        dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
        const block_tqk_had_prod4_d256 * blk = (const block_tqk_had_prod4_d256 *)
            ((const char *)src0 + i01*nb01 + i11*nb02 + i12*nb03) + block_in_row;
        float norm = __half2float(blk->norm), rnorm = __half2float(blk->rnorm);
        float rot[256];
        for (int j = 0; j < 256; j++) rot[j] = tq_c8_d256[tq_up3(blk->qs, j)];
        tq_fwht_local<256>(rot);
        for (int j = 0; j < 256; j++) rot[j] *= norm;
        float corr[256];
        for (int j = 0; j < 256; j++) corr[j] = tq_sign_bit(blk->signs, j);
        tq_fwht_local<256>(corr);
        float qjl_scale = QJL_SCALE_256 * rnorm;
        for (int j = 0; j < 256; j++) rot[j] += qjl_scale * corr[j];
        const int64_t base = block_in_row * 256;
        for (int j = 0; j < 256; j++) dst_row[base + j] = ggml_cuda_cast<dst_t>(rot[j]);
    }
}

template <typename dst_t>
static __global__ void k_get_rows_tq_5hi_3lo_had_d256(
        const void * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int32_t * __restrict__ chmap, const int32_t n_kv_heads, const int32_t layer_idx,
        const int64_t ne00,
        const int64_t ne11, const int64_t ne12,
        const size_t s1, const size_t s2, const size_t s3,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12) {

    for (int64_t z = blockIdx.z; z < ne11 * ne12; z += gridDim.z) {
        const int64_t block_in_row = threadIdx.x;
        if (block_in_row >= ne00 / 256) return;
        const int i11 = z / ne12, i12 = z % ne12, i10 = blockIdx.x;
        const int i01 = src1[i10*s10 + i11*s11 + i12*s12];
        dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
        const block_tqk_5hi_3lo_d256 * blk = (const block_tqk_5hi_3lo_d256 *)
            ((const char *)src0 + i01*nb01 + i11*nb02 + i12*nb03) + block_in_row;
        const int head = (int)(block_in_row % n_kv_heads);
        const int32_t * perm = chmap + ((int64_t)layer_idx * n_kv_heads + head) * 256;
        float norm_hi = __half2float(blk->norm_hi), norm_lo = __half2float(blk->norm_lo);
        float rnorm_hi = __half2float(blk->rnorm_hi);
        // Decode hi: 64 outliers, 4-bit + QJL (H64 transform)
        float hi[64];
        for (int j = 0; j < 64; j++) hi[j] = tq_c16_d64[tq_up4(blk->qs_hi, j)];
        tq_fwht_local<64>(hi);
        for (int j = 0; j < 64; j++) hi[j] *= norm_hi;
        float corr[64];
        for (int j = 0; j < 64; j++) corr[j] = tq_sign_bit(blk->signs_hi, j);
        tq_fwht_local<64>(corr);
        float qjl_scale = QJL_SCALE_64 * rnorm_hi;
        for (int j = 0; j < 64; j++) hi[j] += qjl_scale * corr[j];
        // Decode lo: 192 regular, 3-bit, shared 192-D norm → d192 centroids (3×H64)
        float lo[192];
        for (int j = 0; j < 192; j++) lo[j] = tq_c8_d192[tq_up3(blk->qs_lo, j)];
        for (int b = 0; b < 3; b++) tq_fwht_local<64>(lo + b * 64);
        for (int j = 0; j < 192; j++) lo[j] *= norm_lo;
        // Inverse-permute via channel map
        const int64_t base = block_in_row * 256;
        for (int j = 0; j < 64; j++)  dst_row[base + perm[j]] = ggml_cuda_cast<dst_t>(hi[j]);
        for (int j = 0; j < 192; j++) dst_row[base + perm[64 + j]] = ggml_cuda_cast<dst_t>(lo[j]);
    }
}

// ---------------------------------------------------------------------------
// d=256 non-split host dispatch (template-based, no layer_idx needed)
// ---------------------------------------------------------------------------

#define DEFINE_TQ_D256_GET_ROWS_DISPATCH(suffix, kernel, blk_sz) \
template <typename dst_t> \
void get_rows_tq_##suffix( \
        const void * src0_d, const int32_t * src1_d, dst_t * dst_d, \
        int64_t ne00, size_t nb01, size_t nb02, size_t nb03, \
        int64_t ne10, int64_t ne11, int64_t ne12, \
        size_t nb10, size_t nb11, size_t nb12, \
        size_t nb1, size_t nb2, size_t nb3, \
        cudaStream_t stream) { \
    const int64_t n_blocks = ne00 / blk_sz; \
    const dim3 block_dims(n_blocks, 1, 1); \
    const dim3 grid_dims(ne10, 1, MIN(ne11 * ne12, (int64_t)UINT16_MAX)); \
    kernel<<<grid_dims, block_dims, 0, stream>>>( \
        src0_d, src1_d, dst_d, ne00, ne11, ne12, \
        nb1/sizeof(dst_t), nb2/sizeof(dst_t), nb3/sizeof(dst_t), \
        nb01, nb02, nb03, nb10/sizeof(int32_t), nb11/sizeof(int32_t), nb12/sizeof(int32_t)); \
}

DEFINE_TQ_D256_GET_ROWS_DISPATCH(had_mse4_d256,  k_get_rows_tq_had_mse4_d256,  256)
DEFINE_TQ_D256_GET_ROWS_DISPATCH(had_prod5_d256, k_get_rows_tq_had_prod5_d256, 256)
DEFINE_TQ_D256_GET_ROWS_DISPATCH(had_prod4_d256, k_get_rows_tq_had_prod4_d256, 256)

#define INSTANTIATE_TQ_D256_GET_ROWS(func) \
    template void func<float>(const void*, const int32_t*, float*, int64_t, size_t, size_t, size_t, int64_t, int64_t, int64_t, size_t, size_t, size_t, size_t, size_t, size_t, cudaStream_t); \
    template void func<half>(const void*, const int32_t*, half*, int64_t, size_t, size_t, size_t, int64_t, int64_t, int64_t, size_t, size_t, size_t, size_t, size_t, size_t, cudaStream_t); \
    template void func<int32_t>(const void*, const int32_t*, int32_t*, int64_t, size_t, size_t, size_t, int64_t, int64_t, int64_t, size_t, size_t, size_t, size_t, size_t, size_t, cudaStream_t); \
    template void func<nv_bfloat16>(const void*, const int32_t*, nv_bfloat16*, int64_t, size_t, size_t, size_t, int64_t, int64_t, int64_t, size_t, size_t, size_t, size_t, size_t, size_t, cudaStream_t);

INSTANTIATE_TQ_D256_GET_ROWS(get_rows_tq_had_mse4_d256)
INSTANTIATE_TQ_D256_GET_ROWS(get_rows_tq_had_prod5_d256)
INSTANTIATE_TQ_D256_GET_ROWS(get_rows_tq_had_prod4_d256)

// d=256 split type dispatch is below after macro definition

// ---------------------------------------------------------------------------
// Host-side dispatch functions
// ---------------------------------------------------------------------------

template <typename dst_t>
void get_rows_tq_had_mse4(
        const void * src0_d, const int32_t * src1_d, dst_t * dst_d,
        int64_t ne00, size_t nb01, size_t nb02, size_t nb03,
        int64_t ne10, int64_t ne11, int64_t ne12,
        size_t nb10, size_t nb11, size_t nb12,
        size_t nb1, size_t nb2, size_t nb3,
        cudaStream_t stream) {

    const int64_t n_blocks = ne00 / 128;
    const dim3 block_dims(n_blocks, 1, 1);
    const dim3 grid_dims(ne10, 1, MIN(ne11 * ne12, (int64_t)UINT16_MAX));

    const size_t s1_ = nb1 / sizeof(dst_t);
    const size_t s2_ = nb2 / sizeof(dst_t);
    const size_t s3_ = nb3 / sizeof(dst_t);
    const size_t s10 = nb10 / sizeof(int32_t);
    const size_t s11 = nb11 / sizeof(int32_t);
    const size_t s12 = nb12 / sizeof(int32_t);

    k_get_rows_tq_had_mse4<<<grid_dims, block_dims, 0, stream>>>(
        src0_d, src1_d, dst_d, ne00, ne11, ne12, s1_, s2_, s3_, nb01, nb02, nb03, s10, s11, s12);
}

template <typename dst_t>
void get_rows_tq_had_prod5(
        const void * src0_d, const int32_t * src1_d, dst_t * dst_d,
        int64_t ne00, size_t nb01, size_t nb02, size_t nb03,
        int64_t ne10, int64_t ne11, int64_t ne12,
        size_t nb10, size_t nb11, size_t nb12,
        size_t nb1, size_t nb2, size_t nb3,
        cudaStream_t stream) {

    const int64_t n_blocks = ne00 / 128;
    const dim3 block_dims(n_blocks, 1, 1);
    const dim3 grid_dims(ne10, 1, MIN(ne11 * ne12, (int64_t)UINT16_MAX));

    const size_t s1_ = nb1 / sizeof(dst_t);
    const size_t s2_ = nb2 / sizeof(dst_t);
    const size_t s3_ = nb3 / sizeof(dst_t);
    const size_t s10 = nb10 / sizeof(int32_t);
    const size_t s11 = nb11 / sizeof(int32_t);
    const size_t s12 = nb12 / sizeof(int32_t);

    k_get_rows_tq_had_prod5<<<grid_dims, block_dims, 0, stream>>>(
        src0_d, src1_d, dst_d, ne00, ne11, ne12, s1_, s2_, s3_, nb01, nb02, nb03, s10, s11, s12);
}

template <typename dst_t>
void get_rows_tq_had_prod4(
        const void * src0_d, const int32_t * src1_d, dst_t * dst_d,
        int64_t ne00, size_t nb01, size_t nb02, size_t nb03,
        int64_t ne10, int64_t ne11, int64_t ne12,
        size_t nb10, size_t nb11, size_t nb12,
        size_t nb1, size_t nb2, size_t nb3,
        cudaStream_t stream) {

    const int64_t n_blocks = ne00 / 128;
    const dim3 block_dims(n_blocks, 1, 1);
    const dim3 grid_dims(ne10, 1, MIN(ne11 * ne12, (int64_t)UINT16_MAX));

    const size_t s1_ = nb1 / sizeof(dst_t);
    const size_t s2_ = nb2 / sizeof(dst_t);
    const size_t s3_ = nb3 / sizeof(dst_t);
    const size_t s10 = nb10 / sizeof(int32_t);
    const size_t s11 = nb11 / sizeof(int32_t);
    const size_t s12 = nb12 / sizeof(int32_t);

    k_get_rows_tq_had_prod4<<<grid_dims, block_dims, 0, stream>>>(
        src0_d, src1_d, dst_d, ne00, ne11, ne12, s1_, s2_, s3_, nb01, nb02, nb03, s10, s11, s12);
}

// ---------------------------------------------------------------------------
// Split-type get_rows — tensor-level dispatch (extracts layer_idx from name)
// ---------------------------------------------------------------------------

// Helper: extract layer index from KV cache tensor name (e.g. "cache_k_l5")
static int32_t tq_extract_layer_idx(const ggml_tensor * src0) {
    // Walk through view chain to find root tensor with the name
    const ggml_tensor * root = src0;
    while (root->view_src) root = root->view_src;
    const char * lp = strstr(root->name, "_l");
    return lp ? (int32_t)atoi(lp + 2) : 0;
}

#define DEFINE_TQ_SPLIT_GET_ROWS_OP(suffix, kernel_name, blk_sz) \
void ggml_cuda_op_get_rows_tq_##suffix(ggml_backend_cuda_context & ctx, ggml_tensor * dst) { \
    const ggml_tensor * src0 = dst->src[0]; \
    const ggml_tensor * src1 = dst->src[1]; \
    GGML_ASSERT(src1->type == GGML_TYPE_I32); \
    GGML_TENSOR_BINARY_OP_LOCALS \
    cudaStream_t stream = ctx.stream(); \
    int32_t layer_idx = tq_extract_layer_idx(src0); \
    int32_t * chmap = ggml_cuda_get_tq_channel_map_device(); \
    int n_kv_heads = ggml_cuda_get_tq_chmap_n_heads(); \
    if (n_kv_heads < 1) n_kv_heads = (int)(ne00 / (blk_sz)); \
    const int64_t n_blocks = ne00 / (blk_sz); \
    const dim3 block_dims(n_blocks, 1, 1); \
    const dim3 grid_dims(ne10, 1, MIN(ne11 * ne12, (int64_t)UINT16_MAX)); \
    const size_t s1_ = nb1 / ggml_type_size(dst->type); \
    const size_t s2_ = nb2 / ggml_type_size(dst->type); \
    const size_t s3_ = nb3 / ggml_type_size(dst->type); \
    const size_t s10 = nb10 / sizeof(int32_t); \
    const size_t s11 = nb11 / sizeof(int32_t); \
    const size_t s12 = nb12 / sizeof(int32_t); \
    if (dst->type == GGML_TYPE_F32) { \
        kernel_name<<<grid_dims, block_dims, 0, stream>>>( \
            src0->data, (const int32_t *)src1->data, (float *)dst->data, chmap, n_kv_heads, layer_idx, \
            ne00, ne11, ne12, s1_, s2_, s3_, nb01, nb02, nb03, s10, s11, s12); \
    } else if (dst->type == GGML_TYPE_F16) { \
        kernel_name<<<grid_dims, block_dims, 0, stream>>>( \
            src0->data, (const int32_t *)src1->data, (half *)dst->data, chmap, n_kv_heads, layer_idx, \
            ne00, ne11, ne12, s1_, s2_, s3_, nb01, nb02, nb03, s10, s11, s12); \
    } else { \
        GGML_ABORT("unsupported dst type for TQ split get_rows"); \
    } \
}

DEFINE_TQ_SPLIT_GET_ROWS_OP(5hi_3lo_had,      k_get_rows_tq_5hi_3lo_had,      128)
DEFINE_TQ_SPLIT_GET_ROWS_OP(6hi_3lo_had,      k_get_rows_tq_6hi_3lo_had,      128)
DEFINE_TQ_SPLIT_GET_ROWS_OP(2hi_1lo_had,      k_get_rows_tq_2hi_1lo_had,      128)
DEFINE_TQ_SPLIT_GET_ROWS_OP(3hi_2lo_had,      k_get_rows_tq_3hi_2lo_had,      128)
DEFINE_TQ_SPLIT_GET_ROWS_OP(5hi_3lo_had_d256, k_get_rows_tq_5hi_3lo_had_d256, 256)

// Explicit template instantiations for non-split types (still use template dispatch)
#define INSTANTIATE_TQ_GET_ROWS(func) \
    template void func<float>(const void*, const int32_t*, float*, int64_t, size_t, size_t, size_t, int64_t, int64_t, int64_t, size_t, size_t, size_t, size_t, size_t, size_t, cudaStream_t); \
    template void func<half>(const void*, const int32_t*, half*, int64_t, size_t, size_t, size_t, int64_t, int64_t, int64_t, size_t, size_t, size_t, size_t, size_t, size_t, cudaStream_t); \
    template void func<int32_t>(const void*, const int32_t*, int32_t*, int64_t, size_t, size_t, size_t, int64_t, int64_t, int64_t, size_t, size_t, size_t, size_t, size_t, size_t, cudaStream_t); \
    template void func<nv_bfloat16>(const void*, const int32_t*, nv_bfloat16*, int64_t, size_t, size_t, size_t, int64_t, int64_t, int64_t, size_t, size_t, size_t, size_t, size_t, size_t, cudaStream_t);

INSTANTIATE_TQ_GET_ROWS(get_rows_tq_had_mse4)
INSTANTIATE_TQ_GET_ROWS(get_rows_tq_had_prod5)
INSTANTIATE_TQ_GET_ROWS(get_rows_tq_had_prod4)
