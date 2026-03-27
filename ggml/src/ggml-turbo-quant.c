// TurboQuant CPU quantize/dequantize stubs
// TODO(TurboQuant): Replace with actual PolarQuant + QJL implementation

#include "ggml-quants.h"

#include <stdio.h>
#include <stdlib.h>

void quantize_row_turbo3_0_ref(const float * GGML_RESTRICT x, block_turbo3_0 * GGML_RESTRICT y, int64_t k) {
    (void)x; (void)y; (void)k;
    fprintf(stderr, "%s: TurboQuant 3-bit quantization not yet implemented\n", __func__);
    abort();
}

void dequantize_row_turbo3_0(const block_turbo3_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    (void)x; (void)y; (void)k;
    fprintf(stderr, "%s: TurboQuant 3-bit dequantization not yet implemented\n", __func__);
    abort();
}

void quantize_row_turbo4_0_ref(const float * GGML_RESTRICT x, block_turbo4_0 * GGML_RESTRICT y, int64_t k) {
    (void)x; (void)y; (void)k;
    fprintf(stderr, "%s: TurboQuant 4-bit quantization not yet implemented\n", __func__);
    abort();
}

void dequantize_row_turbo4_0(const block_turbo4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    (void)x; (void)y; (void)k;
    fprintf(stderr, "%s: TurboQuant 4-bit dequantization not yet implemented\n", __func__);
    abort();
}
