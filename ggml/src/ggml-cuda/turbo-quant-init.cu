// TurboQuant rotation matrix initialization (this TU's copy)
//
// Each TU that includes turbo-quant-cuda.cuh gets its own static __device__
// rotation matrices, lazily initialized on first use via
// tq_device_init_rotations_kernel. This file exists so that turbo-quant-init.cu
// is part of the build and its copy of the kernel is compiled.

#include "common.cuh"
#include "turbo-quant-cuda.cuh"

// Lazy init for this TU (if it's ever used directly)
static void tq_ensure_rotations_init(cudaStream_t stream) {
    static bool initialized = false;
    if (!initialized) {
        tq_device_init_rotations_kernel<<<1, 1, 0, stream>>>();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
        initialized = true;
    }
    (void)stream;
}

// Keep this TU non-empty so it links properly
void tq_cuda_init_placeholder() {
    (void)tq_ensure_rotations_init;
}
