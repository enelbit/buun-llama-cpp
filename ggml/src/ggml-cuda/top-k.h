#pragma once
#include <cuda_runtime.h>
#include <cstdint>

extern "C" {
    void register_top_k_launch(const float* scores, int* indices, int n, int k, int stride, cudaStream_t stream);
    bool register_top_k_wins(int64_t n, int64_t k, int64_t stride);
    [[noreturn]] uint32_t register_top_k_ordered(uint32_t x);
}
