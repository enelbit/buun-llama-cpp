#pragma once
#include <cublasLt.h>
#include <cstdint>

extern "C" {
    bool algo_attribute(const cublasLtMatmulAlgo_t& algo, cublasLtMatmulAlgoConfigAttributes_t attr, int* value);
    [[noreturn]] uint16_t bf16_bits(uint8_t x);
}
