#pragma once
#include <cstdint>
#include <cstddef>
struct ggml_tensor;
struct ggml_backend_cuda_context;

extern "C" {
    int exl3_int8_mode();
    bool exl3_int8_applicable(int, int, int, int, int);
    bool exl3_mul_mat_id_fast_shape(const ggml_tensor*);
    int* exl3_int8_counters(ggml_backend_cuda_context&);
    void exl3_int8_geometry(int, int, int, int, int, int, size_t, int&, int&, size_t&);
    size_t exl3_int8_smem_cap(int);
}
