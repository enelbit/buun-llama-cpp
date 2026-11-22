#pragma once

#include "common.cuh"
#include <math.h>
#include <stdint.h>

// ============================================================================
// Split attention micro-kernels (port from ref-kvsplit, adapted for CUDA sm_61)
// ============================================================================

// 1. Extract LSE from dst_meta (float2: {KQ_max, KQ_sum}) → lse = max + log(sum)
static __global__ void fattn_extract_lse(const float2 * __restrict__ meta, float * __restrict__ lse, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float2 m = meta[i];
        lse[i] = (m.y <= 0.0f) ? -INFINITY : m.x + logf(m.y);
    }
}

// 2. Copy Q from VRAM → pinned host memory (split->q)
//    Layout: Q[i * nb1 + h * nb2 + t] → q[(h * n_tok + i) * D + t]
static __global__ void kv_split_q(const char * __restrict__ Q, const int64_t nb1, const int64_t nb2,
                                  float * __restrict__ q, const int n_tok, const int n_head, const int D) {
    const int i = blockIdx.x; // token
    if (i >= n_tok) return;
    for (int h = blockIdx.y; h < n_head; h += gridDim.y) {
        const float * q_row = (const float *)(Q + (int64_t)i * nb1 + (int64_t)h * nb2);
        float * dst_row = q + (((int64_t)h * n_tok) + i) * D;
        for (int t = threadIdx.x; t < D; t += blockDim.x) {
            dst_row[t] = q_row[t];
        }
    }
    __threadfence_system();
}

// 3. Signal CPU: q_ready = tag (1 thread)
static __global__ void kv_split_signal(const int32_t * __restrict__ tag, uint32_t * __restrict__ q_ready) {
    __threadfence_system();
    atomicExch(q_ready, (uint32_t)tag[1]);
    __threadfence_system();
}

// 4. Wait for CPU: out_ready >= tag (1 thread, spin with timeout)
static __global__ void kv_split_wait(const int32_t * __restrict__ tag, const uint32_t * __restrict__ out_ready) {
    const uint32_t t = (uint32_t)tag[1];
    const uint64_t t0 = (uint64_t)clock64();
    uint32_t * ready_mut = const_cast<uint32_t *>(out_ready);
    while (atomicAdd(ready_mut, 0u) < t) {
        if ((uint64_t)clock64() - t0 > 2000000000ULL) {
            __trap();
        }
    }
    __threadfence_system();
}

// 5. Merge: dst = merge(dst_gpu, lse_gpu, out_cpu, lse_cpu) via log-sum-exp
//    Grid: (n_tok, n_head), Block: D threads
static __global__ void kv_split_merge(char * __restrict__ dst, const int64_t nb1, const int64_t nb2,
                                      float * __restrict__ lse,
                                      const float * __restrict__ out, const float * __restrict__ out_lse,
                                      const int n_tok, const int n_head, const int D) {
    const int i = blockIdx.x; // token
    if (i >= n_tok) return;
    for (int h = blockIdx.y; h < n_head; h += gridDim.y) {
        const int64_t row = (int64_t)i * n_head + h;
        float * acc = (float *)(dst + (int64_t)i * nb2 + (int64_t)h * nb1);

        const float L = lse[row];     // GPU LSE
        const float l = out_lse[row]; // CPU LSE
        const float m = fmaxf(L, l);

        if (m == -INFINITY) {
            for (int t = threadIdx.x; t < D; t += blockDim.x) {
                acc[t] = 0.0f;
            }
            continue;
        }

        const float wa = (L == -INFINITY) ? 0.0f : expf(L - m);
        const float wb = (l == -INFINITY) ? 0.0f : expf(l - m);
        const float denom = wa + wb;

        for (int t = threadIdx.x; t < D; t += blockDim.x) {
            acc[t] = ((wa > 0.0f ? wa * acc[t] : 0.0f) + (wb > 0.0f ? wb * out[row * D + t] : 0.0f)) / denom;
        }

        if (threadIdx.x == 0) {
            lse[row] = m + logf(denom);
        }
    }
}
