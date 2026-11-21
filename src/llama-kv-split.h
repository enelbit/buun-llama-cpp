#pragma once

#include <cstdint>
#include <cstddef>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <vector>
#include <atomic>
#include "ggml.h"

// CPU worker for split attention: attends over K/V rows that are in host memory
// (beyond the GPU ring capacity), synchronized with the GPU via pinned flags.
struct llama_kv_split {
    struct layer_info {
        ggml_tensor * k = nullptr;  // permuted K tensor (from graph)
        ggml_tensor * v = nullptr;  // permuted V tensor (from graph)
        uint32_t      n_head  = 0;  // query heads
        uint32_t      head_dim = 0; // d
        float         scale   = 0.0f;
    };

    // Pinned buffers shared with GPU (allocated via cudaHostAlloc)
    float    * split_q     = nullptr;  // [n_layer * max_tok * n_head * head_dim]
    float    * split_out   = nullptr;  // [n_layer * max_tok * n_head * head_dim]
    float    * split_lse   = nullptr;  // [n_layer * max_tok * n_head]
    uint32_t * split_flags = nullptr;  // [2 * n_layer]: q_ready[0..n_layer-1], out_ready[n_layer..2*n_layer-1]

    // Config
    uint32_t n_layer = 0;
    uint32_t max_tok = 0;
    uint32_t n_threads = 0;
    std::vector<layer_info> layers;

    // Worker state
    std::thread worker;
    std::mutex mtx;
    std::condition_variable cv;
    struct job {
        uint32_t tag;
        uint32_t n_rows;
        uint32_t n_tok;
    };
    std::deque<job> jobs;
    std::atomic<bool> stop{false};
    uint32_t tag = 0;

    // CPU compute resources
    std::vector<float> split_work;
    void * split_pool = nullptr;  // ggml_threadpool

    // ggml_fa_split descriptors (one per layer, passed to set_split)
    struct ggml_fa_split * split_desc = nullptr;  // [n_layer]

    void init(uint32_t n_layer_, uint32_t max_tok_, uint32_t n_threads_, void * pinned_base, size_t pinned_size);
    void set_layer(int32_t ikv, ggml_tensor * k, ggml_tensor * v, uint32_t n_head, uint32_t head_dim, float scale);
    void start();
    void stop_worker();
    void submit(uint32_t n_rows, uint32_t n_tok);
    void worker_main();
    void split_layer(int32_t ikv, uint32_t n_rows, uint32_t n_tok);
    ~llama_kv_split();
};
