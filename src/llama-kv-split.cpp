#include "llama-kv-split.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstring>

static inline size_t align_up_256(size_t n) {
    return (n + 255u) & ~size_t(255u);
}

llama_kv_split::~llama_kv_split() {
    stop_worker();
    delete[] split_desc;
    split_desc = nullptr;
}

void llama_kv_split::init(uint32_t n_layer_, uint32_t max_tok_, uint32_t n_threads_, void * pinned_base, size_t pinned_size) {
    n_layer   = n_layer_;
    max_tok   = max_tok_;
    n_threads = std::max(1u, n_threads_);
    layers.assign(n_layer, {});

    delete[] split_desc;
    split_desc  = nullptr;
    split_q     = nullptr;
    split_out   = nullptr;
    split_lse   = nullptr;
    split_flags = nullptr;

    if (n_layer == 0 || max_tok == 0 || pinned_base == nullptr || pinned_size == 0) {
        return;
    }

    split_desc = new ggml_fa_split[n_layer];
    std::memset(split_desc, 0, sizeof(ggml_fa_split) * n_layer);

    const size_t flg_bytes = align_up_256((size_t)2 * n_layer * sizeof(uint32_t));
    if (flg_bytes + 3 * 256u > pinned_size) {
        delete[] split_desc;
        split_desc = nullptr;
        return;
    }

    const size_t usable_bytes = pinned_size - flg_bytes;
    // Reparto 16:16:1 entre split_q, split_out y split_lse (alineado a 256 bytes)
    const size_t q_bytes   = align_up_256((usable_bytes * 16u) / 33u);
    const size_t out_bytes = align_up_256((usable_bytes * 16u) / 33u);
    const size_t lse_bytes = align_up_256(usable_bytes > (q_bytes + out_bytes) ? (usable_bytes - q_bytes - out_bytes) : 256u);

    if (q_bytes + out_bytes + lse_bytes + flg_bytes > pinned_size) {
        delete[] split_desc;
        split_desc = nullptr;
        return;
    }

    auto * p = static_cast<uint8_t *>(pinned_base);
    split_q     = reinterpret_cast<float *>(p);
    split_out   = reinterpret_cast<float *>(p + q_bytes);
    split_lse   = reinterpret_cast<float *>(p + q_bytes + out_bytes);
    split_flags = reinterpret_cast<uint32_t *>(p + q_bytes + out_bytes + lse_bytes);
    std::memset(split_flags, 0, flg_bytes);
}

void llama_kv_split::set_layer(int32_t ikv, ggml_tensor * k, ggml_tensor * v, uint32_t n_head, uint32_t head_dim, float scale) {
    if (ikv < 0 || (uint32_t)ikv >= n_layer) {
        return;
    }
    layers[ikv] = { k, v, n_head, head_dim, scale };

    if (split_desc == nullptr) {
        return;
    }

    const size_t qv_stride  = (size_t)max_tok * n_head * head_dim;
    const size_t lse_stride = (size_t)max_tok * n_head;

    split_desc[ikv].q         = split_q     ? (split_q     + (size_t)ikv * qv_stride)  : nullptr;
    split_desc[ikv].q_ready   = split_flags ? (split_flags + (size_t)ikv)              : nullptr;
    split_desc[ikv].out       = split_out   ? (split_out   + (size_t)ikv * qv_stride)  : nullptr;
    split_desc[ikv].lse       = split_lse   ? (split_lse   + (size_t)ikv * lse_stride) : nullptr;
    split_desc[ikv].out_ready = split_flags ? (split_flags + (size_t)n_layer + ikv)    : nullptr;
    split_desc[ikv].scale     = scale;
}

void llama_kv_split::start() {
    if (worker.joinable()) return;
    worker = std::thread([this]() { worker_main(); });
}

void llama_kv_split::stop_worker() {
    stop.store(true, std::memory_order_release);
    cv.notify_all();
    if (worker.joinable()) {
        worker.join();
    }
    if (split_pool) {
        
        split_pool = nullptr;
    }
}

void llama_kv_split::submit(uint32_t n_rows, uint32_t n_tok) {
    tag++;
    {
        std::lock_guard<std::mutex> lock(mtx);
        jobs.push_back({ tag, n_rows, n_tok });
    }
    cv.notify_one();
}

void llama_kv_split::worker_main() {
    
    

    const bool debug = getenv("LLAMA_KV_SPLIT_DEBUG") != nullptr;
    const bool noop  = getenv("LLAMA_KV_SPLIT_NOOP") != nullptr;
    if (noop) {
        fprintf(stderr, "llama_kv_split: NOOP mode (results wrong, measuring only)\n");
    }

    const int32_t n_layer = (int32_t) layers.size();
    int64_t t_wait = 0, t_attend = 0, n_done = 0;

    for (;;) {
        job j;
        {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [&]() { return stop.load(std::memory_order_acquire) || !jobs.empty(); });
            if (stop.load(std::memory_order_acquire)) return;
            j = jobs.front();
            jobs.pop_front();
        }

        for (int32_t ikv = 0; ikv < n_layer; ++ikv) {
            const int64_t t0 = ggml_time_us();

            // Wait for GPU to publish Q
            for (uint32_t i = 0; __atomic_load_n(split_flags + ikv, __ATOMIC_ACQUIRE) < j.tag; ++i) {
                if (stop.load(std::memory_order_acquire)) return;
                std::this_thread::yield();
                if (i % 4096 == 0) {
                    // timeout check (simplified)
                }
            }
            const int64_t t1 = ggml_time_us();

            if (noop) {
                const auto & li = layers[ikv];
                float * lse = split_lse + (size_t)ikv * max_tok * li.n_head;
                std::fill(lse, lse + (size_t)j.n_tok * li.n_head, -INFINITY);
            } else {
                split_layer(ikv, j.n_rows, j.n_tok);
            }

            // Signal GPU that CPU is done
            __atomic_store_n(split_flags + n_layer + ikv, j.tag, __ATOMIC_RELEASE);

            t_wait   += ikv > 0 ? t1 - t0 : 0;
            t_attend += ggml_time_us() - t1;
        }

        if (debug && ++n_done % 32 == 0) {
            fprintf(stderr, "llama_kv_split: %u rows, %u tok: waited %.1f ms, attended %.1f ms\n",
                    j.n_rows, j.n_tok, t_wait/32e3, t_attend/32e3);
            t_wait = t_attend = 0;
        }
    }
}

void llama_kv_split::split_layer(int32_t ikv, uint32_t n_rows, uint32_t n_tok) {
    const auto & li = layers[ikv];
    ggml_tensor * K = li.k;
    ggml_tensor * V = li.v;

    ggml_init_params params = {
        /*.mem_size   =*/ 32*ggml_tensor_overhead() + ggml_graph_overhead_custom(16, false),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);

    // Q: from pinned buffer (written by GPU before signaling q_ready)
    ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, li.head_dim, n_tok, li.n_head);
    q->data = split_q + (size_t)ikv * max_tok * li.n_head * li.head_dim;

    // K/V: view of the host tensor, limited to n_rows tokens
    // K layout after permute: [n_embd_head_k, n_kv, n_head_kv, n_stream]
    // We want: [n_embd_head_k, n_rows, n_head_kv, 1]
    ggml_tensor * k = ggml_view_4d(ctx, K, K->ne[0], n_rows, K->ne[2], 1,
            K->nb[1], K->nb[2], K->nb[3], 0);
    ggml_tensor * v = ggml_view_4d(ctx, V, V->ne[0], n_rows, V->ne[2], 1,
            V->nb[1], V->nb[2], V->nb[3], 0);

    // Flash attention
    ggml_tensor * fa = ggml_flash_attn_ext(ctx, q, k, v, nullptr, li.scale, 0.0f, 0.0f);
    fa->data = split_out + (size_t)ikv * max_tok * li.n_head * li.head_dim;
    ggml_flash_attn_ext_set_lse(fa, split_lse + (size_t)ikv * max_tok * li.n_head);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(gf, fa);

    ggml_cplan plan = ggml_graph_plan(gf, n_threads, nullptr);
    if (plan.work_size > split_work.size()) {
        split_work.resize(plan.work_size);
    }
    plan.work_data = (uint8_t *)split_work.data();
    ggml_graph_compute(gf, &plan);

    ggml_free(ctx);
}
