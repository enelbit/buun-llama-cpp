# Command & Configuration Guide: Dense Models vs. MoE Models

This guide documents all working CLI flags and `models.ini` configurations in `kv-stream-pascal`, separating **Dense / Hybrid-Dense models** from **Mixture-of-Experts (MoE) models** so users do not combine conflicting options.

---

## 1. Critical Flag Interaction Rules

1. `--fit on --fit-budget <MiB>` vs `--fit off -ngl 99`:
   - When pinning layers manually with `-ot` (`--override-tensor`) or `-ngl 99`, set `--fit off` (`fit = off`). Leaving `--fit on` without `--fit-budget` can override manual placement or abort when `n_gpu_layers` is fixed.
   - When combining `--fit on --fit-budget <MiB>` with `--kv-stream-stage-mib <MiB>`, the allocator automatically deducts `kv_stream_stage_mib + 32 MiB` before fitting weights and context.
2. `--no-kv-offload` (`-nkvo`) + `--kv-stream-stage-mib` vs Dynamic VBR (`-ct vbr`):
   - Dynamic VBR (`-ct vbr` / implicit VBR) requires GPU-resident KV pools (`offload_kqv = true`).
   - When `-nkvo --kv-stream-stage-mib <MiB>` is active, KV pages live in pinned host RAM and stream through a VRAM ring buffer; use concrete KV types (`q8_0`, `turbo8`, `turbo4`, `f16`).
3. `--override-tensor` (`-ot`) vs `--n-cpu-moe` (`-ncmoe`) / `--cpu-moe` (`-cmoe`):
   - Overrides are matched in order of appearance (first regex match wins). Never use `--n-cpu-moe` on a Dense model (it only matches `ffn_.*_exps`). On MoE models, place specific `-ot` rules before `-ncmoe`.
4. `--moe-cache` vs `--no-repack`:
   - `--moe-cache` is only for MoE models on `sm_70+` (Turing/Ampere/Ada/Hopper) or RDNA3/4 (`gfx1100+`). On Pascal (`sm_61`), GPU expert caching is disabled while CPU AVX2/FMA weight repacking remains active unless `--no-repack` is explicitly passed.
5. `--reasoning auto` vs `--chat-template-kwargs`:
   - Use `--reasoning auto` (with optional `--reasoning-budget N`) instead of passing `enable_thinking` in `--chat-template-kwargs`. In shell CLI, always wrap JSON in single quotes: `'{"key":true}'`.

---

## 2. Section A — Dense & Hybrid-Dense Models (4B, 8B, 9B, 27B)

### What to Use vs. Avoid on Dense Models
- USE: Full GPU offload (`--fit off -ngl 99`) when weights + KV fit in VRAM, or partial FFN CPU offload (`-ot "blk\.([0-N])\.ffn_(gate|up|down)\.weight=CPU"`) when weights exceed VRAM (e.g., 9B or 27B on 6 GB VRAM).
- USE: `-nkvo --kv-stream-stage-mib 400` whenever context length (131K–262K) exceeds remaining VRAM.
- USE: `--spec-type draft-mtp` on native MTP models (`Qwen3.5-4B-MTP`, `Qwen3.5-9B-MTP`); `create_mtp_context()` automatically caps draft `n_ctx <= 32768`, bounds `n_batch`/`n_ubatch`, and disables duplicate KV staging.
- AVOID: Do NOT pass `--moe-cache`, `--n-cpu-moe`, `--cpu-moe`, or `ffn_.*_exps` regexes (dense models have no routed expert tensors).

### A1. Terminal (CLI) Examples — Dense Models

    # Small Dense + MTP in Full VRAM (131K context, ~37 t/s)
    ./build-pascal/bin/llama-server \
      --model /path/to/Qwen3.5-4B-MTP-Q5_K_M.gguf \
      --mmproj /path/to/Qwen3.5-4B-mmproj-Q8_0.gguf \
      --fit off --n-gpu-layers 99 --no-mmproj-offload \
      --flash-attn on --cache-type-k turbo4 --cache-type-v turbo4 \
      --ctx-size 131072 --parallel 1 --no-kv-unified \
      --batch-size 1024 --ubatch-size 256 --threads 6 --load-mode none \
      --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.85 \
      --reasoning auto --reasoning-budget 4096 \
      --port 18999

    # Small Dense + MTP with kv-stream (262K context under 5.2 GB VRAM)
    ./build-pascal/bin/llama-server \
      --model /path/to/Qwen3.5-4B-MTP-Q5_K_S.gguf \
      --mmproj /path/to/Qwen3.5-4B-mmproj-Q8_0.gguf \
      --fit off --n-gpu-layers 99 --no-mmproj-offload \
      --no-kv-offload --kv-stream-stage-mib 400 --no-kv-unified \
      --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
      --ctx-size 262144 --batch-size 2048 --ubatch-size 512 --parallel 1 \
      --threads 6 --load-mode none \
      --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.80 \
      --reasoning auto --reasoning-budget 4096 \
      --port 18999

    # Medium Dense (9B) with Partial FFN Offload + kv-stream
    ./build-pascal/bin/llama-server \
      --model /path/to/Qwen3.5-9B-Q5_K_S.gguf \
      --mmproj /path/to/mmproj-Qwen3.5-9B-Q8_0.gguf \
      --fit off --n-gpu-layers 99 \
      --override-tensor "blk\.([0-9]|1[0-6])\.ffn_(gate|up|down)\.weight=CPU" \
      --no-mmproj-offload --load-mode none \
      --no-kv-offload --kv-stream-stage-mib 400 --no-kv-unified \
      --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
      --ctx-size 131072 --batch-size 1024 --ubatch-size 256 --parallel 1 --threads 6 \
      --reasoning auto --reasoning-budget 4096 \
      --port 18999

### A2. models.ini Examples — Dense Models

    [qwen3.5-4b-mtp-q5-ks]
    model = /path/to/Qwen3.5-4B-MTP-Q5_K_S.gguf
    mmproj = /path/to/Qwen3.5-4B-mmproj-Q8_0.gguf
    alias = qwen3.5-4b-mtp-q5-ks
    fit = off
    n-gpu-layers = 99
    no-mmproj-offload = true
    no-kv-offload = true
    kv-stream-stage-mib = 400
    cache-type-k = q8_0
    cache-type-v = q8_0
    ctx-size = 262144
    batch-size = 2048
    ubatch-size = 512
    threads = 6
    load-mode = none
    spec-type = draft-mtp
    spec-draft-n-max = 2
    spec-draft-p-min = 0.85
    reasoning = auto
    reasoning-budget = 4096
    reasoning-budget-message = "Concluye tu razonamiento y proporciona la respuesta final. Responde ahora."

    [qwen3.5-9b-instruct-q5-ks]
    model = /path/to/Qwen3.5-9B-Q5_K_S.gguf
    mmproj = /path/to/mmproj-Qwen3.5-9B-Q8_0.gguf
    alias = qwen3.5-9b-instruct-q5-ks
    fit = off
    n-gpu-layers = 99
    override-tensor = blk\.([0-9]|1[0-6])\.ffn_(gate|up|down)\.weight=CPU
    no-mmproj-offload = true
    no-kv-offload = true
    kv-stream-stage-mib = 400
    load-mode = none
    ctx-size = 131072
    ubatch-size = 256
    reasoning = auto
    reasoning-budget = 4096
    reasoning-budget-message = "Concluye tu razonamiento y proporciona la respuesta final. Responde ahora."

---

## 3. Section B — Mixture-of-Experts (MoE) Models (Qwen3.6-35B-A3B, DeepSeek-V4, Qwen3.8-Flash)

### What to Use vs. Avoid on MoE Models
- USE: Selective routed-expert placement (`-ot "blk\.2\.ffn_(up|down)_exps\.weight=CPU,blk\.([3-9]|[1-3][0-9]|40)\.ffn_.*_exps\.weight=CPU"`) or `--n-cpu-moe N` / `--cpu-moe` so attention, shared experts, and hot early routed experts stay in VRAM while cold routed experts run on CPU with AVX2/FMA repacking.
- USE: `--load-mode none` when placing experts on CPU so the loader repacks weights directly into aligned RAM without `mmap` page-fault stalls.
- USE: `-nkvo --kv-stream-stage-mib 400` (or `64`–`256` to trigger CPU+GPU split attention on ultra-constrained VRAM) for 262K context.
- USE: `--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.85` for native MTP MoE models.
- AVOID: Do NOT use dense FFN regexes (`ffn_(gate|up|down)\.weight=CPU` without `_exps`), which evict shared/dense layers instead of routed experts.
- AVOID: Do NOT pass `--no-repack` on Pascal (`sm_61`) unless benchmarking canonical CPU weights, as repacking improves CPU expert throughput.

### B1. Terminal (CLI) Examples — MoE Models

    # Qwen3.6-35B-A3B-MTP (262K context on 6 GB VRAM, ~16.5 t/s sustained)
    ./build-pascal/bin/llama-server \
      --model /path/to/Qwen3.6-35B-A3B-MTP-UD-Q5_K_S.gguf \
      --mmproj /path/to/mmproj-Q8_0.gguf \
      --fit off --n-gpu-layers 99 --no-mmproj-offload \
      --override-tensor "blk\.2\.ffn_(up|down)_exps\.weight=CPU,blk\.([3-9]|[1-3][0-9]|40)\.ffn_.*_exps\.weight=CPU" \
      --no-kv-offload --kv-stream-stage-mib 400 --no-kv-unified \
      --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
      --ctx-size 262144 --batch-size 1024 --ubatch-size 512 --parallel 1 \
      --threads 6 --load-mode none \
      --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.85 \
      --reasoning auto --reasoning-budget 4096 \
      --port 18999

    # Adaptive MoE Placement (--fit on --fit-budget 5150 --kv-stream-stage-mib 400)
    ./build-pascal/bin/llama-server \
      --model /path/to/Qwen3.6-35B-A3B-MTP-UD-Q5_K_S.gguf \
      --fit on --fit-budget 5150 \
      --cpu-moe --no-kv-offload --kv-stream-stage-mib 400 \
      --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
      --ctx-size 131072 --parallel 1 --threads 6 --load-mode none \
      --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.85 \
      --port 18999

### B2. models.ini Examples — MoE Models

    [qwen3.6-35b-a3b-q5ks]
    model = /path/to/Qwen3.6-35B-A3B-MTP-UD-Q5_K_S.gguf
    mmproj = /path/to/mmproj-Q8_0.gguf
    alias = qwen3.6-35b-a3b-q5ks
    fit = off
    n-gpu-layers = 99
    no-mmproj-offload = true
    override-tensor = blk\.2\.ffn_(up|down)_exps\.weight=CPU,blk\.([3-9]|[1-3][0-9]|40)\.ffn_.*_exps\.weight=CPU
    no-kv-offload = true
    kv-stream-stage-mib = 400
    cache-type-k = q8_0
    cache-type-v = q8_0
    ctx-size = 262144
    batch-size = 1024
    ubatch-size = 512
    parallel = 1
    threads = 6
    load-mode = none
    spec-type = draft-mtp
    spec-draft-n-max = 2
    spec-draft-p-min = 0.85
    reasoning = auto
    reasoning-budget = 4096
    reasoning-budget-message = "Concluye tu razonamiento y proporciona la respuesta final. Responde ahora."

    [qwen3.6-35b-a3b-mtp-q4-ks]
    model = /path/to/Qwen3.6-35B-A3B-bartowsky-Q4_K_S.gguf
    mmproj = /path/to/mmproj-Q8_0.gguf
    alias = qwen3.6-35b-a3b-mtp-q4-ks
    fit = off
    n-gpu-layers = 99
    n-cpu-moe = 99
    no-mmproj-offload = true
    no-kv-offload = true
    kv-stream-stage-mib = 400
    load-mode = none
    ctx-size = 131072
    temperature = 0.80
    top-k = 20
    top-p = 0.95
    min-p = 0
    reasoning = auto
    reasoning-budget = 4096
    reasoning-budget-message = "Concluye tu razonamiento y proporciona la respuesta final. Responde ahora."
