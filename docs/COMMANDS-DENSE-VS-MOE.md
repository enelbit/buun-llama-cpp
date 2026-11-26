# Master Command & Configuration Reference (`Terminal CLI` vs. `models.ini`)

> All flags listed below have been verified directly against `./build-pascal/bin/llama-server --help`.
> If **Compatibility** says **All (MoE / Hybrid / Dense)**, the flag works across all three model families.

---

## 0. Master Flag Reference Table (`CLI` vs. `models.ini` & Architecture Compatibility)

### 0.1 Model, Multimodal & Identification

| Terminal CLI Flag | `models.ini` Key | Example Value | Compatibility & Usage Notes |
|---|---|---|---|
| `--model <path>` | `model = <path>` | `--model /path/model.gguf` | **All.** Required model path. |
| `--mmproj <path>` | `mmproj = <path>` | `--mmproj /path/mmproj.gguf` | **MoE / Hybrid** (multimodal). Omit on text-only models (`DeepSeek-R1-8B`). |
| `--no-mmproj-offload` | `no-mmproj-offload = true` | `--no-mmproj-offload` | **MoE / Hybrid** with `mmproj`. Keeps vision encoder in RAM to save ~1.1 GiB VRAM. |
| `--mmproj-gpu-swap` | `mmproj-gpu-swap = true` | `--mmproj-gpu-swap` | **MoE / Hybrid** with `mmproj` + `draft-mtp` (without `-nkvo`). Swaps MTP out during image encoding. |
| `--image-min-tokens <N>` | `image-min-tokens = <N>` | `--image-min-tokens 2048` | **MoE / Hybrid** with `mmproj`. Minimum visual token budget per image. |
| `--alias <name>` | `alias = <name>` | `--alias qwen3.5-4b-mtp-q5ks` | **All.** In `models.ini`, must match the `[section-name]`. |

### 0.2 GPU Device Selection & Multi-GPU Splitting

| Terminal CLI Flag | `models.ini` Key | Example Value | Compatibility & Usage Notes |
|---|---|---|---|
| `-dev, --device <dev1,dev2|none>` | `device = <dev1,dev2|none>` | `--device CUDA0` / `CUDA0,CUDA1` | **All.** Selects target GPU(s) (`--list-devices` lists IDs). `none` forces CPU-only execution. |
| `-sm, --split-mode {none,layer,row,tensor}` | `split-mode = <mode>` | `--split-mode layer` | **All (Multi-GPU).** `none`: single GPU; `layer`: splits layers across GPUs (best with `-ot` / MoE); `row` / `tensor`: tensor-parallel split. |
| `-ts, --tensor-split <N0,N1,...>` | `tensor-split = <N0,N1,...>` | `--tensor-split 24,6` | **All (Multi-GPU).** Proportional VRAM split across GPUs (e.g. RTX 3090 + GTX 1060). |
| `-mg, --main-gpu <INDEX>` | `main-gpu = <INDEX>` | `--main-gpu 0` | **All.** Primary GPU for scratch buffers and the `kv-stream` staging ring. |

### 0.3 Memory Fit, Layer Placement, MoE Offload & Loading Modes

| Terminal CLI Flag | `models.ini` Key | Example Value | Compatibility & Usage Notes |
|---|---|---|---|
| `--fit <on|off>` | `fit = <on|off>` | `--fit on` / `--fit off` | **All.** Use `off` when manually pinning `-ngl 99` + `-ot`; use `on` with `--fit-budget` or `--fit-target`. |
| `--fit-budget <MiB>` | `fit-budget = <MiB>` | `--fit-budget 5190` | **All (with `--fit on`).** Hard process VRAM ceiling in MiB (deducts `kv-stream-stage-mib + 32 MiB`). |
| `--fit-target <MiB>` | `fit-target = <MiB>` | `--fit-target 300` | **All (with `--fit on`).** Free VRAM headroom target in MiB (alternative to `--fit-budget`). |
| `-ngl, --n-gpu-layers <N>` | `n-gpu-layers = <N>` | `--n-gpu-layers 99` / `0` | **All.** `99` = full GPU offload (with `--fit off`); `0` = CPU-only. |
| `-ot, --override-tensor "<regex>"` | `override-tensor = <regex>` | See Section 2 & 3 | **MoE & Hybrid.** Pins matching tensors to `CPU` (e.g. `ffn_.*_exps` on MoE; `ffn_(gate|up|down)` on Hybrid). |
| `-cmoe, --cpu-moe` | `cpu-moe = true` | `--cpu-moe` | **MoE only.** Offloads all routed experts (`ffn_.*_exps`) to CPU. |
| `-ncmoe, --n-cpu-moe <N>` | `n-cpu-moe = <N>` | `--n-cpu-moe 38` | **MoE only.** Offloads routed experts of the first `N` layers to CPU. |
| `--moe-cache <auto|on|off|soft|MiB>` | `moe-cache = <val>` | `--moe-cache off` | **MoE only (`sm_70+` / `gfx1100+`).** Disabled on Pascal `sm_61` while preserving CPU AVX2/FMA repacking. |
| `--moe-cache-cpu-overlap <auto|0..8>` | `moe-cache-cpu-overlap = <val>` | `--moe-cache-cpu-overlap 2` | **MoE only (`sm_70+` / `gfx1100+`).** Runs `N` cached expert rows on CPU in parallel with GPU cache. |
| `--moe-cache-expert-parallel <auto|0|1>` | `moe-cache-expert-parallel = <val>` | `--moe-cache-expert-parallel auto` | **MoE Multi-GPU (`sm_70+` / `gfx1100+`).** Splits cached expert rows across 2-3 GPUs. |
| `--load-mode <mmap|none|mlock>` | `load-mode = <val>` | `--load-mode none` / `mmap` / `mlock` | **All.** `none`: fastest when tensors run on CPU (direct AVX2/FMA repack); `mmap`: instant startup; `mlock`: locks weights in RAM. |
| `--no-repack` | `no-repack = true` | *(Omit by default)* | **All.** Disables CPU AVX2/FMA weight repacking (only use if forced by extreme RAM limits). |
| `--repack-cache <DIR>` | `repack-cache = <DIR>` | `--repack-cache /tmp/repack` | **All (safetensors SSD offload).** Reuses prepared host weights across launches on Linux. |

### 0.4 Context, PCIe `kv-stream`, KV Codecs & VBR

| Terminal CLI Flag | `models.ini` Key | Example Value | Compatibility & Usage Notes |
|---|---|---|---|
| `-c, --ctx-size <N>` | `ctx-size = <N>` | `--ctx-size 262144` | **All.** Context window size (up to `262144` tokens under 5.2 GB VRAM with `kv-stream`). |
| `-nkvo, --no-kv-offload` | `no-kv-offload = true` | `--no-kv-offload` | **All.** Places KV cache in pinned host RAM (required for `--kv-stream-stage-mib`). |
| `--kv-stream-stage-mib <MiB>` | `kv-stream-stage-mib = <MiB>` | `--kv-stream-stage-mib 400` | **All (with `-nkvo`).** Bounded VRAM DMA staging ring (`64`-`512 MiB`) + CPU/GPU split attention. |
| `--kv-stream-min-mib <MiB>` | `kv-stream-min-mib = <MiB>` | `--kv-stream-min-mib 64` | **All (with `kv-stream`).** Minimum KV streaming pool floor in MiB (`0` = no explicit CLI floor). |
| `--kv-stream-max-mib <MiB>` | `kv-stream-max-mib = <MiB>` | `--kv-stream-max-mib 512` | **All (with `kv-stream`).** Maximum KV streaming pool ceiling in MiB (`0` = no ceiling). |
| `-fa, --flash-attn <on|off>` | `flash-attn = <on|off>` | `--flash-attn on` | **All.** Required for `kv-stream`, `split attention`, VBR, and quantized KV caches. |
| `-ctk, --cache-type-k <type>` | `cache-type-k = <type>` | `q8_0` / `turbo8` / `turbo4` / `f16` | **All.** Use `q8_0` or `turbo8` with `kv-stream` (do not use `vbr` together with `-nkvo`). |
| `-ctv, --cache-type-v <type>` | `cache-type-v = <type>` | `q8_0` / `turbo8` / `turbo4` / `f16` | **All.** Pair with `cache-type-k`. |
| `--vbr-entry <TIER>` | `vbr-entry = <TIER>` | `--vbr-entry t8` | **All (GPU VBR only, without `-nkvo`).** Entry tier (`f16`, `t8`, `t4`). Incompatible with `-nkvo`. |
| `--vbr-floor <TIER|BITS>` | `vbr-floor = <TIER|BITS>` | `--vbr-floor t4` | **All (GPU VBR only, without `-nkvo`).** Minimum quality floor (`t4`, `t3_tcq`, `t2_tcq`, `t1_tcq`). |
| `--vbr-vram <SIZE>` | `vbr-vram = <SIZE>` | `--vbr-vram 1200M` | **All (GPU VBR only, without `-nkvo`).** Fixed VRAM budget for dynamic VBR cache. |
| `--kv-unified` / `--no-kv-unified` | `kv-unified = true` / `no-kv-unified = true` | `--no-kv-unified` (with `parallel = 1`) | **All.** Use `--no-kv-unified` for single-slot (`parallel = 1`); use `--kv-unified` when `parallel > 1`. |
| `-np, --parallel <N>` | `parallel = <N>` | `--parallel 1` | **All.** Keep at `1` to dedicate the full `131K`-`262K` context window to a single session. |
| `-b, --batch-size <N>` | `batch-size = <N>` | `--batch-size 1024` / `2048` | **All.** Logical prompt-processing batch size. |
| `-ub, --ubatch-size <N>` | `ubatch-size = <N>` | `--ubatch-size 512` / `256` | **All.** Physical CUDA micro-batch size (`256` saves ~200 MiB compute VRAM vs `512`). |
| `-t, --threads <N>` | `threads = <N>` | `--threads 6` | **All.** Set to physical CPU cores (`6` on Ryzen 5 3600) to avoid SMT contention. |
| `-ctxcp, --ctx-checkpoints <N>` | `ctx-checkpoints = <N>` | `--ctx-checkpoints 16` | **Hybrid DeltaNet (`Qwen3.5/3.6/3.8`).** Max recurrent SSM checkpoints per slot for fast prefix reuse. |
| `--no-context-shift` | `no-context-shift = true` | `--no-context-shift` | **All.** Stops cleanly when `ctx-size` is full instead of shifting old KV rows (recommended with `kv-stream`). |

### 0.5 Speculative Decoding (`draft-mtp` & External Sidecars)

| Terminal CLI Flag | `models.ini` Key | Example Value | Compatibility & Usage Notes |
|---|---|---|---|
| `--spec-type draft-mtp` | `spec-type = draft-mtp` | `--spec-type draft-mtp` | **Models with built-in or sidecar MTP** (`Qwen3.5-MTP`, `Qwen3.6-MTP`, `Qwen3.8+MTP`). Omit on `DeepSeek-R1-8B`. |
| `-md, --spec-draft-model <path>` | `spec-draft-model = <path>` | `--spec-draft-model /path/mtp.gguf` | **External drafter GGUF only** (e.g. `mtp-Qwen3.8-27B-Q4_0.gguf`). Omit when MTP is embedded. |
| `-ngld, --n-gpu-layers-draft <N>` | `n-gpu-layers-draft = <N>` | `--n-gpu-layers-draft 0` | **External drafter GGUF only.** `0` keeps external draft model in CPU RAM without spending VRAM. |
| `-devd, --spec-draft-device <dev>` | `spec-draft-device = <dev>` | `--spec-draft-device none` | **External drafter GGUF only.** Selects device for external draft model (`none` = CPU). |
| `-otd, --override-tensor-draft "<rx>"` | `override-tensor-draft = <rx>` | `-otd "exps=CPU"` | **External MoE drafter only.** Offloads specific draft tensors to CPU. |
| `--spec-draft-n-max <N>` | `spec-draft-n-max = <N>` | `--spec-draft-n-max 2` | **With `draft-mtp`.** Max draft tokens per step (`2` optimal on 6 GB VRAM; `1` on heavier 9B/27B models). |
| `--spec-draft-p-min <float>` | `spec-draft-p-min = <float>` | `--spec-draft-p-min 0.85` | **With `draft-mtp`.** Minimum draft token probability threshold to avoid costly rollbacks. |

### 0.6 Reasoning, Templates, Sampling & Router Server (Open WebUI 0.13.3)

| Terminal CLI Flag | `models.ini` Key | Example Value | Compatibility & Usage Notes |
|---|---|---|---|
| `--jinja` | `jinja = true` | `--jinja` | **All.** Enables Jinja chat templating (required for tool calling and structured reasoning). |
| `--reasoning <auto|on|off>` | `reasoning = <val>` | `--reasoning auto` | **All reasoning models (`Qwen3.5/3.6/3.8`, `DeepSeek-R1`).** |
| `--reasoning-budget <N>` | `reasoning-budget = <N>` | `--reasoning-budget 4096` | **All reasoning models.** Caps `<think>` tokens before forcing the final answer. |
| `--reasoning-budget-message "<txt>"` | `reasoning-budget-message = "<txt>"` | `"Concluye tu razonamiento..."` | **All reasoning models.** Message injected when `reasoning-budget` is reached. |
| `--chat-template-file <path>` | `chat-template-file = <path>` | `--chat-template-file /path/tpl.jinja` | **All.** Overrides the embedded GGUF Jinja template. |
| `--chat-template-kwargs '<json>'` | `chat-template-kwargs = <json>` | CLI: `'{"preserve_thinking":true}'`<br>INI: `{"preserve_thinking":true}` | **All.** Wrap in single quotes `'...'` in bash CLI; omit outer single quotes in `models.ini`. |
| `--temp` / `--top-k` / `--top-p` / `--min-p` | `temperature` / `top-k` / `top-p` / `min-p` | `temperature = 0.6` | **All.** Default sampling parameters for the model alias. |
| `--repeat-penalty <float>` | `repeat-penalty = <float>` | `--repeat-penalty 1.05` | **All.** Repetition penalty (`1.0` = disabled). |
| `--presence-penalty <float>` | `presence-penalty = <float>` | `--presence-penalty 0.0` | **All.** OpenAI-compatible presence penalty. |
| `--seed <N>` | `seed = <N>` | `--seed 42` | **All.** Deterministic RNG seed (`-1` = random). |
| `-cram, --cache-ram <MiB>` | *(Server startup CLI)* | `--cache-ram 4096` | **All.** Host RAM prefix/checkpoint cache size in MiB (`-1` = unlimited, `0` = disabled). |
| `--slot-save-path <DIR>` | *(Server startup CLI)* | `--slot-save-path /tmp/slots` | **All.** Directory for saving/restoring slot KV states on disk. |
| `--models-preset <models.ini>` | *(Server startup CLI)* | `--models-preset models.ini` | **All (Router Mode).** Exposes all sections in `models.ini` to Open WebUI 0.13.3. |
| `--models-max <N>` | *(Server startup CLI)* | `--models-max 1` | **All (Router Mode).** Max models loaded simultaneously (`1` required on 6 GB VRAM). |
| `--sleep-idle-seconds <SEC>` | *(Server startup CLI)* | `--sleep-idle-seconds 600` | **All (Router Mode).** Unloads idle model from VRAM after `N` seconds. |
| `--host <IP>` / `--port <PORT>` | *(Server startup CLI)* | `--host 127.0.0.1 --port 18999` | **All.** HTTP bind address and port (`0.0.0.0` for LAN/Docker). |
| `--api-key <KEY>` | *(Server startup CLI)* | `--api-key sk-local-1234` | **All.** Requires `Authorization: Bearer <KEY>` on `/v1/*` endpoints. |
| `--cors-origins <ORIGINS>` | *(Server startup CLI)* | `--cors-origins localhost` | **All.** Restricts browser CORS origins (`*` by default). |

---

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
