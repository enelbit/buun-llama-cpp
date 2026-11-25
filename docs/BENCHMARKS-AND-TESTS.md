# Benchmarks, Validation Tests, and Methodology (`kv-stream-pascal`)

## 1. Reference Test Hardware & Environment
- **GPU:** NVIDIA GeForce GTX 1060 6 GB (`sm_61`, Pascal, 6144 MiB VRAM; ~293 MiB used by Wayland compositor, strict server budget <= 5200 MiB).
- **CPU & RAM:** AMD Ryzen 5 3600 (6C/12T), 48 GB DDR4 system RAM.
- **OS & Compiler:** CachyOS Linux x86_64, GCC 14.3.1, CUDA Toolkit (`-DCMAKE_CUDA_ARCHITECTURES=61 -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON`).

## 2. Test Methodology
1. **Sustained 2000-Token Generation Stress Test:**
   - Endpoint: `POST /completion` on port `18999` with `"n_predict": 2000`, `"ignore_eos": true`, `"temperature": 0.7`.
   - Purpose: Force continuous KV-cache growth, CUDA graph reuse, and speculative drafting (`draft-mtp`) across 2000 consecutive decode steps without early EOS termination.
2. **Split Attention Ring Overflow Test (`n_kv_total > max_ring_rows`):**
   - Configured with `--kv-stream-stage-mib 64` on a 10-attention-layer MoE model (`Qwen3.6-35B-A3B`) so active context rows exceed the resident GPU ring capacity, forcing `kv_split_q` -> CPU `ggml_flash_attn_ext` worker -> `kv_split_merge` (Log-Sum-Exp fusion) alongside `draft-mtp`.
3. **Budget Independence & Shortfall Diagnostics (`--fit-budget` vs `--kv-stream-stage-mib`):**
   - **Case A:** `--fit on --fit-budget 5150` alone.
   - **Case B:** `--fit off --kv-stream-stage-mib 400` alone.
   - **Case C1:** `--fit on --fit-budget 300 --kv-stream-stage-mib 400` (provokes diagnostic shortfall, triggers safe fallback to `0` GPU layers without OOM).
   - **Case C2:** `--fit on --fit-budget 5150 --kv-stream-stage-mib 400` (deducts stage + conversion headroom before layer placement).

## 3. Verified Results Matrix

| Model & Quant | Context | KV Cache & Staging | Speculative Decode | VRAM (`nvidia-smi`) | Prefill (`pp`) | Sustained Decode (`tg`, 2000 tok) | MTP Accept |
|---|---|---|---|---|---|---|---|
| **Qwen3.6-35B-A3B-MTP** (`UD-Q5_K_S`) | 262,144 | `q8_0`, `-nkvo`, `stage=400 MiB` | `draft-mtp` (`n=2, p=0.85`) | **5,172 MiB** | 12.1 t/s | **16.51 t/s** | 64.4% |
| **Qwen3.6-35B-A3B-MTP** (`UD-Q5_K_S`) | 262,144 | `q8_0`, `-nkvo`, `stage=64 MiB` (Split Attn) | `draft-mtp` (`n=2, p=0.85`) | **4,690 MiB** | 12.54 t/s | **16.58 t/s** | 64.4% |
| **Qwen3.5-4B-MTP** (`Q5_K_S`) | 262,144 | `q8_0`, `-nkvo`, `stage=400 MiB` | `draft-mtp` (`n=3, p=0.80`) | **5,214 MiB** | 38.43 t/s | **23.32 t/s** | 50.1% |
| **Qwen3.5-4B-MTP** (`Q5_K_S`) | 131,072 | `turbo8`, `-nkvo`, `stage=400 MiB`, `fit=5190` | `draft-mtp` (`n=3, p=0.80`) | **< 5,190 MiB** | 19.70 t/s | **31.09 t/s** (peak 36.83) | 81.4% |
| **Qwen3.5-4B-MTP** (`Q5_K_S`) | 131,072 | `turbo8` (native `fattn`), `-nkvo`, `stage=400 MiB` | `draft-mtp` (`n=2, p=0.85`) | **4,280 MiB** | 22.30 t/s | **22.17 t/s** | 45.2% |
| **Qwen3.5-4B-MTP** (`Q5_K_M`) | 131,072 | `turbo8` (full VRAM, no `kv-stream`) | `draft-mtp` (`n=1, p=0.85`) | 5,640 MiB | 40.82 t/s | **37.54 t/s** | 52.5% |
