# qwen-inference

Python package for the Qwen3.5-4B inference server used in competition submissions.

## Usage

From the repo root:

```bash
uv run --package qwen-inference qwen-serve
```

Or from this directory:

```bash
uv run qwen-serve
```

Implement your optimized server in `src/qwen_inference/serve.py`.

## Docker

Download weights into this package directory, then build:

```bash
uv run --directory ../scripts download_weights.py
docker build -t my-submission:latest .
```

## A10 Specs

Reference hardware for this package. The competition runs on **A10G** (AWS `g5.xlarge`), which uses the same GA102 die and CUDA architecture as the A10 — kernel tuning targets `sm_86` for both.

Sources: [NVIDIA A10 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a10/pdf/datasheet-new/nvidia-a10-datasheet.pdf), [GA102 whitepaper](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf), [Ampere tuning guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/).

### GPU overview

| Spec | Value |
| --- | --- |
| GPU | GA102-890 (partial GA102 die) |
| Architecture | Ampere |
| Process | Samsung 8 nm |
| Compute capability | **8.6** (`sm_86`) |
| SMs | 72 |
| CUDA cores | 9,216 (128 per SM) |
| Tensor cores | 288 (4 per SM, 3rd gen) |
| RT cores | 72 (1 per SM, 2nd gen) |
| Clock (base / boost) | 885 / 1,695 MHz |
| TDP | 150 W |
| FP64 | Limited scalar only (~1/64 of FP32); **no FP64 tensor cores** |

Compile for this GPU explicitly: `-arch=sm_86` (or `compute_86` / `sm_86` in `-gencode`). Binaries built for CC 8.0 (A100) run on A10 but miss the 2× FP32 throughput per SM that CC 8.6 provides.

### Memory

| Spec | Value |
| --- | --- |
| VRAM | 24 GB GDDR6 (ECC supported) |
| Bus width | 384-bit (12 × 32-bit controllers) |
| Peak bandwidth | 600 GB/s |
| L2 cache | 6 MB (512 KB per memory controller) |
| Host interconnect | PCIe 4.0 ×16 (~64 GB/s) |

For LLM inference, 24 GB fits Qwen3.5-4B comfortably; bandwidth (not raw FLOPS) often dominates prefill and large-batch decode.

### SM architecture (kernel-relevant)

Each SM is a GA10x block with:

- **128 FP32 CUDA cores** — can execute 128 FP32 *or* 64 FP32 + 64 INT32 ops per clock (dual datapath).
- **4 warp schedulers**, **4 dispatch units** — up to 2 warps issued per clock.
- **4 third-gen tensor cores** (one per processing block).
- **256 KB register file** (64K × 32-bit registers).
- **128 KB unified L1 / shared memory** (runtime carveout; see below).
- **Shared memory bandwidth**: 128 bytes/clock per SM.
- **32 shared-memory banks**, 4 bytes wide — avoid bank conflicts when lanes in a warp hit the same bank.

Ampere features useful in custom kernels:

- **`cp.async`** — async global → shared copies (bypass L1, overlap with compute).
- **Split arrive/wait barriers** in shared memory (producer–consumer pipelines).
- **Hardware warp reductions** (`__reduce_add_sync`, min, max, and/or/xor).
- **2:4 structured sparsity** on tensor cores (doubles effective matmul throughput when weights are sparse).

### Occupancy limits (CC 8.6 — differs from A100 / CC 8.0)

| Limit | CC 8.6 (A10) | CC 8.0 (A100) |
| --- | --- | --- |
| Warps per SM | **48** | 64 |
| Threads per SM | **1,536** | 2,048 |
| Thread blocks per SM | **16** | 32 |
| Threads per block | 1,024 | 1,024 |
| Registers per SM | 64K × 32-bit | 64K × 32-bit |
| Max registers per thread | 255 | 255 |
| Shared memory per SM | **100 KB max** | 164 KB max |
| Shared memory per block | **99 KB max** | 163 KB max |

Shared-memory carveout options (via `cudaFuncAttributePreferredSharedMemoryCarveout`): **0, 8, 16, 32, 64, 100 KB** per SM. CUDA reserves 1 KB per block. Static `__shared__` allocations stay capped at **48 KB** unless you opt in to larger dynamic shared memory.

Grid limits: max block dimensions `(1024, 1024, 64)`; max grid `(2³¹−1, 65535, 65535)`. Warp size is always **32** — launch thread counts as multiples of 32 for full warp utilization.

### L1 / shared carveout (compute mode)

Total per SM: 128 KB combined L1 + shared. Typical compute configurations:

| L1 cache | Shared memory |
| --- | --- |
| 128 KB | 0 KB |
| 120 KB | 8 KB |
| 112 KB | 16 KB |
| 96 KB | 32 KB |
| 64 KB | 64 KB |
| 28 KB | 100 KB |

Kernels heavy on `__shared__` tiling (GEMM, attention) benefit from larger shared carveouts; memory-bound kernels may prefer more L1.

### Tensor cores (3rd gen)

Supported dtypes and peak throughput (dense | 2:4 sparse*):

| Precision | Peak throughput |
| --- | --- |
| FP32 (CUDA cores) | 31.2 TFLOPS |
| TF32 | 62.5 \| 125 TFLOPS |
| FP16 / BF16 | 125 \| 250 TFLOPS |
| INT8 | 250 \| 500 TOPS |
| INT4 | 500 \| 1,000 TOPS |

\*Sparse figures require 2:4 structured sparsity in weights.

Native MMA tile sizes (Ampere HMMA/IMMA):

| Instruction | Input formats | Accumulator | Tile (M×N×K) |
| --- | --- | --- | --- |
| HMMA | FP16, BF16 | FP16 / FP32 | 16×8×8, 16×8×16 |
| HMMA | TF32 | FP32 | 16×8×4 |
| IMMA | INT8 | INT32 | 8×8×16, 16×8×16, 16×8×32 |
| IMMA | INT4 | INT32 | 8×8×32, 16×8×32, 16×8×64 |

Use `mma.sync` / WMMA APIs or libraries (cuBLAS, CUTLASS) rather than hand-rolling unless profiling shows a win. For inference on A10, **FP16/BF16/INT8 tensor paths** are the main lever; TF32 is primarily a training format.

### Practical tuning notes

- **Occupancy**: A10 caps at 48 warps/SM — register-heavy kernels drop occupancy faster than on A100.
- **Memory coalescing**: L1 acts as a coalescing buffer; ensure consecutive threads access consecutive addresses.
- **Block sizing**: With 72 SMs, aim for enough blocks to fill the GPU (often 72–576+ depending on occupancy). Undersized grids leave SMs idle.
- **A10 vs A10G**: Same silicon and kernel limits; A10G (AWS) typically boosts higher (~1,710 MHz) so absolute latency differs, not architecture.
