# qwen-inference

[Challenge link](https://adaptfm.gitlab.io/call-for-competition/)

This repo is a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/) with two independently runnable projects:

| Path | Purpose |
| --- | --- |
| [`scripts/`](scripts/) | Utility scripts (e.g. downloading model weights) |
| [`qwen-inference/`](qwen-inference/) | Inference server package for Docker submissions |

## Quick start

### Requirements

- Linux with Python 3.12 and [`uv`](https://docs.astral.sh/uv/) installed.
- NVIDIA GPU with enough VRAM for Qwen3.5-4B; the target competition hardware is A10G/A10-class (`sm_86`, 24 GB).
- NVIDIA driver/CUDA runtime compatible with the installed PyTorch build. When the `vllm` dependency group is installed, PyTorch comes from the vLLM wheel (currently `torch 2.10.x`).
- Model weights downloaded into `qwen-inference/qwen-weights/`.
- For local profiling scripts: `curl`, `python3`, and optionally NVIDIA Nsight Systems (`nsys`) for CUDA traces.

Install the workspace dependencies (including vLLM for the `vllm` backend):

```bash
UV_TORCH_BACKEND=auto uv sync --package qwen-inference --group dev --group vllm
```

Download model weights:

```bash
uv run --directory scripts download_weights.py
```

Run the inference server locally:

```bash
uv run --package qwen-inference qwen-serve
```

The server supports three backends:

```bash
# Baseline: Hugging Face Transformers model.generate()
uv run --package qwen-inference qwen-serve --mode baseline

# Custom: optimized loader/backend path
uv run --package qwen-inference qwen-serve --mode custom

# vLLM: competition-style serving baseline (optional dependency)
uv run --package qwen-inference qwen-serve --mode vllm
```

You can also use `INFERENCE_MODE=baseline`, `INFERENCE_MODE=custom`, or `INFERENCE_MODE=vllm`.

### vLLM backend

vLLM is installed via the `vllm` dependency group on Python 3.12. It is loaded lazily only when `--mode vllm` is selected, so `baseline` and `custom` runs do not require vLLM to be installed unless you sync that group.

```bash
UV_TORCH_BACKEND=auto uv sync --package qwen-inference --group dev --group vllm
uv run --package qwen-inference qwen-serve --mode vllm
```

Qwen3.5 support currently requires `transformers` from the upstream Git repository; `uv sync` handles that automatically via `[tool.uv.sources]`.

Build a submission image:

```bash
docker build -t my-submission:latest qwen-inference/
```

## Profiling

[`scripts/profile.sh`](scripts/profile.sh) starts the inference server, sends eval-shaped requests, and writes results under `profiles/`. Run it from the repo root (or from `scripts/`; it `cd`s to the repo root automatically).

### Prerequisites

- Dependencies installed: `UV_TORCH_BACKEND=auto uv sync --package qwen-inference --group dev --group vllm`
- Model weights in `qwen-inference/qwen-weights/`
- `uv`, `curl`, and `python3` on `PATH`
- For `nsys-forward`: Nsight Systems (`nsys`) and a working host importer (see below)

### Usage

```text
./scripts/profile.sh [baseline|custom|vllm|all] [short|medium|long|all] [latency|cuda-forward|nsys-forward]
```

| Argument | Values | Default |
| --- | --- | --- |
| Backend | `baseline`, `custom`, `vllm`, `all` | `baseline` |
| Prompt size | `short`, `medium`, `long`, `all` | `short` |
| Profile kind | `latency`, `cuda-forward`, `nsys-forward` | `latency` |

Prompt sizes match [`evals/run_eval_local.py`](evals/run_eval_local.py):

| Size | Prompt | `max_tokens` |
| --- | --- | --- |
| `short` | synthetic ~64 tokens | 128 |
| `medium` | synthetic ~2048 tokens | 256 |
| `long` | synthetic ~8192 tokens | 256 |
| `all` | latency mode only; runs short, medium, and long | — |

### Output layout

```text
profiles/
├── latency/   # .json latency summaries
├── logs/      # server .log files
└── traces/    # .trace.json (cuda-forward) or .nsys-rep (nsys-forward)
```

File names follow:

```text
profiles/<subdir>/<mode>-<prompt-size>-<profile-kind>-<timestamp>.<ext>
```

### Environment variables

| Variable | Default | Used by |
| --- | --- | --- |
| `WARMUP_RUNS` | `3` | all modes |
| `LATENCY_RUNS` | `50` | `latency` |
| `DECODE_STEPS` | `4` | `cuda-forward`, `nsys-forward` |
| `HOST` | `127.0.0.1` | server bind / client URL |
| `PORT` | `8080` | server bind / client URL |
| `PROFILE_DIR` | `profiles` | output root |
| `FILLER` | `The quick brown fox jumps over the lazy dog. ` | synthetic prompt text |

### Common commands

```bash
# End-to-end latency for one backend and prompt size
./scripts/profile.sh baseline short latency

# All three backends on the short prompt (latency only)
./scripts/profile.sh all short latency

# All prompt sizes for baseline latency (competition comparison)
LATENCY_RUNS=50 ./scripts/profile.sh baseline all latency

# Small CUDA kernel trace: 1 prefill + 4 decode forwards
DECODE_STEPS=4 ./scripts/profile.sh baseline short cuda-forward

# Compare baseline vs vLLM forward kernels with the same prompt/decode count
DECODE_STEPS=4 ./scripts/profile.sh baseline short cuda-forward
DECODE_STEPS=4 ./scripts/profile.sh vllm short cuda-forward

# Profile all backends with the same forward-pass settings
DECODE_STEPS=4 ./scripts/profile.sh all short cuda-forward
```

### End-to-end latency (`latency`)

Latency mode does not collect CUDA traces. It starts the server normally, sends warmup requests, then records mean/median/min/max per-run latency plus delta and speedup versus competition baselines from [`competition-guide.md`](competition-guide.md) (short=2582 ms, medium=5441 ms, long=6576 ms).

```bash
./scripts/profile.sh baseline short latency
LATENCY_RUNS=50 ./scripts/profile.sh vllm all latency
```

Outputs:

```text
profiles/latency/<mode>-<prompt-size>-latency-<timestamp>.json
profiles/logs/<mode>-<prompt-size>-latency-<timestamp>.log
```

Use this mode when comparing against eval latency; it includes HTTP handling, tokenization, generation, and response decoding.

### CUDA forward trace (`cuda-forward`)

CUDA forward mode captures a small Chrome trace: one prefill forward plus `DECODE_STEPS` single-token decode forwards (default 4). This is better for inspecting kernel launches than tracing a full 128-token generation.

- **baseline** / **custom**: `torch.profiler` via `/profile/forward`
- **vllm**: vLLM's built-in torch profiler with the same bounded decode count

```bash
DECODE_STEPS=4 ./scripts/profile.sh baseline short cuda-forward
DECODE_STEPS=4 ./scripts/profile.sh vllm short cuda-forward
```

Outputs:

```text
profiles/traces/<mode>-<prompt-size>-cuda-forward-<timestamp>.trace.json
profiles/logs/<mode>-<prompt-size>-cuda-forward-<timestamp>.log
```

Open traces in `chrome://tracing` or [Perfetto](https://ui.perfetto.dev/).

### Nsight Systems forward trace (`nsys-forward`)

Use `nsys-forward` when you need an Nsight Systems `.nsys-rep` file. The script sends warmup forward requests, then one measured request wrapped with `cudaProfilerApi`.

```bash
DECODE_STEPS=4 ./scripts/profile.sh baseline short nsys-forward
```

Outputs, when Nsight import succeeds:

```text
profiles/traces/<mode>-<prompt-size>-nsys-forward-<timestamp>.nsys-rep
profiles/logs/<mode>-<prompt-size>-nsys-forward-<timestamp>.log
```

Summarize with:

```bash
nsys stats profiles/traces/<file>.nsys-rep
```

Nsight mode requires:

- `nsys` on `PATH` (the script can attempt `apt-get` install on Debian/Ubuntu)
- the Nsight Systems host importer at `/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter`
- importer dependencies that can run on the host

If the importer cannot run, Nsight may produce only a raw `.qdstrm` file. `nsys stats` cannot read `.qdstrm`; it needs `.nsys-rep` or `.sqlite`. The script validates the importer before collecting so this fails early instead of creating another unusable raw capture. On this machine, the Lambda Labs `nsight-systems` package expects a `libssh` symbol version `LIBSSH_4_9_0`, while Ubuntu 22.04's stock `libssh` only provides older symbol versions. Install a compatible Nsight Systems host package/libssh combination before using `nsys-forward`.

### Notes

- `PROMPT_SIZE=all` is only supported for `latency`; run `short`, `medium`, and `long` separately for forward-pass modes.
- `MODE=all` runs `baseline`, then `custom`, then `vllm` with the same settings.
- First `vllm` server start can take several minutes while vLLM compiles; later runs are faster.
- `profiles/` is gitignored.

## Benchmarking

[`scripts/benchmark.sh`](scripts/benchmark.sh) runs the competition eval harness
([`evals/run_eval_local.py`](evals/run_eval_local.py)) against a chosen set of
model weights. It starts the inference server, waits for `/ping`, runs the
requested benchmark(s), then shuts the server down.

```text
./scripts/benchmark.sh [quality|latency|both] [MODEL_WEIGHTS_DIR]
```

| Argument | Values | Default |
| --- | --- | --- |
| Benchmark kind | `quality`, `latency`, `both` | `both` |
| Model weights | path to a weights directory | `qwen-inference/qwen-weights` |

The kind maps to the harness `EVAL_MODE` (`quality` → `quality`, `latency` →
`latency`, `both` → `full`). The weights path is exported as `MODEL_DIR`, which
is how `qwen-serve` selects weights for every backend.

### Prerequisites

- Workspace deps installed: `UV_TORCH_BACKEND=auto uv sync --package qwen-inference --group dev --group vllm`
- The eval harness deps importable by `EVAL_PYTHON` (default `python3`):
  `pip install lm-eval==0.4.11 langdetect immutabledict`
  (`run_eval_local.py` imports `lm_eval` at module load, so this is required for
  every kind, including `latency`.) Point at a different interpreter with
  `EVAL_PYTHON=/path/to/python`.
- For quality benchmarks, the MMLU-Pro / IFEval / GPQA-Diamond datasets must be
  available in `HF_HOME` (the harness runs with `HF_HUB_OFFLINE=1`).

### Examples

```bash
# Serve the INT4 (GPTQ) checkpoint and run both quality + latency
./scripts/benchmark.sh both qwen-inference/qwen-weights-quantized

# Quality gates only, on the full set (not the 10% dev sample)
QUALITY_LIMIT=1.0 ./scripts/benchmark.sh quality qwen-inference/qwen-weights-quantized

# Latency only against the bf16 baseline weights
LATENCY_RUNS=50 ./scripts/benchmark.sh latency qwen-inference/qwen-weights
```

### Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `INFERENCE_MODE` / `MODE` | `vllm` | Serving backend (`baseline`, `custom`, `vllm`) |
| `QUALITY_LIMIT` | `0.1` | Fraction of each quality task to run (`1.0` = full) |
| `NUM_CONCURRENT` | `8` | Concurrent quality requests |
| `LATENCY_RUNS` | `50` | Measured latency runs (passed as `NUM_RUNS`) |
| `EVAL_PYTHON` | `python3` | Interpreter with the eval harness installed |
| `HOST` / `PORT` | `127.0.0.1` / `8080` | Server bind / client URL |
| `SERVER_WAIT_SECONDS` | `600` | Max wait for `/ping` |
| `RESULTS_DIR` | `profiles/benchmarks` | Output root for result JSON |

### Output layout

```text
profiles/
├── benchmarks/  # <mode>-<weights>-<kind>-<timestamp>.json result summaries
└── logs/        # server .log files
```

## Quantization

[`scripts/quantize_gptq.py`](scripts/quantize_gptq.py) produces an INT4 (GPTQ
W4A16) `compressed-tensors` checkpoint that vLLM serves via the Marlin kernel on
Ampere (A10G). Run it in a dedicated environment (kept out of the uv workspace
because the `llmcompressor` line that supports the git `transformers` build
conflicts with the `vllm`-pinned `compressed-tensors`):

```bash
uv venv .venv-quantize --python 3.12
UV_TORCH_BACKEND=auto uv pip install --python .venv-quantize -r scripts/quantize-requirements.txt
.venv-quantize/bin/python scripts/quantize_gptq.py
```

By default it reads `qwen-inference/qwen-weights` and writes
`qwen-inference/qwen-weights-quantized`. Validate the result with
`./scripts/benchmark.sh both qwen-inference/qwen-weights-quantized` before
submitting — especially the GPQA-Diamond quality gate.

See [`competition-guide.md`](competition-guide.md) for full competition details.
