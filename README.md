# qwen-inference

[Challenge link](https://adaptfm.gitlab.io/call-for-competition/)

This repo is a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/) with two independently runnable projects:

| Path | Purpose |
| --- | --- |
| [`scripts/`](scripts/) | Utility scripts (e.g. downloading model weights) |
| [`qwen-inference/`](qwen-inference/) | Inference server package for Docker submissions |

## Quick start

### Requirements

- Linux with Python `>=3.13` and [`uv`](https://docs.astral.sh/uv/) installed.
- NVIDIA GPU with enough VRAM for Qwen3.5-4B; the target competition hardware is A10G/A10-class (`sm_86`, 24 GB).
- NVIDIA driver/CUDA runtime compatible with the installed PyTorch build. This workspace currently pins `torch==2.9.1` in the `qwen-inference` dev dependency group.
- Model weights downloaded into `qwen-inference/qwen-weights/`.
- For local profiling scripts: `curl`, `python3`, and optionally NVIDIA Nsight Systems (`nsys`) for CUDA traces.

Install the workspace dependencies:

```bash
uv sync --package qwen-inference --group dev
```

Download model weights:

```bash
uv run --directory scripts download_weights.py
```

Run the inference server locally:

```bash
uv run --package qwen-inference qwen-serve
```

The server supports two backends:

```bash
# Baseline: Hugging Face Transformers model.generate()
uv run --package qwen-inference qwen-serve --mode baseline

# Custom: optimized loader/backend path
uv run --package qwen-inference qwen-serve --mode custom
```

You can also use `INFERENCE_MODE=baseline` or `INFERENCE_MODE=custom`.

Build a submission image:

```bash
docker build -t my-submission:latest qwen-inference/
```

## Profiling

Use `scripts/profile.sh` from the repo root or from `scripts/`:

```bash
./scripts/profile.sh baseline short latency
```

Arguments are:

```text
scripts/profile.sh [baseline|custom] [short|medium|long|all] [latency|cuda-forward]
```

Prompt sizes match `evals/run_eval_local.py`:

- `short`: synthetic ~64-token prompt, `max_tokens=128`
- `medium`: synthetic ~2048-token prompt, `max_tokens=256`
- `long`: synthetic ~8192-token prompt, `max_tokens=256`
- `all`: latency mode only; runs short, medium, and long

### End-To-End Latency

Latency mode does not collect CUDA traces. It starts the server normally, sends eval-shaped warmup requests, then records average, median, min, max, and per-run latency:

```bash
LATENCY_RUNS=50 WARMUP_RUNS=5 ./scripts/profile.sh baseline all latency
```

Outputs:

```text
profiles/<mode>-<prompt-size>-latency-<timestamp>.latency.json
profiles/<mode>-<prompt-size>-latency-<timestamp>.server.log
```

This is the right mode for comparing against eval latency because it includes HTTP handling, tokenization, generation, and response decoding.

### CUDA Forward-Pass Trace

CUDA forward mode captures a much smaller Nsight Systems trace: one prefill forward plus a few single-token decode forwards. This is better for inspecting kernel-level behavior than tracing a full `model.generate()` request with 128+ decode iterations.

```bash
DECODE_STEPS=4 WARMUP_RUNS=5 ./scripts/profile.sh baseline short cuda-forward
```

Outputs, when Nsight import succeeds:

```text
profiles/<mode>-<prompt-size>-cuda-forward-<timestamp>.nsys-rep
profiles/<mode>-<prompt-size>-cuda-forward-<timestamp>.server.log
```

Summarize with:

```bash
nsys stats profiles/<file>.nsys-rep
```

CUDA forward mode requires:

- `nsys` on `PATH`
- the Nsight Systems host importer at `/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter`
- importer dependencies that can run on the host

If the importer cannot run, Nsight may produce only a raw `.qdstrm` file. `nsys stats` cannot read `.qdstrm`; it needs `.nsys-rep` or `.sqlite`. On this machine, the Lambda Labs `nsight-systems` package expects a `libssh` symbol version `LIBSSH_4_9_0`, while Ubuntu 22.04's stock `libssh` only provides older symbol versions. Install a compatible Nsight Systems host package/libssh combination before using `cuda-forward`.

See [`competition-guide.md`](competition-guide.md) for full competition details.
