# qwen-inference

[Challenge link](https://adaptfm.gitlab.io/call-for-competition/)

This repo is a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/) with two independently runnable projects:

| Path | Purpose |
| --- | --- |
| [`scripts/`](scripts/) | Utility scripts (e.g. downloading model weights) |
| [`qwen-inference/`](qwen-inference/) | Inference server package for Docker submissions |

## Quick start

Download model weights:

```bash
uv run --directory scripts download_weights.py
```

Run the inference server locally (after implementing `serve.py`):

```bash
uv run --package qwen-inference qwen-serve
```

Build a submission image:

```bash
docker build -t my-submission:latest qwen-inference/
```

See [`competition-guide.md`](competition-guide.md) for full competition details.
