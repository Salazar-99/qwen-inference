"""Docker entrypoint for the Qwen inference server."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import uvicorn
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen_inference.backends.custom.loader import DEFAULT_MODEL_DIR, load_weights
from qwen_inference.server import (
    app,
    configure_baseline,
    configure_custom,
    configure_vllm,
    register_async_initializer,
)
from qwen_inference.tokenizer import Tokenizer

DEFAULT_HOST = os.environ.get("HOST", "0.0.0.0")
DEFAULT_PORT = int(os.environ.get("PORT", "8080"))
DEFAULT_MODE = os.environ.get("INFERENCE_MODE", "custom")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Qwen inference server.")
    parser.add_argument(
        "--mode",
        choices=("baseline", "custom", "vllm"),
        default=DEFAULT_MODE,
        help='Serving backend to use. Can also be set with INFERENCE_MODE.',
    )
    return parser.parse_args()


def _load_custom_backend(model_dir: Path) -> None:
    with open(model_dir / "config.json") as config_file:
        config = json.load(config_file)

    weights = load_weights(config, model_dir)
    tokenizer = Tokenizer(model_dir)
    configure_custom(weights, tokenizer)


def _load_baseline_backend(model_dir: Path) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype).to(device)
    model.eval()
    configure_baseline(model, tokenizer, device=device)


def _load_vllm_backend(model_dir: Path) -> None:
    os.environ.setdefault("MODEL_DIR", str(model_dir))

    try:
        import vllm  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "vLLM is not installed. Install dependencies with "
            "'UV_TORCH_BACKEND=auto uv sync --package qwen-inference --group dev --group vllm', "
            "then rerun with --mode vllm."
        ) from exc

    # The async vLLM engine must be built inside the event loop; defer it to the
    # server's startup lifespan instead of constructing it here.
    register_async_initializer(lambda: configure_vllm(str(model_dir)))


def main() -> None:
    args = parse_args()
    model_dir = Path(DEFAULT_MODEL_DIR)

    if args.mode == "baseline":
        _load_baseline_backend(model_dir)
    elif args.mode == "vllm":
        _load_vllm_backend(model_dir)
    else:
        _load_custom_backend(model_dir)

    uvicorn.run(app, host=DEFAULT_HOST, port=DEFAULT_PORT, log_level="info")


if __name__ == "__main__":
    main()
