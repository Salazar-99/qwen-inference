"""Docker entrypoint for the Qwen inference server."""

from __future__ import annotations

import json
import os
from pathlib import Path

import uvicorn

from qwen_inference.loader import DEFAULT_MODEL_DIR, load_weights
from qwen_inference.server import app, configure
from qwen_inference.tokenizer import Tokenizer

DEFAULT_HOST = os.environ.get("HOST", "0.0.0.0")
DEFAULT_PORT = int(os.environ.get("PORT", "8080"))


def main() -> None:
    model_dir = Path(DEFAULT_MODEL_DIR)
    with open(model_dir / "config.json") as config_file:
        config = json.load(config_file)

    weights = load_weights(config, model_dir)
    tokenizer = Tokenizer(model_dir)
    configure(weights, tokenizer)

    uvicorn.run(app, host=DEFAULT_HOST, port=DEFAULT_PORT, log_level="info")


if __name__ == "__main__":
    main()
