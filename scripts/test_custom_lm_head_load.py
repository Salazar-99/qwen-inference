#!/usr/bin/env python3
"""Smoke-test vLLM loading of a custom lm_head checkpoint."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "qwen-inference" / "src"))

os.environ.setdefault("VLLM_WARMUP", "1")
os.environ.setdefault("VLLM_GPU_MEMORY_UTILIZATION", "0.85")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=REPO_ROOT / "qwen-inference" / "qwen-weights-w4a16-lmhead",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    model_dir = str(args.model_dir.resolve())
    os.environ["MODEL_DIR"] = model_dir

    from qwen_inference.backends.vllm import (
        VllmBackend,
    )

    backend = await VllmBackend.create(model_dir)
    text = await backend.generate_from_prompt(
        "Hello",
        max_tokens=4,
        temperature=0.0,
    )
    print("generation:", repr(text))


if __name__ == "__main__":
    asyncio.run(main())
