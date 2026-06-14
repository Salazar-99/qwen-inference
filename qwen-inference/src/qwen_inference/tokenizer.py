"""HuggingFace fast (Rust) tokenizer for Qwen3.5-4B."""

from __future__ import annotations

import os
from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizerBase

DEFAULT_MODEL_DIR = os.environ.get(
    "MODEL_DIR",
    str(Path(__file__).resolve().parents[2] / "qwen-weights"),
)


class Tokenizer:
    """Qwen3.5-4B tokenizer backed by HuggingFace's Rust tokenizers library."""

    def __init__(self, model_dir: str | Path | None = None) -> None:
        path = Path(model_dir or DEFAULT_MODEL_DIR)
        if not path.is_dir():
            raise FileNotFoundError(f"Tokenizer directory not found: {path}")

        self._tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(path)

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = False) -> str:
        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
