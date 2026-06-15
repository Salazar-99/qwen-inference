"""Custom optimized inference backend package."""

from qwen_inference.backends.custom.backend import CustomBackend
from qwen_inference.backends.custom.loader import DEFAULT_MODEL_DIR, load_weights

__all__ = [
    "CustomBackend",
    "DEFAULT_MODEL_DIR",
    "load_weights",
]
