"""vLLM general plugin: register custom Qwen3.5 lm_head model classes.

Loaded in every vLLM process (API server, engine core, and GPU workers) via
the ``vllm.general_plugins`` entry point, so ModelRegistry overrides survive
the spawn-based worker model.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

CUSTOM_MODEL_FILENAME = "modeling_qwen3_custom.py"
CUSTOM_MODEL_ARCHITECTURES = frozenset(
    {
        "Qwen3_5ForConditionalGenerationCustom",
        "Qwen3ForCausalLMCustom",
    }
)


def register_custom_vllm_models() -> None:
    model_dir = os.environ.get("MODEL_DIR", "").strip()
    if not model_dir:
        return

    model_path = Path(model_dir)
    config_path = model_path / "config.json"
    if not config_path.exists():
        return

    with open(config_path) as config_file:
        config = json.load(config_file)

    architectures = config.get("architectures") or []
    if not any(arch in CUSTOM_MODEL_ARCHITECTURES for arch in architectures):
        return

    custom_model_path = model_path / CUSTOM_MODEL_FILENAME
    if not custom_model_path.exists():
        logger.warning(
            "Checkpoint declares custom architecture %s but is missing %s",
            architectures,
            CUSTOM_MODEL_FILENAME,
        )
        return

    model_dir_str = str(model_path.resolve())
    if model_dir_str not in sys.path:
        sys.path.insert(0, model_dir_str)

    from modeling_qwen3_custom import Qwen3_5ForConditionalGenerationCustom
    from vllm.model_executor.models import ModelRegistry

    for arch in architectures:
        if arch in CUSTOM_MODEL_ARCHITECTURES:
            ModelRegistry.register_model(arch, Qwen3_5ForConditionalGenerationCustom)
            logger.info("Registered custom vLLM model architecture: %s", arch)
