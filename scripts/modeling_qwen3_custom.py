"""Custom vLLM model classes for Qwen3.5 with a quantized lm_head.

vLLM's stock Qwen3.5 path ties lm_head to VocabParallelEmbedding, which has no
weight_scale slot and cannot load compressed-tensors INT4 weights.  This module
replaces lm_head with ColumnParallelLinear so the Marlin W4A16 kernel used by
every other linear layer also serves the 152k-vocab projection.

Place this file in the checkpoint directory and set config.json architectures to
["Qwen3_5ForConditionalGenerationCustom"].  The server registers the class via
ModelRegistry before constructing the vLLM engine.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
)
from vllm.model_executor.models.qwen3_vl import Qwen3_VisionTransformer
from vllm.model_executor.models.utils import AutoWeightsLoader, maybe_prefix
from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5Config


class Qwen3_5ForCausalLMCustom(Qwen3_5ForCausalLM):
    """Qwen3.5 language model with a quantized ColumnParallelLinear lm_head."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_text_config
        if not get_pp_group().is_last_rank:
            return

        # Parent may have tied lm_head to embed_tokens; replace with a quantized
        # linear that exposes weight_scale / weight_packed like every other layer.
        self.lm_head = ColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=config.vocab_size,
            bias=False,
            gather_output=False,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights, ensuring quantized lm_head tensors are not skipped."""
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["mtp."],
        )
        return loader.load_weights(weights)


class Qwen3_5ForConditionalGenerationCustom(Qwen3_5ForConditionalGeneration):
    """Multimodal Qwen3.5 wrapper that uses Qwen3_5ForCausalLMCustom."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model") -> None:
        # Qwen3_5ForConditionalGeneration mixes in protocol classes without
        # nn.Module.__init__; mirror the stock constructor but swap the LM class.
        nn.Module.__init__(self)
        self.update_packed_mapping(enable_lora=vllm_config.lora_config is not None)
        config: Qwen3_5Config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        self.is_multimodal_pruning_enabled = False

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Qwen3_VisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "visual"),
            )

        with self._mark_language_model(vllm_config):
            self.language_model = Qwen3_5ForCausalLMCustom(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["mtp."],
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
