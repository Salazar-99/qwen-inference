from __future__ import annotations

import os
from pathlib import Path

import torch
from safetensors.torch import load_file

DEFAULT_MODEL_DIR = os.environ.get(
    "MODEL_DIR",
    str(Path(__file__).resolve().parents[4] / "qwen-weights"),
)


def _to_gpu_weight(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.cuda().bfloat16().contiguous()


def load_weights(
    config: dict,
    weights_dir: str | Path,
) -> dict[str, torch.Tensor]:
    """Load model weights from safetensors shards in a directory onto the GPU."""
    cfg = config.get("text_config", config)
    paths = sorted(Path(weights_dir).glob("model*.safetensors"))

    # Parse raw flat safetensors pointers (zero-copy mmap structure)
    raw_weights: dict[str, torch.Tensor] = {}
    for path in paths:
        raw_weights.update(load_file(str(path), device="cpu"))

    def raw_language_weight(name: str) -> torch.Tensor:
        """Fetch a text-model weight from either HF causal-LM or multimodal shard names."""
        direct_key = f"model.{name}"
        multimodal_key = f"model.language_model.{name}"
        if direct_key in raw_weights:
            return raw_weights[direct_key]
        return raw_weights[multimodal_key]

    # Build an optimized dictionary container to hold our final GPU pointers
    gpu_model_weights: dict[str, torch.Tensor] = {}

    # Map input embeddings cleanly
    gpu_model_weights["embed_tokens"] = _to_gpu_weight(
        raw_language_weight("embed_tokens.weight")
    )
    gpu_model_weights["norm"] = _to_gpu_weight(raw_language_weight("norm.weight"))

    # Iterate through the layers specified in the config
    for idx in range(cfg["num_hidden_layers"]):
        layer_type = cfg["layer_types"][idx]

        # Every layer, whether full or linear, utilizes an input layer norm
        # Qwen 3.5 uses RMSNorm. We read it out and flag it as a contiguous chunk.
        gpu_model_weights[f"layer.{idx}.input_norm"] = _to_gpu_weight(
            raw_language_weight(f"layers.{idx}.input_layernorm.weight")
        )
        gpu_model_weights[f"layer.{idx}.post_norm"] = _to_gpu_weight(
            raw_language_weight(f"layers.{idx}.post_attention_layernorm.weight")
        )

        if layer_type == "full_attention":
            # Extract GQA tensors
            q = raw_language_weight(f"layers.{idx}.self_attn.q_proj.weight")
            k = raw_language_weight(f"layers.{idx}.self_attn.k_proj.weight")
            v = raw_language_weight(f"layers.{idx}.self_attn.v_proj.weight")

            # Performance optimization: Fuse QKV into an interconnected block for our Triton GEMM
            gpu_model_weights[f"layer.{idx}.attn.fused_qkv"] = _to_gpu_weight(
                torch.cat([q, k, v], dim=0)
            )
            gpu_model_weights[f"layer.{idx}.attn.o_proj"] = _to_gpu_weight(
                raw_language_weight(f"layers.{idx}.self_attn.o_proj.weight")
            )
            gpu_model_weights[f"layer.{idx}.attn.q_norm"] = _to_gpu_weight(
                raw_language_weight(f"layers.{idx}.self_attn.q_norm.weight")
            )
            gpu_model_weights[f"layer.{idx}.attn.k_norm"] = _to_gpu_weight(
                raw_language_weight(f"layers.{idx}.self_attn.k_norm.weight")
            )

        elif layer_type == "linear_attention":
            # Pull separate weights tailored for the Gated DeltaNet linear attention equations
            for name in (
                "dt_bias",
                "A_log",
                "conv1d.weight",
                "norm.weight",
                "out_proj.weight",
                "in_proj_qkv.weight",
                "in_proj_z.weight",
                "in_proj_b.weight",
                "in_proj_a.weight",
            ):
                gpu_model_weights[f"layer.{idx}.linear_attn.{name}"] = _to_gpu_weight(
                    raw_language_weight(f"layers.{idx}.linear_attn.{name}")
                )

        # Handle the standard SwiGLU MLP layers (common across both block variations)
        # `"hidden_act": "silu"` and `"intermediate_size": 9216` tell us how to dimension the gate
        gpu_model_weights[f"layer.{idx}.mlp.gate_up_proj"] = _to_gpu_weight(
            torch.cat(
                [
                    raw_language_weight(f"layers.{idx}.mlp.gate_proj.weight"),
                    raw_language_weight(f"layers.{idx}.mlp.up_proj.weight"),
                ],
                dim=0,
            )
        )

        gpu_model_weights[f"layer.{idx}.mlp.down_proj"] = _to_gpu_weight(
            raw_language_weight(f"layers.{idx}.mlp.down_proj.weight")
        )

    return gpu_model_weights
