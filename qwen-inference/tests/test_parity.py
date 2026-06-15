from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen_inference.backends.custom.loader import DEFAULT_MODEL_DIR, load_weights


def _hf_state_dict_from_loaded_weights(
    loaded_weights: dict[str, torch.Tensor],
    config: dict,
) -> dict[str, torch.Tensor]:
    cfg = config.get("text_config", config)
    state_dict = {
        "model.embed_tokens.weight": loaded_weights["embed_tokens"],
        "model.norm.weight": loaded_weights["norm"],
        "lm_head.weight": loaded_weights["embed_tokens"],
    }

    for idx, layer_type in enumerate(cfg["layer_types"]):
        state_dict[f"model.layers.{idx}.input_layernorm.weight"] = loaded_weights[
            f"layer.{idx}.input_norm"
        ]
        state_dict[f"model.layers.{idx}.post_attention_layernorm.weight"] = loaded_weights[
            f"layer.{idx}.post_norm"
        ]

        if layer_type == "full_attention":
            fused_qkv = loaded_weights[f"layer.{idx}.attn.fused_qkv"]
            kv_size = cfg["num_key_value_heads"] * cfg["head_dim"]
            q_size = fused_qkv.shape[0] - (2 * kv_size)
            q, k, v = fused_qkv.split([q_size, kv_size, kv_size], dim=0)

            state_dict[f"model.layers.{idx}.self_attn.q_proj.weight"] = q
            state_dict[f"model.layers.{idx}.self_attn.k_proj.weight"] = k
            state_dict[f"model.layers.{idx}.self_attn.v_proj.weight"] = v
            state_dict[f"model.layers.{idx}.self_attn.o_proj.weight"] = loaded_weights[
                f"layer.{idx}.attn.o_proj"
            ]
            state_dict[f"model.layers.{idx}.self_attn.q_norm.weight"] = loaded_weights[
                f"layer.{idx}.attn.q_norm"
            ]
            state_dict[f"model.layers.{idx}.self_attn.k_norm.weight"] = loaded_weights[
                f"layer.{idx}.attn.k_norm"
            ]
        else:
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
                state_dict[f"model.layers.{idx}.linear_attn.{name}"] = loaded_weights[
                    f"layer.{idx}.linear_attn.{name}"
                ]

        gate, up = loaded_weights[f"layer.{idx}.mlp.gate_up_proj"].split(
            cfg["intermediate_size"],
            dim=0,
        )
        state_dict[f"model.layers.{idx}.mlp.gate_proj.weight"] = gate
        state_dict[f"model.layers.{idx}.mlp.up_proj.weight"] = up
        state_dict[f"model.layers.{idx}.mlp.down_proj.weight"] = loaded_weights[
            f"layer.{idx}.mlp.down_proj"
        ]

    return state_dict


@pytest.mark.gpu
@pytest.mark.integration
def test_load_weights_matches_transformers_logits() -> None:
    weights_dir = Path(DEFAULT_MODEL_DIR)
    if not weights_dir.exists():
        pytest.skip(f"Model weights not found at {weights_dir}")
    if not any(weights_dir.glob("model*.safetensors")):
        pytest.skip(f"No safetensors shards found at {weights_dir}")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required because load_weights moves tensors to GPU")

    with open(weights_dir / "config.json") as config_file:
        config = json.load(config_file)

    tokenizer = AutoTokenizer.from_pretrained(weights_dir)
    inputs = tokenizer("loader parity check", return_tensors="pt").to("cuda")

    model = AutoModelForCausalLM.from_pretrained(
        weights_dir,
        dtype=torch.bfloat16,
    ).to("cuda")
    model.eval()

    with torch.inference_mode():
        transformers_logits = model(**inputs, use_cache=False).logits.detach().cpu()

        loaded_weights = load_weights(config, weights_dir)
        loaded_state_dict = _hf_state_dict_from_loaded_weights(loaded_weights, config)
        missing_keys = set(model.state_dict()) - set(loaded_state_dict)
        assert not missing_keys

        model.load_state_dict(loaded_state_dict, strict=True)
        loader_logits = model(**inputs, use_cache=False).logits.detach().cpu()

    torch.testing.assert_close(loader_logits, transformers_logits, rtol=0, atol=0)
