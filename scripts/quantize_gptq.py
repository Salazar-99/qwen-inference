"""Quantize Qwen3.5-4B with llm-compressor.

Two schemes are supported via --scheme:

  W4A16 (default) — GPTQ weight-only INT4, 16-bit activations.
    Served by vLLM through the Marlin INT4 kernel on Ampere (A10G / sm_86).
    Best for decode-bound workloads (short category): cuts the bytes read per
    weight by 4×, which is the bottleneck when batch=1 GEMVs re-read all
    weights every token.

  W8A8 — SmoothQuant INT8 weights + INT8 activations.
    Served by vLLM through the INT8 tensor-core GEMM path on Ampere.
    Best for prefill-heavy workloads (medium/long categories): activations are
    also quantized so the matmuls run through INT8 tensor cores (~2× GEMM
    throughput vs bf16). W8A8 also halves weight bandwidth vs bf16, so decode
    still improves — just not as much as W4A16.

Run in a DEDICATED, isolated environment -- NOT the uv workspace:

    uv venv .venv-quantize --python 3.12
    UV_TORCH_BACKEND=auto uv pip install --python .venv-quantize \\
        -r scripts/quantize-requirements.txt

    # W4A16 (default)
    .venv-quantize/bin/python scripts/quantize_gptq.py

    # W8A8
    .venv-quantize/bin/python scripts/quantize_gptq.py --scheme W8A8

Output directories:
  W4A16 → qwen-inference/qwen-weights-w4a16
  W8A8  → qwen-inference/qwen-weights-w8a8

Notes:
- `lm_head` is left unquantized by default (quality-sensitive vocab projection);
  the gated-delta-net conv/recurrent kernels are not nn.Linear so they are
  untouched automatically.
- Always validate against the competition quality gates (especially
  GPQA-Diamond) before submitting.
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = REPO_ROOT / "qwen-inference" / "qwen-weights"
DEFAULT_OUTPUT_DIRS = {
    "W4A16": REPO_ROOT / "qwen-inference" / "qwen-weights-w4a16",
    "W8A8": REPO_ROOT / "qwen-inference" / "qwen-weights-w8a8",
}
DEFAULT_OUTPUT_DIRS_LM_HEAD = {
    "W4A16": REPO_ROOT / "qwen-inference" / "qwen-weights-w4a16-lmhead",
    "W8A8": REPO_ROOT / "qwen-inference" / "qwen-weights-w8a8-lmhead",
}
CUSTOM_MODEL_FILENAME = "modeling_qwen3_custom.py"
CUSTOM_MODEL_CLASS = "Qwen3_5ForConditionalGenerationCustom"
CUSTOM_MODEL_TEMPLATE = Path(__file__).resolve().parent / CUSTOM_MODEL_FILENAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Path to the source (bf16) weights.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Where to write the quantized checkpoint (default: scheme-specific dir).")
    parser.add_argument(
        "--scheme",
        choices=["W4A16", "W8A8"],
        default="W4A16",
        help="Quantization scheme. W4A16 uses GPTQ weight-only INT4 (best for decode). W8A8 uses SmoothQuant INT8 weights+activations (best for prefill). Default: W4A16.",
    )
    parser.add_argument("--dataset", default="HuggingFaceH4/ultrachat_200k", help="HF calibration dataset.")
    parser.add_argument("--dataset-split", default="train_sft", help="Dataset split to sample from.")
    parser.add_argument("--num-samples", type=int, default=512, help="Number of calibration samples.")
    parser.add_argument("--seq-len", type=int, default=2048, help="Max calibration sequence length.")
    parser.add_argument(
        "--ignore",
        nargs="*",
        default=["lm_head", "re:.*visual.*", "re:.*mtp.*"],
        help=(
            "Module names/regexes to leave in full precision. Defaults skip the "
            "lm_head, the vision tower (model.visual.*), and the MTP head "
            "(mtp.*): the vision/MTP layers receive no activations from text-only "
            "calibration, so quantizing them would be meaningless or fail."
        ),
    )
    parser.add_argument(
        "--quantize-lm-head",
        action="store_true",
        default=False,
        help=(
            "Also quantize the lm_head (152k-vocab projection, 13.5%% of decode time). "
            "Excluded by default: vLLM implements lm_head as VocabParallelEmbedding which "
            "has no weight_scale slot, so the checkpoint fails to load. Only enable if "
            "using a custom loader that supports quantized lm_head."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Calibration sampling seed.")
    # W8A8 / SmoothQuant knob — controls how aggressively outliers are migrated
    # from activations to weights before INT8 quantization. 0.8 is the value
    # recommended in the SmoothQuant paper for most transformer models.
    parser.add_argument("--smoothing-strength", type=float, default=0.8, help="SmoothQuant migration strength α (W8A8 only). Default: 0.8.")
    return parser.parse_args()


def build_calibration_dataset(dataset: str, split: str, num_samples: int, seq_len: int, seed: int, tokenizer):
    from datasets import load_dataset

    ds = load_dataset(dataset, split=split)
    ds = ds.shuffle(seed=seed).select(range(min(num_samples, len(ds))))

    def to_text(example):
        messages = example.get("messages")
        if messages is not None:
            text = tokenizer.apply_chat_template(messages, tokenize=False)
        else:
            text = example.get("text", "")
        return {"text": text}

    def tokenize(example):
        return tokenizer(
            example["text"],
            padding=False,
            truncation=True,
            max_length=seq_len,
            add_special_tokens=False,
        )

    ds = ds.map(to_text)
    ds = ds.map(tokenize, remove_columns=ds.column_names)
    return ds


def build_recipe_w4a16(ignore: list[str]):
    from llmcompressor.modifiers.quantization import GPTQModifier

    return GPTQModifier(targets="Linear", scheme="W4A16", ignore=ignore)


def build_recipe_w8a8(model, ignore: list[str], smoothing_strength: float):
    from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
    from llmcompressor.modifiers.quantization import QuantizationModifier

    # Qwen3_5ForConditionalGeneration is not in llm-compressor's architecture
    # registry, so the default mapping resolution falls back to global regexes
    # (re:.*input_layernorm) that match all 32 layers at once, causing:
    #   "SmoothQuant must match a single smooth layer for each mapping"
    # Fix: enumerate layers from the loaded model and generate one explicit
    # (balance_layers, smooth_layer) pair per transformer block.
    # Locate the transformer layer list via named_modules so the prefix exactly
    # matches what llm-compressor sees (avoids hardcoding nesting depth).
    # Qwen3.5 multimodal: model.model.language_model.layers → named path "model.language_model.layers"
    import torch.nn as nn
    layers = None
    prefix_base = None
    for mod_name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) > 1:
            first = mod[0]
            if hasattr(first, "self_attn") or hasattr(first, "mlp"):
                layers = mod
                prefix_base = mod_name  # e.g. "model.language_model.layers"
                break
    if layers is None:
        raise RuntimeError("Could not find transformer layers ModuleList in model.")

    mappings = []
    for i, layer in enumerate(layers):
        p = f"{prefix_base}.{i}"
        # Full-attention layers have q/k/v_proj; GDN (linear-attn) layers do not.
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "q_proj"):
            mappings.append((
                [f"{p}.self_attn.q_proj", f"{p}.self_attn.k_proj", f"{p}.self_attn.v_proj"],
                f"{p}.input_layernorm",
            ))
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate_proj"):
            mappings.append((
                [f"{p}.mlp.gate_proj", f"{p}.mlp.up_proj"],
                f"{p}.post_attention_layernorm",
            ))

    print(f"  SmoothQuant: generated {len(mappings)} explicit per-layer mappings.")
    # SmoothQuant migrates activation outliers into weights before quantization,
    # making both sides amenable to INT8 rounding. QuantizationModifier then
    # applies RTN W8A8 (static-per-tensor weights, dynamic-per-token activations),
    # which vLLM dispatches to INT8 cuBLAS / tensor-core GEMMs on Ampere.
    return [
        SmoothQuantModifier(smoothing_strength=smoothing_strength, mappings=mappings),
        QuantizationModifier(targets="Linear", scheme="W8A8", ignore=ignore),
    ]


def _patch_config_for_quantized_lm_head(config_path: Path) -> None:
    """Update config.json for the custom vLLM lm_head loader."""
    import json

    with open(config_path) as config_file:
        config = json.load(config_file)

    config["architectures"] = [CUSTOM_MODEL_CLASS]
    config["tie_word_embeddings"] = False
    if "text_config" in config:
        config["text_config"]["tie_word_embeddings"] = False

    quant_cfg = config.get("quantization_config") or {}
    ignore = quant_cfg.get("ignore") or []
    quant_cfg["ignore"] = [entry for entry in ignore if entry != "lm_head"]
    config["quantization_config"] = quant_cfg

    with open(config_path, "w") as config_file:
        json.dump(config, config_file, indent=2)
        config_file.write("\n")


def _install_custom_model_file(output_dir: Path) -> None:
    import shutil

    if not CUSTOM_MODEL_TEMPLATE.exists():
        raise SystemExit(f"Missing custom model template: {CUSTOM_MODEL_TEMPLATE}")

    dst = output_dir / CUSTOM_MODEL_FILENAME
    shutil.copy2(CUSTOM_MODEL_TEMPLATE, dst)
    print(f"  installed custom vLLM model: {dst.name}")


def main() -> None:
    args = parse_args()

    model_dir = args.model_dir.resolve()
    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
    elif args.quantize_lm_head:
        output_dir = DEFAULT_OUTPUT_DIRS_LM_HEAD[args.scheme].resolve()
    else:
        output_dir = DEFAULT_OUTPUT_DIRS[args.scheme].resolve()

    if not model_dir.exists():
        raise SystemExit(
            f"Model weights not found at {model_dir}. "
            "Run scripts/download_weights.py first or pass --model-dir."
        )

    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoTokenizer,
    )
    from llmcompressor import oneshot

    print(f"Loading model from {model_dir} ...")
    load_kwargs = dict(dtype="auto", device_map="auto", trust_remote_code=True)
    # Qwen3.5 is a multimodal Qwen3_5ForConditionalGeneration (text tower +
    # vision tower + MTP head). It MUST be loaded as the full model so the saved
    # checkpoint keeps the multimodal config and `model.language_model.*` /
    # `model.visual.*` / `mtp.*` weight names that vLLM expects. Loading it as a
    # plain CausalLM flattens the config and drops the vision/MTP weights, which
    # vLLM then refuses to serve.
    try:
        model = AutoModelForImageTextToText.from_pretrained(str(model_dir), **load_kwargs)
    except (ValueError, KeyError, OSError) as exc:
        print(f"  AutoModelForImageTextToText failed ({exc}); falling back to CausalLM.")
        model = AutoModelForCausalLM.from_pretrained(str(model_dir), **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)

    print(f"Building {args.num_samples} calibration samples from {args.dataset} ...")
    calibration_ds = build_calibration_dataset(
        args.dataset,
        args.dataset_split,
        args.num_samples,
        args.seq_len,
        args.seed,
        tokenizer,
    )

    ignore = list(args.ignore)
    if args.quantize_lm_head and "lm_head" in ignore:
        ignore.remove("lm_head")
        print("lm_head will be quantized (--quantize-lm-head set).")

    if args.scheme == "W4A16":
        recipe = build_recipe_w4a16(ignore)
        print("Running GPTQ W4A16 quantization (this can take a while) ...")
    else:
        recipe = build_recipe_w8a8(model, ignore, args.smoothing_strength)
        print(f"Running SmoothQuant W8A8 quantization (α={args.smoothing_strength}, this can take a while) ...")

    oneshot(
        model=model,
        dataset=calibration_ds,
        recipe=recipe,
        max_seq_length=args.seq_len,
        num_calibration_samples=args.num_samples,
    )

    print(f"Saving compressed checkpoint to {output_dir} ...")
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir), save_compressed=True)
    tokenizer.save_pretrained(str(output_dir))

    # Qwen3.5 is multimodal, so vLLM builds an image/video processor at load time
    # and needs the processor configs that save_pretrained does not emit. Copy any
    # non-weight aux files from the source dir that are missing in the output.
    import shutil

    aux_files = [
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "processor_config.json",
        "merges.txt",
        "vocab.json",
    ]
    for name in aux_files:
        src = model_dir / name
        dst = output_dir / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            print(f"  copied aux file: {name}")

    if args.quantize_lm_head:
        _patch_config_for_quantized_lm_head(output_dir / "config.json")
        _install_custom_model_file(output_dir)
        print(
            f"Patched config for {CUSTOM_MODEL_CLASS} "
            "(tie_word_embeddings=false, lm_head removed from ignore)."
        )

    print("Done. Serve with vLLM by pointing the model path at:")
    print(f"  {output_dir}")


if __name__ == "__main__":
    main()
