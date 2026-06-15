"""Quantize Qwen3.5-4B to INT4 (GPTQ W4A16) with llm-compressor.

This produces a `compressed-tensors` checkpoint that vLLM serves through the
Marlin INT4 kernel on Ampere (A10G / sm_86), which is the highest-leverage win
for the decode-bound short category (weights are the bottleneck, not compute).

Run it in a DEDICATED, isolated environment -- NOT the uv workspace. The
`llmcompressor` line that supports transformers 5.x (Qwen3.5) needs
`compressed-tensors>=0.17.2a2`, which conflicts with the `compressed-tensors`
version pinned by `vllm==0.19.1` in the server env. Quantization is a one-shot
offline step and does not need `vllm`, so keep it separate:

    uv venv .venv-quantize --python 3.12
    UV_TORCH_BACKEND=auto uv pip install --python .venv-quantize \
        -r scripts/quantize-requirements.txt
    .venv-quantize/bin/python scripts/quantize_gptq.py

The isolated env still uses the git build of `transformers` (5.13.x dev), which
is required to load the Qwen3.5 hybrid architecture for the calibration pass.

By default it reads weights from `qwen-inference/qwen-weights` and writes the
quantized checkpoint to `qwen-inference/qwen-weights-quantized`.

Notes:
- W4A16 = 4-bit weights, 16-bit activations (weight-only). This is the right
  scheme for Ampere, which has no FP8 tensor cores. It maximizes the decode
  speedup; for prefill-heavy categories also evaluate INT8 W8A8.
- `lm_head` is left unquantized by default (quality-sensitive vocab projection);
  the gated-delta-net conv/recurrent kernels are not `nn.Linear` so they are
  untouched automatically.
- Always validate against the competition quality gates (especially
  GPQA-Diamond) before submitting.
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = REPO_ROOT / "qwen-inference" / "qwen-weights"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "qwen-inference" / "qwen-weights-quantized"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Path to the source (bf16) weights.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Where to write the INT4 checkpoint.")
    parser.add_argument("--dataset", default="HuggingFaceH4/ultrachat_200k", help="HF calibration dataset.")
    parser.add_argument("--dataset-split", default="train_sft", help="Dataset split to sample from.")
    parser.add_argument("--num-samples", type=int, default=512, help="Number of calibration samples.")
    parser.add_argument("--seq-len", type=int, default=2048, help="Max calibration sequence length.")
    parser.add_argument(
        "--ignore",
        nargs="*",
        default=["lm_head"],
        help="Module names/regexes to leave in full precision.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Calibration sampling seed.")
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


def main() -> None:
    args = parse_args()

    model_dir = args.model_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not model_dir.exists():
        raise SystemExit(
            f"Model weights not found at {model_dir}. "
            "Run scripts/download_weights.py first or pass --model-dir."
        )

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier

    print(f"Loading model from {model_dir} ...")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
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

    recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=list(args.ignore))

    print("Running GPTQ W4A16 quantization (this can take a while) ...")
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
    print("Done. Serve with vLLM by pointing the model path at:")
    print(f"  {output_dir}")


if __name__ == "__main__":
    main()
