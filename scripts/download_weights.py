from pathlib import Path

from huggingface_hub import snapshot_download

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "qwen-inference" / "qwen-weights"

snapshot_download("Qwen/Qwen3.5-4B", local_dir=str(OUTPUT_DIR))
