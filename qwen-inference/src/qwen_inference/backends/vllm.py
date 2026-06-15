"""vLLM inference backend."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import time
from collections.abc import Iterator
from typing import Any

import torch
from fastapi import HTTPException

from qwen_inference.types import ChatMessage


def _build_profiler_config(profile_dir: str) -> Any:
    config_kwargs = {
        "profiler": "torch",
        "torch_profiler_dir": profile_dir,
        "delay_iterations": 0,
        "max_iterations": 0,
    }
    try:
        from vllm.config import ProfilerConfig

        return ProfilerConfig(**config_kwargs)
    except ImportError:
        return config_kwargs


def _count_prompt_tokens(tokenizer: Any, prompt: str) -> int:
    if hasattr(tokenizer, "encode"):
        return len(tokenizer.encode(prompt))
    return len(tokenizer(prompt)["input_ids"])


def _is_trace_file(path: Path) -> bool:
    name = path.name
    return (
        path.suffix == ".json"
        or name.endswith(".trace.json")
        or name.endswith(".pt.trace.json")
        or name.endswith(".pt.trace.json.gz")
    )


def _export_latest_trace(profiler_dir: Path, trace_path: Path) -> None:
    trace_candidates = sorted(
        (
            path
            for path in profiler_dir.rglob("*")
            if path.is_file() and _is_trace_file(path)
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not trace_candidates:
        raise HTTPException(
            status_code=500,
            detail=f"No torch profiler trace found under {profiler_dir}",
        )

    trace_path.parent.mkdir(parents=True, exist_ok=True)
    source = trace_candidates[0]
    if source.name.endswith(".gz"):
        import gzip

        with gzip.open(source, "rb") as compressed, trace_path.open("wb") as output:
            shutil.copyfileobj(compressed, output)
    else:
        shutil.copy2(source, trace_path)


@contextmanager
def _nsys_capture(enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return

    if not torch.cuda.is_available():
        yield
        return

    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    try:
        yield
        torch.cuda.synchronize()
    finally:
        torch.cuda.cudart().cudaProfilerStop()


class VllmBackend:
    def __init__(self, model_dir: str) -> None:
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise HTTPException(
                status_code=503,
                detail=(
                    "vLLM is not installed. Install dependencies with "
                    "'UV_TORCH_BACKEND=auto uv sync --package qwen-inference --group dev --group vllm'."
                ),
            ) from exc

        self._sampling_params_cls = SamplingParams
        self._profiler_dir = Path(
            os.environ.get("VLLM_PROFILER_DIR", "/tmp/vllm-profiler"),
        )
        self._profiler_dir.mkdir(parents=True, exist_ok=True)
        self.llm = LLM(
            model=model_dir,
            dtype="bfloat16",
            trust_remote_code=True,
            profiler_config=_build_profiler_config(str(self._profiler_dir)),
        )
        self.tokenizer = self.llm.get_tokenizer()

    def _sampling_params(self, *, max_tokens: int, temperature: float) -> Any:
        kwargs: dict[str, Any] = {"max_tokens": max_tokens}
        if temperature > 0:
            kwargs["temperature"] = temperature
        else:
            kwargs["temperature"] = 0.0
        return self._sampling_params_cls(**kwargs)

    def _profile_sampling_params(self, *, decode_steps: int) -> Any:
        decode_steps = max(0, decode_steps)
        if decode_steps == 0:
            return self._sampling_params_cls(
                temperature=0.0,
                max_tokens=1,
                min_tokens=0,
            )

        return self._sampling_params_cls(
            temperature=0.0,
            max_tokens=decode_steps,
            min_tokens=decode_steps,
            ignore_eos=True,
        )

    def _generate_text(self, prompt: str, *, max_tokens: int, temperature: float) -> str:
        outputs = self.llm.generate(
            [prompt],
            self._sampling_params(max_tokens=max_tokens, temperature=temperature),
        )
        return outputs[0].outputs[0].text

    def _run_profiled_generate(self, prompt: str, *, decode_steps: int) -> None:
        self.llm.generate(
            [prompt],
            self._profile_sampling_params(decode_steps=decode_steps),
        )

    def generate_from_prompt(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
    ) -> str:
        return self._generate_text(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def generate_from_messages(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int,
        temperature: float,
        thinking: bool | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> str:
        template_kwargs = dict(chat_template_kwargs or {})
        if thinking is not None:
            template_kwargs["enable_thinking"] = thinking

        prompt = self.tokenizer.apply_chat_template(
            [message.model_dump() for message in messages],
            add_generation_prompt=True,
            tokenize=False,
            **template_kwargs,
        )
        return self._generate_text(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def profile_forward_passes(
        self,
        prompt: str,
        *,
        decode_steps: int,
        profile: bool = False,
        trace_path: str | None = None,
    ) -> dict[str, Any]:
        decode_steps = max(0, decode_steps)
        prompt_tokens = _count_prompt_tokens(self.tokenizer, prompt)

        if trace_path:
            trace_output = Path(trace_path)
            profile_prefix = trace_output.stem
            self.llm.start_profile(profile_prefix=profile_prefix)
            try:
                self._run_profiled_generate(prompt, decode_steps=decode_steps)
            finally:
                self.llm.stop_profile()
                time.sleep(1)
            _export_latest_trace(self._profiler_dir, trace_output)
        elif profile:
            with _nsys_capture(True):
                self._run_profiled_generate(prompt, decode_steps=decode_steps)
        else:
            self._run_profiled_generate(prompt, decode_steps=decode_steps)

        return {
            "prompt_tokens": prompt_tokens,
            "prefill_forwards": 1,
            "decode_forwards": decode_steps,
            "profiler": "vllm-torch" if trace_path else "vllm-nsys" if profile else "none",
        }
