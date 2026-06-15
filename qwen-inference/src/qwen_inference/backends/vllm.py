"""vLLM inference backend."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
import inspect
import logging
import os
from pathlib import Path
import shutil
import time
import uuid
from collections.abc import Iterator
from typing import Any

import torch
from fastapi import HTTPException

from qwen_inference.types import ChatMessage

logger = logging.getLogger(__name__)


async def _maybe_await(value: Any) -> Any:
    """Await ``value`` if it is awaitable, else return it as-is.

    vLLM's async engine methods (``get_tokenizer``, ``start_profile`` …) are
    coroutines on the V1 engine but plain calls on some builds; this keeps the
    call sites agnostic.
    """
    if inspect.isawaitable(value):
        return await value
    return value


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


# Generic geometric sweep of prompt lengths (in tokens) to warm before serving.
# Spans the serving range so the per-shape Triton autotune + JIT compile and
# CUDA-graph capture for the gated-delta-net (fla) attention path happen at
# startup rather than on the first request. This covers both the latency
# benchmark sizes (64/2048/8192) and the quality-eval prompt lengths in between,
# without being tied to any specific benchmark's exact values.
WARMUP_PROMPT_TOKENS = (64, 256, 1024, 4096, 8192)

# Same filler string the benchmark uses (~10 tokens per repetition), so the
# prompts we build tokenize to the same lengths the benchmark drives.
_WARMUP_FILLER = "The quick brown fox jumps over the lazy dog. "


def _warmup_prompt_token_counts() -> list[int]:
    raw = os.environ.get("VLLM_WARMUP_PROMPT_TOKENS")
    if not raw:
        return list(WARMUP_PROMPT_TOKENS)
    counts = [int(part) for part in raw.split(",") if part.strip()]
    return counts or list(WARMUP_PROMPT_TOKENS)


def _build_async_engine(model_dir: str) -> Any:
    """Construct the vLLM async engine (continuous batching across requests).

    This deployment runs the V1 engine (see ``vllm/v1/...`` in the profiler
    traces), whose async front-end is ``AsyncLLM``. We fall back to the legacy
    ``AsyncLLMEngine`` shim for older builds. Must be called from within a
    running event loop so the engine can start its background output handler.
    """
    try:
        from vllm import AsyncEngineArgs
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "vLLM is not installed. Install dependencies with "
                "'UV_TORCH_BACKEND=auto uv sync --package qwen-inference --group dev --group vllm'."
            ),
        ) from exc

    max_num_seqs = max(1, int(os.environ.get("VLLM_MAX_NUM_SEQS", "16")))
    engine_args = AsyncEngineArgs(
        model=model_dir,
        dtype="bfloat16",
        trust_remote_code=True,
        max_num_seqs=max_num_seqs,
    )

    try:
        from vllm.v1.engine.async_llm import AsyncLLM

        return AsyncLLM.from_engine_args(engine_args)
    except ImportError:
        from vllm import AsyncLLMEngine

        return AsyncLLMEngine.from_engine_args(engine_args)


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
    """Async vLLM backend.

    Built with :meth:`create` from inside the server's event loop so the single
    shared ``AsyncLLM`` engine drives continuous batching: concurrent requests
    are concurrent coroutines on one loop, co-scheduled into the same decode
    batch instead of serializing.
    """

    def __init__(self, engine: Any, tokenizer: Any, sampling_params_cls: Any, profiler_dir: Path) -> None:
        self._engine = engine
        self.tokenizer = tokenizer
        self._sampling_params_cls = sampling_params_cls
        self._profiler_dir = profiler_dir

    @classmethod
    async def create(cls, model_dir: str) -> "VllmBackend":
        try:
            from vllm import SamplingParams
        except ImportError as exc:
            raise HTTPException(
                status_code=503,
                detail=(
                    "vLLM is not installed. Install dependencies with "
                    "'UV_TORCH_BACKEND=auto uv sync --package qwen-inference --group dev --group vllm'."
                ),
            ) from exc

        profiler_dir = Path(os.environ.get("VLLM_PROFILER_DIR", "/tmp/vllm-profiler"))
        profiler_dir.mkdir(parents=True, exist_ok=True)
        # The async/server torch profiler is enabled via env var; it must be set
        # before the engine is constructed for start_profile()/stop_profile().
        os.environ.setdefault("VLLM_TORCH_PROFILER_DIR", str(profiler_dir))

        engine = _build_async_engine(model_dir)
        tokenizer = await _maybe_await(engine.get_tokenizer())
        backend = cls(engine, tokenizer, SamplingParams, profiler_dir)

        if _env_flag("VLLM_WARMUP", default=True):
            await backend._warmup()
        return backend

    async def _agenerate(self, prompt: str, params: Any) -> str:
        """Drive one request through the async engine to completion."""
        request_id = f"qwen-{uuid.uuid4().hex}"
        final = None
        async for output in self._engine.generate(prompt, params, request_id):
            final = output
        if final is None or not final.outputs:
            return ""
        return final.outputs[0].text

    async def _warmup(self) -> None:
        """Run representative generations so kernels are compiled before serving.

        The first real forward pass otherwise pays a large one-time cost:
        Triton JIT compilation + autotuning of the gated-delta-net (fla) linear
        attention kernels, torch._dynamo AOT compilation, and CUDA-graph capture.
        These costs are per prefill shape, so we warm a geometric sweep of prompt
        lengths across the serving range (see WARMUP_PROMPT_TOKENS) and force a few
        decode steps so the decode-path kernels compile too. Running them
        concurrently also exercises the batched decode path. The idle time lands
        here at startup (before the server is ready) instead of on the first
        request.
        """
        decode_steps = max(1, int(os.environ.get("VLLM_WARMUP_DECODE_STEPS", "8")))
        token_counts = _warmup_prompt_token_counts()
        # Mirror the benchmark's prompt construction (~10 tokens per filler repeat)
        # so each prompt tokenizes to its target length.
        prompts = [
            _WARMUP_FILLER * max(1, num_tokens // 10) for num_tokens in token_counts
        ]
        params = self._sampling_params_cls(
            temperature=0.0,
            max_tokens=decode_steps,
            min_tokens=decode_steps,
            ignore_eos=True,
        )

        start = time.perf_counter()
        try:
            await asyncio.gather(*(self._agenerate(p, params) for p in prompts))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:  # noqa: BLE001 - warmup must never block serving
            logger.exception("vLLM warmup failed; first request may be slow")
            return
        logger.info(
            "vLLM warmup complete in %.2fs (prompt tokens=%s, %d decode steps each)",
            time.perf_counter() - start,
            token_counts,
            decode_steps,
        )

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

    async def generate_from_prompt(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
    ) -> str:
        params = self._sampling_params(max_tokens=max_tokens, temperature=temperature)
        return await self._agenerate(prompt, params)

    async def generate_from_messages(
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
        params = self._sampling_params(max_tokens=max_tokens, temperature=temperature)
        return await self._agenerate(prompt, params)

    async def profile_forward_passes(
        self,
        prompt: str,
        *,
        decode_steps: int,
        profile: bool = False,
        trace_path: str | None = None,
    ) -> dict[str, Any]:
        decode_steps = max(0, decode_steps)
        prompt_tokens = _count_prompt_tokens(self.tokenizer, prompt)
        params = self._profile_sampling_params(decode_steps=decode_steps)

        if trace_path:
            trace_output = Path(trace_path)
            await _maybe_await(self._engine.start_profile())
            try:
                await self._agenerate(prompt, params)
            finally:
                await _maybe_await(self._engine.stop_profile())
                await asyncio.sleep(1)
            _export_latest_trace(self._profiler_dir, trace_output)
        elif profile:
            with _nsys_capture(True):
                await self._agenerate(prompt, params)
        else:
            await self._agenerate(prompt, params)

        return {
            "prompt_tokens": prompt_tokens,
            "prefill_forwards": 1,
            "decode_forwards": decode_steps,
            "profiler": "vllm-torch" if trace_path else "vllm-nsys" if profile else "none",
        }
