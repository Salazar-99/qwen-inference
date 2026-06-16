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


def _profiler_dir() -> Path:
    profiler_dir = Path(os.environ.get("VLLM_PROFILER_DIR", "/tmp/vllm-profiler"))
    profiler_dir.mkdir(parents=True, exist_ok=True)
    return profiler_dir


def _profiler_config(profiler_dir: Path) -> Any:
    """Torch profiler settings for vLLM's worker-side CUDA traces.

    vLLM reads these via ``ProfilerConfig`` on ``AsyncEngineArgs`` (not env vars).
    ``record_shapes`` / ``profile_memory`` mirror the baseline backend's
    ``torch.profiler.profile`` options so Chrome traces include per-op tensor dims.
    """
    from vllm.config import ProfilerConfig

    return ProfilerConfig(
        profiler="torch",
        torch_profiler_dir=str(profiler_dir),
        torch_profiler_record_shapes=_env_flag(
            "VLLM_TORCH_PROFILER_RECORD_SHAPES", default=True
        ),
        torch_profiler_with_memory=_env_flag(
            "VLLM_TORCH_PROFILER_WITH_MEMORY", default=True
        ),
        # GPU kernels are profiled in the worker; skip AsyncLLM's CPU-only frontend
        # profiler to avoid duplicate traces and extra overhead.
        ignore_frontend=True,
    )


def _build_async_engine(model_dir: str, profiler_dir: Path) -> Any:
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

    # Chunked prefill: disabled by default for single-request latency.
    # When enabled (VLLM_CHUNKED_PREFILL=1) it triggers mixed prefill+decode
    # batching which bypasses the Marlin INT4 GEMM path for prefill chunks,
    # replacing it with cuBLAS bf16 GEMMs. Net effect on single-request short
    # latency: neutral (1158ms vs 1167ms), profile wall time worse (+17%).
    # May benefit multi-request throughput but hurts this competition's metric.
    enable_chunked_prefill = _env_flag("VLLM_CHUNKED_PREFILL", default=False)

    # Speculative decoding: off by default, opt-in via VLLM_SPEC_DECODE.
    #
    #   VLLM_SPEC_DECODE=ngram_gpu  — on-GPU n-gram lookup (no CPU sync).
    #     Finds the trailing n-gram of the current context in the
    #     prompt/output history and proposes the tokens that followed it.
    #     VLLM_NUM_SPEC_TOKENS (default 5): draft tokens per step.
    #     VLLM_SPEC_NGRAM_MIN/MAX (default 1/4): n-gram match window.
    #
    #   VLLM_SPEC_DECODE=mtp  — Qwen3.5 built-in MTP head.
    #     Uses the mtp.* transformer layer shipped with the checkpoint to
    #     predict 1 extra token in the same forward pass budget.
    #     VLLM_NUM_SPEC_TOKENS (default 1): set to mtp_num_hidden_layers.
    spec_decode_method = os.environ.get("VLLM_SPEC_DECODE", "").strip().lower()
    spec_decode_config: dict | None = None
    # ngram_gpu has a CUDA-graph sym_shape_indices bug at max_num_seqs=1 in
    # vLLM 0.19.1; fall back to the stable CPU-numba ngram proposer.
    if spec_decode_method == "ngram_gpu":
        spec_decode_method = "ngram"
    if spec_decode_method in ("ngram", "ngram_gpu"):
        num_spec_tokens = int(os.environ.get("VLLM_NUM_SPEC_TOKENS", "5"))
        ngram_min = int(os.environ.get("VLLM_SPEC_NGRAM_MIN", "1"))
        ngram_max = int(os.environ.get("VLLM_SPEC_NGRAM_MAX", "4"))
        spec_decode_config = {
            "method": spec_decode_method,
            "num_speculative_tokens": num_spec_tokens,
            "prompt_lookup_min": ngram_min,
            "prompt_lookup_max": ngram_max,
        }
    elif spec_decode_method == "mtp":
        num_spec_tokens = int(os.environ.get("VLLM_NUM_SPEC_TOKENS", "1"))
        spec_decode_config = {
            "method": "qwen3_5_mtp",
            "model": model_dir,
            "num_speculative_tokens": num_spec_tokens,
        }

    max_num_seqs = max(1, int(os.environ.get("VLLM_MAX_NUM_SEQS", "16")))
    # Spec decode's dummy profiling pass allocates activations for
    # max_num_seqs × (max_model_len + num_spec_tokens) tokens, which OOMs the
    # vision tower's dummy forward on A10 (22 GB) at default max_num_seqs=16.
    # The competition benchmark is single-request, so cap to 1.
    if spec_decode_config and max_num_seqs > 1:
        max_num_seqs = 1
    # Default to 32768: covers 8192 prompt + 12288 GPQA output + buffer.
    # vLLM's default of 262144 OOMs the vision tower dummy forward at startup.
    max_model_len = int(os.environ.get("VLLM_MAX_MODEL_LEN", "0")) or 32768

    gpu_memory_utilization = float(
        os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.90")
    )
    engine_args = AsyncEngineArgs(
        model=model_dir,
        dtype="bfloat16",
        trust_remote_code=True,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        enable_chunked_prefill=enable_chunked_prefill,
        speculative_config=spec_decode_config,
        gpu_memory_utilization=gpu_memory_utilization,
        profiler_config=_profiler_config(profiler_dir),
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

        profiler_dir = _profiler_dir()
        engine = _build_async_engine(model_dir, profiler_dir)
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
