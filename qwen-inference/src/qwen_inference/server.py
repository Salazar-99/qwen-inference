"""FastAPI routes and request handlers for the Qwen inference server."""

from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
import os
from pathlib import Path
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from qwen_inference.backends import (
    BaselineTransformersBackend,
    CustomBackend,
    InferenceBackend,
    VllmBackend,
)
from qwen_inference.types import ChatMessage

DEFAULT_MODEL = os.environ.get("MODEL_NAME", "Qwen/Qwen3.5-4B")

_backend: "InferenceBackend | None" = None
_model_ready = False
# Backends that must be built inside the event loop (the async vLLM engine)
# register a coroutine factory here; the FastAPI lifespan runs it on startup,
# before the server reports ready.
_pending_async_init: "Callable[[], Awaitable[None]] | None" = None


class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.0


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int = 128
    temperature: float = 0.0
    thinking: bool | None = None
    stream: bool | None = None
    chat_template_kwargs: dict[str, Any] | None = None


class InvocationRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    prompt: str | None = None
    messages: list[ChatMessage] | None = None
    max_tokens: int = 128
    temperature: float = 0.0
    thinking: bool | None = None
    stream: bool | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    profile: bool = False


class ForwardProfileRequest(BaseModel):
    prompt: str
    decode_steps: int = 4
    profile: bool = True
    trace_path: str | None = None


def configure(weights: dict[str, Any], tokenizer: Any) -> None:
    """Attach loaded custom model weights and tokenizer for request handlers."""
    configure_custom(weights, tokenizer)


def configure_custom(weights: dict[str, Any], tokenizer: Any) -> None:
    """Use the optimized/custom inference backend."""
    _set_backend(CustomBackend(weights, tokenizer))


def configure_baseline(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    *,
    device: str,
) -> None:
    """Use Hugging Face Transformers directly for baseline profiling."""
    _set_backend(BaselineTransformersBackend(model, tokenizer, device=device))


async def configure_vllm(model_dir: str) -> None:
    """Use vLLM (async engine) for competition-style serving."""
    _set_backend(await VllmBackend.create(model_dir))


def register_async_initializer(factory: "Callable[[], Awaitable[None]]") -> None:
    """Defer backend construction to the event loop (run by the lifespan)."""
    global _pending_async_init
    _pending_async_init = factory


def _set_backend(backend: InferenceBackend) -> None:
    global _backend, _model_ready
    _backend = backend
    _model_ready = True


async def generate_from_prompt(
    prompt: str,
    *,
    max_tokens: int,
    temperature: float,
) -> str:
    """Generate raw text completion (no chat template, no thinking)."""
    _ensure_model_ready()
    assert _backend is not None
    return await _backend.generate_from_prompt(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )


async def generate_from_messages(
    messages: list[ChatMessage],
    *,
    max_tokens: int,
    temperature: float,
    thinking: bool | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> str:
    """Generate assistant text from chat messages."""
    _ensure_model_ready()
    assert _backend is not None
    return await _backend.generate_from_messages(
        messages,
        max_tokens=max_tokens,
        temperature=temperature,
        thinking=thinking,
        chat_template_kwargs=chat_template_kwargs,
    )


async def profile_forward_passes(prompt: str, *, decode_steps: int) -> dict[str, Any]:
    _ensure_model_ready()
    assert _backend is not None
    return await _backend.profile_forward_passes(prompt, decode_steps=decode_steps)


async def profile_forward_passes_with_options(
    prompt: str,
    *,
    decode_steps: int,
    profile: bool = False,
    trace_path: str | None = None,
) -> dict[str, Any]:
    _ensure_model_ready()
    assert _backend is not None
    return await _backend.profile_forward_passes(
        prompt,
        decode_steps=decode_steps,
        profile=profile,
        trace_path=trace_path,
    )


def text_completion_response(text: str) -> dict[str, Any]:
    return {"choices": [{"text": text}]}


def chat_completion_response(text: str, *, model: str) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
    }


def _ensure_model_ready() -> None:
    if not _model_ready:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")


def _thinking_enabled(
    thinking: bool | None,
    chat_template_kwargs: dict[str, Any] | None,
) -> bool | None:
    if thinking is not None:
        return thinking
    if chat_template_kwargs and "enable_thinking" in chat_template_kwargs:
        return bool(chat_template_kwargs["enable_thinking"])
    return None


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


@contextmanager
def _profile_capture(enabled: bool, trace_path: str | None = None) -> Iterator[None]:
    if trace_path:
        output_path = Path(trace_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as profiler:
            yield
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        profiler.export_chrome_trace(str(output_path))
        return

    with _nsys_capture(enabled):
        yield


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Build any deferred backend (the async vLLM engine) inside the running loop,
    # including warmup, before the server starts accepting requests.
    global _pending_async_init
    if _pending_async_init is not None:
        await _pending_async_init()
        _pending_async_init = None
    yield


app = FastAPI(title="Qwen Inference Server", lifespan=_lifespan)


@app.get("/ping")
def ping() -> dict[str, str]:
    _ensure_model_ready()
    return {"status": "ok"}


@app.post("/invocations")
async def invocations(request: InvocationRequest) -> dict[str, Any]:
    _ensure_model_ready()

    if request.messages is not None:
        if request.stream:
            raise HTTPException(status_code=501, detail="Streaming is not implemented yet")
        with _nsys_capture(request.profile):
            text = await generate_from_messages(
                request.messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                thinking=_thinking_enabled(request.thinking, request.chat_template_kwargs),
                chat_template_kwargs=request.chat_template_kwargs,
            )
        return chat_completion_response(text, model=DEFAULT_MODEL)

    if request.prompt is not None:
        with _nsys_capture(request.profile):
            text = await generate_from_prompt(
                request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
        return text_completion_response(text)

    raise HTTPException(
        status_code=400,
        detail='Request must include either "prompt" or "messages"',
    )


@app.post("/profile/forward")
async def profile_forward(request: ForwardProfileRequest) -> dict[str, Any]:
    _ensure_model_ready()
    assert _backend is not None

    if isinstance(_backend, VllmBackend):
        return await profile_forward_passes_with_options(
            request.prompt,
            decode_steps=request.decode_steps,
            profile=request.profile,
            trace_path=request.trace_path,
        )

    with _profile_capture(request.profile, request.trace_path):
        return await profile_forward_passes(
            request.prompt,
            decode_steps=request.decode_steps,
        )


@app.post("/v1/completions")
async def v1_completions(request: CompletionRequest) -> dict[str, Any]:
    _ensure_model_ready()
    text = await generate_from_prompt(
        request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )
    return text_completion_response(text)


@app.post("/v1/chat/completions")
async def v1_chat_completions(request: ChatCompletionRequest) -> dict[str, Any]:
    _ensure_model_ready()

    if request.stream:
        raise HTTPException(status_code=501, detail="Streaming is not implemented yet")

    text = await generate_from_messages(
        request.messages,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        thinking=_thinking_enabled(request.thinking, request.chat_template_kwargs),
        chat_template_kwargs=request.chat_template_kwargs,
    )
    return chat_completion_response(text, model=request.model or DEFAULT_MODEL)
