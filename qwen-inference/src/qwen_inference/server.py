"""FastAPI routes and request handlers for the Qwen inference server."""

from __future__ import annotations

import os
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

DEFAULT_MODEL = os.environ.get("MODEL_NAME", "Qwen/Qwen3.5-4B")

_weights: dict[str, Any] | None = None
_tokenizer: Any | None = None
_model_ready = False


class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.0


class ChatMessage(BaseModel):
    role: str
    content: str


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


def configure(weights: dict[str, Any], tokenizer: Any) -> None:
    """Attach loaded model weights and tokenizer for request handlers."""
    global _weights, _tokenizer, _model_ready
    _weights = weights
    _tokenizer = tokenizer
    _model_ready = True


def generate_from_prompt(
    prompt: str,
    *,
    max_tokens: int,
    temperature: float,
) -> str:
    """Generate raw text completion (no chat template, no thinking)."""
    raise HTTPException(
        status_code=501,
        detail="Prompt completion is not implemented yet",
    )


def generate_from_messages(
    messages: list[ChatMessage],
    *,
    max_tokens: int,
    temperature: float,
    thinking: bool | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> str:
    """Generate assistant text from chat messages."""
    raise NotImplementedError("Chat completion is not implemented yet")


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


app = FastAPI(title="Qwen Inference Server")


@app.get("/ping")
def ping() -> dict[str, str]:
    _ensure_model_ready()
    return {"status": "ok"}


@app.post("/invocations")
def invocations(request: InvocationRequest) -> dict[str, Any]:
    _ensure_model_ready()

    if request.messages is not None:
        if request.stream:
            raise HTTPException(status_code=501, detail="Streaming is not implemented yet")
        text = generate_from_messages(
            request.messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            thinking=_thinking_enabled(request.thinking, request.chat_template_kwargs),
            chat_template_kwargs=request.chat_template_kwargs,
        )
        return chat_completion_response(text, model=DEFAULT_MODEL)

    if request.prompt is not None:
        text = generate_from_prompt(
            request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        return text_completion_response(text)

    raise HTTPException(
        status_code=400,
        detail='Request must include either "prompt" or "messages"',
    )


@app.post("/v1/completions")
def v1_completions(request: CompletionRequest) -> dict[str, Any]:
    _ensure_model_ready()
    text = generate_from_prompt(
        request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )
    return text_completion_response(text)


@app.post("/v1/chat/completions")
def v1_chat_completions(request: ChatCompletionRequest) -> dict[str, Any]:
    _ensure_model_ready()

    if request.stream:
        raise HTTPException(status_code=501, detail="Streaming is not implemented yet")

    text = generate_from_messages(
        request.messages,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        thinking=_thinking_enabled(request.thinking, request.chat_template_kwargs),
        chat_template_kwargs=request.chat_template_kwargs,
    )
    return chat_completion_response(text, model=request.model or DEFAULT_MODEL)
