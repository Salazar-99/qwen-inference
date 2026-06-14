"""FastAPI routes and request handlers for the Qwen inference server."""

from __future__ import annotations

from contextlib import contextmanager
import os
import uuid
from collections.abc import Iterator
from typing import Any, Protocol

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

DEFAULT_MODEL = os.environ.get("MODEL_NAME", "Qwen/Qwen3.5-4B")

_backend: "InferenceBackend | None" = None
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
    profile: bool = False


class ForwardProfileRequest(BaseModel):
    prompt: str
    decode_steps: int = 4
    profile: bool = True


class InferenceBackend(Protocol):
    def generate_from_prompt(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
    ) -> str: ...

    def generate_from_messages(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int,
        temperature: float,
        thinking: bool | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> str: ...

    def profile_forward_passes(self, prompt: str, *, decode_steps: int) -> dict[str, Any]: ...


class CustomBackend:
    def __init__(self, weights: dict[str, Any], tokenizer: Any) -> None:
        self.weights = weights
        self.tokenizer = tokenizer

    def generate_from_prompt(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
    ) -> str:
        raise HTTPException(
            status_code=501,
            detail="Custom prompt completion is not implemented yet",
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
        raise HTTPException(
            status_code=501,
            detail="Custom chat completion is not implemented yet",
        )

    def profile_forward_passes(self, prompt: str, *, decode_steps: int) -> dict[str, Any]:
        raise HTTPException(
            status_code=501,
            detail="Custom forward-pass profiling is not implemented yet",
        )


class BaselineTransformersBackend:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        *,
        device: str,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)

    def generate_from_prompt(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
    ) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        return self._generate(inputs, max_tokens=max_tokens, temperature=temperature)

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

        input_ids = self.tokenizer.apply_chat_template(
            [message.model_dump() for message in messages],
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True,
            **template_kwargs,
        ).to(self.device)
        return self._generate(
            {"input_ids": input_ids},
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def _generate(
        self,
        inputs: dict[str, torch.Tensor],
        *,
        max_tokens: int,
        temperature: float,
    ) -> str:
        input_length = inputs["input_ids"].shape[-1]
        generation_kwargs: dict[str, Any] = {
            **inputs,
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0:
            generation_kwargs["temperature"] = temperature

        with torch.inference_mode():
            output_ids = self.model.generate(**generation_kwargs)

        new_tokens = output_ids[0, input_length:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def profile_forward_passes(self, prompt: str, *, decode_steps: int) -> dict[str, Any]:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs["input_ids"].shape[-1]
        decode_steps = max(0, decode_steps)

        with torch.inference_mode():
            outputs = self.model(**inputs, use_cache=True)
            past_key_values = outputs.past_key_values

            attention_mask = inputs.get("attention_mask")
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            for _ in range(decode_steps):
                if attention_mask is not None:
                    attention_mask = torch.cat(
                        [attention_mask, torch.ones_like(next_token)],
                        dim=-1,
                    )
                outputs = self.model(
                    input_ids=next_token,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        return {
            "prompt_tokens": input_length,
            "prefill_forwards": 1,
            "decode_forwards": decode_steps,
        }


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


def _set_backend(backend: InferenceBackend) -> None:
    global _backend, _model_ready
    _backend = backend
    _model_ready = True


def generate_from_prompt(
    prompt: str,
    *,
    max_tokens: int,
    temperature: float,
) -> str:
    """Generate raw text completion (no chat template, no thinking)."""
    _ensure_model_ready()
    assert _backend is not None
    return _backend.generate_from_prompt(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
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
    _ensure_model_ready()
    assert _backend is not None
    return _backend.generate_from_messages(
        messages,
        max_tokens=max_tokens,
        temperature=temperature,
        thinking=thinking,
        chat_template_kwargs=chat_template_kwargs,
    )


def profile_forward_passes(prompt: str, *, decode_steps: int) -> dict[str, Any]:
    _ensure_model_ready()
    assert _backend is not None
    return _backend.profile_forward_passes(prompt, decode_steps=decode_steps)


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
        with _nsys_capture(request.profile):
            text = generate_from_messages(
                request.messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                thinking=_thinking_enabled(request.thinking, request.chat_template_kwargs),
                chat_template_kwargs=request.chat_template_kwargs,
            )
        return chat_completion_response(text, model=DEFAULT_MODEL)

    if request.prompt is not None:
        with _nsys_capture(request.profile):
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


@app.post("/profile/forward")
def profile_forward(request: ForwardProfileRequest) -> dict[str, Any]:
    _ensure_model_ready()
    with _nsys_capture(request.profile):
        return profile_forward_passes(
            request.prompt,
            decode_steps=request.decode_steps,
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
