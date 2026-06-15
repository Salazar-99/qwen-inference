"""Backend interface for inference implementations."""

from __future__ import annotations

from typing import Any, Protocol

from qwen_inference.types import ChatMessage


class InferenceBackend(Protocol):
    async def generate_from_prompt(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
    ) -> str: ...

    async def generate_from_messages(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int,
        temperature: float,
        thinking: bool | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> str: ...

    async def profile_forward_passes(
        self,
        prompt: str,
        *,
        decode_steps: int,
        profile: bool = False,
        trace_path: str | None = None,
    ) -> dict[str, Any]: ...
