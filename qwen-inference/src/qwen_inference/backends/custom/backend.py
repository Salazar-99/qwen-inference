"""Custom optimized inference backend."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from qwen_inference.types import ChatMessage


class CustomBackend:
    def __init__(self, weights: dict[str, Any], tokenizer: Any) -> None:
        self.weights = weights
        self.tokenizer = tokenizer

    async def generate_from_prompt(
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

    async def generate_from_messages(
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

    async def profile_forward_passes(
        self,
        prompt: str,
        *,
        decode_steps: int,
        profile: bool = False,
        trace_path: str | None = None,
    ) -> dict[str, Any]:
        raise HTTPException(
            status_code=501,
            detail="Custom forward-pass profiling is not implemented yet",
        )
