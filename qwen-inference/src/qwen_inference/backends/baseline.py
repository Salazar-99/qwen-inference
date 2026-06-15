"""Hugging Face Transformers baseline inference backend."""

from __future__ import annotations

import asyncio
from typing import Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from qwen_inference.types import ChatMessage


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

    async def generate_from_prompt(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
    ) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        # HF generate is blocking; run it off the event loop.
        return await asyncio.to_thread(
            self._generate, inputs, max_tokens=max_tokens, temperature=temperature
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
        return await asyncio.to_thread(
            self._generate,
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

    async def profile_forward_passes(
        self,
        prompt: str,
        *,
        decode_steps: int,
        profile: bool = False,
        trace_path: str | None = None,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._profile_forward_passes, prompt, decode_steps=decode_steps
        )

    def _profile_forward_passes(
        self,
        prompt: str,
        *,
        decode_steps: int,
    ) -> dict[str, Any]:
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
