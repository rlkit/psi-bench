"""OpenAI-compatible chat client.

This is a thin wrapper over the ``openai`` SDK with two conveniences:

1. Works with any OpenAI-compatible endpoint (OpenAI, sglang, vLLM, etc.) via
   ``base_url``.
2. Accepts our typed :class:`psibench.schemas.messages.Message` objects directly.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Iterable

from openai import OpenAI
from pydantic import BaseModel

from psibench.schemas.messages import Message


class Usage(BaseModel):
    """Per-call token usage, plus a running total on the owning client."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0

    def add(self, other: "Usage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens
        self.calls += other.calls

    @classmethod
    def from_response(cls, usage: Any) -> "Usage":
        if usage is None:
            return cls()
        return cls(
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage, "total_tokens", 0) or 0,
            calls=1,
        )


def _messages_to_dicts(messages: Iterable[Message | dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if isinstance(m, Message):
            d = m.model_dump(exclude_none=True)
        else:
            d = {k: v for k, v in m.items() if v is not None}
        out.append(d)
    return out


class ChatClient:
    """OpenAI-compatible chat completion client.

    ``provider`` is informational (``openai``, ``sglang``, ``vllm``, etc.); routing
    is determined by ``base_url`` + ``api_key``. For sglang-served models you
    typically pass ``base_url="http://localhost:30000/v1"``.
    """

    def __init__(
        self,
        model: str,
        *,
        provider: str = "openai",
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        timeout: float = 60.0,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.model = model
        self.provider = provider
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.extra_headers = extra_headers or {}

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
        self._client = OpenAI(
            api_key=resolved_key,
            base_url=base_url,
            timeout=timeout,
        )
        self.last_usage: Usage = Usage()
        self.cumulative_usage: Usage = Usage()

    def complete(
        self,
        messages: Iterable[Message | dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float | None = None,
        response_format: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> Any:
        """Return the raw ChatCompletion response."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": _messages_to_dicts(messages),
            "temperature": self.temperature if temperature is None else temperature,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if tools:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if response_format is not None:
            kwargs["response_format"] = response_format
        if self.extra_headers:
            kwargs["extra_headers"] = self.extra_headers
        if extra_body is not None:
            kwargs["extra_body"] = extra_body
        res = self._client.chat.completions.create(**kwargs)
        self.last_usage = Usage.from_response(getattr(res, "usage", None))
        self.cumulative_usage.add(self.last_usage)
        return res

    def complete_text(
        self,
        messages: Iterable[Message | dict[str, Any]],
        **kwargs: Any,
    ) -> str:
        """Return just the assistant message content as a string."""
        res = self.complete(messages, **kwargs)
        content = res.choices[0].message.content
        return content or ""


@lru_cache(maxsize=16)
def _cached_client(
    model: str,
    provider: str,
    base_url: str | None,
    api_key: str | None,
    temperature: float,
    max_tokens: int | None,
    timeout: float,
) -> ChatClient:
    return ChatClient(
        model=model,
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )


def chat_completion(
    messages: Iterable[Message | dict[str, Any]],
    *,
    model: str,
    provider: str = "openai",
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    timeout: float = 60.0,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    response_format: dict[str, Any] | None = None,
) -> Any:
    """One-shot chat completion helper that reuses a cached client."""
    client = _cached_client(model, provider, base_url, api_key, temperature, max_tokens, timeout)
    return client.complete(
        messages,
        tools=tools,
        tool_choice=tool_choice,
        response_format=response_format,
    )
