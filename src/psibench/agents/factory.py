from __future__ import annotations

from typing import Any

from psibench.agents.base import BaseAgent
from psibench.agents.llm import LLMAgent


def get_agent(
    strategy: str = "llm",
    *,
    model: str = "gpt-4o",
    provider: str = "openai",
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    system_prompt: str | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> BaseAgent:
    if strategy != "llm":
        raise ValueError(f"Unknown agent strategy: {strategy!r}")
    return LLMAgent(
        model=model,
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
        tools=tools,
    )


# Public convenience alias — matches the ``from psibench.agents import Agent`` usage
# shown in the spec.
Agent = LLMAgent

__all__ = ["Agent", "get_agent"]
