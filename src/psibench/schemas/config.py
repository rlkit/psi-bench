"""Typed configuration objects for models, agents, users, judges, and envs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

MaliciousBehavior = Literal[
    "none",
    "cheat",
    "prompt_injection",
    "unsafe_request",
    "jailbreak_attempt",
]


class ModelConfig(BaseModel):
    """Config for a single OpenAI-compatible model endpoint."""

    model: str
    provider: str = "openai"
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = None
    timeout: float = 60.0
    extra_headers: dict[str, str] = Field(default_factory=dict)


class AgentConfig(BaseModel):
    model: str
    provider: str = "openai"
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = None
    system_prompt: str | None = None


class UserConfig(BaseModel):
    strategy: Literal["llm", "human"] = "llm"
    model: str = "gpt-4o"
    provider: str = "openai"
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.7
    malicious_behavior: MaliciousBehavior = "none"


class JudgeConfig(BaseModel):
    """Either ``url`` (HTTP) or ``model`` (LLM judge) must be provided."""

    url: str | None = None
    system_prompt: str | None = None
    model: str | None = None
    provider: str = "openai"
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.0
    satisfaction_weight: float = 0.5
    safety_weight: float = 0.5
    timeout: float = 30.0


class EnvConfig(BaseModel):
    env_name: str
    task_index: int | None = None
    task_split: str = "test"
    score_turn: int | Literal["every turn", "end"] = "end"
    max_turns: int = 30
    agent: AgentConfig | None = None
    user: UserConfig = Field(default_factory=UserConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
