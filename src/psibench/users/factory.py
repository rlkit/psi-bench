from __future__ import annotations

from psibench.schemas.config import MaliciousBehavior, UserConfig
from psibench.users.base import BaseUser
from psibench.users.llm_user import LLMUser


def load_user(
    *,
    strategy: str = "llm",
    model: str = "gpt-4o",
    provider: str = "openai",
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.7,
    malicious_behavior: MaliciousBehavior = "none",
) -> BaseUser:
    if strategy != "llm":
        raise ValueError(
            f"Only user_strategy='llm' is supported in psi-bench (got {strategy!r})."
        )
    return LLMUser(
        model=model,
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        malicious_behavior=malicious_behavior,
    )


def load_user_from_config(cfg: UserConfig) -> BaseUser:
    return load_user(
        strategy=cfg.strategy,
        model=cfg.model,
        provider=cfg.provider,
        base_url=cfg.base_url,
        api_key=cfg.api_key,
        temperature=cfg.temperature,
        malicious_behavior=cfg.malicious_behavior,
    )
