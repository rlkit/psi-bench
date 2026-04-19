"""Environment factory."""

from __future__ import annotations

from typing import Any, Callable

from psibench.envs.base import Env, build_env
from psibench.schemas.config import MaliciousBehavior

DomainLoader = Callable[[str], dict[str, Any]]

_DOMAINS: dict[str, DomainLoader] = {}


def register_env(name: str, loader: DomainLoader) -> None:
    """Register a new domain under ``name``.

    ``loader(task_split)`` must return a dict with keys compatible with
    :func:`psibench.envs.base.build_env`: ``data_load_func``, ``tools``,
    ``tasks``, ``wiki``, ``rules``, ``terminate_tools``.
    """
    _DOMAINS[name] = loader


def _load_retail(task_split: str) -> dict[str, Any]:
    from psibench.envs.retail import retail_domain

    return retail_domain(task_split)


def _load_airline(task_split: str) -> dict[str, Any]:
    from psibench.envs.airline import airline_domain

    return airline_domain(task_split)


register_env("retail", _load_retail)
register_env("airline", _load_airline)


def get_env(
    env_name: str,
    *,
    user_strategy: str = "llm",
    user_model: str = "gpt-4o",
    user_provider: str = "openai",
    user_base_url: str | None = None,
    user_api_key: str | None = None,
    user_temperature: float = 0.7,
    user_malicious_behavior: MaliciousBehavior = "none",
    judge_url: str | None = None,
    judge_system_prompt: str | None = None,
    judge_model: str | None = None,
    judge_provider: str = "openai",
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
    judge_temperature: float = 0.0,
    satisfaction_weight: float = 0.5,
    safety_weight: float = 0.5,
    score_turn: int | str = "end",
    max_turns: int = 30,
    task_split: str = "test",
    task_index: int | None = None,
) -> Env:
    """Create an :class:`Env` for the given domain.

    Either ``judge_url`` or ``judge_model`` must be provided.
    """
    if env_name not in _DOMAINS:
        raise ValueError(
            f"Unknown environment {env_name!r}. Registered: {sorted(_DOMAINS)}"
        )
    domain = _DOMAINS[env_name](task_split)
    return build_env(
        **domain,
        user_strategy=user_strategy,
        user_model=user_model,
        user_provider=user_provider,
        user_base_url=user_base_url,
        user_api_key=user_api_key,
        user_temperature=user_temperature,
        user_malicious_behavior=user_malicious_behavior,
        judge_url=judge_url,
        judge_system_prompt=judge_system_prompt,
        judge_model=judge_model,
        judge_provider=judge_provider,
        judge_base_url=judge_base_url,
        judge_api_key=judge_api_key,
        judge_temperature=judge_temperature,
        satisfaction_weight=satisfaction_weight,
        safety_weight=safety_weight,
        score_turn=score_turn,
        max_turns=max_turns,
        task_index=task_index,
    )


__all__ = ["Env", "get_env", "register_env"]
