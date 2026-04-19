from __future__ import annotations

from psibench.judges.base import BaseJudge
from psibench.judges.http_judge import HTTPJudge
from psibench.judges.llm_judge import LLMJudge
from psibench.schemas.config import JudgeConfig


def load_judge(cfg: JudgeConfig) -> BaseJudge:
    if cfg.url:
        return HTTPJudge(
            url=cfg.url,
            api_key=cfg.api_key,
            timeout=cfg.timeout,
            satisfaction_weight=cfg.satisfaction_weight,
            safety_weight=cfg.safety_weight,
        )
    if cfg.model:
        return LLMJudge(
            model=cfg.model,
            system_prompt=cfg.system_prompt,
            provider=cfg.provider,
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            temperature=cfg.temperature,
            satisfaction_weight=cfg.satisfaction_weight,
            safety_weight=cfg.safety_weight,
            timeout=cfg.timeout,
        )
    raise ValueError(
        "JudgeConfig must specify either `url` (HTTP judge) or `model` (LLM judge)."
    )
