from __future__ import annotations

import abc
from typing import Any

from pydantic import BaseModel, Field

from psibench.schemas.messages import Message
from psibench.schemas.reward import Reward


class JudgeInput(BaseModel):
    """Structured payload sent to the judge."""

    env_name: str
    task_instruction: str | None = None
    user_profile: dict[str, Any] = Field(default_factory=dict)
    malicious_behavior: str = "none"
    conversation: list[Message]
    latest_agent_response: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BaseJudge(abc.ABC):
    @abc.abstractmethod
    def evaluate(self, payload: JudgeInput) -> Reward:
        ...
