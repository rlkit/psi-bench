"""HTTP judge — POSTs the :class:`JudgeInput` to an external URL."""

from __future__ import annotations

import httpx

from psibench.judges.base import BaseJudge, JudgeInput
from psibench.schemas.reward import Reward


class HTTPJudge(BaseJudge):
    def __init__(
        self,
        url: str,
        *,
        api_key: str | None = None,
        timeout: float = 30.0,
        satisfaction_weight: float = 0.5,
        safety_weight: float = 0.5,
    ) -> None:
        self.url = url
        self.timeout = timeout
        self.satisfaction_weight = satisfaction_weight
        self.safety_weight = safety_weight
        self._headers: dict[str, str] = {}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

    def evaluate(self, payload: JudgeInput) -> Reward:
        with httpx.Client(timeout=self.timeout) as client:
            r = client.post(
                self.url,
                json=payload.model_dump(mode="json"),
                headers=self._headers,
            )
            r.raise_for_status()
            data = r.json()

        sat = float(data.get("user_satisfaction_score", 0.0))
        safety = float(data.get("safety_score", 0.0))
        total = data.get("total_score")
        if total is None:
            total = Reward.compute_total(
                sat, safety, self.satisfaction_weight, self.safety_weight
            )
        return Reward(
            user_satisfaction_score=sat,
            safety_score=safety,
            total_score=float(total),
            reasoning=data.get("reasoning"),
            raw=data,
        )
