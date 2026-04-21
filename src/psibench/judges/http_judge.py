"""HTTP judge — POSTs the :class:`JudgeInput` to an external URL."""

from __future__ import annotations

import httpx

from psibench.judges.base import BaseJudge, JudgeInput
from psibench.schemas.reward import Reward, RewardInfo


class HTTPJudge(BaseJudge):
    def __init__(
        self,
        url: str,
        *,
        api_key: str | None = None,
        timeout: float = 30.0,
        satisfaction_weight: float = 0.25,
        safety_weight: float = 0.25,
        task_accuracy_weight: float = 0.25,
        output_match_weight: float = 0.25,
    ) -> None:
        self.url = url
        self.timeout = timeout
        self.satisfaction_weight = satisfaction_weight
        self.safety_weight = safety_weight
        self.task_accuracy_weight = task_accuracy_weight
        self.output_match_weight = output_match_weight
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

        per_output_raw = data.get("per_output") or {}
        per_output: dict[str, bool] = {str(k): bool(v) for k, v in per_output_raw.items()}

        output_match: float | None
        if payload.expected_outputs:
            if "output_match" in data and data["output_match"] is not None:
                output_match = float(data["output_match"])
            elif per_output:
                output_match = sum(1 for v in per_output.values() if v) / max(len(per_output), 1)
            else:
                output_match = 0.0
        else:
            output_match = None

        ta_meta = payload.metadata.get("task_accuracy")
        task_accuracy: float | None = float(ta_meta) if ta_meta is not None else None

        total, applicable, weights_used = Reward.compute_total(
            sat,
            safety,
            task_accuracy=task_accuracy,
            output_match=output_match,
            satisfaction_weight=self.satisfaction_weight,
            safety_weight=self.safety_weight,
            task_accuracy_weight=self.task_accuracy_weight,
            output_match_weight=self.output_match_weight,
        )

        return Reward(
            user_satisfaction_score=sat,
            safety_score=safety,
            task_accuracy=task_accuracy,
            output_match=output_match,
            total_score=float(total),
            reasoning=data.get("reasoning"),
            raw=data,
            info=RewardInfo(
                applicable_components=applicable,
                weights_used=weights_used,
                per_output=per_output,
                output_match_reasoning=data.get("reasoning"),
            ),
        )
