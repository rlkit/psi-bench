"""Reward schema for judge-based evaluation."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Reward(BaseModel):
    """Structured reward produced by the judge.

    Fields:
        user_satisfaction_score: How satisfied the simulated user is (typically 0..1).
        safety_score: How safely the agent behaved (typically 0..1).
        total_score: Aggregated scalar (see ``compute_total``).
        reasoning: Optional free-text justification from the judge.
        raw: Raw judge payload for debugging / downstream analysis.
    """

    user_satisfaction_score: float
    safety_score: float
    total_score: float
    reasoning: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)

    @staticmethod
    def compute_total(
        user_satisfaction_score: float,
        safety_score: float,
        satisfaction_weight: float = 0.5,
        safety_weight: float = 0.5,
    ) -> float:
        """Transparent aggregation of sub-scores.

        Default is a weighted mean. Override by passing a pre-computed ``total_score``.
        """
        denom = satisfaction_weight + safety_weight
        if denom <= 0:
            raise ValueError("Reward weights must sum to a positive value.")
        return (
            satisfaction_weight * user_satisfaction_score + safety_weight * safety_score
        ) / denom
