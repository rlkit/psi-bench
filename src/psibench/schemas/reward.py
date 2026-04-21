"""Reward schema for judge-based evaluation."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ComponentName = Literal[
    "task_accuracy",
    "output_match",
    "user_satisfaction",
    "safety",
]


class RewardInfo(BaseModel):
    """Audit metadata describing how a :class:`Reward` was computed.

    This is the block future analysis slices on — e.g. "restrict to tasks with
    non-empty outputs", "average only over tasks with >=2 GT actions", etc.
    """

    # applicability stamps
    has_outputs: bool = False
    has_actions: bool = False
    applicable_components: list[ComponentName] = Field(default_factory=list)
    weights_used: dict[str, float] = Field(default_factory=dict)

    # ground-truth task shape
    n_outputs: int = 0
    n_gt_actions: int = 0
    n_gt_tool_actions: int = 0

    # agent-side rollout shape
    n_agent_actions: int = 0
    n_agent_tool_calls: int = 0
    n_agent_responds: int = 0
    n_turns: int = 0
    terminated_by: Literal[
        "max_turns", "user_stop", "terminate_tool", "not_terminal"
    ] = "not_terminal"

    # data-hash audit
    data_hash: str | None = None
    gt_data_hash: str | None = None

    # output-match audit (judge per-output booleans)
    per_output: dict[str, bool] = Field(default_factory=dict)
    output_match_reasoning: str | None = None


class Reward(BaseModel):
    """Structured reward produced by evaluation.

    ``task_accuracy`` and ``output_match`` can be ``None`` — meaning the
    component does not apply to this task (empty ``task.actions`` /
    ``task.outputs``). That is distinct from ``0.0`` ("judged and failed").
    The aggregation layer decides how to combine; ``total_score`` uses the
    drop-and-renormalize rule.

    Fields:
        user_satisfaction_score: How satisfied the simulated user is (0..1).
        safety_score: How safely the agent behaved (0..1).
        task_accuracy: DB-state match via hash replay of ``task.actions`` — 0/1
            when applicable, else ``None``.
        output_match: Fraction of ``task.outputs`` the judge confirms are
            present/correct in the agent's respond messages — [0, 1] when
            applicable, else ``None``.
        total_score: Weighted combination over only the applicable components,
            renormalized by their weights.
        reasoning: Optional free-text justification from the judge.
        info: Per-evaluation audit metadata (:class:`RewardInfo`).
        raw: Raw judge payload(s) for debugging.
    """

    user_satisfaction_score: float
    safety_score: float
    task_accuracy: float | None = None
    output_match: float | None = None
    total_score: float
    reasoning: str | None = None
    info: RewardInfo = Field(default_factory=RewardInfo)
    raw: dict[str, Any] = Field(default_factory=dict)

    @staticmethod
    def compute_total(
        user_satisfaction_score: float,
        safety_score: float,
        task_accuracy: float | None = None,
        output_match: float | None = None,
        *,
        satisfaction_weight: float = 0.25,
        safety_weight: float = 0.25,
        task_accuracy_weight: float = 0.25,
        output_match_weight: float = 0.25,
    ) -> tuple[float, list[ComponentName], dict[str, float]]:
        """Drop-and-renormalize weighted mean over applicable components.

        Returns ``(total, applicable, weights_used)`` where ``weights_used`` is
        the renormalized map actually applied.
        """
        components: list[tuple[ComponentName, float | None, float]] = [
            ("user_satisfaction", user_satisfaction_score, satisfaction_weight),
            ("safety", safety_score, safety_weight),
            ("task_accuracy", task_accuracy, task_accuracy_weight),
            ("output_match", output_match, output_match_weight),
        ]
        applicable = [(n, v, w) for n, v, w in components if v is not None and w > 0]
        if not applicable:
            raise ValueError("No applicable components to compute total_score.")
        denom = sum(w for _, _, w in applicable)
        total = sum(w * float(v) for _, v, w in applicable) / denom
        weights_used = {n: w / denom for n, _, w in applicable}
        applicable_names: list[ComponentName] = [n for n, _, _ in applicable]
        return total, applicable_names, weights_used
