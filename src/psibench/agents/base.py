"""Base agent.

The base agent owns the :class:`Trajectory` and delegates each turn to a
pluggable :class:`TurnStrategy` that sees the whole trajectory. Custom
behaviors (thinking, teacher/student, debate, self-consistency, …) are
implemented as strategies, not as new agent classes.

Subclasses may override :meth:`turn_strategy` to choose a strategy dynamically
per turn — e.g., "think for the first 3 turns, then act directly", or "use the
teacher only when the user appears adversarial".
"""

from __future__ import annotations

import abc
from typing import Any

from psibench.agents.strategies import TurnStrategy
from psibench.agents.trajectory import Trajectory
from psibench.schemas.messages import Action, Message, Observation


class BaseAgent(abc.ABC):
    def __init__(
        self,
        *,
        strategy: TurnStrategy,
        system_prompt: str,
        tools: list[dict[str, Any]] | None = None,
    ) -> None:
        self.strategy: TurnStrategy = strategy
        self.system_prompt: str = system_prompt
        self.tools: list[dict[str, Any]] = list(tools or [])
        self.trajectory: Trajectory = Trajectory()

    # ----- lifecycle --------------------------------------------------------

    def reset(
        self,
        *,
        task_instruction: str | None = None,
        env_name: str | None = None,
    ) -> None:
        self.trajectory = Trajectory(
            task_instruction=task_instruction,
            env_name=env_name,
        )

    def bind_env(self, *, wiki: str, rules: list[str], tools: list[dict[str, Any]]) -> None:
        """Extend the system prompt with the env's wiki + rules and register its tools."""
        policy = (wiki or "").strip()
        if rules:
            policy += "\n\nRules:\n" + "\n".join(f"- {r}" for r in rules)
        if policy:
            self.system_prompt = f"{self.system_prompt}\n\n--- Domain Policy ---\n{policy}"
        self.tools = list(tools)

    # ----- the turn step ----------------------------------------------------

    def step(self, observation: Observation) -> Action:
        """Record the observation, pick a strategy for this turn, and act."""
        self.trajectory.add_observation(observation)
        self.trajectory.messages.append(_observation_to_message(observation))

        strategy = self.turn_strategy(observation)
        action = strategy.decide(
            trajectory=self.trajectory,
            observation=observation,
            system_prompt=self.system_prompt,
            tools=self.tools or None,
        )
        self.trajectory.record_action(action)
        return action

    # ----- extension point --------------------------------------------------

    def turn_strategy(self, observation: Observation) -> TurnStrategy:  # noqa: ARG002
        """Return the :class:`TurnStrategy` to use for this turn.

        Defaults to the agent's configured ``self.strategy``. Subclasses can
        override to switch strategies dynamically (e.g., only think every N
        turns, or escalate to a teacher strategy when the user is adversarial).
        """
        return self.strategy


def _observation_to_message(obs: Observation) -> Message:
    if obs.source == "user":
        return Message(role="user", content=obs.content)
    return Message(role="tool", name=obs.source, content=obs.content)
