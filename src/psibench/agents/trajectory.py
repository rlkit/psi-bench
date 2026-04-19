"""Trajectory — the full history an agent sees across turns.

Agents and their :class:`TurnStrategy` receive a ``Trajectory`` each step, which
includes every observation, every action, the flattened OpenAI-style message
log, and any strategy-specific scratchpad entries (``thoughts``, ``notes``).
This is what enables "thinking" strategies, teacher/student setups, and other
multi-pass or multi-agent policies on top of the same base agent.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from psibench.schemas.messages import Action, Message, Observation


class Turn(BaseModel):
    observation: Observation
    action: Action | None = None
    thoughts: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Trajectory(BaseModel):
    """All the agent-visible state accumulated over an episode."""

    task_instruction: str | None = None
    env_name: str | None = None
    turns: list[Turn] = Field(default_factory=list)
    messages: list[Message] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # -- convenience --------------------------------------------------------

    @property
    def last(self) -> Turn | None:
        return self.turns[-1] if self.turns else None

    @property
    def num_turns(self) -> int:
        return len(self.turns)

    def add_observation(self, observation: Observation) -> Turn:
        turn = Turn(observation=observation)
        self.turns.append(turn)
        return turn

    def record_action(self, action: Action) -> None:
        if not self.turns:
            raise RuntimeError("record_action called before any observation")
        self.turns[-1].action = action

    def record_thought(self, thought: str) -> None:
        if self.turns:
            self.turns[-1].thoughts.append(thought)
        else:
            self.notes.append(thought)
