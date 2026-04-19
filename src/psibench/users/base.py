from __future__ import annotations

import abc

from psibench.agents.trajectory import Trajectory


class BaseUser(abc.ABC):
    """A simulated user the agent converses with.

    Like the agent, the user is given read access to the full :class:`Trajectory`
    on every turn — so a user strategy can condition its reply on all prior
    tool calls, tool results, and agent messages, not just the visible chat.
    """

    @abc.abstractmethod
    def reset(
        self,
        instruction: str | None = None,
        *,
        trajectory: Trajectory | None = None,
    ) -> str:
        """Initialize the user and return the opening message."""

    @abc.abstractmethod
    def step(
        self,
        agent_message: str,
        *,
        trajectory: Trajectory | None = None,
    ) -> str:
        """Produce the user's next reply, optionally conditioned on the full trajectory."""

    @property
    @abc.abstractmethod
    def is_done(self) -> bool:
        """Whether the user has signaled the conversation is over."""
