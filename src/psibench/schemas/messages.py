"""Core message / action / observation schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

RESPOND_ACTION_NAME = "respond"


Role = Literal["system", "user", "assistant", "tool"]


class Message(BaseModel):
    """A single conversation message."""

    role: Role
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class Action(BaseModel):
    """An action the agent takes — either a tool call or a text response."""

    name: str
    kwargs: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def respond(cls, content: str) -> Action:
        return cls(name=RESPOND_ACTION_NAME, kwargs={"content": content})

    @property
    def is_respond(self) -> bool:
        return self.name == RESPOND_ACTION_NAME


class Observation(BaseModel):
    """What the environment returns to the agent each step."""

    content: str
    source: str = "user"  # "user" or tool name
    turn: int = 0
    done: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class Task(BaseModel):
    """A benchmark task definition.

    ``actions`` and ``outputs`` are legacy fields carried over from the original
    domain data files; they are informational only — evaluation is done by the
    judge, not by comparing against them.
    """

    user_id: str = ""
    instruction: str = ""
    actions: list[Action] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    annotator: str | int | None = None
