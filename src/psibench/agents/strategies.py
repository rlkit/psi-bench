"""Pluggable turn strategies.

A :class:`TurnStrategy` decides the next :class:`Action` given the complete
:class:`Trajectory` so far. This decouples *how* a turn is computed (single
LLM call, think-then-act, teacher/student, debate, tool-use planning, etc.)
from the agent's outer loop.

Built-in strategies:

- :class:`DirectStrategy` — one LLM call with optional tool use.
- :class:`ThinkingStrategy` — private chain-of-thought, then a public action.
- :class:`TeacherStudentStrategy` — a "teacher" model writes guidance for a
  "student" model that produces the final action.

Custom strategies can be written by subclassing :class:`TurnStrategy` — they
get full read access to the trajectory and can call any model they like.
"""

from __future__ import annotations

import abc
from typing import Any

from psibench.agents.trajectory import Trajectory
from psibench.models.client import ChatClient
from psibench.schemas.messages import RESPOND_ACTION_NAME, Action, Message, Observation
from psibench.utils.parsing import parse_tool_call_args


class TurnStrategy(abc.ABC):
    """Given a trajectory, return the next action.

    Implementations have read-write access to ``trajectory`` (they can append
    to ``trajectory.notes`` or ``trajectory.turns[-1].thoughts`` for
    transparency), and may use the :class:`ChatClient` interface to call any
    OpenAI-compatible model — including multiple models per turn for
    teacher/student-style policies.
    """

    @abc.abstractmethod
    def decide(
        self,
        *,
        trajectory: Trajectory,
        observation: Observation,
        system_prompt: str,
        tools: list[dict[str, Any]] | None,
    ) -> Action:
        ...


# ---------------------------------------------------------------------------
# Direct strategy
# ---------------------------------------------------------------------------


class DirectStrategy(TurnStrategy):
    """A single LLM call; may emit a tool call or a respond action.

    If the first completion produces neither a tool call nor any visible
    ``content`` (e.g. Qwen3 exhausted its budget inside ``<think>``), we
    retry once with thinking disabled and an explicit "commit to an action
    now" nudge so the benchmark doesn't hang on empty assistant turns.
    """

    def __init__(self, client: ChatClient) -> None:
        self.client = client

    def decide(
        self,
        *,
        trajectory: Trajectory,
        observation: Observation,
        system_prompt: str,
        tools: list[dict[str, Any]] | None,
    ) -> Action:
        messages = _build_messages(trajectory, system_prompt)
        res = self.client.complete(messages, tools=tools or None)
        msg = res.choices[0].message
        _record_reasoning(msg, trajectory)

        if _is_empty_message(msg):
            nudge = Message(
                role="system",
                content=(
                    "Your previous turn produced no visible reply and no tool "
                    "call. Commit to an action now: either call exactly one "
                    "tool, or reply to the user in plain text. Do not think "
                    "further — respond immediately."
                ),
            )
            retry_messages = messages + [nudge]
            res = self.client.complete(
                retry_messages,
                tools=tools or None,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            msg = res.choices[0].message
            _record_reasoning(msg, trajectory)

        action = _message_to_action(msg)
        trajectory.messages.append(_assistant_message(msg, action))
        return action


# ---------------------------------------------------------------------------
# Thinking strategy (chain-of-thought)
# ---------------------------------------------------------------------------


class ThinkingStrategy(TurnStrategy):
    """First produce a private thought, then decide on an action.

    The thought is stored on the trajectory but never sent to the user or the
    tool runtime; only the final action is emitted. When ``thinker_client`` is
    ``None``, the same client is used for both passes.
    """

    def __init__(
        self,
        client: ChatClient,
        *,
        thinker_client: ChatClient | None = None,
        think_prompt: str = (
            "Think step-by-step about what to do next. Consider the user's goal, "
            "the domain policy, and any safety concerns. Write a brief private "
            "reasoning trace — it will NOT be sent to the user."
        ),
    ) -> None:
        self.client = client
        self.thinker = thinker_client or client
        self.think_prompt = think_prompt

    def decide(
        self,
        *,
        trajectory: Trajectory,
        observation: Observation,
        system_prompt: str,
        tools: list[dict[str, Any]] | None,
    ) -> Action:
        base_messages = _build_messages(trajectory, system_prompt)

        thought_messages = base_messages + [
            Message(role="system", content=self.think_prompt),
        ]
        thought = self.thinker.complete_text(thought_messages)
        trajectory.record_thought(thought)

        act_messages = base_messages + [
            Message(role="system", content=f"[Private reasoning]\n{thought}"),
        ]
        res = self.client.complete(act_messages, tools=tools or None)
        msg = res.choices[0].message
        _record_reasoning(msg, trajectory)
        action = _message_to_action(msg)
        trajectory.messages.append(_assistant_message(msg, action))
        return action


# ---------------------------------------------------------------------------
# Teacher / student strategy
# ---------------------------------------------------------------------------


class TeacherStudentStrategy(TurnStrategy):
    """A teacher model critiques the trajectory and writes guidance; a student
    model uses that guidance to emit the final action.

    Useful for distillation, oversight, or evaluating how well a smaller
    student model can follow a stronger teacher's advice.
    """

    def __init__(
        self,
        *,
        student_client: ChatClient,
        teacher_client: ChatClient,
        teacher_system_prompt: str = (
            "You are a supervising teacher observing a student agent. Given the "
            "conversation so far, write concise tactical guidance for the "
            "student's next turn. Be specific about what tools to consider, "
            "what to avoid, and any policy concerns. Do not produce the final "
            "response yourself — only the guidance."
        ),
    ) -> None:
        self.student = student_client
        self.teacher = teacher_client
        self.teacher_system_prompt = teacher_system_prompt

    def decide(
        self,
        *,
        trajectory: Trajectory,
        observation: Observation,
        system_prompt: str,
        tools: list[dict[str, Any]] | None,
    ) -> Action:
        base_messages = _build_messages(trajectory, system_prompt)

        teacher_messages = [
            Message(role="system", content=self.teacher_system_prompt),
            *base_messages[1:],
        ]
        guidance = self.teacher.complete_text(teacher_messages)
        trajectory.record_thought(f"[Teacher guidance] {guidance}")

        student_messages = base_messages + [
            Message(role="system", content=f"Teacher guidance for this turn:\n{guidance}"),
        ]
        res = self.student.complete(student_messages, tools=tools or None)
        msg = res.choices[0].message
        _record_reasoning(msg, trajectory)
        action = _message_to_action(msg)
        trajectory.messages.append(_assistant_message(msg, action))
        return action


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_messages(trajectory: Trajectory, system_prompt: str) -> list[Message]:
    msgs: list[Message] = [Message(role="system", content=system_prompt)]
    msgs.extend(trajectory.messages)
    return msgs


def _is_empty_message(msg: Any) -> bool:
    """True if the model emitted neither visible content nor a tool call."""
    tool_calls = getattr(msg, "tool_calls", None) or []
    content = getattr(msg, "content", None) or ""
    return not tool_calls and not content.strip()


def _record_reasoning(msg: Any, trajectory: Trajectory) -> None:
    """Stash a reasoning-parser's ``reasoning_content`` on the current turn."""
    reasoning = getattr(msg, "reasoning_content", None)
    if reasoning:
        trajectory.record_thought(f"[agent reasoning] {reasoning}")


def _message_to_action(msg: Any) -> Action:
    tool_calls = getattr(msg, "tool_calls", None) or []
    if tool_calls:
        call = tool_calls[0]
        return Action(
            name=call.function.name,
            kwargs=parse_tool_call_args(call.function.arguments),
        )
    return Action(name=RESPOND_ACTION_NAME, kwargs={"content": msg.content or ""})


def _assistant_message(msg: Any, action: Action) -> Message:
    if not action.is_respond:
        return Message(
            role="assistant",
            content=msg.content,
            tool_calls=[
                {
                    "id": getattr(msg.tool_calls[0], "id", None),
                    "type": "function",
                    "function": {
                        "name": action.name,
                        "arguments": msg.tool_calls[0].function.arguments,
                    },
                }
            ],
        )
    return Message(role="assistant", content=action.kwargs.get("content", ""))


__all__ = [
    "DirectStrategy",
    "TeacherStudentStrategy",
    "ThinkingStrategy",
    "TurnStrategy",
]
