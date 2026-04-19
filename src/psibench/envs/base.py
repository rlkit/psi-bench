"""Base environment — judge-based, no hardcoded reward logic.

Design summary (see ``CLAUDE.md`` for the spec)::

    env = get_env(...)
    observation = env.reset()
    while True:
        action = agent.step(observation)
        observation = env.step(action)
        if env.should_evaluate():
            reward = env.evaluate()
        if env.is_terminal():
            break
"""

from __future__ import annotations

from typing import Any, Callable

from psibench.agents.trajectory import Trajectory
from psibench.judges.base import BaseJudge, JudgeInput
from psibench.judges.factory import load_judge
from psibench.schemas.config import JudgeConfig, MaliciousBehavior
from psibench.schemas.messages import (
    RESPOND_ACTION_NAME,
    Action,
    Message,
    Observation,
    Task,
)
from psibench.schemas.reward import Reward
from psibench.tools.base import Tool
from psibench.users.base import BaseUser
from psibench.users.factory import load_user


ScoreTurnSpec = int | str  # int N, "every turn", or "end"


class Env:
    """Generic multi-turn benchmark environment.

    The agent emits :class:`Action`s (either a ``respond`` text action or a
    named tool call). Tool calls mutate ``self.data`` via :class:`Tool`
    subclasses; ``respond`` actions go to the simulated :class:`BaseUser`,
    whose reply becomes the next observation.

    Termination is explicit — no ``done`` reward-termination coupling:
      * ``max_turns`` reached,
      * the user emits the stop sentinel,
      * the agent invokes a registered terminate tool.
    """

    def __init__(
        self,
        *,
        env_name: str,
        data_load_func: Callable[[], dict[str, Any]],
        tools: list[type[Tool]],
        tasks: list[Task],
        wiki: str,
        rules: list[str],
        user: BaseUser,
        judge: BaseJudge,
        task_index: int | None = None,
        score_turn: ScoreTurnSpec = "end",
        max_turns: int = 30,
        malicious_behavior: MaliciousBehavior = "none",
        terminate_tools: list[str] | None = None,
    ) -> None:
        self.env_name = env_name
        self.data_load_func = data_load_func
        self.tools_map: dict[str, type[Tool]] = {
            t.get_info()["function"]["name"]: t for t in tools
        }
        self.tools_info: list[dict[str, Any]] = [t.get_info() for t in tools]
        self.tasks = tasks
        self.wiki = wiki
        self.rules = rules
        self.user = user
        self.judge = judge
        self.score_turn = score_turn
        self.max_turns = max_turns
        self.malicious_behavior = malicious_behavior
        self.terminate_tools = list(terminate_tools or [])

        if not tasks:
            raise ValueError(f"Environment {env_name!r} has no tasks.")
        self.task_index: int = 0 if task_index is None else task_index
        self.task: Task = tasks[self.task_index]
        self.data: dict[str, Any] = data_load_func()
        self.history: list[Message] = []
        self.actions: list[Action] = []
        self.turn: int = 0
        self._terminal: bool = False
        self._last_agent_response: str | None = None
        # A trajectory mirror kept by the env; both the user simulator and the
        # judge receive this so they always see the same state as the agent.
        self.trajectory: Trajectory = Trajectory(env_name=env_name)

    # ---- lifecycle ---------------------------------------------------------

    def reset(self, task_index: int | None = None) -> Observation:
        if task_index is not None:
            self.task_index = task_index
        self.task = self.tasks[self.task_index]
        self.data = self.data_load_func()
        self.history = []
        self.actions = []
        self.turn = 0
        self._terminal = False
        self._last_agent_response = None
        self.trajectory = Trajectory(
            env_name=self.env_name,
            task_instruction=self.task.instruction,
        )

        opening = self.user.reset(instruction=self.task.instruction, trajectory=self.trajectory)
        self.history.append(Message(role="user", content=opening))
        obs = Observation(content=opening, source="user", turn=self.turn)
        self.trajectory.messages.append(Message(role="user", content=opening))
        self.trajectory.add_observation(obs)
        return obs

    def step(self, action: Action) -> Observation:
        if self._terminal:
            raise RuntimeError("Environment is terminal; call reset() first.")

        self.actions.append(action)
        self.trajectory.record_action(action)
        self.turn += 1

        if action.is_respond:
            agent_text = action.kwargs.get("content", "")
            self._last_agent_response = agent_text
            agent_msg = Message(role="assistant", content=agent_text)
            self.history.append(agent_msg)
            self.trajectory.messages.append(agent_msg)
            user_reply = self.user.step(agent_text, trajectory=self.trajectory)
            user_msg = Message(role="user", content=user_reply)
            self.history.append(user_msg)
            self.trajectory.messages.append(user_msg)
            obs = Observation(
                content=user_reply,
                source="user",
                turn=self.turn,
                done=self.user.is_done,
            )
            if self.user.is_done:
                self._terminal = True
        elif action.name in self.tools_map:
            try:
                result = self.tools_map[action.name].invoke(data=self.data, **action.kwargs)
            except Exception as e:  # noqa: BLE001
                result = f"Error: {e}"
            tool_text = str(result)
            tc_msg = Message(
                role="assistant",
                content=None,
                tool_calls=[{"name": action.name, "arguments": action.kwargs}],
            )
            tool_msg = Message(role="tool", name=action.name, content=tool_text)
            self.history.append(tc_msg)
            self.history.append(tool_msg)
            self.trajectory.messages.append(tc_msg)
            self.trajectory.messages.append(tool_msg)
            obs = Observation(content=tool_text, source=action.name, turn=self.turn)
            if action.name in self.terminate_tools:
                self._terminal = True
                obs.done = True
        else:
            err = f"Unknown action {action.name!r}"
            err_msg = Message(role="tool", name=action.name, content=err)
            self.history.append(err_msg)
            self.trajectory.messages.append(err_msg)
            obs = Observation(content=err, source=action.name, turn=self.turn)

        if self.turn >= self.max_turns:
            self._terminal = True
            obs.done = True
        self.trajectory.add_observation(obs)
        return obs

    # ---- evaluation --------------------------------------------------------

    def should_evaluate(self) -> bool:
        if self.score_turn == "end":
            return self._terminal
        if self.score_turn == "every turn":
            return self.turn > 0
        if isinstance(self.score_turn, int):
            if self.score_turn <= 0:
                return self._terminal
            return self.turn > 0 and (self.turn % self.score_turn == 0 or self._terminal)
        raise ValueError(f"Invalid score_turn: {self.score_turn!r}")

    def is_terminal(self) -> bool:
        return self._terminal

    def evaluate(self) -> Reward:
        payload = JudgeInput(
            env_name=self.env_name,
            task_instruction=self.task.instruction,
            user_profile={"user_id": self.task.user_id},
            malicious_behavior=self.malicious_behavior,
            conversation=list(self.history),
            latest_agent_response=self._last_agent_response,
            metadata={"turn": self.turn, "task_index": self.task_index},
        )
        return self.judge.evaluate(payload)


# --- helper to build an Env from loose kwargs (used by get_env) --------------


def build_env(
    *,
    env_name: str,
    data_load_func: Callable[[], dict[str, Any]],
    tools: list[type[Tool]],
    tasks: list[Task],
    wiki: str,
    rules: list[str],
    user_strategy: str = "llm",
    user_model: str = "gpt-4o",
    user_provider: str = "openai",
    user_base_url: str | None = None,
    user_api_key: str | None = None,
    user_temperature: float = 0.7,
    user_malicious_behavior: MaliciousBehavior = "none",
    judge_url: str | None = None,
    judge_system_prompt: str | None = None,
    judge_model: str | None = None,
    judge_provider: str = "openai",
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
    judge_temperature: float = 0.0,
    satisfaction_weight: float = 0.5,
    safety_weight: float = 0.5,
    score_turn: ScoreTurnSpec = "end",
    max_turns: int = 30,
    task_index: int | None = None,
    terminate_tools: list[str] | None = None,
) -> Env:
    user = load_user(
        strategy=user_strategy,
        model=user_model,
        provider=user_provider,
        base_url=user_base_url,
        api_key=user_api_key,
        temperature=user_temperature,
        malicious_behavior=user_malicious_behavior,
    )
    judge = load_judge(
        JudgeConfig(
            url=judge_url,
            system_prompt=judge_system_prompt,
            model=judge_model,
            provider=judge_provider,
            base_url=judge_base_url,
            api_key=judge_api_key,
            temperature=judge_temperature,
            satisfaction_weight=satisfaction_weight,
            safety_weight=safety_weight,
        )
    )
    return Env(
        env_name=env_name,
        data_load_func=data_load_func,
        tools=tools,
        tasks=tasks,
        wiki=wiki,
        rules=rules,
        user=user,
        judge=judge,
        task_index=task_index,
        score_turn=score_turn,
        max_turns=max_turns,
        malicious_behavior=user_malicious_behavior,
        terminate_tools=terminate_tools,
    )


__all__ = ["Env", "RESPOND_ACTION_NAME", "build_env"]
