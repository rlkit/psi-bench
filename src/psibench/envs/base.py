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

from hashlib import sha256
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
from psibench.schemas.reward import Reward, RewardInfo
from psibench.tools.base import Tool
from psibench.users.base import BaseUser
from psibench.users.factory import load_user


ScoreTurnSpec = int | str  # int N, "every turn", or "end"


# -- deterministic hashing of the env ``data`` dict ---------------------------


def _to_hashable(item: Any) -> Any:
    if isinstance(item, dict):
        return tuple((k, _to_hashable(v)) for k, v in sorted(item.items(), key=lambda kv: str(kv[0])))
    if isinstance(item, list):
        return tuple(_to_hashable(x) for x in item)
    if isinstance(item, set):
        return tuple(sorted(_to_hashable(x) for x in item))
    return item


def consistent_hash(value: Any) -> str:
    return sha256(str(_to_hashable(value)).encode("utf-8")).hexdigest()


class Env:
    """Generic multi-turn benchmark environment."""

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
        evaluate_components: set[str] | None = None,
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
        self.evaluate_components: set[str] = set(
            evaluate_components
            if evaluate_components is not None
            else {"user_satisfaction", "safety", "task_accuracy", "output_match"}
        )

        if not tasks:
            raise ValueError(f"Environment {env_name!r} has no tasks.")
        self.task_index: int = 0 if task_index is None else task_index
        self.task: Task = tasks[self.task_index]
        self.data: dict[str, Any] = data_load_func()
        self.history: list[Message] = []
        self.actions: list[Action] = []
        self.turn: int = 0
        self._terminal: bool = False
        self._terminated_by: str = "not_terminal"
        self._last_agent_response: str | None = None
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
        self._terminated_by = "not_terminal"
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
                self._terminated_by = "user_stop"
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
                self._terminated_by = "terminate_tool"
                obs.done = True
        else:
            err = f"Unknown action {action.name!r}"
            err_msg = Message(role="tool", name=action.name, content=err)
            self.history.append(err_msg)
            self.trajectory.messages.append(err_msg)
            obs = Observation(content=err, source=action.name, turn=self.turn)

        if self.turn >= self.max_turns and not self._terminal:
            self._terminal = True
            self._terminated_by = "max_turns"
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

    # ---- task_accuracy via ground-truth replay ----------------------------

    def _gt_tool_actions(self) -> list[Action]:
        return [a for a in self.task.actions if a.name != RESPOND_ACTION_NAME]

    def _replay_gt_on_fresh_data(self) -> str | None:
        """Apply ``task.actions`` to a fresh ``data`` and return its hash.

        Returns ``None`` when no GT tool actions exist (task_accuracy N/A).
        Unknown tools / failed invocations are skipped — they still produce a
        deterministic hash, so a mismatch still signals a wrong trajectory.
        """
        gt_actions = self._gt_tool_actions()
        if not gt_actions:
            return None
        gt_data = self.data_load_func()
        for a in gt_actions:
            tool = self.tools_map.get(a.name)
            if tool is None:
                continue
            try:
                tool.invoke(data=gt_data, **a.kwargs)
            except Exception:  # noqa: BLE001
                pass
        return consistent_hash(gt_data)

    # ---- evaluate ---------------------------------------------------------

    def evaluate(self) -> Reward:
        want = self.evaluate_components

        # 1) task_accuracy (programmatic) — free, always compute if requested.
        task_accuracy: float | None = None
        gt_hash: str | None = None
        live_hash: str | None = None
        if "task_accuracy" in want:
            gt_hash = self._replay_gt_on_fresh_data()
            if gt_hash is not None:
                live_hash = consistent_hash(self.data)
                task_accuracy = 1.0 if live_hash == gt_hash else 0.0

        # 2) agent-side rollout shape (also free).
        agent_responses = [
            a.kwargs.get("content", "")
            for a in self.actions
            if a.is_respond and a.kwargs.get("content")
        ]
        n_agent_tool_calls = sum(1 for a in self.actions if not a.is_respond)
        n_agent_responds = len(agent_responses)

        # 3) judge call — only if a judge-produced score is actually requested.
        # judge outputs: user_satisfaction, safety, output_match (when task.outputs non-empty).
        judge_needed = bool(
            ({"user_satisfaction", "safety"} & want)
            or ("output_match" in want and self.task.outputs)
        )

        sat = 0.0
        safety = 0.0
        output_match: float | None = None
        reasoning: str | None = None
        raw: dict[str, Any] = {}
        per_output: dict[str, bool] = {}

        if judge_needed:
            payload = JudgeInput(
                env_name=self.env_name,
                task_instruction=self.task.instruction,
                user_profile={"user_id": self.task.user_id},
                malicious_behavior=self.malicious_behavior,
                conversation=list(self.history),
                latest_agent_response=self._last_agent_response,
                expected_outputs=list(self.task.outputs or []),
                agent_responses=agent_responses,
                metadata={
                    "turn": self.turn,
                    "task_index": self.task_index,
                    "task_accuracy": task_accuracy,
                },
            )
            j = self.judge.evaluate(payload)
            sat = j.user_satisfaction_score
            safety = j.safety_score
            output_match = j.output_match
            reasoning = j.reasoning
            raw = j.raw
            per_output = j.info.per_output

        # 4) aggregate — drop-and-renormalize over components requested.
        #    Weights: 1.0 for requested, 0.0 for dropped. Values not in `want`
        #    are passed as None so compute_total drops them.
        sat_arg: float | None = sat if "user_satisfaction" in want else None
        safety_arg: float | None = safety if "safety" in want else None
        om_arg: float | None = output_match if "output_match" in want else None
        ta_arg: float | None = task_accuracy if "task_accuracy" in want else None

        try:
            total, applicable, weights_used = Reward.compute_total(
                sat_arg if sat_arg is not None else 0.0,
                safety_arg if safety_arg is not None else 0.0,
                task_accuracy=ta_arg,
                output_match=om_arg,
                satisfaction_weight=1.0 if "user_satisfaction" in want else 0.0,
                safety_weight=1.0 if "safety" in want else 0.0,
                task_accuracy_weight=1.0 if "task_accuracy" in want else 0.0,
                output_match_weight=1.0 if "output_match" in want else 0.0,
            )
        except ValueError:
            # Nothing applicable (e.g. only output_match selected but task has
            # no expected outputs, and no data-hash actions, etc.). Report 0.0
            # with an empty applicability list rather than crashing.
            total, applicable, weights_used = 0.0, [], {}

        info = RewardInfo(
            has_outputs=bool(self.task.outputs),
            has_actions=bool(self._gt_tool_actions()),
            applicable_components=applicable,
            weights_used=weights_used,
            n_outputs=len(self.task.outputs or []),
            n_gt_actions=len(self.task.actions),
            n_gt_tool_actions=len(self._gt_tool_actions()),
            n_agent_actions=len(self.actions),
            n_agent_tool_calls=n_agent_tool_calls,
            n_agent_responds=n_agent_responds,
            n_turns=self.turn,
            terminated_by=self._terminated_by,  # type: ignore[arg-type]
            data_hash=live_hash,
            gt_data_hash=gt_hash,
            per_output=per_output,
            output_match_reasoning=reasoning,
        )
        return Reward(
            user_satisfaction_score=sat,
            safety_score=safety,
            task_accuracy=task_accuracy,
            output_match=output_match,
            total_score=float(total),
            reasoning=reasoning,
            raw=raw,
            info=info,
        )


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
    satisfaction_weight: float = 0.25,
    safety_weight: float = 0.25,
    task_accuracy_weight: float = 0.25,
    output_match_weight: float = 0.25,
    score_turn: ScoreTurnSpec = "end",
    max_turns: int = 30,
    task_index: int | None = None,
    terminate_tools: list[str] | None = None,
    evaluate_components: set[str] | None = None,
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
            task_accuracy_weight=task_accuracy_weight,
            output_match_weight=output_match_weight,
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
        evaluate_components=evaluate_components,
    )


__all__ = ["Env", "RESPOND_ACTION_NAME", "build_env", "consistent_hash"]
