# psi-bench

Benchmark environments where an **LLM agent** interacts with an **LLM-simulated
user**, and the whole interaction is scored by an external **judge** (HTTP API
or LLM). Ships with **retail** and **airline** domains out of the box.

- Multi-turn agent ↔ user conversations with tool use
- LLM user simulation with optional adversarial profiles (`cheat`,
  `prompt_injection`, `unsafe_request`, `jailbreak_attempt`)
- Judge-based rewards — no hardcoded success rules
- Typed `Reward` object with `user_satisfaction_score`, `safety_score`, `total_score`
- Pluggable **turn strategies** — Direct, Thinking, Teacher/Student — that
  have full access to the trajectory
- OpenAI-compatible clients, so sglang / vLLM / local servers just work

## Install

```bash
uv venv
uv pip install -e ".[dev]"
# or
uv sync
```

## Quickstart

```python
from psibench.envs import get_env
from psibench.agents import Agent

env = get_env(
    env_name="retail",
    user_strategy="llm",
    user_model="gpt-4o",
    user_provider="openai",
    user_malicious_behavior="cheat",
    judge_url="http://localhost:8000/judge",             # OR use an LLM judge:
    # judge_model="gpt-4o",
    # judge_system_prompt="You are a judge that scores user satisfaction and safety.",
    score_turn=3,
    task_index=0,
)

agent = Agent(model="gpt-4o", provider="openai", temperature=0.0)
agent.bind_env(wiki=env.wiki, rules=env.rules, tools=env.tools_info)

observation = env.reset()
agent.reset(task_instruction=env.task.instruction, env_name=env.env_name)

while True:
    action = agent.step(observation)
    observation = env.step(action)
    if env.should_evaluate():
        reward = env.evaluate()
        print(reward)
    if env.is_terminal():
        break
```

## Environment concepts

| Concept | What it is |
| --- | --- |
| **Env** | Orchestrates a single episode: resets data, runs agent ↔ user turns, invokes tools, calls the judge. No hardcoded reward. |
| **User** | `LLMUser` drives the simulated customer. Configurable model + provider + adversarial profile. Receives the full trajectory each turn. |
| **Agent** | `BaseAgent` + pluggable `TurnStrategy`. Owns a `Trajectory` and gets domain tools + wiki/rules via `bind_env`. |
| **Judge** | `HTTPJudge` (posts the payload to `judge_url`) or `LLMJudge` (uses an OpenAI-compatible model). Returns a typed `Reward`. |
| **Reward** | `user_satisfaction_score`, `safety_score`, `total_score` (weighted mean by default), plus optional `reasoning`. |
| **Trajectory** | Every turn's observation + action + thoughts + flattened message log. Shared with the user simulator so it can react to the agent's tool use, not just its chat. |

An episode terminates when:
- the user emits `###STOP###`,
- the agent calls a registered terminate tool (e.g. `transfer_to_human_agents`),
- or `max_turns` is reached.

## Judge modes

### HTTP judge (external service)

```python
env = get_env(..., judge_url="https://judge.example.com/score")
```

The service receives a JSON body matching
`psibench.judges.base.JudgeInput` and must return JSON with
`user_satisfaction_score`, `safety_score`, optional `total_score`, and
optional `reasoning`.

### LLM judge (OpenAI-compatible)

```python
env = get_env(
    ...,
    judge_model="gpt-4o",
    judge_system_prompt="You are a strict evaluator ...",
)
```

## Turn strategies

Agents delegate each turn to a `TurnStrategy` that has read/write access to the
full `Trajectory`. Three are built in:

```python
from psibench.agents import LLMAgent, ThinkingStrategy, TeacherStudentStrategy
from psibench.models import ChatClient

# Thinking: private CoT, then act
agent = LLMAgent(model="gpt-4o")
agent.strategy = ThinkingStrategy(agent.client)

# Teacher/student: stronger teacher writes guidance, smaller student acts
teacher = ChatClient(model="gpt-4o")
student = ChatClient(model="gpt-4o-mini")
agent = LLMAgent(model="gpt-4o-mini",
                 strategy=TeacherStudentStrategy(
                     student_client=student, teacher_client=teacher))
```

Implement a custom strategy by subclassing `TurnStrategy.decide()` — it gets
`trajectory`, `observation`, `system_prompt`, and `tools`, and can call any
number of models. Override `BaseAgent.turn_strategy()` to switch strategies
dynamically per turn.

## Running with OpenAI

```python
get_env(..., user_provider="openai", user_model="gpt-4o")
```

`OPENAI_API_KEY` is read from the environment, or can be passed via
`user_api_key` / `judge_api_key`.

## Running with sglang / vLLM / other OpenAI-compatible servers

Point `*_base_url` at the server:

```python
env = get_env(
    env_name="retail",
    user_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    user_provider="sglang",
    user_base_url="http://localhost:30000/v1",
    user_api_key="EMPTY",
    judge_model="meta-llama/Meta-Llama-3.1-70B-Instruct",
    judge_base_url="http://localhost:30001/v1",
    judge_api_key="EMPTY",
    ...
)
agent = Agent(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    provider="sglang",
    base_url="http://localhost:30000/v1",
    api_key="EMPTY",
)
```

## Adding a new environment

1. Create a module with `load_data`, `TOOLS`, `TASKS`, `WIKI`, `RULES`.
2. Expose a `my_domain(task_split)` function returning
   `{env_name, data_load_func, tools, tasks, wiki, rules, terminate_tools}`.
3. Register it:

   ```python
   from psibench.envs import register_env
   from my_pkg.env import my_domain

   register_env("my_domain", my_domain)
   ```

Now `get_env("my_domain", ...)` works.

## Dev commands

```bash
uv sync                      # install
uv run pytest                # tests
uv run ruff check src tests  # lint
uv run ruff format src tests # format
uv run mypy src              # type-check
uv build                     # build sdist + wheel
uv publish                   # publish to PyPI
```

## License

MIT.
