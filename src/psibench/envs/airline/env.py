"""Airline domain descriptor — data, tools, tasks, wiki, rules."""

from __future__ import annotations

from typing import Any

from psibench.envs.airline.data import load_data
from psibench.envs.airline.rules import RULES
from psibench.envs.airline.tools import ALL_TOOLS
from psibench.envs.airline.wiki import WIKI
from psibench.schemas.messages import Task


TERMINATE_TOOLS: list[str] = ["transfer_to_human_agents"]


def load_tasks(task_split: str = "test") -> list[Task]:
    if task_split != "test":
        raise ValueError(f"Unknown task split: {task_split}")
    from psibench.envs.airline.tasks_test import TASKS as tasks
    return list(tasks)


def airline_domain(task_split: str = "test") -> dict[str, Any]:
    return {
        "env_name": "airline",
        "data_load_func": load_data,
        "tools": list(ALL_TOOLS),
        "tasks": load_tasks(task_split),
        "wiki": WIKI,
        "rules": list(RULES),
        "terminate_tools": list(TERMINATE_TOOLS),
    }


__all__ = ["airline_domain", "load_tasks", "TERMINATE_TOOLS"]
