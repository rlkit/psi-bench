"""Retail domain descriptor — data, tools, tasks, wiki, rules."""

from __future__ import annotations

from typing import Any

from psibench.envs.retail.data import load_data
from psibench.envs.retail.rules import RULES
from psibench.envs.retail.tools import ALL_TOOLS
from psibench.envs.retail.wiki import WIKI
from psibench.schemas.messages import Task


TERMINATE_TOOLS: list[str] = ["transfer_to_human_agents"]


def load_tasks(task_split: str = "test") -> list[Task]:
    if task_split == "test":
        from psibench.envs.retail.tasks_test import TASKS_TEST as tasks
    elif task_split == "train":
        from psibench.envs.retail.tasks_train import TASKS_TRAIN as tasks
    elif task_split == "dev":
        from psibench.envs.retail.tasks_dev import TASKS_DEV as tasks
    else:
        raise ValueError(f"Unknown task split: {task_split}")
    return list(tasks)


def retail_domain(task_split: str = "test") -> dict[str, Any]:
    """Return components needed by :func:`psibench.envs.base.build_env`."""
    return {
        "env_name": "retail",
        "data_load_func": load_data,
        "tools": list(ALL_TOOLS),
        "tasks": load_tasks(task_split),
        "wiki": WIKI,
        "rules": list(RULES),
        "terminate_tools": list(TERMINATE_TOOLS),
    }


__all__ = ["load_tasks", "retail_domain", "TERMINATE_TOOLS"]
