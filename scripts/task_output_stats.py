"""Compute and plot task.outputs + task.actions stats for retail and airline."""

from __future__ import annotations

import statistics
from pathlib import Path

import matplotlib.pyplot as plt

from psibench.envs.airline.tasks_test import TASKS as airline_tasks_test
from psibench.envs.retail.tasks_dev import TASKS_DEV as retail_tasks_dev
from psibench.envs.retail.tasks_test import TASKS_TEST as retail_tasks_test
from psibench.envs.retail.tasks_train import TASKS_TRAIN as retail_tasks_train

RESPOND_ACTION_NAME = "respond"

# pastel palette
C_TOTAL = "#AEC6E4"      # pastel blue
C_WITH = "#A8D5BA"       # pastel green
C_WITHOUT = "#F7B7A3"    # pastel coral
C_ACTIONS = "#C8A2D4"    # pastel purple
C_TOOL = "#F6D186"       # pastel yellow


def _outputs_of(t: object) -> list:
    if isinstance(t, dict):
        return t.get("outputs") or []
    return getattr(t, "outputs", None) or []


def _actions_of(t: object) -> list:
    if isinstance(t, dict):
        return t.get("actions") or []
    return getattr(t, "actions", None) or []


def _action_name(a: object) -> str:
    if isinstance(a, dict):
        return a.get("name", "")
    return getattr(a, "name", "") or ""


def summarize(name: str, tasks: list) -> dict[str, float | int | list[int]]:
    total = len(tasks)

    with_outputs = sum(1 for t in tasks if _outputs_of(t))
    without_outputs = total - with_outputs

    n_actions = [len(_actions_of(t)) for t in tasks]
    n_tool_actions = [
        sum(1 for a in _actions_of(t) if _action_name(a) != RESPOND_ACTION_NAME)
        for t in tasks
    ]
    with_actions = sum(1 for n in n_tool_actions if n > 0)
    without_actions = total - with_actions

    def _safe(fn, xs):
        return fn(xs) if xs else 0

    mean_actions = statistics.mean(n_actions) if n_actions else 0.0
    median_actions = statistics.median(n_actions) if n_actions else 0.0
    max_actions = _safe(max, n_actions)
    min_actions = _safe(min, n_actions)

    mean_tool = statistics.mean(n_tool_actions) if n_tool_actions else 0.0
    total_actions = sum(n_actions)
    total_tool_actions = sum(n_tool_actions)

    print(
        f"{name:<15} total={total:<4} "
        f"outputs[+={with_outputs:<3} -={without_outputs:<3}] "
        f"actions[+={with_actions:<3} -={without_actions:<3}] "
        f"n_actions(mean={mean_actions:.2f} median={median_actions:.1f} "
        f"min={min_actions} max={max_actions}) "
        f"n_tool_actions(mean={mean_tool:.2f} sum={total_tool_actions})"
    )
    return {
        "total": total,
        "with_outputs": with_outputs,
        "without_outputs": without_outputs,
        "with_actions": with_actions,
        "without_actions": without_actions,
        "n_actions": n_actions,
        "n_tool_actions": n_tool_actions,
        "mean_actions": mean_actions,
        "median_actions": median_actions,
        "max_actions": max_actions,
        "min_actions": min_actions,
        "mean_tool_actions": mean_tool,
        "total_actions": total_actions,
        "total_tool_actions": total_tool_actions,
    }


def _label_bars(ax, bars, fmt="{:g}"):
    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            fmt.format(h),
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_outputs_grouped(stats, out_path: Path) -> None:
    labels = list(stats.keys())
    with_vals = [stats[k]["with_outputs"] for k in labels]
    without_vals = [stats[k]["without_outputs"] for k in labels]
    total_vals = [stats[k]["total"] for k in labels]

    x = range(len(labels))
    width = 0.27

    fig, ax = plt.subplots(figsize=(9, 5.5))
    b1 = ax.bar([i - width for i in x], total_vals, width, label="total", color=C_TOTAL, edgecolor="white")
    b2 = ax.bar(x, with_vals, width, label="outputs non-empty", color=C_WITH, edgecolor="white")
    b3 = ax.bar([i + width for i in x], without_vals, width, label="outputs empty", color=C_WITHOUT, edgecolor="white")
    for bars in (b1, b2, b3):
        _label_bars(ax, bars, "{:.0f}")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("number of tasks")
    ax.set_title("psi-bench — task.outputs coverage")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved: {out_path}")


def plot_outputs_stacked(stats, out_path: Path) -> None:
    labels = list(stats.keys())
    with_vals = [stats[k]["with_outputs"] for k in labels]
    without_vals = [stats[k]["without_outputs"] for k in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, with_vals, label="outputs non-empty", color=C_WITH, edgecolor="white")
    ax.bar(labels, without_vals, bottom=with_vals, label="outputs empty", color=C_WITHOUT, edgecolor="white")

    for i, (w, wo) in enumerate(zip(with_vals, without_vals)):
        total = w + wo
        if w > 0:
            ax.text(i, w / 2, f"{w}", ha="center", va="center", fontsize=10)
        if wo > 0:
            ax.text(i, w + wo / 2, f"{wo}", ha="center", va="center", fontsize=10)
        ax.text(i, total + max(total * 0.02, 1), f"total={total}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("number of tasks")
    ax.set_title("psi-bench — task.outputs coverage (stacked)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved: {out_path}")


def plot_actions_summary(stats, out_path: Path) -> None:
    """Mean (and max) number of GT actions per task, per split."""
    labels = list(stats.keys())
    mean_all = [stats[k]["mean_actions"] for k in labels]
    mean_tool = [stats[k]["mean_tool_actions"] for k in labels]
    max_all = [stats[k]["max_actions"] for k in labels]

    x = range(len(labels))
    width = 0.27

    fig, ax = plt.subplots(figsize=(9, 5.5))
    b1 = ax.bar([i - width for i in x], mean_all, width, label="mean actions", color=C_ACTIONS, edgecolor="white")
    b2 = ax.bar(x, mean_tool, width, label="mean tool actions (excl. respond)", color=C_TOOL, edgecolor="white")
    b3 = ax.bar([i + width for i in x], max_all, width, label="max actions", color=C_TOTAL, edgecolor="white")
    for bars, fmt in ((b1, "{:.2f}"), (b2, "{:.2f}"), (b3, "{:.0f}")):
        _label_bars(ax, bars, fmt)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("actions per task")
    ax.set_title("psi-bench — ground-truth actions per task")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved: {out_path}")


def plot_actions_histogram(stats, out_path: Path) -> None:
    """Distribution of GT action counts per task, per split (overlaid)."""
    colors = [C_WITH, C_WITHOUT, C_ACTIONS, C_TOTAL, C_TOOL]
    fig, axes = plt.subplots(1, len(stats), figsize=(4 * len(stats), 4.2), sharey=True)
    if len(stats) == 1:
        axes = [axes]
    for ax, (label, s), color in zip(axes, stats.items(), colors):
        data = s["n_actions"]
        if not data:
            ax.set_title(f"{label}\n(no tasks)")
            continue
        bins = range(0, max(data) + 2)
        ax.hist(data, bins=bins, color=color, edgecolor="white", align="left")
        ax.set_title(f"{label}\nmean={s['mean_actions']:.2f} max={s['max_actions']}")
        ax.set_xlabel("# actions per task")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("# tasks")
    fig.suptitle("psi-bench — distribution of GT actions per task", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved: {out_path}")


def plot_has_actions(stats, out_path: Path) -> None:
    """How many tasks in each split have >=1 GT tool action (excluding respond)."""
    labels = list(stats.keys())
    with_vals = [stats[k]["with_actions"] for k in labels]
    without_vals = [stats[k]["without_actions"] for k in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, with_vals, label="has tool actions (>=1)", color=C_WITH, edgecolor="white")
    ax.bar(labels, without_vals, bottom=with_vals, label="no tool actions", color=C_WITHOUT, edgecolor="white")

    for i, (w, wo) in enumerate(zip(with_vals, without_vals)):
        total = w + wo
        if w > 0:
            ax.text(i, w / 2, f"{w}", ha="center", va="center", fontsize=10)
        if wo > 0:
            ax.text(i, w + wo / 2, f"{wo}", ha="center", va="center", fontsize=10)
        ax.text(i, total + max(total * 0.02, 1), f"total={total}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("number of tasks")
    ax.set_title("psi-bench — task.actions coverage (tool actions only)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved: {out_path}")


def main() -> None:
    stats: dict[str, dict] = {}
    stats["retail train"] = summarize("retail train", retail_tasks_train)
    stats["retail dev"] = summarize("retail dev", retail_tasks_dev)
    stats["retail test"] = summarize("retail test", retail_tasks_test)
    stats["airline test"] = summarize("airline test", airline_tasks_test)

    plots_dir = Path(__file__).resolve().parent.parent / "plots"
    plots_dir.mkdir(exist_ok=True)

    plot_outputs_grouped(stats, plots_dir / "task_outputs_coverage_grouped.png")
    plot_outputs_stacked(stats, plots_dir / "task_outputs_coverage_stacked.png")
    plot_has_actions(stats, plots_dir / "task_actions_coverage_stacked.png")
    plot_actions_summary(stats, plots_dir / "task_actions_summary.png")
    plot_actions_histogram(stats, plots_dir / "task_actions_histogram.png")


if __name__ == "__main__":
    main()
