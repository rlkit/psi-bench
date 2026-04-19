"""Smoke tests that don't need network access."""

from __future__ import annotations

from psibench.schemas.reward import Reward


def test_reward_compute_total_default_is_mean():
    total = Reward.compute_total(1.0, 0.0)
    assert abs(total - 0.5) < 1e-9


def test_reward_compute_total_weights():
    total = Reward.compute_total(1.0, 0.0, satisfaction_weight=0.8, safety_weight=0.2)
    assert abs(total - 0.8) < 1e-9


def test_retail_domain_loads():
    from psibench.envs.retail import retail_domain

    d = retail_domain("test")
    assert d["env_name"] == "retail"
    assert d["tools"]
    assert d["tasks"]
    assert "transfer_to_human_agents" in d["terminate_tools"]


def test_airline_domain_loads():
    from psibench.envs.airline import airline_domain

    d = airline_domain("test")
    assert d["env_name"] == "airline"
    assert d["tools"]
    assert d["tasks"]
