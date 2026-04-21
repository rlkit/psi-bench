#!/usr/bin/env bash
# Smoke-test the evaluate_components gating on Env.evaluate().
#
# No agent rollout — we just call env.evaluate() directly and confirm:
#   * when only task_accuracy is requested, the judge LLM is NOT called
#   * when user_satisfaction/safety are requested, the judge IS called
#
# Requires sglang to be up at BASE_URL (judge client is constructed even when
# unused, so the base URL just needs to be reachable for model discovery —
# the probe below does not actually make a judge request in the skip case).
#
# Usage:
#   ./scripts/test_eval_components.sh
#   BASE_URL=http://localhost:30000/v1 MODEL=Qwen/Qwen3-4B ./scripts/test_eval_components.sh

set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:30000/v1}"
API_KEY="${API_KEY:-EMPTY}"
MODEL="${MODEL:-Qwen/Qwen3-4B}"
TASK_INDEX="${TASK_INDEX:-0}"

echo "[test_eval_components] base_url=${BASE_URL}  model=${MODEL}  task_index=${TASK_INDEX}"

uv run python - <<PY
from psibench.envs import get_env

BASE_URL = "${BASE_URL}"
API_KEY  = "${API_KEY}"
MODEL    = "${MODEL}"
TI       = int("${TASK_INDEX}")

def mk(components):
    return get_env(
        env_name="retail",
        user_strategy="llm", user_model=MODEL, user_provider="openai",
        user_base_url=BASE_URL, user_api_key=API_KEY,
        judge_model=MODEL, judge_provider="openai",
        judge_base_url=BASE_URL, judge_api_key=API_KEY,
        task_split="test", task_index=TI,
        score_turn="end", max_turns=4,
        evaluate_components=components,
    )

print()
print("=== A) components = {'task_accuracy'} — judge should NOT run ===")
env = mk({"task_accuracy"})
r = env.evaluate()
judge_calls = env.judge.client.cumulative_usage.calls
print(f"  task_accuracy     = {r.task_accuracy}")
print(f"  output_match      = {r.output_match} (expect None)")
print(f"  user_satisfaction = {r.user_satisfaction_score} (expect 0.0 — judge skipped)")
print(f"  safety            = {r.safety_score} (expect 0.0 — judge skipped)")
print(f"  total_score       = {r.total_score}  (== task_accuracy)")
print(f"  applicable        = {r.info.applicable_components}")
print(f"  weights_used      = {r.info.weights_used}")
print(f"  judge calls       = {judge_calls}")
assert judge_calls == 0, f"judge should not be called; got {judge_calls} calls"
assert r.output_match is None
print("  [ok] judge was skipped")

print()
print("=== B) components = {'user_satisfaction','safety'} — judge SHOULD run ===")
env = mk({"user_satisfaction", "safety"})
r = env.evaluate()
judge_calls = env.judge.client.cumulative_usage.calls
print(f"  user_satisfaction = {r.user_satisfaction_score}")
print(f"  safety            = {r.safety_score}")
print(f"  task_accuracy     = {r.task_accuracy} (expect None — not requested)")
print(f"  total_score       = {r.total_score}")
print(f"  applicable        = {r.info.applicable_components}")
print(f"  weights_used      = {r.info.weights_used}")
print(f"  judge calls       = {judge_calls}")
assert judge_calls == 1, f"judge should be called once; got {judge_calls} calls"
assert r.task_accuracy is None
print("  [ok] judge ran")

print()
print("=== C) components = 'all' — task_accuracy + satisfaction + safety (output_match is N/A for task 0, outputs=[]) ===")
env = mk({"user_satisfaction", "safety", "task_accuracy", "output_match"})
r = env.evaluate()
judge_calls = env.judge.client.cumulative_usage.calls
print(f"  user_satisfaction = {r.user_satisfaction_score}")
print(f"  safety            = {r.safety_score}")
print(f"  task_accuracy     = {r.task_accuracy}")
print(f"  output_match      = {r.output_match} (task 0 has no outputs → None)")
print(f"  total_score       = {r.total_score}")
print(f"  applicable        = {r.info.applicable_components}")
print(f"  weights_used      = {r.info.weights_used}")
print(f"  has_outputs       = {r.info.has_outputs}  n_outputs={r.info.n_outputs}")
print(f"  has_actions       = {r.info.has_actions}  n_gt_tool_actions={r.info.n_gt_tool_actions}")
print(f"  judge calls       = {judge_calls}")

print()
print("[done] all component-gating assertions passed")
PY
