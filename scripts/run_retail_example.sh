#!/usr/bin/env bash
# Drive examples/retail_qwen3_4b.py end-to-end (real agent rollout + evaluate).
#
# Three scenarios are run back-to-back:
#   1. components=task_accuracy            (judge NOT invoked, fastest)
#   2. components=user_satisfaction,safety (judge invoked once)
#   3. components=all                      (judge invoked once; full reward)
#
# Each scenario is a short rollout (MAX_TURNS default = 8) so the whole
# script finishes in minutes rather than tens of minutes. Logs are teed to
# runs/retail_<scenario>_<task>_<ts>.log so you can inspect them afterwards.
#
# Prereqs:
#   ./scripts/serve_qwen3_4b.sh    # sglang on http://0.0.0.0:30000
#
# Usage:
#   ./scripts/run_retail_example.sh
#   TASK_INDEX=37 ./scripts/run_retail_example.sh   # task 37 has non-empty outputs
#   MAX_TURNS=12 TASK_INDEX=2 ./scripts/run_retail_example.sh
#   SCENARIOS="task_accuracy,all" ./scripts/run_retail_example.sh
#   # run just one scenario:
#   ./scripts/run_retail_example.sh user_satisfaction,safety

set -euo pipefail

cd "$(dirname "$0")/.."

BASE_URL="${BASE_URL:-http://localhost:30000/v1}"
API_KEY="${API_KEY:-EMPTY}"
MODEL="${MODEL:-Qwen/Qwen3-4B}"
TASK_INDEX="${TASK_INDEX:-0}"
MAX_TURNS="${MAX_TURNS:-8}"
SCORE_TURN="${SCORE_TURN:-end}"
MALICIOUS="${MALICIOUS:-none}"
TASK_SPLIT="${TASK_SPLIT:-test}"
RUNS_DIR="${RUNS_DIR:-runs}"

mkdir -p "${RUNS_DIR}"

# Pre-flight: confirm the sglang server is reachable.
if ! curl -sSf -m 3 "${BASE_URL}/models" >/dev/null; then
  echo "[run_retail_example] ERROR: ${BASE_URL}/models is not reachable."
  echo "  Start the server first: ./scripts/serve_qwen3_4b.sh"
  exit 1
fi

# Build the scenario list. If the user passed one scenario as $1, use just that.
# Otherwise the SCENARIOS env var (or the default 3) drives it.
if [[ $# -ge 1 ]]; then
  SCENARIOS="$1"
else
  SCENARIOS="${SCENARIOS:-task_accuracy|user_satisfaction,safety|all}"
fi

# Split on | so we can embed commas inside a single scenario.
IFS='|' read -r -a SCENARIO_ARR <<<"${SCENARIOS//,/ }"
# Re-split: if the user supplied "task_accuracy|all", honor |; otherwise fall
# back to the default '|' separated scenarios above.
IFS='|' read -r -a SCENARIO_ARR <<<"${SCENARIOS}"

ts="$(date +%Y%m%d_%H%M%S)"
overall_rc=0

for scenario in "${SCENARIO_ARR[@]}"; do
  safe="${scenario//,/+}"
  log="${RUNS_DIR}/retail_${safe}_t${TASK_INDEX}_${ts}.log"
  echo
  echo "=============================================================="
  echo "[run_retail_example] scenario=${scenario}"
  echo "                     task=${TASK_INDEX}  split=${TASK_SPLIT}"
  echo "                     max_turns=${MAX_TURNS}  score_turn=${SCORE_TURN}"
  echo "                     log=${log}"
  echo "=============================================================="

  set +e
  uv run python examples/retail_qwen3_4b.py \
    --task-index "${TASK_INDEX}" \
    --task-split "${TASK_SPLIT}" \
    --components "${scenario}" \
    --max-turns "${MAX_TURNS}" \
    --score-turn "${SCORE_TURN}" \
    --malicious "${MALICIOUS}" \
    --model "${MODEL}" \
    --base-url "${BASE_URL}" \
    --api-key "${API_KEY}" \
    2>&1 | tee "${log}"
  rc=${PIPESTATUS[0]}
  set -e

  if [[ ${rc} -ne 0 ]]; then
    echo "[run_retail_example] scenario '${scenario}' FAILED (exit=${rc})"
    overall_rc=${rc}
    continue
  fi

  # Extract and print the one-line summary (the JUDGE block).
  echo
  echo "[summary for '${scenario}']"
  grep -E "user_satisfaction=|applicable=|has_outputs=|data_hash\[|per_output=" "${log}" | sed 's/^/  /' || true
done

echo
if [[ ${overall_rc} -eq 0 ]]; then
  echo "[run_retail_example] all scenarios completed successfully."
else
  echo "[run_retail_example] one or more scenarios failed (last rc=${overall_rc})."
fi
exit ${overall_rc}
