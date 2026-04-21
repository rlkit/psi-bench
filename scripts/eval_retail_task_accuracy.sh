#!/usr/bin/env bash
# Thin wrapper around scripts/eval_retail_task_accuracy.py.
#
# Runs all retail-test tasks (115) with components=task_accuracy, saves each
# task's trajectory + reward under runs/retail_task_accuracy_test_<ts>/.
#
# Examples:
#   ./scripts/eval_retail_task_accuracy.sh                # all 115 test tasks
#   LIMIT=5 ./scripts/eval_retail_task_accuracy.sh        # just the first 5
#   START=40 LIMIT=20 ./scripts/eval_retail_task_accuracy.sh
#   TASK_SPLIT=dev ./scripts/eval_retail_task_accuracy.sh # 20 dev tasks
#   MAX_TURNS=16 VERBOSE=1 ./scripts/eval_retail_task_accuracy.sh
#   OUT_DIR=runs/my_run ./scripts/eval_retail_task_accuracy.sh
#
# Resume: re-running with the same OUT_DIR skips task files that already exist.
# Pass FORCE=1 to overwrite them.

set -euo pipefail
cd "$(dirname "$0")/.."

BASE_URL="${BASE_URL:-http://localhost:30000/v1}"
API_KEY="${API_KEY:-EMPTY}"
MODEL="${MODEL:-Qwen/Qwen3-4B}"
TASK_SPLIT="${TASK_SPLIT:-test}"
MAX_TURNS="${MAX_TURNS:-20}"
START="${START:-0}"

if ! curl -sSf -m 3 "${BASE_URL}/models" >/dev/null; then
  echo "[eval_retail_task_accuracy] ERROR: ${BASE_URL}/models not reachable."
  echo "  Start the server first: ./scripts/serve_qwen3_4b.sh"
  exit 1
fi

ARGS=(
  --task-split "${TASK_SPLIT}"
  --start "${START}"
  --max-turns "${MAX_TURNS}"
  --model "${MODEL}"
  --base-url "${BASE_URL}"
  --api-key "${API_KEY}"
)
if [[ -n "${LIMIT:-}" ]]; then ARGS+=(--limit "${LIMIT}"); fi
if [[ -n "${OUT_DIR:-}" ]]; then ARGS+=(--out-dir "${OUT_DIR}"); fi
if [[ -n "${USER_MALICIOUS:-}" ]]; then ARGS+=(--user-malicious "${USER_MALICIOUS}"); fi
if [[ -n "${FORCE:-}" && "${FORCE}" != "0" ]]; then ARGS+=(--force); fi
if [[ -n "${VERBOSE:-}" && "${VERBOSE}" != "0" ]]; then ARGS+=(--verbose); fi

echo "[eval_retail_task_accuracy] uv run python scripts/eval_retail_task_accuracy.py ${ARGS[*]}"
exec uv run python scripts/eval_retail_task_accuracy.py "${ARGS[@]}"
