#!/usr/bin/env bash
# Launch an sglang OpenAI-compatible server for Qwen/Qwen3-4B.
#
# Usage:
#   ./scripts/serve_qwen3_4b.sh                # start with defaults
#   PORT=30001 ./scripts/serve_qwen3_4b.sh     # override port
#   HOST=0.0.0.0 PORT=8000 ./scripts/serve_qwen3_4b.sh
#
# Once running, point psi-bench at it with an OpenAI-compatible client:
#   base_url = "http://${HOST}:${PORT}/v1"
#   api_key  = "EMPTY"
#   model    = "Qwen/Qwen3-4B"

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3-4B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
TP_SIZE="${TP_SIZE:-1}"
DP_SIZE="${DP_SIZE:-1}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-64}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
DTYPE="${DTYPE:-bfloat16}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen}"

echo "[serve_qwen3_4b] starting sglang server"
echo "  model             = ${MODEL_PATH}"
echo "  served-model-name = ${SERVED_MODEL_NAME}"
echo "  host:port         = ${HOST}:${PORT}"
echo "  tp / dp           = ${TP_SIZE} / ${DP_SIZE}"
echo "  context-length    = ${CONTEXT_LENGTH}"
echo "  dtype             = ${DTYPE}"

ARGS=(
  --model-path "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host "${HOST}"
  --port "${PORT}"
  --tp-size "${TP_SIZE}"
  --dp-size "${DP_SIZE}"
  --mem-fraction-static "${MEM_FRACTION_STATIC}"
  --max-running-requests "${MAX_RUNNING_REQUESTS}"
  --context-length "${CONTEXT_LENGTH}"
  --dtype "${DTYPE}"
)

if [[ -n "${REASONING_PARSER}" && "${REASONING_PARSER}" != "none" ]]; then
  ARGS+=(--reasoning-parser "${REASONING_PARSER}")
fi

if [[ -n "${TOOL_CALL_PARSER}" && "${TOOL_CALL_PARSER}" != "none" ]]; then
  ARGS+=(--tool-call-parser "${TOOL_CALL_PARSER}")
fi

if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
  ARGS+=(--trust-remote-code)
fi

if command -v uv >/dev/null 2>&1; then
  exec uv run python -m sglang.launch_server "${ARGS[@]}"
else
  exec python -m sglang.launch_server "${ARGS[@]}"
fi
