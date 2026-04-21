#!/usr/bin/env bash
# SLURM job: launch sglang for the model, wait for readiness, then run the
# full-split retail task_accuracy eval against it. Kills the server on exit.
#
# One job = one GPU + one model + one eval split. Logs go to:
#   slurm-<jobid>.out          (SLURM default; everything below)
#   runs/slurm_<jobid>/server.log
#   runs/slurm_<jobid>/eval.log
#
# Submit:
#   sbatch scripts/slurm_serve_and_eval.sbatch
#   sbatch --export=ALL,TASK_SPLIT=dev,LIMIT=5 scripts/slurm_serve_and_eval.sbatch
#   sbatch --export=ALL,MODEL=Qwen/Qwen3-8B scripts/slurm_serve_and_eval.sbatch
#
# REPLACE-ME defaults below (--account, --partition, --qos) must match your
# cluster; the rest are reasonable for Qwen3-4B on one 80GB GPU.
# ----------------------------------------------------------------------------
#SBATCH --job-name=psibench_retail_eval
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --constraint=h100
#SBATCH --time=08:00:00
# #SBATCH --account=REPLACE_ME
# # SBATCH --partition=highgpu
# #SBATCH --qos=REPLACE_ME
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=avinash@ucf.edu

set -euo pipefail

# ---- overridable via --export=ALL,VAR=value --------------------------------
MODEL="${MODEL:-Qwen/Qwen3-4B}"
TASK_SPLIT="${TASK_SPLIT:-test}"
MAX_TURNS="${MAX_TURNS:-20}"
LIMIT="${LIMIT:-}"              # e.g. 5 for a smoke run; empty = all tasks
START="${START:-0}"
USER_MALICIOUS="${USER_MALICIOUS:-none}"
TP_SIZE="${TP_SIZE:-1}"
DP_SIZE="${DP_SIZE:-1}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-64}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
DTYPE="${DTYPE:-bfloat16}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-600}"   # seconds to wait for /v1/models

# Bind the server to localhost on a per-job port to avoid collisions on shared
# nodes. Override PORT from --export if you need a specific one.
HOST="${HOST:-127.0.0.1}"
JOB_ID="${SLURM_JOB_ID:-$$}"
PORT_DEFAULT=$((30000 + (JOB_ID % 1000)))
PORT="${PORT:-${PORT_DEFAULT}}"
BASE_URL="http://${HOST}:${PORT}/v1"
API_KEY="${API_KEY:-EMPTY}"

# ---- working dir + logs ----------------------------------------------------
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
RUN_DIR="runs/slurm_${JOB_ID}"
mkdir -p "${RUN_DIR}"
SERVER_LOG="${RUN_DIR}/server.log"
EVAL_LOG="${RUN_DIR}/eval.log"
SERVER_PID_FILE="${RUN_DIR}/server.pid"

echo "[slurm] job_id=${JOB_ID}  host=${HOST}  port=${PORT}  model=${MODEL}"
echo "[slurm] split=${TASK_SPLIT}  max_turns=${MAX_TURNS}  limit=${LIMIT:-<all>}  start=${START}"
echo "[slurm] run_dir=${RUN_DIR}"
nvidia-smi || true

# ---- start sglang in the background ---------------------------------------
echo "[slurm] launching sglang server → ${SERVER_LOG}"
(
  uv run python -m sglang.launch_server \
    --model-path "${MODEL}" \
    --served-model-name "${MODEL}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --tp-size "${TP_SIZE}" \
    --dp-size "${DP_SIZE}" \
    --mem-fraction-static "${MEM_FRACTION_STATIC}" \
    --max-running-requests "${MAX_RUNNING_REQUESTS}" \
    --context-length "${CONTEXT_LENGTH}" \
    --dtype "${DTYPE}" \
    --reasoning-parser "${REASONING_PARSER}" \
    --tool-call-parser "${TOOL_CALL_PARSER}" \
    --trust-remote-code \
    >"${SERVER_LOG}" 2>&1
) &
SERVER_PID=$!
echo "${SERVER_PID}" >"${SERVER_PID_FILE}"

# Best-effort cleanup on exit (normal, error, or SIGTERM from SLURM).
cleanup() {
  echo "[slurm] cleanup: stopping server pid=${SERVER_PID}"
  if kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -TERM "${SERVER_PID}" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "${SERVER_PID}" 2>/dev/null || break
      sleep 1
    done
    kill -KILL "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# ---- wait until the server answers /v1/models -----------------------------
echo "[slurm] waiting for ${BASE_URL}/models (timeout=${SERVER_READY_TIMEOUT}s)"
deadline=$(( $(date +%s) + SERVER_READY_TIMEOUT ))
until curl -sSf -m 3 "${BASE_URL}/models" >/dev/null 2>&1; do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[slurm] sglang died during startup; tail of server.log:"
    tail -n 80 "${SERVER_LOG}" || true
    exit 1
  fi
  if [[ $(date +%s) -ge ${deadline} ]]; then
    echo "[slurm] server did not become ready within ${SERVER_READY_TIMEOUT}s"
    tail -n 80 "${SERVER_LOG}" || true
    exit 1
  fi
  sleep 3
done
echo "[slurm] server is up"
curl -sS "${BASE_URL}/models" | head -c 400 || true
echo

# ---- run the evaluation ---------------------------------------------------
EVAL_ARGS=(
  --task-split "${TASK_SPLIT}"
  --start "${START}"
  --max-turns "${MAX_TURNS}"
  --model "${MODEL}"
  --base-url "${BASE_URL}"
  --api-key "${API_KEY}"
  --user-malicious "${USER_MALICIOUS}"
  --out-dir "${RUN_DIR}/eval"
)
if [[ -n "${LIMIT}" ]]; then EVAL_ARGS+=(--limit "${LIMIT}"); fi
if [[ -n "${FORCE:-}" && "${FORCE}" != "0" ]]; then EVAL_ARGS+=(--force); fi
if [[ -n "${VERBOSE:-}" && "${VERBOSE}" != "0" ]]; then EVAL_ARGS+=(--verbose); fi

echo "[slurm] uv run python scripts/eval_retail_task_accuracy.py ${EVAL_ARGS[*]}"
uv run python scripts/eval_retail_task_accuracy.py "${EVAL_ARGS[@]}" 2>&1 | tee "${EVAL_LOG}"
rc=${PIPESTATUS[0]}

echo "[slurm] eval rc=${rc}"
exit "${rc}"
