#!/usr/bin/env bash
# Install psi-bench dependencies with the correct extra for the current machine.
#
# Logic:
#   - macOS / no GPU     -> base install (no sglang)
#   - Linux + H100/H200  -> --extra h100
#   - Linux + A100       -> --extra a100
#   - Linux + L40/L40S/L4/4090 -> --extra l40
#   - Linux + other GPU  -> --extra serve
#
# Usage:
#   ./scripts/install_deps.sh              # auto-detect
#   ./scripts/install_deps.sh h100         # force a specific extra
#   EXTRA=a100 ./scripts/install_deps.sh   # same, via env

set -euo pipefail

EXTRA="${1:-${EXTRA:-}}"

if [[ -z "${EXTRA}" ]]; then
  if [[ "$(uname -s)" != "Linux" ]]; then
    EXTRA="base"
  elif ! command -v nvidia-smi >/dev/null 2>&1; then
    EXTRA="base"
  else
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || true)"
    echo "[install_deps] detected GPU: ${GPU_NAME:-<none>}"
    shopt -s nocasematch
    case "${GPU_NAME}" in
      *H100*|*H200*)            EXTRA="h100" ;;
      *A100*)                   EXTRA="a100" ;;
      *L40*|*L4\ *|*L4$|*4090*) EXTRA="l40"  ;;
      "")                       EXTRA="base" ;;
      *)                        EXTRA="serve" ;;
    esac
    shopt -u nocasematch
  fi
fi

echo "[install_deps] selected extra: ${EXTRA}"

if [[ "${EXTRA}" == "base" ]]; then
  exec uv sync
else
  exec uv sync --extra "${EXTRA}"
fi
