#!/bin/bash
set -euo pipefail

# Usage:
#   ./scripts/response/run_response_local.sh \
#     --unperturbed-dir <dir> \
#     --response-config <path> \
#     --output-dir <dir> \
#     --transient <t> \
#     --workers <n> \
#     [--flush-every <n>] \
#     [--base-seed <n>]
#
# Example:
#   ./scripts/response/run_response_local.sh \
#     --unperturbed-dir results/linear_response/poisson/critical/n1000/unperturbed_runs/graph_0001 \
#     --response-config configs/linear_response/poisson/perturbed_runs/critical/response_config_constant.json \
#     --output-dir results/linear_response/poisson/critical/n1000/perturbed_runs/graph_0001 \
#     --transient 5000 \
#     --workers 4 \
#     --flush-every 50

export HDF5_USE_FILE_LOCKING=FALSE

UNPERTURBED_DIR=""
RESPONSE_CONFIG=""
OUTPUT_DIR=""
NUM_WORKERS=""
FLUSH_EVERY="50"
BASE_SEED=""
TRANSIENT=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --unperturbed-dir) UNPERTURBED_DIR="$2"; shift 2;;
    --response-config) RESPONSE_CONFIG="$2"; shift 2;;
    --output-dir) OUTPUT_DIR="$2"; shift 2;;
    --transient) TRANSIENT="$2"; shift 2;;
    --workers) NUM_WORKERS="$2"; shift 2;;
    --flush-every) FLUSH_EVERY="$2"; shift 2;;
    --base-seed) BASE_SEED="$2"; shift 2;;
    *) echo "Unknown argument: $1"; exit 2;;
  esac
done

: "${UNPERTURBED_DIR:?--unperturbed-dir is required}"
: "${RESPONSE_CONFIG:?--response-config is required}"
: "${OUTPUT_DIR:?--output-dir is required}"
: "${TRANSIENT:?--transient is required}"
: "${NUM_WORKERS:?--workers is required}"

if [ "$NUM_WORKERS" -le 0 ]; then
  echo "NUM_WORKERS must be >= 1"
  exit 1
fi
if [ "$FLUSH_EVERY" -le 0 ]; then
  echo "FLUSH_EVERY must be >= 1"
  exit 1
fi

run_worker() {
  local worker_id="$1"
  cmd=(uv run python scripts/run_response.py worker
    --unperturbed-dir "$UNPERTURBED_DIR"
    --response-config "$RESPONSE_CONFIG"
    --output-dir "$OUTPUT_DIR"
    --transient "$TRANSIENT"
    --worker-id "$worker_id"
    --num-workers "$NUM_WORKERS"
    --flush-every "$FLUSH_EVERY"
  )
  if [ -n "$BASE_SEED" ]; then
    cmd+=(--base-seed "$BASE_SEED")
  fi
  "${cmd[@]}"
}

pids=()
for worker_id in $(seq 0 $((NUM_WORKERS - 1))); do
  run_worker "$worker_id" &
  pids+=("$!")
done

cleanup() {
  echo "Stopping workers..."
  kill 0 2>/dev/null || true
}
trap cleanup EXIT INT TERM

fail=0
for pid in "${pids[@]}"; do
  if wait "$pid"; then
    echo "Worker pid=$pid finished ok"
  else
    rc=$?
    echo "Worker pid=$pid failed (exit=$rc)"
    fail=1
  fi
done

if [ "$fail" -ne 0 ]; then
  echo "One or more workers failed; skipping aggregation."
  exit 1
fi

uv run python scripts/run_response.py aggregate \
  --output-dir "$OUTPUT_DIR"
