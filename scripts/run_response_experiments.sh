#!/bin/bash
set -euo pipefail

# Usage:
#   ./scripts/run_response_experiments.sh CONFIG_PATH TABLE_PATH OUTPUT_DIR NUM_WORKERS [FLUSH_EVERY]
#
# Example:
#   ./scripts/run_response_experiments.sh configs/config_lr.json results/response_samples.tsv results/response 4 20

CONFIG_PATH="${1:?config path required}"
TABLE_PATH="${2:?table path required}"
OUTPUT_DIR="${3:?output dir required}"
NUM_WORKERS="${4:?num workers required}"
FLUSH_EVERY="${5:-10}"

if [ "$NUM_WORKERS" -le 0 ]; then
  echo "NUM_WORKERS must be >= 1"
  exit 1
fi
if [ "$FLUSH_EVERY" -le 0 ]; then
  echo "FLUSH_EVERY must be >= 1"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

run_worker() {
  local worker_id="$1"
  uv run python scripts/run_response_parallel.py worker \
    --config "$CONFIG_PATH" \
    --table "$TABLE_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --worker-id "$worker_id" \
    --num-workers "$NUM_WORKERS" \
    --flush-every "$FLUSH_EVERY"
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

uv run python scripts/run_response_parallel.py aggregate \
  --output-dir "$OUTPUT_DIR"
