#!/bin/bash
set -euo pipefail

# Usage:
#   ./scripts/single/run_simulations.sh \
#     --config <path> --output-dir <dir> --num-graphs <n> [--workers <n>]
#
# Example:
#   ./scripts/single/run_simulations.sh \
#     --config configs/linear_response/config_lr.json \
#     --output-dir results/lr \
#     --num-graphs 10 \
#     --workers 3

CONFIG_PATH=""
OUTPUT_DIR=""
NUM_GRAPHS=""
NUM_WORKERS="1"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --config)     CONFIG_PATH="$2"; shift 2;;
    --output-dir) OUTPUT_DIR="$2"; shift 2;;
    --num-graphs) NUM_GRAPHS="$2"; shift 2;;
    --workers)    NUM_WORKERS="$2"; shift 2;;
    *) echo "Unknown argument: $1"; exit 2;;
  esac
done

: "${CONFIG_PATH:?--config is required}"
: "${OUTPUT_DIR:?--output-dir is required}"
: "${NUM_GRAPHS:?--num-graphs is required}"

if [ "$NUM_WORKERS" -le 0 ]; then
  echo "NUM_WORKERS must be >= 1"
  exit 1
fi

if [ "$NUM_GRAPHS" -le 0 ]; then
  echo "NUM_GRAPHS must be >= 1"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

run_worker() {
  local worker_id="$1"
  for idx in $(seq 1 "$NUM_GRAPHS"); do
    if [ $(( (idx - 1) % NUM_WORKERS )) -ne "$worker_id" ]; then
      continue
    fi
    run_id=$(printf "graph_%04d" "$idx")
    echo "Worker $worker_id running $run_id ($idx / $NUM_GRAPHS)"
    uv run python scripts/run_simulation.py \
      --config "$CONFIG_PATH" \
      --output-dir "$OUTPUT_DIR" \
      --run-id "$run_id"
  done
}

for worker_id in $(seq 0 $((NUM_WORKERS - 1))); do
  run_worker "$worker_id" &
done

wait
