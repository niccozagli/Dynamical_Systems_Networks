#!/bin/bash
set -euo pipefail

# Usage:
#   ./scripts/run_linear_response_unperturbed_experiments.sh --config <path> --table <path> --output-dir <dir> --workers <n>
#
# Example:
#   ./scripts/run_linear_response_unperturbed_experiments.sh \
#     --config configs/linear_response/config_lr.json \
#     --table params/linear_response/lr_seeds.tsv \
#     --output-dir results/lr \
#     --workers 3

CONFIG_PATH=""
TABLE_PATH=""
OUTPUT_DIR=""
NUM_WORKERS=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --config)     CONFIG_PATH="$2"; shift 2;;
    --table)      TABLE_PATH="$2"; shift 2;;
    --output-dir) OUTPUT_DIR="$2"; shift 2;;
    --workers)    NUM_WORKERS="$2"; shift 2;;
    *) echo "Unknown argument: $1"; exit 2;;
  esac
done

: "${CONFIG_PATH:?--config is required}"
: "${TABLE_PATH:?--table is required}"
: "${OUTPUT_DIR:?--output-dir is required}"
: "${NUM_WORKERS:?--workers is required}"

if [ "$NUM_WORKERS" -le 0 ]; then
  echo "NUM_WORKERS must be >= 1"
  exit 1
fi

row_count=$(python - "$TABLE_PATH" <<'PY'
import csv, sys
from pathlib import Path
path = Path(sys.argv[1])
delim = "," if path.suffix.lower() == ".csv" else "\t"
with path.open("r", newline="") as fh:
    lines = [line for line in fh if line.strip() and not line.lstrip().startswith("#")]
    reader = csv.DictReader(lines, delimiter=delim)
    rows = list(reader)
print(len(rows))
PY
)

if [ "$row_count" -le 0 ]; then
  echo "No rows found in $TABLE_PATH"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

run_worker() {
  local worker_id="$1"
  for row in $(seq 1 "$row_count"); do
    if [ $(( (row - 1) % NUM_WORKERS )) -ne "$worker_id" ]; then
      continue
    fi
    echo "Worker $worker_id running row $row / $row_count"
    uv run python scripts/run_simulation.py \
      --config "$CONFIG_PATH" \
      --params-table "$TABLE_PATH" \
      --row-index "$row" \
      --output-dir "$OUTPUT_DIR"
  done
}

for worker_id in $(seq 0 $((NUM_WORKERS - 1))); do
  run_worker "$worker_id" &
done

wait
