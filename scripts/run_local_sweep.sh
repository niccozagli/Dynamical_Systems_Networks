#!/bin/bash
set -euo pipefail

CONFIG_PATH="${1:-configs/config_Kuramoto_ER.json}"
TABLE_PATH="${2:-params/sweep_theta_Kuramoto_ER.tsv}"
OUTPUT_DIR="${3:-results/sweep_theta_Kuramoto_ER_local}"
REPS="${4:-5}"
JOBS="${5:-1}"

row_count=$(python - "$TABLE_PATH" <<'PY'
import csv
from pathlib import Path
import sys

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

total_runs=$((row_count * REPS))
run_index=0

if [ "$JOBS" -le 1 ]; then
  for row in $(seq 1 "$row_count"); do
    for rep in $(seq 1 "$REPS"); do
      run_index=$((run_index + 1))
      echo "Run $run_index / $total_runs (row $row, rep $rep)"
      start_ts=$(date +%s)
      run_id="row_${row}_rep_${rep}"
      uv run python scripts/run_simulation.py \
        --config "$CONFIG_PATH" \
        --params-table "$TABLE_PATH" \
        --row-index "$row" \
        --output-dir "$OUTPUT_DIR" \
        --run-id "$run_id"
      end_ts=$(date +%s)
      echo "Finished run $run_index in $((end_ts - start_ts))s"
    done
  done
else
  export CONFIG_PATH TABLE_PATH OUTPUT_DIR
  export TOTAL_RUNS="$total_runs"
  tasks_file=$(mktemp)
  idx=0
  for row in $(seq 1 "$row_count"); do
    for rep in $(seq 1 "$REPS"); do
      idx=$((idx + 1))
      echo "$idx $row $rep" >> "$tasks_file"
    done
  done
  xargs -n3 -P "$JOBS" bash -c '
    idx="$1"
    row="$2"
    rep="$3"
    echo "Run ${idx} / ${TOTAL_RUNS} (row ${row}, rep ${rep})"
    start_ts=$(date +%s)
    run_id="row_${row}_rep_${rep}"
    uv run python scripts/run_simulation.py \
      --config "$CONFIG_PATH" \
      --params-table "$TABLE_PATH" \
      --row-index "$row" \
      --output-dir "$OUTPUT_DIR" \
      --run-id "$run_id"
    end_ts=$(date +%s)
    echo "Finished run ${idx} / ${TOTAL_RUNS} (row ${row}, rep ${rep}) in $((end_ts - start_ts))s"
  ' _ < "$tasks_file"
  rm -f "$tasks_file"
fi
