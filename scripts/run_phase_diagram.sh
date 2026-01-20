#!/bin/bash
set -euo pipefail

# Usage:
#   ./scripts/run_phase_diagram.sh [CONFIG_PATH] [TABLE_PATH] [OUTPUT_DIR] \
#     [GRAPH_REALIZATIONS] [NOISE_REALIZATIONS] [WORKERS] [FLUSH_EVERY] [BASE_SEED]
#
# Example:
#   ./scripts/run_phase_diagram.sh \
#     configs/config_double_well_configuration_model_poisson.json \
#     params/sweep_theta_double_well_configuration_model_poisson.tsv \
#     results/phase_diagram_double_well_configuration_model_poisson \
#     50 20 4 10
#
# Notes:
# - One sweep row (parameter set) at a time; each row spawns WORKERS processes.
# - Each worker writes results to results/<run_id>/worker_<id>.h5.
# - After workers finish for a row, the script writes results/<run_id>/aggregate.h5.
# - If BASE_SEED is omitted, NumPy uses OS entropy for non-reproducible seeds.

CONFIG_PATH="${1:-configs/config_Kuramoto_ER.json}"
TABLE_PATH="${2:-params/sweep_theta_Kuramoto_ER.tsv}"
OUTPUT_DIR="${3:-results/sweep_theta_Kuramoto_ER_local}"
GRAPH_REALIZATIONS="${4:-5}"
NOISE_REALIZATIONS="${5:-3}"
WORKERS="${6:-1}"
FLUSH_EVERY="${7:-10}"
BASE_SEED="${8:-}"

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

get_run_id() {
  python - "$TABLE_PATH" "$1" <<'PY'
import csv
from pathlib import Path
import sys

path = Path(sys.argv[1])
row_index = int(sys.argv[2])
delim = "," if path.suffix.lower() == ".csv" else "\t"
with path.open("r", newline="") as fh:
    lines = [line for line in fh if line.strip() and not line.lstrip().startswith("#")]
    reader = csv.DictReader(lines, delimiter=delim)
    for idx, row in enumerate(reader, start=1):
        if idx == row_index:
            run_id = row.get("run_id")
            if run_id:
                print(run_id)
                sys.exit(0)
            break
print(f"row_{row_index}")
PY
}

for row in $(seq 1 "$row_count"); do
  run_id="$(get_run_id "$row")"
  echo "Row $row / $row_count -> run_id=$run_id"
  start_ts=$(date +%s)

  for worker_id in $(seq 0 $((WORKERS - 1))); do
    cmd=(uv run python scripts/run_bulk_parallel.py worker
      --config "$CONFIG_PATH"
      --params-table "$TABLE_PATH"
      --row-index "$row"
      --output-dir "$OUTPUT_DIR"
      --run-id "$run_id"
      --graph-realizations "$GRAPH_REALIZATIONS"
      --noise-realizations "$NOISE_REALIZATIONS"
      --num-workers "$WORKERS"
      --worker-id "$worker_id"
      --flush-every "$FLUSH_EVERY"
    )
    if [ -n "$BASE_SEED" ]; then
      cmd+=(--base-seed "$BASE_SEED")
    fi
    "${cmd[@]}" &
  done

  # Wait for all workers for this row.
  wait

  # Merge worker aggregates into a single file for this row.
  uv run python scripts/run_bulk_parallel.py aggregate \
    --output-dir "$OUTPUT_DIR" \
    --run-id "$run_id"

  end_ts=$(date +%s)
  echo "Finished row $row in $((end_ts - start_ts))s"
done
