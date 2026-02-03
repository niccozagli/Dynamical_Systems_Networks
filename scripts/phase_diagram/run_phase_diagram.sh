#!/bin/bash
set -euo pipefail

# Usage:
#   ./scripts/phase_diagram/run_phase_diagram.sh [CONFIG_PATH] [TABLE_PATH] [OUTPUT_DIR] \
#     [GRAPH_REALIZATIONS] [NOISE_REALIZATIONS] [WORKERS] [FLUSH_EVERY] [BASE_SEED]
#
# Example:
#   ./scripts/phase_diagram/run_phase_diagram.sh \
#     configs/config_double_well_configuration_model_poisson.json \
#     params/sweep_theta_double_well_configuration_model_poisson.tsv \
#     results/phase_diagram_double_well_configuration_model_poisson \
#     50 20 4 10
#
# Notes:
# - One sweep row (parameter set) at a time; each row spawns WORKERS processes.
# - Each worker writes results to results/row_<index>/worker_<id>.h5.
# - After workers finish for a row, the script writes results/row_<index>/aggregate.h5.
# - If BASE_SEED is omitted, NumPy uses OS entropy for non-reproducible seeds.

CONFIG_PATH="${1:-configs/config_Kuramoto_ER.json}"
TABLE_PATH="${2:-params/sweep_theta_Kuramoto_ER.tsv}"
OUTPUT_DIR="${3:-results/sweep_theta_Kuramoto_ER_local}"
GRAPH_REALIZATIONS="${4:-5}"
NOISE_REALIZATIONS="${5:-3}"
WORKERS="${6:-1}"
FLUSH_EVERY="${7:-10}"
BASE_SEED="${8:-}"

UTILS="scripts/phase_diagram/phase_diagram_utils.py"
row_count=$(python "$UTILS" count-rows --table "$TABLE_PATH")

if [ "$row_count" -le 0 ]; then
  echo "No rows found in $TABLE_PATH"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

get_run_id() {
  python "$UTILS" row-id --row-index "$1"
}

write_config_used() {
  python "$UTILS" write-config-used \
    --config "$CONFIG_PATH" \
    --table "$TABLE_PATH" \
    --row-index "$1" \
    --output-dir "$OUTPUT_DIR" \
    --run-id "$2" \
    --graph-realizations "$GRAPH_REALIZATIONS" \
    --noise-realizations "$NOISE_REALIZATIONS"
}

for row in $(seq 1 "$row_count"); do
  run_id="$(get_run_id "$row")"
  echo "Row $row / $row_count -> row_id=$run_id"
  start_ts=$(date +%s)
  write_config_used "$row" "$run_id"

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
