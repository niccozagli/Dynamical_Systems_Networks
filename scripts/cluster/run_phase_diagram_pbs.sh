#!/bin/bash
#PBS -N run_phase_diagram
#PBS -q medium
#PBS -m be
#PBS -j oe

# --- NEW: request 1 node with max cores on medium ---
#PBS -l nodes=1:ppn=8
#PBS -l mem=8gb

set -euo pipefail
cd "${PBS_O_WORKDIR:?PBS_O_WORKDIR is not set}"

# ---- Live log (independent of PBS staging) ----
LOGDIR="$PBS_O_WORKDIR/trash"
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/${PBS_JOBNAME}.${PBS_JOBID}.log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "=== START $(date) ==="
echo "Host: $(hostname)"
echo "JobID: ${PBS_JOBID:-n/a}"
echo "Workdir: $PBS_O_WORKDIR"
echo

# Parameters passed via qsub -v
ARGS="${ARGS:-}"
: "${ARGS:?Set ARGS via qsub -v, e.g. ARGS=\"--config configs/foo.json --table params/sweep.tsv --output-dir results/out --graph-realizations 5 --noise-realizations 3 --flush-every 10 --base-seed 123\"}"
echo "ARGS: $ARGS"
echo

# Keep numerical libs single-threaded (we do process-level parallelism via WORKERS)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# Disable HDF5 locking on network FS otherwise cluster can hang
export HDF5_USE_FILE_LOCKING=FALSE

# Local disk for uv cache, temp, and venv
export UV_CACHE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/uv-cache.XXXXXX")"
export TMPDIR="$(mktemp -d "${TMPDIR:-/tmp}/uv-tmp.XXXXXX")"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"

export UV_PROJECT_ENVIRONMENT="${TMPDIR:-/tmp}/uv-venv-$USER-${PBS_JOBID}"
rm -rf "$UV_PROJECT_ENVIRONMENT"

cleanup() {
  echo
  echo "=== CLEANUP $(date) ==="
  rm -rf "$UV_CACHE_DIR" "$TMPDIR" "$UV_PROJECT_ENVIRONMENT"
}
trap cleanup EXIT

echo "[$(date)] starting uv sync"
uv sync --frozen --no-progress
echo "[$(date)] finished uv sync"
echo

# ---- Phase diagram runner ----
# Usage:
#   qsub -v ARGS="--config CONFIG --table TABLE --output-dir OUT \
#     --graph-realizations N --noise-realizations N --flush-every N [--base-seed N] [--workers N]" \
#     scripts/cluster/run_phase_diagram_pbs.sh
# Note: WORKERS defaults to allocated cores on one node.

CONFIG_PATH="configs/config_Kuramoto_ER.json"
TABLE_PATH="params/sweep_theta_Kuramoto_ER.tsv"
OUTPUT_DIR="results/sweep_theta_Kuramoto_ER_local"
GRAPH_REALIZATIONS="5"
NOISE_REALIZATIONS="3"
FLUSH_EVERY="10"
BASE_SEED=""
WORKERS="${PBS_NUM_PPN:-${PBS_NP:-8}}"

set -- $ARGS
while [ "$#" -gt 0 ]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --table)
      TABLE_PATH="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --graph-realizations)
      GRAPH_REALIZATIONS="$2"
      shift 2
      ;;
    --noise-realizations)
      NOISE_REALIZATIONS="$2"
      shift 2
      ;;
    --flush-every)
      FLUSH_EVERY="$2"
      shift 2
      ;;
    --base-seed)
      BASE_SEED="$2"
      shift 2
      ;;
    --workers)
      WORKERS="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 2
      ;;
  esac
done

echo "Allocated cores (PBS_NUM_PPN/PBS_NP) = ${PBS_NUM_PPN:-${PBS_NP:-unknown}}; using WORKERS=$WORKERS"
echo

# --- NEW: make output job-unique to avoid collisions across multiple PBS jobs ---
OUTPUT_DIR="${OUTPUT_DIR%/}/job_${PBS_JOBID}"
mkdir -p "$OUTPUT_DIR"
echo "Output root for this job: $OUTPUT_DIR"
echo

row_count=$("$UV_PROJECT_ENVIRONMENT/bin/python" - "$TABLE_PATH" <<'PY'
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

get_run_id() {
  "$UV_PROJECT_ENVIRONMENT/bin/python" - "$TABLE_PATH" "$1" <<'PY'
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
    cmd=("$UV_PROJECT_ENVIRONMENT/bin/python" -u -X faulthandler scripts/run_bulk_parallel.py worker
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
  "$UV_PROJECT_ENVIRONMENT/bin/python" -u -X faulthandler scripts/run_bulk_parallel.py aggregate \
    --output-dir "$OUTPUT_DIR" \
    --run-id "$run_id"

  end_ts=$(date +%s)
  echo "Finished row $row in $((end_ts - start_ts))s"
done

echo "=== END $(date) ==="
