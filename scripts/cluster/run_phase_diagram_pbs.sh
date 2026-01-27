#!/bin/bash
#PBS -N run_phase_diagram
#PBS -q medium
#PBS -m be
#PBS -j oe

# Request 1 node with 8 cores on medium
#PBS -l nodes=1:ppn=8
#PBS -l mem=4gb

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

# Parameters can be passed either via qsub -v ARGS="..." or directly as CLI flags.
# Example (qsub):
#   qsub -e trash -o trash -v ARGS="--config configs/config_double_well_configuration_model_poisson.json \
#   --table params/sweep_theta_double_well_configuration_model_poisson.tsv \
#   --output-dir results/phase_diagram_double_well \
#   --graph-realizations 10 \
#   --noise-realizations 5 \
#   --flush-every 5" scripts/cluster/run_phase_diagram_pbs.sh
# Example (direct flags):
#   qsub -e trash -o trash scripts/cluster/run_phase_diagram_pbs.sh --config ... --table ... --output-dir ... --graph-realizations 10 --noise-realizations 5 --flush-every 5
ARGS="${ARGS:-}"
if [ "$#" -gt 0 ]; then
  argv=("$@")
  echo "CLI ARGS: ${argv[*]}"
elif [ -n "$ARGS" ]; then
  read -r -a argv <<< "$ARGS"
  echo "ENV ARGS: $ARGS"
else
  echo "No arguments provided. Pass flags directly or via ARGS=..."
  exit 2
fi
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
CONFIG_PATH="configs/config_Kuramoto_ER.json"
TABLE_PATH="params/sweep_theta_Kuramoto_ER.tsv"
OUTPUT_DIR="results/sweep_theta_Kuramoto_ER_local"
GRAPH_REALIZATIONS="5"
NOISE_REALIZATIONS="3"
FLUSH_EVERY="10"
BASE_SEED=""
WORKERS="${PBS_NUM_PPN:-${PBS_NP:-8}}"

set -- "${argv[@]}"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --config)            CONFIG_PATH="$2"; shift 2;;
    --table)             TABLE_PATH="$2"; shift 2;;
    --output-dir)        OUTPUT_DIR="$2"; shift 2;;
    --graph-realizations) GRAPH_REALIZATIONS="$2"; shift 2;;
    --noise-realizations) NOISE_REALIZATIONS="$2"; shift 2;;
    --flush-every)       FLUSH_EVERY="$2"; shift 2;;
    --base-seed)         BASE_SEED="$2"; shift 2;;
    --workers)           WORKERS="$2"; shift 2;;
    *) echo "Unknown argument: $1"; exit 2;;
  esac
done

ALLOC_CORES="${PBS_NUM_PPN:-${PBS_NP:-8}}"
if [ "$WORKERS" -gt "$ALLOC_CORES" ]; then
  echo "WARNING: WORKERS=$WORKERS > allocated cores=$ALLOC_CORES; capping to $ALLOC_CORES"
  WORKERS="$ALLOC_CORES"
fi

echo "Allocated cores (PBS_NUM_PPN/PBS_NP) = ${PBS_NUM_PPN:-${PBS_NP:-unknown}}; using WORKERS=$WORKERS"
echo

# Make output job-unique to avoid collisions across PBS jobs
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

  echo "[$(date)] launching $WORKERS workers for run_id=$run_id"
  pids=()

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

    pid=$!
    pids+=("$pid")
    echo "[$(date)] started worker_id=$worker_id pid=$pid"
  done

  echo "[$(date)] entering wait for run_id=$run_id"
  fail=0
  for pid in "${pids[@]}"; do
    if wait "$pid"; then
      echo "[$(date)] worker pid=$pid finished ok"
    else
      rc=$?
      echo "[$(date)] worker pid=$pid failed (exit=$rc)"
      fail=1
    fi
  done
  echo "[$(date)] all workers waited for run_id=$run_id (fail=$fail)"

  if [ "$fail" -ne 0 ]; then
    echo "One or more workers failed for run_id=$run_id; skipping aggregation."
    continue
  fi

  echo "[$(date)] starting aggregation for run_id=$run_id"
  "$UV_PROJECT_ENVIRONMENT/bin/python" -u -X faulthandler scripts/run_bulk_parallel.py aggregate \
    --output-dir "$OUTPUT_DIR" \
    --run-id "$run_id"
  echo "[$(date)] finished aggregation for run_id=$run_id"

  end_ts=$(date +%s)
  echo "Finished row $row in $((end_ts - start_ts))s"
done

echo "=== END $(date) ==="
