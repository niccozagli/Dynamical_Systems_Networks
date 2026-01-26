#!/bin/bash
#PBS -N run_lr_unperturbed
#PBS -q medium
#PBS -m be
#PBS -j oe
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

# Parameters passed via qsub -v
# Example:
#   qsub -e trash -o trash -v ARGS="--config configs/config_lr.json \
#   --table params/lr_seeds.tsv \
#   --output-dir results/lr_unperturbed \
#   --workers 8" scripts/cluster/run_linear_response_unperturbed_pbs.sh
ARGS="${ARGS:-}"
: "${ARGS:?Set ARGS via qsub -v, e.g. ARGS=\"--config configs/foo.json --table params/seeds.tsv --output-dir results/out --workers 8\"}"
echo "ARGS: $ARGS"
echo

# Keep numerical libs single-threaded (process-level parallelism)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# Disable HDF5 locking on network FS
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

# ---- Unperturbed runner ----
CONFIG_PATH="configs/config_lr.json"
TABLE_PATH="params/lr_seeds.tsv"
OUTPUT_DIR="results/lr_unperturbed"
WORKERS="${PBS_NUM_PPN:-${PBS_NP:-8}}"

read -r -a argv <<< "$ARGS"
set -- "${argv[@]}"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --config)     CONFIG_PATH="$2"; shift 2;;
    --table)      TABLE_PATH="$2"; shift 2;;
    --output-dir) OUTPUT_DIR="$2"; shift 2;;
    --workers)    WORKERS="$2"; shift 2;;
    *) echo "Unknown argument: $1"; exit 2;;
  esac
done

ALLOC_CORES="${PBS_NUM_PPN:-${PBS_NP:-8}}"
if [ "$WORKERS" -gt "$ALLOC_CORES" ]; then
  echo "WARNING: WORKERS=$WORKERS > allocated cores=$ALLOC_CORES; capping to $ALLOC_CORES"
  WORKERS="$ALLOC_CORES"
fi

echo "Allocated cores = ${ALLOC_CORES}; using WORKERS=$WORKERS"
echo

# Make output job-unique to avoid collisions across PBS jobs
OUTPUT_DIR="${OUTPUT_DIR%/}/job_${PBS_JOBID}"
mkdir -p "$OUTPUT_DIR"
echo "Output root for this job: $OUTPUT_DIR"
echo

row_count=$("$UV_PROJECT_ENVIRONMENT/bin/python" - "$TABLE_PATH" <<'PY'
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

run_worker() {
  local worker_id="$1"
  for row in $(seq 1 "$row_count"); do
    if [ $(( (row - 1) % WORKERS )) -ne "$worker_id" ]; then
      continue
    fi
    echo "Worker $worker_id running row $row / $row_count"
    "$UV_PROJECT_ENVIRONMENT/bin/python" -u -X faulthandler scripts/run_simulation.py \
      --config "$CONFIG_PATH" \
      --params-table "$TABLE_PATH" \
      --row-index "$row" \
      --output-dir "$OUTPUT_DIR"
  done
}

pids=()
for worker_id in $(seq 0 $((WORKERS - 1))); do
  run_worker "$worker_id" &
  pids+=("$!")
done

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

echo "Workers done (fail=$fail)"
echo "=== END $(date) ==="
