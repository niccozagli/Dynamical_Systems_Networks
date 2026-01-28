#!/bin/bash
#PBS -N run_response
#PBS -q medium
#PBS -m be
#PBS -j oe
#PBS -l nodes=1:ppn=8
#PBS -l mem=6gb

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
#   qsub -e trash -o trash -v ARGS="--config configs/config_lr.json \
#   --table params/response_samples.tsv \
#   --output-dir results/response \
#   --workers 8 \
#   --flush-every 10" scripts/cluster/run_response_experiments_pbs.sh
# Example (direct flags):
#   qsub -e trash -o trash scripts/cluster/run_response_experiments_pbs.sh --config ... --table ... --output-dir ... --workers 8
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

# ---- Response runner ----
CONFIG_PATH="configs/config_lr.json"
TABLE_PATH="params/response_samples.tsv"
OUTPUT_DIR="results/response"
FLUSH_EVERY="10"
WORKERS="${PBS_NUM_PPN:-${PBS_NP:-8}}"

set -- "${argv[@]}"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --config)     CONFIG_PATH="$2"; shift 2;;
    --table)      TABLE_PATH="$2"; shift 2;;
    --output-dir) OUTPUT_DIR="$2"; shift 2;;
    --workers)    WORKERS="$2"; shift 2;;
    --flush-every) FLUSH_EVERY="$2"; shift 2;;
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

run_worker() {
  local worker_id="$1"
  "$UV_PROJECT_ENVIRONMENT/bin/python" -u -X faulthandler scripts/run_response_parallel.py worker \
    --config "$CONFIG_PATH" \
    --table "$TABLE_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --worker-id "$worker_id" \
    --num-workers "$WORKERS" \
    --flush-every "$FLUSH_EVERY"
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
if [ "$fail" -ne 0 ]; then
  exit 1
fi

"$UV_PROJECT_ENVIRONMENT/bin/python" -u -X faulthandler scripts/run_response_parallel.py aggregate \
  --output-dir "$OUTPUT_DIR"

echo "=== END $(date) ==="
