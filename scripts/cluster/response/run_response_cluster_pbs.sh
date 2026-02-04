#!/bin/bash
#PBS -N run_response_parallel
#PBS -q medium
#PBS -m be
#PBS -j oe
#PBS -l nodes=1:ppn=8
#PBS -l mem=5gb

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
#   qsub -e trash -o trash -v ARGS="--unperturbed-dir ... --response-config ... --output-dir ... \
#     --transient 5000 --workers 8 --job-id 0 --num-jobs 4 --flush-every 50" \
#     scripts/cluster/response/run_response_cluster_pbs.sh
# Example (direct flags):
#   qsub -e trash -o trash scripts/cluster/response/run_response_cluster_pbs.sh --unperturbed-dir ... --response-config ... \
#     --output-dir ... --transient 5000 --workers 8 --job-id 0 --num-jobs 4 --flush-every 50
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

UNPERTURBED_DIR=""
RESPONSE_CONFIG=""
OUTPUT_DIR=""
TRANSIENT=""
WORKERS="${PBS_NUM_PPN:-${PBS_NP:-8}}"
JOB_ID="0"
NUM_JOBS="1"
FLUSH_EVERY="50"
BASE_SEED=""

set -- "${argv[@]}"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --unperturbed-dir) UNPERTURBED_DIR="$2"; shift 2;;
    --response-config) RESPONSE_CONFIG="$2"; shift 2;;
    --output-dir) OUTPUT_DIR="$2"; shift 2;;
    --transient) TRANSIENT="$2"; shift 2;;
    --workers) WORKERS="$2"; shift 2;;
    --job-id) JOB_ID="$2"; shift 2;;
    --num-jobs) NUM_JOBS="$2"; shift 2;;
    --flush-every) FLUSH_EVERY="$2"; shift 2;;
    --base-seed) BASE_SEED="$2"; shift 2;;
    *) echo "Unknown argument: $1"; exit 2;;
  esac
done

: "${UNPERTURBED_DIR:?--unperturbed-dir is required}"
: "${RESPONSE_CONFIG:?--response-config is required}"
: "${OUTPUT_DIR:?--output-dir is required}"
: "${TRANSIENT:?--transient is required}"

if [ "$WORKERS" -le 0 ]; then
  echo "WORKERS must be >= 1"
  exit 1
fi
if [ "$FLUSH_EVERY" -le 0 ]; then
  echo "FLUSH_EVERY must be >= 1"
  exit 1
fi
if [ "$NUM_JOBS" -le 0 ]; then
  echo "NUM_JOBS must be >= 1"
  exit 1
fi
if [ "$JOB_ID" -lt 0 ] || [ "$JOB_ID" -ge "$NUM_JOBS" ]; then
  echo "JOB_ID must be in [0, NUM_JOBS)"
  exit 1
fi

ALLOC_CORES="${PBS_NUM_PPN:-${PBS_NP:-8}}"
if [ "$WORKERS" -gt "$ALLOC_CORES" ]; then
  echo "WARNING: WORKERS=$WORKERS > allocated cores=$ALLOC_CORES; capping to $ALLOC_CORES"
  WORKERS="$ALLOC_CORES"
fi

echo "Allocated cores = ${ALLOC_CORES}; using WORKERS=$WORKERS"
echo "Job partition: job_id=$JOB_ID / num_jobs=$NUM_JOBS"
echo

PYTHON_BIN="$UV_PROJECT_ENVIRONMENT/bin/python"

run_worker() {
  local worker_id="$1"
  cmd=("$PYTHON_BIN" -u -X faulthandler scripts/run_response.py worker
    --unperturbed-dir "$UNPERTURBED_DIR"
    --response-config "$RESPONSE_CONFIG"
    --output-dir "$OUTPUT_DIR"
    --transient "$TRANSIENT"
    --worker-id "$worker_id"
    --num-workers "$WORKERS"
    --job-id "$JOB_ID"
    --num-jobs "$NUM_JOBS"
    --flush-every "$FLUSH_EVERY"
  )
  if [ -n "$BASE_SEED" ]; then
    cmd+=(--base-seed "$BASE_SEED")
  fi
  "${cmd[@]}"
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

if [ "$fail" -ne 0 ]; then
  echo "One or more workers failed; skipping aggregation."
  exit 1
fi

# Aggregation is typically run once after all jobs finish. You can run it here
# if you are using a single job.
if [ "$NUM_JOBS" -eq 1 ]; then
  "$PYTHON_BIN" -u -X faulthandler scripts/run_response.py aggregate \
    --output-dir "$OUTPUT_DIR"
fi

echo "=== END $(date) ==="
