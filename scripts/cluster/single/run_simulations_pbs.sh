#!/bin/bash
#PBS -N run_simulations
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

# Parameters can be passed either via qsub -v ARGS="..." or directly as CLI flags.
# Example (qsub, multi-run):
#   qsub -e trash -o trash -v ARGS="--config configs/foo.json --output-dir results/out --num-graphs 10 --workers 8" \
#     scripts/cluster/single/run_simulations_pbs.sh
# Example (qsub, single-run):
#   qsub -e trash -o trash -v ARGS="--config configs/foo.json --output-dir results/out --run-id test" \
#     scripts/cluster/single/run_simulations_pbs.sh
# Example (direct flags):
#   qsub -e trash -o trash scripts/cluster/single/run_simulations_pbs.sh --config ... --output-dir ... --num-graphs 10 --workers 8
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

# ---- Runner ----
CONFIG_PATH=""
OUTPUT_DIR=""
NUM_GRAPHS=""
RUN_ID=""
WORKERS="${PBS_NUM_PPN:-${PBS_NP:-1}}"

set -- "${argv[@]}"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --config)     CONFIG_PATH="$2"; shift 2;;
    --output-dir) OUTPUT_DIR="$2"; shift 2;;
    --num-graphs) NUM_GRAPHS="$2"; shift 2;;
    --run-id)     RUN_ID="$2"; shift 2;;
    --workers)    WORKERS="$2"; shift 2;;
    *) echo "Unknown argument: $1"; exit 2;;
  esac
done

: "${CONFIG_PATH:?--config is required}"
: "${OUTPUT_DIR:?--output-dir is required}"

if [ -n "$RUN_ID" ] && [ -n "$NUM_GRAPHS" ]; then
  echo "Provide only one of --run-id or --num-graphs"
  exit 2
fi

if [ -z "$RUN_ID" ] && [ -z "$NUM_GRAPHS" ]; then
  echo "Provide either --run-id (single run) or --num-graphs (multiple runs)"
  exit 2
fi

if [ -n "$NUM_GRAPHS" ] && [ "$NUM_GRAPHS" -le 0 ]; then
  echo "NUM_GRAPHS must be >= 1"
  exit 1
fi

ALLOC_CORES="${PBS_NUM_PPN:-${PBS_NP:-1}}"
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

PYTHON_BIN="$UV_PROJECT_ENVIRONMENT/bin/python"

if [ -n "$RUN_ID" ]; then
  echo "[$(date)] launching single simulation run_id=$RUN_ID"
  "$PYTHON_BIN" -u -X faulthandler scripts/run_simulation.py \
    --config "$CONFIG_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --run-id "$RUN_ID"
  echo "[$(date)] single simulation finished"
  echo "=== END $(date) ==="
  exit 0
fi

run_worker() {
  local worker_id="$1"
  for idx in $(seq 1 "$NUM_GRAPHS"); do
    if [ $(( (idx - 1) % WORKERS )) -ne "$worker_id" ]; then
      continue
    fi
    run_id=$(printf "graph_%04d" "$idx")
    echo "Worker $worker_id running $run_id ($idx / $NUM_GRAPHS)"
    "$PYTHON_BIN" -u -X faulthandler scripts/run_simulation.py \
      --config "$CONFIG_PATH" \
      --output-dir "$OUTPUT_DIR" \
      --run-id "$run_id"
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
