#!/bin/bash
#PBS -N run_simulation
#PBS -q standard
#PBS -m be
#PBS -j oe

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
#   qsub -e trash -o trash -v ARGS="--config configs/foo.json --output-dir results/out --run-id test" scripts/cluster/run_simulation_pbs.sh
# Example (direct flags):
#   qsub -e trash -o trash scripts/cluster/run_simulation_pbs.sh --config configs/foo.json --output-dir results/out --run-id test
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

# Keep numerical libs single-threaded (standard queue = 1 core)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

#  Disable HDF5 locking on network FS otherwise cluster hangs
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

echo "[$(date)] launching simulation"
# -u: unbuffered stdout/stderr (so logs appear immediately)
# -X faulthandler: enables faulthandler (add dump_traceback_later in Python for periodic traces)
"$UV_PROJECT_ENVIRONMENT/bin/python" -u -X faulthandler scripts/run_simulation.py "${argv[@]}"
echo "[$(date)] simulation finished"

echo "=== END $(date) ==="
