#!/bin/bash
#PBS -N build_response_table
#PBS -q short
#PBS -m be
#PBS -j oe
#PBS -l nodes=1:ppn=1
#PBS -l mem=2gb

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
#   qsub -e trash -o trash -v ARGS="--unperturbed-root results/lr_unperturbed/job_XXXX \
#   --output-dir params \
#   --transient 5000 \
#   --integrator-tmin 0 --integrator-tmax 200 --integrator-dt 0.01 \
#   --stats-every 10 --state-every 100" scripts/cluster/run_build_response_table_pbs.sh
ARGS="${ARGS:-}"
: "${ARGS:?Set ARGS via qsub -v, e.g. ARGS=\"--unperturbed-root results/... --output params/... --transient 5000\"}"
echo "ARGS: $ARGS"
echo

# Keep numerical libs single-threaded
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

read -r -a argv <<< "$ARGS"
set -- "${argv[@]}"

"$UV_PROJECT_ENVIRONMENT/bin/python" -u -X faulthandler scripts/build_response_table.py "$@"

echo "=== END $(date) ==="
