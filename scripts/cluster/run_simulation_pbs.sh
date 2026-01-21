#!/bin/bash
#PBS -N run_simulation
#PBS -q standard
#PBS -m be
#PBS -j oe

set -euo pipefail
cd "${PBS_O_WORKDIR:?PBS_O_WORKDIR is not set}"

# Parameters passed via qsub -v
ARGS="${ARGS:-}"
: "${ARGS:?Set ARGS via qsub -v, e.g. ARGS=\"--config params/foo.json --seed 123\"}"

# Keep numerical libs single-threaded (standard queue = 1 core)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# Local disk for uv cache, temp, and venv
export UV_CACHE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/uv-cache.XXXXXX")"
export TMPDIR="$(mktemp -d "${TMPDIR:-/tmp}/uv-tmp.XXXXXX")"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"

export UV_PROJECT_ENVIRONMENT="${TMPDIR:-/tmp}/uv-venv-$USER-${PBS_JOBID}"
rm -rf "$UV_PROJECT_ENVIRONMENT"

cleanup() {
  rm -rf "$UV_CACHE_DIR" "$TMPDIR" "$UV_PROJECT_ENVIRONMENT"
}
trap cleanup EXIT

# Deterministic install from lockfile
uv sync --frozen --no-progress

# Run your (hardcoded) simulation entrypoint
"$UV_PROJECT_ENVIRONMENT/bin/python" scripts/run_simulation.py $ARGS
