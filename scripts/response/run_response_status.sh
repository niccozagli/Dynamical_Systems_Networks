#!/bin/bash
set -euo pipefail

# Usage:
#   ./scripts/response/run_response_status.sh --output-dir <dir> [--per-worker]
#
# Example:
#   ./scripts/response/run_response_status.sh \
#     --output-dir results/linear_response/poisson_annealed/perturbed_runs/degree_weighted/critical/n1000/graph_0001/alpha_rot_gamma1p0_eps01

export HDF5_USE_FILE_LOCKING=FALSE
uv run python scripts/response_status.py "$@"
