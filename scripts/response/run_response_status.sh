#!/bin/bash
set -euo pipefail

# Usage:
#   ./scripts/response/run_response_status.sh --output-dir <dir> [--per-worker]
#
# Example:
#   ./scripts/response/run_response_status.sh \
#     --output-dir results/linear_response/poisson/perturbed_runs/constant/critical/n1000/graph_0001/eps001

export HDF5_USE_FILE_LOCKING=FALSE
uv run python scripts/response_status.py "$@"
