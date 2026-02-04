#!/bin/bash
set -euo pipefail

# Usage:
#   ./scripts/response/run_response_aggregate.sh --output-dir <dir>
#
# Example:
#   ./scripts/response/run_response_aggregate.sh \
#     --output-dir results/linear_response/poisson/critical/n1000/perturbed_runs/graph_0001

export HDF5_USE_FILE_LOCKING=FALSE
uv run python scripts/run_response.py aggregate "$@"
