#!/bin/bash
set -euo pipefail
cd "${PBS_O_WORKDIR:-$(pwd)}"

# Usage:
#   ./scripts/cluster/response/run_response_aggregate.sh --output-dir <dir>
#
# Example:
#   ./scripts/cluster/response/run_response_aggregate.sh \
#     --output-dir results/linear_response/poisson/perturbed_runs/constant/critical/n1000/graph_0001/eps001

export HDF5_USE_FILE_LOCKING=FALSE
uv run python scripts/run_response.py aggregate "$@"
