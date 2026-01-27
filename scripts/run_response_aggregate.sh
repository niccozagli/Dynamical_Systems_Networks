#!/bin/bash
set -euo pipefail

# Usage:
#   ./scripts/run_response_aggregate.sh --output-dir <dir> [--pattern <glob>]
#
# Example:
#   ./scripts/run_response_aggregate.sh \
#     --output-dir results/linear_response/poisson/critical/n1000/perturbed_runs/job_12345
#
# Note:
#   --output-dir should point to the job/run folder that contains worker_*.h5.

uv run python scripts/run_response_parallel.py aggregate "$@"
