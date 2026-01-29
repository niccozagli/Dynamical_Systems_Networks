#!/bin/bash
set -euo pipefail

# Usage:
#   ./scripts/run_merge_response_aggregates.sh --inputs <agg1> <agg2> ... --output <merged>
#
# Example:
#   ./scripts/run_merge_response_aggregates.sh \
#     --inputs results/.../job_1/aggregate.h5 results/.../job_2/aggregate.h5 \
#     --output results/.../merged/aggregate.h5

uv run python scripts/merge_response_aggregates.py "$@"
