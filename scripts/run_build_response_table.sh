#!/bin/bash
set -euo pipefail

# Usage:
#   ./scripts/run_build_response_table.sh --unperturbed-root <dir> --output <path> --transient <t> [overrides]
#   ./scripts/run_build_response_table.sh --unperturbed-root <dir> --output-dir <dir> --transient <t> [overrides]
#
# Example:
#   ./scripts/run_build_response_table.sh \
#     --unperturbed-root results/lr_unperturbed/job_XXXX \
#     --output-dir params \
#     --transient 5000 \
#     --integrator-tmin 0 --integrator-tmax 200 --integrator-dt 0.01 \
#     --stats-every 10 --state-every 100

uv run python scripts/build_response_table.py "$@"
