#!/bin/bash
set -euo pipefail

# Usage:
#   ./scripts/response/run_build_response_table.sh --unperturbed-root <dir> --output <path> --transient <t>
#   ./scripts/response/run_build_response_table.sh --unperturbed-root <dir> --output-dir <dir> --transient <t>
#
# Example:
#   ./scripts/response/run_build_response_table.sh \
#     --unperturbed-root results/lr_unperturbed/job_XXXX \
#     --output-dir params \
#     --transient 5000

uv run python scripts/build_response_table.py "$@"
