#!/bin/bash
set -euo pipefail
cd "${PBS_O_WORKDIR:-$(pwd)}"

# Thin wrapper for login-node usage (kept here for consistency with cluster scripts).
# Usage matches scripts/run_claim_response_chunk.sh.

uv run python scripts/claim_response_chunk.py "$@"
