#!/bin/bash
set -euo pipefail

# Usage:
#   ./scripts/run_claim_response_chunk.sh --table <response_samples.tsv> --output-dir <dir> --chunk-size <N>
#
# Example:
#   ./scripts/run_claim_response_chunk.sh \
#     --table results/linear_response/poisson_n1000/response_samples.tsv \
#     --output-dir results/linear_response/poisson_n1000/chunks \
#     --chunk-size 50000 --randomize

uv run python scripts/claim_response_chunk.py "$@"
