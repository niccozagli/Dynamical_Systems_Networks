#!/bin/bash
set -euo pipefail

# Usage:
#   ./scripts/cluster/response/submit_response_jobs.sh \
#     --network-name <name> \
#     --setting <critical|far> \
#     --n <N> \
#     --graph <NNNN> \
#     --perturbation <type> \
#     --epsilon-tag <tag> \
#     --num-jobs <J> \
#     --workers <W> \
#     --transient <t> \
#     --flush-every <n> \
#     --sample-dt <dt>
#
# Example:
#   ./scripts/cluster/response/submit_response_jobs.sh \
#     --network-name poisson \
#     --setting critical \
#     --n 1000 \
#     --graph 0001 \
#     --perturbation constant \
#     --epsilon-tag 001 \
#     --num-jobs 10 \
#     --workers 8 \
#     --transient 5000 \
#     --flush-every 10 \
#     --sample-dt 10

NETWORK_NAME=""
SETTING=""
N=""
GRAPH=""
PERTURBATION=""
EPS_TAG=""
NUM_JOBS=""
WORKERS=""
TRANSIENT=""
FLUSH_EVERY=""
SAMPLE_DT=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --network-name) NETWORK_NAME="$2"; shift 2;;
    --setting) SETTING="$2"; shift 2;;
    --n) N="$2"; shift 2;;
    --graph) GRAPH="$2"; shift 2;;
    --perturbation) PERTURBATION="$2"; shift 2;;
    --epsilon-tag) EPS_TAG="$2"; shift 2;;
    --num-jobs) NUM_JOBS="$2"; shift 2;;
    --workers) WORKERS="$2"; shift 2;;
    --transient) TRANSIENT="$2"; shift 2;;
    --flush-every) FLUSH_EVERY="$2"; shift 2;;
    --sample-dt) SAMPLE_DT="$2"; shift 2;;
    *) echo "Unknown argument: $1"; exit 2;;
  esac
done

: "${NETWORK_NAME:?--network-name is required}"
: "${SETTING:?--setting is required}"
: "${N:?--n is required}"
: "${GRAPH:?--graph is required}"
: "${PERTURBATION:?--perturbation is required}"
: "${EPS_TAG:?--epsilon-tag is required}"
: "${NUM_JOBS:?--num-jobs is required}"
: "${WORKERS:?--workers is required}"
: "${TRANSIENT:?--transient is required}"
: "${FLUSH_EVERY:?--flush-every is required}"

if [ "$NUM_JOBS" -le 0 ] || [ "$WORKERS" -le 0 ]; then
  echo "NUM_JOBS and WORKERS must be >= 1"
  exit 1
fi
if [ "$FLUSH_EVERY" -le 0 ]; then
  echo "FLUSH_EVERY must be >= 1"
  exit 1
fi

unperturbed_dir="results/linear_response/${NETWORK_NAME}/unperturbed_runs/${SETTING}/n${N}/graph_${GRAPH}"
response_config="configs/linear_response/${NETWORK_NAME}/perturbed_runs/${SETTING}/response_config_${PERTURBATION}_eps${EPS_TAG}.json"
output_dir="results/linear_response/${NETWORK_NAME}/perturbed_runs/${PERTURBATION}/${SETTING}/n${N}/graph_${GRAPH}/eps${EPS_TAG}"

echo "Submitting response jobs:"
echo "  unperturbed_dir = ${unperturbed_dir}"
echo "  response_config = ${response_config}"
echo "  output_dir      = ${output_dir}"
echo "  num_jobs        = ${NUM_JOBS}"
echo "  workers         = ${WORKERS}"
echo "  transient       = ${TRANSIENT}"
echo "  flush_every     = ${FLUSH_EVERY}"
if [ -n "$SAMPLE_DT" ]; then
  echo "  sample_dt       = ${SAMPLE_DT}"
fi
echo

for job_id in $(seq 0 $((NUM_JOBS - 1))); do
  cmd=(qsub -e trash -o trash -v ARGS="--unperturbed-dir ${unperturbed_dir} \
--response-config ${response_config} \
--output-dir ${output_dir} \
--transient ${TRANSIENT} \
--workers ${WORKERS} \
--job-id ${job_id} --num-jobs ${NUM_JOBS} \
--flush-every ${FLUSH_EVERY}" \
    scripts/cluster/response/run_response_cluster_pbs.sh)
  if [ -n "$SAMPLE_DT" ]; then
    cmd=(qsub -e trash -o trash -v ARGS="--unperturbed-dir ${unperturbed_dir} \
--response-config ${response_config} \
--output-dir ${output_dir} \
--transient ${TRANSIENT} \
--workers ${WORKERS} \
--job-id ${job_id} --num-jobs ${NUM_JOBS} \
--flush-every ${FLUSH_EVERY} \
--sample-dt ${SAMPLE_DT}" \
      scripts/cluster/response/run_response_cluster_pbs.sh)
  fi
  job_output="$("${cmd[@]}")"
  echo "submitted job_id=${job_id} qsub_id=${job_output}"
done
