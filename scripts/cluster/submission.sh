#!/bin/bash
#PBS -N check_scratch
#PBS -q standard
#PBS -l select=1:ncpus=1:mem=1gb
#PBS -l walltime=00:02:00
#PBS -o /tmp/check_scratch.out
#PBS -e /tmp/check_scratch.err

set -euo pipefail

echo "HOST=$(hostname)"
echo "TMPDIR=${TMPDIR:-}"
echo "PBS_JOBFS=${PBS_JOBFS:-}"
echo "SLURM_TMPDIR=${SLURM_TMPDIR:-}"
echo "PWD=$PWD"

# Quick write test (1GB) if TMPDIR is set
if [ -n "${TMPDIR:-}" ]; then
  echo "Writing 1GB to $TMPDIR..."
  dd if=/dev/zero of="$TMPDIR/testfile" bs=1M count=1024 status=progress
  rm -f "$TMPDIR/testfile"
fi

# Quick write test on shared scratch
SCRATCH=/scratchcomp02/$USER
mkdir -p "$SCRATCH"
echo "Writing 1GB to $SCRATCH..."
dd if=/dev/zero of="$SCRATCH/testfile" bs=1M count=1024 status=progress
rm -f "$SCRATCH/testfile"
