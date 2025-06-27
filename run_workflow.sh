#!/usr/bin/env bash
#
# run_workflow.sh — Dispatch the "run-runner.yml" workflow via GitHub CLI
#
# Usage:
#   ./run_workflow.sh <number_of_circuits> <batch_size> <time_to_run> <mid_site> <Lx> <Ly>
#
# Example:
#   ./run_workflow.sh 100 10 1000 4 3 3

set -euo pipefail

if ! command -v gh &> /dev/null; then
  echo "Error: GitHub CLI (gh) not found. Please install from https://cli.github.com/" >&2
  exit 1
fi

if [[ $# -ne 6 ]]; then
  echo "Usage: $0 <number_of_circuits> <batch_size> <time_to_run> <mid_site> <Lx> <Ly>" >&2
  exit 1
fi

NUMBER_OF_CIRCUITS=$1
BATCH_SIZE=$2
TIME_TO_RUN=$10
MID_SITE=$2
LX=$2
LY=$2

WORKFLOW_FILE="run-runner.yml"   # name under .github/workflows/

echo "Dispatching GitHub Actions workflow '$WORKFLOW_FILE' with inputs:"
echo "  number_of_circuits = $NUMBER_OF_CIRCUITS"
echo "  batch_size         = $BATCH_SIZE"
echo "  time_to_run        = $TIME_TO_RUN"
echo "  mid_site           = $MID_SITE"
echo "  Lx                 = $LX"
echo "  Ly                 = $LY"
echo

gh workflow run "$WORKFLOW_FILE" \
  --field number_of_circuits="$NUMBER_OF_CIRCUITS" \
  --field batch_size="$BATCH_SIZE" \
  --field time_to_run="$TIME_TO_RUN" \
  --field mid_site="$MID_SITE" \
  --field Lx="$LX" \
  --field Ly="$LY"

echo
echo "✅ Workflow dispatched. To watch its progress, run: gh run watch"
