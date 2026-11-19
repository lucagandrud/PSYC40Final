#!/usr/bin/env bash
# Run training jobs from a job file, N at a time, each with 2 CPU threads.
# Job file: one job per line = arguments to `python -m src.train` (comments with # allowed).
# Usage: bash scripts/run_jobs.sh <jobfile> [parallel=4]
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
JOBS="$1"; PAR="${2:-4}"
LOG="$ROOT/results/jobs.log"
mkdir -p results
grep -v '^\s*#' "$JOBS" | grep -v '^\s*$' | \
  xargs -P "$PAR" -I{} bash -c 'echo "$(date "+%F %T") start: {}" >> '"$LOG"'; .venv/bin/python -m src.train {} --threads 2 >> '"$LOG"' 2>&1; echo "$(date "+%F %T") end: {}" >> '"$LOG"
echo "$(date '+%F %T') all jobs in $JOBS finished" >> "$LOG"
