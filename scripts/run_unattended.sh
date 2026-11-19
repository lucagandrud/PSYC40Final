#!/usr/bin/env bash
# Start downloads (3 workers) + automatic QC, detached from the terminal and
# protected from idle sleep. Safe to re-run: finished sessions are skipped.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
nohup caffeinate -i bash -c '
  for i in 1 2 3; do bash scripts/download_data.sh & sleep 2; done
  bash scripts/qc_watcher.sh &
  wait
' >> data/runner.log 2>&1 &
echo $! > data/runner.pid
echo "started runner (pid $(cat data/runner.pid)); progress: bash scripts/status.sh"
