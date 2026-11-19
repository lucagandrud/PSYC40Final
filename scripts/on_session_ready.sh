#!/usr/bin/env bash
# Runs on each newly verified session (called by qc_watcher.sh).
# Steps are added as the pipeline grows; each one is skipped until its script exists.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
S="$1"
cd "$ROOT"
# 1a. inventory of every variable in every file
"$PY" -I scripts/inventory.py "data/raw/$S" > /dev/null
# 1b/1c. load, preprocess, validate (appends reports/qc_table.csv)
if [ -f scripts/process_session.py ]; then "$PY" -I scripts/process_session.py "$S"; fi
# 1d. regenerate the cross-session QC report
if [ -f scripts/build_qc_report.py ]; then "$PY" -I scripts/build_qc_report.py; fi
