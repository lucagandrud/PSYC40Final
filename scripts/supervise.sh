#!/usr/bin/env bash
# Keeps the download + QC pipeline going unattended. Whenever the downloaders and the QC
# watcher have both stopped but queued sessions are still not downloaded (e.g. after
# network resets), it clears the failure list and relaunches run_unattended.sh.
# Gives up after 3 relaunches. Log: data/supervisor.log
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOG="$ROOT/data/supervisor.log"
log() { echo "$(date '+%F %T') [supervisor] $*" >> "$LOG"; }
log "started"
for round in 1 2 3; do
  while pgrep -f "scripts/download_data.sh" >/dev/null || pgrep -f "scripts/qc_watcher.sh" >/dev/null; do
    sleep 30
  done
  pending=""
  for s in $(grep -v '^#' data/manifests/download_queue.txt | awk 'NF {print $1}'); do
    [ -f "data/raw/$s/.verified" ] || pending="$pending $s"
  done
  if [ -z "$pending" ]; then log "all queued sessions downloaded; exiting"; exit 0; fi
  log "relaunch $round for:$pending"
  : > data/manifests/download_failures.txt
  bash scripts/run_unattended.sh >> "$LOG" 2>&1
  sleep 60
done
log "gave up after 3 relaunches"
