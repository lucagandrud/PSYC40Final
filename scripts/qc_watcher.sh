#!/usr/bin/env bash
# Watches data/raw for verified sessions and runs scripts/on_session_ready.sh on
# each one whose QC is missing or older than scripts/QC_VERSION.
# Bumping QC_VERSION makes every session re-run automatically.
# Exits when every queued session is verified-or-failed and QC'd, and no
# downloader is still running.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="$ROOT/data/raw"; MAN="$ROOT/data/manifests"
LOG="$ROOT/data/qc.log"
log() { echo "$(date '+%F %T') [qc] $*" | tee -a "$LOG"; }

while true; do
  VER=$(cat "$ROOT/scripts/QC_VERSION" 2>/dev/null || echo 0)
  for d in "$RAW"/*/; do
    S=$(basename "$d")
    [ -f "$d/.verified" ] || continue
    [ "$(cat "$d/.qc_done" 2>/dev/null)" = "$VER" ] && continue
    [ "$(cat "$d/.qc_failed" 2>/dev/null)" = "$VER" ] && continue
    log "QC start $S (v$VER)"
    if bash "$ROOT/scripts/on_session_ready.sh" "$S" >> "$LOG" 2>&1; then
      echo "$VER" > "$d/.qc_done"; rm -f "$d/.qc_failed"; log "QC done $S"
    else
      echo "$VER" > "$d/.qc_failed"; log "QC ERROR $S (see data/qc.log)"
    fi
  done
  # stop once nothing is left to download and nothing is left to check
  pending=0
  for s in $(grep -v '^#' "$MAN/download_queue.txt" | awk 'NF {print $1}'); do
    if cut -f1 "$MAN/download_failures.txt" 2>/dev/null | grep -qx "$s"; then continue; fi
    if [ ! -f "$RAW/$s/.verified" ]; then pending=1; break; fi
    [ "$(cat "$RAW/$s/.qc_done" 2>/dev/null)" = "$VER" ] || [ -f "$RAW/$s/.qc_failed" ] || { pending=1; break; }
  done
  if [ "$pending" -eq 0 ] && ! pgrep -f "scripts/download_data.sh" >/dev/null; then
    log "all queued sessions processed; watcher exiting"; break
  fi
  sleep 20
done
