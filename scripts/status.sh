#!/usr/bin/env bash
# One-screen progress: queue status per session, in-flight downloads, recent problems.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="$ROOT/data/raw"; MAN="$ROOT/data/manifests"
printf "%-20s %-12s %s\n" SESSION DOWNLOAD QC
for s in $(grep -v '^#' "$MAN/download_queue.txt" | awk 'NF {print $1}'); do
  if [ -f "$RAW/$s/.verified" ]; then d=done
  elif [ -f "$RAW/$s.tar.gz" ]; then d="$(( $(stat -f %z "$RAW/$s.tar.gz") / 1048576 )) MB"
  elif cut -f1 "$MAN/download_failures.txt" 2>/dev/null | grep -qx "$s"; then d=FAILED
  else d=queued; fi
  if [ -f "$RAW/$s/.qc_done" ]; then q=done; elif [ -f "$RAW/$s/.qc_failed" ]; then q=ERROR; else q=-; fi
  printf "%-20s %-12s %s\n" "$s" "$d" "$q"
done
echo; pgrep -f "scripts/download_data.sh" >/dev/null && echo "downloader: running" || echo "downloader: stopped"
pgrep -f "scripts/qc_watcher.sh" >/dev/null && echo "qc watcher: running" || echo "qc watcher: stopped"
grep -hE "FLAG|ERROR|MISMATCH" "$ROOT/data/download.log" "$ROOT/data/qc.log" 2>/dev/null | tail -3
