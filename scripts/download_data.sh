#!/usr/bin/env bash
# Queue-driven, idempotent download of CRCNS fcx-1 sessions into data/raw/<session>/.
#
# Usage: scripts/download_data.sh            (works through data/manifests/download_queue.txt)
#        scripts/download_data.sh S1 S2 ...  (explicit sessions)
# Several workers can run at once; a per-session lock stops them colliding.
# The queue file is re-read before each pick, so it can be edited while running.
#
# Per session: free-space check (1.5x tarball) -> download -> MD5 vs checksums.md5
# (one retry) -> extract without *.eeg / *ClusterQualityMeasures* -> compare files
# with 0_data_file_list.txt -> delete tarball -> write .verified marker.
# QC runs separately (scripts/qc_watcher.sh) on every verified session.
# Credentials: env vars or ~/.crcns_credentials (see crcns_fetch.sh).
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MAN="$ROOT/data/manifests"
RAW="$ROOT/data/raw"
LOG="$ROOT/data/download.log"
QUEUE="$MAN/download_queue.txt"
[ -f "$QUEUE" ] || cp "$ROOT/scripts/download_queue.txt" "$QUEUE"   # repo copy of the queue
FETCH="$ROOT/scripts/crcns_fetch.sh"
MISSING="$MAN/missing_files.csv"
FAILED="$MAN/download_failures.txt"
mkdir -p "$RAW"; touch "$FAILED"
[ -f "$MISSING" ] || echo "session,file,expected_bytes,actual_bytes" > "$MISSING"

log() { echo "$(date '+%F %T') [$$] $*" | tee -a "$LOG"; }

CUR_LOCK=""
release() { [ -n "$CUR_LOCK" ] && rm -rf "$CUR_LOCK"; CUR_LOCK=""; }
trap release EXIT INT TERM

tar_size()     { awk -v p="data/$1.tar.gz" '$1==p {print $2}' "$MAN/filelist.txt"; }
expected_md5() { awk -v p="data/$1.tar.gz" '$2==p {print $1}' "$MAN/checksums.md5"; }
flag()         { log "FLAG $1: $2"; echo "$1	$2" >> "$FAILED"; release; }

candidates() {
  if [ "$#" -gt 0 ]; then printf '%s\n' "$@"; else grep -v '^#' "$QUEUE" | awk 'NF {print $1}'; fi
}

# Pick and lock the next session that is not verified, failed, or busy.
next_session() {
  local s
  for s in $(candidates "$@"); do
    [ -f "$RAW/$s/.verified" ] && continue
    cut -f1 "$FAILED" | grep -qx "$s" && continue
    # a lock from the older worker generation (without RESERVED) means it is downloading
    if [ -d "$RAW/.lock_$s" ] && [ ! -f "$RAW/.lock_$s/RESERVED" ]; then continue; fi
    if mkdir "$RAW/.qlock_$s" 2>/dev/null; then echo "$s"; return 0; fi
  done
  return 1
}

check_files() {
  local S="$1" n_missing=0 n_ok=0 rel size f actual
  while read -r rel size; do
    f="$RAW/$rel"
    if [ ! -f "$f" ]; then
      echo "$S,$rel,$size,MISSING" >> "$MISSING"; n_missing=$((n_missing+1))
    else
      actual=$(stat -f %z "$f" 2>/dev/null || stat -c %s "$f")
      if [ "$actual" != "$size" ]; then echo "$S,$rel,$size,$actual" >> "$MISSING"; n_missing=$((n_missing+1))
      else n_ok=$((n_ok+1)); fi
    fi
  done < <(sed -n "s/^\`\.\/\(${S}\/[^']*\)' \([0-9]*\)$/\1 \2/p" "$MAN/0_data_file_list.txt" \
             | grep -v -e '\.eeg ' -e 'ClusterQualityMeasures')
  log "$S file check: $n_ok ok, $n_missing missing or wrong size"
}

while S=$(next_session "$@"); do
  CUR_LOCK="$RAW/.qlock_$S"
  DEST="$RAW/$S"; TGZ="$RAW/$S.tar.gz"
  SIZE=$(tar_size "$S"); WANT=$(expected_md5 "$S")
  if [ -z "$SIZE" ] || [ -z "$WANT" ]; then flag "$S" "not in filelist/checksums"; continue; fi

  FREE=$(( $(df -k "$RAW" | awk 'NR==2 {print $4}') * 1024 ))
  NEED=$(( SIZE * 3 / 2 ))
  if [ "$FREE" -lt "$NEED" ]; then
    flag "$S" "not enough disk (free $((FREE/1048576)) MB, need $((NEED/1048576)) MB)"; continue
  fi

  OK=0
  for attempt in 1 2 3 4; do
    rm -f "$TGZ"   # server does not support resume; always start clean
    [ "$attempt" -gt 1 ] && sleep $((30 * (attempt - 1)))   # back off after a failed attempt
    log "download $S attempt $attempt ($((SIZE/1048576)) MB)"
    t0=$(date +%s)
    if ! bash "$FETCH" "data/$S.tar.gz" "$TGZ" >> "$LOG" 2>&1; then
      log "download error $S attempt $attempt"; continue
    fi
    dt=$(( $(date +%s) - t0 )); [ "$dt" -gt 0 ] || dt=1
    GOT=$( (md5 -q "$TGZ" 2>/dev/null) || md5sum "$TGZ" | awk '{print $1}')
    if [ "$GOT" = "$WANT" ]; then log "md5 OK $S ($((SIZE/1048576/dt)) MB/s)"; OK=1; break; fi
    log "md5 MISMATCH $S attempt $attempt (got $GOT want $WANT)"
  done
  if [ "$OK" -ne 1 ]; then rm -f "$TGZ"; flag "$S" "download or md5 failed 4 times"; continue; fi

  log "extract $S"
  if ! tar -xzf "$TGZ" -C "$RAW" --exclude='*.eeg' --exclude='*ClusterQualityMeasures*'; then
    flag "$S" "extraction failed"; continue
  fi
  rm -f "$TGZ"
  if [ ! -d "$DEST" ]; then flag "$S" "archive did not create $DEST"; continue; fi
  check_files "$S"
  echo "md5=$WANT verified=$(date '+%F %T')" > "$DEST/.verified"
  log "done $S"
  release
done
log "worker finished (queue empty or all remaining sessions busy)"
