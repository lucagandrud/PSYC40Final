#!/usr/bin/env bash
# Idempotent, resumable download of CRCNS fcx-1 sessions into data/raw/<session>/.
#
# Usage: scripts/download_data.sh [SESSION ...]
#   default: every session in data/manifests/filelist.txt, BWRat20_101013 and
#   20140526_277um first, then the rest smallest first.
# Several copies can run at once; a per-session lock stops them colliding.
#
# Per session: free-space check (1.5x tarball) -> download -> MD5 check against
# checksums.md5 (one retry) -> extract without *.eeg / *ClusterQualityMeasures*
# -> compare files with 0_data_file_list.txt -> delete tarball -> mark verified.
# Credentials: env vars or ~/.crcns_credentials (see crcns_fetch.sh).
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MAN="$ROOT/data/manifests"
RAW="$ROOT/data/raw"
LOG="$ROOT/data/download.log"
FETCH="$ROOT/scripts/crcns_fetch.sh"
MISSING="$MAN/missing_files.csv"
MD5FAIL="$MAN/md5_failures.txt"
mkdir -p "$RAW"
[ -f "$MISSING" ] || echo "session,file,expected_bytes,actual_bytes" > "$MISSING"

log() { echo "$(date '+%F %T') [$$] $*" | tee -a "$LOG"; }

CUR_LOCK=""
cleanup() { [ -n "$CUR_LOCK" ] && rm -rf "$CUR_LOCK"; }
trap cleanup EXIT INT TERM

for f in filelist.txt checksums.md5 0_data_file_list.txt; do
  [ -f "$MAN/$f" ] || { echo "missing $MAN/$f; fetch the manifests first"; exit 1; }
done

tar_size() {  # bytes, from filelist.txt
  awk -v p="data/$1.tar.gz" '$1==p {print $2}' "$MAN/filelist.txt"
}
expected_md5() {
  awk -v p="data/$1.tar.gz" '$2==p {print $1}' "$MAN/checksums.md5"
}

if [ "$#" -gt 0 ]; then
  SESSIONS="$*"
else
  FIRST="BWRat20_101013 20140526_277um"
  REST=$(awk '$1 ~ /^data\/.*\.tar\.gz$/ {s=$1; sub(/^data\//,"",s); sub(/\.tar\.gz$/,"",s); print $2, s}' \
           "$MAN/filelist.txt" | sort -n | awk '{print $2}')
  SESSIONS="$FIRST"
  for s in $REST; do
    case " $FIRST " in *" $s "*) ;; *) SESSIONS="$SESSIONS $s";; esac
  done
fi

check_files() {  # compare extracted files against the authors' file list
  local S="$1" n_missing=0 n_ok=0
  while read -r rel size; do
    local f="$RAW/$rel"
    if [ ! -f "$f" ]; then
      echo "$S,$rel,$size,MISSING" >> "$MISSING"; n_missing=$((n_missing+1))
    else
      local actual; actual=$(stat -f %z "$f" 2>/dev/null || stat -c %s "$f")
      if [ "$actual" != "$size" ]; then
        echo "$S,$rel,$size,$actual" >> "$MISSING"; n_missing=$((n_missing+1))
      else
        n_ok=$((n_ok+1))
      fi
    fi
  done < <(sed -n "s/^\`\.\/\(${S}\/[^']*\)' \([0-9]*\)$/\1 \2/p" "$MAN/0_data_file_list.txt" \
             | grep -v -e '\.eeg ' -e 'ClusterQualityMeasures')
  log "$S file check: $n_ok ok, $n_missing missing or wrong size"
}

for S in $SESSIONS; do
  DEST="$RAW/$S"
  TGZ="$RAW/$S.tar.gz"
  if [ -f "$DEST/.verified" ]; then log "skip $S (already verified)"; continue; fi
  if ! mkdir "$RAW/.lock_$S" 2>/dev/null; then log "skip $S (another worker has it)"; continue; fi
  CUR_LOCK="$RAW/.lock_$S"

  SIZE=$(tar_size "$S"); WANT=$(expected_md5 "$S")
  if [ -z "$SIZE" ] || [ -z "$WANT" ]; then log "FLAG $S: not in filelist/checksums"; rm -rf "$CUR_LOCK"; CUR_LOCK=""; continue; fi

  FREE=$(( $(df -k "$RAW" | awk 'NR==2 {print $4}') * 1024 ))
  NEED=$(( SIZE * 3 / 2 ))
  if [ "$FREE" -lt "$NEED" ]; then
    log "FLAG $S: not enough disk (free $((FREE/1048576)) MB, need $((NEED/1048576)) MB); skipping"
    rm -rf "$CUR_LOCK"; CUR_LOCK=""; continue
  fi

  OK=0
  for attempt in 1 2; do
    log "download $S attempt $attempt ($((SIZE/1048576)) MB)"
    t0=$(date +%s)
    if ! bash "$FETCH" "data/$S.tar.gz" "$TGZ" >> "$LOG" 2>&1; then
      log "download error $S attempt $attempt"; rm -f "$TGZ"; continue
    fi
    dt=$(( $(date +%s) - t0 )); [ "$dt" -gt 0 ] || dt=1
    GOT=$( (md5 -q "$TGZ" 2>/dev/null) || md5sum "$TGZ" | awk '{print $1}')
    if [ "$GOT" = "$WANT" ]; then
      log "md5 OK $S ($((SIZE/1048576/dt)) MB/s)"; OK=1; break
    fi
    log "md5 MISMATCH $S attempt $attempt (got $GOT want $WANT)"; rm -f "$TGZ"
  done
  if [ "$OK" -ne 1 ]; then
    log "FLAG $S: md5 failed twice"; echo "$S" >> "$MD5FAIL"
    rm -rf "$CUR_LOCK"; CUR_LOCK=""; continue
  fi

  log "extract $S"
  if ! tar -xzf "$TGZ" -C "$RAW" --exclude='*.eeg' --exclude='*ClusterQualityMeasures*'; then
    log "FLAG $S: extraction failed"; rm -rf "$CUR_LOCK"; CUR_LOCK=""; continue
  fi
  rm -f "$TGZ"
  if [ ! -d "$DEST" ]; then log "FLAG $S: archive did not create $DEST"; rm -rf "$CUR_LOCK"; CUR_LOCK=""; continue; fi
  check_files "$S"
  echo "md5=$WANT verified=$(date '+%F %T')" > "$DEST/.verified"
  log "done $S"
  rm -rf "$CUR_LOCK"; CUR_LOCK=""
done
log "worker finished"
