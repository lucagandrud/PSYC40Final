#!/usr/bin/env bash
# Fetch one file from CRCNS fcx-1 via the login-form POST.
# Usage: crcns_fetch.sh <path-under-fcx-1> <output-file>
# Credentials come from env or ~/.crcns_credentials; the password is passed via
# a file descriptor so it never appears in `ps` or logs.
set -euo pipefail
REMOTE="$1"; OUT="$2"
CRED_FILE="${CRCNS_CRED_FILE:-$HOME/.crcns_credentials}"
if [ -z "${CRCNS_USERNAME:-}" ] && [ -f "$CRED_FILE" ]; then
  # shellcheck disable=SC1090
  source "$CRED_FILE"
fi
: "${CRCNS_USERNAME:?set CRCNS_USERNAME or create ~/.crcns_credentials}"
: "${CRCNS_PASSWORD:?set CRCNS_PASSWORD or create ~/.crcns_credentials}"
FN="fcx-1/${REMOTE}"
curl -sS -L --fail --retry 3 --retry-delay 5 -C - -o "$OUT" \
  --data-urlencode "username=${CRCNS_USERNAME}" \
  --data-urlencode password@<(printf '%s' "$CRCNS_PASSWORD") \
  --data-urlencode "fn=${FN}" \
  --data-urlencode "submit=Login" \
  "https://portal.nersc.gov/project/crcns/download/${FN}"
# A failed login returns HTML instead of the file.
if head -c 512 "$OUT" | grep -qi '<html\|<!doctype'; then
  case "$OUT" in *.html|*.htm) ;; *)
    echo "ERROR: got an HTML page instead of $REMOTE (login failed or bad path)" >&2
    rm -f "$OUT"; exit 2;;
  esac
fi
