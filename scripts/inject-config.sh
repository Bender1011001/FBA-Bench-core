#!/usr/bin/env bash
set -euo pipefail

# Inject production values into site/config.js based on environment variables.
# This script performs in-place replacements only where @inject markers exist.
# Usage:
#   ANALYTICS_ENABLED=true ANALYTICS_TRACKING_ID=G-XXXX ANALYTICS_ENDPOINT=https://analytics.example.com \
#   ANALYTICS_RESPECT_DNT=false ENVIRONMENT=production bash scripts/inject-config.sh
#
# Windows note: Run under Git Bash or WSL. For PowerShell, you can set env vars like:
#   $env:ANALYTICS_ENABLED="true"; $env:ANALYTICS_TRACKING_ID="G-XXXX"; bash scripts/inject-config.sh

CONFIG_FILE="site/config.js"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Error: $CONFIG_FILE not found." >&2
  exit 1
fi

# Read envs; provide defaults to avoid accidental enabling
ANALYTICS_ENABLED="${ANALYTICS_ENABLED:-false}"
ANALYTICS_TRACKING_ID="${ANALYTICS_TRACKING_ID:-}"
ANALYTICS_ENDPOINT="${ANALYTICS_ENDPOINT:-}"
ANALYTICS_RESPECT_DNT="${ANALYTICS_RESPECT_DNT:-true}"
ENVIRONMENT="${ENVIRONMENT:-production}"

# Detect sed flavor (macOS vs GNU)
if sed --version >/dev/null 2>&1; then
  SED_INPLACE=(-i)
else
  # macOS/BSD sed needs empty string arg for -i
  SED_INPLACE=(-i '')
fi

# Helper: boolean normalization to "true" or "false"
to_bool() {
  case "${1,,}" in
    1|true|yes|on) echo "true" ;;
    0|false|no|off) echo "false" ;;
    *) echo "false" ;;
  esac
}

AE="$(to_bool "$ANALYTICS_ENABLED")"
ARDNT="$(to_bool "$ANALYTICS_RESPECT_DNT")"

# Replace booleans
sed "${SED_INPLACE[@]}" \
  -e "s/\(enabled:\s*\)false\(.*@inject ANALYTICS_ENABLED.*\)/\1${AE}\2/" \
  -e "s/\(respectDNT:\s*\)true\(.*@inject ANALYTICS_RESPECT_DNT.*\)/\1${ARDNT}\2/" \
  "$CONFIG_FILE"

# Replace strings (escape slashes)
ESC_TID="${ANALYTICS_TRACKING_ID//\//\\/}"
ESC_EP="${ANALYTICS_ENDPOINT//\//\\/}"
ESC_ENV="${ENVIRONMENT//\//\\/}"

sed "${SED_INPLACE[@]}" \
  -e "s/\(trackingId:\s*\)''\(.*@inject ANALYTICS_TRACKING_ID.*\)/\1'${ESC_TID}'\2/" \
  -e "s/\(endpoint:\s*\)''\(.*@inject ANALYTICS_ENDPOINT.*\)/\1'${ESC_EP}'\2/" \
  -e "s/\(environment:\s*\)'development'\(.*@inject ENVIRONMENT.*\)/\1'${ESC_ENV}'\2/" \
  "$CONFIG_FILE"

echo "Injected analytics config into ${CONFIG_FILE} (enabled=${AE}, respectDNT=${ARDNT}, trackingId set? $([[ -n \"$ANALYTICS_TRACKING_ID\" ]] && echo yes || echo no), endpoint set? $([[ -n \"$ANALYTICS_ENDPOINT\" ]] && echo yes || echo no), env='${ENVIRONMENT}')."
echo "Note: Do not commit ${CONFIG_FILE} with production secrets. Use this only at build/deploy time."
