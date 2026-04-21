#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  install_xmutil_app.sh <app_dir>
USAGE
}

if [ $# -ne 1 ]; then
  usage >&2
  exit 1
fi

APP_DIR="$(realpath "$1")"
[ -d "$APP_DIR" ] || { echo "App directory not found: $APP_DIR" >&2; exit 1; }
APP_NAME="$(basename "$APP_DIR")"
DST_DIR="/lib/firmware/xilinx/$APP_NAME"

run_sudo() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  else
    sudo "$@"
  fi
}

mapfile -t DTBOS < <(find "$APP_DIR" -maxdepth 1 -type f -name '*.dtbo' | sort)
mapfile -t BITSTREAMS < <(find "$APP_DIR" -maxdepth 1 -type f \( -name '*.bit.bin' -o -name '*.bin' \) | sort)
SHELL_JSON="$APP_DIR/shell.json"

[ -f "$SHELL_JSON" ] || { echo "Missing shell.json in $APP_DIR" >&2; exit 1; }
[ "${#DTBOS[@]}" -gt 0 ] || { echo "Missing *.dtbo in $APP_DIR" >&2; exit 1; }
[ "${#BITSTREAMS[@]}" -gt 0 ] || { echo "Missing *.bit.bin or *.bin in $APP_DIR" >&2; exit 1; }

run_sudo mkdir -p "$DST_DIR"
for src in "$SHELL_JSON" "${DTBOS[@]}" "${BITSTREAMS[@]}"; do
  run_sudo install -m 0644 "$src" "$DST_DIR/$(basename "$src")"
done
run_sudo ls -lah "$DST_DIR"
