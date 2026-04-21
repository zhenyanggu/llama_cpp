#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  board_init_overlay.sh [options]

Options:
  --remote-root <path>
  --app-name <name>
  --sudo-password-env <var>
  --output <path>
USAGE
}

REMOTE_ROOT="$HOME/aicas"
APP_NAME="double_dma_overlayapp"
SUDO_PASSWORD_ENV="BOARD_SUDO_PASSWORD"
OUTPUT_PATH=""

while [ $# -gt 0 ]; do
  case "$1" in
    --remote-root) REMOTE_ROOT="$2"; shift 2 ;;
    --app-name) APP_NAME="$2"; shift 2 ;;
    --sudo-password-env) SUDO_PASSWORD_ENV="$2"; shift 2 ;;
    --output) OUTPUT_PATH="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [ -n "$OUTPUT_PATH" ]; then
  mkdir -p "$(dirname "$OUTPUT_PATH")"
  exec > >(tee "$OUTPUT_PATH") 2>&1
fi

run_sudo() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
    return 0
  fi
  if sudo -n true >/dev/null 2>&1; then
    sudo "$@"
    return 0
  fi
  local password="${!SUDO_PASSWORD_ENV:-}"
  [ -n "$password" ] || { echo "Missing sudo password in env var: $SUDO_PASSWORD_ENV" >&2; exit 1; }
  printf '%s\n' "$password" | sudo -S -p '' "$@"
}

APP_DIR="$REMOTE_ROOT/board_support/overlay/$APP_NAME"
DRIVER_SRC="$REMOTE_ROOT/board_support/driver/npu_kv260.ko"
DST_DIR="/lib/firmware/xilinx/$APP_NAME"
GROUP_NAME="$(id -gn)"

[ -x /usr/bin/xmutil ] || { echo "xmutil not found at /usr/bin/xmutil" >&2; exit 1; }
[ -f "$APP_DIR/shell.json" ] || { echo "Missing shell.json under $APP_DIR" >&2; exit 1; }
[ -f "$APP_DIR/double_dma.dtbo" ] || { echo "Missing double_dma.dtbo under $APP_DIR" >&2; exit 1; }
[ -f "$APP_DIR/double_dma.bit.bin" ] || { echo "Missing double_dma.bit.bin under $APP_DIR" >&2; exit 1; }
[ -f "$DRIVER_SRC" ] || { echo "Missing driver under $DRIVER_SRC" >&2; exit 1; }

echo "[install overlay app]"
run_sudo mkdir -p "$DST_DIR"
run_sudo install -m 0644 "$APP_DIR/shell.json" "$DST_DIR/shell.json"
run_sudo install -m 0644 "$APP_DIR/double_dma.dtbo" "$DST_DIR/double_dma.dtbo"
run_sudo install -m 0644 "$APP_DIR/double_dma.bit.bin" "$DST_DIR/double_dma.bit.bin"
if [ -f "$APP_DIR/pl.dtsi" ]; then
  run_sudo install -m 0644 "$APP_DIR/pl.dtsi" "$DST_DIR/pl.dtsi"
fi

echo
echo "[xmutil loadapp]"
run_sudo xmutil unloadapp || true
run_sudo xmutil loadapp "$APP_NAME"

echo
echo "[driver load]"
if grep -q '^npu_kv260 ' /proc/modules; then
  run_sudo rmmod npu_kv260 || true
fi
run_sudo insmod "$DRIVER_SRC"

if [ -e /dev/npu_kv260 ]; then
  run_sudo chgrp "$GROUP_NAME" /dev/npu_kv260
  run_sudo chmod 660 /dev/npu_kv260
fi

echo
echo "[overlay dir]"
run_sudo ls -lah "$DST_DIR"

echo
echo "[xmutil listapps]"
run_sudo xmutil listapps || true

echo
echo "[/dev/npu_kv260]"
ls -l /dev/npu_kv260 2>/dev/null || true

echo
echo "[/proc/modules]"
grep '^npu_kv260 ' /proc/modules || true

echo
echo "[/sys/bus/platform/devices]"
ls /sys/bus/platform/devices 2>/dev/null | grep -Ei 'npu|a0000000' || true

echo
echo "[summary]"
run_sudo xmutil listapps 2>/dev/null | grep -Eq "(^|[[:space:]])$APP_NAME([[:space:]]|$)"
[ -e /dev/npu_kv260 ]
grep -q '^npu_kv260 ' /proc/modules
ls /sys/bus/platform/devices 2>/dev/null | grep -Eiq 'npu|a0000000'
echo "xmutil_app=1 dev=1 module=1 platform=1"
