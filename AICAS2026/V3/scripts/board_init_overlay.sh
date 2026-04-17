#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  board_init_npu.sh [options]

Initialize the KV260 NPU environment on the board.

Options:
  --remote-root <path>       Remote root for staged board support files.
  --sudo-password-env <var>  Environment variable that stores the sudo password.
  --output <path>            Write the readiness log to this path.
  -h, --help                 Show this help.
EOF
}

REMOTE_ROOT="$HOME/aicas"
SUDO_PASSWORD_ENV="BOARD_SUDO_PASSWORD"
OUTPUT_PATH=""

while [ $# -gt 0 ]; do
  case "$1" in
    --remote-root)
      REMOTE_ROOT="$2"
      shift 2
      ;;
    --sudo-password-env)
      SUDO_PASSWORD_ENV="$2"
      shift 2
      ;;
    --output)
      OUTPUT_PATH="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
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
  [ -n "$password" ] || {
    echo "Missing sudo password in env var: $SUDO_PASSWORD_ENV" >&2
    exit 1
  }

  printf '%s\n' "$password" | sudo -S -p '' "$@"
}

MYNPU_SRC="$REMOTE_ROOT/board_support/mynpu"
DRIVER_SRC="$REMOTE_ROOT/board_support/driver/npu_kv260.ko"
FIRMWARE_DST="/lib/firmware/xilinx/mynpu"
GROUP_NAME="$(id -gn)"

[ -x /usr/bin/xmutil ] || {
  echo "xmutil not found at /usr/bin/xmutil" >&2
  exit 1
}
[ -f "$MYNPU_SRC/KV260_410.bin" ] || {
  echo "Missing mynpu bitstream under $MYNPU_SRC" >&2
  exit 1
}
[ -f "$MYNPU_SRC/KV260_410.dtbo" ] || {
  echo "Missing mynpu overlay under $MYNPU_SRC" >&2
  exit 1
}
[ -f "$MYNPU_SRC/shell.json" ] || {
  echo "Missing mynpu shell.json under $MYNPU_SRC" >&2
  exit 1
}
[ -f "$DRIVER_SRC" ] || {
  echo "Missing driver under $DRIVER_SRC" >&2
  exit 1
}

echo "[install firmware]"
run_sudo mkdir -p "$FIRMWARE_DST"
run_sudo install -m 0644 "$MYNPU_SRC/KV260_410.bin" "$FIRMWARE_DST/KV260_410.bin"
run_sudo install -m 0644 "$MYNPU_SRC/KV260_410.dtbo" "$FIRMWARE_DST/KV260_410.dtbo"
run_sudo install -m 0644 "$MYNPU_SRC/shell.json" "$FIRMWARE_DST/shell.json"

echo
echo "[xmutil loadapp]"
run_sudo xmutil unloadapp || true
run_sudo xmutil loadapp mynpu

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
echo "[firmware dir]"
run_sudo ls -lah "$FIRMWARE_DST"

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

has_app=0
has_dev=0
has_module=0
has_platform=0

if run_sudo xmutil listapps 2>/dev/null | grep -Eq '(^|[[:space:]])mynpu([[:space:]]|$)'; then
  has_app=1
fi
if [ -e /dev/npu_kv260 ]; then
  has_dev=1
fi
if grep -q '^npu_kv260 ' /proc/modules; then
  has_module=1
fi
if ls /sys/bus/platform/devices 2>/dev/null | grep -Eiq 'npu|a0000000'; then
  has_platform=1
fi

echo
echo "[summary]"
echo "xmutil_mynpu=$has_app dev=$has_dev module=$has_module platform=$has_platform"

if [ "$has_app" -ne 1 ] || [ "$has_dev" -ne 1 ] || [ "$has_module" -ne 1 ] || [ "$has_platform" -ne 1 ]; then
  exit 1
fi
