#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  board_init_overlay.sh [options]

Initialize KV260 overlay + NPU driver for V2 flow.

Options:
  --host <host>              KV260 host (default: 192.168.0.10)
  --user <user>              KV260 user (default: ubuntu)
  --app <name>               xmutil app name (default: INT32_dequant_overlayapp)
  --remote-root <path>       Remote root (default: /home/ubuntu/aicas)
  --sudo-password <pass>     Board sudo password (default: env BOARD_SUDO_PASSWORD)
  -h, --help                 Show this help
EOF
}

HOST="192.168.0.10"
USER_NAME="ubuntu"
APP_NAME="INT32_dequant_overlayapp"
REMOTE_ROOT="/home/ubuntu/aicas"
SUDO_PASS="${BOARD_SUDO_PASSWORD:-}"

while [ $# -gt 0 ]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --user) USER_NAME="$2"; shift 2 ;;
    --app) APP_NAME="$2"; shift 2 ;;
    --remote-root) REMOTE_ROOT="$2"; shift 2 ;;
    --sudo-password) SUDO_PASS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage >&2; exit 1 ;;
  esac
done

[ -n "$SUDO_PASS" ] || { echo "Missing sudo password: pass --sudo-password or export BOARD_SUDO_PASSWORD" >&2; exit 1; }
TARGET="${USER_NAME}@${HOST}"

ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR "$TARGET" "bash -s" <<EOF
set -euo pipefail
PASS='${SUDO_PASS//\'/\'\"\'\"\'}'
APP_NAME='${APP_NAME//\'/\'\"\'\"\'}'
REMOTE_ROOT='${REMOTE_ROOT//\'/\'\"\'\"\'}'

run_sudo() { printf '%s\n' "\$PASS" | sudo -S -p '' "\$@"; }

run_sudo xmutil unloadapp || true
run_sudo xmutil loadapp "\$APP_NAME"

if [ -f "\$REMOTE_ROOT/board_support/driver/npu_kv260.ko" ]; then
  if grep -q '^npu_kv260 ' /proc/modules; then
    run_sudo rmmod npu_kv260 || true
  fi
  run_sudo insmod "\$REMOTE_ROOT/board_support/driver/npu_kv260.ko"
fi

if [ -e /dev/npu_kv260 ]; then
  run_sudo chgrp "\$(id -gn)" /dev/npu_kv260
  run_sudo chmod 660 /dev/npu_kv260
fi

run_sudo xmutil listapps || true
ls -l /dev/npu_kv260 || true
grep '^npu_kv260 ' /proc/modules || true
EOF
