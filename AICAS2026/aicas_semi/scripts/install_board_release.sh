#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  install_board_release.sh [options]

Install a prepared AICAS semi board payload onto the KV260.

Options:
  --host <host>             KV260 host (default: 192.168.0.10)
  --user <user>             SSH user (default: ubuntu)
  --payload-dir <path>      Local payload directory
  --remote-root <path>      Remote release root (default: /home/ubuntu/aicas-semi-release)
  --run-link <path>         Remote convenience script path (default: /home/ubuntu/run_semi.sh)
  --sudo-password <pass>    Board sudo password (default: BOARD_SUDO_PASSWORD or 123456)
  --skip-firmware           Do not install overlay apps into /lib/firmware/xilinx
  --verify-only             Do not copy, only run fast remote sanity checks
  -h, --help                Show this help
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAYLOAD_DIR="/tmp/aicas-semi-board-release"
HOST="192.168.0.10"
USER_NAME="ubuntu"
REMOTE_ROOT="/home/ubuntu/aicas-semi-release"
RUN_LINK="/home/ubuntu/run_semi.sh"
SUDO_PASSWORD="${BOARD_SUDO_PASSWORD:-123456}"
SKIP_FIRMWARE=0
VERIFY_ONLY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --user) USER_NAME="$2"; shift 2 ;;
    --payload-dir) PAYLOAD_DIR="$2"; shift 2 ;;
    --remote-root) REMOTE_ROOT="$2"; shift 2 ;;
    --run-link) RUN_LINK="$2"; shift 2 ;;
    --sudo-password) SUDO_PASSWORD="$2"; shift 2 ;;
    --skip-firmware) SKIP_FIRMWARE=1; shift ;;
    --verify-only) VERIFY_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

TARGET="$USER_NAME@$HOST"
PAYLOAD_DIR="$(cd "$PAYLOAD_DIR" && pwd)"
SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR)

require_path() {
  local path="$1"
  local label="$2"
  [ -e "$path" ] || { echo "Missing $label: $path" >&2; exit 1; }
}

require_path "$PAYLOAD_DIR/run_semi.sh" "payload run script"
require_path "$PAYLOAD_DIR/bin/llama-server" "payload llama-server"

if [ "$VERIFY_ONLY" -ne 1 ]; then
  ssh "${SSH_OPTS[@]}" "$TARGET" "mkdir -p '$REMOTE_ROOT'"
  if command -v rsync >/dev/null 2>&1 && ssh "${SSH_OPTS[@]}" "$TARGET" "command -v rsync >/dev/null 2>&1"; then
    rsync -a --delete -e "ssh ${SSH_OPTS[*]}" "$PAYLOAD_DIR/" "$TARGET:$REMOTE_ROOT/"
  else
    tar -C "$PAYLOAD_DIR" -cf - . | ssh "${SSH_OPTS[@]}" "$TARGET" "rm -rf '$REMOTE_ROOT' && mkdir -p '$REMOTE_ROOT' && cd '$REMOTE_ROOT' && tar -xf -"
  fi

  ssh "${SSH_OPTS[@]}" "$TARGET" "chmod +x '$REMOTE_ROOT/run_semi.sh' '$REMOTE_ROOT/code/run_semi_eval.sh' && ln -sfn '$REMOTE_ROOT/run_semi.sh' '$RUN_LINK'"

  if [ "$SKIP_FIRMWARE" -ne 1 ]; then
    ssh "${SSH_OPTS[@]}" "$TARGET" "SUDO_PASSWORD='$SUDO_PASSWORD' REMOTE_ROOT='$REMOTE_ROOT' bash -s" <<'EOF'
set -euo pipefail
install_app() {
  local app="$1"
  local src="$REMOTE_ROOT/board/overlays/$app"
  local dst="/lib/firmware/xilinx/$app"
  [ -d "$src" ] || { echo "Missing overlay payload: $src" >&2; exit 1; }
  printf '%s\n' "$SUDO_PASSWORD" | sudo -S mkdir -p "$dst"
  printf '%s\n' "$SUDO_PASSWORD" | sudo -S cp "$src"/* "$dst/"
}
install_app prefill_190m_qkpipe_20260608_0206_app
install_app dec_200m_latest_0609a_app
EOF
  fi
fi

ssh "${SSH_OPTS[@]}" "$TARGET" "cd '$REMOTE_ROOT' && chmod +x run_semi.sh && test -f data/FullTest.json && test -d data/images && test -x bin/llama-server && test -f models/text.gguf && test -f models/mmproj.gguf"

echo "Installed and sanity-checked: $TARGET:$REMOTE_ROOT"
echo "Board entrypoint: $RUN_LINK"
echo "Smoke test: bash $RUN_LINK --quick-smoke"
