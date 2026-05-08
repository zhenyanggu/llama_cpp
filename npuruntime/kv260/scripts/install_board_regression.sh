#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  install_board_regression.sh [options]

Create or update the self-contained KV260 overlay regression environment on a board.
This script runs on the host. The installed regression runs later entirely on the board.

Options:
  --host <host>              Board host/IP (default: 192.168.0.10)
  --user <user>              Board SSH user (default: ubuntu)
  --remote-root <path>       Board install root (default: /home/ubuntu/kv260-regression)
  --sdk-env <path>           KV260 SDK environment script
  --driver-ko <path>         npu_kv260.ko to install
  --skip-build               Reuse existing runtime test binaries
  --ssh-password-env <var>   Env var containing SSH password (default: KV260_SSH_PASSWORD)
  --sudo-password-env <var>  Env var name documented for board sudo (default: KV260_SUDO_PASSWORD)
  --dry-run                  Print actions without copying to the board
  -h, --help                 Show this help

Examples:
  bash npuruntime/kv260/scripts/install_board_regression.sh
  KV260_SSH_PASSWORD=123456 bash npuruntime/kv260/scripts/install_board_regression.sh --host 192.168.0.10
USAGE
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
KV260_DIR="$ROOT_DIR/npuruntime/kv260"
RUNTIME_DIR="$KV260_DIR/runtime"
DOC_PATH="$KV260_DIR/docs/BOARD_OVERLAY_REGRESSION.md"
BOARD_RUNNER="$KV260_DIR/scripts/run_regression.sh"

HOST="192.168.0.10"
USER_NAME="ubuntu"
REMOTE_ROOT="/home/ubuntu/kv260-regression"
SDK_ENV="/home/gugugu/petalinux/sdk/kv260-2025.1/environment-setup-cortexa72-cortexa53-amd-linux"
DRIVER_KO="$KV260_DIR/driver/npu_kv260.ko"
SKIP_BUILD=0
SSH_PASSWORD_ENV="KV260_SSH_PASSWORD"
SUDO_PASSWORD_ENV="KV260_SUDO_PASSWORD"
DRY_RUN=0

while [ $# -gt 0 ]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --user) USER_NAME="$2"; shift 2 ;;
    --remote-root) REMOTE_ROOT="$2"; shift 2 ;;
    --sdk-env) SDK_ENV="$2"; shift 2 ;;
    --driver-ko) DRIVER_KO="$2"; shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --ssh-password-env) SSH_PASSWORD_ENV="$2"; shift 2 ;;
    --sudo-password-env) SUDO_PASSWORD_ENV="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

TARGET="${USER_NAME}@${HOST}"
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR)
ASKPASS_DIR=""

cleanup() {
  if [ -n "$ASKPASS_DIR" ]; then
    rm -rf "$ASKPASS_DIR"
  fi
}
trap cleanup EXIT

setup_askpass_if_needed() {
  local password="${!SSH_PASSWORD_ENV:-}"
  if [ -z "$password" ]; then
    SSH_OPTS+=(-o BatchMode=yes)
    return 0
  fi

  ASKPASS_DIR="$(mktemp -d)"
  chmod 700 "$ASKPASS_DIR"
  printf '%s' "$password" > "$ASKPASS_DIR/password"
  chmod 600 "$ASKPASS_DIR/password"
  cat > "$ASKPASS_DIR/askpass.sh" <<ASKPASS
#!/usr/bin/env bash
cat "$ASKPASS_DIR/password"
ASKPASS
  chmod 700 "$ASKPASS_DIR/askpass.sh"
  SSH_OPTS+=(-o BatchMode=no)
}

ssh_cmd() {
  if [ -n "$ASKPASS_DIR" ] && command -v setsid >/dev/null 2>&1; then
    DISPLAY=none SSH_ASKPASS="$ASKPASS_DIR/askpass.sh" SSH_ASKPASS_REQUIRE=force \
      setsid -w ssh "${SSH_OPTS[@]}" "$TARGET" "$@"
  else
    ssh "${SSH_OPTS[@]}" "$TARGET" "$@"
  fi
}

tar_to_remote() {
  local src_dir="$1"
  local remote_root_q
  printf -v remote_root_q '%q' "$REMOTE_ROOT"
  if [ -n "$ASKPASS_DIR" ] && command -v setsid >/dev/null 2>&1; then
    tar -C "$src_dir" -cf - . | DISPLAY=none SSH_ASKPASS="$ASKPASS_DIR/askpass.sh" SSH_ASKPASS_REQUIRE=force \
      setsid -w ssh "${SSH_OPTS[@]}" "$TARGET" "mkdir -p $remote_root_q && tar -C $remote_root_q -xpf -"
  else
    tar -C "$src_dir" -cf - . | ssh "${SSH_OPTS[@]}" "$TARGET" "mkdir -p $remote_root_q && tar -C $remote_root_q -xpf -"
  fi
}

require_file() {
  local path="$1"
  local label="$2"
  [ -f "$path" ] || { echo "Missing $label: $path" >&2; exit 2; }
}

require_executable() {
  local path="$1"
  local label="$2"
  [ -x "$path" ] || { echo "Missing or non-executable $label: $path" >&2; exit 2; }
}

build_runtime_tests() {
  [ -f "$SDK_ENV" ] || { echo "Missing SDK env: $SDK_ENV" >&2; exit 2; }
  unset LD_LIBRARY_PATH || true
  # shellcheck disable=SC1090
  source "$SDK_ENV"
  make -C "$RUNTIME_DIR" all
}

RUNTIME_BINS=(
  kv260_npu_smoke_test
  kv260_runtime_init_test
  kv260_dma_loopback_test
  kv260_dma_acc_int32_fp32_test
  kv260_mvin_problem_case_test
  kv260_dma_double_mvin_async_test
  kv260_layer_gemm_replay_test
  kv260_mmproj_layer_asym_w8a8_test
  kv260_overlay_switch_test
)

if [ "$SKIP_BUILD" -ne 1 ]; then
  echo "== build runtime tests =="
  build_runtime_tests
else
  echo "== skip build =="
fi

require_file "$DRIVER_KO" "driver module"
require_file "$DOC_PATH" "board documentation"
require_executable "$BOARD_RUNNER" "board runner"
for bin in "${RUNTIME_BINS[@]}"; do
  require_executable "$RUNTIME_DIR/$bin" "$bin"
done

STAGING="$(mktemp -d)"
trap 'rm -rf "$STAGING"; cleanup' EXIT
mkdir -p "$STAGING/bin" "$STAGING/driver" "$STAGING/scripts" "$STAGING/runs"
cp "$DRIVER_KO" "$STAGING/driver/npu_kv260.ko"
cp "$BOARD_RUNNER" "$STAGING/scripts/run_regression.sh"
cp "$DOC_PATH" "$STAGING/README.md"
for bin in "${RUNTIME_BINS[@]}"; do
  cp "$RUNTIME_DIR/$bin" "$STAGING/bin/$bin"
done
chmod +x "$STAGING/scripts/run_regression.sh" "$STAGING/bin"/*

cat > "$STAGING/INSTALL_META.txt" <<META
installed_from=$ROOT_DIR
remote_root=$REMOTE_ROOT
sdk_env=$SDK_ENV
driver_ko=$DRIVER_KO
sudo_password_env=$SUDO_PASSWORD_ENV
created_at=$(date -Is)
META

if [ "$DRY_RUN" -eq 1 ]; then
  echo "== dry run =="
  echo "target=$TARGET"
  echo "remote_root=$REMOTE_ROOT"
  echo "payload files:"
  find "$STAGING" -type f -printf '%P\n' | sort
  exit 0
fi

setup_askpass_if_needed

echo "== probe ssh =="
ssh_cmd "printf 'connected: '; hostname; mkdir -p '$REMOTE_ROOT'"

echo "== upload regression payload =="
tar_to_remote "$STAGING"

echo "== set executable bits =="
ssh_cmd "chmod +x '$REMOTE_ROOT/scripts/run_regression.sh' '$REMOTE_ROOT/bin/'* && find '$REMOTE_ROOT' -maxdepth 2 -type f | sort"

cat <<EOF_DONE

Installed KV260 regression environment on $TARGET:$REMOTE_ROOT
Board-side run command:
  cd $REMOTE_ROOT
  ./scripts/run_regression.sh --app-name double_dma_overlayapp

If sudo needs a password on the board:
  export $SUDO_PASSWORD_ENV='<board-sudo-password>'
  ./scripts/run_regression.sh
EOF_DONE
