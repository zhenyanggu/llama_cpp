#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  cleanup_kv260_board.sh [options]

Clean KV260 board development leftovers for a reproducible AICAS semi image.
Default mode is dry-run. With --execute, paths are moved into a quarantine
directory first. Nothing is permanently removed unless --delete-quarantine is
also supplied.

Options:
  --host <host>              KV260 host (default: 192.168.0.10)
  --user <user>              SSH user (default: ubuntu)
  --sudo-password <pass>     Board sudo password (default: BOARD_SUDO_PASSWORD or 123456)
  --release-root <path>      Release root to preserve (default: /home/ubuntu/aicas-semi-release)
  --execute                  Move cleanup candidates into quarantine
  --delete-quarantine        Permanently remove the quarantine created by this run
  --quarantine-root <path>   Quarantine root (default: /home/ubuntu/.aicas-cleanup-quarantine)
  -h, --help                 Show this help
USAGE
}

HOST="192.168.0.10"
USER_NAME="ubuntu"
SUDO_PASSWORD="${BOARD_SUDO_PASSWORD:-123456}"
RELEASE_ROOT="/home/ubuntu/aicas-semi-release"
EXECUTE=0
DELETE_QUARANTINE=0
QUARANTINE_ROOT="/home/ubuntu/.aicas-cleanup-quarantine"

while [ $# -gt 0 ]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --user) USER_NAME="$2"; shift 2 ;;
    --sudo-password) SUDO_PASSWORD="$2"; shift 2 ;;
    --release-root) RELEASE_ROOT="$2"; shift 2 ;;
    --execute) EXECUTE=1; shift ;;
    --delete-quarantine) DELETE_QUARANTINE=1; shift ;;
    --quarantine-root) QUARANTINE_ROOT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

TARGET="$USER_NAME@$HOST"
SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR)
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
REMOTE_LOG_DIR="/home/ubuntu/aicas-cleanup-$RUN_ID"

ssh "${SSH_OPTS[@]}" "$TARGET" \
  "SUDO_PASSWORD='$SUDO_PASSWORD' RELEASE_ROOT='$RELEASE_ROOT' EXECUTE='$EXECUTE' DELETE_QUARANTINE='$DELETE_QUARANTINE' QUARANTINE_ROOT='$QUARANTINE_ROOT' REMOTE_LOG_DIR='$REMOTE_LOG_DIR' RUN_ID='$RUN_ID' bash -s" <<'EOF'
set -euo pipefail

sudo_sh() {
  printf '%s\n' "$SUDO_PASSWORD" | sudo -S "$@"
}

mkdir -p "$REMOTE_LOG_DIR"
df -h > "$REMOTE_LOG_DIR/df_before.txt"
du -xh --max-depth=1 /home/ubuntu 2>/dev/null | sort -h > "$REMOTE_LOG_DIR/du_home_before.txt" || true
du -xh --max-depth=1 /lib/firmware/xilinx 2>/dev/null | sort -h > "$REMOTE_LOG_DIR/du_firmware_before.txt" || true

QUARANTINE="$QUARANTINE_ROOT/$RUN_ID"
HOME_CANDIDATES=(
  /home/ubuntu/aicas-semi
  /home/ubuntu/aicas
  /home/ubuntu/kv260-llamaserver-throughput
  /home/ubuntu/versa-kvquant-diag
  /home/ubuntu/versa-decode-runs
  /home/ubuntu/versa-runtime-kvquant
  /home/ubuntu/overlap-ablation
  /home/ubuntu/operator-microbench
  /home/ubuntu/attn-repro
  /home/ubuntu/versa-runtime-qk-pv
  /home/ubuntu/attention-overlap-ablation
  /home/ubuntu/roofline-kernel
  /home/ubuntu/versa-fastpath-runtime
  /home/ubuntu/overlay-upload
  /home/ubuntu/overlay_stage
  /home/ubuntu/overlay_staging
  /home/ubuntu/overlay-upload
  /home/ubuntu/versa-decode-runtime
  /home/ubuntu/versa-runtime-regression
  /home/ubuntu/APPS
  /home/ubuntu/dec_qkfix_test
  /home/ubuntu/versa_prefill_app.staging
  /home/ubuntu/acc_eval.py
  /home/ubuntu/install_xmutil_app.sh
  /home/ubuntu/kv260-kvcache-maint-bench
  /home/ubuntu/kv260_ffn_swiglu_actscale_test.fp32
  /home/ubuntu/kv260_gemv_modes_test
  /home/ubuntu/kv260_runtime_init_test_cma1200
  /home/ubuntu/kv260_stream_gemv_api_test
  /home/ubuntu/kv260_stream_gemv_api_test_headrt_currenttest
  /home/ubuntu/kv260_stream_gemv_api_test_random_headrt
  /home/ubuntu/llama-mtmd-profiler
  /home/ubuntu/npu_kv260.ko
  /home/ubuntu/npu_kv260_uapi.h
  /home/ubuntu/quick_server.pid
  /home/ubuntu/set_ip.sh
  /home/ubuntu/test.sh
  /home/ubuntu/unload.sh
)

while IFS= read -r path; do
  HOME_CANDIDATES+=("$path")
done < <(find /home/ubuntu -maxdepth 1 -mindepth 1 \( \
  -name 'upload-*' -o \
  -name 'decode_*_app*' -o \
  -name 'dec_*_app*' -o \
  -name 'prefill_*_app*' -o \
  -name 'T_NPU_*_app*' -o \
  -name 'npu-driver-current' \
\) -print 2>/dev/null | sort)

PRESERVE_FIRMWARE=(
  /lib/firmware/xilinx/k24-starter-kits
  /lib/firmware/xilinx/k26-starter-kits
  /lib/firmware/xilinx/prefill_190m_qkpipe_20260608_0206_app
  /lib/firmware/xilinx/dec_200m_latest_0609a_app
)

{
  echo "# Cleanup run: $RUN_ID"
  echo "# execute=$EXECUTE delete_quarantine=$DELETE_QUARANTINE"
  echo "# preserve release root: $RELEASE_ROOT"
  echo
  echo "## Home candidates"
  for path in "${HOME_CANDIDATES[@]}"; do
    [ -e "$path" ] || continue
    [ "$path" = "$RELEASE_ROOT" ] && continue
    printf '%s\n' "$path"
  done
  echo
  echo "## Firmware candidates"
  for path in /lib/firmware/xilinx/*; do
    [ -e "$path" ] || continue
    keep=0
    for preserve in "${PRESERVE_FIRMWARE[@]}"; do
      [ "$path" = "$preserve" ] && keep=1
    done
    [ "$keep" -eq 1 ] && continue
    printf '%s\n' "$path"
  done
} > "$REMOTE_LOG_DIR/cleanup_manifest.txt"

if [ "$EXECUTE" -ne 1 ]; then
  cat "$REMOTE_LOG_DIR/cleanup_manifest.txt"
  echo
  echo "Dry-run only. Re-run with --execute to move these paths into quarantine."
  exit 0
fi

mkdir -p "$QUARANTINE/home" "$QUARANTINE/firmware_xilinx"

move_home_path() {
  local path="$1"
  [ -e "$path" ] || return 0
  [ "$path" = "$RELEASE_ROOT" ] && return 0
  mv "$path" "$QUARANTINE/home/$(basename "$path")"
}

move_firmware_path() {
  local path="$1"
  [ -e "$path" ] || return 0
  local keep=0
  for preserve in "${PRESERVE_FIRMWARE[@]}"; do
    [ "$path" = "$preserve" ] && keep=1
  done
  [ "$keep" -eq 1 ] && return 0
  sudo_sh mv "$path" "$QUARANTINE/firmware_xilinx/$(basename "$path")"
}

for path in "${HOME_CANDIDATES[@]}"; do
  move_home_path "$path"
done
for path in /lib/firmware/xilinx/*; do
  move_firmware_path "$path"
done

df -h > "$REMOTE_LOG_DIR/df_after_quarantine.txt"
du -xh --max-depth=1 /home/ubuntu 2>/dev/null | sort -h > "$REMOTE_LOG_DIR/du_home_after_quarantine.txt" || true
du -xh --max-depth=1 /lib/firmware/xilinx 2>/dev/null | sort -h > "$REMOTE_LOG_DIR/du_firmware_after_quarantine.txt" || true

if [ "$DELETE_QUARANTINE" -eq 1 ]; then
  sudo_sh rm -rf "$QUARANTINE"
  df -h > "$REMOTE_LOG_DIR/df_after_delete.txt"
fi

echo "Cleanup logs: $REMOTE_LOG_DIR"
echo "Quarantine: $QUARANTINE"
EOF
