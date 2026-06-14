#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_attention_overlap_ablation_kv260.sh [options]

Runs the prefill attention P-matrix overlap ablation on KV260 and writes JSON.

Options:
  --host <host>                 KV260 host (default: 192.168.0.10)
  --user <user>                 KV260 user (default: ubuntu)
  --sudo-password <password>    Board sudo password (default: 123456)
  --run-id <id>                 Run id (default: timestamp)
  --output-root <path>          Local output root
  --remote-root <path>          Remote run root
  --sdk-env <path>              KV260 SDK env script
  --prefill-overlay-app <name>  Prefill overlay app
  --prefill-overlay-dir <path>  Local prefill overlay directory
  --driver-ko <path>            Local npu_kv260.ko
  --heads <n>                   Number of q heads to pipeline (default: 5)
  --single-head                 Run old single-head intra-head PV overlap only
  --skip-build                  Reuse existing runtime binary
  --skip-overlay-sync           Do not copy local overlay dir to /lib/firmware/xilinx
  --skip-driver-reload          Do not reload npu_kv260.ko
  -h, --help                    Show help
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

HOST="192.168.0.10"
USER_NAME="ubuntu"
SUDO_PASSWORD="${BOARD_SUDO_PASSWORD:-123456}"
RUN_ID="$(date +%Y%m%dT%H%M%S)-attention-overlap-ablation"
OUTPUT_ROOT="$REPO_DIR/AICAS2026/aicas_semi/results/attention_overlap_ablation"
REMOTE_ROOT="/home/ubuntu/attention-overlap-ablation"
SDK_ENV="/home/gugugu/petalinux/sdk/kv260-2025.1/environment-setup-cortexa72-cortexa53-amd-linux"
PREFILL_OVERLAY_APP="prefill_190m_qkpipe_20260608_0206_app"
PREFILL_OVERLAY_DIR="/mnt/c/vivado/KV260/out/prefill_190m_qkpipe_20260608_0206_app"
PREFILL_RUNTIME_DIR="/mnt/c/vivado/VersaEdge/Versa_P/runtime"
DRIVER_KO="/home/gugugu/work/kv260-xilinx-6.8-driver-env/npu_kv260_build/npu_kv260.ko"
HEADS=5
SINGLE_HEAD=0
SKIP_BUILD=0
SKIP_OVERLAY_SYNC=0
SKIP_DRIVER_RELOAD=0

while [ $# -gt 0 ]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --user) USER_NAME="$2"; shift 2 ;;
    --sudo-password) SUDO_PASSWORD="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --remote-root) REMOTE_ROOT="$2"; shift 2 ;;
    --sdk-env) SDK_ENV="$2"; shift 2 ;;
    --prefill-overlay-app) PREFILL_OVERLAY_APP="$2"; shift 2 ;;
    --prefill-overlay-dir) PREFILL_OVERLAY_DIR="$2"; shift 2 ;;
    --driver-ko) DRIVER_KO="$2"; shift 2 ;;
    --heads) HEADS="$2"; shift 2 ;;
    --single-head) SINGLE_HEAD=1; shift ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --skip-overlay-sync) SKIP_OVERLAY_SYNC=1; shift ;;
    --skip-driver-reload) SKIP_DRIVER_RELOAD=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

TARGET="$USER_NAME@$HOST"
LOCAL_RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
REMOTE_RUN_DIR="$REMOTE_ROOT/$RUN_ID"
SSH_OPTS=(
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
  -o LogLevel=ERROR
)

mkdir -p "$LOCAL_RUN_DIR"

run_ssh() {
  ssh "${SSH_OPTS[@]}" "$TARGET" "$@"
}

build_binaries() {
  if [ "$SKIP_BUILD" -eq 1 ]; then
    return
  fi
  (
    set -euo pipefail
    source "$SDK_ENV"
    make -C "$PREFILL_RUNTIME_DIR" \
      versa_p_runtime_init_test \
      versa_p_attention_qk_pv_log8_test \
      -j1
  )
}

stage_remote() {
  run_ssh "mkdir -p '$REMOTE_RUN_DIR/bin' '$REMOTE_RUN_DIR/logs'"
  scp "${SSH_OPTS[@]}" \
    "$PREFILL_RUNTIME_DIR/build/bin/versa_p_runtime_init_test" \
    "$PREFILL_RUNTIME_DIR/build/bin/versa_p_attention_qk_pv_log8_test" \
    "$TARGET:$REMOTE_RUN_DIR/bin/"
}

sync_overlay() {
  if [ "$SKIP_OVERLAY_SYNC" -eq 1 ]; then
    return
  fi
  if [ ! -d "$PREFILL_OVERLAY_DIR" ]; then
    echo "Overlay directory not found: $PREFILL_OVERLAY_DIR" >&2
    exit 2
  fi
  scp -r "${SSH_OPTS[@]}" "$PREFILL_OVERLAY_DIR" "$TARGET:/home/$USER_NAME/upload-$PREFILL_OVERLAY_APP"
  run_ssh "
    set -e
    echo '$SUDO_PASSWORD' | sudo -S rm -rf '/lib/firmware/xilinx/$PREFILL_OVERLAY_APP'
    echo '$SUDO_PASSWORD' | sudo -S mkdir -p '/lib/firmware/xilinx/$PREFILL_OVERLAY_APP'
    echo '$SUDO_PASSWORD' | sudo -S cp -a '/home/$USER_NAME/upload-$PREFILL_OVERLAY_APP/.' '/lib/firmware/xilinx/$PREFILL_OVERLAY_APP/'
  "
}

load_overlay() {
  run_ssh "
    set -e
    echo '$SUDO_PASSWORD' | sudo -S xmutil unloadapp >/dev/null 2>&1 || true
    echo '$SUDO_PASSWORD' | sudo -S xmutil loadapp '$PREFILL_OVERLAY_APP'
    echo '$SUDO_PASSWORD' | sudo -S chmod 666 /dev/npu_kv260 2>/dev/null || true
    grep -E 'firmware-name|compatible|interrupts|reg =' '/lib/firmware/xilinx/$PREFILL_OVERLAY_APP/pl.dtsi' || true
    ls -l /dev/npu_kv260
  " | tee "$LOCAL_RUN_DIR/load_${PREFILL_OVERLAY_APP}.log"
}

reload_driver() {
  if [ "$SKIP_DRIVER_RELOAD" -eq 1 ]; then
    return
  fi
  if [ ! -f "$DRIVER_KO" ]; then
    echo "Driver module not found: $DRIVER_KO" >&2
    exit 2
  fi
  run_ssh "mkdir -p /home/$USER_NAME/npu-driver-current"
  scp "${SSH_OPTS[@]}" "$DRIVER_KO" "$TARGET:/home/$USER_NAME/npu-driver-current/npu_kv260.ko"
  run_ssh "
    set -e
    echo '$SUDO_PASSWORD' | sudo -S rmmod npu_kv260 2>/dev/null || true
    echo '$SUDO_PASSWORD' | sudo -S insmod '/home/$USER_NAME/npu-driver-current/npu_kv260.ko'
    echo '$SUDO_PASSWORD' | sudo -S chmod 666 /dev/npu_kv260
    lsmod | grep '^npu_kv260'
  " | tee "$LOCAL_RUN_DIR/reload_driver.log"
}

run_remote() {
  sync_overlay
  load_overlay
  reload_driver
  run_ssh "
    set -e
    set -o pipefail
    cd '$REMOTE_RUN_DIR/bin'
    ./versa_p_runtime_init_test 2>&1 | tee '$REMOTE_RUN_DIR/logs/prefill_runtime_init.log'
    if [ '$SINGLE_HEAD' -eq 1 ]; then
      ./versa_p_attention_qk_pv_log8_test --p-overlap-only 2>&1 | tee '$REMOTE_RUN_DIR/logs/attention_p_overlap_ablation.log'
    else
      ./versa_p_attention_qk_pv_log8_test --p-overlap-heads '$HEADS' 2>&1 | tee '$REMOTE_RUN_DIR/logs/attention_p_overlap_ablation.log'
    fi
  "
}

pull_and_analyze() {
  scp -r "${SSH_OPTS[@]}" "$TARGET:$REMOTE_RUN_DIR/logs/." "$LOCAL_RUN_DIR/"
  python3 "$SCRIPT_DIR/analyze_attention_overlap_ablation.py" \
    --run-id "$RUN_ID" \
    --log "$LOCAL_RUN_DIR/attention_p_overlap_ablation.log" \
    --overlay "$PREFILL_OVERLAY_APP" \
    --out-json "$LOCAL_RUN_DIR/attention_p_overlap_ablation_summary.json" \
    --notes "$LOCAL_RUN_DIR/attention_p_overlap_ablation_notes.md"
}

build_binaries
stage_remote
run_remote
pull_and_analyze

echo "Attention overlap ablation results: $LOCAL_RUN_DIR"
