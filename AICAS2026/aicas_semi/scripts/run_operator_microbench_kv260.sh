#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_operator_microbench_kv260.sh [options]

Runs isolated KV260 operator microbenchmarks for plotting NPU/CPU speedup.
This does not start llama-server and does not run an end-to-end inference phase.

Options:
  --host <host>                 KV260 host (default: 192.168.0.10)
  --user <user>                 KV260 user (default: ubuntu)
  --sudo-password <password>    Board sudo password (default: 123456)
  --run-id <id>                 Run id (default: timestamp)
  --output-root <path>          Local output root
  --remote-root <path>          Remote run root (default: /home/ubuntu/operator-microbench)
  --sdk-env <path>              KV260 SDK env script
  --prefill-overlay-app <name>  Prefill overlay app
  --prefill-overlay-dir <path>  Local prefill overlay directory
  --decode-overlay-app <name>   Decode overlay app
  --decode-overlay-dir <path>   Local decode overlay directory
  --driver-ko <path>            Local npu_kv260.ko
  --cpu-repeats <n>             CPU reference repeats (default: 1)
  --skip-build                  Reuse existing runtime binaries
  --skip-overlay-sync           Do not copy local overlay dirs to /lib/firmware/xilinx
  --skip-driver-reload          Do not reload npu_kv260.ko
  -h, --help                    Show help
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CODE_DIR="$REPO_DIR/AICAS2026/aicas_semi/code"

HOST="192.168.0.10"
USER_NAME="ubuntu"
SUDO_PASSWORD="${BOARD_SUDO_PASSWORD:-123456}"
RUN_ID="$(date +%Y%m%dT%H%M%S)-operator-microbench"
OUTPUT_ROOT="$REPO_DIR/AICAS2026/aicas_semi/results/operator_microbench"
REMOTE_ROOT="/home/ubuntu/operator-microbench"
SDK_ENV="/home/gugugu/petalinux/sdk/kv260-2025.1/environment-setup-cortexa72-cortexa53-amd-linux"
PREFILL_OVERLAY_APP="prefill_190m_qkpipe_20260608_0206_app"
PREFILL_OVERLAY_DIR="/mnt/c/vivado/KV260/out/prefill_190m_qkpipe_20260608_0206_app"
DECODE_OVERLAY_APP="dec_200m_latest_0609a_app"
DECODE_OVERLAY_DIR="/mnt/c/vivado/KV260/out/dec_200m_latest_0609a_app"
DRIVER_KO="/home/gugugu/work/kv260-xilinx-6.8-driver-env/npu_kv260_build/npu_kv260.ko"
CPU_REPEATS=1
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
    --decode-overlay-app) DECODE_OVERLAY_APP="$2"; shift 2 ;;
    --decode-overlay-dir) DECODE_OVERLAY_DIR="$2"; shift 2 ;;
    --driver-ko) DRIVER_KO="$2"; shift 2 ;;
    --cpu-repeats) CPU_REPEATS="$2"; shift 2 ;;
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
PREFILL_RUNTIME_DIR="/mnt/c/vivado/VersaEdge/Versa_P/runtime"
DECODE_RUNTIME_DIR="/mnt/c/vivado/versa_decode/sw/kv260/runtime"
CPU_BIN_LOCAL="$LOCAL_RUN_DIR/operator_microbench_cpu"

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
    ${CXX:-g++} -std=c++17 -O3 -DNDEBUG "$CODE_DIR/operator_microbench_cpu.cpp" -o "$CPU_BIN_LOCAL"
    make -C "$PREFILL_RUNTIME_DIR" \
      versa_p_runtime_init_test \
      versa_p_layer_phase_profile_test \
      versa_p_attention_qk_pv_log8_test \
      -j1
    make -C "$DECODE_RUNTIME_DIR" \
      kv260_runtime_init_test \
      kv260_decode_full_stream_perf_test \
      -j1
  )
}

stage_remote() {
  run_ssh "mkdir -p '$REMOTE_RUN_DIR/bin' '$REMOTE_RUN_DIR/logs'"
  scp "${SSH_OPTS[@]}" "$CPU_BIN_LOCAL" "$TARGET:$REMOTE_RUN_DIR/bin/"
  scp "${SSH_OPTS[@]}" \
    "$PREFILL_RUNTIME_DIR/build/bin/versa_p_runtime_init_test" \
    "$PREFILL_RUNTIME_DIR/build/bin/versa_p_layer_phase_profile_test" \
    "$PREFILL_RUNTIME_DIR/build/bin/versa_p_attention_qk_pv_log8_test" \
    "$DECODE_RUNTIME_DIR/build/bin/kv260_runtime_init_test" \
    "$DECODE_RUNTIME_DIR/build/bin/kv260_decode_full_stream_perf_test" \
    "$TARGET:$REMOTE_RUN_DIR/bin/"
}

sync_overlay() {
  local app="$1"
  local dir="$2"
  if [ "$SKIP_OVERLAY_SYNC" -eq 1 ]; then
    return
  fi
  if [ ! -d "$dir" ]; then
    echo "Overlay directory not found: $dir" >&2
    exit 2
  fi
  scp -r "${SSH_OPTS[@]}" "$dir" "$TARGET:/home/$USER_NAME/upload-$app"
  run_ssh "
    set -e
    echo '$SUDO_PASSWORD' | sudo -S rm -rf '/lib/firmware/xilinx/$app'
    echo '$SUDO_PASSWORD' | sudo -S mkdir -p '/lib/firmware/xilinx/$app'
    echo '$SUDO_PASSWORD' | sudo -S cp -a '/home/$USER_NAME/upload-$app/.' '/lib/firmware/xilinx/$app/'
  "
}

load_overlay() {
  local app="$1"
  run_ssh "
    set -e
    echo '$SUDO_PASSWORD' | sudo -S xmutil unloadapp >/dev/null 2>&1 || true
    echo '$SUDO_PASSWORD' | sudo -S xmutil loadapp '$app'
    echo '$SUDO_PASSWORD' | sudo -S chmod 666 /dev/npu_kv260 2>/dev/null || true
    grep -E 'firmware-name|compatible|interrupts|reg =' '/lib/firmware/xilinx/$app/pl.dtsi' || true
  " | tee "$LOCAL_RUN_DIR/load_${app}.log"
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

run_remote_benchmarks() {
  run_ssh "
    set -e
    cd '$REMOTE_RUN_DIR/bin'
    ./operator_microbench_cpu --repeats '$CPU_REPEATS' 2>&1 | tee '$REMOTE_RUN_DIR/logs/cpu_operator_microbench.log'
  "

  sync_overlay "$PREFILL_OVERLAY_APP" "$PREFILL_OVERLAY_DIR"
  load_overlay "$PREFILL_OVERLAY_APP"
  reload_driver
  run_ssh "
    set -e
    cd '$REMOTE_RUN_DIR/bin'
    ./versa_p_runtime_init_test 2>&1 | tee '$REMOTE_RUN_DIR/logs/prefill_runtime_init.log'
    ./versa_p_layer_phase_profile_test --layer m5_ffn_up_768x768x2048 2>&1 | tee '$REMOTE_RUN_DIR/logs/prefill_gemm.log'
    ./versa_p_attention_qk_pv_log8_test 2>&1 | tee '$REMOTE_RUN_DIR/logs/prefill_attention.log'
  "

  sync_overlay "$DECODE_OVERLAY_APP" "$DECODE_OVERLAY_DIR"
  load_overlay "$DECODE_OVERLAY_APP"
  reload_driver
  run_ssh "
    set -e
    cd '$REMOTE_RUN_DIR/bin'
    ./kv260_runtime_init_test 2>&1 | tee '$REMOTE_RUN_DIR/logs/decode_runtime_init.log'
    timeout 900s ./kv260_decode_full_stream_perf_test 2>&1 | tee '$REMOTE_RUN_DIR/logs/decode_full_stream.log'
  "
}

pull_and_analyze() {
  scp -r "${SSH_OPTS[@]}" "$TARGET:$REMOTE_RUN_DIR/logs/." "$LOCAL_RUN_DIR/"
  python3 "$SCRIPT_DIR/analyze_operator_microbench.py" \
    --run-id "$RUN_ID" \
    --cpu-log "$LOCAL_RUN_DIR/cpu_operator_microbench.log" \
    --prefill-gemm-log "$LOCAL_RUN_DIR/prefill_gemm.log" \
    --prefill-attention-log "$LOCAL_RUN_DIR/prefill_attention.log" \
    --decode-log "$LOCAL_RUN_DIR/decode_full_stream.log" \
    --prefill-overlay "$PREFILL_OVERLAY_APP" \
    --decode-overlay "$DECODE_OVERLAY_APP" \
    --out-json "$LOCAL_RUN_DIR/operator_speedup_summary.json" \
    --notes "$LOCAL_RUN_DIR/operator_speedup_notes.md"
}

build_binaries
stage_remote
run_remote_benchmarks
pull_and_analyze

echo "Operator microbench results: $LOCAL_RUN_DIR"
