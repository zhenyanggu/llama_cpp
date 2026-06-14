#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_semi.sh [options]

Run the self-contained AICAS semi-final evaluation on the KV260 image.

Options:
  --release-root <path>     Release payload root (default: /home/ubuntu/aicas-semi-release)
  --run-id <id>             Result directory name (default: UTC timestamp)
  --output-dir <path>       Results root (default: <release-root>/results)
  --port <port>             llama-server port (default: 8080)
  --threads <n>             llama-server threads (default: 4)
  --sudo-password <pass>    Board sudo password (default: 123456)
  --power-path <path>       hwmon power input path (default: /sys/class/hwmon/hwmon2/power1_input)
  --sample-hz <hz>          Energy sample rate (default: 100)
  --throughput-max-tokens <n> Throughput output token budget (default: 1024)
  --energy-max-tokens <n>   Energy output token budget (default: 1024)
  --sample-json <path>      Reuse an existing sampled accuracy JSON
  --acc-sample-json <path>  Accuracy sample JSON to test
  --acc-ori <ratio>         Forwarded to score_submission.py
  --skip-acc                Skip accuracy and produce throughput/energy/ttft only
  --fast-acc                Run only one accuracy sample, then continue other tests
  --cpu-mode                Run pure CPU fp16 baseline
  --cpu-model <path>        CPU text model GGUF (default: <release-root>/models/text-f16.gguf)
  --cpu-mmproj <path>       CPU mmproj GGUF (default: <release-root>/models/mmproj-f16.gguf)
  --quick-smoke             Run hardware init + 1-token request only
  -h, --help                Show this help
USAGE
}

RELEASE_ROOT="/home/ubuntu/aicas-semi-release"
RUN_ID=""
OUTPUT_ROOT=""
PORT="8080"
THREADS="4"
SUDO_PASSWORD="123456"
POWER_PATH="/sys/class/hwmon/hwmon2/power1_input"
SAMPLE_HZ="100"
THROUGHPUT_MAX_TOKENS="1024"
ENERGY_MAX_TOKENS="1024"
SAMPLE_JSON=""
ACC_ORI=""
RUN_ACC=1
FAST_ACC=0
CPU_MODE=0
CPU_MODEL=""
CPU_MMPROJ=""
QUICK_SMOKE=0

while [ $# -gt 0 ]; do
  case "$1" in
    --release-root) RELEASE_ROOT="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --output-dir) OUTPUT_ROOT="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --sudo-password) SUDO_PASSWORD="$2"; shift 2 ;;
    --power-path) POWER_PATH="$2"; shift 2 ;;
    --sample-hz) SAMPLE_HZ="$2"; shift 2 ;;
    --throughput-max-tokens) THROUGHPUT_MAX_TOKENS="$2"; shift 2 ;;
    --energy-max-tokens) ENERGY_MAX_TOKENS="$2"; shift 2 ;;
    --sample-json|--acc-sample-json) SAMPLE_JSON="$2"; shift 2 ;;
    --acc-ori) ACC_ORI="$2"; shift 2 ;;
    --skip-acc) RUN_ACC=0; shift ;;
    --fast-acc) FAST_ACC=1; shift ;;
    --cpu-mode) CPU_MODE=1; shift ;;
    --cpu-model) CPU_MODEL="$2"; shift 2 ;;
    --cpu-mmproj) CPU_MMPROJ="$2"; shift 2 ;;
    --quick-smoke) QUICK_SMOKE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

CALLER_CWD="$(pwd)"
RELEASE_ROOT="$(cd "$RELEASE_ROOT" && pwd)"
if [ -z "$RUN_ID" ]; then
  RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-semi-board"
fi
if [ -z "$OUTPUT_ROOT" ]; then
  OUTPUT_ROOT="$RELEASE_ROOT/results"
fi
if [ -z "$CPU_MODEL" ]; then
  CPU_MODEL="$RELEASE_ROOT/models/text-f16.gguf"
fi
if [ -z "$CPU_MMPROJ" ]; then
  CPU_MMPROJ="$RELEASE_ROOT/models/mmproj-f16.gguf"
fi
case "$CPU_MODEL" in
  /*) ;;
  *) CPU_MODEL="$CALLER_CWD/$CPU_MODEL" ;;
esac
case "$CPU_MMPROJ" in
  /*) ;;
  *) CPU_MMPROJ="$CALLER_CWD/$CPU_MMPROJ" ;;
esac

BIN="$RELEASE_ROOT/bin/llama-server"
CODE_DIR="$RELEASE_ROOT/code"
DATA_DIR="$RELEASE_ROOT/data"
MODEL="$RELEASE_ROOT/models/text.gguf"
MMPROJ="$RELEASE_ROOT/models/mmproj.gguf"
if [ "$CPU_MODE" -eq 1 ]; then
  MODEL="$CPU_MODEL"
  MMPROJ="$CPU_MMPROJ"
fi
LIB_DIR="$RELEASE_ROOT/lib"
DRIVER_KO="$RELEASE_ROOT/board/driver/npu_kv260.ko"
PREFILL_APP="prefill_190m_qkpipe_20260608_0206_app"
DECODE_APP="dec_200m_latest_0609a_app"
PREFILL_FW="xilinx/$PREFILL_APP/prefill_190m_qkpipe_20260608_0206.bit.bin"
DECODE_FW="xilinx/$DECODE_APP/dec_200m_latest_0609a.bit.bin"
RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
LOG_DIR="$RUN_DIR/logs"
RESULT_DIR="$RUN_DIR/results"
SWITCH_SCRIPT="$RUN_DIR/switch_npu_overlay.sh"
SERVER_PID=""

require_file() {
  local path="$1"
  local label="$2"
  [ -f "$path" ] || { echo "Missing $label: $path" >&2; exit 1; }
}

sudo_sh() {
  printf '%s\n' "$SUDO_PASSWORD" | sudo -S "$@"
}

require_file "$BIN" "llama-server"
require_file "$MODEL" "text model"
require_file "$MMPROJ" "mmproj"
if [ "$CPU_MODE" -ne 1 ]; then
  require_file "$DRIVER_KO" "npu_kv260.ko"
  require_file "$RELEASE_ROOT/config/env.sh" "runtime env"
fi
require_file "$CODE_DIR/run_semi_eval.sh" "semi eval runner"
require_file "$CODE_DIR/test2.jpg" "throughput image"
require_file "$CODE_DIR/ttft_config.board.json" "board TTFT config"
require_file "$DATA_DIR/FullTest.json" "FullTest.json"
if [ -n "$SAMPLE_JSON" ]; then
  case "$SAMPLE_JSON" in
    /*) ;;
    *) SAMPLE_JSON="$CALLER_CWD/$SAMPLE_JSON" ;;
  esac
  require_file "$SAMPLE_JSON" "accuracy sample JSON"
fi

mkdir -p "$LOG_DIR" "$RESULT_DIR"

log_stage() {
  printf '\n[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

cleanup() {
  if [ -n "${SERVER_PID:-}" ]; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" >/dev/null 2>&1 || true
  fi
  local pids
  pids="$(pgrep -f "llama-server --host 127\\.0\\.0\\.1 --port $PORT" 2>/dev/null || true)"
  if [ -n "$pids" ]; then
    sudo_sh kill $pids >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

wait_ready() {
  local ready=0
  for _ in $(seq 1 120); do
    if python3 - <<PY
import json
import urllib.request
with urllib.request.urlopen("http://127.0.0.1:$PORT/v1/models", timeout=5) as resp:
    payload = json.load(resp)
assert any(item.get("id") == "smolvlm2-gguf" for item in payload.get("data", [])), payload
PY
    then
      ready=1
      break
    fi
    sleep 2
  done
  [ "$ready" -eq 1 ]
}

write_switch_script() {
  cat > "$SWITCH_SCRIPT" <<'EOS'
#!/usr/bin/env bash
set -euo pipefail
target_app="$1"
state_file="/tmp/aicas_npu_overlay_state"
prefill_app="${AICAS_PREFILL_APP:?}"
decode_app="${AICAS_DECODE_APP:?}"
prefill_fw="${AICAS_PREFILL_FW:?}"
decode_fw="${AICAS_DECODE_FW:?}"
sudo_password="${SUDO_PASSWORD:?}"

current=""
if [ -f "$state_file" ]; then
  current="$(cat "$state_file" 2>/dev/null || true)"
fi
if [ "$current" = "$target_app" ] && [ -e /dev/npu_kv260 ]; then
  echo "NPU overlay already active: $target_app"
  exit 0
fi

case "$target_app" in
  "$prefill_app") firmware="$prefill_fw" ;;
  "$decode_app") firmware="$decode_fw" ;;
  *) echo "Unknown NPU overlay target: $target_app" >&2; exit 1 ;;
esac

echo "Switching NPU bitstream: ${current:-none} -> $target_app ($firmware)"
printf '%s\n' "$sudo_password" | sudo -S sh -c "echo 0 > /sys/class/fpga_manager/fpga0/flags"
printf '%s\n' "$sudo_password" | sudo -S sh -c "echo '$firmware' > /sys/class/fpga_manager/fpga0/firmware"
for _ in $(seq 1 60); do
  state="$(cat /sys/class/fpga_manager/fpga0/state 2>/dev/null || true)"
  [ "$state" = "operating" ] && break
  sleep 0.1
done
state="$(cat /sys/class/fpga_manager/fpga0/state 2>/dev/null || true)"
if [ "$state" != "operating" ]; then
  echo "FPGA manager did not reach operating after loading $firmware; state=$state" >&2
  exit 1
fi
if [ ! -e /dev/npu_kv260 ]; then
  echo "NPU device missing after bitstream switch to $target_app" >&2
  exit 1
fi
printf '%s\n' "$sudo_password" | sudo -S chmod 666 /dev/npu_kv260
printf '%s\n' "$target_app" > "$state_file"
EOS
  chmod +x "$SWITCH_SCRIPT"
}

init_npu() {
  log_stage "Initializing NPU overlay and driver"
  sudo_sh rmmod npu_kv260 >/dev/null 2>&1 || true
  sudo_sh xmutil unloadapp >/dev/null 2>&1 || true
  sudo_sh xmutil loadapp "$PREFILL_APP"
  if ! lsmod | grep -q '^npu_kv260 '; then
    sudo_sh insmod "$DRIVER_KO" max_buffer_mb=1500
  fi

  if [ ! -e /dev/npu_kv260 ]; then
    for dev in /sys/bus/platform/devices/*.npu_generic /sys/bus/platform/devices/*Versa*; do
      [ -e "$dev" ] || continue
      dev_name="$(basename "$dev")"
      sudo_sh sh -c "echo npu_kv260 > '$dev/driver_override'" >/dev/null 2>&1 || true
      sudo_sh sh -c "echo '$dev_name' > /sys/bus/platform/drivers/npu_kv260/bind" >/dev/null 2>&1 || true
      [ -e /dev/npu_kv260 ] && break
    done
  fi
  for _ in $(seq 1 30); do
    [ -e /dev/npu_kv260 ] && break
    sleep 1
  done
  [ -e /dev/npu_kv260 ] || { echo "NPU device did not appear after loading $PREFILL_APP" >&2; exit 1; }
  sudo_sh chmod 666 /dev/npu_kv260
  printf '%s\n' "$PREFILL_APP" > /tmp/aicas_npu_overlay_state
  write_switch_script
  log_stage "NPU ready: $PREFILL_APP"
}

start_server() {
  cleanup
  if [ "$CPU_MODE" -eq 1 ]; then
    log_stage "Starting llama-server in CPU fp16 baseline mode on port $PORT"
    export MTMD_BACKEND_DEVICE=CPU
    export LLAMA_MTMD_MERGE_PREFILL=0
    export GGML_NPU_TEXT_PREFILL_DYNAMIC=0
    export GGML_NPU_PRELOAD_WEIGHTS_ON_LOAD=0
    export AICAS_TEXT_PREFILL_LOG8PV_NPU=0
    export AICAS_MMPROJ_LOG8PV_NPU=0
    export AICAS_TEXT_DECODE_AWQ_NPU=0
    export AICAS_TEXT_DECODE_AWQ_NPU_REQUIRE_ACTIVE=0
    export AICAS_TEXT_DECODE_AWQ_FUSED_FFN_NPU=0
    export AICAS_TEXT_DECODE_ATTN_NPU=0
    export AICAS_TEXT_LM_HEAD_W8A16_NPU=0
    unset AICAS_NPU_PREFILL_SWITCH_CMD
    unset AICAS_NPU_DECODE_SWITCH_CMD
  else
    log_stage "Starting llama-server on port $PORT"
    export RELEASE_ROOT
    # shellcheck disable=SC1091
    source "$RELEASE_ROOT/config/env.sh"

    export MTMD_BACKEND_DEVICE=NPU
    export LLAMA_MTMD_MERGE_PREFILL=1
    export LLAMA_MTMD_MERGE_PREFILL_TRACE=0
    export GGML_NPU_TEXT_PREFILL_DYNAMIC=1
    export GGML_NPU_EAGER_INIT=1
    export VERSA_P_CMA_SIZE=1408M
    export VERSA_P_CMA_HEAP_OFFSET=0
    export VERSA_P_CMA_HEAP_SIZE=832M
    export NPU_CMA_SIZE=1408M
    export NPU_CMA_HEAP_OFFSET=832M
    export NPU_CMA_HEAP_SIZE=576M
    export AICAS_TEXT_DECODE_AWQ_NPU=1
    export AICAS_TEXT_DECODE_AWQ_NPU_GEMV_ONLY=0
    export AICAS_TEXT_DECODE_AWQ_FUSED_FFN_NPU=1
    export AICAS_TEXT_DECODE_ATTN_NPU=1
    export AICAS_TEXT_DECODE_AWQ_NPU_REQUIRE_ACTIVE=1
    export AICAS_TEXT_DECODE_ATTN_NPU_CPU_ROPE=0
    export AICAS_TEXT_DECODE_ATTN_NPU_HW_KV_QUANT=1
    export AICAS_TEXT_DECODE_ATTN_NPU_PV_SCALE_BOOST_LOG2=0
    export AICAS_TEXT_DECODE_ATTN_NPU_PV_SCALE_BOOST_FOLD_O=0
    export AICAS_TEXT_DECODE_AWQ_NPU_PRELOAD_CMA=1
    export AICAS_TEXT_DECODE_AWQ_PINGPONG=1
    export AICAS_TEXT_DECODE_AWQ_HW_F32_MVOUT=1
    export GGML_NPU_PRELOAD_WEIGHTS_ON_LOAD=1
    export AICAS_TEXT_DECODE_AWQ_NPU_PRELOAD=1
    export AICAS_PREFILL_APP="$PREFILL_APP"
    export AICAS_DECODE_APP="$DECODE_APP"
    export AICAS_PREFILL_FW="$PREFILL_FW"
    export AICAS_DECODE_FW="$DECODE_FW"
    export SUDO_PASSWORD
    export AICAS_NPU_PREFILL_SWITCH_CMD="SUDO_PASSWORD='$SUDO_PASSWORD' AICAS_PREFILL_APP='$PREFILL_APP' AICAS_DECODE_APP='$DECODE_APP' AICAS_PREFILL_FW='$PREFILL_FW' AICAS_DECODE_FW='$DECODE_FW' '$SWITCH_SCRIPT' '$PREFILL_APP'"
    export AICAS_NPU_DECODE_SWITCH_CMD="SUDO_PASSWORD='$SUDO_PASSWORD' AICAS_PREFILL_APP='$PREFILL_APP' AICAS_DECODE_APP='$DECODE_APP' AICAS_PREFILL_FW='$PREFILL_FW' AICAS_DECODE_FW='$DECODE_FW' '$SWITCH_SCRIPT' '$DECODE_APP'"
  fi

  server_args=(
    --host 127.0.0.1
    --port "$PORT"
    --alias smolvlm2-gguf
    -m "$MODEL"
    --mmproj "$MMPROJ"
    -t "$THREADS"
    --ubatch-size 1024
    --log-disable
    --no-warmup
  )
  if [ "$CPU_MODE" -eq 1 ]; then
    server_args+=(--cache-type-k f16 --cache-type-v f16 --flash-attn auto)
  else
    server_args+=(--cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on)
  fi
  cd "$RUN_DIR"
  env LD_LIBRARY_PATH="$LIB_DIR:${LD_LIBRARY_PATH:-}" \
    "$BIN" "${server_args[@]}" \
      > "$LOG_DIR/server.log" 2>&1 &
  SERVER_PID=$!

  if ! wait_ready; then
    tail -n 160 "$LOG_DIR/server.log" >&2 || true
    echo "llama-server did not become ready" >&2
    exit 1
  fi
  log_stage "llama-server ready: http://127.0.0.1:$PORT/v1"
}

if [ "$CPU_MODE" -eq 1 ]; then
  log_stage "CPU mode enabled: skipping NPU overlay and driver init"
else
  init_npu
fi
start_server

if [ "$QUICK_SMOKE" -eq 1 ]; then
  log_stage "Starting quick smoke request"
  python3 "$CODE_DIR/throughput_eval.py" \
    -i "$CODE_DIR/test2.jpg" \
    -o "$RESULT_DIR/quick_smoke.json" \
    --base-url "http://127.0.0.1:$PORT/v1" \
    --model smolvlm2-gguf \
    --max-tokens 1 \
    --ignore-eos \
    --no-cache-prompt \
    --request-timeout 600
  log_stage "Finished quick smoke request: $RESULT_DIR/quick_smoke.json"
  echo "Quick smoke results: $RESULT_DIR"
  exit 0
fi

runner_args=(
  --code-dir "$CODE_DIR"
  --full-test-json "$DATA_DIR/FullTest.json"
  --image-root "$DATA_DIR/images"
  --throughput-image "$CODE_DIR/test2.jpg"
  --ttft-config "$CODE_DIR/ttft_config.board.json"
  --output-dir "$RESULT_DIR"
  --base-url "http://127.0.0.1:$PORT/v1"
  --model smolvlm2-gguf
  --power-path "$POWER_PATH"
  --sample-hz "$SAMPLE_HZ"
  --throughput-max-tokens "$THROUGHPUT_MAX_TOKENS"
  --energy-max-tokens "$ENERGY_MAX_TOKENS"
  --acc-max-tokens 100
  --no-ttft-cache-prompt
)
if [ "$RUN_ACC" -eq 1 ]; then
  runner_args+=(--run-acc)
else
  runner_args+=(--skip-acc)
fi
if [ -n "$SAMPLE_JSON" ]; then
  runner_args+=(--sample-json "$SAMPLE_JSON")
fi
if [ "$FAST_ACC" -eq 1 ]; then
  runner_args+=(--fast-acc)
fi
if [ -n "$ACC_ORI" ]; then
  runner_args+=(--acc-ori "$ACC_ORI")
fi

log_stage "Starting full semi evaluation: run_id=$RUN_ID"
bash "$CODE_DIR/run_semi_eval.sh" "${runner_args[@]}" | tee "$LOG_DIR/eval.log"
log_stage "Finished full semi evaluation: $RESULT_DIR"

cat > "$RUN_DIR/run_meta.json" <<EOF
{
  "run_id": "$RUN_ID",
  "release_root": "$RELEASE_ROOT",
  "result_dir": "$RESULT_DIR",
  "port": $PORT,
  "threads": $THREADS,
  "mode": "$([ "$CPU_MODE" -eq 1 ] && printf cpu-fp16 || printf npu)",
  "model": "$MODEL",
  "mmproj": "$MMPROJ",
  "power_path": "$POWER_PATH",
  "sample_hz": "$SAMPLE_HZ",
  "throughput_max_tokens": $THROUGHPUT_MAX_TOKENS,
  "energy_max_tokens": $ENERGY_MAX_TOKENS,
  "fast_acc": $FAST_ACC
}
EOF

echo "Results: $RESULT_DIR"
