#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_v1_submission.sh [options]

Run the official V1 submission flow for:
  Q8_0 text model + W8A8 mixed_v1 mmproj on KV260.

Options:
  --host <host>              KV260 host.
  --user <user>              KV260 username.
  --remote-root <path>       Remote root directory.
  --sdk-env <path>           KV260 SDK env script.
  --build-dir <path>         Cross-build directory.
  --run-id <id>              Override the run id.
  --threads <n>              llama-server thread count.
  --port <port>              llama-server port.
  --skip-build               Reuse the existing build output.
  --skip-board-init          Assume the board was initialized already.
  --sudo-password-env <var>  Env var containing the board sudo password.
  -h, --help                 Show this help.
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
V1_DIR="$ROOT_DIR/AICAS2026/V1"
PAYLOAD_DIR="$V1_DIR/payload"
MANIFEST_DIR="$V1_DIR/manifests"
RESULTS_ROOT="$V1_DIR/results/official"
PREPARE_SCRIPT="$V1_DIR/scripts/prepare_v1_bundle.sh"
BOARD_INIT_SCRIPT="$V1_DIR/scripts/board_init_npu.sh"
SUMMARY_SCRIPT="$V1_DIR/scripts/summarize_results.py"

HOST="192.168.0.10"
USER_NAME="ubuntu"
REMOTE_ROOT="/home/ubuntu/aicas"
SDK_ENV="/home/gugugu/petalinux/sdk/kv260-2025.1/environment-setup-cortexa72-cortexa53-amd-linux"
BUILD_DIR="$ROOT_DIR/build-kv260-npu"
RUN_ID=""
THREADS="4"
PORT="8100"
SKIP_BUILD=0
SKIP_BOARD_INIT=0
SUDO_PASSWORD_ENV="BOARD_SUDO_PASSWORD"
MODEL_ALIAS="smolvlm2-gguf-npu-q8"

NPU_SPM_BYTES="${GGML_NPU_SPM_BYTES:-524288}"
NPU_ACC_BYTES="${GGML_NPU_ACC_BYTES:-524288}"
NPU_GUARD_BYTES="${GGML_NPU_GUARD_BYTES:-4096}"
NPU_STAGE2_K_BYTES="${GGML_NPU_STAGE2_K_BYTES:-4096}"
NPU_CMA_BYTES="${NPU_CMA_SIZE:-256M}"

SSH_OPTS=(
  -o BatchMode=yes
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
  -o LogLevel=ERROR
)

while [ $# -gt 0 ]; do
  case "$1" in
    --host)
      HOST="$2"
      shift 2
      ;;
    --user)
      USER_NAME="$2"
      shift 2
      ;;
    --remote-root)
      REMOTE_ROOT="$2"
      shift 2
      ;;
    --sdk-env)
      SDK_ENV="$2"
      shift 2
      ;;
    --build-dir)
      BUILD_DIR="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --threads)
      THREADS="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --skip-build)
      SKIP_BUILD=1
      shift
      ;;
    --skip-board-init)
      SKIP_BOARD_INIT=1
      shift
      ;;
    --sudo-password-env)
      SUDO_PASSWORD_ENV="$2"
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

TARGET="${USER_NAME}@${HOST}"
REMOTE_SHARED_DIR="$REMOTE_ROOT/shared"
REMOTE_GGUF_DIR="$REMOTE_SHARED_DIR/gguf"
REMOTE_EVAL_DIR="$REMOTE_SHARED_DIR/eval"
REMOTE_DATA_DIR="$REMOTE_SHARED_DIR/data"
REMOTE_LIB_DIR="$REMOTE_SHARED_DIR/lib"
REMOTE_BOARD_SUPPORT_DIR="$REMOTE_ROOT/board_support"
REMOTE_BOARD_MYNPU_DIR="$REMOTE_BOARD_SUPPORT_DIR/mynpu"
REMOTE_BOARD_DRIVER_DIR="$REMOTE_BOARD_SUPPORT_DIR/driver"
REMOTE_SCRIPTS_DIR="$REMOTE_ROOT/scripts"
REMOTE_RUNS_DIR="$REMOTE_ROOT/runs/submission-q8-w8a8"

remote_file_size() {
  ssh "${SSH_OPTS[@]}" "$TARGET" "stat -c %s '$1' 2>/dev/null || true"
}

sync_file_if_needed() {
  local src="$1"
  local dst="$2"
  local label="$3"
  local local_size remote_size remote_dir

  local_size="$(stat -c %s "$src")"
  remote_size="$(remote_file_size "$dst")"

  if [ -n "$remote_size" ] && [ "$remote_size" = "$local_size" ]; then
    echo "Reusing $label at $dst"
    return 0
  fi

  remote_dir="$(dirname "$dst")"
  ssh "${SSH_OPTS[@]}" "$TARGET" "mkdir -p '$remote_dir'"
  scp "${SSH_OPTS[@]}" "$src" "$TARGET:$dst"
}

sync_required_data() {
  local manifest="$MANIFEST_DIR/required_data_files.txt"
  ssh "${SSH_OPTS[@]}" "$TARGET" "mkdir -p '$REMOTE_DATA_DIR'"
  tar -C "$PAYLOAD_DIR/data/images" -cf - -T "$manifest" | \
    ssh "${SSH_OPTS[@]}" "$TARGET" "cd '$REMOTE_DATA_DIR' && tar -xmf -"
}

write_run_meta() {
  local local_run_dir="$1"
  local remote_run_dir="$2"
  local status="$3"
  local remote_exit_code="$4"

  RUN_META_RUN_ID="$RUN_ID" \
  RUN_META_STATUS="$status" \
  RUN_META_REMOTE_EXIT_CODE="$remote_exit_code" \
  RUN_META_BUILD_DIR="$BUILD_DIR" \
  RUN_META_SDK_ENV="$SDK_ENV" \
  RUN_META_MODEL="$PAYLOAD_DIR/gguf/SmolVLM2-500M-Video-Instruct-Q8_0.gguf" \
  RUN_META_MMPROJ="$PAYLOAD_DIR/gguf/mmproj-SmolVLM2-500M-Video-Instruct-w8a8-mixed-v1.gguf" \
  RUN_META_HOST="$HOST" \
  RUN_META_USER="$USER_NAME" \
  RUN_META_REMOTE_ROOT="$REMOTE_ROOT" \
  RUN_META_REMOTE_RUN_DIR="$remote_run_dir" \
  RUN_META_THREADS="$THREADS" \
  RUN_META_PORT="$PORT" \
  RUN_META_MODEL_ALIAS="$MODEL_ALIAS" \
  RUN_META_NPU_SPM_BYTES="$NPU_SPM_BYTES" \
  RUN_META_NPU_ACC_BYTES="$NPU_ACC_BYTES" \
  RUN_META_NPU_GUARD_BYTES="$NPU_GUARD_BYTES" \
  RUN_META_NPU_STAGE2_K_BYTES="$NPU_STAGE2_K_BYTES" \
  RUN_META_NPU_CMA_BYTES="$NPU_CMA_BYTES" \
  python3 - "$local_run_dir/run_meta.json" <<'PY'
import json
import os
import subprocess
import sys

git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
git_status = subprocess.check_output(["git", "status", "--short"], text=True).splitlines()

payload = {
    "run_id": os.environ["RUN_META_RUN_ID"],
    "run_type": "submission-q8_0-w8a8mmproj",
    "run_status": os.environ["RUN_META_STATUS"],
    "remote_exit_code": int(os.environ["RUN_META_REMOTE_EXIT_CODE"]) if os.environ["RUN_META_REMOTE_EXIT_CODE"] else None,
    "git_head": git_head,
    "git_status_short": git_status,
    "sdk_env": os.environ["RUN_META_SDK_ENV"],
    "build_dir": os.environ["RUN_META_BUILD_DIR"],
    "model": os.environ["RUN_META_MODEL"],
    "mmproj": os.environ["RUN_META_MMPROJ"],
    "remote_host": os.environ["RUN_META_HOST"],
    "remote_user": os.environ["RUN_META_USER"],
    "remote_root": os.environ["RUN_META_REMOTE_ROOT"],
    "remote_run_dir": os.environ["RUN_META_REMOTE_RUN_DIR"],
    "threads": int(os.environ["RUN_META_THREADS"]),
    "port": int(os.environ["RUN_META_PORT"]),
    "model_alias": os.environ["RUN_META_MODEL_ALIAS"],
    "runtime_env": {
        "MTMD_BACKEND_DEVICE": "NPU",
        "GGML_NPU_SPM_BYTES": os.environ["RUN_META_NPU_SPM_BYTES"],
        "GGML_NPU_ACC_BYTES": os.environ["RUN_META_NPU_ACC_BYTES"],
        "GGML_NPU_GUARD_BYTES": os.environ["RUN_META_NPU_GUARD_BYTES"],
        "GGML_NPU_STAGE2_K_BYTES": os.environ["RUN_META_NPU_STAGE2_K_BYTES"],
        "NPU_CMA_SIZE": os.environ["RUN_META_NPU_CMA_BYTES"],
    },
}

with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")
PY
}

run_remote_eval() {
  local remote_run_dir="$1"
  ssh "${SSH_OPTS[@]}" "$TARGET" "bash -s" <<EOF
set -euo pipefail

RUN_DIR='$remote_run_dir'
REMOTE_MODEL='$REMOTE_GGUF_DIR/SmolVLM2-500M-Video-Instruct-Q8_0.gguf'
REMOTE_MMPROJ='$REMOTE_GGUF_DIR/mmproj-SmolVLM2-500M-Video-Instruct-w8a8-mixed-v1.gguf'
REMOTE_EVAL_DIR='$REMOTE_EVAL_DIR'
REMOTE_DATA_DIR='$REMOTE_DATA_DIR'
REMOTE_LIB_DIR='$REMOTE_LIB_DIR'
PORT='$PORT'
THREADS='$THREADS'
MODEL_ALIAS='$MODEL_ALIAS'
NPU_SPM_BYTES='$NPU_SPM_BYTES'
NPU_ACC_BYTES='$NPU_ACC_BYTES'
NPU_GUARD_BYTES='$NPU_GUARD_BYTES'
NPU_STAGE2_K_BYTES='$NPU_STAGE2_K_BYTES'
NPU_CMA_BYTES='$NPU_CMA_BYTES'

cleanup() {
  if [ -n "\${SERVER_PID:-}" ]; then
    kill "\$SERVER_PID" >/dev/null 2>&1 || true
    wait "\$SERVER_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

cd "\$RUN_DIR"
rm -f server.log throughput_metrics.json acc_100sample.json

env \
  LD_LIBRARY_PATH="\$REMOTE_LIB_DIR:\${LD_LIBRARY_PATH:-}" \
  MTMD_BACKEND_DEVICE=NPU \
  GGML_NPU_SPM_BYTES="\$NPU_SPM_BYTES" \
  GGML_NPU_ACC_BYTES="\$NPU_ACC_BYTES" \
  GGML_NPU_GUARD_BYTES="\$NPU_GUARD_BYTES" \
  GGML_NPU_STAGE2_K_BYTES="\$NPU_STAGE2_K_BYTES" \
  NPU_CMA_SIZE="\$NPU_CMA_BYTES" \
  AICAS_MMPROJ_W8A8_DEBUG=1 \
  ./llama-server \
    --no-warmup \
    --host 127.0.0.1 \
    --port "\$PORT" \
    --alias "\$MODEL_ALIAS" \
    -m "\$REMOTE_MODEL" \
    --mmproj "\$REMOTE_MMPROJ" \
    -t "\$THREADS" \
    > server.log 2>&1 &
SERVER_PID=\$!

READY=0
for _ in \$(seq 1 90); do
  if python3 - <<'PY'
import json
import urllib.request

with urllib.request.urlopen('http://127.0.0.1:$PORT/v1/models', timeout=5) as resp:
    payload = json.load(resp)

assert any(item.get('id') == '$MODEL_ALIAS' for item in payload.get('data', [])), payload
PY
  then
    READY=1
    break
  fi
  sleep 2
done

if [ "\$READY" -ne 1 ]; then
  tail -n 120 server.log >&2 || true
  echo "llama-server did not become ready" >&2
  exit 1
fi

python3 "\$REMOTE_EVAL_DIR/throughput_eval.py" \
  -i "\$REMOTE_DATA_DIR/IIIT5K/test/2543_2.png" \
  -o "\$RUN_DIR/throughput_metrics.json" \
  --base-url "http://127.0.0.1:\$PORT/v1" \
  --model "\$MODEL_ALIAS"

python3 "\$REMOTE_EVAL_DIR/acc_eval.py" \
  --image_folder "\$REMOTE_DATA_DIR" \
  --OCRBench_file "\$REMOTE_EVAL_DIR/sampled_100.json" \
  --output_folder "\$RUN_DIR" \
  --save_name acc_100sample \
  --base-url "http://127.0.0.1:\$PORT/v1" \
  --model "\$MODEL_ALIAS" \
  --request-timeout 600 \
  --progress-every 10
EOF
}

prepare_args=(--sdk-env "$SDK_ENV" --build-dir "$BUILD_DIR")
if [ "$SKIP_BUILD" -eq 1 ]; then
  prepare_args+=(--skip-build)
fi
bash "$PREPARE_SCRIPT" "${prepare_args[@]}"

if [ -z "$RUN_ID" ]; then
  RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-q8_0-w8a8mmproj"
fi

LOCAL_RUN_DIR="$RESULTS_ROOT/$RUN_ID"
REMOTE_RUN_DIR="$REMOTE_RUNS_DIR/$RUN_ID"
mkdir -p "$LOCAL_RUN_DIR"

ssh "${SSH_OPTS[@]}" "$TARGET" "mkdir -p '$REMOTE_ROOT' '$REMOTE_SHARED_DIR' '$REMOTE_GGUF_DIR' '$REMOTE_EVAL_DIR' '$REMOTE_DATA_DIR' '$REMOTE_LIB_DIR' '$REMOTE_BOARD_MYNPU_DIR' '$REMOTE_BOARD_DRIVER_DIR' '$REMOTE_SCRIPTS_DIR' '$REMOTE_RUN_DIR'"

sync_file_if_needed "$PAYLOAD_DIR/gguf/SmolVLM2-500M-Video-Instruct-Q8_0.gguf" "$REMOTE_GGUF_DIR/SmolVLM2-500M-Video-Instruct-Q8_0.gguf" "Q8_0 model"
sync_file_if_needed "$PAYLOAD_DIR/gguf/mmproj-SmolVLM2-500M-Video-Instruct-w8a8-mixed-v1.gguf" "$REMOTE_GGUF_DIR/mmproj-SmolVLM2-500M-Video-Instruct-w8a8-mixed-v1.gguf" "W8A8 mmproj"
sync_file_if_needed "$PAYLOAD_DIR/eval/throughput_eval.py" "$REMOTE_EVAL_DIR/throughput_eval.py" "throughput_eval.py"
sync_file_if_needed "$PAYLOAD_DIR/eval/acc_eval.py" "$REMOTE_EVAL_DIR/acc_eval.py" "acc_eval.py"
sync_file_if_needed "$PAYLOAD_DIR/eval/llama_server_client.py" "$REMOTE_EVAL_DIR/llama_server_client.py" "llama_server_client.py"
sync_file_if_needed "$PAYLOAD_DIR/data/sampled_100.json" "$REMOTE_EVAL_DIR/sampled_100.json" "sampled_100.json"
sync_file_if_needed "$PAYLOAD_DIR/board_support/driver/npu_kv260.ko" "$REMOTE_BOARD_DRIVER_DIR/npu_kv260.ko" "npu_kv260.ko"
sync_file_if_needed "$PAYLOAD_DIR/board_support/mynpu/KV260_410.bin" "$REMOTE_BOARD_MYNPU_DIR/KV260_410.bin" "KV260_410.bin"
sync_file_if_needed "$PAYLOAD_DIR/board_support/mynpu/KV260_410.dtbo" "$REMOTE_BOARD_MYNPU_DIR/KV260_410.dtbo" "KV260_410.dtbo"
sync_file_if_needed "$PAYLOAD_DIR/board_support/mynpu/shell.json" "$REMOTE_BOARD_MYNPU_DIR/shell.json" "mynpu shell.json"
sync_file_if_needed "$PAYLOAD_DIR/bin/llama-server-kv260-npu" "$REMOTE_RUN_DIR/llama-server" "llama-server"
sync_file_if_needed "$BOARD_INIT_SCRIPT" "$REMOTE_SCRIPTS_DIR/board_init_npu.sh" "board_init_npu.sh"

for lib in "$PAYLOAD_DIR"/runtime_libs/*; do
  [ -e "$lib" ] || continue
  sync_file_if_needed "$lib" "$REMOTE_LIB_DIR/$(basename "$lib")" "runtime lib $(basename "$lib")"
done

sync_required_data
ssh "${SSH_OPTS[@]}" "$TARGET" "chmod +x '$REMOTE_RUN_DIR/llama-server' '$REMOTE_SCRIPTS_DIR/board_init_npu.sh'"

if [ "$SKIP_BOARD_INIT" -ne 1 ]; then
  password_value="${!SUDO_PASSWORD_ENV:-}"
  [ -n "$password_value" ] || {
    echo "Missing board sudo password in env var: $SUDO_PASSWORD_ENV" >&2
    exit 1
  }
  password_escaped="$(printf '%q' "$password_value")"
  ssh "${SSH_OPTS[@]}" "$TARGET" "bash -lc 'BOARD_SUDO_PASSWORD=$password_escaped \"$REMOTE_SCRIPTS_DIR/board_init_npu.sh\" --remote-root \"$REMOTE_ROOT\" --output \"$REMOTE_RUN_DIR/board_readiness.txt\"'"
fi

REMOTE_RC=0
if run_remote_eval "$REMOTE_RUN_DIR"; then
  REMOTE_RC=0
else
  REMOTE_RC=$?
fi

scp "${SSH_OPTS[@]}" "$TARGET:$REMOTE_RUN_DIR/throughput_metrics.json" "$LOCAL_RUN_DIR/" >/dev/null 2>&1 || true
scp "${SSH_OPTS[@]}" "$TARGET:$REMOTE_RUN_DIR/acc_100sample.json" "$LOCAL_RUN_DIR/" >/dev/null 2>&1 || true
scp "${SSH_OPTS[@]}" "$TARGET:$REMOTE_RUN_DIR/server.log" "$LOCAL_RUN_DIR/" >/dev/null 2>&1 || true
scp "${SSH_OPTS[@]}" "$TARGET:$REMOTE_RUN_DIR/board_readiness.txt" "$LOCAL_RUN_DIR/" >/dev/null 2>&1 || true

run_status="completed"
if [ "$REMOTE_RC" -ne 0 ]; then
  run_status="failed"
fi

write_run_meta "$LOCAL_RUN_DIR" "$REMOTE_RUN_DIR" "$run_status" "$REMOTE_RC"
scp "${SSH_OPTS[@]}" "$LOCAL_RUN_DIR/run_meta.json" "$TARGET:$REMOTE_RUN_DIR/run_meta.json" >/dev/null 2>&1 || true

if [ -f "$LOCAL_RUN_DIR/throughput_metrics.json" ] && [ -f "$LOCAL_RUN_DIR/acc_100sample.json" ]; then
  python3 "$SUMMARY_SCRIPT" \
    --throughput "$LOCAL_RUN_DIR/throughput_metrics.json" \
    --acc-json "$LOCAL_RUN_DIR/acc_100sample.json" \
    --run-meta "$LOCAL_RUN_DIR/run_meta.json" \
    --output-json "$LOCAL_RUN_DIR/summary_metrics.json" \
    --output-md "$LOCAL_RUN_DIR/RESULTS.md"
fi

bash "$PREPARE_SCRIPT" --skip-build --sdk-env "$SDK_ENV" --build-dir "$BUILD_DIR"

echo "Local results directory: $LOCAL_RUN_DIR"
echo "Remote run directory: $REMOTE_RUN_DIR"

if [ "$REMOTE_RC" -ne 0 ]; then
  echo "Remote evaluation failed. Check $LOCAL_RUN_DIR for partial artifacts." >&2
  exit "$REMOTE_RC"
fi
