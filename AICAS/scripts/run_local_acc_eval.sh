#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_local_acc_eval.sh [options]

Run AICAS/acc_eval.py on the local machine in one command:
1) prepare Python env (optional venv only)
2) start local llama-server
3) run accuracy eval

Options:
  --server-bin <path>      Path to llama-server binary.
  --model <f16|int8|path>  Model preset or custom model gguf path.
  --mmproj <path>          Path to mmproj gguf.
  --image-folder <path>    Image root folder used by acc_eval.py.
  --ocrbench-file <path>   OCRBench json input file.
  --output-folder <path>   Output directory for JSON + server.log.
  --save-name <name>       Output json basename (default: SmolVLM2).
  --port <port>            llama-server port (default: 8080).
  --threads <n>            llama-server thread count.
  --python <python>        Python interpreter for venv creation.
  --venv-dir <path>        Venv dir (default: AICAS/.venv-acc-eval).
  --request-timeout <sec>  acc_eval.py 单请求超时，默认 300。
  --request-retries <n>    acc_eval.py 请求重试次数，默认 2。
  --retry-delay <sec>      acc_eval.py 重试间隔，默认 2。
  --skip-venv              Use current Python directly.
  --no-install             Accepted for backward compatibility and ignored.
  -h, --help               Show this help.
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AICAS_DIR="$ROOT_DIR/AICAS"

SERVER_BIN=""
MODEL_F16="$AICAS_DIR/gguf/SmolVLM2-500M-Video-Instruct-f16.gguf"
MMPROJ_F16="$AICAS_DIR/gguf/mmproj-SmolVLM2-500M-Video-Instruct-f16.gguf"
MODEL_INT8="$AICAS_DIR/gguf/SmolVLM2-500M-Video-Instruct-Q8_0.gguf"
MMPROJ_INT8="$AICAS_DIR/gguf/mmproj-SmolVLM2-500M-Video-Instruct-Q8_0.gguf"

MODEL="$MODEL_F16"
MMPROJ="$MMPROJ_F16"
IMAGE_FOLDER="$AICAS_DIR/data"
OCRBENCH_FILE="$AICAS_DIR/sampled.json"
OUTPUT_FOLDER="$AICAS_DIR/output/local-acc-eval"
SAVE_NAME="SmolVLM2"
PORT="8080"
THREADS="$(nproc 2>/dev/null || echo 4)"
PYTHON_BIN="python3"
VENV_DIR="$AICAS_DIR/.venv-acc-eval"
MODEL_ALIAS="smolvlm2-gguf"
REQUEST_TIMEOUT="300"
REQUEST_RETRIES="2"
RETRY_DELAY="2"

SKIP_VENV=0
NO_INSTALL=0
SERVER_PID=""
EVAL_PYTHON=""
MMPROJ_EXPLICIT=0

while [ $# -gt 0 ]; do
  case "$1" in
    --server-bin)
      SERVER_BIN="$2"
      shift 2
      ;;
    --model)
      case "$2" in
        f16)
          MODEL="$MODEL_F16"
          if [ "$MMPROJ_EXPLICIT" -eq 0 ]; then
            MMPROJ="$MMPROJ_F16"
          fi
          ;;
        int8)
          MODEL="$MODEL_INT8"
          if [ "$MMPROJ_EXPLICIT" -eq 0 ]; then
            MMPROJ="$MMPROJ_INT8"
          fi
          ;;
        *)
          MODEL="$2"
          ;;
      esac
      shift 2
      ;;
    --mmproj)
      MMPROJ="$2"
      MMPROJ_EXPLICIT=1
      shift 2
      ;;
    --image-folder)
      IMAGE_FOLDER="$2"
      shift 2
      ;;
    --ocrbench-file)
      OCRBENCH_FILE="$2"
      shift 2
      ;;
    --output-folder)
      OUTPUT_FOLDER="$2"
      shift 2
      ;;
    --save-name)
      SAVE_NAME="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --threads)
      THREADS="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --venv-dir)
      VENV_DIR="$2"
      shift 2
      ;;
    --request-timeout)
      REQUEST_TIMEOUT="$2"
      shift 2
      ;;
    --request-retries)
      REQUEST_RETRIES="$2"
      shift 2
      ;;
    --retry-delay)
      RETRY_DELAY="$2"
      shift 2
      ;;
    --skip-venv)
      SKIP_VENV=1
      shift
      ;;
    --no-install)
      NO_INSTALL=1
      shift
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

require_path() {
  local path="$1"
  local name="$2"
  [ -e "$path" ] || { echo "Missing $name: $path" >&2; exit 1; }
}

ensure_exec() {
  local path="$1"
  if [ ! -x "$path" ]; then
    chmod +x "$path" 2>/dev/null || true
  fi
  [ -x "$path" ]
}

is_arch_compatible() {
  local bin="$1"
  local host_arch file_desc
  host_arch="$(uname -m)"
  file_desc="$(file "$bin" 2>/dev/null || true)"

  case "$host_arch" in
    x86_64)
      [[ "$file_desc" == *"x86-64"* ]]
      ;;
    aarch64|arm64)
      [[ "$file_desc" == *"ARM aarch64"* ]]
      ;;
    *)
      return 0
      ;;
  esac
}

pick_server_bin() {
  local candidate

  if [ -n "$SERVER_BIN" ]; then
    require_path "$SERVER_BIN" "llama-server"
    ensure_exec "$SERVER_BIN" || { echo "Not executable: $SERVER_BIN" >&2; exit 1; }
    is_arch_compatible "$SERVER_BIN" || {
      echo "Architecture mismatch for --server-bin: $SERVER_BIN" >&2
      echo "Host arch: $(uname -m), file: $(file "$SERVER_BIN")" >&2
      exit 1
    }
    return 0
  fi

  for candidate in \
    "$ROOT_DIR/build-host/bin/llama-server" \
    "$ROOT_DIR/build/bin/llama-server" \
    "$AICAS_DIR/bin/llama-server"; do
    if [ -e "$candidate" ] && ensure_exec "$candidate" && is_arch_compatible "$candidate"; then
      SERVER_BIN="$candidate"
      return 0
    fi
  done

  echo "No compatible llama-server found, trying to build host binary..." >&2
  cmake -S "$ROOT_DIR" -B "$ROOT_DIR/build-host" -DCMAKE_BUILD_TYPE=Release >/dev/null
  cmake --build "$ROOT_DIR/build-host" --target llama-server -j"$THREADS"

  candidate="$ROOT_DIR/build-host/bin/llama-server"
  if [ -e "$candidate" ] && ensure_exec "$candidate" && is_arch_compatible "$candidate"; then
    SERVER_BIN="$candidate"
    return 0
  fi

  echo "Failed to locate/build a compatible llama-server binary." >&2
  exit 1
}

setup_python() {
  if [ "$SKIP_VENV" -eq 1 ]; then
    EVAL_PYTHON="$PYTHON_BIN"
  else
    if [ ! -d "$VENV_DIR" ]; then
      "$PYTHON_BIN" -m venv "$VENV_DIR"
    fi
    EVAL_PYTHON="$VENV_DIR/bin/python"
  fi

  if ! "$EVAL_PYTHON" -V >/dev/null 2>&1
  then
    echo "Selected Python interpreter is not runnable: $EVAL_PYTHON" >&2
    exit 1
  fi
}

wait_for_server() {
  local attempts=90
  local i
  for i in $(seq 1 "$attempts"); do
    if "$EVAL_PYTHON" - "$PORT" "$MODEL_ALIAS" <<'PY' >/dev/null 2>&1
import json
import sys
import urllib.request

port = sys.argv[1]
model_id = sys.argv[2]
url = f"http://127.0.0.1:{port}/v1/models"
with urllib.request.urlopen(url, timeout=3) as resp:
    payload = json.load(resp)
if not any(item.get("id") == model_id for item in payload.get("data", [])):
    raise SystemExit(2)
PY
    then
      return 0
    fi
    sleep 2
  done
  return 1
}

cleanup() {
  if [ -n "$SERVER_PID" ]; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" >/dev/null 2>&1 || true
  fi
}

require_path "$AICAS_DIR/acc_eval.py" "acc_eval.py"
require_path "$MODEL" "model gguf"
require_path "$MMPROJ" "mmproj gguf"
require_path "$IMAGE_FOLDER" "image folder"
require_path "$OCRBENCH_FILE" "OCRBench input json"

pick_server_bin
setup_python

mkdir -p "$OUTPUT_FOLDER"
SERVER_LOG="$OUTPUT_FOLDER/server.log"
BASE_URL="http://127.0.0.1:$PORT/v1"
rm -f "$SERVER_LOG"

echo "Using llama-server: $SERVER_BIN"
echo "Using OCRBench file: $OCRBENCH_FILE"
echo "Results output dir: $OUTPUT_FOLDER"
echo "Using base URL: $BASE_URL"

trap cleanup EXIT

LD_LIBRARY_PATH="$(dirname "$SERVER_BIN"):${LD_LIBRARY_PATH:-}" \
  "$SERVER_BIN" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --alias "$MODEL_ALIAS" \
    -m "$MODEL" \
    --mmproj "$MMPROJ" \
    -t "$THREADS" \
    >"$SERVER_LOG" 2>&1 &
SERVER_PID="$!"

if ! wait_for_server; then
  echo "llama-server did not become ready, tail of log:" >&2
  tail -n 80 "$SERVER_LOG" >&2 || true
  exit 1
fi

"$EVAL_PYTHON" "$AICAS_DIR/acc_eval.py" \
  --image_folder "$IMAGE_FOLDER" \
  --OCRBench_file "$OCRBENCH_FILE" \
  --output_folder "$OUTPUT_FOLDER" \
  --save_name "$SAVE_NAME" \
  --base-url "$BASE_URL" \
  --request-timeout "$REQUEST_TIMEOUT" \
  --request-retries "$REQUEST_RETRIES" \
  --retry-delay "$RETRY_DELAY"

echo
echo "Accuracy evaluation finished."
echo "Result JSON: $OUTPUT_FOLDER/$SAVE_NAME.json"
echo "Server log:  $SERVER_LOG"
