#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  deploy_and_run_kv260_cpu_eval.sh [options]

Build the current llama.cpp workspace for KV260 on the host, stage shared
assets under /home/ubuntu/aicas on the board, run CPU f16 throughput + acc,
and pull the results back to the host.

Options:
  --host <host>              KV260 host (default: 192.168.0.10).
  --user <user>              KV260 user (default: ubuntu).
  --remote-root <path>       Remote root directory (default: /home/ubuntu/aicas).
  --results-dir <path>       Local results root (default: AICAS/kv260_results/cpu-f16).
  --model <f16|path>         Model preset or custom model GGUF path.
  --mmproj <path>            Path to mmproj GGUF.
  --ocrbench-file <path>     Accuracy input JSON (default: AICAS/sampled.json).
  --threads <n>              llama-server thread count (default: 4).
  --port <port>              llama-server port (default: 8080).
  --sdk-env <path>           KV260 SDK env script path.
  --build-dir <path>         Cross-build directory (default: build-kv260).
  --run-id <id>              Override run id.
  --throughput-image <path>  Image used for throughput_eval.py.
  --skip-build               Reuse existing build output.
  --force-sync-shared        Re-copy shared assets even if markers exist.
  -h, --help                 Show this help.
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AICAS_DIR="$ROOT_DIR/AICAS"

HOST="192.168.0.10"
USER_NAME="ubuntu"
REMOTE_ROOT="/home/ubuntu/aicas"
RESULTS_ROOT="$AICAS_DIR/kv260_results/cpu-f16"
SDK_ENV="/home/gugugu/petalinux/sdk/kv260-2025.1/environment-setup-cortexa72-cortexa53-amd-linux"
BUILD_DIR="$ROOT_DIR/build-kv260"
MODEL_F16="$AICAS_DIR/gguf/SmolVLM2-500M-Video-Instruct-f16.gguf"
MMPROJ_F16="$AICAS_DIR/gguf/mmproj-SmolVLM2-500M-Video-Instruct-f16.gguf"
MODEL="$MODEL_F16"
MMPROJ="$MMPROJ_F16"
OCRBENCH_FILE="$AICAS_DIR/sampled.json"
THROUGHPUT_IMAGE="$AICAS_DIR/data/IIIT5K/test/2543_2.png"
THREADS="4"
PORT="8080"
RUN_ID=""
SKIP_BUILD=0
FORCE_SYNC_SHARED=0
MODEL_ALIAS="smolvlm2-gguf"
SAVE_NAME="SmolVLM2"

SSH_OPTS=(
  -o BatchMode=yes
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
  -o LogLevel=ERROR
)

BUILD_FLAGS=(
  -DCMAKE_BUILD_TYPE=Release
  -DBUILD_SHARED_LIBS=OFF
  -DLLAMA_CURL=OFF
  -DGGML_BLAS=OFF
  -DLLAMA_BUILD_TESTS=OFF
  -DLLAMA_BUILD_EXAMPLES=OFF
  -DLLAMA_BUILD_TOOLS=ON
  -DLLAMA_BUILD_SERVER=ON
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
    --results-dir)
      RESULTS_ROOT="$2"
      shift 2
      ;;
    --model)
      case "$2" in
        f16)
          MODEL="$MODEL_F16"
          ;;
        *)
          MODEL="$2"
          ;;
      esac
      shift 2
      ;;
    --mmproj)
      MMPROJ="$2"
      shift 2
      ;;
    --ocrbench-file)
      OCRBENCH_FILE="$2"
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
    --throughput-image)
      THROUGHPUT_IMAGE="$2"
      shift 2
      ;;
    --skip-build)
      SKIP_BUILD=1
      shift
      ;;
    --force-sync-shared)
      FORCE_SYNC_SHARED=1
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

TARGET="${USER_NAME}@${HOST}"
SHARED_DIR="$REMOTE_ROOT/shared"
REMOTE_EVAL_DIR="$SHARED_DIR/eval"
REMOTE_GGUF_DIR="$SHARED_DIR/gguf"
REMOTE_DATA_DIR="$SHARED_DIR/data"
REMOTE_LIB_DIR="$SHARED_DIR/lib"
REMOTE_DATA_MANIFEST="$REMOTE_EVAL_DIR/.required-data-files.txt"
REMOTE_CPU_RUNS_DIR="$REMOTE_ROOT/runs/cpu-f16"
REMOTE_NPU_RUNS_DIR="$REMOTE_ROOT/runs/npu"

SERVER_BIN="$BUILD_DIR/bin/llama-server"
RESULTS_DIR=""
RUN_DIR=""
RUN_META_FILE=""
GIT_STATUS_FILE=""
DATA_MANIFEST_FILE=""
DATA_MANIFEST_COUNT="0"
STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
GIT_HEAD="$(git -C "$ROOT_DIR" rev-parse HEAD)"
SDK_ROOT=""
SDK_SYSROOT_LIB_DIR=""
LOCAL_RUNTIME_LIB_DIR="$AICAS_DIR/third_party/kv260_runtime/lib"

MODEL_BASENAME=""
MMPROJ_BASENAME=""
OCRBENCH_BASENAME=""
THROUGHPUT_IMAGE_REL=""
REMOTE_MODEL=""
REMOTE_MMPROJ=""
REMOTE_OCRBENCH=""
REMOTE_THROUGHPUT_IMAGE=""

require_path() {
  local path="$1"
  local name="$2"
  [ -e "$path" ] || { echo "Missing $name: $path" >&2; exit 1; }
}

canonicalize_path() {
  python3 - "$1" <<'PY'
import os
import sys

print(os.path.realpath(sys.argv[1]))
PY
}

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

  if [ "$FORCE_SYNC_SHARED" -ne 1 ] && [ -n "$remote_size" ] && [ "$remote_size" = "$local_size" ]; then
    echo "Reusing $label at $dst"
    return 0
  fi

  echo "Syncing $label to $dst"
  remote_dir="$(dirname "$dst")"
  ssh "${SSH_OPTS[@]}" "$TARGET" "mkdir -p '$remote_dir'"
  scp "${SSH_OPTS[@]}" "$src" "$TARGET:$dst"
}

prepare_runtime_libs() {
  local lib
  mkdir -p "$LOCAL_RUNTIME_LIB_DIR"
  rm -f "$LOCAL_RUNTIME_LIB_DIR"/libgomp.so*

  for lib in libgomp.so.1 libgomp.so.1.0.0; do
    if [ -e "$SDK_SYSROOT_LIB_DIR/$lib" ]; then
      cp -a "$SDK_SYSROOT_LIB_DIR/$lib" "$LOCAL_RUNTIME_LIB_DIR/"
    fi
  done

  require_path "$LOCAL_RUNTIME_LIB_DIR/libgomp.so.1" "KV260 runtime libgomp"
}

sync_runtime_libs() {
  local lib
  prepare_runtime_libs
  ssh "${SSH_OPTS[@]}" "$TARGET" "mkdir -p '$REMOTE_LIB_DIR'"

  for lib in "$LOCAL_RUNTIME_LIB_DIR"/libgomp.so*; do
    [ -e "$lib" ] || continue
    sync_file_if_needed "$lib" "$REMOTE_LIB_DIR/$(basename "$lib")" "runtime lib $(basename "$lib")"
  done
}

sync_gguf() {
  sync_file_if_needed "$MODEL" "$REMOTE_MODEL" "model GGUF"
  sync_file_if_needed "$MMPROJ" "$REMOTE_MMPROJ" "mmproj GGUF"
}

sync_eval_payload() {
  sync_file_if_needed "$AICAS_DIR/acc_eval.py" "$REMOTE_EVAL_DIR/acc_eval.py" "acc_eval.py"
  sync_file_if_needed "$AICAS_DIR/throughput_eval.py" "$REMOTE_EVAL_DIR/throughput_eval.py" "throughput_eval.py"
  sync_file_if_needed "$AICAS_DIR/llama_server_client.py" "$REMOTE_EVAL_DIR/llama_server_client.py" "llama_server_client.py"
  sync_file_if_needed "$OCRBENCH_FILE" "$REMOTE_OCRBENCH" "$OCRBENCH_BASENAME"
}

build_data_manifest() {
  DATA_MANIFEST_FILE="$(mktemp)"
  python3 - "$OCRBENCH_FILE" "$THROUGHPUT_IMAGE" "$AICAS_DIR/data" "$DATA_MANIFEST_FILE" <<'PY'
import json
import os
import sys
from pathlib import Path

ocrbench_path = Path(sys.argv[1]).resolve()
throughput_image = Path(sys.argv[2]).resolve()
data_root = Path(sys.argv[3]).resolve()
manifest_path = Path(sys.argv[4])

paths = set()
if throughput_image.is_relative_to(data_root):
    paths.add(str(throughput_image.relative_to(data_root)))
else:
    raise SystemExit(f"Throughput image is not under data root: {throughput_image}")

with ocrbench_path.open("r", encoding="utf-8") as handle:
    items = json.load(handle)

for item in items:
    image_rel = item.get("image_path")
    if not image_rel:
        continue
    image_path = (data_root / image_rel).resolve()
    if not image_path.is_relative_to(data_root):
        raise SystemExit(f"Image path escapes data root: {image_rel}")
    if not image_path.exists():
        raise SystemExit(f"Missing OCRBench image: {image_path}")
    paths.add(str(image_path.relative_to(data_root)))

manifest_path.write_text("".join(f"{path}\n" for path in sorted(paths)), encoding="utf-8")
PY
}

remote_has_manifest_files() {
  scp "${SSH_OPTS[@]}" "$DATA_MANIFEST_FILE" "$TARGET:$REMOTE_DATA_MANIFEST" >/dev/null
  ssh "${SSH_OPTS[@]}" "$TARGET" "python3 - '$REMOTE_DATA_DIR' '$REMOTE_DATA_MANIFEST' <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
missing = []

for line in manifest_path.read_text(encoding='utf-8').splitlines():
    rel = line.strip()
    if not rel:
        continue
    if not (root / rel).exists():
        missing.append(rel)
        if len(missing) >= 10:
            break

if missing:
    print('Missing data files:', file=sys.stderr)
    for rel in missing:
        print(rel, file=sys.stderr)
    raise SystemExit(1)
PY" < "$DATA_MANIFEST_FILE"
}

sync_data_payload() {
  local manifest_count
  build_data_manifest
  manifest_count="$(wc -l < "$DATA_MANIFEST_FILE")"
  DATA_MANIFEST_COUNT="$manifest_count"

  if [ "$FORCE_SYNC_SHARED" -ne 1 ] && remote_has_manifest_files; then
    echo "Reusing data payload at $REMOTE_DATA_DIR for $manifest_count required files"
    return 0
  fi

  echo "Syncing $manifest_count required data files to $REMOTE_DATA_DIR"
  ssh "${SSH_OPTS[@]}" "$TARGET" "mkdir -p '$REMOTE_DATA_DIR'"
  tar -C "$AICAS_DIR/data" -cf - -T "$DATA_MANIFEST_FILE" | \
    ssh "${SSH_OPTS[@]}" "$TARGET" "cd '$REMOTE_DATA_DIR' && tar -xmf -"
}

build_server() {
  if [ "$SKIP_BUILD" -eq 1 ]; then
    require_path "$SERVER_BIN" "llama-server binary"
    return 0
  fi

  require_path "$SDK_ENV" "KV260 SDK env script"

  echo "Building KV260 llama-server into $BUILD_DIR"
  rm -rf "$BUILD_DIR"
  (
    set -euo pipefail
    unset LD_LIBRARY_PATH
    # shellcheck disable=SC1090
    source "$SDK_ENV" >/dev/null
    cmake -S "$ROOT_DIR" -B "$BUILD_DIR" -G Ninja "${BUILD_FLAGS[@]}"
    cmake --build "$BUILD_DIR" --target llama-server -j"$(nproc)"
  )
}

verify_server_bin() {
  local file_desc
  require_path "$SERVER_BIN" "llama-server binary"
  file_desc="$(file "$SERVER_BIN")"
  echo "Built binary: $file_desc"
  [[ "$file_desc" == *"ARM aarch64"* ]] || {
    echo "Expected an ARM aarch64 llama-server, got: $file_desc" >&2
    exit 1
  }
}

prepare_run_id() {
  if [ -z "$RUN_ID" ]; then
    RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$(git -C "$ROOT_DIR" rev-parse --short HEAD)-cpu-f16"
  fi

  RESULTS_DIR="$RESULTS_ROOT/$RUN_ID"
  RUN_DIR="$REMOTE_CPU_RUNS_DIR/$RUN_ID"
  RUN_META_FILE="$RESULTS_DIR/run_meta.json"
  mkdir -p "$RESULTS_DIR"
}

write_run_meta() {
  local completed_at="$1"
  local run_status="$2"
  local remote_exit_code="$3"

  RUN_META_RUN_ID="$RUN_ID" \
  RUN_META_STARTED_AT="$STARTED_AT" \
  RUN_META_COMPLETED_AT="$completed_at" \
  RUN_META_STATUS="$run_status" \
  RUN_META_REMOTE_EXIT_CODE="$remote_exit_code" \
  RUN_META_GIT_HEAD="$GIT_HEAD" \
  RUN_META_SDK_ENV="$SDK_ENV" \
  RUN_META_BUILD_DIR="$BUILD_DIR" \
  RUN_META_SERVER_BIN="$SERVER_BIN" \
  RUN_META_MODEL="$MODEL" \
  RUN_META_MMPROJ="$MMPROJ" \
  RUN_META_OCRBENCH_FILE="$OCRBENCH_FILE" \
  RUN_META_THROUGHPUT_IMAGE="$THROUGHPUT_IMAGE" \
  RUN_META_THREADS="$THREADS" \
  RUN_META_PORT="$PORT" \
  RUN_META_MODEL_ALIAS="$MODEL_ALIAS" \
  RUN_META_HOST="$HOST" \
  RUN_META_USER="$USER_NAME" \
  RUN_META_REMOTE_ROOT="$REMOTE_ROOT" \
  RUN_META_SHARED_DIR="$SHARED_DIR" \
  RUN_META_REMOTE_LIB_DIR="$REMOTE_LIB_DIR" \
  RUN_META_RUN_DIR="$RUN_DIR" \
  RUN_META_RESULTS_DIR="$RESULTS_DIR" \
  RUN_META_REQUIRED_DATA_FILES="$DATA_MANIFEST_COUNT" \
  python3 - "$RUN_META_FILE" "$GIT_STATUS_FILE" "$OCRBENCH_FILE" <<'PY'
import json
import os
import sys
from pathlib import Path

output_path = Path(sys.argv[1])
git_status_path = Path(sys.argv[2])
ocrbench_path = Path(sys.argv[3])

status_lines = git_status_path.read_text(encoding="utf-8").splitlines()
with ocrbench_path.open("r", encoding="utf-8") as handle:
    ocrbench_data = json.load(handle)

payload = {
    "run_id": os.environ["RUN_META_RUN_ID"],
    "run_type": "cpu-f16",
    "run_started_at": os.environ["RUN_META_STARTED_AT"],
    "run_completed_at": os.environ["RUN_META_COMPLETED_AT"],
    "run_status": os.environ["RUN_META_STATUS"],
    "remote_exit_code": (
        int(os.environ["RUN_META_REMOTE_EXIT_CODE"])
        if os.environ["RUN_META_REMOTE_EXIT_CODE"]
        else None
    ),
    "git_head": os.environ["RUN_META_GIT_HEAD"],
    "repo_dirty": bool(status_lines),
    "git_status_short": status_lines,
    "sdk_env": os.environ["RUN_META_SDK_ENV"],
    "build_dir": os.environ["RUN_META_BUILD_DIR"],
    "server_bin": os.environ["RUN_META_SERVER_BIN"],
    "build_flags": [
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_SHARED_LIBS=OFF",
        "-DLLAMA_CURL=OFF",
        "-DGGML_BLAS=OFF",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_EXAMPLES=OFF",
        "-DLLAMA_BUILD_TOOLS=ON",
        "-DLLAMA_BUILD_SERVER=ON",
    ],
    "model": os.environ["RUN_META_MODEL"],
    "mmproj": os.environ["RUN_META_MMPROJ"],
    "ocrbench_file": os.environ["RUN_META_OCRBENCH_FILE"],
    "ocrbench_items": len(ocrbench_data),
    "throughput_image": os.environ["RUN_META_THROUGHPUT_IMAGE"],
    "threads": int(os.environ["RUN_META_THREADS"]),
    "port": int(os.environ["RUN_META_PORT"]),
    "model_alias": os.environ["RUN_META_MODEL_ALIAS"],
    "remote_host": os.environ["RUN_META_HOST"],
    "remote_user": os.environ["RUN_META_USER"],
    "remote_root": os.environ["RUN_META_REMOTE_ROOT"],
    "remote_shared_dir": os.environ["RUN_META_SHARED_DIR"],
    "remote_lib_dir": os.environ["RUN_META_REMOTE_LIB_DIR"],
    "remote_run_dir": os.environ["RUN_META_RUN_DIR"],
    "local_results_dir": os.environ["RUN_META_RESULTS_DIR"],
    "required_data_files": int(os.environ["RUN_META_REQUIRED_DATA_FILES"]),
}

output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

stage_remote_run_dir() {
  ssh "${SSH_OPTS[@]}" "$TARGET" "mkdir -p '$RUN_DIR' '$REMOTE_NPU_RUNS_DIR'"
  scp "${SSH_OPTS[@]}" "$SERVER_BIN" "$TARGET:$RUN_DIR/llama-server"
  scp "${SSH_OPTS[@]}" "$RUN_META_FILE" "$TARGET:$RUN_DIR/run_meta.json"
  ssh "${SSH_OPTS[@]}" "$TARGET" "chmod +x '$RUN_DIR/llama-server'"
}

verify_remote_server_bin() {
  ssh "${SSH_OPTS[@]}" "$TARGET" "cd '$RUN_DIR' && if LD_LIBRARY_PATH='$REMOTE_LIB_DIR:\${LD_LIBRARY_PATH:-}' ldd ./llama-server | grep -q 'not found'; then LD_LIBRARY_PATH='$REMOTE_LIB_DIR:\${LD_LIBRARY_PATH:-}' ldd ./llama-server; exit 1; fi"
}

run_remote_eval() {
  ssh "${SSH_OPTS[@]}" "$TARGET" "bash -s" <<EOF
set -euo pipefail
RUN_DIR='$RUN_DIR'
REMOTE_MODEL='$REMOTE_MODEL'
REMOTE_MMPROJ='$REMOTE_MMPROJ'
REMOTE_OCRBENCH='$REMOTE_OCRBENCH'
REMOTE_DATA_DIR='$REMOTE_DATA_DIR'
REMOTE_EVAL_DIR='$REMOTE_EVAL_DIR'
REMOTE_THROUGHPUT_IMAGE='$REMOTE_THROUGHPUT_IMAGE'
PORT='$PORT'
THREADS='$THREADS'
MODEL_ALIAS='$MODEL_ALIAS'
SAVE_NAME='$SAVE_NAME'
REMOTE_LIB_DIR='$REMOTE_LIB_DIR'

cleanup() {
  if [ -n "\${SERVER_PID:-}" ]; then
    kill "\$SERVER_PID" >/dev/null 2>&1 || true
    wait "\$SERVER_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

cd "\$RUN_DIR"
rm -f server.log throughput_metrics.json "\$SAVE_NAME.json"

env \
  LD_LIBRARY_PATH="\$REMOTE_LIB_DIR:\${LD_LIBRARY_PATH:-}" \
./llama-server \
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
  -i "\$REMOTE_THROUGHPUT_IMAGE" \
  -o "\$RUN_DIR/throughput_metrics.json" \
  --base-url "http://127.0.0.1:\$PORT/v1" \
  --model "\$MODEL_ALIAS"

python3 "\$REMOTE_EVAL_DIR/acc_eval.py" \
  --image_folder "\$REMOTE_DATA_DIR" \
  --OCRBench_file "\$REMOTE_OCRBENCH" \
  --output_folder "\$RUN_DIR" \
  --save_name "\$SAVE_NAME" \
  --base-url "http://127.0.0.1:\$PORT/v1" \
  --model "\$MODEL_ALIAS"
EOF
}

pull_artifact_if_present() {
  local remote_name="$1"
  scp "${SSH_OPTS[@]}" "$TARGET:$RUN_DIR/$remote_name" "$RESULTS_DIR/" >/dev/null 2>&1 || true
}

require_path "$MODEL" "model GGUF"
require_path "$MMPROJ" "mmproj GGUF"
require_path "$OCRBENCH_FILE" "OCRBench JSON"
require_path "$THROUGHPUT_IMAGE" "throughput image"
[ -d "$AICAS_DIR/data" ] || {
  echo "Missing data root: $AICAS_DIR/data" >&2
  exit 1
}

MODEL="$(canonicalize_path "$MODEL")"
MMPROJ="$(canonicalize_path "$MMPROJ")"
OCRBENCH_FILE="$(canonicalize_path "$OCRBENCH_FILE")"
THROUGHPUT_IMAGE="$(canonicalize_path "$THROUGHPUT_IMAGE")"
RESULTS_ROOT="$(canonicalize_path "$RESULTS_ROOT")"
SDK_ENV="$(canonicalize_path "$SDK_ENV")"
SDK_ROOT="$(cd "$(dirname "$SDK_ENV")" && pwd)"
SDK_SYSROOT_LIB_DIR="$SDK_ROOT/sysroots/cortexa72-cortexa53-amd-linux/usr/lib"

[ "${THROUGHPUT_IMAGE#"$AICAS_DIR/data/"}" != "$THROUGHPUT_IMAGE" ] || {
  echo "--throughput-image must live under $AICAS_DIR/data so it can be staged into shared/data." >&2
  exit 1
}

MODEL_BASENAME="$(basename "$MODEL")"
MMPROJ_BASENAME="$(basename "$MMPROJ")"
OCRBENCH_BASENAME="$(basename "$OCRBENCH_FILE")"
THROUGHPUT_IMAGE_REL="${THROUGHPUT_IMAGE#"$AICAS_DIR/data/"}"
REMOTE_MODEL="$REMOTE_GGUF_DIR/$MODEL_BASENAME"
REMOTE_MMPROJ="$REMOTE_GGUF_DIR/$MMPROJ_BASENAME"
REMOTE_OCRBENCH="$REMOTE_EVAL_DIR/$OCRBENCH_BASENAME"
REMOTE_THROUGHPUT_IMAGE="$REMOTE_DATA_DIR/$THROUGHPUT_IMAGE_REL"

GIT_STATUS_FILE="$(mktemp)"
trap 'rm -f "$GIT_STATUS_FILE" "$DATA_MANIFEST_FILE"' EXIT
git -C "$ROOT_DIR" status --short > "$GIT_STATUS_FILE"

prepare_run_id
build_server
verify_server_bin

ssh "${SSH_OPTS[@]}" "$TARGET" "mkdir -p '$SHARED_DIR' '$REMOTE_GGUF_DIR' '$REMOTE_EVAL_DIR' '$REMOTE_DATA_DIR' '$REMOTE_LIB_DIR' '$REMOTE_CPU_RUNS_DIR' '$REMOTE_NPU_RUNS_DIR'"
sync_gguf
sync_eval_payload
sync_data_payload
sync_runtime_libs

write_run_meta "" "prepared" ""
stage_remote_run_dir
verify_remote_server_bin

REMOTE_RC=0
if run_remote_eval; then
  REMOTE_RC=0
else
  REMOTE_RC=$?
fi

COMPLETED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
RUN_STATUS="completed"
if [ "$REMOTE_RC" -ne 0 ]; then
  RUN_STATUS="failed"
fi

write_run_meta "$COMPLETED_AT" "$RUN_STATUS" "$REMOTE_RC"
scp "${SSH_OPTS[@]}" "$RUN_META_FILE" "$TARGET:$RUN_DIR/run_meta.json" >/dev/null 2>&1 || true

pull_artifact_if_present "throughput_metrics.json"
pull_artifact_if_present "$SAVE_NAME.json"
pull_artifact_if_present "server.log"

echo
echo "Results directory: $RESULTS_DIR"
echo "Remote run directory: $RUN_DIR"

if [ "$REMOTE_RC" -ne 0 ]; then
  echo "Remote evaluation failed. Pulled any available artifacts for inspection." >&2
  exit "$REMOTE_RC"
fi
