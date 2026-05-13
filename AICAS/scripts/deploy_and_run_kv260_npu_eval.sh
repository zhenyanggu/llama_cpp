#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  deploy_and_run_kv260_npu_eval.sh [options]

Build the current llama.cpp workspace for KV260 with GGML_NPU enabled, run a
read-only NPU readiness probe on the board, stage shared assets under
/home/ubuntu/aicas, run mmproj W8A8 NPU throughput + acc, and pull the results
back to the host.

Options:
  --host <host>              KV260 host (default: 192.168.0.10).
  --user <user>              KV260 user (default: ubuntu).
  --remote-root <path>       Remote root directory (default: /home/ubuntu/aicas).
  --results-dir <path>       Local results root (default: AICAS/kv260_results/npu-w8a8).
  --model <f16|q8_0|path>    Model preset or custom model GGUF path.
  --mmproj <mixed_v1|w8a8|f16|q8_0|path>
                             mmproj preset or custom mmproj GGUF path.
  --ocrbench-file <path>     Accuracy input JSON (default: AICAS/sampled.json).
  --threads <n>              llama-server thread count (default: 4).
  --port <port>              llama-server port (default: 8081).
  --sdk-env <path>           KV260 SDK env script path.
  --build-dir <path>         Cross-build directory (default: build-kv260-npu-current).
  --run-id <id>              Override run id.
  --throughput-image <path>  Image used for throughput_eval.py.
  --no-server-release-mode   Do not add throughput-oriented server flags.
  --throughput-only          Run throughput_eval only (skip acc_eval).
  --skip-build               Reuse existing build output.
  --skip-readiness-probe     Skip the non-root readiness probe.
  --build-only               Build the NPU-enabled llama-server and exit.
  --force-sync-shared        Re-copy shared assets even if markers exist.
  -h, --help                 Show this help.

Environment overrides for remote llama-server:
  GGML_NPU_SPM_BYTES         Default: 524288
  GGML_NPU_ACC_BYTES         Default: 524288
  GGML_NPU_GUARD_BYTES       Default: 4096
  GGML_NPU_STAGE2_K_BYTES    Default: 4096
  GGML_NPU_TILE_ALIGN_DEBUG  Enable single-tile NPU-vs-CPU reference check
  GGML_NPU_TILE_ALIGN_LAYER_ID  Optional layer filter (default: all)
  GGML_NPU_TILE_ALIGN_TILE_INDEX Optional tile filter (default: all)
  GGML_NPU_TILE_ALIGN_MAX_LOGS   Max logs emitted (default: 1)
  GGML_NPU_TILE_ALIGN_ABS_TOL    Absolute diff tolerance (default: 1e-3)
  GGML_NPU_AICAS_BIAS_MODE       AICAS bias path mode: auto|precomp|raw
  GGML_NPU_DISABLE_FOLD_OUTPUT   Disable NPU fold-output fast path when set
  GGML_NPU_TILING_MODE           Tiling mode: auto|activation|weight
  GGML_NPU_LAMBDA_DMA            DMA launch penalty in bytes for tiling search
  GGML_NPU_PROFILE_JSON          Remote profile JSON path, or "auto" for run dir
  GGML_NPU_PROFILE_LEVEL         Profile level: compact|diagnostic|shape|layer|full
  NPU_PROFILE_OUT                Remote runtime profile JSON path, or "auto" for run dir
  NPU_CMA_SIZE               Default: 256M
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AICAS_DIR="$ROOT_DIR/AICAS"

HOST="192.168.0.10"
USER_NAME="ubuntu"
REMOTE_ROOT="/home/ubuntu/aicas"
RESULTS_ROOT="$AICAS_DIR/kv260_results/npu-w8a8"
SDK_ENV="/home/gugugu/petalinux/sdk/kv260-2025.1/environment-setup-cortexa72-cortexa53-amd-linux"
BUILD_DIR="$ROOT_DIR/build-kv260-npu-current"

MODEL_F16="$AICAS_DIR/gguf/SmolVLM2-500M-Video-Instruct-f16.gguf"
MODEL_Q8_0="$AICAS_DIR/gguf/SmolVLM2-500M-Video-Instruct-Q8_0.gguf"
MODEL_AWQ_SQ="$AICAS_DIR/output/text-decode-awq-repro/calib16-a0p125-g32/text_sq_prefill_decode_awq_calib16_a0p125_g32_kvq8_scale_f16.gguf"
MMPROJ_F16="$AICAS_DIR/gguf/mmproj-SmolVLM2-500M-Video-Instruct-f16.gguf"
MMPROJ_Q8_0="$AICAS_DIR/gguf/mmproj-SmolVLM2-500M-Video-Instruct-Q8_0.gguf"
MMPROJ_W8A8="$AICAS_DIR/gguf/mmproj-SmolVLM2-500M-Video-Instruct-w8a8.gguf"
MMPROJ_W8A8_MIXED_V1="$AICAS_DIR/gguf/mmproj-SmolVLM2-500M-Video-Instruct-w8a8-mixed-v1.gguf"
MMPROJ_SMOOTHQUANT="$AICAS_DIR/output/smoothquant/full-alpha-0_5-minmax/mmproj.gguf"

MODEL="$MODEL_AWQ_SQ"
MMPROJ="$MMPROJ_SMOOTHQUANT"
OCRBENCH_FILE="$AICAS_DIR/sampled.json"
THROUGHPUT_IMAGE="$AICAS_DIR/data/IIIT5K/test/2543_2.png"
THREADS="4"
PORT="8081"
RUN_ID=""
SKIP_BUILD=0
SKIP_READINESS_PROBE=0
BUILD_ONLY=0
FORCE_SYNC_SHARED=0
THROUGHPUT_ONLY=0
SERVER_RELEASE_MODE=1
MODEL_ALIAS="smolvlm2-gguf-npu"
SAVE_NAME="SmolVLM2_npu_w8a8"

NPU_SPM_BYTES="${GGML_NPU_SPM_BYTES:-524288}"
NPU_ACC_BYTES="${GGML_NPU_ACC_BYTES:-524288}"
NPU_GUARD_BYTES="${GGML_NPU_GUARD_BYTES:-4096}"
NPU_STAGE2_K_BYTES="${GGML_NPU_STAGE2_K_BYTES:-4096}"
NPU_CMA_BYTES="${NPU_CMA_SIZE:-256M}"
NPU_TILE_ALIGN_DEBUG="${GGML_NPU_TILE_ALIGN_DEBUG:-}"
NPU_TILE_ALIGN_LAYER_ID="${GGML_NPU_TILE_ALIGN_LAYER_ID:-}"
NPU_TILE_ALIGN_TILE_INDEX="${GGML_NPU_TILE_ALIGN_TILE_INDEX:-}"
NPU_TILE_ALIGN_MAX_LOGS="${GGML_NPU_TILE_ALIGN_MAX_LOGS:-}"
NPU_TILE_ALIGN_ABS_TOL="${GGML_NPU_TILE_ALIGN_ABS_TOL:-}"
NPU_AICAS_BIAS_MODE="${GGML_NPU_AICAS_BIAS_MODE:-}"
NPU_DISABLE_FOLD_OUTPUT="${GGML_NPU_DISABLE_FOLD_OUTPUT:-}"
NPU_TILING_MODE="${GGML_NPU_TILING_MODE:-}"
NPU_LAMBDA_DMA="${GGML_NPU_LAMBDA_DMA:-}"
NPU_SPM_FACTOR_A="${GGML_NPU_SPM_FACTOR_A:-}"
NPU_SPM_FACTOR_B="${GGML_NPU_SPM_FACTOR_B:-}"
NPU_TK_ALIGN="${GGML_NPU_TK_ALIGN:-}"
NPU_MAX_U="${GGML_NPU_MAX_U:-}"
NPU_MAX_V="${GGML_NPU_MAX_V:-}"
NPU_MAX_TK="${GGML_NPU_MAX_TK:-}"
NPU_PROFILE_JSON="${GGML_NPU_PROFILE_JSON:-}"
NPU_PROFILE_LEVEL="${GGML_NPU_PROFILE_LEVEL:-}"
NPU_RUNTIME_PROFILE_OUT="${NPU_PROFILE_OUT:-}"

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
  -DGGML_NPU=ON
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
        q8_0|q8)
          MODEL="$MODEL_Q8_0"
          ;;
        *)
          MODEL="$2"
          ;;
      esac
      shift 2
      ;;
    --mmproj)
      case "$2" in
        mixed_v1)
          MMPROJ="$MMPROJ_W8A8_MIXED_V1"
          ;;
        w8a8)
          MMPROJ="$MMPROJ_W8A8"
          ;;
        f16)
          MMPROJ="$MMPROJ_F16"
          ;;
        q8_0|q8)
          MMPROJ="$MMPROJ_Q8_0"
          ;;
        *)
          MMPROJ="$2"
          ;;
      esac
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
    --throughput-only)
      THROUGHPUT_ONLY=1
      shift
      ;;
    --no-server-release-mode)
      SERVER_RELEASE_MODE=0
      shift
      ;;
    --skip-build)
      SKIP_BUILD=1
      shift
      ;;
    --skip-readiness-probe)
      SKIP_READINESS_PROBE=1
      shift
      ;;
    --build-only)
      BUILD_ONLY=1
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
REMOTE_NPU_RUNS_DIR="$REMOTE_ROOT/runs/npu-w8a8"

SERVER_BIN="$BUILD_DIR/bin/llama-server"
RESULTS_DIR=""
RUN_DIR=""
RUN_META_FILE=""
GIT_STATUS_FILE=""
DATA_MANIFEST_FILE=""
READINESS_LOG_FILE=""
REMOTE_NPU_PROFILE_JSON=""
REMOTE_NPU_RUNTIME_PROFILE_JSON=""
DATA_MANIFEST_COUNT="0"
SERVER_BIN_SHA256=""
REMOTE_SERVER_BIN_SHA256=""
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
  rm -f \
    "$LOCAL_RUNTIME_LIB_DIR"/libgomp.so* \
    "$LOCAL_RUNTIME_LIB_DIR"/libstdc++.so* \
    "$LOCAL_RUNTIME_LIB_DIR"/libgcc_s.so*

  for lib in \
    libgomp.so libgomp.so.1 libgomp.so.1.0.0 \
    libstdc++.so libstdc++.so.6 libstdc++.so.6.0.32 \
    libgcc_s.so libgcc_s.so.1; do
    if [ -e "$SDK_SYSROOT_LIB_DIR/$lib" ]; then
      cp -a "$SDK_SYSROOT_LIB_DIR/$lib" "$LOCAL_RUNTIME_LIB_DIR/"
    fi
  done

  require_path "$LOCAL_RUNTIME_LIB_DIR/libgomp.so.1" "KV260 runtime libgomp"
  require_path "$LOCAL_RUNTIME_LIB_DIR/libstdc++.so.6" "KV260 runtime libstdc++"
  require_path "$LOCAL_RUNTIME_LIB_DIR/libgcc_s.so.1" "KV260 runtime libgcc_s"
}

sync_runtime_libs() {
  local lib
  prepare_runtime_libs
  ssh "${SSH_OPTS[@]}" "$TARGET" "mkdir -p '$REMOTE_LIB_DIR'"

  for lib in \
    "$LOCAL_RUNTIME_LIB_DIR"/libgomp.so* \
    "$LOCAL_RUNTIME_LIB_DIR"/libstdc++.so* \
    "$LOCAL_RUNTIME_LIB_DIR"/libgcc_s.so*; do
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

  echo "Building KV260 NPU llama-server into $BUILD_DIR"
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
  SERVER_BIN_SHA256="$(sha256sum "$SERVER_BIN" | awk '{print $1}')"
  echo "Built binary: $file_desc"
  [[ "$file_desc" == *"ARM aarch64"* ]] || {
    echo "Expected an ARM aarch64 llama-server, got: $file_desc" >&2
    exit 1
  }
}

verify_npu_symbols() {
  local nm_output
  nm_output="$(nm -C "$SERVER_BIN")"
  grep -q 'ggml_backend_npu_init' <<< "$nm_output"
  grep -q 'ggml_backend_npu_w8a8_register' <<< "$nm_output"
}

prepare_run_id() {
  if [ -z "$RUN_ID" ]; then
    RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$(git -C "$ROOT_DIR" rev-parse --short HEAD)-npu-w8a8"
  fi

  RESULTS_DIR="$RESULTS_ROOT/$RUN_ID"
  RUN_DIR="$REMOTE_NPU_RUNS_DIR/$RUN_ID"
  RUN_META_FILE="$RESULTS_DIR/run_meta.json"
  READINESS_LOG_FILE="$RESULTS_DIR/readiness_probe.txt"
  if [ "$NPU_PROFILE_JSON" = "auto" ]; then
    REMOTE_NPU_PROFILE_JSON="$RUN_DIR/ggml_npu_profile.json"
  else
    REMOTE_NPU_PROFILE_JSON="$NPU_PROFILE_JSON"
  fi
  if [ "$NPU_RUNTIME_PROFILE_OUT" = "auto" ]; then
    REMOTE_NPU_RUNTIME_PROFILE_JSON="$RUN_DIR/npu_runtime_profile.json"
  else
    REMOTE_NPU_RUNTIME_PROFILE_JSON="$NPU_RUNTIME_PROFILE_OUT"
  fi
  mkdir -p "$RESULTS_DIR"
}

probe_remote_npu_readiness() {
  local probe_output=""

  echo "Running read-only NPU readiness probe on $TARGET"
if ! probe_output="$(ssh "${SSH_OPTS[@]}" "$TARGET" "bash -s" <<'EOF'
set -euo pipefail

export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH"

xmutil_output=""
if [ -x /usr/bin/xmutil ]; then
  xmutil_output="$(/usr/bin/xmutil listapps 2>/dev/null || true)"
elif command -v xmutil >/dev/null 2>&1; then
  xmutil_output="$(xmutil listapps 2>/dev/null || true)"
fi

dev_line=""
if [ -e /dev/npu_kv260 ]; then
  dev_line="$(ls -l /dev/npu_kv260)"
fi

module_line="$(grep '^npu_kv260 ' /proc/modules || true)"
platform_matches="$(ls /sys/bus/platform/devices 2>/dev/null | grep -Ei 'npu|a0000000' || true)"

has_xmutil=0
if printf '%s\n' "$xmutil_output" | grep -Eq '(^|[[:space:]])mynpu([[:space:]]|$)'; then
  has_xmutil=1
fi

has_dev=0
if [ -n "$dev_line" ]; then
  has_dev=1
fi

has_module=0
if [ -n "$module_line" ]; then
  has_module=1
fi

has_platform=0
if [ -n "$platform_matches" ]; then
  has_platform=1
fi

printf '== xmutil ==\n'
if [ -n "$xmutil_output" ]; then
  printf '%s\n' "$xmutil_output"
else
  printf '(xmutil unavailable or mynpu not listed)\n'
fi

printf '== dev ==\n'
if [ "$has_dev" -eq 1 ]; then
  printf '%s\n' "$dev_line"
else
  printf '(missing /dev/npu_kv260)\n'
fi

printf '== modules ==\n'
if [ "$has_module" -eq 1 ]; then
  printf '%s\n' "$module_line"
else
  printf '(npu_kv260 module not loaded)\n'
fi

printf '== platform ==\n'
if [ "$has_platform" -eq 1 ]; then
  printf '%s\n' "$platform_matches"
else
  printf '(no NPU platform device match)\n'
fi

printf '== summary ==\n'
printf 'xmutil_mynpu=%d dev=%d module=%d platform=%d\n' \
  "$has_xmutil" "$has_dev" "$has_module" "$has_platform"

if [ "$has_xmutil" -eq 1 ] && \
   [ "$has_dev" -eq 1 ] && \
   [ "$has_module" -eq 1 ] && \
   [ "$has_platform" -eq 1 ]; then
  exit 0
fi

exit 9
EOF
)"; then
    printf '%s\n' "$probe_output" | tee "$READINESS_LOG_FILE"
    echo "Remote NPU readiness check failed. See $READINESS_LOG_FILE" >&2
    echo "Expected mynpu in xmutil, /dev/npu_kv260, the npu_kv260 module, and an NPU platform device before deployment." >&2
    return 1
  fi

  printf '%s\n' "$probe_output" | tee "$READINESS_LOG_FILE"
  return 0
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
  RUN_META_SERVER_RELEASE_MODE="$SERVER_RELEASE_MODE" \
  RUN_META_HOST="$HOST" \
  RUN_META_USER="$USER_NAME" \
  RUN_META_REMOTE_ROOT="$REMOTE_ROOT" \
  RUN_META_SHARED_DIR="$SHARED_DIR" \
  RUN_META_REMOTE_LIB_DIR="$REMOTE_LIB_DIR" \
  RUN_META_RUN_DIR="$RUN_DIR" \
  RUN_META_RESULTS_DIR="$RESULTS_DIR" \
  RUN_META_REQUIRED_DATA_FILES="$DATA_MANIFEST_COUNT" \
  RUN_META_READINESS_LOG="$READINESS_LOG_FILE" \
  RUN_META_SKIP_READINESS_PROBE="$SKIP_READINESS_PROBE" \
  RUN_META_SERVER_BIN_SHA256="$SERVER_BIN_SHA256" \
  RUN_META_REMOTE_SERVER_BIN_SHA256="$REMOTE_SERVER_BIN_SHA256" \
  RUN_META_NPU_SPM_BYTES="$NPU_SPM_BYTES" \
  RUN_META_NPU_ACC_BYTES="$NPU_ACC_BYTES" \
  RUN_META_NPU_GUARD_BYTES="$NPU_GUARD_BYTES" \
  RUN_META_NPU_STAGE2_K_BYTES="$NPU_STAGE2_K_BYTES" \
  RUN_META_NPU_CMA_BYTES="$NPU_CMA_BYTES" \
  RUN_META_AICAS_MMPROJ_W8A8_DEBUG="${AICAS_MMPROJ_W8A8_DEBUG:-}" \
  RUN_META_NPU_TILE_ALIGN_DEBUG="$NPU_TILE_ALIGN_DEBUG" \
  RUN_META_NPU_TILE_ALIGN_LAYER_ID="$NPU_TILE_ALIGN_LAYER_ID" \
  RUN_META_NPU_TILE_ALIGN_TILE_INDEX="$NPU_TILE_ALIGN_TILE_INDEX" \
  RUN_META_NPU_TILE_ALIGN_MAX_LOGS="$NPU_TILE_ALIGN_MAX_LOGS" \
  RUN_META_NPU_TILE_ALIGN_ABS_TOL="$NPU_TILE_ALIGN_ABS_TOL" \
  RUN_META_NPU_AICAS_BIAS_MODE="$NPU_AICAS_BIAS_MODE" \
  RUN_META_NPU_DISABLE_FOLD_OUTPUT="$NPU_DISABLE_FOLD_OUTPUT" \
  RUN_META_NPU_TILING_MODE="$NPU_TILING_MODE" \
  RUN_META_NPU_LAMBDA_DMA="$NPU_LAMBDA_DMA" \
  RUN_META_NPU_SPM_FACTOR_A="$NPU_SPM_FACTOR_A" \
  RUN_META_NPU_SPM_FACTOR_B="$NPU_SPM_FACTOR_B" \
  RUN_META_NPU_TK_ALIGN="$NPU_TK_ALIGN" \
  RUN_META_NPU_MAX_U="$NPU_MAX_U" \
  RUN_META_NPU_MAX_V="$NPU_MAX_V" \
  RUN_META_NPU_MAX_TK="$NPU_MAX_TK" \
  RUN_META_NPU_PROFILE_JSON="$REMOTE_NPU_PROFILE_JSON" \
  RUN_META_NPU_PROFILE_LEVEL="$NPU_PROFILE_LEVEL" \
  RUN_META_NPU_RUNTIME_PROFILE_OUT="$REMOTE_NPU_RUNTIME_PROFILE_JSON" \
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
    "run_type": "npu-w8a8",
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
    "server_bin_sha256": os.environ["RUN_META_SERVER_BIN_SHA256"],
    "remote_server_bin_sha256": os.environ["RUN_META_REMOTE_SERVER_BIN_SHA256"],
    "build_flags": [
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_SHARED_LIBS=OFF",
        "-DLLAMA_CURL=OFF",
        "-DGGML_BLAS=OFF",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_EXAMPLES=OFF",
        "-DLLAMA_BUILD_TOOLS=ON",
        "-DLLAMA_BUILD_SERVER=ON",
        "-DGGML_NPU=ON",
    ],
    "model": os.environ["RUN_META_MODEL"],
    "mmproj": os.environ["RUN_META_MMPROJ"],
    "ocrbench_file": os.environ["RUN_META_OCRBENCH_FILE"],
    "ocrbench_items": len(ocrbench_data),
    "throughput_image": os.environ["RUN_META_THROUGHPUT_IMAGE"],
    "threads": int(os.environ["RUN_META_THREADS"]),
    "port": int(os.environ["RUN_META_PORT"]),
    "model_alias": os.environ["RUN_META_MODEL_ALIAS"],
    "server_release_mode": os.environ["RUN_META_SERVER_RELEASE_MODE"] == "1",
    "server_args": (
        ["--log-disable", "--no-warmup"]
        if os.environ["RUN_META_SERVER_RELEASE_MODE"] == "1"
        else []
    ),
    "remote_host": os.environ["RUN_META_HOST"],
    "remote_user": os.environ["RUN_META_USER"],
    "remote_root": os.environ["RUN_META_REMOTE_ROOT"],
    "remote_shared_dir": os.environ["RUN_META_SHARED_DIR"],
    "remote_lib_dir": os.environ["RUN_META_REMOTE_LIB_DIR"],
    "remote_run_dir": os.environ["RUN_META_RUN_DIR"],
    "local_results_dir": os.environ["RUN_META_RESULTS_DIR"],
    "required_data_files": int(os.environ["RUN_META_REQUIRED_DATA_FILES"]),
    "readiness_log": os.environ["RUN_META_READINESS_LOG"],
    "skip_readiness_probe": os.environ["RUN_META_SKIP_READINESS_PROBE"] == "1",
    "runtime_env": {
        "MTMD_BACKEND_DEVICE": "NPU",
        "GGML_NPU_SPM_BYTES": os.environ["RUN_META_NPU_SPM_BYTES"],
        "GGML_NPU_ACC_BYTES": os.environ["RUN_META_NPU_ACC_BYTES"],
        "GGML_NPU_GUARD_BYTES": os.environ["RUN_META_NPU_GUARD_BYTES"],
        "GGML_NPU_STAGE2_K_BYTES": os.environ["RUN_META_NPU_STAGE2_K_BYTES"],
        "NPU_CMA_SIZE": os.environ["RUN_META_NPU_CMA_BYTES"],
        "AICAS_MMPROJ_W8A8_DEBUG": os.environ["RUN_META_AICAS_MMPROJ_W8A8_DEBUG"],
        "GGML_NPU_TILE_ALIGN_DEBUG": os.environ["RUN_META_NPU_TILE_ALIGN_DEBUG"],
        "GGML_NPU_TILE_ALIGN_LAYER_ID": os.environ["RUN_META_NPU_TILE_ALIGN_LAYER_ID"],
        "GGML_NPU_TILE_ALIGN_TILE_INDEX": os.environ["RUN_META_NPU_TILE_ALIGN_TILE_INDEX"],
        "GGML_NPU_TILE_ALIGN_MAX_LOGS": os.environ["RUN_META_NPU_TILE_ALIGN_MAX_LOGS"],
        "GGML_NPU_TILE_ALIGN_ABS_TOL": os.environ["RUN_META_NPU_TILE_ALIGN_ABS_TOL"],
        "GGML_NPU_AICAS_BIAS_MODE": os.environ["RUN_META_NPU_AICAS_BIAS_MODE"],
        "GGML_NPU_DISABLE_FOLD_OUTPUT": os.environ["RUN_META_NPU_DISABLE_FOLD_OUTPUT"],
        "GGML_NPU_TILING_MODE": os.environ["RUN_META_NPU_TILING_MODE"],
        "GGML_NPU_LAMBDA_DMA": os.environ["RUN_META_NPU_LAMBDA_DMA"],
        "GGML_NPU_SPM_FACTOR_A": os.environ["RUN_META_NPU_SPM_FACTOR_A"],
        "GGML_NPU_SPM_FACTOR_B": os.environ["RUN_META_NPU_SPM_FACTOR_B"],
        "GGML_NPU_TK_ALIGN": os.environ["RUN_META_NPU_TK_ALIGN"],
        "GGML_NPU_MAX_U": os.environ["RUN_META_NPU_MAX_U"],
        "GGML_NPU_MAX_V": os.environ["RUN_META_NPU_MAX_V"],
        "GGML_NPU_MAX_TK": os.environ["RUN_META_NPU_MAX_TK"],
        "GGML_NPU_PROFILE_JSON": os.environ["RUN_META_NPU_PROFILE_JSON"],
        "GGML_NPU_PROFILE_LEVEL": os.environ["RUN_META_NPU_PROFILE_LEVEL"],
        "NPU_PROFILE_OUT": os.environ["RUN_META_NPU_RUNTIME_PROFILE_OUT"],
    },
}

output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

stage_remote_run_dir() {
  ssh "${SSH_OPTS[@]}" "$TARGET" "mkdir -p '$RUN_DIR' '$REMOTE_NPU_RUNS_DIR'"
  scp "${SSH_OPTS[@]}" "$SERVER_BIN" "$TARGET:$RUN_DIR/llama-server"
  scp "${SSH_OPTS[@]}" "$RUN_META_FILE" "$TARGET:$RUN_DIR/run_meta.json"
  ssh "${SSH_OPTS[@]}" "$TARGET" "chmod +x '$RUN_DIR/llama-server'"
  REMOTE_SERVER_BIN_SHA256="$(ssh "${SSH_OPTS[@]}" "$TARGET" "sha256sum '$RUN_DIR/llama-server' | awk '{print \$1}'")"
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
NPU_SPM_BYTES='$NPU_SPM_BYTES'
NPU_ACC_BYTES='$NPU_ACC_BYTES'
NPU_GUARD_BYTES='$NPU_GUARD_BYTES'
NPU_STAGE2_K_BYTES='$NPU_STAGE2_K_BYTES'
NPU_CMA_BYTES='$NPU_CMA_BYTES'
AICAS_MMPROJ_W8A8_DEBUG='${AICAS_MMPROJ_W8A8_DEBUG:-}'
GGML_NPU_TILE_ALIGN_DEBUG='$NPU_TILE_ALIGN_DEBUG'
GGML_NPU_TILE_ALIGN_LAYER_ID='$NPU_TILE_ALIGN_LAYER_ID'
GGML_NPU_TILE_ALIGN_TILE_INDEX='$NPU_TILE_ALIGN_TILE_INDEX'
GGML_NPU_TILE_ALIGN_MAX_LOGS='$NPU_TILE_ALIGN_MAX_LOGS'
GGML_NPU_TILE_ALIGN_ABS_TOL='$NPU_TILE_ALIGN_ABS_TOL'
GGML_NPU_AICAS_BIAS_MODE='$NPU_AICAS_BIAS_MODE'
GGML_NPU_DISABLE_FOLD_OUTPUT='$NPU_DISABLE_FOLD_OUTPUT'
GGML_NPU_TILING_MODE='$NPU_TILING_MODE'
GGML_NPU_LAMBDA_DMA='$NPU_LAMBDA_DMA'
GGML_NPU_SPM_FACTOR_A='$NPU_SPM_FACTOR_A'
GGML_NPU_SPM_FACTOR_B='$NPU_SPM_FACTOR_B'
GGML_NPU_TK_ALIGN='$NPU_TK_ALIGN'
GGML_NPU_MAX_U='$NPU_MAX_U'
GGML_NPU_MAX_V='$NPU_MAX_V'
GGML_NPU_MAX_TK='$NPU_MAX_TK'
GGML_NPU_PROFILE_JSON='$REMOTE_NPU_PROFILE_JSON'
GGML_NPU_PROFILE_LEVEL='$NPU_PROFILE_LEVEL'
NPU_PROFILE_OUT='$REMOTE_NPU_RUNTIME_PROFILE_JSON'
SERVER_RELEASE_MODE='$SERVER_RELEASE_MODE'

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
  MTMD_BACKEND_DEVICE=NPU \
  GGML_NPU_SPM_BYTES="\$NPU_SPM_BYTES" \
  GGML_NPU_ACC_BYTES="\$NPU_ACC_BYTES" \
  GGML_NPU_GUARD_BYTES="\$NPU_GUARD_BYTES" \
  GGML_NPU_STAGE2_K_BYTES="\$NPU_STAGE2_K_BYTES" \
  NPU_CMA_SIZE="\$NPU_CMA_BYTES" \
  AICAS_MMPROJ_W8A8_DEBUG="\$AICAS_MMPROJ_W8A8_DEBUG" \
  GGML_NPU_TILE_ALIGN_DEBUG="\$GGML_NPU_TILE_ALIGN_DEBUG" \
  GGML_NPU_TILE_ALIGN_LAYER_ID="\$GGML_NPU_TILE_ALIGN_LAYER_ID" \
  GGML_NPU_TILE_ALIGN_TILE_INDEX="\$GGML_NPU_TILE_ALIGN_TILE_INDEX" \
  GGML_NPU_TILE_ALIGN_MAX_LOGS="\$GGML_NPU_TILE_ALIGN_MAX_LOGS" \
  GGML_NPU_TILE_ALIGN_ABS_TOL="\$GGML_NPU_TILE_ALIGN_ABS_TOL" \
  GGML_NPU_AICAS_BIAS_MODE="\$GGML_NPU_AICAS_BIAS_MODE" \
  GGML_NPU_DISABLE_FOLD_OUTPUT="\$GGML_NPU_DISABLE_FOLD_OUTPUT" \
  GGML_NPU_TILING_MODE="\$GGML_NPU_TILING_MODE" \
  GGML_NPU_LAMBDA_DMA="\$GGML_NPU_LAMBDA_DMA" \
  GGML_NPU_SPM_FACTOR_A="\$GGML_NPU_SPM_FACTOR_A" \
  GGML_NPU_SPM_FACTOR_B="\$GGML_NPU_SPM_FACTOR_B" \
  GGML_NPU_TK_ALIGN="\$GGML_NPU_TK_ALIGN" \
  GGML_NPU_MAX_U="\$GGML_NPU_MAX_U" \
  GGML_NPU_MAX_V="\$GGML_NPU_MAX_V" \
  GGML_NPU_MAX_TK="\$GGML_NPU_MAX_TK" \
  GGML_NPU_PROFILE_JSON="\$GGML_NPU_PROFILE_JSON" \
  GGML_NPU_PROFILE_LEVEL="\$GGML_NPU_PROFILE_LEVEL" \
  NPU_PROFILE_OUT="\$NPU_PROFILE_OUT" \
./llama-server \
  --host 127.0.0.1 \
  --port "\$PORT" \
  --alias "\$MODEL_ALIAS" \
  -m "\$REMOTE_MODEL" \
  --mmproj "\$REMOTE_MMPROJ" \
  -t "\$THREADS" \
  \$( [ "\$SERVER_RELEASE_MODE" = "1" ] && printf '%s ' --log-disable --no-warmup ) \
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

if [ "$THROUGHPUT_ONLY" -ne 1 ]; then
python3 "\$REMOTE_EVAL_DIR/acc_eval.py" \
  --image_folder "\$REMOTE_DATA_DIR" \
  --OCRBench_file "\$REMOTE_OCRBENCH" \
  --output_folder "\$RUN_DIR" \
  --save_name "\$SAVE_NAME" \
  --base-url "http://127.0.0.1:\$PORT/v1" \
  --model "\$MODEL_ALIAS"
fi
EOF
}

pull_artifact_if_present() {
  local remote_name="$1"
  scp "${SSH_OPTS[@]}" "$TARGET:$RUN_DIR/$remote_name" "$RESULTS_DIR/" >/dev/null 2>&1 || true
}

build_server
verify_server_bin
verify_npu_symbols

if [ "$BUILD_ONLY" -eq 1 ]; then
  echo
  echo "Built KV260 NPU llama-server:"
  echo "  build  : $BUILD_DIR"
  echo "  binary : $SERVER_BIN"
  exit 0
fi

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

if [ "$SKIP_READINESS_PROBE" -eq 1 ]; then
  printf '%s\n' \
    'Readiness probe skipped by --skip-readiness-probe.' \
    'Assuming manual root-validated NPU bring-up was completed before this run.' \
    | tee "$READINESS_LOG_FILE" >/dev/null
else
  if ! probe_remote_npu_readiness; then
    COMPLETED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    write_run_meta "$COMPLETED_AT" "blocked_not_ready" ""
    exit 1
  fi
fi

ssh "${SSH_OPTS[@]}" "$TARGET" "mkdir -p '$SHARED_DIR' '$REMOTE_GGUF_DIR' '$REMOTE_EVAL_DIR' '$REMOTE_DATA_DIR' '$REMOTE_LIB_DIR' '$REMOTE_NPU_RUNS_DIR'"
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
pull_artifact_if_present "ggml_npu_profile.json"
pull_artifact_if_present "npu_runtime_profile.json"

echo
echo "Results directory: $RESULTS_DIR"
echo "Remote run directory: $RUN_DIR"

if [ "$REMOTE_RC" -ne 0 ]; then
  echo "Remote evaluation failed. Pulled any available artifacts for inspection." >&2
  exit "$REMOTE_RC"
fi
