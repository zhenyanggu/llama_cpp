#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_submission.sh [options]

Options:
  --host <host>
  --user <user>
  --remote-root <path>
  --run-id <id>
  --threads <n>
  --port <port>
  --skip-board-init
  --skip-validate
  --skip-acc
  --cache-type-k <t>
  --cache-type-v <t>
  --mmproj <path>
  --sample-json <path>
  --sample-images-root <path>
  --throughput-image-rel <relpath>
  --probe-only
  --probe-text <text>
  --probe-max-tokens <n>
  --probe-timeout <seconds>
  --probe-poll-interval <seconds>
  --probe-strace
  --sudo-password <pass>
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PAYLOAD_DIR="$BUNDLE_DIR/payload"
RESULTS_ROOT="$BUNDLE_DIR/results/official"
VALIDATE_SCRIPT="$SCRIPT_DIR/validate_bundle.sh"
BOARD_INIT_SCRIPT="$SCRIPT_DIR/board_init_overlay.sh"
SUMMARY_SCRIPT="$SCRIPT_DIR/summarize_results.py"
REPLAY_SCRIPT="$SCRIPT_DIR/run_layer_gemm_replay.sh"
SOURCE_STATE_DIR="$BUNDLE_DIR/source_state"

HOST="192.168.0.10"
USER_NAME="ubuntu"
REMOTE_ROOT="/home/ubuntu/aicas"
RUN_ID=""
THREADS="4"
PORT="8081"
SKIP_BOARD_INIT=0
SKIP_VALIDATE=0
SKIP_ACC=0
MODEL_ALIAS="smolvlm2-gguf-npu-versavlm"
CACHE_TYPE_K=""
CACHE_TYPE_V=""
MMPROJ_FILE="$PAYLOAD_DIR/gguf/mmproj-fallback-search-fb_attn_k-per-tensor.gguf"
SAMPLE_JSON="$PAYLOAD_DIR/data/sampled_100.json"
SAMPLE_IMAGES_ROOT="$PAYLOAD_DIR/data/images"
THROUGHPUT_IMAGE_REL="IIIT5K/test/2543_2.png"
PROBE_ONLY=0
PROBE_TEXT="Please OCR this image briefly and summarize the visible content in one short paragraph."
PROBE_MAX_TOKENS="16"
PROBE_TIMEOUT="120"
PROBE_POLL_INTERVAL="5"
PROBE_STRACE=0
SUDO_PASSWORD="${BOARD_SUDO_PASSWORD:-}"

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
    --host) HOST="$2"; shift 2 ;;
    --user) USER_NAME="$2"; shift 2 ;;
    --remote-root) REMOTE_ROOT="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --skip-board-init) SKIP_BOARD_INIT=1; shift ;;
    --skip-validate) SKIP_VALIDATE=1; shift ;;
    --skip-acc) SKIP_ACC=1; shift ;;
    --cache-type-k) CACHE_TYPE_K="$2"; shift 2 ;;
    --cache-type-v) CACHE_TYPE_V="$2"; shift 2 ;;
    --mmproj) MMPROJ_FILE="$2"; shift 2 ;;
    --sample-json) SAMPLE_JSON="$2"; shift 2 ;;
    --sample-images-root) SAMPLE_IMAGES_ROOT="$2"; shift 2 ;;
    --throughput-image-rel) THROUGHPUT_IMAGE_REL="$2"; shift 2 ;;
    --probe-only) PROBE_ONLY=1; shift ;;
    --probe-text) PROBE_TEXT="$2"; shift 2 ;;
    --probe-max-tokens) PROBE_MAX_TOKENS="$2"; shift 2 ;;
    --probe-timeout) PROBE_TIMEOUT="$2"; shift 2 ;;
    --probe-poll-interval) PROBE_POLL_INTERVAL="$2"; shift 2 ;;
    --probe-strace) PROBE_STRACE=1; shift ;;
    --sudo-password) SUDO_PASSWORD="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

TARGET="${USER_NAME}@${HOST}"
REMOTE_SHARED_DIR="$REMOTE_ROOT/shared"
REMOTE_GGUF_DIR="$REMOTE_SHARED_DIR/gguf"
REMOTE_EVAL_DIR="$REMOTE_SHARED_DIR/eval"
REMOTE_DATA_DIR="$REMOTE_SHARED_DIR/data"
REMOTE_LIB_DIR="$REMOTE_SHARED_DIR/lib"
REMOTE_BOARD_SUPPORT_DIR="$REMOTE_ROOT/board_support"
REMOTE_BOARD_OVERLAY_DIR="$REMOTE_BOARD_SUPPORT_DIR/overlay/double_dma_overlayapp"
REMOTE_BOARD_DRIVER_DIR="$REMOTE_BOARD_SUPPORT_DIR/driver"
REMOTE_BOARD_QSPI_DIR="$REMOTE_BOARD_SUPPORT_DIR/qspi"
REMOTE_BOARD_TESTS_DIR="$REMOTE_BOARD_SUPPORT_DIR/tests"
REMOTE_SCRIPTS_DIR="$REMOTE_ROOT/scripts"
REMOTE_RUNS_DIR="$REMOTE_ROOT/runs/submission-versavlm"

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

build_custom_manifest() {
  local sample_json="$1"
  local images_root="$2"
  local manifest="$3"
  python3 - "$sample_json" "$images_root" "$manifest" <<'PY'
import json
import sys
from pathlib import Path
sample_json = Path(sys.argv[1]).resolve()
images_root = Path(sys.argv[2]).resolve()
manifest = Path(sys.argv[3]).resolve()
items = json.loads(sample_json.read_text(encoding='utf-8'))
paths = []
seen = set()
for item in items:
    rel = item.get('image_path')
    if not rel or rel in seen:
        continue
    src = (images_root / rel).resolve()
    if not src.exists():
        raise SystemExit(f'Missing sampled image: {src}')
    seen.add(rel)
    paths.append(rel)
manifest.write_text(''.join(f"{p}\n" for p in sorted(paths)), encoding='utf-8')
PY
}

sync_sample_payload() {
  local local_json="$1"
  local local_root="$2"
  local remote_json="$3"
  local remote_root="$4"
  local manifest="$5"
  scp "${SSH_OPTS[@]}" "$local_json" "$TARGET:$remote_json"
  tar -C "$local_root" -cf - -T "$manifest" | ssh "${SSH_OPTS[@]}" "$TARGET" "mkdir -p '$remote_root' && cd '$remote_root' && tar -xmf -"
}

probe_remote_ready() {
  ssh "${SSH_OPTS[@]}" "$TARGET" "bash -s" <<'PROBE'
set -euo pipefail
printf '== xmutil ==\n'
/usr/bin/xmutil listapps 2>/dev/null || true
printf '== dev ==\n'
ls -l /dev/npu_kv260 2>/dev/null || true
printf '== modules ==\n'
grep '^npu_kv260 ' /proc/modules || true
printf '== platform ==\n'
ls /sys/bus/platform/devices 2>/dev/null | grep -Ei 'npu|a0000000' || true
PROBE
}

read_optional_file() {
  local path="$1"
  if [ -f "$path" ]; then
    cat "$path"
  fi
}

write_run_meta() {
  local local_run_dir="$1"
  local remote_run_dir="$2"
  local status="$3"
  local remote_exit_code="$4"
  local board_init_log="$5"
  local readiness_log="$6"
  local git_head git_status
  git_head="$(read_optional_file "$SOURCE_STATE_DIR/git_head.txt" | tr -d '\n')"
  git_status="$(read_optional_file "$SOURCE_STATE_DIR/git_status.txt")"
  RUN_META_RUN_ID="$RUN_ID" \
  RUN_META_STATUS="$status" \
  RUN_META_REMOTE_EXIT_CODE="$remote_exit_code" \
  RUN_META_GIT_HEAD="$git_head" \
  RUN_META_GIT_STATUS="$git_status" \
  RUN_META_MODEL="payload/gguf/SmolVLM2-500M-Video-Instruct-Q8_0.gguf" \
  RUN_META_MMPROJ="$(realpath --relative-to="$BUNDLE_DIR" "$MMPROJ_FILE")" \
  RUN_META_HOST="$HOST" \
  RUN_META_USER="$USER_NAME" \
  RUN_META_REMOTE_ROOT="$REMOTE_ROOT" \
  RUN_META_REMOTE_RUN_DIR="$remote_run_dir" \
  RUN_META_THREADS="$THREADS" \
  RUN_META_PORT="$PORT" \
  RUN_META_MODEL_ALIAS="$MODEL_ALIAS" \
  RUN_META_SAMPLE_JSON="$(realpath --relative-to="$BUNDLE_DIR" "$SAMPLE_JSON")" \
  RUN_META_SAMPLE_IMAGES_ROOT="$(realpath --relative-to="$BUNDLE_DIR" "$SAMPLE_IMAGES_ROOT")" \
  RUN_META_THROUGHPUT_IMAGE_REL="$THROUGHPUT_IMAGE_REL" \
  RUN_META_PROBE_ONLY="$PROBE_ONLY" \
  RUN_META_PROBE_TEXT="$PROBE_TEXT" \
  RUN_META_PROBE_MAX_TOKENS="$PROBE_MAX_TOKENS" \
  RUN_META_PROBE_TIMEOUT="$PROBE_TIMEOUT" \
  RUN_META_PROBE_POLL_INTERVAL="$PROBE_POLL_INTERVAL" \
  RUN_META_PROBE_STRACE="$PROBE_STRACE" \
  RUN_META_BOARD_INIT_LOG="$(realpath --relative-to="$BUNDLE_DIR" "$board_init_log")" \
  RUN_META_READINESS_LOG="$(realpath --relative-to="$BUNDLE_DIR" "$readiness_log")" \
  RUN_META_NPU_SPM_BYTES="$NPU_SPM_BYTES" \
  RUN_META_NPU_ACC_BYTES="$NPU_ACC_BYTES" \
  RUN_META_NPU_GUARD_BYTES="$NPU_GUARD_BYTES" \
  RUN_META_NPU_STAGE2_K_BYTES="$NPU_STAGE2_K_BYTES" \
  RUN_META_NPU_CMA_BYTES="$NPU_CMA_BYTES" \
  python3 - "$local_run_dir/run_meta.json" <<'PY'
import json
import os
import sys
payload = {
    'run_id': os.environ['RUN_META_RUN_ID'],
    'run_type': 'submission-versavlm',
    'run_status': os.environ['RUN_META_STATUS'],
    'remote_exit_code': int(os.environ['RUN_META_REMOTE_EXIT_CODE']) if os.environ['RUN_META_REMOTE_EXIT_CODE'] else None,
    'git_head': os.environ.get('RUN_META_GIT_HEAD') or None,
    'git_status_short': [line for line in os.environ.get('RUN_META_GIT_STATUS', '').splitlines() if line],
    'model': os.environ['RUN_META_MODEL'],
    'mmproj': os.environ['RUN_META_MMPROJ'],
    'remote_host': os.environ['RUN_META_HOST'],
    'remote_user': os.environ['RUN_META_USER'],
    'remote_root': os.environ['RUN_META_REMOTE_ROOT'],
    'remote_run_dir': os.environ['RUN_META_REMOTE_RUN_DIR'],
    'threads': int(os.environ['RUN_META_THREADS']),
    'port': int(os.environ['RUN_META_PORT']),
    'model_alias': os.environ['RUN_META_MODEL_ALIAS'],
    'sample_json': os.environ['RUN_META_SAMPLE_JSON'],
    'sample_images_root': os.environ['RUN_META_SAMPLE_IMAGES_ROOT'],
    'throughput_image_rel': os.environ['RUN_META_THROUGHPUT_IMAGE_REL'],
    'probe_only': os.environ['RUN_META_PROBE_ONLY'] == '1',
    'probe_text': os.environ['RUN_META_PROBE_TEXT'],
    'probe_max_tokens': int(os.environ['RUN_META_PROBE_MAX_TOKENS']),
    'probe_timeout_s': float(os.environ['RUN_META_PROBE_TIMEOUT']),
    'probe_poll_interval_s': float(os.environ['RUN_META_PROBE_POLL_INTERVAL']),
    'probe_strace': os.environ['RUN_META_PROBE_STRACE'] == '1',
    'board_init_log': os.environ['RUN_META_BOARD_INIT_LOG'],
    'readiness_log': os.environ['RUN_META_READINESS_LOG'],
    'server_args': ['--log-disable', '--no-warmup'],
    'runtime_env': {
        'MTMD_BACKEND_DEVICE': 'NPU',
        'GGML_NPU_SPM_BYTES': os.environ['RUN_META_NPU_SPM_BYTES'],
        'GGML_NPU_ACC_BYTES': os.environ['RUN_META_NPU_ACC_BYTES'],
        'GGML_NPU_GUARD_BYTES': os.environ['RUN_META_NPU_GUARD_BYTES'],
        'GGML_NPU_STAGE2_K_BYTES': os.environ['RUN_META_NPU_STAGE2_K_BYTES'],
        'NPU_CMA_SIZE': os.environ['RUN_META_NPU_CMA_BYTES'],
    },
}
with open(sys.argv[1], 'w', encoding='utf-8') as handle:
    json.dump(payload, handle, indent=2)
    handle.write('\n')
PY
}

run_remote_eval() {
  local remote_run_dir="$1"
  local remote_sample_json="$2"
  local remote_throughput_image="$3"
  local skip_acc="$4"
  local cache_type_k="$5"
  local cache_type_v="$6"
  local mmproj_file="$7"
  local probe_only="$8"
  local probe_text="$9"
  local probe_max_tokens="${10}"
  local probe_timeout="${11}"
  local probe_poll_interval="${12}"
  local probe_strace="${13}"
  local mmproj_basename
  mmproj_basename="$(basename "$mmproj_file")"
  ssh "${SSH_OPTS[@]}" "$TARGET" "bash -s" <<EOF_RUN
set -euo pipefail
RUN_DIR='$remote_run_dir'
REMOTE_MODEL='$REMOTE_GGUF_DIR/SmolVLM2-500M-Video-Instruct-Q8_0.gguf'
REMOTE_MMPROJ='$REMOTE_GGUF_DIR/$mmproj_basename'
REMOTE_EVAL_DIR='$REMOTE_EVAL_DIR'
REMOTE_DATA_DIR='$REMOTE_DATA_DIR'
REMOTE_LIB_DIR='$REMOTE_LIB_DIR'
REMOTE_SAMPLE_JSON='$remote_sample_json'
REMOTE_THROUGHPUT_IMAGE='$remote_throughput_image'
PORT='$PORT'
THREADS='$THREADS'
MODEL_ALIAS='$MODEL_ALIAS'
CACHE_TYPE_K='$cache_type_k'
CACHE_TYPE_V='$cache_type_v'
PROBE_ONLY='$probe_only'
PROBE_TEXT='$probe_text'
PROBE_MAX_TOKENS='$probe_max_tokens'
PROBE_TIMEOUT='$probe_timeout'
PROBE_POLL_INTERVAL='$probe_poll_interval'
PROBE_STRACE='$probe_strace'
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
CACHE_ARGS=()
if [ -n "\$CACHE_TYPE_K" ]; then
  CACHE_ARGS+=( -ctk "\$CACHE_TYPE_K" )
fi
if [ -n "\$CACHE_TYPE_V" ]; then
  CACHE_ARGS+=( -ctv "\$CACHE_TYPE_V" )
fi
cd "\$RUN_DIR"
rm -f server.log throughput_metrics.json acc_100sample.json probe_result.json probe_slots.json dmesg_before.txt dmesg_after.txt
ENV_ARGS=(
  LD_LIBRARY_PATH="\$REMOTE_LIB_DIR:\${LD_LIBRARY_PATH:-}" \
  MTMD_BACKEND_DEVICE=NPU \
  GGML_NPU_SPM_BYTES="\$NPU_SPM_BYTES" \
  GGML_NPU_ACC_BYTES="\$NPU_ACC_BYTES" \
  GGML_NPU_GUARD_BYTES="\$NPU_GUARD_BYTES" \
  GGML_NPU_STAGE2_K_BYTES="\$NPU_STAGE2_K_BYTES" \
  NPU_CMA_SIZE="\$NPU_CMA_BYTES" \
)
if [ -n "\${GGML_NPU_DEBUG_LOG:-}" ]; then
  ENV_ARGS+=( GGML_NPU_DEBUG_LOG="\${GGML_NPU_DEBUG_LOG:-}" )
fi
if [ -n "\${AICAS_MMPROJ_W8A8_DEBUG:-}" ]; then
  ENV_ARGS+=( AICAS_MMPROJ_W8A8_DEBUG="\${AICAS_MMPROJ_W8A8_DEBUG:-}" )
fi
if [ -n "\${GGML_NPU_AICAS_BIAS_MODE:-}" ]; then
  ENV_ARGS+=( GGML_NPU_AICAS_BIAS_MODE="\${GGML_NPU_AICAS_BIAS_MODE:-}" )
fi
if [ -n "\${MTMD_DEBUG_GRAPH:-}" ]; then
  ENV_ARGS+=( MTMD_DEBUG_GRAPH="\${MTMD_DEBUG_GRAPH:-}" )
fi
if [ -n "\${MTMD_MAX_NODES:-}" ]; then
  ENV_ARGS+=( MTMD_MAX_NODES="\${MTMD_MAX_NODES:-}" )
fi
if [ -n "\${MTMD_DUMP_DOT:-}" ]; then
  ENV_ARGS+=( MTMD_DUMP_DOT="\${MTMD_DUMP_DOT:-}" )
fi
if [ -n "\${AICAS_MMPROJ_DEQUANT_STATS_FILE:-}" ]; then
  ENV_ARGS+=( AICAS_MMPROJ_DEQUANT_STATS_FILE="\${AICAS_MMPROJ_DEQUANT_STATS_FILE:-}" )
fi
if [ -n "\${AICAS_MMPROJ_ACT_STATS_FILE:-}" ]; then
  ENV_ARGS+=( AICAS_MMPROJ_ACT_STATS_FILE="\${AICAS_MMPROJ_ACT_STATS_FILE:-}" )
fi
if [ -n "\${AICAS_MMPROJ_ACT_SAMPLES:-}" ]; then
  ENV_ARGS+=( AICAS_MMPROJ_ACT_SAMPLES="\${AICAS_MMPROJ_ACT_SAMPLES:-}" )
fi
SERVER_CMD=(
  ./llama-server \
    --log-disable \
    --no-warmup \
    --host 127.0.0.1 \
    --port "\$PORT" \
    --alias "\$MODEL_ALIAS" \
    -m "\$REMOTE_MODEL" \
    --mmproj "\$REMOTE_MMPROJ" \
    -t "\$THREADS" \
    "\${CACHE_ARGS[@]}"
)
if [ "\$PROBE_STRACE" -eq 1 ]; then
  env "\${ENV_ARGS[@]}" strace -ff -tt -o strace "\${SERVER_CMD[@]}" > server.log 2>&1 &
else
  env "\${ENV_ARGS[@]}" "\${SERVER_CMD[@]}" > server.log 2>&1 &
fi
SERVER_PID=\$!
READY=0
for _ in \$(seq 1 90); do
  if python3 - <<PY
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
if command -v dmesg >/dev/null 2>&1; then
  dmesg | tail -n 200 > "\$RUN_DIR/dmesg_before.txt" 2>/dev/null || true
fi
if [ "\$PROBE_ONLY" -eq 1 ]; then
  python3 "\$REMOTE_EVAL_DIR/mmproj_probe.py" \
    --image "\$REMOTE_THROUGHPUT_IMAGE" \
    --output "\$RUN_DIR/probe_result.json" \
    --slots-output "\$RUN_DIR/probe_slots.json" \
    --base-url "http://127.0.0.1:\$PORT/v1" \
    --model "\$MODEL_ALIAS" \
    --prompt "\$PROBE_TEXT" \
    --max-tokens "\$PROBE_MAX_TOKENS" \
    --request-timeout "\$PROBE_TIMEOUT" \
    --poll-interval "\$PROBE_POLL_INTERVAL"
else
  python3 "\$REMOTE_EVAL_DIR/throughput_eval.py" \
    -i "\$REMOTE_THROUGHPUT_IMAGE" \
    -o "\$RUN_DIR/throughput_metrics.json" \
    --base-url "http://127.0.0.1:\$PORT/v1" \
    --model "\$MODEL_ALIAS"
  if [ "$skip_acc" -eq 1 ]; then
    echo "[run_submission] skipped acc_eval by request"
  else
  python3 "\$REMOTE_EVAL_DIR/acc_eval.py" \
    --image_folder "\$REMOTE_DATA_DIR" \
    --OCRBench_file "\$REMOTE_SAMPLE_JSON" \
    --output_folder "\$RUN_DIR" \
    --save_name acc_100sample \
    --base-url "http://127.0.0.1:\$PORT/v1" \
    --model "\$MODEL_ALIAS" \
    --request-timeout 600 \
    --progress-every 10
  fi
fi
if command -v dmesg >/dev/null 2>&1; then
  dmesg | tail -n 200 > "\$RUN_DIR/dmesg_after.txt" 2>/dev/null || true
fi
EOF_RUN
}

if [ "$SKIP_VALIDATE" -ne 1 ]; then
  bash "$VALIDATE_SCRIPT"
fi

SAMPLE_JSON="$(realpath "$SAMPLE_JSON")"
SAMPLE_IMAGES_ROOT="$(realpath "$SAMPLE_IMAGES_ROOT")"

if [ -z "$RUN_ID" ]; then
  RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-versavlm-release"
fi

LOCAL_RUN_DIR="$RESULTS_ROOT/$RUN_ID"
REMOTE_RUN_DIR="$REMOTE_RUNS_DIR/$RUN_ID"
LOCAL_MANIFEST="$LOCAL_RUN_DIR/sample_manifest.txt"
mkdir -p "$LOCAL_RUN_DIR"

build_custom_manifest "$SAMPLE_JSON" "$SAMPLE_IMAGES_ROOT" "$LOCAL_MANIFEST"

ssh "${SSH_OPTS[@]}" "$TARGET" "mkdir -p '$REMOTE_ROOT' '$REMOTE_SHARED_DIR' '$REMOTE_GGUF_DIR' '$REMOTE_EVAL_DIR' '$REMOTE_DATA_DIR' '$REMOTE_LIB_DIR' '$REMOTE_BOARD_OVERLAY_DIR' '$REMOTE_BOARD_DRIVER_DIR' '$REMOTE_BOARD_QSPI_DIR' '$REMOTE_BOARD_TESTS_DIR/bin' '$REMOTE_BOARD_TESTS_DIR/src' '$REMOTE_SCRIPTS_DIR' '$REMOTE_RUN_DIR'"

sync_file_if_needed "$PAYLOAD_DIR/gguf/SmolVLM2-500M-Video-Instruct-Q8_0.gguf" "$REMOTE_GGUF_DIR/SmolVLM2-500M-Video-Instruct-Q8_0.gguf" "Q8_0 model"
sync_file_if_needed "$MMPROJ_FILE" "$REMOTE_GGUF_DIR/$(basename "$MMPROJ_FILE")" "mmproj $(basename "$MMPROJ_FILE")"
sync_file_if_needed "$PAYLOAD_DIR/eval/throughput_eval.py" "$REMOTE_EVAL_DIR/throughput_eval.py" "throughput_eval.py"
sync_file_if_needed "$PAYLOAD_DIR/eval/acc_eval.py" "$REMOTE_EVAL_DIR/acc_eval.py" "acc_eval.py"
sync_file_if_needed "$PAYLOAD_DIR/eval/llama_server_client.py" "$REMOTE_EVAL_DIR/llama_server_client.py" "llama_server_client.py"
sync_file_if_needed "$PAYLOAD_DIR/eval/mmproj_probe.py" "$REMOTE_EVAL_DIR/mmproj_probe.py" "mmproj_probe.py"
sync_file_if_needed "$PAYLOAD_DIR/board_support/driver/npu_kv260.ko" "$REMOTE_BOARD_DRIVER_DIR/npu_kv260.ko" "npu_kv260.ko"
for overlay_file in "$PAYLOAD_DIR/board_support/overlay/double_dma_overlayapp"/*; do
  [ -e "$overlay_file" ] || continue
  sync_file_if_needed "$overlay_file" "$REMOTE_BOARD_OVERLAY_DIR/$(basename "$overlay_file")" "overlay $(basename "$overlay_file")"
done
sync_file_if_needed "$PAYLOAD_DIR/board_support/qspi/BOOT-k26-smk-sdt-v1.05-20250912165210.bin" "$REMOTE_BOARD_QSPI_DIR/BOOT-k26-smk-sdt-v1.05-20250912165210.bin" "QSPI boot image"
for test_file in "$PAYLOAD_DIR/board_support/tests/bin"/* "$PAYLOAD_DIR/board_support/tests/src"/*; do
  [ -e "$test_file" ] || continue
  if [[ "$test_file" == *"/bin/"* ]]; then
    sync_file_if_needed "$test_file" "$REMOTE_BOARD_TESTS_DIR/bin/$(basename "$test_file")" "test $(basename "$test_file")"
  else
    sync_file_if_needed "$test_file" "$REMOTE_BOARD_TESTS_DIR/src/$(basename "$test_file")" "test source $(basename "$test_file")"
  fi
done
sync_file_if_needed "$PAYLOAD_DIR/bin/llama-server-kv260-npu" "$REMOTE_RUN_DIR/llama-server" "llama-server"
sync_file_if_needed "$BOARD_INIT_SCRIPT" "$REMOTE_SCRIPTS_DIR/board_init_overlay.sh" "board_init_overlay.sh"
sync_file_if_needed "$SCRIPT_DIR/install_xmutil_app.sh" "$REMOTE_SCRIPTS_DIR/install_xmutil_app.sh" "install_xmutil_app.sh"
sync_file_if_needed "$REPLAY_SCRIPT" "$REMOTE_SCRIPTS_DIR/run_layer_gemm_replay.sh" "run_layer_gemm_replay.sh"
for lib in "$PAYLOAD_DIR"/runtime_libs/*; do
  [ -e "$lib" ] || continue
  sync_file_if_needed "$lib" "$REMOTE_LIB_DIR/$(basename "$lib")" "runtime lib $(basename "$lib")"
done

REMOTE_SAMPLE_JSON="$REMOTE_EVAL_DIR/$(basename "$SAMPLE_JSON")"
sync_sample_payload "$SAMPLE_JSON" "$SAMPLE_IMAGES_ROOT" "$REMOTE_SAMPLE_JSON" "$REMOTE_DATA_DIR" "$LOCAL_MANIFEST"
ssh "${SSH_OPTS[@]}" "$TARGET" "chmod +x '$REMOTE_RUN_DIR/llama-server' '$REMOTE_SCRIPTS_DIR/board_init_overlay.sh' '$REMOTE_SCRIPTS_DIR/install_xmutil_app.sh'"

if [ "$SKIP_BOARD_INIT" -ne 1 ]; then
  [ -n "$SUDO_PASSWORD" ] || { echo "Missing board sudo password: use --sudo-password or export BOARD_SUDO_PASSWORD" >&2; exit 1; }
  password_escaped="$(printf '%q' "$SUDO_PASSWORD")"
  ssh "${SSH_OPTS[@]}" "$TARGET" "bash -lc 'BOARD_SUDO_PASSWORD=$password_escaped \"$REMOTE_SCRIPTS_DIR/board_init_overlay.sh\" --remote-root \"$REMOTE_ROOT\" --output \"$REMOTE_RUN_DIR/board_init.log\"'"
fi

probe_remote_ready > "$LOCAL_RUN_DIR/readiness_probe.txt"
REMOTE_RC=0
  if run_remote_eval "$REMOTE_RUN_DIR" "$REMOTE_SAMPLE_JSON" "$REMOTE_DATA_DIR/$THROUGHPUT_IMAGE_REL" "$SKIP_ACC" "$CACHE_TYPE_K" "$CACHE_TYPE_V" "$MMPROJ_FILE" "$PROBE_ONLY" "$PROBE_TEXT" "$PROBE_MAX_TOKENS" "$PROBE_TIMEOUT" "$PROBE_POLL_INTERVAL" "$PROBE_STRACE"; then
  REMOTE_RC=0
else
  REMOTE_RC=$?
fi

for artifact in throughput_metrics.json acc_100sample.json server.log board_init.log run_meta.json probe_result.json probe_slots.json dmesg_before.txt dmesg_after.txt; do
  scp "${SSH_OPTS[@]}" "$TARGET:$REMOTE_RUN_DIR/$artifact" "$LOCAL_RUN_DIR/" >/dev/null 2>&1 || true
done
scp "${SSH_OPTS[@]}" "$TARGET:$REMOTE_RUN_DIR/strace*" "$LOCAL_RUN_DIR/" >/dev/null 2>&1 || true

run_status="completed"
if [ "$REMOTE_RC" -ne 0 ]; then
  run_status="failed"
fi
write_run_meta "$LOCAL_RUN_DIR" "$REMOTE_RUN_DIR" "$run_status" "$REMOTE_RC" "$LOCAL_RUN_DIR/board_init.log" "$LOCAL_RUN_DIR/readiness_probe.txt"
scp "${SSH_OPTS[@]}" "$LOCAL_RUN_DIR/run_meta.json" "$TARGET:$REMOTE_RUN_DIR/run_meta.json" >/dev/null 2>&1 || true

if [ -f "$LOCAL_RUN_DIR/probe_result.json" ]; then
  echo "[probe_result.json]"
  cat "$LOCAL_RUN_DIR/probe_result.json"
fi

if [ -f "$LOCAL_RUN_DIR/throughput_metrics.json" ]; then
  echo "[throughput_metrics.json]"
  cat "$LOCAL_RUN_DIR/throughput_metrics.json"
fi

if [ "$PROBE_ONLY" -eq 1 ]; then
  echo "[mode] probe-only"
elif [ "$SKIP_ACC" -eq 1 ] && [ ! -f "$LOCAL_RUN_DIR/acc_100sample.json" ]; then
  echo "[acc] skipped by --skip-acc"
fi

if [ -f "$LOCAL_RUN_DIR/throughput_metrics.json" ] && [ -f "$LOCAL_RUN_DIR/acc_100sample.json" ]; then
  python3 "$SUMMARY_SCRIPT" \
    --throughput "$LOCAL_RUN_DIR/throughput_metrics.json" \
    --acc-json "$LOCAL_RUN_DIR/acc_100sample.json" \
    --run-meta "$LOCAL_RUN_DIR/run_meta.json" \
    --output-json "$LOCAL_RUN_DIR/summary_metrics.json" \
    --output-zh "$LOCAL_RUN_DIR/RESULTS.zh-CN.md" \
    --output-en "$LOCAL_RUN_DIR/RESULTS.en.md"
fi

bash "$VALIDATE_SCRIPT" --update-manifests

echo "Local results directory: $LOCAL_RUN_DIR"
echo "Remote run directory: $REMOTE_RUN_DIR"

if [ "$REMOTE_RC" -ne 0 ]; then
  echo "Remote evaluation failed. Check $LOCAL_RUN_DIR for partial artifacts." >&2
  exit "$REMOTE_RC"
fi
