#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_v2_eval.sh [options]

Run KV260 NPU eval with V2 output layout.

Options:
  --host <host>              KV260 host (default: 192.168.0.10)
  --user <user>              KV260 user (default: ubuntu)
  --remote-root <path>       Remote root (default: /home/ubuntu/aicas)
  --run-id <id>              Run id (default: auto UTC timestamp)
  --samples <5|100>          Dataset size (default: 5)
  --build-dir <path>         Cross-build directory (default: build-kv260-npu-current)
  --skip-build               Reuse existing build
  --skip-readiness-probe     Skip xmutil readiness probe
  -h, --help                 Show this help

Fixed artifacts in V2:
  text model: SmolVLM2-500M-Video-Instruct-Q8_0.gguf
  mmproj:    mmproj-fallback-search-fb_attn_k-per-tensor.gguf
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
HOST="192.168.0.10"
USER_NAME="ubuntu"
REMOTE_ROOT="/home/ubuntu/aicas"
RUN_ID=""
SAMPLES="5"
SKIP_BUILD=0
SKIP_READINESS=1
BUILD_DIR="$ROOT_DIR/build-kv260-npu-current"
MODEL_GGUF="$ROOT_DIR/AICAS/gguf/SmolVLM2-500M-Video-Instruct-Q8_0.gguf"
MMPROJ_GGUF="$ROOT_DIR/AICAS/gguf/mmproj-fallback-search-fb_attn_k-per-tensor.gguf"

while [ $# -gt 0 ]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --user) USER_NAME="$2"; shift 2 ;;
    --remote-root) REMOTE_ROOT="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --samples) SAMPLES="$2"; shift 2 ;;
    --build-dir) BUILD_DIR="$2"; shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --skip-readiness-probe) SKIP_READINESS=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [ -z "$RUN_ID" ]; then
  RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-v2-acc${SAMPLES}"
fi

OCR_FILE="$ROOT_DIR/AICAS2026/V1_412/kv260_sampled5.json"
if [ "$SAMPLES" = "100" ]; then
  OCR_FILE="$ROOT_DIR/AICAS/sampled.json"
fi

CMD=(
  bash "$ROOT_DIR/AICAS/scripts/deploy_and_run_kv260_npu_eval.sh"
  --host "$HOST"
  --user "$USER_NAME"
  --remote-root "$REMOTE_ROOT"
  --results-dir "$ROOT_DIR/AICAS2026/V2/results/official"
  --build-dir "$BUILD_DIR"
  --model "$MODEL_GGUF"
  --mmproj "$MMPROJ_GGUF"
  --ocrbench-file "$OCR_FILE"
  --run-id "$RUN_ID"
)

if [ "$SKIP_BUILD" -eq 1 ]; then
  CMD+=(--skip-build)
fi
if [ "$SKIP_READINESS" -eq 1 ]; then
  CMD+=(--skip-readiness-probe)
fi

"${CMD[@]}"
