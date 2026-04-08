#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  deploy_and_run_kv260_baseline.sh [--host <host>] [--user <user>]
                                   [--remote-dir <dir>] [--results-dir <dir>]

Deploys the local AICAS baseline payload to KV260, patches the remote copy,
starts llama-server, runs throughput and 100-sample accuracy, then pulls
results back to the host.
EOF
}

HOST="192.168.0.10"
USER_NAME="petalinux"
REMOTE_DIR="~/npux/aicas_official_baseline_20260407"
RESULTS_DIR=""

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
    --remote-dir)
      REMOTE_DIR="$2"
      shift 2
      ;;
    --results-dir)
      RESULTS_DIR="$2"
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

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -z "$RESULTS_DIR" ]; then
  RESULTS_DIR="$ROOT_DIR/kv260_results/2026-04-07-baseline-sampled"
fi

LIB_DIR="$ROOT_DIR/third_party/kv260_runtime/lib"
PREP_SCRIPT="$ROOT_DIR/scripts/prepare_kv260_runtime_libs.sh"

for path in "$ROOT_DIR/bin" "$ROOT_DIR/gguf" "$ROOT_DIR/data" "$ROOT_DIR/acc_eval.py" "$ROOT_DIR/throughput_eval.py" "$ROOT_DIR/sampled.json"; do
  [ -e "$path" ] || { echo "Missing required path: $path" >&2; exit 1; }
done

if [ ! -d "$LIB_DIR" ] || [ -z "$(find "$LIB_DIR" -maxdepth 1 \( -type f -o -type l \) -print -quit)" ]; then
  "$PREP_SCRIPT" --root "$ROOT_DIR"
fi

mkdir -p "$RESULTS_DIR"

REMOTE_DIR_EXPANDED="$(ssh "${USER_NAME}@${HOST}" "eval printf '%s' $REMOTE_DIR")"
echo "Remote directory: $REMOTE_DIR_EXPANDED"

ssh "${USER_NAME}@${HOST}" "mkdir -p '$REMOTE_DIR_EXPANDED/results' '$REMOTE_DIR_EXPANDED/third_party/lib'"

ssh "${USER_NAME}@${HOST}" "rm -rf \
  '$REMOTE_DIR_EXPANDED/bin' \
  '$REMOTE_DIR_EXPANDED/gguf' \
  '$REMOTE_DIR_EXPANDED/data' \
  '$REMOTE_DIR_EXPANDED/third_party/lib' && \
  rm -f \
  '$REMOTE_DIR_EXPANDED/acc_eval.py' \
  '$REMOTE_DIR_EXPANDED/throughput_eval.py' \
  '$REMOTE_DIR_EXPANDED/sampled.json' \
  '$REMOTE_DIR_EXPANDED/server.log' \
  '$REMOTE_DIR_EXPANDED/server.pid' \
  '$REMOTE_DIR_EXPANDED/results/throughput_metrics.json' \
  '$REMOTE_DIR_EXPANDED/results/SmolVLM2.json' && \
  mkdir -p '$REMOTE_DIR_EXPANDED/results' '$REMOTE_DIR_EXPANDED/third_party/lib'"

tar -C "$ROOT_DIR" -cf - \
  bin \
  gguf \
  data \
  acc_eval.py \
  throughput_eval.py \
  sampled.json | \
  ssh "${USER_NAME}@${HOST}" "cd '$REMOTE_DIR_EXPANDED' && tar -xmf -"

tar -C "$LIB_DIR" -cf - . | \
  ssh "${USER_NAME}@${HOST}" "cd '$REMOTE_DIR_EXPANDED/third_party/lib' && tar -xmf -"

ssh "${USER_NAME}@${HOST}" "python3 - <<'PY'
from pathlib import Path
path = Path('$REMOTE_DIR_EXPANDED/throughput_eval.py')
text = path.read_text(encoding='utf-8')
old = 'model=\"local-model\"'
new = 'model=\"smolvlm2-gguf\"'
if old not in text and new not in text:
    raise SystemExit('Expected model token not found in throughput_eval.py')
path.write_text(text.replace(old, new), encoding='utf-8')
PY"

ssh "${USER_NAME}@${HOST}" "cd '$REMOTE_DIR_EXPANDED' && \
  if LD_LIBRARY_PATH='$REMOTE_DIR_EXPANDED/bin:$REMOTE_DIR_EXPANDED/third_party/lib' \
       ldd ./bin/llama-server | grep -q 'not found'; then \
    LD_LIBRARY_PATH='$REMOTE_DIR_EXPANDED/bin:$REMOTE_DIR_EXPANDED/third_party/lib' \
      ldd ./bin/llama-server; \
    exit 1; \
  fi"

ssh "${USER_NAME}@${HOST}" "bash -s" <<EOF
set -euo pipefail
cd '$REMOTE_DIR_EXPANDED'

cleanup() {
  if [ -n "\${SERVER_PID:-}" ]; then
    kill "\$SERVER_PID" >/dev/null 2>&1 || true
    wait "\$SERVER_PID" >/dev/null 2>&1 || true
  fi
  rm -f server.pid
}
trap cleanup EXIT

rm -f server.log server.pid

env \
  LD_LIBRARY_PATH='$REMOTE_DIR_EXPANDED/bin:$REMOTE_DIR_EXPANDED/third_party/lib' \
  ./bin/llama-server \
    --host 127.0.0.1 \
    --port 8080 \
    --alias smolvlm2-gguf \
    -m ./gguf/SmolVLM2-500M-Video-Instruct-Q8_0.gguf \
    --mmproj ./gguf/mmproj-SmolVLM2-500M-Video-Instruct-Q8_0.gguf \
    -t 4 \
    > server.log 2>&1 &
SERVER_PID=\$!
echo "\$SERVER_PID" > server.pid

READY=0
for _ in \$(seq 1 60); do
  if python3 - <<'PY'
import json
import urllib.request
with urllib.request.urlopen('http://127.0.0.1:8080/v1/models', timeout=5) as resp:
    payload = json.load(resp)
assert any(item.get('id') == 'smolvlm2-gguf' for item in payload.get('data', [])), payload
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

python3 throughput_eval.py \
  -i ./data/IIIT5K/test/2543_2.png \
  -o ./results/throughput_metrics.json

python3 acc_eval.py \
  --image_folder ./data \
  --OCRBench_file ./sampled.json \
  --output_folder ./results \
  --save_name SmolVLM2
EOF

scp \
  "${USER_NAME}@${HOST}:$REMOTE_DIR_EXPANDED/results/throughput_metrics.json" \
  "${USER_NAME}@${HOST}:$REMOTE_DIR_EXPANDED/results/SmolVLM2.json" \
  "${USER_NAME}@${HOST}:$REMOTE_DIR_EXPANDED/server.log" \
  "$RESULTS_DIR/"

echo
echo "Pulled results into: $RESULTS_DIR"
