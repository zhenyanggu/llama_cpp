#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_semi_eval.sh [options]

Options:
  --full-test-json <path>
  --image-root <path>
  --throughput-image <path>
  --ttft-config <path>
  --output-dir <path>
  --base-url <url>
  --model <name>
  --power-path <path>
  --sample-hz <hz>
  --acc-ori <ratio>
USAGE
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CODE_DIR="$ROOT_DIR/AICAS2026/aicas_semi/code"

FULL_TEST_JSON="$ROOT_DIR/AICAS/FullTest.json"
IMAGE_ROOT="$ROOT_DIR/AICAS2026/V3/payload/data/images"
THROUGHPUT_IMAGE="$ROOT_DIR/AICAS2026/aicas_semi/code/test2.jpg"
TTFT_CONFIG="$CODE_DIR/ttft_config.json"
OUTPUT_DIR="$ROOT_DIR/AICAS2026/aicas_semi/results/latest"
BASE_URL="http://127.0.0.1:8080/v1"
MODEL="local-model"
POWER_PATH="/sys/class/hwmon/hwmon2/power1_input"
SAMPLE_HZ="100"
ACC_ORI=""

while [ $# -gt 0 ]; do
  case "$1" in
    --full-test-json) FULL_TEST_JSON="$2"; shift 2 ;;
    --image-root) IMAGE_ROOT="$2"; shift 2 ;;
    --throughput-image) THROUGHPUT_IMAGE="$2"; shift 2 ;;
    --ttft-config) TTFT_CONFIG="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --base-url) BASE_URL="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --power-path) POWER_PATH="$2"; shift 2 ;;
    --sample-hz) SAMPLE_HZ="$2"; shift 2 ;;
    --acc-ori) ACC_ORI="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

mkdir -p "$OUTPUT_DIR"

python3 "$CODE_DIR/sample.py" -i "$FULL_TEST_JSON" -o "$OUTPUT_DIR/sample_30.json"
python3 "$CODE_DIR/acc_eval.py" \
  -i "$IMAGE_ROOT" \
  -d "$OUTPUT_DIR/sample_30.json" \
  -o "$OUTPUT_DIR/acc_eval_results.json" \
  --base-url "$BASE_URL" \
  --model "$MODEL"
python3 "$CODE_DIR/throughput_eval.py" \
  -i "$THROUGHPUT_IMAGE" \
  -o "$OUTPUT_DIR/throughput_metrics.json" \
  --base-url "$BASE_URL" \
  --model "$MODEL"
python3 "$CODE_DIR/energy_eval.py" \
  -i "$THROUGHPUT_IMAGE" \
  -o "$OUTPUT_DIR/energy_metrics.json" \
  --sample_hz "$SAMPLE_HZ" \
  --power_path "$POWER_PATH" \
  --base-url "$BASE_URL" \
  --model "$MODEL"
python3 "$CODE_DIR/ttft_eval_multiprompt.py" \
  -c "$TTFT_CONFIG" \
  -o "$OUTPUT_DIR/ttft_eval_results.json"
python3 "$CODE_DIR/merge_results.py" \
  --acc "$OUTPUT_DIR/acc_eval_results.json" \
  --throughput "$OUTPUT_DIR/throughput_metrics.json" \
  --energy "$OUTPUT_DIR/energy_metrics.json" \
  --ttft "$OUTPUT_DIR/ttft_eval_results.json" \
  -o "$OUTPUT_DIR/aicas_submission.json"

score_args=(--merged "$OUTPUT_DIR/aicas_submission.json" --output "$OUTPUT_DIR/score_report.json")
if [ -n "$ACC_ORI" ]; then
  score_args+=(--acc-ori "$ACC_ORI")
fi
python3 "$CODE_DIR/score_submission.py" "${score_args[@]}"

echo "Results: $OUTPUT_DIR"
