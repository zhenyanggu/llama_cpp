#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_semi_eval.sh [options]

Options:
  --root-dir <path>
  --code-dir <path>
  --full-test-json <path>
  --image-root <path>
  --throughput-image <path>
  --throughput-max-tokens <n>     Max generated tokens for throughput test
  --ttft-config <path>
  --output-dir <path>
  --base-url <url>
  --model <name>
  --power-path <path>
  --sample-hz <hz>
  --energy-max-tokens <n>         Max generated tokens for energy test
  --sample-json <path>
  --acc-sample-json <path>        Accuracy sample JSON to test
  --acc-only
  --fast-acc                     Run only the first accuracy sample
  --run-throughput-profile
  --throughput-profile-only
  --throughput-profile-output <path>
  --throughput-profile-max-tokens <n>
  --throughput-profile-prompt <text>
  --throughput-profile-artifacts <path>
  --mtmd-summary-json <path>
  --npu-profile-json <path>
  --npu-profile-level <level>
  --run-acc                      Run accuracy (default)
  --skip-acc                     Skip accuracy
  --acc-max-tokens <n>            Max generated tokens for accuracy requests (default: 100)
  --acc-cache-prompt              Send cache_prompt=true for accuracy requests
  --acc-no-cache-prompt           Send cache_prompt=false for accuracy requests
  --ttft-cache-prompt             Send cache_prompt=true for TTFT requests
  --no-ttft-cache-prompt          Send cache_prompt=false for TTFT requests (default)
  --stop-after-throughput         Stop after acc + throughput, skipping energy/ttft/merge
  --acc-ori <ratio>
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

if [ -d "$SCRIPT_DIR/../code" ]; then
  CODE_DIR="$(cd "$SCRIPT_DIR/../code" && pwd)"
else
  CODE_DIR="$ROOT_DIR/AICAS2026/aicas_semi/code"
fi

FULL_TEST_JSON="$ROOT_DIR/AICAS/FullTest.json"
IMAGE_ROOT="$ROOT_DIR/AICAS2026/V3/payload/data/images"
THROUGHPUT_IMAGE="$CODE_DIR/test2.jpg"
THROUGHPUT_MAX_TOKENS="1024"
TTFT_CONFIG="$CODE_DIR/ttft_config.json"
OUTPUT_DIR="$ROOT_DIR/AICAS2026/aicas_semi/results/latest"
BASE_URL="http://127.0.0.1:8080/v1"
MODEL="local-model"
POWER_PATH="/sys/class/hwmon/hwmon2/power1_input"
SAMPLE_HZ="100"
ENERGY_MAX_TOKENS="1024"
SAMPLE_JSON=""
ACC_ONLY=0
FAST_ACC=0
RUN_THROUGHPUT_PROFILE=0
THROUGHPUT_PROFILE_ONLY=0
THROUGHPUT_PROFILE_OUTPUT=""
THROUGHPUT_PROFILE_MAX_TOKENS="1"
THROUGHPUT_PROFILE_PROMPT="Describe this image briefly."
THROUGHPUT_PROFILE_ARTIFACTS=""
MTMD_SUMMARY_JSON=""
NPU_PROFILE_JSON=""
NPU_PROFILE_LEVEL="diagnostic"
RUN_ACC=1
ACC_MAX_TOKENS="100"
ACC_CACHE_PROMPT=""
TTFT_CACHE_PROMPT="--no-cache-prompt"
STOP_AFTER_THROUGHPUT=0
ACC_ORI=""

while [ $# -gt 0 ]; do
  case "$1" in
    --root-dir) ROOT_DIR="$2"; shift 2 ;;
    --code-dir) CODE_DIR="$2"; shift 2 ;;
    --full-test-json) FULL_TEST_JSON="$2"; shift 2 ;;
    --image-root) IMAGE_ROOT="$2"; shift 2 ;;
    --throughput-image) THROUGHPUT_IMAGE="$2"; shift 2 ;;
    --throughput-max-tokens) THROUGHPUT_MAX_TOKENS="$2"; shift 2 ;;
    --ttft-config) TTFT_CONFIG="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --base-url) BASE_URL="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --power-path) POWER_PATH="$2"; shift 2 ;;
    --sample-hz) SAMPLE_HZ="$2"; shift 2 ;;
    --energy-max-tokens) ENERGY_MAX_TOKENS="$2"; shift 2 ;;
    --sample-json|--acc-sample-json) SAMPLE_JSON="$2"; shift 2 ;;
    --acc-only) ACC_ONLY=1; RUN_ACC=1; shift ;;
    --fast-acc) FAST_ACC=1; shift ;;
    --run-throughput-profile) RUN_THROUGHPUT_PROFILE=1; shift ;;
    --throughput-profile-only) RUN_THROUGHPUT_PROFILE=1; THROUGHPUT_PROFILE_ONLY=1; shift ;;
    --throughput-profile-output) THROUGHPUT_PROFILE_OUTPUT="$2"; shift 2 ;;
    --throughput-profile-max-tokens) THROUGHPUT_PROFILE_MAX_TOKENS="$2"; shift 2 ;;
    --throughput-profile-prompt) THROUGHPUT_PROFILE_PROMPT="$2"; shift 2 ;;
    --throughput-profile-artifacts) THROUGHPUT_PROFILE_ARTIFACTS="$2"; shift 2 ;;
    --mtmd-summary-json) MTMD_SUMMARY_JSON="$2"; shift 2 ;;
    --npu-profile-json) NPU_PROFILE_JSON="$2"; shift 2 ;;
    --npu-profile-level) NPU_PROFILE_LEVEL="$2"; shift 2 ;;
    --run-acc) RUN_ACC=1; shift ;;
    --skip-acc) RUN_ACC=0; shift ;;
    --acc-max-tokens) ACC_MAX_TOKENS="$2"; shift 2 ;;
    --acc-cache-prompt) ACC_CACHE_PROMPT="--cache-prompt"; shift ;;
    --acc-no-cache-prompt) ACC_CACHE_PROMPT="--no-cache-prompt"; shift ;;
    --ttft-cache-prompt) TTFT_CACHE_PROMPT="--cache-prompt"; shift ;;
    --no-ttft-cache-prompt) TTFT_CACHE_PROMPT="--no-cache-prompt"; shift ;;
    --stop-after-throughput) STOP_AFTER_THROUGHPUT=1; shift ;;
    --acc-ori) ACC_ORI="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

mkdir -p "$OUTPUT_DIR"

log_stage() {
  printf '\n[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

TEST_STAGE_COMPLETED=0

mark_test_completed() {
  TEST_STAGE_COMPLETED=1
}

sleep_before_next_test() {
  if [ "$TEST_STAGE_COMPLETED" -eq 1 ]; then
    sleep 1
  fi
}

if [ "$RUN_ACC" -eq 1 ]; then
  log_stage "Starting sample preparation for accuracy"
  if [ -z "$SAMPLE_JSON" ]; then
    SAMPLE_JSON="$OUTPUT_DIR/sample_30.json"
    python3 "$CODE_DIR/sample.py" -i "$FULL_TEST_JSON" -o "$SAMPLE_JSON"
  fi
  if [ "$FAST_ACC" -eq 1 ]; then
    FAST_SAMPLE_JSON="$OUTPUT_DIR/sample_fast_acc_1.json"
    python3 - "$SAMPLE_JSON" "$FAST_SAMPLE_JSON" <<'PY'
import json
import sys

src, dst = sys.argv[1:3]
with open(src, "r", encoding="utf-8") as handle:
    items = json.load(handle)
if not items:
    raise SystemExit(f"accuracy sample file is empty: {src}")
with open(dst, "w", encoding="utf-8") as handle:
    json.dump(items[:1], handle, ensure_ascii=False, indent=2)
PY
    SAMPLE_JSON="$FAST_SAMPLE_JSON"
    log_stage "Fast acc enabled: using first accuracy sample only ($SAMPLE_JSON)"
  fi

  log_stage "Checking sampled accuracy images"
  python3 - "$IMAGE_ROOT" "$SAMPLE_JSON" <<'PY'
import json
import os
import sys

image_root = sys.argv[1]
sample_json = sys.argv[2]

with open(sample_json, "r", encoding="utf-8") as handle:
    items = json.load(handle)

missing = []
for item in items:
    image_path = item.get("image_path")
    if not image_path:
        continue
    full_path = os.path.join(image_root, image_path)
    if not os.path.exists(full_path):
        missing.append(image_path)
        if len(missing) >= 10:
            break

if missing:
    print("Sampled accuracy set contains missing images:", file=sys.stderr)
    for rel in missing:
        print(rel, file=sys.stderr)
    raise SystemExit(1)
PY

  log_stage "Starting accuracy evaluation"
  python3 "$CODE_DIR/acc_eval.py" \
    -i "$IMAGE_ROOT" \
    -d "$SAMPLE_JSON" \
    -o "$OUTPUT_DIR/acc_eval_results.json" \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --max-tokens "$ACC_MAX_TOKENS" \
    ${ACC_CACHE_PROMPT:+"$ACC_CACHE_PROMPT"}
  log_stage "Finished accuracy evaluation: $OUTPUT_DIR/acc_eval_results.json"
  mark_test_completed
fi

if [ "$ACC_ONLY" -eq 1 ]; then
  echo "Accuracy results: $OUTPUT_DIR/acc_eval_results.json"
  exit 0
fi

if [ "$THROUGHPUT_PROFILE_ONLY" -ne 1 ]; then
  sleep_before_next_test
  log_stage "Starting throughput evaluation"
  python3 "$CODE_DIR/throughput_eval.py" \
    -i "$THROUGHPUT_IMAGE" \
    -o "$OUTPUT_DIR/throughput_metrics.json" \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --max-tokens "$THROUGHPUT_MAX_TOKENS"
  log_stage "Finished throughput evaluation: $OUTPUT_DIR/throughput_metrics.json"
  mark_test_completed
fi

if [ "$RUN_THROUGHPUT_PROFILE" -eq 1 ]; then
  if [ -z "$THROUGHPUT_PROFILE_OUTPUT" ]; then
    THROUGHPUT_PROFILE_OUTPUT="$OUTPUT_DIR/throughput_metrics_profile.json"
  fi
  if [ -z "$THROUGHPUT_PROFILE_ARTIFACTS" ]; then
    THROUGHPUT_PROFILE_ARTIFACTS="$OUTPUT_DIR/throughput_profile_artifacts.json"
  fi

  sleep_before_next_test
  log_stage "Starting throughput profile evaluation"
  python3 "$CODE_DIR/throughput_eval.py" \
    -i "$THROUGHPUT_IMAGE" \
    -o "$THROUGHPUT_PROFILE_OUTPUT" \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --max-tokens "$THROUGHPUT_PROFILE_MAX_TOKENS" \
    --prompt "$THROUGHPUT_PROFILE_PROMPT"
  log_stage "Finished throughput profile evaluation: $THROUGHPUT_PROFILE_OUTPUT"
  mark_test_completed

  python3 - "$THROUGHPUT_PROFILE_ARTIFACTS" "$THROUGHPUT_PROFILE_OUTPUT" "$MTMD_SUMMARY_JSON" "$NPU_PROFILE_JSON" "$NPU_PROFILE_LEVEL" <<'PY'
import json
import os
import sys

out_path, metrics_path, mtmd_path, npu_path, npu_level = sys.argv[1:6]

def info(path):
    return {
        "path": path,
        "exists": bool(path) and os.path.exists(path),
        "bytes": os.path.getsize(path) if path and os.path.exists(path) else 0,
    }

payload = {
    "profile_kind": "aicas_semi_throughput_profile",
    "metrics": info(metrics_path),
    "mtmd_prefill_summary": info(mtmd_path),
    "ggml_npu_profile": info(npu_path),
    "npu_profile_level": npu_level,
}

os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
with open(out_path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=False, indent=2)
PY
fi

if [ "$STOP_AFTER_THROUGHPUT" -eq 1 ]; then
  echo "Stopped after throughput: $OUTPUT_DIR/throughput_metrics.json"
  exit 0
fi

if [ "$THROUGHPUT_PROFILE_ONLY" -eq 1 ]; then
  echo "Profile results: $OUTPUT_DIR"
  exit 0
fi

sleep_before_next_test
log_stage "Starting energy evaluation"
python3 "$CODE_DIR/energy_eval.py" \
  -i "$THROUGHPUT_IMAGE" \
  -o "$OUTPUT_DIR/energy_metrics.json" \
  --sample_hz "$SAMPLE_HZ" \
  --power_path "$POWER_PATH" \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --max-tokens "$ENERGY_MAX_TOKENS"
log_stage "Finished energy evaluation: $OUTPUT_DIR/energy_metrics.json"
mark_test_completed

sleep_before_next_test
log_stage "Starting TTFT evaluation"
python3 "$CODE_DIR/ttft_eval_multiprompt.py" \
  -c "$TTFT_CONFIG" \
  -o "$OUTPUT_DIR/ttft_eval_results.json" \
  "$TTFT_CACHE_PROMPT"
log_stage "Finished TTFT evaluation: $OUTPUT_DIR/ttft_eval_results.json"
mark_test_completed

merge_args=(
  --throughput "$OUTPUT_DIR/throughput_metrics.json"
  --energy "$OUTPUT_DIR/energy_metrics.json"
  --ttft "$OUTPUT_DIR/ttft_eval_results.json"
  -o "$OUTPUT_DIR/aicas_submission.json"
)

if [ "$RUN_ACC" -eq 1 ]; then
  merge_args=(--acc "$OUTPUT_DIR/acc_eval_results.json" "${merge_args[@]}")
fi

log_stage "Merging metric files"
python3 "$CODE_DIR/merge_results.py" "${merge_args[@]}"

score_args=(--merged "$OUTPUT_DIR/aicas_submission.json" --output "$OUTPUT_DIR/score_report.json")
if [ -n "$ACC_ORI" ]; then
  score_args+=(--acc-ori "$ACC_ORI")
fi
log_stage "Scoring merged submission"
python3 "$CODE_DIR/score_submission.py" "${score_args[@]}"

echo "Results: $OUTPUT_DIR"
