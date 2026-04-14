#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AICAS_DIR="$ROOT_DIR/AICAS"
RUN_LOCAL_SCRIPT="$AICAS_DIR/scripts/run_local_acc_eval.sh"
STRICT_MMPROJ="$AICAS_DIR/gguf/mmproj-SmolVLM2-500M-Video-Instruct-w8a8-mixed-v1-pt-strict.gguf"
OCR_BENCH="$AICAS_DIR/sampled.json"
SUMMARY_PATH="$AICAS_DIR/artifacts/strict_pt_dequant_mode_eval_summary.json"
OUTPUT_BASE="$AICAS_DIR/output/strict-pt-dequant-mode"

DEQUANT_MODES=(off versa_q8_24 versa_q8_24_fp_reconstruct scale_shift_i32_round scale_shift_fp_reconstruct)
TEXT_MODELS=(f16 int8)

if [ ! -x "$RUN_LOCAL_SCRIPT" ]; then
  echo "Missing or non-executable run_local_acc_eval.sh at $RUN_LOCAL_SCRIPT" >&2
  exit 1
fi

mkdir -p "$(dirname "$SUMMARY_PATH")"

tmp_rows="$(mktemp)"
trap 'rm -f "$tmp_rows"' EXIT

for mode in "${DEQUANT_MODES[@]}"; do
  for text_model in "${TEXT_MODELS[@]}"; do
    output_dir="$OUTPUT_BASE/$mode/$text_model"
    save_name="strict_pt_${mode}_${text_model}"
    stats_json="$output_dir/dequant_stats.json"
    echo "Running strict-pt eval: mode=$mode text_model=$text_model"
    rm -rf "$output_dir"
    (
      export AICAS_MMPROJ_DEQUANT_SIM="$mode"
      if [ "$mode" != "off" ]; then
        export AICAS_MMPROJ_DEQUANT_STATS_FILE="$stats_json"
      else
        unset AICAS_MMPROJ_DEQUANT_STATS_FILE
      fi
      "$RUN_LOCAL_SCRIPT" \
        --model "$text_model" \
        --mmproj "$STRICT_MMPROJ" \
        --ocrbench-file "$OCR_BENCH" \
        --output-folder "$output_dir" \
        --save-name "$save_name"
    )
    result_json="$output_dir/$save_name.json"
    if [ ! -f "$result_json" ]; then
      echo "Missing result JSON: $result_json" >&2
      exit 1
    fi
    row_json=$(
      python3 - "$mode" "$text_model" "$result_json" "$stats_json" <<'PY'
import json, os, sys
mode, text_model, result_json, stats_json = sys.argv[1:]
with open(result_json, "r", encoding="utf-8") as fh:
    data = json.load(fh)
score = sum(int(entry.get("result", 0)) for entry in data)
errors = sum(1 for entry in data if str(entry.get("predict", "")).startswith("API_ERROR"))
stats_payload = None
if os.path.exists(stats_json):
    with open(stats_json, "r", encoding="utf-8") as fh:
        stats_payload = json.load(fh)
row = {
    "variant": "strict_pt_mixed_v1",
    "text_model": text_model,
    "dequant_sim": mode,
    "score": score,
    "errors": errors,
    "result_json": os.path.realpath(result_json),
    "dequant_stats_json": os.path.realpath(stats_json) if stats_payload is not None else "",
    "dequant_stats": stats_payload["counters"] if isinstance(stats_payload, dict) and "counters" in stats_payload else {},
    "notes": f"Strict per-tensor mmproj with {text_model} text model and {mode} dequant simulation."
}
print(json.dumps(row))
PY
    )
    echo "$row_json" >> "$tmp_rows"
  done
done

python3 - <<PY
import json, pathlib
rows = []
with open("$tmp_rows") as fh:
    for line in fh:
        rows.append(json.loads(line))
summary = {
    "schema": "aicas.mmproj.strict-pt-dequant-mode-eval-summary.v1",
    "eval_json": "$OCR_BENCH",
    "rows": rows,
}
pathlib.Path("$SUMMARY_PATH").write_text(json.dumps(summary, indent=2))
PY

echo "Summary written to $SUMMARY_PATH"
