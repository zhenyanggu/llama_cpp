#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

SERVER_BIN="${SERVER_BIN:-$REPO_DIR/build-kv260-semi/bin/llama-server}"
PORT="${PORT:-18080}"
RUN_PREFIX="${RUN_PREFIX:-$(date +%Y%m%dT%H%M%S)-acc-seed-sweep}"
RESULT_ROOT="$REPO_DIR/AICAS2026/aicas_semi/results/kv260"
SUMMARY="$RESULT_ROOT/${RUN_PREFIX}.tsv"

if [ -n "${SEEDS:-}" ]; then
  read -r -a SEEDS <<< "$SEEDS"
else
  SEEDS=(20260615 20260616 20260617)
fi

OFFICIAL_FP16_MODEL="$REPO_DIR/AICAS/gguf/SmolVLM2-500M-Video-Instruct-f16.gguf"
OFFICIAL_FP16_MMPROJ="$REPO_DIR/AICAS/gguf/mmproj-SmolVLM2-500M-Video-Instruct-f16.gguf"

common_args=(
  --skip-build
  --server-bin "$SERVER_BIN"
  --run-acc
  --acc-only
  --port "$PORT"
)

baseline_disable_env=(
  --no-decode-npu
  --mtmd-backend-device CPU
  --no-npu-text-prefill-dynamic
  --no-npu-shape-table
  --no-npu-preload-weights
  --model "$OFFICIAL_FP16_MODEL"
  --mmproj "$OFFICIAL_FP16_MMPROJ"
  --server-env AICAS_TEXT_PREFILL_LOG8PV_NPU=0
  --server-env AICAS_TEXT_PREFILL_LOG8PV_DIRECT_KV=0
  --server-env AICAS_TEXT_PREFILL_ATTN_LOG8PV=0
  --server-env AICAS_TEXT_PREFILL_LOG8PV_BIT_ACCURATE=0
  --server-env AICAS_MMPROJ_LOG8PV_NPU=0
  --server-env AICAS_MMPROJ_ATTN_PRECISION=f16
  --server-env AICAS_TEXT_TOKEN_EMBD_W8A16=0
  --server-env AICAS_TEXT_DECODE_AWQ=0
  --server-env AICAS_TEXT_DECODE_AWQ_NPU=0
  --server-env AICAS_TEXT_DECODE_AWQ_FUSED_FFN_NPU=0
  --server-env AICAS_TEXT_DECODE_ATTN_NPU=0
  --server-env AICAS_TEXT_DECODE_ATTN_W8A16_RTL=0
  --server-env AICAS_TEXT_LM_HEAD_W8A16_NPU=0
  --server-env AICAS_TEXT_LM_HEAD_W8A16_DP128=0
  --server-env AICAS_TEXT_LM_HEAD_W16A16_DP128=0
)

mkdir -p "$RESULT_ROOT"
printf 'run_id\tconfig\tseed\tcorrect\ttotal\taccuracy\tresult_json\n' > "$SUMMARY"

summarize_run() {
  local run_id="$1"
  local config="$2"
  local seed="$3"
  local result_json="$RESULT_ROOT/$run_id/results/acc_eval_results.json"

  python3 - "$result_json" "$run_id" "$config" "$seed" >> "$SUMMARY" <<'PY'
import json
import sys

path, run_id, config, seed = sys.argv[1:5]
with open(path, "r", encoding="utf-8") as handle:
    rows = json.load(handle)
total = len(rows)
correct = sum(int(row.get("result", 0)) for row in rows)
acc = correct / total if total else 0.0
print(f"{run_id}\t{config}\t{seed}\t{correct}\t{total}\t{acc:.6f}\t{path}")
PY
}

run_one() {
  local config="$1"
  local seed="$2"
  local run_id="${RUN_PREFIX}-${config}-seed${seed}"

  echo "===== RUN $run_id ====="
  if [ "$config" = "npu" ]; then
    bash "$SCRIPT_DIR/run_semi_eval_kv260.sh" \
      "${common_args[@]}" \
      --acc-sample-seed "$seed" \
      --run-id "$run_id"
  elif [ "$config" = "fp16" ]; then
    bash "$SCRIPT_DIR/run_semi_eval_kv260.sh" \
      "${common_args[@]}" \
      "${baseline_disable_env[@]}" \
      --acc-sample-seed "$seed" \
      --run-id "$run_id"
  else
    echo "unknown config: $config" >&2
    return 1
  fi
  summarize_run "$run_id" "$config" "$seed"
  echo "===== DONE $run_id ====="
}

for seed in "${SEEDS[@]}"; do
  run_one npu "$seed"
  run_one fp16 "$seed"
done

echo "Summary: $SUMMARY"
cat "$SUMMARY"
