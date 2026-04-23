#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AICAS_DIR="$ROOT_DIR/AICAS"
RUNNER="$AICAS_DIR/scripts/run_local_acc_eval.sh"
OUT_ROOT="${1:-$AICAS_DIR/output/acc-matrix-20260423}"
BASE_PORT="${BASE_PORT:-8090}"

TEXT_PC_MODEL="$AICAS_DIR/output/text-smoothquant/q8base-a05/model.gguf"
TEXT_PT_MODEL="$AICAS_DIR/output/text-smoothquant-q80-refresh/q80_a05_pt_model.gguf"
MMPROJ_PC_MODEL="$AICAS_DIR/output/smoothquant/full-alpha-0_5-minmax/mmproj.gguf"
MMPROJ_PT_MODEL="$AICAS_DIR/output/smoothquant-per-tensor/full-alpha-0_5-minmax/mmproj.gguf"

mkdir -p "$OUT_ROOT"

RESULTS_CSV="$OUT_ROOT/results.csv"
UNSUPPORTED_CSV="$OUT_ROOT/unsupported_cases.csv"

cat >"$RESULTS_CSV" <<'EOF'
case_id,mmproj_granularity,text_granularity,prefill_sq,decode_mode,kv_cache,k_cache_type,v_cache_type,correct,total,accuracy_pct,result_json,output_dir
EOF

cat >"$UNSUPPORTED_CSV" <<'EOF'
case_id,mmproj_granularity,text_prefill_granularity,requested_decode_mode,kv_cache,reason
EOF

append_result_row() {
    local case_id="$1"
    local mmproj_granularity="$2"
    local text_granularity="$3"
    local decode_mode="$4"
    local kv_cache="$5"
    local k_cache_type="$6"
    local v_cache_type="$7"
    local result_json="$8"
    local output_dir="$9"

    python3 - "$RESULTS_CSV" "$case_id" "$mmproj_granularity" "$text_granularity" "$decode_mode" "$kv_cache" "$k_cache_type" "$v_cache_type" "$result_json" "$output_dir" <<'PY'
import csv
import json
import pathlib
import sys

csv_path = pathlib.Path(sys.argv[1])
case_id = sys.argv[2]
mmproj_granularity = sys.argv[3]
text_granularity = sys.argv[4]
decode_mode = sys.argv[5]
kv_cache = sys.argv[6]
k_cache_type = sys.argv[7]
v_cache_type = sys.argv[8]
result_json = pathlib.Path(sys.argv[9])
output_dir = sys.argv[10]

with result_json.open("r", encoding="utf-8") as f:
    data = json.load(f)

if isinstance(data, dict) and "score" in data and "results" in data:
    rows = data["results"]
else:
    rows = data

correct = sum(1 for row in rows if row.get("result") == 1)
total = len(rows)
accuracy = (100.0 * correct / total) if total else 0.0

with csv_path.open("a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        case_id,
        mmproj_granularity,
        text_granularity,
        "sq_w8a8_act_pertensor",
        decode_mode,
        kv_cache,
        k_cache_type,
        v_cache_type,
        correct,
        total,
        f"{accuracy:.2f}",
        str(result_json),
        output_dir,
    ])
PY
}

pick_free_port() {
    local port="$BASE_PORT"
    while true; do
        if ! ss -ltnH 2>/dev/null | awk '{print $4}' | grep -Eq "(^|:)${port}\$"; then
            BASE_PORT=$((port + 1))
            printf '%s\n' "$port"
            return 0
        fi
        port=$((port + 1))
    done
}

run_case() {
    local mmproj_granularity="$1"
    local text_granularity="$2"
    local decode_mode="$3"
    local kv_cache="$4"

    local text_model mmproj_model enable_decode case_id case_dir save_name port
    local k_cache_type="f16"
    local v_cache_type="f16"

    case "$text_granularity" in
        per_channel) text_model="$TEXT_PC_MODEL" ;;
        per_tensor) text_model="$TEXT_PT_MODEL" ;;
        *) echo "unknown text granularity: $text_granularity" >&2; exit 1 ;;
    esac

    case "$mmproj_granularity" in
        per_channel) mmproj_model="$MMPROJ_PC_MODEL" ;;
        per_tensor) mmproj_model="$MMPROJ_PT_MODEL" ;;
        *) echo "unknown mmproj granularity: $mmproj_granularity" >&2; exit 1 ;;
    esac

    case "$decode_mode" in
        q80_grouped) enable_decode="0" ;;
        sq_match_text) enable_decode="1" ;;
        *) echo "unknown decode mode: $decode_mode" >&2; exit 1 ;;
    esac

    case "$kv_cache" in
        fp16)
            k_cache_type="f16"
            v_cache_type="f16"
            ;;
        int8)
            k_cache_type="q8_0"
            v_cache_type="q8_0"
            ;;
        *)
            echo "unknown kv cache mode: $kv_cache" >&2
            exit 1
            ;;
    esac

    case_id="mmproj-${mmproj_granularity}__text-${text_granularity}__decode-${decode_mode}__kv-${kv_cache}"
    case_dir="$OUT_ROOT/$case_id"
    save_name="$case_id"
    port="$(pick_free_port)"

    echo "=== running $case_id ==="
    rm -rf "$case_dir"
    mkdir -p "$case_dir"

    local -a cmd=(
        "$RUNNER"
        --model "$text_model"
        --mmproj "$mmproj_model"
        --output-folder "$case_dir"
        --save-name "$save_name"
        --port "$port"
    )

    if [ "$kv_cache" = "fp16" ]; then
        env \
            AICAS_TEXT_SQ_ENABLE_DECODE_GEMV="$enable_decode" \
            LLAMA_ARG_CACHE_TYPE_K="$k_cache_type" \
            LLAMA_ARG_CACHE_TYPE_V="$v_cache_type" \
            "${cmd[@]}"
    else
        env \
            AICAS_TEXT_SQ_ENABLE_DECODE_GEMV="$enable_decode" \
            LLAMA_ARG_CACHE_TYPE_K="$k_cache_type" \
            LLAMA_ARG_CACHE_TYPE_V="$v_cache_type" \
            LLAMA_ARG_FLASH_ATTN=on \
            "${cmd[@]}"
    fi

    append_result_row \
        "$case_id" \
        "$mmproj_granularity" \
        "$text_granularity" \
        "$decode_mode" \
        "$kv_cache" \
        "$k_cache_type" \
        "$v_cache_type" \
        "$case_dir/$save_name.json" \
        "$case_dir"
}

append_unsupported() {
    local mmproj_granularity="$1"
    local text_prefill_granularity="$2"
    local requested_decode_mode="$3"
    local kv_cache="$4"
    local case_id="mmproj-${mmproj_granularity}__prefill-${text_prefill_granularity}__decode-${requested_decode_mode}__kv-${kv_cache}"

    printf '%s,%s,%s,%s,%s,%s\n' \
        "$case_id" \
        "$mmproj_granularity" \
        "$text_prefill_granularity" \
        "$requested_decode_mode" \
        "$kv_cache" \
        "current runtime/model metadata use one text SQ granularity for both prefill and decode; this mismatched combination is not representable" \
        >>"$UNSUPPORTED_CSV"
}

for mmproj_granularity in per_channel per_tensor; do
    for text_granularity in per_channel per_tensor; do
        for kv_cache in fp16 int8; do
            run_case "$mmproj_granularity" "$text_granularity" q80_grouped "$kv_cache"
            run_case "$mmproj_granularity" "$text_granularity" sq_match_text "$kv_cache"
        done
    done
done

for mmproj_granularity in per_channel per_tensor; do
    for kv_cache in fp16 int8; do
        append_unsupported "$mmproj_granularity" per_channel sq_per_tensor "$kv_cache"
        append_unsupported "$mmproj_granularity" per_tensor sq_per_channel "$kv_cache"
    done
done

echo "results: $RESULTS_CSV"
echo "unsupported: $UNSUPPORTED_CSV"
