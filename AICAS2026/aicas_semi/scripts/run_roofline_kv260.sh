#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_roofline_kv260.sh [options]

Collect roofline workload samples and aggregate plotting CSV/JSON data.

This script expects CPU and NPU llama-server endpoints to be reachable. For NPU
profile artifacts it can also call the existing KV260 semi wrapper.

Options:
  --run-id <id>                  Run id (default: UTC timestamp).
  --output-root <path>           Output root (default: AICAS2026/aicas_semi/results/roofline).
  --cpu-base-url <url>           CPU FP16 endpoint, usually http://host:port/v1.
  --npu-base-url <url>           NPU default endpoint, usually http://host:port/v1.
  --cpu-model <name>             CPU model alias (default: smolvlm2-gguf).
  --npu-model <name>             NPU model alias (default: smolvlm2-gguf).
  --image <path>                 Image for image_mmproj (default: code/test2.jpg).
  --repeats <n>                  Latency repeats per workload/hardware (default: 3).
  --text-prefill-tokens <n>      Exact text-only prefill tokens (default: 512).
  --decode-tokens <n>            Decode tokens for average decode point (default: 128).
  --op-convention <mac2|mac1>    Accepted for metadata; only mac2 is supported.
  --cpu-profile-dir <path>       Existing CPU profile results dir.
  --npu-prefill-profile-dir <path>
                                 Existing NPU prefill profile results dir.
  --npu-prefill-latency-profile-dir <path>
                                 Existing low-overhead mtmd latency results dir.
  --npu-decode-profile-dir <path>
                                 Existing NPU decode profile results dir.
  --run-npu-profile              Run existing KV260 wrapper to collect NPU profiles.
  --npu-wrapper-extra <arg>      Extra argument passed to run_semi_eval_kv260.sh; repeatable.
  --microbench-log <path>        Microbench log for hardware_roofs.csv; repeatable.
  --cpu-peak-tops <v>            Override CPU roof peak compute in TOPS.
  --cpu-memory-bandwidth-gbs <v> Override CPU roof memory bandwidth in GB/s.
  --npu-prefill-peak-tops <v>    Override NPU prefill overlay peak compute in TOPS.
  --npu-prefill-memory-bandwidth-gbs <v>
                                 Override NPU prefill overlay bandwidth in GB/s.
  --npu-decode-peak-tops <v>     Override NPU decode overlay peak compute in TOPS.
  --npu-decode-memory-bandwidth-gbs <v>
                                 Override NPU decode overlay bandwidth in GB/s.
  --analyze-only                 Do not collect latency, only aggregate existing inputs.
  -h, --help                     Show help.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CODE_DIR="$REPO_DIR/AICAS2026/aicas_semi/code"
OUTPUT_ROOT="$REPO_DIR/AICAS2026/aicas_semi/results/roofline"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-roofline"
CPU_BASE_URL=""
NPU_BASE_URL=""
CPU_MODEL="smolvlm2-gguf"
NPU_MODEL="smolvlm2-gguf"
IMAGE="$CODE_DIR/test2.jpg"
REPEATS=3
TEXT_PREFILL_TOKENS=512
DECODE_TOKENS=128
OP_CONVENTION="mac2"
CPU_PROFILE_DIR=""
NPU_PREFILL_PROFILE_DIR=""
NPU_PREFILL_LATENCY_PROFILE_DIR=""
NPU_DECODE_PROFILE_DIR=""
RUN_NPU_PROFILE=0
ANALYZE_ONLY=0
NPU_WRAPPER_EXTRA=()
MICROBENCH_LOGS=()
CPU_PEAK_TOPS=""
CPU_MEMORY_BANDWIDTH_GBS=""
NPU_PREFILL_PEAK_TOPS=""
NPU_PREFILL_MEMORY_BANDWIDTH_GBS=""
NPU_DECODE_PEAK_TOPS=""
NPU_DECODE_MEMORY_BANDWIDTH_GBS=""

while [ $# -gt 0 ]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --cpu-base-url) CPU_BASE_URL="$2"; shift 2 ;;
    --npu-base-url) NPU_BASE_URL="$2"; shift 2 ;;
    --cpu-model) CPU_MODEL="$2"; shift 2 ;;
    --npu-model) NPU_MODEL="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --text-prefill-tokens) TEXT_PREFILL_TOKENS="$2"; shift 2 ;;
    --decode-tokens) DECODE_TOKENS="$2"; shift 2 ;;
    --op-convention) OP_CONVENTION="$2"; shift 2 ;;
    --cpu-profile-dir) CPU_PROFILE_DIR="$2"; shift 2 ;;
    --npu-prefill-profile-dir) NPU_PREFILL_PROFILE_DIR="$2"; shift 2 ;;
    --npu-prefill-latency-profile-dir) NPU_PREFILL_LATENCY_PROFILE_DIR="$2"; shift 2 ;;
    --npu-decode-profile-dir) NPU_DECODE_PROFILE_DIR="$2"; shift 2 ;;
    --run-npu-profile) RUN_NPU_PROFILE=1; shift ;;
    --npu-wrapper-extra) NPU_WRAPPER_EXTRA+=("$2"); shift 2 ;;
    --microbench-log) MICROBENCH_LOGS+=("$2"); shift 2 ;;
    --cpu-peak-tops) CPU_PEAK_TOPS="$2"; shift 2 ;;
    --cpu-memory-bandwidth-gbs) CPU_MEMORY_BANDWIDTH_GBS="$2"; shift 2 ;;
    --npu-prefill-peak-tops) NPU_PREFILL_PEAK_TOPS="$2"; shift 2 ;;
    --npu-prefill-memory-bandwidth-gbs) NPU_PREFILL_MEMORY_BANDWIDTH_GBS="$2"; shift 2 ;;
    --npu-decode-peak-tops) NPU_DECODE_PEAK_TOPS="$2"; shift 2 ;;
    --npu-decode-memory-bandwidth-gbs) NPU_DECODE_MEMORY_BANDWIDTH_GBS="$2"; shift 2 ;;
    --analyze-only) ANALYZE_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [ "$OP_CONVENTION" != "mac2" ]; then
  echo "Only --op-convention mac2 is supported by the analyzer." >&2
  exit 2
fi
if [ "$REPEATS" -le 0 ]; then
  echo "--repeats must be positive" >&2
  exit 2
fi

RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
CPU_LATENCY_DIR="$RUN_DIR/latency/cpu"
NPU_LATENCY_DIR="$RUN_DIR/latency/npu"
ANALYSIS_DIR="$RUN_DIR/analysis"
mkdir -p "$RUN_DIR" "$CPU_LATENCY_DIR" "$NPU_LATENCY_DIR" "$ANALYSIS_DIR"

write_meta() {
  python3 - "$RUN_DIR/run_meta.json" <<PY
import json
from pathlib import Path

payload = {
    "schema": "aicas_roofline_run.v1",
    "run_id": "$RUN_ID",
    "repo_dir": "$REPO_DIR",
    "cpu_base_url": "$CPU_BASE_URL",
    "npu_base_url": "$NPU_BASE_URL",
    "cpu_model": "$CPU_MODEL",
    "npu_model": "$NPU_MODEL",
    "image": "$IMAGE",
    "repeats": int("$REPEATS"),
    "text_prefill_tokens": int("$TEXT_PREFILL_TOKENS"),
    "decode_tokens": int("$DECODE_TOKENS"),
    "op_convention": "$OP_CONVENTION",
    "cpu_peak_tops": "$CPU_PEAK_TOPS",
    "cpu_memory_bandwidth_gbs": "$CPU_MEMORY_BANDWIDTH_GBS",
    "npu_prefill_peak_tops": "$NPU_PREFILL_PEAK_TOPS",
    "npu_prefill_memory_bandwidth_gbs": "$NPU_PREFILL_MEMORY_BANDWIDTH_GBS",
    "npu_decode_peak_tops": "$NPU_DECODE_PEAK_TOPS",
    "npu_decode_memory_bandwidth_gbs": "$NPU_DECODE_MEMORY_BANDWIDTH_GBS",
}
Path("$RUN_DIR/run_meta.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

collect_endpoint() {
  local label="$1"
  local base_url="$2"
  local model="$3"
  local out_dir="$4"
  if [ -z "$base_url" ]; then
    return 0
  fi
  mkdir -p "$out_dir"
  local workload repeat max_tokens
  for workload in image_mmproj text_prefill_512 text_decode_128avg; do
    for repeat in $(seq 1 "$REPEATS"); do
      max_tokens=1
      if [ "$workload" = "text_decode_128avg" ]; then
        max_tokens="$DECODE_TOKENS"
      fi
      python3 "$CODE_DIR/roofline_eval.py" \
        --workload "$workload" \
        --base-url "$base_url" \
        --model "$model" \
        --image "$IMAGE" \
        --target-prompt-tokens "$TEXT_PREFILL_TOKENS" \
        --max-tokens "$max_tokens" \
        --ignore-eos \
        --no-cache-prompt \
        -o "$out_dir/${workload}_r${repeat}.json"
    done
  done
  echo "Collected $label latency samples in $out_dir"
}

if [ "$ANALYZE_ONLY" -ne 1 ]; then
  write_meta
  collect_endpoint "CPU" "$CPU_BASE_URL" "$CPU_MODEL" "$CPU_LATENCY_DIR"
  collect_endpoint "NPU" "$NPU_BASE_URL" "$NPU_MODEL" "$NPU_LATENCY_DIR"
fi

if [ "$RUN_NPU_PROFILE" -eq 1 ]; then
  PROFILE_RUN_ID="${RUN_ID}-npu-profile"
  bash "$SCRIPT_DIR/run_semi_eval_kv260.sh" \
    --run-id "$PROFILE_RUN_ID" \
    --skip-acc \
    --throughput-profile-only \
    --throughput-profile-max-tokens "$DECODE_TOKENS" \
    --decode-profile \
    --prefill-profile-mode diagnostic \
    --npu-profile-level diagnostic \
    "${NPU_WRAPPER_EXTRA[@]}"
  NPU_PREFILL_PROFILE_DIR="$REPO_DIR/AICAS2026/aicas_semi/results/kv260/$PROFILE_RUN_ID/results"
  NPU_DECODE_PROFILE_DIR="$NPU_PREFILL_PROFILE_DIR"
fi

analyze_args=(
  --out-dir "$ANALYSIS_DIR"
)
if [ -n "$CPU_PROFILE_DIR" ]; then
  analyze_args+=(--cpu-profile-dir "$CPU_PROFILE_DIR")
fi
if [ -d "$CPU_LATENCY_DIR" ]; then
  analyze_args+=(--cpu-latency-dir "$CPU_LATENCY_DIR")
fi
if [ -n "$NPU_PREFILL_PROFILE_DIR" ]; then
  analyze_args+=(--npu-prefill-profile-dir "$NPU_PREFILL_PROFILE_DIR")
fi
if [ -n "$NPU_PREFILL_LATENCY_PROFILE_DIR" ]; then
  analyze_args+=(--npu-prefill-latency-profile-dir "$NPU_PREFILL_LATENCY_PROFILE_DIR")
fi
if [ -n "$NPU_DECODE_PROFILE_DIR" ]; then
  analyze_args+=(--npu-decode-profile-dir "$NPU_DECODE_PROFILE_DIR")
fi
if [ -d "$NPU_LATENCY_DIR" ]; then
  analyze_args+=(--npu-latency-dir "$NPU_LATENCY_DIR")
fi
for log in "${MICROBENCH_LOGS[@]}"; do
  analyze_args+=(--microbench-log "$log")
done
if [ -n "$CPU_PEAK_TOPS" ]; then
  analyze_args+=(--cpu-peak-tops "$CPU_PEAK_TOPS")
fi
if [ -n "$CPU_MEMORY_BANDWIDTH_GBS" ]; then
  analyze_args+=(--cpu-memory-bandwidth-gbs "$CPU_MEMORY_BANDWIDTH_GBS")
fi
if [ -n "$NPU_PREFILL_PEAK_TOPS" ]; then
  analyze_args+=(--npu-prefill-peak-tops "$NPU_PREFILL_PEAK_TOPS")
fi
if [ -n "$NPU_PREFILL_MEMORY_BANDWIDTH_GBS" ]; then
  analyze_args+=(--npu-prefill-memory-bandwidth-gbs "$NPU_PREFILL_MEMORY_BANDWIDTH_GBS")
fi
if [ -n "$NPU_DECODE_PEAK_TOPS" ]; then
  analyze_args+=(--npu-decode-peak-tops "$NPU_DECODE_PEAK_TOPS")
fi
if [ -n "$NPU_DECODE_MEMORY_BANDWIDTH_GBS" ]; then
  analyze_args+=(--npu-decode-memory-bandwidth-gbs "$NPU_DECODE_MEMORY_BANDWIDTH_GBS")
fi

python3 "$SCRIPT_DIR/analyze_roofline.py" "${analyze_args[@]}"

echo
echo "Roofline run directory: $RUN_DIR"
echo "Points CSV: $ANALYSIS_DIR/roofline_points.csv"
echo "Hardware CSV: $ANALYSIS_DIR/hardware_roofs.csv"
