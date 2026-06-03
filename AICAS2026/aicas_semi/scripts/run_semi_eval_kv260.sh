#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_semi_eval_kv260.sh [options]

Runs the AICAS 2026 semi-final eval on a KV260 board with a user-launched
llama-server and shared-asset reuse.

Options:
  --host <host>                 KV260 host (default: 192.168.0.10)
  --user <user>                 KV260 user (default: ubuntu)
  --remote-root <path>          Remote root (default: /home/ubuntu/aicas-semi)
  --sdk-env <path>              KV260 SDK env script
  --build-dir <path>            Cross-build dir (default: build-kv260-semi-arm)
  --server-bin <path>           Reuse existing KV260 llama-server binary
  --skip-build                  Skip cross-build and reuse --server-bin/build output
  --best-config-json <path>     Load model/mmproj/server env from a current-best config JSON
                                (default: ../model-quant/docs/aicas-current-best-config.json)
  --model <path>                Text model GGUF path
  --mmproj <path>               mmproj GGUF path
  --overlay-app <name>          Prefill overlay app to load (default: prefill_attention_log8_qcount_100m_0603_app)
  --decode-overlay-app <name>   Decode overlay app for W4A16 GEMV (default: decode_disable_rsk_irqfix_0528_app)
  --decode-overlay-dir <path>   Local decode overlay directory
  --generic-fastpath            Use generic prefill/decode apps
  --overlay-switch-mode <mode>  Overlay switch mode: fast|loadapp (default: fast)
  --remote-npu-driver-ko <path> Use an existing board-side npu_kv260.ko instead of staging local driver
  --prefill-bitstream-fw <path> FPGA manager firmware path for generic prefill
  --decode-bitstream-fw <path>  FPGA manager firmware path for generic decode
  --no-decode-npu               Disable decode AWQ W4A16 NPU offload and overlay switching
  --sudo-password <password>    Board sudo password (default: 123456)
  --port <port>                 llama-server port (default: 8080)
  --threads <n>                 llama-server threads (default: 4)
  --ubatch-size <n>             llama-server physical ubatch size (default: 1024)
  --cache-type-k <type>         llama-server K cache type (default: q8_0)
  --cache-type-v <type>         llama-server V cache type (default: q8_0)
  --flash-attn <on|off|auto>    llama-server flash attention mode (default: auto)
  --ctx-size <n>                llama-server context size
  --mtmd-backend-device <name>  MTMD_BACKEND_DEVICE for mmproj (default: NPU)
  --alias <name>                Model alias (default: smolvlm2-gguf)
  --output-dir <path>           Local results root
  --run-id <id>                 Override run id
  --run-acc                     Run accuracy
  --skip-acc                    Skip accuracy and run throughput/energy/ttft only (default)
  --throughput-only             Run only throughput_eval.py, skipping acc/energy/ttft/merge
  --energy-only                 Run only energy_eval.py, skipping acc/throughput/ttft/merge
  --energy-max-tokens <n>       Max generated tokens for energy_eval.py (default: 128)
  --correctness-only            Run a short token-limited image generation sanity check only
  --correctness-max-tokens <n>  Max generated tokens for correctness-only (default: 1)
  --correctness-prompt <text>   Prompt for correctness-only
  --ttft-only                   Run only ttft_eval_multiprompt.py, skipping acc/throughput/energy/merge
  --run-throughput-profile      Run an extra profiled throughput pass after normal eval
  --throughput-profile-only     Only run the profiled throughput pass
  --throughput-profile-max-tokens <n>
                                Max generated tokens for the profiled pass (default: 4096)
  --throughput-profile-prompt <text>
                                Prompt for profiled pass (default: throughput_eval.py LONG_PROMPT)
  --prefill-profile-mode <mode> Profile overhead mode: summary|aggregate|diagnostic (default: summary)
  --npu-profile-level <level>   GGML_NPU_PROFILE_LEVEL for profile pass (default: diagnostic)
  --no-profile-compare          Disable extra decode CPU/NPU compare during profile pass
  --npu-shape-table <path>      Optional GGML_NPU_SHAPE_TABLE_JSON file for text W8A8 GEMM fast path
  --no-npu-shape-table          Disable text W8A8 shape-table offload
  --npu-preload-weights         Preload NPU weights into CMA and keep them resident (default)
  --no-npu-preload-weights      Disable NPU packed-weight preload
  --npu-shape-record            Record observed text W8A8 GEMM shapes to results/npu_shape_record.json
  --npu-wait-policy <mode>      Runtime wait mode: hybrid|spin|irq|adaptive
  --npu-wait-policy-json <path> Policy JSON for --npu-wait-policy adaptive
  --npu-wait-trace              Record per-instruction NPU wait trace to results/npu_wait_trace.jsonl
  --trace-ubatch                Record pre/post microbatch sizes to results/ubatch_trace.jsonl
  --merge-prefill               Merge multimodal prompt into one embedding prefill (default)
  --no-merge-prefill            Disable merged multimodal embedding prefill
  --merge-prefill-trace         Log merged-prefill entry/fallback/success reasons
  --ttft-cache-prompt           Send cache_prompt=true in TTFT requests
  --no-ttft-cache-prompt        Send cache_prompt=false in TTFT requests
  --npu-mmproj-only             Use NPU for mmproj only; keep text W8A8 prefill on CPU
  --npu-text-prefill-dynamic    Enable dynamic NPU text prefill GEMM shapes (default)
  --no-npu-text-prefill-dynamic Disable dynamic NPU text prefill GEMM shapes
  --server-env <KEY=VALUE>      Add an environment assignment to llama-server; repeatable
  --require-mmproj-bfp8m-npu    Require mmproj BFP8-M attention to use the NPU raw GEMM path
  --acc-sample-mode <mode>      available|official (default: available)
  --force-sync-shared           Re-upload shared assets even if same-size files exist
  --power-path <path>           Board hwmon path
  --sample-hz <hz>              Energy sampling rate
  --acc-ori <ratio>             Forwarded to score_submission.py
  -h, --help                    Show help
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CODE_DIR="$REPO_DIR/AICAS2026/aicas_semi/code"
DEFAULT_NPU_SHAPE_TABLE="$REPO_DIR/AICAS2026/aicas_semi/config/semi_npu_shapes.json"
DEFAULT_BEST_CONFIG_JSON="$REPO_DIR/../model-quant/docs/aicas-current-best-config.json"
VERSA_PREFILL_OVERLAY_DIR="/mnt/c/vivado/KV260/out/prefill_attention_log8_qcount_100m_0603_app"
DEFAULT_DECODE_OVERLAY_DIR="/mnt/c/vivado/KV260/out/decode_disable_rsk_irqfix_0528_app"
GENERIC_PREFILL_OVERLAY_DIR="$VERSA_PREFILL_OVERLAY_DIR"
GENERIC_DECODE_OVERLAY_DIR="/mnt/c/vivado/KV260/out/decode_generic_light_0529_app"
GENERIC_PREFILL_FW_DEFAULT="xilinx/prefill_attention_log8_qcount_100m_0603_app/prefill_attention_log8_qcount_100m_0603.bit.bin"
GENERIC_DECODE_FW_DEFAULT="xilinx/decode_generic_light_0529_app/decode_generic_light_0529.bit.bin"
NPU_DRIVER_KO="$REPO_DIR/npuruntime/kv260/driver/npu_kv260.ko"
NPU_DRIVER_MAX_BUFFER_MB="${NPU_DRIVER_MAX_BUFFER_MB:-1500}"

HOST="192.168.0.10"
USER_NAME="ubuntu"
REMOTE_ROOT="/home/ubuntu/aicas-semi"
SDK_ENV="/home/gugugu/petalinux/sdk/kv260-2025.1/environment-setup-cortexa72-cortexa53-amd-linux"
BUILD_DIR="$REPO_DIR/build-kv260-semi-arm"
SERVER_BIN=""
SKIP_BUILD=0
MODEL_PATH="$REPO_DIR/../model-quant/AICAS/output/group128-hparam-search/text_awq/calib16-a0_12-g128-sym-lmhead/text_sq_a05_decode_awq_sym_lmhead_calib16_a0_12_g128_scale_f16.gguf"
MMPROJ_PATH="$REPO_DIR/../model-quant/AICAS/output/group128-hparam-search/mmproj_sq/a0_7-minmax-per_channel/mmproj.gguf"
BEST_CONFIG_JSON="$DEFAULT_BEST_CONFIG_JSON"
BEST_CONFIG_ENV=""
EXTRA_SERVER_ENV=""
TEXT_PREFILL_LOG8PV_CALIB_PATH=""
MMPROJ_LOG8PV_CALIB_PATH=""
MODEL_EXPLICIT=0
MMPROJ_EXPLICIT=0
OVERLAY_APP="prefill_attention_log8_qcount_100m_0603_app"
DECODE_OVERLAY_APP="decode_disable_rsk_irqfix_0528_app"
DECODE_OVERLAY_DIR="$DEFAULT_DECODE_OVERLAY_DIR"
OVERLAY_APP_EXPLICIT=0
DECODE_OVERLAY_APP_EXPLICIT=0
DECODE_OVERLAY_DIR_EXPLICIT=0
GENERIC_FASTPATH=0
NPU_OVERLAY_SWITCH_MODE="fast"
REMOTE_NPU_DRIVER_KO_OVERRIDE=""
GENERIC_PREFILL_FW="$GENERIC_PREFILL_FW_DEFAULT"
GENERIC_DECODE_FW="$GENERIC_DECODE_FW_DEFAULT"
GENERIC_PREFILL_FW_EXPLICIT=0
GENERIC_DECODE_FW_EXPLICIT=0
DECODE_NPU=0
SUDO_PASSWORD="${BOARD_SUDO_PASSWORD:-123456}"
PORT="8080"
THREADS="4"
UBATCH_SIZE="1024"
CACHE_TYPE_K="q8_0"
CACHE_TYPE_V="q8_0"
FLASH_ATTN="auto"
CTX_SIZE=""
MTMD_BACKEND_DEVICE="NPU"
MODEL_ALIAS="smolvlm2-gguf"
OUTPUT_ROOT="$REPO_DIR/AICAS2026/aicas_semi/results/kv260"
RUN_ID=""
RUN_ACC=0
RUN_THROUGHPUT_ONLY=0
RUN_ENERGY_ONLY=0
ENERGY_MAX_TOKENS="128"
RUN_CORRECTNESS_ONLY=0
CORRECTNESS_MAX_TOKENS="1"
CORRECTNESS_PROMPT="Please describe the image concisely and accurately."
RUN_TTFT_ONLY=0
RUN_THROUGHPUT_PROFILE=0
THROUGHPUT_PROFILE_ONLY=0
THROUGHPUT_PROFILE_MAX_TOKENS="4096"
THROUGHPUT_PROFILE_PROMPT=""
PREFILL_PROFILE_MODE="summary"
NPU_PROFILE_LEVEL="diagnostic"
PROFILE_COMPARE=1
NPU_SHAPE_TABLE=""
if [ -f "$DEFAULT_NPU_SHAPE_TABLE" ]; then
  NPU_SHAPE_TABLE="$DEFAULT_NPU_SHAPE_TABLE"
fi
NPU_SHAPE_RECORD=0
NPU_PRELOAD_WEIGHTS=1
NPU_WAIT_POLICY=""
NPU_WAIT_POLICY_JSON=""
NPU_WAIT_TRACE=0
TRACE_UBATCH=0
MERGE_PREFILL=1
MERGE_PREFILL_TRACE=0
TTFT_CACHE_PROMPT=""
NPU_TEXT_PREFILL_DYNAMIC=1
ACC_SAMPLE_MODE="available"
FORCE_SYNC_SHARED=0
POWER_PATH="/sys/class/hwmon/hwmon2/power1_input"
SAMPLE_HZ="100"
ACC_ORI=""

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
    --sdk-env) SDK_ENV="$2"; shift 2 ;;
    --build-dir) BUILD_DIR="$2"; shift 2 ;;
    --server-bin) SERVER_BIN="$2"; shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --best-config-json) BEST_CONFIG_JSON="$2"; shift 2 ;;
    --model) MODEL_PATH="$2"; MODEL_EXPLICIT=1; shift 2 ;;
    --mmproj) MMPROJ_PATH="$2"; MMPROJ_EXPLICIT=1; shift 2 ;;
    --overlay-app) OVERLAY_APP="$2"; OVERLAY_APP_EXPLICIT=1; shift 2 ;;
    --decode-overlay-app) DECODE_OVERLAY_APP="$2"; DECODE_OVERLAY_APP_EXPLICIT=1; shift 2 ;;
    --decode-overlay-dir) DECODE_OVERLAY_DIR="$2"; DECODE_OVERLAY_DIR_EXPLICIT=1; shift 2 ;;
    --generic-fastpath) GENERIC_FASTPATH=1; shift ;;
    --overlay-switch-mode) NPU_OVERLAY_SWITCH_MODE="$2"; shift 2 ;;
    --remote-npu-driver-ko) REMOTE_NPU_DRIVER_KO_OVERRIDE="$2"; shift 2 ;;
    --prefill-bitstream-fw) GENERIC_PREFILL_FW="$2"; GENERIC_PREFILL_FW_EXPLICIT=1; shift 2 ;;
    --decode-bitstream-fw) GENERIC_DECODE_FW="$2"; GENERIC_DECODE_FW_EXPLICIT=1; shift 2 ;;
    --no-decode-npu) DECODE_NPU=0; shift ;;
    --sudo-password) SUDO_PASSWORD="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --ubatch-size) UBATCH_SIZE="$2"; shift 2 ;;
    --cache-type-k) CACHE_TYPE_K="$2"; shift 2 ;;
    --cache-type-v) CACHE_TYPE_V="$2"; shift 2 ;;
    --flash-attn) FLASH_ATTN="$2"; shift 2 ;;
    --ctx-size) CTX_SIZE="$2"; shift 2 ;;
    --mtmd-backend-device) MTMD_BACKEND_DEVICE="$2"; shift 2 ;;
    --alias) MODEL_ALIAS="$2"; shift 2 ;;
    --output-dir) OUTPUT_ROOT="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --run-acc) RUN_ACC=1; shift ;;
    --skip-acc) RUN_ACC=0; shift ;;
    --throughput-only) RUN_THROUGHPUT_ONLY=1; shift ;;
    --energy-only) RUN_ENERGY_ONLY=1; shift ;;
    --energy-max-tokens) ENERGY_MAX_TOKENS="$2"; shift 2 ;;
    --correctness-only) RUN_CORRECTNESS_ONLY=1; shift ;;
    --correctness-max-tokens) CORRECTNESS_MAX_TOKENS="$2"; shift 2 ;;
    --correctness-prompt) CORRECTNESS_PROMPT="$2"; shift 2 ;;
    --ttft-only) RUN_TTFT_ONLY=1; shift ;;
    --run-throughput-profile) RUN_THROUGHPUT_PROFILE=1; shift ;;
    --throughput-profile-only) RUN_THROUGHPUT_PROFILE=1; THROUGHPUT_PROFILE_ONLY=1; shift ;;
    --throughput-profile-max-tokens) THROUGHPUT_PROFILE_MAX_TOKENS="$2"; shift 2 ;;
    --throughput-profile-prompt) THROUGHPUT_PROFILE_PROMPT="$2"; shift 2 ;;
    --prefill-profile-mode) PREFILL_PROFILE_MODE="$2"; shift 2 ;;
    --npu-profile-level) NPU_PROFILE_LEVEL="$2"; shift 2 ;;
    --no-profile-compare) PROFILE_COMPARE=0; shift ;;
    --npu-shape-table) NPU_SHAPE_TABLE="$2"; shift 2 ;;
    --no-npu-shape-table) NPU_SHAPE_TABLE=""; shift ;;
    --npu-preload-weights) NPU_PRELOAD_WEIGHTS=1; shift ;;
    --no-npu-preload-weights) NPU_PRELOAD_WEIGHTS=0; shift ;;
    --npu-shape-record) NPU_SHAPE_RECORD=1; shift ;;
    --npu-wait-policy) NPU_WAIT_POLICY="$2"; shift 2 ;;
    --npu-wait-policy-json) NPU_WAIT_POLICY_JSON="$2"; shift 2 ;;
    --npu-wait-trace) NPU_WAIT_TRACE=1; shift ;;
    --trace-ubatch) TRACE_UBATCH=1; shift ;;
    --merge-prefill) MERGE_PREFILL=1; shift ;;
    --no-merge-prefill) MERGE_PREFILL=0; shift ;;
    --merge-prefill-trace) MERGE_PREFILL_TRACE=1; shift ;;
    --ttft-cache-prompt) TTFT_CACHE_PROMPT=1; shift ;;
    --no-ttft-cache-prompt) TTFT_CACHE_PROMPT=0; shift ;;
    --npu-mmproj-only) MTMD_BACKEND_DEVICE="NPU"; NPU_SHAPE_TABLE=""; NPU_SHAPE_RECORD=0; NPU_TEXT_PREFILL_DYNAMIC=0; shift ;;
    --npu-text-prefill-dynamic) NPU_TEXT_PREFILL_DYNAMIC=1; shift ;;
    --no-npu-text-prefill-dynamic) NPU_TEXT_PREFILL_DYNAMIC=0; shift ;;
    --server-env)
      if [[ "$2" != *=* ]]; then
        echo "--server-env expects KEY=VALUE" >&2
        exit 1
      fi
      EXTRA_SERVER_ENV="${EXTRA_SERVER_ENV:+$EXTRA_SERVER_ENV }$2"
      shift 2
      ;;
    --require-mmproj-bfp8m-npu)
      EXTRA_SERVER_ENV="${EXTRA_SERVER_ENV:+$EXTRA_SERVER_ENV }AICAS_MMPROJ_BFP8M_NPU=1 AICAS_MMPROJ_BFP8M_NPU_REQUIRE=1"
      shift
      ;;
    --acc-sample-mode) ACC_SAMPLE_MODE="$2"; shift 2 ;;
    --force-sync-shared) FORCE_SYNC_SHARED=1; shift ;;
    --power-path) POWER_PATH="$2"; shift 2 ;;
    --sample-hz) SAMPLE_HZ="$2"; shift 2 ;;
    --acc-ori) ACC_ORI="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

load_best_config_json() {
  [ -n "$BEST_CONFIG_JSON" ] || return 0

  BEST_CONFIG_JSON="$(realpath "$BEST_CONFIG_JSON")"
  eval "$(
    python3 - "$BEST_CONFIG_JSON" "$REPO_DIR" "$MODEL_EXPLICIT" "$MMPROJ_EXPLICIT" <<'PY'
import json
import os
import shlex
import sys

cfg_path, repo_dir, model_explicit, mmproj_explicit = sys.argv[1:5]
with open(cfg_path, "r", encoding="utf-8") as handle:
    cfg = json.load(handle)

def repo_abs(path):
    if not path:
        return ""
    if os.path.isabs(path):
        return path
    cfg_dir = os.path.dirname(cfg_path)
    cfg_root = os.path.dirname(cfg_dir) if os.path.basename(cfg_dir) == "docs" else cfg_dir
    repo_candidate = os.path.join(repo_dir, path)
    cfg_candidate = os.path.join(cfg_root, path)
    if os.path.exists(cfg_candidate) and not os.path.exists(repo_candidate):
        return cfg_candidate
    return repo_candidate

assignments = []
if model_explicit != "1":
    assignments.append(("MODEL_PATH", repo_abs(cfg.get("text_model", ""))))
if mmproj_explicit != "1":
    assignments.append(("MMPROJ_PATH", repo_abs(cfg.get("mmproj", ""))))

env = cfg.get("env", {})
if not isinstance(env, dict):
    raise SystemExit("best-config JSON field 'env' must be an object")
schema = str(cfg.get("schema", ""))
if "nobfp" in schema or os.path.basename(cfg_path) == "aicas-current-best-nobfp-config.json":
    env["AICAS_MMPROJ_ATTN_PRECISION"] = "f16"
env_parts = []
for key, value in env.items():
    if not isinstance(key, str) or not key.replace("_", "").isalnum():
        raise SystemExit(f"invalid environment key in best config: {key!r}")
    if key == "AICAS_TEXT_PREFILL_LOG8PV_CALIB_FILE":
        assignments.append(("TEXT_PREFILL_LOG8PV_CALIB_PATH", repo_abs(str(value))))
        continue
    if key == "AICAS_MMPROJ_LOG8PV_CALIB_FILE":
        assignments.append(("MMPROJ_LOG8PV_CALIB_PATH", repo_abs(str(value))))
        continue
    env_parts.append(f"{key}={shlex.quote(str(value))}")
assignments.append(("BEST_CONFIG_ENV", " ".join(env_parts)))

for key, value in assignments:
    print(f"{key}={shlex.quote(value)}")
PY
  )"
  echo "[config] loaded best config: $BEST_CONFIG_JSON"
}

load_best_config_json
if [ "$GENERIC_FASTPATH" -eq 1 ]; then
  if [ "$OVERLAY_APP_EXPLICIT" -ne 1 ]; then
    OVERLAY_APP="prefill_attention_log8_qcount_100m_0603_app"
  fi
  if [ "$DECODE_OVERLAY_APP_EXPLICIT" -ne 1 ]; then
    DECODE_OVERLAY_APP="decode_generic_light_0529_app"
  fi
  if [ "$DECODE_OVERLAY_DIR_EXPLICIT" -ne 1 ]; then
    if [ "$DECODE_OVERLAY_APP_EXPLICIT" -eq 1 ]; then
      DECODE_OVERLAY_DIR=""
    else
      DECODE_OVERLAY_DIR="$GENERIC_DECODE_OVERLAY_DIR"
    fi
  fi
  if [ "$GENERIC_PREFILL_FW_EXPLICIT" -ne 1 ] && [ "$OVERLAY_APP_EXPLICIT" -eq 1 ]; then
    prefill_fw_stem="${OVERLAY_APP%_app}"
    GENERIC_PREFILL_FW="xilinx/$OVERLAY_APP/$prefill_fw_stem.bit.bin"
  fi
  if [ "$GENERIC_DECODE_FW_EXPLICIT" -ne 1 ] && [ "$DECODE_OVERLAY_APP_EXPLICIT" -eq 1 ]; then
    decode_fw_stem="${DECODE_OVERLAY_APP%_app}"
    GENERIC_DECODE_FW="xilinx/$DECODE_OVERLAY_APP/$decode_fw_stem.bit.bin"
  fi
fi
if [ -n "$EXTRA_SERVER_ENV" ]; then
  BEST_CONFIG_ENV="${BEST_CONFIG_ENV:+$BEST_CONFIG_ENV }$EXTRA_SERVER_ENV"
fi
BEST_CONFIG_ENV="VERSA_P_CMA_SIZE=1408M VERSA_P_CMA_HEAP_OFFSET=0 VERSA_P_CMA_HEAP_SIZE=832M NPU_CMA_SIZE=1408M NPU_CMA_HEAP_OFFSET=832M NPU_CMA_HEAP_SIZE=576M AICAS_TEXT_PREFILL_LOG8PV_NPU=1 AICAS_TEXT_PREFILL_LOG8PV_DIRECT_KV=1 AICAS_MMPROJ_LOG8PV_NPU=1${BEST_CONFIG_ENV:+ $BEST_CONFIG_ENV}"

if [ "$DECODE_NPU" -eq 1 ]; then
  [ -n "$DECODE_OVERLAY_APP" ] || {
    echo "--decode-overlay-app must not be empty when decode NPU is enabled" >&2
    exit 1
  }
  [ "$GENERIC_FASTPATH" -eq 1 ] || [ -d "$DECODE_OVERLAY_DIR" ] || {
    echo "Decode overlay directory not found: $DECODE_OVERLAY_DIR" >&2
    exit 1
  }
fi

case "$ACC_SAMPLE_MODE" in
  available|official) ;;
  *)
    echo "--acc-sample-mode must be one of: available, official" >&2
    exit 1
    ;;
esac

case "$NPU_OVERLAY_SWITCH_MODE" in
  fast|loadapp) ;;
  *)
    echo "--overlay-switch-mode must be one of: fast, loadapp" >&2
    exit 1
    ;;
esac

case "$PREFILL_PROFILE_MODE" in
  summary|aggregate|diagnostic) ;;
  *)
    echo "--prefill-profile-mode must be one of: summary, aggregate, diagnostic" >&2
    exit 1
    ;;
esac

case "$UBATCH_SIZE" in
  ''|*[!0-9]*)
    echo "--ubatch-size must be a positive integer" >&2
    exit 1
    ;;
  *)
    if [ "$UBATCH_SIZE" -le 0 ]; then
      echo "--ubatch-size must be a positive integer" >&2
      exit 1
    fi
    ;;
esac

case "$CORRECTNESS_MAX_TOKENS" in
  ''|*[!0-9]*)
    echo "--correctness-max-tokens must be a positive integer" >&2
    exit 1
    ;;
  *)
    if [ "$CORRECTNESS_MAX_TOKENS" -le 0 ]; then
      echo "--correctness-max-tokens must be a positive integer" >&2
      exit 1
    fi
    ;;
esac

case "$ENERGY_MAX_TOKENS" in
  ''|*[!0-9]*)
    echo "--energy-max-tokens must be a positive integer" >&2
    exit 1
    ;;
  *)
    if [ "$ENERGY_MAX_TOKENS" -le 0 ]; then
      echo "--energy-max-tokens must be a positive integer" >&2
      exit 1
    fi
    ;;
esac

case "$NPU_WAIT_POLICY" in
  ""|hybrid|spin|irq|adaptive) ;;
  *)
    echo "--npu-wait-policy must be one of: hybrid, spin, irq, adaptive" >&2
    exit 1
    ;;
esac

TARGET="$USER_NAME@$HOST"
MODEL_PATH="$(realpath "$MODEL_PATH")"
MMPROJ_PATH="$(realpath "$MMPROJ_PATH")"
if [ -n "$TEXT_PREFILL_LOG8PV_CALIB_PATH" ]; then
  TEXT_PREFILL_LOG8PV_CALIB_PATH="$(realpath "$TEXT_PREFILL_LOG8PV_CALIB_PATH")"
fi
if [ -n "$MMPROJ_LOG8PV_CALIB_PATH" ]; then
  MMPROJ_LOG8PV_CALIB_PATH="$(realpath "$MMPROJ_LOG8PV_CALIB_PATH")"
fi
[ -f "$MODEL_PATH" ] || { echo "Model not found: $MODEL_PATH" >&2; exit 1; }
[ -f "$MMPROJ_PATH" ] || { echo "mmproj not found: $MMPROJ_PATH" >&2; exit 1; }
if [ -n "$TEXT_PREFILL_LOG8PV_CALIB_PATH" ]; then
  [ -f "$TEXT_PREFILL_LOG8PV_CALIB_PATH" ] || { echo "Text log8PV calibration not found: $TEXT_PREFILL_LOG8PV_CALIB_PATH" >&2; exit 1; }
fi
if [ -n "$MMPROJ_LOG8PV_CALIB_PATH" ]; then
  [ -f "$MMPROJ_LOG8PV_CALIB_PATH" ] || { echo "mmproj log8PV calibration not found: $MMPROJ_LOG8PV_CALIB_PATH" >&2; exit 1; }
fi
THROUGHPUT_PROFILE_PROMPT_B64="$(printf '%s' "$THROUGHPUT_PROFILE_PROMPT" | base64 -w0)"
CORRECTNESS_PROMPT_B64="$(printf '%s' "$CORRECTNESS_PROMPT" | base64 -w0)"
if [ -n "$NPU_SHAPE_TABLE" ]; then
  NPU_SHAPE_TABLE="$(realpath "$NPU_SHAPE_TABLE")"
fi
if [ -n "$NPU_WAIT_POLICY_JSON" ]; then
  NPU_WAIT_POLICY_JSON="$(realpath "$NPU_WAIT_POLICY_JSON")"
fi
mkdir -p "$OUTPUT_ROOT"
OUTPUT_ROOT="$(realpath -m "$OUTPUT_ROOT")"

if [ -z "$RUN_ID" ]; then
  RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-semi-kv260"
fi

LOCAL_RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
LOCAL_STAGE_DIR="$LOCAL_RUN_DIR/stage"
LOCAL_RESULT_DIR="$LOCAL_RUN_DIR/results"
mkdir -p "$LOCAL_STAGE_DIR" "$LOCAL_RESULT_DIR"

REMOTE_SHARED_DIR="$REMOTE_ROOT/shared"
REMOTE_RUN_DIR="$REMOTE_ROOT/runs/$RUN_ID"
REMOTE_LIB_DIR="$REMOTE_SHARED_DIR/runtime_libs"
REMOTE_CODE_DIR="$REMOTE_RUN_DIR/code"
REMOTE_DATA_DIR="$REMOTE_RUN_DIR/data"
REMOTE_RESULTS_DIR="$REMOTE_RUN_DIR/results"
REMOTE_SERVER_BIN="$REMOTE_RUN_DIR/llama-server"
REMOTE_MODEL="$REMOTE_SHARED_DIR/$(basename "$MODEL_PATH")"
REMOTE_MMPROJ_BASENAME="$(basename "$MMPROJ_PATH")"
if [ -n "$BEST_CONFIG_JSON" ]; then
  REMOTE_MMPROJ_BASENAME="$(printf '%s' "$MMPROJ_PATH" | sha1sum | cut -c1-10)-$REMOTE_MMPROJ_BASENAME"
fi
REMOTE_MMPROJ="$REMOTE_SHARED_DIR/$REMOTE_MMPROJ_BASENAME"
REMOTE_TEXT_PREFILL_LOG8PV_CALIB=""
REMOTE_MMPROJ_LOG8PV_CALIB=""
if [ -n "$TEXT_PREFILL_LOG8PV_CALIB_PATH" ]; then
  REMOTE_TEXT_PREFILL_LOG8PV_CALIB="$REMOTE_SHARED_DIR/$(basename "$TEXT_PREFILL_LOG8PV_CALIB_PATH")"
  BEST_CONFIG_ENV="${BEST_CONFIG_ENV:+$BEST_CONFIG_ENV }AICAS_TEXT_PREFILL_LOG8PV_CALIB_FILE=$REMOTE_TEXT_PREFILL_LOG8PV_CALIB"
fi
if [ -n "$MMPROJ_LOG8PV_CALIB_PATH" ]; then
  REMOTE_MMPROJ_LOG8PV_CALIB="$REMOTE_SHARED_DIR/$(basename "$MMPROJ_LOG8PV_CALIB_PATH")"
  BEST_CONFIG_ENV="${BEST_CONFIG_ENV:+$BEST_CONFIG_ENV }AICAS_MMPROJ_LOG8PV_CALIB_FILE=$REMOTE_MMPROJ_LOG8PV_CALIB"
fi
REMOTE_NPU_DRIVER_KO="${REMOTE_NPU_DRIVER_KO_OVERRIDE:-$REMOTE_RUN_DIR/npu_kv260.ko}"
REMOTE_FULL_TEST_JSON="$REMOTE_DATA_DIR/FullTest.json"
REMOTE_IMAGE_ROOT="$REMOTE_DATA_DIR/images"
REMOTE_TTFT_CONFIG="$REMOTE_CODE_DIR/ttft_config.remote.json"
REMOTE_SAMPLE_JSON=""
REMOTE_THROUGHPUT_PROFILE_METRICS="$REMOTE_RESULTS_DIR/throughput_metrics_profile.json"
REMOTE_THROUGHPUT_PROFILE_ARTIFACTS="$REMOTE_RESULTS_DIR/throughput_profile_artifacts.json"
REMOTE_MTMD_SUMMARY="$REMOTE_RESULTS_DIR/mtmd_prefill_summary.json"
REMOTE_TEXT_CPU_PROFILE="$REMOTE_RESULTS_DIR/text_cpu_profile.jsonl"
REMOTE_NPU_PROFILE_JSON="$REMOTE_RESULTS_DIR/ggml_npu_profile.json"
REMOTE_NPU_PROFILE_MANIFEST="$REMOTE_RESULTS_DIR/ggml_npu_profile_manifest.json"
REMOTE_NPU_DECODE_PROFILE_JSONL="$REMOTE_RESULTS_DIR/ggml_npu_decode_profile.jsonl"
REMOTE_LOG8PV_ATTN_PROFILE_JSONL="$REMOTE_RESULTS_DIR/log8pv_attention_npu_profile.jsonl"
REMOTE_FUSED_FFN_COMPARE_JSONL="$REMOTE_RESULTS_DIR/decode_swiglu_ffn_compare.jsonl"
REMOTE_NPU_OVERLAY_PROFILE_JSONL="$REMOTE_RESULTS_DIR/npu_overlay_switch_profile.jsonl"
REMOTE_PROFILE_SERVER_LOG="$REMOTE_RUN_DIR/server_profile.log"
REMOTE_NPU_SHAPE_TABLE=""
if [ -n "$NPU_SHAPE_TABLE" ]; then
  REMOTE_NPU_SHAPE_TABLE="$REMOTE_CODE_DIR/$(basename "$NPU_SHAPE_TABLE")"
fi
REMOTE_NPU_SHAPE_RECORD="$REMOTE_RESULTS_DIR/npu_shape_record.json"
REMOTE_UBATCH_TRACE="$REMOTE_RESULTS_DIR/ubatch_trace.jsonl"
REMOTE_NPU_WAIT_TRACE="$REMOTE_RESULTS_DIR/npu_wait_trace.jsonl"
REMOTE_NPU_WAIT_POLICY_JSON=""
if [ -n "$NPU_WAIT_POLICY_JSON" ]; then
  REMOTE_NPU_WAIT_POLICY_JSON="$REMOTE_CODE_DIR/$(basename "$NPU_WAIT_POLICY_JSON")"
fi
REMOTE_NPU_OVERLAY_SWITCH="$REMOTE_RUN_DIR/switch_npu_overlay.sh"

copy_once_by_size() {
  local src="$1"
  local dst="$2"
  local label="$3"
  local local_size
  local remote_size

  local_size="$(stat -c%s "$src")"
  if [ "$FORCE_SYNC_SHARED" -ne 1 ] && ssh "${SSH_OPTS[@]}" "$TARGET" "test -f '$dst'"; then
    remote_size="$(ssh "${SSH_OPTS[@]}" "$TARGET" "stat -c%s '$dst' 2>/dev/null || wc -c < '$dst'")"
    if [ "$remote_size" = "$local_size" ]; then
      echo "[reuse] $label: $dst"
      return 0
    fi
  fi

  echo "[sync] $label -> $dst"
  ssh "${SSH_OPTS[@]}" "$TARGET" "mkdir -p '$(dirname "$dst")'"
  scp "${SSH_OPTS[@]}" "$src" "$TARGET:$dst"
}

stage_overlay_app() {
  local app="$1"
  local src_dir="$2"
  local overlay_files=()
  [ -d "$src_dir" ] || {
    echo "Overlay directory not found for $app: $src_dir" >&2
    exit 1
  }
  mapfile -t overlay_files < <(find "$src_dir" -maxdepth 1 -type f -print)
  [ "${#overlay_files[@]}" -gt 0 ] || {
    echo "Overlay directory has no regular files for $app: $src_dir" >&2
    exit 1
  }

  local remote_tmp="$REMOTE_RUN_DIR/overlay_staging/$app"
  ssh "${SSH_OPTS[@]}" "$TARGET" "rm -rf '$remote_tmp' && mkdir -p '$remote_tmp'"
  scp "${SSH_OPTS[@]}" "${overlay_files[@]}" "$TARGET:$remote_tmp/"
  ssh "${SSH_OPTS[@]}" "$TARGET" "SUDO_PASSWORD='$SUDO_PASSWORD' REMOTE_TMP='$remote_tmp' OVERLAY_APP='$app' bash -s" <<'EOF'
set -euo pipefail
echo "$SUDO_PASSWORD" | sudo -S mkdir -p "/lib/firmware/xilinx/$OVERLAY_APP"
echo "$SUDO_PASSWORD" | sudo -S cp "$REMOTE_TMP/"* "/lib/firmware/xilinx/$OVERLAY_APP/"
EOF
}

ensure_remote_overlay_app() {
  local app="$1"
  if ssh "${SSH_OPTS[@]}" "$TARGET" "test -d '/lib/firmware/xilinx/$app'"; then
    echo "[reuse] overlay app already installed on board: $app"
    return 0
  fi
  echo "Overlay app not installed on board and local directory was not found: $app" >&2
  exit 1
}

stage_overlay_apps() {
  if [ "$GENERIC_FASTPATH" -eq 1 ]; then
    if [ "$OVERLAY_APP_EXPLICIT" -eq 0 ] && [ -d "$GENERIC_PREFILL_OVERLAY_DIR" ]; then
      stage_overlay_app "$OVERLAY_APP" "$GENERIC_PREFILL_OVERLAY_DIR"
    else
      ensure_remote_overlay_app "$OVERLAY_APP"
    fi
    if [ "$DECODE_NPU" -eq 1 ]; then
      if [ -n "$DECODE_OVERLAY_DIR" ] && [ -d "$DECODE_OVERLAY_DIR" ]; then
        stage_overlay_app "$DECODE_OVERLAY_APP" "$DECODE_OVERLAY_DIR"
      else
        ensure_remote_overlay_app "$DECODE_OVERLAY_APP"
      fi
    fi
    return 0
  fi
  if [ "$OVERLAY_APP" = "versa_prefill_profile_app" ]; then
    stage_overlay_app "$OVERLAY_APP" "$VERSA_PREFILL_OVERLAY_DIR"
  fi
  if [ "$DECODE_NPU" -eq 1 ]; then
    stage_overlay_app "$DECODE_OVERLAY_APP" "$DECODE_OVERLAY_DIR"
  fi
}

build_server() {
  if [ -n "$SERVER_BIN" ]; then
    SERVER_BIN="$(realpath "$SERVER_BIN")"
    return 0
  fi

  if [ "$SKIP_BUILD" -eq 1 ]; then
    SERVER_BIN="$BUILD_DIR/bin/llama-server"
    SERVER_BIN="$(realpath "$SERVER_BIN")"
    return 0
  fi

  [ -f "$SDK_ENV" ] || {
    echo "SDK env script not found: $SDK_ENV" >&2
    exit 1
  }

  mkdir -p "$BUILD_DIR"
  (
    set -euo pipefail
    unset LD_LIBRARY_PATH
    # shellcheck disable=SC1090
    source "$SDK_ENV" >/dev/null
    cmake -S "$REPO_DIR" -B "$BUILD_DIR" -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=OFF \
      -DLLAMA_CURL=OFF \
      -DGGML_BLAS=OFF \
      -DGGML_NPU=ON \
      -DLLAMA_BUILD_TESTS=OFF \
      -DLLAMA_BUILD_EXAMPLES=OFF \
      -DLLAMA_BUILD_TOOLS=ON \
      -DLLAMA_BUILD_SERVER=ON
    cmake --build "$BUILD_DIR" --target llama-server -j"$(nproc)"
  )

  SERVER_BIN="$BUILD_DIR/bin/llama-server"
  SERVER_BIN="$(realpath "$SERVER_BIN")"
}

prepare_ttft_assets() {
  local small_image="$LOCAL_STAGE_DIR/small.jpg"
  local ttft_config="$LOCAL_STAGE_DIR/ttft_config.remote.json"

  cp "$CODE_DIR/test2.jpg" "$LOCAL_STAGE_DIR/test2.jpg"

  if python3 - "$CODE_DIR/test2.jpg" "$small_image" <<'PY'
import sys
from PIL import Image

src, dst = sys.argv[1], sys.argv[2]
img = Image.open(src)
img = img.resize((512, 512), Image.LANCZOS)
img.save(dst, "JPEG")
PY
  then
    :
  elif [ -f "$REPO_DIR/AICAS2026/aicas_semi/results/current/small.jpg" ]; then
    cp "$REPO_DIR/AICAS2026/aicas_semi/results/current/small.jpg" "$small_image"
  else
    echo "Failed to prepare TTFT small image: local Pillow missing and no fallback small.jpg found" >&2
    exit 1
  fi

  python3 - "$ttft_config" "$MODEL_ALIAS" "$PORT" <<'PY'
import json
import sys

output_path = sys.argv[1]
model_alias = sys.argv[2]
port = sys.argv[3]

cfg = {
    "server_url": f"http://127.0.0.1:{port}/v1",
    "model": model_alias,
    "images": [
        {"name": "large", "source": "test2.jpg"},
        {"name": "small", "source": "small.jpg"},
    ],
    "prompts": [
        {
            "name": "short",
            "text": "Please describe what you see in this image concisely and mention the most important elements.",
        },
        {
            "name": "long",
            "text": "Please perform a thorough visual analysis of the provided image. Begin by identifying and describing every object, person, and architectural or natural element visible, noting their spatial arrangement, relative sizes, and interactions. Examine the lighting conditions: identify the apparent light sources, direction, intensity, and the resulting highlights and shadows across different surfaces. Analyze the color palette, including dominant, accent, and background colors, and discuss how they contribute to the visual hierarchy and emotional impact of the scene. Describe the composition, noting the use of lines, frames, rule of thirds, symmetry, or asymmetry, and how these guide the viewer's attention. Evaluate the camera perspective, including angle, distance, and focus, and discuss the depth of field, foreground-midground-background relationships, and any visual layering techniques. Consider whether the image makes use of photographic or artistic principles such as leading lines, contrast, repetition, or negative space, and how they affect interpretation. Assess the mood, atmosphere, and any narrative subtext present, and relate them to the visual elements analyzed. Formulate a concluding interpretation that synthesizes all levels of analysis into a cohesive understanding of the image.",
        },
        {
            "name": "xlong",
            "text": "Please conduct an exhaustive multi-faceted analysis of the provided image. You are to operate as a team of four specialized analysts whose findings must be synthesized into a single comprehensive report. Do not summarize prematurely; each section must be fully developed with detailed observations and evidence-based reasoning.\n\nSECTION 1: FORENSIC VISUAL DECONSTRUCTION\nCatalog every discrete entity in the image: humans with clothing details, poses, expressions, gestures, apparent demographics and relationships between them; animals by species, posture, and activity; natural elements including vegetation types, geological features, water bodies, sky conditions and cloud formations; architectural structures noting building materials, architectural styles, structural condition, fenestration patterns; and manufactured objects such as furniture, vehicles, signage, tools, electronics, and decorative items. For each entity specify its depth layer as foreground, midground, background, or transitional, and its apparent state as static, dynamic, intact, or damaged.\n\nSECTION 2: ILLUMINATION AND CHROMATIC ANALYSIS\nPerform a complete lighting audit. Identify every apparent or implied light source, both natural, based on sun position relative to the scene, skylight, and reflections, and artificial, noting type, color temperature, and wattage implied by brightness. Analyze the directionality of each source and its diffusion characteristics as hard versus soft shadows. Deconstruct the full color palette: list all dominant hues with approximate hex equivalents or descriptive names, identify accent colors, and note color harmonies as complementary, analogous, or triadic. Evaluate contrast ratios between adjacent regions, highlight clipping and shadow crushing, and the overall tonal range. Discuss how lighting and color interact to create depth cues, atmospheric perspective, material differentiation, and mood modulation.\n\nSECTION 3: COMPOSITIONAL ARCHITECTURE AND VISUAL GRAMMAR\nAnalyze the geometric structure of the image. Identify all major lines including horizon lines, leading lines, and implied lines formed by gazes or gestures, and classify them as diagonal, vertical, horizontal, curved, or converging. Locate vanishing points and reconstruct the perspective system as one-point, two-point, three-point, or atmospheric. Evaluate the use of the rule of thirds, golden ratio, triangular composition, radial balance, or any other formal compositional scheme. Identify framing devices such as archways, window frames, overhanging branches, or architectural elements, and discuss how they isolate or emphasize the subject. Analyze symmetry and asymmetry, visual weight distribution, and the hierarchy of focal points. Describe the predicted eye-path a typical viewer would follow, justifying each step with compositional evidence.\n\nSECTION 4: CONTEXTUAL AND SEMANTIC INTERPRETATION\nBased on all visible evidence, deduce the setting as urban, rural, indoor, outdoor, natural, staged, candid, and estimate the probable historical period with justification. Estimate the time of day, season, and weather conditions at the moment of image capture. If the image appears to reference any cultural, historical, or artistic contexts, identify them with specific reasoning. Interpret the emotional tone of the image as joyful, melancholic, tense, serene, or chaotic, and identify which specific visual elements most strongly contribute to that tone. Formulate a thesis regarding the intended message or purpose of the image, considering whether it serves documentary, artistic, commercial, or personal functions. Provide a closing synthesis that integrates the key insights from all four analytical dimensions into a unified interpretation of the image.",
        },
    ],
    "output": "ttft_eval_results.json",
}

with open(output_path, "w", encoding="utf-8") as handle:
    json.dump(cfg, handle, indent=2, ensure_ascii=False)
PY
}

prepare_acc_sample() {
  if [ "$RUN_ACC" -ne 1 ]; then
    return 0
  fi

  if [ "$ACC_SAMPLE_MODE" = "official" ]; then
    REMOTE_SAMPLE_JSON=""
    return 0
  fi

  REMOTE_SAMPLE_JSON="$REMOTE_DATA_DIR/sample_30_available.json"
  python3 - "$REPO_DIR/AICAS/FullTest.json" "$REPO_DIR/AICAS2026/V3/payload/data/images" "$LOCAL_STAGE_DIR/sample_30_available.json" <<'PY'
import json
import os
import random
import sys
from collections import defaultdict

full_test_path, image_root, output_path = sys.argv[1:4]
random.seed(12345)

targets = [
    ("Regular Text Recognition", 3),
    ("Irregular Text Recognition", 3),
    ("Artistic Text Recognition", 3),
    ("Handwriting Recognition", 3),
    ("Digit String Recognition", 3),
    ("Non-Semantic Text Recognition", 3),
    ("Scene Text-centric VQA", 12),
]

with open(full_test_path, "r", encoding="utf-8") as handle:
    items = json.load(handle)

by_type = defaultdict(list)
for item in items:
    image_path = item.get("image_path")
    if not image_path:
      continue
    if os.path.exists(os.path.join(image_root, image_path)):
      by_type[item.get("type")].append(item)

selected = []
for type_name, target_count in targets:
    available = by_type.get(type_name, [])
    if len(available) < target_count:
        raise SystemExit(f"Not enough available samples for {type_name}: need {target_count}, have {len(available)}")
    selected.extend(random.sample(available, target_count))

with open(output_path, "w", encoding="utf-8") as handle:
    json.dump(selected, handle, ensure_ascii=False, indent=2)
PY
}

stage_remote_tree() {
  ssh "${SSH_OPTS[@]}" "$TARGET" "mkdir -p '$REMOTE_RUN_DIR' '$REMOTE_CODE_DIR' '$REMOTE_RESULTS_DIR' '$REMOTE_SHARED_DIR' '$REMOTE_LIB_DIR'"

  stage_overlay_apps

  copy_once_by_size "$SERVER_BIN" "$REMOTE_SERVER_BIN" "llama-server"
  if [ -n "$REMOTE_NPU_DRIVER_KO_OVERRIDE" ]; then
    ssh "${SSH_OPTS[@]}" "$TARGET" "test -f '$REMOTE_NPU_DRIVER_KO'" || {
      echo "Remote NPU driver not found: $REMOTE_NPU_DRIVER_KO" >&2
      exit 1
    }
    echo "[reuse] board-side npu_kv260.ko: $REMOTE_NPU_DRIVER_KO"
  else
    copy_once_by_size "$NPU_DRIVER_KO" "$REMOTE_NPU_DRIVER_KO" "npu_kv260.ko"
  fi
  copy_once_by_size "$MODEL_PATH" "$REMOTE_MODEL" "model"
  copy_once_by_size "$MMPROJ_PATH" "$REMOTE_MMPROJ" "mmproj"
  if [ -n "$TEXT_PREFILL_LOG8PV_CALIB_PATH" ]; then
    copy_once_by_size "$TEXT_PREFILL_LOG8PV_CALIB_PATH" "$REMOTE_TEXT_PREFILL_LOG8PV_CALIB" "text log8PV calibration"
  fi
  if [ -n "$MMPROJ_LOG8PV_CALIB_PATH" ]; then
    copy_once_by_size "$MMPROJ_LOG8PV_CALIB_PATH" "$REMOTE_MMPROJ_LOG8PV_CALIB" "mmproj log8PV calibration"
  fi

  for lib in libstdc++.so.6 libgcc_s.so.1 libgomp.so.1; do
    copy_once_by_size \
      "/home/gugugu/.codex/skills/kv260-llamaserver-throughput/assets/runtime_libs/$lib" \
      "$REMOTE_LIB_DIR/$lib" \
      "runtime lib $lib"
  done

  scp "${SSH_OPTS[@]}" "$CODE_DIR/"*.py "$TARGET:$REMOTE_CODE_DIR/"
  scp "${SSH_OPTS[@]}" "$LOCAL_STAGE_DIR/test2.jpg" "$LOCAL_STAGE_DIR/small.jpg" "$LOCAL_STAGE_DIR/ttft_config.remote.json" "$TARGET:$REMOTE_CODE_DIR/"
  scp "${SSH_OPTS[@]}" "$SCRIPT_DIR/run_semi_eval.sh" "$TARGET:$REMOTE_RUN_DIR/"
  if [ -n "$NPU_SHAPE_TABLE" ]; then
    scp "${SSH_OPTS[@]}" "$NPU_SHAPE_TABLE" "$TARGET:$REMOTE_NPU_SHAPE_TABLE"
  fi
  if [ -n "$NPU_WAIT_POLICY_JSON" ]; then
    scp "${SSH_OPTS[@]}" "$NPU_WAIT_POLICY_JSON" "$TARGET:$REMOTE_NPU_WAIT_POLICY_JSON"
  fi

  if [ "$RUN_ACC" -eq 1 ]; then
    ssh "${SSH_OPTS[@]}" "$TARGET" "mkdir -p '$REMOTE_DATA_DIR'"
    copy_once_by_size "$REPO_DIR/AICAS/FullTest.json" "$REMOTE_FULL_TEST_JSON" "FullTest.json"
    if [ "$FORCE_SYNC_SHARED" -eq 1 ] || ! ssh "${SSH_OPTS[@]}" "$TARGET" "test -d '$REMOTE_IMAGE_ROOT'"; then
      rm -rf "$LOCAL_STAGE_DIR/images"
      cp -R "$REPO_DIR/AICAS2026/V3/payload/data/images" "$LOCAL_STAGE_DIR/images"
      scp -r "${SSH_OPTS[@]}" "$LOCAL_STAGE_DIR/images" "$TARGET:$REMOTE_DATA_DIR/"
    fi
    if [ -n "$REMOTE_SAMPLE_JSON" ]; then
      scp "${SSH_OPTS[@]}" "$LOCAL_STAGE_DIR/sample_30_available.json" "$TARGET:$REMOTE_SAMPLE_JSON"
    fi
  fi
}

run_remote_eval() {
  local remote_acc_flag=""
  local remote_sample_flag=""
  local remote_acc_ori_flag=""

  if [ "$RUN_ACC" -eq 1 ]; then
    remote_acc_flag="--run-acc"
    if [ -n "$REMOTE_SAMPLE_JSON" ]; then
      remote_sample_flag="--sample-json '$REMOTE_SAMPLE_JSON'"
    fi
  else
    remote_acc_flag="--skip-acc"
  fi

  if [ -n "$ACC_ORI" ]; then
    remote_acc_ori_flag="--acc-ori '$ACC_ORI'"
  fi

  local npu_runtime_env_extra=""
  local env_name
  for env_name in \
    GGML_NPU_TILE_ALIGN_DEBUG \
    GGML_NPU_TILE_ALIGN_LAYER_ID \
    GGML_NPU_TILE_ALIGN_LAYER_BEGIN \
    GGML_NPU_TILE_ALIGN_LAYER_END \
    GGML_NPU_TILE_ALIGN_TILE_INDEX \
    GGML_NPU_TILE_ALIGN_MAX_LOGS \
    GGML_NPU_TILE_ALIGN_ABS_TOL \
    GGML_NPU_DUMP_TILE_DIR \
    GGML_NPU_DUMP_TILE_LAYER_ID \
    GGML_NPU_DUMP_TILE_INDEX \
    GGML_NPU_FORCE_RAW_ACC_MVOUT \
    GGML_NPU_FORCE_FP32_MVOUT \
    GGML_NPU_FORCE_RELOAD_ACTIVATIONS \
    GGML_NPU_FORCE_RELOAD_WEIGHTS \
    GGML_NPU_DEBUG_LOG; do
    local env_value="${!env_name:-}"
    if [ -n "$env_value" ]; then
      npu_runtime_env_extra+=" $env_name=$(printf "%q" "$env_value")"
    fi
  done
  local npu_runtime_env_extra_b64
  npu_runtime_env_extra_b64="$(printf '%s' "$npu_runtime_env_extra" | base64 -w0)"

  ssh "${SSH_OPTS[@]}" "$TARGET" "RUN_DIR='$REMOTE_RUN_DIR' REMOTE_LIB_DIR='$REMOTE_LIB_DIR' REMOTE_MODEL='$REMOTE_MODEL' REMOTE_MMPROJ='$REMOTE_MMPROJ' REMOTE_NPU_DRIVER_KO='$REMOTE_NPU_DRIVER_KO' NPU_DRIVER_MAX_BUFFER_MB='$NPU_DRIVER_MAX_BUFFER_MB' SUDO_PASSWORD='$SUDO_PASSWORD' OVERLAY_APP='$OVERLAY_APP' DECODE_NPU='$DECODE_NPU' DECODE_OVERLAY_APP='$DECODE_OVERLAY_APP' GENERIC_FASTPATH='$GENERIC_FASTPATH' NPU_OVERLAY_SWITCH_MODE='$NPU_OVERLAY_SWITCH_MODE' GENERIC_PREFILL_FW='$GENERIC_PREFILL_FW' GENERIC_DECODE_FW='$GENERIC_DECODE_FW' REMOTE_NPU_OVERLAY_SWITCH='$REMOTE_NPU_OVERLAY_SWITCH' PORT='$PORT' THREADS='$THREADS' UBATCH_SIZE='$UBATCH_SIZE' CACHE_TYPE_K='$CACHE_TYPE_K' CACHE_TYPE_V='$CACHE_TYPE_V' FLASH_ATTN='$FLASH_ATTN' CTX_SIZE='$CTX_SIZE' MTMD_BACKEND_DEVICE='$MTMD_BACKEND_DEVICE' MODEL_ALIAS='$MODEL_ALIAS' POWER_PATH='$POWER_PATH' SAMPLE_HZ='$SAMPLE_HZ' RUN_THROUGHPUT_ONLY='$RUN_THROUGHPUT_ONLY' RUN_ENERGY_ONLY='$RUN_ENERGY_ONLY' ENERGY_MAX_TOKENS='$ENERGY_MAX_TOKENS' RUN_CORRECTNESS_ONLY='$RUN_CORRECTNESS_ONLY' CORRECTNESS_MAX_TOKENS='$CORRECTNESS_MAX_TOKENS' CORRECTNESS_PROMPT_B64='$CORRECTNESS_PROMPT_B64' RUN_TTFT_ONLY='$RUN_TTFT_ONLY' RUN_THROUGHPUT_PROFILE='$RUN_THROUGHPUT_PROFILE' THROUGHPUT_PROFILE_ONLY='$THROUGHPUT_PROFILE_ONLY' THROUGHPUT_PROFILE_MAX_TOKENS='$THROUGHPUT_PROFILE_MAX_TOKENS' THROUGHPUT_PROFILE_PROMPT_B64='$THROUGHPUT_PROFILE_PROMPT_B64' PREFILL_PROFILE_MODE='$PREFILL_PROFILE_MODE' NPU_PROFILE_LEVEL='$NPU_PROFILE_LEVEL' PROFILE_COMPARE='$PROFILE_COMPARE' REMOTE_THROUGHPUT_PROFILE_METRICS='$REMOTE_THROUGHPUT_PROFILE_METRICS' REMOTE_THROUGHPUT_PROFILE_ARTIFACTS='$REMOTE_THROUGHPUT_PROFILE_ARTIFACTS' REMOTE_MTMD_SUMMARY='$REMOTE_MTMD_SUMMARY' REMOTE_TEXT_CPU_PROFILE='$REMOTE_TEXT_CPU_PROFILE' REMOTE_NPU_PROFILE_JSON='$REMOTE_NPU_PROFILE_JSON' REMOTE_NPU_PROFILE_MANIFEST='$REMOTE_NPU_PROFILE_MANIFEST' REMOTE_NPU_DECODE_PROFILE_JSONL='$REMOTE_NPU_DECODE_PROFILE_JSONL' REMOTE_LOG8PV_ATTN_PROFILE_JSONL='$REMOTE_LOG8PV_ATTN_PROFILE_JSONL' REMOTE_FUSED_FFN_COMPARE_JSONL='$REMOTE_FUSED_FFN_COMPARE_JSONL' REMOTE_NPU_OVERLAY_PROFILE_JSONL='$REMOTE_NPU_OVERLAY_PROFILE_JSONL' REMOTE_PROFILE_SERVER_LOG='$REMOTE_PROFILE_SERVER_LOG' REMOTE_NPU_SHAPE_TABLE='$REMOTE_NPU_SHAPE_TABLE' NPU_SHAPE_RECORD='$NPU_SHAPE_RECORD' NPU_PRELOAD_WEIGHTS='$NPU_PRELOAD_WEIGHTS' REMOTE_NPU_SHAPE_RECORD='$REMOTE_NPU_SHAPE_RECORD' NPU_WAIT_POLICY='$NPU_WAIT_POLICY' NPU_WAIT_TRACE='$NPU_WAIT_TRACE' REMOTE_NPU_WAIT_TRACE='$REMOTE_NPU_WAIT_TRACE' REMOTE_NPU_WAIT_POLICY_JSON='$REMOTE_NPU_WAIT_POLICY_JSON' TRACE_UBATCH='$TRACE_UBATCH' REMOTE_UBATCH_TRACE='$REMOTE_UBATCH_TRACE' MERGE_PREFILL='$MERGE_PREFILL' MERGE_PREFILL_TRACE='$MERGE_PREFILL_TRACE' TTFT_CACHE_PROMPT='$TTFT_CACHE_PROMPT' NPU_TEXT_PREFILL_DYNAMIC='$NPU_TEXT_PREFILL_DYNAMIC' NPU_RUNTIME_ENV_EXTRA_B64='$npu_runtime_env_extra_b64' BEST_CONFIG_ENV='$BEST_CONFIG_ENV' bash -s" <<EOF
set -euo pipefail

cd "\$RUN_DIR"
rm -f server.log server.pid
mkdir -p results
THROUGHPUT_PROFILE_PROMPT="\$(printf '%s' "\$THROUGHPUT_PROFILE_PROMPT_B64" | base64 -d)"
CORRECTNESS_PROMPT="\$(printf '%s' "\$CORRECTNESS_PROMPT_B64" | base64 -d)"

cleanup() {
  if [ -f server.pid ]; then
    local pid
    pid="\$(cat server.pid)"
    if [ -n "\$pid" ]; then
      kill "\$pid" >/dev/null 2>&1 || true
    fi
  fi
}
trap cleanup EXIT

echo "\$SUDO_PASSWORD" | sudo -S rmmod npu_kv260 >/dev/null 2>&1 || true
echo "\$SUDO_PASSWORD" | sudo -S xmutil unloadapp >/dev/null 2>&1 || true
echo "\$SUDO_PASSWORD" | sudo -S rmmod npu_kv260 >/dev/null 2>&1 || true
echo "\$SUDO_PASSWORD" | sudo -S xmutil loadapp "\$OVERLAY_APP"
if lsmod | grep -q '^npu_kv260 '; then
  echo "\$SUDO_PASSWORD" | sudo -S rmmod npu_kv260 >/dev/null 2>&1 || true
fi
if lsmod | grep -q '^npu_kv260 '; then
  echo "[reuse] loaded npu_kv260 module"
else
  echo "\$SUDO_PASSWORD" | sudo -S insmod "\$REMOTE_NPU_DRIVER_KO" max_buffer_mb="\$NPU_DRIVER_MAX_BUFFER_MB"
fi
for _ in \$(seq 1 30); do
  [ -e /dev/npu_kv260 ] && break
  sleep 1
done
if [ ! -e /dev/npu_kv260 ]; then
  echo "NPU device did not appear after loading \$OVERLAY_APP" >&2
  lsmod | grep npu_kv260 >&2 || true
  ls -l /sys/bus/platform/devices/*Versa* /sys/bus/platform/drivers/npu_kv260 2>&2 || true
  exit 1
fi
echo "\$SUDO_PASSWORD" | sudo -S chmod 666 /dev/npu_kv260
printf '%s\n' "\$OVERLAY_APP" > /tmp/aicas_npu_overlay_state

if [ "\$GENERIC_FASTPATH" = "1" ] && [ "\$NPU_OVERLAY_SWITCH_MODE" = "fast" ]; then
cat > "\$REMOTE_NPU_OVERLAY_SWITCH" <<'EOSWITCH'
#!/usr/bin/env bash
set -euo pipefail
target_app="\$1"
state_file="/tmp/aicas_npu_overlay_state"
prefill_app="\${GENERIC_PREFILL_APP:?}"
decode_app="\${GENERIC_DECODE_APP:?}"
prefill_fw="\${GENERIC_PREFILL_FW:?}"
decode_fw="\${GENERIC_DECODE_FW:?}"
current=""
if [ -f "\$state_file" ]; then
  current="\$(cat "\$state_file" 2>/dev/null || true)"
fi
if [ "\$current" = "\$target_app" ] && [ -e /dev/npu_kv260 ]; then
  echo "NPU overlay already active: \$target_app"
  exit 0
fi
case "\$target_app" in
  "\$prefill_app") firmware="\$prefill_fw" ;;
  "\$decode_app") firmware="\$decode_fw" ;;
  *)
    echo "Unknown generic NPU overlay target: \$target_app" >&2
    exit 1
    ;;
esac
echo "Switching NPU bitstream: \${current:-none} -> \$target_app (\$firmware)"
echo "\$SUDO_PASSWORD" | sudo -S sh -c "echo 0 > /sys/class/fpga_manager/fpga0/flags"
echo "\$SUDO_PASSWORD" | sudo -S sh -c "echo '\$firmware' > /sys/class/fpga_manager/fpga0/firmware"
for _ in \$(seq 1 60); do
  state="\$(cat /sys/class/fpga_manager/fpga0/state 2>/dev/null || true)"
  [ "\$state" = "operating" ] && break
  sleep 0.1
done
state="\$(cat /sys/class/fpga_manager/fpga0/state 2>/dev/null || true)"
if [ "\$state" != "operating" ]; then
  echo "FPGA manager did not reach operating after loading \$firmware; state=\$state" >&2
  exit 1
fi
if [ ! -e /dev/npu_kv260 ]; then
  echo "NPU device missing after bitstream switch to \$target_app; generic DT/driver must stay loaded" >&2
  exit 1
fi
echo "\$SUDO_PASSWORD" | sudo -S chmod 666 /dev/npu_kv260
printf '%s\n' "\$target_app" > "\$state_file"
EOSWITCH
else
cat > "\$REMOTE_NPU_OVERLAY_SWITCH" <<'EOSWITCH'
#!/usr/bin/env bash
set -euo pipefail
target_app="\$1"
state_file="/tmp/aicas_npu_overlay_state"
current=""
if [ -f "\$state_file" ]; then
  current="\$(cat "\$state_file" 2>/dev/null || true)"
fi
if [ "\$current" = "\$target_app" ] && [ -e /dev/npu_kv260 ]; then
  echo "NPU overlay already active: \$target_app"
  exit 0
fi
echo "Switching NPU overlay with xmutil: \${current:-none} -> \$target_app"
echo "\$SUDO_PASSWORD" | sudo -S xmutil unloadapp >/dev/null 2>&1 || true
echo "\$SUDO_PASSWORD" | sudo -S xmutil loadapp "\$target_app"
if ! lsmod | grep -q '^npu_kv260 '; then
  echo "\$SUDO_PASSWORD" | sudo -S insmod "\$REMOTE_NPU_DRIVER_KO" max_buffer_mb="\${NPU_DRIVER_MAX_BUFFER_MB:-1500}"
fi
for _ in \$(seq 1 30); do
  [ -e /dev/npu_kv260 ] && break
  sleep 1
done
if [ ! -e /dev/npu_kv260 ]; then
  echo "NPU device missing after xmutil switch to \$target_app" >&2
  exit 1
fi
echo "\$SUDO_PASSWORD" | sudo -S chmod 666 /dev/npu_kv260
printf '%s\n' "\$target_app" > "\$state_file"
EOSWITCH
fi
chmod +x "\$REMOTE_NPU_OVERLAY_SWITCH"

wait_ready() {
  local ready=0
  for _ in \$(seq 1 120); do
    if python3 - <<PY
import json
import urllib.request

with urllib.request.urlopen('http://127.0.0.1:$PORT/v1/models', timeout=5) as resp:
    payload = json.load(resp)

assert any(item.get('id') == '$MODEL_ALIAS' for item in payload.get('data', [])), payload
PY
    then
      ready=1
      break
    fi
    sleep 2
  done
  [ "\$ready" -eq 1 ]
}

stop_server() {
  if [ -f server.pid ]; then
    local pid
    pid="\$(cat server.pid)"
    if [ -n "\$pid" ]; then
      echo "\$SUDO_PASSWORD" | sudo -S kill "\$pid" >/dev/null 2>&1 || true
      for _ in \$(seq 1 40); do
        if ! echo "\$SUDO_PASSWORD" | sudo -S kill -0 "\$pid" >/dev/null 2>&1; then
          break
        fi
        sleep 0.25
      done
      if echo "\$SUDO_PASSWORD" | sudo -S kill -0 "\$pid" >/dev/null 2>&1; then
        echo "\$SUDO_PASSWORD" | sudo -S kill -9 "\$pid" >/dev/null 2>&1 || true
      fi
      wait "\$pid" >/dev/null 2>&1 || true
    fi
    rm -f server.pid
  fi
  echo "\$SUDO_PASSWORD" | sudo -S pkill -f "./llama-server --host 127.0.0.1 --port \$PORT" >/dev/null 2>&1 || true
}

start_server() {
  local log_path="\$1"
  local enable_profile="\$2"
  stop_server
  pkill -f "./llama-server --host 127.0.0.1 --port \$PORT" >/dev/null 2>&1 || true

  local profile_env=""
  local fused_ffn_compare_env=""
  local npu_shape_env=""
  local ubatch_trace_env=""
  local decode_overlay_env=""
  local wait_env=""
  local ctx_arg=""
  local npu_runtime_env="GGML_NPU_EAGER_INIT=1 VERSA_P_CMA_SIZE=1408M VERSA_P_CMA_HEAP_OFFSET=0 VERSA_P_CMA_HEAP_SIZE=832M NPU_CMA_SIZE=1408M NPU_CMA_HEAP_OFFSET=832M NPU_CMA_HEAP_SIZE=576M"
  if [ -n "\$NPU_RUNTIME_ENV_EXTRA_B64" ]; then
    npu_runtime_env="\$npu_runtime_env \$(printf '%s' "\$NPU_RUNTIME_ENV_EXTRA_B64" | base64 -d)"
  fi
  local text_prefill_env="LLAMA_MTMD_MERGE_PREFILL='\$MERGE_PREFILL' LLAMA_MTMD_MERGE_PREFILL_TRACE='\$MERGE_PREFILL_TRACE' GGML_NPU_TEXT_PREFILL_DYNAMIC='\$NPU_TEXT_PREFILL_DYNAMIC'"
  if [ "\$DECODE_NPU" = "1" ]; then
    decode_overlay_env="AICAS_TEXT_DECODE_AWQ_NPU=1 AICAS_TEXT_DECODE_AWQ_NPU_REQUIRE_ACTIVE=1 AICAS_NPU_PREFILL_SWITCH_CMD='SUDO_PASSWORD=\"\$SUDO_PASSWORD\" REMOTE_NPU_DRIVER_KO=\"\$REMOTE_NPU_DRIVER_KO\" NPU_DRIVER_MAX_BUFFER_MB=\"\$NPU_DRIVER_MAX_BUFFER_MB\" GENERIC_PREFILL_APP=\"\$OVERLAY_APP\" GENERIC_DECODE_APP=\"\$DECODE_OVERLAY_APP\" GENERIC_PREFILL_FW=\"\$GENERIC_PREFILL_FW\" GENERIC_DECODE_FW=\"\$GENERIC_DECODE_FW\" \"\$REMOTE_NPU_OVERLAY_SWITCH\" \"\$OVERLAY_APP\"' AICAS_NPU_DECODE_SWITCH_CMD='SUDO_PASSWORD=\"\$SUDO_PASSWORD\" REMOTE_NPU_DRIVER_KO=\"\$REMOTE_NPU_DRIVER_KO\" NPU_DRIVER_MAX_BUFFER_MB=\"\$NPU_DRIVER_MAX_BUFFER_MB\" GENERIC_PREFILL_APP=\"\$OVERLAY_APP\" GENERIC_DECODE_APP=\"\$DECODE_OVERLAY_APP\" GENERIC_PREFILL_FW=\"\$GENERIC_PREFILL_FW\" GENERIC_DECODE_FW=\"\$GENERIC_DECODE_FW\" \"\$REMOTE_NPU_OVERLAY_SWITCH\" \"\$DECODE_OVERLAY_APP\"'"
    if [ "\$RUN_CORRECTNESS_ONLY" = "1" ] || { [ "\$enable_profile" = "1" ] && [ "\$PROFILE_COMPARE" = "1" ]; }; then
      rm -f "\$REMOTE_FUSED_FFN_COMPARE_JSONL"
      fused_ffn_compare_env="AICAS_TEXT_DECODE_AWQ_FUSED_FFN_COMPARE=1 AICAS_TEXT_DECODE_AWQ_FUSED_FFN_COMPARE_LIMIT=1 AICAS_TEXT_DECODE_AWQ_FUSED_FFN_COMPARE_JSONL='\$REMOTE_FUSED_FFN_COMPARE_JSONL'"
    fi
  fi
  local best_config_env="\$BEST_CONFIG_ENV"
  if [ -n "\$REMOTE_NPU_SHAPE_TABLE" ]; then
    npu_shape_env="GGML_NPU_SHAPE_TABLE_JSON='\$REMOTE_NPU_SHAPE_TABLE'"
  fi
  if [ "\$NPU_PRELOAD_WEIGHTS" = "1" ]; then
    npu_shape_env="\$npu_shape_env GGML_NPU_PRELOAD_WEIGHTS_ON_LOAD=1"
    if [ "\$DECODE_NPU" = "1" ]; then
      npu_shape_env="\$npu_shape_env AICAS_TEXT_DECODE_AWQ_NPU_PRELOAD=1"
    fi
  fi
  if [ "\$NPU_SHAPE_RECORD" = "1" ]; then
    rm -f "\$REMOTE_NPU_SHAPE_RECORD"
    npu_shape_env="\$npu_shape_env GGML_NPU_SHAPE_RECORD_JSON='\$REMOTE_NPU_SHAPE_RECORD' GGML_NPU_PROFILE_TILING_SEARCH=1"
  fi
  if [ "\$enable_profile" = "1" ]; then
    rm -f "\$REMOTE_THROUGHPUT_PROFILE_METRICS" "\$REMOTE_THROUGHPUT_PROFILE_ARTIFACTS" "\$REMOTE_MTMD_SUMMARY" "\$REMOTE_TEXT_CPU_PROFILE" "\$REMOTE_NPU_PROFILE_JSON" "\$REMOTE_NPU_PROFILE_MANIFEST" "\$REMOTE_NPU_DECODE_PROFILE_JSONL" "\$REMOTE_LOG8PV_ATTN_PROFILE_JSONL" "\$REMOTE_NPU_OVERLAY_PROFILE_JSONL"
    profile_env="LLAMA_MTMD_PREFILL_SUMMARY_JSON='\$REMOTE_MTMD_SUMMARY' AICAS_LOG8PV_NPU_PROFILE_JSONL='\$REMOTE_LOG8PV_ATTN_PROFILE_JSONL' AICAS_NPU_OVERLAY_PROFILE_JSONL='\$REMOTE_NPU_OVERLAY_PROFILE_JSONL'"
    if [ "\$PREFILL_PROFILE_MODE" = "aggregate" ]; then
      profile_env="\$profile_env LLAMA_MTMD_CPU_OP_PROFILE=aggregate LLAMA_TEXT_CPU_PROFILE_JSON='\$REMOTE_TEXT_CPU_PROFILE' LLAMA_TEXT_CPU_PROFILE_MODE=aggregate GGML_NPU_PROFILE_JSON='\$REMOTE_NPU_PROFILE_JSON' GGML_NPU_PROFILE_MANIFEST_JSON='\$REMOTE_NPU_PROFILE_MANIFEST' GGML_NPU_PROFILE_LEVEL=aggregate"
    elif [ "\$PREFILL_PROFILE_MODE" = "diagnostic" ]; then
      profile_env="\$profile_env LLAMA_MTMD_CPU_OP_PROFILE=1 LLAMA_TEXT_CPU_PROFILE_JSON='\$REMOTE_TEXT_CPU_PROFILE' GGML_NPU_PROFILE_JSON='\$REMOTE_NPU_PROFILE_JSON' GGML_NPU_PROFILE_MANIFEST_JSON='\$REMOTE_NPU_PROFILE_MANIFEST' GGML_NPU_PROFILE_LEVEL='\$NPU_PROFILE_LEVEL' GGML_NPU_DECODE_PROFILE_JSONL='\$REMOTE_NPU_DECODE_PROFILE_JSONL' GGML_NPU_DECODE_PROFILE_FLUSH_RECORDS='\${GGML_NPU_DECODE_PROFILE_FLUSH_RECORDS:-100000}'"
    fi
  fi
  if [ "\$TRACE_UBATCH" = "1" ]; then
    rm -f "\$REMOTE_UBATCH_TRACE"
    ubatch_trace_env="LLAMA_UBATCH_TRACE_JSONL='\$REMOTE_UBATCH_TRACE'"
  fi
  if [ -n "\$NPU_WAIT_POLICY" ]; then
    wait_env="\$wait_env NPU_WAIT_POLICY='\$NPU_WAIT_POLICY'"
  fi
  if [ -n "\$REMOTE_NPU_WAIT_POLICY_JSON" ]; then
    wait_env="\$wait_env NPU_WAIT_POLICY_JSON='\$REMOTE_NPU_WAIT_POLICY_JSON'"
  fi
  if [ "\$NPU_WAIT_TRACE" = "1" ]; then
    rm -f "\$REMOTE_NPU_WAIT_TRACE"
    wait_env="\$wait_env NPU_WAIT_TRACE_JSONL='\$REMOTE_NPU_WAIT_TRACE'"
  fi
  local log_disable_arg="--log-disable"
  local server_env_for_log_check="\$npu_runtime_env \$best_config_env \$npu_shape_env \$profile_env \$ubatch_trace_env \$wait_env"
  if [ "\$MERGE_PREFILL_TRACE" = "1" ] || printf '%s' "\$server_env_for_log_check" | grep -q 'GGML_NPU_TILE_ALIGN_DEBUG\\|GGML_NPU_DEBUG_LOG\\|GGML_NPU_PROFILE_TILING_SEARCH\\|AICAS_MMPROJ_W8A8_DEBUG\\|AICAS_TEXT_DECODE_AWQ_NPU_DEBUG'; then
    log_disable_arg=""
  fi
  if [ -n "\$CTX_SIZE" ]; then
    ctx_arg="--ctx-size '\$CTX_SIZE'"
  fi

  bash -lc "cd '\$RUN_DIR' && env LD_LIBRARY_PATH='\$REMOTE_LIB_DIR:\${LD_LIBRARY_PATH:-}' MTMD_BACKEND_DEVICE='\$MTMD_BACKEND_DEVICE' \$npu_runtime_env \$text_prefill_env \$decode_overlay_env \$best_config_env \$npu_shape_env \$profile_env \$fused_ffn_compare_env \$ubatch_trace_env \$wait_env ./llama-server --host 127.0.0.1 --port '\$PORT' --alias '\$MODEL_ALIAS' -m '\$REMOTE_MODEL' --mmproj '\$REMOTE_MMPROJ' --cache-type-k '\$CACHE_TYPE_K' --cache-type-v '\$CACHE_TYPE_V' --flash-attn '\$FLASH_ATTN' \$ctx_arg -t '\$THREADS' --ubatch-size '\$UBATCH_SIZE' \$log_disable_arg --no-warmup > '\$log_path' 2>&1 & echo \\\$! > server.pid"

  if ! wait_ready; then
    tail -n 120 "\$log_path" >&2 || true
    echo "llama-server did not become ready" >&2
    exit 1
  fi
}

if [ "\$RUN_CORRECTNESS_ONLY" = "1" ] && [ "\$THROUGHPUT_PROFILE_ONLY" -ne 1 ]; then
  start_server server.log 0

  correctness_cmd=(python3 "\$RUN_DIR/code/throughput_eval.py" \
    -i "\$RUN_DIR/code/test2.jpg" \
    -o "\$RUN_DIR/results/correctness_metrics.json" \
    --base-url "http://127.0.0.1:\$PORT/v1" \
    --model "\$MODEL_ALIAS" \
    --max-tokens "\$CORRECTNESS_MAX_TOKENS")
  if [ -n "\$CORRECTNESS_PROMPT" ]; then
    correctness_cmd+=(--prompt "\$CORRECTNESS_PROMPT")
  fi
  "\${correctness_cmd[@]}"

  stop_server
elif [ "\$RUN_THROUGHPUT_ONLY" = "1" ] && [ "\$THROUGHPUT_PROFILE_ONLY" -ne 1 ]; then
  start_server server.log 0

  python3 "\$RUN_DIR/code/throughput_eval.py" \
    -i "\$RUN_DIR/code/test2.jpg" \
    -o "\$RUN_DIR/results/throughput_metrics.json" \
    --base-url "http://127.0.0.1:\$PORT/v1" \
    --model "\$MODEL_ALIAS"

  stop_server
elif [ "\$RUN_ENERGY_ONLY" = "1" ] && [ "\$THROUGHPUT_PROFILE_ONLY" -ne 1 ]; then
  start_server server.log 0

  python3 "\$RUN_DIR/code/energy_eval.py" \
    -i "\$RUN_DIR/code/test2.jpg" \
    -o "\$RUN_DIR/results/energy_metrics.json" \
    --base-url "http://127.0.0.1:\$PORT/v1" \
    --model "\$MODEL_ALIAS" \
    --power_path "\$POWER_PATH" \
    --sample_hz "\$SAMPLE_HZ" \
    --max-tokens "\$ENERGY_MAX_TOKENS" \
    --request-timeout 1800

  stop_server
elif [ "\$RUN_TTFT_ONLY" = "1" ] && [ "\$THROUGHPUT_PROFILE_ONLY" -ne 1 ]; then
  ttft_profile=0
  if [ "\$TRACE_UBATCH" = "1" ] || [ "\$NPU_SHAPE_RECORD" = "1" ] || [ "\$MERGE_PREFILL_TRACE" = "1" ]; then
    ttft_profile=1
  fi
  start_server server.log "\$ttft_profile"
  sleep 60
  ttft_cache_arg=()
  if [ "\$TTFT_CACHE_PROMPT" = "1" ]; then
    ttft_cache_arg=(--cache-prompt)
  elif [ "\$TTFT_CACHE_PROMPT" = "0" ]; then
    ttft_cache_arg=(--no-cache-prompt)
  fi

  if ! python3 "\$RUN_DIR/code/ttft_eval_multiprompt.py" \
      -c "\$RUN_DIR/code/ttft_config.remote.json" \
      -o "\$RUN_DIR/results/ttft_eval_results.json" \
      "\${ttft_cache_arg[@]}" \
      --request-timeout 3600; then
    echo "[ttft-only] first TTFT attempt failed; retrying after multimodal lazy init" >&2
    sleep 30
    if ! python3 "\$RUN_DIR/code/ttft_eval_multiprompt.py" \
        -c "\$RUN_DIR/code/ttft_config.remote.json" \
        -o "\$RUN_DIR/results/ttft_eval_results.json" \
        "\${ttft_cache_arg[@]}" \
        --request-timeout 3600; then
      stop_server
      exit 1
    fi
  fi

  stop_server
elif [ "\$THROUGHPUT_PROFILE_ONLY" -ne 1 ]; then
  start_server server.log 0

  bash ./run_semi_eval.sh \
    --code-dir "\$RUN_DIR/code" \
    --throughput-image "\$RUN_DIR/code/test2.jpg" \
    --ttft-config "\$RUN_DIR/code/ttft_config.remote.json" \
    --output-dir "\$RUN_DIR/results" \
    --base-url "http://127.0.0.1:\$PORT/v1" \
    --model "\$MODEL_ALIAS" \
    --power-path "\$POWER_PATH" \
    --sample-hz "\$SAMPLE_HZ" \
    $remote_acc_flag \
    $remote_sample_flag \
    $remote_acc_ori_flag \
    $( [ "$RUN_ACC" -eq 1 ] && printf '%s ' --full-test-json "'$REMOTE_FULL_TEST_JSON'" --image-root "'$REMOTE_IMAGE_ROOT'" )
fi

if [ "\$RUN_THROUGHPUT_PROFILE" -eq 1 ]; then
  start_server "\$REMOTE_PROFILE_SERVER_LOG" 1

  profile_eval_cmd=(python3 "\$RUN_DIR/code/throughput_eval.py" \
    -i "\$RUN_DIR/code/test2.jpg" \
    -o "\$REMOTE_THROUGHPUT_PROFILE_METRICS" \
    --base-url "http://127.0.0.1:\$PORT/v1" \
    --model "\$MODEL_ALIAS" \
    --max-tokens "\$THROUGHPUT_PROFILE_MAX_TOKENS")
  if [ -n "\$THROUGHPUT_PROFILE_PROMPT" ]; then
    profile_eval_cmd+=(--prompt "\$THROUGHPUT_PROFILE_PROMPT")
  fi
  "\${profile_eval_cmd[@]}"

  stop_server
  for _ in \$(seq 1 40); do
    if [ -f "\$REMOTE_NPU_DECODE_PROFILE_JSONL" ]; then
      break
    fi
    sleep 0.25
  done

  python3 - "\$REMOTE_THROUGHPUT_PROFILE_ARTIFACTS" "\$REMOTE_THROUGHPUT_PROFILE_METRICS" "\$REMOTE_MTMD_SUMMARY" "\$REMOTE_TEXT_CPU_PROFILE" "\$REMOTE_NPU_PROFILE_JSON" "\$REMOTE_NPU_PROFILE_MANIFEST" "\$REMOTE_NPU_DECODE_PROFILE_JSONL" "\$REMOTE_FUSED_FFN_COMPARE_JSONL" "\$REMOTE_NPU_OVERLAY_PROFILE_JSONL" "\$NPU_PROFILE_LEVEL" "\$PREFILL_PROFILE_MODE" <<'PY'
import json
import os
import sys

out_path, metrics_path, mtmd_path, text_cpu_path, npu_path, manifest_path, decode_path, fused_ffn_path, overlay_path, npu_level, prefill_mode = sys.argv[1:12]

def info(path):
    return {
        "path": path,
        "exists": bool(path) and os.path.exists(path),
        "bytes": os.path.getsize(path) if path and os.path.exists(path) else 0,
    }

payload = {
    "profile_kind": "aicas_semi_kv260_throughput_profile",
    "metrics": info(metrics_path),
    "mtmd_prefill_summary": info(mtmd_path),
    "text_cpu_profile": info(text_cpu_path),
    "ggml_npu_profile": info(npu_path),
    "ggml_npu_manifest": info(manifest_path),
    "ggml_npu_decode_profile": info(decode_path),
    "decode_swiglu_ffn_compare": info(fused_ffn_path),
    "npu_overlay_switch_profile": info(overlay_path),
    "npu_profile_level": npu_level,
    "prefill_profile_mode": prefill_mode,
}

with open(out_path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=False, indent=2)
PY
fi

if [ "\$NPU_SHAPE_RECORD" = "1" ]; then
  python3 - "\$REMOTE_NPU_SHAPE_RECORD" "\$RUN_DIR/server.log" "\$REMOTE_PROFILE_SERVER_LOG" <<'PY'
import json
import os
import re
import sys
from collections import Counter

out_path, *log_paths = sys.argv[1:]
pat = re.compile(
    r"M=(?P<M>\d+) N=(?P<N>\d+) K=(?P<K>\d+).*?"
    r"valid=(?P<valid>\d+) tm=(?P<tm>\d+) tn=(?P<tn>\d+) tk=(?P<tk>\d+)"
)
counter = Counter()
for path in log_paths:
    if not path or not os.path.exists(path):
        continue
    source = os.path.basename(path)
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for lineno, line in enumerate(handle, 1):
            if "npu_create_mul_mat_plan:" not in line:
                continue
            m = pat.search(line)
            if not m:
                continue
            key = (
                source,
                int(m.group("M")),
                int(m.group("N")),
                int(m.group("K")),
                int(m.group("valid")),
                int(m.group("tm")),
                int(m.group("tn")),
                int(m.group("tk")),
            )
            counter[key] += 1

records = [
    {
        "source": source,
        "M": M,
        "N": N,
        "K": K,
        "valid": bool(valid),
        "tm": tm,
        "tn": tn,
        "tk": tk,
        "count": count,
    }
    for (source, M, N, K, valid, tm, tn, tk), count in sorted(counter.items())
]

with open(out_path, "w", encoding="utf-8") as handle:
    json.dump({
        "profile_kind": "aicas_semi_kv260_npu_shape_record",
        "records": records,
    }, handle, ensure_ascii=False, indent=2)
PY
fi
EOF
}

pull_results() {
  scp "${SSH_OPTS[@]}" "$TARGET:$REMOTE_RUN_DIR/server.log" "$LOCAL_RUN_DIR/" >/dev/null 2>&1 || true
  if [ "$RUN_THROUGHPUT_PROFILE" -eq 1 ]; then
    scp "${SSH_OPTS[@]}" "$TARGET:$REMOTE_RUN_DIR/server_profile.log" "$LOCAL_RUN_DIR/" >/dev/null 2>&1 || true
  fi
  scp -r "${SSH_OPTS[@]}" "$TARGET:$REMOTE_RESULTS_DIR/." "$LOCAL_RESULT_DIR/" >/dev/null
}

build_server
prepare_ttft_assets
prepare_acc_sample
stage_remote_tree
run_remote_eval
pull_results

echo "Local results: $LOCAL_RESULT_DIR"
