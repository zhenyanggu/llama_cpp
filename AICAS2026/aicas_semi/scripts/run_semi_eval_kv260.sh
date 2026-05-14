#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_semi_eval_kv260.sh [options]

Runs the AICAS 2026 semi-final eval on a KV260 board with a sudo-launched
llama-server and shared-asset reuse.

Options:
  --host <host>                 KV260 host (default: 192.168.0.10)
  --user <user>                 KV260 user (default: ubuntu)
  --remote-root <path>          Remote root (default: /home/ubuntu/aicas-semi)
  --sdk-env <path>              KV260 SDK env script
  --build-dir <path>            Cross-build dir (default: build-kv260-semi)
  --server-bin <path>           Reuse existing KV260 llama-server binary
  --skip-build                  Skip cross-build and reuse --server-bin/build output
  --model <path>                Text model GGUF path
  --mmproj <path>               mmproj GGUF path
  --overlay-app <name>          Overlay app to load (required, e.g. GEMM_v6_app)
  --sudo-password <password>    Board sudo password (default: 123456)
  --port <port>                 llama-server port (default: 8080)
  --threads <n>                 llama-server threads (default: 4)
  --ubatch-size <n>             llama-server physical ubatch size (default: 1024)
  --mtmd-backend-device <name>  MTMD_BACKEND_DEVICE for mmproj (default: NPU)
  --alias <name>                Model alias (default: smolvlm2-gguf)
  --output-dir <path>           Local results root
  --run-id <id>                 Override run id
  --run-acc                     Run accuracy (default)
  --skip-acc                    Skip accuracy and run throughput/energy/ttft only
  --run-throughput-profile      Run an extra profiled throughput pass after normal eval
  --throughput-profile-only     Only run the profiled throughput pass
  --throughput-profile-max-tokens <n>
                                Max generated tokens for the profiled pass (default: 1)
  --throughput-profile-prompt <text>
                                Prompt for profiled pass (default: short image description)
  --npu-profile-level <level>   GGML_NPU_PROFILE_LEVEL for profile pass (default: diagnostic)
  --npu-shape-table <path>      Optional GGML_NPU_SHAPE_TABLE_JSON file for text W8A8 GEMM fast path
  --npu-shape-record            Record observed text W8A8 GEMM shapes to results/npu_shape_record.json
  --trace-ubatch                Record pre/post microbatch sizes to results/ubatch_trace.jsonl
  --merge-prefill               Merge multimodal prompt into one embedding prefill (default)
  --no-merge-prefill            Disable merged multimodal embedding prefill
  --npu-text-prefill-dynamic    Enable dynamic NPU text prefill GEMM shapes (default)
  --no-npu-text-prefill-dynamic Disable dynamic NPU text prefill GEMM shapes
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

HOST="192.168.0.10"
USER_NAME="ubuntu"
REMOTE_ROOT="/home/ubuntu/aicas-semi"
SDK_ENV="/home/gugugu/petalinux/sdk/kv260-2025.1/environment-setup-cortexa72-cortexa53-amd-linux"
BUILD_DIR="$REPO_DIR/build-kv260-semi"
SERVER_BIN=""
SKIP_BUILD=0
MODEL_PATH="$REPO_DIR/AICAS/output/text-decode-awq-repro/calib16-a0p125-g32/text_sq_prefill_decode_awq_calib16_a0p125_g32_kvq8_scale_f16.gguf"
MMPROJ_PATH="$REPO_DIR/AICAS/output/smoothquant/full-alpha-0_5-minmax/mmproj.gguf"
OVERLAY_APP=""
SUDO_PASSWORD="${BOARD_SUDO_PASSWORD:-123456}"
PORT="8080"
THREADS="4"
UBATCH_SIZE="1024"
MTMD_BACKEND_DEVICE="NPU"
MODEL_ALIAS="smolvlm2-gguf"
OUTPUT_ROOT="$REPO_DIR/AICAS2026/aicas_semi/results/kv260"
RUN_ID=""
RUN_ACC=1
RUN_THROUGHPUT_PROFILE=0
THROUGHPUT_PROFILE_ONLY=0
THROUGHPUT_PROFILE_MAX_TOKENS="1"
THROUGHPUT_PROFILE_PROMPT="Describe this image briefly."
NPU_PROFILE_LEVEL="diagnostic"
NPU_SHAPE_TABLE=""
if [ -f "$DEFAULT_NPU_SHAPE_TABLE" ]; then
  NPU_SHAPE_TABLE="$DEFAULT_NPU_SHAPE_TABLE"
fi
NPU_SHAPE_RECORD=0
TRACE_UBATCH=0
MERGE_PREFILL=1
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
    --model) MODEL_PATH="$2"; shift 2 ;;
    --mmproj) MMPROJ_PATH="$2"; shift 2 ;;
    --overlay-app) OVERLAY_APP="$2"; shift 2 ;;
    --sudo-password) SUDO_PASSWORD="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --ubatch-size) UBATCH_SIZE="$2"; shift 2 ;;
    --mtmd-backend-device) MTMD_BACKEND_DEVICE="$2"; shift 2 ;;
    --alias) MODEL_ALIAS="$2"; shift 2 ;;
    --output-dir) OUTPUT_ROOT="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --run-acc) RUN_ACC=1; shift ;;
    --skip-acc) RUN_ACC=0; shift ;;
    --run-throughput-profile) RUN_THROUGHPUT_PROFILE=1; shift ;;
    --throughput-profile-only) RUN_THROUGHPUT_PROFILE=1; THROUGHPUT_PROFILE_ONLY=1; shift ;;
    --throughput-profile-max-tokens) THROUGHPUT_PROFILE_MAX_TOKENS="$2"; shift 2 ;;
    --throughput-profile-prompt) THROUGHPUT_PROFILE_PROMPT="$2"; shift 2 ;;
    --npu-profile-level) NPU_PROFILE_LEVEL="$2"; shift 2 ;;
    --npu-shape-table) NPU_SHAPE_TABLE="$2"; shift 2 ;;
    --npu-shape-record) NPU_SHAPE_RECORD=1; shift ;;
    --trace-ubatch) TRACE_UBATCH=1; shift ;;
    --merge-prefill) MERGE_PREFILL=1; shift ;;
    --no-merge-prefill) MERGE_PREFILL=0; shift ;;
    --npu-text-prefill-dynamic) NPU_TEXT_PREFILL_DYNAMIC=1; shift ;;
    --no-npu-text-prefill-dynamic) NPU_TEXT_PREFILL_DYNAMIC=0; shift ;;
    --acc-sample-mode) ACC_SAMPLE_MODE="$2"; shift 2 ;;
    --force-sync-shared) FORCE_SYNC_SHARED=1; shift ;;
    --power-path) POWER_PATH="$2"; shift 2 ;;
    --sample-hz) SAMPLE_HZ="$2"; shift 2 ;;
    --acc-ori) ACC_ORI="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

[ -n "$OVERLAY_APP" ] || {
  echo "--overlay-app is required (for example: --overlay-app GEMM_v6_app)" >&2
  exit 1
}

case "$ACC_SAMPLE_MODE" in
  available|official) ;;
  *)
    echo "--acc-sample-mode must be one of: available, official" >&2
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

TARGET="$USER_NAME@$HOST"
MODEL_PATH="$(realpath "$MODEL_PATH")"
MMPROJ_PATH="$(realpath "$MMPROJ_PATH")"
THROUGHPUT_PROFILE_PROMPT_B64="$(printf '%s' "$THROUGHPUT_PROFILE_PROMPT" | base64 -w0)"
if [ -n "$NPU_SHAPE_TABLE" ]; then
  NPU_SHAPE_TABLE="$(realpath "$NPU_SHAPE_TABLE")"
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
REMOTE_MMPROJ="$REMOTE_SHARED_DIR/$(basename "$MMPROJ_PATH")"
REMOTE_FULL_TEST_JSON="$REMOTE_DATA_DIR/FullTest.json"
REMOTE_IMAGE_ROOT="$REMOTE_DATA_DIR/images"
REMOTE_TTFT_CONFIG="$REMOTE_CODE_DIR/ttft_config.remote.json"
REMOTE_SAMPLE_JSON=""
REMOTE_THROUGHPUT_PROFILE_METRICS="$REMOTE_RESULTS_DIR/throughput_metrics_profile.json"
REMOTE_THROUGHPUT_PROFILE_ARTIFACTS="$REMOTE_RESULTS_DIR/throughput_profile_artifacts.json"
REMOTE_MTMD_SUMMARY="$REMOTE_RESULTS_DIR/mtmd_prefill_summary.json"
REMOTE_NPU_PROFILE_JSON="$REMOTE_RESULTS_DIR/ggml_npu_profile.json"
REMOTE_NPU_PROFILE_MANIFEST="$REMOTE_RESULTS_DIR/ggml_npu_profile_manifest.json"
REMOTE_PROFILE_SERVER_LOG="$REMOTE_RUN_DIR/server_profile.log"
REMOTE_NPU_SHAPE_TABLE=""
if [ -n "$NPU_SHAPE_TABLE" ]; then
  REMOTE_NPU_SHAPE_TABLE="$REMOTE_CODE_DIR/$(basename "$NPU_SHAPE_TABLE")"
fi
REMOTE_NPU_SHAPE_RECORD="$REMOTE_RESULTS_DIR/npu_shape_record.json"
REMOTE_UBATCH_TRACE="$REMOTE_RESULTS_DIR/ubatch_trace.jsonl"

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

  python3 - "$ttft_config" "$MODEL_ALIAS" <<'PY'
import json
import sys

output_path = sys.argv[1]
model_alias = sys.argv[2]

cfg = {
    "server_url": "http://127.0.0.1:8080/v1",
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

  copy_once_by_size "$SERVER_BIN" "$REMOTE_SERVER_BIN" "llama-server"
  copy_once_by_size "$MODEL_PATH" "$REMOTE_MODEL" "model"
  copy_once_by_size "$MMPROJ_PATH" "$REMOTE_MMPROJ" "mmproj"

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

  ssh "${SSH_OPTS[@]}" "$TARGET" "RUN_DIR='$REMOTE_RUN_DIR' REMOTE_LIB_DIR='$REMOTE_LIB_DIR' REMOTE_MODEL='$REMOTE_MODEL' REMOTE_MMPROJ='$REMOTE_MMPROJ' SUDO_PASSWORD='$SUDO_PASSWORD' OVERLAY_APP='$OVERLAY_APP' PORT='$PORT' THREADS='$THREADS' UBATCH_SIZE='$UBATCH_SIZE' MTMD_BACKEND_DEVICE='$MTMD_BACKEND_DEVICE' MODEL_ALIAS='$MODEL_ALIAS' POWER_PATH='$POWER_PATH' SAMPLE_HZ='$SAMPLE_HZ' RUN_THROUGHPUT_PROFILE='$RUN_THROUGHPUT_PROFILE' THROUGHPUT_PROFILE_ONLY='$THROUGHPUT_PROFILE_ONLY' THROUGHPUT_PROFILE_MAX_TOKENS='$THROUGHPUT_PROFILE_MAX_TOKENS' THROUGHPUT_PROFILE_PROMPT_B64='$THROUGHPUT_PROFILE_PROMPT_B64' NPU_PROFILE_LEVEL='$NPU_PROFILE_LEVEL' REMOTE_THROUGHPUT_PROFILE_METRICS='$REMOTE_THROUGHPUT_PROFILE_METRICS' REMOTE_THROUGHPUT_PROFILE_ARTIFACTS='$REMOTE_THROUGHPUT_PROFILE_ARTIFACTS' REMOTE_MTMD_SUMMARY='$REMOTE_MTMD_SUMMARY' REMOTE_NPU_PROFILE_JSON='$REMOTE_NPU_PROFILE_JSON' REMOTE_NPU_PROFILE_MANIFEST='$REMOTE_NPU_PROFILE_MANIFEST' REMOTE_PROFILE_SERVER_LOG='$REMOTE_PROFILE_SERVER_LOG' REMOTE_NPU_SHAPE_TABLE='$REMOTE_NPU_SHAPE_TABLE' NPU_SHAPE_RECORD='$NPU_SHAPE_RECORD' REMOTE_NPU_SHAPE_RECORD='$REMOTE_NPU_SHAPE_RECORD' TRACE_UBATCH='$TRACE_UBATCH' REMOTE_UBATCH_TRACE='$REMOTE_UBATCH_TRACE' MERGE_PREFILL='$MERGE_PREFILL' NPU_TEXT_PREFILL_DYNAMIC='$NPU_TEXT_PREFILL_DYNAMIC' bash -s" <<EOF
set -euo pipefail

cd "\$RUN_DIR"
rm -f server.log server.pid
mkdir -p results
THROUGHPUT_PROFILE_PROMPT="\$(printf '%s' "\$THROUGHPUT_PROFILE_PROMPT_B64" | base64 -d)"

cleanup() {
  if [ -f server.pid ]; then
    local pid
    pid="\$(cat server.pid)"
    if [ -n "\$pid" ]; then
      echo "\$SUDO_PASSWORD" | sudo -S kill "\$pid" >/dev/null 2>&1 || true
    fi
  fi
}
trap cleanup EXIT

echo "\$SUDO_PASSWORD" | sudo -S xmutil unloadapp >/dev/null 2>&1 || true
echo "\$SUDO_PASSWORD" | sudo -S xmutil loadapp "\$OVERLAY_APP"

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
      wait "\$pid" >/dev/null 2>&1 || true
    fi
    rm -f server.pid
  fi
}

start_server() {
  local log_path="\$1"
  local enable_profile="\$2"
  stop_server
  echo "\$SUDO_PASSWORD" | sudo -S pkill -f "./llama-server --host 127.0.0.1 --port \$PORT" >/dev/null 2>&1 || true

  local profile_env=""
  local npu_shape_env=""
  local ubatch_trace_env=""
  local text_prefill_env="LLAMA_MTMD_MERGE_PREFILL='\$MERGE_PREFILL' GGML_NPU_TEXT_PREFILL_DYNAMIC='\$NPU_TEXT_PREFILL_DYNAMIC'"
  if [ -n "\$REMOTE_NPU_SHAPE_TABLE" ]; then
    npu_shape_env="GGML_NPU_PRELOAD_WEIGHTS_ON_LOAD=1 GGML_NPU_SHAPE_TABLE_JSON='\$REMOTE_NPU_SHAPE_TABLE'"
  fi
  if [ "\$NPU_SHAPE_RECORD" = "1" ]; then
    rm -f "\$REMOTE_NPU_SHAPE_RECORD"
    npu_shape_env="\$npu_shape_env GGML_NPU_SHAPE_RECORD_JSON='\$REMOTE_NPU_SHAPE_RECORD'"
  fi
  if [ "\$enable_profile" = "1" ]; then
    rm -f "\$REMOTE_THROUGHPUT_PROFILE_METRICS" "\$REMOTE_THROUGHPUT_PROFILE_ARTIFACTS" "\$REMOTE_MTMD_SUMMARY" "\$REMOTE_NPU_PROFILE_JSON" "\$REMOTE_NPU_PROFILE_MANIFEST"
    profile_env="LLAMA_MTMD_PREFILL_SUMMARY_JSON='\$REMOTE_MTMD_SUMMARY' GGML_NPU_PROFILE_JSON='\$REMOTE_NPU_PROFILE_JSON' GGML_NPU_PROFILE_MANIFEST_JSON='\$REMOTE_NPU_PROFILE_MANIFEST' GGML_NPU_PROFILE_LEVEL='\$NPU_PROFILE_LEVEL'"
  fi
  if [ "\$TRACE_UBATCH" = "1" ]; then
    rm -f "\$REMOTE_UBATCH_TRACE"
    ubatch_trace_env="LLAMA_UBATCH_TRACE_JSONL='\$REMOTE_UBATCH_TRACE'"
  fi

  echo "\$SUDO_PASSWORD" | sudo -S bash -lc "cd '\$RUN_DIR' && env LD_LIBRARY_PATH='\$REMOTE_LIB_DIR:\${LD_LIBRARY_PATH:-}' MTMD_BACKEND_DEVICE='\$MTMD_BACKEND_DEVICE' \$text_prefill_env \$npu_shape_env \$profile_env \$ubatch_trace_env ./llama-server --host 127.0.0.1 --port '\$PORT' --alias '\$MODEL_ALIAS' -m '\$REMOTE_MODEL' --mmproj '\$REMOTE_MMPROJ' --cache-type-k q8_0 --cache-type-v q8_0 -t '\$THREADS' --ubatch-size '\$UBATCH_SIZE' --log-disable --no-warmup > '\$log_path' 2>&1 & echo \\\$! > server.pid"

  if ! wait_ready; then
    tail -n 120 "\$log_path" >&2 || true
    echo "llama-server did not become ready" >&2
    exit 1
  fi
}

if [ "\$THROUGHPUT_PROFILE_ONLY" -ne 1 ]; then
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

  python3 "\$RUN_DIR/code/throughput_eval.py" \
    -i "\$RUN_DIR/code/test2.jpg" \
    -o "\$REMOTE_THROUGHPUT_PROFILE_METRICS" \
    --base-url "http://127.0.0.1:\$PORT/v1" \
    --model "\$MODEL_ALIAS" \
    --max-tokens "\$THROUGHPUT_PROFILE_MAX_TOKENS" \
    --prompt "\$THROUGHPUT_PROFILE_PROMPT"

  python3 - "\$REMOTE_THROUGHPUT_PROFILE_ARTIFACTS" "\$REMOTE_THROUGHPUT_PROFILE_METRICS" "\$REMOTE_MTMD_SUMMARY" "\$REMOTE_NPU_PROFILE_JSON" "\$REMOTE_NPU_PROFILE_MANIFEST" "\$NPU_PROFILE_LEVEL" <<'PY'
import json
import os
import sys

out_path, metrics_path, mtmd_path, npu_path, manifest_path, npu_level = sys.argv[1:7]

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
    "ggml_npu_profile": info(npu_path),
    "ggml_npu_manifest": info(manifest_path),
    "npu_profile_level": npu_level,
}

with open(out_path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=False, indent=2)
PY

  stop_server
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
