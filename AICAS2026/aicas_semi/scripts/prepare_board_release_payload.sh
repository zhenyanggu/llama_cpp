#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  prepare_board_release_payload.sh [options]

Build a self-contained KV260 board payload for the AICAS semi-final image.

Options:
  --output-dir <path>          Payload directory (default: /tmp/aicas-semi-board-release)
  --server-bin <path>          Prebuilt KV260 llama-server
  --best-config-json <path>    Current-best config JSON
  --model <path>               Text model GGUF
  --mmproj <path>              mmproj GGUF
  --cpu-model <path>           CPU baseline fp16 text model GGUF
  --cpu-mmproj <path>          CPU baseline fp16 mmproj GGUF
  --data-root <path>           Full OCRBench image root
  --full-test-json <path>      FullTest.json
  --official-semi-dir <path>   Official aicas_semi package
  --prefill-overlay-dir <path> Final prefill overlay app directory
  --decode-overlay-dir <path>  Final decode overlay app directory
  --driver-ko <path>           npu_kv260.ko
  --runtime-lib-dir <path>     Runtime library directory
  --no-data                    Do not copy OCRBench images
  --clean                      Remove output directory before writing
  -h, --help                   Show this help
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

OUTPUT_DIR="/tmp/aicas-semi-board-release"
SERVER_BIN="$REPO_DIR/build-kv260-semi/bin/llama-server"
BEST_CONFIG_JSON="$REPO_DIR/docs/aicas-current-best-config.json"
MODEL_PATH=""
MMPROJ_PATH=""
CPU_MODEL_PATH="$REPO_DIR/AICAS/gguf/SmolVLM2-500M-Video-Instruct-f16.gguf"
CPU_MMPROJ_PATH="$REPO_DIR/AICAS/gguf/mmproj-SmolVLM2-500M-Video-Instruct-f16.gguf"
DATA_ROOT="$REPO_DIR/AICAS/data"
FULL_TEST_JSON="$REPO_DIR/AICAS/FullTest.json"
OFFICIAL_SEMI_DIR="/mnt/c/Users/顾振阳/Downloads/aicas_semi"
PREFILL_OVERLAY_DIR="/mnt/c/vivado/KV260/out/prefill_190m_qkpipe_20260608_0206_app"
DECODE_OVERLAY_DIR="/mnt/c/vivado/KV260/out/dec_200m_latest_0609a_app"
DRIVER_KO="$REPO_DIR/npuruntime/kv260/driver/npu_kv260.ko"
RUNTIME_LIB_DIR="/home/gugugu/.codex/skills/kv260-llamaserver-throughput/assets/runtime_libs"
COPY_DATA=1
CLEAN=0

while [ $# -gt 0 ]; do
  case "$1" in
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --server-bin) SERVER_BIN="$2"; shift 2 ;;
    --best-config-json) BEST_CONFIG_JSON="$2"; shift 2 ;;
    --model) MODEL_PATH="$2"; shift 2 ;;
    --mmproj) MMPROJ_PATH="$2"; shift 2 ;;
    --cpu-model) CPU_MODEL_PATH="$2"; shift 2 ;;
    --cpu-mmproj) CPU_MMPROJ_PATH="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --full-test-json) FULL_TEST_JSON="$2"; shift 2 ;;
    --official-semi-dir) OFFICIAL_SEMI_DIR="$2"; shift 2 ;;
    --prefill-overlay-dir) PREFILL_OVERLAY_DIR="$2"; shift 2 ;;
    --decode-overlay-dir) DECODE_OVERLAY_DIR="$2"; shift 2 ;;
    --driver-ko) DRIVER_KO="$2"; shift 2 ;;
    --runtime-lib-dir) RUNTIME_LIB_DIR="$2"; shift 2 ;;
    --no-data) COPY_DATA=0; shift ;;
    --clean) CLEAN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

require_path() {
  local path="$1"
  local label="$2"
  [ -e "$path" ] || { echo "Missing $label: $path" >&2; exit 1; }
}

realpath_existing() {
  local path="$1"
  python3 - "$path" <<'PY'
import os
import sys
print(os.path.realpath(sys.argv[1]))
PY
}

resolve_best_config_paths() {
  eval "$(
    python3 - "$BEST_CONFIG_JSON" "$REPO_DIR" "$MODEL_PATH" "$MMPROJ_PATH" <<'PY'
import json
import os
import shlex
import sys

cfg_path, repo_dir, model_override, mmproj_override = sys.argv[1:5]
cfg_path = os.path.realpath(cfg_path)
repo_dir = os.path.realpath(repo_dir)
with open(cfg_path, "r", encoding="utf-8") as handle:
    cfg = json.load(handle)

def resolve(path: str) -> str:
    if not path:
        return ""
    if os.path.isabs(path):
        return path
    cfg_dir = os.path.dirname(cfg_path)
    cfg_root = os.path.dirname(cfg_dir) if os.path.basename(cfg_dir) == "docs" else cfg_dir
    candidates = [
        os.path.join(repo_dir, path),
        os.path.join(cfg_root, path),
        os.path.join(os.path.dirname(repo_dir), "model-quant", path),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]

model = model_override or resolve(str(cfg.get("text_model", "")))
mmproj = mmproj_override or resolve(str(cfg.get("mmproj", "")))
text_calib = ""
mmproj_calib = ""
for key, value in (cfg.get("env") or {}).items():
    if key == "AICAS_TEXT_PREFILL_LOG8PV_CALIB_FILE":
        text_calib = resolve(str(value))
    elif key == "AICAS_MMPROJ_LOG8PV_CALIB_FILE":
        mmproj_calib = resolve(str(value))

print(f"MODEL_PATH={shlex.quote(model)}")
print(f"MMPROJ_PATH={shlex.quote(mmproj)}")
print(f"TEXT_CALIB_PATH={shlex.quote(text_calib)}")
print(f"MMPROJ_CALIB_PATH={shlex.quote(mmproj_calib)}")
PY
  )"
}

copy_file() {
  local src="$1"
  local dst="$2"
  mkdir -p "$(dirname "$dst")"
  cp -a "$src" "$dst"
}

copy_overlay_app() {
  local src_dir="$1"
  local dst_dir="$2"
  local label="$3"
  rm -rf "$dst_dir"
  mkdir -p "$dst_dir"
  local copied=0
  for pattern in '*.bit.bin' '*.dtbo' 'pl.dtsi' 'shell.json'; do
    for file in "$src_dir"/$pattern; do
      [ -e "$file" ] || continue
      cp -a "$file" "$dst_dir/"
      copied=$((copied + 1))
    done
  done
  [ "$copied" -gt 0 ] || { echo "No overlay files copied for $label from $src_dir" >&2; exit 1; }
}

write_board_ttft_config() {
  python3 - "$OUTPUT_DIR/code/ttft_config.json" "$OUTPUT_DIR/code/ttft_config.board.json" <<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
cfg = json.loads(src.read_text(encoding="utf-8"))
cfg["server_url"] = "http://127.0.0.1:8080/v1"
cfg["model"] = "smolvlm2-gguf"
for image in cfg.get("images", []):
    if image.get("name") == "small":
        image.pop("resize", None)
        image["source"] = "small.jpg"
cfg["output"] = "ttft_eval_results.json"
dst.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
}

prepare_small_image() {
  local src="$OUTPUT_DIR/code/test2.jpg"
  local dst="$OUTPUT_DIR/code/small.jpg"
  if python3 - "$src" "$dst" <<'PY'
import sys
from PIL import Image
src, dst = sys.argv[1:3]
img = Image.open(src)
img = img.resize((512, 512), Image.LANCZOS)
img.save(dst, "JPEG")
PY
  then
    return 0
  fi
  if [ -f "$REPO_DIR/AICAS2026/aicas_semi/results/current/small.jpg" ]; then
    cp -a "$REPO_DIR/AICAS2026/aicas_semi/results/current/small.jpg" "$dst"
    return 0
  fi
  echo "Failed to create small.jpg: install Pillow on host or provide results/current/small.jpg" >&2
  exit 1
}

write_env_sh() {
  python3 - "$BEST_CONFIG_JSON" "$OUTPUT_DIR/config/env.sh" <<'PY'
import json
import shlex
import sys
from pathlib import Path

cfg_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
env = cfg.get("env") or {}
lines = [
    "# Generated by prepare_board_release_payload.sh",
    ': "${RELEASE_ROOT:?RELEASE_ROOT must be set before sourcing env.sh}"',
    "export VERSA_P_CMA_SIZE=1408M",
    "export VERSA_P_CMA_HEAP_OFFSET=0",
    "export VERSA_P_CMA_HEAP_SIZE=832M",
    "export NPU_CMA_SIZE=1408M",
    "export NPU_CMA_HEAP_OFFSET=832M",
    "export NPU_CMA_HEAP_SIZE=576M",
    "export AICAS_TEXT_PREFILL_LOG8PV_NPU=1",
    "export AICAS_TEXT_PREFILL_LOG8PV_DIRECT_KV=1",
    "export AICAS_MMPROJ_LOG8PV_NPU=1",
]
for key, value in env.items():
    raw_line = None
    if key == "AICAS_TEXT_PREFILL_LOG8PV_CALIB_FILE":
        raw_line = 'export AICAS_TEXT_PREFILL_LOG8PV_CALIB_FILE="${RELEASE_ROOT}/config/weighted40-text-prefill-log8pv-v1.json"'
    elif key == "AICAS_MMPROJ_LOG8PV_CALIB_FILE":
        raw_line = 'export AICAS_MMPROJ_LOG8PV_CALIB_FILE="${RELEASE_ROOT}/config/weighted40-mmproj-log8pv-v2.json"'
    if not key.replace("_", "").isalnum():
        raise SystemExit(f"invalid env key: {key}")
    if raw_line is not None:
        lines.append(raw_line)
    else:
        lines.append(f"export {key}={shlex.quote(str(value))}")
out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

write_release_meta() {
  python3 - "$OUTPUT_DIR/release_meta.json" "$REPO_DIR" "$SERVER_BIN" "$MODEL_PATH" "$MMPROJ_PATH" "$DATA_ROOT" <<'PY'
import json
import os
import subprocess
import sys
from pathlib import Path

out, repo, server, model, mmproj, data_root = sys.argv[1:7]
def cmd(args):
    try:
        return subprocess.check_output(args, text=True).strip()
    except Exception:
        return ""
payload = {
    "release_kind": "aicas-semi-kv260-board-payload",
    "git_head": cmd(["git", "-C", repo, "rev-parse", "HEAD"]),
    "git_status_short": cmd(["git", "-C", repo, "status", "--short"]).splitlines(),
    "server_bin": os.path.realpath(server),
    "model": os.path.realpath(model),
    "mmproj": os.path.realpath(mmproj),
    "data_root": os.path.realpath(data_root),
}
Path(out).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
}

write_manifests() {
  (
    cd "$OUTPUT_DIR"
    python3 - <<'PY'
import json
from pathlib import Path
records = []
for path in sorted(p for p in Path(".").rglob("*") if p.is_file()):
    records.append({"path": path.as_posix(), "size_bytes": path.stat().st_size})
Path("manifest.json").write_text(json.dumps({"file_count": len(records), "files": records}, indent=2) + "\n", encoding="utf-8")
PY
  )
}

resolve_best_config_paths

SERVER_BIN="$(realpath_existing "$SERVER_BIN")"
BEST_CONFIG_JSON="$(realpath_existing "$BEST_CONFIG_JSON")"
MODEL_PATH="$(realpath_existing "$MODEL_PATH")"
MMPROJ_PATH="$(realpath_existing "$MMPROJ_PATH")"
CPU_MODEL_PATH="$(realpath_existing "$CPU_MODEL_PATH")"
CPU_MMPROJ_PATH="$(realpath_existing "$CPU_MMPROJ_PATH")"
DATA_ROOT="$(realpath_existing "$DATA_ROOT")"
FULL_TEST_JSON="$(realpath_existing "$FULL_TEST_JSON")"
OFFICIAL_SEMI_DIR="$(realpath_existing "$OFFICIAL_SEMI_DIR")"
PREFILL_OVERLAY_DIR="$(realpath_existing "$PREFILL_OVERLAY_DIR")"
DECODE_OVERLAY_DIR="$(realpath_existing "$DECODE_OVERLAY_DIR")"
DRIVER_KO="$(realpath_existing "$DRIVER_KO")"
RUNTIME_LIB_DIR="$(realpath_existing "$RUNTIME_LIB_DIR")"
TEXT_CALIB_PATH="$(realpath_existing "$TEXT_CALIB_PATH")"
MMPROJ_CALIB_PATH="$(realpath_existing "$MMPROJ_CALIB_PATH")"

require_path "$SERVER_BIN" "llama-server"
require_path "$BEST_CONFIG_JSON" "best config"
require_path "$MODEL_PATH" "text model"
require_path "$MMPROJ_PATH" "mmproj"
require_path "$CPU_MODEL_PATH" "CPU baseline fp16 text model"
require_path "$CPU_MMPROJ_PATH" "CPU baseline fp16 mmproj"
require_path "$FULL_TEST_JSON" "FullTest.json"
require_path "$OFFICIAL_SEMI_DIR" "official semi package"
require_path "$PREFILL_OVERLAY_DIR" "prefill overlay"
require_path "$DECODE_OVERLAY_DIR" "decode overlay"
require_path "$DRIVER_KO" "driver"
require_path "$RUNTIME_LIB_DIR/libstdc++.so.6" "libstdc++.so.6"
require_path "$RUNTIME_LIB_DIR/libgcc_s.so.1" "libgcc_s.so.1"
require_path "$RUNTIME_LIB_DIR/libgomp.so.1" "libgomp.so.1"
require_path "$TEXT_CALIB_PATH" "text log8PV calibration"
require_path "$MMPROJ_CALIB_PATH" "mmproj log8PV calibration"
if [ "$COPY_DATA" -eq 1 ]; then
  require_path "$DATA_ROOT" "OCRBench data root"
fi

file_desc="$(file "$SERVER_BIN")"
[[ "$file_desc" == *"ARM aarch64"* ]] || {
  echo "Expected ARM aarch64 llama-server, got: $file_desc" >&2
  exit 1
}

if [ "$CLEAN" -eq 1 ]; then
  rm -rf "$OUTPUT_DIR"
fi
mkdir -p "$OUTPUT_DIR"/{bin,lib,models,data,code,config,board/driver,board/overlays,official_reference,results,logs}

copy_file "$SERVER_BIN" "$OUTPUT_DIR/bin/llama-server"
copy_file "$MODEL_PATH" "$OUTPUT_DIR/models/text.gguf"
copy_file "$MMPROJ_PATH" "$OUTPUT_DIR/models/mmproj.gguf"
copy_file "$CPU_MODEL_PATH" "$OUTPUT_DIR/models/text-f16.gguf"
copy_file "$CPU_MMPROJ_PATH" "$OUTPUT_DIR/models/mmproj-f16.gguf"
copy_file "$FULL_TEST_JSON" "$OUTPUT_DIR/data/FullTest.json"
copy_file "$DRIVER_KO" "$OUTPUT_DIR/board/driver/npu_kv260.ko"
copy_file "$TEXT_CALIB_PATH" "$OUTPUT_DIR/config/weighted40-text-prefill-log8pv-v1.json"
copy_file "$MMPROJ_CALIB_PATH" "$OUTPUT_DIR/config/weighted40-mmproj-log8pv-v2.json"

cp -a "$RUNTIME_LIB_DIR"/libstdc++.so* "$OUTPUT_DIR/lib/"
cp -a "$RUNTIME_LIB_DIR"/libgcc_s.so* "$OUTPUT_DIR/lib/"
cp -a "$RUNTIME_LIB_DIR"/libgomp.so* "$OUTPUT_DIR/lib/"

copy_overlay_app "$PREFILL_OVERLAY_DIR" "$OUTPUT_DIR/board/overlays/prefill_190m_qkpipe_20260608_0206_app" "prefill"
copy_overlay_app "$DECODE_OVERLAY_DIR" "$OUTPUT_DIR/board/overlays/dec_200m_latest_0609a_app" "decode"

cp -a "$REPO_DIR/AICAS2026/aicas_semi/code/"*.py "$OUTPUT_DIR/code/"
cp -a "$REPO_DIR/AICAS2026/aicas_semi/code/test2.jpg" "$OUTPUT_DIR/code/"
cp -a "$REPO_DIR/AICAS2026/aicas_semi/code/ttft_config.json" "$OUTPUT_DIR/code/"
cp -a "$REPO_DIR/AICAS2026/aicas_semi/scripts/run_semi_eval.sh" "$OUTPUT_DIR/code/run_semi_eval.sh"
cp -a "$REPO_DIR/AICAS2026/aicas_semi/scripts/board_run_semi.sh" "$OUTPUT_DIR/run_semi.sh"
chmod +x "$OUTPUT_DIR/run_semi.sh" "$OUTPUT_DIR/code/run_semi_eval.sh"

rm -rf "$OUTPUT_DIR/official_reference"
mkdir -p "$OUTPUT_DIR/official_reference"
cp -a "$OFFICIAL_SEMI_DIR/." "$OUTPUT_DIR/official_reference/"

prepare_small_image
write_board_ttft_config
write_env_sh

if [ "$COPY_DATA" -eq 1 ]; then
  rm -rf "$OUTPUT_DIR/data/images"
  mkdir -p "$OUTPUT_DIR/data/images"
  cp -a "$DATA_ROOT/." "$OUTPUT_DIR/data/images/"
fi

write_release_meta
write_manifests

echo "Payload ready: $OUTPUT_DIR"
du -sh "$OUTPUT_DIR"
