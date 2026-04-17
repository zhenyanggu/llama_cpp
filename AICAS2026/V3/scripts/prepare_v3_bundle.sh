#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  prepare_v1_bundle.sh [options]

Prepare the self-contained V1 submission bundle under AICAS2026/V1.

Options:
  --sdk-env <path>      KV260 SDK env script.
  --build-dir <path>    Cross-build directory for llama-server.
  --skip-build          Reuse the existing build output.
  -h, --help            Show this help.
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
V1_DIR="$ROOT_DIR/AICAS2026/V1"
PAYLOAD_DIR="$V1_DIR/payload"
MANIFEST_DIR="$V1_DIR/manifests"
SOURCE_STATE_DIR="$V1_DIR/source_state"
RESULTS_DIR="$V1_DIR/results/official"

SDK_ENV="/home/gugugu/petalinux/sdk/kv260-2025.1/environment-setup-cortexa72-cortexa53-amd-linux"
BUILD_DIR="$ROOT_DIR/build-kv260-npu"
SKIP_BUILD=0

MODEL_SRC="$ROOT_DIR/AICAS/gguf/SmolVLM2-500M-Video-Instruct-Q8_0.gguf"
MMPROJ_SRC="$ROOT_DIR/AICAS/gguf/mmproj-SmolVLM2-500M-Video-Instruct-w8a8-mixed-v1.gguf"
THROUGHPUT_SRC="$ROOT_DIR/AICAS/throughput_eval.py"
ACC_SRC="$ROOT_DIR/AICAS/acc_eval.py"
CLIENT_SRC="$ROOT_DIR/AICAS/llama_server_client.py"
SAMPLED_JSON_SRC="$ROOT_DIR/AICAS/sampled.json"
DATA_ROOT="$ROOT_DIR/AICAS/data"
DRIVER_SRC="$ROOT_DIR/npuruntime/kv260/driver/npu_kv260.ko"
MYNPU_SRC_DIR="$ROOT_DIR/AICAS2026/V1_412/mynpu"
SERVER_SRC="$BUILD_DIR/bin/llama-server"

BUILD_FLAGS=(
  -DCMAKE_BUILD_TYPE=Release
  -DBUILD_SHARED_LIBS=OFF
  -DLLAMA_CURL=OFF
  -DGGML_BLAS=OFF
  -DLLAMA_BUILD_TESTS=OFF
  -DLLAMA_BUILD_EXAMPLES=OFF
  -DLLAMA_BUILD_TOOLS=ON
  -DLLAMA_BUILD_SERVER=ON
  -DGGML_NPU=ON
)

while [ $# -gt 0 ]; do
  case "$1" in
    --sdk-env)
      SDK_ENV="$2"
      shift 2
      ;;
    --build-dir)
      BUILD_DIR="$2"
      SERVER_SRC="$BUILD_DIR/bin/llama-server"
      shift 2
      ;;
    --skip-build)
      SKIP_BUILD=1
      shift
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

require_path() {
  local path="$1"
  local label="$2"
  [ -e "$path" ] || {
    echo "Missing $label: $path" >&2
    exit 1
  }
}

build_server() {
  if [ "$SKIP_BUILD" -eq 1 ]; then
    require_path "$SERVER_SRC" "llama-server binary"
    return 0
  fi

  require_path "$SDK_ENV" "KV260 SDK env script"

  (
    set -euo pipefail
    unset LD_LIBRARY_PATH
    # shellcheck disable=SC1090
    source "$SDK_ENV" >/dev/null
    cmake -S "$ROOT_DIR" -B "$BUILD_DIR" -G Ninja "${BUILD_FLAGS[@]}"
    cmake --build "$BUILD_DIR" --target llama-server -j"$(nproc)"
  )
}

verify_server_bin() {
  local file_desc
  require_path "$SERVER_SRC" "llama-server binary"
  file_desc="$(file "$SERVER_SRC")"
  [[ "$file_desc" == *"ARM aarch64"* ]] || {
    echo "Expected an ARM aarch64 llama-server, got: $file_desc" >&2
    exit 1
  }
}

prepare_dirs() {
  rm -rf \
    "$PAYLOAD_DIR/bin" \
    "$PAYLOAD_DIR/board_support" \
    "$PAYLOAD_DIR/data" \
    "$PAYLOAD_DIR/eval" \
    "$PAYLOAD_DIR/gguf" \
    "$PAYLOAD_DIR/runtime_libs" \
    "$SOURCE_STATE_DIR" \
    "$MANIFEST_DIR"

  mkdir -p \
    "$PAYLOAD_DIR/bin" \
    "$PAYLOAD_DIR/board_support/driver" \
    "$PAYLOAD_DIR/board_support/mynpu" \
    "$PAYLOAD_DIR/data/images" \
    "$PAYLOAD_DIR/eval" \
    "$PAYLOAD_DIR/gguf" \
    "$PAYLOAD_DIR/runtime_libs" \
    "$SOURCE_STATE_DIR" \
    "$MANIFEST_DIR" \
    "$RESULTS_DIR"
}

copy_runtime_libs() {
  local sdk_root sdk_lib_dir fallback_lib_dir lib
  sdk_root="$(cd "$(dirname "$SDK_ENV")" && pwd)"
  sdk_lib_dir="$sdk_root/sysroots/cortexa72-cortexa53-amd-linux/usr/lib"
  fallback_lib_dir="$ROOT_DIR/AICAS/third_party/kv260_runtime/lib"

  require_path "$sdk_lib_dir" "SDK sysroot library directory"

  for lib in \
    libgomp.so libgomp.so.1 libgomp.so.1.0.0 \
    libstdc++.so libstdc++.so.6 libstdc++.so.6.0.32 \
    libgcc_s.so libgcc_s.so.1; do
    if [ -e "$fallback_lib_dir/$lib" ]; then
      cp -a "$fallback_lib_dir/$lib" "$PAYLOAD_DIR/runtime_libs/"
    elif [ -e "$sdk_lib_dir/$lib" ]; then
      cp -a "$sdk_lib_dir/$lib" "$PAYLOAD_DIR/runtime_libs/"
    fi
  done

  require_path "$PAYLOAD_DIR/runtime_libs/libgomp.so.1" "libgomp"
  require_path "$PAYLOAD_DIR/runtime_libs/libstdc++.so.6" "libstdc++"
  require_path "$PAYLOAD_DIR/runtime_libs/libgcc_s.so.1" "libgcc_s"
}

copy_fixed_payload() {
  cp -a "$SERVER_SRC" "$PAYLOAD_DIR/bin/llama-server-kv260-npu"
  cp -a "$MODEL_SRC" "$PAYLOAD_DIR/gguf/"
  cp -a "$MMPROJ_SRC" "$PAYLOAD_DIR/gguf/"
  cp -a "$THROUGHPUT_SRC" "$PAYLOAD_DIR/eval/"
  cp -a "$ACC_SRC" "$PAYLOAD_DIR/eval/"
  cp -a "$CLIENT_SRC" "$PAYLOAD_DIR/eval/"
  cp -a "$DRIVER_SRC" "$PAYLOAD_DIR/board_support/driver/"
  cp -a "$MYNPU_SRC_DIR"/KV260_410.bin "$PAYLOAD_DIR/board_support/mynpu/"
  cp -a "$MYNPU_SRC_DIR"/KV260_410.dtbo "$PAYLOAD_DIR/board_support/mynpu/"
  cp -a "$MYNPU_SRC_DIR"/shell.json "$PAYLOAD_DIR/board_support/mynpu/"
}

prepare_data_payload() {
  python3 - "$SAMPLED_JSON_SRC" "$DATA_ROOT" "$PAYLOAD_DIR/data/sampled_100.json" "$PAYLOAD_DIR/data/images" "$MANIFEST_DIR/required_data_files.txt" <<'PY'
import json
import shutil
import sys
from pathlib import Path

sampled_json = Path(sys.argv[1]).resolve()
data_root = Path(sys.argv[2]).resolve()
output_json = Path(sys.argv[3]).resolve()
images_root = Path(sys.argv[4]).resolve()
manifest_path = Path(sys.argv[5]).resolve()

items = json.loads(sampled_json.read_text(encoding="utf-8"))
paths = []
seen = set()

for item in items:
    rel = item.get("image_path")
    if not rel or rel in seen:
        continue
    src = (data_root / rel).resolve()
    if not src.exists():
        raise SystemExit(f"Missing sampled image: {src}")
    if data_root not in src.parents and src != data_root:
        raise SystemExit(f"Image escapes data root: {src}")
    seen.add(rel)
    paths.append(rel)
    dst = images_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

output_json.write_text(json.dumps(items, indent=2) + "\n", encoding="utf-8")
manifest_path.write_text("".join(f"{rel}\n" for rel in sorted(paths)), encoding="utf-8")
PY
}

snapshot_source_state() {
  git -C "$ROOT_DIR" rev-parse HEAD > "$SOURCE_STATE_DIR/git_head.txt"
  git -C "$ROOT_DIR" status --short > "$SOURCE_STATE_DIR/git_status.txt"
  git -C "$ROOT_DIR" diff --binary HEAD > "$SOURCE_STATE_DIR/working_tree.patch"
  git -C "$ROOT_DIR" ls-files --others --exclude-standard > "$SOURCE_STATE_DIR/untracked_files.txt"
}

write_build_meta() {
  BUILD_META_SERVER_SRC="$SERVER_SRC" \
  BUILD_META_SERVER_DST="$PAYLOAD_DIR/bin/llama-server-kv260-npu" \
  BUILD_META_BUILD_DIR="$BUILD_DIR" \
  BUILD_META_SDK_ENV="$SDK_ENV" \
  python3 - "$PAYLOAD_DIR/bin/llama-server-kv260-npu.build.json" <<'PY'
import json
import os
import sys
from pathlib import Path

payload = {
    "server_source": os.environ["BUILD_META_SERVER_SRC"],
    "server_bundle_path": os.environ["BUILD_META_SERVER_DST"],
    "build_dir": os.environ["BUILD_META_BUILD_DIR"],
    "sdk_env": os.environ["BUILD_META_SDK_ENV"],
    "build_flags": [
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_SHARED_LIBS=OFF",
        "-DLLAMA_CURL=OFF",
        "-DGGML_BLAS=OFF",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_EXAMPLES=OFF",
        "-DLLAMA_BUILD_TOOLS=ON",
        "-DLLAMA_BUILD_SERVER=ON",
        "-DGGML_NPU=ON",
    ],
}

Path(sys.argv[1]).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

write_manifests() {
  python3 - "$V1_DIR" "$MANIFEST_DIR/bundle_manifest.json" "$MANIFEST_DIR/SHA256SUMS" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

v1_dir = Path(sys.argv[1]).resolve()
manifest_path = Path(sys.argv[2]).resolve()
sha_path = Path(sys.argv[3]).resolve()

roots = [
    v1_dir / "README.md",
    v1_dir / "docs",
    v1_dir / "payload",
    v1_dir / "results" / "official",
    v1_dir / "scripts",
    v1_dir / "source_state",
]

records = []
sha_lines = []

for root in roots:
    if not root.exists():
      continue
    if root.is_file():
      paths = [root]
    else:
      paths = sorted(path for path in root.rglob("*") if path.is_file())
    for path in paths:
      rel = path.relative_to(v1_dir).as_posix()
      digest = hashlib.sha256(path.read_bytes()).hexdigest()
      records.append({
          "path": rel,
          "size_bytes": path.stat().st_size,
          "sha256": digest,
      })
      sha_lines.append(f"{digest}  {rel}")

payload = {
    "bundle_root": str(v1_dir),
    "file_count": len(records),
    "files": records,
}

manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
sha_path.write_text("\n".join(sha_lines) + ("\n" if sha_lines else ""), encoding="utf-8")
PY
}

require_path "$MODEL_SRC" "Q8_0 model"
require_path "$MMPROJ_SRC" "W8A8 mixed_v1 mmproj"
require_path "$THROUGHPUT_SRC" "throughput_eval.py"
require_path "$ACC_SRC" "acc_eval.py"
require_path "$CLIENT_SRC" "llama_server_client.py"
require_path "$SAMPLED_JSON_SRC" "sampled.json"
require_path "$DATA_ROOT" "AICAS data root"
require_path "$DRIVER_SRC" "KV260 NPU driver"
require_path "$MYNPU_SRC_DIR/KV260_410.bin" "mynpu bitstream"
require_path "$MYNPU_SRC_DIR/KV260_410.dtbo" "mynpu overlay"
require_path "$MYNPU_SRC_DIR/shell.json" "mynpu shell.json"

build_server
verify_server_bin
prepare_dirs
copy_runtime_libs
copy_fixed_payload
prepare_data_payload
snapshot_source_state
write_build_meta
write_manifests

echo "Prepared V1 bundle at $V1_DIR"
