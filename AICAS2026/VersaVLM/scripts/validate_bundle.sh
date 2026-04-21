#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  validate_bundle.sh [--update-manifests]

Validate that the VersaVLM bundle is self-contained.
Optionally refresh bundle manifests and SHA256 checksums.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PAYLOAD_DIR="$BUNDLE_DIR/payload"
MANIFEST_DIR="$BUNDLE_DIR/manifests"
RESULTS_DIR="$BUNDLE_DIR/results/official"
UPDATE_MANIFESTS=0

while [ $# -gt 0 ]; do
  case "$1" in
    --update-manifests) UPDATE_MANIFESTS=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

require_path() {
  local path="$1"
  local label="$2"
  [ -e "$path" ] || { echo "Missing $label: $path" >&2; exit 1; }
}

require_path "$BUNDLE_DIR/README.zh-CN.md" "Chinese README"
require_path "$BUNDLE_DIR/README.en.md" "English README"
require_path "$BUNDLE_DIR/docs/REPRODUCE.zh-CN.md" "Chinese reproduce doc"
require_path "$BUNDLE_DIR/docs/REPRODUCE.en.md" "English reproduce doc"
require_path "$PAYLOAD_DIR/bin/llama-server-kv260-npu" "KV260 llama-server"
require_path "$PAYLOAD_DIR/gguf/SmolVLM2-500M-Video-Instruct-Q8_0.gguf" "text model"
require_path "$PAYLOAD_DIR/gguf/mmproj-fallback-search-fb_attn_k-per-tensor.gguf" "mmproj"
require_path "$PAYLOAD_DIR/eval/throughput_eval.py" "throughput_eval.py"
require_path "$PAYLOAD_DIR/eval/acc_eval.py" "acc_eval.py"
require_path "$PAYLOAD_DIR/eval/llama_server_client.py" "llama_server_client.py"
require_path "$PAYLOAD_DIR/data/sampled_100.json" "sampled_100.json"
require_path "$PAYLOAD_DIR/data/images/IIIT5K/test/2543_2.png" "throughput sample image"
require_path "$PAYLOAD_DIR/board_support/overlay/double_dma_overlayapp/double_dma.bit.bin" "overlay bitstream"
require_path "$PAYLOAD_DIR/board_support/overlay/double_dma_overlayapp/double_dma.dtbo" "overlay dtbo"
require_path "$PAYLOAD_DIR/board_support/overlay/double_dma_overlayapp/shell.json" "overlay shell.json"
require_path "$PAYLOAD_DIR/board_support/driver/npu_kv260.ko" "NPU driver"
require_path "$PAYLOAD_DIR/board_support/qspi/BOOT-k26-smk-sdt-v1.05-20250912165210.bin" "QSPI reference image"
require_path "$PAYLOAD_DIR/board_support/tests/bin/kv260_mmproj_layer_asym_w8a8_test" "layer test binary"
require_path "$PAYLOAD_DIR/board_support/tests/bin/kv260_layer_gemm_replay_test" "replay test binary"
require_path "$PAYLOAD_DIR/runtime_libs/libgomp.so.1.0.0" "libgomp"
require_path "$PAYLOAD_DIR/runtime_libs/libstdc++.so.6.0.32" "libstdc++"
require_path "$PAYLOAD_DIR/runtime_libs/libgcc_s.so.1" "libgcc_s"

python3 - "$PAYLOAD_DIR/data/sampled_100.json" <<'PY'
import json
import sys
from pathlib import Path
items = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
if len(items) != 100:
    raise SystemExit(f"Expected 100 sampled items, got {len(items)}")
PY

if [ "$UPDATE_MANIFESTS" -eq 1 ]; then
  mkdir -p "$MANIFEST_DIR"
  python3 - "$BUNDLE_DIR" "$MANIFEST_DIR/bundle_manifest.json" "$MANIFEST_DIR/SHA256SUMS" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
bundle_dir = Path(sys.argv[1]).resolve()
manifest_path = Path(sys.argv[2]).resolve()
sha_path = Path(sys.argv[3]).resolve()
roots = [
    bundle_dir / 'README.zh-CN.md',
    bundle_dir / 'README.en.md',
    bundle_dir / 'docs',
    bundle_dir / 'manifests' / 'required_data_files.txt',
    bundle_dir / 'payload',
    bundle_dir / 'results' / 'official',
    bundle_dir / 'scripts',
    bundle_dir / 'source_state',
]
records = []
sha_lines = []
for root in roots:
    if not root.exists():
        continue
    paths = [root] if root.is_file() else sorted(path for path in root.rglob('*') if path.is_file())
    for path in paths:
        rel = path.relative_to(bundle_dir).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append({'path': rel, 'size_bytes': path.stat().st_size, 'sha256': digest})
        sha_lines.append(f'{digest}  {rel}')
manifest_path.write_text(json.dumps({'bundle_root': '.', 'file_count': len(records), 'files': records}, indent=2) + '\n', encoding='utf-8')
sha_path.write_text('\n'.join(sha_lines) + ('\n' if sha_lines else ''), encoding='utf-8')
PY
fi

echo "Bundle validation passed: $BUNDLE_DIR"
