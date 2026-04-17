#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  make_one_layer_mmproj_gguf.sh [options]

Build two one-layer W8A8 mmproj GGUFs for fast NPU bring-up:
  1) tiny profile:      v.blk.0.attn_q.weight (768x768)
  2) real profile:      v.blk.0.ffn_down.weight (3072x768)

Options:
  --input-gguf <path>       Source mmproj f16 gguf
  --layer-manifest <path>   layer_manifest.json
  --base-policy <path>      Base quant/policy json (contains act params)
  --out-dir <path>          Output dir for generated policies and pack summaries
  --gguf-dir <path>         Output dir for generated gguf files
  -h, --help                Show help
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
INPUT_GGUF="$ROOT_DIR/AICAS/gguf/mmproj-SmolVLM2-500M-Video-Instruct-f16.gguf"
LAYER_MANIFEST="$ROOT_DIR/AICAS/artifacts/layer_manifest.json"
BASE_POLICY="$ROOT_DIR/AICAS/artifacts/quant_params.strict_pt.json"
OUT_DIR="$ROOT_DIR/AICAS/artifacts/one_layer"
GGUF_DIR="$ROOT_DIR/AICAS/gguf"

while [ $# -gt 0 ]; do
  case "$1" in
    --input-gguf) INPUT_GGUF="$2"; shift 2 ;;
    --layer-manifest) LAYER_MANIFEST="$2"; shift 2 ;;
    --base-policy) BASE_POLICY="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --gguf-dir) GGUF_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage >&2; exit 1 ;;
  esac
done

mkdir -p "$OUT_DIR" "$GGUF_DIR"

require_file() {
  local path="$1"
  local name="$2"
  if [ ! -f "$path" ]; then
    echo "Missing $name: $path" >&2
    exit 1
  fi
}

require_file "$INPUT_GGUF" "input gguf"
require_file "$LAYER_MANIFEST" "layer manifest"
require_file "$BASE_POLICY" "base policy"

make_single() {
  local target_tensor="$1"
  local tag="$2"
  local policy_out="$OUT_DIR/policy_one_layer_${tag}.json"
  local pack_summary="$OUT_DIR/pack_summary_one_layer_${tag}.json"
  local gguf_out="$GGUF_DIR/mmproj-one-layer-${tag}-per-tensor.gguf"

  python3 "$ROOT_DIR/AICAS/tools/mmproj_make_single_layer_policy.py" \
    --input-policy "$BASE_POLICY" \
    --target-tensor "$target_tensor" \
    --output "$policy_out"

  python3 "$ROOT_DIR/AICAS/tools/mmproj_pack_gguf.py" \
    --input-gguf "$INPUT_GGUF" \
    --layer-manifest "$LAYER_MANIFEST" \
    --quant-params "$policy_out" \
    --output-gguf "$gguf_out" \
    --output-quant-params "$pack_summary" \
    --mode quantize \
    --weight-granularity per_tensor

  echo "Generated: $gguf_out"
}

make_single "v.blk.0.attn_q.weight" "tiny_attn_q"
make_single "v.blk.0.ffn_down.weight" "real_ffn_down"

echo "Done. one-layer ggufs are in: $GGUF_DIR"
