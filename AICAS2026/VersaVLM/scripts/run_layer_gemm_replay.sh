#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_layer_gemm_replay.sh --profile-json <path> [options] [-- <extra replay args>]

Extract one layer tile shape from a ggml_npu_profile.json file, then run the
bundled kv260_layer_gemm_replay_test on KV260.

Options:
  --profile-json <path>      Profile json produced by a previous run
  --layer-id <int>           Layer id in profile (default: first semantic_op == ffn_down)
  --tile-index <int>         Tile index in selected layer (default: 0)
  --host <host>              KV260 host (default: 192.168.0.10)
  --user <user>              KV260 user (default: ubuntu)
  --remote-root <path>       Remote root (default: /home/ubuntu/aicas)
  -h, --help                 Show help
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOCAL_BIN="$BUNDLE_DIR/payload/board_support/tests/bin/kv260_layer_gemm_replay_test"
PROFILE_JSON=""
LAYER_ID=""
TILE_INDEX="0"
HOST="192.168.0.10"
USER_NAME="ubuntu"
REMOTE_ROOT="/home/ubuntu/aicas"
EXTRA_ARGS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --profile-json) PROFILE_JSON="$2"; shift 2 ;;
    --layer-id) LAYER_ID="$2"; shift 2 ;;
    --tile-index) TILE_INDEX="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --user) USER_NAME="$2"; shift 2 ;;
    --remote-root) REMOTE_ROOT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    *) echo "Unknown arg: $1" >&2; usage >&2; exit 1 ;;
  esac
done

[ -n "$PROFILE_JSON" ] || { echo "--profile-json is required" >&2; exit 1; }
[ -f "$PROFILE_JSON" ] || { echo "Missing profile json: $PROFILE_JSON" >&2; exit 1; }
[ -x "$LOCAL_BIN" ] || { echo "Missing replay binary: $LOCAL_BIN" >&2; exit 1; }

readarray -t LAYER_INFO < <(
  python3 - "$PROFILE_JSON" "$LAYER_ID" "$TILE_INDEX" <<'PY'
import json
import sys
profile_path, layer_id_arg, tile_idx_arg = sys.argv[1], sys.argv[2], sys.argv[3]
tile_idx = int(tile_idx_arg)
with open(profile_path, 'r', encoding='utf-8') as f:
    profile = json.load(f)
nodes = profile.get('nodes', [])
if not nodes:
    raise SystemExit('No nodes in profile json')
selected = None
if layer_id_arg:
    target = int(layer_id_arg)
    for node in nodes:
        if int(node.get('layer_id', -1)) == target:
            selected = node
            break
    if selected is None:
        raise SystemExit(f'Layer id not found: {target}')
else:
    for node in nodes:
        if node.get('semantic_op') == 'ffn_down':
            selected = node
            break
    if selected is None:
        selected = nodes[0]
tiles = selected.get('tiles', [])
if not tiles:
    raise SystemExit('Selected node has no tiles')
if tile_idx < 0 or tile_idx >= len(tiles):
    raise SystemExit(f'tile-index out of range: {tile_idx}, tiles={len(tiles)}')
tile = tiles[tile_idx]
print(int(selected.get('layer_id', -1)))
print(str(selected.get('semantic_op', '')))
print(str(selected.get('root_op_name', '')))
print(int(tile.get('m', selected.get('m'))))
print(int(tile.get('n', selected.get('n'))))
print(int(tile.get('k', selected.get('k'))))
print(int(tile_idx))
PY
)

SEL_LAYER_ID="${LAYER_INFO[0]}"
SEL_SEMANTIC="${LAYER_INFO[1]}"
SEL_ROOT_OP="${LAYER_INFO[2]}"
M="${LAYER_INFO[3]}"
N="${LAYER_INFO[4]}"
K="${LAYER_INFO[5]}"
SEL_TILE_INDEX="${LAYER_INFO[6]}"

echo "Selected layer: id=$SEL_LAYER_ID semantic=$SEL_SEMANTIC root_op=$SEL_ROOT_OP tile=$SEL_TILE_INDEX"
echo "Replay shape: m=$M n=$N k=$K"

REMOTE_DIR="$REMOTE_ROOT/shared/runtime"
REMOTE_BIN="$REMOTE_DIR/kv260_layer_gemm_replay_test"
TARGET="${USER_NAME}@${HOST}"
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR)

ssh "${SSH_OPTS[@]}" "$TARGET" "mkdir -p '$REMOTE_DIR'"
scp "${SSH_OPTS[@]}" "$LOCAL_BIN" "$TARGET:$REMOTE_BIN"

REMOTE_CMD=(
  "$REMOTE_BIN"
  --m "$M"
  --n "$N"
  --k "$K"
  --loops 1
  --double-mvin
)
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
  REMOTE_CMD+=("${EXTRA_ARGS[@]}")
fi
printf -v REMOTE_CMD_STR "%q " "${REMOTE_CMD[@]}"
echo "Running on board: $REMOTE_CMD_STR"
ssh "${SSH_OPTS[@]}" "$TARGET" "$REMOTE_CMD_STR"
