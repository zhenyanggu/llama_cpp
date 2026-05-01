#!/usr/bin/env bash
set -euo pipefail

SCRIPT_SRC="/home/gugugu/.codex/skills/kv260-llamaserver-throughput/scripts/run_kv260_latest.sh"
SCRIPT_DIR_REAL="/home/gugugu/.codex/skills/kv260-llamaserver-throughput/scripts"
SKILL_DIR_REAL="/tmp/kv260-llamaserver-throughput"
TMP_SCRIPT="$(mktemp /tmp/run_kv260_latest_3dma_perchannel.XXXXXX.sh)"
TMP_ENV_EXPORTS="$(mktemp /tmp/run_kv260_latest_3dma_perchannel_env.XXXXXX.sh)"

cleanup() {
  rm -f "$TMP_SCRIPT"
  rm -f "$TMP_ENV_EXPORTS"
}
trap cleanup EXIT

while IFS='=' read -r name value; do
  case "$name" in
    GGML_NPU_*|LLAMA_MTMD_*|MTMD_PROFILE_*)
      printf 'export %s=%q\n' "$name" "$value" >> "$TMP_ENV_EXPORTS"
      ;;
  esac
done < <(env)

sed \
  -e "s|^SCRIPT_DIR=.*$|SCRIPT_DIR=\"$SCRIPT_DIR_REAL\"|" \
  -e "s|^SKILL_DIR=.*$|SKILL_DIR=\"$SKILL_DIR_REAL\"|" \
  -e "/export MTMD_BACKEND_DEVICE=NPU/r $TMP_ENV_EXPORTS" \
  -e 's/double_dma_overlayapp/3DMA_perchannel_app/g' \
  "$SCRIPT_SRC" > "$TMP_SCRIPT"
chmod +x "$TMP_SCRIPT"

exec "$TMP_SCRIPT" \
  --runtime-libs-dir /home/gugugu/.codex/skills/kv260-llamaserver-throughput/assets/runtime_libs \
  "$@"
