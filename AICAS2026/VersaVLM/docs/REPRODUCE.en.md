# VersaVLM Reproduction Guide

## Goal

Reproduce the official KV260 result on Ubuntu with:

- Text model: `payload/gguf/SmolVLM2-500M-Video-Instruct-Q8_0.gguf`
- mmproj: `payload/gguf/mmproj-fallback-search-fb_attn_k-per-tensor.gguf`
- Backend: `MTMD_BACKEND_DEVICE=NPU`
- Overlay: `double_dma_overlayapp`
- Outputs: `throughput_metrics.json` and `acc_100sample.json`

## Prerequisites

Host:

- `ssh`, `scp`, and `python3`
- network access to the KV260 board

Board:

- official Ubuntu image
- `xmutil` installed
- sudo access for the target user
- enough free space under the selected `--remote-root`

## Standard Flow

From the `VersaVLM/` root directory:

```bash
bash scripts/validate_bundle.sh
export BOARD_SUDO_PASSWORD='<board sudo password>'
bash scripts/run_submission.sh \
  --host <board_ip> \
  --user <board_user> \
  --remote-root <remote_root>
```

This performs the following steps:

1. validates that the bundle is self-contained
2. syncs models, scripts, images, overlay, and driver to the board
3. installs the overlay under `/lib/firmware/xilinx/double_dma_overlayapp/`
4. runs `xmutil loadapp double_dma_overlayapp`
5. loads `npu_kv260.ko`
6. starts `llama-server` on the board
7. runs throughput evaluation
8. runs the 100-sample accuracy evaluation
9. pulls the results back into `results/official/<run_id>/`

## Common Variants

If overlay and driver are already loaded correctly, rerun only the evaluation:

```bash
bash scripts/run_submission.sh \
  --host <board_ip> \
  --user <board_user> \
  --remote-root <remote_root> \
  --skip-board-init
```

If you want to skip bundle validation:

```bash
bash scripts/run_submission.sh \
  --host <board_ip> \
  --user <board_user> \
  --remote-root <remote_root> \
  --skip-validate
```

## Replacing the Sample Set

```bash
bash scripts/run_submission.sh \
  --host <board_ip> \
  --user <board_user> \
  --remote-root <remote_root> \
  --sample-json /path/to/custom.json \
  --sample-images-root /path/to/custom_images
```

## Output Locations

Local:

```text
results/official/<run_id>/
```

Board:

```text
<remote_root>/runs/submission-versavlm/<run_id>/
```
