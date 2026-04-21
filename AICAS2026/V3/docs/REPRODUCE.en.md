# V3 Reproduction Guide

## Goal

Reproduce the official KV260 result on Ubuntu with:

- Text model: `SmolVLM2-500M-Video-Instruct-Q8_0.gguf`
- mmproj: `mmproj-fallback-search-fb_attn_k-per-tensor.gguf`
- Backend: `MTMD_BACKEND_DEVICE=NPU`
- Overlay: `double_dma_overlayapp`
- Outputs: `throughput_metrics.json` and `acc_100sample.json`

## Prerequisites

Host:

- `ssh`, `scp`, and `python3`
- Network access to the KV260 board
- KV260 SDK env only if a rebuild is required

Board:

- Official Ubuntu image
- `xmutil` installed
- sudo access for the `ubuntu` user
- enough free space under `/home/ubuntu/aicas`

## One-Shot Flow

```bash
cd /home/gugugu/work/llama.cpp-kv260-20260407
export BOARD_SUDO_PASSWORD='123456'
bash AICAS2026/V3/scripts/run_v3_submission.sh --skip-build
```

This performs the following steps:

1. Refresh `AICAS2026/V3/payload/`
2. Sync models, scripts, images, overlay, and driver to the board
3. Install the overlay under `/lib/firmware/xilinx/double_dma_overlayapp/`
4. Run `xmutil loadapp double_dma_overlayapp`
5. Load `npu_kv260.ko`
6. Start `llama-server` on the board
7. Run throughput evaluation
8. Run 100-sample accuracy evaluation
9. Pull all outputs back into `results/official/<run_id>/`

## Common Variants

Skip rebuild:

```bash
bash AICAS2026/V3/scripts/run_v3_submission.sh --skip-build
```

If overlay and driver are already loaded correctly, rerun only the evaluation:

```bash
bash AICAS2026/V3/scripts/run_v3_submission.sh --skip-build --skip-board-init
```

Target a different board:

```bash
bash AICAS2026/V3/scripts/run_v3_submission.sh \
  --host <board_ip> \
  --user <board_user> \
  --remote-root <remote_root> \
  --sudo-password <password> \
  --skip-build
```

## Replacing the Sample Set

```bash
bash AICAS2026/V3/scripts/run_v3_submission.sh \
  --sample-json /path/to/custom.json \
  --sample-images-root /path/to/custom_images \
  --skip-build
```

Requirements:

- the JSON schema must match `sampled_100.json`
- every `image_path` must resolve under `--sample-images-root`

## Output Locations

Local:

```text
AICAS2026/V3/results/official/<run_id>/
```

Board:

```text
/home/ubuntu/aicas/runs/submission-v3/<run_id>/
```
