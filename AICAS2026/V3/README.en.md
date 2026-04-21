# AICAS2026 V3 Submission Bundle

This is the organizer-facing release bundle.

Official target configuration:

- Text model: `SmolVLM2-500M-Video-Instruct-Q8_0.gguf`
- mmproj: `mmproj-fallback-search-fb_attn_k-per-tensor.gguf`
- Device: Kria KV260
- OS: official Ubuntu
- Backend: `MTMD_BACKEND_DEVICE=NPU`
- Overlay app: `double_dma_overlayapp`

## Directory Layout

- `docs/`
  Reproduction, host environment, and board environment notes in both Chinese and English.
- `scripts/`
  Official automation scripts. `run_v3_submission.sh` is the one-shot entrypoint.
- `payload/`
  Self-contained runtime payload: binaries, models, eval scripts, overlay, driver, test programs, and minimal dataset.
- `results/official/`
  Official release-mode outputs, one directory per `run_id`.
- `manifests/`
  Bundle manifest and SHA256 checksums.
- `source_state/`
  Source snapshot metadata for the exact workspace used.

## One-Command Run

On the host:

```bash
cd /home/gugugu/work/llama.cpp-kv260-20260407
export BOARD_SUDO_PASSWORD='123456'
bash AICAS2026/V3/scripts/run_v3_submission.sh --skip-build
```

Default board target:

- SSH: `ubuntu@192.168.0.10`
- Remote root: `/home/ubuntu/aicas`

If the organizer uses a different board setup, override the parameters:

```bash
bash AICAS2026/V3/scripts/run_v3_submission.sh \
  --host <board_ip> \
  --user <board_user> \
  --remote-root <remote_root> \
  --sudo-password <board_sudo_password> \
  --skip-build
```

## Included Assets

- `double_dma_overlayapp` overlay files
- `npu_kv260.ko` driver
- `llama-server-kv260-npu` board binary
- `throughput_eval.py` / `acc_eval.py`
- Two board-side NPU layer test binaries and their source files
- Minimal official 100-sample set: `sampled_100.json` plus required images
- QSPI reference image: `BOOT-k26-smk-sdt-v1.05-20250912165210.bin`

## Replacing the Sample Set

The default run uses `payload/data/sampled_100.json` and the image subset under `payload/data/images/`.

To run on a different sample set:

1. Prepare a JSON file with the same schema as `sampled_100.json`.
2. Prepare an image root matching the `image_path` fields.
3. Run:

```bash
bash AICAS2026/V3/scripts/run_v3_submission.sh \
  --sample-json /path/to/custom.json \
  --sample-images-root /path/to/custom_images \
  --skip-build
```

## Output Files

Official outputs are written to:

```text
AICAS2026/V3/results/official/<run_id>/
```

Key artifacts:

- `throughput_metrics.json`
- `acc_100sample.json`
- `summary_metrics.json`
- `RESULTS.zh-CN.md`
- `RESULTS.en.md`
- `server.log`
- `board_init.log`
- `readiness_probe.txt`
- `run_meta.json`
