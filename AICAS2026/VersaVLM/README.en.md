# VersaVLM Submission Bundle

This is a self-contained organizer-facing delivery directory. The organizer only needs the full `VersaVLM/` folder and does not need the original repository. If reproduction issues occur, or if the modified llama.cpp / RTL project source code is needed, contact `zhenyanggu21@gmail.com` or `+86-15611617103`.

https://github.com/zhenyanggu/llama_cpp
https://github.com/zhenyanggu/VersaVLM

Official target configuration:

- Text model: `payload/gguf/SmolVLM2-500M-Video-Instruct-Q8_0.gguf`
- mmproj: `payload/gguf/mmproj-fallback-search-fb_attn_k-per-tensor.gguf`
- Device: Kria KV260
- OS: official Ubuntu
- Backend: `MTMD_BACKEND_DEVICE=NPU`
- Overlay app: `double_dma_overlayapp`

## Directory Layout

- `docs/`
  Reproduction and environment notes in both Chinese and English.
- `scripts/`
  Execution scripts. The main entrypoint is `scripts/run_submission.sh`.
- `payload/`
  Self-contained runtime payload: binaries, models, eval scripts, overlay, driver, test programs, and minimal dataset.
- `results/official/`
  Official release-mode outputs.
- `manifests/`
  Bundle manifest and SHA256 checksums.
- `source_state/`
  Snapshot from the original development workspace, kept only for provenance.
- `docs/OPTIMIZATION.zh-CN.md` / `docs/OPTIMIZATION.en.md`
  NPU acceleration and optimization method notes.

## Minimal Organizer Workflow

It is recommended to connect to the board from the host through SSH and use the provided scripts for one-command execution. From the `VersaVLM/` root directory:

```bash
bash scripts/validate_bundle.sh
export BOARD_SUDO_PASSWORD='<board sudo password>'
bash scripts/run_submission.sh \
  --host <board_ip> \
  --user <board_user> \
  --remote-root <remote_root>
```

If you prefer not to use an environment variable, pass the sudo password explicitly:

```bash
bash scripts/run_submission.sh \
  --host <board_ip> \
  --user <board_user> \
  --remote-root <remote_root> \
  --sudo-password <board_sudo_password>
```

## Included Assets

- `double_dma_overlayapp` overlay files
- `npu_kv260.ko` driver
- `llama-server-kv260-npu` board binary
- `throughput_eval.py` / `acc_eval.py`
- Two board-side NPU layer test binaries and their source files
- Official 100-sample subset `sampled_100.json` plus required images
- QSPI reference image `BOOT-k26-smk-sdt-v1.05-20250912165210.bin`

## Replacing the Sample Set

Default inputs:

- `payload/data/sampled_100.json`
- `payload/data/images/`

To run on a different sample set:

```bash
bash scripts/run_submission.sh \
  --host <board_ip> \
  --user <board_user> \
  --remote-root <remote_root> \
  --sample-json /path/to/custom.json \
  --sample-images-root /path/to/custom_images
```

Requirements:

- the JSON schema must match `payload/data/sampled_100.json`
- every `image_path` must resolve under `--sample-images-root`

## Output Location

Local outputs are written to:

```text
results/official/<run_id>/
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
