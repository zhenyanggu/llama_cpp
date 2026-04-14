# Reproduce V1

## Goal

Reproduce the official V1 result on a clean KV260 board with:

- text model: `SmolVLM2-500M-Video-Instruct-Q8_0.gguf`
- mmproj: `mmproj-SmolVLM2-500M-Video-Instruct-w8a8-mixed-v1.gguf`
- backend: `MTMD_BACKEND_DEVICE=NPU`
- outputs: `throughput_metrics.json` and `acc_100sample.json`

## Preconditions

Host:

- repo path: `/home/gugugu/work/llama.cpp-kv260-20260407`
- KV260 SDK env available
- SSH access to `ubuntu@192.168.0.10`

Board:

- Ubuntu image with `xmutil`
- sudo access for the `ubuntu` user
- enough free space under `/home` for `/home/ubuntu/aicas`

## One Command Flow

```bash
cd /home/gugugu/work/llama.cpp-kv260-20260407
export BOARD_SUDO_PASSWORD='<board sudo password>'
bash AICAS2026/V1/scripts/run_v1_submission.sh
```

That command does the following:

1. Builds or reuses the KV260 NPU `llama-server`.
2. Generates `AICAS2026/V1/payload/`.
3. Syncs the official payload to `/home/ubuntu/aicas/`.
4. Installs `mynpu` under `/lib/firmware/xilinx/mynpu/`.
5. Runs `xmutil loadapp mynpu`.
6. Loads `npu_kv260.ko` and fixes `/dev/npu_kv260` permissions.
7. Starts `llama-server` on the board.
8. Runs throughput on `IIIT5K/test/2543_2.png`.
9. Runs 100-sample accuracy on `sampled_100.json`.
10. Pulls all official artifacts back to `AICAS2026/V1/results/official/<run_id>/`.

## Expected Outputs

Local:

- `results/official/<run_id>/board_readiness.txt`
- `results/official/<run_id>/server.log`
- `results/official/<run_id>/throughput_metrics.json`
- `results/official/<run_id>/acc_100sample.json`
- `results/official/<run_id>/run_meta.json`
- `results/official/<run_id>/summary_metrics.json`
- `results/official/<run_id>/RESULTS.md`

Board:

- `/home/ubuntu/aicas/shared/gguf/...`
- `/home/ubuntu/aicas/shared/eval/...`
- `/home/ubuntu/aicas/shared/data/...`
- `/home/ubuntu/aicas/shared/lib/...`
- `/home/ubuntu/aicas/board_support/...`
- `/home/ubuntu/aicas/runs/submission-q8-w8a8/<run_id>/...`

## Manual Recovery

If the run stops after board initialization, do not move files by hand. Re-run:

```bash
bash AICAS2026/V1/scripts/run_v1_submission.sh --skip-build
```

If the board was already initialized correctly and only the evaluation needs to be repeated:

```bash
bash AICAS2026/V1/scripts/run_v1_submission.sh --skip-build --skip-board-init
```

## Notes

- The official payload only includes the 100-sample minimal dataset, not the full `AICAS/data` tree.
- `AICAS2026/V1_412` is archive-only and is not part of the official reproduction chain.
- The board-side Python flow uses the system `python3`; no extra Python package installation is required by the current V1 evaluation scripts.
