# AICAS2026 V1 Submission Bundle

This directory is the only entrypoint for the official V1 submission flow.

- Official target: `SmolVLM2-500M-Video-Instruct-Q8_0.gguf` + `mmproj-SmolVLM2-500M-Video-Instruct-w8a8-mixed-v1.gguf`
- Official outputs: one `throughput_metrics.json` and one `acc_100sample.json`
- Board target: `ubuntu@192.168.0.10`
- Remote root: `/home/ubuntu/aicas`

`AICAS2026/V1_412` remains the internal archive of earlier manual experiments and raw profiling artifacts. Do not use `V1_412` as the reproduction entrypoint anymore.

## Quick Start

On the host:

```bash
cd /home/gugugu/work/llama.cpp-kv260-20260407
export BOARD_SUDO_PASSWORD='<board sudo password>'
bash AICAS2026/V1/scripts/run_v1_submission.sh
```

After the run:

- official results are under `AICAS2026/V1/results/official/<run_id>/`
- the board is organized under `/home/ubuntu/aicas/`
- the generated bundle payload is under `AICAS2026/V1/payload/`

## Directory Roles

- `docs/`
  Host, board, and reproduction instructions.
- `scripts/`
  The official automation entrypoints.
- `payload/`
  Generated submission payload: binary, models, eval scripts, board support files, runtime libs, and the 100-sample minimal dataset.
- `results/official/`
  Official run outputs, one directory per run id.
- `manifests/`
  Generated checksums and payload manifests.
- `source_state/`
  Frozen source snapshot metadata for the current host workspace.

## Official Scripts

- `scripts/prepare_v1_bundle.sh`
  Build or reuse the KV260 `llama-server`, collect the fixed payload, and generate manifests.
- `scripts/board_init_npu.sh`
  Install `mynpu`, load the app, insert `npu_kv260.ko`, fix `/dev/npu_kv260` permissions, and emit a readiness log.
- `scripts/run_v1_submission.sh`
  The only official end-to-end run entrypoint.
- `scripts/summarize_results.py`
  Merge the throughput and accuracy outputs into a compact summary.

## Current Rules

- Do not save ad hoc logs, JSON files, or model variants directly under `/home/ubuntu`.
- Do not save new official artifacts under `AICAS2026/V1_412`.
- Do not mix development-only outputs from `AICAS/` into official `results/official/`.
- If a run fails, keep the partial artifacts inside the run directory instead of renaming them manually.
