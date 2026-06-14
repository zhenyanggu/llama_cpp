# KV260 AICAS Semi Board Release

This directory contains the scripts for preparing a clean, self-contained KV260
image payload for the AICAS 2026 semi-final evaluation.

## Host Preparation

Build or reuse the KV260 `llama-server`, then prepare the payload:

```bash
bash AICAS2026/aicas_semi/scripts/prepare_board_release_payload.sh --clean
```

Default payload output:

```text
/tmp/aicas-semi-board-release
```

The payload includes:

- prebuilt `bin/llama-server`
- current best text model and mmproj
- full OCRBench image data and `FullTest.json`
- adapted semi-final eval scripts
- official `aicas_semi` package under `official_reference/`
- `npu_kv260.ko`
- final prefill/decode overlay apps
- runtime libraries
- `manifest.json` with file paths and sizes

## Board Install

Free board space first using dry-run inspection:

```bash
bash AICAS2026/aicas_semi/scripts/cleanup_kv260_board.sh
```

After reviewing the printed manifest, move old development artifacts into
quarantine:

```bash
bash AICAS2026/aicas_semi/scripts/cleanup_kv260_board.sh --execute
```

Install the prepared payload:

```bash
bash AICAS2026/aicas_semi/scripts/install_board_release.sh \
  --payload-dir /tmp/aicas-semi-board-release
```

The board entrypoint is:

```text
/home/ubuntu/run_semi.sh
```

## Board Run

Quick hardware and request smoke test:

```bash
bash /home/ubuntu/run_semi.sh --quick-smoke
```

Full semi-final flow:

```bash
bash /home/ubuntu/run_semi.sh --run-id final-image-check
```

Results are written under:

```text
/home/ubuntu/aicas-semi-release/results/<run-id>/results
```

Expected outputs include:

- `sample_30.json`
- `acc_eval_results.json`
- `throughput_metrics.json`
- `energy_metrics.json`
- `ttft_eval_results.json`
- `aicas_submission.json`
- `score_report.json`

## Permanent Deletion

Only after the release payload passes smoke/full validation, delete the
quarantine:

```bash
bash AICAS2026/aicas_semi/scripts/cleanup_kv260_board.sh \
  --execute \
  --delete-quarantine
```

The cleanup script preserves `/home/ubuntu/aicas-semi-release`,
`/home/ubuntu/run_semi.sh`, stock Xilinx firmware apps, and the final prefill
and decode overlay apps.
