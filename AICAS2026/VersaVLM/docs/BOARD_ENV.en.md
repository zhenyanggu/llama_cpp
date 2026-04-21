# Board Environment Notes

## Official System Assumption

This bundle targets the standard KV260 Ubuntu flow.

The board is expected to provide:

- `/usr/bin/xmutil`
- working sudo access
- writable `/lib/firmware/xilinx/`
- writable `--remote-root`

## Board Layout

The scripts create the following layout under `--remote-root`:

- `shared/gguf/`
- `shared/eval/`
- `shared/data/`
- `shared/lib/`
- `board_support/overlay/double_dma_overlayapp/`
- `board_support/driver/`
- `board_support/qspi/`
- `board_support/tests/`
- `runs/submission-versavlm/<run_id>/`

## Overlay and Driver Initialization

Board initialization script:

```bash
scripts/board_init_overlay.sh
```

This script:

1. installs `double_dma_overlayapp`
2. runs `xmutil unloadapp` / `xmutil loadapp`
3. loads `npu_kv260.ko`
4. fixes `/dev/npu_kv260` permissions
5. emits readiness details

## QSPI Reference Image

The bundle includes:

```text
payload/board_support/qspi/BOOT-k26-smk-sdt-v1.05-20250912165210.bin
```

It is provided as a reference only and is not flashed automatically.
