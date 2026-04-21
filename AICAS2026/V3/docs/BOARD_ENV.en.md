# Board Environment Notes

## Official System Assumption

This bundle targets the standard KV260 Ubuntu flow.

The board is expected to provide:

- `/usr/bin/xmutil`
- working sudo access
- writable `/lib/firmware/xilinx/`
- writable `/home/ubuntu/aicas/`

## Directory Layout

The official board-side layout is:

- `/home/ubuntu/aicas/shared/gguf/`
- `/home/ubuntu/aicas/shared/eval/`
- `/home/ubuntu/aicas/shared/data/`
- `/home/ubuntu/aicas/shared/lib/`
- `/home/ubuntu/aicas/board_support/overlay/double_dma_overlayapp/`
- `/home/ubuntu/aicas/board_support/driver/`
- `/home/ubuntu/aicas/board_support/qspi/`
- `/home/ubuntu/aicas/board_support/tests/`
- `/home/ubuntu/aicas/runs/submission-v3/<run_id>/`

## Overlay and Driver

Board initialization script:

```bash
/home/ubuntu/aicas/scripts/board_init_overlay.sh
```

This script:

1. installs `double_dma_overlayapp`
2. runs `xmutil unloadapp` / `xmutil loadapp`
3. loads `npu_kv260.ko`
4. fixes `/dev/npu_kv260` permissions
5. emits readiness details

## QSPI Reference Image

The bundle includes the following reference file:

```text
BOOT-k26-smk-sdt-v1.05-20250912165210.bin
```

It was copied from:

```text
/mnt/c/Users/顾振阳/Downloads/
```

It is included for reference only and is not flashed automatically.
