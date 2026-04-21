# 板端环境说明

## 官方系统假设

当前交付面向 KV260 官方 Ubuntu 流程。

要求板端具备：

- `/usr/bin/xmutil`
- 可用 sudo
- 可写 `/lib/firmware/xilinx/`
- 可写 `--remote-root`

## 板端目录布局

脚本会在 `--remote-root` 下建立如下目录：

- `shared/gguf/`
- `shared/eval/`
- `shared/data/`
- `shared/lib/`
- `board_support/overlay/double_dma_overlayapp/`
- `board_support/driver/`
- `board_support/qspi/`
- `board_support/tests/`
- `runs/submission-versavlm/<run_id>/`

## Overlay 与驱动初始化

板端初始化脚本：

```bash
scripts/board_init_overlay.sh
```

该脚本会：

1. 安装 `double_dma_overlayapp`
2. 执行 `xmutil unloadapp` / `xmutil loadapp`
3. 加载 `npu_kv260.ko`
4. 修复 `/dev/npu_kv260` 权限
5. 输出 readiness 信息

## QSPI 参考镜像

交付中包含：

```text
payload/board_support/qspi/BOOT-k26-smk-sdt-v1.05-20250912165210.bin
```

它仅作为参考文件提供，不会被脚本自动刷写。
