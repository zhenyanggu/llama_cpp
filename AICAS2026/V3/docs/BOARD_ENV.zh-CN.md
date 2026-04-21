# 板端环境说明

## 官方系统假设

当前交付基于 KV260 官方 Ubuntu 流程。

要求板端具备：

- `/usr/bin/xmutil`
- 可用的 sudo
- 可写的 `/lib/firmware/xilinx/`
- 可写的 `/home/ubuntu/aicas/`

## 目录布局

官方板端布局如下：

- `/home/ubuntu/aicas/shared/gguf/`
- `/home/ubuntu/aicas/shared/eval/`
- `/home/ubuntu/aicas/shared/data/`
- `/home/ubuntu/aicas/shared/lib/`
- `/home/ubuntu/aicas/board_support/overlay/double_dma_overlayapp/`
- `/home/ubuntu/aicas/board_support/driver/`
- `/home/ubuntu/aicas/board_support/qspi/`
- `/home/ubuntu/aicas/board_support/tests/`
- `/home/ubuntu/aicas/runs/submission-v3/<run_id>/`

## Overlay 与驱动

板端初始化脚本：

```bash
/home/ubuntu/aicas/scripts/board_init_overlay.sh
```

该脚本会：

1. 安装 `double_dma_overlayapp`
2. 执行 `xmutil unloadapp` / `xmutil loadapp`
3. 加载 `npu_kv260.ko`
4. 修复 `/dev/npu_kv260` 权限
5. 输出 readiness 信息

## QSPI 参考镜像

交付中包含以下参考文件：

```text
BOOT-k26-smk-sdt-v1.05-20250912165210.bin
```

它来自：

```text
/mnt/c/Users/顾振阳/Downloads/
```

仅作为参考随包提供，不会被自动刷写。
