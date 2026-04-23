# VersaVLM 复现说明

## 目标

在 KV260 官方 Ubuntu 系统上复现以下正式结果：

- 文本模型：`payload/gguf/SmolVLM2-500M-Video-Instruct-Q8_0.gguf`
- mmproj：`payload/gguf/mmproj-fallback-search-fb_attn_k-per-tensor.gguf`
- 后端：`MTMD_BACKEND_DEVICE=NPU`
- overlay：`double_dma_overlayapp`
- 输出：`throughput_metrics.json` 与 `acc_100sample.json`

## 前置条件

主机：

- 已安装 `ssh` / `scp` / `python3`
- 能访问 KV260

板端：

- 官方 Ubuntu
- 自带 `xmutil`
- 目标用户有 sudo 权限
- 有足够空间容纳 `--remote-root` 下的运行文件

## 标准执行流程

在 `VersaVLM/` 根目录执行：

```bash
bash scripts/validate_bundle.sh
export BOARD_SUDO_PASSWORD='<board sudo password>'
bash scripts/run_submission.sh \
  --host <board_ip> \
  --user <board_user> \
  --remote-root <remote_root>
```

该命令会完成：

1. 校验 bundle 自包含完整性
2. 同步模型、脚本、图片、overlay、驱动到板端
3. 安装 overlay 到 `/lib/firmware/xilinx/double_dma_overlayapp/`
4. 执行 `xmutil loadapp double_dma_overlayapp`
5. 加载 `npu_kv260.ko`
6. 启动板端 `llama-server`
7. 跑 throughput
8. 若未加 `--skip-acc`，则跑 100 样本 acc eval
9. 拉回结果到 `results/official/<run_id>/`

## 常用变体

如果 overlay 和驱动已经正确加载，只重跑评测：

```bash
bash scripts/run_submission.sh \
  --host <board_ip> \
  --user <board_user> \
  --remote-root <remote_root> \
  --skip-board-init
```

如果只想跳过 bundle 校验：

```bash
bash scripts/run_submission.sh \
  --host <board_ip> \
  --user <board_user> \
  --remote-root <remote_root> \
  --skip-validate
```

如果想先只跑 throughput（跳过 acc）：

```bash
bash scripts/run_submission.sh \
  --host <board_ip> \
  --user <board_user> \
  --remote-root <remote_root> \
  --skip-board-init \
  --skip-acc
```

## 更换测试样本

```bash
bash scripts/run_submission.sh \
  --host <board_ip> \
  --user <board_user> \
  --remote-root <remote_root> \
  --sample-json /path/to/custom.json \
  --sample-images-root /path/to/custom_images
```

## 结果位置

本地：

```text
results/official/<run_id>/
```

板端：

```text
<remote_root>/runs/submission-versavlm/<run_id>/
```
