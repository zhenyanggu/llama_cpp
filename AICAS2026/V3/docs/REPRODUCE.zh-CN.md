# V3 复现说明

## 目标

在 KV260 官方 Ubuntu 系统上复现以下正式结果：

- 文本模型：`SmolVLM2-500M-Video-Instruct-Q8_0.gguf`
- mmproj：`mmproj-fallback-search-fb_attn_k-per-tensor.gguf`
- 后端：`MTMD_BACKEND_DEVICE=NPU`
- overlay：`double_dma_overlayapp`
- 输出：`throughput_metrics.json` 与 `acc_100sample.json`

## 前置条件

主机：

- 已安装 SSH / `scp` / `python3`
- 可访问 KV260
- 如果需要重新构建，则需准备 KV260 SDK env

板端：

- 官方 Ubuntu
- 自带 `xmutil`
- `ubuntu` 用户有 sudo 权限
- `/home/ubuntu` 下有足够空间用于 `/home/ubuntu/aicas`

## 一键流程

```bash
cd /home/gugugu/work/llama.cpp-kv260-20260407
export BOARD_SUDO_PASSWORD='123456'
bash AICAS2026/V3/scripts/run_v3_submission.sh --skip-build
```

该命令会完成：

1. 生成或刷新 `AICAS2026/V3/payload/`
2. 同步模型、脚本、图片、overlay、驱动到板端
3. 安装 overlay 到 `/lib/firmware/xilinx/double_dma_overlayapp/`
4. 执行 `xmutil loadapp double_dma_overlayapp`
5. 加载 `npu_kv260.ko`
6. 启动板端 `llama-server`
7. 跑 throughput
8. 跑 100 样本 acc eval
9. 拉回所有结果到本地 `results/official/<run_id>/`

## 常用变体

只跳过重新构建：

```bash
bash AICAS2026/V3/scripts/run_v3_submission.sh --skip-build
```

如果板端 overlay 和驱动已经正确加载，只重跑评测：

```bash
bash AICAS2026/V3/scripts/run_v3_submission.sh --skip-build --skip-board-init
```

指定不同板卡：

```bash
bash AICAS2026/V3/scripts/run_v3_submission.sh \
  --host <board_ip> \
  --user <board_user> \
  --remote-root <remote_root> \
  --sudo-password <password> \
  --skip-build
```

## 更换测试样本

```bash
bash AICAS2026/V3/scripts/run_v3_submission.sh \
  --sample-json /path/to/custom.json \
  --sample-images-root /path/to/custom_images \
  --skip-build
```

要求：

- JSON 结构与 `sampled_100.json` 一致
- `image_path` 必须能在 `--sample-images-root` 下找到对应图片

## 结果位置

本地：

```text
AICAS2026/V3/results/official/<run_id>/
```

板端：

```text
/home/ubuntu/aicas/runs/submission-v3/<run_id>/
```
