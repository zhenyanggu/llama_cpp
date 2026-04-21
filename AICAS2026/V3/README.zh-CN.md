# AICAS2026 V3 提交包

这是面向比赛主办方的正式交付目录。

当前官方目标组合：

- 文本模型：`SmolVLM2-500M-Video-Instruct-Q8_0.gguf`
- mmproj：`mmproj-fallback-search-fb_attn_k-per-tensor.gguf`
- 设备：Kria KV260
- 系统：官方 Ubuntu
- 后端：`MTMD_BACKEND_DEVICE=NPU`
- overlay app：`double_dma_overlayapp`

## 目录说明

- `docs/`
  复现、主机环境、板端环境说明，中英文各一份。
- `scripts/`
  官方执行脚本。`run_v3_submission.sh` 是一键入口。
- `payload/`
  自包含运行载荷，包括二进制、模型、评测脚本、overlay、驱动、测试程序、最小数据集。
- `results/official/`
  官方 release 模式运行结果，每次运行一个 `run_id` 目录。
- `manifests/`
  清单与 SHA256。
- `source_state/`
  当前工作区源码快照信息。

## 一键运行

在主机上执行：

```bash
cd /home/gugugu/work/llama.cpp-kv260-20260407
export BOARD_SUDO_PASSWORD='123456'
bash AICAS2026/V3/scripts/run_v3_submission.sh --skip-build
```

默认目标板：

- SSH: `ubuntu@192.168.0.10`
- 远端目录：`/home/ubuntu/aicas`

如果主办方环境不同，可通过参数覆盖：

```bash
bash AICAS2026/V3/scripts/run_v3_submission.sh \
  --host <board_ip> \
  --user <board_user> \
  --remote-root <remote_root> \
  --sudo-password <board_sudo_password> \
  --skip-build
```

## 当前交付包含

- `double_dma_overlayapp` overlay 文件
- `npu_kv260.ko` 驱动
- `llama-server-kv260-npu` 板端二进制
- `throughput_eval.py` / `acc_eval.py`
- 两个板端 NPU 单层测试程序及对应源码
- 100 个官方采样样本 `sampled_100.json` 与所需图片子集
- QSPI 参考镜像 `BOOT-k26-smk-sdt-v1.05-20250912165210.bin`

## 样本替换

默认使用 `payload/data/sampled_100.json` 和 `payload/data/images/` 下的最小图片集。

如果主办方希望替换测试样本：

1. 准备新的 JSON，字段格式与 `sampled_100.json` 保持一致。
2. 准备与 JSON 中 `image_path` 对应的图片目录。
3. 运行：

```bash
bash AICAS2026/V3/scripts/run_v3_submission.sh \
  --sample-json /path/to/custom.json \
  --sample-images-root /path/to/custom_images \
  --skip-build
```

## 结果文件

正式运行后，本地结果位于：

```text
AICAS2026/V3/results/official/<run_id>/
```

其中关键文件包括：

- `throughput_metrics.json`
- `acc_100sample.json`
- `summary_metrics.json`
- `RESULTS.zh-CN.md`
- `RESULTS.en.md`
- `server.log`
- `board_init.log`
- `readiness_probe.txt`
- `run_meta.json`
