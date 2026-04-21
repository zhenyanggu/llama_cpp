# VersaVLM 提交包

这是一个自包含的主办方交付目录。主办方只需要拿到整个 `VersaVLM/` 目录，就可以按照文档完成复现，不依赖原始仓库。如果复现过程中出现问题，请联系zhenyanggu21@gmail.com，或者+86-15611617103。

相关源代码已开源：
https://github.com/zhenyanggu/llama_cpp
https://github.com/zhenyanggu/VersaVLM

当前正式目标组合：

- 文本模型：`payload/gguf/SmolVLM2-500M-Video-Instruct-Q8_0.gguf`
- mmproj：`payload/gguf/mmproj-fallback-search-fb_attn_k-per-tensor.gguf`
- 设备：Kria KV260
- 系统：官方 Ubuntu
- 后端：`MTMD_BACKEND_DEVICE=NPU`
- overlay app：`double_dma_overlayapp`

## 目录说明

- `docs/`
  复现与环境说明，中英文各一份。
- `scripts/`
  执行脚本。主入口是 `scripts/run_submission.sh`。
- `payload/`
  自包含运行载荷，包括二进制、模型、评测脚本、overlay、驱动、测试程序、最小数据集。
- `results/official/`
  正式 release 模式运行结果。
- `manifests/`
  清单与 SHA256。
- `source_state/`
  原始开发工作区快照，仅用于溯源，不参与主办方执行。

## 主办方最小执行流程

推荐在主机中使用ssh连接板卡以使用脚本一键运行：在主机进入 `VersaVLM/` 根目录后执行：

```bash
bash scripts/validate_bundle.sh
export BOARD_SUDO_PASSWORD='<board sudo password>'
bash scripts/run_submission.sh \
  --host <board_ip> \
  --user <board_user> \
  --remote-root <remote_root>
```

如果不想使用环境变量，也可以显式传入 sudo 密码：

```bash
bash scripts/run_submission.sh \
  --host <board_ip> \
  --user <board_user> \
  --remote-root <remote_root> \
  --sudo-password <board_sudo_password>
```

## 当前交付包含

- `double_dma_overlayapp` overlay 文件
- `npu_kv260.ko` 驱动
- `llama-server-kv260-npu` 板端二进制
- `throughput_eval.py` / `acc_eval.py`
- 两个板端 NPU 单层测试程序及对应源码
- 官方采样 100 条 `sampled_100.json` 与所需图片子集
- QSPI 参考镜像 `BOOT-k26-smk-sdt-v1.05-20250912165210.bin`

## 更换样本

默认使用：

- `payload/data/sampled_100.json`
- `payload/data/images/`

如果主办方希望替换样本：

```bash
bash scripts/run_submission.sh \
  --host <board_ip> \
  --user <board_user> \
  --remote-root <remote_root> \
  --sample-json /path/to/custom.json \
  --sample-images-root /path/to/custom_images
```

要求：

- JSON 结构与 `payload/data/sampled_100.json` 一致
- `image_path` 必须能在 `--sample-images-root` 下找到对应图片

## 结果位置

本地结果位于：

```text
results/official/<run_id>/
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
