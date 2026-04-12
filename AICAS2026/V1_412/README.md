# KV260 mmproj W8A8 NPU V1_412

## 目录说明

本目录汇总 2026-04-12 在 KV260 板端完成的 `mmproj W8A8 + NPU` 真实测试结果，目标是让同仓库用户可以按相同步骤复现。

同目录原始产物：

- `manual-npu-debug-8090.log`
  最终成功跑通 true NPU 路径时的 server 原始日志。
- `manual-throughput-npu-8090.json`
  throughput 原始指标 JSON。
- `SmolVLM2_npu_w8a8_5sample.json`
  5 样本 acc 原始结果 JSON。
- `kv260_sampled5.json`
  本次 5 样本 acc 输入集。
- `board_readiness_snapshot.txt`
  本次整理时板端 readiness 原始快照。
- `commands_used.txt`
  实际复现命令模板。

## 最终结论

- 板端 readiness：满足。
- server 启动：成功。
- 是否真的跑到 NPU：是。
- throughput：成功。
- 5 样本 acc：成功，`3/5`。
- 100 样本 acc：本次未继续执行。

## 真实性能

来自 `manual-throughput-npu-8090.json`：

- Prompt Tokens: `188`
- Completion Tokens: `327`
- Total Tokens: `515`
- Prefill Time: `115201.381 ms`
- Decode Time: `58100.173 ms`
- Total Time: `173301.554 ms`
- Prefill Speed: `1.631924881178291 t/s`
- Decode Speed: `5.628210435793367 t/s`

5 样本 acc：

- 得分：`3/5`
- 失败样本：
  - `svt/image/img_0046.jpg`
    - GT: `center`
    - Pred: `CHANNEL`
  - `IIIT5K/test/1023_14.png`
    - GT: `83KM`
    - Pred: `83 KM`

## true NPU 路径证据

在 `manual-npu-debug-8090.log` 中可以直接看到：

- `clip_ctx: CLIP using NPU backend`
- `register_aicas_w8a8_for_npu: registered 48 AICAS W8A8 tensors for NPU (0 failed)`
- 多次 `npu_backend_graph_plan_compute: start ...`
- 多次 `npu_compute_node: leave root=...`

这说明：

- mmproj/vision 路径不是 CPU fallback。
- 调度器实际把一批 `MUL_MAT/ADD` 节点分给了 NPU backend。
- NPU backend 实际完成了 node compute，而不是只初始化成功。

## 复现前提

主机侧：

- 仓库路径：`/home/gugugu/work/llama.cpp-kv260-20260407`
- 复用 NPU 交叉编译目录：`build-kv260-npu`
- 不覆盖 CPU 基线目录：`build-kv260`
- SDK 路径：`/home/gugugu/petalinux/sdk/kv260-2025.1`

板端：

- SSH 用户：`ubuntu@192.168.0.10`
- `mynpu` app 可 load
- `npu_kv260.ko` 与板端内核 vermagic 匹配
- 每次 `xmutil loadapp mynpu` 后，通常都需要重新给 `/dev/npu_kv260` 赋权限

依赖当前仓库状态里的 NPU 路径修复，关键文件：

- `tools/mtmd/clip.cpp`
- `ggml/src/ggml-npu/ggml-npu.cpp`
- `ggml/src/ggml-npu/ggml-npu-exec.cpp`
- `ggml/src/ggml-npu/ggml-npu-plan.cpp`

## 复现步骤

### 1. 板端 readiness

先在板端 unload/load app，再插入驱动并检查：

```bash
sudo xmutil unloadapp || true
sudo xmutil loadapp mynpu
sudo insmod /home/ubuntu/npu_kv260.ko
sudo chgrp ubuntu /dev/npu_kv260
sudo chmod 660 /dev/npu_kv260
```

检查点：

- `xmutil listapps` 里能看到 `mynpu`
- `/dev/npu_kv260` 存在
- `/proc/modules` 里有 `npu_kv260`
- `/sys/bus/platform/devices` 里有 `a0000000.T_NPU_FPGA`

### 2. 只重建 NPU server

```bash
cd /home/gugugu/work/llama.cpp-kv260-20260407
cmake --build build-kv260-npu --target llama-server -j8
```

### 3. 部署到板端

```bash
scp build-kv260-npu/bin/llama-server \
  ubuntu@192.168.0.10:/home/ubuntu/llama-server-npu-test
```

### 4. 启动板端 server

```bash
ssh ubuntu@192.168.0.10

env \
  LD_LIBRARY_PATH=/home/ubuntu/aicas/shared/lib \
  MTMD_BACKEND_DEVICE=NPU \
  GGML_NPU_SPM_BYTES=524288 \
  GGML_NPU_ACC_BYTES=524288 \
  GGML_NPU_GUARD_BYTES=4096 \
  GGML_NPU_STAGE2_K_BYTES=4096 \
  NPU_CMA_SIZE=256M \
  AICAS_MMPROJ_W8A8_DEBUG=1 \
  LLAMA_LOG_VERBOSITY=1 \
  /home/ubuntu/llama-server-npu-test \
    -v \
    --no-warmup \
    --host 127.0.0.1 \
    --port 8090 \
    --alias smolvlm2-gguf-npu \
    -m /home/ubuntu/aicas/shared/gguf/SmolVLM2-500M-Video-Instruct-f16.gguf \
    --mmproj /home/ubuntu/aicas/shared/gguf/mmproj-SmolVLM2-500M-Video-Instruct-w8a8-mixed-v1.gguf \
    -t 4
```

### 5. throughput

```bash
ssh ubuntu@192.168.0.10 \
  "python3 /home/ubuntu/aicas/shared/eval/throughput_eval.py \
    -i /home/ubuntu/aicas/shared/data/IIIT5K/test/2543_2.png \
    -o /home/ubuntu/manual-throughput-npu-8090.json \
    --base-url http://127.0.0.1:8090/v1 \
    --model smolvlm2-gguf-npu"
```

### 6. 5 样本 acc

先把本目录里的 `kv260_sampled5.json` 传到板端：

```bash
scp AICAS2026/V1_412/kv260_sampled5.json \
  ubuntu@192.168.0.10:/home/ubuntu/kv260_sampled5.json
```

再执行：

```bash
ssh ubuntu@192.168.0.10 \
  "cd /home/ubuntu/aicas/shared/eval && \
   python3 ./acc_eval.py \
     --image_folder /home/ubuntu/aicas/shared/data \
     --OCRBench_file /home/ubuntu/kv260_sampled5.json \
     --output_folder /home/ubuntu/acc-npu-8090 \
     --save_name SmolVLM2_npu_w8a8_5sample \
     --base-url http://127.0.0.1:8090/v1 \
     --model smolvlm2-gguf-npu \
     --request-timeout 600 \
     --progress-every 1"
```

## 这次为跑通做过的关键修复

- `clip.cpp`
  - `MTMD_BACKEND_DEVICE=NPU` 时显式走 `ggml_backend_npu_init()`，避免 vision/mmproj 回退到 CPU。
- `ggml-npu.cpp`
  - 补上 `graph_compute`，修掉之前调用空函数指针导致的 `SIGSEGV`。
  - NPU host buffer 改为对齐分配，修掉 compute meta buffer 的对齐损耗问题。
- `ggml-npu-plan.cpp`
  - 对缺少静态激活量化参数的非 AICAS-W8A8 节点，提前判定为不支持 NPU，避免运行时 500。

## 建议

- 如果继续做 `100` 样本 acc，建议继续复用同一套参数和当前 repo 状态。
- 如果重新 `xmutil loadapp mynpu`，不要忘记重新处理 `/dev/npu_kv260` 权限。
