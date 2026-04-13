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

## Q8_0 版本（不做 profile）

本节固定使用：

- 主模型：`/home/ubuntu/aicas/shared/gguf/SmolVLM2-500M-Video-Instruct-Q8_0.gguf`
- mmproj：`/home/ubuntu/aicas/shared/gguf/mmproj-SmolVLM2-500M-Video-Instruct-w8a8-mixed-v1.gguf`

只做真实性能和 acc，不开启 profile 相关环境变量。

### 1. 确认模型在板端

如果板端还没有 Q8_0 主模型，先从主机传输：

```bash
scp AICAS/gguf/SmolVLM2-500M-Video-Instruct-Q8_0.gguf \
  ubuntu@192.168.0.10:/home/ubuntu/aicas/shared/gguf/
```

### 2. 启动 Q8_0 + NPU server（无 profile）

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
  /home/ubuntu/llama-server-npu-test \
    --no-warmup \
    --host 127.0.0.1 \
    --port 8100 \
    --alias smolvlm2-gguf-npu-q8 \
    -m /home/ubuntu/aicas/shared/gguf/SmolVLM2-500M-Video-Instruct-Q8_0.gguf \
    --mmproj /home/ubuntu/aicas/shared/gguf/mmproj-SmolVLM2-500M-Video-Instruct-w8a8-mixed-v1.gguf \
    -t 4
```

### 3. throughput

```bash
ssh ubuntu@192.168.0.10 \
  "python3 /home/ubuntu/aicas/shared/eval/throughput_eval.py \
    -i /home/ubuntu/aicas/shared/data/IIIT5K/test/2543_2.png \
    -o /home/ubuntu/manual-throughput-npu-8100-q8_0.json \
    --base-url http://127.0.0.1:8100/v1 \
    --model smolvlm2-gguf-npu-q8"
```

### 4. acc测试

如果板端没有样本文件，先传输：

```bash
scp AICAS2026/V1_412/kv260_sampled5.json \
  ubuntu@192.168.0.10:/home/ubuntu/kv260_sampled5.json
```

执行：这里跑100个样本的测试，正确率为50/100

```bash
ssh ubuntu@192.168.0.10 \
  "cd /home/ubuntu/aicas/shared/eval && \
   python3 ./acc_eval.py \
     --image_folder /home/ubuntu/aicas/shared/data \
     --OCRBench_file /home/ubuntu/aicas/shared/eval/sampled.json \
     --output_folder /home/ubuntu/acc-npu-8100-q8_0 \
     --save_name SmolVLM2_npu_q8_0_100sample \
     --base-url http://127.0.0.1:8100/v1 \
     --model smolvlm2-gguf-npu-q8 \
     --request-timeout 600 \
     --progress-every 1"
```

## 如何开始 Profile（KV260）

下面给两条线：

- 真实链路 profile（`llama-server + throughput_eval.py`）：看端到端 prefill/decode。
- 深度 mmproj profile（`llama-mtmd-profiler`）：看 mmproj 的 operator / NPU stage 明细。

### A. 真实链路 profile（推荐先跑）

#### A1. 启动 NPU server 并开启 profile 输出

如何关闭之前的服务：

```
ps -ef
pkill -9 -f "/home/ubuntu/llama-server-npu-test" || true
pkill -9 -f "/home/ubuntu/llama-server-cpu-test" || true
pkill -9 -f "llama-server" || true
```



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
  LLAMA_MTMD_PREFILL_SUMMARY_JSON=/home/ubuntu/mtmd_prefill_summary_npu.json \
  GGML_NPU_PROFILE_JSON=/home/ubuntu/mmproj_raw_npu_node_trace.json \
  NPU_PROFILE_OUT=/home/ubuntu/mmproj_raw_runtime.json \
  /home/ubuntu/llama-server-npu-test \
    --no-warmup \
    --host 127.0.0.1 \
    --port 8092 \
    --alias smolvlm2-gguf-npu \
    -m /home/ubuntu/aicas/shared/gguf/SmolVLM2-500M-Video-Instruct-f16.gguf \
    --mmproj /home/ubuntu/aicas/shared/gguf/mmproj-SmolVLM2-500M-Video-Instruct-w8a8-mixed-v1.gguf \
    -t 4 \
    >/home/ubuntu/server-npu-summary.log 2>&1
```

#### A2. 触发一次 throughput 请求

```bash
ssh ubuntu@192.168.0.10 \
  "python3 /home/ubuntu/aicas/shared/eval/throughput_eval.py \
    -i /home/ubuntu/aicas/shared/data/IIIT5K/test/2543_2.png \
    -o /home/ubuntu/manual-throughput-npu-8092.json \
    --base-url http://127.0.0.1:8092/v1 \
    --model smolvlm2-gguf-npu"
```

#### A3. 拉回 profile 产物并快速检查

```bash
scp ubuntu@192.168.0.10:/home/ubuntu/mtmd_prefill_summary_npu.json AICAS2026/V1_412/
scp ubuntu@192.168.0.10:/home/ubuntu/mmproj_raw_npu_node_trace.json AICAS2026/V1_412/
scp ubuntu@192.168.0.10:/home/ubuntu/mmproj_raw_runtime.json AICAS2026/V1_412/
scp ubuntu@192.168.0.10:/home/ubuntu/manual-throughput-npu-8092.json AICAS2026/V1_412/

jq '.summary' AICAS2026/V1_412/mmproj_raw_runtime.json
jq '.summary' AICAS2026/V1_412/mmproj_raw_npu_node_trace.json
jq '.chunks' AICAS2026/V1_412/mtmd_prefill_summary_npu.json
```

### B. 深度 mmproj profile（operator 级）

先构建并部署：

```bash
cd /home/gugugu/work/llama.cpp-kv260-20260407
cmake --build build-kv260-npu --target llama-mtmd-profiler -j8
scp build-kv260-npu/bin/llama-mtmd-profiler \
  ubuntu@192.168.0.10:/home/ubuntu/llama-mtmd-profiler
```

在板端执行（会直接生成四份 raw JSON）：

```bash
ssh ubuntu@192.168.0.10 \
  "env LD_LIBRARY_PATH=/home/ubuntu/aicas/shared/lib MTMD_BACKEND_DEVICE=NPU \
    /home/ubuntu/llama-mtmd-profiler \
      --mmproj-only \
      -m /home/ubuntu/aicas/shared/gguf/SmolVLM2-500M-Video-Instruct-f16.gguf \
      --mmproj /home/ubuntu/aicas/shared/gguf/mmproj-SmolVLM2-500M-Video-Instruct-w8a8-mixed-v1.gguf \
      --image /home/ubuntu/aicas/shared/data/IIIT5K/test/2543_2.png \
      -t 4 \
      --profile-output /home/ubuntu/mmproj_raw_run.json \
      --profile-mmproj-output /home/ubuntu/mmproj_raw_operator.json \
      --npu-runtime-profile-output /home/ubuntu/mmproj_raw_runtime.json \
      --npu-node-trace-output /home/ubuntu/mmproj_raw_npu_node_trace.json"
```

回主机后可复用本目录脚本生成汇总：

```bash
python3 AICAS2026/V1_412/generate_mmproj_profile_report.py \
  --mmproj-run AICAS2026/V1_412/mmproj_raw_run.json \
  --mmproj-operator AICAS2026/V1_412/mmproj_raw_operator.json \
  --npu-runtime-profile AICAS2026/V1_412/mmproj_raw_runtime.json \
  --npu-node-trace AICAS2026/V1_412/mmproj_raw_npu_node_trace.json \
  --board kv260 \
  --output-json AICAS2026/V1_412/mmproj_profile_report.json \
  --output-md AICAS2026/V1_412/mmproj_performance_analysis.md
```

## 如何测试不同版本模型 / mmproj

建议固定输入与参数，只替换模型版本，这样结果可比：

- 固定：同一张测试图、`-t 4`、同一个 `throughput_eval.py`。
- 变更：`-m`（主模型版本）和 `--mmproj`（投影头版本）。

示例（NPU 路径，循环测多个 mmproj 版本）：

```bash
ssh ubuntu@192.168.0.10 '
set -e
MODEL=/home/ubuntu/aicas/shared/gguf/SmolVLM2-500M-Video-Instruct-f16.gguf
for MMPROJ in \
  /home/ubuntu/aicas/shared/gguf/mmproj-SmolVLM2-500M-Video-Instruct-w8a8-mixed-v1.gguf \
  /home/ubuntu/aicas/shared/gguf/mmproj-SmolVLM2-500M-Video-Instruct-w8a8-mixed-v2.gguf
do
  TAG=$(basename "$MMPROJ" .gguf)
  pkill -f llama-server-npu-test || true
  env LD_LIBRARY_PATH=/home/ubuntu/aicas/shared/lib \
      MTMD_BACKEND_DEVICE=NPU \
      /home/ubuntu/llama-server-npu-test \
      --no-warmup --host 127.0.0.1 --port 8095 --alias smolvlm2-gguf-npu \
      -m "$MODEL" --mmproj "$MMPROJ" -t 4 \
      >/home/ubuntu/server-$TAG.log 2>&1 &
  sleep 3
  python3 /home/ubuntu/aicas/shared/eval/throughput_eval.py \
      -i /home/ubuntu/aicas/shared/data/IIIT5K/test/2543_2.png \
      -o /home/ubuntu/throughput-$TAG.json \
      --base-url http://127.0.0.1:8095/v1 \
      --model smolvlm2-gguf-npu
done
'
```

如果某个版本文件不存在，先在板端 `ls /home/ubuntu/aicas/shared/gguf/` 确认真实文件名，再替换命令中的路径。

## 如何测试纯 CPU 和 CPU+NPU

核心区别只有一条：

- 纯 CPU：`MTMD_BACKEND_DEVICE=CPU`（或不设置，且不配 NPU profile 环境）。
- CPU+NPU：`MTMD_BACKEND_DEVICE=NPU` + NPU 相关环境。

### 纯 CPU 基线

```bash
ssh ubuntu@192.168.0.10

env \
  LD_LIBRARY_PATH=/home/ubuntu/aicas/shared/lib \
  MTMD_BACKEND_DEVICE=CPU \
  LLAMA_MTMD_PREFILL_SUMMARY_JSON=/home/ubuntu/mtmd_prefill_summary_cpu.json \
  /home/ubuntu/llama-server-cpu-test \
    --no-warmup \
    --host 127.0.0.1 \
    --port 8093 \
    --alias smolvlm2-gguf-cpu \
    -m /home/ubuntu/aicas/shared/gguf/SmolVLM2-500M-Video-Instruct-f16.gguf \
    --mmproj /home/ubuntu/aicas/shared/gguf/mmproj-SmolVLM2-500M-Video-Instruct-w8a8-mixed-v1.gguf \
    -t 4 \
    >/home/ubuntu/server-cpu-summary.log 2>&1
```

```bash
ssh ubuntu@192.168.0.10 \
  "python3 /home/ubuntu/aicas/shared/eval/throughput_eval.py \
    -i /home/ubuntu/aicas/shared/data/IIIT5K/test/2543_2.png \
    -o /home/ubuntu/manual-throughput-cpu-8093.json \
    --base-url http://127.0.0.1:8093/v1 \
    --model smolvlm2-gguf-cpu"
```

### CPU+NPU 对照

```bash
ssh ubuntu@192.168.0.10 \
  "python3 /home/ubuntu/aicas/shared/eval/throughput_eval.py \
    -i /home/ubuntu/aicas/shared/data/IIIT5K/test/2543_2.png \
    -o /home/ubuntu/manual-throughput-npu-8092.json \
    --base-url http://127.0.0.1:8092/v1 \
    --model smolvlm2-gguf-npu"
```

最后对比：

```bash
jq '{prefill_ms:.prefill_ms, decode_ms:.decode_ms, total_ms:.total_ms, prefill_tps:.prefill_tps, decode_tps:.decode_tps}' \
  AICAS2026/V1_412/manual-throughput-cpu-8093.json
jq '{prefill_ms:.prefill_ms, decode_ms:.decode_ms, total_ms:.total_ms, prefill_tps:.prefill_tps, decode_tps:.decode_tps}' \
  AICAS2026/V1_412/manual-throughput-npu-8092.json
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
