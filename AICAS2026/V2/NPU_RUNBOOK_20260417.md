# KV260 NPU Runbook (2026-04-17)

This document records the confirmed KV260 NPU configuration that:

- completes `throughput-only` successfully
- produces readable text instead of gibberish
- reproduces the expected `acc5 = 3/5`
- restores correct `precomp` bias compensation semantics

## Canonical Version

- repo: `llama.cpp-kv260-20260407`
- git head: `20be41e5fca6e67715e8cc1af26c55d6ec616191`
- canonical build dir: `build-kv260-npu-current`
- runtime binary sha256:
  `0bd944b7bc125bb188ddc6b3c6f1c45f85834f7442dc8307c1c794614f88e876`

Do not mix this with `build-kv260-npu`.

## Model Pair

- text model:
  `AICAS/gguf/SmolVLM2-500M-Video-Instruct-Q8_0.gguf`
- mmproj:
  `AICAS/gguf/mmproj-fallback-search-fb_attn_k-per-tensor.gguf`

## Confirmed Working Bias Semantics

The working path is:

- AICAS W8A8 default bias mode resolves to `precomp`
- loader pre-fuses compensation into bias tensors by default
- runtime postprocess adds:
  - model bias only

Formula:

```text
fused_bias[m] = bias[m] - act_scale * w_scale[m] * act_zero_point_i8 * sum_w[m]
out = mvout_fp32 * w_scale[m] + fused_bias[m]
```

Notes:

- explicit `GGML_NPU_AICAS_BIAS_MODE=raw` remains available as fallback/debug path.
- `precomp` was previously broken because NPU backend `ggml_backend_tensor_get/set()` ignored tensor offsets and read/wrote from buffer base.
- fixing NPU buffer I/O to use `tensor->data + offset` restored `precomp`.

## Root Cause Fixed

The bug was in:

- [ggml-npu.cpp](/home/gugugu/work/llama.cpp-kv260-20260407/ggml/src/ggml-npu/ggml-npu.cpp)

Before the fix:

- `npu_buffer_memset_tensor`
- `npu_buffer_set_tensor`
- `npu_buffer_get_tensor`

used `buffer_base + offset` instead of `tensor->data + offset`.

Impact:

- tensors allocated from one shared backend buffer were read/written as if every tensor started at offset 0
- loader-side `clip_aicas_fuse_bias_compensation()` uses `ggml_backend_tensor_get/set()`
- so `precomp` fused the wrong memory region and semantic output collapsed

Validation:

- host-native test `test-npu-buffer-io` now passes and covers:
  - non-zero-offset tensor writes
  - view tensor writes
  - partial memset
- board-side `precomp` one-sample OCR is readable again
- board-side full V2 `acc5` returns `3/5`

## Extra Stability Fix Included

Runtime DMA launch no longer hard-fails immediately on residual busy bits.
Before launching MVIN/MVOUT async, it now waits for that DMA channel to become idle.

This fixed the observed board-side crash:

```text
DMA1 MVIN is busy
```

## Reproduction Commands

### 1. Build

```bash
cmake --build build-kv260-npu-current --target llama-server -j$(nproc)
```

### 1.1 Board sudo password

Current lab board password:

```bash
export BOARD_SUDO_PASSWORD=123456
```

Use it with overlay/driver init scripts that require sudo on `ubuntu@192.168.0.10`.

### 2. Throughput-only

```bash
bash AICAS/scripts/deploy_and_run_kv260_npu_eval.sh \
  --host 192.168.0.10 \
  --user ubuntu \
  --remote-root /home/ubuntu/aicas \
  --results-dir /home/gugugu/work/llama.cpp-kv260-20260407/AICAS2026/V2/results/official \
  --build-dir /home/gugugu/work/llama.cpp-kv260-20260407/build-kv260-npu-current \
  --model /home/gugugu/work/llama.cpp-kv260-20260407/AICAS/gguf/SmolVLM2-500M-Video-Instruct-Q8_0.gguf \
  --mmproj /home/gugugu/work/llama.cpp-kv260-20260407/AICAS/gguf/mmproj-fallback-search-fb_attn_k-per-tensor.gguf \
  --ocrbench-file /home/gugugu/work/llama.cpp-kv260-20260407/AICAS2026/V1_412/kv260_sampled5.json \
  --run-id 20260417T-auto-default-throughput-only-fixed \
  --skip-build \
  --skip-readiness-probe \
  --throughput-only
```

Result directory:

- [20260417T-auto-default-throughput-only-fixed](/home/gugugu/work/llama.cpp-kv260-20260407/AICAS2026/V2/results/official/20260417T-auto-default-throughput-only-fixed)

### 3. Host-native backend regression

```bash
cmake -B build-host-npu-debug -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NPU=ON \
  -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_CURL=OFF

cmake --build build-host-npu-debug --target test-npu-buffer-io -j$(nproc)

./build-host-npu-debug/bin/test-npu-buffer-io
```

Expected:

```text
test-npu-buffer-io: ok
```

### 4. Explicit `precomp` one-sample OCR

```bash
GGML_NPU_AICAS_BIAS_MODE=precomp \
bash AICAS/scripts/deploy_and_run_kv260_npu_eval.sh \
  --host 192.168.0.10 \
  --user ubuntu \
  --remote-root /home/ubuntu/aicas \
  --results-dir /home/gugugu/work/llama.cpp-kv260-20260407/AICAS2026/V2/results/official \
  --build-dir /home/gugugu/work/llama.cpp-kv260-20260407/build-kv260-npu-current \
  --model /home/gugugu/work/llama.cpp-kv260-20260407/AICAS/gguf/SmolVLM2-500M-Video-Instruct-Q8_0.gguf \
  --mmproj /home/gugugu/work/llama.cpp-kv260-20260407/AICAS/gguf/mmproj-fallback-search-fb_attn_k-per-tensor.gguf \
  --ocrbench-file /home/gugugu/work/llama.cpp-kv260-20260407/AICAS2026/V2/results/official/one_sample_2543.json \
  --run-id 20260417T-precomp-fix-onesample \
  --skip-build \
  --skip-readiness-probe
```

Result directory:

- [20260417T-precomp-fix-onesample](/home/gugugu/work/llama.cpp-kv260-20260407/AICAS2026/V2/results/official/20260417T-precomp-fix-onesample)

### 5. Explicit `precomp` V2 acc5

```bash
GGML_NPU_AICAS_BIAS_MODE=precomp \
bash AICAS2026/V2/scripts/run_v2_eval.sh \
  --build-dir /home/gugugu/work/llama.cpp-kv260-20260407/build-kv260-npu-current \
  --samples 5 \
  --skip-build \
  --skip-readiness-probe \
  --run-id 20260417T-v2-acc5-precomp-fixed
```

Result directory:

- [20260417T-v2-acc5-precomp-fixed](/home/gugugu/work/llama.cpp-kv260-20260407/AICAS2026/V2/results/official/20260417T-v2-acc5-precomp-fixed)

### 6. Default `precomp` V2 acc5

```bash
env -u GGML_NPU_AICAS_BIAS_MODE \
bash AICAS2026/V2/scripts/run_v2_eval.sh \
  --build-dir /home/gugugu/work/llama.cpp-kv260-20260407/build-kv260-npu-current \
  --samples 5 \
  --skip-build \
  --skip-readiness-probe \
  --run-id 20260417T-v2-acc5-default-precomp-fixed
```

Result directory:

- [20260417T-v2-acc5-default-precomp-fixed](/home/gugugu/work/llama.cpp-kv260-20260407/AICAS2026/V2/results/official/20260417T-v2-acc5-default-precomp-fixed)

## Confirmed Results

### Throughput-only

From [throughput_metrics.json](/home/gugugu/work/llama.cpp-kv260-20260407/AICAS2026/V2/results/official/20260417T-auto-default-throughput-only-fixed/throughput_metrics.json):

- prompt tokens: `188`
- completion tokens: `252`
- total tokens: `440`
- prompt ms: `63784.377`
- decode ms: `31896.771`
- total ms: `95681.148`
- prefill speed: `2.95 tok/s`
- decode speed: `7.90 tok/s`

### Explicit `precomp` one-sample

From [SmolVLM2_npu_w8a8.json](/home/gugugu/work/llama.cpp-kv260-20260407/AICAS2026/V2/results/official/20260417T-precomp-fix-onesample/SmolVLM2_npu_w8a8.json):

- final score: `1/1`
- prediction: `The image contains the word "Villa" written in a cursive font.`

### Explicit `precomp` acc5

From [throughput_metrics.json](/home/gugugu/work/llama.cpp-kv260-20260407/AICAS2026/V2/results/official/20260417T-v2-acc5-precomp-fixed/throughput_metrics.json):

- prompt tokens: `188`
- completion tokens: `171`
- total tokens: `359`
- prompt ms: `62422.422`
- decode ms: `21086.481`
- total ms: `83508.903`
- prefill speed: `3.01 tok/s`
- decode speed: `8.11 tok/s`

From [SmolVLM2_npu_w8a8.json](/home/gugugu/work/llama.cpp-kv260-20260407/AICAS2026/V2/results/official/20260417T-v2-acc5-precomp-fixed/SmolVLM2_npu_w8a8.json):

- final score: `3/5`

### Default `precomp` acc5

From [SmolVLM2_npu_w8a8.json](/home/gugugu/work/llama.cpp-kv260-20260407/AICAS2026/V2/results/official/20260417T-v2-acc5-default-precomp-fixed/SmolVLM2_npu_w8a8.json):

- final score: `3/5`

Per-sample summary:

1. `IIIT5K/test/2543_2.png`
   answer=`VILLA`, predict=`Villa`
2. `IIIT5K/test/2174_5.png`
   answer=`SWAD`, predict=`Swad`
3. `IC13_857/imgs/000000229.jpg`
   answer=`students`, predict=`Students`
4. `svt/image/img_0046.jpg`
   answer=`center`, predict=`CHANNEL`
5. `IIIT5K/test/1023_14.png`
   answer=`83KM`, predict=`83 KM`

## One-Sample Bias/Fold Matrix

Fixed input: `IIIT5K/test/2543_2.png`, question: `what is written in the image?`

- `precomp + fold`: correct, readable (`Villa`)
- `raw + fold`: correct, readable
- `raw + nofold`: correct, readable

Conclusion:

- fold-output reconstruction is not the primary blocker
- precomputed bias-compensation semantics are now restored

## Async Copy Status

The current execution path does use runtime DMA async APIs for input staging:

- activation tile via `npu_dma_mvin_async(0, ...)`
- weight tile via `npu_dma_mvin_async(1, ...)`
- then `npu_dma_wait_mvin((1u << 0) | (1u << 1))`

Reference:

- [ggml-npu-exec.cpp](/home/gugugu/work/llama.cpp-kv260-20260407/ggml/src/ggml-npu/ggml-npu-exec.cpp:580)

Important limitation:

- the backend interface itself is still marked non-async
  - `.set_tensor_async = nullptr`
  - `.get_tensor_async = nullptr`
  - `.cpy_tensor_async = nullptr`
  - `.async = false`

So:

- hardware/runtime level: yes, dual-DMA async MVIN is used
- ggml backend API level: no, this backend is not exposed as an async backend
