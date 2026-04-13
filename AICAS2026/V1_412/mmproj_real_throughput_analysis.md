# mmproj Real Throughput Performance Analysis

## 1. Scope and Method

- Real throughput path: `llama-server + throughput_eval.py`, same image / model / mmproj / threads=`4`.
- Real CPU/NPU wall-clock comes from:
  - `mtmd_prefill_summary_cpu.json`
  - `mtmd_prefill_summary_npu.json`
  - `manual-throughput-cpu-8093.json`
  - `manual-throughput-npu-8092.json`
- CPU time of NPU-offloaded layers is an estimate:
  - board-side `mmproj_cpu_raw_operator.json` operator shares are scaled to the real CPU `mmproj.encode_us`.
  - on CPU this path appears as `MAP_CUSTOM3 + ADD`, not `MUL_MAT + ADD`.
- Semantic split of NPU-offloaded layers is also an estimate:
  - unique fused NPU nodes are taken from `mmproj_raw_npu_node_trace.json` with `root_op_name == ADD`,
  - then scaled to the real `npu_total_node_us`.
- Therefore:
  - total real wall time is exact;
  - per-semantic breakdown is a calibrated estimate.

## 2. Real Throughput Result

| Metric | CPU (ms) | NPU (ms) | Delta (NPU-CPU, ms) |
| --- | ---: | ---: | ---: |
| Prefill total | 58841.312 | 115182.503 | 56341.191 |
| mmproj encode | 43879.914 | 100174.206 | 56294.292 |
| mmproj decode-to-text-embd | 4897.558 | 4996.479 | 98.921 |
| Prefill minus mmproj encode | 14961.398 | 15008.297 | 46.899 |

结论：真实 prefill 的 `56.341 s` 差值里，`56.294 s` 来自 `mmproj encode`。  
`mmproj decode` 和其余 text prefill 基本没有差异。

## 3. NPU-Offloaded Layers: CPU Should-Be vs NPU Real

NPU 实际卸载层总 wall time: `68168.161 ms`。  
CPU 版本对应层估算总时间: `11674.490 ms`。  
放大约 `5.839x`。

| Semantic op | Layers | CPU est (ms) | NPU real (ms) | Delta (ms) | Slowdown | NPU pack (ms) | NPU copy (ms) | NPU DMA in (ms) | NPU GEMM (ms) | NPU DMA out (ms) | NPU post (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| attn_q | 12 | 1622.038 | 9724.408 | 8102.370 | 5.995x | 8274.809 | 132.436 | 469.094 | 117.416 | 62.470 | 643.912 |
| attn_k | 12 | 1622.511 | 9720.809 | 8098.298 | 5.991x | 8273.060 | 131.710 | 468.522 | 117.440 | 62.297 | 643.484 |
| attn_v | 12 | 1623.910 | 9718.658 | 8094.748 | 5.985x | 8271.853 | 131.208 | 468.799 | 117.609 | 62.237 | 642.536 |
| ffn_down | 12 | 6806.031 | 39004.286 | 32198.255 | 5.731x | 33074.124 | 637.056 | 1875.424 | 469.998 | 249.153 | 2601.819 |
| **Total** | **48** | **11674.490** | **68168.161** | **56493.671** | **5.839x** |  |  |  |  |  |  |

代表性单层平均值：

- `attn_q/k/v`：CPU 每层约 `135 ms`，NPU 每层约 `809~810 ms`
- `ffn_down`：CPU 每层约 `567.169 ms`，NPU 每层约 `3250.357 ms`

## 4. What NPU-Offloaded Time Is Made Of

### 4.1 User-space wall-time decomposition of offloaded layers

| Stage | Time (ms) | Share of offloaded wall time |
| --- | ---: | ---: |
| Activation pack / quantize | 57893.845 | 84.93% |
| Host copy | 1032.410 | 1.51% |
| DMA in (user-space timed) | 3281.840 | 4.81% |
| GEMM call (user-space timed) | 822.463 | 1.21% |
| DMA out (user-space timed) | 436.157 | 0.64% |
| Postprocess / dequant+bias+writeback | 4531.750 | 6.65% |
| Other control overhead | 169.696 | 0.25% |

结论：在真实 throughput 口径下，NPU 卸载层的最大开销仍然是 host 侧 `activation pack / 量化`，约 `57.894 s`，占卸载层 wall time 的 `84.93%`。真正的 `GEMM` 只有约 `0.822 s`。

### 4.2 Runtime internal stage split

| Runtime stage | Time (ms) |
| --- | ---: |
| DMA in | 3029.437 |
| Compute | 703.869 |
| DMA out | 320.265 |
| wait_irq | 3743.142 |

`runtime_*` 是 runtime 内部阶段子计时，和上面的 user-space wall-time 不是同一统计口径，不能直接相加。  
它只说明在 runtime 内部，`wait_irq` 比 `compute` 更重。

## 5. Did Remaining CPU Layers Regress?

| Item | Time (ms) |
| --- | ---: |
| CPU-only non-offloaded mmproj layers (estimated) | 32205.424 |
| NPU run residual CPU mmproj time (real) | 32006.045 |
| Delta | -199.379 |

结论：没有看到明显退化。  
`32.205 s` vs `32.006 s` 的差异只有约 `-0.199 s`，量级不到 `1%`。

换句话说，NPU 版本慢，不是因为剩余 CPU 层被拖慢了；主要就是被卸载的那 48 个语义层本身变慢了。

## 6. Final Conclusions

1. 真实 throughput 下，NPU prefill 比 CPU 多 `56.341 s`
2. 这个差值几乎全部来自 `mmproj encode`，而不是 text prefill 主干
3. 被卸载到 NPU 的 48 个语义层，CPU 版本本应约 `11.675 s`，NPU 版实际约 `68.168 s`，放大约 `5.84x`
4. 最大瓶颈是 host 侧 `activation pack / 量化`，不是 NPU compute，也不是剩余 CPU 层退化
5. runtime 内部看，`wait_irq` 也明显高于 `compute`，但它是次级问题；第一优先级仍然是先处理 activation pack
