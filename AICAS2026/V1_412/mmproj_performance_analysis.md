# KV260 mmproj Performance Analysis

## Summary

- Board: `ubuntu@192.168.0.10`
- Model: `/home/ubuntu/aicas/shared/gguf/SmolVLM2-500M-Video-Instruct-f16.gguf`
- mmproj: `/home/ubuntu/aicas/shared/gguf/mmproj-SmolVLM2-500M-Video-Instruct-w8a8-mixed-v1.gguf`
- Image: `/home/ubuntu/aicas/shared/data/IIIT5K/test/2543_2.png`
- Threads: `4`
- mmproj total profiled time: `1720963.561 ms`
- NPU-covered node trace time: `135323.411 ms` (`7.86%` of mmproj replay)
- Runtime NPU total: `135077.191 ms`
- Hottest NPU layer: `ffn_down` / `node_337` at `3258.667 ms`
- Hottest ggml operator: `MUL_MAT` at `613153.291 ms`
- Hottest ggml node: `node_403` at `25447.102 ms`

## NPU Stage Breakdown

| Stage | Time (ms) |
| --- | --- |
| Runtime DMA In | 6136.353 |
| Runtime Compute | 1442.492 |
| Runtime DMA Out | 674.321 |
| Runtime Layout | 0.000 |
| Runtime Wait IRQ | 7431.568 |
| Host Pack/Copy/Post | 125638.624 |
| ggml-npu GEMM Call | 1648.123 |

## Hot NPU Layers

| Layer | Semantic | M/N/K | Tiles | Node ms | Runtime ms | DMA In ms | Compute ms | Wait IRQ ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 79 | ffn_down | 3072/1024/768 | 3072 | 3258.667 | 3258.648 | 146.078 | 34.535 | 176.865 |
| 71 | ffn_down | 3072/1024/768 | 3072 | 3258.191 | 3258.171 | 146.153 | 34.396 | 176.713 |
| 95 | ffn_down | 3072/1024/768 | 3072 | 3258.056 | 3258.015 | 146.394 | 34.445 | 176.754 |
| 87 | ffn_down | 3072/1024/768 | 3072 | 3257.224 | 3257.204 | 146.092 | 34.480 | 176.781 |
| 47 | ffn_down | 3072/1024/768 | 3072 | 3257.108 | 3257.089 | 146.156 | 34.276 | 176.771 |
| 39 | ffn_down | 3072/1024/768 | 3072 | 3256.550 | 3256.530 | 145.947 | 34.446 | 176.823 |
| 31 | ffn_down | 3072/1024/768 | 3072 | 3256.405 | 3256.384 | 146.017 | 34.271 | 176.875 |
| 23 | ffn_down | 3072/1024/768 | 3072 | 3255.984 | 3255.963 | 145.867 | 34.393 | 176.886 |
| 63 | ffn_down | 3072/1024/768 | 3072 | 3255.601 | 3255.581 | 146.179 | 34.341 | 176.879 |
| 55 | ffn_down | 3072/1024/768 | 3072 | 3255.575 | 3255.554 | 146.196 | 34.432 | 176.975 |
| 15 | ffn_down | 3072/1024/768 | 3072 | 3253.944 | 3253.922 | 145.668 | 34.177 | 176.715 |
| 7 | ffn_down | 3072/1024/768 | 3072 | 3253.125 | 3253.104 | 145.515 | 34.013 | 176.805 |
| 78 | ffn_down | 3072/1024/768 | 3072 | 3188.248 | 3188.228 | 146.224 | 34.503 | 177.014 |
| 6 | ffn_down | 3072/1024/768 | 3072 | 3185.249 | 3185.204 | 145.668 | 34.102 | 176.905 |
| 70 | ffn_down | 3072/1024/768 | 3072 | 3184.306 | 3184.286 | 146.216 | 34.473 | 176.864 |

## Semantic Op Summary

| Semantic Op | Layers | Node ms | Runtime ms | Runtime Compute ms | Examples |
| --- | --- | --- | --- | --- | --- |
| ffn_down | 24 | 77279.687 | 77279.144 | 824.634 | node_337, node_304, node_403 |
| attn_q | 24 | 19512.640 | 19267.962 | 205.833 | v.blk.0.attn_q.weight, node_82, node_214 |
| attn_k | 24 | 19269.017 | 19268.515 | 205.939 | node_349, node_382, node_151 |
| attn_v | 24 | 19262.067 | 19261.570 | 206.087 | node_187, node_385, node_253 |

## ggml Operator Summary

| Operator | Time ms | Share % | Events |
| --- | --- | --- | --- |
| MUL_MAT | 613153.291 | 35.63 | 98 |
| ADD | 595007.606 | 34.57 | 123 |
| RESHAPE | 382058.223 | 22.20 | 42 |
| PERMUTE | 129184.306 | 7.51 | 51 |
| SOFT_MAX | 924.995 | 0.05 | 12 |
| GELU | 336.109 | 0.02 | 12 |
| CONT | 193.573 | 0.01 | 28 |
| NORM | 70.508 | 0.00 | 25 |
| MUL | 26.425 | 0.00 | 25 |
| IM2COL | 8.520 | 0.00 | 1 |
| TRANSPOSE | 0.005 | 0.00 | 1 |

## ggml Hot Nodes

| Node | Operator | Time ms | Share % |
| --- | --- | --- | --- |
| node_403 | ADD | 25447.102 | 1.48 |
| v.blk.11.ffn_down.weight | MUL_MAT | 24903.290 | 1.45 |
| node_370 | ADD | 23626.091 | 1.37 |
| v.blk.10.ffn_down.weight | MUL_MAT | 23267.174 | 1.35 |
| node_382 | ADD | 22616.447 | 1.31 |
| node_337 | ADD | 22516.793 | 1.31 |
| node_379 | ADD | 22030.273 | 1.28 |
| node_385 | ADD | 21939.713 | 1.27 |
| v.blk.11.attn_k.weight | MUL_MAT | 21687.062 | 1.26 |
| v.blk.11.attn_v.weight | MUL_MAT | 21494.102 | 1.25 |
| v.blk.11.attn_q.weight | MUL_MAT | 21177.866 | 1.23 |
|  (reshaped) | RESHAPE | 21125.267 | 1.23 |

## Non-NPU Hot Nodes

| Node | Operator | Time ms | Share % |
| --- | --- | --- | --- |
|  (reshaped) | RESHAPE | 21125.267 | 1.23 |
|  (reshaped) (permuted) | PERMUTE | 20786.649 | 1.21 |
|  (reshaped) | RESHAPE | 20771.508 | 1.21 |
|  (reshaped) | RESHAPE | 20320.740 | 1.18 |
|  (reshaped) | RESHAPE | 20104.922 | 1.17 |
|  (reshaped) (permuted) | PERMUTE | 19189.068 | 1.12 |

## Notes

- `mmproj_operator` totals come from ggml eval-callback replay of the full image-encode graph.
- `npu_node_trace` and `npu_runtime_profile` only cover the NPU-offloaded subset, so totals are expected to be smaller.
- Runtime `wait_irq` is counted separately and is useful for understanding hardware-side stall/latency not visible from the ggml wall-clock split.
- Host `pack/copy/postprocess` comes from `ggml-npu` user-space instrumentation, not from the runtime JSON.
