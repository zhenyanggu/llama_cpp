# 比赛冲刺优化计划

更新日期：2026-04-22

## 1. 前提更新

这版文档按你刚补充的约束重写，几个判断需要先明确：

1. `activation_pack` 和 `postprocess` 已经做过优化，旧的 `2026-04-16` 延迟分解不能再作为当前主判断依据。
2. `32x32 SA -> 16x16 SA` 现在不是“可选优化”，而是主线任务。原因不是理论峰值更高，而是当前版本资源占用过高，设计几乎放不下，同时也不利于时序收敛。
3. `多 DMA` 也应从支线提升到主线。原因不是锦上添花，而是当前对 DRAM 的利用率偏低，按你的观察可能只有约 `3 GB/s`，说明 memory system 还有明显空间。
4. `动态形状` 仍然重要，但建议放到 `instruction control / instruction buffer` 版本之后做，避免重复开发。
5. 量化 ABI 必须先冻结，否则 `16x16`、`多 DMA`、`VPU`、`动态形状`、编译器这些工作都会反复返工。

## 2. 当前判断

### 2.1 当前稳定基线

- 当前稳定提交版本可参考 `AICAS2026/V3/results/official/20260417T092613Z-v3-release/summary_metrics.json`
- 该版本指标：
  - `prompt_ms = 53624.371`
  - `decode_ms = 21224.736`
  - `prefill_tps = 3.506`
  - `decode_tps = 8.057`
  - `acc = 58/100`

### 2.2 当前实现状态

- 软件侧当前默认仍是 `32x32` tile，见 `ggml/src/ggml-npu/ggml-npu-common.h`
- runtime 当前默认仍是 `2 DMA`，见 `npuruntime/kv260/runtime/npu_runtime.h`
- RTL 侧已经有 `inst_ctrl`、`dma`、`VPU`、`systolic_array`、`scratchpad` 的文档基础，可作为后续重构锚点

这意味着：

- `16x16 SA` 不只是 RTL 改一下参数，还会牵动编译器 tile 规则、SPM 映射、runtime 调度和性能评估
- `多 DMA` 也不只是多复制一个 DMA，而是 `SPM banking / crossbar / AXI port / runtime 提交策略` 的联动问题

### 2.3 现在真正需要回答的三个问题

在 `activation_pack/postprocess` 已优化之后，当前主线需要重新围绕下面三个问题组织：

1. 在新的 mmproj/prefill 路径里，热点到底是 `compute`、`DMA`、`wait` 还是残余 CPU 算子？
2. 对硬件而言，当前更硬的约束到底是算力不足，还是 `资源/时序/带宽`？
3. 对比赛成绩而言，当前是否仍然应该优先压 `prefill/mmproj`，还是 `decode/GEMV` 已经抬头到足以值得单独开硬件支线？

### 2.4 更新后的总判断

- `prefill/mmproj` 仍然大概率是主战场，除非你拿到新的拆解证明 `decode` 已经接近它
- 当前更像是“系统级瓶颈”而不是单纯“算子级瓶颈”
- 主线应该从旧的 `pack/postprocess` 转为新的三条主线：
  1. 量化 ABI 冻结
  2. `16x16 SA` 资源/时序收敛
  3. memory system 扩展，包括 `banked SPM + grouped crossbar + 3~4 DMA + AXI 口扩展`

## 3. 对你原计划的逐项评估

| 项目 | 更新后的结论 | 主要原因 | 建议 |
| --- | --- | --- | --- |
| 1. 更优量化策略 | 必做 | 这是所有后续硬件接口的 ABI 基线 | 先冻结 `SmoothQuant` 主线，`AWQ` 作为支线 |
| 2. 动态形状 | 必做，但放在指令版本之后 | prefill/decode 都需要；如果先做静态寄存器版本，后面切指令流容易返工 | 先做 instruction ABI，再做 dynamic shape |
| 3. 改 16x16 PE | 必做，且是当前主线 | 当前资源占用太高，几乎放不下；同时有利于时序收敛 | 立刻启动 `16x16 + 编译器 tiling + 功能验证 + 资源/Fmax 对比` |
| 4. SPM 改为分组 crossbar | 高优先级 | 这是多 DMA、流水线、时序优化的基础设施 | 放到阶段二主线，和多 DMA 并行推进 |
| 5. 指令控制 + 指令 buffer | 高价值 | 是动态形状、流水线、复杂调度的基础设施 | 阶段一先定 ABI，阶段二主线落地 |
| 6. Debug bias 路径 | 立刻做 | 这是正确性问题，不是优化问题 | 阶段一高优先级 |
| 7. per-channel 的 int32-fp32 量化/反量化 | 高价值，但不要卡住主线 | 直接影响精度和更多层卸载；但当前主线先收敛 `per-tensor` 更稳 | 阶段一设计 ABI 预留，阶段二按精度需求落地 |
| 8. 3~4 个 DMA | 高优先级主线 | 当前 DRAM 利用率偏低，双 DMA 基线已经有，继续扩展值得做 | 和 `SPM banking / AXI 口 / runtime 调度` 联动推进 |
| 9. NPU 流水线 | 高风险高收益 | 依赖指令流、banked SPM、多 DMA、hazard 处理 | 放在阶段二后半段 |
| 10. 加 VPU 加速 GEMV | 可以做，但不是第一主线 | 需要先确认比赛模型、量化方式、精度要求和 decode 占比 | 先把 GEMM 主线收敛，再决定是否立项 |
| 11. CPU 侧优化 | 立刻做 | 成本低、回报快、对比赛最友好 | 阶段一必须做 |
| 12. CPU/NPU 流水线并行 | 值得做 | 软件级 overlap 往往比新开硬件支线更快见效 | 先做软件侧 overlap，再考虑更深层并行 |
| 13. 时序提升到 200M | 必做 | 但它不是独立任务，而是 SA、SPM、DMA、inst_ctrl 调整后的结果 | 从阶段一开始绑定推进 |

## 4. 我建议新增的三项任务

这三项不在你原始清单里，但现在优先级很高。

### 4.1 新版 profiling 和带宽测量自动化

旧 profile 已经过时，必须建立新的固定输出。

建议每次实验至少固定产出：

- 端到端：`prompt_ms / decode_ms / acc`
- mmproj/prefill 拆解
- node 级：`pack / dma_in / gemm / dma_out / postprocess / residual_cpu`
- runtime 级：`dma busy / compute busy / wait irq / idle`
- 带宽：`DDR GB/s`、各 `AXI` 口利用率、`SmartConnect` 压力
- SA 利用率：tile 利用率、边角浪费、空泡比例
- Vivado：`LUT / DSP / BRAM / URAM / WNS / TNS / Fmax`

### 4.2 `16x16 SA` 与编译器联动工作包

这件事不能只交给 RTL。

最少要同步修改：

- tile 切分规则
- SPM/ACC 容量假设
- 编译器 codegen
- runtime 描述结构
- 精度和性能回归脚本

如果 `16x16` 只是 RTL 改了，软件栈还按 `32x32` 思路切 tile，后面数据没有意义。

### 4.3 memory system scaling study

既然当前 DRAM 利用率偏低，那就不要只说“上 4 DMA”，而要把路径打透。

建议固定评估：

- `2 DMA / 3 DMA / 4 DMA`
- `2 AXI / 4 AXI`
- grouped crossbar 前后差异
- SmartConnect 开销
- 不同 burst/队列深度
- 在 `16x16` 下的实际带宽需求和满足程度

## 5. 推荐优先级

### 5.1 主线优先级

1. 冻结量化 ABI，先确定 `SmoothQuant` 主线
2. 在当前“已优化 pack/postprocess”的版本上重跑延迟分解和带宽分解
3. 修 `bias` 路径，统一 `raw/precomp/fold-output` 语义
4. 做 CPU 侧 quick win
5. 推进 `16x16 SA + compiler/runtime` 联动适配
6. 定义 instruction ABI 与 compiler/runtime 接口
7. 推进 `SPM banking / grouped crossbar`
8. 推进 `3~4 DMA / 4 AXI 口 / runtime 调度`
9. 在 instruction 版本上支持 dynamic shape
10. 做 NPU pipeline
11. 根据精度压力落地 per-channel `int32-fp32`
12. 只有在 decode 占比和量化方式明确后，再决定是否投入 `VPU/GEMV`
13. 全程绑定 `timing close >= 200 MHz`

### 5.2 当前不建议再作为主线的问题

- 不要再把 `activation_pack` 和 `postprocess` 当成当前主线优化目标
- 不要在量化 ABI 未冻结前大改 RTL
- 不要在没有 `16x16` 编译器适配前只做 RTL 单边实验
- 不要在没有 `SPM banking / AXI` 设计前直接喊“上 4 DMA”
- 不要过早把 `VPU` 定成比赛主线
- 不要同时开 `AWQ`、`VPU`、`深流水线` 三条高风险支线

## 6. 建议的阶段安排

### 6.1 阶段一：到 2026-05-01

目标：冻结 ABI，拿到新的真实瓶颈，启动必须做的主线分支，但避免大返工。

### 6.1.1 必须完成

1. 完成全量化 `SQ mmproj` 测试，输出精度、时延、延迟分解、带宽分解
2. 冻结量化 ABI v1
3. Debug `bias` 路径，统一 `raw/precomp/fold-output` 语义
4. 跑一轮 CPU 侧 quick win
5. 起 `16x16` 版本，连同编译器 tile 规则一起做功能验证
6. 写清 instruction ABI 最小版本，不要求 RTL 全部落地
7. 明确 `decode GEMV` 的需求边界，确认 VPU 是否真的值得进主线

### 6.1.2 阶段一的量化 ABI 建议

- 激活：CPU 侧完成 `fp32 -> int8/uint8`
- `SmoothQuant` 的 `x / s` 逻辑仍在 CPU 侧完成
- 权重：先 `per-tensor`
- 输出重建：先支持 `int32 -> fp32`
- ABI 层面预留 `per-channel` 字段，但不要求阶段一全部落地

### 6.1.3 阶段一的验收标准

- 同一模型/同一量化文件，软件、编译器、RTL 的量化语义完全一致
- 至少拿到一份新的 `mmproj` profile 报告，且不再依赖旧的 `2026-04-16` 结论
- 至少拿到一份新的带宽报告，能回答当前 DDR 利用率到底是多少
- `16x16` 版本至少完成功能打通，并拿到初版资源/Fmax 数据
- 能明确回答当前主瓶颈更偏 `compute`、`bandwidth` 还是 `resource/timing`

### 6.2 阶段二：2026-05-06 到 2026-05-20

目标：围绕 `16x16 + instruction + memory system` 做架构收敛。

### 6.2.1 建议主线

1. 完成 `16x16 SA` 的 RTL、编译器、runtime 适配
2. SPM 改成 `grouped/banked crossbar`，同时做寄存器插入
3. 实现 `3~4 DMA`，目标是吃满带宽并降低 `AXI SmartConnect` 开销
4. 实现指令控制方式，加入 instruction buffer、取指、译码
5. 在 instruction 版本上支持 dynamic shape
6. 做 NPU pipeline
7. 按精度需求落地 per-channel `int32-fp32`
8. 只有在收益明确时，再做 VPU prototype

### 6.2.2 阶段二的顺序建议

建议拆成两条并行主线：

主线 A，算力/控制：

1. `16x16 SA`
2. `instruction ABI`
3. `dynamic shape`
4. `pipeline`

主线 B，存储/带宽/时序：

1. `SPM banking / grouped crossbar`
2. `multi-DMA`
3. `AXI 口扩展`
4. `timing close`

原因：

- `16x16` 会改变 tile 和 buffer 假设，不应放到很后面
- 没有 instruction ABI，动态形状和流水线都不稳
- 没有 banked SPM，多 DMA 很容易互相打架
- 当前既然怀疑 DRAM 利用率低，就必须尽快建立带宽主线，而不是放到最后

### 6.3 阶段三：2026-05-20 到 2026-05-28

目标：只保留已经证明有效的优化，做最后收敛和提交。

### 6.3.1 建议主线

1. 锁定最终架构配置，不再引入新的大支线
2. 做 timing close，目标 `>= 200 MHz`
3. 做端到端精度和吞吐回归
4. 完成提交物整理
5. 如果还有余力，再评估 `VPU GEMV` 或更深层 CPU/NPU 并行

### 6.3.2 阶段三不建议做的事

- 临近提交时切换量化主线
- 临近提交时重新改 SA 阵列规模
- 临近提交时重写编译器调度
- 临近提交时新开 AWQ 主线

## 7. 推荐的并行分工

### 7.1 1 个人

按这条线做：

1. 量化 ABI 冻结
2. 新 profile 和带宽测量
3. bias 正确性
4. CPU quick win
5. `16x16 + 编译器` 打通
6. instruction ABI

不要同时开 `AWQ`、`VPU`、`深流水线` 三条支线。

### 7.2 2 个人

角色 A：软件/量化/验证

- `SmoothQuant` 主线
- 精度回归
- `bias` 路径
- CPU quick win
- 新版 profiling

角色 B：RTL/架构

- `16x16 SA`
- `SPM banking / crossbar`
- `multi-DMA`
- `timing close`

### 7.3 3 个人

角色 A：量化与精度

- `SQ` 主线
- per-channel
- `AWQ` 支线预研

角色 B：runtime/compiler

- ggml/runtime/profile
- `16x16` tile 适配
- instruction ABI
- dynamic shape

角色 C：RTL

- `16x16 SA`
- `inst_ctrl`
- `SPM/crossbar`
- `DMA`
- `timing`

### 7.4 4 个人及以上

再加一条实验支线：

角色 D：实验特性

- `VPU/GEMV` feasibility
- CPU/NPU pipeline
- 更深的 scheduler overlap
- `AWQ` 或其他激进方案

## 8. 真正的性能瓶颈应该怎么确认

不要再靠感觉判断，固定看下面六类数据：

1. 端到端：`prompt_ms / decode_ms / acc`
2. mmproj/prefill 拆解：图像侧、文本侧、残余 CPU 占比
3. node/runtime profile：`pack / dma_in / gemm / dma_out / postprocess / wait`
4. 带宽：`DDR GB/s`、`AXI` 口利用率、`SmartConnect` 压力
5. SA 利用率：tile 利用率、边角浪费、空泡比例
6. Vivado：`LUT / DSP / BRAM / URAM / WNS / TNS / Fmax`

判断规则建议直接定死：

- 如果 `DDR GB/s` 明显低于预期，同时大量时间耗在 `wait dma / wait buffer`，那主瓶颈是 memory system
- 如果资源接近打满，或者 `WNS/TNS` 很差，`32x32` 很难关时序，那主瓶颈是架构规模和实现方式
- 如果带宽健康、时序健康，但 `compute busy` 明显最长，那才说明是纯算力瓶颈

## 9. 最终建议

### 9.1 主线

- 先冻结 `SmoothQuant` 主线
- 先拿新的延迟/带宽分解，不再依赖旧 profile
- 把 `16x16 SA` 当成必须收敛的主线，而不是实验项
- 把 `multi-DMA` 当成 memory system 主线，而不是后续锦上添花
- 先完成 instruction ABI，再做 dynamic shape
- 全程围绕 `>= 200 MHz` 做结构取舍

### 9.2 支线

- `AWQ`
- `VPU GEMV`
- 更激进的 CPU/NPU 深流水并行

### 9.3 一句话版本

当前最应该做的已经不是继续盯 `activation_pack/postprocess`，而是“先冻结量化 ABI，再围绕 `16x16 SA` 和 `memory system` 做架构收敛”，同时用新的 profile 把真实瓶颈重新量出来。
