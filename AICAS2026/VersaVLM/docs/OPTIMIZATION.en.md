# Optimization Method Notes

## 1. Background and Problem

For the target VLM-style multimodal model, the `prefill` stage is becoming the main latency bottleneck.

There are two main reasons:

1. The image-side `mmproj` path contains a large amount of matrix multiplication and takes a significant share of runtime.
2. After visual encoding and projection, the image path produces many visual tokens, which further increases the subsequent `prefill` workload and reduces effective prefill throughput.

Therefore, for this class of models, reducing `mmproj` latency is more important than optimizing only the text decoding path.

## 2. Overall Approach

The optimization strategy is:

- accelerate the dominant `mulmat` operators inside `mmproj`
- use an NPU based on a systolic array for the main matrix multiplications
- quantize part of the precision-insensitive `mulmat` layers under an accuracy-oriented policy
- offload those quantized operators to the NPU, while the CPU handles control, synchronization, and a small amount of post-processing

## 3. Why NPU Acceleration for mmproj

`mmproj` is dominated by linear layers, which are essentially dense matrix multiplications.

These operators have the following properties:

- large MAC count
- regular data access pattern
- stable operator structure
- good suitability for static scheduling and tiled execution

A systolic-array-based NPU is efficient for this type of `mulmat` workload and can improve throughput while reducing CPU-side compute pressure.

## 4. Quantization Scheme

The default GGUF quantization formats are more CPU-oriented and are not ideal for the current NPU data path and compute format. For that reason, we did not directly reuse the default GGUF quantization scheme for the NPU path.

Instead, we use a hardware-friendly, accuracy-oriented scheme:

- activations: static asymmetric quantization
- weights: static symmetric per-tensor quantization
- data format: `W8A8`

Why this scheme was chosen:

- asymmetric activation quantization better matches the distribution of visual features and reduces zero-point related error
- symmetric per-tensor weight quantization is simple, hardware-efficient, and easy to implement on the current NPU
- `W8A8` provides a practical balance between accuracy and hardware cost in the current system

## 5. NPU Execution Flow

The current hardware/software execution flow is:

1. the CPU performs graph execution, task partitioning, and NPU scheduling
2. the CPU programs the NPU through MMIO registers, including dimensions, addresses, scales, and zero-points
3. input, weight, and output buffers are placed in CMA so that the NPU DMA engines can access them directly
4. the NPU DMA moves data from CMA to on-chip buffers
5. the systolic array executes the main `mulmat` computation
6. the result is dequantized / post-processed and written back to CMA
7. the CPU continues the remaining graph execution when needed

In this design, the CPU acts mainly as the controller instead of executing the main matrix multiplication itself.

## 6. Key Optimizations Already Implemented

### 6.1 Partial mmproj mulmat offload to the NPU

A subset of precision-insensitive linear layers inside `mmproj` was statically quantized to `W8A8` and offloaded to the NPU.

This is the foundation of the current throughput improvement.

### 6.2 NPU-oriented quantization data path

To match the current NPU compute path, a dedicated quantization / dequantization flow was implemented instead of relying on the default GGUF layout. This includes:

- static activation scale / zero-point handling
- static per-tensor weight scale handling
- corresponding bias, compensation, and output reconstruction logic

### 6.3 int32-to-fp32 dequantization unit

To reduce CPU-side post-processing overhead, an `int32 -> fp32` dequantization unit was added to the NPU result path.

Its purpose is to:

- reduce per-element reconstruction work on the CPU
- shorten the post-processing path
- improve overall `mmproj` completion time

### 6.4 Dual-DMA / multi-DMA parallel transfer

To reduce transfer overhead, multi-DMA parallel transfer was introduced so that input and intermediate data movement can overlap better with computation.

The goal is to reduce time spent in:

- `mvin`
- `mvout`
- idle waiting for on-chip buffer availability

### 6.5 Reduced host-side extra copies

During optimization, unnecessary host-side copies were reduced or restructured in order to avoid repeating expensive memory movement during every inference.

Examples include:

- reducing repeated weight copies
- reusing NPU-accessible buffers whenever possible
- shortening the CPU-side post-processing chain

### 6.6 Tiling and operator-level scheduling

To fit on-chip storage limits and DMA granularity, `mulmat` is executed in tiles, with runtime parameters tuned to the hardware capacity, including:

- SPM usage
- accumulator buffer usage
- guard region size
- stage-2 K partition settings

These parameters directly affect throughput, on-chip buffer utilization, and runtime stability.

### 6.7 Graph fusion and offload planning in software

On the software side, graph patterns are recognized and fused to reduce unnecessary host-side operations and operator switching overhead. Examples include:

- fusion detection for `ADD(MUL_MAT, bias)`
- avoiding duplicate offload of a fused `ADD` and its child `MUL_MAT`
- selecting different execution modes based on bias, compensation, and scale requirements

## 7. Current Benefit

The current version mainly improves the system in the following ways:

- the dominant `mulmat` hotspots in `mmproj` are moved from the CPU to the NPU
- prefill throughput is measurably improved
- end-to-end latency under multimodal input is reduced
- the system is closer to a practically usable state for the competition workload

This also confirms an important point:

- for VLM workloads, optimizing only text decoding is not sufficient
- the image-side `mmproj` path and visual-token expansion must be addressed together

## 8. Current Limitations

Although the current version already accelerates part of the `mmproj` path effectively, several limitations remain:

1. only part of the layers are currently offloaded, and a substantial amount of work still runs on the CPU
2. the current quantization scheme is still conservative, prioritizing usable accuracy before broader offload coverage
3. the decoding stage is still dominated by CPU-side `GEMV`, so decode throughput has not yet been specifically optimized
4. some post-processing and data organization overhead still remains
5. weight quantization is still primarily per-tensor, leaving room for future improvement

## 9. Future Optimization Targets

### 9.1 Better quantization to offload more layers to the NPU

Future work will continue exploring data representations that are better suited to both the model and the hardware, with goals including:

- better accuracy retention
- lower accumulated quantization error
- wider operator offload coverage
- higher NPU utilization

Possible directions include:

- finer-grained weight quantization
- more stable activation calibration
- layer-wise / channel-wise mixed strategies
- scale / zero-point layouts that map better to the hardware pipeline

### 9.2 A dedicated GEMV unit for decoding

The main hotspot in the `decode` stage is closer to `GEMV` than `GEMM`.

Future work will therefore consider adding a `GEMV` unit to accelerate:

- token-by-token decoding linear layers
- matrix-vector operations under small-batch, low-parallelism conditions

This is expected to be a key direction for improving decode throughput.

### 9.3 More overlap between transfer and computation

Further work will continue on:

- improved multi-DMA scheduling
- better transfer / compute overlap
- better ping-pong and double-buffer mechanisms
- lower host synchronization frequency

### 9.4 Lower host-side post-processing cost

Although the `int32 -> fp32` dequantization unit already reduces part of the CPU burden, there is still room for further improvement. Future directions include:

- completing more output reconstruction logic inside the NPU
- reducing direct CMA reads and writes on the CPU side
- reducing scalar CPU-side post-processing
- improving vectorization and cache-friendly implementations

### 9.5 Better graph-level fusion and execution planning

The software stack can be further improved through:

- more pattern fusion
- layer grouping
- better offload planning
- fewer redundant quantize / dequantize / intermediate format conversions

## 10. Summary

The NPU used in this work is itself a general-purpose NPU and had already been able to run traditional CNN models effectively. The focus of the current work is to extend that general hardware/software stack toward VLM and LLM workloads through software-stack changes, quantization support, and execution-path optimization.

The overall path can be summarized as follows:

- identify `mmproj` as the key bottleneck in `prefill`
- accelerate the dominant `mulmat` operators with a systolic-array-based NPU
- use a static `W8A8` quantization scheme oriented toward both accuracy and hardware feasibility
- reduce system-level overhead through DMA optimization, a dequantization unit, tiling, and execution planning
- gradually expand NPU offload coverage while keeping the result usable

Future progress will continue along three main directions: broader layer offload, stronger decode acceleration, and lower overall system overhead.
