# 编译器设置

# 基础编译选项：C++17标准，显示所有警告，pthread支持(通常runtime需要)
# CXXFLAGS := -std=c++17 -Wall -pthread

# 源文件定义
RUNTIME_SRC := ../npu_runtime.cpp
RUNTIME_HDR := ../npu_runtime.h ../npu_regs_compat.h ../npu_regs.h

# 目标程序列表
# PREFIX := zynq7020
PREFIX := zcu102
TARGETS := $(addprefix $(PREFIX)_,gelu layer_norm softmax dma_accurate dma_delay dma_32bit dma_mvin_repro dma_strided_readback vgg16_fc_repro resnet50_stage4_repro gemm gemm_debug latency matadd matadd_256x128 transpose resample layout conv)

# ==========================================
# 模式配置 (Debug vs Release)
# ==========================================

# 默认模式为 release，命令行可以通过 make MODE=debug 覆盖
# MODE ?= debug
MODE ?= release
# 是否启用影子寄存器 (默认)
DNPU_SHADOW_REGS=1
CXXFLAGS += -DNPU_SHADOW_REGS=$(DNPU_SHADOW_REGS)

# CPU 位宽配置：32 或 64（默认 64）
CPU_WIDTH ?= 64

ifeq ($(MODE), debug)
    # Debug模式: 
    # -g: 生成调试符号
    # -O0: 关闭优化，方便调试
    # -DNPU_DEBUG: 定义宏，启用 runtime 中的详细打印
    CXXFLAGS += -DNPU_DEBUG
    BUILD_INFO := [Debug Mode] - Detailed logging enabled, Optimization OFF
else
    # Release模式:
    # -O3: 最高级别优化，性能最佳
    # (不定义 NPU_DEBUG，代码中的日志宏会被编译为空)
    CXXFLAGS += -O3
    BUILD_INFO := [Release Mode] - Best performance, Logging OFF
endif

# 传递 CPU 位宽宏到编译器
CXXFLAGS += -DNPU_CPU_WIDTH=$(CPU_WIDTH)

# Profiling 开关 (NPU_PROFILE)
# 用法: make PROFILE=1
PROFILE ?= 0
ifeq ($(PROFILE), 1)
	CXXFLAGS += -DNPU_PROFILE
endif

# ==========================================
# Hybrid Polling 参数
# ==========================================

# 自旋轮询次数
NPU_POLL_SPIN_COUNT ?= 2000
# 让出轮询次数
NPU_POLL_YIELD_COUNT ?= 20
# 每次让出的微秒数
NPU_POLL_YIELD_US ?= 1

# 传递轮询参数到编译器
CXXFLAGS += -DNPU_POLL_SPIN_COUNT=$(NPU_POLL_SPIN_COUNT)
CXXFLAGS += -DNPU_POLL_YIELD_COUNT=$(NPU_POLL_YIELD_COUNT)
CXXFLAGS += -DNPU_POLL_YIELD_US=$(NPU_POLL_YIELD_US)

# ==========================================
# 编译规则
# ==========================================

.PHONY: all clean help

# 默认目标
all: info $(TARGETS)

# 打印当前编译模式信息
info:
	@echo "--------------------------------------------------"
	@echo "$(BUILD_INFO)"
	@echo "--------------------------------------------------"

# 具体目标的编译规则
# $^ 表示所有依赖文件，$@ 表示目标文件名

$(PREFIX)_gelu: gelu_test.cpp $(RUNTIME_SRC) $(RUNTIME_HDR)
	$(CXX) $(CXXFLAGS) gelu_test.cpp $(RUNTIME_SRC) -o $@

$(PREFIX)_layer_norm: layernorm_test.cpp $(RUNTIME_SRC) $(RUNTIME_HDR)
	$(CXX) $(CXXFLAGS) layernorm_test.cpp $(RUNTIME_SRC) -o $@

$(PREFIX)_softmax: softmax_test.cpp $(RUNTIME_SRC) $(RUNTIME_HDR)
	$(CXX) $(CXXFLAGS) softmax_test.cpp $(RUNTIME_SRC) -o $@

$(PREFIX)_dma_accurate: dma_accurate_test.cpp $(RUNTIME_SRC) $(RUNTIME_HDR)
	$(CXX) $(CXXFLAGS) dma_accurate_test.cpp $(RUNTIME_SRC) -o $@	

$(PREFIX)_dma_delay: dma_delay_test.cpp $(RUNTIME_SRC) $(RUNTIME_HDR)
	$(CXX) $(CXXFLAGS) dma_delay_test.cpp $(RUNTIME_SRC) -o $@

$(PREFIX)_dma_32bit: dma_32bit_test.cpp $(RUNTIME_SRC) $(RUNTIME_HDR)
	$(CXX) $(CXXFLAGS) dma_32bit_test.cpp $(RUNTIME_SRC) -o $@

$(PREFIX)_dma_mvin_repro: mvin_repro_test.cpp $(RUNTIME_SRC) $(RUNTIME_HDR)
	$(CXX) $(CXXFLAGS) mvin_repro_test.cpp $(RUNTIME_SRC) -o $@

$(PREFIX)_dma_strided_readback: ../../../model_test/runtime/zcu102_dma_strided_readback.cpp $(RUNTIME_SRC) $(RUNTIME_HDR)
	$(CXX) $(CXXFLAGS) ../../../model_test/runtime/zcu102_dma_strided_readback.cpp $(RUNTIME_SRC) -o $@

$(PREFIX)_vgg16_fc_repro: ../../../model_test/runtime/zcu102_vgg16_fc_repro.cpp $(RUNTIME_SRC) $(RUNTIME_HDR)
	$(CXX) $(CXXFLAGS) ../../../model_test/runtime/zcu102_vgg16_fc_repro.cpp $(RUNTIME_SRC) -o $@

$(PREFIX)_resnet50_stage4_repro: ../../../model_test/runtime/zcu102_resnet50_stage4_repro.cpp $(RUNTIME_SRC) $(RUNTIME_HDR)
	$(CXX) $(CXXFLAGS) ../../../model_test/runtime/zcu102_resnet50_stage4_repro.cpp $(RUNTIME_SRC) -o $@

$(PREFIX)_gemm: gemm_test.cpp $(RUNTIME_SRC) $(RUNTIME_HDR)
	$(CXX) $(CXXFLAGS) gemm_test.cpp $(RUNTIME_SRC) -o $@

$(PREFIX)_gemm_debug: gemm_debug.cpp $(RUNTIME_SRC) $(RUNTIME_HDR)
	$(CXX) $(CXXFLAGS) gemm_debug.cpp $(RUNTIME_SRC) -o $@

$(PREFIX)_latency: runtime_latency_test.cpp $(RUNTIME_SRC) $(RUNTIME_HDR)
	$(CXX) $(CXXFLAGS) runtime_latency_test.cpp $(RUNTIME_SRC) -o $@

$(PREFIX)_matadd: matadd_test.cpp $(RUNTIME_SRC) $(RUNTIME_HDR)
	$(CXX) $(CXXFLAGS) matadd_test.cpp $(RUNTIME_SRC) -o $@

$(PREFIX)_matadd_256x128: matadd_256x128_test.cpp $(RUNTIME_SRC) $(RUNTIME_HDR)
	$(CXX) $(CXXFLAGS) matadd_256x128_test.cpp $(RUNTIME_SRC) -o $@

$(PREFIX)_transpose: transpose_test.cpp $(RUNTIME_SRC) $(RUNTIME_HDR)
	$(CXX) $(CXXFLAGS) transpose_test.cpp $(RUNTIME_SRC) -o $@

$(PREFIX)_resample: resample_test.cpp $(RUNTIME_SRC) $(RUNTIME_HDR)
	$(CXX) $(CXXFLAGS) resample_test.cpp $(RUNTIME_SRC) -o $@

$(PREFIX)_layout: layout_convert_test.cpp $(RUNTIME_SRC) $(RUNTIME_HDR)
	$(CXX) $(CXXFLAGS) layout_convert_test.cpp $(RUNTIME_SRC) -o $@

$(PREFIX)_conv: conv_test.cpp $(RUNTIME_SRC) $(RUNTIME_HDR)
	$(CXX) $(CXXFLAGS) conv_test.cpp $(RUNTIME_SRC) -o $@

# 清理规则
clean:
	rm -f $(TARGETS)
	@echo "Cleaned all targets."

# 帮助信息
help:
	@echo "Usage:"
	@echo "  make               Build in Release mode (default)"
	@echo "  make MODE=debug    Build in Debug mode (with logs)"
	@echo "  make clean         Remove generated executables"