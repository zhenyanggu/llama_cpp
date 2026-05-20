CXX ?= g++
CC ?= gcc
AR ?= ar

CXXFLAGS ?= -std=c++17 -Wall -Wextra -O2
CXXFLAGS += -DNPU_CPU_WIDTH=64 -DNPU_SHADOW_REGS=1 -I. -I../driver
CFLAGS ?= -std=c11 -Wall -Wextra -O2
CFLAGS += -I../driver

RUNTIME_BACKEND ?= versa_p
RUNTIME_SRC := npu_runtime.cpp
RUNTIME_OBJS := npu_runtime.o
RUNTIME_HDR := npu_runtime.h npu_regs_compat.h npu_regs.h ../driver/npu_kv260_uapi.h

ifeq ($(RUNTIME_BACKEND),versa_p)
VERSA_P_DIR := versa_p
CXXFLAGS += -I$(VERSA_P_DIR)/include -I$(VERSA_P_DIR)/src
RUNTIME_SRC := npu_runtime_versa_p.cpp
RUNTIME_OBJS := npu_runtime.o versa_p_device.o versa_p_mmio.o versa_p_commands.o versa_p_gemm_plan.o
RUNTIME_HDR += \
	$(VERSA_P_DIR)/include/versa_p_error.h \
	$(VERSA_P_DIR)/include/versa_p_regs.h \
	$(VERSA_P_DIR)/include/versa_p_runtime.h \
	$(VERSA_P_DIR)/src/versa_p_internal.h
endif

.PHONY: all clean

all: libnpu_kv260_runtime.a kv260_npu_smoke_test kv260_runtime_init_test kv260_dma_loopback_test kv260_dma_acc_int32_fp32_test kv260_mvin_problem_case_test kv260_dma_double_mvin_async_test kv260_gemm_plan_test kv260_network_gemm_layers_test kv260_layer_gemm_replay_test kv260_mmproj_layer_asym_w8a8_test

npu_runtime.o: $(RUNTIME_SRC) $(RUNTIME_HDR)
	$(CXX) $(CXXFLAGS) -c $(RUNTIME_SRC) -o $@

versa_p_device.o: versa_p/src/device.cpp $(RUNTIME_HDR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

versa_p_mmio.o: versa_p/src/mmio.cpp $(RUNTIME_HDR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

versa_p_commands.o: versa_p/src/commands.cpp $(RUNTIME_HDR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

versa_p_gemm_plan.o: versa_p/src/gemm_plan.cpp $(RUNTIME_HDR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

libnpu_kv260_runtime.a: $(RUNTIME_OBJS)
	$(AR) rcs $@ $^

kv260_npu_smoke_test: smoke_test.c ../driver/npu_kv260_uapi.h
	$(CC) $(CFLAGS) smoke_test.c -o $@ $(LDFLAGS) $(LDLIBS)

kv260_runtime_init_test: runtime_init_test.cpp libnpu_kv260_runtime.a $(RUNTIME_HDR)
	$(CXX) $(CXXFLAGS) runtime_init_test.cpp libnpu_kv260_runtime.a -o $@ $(LDFLAGS) $(LDLIBS)

kv260_dma_loopback_test: dma_loopback_test.cpp libnpu_kv260_runtime.a $(RUNTIME_HDR)
	$(CXX) $(CXXFLAGS) dma_loopback_test.cpp libnpu_kv260_runtime.a -o $@ $(LDFLAGS) $(LDLIBS)

kv260_dma_acc_int32_fp32_test: dma_acc_int32_fp32_test.cpp libnpu_kv260_runtime.a $(RUNTIME_HDR)
	$(CXX) $(CXXFLAGS) dma_acc_int32_fp32_test.cpp libnpu_kv260_runtime.a -o $@ $(LDFLAGS) $(LDLIBS)

kv260_mvin_problem_case_test: mvin_problem_case_test.cpp libnpu_kv260_runtime.a $(RUNTIME_HDR)
	$(CXX) $(CXXFLAGS) mvin_problem_case_test.cpp libnpu_kv260_runtime.a -o $@ $(LDFLAGS) $(LDLIBS)

kv260_dma_double_mvin_async_test: dma_double_mvin_async_test.cpp libnpu_kv260_runtime.a $(RUNTIME_HDR)
	$(CXX) $(CXXFLAGS) dma_double_mvin_async_test.cpp libnpu_kv260_runtime.a -o $@ $(LDFLAGS) $(LDLIBS)

kv260_gemm_plan_test: gemm_plan_test.cpp libnpu_kv260_runtime.a $(RUNTIME_HDR)
	$(CXX) $(CXXFLAGS) gemm_plan_test.cpp libnpu_kv260_runtime.a -o $@ $(LDFLAGS) $(LDLIBS)

kv260_network_gemm_layers_test: network_gemm_layers_test.cpp libnpu_kv260_runtime.a $(RUNTIME_HDR)
	$(CXX) $(CXXFLAGS) network_gemm_layers_test.cpp libnpu_kv260_runtime.a -o $@ $(LDFLAGS) $(LDLIBS)

kv260_layer_gemm_replay_test: layer_gemm_replay_test.cpp libnpu_kv260_runtime.a $(RUNTIME_HDR)
	$(CXX) $(CXXFLAGS) layer_gemm_replay_test.cpp libnpu_kv260_runtime.a -o $@ $(LDFLAGS) $(LDLIBS)

kv260_mmproj_layer_asym_w8a8_test: mmproj_layer_asym_w8a8_test.cpp libnpu_kv260_runtime.a $(RUNTIME_HDR)
	$(CXX) $(CXXFLAGS) mmproj_layer_asym_w8a8_test.cpp libnpu_kv260_runtime.a -o $@ $(LDFLAGS) $(LDLIBS)

clean:
	rm -f npu_runtime.o versa_p_device.o versa_p_mmio.o versa_p_commands.o versa_p_gemm_plan.o libnpu_kv260_runtime.a kv260_npu_smoke_test kv260_runtime_init_test kv260_dma_loopback_test kv260_dma_acc_int32_fp32_test kv260_mvin_problem_case_test kv260_dma_double_mvin_async_test kv260_gemm_plan_test kv260_network_gemm_layers_test kv260_layer_gemm_replay_test kv260_mmproj_layer_asym_w8a8_test
