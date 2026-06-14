CXX ?= g++
CC ?= gcc
AR ?= ar

ROOT_DIR := .
INC_DIR := $(ROOT_DIR)/include
SRC_DIR := $(ROOT_DIR)/src
TEST_DIR := $(ROOT_DIR)/tests
BUILD_DIR := $(ROOT_DIR)/build
OBJ_DIR := $(BUILD_DIR)/obj
LIB_DIR := $(BUILD_DIR)/lib
BIN_DIR := $(BUILD_DIR)/bin

CXXFLAGS ?= -std=c++17 -Wall -Wextra -O2
CXXFLAGS += -DNPU_CPU_WIDTH=64 -DNPU_SHADOW_REGS=1 -I$(INC_DIR) -I$(SRC_DIR) -I../driver
CFLAGS ?= -std=c11 -Wall -Wextra -O2
CFLAGS += -I$(INC_DIR) -I../driver

RUNTIME_SRC := $(SRC_DIR)/npu_runtime.cpp
RUNTIME_HDR := $(INC_DIR)/npu_runtime.h $(INC_DIR)/npu_kv260_uapi.h
RUNTIME_PRIVATE_HDR := $(SRC_DIR)/npu_regs_compat.h $(SRC_DIR)/npu_regs.h
RUNTIME_OBJ := $(OBJ_DIR)/npu_runtime.o
RUNTIME_LIB := $(LIB_DIR)/libnpu_kv260_runtime.a

# Default board regression for the current public stream API.
TARGET_NAMES := kv260_npu_smoke_test kv260_runtime_init_test kv260_runtime_api_contract_test kv260_stream_gemv_api_test
TARGETS := $(addprefix $(BIN_DIR)/,$(TARGET_NAMES))
STALE_TARGET_NAMES := kv260_dma_loopback_test kv260_dma_acc_int32_fp32_test kv260_bias_reload_test
STALE_TARGETS := $(addprefix $(BIN_DIR)/,$(STALE_TARGET_NAMES))

.PHONY: all clean $(TARGET_NAMES)

all: $(RUNTIME_LIB) $(TARGETS)

$(OBJ_DIR) $(LIB_DIR) $(BIN_DIR):
	mkdir -p $@

$(RUNTIME_OBJ): $(RUNTIME_SRC) $(RUNTIME_HDR) $(RUNTIME_PRIVATE_HDR) | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(RUNTIME_LIB): $(RUNTIME_OBJ) | $(LIB_DIR)
	$(AR) rcs $@ $<

$(BIN_DIR)/kv260_npu_smoke_test: $(TEST_DIR)/smoke_test.c $(INC_DIR)/npu_kv260_uapi.h | $(BIN_DIR)
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

$(BIN_DIR)/kv260_runtime_init_test: $(TEST_DIR)/runtime_init_test.cpp $(RUNTIME_LIB) $(RUNTIME_HDR) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $< $(RUNTIME_LIB) -o $@ $(LDFLAGS) $(LDLIBS)

$(BIN_DIR)/kv260_runtime_api_contract_test: $(TEST_DIR)/runtime_api_contract_test.cpp $(RUNTIME_LIB) $(RUNTIME_HDR) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $< $(RUNTIME_LIB) -o $@ $(LDFLAGS) $(LDLIBS)

$(BIN_DIR)/kv260_stream_gemv_api_test: $(TEST_DIR)/stream_gemv_api_test.cpp $(RUNTIME_LIB) $(RUNTIME_HDR) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $< $(RUNTIME_LIB) -o $@ $(LDFLAGS) $(LDLIBS)

$(BIN_DIR)/kv260_stream_gemv_perf_test: $(TEST_DIR)/stream_gemv_perf_test.cpp $(RUNTIME_LIB) $(RUNTIME_HDR) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $< $(RUNTIME_LIB) -o $@ $(LDFLAGS) $(LDLIBS)

$(BIN_DIR)/kv260_stream_attn_pv_dump_repro_test: $(TEST_DIR)/stream_attn_pv_dump_repro_test.cpp $(RUNTIME_LIB) $(RUNTIME_HDR) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $< $(RUNTIME_LIB) -o $@ $(LDFLAGS) $(LDLIBS)

$(BIN_DIR)/kv260_ffn_swiglu_actscale_test: $(TEST_DIR)/ffn_swiglu_actscale_test.cpp $(RUNTIME_LIB) $(RUNTIME_HDR) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $< $(RUNTIME_LIB) -o $@ $(LDFLAGS) $(LDLIBS)

$(BIN_DIR)/kv260_decode_full_stream_perf_test: $(TEST_DIR)/decode_full_stream_perf_test.cpp $(RUNTIME_LIB) $(RUNTIME_HDR) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $< $(RUNTIME_LIB) -o $@ $(LDFLAGS) $(LDLIBS)

kv260_npu_smoke_test: $(BIN_DIR)/kv260_npu_smoke_test
kv260_runtime_init_test: $(BIN_DIR)/kv260_runtime_init_test
kv260_runtime_api_contract_test: $(BIN_DIR)/kv260_runtime_api_contract_test
kv260_stream_gemv_api_test: $(BIN_DIR)/kv260_stream_gemv_api_test
kv260_stream_gemv_perf_test: $(BIN_DIR)/kv260_stream_gemv_perf_test
kv260_stream_attn_pv_dump_repro_test: $(BIN_DIR)/kv260_stream_attn_pv_dump_repro_test
kv260_ffn_swiglu_actscale_test: $(BIN_DIR)/kv260_ffn_swiglu_actscale_test
kv260_decode_full_stream_perf_test: $(BIN_DIR)/kv260_decode_full_stream_perf_test

clean:
	rm -rf $(BUILD_DIR)
