CXX ?= g++
CC ?= gcc
AR ?= ar

CXXFLAGS ?= -std=c++17 -Wall -Wextra -O2
CXXFLAGS += -DNPU_CPU_WIDTH=64 -DNPU_SHADOW_REGS=1 -I. -I../driver
CFLAGS ?= -std=c11 -Wall -Wextra -O2
CFLAGS += -I../driver

RUNTIME_SRC := npu_runtime.cpp
RUNTIME_HDR := npu_runtime.h npu_regs_compat.h npu_regs.h ../driver/npu_kv260_uapi.h

.PHONY: all clean

all: libnpu_kv260_runtime.a kv260_npu_smoke_test kv260_runtime_init_test kv260_dma_loopback_test

npu_runtime.o: $(RUNTIME_SRC) $(RUNTIME_HDR)
	$(CXX) $(CXXFLAGS) -c $(RUNTIME_SRC) -o $@

libnpu_kv260_runtime.a: npu_runtime.o
	$(AR) rcs $@ $<

kv260_npu_smoke_test: smoke_test.c ../driver/npu_kv260_uapi.h
	$(CC) $(CFLAGS) smoke_test.c -o $@ $(LDFLAGS) $(LDLIBS)

kv260_runtime_init_test: runtime_init_test.cpp libnpu_kv260_runtime.a $(RUNTIME_HDR)
	$(CXX) $(CXXFLAGS) runtime_init_test.cpp libnpu_kv260_runtime.a -o $@ $(LDFLAGS) $(LDLIBS)

kv260_dma_loopback_test: dma_loopback_test.cpp libnpu_kv260_runtime.a $(RUNTIME_HDR)
	$(CXX) $(CXXFLAGS) dma_loopback_test.cpp libnpu_kv260_runtime.a -o $@ $(LDFLAGS) $(LDLIBS)

clean:
	rm -f npu_runtime.o libnpu_kv260_runtime.a kv260_npu_smoke_test kv260_runtime_init_test kv260_dma_loopback_test
