#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <cmath>
#include <cstdint>

// 引入 Runtime 头文件 (假设保存为 npu_runtime.h)
#include "../npu_runtime.h"

using namespace std;
using namespace std::chrono;

// ==========================================
// 辅助工具
// ==========================================

// 获取高精度时间 (微秒)
double get_time_us() {
    auto now = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(now.time_since_epoch());
    return (double)duration.count();
}

void print_separator() {
    cout << "------------------------------------------------------------" << endl;
}

// ==========================================
// 测试 1: 指令启动延迟 (Latency)
// 目标：测量“软件栈 + 驱动 + 硬件启动”的固定开销
// 方法：搬运极小数据 (1行 x 32列 int8)，耗时几乎全为指令开销
// ==========================================
void test_instruction_overhead() {
    print_separator();
    cout << "[Test 1] Instruction Latency (Software + Hardware Overhead)" << endl;

    // 1. 准备数据
    // 这是一个极小的 payload：1行，32列 (32 Bytes)
    // 对应硬件的一个原子操作或最小总线宽度
    uint16_t cols = 32; 
    uint16_t rows = 1;
    uint16_t stride = cols; // 连续搬运，stride = 列数
    size_t transfer_size = cols * rows * sizeof(uint8_t);

    void* host_ptr = npu_mem_alloc(4096); // 分配一页，足够用
    if (!host_ptr) {
        cerr << "Memory alloc failed!" << endl;
        return;
    }
    // 填充 0xAA 用于调试
    memset(host_ptr, 0xAA, transfer_size);

    uint32_t sram_addr = 0x0;
    int iterations = 1000;

    // 2. 预热 (Warmup) - 消除首次缺页中断影响
    npu_dma_mvin(
        /*host_ptr=*/host_ptr,
        /*sram_addr=*/sram_addr,
        /*col_num=*/cols - 1,
        /*row_num=*/rows - 1,
        /*sram_stride=*/stride,
        /*dram_stride=*/stride,
        /*precision=*/1,
        /*input_type=*/0,
        /*dest=*/0,
        /*is_bias=*/false,
        /*is_quant=*/false,
        /*quant_zero=*/0,
        /*quant_scale=*/0,
        /*quant_shift=*/0
    );

    // 3. 开始测试
    double start_time = get_time_us();

    for (int i = 0; i < iterations; i++) {
        npu_dma_mvin(
            /*host_ptr=*/host_ptr,
            /*sram_addr=*/sram_addr,
            /*col_num=*/cols - 1,
            /*row_num=*/rows - 1,
            /*sram_stride=*/stride,
            /*dram_stride=*/stride,
            /*precision=*/1,
            /*input_type=*/0,
            /*dest=*/0,
            /*is_bias=*/false,
            /*is_quant=*/false,
            /*quant_zero=*/0,
            /*quant_scale=*/0,
            /*quant_shift=*/0
        );
    }

    double end_time = get_time_us();

    // 4. 计算结果
    double total_us = end_time - start_time;
    double avg_latency_us = total_us / iterations;

    cout << "  Payload:      " << transfer_size << " Bytes (int8)" << endl;
    cout << "  Iterations:   " << iterations << endl;
    cout << "  Avg Latency:  " << avg_latency_us << " us / instruction" << endl;
    
    npu_mem_free(host_ptr);
}

// ==========================================
// 测试 2: 数据搬运带宽 (Bandwidth)
// 目标：测量 DMA 在搬运 int8 大块数据时的吞吐率
// ==========================================
void test_data_transport_overhead() {
    print_separator();
    cout << "[Test 2] Data Transport Bandwidth (int8 Matrix)" << endl;

    // 1. 定义矩阵维度 (64KB 数据量)
    // 为了模拟真实场景，我们设置一个稍大的矩阵
    // 宽度 (Cols) = 1024 (int8)
    // 高度 (Rows) = 256
    // 总大小 = 256 * 1024 = 262144 Bytes = 256KB
    uint16_t cols = 1024;
    uint16_t rows = 257;
    uint16_t stride = cols; // 连续搬运，stride 等于宽度
    size_t transfer_size = cols * rows * sizeof(uint8_t);

    void* src_ptr = npu_mem_alloc(transfer_size);
    void* dst_ptr = npu_mem_alloc(transfer_size);

    if (!src_ptr || !dst_ptr) {
        cerr << "Memory alloc failed!" << endl;
        return;
    }

    // 初始化数据 (int8)
    // src 填 0x55 (01010101)
    // dst 填 0x00
    memset(src_ptr, 0x55, transfer_size);
    memset(dst_ptr, 0x00, transfer_size);

    int iterations = 500; // 跑 500 次以取平均值
    uint32_t sram_addr = 0x0;

    // Warmup
    npu_dma_mvin(
        /*host_ptr=*/src_ptr,
        /*sram_addr=*/sram_addr,
        /*col_num=*/cols - 1,
        /*row_num=*/rows - 1,
        /*sram_stride=*/stride,
        /*dram_stride=*/stride,
        /*precision=*/1,
        /*input_type=*/0,
        /*dest=*/0,
        /*is_bias=*/false,
        /*is_quant=*/false,
        /*quant_zero=*/0,
        /*quant_scale=*/0,
        /*quant_shift=*/0
    );

    // -------------------------------------------------
    // Phase A: MVIN 测试 (DDR -> SRAM)
    // -------------------------------------------------
    double start = get_time_us();
    for (int i = 0; i < iterations; i++) {
        npu_dma_mvin(
            /*host_ptr=*/src_ptr,
            /*sram_addr=*/sram_addr,
            /*col_num=*/cols - 1,
            /*row_num=*/rows - 1,
            /*sram_stride=*/stride,
            /*dram_stride=*/stride,
            /*precision=*/1,
            /*input_type=*/0,
            /*dest=*/0,
            /*is_bias=*/false,
            /*is_quant=*/false,
            /*quant_zero=*/0,
            /*quant_scale=*/0,
            /*quant_shift=*/0
        );
    }
    double duration_us = get_time_us() - start;

    double total_bytes_mb = (double)(transfer_size * iterations) / (1024.0 * 1024.0);
    double duration_s = duration_us / 1000000.0;
    double bw_mvin = total_bytes_mb / duration_s;

    cout << "  Matrix Shape: " << rows << "x" << cols << " (int8)" << endl;
    cout << "  Block Size:   " << transfer_size / 1024.0 << " KB" << endl;
    cout << "  [MVIN] Bandwidth:  " << fixed << setprecision(2) << bw_mvin << " MB/s" << endl;

    // -------------------------------------------------
    // Phase B: MVOUT 测试 (SRAM -> DDR)
    // -------------------------------------------------
    start = get_time_us();
    for (int i = 0; i < iterations; i++) {
        npu_dma_mvout(
            /*host_ptr=*/dst_ptr,
            /*sram_addr=*/sram_addr,
            /*col_num=*/cols - 1,
            /*row_num=*/rows - 1,
            /*sram_stride=*/stride,
            /*dram_stride=*/stride,
            /*precision=*/1,
            /*output_type=*/0,
            /*source=*/0,
            /*is_quant=*/false,
            /*quant_zero=*/0,
            /*quant_scale=*/0,
            /*quant_shift=*/0
        );
    }
    duration_us = get_time_us() - start;
    
    double bw_mvout = total_bytes_mb / (duration_us / 1000000.0);
    cout << "  [MVOUT] Bandwidth: " << fixed << setprecision(2) << bw_mvout << " MB/s" << endl;

    // -------------------------------------------------
    // Phase C: 数据正确性校验 (Loopback Check)
    // -------------------------------------------------
    uint8_t* p_in = (uint8_t*)src_ptr;
    uint8_t* p_out = (uint8_t*)dst_ptr;
    bool pass = true;
    int err_cnt = 0;

    // 随机抽查或全量检查
    for (size_t i = 0; i < transfer_size; i++) {
        if (p_in[i] != p_out[i]) {
            pass = false;
            if (err_cnt++ < 5) {
                printf("  [Error] Index %zu: Expect 0x%02X, Got 0x%02X\n", i, p_in[i], p_out[i]);
            }
        }
    }

    if (pass) {
        cout << "  [Check] Data Verification PASS." << endl;
    } else {
        cout << "  [Check] Data Verification FAILED." << endl;
    }

    npu_mem_free(src_ptr);
    npu_mem_free(dst_ptr);
}

int main() {
    cout << "=== NPU Performance Benchmark (int8 mode) ===" << endl;

    // 1. 初始化 Runtime
    if (npu_init() != 0) {
        cerr << "Error: NPU init failed. Check kernel driver." << endl;
        return -1;
    }

    try {
        // 2. 运行延迟测试
        test_instruction_overhead();

        // 3. 运行带宽测试
        test_data_transport_overhead();

    } catch (const std::exception& e) {
        cerr << "Exception: " << e.what() << endl;
    }

    npu_destroy();
    return 0;
}