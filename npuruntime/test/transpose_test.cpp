/**
 * @file transpose_test.cpp
 * @brief NPU Transpose Test - 验证矩阵转置功能的硬件可靠性、计算正确性和延迟
 * 
 * 测试内容:
 * 1. 可靠性测试: 多次运行验证结果一致性
 * 2. 正确性测试: 与软件参考结果对比
 * 3. 延迟测试: 测量不同规模的执行时间
 * 
 * 使用方法:
 *   make $(PREFIX)_transpose
 *   ./$(PREFIX)_transpose
 */

#include "../npu_runtime.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// ==========================================
// 编译时配置
// ==========================================

// 最大允许的误差
#ifndef MAX_DIFF
#define MAX_DIFF 0
#endif

// 可靠性测试重复次数
#ifndef RELIABILITY_RUNS
#define RELIABILITY_RUNS 1
#endif

// 延迟测试重复次数 (用于统计平均)
#ifndef LATENCY_RUNS
#define LATENCY_RUNS 1
#endif

// ==========================================
// 测试用例结构
// ==========================================

struct TransposeCase {
    std::string name;
    int rows;
    int cols;
    bool random;          // 是否使用随机数据
    bool test_latency;    // 是否进行延迟测试
};

// ==========================================
// 测试统计
// ==========================================

struct TestStats {
    size_t total_elements;
    size_t mismatches;
    int max_diff;
    double avg_latency_us;    // 平均延迟 (微秒)
    double min_latency_us;
    double max_latency_us;
};

// ==========================================
// 软件参考实现: 矩阵转置
// ==========================================

static void software_transpose(const int8_t* input, int8_t* output, int rows, int cols) {
    // 输入: rows x cols 矩阵 (行优先存储)
    // 输出: cols x rows 矩阵 (行优先存储)
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            // input[r][c] -> output[c][r]
            output[c * rows + r] = input[r * cols + c];
        }
    }
}

// ==========================================
// 打印二维矩阵 (行优先)
// ==========================================

static void print_matrix_2d(
    const std::string& title,
    const int8_t* data,
    int rows,
    int cols
) {
    std::cout << "\n[Matrix] " << title << " (" << rows << " x " << cols << ")\n";
    for (int r = 0; r < rows; ++r) {
        std::cout << "  ";
        for (int c = 0; c < cols; ++c) {
            std::cout << std::setw(4) << static_cast<int>(data[r * cols + c]) << " ";
        }
        std::cout << "\n";
    }
}

// ==========================================
// 单个测试用例执行
// ==========================================

static TestStats run_transpose_case(
    const TransposeCase& tc,
    int8_t* host_in,
    int8_t* host_out
) {
    TestStats stats = {0, 0, 0, 0.0, 1e9, 0.0};
    
    const int rows = tc.rows;
    const int cols = tc.cols;
    const size_t input_size = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    const size_t output_size = input_size; // 转置后大小相同
    
    std::cout << "\n========================================\n";
    std::cout << "[TestCase] " << tc.name 
              << " (" << rows << " x " << cols << " -> " << cols << " x " << rows << ")\n";
    std::cout << "========================================\n";
    
    // 1. 生成测试数据
    std::mt19937 rng(2025 + rows * 17 + cols * 31);
    if (tc.random) {
        std::uniform_int_distribution<int> dist(-128, 127);
        for (size_t i = 0; i < input_size; ++i) {
            host_in[i] = static_cast<int8_t>(dist(rng));
        }
    } else {
        // 使用确定性模式便于调试
        for (size_t i = 0; i < input_size; ++i) {
            host_in[i] = static_cast<int8_t>((i % 256) - 128);
        }
    }
    
    // 2. 计算软件参考结果
    std::vector<int8_t> ref_out(output_size);
    software_transpose(host_in, ref_out.data(), rows, cols);
    
    std::memset(host_out, 0xAA, output_size);  // 填充标记值
    
    // SPM 地址布局 (字节地址)
    // 注意: 输出地址必须避开输入区域，避免 transpose 读写重叠导致结果污染。
    const uint32_t SPM_ADDR_IN = 0x0000;
    const uint32_t SPM_ADDR_LIMIT = 0x80000; // 与其他测试保持一致: 512KB
    const uint32_t input_end_aligned = (static_cast<uint32_t>(input_size) + 63u) & ~63u;
    const uint32_t SPM_ADDR_OUT = input_end_aligned;

    if (SPM_ADDR_OUT + static_cast<uint32_t>(output_size) > SPM_ADDR_LIMIT) {
        std::cerr << "[FATAL] SPM out of range for case " << tc.name
                  << ": input_size=" << input_size
                  << ", output_size=" << output_size
                  << ", out_end=0x" << std::hex
                  << (SPM_ADDR_OUT + static_cast<uint32_t>(output_size))
                  << " > limit=0x" << SPM_ADDR_LIMIT << std::dec << "\n";
        stats.total_elements = output_size;
        stats.mismatches = output_size;
        stats.max_diff = 255;
        stats.avg_latency_us = 0.0;
        stats.min_latency_us = 0.0;
        stats.max_latency_us = 0.0;
        return stats;
    }
    
    // ========================================
    // 3. MVIN: 将数据从 Host 搬入 SPM
    // ========================================
    
    npu_dma_mvin(
        /*host_ptr=*/host_in,
        /*sram_addr=*/SPM_ADDR_IN,
        /*col_num=*/static_cast<uint16_t>(cols - 1),
        /*row_num=*/static_cast<uint16_t>(rows - 1),
        /*sram_stride=*/static_cast<uint16_t>(cols),
        /*dram_stride=*/static_cast<uint16_t>(cols),
        /*precision=*/1,
        /*input_type=*/0,
        /*dest=*/0,
        /*is_bias=*/false,
        /*is_quant=*/false,
        /*quant_zero=*/0,
        /*quant_scale=*/0,
        /*quant_shift=*/0
    );
    
    // ========================================
    // 4. 可靠性 & 延迟测试: 多次执行 TRANSPOSE
    // ========================================
    
    std::vector<double> latencies;
    int reliability_pass = 0;
    
    int num_runs = tc.test_latency ? LATENCY_RUNS : RELIABILITY_RUNS;
    
    for (int run = 0; run < num_runs; ++run) {
        std::memset(host_out, 0xAA, output_size);
        
        // 计时开始
        auto t_start = std::chrono::high_resolution_clock::now();
        
        // 执行 TRANSPOSE
        npu_transpose_run(
            /*input_sram_addr=*/SPM_ADDR_IN,
            /*output_sram_addr=*/SPM_ADDR_OUT,
            /*col_num=*/static_cast<uint16_t>(cols - 1),
            /*row_num=*/static_cast<uint16_t>(rows - 1),
            /*out_padding_row=*/false,
            /*out_padding_col=*/false
        );
        
        // 计时结束
        auto t_end = std::chrono::high_resolution_clock::now();
        double latency_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();
        latencies.push_back(latency_us);
        
        if (latency_us < stats.min_latency_us) stats.min_latency_us = latency_us;
        if (latency_us > stats.max_latency_us) stats.max_latency_us = latency_us;
        
        // MVOUT: 将结果从 SPM 搬出到 Host
        npu_dma_mvout(
            /*host_ptr=*/host_out,
            /*sram_addr=*/SPM_ADDR_OUT,
            /*col_num=*/static_cast<uint16_t>(rows - 1),
            /*row_num=*/static_cast<uint16_t>(cols - 1),
            /*sram_stride=*/static_cast<uint16_t>(rows),
            /*dram_stride=*/static_cast<uint16_t>(rows),
            /*precision=*/1,
            /*output_type=*/0,
            /*source=*/0,
            /*is_quant=*/false,
            /*quant_zero=*/0,
            /*quant_scale=*/0,
            /*quant_shift=*/0
        );
        
        // 验证结果
        bool run_pass = true;
        for (size_t i = 0; i < output_size; ++i) {
            int diff = std::abs(static_cast<int>(host_out[i]) - static_cast<int>(ref_out[i]));
            if (diff > MAX_DIFF) {
                run_pass = false;
                break;
            }
        }
        
        if (run_pass) {
            reliability_pass++;
            std::cout << "  Run " << (run + 1) << "/" << num_runs 
                      << ": PASS (latency=" << std::fixed << std::setprecision(2) 
                      << latency_us << " us)\n";
        } else {
            std::cout << "  Run " << (run + 1) << "/" << num_runs << ": FAIL\n";
        }
    }
    
    // ========================================
    // 5. 统计结果
    // ========================================
    
    stats.total_elements = output_size;
    
    // 计算延迟统计
    double sum_latency = 0.0;
    for (double lat : latencies) {
        sum_latency += lat;
    }
    stats.avg_latency_us = sum_latency / latencies.size();
    
    // 最终验证 (使用最后一次运行的结果)
    stats.mismatches = 0;
    stats.max_diff = 0;
    for (size_t i = 0; i < output_size; ++i) {
        int diff = std::abs(static_cast<int>(host_out[i]) - static_cast<int>(ref_out[i]));
        if (diff > MAX_DIFF) stats.mismatches++;
        if (diff > stats.max_diff) stats.max_diff = diff;
    }
    
    // 打印样本数据 (显示部分转置前后的对比)
    std::cout << "\n[Sample] Input (first row):\n  ";
    int print_cols = std::min(8, cols);
    for (int c = 0; c < print_cols; ++c) {
        std::cout << std::setw(4) << static_cast<int>(host_in[c]) << " ";
    }
    if (cols > 8) std::cout << "...";
    std::cout << "\n";
    
    std::cout << "[Sample] Output (first row, should be input's first column):\n  ";
    int print_out_cols = std::min(8, rows);
    for (int r = 0; r < print_out_cols; ++r) {
        std::cout << std::setw(4) << static_cast<int>(host_out[r]) << " ";
    }
    if (rows > 8) std::cout << "...";
    std::cout << "\n";
    
    std::cout << "[Sample] Reference (first row):\n  ";
    for (int r = 0; r < print_out_cols; ++r) {
        std::cout << std::setw(4) << static_cast<int>(ref_out[r]) << " ";
    }
    if (rows > 8) std::cout << "...";
    std::cout << "\n";
    
    // 打印总结
    std::cout << "\n[Result Summary]\n";
    std::cout << "  Reliability: " << reliability_pass << "/" << num_runs << " runs passed\n";
    std::cout << "  Correctness: " << (output_size - stats.mismatches) << "/" << output_size 
              << " elements match (max_diff=" << stats.max_diff << ")\n";
    std::cout << "  Latency: avg=" << std::fixed << std::setprecision(2) << stats.avg_latency_us 
              << " us, min=" << stats.min_latency_us 
              << " us, max=" << stats.max_latency_us << " us\n";
    
    if (tc.test_latency) {
        double throughput_mb = (input_size * 2.0) / stats.avg_latency_us;  // 读写各一次
        std::cout << "  Throughput: " << std::fixed << std::setprecision(2) 
                  << throughput_mb << " MB/s\n";
    }

    // 如果结果错误，打印错误范围 (每个 case 仅打印一次)
    if (stats.mismatches > 0) {
        const int out_rows = cols;
        const int out_cols = rows;

        // 找出错误范围 (行列索引)
        int min_row = out_rows, max_row = -1;
        int min_col = out_cols, max_col = -1;
        
        for (int r = 0; r < out_rows; ++r) {
            for (int c = 0; c < out_cols; ++c) {
                int idx = r * out_cols + c;
                int diff = std::abs(static_cast<int>(host_out[idx]) - static_cast<int>(ref_out[idx]));
                if (diff > MAX_DIFF) {
                    min_row = std::min(min_row, r);
                    max_row = std::max(max_row, r);
                    min_col = std::min(min_col, c);
                    max_col = std::max(max_col, c);
                }
            }
        }
        
        // 打印错误范围
        std::cout << "\n[ERROR RANGE] Rows [" << min_row << ", " << max_row 
                  << "], Cols [" << min_col << ", " << max_col << "]\n";
        
        // 打印错误范围内的数据对比
        // std::cout << "[Detail] Row\tCol\tExpected\tActual\tDiff\n";
        // for (int r = min_row; r <= max_row; ++r) {
        //     for (int c = min_col; c <= max_col; ++c) {
        //         int idx = r * cols + c;
        //         int expected = static_cast<int>(ref_out[idx]);
        //         int actual = static_cast<int>(host_out[idx]);
        //         int diff = std::abs(expected - actual);
        //         if (diff > MAX_DIFF) {
        //             std::cout << "        " << r << "\t" << c << "\t" 
        //                       << expected << "\t\t" << actual << "\t" << diff << "\n";
        //         }
        //     }
        // }
    }
    
    return stats;
}

// ==========================================
// Main
// ==========================================

int main() {
    std::cout << "=========================================\n";
    std::cout << "   NPU Transpose Test (Runtime API)\n";
    std::cout << "=========================================\n";
    std::cout << "Tests: Reliability, Correctness, Latency\n";
    std::cout << "MAX_DIFF=" << MAX_DIFF 
              << ", RELIABILITY_RUNS=" << RELIABILITY_RUNS 
              << ", LATENCY_RUNS=" << LATENCY_RUNS << "\n";
    
    // 初始化 NPU Runtime
    if (npu_init() != 0) {
        std::cerr << "[FATAL] NPU initialization failed!\n";
        return -1;
    }
    
    npu_reset();
    
    // ========================================
    // 定义测试用例
    // ========================================
    
    std::vector<TransposeCase> test_cases = {
        // 基础功能测试
        // {"Small_4x4",           4,   4,  false, false},
        // {"Small_8x8",           8,   8,  false, false},
        // {"Rect_4x8",            4,   8,  false, false},
        // {"Rect_8x4",            8,   4,  false, false},
        
        // // 中等规模测试
        // {"Medium_16x16",       16,  16,  false,  false},
        // {"Medium_32x32",       32,  32,  false,  false},
        // {"Medium_16x32",       16,  32,  false,  false},
        // {"Medium_32x16",       32,  16,  false,  false},
        
        // // 大规模测试 (延迟测试)
        // {"Large_64x64",        64,  64,  false,  true},
        // {"Large_64x64",        60,  60,  false,  true},
        // {"Large_128x128",     128, 128,  false,  true},
        // {"Large_256x256",     256, 256,  false,  true},
        
        {"real_128x512", 128, 512,  false,   false},
        {"real_128x640", 128, 640,  false,   false},
        {"real_128x768", 128, 768,  false,   false},
    };
    
    // 分配 Host 缓冲区 (使用最大尺寸)
    const size_t MAX_SIZE = 256 * 768;
    int8_t* host_in  = static_cast<int8_t*>(npu_mem_alloc(MAX_SIZE));
    int8_t* host_out = static_cast<int8_t*>(npu_mem_alloc(MAX_SIZE));
    
    if (!host_in || !host_out) {
        std::cerr << "[FATAL] Memory allocation failed!\n";
        npu_destroy();
        return -1;
    }
    
    // ========================================
    // 运行测试用例
    // ========================================
    
    int total_tests = 0;
    int passed_tests = 0;
    
    for (const auto& tc : test_cases) {
        TestStats stats = run_transpose_case(tc, host_in, host_out);
        
        total_tests++;
        if (stats.mismatches == 0) {
            passed_tests++;
            std::cout << "[PASS] " << tc.name << "\n";
        } else {
            std::cout << "[FAIL] " << tc.name 
                      << " (" << stats.mismatches << " mismatches)\n";
        }
    }
    
    // ========================================
    // 总结
    // ========================================
    
    std::cout << "\n=========================================\n";
    std::cout << "   Test Summary\n";
    std::cout << "=========================================\n";
    std::cout << "Total: " << passed_tests << "/" << total_tests << " tests passed\n";
    
    // 清理
    npu_mem_free(host_in);
    npu_mem_free(host_out);
    npu_destroy();
    
    return (passed_tests == total_tests) ? 0 : 1;
}
