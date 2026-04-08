/**
 * @file resample_test.cpp
 * @brief NPU Resample Test - 验证 2x2 Max Pooling 的硬件可靠性、计算正确性和延迟
 * 
 * 测试内容:
 * 1. 可靠性测试: 多次运行验证结果一致性
 * 2. 正确性测试: 与软件参考结果对比
 * 3. 延迟测试: 测量不同规模的执行时间
 * 
 * 当前硬件仅支持 2x2 Max Pooling (stride=2)
 * 
 * 使用方法:
 *   make $(PREFIX)_resample
 *   ./$(PREFIX)_resample
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

struct ResampleCase {
    std::string name;
    int channels;         // 输入通道数 C
    int rows;             // 输入高度 H
    int cols;             // 输入宽度 W
    uint8_t type;         // 0=downsample, 1=upsample, 2=pooling
    uint8_t op;           // 0=max/nearest, 1=avg
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
// 简单 SPM 地址分配器
// ==========================================

struct SpmRegion {
    uint32_t addr;
    size_t size;
};

class SimpleSpmAllocator {
public:
    explicit SimpleSpmAllocator(uint32_t base = 0, uint32_t limit = (1u << 19))
        : base_(base), cursor_(base), limit_(limit) {}

    bool alloc(size_t bytes, SpmRegion& out, uint32_t align = 64) {
        uint32_t aligned = align_up(cursor_, align);
        uint64_t end = static_cast<uint64_t>(aligned) + static_cast<uint64_t>(bytes);
        if (end > limit_) {
            return false;
        }
        out.addr = aligned;
        out.size = bytes;
        cursor_ = static_cast<uint32_t>(end);
        return true;
    }

private:
    static uint32_t align_up(uint32_t v, uint32_t align) {
        if (align == 0) return v;
        return (v + align - 1) & ~(align - 1);
    }

    uint32_t base_;
    uint32_t cursor_;
    uint32_t limit_;
};

// ==========================================
// 软件参考实现: 2x2 Max Pooling (stride=2)
// ==========================================

static void software_maxpool_2x2(const int8_t* input, int8_t* output, 
                                  int in_rows, int in_cols,
                                  int& out_rows, int& out_cols) {
    // 输出尺寸: (in_rows / 2) x (in_cols / 2)
    out_rows = in_rows / 2;
    out_cols = in_cols / 2;
    
    for (int or_ = 0; or_ < out_rows; ++or_) {
        for (int oc = 0; oc < out_cols; ++oc) {
            int ir = or_ * 2;
            int ic = oc * 2;
            
            // 2x2 窗口的 4 个元素
            int8_t v00 = input[ir * in_cols + ic];
            int8_t v01 = input[ir * in_cols + ic + 1];
            int8_t v10 = input[(ir + 1) * in_cols + ic];
            int8_t v11 = input[(ir + 1) * in_cols + ic + 1];
            
            // 取最大值
            int8_t max_val = v00;
            if (v01 > max_val) max_val = v01;
            if (v10 > max_val) max_val = v10;
            if (v11 > max_val) max_val = v11;
            
            output[or_ * out_cols + oc] = max_val;
        }
    }
}

// ==========================================
// 软件参考实现: 2x2 Average Pooling (stride=2)
// ==========================================

static void software_avgpool_2x2(const int8_t* input, int8_t* output, 
                                  int in_rows, int in_cols,
                                  int& out_rows, int& out_cols) {
    out_rows = in_rows / 2;
    out_cols = in_cols / 2;
    
    for (int or_ = 0; or_ < out_rows; ++or_) {
        for (int oc = 0; oc < out_cols; ++oc) {
            int ir = or_ * 2;
            int ic = oc * 2;
            
            int8_t v00 = input[ir * in_cols + ic];
            int8_t v01 = input[ir * in_cols + ic + 1];
            int8_t v10 = input[(ir + 1) * in_cols + ic];
            int8_t v11 = input[(ir + 1) * in_cols + ic + 1];
            
            // 平均值 (四舍五入)
            int sum = static_cast<int>(v00) + static_cast<int>(v01) + 
                      static_cast<int>(v10) + static_cast<int>(v11);
            int avg = (sum >= 0) ? (sum + 2) / 4 : (sum - 2) / 4;
            
            // 饱和到 int8 范围
            if (avg > 127) avg = 127;
            if (avg < -128) avg = -128;
            
            output[or_ * out_cols + oc] = static_cast<int8_t>(avg);
        }
    }
}

// ==========================================
// 软件参考实现: 2x Nearest Upsample
// ==========================================

static void software_upsample_nearest_2x(const int8_t* input, int8_t* output,
                                         int in_rows, int in_cols,
                                         int& out_rows, int& out_cols) {
    out_rows = in_rows * 2;
    out_cols = in_cols * 2;

    for (int r = 0; r < in_rows; ++r) {
        for (int c = 0; c < in_cols; ++c) {
            int8_t v = input[r * in_cols + c];
            int rr = r * 2;
            int cc = c * 2;
            output[rr * out_cols + cc] = v;
            output[rr * out_cols + cc + 1] = v;
            output[(rr + 1) * out_cols + cc] = v;
            output[(rr + 1) * out_cols + cc + 1] = v;
        }
    }
}

// ==========================================
// 单个测试用例执行
// ==========================================

static TestStats run_resample_case(
    const ResampleCase& tc,
    int8_t* host_in,
    int8_t* host_out
) {
    const bool matrix_layout_mode = (tc.name.find("RowCol_") != std::string::npos);
    const bool is_debug_case = matrix_layout_mode;
    TestStats stats = {0, 0, 0, 0.0, 1e9, 0.0};
    
    const int in_c = tc.channels;
    const int in_h = tc.rows;
    const int in_w = tc.cols;
    const int matrix_in_rows = matrix_layout_mode ? in_h : (in_c * in_h);
    const int matrix_in_cols = in_w;
    const size_t input_size = static_cast<size_t>(matrix_in_rows) * static_cast<size_t>(matrix_in_cols);
    
    // 输出尺寸根据模式决定
    int out_h = (tc.type == RESAMPLE_TYPE_UPSAMPLE) ? (matrix_in_rows * 2) : (matrix_in_rows / 2);
    int out_w = (tc.type == RESAMPLE_TYPE_UPSAMPLE) ? (matrix_in_cols * 2) : ((matrix_in_cols + 1) / 2);
    const size_t output_size = static_cast<size_t>(out_h) * static_cast<size_t>(out_w);
    
    std::cout << "\n========================================\n";
    std::cout << "[TestCase] " << tc.name 
              << " (C=" << in_c << ", H=" << in_h << ", W=" << in_w
              << " -> H=" << out_h << ", W=" << out_w << ")\n";
    std::cout << "  Type=" << (tc.type == 0 ? "downsample" : (tc.type == 1 ? "upsample" : "pooling"))
              << ", Op=" << (tc.op == 0 ? "max/nearest" : "avg") << "\n";
    std::cout << "========================================\n";
    if (is_debug_case) {
        std::cout << "[Debug] Case=" << tc.name
                  << ", in_c=" << in_c
                  << ", in_h=" << in_h
                  << ", in_w=" << in_w
                  << ", matrix_rows=" << matrix_in_rows
                  << ", matrix_cols=" << matrix_in_cols
                  << ", input_size=" << input_size
                  << ", output_size=" << output_size << "\n";
    }
    
    // 1. 生成测试数据
    std::mt19937 rng(2025 + in_c * 17 + in_h * 31 + in_w * 13);
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
    int ref_out_rows, ref_out_cols;
    if (tc.type == RESAMPLE_TYPE_UPSAMPLE) {
        software_upsample_nearest_2x(host_in, ref_out.data(), matrix_in_rows, matrix_in_cols, ref_out_rows, ref_out_cols);
    } else if (tc.op == RESAMPLE_OP_MAX) {
        software_maxpool_2x2(host_in, ref_out.data(), matrix_in_rows, matrix_in_cols, ref_out_rows, ref_out_cols);
    } else {
        software_avgpool_2x2(host_in, ref_out.data(), matrix_in_rows, matrix_in_cols, ref_out_rows, ref_out_cols);
    }
    
    std::memset(host_out, 0xAA, output_size);  // 填充标记值

    // SPM 地址布局（动态分配，避免硬编码）
    SimpleSpmAllocator spm_alloc;
    SpmRegion spm_in{}, spm_out{};
    if (!spm_alloc.alloc(input_size, spm_in, 64) || !spm_alloc.alloc(output_size, spm_out, 64)) {
        throw std::runtime_error("SPM allocation failed: input/output region exceeds SPM capacity");
    }
    const uint32_t SPM_ADDR_IN  = spm_in.addr;
    const uint32_t SPM_ADDR_OUT = spm_out.addr;
    
    // ========================================
    // 3. MVIN: 将数据从 Host 搬入 SPM
    // ========================================
    // 对于 CxHxW contiguous layout:
    // row_num = 64 通道 - 1 = 63 (每个通道是完整的 56x56 = 3136 个元素)
    // col_num = 56x56 - 1 = 3135
    // stride = 56x56 = 3136
    
    int mvin_row_num = matrix_in_rows - 1;
    int mvin_col_num = matrix_in_cols - 1;
    int mvin_stride = matrix_in_cols;

    // 先清零 SPM 输入区域，避免历史残留数据影响
    // 注意：DMA host_ptr 必须来自 npu_mem_alloc 对应的 NPU 映射内存
    int8_t* zero_in = static_cast<int8_t*>(npu_mem_alloc(input_size));
    if (!zero_in) {
        throw std::runtime_error("Failed to allocate zero_in buffer from NPU memory");
    }
    std::memset(zero_in, 0, input_size);
    npu_dma_mvin(
        /*host_ptr=*/zero_in,
        /*sram_addr=*/SPM_ADDR_IN,
        /*col_num=*/static_cast<uint32_t>(mvin_col_num),
        /*row_num=*/static_cast<uint32_t>(mvin_row_num),
        /*sram_stride=*/static_cast<uint16_t>(mvin_stride),
        /*dram_stride=*/static_cast<uint32_t>(mvin_stride),
        /*precision=*/0,
        /*input_type=*/0,
        /*dest=*/0,
        /*is_bias=*/false,
        /*is_quant=*/false,
        /*quant_zero=*/0,
        /*quant_scale=*/0,
        /*quant_shift=*/0
    );

    // 先清零 SPM 输出区域，避免历史残留数据影响
    int zero_out_row_num = out_h - 1;
    int zero_out_col_num = out_w - 1;
    int zero_out_stride = out_w;
    int8_t* zero_out = static_cast<int8_t*>(npu_mem_alloc(output_size));
    if (!zero_out) {
        npu_mem_free(zero_in);
        throw std::runtime_error("Failed to allocate zero_out buffer from NPU memory");
    }
    std::memset(zero_out, 0, output_size);
    npu_dma_mvin(
        /*host_ptr=*/zero_out,
        /*sram_addr=*/SPM_ADDR_OUT,
        /*col_num=*/static_cast<uint32_t>(zero_out_col_num),
        /*row_num=*/static_cast<uint32_t>(zero_out_row_num),
        /*sram_stride=*/static_cast<uint16_t>(zero_out_stride),
        /*dram_stride=*/static_cast<uint32_t>(zero_out_stride),
        /*precision=*/0,
        /*input_type=*/0,
        /*dest=*/0,
        /*is_bias=*/false,
        /*is_quant=*/false,
        /*quant_zero=*/0,
        /*quant_scale=*/0,
        /*quant_shift=*/0
    );

    npu_mem_free(zero_in);
    npu_mem_free(zero_out);
    
    npu_dma_mvin(
        /*host_ptr=*/host_in,
        /*sram_addr=*/SPM_ADDR_IN,
        /*col_num=*/static_cast<uint32_t>(mvin_col_num),
        /*row_num=*/static_cast<uint32_t>(mvin_row_num),
        /*sram_stride=*/static_cast<uint16_t>(mvin_stride),
        /*dram_stride=*/static_cast<uint32_t>(mvin_stride),
        /*precision=*/0,  // 与日志一致
        /*input_type=*/0,
        /*dest=*/0,
        /*is_bias=*/false,
        /*is_quant=*/false,
        /*quant_zero=*/0,
        /*quant_scale=*/0,
        /*quant_shift=*/0
    );
    if (is_debug_case) {
        std::cout << "[Debug] MVIN done: SPM_ADDR_IN=0x" << std::hex << SPM_ADDR_IN
                  << std::dec << ", col_num=" << mvin_col_num 
                  << ", row_num=" << mvin_row_num 
                  << ", stride=" << mvin_stride << "\n";
    }
    
    // ========================================
    // 4. 可靠性 & 延迟测试: 多次执行 RESAMPLE
    // ========================================
    
    std::vector<double> latencies;
    int reliability_pass = 0;
    
    int num_runs = tc.test_latency ? LATENCY_RUNS : RELIABILITY_RUNS;
    
    for (int run = 0; run < num_runs; ++run) {
        std::memset(host_out, 0xAA, output_size);
        
        // 计时开始
        auto t_start = std::chrono::high_resolution_clock::now();
        
        // 执行 RESAMPLE
        // 说明：实测当 input_row_num 较大时，硬件会出现后半段输出为 0 的现象。
        // 这里采用按行分块执行，规避单次 row 过大的问题。
        const int total_in_rows = matrix_in_rows;
        const int in_cols_hw = matrix_in_cols;
        const int out_cols_hw = out_w;
        const int MAX_RESAMPLE_ROWS_PER_RUN = 15360;  // 经验值：1536 可稳定工作

        for (int row_base = 0; row_base < total_in_rows; ) {
            int cur_rows = std::min(MAX_RESAMPLE_ROWS_PER_RUN, total_in_rows - row_base);
            // 2x2 downsample/pooling 要求偶数行
            if (tc.type != RESAMPLE_TYPE_UPSAMPLE && (cur_rows & 1) != 0) {
                cur_rows -= 1;
            }
            if (cur_rows <= 0) {
                break;
            }

            uint32_t in_addr_chunk = SPM_ADDR_IN + static_cast<uint32_t>(row_base * in_cols_hw);
            int out_row_base = (tc.type == RESAMPLE_TYPE_UPSAMPLE) ? (row_base * 2) : (row_base / 2);
            uint32_t out_addr_chunk = SPM_ADDR_OUT + static_cast<uint32_t>(out_row_base * out_cols_hw);

            npu_resample_run(
                /*resample_type=*/tc.type,
                /*resample_op=*/tc.op,
                /*input_sram_addr=*/in_addr_chunk,
                /*output_sram_addr=*/out_addr_chunk,
                /*input_col_num=*/static_cast<uint16_t>(matrix_in_cols - 1),
                /*input_row_num=*/static_cast<uint16_t>(cur_rows - 1)
            );

            if (is_debug_case) {
                std::cout << "[Debug] RESAMPLE chunk: row_base=" << row_base
                          << ", cur_rows=" << cur_rows
                          << ", in_addr=0x" << std::hex << in_addr_chunk
                          << ", out_addr=0x" << out_addr_chunk << std::dec << "\n";
            }

            row_base += cur_rows;
        }
        
        // 计时结束
        auto t_end = std::chrono::high_resolution_clock::now();
        double latency_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();
        latencies.push_back(latency_us);
        
        if (latency_us < stats.min_latency_us) stats.min_latency_us = latency_us;
        if (latency_us > stats.max_latency_us) stats.max_latency_us = latency_us;
        
        // MVOUT: 将结果从 SPM 搬出到 Host
        // 输出格式: row_num = 通道数-1, col_num = H*W-1
        int mvout_row_num = out_h - 1;
        int mvout_col_num = out_w - 1;
        int mvout_stride = out_w;
        
        npu_dma_mvout(
            /*host_ptr=*/host_out,
            /*sram_addr=*/SPM_ADDR_OUT,
            /*col_num=*/static_cast<uint32_t>(mvout_col_num),
            /*row_num=*/static_cast<uint32_t>(mvout_row_num),
            /*sram_stride=*/static_cast<uint16_t>(mvout_stride),
            /*dram_stride=*/static_cast<uint32_t>(mvout_stride),
            /*precision=*/0,  // 与日志一致
            /*output_type=*/0,
            /*source=*/0,
            /*is_quant=*/false,
            /*quant_zero=*/0,
            /*quant_scale=*/0,
            /*quant_shift=*/0
        );
        if (is_debug_case) {
            std::cout << "[Debug] MVOUT done: SPM_ADDR_OUT=0x" << std::hex << SPM_ADDR_OUT
                      << std::dec << ", col_num=" << mvout_col_num 
                      << ", row_num=" << mvout_row_num 
                      << ", stride=" << mvout_stride << "\n";
        }
        
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
            if (is_debug_case) {
                std::cout << "  [Debug] First mismatches (up to 16):\n";
                std::cout << "    idx | hw | sw | diff\n";
                std::cout << "    ----+----+----+-----\n";
                int printed = 0;
                size_t first_mismatch_idx = output_size;
                for (size_t i = 0; i < output_size && printed < 16; ++i) {
                    int diff = static_cast<int>(host_out[i]) - static_cast<int>(ref_out[i]);
                    if (std::abs(diff) > MAX_DIFF) {
                        if (first_mismatch_idx == output_size) first_mismatch_idx = i;
                        std::cout << "    " << std::setw(3) << i
                                  << " | " << std::setw(3) << static_cast<int>(host_out[i])
                                  << " | " << std::setw(3) << static_cast<int>(ref_out[i])
                                  << " | " << std::setw(4) << diff << "\n";
                        printed++;
                    }
                }
                if (printed == 0) {
                    std::cout << "    (none)\n";
                }

                // 简单定位：检查是否出现“整体右移一位”的模式
                if (first_mismatch_idx < output_size && first_mismatch_idx > 0) {
                    size_t check_n = std::min<size_t>(64, output_size - first_mismatch_idx);
                    size_t shift_hit = 0;
                    for (size_t k = 0; k < check_n; ++k) {
                        size_t i = first_mismatch_idx + k;
                        if (i > 0 && host_out[i] == ref_out[i - 1]) {
                            shift_hit++;
                        }
                    }
                    std::cout << "  [Debug] ShiftCheck(first_mismatch=" << first_mismatch_idx
                              << ", window=" << check_n
                              << "): hw[i]==sw[i-1] hits=" << shift_hit << "\n";
                }
            }
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
    
    // 打印样本数据
    std::cout << "\n[Sample] Input (first 2x2 window):\n";
    if (matrix_in_rows >= 2 && matrix_in_cols >= 2) {
        std::cout << "  [" << std::setw(4) << static_cast<int>(host_in[0]) 
                  << " " << std::setw(4) << static_cast<int>(host_in[1]) << "]\n";
        std::cout << "  [" << std::setw(4) << static_cast<int>(host_in[matrix_in_cols]) 
                  << " " << std::setw(4) << static_cast<int>(host_in[matrix_in_cols + 1]) << "]\n";
    }
    
    std::cout << "[Sample] Output (first element): " 
              << static_cast<int>(host_out[0]) << "\n";
    std::cout << "[Sample] Reference (first element): " 
              << static_cast<int>(ref_out[0]) << "\n";
    if (is_debug_case) {
        std::cout << "[Debug] Input tail elements:\n";
        if (input_size >= 4) {
            std::cout << "  last4: "
                      << static_cast<int>(host_in[input_size - 4]) << ", "
                      << static_cast<int>(host_in[input_size - 3]) << ", "
                      << static_cast<int>(host_in[input_size - 2]) << ", "
                      << static_cast<int>(host_in[input_size - 1]) << "\n";
        }
        std::cout << "[Debug] Output head/tail elements:\n";
        if (output_size >= 4) {
            std::cout << "  head4: "
                      << static_cast<int>(host_out[0]) << ", "
                      << static_cast<int>(host_out[1]) << ", "
                      << static_cast<int>(host_out[2]) << ", "
                      << static_cast<int>(host_out[3]) << "\n";
            std::cout << "  tail4: "
                      << static_cast<int>(host_out[output_size - 4]) << ", "
                      << static_cast<int>(host_out[output_size - 3]) << ", "
                      << static_cast<int>(host_out[output_size - 2]) << ", "
                      << static_cast<int>(host_out[output_size - 1]) << "\n";
        }
    //     std::cout << "\n[Debug] HW Output (" << out_rows << " x " << out_cols << "):\n";
    //     for (int r = 0; r < out_rows; ++r) {
    //         std::cout << "  ";
    //         for (int c = 0; c < out_cols; ++c) {
    //             size_t idx = static_cast<size_t>(r) * out_cols + c;
    //             std::cout << std::setw(4) << static_cast<int>(host_out[idx]);
    //         }
    //         std::cout << "\n";
    //     }
    //     std::cout << "\n[Debug] SW Reference (" << out_rows << " x " << out_cols << "):\n";
    //     for (int r = 0; r < out_rows; ++r) {
    //         std::cout << "  ";
    //         for (int c = 0; c < out_cols; ++c) {
    //             size_t idx = static_cast<size_t>(r) * out_cols + c;
    //             std::cout << std::setw(4) << static_cast<int>(ref_out[idx]);
    //         }
    //         std::cout << "\n";
    //     }
    }
    
    // 打印前几个输出结果对比
    int print_count = std::min(8, static_cast<int>(output_size));
    std::cout << "\n[Sample] First " << print_count << " output elements:\n";
    std::cout << "  idx | HW out | SW ref | diff\n";
    std::cout << "  ----+--------+--------+------\n";
    for (int i = 0; i < print_count; ++i) {
        int diff = static_cast<int>(host_out[i]) - static_cast<int>(ref_out[i]);
        std::cout << "  " << std::setw(3) << i 
                  << " | " << std::setw(6) << static_cast<int>(host_out[i])
                  << " | " << std::setw(6) << static_cast<int>(ref_out[i])
                  << " | " << std::setw(4) << diff << "\n";
    }
    
    // 打印总结
    std::cout << "\n[Result Summary]\n";
    std::cout << "  Reliability: " << reliability_pass << "/" << num_runs << " runs passed\n";
    std::cout << "  Correctness: " << (output_size - stats.mismatches) << "/" << output_size 
              << " elements match (max_diff=" << stats.max_diff << ")\n";
    std::cout << "  Latency: avg=" << std::fixed << std::setprecision(2) << stats.avg_latency_us 
              << " us, min=" << stats.min_latency_us 
              << " us, max=" << stats.max_latency_us << " us\n";
    
    if (tc.test_latency) {
        double throughput_mb = (input_size + output_size) / stats.avg_latency_us;
        std::cout << "  Throughput: " << std::fixed << std::setprecision(2) 
                  << throughput_mb << " MB/s\n";
    }
    
    return stats;
}

// ==========================================
// Main
// ==========================================

int main() {
    std::cout << "=========================================\n";
    std::cout << "   NPU Resample Test (Runtime API)\n";
    std::cout << "=========================================\n";
    std::cout << "Tests: 2x2 Resample/Pooling (stride=2)\n";
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
    
    // 注意: 复现 upsample 超时 case
    std::vector<ResampleCase> test_cases = {
        // 目标复现（与日志一致）:
        // mvin: col_num=54, row_num=3519, stride=55
        // resample: type=0, op=0, input_col_num=54, input_row_num=3519
        // mvout: col_num=27, row_num=1791, stride=28
        {"Downsample_RowCol_H3520_W55_Repro", 1, 3520, 56, RESAMPLE_TYPE_DOWNSAMPLE, RESAMPLE_OP_MAX, true, true},

        // 定位用对照组（只改一个维度）
        // {"Downsample_RowCol_H3520_W56_EvenW", 1, 3520, 56, RESAMPLE_TYPE_DOWNSAMPLE, RESAMPLE_OP_MAX, true, true},
        // {"Downsample_RowCol_H3520_W54_EvenW", 1, 3520, 54, RESAMPLE_TYPE_DOWNSAMPLE, RESAMPLE_OP_MAX, true, true},
        // {"Downsample_RowCol_H2048_W55_LowH",  1, 2048, 55, RESAMPLE_TYPE_DOWNSAMPLE, RESAMPLE_OP_MAX, true, true},
        
        // 基础功能测试 - Max Pooling (已注释)
        // {"MaxPool_C1_H4_W4",      1,   4,   4, RESAMPLE_TYPE_POOLING, RESAMPLE_OP_MAX, false, false},
        // {"MaxPool_C1_H8_W8",      1,   8,   8, RESAMPLE_TYPE_POOLING, RESAMPLE_OP_MAX, false, false},
        // {"MaxPool_C1_H16_W16",    1,  16,  16, RESAMPLE_TYPE_POOLING, RESAMPLE_OP_MAX, true,  false},
        
        // // 非方形矩阵
        // {"MaxPool_C1_H8_W16",     1,   8,  16, RESAMPLE_TYPE_POOLING, RESAMPLE_OP_MAX, true,  false},
        // {"MaxPool_C1_H16_W8",     1,  16,   8, RESAMPLE_TYPE_POOLING, RESAMPLE_OP_MAX, true,  false},
        
        // // 中等规模
        // {"MaxPool_C1_H32_W32",    1,  32,  32, RESAMPLE_TYPE_POOLING, RESAMPLE_OP_MAX, true,  false},
        // {"MaxPool_C1_H64_W64",    1,  64,  64, RESAMPLE_TYPE_POOLING, RESAMPLE_OP_MAX, true,  true},
        
        // // 大规模测试 (延迟测试) (已注释)
        // {"MaxPool_C1_H130_W130",  1, 130, 130, RESAMPLE_TYPE_POOLING, RESAMPLE_OP_MAX, true,  true},
        // {"MaxPool_C1_H200_W200",  1, 200, 200, RESAMPLE_TYPE_POOLING, RESAMPLE_OP_MAX, true,  true},
        // {"MaxPool_C1_H256_W200",  1, 256, 200, RESAMPLE_TYPE_POOLING, RESAMPLE_OP_MAX, true,  true},
        // {"MaxPool_C1_H256_W256",  1, 256, 256, RESAMPLE_TYPE_POOLING, RESAMPLE_OP_MAX, true,  true},
        // {"MaxPool_C1_H300_W300",  1, 300, 300, RESAMPLE_TYPE_POOLING, RESAMPLE_OP_MAX, true,  true},
        
        // 下采样模式测试 (与 pooling 类似，使用 max)
        // {"Downsample_C1_H32_W32", 1,  32,  32, RESAMPLE_TYPE_DOWNSAMPLE, RESAMPLE_OP_MAX, true, false},
        
        // Average Pooling 测试 (如果硬件支持)
        // {"AvgPool_C1_H32_W32",    1,  32,  32, RESAMPLE_TYPE_POOLING, RESAMPLE_OP_AVG, true, false},
    };
    
    // 分配 Host 缓冲区 (按 test_cases 动态计算最大输入/输出)
    size_t max_input_size = 0;
    size_t max_output_size = 0;
    for (const auto& tc : test_cases) {
        const bool matrix_layout_mode = (tc.name == "Downsample_Row3519_Col54_Repro");
        const int matrix_rows = matrix_layout_mode ? tc.rows : (tc.channels * tc.rows);
        const int matrix_cols = tc.cols;

        size_t in_sz = static_cast<size_t>(matrix_rows) * static_cast<size_t>(matrix_cols);
        int oh = (tc.type == RESAMPLE_TYPE_UPSAMPLE) ? (matrix_rows * 2) : (matrix_rows / 2);
        int ow = (tc.type == RESAMPLE_TYPE_UPSAMPLE) ? (matrix_cols * 2) : ((matrix_cols + 1) / 2);
        size_t out_sz = static_cast<size_t>(oh) * static_cast<size_t>(ow);
        if (in_sz > max_input_size) max_input_size = in_sz;
        if (out_sz > max_output_size) max_output_size = out_sz;
    }

    int8_t* host_in  = static_cast<int8_t*>(npu_mem_alloc(max_input_size));
    int8_t* host_out = static_cast<int8_t*>(npu_mem_alloc(max_output_size));
    
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
        TestStats stats = run_resample_case(tc, host_in, host_out);
        
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
