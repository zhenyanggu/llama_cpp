#include "../npu_runtime.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <cmath>

using Clock = std::chrono::steady_clock;

static uint64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               Clock::now().time_since_epoch())
        .count();
}

// 结构体存储详细测量结果
struct LatencyResult {
    const char* name;
    double min_ns;
    double max_ns;
    double avg_ns;
    double median_ns;
    int iterations;
};

static std::vector<LatencyResult> g_results;

template <typename Func>
static void measure_latency(const char* name, int iterations, Func&& func) {
    if (iterations <= 0) return;

    std::vector<uint64_t> samples;
    samples.reserve(iterations);

    // Warmup (多次预热)
    for (int i = 0; i < 3; ++i) {
        func();
    }

    // 收集每次迭代的样本
    for (int i = 0; i < iterations; ++i) {
        uint64_t start = now_ns();
        func();
        uint64_t end = now_ns();
        samples.push_back(end - start);
    }

    // 统计分析
    std::sort(samples.begin(), samples.end());
    
    double min_ns = static_cast<double>(samples.front());
    double max_ns = static_cast<double>(samples.back());
    double median_ns = static_cast<double>(samples[iterations / 2]);
    
    uint64_t sum = 0;
    for (auto s : samples) sum += s;
    double avg_ns = static_cast<double>(sum) / static_cast<double>(iterations);

    // 保存结果
    g_results.push_back({name, min_ns, max_ns, avg_ns, median_ns, iterations});

    std::cout << "[Latency] " << std::setw(20) << std::left << name 
              << ": avg=" << std::setw(10) << avg_ns 
              << " min=" << std::setw(10) << min_ns 
              << " max=" << std::setw(10) << max_ns 
              << " median=" << median_ns << " ns"
              << " (n=" << iterations << ")" << std::endl;
}

int main(int argc, char** argv) {
    int iterations = 10;
    if (argc >= 2) {
        iterations = std::atoi(argv[1]);
    }

    if (npu_init() != 0) {
        std::cerr << "npu_init failed" << std::endl;
        return -1;
    }

    npu_reset();

    // Allocate host buffers in NPU DDR region
    const uint16_t cols = 32;
    const uint16_t rows = 1;
    const uint16_t stride = cols;
    const size_t buf_size = 4096;

    void* host_a = npu_mem_alloc(buf_size);
    void* host_b = npu_mem_alloc(buf_size);
    void* host_out = npu_mem_alloc(buf_size);

    if (!host_a || !host_b || !host_out) {
        std::cerr << "npu_mem_alloc failed" << std::endl;
        npu_destroy();
        return -1;
    }

    std::memset(host_a, 0x11, buf_size);
    std::memset(host_b, 0x22, buf_size);
    std::memset(host_out, 0x00, buf_size);

    const uint32_t sram_a = 0x0000;
    const uint32_t sram_b = 0x1000;
    const uint32_t sram_out = 0x2000;
    const uint32_t sram_bias = 0x3000;
    const uint32_t sram_output = 0x4000;

    // Baseline (empty loop) to estimate timing + loop overhead
    measure_latency("baseline_empty", iterations, [&]() {
        asm volatile("" ::: "memory");
    });

    // API-only overhead (test stubs; no hardware work)
    measure_latency("npu_dma_mvin_test", iterations, [&]() {
        npu_dma_mvin_test(
            /*host_ptr=*/host_a,
            /*sram_addr=*/sram_a,
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
            /*quant_shift=*/0);
    });

    measure_latency("npu_dma_mvout_test", iterations, [&]() {
        npu_dma_mvout_test(
            /*host_ptr=*/host_out,
            /*sram_addr=*/sram_a,
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
            /*quant_shift=*/0);
    });

    // MVIN latency
    measure_latency("npu_dma_mvin", iterations, [&]() {
        npu_dma_mvin(
            /*host_ptr=*/host_a,
            /*sram_addr=*/sram_a,
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
            /*quant_shift=*/0);
    });

    // MVOUT latency (after a MVIN to ensure data is ready)
    npu_dma_mvin(
        /*host_ptr=*/host_a,
        /*sram_addr=*/sram_a,
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
    measure_latency("npu_dma_mvout", iterations, [&]() {
        npu_dma_mvout(
            /*host_ptr=*/host_out,
            /*sram_addr=*/sram_a,
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
            /*quant_shift=*/0);
    });

    // SFU latency (GELU)
    npu_dma_mvin(
        /*host_ptr=*/host_a,
        /*sram_addr=*/sram_a,
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
    measure_latency("npu_sfu_run", iterations, [&]() {
        npu_sfu_run(
            /*op_type=*/SFU_OP_GELU,
            /*int_type=*/0,
            /*is_quant=*/false,
            /*input_sram_addr=*/sram_a,
            /*input_col_num=*/cols - 1,
            /*input_row_num=*/rows - 1,
            /*output_sram_addr=*/sram_out,
            /*input_zeropoint=*/0,
            /*output_zeropoint=*/0,
            /*input_scale=*/0,
            /*input_scale_shift=*/0,
            /*output_scale=*/0,
            /*output_scale_shift=*/0);
    });

    // GEMM latency (very small shape)
    npu_dma_mvin(
        /*host_ptr=*/host_a,
        /*sram_addr=*/sram_a,
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
    npu_dma_mvin(
        /*host_ptr=*/host_b,
        /*sram_addr=*/sram_b,
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
    measure_latency("npu_gemm_run", iterations, [&]() {
        npu_gemm_run(
            /*dataflow=*/1,
            /*int_type=*/0,
            /*optype=*/0,
            /*accout_dest=*/0,
            /*input_a_zeropoint=*/0,
            /*input_b_zeropoint=*/0,
            /*output_zeropoint=*/0,
            /*output_scale=*/0,
            /*output_scaleshift=*/0,
            /*biaspsum_addr=*/sram_bias,
            /*biaspsum_stride=*/stride,
            /*biaspsum_width=*/1,
            /*biaspsum_height=*/1,
            /*output_addr=*/sram_output,
            /*output_stride=*/stride,
            /*isaccu=*/0,
            /*relu=*/0,
            /*relu_type=*/0,
            /*is_bias=*/0,
            /*input_a_addr=*/sram_a,
            /*input_a_col_num=*/cols - 1,
            /*input_a_row_num=*/rows - 1,
            /*input_a_stride=*/stride,
            /*input_b_addr=*/sram_b,
            /*input_b_col_num=*/cols - 1,
            /*input_b_row_num=*/rows - 1,
            /*input_b_stride=*/stride);
    });

    // CONV latency (1x1 kernel, minimal shape)
    npu_dma_mvin(
        /*host_ptr=*/host_a,
        /*sram_addr=*/sram_a,
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
    npu_dma_mvin(
        /*host_ptr=*/host_b,
        /*sram_addr=*/sram_b,
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
    measure_latency("npu_conv_run", iterations, [&]() {
        npu_conv_run(
            /*pad_top=*/0,
            /*pad_bottom=*/0,
            /*pad_left=*/0,
            /*pad_right=*/0,
            /*pad_mode=*/0,
            /*weight_shape_m1=*/0,
            /*weight_stride_m1=*/0,
            /*weight_dilation_m1=*/0,
            /*is_group_conv=*/0,
            /*int_type=*/0,
            /*op_type=*/1,
            /*dataflow_mode=*/0,
            /*accout_dest=*/0,
            /*input_a_zeropoint=*/0,
            /*input_b_zeropoint=*/0,
            /*input_a_addr=*/sram_a,
            /*input_a_col_num_m1=*/0,
            /*input_a_row_num_m1=*/0,
            /*input_a_stride=*/stride,
            /*input_b_addr=*/sram_b,
            /*input_b_col_num_m1=*/0,
            /*input_b_row_num_m1=*/0,
            /*input_b_stride=*/stride,
            /*biaspsum_width=*/1,
            /*biaspsum_height=*/1,
            /*biaspsum_addr=*/sram_bias,
            /*biaspsum_stride=*/stride,
            /*output_addr=*/sram_output,
            /*output_stride=*/stride,
            /*is_accumulate=*/0,
            /*relu_enable=*/0,
            /*relu_type=*/0,
            /*is_bias=*/0,
            /*output_zeropoint=*/0,
            /*quant_scale=*/0,
            /*quant_scaleshift=*/0);
    });

    npu_mem_free(host_a);
    npu_mem_free(host_b);
    npu_mem_free(host_out);
    npu_destroy();

    // 打印汇总结果
    std::cout << "\n========== Latency Summary ==========\n";
    std::cout << std::setw(22) << std::left << "Test"
              << std::setw(12) << "Avg(ns)"
              << std::setw(12) << "Min(ns)"
              << std::setw(12) << "Median(ns)" << std::endl;
    std::cout << std::string(58, '-') << std::endl;
    
    double baseline_avg = 0;
    for (const auto& r : g_results) {
        if (std::strcmp(r.name, "baseline_empty") == 0) {
            baseline_avg = r.avg_ns;
        }
        std::cout << std::setw(22) << std::left << r.name
                  << std::setw(12) << std::fixed << std::setprecision(1) << r.avg_ns
                  << std::setw(12) << r.min_ns
                  << std::setw(12) << r.median_ns << std::endl;
    }
    
    // 计算校正后的延迟 (减去 baseline)
    if (baseline_avg > 0) {
        std::cout << "\n========== Corrected Latency (- baseline) ==========\n";
        for (const auto& r : g_results) {
            if (std::strcmp(r.name, "baseline_empty") == 0) continue;
            double corrected = r.avg_ns - baseline_avg;
            std::cout << std::setw(22) << std::left << r.name
                      << std::setw(12) << std::fixed << std::setprecision(1) << corrected << " ns" << std::endl;
        }
    }

    return 0;
}
