#include "../npu_runtime.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

struct FixedPointParams {
    int16_t scale;
    int16_t shift;
};

static FixedPointParams getFixedPointParams(double scale) {
    if (std::abs(scale) < 1e-8) return {0, 0};

    int exponent = 0;
    double mantissa = std::frexp(scale, &exponent);

    double mantissa_scaled = std::round(mantissa * 32768.0);

    if (mantissa_scaled >= 32768.0) {
        mantissa_scaled /= 2.0;
        exponent += 1;
    }

    return {static_cast<int16_t>(mantissa_scaled), static_cast<int16_t>(exponent - 15)};
}

static float gelu_ref(float x) {
    const float kInvSqrt2 = 0.70710678118f;
    return 0.5f * x * (1.0f + std::erf(x * kInvSqrt2));
}

static int8_t quantize_symmetric(float x, float scale) {
    if (scale <= 0.0f) return 0;
    float q = std::round(x / scale);
    if (q > 127.0f) q = 127.0f;
    if (q < -128.0f) q = -128.0f;
    return static_cast<int8_t>(q);
}

static float dequantize_symmetric(int8_t q, float scale) {
    return static_cast<float>(q) * scale;
}

struct TestStats {
    int total = 0;
    int pass = 0;
    int exact = 0;
    int max_abs_diff = 0;
};

static bool run_gelu_case(
    int8_t* host_in,
    int8_t* host_out,
    float input_value,
    float input_scale,
    float output_scale,
    const uint32_t input_sram_addr,
    const uint32_t output_sram_addr,
    TestStats& stats,
    int case_index,
    int log_limit,
    const std::string& test_name,
    std::ofstream& mismatch_log,
    int& mismatch_count
) {
    if (input_scale <= 0.0f || output_scale <= 0.0f) {
        return false;
    }

    const uint32_t input_zeropoint = 0;
    const uint16_t output_zeropoint = 0;

    FixedPointParams in_fp = getFixedPointParams(input_scale);
    FixedPointParams out_fp = getFixedPointParams(1.0 / output_scale);

    std::memset(host_in, 0, 64);
    std::memset(host_out, 0, 64);

    int8_t q_in = quantize_symmetric(input_value, input_scale);
    host_in[0] = q_in;

    npu_dma_mvin(
        /*host_ptr=*/host_in,
        /*sram_addr=*/input_sram_addr,
        /*col_num=*/0,
        /*row_num=*/0,
        /*sram_stride=*/0,
        /*dram_stride=*/0,
        /*precision=*/1,
        /*input_type=*/0,
        /*dest=*/0,
        /*is_bias=*/false,
        /*is_quant=*/false,
        /*quant_zero=*/0,
        /*quant_scale=*/0,
        /*quant_shift=*/0
    );

    npu_sfu_run(
        /*op_type=*/SFU_OP_GELU,
        /*int_type=*/0,
        /*is_quant=*/true,
        /*input_sram_addr=*/input_sram_addr,
        /*input_col_num=*/0,
        /*input_row_num=*/0,
        /*output_sram_addr=*/output_sram_addr,
        /*input_zeropoint=*/input_zeropoint,
        /*output_zeropoint=*/output_zeropoint,
        /*input_scale=*/static_cast<uint16_t>(in_fp.scale),
        /*input_scale_shift=*/static_cast<uint16_t>(in_fp.shift),
        /*output_scale=*/static_cast<uint16_t>(out_fp.scale),
        /*output_scale_shift=*/static_cast<uint16_t>(out_fp.shift)
    );

    npu_dma_mvout(
        /*host_ptr=*/host_out,
        /*sram_addr=*/output_sram_addr,
        /*col_num=*/1,
        /*row_num=*/1,
        /*sram_stride=*/1,
        /*dram_stride=*/1,
        /*precision=*/1,
        /*output_type=*/0,
        /*source=*/0,
        /*is_quant=*/false,
        /*quant_zero=*/0,
        /*quant_scale=*/0,
        /*quant_shift=*/0
    );

    int8_t q_out = host_out[0];
    int8_t ref_in_quant = quantize_symmetric(input_value, input_scale);
    float ref_in_after_dequant = dequantize_symmetric(ref_in_quant, input_scale);
    float ref = gelu_ref(ref_in_after_dequant);
    int8_t q_ref = quantize_symmetric(ref, output_scale);

    float deq_out = dequantize_symmetric(q_out, output_scale);
    float deq_ref = dequantize_symmetric(q_ref, output_scale);

    int diff = std::abs(static_cast<int>(q_out) - static_cast<int>(q_ref));
    bool ok = diff <= 1;

    stats.total++;
    if (ok) stats.pass++;
    if (diff == 0) stats.exact++;
    if (diff > stats.max_abs_diff) stats.max_abs_diff = diff;

    mismatch_log << test_name << ","
                 << case_index << ","
                 << std::setprecision(9) << input_value << ","
                 << input_scale << ","
                 << output_scale << ","
                 << static_cast<int>(q_in) << ","
                 << static_cast<int>(q_out) << ","
                 << static_cast<int>(q_ref) << ","
                 << diff << ","
                 << in_fp.scale << "," << in_fp.shift << ","
                 << out_fp.scale << "," << out_fp.shift << ","
                 << std::setprecision(9) << deq_out << ","
                 << deq_ref << ","
                 << (ok ? "pass" : "fail")
                 << "\n";

    if (!ok) {
        if (case_index < log_limit) {
            std::cout << "[Mismatch] case=" << case_index
                      << " in=" << input_value
                      << " in_scale=" << input_scale
                      << " out_scale=" << output_scale
                      << " q_in=" << static_cast<int>(q_in)
                      << " q_out=" << static_cast<int>(q_out)
                      << " q_ref=" << static_cast<int>(q_ref)
                      << " diff=" << diff << std::endl;
        }

        mismatch_count++;
    }

    return ok;
}

static void run_grid_tests(int8_t* host_in, int8_t* host_out, std::ofstream& mismatch_log, int& mismatch_count) {
    std::cout << "===== GELU Grid Test (Symmetric Quant) =====" << std::endl;

    const std::vector<float> input_scales = {0.00390625f, 0.0078125f, 0.015625f, 0.03125f, 0.05f};
    const std::vector<float> output_scales = {0.00390625f, 0.0078125f, 0.015625f, 0.03125f, 0.05f};

    const float x_min = -6.0f;
    const float x_max = 6.0f;
    const float x_step = 0.25f;

    const uint32_t input_sram_addr = 0x0;
    const uint32_t output_sram_addr = 0x100;

    TestStats stats;
    int case_index = 0;
    for (float in_scale : input_scales) {
        for (float out_scale : output_scales) {
            for (float x = x_min; x <= x_max + 1e-6f; x += x_step) {
                run_gelu_case(
                    host_in,
                    host_out,
                    x,
                    in_scale,
                    out_scale,
                    input_sram_addr,
                    output_sram_addr,
                    stats,
                    case_index,
                    10,
                    "grid",
                    mismatch_log,
                    mismatch_count
                );
                case_index++;
            }
        }
    }

    std::cout << "[Grid] Total=" << stats.total
              << " Pass=" << stats.pass
              << " Exact=" << stats.exact
              << " MaxAbsDiff=" << stats.max_abs_diff << std::endl;
}

static void run_random_tests(int8_t* host_in, int8_t* host_out, std::ofstream& mismatch_log, int& mismatch_count) {
    std::cout << "===== GELU Random Test (Symmetric Quant) =====" << std::endl;

    const int num_cases = 2000;
    const float x_min = -8.0f;
    const float x_max = 8.0f;
    const float scale_min = 0.0025f;
    const float scale_max = 0.08f;

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist_x(x_min, x_max);
    std::uniform_real_distribution<float> dist_scale(scale_min, scale_max);

    const uint32_t input_sram_addr = 0x0;
    const uint32_t output_sram_addr = 0x100;

    TestStats stats;
    for (int i = 0; i < num_cases; ++i) {
        float x = dist_x(rng);
        float in_scale = dist_scale(rng);
        float out_scale = dist_scale(rng);
        run_gelu_case(
            host_in,
            host_out,
            x,
            in_scale,
            out_scale,
            input_sram_addr,
            output_sram_addr,
            stats,
            i,
            10,
            "random",
            mismatch_log,
            mismatch_count
        );
    }

    std::cout << "[Random] Total=" << stats.total
              << " Pass=" << stats.pass
              << " Exact=" << stats.exact
              << " MaxAbsDiff=" << stats.max_abs_diff << std::endl;
}

int main() {
    std::cout << "===== NPU GELU Comprehensive Test (Symmetric Quant) =====" << std::endl;

    if (npu_init() != 0) {
        std::cerr << "Error: NPU init failed. Check kernel driver." << std::endl;
        return -1;
    }

    int8_t* host_in = static_cast<int8_t*>(npu_mem_alloc(64));
    int8_t* host_out = static_cast<int8_t*>(npu_mem_alloc(64));
    if (!host_in || !host_out) {
        std::cerr << "Error: NPU memory alloc failed." << std::endl;
        npu_destroy();
        return -1;
    }

    std::ofstream mismatch_log("gelu_mismatch_cases.csv", std::ios::out | std::ios::trunc);
    mismatch_log << "test,case,input_value,input_scale,output_scale,q_in,q_out,q_ref,diff,in_scale,in_shift,out_scale,out_shift,npu_float,ref_float,status\n";
    int mismatch_count = 0;

    run_grid_tests(host_in, host_out, mismatch_log, mismatch_count);
    run_random_tests(host_in, host_out, mismatch_log, mismatch_count);

    mismatch_log.flush();
    mismatch_log.close();

    std::cout << "Mismatch cases saved to gelu_mismatch_cases.csv, count=" << mismatch_count << std::endl;

    npu_mem_free(host_in);
    npu_mem_free(host_out);
    npu_destroy();

    return 0;
}
