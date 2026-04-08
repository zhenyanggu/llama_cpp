#include "../npu_runtime.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
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

static float compute_symmetric_scale(const std::vector<float>& values) {
    float max_abs = 0.0f;
    for (float v : values) {
        float a = std::fabs(v);
        if (a > max_abs) max_abs = a;
    }
    if (max_abs < 1e-8f) {
        return 1.0f / 127.0f;
    }
    return max_abs / 127.0f;
}

static std::vector<float> layernorm_ref_matrix(
    const std::vector<float>& x,
    size_t rows,
    size_t cols,
    float eps = 1e-5f
) {
    std::vector<float> y(x.size(), 0.0f);
    if (rows == 0 || cols == 0 || x.size() != rows * cols) return y;

    for (size_t r = 0; r < rows; ++r) {
        float mean = 0.0f;
        float var = 0.0f;
        for (size_t c = 0; c < cols; ++c) {
            float v = x[r * cols + c];
            mean += v;
        }
        mean /= static_cast<float>(cols);

        for (size_t c = 0; c < cols; ++c) {
            float v = x[r * cols + c];
            float diff = v - mean;
            var += diff * diff;
        }
        var /= static_cast<float>(cols);
        float inv_std = 1.0f / std::sqrt(var + eps);

        for (size_t c = 0; c < cols; ++c) {
            float v = x[r * cols + c];
            y[r * cols + c] = (v - mean) * inv_std;
        }
    }
    return y;
}

struct TestStats {
    int total = 0;
    int pass = 0;
    int exact = 0;
    int max_abs_diff = 0;
};

static void run_layernorm_case(
    int8_t* host_in,
    int8_t* host_out,
    const std::vector<float>& input_values,
    const std::string& test_name,
    size_t rows,
    size_t cols,
    float input_scale,
    float output_scale,
    uint32_t input_sram_addr,
    uint32_t output_sram_addr,
    TestStats& stats,
    int case_index,
    int log_limit,
    std::ofstream& mismatch_log,
    int& mismatch_count
) {
    const uint32_t input_zeropoint = 0;
    const uint16_t output_zeropoint = 0;

    FixedPointParams in_fp = getFixedPointParams(input_scale);
    FixedPointParams out_fp = getFixedPointParams(1.0 / output_scale);

    const size_t n = input_values.size();
    if (rows == 0 || cols == 0 || n != rows * cols || cols > 1024 || n > 1024) {
        return;
    }

    std::memset(host_in, 0, n);
    std::memset(host_out, 0, n);

    for (size_t i = 0; i < n; ++i) {
        host_in[i] = quantize_symmetric(input_values[i], input_scale);
    }

    npu_dma_mvin(
        /*host_ptr=*/host_in,
        /*sram_addr=*/input_sram_addr,
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

    npu_sfu_run(
        /*op_type=*/SFU_OP_LAYERNORM,
        /*int_type=*/0,
        /*is_quant=*/true,
        /*input_sram_addr=*/input_sram_addr,
        /*input_col_num=*/static_cast<uint16_t>(cols - 1),
        /*input_row_num=*/static_cast<uint16_t>(rows - 1),
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
        /*col_num=*/static_cast<uint16_t>(cols - 1),
        /*row_num=*/static_cast<uint16_t>(rows - 1),
        /*sram_stride=*/static_cast<uint16_t>(cols),
        /*dram_stride=*/static_cast<uint16_t>(cols),
        /*precision=*/1,
        /*output_type=*/0,
        /*source=*/0,
        /*is_quant=*/false,
        /*quant_zero=*/0,
        /*quant_scale=*/0,
        /*quant_shift=*/0
    );

    std::vector<float> ref = layernorm_ref_matrix(input_values, rows, cols);

    for (size_t i = 0; i < n; ++i) {
        int8_t q_out = host_out[i];
        int8_t q_ref = quantize_symmetric(ref[i], output_scale);

        int diff = std::abs(static_cast<int>(q_out) - static_cast<int>(q_ref));
        bool ok = diff <= 4;

        stats.total++;
        if (ok) stats.pass++;
        if (diff == 0) stats.exact++;
        if (diff > stats.max_abs_diff) stats.max_abs_diff = diff;

        int8_t q_in = host_in[i];

        mismatch_log << test_name << "," << case_index << "," << rows << "," << cols << "," << i << ","
                     << static_cast<int>(q_in) << ","
                     << std::setprecision(9) << input_values[i] << ","
                     << input_scale << "," << output_scale << ","
                     << static_cast<int>(q_out) << ","
                     << static_cast<int>(q_ref) << ","
                     << diff << ","
                     << in_fp.scale << "," << in_fp.shift << ","
                     << out_fp.scale << "," << out_fp.shift << ","
                     << std::setprecision(9) << dequantize_symmetric(q_out, output_scale) << ","
                     << ref[i] << ","
                     << (ok ? "pass" : "fail")
                     << "\n";

        if (!ok && mismatch_count < log_limit) {
            std::cout << "[Mismatch] test=" << test_name
                      << " case=" << case_index
                      << " rows=" << rows
                      << " cols=" << cols
                      << " idx=" << i
                      << " in=" << input_values[i]
                      << " in_scale=" << input_scale
                      << " out_scale=" << output_scale
                      << " q_out=" << static_cast<int>(q_out)
                      << " q_ref=" << static_cast<int>(q_ref)
                      << " diff=" << diff << std::endl;
            mismatch_count++;
        }
    }
}

static std::vector<float> make_pattern_vector(size_t n, int pattern_id) {
    std::vector<float> v(n, 0.0f);
    if (n == 0) return v;

    switch (pattern_id) {
        case 0: // all zeros
            std::fill(v.begin(), v.end(), 0.0f);
            break;
        case 1: // all same positive
            std::fill(v.begin(), v.end(), 1.0f);
            break;
        case 2: // all same negative
            std::fill(v.begin(), v.end(), -1.0f);
            break;
        case 3: // ramp
            for (size_t i = 0; i < n; ++i) v[i] = static_cast<float>(i) / static_cast<float>(n);
            break;
        case 4: // mild alternating
            for (size_t i = 0; i < n; ++i) v[i] = (i % 2 == 0) ? 1.0f : -1.0f;
            break;
        case 5: // one hot
            v[n / 2] = 4.0f;
            break;
        default:
            for (size_t i = 0; i < n; ++i) v[i] = 0.0f;
            break;
    }

    return v;
}

static TestStats run_grid_tests(int8_t* host_in, int8_t* host_out, std::ofstream& mismatch_log, int& mismatch_count) {
    std::cout << "===== LayerNorm Grid Test (Symmetric Quant) =====" << std::endl;

    const std::vector<size_t> cols_list = {16, 32, 64, 128, 256};
    const std::vector<size_t> rows_list = {1, 2, 4};

    const uint32_t input_sram_addr = 0x0;
    const uint32_t output_sram_addr = 0x2000;

    TestStats stats;
    int case_index = 0;

    for (size_t rows : rows_list) {
        for (size_t cols : cols_list) {
            size_t n = rows * cols;
            if (n == 0 || n > 1024) continue;

            for (int pattern_id = 0; pattern_id <= 5; ++pattern_id) {
                std::vector<float> vec = make_pattern_vector(n, pattern_id);
                float input_scale = compute_symmetric_scale(vec);
                std::vector<float> ref = layernorm_ref_matrix(vec, rows, cols);
                float output_scale = compute_symmetric_scale(ref);

                run_layernorm_case(
                    host_in,
                    host_out,
                    vec,
                    "grid",
                    rows,
                    cols,
                    input_scale,
                    output_scale,
                    input_sram_addr,
                    output_sram_addr,
                    stats,
                    case_index,
                    10,
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
    return stats;
}

static TestStats run_random_tests(int8_t* host_in, int8_t* host_out, std::ofstream& mismatch_log, int& mismatch_count) {
    std::cout << "===== LayerNorm Random Test (Symmetric Quant) =====" << std::endl;

    const int num_cases = 80;
    const uint32_t input_sram_addr = 0x0;
    const uint32_t output_sram_addr = 0x2000;

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::uniform_int_distribution<int> dist_rows(1, 8);
    std::uniform_int_distribution<int> dist_cols(8, 256);

    TestStats stats;

    for (int i = 0; i < num_cases; ++i) {
        size_t rows = static_cast<size_t>(dist_rows(rng));
        size_t cols = static_cast<size_t>(dist_cols(rng));
        size_t n = rows * cols;
        if (n == 0 || n > 1024) {
            --i;
            continue;
        }

        std::vector<float> vec(n);
        for (float& v : vec) v = dist(rng);

        std::vector<float> ref = layernorm_ref_matrix(vec, rows, cols);
        float input_scale = compute_symmetric_scale(vec);
        float output_scale = compute_symmetric_scale(ref);

        run_layernorm_case(
            host_in,
            host_out,
            vec,
            "random",
            rows,
            cols,
            input_scale,
            output_scale,
            input_sram_addr,
            output_sram_addr,
            stats,
            i,
            10,
            mismatch_log,
            mismatch_count
        );
    }

    std::cout << "[Random] Total=" << stats.total
              << " Pass=" << stats.pass
              << " Exact=" << stats.exact
              << " MaxAbsDiff=" << stats.max_abs_diff << std::endl;
    return stats;
}

static TestStats run_transformer_tests(int8_t* host_in, int8_t* host_out, std::ofstream& mismatch_log, int& mismatch_count) {
    std::cout << "===== LayerNorm Transformer Shapes =====" << std::endl;

    struct Shape {
        size_t rows;
        size_t cols;
        const char* name;
    };

    const std::vector<Shape> shapes = {
        {1, 768, "bert_768"},
        // {1, 1024, "gpt_1024"},
        {4, 256, "attn_4x256"},
        {8, 128, "attn_8x128"}
    };

    const uint32_t input_sram_addr = 0x0;
    const uint32_t output_sram_addr = 0x2000;

    std::mt19937 rng(2024);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    TestStats stats;
    int case_index = 0;

    for (const auto& sh : shapes) {
        size_t n = sh.rows * sh.cols;
        if (n == 0 || n > 1024 || sh.cols > 1024) continue;

        std::vector<float> vec(n);
        for (float& v : vec) v = dist(rng);

        std::vector<float> ref = layernorm_ref_matrix(vec, sh.rows, sh.cols);
        float input_scale = compute_symmetric_scale(vec);
        float output_scale = compute_symmetric_scale(ref);

        run_layernorm_case(
            host_in,
            host_out,
            vec,
            sh.name,
            sh.rows,
            sh.cols,
            input_scale,
            output_scale,
            input_sram_addr,
            output_sram_addr,
            stats,
            case_index,
            10,
            mismatch_log,
            mismatch_count
        );
        case_index++;
    }

    std::cout << "[Transformer] Total=" << stats.total
              << " Pass=" << stats.pass
              << " Exact=" << stats.exact
              << " MaxAbsDiff=" << stats.max_abs_diff << std::endl;
    return stats;
}

int main() {
    std::cout << "===== NPU LayerNorm Test (Symmetric Quant) =====" << std::endl;

    if (npu_init() != 0) {
        std::cerr << "Error: NPU init failed. Check kernel driver." << std::endl;
        return -1;
    }

    int8_t* host_in = static_cast<int8_t*>(npu_mem_alloc(1024));
    int8_t* host_out = static_cast<int8_t*>(npu_mem_alloc(1024));
    if (!host_in || !host_out) {
        std::cerr << "Error: NPU memory alloc failed." << std::endl;
        npu_destroy();
        return -1;
    }

    std::ofstream mismatch_log("layernorm_mismatch_cases.csv", std::ios::out | std::ios::trunc);
    mismatch_log << "test,case,rows,cols,idx,q_in,input_value,input_scale,output_scale,q_out,q_ref,diff,in_scale,in_shift,out_scale,out_shift,npu_float,ref_float,status\n";
    int mismatch_count = 0;

    TestStats grid_stats = run_grid_tests(host_in, host_out, mismatch_log, mismatch_count);
    TestStats rand_stats = run_random_tests(host_in, host_out, mismatch_log, mismatch_count);
    TestStats trans_stats = run_transformer_tests(host_in, host_out, mismatch_log, mismatch_count);

    mismatch_log.flush();
    mismatch_log.close();

    int total = grid_stats.total + rand_stats.total + trans_stats.total;
    int pass = grid_stats.pass + rand_stats.pass + trans_stats.pass;
    int fail = total - pass;
    std::cout << "Mismatch cases saved to layernorm_mismatch_cases.csv" << std::endl;
    std::cout << "Summary: total=" << total << ", pass=" << pass << ", fail=" << fail << std::endl;

    npu_mem_free(host_in);
    npu_mem_free(host_out);
    npu_destroy();

    return 0;
}
