#include "../npu_runtime.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// Compile-time switch: 1=use bias, 0=all bias are zeros
#ifndef USE_BIAS
#define USE_BIAS 0
#endif

// Compile-time switch: max allowed absolute diff before counting as mismatch
#ifndef MAX_DIFF
#define MAX_DIFF 2
#endif

// Total SPM byte size visible to this test.
#ifndef TEST_SPM_SIZE_BYTES
#define TEST_SPM_SIZE_BYTES (1u << 19)
#endif

struct GemmCase {
    std::string name;
    int M;
    int N;
    int K;
    bool random;
};

struct MulShift {
    int16_t scale;
    int16_t shift;
};

static MulShift pick_output_scale_shift(float M) {
    MulShift r{0, 0};
    if (M <= 0.0f) return r;

    int bestN = -1;
    int32_t bestScale = 0;

    for (int N = 0; N <= 30; N++) {
        double s = static_cast<double>(M) * static_cast<double>(1u << N);
        int32_t sc = static_cast<int32_t>(std::llround(s));
        if (sc <= 0) continue;
        if (sc > 32767) break;
        bestN = N;
        bestScale = sc;
    }
    if (bestN < 0) {
        r.scale = 1;
        r.shift = -30;
        return r;
    }
    r.scale = static_cast<int16_t>(bestScale);
    r.shift = static_cast<int16_t>(-bestN);
    return r;
}

static float calc_symmetric_scale_fp32(const std::vector<float>& x) {
    float max_abs = 0.0f;
    for (float v : x) {
        float a = std::fabs(v);
        if (a > max_abs) max_abs = a;
    }
    if (max_abs < 1e-12f) return 1.0f;
    return max_abs / 127.0f;
}

static int8_t quant_i32_to_i8(int32_t x, int16_t scale, int16_t shift) {
    int64_t v = static_cast<int64_t>(x) * static_cast<int64_t>(scale);
    if (shift < 0) {
        int s = -shift;
        if (s > 0) {
            int64_t round = 1LL << (s - 1);
            v = (v >= 0) ? (v + round) : (v - round);
        }
        v >>= s;
    } else if (shift > 0) {
        v <<= shift;
    }

    if (v > 127) v = 127;
    if (v < -128) v = -128;
    return static_cast<int8_t>(v);
}

static int8_t quant_fp32_to_i8(float x, float scale) {
    if (scale <= 0.0f) return 0;
    int32_t q = static_cast<int32_t>(std::lrint(x / scale));
    if (q > 127) q = 127;
    if (q < -128) q = -128;
    return static_cast<int8_t>(q);
}

static int32_t quant_fp32_to_i32(float x, float scale) {
    if (scale <= 0.0f) return 0;
    return static_cast<int32_t>(std::lrint(x / scale));
}

static void fill_inputs_fp32(
    std::vector<float>& A,
    std::vector<float>& B,
    std::vector<float>& Bias,
    int M,
    int N,
    int K,
    bool random,
    std::mt19937& rng
) {
    const int a_size = M * K;
    const int b_size = K * N;
    const int bias_size = N;

    A.resize(a_size);
    B.resize(b_size);
    Bias.resize(bias_size);

#if !USE_BIAS
    std::fill(Bias.begin(), Bias.end(), 0.0f);
    if (random) {
        std::normal_distribution<float> dist_a(0.0f, 1.0f);
        std::normal_distribution<float> dist_b(0.0f, 1.0f);
        for (int i = 0; i < a_size; ++i) {
            A[i] = dist_a(rng);
        }
        for (int i = 0; i < b_size; ++i) {
            B[i] = dist_b(rng);
        }
        return;
    }
    for (int i = 0; i < a_size; ++i) {
        A[i] = -1.0f + 0.01f * static_cast<float>(i % 200);
    }
    for (int i = 0; i < b_size; ++i) {
        B[i] = -1.0f + 0.01f * static_cast<float>(i % 200);
    }
    return;
#endif

    if (random) {
        std::normal_distribution<float> dist_a(0.0f, 1.0f);
        std::normal_distribution<float> dist_b(0.0f, 1.0f);
        std::normal_distribution<float> dist_bias(0.0f, 0.5f);

        for (int i = 0; i < a_size; ++i) {
            A[i] = dist_a(rng);
        }
        for (int i = 0; i < b_size; ++i) {
            B[i] = dist_b(rng);
        }
        for (int i = 0; i < bias_size; ++i) {
            Bias[i] = dist_bias(rng);
        }
        return;
    }

    for (int i = 0; i < a_size; ++i) {
        A[i] = -1.0f + 0.01f * static_cast<float>(i % 200);
    }
    for (int i = 0; i < b_size; ++i) {
        B[i] = -1.0f + 0.01f * static_cast<float>(i % 200);
    }
    for (int i = 0; i < bias_size; ++i) {
        Bias[i] = -0.5f + 0.01f * static_cast<float>(i % 100);
    }
}

static std::vector<float> reference_gemm_fp32(
    const std::vector<float>& A,
    const std::vector<float>& B,
    const std::vector<float>& Bias,
    int M,
    int N,
    int K
) {
    std::vector<float> out(M * N, 0.0f);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            // Bias is row-broadcasted: same bias vector for every row.
            float acc = Bias[j];
            for (int k = 0; k < K; ++k) {
                acc += A[i * K + k] * B[k * N + j];
            }
            out[i * N + j] = acc;
        }
    }
    return out;
}

struct CaseStats {
    size_t total = 0;
    size_t mismatch = 0;
};

static CaseStats run_case(
    const GemmCase& tc,
    int8_t* host_a,
    int8_t* host_b,
    int32_t* host_bias,
    int8_t* host_out
) {
    const int M = tc.M;
    const int N = tc.N;
    const int K = tc.K;

    std::mt19937 rng(2026 + M * 131 + N * 17 + K * 3);
    std::vector<float> A_fp32;
    std::vector<float> B_fp32;
    std::vector<float> Bias_fp32;
    fill_inputs_fp32(A_fp32, B_fp32, Bias_fp32, M, N, K, tc.random, rng);

    // Reference GEMM in fp32 (ideal domain).
    // Quantization note: A/B are quantized to int8 with symmetric scales Si1/Si2.
    // Bias is quantized to int32 using scale (Si1 * Si2), row-broadcasted.
    // Output int8 scale So is computed from fp32 reference outputs.
    // The int32->int8 requant multiplier is M = (Si1 * Si2) / So.
    std::vector<float> ref_fp32 = reference_gemm_fp32(A_fp32, B_fp32, Bias_fp32, M, N, K);
    float Si1 = calc_symmetric_scale_fp32(A_fp32);
    float Si2 = calc_symmetric_scale_fp32(B_fp32);
    float So = calc_symmetric_scale_fp32(ref_fp32);
    float Mscale = (Si1 * Si2) / So;
    MulShift ms = pick_output_scale_shift(Mscale);

    std::vector<int8_t> A;
    std::vector<int8_t> B;
    std::vector<int32_t> Bias;
    A.resize(A_fp32.size());
    B.resize(B_fp32.size());
    Bias.resize(Bias_fp32.size());

    for (size_t i = 0; i < A_fp32.size(); ++i) {
        A[i] = quant_fp32_to_i8(A_fp32[i], Si1);
    }
    for (size_t i = 0; i < B_fp32.size(); ++i) {
        B[i] = quant_fp32_to_i8(B_fp32[i], Si2);
    }
    float Sb = Si1 * Si2;
    for (size_t i = 0; i < Bias_fp32.size(); ++i) {
        Bias[i] = quant_fp32_to_i32(Bias_fp32[i], Sb);
    }

    // Reference GEMM in int32 (hardware domain): int8 A/B + int32 bias.
    std::vector<int32_t> ref_acc_int32(M * N, 0);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            int32_t acc = Bias[j];
            for (int k = 0; k < K; ++k) {
                acc += static_cast<int32_t>(A[i * K + k]) * static_cast<int32_t>(B[k * N + j]);
            }
            ref_acc_int32[i * N + j] = acc;
        }
    }

    // Reference GEMM output in int8 (quantized from int32 accumulator).
    std::vector<int8_t> ref_q_int8(M * N, 0);
    for (size_t i = 0; i < ref_acc_int32.size(); ++i) {
        ref_q_int8[i] = quant_i32_to_i8(ref_acc_int32[i], ms.scale, ms.shift);
    }

    // std::cout << "[Bias] " << tc.name << " N=" << N << std::endl;
    // for (int j = 0; j < N; ++j) {
    //     std::cout << "  j=" << j
    //               << " fp32=" << std::fixed << std::setprecision(6) << Bias_fp32[j]
    //               << " q=" << Bias[j]
    //               << std::endl;
    // }

    std::memcpy(host_a, A.data(), A.size());
    std::memcpy(host_b, B.data(), B.size());
    std::memcpy(host_bias, Bias.data(), Bias.size() * sizeof(int32_t));
    std::memset(host_out, 0, ref_fp32.size() * sizeof(int8_t));

    // SRAM/ACC layout convention (byte addressing):
    // - A and B are int8 in SPM (1 byte per element)
    // - Bias is int32 in ACC (4 bytes per element), row-broadcasted
    // - Output is int8 in SPM when accout_dest=0
    //
    // Dynamic SPM placement to avoid overlap when shapes are large
    // (e.g. K > 256 with M=32 makes A bigger than 0x2000 bytes).
    auto align_up = [](uint32_t x, uint32_t a) -> uint32_t {
        return (x + a - 1u) & ~(a - 1u);
    };
    constexpr uint32_t SPM_ALIGN = 64;                 // 64B alignment
    constexpr uint32_t SPM_SIZE_BYTES = TEST_SPM_SIZE_BYTES;
    constexpr uint32_t SPM_HALF_BYTES = SPM_SIZE_BYTES / 2;

    const uint32_t a_size_bytes = static_cast<uint32_t>(A.size());
    const uint32_t b_size_bytes = static_cast<uint32_t>(B.size());
    const uint32_t out_size_bytes = static_cast<uint32_t>(ref_fp32.size());

    // Layout for post-RTL-fix validation:
    // - A/B in lower half
    // - OUT in upper half
    const uint32_t SRAM_ADDR_A = 0;
    const uint32_t SRAM_ADDR_B = align_up(SRAM_ADDR_A + a_size_bytes, SPM_ALIGN);
    const uint32_t SRAM_ADDR_OUT = align_up(SPM_HALF_BYTES, SPM_ALIGN);
    const uint32_t ACC_ADDR_BIAS = 0x0000;

    if (SRAM_ADDR_B + b_size_bytes > SPM_HALF_BYTES ||
        SRAM_ADDR_OUT + out_size_bytes > SPM_SIZE_BYTES) {
        std::cerr << "[Error] SPM overflow in " << tc.name
                  << ": A=" << a_size_bytes
                  << " B=" << b_size_bytes
                  << " O=" << out_size_bytes
                  << " A@0x" << std::hex << SRAM_ADDR_A
                  << " B@0x" << SRAM_ADDR_B
                  << " O@0x" << SRAM_ADDR_OUT
                  << " O_end=0x" << (SRAM_ADDR_OUT + out_size_bytes - 1)
                  << " B_end=0x" << (SRAM_ADDR_B + b_size_bytes - 1)
                  << std::dec
                  << " SPM=" << SPM_SIZE_BYTES
                  << " HALF=" << SPM_HALF_BYTES << std::endl;
        CaseStats stats;
        stats.total = ref_fp32.size();
        stats.mismatch = stats.total;
        return stats;
    }

    npu_dma_mvin(
        /*host_ptr=*/host_a,
        /*sram_addr=*/SRAM_ADDR_A,
        /*col_num=*/static_cast<uint32_t>(A.size() - 1),
        /*row_num=*/0,
        /*sram_stride=*/0,
        /*dram_stride=*/0,
        /*precision=*/ 1,
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
        /*sram_addr=*/SRAM_ADDR_B,
        /*col_num=*/static_cast<uint32_t>(B.size() - 1),
        /*row_num=*/0,
        /*sram_stride=*/0,
        /*dram_stride=*/0,
        /*precision=*/1,
        /*input_type=*/1,
        /*dest=*/0,
        /*is_bias=*/false,
        /*is_quant=*/false,
        /*quant_zero=*/0,
        /*quant_scale=*/0,
        /*quant_shift=*/0
    );

    // Tiling strategy:
    // - Tile along M (rows) and N (cols) with 32x32 tiles.
    // - K is not tiled here (full K for each tile), matching OS dataflow.
    // - For hardware constraints: input_a_row_num/col_num and input_b_row_num/col_num
    //   are 0-based (m1) sizes; bias/output use actual tile sizes.
    // - Bias is row-broadcasted: load a 32-wide bias vector (int32) into ACC with is_bias=1.
    //   Hardware only accepts 32 bias values at a time; when switching to a new 32-row tile,
    //   reload bias for each 32-column tile (tn <= 32).
    const int TILE_M = 32;
    const int TILE_N = 32;

    for (int row = 0; row < M; row += TILE_M) {
        for (int col = 0; col < N; col += TILE_N) {
            const int tm = std::min(TILE_M, M - row);
            const int tn = std::min(TILE_N, N - col);
            if (tm <= 0 || tn <= 0) continue;

            // Addressing detail:
            // A is stored row-major MxK (stride K bytes).
            // B is stored row-major KxN (stride N bytes).
            // Bias/Output are row-major MxN.
            const uint32_t out_addr = SRAM_ADDR_OUT + static_cast<uint32_t>(row * N + col);
            const uint32_t a_addr = SRAM_ADDR_A + static_cast<uint32_t>(row * K);
            const uint32_t b_addr = SRAM_ADDR_B + static_cast<uint32_t>(col);

            // Load bias vector (tn elements, <=32) into ACC, with is_bias=1.
            // If USE_BIAS=0, Bias is all zeros; still load to keep a uniform path.
            npu_dma_mvin(
                /*host_ptr=*/host_bias + col,
                /*sram_addr=*/ACC_ADDR_BIAS,
                /*col_num=*/static_cast<uint32_t>(tn - 1),
                /*row_num=*/0,
                /*sram_stride=*/0,
                /*dram_stride=*/0,
                /*precision=*/1,
                /*input_type=*/2,
                /*dest=*/1,
                /*is_bias=*/true,
                /*is_quant=*/false,
                /*quant_zero=*/0,
                /*quant_scale=*/0,
                /*quant_shift=*/0
            );

            npu_gemm_run(
                /*dataflow=*/1,
                /*int_type=*/0,
                /*optype=*/0,
                /*accout_dest=*/0,
                /*input_a_zeropoint=*/0,
                /*input_b_zeropoint=*/0,
                /*output_zeropoint=*/0,
                /*output_scale=*/static_cast<uint16_t>(ms.scale),
                /*output_scaleshift=*/static_cast<uint16_t>(ms.shift),
                /*biaspsum_addr=*/ACC_ADDR_BIAS,
                /*biaspsum_stride=*/static_cast<uint16_t>(N),
                /*biaspsum_width=*/static_cast<uint8_t>(tn),
                /*biaspsum_height=*/static_cast<uint8_t>(tm),
                /*output_addr=*/out_addr,
                /*output_stride=*/static_cast<uint16_t>(N),
                /*isaccu=*/1,
                /*relu=*/0,
                /*relu_type=*/0,
                /*is_bias=*/1,
                /*input_a_addr=*/a_addr,
                /*input_a_col_num=*/static_cast<uint16_t>(K - 1),
                /*input_a_row_num=*/static_cast<uint8_t>(tm - 1),
                /*input_a_stride=*/static_cast<uint16_t>(K),
                /*input_b_addr=*/b_addr,
                /*input_b_col_num=*/static_cast<uint8_t>(tn - 1),
                /*input_b_row_num=*/static_cast<uint16_t>(K - 1),
                /*input_b_stride=*/static_cast<uint16_t>(N)
            );
        }
    }

    npu_dma_mvout(
        /*host_ptr=*/host_out,
        /*sram_addr=*/SRAM_ADDR_OUT,
        /*col_num=*/static_cast<uint32_t>(ref_fp32.size() - 1),
        /*row_num=*/0,
        /*sram_stride=*/0,
        /*dram_stride=*/0,
        /*precision=*/1,
        /*output_type=*/0,
        /*source=*/0,
        /*is_quant=*/false,
        /*quant_zero=*/0,
        /*quant_scale=*/0,
        /*quant_shift=*/0
    );

    int mismatch = 0;
    int max_diff = 0;
    const int print_count = std::min<int>(16, static_cast<int>(ref_fp32.size()));
    std::cout << "[Quant] Si1=" << std::fixed << std::setprecision(6) << Si1
              << " Si2=" << Si2
              << " So=" << So
              << " Sb=" << Sb
              << " M=" << Mscale
              << " scale=" << ms.scale
              << " shift=" << ms.shift << std::endl;
    for (size_t i = 0; i < ref_fp32.size(); ++i) {
        int8_t ref_q = ref_q_int8[i];
        int8_t hw_q = host_out[i];
        int diff = std::abs(static_cast<int>(hw_q) - static_cast<int>(ref_q));
        if (diff > MAX_DIFF && mismatch < 5) {
            std::cout << "[Mismatch] " << tc.name
                      << " idx=" << i
                      << " hw_q=" << static_cast<int>(hw_q)
                      << " ref_q=" << static_cast<int>(ref_q)
                      << " acc=" << ref_acc_int32[i]
                      << " diff=" << diff << std::endl;
        }
        if (diff > MAX_DIFF) mismatch++;
        if (diff > max_diff) max_diff = diff;
    }

    std::cout << "[Sample] " << tc.name << " first " << print_count << " elems:" << std::endl;
    for (int i = 0; i < print_count; ++i) {
        int row = i / N;
        int col = i % N;
        int32_t acc_ref = ref_acc_int32[i];
        int8_t ref_q = ref_q_int8[i];
        std::cout << "  (" << row << "," << col << ") hw_q=" << static_cast<int>(host_out[i])
                  << " ref_q=" << static_cast<int>(ref_q)
                  << " acc=" << acc_ref << std::endl;
    }

    // Formatted channel dump (pick one testcase + one channel for inspection).
    // Channel is interpreted as output column index.
    if (tc.name == "tiled_35x42x38") {
        const int channel = 0;
        std::cout << "[ChannelDump] " << tc.name << " col=" << channel << std::endl;
        std::cout << "  row | hw_q | ref_q | acc" << std::endl;
        std::cout << "  ----+------+-------+----------------" << std::endl;
        for (int r = 0; r < M; ++r) {
            int idx = r * N + channel;
            int32_t acc_ref = ref_acc_int32[idx];
            std::cout << std::setw(5) << r << " | "
                      << std::setw(10) << static_cast<int>(host_out[idx]) << " | "
                      << std::setw(10) << static_cast<int>(ref_q_int8[idx]) << " | "
                      << std::setw(14) << acc_ref << std::endl;
        }
    }

    std::cout << "[Case] " << tc.name
              << " M=" << M
              << " N=" << N
              << " K=" << K
              << " random=" << (tc.random ? "yes" : "no")
              << " mismatches=" << mismatch
              << " max_diff=" << max_diff << std::endl;

    CaseStats stats;
    stats.total = ref_fp32.size();
    stats.mismatch = static_cast<size_t>(mismatch);
    return stats;
}

int main() {
    std::cout << "===== NPU GEMM Test (Runtime API) =====" << std::endl;

    if (npu_init() != 0) {
        std::cerr << "Error: NPU init failed. Check kernel driver." << std::endl;
        return -1;
    }

    npu_reset();

    std::vector<GemmCase> cases = {
        // {"attn_32x32x1", 32, 32, 1, false},
        // {"tiled_35x42x38", 35, 42, 38, false},
        // {"fc_4x128x128", 4, 128, 128, false},
        // {"mlp_8x64x256", 8, 64, 256, false},
        // {"proj_16x128x64", 16, 128, 64, false},
        {"test_128*512x512", 128, 512, 512, false}
        // {"attn_32x32x128", 32, 32, 128, false}
    };

    // std::mt19937 rng(2025);
    // std::uniform_int_distribution<int> dist_m(1, 64);
    // std::uniform_int_distribution<int> dist_n(1, 64);
    // std::uniform_int_distribution<int> dist_k(1, 64);
    // for (int i = 0; i < 20; ++i) {
    //     int M = dist_m(rng);
    //     int N = dist_n(rng);
    //     int K = dist_k(rng);
    //     cases.push_back({"random_" + std::to_string(i), M, N, K, true});
    // }

    size_t max_a = 0;
    size_t max_b = 0;
    size_t max_o = 0;
    size_t max_n = 0;
    for (const auto& tc : cases) {
        max_a = std::max(max_a, static_cast<size_t>(tc.M * tc.K));
        max_b = std::max(max_b, static_cast<size_t>(tc.K * tc.N));
        max_o = std::max(max_o, static_cast<size_t>(tc.M * tc.N));
        max_n = std::max(max_n, static_cast<size_t>(tc.N));
    }

    int8_t* host_a = static_cast<int8_t*>(npu_mem_alloc(max_a));
    int8_t* host_b = static_cast<int8_t*>(npu_mem_alloc(max_b));
    int32_t* host_bias = static_cast<int32_t*>(npu_mem_alloc(max_n * sizeof(int32_t)));
    int8_t* host_out = static_cast<int8_t*>(npu_mem_alloc(max_o * sizeof(int8_t)));

    if (!host_a || !host_b || !host_bias || !host_out) {
        std::cerr << "Error: NPU memory alloc failed." << std::endl;
        npu_destroy();
        return -1;
    }

    int pass = 0;
    size_t total_points = 0;
    size_t fail_points = 0;
    for (const auto& tc : cases) {
        CaseStats stats = run_case(tc, host_a, host_b, host_bias, host_out);
        total_points += stats.total;
        fail_points += stats.mismatch;
        if (stats.mismatch == 0) pass++;
    }
    size_t pass_points = total_points - fail_points;

    std::cout << "Summary: total=" << cases.size() << " pass=" << pass
              << " fail=" << (cases.size() - pass) << std::endl;
    std::cout << "Points: total=" << total_points
              << " pass=" << pass_points
              << " fail=" << fail_points << std::endl;

    npu_mem_free(host_a);
    npu_mem_free(host_b);
    npu_mem_free(host_bias);
    npu_mem_free(host_out);
    npu_destroy();

    return 0;
}
