/**
 * @file gemm_debug.cpp
 * @brief 渐进式GEMM调试测试 - 用于定位硬件/软件错误
 * 
 * 测试方案：从最简单的情况逐步增加复杂度
 * Level 0: 1x1x1 标量乘法 (最基础验证)
 * Level 1: 1xNx1 行向量×列向量 (验证累加)
 * Level 2: Mx1x1 列向量×行向量 (验证多行输出)
 * Level 3: 1x1xK 向量点积 (验证K维累加)
 * Level 4: MxNx1 外积 (验证M*N输出)
 * Level 5: 1xNxK 行向量GEMV (验证完整累加)
 * Level 6: Mx1xK 列向量GEMV
 * Level 7: MxNxK 小矩阵 (2x2, 4x4, 8x8)
 * Level 8: 32x32xK 满阵 (验证完整PE阵列)
 * 
 * 使用固定的已知值，便于手工验证
 */

#include "../npu_runtime.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

// ============================================
// 配置
// ============================================
#define PRINT_VERBOSE 1  // 打印详细信息

// SRAM 地址布局
constexpr uint32_t SRAM_ADDR_A   = 0x0000;
constexpr uint32_t SRAM_ADDR_B   = 0x2000;
constexpr uint32_t SRAM_ADDR_OUT = 0x6000;
constexpr uint32_t ACC_ADDR_BIAS = 0x0000;
constexpr uint32_t ACC_ADDR_OUT = 0x0000;

// ============================================
// 工具函数
// ============================================

static void print_matrix_i8(const char* name, const int8_t* data, int rows, int cols) {
    std::cout << name << " (" << rows << "x" << cols << "):" << std::endl;
    for (int i = 0; i < rows && i < 16; ++i) {
        std::cout << "  [";
        for (int j = 0; j < cols && j < 16; ++j) {
            std::cout << std::setw(4) << static_cast<int>(data[i * cols + j]);
            if (j < cols - 1 && j < 15) std::cout << ",";
        }
        if (cols > 16) std::cout << "...";
        std::cout << "]" << std::endl;
    }
    if (rows > 16) std::cout << "  ..." << std::endl;
}

static void print_matrix_i32(const char* name, const int32_t* data, int rows, int cols) {
    std::cout << name << " (" << rows << "x" << cols << "):" << std::endl;
    for (int i = 0; i < rows && i < 16; ++i) {
        std::cout << "  [";
        for (int j = 0; j < cols && j < 16; ++j) {
            std::cout << std::setw(8) << data[i * cols + j];
            if (j < cols - 1 && j < 15) std::cout << ",";
        }
        if (cols > 16) std::cout << "...";
        std::cout << "]" << std::endl;
    }
    if (rows > 16) std::cout << "  ..." << std::endl;
}

// 软件参考GEMM: C = A * B (无bias)
// A: MxK (row-major), B: KxN (row-major), C: MxN (row-major)
static void reference_gemm_i8_to_i32(
    const int8_t* A, const int8_t* B, int32_t* C,
    int M, int N, int K
) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            int32_t acc = 0;
            for (int k = 0; k < K; ++k) {
                acc += static_cast<int32_t>(A[i * K + k]) * static_cast<int32_t>(B[k * N + j]);
            }
            C[i * N + j] = acc;
        }
    }
}

static void reference_gemm_asymmetric_a_u8_to_i32(
    const int8_t* A_raw, const int8_t* B, int32_t* C,
    int M, int N, int K
) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            int32_t acc = 0;
            for (int k = 0; k < K; ++k) {
                const uint8_t a_u8 = static_cast<uint8_t>(A_raw[i * K + k]);
                const int32_t a_signed = static_cast<int32_t>(a_u8) - 128;
                acc += a_signed * static_cast<int32_t>(B[k * N + j]);
            }
            C[i * N + j] = acc;
        }
    }
}

// 比较结果
static int compare_results(
    const int32_t* hw, const int32_t* ref, int size,
    int max_print = 10
) {
    int mismatches = 0;
    int max_diff = 0;
    for (int i = 0; i < size; ++i) {
        int diff = std::abs(hw[i] - ref[i]);
        if (diff > 0) {
            if (mismatches < max_print) {
                std::cout << "  MISMATCH [" << i << "]: hw=" << hw[i] 
                          << " ref=" << ref[i] << " diff=" << diff << std::endl;
            }
            ++mismatches;
            if (diff > max_diff) max_diff = diff;
        }
    }
    if (mismatches > max_print) {
        std::cout << "  ... and " << (mismatches - max_print) << " more mismatches" << std::endl;
    }
    if (mismatches > 0) {
        std::cout << "  Max diff: " << max_diff << std::endl;
    }
    return mismatches;
}

// ============================================
// 测试等级
// ============================================

struct TestResult {
    bool pass;
    int mismatches;
    std::string message;
};

// Level 0: 1x1x1 标量乘法
// A = [a], B = [b], C = [a*b]
static TestResult test_level0_scalar(
    int8_t* host_a, int8_t* host_b, int32_t* host_out
) {
    std::cout << "\n=== Level 0: 1x1x1 Scalar Multiplication ===" << std::endl;
    
    constexpr int M = 1, N = 1, K = 1;
    const int8_t a_val = 3;
    const int8_t b_val = 7;
    const int32_t expected = a_val * b_val; // 21
    
    host_a[0] = a_val;
    host_b[0] = b_val;
    host_out[0] = 0xDEADBEEF; // 标记用于检测是否被写入
    
    std::cout << "  A[0]=" << static_cast<int>(a_val) 
              << " B[0]=" << static_cast<int>(b_val) 
              << " Expected=" << expected << std::endl;

    // MVIN A
    npu_dma_mvin(host_a, SRAM_ADDR_A, K - 1, M - 1, K, K, 1, 0, false, false, false, 0, 0, 0);
    
    // MVIN B  
    npu_dma_mvin(host_b, SRAM_ADDR_B, N - 1, K - 1, N, N, 1, 1, false, false, false, 0, 0, 0);

    // GEMM: accout_dest=1 -> 输出到ACC (int32)
    npu_gemm_run(
        /*dataflow=*/1,         // 0=im2col & OS, 1=OS only
        /*int_type=*/0,         // 0=int8
        /*optype=*/0,           // 0=GEMM
        /*accout_dest=*/1,      // 0=SPM, 1=ACC
        /*input_a_zeropoint=*/0,
        /*input_b_zeropoint=*/0,
        /*output_zeropoint=*/0, // ACC->SPM zeropoint (unused when accout_dest=1)
        /*output_scale=*/1,     // ACC->SPM scale
        /*output_scaleshift=*/0,
        /*biaspsum_addr=*/ACC_ADDR_BIAS,
        /*biaspsum_stride=*/N,
        /*biaspsum_width=*/N,
        /*biaspsum_height=*/M,
        /*output_addr=*/ACC_ADDR_OUT,
        /*output_stride=*/N,
        /*isaccu=*/0,           // 0=No accumulate, 1=Accumulate with psum
        /*relu=*/0,
        /*relu_type=*/0,
        /*is_bias=*/0,          // 0=accumulate psum, 1=accumulate bias
        /*input_a_addr=*/SRAM_ADDR_A,
        /*input_a_col_num=*/K - 1,
        /*input_a_row_num=*/M - 1,
        /*input_a_stride=*/K,
        /*input_b_addr=*/SRAM_ADDR_B,
        /*input_b_col_num=*/N - 1,
        /*input_b_row_num=*/K - 1,
        /*input_b_stride=*/N
    );

    // MVOUT from ACC
    npu_dma_mvout(host_out, ACC_ADDR_OUT, M * N - 1, 0, 0, 0, 1, 1, true, false, 0, 0, 0);

    std::cout << "  HW Result: " << host_out[0] << std::endl;
    
    TestResult result;
    result.pass = (host_out[0] == expected);
    result.mismatches = result.pass ? 0 : 1;
    result.message = result.pass ? "PASS" : "FAIL - scalar multiply broken";
    
    std::cout << "  " << result.message << std::endl;
    return result;
}

static TestResult test_level0_asymmetric_activations(
    int8_t* host_a, int8_t* host_b, int32_t* host_out
) {
    std::cout << "\n=== Level 0A: GEMM Asymmetric Activations ===" << std::endl;

    constexpr int M = 2, N = 1, K = 1;
    host_a[0] = static_cast<int8_t>(0x00); // raw 0   -> -128
    host_a[1] = static_cast<int8_t>(0xFF); // raw 255 -> 127
    host_b[0] = 1;
    std::memset(host_out, 0, M * N * sizeof(int32_t));

    int32_t ref[M * N] = {};
    reference_gemm_asymmetric_a_u8_to_i32(host_a, host_b, ref, M, N, K);

    npu_dma_mvin(host_a, SRAM_ADDR_A, K - 1, M - 1, K, K, 1, 0, false, false, false, 0, 0, 0);
    npu_dma_mvin(host_b, SRAM_ADDR_B, N - 1, K - 1, N, N, 1, 1, false, false, false, 0, 0, 0);

    npu_gemm_run(
        /*dataflow=*/1,
        /*int_type=*/0,
        /*optype=*/0,
        /*accout_dest=*/1,
        /*input_a_zeropoint=*/0,
        /*input_b_zeropoint=*/0,
        /*output_zeropoint=*/0,
        /*output_scale=*/1,
        /*output_scaleshift=*/0,
        /*biaspsum_addr=*/ACC_ADDR_BIAS,
        /*biaspsum_stride=*/N,
        /*biaspsum_width=*/N,
        /*biaspsum_height=*/M,
        /*output_addr=*/ACC_ADDR_OUT,
        /*output_stride=*/N,
        /*isaccu=*/0,
        /*relu=*/0,
        /*relu_type=*/0,
        /*is_bias=*/0,
        /*input_a_addr=*/SRAM_ADDR_A,
        /*input_a_col_num=*/K - 1,
        /*input_a_row_num=*/M - 1,
        /*input_a_stride=*/K,
        /*input_b_addr=*/SRAM_ADDR_B,
        /*input_b_col_num=*/N - 1,
        /*input_b_row_num=*/K - 1,
        /*input_b_stride=*/N,
        /*asymmetric_activations=*/true
    );

    npu_dma_mvout(host_out, ACC_ADDR_OUT, M * N - 1, 0, 0, 0, 1, 1, true, false, 0, 0, 0);

    std::cout << "  Expected: [" << ref[0] << ", " << ref[1] << "]" << std::endl;
    std::cout << "  HW      : [" << host_out[0] << ", " << host_out[1] << "]" << std::endl;

    const int mismatches = compare_results(host_out, ref, M * N);
    TestResult result;
    result.pass = (mismatches == 0);
    result.mismatches = mismatches;
    result.message = result.pass ? "PASS" : "FAIL - asymmetric activation path broken";
    std::cout << "  " << result.message << std::endl;
    return result;
}

// Level 1: 1xNx1 行向量 × 列向量 (N个独立乘法，无累加)
// A = [a], B = [b0, b1, ..., b_{N-1}], C = [a*b0, a*b1, ..., a*b_{N-1}]
static TestResult test_level1_row_times_col(
    int8_t* host_a, int8_t* host_b, int32_t* host_out,
    int N = 4
) {
    std::cout << "\n=== Level 1: 1x" << N << "x1 Row×Col (N independent mults) ===" << std::endl;
    
    constexpr int M = 1, K = 1;
    const int8_t a_val = 5;
    
    host_a[0] = a_val;
    for (int j = 0; j < N; ++j) {
        host_b[j] = static_cast<int8_t>(j + 1); // B = [1, 2, 3, ...]
    }
    std::memset(host_out, 0, N * sizeof(int32_t));

    // 计算参考结果
    std::vector<int32_t> ref(N);
    for (int j = 0; j < N; ++j) {
        ref[j] = a_val * host_b[j];
    }

#if PRINT_VERBOSE
    print_matrix_i8("A", host_a, M, K);
    print_matrix_i8("B", host_b, K, N);
    print_matrix_i32("Expected C", ref.data(), M, N);
#endif

    npu_dma_mvin(host_a, SRAM_ADDR_A, K - 1, M - 1, K, K, 1, 0, false, false, false, 0, 0, 0);
    npu_dma_mvin(host_b, SRAM_ADDR_B, N - 1, K - 1, N, N, 1, 1, false, false, false, 0, 0, 0);

    npu_gemm_run(
        /*dataflow=*/1,         // 0=im2col & OS, 1=OS only
        /*int_type=*/0,         // 0=int8
        /*optype=*/0,           // 0=GEMM
        /*accout_dest=*/1,      // 0=SPM, 1=ACC
        /*input_a_zeropoint=*/0,
        /*input_b_zeropoint=*/0,
        /*output_zeropoint=*/0, // ACC->SPM zeropoint (unused when accout_dest=1)
        /*output_scale=*/1,     // ACC->SPM scale
        /*output_scaleshift=*/0,
        /*biaspsum_addr=*/ACC_ADDR_BIAS,
        /*biaspsum_stride=*/N,
        /*biaspsum_width=*/N,
        /*biaspsum_height=*/M,
        /*output_addr=*/ACC_ADDR_OUT,
        /*output_stride=*/N,
        /*isaccu=*/0,           // 0=No accumulate, 1=Accumulate with psum
        /*relu=*/0,
        /*relu_type=*/0,
        /*is_bias=*/0,          // 0=accumulate psum, 1=accumulate bias
        /*input_a_addr=*/SRAM_ADDR_A,
        /*input_a_col_num=*/K - 1,
        /*input_a_row_num=*/M - 1,
        /*input_a_stride=*/K,
        /*input_b_addr=*/SRAM_ADDR_B,
        /*input_b_col_num=*/N - 1,
        /*input_b_row_num=*/K - 1,
        /*input_b_stride=*/N
    );

    npu_dma_mvout(host_out, ACC_ADDR_OUT, M * N - 1, 0, 0, 0, 1, 1, true, false, 0, 0, 0);

#if PRINT_VERBOSE
    print_matrix_i32("HW C", host_out, M, N);
#endif

    int mismatches = compare_results(host_out, ref.data(), N);
    
    TestResult result;
    result.pass = (mismatches == 0);
    result.mismatches = mismatches;
    result.message = result.pass ? "PASS" : "FAIL - N independent mults broken";
    
    std::cout << "  " << result.message << std::endl;
    return result;
}

// Level 2: Mx1x1 列向量 × 行向量 (M行输出，每行1个元素)
static TestResult test_level2_col_times_row(
    int8_t* host_a, int8_t* host_b, int32_t* host_out,
    int M = 4
) {
    std::cout << "\n=== Level 2: " << M << "x1x1 Col×Row (M rows output) ===" << std::endl;
    
    constexpr int N = 1, K = 1;
    const int8_t b_val = 3;
    
    for (int i = 0; i < M; ++i) {
        host_a[i] = static_cast<int8_t>(i + 1); // A = [1, 2, 3, ...]^T
    }
    host_b[0] = b_val;
    std::memset(host_out, 0, M * sizeof(int32_t));

    std::vector<int32_t> ref(M);
    for (int i = 0; i < M; ++i) {
        ref[i] = host_a[i] * b_val;
    }

#if PRINT_VERBOSE
    print_matrix_i8("A", host_a, M, K);
    print_matrix_i8("B", host_b, K, N);
    print_matrix_i32("Expected C", ref.data(), M, N);
#endif

    npu_dma_mvin(host_a, SRAM_ADDR_A, K - 1, M - 1, K, K, 1, 1, false, false, false, 0, 0, 0);
    
    npu_dma_mvin(host_b, SRAM_ADDR_B, N - 1, K - 1, N, N, 1, 1, false, false, false, 0, 0, 0);

    // Read back data from SPM after MVIN to verify correctness
    int8_t* spm_a_read = static_cast<int8_t*>(npu_mem_alloc(M * K));
    int8_t* spm_b_read = static_cast<int8_t*>(npu_mem_alloc(4));
    if (!spm_a_read || !spm_b_read) {
        std::cerr << "Error: Readback buffer alloc failed." << std::endl;
        if (spm_a_read) npu_mem_free(spm_a_read);
        if (spm_b_read) npu_mem_free(spm_b_read);
        TestResult fail_result;
        fail_result.pass = false;
        fail_result.mismatches = 1;
        fail_result.message = "FAIL - readback alloc failed";
        return fail_result;
    }
    npu_dma_mvout(spm_a_read, SRAM_ADDR_A, static_cast<uint16_t>(M * K - 1), 0, 0, 0, 1, 0, false, false, 0, 0, 0);
    npu_dma_mvout(spm_b_read, SRAM_ADDR_B, static_cast<uint16_t>(4), 0, 0, 0, 1, 0, false, false, 0, 0, 0);
#if PRINT_VERBOSE
    print_matrix_i8("SPM A (readback)", spm_a_read, M, K);
    print_matrix_i8("SPM B (readback)", spm_b_read, 1, 4);
#endif
    npu_mem_free(spm_a_read);
    npu_mem_free(spm_b_read);

    npu_gemm_run(
        /*dataflow=*/1,         // 0=im2col & OS, 1=OS only
        /*int_type=*/0,         // 0=int8
        /*optype=*/0,           // 0=GEMM
        /*accout_dest=*/1,      // 0=SPM, 1=ACC
        /*input_a_zeropoint=*/0,
        /*input_b_zeropoint=*/0,
        /*output_zeropoint=*/0, // ACC->SPM zeropoint (unused when accout_dest=1)
        /*output_scale=*/1,     // ACC->SPM scale
        /*output_scaleshift=*/0,
        /*biaspsum_addr=*/ACC_ADDR_BIAS,
        /*biaspsum_stride=*/N,
        /*biaspsum_width=*/N,
        /*biaspsum_height=*/M,
        /*output_addr=*/ACC_ADDR_OUT,
        /*output_stride=*/N,
        /*isaccu=*/0,           // 0=No accumulate, 1=Accumulate with psum
        /*relu=*/0,
        /*relu_type=*/0,
        /*is_bias=*/0,          // 0=accumulate psum, 1=accumulate bias
        /*input_a_addr=*/SRAM_ADDR_A,
        /*input_a_col_num=*/K - 1,
        /*input_a_row_num=*/M - 1,
        /*input_a_stride=*/K,
        /*input_b_addr=*/SRAM_ADDR_B,
        /*input_b_col_num=*/N - 1,
        /*input_b_row_num=*/K - 1,
        /*input_b_stride=*/N
    );

    npu_dma_mvout(host_out, ACC_ADDR_OUT, M * N - 1, 0, 1, 1, 1, 1, true, false, 0, 0, 0);

#if PRINT_VERBOSE
    print_matrix_i32("HW C", host_out, M, N);
#endif

    int mismatches = compare_results(host_out, ref.data(), M);
    
    TestResult result;
    result.pass = (mismatches == 0);
    result.mismatches = mismatches;
    result.message = result.pass ? "PASS" : "FAIL - M rows output broken";
    
    std::cout << "  " << result.message << std::endl;
    return result;
}

// Level 3: 1x1xK 向量点积 (验证K维累加)
static TestResult test_level3_dot_product(
    int8_t* host_a, int8_t* host_b, int32_t* host_out,
    int K = 4
) {
    std::cout << "\n=== Level 3: 1x1x" << K << " Dot Product (K-dim accumulation) ===" << std::endl;
    
    constexpr int M = 1, N = 1;
    
    for (int k = 0; k < K; ++k) {
        host_a[k] = static_cast<int8_t>(k + 1);     // A = [1, 2, 3, ...]
        host_b[k] = static_cast<int8_t>(K - k);     // B = [K, K-1, K-2, ...]^T
    }
    host_out[0] = 0;

    int32_t expected = 0;
    for (int k = 0; k < K; ++k) {
        expected += static_cast<int32_t>(host_a[k]) * static_cast<int32_t>(host_b[k]);
    }

#if PRINT_VERBOSE
    print_matrix_i8("A (1xK)", host_a, M, K);
    print_matrix_i8("B (Kx1)", host_b, K, N);
    std::cout << "  Expected dot product: " << expected << std::endl;
#endif

    npu_dma_mvin(host_a, SRAM_ADDR_A, K - 1, M - 1, K, K, 1, 0, false, false, false, 0, 0, 0);
    npu_dma_mvin(host_b, SRAM_ADDR_B, N - 1, K - 1, N, N, 1, 1, false, false, false, 0, 0, 0);

    npu_gemm_run(
        /*dataflow=*/1,         // 0=im2col & OS, 1=OS only
        /*int_type=*/0,         // 0=int8
        /*optype=*/0,           // 0=GEMM
        /*accout_dest=*/1,      // 0=SPM, 1=ACC
        /*input_a_zeropoint=*/0,
        /*input_b_zeropoint=*/0,
        /*output_zeropoint=*/0, // ACC->SPM zeropoint (unused when accout_dest=1)
        /*output_scale=*/1,     // ACC->SPM scale
        /*output_scaleshift=*/0,
        /*biaspsum_addr=*/ACC_ADDR_BIAS,
        /*biaspsum_stride=*/N,
        /*biaspsum_width=*/N,
        /*biaspsum_height=*/M,
        /*output_addr=*/ACC_ADDR_OUT,
        /*output_stride=*/N,
        /*isaccu=*/0,           // 0=No accumulate, 1=Accumulate with psum
        /*relu=*/0,
        /*relu_type=*/0,
        /*is_bias=*/0,          // 0=accumulate psum, 1=accumulate bias
        /*input_a_addr=*/SRAM_ADDR_A,
        /*input_a_col_num=*/K - 1,
        /*input_a_row_num=*/M - 1,
        /*input_a_stride=*/K,
        /*input_b_addr=*/SRAM_ADDR_B,
        /*input_b_col_num=*/N - 1,
        /*input_b_row_num=*/K - 1,
        /*input_b_stride=*/N
    );

    npu_dma_mvout(host_out, ACC_ADDR_OUT, M * N - 1, 0, 0, 0, 1, 1, true, false, 0, 0, 0);

    std::cout << "  HW Result: " << host_out[0] << std::endl;

    TestResult result;
    result.pass = (host_out[0] == expected);
    result.mismatches = result.pass ? 0 : 1;
    result.message = result.pass ? "PASS" : "FAIL - K-dim accumulation broken";
    
    std::cout << "  " << result.message << std::endl;
    return result;
}

// Level 4: MxNx1 外积 (M*N输出，每个只有1次乘法)
static TestResult test_level4_outer_product(
    int8_t* host_a, int8_t* host_b, int32_t* host_out,
    int M = 4, int N = 4
) {
    std::cout << "\n=== Level 4: " << M << "x" << N << "x1 Outer Product ===" << std::endl;
    
    constexpr int K = 1;
    
    for (int i = 0; i < M; ++i) {
        host_a[i] = static_cast<int8_t>(i + 1);
    }
    for (int j = 0; j < N; ++j) {
        host_b[j] = static_cast<int8_t>(j + 1);
    }
    std::memset(host_out, 0, M * N * sizeof(int32_t));

    std::vector<int32_t> ref(M * N);
    reference_gemm_i8_to_i32(host_a, host_b, ref.data(), M, N, K);

#if PRINT_VERBOSE
    print_matrix_i8("A", host_a, M, K);
    print_matrix_i8("B", host_b, K, N);
    print_matrix_i32("Expected C", ref.data(), M, N);
#endif

    npu_dma_mvin(host_a, SRAM_ADDR_A, K - 1, M - 1, K, K, 1, 0, false, false, false, 0, 0, 0);
    npu_dma_mvin(host_b, SRAM_ADDR_B, N - 1, K - 1, N, N, 1, 1, false, false, false, 0, 0, 0);

    npu_gemm_run(
        /*dataflow=*/1,         // 0=im2col & OS, 1=OS only
        /*int_type=*/0,         // 0=int8
        /*optype=*/0,           // 0=GEMM
        /*accout_dest=*/1,      // 0=SPM, 1=ACC
        /*input_a_zeropoint=*/0,
        /*input_b_zeropoint=*/0,
        /*output_zeropoint=*/0, // ACC->SPM zeropoint (unused when accout_dest=1)
        /*output_scale=*/1,     // ACC->SPM scale
        /*output_scaleshift=*/0,
        /*biaspsum_addr=*/ACC_ADDR_BIAS,
        /*biaspsum_stride=*/N,
        /*biaspsum_width=*/N,
        /*biaspsum_height=*/M,
        /*output_addr=*/ACC_ADDR_OUT,
        /*output_stride=*/N,
        /*isaccu=*/0,           // 0=No accumulate, 1=Accumulate with psum
        /*relu=*/0,
        /*relu_type=*/0,
        /*is_bias=*/0,          // 0=accumulate psum, 1=accumulate bias
        /*input_a_addr=*/SRAM_ADDR_A,
        /*input_a_col_num=*/K - 1,
        /*input_a_row_num=*/M - 1,
        /*input_a_stride=*/K,
        /*input_b_addr=*/SRAM_ADDR_B,
        /*input_b_col_num=*/N - 1,
        /*input_b_row_num=*/K - 1,
        /*input_b_stride=*/N
    );

    npu_dma_mvout(host_out, ACC_ADDR_OUT, M * N - 1, 0, 0, 0, 1, 1, true, false, 0, 0, 0);

#if PRINT_VERBOSE
    print_matrix_i32("HW C", host_out, M, N);
#endif

    int mismatches = compare_results(host_out, ref.data(), M * N);
    
    TestResult result;
    result.pass = (mismatches == 0);
    result.mismatches = mismatches;
    result.message = result.pass ? "PASS" : "FAIL - outer product broken";
    
    std::cout << "  " << result.message << std::endl;
    return result;
}

// Level 5: 1xNxK GEMV (行向量 × 矩阵)
static TestResult test_level5_gemv_row(
    int8_t* host_a, int8_t* host_b, int32_t* host_out,
    int N = 4, int K = 4
) {
    std::cout << "\n=== Level 5: 1x" << N << "x" << K << " Row GEMV ===" << std::endl;
    
    constexpr int M = 1;
    
    for (int k = 0; k < K; ++k) {
        host_a[k] = static_cast<int8_t>(k + 1);
    }
    for (int k = 0; k < K; ++k) {
        for (int j = 0; j < N; ++j) {
            host_b[k * N + j] = static_cast<int8_t>((k + j) % 10 + 1);
        }
    }
    std::memset(host_out, 0, N * sizeof(int32_t));

    std::vector<int32_t> ref(N);
    reference_gemm_i8_to_i32(host_a, host_b, ref.data(), M, N, K);

#if PRINT_VERBOSE
    print_matrix_i8("A (1xK)", host_a, M, K);
    print_matrix_i8("B (KxN)", host_b, K, N);
    print_matrix_i32("Expected C", ref.data(), M, N);
#endif

    npu_dma_mvin(host_a, SRAM_ADDR_A, K - 1, M - 1, K, K, 1, 0, false, false, false, 0, 0, 0);
    npu_dma_mvin(host_b, SRAM_ADDR_B, K * N - 1, 0, 0, 0, 1, 1, false, false, false, 0, 0, 0);

    npu_gemm_run(
        /*dataflow=*/1,         // 0=im2col & OS, 1=OS only
        /*int_type=*/0,         // 0=int8
        /*optype=*/0,           // 0=GEMM
        /*accout_dest=*/1,      // 0=SPM, 1=ACC
        /*input_a_zeropoint=*/0,
        /*input_b_zeropoint=*/0,
        /*output_zeropoint=*/0, // ACC->SPM zeropoint (unused when accout_dest=1)
        /*output_scale=*/1,     // ACC->SPM scale
        /*output_scaleshift=*/0,
        /*biaspsum_addr=*/ACC_ADDR_BIAS,
        /*biaspsum_stride=*/N,
        /*biaspsum_width=*/N,
        /*biaspsum_height=*/M,
        /*output_addr=*/ACC_ADDR_OUT,
        /*output_stride=*/N,
        /*isaccu=*/0,           // 0=No accumulate, 1=Accumulate with psum
        /*relu=*/0,
        /*relu_type=*/0,
        /*is_bias=*/0,          // 0=accumulate psum, 1=accumulate bias
        /*input_a_addr=*/SRAM_ADDR_A,
        /*input_a_col_num=*/K - 1,
        /*input_a_row_num=*/M - 1,
        /*input_a_stride=*/K,
        /*input_b_addr=*/SRAM_ADDR_B,
        /*input_b_col_num=*/N - 1,
        /*input_b_row_num=*/K - 1,
        /*input_b_stride=*/N
    );

    npu_dma_mvout(host_out, ACC_ADDR_OUT, N - 1, 0, 0, 0, 1, 1, true, false, 0, 0, 0);

#if PRINT_VERBOSE
    print_matrix_i32("HW C", host_out, M, N);
#endif

    int mismatches = compare_results(host_out, ref.data(), N);
    
    TestResult result;
    result.pass = (mismatches == 0);
    result.mismatches = mismatches;
    result.message = result.pass ? "PASS" : "FAIL - row GEMV broken";
    
    std::cout << "  " << result.message << std::endl;
    return result;
}

// Level 6: Mx1xK GEMV (矩阵 × 列向量)
static TestResult test_level6_gemv_col(
    int8_t* host_a, int8_t* host_b, int32_t* host_out,
    int M = 4, int K = 4
) {
    std::cout << "\n=== Level 6: " << M << "x1x" << K << " Col GEMV ===" << std::endl;
    
    constexpr int N = 1;
    
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            host_a[i * K + k] = static_cast<int8_t>((i + k) % 10 + 1);
        }
    }
    for (int k = 0; k < K; ++k) {
        host_b[k] = static_cast<int8_t>(k + 1);
    }
    std::memset(host_out, 0, M * sizeof(int32_t));

    std::vector<int32_t> ref(M);
    reference_gemm_i8_to_i32(host_a, host_b, ref.data(), M, N, K);

#if PRINT_VERBOSE
    print_matrix_i8("A (MxK)", host_a, M, K);
    print_matrix_i8("B (Kx1)", host_b, K, N);
    print_matrix_i32("Expected C", ref.data(), M, N);
#endif

    npu_dma_mvin(host_a, SRAM_ADDR_A, M * K - 1, 0, 0, 0, 1, 0, false, false, false, 0, 0, 0);
    npu_dma_mvin(host_b, SRAM_ADDR_B, N - 1, K - 1, N, N, 1, 1, false, false, false, 0, 0, 0);

    npu_gemm_run(
        /*dataflow=*/1,         // 0=im2col & OS, 1=OS only
        /*int_type=*/0,         // 0=int8
        /*optype=*/0,           // 0=GEMM
        /*accout_dest=*/1,      // 0=SPM, 1=ACC
        /*input_a_zeropoint=*/0,
        /*input_b_zeropoint=*/0,
        /*output_zeropoint=*/0, // ACC->SPM zeropoint (unused when accout_dest=1)
        /*output_scale=*/1,     // ACC->SPM scale
        /*output_scaleshift=*/0,
        /*biaspsum_addr=*/ACC_ADDR_BIAS,
        /*biaspsum_stride=*/N,
        /*biaspsum_width=*/N,
        /*biaspsum_height=*/M,
        /*output_addr=*/ACC_ADDR_OUT,
        /*output_stride=*/N,
        /*isaccu=*/0,           // 0=No accumulate, 1=Accumulate with psum
        /*relu=*/0,
        /*relu_type=*/0,
        /*is_bias=*/0,          // 0=accumulate psum, 1=accumulate bias
        /*input_a_addr=*/SRAM_ADDR_A,
        /*input_a_col_num=*/K - 1,
        /*input_a_row_num=*/M - 1,
        /*input_a_stride=*/K,
        /*input_b_addr=*/SRAM_ADDR_B,
        /*input_b_col_num=*/N - 1,
        /*input_b_row_num=*/K - 1,
        /*input_b_stride=*/N
    );

    npu_dma_mvout(host_out, ACC_ADDR_OUT, M - 1, 0, 0, 0, 1, 1, true, false, 0, 0, 0);

#if PRINT_VERBOSE
    print_matrix_i32("HW C", host_out, M, N);
#endif

    int mismatches = compare_results(host_out, ref.data(), M);
    
    TestResult result;
    result.pass = (mismatches == 0);
    result.mismatches = mismatches;
    result.message = result.pass ? "PASS" : "FAIL - col GEMV broken";
    
    std::cout << "  " << result.message << std::endl;
    return result;
}

// Level 7: 小矩阵GEMM
static TestResult test_level7_small_gemm(
    int8_t* host_a, int8_t* host_b, int32_t* host_out,
    int M, int N, int K
) {
    std::cout << "\n=== Level 7: " << M << "x" << N << "x" << K << " Small GEMM ===" << std::endl;
    
    // 使用简单的固定模式填充
    for (int i = 0; i < M * K; ++i) {
        host_a[i] = static_cast<int8_t>((i % 10) + 1);
    }
    for (int i = 0; i < K * N; ++i) {
        host_b[i] = static_cast<int8_t>((i % 10) + 1);
    }
    std::memset(host_out, 0, M * N * sizeof(int32_t));

    std::vector<int32_t> ref(M * N);
    reference_gemm_i8_to_i32(host_a, host_b, ref.data(), M, N, K);

#if PRINT_VERBOSE
    print_matrix_i8("A", host_a, M, K);
    print_matrix_i8("B", host_b, K, N);
    print_matrix_i32("Expected C", ref.data(), M, N);
#endif

    // MVIN as flat arrays
    npu_dma_mvin(host_a, SRAM_ADDR_A, M * K - 1, 0, 0, 0, 1, 0, false, false, false, 0, 0, 0);
    npu_dma_mvin(host_b, SRAM_ADDR_B, K * N - 1, 0, 0, 0, 1, 1, false, false, false, 0, 0, 0);

    npu_gemm_run(
        /*dataflow=*/1,         // 0=im2col & OS, 1=OS only
        /*int_type=*/0,         // 0=int8
        /*optype=*/0,           // 0=GEMM
        /*accout_dest=*/1,      // 0=SPM, 1=ACC
        /*input_a_zeropoint=*/0,
        /*input_b_zeropoint=*/0,
        /*output_zeropoint=*/0, // ACC->SPM zeropoint (unused when accout_dest=1)
        /*output_scale=*/1,     // ACC->SPM scale
        /*output_scaleshift=*/0,
        /*biaspsum_addr=*/ACC_ADDR_BIAS,
        /*biaspsum_stride=*/N,
        /*biaspsum_width=*/N,
        /*biaspsum_height=*/M,
        /*output_addr=*/ACC_ADDR_OUT,
        /*output_stride=*/N,
        /*isaccu=*/0,           // 0=No accumulate, 1=Accumulate with psum
        /*relu=*/0,
        /*relu_type=*/0,
        /*is_bias=*/0,          // 0=accumulate psum, 1=accumulate bias
        /*input_a_addr=*/SRAM_ADDR_A,
        /*input_a_col_num=*/K - 1,
        /*input_a_row_num=*/M - 1,
        /*input_a_stride=*/K,
        /*input_b_addr=*/SRAM_ADDR_B,
        /*input_b_col_num=*/N - 1,
        /*input_b_row_num=*/K - 1,
        /*input_b_stride=*/N
    );

    npu_dma_mvout(host_out, ACC_ADDR_OUT, M * N - 1, 0, 0, 0, 1, 1, true, false, 0, 0, 0);

#if PRINT_VERBOSE
    print_matrix_i32("HW C", host_out, M, N);
#endif

    int mismatches = compare_results(host_out, ref.data(), M * N);
    
    TestResult result;
    result.pass = (mismatches == 0);
    result.mismatches = mismatches;
    result.message = result.pass ? "PASS" : ("FAIL - " + std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K) + " GEMM broken");
    
    std::cout << "  " << result.message << std::endl;
    return result;
}

// Level 8: 32x32xK 满PE阵列GEMM
static TestResult test_level8_full_array(
    int8_t* host_a, int8_t* host_b, int32_t* host_out,
    int K = 1
) {
    constexpr int M = 32, N = 32;
    std::cout << "\n=== Level 8: 32x32x" << K << " Full PE Array ===" << std::endl;
    
    for (int i = 0; i < M * K; ++i) {
        host_a[i] = static_cast<int8_t>((i % 5) + 1);
    }
    for (int i = 0; i < K * N; ++i) {
        host_b[i] = static_cast<int8_t>((i % 5) + 1);
    }
    std::memset(host_out, 0, M * N * sizeof(int32_t));

    std::vector<int32_t> ref(M * N);
    reference_gemm_i8_to_i32(host_a, host_b, ref.data(), M, N, K);

    npu_dma_mvin(host_a, SRAM_ADDR_A, M * K - 1, 0, 0, 0, 1, 0, false, false, false, 0, 0, 0);
    npu_dma_mvin(host_b, SRAM_ADDR_B, K * N - 1, 0, 0, 0, 1, 1, false, false, false, 0, 0, 0);

    npu_gemm_run(
        1, 0, 0, 1, 0, 0, 0, 1, 0,
        ACC_ADDR_BIAS, N, N, M,
        ACC_ADDR_OUT, N, 0, 0, 0, 0,
        SRAM_ADDR_A, K - 1, M - 1, K,
        SRAM_ADDR_B, N - 1, K - 1, N
    );

    npu_dma_mvout(host_out, ACC_ADDR_OUT, M * N - 1, 0, 0, 0, 1, 1, true, false, 0, 0, 0);

    // 只打印部分结果
    std::cout << "  Reference C (4x4 corner):" << std::endl;
    for (int i = 0; i < 4; ++i) {
        std::cout << "    [";
        for (int j = 0; j < 4; ++j) {
            std::cout << std::setw(6) << ref[i * N + j];
            if (j < 3) std::cout << ",";
        }
        std::cout << "]" << std::endl;
    }
    
    std::cout << "  HW C (4x4 corner):" << std::endl;
    for (int i = 0; i < 4; ++i) {
        std::cout << "    [";
        for (int j = 0; j < 4; ++j) {
            std::cout << std::setw(6) << host_out[i * N + j];
            if (j < 3) std::cout << ",";
        }
        std::cout << "]" << std::endl;
    }

    int mismatches = compare_results(host_out, ref.data(), M * N);
    
    TestResult result;
    result.pass = (mismatches == 0);
    result.mismatches = mismatches;
    result.message = result.pass ? "PASS" : "FAIL - 32x32 full array broken";
    
    std::cout << "  " << result.message << std::endl;
    return result;
}

// ============================================
// Main
// ============================================

int main() {
    std::cout << "============================================" << std::endl;
    std::cout << "     NPU GEMM Progressive Debug Test        " << std::endl;
    std::cout << "============================================" << std::endl;

    if (npu_init() != 0) {
        std::cerr << "Error: NPU init failed." << std::endl;
        return -1;
    }
    npu_reset();

    // 分配足够大的缓冲区
    constexpr size_t MAX_SIZE = 32 * 256; // 足够容纳 32x256 矩阵
    int8_t* host_a = static_cast<int8_t*>(npu_mem_alloc(MAX_SIZE));
    int8_t* host_b = static_cast<int8_t*>(npu_mem_alloc(MAX_SIZE));
    int32_t* host_out = static_cast<int32_t*>(npu_mem_alloc(MAX_SIZE * sizeof(int32_t)));

    if (!host_a || !host_b || !host_out) {
        std::cerr << "Error: Memory allocation failed." << std::endl;
        npu_destroy();
        return -1;
    }

    std::vector<TestResult> results;

    // ========== 执行测试 ==========
    
    // Level 0: 标量
    // results.push_back(test_level0_scalar(host_a, host_b, host_out));
    // if (!results.back().pass) {
    //     std::cout << "\n*** STOP: Level 0 failed - basic scalar multiply broken ***" << std::endl;
    //     goto summary;
    // }

	    // // Level 1: 1xNx1
	    // results.push_back(test_level1_row_times_col(host_a, host_b, host_out, 4));
	    // if (!results.back().pass) {
	    //     std::cout << "\n*** STOP: Level 1 failed - multiple outputs broken ***" << std::endl;
	    //     goto summary;
	    // }

	    results.push_back(test_level0_scalar(host_a, host_b, host_out));
	    if (!results.back().pass) {
	        std::cout << "\n*** STOP: Level 0 failed - basic scalar multiply broken ***" << std::endl;
	        goto summary;
	    }

	    results.push_back(test_level0_asymmetric_activations(host_a, host_b, host_out));
	    if (!results.back().pass) {
	        std::cout << "\n*** STOP: Level 0A failed - asymmetric activation handling broken ***" << std::endl;
	        goto summary;
	    }

	    // Level 2: Mx1x1
	    results.push_back(test_level2_col_times_row(host_a, host_b, host_out, 4));
    if (!results.back().pass) {
        std::cout << "\n*** STOP: Level 2 failed - multiple rows broken ***" << std::endl;
        goto summary;
    }

    // Level 3: 1x1xK 点积
    for (int K : {2, 4, 8, 16, 32}) {
        results.push_back(test_level3_dot_product(host_a, host_b, host_out, K));
        if (!results.back().pass) {
            std::cout << "\n*** STOP: Level 3 (K=" << K << ") failed - accumulation broken ***" << std::endl;
            goto summary;
        }
    }

    // Level 4: 外积
    results.push_back(test_level4_outer_product(host_a, host_b, host_out, 4, 4));
    if (!results.back().pass) {
        std::cout << "\n*** STOP: Level 4 failed - outer product broken ***" << std::endl;
        goto summary;
    }

    // Level 5: Row GEMV
    results.push_back(test_level5_gemv_row(host_a, host_b, host_out, 4, 4));
    if (!results.back().pass) {
        std::cout << "\n*** STOP: Level 5 failed - row GEMV broken ***" << std::endl;
        goto summary;
    }

    // Level 6: Col GEMV
    results.push_back(test_level6_gemv_col(host_a, host_b, host_out, 4, 4));
    if (!results.back().pass) {
        std::cout << "\n*** STOP: Level 6 failed - col GEMV broken ***" << std::endl;
        goto summary;
    }

    // Level 7: 小矩阵 (逐步增大)
    for (auto [M, N, K] : std::vector<std::tuple<int,int,int>>{{2,2,2}, {4,4,4}, {8,8,8}, {16,16,16}}) {
        results.push_back(test_level7_small_gemm(host_a, host_b, host_out, M, N, K));
        if (!results.back().pass) {
            std::cout << "\n*** STOP: Level 7 (" << M << "x" << N << "x" << K << ") failed ***" << std::endl;
            goto summary;
        }
    }

    // Level 8: 32x32 满阵
    for (int K : {1, 2, 4, 8, 16, 32}) {
        results.push_back(test_level8_full_array(host_a, host_b, host_out, K));
        if (!results.back().pass) {
            std::cout << "\n*** STOP: Level 8 (32x32x" << K << ") failed ***" << std::endl;
            goto summary;
        }
    }

summary:
    // ========== 汇总 ==========
    std::cout << "\n============================================" << std::endl;
    std::cout << "                  SUMMARY                   " << std::endl;
    std::cout << "============================================" << std::endl;
    
    int passed = 0, failed = 0;
    for (const auto& r : results) {
        if (r.pass) ++passed; else ++failed;
    }
    
    std::cout << "Total: " << results.size() << " tests" << std::endl;
    std::cout << "Passed: " << passed << std::endl;
    std::cout << "Failed: " << failed << std::endl;
    
    if (failed > 0) {
        std::cout << "\n错误定位建议:" << std::endl;
        std::cout << "- Level 0 失败: 检查基础DMA/GEMM指令编码" << std::endl;
        std::cout << "- Level 1-2 失败: 检查输出地址计算/stride" << std::endl;
        std::cout << "- Level 3 失败: 检查累加逻辑/PE内部状态" << std::endl;
        std::cout << "- Level 4-6 失败: 检查数据路由/广播逻辑" << std::endl;
        std::cout << "- Level 7-8 失败: 检查边界情况/满阵调度" << std::endl;
    }

    npu_mem_free(host_a);
    npu_mem_free(host_b);
    npu_mem_free(host_out);
    npu_destroy();

    return failed > 0 ? 1 : 0;
}
