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

// Compile-time switch: max allowed absolute diff before counting as mismatch
#ifndef MAX_DIFF
#define MAX_DIFF 0
#endif

// For quantized (fp-domain) tests, allow slightly more tolerance due to
// cascaded quantization errors (MVIN dequant + MATADD output quant)
#ifndef MAX_DIFF_QUANT
#define MAX_DIFF_QUANT 2
#endif

// ==========================================
// Common structures
// ==========================================

struct MataddCase {
    std::string name;
    int col_num;
    int row_num;
    bool random;
};

struct CaseStats {
    size_t total    = 0;
    size_t mismatch = 0;
};

// ==========================================
// Quantization helpers (matching HW behavior)
// ==========================================

// Hardware MVIN dequant formula (int8 -> int32):
//   mul = quant_scale * int8_val            (signed 16x8 -> 24-bit)
//   if scale_shift <= 0:  result = mul >>> (-scale_shift)   (arithmetic right shift)
//   if scale_shift >  0:  result = mul <<  scale_shift      (left shift)
//   result = clip(result, INT32_MIN, INT32_MAX)
static int32_t hw_mvin_dequant(int8_t val, int16_t scale, int16_t shift) {
    int64_t mul = static_cast<int64_t>(scale) * static_cast<int64_t>(val);
    int64_t result;
    if (shift <= 0) {
        int s = -shift;
        result = (s > 0) ? (mul >> s) : mul;  // arithmetic right shift (C++ on signed)
    } else {
        result = mul << shift;
    }
    if (result > INT32_MAX) result = INT32_MAX;
    if (result < INT32_MIN) result = INT32_MIN;
    return static_cast<int32_t>(result);
}

// Hardware MATADD output quant formula (int32 -> int8):
//   mul = output_scale * acc_val            (signed 16x32 -> 48-bit)
//   if scale_shift <= 0:  result = mul >>> (-scale_shift)
//   if scale_shift >  0:  result = mul <<  scale_shift
//   result = clip(result, -128, 127)
static int8_t hw_output_quant(int32_t val, int16_t scale, int16_t shift) {
    int64_t mul = static_cast<int64_t>(scale) * static_cast<int64_t>(val);
    int64_t result;
    if (shift <= 0) {
        int s = -shift;
        result = (s > 0) ? (mul >> s) : mul;
    } else {
        result = mul << shift;
    }
    if (result > 127)  result = 127;
    if (result < -128) result = -128;
    return static_cast<int8_t>(result);
}

// Symmetric quantization: compute scale = max(|x|) / 127
static float calc_symmetric_scale(const std::vector<float>& x) {
    float max_abs = 0.0f;
    for (float v : x) {
        float a = std::fabs(v);
        if (a > max_abs) max_abs = a;
    }
    if (max_abs < 1e-12f) return 1.0f;
    return max_abs / 127.0f;
}

// Quantize fp32 to int8 with symmetric scale
static int8_t quant_fp32_to_i8(float x, float scale) {
    if (scale <= 0.0f) return 0;
    int32_t q = static_cast<int32_t>(std::lrint(x / scale));
    if (q > 127)  q = 127;
    if (q < -128) q = -128;
    return static_cast<int8_t>(q);
}

// Pick (scale, shift) to approximate a float multiplier M as:
//   M ≈ scale * 2^shift   (shift <= 0, i.e. right-shift)
// scale fits in int16 (max 32767), shift is negative
struct MulShift {
    int16_t scale;
    int16_t shift;
};

static MulShift pick_scale_shift(float M) {
    MulShift r{0, 0};
    if (M <= 0.0f) return r;
    if (M >= 32767.0f) {
        r.scale = 32767;
        r.shift = 0;
        return r;
    }

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

// ==========================================
// Helper: saturating int8 add (for raw tests)
// ==========================================
static int8_t sat_add_i8(int8_t a, int8_t b) {
    int32_t s = static_cast<int32_t>(a) + static_cast<int32_t>(b);
    if (s > 127)  s = 127;
    if (s < -128) s = -128;
    return static_cast<int8_t>(s);
}

// ==========================================
// Fill int8 input matrices (raw test)
// ==========================================
static void fill_inputs_i8(
    std::vector<int8_t>& A,
    std::vector<int8_t>& B,
    int size,
    bool random,
    std::mt19937& rng
) {
    A.resize(size);
    B.resize(size);

    if (random) {
        std::uniform_int_distribution<int> dist(-128, 127);
        for (int i = 0; i < size; ++i) {
            A[i] = static_cast<int8_t>(dist(rng));
            B[i] = static_cast<int8_t>(dist(rng));
        }
    } else {
        for (int i = 0; i < size; ++i) {
            A[i] = static_cast<int8_t>((i % 256) - 128);
            B[i] = static_cast<int8_t>(((i * 3 + 7) % 256) - 128);
        }
    }
}

// ==========================================
// Fill fp32 input matrices (quantized test)
// ==========================================
static void fill_inputs_fp32(
    std::vector<float>& A,
    std::vector<float>& B,
    int size,
    bool random,
    std::mt19937& rng
) {
    A.resize(size);
    B.resize(size);

    if (random) {
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (int i = 0; i < size; ++i) {
            A[i] = dist(rng);
            B[i] = dist(rng);
        }
    } else {
        for (int i = 0; i < size; ++i) {
            A[i] = -1.0f + 0.01f * static_cast<float>(i % 200);
            B[i] = -0.5f + 0.008f * static_cast<float>(i % 125);
        }
    }
}

// ==========================================
// Reference: element-wise saturating add
// ==========================================
static std::vector<int8_t> reference_matadd_i8(
    const std::vector<int8_t>& A,
    const std::vector<int8_t>& B,
    int size
) {
    std::vector<int8_t> out(size);
    for (int i = 0; i < size; ++i) {
        out[i] = sat_add_i8(A[i], B[i]);
    }
    return out;
}

// ======================================================================
// Part 1: Raw int8 matadd test (identity quant, no rescaling)
// ======================================================================

static CaseStats run_case_raw(
    const MataddCase& tc,
    int8_t*  host_a,
    int8_t*  host_b,
    int8_t*  host_out
) {
    const int col_num = tc.col_num;
    const int row_num = tc.row_num;
    const int mat_size = col_num * row_num;

    std::mt19937 rng(2026 + col_num * 131 + row_num * 17);

    // 1. Generate input data
    std::vector<int8_t> A, B;
    fill_inputs_i8(A, B, mat_size, tc.random, rng);

    // 2. Compute reference result
    std::vector<int8_t> ref = reference_matadd_i8(A, B, mat_size);

    // 3. Copy input data to host buffers, clear output
    std::memcpy(host_a, A.data(), mat_size);
    std::memcpy(host_b, B.data(), mat_size);
    std::memset(host_out, 0, mat_size);

    // SRAM/ACC address layout (byte addressing):
    // - MATADD reads A and B from ACC
    // - MATADD writes output to SPM
    const uint32_t ACC_ADDR_A   = 0x0000;
    const uint32_t ACC_ADDR_B   = 0x2000;
    const uint32_t SPM_ADDR_OUT = 0x4000;

    // 4. MVIN Matrix A into ACC (dest=1, is_quant=1, scale=1, shift=0 => identity)
    npu_dma_mvin(
        /*host_ptr=*/host_a,
        /*sram_addr=*/ACC_ADDR_A,
        /*col_num=*/static_cast<uint16_t>(mat_size - 1),
        /*row_num=*/0,
        /*sram_stride=*/0,
        /*dram_stride=*/0,
        /*precision=*/1,
        /*input_type=*/0,
        /*dest=*/1,          // ACC
        /*is_bias=*/false,
        /*is_quant=*/true,
        /*quant_zero=*/0,
        /*quant_scale=*/1,
        /*quant_shift=*/0
    );

    // 5. MVIN Matrix B into ACC (dest=1, is_quant=1, scale=1, shift=0 => identity)
    npu_dma_mvin(
        /*host_ptr=*/host_b,
        /*sram_addr=*/ACC_ADDR_B,
        /*col_num=*/static_cast<uint16_t>(mat_size - 1),
        /*row_num=*/0,
        /*sram_stride=*/0,
        /*dram_stride=*/0,
        /*precision=*/1,
        /*input_type=*/0,
        /*dest=*/1,          // ACC
        /*is_bias=*/false,
        /*is_quant=*/true,
        /*quant_zero=*/0,
        /*quant_scale=*/1,
        /*quant_shift=*/0
    );

    // 6. Run MATADD (output_scale=1, output_scaleshift=0 => no requant)
    npu_matadd_run(
        /*input_a_addr=*/ACC_ADDR_A,
        /*input_b_addr=*/ACC_ADDR_B,
        /*output_addr=*/SPM_ADDR_OUT,
        /*col_num=*/static_cast<uint8_t>(col_num),
        /*row_num=*/static_cast<uint8_t>(row_num),
        /*output_zeropoint=*/0,
        /*output_scale=*/1,
        /*output_scaleshift=*/0
    );

    // 7. MVOUT result from SPM (source=0)
    npu_dma_mvout(
        /*host_ptr=*/host_out,
        /*sram_addr=*/SPM_ADDR_OUT,
        /*col_num=*/static_cast<uint16_t>(mat_size - 1),
        /*row_num=*/0,
        /*sram_stride=*/0,
        /*dram_stride=*/0,
        /*precision=*/1,
        /*output_type=*/0,   // final result
        /*source=*/0,        // SPM
        /*is_quant=*/false,
        /*quant_zero=*/0,
        /*quant_scale=*/0,
        /*quant_shift=*/0
    );

    // 8. Compare
    int mismatch = 0;
    int max_diff = 0;
    const int print_count = std::min<int>(16, mat_size);

    for (int i = 0; i < mat_size; ++i) {
        int diff = std::abs(static_cast<int>(host_out[i]) - static_cast<int>(ref[i]));
        if (diff > MAX_DIFF && mismatch < 5) {
            int row = i / col_num;
            int col = i % col_num;
            std::cout << "[Mismatch] " << tc.name
                      << " idx=" << i
                      << " (" << row << "," << col << ")"
                      << " hw=" << static_cast<int>(host_out[i])
                      << " ref=" << static_cast<int>(ref[i])
                      << " a=" << static_cast<int>(A[i])
                      << " b=" << static_cast<int>(B[i])
                      << " diff=" << diff << std::endl;
        }
        if (diff > MAX_DIFF) mismatch++;
        if (diff > max_diff) max_diff = diff;
    }

    // Print first few elements
    std::cout << "[Sample] " << tc.name << " first " << print_count << " elems:" << std::endl;
    for (int i = 0; i < print_count; ++i) {
        int row = i / col_num;
        int col = i % col_num;
        std::cout << "  (" << row << "," << col << ")"
                  << " a=" << std::setw(4) << static_cast<int>(A[i])
                  << " b=" << std::setw(4) << static_cast<int>(B[i])
                  << " hw=" << std::setw(4) << static_cast<int>(host_out[i])
                  << " ref=" << std::setw(4) << static_cast<int>(ref[i])
                  << std::endl;
    }

    std::cout << "[Case] " << tc.name
              << " col=" << col_num
              << " row=" << row_num
              << " random=" << (tc.random ? "yes" : "no")
              << " mismatches=" << mismatch
              << " max_diff=" << max_diff << std::endl;

    CaseStats stats;
    stats.total    = static_cast<size_t>(mat_size);
    stats.mismatch = static_cast<size_t>(mismatch);
    return stats;
}

// ======================================================================
// Part 2: Quantized fp32-domain matadd test
//
// Simulates a real network residual-add:
//   A_fp32, B_fp32 (e.g. two layer outputs)
//   -> symmetric quantize to int8 with scales Sa, Sb
//   -> MVIN dequant into ACC with per-input rescaling
//   -> MATADD (int32 add in ACC)
//   -> output quant (int32 -> int8) via MATADD output_scale/shift
//   -> MVOUT int8 result
//   -> dequant back to fp32 and compare vs fp32 reference
//
// Quantization scheme (symmetric, zero_point = 0):
//   A_q = round(A_fp / Sa),    A_fp ≈ A_q * Sa
//   B_q = round(B_fp / Sb),    B_fp ≈ B_q * Sb
//   C_fp = A_fp + B_fp,        So = max(|C_fp|) / 127
//   C_q  = round(C_fp / So)
//
// Hardware mapping:
//   Pick a fixed-point precision N (number of fractional bits in ACC).
//   MVIN A: quant_scale_a = round(Sa/So * 2^N), shift = 0
//     => ACC_A[i] = A_q[i] * scale_a ≈ (A_fp[i]/So) * 2^N
//   MVIN B: quant_scale_b = round(Sb/So * 2^N), shift = 0
//     => ACC_B[i] = B_q[i] * scale_b ≈ (B_fp[i]/So) * 2^N
//   MATADD: ACC_sum = ACC_A + ACC_B ≈ (C_fp/So) * 2^N
//   Output quant: out = clip((1 * ACC_sum) >>> N, -128, 127)
//     => output_scale = 1, output_scaleshift = -N
//     => out ≈ C_fp / So = C_q
// ======================================================================

static CaseStats run_case_quant(
    const MataddCase& tc,
    int8_t*  host_a,
    int8_t*  host_b,
    int8_t*  host_out
) {
    const int col_num = tc.col_num;
    const int row_num = tc.row_num;
    const int mat_size = col_num * row_num;

    std::mt19937 rng(42 + col_num * 97 + row_num * 53);

    // ---- 1. Generate fp32 data ----
    std::vector<float> A_fp, B_fp;
    fill_inputs_fp32(A_fp, B_fp, mat_size, tc.random, rng);

    // ---- 2. Compute fp32 reference ----
    std::vector<float> C_fp(mat_size);
    for (int i = 0; i < mat_size; ++i) {
        C_fp[i] = A_fp[i] + B_fp[i];
    }

    // ---- 3. Compute quantization scales ----
    float Sa = calc_symmetric_scale(A_fp);
    float Sb = calc_symmetric_scale(B_fp);
    float So = calc_symmetric_scale(C_fp);

    // ---- 4. Quantize inputs to int8 ----
    std::vector<int8_t> A_q(mat_size), B_q(mat_size);
    for (int i = 0; i < mat_size; ++i) {
        A_q[i] = quant_fp32_to_i8(A_fp[i], Sa);
        B_q[i] = quant_fp32_to_i8(B_fp[i], Sb);
    }

    // ---- 5. Compute MVIN dequant scales ----
    // We want ACC_A[i] = A_q[i] * mvin_scale_a ≈ (A_fp[i]/So) * 2^N
    // So mvin_scale_a ≈ (Sa/So) * 2^N.
    // Similarly mvin_scale_b ≈ (Sb/So) * 2^N.
    // Output: out = clip((1 * ACC_sum) >>> N) ≈ C_q
    //
    // We pick N so that the MVIN scales fit in int16 (max 32767).
    float ratio_a = Sa / So;
    float ratio_b = Sb / So;
    float max_ratio = std::max(ratio_a, ratio_b);

    // Find largest N such that max_ratio * 2^N <= 32767
    int N = 0;
    while (N < 30 && max_ratio * static_cast<float>(1 << (N + 1)) <= 32767.0f) {
        N++;
    }

    int16_t mvin_scale_a = static_cast<int16_t>(std::lrint(ratio_a * static_cast<float>(1 << N)));
    int16_t mvin_scale_b = static_cast<int16_t>(std::lrint(ratio_b * static_cast<float>(1 << N)));

    // Output quant: scale=1, shift=-N  =>  out = clip(ACC_sum >>> N)
    int16_t out_scale = 1;
    int16_t out_shift = static_cast<int16_t>(-N);

    // ---- 6. Software reference (exact same integer math as HW) ----
    std::vector<int32_t> ref_acc(mat_size);
    std::vector<int8_t> ref_q(mat_size);
    for (int i = 0; i < mat_size; ++i) {
        int32_t acc_a = hw_mvin_dequant(A_q[i], mvin_scale_a, 0);
        int32_t acc_b = hw_mvin_dequant(B_q[i], mvin_scale_b, 0);
        ref_acc[i] = acc_a + acc_b;
        ref_q[i] = hw_output_quant(ref_acc[i], out_scale, out_shift);
    }

    // Ideal fp32 reference quantized
    std::vector<int8_t> ideal_q(mat_size);
    for (int i = 0; i < mat_size; ++i) {
        ideal_q[i] = quant_fp32_to_i8(C_fp[i], So);
    }

    // ---- 7. Copy data to host buffers ----
    std::memcpy(host_a, A_q.data(), mat_size);
    std::memcpy(host_b, B_q.data(), mat_size);
    std::memset(host_out, 0, mat_size);

    const uint32_t ACC_ADDR_A   = 0x0000;
    const uint32_t ACC_ADDR_B   = 0x2000;
    const uint32_t SPM_ADDR_OUT = 0x4000;

    // ---- 8. MVIN A with dequant scale ----
    npu_dma_mvin(
        /*host_ptr=*/host_a,
        /*sram_addr=*/ACC_ADDR_A,
        /*col_num=*/static_cast<uint16_t>(mat_size - 1),
        /*row_num=*/0,
        /*sram_stride=*/0,
        /*dram_stride=*/0,
        /*precision=*/1,
        /*input_type=*/0,
        /*dest=*/1,          // ACC
        /*is_bias=*/false,
        /*is_quant=*/true,
        /*quant_zero=*/0,
        /*quant_scale=*/static_cast<uint16_t>(mvin_scale_a),
        /*quant_shift=*/0
    );

    // ---- 9. MVIN B with dequant scale ----
    npu_dma_mvin(
        /*host_ptr=*/host_b,
        /*sram_addr=*/ACC_ADDR_B,
        /*col_num=*/static_cast<uint16_t>(mat_size - 1),
        /*row_num=*/0,
        /*sram_stride=*/0,
        /*dram_stride=*/0,
        /*precision=*/1,
        /*input_type=*/0,
        /*dest=*/1,          // ACC
        /*is_bias=*/false,
        /*is_quant=*/true,
        /*quant_zero=*/0,
        /*quant_scale=*/static_cast<uint16_t>(mvin_scale_b),
        /*quant_shift=*/0
    );

    // ---- 10. MATADD with output requant ----
    npu_matadd_run(
        /*input_a_addr=*/ACC_ADDR_A,
        /*input_b_addr=*/ACC_ADDR_B,
        /*output_addr=*/SPM_ADDR_OUT,
        /*col_num=*/static_cast<uint8_t>(col_num),
        /*row_num=*/static_cast<uint8_t>(row_num),
        /*output_zeropoint=*/0,
        /*output_scale=*/static_cast<uint16_t>(out_scale),
        /*output_scaleshift=*/static_cast<uint16_t>(out_shift)
    );

    // ---- 11. MVOUT from SPM ----
    npu_dma_mvout(
        /*host_ptr=*/host_out,
        /*sram_addr=*/SPM_ADDR_OUT,
        /*col_num=*/static_cast<uint16_t>(mat_size - 1),
        /*row_num=*/0,
        /*sram_stride=*/0,
        /*dram_stride=*/0,
        /*precision=*/1,
        /*output_type=*/0,
        /*source=*/0,        // SPM
        /*is_quant=*/false,
        /*quant_zero=*/0,
        /*quant_scale=*/0,
        /*quant_shift=*/0
    );

    // ---- 12. Compare ----
    int mismatch_hw_ref = 0;   // HW vs integer-exact reference
    int mismatch_hw_ideal = 0; // HW vs ideal fp32 reference
    int max_diff_ref = 0;
    int max_diff_ideal = 0;
    float max_fp_err = 0.0f;
    const int print_count = std::min<int>(16, mat_size);

    for (int i = 0; i < mat_size; ++i) {
        int hw_val = static_cast<int>(host_out[i]);
        int ref_val = static_cast<int>(ref_q[i]);
        int ideal_val = static_cast<int>(ideal_q[i]);

        int diff_ref = std::abs(hw_val - ref_val);
        int diff_ideal = std::abs(hw_val - ideal_val);

        // fp32 error: dequant HW output and compare to C_fp
        float hw_fp = static_cast<float>(host_out[i]) * So;
        float fp_err = std::fabs(hw_fp - C_fp[i]);
        if (fp_err > max_fp_err) max_fp_err = fp_err;

        if (diff_ref > MAX_DIFF && mismatch_hw_ref < 5) {
            int row = i / col_num;
            int col = i % col_num;
            std::cout << "[Mismatch-HW-Ref] " << tc.name
                      << " idx=" << i << " (" << row << "," << col << ")"
                      << " hw=" << hw_val
                      << " ref=" << ref_val
                      << " ideal=" << ideal_val
                      << " acc=" << ref_acc[i]
                      << " diff=" << diff_ref << std::endl;
        }
        if (diff_ref > MAX_DIFF) mismatch_hw_ref++;
        if (diff_ref > max_diff_ref) max_diff_ref = diff_ref;

        if (diff_ideal > MAX_DIFF_QUANT) mismatch_hw_ideal++;
        if (diff_ideal > max_diff_ideal) max_diff_ideal = diff_ideal;
    }

    // Print quant parameters
    std::cout << "[Quant] " << tc.name
              << " Sa=" << std::fixed << std::setprecision(6) << Sa
              << " Sb=" << Sb
              << " So=" << So
              << " ratio_a=" << ratio_a
              << " ratio_b=" << ratio_b
              << " N=" << N
              << " mvin_scale_a=" << mvin_scale_a
              << " mvin_scale_b=" << mvin_scale_b
              << " out_shift=" << out_shift << std::endl;

    // Print first few elements with full detail
    std::cout << "[Sample] " << tc.name << " first " << print_count << " elems:" << std::endl;
    for (int i = 0; i < print_count; ++i) {
        int row = i / col_num;
        int col = i % col_num;
        float hw_fp = static_cast<float>(host_out[i]) * So;
        std::cout << "  (" << row << "," << col << ")"
                  << " A_fp=" << std::setw(8) << std::fixed << std::setprecision(4) << A_fp[i]
                  << " B_fp=" << std::setw(8) << B_fp[i]
                  << " C_fp=" << std::setw(8) << C_fp[i]
                  << " | A_q=" << std::setw(4) << static_cast<int>(A_q[i])
                  << " B_q=" << std::setw(4) << static_cast<int>(B_q[i])
                  << " | hw=" << std::setw(4) << static_cast<int>(host_out[i])
                  << " ref=" << std::setw(4) << static_cast<int>(ref_q[i])
                  << " ideal=" << std::setw(4) << static_cast<int>(ideal_q[i])
                  << " | hw_fp=" << std::setw(8) << hw_fp
                  << std::endl;
    }

    std::cout << "[Case] " << tc.name
              << " col=" << col_num
              << " row=" << row_num
              << " random=" << (tc.random ? "yes" : "no")
              << " mismatch_hw_ref=" << mismatch_hw_ref
              << " mismatch_hw_ideal=" << mismatch_hw_ideal
              << " max_diff_ref=" << max_diff_ref
              << " max_diff_ideal=" << max_diff_ideal
              << " max_fp_err=" << max_fp_err << std::endl;

    CaseStats stats;
    stats.total    = static_cast<size_t>(mat_size);
    // Use hw-vs-ref (exact integer match) as the pass/fail criterion
    stats.mismatch = static_cast<size_t>(mismatch_hw_ref);
    return stats;
}

// ==========================================
// Main
// ==========================================
int main() {
    std::cout << "===== NPU MATADD Test (Runtime API) =====" << std::endl;

    if (npu_init() != 0) {
        std::cerr << "npu_init() failed!" << std::endl;
        return 1;
    }

    npu_reset();

    // --------------------------------------------------
    // Part 1: Raw int8 tests (identity quant)
    // --------------------------------------------------
    std::vector<MataddCase> raw_cases = {
        {"raw_4x4",          4,   4, false},
        {"raw_24x15",       24,  15, false},
        {"raw_32x32",       32,  32, false},
        {"raw_16x8",        16,   8, false},
        {"raw_8x24",         8,  24, false},
        {"raw_1x1",          1,   1, false},
        {"raw_rand_24x15",  24,  15, true},
        {"raw_rand_32x32",  32,  32, true},
        {"raw_rand_16x20",  16,  20, true},
        {"raw_rand_10x10",  10,  10, true},
    };

    // --------------------------------------------------
    // Part 2: Quantized fp32-domain tests (real network flow)
    // --------------------------------------------------
    std::vector<MataddCase> quant_cases = {
        {"quant_4x4",         4,   4, false},
        {"quant_24x15",      24,  15, false},
        {"quant_32x32",      32,  32, false},
        {"quant_16x8",       16,   8, false},
        {"quant_8x24",        8,  24, false},
        {"quant_1x1",         1,   1, false},
        {"quant_rand_24x15", 24,  15, true},
        {"quant_rand_32x32", 32,  32, true},
        {"quant_rand_16x20", 16,  20, true},
        {"quant_rand_10x10", 10,  10, true},
        {"quant_rand_30x25", 30,  25, true},
        {"quant_rand_12x32", 12,  32, true},
    };

    // Compute max buffer size needed
    size_t max_size = 0;
    for (const auto& tc : raw_cases) {
        size_t s = static_cast<size_t>(tc.col_num) * tc.row_num;
        if (s > max_size) max_size = s;
    }
    for (const auto& tc : quant_cases) {
        size_t s = static_cast<size_t>(tc.col_num) * tc.row_num;
        if (s > max_size) max_size = s;
    }

    // Allocate host buffers via NPU allocator
    int8_t* host_a   = static_cast<int8_t*>(npu_mem_alloc(max_size));
    int8_t* host_b   = static_cast<int8_t*>(npu_mem_alloc(max_size));
    int8_t* host_out = static_cast<int8_t*>(npu_mem_alloc(max_size));

    if (!host_a || !host_b || !host_out) {
        std::cerr << "npu_mem_alloc() failed!" << std::endl;
        npu_destroy();
        return 1;
    }

    int total_cases = 0;
    int pass = 0;
    size_t total_points = 0;
    size_t fail_points  = 0;

    // Run raw tests
    std::cout << "\n===== Part 1: Raw Int8 MATADD Tests =====" << std::endl;
    for (const auto& tc : raw_cases) {
        CaseStats st = run_case_raw(tc, host_a, host_b, host_out);
        total_points += st.total;
        fail_points  += st.mismatch;
        if (st.mismatch == 0) pass++;
        total_cases++;
    }

    // Run quantized tests
    std::cout << "\n===== Part 2: Quantized FP32-Domain MATADD Tests =====" << std::endl;
    for (const auto& tc : quant_cases) {
        CaseStats st = run_case_quant(tc, host_a, host_b, host_out);
        total_points += st.total;
        fail_points  += st.mismatch;
        if (st.mismatch == 0) pass++;
        total_cases++;
    }

    size_t pass_points = total_points - fail_points;

    std::cout << "\n========== Summary ==========" << std::endl;
    std::cout << "Cases:  total=" << total_cases
              << " pass=" << pass
              << " fail=" << (total_cases - pass) << std::endl;
    std::cout << "Points: total=" << total_points
              << " pass=" << pass_points
              << " fail=" << fail_points << std::endl;

    npu_mem_free(host_a);
    npu_mem_free(host_b);
    npu_mem_free(host_out);
    npu_destroy();

    return (fail_points == 0) ? 0 : 1;
}
