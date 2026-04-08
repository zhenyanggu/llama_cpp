/**
/**
 * @file layout_convert_test.cpp
 * @brief Layout Convert Test - 验证 NCHW <-> NCHWC32 / NHWC 转换正确性
 *
 * 使用方法:
 *   make $(PREFIX)_layout
 *   ./$(PREFIX)_layout
 */

#include "../npu_runtime.h"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

// ==========================================
// 测试用例结构
// ==========================================

struct LayoutCase {
    std::string name;
    uint16_t n;
    uint16_t c;
    uint16_t h;
    uint16_t w;
};

// ==========================================
// 软件参考实现
// ==========================================

static void nchw_to_nhwc(const int8_t* in, int8_t* out,
                         uint16_t n, uint16_t c, uint16_t h, uint16_t w) {
    for (uint16_t bn = 0; bn < n; ++bn) {
        for (uint16_t hh = 0; hh < h; ++hh) {
            for (uint16_t ww = 0; ww < w; ++ww) {
                for (uint16_t cc = 0; cc < c; ++cc) {
                    size_t in_idx = ((bn * c + cc) * h + hh) * w + ww;
                    size_t out_idx = ((bn * h + hh) * w + ww) * c + cc;
                    out[out_idx] = in[in_idx];
                }
            }
        }
    }
}

static void nhwc_to_nchw(const int8_t* in, int8_t* out,
                         uint16_t n, uint16_t c, uint16_t h, uint16_t w) {
    for (uint16_t bn = 0; bn < n; ++bn) {
        for (uint16_t hh = 0; hh < h; ++hh) {
            for (uint16_t ww = 0; ww < w; ++ww) {
                for (uint16_t cc = 0; cc < c; ++cc) {
                    size_t in_idx = ((bn * h + hh) * w + ww) * c + cc;
                    size_t out_idx = ((bn * c + cc) * h + hh) * w + ww;
                    out[out_idx] = in[in_idx];
                }
            }
        }
    }
}

static void nchw_to_nchwc32(const int8_t* in, int8_t* out,
                            uint16_t n, uint16_t c, uint16_t h, uint16_t w) {
    uint16_t groups = static_cast<uint16_t>((c + 31) / 32);
    for (uint16_t bn = 0; bn < n; ++bn) {
        for (uint16_t g = 0; g < groups; ++g) {
            for (uint16_t hh = 0; hh < h; ++hh) {
                for (uint16_t ww = 0; ww < w; ++ww) {
                    for (uint16_t ci = 0; ci < 32; ++ci) {
                        uint16_t cc = static_cast<uint16_t>(g * 32 + ci);
                        size_t out_idx = ((((bn * groups + g) * h + hh) * w + ww) * 32) + ci;
                        if (cc < c) {
                            size_t in_idx = ((bn * c + cc) * h + hh) * w + ww;
                            out[out_idx] = in[in_idx];
                        } else {
                            out[out_idx] = 0;
                        }
                    }
                }
            }
        }
    }
}

static void nchwc32_to_nchw(const int8_t* in, int8_t* out,
                            uint16_t n, uint16_t c, uint16_t h, uint16_t w) {
    uint16_t groups = c / 32;
    for (uint16_t bn = 0; bn < n; ++bn) {
        for (uint16_t g = 0; g < groups; ++g) {
            for (uint16_t hh = 0; hh < h; ++hh) {
                for (uint16_t ww = 0; ww < w; ++ww) {
                    for (uint16_t ci = 0; ci < 32; ++ci) {
                        uint16_t cc = static_cast<uint16_t>(g * 32 + ci);
                        size_t in_idx = ((((bn * groups + g) * h + hh) * w + ww) * 32) + ci;
                        size_t out_idx = ((bn * c + cc) * h + hh) * w + ww;
                        out[out_idx] = in[in_idx];
                    }
                }
            }
        }
    }
}

// ==========================================
// 工具函数
// ==========================================

static size_t count_mismatch(const int8_t* a, const int8_t* b, size_t size) {
    size_t mismatches = 0;
    for (size_t i = 0; i < size; ++i) {
        if (a[i] != b[i]) {
            ++mismatches;
        }
    }
    return mismatches;
}

// ==========================================
// 单个测试用例执行
// ==========================================

static bool run_layout_case(const LayoutCase& tc,
                            int8_t* host_in,
                            int8_t* host_out,
                            int8_t* host_back) {
    const uint32_t hw = static_cast<uint32_t>(tc.h) * static_cast<uint32_t>(tc.w);
    const size_t input_size = static_cast<size_t>(tc.n) * tc.c * hw;
    const uint16_t groups = (tc.c >= 32) ? static_cast<uint16_t>((tc.c + 31) / 32) : 0;
    const size_t output_size = (tc.c < 32)
        ? input_size
        : static_cast<size_t>(tc.n) * groups * hw * 32u;
    const bool need_backward = (tc.c < 32) || (tc.c % 32 == 0);

    std::cout << "\n========================================\n";
    std::cout << "[TestCase] " << tc.name
              << " (N=" << tc.n << ", C=" << tc.c << ", H=" << tc.h << ", W=" << tc.w << ")\n";
    std::cout << "========================================\n";

    // 1. 生成输入
    for (size_t i = 0; i < input_size; ++i) {
        host_in[i] = static_cast<int8_t>((i * 7) % 127);
    }

    // 2. 软件参考
    std::vector<int8_t> ref_out(output_size, 0);
    std::vector<int8_t> ref_back(input_size, 0);

    if (tc.c < 32) {
        nchw_to_nhwc(host_in, ref_out.data(), tc.n, tc.c, tc.h, tc.w);
        nhwc_to_nchw(ref_out.data(), ref_back.data(), tc.n, tc.c, tc.h, tc.w);
    } else {
        nchw_to_nchwc32(host_in, ref_out.data(), tc.n, tc.c, tc.h, tc.w);
        if (need_backward) {
            nchwc32_to_nchw(ref_out.data(), ref_back.data(), tc.n, tc.c, tc.h, tc.w);
        }
    }

    std::memset(host_out, 0, output_size);
    std::memset(host_back, 0, input_size);

    // SPM 地址: 0x40000 分成两半
    const uint32_t SPM_ADDR_IN = 0x00000;
    const uint32_t SPM_ADDR_OUT = 0x20000;

    // 3. MVIN: Host -> SPM_IN
    npu_dma_mvin(
        host_in,
        SPM_ADDR_IN,
        static_cast<uint16_t>(input_size - 1),
        0,
        static_cast<uint16_t>(input_size - 1),
        static_cast<uint32_t>(input_size - 1),
        1,
        0,
        0,
        false,
        false,
        0,
        0,
        0
    );

    // 4. Layout Convert (Forward)
    npu_layout_nchw_to_nchwc32(SPM_ADDR_IN, SPM_ADDR_OUT, tc.n, tc.c, tc.h, tc.w);

    // 5. MVOUT: SPM_OUT -> Host
    npu_dma_mvout(
        host_out,
        SPM_ADDR_OUT,
        static_cast<uint16_t>(output_size - 1),
        0,
        static_cast<uint16_t>(output_size - 1),
        static_cast<uint32_t>(output_size - 1),
        1,
        0,
        0,
        false,
        0,
        0,
        0
    );

    // 6. 将前一半写入脏数据，再做反向转换回写前一半
    if (need_backward) {
        std::memset(host_in, 0x5A, input_size);
        npu_dma_mvin(
            host_in,
            SPM_ADDR_IN,
            static_cast<uint16_t>(input_size - 1),
            0,
            static_cast<uint16_t>(input_size - 1),
            static_cast<uint32_t>(input_size - 1),
            1,
            0,
            0,
            false,
            false,
            0,
            0,
            0
        );

        // 7. Layout Convert (Backward) - 输出回写前一半
        npu_layout_nchwc32_to_nchw(SPM_ADDR_OUT, SPM_ADDR_IN, tc.n, tc.c, tc.h, tc.w);

        // 8. MVOUT: SPM_IN -> Host
        npu_dma_mvout(
            host_back,
            SPM_ADDR_IN,
            static_cast<uint16_t>(input_size - 1),
            0,
            static_cast<uint16_t>(input_size - 1),
            static_cast<uint32_t>(input_size - 1),
            1,
            0,
            0,
            false,
            0,
            0,
            0
        );
    }

    // 9. 比较结果
    size_t mismatches_out = count_mismatch(host_out, ref_out.data(), output_size);
    size_t mismatches_back = need_backward ? count_mismatch(host_back, ref_back.data(), input_size) : 0;

    if (mismatches_out == 0 && (!need_backward || mismatches_back == 0)) {
        std::cout << (need_backward ? "[PASS] Forward/Backward layout conversion" : "[PASS] Forward layout conversion") << "\n";
        return true;
    }

    std::cout << "[FAIL] mismatches (forward=" << mismatches_out;
    if (need_backward) {
        std::cout << ", backward=" << mismatches_back;
    }
    std::cout << ")\n";
    return false;
}

// ==========================================
// Main
// ==========================================

int main() {
    std::cout << "=========================================\n";
    std::cout << "   NPU Layout Convert Test (Runtime API)\n";
    std::cout << "=========================================\n";

    if (npu_init() != 0) {
        std::cerr << "[FATAL] NPU initialization failed!\n";
        return -1;
    }

    npu_reset();

    // 模拟卷积网络中的两类层：
    // 1) 首层输入 C<32 -> NHWC
    // 2) 中间层 C>=32 -> NCHWC32
    std::vector<LayoutCase> cases = {
        {"Layer0_C3", 1, 3, 8, 8},
        {"Layer1_C48", 1, 48, 8, 8},
        {"Layer1_C64", 1, 64, 8, 8},
        {"Layer2_C64", 1, 64, 32, 32},
        {"Layer3_C96", 1, 96, 16, 16},
        {"Layer4_C512", 1, 512, 3, 3},
        // Tests for Channel Padding (C % 32 != 0)
        {"Layer_Pad_C33", 1, 33, 8, 8},  // 32 + 1
        {"Layer_Pad_C47", 1, 47, 8, 8},  // 32 + 15
        {"Layer_Pad_C63", 1, 63, 8, 8}   // 32 + 31 (64-1)
    };

    const size_t SPM_HALF_SIZE = 0x20000; // 0x40000 分两半
    size_t max_size = SPM_HALF_SIZE;
    int8_t* host_in = static_cast<int8_t*>(npu_mem_alloc(max_size));
    int8_t* host_out = static_cast<int8_t*>(npu_mem_alloc(max_size));
    int8_t* host_back = static_cast<int8_t*>(npu_mem_alloc(max_size));

    if (!host_in || !host_out || !host_back) {
        std::cerr << "[FATAL] Memory allocation failed!\n";
        npu_destroy();
        return -1;
    }

    int total_tests = 0;
    int passed_tests = 0;

    for (const auto& tc : cases) {
        uint32_t hw = static_cast<uint32_t>(tc.h) * static_cast<uint32_t>(tc.w);
        if (hw > 4096) {
            std::cout << "[SKIP] " << tc.name << " (H*W exceeds 4096)\n";
            continue;
        }
        size_t input_size = static_cast<size_t>(tc.n) * tc.c * hw;
        uint16_t groups = (tc.c >= 32) ? static_cast<uint16_t>((tc.c + 31) / 32) : 0;
        size_t output_size = (tc.c < 32)
            ? input_size
            : static_cast<size_t>(tc.n) * groups * hw * 32u;
        size_t required = std::max(input_size, output_size);
        if (required > SPM_HALF_SIZE) {
            std::cout << "[SKIP] " << tc.name << " (buffer too small)\n";
            continue;
        }

        total_tests++;
        if (run_layout_case(tc, host_in, host_out, host_back)) {
            passed_tests++;
        }
    }

    std::cout << "\n=========================================\n";
    std::cout << "   Test Summary\n";
    std::cout << "=========================================\n";
    std::cout << "Total: " << passed_tests << "/" << total_tests << " tests passed\n";

    npu_mem_free(host_in);
    npu_mem_free(host_out);
    npu_mem_free(host_back);
    npu_destroy();

    return (passed_tests == total_tests) ? 0 : 1;
}
