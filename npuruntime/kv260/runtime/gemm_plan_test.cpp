#include "npu_runtime.h"
#include "npu_kv260_uapi.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <vector>

namespace {

constexpr uint32_t kSpmA = 0x00000000;
constexpr uint32_t kSpmB = 0x00010000;
constexpr uint32_t kAccOut = 0x00004000;
constexpr uint32_t kAccScratch = 0x00020000;

struct GemmPlanCase {
    const char* name;
    uint16_t m;
    uint16_t n;
    uint16_t k;
    uint16_t a_stride;
    uint16_t b_stride;
    uint16_t out_stride;
    uint32_t b_dram_stride = 0;
};

bool write_reg32(int fd, uint32_t offset, uint32_t value) {
    npu_kv260_reg_access access = {offset, value};
    if (ioctl(fd, NPU_KV260_IOC_REG_WRITE, &access) != 0) {
        std::fprintf(stderr, "reg_write32 offset=0x%03x failed: %s\n", offset, std::strerror(errno));
        return false;
    }
    return true;
}

bool read_reg32(int fd, uint32_t offset, uint32_t* value) {
    npu_kv260_reg_access access = {offset, 0};
    if (ioctl(fd, NPU_KV260_IOC_REG_READ, &access) != 0) {
        std::fprintf(stderr, "reg_read32 offset=0x%03x failed: %s\n", offset, std::strerror(errno));
        return false;
    }
    *value = access.value;
    return true;
}

bool expect_reg32(int fd, uint32_t offset, uint32_t expected) {
    uint32_t got = 0;
    if (!read_reg32(fd, offset, &got)) {
        return false;
    }
    if (got != expected) {
        std::fprintf(stderr, "reg_mismatch offset=0x%03x expected=0x%08x got=0x%08x\n", offset, expected, got);
        return false;
    }
    return true;
}

bool check_gemm_plan_register_io() {
    int fd = open(NPU_KV260_DEV_PATH, O_RDWR | O_CLOEXEC);
    if (fd < 0) {
        std::fprintf(stderr, "open %s failed: %s\n", NPU_KV260_DEV_PATH, std::strerror(errno));
        return false;
    }

    bool ok = true;
    ok = ok && write_reg32(fd, RegOffset::GEMM_PLAN_0, 0x00210000u);
    ok = ok && write_reg32(fd, RegOffset::GEMM_PLAN_0 + 4, 0x5a5aa5a5u);
    ok = ok && write_reg32(fd, RegOffset::GEMM_PLAN_1, kAccScratch);
    ok = ok && write_reg32(fd, RegOffset::GEMM_PLAN_1 + 4, 0x00000000u);
    ok = ok && write_reg32(fd, RegOffset::GEMM_PLAN_2, 0xcafebabeu);
    ok = ok && write_reg32(fd, RegOffset::GEMM_PLAN_2 + 4, 0x0badf00du);

    ok = ok && expect_reg32(fd, RegOffset::GEMM_PLAN_0, 0x00210000u);
    ok = ok && expect_reg32(fd, RegOffset::GEMM_PLAN_0 + 4, 0x5a5aa5a5u);
    ok = ok && expect_reg32(fd, RegOffset::GEMM_PLAN_1, kAccScratch);
    ok = ok && expect_reg32(fd, RegOffset::GEMM_PLAN_1 + 4, 0x00000000u);
    ok = ok && expect_reg32(fd, RegOffset::GEMM_PLAN_2, 0xcafebabeu);
    ok = ok && expect_reg32(fd, RegOffset::GEMM_PLAN_2 + 4, 0x0badf00du);

    close(fd);
    std::printf("gemm_plan_reg_io=%s\n", ok ? "ok" : "fail");
    return ok;
}

int8_t pattern_a(uint16_t row, uint16_t col) {
    const int value = (static_cast<int>(row) * 7 + static_cast<int>(col) * 3 + 5) % 17;
    return static_cast<int8_t>(value - 8);
}

int8_t pattern_b(uint16_t row, uint16_t col) {
    const int value = (static_cast<int>(row) * 5 + static_cast<int>(col) * 11 + 9) % 13;
    return static_cast<int8_t>(value - 6);
}

void fill_matrix_a(int8_t* ptr, const GemmPlanCase& tc) {
    std::memset(ptr, 0, static_cast<size_t>(tc.m) * tc.a_stride);
    for (uint16_t r = 0; r < tc.m; ++r) {
        int8_t* row = ptr + static_cast<size_t>(r) * tc.a_stride;
        for (uint16_t c = 0; c < tc.k; ++c) {
            row[c] = pattern_a(r, c);
        }
    }
}

void fill_matrix_b(int8_t* ptr, const GemmPlanCase& tc) {
    const uint32_t b_dram_stride = tc.b_dram_stride ? tc.b_dram_stride : tc.b_stride;
    std::memset(ptr, 0, static_cast<size_t>(tc.k) * b_dram_stride);
    for (uint16_t r = 0; r < tc.k; ++r) {
        int8_t* row = ptr + static_cast<size_t>(r) * b_dram_stride;
        for (uint16_t c = 0; c < tc.n; ++c) {
            row[c] = pattern_b(r, c);
        }
    }
}

std::vector<int32_t> build_golden(const GemmPlanCase& tc) {
    std::vector<int32_t> golden(static_cast<size_t>(tc.m) * tc.n, 0);
    for (uint16_t r = 0; r < tc.m; ++r) {
        for (uint16_t c = 0; c < tc.n; ++c) {
            int32_t sum = 0;
            for (uint16_t kk = 0; kk < tc.k; ++kk) {
                sum += static_cast<int32_t>(pattern_a(r, kk)) *
                       static_cast<int32_t>(pattern_b(kk, c));
            }
            golden[static_cast<size_t>(r) * tc.n + c] = sum;
        }
    }
    return golden;
}

bool run_gemm_plan_case(const GemmPlanCase& tc) {
    const size_t a_bytes = static_cast<size_t>(tc.m) * tc.a_stride;
    const uint32_t b_dram_stride = tc.b_dram_stride ? tc.b_dram_stride : tc.b_stride;
    const size_t b_bytes = static_cast<size_t>(tc.k) * b_dram_stride;
    const size_t out_elems = static_cast<size_t>(tc.m) * tc.out_stride;
    const size_t out_bytes = out_elems * sizeof(int32_t);

    auto* a = static_cast<int8_t*>(npu_mem_alloc(a_bytes));
    auto* b = static_cast<int8_t*>(npu_mem_alloc(b_bytes));
    auto* out = static_cast<int32_t*>(npu_mem_alloc(out_bytes));
    if (!a || !b || !out) {
        std::fprintf(stderr, "[%s] npu_mem_alloc failed\n", tc.name);
        if (a) npu_mem_free(a);
        if (b) npu_mem_free(b);
        if (out) npu_mem_free(out);
        return false;
    }

    fill_matrix_a(a, tc);
    fill_matrix_b(b, tc);
    std::memset(out, 0, out_bytes);
    const std::vector<int32_t> golden = build_golden(tc);

    const MvinConfig act_mvin = {
        a,
        kSpmA,
        static_cast<uint32_t>(tc.k - 1),
        static_cast<uint32_t>(tc.m - 1),
        tc.a_stride,
        tc.a_stride,
        1,
        0,
        false,
        false,
        false,
        0,
        0,
        0,
    };
    const MvinConfig weight_mvin = {
        b,
        kSpmB,
        static_cast<uint32_t>(tc.n - 1),
        static_cast<uint32_t>(tc.k - 1),
        tc.b_stride,
        b_dram_stride,
        1,
        1,
        false,
        false,
        false,
        0,
        0,
        0,
    };
    npu_dma_mvin_async(0, &act_mvin);
    npu_dma_wait_mvin(1u << 0);
    npu_dma_mvin_async(1, &weight_mvin);
    npu_dma_wait_mvin(1u << 1);

    npu_gemm_plan_run(kSpmA, kSpmB, kAccOut, kAccScratch, 0,
                      tc.m, tc.n, tc.k,
                      tc.a_stride, tc.b_stride, tc.out_stride, 0,
                      false);

    npu_dma_mvout(out, kAccOut, static_cast<uint32_t>(tc.n - 1), static_cast<uint32_t>(tc.m - 1),
                  tc.out_stride, tc.out_stride, 1, 1, true, false, 0, 0);

    uint32_t mismatches = 0;
    uint16_t first_row = 0;
    uint16_t first_col = 0;
    int32_t first_expected = 0;
    int32_t first_got = 0;
    for (uint16_t r = 0; r < tc.m; ++r) {
        const int32_t* row = out + static_cast<size_t>(r) * tc.out_stride;
        for (uint16_t c = 0; c < tc.n; ++c) {
            const int32_t expected = golden[static_cast<size_t>(r) * tc.n + c];
            if (row[c] != expected) {
                if (mismatches == 0) {
                    first_row = r;
                    first_col = c;
                    first_expected = expected;
                    first_got = row[c];
                }
                ++mismatches;
            }
        }
    }

    if (mismatches != 0) {
        std::fprintf(stderr,
                     "[%s] fail mismatches=%u first_pos=(%u,%u) expected=%d got=%d shape=%ux%ux%u stride=[%u,%u,%u]\n",
                     tc.name, mismatches, first_row, first_col, first_expected, first_got,
                     tc.m, tc.n, tc.k, tc.a_stride, tc.b_stride, tc.out_stride);
    } else {
        int32_t min_expected = golden.front();
        int32_t max_expected = golden.front();
        for (int32_t value : golden) {
            min_expected = std::min(min_expected, value);
            max_expected = std::max(max_expected, value);
        }
        std::printf("[%s] ok shape=%ux%ux%u stride=[%u,%u,%u] golden_range=[%d,%d]\n",
                    tc.name, tc.m, tc.n, tc.k, tc.a_stride, tc.b_stride, tc.out_stride,
                    min_expected, max_expected);
    }

    npu_mem_free(a);
    npu_mem_free(b);
    npu_mem_free(out);
    return mismatches == 0;
}

bool run_gemm_plan_suite() {
    std::vector<GemmPlanCase> cases = {
        {"edge_tiles_17x19x33",         17,   19,   33,   33,   19,   19},
        {"mainpath_16x1024x16",         16,   16, 1024, 1024,   16,   16},
        {"mainpath_64x512x64",          64,   64,  512,  512,   64,   64},
        {"mmproj_tail_b64_128x64x64",  128,   64,   64,   64,   64,   64},
        {"mmproj_tail_stride240_128x64x64", 128, 64, 64, 64, 240, 240, 240},
        {"mmproj_cma_tail_128x64x64",  128,   64,   64,   64,  240,  240, 1024},
        {"mmproj_full_128x240x64",     128,  240,   64,   64,  240,  240, 1024},
    };
    if (std::getenv("NPU_GEMM_PLAN_DIAG")) {
        const GemmPlanCase diag_cases[] = {
            {"diag_16x16x16",            16,   16,   16,   16,   16,   16},
            {"diag_16x16x32",            16,   16,   32,   32,   16,   16},
            {"diag_16x16x33",            16,   16,   33,   33,   16,   16},
            {"diag_16x16x34",            16,   16,   34,   34,   16,   16},
            {"diag_16x16x48",            16,   16,   48,   48,   16,   16},
            {"diag_17x16x48",            17,   16,   48,   48,   16,   16},
            {"diag_16x17x48",            16,   17,   48,   48,   17,   17},
            {"diag_16x16x64",            16,   16,   64,   64,   16,   16},
        };
        cases.insert(cases.end(), std::begin(diag_cases), std::end(diag_cases));
    }
    if (const char* only = std::getenv("NPU_GEMM_PLAN_ONLY")) {
        std::vector<GemmPlanCase> filtered;
        for (const auto& tc : cases) {
            if (std::strcmp(tc.name, only) == 0) {
                filtered.push_back(tc);
            }
        }
        if (filtered.empty()) {
            std::fprintf(stderr, "unknown NPU_GEMM_PLAN_ONLY case: %s\n", only);
            return false;
        }
        cases.swap(filtered);
    }

    bool all_ok = true;
    for (const auto& tc : cases) {
        if (!run_gemm_plan_case(tc)) {
            all_ok = false;
        }
    }
    return all_ok;
}

} // namespace

int main() {
    std::puts("kv260_gemm_plan_test: register IO + multi-scenario GEMM plan suite");

    if (npu_init() != 0) {
        std::fprintf(stderr, "npu_init failed\n");
        return 1;
    }

    const bool reg_ok = check_gemm_plan_register_io();
    const bool suite_ok = run_gemm_plan_suite();

    npu_destroy();

    if (!reg_ok || !suite_ok) {
        std::puts("kv260_gemm_plan_test=fail");
        return 3;
    }

    std::puts("kv260_gemm_plan_test=ok");
    return 0;
}
