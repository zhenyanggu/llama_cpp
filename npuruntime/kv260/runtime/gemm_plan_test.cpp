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

uint32_t packed_w_bytes(uint32_t k, uint32_t n) {
    return k * ((n + 31u) / 32u) * 32u;
}

void pack_matrix_b_versa(
        const std::vector<int8_t>& src,
        uint32_t src_k0,
        uint32_t total_n,
        uint32_t src_n0,
        uint32_t k_count,
        uint32_t n_count,
        int8_t* dst) {
    std::memset(dst, 0, packed_w_bytes(k_count, n_count));
    const uint32_t groups = (n_count + 31u) / 32u;
    for (uint32_t group = 0; group < groups; ++group) {
        const uint32_t group_cols = std::min(32u, n_count - group * 32u);
        for (uint32_t kk = 0; kk < k_count; ++kk) {
            for (uint32_t lane = 0; lane < group_cols; ++lane) {
                dst[((size_t)group * k_count + kk) * 32u + lane] =
                    src[(size_t)(src_k0 + kk) * total_n + src_n0 + group * 32u + lane];
            }
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

std::vector<int32_t> build_golden_chunk(
        const std::vector<int8_t>& a,
        uint32_t a_stride,
        const std::vector<int8_t>& b,
        uint32_t rows,
        uint32_t n,
        uint32_t total_k,
        uint32_t total_m,
        uint32_t m0,
        uint32_t k0,
        uint32_t k_count,
        bool accumulate,
        const std::vector<int32_t>& base,
        const std::vector<int32_t>& bias,
        bool u8_minus_128) {
    std::vector<int32_t> out = accumulate ? base : std::vector<int32_t>((size_t)rows * n, 0);
    if (!accumulate && !bias.empty()) {
        for (uint32_t r = 0; r < rows; ++r) {
            for (uint32_t c = 0; c < n; ++c) {
                out[(size_t)r * n + c] = bias[m0 + c];
            }
        }
    }
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < n; ++c) {
            int32_t sum = 0;
            for (uint32_t kk = 0; kk < k_count; ++kk) {
                const int32_t aval = u8_minus_128
                    ? static_cast<int32_t>(static_cast<uint8_t>(a[(size_t)r * a_stride + k0 + kk])) - 128
                    : static_cast<int32_t>(a[(size_t)r * a_stride + k0 + kk]);
                sum += aval *
                       static_cast<int32_t>(b[(size_t)(k0 + kk) * total_m + m0 + c]);
            }
            out[(size_t)r * n + c] += sum;
        }
    }
    (void)total_k;
    return out;
}

bool check_i32_matrix(
        const char* name,
        const int32_t* got,
        uint32_t got_stride,
        const std::vector<int32_t>& ref,
        uint32_t rows,
        uint32_t n) {
    uint32_t mismatches = 0;
    uint32_t first_row = 0;
    uint32_t first_col = 0;
    int32_t first_expected = 0;
    int32_t first_got = 0;
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < n; ++c) {
            const int32_t expected = ref[(size_t)r * n + c];
            const int32_t value = got[(size_t)r * got_stride + c];
            if (value != expected) {
                if (mismatches == 0) {
                    first_row = r;
                    first_col = c;
                    first_expected = expected;
                    first_got = value;
                }
                ++mismatches;
            }
        }
    }
    if (mismatches != 0) {
        std::fprintf(stderr,
                     "[%s] fail mismatches=%u first_pos=(%u,%u) expected=%d got=%d\n",
                     name, mismatches, first_row, first_col, first_expected, first_got);
        return false;
    }
    std::printf("[%s] ok\n", name);
    return true;
}

bool run_w_pingpong_large_case() {
    constexpr uint32_t rows = 256;
    constexpr uint32_t n = 480;
    constexpr uint32_t total_m = 960;
    constexpr uint32_t total_k = 960;
    constexpr uint32_t k_chunks[] = {768, 192};
    constexpr uint32_t a_stride = 960;
    constexpr uint32_t out_stride = 480;
    constexpr bool u8_minus_128 = true;

    std::vector<int8_t> a((size_t)rows * a_stride, 0);
    std::vector<int8_t> b((size_t)total_k * total_m, 0);
    std::vector<int32_t> bias(total_m, 0);
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t k = 0; k < total_k; ++k) {
            if (u8_minus_128) {
                const uint32_t payload = (r * 19u + k * 7u + 53u) & 0xffu;
                a[(size_t)r * a_stride + k] = static_cast<int8_t>(static_cast<uint8_t>(payload));
            } else {
                a[(size_t)r * a_stride + k] = pattern_a((uint16_t)r, (uint16_t)k);
            }
        }
    }
    for (uint32_t k = 0; k < total_k; ++k) {
        for (uint32_t c = 0; c < total_m; ++c) {
            const int32_t value = (int32_t)((k * 13u + c * 5u + 11u) % 255u) - 127;
            b[(size_t)k * total_m + c] = static_cast<int8_t>(value);
        }
    }
    for (uint32_t c = 0; c < total_m; ++c) {
        bias[c] = static_cast<int32_t>((c * 17u) % 1009u) - 503;
    }

    auto* a_cma = static_cast<int8_t*>(npu_mem_alloc((size_t)rows * 768));
    auto* w0_cma = static_cast<int8_t*>(npu_mem_alloc(packed_w_bytes(768, n)));
    auto* w1_cma = static_cast<int8_t*>(npu_mem_alloc(packed_w_bytes(768, n)));
    auto* bias_cma = static_cast<int32_t*>(npu_mem_alloc(total_m * sizeof(int32_t)));
    auto* out_cma = static_cast<int32_t*>(npu_mem_alloc((size_t)rows * out_stride * sizeof(int32_t)));
    if (!a_cma || !w0_cma || !w1_cma || !bias_cma || !out_cma) {
        std::fprintf(stderr, "[w_pingpong_large] npu_mem_alloc failed\n");
        if (a_cma) npu_mem_free(a_cma);
        if (w0_cma) npu_mem_free(w0_cma);
        if (w1_cma) npu_mem_free(w1_cma);
        if (bias_cma) npu_mem_free(bias_cma);
        if (out_cma) npu_mem_free(out_cma);
        return false;
    }
    std::memcpy(bias_cma, bias.data(), total_m * sizeof(int32_t));
    std::memset(out_cma, 0, (size_t)rows * out_stride * sizeof(int32_t));
    npu_dma_mvin(
        bias_cma, 0, total_m, 0, (uint16_t)total_m, total_m,
        1, 2, true, true, false, 0, 0, 0);

    bool ok = true;
    uint32_t k0 = 0;
    uint8_t current_w_bank = 0;
    int8_t* current_w = w0_cma;
    int8_t* next_w = w1_cma;
    std::vector<int32_t> ref((size_t)rows * n, 0);

    uint32_t m0 = 0;
    pack_matrix_b_versa(b, 0, total_m, m0, k_chunks[0], n, current_w);
    MvinConfig weight_mvin = {
        current_w, kSpmB, n, k_chunks[0],
        (uint16_t)n, n, 1, 1, false, false, false, 0, 0, 0,
    };
    npu_dma_mvin_w_async_bank(current_w_bank, &weight_mvin);
    npu_dma_wait_w_bank(current_w_bank);

    constexpr size_t n_k_chunks = sizeof(k_chunks) / sizeof(k_chunks[0]);
    for (uint32_t macro_m = 0; macro_m < 2; ++macro_m) {
    m0 = macro_m * n;
    k0 = 0;
    if (macro_m != 0) {
        current_w_bank ^= 1u;
        std::swap(current_w, next_w);
        pack_matrix_b_versa(b, 0, total_m, m0, k_chunks[0], n, current_w);
        weight_mvin.host_ptr = current_w;
        weight_mvin.row_num = k_chunks[0];
        npu_dma_mvin_w_async_bank(current_w_bank, &weight_mvin);
        npu_dma_wait_w_bank(current_w_bank);
        std::fill(ref.begin(), ref.end(), 0);
    }
    for (size_t chunk_idx = 0; chunk_idx < n_k_chunks; ++chunk_idx) {
        const uint32_t k_count = k_chunks[chunk_idx];
        for (uint32_t r = 0; r < rows; ++r) {
            std::memcpy(a_cma + (size_t)r * k_count,
                        a.data() + (size_t)r * a_stride + k0,
                        k_count);
        }
        const MvinConfig act_mvin = {
            a_cma, kSpmA, k_count, rows,
            (uint16_t)k_count, k_count, 1, 0, false, false, u8_minus_128, 0, 0, 0,
        };
        npu_dma_mvin_a_async_bank(0, &act_mvin);
        npu_dma_wait_a_bank(0);

        npu_gemm_plan_start_ex_bank(
            current_w_bank,
            kSpmA,
            kSpmB,
            kAccOut,
            kAccScratch,
            m0 * sizeof(int32_t),
            (uint16_t)rows,
            (uint16_t)n,
            (uint16_t)k_count,
            (uint16_t)k_count,
            (uint16_t)n,
            (uint16_t)out_stride,
            0,
            chunk_idx == 0,
            chunk_idx != 0,
            u8_minus_128);

        const bool have_next = chunk_idx + 1 < n_k_chunks;
        uint8_t next_bank = current_w_bank ^ 1u;
        if (have_next) {
            const uint32_t next_k0 = k0 + k_count;
            const uint32_t next_k_count = k_chunks[chunk_idx + 1];
            pack_matrix_b_versa(b, next_k0, total_m, m0, next_k_count, n, next_w);
            MvinConfig next_weight_mvin = {
                next_w, kSpmB, n, next_k_count,
                (uint16_t)n, n, 1, 1, false, false, false, 0, 0, 0,
            };
            npu_dma_mvin_w_async_bank(next_bank, &next_weight_mvin);
        }

        npu_gemm_plan_wait();
        if (have_next) {
            npu_dma_wait_w_bank(next_bank);
        }
        ref = build_golden_chunk(a, a_stride, b, rows, n, total_k,
                                 total_m, m0, k0, k_count, chunk_idx != 0, ref,
                                 bias, u8_minus_128);
        if (!have_next) {
            npu_dma_mvout(out_cma, kAccOut, n, rows,
                          out_stride, out_stride, 1, 1, true, false, 0, 0);
            char name[96];
            std::snprintf(name, sizeof(name), "w_pingpong_large_m%u_final", m0);
            ok = check_i32_matrix(name, out_cma, out_stride, ref, rows, n) && ok;
        }

        if (have_next) {
            k0 += k_count;
            current_w_bank = next_bank;
            std::swap(current_w, next_w);
        }
    }
    }

    npu_mem_free(a_cma);
    npu_mem_free(w0_cma);
    npu_mem_free(w1_cma);
    npu_mem_free(bias_cma);
    npu_mem_free(out_cma);
    std::printf("w_pingpong_large=%s\n", ok ? "ok" : "fail");
    return ok;
}

bool run_ffnup_u256_v512_repro_case() {
    constexpr uint32_t rows = 256;
    constexpr uint32_t total_m = 768;
    constexpr uint32_t total_k = 3072;
    constexpr uint32_t k_chunk = 768;
    constexpr uint32_t a_stride = total_k;
    constexpr uint32_t out_stride = total_m;
    constexpr bool u8_minus_128 = true;
    const uint32_t macro_cols[] = {512, 256};
    const uint32_t macro_m0[] = {0, 512};

    std::vector<int8_t> a((size_t)rows * a_stride, 0);
    std::vector<int8_t> b((size_t)total_k * total_m, 0);
    std::vector<int32_t> bias(total_m, 0);
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t k = 0; k < total_k; ++k) {
            const uint32_t payload = (r * 19u + k * 7u + 53u) & 0xffu;
            a[(size_t)r * a_stride + k] =
                static_cast<int8_t>(static_cast<uint8_t>(payload));
        }
    }
    for (uint32_t k = 0; k < total_k; ++k) {
        for (uint32_t c = 0; c < total_m; ++c) {
            const int32_t value = (int32_t)((k * 13u + c * 5u + 11u) % 255u) - 127;
            b[(size_t)k * total_m + c] = static_cast<int8_t>(value);
        }
    }
    for (uint32_t c = 0; c < total_m; ++c) {
        bias[c] = static_cast<int32_t>((c * 17u) % 1009u) - 503;
    }

    auto* a_cma = static_cast<int8_t*>(npu_mem_alloc((size_t)rows * k_chunk));
    auto* w0_cma = static_cast<int8_t*>(npu_mem_alloc(packed_w_bytes(k_chunk, 512)));
    auto* w1_cma = static_cast<int8_t*>(npu_mem_alloc(packed_w_bytes(k_chunk, 512)));
    auto* bias_cma = static_cast<int32_t*>(npu_mem_alloc(total_m * sizeof(int32_t)));
    auto* out_cma = static_cast<int32_t*>(npu_mem_alloc((size_t)rows * out_stride * sizeof(int32_t)));
    if (!a_cma || !w0_cma || !w1_cma || !bias_cma || !out_cma) {
        std::fprintf(stderr, "[ffnup_u256_v512] npu_mem_alloc failed\n");
        return false;
    }
    std::memcpy(bias_cma, bias.data(), total_m * sizeof(int32_t));
    std::memset(out_cma, 0, (size_t)rows * out_stride * sizeof(int32_t));
    npu_dma_mvin(
        bias_cma, 0, total_m, 0, (uint16_t)total_m, total_m,
        1, 2, true, true, false, 0, 0, 0);

    uint8_t current_w_bank = 0;
    uint8_t current_a_bank = 1;
    bool prefetched_a_valid = false;
    int8_t* current_w = w0_cma;
    int8_t* next_w = w1_cma;

    for (uint32_t macro = 0; macro < 2; ++macro) {
        const uint32_t cols = macro_cols[macro];
        const uint32_t m0 = macro_m0[macro];
        current_w_bank = 0;
        current_a_bank ^= 1u;
        prefetched_a_valid = false;
        current_w = w0_cma;
        next_w = w1_cma;

        pack_matrix_b_versa(b, 0, total_m, m0, k_chunk, cols, current_w);
        MvinConfig weight_mvin = {
            current_w, kSpmB, cols, k_chunk,
            (uint16_t)cols, cols, 1, 1, false, false, false, 0, 0, 0,
        };
        npu_dma_mvin_w_async_bank(current_w_bank, &weight_mvin);
        npu_dma_wait_w_bank(current_w_bank);

        for (uint32_t chunk = 0; chunk < 4; ++chunk) {
            const uint32_t k0 = chunk * k_chunk;
            if (!prefetched_a_valid) {
                for (uint32_t r = 0; r < rows; ++r) {
                    std::memcpy(a_cma + (size_t)r * k_chunk,
                                a.data() + (size_t)r * a_stride + k0,
                                k_chunk);
                }
                const MvinConfig act_mvin = {
                    a_cma, kSpmA, k_chunk, rows,
                    (uint16_t)k_chunk, k_chunk, 1, 0, false, false, u8_minus_128, 0, 0, 0,
                };
                npu_dma_mvin_a_async_bank(current_a_bank, &act_mvin);
                npu_dma_wait_a_bank(current_a_bank);
            }

            npu_gemm_plan_start_ex_banks(
                current_a_bank,
                current_w_bank,
                macro == 0 ? 0 : 1,
                kSpmA,
                kSpmB,
                kAccOut,
                kAccScratch,
                chunk == 0 ? m0 * sizeof(int32_t) : 0,
                (uint16_t)rows,
                (uint16_t)cols,
                (uint16_t)k_chunk,
                (uint16_t)k_chunk,
                (uint16_t)cols,
                (uint16_t)out_stride,
                0,
                chunk == 0,
                chunk != 0,
                u8_minus_128);

            const bool have_next = chunk + 1 < 4;
            const uint8_t next_w_bank = current_w_bank ^ 1u;
            const uint8_t next_a_bank = current_a_bank ^ 1u;
            if (have_next) {
                const uint32_t next_k0 = k0 + k_chunk;
                for (uint32_t r = 0; r < rows; ++r) {
                    std::memcpy(a_cma + (size_t)r * k_chunk,
                                a.data() + (size_t)r * a_stride + next_k0,
                                k_chunk);
                }
                const MvinConfig next_act_mvin = {
                    a_cma, kSpmA, k_chunk, rows,
                    (uint16_t)k_chunk, k_chunk, 1, 0, false, false, u8_minus_128, 0, 0, 0,
                };
                npu_dma_mvin_a_async_bank(next_a_bank, &next_act_mvin);
                pack_matrix_b_versa(b, next_k0, total_m, m0, k_chunk, cols, next_w);
                MvinConfig next_weight_mvin = {
                    next_w, kSpmB, cols, k_chunk,
                    (uint16_t)cols, cols, 1, 1, false, false, false, 0, 0, 0,
                };
                npu_dma_mvin_w_async_bank(next_w_bank, &next_weight_mvin);
                npu_dma_wait_a_bank(next_a_bank);
                npu_dma_wait_w_bank(next_w_bank);
            }
            npu_gemm_plan_wait();
            if (have_next) {
                current_w_bank = next_w_bank;
                current_a_bank = next_a_bank;
                prefetched_a_valid = true;
                std::swap(current_w, next_w);
            } else if (macro + 1 < 2) {
                current_a_bank ^= 1u;
                for (uint32_t r = 0; r < rows; ++r) {
                    std::memcpy(a_cma + (size_t)r * k_chunk,
                                a.data() + (size_t)r * a_stride,
                                k_chunk);
                }
                const MvinConfig next_macro_act_mvin = {
                    a_cma, kSpmA, k_chunk, rows,
                    (uint16_t)k_chunk, k_chunk, 1, 0, false, false, u8_minus_128, 0, 0, 0,
                };
                npu_dma_mvin_a_async_bank(current_a_bank, &next_macro_act_mvin);
                npu_dma_wait_a_bank(current_a_bank);
            }
        }

        const MvoutConfig mvout_cfg {
            out_cma + m0,
            kAccOut,
            cols,
            rows,
            (uint16_t)out_stride,
            out_stride,
            1,
            1,
            true,
            false,
            0,
            0,
            false,
        };
        npu_dma_mvout_async_bank(macro == 0 ? 0 : 1, 2, &mvout_cfg);
        if (macro != 0) {
            npu_dma_wait_mvout(1u << 2);
        }
    }

    npu_mem_free(a_cma);
    npu_mem_free(w0_cma);
    npu_mem_free(w1_cma);
    npu_mem_free(bias_cma);
    npu_mem_free(out_cma);
    std::puts("ffnup_u256_v512_repro=ok");
    return true;
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
    if (std::getenv("NPU_GEMM_PLAN_FFNUP_U256_V512_REPRO")) {
        return run_ffnup_u256_v512_repro_case();
    }
    if (std::getenv("NPU_GEMM_PLAN_W_PINGPONG_LARGE")) {
        return run_w_pingpong_large_case();
    }

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

    const bool reg_ok = std::getenv("NPU_GEMM_PLAN_SKIP_REG_IO")
        ? true
        : check_gemm_plan_register_io();
    const bool suite_ok = run_gemm_plan_suite();

    npu_destroy();

    if (!reg_ok || !suite_ok) {
        std::puts("kv260_gemm_plan_test=fail");
        return 3;
    }

    std::puts("kv260_gemm_plan_test=ok");
    return 0;
}
