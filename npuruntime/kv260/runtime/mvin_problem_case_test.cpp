#include "npu_runtime.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>

namespace {

struct Config {
    uint32_t sram_addr = 0;
    uint32_t col_num = 767;       // 768 elements per row
    uint32_t row_num = 31;        // 32 rows
    uint16_t sram_stride = 768;
    uint32_t dram_stride = 768;
    int loops = 100;
};

uint32_t parse_u32(const char * s, uint32_t fallback) {
    if (s == nullptr || s[0] == '\0') {
        return fallback;
    }
    const unsigned long long v = std::strtoull(s, nullptr, 0);
    return static_cast<uint32_t>(v);
}

Config parse_config(int argc, char ** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--loops") == 0 && i + 1 < argc) {
            const int v = std::atoi(argv[++i]);
            cfg.loops = v > 0 ? v : cfg.loops;
            continue;
        }
        if (std::strcmp(argv[i], "--col") == 0 && i + 1 < argc) {
            cfg.col_num = parse_u32(argv[++i], cfg.col_num);
            continue;
        }
        if (std::strcmp(argv[i], "--row") == 0 && i + 1 < argc) {
            cfg.row_num = parse_u32(argv[++i], cfg.row_num);
            continue;
        }
        if (std::strcmp(argv[i], "--sram-stride") == 0 && i + 1 < argc) {
            cfg.sram_stride = static_cast<uint16_t>(parse_u32(argv[++i], cfg.sram_stride));
            continue;
        }
        if (std::strcmp(argv[i], "--dram-stride") == 0 && i + 1 < argc) {
            cfg.dram_stride = parse_u32(argv[++i], cfg.dram_stride);
            continue;
        }
        if (std::strcmp(argv[i], "--sram-addr") == 0 && i + 1 < argc) {
            cfg.sram_addr = parse_u32(argv[++i], cfg.sram_addr);
            continue;
        }
    }
    return cfg;
}

} // namespace

int main(int argc, char ** argv) {
    const Config cfg = parse_config(argc, argv);
    const size_t rows = static_cast<size_t>(cfg.row_num + 1);
    const size_t cols = static_cast<size_t>(cfg.col_num + 1);
    const size_t bytes = rows * cols;
    const int loops = cfg.loops;

    std::printf("mvin_problem_case_test: loops=%d rows=%zu cols=%zu bytes=%zu\n",
                loops, rows, cols, bytes);
    std::printf(
        "MVIN params: col=%u row=%u sram_stride=%u dram_stride=%u precision=1 input_type=0 dest=SPM\n",
        cfg.col_num, cfg.row_num, cfg.sram_stride, cfg.dram_stride);

    if (npu_init() != 0) {
        std::perror("npu_init");
        return 1;
    }
    npu_reset();

    auto * src = static_cast<uint8_t *>(npu_mem_alloc(bytes));
    auto * dst = static_cast<uint8_t *>(npu_mem_alloc(bytes));
    if (!src || !dst) {
        std::fprintf(stderr, "npu_mem_alloc failed\n");
        if (src) npu_mem_free(src);
        if (dst) npu_mem_free(dst);
        npu_destroy();
        return 2;
    }

    for (size_t i = 0; i < bytes; ++i) {
        src[i] = static_cast<uint8_t>((i * 131 + 17) & 0xFF);
    }

    uint64_t total_mvin_ns = 0;
    uint64_t total_mvout_ns = 0;
    for (int it = 0; it < loops; ++it) {
        std::memset(dst, 0, bytes);

        const auto t0 = std::chrono::steady_clock::now();
        npu_dma_mvin(
            src,
            cfg.sram_addr,
            cfg.col_num,
            cfg.row_num,
            cfg.sram_stride,
            cfg.dram_stride,
            1,      // precision
            0,      // input_type: IFM
            false,  // dest: SPM
            false,  // is_bias
            false,  // is_quant
            0,
            0,
            0);
        const auto t1 = std::chrono::steady_clock::now();

        npu_dma_mvout(
            dst,
            cfg.sram_addr,
            cfg.col_num,
            cfg.row_num,
            cfg.sram_stride,
            cfg.dram_stride,
            1,      // precision
            0,      // output_type: int8 from SPM
            false,  // source: SPM
            false,  // is_quant
            0,
            0);
        const auto t2 = std::chrono::steady_clock::now();

        total_mvin_ns += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
        total_mvout_ns += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count());

        size_t mismatches = 0;
        for (size_t i = 0; i < bytes; ++i) {
            if (dst[i] != src[i]) {
                if (mismatches < 8) {
                    std::fprintf(stderr, "iter=%d mismatch idx=%zu src=0x%02x dst=0x%02x\n",
                                 it, i, src[i], dst[i]);
                }
                ++mismatches;
            }
        }
        if (mismatches) {
            std::fprintf(stderr, "iter=%d mismatches=%zu\n", it, mismatches);
            npu_mem_free(src);
            npu_mem_free(dst);
            npu_destroy();
            return 3;
        }

        if ((it + 1) % 10 == 0 || it == loops - 1) {
            std::printf("iter=%d/%d ok\n", it + 1, loops);
        }
    }

    const double avg_mvin_us = static_cast<double>(total_mvin_ns) / loops / 1000.0;
    const double avg_mvout_us = static_cast<double>(total_mvout_ns) / loops / 1000.0;
    std::printf("done: avg_mvin_us=%.3f avg_mvout_us=%.3f\n", avg_mvin_us, avg_mvout_us);

    npu_mem_free(src);
    npu_mem_free(dst);
    npu_destroy();
    return 0;
}

