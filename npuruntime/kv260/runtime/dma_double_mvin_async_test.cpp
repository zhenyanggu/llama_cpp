#include "npu_runtime.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

struct Config {
    uint32_t bytes_per_dma = 16 * 1024; // 16 KiB
    uint32_t sram_addr_dma0 = 0x0000;
    uint32_t sram_addr_dma1 = 0x4000;
    int loops = 100;
    bool use_double_api = false;
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
        if (std::strcmp(argv[i], "--bytes") == 0 && i + 1 < argc) {
            cfg.bytes_per_dma = parse_u32(argv[++i], cfg.bytes_per_dma);
            continue;
        }
        if (std::strcmp(argv[i], "--sram0") == 0 && i + 1 < argc) {
            cfg.sram_addr_dma0 = parse_u32(argv[++i], cfg.sram_addr_dma0);
            continue;
        }
        if (std::strcmp(argv[i], "--sram1") == 0 && i + 1 < argc) {
            cfg.sram_addr_dma1 = parse_u32(argv[++i], cfg.sram_addr_dma1);
            continue;
        }
        if (std::strcmp(argv[i], "--loops") == 0 && i + 1 < argc) {
            const int v = std::atoi(argv[++i]);
            cfg.loops = v > 0 ? v : cfg.loops;
            continue;
        }
        if (std::strcmp(argv[i], "--use-double-api") == 0) {
            cfg.use_double_api = true;
            continue;
        }
    }
    if (cfg.bytes_per_dma == 0) {
        cfg.bytes_per_dma = 1;
    }
    return cfg;
}

bool check_equal(const uint8_t * a, const uint8_t * b, size_t n, const char * tag, int it) {
    size_t mismatches = 0;
    size_t first_idx = 0;
    for (size_t i = 0; i < n; ++i) {
        if (a[i] != b[i]) {
            if (mismatches == 0) {
                first_idx = i;
            }
            ++mismatches;
        }
    }
    if (mismatches == 0) {
        return true;
    }
    std::fprintf(stderr,
                 "iter=%d %s mismatch count=%zu first_idx=%zu src=0x%02x dst=0x%02x\n",
                 it,
                 tag,
                 mismatches,
                 first_idx,
                 a[first_idx],
                 b[first_idx]);
    return false;
}

double mib_per_second(uint64_t bytes, uint64_t ns) {
    if (ns == 0) return 0.0;
    const double seconds = static_cast<double>(ns) / 1.0e9;
    return (static_cast<double>(bytes) / (1024.0 * 1024.0)) / seconds;
}

} // namespace

int main(int argc, char ** argv) {
    const Config cfg = parse_config(argc, argv);
    const uint32_t col_num = cfg.bytes_per_dma - 1;
    const uint32_t row_num = 0;

    std::printf(
        "kv260_dma_double_mvin_async_test: loops=%d bytes_per_dma=%u total_bytes=%u sram0=0x%08x sram1=0x%08x mode=%s\n",
        cfg.loops,
        cfg.bytes_per_dma,
        cfg.bytes_per_dma * 2,
        cfg.sram_addr_dma0,
        cfg.sram_addr_dma1,
        cfg.use_double_api ? "npu_dma_double_mvin" : "mvin_async+wait");

    if (npu_init() != 0) {
        std::perror("npu_init");
        return 1;
    }
    npu_reset();

    auto * src0 = static_cast<uint8_t *>(npu_mem_alloc(cfg.bytes_per_dma));
    auto * src1 = static_cast<uint8_t *>(npu_mem_alloc(cfg.bytes_per_dma));
    auto * dst0 = static_cast<uint8_t *>(npu_mem_alloc(cfg.bytes_per_dma));
    auto * dst1 = static_cast<uint8_t *>(npu_mem_alloc(cfg.bytes_per_dma));
    if (!src0 || !src1 || !dst0 || !dst1) {
        std::fprintf(stderr, "npu_mem_alloc failed\n");
        if (src0) npu_mem_free(src0);
        if (src1) npu_mem_free(src1);
        if (dst0) npu_mem_free(dst0);
        if (dst1) npu_mem_free(dst1);
        npu_destroy();
        return 2;
    }

    for (uint32_t i = 0; i < cfg.bytes_per_dma; ++i) {
        src0[i] = static_cast<uint8_t>((i * 13u + 7u) & 0xffu);
        src1[i] = static_cast<uint8_t>((i * 29u + 19u) & 0xffu);
    }

    MvinConfig mvin0{};
    mvin0.host_ptr = src0;
    mvin0.sram_addr = cfg.sram_addr_dma0;
    mvin0.col_num = col_num;
    mvin0.row_num = row_num;
    mvin0.sram_stride = 0;
    mvin0.dram_stride = 0;
    mvin0.precision = 1;
    mvin0.input_type = 0;
    mvin0.dest = false;
    mvin0.is_bias = false;
    mvin0.is_quant = false;
    mvin0.quant_zero = 0;
    mvin0.quant_scale = 0;
    mvin0.quant_shift = 0;

    MvinConfig mvin1 = mvin0;
    mvin1.host_ptr = src1;
    mvin1.sram_addr = cfg.sram_addr_dma1;

    uint64_t total_double_mvin_ns = 0;
    uint64_t total_mvout_ns = 0;

    for (int it = 0; it < cfg.loops; ++it) {
        std::memset(dst0, 0, cfg.bytes_per_dma);
        std::memset(dst1, 0, cfg.bytes_per_dma);

        const auto t0 = std::chrono::steady_clock::now();
        if (cfg.use_double_api) {
            npu_dma_double_mvin(&mvin0, &mvin1);
        } else {
            npu_dma_mvin_async(0, &mvin0);
            npu_dma_mvin_async(1, &mvin1);
            npu_dma_wait_mvin((1u << 0) | (1u << 1));
        }
        const auto t1 = std::chrono::steady_clock::now();

        npu_dma_mvout(
            dst0,
            cfg.sram_addr_dma0,
            col_num,
            row_num,
            0,
            0,
            1,
            0,
            false,
            false,
            0,
            0);
        npu_dma_mvout(
            dst1,
            cfg.sram_addr_dma1,
            col_num,
            row_num,
            0,
            0,
            1,
            0,
            false,
            false,
            0,
            0);
        const auto t2 = std::chrono::steady_clock::now();

        total_double_mvin_ns += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
        total_mvout_ns += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count());

        if (!check_equal(src0, dst0, cfg.bytes_per_dma, "dma0", it) ||
            !check_equal(src1, dst1, cfg.bytes_per_dma, "dma1", it)) {
            npu_mem_free(src0);
            npu_mem_free(src1);
            npu_mem_free(dst0);
            npu_mem_free(dst1);
            npu_destroy();
            return 3;
        }

        if ((it + 1) % 10 == 0 || it == cfg.loops - 1) {
            std::printf("iter=%d/%d ok\n", it + 1, cfg.loops);
        }
    }

    const uint64_t total_mvin_bytes = static_cast<uint64_t>(cfg.bytes_per_dma) * 2ULL * static_cast<uint64_t>(cfg.loops);
    const uint64_t total_mvout_bytes = total_mvin_bytes;
    std::printf("double_mvin_total_ns=%llu avg_us=%.3f bw_mib_s=%.2f\n",
                static_cast<unsigned long long>(total_double_mvin_ns),
                static_cast<double>(total_double_mvin_ns) / static_cast<double>(cfg.loops) / 1000.0,
                mib_per_second(total_mvin_bytes, total_double_mvin_ns));
    std::printf("mvout_total_ns=%llu avg_us=%.3f bw_mib_s=%.2f\n",
                static_cast<unsigned long long>(total_mvout_ns),
                static_cast<double>(total_mvout_ns) / static_cast<double>(cfg.loops) / 1000.0,
                mib_per_second(total_mvout_bytes, total_mvout_ns));
    std::puts("kv260_dma_double_mvin_async_test=ok");

    npu_mem_free(src0);
    npu_mem_free(src1);
    npu_mem_free(dst0);
    npu_mem_free(dst1);
    npu_destroy();
    return 0;
}

