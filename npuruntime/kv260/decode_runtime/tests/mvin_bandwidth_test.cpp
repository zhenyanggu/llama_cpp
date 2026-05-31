#include "npu_runtime.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr uint32_t kSpmBytes = 512 * 1024;
constexpr uint32_t kMvinAlignBytes = 256;
constexpr uint32_t kDefaultSpmBase = 0x0000;
constexpr double kDefaultPeakGBps = 12.8; // 4 * 128-bit AXI at 200 MHz.

struct NpuBuffer {
    void* ptr = nullptr;
    size_t bytes = 0;

    explicit NpuBuffer(size_t size) : ptr(npu_mem_alloc(size)), bytes(size) {}
    ~NpuBuffer() {
        if (ptr) npu_mem_free(ptr);
    }

    NpuBuffer(const NpuBuffer&) = delete;
    NpuBuffer& operator=(const NpuBuffer&) = delete;

    uint8_t* data() { return static_cast<uint8_t*>(ptr); }
};

uint64_t elapsed_ns(Clock::time_point begin, Clock::time_point end) {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
}

double ns_to_ms(uint64_t ns) {
    return static_cast<double>(ns) / 1000000.0;
}

double bytes_per_ns_to_gbps(uint64_t bytes, uint64_t ns) {
    if (ns == 0) {
        return 0.0;
    }
    return static_cast<double>(bytes) / static_cast<double>(ns);
}

int get_env_int(const char* name, int fallback) {
    const char* value = std::getenv(name);
    if (!value || value[0] == '\0') {
        return fallback;
    }
    const int parsed = std::atoi(value);
    return parsed > 0 ? parsed : fallback;
}

double get_env_double(const char* name, double fallback) {
    const char* value = std::getenv(name);
    if (!value || value[0] == '\0') {
        return fallback;
    }
    const double parsed = std::atof(value);
    return parsed > 0.0 ? parsed : fallback;
}

uint32_t align_down(uint32_t value, uint32_t alignment) {
    return value & ~(alignment - 1u);
}

void fill_pattern(uint8_t* data, size_t bytes) {
    for (size_t i = 0; i < bytes; ++i) {
        data[i] = static_cast<uint8_t>((i * 131u + 17u) & 0xffu);
    }
}

void mvin_sync(void* ptr, uint32_t spm_addr, uint32_t bytes) {
    npu_dma_mvin(ptr,
                 spm_addr,
                 bytes - 1u,
                 0,
                 0,
                 0,
                 1,
                 1,
                 false,
                 false,
                 false,
                 0,
                 0,
                 0);
}

void mvin_async_wait(void* ptr, uint32_t spm_addr, uint32_t bytes) {
    MvinConfig cfg = {};
    cfg.host_ptr = ptr;
    cfg.sram_addr = spm_addr;
    cfg.col_num = bytes - 1u;
    cfg.row_num = 0;
    cfg.sram_stride = 0;
    cfg.dram_stride = 0;
    cfg.precision = 1;
    cfg.input_type = 1;
    cfg.dest = false;
    cfg.is_bias = false;
    cfg.is_quant = false;
    cfg.quant_zero = 0;
    cfg.quant_scale = 0;
    cfg.quant_shift = 0;

    npu_dma_mvin_async(0, &cfg);
    npu_dma_wait_mvin(1u << 0);
}

struct CaseResult {
    uint32_t bytes = 0;
    int iterations = 0;
    uint64_t total_ns = 0;
    double gbps = 0.0;
    double util = 0.0;
};

CaseResult run_case(const char* mode, void* ptr, uint32_t bytes, int iterations,
                    int warmup, double peak_gbps) {
    for (int i = 0; i < warmup; ++i) {
        if (std::strcmp(mode, "async") == 0) {
            mvin_async_wait(ptr, kDefaultSpmBase, bytes);
        } else {
            mvin_sync(ptr, kDefaultSpmBase, bytes);
        }
    }

    const auto begin = Clock::now();
    for (int i = 0; i < iterations; ++i) {
        if (std::strcmp(mode, "async") == 0) {
            mvin_async_wait(ptr, kDefaultSpmBase, bytes);
        } else {
            mvin_sync(ptr, kDefaultSpmBase, bytes);
        }
    }
    const uint64_t total_ns = elapsed_ns(begin, Clock::now());
    const uint64_t total_bytes = static_cast<uint64_t>(bytes) *
                                 static_cast<uint64_t>(iterations);
    const double gbps = bytes_per_ns_to_gbps(total_bytes, total_ns);

    CaseResult result = {};
    result.bytes = bytes;
    result.iterations = iterations;
    result.total_ns = total_ns;
    result.gbps = gbps;
    result.util = peak_gbps > 0.0 ? gbps * 100.0 / peak_gbps : 0.0;
    return result;
}

std::vector<uint32_t> default_sizes(uint32_t max_bytes) {
    const uint32_t raw_sizes[] = {
        4 * 1024,
        16 * 1024,
        64 * 1024,
        128 * 1024,
        256 * 1024,
        384 * 1024,
        512 * 1024,
    };

    std::vector<uint32_t> sizes;
    for (uint32_t size : raw_sizes) {
        const uint32_t aligned = align_down(std::min(size, max_bytes), kMvinAlignBytes);
        if (aligned == 0 || aligned > max_bytes) {
            continue;
        }
        if (sizes.empty() || sizes.back() != aligned) {
            sizes.push_back(aligned);
        }
    }
    return sizes;
}

} // namespace

int main() {
    std::puts("kv260_mvin_bandwidth_test: continuous MVIN bandwidth profiling");

    if (npu_init() != 0) {
        std::perror("npu_init");
        return 1;
    }

    const double peak_gbps = get_env_double("KV260_DDR_PEAK_GBPS", kDefaultPeakGBps);
    const int base_iterations = get_env_int("MVIN_BENCH_ITERS", 256);
    const int warmup = get_env_int("MVIN_BENCH_WARMUP", 8);
    const uint32_t max_bytes =
        align_down(static_cast<uint32_t>(std::min<int>(get_env_int("MVIN_BENCH_MAX_KB", 512), 512)) *
                       1024u,
                   kMvinAlignBytes);

    if (max_bytes == 0 || max_bytes > kSpmBytes) {
        std::fprintf(stderr, "invalid MVIN_BENCH_MAX_KB; max must be 1..512 KiB\n");
        npu_destroy();
        return 2;
    }

    NpuBuffer src(max_bytes);
    if (!src.ptr) {
        std::fprintf(stderr, "npu_mem_alloc failed for %u bytes\n", max_bytes);
        npu_destroy();
        return 2;
    }
    fill_pattern(src.data(), max_bytes);

    npu_reset();
    std::printf("peak_gbps=%.3f max_transfer=%u bytes warmup=%d base_iters=%d\n",
                peak_gbps, max_bytes, warmup, base_iterations);
    std::puts("mode,bytes,iters,total_ms,avg_us,gbps,util_percent");

    const std::vector<uint32_t> sizes = default_sizes(max_bytes);
    for (const char* mode : {"sync", "async"}) {
        for (uint32_t bytes : sizes) {
            const int iterations = std::max(16, base_iterations * 65536 / static_cast<int>(bytes));
            const CaseResult result = run_case(mode, src.ptr, bytes, iterations, warmup, peak_gbps);
            std::printf("%s,%u,%d,%.3f,%.3f,%.3f,%.2f\n",
                        mode,
                        result.bytes,
                        result.iterations,
                        ns_to_ms(result.total_ns),
                        static_cast<double>(result.total_ns) / result.iterations / 1000.0,
                        result.gbps,
                        result.util);
        }
    }

    npu_destroy();
    std::puts("kv260_mvin_bandwidth_test=ok");
    return 0;
}
