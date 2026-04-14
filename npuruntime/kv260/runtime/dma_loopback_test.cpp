#include "npu_runtime.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

constexpr uint32_t kSpmAddr = 0x0000;
constexpr uint32_t kBytes = 64 * 1024;
constexpr uint32_t kColNum = kBytes - 1;

double mib_per_second(uint32_t bytes, std::chrono::nanoseconds elapsed)
{
    const double seconds = static_cast<double>(elapsed.count()) / 1.0e9;
    if (seconds <= 0.0) return 0.0;
    return (static_cast<double>(bytes) / (1024.0 * 1024.0)) / seconds;
}

} // namespace

int main()
{
    std::puts("kv260_dma_loopback_test: using KV260 runtime C API");
    std::printf("runtime_call=npu_init device=/dev/npu_kv260\n");
    if (npu_init() != 0) {
        std::perror("npu_init");
        return 1;
    }

    std::printf("runtime_call=npu_reset\n");
    npu_reset();

    auto *src = static_cast<uint8_t *>(npu_mem_alloc(kBytes));
    auto *dst = static_cast<uint8_t *>(npu_mem_alloc(kBytes));
    if (!src || !dst) {
        std::fprintf(stderr, "npu_mem_alloc failed\n");
        if (src) npu_mem_free(src);
        if (dst) npu_mem_free(dst);
        npu_destroy();
        return 2;
    }
    std::printf("runtime_call=npu_mem_alloc bytes=%u src=%p dst=%p\n",
                kBytes,
                static_cast<void *>(src),
                static_cast<void *>(dst));

    for (uint32_t i = 0; i < kBytes; ++i) {
        src[i] = static_cast<uint8_t>((i * 13u + 7u) & 0xffu);
    }
    std::memset(dst, 0, kBytes);

    std::printf(
        "runtime_call=npu_dma_mvin bytes=%u sram_addr=0x%08x col_num=%u row_num=0 sram_stride=0 dram_stride=0 precision=1 input_type=0 dest=SPM\n",
        kBytes,
        kSpmAddr,
        kColNum);
    const auto mvin_begin = std::chrono::steady_clock::now();
    npu_dma_mvin(
        src,
        kSpmAddr,
        kColNum,
        0,
        0,
        0,
        1,
        0,
        false,
        false,
        false,
        0,
        0,
        0);
    const auto mvin_end = std::chrono::steady_clock::now();

    std::printf(
        "runtime_call=npu_dma_mvout bytes=%u sram_addr=0x%08x col_num=%u row_num=0 sram_stride=0 dram_stride=0 precision=1 output_type=0 source=SPM\n",
        kBytes,
        kSpmAddr,
        kColNum);
    const auto mvout_begin = std::chrono::steady_clock::now();
    npu_dma_mvout(
        dst,
        kSpmAddr,
        kColNum,
        0,
        0,
        0,
        1,
        0,
        false,
        false,
        0,
        0);
    const auto mvout_end = std::chrono::steady_clock::now();

    const auto mvin_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(mvin_end - mvin_begin);
    const auto mvout_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(mvout_end - mvout_begin);
    const auto total_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(mvout_end - mvin_begin);

    std::printf("bandwidth_mvin bytes=%u time_ns=%lld mib_per_s=%.2f\n",
                kBytes,
                static_cast<long long>(mvin_ns.count()),
                mib_per_second(kBytes, mvin_ns));
    std::printf("bandwidth_mvout bytes=%u time_ns=%lld mib_per_s=%.2f\n",
                kBytes,
                static_cast<long long>(mvout_ns.count()),
                mib_per_second(kBytes, mvout_ns));
    std::printf("bandwidth_loopback bytes=%u time_ns=%lld mib_per_s=%.2f\n",
                kBytes * 2,
                static_cast<long long>(total_ns.count()),
                mib_per_second(kBytes * 2, total_ns));

    uint32_t mismatches = 0;
    uint32_t first_mismatch = 0;
    for (uint32_t i = 0; i < kBytes; ++i) {
        if (dst[i] != src[i]) {
            if (mismatches == 0) first_mismatch = i;
            ++mismatches;
        }
    }

    if (mismatches) {
        std::fprintf(stderr,
                     "kv260_dma_loopback_test=fail mismatches=%u first=%u src=0x%02x dst=0x%02x\n",
                     mismatches,
                     first_mismatch,
                     src[first_mismatch],
                     dst[first_mismatch]);
    } else {
        std::puts("kv260_dma_loopback_test=ok");
    }

    npu_mem_free(src);
    npu_mem_free(dst);
    npu_destroy();
    return mismatches ? 3 : 0;
}
