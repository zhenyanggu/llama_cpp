#include "npu_regs_compat.h"
#include "npu_kv260_uapi.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr uint32_t kMatvecIrqMask = BIT_START_MATVEC;

uint64_t elapsed_ns(Clock::time_point begin, Clock::time_point end) {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
}

int get_env_int(const char* name, int fallback) {
    const char* value = std::getenv(name);
    if (!value || value[0] == '\0') {
        return fallback;
    }
    char* end = nullptr;
    errno = 0;
    long parsed = std::strtol(value, &end, 0);
    if (errno || end == value || parsed <= 0) {
        return fallback;
    }
    return static_cast<int>(parsed);
}

void fail_errno(const char* what) {
    throw std::runtime_error(std::string(what) + ": " + std::strerror(errno));
}

volatile uint32_t* reg32(volatile uint32_t* regs, uint32_t offset) {
    return regs + (offset / sizeof(uint32_t));
}

uint32_t read_reg(volatile uint32_t* regs, uint32_t offset) {
    return *reg32(regs, offset);
}

void write_reg(volatile uint32_t* regs, uint32_t offset, uint32_t value) {
    *reg32(regs, offset) = value;
}

void write_reg64(volatile uint32_t* regs, uint32_t offset, uint64_t value) {
    volatile uint64_t* ptr =
        reinterpret_cast<volatile uint64_t*>(reinterpret_cast<volatile char*>(const_cast<uint32_t*>(regs)) + offset);
    *ptr = value;
}

void set_irq_mode(int fd, uint32_t mode) {
    if (ioctl(fd, NPU_KV260_IOC_SET_IRQ_MODE, mode) < 0) {
        fail_errno("ioctl SET_IRQ_MODE");
    }
}

void clear_irqs(volatile uint32_t* regs) {
    write_reg(regs, RegOffset::IAR, NPU_REGS__IAR__ACK_bm);
}

void program_zero_matvec(volatile uint32_t* regs) {
    write_reg64(regs, RegOffset::MATVEC_CTRL_0, 0);
    write_reg64(regs, RegOffset::MATVEC_CTRL_1, 0);
}

uint64_t bench_empty_loop(int iterations) {
    volatile uint32_t sink = 0;
    const auto begin = Clock::now();
    for (int i = 0; i < iterations; ++i) {
        sink += static_cast<uint32_t>(i);
        asm volatile("" ::: "memory");
    }
    const auto end = Clock::now();
    if (sink == 0xFFFFFFFFu) {
        std::printf("sink=%u\n", sink);
    }
    return elapsed_ns(begin, end);
}

uint64_t bench_mmio_group(volatile uint32_t* regs, int iterations, int writes_per_iter) {
    const auto begin = Clock::now();
    for (int i = 0; i < iterations; ++i) {
        const uint64_t value = 0x5a00000000000000ull ^ static_cast<uint64_t>(i);
        write_reg64(regs, RegOffset::MATVEC_CTRL_0, value);
        if (writes_per_iter >= 2) {
            write_reg64(regs, RegOffset::MATVEC_CTRL_1, value + 1u);
        }
        if (writes_per_iter >= 3) {
            write_reg64(regs, RegOffset::SFU_INPUT, value + 2u);
        }
        asm volatile("" ::: "memory");
    }
    const auto end = Clock::now();
    return elapsed_ns(begin, end);
}

void print_mmio_result(const char* name, uint64_t total_ns, uint64_t empty_ns,
                       int iterations, int writes_per_iter) {
    const double raw_iter = static_cast<double>(total_ns) / iterations;
    const double corrected_iter =
        static_cast<double>(total_ns > empty_ns ? total_ns - empty_ns : total_ns) / iterations;
    std::printf("%s iterations=%d writes_per_iter=%d total_ns=%llu raw_iter_ns=%.2f "
                "corrected_iter_ns=%.2f corrected_per_write_ns=%.2f\n",
                name,
                iterations,
                writes_per_iter,
                static_cast<unsigned long long>(total_ns),
                raw_iter,
                corrected_iter,
                corrected_iter / writes_per_iter);
}

struct SampleStats {
    uint64_t min_ns = 0;
    uint64_t p50_ns = 0;
    uint64_t p95_ns = 0;
    uint64_t max_ns = 0;
    double avg_ns = 0.0;
};

SampleStats summarize(std::vector<uint64_t> samples) {
    if (samples.empty()) {
        return {};
    }
    std::sort(samples.begin(), samples.end());
    uint64_t sum = 0;
    for (uint64_t value : samples) {
        sum += value;
    }
    SampleStats stats;
    stats.min_ns = samples.front();
    stats.p50_ns = samples[samples.size() / 2];
    stats.p95_ns = samples[(samples.size() * 95) / 100];
    stats.max_ns = samples.back();
    stats.avg_ns = static_cast<double>(sum) / samples.size();
    return stats;
}

void print_wait_stats(const char* name, const SampleStats& stats, int iterations) {
    std::printf("%s iterations=%d min_ns=%llu avg_ns=%.2f p50_ns=%llu p95_ns=%llu max_ns=%llu\n",
                name,
                iterations,
                static_cast<unsigned long long>(stats.min_ns),
                stats.avg_ns,
                static_cast<unsigned long long>(stats.p50_ns),
                static_cast<unsigned long long>(stats.p95_ns),
                static_cast<unsigned long long>(stats.max_ns));
}

uint64_t run_zero_matvec_poll_once(volatile uint32_t* regs, int spin_limit) {
    clear_irqs(regs);
    program_zero_matvec(regs);

    const auto begin = Clock::now();
    write_reg(regs, RegOffset::START, BIT_START_MATVEC);
    for (int i = 0; i < spin_limit; ++i) {
        const uint32_t isr = read_reg(regs, RegOffset::ISR);
        if (isr & kMatvecIrqMask) {
            clear_irqs(regs);
            return elapsed_ns(begin, Clock::now());
        }
    }
    throw std::runtime_error("poll wait timeout for zero MATVEC");
}

uint64_t run_zero_matvec_kernel_irq_once(int fd, volatile uint32_t* regs) {
    uint32_t status = 0;
    clear_irqs(regs);
    program_zero_matvec(regs);

    const auto begin = Clock::now();
    write_reg(regs, RegOffset::START, BIT_START_MATVEC);
    if (ioctl(fd, NPU_KV260_IOC_WAIT_IRQ, &status) < 0) {
        fail_errno("ioctl WAIT_IRQ");
    }
    const uint64_t ns = elapsed_ns(begin, Clock::now());
    if ((status & kMatvecIrqMask) == 0) {
        std::fprintf(stderr, "warning: kernel IRQ status 0x%08x lacks MATVEC bit 0x%08x\n",
                     status, kMatvecIrqMask);
    }
    return ns;
}

SampleStats bench_poll_wait(volatile uint32_t* regs, int iterations, int spin_limit) {
    std::vector<uint64_t> samples;
    samples.reserve(iterations);
    for (int i = 0; i < iterations; ++i) {
        samples.push_back(run_zero_matvec_poll_once(regs, spin_limit));
    }
    return summarize(std::move(samples));
}

SampleStats bench_kernel_irq_wait(int fd, volatile uint32_t* regs, int iterations) {
    std::vector<uint64_t> samples;
    samples.reserve(iterations);
    for (int i = 0; i < iterations; ++i) {
        samples.push_back(run_zero_matvec_kernel_irq_once(fd, regs));
    }
    return summarize(std::move(samples));
}

}  // namespace

int main() {
    const int mmio_iters = get_env_int("NPU_MMIO_BENCH_ITERS", 100000);
    const int irq_iters = get_env_int("NPU_IRQ_BENCH_ITERS", 2000);
    const int poll_spin_limit = get_env_int("NPU_IRQ_POLL_SPIN_LIMIT", 1000000);
    const bool run_legacy_zero_matvec_irq =
        get_env_int("NPU_IRQ_BENCH_ZERO_MATVEC", 0) != 0;

    int fd = open(NPU_KV260_DEV_PATH, O_RDWR);
    if (fd < 0) {
        std::perror("open " NPU_KV260_DEV_PATH);
        return 1;
    }

    try {
        npu_kv260_info info = {};
        if (ioctl(fd, NPU_KV260_IOC_GET_INFO, &info) < 0) {
            fail_errno("ioctl GET_INFO");
        }

        volatile uint32_t* regs = static_cast<volatile uint32_t*>(
            mmap(nullptr, NPU_KV260_REG_MMAP_SIZE, PROT_READ | PROT_WRITE,
                 MAP_SHARED, fd, NPU_KV260_MMAP_REGS_OFFSET));
        if (regs == MAP_FAILED) {
            fail_errno("mmap registers");
        }

        std::printf("kv260_mmio_irq_bench_test: regs_phys=0x%llx regs_size=0x%x has_irq=%u\n",
                    static_cast<unsigned long long>(info.regs_phys),
                    info.regs_size,
                    info.has_irq);

        set_irq_mode(fd, NPU_KV260_IRQ_MODE_USERSPACE);
        clear_irqs(regs);

        const uint64_t empty_ns = bench_empty_loop(mmio_iters);
        std::printf("empty_loop iterations=%d total_ns=%llu raw_iter_ns=%.2f\n",
                    mmio_iters,
                    static_cast<unsigned long long>(empty_ns),
                    static_cast<double>(empty_ns) / mmio_iters);
        for (int writes = 1; writes <= 3; ++writes) {
            const uint64_t total_ns = bench_mmio_group(regs, mmio_iters, writes);
            print_mmio_result("mmio_mmap_write64", total_ns, empty_ns, mmio_iters, writes);
        }

        if (run_legacy_zero_matvec_irq) {
            set_irq_mode(fd, NPU_KV260_IRQ_MODE_USERSPACE);
            const SampleStats poll_stats = bench_poll_wait(regs, irq_iters, poll_spin_limit);
            print_wait_stats("zero_matvec_userspace_poll", poll_stats, irq_iters);

            if (info.has_irq) {
                set_irq_mode(fd, NPU_KV260_IRQ_MODE_KERNEL);
                const SampleStats irq_stats = bench_kernel_irq_wait(fd, regs, irq_iters);
                print_wait_stats("zero_matvec_kernel_irq", irq_stats, irq_iters);
                set_irq_mode(fd, NPU_KV260_IRQ_MODE_USERSPACE);
            } else {
                std::puts("zero_matvec_kernel_irq=skipped has_irq=0");
            }
        } else {
            std::puts("zero_matvec_irq_bench=skipped enable_with_NPU_IRQ_BENCH_ZERO_MATVEC=1");
        }

        munmap(const_cast<uint32_t*>(regs), NPU_KV260_REG_MMAP_SIZE);
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "kv260_mmio_irq_bench_test error: %s\n", ex.what());
        close(fd);
        return 2;
    }

    close(fd);
    std::puts("kv260_mmio_irq_bench_test=ok");
    return 0;
}
