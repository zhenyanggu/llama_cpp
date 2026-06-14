#include "npu_kv260_uapi.h"

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#define REG_MER 0x108U
#define DEFAULT_SIZE (1U << 20)

static uint64_t parse_size(const char *text, uint64_t fallback)
{
    char *end = NULL;
    unsigned long long value;
    uint64_t multiplier = 1;

    if (!text || text[0] == '\0')
        return fallback;

    errno = 0;
    value = strtoull(text, &end, 0);
    if (errno || end == text)
        return fallback;

    if (*end == 'K' || *end == 'k') {
        multiplier = 1024ULL;
    } else if (*end == 'M' || *end == 'm') {
        multiplier = 1024ULL * 1024ULL;
    } else if (*end == 'G' || *end == 'g') {
        multiplier = 1024ULL * 1024ULL * 1024ULL;
    } else if (*end != '\0') {
        return fallback;
    }

    return (uint64_t)value * multiplier;
}

static void fail_errno(const char *what)
{
    perror(what);
    exit(1);
}

int main(int argc, char **argv)
{
    uint64_t requested_size = DEFAULT_SIZE;
    struct npu_kv260_info info;
    struct npu_kv260_buffer_request req;
    struct npu_kv260_reg_access reg;
    volatile uint32_t *regs;
    uint8_t *data;
    uint64_t off;
    uint32_t mer_via_mmap;
    const char *env;
    int fd;

    env = getenv("NPU_CMA_SIZE");
    if (env)
        requested_size = parse_size(env, requested_size);
    if (argc > 1)
        requested_size = parse_size(argv[1], requested_size);

    fd = open(NPU_KV260_DEV_PATH, O_RDWR);
    if (fd < 0)
        fail_errno("open " NPU_KV260_DEV_PATH);

    if (ioctl(fd, NPU_KV260_IOC_GET_INFO, &info) < 0)
        fail_errno("ioctl GET_INFO");

    printf("regs_phys=0x%llx regs_size=0x%x dma_bits=%u has_irq=%u\n",
           (unsigned long long)info.regs_phys,
           info.regs_size,
           info.dma_addr_bits,
           info.has_irq);

    req.size = requested_size;
    req.dma_addr = 0;
    if (ioctl(fd, NPU_KV260_IOC_ALLOC_BUFFER, &req) < 0)
        fail_errno("ioctl ALLOC_BUFFER");

    if (req.dma_addr == 0) {
        fprintf(stderr, "driver returned zero DMA address\n");
        return 1;
    }

    printf("cma_dma=0x%llx cma_size=0x%llx\n",
           (unsigned long long)req.dma_addr,
           (unsigned long long)req.size);

    data = mmap(NULL, req.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd,
                NPU_KV260_MMAP_BUFFER_OFFSET);
    if (data == MAP_FAILED)
        fail_errno("mmap CMA buffer");

    for (off = 0; off < req.size; off += 4096)
        data[off] = (uint8_t)((off / 4096) & 0xff);

    for (off = 0; off < req.size; off += 4096) {
        uint8_t expected = (uint8_t)((off / 4096) & 0xff);
        if (data[off] != expected) {
            fprintf(stderr,
                    "CMA mmap verify failed at 0x%llx: got 0x%02x expected 0x%02x\n",
                    (unsigned long long)off, data[off], expected);
            return 1;
        }
    }
    puts("cma_mmap_rw=ok");

    regs = mmap(NULL, NPU_KV260_REG_MMAP_SIZE, PROT_READ | PROT_WRITE,
                MAP_SHARED, fd, NPU_KV260_MMAP_REGS_OFFSET);
    if (regs == MAP_FAILED)
        fail_errno("mmap registers");

    mer_via_mmap = regs[REG_MER / sizeof(uint32_t)];
    reg.offset = REG_MER;
    reg.value = 0;
    if (ioctl(fd, NPU_KV260_IOC_REG_READ, &reg) < 0)
        fail_errno("ioctl REG_READ");
    if (ioctl(fd, NPU_KV260_IOC_REG_WRITE, &reg) < 0)
        fail_errno("ioctl REG_WRITE");

    printf("reg_mer_mmap=0x%08x reg_mer_ioctl=0x%08x\n",
           mer_via_mmap, reg.value);

    munmap((void *)regs, NPU_KV260_REG_MMAP_SIZE);
    munmap(data, req.size);

    if (ioctl(fd, NPU_KV260_IOC_FREE_BUFFER) < 0)
        fail_errno("ioctl FREE_BUFFER");

    close(fd);
    puts("kv260_npu_smoke_test=ok");
    return 0;
}
