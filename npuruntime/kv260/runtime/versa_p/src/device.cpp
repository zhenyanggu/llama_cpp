#include "versa_p_internal.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <stdio.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace {

uint32_t parse_size_env(const char *env, uint32_t fallback)
{
    if (!env || env[0] == '\0') {
        return fallback;
    }

    char *end = nullptr;
    unsigned long long parsed = strtoull(env, &end, 0);
    if (end == env || parsed == 0) {
        return fallback;
    }

    while (*end == ' ' || *end == '\t') {
        ++end;
    }

    unsigned long long multiplier = 1;
    if (*end != '\0') {
        if (*end == 'k' || *end == 'K') {
            multiplier = 1024ull;
            ++end;
        } else if (*end == 'm' || *end == 'M') {
            multiplier = 1024ull * 1024ull;
            ++end;
        } else if (*end == 'g' || *end == 'G') {
            multiplier = 1024ull * 1024ull * 1024ull;
            ++end;
        } else {
            return fallback;
        }
        if (*end == 'i' || *end == 'I') {
            ++end;
        }
        if (*end == 'b' || *end == 'B') {
            ++end;
        }
        while (*end == ' ' || *end == '\t') {
            ++end;
        }
        if (*end != '\0') {
            return fallback;
        }
    }

    if (parsed > std::numeric_limits<uint32_t>::max() / multiplier) {
        return fallback;
    }
    return (uint32_t)(parsed * multiplier);
}

uint32_t resolve_cma_size(const versa_p_options *options)
{
    if (options && options->cma_size != 0) {
        return options->cma_size;
    }
    return parse_size_env(getenv("VERSA_P_CMA_SIZE"), VERSA_P_DEFAULT_CMA_BYTES);
}

uint32_t resolve_heap_offset(void)
{
    return parse_size_env(getenv("VERSA_P_CMA_HEAP_OFFSET"), 0);
}

uint32_t resolve_heap_size(uint32_t cma_size, uint32_t heap_offset)
{
    if (heap_offset >= cma_size) {
        return 0;
    }
    return parse_size_env(getenv("VERSA_P_CMA_HEAP_SIZE"), cma_size - heap_offset);
}

uint32_t resolve_cma_alloc_flags(void)
{
    const char *env = getenv("VERSA_P_CMA_CACHEABLE");
    if (!env || env[0] == '\0') {
        env = getenv("NPU_CMA_CACHEABLE");
    }
    if (!env || env[0] == '\0' || env[0] == '0') {
        return 0;
    }
    uint32_t flags = NPU_KV260_BUFFER_FLAG_CACHEABLE;
    if (strcmp(env, "required") == 0 || strcmp(env, "require") == 0) {
        flags |= NPU_KV260_BUFFER_FLAG_CACHEABLE_REQUIRED;
    }
    return flags;
}

const char *resolve_dev_path(const versa_p_options *options)
{
    if (options && options->dev_path && options->dev_path[0] != '\0') {
        return options->dev_path;
    }
    return NPU_KV260_DEV_PATH;
}

void reset_software_state(versa_p_device *dev)
{
    for (bool &inflight : dev->api_inflight) {
        inflight = false;
    }
    dev->active_a_bank = 0;
    dev->a_bank[0] = VersaPABankState{};
    dev->a_bank[1] = VersaPABankState{};
    dev->pending_mvin_a_bank = 0;
    dev->mvin_a_inflight_bank = 0;
    dev->pending_mvin_a_m = 0;
    dev->pending_mvin_a_k = 0;
    dev->active_w_bank = 1;
    dev->w_bank[0] = VersaPBankState{};
    dev->w_bank[1] = VersaPBankState{};
    dev->pending_mvin_w_bank = 0;
    dev->pending_mvin_w_k = 0;
    dev->pending_mvin_w_n = 0;
    dev->gemm_inflight_a_bank = 0;
    dev->gemm_inflight_o_bank = 0;
    dev->mvout_inflight_o_bank = 0;
    dev->pending_mvout_dma_addr = 0;
    dev->pending_mvout_bytes = 0;
}

} // namespace

const char *versa_p_status_string(int status)
{
    switch (status) {
    case VERSA_P_OK: return "ok";
    case VERSA_P_ERR_INVAL: return "invalid argument";
    case VERSA_P_ERR_IO: return "io error";
    case VERSA_P_ERR_NOMEM: return "out of memory";
    case VERSA_P_ERR_TIMEOUT: return "timeout";
    case VERSA_P_ERR_BUSY: return "busy";
    case VERSA_P_ERR_ALIGNMENT: return "alignment error";
    case VERSA_P_ERR_ILLEGAL_SHAPE: return "illegal shape";
    case VERSA_P_ERR_RESOURCE_CONFLICT: return "resource conflict";
    case VERSA_P_ERR_BANK_CONFLICT: return "bank conflict";
    case VERSA_P_ERR_BANK_NOT_VALID: return "bank not valid";
    case VERSA_P_ERR_SHAPE_MISMATCH: return "shape mismatch";
    case VERSA_P_ERR_UNSUPPORTED_MODE: return "unsupported mode";
    case VERSA_P_ERR_ILLEGAL_FLAGS: return "illegal flags";
    case VERSA_P_ERR_HARDWARE: return "hardware error";
    default: return "unknown";
    }
}

int versa_p_init(versa_p_device **out_dev, const versa_p_options *options)
{
    if (!out_dev) {
        return VERSA_P_ERR_INVAL;
    }
    *out_dev = nullptr;

    versa_p_device *dev = new versa_p_device();
    dev->fd = open(resolve_dev_path(options), O_RDWR);
    if (dev->fd < 0) {
        delete dev;
        return VERSA_P_ERR_IO;
    }

    npu_kv260_buffer_request req = {};
    npu_kv260_buffer_request_ex req_ex = {};
    const uint32_t alloc_flags = resolve_cma_alloc_flags();
    bool allocated = false;
    if (alloc_flags != 0) {
        req_ex.size = resolve_cma_size(options);
        req_ex.flags = alloc_flags;
        if (ioctl(dev->fd, NPU_KV260_IOC_ALLOC_BUFFER_EX, &req_ex) == 0) {
            allocated = true;
            req.size = req_ex.size;
            req.dma_addr = req_ex.dma_addr;
        } else if ((alloc_flags & NPU_KV260_BUFFER_FLAG_CACHEABLE_REQUIRED) != 0) {
            close(dev->fd);
            delete dev;
            return VERSA_P_ERR_IO;
        }
    }
    if (!allocated) {
        req.size = resolve_cma_size(options);
        if (ioctl(dev->fd, NPU_KV260_IOC_ALLOC_BUFFER, &req) == 0) {
            allocated = true;
        }
    }
    if (!allocated) {
        close(dev->fd);
        delete dev;
        return VERSA_P_ERR_IO;
    }
    if (req.dma_addr == 0 || req.size == 0 ||
        req.dma_addr > std::numeric_limits<uint32_t>::max() ||
        req.size > std::numeric_limits<uint32_t>::max()) {
        ioctl(dev->fd, NPU_KV260_IOC_FREE_BUFFER);
        close(dev->fd);
        delete dev;
        return VERSA_P_ERR_IO;
    }

    dev->cma_dma = (uint32_t)req.dma_addr;
    dev->cma_size = (uint32_t)req.size;
    dev->regs = mmap(nullptr, NPU_KV260_REG_MMAP_SIZE, PROT_READ | PROT_WRITE,
                     MAP_SHARED, dev->fd, (off_t)NPU_KV260_MMAP_REGS_OFFSET);
    if (dev->regs == MAP_FAILED) {
        dev->regs = nullptr;
        ioctl(dev->fd, NPU_KV260_IOC_FREE_BUFFER);
        close(dev->fd);
        delete dev;
        return VERSA_P_ERR_IO;
    }

    dev->cma = mmap(nullptr, dev->cma_size, PROT_READ | PROT_WRITE, MAP_SHARED,
                    dev->fd, (off_t)NPU_KV260_MMAP_BUFFER_OFFSET);
    if (dev->cma == MAP_FAILED) {
        dev->cma = nullptr;
        munmap(dev->regs, NPU_KV260_REG_MMAP_SIZE);
        ioctl(dev->fd, NPU_KV260_IOC_FREE_BUFFER);
        close(dev->fd);
        delete dev;
        return VERSA_P_ERR_IO;
    }

    (void)ioctl(dev->fd, NPU_KV260_IOC_GET_INFO, &dev->info);
    const uint32_t heap_offset = resolve_heap_offset();
    const uint32_t heap_size = resolve_heap_size(dev->cma_size, heap_offset);
    if (heap_size == 0 || heap_size > dev->cma_size - heap_offset) {
        versa_p_destroy(dev);
        return VERSA_P_ERR_INVAL;
    }
    dev->blocks.push_back(VersaPBlock{heap_offset, heap_size, true});
    reset_software_state(dev);

    int rc = versa_p_reset(dev);
    if (rc != VERSA_P_OK) {
        versa_p_destroy(dev);
        return rc;
    }

    *out_dev = dev;
    return VERSA_P_OK;
}

void versa_p_destroy(versa_p_device *dev)
{
    if (!dev) {
        return;
    }
    if (dev->fd >= 0 && dev->regs) {
        (void)versa_p_reset(dev);
    }
    if (dev->regs) {
        munmap(dev->regs, NPU_KV260_REG_MMAP_SIZE);
    }
    if (dev->cma) {
        munmap(dev->cma, dev->cma_size);
    }
    if (dev->fd >= 0) {
        ioctl(dev->fd, NPU_KV260_IOC_FREE_BUFFER);
        close(dev->fd);
    }
    delete dev;
}

int versa_p_reset(versa_p_device *dev)
{
    if (!dev || dev->fd < 0) {
        return VERSA_P_ERR_INVAL;
    }
    if (ioctl(dev->fd, NPU_KV260_IOC_RESET_DEV) < 0) {
        return VERSA_P_ERR_IO;
    }
    reset_software_state(dev);
    versa_p_write64(dev, VERSA_P_REG_GLOBAL_CLEAR, 1ull);
    return VERSA_P_OK;
}

int versa_p_get_info(versa_p_device *dev, versa_p_device_info *out_info)
{
    if (!dev || !out_info) {
        return VERSA_P_ERR_INVAL;
    }
    npu_kv260_info info = {};
    if (ioctl(dev->fd, NPU_KV260_IOC_GET_INFO, &info) == 0) {
        dev->info = info;
    }
    out_info->regs_phys = dev->info.regs_phys;
    out_info->regs_size = dev->info.regs_size;
    out_info->dma_addr_bits = dev->info.dma_addr_bits;
    out_info->has_irq = dev->info.has_irq;
    out_info->flags = dev->info.flags;
    out_info->cma_dma_addr = dev->cma_dma;
    out_info->cma_size = dev->cma_size;
    out_info->cma_vaddr = dev->cma;
    return VERSA_P_OK;
}

int versa_p_mem_alloc(versa_p_device *dev, uint32_t size, uint32_t alignment,
                      versa_p_buffer *out_buffer)
{
    if (!dev || !out_buffer || size == 0) {
        return VERSA_P_ERR_INVAL;
    }
    if (alignment == 0) {
        alignment = VERSA_P_ALIGN_BYTES;
    }
    if ((alignment & (alignment - 1u)) != 0) {
        return VERSA_P_ERR_INVAL;
    }
    size = versa_p_align_up(size, VERSA_P_ALIGN_BYTES);

    for (size_t i = 0; i < dev->blocks.size(); ++i) {
        VersaPBlock &block = dev->blocks[i];
        if (!block.free) {
            continue;
        }
        uint32_t aligned = versa_p_align_up(block.offset, alignment);
        uint32_t pad = aligned - block.offset;
        if (block.size < pad || block.size - pad < size) {
            continue;
        }

        std::vector<VersaPBlock> replacement;
        if (pad != 0) {
            replacement.push_back(VersaPBlock{block.offset, pad, true});
        }
        replacement.push_back(VersaPBlock{aligned, size, false});
        uint32_t tail = block.size - pad - size;
        if (tail != 0) {
            replacement.push_back(VersaPBlock{aligned + size, tail, true});
        }
        dev->blocks.erase(dev->blocks.begin() + (long)i);
        dev->blocks.insert(dev->blocks.begin() + (long)i,
                           replacement.begin(), replacement.end());

        out_buffer->offset = aligned;
        out_buffer->size = size;
        out_buffer->vaddr = (uint8_t *)dev->cma + aligned;
        out_buffer->dma_addr = dev->cma_dma + aligned;
        return VERSA_P_OK;
    }
    return VERSA_P_ERR_NOMEM;
}

void versa_p_mem_free(versa_p_device *dev, versa_p_buffer *buffer)
{
    if (!dev || !buffer || !buffer->vaddr || buffer->size == 0) {
        return;
    }
    for (size_t i = 0; i < dev->blocks.size(); ++i) {
        if (dev->blocks[i].offset == buffer->offset && !dev->blocks[i].free) {
            dev->blocks[i].free = true;
            if (i + 1 < dev->blocks.size() && dev->blocks[i + 1].free) {
                dev->blocks[i].size += dev->blocks[i + 1].size;
                dev->blocks.erase(dev->blocks.begin() + (long)i + 1);
            }
            if (i > 0 && dev->blocks[i - 1].free) {
                dev->blocks[i - 1].size += dev->blocks[i].size;
                dev->blocks.erase(dev->blocks.begin() + (long)i);
            }
            break;
        }
    }
    buffer->vaddr = nullptr;
    buffer->dma_addr = 0;
    buffer->offset = 0;
    buffer->size = 0;
}

uint32_t versa_p_dma_addr(versa_p_device *dev, const void *ptr)
{
    if (!dev || !ptr || !dev->cma) {
        return 0;
    }
    const uint8_t *base = (const uint8_t *)dev->cma;
    const uint8_t *p = (const uint8_t *)ptr;
    if (p < base || p >= base + dev->cma_size) {
        return 0;
    }
    return dev->cma_dma + (uint32_t)(p - base);
}

int versa_p_sync_dma_range(versa_p_device *dev, uint32_t dma_addr,
                           size_t bytes, uint32_t target,
                           uint32_t direction)
{
    if (!dev || bytes == 0) {
        return VERSA_P_ERR_INVAL;
    }
    if (dma_addr < dev->cma_dma ||
        (uint64_t)dma_addr + bytes > (uint64_t)dev->cma_dma + dev->cma_size) {
        return VERSA_P_ERR_INVAL;
    }
    if ((dev->info.flags & NPU_KV260_INFO_FLAG_CACHEABLE_BUFFER) == 0) {
        return VERSA_P_OK;
    }

    npu_kv260_buffer_sync sync = {};
    sync.cma_offset = (uint64_t)(dma_addr - dev->cma_dma);
    sync.size = bytes;
    sync.target = target;
    sync.direction = direction;
    if (ioctl(dev->fd, NPU_KV260_IOC_SYNC_BUFFER, &sync) != 0) {
        return VERSA_P_ERR_IO;
    }
    return VERSA_P_OK;
}

int versa_p_sync_for_cpu(versa_p_device *dev, const void *cma_ptr,
                         size_t bytes)
{
    uint32_t dma_addr = versa_p_dma_addr(dev, cma_ptr);
    if (dma_addr == 0) {
        return VERSA_P_ERR_INVAL;
    }
    return versa_p_sync_dma_range(dev, dma_addr, bytes,
                                  NPU_KV260_SYNC_FOR_CPU,
                                  NPU_KV260_SYNC_FROM_DEVICE);
}

int versa_p_sync_for_device(versa_p_device *dev, const void *cma_ptr,
                            size_t bytes)
{
    uint32_t dma_addr = versa_p_dma_addr(dev, cma_ptr);
    if (dma_addr == 0) {
        return VERSA_P_ERR_INVAL;
    }
    return versa_p_sync_dma_range(dev, dma_addr, bytes,
                                  NPU_KV260_SYNC_FOR_DEVICE,
                                  NPU_KV260_SYNC_TO_DEVICE);
}

int versa_p_dma_copy_from_cma(versa_p_device *dev, const void *cma_ptr,
                              void *dst, size_t bytes,
                              uint32_t chunk_bytes)
{
    if (!dev || !cma_ptr || !dst || bytes == 0 || !dev->cma) {
        return VERSA_P_ERR_INVAL;
    }

    const uint8_t *base = (const uint8_t *)dev->cma;
    const uint8_t *p = (const uint8_t *)cma_ptr;
    if (p < base || p > base + dev->cma_size) {
        return VERSA_P_ERR_INVAL;
    }

    const uint64_t offset = (uint64_t)(p - base);
    if (offset > dev->cma_size || bytes > (size_t)(dev->cma_size - offset)) {
        return VERSA_P_ERR_INVAL;
    }

    npu_kv260_dma_copy copy = {};
    int rc = versa_p_sync_for_cpu(dev, cma_ptr, bytes);
    if (rc != VERSA_P_OK) {
        return rc;
    }
    copy.cma_offset = offset;
    copy.user_addr = (uint64_t)(uintptr_t)dst;
    copy.size = bytes;
    copy.direction = NPU_KV260_DMA_COPY_CMA_TO_USER;
    copy.chunk_bytes = chunk_bytes;
    if (ioctl(dev->fd, NPU_KV260_IOC_DMA_COPY, &copy) != 0) {
        if (errno == EOPNOTSUPP || errno == ENOTTY || errno == EINVAL) {
            return VERSA_P_ERR_UNSUPPORTED_MODE;
        }
        return VERSA_P_ERR_IO;
    }
    return VERSA_P_OK;
}
