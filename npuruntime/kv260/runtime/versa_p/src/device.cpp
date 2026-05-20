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

uint32_t resolve_cma_size(const versa_p_options *options)
{
    if (options && options->cma_size != 0) {
        return options->cma_size;
    }
    const char *env = getenv("VERSA_P_CMA_SIZE");
    if (env && env[0] != '\0') {
        char *end = nullptr;
        unsigned long parsed = strtoul(env, &end, 0);
        if (end != env && parsed != 0 &&
            parsed <= std::numeric_limits<uint32_t>::max()) {
            return (uint32_t)parsed;
        }
    }
    return VERSA_P_DEFAULT_CMA_BYTES;
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
    dev->active_w_bank = 1;
    dev->w_bank[0] = VersaPBankState{};
    dev->w_bank[1] = VersaPBankState{};
    dev->pending_mvin_w_bank = 0;
    dev->pending_mvin_w_k = 0;
    dev->pending_mvin_w_n = 0;
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
    req.size = resolve_cma_size(options);
    if (ioctl(dev->fd, NPU_KV260_IOC_ALLOC_BUFFER, &req) < 0) {
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
    dev->blocks.push_back(VersaPBlock{0, dev->cma_size, true});
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
