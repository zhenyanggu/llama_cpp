#ifndef NPU_KV260_UAPI_H
#define NPU_KV260_UAPI_H

#include <linux/ioctl.h>
#include <linux/types.h>

#define NPU_KV260_DEVICE_NAME "npu_kv260"
#define NPU_KV260_DEV_PATH "/dev/npu_kv260"

#define NPU_KV260_MMAP_BUFFER_OFFSET 0x00000000ULL
#define NPU_KV260_MMAP_REGS_OFFSET   0x10000000ULL
#define NPU_KV260_REG_MMAP_SIZE      0x10000U

#define NPU_KV260_IRQ_MODE_KERNEL    0U
#define NPU_KV260_IRQ_MODE_USERSPACE 1U

#define NPU_KV260_HW_MAGIC       0x4e505547U
#define NPU_KV260_HW_ABI_VERSION 1U
#define NPU_KV260_MODE_UNKNOWN   0U
#define NPU_KV260_MODE_PREFILL   1U
#define NPU_KV260_MODE_DECODE    2U

struct npu_kv260_buffer_request {
    __u64 size;
    __u64 dma_addr;
};

struct npu_kv260_buffer_info {
    __u64 size;
    __u64 dma_addr;
};

struct npu_kv260_info {
    __u64 regs_phys;
    __u32 regs_size;
    __u32 dma_addr_bits;
    __u32 has_irq;
    __u32 flags;
};

struct npu_kv260_reg_access {
    __u32 offset;
    __u32 value;
};

struct npu_kv260_hw_state {
    __u32 magic;
    __u32 abi_version;
    __u32 mode_id;
    __u32 caps;
    __u32 status;
    __u32 error;
};

struct npu_kv260_clock_rate {
    __u64 requested_hz;
    __u64 actual_hz;
};

#define NPU_KV260_IOC_MAGIC 'N'

#define NPU_KV260_IOC_WAIT_IRQ        _IOR(NPU_KV260_IOC_MAGIC, 1, __u32)
#define NPU_KV260_IOC_GET_BUFFER_INFO _IOR(NPU_KV260_IOC_MAGIC, 2, struct npu_kv260_buffer_info)
#define NPU_KV260_IOC_RESET_DEV       _IO(NPU_KV260_IOC_MAGIC, 3)
#define NPU_KV260_IOC_SET_IRQ_MODE    _IOW(NPU_KV260_IOC_MAGIC, 4, __u32)
#define NPU_KV260_IOC_ALLOC_BUFFER    _IOWR(NPU_KV260_IOC_MAGIC, 5, struct npu_kv260_buffer_request)
#define NPU_KV260_IOC_FREE_BUFFER     _IO(NPU_KV260_IOC_MAGIC, 6)
#define NPU_KV260_IOC_GET_INFO        _IOR(NPU_KV260_IOC_MAGIC, 7, struct npu_kv260_info)
#define NPU_KV260_IOC_REG_READ        _IOWR(NPU_KV260_IOC_MAGIC, 8, struct npu_kv260_reg_access)
#define NPU_KV260_IOC_REG_WRITE       _IOW(NPU_KV260_IOC_MAGIC, 9, struct npu_kv260_reg_access)
#define NPU_KV260_IOC_REINIT          _IOR(NPU_KV260_IOC_MAGIC, 13, struct npu_kv260_hw_state)
#define NPU_KV260_IOC_GET_HW_STATE    _IOR(NPU_KV260_IOC_MAGIC, 14, struct npu_kv260_hw_state)
#define NPU_KV260_IOC_SET_CLOCK_RATE  _IOWR(NPU_KV260_IOC_MAGIC, 15, struct npu_kv260_clock_rate)

#endif
