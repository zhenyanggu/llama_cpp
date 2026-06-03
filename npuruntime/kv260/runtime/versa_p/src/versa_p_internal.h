#ifndef VERSA_P_INTERNAL_H
#define VERSA_P_INTERNAL_H

#include "versa_p_runtime.h"
#include "npu_kv260_uapi.h"

#include <stdint.h>
#include <sys/types.h>
#include <vector>

struct VersaPBlock {
    uint32_t offset;
    uint32_t size;
    bool free;
};

struct VersaPBankState {
    bool loaded = false;
    uint16_t k = 0;
    uint16_t n = 0;
};

struct VersaPABankState {
    bool loaded = false;
    uint16_t m = 0;
    uint16_t k = 0;
};

struct versa_p_device {
    int fd = -1;
    void *regs = nullptr;
    void *cma = nullptr;
    uint32_t cma_dma = 0;
    uint32_t cma_size = 0;
    npu_kv260_info info = {};
    npu_kv260_hw_state hw = {};
    std::vector<VersaPBlock> blocks;
    bool api_inflight[5] = {};
    uint8_t active_a_bank = 0;
    VersaPABankState a_bank[2];
    uint8_t pending_mvin_a_bank = 0;
    uint8_t mvin_a_inflight_bank = 0;
    uint16_t pending_mvin_a_m = 0;
    uint16_t pending_mvin_a_k = 0;
    uint8_t active_w_bank = 1;
    VersaPBankState w_bank[2];
    uint8_t pending_mvin_w_bank = 0;
    uint16_t pending_mvin_w_k = 0;
    uint16_t pending_mvin_w_n = 0;
    uint8_t gemm_inflight_a_bank = 0;
    uint8_t gemm_inflight_o_bank = 0;
    uint8_t mvout_inflight_o_bank = 0;
    uint32_t pending_mvout_dma_addr = 0;
    uint32_t pending_mvout_bytes = 0;
};

static inline bool versa_p_is_aligned(uint64_t value, uint32_t alignment)
{
    return alignment != 0 && (value & (uint64_t)(alignment - 1u)) == 0;
}

static inline uint32_t versa_p_align_up(uint32_t value, uint32_t alignment)
{
    return (value + alignment - 1u) & ~(alignment - 1u);
}

uint32_t versa_p_status_offset(versa_p_api api);
int versa_p_hw_error_to_status(uint8_t error_code);
void versa_p_mark_api_complete(versa_p_device *dev, versa_p_api api);
int versa_p_sync_dma_range(versa_p_device *dev, uint32_t dma_addr,
                           size_t bytes, uint32_t target,
                           uint32_t direction);

#endif
