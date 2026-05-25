#include "versa_p_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

namespace {

uint64_t monotonic_ms()
{
    timespec ts = {};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000ull + (uint64_t)ts.tv_nsec / 1000000ull;
}

const char *api_name(versa_p_api api)
{
    switch (api) {
    case VERSA_P_API_MVIN_A: return "mvin_a";
    case VERSA_P_API_MVIN_W: return "mvin_w";
    case VERSA_P_API_MVIN_META: return "mvin_meta";
    case VERSA_P_API_GEMM_I8: return "gemm_i8";
    case VERSA_P_API_MVOUT: return "mvout";
    default: return "unknown";
    }
}

bool debug_enabled()
{
    const char *env = getenv("VERSA_P_DEBUG");
    return env && env[0] != '\0' && env[0] != '0';
}

} // namespace

uint32_t versa_p_status_offset(versa_p_api api)
{
    switch (api) {
    case VERSA_P_API_MVIN_A: return VERSA_P_REG_MVIN_A_STATUS;
    case VERSA_P_API_MVIN_W: return VERSA_P_REG_MVIN_W_STATUS;
    case VERSA_P_API_MVIN_META: return VERSA_P_REG_MVIN_META_STATUS;
    case VERSA_P_API_GEMM_I8: return VERSA_P_REG_GEMM_STATUS;
    case VERSA_P_API_MVOUT: return VERSA_P_REG_MVOUT_STATUS;
    default: return 0;
    }
}

int versa_p_hw_error_to_status(uint8_t error_code)
{
    switch (error_code) {
    case VERSA_P_HW_ERR_NONE: return VERSA_P_OK;
    case VERSA_P_HW_ERR_ILLEGAL_SHAPE: return VERSA_P_ERR_ILLEGAL_SHAPE;
    case VERSA_P_HW_ERR_START_WHILE_BUSY: return VERSA_P_ERR_BUSY;
    case VERSA_P_HW_ERR_RESOURCE_CONFLICT: return VERSA_P_ERR_RESOURCE_CONFLICT;
    case VERSA_P_HW_ERR_BANK_CONFLICT: return VERSA_P_ERR_BANK_CONFLICT;
    case VERSA_P_HW_ERR_BANK_NOT_VALID: return VERSA_P_ERR_BANK_NOT_VALID;
    case VERSA_P_HW_ERR_SHAPE_MISMATCH: return VERSA_P_ERR_SHAPE_MISMATCH;
    case VERSA_P_HW_ERR_ALIGNMENT: return VERSA_P_ERR_ALIGNMENT;
    case VERSA_P_HW_ERR_UNSUPPORTED_MODE: return VERSA_P_ERR_UNSUPPORTED_MODE;
    case VERSA_P_HW_ERR_ILLEGAL_FLAGS: return VERSA_P_ERR_ILLEGAL_FLAGS;
    default: return VERSA_P_ERR_HARDWARE;
    }
}

void versa_p_mark_api_complete(versa_p_device *dev, versa_p_api api)
{
    if (!dev || api < VERSA_P_API_MVIN_A || api > VERSA_P_API_MVOUT) {
        return;
    }
    dev->api_inflight[(int)api] = false;
    if (api == VERSA_P_API_MVIN_A) {
        VersaPABankState &bank = dev->a_bank[dev->pending_mvin_a_bank & 1u];
        bank.loaded = true;
        bank.m = dev->pending_mvin_a_m;
        bank.k = dev->pending_mvin_a_k;
    }
    if (api == VERSA_P_API_MVIN_W) {
        VersaPBankState &bank = dev->w_bank[dev->pending_mvin_w_bank & 1u];
        bank.loaded = true;
        bank.k = dev->pending_mvin_w_k;
        bank.n = dev->pending_mvin_w_n;
    }
}

uint64_t versa_p_read64(versa_p_device *dev, uint32_t offset)
{
    if (!dev || !dev->regs || (offset & 7u) != 0) {
        return 0;
    }
    volatile uint64_t *reg = (volatile uint64_t *)((uint8_t *)dev->regs + offset);
    return *reg;
}

void versa_p_write64(versa_p_device *dev, uint32_t offset, uint64_t value)
{
    if (!dev || !dev->regs || (offset & 7u) != 0) {
        return;
    }
    volatile uint64_t *reg = (volatile uint64_t *)((uint8_t *)dev->regs + offset);
    *reg = value;
}

int versa_p_read_status(versa_p_device *dev, versa_p_api api,
                        versa_p_reg_status *out_status)
{
    if (!dev || !out_status || api < VERSA_P_API_MVIN_A || api > VERSA_P_API_MVOUT) {
        return VERSA_P_ERR_INVAL;
    }
    *out_status = versa_p_decode_status(versa_p_read64(dev, versa_p_status_offset(api)));
    return VERSA_P_OK;
}

int versa_p_profile_start(versa_p_device *dev)
{
    if (!dev) {
        return VERSA_P_ERR_INVAL;
    }
    versa_p_write64(dev, VERSA_P_REG_PROFILE_CTRL, 0x3ull);
    return VERSA_P_OK;
}

int versa_p_profile_stop(versa_p_device *dev)
{
    if (!dev) {
        return VERSA_P_ERR_INVAL;
    }
    versa_p_write64(dev, VERSA_P_REG_PROFILE_CTRL, 0x0ull);
    return VERSA_P_OK;
}

int versa_p_profile_read(versa_p_device *dev,
                         versa_p_profile_counters *out_counters)
{
    if (!dev || !out_counters) {
        return VERSA_P_ERR_INVAL;
    }
    out_counters->global_cycles =
        versa_p_read64(dev, VERSA_P_REG_PROFILE_GLOBAL);
    out_counters->mvin_a_busy_cycles =
        versa_p_read64(dev, VERSA_P_REG_PROFILE_A_BUSY);
    out_counters->mvin_w_busy_cycles =
        versa_p_read64(dev, VERSA_P_REG_PROFILE_W_BUSY);
    out_counters->mvin_meta_busy_cycles =
        versa_p_read64(dev, VERSA_P_REG_PROFILE_META_BUSY);
    out_counters->gemm_busy_cycles =
        versa_p_read64(dev, VERSA_P_REG_PROFILE_GEMM_BUSY);
    out_counters->mvout_busy_cycles =
        versa_p_read64(dev, VERSA_P_REG_PROFILE_MVOUT_BUSY);
    out_counters->busy_any_cycles =
        versa_p_read64(dev, VERSA_P_REG_PROFILE_BUSY_ANY);
    out_counters->busy_multi_cycles =
        versa_p_read64(dev, VERSA_P_REG_PROFILE_BUSY_MULTI);

    const uint32_t r_beats_regs[4] = {
        VERSA_P_REG_PROFILE_AXI0_R_BEATS,
        VERSA_P_REG_PROFILE_AXI1_R_BEATS,
        VERSA_P_REG_PROFILE_AXI2_R_BEATS,
        VERSA_P_REG_PROFILE_AXI3_R_BEATS,
    };
    const uint32_t w_beats_regs[4] = {
        VERSA_P_REG_PROFILE_AXI0_W_BEATS,
        VERSA_P_REG_PROFILE_AXI1_W_BEATS,
        VERSA_P_REG_PROFILE_AXI2_W_BEATS,
        VERSA_P_REG_PROFILE_AXI3_W_BEATS,
    };
    const uint32_t ar_stall_regs[4] = {
        VERSA_P_REG_PROFILE_AXI0_AR_STALL,
        VERSA_P_REG_PROFILE_AXI1_AR_STALL,
        VERSA_P_REG_PROFILE_AXI2_AR_STALL,
        VERSA_P_REG_PROFILE_AXI3_AR_STALL,
    };
    const uint32_t r_stall_regs[4] = {
        VERSA_P_REG_PROFILE_AXI0_R_STALL,
        VERSA_P_REG_PROFILE_AXI1_R_STALL,
        VERSA_P_REG_PROFILE_AXI2_R_STALL,
        VERSA_P_REG_PROFILE_AXI3_R_STALL,
    };
    const uint32_t aw_stall_regs[4] = {
        VERSA_P_REG_PROFILE_AXI0_AW_STALL,
        VERSA_P_REG_PROFILE_AXI1_AW_STALL,
        VERSA_P_REG_PROFILE_AXI2_AW_STALL,
        VERSA_P_REG_PROFILE_AXI3_AW_STALL,
    };
    const uint32_t w_stall_regs[4] = {
        VERSA_P_REG_PROFILE_AXI0_W_STALL,
        VERSA_P_REG_PROFILE_AXI1_W_STALL,
        VERSA_P_REG_PROFILE_AXI2_W_STALL,
        VERSA_P_REG_PROFILE_AXI3_W_STALL,
    };
    for (int i = 0; i < 4; ++i) {
        out_counters->axi_r_beats[i] = versa_p_read64(dev, r_beats_regs[i]);
        out_counters->axi_w_beats[i] = versa_p_read64(dev, w_beats_regs[i]);
        out_counters->axi_ar_stall_cycles[i] = versa_p_read64(dev, ar_stall_regs[i]);
        out_counters->axi_r_stall_cycles[i] = versa_p_read64(dev, r_stall_regs[i]);
        out_counters->axi_aw_stall_cycles[i] = versa_p_read64(dev, aw_stall_regs[i]);
        out_counters->axi_w_stall_cycles[i] = versa_p_read64(dev, w_stall_regs[i]);
    }
    return VERSA_P_OK;
}

int versa_p_wait(versa_p_device *dev, versa_p_api api, uint32_t timeout_ms)
{
    if (!dev || api < VERSA_P_API_MVIN_A || api > VERSA_P_API_MVOUT) {
        return VERSA_P_ERR_INVAL;
    }
    const uint64_t start = monotonic_ms();
    for (;;) {
        versa_p_reg_status status = {};
        int rc = versa_p_read_status(dev, api, &status);
        if (rc != VERSA_P_OK) {
            return rc;
        }
        if (status.error) {
            dev->api_inflight[(int)api] = false;
            versa_p_write64(dev, versa_p_status_offset(api), VERSA_P_STATUS_ERROR_MASK);
            return versa_p_hw_error_to_status(status.error_code);
        }
        if (status.done) {
            if (api == VERSA_P_API_MVOUT && dev->pending_mvout_bytes != 0) {
                rc = versa_p_sync_dma_range(
                    dev, dev->pending_mvout_dma_addr, dev->pending_mvout_bytes,
                    NPU_KV260_SYNC_FOR_CPU,
                    NPU_KV260_SYNC_FROM_DEVICE);
                if (rc != VERSA_P_OK) {
                    dev->api_inflight[(int)api] = false;
                    return rc;
                }
                dev->pending_mvout_dma_addr = 0;
                dev->pending_mvout_bytes = 0;
            }
            versa_p_mark_api_complete(dev, api);
            versa_p_write64(dev, versa_p_status_offset(api), VERSA_P_STATUS_DONE_MASK);
            return VERSA_P_OK;
        }
        if (timeout_ms != 0 && monotonic_ms() - start >= timeout_ms) {
            if (debug_enabled()) {
                fprintf(stderr,
                        "versa_p_wait timeout api=%s raw=0x%016llx busy=%u done=%u error=%u err=%u accepted=%u\n",
                        api_name(api),
                        (unsigned long long)status.raw,
                        status.busy,
                        status.done,
                        status.error,
                        status.error_code,
                        status.accepted_count);
            }
            return VERSA_P_ERR_TIMEOUT;
        }
        usleep(100);
    }
}
