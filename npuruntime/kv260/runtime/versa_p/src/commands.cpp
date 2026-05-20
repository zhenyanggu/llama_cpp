#include "versa_p_internal.h"

namespace {

bool valid_api_idle(versa_p_device *dev, versa_p_api api)
{
    return dev && !dev->api_inflight[(int)api];
}

bool resource_busy_for(versa_p_device *dev, versa_p_api api)
{
    switch (api) {
    case VERSA_P_API_MVIN_A:
        return dev->api_inflight[VERSA_P_API_GEMM_I8];
    case VERSA_P_API_MVIN_META:
        return dev->api_inflight[VERSA_P_API_GEMM_I8] ||
               dev->api_inflight[VERSA_P_API_MVOUT];
    case VERSA_P_API_GEMM_I8:
        return dev->api_inflight[VERSA_P_API_MVIN_A] ||
               dev->api_inflight[VERSA_P_API_MVIN_META] ||
               dev->api_inflight[VERSA_P_API_MVOUT];
    case VERSA_P_API_MVOUT:
        return dev->api_inflight[VERSA_P_API_MVIN_META] ||
               dev->api_inflight[VERSA_P_API_GEMM_I8];
    default:
        return false;
    }
}

int validate_mvin_a(const versa_p_mvin_a_desc *desc)
{
    if (!desc || desc->m == 0 || desc->k == 0 || desc->k > VERSA_P_K_MAX ||
        desc->dram_row_stride_bytes < desc->k) {
        return VERSA_P_ERR_ILLEGAL_SHAPE;
    }
    if (!versa_p_is_aligned(desc->dram_base, VERSA_P_ALIGN_BYTES)) {
        return VERSA_P_ERR_ALIGNMENT;
    }
    return VERSA_P_OK;
}

int validate_mvin_w(versa_p_device *dev, const versa_p_mvin_w_desc *desc)
{
    if (!desc || desc->k == 0 || desc->n == 0 || desc->k > VERSA_P_K_MAX ||
        desc->w_bank > 1) {
        return VERSA_P_ERR_ILLEGAL_SHAPE;
    }
    if (!versa_p_is_aligned(desc->dram_base, VERSA_P_ALIGN_BYTES)) {
        return VERSA_P_ERR_ALIGNMENT;
    }
    if (desc->w_bank == dev->active_w_bank) {
        return VERSA_P_ERR_BANK_CONFLICT;
    }
    return VERSA_P_OK;
}

int validate_mvin_meta(const versa_p_mvin_meta_desc *desc)
{
    if (!desc || desc->byte_count == 0 || desc->meta_type > 3) {
        return VERSA_P_ERR_ILLEGAL_SHAPE;
    }
    if (!versa_p_is_aligned(desc->dram_base, VERSA_P_ALIGN_BYTES) ||
        !versa_p_is_aligned(desc->meta_offset_bytes, VERSA_P_ALIGN_BYTES)) {
        return VERSA_P_ERR_ALIGNMENT;
    }
    if (desc->meta_offset_bytes > VERSA_P_META_BYTES ||
        desc->byte_count > VERSA_P_META_BYTES - desc->meta_offset_bytes) {
        return VERSA_P_ERR_ILLEGAL_SHAPE;
    }
    return VERSA_P_OK;
}

int validate_gemm_i8(versa_p_device *dev, const versa_p_gemm_i8_desc *desc)
{
    if (!desc || desc->m == 0 || desc->n == 0 || desc->k == 0 ||
        desc->k > VERSA_P_K_MAX || desc->w_bank > 1) {
        return VERSA_P_ERR_ILLEGAL_SHAPE;
    }
    if (desc->accumulate_en && desc->add_bias_en) {
        return VERSA_P_ERR_ILLEGAL_FLAGS;
    }
    const VersaPBankState &bank = dev->w_bank[desc->w_bank];
    if (!bank.loaded) {
        return VERSA_P_ERR_BANK_NOT_VALID;
    }
    if (bank.k != desc->k || desc->n > bank.n) {
        return VERSA_P_ERR_SHAPE_MISMATCH;
    }
    if (desc->add_bias_en &&
        !versa_p_is_aligned(desc->bias_offset_bytes, VERSA_P_ALIGN_BYTES)) {
        return VERSA_P_ERR_ALIGNMENT;
    }
    return VERSA_P_OK;
}

int validate_mvout(const versa_p_mvout_desc *desc)
{
    if (!desc || desc->m == 0 || desc->n == 0 ||
        (desc->output_stride_n != 0 && desc->output_stride_n < desc->n)) {
        return VERSA_P_ERR_ILLEGAL_SHAPE;
    }
    if (!versa_p_is_aligned(desc->dram_base, VERSA_P_ALIGN_BYTES)) {
        return VERSA_P_ERR_ALIGNMENT;
    }
    if (desc->mode > VERSA_P_MVOUT_FP32_PER_CHANNEL_Q8_24) {
        return VERSA_P_ERR_UNSUPPORTED_MODE;
    }
    if (desc->mode == VERSA_P_MVOUT_FP32_PER_CHANNEL_Q8_24 &&
        !versa_p_is_aligned(desc->scale_param, VERSA_P_ALIGN_BYTES)) {
        return VERSA_P_ERR_ALIGNMENT;
    }
    return VERSA_P_OK;
}

void mark_started(versa_p_device *dev, versa_p_api api)
{
    dev->api_inflight[(int)api] = true;
}

} // namespace

int versa_p_start_mvin_a(versa_p_device *dev, const versa_p_mvin_a_desc *desc)
{
    if (!dev) {
        return VERSA_P_ERR_INVAL;
    }
    if (!valid_api_idle(dev, VERSA_P_API_MVIN_A)) {
        return VERSA_P_ERR_BUSY;
    }
    if (resource_busy_for(dev, VERSA_P_API_MVIN_A)) {
        return VERSA_P_ERR_RESOURCE_CONFLICT;
    }
    int rc = validate_mvin_a(desc);
    if (rc != VERSA_P_OK) {
        return rc;
    }

    uint64_t desc0 = (uint64_t)desc->dram_base |
                     ((uint64_t)desc->dram_row_stride_bytes << 32);
    uint64_t desc1 = (uint64_t)desc->m |
                     ((uint64_t)desc->k << 16) |
                     ((uint64_t)(desc->u8_minus_128 != 0) << 32) |
                     (1ull << VERSA_P_START_MVIN_A_BIT);
    versa_p_write64(dev, VERSA_P_REG_MVIN_A_DESC0, desc0);
    versa_p_write64(dev, VERSA_P_REG_MVIN_A_DESC1, desc1);
    mark_started(dev, VERSA_P_API_MVIN_A);
    return VERSA_P_OK;
}

int versa_p_start_mvin_w(versa_p_device *dev, const versa_p_mvin_w_desc *desc)
{
    if (!dev) {
        return VERSA_P_ERR_INVAL;
    }
    if (!valid_api_idle(dev, VERSA_P_API_MVIN_W)) {
        return VERSA_P_ERR_BUSY;
    }
    int rc = validate_mvin_w(dev, desc);
    if (rc != VERSA_P_OK) {
        return rc;
    }

    uint64_t desc1 = (uint64_t)desc->k |
                     ((uint64_t)desc->n << 16) |
                     ((uint64_t)(desc->w_bank & 1u) << 32) |
                     (1ull << VERSA_P_START_MVIN_W_BIT);
    versa_p_write64(dev, VERSA_P_REG_MVIN_W_DESC0, desc->dram_base);
    versa_p_write64(dev, VERSA_P_REG_MVIN_W_DESC1, desc1);
    dev->w_bank[desc->w_bank].loaded = false;
    dev->pending_mvin_w_bank = desc->w_bank;
    dev->pending_mvin_w_k = desc->k;
    dev->pending_mvin_w_n = desc->n;
    mark_started(dev, VERSA_P_API_MVIN_W);
    return VERSA_P_OK;
}

int versa_p_start_mvin_meta(versa_p_device *dev,
                            const versa_p_mvin_meta_desc *desc)
{
    if (!dev) {
        return VERSA_P_ERR_INVAL;
    }
    if (!valid_api_idle(dev, VERSA_P_API_MVIN_META)) {
        return VERSA_P_ERR_BUSY;
    }
    if (resource_busy_for(dev, VERSA_P_API_MVIN_META)) {
        return VERSA_P_ERR_RESOURCE_CONFLICT;
    }
    int rc = validate_mvin_meta(desc);
    if (rc != VERSA_P_OK) {
        return rc;
    }

    uint64_t desc0 = (uint64_t)desc->dram_base |
                     ((uint64_t)desc->meta_offset_bytes << 32);
    uint64_t desc1 = (uint64_t)desc->byte_count |
                     ((uint64_t)(desc->meta_type & 3u) << 32) |
                     (1ull << VERSA_P_START_MVIN_META_BIT);
    versa_p_write64(dev, VERSA_P_REG_MVIN_META_DESC0, desc0);
    versa_p_write64(dev, VERSA_P_REG_MVIN_META_DESC1, desc1);
    mark_started(dev, VERSA_P_API_MVIN_META);
    return VERSA_P_OK;
}

int versa_p_start_gemm_i8(versa_p_device *dev, const versa_p_gemm_i8_desc *desc)
{
    if (!dev) {
        return VERSA_P_ERR_INVAL;
    }
    if (!valid_api_idle(dev, VERSA_P_API_GEMM_I8)) {
        return VERSA_P_ERR_BUSY;
    }
    if (resource_busy_for(dev, VERSA_P_API_GEMM_I8)) {
        return VERSA_P_ERR_RESOURCE_CONFLICT;
    }
    int rc = validate_gemm_i8(dev, desc);
    if (rc != VERSA_P_OK) {
        return rc;
    }

    uint64_t desc0 = (uint64_t)desc->m |
                     ((uint64_t)desc->n << 16) |
                     ((uint64_t)desc->k << 32) |
                     ((uint64_t)(desc->w_bank & 1u) << 48) |
                     ((uint64_t)(desc->accumulate_en != 0) << 49) |
                     ((uint64_t)(desc->add_bias_en != 0) << 50) |
                     (1ull << VERSA_P_START_GEMM_BIT);
    versa_p_write64(dev, VERSA_P_REG_GEMM_DESC1, desc->bias_offset_bytes);
    versa_p_write64(dev, VERSA_P_REG_GEMM_DESC0, desc0);
    dev->active_w_bank = desc->w_bank;
    mark_started(dev, VERSA_P_API_GEMM_I8);
    return VERSA_P_OK;
}

int versa_p_start_mvout(versa_p_device *dev, const versa_p_mvout_desc *desc)
{
    if (!dev) {
        return VERSA_P_ERR_INVAL;
    }
    if (!valid_api_idle(dev, VERSA_P_API_MVOUT)) {
        return VERSA_P_ERR_BUSY;
    }
    if (resource_busy_for(dev, VERSA_P_API_MVOUT)) {
        return VERSA_P_ERR_RESOURCE_CONFLICT;
    }
    int rc = validate_mvout(desc);
    if (rc != VERSA_P_OK) {
        return rc;
    }

    uint64_t desc0 = (uint64_t)desc->dram_base |
                     ((uint64_t)desc->scale_param << 32);
    uint64_t desc1 = (uint64_t)desc->m |
                     ((uint64_t)desc->n << 16) |
                     ((uint64_t)desc->output_stride_n << 32) |
                     ((uint64_t)(desc->mode & 3u) << 48) |
                     (1ull << VERSA_P_START_MVOUT_BIT);
    versa_p_write64(dev, VERSA_P_REG_MVOUT_DESC0, desc0);
    versa_p_write64(dev, VERSA_P_REG_MVOUT_DESC1, desc1);
    mark_started(dev, VERSA_P_API_MVOUT);
    return VERSA_P_OK;
}

int versa_p_mvin_a(versa_p_device *dev, const versa_p_mvin_a_desc *desc,
                   uint32_t timeout_ms)
{
    int rc = versa_p_start_mvin_a(dev, desc);
    return rc == VERSA_P_OK ? versa_p_wait(dev, VERSA_P_API_MVIN_A, timeout_ms) : rc;
}

int versa_p_mvin_w(versa_p_device *dev, const versa_p_mvin_w_desc *desc,
                   uint32_t timeout_ms)
{
    int rc = versa_p_start_mvin_w(dev, desc);
    return rc == VERSA_P_OK ? versa_p_wait(dev, VERSA_P_API_MVIN_W, timeout_ms) : rc;
}

int versa_p_mvin_meta(versa_p_device *dev, const versa_p_mvin_meta_desc *desc,
                      uint32_t timeout_ms)
{
    int rc = versa_p_start_mvin_meta(dev, desc);
    return rc == VERSA_P_OK ? versa_p_wait(dev, VERSA_P_API_MVIN_META, timeout_ms) : rc;
}

int versa_p_gemm_i8(versa_p_device *dev, const versa_p_gemm_i8_desc *desc,
                    uint32_t timeout_ms)
{
    int rc = versa_p_start_gemm_i8(dev, desc);
    return rc == VERSA_P_OK ? versa_p_wait(dev, VERSA_P_API_GEMM_I8, timeout_ms) : rc;
}

int versa_p_mvout(versa_p_device *dev, const versa_p_mvout_desc *desc,
                  uint32_t timeout_ms)
{
    int rc = versa_p_start_mvout(dev, desc);
    return rc == VERSA_P_OK ? versa_p_wait(dev, VERSA_P_API_MVOUT, timeout_ms) : rc;
}
