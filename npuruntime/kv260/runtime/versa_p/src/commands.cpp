#include "versa_p_internal.h"

namespace {

bool valid_api_idle(versa_p_device *dev, versa_p_api api)
{
    return dev && !dev->api_inflight[(int)api];
}

bool is_multiple_of(uint32_t value, uint32_t alignment)
{
    return alignment != 0 && (value % alignment) == 0;
}

bool mvin_a_conflicts(versa_p_device *dev, uint8_t a_bank)
{
    return dev->api_inflight[VERSA_P_API_GEMM_I8] &&
           dev->gemm_inflight_a_bank == a_bank;
}

bool gemm_conflicts(versa_p_device *dev, uint8_t a_bank, uint8_t o_bank)
{
    return (dev->api_inflight[VERSA_P_API_MVIN_A] &&
            dev->mvin_a_inflight_bank == a_bank) ||
           dev->api_inflight[VERSA_P_API_MVIN_META] ||
           (dev->api_inflight[VERSA_P_API_MVOUT] &&
            dev->mvout_inflight_o_bank == o_bank);
}

bool mvout_conflicts(versa_p_device *dev, uint8_t o_bank)
{
    return dev->api_inflight[VERSA_P_API_MVIN_META] ||
           (dev->api_inflight[VERSA_P_API_GEMM_I8] &&
            dev->gemm_inflight_o_bank == o_bank);
}

bool meta_conflicts(versa_p_device *dev)
{
    return dev->api_inflight[VERSA_P_API_GEMM_I8] ||
           dev->api_inflight[VERSA_P_API_MVOUT];
}

int validate_mvin_a(const versa_p_mvin_a_desc *desc)
{
    if (!desc || desc->m == 0 || desc->k == 0 || desc->k > VERSA_P_K_MAX ||
        desc->a_bank > 1 ||
        desc->dram_row_stride_bytes < desc->k ||
        !is_multiple_of(desc->m, VERSA_P_HW_ALIGN_ELEMS) ||
        !is_multiple_of(desc->k, VERSA_P_HW_ALIGN_ELEMS) ||
        !is_multiple_of(desc->dram_row_stride_bytes, VERSA_P_HW_ALIGN_ELEMS)) {
        return VERSA_P_ERR_ILLEGAL_SHAPE;
    }
    if (!versa_p_is_aligned(desc->dram_base, VERSA_P_DMA_ALIGN_BYTES)) {
        return VERSA_P_ERR_ALIGNMENT;
    }
    return VERSA_P_OK;
}

int validate_mvin_w(versa_p_device *dev, const versa_p_mvin_w_desc *desc)
{
    if (!desc || desc->k == 0 || desc->n == 0 || desc->k > VERSA_P_K_MAX ||
        desc->w_bank > 1 ||
        !is_multiple_of(desc->k, VERSA_P_HW_ALIGN_ELEMS) ||
        !is_multiple_of(desc->n, VERSA_P_HW_ALIGN_ELEMS)) {
        return VERSA_P_ERR_ILLEGAL_SHAPE;
    }
    if (!versa_p_is_aligned(desc->dram_base, VERSA_P_DMA_ALIGN_BYTES)) {
        return VERSA_P_ERR_ALIGNMENT;
    }
    if (desc->w_bank == dev->active_w_bank) {
        return VERSA_P_ERR_BANK_CONFLICT;
    }
    return VERSA_P_OK;
}

int validate_mvin_meta(const versa_p_mvin_meta_desc *desc)
{
    if (!desc || desc->byte_count == 0 || desc->meta_type > 3 ||
        !is_multiple_of(desc->byte_count, VERSA_P_DMA_ALIGN_BYTES)) {
        return VERSA_P_ERR_ILLEGAL_SHAPE;
    }
    if (!versa_p_is_aligned(desc->dram_base, VERSA_P_DMA_ALIGN_BYTES) ||
        !versa_p_is_aligned(desc->meta_offset_bytes, VERSA_P_DMA_ALIGN_BYTES)) {
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
        desc->k > VERSA_P_K_MAX || desc->a_bank > 1 ||
        desc->w_bank > 1 || desc->o_bank > 1 ||
        !is_multiple_of(desc->m, VERSA_P_HW_ALIGN_ELEMS) ||
        !is_multiple_of(desc->n, VERSA_P_HW_ALIGN_ELEMS) ||
        !is_multiple_of(desc->k, VERSA_P_HW_ALIGN_ELEMS)) {
        return VERSA_P_ERR_ILLEGAL_SHAPE;
    }
    if (desc->accumulate_en && desc->add_bias_en) {
        return VERSA_P_ERR_ILLEGAL_FLAGS;
    }
    const VersaPBankState &bank = dev->w_bank[desc->w_bank];
    if (!bank.loaded) {
        return VERSA_P_ERR_BANK_NOT_VALID;
    }
    const VersaPABankState &a_bank = dev->a_bank[desc->a_bank];
    if (!a_bank.loaded) {
        return VERSA_P_ERR_BANK_NOT_VALID;
    }
    if (a_bank.m != desc->m || a_bank.k != desc->k) {
        return VERSA_P_ERR_SHAPE_MISMATCH;
    }
    if (bank.k != desc->k || bank.n != desc->n) {
        return VERSA_P_ERR_SHAPE_MISMATCH;
    }
    if (desc->add_bias_en &&
        !versa_p_is_aligned(desc->bias_offset_bytes, VERSA_P_ALIGN_BYTES)) {
        return VERSA_P_ERR_ALIGNMENT;
    }
    return VERSA_P_OK;
}

int validate_attention_qk(versa_p_device *dev,
                          const versa_p_attention_qk_desc *desc)
{
    if (!desc || desc->token_count < 32 || desc->token_count > 1024 ||
        desc->q_rows == 0 || desc->q_bank > 1 || desc->k_bank > 1 ||
        desc->o_bank > 1 || desc->gamma16_fix == 0 ||
        !is_multiple_of(desc->token_count, VERSA_P_HW_ALIGN_ELEMS) ||
        !is_multiple_of(desc->q_row_start, VERSA_P_HW_ALIGN_ELEMS) ||
        !is_multiple_of(desc->q_rows, VERSA_P_HW_ALIGN_ELEMS) ||
        (uint32_t)desc->q_row_start + desc->q_rows > desc->token_count) {
        return VERSA_P_ERR_ILLEGAL_SHAPE;
    }
    const VersaPABankState &q_bank = dev->a_bank[desc->q_bank];
    if (!q_bank.loaded) {
        return VERSA_P_ERR_BANK_NOT_VALID;
    }
    if (q_bank.k != 64 ||
        q_bank.m < (uint32_t)desc->q_row_start + desc->q_rows) {
        return VERSA_P_ERR_SHAPE_MISMATCH;
    }
    const VersaPBankState &k_bank = dev->w_bank[desc->k_bank];
    if (!k_bank.loaded) {
        return VERSA_P_ERR_BANK_NOT_VALID;
    }
    if (k_bank.k != 64 || k_bank.n < desc->token_count) {
        return VERSA_P_ERR_SHAPE_MISMATCH;
    }
    return VERSA_P_OK;
}

int validate_pv_log8(versa_p_device *dev, const versa_p_pv_log8_desc *desc)
{
    if (!desc || desc->m == 0 || desc->n == 0 || desc->k == 0 ||
        desc->k > VERSA_P_K_MAX || desc->p_bank > 1 ||
        desc->v_bank > 1 || desc->o_bank > 1 ||
        !is_multiple_of(desc->m, VERSA_P_HW_ALIGN_ELEMS) ||
        !is_multiple_of(desc->n, VERSA_P_HW_ALIGN_ELEMS) ||
        !is_multiple_of(desc->k, VERSA_P_HW_ALIGN_ELEMS)) {
        return VERSA_P_ERR_ILLEGAL_SHAPE;
    }
    const VersaPABankState &p_bank = dev->a_bank[desc->p_bank];
    if (!p_bank.loaded) {
        return VERSA_P_ERR_BANK_NOT_VALID;
    }
    if (p_bank.m != desc->m || p_bank.k != desc->k) {
        return VERSA_P_ERR_SHAPE_MISMATCH;
    }
    const VersaPBankState &v_bank = dev->w_bank[desc->v_bank];
    if (!v_bank.loaded) {
        return VERSA_P_ERR_BANK_NOT_VALID;
    }
    if (v_bank.k != desc->k || v_bank.n != desc->n) {
        return VERSA_P_ERR_SHAPE_MISMATCH;
    }
    return VERSA_P_OK;
}

int validate_mvout(const versa_p_mvout_desc *desc)
{
    if (!desc || desc->m == 0 || desc->n == 0 || desc->o_bank > 1 ||
        (desc->output_stride_n != 0 && desc->output_stride_n < desc->n) ||
        !is_multiple_of(desc->m, VERSA_P_HW_ALIGN_ELEMS) ||
        !is_multiple_of(desc->n, VERSA_P_HW_ALIGN_ELEMS) ||
        (desc->output_stride_n != 0 &&
         !is_multiple_of(desc->output_stride_n, VERSA_P_HW_ALIGN_ELEMS))) {
        return VERSA_P_ERR_ILLEGAL_SHAPE;
    }
    if (!versa_p_is_aligned(desc->dram_base, VERSA_P_DMA_ALIGN_BYTES)) {
        return VERSA_P_ERR_ALIGNMENT;
    }
    if (desc->mode > VERSA_P_MVOUT_ATTENTION_QK_LOGP) {
        return VERSA_P_ERR_UNSUPPORTED_MODE;
    }
    if (desc->mode == VERSA_P_MVOUT_FP32_PER_CHANNEL_Q8_24 &&
        !versa_p_is_aligned(desc->scale_param, VERSA_P_DMA_ALIGN_BYTES)) {
        return VERSA_P_ERR_ALIGNMENT;
    }
    return VERSA_P_OK;
}

int validate_attention_logp_mvout(const versa_p_attention_logp_mvout_desc *desc)
{
    if (!desc || desc->token_count < 32 || desc->token_count > 1024 ||
        desc->q_rows == 0 || desc->o_bank > 1 ||
        !is_multiple_of(desc->token_count, VERSA_P_HW_ALIGN_ELEMS) ||
        !is_multiple_of(desc->q_row_start, VERSA_P_HW_ALIGN_ELEMS) ||
        !is_multiple_of(desc->q_rows, VERSA_P_HW_ALIGN_ELEMS) ||
        (uint32_t)desc->q_row_start + desc->q_rows > desc->token_count ||
        (desc->output_stride_bytes != 0 &&
         desc->output_stride_bytes < desc->token_count)) {
        return VERSA_P_ERR_ILLEGAL_SHAPE;
    }
    if (!versa_p_is_aligned(desc->dram_base, VERSA_P_DMA_ALIGN_BYTES)) {
        return VERSA_P_ERR_ALIGNMENT;
    }
    if (desc->output_stride_bytes != 0 &&
        !is_multiple_of(desc->output_stride_bytes, VERSA_P_DMA_ALIGN_BYTES)) {
        return VERSA_P_ERR_ILLEGAL_SHAPE;
    }
    return VERSA_P_OK;
}

void mark_started(versa_p_device *dev, versa_p_api api)
{
    dev->api_inflight[(int)api] = true;
}

size_t strided_span_bytes(uint32_t rows, uint32_t row_bytes, uint32_t stride_bytes)
{
    if (rows == 0 || row_bytes == 0) {
        return 0;
    }
    return (size_t)(rows - 1u) * stride_bytes + row_bytes;
}

size_t packed_w_span_bytes(uint32_t k, uint32_t n)
{
    const uint32_t groups = (n + 31u) / 32u;
    return (size_t)k * groups * 32u;
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
    int rc = validate_mvin_a(desc);
    if (rc != VERSA_P_OK) {
        return rc;
    }
    if (mvin_a_conflicts(dev, desc->a_bank)) {
        return VERSA_P_ERR_RESOURCE_CONFLICT;
    }
    rc = versa_p_sync_dma_range(
        dev, desc->dram_base,
        strided_span_bytes(desc->m, desc->k, desc->dram_row_stride_bytes),
        NPU_KV260_SYNC_FOR_DEVICE,
        NPU_KV260_SYNC_TO_DEVICE);
    if (rc != VERSA_P_OK) {
        return rc;
    }

    uint64_t desc0 = (uint64_t)desc->dram_base |
                     ((uint64_t)desc->dram_row_stride_bytes << 32);
    uint64_t desc1 = (uint64_t)desc->m |
                     ((uint64_t)desc->k << 16) |
                     ((uint64_t)(desc->u8_minus_128 != 0) << 32) |
                     ((uint64_t)(desc->a_bank & 1u) << VERSA_P_MVIN_A_BANK_BIT) |
                     (1ull << VERSA_P_START_MVIN_A_BIT);
    versa_p_write64(dev, VERSA_P_REG_MVIN_A_DESC0, desc0);
    versa_p_write64(dev, VERSA_P_REG_MVIN_A_DESC1, desc1);
    dev->a_bank[desc->a_bank].loaded = false;
    dev->pending_mvin_a_bank = desc->a_bank;
    dev->mvin_a_inflight_bank = desc->a_bank;
    dev->pending_mvin_a_m = desc->m;
    dev->pending_mvin_a_k = desc->k;
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
    rc = versa_p_sync_dma_range(
        dev, desc->dram_base,
        packed_w_span_bytes(desc->k, desc->n),
        NPU_KV260_SYNC_FOR_DEVICE,
        NPU_KV260_SYNC_TO_DEVICE);
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
    if (meta_conflicts(dev)) {
        return VERSA_P_ERR_RESOURCE_CONFLICT;
    }
    int rc = validate_mvin_meta(desc);
    if (rc != VERSA_P_OK) {
        return rc;
    }
    rc = versa_p_sync_dma_range(
        dev, desc->dram_base, desc->byte_count,
        NPU_KV260_SYNC_FOR_DEVICE,
        NPU_KV260_SYNC_TO_DEVICE);
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
    int rc = validate_gemm_i8(dev, desc);
    if (rc != VERSA_P_OK) {
        return rc;
    }
    if (gemm_conflicts(dev, desc->a_bank, desc->o_bank)) {
        return VERSA_P_ERR_RESOURCE_CONFLICT;
    }

    uint64_t desc0 = (uint64_t)desc->m |
                     ((uint64_t)desc->n << 16) |
                     ((uint64_t)desc->k << 32) |
                     ((uint64_t)(desc->w_bank & 1u) << 48) |
                     ((uint64_t)(desc->accumulate_en != 0) << 49) |
                     ((uint64_t)(desc->add_bias_en != 0) << 50) |
                     ((uint64_t)(desc->a_bank & 1u) << VERSA_P_GEMM_A_BANK_BIT) |
                     ((uint64_t)(desc->o_bank & 1u) << 53) |
                     (1ull << VERSA_P_START_GEMM_BIT);
    versa_p_write64(dev, VERSA_P_REG_GEMM_DESC1, desc->bias_offset_bytes);
    versa_p_write64(dev, VERSA_P_REG_GEMM_DESC0, desc0);
    dev->active_a_bank = desc->a_bank;
    dev->active_w_bank = desc->w_bank;
    dev->gemm_inflight_a_bank = desc->a_bank;
    dev->gemm_inflight_o_bank = desc->o_bank;
    mark_started(dev, VERSA_P_API_GEMM_I8);
    return VERSA_P_OK;
}

int versa_p_start_attention_qk_logp(versa_p_device *dev,
                                    const versa_p_attention_qk_desc *desc)
{
    if (!dev) {
        return VERSA_P_ERR_INVAL;
    }
    if (!valid_api_idle(dev, VERSA_P_API_GEMM_I8)) {
        return VERSA_P_ERR_BUSY;
    }
    int rc = validate_attention_qk(dev, desc);
    if (rc != VERSA_P_OK) {
        return rc;
    }
    if (gemm_conflicts(dev, desc->q_bank, desc->o_bank) ||
        (dev->api_inflight[VERSA_P_API_MVIN_W] &&
         dev->pending_mvin_w_bank == desc->k_bank)) {
        return VERSA_P_ERR_RESOURCE_CONFLICT;
    }

    uint64_t desc0 = (uint64_t)desc->q_rows |
                     ((uint64_t)desc->token_count << 16) |
                     (64ull << 32) |
                     ((uint64_t)(desc->k_bank & 1u) << 48) |
                     ((uint64_t)(desc->causal_mask != 0) << 50) |
                     ((uint64_t)(desc->q_bank & 1u) << VERSA_P_GEMM_A_BANK_BIT) |
                     ((uint64_t)(desc->o_bank & 1u) << 53) |
                     ((uint64_t)VERSA_P_GEMM_MODE_ATTENTION_QK << 54) |
                     (1ull << VERSA_P_START_GEMM_BIT);
    uint64_t desc1 = (uint64_t)desc->gamma16_fix |
                     ((uint64_t)desc->q_row_start << 32);
    versa_p_write64(dev, VERSA_P_REG_GEMM_DESC1, desc1);
    versa_p_write64(dev, VERSA_P_REG_GEMM_DESC0, desc0);
    dev->active_a_bank = desc->q_bank;
    dev->gemm_inflight_a_bank = desc->q_bank;
    dev->gemm_inflight_o_bank = desc->o_bank;
    mark_started(dev, VERSA_P_API_GEMM_I8);
    return VERSA_P_OK;
}

int versa_p_start_gemm_pv_log8(versa_p_device *dev,
                               const versa_p_pv_log8_desc *desc)
{
    if (!dev) {
        return VERSA_P_ERR_INVAL;
    }
    if (!valid_api_idle(dev, VERSA_P_API_GEMM_I8)) {
        return VERSA_P_ERR_BUSY;
    }
    int rc = validate_pv_log8(dev, desc);
    if (rc != VERSA_P_OK) {
        return rc;
    }
    if (gemm_conflicts(dev, desc->p_bank, desc->o_bank)) {
        return VERSA_P_ERR_RESOURCE_CONFLICT;
    }

    uint64_t desc0 = (uint64_t)desc->m |
                     ((uint64_t)desc->n << 16) |
                     ((uint64_t)desc->k << 32) |
                     ((uint64_t)(desc->v_bank & 1u) << 48) |
                     ((uint64_t)(desc->p_bank & 1u) << VERSA_P_GEMM_A_BANK_BIT) |
                     ((uint64_t)(desc->o_bank & 1u) << 53) |
                     ((uint64_t)VERSA_P_GEMM_MODE_PV_LOG8_U16I8 << 54) |
                     (1ull << VERSA_P_START_GEMM_BIT);
    versa_p_write64(dev, VERSA_P_REG_GEMM_DESC1, 0);
    versa_p_write64(dev, VERSA_P_REG_GEMM_DESC0, desc0);
    dev->active_a_bank = desc->p_bank;
    dev->active_w_bank = desc->v_bank;
    dev->gemm_inflight_a_bank = desc->p_bank;
    dev->gemm_inflight_o_bank = desc->o_bank;
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
    int rc = validate_mvout(desc);
    if (rc != VERSA_P_OK) {
        return rc;
    }
    if (mvout_conflicts(dev, desc->o_bank)) {
        return VERSA_P_ERR_RESOURCE_CONFLICT;
    }

    uint64_t desc0 = (uint64_t)desc->dram_base |
                     ((uint64_t)desc->scale_param << 32);
    const uint32_t output_stride_n = desc->output_stride_n ?
                                     desc->output_stride_n : desc->n;
    uint64_t desc1 = (uint64_t)desc->m |
                     ((uint64_t)desc->n << 16) |
                     ((uint64_t)output_stride_n << 32) |
                     ((uint64_t)(desc->mode & 3u) << 48) |
                     ((uint64_t)(desc->o_bank & 1u) << 51) |
                     (1ull << VERSA_P_START_MVOUT_BIT);
    versa_p_write64(dev, VERSA_P_REG_MVOUT_DESC0, desc0);
    versa_p_write64(dev, VERSA_P_REG_MVOUT_DESC1, desc1);
    dev->pending_mvout_dma_addr = desc->dram_base;
    if (desc->mode == VERSA_P_MVOUT_ATTENTION_QK_LOGP) {
        dev->pending_mvout_bytes =
            (uint32_t)strided_span_bytes(desc->m, desc->n, output_stride_n);
    } else {
        dev->pending_mvout_bytes =
            (uint32_t)strided_span_bytes(desc->m, desc->n * sizeof(uint32_t),
                                         output_stride_n * sizeof(uint32_t));
    }
    dev->mvout_inflight_o_bank = desc->o_bank;
    mark_started(dev, VERSA_P_API_MVOUT);
    return VERSA_P_OK;
}

int versa_p_start_mvout_attention_logp(
    versa_p_device *dev, const versa_p_attention_logp_mvout_desc *desc)
{
    if (!dev) {
        return VERSA_P_ERR_INVAL;
    }
    if (!valid_api_idle(dev, VERSA_P_API_MVOUT)) {
        return VERSA_P_ERR_BUSY;
    }
    int rc = validate_attention_logp_mvout(desc);
    if (rc != VERSA_P_OK) {
        return rc;
    }
    if (mvout_conflicts(dev, desc->o_bank)) {
        return VERSA_P_ERR_RESOURCE_CONFLICT;
    }

    const uint16_t stride = desc->output_stride_bytes ?
                            desc->output_stride_bytes : desc->token_count;
    uint64_t desc0 = (uint64_t)desc->dram_base |
                     ((uint64_t)(desc->q_row_start |
                                 ((uint32_t)(desc->causal_mask != 0) << 16)) << 32);
    uint64_t desc1 = (uint64_t)desc->q_rows |
                     ((uint64_t)desc->token_count << 16) |
                     ((uint64_t)stride << 32) |
                     ((uint64_t)VERSA_P_MVOUT_ATTENTION_QK_LOGP << 48) |
                     ((uint64_t)(desc->o_bank & 1u) << 51) |
                     (1ull << VERSA_P_START_MVOUT_BIT);
    versa_p_write64(dev, VERSA_P_REG_MVOUT_DESC0, desc0);
    versa_p_write64(dev, VERSA_P_REG_MVOUT_DESC1, desc1);
    dev->pending_mvout_dma_addr = desc->dram_base;
    dev->pending_mvout_bytes =
        (uint32_t)strided_span_bytes(desc->q_rows, desc->token_count, stride);
    dev->mvout_inflight_o_bank = desc->o_bank;
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

int versa_p_attention_qk_logp(versa_p_device *dev,
                              const versa_p_attention_qk_desc *desc,
                              uint32_t timeout_ms)
{
    int rc = versa_p_start_attention_qk_logp(dev, desc);
    return rc == VERSA_P_OK ? versa_p_wait(dev, VERSA_P_API_GEMM_I8, timeout_ms) : rc;
}

int versa_p_gemm_pv_log8(versa_p_device *dev,
                         const versa_p_pv_log8_desc *desc,
                         uint32_t timeout_ms)
{
    int rc = versa_p_start_gemm_pv_log8(dev, desc);
    return rc == VERSA_P_OK ? versa_p_wait(dev, VERSA_P_API_GEMM_I8, timeout_ms) : rc;
}

int versa_p_mvout(versa_p_device *dev, const versa_p_mvout_desc *desc,
                  uint32_t timeout_ms)
{
    int rc = versa_p_start_mvout(dev, desc);
    return rc == VERSA_P_OK ? versa_p_wait(dev, VERSA_P_API_MVOUT, timeout_ms) : rc;
}

int versa_p_mvout_attention_logp(
    versa_p_device *dev, const versa_p_attention_logp_mvout_desc *desc,
    uint32_t timeout_ms)
{
    int rc = versa_p_start_mvout_attention_logp(dev, desc);
    return rc == VERSA_P_OK ? versa_p_wait(dev, VERSA_P_API_MVOUT, timeout_ms) : rc;
}
