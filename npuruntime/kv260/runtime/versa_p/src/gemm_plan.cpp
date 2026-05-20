#include "versa_p_internal.h"

#include <algorithm>
#include <cstring>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace {

constexpr uint32_t kBiasMetaOffset = 0;
constexpr uint32_t kScaleMetaOffsetWithBias = 2048;

uint32_t min_u32(uint32_t a, uint32_t b)
{
    return a < b ? a : b;
}

uint32_t packed_w_bytes(uint32_t k, uint32_t n)
{
    uint32_t n_groups = (n + 31u) / 32u;
    return k * n_groups * 32u;
}

void copy_32b_lane(const int8_t *src, int8_t *dst)
{
#if defined(__aarch64__)
    int8x16_t lo = vld1q_s8(src);
    int8x16_t hi = vld1q_s8(src + 16);
    vst1q_s8(dst, lo);
    vst1q_s8(dst + 16, hi);
#else
    const uint64_t *s64 = (const uint64_t *)src;
    uint64_t *d64 = (uint64_t *)dst;
    d64[0] = s64[0];
    d64[1] = s64[1];
    d64[2] = s64[2];
    d64[3] = s64[3];
#endif
}

void copy_bytes_vector(const void *src_void, void *dst_void, size_t bytes)
{
#if defined(__aarch64__)
    const uint8_t *src = (const uint8_t *)src_void;
    uint8_t *dst = (uint8_t *)dst_void;
    while (bytes >= 64u) {
        __builtin_prefetch(src + 256, 0, 0);
        uint8x16_t v0 = vld1q_u8(src);
        uint8x16_t v1 = vld1q_u8(src + 16);
        uint8x16_t v2 = vld1q_u8(src + 32);
        uint8x16_t v3 = vld1q_u8(src + 48);
        vst1q_u8(dst, v0);
        vst1q_u8(dst + 16, v1);
        vst1q_u8(dst + 32, v2);
        vst1q_u8(dst + 48, v3);
        src += 64;
        dst += 64;
        bytes -= 64;
    }
    while (bytes >= 16u) {
        uint8x16_t v = vld1q_u8(src);
        vst1q_u8(dst, v);
        src += 16;
        dst += 16;
        bytes -= 16;
    }
    if (bytes != 0) {
        std::memcpy(dst, src, bytes);
    }
#else
    std::memcpy(dst_void, src_void, bytes);
#endif
}

void pack_w_tile(const int8_t *src, uint32_t src_stride_n, uint32_t k0,
                 uint32_t n0, uint32_t k_count, uint32_t n_count,
                 int8_t *dst)
{
    const uint32_t n_groups = (n_count + 31u) / 32u;
    std::memset(dst, 0, packed_w_bytes(k_count, n_count));
    for (uint32_t group = 0; group < n_groups; ++group) {
        uint32_t group_cols = min_u32(32u, n_count - group * 32u);
        for (uint32_t kk = 0; kk < k_count; ++kk) {
            int8_t *d = dst + ((size_t)group * k_count + kk) * 32u;
            const int8_t *s = src + (size_t)(k0 + kk) * src_stride_n +
                              n0 + group * 32u;
            if (group_cols == 32u) {
                copy_32b_lane(s, d);
            } else {
                std::memcpy(d, s, group_cols);
            }
        }
    }
}

void copy_a_tile(const int8_t *src, uint32_t stride_bytes, uint32_t m0,
                 uint32_t k0, uint32_t m_count, uint32_t k_count,
                 int8_t *dst)
{
    for (uint32_t mm = 0; mm < m_count; ++mm) {
        std::memcpy(dst + mm * k_count,
                    src + (m0 + mm) * stride_bytes + k0,
                    k_count);
    }
}

void copy_output_tile_32b(const void *src_void, uint32_t src_stride_n,
                          void *dst_void, uint32_t dst_stride_n,
                          uint32_t m_count, uint32_t n_count)
{
    const uint32_t *src = (const uint32_t *)src_void;
    uint32_t *dst = (uint32_t *)dst_void;
    if (src_stride_n == n_count && dst_stride_n == n_count) {
        copy_bytes_vector(src, dst, (size_t)m_count * n_count * sizeof(uint32_t));
        return;
    }
    for (uint32_t mm = 0; mm < m_count; ++mm) {
        const uint32_t *s = src + (size_t)mm * src_stride_n;
        uint32_t *d = dst + (size_t)mm * dst_stride_n;
        copy_bytes_vector(s, d, (size_t)n_count * sizeof(uint32_t));
    }
}

int load_metadata_if_needed(versa_p_device *dev, const versa_p_gemm_plan *plan,
                            uint32_t n0, uint32_t n_count,
                            versa_p_buffer *meta_buf,
                            uint32_t *scale_meta_offset)
{
    const uint32_t timeout_ms = plan->timeout_ms ? plan->timeout_ms : 10000u;
    uint32_t bytes = 0;
    if (plan->bias_i32) {
        bytes += versa_p_align_up(n_count * sizeof(int32_t), VERSA_P_ALIGN_BYTES);
    }
    if (plan->mvout_mode == VERSA_P_MVOUT_FP32_PER_CHANNEL_Q8_24) {
        bytes += versa_p_align_up(n_count * sizeof(int32_t), VERSA_P_ALIGN_BYTES);
    }
    if (bytes == 0) {
        return VERSA_P_OK;
    }

    int rc = versa_p_mem_alloc(dev, bytes, VERSA_P_ALIGN_BYTES, meta_buf);
    if (rc != VERSA_P_OK) {
        return rc;
    }
    uint8_t *cursor = (uint8_t *)meta_buf->vaddr;
    if (plan->bias_i32) {
        std::memcpy(cursor, plan->bias_i32 + n0, n_count * sizeof(int32_t));
        uint32_t bias_bytes = versa_p_align_up(n_count * sizeof(int32_t),
                                               VERSA_P_ALIGN_BYTES);
        versa_p_mvin_meta_desc desc = {};
        desc.dram_base = meta_buf->dma_addr;
        desc.meta_offset_bytes = kBiasMetaOffset;
        desc.byte_count = n_count * (uint32_t)sizeof(int32_t);
        desc.meta_type = VERSA_P_META_BIAS;
        rc = versa_p_mvin_meta(dev, &desc, timeout_ms);
        if (rc != VERSA_P_OK) {
            return rc;
        }
        cursor += bias_bytes;
    }
    if (plan->mvout_mode == VERSA_P_MVOUT_FP32_PER_CHANNEL_Q8_24) {
        if (!plan->scale_q8_24) {
            return VERSA_P_ERR_INVAL;
        }
        const uint32_t scale_offset = plan->bias_i32 ? kScaleMetaOffsetWithBias : 0;
        if (scale_meta_offset) {
            *scale_meta_offset = scale_offset;
        }
        std::memcpy(cursor, plan->scale_q8_24 + n0, n_count * sizeof(int32_t));
        versa_p_mvin_meta_desc desc = {};
        desc.dram_base = meta_buf->dma_addr +
                         (uint32_t)(cursor - (uint8_t *)meta_buf->vaddr);
        desc.meta_offset_bytes = scale_offset;
        desc.byte_count = n_count * (uint32_t)sizeof(int32_t);
        desc.meta_type = VERSA_P_META_SCALE;
        rc = versa_p_mvin_meta(dev, &desc, timeout_ms);
    }
    return rc;
}

int validate_plan(const versa_p_gemm_plan *plan)
{
    if (!plan || !plan->a || !plan->w || !plan->output ||
        plan->m == 0 || plan->n == 0 || plan->k == 0) {
        return VERSA_P_ERR_INVAL;
    }
    if (plan->a_stride_bytes < plan->k || plan->w_stride_n < plan->n) {
        return VERSA_P_ERR_ILLEGAL_SHAPE;
    }
    if (plan->output_stride_n != 0 && plan->output_stride_n < plan->n) {
        return VERSA_P_ERR_ILLEGAL_SHAPE;
    }
    if (plan->mvout_mode > VERSA_P_MVOUT_FP32_PER_CHANNEL_Q8_24) {
        return VERSA_P_ERR_UNSUPPORTED_MODE;
    }
    if (plan->mvout_mode == VERSA_P_MVOUT_FP32_PER_CHANNEL_Q8_24 &&
        !plan->scale_q8_24) {
        return VERSA_P_ERR_INVAL;
    }
    return VERSA_P_OK;
}

} // namespace

int versa_p_gemm_plan_run(versa_p_device *dev, const versa_p_gemm_plan *plan)
{
    if (!dev) {
        return VERSA_P_ERR_INVAL;
    }
    int rc = validate_plan(plan);
    if (rc != VERSA_P_OK) {
        return rc;
    }

    const uint32_t timeout_ms = plan->timeout_ms ? plan->timeout_ms : 10000u;
    const uint32_t output_stride_n = plan->output_stride_n ?
                                     plan->output_stride_n : plan->n;
    uint8_t next_w_bank = 0;
    versa_p_buffer full_out_buf = {};
    uint32_t full_out_bytes =
        versa_p_align_up((uint32_t)((size_t)plan->m * output_stride_n *
                                    sizeof(uint32_t)),
                         VERSA_P_ALIGN_BYTES);
    rc = versa_p_mem_alloc(dev, full_out_bytes, VERSA_P_ALIGN_BYTES,
                           &full_out_buf);
    if (rc != VERSA_P_OK) {
        return rc;
    }

    for (uint32_t m0 = 0; m0 < plan->m; m0 += VERSA_P_M_TILE) {
        uint32_t m_count = min_u32(VERSA_P_M_TILE, plan->m - m0);
        for (uint32_t n0 = 0; n0 < plan->n; n0 += VERSA_P_N_TILE) {
            uint32_t n_count = min_u32(VERSA_P_N_TILE, plan->n - n0);
            versa_p_buffer meta_buf = {};
            uint32_t scale_meta_offset = 0;
            rc = load_metadata_if_needed(dev, plan, n0, n_count, &meta_buf,
                                         &scale_meta_offset);
            if (rc != VERSA_P_OK) {
                versa_p_mem_free(dev, &meta_buf);
                versa_p_mem_free(dev, &full_out_buf);
                return rc;
            }

            for (uint32_t k0 = 0; k0 < plan->k; k0 += VERSA_P_K_TILE) {
                uint32_t k_count = min_u32(VERSA_P_K_TILE, plan->k - k0);
                versa_p_buffer a_buf = {};
                versa_p_buffer w_buf = {};
                uint32_t a_bytes = versa_p_align_up(m_count * k_count,
                                                    VERSA_P_ALIGN_BYTES);
                uint32_t w_bytes = versa_p_align_up(packed_w_bytes(k_count, n_count),
                                                    VERSA_P_ALIGN_BYTES);
                rc = versa_p_mem_alloc(dev, a_bytes, VERSA_P_ALIGN_BYTES, &a_buf);
                if (rc == VERSA_P_OK) {
                    rc = versa_p_mem_alloc(dev, w_bytes, VERSA_P_ALIGN_BYTES, &w_buf);
                }
                if (rc != VERSA_P_OK) {
                    versa_p_mem_free(dev, &a_buf);
                    versa_p_mem_free(dev, &w_buf);
                    versa_p_mem_free(dev, &meta_buf);
                    versa_p_mem_free(dev, &full_out_buf);
                    return rc;
                }

                copy_a_tile(plan->a, plan->a_stride_bytes, m0, k0,
                            m_count, k_count, (int8_t *)a_buf.vaddr);
                pack_w_tile(plan->w, plan->w_stride_n, k0, n0,
                            k_count, n_count, (int8_t *)w_buf.vaddr);

                versa_p_mvin_a_desc a_desc = {};
                a_desc.dram_base = a_buf.dma_addr;
                a_desc.dram_row_stride_bytes = k_count;
                a_desc.m = (uint16_t)m_count;
                a_desc.k = (uint16_t)k_count;
                rc = versa_p_mvin_a(dev, &a_desc, timeout_ms);
                if (rc == VERSA_P_OK) {
                    uint8_t bank = next_w_bank;
                    if (bank == dev->active_w_bank) {
                        bank ^= 1u;
                    }
                    versa_p_mvin_w_desc w_desc = {};
                    w_desc.dram_base = w_buf.dma_addr;
                    w_desc.k = (uint16_t)k_count;
                    w_desc.n = (uint16_t)n_count;
                    w_desc.w_bank = bank;
                    rc = versa_p_mvin_w(dev, &w_desc, timeout_ms);
                    if (rc == VERSA_P_OK) {
                        versa_p_gemm_i8_desc gemm = {};
                        gemm.m = (uint16_t)m_count;
                        gemm.n = (uint16_t)n_count;
                        gemm.k = (uint16_t)k_count;
                        gemm.w_bank = bank;
                        gemm.accumulate_en = (k0 != 0);
                        gemm.add_bias_en = (k0 == 0 && plan->bias_i32 != nullptr);
                        gemm.bias_offset_bytes = kBiasMetaOffset;
                        rc = versa_p_gemm_i8(dev, &gemm, timeout_ms);
                    }
                    next_w_bank = bank ^ 1u;
                }

                versa_p_mem_free(dev, &a_buf);
                versa_p_mem_free(dev, &w_buf);
                if (rc != VERSA_P_OK) {
                    versa_p_mem_free(dev, &meta_buf);
                    versa_p_mem_free(dev, &full_out_buf);
                    return rc;
                }
            }

            uint32_t scale_param = plan->tensor_scale_q8_24;
            if (plan->mvout_mode == VERSA_P_MVOUT_FP32_PER_CHANNEL_Q8_24) {
                scale_param = scale_meta_offset;
            }
            versa_p_mvout_desc mvout = {};
            mvout.dram_base = full_out_buf.dma_addr +
                              (uint32_t)(((size_t)m0 * output_stride_n + n0) *
                                         sizeof(uint32_t));
            mvout.scale_param = scale_param;
            mvout.m = (uint16_t)m_count;
            mvout.n = (uint16_t)n_count;
            mvout.output_stride_n = (uint16_t)output_stride_n;
            mvout.mode = plan->mvout_mode;
            rc = versa_p_mvout(dev, &mvout, timeout_ms);
            versa_p_mem_free(dev, &meta_buf);
            if (rc != VERSA_P_OK) {
                versa_p_mem_free(dev, &full_out_buf);
                return rc;
            }
        }
    }

    copy_output_tile_32b(full_out_buf.vaddr, output_stride_n, plan->output,
                         output_stride_n, plan->m, plan->n);
    versa_p_mem_free(dev, &full_out_buf);
    return VERSA_P_OK;
}
