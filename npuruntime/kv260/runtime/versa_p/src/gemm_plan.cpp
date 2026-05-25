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

uint32_t align_elems_32(uint32_t value)
{
    return versa_p_align_up(value, VERSA_P_HW_ALIGN_ELEMS);
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
                 uint32_t n0, uint32_t k_real, uint32_t n_real,
                 uint32_t k_hw, uint32_t n_hw,
                 int8_t *dst)
{
    const uint32_t n_groups = n_hw / 32u;
    std::memset(dst, 0, packed_w_bytes(k_hw, n_hw));
    for (uint32_t group = 0; group < n_groups; ++group) {
        const uint32_t group_col_base = group * 32u;
        uint32_t group_cols = 0;
        if (group_col_base < n_real) {
            group_cols = min_u32(32u, n_real - group_col_base);
        }
        for (uint32_t kk = 0; kk < k_hw; ++kk) {
            int8_t *d = dst + ((size_t)group * k_hw + kk) * 32u;
            if (kk >= k_real || group_cols == 0) {
                continue;
            }
            const int8_t *s = src + (size_t)(k0 + kk) * src_stride_n +
                              n0 + group_col_base;
            if (group_cols == 32u) {
                copy_32b_lane(s, d);
            } else {
                std::memcpy(d, s, group_cols);
            }
        }
    }
}

void copy_a_tile(const int8_t *src, uint32_t stride_bytes, uint32_t m0,
                 uint32_t k0, uint32_t m_real, uint32_t k_real,
                 uint32_t m_hw, uint32_t k_hw,
                 int8_t *dst)
{
    std::memset(dst, 0, (size_t)m_hw * k_hw);
    for (uint32_t mm = 0; mm < m_real; ++mm) {
        std::memcpy(dst + mm * k_hw,
                    src + (m0 + mm) * stride_bytes + k0,
                    k_real);
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
                            uint32_t n0, uint32_t n_real, uint32_t n_hw,
                            versa_p_buffer *meta_buf,
                            uint32_t *scale_meta_offset)
{
    const uint32_t timeout_ms = plan->timeout_ms ? plan->timeout_ms : 10000u;
    uint32_t bytes = 0;
    if (plan->bias_i32) {
        bytes += versa_p_align_up(n_hw * sizeof(int32_t), VERSA_P_ALIGN_BYTES);
    }
    if (plan->mvout_mode == VERSA_P_MVOUT_FP32_PER_CHANNEL_Q8_24) {
        bytes += versa_p_align_up(n_hw * sizeof(int32_t), VERSA_P_ALIGN_BYTES);
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
        std::memset(cursor, 0, n_hw * sizeof(int32_t));
        std::memcpy(cursor, plan->bias_i32 + n0, n_real * sizeof(int32_t));
        uint32_t bias_bytes = versa_p_align_up(n_hw * sizeof(int32_t),
                                               VERSA_P_ALIGN_BYTES);
        versa_p_mvin_meta_desc desc = {};
        desc.dram_base = meta_buf->dma_addr;
        desc.meta_offset_bytes = kBiasMetaOffset;
        desc.byte_count = n_hw * (uint32_t)sizeof(int32_t);
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
        std::memset(cursor, 0, n_hw * sizeof(int32_t));
        std::memcpy(cursor, plan->scale_q8_24 + n0, n_real * sizeof(int32_t));
        versa_p_mvin_meta_desc desc = {};
        desc.dram_base = meta_buf->dma_addr +
                         (uint32_t)(cursor - (uint8_t *)meta_buf->vaddr);
        desc.meta_offset_bytes = scale_offset;
        desc.byte_count = n_hw * (uint32_t)sizeof(int32_t);
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
    const uint32_t output_stride_hw = align_elems_32(output_stride_n);
    const uint32_t m_total_hw = align_elems_32(plan->m);
    uint8_t next_w_bank = 0;
    uint8_t next_a_bank = 0;
    versa_p_buffer full_out_buf = {};
    uint32_t full_out_bytes =
        versa_p_align_up((uint32_t)((size_t)m_total_hw * output_stride_hw *
                                    sizeof(uint32_t)),
                         VERSA_P_ALIGN_BYTES);
    rc = versa_p_mem_alloc(dev, full_out_bytes, VERSA_P_ALIGN_BYTES,
                           &full_out_buf);
    if (rc != VERSA_P_OK) {
        return rc;
    }

    for (uint32_t m0 = 0; m0 < plan->m; m0 += VERSA_P_M_TILE) {
        uint32_t m_count = min_u32(VERSA_P_M_TILE, plan->m - m0);
        uint32_t m_hw = align_elems_32(m_count);
        for (uint32_t n0 = 0; n0 < plan->n; n0 += VERSA_P_N_TILE) {
            uint32_t n_count = min_u32(VERSA_P_N_TILE, plan->n - n0);
            uint32_t n_hw = align_elems_32(n_count);
            uint8_t o_bank = (uint8_t)(((m0 / VERSA_P_M_TILE) +
                                        (n0 / VERSA_P_N_TILE)) & 1u);
            versa_p_buffer meta_buf = {};
            uint32_t scale_meta_offset = 0;
            rc = load_metadata_if_needed(dev, plan, n0, n_count, n_hw, &meta_buf,
                                         &scale_meta_offset);
            if (rc != VERSA_P_OK) {
                versa_p_mem_free(dev, &meta_buf);
                versa_p_mem_free(dev, &full_out_buf);
                return rc;
            }

            struct TileSlot {
                versa_p_buffer a_buf = {};
                versa_p_buffer w_buf = {};
                uint32_t k0 = 0;
                uint32_t k_count = 0;
                uint32_t k_hw = 0;
                uint8_t a_bank = 0;
                uint8_t w_bank = 0;
            };

            TileSlot slots[2];
            const uint32_t max_k_hw = align_elems_32(min_u32(VERSA_P_K_TILE, plan->k));
            const uint32_t a_bytes = versa_p_align_up(m_hw * max_k_hw,
                                                      VERSA_P_ALIGN_BYTES);
            const uint32_t w_bytes =
                versa_p_align_up(packed_w_bytes(max_k_hw, n_hw),
                                 VERSA_P_ALIGN_BYTES);
            for (TileSlot &slot : slots) {
                rc = versa_p_mem_alloc(dev, a_bytes, VERSA_P_ALIGN_BYTES, &slot.a_buf);
                if (rc == VERSA_P_OK) {
                    rc = versa_p_mem_alloc(dev, w_bytes, VERSA_P_ALIGN_BYTES,
                                           &slot.w_buf);
                }
                if (rc != VERSA_P_OK) {
                    versa_p_mem_free(dev, &slots[0].a_buf);
                    versa_p_mem_free(dev, &slots[0].w_buf);
                    versa_p_mem_free(dev, &slots[1].a_buf);
                    versa_p_mem_free(dev, &slots[1].w_buf);
                    versa_p_mem_free(dev, &meta_buf);
                    versa_p_mem_free(dev, &full_out_buf);
                    return rc;
                }
            }

            auto free_slots = [&]() {
                versa_p_mem_free(dev, &slots[0].a_buf);
                versa_p_mem_free(dev, &slots[0].w_buf);
                versa_p_mem_free(dev, &slots[1].a_buf);
                versa_p_mem_free(dev, &slots[1].w_buf);
            };

            auto preload_slot = [&](TileSlot &slot, uint32_t k0,
                                    uint8_t a_bank, uint8_t w_bank) -> int {
                slot.k0 = k0;
                slot.k_count = min_u32(VERSA_P_K_TILE, plan->k - k0);
                slot.k_hw = align_elems_32(slot.k_count);
                slot.a_bank = a_bank;
                slot.w_bank = w_bank;
                copy_a_tile(plan->a, plan->a_stride_bytes, m0, k0,
                            m_count, slot.k_count, m_hw, slot.k_hw,
                            (int8_t *)slot.a_buf.vaddr);
                pack_w_tile(plan->w, plan->w_stride_n, k0, n0,
                            slot.k_count, n_count, slot.k_hw, n_hw,
                            (int8_t *)slot.w_buf.vaddr);

                versa_p_mvin_a_desc a_desc = {};
                a_desc.dram_base = slot.a_buf.dma_addr;
                a_desc.dram_row_stride_bytes = slot.k_hw;
                a_desc.m = (uint16_t)m_hw;
                a_desc.k = (uint16_t)slot.k_hw;
                a_desc.a_bank = slot.a_bank;
                int preload_rc = versa_p_start_mvin_a(dev, &a_desc);
                if (preload_rc != VERSA_P_OK) {
                    return preload_rc;
                }

                versa_p_mvin_w_desc w_desc = {};
                w_desc.dram_base = slot.w_buf.dma_addr;
                w_desc.k = (uint16_t)slot.k_hw;
                w_desc.n = (uint16_t)n_hw;
                w_desc.w_bank = slot.w_bank;
                preload_rc = versa_p_start_mvin_w(dev, &w_desc);
                if (preload_rc != VERSA_P_OK) {
                    (void)versa_p_wait(dev, VERSA_P_API_MVIN_A, timeout_ms);
                    return preload_rc;
                }
                preload_rc = versa_p_wait(dev, VERSA_P_API_MVIN_A, timeout_ms);
                if (preload_rc != VERSA_P_OK) {
                    return preload_rc;
                }
                return versa_p_wait(dev, VERSA_P_API_MVIN_W, timeout_ms);
            };

            uint8_t run_slot = 0;
            uint8_t load_a_bank = next_a_bank;
            uint8_t load_w_bank = next_w_bank;
            if (load_w_bank == dev->active_w_bank) {
                load_w_bank ^= 1u;
            }
            rc = preload_slot(slots[run_slot], 0, load_a_bank, load_w_bank);
            if (rc != VERSA_P_OK) {
                free_slots();
                versa_p_mem_free(dev, &meta_buf);
                versa_p_mem_free(dev, &full_out_buf);
                return rc;
            }

            while (rc == VERSA_P_OK) {
                TileSlot &cur = slots[run_slot];
                versa_p_gemm_i8_desc gemm = {};
                gemm.m = (uint16_t)m_hw;
                gemm.n = (uint16_t)n_hw;
                gemm.k = (uint16_t)cur.k_hw;
                gemm.a_bank = cur.a_bank;
                gemm.w_bank = cur.w_bank;
                gemm.o_bank = o_bank;
                gemm.accumulate_en = (cur.k0 != 0);
                gemm.add_bias_en = (cur.k0 == 0 && plan->bias_i32 != nullptr);
                gemm.bias_offset_bytes = kBiasMetaOffset;
                rc = versa_p_start_gemm_i8(dev, &gemm);
                if (rc != VERSA_P_OK) {
                    break;
                }

                const uint32_t following_k0 = cur.k0 + VERSA_P_K_TILE;
                const bool have_following = following_k0 < plan->k;
                if (have_following) {
                    TileSlot &next = slots[run_slot ^ 1u];
                    rc = preload_slot(next, following_k0,
                                      (uint8_t)(cur.a_bank ^ 1u),
                                      (uint8_t)(cur.w_bank ^ 1u));
                    if (rc != VERSA_P_OK) {
                        (void)versa_p_wait(dev, VERSA_P_API_GEMM_I8, timeout_ms);
                        break;
                    }
                }

                rc = versa_p_wait(dev, VERSA_P_API_GEMM_I8, timeout_ms);
                if (rc != VERSA_P_OK || !have_following) {
                    next_a_bank = cur.a_bank ^ 1u;
                    next_w_bank = cur.w_bank ^ 1u;
                    break;
                }
                run_slot ^= 1u;
            }

            free_slots();
            if (rc != VERSA_P_OK) {
                versa_p_mem_free(dev, &meta_buf);
                versa_p_mem_free(dev, &full_out_buf);
                return rc;
            }

            uint32_t scale_param = plan->tensor_scale_q8_24;
            if (plan->mvout_mode == VERSA_P_MVOUT_FP32_PER_CHANNEL_Q8_24) {
                scale_param = scale_meta_offset;
            }
            versa_p_mvout_desc mvout = {};
            mvout.dram_base = full_out_buf.dma_addr +
                              (uint32_t)(((size_t)m0 * output_stride_hw + n0) *
                                         sizeof(uint32_t));
            mvout.scale_param = scale_param;
            mvout.m = (uint16_t)m_hw;
            mvout.n = (uint16_t)n_hw;
            mvout.output_stride_n = (uint16_t)output_stride_hw;
            mvout.mode = plan->mvout_mode;
            mvout.o_bank = o_bank;
            rc = versa_p_mvout(dev, &mvout, timeout_ms);
            versa_p_mem_free(dev, &meta_buf);
            if (rc != VERSA_P_OK) {
                versa_p_mem_free(dev, &full_out_buf);
                return rc;
            }
        }
    }

    copy_output_tile_32b(full_out_buf.vaddr, output_stride_hw, plan->output,
                         output_stride_n, plan->m, plan->n);
    versa_p_mem_free(dev, &full_out_buf);
    return VERSA_P_OK;
}
