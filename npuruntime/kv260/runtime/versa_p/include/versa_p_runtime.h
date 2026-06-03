#ifndef VERSA_P_RUNTIME_H
#define VERSA_P_RUNTIME_H

#include "versa_p_error.h"
#include "versa_p_regs.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VERSA_P_DEFAULT_CMA_BYTES      (1024u * 1024u * 1024u)
#define VERSA_P_M_TILE                 256u
#define VERSA_P_K_TILE                 768u
#define VERSA_P_N_TILE                 480u
#define VERSA_P_K_MAX                  4096u
#define VERSA_P_ALIGN_BYTES            16u
#define VERSA_P_HW_ALIGN_ELEMS         32u
#define VERSA_P_DMA_ALIGN_BYTES        16u
#define VERSA_P_META_BYTES             (1024u * 16u)

typedef struct versa_p_device versa_p_device;

typedef struct versa_p_options {
    const char *dev_path;
    uint32_t cma_size;
} versa_p_options;

typedef struct versa_p_device_info {
    uint64_t regs_phys;
    uint32_t regs_size;
    uint32_t dma_addr_bits;
    uint32_t has_irq;
    uint32_t flags;
    uint32_t cma_dma_addr;
    uint32_t cma_size;
    void *cma_vaddr;
} versa_p_device_info;

typedef struct versa_p_hw_state {
    uint32_t magic;
    uint32_t abi_version;
    uint32_t mode_id;
    uint32_t caps;
    uint32_t status;
    uint32_t error;
} versa_p_hw_state;

#define VERSA_P_INFO_FLAG_DMA_COPY         0x00000001u
#define VERSA_P_INFO_FLAG_CACHEABLE_BUFFER 0x00000002u

typedef struct versa_p_buffer {
    void *vaddr;
    uint32_t dma_addr;
    uint32_t size;
    uint32_t offset;
} versa_p_buffer;

typedef struct versa_p_mvin_a_desc {
    uint32_t dram_base;
    uint32_t dram_row_stride_bytes;
    uint16_t m;
    uint16_t k;
    uint8_t a_bank;
    uint8_t u8_minus_128;
} versa_p_mvin_a_desc;

typedef struct versa_p_mvin_w_desc {
    uint32_t dram_base;
    uint16_t k;
    uint16_t n;
    uint8_t w_bank;
} versa_p_mvin_w_desc;

typedef struct versa_p_mvin_meta_desc {
    uint32_t dram_base;
    uint32_t meta_offset_bytes;
    uint32_t byte_count;
    uint8_t meta_type;
} versa_p_mvin_meta_desc;

typedef struct versa_p_gemm_i8_desc {
    uint16_t m;
    uint16_t n;
    uint16_t k;
    uint8_t a_bank;
    uint8_t w_bank;
    uint8_t accumulate_en;
    uint8_t add_bias_en;
    uint32_t bias_offset_bytes;
    uint8_t o_bank;
} versa_p_gemm_i8_desc;

typedef struct versa_p_attention_qk_desc {
    uint16_t token_count;
    uint16_t q_row_start;
    uint16_t q_rows;
    uint32_t gamma16_fix;
    uint8_t q_bank;
    uint8_t k_bank;
    uint8_t o_bank;
    uint8_t causal_mask;
} versa_p_attention_qk_desc;

typedef struct versa_p_pv_log8_desc {
    uint16_t m;
    uint16_t n;
    uint16_t k;
    uint8_t p_bank;
    uint8_t v_bank;
    uint8_t o_bank;
} versa_p_pv_log8_desc;

typedef struct versa_p_mvout_desc {
    uint32_t dram_base;
    uint32_t scale_param;
    uint16_t m;
    uint16_t n;
    uint16_t output_stride_n;
    uint8_t mode;
    uint8_t o_bank;
} versa_p_mvout_desc;

typedef struct versa_p_attention_logp_mvout_desc {
    uint32_t dram_base;
    uint16_t token_count;
    uint16_t q_row_start;
    uint16_t q_rows;
    uint16_t output_stride_bytes;
    uint8_t o_bank;
    uint8_t causal_mask;
} versa_p_attention_logp_mvout_desc;

typedef struct versa_p_gemm_plan {
    const int8_t *a;
    uint32_t a_stride_bytes;
    const int8_t *w;
    uint32_t w_stride_n;
    const int32_t *bias_i32;
    const int32_t *scale_q8_24;
    uint32_t tensor_scale_q8_24;
    void *output;
    uint32_t output_stride_n;
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint8_t mvout_mode;
    uint32_t timeout_ms;
} versa_p_gemm_plan;

typedef struct versa_p_profile_counters {
    uint64_t global_cycles;
    uint64_t mvin_a_busy_cycles;
    uint64_t mvin_w_busy_cycles;
    uint64_t mvin_meta_busy_cycles;
    uint64_t gemm_busy_cycles;
    uint64_t mvout_busy_cycles;
    uint64_t busy_any_cycles;
    uint64_t busy_multi_cycles;
    uint64_t axi_r_beats[4];
    uint64_t axi_w_beats[4];
    uint64_t axi_ar_stall_cycles[4];
    uint64_t axi_r_stall_cycles[4];
    uint64_t axi_aw_stall_cycles[4];
    uint64_t axi_w_stall_cycles[4];
} versa_p_profile_counters;

int versa_p_init(versa_p_device **out_dev, const versa_p_options *options);
void versa_p_destroy(versa_p_device *dev);
int versa_p_reset(versa_p_device *dev);
int versa_p_reinit(versa_p_device *dev);
int versa_p_get_info(versa_p_device *dev, versa_p_device_info *out_info);
int versa_p_get_hw_state(versa_p_device *dev, versa_p_hw_state *out_state);

int versa_p_mem_alloc(versa_p_device *dev, uint32_t size, uint32_t alignment,
                      versa_p_buffer *out_buffer);
void versa_p_mem_free(versa_p_device *dev, versa_p_buffer *buffer);
uint32_t versa_p_dma_addr(versa_p_device *dev, const void *ptr);
int versa_p_dma_copy_from_cma(versa_p_device *dev, const void *cma_ptr,
                              void *dst, size_t bytes,
                              uint32_t chunk_bytes);
int versa_p_dma_copy_to_cma(versa_p_device *dev, void *cma_ptr,
                            const void *src, size_t bytes,
                            uint32_t chunk_bytes);
int versa_p_sync_for_cpu(versa_p_device *dev, const void *cma_ptr,
                         size_t bytes);
int versa_p_sync_for_device(versa_p_device *dev, const void *cma_ptr,
                            size_t bytes);

uint64_t versa_p_read64(versa_p_device *dev, uint32_t offset);
void versa_p_write64(versa_p_device *dev, uint32_t offset, uint64_t value);
int versa_p_read_status(versa_p_device *dev, versa_p_api api,
                        versa_p_reg_status *out_status);
int versa_p_profile_start(versa_p_device *dev);
int versa_p_profile_stop(versa_p_device *dev);
int versa_p_profile_read(versa_p_device *dev,
                         versa_p_profile_counters *out_counters);

int versa_p_start_mvin_a(versa_p_device *dev, const versa_p_mvin_a_desc *desc);
int versa_p_start_mvin_w(versa_p_device *dev, const versa_p_mvin_w_desc *desc);
int versa_p_start_mvin_meta(versa_p_device *dev,
                            const versa_p_mvin_meta_desc *desc);
int versa_p_start_gemm_i8(versa_p_device *dev, const versa_p_gemm_i8_desc *desc);
int versa_p_start_attention_qk_logp(versa_p_device *dev,
                                    const versa_p_attention_qk_desc *desc);
int versa_p_start_gemm_pv_log8(versa_p_device *dev,
                               const versa_p_pv_log8_desc *desc);
int versa_p_start_mvout(versa_p_device *dev, const versa_p_mvout_desc *desc);
int versa_p_start_mvout_attention_logp(
    versa_p_device *dev, const versa_p_attention_logp_mvout_desc *desc);

int versa_p_wait(versa_p_device *dev, versa_p_api api, uint32_t timeout_ms);

int versa_p_mvin_a(versa_p_device *dev, const versa_p_mvin_a_desc *desc,
                   uint32_t timeout_ms);
int versa_p_mvin_w(versa_p_device *dev, const versa_p_mvin_w_desc *desc,
                   uint32_t timeout_ms);
int versa_p_mvin_meta(versa_p_device *dev, const versa_p_mvin_meta_desc *desc,
                      uint32_t timeout_ms);
int versa_p_gemm_i8(versa_p_device *dev, const versa_p_gemm_i8_desc *desc,
                    uint32_t timeout_ms);
int versa_p_attention_qk_logp(versa_p_device *dev,
                              const versa_p_attention_qk_desc *desc,
                              uint32_t timeout_ms);
int versa_p_gemm_pv_log8(versa_p_device *dev,
                         const versa_p_pv_log8_desc *desc,
                         uint32_t timeout_ms);
int versa_p_mvout(versa_p_device *dev, const versa_p_mvout_desc *desc,
                  uint32_t timeout_ms);
int versa_p_mvout_attention_logp(
    versa_p_device *dev, const versa_p_attention_logp_mvout_desc *desc,
    uint32_t timeout_ms);

int versa_p_gemm_plan_run(versa_p_device *dev, const versa_p_gemm_plan *plan);

#ifdef __cplusplus
}
#endif

#endif
