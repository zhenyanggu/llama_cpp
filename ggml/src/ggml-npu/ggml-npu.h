#pragma once

#include "ggml-backend.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_npu_reg(void);
GGML_BACKEND_API ggml_backend_t ggml_backend_npu_init(void);
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_npu_buffer_type(void);
GGML_BACKEND_API bool ggml_backend_is_npu(ggml_backend_t backend);

typedef struct ggml_npu_profile_summary {
    int32_t valid;
    int32_t used_npu;
    int64_t node_count;
    int64_t exec_tile_count;
    int64_t weight_pack_count;
    int64_t bias_pack_count;
    int64_t activation_pack_calls;
    int64_t host_copy_activation_calls;
    int64_t host_copy_weight_calls;
    int64_t bias_prepare_calls;
    int64_t dma_in_activation_calls;
    int64_t dma_in_weight_calls;
    int64_t dma_in_bias_calls;
    int64_t dma_in_pair_calls;
    int64_t spm_activation_reuse_hits;
    int64_t spm_weight_reuse_hits;
    int64_t gemm_calls;
    int64_t gemm_plan_calls;
    int64_t dma_out_calls;
    int64_t postprocess_calls;
    int64_t raw_acc_mvout_nodes;
    int64_t raw_acc_mvout_tiles;
    int64_t w_prefetch_calls;
    int64_t w_prefetch_hits;
    int64_t w_prefetch_conflicts;
    int64_t packed_activation_bytes_total;
    int64_t copied_weight_bytes_total;
    int64_t dma_in_activation_bytes_total;
    int64_t dma_in_weight_bytes_total;
    int64_t bias_bytes_total;
    int64_t acc_readback_bytes_total;
    int64_t output_write_bytes_total;
    int64_t total_node_us;
    int64_t setup_runtime_us_total;
    int64_t setup_validate_us_total;
    int64_t setup_buffer_alloc_us_total;
    int64_t setup_cache_alloc_us_total;
    int64_t setup_profile_begin_us_total;
    int64_t activation_pack_us_total;
    int64_t host_copy_activation_us_total;
    int64_t host_copy_weight_us_total;
    int64_t bias_prepare_us_total;
    int64_t dma_in_activation_us_total;
    int64_t dma_in_weight_us_total;
    int64_t dma_in_bias_us_total;
    int64_t dma_in_pair_us_total;
    int64_t w_prefetch_wait_us_total;
    int64_t w_prefetch_hidden_candidate_us_total;
    int64_t gemm_us_total;
    int64_t dma_out_us_total;
    int64_t postprocess_us_total;
    int64_t accounted_us_total;
    int64_t unaccounted_us_total;
    int64_t runtime_total_us;
    int64_t runtime_dma_in_us;
    int64_t runtime_compute_us;
    int64_t runtime_compute_exclusive_us;
    int64_t runtime_dma_out_us;
    int64_t runtime_layout_us;
    int64_t runtime_layout_exclusive_us;
    int64_t runtime_wait_irq_us;
    int64_t runtime_mvin_calls;
    int64_t runtime_compute_calls;
    int64_t runtime_gemm_plan_calls;
    int64_t runtime_mvout_calls;
    int64_t runtime_layout_calls;
} ggml_npu_profile_summary;

GGML_BACKEND_API void ggml_backend_npu_profile_summary_start(void);
GGML_BACKEND_API void ggml_backend_npu_profile_summary_stop(ggml_npu_profile_summary * out);

GGML_BACKEND_API void ggml_backend_npu_w8a8_clear(void);
GGML_BACKEND_API bool ggml_backend_npu_w8a8_register(
        const char * weight_name,
        float act_scale,
    int32_t act_scale_q8_24,
        int32_t act_zero_point_u8,
        const float * weight_scale,
        size_t weight_scale_len,
        const int32_t * sum_w,
        size_t sum_w_len,
        const float * smooth_scale,
        size_t smooth_scale_len);
GGML_BACKEND_API bool ggml_backend_npu_w8a8_preload(const struct ggml_tensor * weight_tensor);
GGML_BACKEND_API void ggml_backend_npu_w8a8_preload_clear(void);

GGML_BACKEND_API bool ggml_backend_npu_i8_gemm_raw_packed(
        const char * op_name,
        const int8_t * weight_kxm,
        int64_t m,
        int64_t k,
        int64_t weight_stride_m,
        const int8_t * act_nxk,
        int64_t n,
        int64_t act_stride_k,
        int32_t * out_nxm,
        int64_t out_stride_m);

GGML_BACKEND_API void * ggml_backend_npu_mem_alloc(size_t size);
GGML_BACKEND_API void ggml_backend_npu_mem_free(void * ptr);
GGML_BACKEND_API void ggml_backend_npu_runtime_shutdown(void);
GGML_BACKEND_API void ggml_backend_npu_decode_runtime_shutdown(void);
GGML_BACKEND_API void ggml_backend_npu_decode_overlay_mark_inactive(void);
GGML_BACKEND_API void ggml_backend_npu_decode_overlay_mark_active(void);

GGML_BACKEND_API bool ggml_backend_npu_i8_gemm_raw_cma(
        const char * op_name,
        const int8_t * weight_cma_kxm,
        int64_t m,
        int64_t k,
        int64_t weight_stride_m,
        const int8_t * act_cma_nxk,
        int64_t n,
        int64_t act_stride_k,
        int32_t * out_nxm,
        int64_t out_stride_m);

GGML_BACKEND_API bool ggml_backend_npu_decode_w4a16_gemv(
        const char * op_name,
        const void * q4_data,
        int64_t packed_k,
        int64_t out_channels,
        int64_t q4_nb1,
        const void * scale_data,
        int scale_type,
        int64_t scale_nb0,
        int64_t scale_nb1,
        const void * zero_data,
        int zero_type,
        int64_t zero_nb0,
        int64_t zero_nb1,
        const void * act_data,
        int act_type,
        int64_t act_nb0,
        int64_t act_nb1,
        const float * smooth_scale,
        int64_t k,
        int64_t n_cols,
        float * dst_data,
        int64_t dst_nb1);

GGML_BACKEND_API bool ggml_backend_npu_decode_w4a16_gemv_ex(
        const char * op_name,
        const void * q4_data,
        int64_t packed_k,
        int64_t out_channels,
        int64_t q4_nb1,
        const void * scale_data,
        int scale_type,
        int64_t scale_nb0,
        int64_t scale_nb1,
        const void * zero_data,
        int zero_type,
        int64_t zero_nb0,
        int64_t zero_nb1,
        const void * act_data,
        int act_type,
        int64_t act_nb0,
        int64_t act_nb1,
        const float * smooth_scale,
        int64_t k,
        int64_t n_cols,
        void * dst_data,
        int dst_type,
        int64_t dst_nb1);

GGML_BACKEND_API bool ggml_backend_npu_decode_w4a16_preload(
        const char * weight_name,
        const void * q4_data,
        int64_t packed_k,
        int64_t out_channels,
        int64_t q4_nb1,
        const void * scale_data,
        int scale_type,
        int64_t scale_nb0,
        int64_t scale_nb1,
        const void * zero_data,
        int zero_type,
        int64_t zero_nb0,
        int64_t zero_nb1,
        int64_t k,
        const float * smooth_scale,
        size_t smooth_scale_len);

#ifdef __cplusplus
}
#endif
