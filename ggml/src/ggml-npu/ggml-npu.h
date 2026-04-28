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
    int64_t gemm_calls;
    int64_t dma_out_calls;
    int64_t postprocess_calls;
    int64_t packed_activation_bytes_total;
    int64_t copied_weight_bytes_total;
    int64_t bias_bytes_total;
    int64_t acc_readback_bytes_total;
    int64_t output_write_bytes_total;
    int64_t total_node_us;
    int64_t activation_pack_us_total;
    int64_t host_copy_activation_us_total;
    int64_t host_copy_weight_us_total;
    int64_t bias_prepare_us_total;
    int64_t dma_in_activation_us_total;
    int64_t dma_in_weight_us_total;
    int64_t dma_in_bias_us_total;
    int64_t dma_in_pair_us_total;
    int64_t gemm_us_total;
    int64_t dma_out_us_total;
    int64_t postprocess_us_total;
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
        size_t sum_w_len);
GGML_BACKEND_API bool ggml_backend_npu_w8a8_preload(const struct ggml_tensor * weight_tensor);
GGML_BACKEND_API void ggml_backend_npu_w8a8_preload_clear(void);

#ifdef __cplusplus
}
#endif
