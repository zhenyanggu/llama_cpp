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

GGML_BACKEND_API void ggml_backend_npu_w8a8_clear(void);
GGML_BACKEND_API bool ggml_backend_npu_w8a8_register(
        const char * weight_name,
        float act_scale,
        int32_t act_zero_point_u8,
        const float * weight_scale,
        size_t weight_scale_len,
        const int32_t * sum_w,
        size_t sum_w_len);

#ifdef __cplusplus
}
#endif
