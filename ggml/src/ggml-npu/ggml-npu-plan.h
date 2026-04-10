#pragma once

#include "ggml-npu-common.h"

namespace ggml_npu {

bool npu_is_fusable_bias_add(const struct ggml_tensor * op, std::string * reason);
bool npu_can_handle_mul_mat(const struct ggml_tensor * op, std::string * reason);
npu_tiling_config npu_default_tiling_config(void);
npu_node_plan npu_create_mul_mat_plan(struct ggml_tensor * op, const npu_tiling_config & config);
bool npu_plan_is_aot_stable(const npu_node_plan & plan, std::string * reason);

void npu_clear_aicas_w8a8_table(void);
bool npu_register_aicas_w8a8(
        const char * weight_name,
        float act_scale,
        int32_t act_zero_point_u8,
        const float * weight_scale,
        size_t weight_scale_len,
        const int32_t * sum_w,
        size_t sum_w_len);

} // namespace ggml_npu
