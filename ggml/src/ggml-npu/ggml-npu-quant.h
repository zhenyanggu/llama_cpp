#pragma once

#include "ggml-npu-common.h"

#include <vector>

namespace ggml_npu {

struct npu_q8_block_pack {
    std::vector<int8_t> quants;
    std::vector<float> scales;
    int64_t logical_k = 0;
    int64_t padded_k = 0;
};

float npu_compute_activation_tensor_scale(
        const struct ggml_tensor * src1);

void npu_compute_activation_dynamic_u8_params(
        const struct ggml_tensor * src1,
        float * scale,
        uint8_t * zero_point_u8);

std::vector<float> npu_compute_weight_output_scales(
        const struct ggml_tensor * src0,
        int64_t m0,
        int64_t m_cols);

bool npu_pack_activation_tile_dynamic_tensor_i8(
        const struct ggml_tensor * src1,
        int64_t n0,
        int64_t n_rows,
        int64_t k0,
        int64_t k_cols,
        int64_t k_stride,
        float scale,
        uint8_t zero_point_u8,
        std::vector<int8_t> * packed,
        std::string * error);

bool npu_pack_activation_tile_static_asym_i8(
        const struct ggml_tensor * src1,
        int64_t n0,
        int64_t n_rows,
        int64_t k0,
        int64_t k_cols,
        int64_t k_stride,
        float scale,
        int32_t zero_point,
        const std::vector<float> * smooth_scale,
        std::vector<int8_t> * packed,
        std::string * error);

bool npu_pack_weight_tile_fixed_i8_transposed(
        const struct ggml_tensor * src0,
        int64_t m0,
        int64_t m_cols,
        int64_t k0,
        int64_t k_rows,
        int64_t m_stride,
        const float * col_scales,
        std::vector<int8_t> * packed,
        std::string * error);

bool npu_pack_weight_tile_prequant_i8_transposed(
        const struct ggml_tensor * src0,
        int64_t m0,
        int64_t m_cols,
        int64_t k0,
        int64_t k_rows,
        int64_t m_stride,
        std::vector<int8_t> * packed,
        std::string * error);

bool npu_pack_activation_tile_q8_0_block(
        const struct ggml_tensor * src1,
        int64_t n0,
        int64_t n_rows,
        int64_t k0,
        int64_t k_cols,
        npu_q8_block_pack * packed,
        std::string * error);

bool npu_pack_weight_tile_q8_0_block_transposed(
        const struct ggml_tensor * src0,
        int64_t m0,
        int64_t m_cols,
        int64_t k0,
        int64_t k_rows,
        npu_q8_block_pack * packed,
        std::string * error);

} // namespace ggml_npu
