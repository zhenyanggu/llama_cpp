#include "ggml-npu-quant.h"

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

extern "C" {
#include "ggml-cpu/quants.h"
}

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace ggml_npu {

static float npu_read_tensor_value_f32(
        const struct ggml_tensor * tensor,
        int64_t i0,
        int64_t i1) {
    if (tensor->type == GGML_TYPE_Q8_0) {
        const char * row_ptr = static_cast<const char *>(tensor->data) + i1 * tensor->nb[1];
        const block_q8_0 * blocks = reinterpret_cast<const block_q8_0 *>(row_ptr);
        const int64_t block_idx = i0 / NPU_Q8_BLOCK;
        const int64_t block_off = i0 % NPU_Q8_BLOCK;
        const float delta = ggml_fp16_to_fp32(blocks[block_idx].d);
        return delta * static_cast<float>(blocks[block_idx].qs[block_off]);
    }

    const char * base = static_cast<const char *>(tensor->data) + i1 * tensor->nb[1] + i0 * tensor->nb[0];

    switch (tensor->type) {
        case GGML_TYPE_F32:
            return *reinterpret_cast<const float *>(base);
        case GGML_TYPE_F16:
            return ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t *>(base));
        case GGML_TYPE_I8:
            return static_cast<float>(*reinterpret_cast<const int8_t *>(base));
        default:
            return 0.0f;
    }
}

static float npu_read_weight_value_f32(
        const struct ggml_tensor * tensor,
        int64_t k,
        int64_t m) {
    if (tensor->type == GGML_TYPE_Q8_0) {
        const char * row_ptr = static_cast<const char *>(tensor->data) + m * tensor->nb[1];
        const block_q8_0 * blocks = reinterpret_cast<const block_q8_0 *>(row_ptr);
        const int64_t block_idx = k / QK8_0;
        const int64_t block_off = k % QK8_0;
        const float delta = ggml_fp16_to_fp32(blocks[block_idx].d);
        return delta * static_cast<float>(blocks[block_idx].qs[block_off]);
    }

    return npu_read_tensor_value_f32(tensor, k, m);
}

float npu_compute_activation_tensor_scale(
        const struct ggml_tensor * src1) {
    float max_abs = 0.0f;
    for (int64_t n = 0; n < src1->ne[1]; ++n) {
        for (int64_t k = 0; k < src1->ne[0]; ++k) {
            const float v = npu_read_tensor_value_f32(src1, k, n);
            max_abs = std::max(max_abs, std::fabs(v));
        }
    }
    return max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
}

void npu_compute_activation_dynamic_u8_params(
        const struct ggml_tensor * src1,
        float * scale,
        uint8_t * zero_point_u8) {
    float min_v = std::numeric_limits<float>::max();
    float max_v = -std::numeric_limits<float>::max();
    for (int64_t n = 0; n < src1->ne[1]; ++n) {
        for (int64_t k = 0; k < src1->ne[0]; ++k) {
            const float v = npu_read_tensor_value_f32(src1, k, n);
            min_v = std::min(min_v, v);
            max_v = std::max(max_v, v);
        }
    }

    float computed_scale = 1.0f;
    int32_t computed_zero_point = 0;
    if (max_v > min_v) {
        computed_scale = (max_v - min_v) / 255.0f;
        if (computed_scale < 1e-8f) {
            computed_scale = 1e-8f;
        }
        computed_zero_point = static_cast<int32_t>(std::lrint(-min_v / computed_scale));
        computed_zero_point = std::max(0, std::min(255, computed_zero_point));
    }

    *scale = computed_scale;
    *zero_point_u8 = static_cast<uint8_t>(computed_zero_point);
}

static int8_t npu_quantize_i8(float value, float scale) {
    if (scale <= 0.0f || !std::isfinite(scale)) {
        return 0;
    }

    int32_t q = static_cast<int32_t>(std::lrint(value / scale));
    q = std::max(-128, std::min(127, q));
    return static_cast<int8_t>(q);
}

std::vector<float> npu_compute_weight_output_scales(
        const struct ggml_tensor * src0,
        int64_t m0,
        int64_t m_cols) {
    std::vector<float> scales(static_cast<size_t>(m_cols), 1.0f);
    for (int64_t m = 0; m < m_cols; ++m) {
        float max_abs = 0.0f;
        for (int64_t k = 0; k < src0->ne[0]; ++k) {
            const float v = npu_read_weight_value_f32(src0, k, m0 + m);
            max_abs = std::max(max_abs, std::fabs(v));
        }
        scales[static_cast<size_t>(m)] = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
    }
    return scales;
}

bool npu_pack_activation_tile_dynamic_tensor_i8(
        const struct ggml_tensor * src1,
        int64_t n0,
        int64_t n_rows,
        int64_t k0,
        int64_t k_cols,
        float scale,
        uint8_t zero_point_u8,
        std::vector<int8_t> * packed,
        std::string * error) {
    if (packed == nullptr) {
        if (error) {
            *error = "激活打包输出为空";
        }
        return false;
    }
    packed->assign(static_cast<size_t>(n_rows * k_cols), 0);
    for (int64_t n = 0; n < n_rows; ++n) {
        for (int64_t k = 0; k < k_cols; ++k) {
            const float v = npu_read_tensor_value_f32(src1, k0 + k, n0 + n);
            int32_t q_u8 = static_cast<int32_t>(std::lrint(v / scale)) + static_cast<int32_t>(zero_point_u8);
            q_u8 = std::max(0, std::min(255, q_u8));
            (*packed)[static_cast<size_t>(n * k_cols + k)] = static_cast<int8_t>(q_u8 - 128);
        }
    }
    return true;
}

bool npu_pack_activation_tile_static_asym_i8(
        const struct ggml_tensor * src1,
        int64_t n0,
        int64_t n_rows,
        int64_t k0,
        int64_t k_cols,
        float scale,
        int32_t zero_point,
        std::vector<int8_t> * packed,
        std::string * error) {
    if (packed == nullptr) {
        if (error) {
            *error = "激活打包输出为空";
        }
        return false;
    }
    packed->assign(static_cast<size_t>(n_rows * k_cols), 0);
    for (int64_t n = 0; n < n_rows; ++n) {
        for (int64_t k = 0; k < k_cols; ++k) {
            const float v = npu_read_tensor_value_f32(src1, k0 + k, n0 + n);
            const float shifted = v / scale + static_cast<float>(zero_point);
            int32_t q = static_cast<int32_t>(std::lrint(shifted));
            q = std::max(-128, std::min(127, q));
            (*packed)[static_cast<size_t>(n * k_cols + k)] = static_cast<int8_t>(q);
        }
    }
    return true;
}

bool npu_pack_weight_tile_fixed_i8_transposed(
        const struct ggml_tensor * src0,
        int64_t m0,
        int64_t m_cols,
        int64_t k0,
        int64_t k_rows,
        const float * col_scales,
        std::vector<int8_t> * packed,
        std::string * error) {
    if (packed == nullptr) {
        if (error) {
            *error = "权重打包输出为空";
        }
        return false;
    }
    packed->assign(static_cast<size_t>(k_rows * m_cols), 0);
    for (int64_t k = 0; k < k_rows; ++k) {
        for (int64_t m = 0; m < m_cols; ++m) {
            const float v = npu_read_weight_value_f32(src0, k0 + k, m0 + m);
            (*packed)[static_cast<size_t>(k * m_cols + m)] = npu_quantize_i8(v, col_scales[m]);
        }
    }
    return true;
}

bool npu_pack_weight_tile_prequant_i8_transposed(
        const struct ggml_tensor * src0,
        int64_t m0,
        int64_t m_cols,
        int64_t k0,
        int64_t k_rows,
        std::vector<int8_t> * packed,
        std::string * error) {
    if (src0 == nullptr || src0->type != GGML_TYPE_I8) {
        if (error) {
            *error = "预量化权重打包仅支持 GGML_TYPE_I8";
        }
        return false;
    }
    if (packed == nullptr) {
        if (error) {
            *error = "权重打包输出为空";
        }
        return false;
    }

    packed->assign(static_cast<size_t>(k_rows * m_cols), 0);
    for (int64_t m = 0; m < m_cols; ++m) {
        const int8_t * row_ptr = reinterpret_cast<const int8_t *>(
            static_cast<const char *>(src0->data) + (m0 + m) * src0->nb[1] + k0 * src0->nb[0]);
        for (int64_t k = 0; k < k_rows; ++k) {
            (*packed)[static_cast<size_t>(k * m_cols + m)] = row_ptr[k];
        }
    }

    return true;
}

static int64_t npu_pad_to_q8_block(int64_t k_cols) {
    return ((k_cols + NPU_Q8_BLOCK - 1) / NPU_Q8_BLOCK) * NPU_Q8_BLOCK;
}

static bool npu_try_pack_activation_q8_0_direct(
        const struct ggml_tensor * src1,
        int64_t n0,
        int64_t n_rows,
        int64_t k0,
        int64_t k_cols,
        npu_q8_block_pack * packed) {
    if (src1->type != GGML_TYPE_Q8_0) {
        return false;
    }
    if ((k0 % NPU_Q8_BLOCK) != 0) {
        return false;
    }

    const int64_t padded_k = npu_pad_to_q8_block(k_cols);
    packed->logical_k = k_cols;
    packed->padded_k = padded_k;
    packed->quants.assign(static_cast<size_t>(n_rows * padded_k), 0);
    packed->scales.assign(static_cast<size_t>(n_rows * (padded_k / NPU_Q8_BLOCK)), 1.0f);

    for (int64_t n = 0; n < n_rows; ++n) {
        const char * row_ptr = static_cast<const char *>(src1->data) + (n0 + n) * src1->nb[1];
        const block_q8_0 * blocks = reinterpret_cast<const block_q8_0 *>(row_ptr) + (k0 / NPU_Q8_BLOCK);
        for (int64_t blk = 0; blk < padded_k / NPU_Q8_BLOCK; ++blk) {
            if (blk * NPU_Q8_BLOCK < k_cols) {
                packed->scales[static_cast<size_t>(n * (padded_k / NPU_Q8_BLOCK) + blk)] =
                    ggml_fp16_to_fp32(blocks[blk].d);
                std::memcpy(
                    packed->quants.data() + static_cast<size_t>(n * padded_k + blk * NPU_Q8_BLOCK),
                    blocks[blk].qs,
                    sizeof(blocks[blk].qs));
            }
        }
    }
    return true;
}

static bool npu_try_pack_weight_q8_0_direct_transposed(
        const struct ggml_tensor * src0,
        int64_t m0,
        int64_t m_cols,
        int64_t k0,
        int64_t k_rows,
        npu_q8_block_pack * packed) {
    if (src0->type != GGML_TYPE_Q8_0) {
        return false;
    }
    if ((k0 % NPU_Q8_BLOCK) != 0) {
        return false;
    }

    const int64_t padded_k = npu_pad_to_q8_block(k_rows);
    packed->logical_k = k_rows;
    packed->padded_k = padded_k;
    packed->quants.assign(static_cast<size_t>(padded_k * m_cols), 0);
    packed->scales.assign(static_cast<size_t>(m_cols * (padded_k / NPU_Q8_BLOCK)), 1.0f);

    for (int64_t m = 0; m < m_cols; ++m) {
        const char * row_ptr = static_cast<const char *>(src0->data) + (m0 + m) * src0->nb[1];
        const block_q8_0 * blocks = reinterpret_cast<const block_q8_0 *>(row_ptr) + (k0 / NPU_Q8_BLOCK);
        for (int64_t blk = 0; blk < padded_k / NPU_Q8_BLOCK; ++blk) {
            if (blk * NPU_Q8_BLOCK < k_rows) {
                packed->scales[static_cast<size_t>(m * (padded_k / NPU_Q8_BLOCK) + blk)] =
                    ggml_fp16_to_fp32(blocks[blk].d);
                for (int64_t kk = 0; kk < NPU_Q8_BLOCK; ++kk) {
                    packed->quants[static_cast<size_t>((blk * NPU_Q8_BLOCK + kk) * m_cols + m)] = blocks[blk].qs[kk];
                }
            }
        }
    }
    return true;
}

bool npu_pack_activation_tile_q8_0_block(
        const struct ggml_tensor * src1,
        int64_t n0,
        int64_t n_rows,
        int64_t k0,
        int64_t k_cols,
        npu_q8_block_pack * packed,
        std::string * error) {
    if (packed == nullptr) {
        if (error) {
            *error = "激活打包输出为空";
        }
        return false;
    }

    if (npu_try_pack_activation_q8_0_direct(src1, n0, n_rows, k0, k_cols, packed)) {
        return true;
    }

    const int64_t padded_k = npu_pad_to_q8_block(k_cols);
    std::vector<float> staging(static_cast<size_t>(n_rows * padded_k), 0.0f);
    std::vector<block_q8_0> qblocks(static_cast<size_t>(n_rows * padded_k / NPU_Q8_BLOCK));

    for (int64_t n = 0; n < n_rows; ++n) {
        float * row = staging.data() + n * padded_k;
        for (int64_t k = 0; k < k_cols; ++k) {
            row[k] = npu_read_tensor_value_f32(src1, k0 + k, n0 + n);
        }
        quantize_row_q8_0(row, qblocks.data() + (n * padded_k / NPU_Q8_BLOCK), padded_k);
    }

    packed->logical_k = k_cols;
    packed->padded_k = padded_k;
    packed->quants.assign(static_cast<size_t>(n_rows * padded_k), 0);
    packed->scales.assign(static_cast<size_t>(n_rows * (padded_k / NPU_Q8_BLOCK)), 1.0f);

    for (int64_t n = 0; n < n_rows; ++n) {
        for (int64_t blk = 0; blk < padded_k / NPU_Q8_BLOCK; ++blk) {
            const block_q8_0 & block = qblocks[static_cast<size_t>(n * (padded_k / NPU_Q8_BLOCK) + blk)];
            packed->scales[static_cast<size_t>(n * (padded_k / NPU_Q8_BLOCK) + blk)] = ggml_fp16_to_fp32(block.d);
            std::memcpy(
                packed->quants.data() + static_cast<size_t>(n * padded_k + blk * NPU_Q8_BLOCK),
                block.qs,
                sizeof(block.qs));
        }
    }

    return true;
}

bool npu_pack_weight_tile_q8_0_block_transposed(
        const struct ggml_tensor * src0,
        int64_t m0,
        int64_t m_cols,
        int64_t k0,
        int64_t k_rows,
        npu_q8_block_pack * packed,
        std::string * error) {
    if (packed == nullptr) {
        if (error) {
            *error = "权重打包输出为空";
        }
        return false;
    }

    if (npu_try_pack_weight_q8_0_direct_transposed(src0, m0, m_cols, k0, k_rows, packed)) {
        return true;
    }

    const int64_t padded_k = npu_pad_to_q8_block(k_rows);
    std::vector<float> staging(static_cast<size_t>(m_cols * padded_k), 0.0f);
    std::vector<block_q8_0> qblocks(static_cast<size_t>(m_cols * padded_k / NPU_Q8_BLOCK));

    for (int64_t m = 0; m < m_cols; ++m) {
        float * row = staging.data() + m * padded_k;
        for (int64_t k = 0; k < k_rows; ++k) {
            row[k] = npu_read_weight_value_f32(src0, k0 + k, m0 + m);
        }
        quantize_row_q8_0(row, qblocks.data() + (m * padded_k / NPU_Q8_BLOCK), padded_k);
    }

    packed->logical_k = k_rows;
    packed->padded_k = padded_k;
    packed->quants.assign(static_cast<size_t>(padded_k * m_cols), 0);
    packed->scales.assign(static_cast<size_t>(m_cols * (padded_k / NPU_Q8_BLOCK)), 1.0f);

    for (int64_t m = 0; m < m_cols; ++m) {
        for (int64_t blk = 0; blk < padded_k / NPU_Q8_BLOCK; ++blk) {
            const block_q8_0 & block = qblocks[static_cast<size_t>(m * (padded_k / NPU_Q8_BLOCK) + blk)];
            packed->scales[static_cast<size_t>(m * (padded_k / NPU_Q8_BLOCK) + blk)] = ggml_fp16_to_fp32(block.d);
            for (int64_t kk = 0; kk < NPU_Q8_BLOCK; ++kk) {
                packed->quants[static_cast<size_t>((blk * NPU_Q8_BLOCK + kk) * m_cols + m)] = block.qs[kk];
            }
        }
    }

    return true;
}

} // namespace ggml_npu
