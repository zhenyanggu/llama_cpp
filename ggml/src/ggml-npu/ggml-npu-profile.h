#pragma once

#include "ggml-npu-common.h"
#include "ggml-npu.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace ggml_npu {

struct npu_profile_tensor_info {
    bool present = false;
    std::string name;
    std::string type;
    bool is_quantized = false;
    std::array<int64_t, GGML_MAX_DIMS> ne = {};
};

struct npu_profile_tile_record {
    int64_t tile_index = -1;
    int64_t m0 = 0;
    int64_t n0 = 0;
    int64_t k0 = 0;
    int64_t m = 0;
    int64_t n = 0;
    int64_t k = 0;
    std::string stage;
    bool needs_bias = false;
    bool writes_output = false;
    int32_t weight_pack_index = -1;
    int32_t bias_pack_index = -1;
    int64_t activation_bytes = 0;
    int64_t weight_bytes = 0;
    int64_t bias_bytes = 0;
    int64_t acc_readback_bytes = 0;
    int64_t output_write_bytes = 0;
    double activation_pack_us = 0.0;
    double host_copy_activation_us = 0.0;
    double host_copy_weight_us = 0.0;
    double bias_prepare_us = 0.0;
    double dma_in_activation_us = 0.0;
    double dma_in_weight_us = 0.0;
    double dma_in_bias_us = 0.0;
    double dma_in_pair_us = 0.0;
    double gemm_us = 0.0;
    double dma_out_us = 0.0;
    double postprocess_us = 0.0;
    double accounted_us = 0.0;
    double unaccounted_us = 0.0;
    double total_us = 0.0;
};

struct npu_profile_node_record {
    int64_t layer_id = -1;
    npu_profile_tensor_info root_tensor;
    npu_profile_tensor_info compute_tensor;
    npu_profile_tensor_info weight_tensor;
    npu_profile_tensor_info activation_tensor;
    npu_profile_tensor_info bias_tensor;
    npu_profile_tensor_info dst_tensor;
    std::string root_op_name;
    std::string compute_op_name;
    std::string semantic_op;
    std::string summary;
    int64_t m = 0;
    int64_t n = 0;
    int64_t k = 0;
    bool use_aicas_w8a8 = false;
    bool shape_table_hit = false;
    std::string shape_table_source;
    float activation_scale = 0.0f;
    int32_t activation_zero_point = 0;
    int64_t first_stage_tm = 0;
    int64_t first_stage_tn = 0;
    int64_t first_stage_tk = 0;
    int64_t stage2_k_block = 0;
    int64_t exec_tile_count = 0;
    int64_t weight_pack_count = 0;
    int64_t bias_pack_count = 0;
    uint64_t spm_bytes = 0;
    uint64_t acc_bytes = 0;
    uint32_t activation_offset = 0;
    uint32_t weight_offset = 0;
    uint32_t accumulator_offset = 0;
    uint32_t bias_accumulator_offset = 0;
    uint32_t output_accumulator_offset = 0;
    int64_t activation_pack_calls = 0;
    int64_t host_copy_activation_calls = 0;
    int64_t host_copy_weight_calls = 0;
    int64_t bias_prepare_calls = 0;
    int64_t dma_in_activation_calls = 0;
    int64_t dma_in_weight_calls = 0;
    int64_t dma_in_bias_calls = 0;
    int64_t dma_in_pair_calls = 0;
    int64_t gemm_calls = 0;
    int64_t gemm_plan_calls = 0;
    int64_t dma_out_calls = 0;
    int64_t postprocess_calls = 0;
    int64_t packed_activation_bytes_total = 0;
    int64_t copied_weight_bytes_total = 0;
    int64_t bias_bytes_total = 0;
    int64_t acc_readback_bytes_total = 0;
    int64_t output_write_bytes_total = 0;
    double activation_pack_us_total = 0.0;
    double host_copy_activation_us_total = 0.0;
    double host_copy_weight_us_total = 0.0;
    double bias_prepare_us_total = 0.0;
    double dma_in_activation_us_total = 0.0;
    double dma_in_weight_us_total = 0.0;
    double dma_in_bias_us_total = 0.0;
    double dma_in_pair_us_total = 0.0;
    double gemm_us_total = 0.0;
    double dma_out_us_total = 0.0;
    double postprocess_us_total = 0.0;
    double accounted_us_total = 0.0;
    double unaccounted_us_total = 0.0;
    double total_node_us = 0.0;
    std::string status = "success";
    std::string error;
    std::vector<npu_profile_tile_record> tiles;
};

struct npu_profile_summary_delta {
    int64_t node_count = 0;
    int64_t exec_tile_count = 0;
    int64_t weight_pack_count = 0;
    int64_t bias_pack_count = 0;
    int64_t activation_pack_calls = 0;
    int64_t host_copy_activation_calls = 0;
    int64_t host_copy_weight_calls = 0;
    int64_t bias_prepare_calls = 0;
    int64_t dma_in_activation_calls = 0;
    int64_t dma_in_weight_calls = 0;
    int64_t dma_in_bias_calls = 0;
    int64_t dma_in_pair_calls = 0;
    int64_t gemm_calls = 0;
    int64_t gemm_plan_calls = 0;
    int64_t dma_out_calls = 0;
    int64_t postprocess_calls = 0;
    int64_t packed_activation_bytes_total = 0;
    int64_t copied_weight_bytes_total = 0;
    int64_t bias_bytes_total = 0;
    int64_t acc_readback_bytes_total = 0;
    int64_t output_write_bytes_total = 0;
    int64_t total_node_us = 0;
    int64_t activation_pack_us_total = 0;
    int64_t host_copy_activation_us_total = 0;
    int64_t host_copy_weight_us_total = 0;
    int64_t bias_prepare_us_total = 0;
    int64_t dma_in_activation_us_total = 0;
    int64_t dma_in_weight_us_total = 0;
    int64_t dma_in_bias_us_total = 0;
    int64_t dma_in_pair_us_total = 0;
    int64_t gemm_us_total = 0;
    int64_t dma_out_us_total = 0;
    int64_t postprocess_us_total = 0;
    int64_t accounted_us_total = 0;
    int64_t unaccounted_us_total = 0;
};

bool npu_profile_enabled();
npu_profile_node_record npu_profile_init_node_record(int64_t layer_id, const npu_node_plan & plan);
void npu_profile_reset();
void npu_profile_add_node_record(npu_profile_node_record record);
void npu_profile_flush();
bool npu_summary_active();
void npu_summary_session_start();
void npu_summary_session_stop(ggml_npu_profile_summary * out);
void npu_summary_add_delta(const npu_profile_summary_delta & delta);

} // namespace ggml_npu
