#pragma once

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ggml_npu {

constexpr int64_t NPU_SA_TILE = 32;
constexpr int64_t NPU_STAGE2_K_TILE = 2048;
constexpr int64_t NPU_Q8_BLOCK = 32;
// TODO: Replace these placeholder capacities with the real platform values.
constexpr size_t NPU_DEFAULT_SPM_BYTES = 512u * 1024u;
// TODO: Replace this placeholder ACC capacity with the real platform value.
constexpr size_t NPU_DEFAULT_ACC_BYTES = 512u * 1024u;
constexpr size_t NPU_DEFAULT_GUARD_BYTES = 4u * 1024u;
constexpr uint32_t NPU_SPM_ALIGNMENT = 32;
constexpr uint32_t NPU_ACC_ALIGNMENT = 4;

enum class npu_memory_space {
    spm,
    acc,
};

struct npu_memory_slice {
    npu_memory_space space = npu_memory_space::spm;
    uint32_t offset = 0;
    uint32_t bytes = 0;
};

struct npu_runtime_layout {
    // Offsets are relative to the beginning of each independent memory space.
    // TODO: If runtime later exposes multiple banks/ports per space, extend this
    // layout to carry the corresponding bank identifier.
    npu_memory_slice activation;
    npu_memory_slice weight;
    npu_memory_slice accumulator;
};

struct npu_tiling_config {
    int64_t sa_rows = NPU_SA_TILE;
    int64_t sa_cols = NPU_SA_TILE;
    size_t spm_bytes = NPU_DEFAULT_SPM_BYTES;
    size_t acc_bytes = NPU_DEFAULT_ACC_BYTES;
    size_t guard_bytes = NPU_DEFAULT_GUARD_BYTES;
    int64_t k_block = 0;
    int64_t stage2_k_block = NPU_STAGE2_K_TILE;
    bool output_in_spm = false;
    npu_runtime_layout layout;
};

struct npu_mn_tile {
    int64_t m0 = 0;
    int64_t n0 = 0;
    int64_t m = 0;
    int64_t n = 0;
};

struct npu_k_tile {
    int64_t k0 = 0;
    int64_t k = 0;
    bool first_k = false;
    bool last_k = false;
    size_t activation_bytes = 0;
    size_t weight_bytes = 0;
};

enum class npu_loop_stage {
    single,
    head,
    body,
    tail,
};

struct npu_exec_tile {
    int64_t m0 = 0;
    int64_t n0 = 0;
    int64_t m = 0;
    int64_t n = 0;
    int64_t k0 = 0;
    int64_t k = 0;
    npu_loop_stage stage = npu_loop_stage::single;
    bool needs_bias = false;
    bool writes_output = false;
    int32_t weight_pack_index = -1;
    int32_t bias_pack_index = -1;
};

struct npu_activation_quant_config {
    bool dynamic = false;
    bool per_tensor = true;
    bool symmetric = false;
    bool valid = false;
    float scale = 1.0f;
    int32_t zero_point = 0;
};

struct npu_aicas_w8a8_config {
    bool valid = false;
    float act_scale = 1.0f;
    int32_t act_scale_q8_24 = 0;
    int32_t act_zero_point_i8 = 0;
    std::vector<float> weight_scale;
    std::vector<int32_t> sum_w;
};

struct npu_prepacked_weight {
    int64_t m0 = 0;
    int64_t m = 0;
    int64_t k0 = 0;
    int64_t k = 0;
    std::vector<float> scales;
    std::vector<int8_t> packed;
};

struct npu_prepacked_bias {
    int64_t m0 = 0;
    int64_t n0 = 0;
    int64_t m = 0;
    std::vector<int32_t> values;
};

struct npu_node_plan {
    const struct ggml_tensor * op = nullptr;
    const struct ggml_tensor * root = nullptr;       // original graph node
    const struct ggml_tensor * src0 = nullptr; // weights in ggml layout [K, M]
    const struct ggml_tensor * src1 = nullptr; // activations in ggml layout [K, N]
    const struct ggml_tensor * bias = nullptr; // optional fused bias
    struct ggml_tensor * dst = nullptr;        // dst in ggml layout [M, N]

    int64_t m = 0;
    int64_t n = 0;
    int64_t k = 0;
    npu_activation_quant_config activation_quant;
    npu_aicas_w8a8_config aicas_w8a8;

    npu_tiling_config config;
    int64_t first_stage_tm = 0;
    int64_t first_stage_tn = 0;
    int64_t first_stage_tk = 0;
    std::vector<npu_mn_tile> mn_tiles;
    std::vector<npu_k_tile> k_tiles;
    std::vector<npu_prepacked_weight> weight_packs;
    std::vector<npu_prepacked_bias> bias_packs;
    std::vector<int32_t> weight_column_sum_q;
    std::vector<npu_exec_tile> exec_tiles;
    std::string summary;
};

struct npu_graph_plan {
    std::vector<npu_node_plan> nodes;
};

bool npu_is_tensor_2d(const struct ggml_tensor * tensor);
bool npu_is_tensor_plain_contiguous(const struct ggml_tensor * tensor);
bool npu_is_supported_weight_type(enum ggml_type type);
bool npu_is_supported_activation_type(enum ggml_type type);
std::string npu_shape_string(const struct ggml_tensor * tensor);

} // namespace ggml_npu
