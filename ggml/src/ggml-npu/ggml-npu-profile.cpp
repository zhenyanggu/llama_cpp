#include "ggml-npu-profile.h"

#include "ggml.h"
#include "npu_runtime.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <unordered_set>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

namespace ggml_npu {

namespace {

struct npu_profile_state {
    std::mutex mutex;
    std::string output_path;
    std::string manifest_path;
    std::vector<npu_profile_node_record> nodes;
};

struct npu_summary_state {
    std::mutex mutex;
    bool active = false;
    ggml_npu_profile_summary summary = {};
};

static npu_profile_state & npu_get_profile_state() {
    static npu_profile_state state;
    return state;
}

static npu_summary_state & npu_get_summary_state() {
    static npu_summary_state state;
    return state;
}

static std::string env_string(const char * key) {
    const char * value = std::getenv(key);
    return value ? value : "";
}

static std::string derive_manifest_path(const std::string & output_path) {
    const std::string override_path = env_string("GGML_NPU_PROFILE_MANIFEST_JSON");
    if (!override_path.empty()) {
        return override_path;
    }

    if (output_path.empty()) {
        return "";
    }

    if (output_path.size() >= 5 && output_path.substr(output_path.size() - 5) == ".json") {
        return output_path.substr(0, output_path.size() - 5) + "_manifest.json";
    }

    return output_path + ".manifest.json";
}

static std::string resolve_output_path() {
    return env_string("GGML_NPU_PROFILE_JSON");
}

static double pct(double numerator, double denominator) {
    if (denominator <= 0.0) {
        return 0.0;
    }
    return numerator / denominator * 100.0;
}

static double profile_accounted_us(const npu_profile_node_record & node) {
    return node.setup_runtime_us_total +
        node.setup_validate_us_total +
        node.setup_buffer_alloc_us_total +
        node.setup_cache_alloc_us_total +
        node.setup_profile_begin_us_total +
        node.activation_pack_us_total +
        node.host_copy_activation_us_total +
        node.host_copy_weight_us_total +
        node.bias_prepare_us_total +
        node.dma_in_pair_us_total +
        node.dma_in_bias_us_total +
        node.w_prefetch_wait_us_total +
        node.a_prefetch_wait_us_total +
        node.o_mvout_wait_us_total +
        node.gemm_us_total +
        node.dma_out_us_total +
        node.postprocess_us_total;
}

static double profile_unaccounted_us(double total_us, double accounted_us) {
    return total_us > accounted_us ? total_us - accounted_us : 0.0;
}

static std::string to_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return (char) std::tolower(c);
    });
    return value;
}

static std::string resolve_profile_level() {
    const std::string level = to_lower(env_string("GGML_NPU_PROFILE_LEVEL"));
    return level.empty() ? "diagnostic" : level;
}

static bool contains_any(const std::string & haystack, const std::vector<std::string> & needles) {
    for (const std::string & needle : needles) {
        if (haystack.find(needle) != std::string::npos) {
            return true;
        }
    }
    return false;
}

static npu_profile_tensor_info capture_tensor_info(const struct ggml_tensor * tensor) {
    npu_profile_tensor_info info;
    if (tensor == nullptr) {
        return info;
    }

    info.present = true;
    info.name = tensor->name;
    info.type = ggml_type_name(tensor->type);
    info.is_quantized = ggml_is_quantized(tensor->type);
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        info.ne[i] = tensor->ne[i];
    }
    return info;
}

static json shape_json(const std::array<int64_t, GGML_MAX_DIMS> & ne) {
    return {ne[0], ne[1], ne[2], ne[3]};
}

static json tensor_json(const npu_profile_tensor_info & info) {
    if (!info.present) {
        return nullptr;
    }

    return {
        {"name", info.name},
        {"type", info.type},
        {"is_quantized", info.is_quantized},
        {"shape", shape_json(info.ne)},
    };
}

static std::string stage_name(npu_loop_stage stage) {
    switch (stage) {
        case npu_loop_stage::single: return "single";
        case npu_loop_stage::head:   return "head";
        case npu_loop_stage::body:   return "body";
        case npu_loop_stage::tail:   return "tail";
    }

    return "unknown";
}

static std::string semantic_op_from_plan(const npu_node_plan & plan) {
    const std::string joined = to_lower(
        std::string(plan.src0 && plan.src0->name[0] ? plan.src0->name : "") + " " +
        std::string(plan.root && plan.root->name[0] ? plan.root->name : "") + " " +
        std::string(plan.dst && plan.dst->name[0] ? plan.dst->name : ""));

    if (contains_any(joined, {"patch_embeddings", "patch_emb", "patch_embedding", "patch_conv"})) {
        return "patch_embedding";
    }
    if (contains_any(joined, {"position_embeddings", "pos_embed", "position_embedding"})) {
        return "position_embedding";
    }
    if (contains_any(joined, {"class_embedding"})) {
        return "class_embedding";
    }
    if (contains_any(joined, {"attn_q", "q_proj", "mm_model_attn_q", ".q.weight", ".q_w"})) {
        return "attn_q";
    }
    if (contains_any(joined, {"attn_k", "k_proj", "mm_model_attn_k", ".k.weight", ".k_w"})) {
        return "attn_k";
    }
    if (contains_any(joined, {"attn_v", "v_proj", "mm_model_attn_v", ".v.weight", ".v_w"})) {
        return "attn_v";
    }
    if (contains_any(joined, {"attn_out_proj", "o_proj", "out_proj", ".o.weight", ".o_w"})) {
        return "attn_out_proj";
    }
    if (contains_any(joined, {"ffn_up", "up_proj", "fc1", "gate_up_proj"})) {
        return "ffn_up";
    }
    if (contains_any(joined, {"ffn_gate", "gate_proj"})) {
        return "ffn_gate";
    }
    if (contains_any(joined, {"ffn_down", "down_proj", "fc2"})) {
        return "ffn_down";
    }
    if (contains_any(joined, {"projection", "projector", "mm_projector"})) {
        return "projector";
    }

    return "unclassified";
}

static json tile_json(const npu_profile_tile_record & tile) {
    return {
        {"tile_index", tile.tile_index},
        {"m0", tile.m0},
        {"n0", tile.n0},
        {"k0", tile.k0},
        {"m", tile.m},
        {"n", tile.n},
        {"k", tile.k},
        {"stage", tile.stage},
        {"needs_bias", tile.needs_bias},
        {"writes_output", tile.writes_output},
        {"weight_pack_index", tile.weight_pack_index},
        {"bias_pack_index", tile.bias_pack_index},
        {"activation_already_in_spm", tile.activation_already_in_spm},
        {"weight_already_in_spm", tile.weight_already_in_spm},
        {"mvin_mask", tile.mvin_mask},
        {"w_bank", tile.w_bank},
        {"a_bank", tile.a_bank},
        {"o_bank", tile.o_bank},
        {"w_prefetch_issued", tile.w_prefetch_issued},
        {"w_prefetch_hit", tile.w_prefetch_hit},
        {"a_prefetch_issued", tile.a_prefetch_issued},
        {"a_prefetch_hit", tile.a_prefetch_hit},
        {"o_mvout_async", tile.o_mvout_async},
        {"activation_bytes", tile.activation_bytes},
        {"weight_bytes", tile.weight_bytes},
        {"bias_bytes", tile.bias_bytes},
        {"acc_readback_bytes", tile.acc_readback_bytes},
        {"output_write_bytes", tile.output_write_bytes},
        {"activation_pack_us", tile.activation_pack_us},
        {"host_copy_activation_us", tile.host_copy_activation_us},
        {"host_copy_weight_us", tile.host_copy_weight_us},
        {"bias_prepare_us", tile.bias_prepare_us},
        {"dma_in_activation_us", tile.dma_in_activation_us},
        {"dma_in_weight_us", tile.dma_in_weight_us},
        {"dma_in_bias_us", tile.dma_in_bias_us},
        {"dma_in_pair_us", tile.dma_in_pair_us},
        {"w_prefetch_wait_us", tile.w_prefetch_wait_us},
        {"a_prefetch_wait_us", tile.a_prefetch_wait_us},
        {"o_mvout_wait_us", tile.o_mvout_wait_us},
        {"o_mvout_hidden_candidate_us", tile.o_mvout_hidden_candidate_us},
        {"gemm_us", tile.gemm_us},
        {"dma_out_us", tile.dma_out_us},
        {"postprocess_us", tile.postprocess_us},
        {"accounted_us", tile.accounted_us},
        {"unaccounted_us", tile.unaccounted_us},
        {"total_us", tile.total_us},
    };
}

static json node_json(const npu_profile_node_record & node, double total_us) {
    json tiles = json::array();
    for (const auto & tile : node.tiles) {
        tiles.push_back(tile_json(tile));
    }

    const double total_dma_in_us = node.dma_in_pair_us_total + node.dma_in_bias_us_total;

    return {
        {"layer_id", node.layer_id},
        {"root_tensor", tensor_json(node.root_tensor)},
        {"compute_tensor", tensor_json(node.compute_tensor)},
        {"weight_tensor", tensor_json(node.weight_tensor)},
        {"activation_tensor", tensor_json(node.activation_tensor)},
        {"bias_tensor", tensor_json(node.bias_tensor)},
        {"dst_tensor", tensor_json(node.dst_tensor)},
        {"root_op_name", node.root_op_name},
        {"compute_op_name", node.compute_op_name},
        {"semantic_op", node.semantic_op},
        {"summary", node.summary},
        {"m", node.m},
        {"n", node.n},
        {"k", node.k},
        {"use_aicas_w8a8", node.use_aicas_w8a8},
        {"shape_table_hit", node.shape_table_hit},
        {"full_pingpong_enabled", node.full_pingpong_enabled},
        {"a_prefetch_enabled", node.a_prefetch_enabled},
        {"o_overlap_enabled", node.o_overlap_enabled},
        {"full_output_cma_active", node.full_output_cma_active},
        {"shape_table_source", node.shape_table_source.empty() ? nullptr : json(node.shape_table_source)},
        {"activation_scale", node.activation_scale},
        {"activation_zero_point", node.activation_zero_point},
        {"first_stage_tm", node.first_stage_tm},
        {"first_stage_tn", node.first_stage_tn},
        {"first_stage_tk", node.first_stage_tk},
        {"stage2_k_block", node.stage2_k_block},
        {"exec_tile_count", node.exec_tile_count},
        {"weight_pack_count", node.weight_pack_count},
        {"bias_pack_count", node.bias_pack_count},
        {"spm_bytes", node.spm_bytes},
        {"acc_bytes", node.acc_bytes},
        {"activation_offset", node.activation_offset},
        {"weight_offset", node.weight_offset},
        {"accumulator_offset", node.accumulator_offset},
        {"bias_accumulator_offset", node.bias_accumulator_offset},
        {"output_accumulator_offset", node.output_accumulator_offset},
        {"activation_pack_calls", node.activation_pack_calls},
        {"activation_pack_async_jobs", node.activation_pack_async_jobs},
        {"activation_pack_async_hits", node.activation_pack_async_hits},
        {"host_copy_activation_calls", node.host_copy_activation_calls},
        {"host_copy_weight_calls", node.host_copy_weight_calls},
        {"bias_prepare_calls", node.bias_prepare_calls},
        {"dma_in_activation_calls", node.dma_in_activation_calls},
        {"dma_in_weight_calls", node.dma_in_weight_calls},
        {"dma_in_bias_calls", node.dma_in_bias_calls},
        {"dma_in_pair_calls", node.dma_in_pair_calls},
        {"spm_activation_reuse_hits", node.spm_activation_reuse_hits},
        {"spm_weight_reuse_hits", node.spm_weight_reuse_hits},
        {"gemm_calls", node.gemm_calls},
        {"gemm_plan_calls", node.gemm_plan_calls},
        {"dma_out_calls", node.dma_out_calls},
        {"postprocess_calls", node.postprocess_calls},
        {"raw_acc_mvout_nodes", node.raw_acc_mvout_nodes},
        {"raw_acc_mvout_tiles", node.raw_acc_mvout_tiles},
        {"w_prefetch_calls", node.w_prefetch_calls},
        {"w_prefetch_hits", node.w_prefetch_hits},
        {"w_prefetch_conflicts", node.w_prefetch_conflicts},
        {"a_prefetch_calls", node.a_prefetch_calls},
        {"a_prefetch_hits", node.a_prefetch_hits},
        {"a_prefetch_conflicts", node.a_prefetch_conflicts},
        {"o_mvout_async_calls", node.o_mvout_async_calls},
        {"packed_activation_bytes_total", node.packed_activation_bytes_total},
        {"copied_weight_bytes_total", node.copied_weight_bytes_total},
        {"dma_in_activation_bytes_total", node.dma_in_activation_bytes_total},
        {"dma_in_weight_bytes_total", node.dma_in_weight_bytes_total},
        {"bias_bytes_total", node.bias_bytes_total},
        {"acc_readback_bytes_total", node.acc_readback_bytes_total},
        {"output_write_bytes_total", node.output_write_bytes_total},
        {"setup_runtime_us_total", node.setup_runtime_us_total},
        {"setup_validate_us_total", node.setup_validate_us_total},
        {"setup_buffer_alloc_us_total", node.setup_buffer_alloc_us_total},
        {"setup_cache_alloc_us_total", node.setup_cache_alloc_us_total},
        {"setup_profile_begin_us_total", node.setup_profile_begin_us_total},
        {"activation_pack_us_total", node.activation_pack_us_total},
        {"activation_pack_async_us_total", node.activation_pack_async_us_total},
        {"activation_pack_wait_us_total", node.activation_pack_wait_us_total},
        {"host_copy_activation_us_total", node.host_copy_activation_us_total},
        {"host_copy_weight_us_total", node.host_copy_weight_us_total},
        {"bias_prepare_us_total", node.bias_prepare_us_total},
        {"dma_in_activation_us_total", node.dma_in_activation_us_total},
        {"dma_in_weight_us_total", node.dma_in_weight_us_total},
        {"dma_in_bias_us_total", node.dma_in_bias_us_total},
        {"dma_in_pair_us_total", node.dma_in_pair_us_total},
        {"w_prefetch_wait_us_total", node.w_prefetch_wait_us_total},
        {"w_prefetch_hidden_candidate_us_total", node.w_prefetch_hidden_candidate_us_total},
        {"a_prefetch_wait_us_total", node.a_prefetch_wait_us_total},
        {"a_prefetch_hidden_candidate_us_total", node.a_prefetch_hidden_candidate_us_total},
        {"o_mvout_wait_us_total", node.o_mvout_wait_us_total},
        {"o_mvout_hidden_candidate_us_total", node.o_mvout_hidden_candidate_us_total},
        {"dma_in_total_us", total_dma_in_us},
        {"gemm_us_total", node.gemm_us_total},
        {"dma_out_us_total", node.dma_out_us_total},
        {"postprocess_us_total", node.postprocess_us_total},
        {"accounted_us_total", node.accounted_us_total},
        {"unaccounted_us_total", node.unaccounted_us_total},
        {"accounted_share_pct", pct(node.accounted_us_total, node.total_node_us)},
        {"total_node_us", node.total_node_us},
        {"share_of_profile_time_pct", pct(node.total_node_us, total_us)},
        {"status", node.status},
        {"error", node.error.empty() ? nullptr : json(node.error)},
        {"tiles", tiles},
    };
}

static json node_json_compact(const npu_profile_node_record & node, double total_us) {
    return {
        {"layer_id", node.layer_id},
        {"semantic_op", node.semantic_op},
        {"compute_op_name", node.compute_op_name},
        {"root_name", node.root_tensor.present ? node.root_tensor.name : ""},
        {"weight_name", node.weight_tensor.present ? node.weight_tensor.name : ""},
        {"m", node.m},
        {"n", node.n},
        {"k", node.k},
        {"shape_table_hit", node.shape_table_hit},
        {"full_pingpong_enabled", node.full_pingpong_enabled},
        {"a_prefetch_enabled", node.a_prefetch_enabled},
        {"o_overlap_enabled", node.o_overlap_enabled},
        {"full_output_cma_active", node.full_output_cma_active},
        {"first_stage_tm", node.first_stage_tm},
        {"first_stage_tn", node.first_stage_tn},
        {"first_stage_tk", node.first_stage_tk},
        {"stage2_k_block", node.stage2_k_block},
        {"exec_tile_count", node.exec_tile_count},
        {"weight_pack_count", node.weight_pack_count},
        {"bias_pack_count", node.bias_pack_count},
        {"setup_runtime_us_total", node.setup_runtime_us_total},
        {"setup_validate_us_total", node.setup_validate_us_total},
        {"setup_buffer_alloc_us_total", node.setup_buffer_alloc_us_total},
        {"setup_cache_alloc_us_total", node.setup_cache_alloc_us_total},
        {"setup_profile_begin_us_total", node.setup_profile_begin_us_total},
        {"activation_pack_us_total", node.activation_pack_us_total},
        {"activation_pack_async_us_total", node.activation_pack_async_us_total},
        {"activation_pack_wait_us_total", node.activation_pack_wait_us_total},
        {"activation_pack_async_jobs", node.activation_pack_async_jobs},
        {"activation_pack_async_hits", node.activation_pack_async_hits},
        {"host_copy_activation_us_total", node.host_copy_activation_us_total},
        {"host_copy_weight_us_total", node.host_copy_weight_us_total},
        {"bias_prepare_us_total", node.bias_prepare_us_total},
        {"dma_in_pair_us_total", node.dma_in_pair_us_total},
        {"dma_in_activation_calls", node.dma_in_activation_calls},
        {"dma_in_weight_calls", node.dma_in_weight_calls},
        {"dma_in_bias_calls", node.dma_in_bias_calls},
        {"dma_in_pair_calls", node.dma_in_pair_calls},
        {"spm_activation_reuse_hits", node.spm_activation_reuse_hits},
        {"spm_weight_reuse_hits", node.spm_weight_reuse_hits},
        {"dma_in_activation_bytes_total", node.dma_in_activation_bytes_total},
        {"dma_in_weight_bytes_total", node.dma_in_weight_bytes_total},
        {"w_prefetch_calls", node.w_prefetch_calls},
        {"w_prefetch_hits", node.w_prefetch_hits},
        {"w_prefetch_conflicts", node.w_prefetch_conflicts},
        {"w_prefetch_wait_us_total", node.w_prefetch_wait_us_total},
        {"w_prefetch_hidden_candidate_us_total", node.w_prefetch_hidden_candidate_us_total},
        {"a_prefetch_calls", node.a_prefetch_calls},
        {"a_prefetch_hits", node.a_prefetch_hits},
        {"a_prefetch_conflicts", node.a_prefetch_conflicts},
        {"a_prefetch_wait_us_total", node.a_prefetch_wait_us_total},
        {"a_prefetch_hidden_candidate_us_total", node.a_prefetch_hidden_candidate_us_total},
        {"o_mvout_async_calls", node.o_mvout_async_calls},
        {"o_mvout_wait_us_total", node.o_mvout_wait_us_total},
        {"o_mvout_hidden_candidate_us_total", node.o_mvout_hidden_candidate_us_total},
        {"raw_acc_mvout_nodes", node.raw_acc_mvout_nodes},
        {"raw_acc_mvout_tiles", node.raw_acc_mvout_tiles},
        {"dma_in_bias_us_total", node.dma_in_bias_us_total},
        {"gemm_us_total", node.gemm_us_total},
        {"dma_out_us_total", node.dma_out_us_total},
        {"postprocess_us_total", node.postprocess_us_total},
        {"packed_activation_bytes_total", node.packed_activation_bytes_total},
        {"copied_weight_bytes_total", node.copied_weight_bytes_total},
        {"acc_readback_bytes_total", node.acc_readback_bytes_total},
        {"output_write_bytes_total", node.output_write_bytes_total},
        {"total_node_us", node.total_node_us},
        {"accounted_us_total", node.accounted_us_total},
        {"unaccounted_us_total", node.unaccounted_us_total},
        {"gemm_calls", node.gemm_calls},
        {"gemm_plan_calls", node.gemm_plan_calls},
        {"share_of_profile_time_pct", pct(node.total_node_us, total_us)},
        {"status", node.status},
        {"error", node.error.empty() ? nullptr : json(node.error)},
    };
}

static std::string shape_signature(const npu_profile_node_record & node) {
    std::ostringstream oss;
    oss << (node.compute_op_name.empty() ? "MUL_MAT" : node.compute_op_name)
        << ":" << node.m
        << "x" << node.n
        << "x" << node.k;
    return oss.str();
}

static std::vector<npu_profile_node_record> sample_nodes_by_shape(
    const std::vector<npu_profile_node_record> & nodes) {
    std::vector<npu_profile_node_record> sampled;
    sampled.reserve(nodes.size());
    std::unordered_set<std::string> seen;
    seen.reserve(nodes.size());

    for (const auto & node : nodes) {
        const std::string sig = shape_signature(node);
        if (seen.insert(sig).second) {
            sampled.push_back(node);
        }
    }

    return sampled;
}

static std::vector<npu_profile_node_record> sample_nodes_first_layer(
    const std::vector<npu_profile_node_record> & nodes) {
    if (nodes.empty()) {
        return {};
    }

    const int64_t target_layer = nodes.front().layer_id;
    std::vector<npu_profile_node_record> sampled;
    for (const auto & node : nodes) {
        if (node.layer_id == target_layer) {
            sampled.push_back(node);
        }
    }
    return sampled;
}

static json manifest_json(const std::vector<npu_profile_node_record> & nodes) {
    json layers = json::array();

    for (const auto & node : nodes) {
        json fused_ops = json::array();
        fused_ops.push_back(node.compute_op_name.empty() ? "MUL_MAT" : node.compute_op_name);
        if (node.bias_tensor.present) {
            fused_ops.push_back("ADD");
        }

        std::string layer_name = node.root_tensor.present && !node.root_tensor.name.empty()
            ? node.root_tensor.name
            : (node.weight_tensor.present ? node.weight_tensor.name : "");
        if (layer_name.empty() && node.dst_tensor.present) {
            layer_name = node.dst_tensor.name;
        }

        layers.push_back({
            {"layer_id", node.layer_id},
            {"onnx_node_name", layer_name},
            {"origin_op_type", node.semantic_op.empty() ? node.compute_op_name : node.semantic_op},
            {"device", "npu"},
            {"fused_ops", fused_ops},
            {"fallback_reason", nullptr},
        });
    }

    return {{"layers", layers}};
}

} // namespace

bool npu_profile_enabled() {
    return !resolve_output_path().empty();
}

npu_profile_node_record npu_profile_init_node_record(int64_t layer_id, const npu_node_plan & plan) {
    npu_profile_node_record record;
    record.layer_id = layer_id;
    record.root_tensor = capture_tensor_info(plan.root);
    record.compute_tensor = capture_tensor_info(plan.op);
    record.weight_tensor = capture_tensor_info(plan.src0);
    record.activation_tensor = capture_tensor_info(plan.src1);
    record.bias_tensor = capture_tensor_info(plan.bias);
    record.dst_tensor = capture_tensor_info(plan.dst);
    record.root_op_name = plan.root ? ggml_op_name(plan.root->op) : "";
    record.compute_op_name = plan.op ? ggml_op_name(plan.op->op) : "";
    record.semantic_op = semantic_op_from_plan(plan);
    record.summary = plan.summary;
    record.m = plan.m;
    record.n = plan.n;
    record.k = plan.k;
    record.use_aicas_w8a8 = plan.aicas_w8a8.valid;
    record.shape_table_hit = plan.shape_table_hit;
    record.shape_table_source = plan.shape_table_source;
    record.activation_scale = plan.activation_quant.scale;
    record.activation_zero_point = plan.activation_quant.zero_point;
    record.first_stage_tm = plan.first_stage_tm;
    record.first_stage_tn = plan.first_stage_tn;
    record.first_stage_tk = plan.first_stage_tk;
    record.stage2_k_block = plan.config.stage2_k_block;
    record.exec_tile_count = static_cast<int64_t>(plan.exec_tiles.size());
    record.weight_pack_count = static_cast<int64_t>(plan.weight_packs.size());
    record.bias_pack_count = static_cast<int64_t>(plan.bias_packs.size());
    record.spm_bytes = plan.config.spm_bytes;
    record.acc_bytes = plan.config.acc_bytes;
    record.activation_offset = plan.config.layout.activation.offset;
    record.weight_offset = plan.config.layout.weight.offset;
    record.accumulator_offset = plan.config.layout.output_accumulator.offset;
    record.bias_accumulator_offset = plan.config.layout.bias_accumulator.offset;
    record.output_accumulator_offset = plan.config.layout.output_accumulator.offset;
    return record;
}

void npu_profile_reset() {
    if (!npu_profile_enabled()) {
        return;
    }

    npu_profile_state & state = npu_get_profile_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.output_path = resolve_output_path();
    state.manifest_path = derive_manifest_path(state.output_path);
    state.nodes.clear();
}

void npu_profile_add_node_record(npu_profile_node_record record) {
    if (!npu_profile_enabled()) {
        return;
    }

    npu_profile_state & state = npu_get_profile_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.output_path = resolve_output_path();
    state.manifest_path = derive_manifest_path(state.output_path);
    state.nodes.push_back(std::move(record));
}

void npu_profile_flush() {
    if (!npu_profile_enabled()) {
        return;
    }

    npu_profile_state & state = npu_get_profile_state();
    std::vector<npu_profile_node_record> snapshot;
    std::string output_path;
    std::string manifest_path;
    std::string profile_level;
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        output_path = resolve_output_path();
        manifest_path = derive_manifest_path(output_path);
        profile_level = resolve_profile_level();
        state.output_path = output_path;
        state.manifest_path = manifest_path;
        snapshot = state.nodes;
    }

    if (output_path.empty()) {
        return;
    }

    double total_us = 0.0;
    double total_setup_runtime_us = 0.0;
    double total_setup_validate_us = 0.0;
    double total_setup_buffer_alloc_us = 0.0;
    double total_setup_cache_alloc_us = 0.0;
    double total_setup_profile_begin_us = 0.0;
    double total_activation_pack_us = 0.0;
    double total_activation_pack_async_us = 0.0;
    double total_activation_pack_wait_us = 0.0;
    double total_host_copy_activation_us = 0.0;
    double total_host_copy_weight_us = 0.0;
    double total_bias_prepare_us = 0.0;
    double total_dma_in_pair_us = 0.0;
    double total_dma_in_bias_us = 0.0;
    double total_w_prefetch_wait_us = 0.0;
    double total_w_prefetch_hidden_candidate_us = 0.0;
    double total_a_prefetch_wait_us = 0.0;
    double total_a_prefetch_hidden_candidate_us = 0.0;
    double total_o_mvout_wait_us = 0.0;
    double total_o_mvout_hidden_candidate_us = 0.0;
    double total_gemm_us = 0.0;
    double total_dma_out_us = 0.0;
    double total_postprocess_us = 0.0;
    int64_t total_bias_prepare_calls = 0;
    int64_t total_activation_pack_async_jobs = 0;
    int64_t total_activation_pack_async_hits = 0;
    int64_t total_dma_in_activation_calls = 0;
    int64_t total_dma_in_weight_calls = 0;
    int64_t total_dma_in_pair_calls = 0;
    int64_t total_dma_in_bias_calls = 0;
    int64_t total_spm_activation_reuse_hits = 0;
    int64_t total_spm_weight_reuse_hits = 0;
    int64_t total_raw_acc_mvout_nodes = 0;
    int64_t total_raw_acc_mvout_tiles = 0;
    int64_t total_w_prefetch_calls = 0;
    int64_t total_w_prefetch_hits = 0;
    int64_t total_w_prefetch_conflicts = 0;
    int64_t total_full_pingpong_nodes = 0;
    int64_t total_a_prefetch_enabled_nodes = 0;
    int64_t total_o_overlap_enabled_nodes = 0;
    int64_t total_full_output_cma_nodes = 0;
    int64_t total_a_prefetch_calls = 0;
    int64_t total_a_prefetch_hits = 0;
    int64_t total_a_prefetch_conflicts = 0;
    int64_t total_o_mvout_async_calls = 0;
    int64_t total_dma_in_activation_bytes = 0;
    int64_t total_dma_in_weight_bytes = 0;

    std::sort(snapshot.begin(), snapshot.end(), [](const auto & lhs, const auto & rhs) {
        return lhs.layer_id < rhs.layer_id;
    });

    for (auto & node : snapshot) {
        if (node.accounted_us_total <= 0.0) {
            node.accounted_us_total = profile_accounted_us(node);
        }
        node.unaccounted_us_total = profile_unaccounted_us(node.total_node_us, node.accounted_us_total);
    }

    double total_accounted_us = 0.0;
    double total_unaccounted_us = 0.0;
    int64_t total_gemm_plan_calls = 0;
    for (const auto & node : snapshot) {
        total_us += node.total_node_us;
        total_setup_runtime_us += node.setup_runtime_us_total;
        total_setup_validate_us += node.setup_validate_us_total;
        total_setup_buffer_alloc_us += node.setup_buffer_alloc_us_total;
        total_setup_cache_alloc_us += node.setup_cache_alloc_us_total;
        total_setup_profile_begin_us += node.setup_profile_begin_us_total;
        total_activation_pack_us += node.activation_pack_us_total;
        total_activation_pack_async_us += node.activation_pack_async_us_total;
        total_activation_pack_wait_us += node.activation_pack_wait_us_total;
        total_host_copy_activation_us += node.host_copy_activation_us_total;
        total_host_copy_weight_us += node.host_copy_weight_us_total;
        total_bias_prepare_us += node.bias_prepare_us_total;
        total_dma_in_pair_us += node.dma_in_pair_us_total;
        total_dma_in_bias_us += node.dma_in_bias_us_total;
        total_w_prefetch_wait_us += node.w_prefetch_wait_us_total;
        total_w_prefetch_hidden_candidate_us += node.w_prefetch_hidden_candidate_us_total;
        total_a_prefetch_wait_us += node.a_prefetch_wait_us_total;
        total_a_prefetch_hidden_candidate_us += node.a_prefetch_hidden_candidate_us_total;
        total_o_mvout_wait_us += node.o_mvout_wait_us_total;
        total_o_mvout_hidden_candidate_us += node.o_mvout_hidden_candidate_us_total;
        total_gemm_us += node.gemm_us_total;
        total_dma_out_us += node.dma_out_us_total;
        total_postprocess_us += node.postprocess_us_total;
        total_accounted_us += node.accounted_us_total;
        total_unaccounted_us += node.unaccounted_us_total;
        total_activation_pack_async_jobs += node.activation_pack_async_jobs;
        total_activation_pack_async_hits += node.activation_pack_async_hits;
        total_bias_prepare_calls += node.bias_prepare_calls;
        total_dma_in_activation_calls += node.dma_in_activation_calls;
        total_dma_in_weight_calls += node.dma_in_weight_calls;
        total_dma_in_pair_calls += node.dma_in_pair_calls;
        total_dma_in_bias_calls += node.dma_in_bias_calls;
        total_spm_activation_reuse_hits += node.spm_activation_reuse_hits;
        total_spm_weight_reuse_hits += node.spm_weight_reuse_hits;
        total_raw_acc_mvout_nodes += node.raw_acc_mvout_nodes;
        total_raw_acc_mvout_tiles += node.raw_acc_mvout_tiles;
        total_w_prefetch_calls += node.w_prefetch_calls;
        total_w_prefetch_hits += node.w_prefetch_hits;
        total_w_prefetch_conflicts += node.w_prefetch_conflicts;
        total_full_pingpong_nodes += node.full_pingpong_enabled ? 1 : 0;
        total_a_prefetch_enabled_nodes += node.a_prefetch_enabled ? 1 : 0;
        total_o_overlap_enabled_nodes += node.o_overlap_enabled ? 1 : 0;
        total_full_output_cma_nodes += node.full_output_cma_active ? 1 : 0;
        total_a_prefetch_calls += node.a_prefetch_calls;
        total_a_prefetch_hits += node.a_prefetch_hits;
        total_a_prefetch_conflicts += node.a_prefetch_conflicts;
        total_o_mvout_async_calls += node.o_mvout_async_calls;
        total_dma_in_activation_bytes += node.dma_in_activation_bytes_total;
        total_dma_in_weight_bytes += node.dma_in_weight_bytes_total;
        total_gemm_plan_calls += node.gemm_plan_calls;
    }

    std::vector<npu_profile_node_record> hot_nodes = snapshot;
    std::sort(hot_nodes.begin(), hot_nodes.end(), [](const auto & lhs, const auto & rhs) {
        if (lhs.total_node_us != rhs.total_node_us) {
            return lhs.total_node_us > rhs.total_node_us;
        }
        return lhs.layer_id < rhs.layer_id;
    });

    json hot = json::array();
    for (size_t i = 0; i < hot_nodes.size() && i < 10; ++i) {
        hot.push_back({
            {"layer_id", hot_nodes[i].layer_id},
            {"semantic_op", hot_nodes[i].semantic_op},
            {"root_name", hot_nodes[i].root_tensor.present ? hot_nodes[i].root_tensor.name : ""},
            {"weight_name", hot_nodes[i].weight_tensor.present ? hot_nodes[i].weight_tensor.name : ""},
            {"total_node_us", hot_nodes[i].total_node_us},
        });
    }

    const bool diagnostic_level = profile_level == "diagnostic";
    const bool shape_sample = profile_level == "shape";
    const bool first_layer_sample = profile_level == "layer" || profile_level == "first_layer";
    const bool compact_level = profile_level == "compact" || diagnostic_level;
    std::vector<npu_profile_node_record> node_records = snapshot;
    if (shape_sample) {
        node_records = sample_nodes_by_shape(snapshot);
    } else if (first_layer_sample) {
        node_records = sample_nodes_first_layer(snapshot);
    }

    json nodes = json::array();
    for (const auto & node : node_records) {
        nodes.push_back(compact_level ? node_json_compact(node, total_us) : node_json(node, total_us));
    }

    json out = {
        {"profile_kind", "ggml_npu_node_trace"},
        {"profile_level", profile_level},
        {"timing_unit", "us"},
        {"note", diagnostic_level
            ? "Diagnostic profile keeps per-node aggregate timings and omits per-tile records to reduce profiling overhead. DMA-in uses additive dma_in_pair_us for dual-DMA launches; activation/weight split is bytes/calls only unless an estimated field is present."
            : (shape_sample
            ? "Node-level wall-clock timings captured around ggml-npu execution; nodes[] is sampled to one record per (op,m,n,k) shape to reduce profiling overhead and JSON size."
            : (first_layer_sample
                ? "Node-level wall-clock timings captured around ggml-npu execution; nodes[] keeps only one layer (the first layer_id in execution order) to reduce profiling overhead and JSON size."
                : (compact_level
                    ? "Compact profile keeps summary/by_semantic_op/hot_nodes and minimal per-node fields (layer/op/shape/time share), omitting tile/tensor/call/byte details."
                    : "Per-node and per-tile wall-clock timings captured around ggml-npu execution. DMA/GEMM timings here are user-space call latencies; NPU runtime profile JSON provides the lower-level stage split including wait_irq.")))},
        {"node_count", snapshot.size()},
        {"recorded_node_count", node_records.size()},
        {"summary", {
            {"total_node_us", total_us},
            {"total_setup_runtime_us", total_setup_runtime_us},
            {"total_setup_validate_us", total_setup_validate_us},
            {"total_setup_buffer_alloc_us", total_setup_buffer_alloc_us},
            {"total_setup_cache_alloc_us", total_setup_cache_alloc_us},
            {"total_setup_profile_begin_us", total_setup_profile_begin_us},
            {"total_activation_pack_us", total_activation_pack_us},
            {"total_activation_pack_async_us", total_activation_pack_async_us},
            {"total_activation_pack_wait_us", total_activation_pack_wait_us},
            {"total_activation_pack_async_jobs", total_activation_pack_async_jobs},
            {"total_activation_pack_async_hits", total_activation_pack_async_hits},
            {"total_host_copy_activation_us", total_host_copy_activation_us},
            {"total_host_copy_weight_us", total_host_copy_weight_us},
            {"total_bias_prepare_us", total_bias_prepare_us},
            {"total_bias_prepare_calls", total_bias_prepare_calls},
            {"total_dma_in_us", total_dma_in_pair_us + total_dma_in_bias_us},
            {"total_dma_in_pair_us", total_dma_in_pair_us},
            {"total_dma_in_pair_calls", total_dma_in_pair_calls},
            {"total_dma_in_activation_calls", total_dma_in_activation_calls},
            {"total_dma_in_weight_calls", total_dma_in_weight_calls},
            {"total_dma_in_bias_us", total_dma_in_bias_us},
            {"total_dma_in_bias_calls", total_dma_in_bias_calls},
            {"total_dma_in_activation_bytes", total_dma_in_activation_bytes},
            {"total_dma_in_weight_bytes", total_dma_in_weight_bytes},
            {"total_spm_activation_reuse_hits", total_spm_activation_reuse_hits},
            {"total_spm_weight_reuse_hits", total_spm_weight_reuse_hits},
            {"total_mvinbias_us", total_dma_in_bias_us},
            {"total_mvinbias_calls", total_dma_in_bias_calls},
            {"total_w_prefetch_calls", total_w_prefetch_calls},
            {"total_w_prefetch_hits", total_w_prefetch_hits},
            {"total_w_prefetch_conflicts", total_w_prefetch_conflicts},
            {"total_w_prefetch_wait_us", total_w_prefetch_wait_us},
            {"total_w_prefetch_hidden_candidate_us", total_w_prefetch_hidden_candidate_us},
            {"total_full_pingpong_nodes", total_full_pingpong_nodes},
            {"total_a_prefetch_enabled_nodes", total_a_prefetch_enabled_nodes},
            {"total_o_overlap_enabled_nodes", total_o_overlap_enabled_nodes},
            {"total_full_output_cma_nodes", total_full_output_cma_nodes},
            {"total_a_prefetch_calls", total_a_prefetch_calls},
            {"total_a_prefetch_hits", total_a_prefetch_hits},
            {"total_a_prefetch_conflicts", total_a_prefetch_conflicts},
            {"total_a_prefetch_wait_us", total_a_prefetch_wait_us},
            {"total_a_prefetch_hidden_candidate_us", total_a_prefetch_hidden_candidate_us},
            {"total_o_mvout_async_calls", total_o_mvout_async_calls},
            {"total_o_mvout_wait_us", total_o_mvout_wait_us},
            {"total_o_mvout_hidden_candidate_us", total_o_mvout_hidden_candidate_us},
            {"total_gemm_us", total_gemm_us},
            {"total_gemm_plan_calls", total_gemm_plan_calls},
            {"total_dma_out_us", total_dma_out_us},
            {"total_postprocess_us", total_postprocess_us},
            {"total_raw_acc_mvout_nodes", total_raw_acc_mvout_nodes},
            {"total_raw_acc_mvout_tiles", total_raw_acc_mvout_tiles},
            {"total_accounted_us", total_accounted_us},
            {"total_unaccounted_us", total_unaccounted_us},
            {"accounted_share_pct", pct(total_accounted_us, total_us)},
            {"hot_nodes", hot},
        }},
        {"nodes", nodes},
    };

    std::ofstream out_stream(output_path);
    if (out_stream.is_open()) {
        out_stream << out.dump(2) << '\n';
    }

    if (!manifest_path.empty()) {
        std::ofstream manifest_stream(manifest_path);
        if (manifest_stream.is_open()) {
            manifest_stream << manifest_json(snapshot).dump(2) << '\n';
        }
    }
}

bool npu_summary_active() {
    npu_summary_state & state = npu_get_summary_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    return state.active;
}

void npu_summary_session_start() {
    npu_summary_state & state = npu_get_summary_state();
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.active = true;
        state.summary = {};
    }

    npu_profile_reset_summary();
}

void npu_summary_session_stop(ggml_npu_profile_summary * out) {
    ggml_npu_profile_summary snapshot = {};
    {
        npu_summary_state & state = npu_get_summary_state();
        std::lock_guard<std::mutex> lock(state.mutex);
        snapshot = state.summary;
        snapshot.valid = 1;
        state.active = false;
        state.summary = {};
    }

    npu_profile_runtime_summary runtime = {};
    npu_profile_get_summary(&runtime);

    snapshot.runtime_total_us = static_cast<int64_t>(runtime.total_ns / 1000);
    snapshot.runtime_dma_in_us = static_cast<int64_t>(runtime.dma_in_ns / 1000);
    snapshot.runtime_compute_us = static_cast<int64_t>(runtime.compute_ns / 1000);
    snapshot.runtime_compute_exclusive_us = static_cast<int64_t>((runtime.compute_ns > runtime.wait_irq_ns ? runtime.compute_ns - runtime.wait_irq_ns : 0) / 1000);
    snapshot.runtime_dma_out_us = static_cast<int64_t>(runtime.dma_out_ns / 1000);
    snapshot.runtime_layout_us = static_cast<int64_t>(runtime.layout_ns / 1000);
    snapshot.runtime_layout_exclusive_us = static_cast<int64_t>((runtime.layout_ns > runtime.wait_irq_ns ? runtime.layout_ns - runtime.wait_irq_ns : 0) / 1000);
    snapshot.runtime_wait_irq_us = static_cast<int64_t>(runtime.wait_irq_ns / 1000);
    snapshot.runtime_mvin_calls = static_cast<int64_t>(runtime.mvin_calls);
    snapshot.runtime_compute_calls = static_cast<int64_t>(runtime.compute_calls);
    snapshot.runtime_gemm_plan_calls = static_cast<int64_t>(runtime.gemm_plan_calls);
    snapshot.runtime_mvout_calls = static_cast<int64_t>(runtime.mvout_calls);
    snapshot.runtime_layout_calls = static_cast<int64_t>(runtime.layout_calls);

    // In dual-DMA runs, runtime dma_in_ns may report 0 despite active DMA work.
    // Fall back to profiled per-node DMA-in totals when we have mvin activity.
    if (snapshot.runtime_dma_in_us == 0) {
        const int64_t profiled_dma_in_us =
            snapshot.dma_in_pair_us_total +
            snapshot.dma_in_bias_us_total;
        if (profiled_dma_in_us > 0) {
            snapshot.runtime_dma_in_us = profiled_dma_in_us;
        }
    }

    if (out != nullptr) {
        *out = snapshot;
    }
}

void npu_summary_add_delta(const npu_profile_summary_delta & delta) {
    npu_summary_state & state = npu_get_summary_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.active) {
        return;
    }

    ggml_npu_profile_summary & summary = state.summary;
    summary.used_npu = 1;
    summary.node_count += delta.node_count;
    summary.exec_tile_count += delta.exec_tile_count;
    summary.weight_pack_count += delta.weight_pack_count;
    summary.bias_pack_count += delta.bias_pack_count;
    summary.activation_pack_calls += delta.activation_pack_calls;
    summary.activation_pack_async_jobs += delta.activation_pack_async_jobs;
    summary.activation_pack_async_hits += delta.activation_pack_async_hits;
    summary.host_copy_activation_calls += delta.host_copy_activation_calls;
    summary.host_copy_weight_calls += delta.host_copy_weight_calls;
    summary.bias_prepare_calls += delta.bias_prepare_calls;
    summary.dma_in_activation_calls += delta.dma_in_activation_calls;
    summary.dma_in_weight_calls += delta.dma_in_weight_calls;
    summary.dma_in_bias_calls += delta.dma_in_bias_calls;
    summary.dma_in_pair_calls += delta.dma_in_pair_calls;
    summary.spm_activation_reuse_hits += delta.spm_activation_reuse_hits;
    summary.spm_weight_reuse_hits += delta.spm_weight_reuse_hits;
    summary.gemm_calls += delta.gemm_calls;
    summary.gemm_plan_calls += delta.gemm_plan_calls;
    summary.dma_out_calls += delta.dma_out_calls;
    summary.postprocess_calls += delta.postprocess_calls;
    summary.raw_acc_mvout_nodes += delta.raw_acc_mvout_nodes;
    summary.raw_acc_mvout_tiles += delta.raw_acc_mvout_tiles;
    summary.w_prefetch_calls += delta.w_prefetch_calls;
    summary.w_prefetch_hits += delta.w_prefetch_hits;
    summary.w_prefetch_conflicts += delta.w_prefetch_conflicts;
    summary.full_pingpong_nodes += delta.full_pingpong_nodes;
    summary.a_prefetch_enabled_nodes += delta.a_prefetch_enabled_nodes;
    summary.o_overlap_enabled_nodes += delta.o_overlap_enabled_nodes;
    summary.full_output_cma_nodes += delta.full_output_cma_nodes;
    summary.a_prefetch_calls += delta.a_prefetch_calls;
    summary.a_prefetch_hits += delta.a_prefetch_hits;
    summary.a_prefetch_conflicts += delta.a_prefetch_conflicts;
    summary.o_mvout_async_calls += delta.o_mvout_async_calls;
    summary.packed_activation_bytes_total += delta.packed_activation_bytes_total;
    summary.copied_weight_bytes_total += delta.copied_weight_bytes_total;
    summary.dma_in_activation_bytes_total += delta.dma_in_activation_bytes_total;
    summary.dma_in_weight_bytes_total += delta.dma_in_weight_bytes_total;
    summary.bias_bytes_total += delta.bias_bytes_total;
    summary.acc_readback_bytes_total += delta.acc_readback_bytes_total;
    summary.output_write_bytes_total += delta.output_write_bytes_total;
    summary.total_node_us += delta.total_node_us;
    summary.setup_runtime_us_total += delta.setup_runtime_us_total;
    summary.setup_validate_us_total += delta.setup_validate_us_total;
    summary.setup_buffer_alloc_us_total += delta.setup_buffer_alloc_us_total;
    summary.setup_cache_alloc_us_total += delta.setup_cache_alloc_us_total;
    summary.setup_profile_begin_us_total += delta.setup_profile_begin_us_total;
    summary.activation_pack_us_total += delta.activation_pack_us_total;
    summary.activation_pack_async_us_total += delta.activation_pack_async_us_total;
    summary.activation_pack_wait_us_total += delta.activation_pack_wait_us_total;
    summary.host_copy_activation_us_total += delta.host_copy_activation_us_total;
    summary.host_copy_weight_us_total += delta.host_copy_weight_us_total;
    summary.bias_prepare_us_total += delta.bias_prepare_us_total;
    summary.dma_in_activation_us_total += delta.dma_in_activation_us_total;
    summary.dma_in_weight_us_total += delta.dma_in_weight_us_total;
    summary.dma_in_bias_us_total += delta.dma_in_bias_us_total;
    summary.dma_in_pair_us_total += delta.dma_in_pair_us_total;
    summary.w_prefetch_wait_us_total += delta.w_prefetch_wait_us_total;
    summary.w_prefetch_hidden_candidate_us_total += delta.w_prefetch_hidden_candidate_us_total;
    summary.a_prefetch_wait_us_total += delta.a_prefetch_wait_us_total;
    summary.a_prefetch_hidden_candidate_us_total += delta.a_prefetch_hidden_candidate_us_total;
    summary.o_mvout_wait_us_total += delta.o_mvout_wait_us_total;
    summary.o_mvout_hidden_candidate_us_total += delta.o_mvout_hidden_candidate_us_total;
    summary.gemm_us_total += delta.gemm_us_total;
    summary.dma_out_us_total += delta.dma_out_us_total;
    summary.postprocess_us_total += delta.postprocess_us_total;
    summary.accounted_us_total += delta.accounted_us_total;
    summary.unaccounted_us_total += delta.unaccounted_us_total;
}

} // namespace ggml_npu
