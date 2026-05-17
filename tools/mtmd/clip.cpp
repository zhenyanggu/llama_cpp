// NOTE: This is modified from clip.cpp only for LLaVA,
// so there might be still unnecessary artifacts hanging around
// I'll gradually clean and extend it
// Note: Even when using identical normalized image inputs (see normalize_image_u8_to_f32()) we have a significant difference in resulting embeddings compared to pytorch
#include "clip.h"
#include "clip-impl.h"
#include "ggml.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"
#ifdef GGML_USE_NPU
#include "ggml/src/ggml-npu/ggml-npu.h"
#endif

#include <nlohmann/json.hpp>

#include <cassert>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <algorithm>
#include <map>
#include <optional>
#include <regex>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <sstream>
#include <cinttypes>
#include <limits>
#include <array>
#include <numeric>
#include <functional>
#include <mutex>
#include <memory>
#include <atomic>

using json = nlohmann::ordered_json;

struct clip_profile_tensor_info {
    bool present = false;
    std::string name;
    std::string op_name;
    std::string type;
    bool is_quantized = false;
    std::array<int64_t, GGML_MAX_DIMS> ne = {};
};

struct clip_profile_node_timing {
    std::string node_name;
    std::string op_name;
    std::string tensor_type;
    bool tensor_is_quantized = false;
    std::array<int64_t, GGML_MAX_DIMS> ne = {};
    std::array<clip_profile_tensor_info, GGML_MAX_SRC> srcs = {};
    double duration_us = 0.0;
    int64_t event_index = 0;
};

struct clip_profile_op_aggregate {
    double duration_us = 0.0;
    int64_t node_count = 0;
};

struct clip_profile_signature_aggregate {
    clip_profile_tensor_info src0;
    clip_profile_tensor_info src1;
    clip_profile_tensor_info dst;
    double duration_us = 0.0;
    int64_t node_count = 0;
    std::vector<std::string> example_node_names;
};

struct clip_profile_node_name_aggregate {
    clip_profile_tensor_info output;
    std::string op_name;
    double duration_us = 0.0;
    int64_t node_count = 0;
};

static clip_profile_tensor_info clip_capture_tensor_info(const ggml_tensor * t) {
    clip_profile_tensor_info info;
    if (t == nullptr) {
        return info;
    }

    info.present = true;
    info.name = t->name;
    info.op_name = ggml_op_desc(t);
    info.type = ggml_type_name(t->type);
    info.is_quantized = ggml_is_quantized(t->type);
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        info.ne[i] = t->ne[i];
    }

    return info;
}

static double clip_pct(double numerator, double denominator) {
    if (denominator <= 0.0) {
        return 0.0;
    }
    return numerator / denominator * 100.0;
}

static json clip_shape_json(const std::array<int64_t, GGML_MAX_DIMS> & ne) {
    return {ne[0], ne[1], ne[2], ne[3]};
}

static json clip_tensor_json(const clip_profile_tensor_info & info) {
    if (!info.present) {
        return nullptr;
    }

    return {
        {"name", info.name},
        {"operator_name", info.op_name},
        {"type", info.type},
        {"is_quantized", info.is_quantized},
        {"shape", clip_shape_json(info.ne)},
    };
}

static void clip_dump_embeddings_if_requested(const ggml_tensor * embeddings, const float * data) {
    const char * dump_path_env = std::getenv("MTMD_DUMP_EMBEDDINGS");
    if (dump_path_env == nullptr || dump_path_env[0] == '\0' || embeddings == nullptr || data == nullptr) {
        return;
    }

    const std::string dump_path = dump_path_env;
    const int64_t n = ggml_nelements(embeddings);
    double sum = 0.0;
    float min_value = 0.0f;
    float max_value = 0.0f;
    float max_abs = 0.0f;
    if (n > 0) {
        min_value = data[0];
        max_value = data[0];
    }
    const int64_t sample_count = std::min<int64_t>(n, 32);
    json first_values = json::array();
    for (int64_t i = 0; i < n; ++i) {
        const float value = data[i];
        sum += value;
        min_value = std::min(min_value, value);
        max_value = std::max(max_value, value);
        max_abs = std::max(max_abs, std::fabs(value));
        if (i < sample_count) {
            first_values.push_back(value);
        }
    }

    {
        std::ofstream out(dump_path, std::ios::binary);
        if (!out) {
            LOG_ERR("%s: failed to open MTMD_DUMP_EMBEDDINGS path '%s'\n", __func__, dump_path.c_str());
            return;
        }
        out.write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(n * (int64_t) sizeof(float)));
        if (!out) {
            LOG_ERR("%s: failed to write MTMD_DUMP_EMBEDDINGS path '%s'\n", __func__, dump_path.c_str());
            return;
        }
    }

    std::array<int64_t, GGML_MAX_DIMS> shape = {};
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        shape[i] = embeddings->ne[i];
    }
    json meta = {
        {"path", dump_path},
        {"name", embeddings->name},
        {"type", ggml_type_name(embeddings->type)},
        {"shape", clip_shape_json(shape)},
        {"n_elements", n},
        {"bytes", n * (int64_t) sizeof(float)},
        {"min", min_value},
        {"max", max_value},
        {"mean", n > 0 ? sum / (double) n : 0.0},
        {"max_abs", max_abs},
        {"first_values", first_values},
    };
    if (const char * backend_env = std::getenv("MTMD_BACKEND_DEVICE")) {
        meta["MTMD_BACKEND_DEVICE"] = backend_env;
    }

    const std::string meta_path = dump_path + ".json";
    std::ofstream meta_out(meta_path);
    if (!meta_out) {
        LOG_ERR("%s: failed to open MTMD_DUMP_EMBEDDINGS metadata path '%s'\n", __func__, meta_path.c_str());
        return;
    }
    meta_out << meta.dump(2) << "\n";
    LOG_INF("%s: dumped final mmproj embeddings to %s (%" PRId64 " f32 values)\n",
        __func__, dump_path.c_str(), n);
}

static std::string clip_sanitize_dump_name(const char * name) {
    std::string result = name != nullptr && name[0] != '\0' ? name : "unnamed";
    for (char & c : result) {
        const bool ok =
            (c >= 'a' && c <= 'z') ||
            (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') ||
            c == '.' || c == '_' || c == '-';
        if (!ok) {
            c = '_';
        }
    }
    return result;
}

static void clip_dump_tensor_f32_to_dir(const std::string & dir, const ggml_tensor * tensor, int index) {
    if (dir.empty() || tensor == nullptr || tensor->type != GGML_TYPE_F32) {
        return;
    }

    std::vector<float> data(static_cast<size_t>(ggml_nelements(tensor)));
    ggml_backend_tensor_get(tensor, data.data(), 0, ggml_nbytes(tensor));

    double sum = 0.0;
    float min_value = data.empty() ? 0.0f : data[0];
    float max_value = data.empty() ? 0.0f : data[0];
    float max_abs = 0.0f;
    for (float value : data) {
        sum += value;
        min_value = std::min(min_value, value);
        max_value = std::max(max_value, value);
        max_abs = std::max(max_abs, std::fabs(value));
    }

    std::array<int64_t, GGML_MAX_DIMS> shape = {};
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        shape[i] = tensor->ne[i];
    }

    std::ostringstream filename;
    filename << dir << "/";
    filename.width(3);
    filename.fill('0');
    filename << index << "_" << clip_sanitize_dump_name(tensor->name) << ".bin";
    const std::string bin_path = filename.str();

    {
        std::ofstream out(bin_path, std::ios::binary);
        if (!out) {
            LOG_ERR("%s: failed to open tensor dump path '%s'\n", __func__, bin_path.c_str());
            return;
        }
        out.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(float)));
        if (!out) {
            LOG_ERR("%s: failed to write tensor dump path '%s'\n", __func__, bin_path.c_str());
            return;
        }
    }

    json meta = {
        {"path", bin_path},
        {"index", index},
        {"name", tensor->name},
        {"op_name", ggml_op_desc(tensor)},
        {"type", ggml_type_name(tensor->type)},
        {"shape", clip_shape_json(shape)},
        {"n_elements", (int64_t) data.size()},
        {"bytes", (int64_t) data.size() * (int64_t) sizeof(float)},
        {"min", min_value},
        {"max", max_value},
        {"mean", data.empty() ? 0.0 : sum / (double) data.size()},
        {"max_abs", max_abs},
    };
    std::ofstream meta_out(bin_path + ".json");
    if (meta_out) {
        meta_out << meta.dump(2) << "\n";
    }
}

static bool clip_is_bias_tensor(const clip_profile_tensor_info & info) {
    return info.present && info.name.size() >= 5 &&
        info.name.compare(info.name.size() - 5, 5, ".bias") == 0;
}

static bool clip_is_large_matmul_bias_add(const clip_profile_node_timing & node) {
    if (node.op_name != "ADD") {
        return false;
    }
    const bool has_bias = clip_is_bias_tensor(node.srcs[0]) || clip_is_bias_tensor(node.srcs[1]);
    if (!has_bias) {
        return false;
    }
    // LayerNorm/patch tiny bias ADDs are real elementwise CPU work. The large
    // projection bias nodes are roots for fused NPU MUL_MAT+ADD execution.
    return node.ne[0] >= 128 && node.ne[1] >= 128;
}

struct clip_npu_execution_index {
    std::unordered_set<std::string> root_names;
};

static bool clip_node_is_npu_executed(
        const clip_profile_node_timing & node,
        const clip_npu_execution_index * npu_index) {
    return npu_index != nullptr &&
        npu_index->root_names.find(node.node_name) != npu_index->root_names.end();
}

static std::string clip_profile_operator_category(
        const clip_profile_node_timing & node,
        const clip_npu_execution_index * npu_index = nullptr) {
    const bool on_npu = clip_node_is_npu_executed(node, npu_index);
    if (on_npu && clip_is_large_matmul_bias_add(node)) {
        return "FUSED_MUL_MAT_BIAS_NPU";
    }
    if (on_npu && node.op_name == "MUL_MAT") {
        return "MUL_MAT_NPU";
    }
    if (node.op_name == "MUL_MAT") {
        return "MUL_MAT_CPU";
    }
    if (node.op_name == "ADD") {
        return "ADD_CPU";
    }
    return node.op_name;
}

static std::string clip_shape_key(const std::array<int64_t, GGML_MAX_DIMS> & ne) {
    return std::to_string(ne[0]) + "x" + std::to_string(ne[1]) + "x" + std::to_string(ne[2]) + "x" + std::to_string(ne[3]);
}

static std::string clip_mul_mat_signature_key(const clip_profile_node_timing & node) {
    return node.srcs[0].type + "|" + clip_shape_key(node.srcs[0].ne) + "|" +
           node.srcs[1].type + "|" + clip_shape_key(node.srcs[1].ne) + "|" +
           node.tensor_type + "|" + clip_shape_key(node.ne);
}

static json clip_node_json(
        const clip_profile_node_timing & node,
        double total_us,
        const clip_npu_execution_index * npu_index = nullptr) {
    json inputs = json::array();
    for (const auto & src : node.srcs) {
        if (src.present) {
            inputs.push_back(clip_tensor_json(src));
        }
    }

    return {
        {"event_index", node.event_index},
        {"node_name", node.node_name},
        {"operator_name", node.op_name},
        {"operator_category", clip_profile_operator_category(node, npu_index)},
        {"device", clip_node_is_npu_executed(node, npu_index) ? "npu" : "cpu"},
        {"is_large_matmul_bias_add", clip_is_large_matmul_bias_add(node)},
        {"output", {
            {"name", node.node_name},
            {"type", node.tensor_type},
            {"is_quantized", node.tensor_is_quantized},
            {"shape", clip_shape_json(node.ne)},
        }},
        {"inputs", inputs},
        {"duration_us", node.duration_us},
        {"duration_ms", node.duration_us / 1000.0},
        {"share_of_total_time_pct", clip_pct(node.duration_us, total_us)},
    };
}

struct clip_mul_mat_split_stats {
    int64_t npu_node_count = 0;
    double npu_duration_us = 0.0;
    int64_t cpu_node_count = 0;
    double cpu_duration_us = 0.0;
    int64_t unmatched_node_count = 0;
    double unmatched_duration_us = 0.0;
};

static clip_npu_execution_index clip_load_npu_execution_index() {
    clip_npu_execution_index index;
    const char * npu_trace_path = std::getenv("GGML_NPU_PROFILE_JSON");
    if (npu_trace_path == nullptr || npu_trace_path[0] == '\0') {
        return index;
    }

    std::ifstream in(npu_trace_path);
    if (!in.is_open()) {
        return index;
    }

    json doc;
    try {
        in >> doc;
    } catch (...) {
        return index;
    }

    const auto it_nodes = doc.find("nodes");
    if (it_nodes == doc.end() || !it_nodes->is_array()) {
        return index;
    }

    for (const auto & node : *it_nodes) {
        if (!node.is_object()) {
            continue;
        }
        const std::string root_name = node.value("root_name", std::string());
        if (!root_name.empty()) {
            index.root_names.insert(root_name);
        }
    }

    return index;
}

static std::optional<clip_mul_mat_split_stats> clip_load_npu_mul_mat_split(
    double mul_mat_total_us, int64_t mul_mat_total_node_count) {
    const char * npu_trace_path = std::getenv("GGML_NPU_PROFILE_JSON");
    if (npu_trace_path == nullptr || npu_trace_path[0] == '\0') {
        return std::nullopt;
    }

    std::ifstream in(npu_trace_path);
    if (!in.is_open()) {
        return std::nullopt;
    }

    json doc;
    try {
        in >> doc;
    } catch (...) {
        return std::nullopt;
    }

    const auto it_nodes = doc.find("nodes");
    if (it_nodes == doc.end() || !it_nodes->is_array()) {
        return std::nullopt;
    }

    clip_mul_mat_split_stats s;
    for (const auto & node : *it_nodes) {
        if (!node.is_object()) {
            continue;
        }
        if (node.value("compute_op_name", std::string()) != "MUL_MAT") {
            continue;
        }
        s.npu_node_count += 1;
        s.npu_duration_us += node.value("total_node_us", 0.0);
    }

    if (s.npu_duration_us > mul_mat_total_us) {
        s.unmatched_duration_us = s.npu_duration_us - mul_mat_total_us;
        s.npu_duration_us = mul_mat_total_us;
    }
    if (s.npu_node_count > mul_mat_total_node_count) {
        s.unmatched_node_count = s.npu_node_count - mul_mat_total_node_count;
        s.npu_node_count = mul_mat_total_node_count;
    }

    s.cpu_node_count = mul_mat_total_node_count - s.npu_node_count;
    if (s.cpu_node_count < 0) {
        s.unmatched_node_count += -s.cpu_node_count;
        s.cpu_node_count = 0;
    }

    const double remaining_us = mul_mat_total_us - s.npu_duration_us;
    if (s.cpu_node_count == 0 && remaining_us > 0.0) {
        // The NPU trace and ggml callback streams are not one-to-one. Do not
        // report the uncorrelated remainder as CPU time.
        s.unmatched_duration_us += remaining_us;
        s.cpu_duration_us = 0.0;
    } else {
        s.cpu_duration_us = std::max(0.0, remaining_us);
    }

    return s;
}

struct clip_profiler {
    bool enabled = false;
    std::string output_path;
    const ggml_tensor * pending_tensor = nullptr;
    int64_t pending_start_us = 0;
    int64_t next_event_index = 0;
    std::vector<clip_profile_node_timing> nodes;

    void init_from_env() {
        const char * path = std::getenv("MTMD_PROFILE_MMPROJ_JSON");
        output_path = path ? path : "";
        enabled = !output_path.empty();
    }

    void reset() {
        pending_tensor = nullptr;
        pending_start_us = 0;
        next_event_index = 0;
        nodes.clear();
    }

    static bool eval_callback(struct ggml_tensor * t, bool ask, void * user_data) {
        auto * profiler = static_cast<clip_profiler *>(user_data);
        if (!profiler || !profiler->enabled) {
            return false;
        }

        if (ask) {
            profiler->pending_tensor = t;
            profiler->pending_start_us = ggml_time_us();
            return true;
        }

        clip_profile_node_timing timing;
        timing.node_name = t->name;
        timing.op_name = ggml_op_desc(t);
        timing.tensor_type = ggml_type_name(t->type);
        timing.tensor_is_quantized = ggml_is_quantized(t->type);
        timing.duration_us = static_cast<double>(ggml_time_us() - profiler->pending_start_us);
        timing.event_index = profiler->next_event_index++;
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            timing.ne[i] = t->ne[i];
        }
        for (int i = 0; i < GGML_MAX_SRC; ++i) {
            timing.srcs[i] = clip_capture_tensor_info(t->src[i]);
        }

        profiler->nodes.push_back(std::move(timing));
        profiler->pending_tensor = nullptr;
        profiler->pending_start_us = 0;

        return true;
    }

    json summarize() const {
        std::unordered_map<std::string, clip_profile_op_aggregate> ops;
        std::unordered_map<std::string, clip_profile_op_aggregate> op_categories;
        std::unordered_map<std::string, clip_profile_signature_aggregate> matmuls;
        std::unordered_map<std::string, clip_profile_node_name_aggregate> node_names;
        const clip_npu_execution_index npu_index = clip_load_npu_execution_index();
        double total_us = 0.0;

        for (const auto & node : nodes) {
            total_us += node.duration_us;
            auto & agg = ops[node.op_name];
            agg.duration_us += node.duration_us;
            agg.node_count += 1;

            auto & category_agg = op_categories[clip_profile_operator_category(node, &npu_index)];
            category_agg.duration_us += node.duration_us;
            category_agg.node_count += 1;

            auto & node_name_agg = node_names[node.node_name + "|" + node.op_name];
            if (node_name_agg.node_count == 0) {
                node_name_agg.output.present = true;
                node_name_agg.output.name = node.node_name;
                node_name_agg.output.type = node.tensor_type;
                node_name_agg.output.is_quantized = node.tensor_is_quantized;
                node_name_agg.output.ne = node.ne;
                node_name_agg.op_name = node.op_name;
            }
            node_name_agg.duration_us += node.duration_us;
            node_name_agg.node_count += 1;

            if (node.op_name == "MUL_MAT" && node.srcs[0].present && node.srcs[1].present) {
                auto & sig = matmuls[clip_mul_mat_signature_key(node)];
                if (sig.node_count == 0) {
                    sig.src0 = node.srcs[0];
                    sig.src1 = node.srcs[1];
                    sig.dst.present = true;
                    sig.dst.name = node.node_name;
                    sig.dst.type = node.tensor_type;
                    sig.dst.is_quantized = node.tensor_is_quantized;
                    sig.dst.ne = node.ne;
                }
                sig.duration_us += node.duration_us;
                sig.node_count += 1;
                if (sig.example_node_names.size() < 3) {
                    if (std::find(sig.example_node_names.begin(), sig.example_node_names.end(), node.node_name) == sig.example_node_names.end()) {
                        sig.example_node_names.push_back(node.node_name);
                    }
                }
            }
        }

        std::vector<std::pair<std::string, clip_profile_op_aggregate>> sorted_ops(ops.begin(), ops.end());
        std::sort(sorted_ops.begin(), sorted_ops.end(), [](const auto & lhs, const auto & rhs) {
            if (lhs.second.duration_us != rhs.second.duration_us) {
                return lhs.second.duration_us > rhs.second.duration_us;
            }
            return lhs.first < rhs.first;
        });

        std::vector<std::pair<std::string, clip_profile_op_aggregate>> sorted_categories(op_categories.begin(), op_categories.end());
        std::sort(sorted_categories.begin(), sorted_categories.end(), [](const auto & lhs, const auto & rhs) {
            if (lhs.second.duration_us != rhs.second.duration_us) {
                return lhs.second.duration_us > rhs.second.duration_us;
            }
            return lhs.first < rhs.first;
        });

        std::vector<std::pair<std::string, clip_profile_signature_aggregate>> sorted_matmuls(matmuls.begin(), matmuls.end());
        std::sort(sorted_matmuls.begin(), sorted_matmuls.end(), [](const auto & lhs, const auto & rhs) {
            if (lhs.second.duration_us != rhs.second.duration_us) {
                return lhs.second.duration_us > rhs.second.duration_us;
            }
            return lhs.first < rhs.first;
        });

        std::vector<std::pair<std::string, clip_profile_node_name_aggregate>> sorted_node_names(node_names.begin(), node_names.end());
        std::sort(sorted_node_names.begin(), sorted_node_names.end(), [](const auto & lhs, const auto & rhs) {
            if (lhs.second.duration_us != rhs.second.duration_us) {
                return lhs.second.duration_us > rhs.second.duration_us;
            }
            return lhs.first < rhs.first;
        });

        std::vector<clip_profile_node_timing> sorted_nodes = nodes;
        std::sort(sorted_nodes.begin(), sorted_nodes.end(), [](const auto & lhs, const auto & rhs) {
            if (lhs.duration_us != rhs.duration_us) {
                return lhs.duration_us > rhs.duration_us;
            }
            return lhs.node_name < rhs.node_name;
        });

        json operators = json::array();
        for (const auto & [name, agg] : sorted_ops) {
            operators.push_back({
                {"operator_name", name},
                {"duration_us", agg.duration_us},
                {"duration_ms", agg.duration_us / 1000.0},
                {"share_of_total_time_pct", clip_pct(agg.duration_us, total_us)},
                {"node_event_count", agg.node_count},
            });
        }

        json operator_categories = json::array();
        for (const auto & [name, agg] : sorted_categories) {
            operator_categories.push_back({
                {"operator_category", name},
                {"duration_us", agg.duration_us},
                {"duration_ms", agg.duration_us / 1000.0},
                {"share_of_total_time_pct", clip_pct(agg.duration_us, total_us)},
                {"node_event_count", agg.node_count},
            });
        }

        json mul_mat_signatures = json::array();
        double mul_mat_total_us = 0.0;
        int64_t mul_mat_total_node_count = 0;
        for (const auto & [_, sig] : sorted_matmuls) {
            mul_mat_total_us += sig.duration_us;
            mul_mat_total_node_count += sig.node_count;
            mul_mat_signatures.push_back({
                {"src0", clip_tensor_json(sig.src0)},
                {"src1", clip_tensor_json(sig.src1)},
                {"dst", clip_tensor_json(sig.dst)},
                {"duration_us", sig.duration_us},
                {"duration_ms", sig.duration_us / 1000.0},
                {"share_of_total_time_pct", clip_pct(sig.duration_us, total_us)},
                {"node_event_count", sig.node_count},
                {"example_node_names", sig.example_node_names},
            });
        }

        json mul_mat_split = nullptr;
        if (const auto split = clip_load_npu_mul_mat_split(mul_mat_total_us, mul_mat_total_node_count); split.has_value()) {
            const auto & s = split.value();
            mul_mat_split = {
                {"manifest_available", true},
                {"npu", {
                    {"node_event_count", s.npu_node_count},
                    {"duration_us", s.npu_duration_us},
                    {"share_of_total_time_pct", clip_pct(s.npu_duration_us, total_us)},
                }},
                {"cpu", {
                    {"node_event_count", s.cpu_node_count},
                    {"duration_us", s.cpu_duration_us},
                    {"share_of_total_time_pct", clip_pct(s.cpu_duration_us, total_us)},
                }},
                {"unmatched", {
                    {"node_event_count", s.unmatched_node_count},
                    {"duration_us", s.unmatched_duration_us},
                    {"share_of_total_time_pct", clip_pct(s.unmatched_duration_us, total_us)},
                }},
                {"correlation_quality", (s.unmatched_node_count == 0 && s.unmatched_duration_us == 0.0) ? "usable" : "partial"},
                {"note", "Only matched NPU node time is attributed to NPU. Remainder is unmatched unless callback and NPU trace counts prove a CPU split."},
            };
        }

        json node_name_summary = json::array();
        for (const auto & [_, agg] : sorted_node_names) {
            node_name_summary.push_back({
                {"node_name", agg.output.name},
                {"operator_name", agg.op_name},
                {"output", clip_tensor_json(agg.output)},
                {"duration_us", agg.duration_us},
                {"duration_ms", agg.duration_us / 1000.0},
                {"share_of_total_time_pct", clip_pct(agg.duration_us, total_us)},
                {"node_event_count", agg.node_count},
            });
        }

        json top_nodes = json::array();
        for (size_t i = 0; i < sorted_nodes.size() && i < 25; ++i) {
            top_nodes.push_back(clip_node_json(sorted_nodes[i], total_us, &npu_index));
        }

        json all_nodes = json::array();
        for (const auto & node : nodes) {
            all_nodes.push_back(clip_node_json(node, total_us, &npu_index));
        }

        return {
            {"profile_kind", "mmproj_operator_latency_share"},
            {"timing_unit", "us"},
            {"note", "Per-node wall-clock timings captured through ggml eval callback. Profiling forces node-by-node synchronized execution, so totals are from profiled replay and operator shares should be interpreted primarily as proportions."},
            {"node_event_count", nodes.size()},
            {"npu_matched_root_count", npu_index.root_names.size()},
            {"total_us", total_us},
            {"total_ms", total_us / 1000.0},
            {"mul_mat_total_us", mul_mat_total_us},
            {"mul_mat_total_ms", mul_mat_total_us / 1000.0},
            {"mul_mat_share_of_total_time_pct", clip_pct(mul_mat_total_us, total_us)},
            {"mul_mat_split", mul_mat_split},
            {"operators", operators},
            {"operator_categories", operator_categories},
            {"mul_mat_signatures", mul_mat_signatures},
            {"node_name_summary", node_name_summary},
            {"top_nodes", top_nodes},
            {"nodes", all_nodes},
        };
    }

    void write_json() const {
        if (!enabled || output_path.empty()) {
            return;
        }

        std::ofstream out(output_path);
        if (!out.is_open()) {
            LOG_ERR("%s: failed to open %s for writing\n", __func__, output_path.c_str());
            return;
        }
        out << summarize().dump(2) << '\n';
    }
};

struct clip_logger_state g_logger_state = {GGML_LOG_LEVEL_CONT, clip_log_callback_default, NULL};

enum ffn_op_type {
    FFN_GELU,
    FFN_GELU_ERF,
    FFN_SILU,
    FFN_GELU_QUICK,
};

enum norm_type {
    NORM_TYPE_NORMAL,
    NORM_TYPE_RMS,
};

//#define CLIP_DEBUG_FUNCTIONS

#ifdef CLIP_DEBUG_FUNCTIONS
static void clip_image_write_image_to_ppm(const clip_image_u8& img, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERR("Failed to open file for writing: %s\n", filename.c_str());
        return;
    }

    // PPM header: P6 format, width, height, and max color value
    file << "P6\n" << img.nx << " " << img.ny << "\n255\n";

    // Write pixel data
    for (size_t i = 0; i < img.buf.size(); i += 3) {
        // PPM expects binary data in RGB format, which matches our image buffer
        file.write(reinterpret_cast<const char*>(&img.buf[i]), 3);
    }

    file.close();
}

static void clip_image_save_to_bmp(const clip_image_u8& img, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERR("Failed to open file for writing: %s\n", filename.c_str());
        return;
    }

    int fileSize = 54 + 3 * img.nx * img.ny; // File header + info header + pixel data
    int bytesPerPixel = 3;
    int widthInBytes = img.nx * bytesPerPixel;
    int paddingAmount = (4 - (widthInBytes % 4)) % 4;
    int stride = widthInBytes + paddingAmount;

    // Bitmap file header
    unsigned char fileHeader[14] = {
        'B','M',     // Signature
        0,0,0,0,    // Image file size in bytes
        0,0,0,0,    // Reserved
        54,0,0,0    // Start of pixel array
    };

    // Total file size
    fileSize = 54 + (stride * img.ny);
    fileHeader[2] = (unsigned char)(fileSize);
    fileHeader[3] = (unsigned char)(fileSize >> 8);
    fileHeader[4] = (unsigned char)(fileSize >> 16);
    fileHeader[5] = (unsigned char)(fileSize >> 24);

    // Bitmap information header (BITMAPINFOHEADER)
    unsigned char infoHeader[40] = {
        40,0,0,0,   // Size of this header (40 bytes)
        0,0,0,0,    // Image width
        0,0,0,0,    // Image height
        1,0,        // Number of color planes
        24,0,       // Bits per pixel
        0,0,0,0,    // No compression
        0,0,0,0,    // Image size (can be 0 for no compression)
        0,0,0,0,    // X pixels per meter (not specified)
        0,0,0,0,    // Y pixels per meter (not specified)
        0,0,0,0,    // Total colors (color table not used)
        0,0,0,0     // Important colors (all are important)
    };

    // Width and height in the information header
    infoHeader[4] = (unsigned char)(img.nx);
    infoHeader[5] = (unsigned char)(img.nx >> 8);
    infoHeader[6] = (unsigned char)(img.nx >> 16);
    infoHeader[7] = (unsigned char)(img.nx >> 24);
    infoHeader[8] = (unsigned char)(img.ny);
    infoHeader[9] = (unsigned char)(img.ny >> 8);
    infoHeader[10] = (unsigned char)(img.ny >> 16);
    infoHeader[11] = (unsigned char)(img.ny >> 24);

    // Write file headers
    file.write(reinterpret_cast<char*>(fileHeader), sizeof(fileHeader));
    file.write(reinterpret_cast<char*>(infoHeader), sizeof(infoHeader));

    // Pixel data
    std::vector<unsigned char> padding(3, 0); // Max padding size to be added to each row
    for (int y = img.ny - 1; y >= 0; --y) { // BMP files are stored bottom-to-top
        for (int x = 0; x < img.nx; ++x) {
            // Each pixel
            size_t pixelIndex = (y * img.nx + x) * 3;
            unsigned char pixel[3] = {
                img.buf[pixelIndex + 2], // BMP stores pixels in BGR format
                img.buf[pixelIndex + 1],
                img.buf[pixelIndex]
            };
            file.write(reinterpret_cast<char*>(pixel), 3);
        }
        // Write padding for the row
        file.write(reinterpret_cast<char*>(padding.data()), paddingAmount);
    }

    file.close();
}

// debug function to convert f32 to u8
static void clip_image_convert_f32_to_u8(const clip_image_f32& src, clip_image_u8& dst) {
    dst.nx = src.nx;
    dst.ny = src.ny;
    dst.buf.resize(3 * src.nx * src.ny);
    for (size_t i = 0; i < src.buf.size(); ++i) {
        dst.buf[i] = static_cast<uint8_t>(std::min(std::max(int(src.buf[i] * 255.0f), 0), 255));
    }
}
#endif


//
// clip layers
//

enum patch_merge_type {
    PATCH_MERGE_FLAT,
    PATCH_MERGE_SPATIAL_UNPAD,
};

struct clip_aicas_w8a8_tensor {
    bool enabled = true;
    std::string policy = "F16_FALLBACK";
    float act_scale = 0.0f;
    int32_t act_scale_q8_24 = 0;
    int32_t act_zero_point = 0;
    std::string act_quant_mode = "asymmetric_u8";
    std::string weight_scale_mode = "per_channel";
    std::vector<float> weight_scale;
    std::vector<int32_t> sum_w;
    std::vector<float> smooth_scale;
    float smooth_alpha = 0.0f;
    float smooth_eps = 0.0f;
    bool smooth_enabled = false;
    bool bias_compensated = false;

    bool uses_symmetric_u8() const {
        return act_quant_mode == "symmetric_u8";
    }

    bool uses_per_tensor_weight_scale() const {
        return weight_scale_mode == "per_tensor";
    }

    size_t expected_weight_scale_len(int64_t out_channels) const {
        return uses_per_tensor_weight_scale() ? size_t(1) : static_cast<size_t>(out_channels);
    }

    bool has_valid_compensation_config(int64_t out_channels) const {
        if (sum_w.empty()) {
            return uses_symmetric_u8() && act_zero_point == 128;
        }
        return sum_w.size() == static_cast<size_t>(out_channels);
    }

    bool has_valid_smooth_config(int64_t in_channels) const {
        if (!smooth_enabled) {
            return smooth_scale.empty();
        }

        return in_channels >= 0 &&
            smooth_scale.size() == static_cast<size_t>(in_channels);
    }

    bool is_npu_compatible(int64_t out_channels) const {
        if (out_channels < 0) {
            if (uses_per_tensor_weight_scale()) {
                return weight_scale.size() == 1 && !sum_w.empty();
            }
            return !weight_scale.empty() && !sum_w.empty() && weight_scale.size() == sum_w.size();
        }
        if (uses_per_tensor_weight_scale()) {
            return weight_scale.size() == 1 &&
                sum_w.size() == static_cast<size_t>(out_channels);
        }
        return weight_scale.size() == static_cast<size_t>(out_channels) &&
            sum_w.size() == static_cast<size_t>(out_channels);
    }
};

static std::string clip_aicas_smooth_scale_tensor_name(const std::string & weight_name) {
    return weight_name + ".aicas_smooth_scale";
}

static bool clip_aicas_fuse_bias_compensation(
        ggml_tensor * bias,
        const clip_aicas_w8a8_tensor & cfg,
        std::string * error) {
    if (bias == nullptr) {
        if (error) {
            *error = "missing bias tensor";
        }
        return false;
    }
    if (bias->ne[1] != 1 || bias->ne[2] != 1 || bias->ne[3] != 1) {
        if (error) {
            *error = "bias must be 1D broadcast";
        }
        return false;
    }
    if (bias->type != GGML_TYPE_F32 && bias->type != GGML_TYPE_F16) {
        if (error) {
            *error = std::string("unsupported bias type: ") + ggml_type_name(bias->type);
        }
        return false;
    }

    const int64_t out_channels = bias->ne[0];
    if (out_channels < 0 || !cfg.is_npu_compatible(out_channels)) {
        if (error) {
            *error = "incompatible W8A8 metadata for bias fusion";
        }
        return false;
    }

    std::vector<uint8_t> raw(ggml_nbytes(bias));
    ggml_backend_tensor_get(bias, raw.data(), 0, raw.size());

    const int32_t act_zero_point_i8 = cfg.act_zero_point - 128;
    for (int64_t i = 0; i < out_channels; ++i) {
        const float weight_scale = cfg.uses_per_tensor_weight_scale()
            ? cfg.weight_scale[0]
            : cfg.weight_scale[static_cast<size_t>(i)];
        const float fused_bias_delta =
            cfg.act_scale * weight_scale * static_cast<float>(act_zero_point_i8 * cfg.sum_w[static_cast<size_t>(i)]);

        if (bias->type == GGML_TYPE_F32) {
            float * values = reinterpret_cast<float *>(raw.data());
            values[i] -= fused_bias_delta;
        } else {
            ggml_fp16_t * values = reinterpret_cast<ggml_fp16_t *>(raw.data());
            const float current = ggml_fp16_to_fp32(values[i]);
            values[i] = ggml_fp32_to_fp16(current - fused_bias_delta);
        }
    }

    ggml_backend_tensor_set(bias, raw.data(), 0, raw.size());
    return true;
}

static bool clip_should_prefuse_aicas_bias_compensation() {
    const char * bias_mode = std::getenv("GGML_NPU_AICAS_BIAS_MODE");
    if (bias_mode == nullptr || bias_mode[0] == '\0') {
        return true;
    }

    if (strcmp(bias_mode, "precomp") == 0) {
        return true;
    }
    if (strcmp(bias_mode, "auto") == 0 || strcmp(bias_mode, "raw") == 0) {
        return false;
    }

    return false;
}

enum class clip_aicas_dequant_sim_mode {
    off,
    versa_q8_24,
    versa_q8_24_fp_reconstruct,
    scale_shift_i32_round,
    scale_shift_fp_reconstruct,
};

enum class clip_mmproj_attn_precision {
    f32,
    f16,
    bf16,
    bfp16m,
};

enum class clip_mmproj_attn_precision_scope {
    core,
    block,
};

enum class clip_aicas_bfp16m_exp_mode {
    kblock,
    per_channel,
    static_per_layer,
};

static const char * clip_mmproj_attn_precision_name(clip_mmproj_attn_precision mode) {
    switch (mode) {
        case clip_mmproj_attn_precision::f32:
            return "f32";
        case clip_mmproj_attn_precision::f16:
            return "f16";
        case clip_mmproj_attn_precision::bf16:
            return "bf16";
        case clip_mmproj_attn_precision::bfp16m:
            return "bfp16m";
    }

    return "f32";
}

static const char * clip_mmproj_attn_precision_scope_name(clip_mmproj_attn_precision_scope scope) {
    switch (scope) {
        case clip_mmproj_attn_precision_scope::core:
            return "core";
        case clip_mmproj_attn_precision_scope::block:
            return "block";
    }

    return "core";
}

static const char * clip_aicas_bfp16m_exp_mode_name(clip_aicas_bfp16m_exp_mode mode) {
    switch (mode) {
        case clip_aicas_bfp16m_exp_mode::kblock:
            return "kblock";
        case clip_aicas_bfp16m_exp_mode::per_channel:
            return "per_channel";
        case clip_aicas_bfp16m_exp_mode::static_per_layer:
            return "static_per_layer";
    }

    return "kblock";
}

static clip_mmproj_attn_precision clip_get_mmproj_attn_precision() {
    const char * env = std::getenv("AICAS_MMPROJ_ATTN_PRECISION");
    if (env == nullptr || env[0] == '\0' || std::strcmp(env, "f32") == 0) {
        return clip_mmproj_attn_precision::f32;
    }
    if (std::strcmp(env, "f16") == 0) {
        return clip_mmproj_attn_precision::f16;
    }
    if (std::strcmp(env, "bf16") == 0) {
        return clip_mmproj_attn_precision::bf16;
    }
    if (std::strcmp(env, "bfp16m") == 0 || std::strcmp(env, "bfp16-m") == 0) {
        return clip_mmproj_attn_precision::bfp16m;
    }

    throw std::runtime_error(string_format(
        "invalid AICAS_MMPROJ_ATTN_PRECISION=%s; expected one of: f32, f16, bf16, bfp16m", env));
}

static clip_mmproj_attn_precision_scope clip_get_mmproj_attn_precision_scope() {
    const char * env = std::getenv("AICAS_MMPROJ_ATTN_PRECISION_SCOPE");
    if (env == nullptr || env[0] == '\0' || std::strcmp(env, "core") == 0) {
        return clip_mmproj_attn_precision_scope::core;
    }
    if (std::strcmp(env, "block") == 0) {
        return clip_mmproj_attn_precision_scope::block;
    }

    throw std::runtime_error(string_format(
        "invalid AICAS_MMPROJ_ATTN_PRECISION_SCOPE=%s; expected one of: core, block", env));
}

static clip_aicas_bfp16m_exp_mode clip_get_bfp16m_exp_mode() {
    const char * env = std::getenv("AICAS_MMPROJ_BFP16M_EXP_MODE");
    if (env == nullptr || env[0] == '\0' || std::strcmp(env, "kblock") == 0) {
        return clip_aicas_bfp16m_exp_mode::kblock;
    }
    if (std::strcmp(env, "per_channel") == 0 || std::strcmp(env, "per-channel") == 0) {
        return clip_aicas_bfp16m_exp_mode::per_channel;
    }
    if (std::strcmp(env, "static") == 0 || std::strcmp(env, "static_per_layer") == 0 || std::strcmp(env, "static-per-layer") == 0) {
        return clip_aicas_bfp16m_exp_mode::static_per_layer;
    }

    throw std::runtime_error(string_format(
        "invalid AICAS_MMPROJ_BFP16M_EXP_MODE=%s; expected one of: kblock, per_channel, static", env));
}

static const char * clip_dequant_sim_mode_name(clip_aicas_dequant_sim_mode mode) {
    switch (mode) {
        case clip_aicas_dequant_sim_mode::off:
            return "off";
        case clip_aicas_dequant_sim_mode::versa_q8_24:
            return "versa_q8_24";
        case clip_aicas_dequant_sim_mode::versa_q8_24_fp_reconstruct:
            return "versa_q8_24_fp_reconstruct";
        case clip_aicas_dequant_sim_mode::scale_shift_i32_round:
            return "scale_shift_i32_round";
        case clip_aicas_dequant_sim_mode::scale_shift_fp_reconstruct:
            return "scale_shift_fp_reconstruct";
    }

    return "off";
}

static clip_aicas_dequant_sim_mode clip_get_dequant_sim_mode() {
    const char * env = std::getenv("AICAS_MMPROJ_DEQUANT_SIM");
    if (env == nullptr || env[0] == '\0' || std::strcmp(env, "off") == 0) {
        return clip_aicas_dequant_sim_mode::off;
    }
    if (std::strcmp(env, "versa_q8_24") == 0) {
        return clip_aicas_dequant_sim_mode::versa_q8_24;
    }
    if (std::strcmp(env, "versa_q8_24_fp_reconstruct") == 0) {
        return clip_aicas_dequant_sim_mode::versa_q8_24_fp_reconstruct;
    }
    if (std::strcmp(env, "scale_shift_i32_round") == 0) {
        return clip_aicas_dequant_sim_mode::scale_shift_i32_round;
    }
    if (std::strcmp(env, "scale_shift_fp_reconstruct") == 0) {
        return clip_aicas_dequant_sim_mode::scale_shift_fp_reconstruct;
    }
    return clip_aicas_dequant_sim_mode::off;
}

static uint32_t clip_f32_to_bits(float value) {
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static float clip_bits_to_f32(uint32_t bits) {
    float value = 0.0f;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

struct clip_ctx;

struct clip_aicas_scale_shift32 {
    int32_t scale = 0;
    int32_t shift = 0;
};

struct clip_aicas_dequant_result {
    float value = 0.0f;
    bool scale_zero = false;
    bool rounded_to_zero = false;
    bool sat_i32 = false;
    bool sat_fp_exp = false;
    bool flush_to_zero_exp = false;
    bool mul_overflow_guard_hit = false;
};

struct clip_aicas_dequant_diag_counters {
    std::atomic<uint64_t> scale_zero {0};
    std::atomic<uint64_t> rounded_to_zero {0};
    std::atomic<uint64_t> sat_i32 {0};
    std::atomic<uint64_t> sat_fp_exp {0};
    std::atomic<uint64_t> flush_to_zero_exp {0};
    std::atomic<uint64_t> mul_overflow_guard_hit {0};

    json to_json() const {
        return {
            {"scale_zero", scale_zero.load(std::memory_order_relaxed)},
            {"rounded_to_zero", rounded_to_zero.load(std::memory_order_relaxed)},
            {"sat_i32", sat_i32.load(std::memory_order_relaxed)},
            {"sat_fp_exp", sat_fp_exp.load(std::memory_order_relaxed)},
            {"flush_to_zero_exp", flush_to_zero_exp.load(std::memory_order_relaxed)},
            {"mul_overflow_guard_hit", mul_overflow_guard_hit.load(std::memory_order_relaxed)},
        };
    }

    bool empty() const {
        return scale_zero.load(std::memory_order_relaxed) == 0 &&
            rounded_to_zero.load(std::memory_order_relaxed) == 0 &&
            sat_i32.load(std::memory_order_relaxed) == 0 &&
            sat_fp_exp.load(std::memory_order_relaxed) == 0 &&
            flush_to_zero_exp.load(std::memory_order_relaxed) == 0 &&
            mul_overflow_guard_hit.load(std::memory_order_relaxed) == 0;
    }
};

struct clip_aicas_w8a8_kernel_userdata {
    const clip_aicas_w8a8_tensor * cfg = nullptr;
    clip_aicas_dequant_sim_mode mode = clip_aicas_dequant_sim_mode::off;
    clip_aicas_dequant_diag_counters * stats = nullptr;
};

struct clip_aicas_bfp16m_userdata {
    int64_t k_block = 64;
    clip_aicas_bfp16m_exp_mode exp_mode = clip_aicas_bfp16m_exp_mode::kblock;
    bool static_exp = false;
    int static_exp_a = 0;
    int static_exp_b = 0;
};

static int64_t clip_get_mmproj_bfp16m_k_block() {
    constexpr int64_t default_k_block = 64;
    const char * env = std::getenv("AICAS_MMPROJ_BFP16M_K_BLOCK");
    if (env == nullptr || env[0] == '\0') {
        return default_k_block;
    }

    const long parsed = std::strtol(env, nullptr, 10);
    if (parsed > 0 && parsed <= std::numeric_limits<int16_t>::max()) {
        return parsed;
    }

    LOG_WRN("%s: ignoring invalid AICAS_MMPROJ_BFP16M_K_BLOCK=%s, using default=%" PRId64 "\n",
            __func__, env, default_k_block);
    return default_k_block;
}

static void clip_bfp16m_quant_dequant_f32(
        struct ggml_tensor * dst,
        const struct ggml_tensor * a,
        int ith,
        int nth,
        void * userdata);

static void clip_bfp16m_mul_mat_f32(
        struct ggml_tensor * dst,
        const struct ggml_tensor * out_template,
        const struct ggml_tensor * a,
        const struct ggml_tensor * b,
        int ith,
        int nth,
        void * userdata);

static void clip_record_aicas_dequant_result(
        clip_aicas_dequant_diag_counters * stats,
        const clip_aicas_dequant_result & result) {
    if (stats == nullptr) {
        return;
    }
    if (result.scale_zero) {
        stats->scale_zero.fetch_add(1, std::memory_order_relaxed);
    }
    if (result.rounded_to_zero) {
        stats->rounded_to_zero.fetch_add(1, std::memory_order_relaxed);
    }
    if (result.sat_i32) {
        stats->sat_i32.fetch_add(1, std::memory_order_relaxed);
    }
    if (result.sat_fp_exp) {
        stats->sat_fp_exp.fetch_add(1, std::memory_order_relaxed);
    }
    if (result.flush_to_zero_exp) {
        stats->flush_to_zero_exp.fetch_add(1, std::memory_order_relaxed);
    }
    if (result.mul_overflow_guard_hit) {
        stats->mul_overflow_guard_hit.fetch_add(1, std::memory_order_relaxed);
    }
}

static int32_t clip_fp32_bits_to_q_fixed(uint32_t fp_bits, int frac_width) {
    const bool sign_bit = (fp_bits >> 31) != 0;
    const uint32_t exp_bits = (fp_bits >> 23) & 0xFFu;
    const uint32_t frac_field = fp_bits & 0x7FFFFFu;
    const bool is_nan = exp_bits == 0xFFu && frac_field != 0;
    const bool is_inf = exp_bits == 0xFFu && frac_field == 0;

    if (is_nan) {
        return 0;
    }
    if (is_inf) {
        return sign_bit ? INT32_MIN : INT32_MAX;
    }
    if (exp_bits == 0 && frac_field == 0) {
        return 0;
    }

    const uint32_t mantissa = exp_bits == 0 ? frac_field : ((1u << 23) | frac_field);
    const int exp_unbiased = exp_bits == 0 ? -126 : static_cast<int>(exp_bits) - 127;
    const int shift_amount = exp_unbiased + (frac_width - 23);

    uint64_t abs_value = 0;
    if (shift_amount >= 0) {
        if (shift_amount >= 40) {
            abs_value = UINT64_MAX;
        } else {
            abs_value = static_cast<uint64_t>(mantissa) << shift_amount;
        }
    } else {
        const int right_shift = -shift_amount;
        if (right_shift < 64) {
            const uint64_t base_value = static_cast<uint64_t>(mantissa) >> right_shift;
            const uint64_t lower_mask = right_shift == 1 ? 0x1ULL : ((1ULL << right_shift) - 1ULL);
            const uint64_t half_ulp = right_shift == 1 ? 0x1ULL : (1ULL << (right_shift - 1));
            const uint64_t remainder = static_cast<uint64_t>(mantissa) & lower_mask;
            abs_value = base_value;
            if (remainder > half_ulp || (remainder == half_ulp && (base_value & 1ULL))) {
                abs_value += 1ULL;
            }
        }
    }

    if (!sign_bit) {
        return abs_value > static_cast<uint64_t>(INT32_MAX) ? INT32_MAX : static_cast<int32_t>(abs_value);
    }

    return abs_value >= 2147483648ULL ? INT32_MIN : -static_cast<int32_t>(abs_value);
}

static int32_t clip_fp32_bits_to_q8_24(uint32_t fp_bits) {
    return clip_fp32_bits_to_q_fixed(fp_bits, 24);
}

static int32_t clip_fixed_mul_to_i32(int32_t int_value, int32_t scale_value, int frac_width, bool * saturated = nullptr) {
    const int64_t product = static_cast<int64_t>(int_value) * static_cast<int64_t>(scale_value);
    const bool product_is_neg = product < 0;
    const uint64_t abs_product = product_is_neg
        ? static_cast<uint64_t>(-product)
        : static_cast<uint64_t>(product);

    uint64_t quotient = abs_product >> frac_width;
    const uint64_t remainder_mask = (1ULL << frac_width) - 1ULL;
    const uint64_t remainder = abs_product & remainder_mask;
    const uint64_t half_ulp = 1ULL << (frac_width - 1);
    if (remainder > half_ulp || (remainder == half_ulp && (quotient & 1ULL))) {
        quotient += 1ULL;
    }

    const int64_t rounded_value = product_is_neg
        ? -static_cast<int64_t>(quotient)
        : static_cast<int64_t>(quotient);
    if (rounded_value > INT32_MAX) {
        if (saturated != nullptr) {
            *saturated = true;
        }
        return INT32_MAX;
    }
    if (rounded_value < INT32_MIN) {
        if (saturated != nullptr) {
            *saturated = true;
        }
        return INT32_MIN;
    }
    if (saturated != nullptr) {
        *saturated = false;
    }
    return static_cast<int32_t>(rounded_value);
}

static uint32_t clip_i32_to_fp32_bits_exact(int32_t int_value) {
    if (int_value == 0) {
        return 0;
    }

    const uint32_t int_bits = static_cast<uint32_t>(int_value);
    const bool sign_bit = (int_bits >> 31) != 0;
    const uint32_t abs_value = sign_bit ? ((~int_bits) + 1u) : int_bits;

    int msb_idx = 0;
    for (int bit_idx = 31; bit_idx >= 0; --bit_idx) {
        if ((abs_value >> bit_idx) & 1u) {
            msb_idx = bit_idx;
            break;
        }
    }

    uint32_t exponent_bits = static_cast<uint32_t>(msb_idx + 127);
    uint32_t mantissa_24 = 0;
    if (msb_idx <= 23) {
        mantissa_24 = abs_value << (23 - msb_idx);
    } else {
        const int right_shift = msb_idx - 23;
        mantissa_24 = abs_value >> right_shift;

        const uint32_t remainder_mask = (1u << right_shift) - 1u;
        const uint32_t remainder_bits = abs_value & remainder_mask;
        const uint32_t half_ulp = 1u << (right_shift - 1);
        if (remainder_bits > half_ulp || (remainder_bits == half_ulp && (mantissa_24 & 1u))) {
            mantissa_24 += 1u;
        }

        if (mantissa_24 == 0x1000000u) {
            exponent_bits += 1u;
            mantissa_24 = 0x800000u;
        }
    }

    const uint32_t frac_bits = mantissa_24 & 0x7FFFFFu;
    return (sign_bit ? 0x80000000u : 0u) | (exponent_bits << 23) | frac_bits;
}

static clip_aicas_scale_shift32 clip_dequant_scale_to_scale_shift(float scale) {
    clip_aicas_scale_shift32 out;
    if (!std::isfinite(scale) || scale == 0.0f) {
        return out;
    }

    int exponent = 0;
    const double abs_scale = std::fabs(static_cast<double>(scale));
    const double mantissa = std::frexp(abs_scale, &exponent);
    int64_t scale_i64 = llrint(mantissa * static_cast<double>(1ULL << 31));
    if (scale_i64 >= (1LL << 31)) {
        scale_i64 >>= 1;
        exponent += 1;
    }

    if (scale_i64 > INT32_MAX) {
        scale_i64 = INT32_MAX;
    }
    if (std::signbit(scale)) {
        scale_i64 = -scale_i64;
    }

    out.scale = static_cast<int32_t>(scale_i64);
    out.shift = exponent - 31;
    return out;
}

static int clip_u64_msb_index(uint64_t value) {
    GGML_ASSERT(value != 0);
#if defined(__GNUC__) || defined(__clang__)
    return 63 - __builtin_clzll(value);
#else
    int idx = 0;
    while ((value >> (idx + 1)) != 0) {
        ++idx;
    }
    return idx;
#endif
}

static int32_t clip_apply_scale_shift_i32_round(
        int32_t acc,
        const clip_aicas_scale_shift32 & scale_shift,
        bool * saturated = nullptr,
        bool * rounded_to_zero = nullptr,
        bool * overflow_guard_hit = nullptr) {
    if (saturated != nullptr) {
        *saturated = false;
    }
    if (rounded_to_zero != nullptr) {
        *rounded_to_zero = false;
    }
    if (overflow_guard_hit != nullptr) {
        *overflow_guard_hit = false;
    }
    if (acc == 0 || scale_shift.scale == 0) {
        return 0;
    }

    const bool product_is_neg = (acc < 0) ^ (scale_shift.scale < 0);
    const uint64_t abs_product = static_cast<uint64_t>(std::llabs(static_cast<long long>(acc))) *
        static_cast<uint64_t>(std::llabs(static_cast<long long>(scale_shift.scale)));
    const uint64_t sat_limit = product_is_neg ? 2147483648ULL : 2147483647ULL;

    uint64_t abs_result = 0;
    if (scale_shift.shift >= 0) {
        const int shift = scale_shift.shift;
        if (shift >= 64) {
            if (overflow_guard_hit != nullptr && abs_product != 0) {
                *overflow_guard_hit = true;
            }
            if (saturated != nullptr && abs_product != 0) {
                *saturated = true;
            }
            return product_is_neg ? INT32_MIN : INT32_MAX;
        }

        const uint64_t limit = sat_limit >> shift;
        if (abs_product > limit) {
            if (overflow_guard_hit != nullptr) {
                *overflow_guard_hit = true;
            }
            if (saturated != nullptr) {
                *saturated = true;
            }
            return product_is_neg ? INT32_MIN : INT32_MAX;
        }
        abs_result = static_cast<uint64_t>(abs_product << shift);
    } else {
        const int right_shift = -scale_shift.shift;
        if (right_shift >= 128) {
            if (rounded_to_zero != nullptr) {
                *rounded_to_zero = true;
            }
            return 0;
        }

        if (right_shift >= 64) {
            if (rounded_to_zero != nullptr) {
                *rounded_to_zero = true;
            }
            return 0;
        }

        uint64_t quotient = abs_product >> right_shift;
        if (right_shift > 0) {
            const uint64_t remainder_mask = (1ULL << right_shift) - 1ULL;
            const uint64_t remainder = abs_product & remainder_mask;
            const uint64_t half_ulp = 1ULL << (right_shift - 1);
            if (remainder > half_ulp || (remainder == half_ulp && (quotient & 1))) {
                quotient += 1;
            }
        }

        if (quotient > sat_limit) {
            if (saturated != nullptr) {
                *saturated = true;
            }
            return product_is_neg ? INT32_MIN : INT32_MAX;
        }
        abs_result = static_cast<uint64_t>(quotient);
    }

    if (abs_result == 0 && rounded_to_zero != nullptr) {
        *rounded_to_zero = true;
    }
    if (abs_result > sat_limit) {
        if (saturated != nullptr) {
            *saturated = true;
        }
        return product_is_neg ? INT32_MIN : INT32_MAX;
    }

    if (!product_is_neg) {
        return static_cast<int32_t>(abs_result);
    }
    if (abs_result == 2147483648ULL) {
        return INT32_MIN;
    }
    return -static_cast<int32_t>(abs_result);
}

static uint32_t clip_u64_to_fp32_bits_with_shift(
        uint64_t abs_value,
        bool sign_bit,
        int32_t value_shift,
        bool * sat_fp_exp = nullptr,
        bool * flush_to_zero_exp = nullptr,
        bool * rounded_to_zero = nullptr) {
    if (sat_fp_exp != nullptr) {
        *sat_fp_exp = false;
    }
    if (flush_to_zero_exp != nullptr) {
        *flush_to_zero_exp = false;
    }
    if (rounded_to_zero != nullptr) {
        *rounded_to_zero = false;
    }
    if (abs_value == 0) {
        return sign_bit ? 0x80000000u : 0u;
    }

    int exponent_unbiased = clip_u64_msb_index(abs_value) + value_shift;
    if (exponent_unbiased > 127) {
        if (sat_fp_exp != nullptr) {
            *sat_fp_exp = true;
        }
        return (sign_bit ? 0x80000000u : 0u) | 0x7F800000u;
    }
    if (exponent_unbiased < -126) {
        if (flush_to_zero_exp != nullptr) {
            *flush_to_zero_exp = true;
        }
        if (rounded_to_zero != nullptr) {
            *rounded_to_zero = true;
        }
        return sign_bit ? 0x80000000u : 0u;
    }

    const int msb_idx = clip_u64_msb_index(abs_value);
    uint64_t mantissa_24 = 0;
    if (msb_idx <= 23) {
        mantissa_24 = abs_value << (23 - msb_idx);
    } else {
        const int right_shift = msb_idx - 23;
        mantissa_24 = abs_value >> right_shift;

        const uint64_t remainder_mask = right_shift == 64 ? UINT64_MAX : ((1ULL << right_shift) - 1ULL);
        const uint64_t remainder_bits = abs_value & remainder_mask;
        const uint64_t half_ulp = 1ULL << (right_shift - 1);
        if (remainder_bits > half_ulp || (remainder_bits == half_ulp && (mantissa_24 & 1ULL))) {
            mantissa_24 += 1ULL;
        }

        if (mantissa_24 == 0x1000000ULL) {
            exponent_unbiased += 1;
            mantissa_24 = 0x800000ULL;
            if (exponent_unbiased > 127) {
                if (sat_fp_exp != nullptr) {
                    *sat_fp_exp = true;
                }
                return (sign_bit ? 0x80000000u : 0u) | 0x7F800000u;
            }
        }
    }

    const uint32_t exponent_bits = static_cast<uint32_t>(exponent_unbiased + 127);
    const uint32_t frac_bits = static_cast<uint32_t>(mantissa_24 & 0x7FFFFFULL);
    return (sign_bit ? 0x80000000u : 0u) | (exponent_bits << 23) | frac_bits;
}

static clip_aicas_dequant_result clip_dequantize_i32_versa_q8_24(int32_t acc, float scale) {
    clip_aicas_dequant_result out;
    const int32_t scale_q8_24 = clip_fp32_bits_to_q8_24(clip_f32_to_bits(scale));
    out.scale_zero = (scale != 0.0f) && (scale_q8_24 == 0);

    bool saturated = false;
    const int32_t rounded = clip_fixed_mul_to_i32(acc, scale_q8_24, 24, &saturated);
    out.sat_i32 = saturated;
    out.rounded_to_zero = (acc != 0) && (scale_q8_24 != 0) && (rounded == 0);
    out.value = clip_bits_to_f32(clip_i32_to_fp32_bits_exact(rounded));
    return out;
}

static clip_aicas_dequant_result clip_dequantize_i32_versa_q8_24_fp_reconstruct(int32_t acc, float scale) {
    clip_aicas_dequant_result out;
    const int32_t scale_q8_24 = clip_fp32_bits_to_q8_24(clip_f32_to_bits(scale));
    out.scale_zero = (scale != 0.0f) && (scale_q8_24 == 0);

    if (acc == 0 || scale_q8_24 == 0) {
        out.value = 0.0f;
        return out;
    }

    const int64_t product = static_cast<int64_t>(acc) * static_cast<int64_t>(scale_q8_24);
    const bool sign_bit = product < 0;
    const uint64_t abs_product = sign_bit
        ? static_cast<uint64_t>(-product)
        : static_cast<uint64_t>(product);
    const uint32_t fp_bits = clip_u64_to_fp32_bits_with_shift(
        abs_product,
        sign_bit,
        -24,
        &out.sat_fp_exp,
        &out.flush_to_zero_exp,
        &out.rounded_to_zero);
    out.value = clip_bits_to_f32(fp_bits);
    return out;
}

static clip_aicas_dequant_result clip_dequantize_i32_scale_shift_i32_round(int32_t acc, float scale) {
    clip_aicas_dequant_result out;
    const clip_aicas_scale_shift32 scale_shift = clip_dequant_scale_to_scale_shift(scale);
    out.scale_zero = (scale != 0.0f) && (scale_shift.scale == 0);

    bool saturated = false;
    bool rounded_to_zero = false;
    bool overflow_guard_hit = false;
    const int32_t rounded = clip_apply_scale_shift_i32_round(
        acc,
        scale_shift,
        &saturated,
        &rounded_to_zero,
        &overflow_guard_hit);
    out.sat_i32 = saturated;
    out.rounded_to_zero = rounded_to_zero;
    out.mul_overflow_guard_hit = overflow_guard_hit;
    out.value = clip_bits_to_f32(clip_i32_to_fp32_bits_exact(rounded));
    return out;
}

static clip_aicas_dequant_result clip_dequantize_i32_scale_shift_fp_reconstruct(int32_t acc, float scale) {
    clip_aicas_dequant_result out;
    const clip_aicas_scale_shift32 scale_shift = clip_dequant_scale_to_scale_shift(scale);
    out.scale_zero = (scale != 0.0f) && (scale_shift.scale == 0);

    if (acc == 0 || scale_shift.scale == 0) {
        out.value = 0.0f;
        return out;
    }

    const int64_t product = static_cast<int64_t>(acc) * static_cast<int64_t>(scale_shift.scale);
    const bool sign_bit = product < 0;
    const uint64_t abs_product = sign_bit
        ? static_cast<uint64_t>(-product)
        : static_cast<uint64_t>(product);
    const uint32_t fp_bits = clip_u64_to_fp32_bits_with_shift(
        abs_product,
        sign_bit,
        scale_shift.shift,
        &out.sat_fp_exp,
        &out.flush_to_zero_exp,
        &out.rounded_to_zero);
    out.value = clip_bits_to_f32(fp_bits);
    return out;
}

static clip_aicas_dequant_result clip_dequantize_i32_aicas(
        int32_t acc,
        float scale,
        clip_aicas_dequant_sim_mode mode) {
    switch (mode) {
        case clip_aicas_dequant_sim_mode::off:
            return {(float) acc * scale, false, false, false, false, false, false};
        case clip_aicas_dequant_sim_mode::versa_q8_24:
            return clip_dequantize_i32_versa_q8_24(acc, scale);
        case clip_aicas_dequant_sim_mode::versa_q8_24_fp_reconstruct:
            return clip_dequantize_i32_versa_q8_24_fp_reconstruct(acc, scale);
        case clip_aicas_dequant_sim_mode::scale_shift_i32_round:
            return clip_dequantize_i32_scale_shift_i32_round(acc, scale);
        case clip_aicas_dequant_sim_mode::scale_shift_fp_reconstruct:
            return clip_dequantize_i32_scale_shift_fp_reconstruct(acc, scale);
    }

    return {(float) acc * scale, false, false, false, false, false, false};
}

struct clip_aicas_activation_stats {
    uint64_t count = 0;
    float min = std::numeric_limits<float>::infinity();
    float max = -std::numeric_limits<float>::infinity();
    int64_t in_channels = 0;
    uint64_t seen_for_reservoir = 0;
    uint64_t reservoir_state = 0x9e3779b97f4a7c15ULL;
    size_t sample_limit = 0;
    std::vector<float> samples;
    std::vector<uint32_t> sample_channels;
    std::vector<float> per_channel_min;
    std::vector<float> per_channel_max;
    std::vector<float> per_channel_absmax;
    std::vector<uint64_t> record_counts;
    std::vector<float> record_absmax;
    std::vector<float> record_min_abs_nonzero;
    std::vector<int> record_bfp16m_exp;

    void ensure_channel_buffers(size_t channels) {
        if (channels == 0) {
            return;
        }
        if (in_channels == 0) {
            in_channels = static_cast<int64_t>(channels);
        }
        GGML_ASSERT(in_channels == static_cast<int64_t>(channels));
        if (per_channel_min.empty()) {
            per_channel_min.assign(channels, std::numeric_limits<float>::infinity());
            per_channel_max.assign(channels, -std::numeric_limits<float>::infinity());
            per_channel_absmax.assign(channels, 0.0f);
        }
    }

    void update(const float * data, size_t channels, size_t cols) {
        if (data == nullptr || channels == 0 || cols == 0) {
            return;
        }

        ensure_channel_buffers(channels);

        const size_t n = channels * cols;
        count += n;
        float rec_absmax = 0.0f;
        float rec_min_abs_nonzero = std::numeric_limits<float>::infinity();
        for (size_t col = 0; col < cols; ++col) {
            const float * col_ptr = data + col * channels;
            for (size_t ch = 0; ch < channels; ++ch) {
                const float v = col_ptr[ch];
                const float av = std::fabs(v);
                min = std::min(min, v);
                max = std::max(max, v);
                per_channel_min[ch] = std::min(per_channel_min[ch], v);
                per_channel_max[ch] = std::max(per_channel_max[ch], v);
                per_channel_absmax[ch] = std::max(per_channel_absmax[ch], av);
                rec_absmax = std::max(rec_absmax, av);
                if (av > 0.0f) {
                    rec_min_abs_nonzero = std::min(rec_min_abs_nonzero, av);
                }

                if (sample_limit == 0) {
                    continue;
                }

                ++seen_for_reservoir;
                if (samples.size() < sample_limit) {
                    samples.push_back(v);
                    sample_channels.push_back(static_cast<uint32_t>(ch));
                    continue;
                }

                reservoir_state = reservoir_state * 6364136223846793005ULL + 1;
                const uint64_t slot = reservoir_state % seen_for_reservoir;
                if (slot < sample_limit) {
                    samples[(size_t) slot] = v;
                    sample_channels[(size_t) slot] = static_cast<uint32_t>(ch);
                }
            }
        }

        int rec_exp = 0;
        if (rec_absmax > 0.0f && std::isfinite(rec_absmax)) {
            rec_exp = (int) std::ceil(std::log2((double) rec_absmax / 32767.0));
        }
        record_counts.push_back((uint64_t) n);
        record_absmax.push_back(rec_absmax);
        record_min_abs_nonzero.push_back(
            std::isfinite(rec_min_abs_nonzero) ? rec_min_abs_nonzero : 0.0f);
        record_bfp16m_exp.push_back(rec_exp);
    }
};

struct clip_aicas_activation_observer {
    clip_ctx * owner = nullptr;
    std::string tensor_name;
};

struct clip_hparams {
    int32_t image_size;
    int32_t patch_size;
    int32_t n_embd;
    int32_t n_ff;
    int32_t projection_dim;
    int32_t n_head;
    int32_t n_layer;
    // idefics3
    int32_t preproc_image_size = 0;
    int32_t proj_scale_factor = 0;

    float image_mean[3];
    float image_std[3];

    // for models using dynamic image size, we need to have a smaller image size to warmup
    // otherwise, user will get OOM everytime they load the model
    int32_t warmup_image_size = 0;
    int32_t warmup_audio_size = 3000;

    ffn_op_type ffn_op = FFN_GELU;

    patch_merge_type mm_patch_merge_type = PATCH_MERGE_FLAT;

    float eps = 1e-6;
    float rope_theta = 0.0;

    std::vector<clip_image_size> image_res_candidates; // for llava-uhd style models
    int32_t image_crop_resolution;
    std::unordered_set<int32_t> vision_feature_layer;
    int32_t attn_window_size = 0;
    int32_t n_wa_pattern = 0;
    int32_t spatial_merge_size = 0;

    // audio
    int32_t n_mel_bins = 0; // whisper preprocessor
    int32_t proj_stack_factor = 0; // ultravox

    // legacy
    bool has_llava_projector = false;
    int minicpmv_version = 0;
    int32_t minicpmv_query_num = 0;         // MiniCPM-V query number
};

struct clip_layer {
    // attention
    ggml_tensor * k_w = nullptr;
    ggml_tensor * k_b = nullptr;
    ggml_tensor * q_w = nullptr;
    ggml_tensor * q_b = nullptr;
    ggml_tensor * v_w = nullptr;
    ggml_tensor * v_b = nullptr;

    ggml_tensor * o_w = nullptr;
    ggml_tensor * o_b = nullptr;

    ggml_tensor * k_norm = nullptr;
    ggml_tensor * q_norm = nullptr;

    // layernorm 1
    ggml_tensor * ln_1_w = nullptr;
    ggml_tensor * ln_1_b = nullptr;

    ggml_tensor * ff_up_w = nullptr;
    ggml_tensor * ff_up_b = nullptr;
    ggml_tensor * ff_gate_w = nullptr;
    ggml_tensor * ff_gate_b = nullptr;
    ggml_tensor * ff_down_w = nullptr;
    ggml_tensor * ff_down_b = nullptr;

    // layernorm 2
    ggml_tensor * ln_2_w = nullptr;
    ggml_tensor * ln_2_b = nullptr;

    // layer scale (no bias)
    ggml_tensor * ls_1_w = nullptr;
    ggml_tensor * ls_2_w = nullptr;
};

struct clip_model {
    clip_modality modality = CLIP_MODALITY_VISION;
    projector_type proj_type = PROJECTOR_TYPE_MLP;
    clip_hparams hparams;

    // embeddings
    ggml_tensor * class_embedding = nullptr;
    ggml_tensor * patch_embeddings_0 = nullptr;
    ggml_tensor * patch_embeddings_1 = nullptr;  // second Conv2D kernel when we decouple Conv3D along temproal dimension (Qwen2VL)
    ggml_tensor * patch_bias = nullptr;
    ggml_tensor * position_embeddings = nullptr;

    ggml_tensor * pre_ln_w = nullptr;
    ggml_tensor * pre_ln_b = nullptr;

    std::vector<clip_layer> layers;

    ggml_tensor * post_ln_w;
    ggml_tensor * post_ln_b;

    ggml_tensor * projection; // TODO: rename it to fc (fully connected layer)
    ggml_tensor * mm_fc_w;
    ggml_tensor * mm_fc_b;

    // LLaVA projection
    ggml_tensor * mm_input_norm_w = nullptr;
    ggml_tensor * mm_input_norm_b = nullptr;
    ggml_tensor * mm_0_w = nullptr;
    ggml_tensor * mm_0_b = nullptr;
    ggml_tensor * mm_2_w = nullptr;
    ggml_tensor * mm_2_b = nullptr;

    ggml_tensor * image_newline = nullptr;

    // Yi type models with mlp+normalization projection
    ggml_tensor * mm_1_w = nullptr; // Yi type models have 0, 1, 3, 4
    ggml_tensor * mm_1_b = nullptr;
    ggml_tensor * mm_3_w = nullptr;
    ggml_tensor * mm_3_b = nullptr;
    ggml_tensor * mm_4_w = nullptr;
    ggml_tensor * mm_4_b = nullptr;

    // GLMV-Edge projection
    ggml_tensor * mm_model_adapter_conv_w = nullptr;
    ggml_tensor * mm_model_adapter_conv_b = nullptr;
    ggml_tensor * mm_glm_tok_boi = nullptr;
    ggml_tensor * mm_glm_tok_eoi = nullptr;

    // MobileVLM projection
    ggml_tensor * mm_model_mlp_1_w = nullptr;
    ggml_tensor * mm_model_mlp_1_b = nullptr;
    ggml_tensor * mm_model_mlp_3_w = nullptr;
    ggml_tensor * mm_model_mlp_3_b = nullptr;
    ggml_tensor * mm_model_block_1_block_0_0_w = nullptr;
    ggml_tensor * mm_model_block_1_block_0_1_w = nullptr;
    ggml_tensor * mm_model_block_1_block_0_1_b = nullptr;
    ggml_tensor * mm_model_block_1_block_1_fc1_w = nullptr;
    ggml_tensor * mm_model_block_1_block_1_fc1_b = nullptr;
    ggml_tensor * mm_model_block_1_block_1_fc2_w = nullptr;
    ggml_tensor * mm_model_block_1_block_1_fc2_b = nullptr;
    ggml_tensor * mm_model_block_1_block_2_0_w = nullptr;
    ggml_tensor * mm_model_block_1_block_2_1_w = nullptr;
    ggml_tensor * mm_model_block_1_block_2_1_b = nullptr;
    ggml_tensor * mm_model_block_2_block_0_0_w = nullptr;
    ggml_tensor * mm_model_block_2_block_0_1_w = nullptr;
    ggml_tensor * mm_model_block_2_block_0_1_b = nullptr;
    ggml_tensor * mm_model_block_2_block_1_fc1_w = nullptr;
    ggml_tensor * mm_model_block_2_block_1_fc1_b = nullptr;
    ggml_tensor * mm_model_block_2_block_1_fc2_w = nullptr;
    ggml_tensor * mm_model_block_2_block_1_fc2_b = nullptr;
    ggml_tensor * mm_model_block_2_block_2_0_w = nullptr;
    ggml_tensor * mm_model_block_2_block_2_1_w = nullptr;
    ggml_tensor * mm_model_block_2_block_2_1_b = nullptr;

    // MobileVLM_V2 projection
    ggml_tensor * mm_model_mlp_0_w = nullptr;
    ggml_tensor * mm_model_mlp_0_b = nullptr;
    ggml_tensor * mm_model_mlp_2_w = nullptr;
    ggml_tensor * mm_model_mlp_2_b = nullptr;
    ggml_tensor * mm_model_peg_0_w = nullptr;
    ggml_tensor * mm_model_peg_0_b = nullptr;

    // MINICPMV projection
    ggml_tensor * mm_model_pos_embed_k = nullptr;
    ggml_tensor * mm_model_query = nullptr;
    ggml_tensor * mm_model_proj = nullptr;
    ggml_tensor * mm_model_kv_proj = nullptr;
    ggml_tensor * mm_model_attn_q_w = nullptr;
    ggml_tensor * mm_model_attn_q_b = nullptr;
    ggml_tensor * mm_model_attn_k_w = nullptr;
    ggml_tensor * mm_model_attn_k_b = nullptr;
    ggml_tensor * mm_model_attn_v_w = nullptr;
    ggml_tensor * mm_model_attn_v_b = nullptr;
    ggml_tensor * mm_model_attn_o_w = nullptr;
    ggml_tensor * mm_model_attn_o_b = nullptr;
    ggml_tensor * mm_model_ln_q_w = nullptr;
    ggml_tensor * mm_model_ln_q_b = nullptr;
    ggml_tensor * mm_model_ln_kv_w = nullptr;
    ggml_tensor * mm_model_ln_kv_b = nullptr;
    ggml_tensor * mm_model_ln_post_w = nullptr;
    ggml_tensor * mm_model_ln_post_b = nullptr;

    // gemma3
    ggml_tensor * mm_input_proj_w = nullptr;
    ggml_tensor * mm_soft_emb_norm_w = nullptr;

    // pixtral
    ggml_tensor * token_embd_img_break = nullptr;
    ggml_tensor * mm_patch_merger_w = nullptr;

    // ultravox / whisper encoder
    ggml_tensor * conv1d_1_w = nullptr;
    ggml_tensor * conv1d_1_b = nullptr;
    ggml_tensor * conv1d_2_w = nullptr;
    ggml_tensor * conv1d_2_b = nullptr;
    ggml_tensor * mm_norm_pre_w = nullptr;
    ggml_tensor * mm_norm_mid_w = nullptr;

    bool aicas_w8a8_enabled = false;
    std::string aicas_w8a8_schema;
    std::unordered_map<std::string, clip_aicas_w8a8_tensor> aicas_w8a8_tensors;
    std::unordered_map<std::string, ggml_tensor *> aicas_smooth_scale_tensors;

    bool audio_has_avgpool() const {
        return proj_type == PROJECTOR_TYPE_QWEN2A
            || proj_type == PROJECTOR_TYPE_VOXTRAL;
    }

    bool audio_has_stack_frames() const {
        return proj_type == PROJECTOR_TYPE_ULTRAVOX
            || proj_type == PROJECTOR_TYPE_VOXTRAL;
    }
};

static void clip_compute_w8a8_mul_mat(
        struct ggml_tensor * dst,
        const struct ggml_tensor * a,
        const struct ggml_tensor * b,
        const struct ggml_tensor * c,
        int ith,
        int nth,
        void * userdata) {
    GGML_UNUSED(a);

    const auto * kernel_userdata = static_cast<const clip_aicas_w8a8_kernel_userdata *>(userdata);
    GGML_ASSERT(kernel_userdata != nullptr);
    GGML_ASSERT(kernel_userdata->cfg != nullptr);
    const auto * cfg = kernel_userdata->cfg;
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(b->type == GGML_TYPE_F32);
    GGML_ASSERT(c->type == GGML_TYPE_I8);

    const int64_t k = c->ne[0];
    const int64_t out_channels = c->ne[1];
    const int64_t n_cols = b->ne[1];

    GGML_ASSERT(b->ne[0] == k);
    GGML_ASSERT(dst->ne[0] == out_channels);
    GGML_ASSERT(dst->ne[1] == n_cols);
    GGML_ASSERT(cfg->weight_scale.size() == cfg->expected_weight_scale_len(out_channels));
    GGML_ASSERT(cfg->has_valid_compensation_config(out_channels));
    GGML_ASSERT(cfg->has_valid_smooth_config(k));
    GGML_ASSERT(cfg->act_scale > 0.0f);

    const int64_t cols_per_thread = (n_cols + nth - 1) / nth;
    const int64_t col_begin = ith * cols_per_thread;
    const int64_t col_end = std::min(n_cols, col_begin + cols_per_thread);
    if (col_begin >= col_end) {
        return;
    }

    const float sa = cfg->act_scale;
    const int32_t za = cfg->act_zero_point;
    const clip_aicas_dequant_sim_mode dequant_sim_mode = kernel_userdata->mode;
    std::vector<int8_t> act_i8(k);

    for (int64_t col = col_begin; col < col_end; ++col) {
        const float * act_col = (const float *) ((const char *) b->data + col * b->nb[1]);
        float * out_col = (float *) ((char *) dst->data + col * dst->nb[1]);

        for (int64_t i = 0; i < k; ++i) {
            const float smooth_scale = cfg->smooth_enabled
                ? cfg->smooth_scale[static_cast<size_t>(i)]
                : 1.0f;
            const float act_value = act_col[i] / smooth_scale;
            int32_t q = (int32_t) lrintf(act_value / sa) + za;
            q = std::max(0, std::min(255, q));
            act_i8[i] = (int8_t) (q - 128);
        }

        for (int64_t j = 0; j < out_channels; ++j) {
            const int8_t * w_col = (const int8_t *) ((const char *) c->data + j * c->nb[1]);
            int32_t acc = 0;
            for (int64_t i = 0; i < k; ++i) {
                acc += (int32_t) act_i8[i] * (int32_t) w_col[i];
            }
            const float weight_scale = cfg->uses_per_tensor_weight_scale()
                ? cfg->weight_scale[0]
                : cfg->weight_scale[j];
            const int32_t correction = (!cfg->sum_w.empty() && !cfg->bias_compensated)
                ? (cfg->act_zero_point - 128) * cfg->sum_w[static_cast<size_t>(j)]
                : 0;
            const float dequant_scale = sa * weight_scale;
            const clip_aicas_dequant_result dequant_result = clip_dequantize_i32_aicas(acc - correction, dequant_scale, dequant_sim_mode);
            clip_record_aicas_dequant_result(kernel_userdata->stats, dequant_result);
            out_col[j] = dequant_result.value;
        }
    }
}

static void clip_collect_activation_f32_passthrough(
        struct ggml_tensor * dst,
        const struct ggml_tensor * a,
        int ith,
        int nth,
        void * userdata);

static int clip_get_max_nodes() {
    constexpr int default_max_nodes = 32768;

    if (const char * env = std::getenv("MTMD_MAX_NODES")) {
        const long parsed = std::strtol(env, nullptr, 10);
        if (parsed > 0 && parsed <= std::numeric_limits<int>::max()) {
            return static_cast<int>(parsed);
        }

        LOG_WRN("%s: ignoring invalid MTMD_MAX_NODES=%s, using default=%d\n",
                __func__, env, default_max_nodes);
    }

    return default_max_nodes;
}

struct clip_ctx {
    clip_model model;

    gguf_context_ptr ctx_gguf;
    ggml_context_ptr ctx_data;

    std::vector<uint8_t> buf_compute_meta;

    std::vector<ggml_backend_t> backend_ptrs;
    std::vector<ggml_backend_buffer_type_t> backend_buft;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_buffer_ptr buf;

    int max_nodes = clip_get_max_nodes();
    ggml_backend_sched_ptr sched;

    // for debugging
    bool debug_graph = false;
    bool debug_dump_dot_done = false;
    std::string debug_dump_dot_path;
    std::string debug_dump_w8a8_tensors_dir;
    size_t debug_dump_w8a8_tensors_max = std::numeric_limits<size_t>::max();
    std::vector<std::string> npu_w8a8_skip_contains;
    clip_profiler profiler;
    std::string last_mmproj_summary_json;
    std::vector<ggml_tensor *> debug_print_tensors;
    std::vector<ggml_tensor *> debug_dump_w8a8_tensors;
    bool aicas_w8a8_debug = false;
    clip_aicas_dequant_sim_mode aicas_dequant_sim_mode = clip_aicas_dequant_sim_mode::off;
    clip_mmproj_attn_precision aicas_mmproj_attn_precision = clip_mmproj_attn_precision::f32;
    clip_mmproj_attn_precision_scope aicas_mmproj_attn_precision_scope = clip_mmproj_attn_precision_scope::core;
    int64_t aicas_mmproj_bfp16m_k_block = 64;
    clip_aicas_bfp16m_exp_mode aicas_bfp16m_exp_mode = clip_aicas_bfp16m_exp_mode::kblock;
    std::string aicas_dequant_stats_path;
    mutable clip_aicas_dequant_diag_counters aicas_dequant_stats;
    std::string aicas_act_stats_path;
    size_t aicas_act_stats_samples_per_tensor = 0;
    mutable std::mutex aicas_act_stats_mutex;
    mutable std::unordered_map<std::string, clip_aicas_activation_stats> aicas_act_stats;
    mutable std::unordered_map<std::string, clip_aicas_activation_observer> aicas_act_observers;
    mutable std::unordered_map<std::string, std::unique_ptr<clip_aicas_w8a8_kernel_userdata>> aicas_w8a8_kernel_userdata_map;
    mutable clip_aicas_bfp16m_userdata aicas_bfp16m_userdata;
    mutable std::vector<std::unique_ptr<clip_aicas_bfp16m_userdata>> aicas_bfp16m_node_userdata;

    clip_ctx(clip_context_params & ctx_params) {
        debug_graph = std::getenv("MTMD_DEBUG_GRAPH") != nullptr;
        const char * dump_dot = std::getenv("MTMD_DUMP_DOT");
        debug_dump_dot_path = dump_dot ? dump_dot : "";
        const char * dump_w8a8_dir = std::getenv("MTMD_DUMP_W8A8_TENSORS_DIR");
        debug_dump_w8a8_tensors_dir = dump_w8a8_dir ? dump_w8a8_dir : "";
        if (const char * dump_w8a8_max = std::getenv("MTMD_DUMP_W8A8_TENSORS_MAX")) {
            const long parsed = std::strtol(dump_w8a8_max, nullptr, 10);
            if (parsed >= 0) {
                debug_dump_w8a8_tensors_max = static_cast<size_t>(parsed);
            }
        }
        if (const char * skip_env = std::getenv("MTMD_NPU_W8A8_SKIP_CONTAINS")) {
            std::stringstream ss(skip_env);
            std::string item;
            while (std::getline(ss, item, ',')) {
                item.erase(item.begin(), std::find_if(item.begin(), item.end(), [](unsigned char ch) {
                    return !std::isspace(ch);
                }));
                item.erase(std::find_if(item.rbegin(), item.rend(), [](unsigned char ch) {
                    return !std::isspace(ch);
                }).base(), item.end());
                if (!item.empty()) {
                    npu_w8a8_skip_contains.push_back(item);
                }
            }
        }
        profiler.init_from_env();
        aicas_w8a8_debug = std::getenv("AICAS_MMPROJ_W8A8_DEBUG") != nullptr;
        aicas_dequant_sim_mode = clip_get_dequant_sim_mode();
        aicas_mmproj_attn_precision = clip_get_mmproj_attn_precision();
        aicas_mmproj_attn_precision_scope = clip_get_mmproj_attn_precision_scope();
        aicas_mmproj_bfp16m_k_block = clip_get_mmproj_bfp16m_k_block();
        aicas_bfp16m_exp_mode = clip_get_bfp16m_exp_mode();
        aicas_bfp16m_userdata.k_block = aicas_mmproj_bfp16m_k_block;
        aicas_bfp16m_userdata.exp_mode = aicas_bfp16m_exp_mode;
        if (aicas_dequant_sim_mode != clip_aicas_dequant_sim_mode::off) {
            LOG_INF("%s: AICAS mmproj dequant simulation enabled: %s\n",
                __func__,
                clip_dequant_sim_mode_name(aicas_dequant_sim_mode));
        }
        if (aicas_mmproj_attn_precision != clip_mmproj_attn_precision::f32) {
            LOG_INF("%s: AICAS mmproj attention precision enabled: %s scope=%s\n",
                __func__,
                clip_mmproj_attn_precision_name(aicas_mmproj_attn_precision),
                clip_mmproj_attn_precision_scope_name(aicas_mmproj_attn_precision_scope));
        }
        if (aicas_mmproj_attn_precision == clip_mmproj_attn_precision::bfp16m) {
            LOG_INF("%s: AICAS mmproj BFP16-M k_block=%" PRId64 " exp_mode=%s\n",
                __func__, aicas_mmproj_bfp16m_k_block, clip_aicas_bfp16m_exp_mode_name(aicas_bfp16m_exp_mode));
        }
        if (const char * stats_path = std::getenv("AICAS_MMPROJ_DEQUANT_STATS_FILE")) {
            aicas_dequant_stats_path = stats_path;
        }
        if (const char * stats_path = std::getenv("AICAS_MMPROJ_ACT_STATS_FILE")) {
            aicas_act_stats_path = stats_path;
            aicas_act_stats_samples_per_tensor = 4096;
            if (const char * samples_env = std::getenv("AICAS_MMPROJ_ACT_SAMPLES")) {
                const long parsed = strtol(samples_env, nullptr, 10);
                if (parsed > 0) {
                    aicas_act_stats_samples_per_tensor = (size_t) parsed;
                }
            }
        }
        backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (!backend_cpu) {
            throw std::runtime_error("failed to initialize CPU backend");
        }
        if (ctx_params.use_gpu) {
            auto backend_name = std::getenv("MTMD_BACKEND_DEVICE");
            if (backend_name != nullptr) {
#ifdef GGML_USE_NPU
                if (std::strcmp(backend_name, "NPU") == 0 || std::strcmp(backend_name, "NPU0") == 0) {
                    backend = ggml_backend_npu_init();
                } else {
                    backend = ggml_backend_init_by_name(backend_name, nullptr);
                }
#else
                backend = ggml_backend_init_by_name(backend_name, nullptr);
#endif
                if (!backend) {
                    LOG_WRN("%s: Warning: Failed to initialize \"%s\" backend, falling back to default GPU backend\n", __func__, backend_name);
                }
            }
            if (!backend) {
                backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
                backend = backend ? backend : ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU, nullptr);
            }
        }

        if (backend) {
            LOG_INF("%s: CLIP using %s backend\n", __func__, ggml_backend_name(backend));
            backend_ptrs.push_back(backend);
            backend_buft.push_back(ggml_backend_get_default_buffer_type(backend));
        } else {
            backend = backend_cpu;
            LOG_INF("%s: CLIP using CPU backend\n", __func__);
        }

        backend_ptrs.push_back(backend_cpu);
        backend_buft.push_back(ggml_backend_get_default_buffer_type(backend_cpu));

        sched.reset(
            ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, false, true)
        );
    }

    ~clip_ctx() {
        flush_aicas_dequant_stats();
        flush_aicas_activation_stats();
        ggml_backend_free(backend);
        if (backend != backend_cpu) {
            ggml_backend_free(backend_cpu);
        }
    }

    bool backend_is_npu() const {
#ifdef GGML_USE_NPU
        return ggml_backend_is_npu(backend);
#else
        return false;
#endif
    }

    bool should_route_w8a8_to_npu(const std::string & weight_name) const {
        if (!backend_is_npu()) {
            return false;
        }
        for (const std::string & needle : npu_w8a8_skip_contains) {
            if (weight_name.find(needle) != std::string::npos) {
                return false;
            }
        }
        return true;
    }

    bool should_preload_aicas_w8a8_for_npu() const {
        const char * v = std::getenv("GGML_NPU_PRELOAD_WEIGHTS_ON_LOAD");
        return v != nullptr && v[0] != '\0' && std::strcmp(v, "0") != 0;
    }

    bool register_aicas_w8a8_for_npu() const {
#ifdef GGML_USE_NPU
        if (!backend_is_npu() || !model.aicas_w8a8_enabled) {
            return true;
        }

        int registered = 0;
        int failed = 0;
        for (const auto & kv : model.aicas_w8a8_tensors) {
            const std::string & tensor_name = kv.first;
            const auto & cfg = kv.second;
            if (!cfg.enabled || cfg.policy != "W8A8") {
                continue;
            }
            if (!cfg.is_npu_compatible(-1)) {
                LOG_WRN(
                    "%s: tensor %s is not compatible with current NPU W8A8 path (act_quant_mode=%s, weight_scale_mode=%s, scale_len=%zu, sumw_len=%zu)\n",
                    __func__,
                    tensor_name.c_str(),
                    cfg.act_quant_mode.c_str(),
                    cfg.weight_scale_mode.c_str(),
                    cfg.weight_scale.size(),
                    cfg.sum_w.size());
                ++failed;
                continue;
            }
            if (cfg.weight_scale.empty() || cfg.sum_w.empty()) {
                ++failed;
                continue;
            }

            const bool ok = ggml_backend_npu_w8a8_register(
                tensor_name.c_str(),
                cfg.act_scale,
                cfg.act_scale_q8_24,
                cfg.act_zero_point,
                cfg.weight_scale.data(),
                cfg.weight_scale.size(),
                cfg.sum_w.data(),
                cfg.sum_w.size(),
                cfg.smooth_scale.empty() ? nullptr : cfg.smooth_scale.data(),
                cfg.smooth_scale.size());
            if (ok) {
                ++registered;
            } else {
                ++failed;
            }
        }

        LOG_INF("%s: registered %d AICAS W8A8 tensors for NPU (%d failed)\n", __func__, registered, failed);
        return failed == 0;
#else
        return false;
#endif
    }

    bool preload_aicas_w8a8_for_npu() const {
#ifdef GGML_USE_NPU
        if (!backend_is_npu() || !model.aicas_w8a8_enabled || ctx_data == nullptr) {
            return true;
        }

        int preloaded = 0;
        int failed = 0;
        for (const auto & kv : model.aicas_w8a8_tensors) {
            const std::string & tensor_name = kv.first;
            const auto & cfg = kv.second;
            if (!cfg.enabled || cfg.policy != "W8A8" || !cfg.is_npu_compatible(-1)) {
                continue;
            }

            ggml_tensor * weight = ggml_get_tensor(ctx_data.get(), tensor_name.c_str());
            if (weight == nullptr) {
                ++failed;
                continue;
            }

            if (ggml_backend_npu_w8a8_preload(weight)) {
                ++preloaded;
            } else {
                ++failed;
            }
        }

        LOG_INF("%s: preloaded %d AICAS W8A8 tensors into NPU CMA (%d failed)\n", __func__, preloaded, failed);
        return failed == 0;
#else
        return false;
#endif
    }

    void force_cpu_backend_for_w8a8() {
        if (backend == backend_cpu) {
            return;
        }

        LOG_WRN("%s: AICAS W8A8 metadata detected, forcing mmproj to CPU backend\n", __func__);
        ggml_backend_free(backend);
        backend = backend_cpu;

        backend_ptrs.clear();
        backend_buft.clear();
        backend_ptrs.push_back(backend_cpu);
        backend_buft.push_back(ggml_backend_get_default_buffer_type(backend_cpu));

        sched.reset(
            ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, false, true)
        );
    }

    // this function is added so that we don't change too much of the existing code
    projector_type proj_type() const {
        return model.proj_type;
    }

    clip_aicas_w8a8_kernel_userdata * get_aicas_w8a8_kernel_userdata(
            const std::string & tensor_name,
            const clip_aicas_w8a8_tensor * cfg) const {
        auto it = aicas_w8a8_kernel_userdata_map.find(tensor_name);
        if (it == aicas_w8a8_kernel_userdata_map.end()) {
            auto userdata = std::make_unique<clip_aicas_w8a8_kernel_userdata>();
            userdata->cfg = cfg;
            userdata->mode = aicas_dequant_sim_mode;
            userdata->stats = const_cast<clip_aicas_dequant_diag_counters *>(&aicas_dequant_stats);
            it = aicas_w8a8_kernel_userdata_map.emplace(tensor_name, std::move(userdata)).first;
        }

        return it->second.get();
    }

    clip_aicas_activation_observer * get_aicas_activation_observer(const std::string & tensor_name) const {
        if (aicas_act_stats_path.empty()) {
            return nullptr;
        }

        std::lock_guard<std::mutex> lock(aicas_act_stats_mutex);

        auto stats_it = aicas_act_stats.find(tensor_name);
        if (stats_it == aicas_act_stats.end()) {
            clip_aicas_activation_stats stats;
            stats.sample_limit = aicas_act_stats_samples_per_tensor;
            stats_it = aicas_act_stats.emplace(tensor_name, std::move(stats)).first;
        }

        auto obs_it = aicas_act_observers.find(tensor_name);
        if (obs_it == aicas_act_observers.end()) {
            clip_aicas_activation_observer observer;
            observer.owner = const_cast<clip_ctx *>(this);
            observer.tensor_name = tensor_name;
            obs_it = aicas_act_observers.emplace(tensor_name, std::move(observer)).first;
        }

        GGML_UNUSED(stats_it);
        return &obs_it->second;
    }

    void record_aicas_activation(
            const std::string & tensor_name,
            const float * data,
            size_t channels,
            size_t cols) {
        if (aicas_act_stats_path.empty()) {
            return;
        }

        std::lock_guard<std::mutex> lock(aicas_act_stats_mutex);
        auto & stats = aicas_act_stats[tensor_name];
        if (stats.sample_limit == 0) {
            stats.sample_limit = aicas_act_stats_samples_per_tensor;
        }
        stats.update(data, channels, cols);
    }

    void flush_aicas_activation_stats() const {
        if (aicas_act_stats_path.empty()) {
            return;
        }

        json out = {
            {"schema", "aicas.mmproj.act_stats.v1"},
            {"samples_per_tensor", aicas_act_stats_samples_per_tensor},
            {"tensors", json::array()},
        };

        std::vector<std::string> names;
        {
            std::lock_guard<std::mutex> lock(aicas_act_stats_mutex);
            names.reserve(aicas_act_stats.size());
            for (const auto & kv : aicas_act_stats) {
                names.push_back(kv.first);
            }
            std::sort(names.begin(), names.end());

            for (const auto & name : names) {
                const auto & stats = aicas_act_stats.at(name);
                if (stats.count == 0) {
                    continue;
                }

                out["tensors"].push_back({
                    {"tensor_name", name},
                    {"count", stats.count},
                    {"min", stats.min},
                    {"max", stats.max},
                    {"in_channels", stats.in_channels},
                    {"per_channel_min", stats.per_channel_min},
                    {"per_channel_max", stats.per_channel_max},
                    {"per_channel_absmax", stats.per_channel_absmax},
                    {"record_counts", stats.record_counts},
                    {"record_absmax", stats.record_absmax},
                    {"record_min_abs_nonzero", stats.record_min_abs_nonzero},
                    {"record_bfp16m_exp", stats.record_bfp16m_exp},
                    {"samples", stats.samples},
                    {"sample_channels", stats.sample_channels},
                });
            }
        }

        std::ofstream fout(aicas_act_stats_path, std::ios::binary);
        if (!fout.is_open()) {
            LOG_ERR("%s: failed to open activation stats file: %s\n", __func__, aicas_act_stats_path.c_str());
            return;
        }
        fout << out.dump(2);
    }

    void flush_aicas_dequant_stats() const {
        if (aicas_dequant_sim_mode == clip_aicas_dequant_sim_mode::off) {
            return;
        }

        const json counters = aicas_dequant_stats.to_json();
        LOG_INF(
            "%s: AICAS mmproj dequant stats mode=%s scale_zero=%" PRIu64 " rounded_to_zero=%" PRIu64 " sat_i32=%" PRIu64 " sat_fp_exp=%" PRIu64 " flush_to_zero_exp=%" PRIu64 " mul_overflow_guard_hit=%" PRIu64 "\n",
            __func__,
            clip_dequant_sim_mode_name(aicas_dequant_sim_mode),
            counters["scale_zero"].get<uint64_t>(),
            counters["rounded_to_zero"].get<uint64_t>(),
            counters["sat_i32"].get<uint64_t>(),
            counters["sat_fp_exp"].get<uint64_t>(),
            counters["flush_to_zero_exp"].get<uint64_t>(),
            counters["mul_overflow_guard_hit"].get<uint64_t>());

        if (aicas_dequant_stats_path.empty()) {
            return;
        }

        json out = {
            {"schema", "aicas.mmproj.dequant_stats.v1"},
            {"mode", clip_dequant_sim_mode_name(aicas_dequant_sim_mode)},
            {"counters", counters},
        };

        std::ofstream fout(aicas_dequant_stats_path, std::ios::binary);
        if (!fout.is_open()) {
            LOG_ERR("%s: failed to open dequant stats file: %s\n", __func__, aicas_dequant_stats_path.c_str());
            return;
        }
        fout << out.dump(2);
    }
};

struct clip_mmproj_summary_aggregate {
    int64_t node_count = 0;
    int64_t elements = 0;
    int64_t bytes = 0;
};

struct clip_mmproj_mul_mat_signature_aggregate {
    clip_profile_tensor_info src0;
    clip_profile_tensor_info src1;
    clip_profile_tensor_info dst;
    int64_t node_count = 0;
    int64_t elements = 0;
    int64_t bytes = 0;
    std::vector<std::string> example_node_names;
};

static bool clip_mmproj_light_summary_enabled() {
    const char * path = std::getenv("LLAMA_MTMD_PREFILL_SUMMARY_JSON");
    return path != nullptr && path[0] != '\0';
}

static std::string clip_to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

static std::string clip_backend_class(ggml_backend_t backend) {
    if (backend == nullptr) {
        return "unknown";
    }

    const char * name_c = ggml_backend_name(backend);
    if (name_c == nullptr || name_c[0] == '\0') {
        return "unknown";
    }

    const std::string name = clip_to_lower(name_c);
    if (name.find("npu") != std::string::npos) {
        return "npu";
    }
    if (name.find("cpu") != std::string::npos) {
        return "cpu";
    }
    return name;
}

static std::string clip_mmproj_op_category(const ggml_tensor * node, const std::string & backend_class) {
    const std::string op = node != nullptr ? ggml_op_desc(node) : "UNKNOWN";

    if (backend_class == "npu") {
        if (op == "MUL_MAT") {
            return "MUL_MAT_NPU";
        }
        return op + "_NPU";
    }

    if (backend_class != "cpu") {
        return op + "_" + backend_class;
    }

    if (op == "MUL_MAT" || op == "MUL_MAT_ID") {
        return "MUL_MAT_CPU";
    }
    if (op == "SOFT_MAX" || op == "FLASH_ATTN_EXT") {
        return "ATTENTION_CPU";
    }
    if (op == "GELU" || op == "SILU" || op == "RELU" || op == "GLU" || op == "UNARY") {
        return "ACTIVATION_CPU";
    }
    if (op == "RMS_NORM" || op == "NORM" || op == "GROUP_NORM" || op == "L2_NORM") {
        return "NORM_CPU";
    }
    if (op == "ADD" || op == "SUB" || op == "MUL" || op == "DIV" ||
            op == "SQR" || op == "SQRT" || op == "SCALE" || op == "CLAMP") {
        return "ELEMENTWISE_CPU";
    }
    if (op == "CONT" || op == "CPY" || op == "RESHAPE" || op == "VIEW" ||
            op == "PERMUTE" || op == "TRANSPOSE" || op == "GET_ROWS") {
        return "LAYOUT_CPU";
    }
    if (op == "POOL_1D" || op == "POOL_2D") {
        return "POOL_CPU";
    }
    return op + "_CPU";
}

static json clip_mmproj_aggregate_map_json(const std::map<std::string, clip_mmproj_summary_aggregate> & values) {
    std::vector<std::pair<std::string, clip_mmproj_summary_aggregate>> sorted(values.begin(), values.end());
    std::sort(sorted.begin(), sorted.end(), [](const auto & a, const auto & b) {
        if (a.second.bytes != b.second.bytes) {
            return a.second.bytes > b.second.bytes;
        }
        if (a.second.elements != b.second.elements) {
            return a.second.elements > b.second.elements;
        }
        return a.first < b.first;
    });

    json out = json::array();
    for (const auto & kv : sorted) {
        out.push_back({
            {"name", kv.first},
            {"node_count", kv.second.node_count},
            {"elements", kv.second.elements},
            {"bytes", kv.second.bytes},
        });
    }
    return out;
}

static std::string clip_mul_mat_signature_key(
        const clip_profile_tensor_info & src0,
        const clip_profile_tensor_info & src1,
        const clip_profile_tensor_info & dst) {
    return src0.type + "|" + clip_shape_key(src0.ne) + "|" +
           src1.type + "|" + clip_shape_key(src1.ne) + "|" +
           dst.type + "|" + clip_shape_key(dst.ne);
}

static json clip_mmproj_mul_mat_signatures_json(
        const std::map<std::string, clip_mmproj_mul_mat_signature_aggregate> & values) {
    std::vector<std::pair<std::string, clip_mmproj_mul_mat_signature_aggregate>> sorted(values.begin(), values.end());
    std::sort(sorted.begin(), sorted.end(), [](const auto & a, const auto & b) {
        if (a.second.bytes != b.second.bytes) {
            return a.second.bytes > b.second.bytes;
        }
        if (a.second.elements != b.second.elements) {
            return a.second.elements > b.second.elements;
        }
        return a.first < b.first;
    });

    json out = json::array();
    for (const auto & kv : sorted) {
        out.push_back({
            {"src0", clip_tensor_json(kv.second.src0)},
            {"src1", clip_tensor_json(kv.second.src1)},
            {"dst", clip_tensor_json(kv.second.dst)},
            {"node_count", kv.second.node_count},
            {"elements", kv.second.elements},
            {"bytes", kv.second.bytes},
            {"example_node_names", kv.second.example_node_names},
        });
    }
    return out;
}

static json clip_build_mmproj_graph_summary(clip_ctx * ctx, ggml_cgraph * gf) {
    std::map<std::string, clip_mmproj_summary_aggregate> by_backend;
    std::map<std::string, clip_mmproj_summary_aggregate> by_category;
    std::map<std::string, clip_mmproj_summary_aggregate> cpu_categories;
    std::map<std::string, clip_mmproj_summary_aggregate> npu_categories;
    std::map<std::string, clip_mmproj_mul_mat_signature_aggregate> cpu_mul_mat_signatures;

    const int node_count = gf != nullptr ? ggml_graph_n_nodes(gf) : 0;
    if (ctx != nullptr && gf != nullptr) {
        for (int i = 0; i < node_count; ++i) {
            ggml_tensor * node = ggml_graph_node(gf, i);
            if (node == nullptr) {
                continue;
            }

            ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(ctx->sched.get(), node);
            const std::string backend_class = clip_backend_class(backend);
            const std::string category = clip_mmproj_op_category(node, backend_class);

            clip_mmproj_summary_aggregate delta;
            delta.node_count = 1;
            delta.elements = ggml_nelements(node);
            delta.bytes = ggml_nbytes(node);

            auto add_delta = [&delta](clip_mmproj_summary_aggregate & dst) {
                dst.node_count += delta.node_count;
                dst.elements += delta.elements;
                dst.bytes += delta.bytes;
            };

            add_delta(by_backend[backend_class]);
            add_delta(by_category[category]);
            if (backend_class == "cpu") {
                add_delta(cpu_categories[category]);
                if (node->op == GGML_OP_MUL_MAT && node->src[0] != nullptr && node->src[1] != nullptr) {
                    const clip_profile_tensor_info src0 = clip_capture_tensor_info(node->src[0]);
                    const clip_profile_tensor_info src1 = clip_capture_tensor_info(node->src[1]);
                    const clip_profile_tensor_info dst = clip_capture_tensor_info(node);
                    auto & sig = cpu_mul_mat_signatures[clip_mul_mat_signature_key(src0, src1, dst)];
                    if (sig.node_count == 0) {
                        sig.src0 = src0;
                        sig.src1 = src1;
                        sig.dst = dst;
                    }
                    sig.node_count += 1;
                    sig.elements += delta.elements;
                    sig.bytes += delta.bytes;
                    if (sig.example_node_names.size() < 4) {
                        sig.example_node_names.emplace_back(node->name);
                    }
                }
            } else if (backend_class == "npu") {
                add_delta(npu_categories[category]);
            }
        }
    }

    return {
        {"profile_kind", "mmproj_non_intrusive_graph_summary"},
        {"instrumentation", "graph_assignment_no_eval_callback"},
        {"note", "Counts and byte sizes describe scheduled graph nodes, not per-node runtime. No eval callback is installed for this summary."},
        {"node_count", node_count},
        {"by_backend", clip_mmproj_aggregate_map_json(by_backend)},
        {"by_operator_category", clip_mmproj_aggregate_map_json(by_category)},
        {"cpu_operator_categories", clip_mmproj_aggregate_map_json(cpu_categories)},
        {"cpu_mul_mat_signatures", clip_mmproj_mul_mat_signatures_json(cpu_mul_mat_signatures)},
        {"npu_operator_categories", clip_mmproj_aggregate_map_json(npu_categories)},
    };
}

static bool clip_should_dump_w8a8_tensor(const clip_ctx * ctx, const ggml_tensor * t, int * index) {
    if (ctx == nullptr || ctx->debug_dump_w8a8_tensors_dir.empty() || t == nullptr) {
        return false;
    }
    for (size_t i = 0; i < ctx->debug_dump_w8a8_tensors.size(); ++i) {
        if (ctx->debug_dump_w8a8_tensors[i] == t) {
            if (index != nullptr) {
                *index = static_cast<int>(i);
            }
            return true;
        }
    }
    return false;
}

static bool clip_eval_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * ctx = static_cast<clip_ctx *>(user_data);
    if (ctx == nullptr) {
        return false;
    }

    int dump_index = -1;
    const bool should_dump = clip_should_dump_w8a8_tensor(ctx, t, &dump_index);
    if (ask) {
        if (ctx->profiler.enabled) {
            clip_profiler::eval_callback(t, ask, &ctx->profiler);
        }
        return ctx->profiler.enabled || should_dump;
    }

    if (should_dump) {
        clip_dump_tensor_f32_to_dir(ctx->debug_dump_w8a8_tensors_dir, t, dump_index);
    }
    if (ctx->profiler.enabled) {
        return clip_profiler::eval_callback(t, ask, &ctx->profiler);
    }
    return true;
}

static void clip_collect_activation_f32_passthrough(
        struct ggml_tensor * dst,
        const struct ggml_tensor * a,
        int ith,
        int nth,
        void * userdata) {
    GGML_UNUSED(nth);

    auto * observer = static_cast<clip_aicas_activation_observer *>(userdata);
    GGML_ASSERT(observer != nullptr);
    GGML_ASSERT(observer->owner != nullptr);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    if (ith != 0) {
        return;
    }

    const size_t nbytes = ggml_nbytes(a);
    GGML_ASSERT(nbytes == ggml_nbytes(dst));
    memcpy(dst->data, a->data, nbytes);
    observer->owner->record_aicas_activation(
        observer->tensor_name,
        (const float *) a->data,
        (size_t) a->ne[0],
        (size_t) ggml_nelements(a) / (size_t) a->ne[0]);
}

static void clip_bfp16m_quant_dequant_f32(
        struct ggml_tensor * dst,
        const struct ggml_tensor * a,
        int ith,
        int nth,
        void * userdata) {
    GGML_UNUSED(nth);

    const auto * cfg = static_cast<const clip_aicas_bfp16m_userdata *>(userdata);
    GGML_ASSERT(cfg != nullptr);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    if (ith != 0) {
        return;
    }

    const int64_t k_block = std::max<int64_t>(1, cfg->k_block);
    const int64_t ne0 = a->ne[0];
    const int64_t ne1 = a->ne[1];
    const int64_t ne2 = a->ne[2];
    const int64_t ne3 = a->ne[3];

    for (int64_t i3 = 0; i3 < ne3; ++i3) {
        for (int64_t i2 = 0; i2 < ne2; ++i2) {
            for (int64_t i1 = 0; i1 < ne1; ++i1) {
                for (int64_t k0 = 0; k0 < ne0; k0 += k_block) {
                    const int64_t k1 = std::min(ne0, k0 + k_block);
                    float max_abs = 0.0f;
                    for (int64_t i0 = k0; i0 < k1; ++i0) {
                        const float v = *(const float *) ((const char *) a->data +
                            i0 * a->nb[0] + i1 * a->nb[1] + i2 * a->nb[2] + i3 * a->nb[3]);
                        max_abs = std::max(max_abs, std::fabs(v));
                    }

                    const int exp = max_abs > 0.0f ? (int) std::ceil(std::log2((double) max_abs / 32767.0)) : 0;
                    const float scale = std::ldexp(1.0f, exp);
                    const float inv_scale = 1.0f / scale;

                    for (int64_t i0 = k0; i0 < k1; ++i0) {
                        const float v = *(const float *) ((const char *) a->data +
                            i0 * a->nb[0] + i1 * a->nb[1] + i2 * a->nb[2] + i3 * a->nb[3]);
                        long q = std::lrint((double) v * inv_scale);
                        q = std::max<long>(-32768, std::min<long>(32767, q));
                        *(float *) ((char *) dst->data +
                            i0 * dst->nb[0] + i1 * dst->nb[1] + i2 * dst->nb[2] + i3 * dst->nb[3]) =
                            (float) q * scale;
                    }
                }
            }
        }
    }
}

static inline float clip_tensor_get_f32_4d(
        const struct ggml_tensor * t,
        int64_t i0,
        int64_t i1,
        int64_t i2,
        int64_t i3) {
    return *(const float *) ((const char *) t->data +
        i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3]);
}

static inline void clip_tensor_set_f32_4d(
        struct ggml_tensor * t,
        int64_t i0,
        int64_t i1,
        int64_t i2,
        int64_t i3,
        float value) {
    *(float *) ((char *) t->data +
        i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3]) = value;
}

static inline int16_t clip_bfp16m_quant_value(float value, int exp) {
    const double scaled = std::ldexp((double) value, -exp);
    long q = std::lrint(scaled);
    q = std::max<long>(-32768, std::min<long>(32767, q));
    return (int16_t) q;
}

static inline int8_t clip_bfp16m_exp_to_i8(int exp) {
    return (int8_t) std::max<int>(std::numeric_limits<int8_t>::min(),
        std::min<int>(std::numeric_limits<int8_t>::max(), exp));
}

static inline int64_t clip_rshift_rne_i64(int64_t value, int shift) {
    if (shift <= 0 || value == 0) {
        return value;
    }
    if (shift >= 64) {
        return 0;
    }

    const bool neg = value < 0;
    const uint64_t abs_v = neg
        ? (uint64_t) (-(value + 1)) + 1
        : (uint64_t) value;
    const uint64_t q = abs_v >> shift;
    const uint64_t rem_mask = (UINT64_C(1) << shift) - 1;
    const uint64_t rem = abs_v & rem_mask;
    const uint64_t half = UINT64_C(1) << (shift - 1);
    uint64_t rounded = q;
    if (rem > half || (rem == half && (q & 1) != 0)) {
        rounded += 1;
    }

    if (neg && rounded == (UINT64_C(1) << 63)) {
        return std::numeric_limits<int64_t>::min();
    }
    if (rounded > (uint64_t) std::numeric_limits<int64_t>::max()) {
        return std::numeric_limits<int64_t>::max();
    }
    const int64_t signed_rounded = (int64_t) rounded;
    return neg ? -signed_rounded : signed_rounded;
}

static inline int64_t clip_saturating_add_i64(int64_t a, int64_t b) {
    if (b > 0 && a > std::numeric_limits<int64_t>::max() - b) {
        return std::numeric_limits<int64_t>::max();
    }
    if (b < 0 && a < std::numeric_limits<int64_t>::min() - b) {
        return std::numeric_limits<int64_t>::min();
    }
    return a + b;
}

static int clip_bfp16m_block_exp_for_a(
        const struct ggml_tensor * a,
        const clip_aicas_bfp16m_userdata * cfg,
        int64_t row,
        int64_t a_i2,
        int64_t a_i3,
        int64_t k0,
        int64_t k1) {
    if (cfg != nullptr && cfg->static_exp) {
        return cfg->static_exp_a;
    }
    float max_abs = 0.0f;
    for (int64_t k = k0; k < k1; ++k) {
        max_abs = std::max(max_abs, std::fabs(clip_tensor_get_f32_4d(a, k, row, a_i2, a_i3)));
    }
    return max_abs > 0.0f ? (int) std::ceil(std::log2((double) max_abs / 32767.0)) : 0;
}

static int clip_bfp16m_block_exp_for_b(
        const struct ggml_tensor * b,
        const clip_aicas_bfp16m_userdata * cfg,
        int64_t col,
        int64_t b_i2,
        int64_t b_i3,
        int64_t k0,
        int64_t k1) {
    if (cfg != nullptr && cfg->static_exp) {
        return cfg->static_exp_b;
    }
    float max_abs = 0.0f;
    for (int64_t k = k0; k < k1; ++k) {
        max_abs = std::max(max_abs, std::fabs(clip_tensor_get_f32_4d(b, k, col, b_i2, b_i3)));
    }
    return max_abs > 0.0f ? (int) std::ceil(std::log2((double) max_abs / 32767.0)) : 0;
}

static void clip_bfp16m_mul_mat_f32(
        struct ggml_tensor * dst,
        const struct ggml_tensor * out_template,
        const struct ggml_tensor * a,
        const struct ggml_tensor * b,
        int ith,
        int nth,
        void * userdata) {
    GGML_UNUSED(out_template);
    GGML_UNUSED(nth);

    const auto * cfg = static_cast<const clip_aicas_bfp16m_userdata *>(userdata);
    GGML_ASSERT(cfg != nullptr);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(a->type == GGML_TYPE_F32);
    GGML_ASSERT(b->type == GGML_TYPE_F32);
    GGML_ASSERT(a->ne[0] == b->ne[0]);
    GGML_ASSERT(dst->ne[0] == a->ne[1]);
    GGML_ASSERT(dst->ne[1] == b->ne[1]);
    GGML_ASSERT(dst->ne[2] == b->ne[2]);
    GGML_ASSERT(dst->ne[3] == b->ne[3]);

    const int64_t k_total = a->ne[0];
    const int64_t configured_k_block = std::max<int64_t>(1, cfg->k_block);
    const int64_t exp_block = (cfg->exp_mode == clip_aicas_bfp16m_exp_mode::per_channel ||
            cfg->exp_mode == clip_aicas_bfp16m_exp_mode::static_per_layer)
        ? k_total
        : configured_k_block;
    if (ith != 0) {
        return;
    }

    const int64_t n_kb = (k_total + exp_block - 1) / exp_block;
    const int64_t a_e_stride_row = n_kb;
    const int64_t a_e_stride_i2 = a->ne[1] * a_e_stride_row;
    const int64_t a_e_stride_i3 = a->ne[2] * a_e_stride_i2;
    const int64_t a_q_stride_row = k_total;
    const int64_t a_q_stride_i2 = a->ne[1] * a_q_stride_row;
    const int64_t a_q_stride_i3 = a->ne[2] * a_q_stride_i2;

    const int64_t b_e_stride_col = n_kb;
    const int64_t b_e_stride_i2 = b->ne[1] * b_e_stride_col;
    const int64_t b_e_stride_i3 = b->ne[2] * b_e_stride_i2;
    const int64_t b_q_stride_col = k_total;
    const int64_t b_q_stride_i2 = b->ne[1] * b_q_stride_col;
    const int64_t b_q_stride_i3 = b->ne[2] * b_q_stride_i2;

    std::vector<int8_t> a_exp((size_t) (a->ne[3] * a_e_stride_i3));
    std::vector<int16_t> a_q((size_t) (a->ne[3] * a_q_stride_i3));
    std::vector<int8_t> b_exp((size_t) (b->ne[3] * b_e_stride_i3));
    std::vector<int16_t> b_q((size_t) (b->ne[3] * b_q_stride_i3));

    for (int64_t i3 = 0; i3 < a->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < a->ne[2]; ++i2) {
            for (int64_t row = 0; row < a->ne[1]; ++row) {
                for (int64_t kb = 0; kb < n_kb; ++kb) {
                    const int64_t k0 = kb * exp_block;
                    const int64_t k1 = std::min(k_total, k0 + exp_block);
                    const int exp = clip_bfp16m_block_exp_for_a(a, cfg, row, i2, i3, k0, k1);
                    const int8_t exp_i8 = clip_bfp16m_exp_to_i8(exp);
                    a_exp[(size_t) (i3 * a_e_stride_i3 + i2 * a_e_stride_i2 + row * a_e_stride_row + kb)] = exp_i8;
                    for (int64_t k = k0; k < k1; ++k) {
                        a_q[(size_t) (i3 * a_q_stride_i3 + i2 * a_q_stride_i2 + row * a_q_stride_row + k)] =
                            clip_bfp16m_quant_value(clip_tensor_get_f32_4d(a, k, row, i2, i3), (int) exp_i8);
                    }
                }
            }
        }
    }

    for (int64_t i3 = 0; i3 < b->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < b->ne[2]; ++i2) {
            for (int64_t col = 0; col < b->ne[1]; ++col) {
                for (int64_t kb = 0; kb < n_kb; ++kb) {
                    const int64_t k0 = kb * exp_block;
                    const int64_t k1 = std::min(k_total, k0 + exp_block);
                    const int exp = clip_bfp16m_block_exp_for_b(b, cfg, col, i2, i3, k0, k1);
                    const int8_t exp_i8 = clip_bfp16m_exp_to_i8(exp);
                    b_exp[(size_t) (i3 * b_e_stride_i3 + i2 * b_e_stride_i2 + col * b_e_stride_col + kb)] = exp_i8;
                    for (int64_t k = k0; k < k1; ++k) {
                        b_q[(size_t) (i3 * b_q_stride_i3 + i2 * b_q_stride_i2 + col * b_q_stride_col + k)] =
                            clip_bfp16m_quant_value(clip_tensor_get_f32_4d(b, k, col, i2, i3), (int) exp_i8);
                    }
                }
            }
        }
    }

    const int64_t total = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
    for (int64_t index = 0; index < total; ++index) {
        int64_t rem = index;
        const int64_t row = rem % dst->ne[0];
        rem /= dst->ne[0];
        const int64_t col = rem % dst->ne[1];
        rem /= dst->ne[1];
        const int64_t i2 = rem % dst->ne[2];
        rem /= dst->ne[2];
        const int64_t i3 = rem;
        const int64_t a_i2 = i2 % a->ne[2];
        const int64_t a_i3 = i3 % a->ne[3];

        bool have_acc = false;
        int acc_exp = 0;
        int64_t acc = 0;
        for (int64_t kb = 0; kb < n_kb; ++kb) {
            const int64_t k0 = kb * exp_block;
            const int64_t k1 = std::min(k_total, k0 + exp_block);
            const int e_a = a_exp[(size_t) (a_i3 * a_e_stride_i3 + a_i2 * a_e_stride_i2 + row * a_e_stride_row + kb)];
            const int e_b = b_exp[(size_t) (i3 * b_e_stride_i3 + i2 * b_e_stride_i2 + col * b_e_stride_col + kb)];
            const int partial_exp = e_a + e_b;

            int64_t partial = 0;
            for (int64_t k = k0; k < k1; ++k) {
                const int16_t qa = a_q[(size_t) (a_i3 * a_q_stride_i3 + a_i2 * a_q_stride_i2 + row * a_q_stride_row + k)];
                const int16_t qb = b_q[(size_t) (i3 * b_q_stride_i3 + i2 * b_q_stride_i2 + col * b_q_stride_col + k)];
                partial += (int32_t) qa * (int32_t) qb;
            }

            if (!have_acc) {
                acc = partial;
                acc_exp = partial_exp;
                have_acc = true;
                continue;
            }

            if (partial_exp > acc_exp) {
                acc = clip_rshift_rne_i64(acc, partial_exp - acc_exp);
                acc_exp = partial_exp;
            }
            acc = clip_saturating_add_i64(acc, clip_rshift_rne_i64(partial, acc_exp - partial_exp));
        }

        clip_tensor_set_f32_4d(dst, row, col, i2, i3, (float) std::ldexp((double) acc, acc_exp));
    }
}

struct clip_graph {
    clip_ctx * ctx;
    const clip_model & model;
    const clip_hparams & hparams;

    // we only support single image per batch
    const clip_image_f32 & img;

    const int patch_size;
    const int n_patches_x;
    const int n_patches_y;
    const int n_patches;
    const int n_embd;
    const int n_head;
    const int d_head;
    const int n_layer;
    const float eps;
    const float kq_scale;

    ggml_context_ptr ctx0_ptr;
    ggml_context * ctx0;
    ggml_cgraph * gf;

    clip_graph(clip_ctx * ctx, const clip_image_f32 & img) :
            ctx(ctx),
            model(ctx->model),
            hparams(model.hparams),
            img(img),
            patch_size(hparams.patch_size),
            n_patches_x(img.nx / patch_size),
            n_patches_y(img.ny / patch_size),
            n_patches(n_patches_x * n_patches_y),
            n_embd(hparams.n_embd),
            n_head(hparams.n_head),
            d_head(n_embd / n_head),
            n_layer(hparams.n_layer),
            eps(hparams.eps),
            kq_scale(1.0f / sqrtf((float)d_head)) {
        struct ggml_init_params params = {
            /*.mem_size   =*/ ctx->buf_compute_meta.size(),
            /*.mem_buffer =*/ ctx->buf_compute_meta.data(),
            /*.no_alloc   =*/ true,
        };
        ctx0_ptr.reset(ggml_init(params));
        ctx0 = ctx0_ptr.get();
        gf = ggml_new_graph_custom(ctx0, ctx->max_nodes, false);
    }

    ggml_cgraph * build_siglip() {
        ggml_tensor * inp = build_inp();

        ggml_tensor * learned_pos_embd = model.position_embeddings;
        if (ctx->proj_type() == PROJECTOR_TYPE_LFM2) {
            learned_pos_embd = resize_position_embeddings();
        }

        ggml_tensor * cur = build_vit(
                                inp, n_patches,
                                NORM_TYPE_NORMAL,
                                hparams.ffn_op,
                                learned_pos_embd,
                                nullptr);

        if (ctx->proj_type() == PROJECTOR_TYPE_GEMMA3) {
            const int batch_size = 1;
            GGML_ASSERT(n_patches_x == n_patches_y);
            const int patches_per_image = n_patches_x;
            const int kernel_size = hparams.proj_scale_factor;

            cur = ggml_transpose(ctx0, cur);
            cur = ggml_cont_4d(ctx0, cur, patches_per_image, patches_per_image, n_embd, batch_size);

            // doing a pool2d to reduce the number of output tokens
            cur = ggml_pool_2d(ctx0, cur, GGML_OP_POOL_AVG, kernel_size, kernel_size, kernel_size, kernel_size, 0, 0);
            cur = ggml_reshape_3d(ctx0, cur, cur->ne[0] * cur->ne[0], n_embd, batch_size);
            cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

            // apply norm before projection
            cur = ggml_rms_norm(ctx0, cur, eps);
            cur = ggml_mul(ctx0, cur, model.mm_soft_emb_norm_w);

            // apply projection
            cur = ggml_mul_mat(ctx0,
                ggml_cont(ctx0, ggml_transpose(ctx0, model.mm_input_proj_w)),
                cur);

        } else if (ctx->proj_type() == PROJECTOR_TYPE_IDEFICS3) {
            // pixel_shuffle
            // https://github.com/huggingface/transformers/blob/0a950e0bbe1ed58d5401a6b547af19f15f0c195e/src/transformers/models/idefics3/modeling_idefics3.py#L578
            const int scale_factor = model.hparams.proj_scale_factor;
            cur = build_patch_merge_permute(cur, scale_factor);
            cur = build_mmproj_linear(model.projection, cur, "projector", -1);

        } else if (ctx->proj_type() == PROJECTOR_TYPE_LFM2) {
            // pixel unshuffle block
            const int scale_factor = model.hparams.proj_scale_factor;
            cur = build_patch_merge_permute(cur, scale_factor);

            // projection
            cur = ggml_norm(ctx0, cur, 1e-5); // default nn.LayerNorm
            cur = ggml_mul(ctx0, cur, model.mm_input_norm_w);
            cur = ggml_add(ctx0, cur, model.mm_input_norm_b);

            cur = ggml_mul_mat(ctx0, model.mm_1_w, cur);
            cur = ggml_add(ctx0, cur, model.mm_1_b);
            cur = ggml_gelu(ctx0, cur);
            cur = ggml_mul_mat(ctx0, model.mm_2_w, cur);
            cur = ggml_add(ctx0, cur, model.mm_2_b);
        } else {
            GGML_ABORT("SigLIP: Unsupported projector type");
        }

        // build the graph
        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    ggml_cgraph * build_pixtral() {
        const int n_merge = hparams.spatial_merge_size;

        // 2D input positions
        ggml_tensor * pos_h = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_patches);
        ggml_set_name(pos_h, "pos_h");
        ggml_set_input(pos_h);

        ggml_tensor * pos_w = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_patches);
        ggml_set_name(pos_w, "pos_w");
        ggml_set_input(pos_w);

        auto add_pos = [&](ggml_tensor * cur, const clip_layer &) {
            return build_rope_2d(ctx0, cur, pos_h, pos_w, hparams.rope_theta, true);
        };

        ggml_tensor * inp = build_inp();
        ggml_tensor * cur = build_vit(
                                inp, n_patches,
                                NORM_TYPE_RMS,
                                hparams.ffn_op,
                                nullptr, // no learned pos embd
                                add_pos);

        // mistral small 3.1 patch merger
        // ref: https://github.com/huggingface/transformers/blob/7a3e208892c06a5e278144eaf38c8599a42f53e7/src/transformers/models/mistral3/modeling_mistral3.py#L67
        if (model.mm_patch_merger_w) {
            GGML_ASSERT(hparams.spatial_merge_size > 0);

            cur = ggml_mul(ctx0, ggml_rms_norm(ctx0, cur, eps), model.mm_input_norm_w);

            // reshape image tokens to 2D grid
            cur = ggml_reshape_3d(ctx0, cur, n_embd, n_patches_x, n_patches_y);
            cur = ggml_permute(ctx0, cur, 2, 0, 1, 3); // [x, y, n_embd]
            cur = ggml_cont(ctx0, cur);

            // torch.nn.functional.unfold is just an im2col under the hood
            // we just need a dummy kernel to make it work
            ggml_tensor * kernel = ggml_view_3d(ctx0, cur, n_merge, n_merge, cur->ne[2], 0, 0, 0);
            cur = ggml_im2col(ctx0, kernel, cur, n_merge, n_merge, 0, 0, 1, 1, true, inp->type);

            // project to n_embd
            cur = ggml_reshape_2d(ctx0, cur, cur->ne[0], cur->ne[1] * cur->ne[2]);
            cur = ggml_mul_mat(ctx0, model.mm_patch_merger_w, cur);
        }

        // LlavaMultiModalProjector (always using GELU activation)
        {
            cur = ggml_mul_mat(ctx0, model.mm_1_w, cur);
            if (model.mm_1_b) {
                cur = ggml_add(ctx0, cur, model.mm_1_b);
            }

            cur = ggml_gelu(ctx0, cur);
            cur = ggml_mul_mat(ctx0, model.mm_2_w, cur);
            if (model.mm_2_b) {
                cur = ggml_add(ctx0, cur, model.mm_2_b);
            }
        }

        // arrangement of the [IMG_BREAK] token
        {
            // not efficient, but works
            // the trick is to view the embeddings as a 3D tensor with shape [n_embd, n_patches_per_row, n_rows]
            // and then concatenate the [IMG_BREAK] token to the end of each row, aka n_patches_per_row dimension
            // after the concatenation, we have a tensor with shape [n_embd, n_patches_per_row + 1, n_rows]

            const int p_y             = n_merge > 0 ? n_patches_y / n_merge : n_patches_y;
            const int p_x             = n_merge > 0 ? n_patches_x / n_merge : n_patches_x;
            const int p_total         = p_x * p_y;
            const int n_embd_text     = cur->ne[0];
            const int n_tokens_output = p_total + p_y - 1; // one [IMG_BREAK] per row, except the last row

            ggml_tensor * tmp = ggml_reshape_3d(ctx0, cur, n_embd_text, p_x, p_y);
            ggml_tensor * tok = ggml_new_tensor_3d(ctx0, tmp->type, n_embd_text, 1, p_y);
            tok = ggml_scale(ctx0, tok, 0.0); // clear the tensor
            tok = ggml_add(ctx0, tok, model.token_embd_img_break);
            tmp = ggml_concat(ctx0, tmp, tok, 1);
            cur = ggml_view_2d(ctx0, tmp,
                n_embd_text, n_tokens_output,
                ggml_row_size(tmp->type, n_embd_text), 0);
        }

        // build the graph
        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    // Qwen2VL and Qwen2.5VL use M-RoPE
    ggml_cgraph * build_qwen2vl() {
        GGML_ASSERT(model.patch_bias == nullptr);
        GGML_ASSERT(model.class_embedding == nullptr);

        const int batch_size       = 1;
        const bool use_window_attn = hparams.n_wa_pattern > 0;
        const int n_wa_pattern     = hparams.n_wa_pattern;
        const int n_pos            = n_patches;
        const int num_position_ids = n_pos * 4; // m-rope requires 4 dim per position

        norm_type norm_t = ctx->proj_type() == PROJECTOR_TYPE_QWEN25VL
            ? NORM_TYPE_RMS // qwen 2.5 vl
            : NORM_TYPE_NORMAL; // qwen 2 vl

        int mrope_sections[4] = {d_head/4, d_head/4, d_head/4, d_head/4};

        ggml_tensor * inp_raw = build_inp_raw();
        ggml_tensor * inp = ggml_conv_2d(ctx0, model.patch_embeddings_0, inp_raw, patch_size, patch_size, 0, 0, 1, 1);

        GGML_ASSERT(img.nx % (patch_size * 2) == 0);
        GGML_ASSERT(img.ny % (patch_size * 2) == 0);

        // second conv dimension
        {
            auto inp_1 = ggml_conv_2d(ctx0, model.patch_embeddings_1, inp_raw, patch_size, patch_size, 0, 0, 1, 1);
            inp = ggml_add(ctx0, inp, inp_1);

            inp = ggml_permute(ctx0, inp, 1, 2, 0, 3);  // [w, h, c, b] -> [c, w, h, b]
            inp = ggml_cont_4d(
                ctx0, inp,
                n_embd * 2, n_patches_x / 2, n_patches_y, batch_size);
            inp = ggml_reshape_4d(
                ctx0, inp,
                n_embd * 2, n_patches_x / 2, 2, batch_size * (n_patches_y / 2));
            inp = ggml_permute(ctx0, inp, 0, 2, 1, 3);
            inp = ggml_cont_3d(
                ctx0, inp,
                n_embd, n_patches_x * n_patches_y, batch_size);
        }

        ggml_tensor * inpL           = inp;
        ggml_tensor * window_mask    = nullptr;
        ggml_tensor * window_idx     = nullptr;
        ggml_tensor * inv_window_idx = nullptr;

        ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, num_position_ids);
        ggml_set_name(positions, "positions");
        ggml_set_input(positions);

        // pre-layernorm
        if (model.pre_ln_w) {
            inpL = build_norm(inpL, model.pre_ln_w, model.pre_ln_b, norm_t, eps, -1);
        }

        if (use_window_attn) {
            // handle window attention inputs
            inv_window_idx = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos / 4);
            ggml_set_name(inv_window_idx, "inv_window_idx");
            ggml_set_input(inv_window_idx);
            // mask for window attention
            window_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_pos, n_pos);
            ggml_set_name(window_mask, "window_mask");
            ggml_set_input(window_mask);

            // inpL shape: [n_embd, n_patches_x * n_patches_y, batch_size]
            GGML_ASSERT(batch_size == 1);
            inpL = ggml_reshape_2d(ctx0, inpL, n_embd * 4, n_patches_x * n_patches_y * batch_size / 4);
            inpL = ggml_get_rows(ctx0, inpL, inv_window_idx);
            inpL = ggml_reshape_3d(ctx0, inpL, n_embd, n_patches_x * n_patches_y, batch_size);
        }

        // loop over layers
        for (int il = 0; il < n_layer; il++) {
            auto & layer = model.layers[il];
            const bool full_attn = use_window_attn ? (il + 1) % n_wa_pattern == 0 : true;

            ggml_tensor * cur = inpL; // inpL = residual, cur = hidden_states

            // layernorm1
            cur = build_norm(cur, layer.ln_1_w, layer.ln_1_b, norm_t, eps, il);
            cb(cur, "ln1", il);

            // self-attention
            {
                ggml_tensor * Qcur = ggml_add(ctx0,
                    ggml_mul_mat(ctx0, layer.q_w, cur), layer.q_b);
                ggml_tensor * Kcur = ggml_add(ctx0,
                    ggml_mul_mat(ctx0, layer.k_w, cur), layer.k_b);
                ggml_tensor * Vcur = ggml_add(ctx0,
                    ggml_mul_mat(ctx0, layer.v_w, cur), layer.v_b);

                Qcur = ggml_reshape_3d(ctx0, Qcur, d_head, n_head, n_patches);
                Kcur = ggml_reshape_3d(ctx0, Kcur, d_head, n_head, n_patches);
                Vcur = ggml_reshape_3d(ctx0, Vcur, d_head, n_head, n_patches);

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                // apply M-RoPE
                Qcur = ggml_rope_multi(
                    ctx0, Qcur, positions, nullptr,
                    d_head/2, mrope_sections, GGML_ROPE_TYPE_VISION, 32768, 10000, 1, 0, 1, 32, 1);
                Kcur = ggml_rope_multi(
                    ctx0, Kcur, positions, nullptr,
                    d_head/2, mrope_sections, GGML_ROPE_TYPE_VISION, 32768, 10000, 1, 0, 1, 32, 1);

                cb(Qcur, "Qcur_rope", il);
                cb(Kcur, "Kcur_rope", il);

                ggml_tensor * attn_mask = full_attn ? nullptr : window_mask;

                cur = build_attn(layer.o_w, layer.o_b,
                    Qcur, Kcur, Vcur, attn_mask, kq_scale, il);
                cb(cur, "attn_out", il);
            }

            // re-add the layer input, e.g., residual
            cur = ggml_add(ctx0, cur, inpL);

            inpL = cur; // inpL = residual, cur = hidden_states

            cb(cur, "ffn_inp", il);

            // layernorm2
            cur = build_norm(cur, layer.ln_2_w, layer.ln_2_b, norm_t, eps, il);
            cb(cur, "ffn_inp_normed", il);

            // ffn
            cur = build_ffn(cur,
                layer.ff_up_w, layer.ff_up_b,
                layer.ff_gate_w, layer.ff_gate_b,
                layer.ff_down_w, layer.ff_down_b,
                hparams.ffn_op, il);

            cb(cur, "ffn_out", il);

            // residual 2
            cur = ggml_add(ctx0, inpL, cur);
            cb(cur, "layer_out", il);

            inpL = cur;
        }

        // post-layernorm
        if (model.post_ln_w) {
            inpL = build_norm(inpL, model.post_ln_w, model.post_ln_b, norm_t, eps, n_layer);
        }

        // multimodal projection
        ggml_tensor * embeddings = inpL;
        embeddings = ggml_reshape_3d(ctx0, embeddings, n_embd * 4, n_pos / 4, batch_size);

        embeddings = ggml_mul_mat(ctx0, model.mm_0_w, embeddings);
        embeddings = ggml_add(ctx0, embeddings, model.mm_0_b);

        // GELU activation
        embeddings = ggml_gelu(ctx0, embeddings);

        // Second linear layer
        embeddings = ggml_mul_mat(ctx0, model.mm_1_w, embeddings);
        embeddings = ggml_add(ctx0, embeddings, model.mm_1_b);

        if (use_window_attn) {
            window_idx = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos / 4);
            ggml_set_name(window_idx, "window_idx");
            ggml_set_input(window_idx);

            // embeddings shape: [n_embd, n_patches_x * n_patches_y, batch_size]
            GGML_ASSERT(batch_size == 1);
            embeddings = ggml_reshape_2d(ctx0, embeddings, hparams.projection_dim, n_patches_x * n_patches_y / 4);
            embeddings = ggml_get_rows(ctx0, embeddings, window_idx);
            embeddings = ggml_reshape_3d(ctx0, embeddings, hparams.projection_dim, n_patches_x * n_patches_y / 4, batch_size);
        }

        // build the graph
        ggml_build_forward_expand(gf, embeddings);

        return gf;
    }

    ggml_cgraph * build_minicpmv() {
        const int batch_size = 1;

        GGML_ASSERT(model.class_embedding == nullptr);
        const int n_pos = n_patches;

        // position embeddings for the projector (not for ViT)
        int n_output_dim = clip_n_mmproj_embd(ctx);
        ggml_tensor * pos_embed = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_output_dim, n_pos, batch_size);
        ggml_set_name(pos_embed, "pos_embed");
        ggml_set_input(pos_embed);

        // for selecting learned pos embd, used by ViT
        struct ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos);
        ggml_set_name(positions, "positions");
        ggml_set_input(positions);

        ggml_tensor * learned_pos_embd = ggml_get_rows(ctx0, model.position_embeddings, positions);

        ggml_tensor * inp = build_inp();
        ggml_tensor * embeddings = build_vit(
                                inp, n_patches,
                                NORM_TYPE_NORMAL,
                                hparams.ffn_op,
                                learned_pos_embd,
                                nullptr);

        // resampler projector (it is just another transformer)

        ggml_tensor * q = model.mm_model_query;
        ggml_tensor * v = ggml_mul_mat(ctx0, model.mm_model_kv_proj, embeddings);

        // norm
        q = build_norm(q, model.mm_model_ln_q_w, model.mm_model_ln_q_b, NORM_TYPE_NORMAL, eps, -1);
        v = build_norm(v, model.mm_model_ln_kv_w, model.mm_model_ln_kv_b, NORM_TYPE_NORMAL, eps, -1);

        // k = v + pos_embed
        ggml_tensor * k = ggml_add(ctx0, v, pos_embed);

        // attention
        {
            int n_embd = clip_n_mmproj_embd(ctx);
            const int d_head = 128;
            int n_head = n_embd/d_head;
            // Use actual config value if available, otherwise fall back to hardcoded values
            int num_query = ctx->model.hparams.minicpmv_query_num;
            ggml_tensor * Q = ggml_add(ctx0,
                ggml_mul_mat(ctx0, model.mm_model_attn_q_w, q),
                model.mm_model_attn_q_b);
            ggml_tensor * K = ggml_add(ctx0,
                ggml_mul_mat(ctx0, model.mm_model_attn_k_w, k),
                model.mm_model_attn_k_b);
            ggml_tensor * V = ggml_add(ctx0,
                ggml_mul_mat(ctx0, model.mm_model_attn_v_w, v),
                model.mm_model_attn_v_b);

            Q = ggml_reshape_3d(ctx0, Q, d_head, n_head, num_query);
            K = ggml_reshape_3d(ctx0, K, d_head, n_head, n_pos);
            V = ggml_reshape_3d(ctx0, V, d_head, n_head, n_pos);

            cb(Q, "resampler_Q", -1);
            cb(K, "resampler_K", -1);
            cb(V, "resampler_V", -1);

            embeddings = build_attn(
                model.mm_model_attn_o_w,
                model.mm_model_attn_o_b,
                Q, K, V, nullptr, kq_scale, -1);
            cb(embeddings, "resampler_attn_out", -1);
        }
        // layernorm
        embeddings = build_norm(embeddings, model.mm_model_ln_post_w, model.mm_model_ln_post_b, NORM_TYPE_NORMAL, eps, -1);

        // projection
        embeddings = ggml_mul_mat(ctx0, model.mm_model_proj, embeddings);

        // build the graph
        ggml_build_forward_expand(gf, embeddings);

        return gf;
    }

    ggml_cgraph * build_internvl() {
        GGML_ASSERT(model.class_embedding != nullptr);
        GGML_ASSERT(model.position_embeddings != nullptr);

        const int n_pos = n_patches + 1;
        ggml_tensor * inp = build_inp();

        // add CLS token
        inp = ggml_concat(ctx0, inp, model.class_embedding, 1);

        // The larger models use a different ViT, which uses RMS norm instead of layer norm
        // ref: https://github.com/ggml-org/llama.cpp/pull/13443#issuecomment-2869786188
        norm_type norm_t = (hparams.n_embd == 3200 && hparams.n_layer == 45)
            ? NORM_TYPE_RMS // 6B ViT (Used by InternVL 2.5/3 - 26B, 38B, 78B)
            : NORM_TYPE_NORMAL; // 300M ViT (Used by all smaller InternVL models)

        ggml_tensor * cur = build_vit(
                                inp, n_pos,
                                norm_t,
                                hparams.ffn_op,
                                model.position_embeddings,
                                nullptr);

        // remove CLS token
        cur = ggml_view_2d(ctx0, cur,
            n_embd, n_patches,
            ggml_row_size(cur->type, n_embd), 0);

        // pixel shuffle
        {
            const int scale_factor = model.hparams.proj_scale_factor;
            const int bsz    = 1; // batch size, always 1 for now since we don't support batching
            const int height = n_patches_y;
            const int width  = n_patches_x;
            GGML_ASSERT(scale_factor > 0);
            cur = ggml_reshape_4d(ctx0, cur, n_embd * scale_factor, height / scale_factor, width, bsz);
            cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
            cur = ggml_cont_4d(ctx0, cur,
                n_embd * scale_factor * scale_factor,
                height / scale_factor,
                width / scale_factor,
                bsz);
            cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
            // flatten to 2D
            cur = ggml_cont_2d(ctx0, cur,
                n_embd * scale_factor * scale_factor,
                cur->ne[1] * cur->ne[2]);
        }

        // projector (always using GELU activation)
        {
            // projector LayerNorm uses pytorch's default eps = 1e-5
            // ref: https://huggingface.co/OpenGVLab/InternVL3-8B-Instruct/blob/a34d3e4e129a5856abfd6aa6de79776484caa14e/modeling_internvl_chat.py#L79
            cur = build_norm(cur, model.mm_0_w, model.mm_0_b, NORM_TYPE_NORMAL, 1e-5, -1);
            cur = ggml_mul_mat(ctx0, model.mm_1_w, cur);
            cur = ggml_add(ctx0, cur, model.mm_1_b);
            cur = ggml_gelu(ctx0, cur);
            cur = ggml_mul_mat(ctx0, model.mm_3_w, cur);
            cur = ggml_add(ctx0, cur, model.mm_3_b);
        }

        // build the graph
        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    ggml_cgraph * build_llama4() {
        GGML_ASSERT(model.class_embedding != nullptr);
        GGML_ASSERT(model.position_embeddings != nullptr);

        const int n_pos = n_patches + 1; // +1 for [CLS]

        // 2D input positions
        ggml_tensor * pos_h = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos);
        ggml_set_name(pos_h, "pos_h");
        ggml_set_input(pos_h);

        ggml_tensor * pos_w = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos);
        ggml_set_name(pos_w, "pos_w");
        ggml_set_input(pos_w);

        ggml_tensor * inp = build_inp_raw();

        // Llama4UnfoldConvolution
        {
            ggml_tensor * kernel = ggml_reshape_4d(ctx0, model.patch_embeddings_0,
                                                    patch_size, patch_size, 3, n_embd);
            inp = ggml_im2col(ctx0, kernel, inp, patch_size, patch_size, 0, 0, 1, 1, true, inp->type);
            inp = ggml_mul_mat(ctx0, model.patch_embeddings_0, inp);
            inp = ggml_reshape_2d(ctx0, inp, n_embd, n_patches);
            cb(inp, "patch_conv", -1);
        }

        // add CLS token
        inp = ggml_concat(ctx0, inp, model.class_embedding, 1);

        // build ViT with 2D position embeddings
        auto add_pos = [&](ggml_tensor * cur, const clip_layer &) {
            // first half is X axis and second half is Y axis
            // ref: https://github.com/huggingface/transformers/blob/40a493c7ed4f19f08eadb0639cf26d49bfa5e180/src/transformers/models/llama4/modeling_llama4.py#L1312
            // ref: https://github.com/Blaizzy/mlx-vlm/blob/a57156aa87b33cca6e5ee6cfc14dd4ef8f611be6/mlx_vlm/models/llama4/vision.py#L441
            return build_rope_2d(ctx0, cur, pos_w, pos_h, hparams.rope_theta, false);
        };
        ggml_tensor * cur = build_vit(
                                inp, n_pos,
                                NORM_TYPE_NORMAL,
                                hparams.ffn_op,
                                model.position_embeddings,
                                add_pos);

        // remove CLS token
        cur = ggml_view_2d(ctx0, cur,
            n_embd, n_patches,
            ggml_row_size(cur->type, n_embd), 0);

        // pixel shuffle
        // based on Llama4VisionPixelShuffleMLP
        // https://github.com/huggingface/transformers/blob/2932f318a20d9e54cc7aea052e040164d85de7d6/src/transformers/models/llama4/modeling_llama4.py#L1151
        {
            const int scale_factor = model.hparams.proj_scale_factor;
            const int bsz = 1; // batch size, always 1 for now since we don't support batching
            GGML_ASSERT(scale_factor > 0);
            GGML_ASSERT(n_patches_x == n_patches_y); // llama4 only supports square images
            cur = ggml_reshape_4d(ctx0, cur,
                n_embd * scale_factor,
                n_patches_x / scale_factor,
                n_patches_y,
                bsz);
            cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
            cur = ggml_cont_4d(ctx0, cur,
                n_embd * scale_factor * scale_factor,
                n_patches_x / scale_factor,
                n_patches_y / scale_factor,
                bsz);
            //cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
            // flatten to 2D
            cur = ggml_cont_2d(ctx0, cur,
                n_embd * scale_factor * scale_factor,
                n_patches / scale_factor / scale_factor);
            cb(cur, "pixel_shuffle", -1);
        }

        // based on Llama4VisionMLP2 (always uses GELU activation, no bias)
        {
            cur = ggml_mul_mat(ctx0, model.mm_model_mlp_1_w, cur);
            cur = ggml_gelu(ctx0, cur);
            cur = ggml_mul_mat(ctx0, model.mm_model_mlp_2_w, cur);
            cur = ggml_gelu(ctx0, cur);
            cb(cur, "adapter_mlp", -1);
        }

        // Llama4MultiModalProjector
        cur = ggml_mul_mat(ctx0, model.mm_model_proj, cur);
        cb(cur, "projected", -1);

        // build the graph
        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    ggml_cgraph * build_kimivl() {
        // 2D input positions
        ggml_tensor * pos_h = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_patches);
        ggml_set_name(pos_h, "pos_h");
        ggml_set_input(pos_h);

        ggml_tensor * pos_w = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_patches);
        ggml_set_name(pos_w, "pos_w");
        ggml_set_input(pos_w);

        ggml_tensor * learned_pos_embd = resize_position_embeddings();

        // build ViT with 2D position embeddings
        auto add_pos = [&](ggml_tensor * cur, const clip_layer &) {
            // first half is X axis and second half is Y axis
            return build_rope_2d(ctx0, cur, pos_w, pos_h, hparams.rope_theta, false);
        };

        ggml_tensor * inp = build_inp();
        ggml_tensor * cur = build_vit(
                                inp, n_patches,
                                NORM_TYPE_NORMAL,
                                hparams.ffn_op,
                                learned_pos_embd,
                                add_pos);

        cb(cur, "vit_out", -1);

        {
            // patch_merger
            const int scale_factor = model.hparams.proj_scale_factor;
            cur = build_patch_merge_permute(cur, scale_factor);

            // projection norm
            int proj_inp_dim = cur->ne[0];
            cur = ggml_view_2d(ctx0, cur,
                n_embd, cur->ne[1] * scale_factor * scale_factor,
                ggml_row_size(cur->type, n_embd), 0);
            cur = ggml_norm(ctx0, cur, 1e-5); // default nn.LayerNorm
            cur = ggml_mul(ctx0, cur, model.mm_input_norm_w);
            cur = ggml_add(ctx0, cur, model.mm_input_norm_b);
            cur = ggml_view_2d(ctx0, cur,
                proj_inp_dim, cur->ne[1] / scale_factor / scale_factor,
                ggml_row_size(cur->type, proj_inp_dim), 0);
            cb(cur, "proj_inp_normed", -1);

            // projection mlp
            cur = ggml_mul_mat(ctx0, model.mm_1_w, cur);
            cur = ggml_add(ctx0, cur, model.mm_1_b);
            cur = ggml_gelu(ctx0, cur);
            cur = ggml_mul_mat(ctx0, model.mm_2_w, cur);
            cur = ggml_add(ctx0, cur, model.mm_2_b);
            cb(cur, "proj_out", -1);
        }

        // build the graph
        ggml_build_forward_expand(gf, cur);

        return gf;
    }

    // this graph is used by llava, granite and glm
    // due to having embedding_stack (used by granite), we cannot reuse build_vit
    ggml_cgraph * build_llava() {
        const int batch_size = 1;
        const int n_pos = n_patches + (model.class_embedding ? 1 : 0);

        GGML_ASSERT(n_patches_x == n_patches_y && "only square images supported");

        // Calculate the deepest feature layer based on hparams and projector type
        int max_feature_layer = n_layer;
        {
            // Get the index of the second to last layer; this is the default for models that have a llava projector
            int il_last = hparams.n_layer - 1;
            int deepest_feature_layer = -1;

            if (ctx->proj_type() == PROJECTOR_TYPE_MINICPMV || ctx->proj_type() == PROJECTOR_TYPE_GLM_EDGE) {
                il_last += 1;
            }

            // If we set explicit vision feature layers, only go up to the deepest one
            // NOTE: only used by granite-vision models for now
            for (const auto & feature_layer : hparams.vision_feature_layer) {
                if (feature_layer > deepest_feature_layer) {
                    deepest_feature_layer = feature_layer;
                }
            }
            max_feature_layer = deepest_feature_layer < 0 ? il_last : deepest_feature_layer;
        }

        ggml_tensor * inp = build_inp();

        // concat class_embeddings and patch_embeddings
        if (model.class_embedding) {
            inp = ggml_concat(ctx0, inp, model.class_embedding, 1);
        }

        ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos);
        ggml_set_name(positions, "positions");
        ggml_set_input(positions);

        inp = ggml_add(ctx0, inp, ggml_get_rows(ctx0, model.position_embeddings, positions));

        ggml_tensor * inpL = inp;

        // pre-layernorm
        if (model.pre_ln_w) {
            inpL = build_norm(inpL, model.pre_ln_w, model.pre_ln_b, NORM_TYPE_NORMAL, eps, -1);
            cb(inpL, "pre_ln", -1);
        }

        std::vector<ggml_tensor *> embedding_stack;
        const auto & vision_feature_layer = hparams.vision_feature_layer;

        // loop over layers
        for (int il = 0; il < max_feature_layer; il++) {
            auto & layer = model.layers[il];
            ggml_tensor * cur = inpL; // inpL = residual, cur = hidden_states

            // If this is an embedding feature layer, save the output.
            // NOTE: 0 index here refers to the input to the encoder.
            if (vision_feature_layer.find(il) != vision_feature_layer.end()) {
                embedding_stack.push_back(cur);
            }

            // layernorm1
            cur = build_norm(cur, layer.ln_1_w, layer.ln_1_b, NORM_TYPE_NORMAL, eps, il);
            cb(cur, "layer_inp_normed", il);

            // self-attention
            {
                ggml_tensor * Qcur = build_mmproj_linear(layer.q_w, cur, "attn_q", il);
                if (layer.q_b) {
                    Qcur = ggml_add(ctx0, Qcur, layer.q_b);
                }

                ggml_tensor * Kcur = build_mmproj_linear(layer.k_w, cur, "attn_k", il);
                if (layer.k_b) {
                    Kcur = ggml_add(ctx0, Kcur, layer.k_b);
                }

                ggml_tensor * Vcur = build_mmproj_linear(layer.v_w, cur, "attn_v", il);
                if (layer.v_b) {
                    Vcur = ggml_add(ctx0, Vcur, layer.v_b);
                }

                Qcur = ggml_reshape_3d(ctx0, Qcur, d_head, n_head, n_pos);
                Kcur = ggml_reshape_3d(ctx0, Kcur, d_head, n_head, n_pos);
                Vcur = ggml_reshape_3d(ctx0, Vcur, d_head, n_head, n_pos);

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                cur = build_attn(layer.o_w, layer.o_b,
                    Qcur, Kcur, Vcur, nullptr, kq_scale, il);
                cb(cur, "attn_out", il);
            }

            // re-add the layer input, e.g., residual
            cur = ggml_add(ctx0, cur, inpL);

            inpL = cur; // inpL = residual, cur = hidden_states

            cb(cur, "ffn_inp", il);

            // layernorm2
            cur = build_norm(cur, layer.ln_2_w, layer.ln_2_b, NORM_TYPE_NORMAL, eps, il);
            cb(cur, "ffn_inp_normed", il);

            // ffn
            cur = build_ffn(cur,
                layer.ff_up_w, layer.ff_up_b,
                layer.ff_gate_w, layer.ff_gate_b,
                layer.ff_down_w, layer.ff_down_b,
                hparams.ffn_op, il);

            cb(cur, "ffn_out", il);

            // residual 2
            cur = ggml_add(ctx0, inpL, cur);
            cb(cur, "layer_out", il);

            inpL = cur;
        }

        // post-layernorm
        if (model.post_ln_w) {
            inpL = build_norm(inpL, model.post_ln_w, model.post_ln_b, NORM_TYPE_NORMAL, eps, -1);
        }

        ggml_tensor * embeddings = inpL;

        // process vision feature layers (used by granite)
        {
            // final layer is a vision feature layer
            if (vision_feature_layer.find(max_feature_layer) != vision_feature_layer.end()) {
                embedding_stack.push_back(inpL);
            }

            // If feature layers are explicitly set, stack them (if we have multiple)
            if (!embedding_stack.empty()) {
                embeddings = embedding_stack[0];
                for (size_t i = 1; i < embedding_stack.size(); i++) {
                    embeddings = ggml_concat(ctx0, embeddings, embedding_stack[i], 0);
                }
            }
        }

        // llava projector (also used by granite)
        if (ctx->model.hparams.has_llava_projector) {
            embeddings = ggml_reshape_2d(ctx0, embeddings, embeddings->ne[0], embeddings->ne[1]);

            ggml_tensor * patches = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_patches);
            ggml_set_name(patches, "patches");
            ggml_set_input(patches);

            // shape [1, 576, 1024]
            // ne is whcn, ne = [1024, 576, 1, 1]
            embeddings = ggml_get_rows(ctx0, embeddings, patches);

            // print_tensor_info(embeddings, "embeddings");

            // llava projector
            if (ctx->proj_type() == PROJECTOR_TYPE_MLP) {
                embeddings = ggml_mul_mat(ctx0, model.mm_0_w, embeddings);
                embeddings = ggml_add(ctx0, embeddings, model.mm_0_b);

                embeddings = ggml_gelu(ctx0, embeddings);
                if (model.mm_2_w) {
                    embeddings = ggml_mul_mat(ctx0, model.mm_2_w, embeddings);
                    embeddings = ggml_add(ctx0, embeddings, model.mm_2_b);
                }
            }
            else if (ctx->proj_type() == PROJECTOR_TYPE_MLP_NORM) {
                embeddings = ggml_mul_mat(ctx0, model.mm_0_w, embeddings);
                embeddings = ggml_add(ctx0, embeddings, model.mm_0_b);
                // ggml_tensor_printf(embeddings, "mm_0_w",0,true,false);
                // First LayerNorm
                embeddings = ggml_norm(ctx0, embeddings, eps);
                embeddings = ggml_add(ctx0, ggml_mul(ctx0, embeddings, model.mm_1_w),
                                    model.mm_1_b);

                // GELU activation
                embeddings = ggml_gelu(ctx0, embeddings);

                // Second linear layer
                embeddings = ggml_mul_mat(ctx0, model.mm_3_w, embeddings);
                embeddings = ggml_add(ctx0, embeddings, model.mm_3_b);

                // Second LayerNorm
                embeddings = ggml_norm(ctx0, embeddings, eps);
                embeddings = ggml_add(ctx0, ggml_mul(ctx0, embeddings, model.mm_4_w),
                                    model.mm_4_b);
            }
            else if (ctx->proj_type() == PROJECTOR_TYPE_LDP) {
                // MobileVLM projector
                int n_patch = 24;
                ggml_tensor * mlp_1 = ggml_mul_mat(ctx0, model.mm_model_mlp_1_w, embeddings);
                mlp_1 = ggml_add(ctx0, mlp_1, model.mm_model_mlp_1_b);
                mlp_1 = ggml_gelu(ctx0, mlp_1);
                ggml_tensor * mlp_3 = ggml_mul_mat(ctx0, model.mm_model_mlp_3_w, mlp_1);
                mlp_3 = ggml_add(ctx0, mlp_3, model.mm_model_mlp_3_b);
                // mlp_3 shape = [1, 576, 2048], ne = [2048, 576, 1, 1]

                // block 1
                ggml_tensor * block_1 = nullptr;
                {
                    // transpose from [1, 576, 2048] --> [1, 2048, 576] --> [1, 2048, 24, 24]
                    mlp_3 = ggml_permute(ctx0, mlp_3, 1, 0, 2, 3);
                    mlp_3 = ggml_cont_4d(ctx0, mlp_3, n_patch, n_patch, mlp_3->ne[1], mlp_3->ne[2]);
                    // stride = 1, padding = 1, bias is nullptr
                    block_1 = ggml_conv_2d_dw(ctx0, model.mm_model_block_1_block_0_0_w, mlp_3, 1, 1, 1, 1, 1, 1);

                    // layer norm
                    // // block_1 shape = [1, 2048, 24, 24], ne = [24, 24, 2048, 1]
                    block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 1, 2, 0, 3));
                    // block_1 shape = [1, 24, 24, 2048], ne = [2048, 24, 24, 1]
                    block_1 = ggml_norm(ctx0, block_1, eps);
                    block_1 = ggml_add(ctx0, ggml_mul(ctx0, block_1, model.mm_model_block_1_block_0_1_w), model.mm_model_block_1_block_0_1_b);
                    block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 2, 0, 1, 3));

                    // block_1 shape = [1, 2048, 24, 24], ne = [24, 24, 2048, 1]
                    // hardswish
                    ggml_tensor * block_1_hw = ggml_hardswish(ctx0, block_1);

                    block_1 = ggml_pool_2d(ctx0, block_1_hw, GGML_OP_POOL_AVG, block_1_hw->ne[0], block_1_hw->ne[1], block_1_hw->ne[0], block_1_hw->ne[1], 0, 0);
                    // block_1 shape = [1, 2048, 1, 1], ne = [1, 1, 2048, 1]
                    // pointwise conv
                    block_1 = ggml_reshape_2d(ctx0, block_1, block_1->ne[0]*block_1->ne[1]*block_1->ne[2], block_1->ne[3]);
                    block_1 = ggml_mul_mat(ctx0, model.mm_model_block_1_block_1_fc1_w, block_1);
                    block_1 = ggml_add(ctx0, block_1, model.mm_model_block_1_block_1_fc1_b);
                    block_1 = ggml_relu(ctx0, block_1);
                    block_1 = ggml_mul_mat(ctx0, model.mm_model_block_1_block_1_fc2_w, block_1);
                    block_1 = ggml_add(ctx0, block_1, model.mm_model_block_1_block_1_fc2_b);
                    block_1 = ggml_hardsigmoid(ctx0, block_1);
                    // block_1_hw shape = [1, 2048, 24, 24], ne = [24, 24, 2048, 1], block_1 shape = [1, 2048], ne = [2048, 1, 1, 1]
                    block_1 = ggml_reshape_4d(ctx0, block_1, 1, 1, block_1->ne[0], block_1->ne[1]);
                    block_1 = ggml_mul(ctx0, block_1_hw, block_1);

                    int w = block_1->ne[0], h = block_1->ne[1];
                    block_1 = ggml_reshape_3d(ctx0, block_1, w*h, block_1->ne[2], block_1->ne[3]);
                    block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 1, 0, 2, 3));

                    // block_1 shape = [1, 24*24, 2048], ne = [24*24, 2048, 1]
                    block_1 = ggml_mul_mat(ctx0, model.mm_model_block_1_block_2_0_w, block_1);
                    block_1 = ggml_reshape_4d(ctx0, block_1, block_1->ne[0], w, h, block_1->ne[3]);

                    // block_1 shape = [1, 24, 24, 2048], ne = [2048, 24, 24, 1]
                    block_1 = ggml_norm(ctx0, block_1, eps);
                    block_1 = ggml_add(ctx0, ggml_mul(ctx0, block_1, model.mm_model_block_1_block_2_1_w), model.mm_model_block_1_block_2_1_b);
                    block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 2, 0, 1, 3));
                    // block1 shape = [1, 2048, 24, 24], ne = [24, 24, 2048, 1]
                    // residual
                    block_1 = ggml_add(ctx0, mlp_3, block_1);
                }

                // block_2
                {
                    // stride = 2
                    block_1 = ggml_conv_2d_dw(ctx0, model.mm_model_block_2_block_0_0_w, block_1, 2, 2, 1, 1, 1, 1);

                    // block_1 shape = [1, 2048, 12, 12], ne = [12, 12, 2048, 1]
                    // layer norm
                    block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 1, 2, 0, 3));
                    // block_1 shape = [1, 12, 12, 2048], ne = [2048, 12, 12, 1]
                    block_1 = ggml_norm(ctx0, block_1, eps);
                    block_1 = ggml_add(ctx0, ggml_mul(ctx0, block_1, model.mm_model_block_2_block_0_1_w), model.mm_model_block_2_block_0_1_b);
                    block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 2, 0, 1, 3));
                    // block_1 shape = [1, 2048, 12, 12], ne = [12, 12, 2048, 1]
                    // hardswish
                    ggml_tensor * block_1_hw = ggml_hardswish(ctx0, block_1);

                    // not sure the parameters is right for globalAvgPooling
                    block_1 = ggml_pool_2d(ctx0, block_1_hw, GGML_OP_POOL_AVG, block_1_hw->ne[0], block_1_hw->ne[1], block_1_hw->ne[0], block_1_hw->ne[1], 0, 0);
                    // block_1 shape = [1, 2048, 1, 1], ne = [1, 1, 2048, 1]
                    // pointwise conv
                    block_1 = ggml_reshape_2d(ctx0, block_1, block_1->ne[0]*block_1->ne[1]*block_1->ne[2], block_1->ne[3]);
                    block_1 = ggml_mul_mat(ctx0, model.mm_model_block_2_block_1_fc1_w, block_1);
                    block_1 = ggml_add(ctx0, block_1, model.mm_model_block_2_block_1_fc1_b);
                    block_1 = ggml_relu(ctx0, block_1);
                    block_1 = ggml_mul_mat(ctx0, model.mm_model_block_2_block_1_fc2_w, block_1);
                    block_1 = ggml_add(ctx0, block_1, model.mm_model_block_2_block_1_fc2_b);
                    block_1 = ggml_hardsigmoid(ctx0, block_1);

                    // block_1_hw shape = [1, 2048, 12, 12], ne = [12, 12, 2048, 1], block_1 shape = [1, 2048, 1, 1], ne = [1, 1, 2048, 1]
                    block_1 = ggml_reshape_4d(ctx0, block_1, 1, 1, block_1->ne[0], block_1->ne[1]);
                    block_1 = ggml_mul(ctx0, block_1_hw, block_1);

                    int w = block_1->ne[0], h = block_1->ne[1];
                    block_1 = ggml_reshape_3d(ctx0, block_1, w*h, block_1->ne[2], block_1->ne[3]);
                    block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 1, 0, 2, 3));
                    // block_1 shape = [1, 24*24, 2048], ne = [24*24, 2048, 1]
                    block_1 = ggml_mul_mat(ctx0, model.mm_model_block_2_block_2_0_w, block_1);
                    block_1 = ggml_reshape_4d(ctx0, block_1, block_1->ne[0], w, h, block_1->ne[3]);


                    // block_1 shape = [1, 12, 12, 2048], ne = [2048, 12, 12, 1]
                    block_1 = ggml_norm(ctx0, block_1, eps);
                    block_1 = ggml_add(ctx0, ggml_mul(ctx0, block_1, model.mm_model_block_2_block_2_1_w), model.mm_model_block_2_block_2_1_b);
                    block_1 = ggml_reshape_3d(ctx0, block_1, block_1->ne[0], block_1->ne[1] * block_1->ne[2], block_1->ne[3]);
                    // block_1 shape = [1, 144, 2048], ne = [2048, 144, 1]
                }
                embeddings = block_1;
            }
            else if (ctx->proj_type() == PROJECTOR_TYPE_LDPV2)
            {
                int n_patch = 24;
                ggml_tensor * mlp_0 = ggml_mul_mat(ctx0, model.mm_model_mlp_0_w, embeddings);
                mlp_0 = ggml_add(ctx0, mlp_0, model.mm_model_mlp_0_b);
                mlp_0 = ggml_gelu(ctx0, mlp_0);
                ggml_tensor * mlp_2 = ggml_mul_mat(ctx0, model.mm_model_mlp_2_w, mlp_0);
                mlp_2 = ggml_add(ctx0, mlp_2, model.mm_model_mlp_2_b);
                // mlp_2 ne = [2048, 576, 1, 1]
                // // AVG Pool Layer 2*2, strides = 2
                mlp_2 = ggml_permute(ctx0, mlp_2, 1, 0, 2, 3);
                // mlp_2 ne = [576, 2048, 1, 1]
                mlp_2 = ggml_cont_4d(ctx0, mlp_2, n_patch, n_patch, mlp_2->ne[1], mlp_2->ne[2]);
                // mlp_2 ne [24, 24, 2048, 1]
                mlp_2 = ggml_pool_2d(ctx0, mlp_2, GGML_OP_POOL_AVG, 2, 2, 2, 2, 0, 0);
                // weight ne = [3, 3, 2048, 1]
                ggml_tensor * peg_0 = ggml_conv_2d_dw(ctx0, model.mm_model_peg_0_w, mlp_2, 1, 1, 1, 1, 1, 1);
                peg_0 = ggml_cont(ctx0, ggml_permute(ctx0, peg_0, 1, 2, 0, 3));
                peg_0 = ggml_add(ctx0, peg_0, model.mm_model_peg_0_b);
                mlp_2 = ggml_cont(ctx0, ggml_permute(ctx0, mlp_2, 1, 2, 0, 3));
                peg_0 = ggml_add(ctx0, peg_0, mlp_2);
                peg_0 = ggml_reshape_3d(ctx0, peg_0, peg_0->ne[0], peg_0->ne[1] * peg_0->ne[2], peg_0->ne[3]);
                embeddings = peg_0;
            }
            else {
                GGML_ABORT("fatal error");
            }
        }

        // glm projector
        else if (ctx->proj_type() == PROJECTOR_TYPE_GLM_EDGE) {
            size_t gridsz = (size_t)sqrt(embeddings->ne[1]);
            embeddings = ggml_permute(ctx0,embeddings,1,0,2,3);
            embeddings = ggml_cont_3d(ctx0, embeddings, gridsz, gridsz, embeddings->ne[1]);
            embeddings = ggml_conv_2d(ctx0, model.mm_model_adapter_conv_w, embeddings, 2, 2, 0, 0, 1, 1);
            embeddings = ggml_reshape_3d(ctx0, embeddings,embeddings->ne[0]*embeddings->ne[1] , embeddings->ne[2], batch_size);
            embeddings = ggml_cont(ctx0, ggml_permute(ctx0,embeddings, 1, 0, 2, 3));
            embeddings = ggml_add(ctx0, embeddings, model.mm_model_adapter_conv_b);
            // GLU
            {
                embeddings = ggml_mul_mat(ctx0, model.mm_model_mlp_0_w, embeddings);
                embeddings = ggml_norm(ctx0, embeddings, eps);
                embeddings = ggml_add(ctx0, ggml_mul(ctx0, embeddings, model.mm_model_ln_q_w), model.mm_model_ln_q_b);
                embeddings = ggml_gelu_inplace(ctx0, embeddings);
                ggml_tensor * x = embeddings;
                embeddings = ggml_mul_mat(ctx0, model.mm_model_mlp_2_w, embeddings);
                x = ggml_mul_mat(ctx0, model.mm_model_mlp_1_w,x);
                embeddings = ggml_swiglu_split(ctx0, embeddings, x);
                embeddings = ggml_mul_mat(ctx0, model.mm_model_mlp_3_w, embeddings);
            }
            // arrangement of BOI/EOI token embeddings
            // note: these embeddings are not present in text model, hence we cannot process them as text tokens
            // see: https://huggingface.co/THUDM/glm-edge-v-2b/blob/main/siglip.py#L53
            {
                embeddings = ggml_concat(ctx0, model.mm_glm_tok_boi, embeddings, 1); // BOI
                embeddings = ggml_concat(ctx0, embeddings, model.mm_glm_tok_eoi, 1); // EOI
            }
        }

        else {
            GGML_ABORT("llava: unknown projector type");
        }

        // build the graph
        ggml_build_forward_expand(gf, embeddings);

        return gf;
    }

    // whisper encoder with custom projector
    ggml_cgraph * build_whisper_enc() {
        const int n_frames = img.nx;
        const int n_pos    = n_frames / 2;
        GGML_ASSERT(model.position_embeddings->ne[1] >= n_pos);

        ggml_tensor * inp = build_inp_raw(1);

        // conv1d block
        {
            // convolution + gelu
            ggml_tensor * cur = ggml_conv_1d_ph(ctx0, model.conv1d_1_w, inp, 1, 1);
            cur = ggml_add(ctx0, cur, model.conv1d_1_b);

            cur = ggml_gelu_erf(ctx0, cur);

            cur = ggml_conv_1d_ph(ctx0, model.conv1d_2_w, cur, 2, 1);
            cur = ggml_add(ctx0, cur, model.conv1d_2_b);

            cur = ggml_gelu_erf(ctx0, cur);
            // transpose
            inp = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
            cb(inp, "after_conv1d", -1);
        }

        // sanity check (only check one layer, but it should be the same for all)
        GGML_ASSERT(model.layers[0].ln_1_w && model.layers[0].ln_1_b);
        GGML_ASSERT(model.layers[0].ln_2_w && model.layers[0].ln_2_b);
        GGML_ASSERT(model.layers[0].q_b);
        GGML_ASSERT(model.layers[0].v_b);
        GGML_ASSERT(!model.layers[0].k_b); // no bias for k
        GGML_ASSERT(model.post_ln_w && model.post_ln_b);

        ggml_tensor * pos_embd_selected = ggml_view_2d(
            ctx0, model.position_embeddings,
            model.position_embeddings->ne[0], n_pos,
            model.position_embeddings->nb[1], 0
        );
        ggml_tensor * cur = build_vit(
                                inp, n_pos,
                                NORM_TYPE_NORMAL,
                                hparams.ffn_op,
                                pos_embd_selected,
                                nullptr);

        cb(cur, "after_transformer", -1);

        if (model.audio_has_stack_frames()) {
            // StackAudioFrames
            // https://huggingface.co/fixie-ai/ultravox-v0_5-llama-3_2-1b/blob/main/ultravox_model.py
            int64_t stride = n_embd * hparams.proj_stack_factor;
            int64_t padded_len = GGML_PAD(ggml_nelements(cur), stride);
            int64_t pad = padded_len - ggml_nelements(cur);
            if (pad > 0) {
                cur = ggml_view_1d(ctx0, cur, ggml_nelements(cur), 0);
                cur = ggml_pad(ctx0, cur, pad, 0, 0, 0);
            }
            cur = ggml_view_2d(ctx0, cur, stride, padded_len / stride,
                                ggml_row_size(cur->type, stride), 0);
            cb(cur, "after_stacked", -1);
        }

        if (ctx->proj_type() == PROJECTOR_TYPE_ULTRAVOX) {
            // UltravoxProjector
            // pre-norm
            cur = ggml_rms_norm(ctx0, cur, 1e-6);
            cur = ggml_mul(ctx0, cur, model.mm_norm_pre_w);

            // ffn in
            cur = ggml_mul_mat(ctx0, model.mm_1_w, cur);

            // swiglu
            // see SwiGLU in ultravox_model.py, the second half passed through is silu, not the first half
            cur = ggml_swiglu_swapped(ctx0, cur);

            // mid-norm
            cur = ggml_rms_norm(ctx0, cur, 1e-6);
            cur = ggml_mul(ctx0, cur, model.mm_norm_mid_w);

            // ffn out
            cur = ggml_mul_mat(ctx0, model.mm_2_w, cur);

        } else if (ctx->proj_type() == PROJECTOR_TYPE_QWEN2A) {
            // projector
            cur = ggml_mul_mat(ctx0, model.mm_fc_w, cur);
            cur = ggml_add(ctx0, cur, model.mm_fc_b);

        } else if (ctx->proj_type() == PROJECTOR_TYPE_VOXTRAL) {
            // projector
            cur = ggml_mul_mat(ctx0, model.mm_1_w, cur);
            cur = ggml_gelu_erf(ctx0, cur);
            cur = ggml_mul_mat(ctx0, model.mm_2_w, cur);

        } else {
            GGML_ABORT("%s: unknown projector type", __func__);
        }

        cb(cur, "projected", -1);

        ggml_build_forward_expand(gf, cur);

        return gf;
    }

private:
    //
    // utility functions
    //

    void cb(ggml_tensor * cur0, const char * name, int il) const {
        if (ctx->debug_graph) {
            ggml_tensor * cur = ggml_cpy(ctx0, cur0, ggml_dup_tensor(ctx0, cur0));
            std::string cur_name = il >= 0 ? std::string(name) + "_" + std::to_string(il) : name;
            ggml_set_name(cur, cur_name.c_str());
            ggml_set_output(cur);
            ggml_build_forward_expand(gf, cur);
            ctx->debug_print_tensors.push_back(cur);
        }
    }

    // siglip2 naflex
    ggml_tensor * resize_position_embeddings() {
        ggml_tensor * pos_embd = model.position_embeddings;
        const int height       = img.ny / patch_size;
        const int width        = img.nx / patch_size;
        const uint32_t mode    = GGML_SCALE_MODE_BILINEAR;
        const int n_per_side   = (int)std::sqrt(pos_embd->ne[1]);

        GGML_ASSERT(pos_embd);

        if (height == n_per_side && width == n_per_side) {
            return pos_embd;
        }

        pos_embd = ggml_reshape_3d(ctx0, pos_embd, n_embd, n_per_side, n_per_side);  // -> (n_embd, n_per_side, n_per_side)
        pos_embd = ggml_permute(ctx0, pos_embd, 2, 0, 1, 3);                         // -> (n_per_side, n_per_side, n_embd)
        pos_embd = ggml_interpolate(ctx0, pos_embd, width, height, n_embd, 1, mode); // -> (width, height, n_embd)
        pos_embd = ggml_permute(ctx0, pos_embd, 1, 2, 0, 3);                         // -> (n_embd, width, height)
        pos_embd = ggml_cont_2d(ctx0, pos_embd, n_embd, width * height);             // -> (n_embd, width * height)

        return pos_embd;
    }

    // build vision transformer (ViT) cgraph
    // this function should cover most of the models
    // if your model has specific features, you should probably duplicate this function
    ggml_tensor * build_vit(
                ggml_tensor * inp,
                int64_t n_pos,
                norm_type norm_t,
                ffn_op_type ffn_t,
                ggml_tensor * learned_pos_embd,
                std::function<ggml_tensor *(ggml_tensor *, const clip_layer &)> add_pos
            ) {
        if (learned_pos_embd) {
            inp = ggml_add(ctx0, inp, learned_pos_embd);
            cb(inp, "pos_embed", -1);
        }

        ggml_tensor * inpL = inp;

        // pre-layernorm
        if (model.pre_ln_w) {
            inpL = build_norm(inpL, model.pre_ln_w, model.pre_ln_b, norm_t, eps, -1);
            cb(inpL, "pre_ln", -1);
        }

        // loop over layers
        for (int il = 0; il < n_layer; il++) {
            auto & layer = model.layers[il];
            ggml_tensor * cur = inpL; // inpL = residual, cur = hidden_states

            // layernorm1
            cur = build_norm(cur, layer.ln_1_w, layer.ln_1_b, norm_t, eps, il);
            cb(cur, "layer_inp_normed", il);

            // self-attention
            {
                ggml_tensor * Qcur = build_mmproj_linear(layer.q_w, cur, "attn_q", il);
                if (layer.q_b) {
                    Qcur = ggml_add(ctx0, Qcur, layer.q_b);
                }

                ggml_tensor * Kcur = build_mmproj_linear(layer.k_w, cur, "attn_k", il);
                if (layer.k_b) {
                    Kcur = ggml_add(ctx0, Kcur, layer.k_b);
                }

                ggml_tensor * Vcur = build_mmproj_linear(layer.v_w, cur, "attn_v", il);
                if (layer.v_b) {
                    Vcur = ggml_add(ctx0, Vcur, layer.v_b);
                }

                if (layer.q_norm) {
                    Qcur = build_norm(Qcur, layer.q_norm, NULL, norm_t, eps, il);
                    cb(Qcur, "Qcur_norm", il);
                }

                if (layer.k_norm) {
                    Kcur = build_norm(Kcur, layer.k_norm, NULL, norm_t, eps, il);
                    cb(Kcur, "Kcur_norm", il);
                }

                Qcur = ggml_reshape_3d(ctx0, Qcur, d_head, n_head, n_pos);
                Kcur = ggml_reshape_3d(ctx0, Kcur, d_head, n_head, n_pos);
                Vcur = ggml_reshape_3d(ctx0, Vcur, d_head, n_head, n_pos);

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                if (add_pos) {
                    Qcur = add_pos(Qcur, layer);
                    Kcur = add_pos(Kcur, layer);
                    cb(Qcur, "Qcur_pos", il);
                    cb(Kcur, "Kcur_pos", il);
                }

                cur = build_attn(layer.o_w, layer.o_b,
                    Qcur, Kcur, Vcur, nullptr, kq_scale, il);
                cb(cur, "attn_out", il);
            }

            if (layer.ls_1_w) {
                cur = ggml_mul(ctx0, cur, layer.ls_1_w);
                cb(cur, "attn_out_scaled", il);
            }

            // re-add the layer input, e.g., residual
            cur = ggml_add(ctx0, cur, inpL);

            inpL = cur; // inpL = residual, cur = hidden_states

            cb(cur, "ffn_inp", il);

            // layernorm2
            cur = build_norm(cur, layer.ln_2_w, layer.ln_2_b, norm_t, eps, il);
            cb(cur, "ffn_inp_normed", il);

            // ffn
            cur = build_ffn(cur,
                layer.ff_up_w, layer.ff_up_b,
                layer.ff_gate_w, layer.ff_gate_b,
                layer.ff_down_w, layer.ff_down_b,
                ffn_t, il);

            cb(cur, "ffn_out", il);

            if (layer.ls_2_w) {
                cur = ggml_mul(ctx0, cur, layer.ls_2_w);
                cb(cur, "ffn_out_scaled", il);
            }

            // residual 2
            cur = ggml_add(ctx0, inpL, cur);
            cb(cur, "layer_out", il);

            inpL = cur;
        }

        if (ctx->model.audio_has_avgpool()) {
            ggml_tensor * cur = inpL;
            cur = ggml_transpose(ctx0, cur);
            cur = ggml_cont(ctx0, cur);
            cur = ggml_pool_1d(ctx0, cur, GGML_OP_POOL_AVG, 2, 2, 0);
            cur = ggml_transpose(ctx0, cur);
            cur = ggml_cont(ctx0, cur);
            inpL = cur;
        }

        // post-layernorm
        if (model.post_ln_w) {
            inpL = build_norm(inpL, model.post_ln_w, model.post_ln_b, norm_t, eps, -1);
        }
        return inpL;
    }

    // build the input after conv2d (inp_raw --> patches)
    // returns tensor with shape [n_embd, n_patches]
    ggml_tensor * build_inp() {
        ggml_tensor * inp_raw = build_inp_raw();
        ggml_tensor * inp = ggml_conv_2d(ctx0, model.patch_embeddings_0, inp_raw, patch_size, patch_size, 0, 0, 1, 1);
        inp = ggml_reshape_2d(ctx0, inp, n_patches, n_embd);
        inp = ggml_cont(ctx0, ggml_transpose(ctx0, inp));
        if (model.patch_bias) {
            inp = ggml_add(ctx0, inp, model.patch_bias);
            cb(inp, "patch_bias", -1);
        }
        return inp;
    }

    ggml_tensor * build_inp_raw(int channels = 3) {
        ggml_tensor * inp_raw = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, img.nx, img.ny, channels);
        ggml_set_name(inp_raw, "inp_raw");
        ggml_set_input(inp_raw);
        return inp_raw;
    }

    ggml_tensor * maybe_observe_mmproj_activation(ggml_tensor * act, const char * tensor_name) const {
        auto * observer = ctx->get_aicas_activation_observer(tensor_name);
        if (observer == nullptr) {
            return act;
        }

        return ggml_map_custom1(
            ctx0,
            act,
            clip_collect_activation_f32_passthrough,
            1,
            observer);
    }

    ggml_tensor * build_mmproj_linear(ggml_tensor * weight, ggml_tensor * act, const char * label, int il) const {
        GGML_ASSERT(weight != nullptr);
        GGML_ASSERT(act != nullptr);
        act = maybe_observe_mmproj_activation(act, weight->name);

        auto it = model.aicas_w8a8_tensors.find(weight->name);
        if (!model.aicas_w8a8_enabled || it == model.aicas_w8a8_tensors.end()) {
            return ggml_mul_mat(ctx0, weight, act);
        }

        const auto & cfg = it->second;
        if (!cfg.enabled || cfg.policy != "W8A8") {
            return ggml_mul_mat(ctx0, weight, act);
        }

        if (ctx->should_route_w8a8_to_npu(weight->name)) {
            if (!ggml_is_contiguous(act)) {
                act = ggml_cont(ctx0, act);
            }
            if (ctx->aicas_w8a8_debug) {
                LOG_DBG(
                    "%s: route %s to NPU ggml_mul_mat (layer=%d, smooth_enabled=%d, weight_scale_mode=%s, scale_len=%zu, sumw_len=%zu)\n",
                    __func__,
                    weight->name,
                    il,
                    cfg.smooth_enabled ? 1 : 0,
                    cfg.weight_scale_mode.c_str(),
                    cfg.weight_scale.size(),
                    cfg.sum_w.size());
            }
            ggml_tensor * out = ggml_mul_mat(ctx0, weight, act);
            ggml_set_name(out, weight->name);
            if (!ctx->debug_dump_w8a8_tensors_dir.empty() &&
                    ctx->debug_dump_w8a8_tensors.size() < ctx->debug_dump_w8a8_tensors_max) {
                ctx->debug_dump_w8a8_tensors.push_back(out);
            }
            return out;
        }

        const bool ok_shape =
            weight->type == GGML_TYPE_I8 &&
            act->type == GGML_TYPE_F32 &&
            cfg.act_scale > 0.0f &&
            weight->ne[2] == 1 && weight->ne[3] == 1 &&
            act->ne[2] == 1 && act->ne[3] == 1 &&
            act->ne[0] == weight->ne[0] &&
            cfg.has_valid_smooth_config(weight->ne[0]) &&
            cfg.weight_scale.size() == cfg.expected_weight_scale_len(weight->ne[1]) &&
            cfg.has_valid_compensation_config(weight->ne[1]);

        if (!ok_shape) {
            if (ctx->aicas_w8a8_debug) {
                LOG_WRN(
                    "%s: fallback to ggml_mul_mat for %s (layer=%d, weight=%s, wtype=%s, acttype=%s, wshape=[%" PRId64 ",%" PRId64 "], ashape=[%" PRId64 ",%" PRId64 "], act_quant_mode=%s, weight_scale_mode=%s, scale_len=%zu, sumw_len=%zu)\n",
                    __func__,
                    label,
                    il,
                    weight->name,
                    ggml_type_name(weight->type),
                    ggml_type_name(act->type),
                    weight->ne[0], weight->ne[1],
                    act->ne[0], act->ne[1],
                    cfg.act_quant_mode.c_str(),
                    cfg.weight_scale_mode.c_str(),
                    cfg.weight_scale.size(),
                    cfg.sum_w.size());
            }
            return ggml_mul_mat(ctx0, weight, act);
        }

        ggml_tensor * out_template = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, weight->ne[1], act->ne[1]);
        auto * kernel_userdata = ctx->get_aicas_w8a8_kernel_userdata(weight->name, &cfg);
        ggml_tensor * out = ggml_map_custom3(
            ctx0,
            out_template,
            act,
            weight,
            clip_compute_w8a8_mul_mat,
            GGML_N_TASKS_MAX,
            kernel_userdata);
        ggml_set_name(out, weight->name);
        if (!ctx->debug_dump_w8a8_tensors_dir.empty() &&
                ctx->debug_dump_w8a8_tensors.size() < ctx->debug_dump_w8a8_tensors_max) {
            ctx->debug_dump_w8a8_tensors.push_back(out);
        }

        if (ctx->aicas_w8a8_debug) {
            LOG_DBG("%s: using W8A8 for %s (layer=%d)\n", __func__, weight->name, il);
        }
        return out;
    }

    ggml_tensor * build_norm(
            ggml_tensor * cur,
            ggml_tensor * mw,
            ggml_tensor * mb,
            norm_type type,
            float norm_eps,
            int il) const {

        cur = type == NORM_TYPE_RMS
            ? ggml_rms_norm(ctx0, cur, norm_eps)
            : ggml_norm(ctx0, cur, norm_eps);

        if (mw || mb) {
            cb(cur, "norm", il);
        }

        if (mw) {
            cur = ggml_mul(ctx0, cur, mw);
            if (mb) {
                cb(cur, "norm_w", il);
            }
        }

        if (mb) {
            cur = ggml_add(ctx0, cur, mb);
        }

        return cur;
    }

    ggml_tensor * build_ffn(
            ggml_tensor * cur,
            ggml_tensor * up,
            ggml_tensor * up_b,
            ggml_tensor * gate,
            ggml_tensor * gate_b,
            ggml_tensor * down,
            ggml_tensor * down_b,
            ffn_op_type type_op,
            int il) const {

        ggml_tensor * tmp = up ? build_mmproj_linear(up, cur, "ffn_up", il) : cur;
        cb(tmp, "ffn_up", il);

        if (up_b) {
            tmp = ggml_add(ctx0, tmp, up_b);
            cb(tmp, "ffn_up_b", il);
        }

        if (gate) {
            cur = build_mmproj_linear(gate, cur, "ffn_gate", il);
            cb(cur, "ffn_gate", il);

            if (gate_b) {
                cur = ggml_add(ctx0, cur, gate_b);
                cb(cur, "ffn_gate_b", il);
            }
        } else {
            cur = tmp;
        }

        // we only support parallel ffn for now
        switch (type_op) {
            case FFN_SILU:
                if (gate) {
                    cur = ggml_swiglu_split(ctx0, cur, tmp);
                    cb(cur, "ffn_swiglu", il);
                } else {
                    cur = ggml_silu(ctx0, cur);
                    cb(cur, "ffn_silu", il);
                } break;
            case FFN_GELU:
                if (gate) {
                    cur = ggml_geglu_split(ctx0, cur, tmp);
                    cb(cur, "ffn_geglu", il);
                } else {
                    cur = ggml_gelu(ctx0, cur);
                    cb(cur, "ffn_gelu", il);
                } break;
            case FFN_GELU_ERF:
                if (gate) {
                    cur = ggml_geglu_erf_split(ctx0, cur, tmp);
                    cb(cur, "ffn_geglu_erf", il);
                } else {
                    cur = ggml_gelu_erf(ctx0, cur);
                    cb(cur, "ffn_gelu_erf", il);
                } break;
            case FFN_GELU_QUICK:
                if (gate) {
                    cur = ggml_geglu_quick_split(ctx0, cur, tmp);
                    cb(cur, "ffn_geglu_quick", il);
                } else {
                    cur = ggml_gelu_quick(ctx0, cur);
                    cb(cur, "ffn_gelu_quick", il);
                } break;
        }

        if (down) {
            cur = build_mmproj_linear(down, cur, "ffn_down", il);
        }

        if (down_b) {
            cb(cur, "ffn_down", il);
        }

        if (down_b) {
            cur = ggml_add(ctx0, cur, down_b);
        }

        return cur;
    }

    ggml_tensor * cast_mmproj_attn_tensor(ggml_tensor * tensor) const {
        ggml_type target_type = GGML_TYPE_F32;
        switch (ctx->aicas_mmproj_attn_precision) {
            case clip_mmproj_attn_precision::f32:
                return tensor;
            case clip_mmproj_attn_precision::f16:
                target_type = GGML_TYPE_F16;
                break;
            case clip_mmproj_attn_precision::bf16:
                target_type = GGML_TYPE_BF16;
                break;
            case clip_mmproj_attn_precision::bfp16m:
                return tensor;
        }

        if (tensor->type == target_type) {
            return tensor;
        }

        return ggml_cast(ctx0, tensor, target_type);
    }

    ggml_tensor * build_mmproj_attn_mul_mat(ggml_tensor * a, ggml_tensor * b) const {
        if (ctx->aicas_mmproj_attn_precision != clip_mmproj_attn_precision::bfp16m) {
            return ggml_mul_mat(ctx0, cast_mmproj_attn_tensor(a), cast_mmproj_attn_tensor(b));
        }

        GGML_ASSERT(a->type == GGML_TYPE_F32);
        GGML_ASSERT(b->type == GGML_TYPE_F32);
        GGML_ASSERT(a->ne[0] == b->ne[0]);
        GGML_ASSERT(b->ne[2] % a->ne[2] == 0);
        GGML_ASSERT(b->ne[3] % a->ne[3] == 0);

        ggml_tensor * out_template = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, a->ne[1], b->ne[1], b->ne[2], b->ne[3]);
        return ggml_map_custom3(
            ctx0,
            out_template,
            a,
            b,
            clip_bfp16m_mul_mat_f32,
            1,
            &ctx->aicas_bfp16m_userdata);
    }

    clip_aicas_bfp16m_userdata * make_static_mmproj_attn_bfp16m_userdata(
            int il,
            bool pv_matmul) const {
        auto node_cfg = std::make_unique<clip_aicas_bfp16m_userdata>(ctx->aicas_bfp16m_userdata);
        if (ctx->aicas_bfp16m_exp_mode == clip_aicas_bfp16m_exp_mode::static_per_layer && il >= 0 && il < 12) {
            static const int q_exp[12] = {-12, -11, -11, -11, -12, -11, -11, -11, -11, -11, -11, -11};
            static const int k_exp[12] = {-12, -11, -11, -11, -12, -11, -11, -11, -11, -11, -11, -11};
            static const int v_exp[12] = {-12, -13, -12, -12, -12, -12, -12, -12, -12, -12, -12, -12};
            static const int p_exp[12] = {-15, -15, -15, -15, -15, -15, -15, -15, -15, -15, -15, -15};
            node_cfg->static_exp = true;
            node_cfg->static_exp_a = pv_matmul ? v_exp[il] : k_exp[il];
            node_cfg->static_exp_b = pv_matmul ? p_exp[il] : q_exp[il];
        }
        clip_aicas_bfp16m_userdata * ptr = node_cfg.get();
        ctx->aicas_bfp16m_node_userdata.push_back(std::move(node_cfg));
        return ptr;
    }

    ggml_tensor * build_mmproj_attn_mul_mat(ggml_tensor * a, ggml_tensor * b, int il, bool pv_matmul) const {
        if (ctx->aicas_mmproj_attn_precision != clip_mmproj_attn_precision::bfp16m) {
            return ggml_mul_mat(ctx0, cast_mmproj_attn_tensor(a), cast_mmproj_attn_tensor(b));
        }

        GGML_ASSERT(a->type == GGML_TYPE_F32);
        GGML_ASSERT(b->type == GGML_TYPE_F32);
        GGML_ASSERT(a->ne[0] == b->ne[0]);
        GGML_ASSERT(b->ne[2] % a->ne[2] == 0);
        GGML_ASSERT(b->ne[3] % a->ne[3] == 0);

        ggml_tensor * out_template = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, a->ne[1], b->ne[1], b->ne[2], b->ne[3]);
        return ggml_map_custom3(
            ctx0,
            out_template,
            a,
            b,
            clip_bfp16m_mul_mat_f32,
            1,
            make_static_mmproj_attn_bfp16m_userdata(il, pv_matmul));
    }

    ggml_tensor * cast_mmproj_attn_tensor_f32(ggml_tensor * tensor) const {
        if (tensor->type == GGML_TYPE_F32) {
            return tensor;
        }
        return ggml_cast(ctx0, tensor, GGML_TYPE_F32);
    }

    bool should_cast_mmproj_attn_block() const {
        return ctx->aicas_mmproj_attn_precision != clip_mmproj_attn_precision::f32 &&
            ctx->aicas_mmproj_attn_precision_scope == clip_mmproj_attn_precision_scope::block;
    }

    ggml_tensor * build_attn(
            ggml_tensor * wo,
            ggml_tensor * wo_b,
            ggml_tensor * q_cur,
            ggml_tensor * k_cur,
            ggml_tensor * v_cur,
            ggml_tensor * kq_mask,
            float kq_scale,
            int il) const {
        // these nodes are added to the graph together so that they are not reordered
        // by doing so, the number of splits in the graph is reduced
        ggml_build_forward_expand(gf, q_cur);
        ggml_build_forward_expand(gf, k_cur);
        ggml_build_forward_expand(gf, v_cur);

        if (should_cast_mmproj_attn_block()) {
            q_cur = cast_mmproj_attn_tensor(q_cur);
            k_cur = cast_mmproj_attn_tensor(k_cur);
            v_cur = cast_mmproj_attn_tensor(v_cur);
        }

        ggml_tensor * q = ggml_permute(ctx0, q_cur, 0, 2, 1, 3);
        if (!ctx->aicas_act_stats_path.empty()) {
            q = ggml_cont(ctx0, q);
            q = maybe_observe_mmproj_activation(q, (il >= 0 ? string_format("mmproj.attn.%d.q_for_qk", il) : "mmproj.attn.resampler.q_for_qk").c_str());
        }
        //cb(q, "q", il);

        ggml_tensor * k = ggml_permute(ctx0, k_cur, 0, 2, 1, 3);
        if (!ctx->aicas_act_stats_path.empty()) {
            k = ggml_cont(ctx0, k);
            k = maybe_observe_mmproj_activation(k, (il >= 0 ? string_format("mmproj.attn.%d.k_for_qk", il) : "mmproj.attn.resampler.k_for_qk").c_str());
        }
        //cb(k, "k", il);

        ggml_tensor * v = ggml_permute(ctx0, v_cur, 1, 2, 0, 3);
        v = ggml_cont(ctx0, v);
        if (!ctx->aicas_act_stats_path.empty()) {
            v = maybe_observe_mmproj_activation(v, (il >= 0 ? string_format("mmproj.attn.%d.v_for_pv", il) : "mmproj.attn.resampler.v_for_pv").c_str());
        }
        //cb(k, "v", il);

        ggml_tensor * cur;

        // TODO @ngxson : support flash attention
        {
            const auto n_tokens = q->ne[1];
            const auto n_head   = q->ne[2];
            // const auto n_kv     = k->ne[1]; // for flash attention

            ggml_tensor * kq = build_mmproj_attn_mul_mat(k, q, il, false);
            // F32 may not needed for vision encoders?
            // ggml_mul_mat_set_prec(kq, GGML_PREC_F32);

            kq = ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale, 0.0f);
            if (!ctx->aicas_act_stats_path.empty()) {
                kq = maybe_observe_mmproj_activation(kq, (il >= 0 ? string_format("mmproj.attn.%d.p_for_pv", il) : "mmproj.attn.resampler.p_for_pv").c_str());
            }

            ggml_tensor * kqv = build_mmproj_attn_mul_mat(v, kq, il, true);
            cur = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
            cur = ggml_cont_2d(ctx0, cur, cur->ne[0]*n_head, n_tokens);
            if (should_cast_mmproj_attn_block()) {
                cur = cast_mmproj_attn_tensor_f32(cast_mmproj_attn_tensor(cur));
            }
        }

        cb(cur, "kqv_out", il);

        if (wo) {
            cur = build_mmproj_linear(wo, cur, "attn_out_proj", il);
        }

        if (wo_b) {
            cur = ggml_add(ctx0, cur, wo_b);
        }

        return cur;
    }

    // implementation of the 2D RoPE without adding a new op in ggml
    // this is not efficient (use double the memory), but works on all backends
    // TODO: there was a more efficient which relies on ggml_view and ggml_rope_ext_inplace, but the rope inplace does not work well with non-contiguous tensors ; we should fix that and revert back to the original implementation in https://github.com/ggml-org/llama.cpp/pull/13065
    static ggml_tensor * build_rope_2d(
        ggml_context * ctx0,
        ggml_tensor * cur,
        ggml_tensor * pos_a, // first half
        ggml_tensor * pos_b, // second half
        const float freq_base,
        const bool interleave_freq
    ) {
        const int64_t n_dim  = cur->ne[0];
        const int64_t n_head = cur->ne[1];
        const int64_t n_pos  = cur->ne[2];

        // for example, if we have cur tensor of shape (n_dim=8, n_head, n_pos)
        // we will have a list of 4 inv_freq: 1e-0, 1e-1, 1e-2, 1e-3
        // first half of cur will use 1e-0, 1e-2 (even)
        // second half of cur will use 1e-1, 1e-3 (odd)
        // the trick here is to rotate just half of n_dim, so inv_freq will automatically be even
        //  ^ don't ask me why, it's math! -2(2i) / n_dim == -2i / (n_dim/2)
        // then for the second half, we use freq_scale to shift the inv_freq
        //  ^ why? replace (2i) with (2i+1) in the above equation
        const float freq_scale_odd = interleave_freq
                                    ? std::pow(freq_base, (float)-2/n_dim)
                                    : 1.0;

        // first half
        ggml_tensor * first;
        {
            first = ggml_view_3d(ctx0, cur,
                n_dim/2, n_head, n_pos,
                ggml_row_size(cur->type, n_dim),
                ggml_row_size(cur->type, n_dim*n_head),
                0);
            first = ggml_rope_ext(
                ctx0,
                first,
                pos_a,      // positions
                nullptr,    // freq factors
                n_dim/2,    // n_dims
                0, 0, freq_base,
                1.0f, 0.0f, 1.0f, 0.0f, 0.0f
            );
        }

        // second half
        ggml_tensor * second;
        {
            second = ggml_view_3d(ctx0, cur,
                n_dim/2, n_head, n_pos,
                ggml_row_size(cur->type, n_dim),
                ggml_row_size(cur->type, n_dim*n_head),
                n_dim/2 * ggml_element_size(cur));
            second = ggml_rope_ext(
                ctx0,
                second,
                pos_b,      // positions
                nullptr,    // freq factors
                n_dim/2,    // n_dims
                0, 0, freq_base,
                freq_scale_odd,
                0.0f, 1.0f, 0.0f, 0.0f
            );
        }

        cur = ggml_concat(ctx0, first, second, 0);
        return cur;
    }

    // aka pixel_shuffle / pixel_unshuffle / patch_merger (Kimi-VL)
    // support dynamic resolution
    ggml_tensor * build_patch_merge_permute(ggml_tensor * cur, int scale_factor) {
        GGML_ASSERT(scale_factor > 1);

        const int n_embd = cur->ne[0];
        int width  = img.nx / patch_size;
        int height = img.ny / patch_size;

        // pad width and height to factor
        const int64_t pad_width  = CLIP_ALIGN(width,  scale_factor) - width;
        const int64_t pad_height = CLIP_ALIGN(height, scale_factor) - height;
        cur = ggml_reshape_3d(ctx0, cur, n_embd, width, height);
        if (pad_width || pad_height) {
            cur     = ggml_pad(ctx0, cur, 0, pad_width, pad_height, 0);
            width  += pad_width;
            height += pad_height;
        }

        // unshuffle h
        cur = ggml_reshape_3d(ctx0, cur, n_embd * scale_factor, width / scale_factor, height);
        cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);

        // unshuffle w
        cur = ggml_cont_3d(ctx0, cur, n_embd * scale_factor * scale_factor, height / scale_factor, width / scale_factor);
        cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);

        cur = ggml_cont_2d(ctx0, cur, cur->ne[0], cur->ne[1] * cur->ne[2]);
        cb(cur, "pixel_shuffle", -1);

        return cur;
    }

};

static ggml_cgraph * clip_image_build_graph(clip_ctx * ctx, const clip_image_f32_batch & imgs) {
    GGML_ASSERT(imgs.entries.size() == 1 && "n_batch > 1 is not supported");
    clip_graph graph(ctx, *imgs.entries[0]);

    ggml_cgraph * res;

    switch (ctx->proj_type()) {
        case PROJECTOR_TYPE_GEMMA3:
        case PROJECTOR_TYPE_IDEFICS3:
        case PROJECTOR_TYPE_LFM2:
            {
                res = graph.build_siglip();
            } break;
        case PROJECTOR_TYPE_PIXTRAL:
            {
                res = graph.build_pixtral();
            } break;
        case PROJECTOR_TYPE_QWEN2VL:
        case PROJECTOR_TYPE_QWEN25VL:
            {
                res = graph.build_qwen2vl();
            } break;
        case PROJECTOR_TYPE_MINICPMV:
            {
                res = graph.build_minicpmv();
            } break;
        case PROJECTOR_TYPE_INTERNVL:
            {
                res = graph.build_internvl();
            } break;
        case PROJECTOR_TYPE_LLAMA4:
            {
                res = graph.build_llama4();
            } break;
        case PROJECTOR_TYPE_ULTRAVOX:
        case PROJECTOR_TYPE_VOXTRAL:
        case PROJECTOR_TYPE_QWEN2A:
            {
                res = graph.build_whisper_enc();
            } break;
        case PROJECTOR_TYPE_KIMIVL:
            {
                res = graph.build_kimivl();
            } break;
        default:
            {
                res = graph.build_llava();
            } break;
    }
    return res;
}

struct clip_model_loader {
    ggml_context_ptr ctx_meta;
    gguf_context_ptr ctx_gguf;

    std::string fname;

    size_t model_size = 0; // in bytes

    bool has_vision = false;
    bool has_audio  = false;

    // TODO @ngxson : we should not pass clip_ctx here, it should be clip_model
    clip_model_loader(const char * fname) : fname(fname) {
        struct ggml_context * meta = nullptr;

        struct gguf_init_params params = {
            /*.no_alloc = */ true,
            /*.ctx      = */ &meta,
        };

        ctx_gguf = gguf_context_ptr(gguf_init_from_file(fname, params));
        if (!ctx_gguf.get()) {
            throw std::runtime_error(string_format("%s: failed to load CLIP model from %s. Does this file exist?\n", __func__, fname));
        }

        ctx_meta.reset(meta);

        const int n_tensors = gguf_get_n_tensors(ctx_gguf.get());

        // print gguf info
        {
            std::string name;
            get_string(KEY_NAME, name, false);
            std::string description;
            get_string(KEY_DESCRIPTION, description, false);
            LOG_INF("%s: model name:   %s\n",  __func__, name.c_str());
            LOG_INF("%s: description:  %s\n",  __func__, description.c_str());
            LOG_INF("%s: GGUF version: %d\n",  __func__, gguf_get_version(ctx_gguf.get()));
            LOG_INF("%s: alignment:    %zu\n", __func__, gguf_get_alignment(ctx_gguf.get()));
            LOG_INF("%s: n_tensors:    %d\n",  __func__, n_tensors);
            LOG_INF("%s: n_kv:         %d\n",  __func__, (int)gguf_get_n_kv(ctx_gguf.get()));
            LOG_INF("\n");
        }

        // modalities
        {
            get_bool(KEY_HAS_VISION_ENC, has_vision, false);
            get_bool(KEY_HAS_AUDIO_ENC,  has_audio,  false);

            if (has_vision) {
                LOG_INF("%s: has vision encoder\n", __func__);
            }
            if (has_audio) {
                LOG_INF("%s: has audio encoder\n", __func__);
            }
        }

        // tensors
        {
            for (int i = 0; i < n_tensors; ++i) {
                const char * name = gguf_get_tensor_name(ctx_gguf.get(), i);
                const size_t offset = gguf_get_tensor_offset(ctx_gguf.get(), i);
                enum ggml_type type = gguf_get_tensor_type(ctx_gguf.get(), i);
                ggml_tensor * cur = ggml_get_tensor(meta, name);
                size_t tensor_size = ggml_nbytes(cur);
                model_size += tensor_size;
                LOG_DBG("%s: tensor[%d]: n_dims = %d, name = %s, tensor_size=%zu, offset=%zu, shape:[%" PRIu64 ", %" PRIu64 ", %" PRIu64 ", %" PRIu64 "], type = %s\n",
                    __func__, i, ggml_n_dims(cur), cur->name, tensor_size, offset, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3], ggml_type_name(type));
            }
        }
    }

    void load_hparams(clip_model & model, clip_modality modality) {
        auto & hparams = model.hparams;
        std::string log_ffn_op; // for logging

        // sanity check
        if (modality == CLIP_MODALITY_VISION) {
            GGML_ASSERT(has_vision);
        } else if (modality == CLIP_MODALITY_AUDIO) {
            GGML_ASSERT(has_audio);
        }
        model.modality = modality;


        // projector type
        std::string proj_type;
        {
            // default key
            get_string(KEY_PROJ_TYPE, proj_type, false);

            // for models with mixed modalities
            if (proj_type.empty()) {
                if (modality == CLIP_MODALITY_VISION) {
                    get_string(KEY_VISION_PROJ_TYPE, proj_type, false);
                } else if (modality == CLIP_MODALITY_AUDIO) {
                    get_string(KEY_AUDIO_PROJ_TYPE, proj_type, false);
                } else {
                    GGML_ABORT("unknown modality");
                }
            }

            model.proj_type = clip_projector_type_from_string(proj_type);

            if (model.proj_type == PROJECTOR_TYPE_UNKNOWN) {
                throw std::runtime_error(string_format("%s: unknown projector type: %s\n", __func__, proj_type.c_str()));
            }

            // correct arch for multimodal models (legacy method)
            if (model.proj_type == PROJECTOR_TYPE_QWEN25O) {
                model.proj_type = modality == CLIP_MODALITY_VISION
                                    ? PROJECTOR_TYPE_QWEN25VL
                                    : PROJECTOR_TYPE_QWEN2A;
            }
        }

        const bool is_vision = model.modality == CLIP_MODALITY_VISION;
        const bool is_audio  = model.modality == CLIP_MODALITY_AUDIO;

        // other hparams
        {
            const char * prefix = is_vision ? "vision" : "audio";
            get_u32(string_format(KEY_N_EMBD,         prefix), hparams.n_embd);
            get_u32(string_format(KEY_N_HEAD,         prefix), hparams.n_head);
            get_u32(string_format(KEY_N_FF,           prefix), hparams.n_ff);
            get_u32(string_format(KEY_N_BLOCK,        prefix), hparams.n_layer);
            get_u32(string_format(KEY_PROJ_DIM,       prefix), hparams.projection_dim);
            get_f32(string_format(KEY_LAYER_NORM_EPS, prefix), hparams.eps);

            if (is_vision) {
                get_u32(KEY_IMAGE_SIZE, hparams.image_size);
                get_u32(KEY_PREPROC_IMAGE_SIZE, hparams.preproc_image_size, false);
                get_u32(KEY_PATCH_SIZE, hparams.patch_size);
                get_u32(KEY_IMAGE_CROP_RESOLUTION, hparams.image_crop_resolution, false);
                get_i32(KEY_MINICPMV_VERSION, hparams.minicpmv_version, false); // legacy
                get_u32(KEY_MINICPMV_QUERY_NUM, hparams.minicpmv_query_num, false);
                if (hparams.minicpmv_query_num == 0) {
                    // Fallback to hardcoded values for legacy models
                    if (hparams.minicpmv_version == 3) {
                        hparams.minicpmv_query_num = 64;
                    } else if (hparams.minicpmv_version == 4) {
                        hparams.minicpmv_query_num = 64;
                    } else if (hparams.minicpmv_version == 5) {
                        hparams.minicpmv_query_num = 64;
                    } else if (hparams.minicpmv_version == 6) {
                        hparams.minicpmv_query_num = 64;
                    } else {
                        hparams.minicpmv_query_num = 96;
                    }
                }
            } else if (is_audio) {
                get_u32(KEY_A_NUM_MEL_BINS, hparams.n_mel_bins);

            } else {
                GGML_ASSERT(false && "unknown modality");
            }

            // for pinpoints, we need to convert it into a list of resolution candidates
            {
                std::vector<int> pinpoints;
                get_arr_int(KEY_IMAGE_GRID_PINPOINTS, pinpoints, false);
                if (!pinpoints.empty()) {
                    for (size_t i = 0; i < pinpoints.size(); i += 2) {
                        hparams.image_res_candidates.push_back({
                            pinpoints[i],
                            pinpoints[i+1],
                        });
                    }
                }
            }

            // default warmup value
            hparams.warmup_image_size = hparams.image_size;

            hparams.has_llava_projector = model.proj_type == PROJECTOR_TYPE_MLP
                                       || model.proj_type == PROJECTOR_TYPE_MLP_NORM
                                       || model.proj_type == PROJECTOR_TYPE_LDP
                                       || model.proj_type == PROJECTOR_TYPE_LDPV2;

            {
                bool use_gelu = false;
                bool use_silu = false;
                get_bool(KEY_USE_GELU, use_gelu, false);
                get_bool(KEY_USE_SILU, use_silu, false);
                if (use_gelu && use_silu) {
                    throw std::runtime_error(string_format("%s: both use_gelu and use_silu are set to true\n", __func__));
                }
                if (use_gelu) {
                    hparams.ffn_op = FFN_GELU;
                    log_ffn_op = "gelu";
                } else if (use_silu) {
                    hparams.ffn_op = FFN_SILU;
                    log_ffn_op = "silu";
                } else {
                    hparams.ffn_op = FFN_GELU_QUICK;
                    log_ffn_op = "gelu_quick";
                }
            }

            {
                std::string mm_patch_merge_type;
                get_string(KEY_MM_PATCH_MERGE_TYPE, mm_patch_merge_type, false);
                if (mm_patch_merge_type == "spatial_unpad") {
                    hparams.mm_patch_merge_type = PATCH_MERGE_SPATIAL_UNPAD;
                }
            }

            if (is_vision) {
                int idx_mean = gguf_find_key(ctx_gguf.get(), KEY_IMAGE_MEAN);
                int idx_std  = gguf_find_key(ctx_gguf.get(), KEY_IMAGE_STD);
                GGML_ASSERT(idx_mean >= 0 && "image_mean not found");
                GGML_ASSERT(idx_std >= 0  && "image_std not found");
                const float * mean_data = (const float *) gguf_get_arr_data(ctx_gguf.get(), idx_mean);
                const float * std_data  = (const float *) gguf_get_arr_data(ctx_gguf.get(), idx_std);
                for (int i = 0; i < 3; ++i) {
                    hparams.image_mean[i] = mean_data[i];
                    hparams.image_std[i]  = std_data[i];
                }
            }

            // Load the vision feature layer indices if they are explicitly provided;
            // if multiple vision feature layers are present, the values will be concatenated
            // to form the final visual features.
            // NOTE: gguf conversions should standardize the values of the vision feature layer to
            // be non-negative, since we use -1 to mark values as unset here.
            std::vector<int> vision_feature_layer;
            get_arr_int(KEY_FEATURE_LAYER, vision_feature_layer, false);
            // convert std::vector to std::unordered_set
            for (auto & layer : vision_feature_layer) {
                hparams.vision_feature_layer.insert(layer);
            }

            // model-specific params
            switch (model.proj_type) {
                case PROJECTOR_TYPE_MINICPMV:
                    {
                        if (hparams.minicpmv_version == 0) {
                            hparams.minicpmv_version = 2; // default to 2 if not set
                        }
                    } break;
                case PROJECTOR_TYPE_IDEFICS3:
                case PROJECTOR_TYPE_LFM2:
                case PROJECTOR_TYPE_INTERNVL:
                    {
                        get_u32(KEY_PROJ_SCALE_FACTOR, hparams.proj_scale_factor, false);
                    } break;
                case PROJECTOR_TYPE_PIXTRAL:
                    {
                        hparams.rope_theta = 10000.0f;
                        hparams.warmup_image_size = hparams.patch_size * 8;
                        // Mistral Small 2506 needs 1024x1024 image size cap to prevent OOM
                        // ref: https://github.com/ggml-org/llama.cpp/issues/14310
                        hparams.image_size = 1024;
                        get_u32(KEY_SPATIAL_MERGE_SIZE, hparams.spatial_merge_size, false);
                    } break;
                case PROJECTOR_TYPE_KIMIVL:
                    {
                        hparams.rope_theta = 10000.0f;
                        hparams.warmup_image_size = hparams.patch_size * 8;
                        get_u32(KEY_PROJ_SCALE_FACTOR, hparams.proj_scale_factor, false);
                    } break;
                case PROJECTOR_TYPE_GEMMA3:
                    {
                        // default value (used by all model sizes in gemma 3 family)
                        // number of patches for each **side** is reduced by a factor of 4
                        hparams.proj_scale_factor = 4;
                        // test model (tinygemma3) has a different value, we optionally read it
                        get_u32(KEY_PROJ_SCALE_FACTOR, hparams.proj_scale_factor, false);
                    } break;
                case PROJECTOR_TYPE_QWEN2VL:
                    {
                        // max image size = sqrt(max_pixels) = 3584
                        // ref: https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/blob/main/preprocessor_config.json
                        // however, the model use unreasonable memory past 1024 size, we force it to 1024 otherwise it's unusable
                        // ref: https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/discussions/10
                        hparams.image_size = 1024;
                        hparams.warmup_image_size = hparams.patch_size * 8;
                    } break;
                case PROJECTOR_TYPE_QWEN25VL:
                    {
                        // max image size = sqrt(max_pixels)
                        // https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/blob/main/preprocessor_config.json
                        // however, the model use unreasonable memory past 1024 size, we force it to 1024 otherwise it's unusable
                        // ref: https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/discussions/10
                        hparams.image_size = 1024;
                        hparams.warmup_image_size = hparams.patch_size * 8;
                        get_u32(KEY_WIN_ATTN_PATTERN, hparams.n_wa_pattern);
                    } break;
                case PROJECTOR_TYPE_LLAMA4:
                    {
                        hparams.rope_theta = 10000.0f;
                        get_u32(KEY_PROJ_SCALE_FACTOR, hparams.proj_scale_factor);
                        set_llava_uhd_res_candidates(model, 3);
                    } break;
                case PROJECTOR_TYPE_ULTRAVOX:
                case PROJECTOR_TYPE_QWEN2A:
                case PROJECTOR_TYPE_VOXTRAL:
                    {
                        bool require_stack = model.proj_type == PROJECTOR_TYPE_ULTRAVOX ||
                                             model.proj_type == PROJECTOR_TYPE_VOXTRAL;
                        get_u32(KEY_A_PROJ_STACK_FACTOR, hparams.proj_stack_factor, require_stack);
                        if (hparams.n_mel_bins != 128) {
                            throw std::runtime_error(string_format("%s: only 128 mel bins are supported for ultravox\n", __func__));
                        }
                        hparams.ffn_op = FFN_GELU_ERF;
                        log_ffn_op = "gelu_erf"; // temporary solution for logging
                    } break;
                default:
                    break;
            }

            load_aicas_w8a8(model);

            LOG_INF("%s: projector:          %s\n", __func__, proj_type.c_str());
            LOG_INF("%s: n_embd:             %d\n", __func__, hparams.n_embd);
            LOG_INF("%s: n_head:             %d\n", __func__, hparams.n_head);
            LOG_INF("%s: n_ff:               %d\n", __func__, hparams.n_ff);
            LOG_INF("%s: n_layer:            %d\n", __func__, hparams.n_layer);
            LOG_INF("%s: ffn_op:             %s\n", __func__, log_ffn_op.c_str());
            LOG_INF("%s: projection_dim:     %d\n", __func__, hparams.projection_dim);
            if (is_vision) {
                LOG_INF("\n--- vision hparams ---\n");
                LOG_INF("%s: image_size:         %d\n", __func__, hparams.image_size);
                LOG_INF("%s: patch_size:         %d\n", __func__, hparams.patch_size);
                LOG_INF("%s: has_llava_proj:     %d\n", __func__, hparams.has_llava_projector);
                LOG_INF("%s: minicpmv_version:   %d\n", __func__, hparams.minicpmv_version);
                LOG_INF("%s: proj_scale_factor:  %d\n", __func__, hparams.proj_scale_factor);
                LOG_INF("%s: n_wa_pattern:       %d\n", __func__, hparams.n_wa_pattern);
            } else if (is_audio) {
                LOG_INF("\n--- audio hparams ---\n");
                LOG_INF("%s: n_mel_bins:         %d\n", __func__, hparams.n_mel_bins);
                LOG_INF("%s: proj_stack_factor:  %d\n", __func__, hparams.proj_stack_factor);
            }
            LOG_INF("\n");
            LOG_INF("%s: model size:         %.2f MiB\n", __func__, model_size / 1024.0 / 1024.0);
            LOG_INF("%s: metadata size:      %.2f MiB\n", __func__, ggml_get_mem_size(ctx_meta.get()) / 1024.0 / 1024.0);
        }
    }

    void load_tensors(clip_ctx & ctx_clip) {
        auto & model = ctx_clip.model;
        auto & hparams = model.hparams;
        std::map<std::string, size_t> tensor_offset;
        std::vector<ggml_tensor *> tensors_to_load;
        size_t extra_tensor_count = 0;

        for (const auto & kv : model.aicas_w8a8_tensors) {
            const auto & cfg = kv.second;
            if (cfg.enabled && cfg.policy == "W8A8" && cfg.smooth_enabled && !cfg.smooth_scale.empty()) {
                ++extra_tensor_count;
            }
        }

        // TODO @ngxson : support both audio and video in the future
        const char * prefix = model.modality == CLIP_MODALITY_AUDIO ? "a" : "v";

        // get offsets
        for (int64_t i = 0; i < gguf_get_n_tensors(ctx_gguf.get()); ++i) {
            const char * name = gguf_get_tensor_name(ctx_gguf.get(), i);
            tensor_offset[name] = gguf_get_data_offset(ctx_gguf.get()) + gguf_get_tensor_offset(ctx_gguf.get(), i);
        }

        // create data context
        struct ggml_init_params params = {
            /*.mem_size =*/ static_cast<size_t>(gguf_get_n_tensors(ctx_gguf.get()) + extra_tensor_count + 1) * ggml_tensor_overhead(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc =*/ true,
        };
        ctx_clip.ctx_data.reset(ggml_init(params));
        if (!ctx_clip.ctx_data) {
            throw std::runtime_error(string_format("%s: failed to init ggml context\n", __func__));
        }

        // helper function
        auto get_tensor = [&](const std::string & name, bool required = true) {
            ggml_tensor * cur = ggml_get_tensor(ctx_meta.get(), name.c_str());
            if (!cur && required) {
                throw std::runtime_error(string_format("%s: unable to find tensor %s\n", __func__, name.c_str()));
            }
            if (cur) {
                tensors_to_load.push_back(cur);
                // add tensors to context
                ggml_tensor * data_tensor = ggml_dup_tensor(ctx_clip.ctx_data.get(), cur);
                ggml_set_name(data_tensor, cur->name);
                cur = data_tensor;
            }
            return cur;
        };

        model.class_embedding = get_tensor(TN_CLASS_EMBD, false);

        model.pre_ln_w = get_tensor(string_format(TN_LN_PRE, prefix, "weight"), false);
        model.pre_ln_b = get_tensor(string_format(TN_LN_PRE, prefix, "bias"),   false);

        model.post_ln_w = get_tensor(string_format(TN_LN_POST, prefix, "weight"), false);
        model.post_ln_b = get_tensor(string_format(TN_LN_POST, prefix, "bias"),   false);

        model.patch_bias = get_tensor(TN_PATCH_BIAS, false);
        model.patch_embeddings_0 = get_tensor(TN_PATCH_EMBD,   false);
        model.patch_embeddings_1 = get_tensor(TN_PATCH_EMBD_1, false);

        model.position_embeddings = get_tensor(string_format(TN_POS_EMBD, prefix), false);

        // layers
        model.layers.resize(hparams.n_layer);
        for (int il = 0; il < hparams.n_layer; ++il) {
            auto & layer = model.layers[il];
            layer.k_w    = get_tensor(string_format(TN_ATTN_K,      prefix, il, "weight"));
            layer.q_w    = get_tensor(string_format(TN_ATTN_Q,      prefix, il, "weight"));
            layer.v_w    = get_tensor(string_format(TN_ATTN_V,      prefix, il, "weight"));
            layer.o_w    = get_tensor(string_format(TN_ATTN_OUTPUT, prefix, il, "weight"));
            layer.k_norm = get_tensor(string_format(TN_ATTN_K_NORM, prefix, il, "weight"), false);
            layer.q_norm = get_tensor(string_format(TN_ATTN_Q_NORM, prefix, il, "weight"), false);
            layer.ln_1_w = get_tensor(string_format(TN_LN_1,        prefix, il, "weight"), false);
            layer.ln_2_w = get_tensor(string_format(TN_LN_2,        prefix, il, "weight"), false);
            layer.ls_1_w = get_tensor(string_format(TN_LS_1,        prefix, il, "weight"), false); // no bias
            layer.ls_2_w = get_tensor(string_format(TN_LS_2,        prefix, il, "weight"), false); // no bias

            layer.k_b    = get_tensor(string_format(TN_ATTN_K,      prefix, il, "bias"), false);
            layer.q_b    = get_tensor(string_format(TN_ATTN_Q,      prefix, il, "bias"), false);
            layer.v_b    = get_tensor(string_format(TN_ATTN_V,      prefix, il, "bias"), false);
            layer.o_b    = get_tensor(string_format(TN_ATTN_OUTPUT, prefix, il, "bias"), false);
            layer.ln_1_b = get_tensor(string_format(TN_LN_1,        prefix, il, "bias"), false);
            layer.ln_2_b = get_tensor(string_format(TN_LN_2,        prefix, il, "bias"), false);

            // ffn
            layer.ff_up_w   = get_tensor(string_format(TN_FFN_UP,   prefix, il, "weight"));
            layer.ff_up_b   = get_tensor(string_format(TN_FFN_UP,   prefix, il, "bias"),   false);
            layer.ff_gate_w = get_tensor(string_format(TN_FFN_GATE, prefix, il, "weight"), false);
            layer.ff_gate_b = get_tensor(string_format(TN_FFN_GATE, prefix, il, "bias"),   false);
            layer.ff_down_w = get_tensor(string_format(TN_FFN_DOWN, prefix, il, "weight"));
            layer.ff_down_b = get_tensor(string_format(TN_FFN_DOWN, prefix, il, "bias"),   false);

            // some models already exported with legacy (incorrect) naming which is quite messy, let's fix it here
            // note: Qwen model converted from the old surgery script has n_ff = 0, so we cannot use n_ff to check!
            bool is_ffn_swapped = (
                    // only old models need this fix
                    model.proj_type == PROJECTOR_TYPE_MLP
                    || model.proj_type == PROJECTOR_TYPE_MLP_NORM
                    || model.proj_type == PROJECTOR_TYPE_LDP
                    || model.proj_type == PROJECTOR_TYPE_LDPV2
                    || model.proj_type == PROJECTOR_TYPE_QWEN2VL
                    || model.proj_type == PROJECTOR_TYPE_QWEN25VL
                    || model.proj_type == PROJECTOR_TYPE_GLM_EDGE
                    || model.proj_type == PROJECTOR_TYPE_GEMMA3
                    || model.proj_type == PROJECTOR_TYPE_IDEFICS3
                    || model.proj_type == PROJECTOR_TYPE_MINICPMV
                ) && layer.ff_up_w && layer.ff_down_w && layer.ff_down_w->ne[0] == hparams.n_embd;
            if (is_ffn_swapped) {
                // swap up and down weights
                ggml_tensor * tmp = layer.ff_up_w;
                layer.ff_up_w = layer.ff_down_w;
                layer.ff_down_w = tmp;
                // swap up and down biases
                tmp = layer.ff_up_b;
                layer.ff_up_b = layer.ff_down_b;
                layer.ff_down_b = tmp;
                if (il == 0) {
                    LOG_WRN("%s: ffn up/down are swapped\n", __func__);
                }
            }
        }

        switch (model.proj_type) {
            case PROJECTOR_TYPE_MLP:
            case PROJECTOR_TYPE_MLP_NORM:
                {
                    // LLaVA projection
                    model.mm_0_w = get_tensor(string_format(TN_LLAVA_PROJ, 0, "weight"), false);
                    model.mm_0_b = get_tensor(string_format(TN_LLAVA_PROJ, 0, "bias"), false);
                    // Yi-type llava
                    model.mm_1_w = get_tensor(string_format(TN_LLAVA_PROJ, 1, "weight"), false);
                    model.mm_1_b = get_tensor(string_format(TN_LLAVA_PROJ, 1, "bias"), false);
                    // missing in Yi-type llava
                    model.mm_2_w = get_tensor(string_format(TN_LLAVA_PROJ, 2, "weight"), false);
                    model.mm_2_b = get_tensor(string_format(TN_LLAVA_PROJ, 2, "bias"), false);
                    // Yi-type llava
                    model.mm_3_w = get_tensor(string_format(TN_LLAVA_PROJ, 3, "weight"), false);
                    model.mm_3_b = get_tensor(string_format(TN_LLAVA_PROJ, 3, "bias"), false);
                    model.mm_4_w = get_tensor(string_format(TN_LLAVA_PROJ, 4, "weight"), false);
                    model.mm_4_b = get_tensor(string_format(TN_LLAVA_PROJ, 4, "bias"), false);
                    if (model.mm_3_w) {
                        // TODO: this is a hack to support Yi-type llava
                        model.proj_type = PROJECTOR_TYPE_MLP_NORM;
                    }
                    model.image_newline = get_tensor(TN_IMAGE_NEWLINE, false);
                } break;
            case PROJECTOR_TYPE_LDP:
                {
                    // MobileVLM projection
                    model.mm_model_mlp_1_w = get_tensor(string_format(TN_MVLM_PROJ_MLP, 1, "weight"));
                    model.mm_model_mlp_1_b = get_tensor(string_format(TN_MVLM_PROJ_MLP, 1, "bias"));
                    model.mm_model_mlp_3_w = get_tensor(string_format(TN_MVLM_PROJ_MLP, 3, "weight"));
                    model.mm_model_mlp_3_b = get_tensor(string_format(TN_MVLM_PROJ_MLP, 3, "bias"));
                    model.mm_model_block_1_block_0_0_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 1, 0, "0.weight"));
                    model.mm_model_block_1_block_0_1_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 1, 0, "1.weight"));
                    model.mm_model_block_1_block_0_1_b = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 1, 0, "1.bias"));
                    model.mm_model_block_1_block_1_fc1_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 1, 1, "fc1.weight"));
                    model.mm_model_block_1_block_1_fc1_b = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 1, 1, "fc1.bias"));
                    model.mm_model_block_1_block_1_fc2_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 1, 1, "fc2.weight"));
                    model.mm_model_block_1_block_1_fc2_b = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 1, 1, "fc2.bias"));
                    model.mm_model_block_1_block_2_0_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 1, 2, "0.weight"));
                    model.mm_model_block_1_block_2_1_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 1, 2, "1.weight"));
                    model.mm_model_block_1_block_2_1_b = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 1, 2, "1.bias"));
                    model.mm_model_block_2_block_0_0_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 2, 0, "0.weight"));
                    model.mm_model_block_2_block_0_1_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 2, 0, "1.weight"));
                    model.mm_model_block_2_block_0_1_b = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 2, 0, "1.bias"));
                    model.mm_model_block_2_block_1_fc1_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 2, 1, "fc1.weight"));
                    model.mm_model_block_2_block_1_fc1_b = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 2, 1, "fc1.bias"));
                    model.mm_model_block_2_block_1_fc2_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 2, 1, "fc2.weight"));
                    model.mm_model_block_2_block_1_fc2_b = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 2, 1, "fc2.bias"));
                    model.mm_model_block_2_block_2_0_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 2, 2, "0.weight"));
                    model.mm_model_block_2_block_2_1_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 2, 2, "1.weight"));
                    model.mm_model_block_2_block_2_1_b = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 2, 2, "1.bias"));
                } break;
            case PROJECTOR_TYPE_LDPV2:
                {
                    // MobilVLM_V2 projection
                    model.mm_model_mlp_0_w = get_tensor(string_format(TN_MVLM_PROJ_MLP, 0, "weight"));
                    model.mm_model_mlp_0_b = get_tensor(string_format(TN_MVLM_PROJ_MLP, 0, "bias"));
                    model.mm_model_mlp_2_w = get_tensor(string_format(TN_MVLM_PROJ_MLP, 2, "weight"));
                    model.mm_model_mlp_2_b = get_tensor(string_format(TN_MVLM_PROJ_MLP, 2, "bias"));
                    model.mm_model_peg_0_w = get_tensor(string_format(TN_MVLM_PROJ_PEG, 0, "weight"));
                    model.mm_model_peg_0_b = get_tensor(string_format(TN_MVLM_PROJ_PEG, 0, "bias"));
                } break;
            case PROJECTOR_TYPE_MINICPMV:
                {
                    // model.mm_model_pos_embed = get_tensor(new_clip->ctx_data, TN_MINICPMV_POS_EMBD);
                    model.mm_model_pos_embed_k = get_tensor(TN_MINICPMV_POS_EMBD_K);
                    model.mm_model_query = get_tensor(TN_MINICPMV_QUERY);
                    model.mm_model_proj = get_tensor(TN_MINICPMV_PROJ);
                    model.mm_model_kv_proj = get_tensor(TN_MINICPMV_KV_PROJ);
                    model.mm_model_attn_q_w = get_tensor(string_format(TN_MINICPMV_ATTN, "q", "weight"));
                    model.mm_model_attn_k_w = get_tensor(string_format(TN_MINICPMV_ATTN, "k", "weight"));
                    model.mm_model_attn_v_w = get_tensor(string_format(TN_MINICPMV_ATTN, "v", "weight"));
                    model.mm_model_attn_q_b = get_tensor(string_format(TN_MINICPMV_ATTN, "q", "bias"));
                    model.mm_model_attn_k_b = get_tensor(string_format(TN_MINICPMV_ATTN, "k", "bias"));
                    model.mm_model_attn_v_b = get_tensor(string_format(TN_MINICPMV_ATTN, "v", "bias"));
                    model.mm_model_attn_o_w = get_tensor(string_format(TN_MINICPMV_ATTN, "out", "weight"));
                    model.mm_model_attn_o_b = get_tensor(string_format(TN_MINICPMV_ATTN, "out", "bias"));
                    model.mm_model_ln_q_w = get_tensor(string_format(TN_MINICPMV_LN, "q", "weight"));
                    model.mm_model_ln_q_b = get_tensor(string_format(TN_MINICPMV_LN, "q", "bias"));
                    model.mm_model_ln_kv_w = get_tensor(string_format(TN_MINICPMV_LN, "kv", "weight"));
                    model.mm_model_ln_kv_b = get_tensor(string_format(TN_MINICPMV_LN, "kv", "bias"));
                    model.mm_model_ln_post_w = get_tensor(string_format(TN_MINICPMV_LN, "post", "weight"));
                    model.mm_model_ln_post_b = get_tensor(string_format(TN_MINICPMV_LN, "post", "bias"));
                } break;
            case PROJECTOR_TYPE_GLM_EDGE:
                {
                    model.mm_model_adapter_conv_w = get_tensor(string_format(TN_GLM_ADAPER_CONV, "weight"));
                    model.mm_model_adapter_conv_b = get_tensor(string_format(TN_GLM_ADAPER_CONV, "bias"));
                    model.mm_model_mlp_0_w = get_tensor(string_format(TN_GLM_ADAPTER_LINEAR, "weight"));
                    model.mm_model_ln_q_w = get_tensor(string_format(TN_GLM_ADAPTER_NORM_1, "weight"));
                    model.mm_model_ln_q_b = get_tensor(string_format(TN_GLM_ADAPTER_NORM_1, "bias"));
                    model.mm_model_mlp_1_w = get_tensor(string_format(TN_GLM_ADAPTER_D_H_2_4H, "weight"));
                    model.mm_model_mlp_2_w = get_tensor(string_format(TN_GLM_ADAPTER_GATE, "weight"));
                    model.mm_model_mlp_3_w = get_tensor(string_format(TN_GLM_ADAPTER_D_4H_2_H, "weight"));
                    model.mm_glm_tok_boi = get_tensor(string_format(TN_TOK_GLM_BOI, "weight"));
                    model.mm_glm_tok_eoi = get_tensor(string_format(TN_TOK_GLM_EOI, "weight"));
                } break;
            case PROJECTOR_TYPE_QWEN2VL:
            case PROJECTOR_TYPE_QWEN25VL:
                {
                    model.mm_0_w = get_tensor(string_format(TN_LLAVA_PROJ, 0, "weight"));
                    model.mm_0_b = get_tensor(string_format(TN_LLAVA_PROJ, 0, "bias"));
                    model.mm_1_w = get_tensor(string_format(TN_LLAVA_PROJ, 2, "weight"));
                    model.mm_1_b = get_tensor(string_format(TN_LLAVA_PROJ, 2, "bias"));
                } break;
            case PROJECTOR_TYPE_GEMMA3:
                {
                    model.mm_input_proj_w = get_tensor(TN_MM_INP_PROJ);
                    model.mm_soft_emb_norm_w = get_tensor(TN_MM_SOFT_EMB_N);
                } break;
            case PROJECTOR_TYPE_IDEFICS3:
                {
                    model.projection = get_tensor(TN_MM_PROJECTOR);
                } break;
            case PROJECTOR_TYPE_LFM2:
            case PROJECTOR_TYPE_KIMIVL:
                {
                    model.mm_input_norm_w = get_tensor(TN_MM_INP_NORM);
                    model.mm_input_norm_b = get_tensor(TN_MM_INP_NORM_B);
                    model.mm_1_w = get_tensor(string_format(TN_LLAVA_PROJ, 1, "weight"));
                    model.mm_1_b = get_tensor(string_format(TN_LLAVA_PROJ, 1, "bias"));
                    model.mm_2_w = get_tensor(string_format(TN_LLAVA_PROJ, 2, "weight"));
                    model.mm_2_b = get_tensor(string_format(TN_LLAVA_PROJ, 2, "bias"));
                } break;
            case PROJECTOR_TYPE_PIXTRAL:
                {
                    model.mm_1_w = get_tensor(string_format(TN_LLAVA_PROJ, 1, "weight"));
                    model.mm_1_b = get_tensor(string_format(TN_LLAVA_PROJ, 1, "bias"), false);
                    model.mm_2_w = get_tensor(string_format(TN_LLAVA_PROJ, 2, "weight"));
                    model.mm_2_b = get_tensor(string_format(TN_LLAVA_PROJ, 2, "bias"), false);
                    // [IMG_BREAK] token embedding
                    model.token_embd_img_break = get_tensor(TN_TOK_IMG_BREAK);
                    // for mistral small 3.1
                    model.mm_input_norm_w   = get_tensor(TN_MM_INP_NORM,     false);
                    model.mm_patch_merger_w = get_tensor(TN_MM_PATCH_MERGER, false);
                } break;
            case PROJECTOR_TYPE_ULTRAVOX:
                {
                    model.conv1d_1_w = get_tensor(string_format(TN_CONV1D, 1, "weight"));
                    model.conv1d_1_b = get_tensor(string_format(TN_CONV1D, 1, "bias"));
                    model.conv1d_2_w = get_tensor(string_format(TN_CONV1D, 2, "weight"));
                    model.conv1d_2_b = get_tensor(string_format(TN_CONV1D, 2, "bias"));
                    model.mm_1_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 1, "weight"));
                    model.mm_2_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 2, "weight"));
                    model.mm_norm_pre_w = get_tensor(string_format(TN_MM_NORM_PRE, "weight"));
                    model.mm_norm_mid_w = get_tensor(string_format(TN_MM_NORM_MID, "weight"));
                } break;
            case PROJECTOR_TYPE_QWEN2A:
                {
                    model.conv1d_1_w = get_tensor(string_format(TN_CONV1D, 1, "weight"));
                    model.conv1d_1_b = get_tensor(string_format(TN_CONV1D, 1, "bias"));
                    model.conv1d_2_w = get_tensor(string_format(TN_CONV1D, 2, "weight"));
                    model.conv1d_2_b = get_tensor(string_format(TN_CONV1D, 2, "bias"));
                    model.mm_fc_w = get_tensor(string_format(TN_MM_AUDIO_FC, "weight"));
                    model.mm_fc_b = get_tensor(string_format(TN_MM_AUDIO_FC, "bias"));
                } break;
            case PROJECTOR_TYPE_VOXTRAL:
                {
                    model.conv1d_1_w = get_tensor(string_format(TN_CONV1D, 1, "weight"));
                    model.conv1d_1_b = get_tensor(string_format(TN_CONV1D, 1, "bias"));
                    model.conv1d_2_w = get_tensor(string_format(TN_CONV1D, 2, "weight"));
                    model.conv1d_2_b = get_tensor(string_format(TN_CONV1D, 2, "bias"));
                    model.mm_1_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 1, "weight"));
                    model.mm_2_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 2, "weight"));
                } break;
            case PROJECTOR_TYPE_INTERNVL:
                {
                    model.mm_0_w = get_tensor(string_format(TN_MVLM_PROJ_MLP, 0, "weight"));
                    model.mm_0_b = get_tensor(string_format(TN_MVLM_PROJ_MLP, 0, "bias"));
                    model.mm_1_w = get_tensor(string_format(TN_MVLM_PROJ_MLP, 1, "weight"));
                    model.mm_1_b = get_tensor(string_format(TN_MVLM_PROJ_MLP, 1, "bias"));
                    model.mm_3_w = get_tensor(string_format(TN_MVLM_PROJ_MLP, 3, "weight"));
                    model.mm_3_b = get_tensor(string_format(TN_MVLM_PROJ_MLP, 3, "bias"));
                } break;
            case PROJECTOR_TYPE_LLAMA4:
                {
                    model.mm_model_proj    = get_tensor(TN_MM_PROJECTOR);
                    model.mm_model_mlp_1_w = get_tensor(string_format(TN_MVLM_PROJ_MLP, 1, "weight"));
                    model.mm_model_mlp_2_w = get_tensor(string_format(TN_MVLM_PROJ_MLP, 2, "weight"));
                } break;
            default:
                GGML_ASSERT(false && "unknown projector type");
        }

        model.aicas_smooth_scale_tensors.clear();
        model.aicas_smooth_scale_tensors.reserve(extra_tensor_count);
        for (const auto & kv : model.aicas_w8a8_tensors) {
            const std::string & weight_name = kv.first;
            const auto & cfg = kv.second;
            if (!cfg.enabled || cfg.policy != "W8A8" || !cfg.smooth_enabled || cfg.smooth_scale.empty()) {
                continue;
            }

            ggml_tensor * smooth_scale = ggml_new_tensor_1d(
                ctx_clip.ctx_data.get(),
                GGML_TYPE_F32,
                static_cast<int64_t>(cfg.smooth_scale.size()));
            GGML_ASSERT(smooth_scale != nullptr);
            ggml_set_name(smooth_scale, clip_aicas_smooth_scale_tensor_name(weight_name).c_str());
            model.aicas_smooth_scale_tensors.emplace(weight_name, smooth_scale);
        }

        // load data
        {
            std::vector<uint8_t> read_buf;

            auto fin = std::ifstream(fname, std::ios::binary);
            if (!fin) {
                throw std::runtime_error(string_format("%s: failed to open %s\n", __func__, fname.c_str()));
            }

            // alloc memory and offload data
            ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(ctx_clip.backend);
            ctx_clip.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_clip.ctx_data.get(), buft));
            ggml_backend_buffer_set_usage(ctx_clip.buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            for (auto & t : tensors_to_load) {
                ggml_tensor * cur = ggml_get_tensor(ctx_clip.ctx_data.get(), t->name);
                const size_t offset = tensor_offset[t->name];
                fin.seekg(offset, std::ios::beg);
                if (!fin) {
                    throw std::runtime_error(string_format("%s: failed to seek for tensor %s\n", __func__, t->name));
                }
                size_t num_bytes = ggml_nbytes(cur);
                if (ggml_backend_buft_is_host(buft)) {
                    // for the CPU and Metal backend, we can read directly into the tensor
                    fin.read(reinterpret_cast<char *>(cur->data), num_bytes);
                } else {
                    // read into a temporary buffer first, then copy to device memory
                    read_buf.resize(num_bytes);
                    fin.read(reinterpret_cast<char *>(read_buf.data()), num_bytes);
                    ggml_backend_tensor_set(cur, read_buf.data(), 0, num_bytes);
                }
            }
            fin.close();

            LOG_DBG("%s: loaded %zu tensors from %s\n", __func__, tensors_to_load.size(), fname.c_str());
        }

        for (const auto & kv : model.aicas_smooth_scale_tensors) {
            const std::string & weight_name = kv.first;
            ggml_tensor * smooth_scale = kv.second;
            GGML_ASSERT(smooth_scale != nullptr);

            const auto cfg_it = model.aicas_w8a8_tensors.find(weight_name);
            GGML_ASSERT(cfg_it != model.aicas_w8a8_tensors.end());
            const auto & cfg = cfg_it->second;
            GGML_ASSERT(cfg.smooth_scale.size() == static_cast<size_t>(ggml_nelements(smooth_scale)));

            ggml_backend_tensor_set(
                smooth_scale,
                cfg.smooth_scale.data(),
                0,
                ggml_nbytes(smooth_scale));
        }

        if (model.aicas_w8a8_enabled && clip_should_prefuse_aicas_bias_compensation()) {
            int fused = 0;
            int disabled = 0;
            for (auto & kv : model.aicas_w8a8_tensors) {
                const std::string & weight_name = kv.first;
                auto & cfg = kv.second;

                if (!cfg.enabled || cfg.policy != "W8A8") {
                    continue;
                }
                if (cfg.sum_w.empty()) {
                    continue;
                }
                if (!(cfg.act_scale > 0.0f)) {
                    cfg.enabled = false;
                    cfg.policy = "F16_FALLBACK";
                    ++disabled;
                    LOG_WRN("%s: disable W8A8 for %s: invalid act_scale\n", __func__, weight_name.c_str());
                    continue;
                }

                const std::string suffix = ".weight";
                if (weight_name.size() <= suffix.size() ||
                    weight_name.compare(weight_name.size() - suffix.size(), suffix.size(), suffix) != 0) {
                    cfg.enabled = false;
                    cfg.policy = "F16_FALLBACK";
                    ++disabled;
                    LOG_WRN("%s: disable W8A8 for %s: cannot derive bias name\n", __func__, weight_name.c_str());
                    continue;
                }

                const std::string bias_name =
                    weight_name.substr(0, weight_name.size() - suffix.size()) + ".bias";
                ggml_tensor * bias = ggml_get_tensor(ctx_clip.ctx_data.get(), bias_name.c_str());
                std::string fuse_error;
                if (!clip_aicas_fuse_bias_compensation(bias, cfg, &fuse_error)) {
                    LOG_WRN("%s: keep W8A8 for %s without bias pre-fusion: %s\n",
                        __func__, weight_name.c_str(), fuse_error.c_str());
                    continue;
                }

                cfg.bias_compensated = true;
                ++fused;
            }

            LOG_INF("%s: fused AICAS W8A8 bias compensation for %d tensors (%d disabled)\n",
                __func__, fused, disabled);
        } else if (model.aicas_w8a8_enabled) {
            LOG_INF("%s: keep original AICAS W8A8 bias tensors for raw runtime compensation path\n", __func__);
        }
    }

    void alloc_compute_meta(clip_ctx & ctx_clip) {
        const auto & hparams = ctx_clip.model.hparams;
        ctx_clip.buf_compute_meta.resize(
                ctx_clip.max_nodes * ggml_tensor_overhead() +
                ggml_graph_overhead_custom(ctx_clip.max_nodes, false));

        // create a fake batch
        clip_image_f32_batch batch;
        clip_image_f32_ptr img(clip_image_f32_init());
        if (ctx_clip.model.modality == CLIP_MODALITY_VISION) {
            img->nx = hparams.warmup_image_size;
            img->ny = hparams.warmup_image_size;
        } else {
            img->nx = hparams.warmup_audio_size;
            img->ny = hparams.n_mel_bins;
        }
        batch.entries.push_back(std::move(img));

        ggml_cgraph * gf = clip_image_build_graph(&ctx_clip, batch);
        ggml_backend_sched_reserve(ctx_clip.sched.get(), gf);

        for (size_t i = 0; i < ctx_clip.backend_ptrs.size(); ++i) {
            ggml_backend_t backend = ctx_clip.backend_ptrs[i];
            ggml_backend_buffer_type_t buft = ctx_clip.backend_buft[i];
            size_t size = ggml_backend_sched_get_buffer_size(ctx_clip.sched.get(), backend);
            if (size > 1) {
                LOG_INF("%s: %10s compute buffer size = %8.2f MiB\n", __func__,
                        ggml_backend_buft_name(buft),
                        size / 1024.0 / 1024.0);
            }
        }
    }

    void get_bool(const std::string & key, bool & output, bool required = true) {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) throw std::runtime_error("Key not found: " + key);
            return;
        }
        output = gguf_get_val_bool(ctx_gguf.get(), i);
    }

    void get_i32(const std::string & key, int & output, bool required = true) {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) throw std::runtime_error("Key not found: " + key);
            return;
        }
        output = gguf_get_val_i32(ctx_gguf.get(), i);
    }

    void get_u32(const std::string & key, int & output, bool required = true) {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) throw std::runtime_error("Key not found: " + key);
            return;
        }
        output = gguf_get_val_u32(ctx_gguf.get(), i);
    }

    void get_f32(const std::string & key, float & output, bool required = true) {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) throw std::runtime_error("Key not found: " + key);
            return;
        }
        output = gguf_get_val_f32(ctx_gguf.get(), i);
    }

    void get_string(const std::string & key, std::string & output, bool required = true) {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) throw std::runtime_error("Key not found: " + key);
            return;
        }
        output = std::string(gguf_get_val_str(ctx_gguf.get(), i));
    }

    void get_arr_int(const std::string & key, std::vector<int> & output, bool required = true) {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) throw std::runtime_error("Key not found: " + key);
            return;
        }
        int n = gguf_get_arr_n(ctx_gguf.get(), i);
        output.resize(n);
        const int32_t * values = (const int32_t *)gguf_get_arr_data(ctx_gguf.get(), i);
        for (int i = 0; i < n; ++i) {
            output[i] = values[i];
        }
    }

    void get_arr_f32(const std::string & key, std::vector<float> & output, bool required = true) {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) throw std::runtime_error("Key not found: " + key);
            return;
        }
        const int n = gguf_get_arr_n(ctx_gguf.get(), i);
        output.resize(n);
        const float * values = (const float *) gguf_get_arr_data(ctx_gguf.get(), i);
        for (int j = 0; j < n; ++j) {
            output[j] = values[j];
        }
    }

    void get_arr_i32(const std::string & key, std::vector<int32_t> & output, bool required = true) {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) throw std::runtime_error("Key not found: " + key);
            return;
        }
        const int n = gguf_get_arr_n(ctx_gguf.get(), i);
        output.resize(n);
        const int32_t * values = (const int32_t *) gguf_get_arr_data(ctx_gguf.get(), i);
        for (int j = 0; j < n; ++j) {
            output[j] = values[j];
        }
    }

    void load_aicas_w8a8(clip_model & model) {
        std::string schema;
        get_string("aicas.w8a8.schema", schema, false);
        if (schema.empty()) {
            return;
        }

        int tensor_count = 0;
        get_i32("aicas.w8a8.tensor_count", tensor_count);
        if (tensor_count < 0) {
            throw std::runtime_error("Invalid aicas.w8a8.tensor_count");
        }

        model.aicas_w8a8_enabled = true;
        model.aicas_w8a8_schema = schema;
        model.aicas_w8a8_tensors.clear();
        model.aicas_w8a8_tensors.reserve((size_t) tensor_count);

        for (int i = 0; i < tensor_count; ++i) {
            const std::string prefix = "aicas.w8a8.tensor." + std::to_string(i) + ".";

            std::string tensor_name;
            get_string(prefix + "name", tensor_name);
            if (tensor_name.empty()) {
                throw std::runtime_error("Empty tensor name in AICAS W8A8 metadata");
            }

            clip_aicas_w8a8_tensor cfg;
            get_bool(prefix + "enabled", cfg.enabled, false);
            get_string(prefix + "policy", cfg.policy, false);
            if (cfg.policy.empty()) {
                cfg.policy = "F16_FALLBACK";
            }

            if (cfg.enabled && cfg.policy == "W8A8") {
                get_f32(prefix + "act_scale", cfg.act_scale);
                get_i32(prefix + "act_scale_q8_24", cfg.act_scale_q8_24, false);
                if (cfg.act_scale_q8_24 == 0 && cfg.act_scale != 0.0f) {
                    cfg.act_scale_q8_24 = clip_fp32_bits_to_q8_24(clip_f32_to_bits(cfg.act_scale));
                }
                int act_zero_point = 0;
                get_i32(prefix + "act_zero_point", act_zero_point);
                cfg.act_zero_point = act_zero_point;
                get_string(prefix + "act_quant_mode", cfg.act_quant_mode, false);
                if (cfg.act_quant_mode.empty()) {
                    cfg.act_quant_mode = "asymmetric_u8";
                }
                if (cfg.act_quant_mode != "asymmetric_u8" && cfg.act_quant_mode != "symmetric_u8") {
                    throw std::runtime_error("Unsupported aicas.w8a8 act_quant_mode: " + cfg.act_quant_mode);
                }
                get_string(prefix + "weight_scale_mode", cfg.weight_scale_mode, false);
                if (cfg.weight_scale_mode.empty()) {
                    cfg.weight_scale_mode = "per_channel";
                }
                if (cfg.weight_scale_mode != "per_channel" && cfg.weight_scale_mode != "per_tensor") {
                    throw std::runtime_error("Unsupported aicas.w8a8 weight_scale_mode: " + cfg.weight_scale_mode);
                }
                get_arr_f32(prefix + "weight_scale", cfg.weight_scale);
                get_arr_i32(prefix + "sum_w", cfg.sum_w, false);
                get_bool(prefix + "smooth_enabled", cfg.smooth_enabled, false);
                get_f32(prefix + "smooth_alpha", cfg.smooth_alpha, false);
                get_f32(prefix + "smooth_eps", cfg.smooth_eps, false);
                get_arr_f32(prefix + "smooth_scale", cfg.smooth_scale, false);

                if (cfg.uses_symmetric_u8() && cfg.act_zero_point != 128) {
                    throw std::runtime_error("symmetric_u8 AICAS W8A8 metadata requires act_zero_point=128");
                }
                if (cfg.smooth_enabled && cfg.smooth_scale.empty()) {
                    throw std::runtime_error("AICAS W8A8 smoothquant metadata requires smooth_scale when smooth_enabled=true");
                }
            }

            model.aicas_w8a8_tensors[tensor_name] = std::move(cfg);
        }

        LOG_INF("%s: AICAS W8A8 enabled, schema=%s, tensors=%d\n",
                __func__, model.aicas_w8a8_schema.c_str(), tensor_count);
    }

    void set_llava_uhd_res_candidates(clip_model & model, const int max_patches_per_side) {
        auto & hparams = model.hparams;
        for (int x = 1; x <= max_patches_per_side; x++) {
            for (int y = 1; y <= max_patches_per_side; y++) {
                if (x == 1 && y == 1) {
                    continue; // skip the first point
                }
                hparams.image_res_candidates.push_back(clip_image_size{
                    x*hparams.image_size,
                    y*hparams.image_size,
                });
            }
        }
    }
};

struct clip_init_result clip_init(const char * fname, struct clip_context_params ctx_params) {
    g_logger_state.verbosity_thold = ctx_params.verbosity;
    clip_ctx * ctx_vision = nullptr;
    clip_ctx * ctx_audio = nullptr;

    try {
        clip_model_loader loader(fname);

        if (loader.has_vision) {
            ctx_vision = new clip_ctx(ctx_params);
            loader.load_hparams(ctx_vision->model, CLIP_MODALITY_VISION);
            if (ctx_vision->model.aicas_w8a8_enabled) {
                const bool npu_compatible = ctx_vision->register_aicas_w8a8_for_npu();
                if (!ctx_vision->backend_is_npu() || !npu_compatible) {
                    ctx_vision->force_cpu_backend_for_w8a8();
                }
            }
            loader.load_tensors(*ctx_vision);
            if (ctx_vision->backend_is_npu() &&
                ctx_vision->model.aicas_w8a8_enabled &&
                ctx_vision->should_preload_aicas_w8a8_for_npu()) {
                ctx_vision->preload_aicas_w8a8_for_npu();
            }
            loader.alloc_compute_meta(*ctx_vision);
        }

        if (loader.has_audio) {
            ctx_audio = new clip_ctx(ctx_params);
            loader.load_hparams(ctx_audio->model, CLIP_MODALITY_AUDIO);
            if (ctx_audio->model.aicas_w8a8_enabled) {
                const bool npu_compatible = ctx_audio->register_aicas_w8a8_for_npu();
                if (!ctx_audio->backend_is_npu() || !npu_compatible) {
                    ctx_audio->force_cpu_backend_for_w8a8();
                }
            }
            loader.load_tensors(*ctx_audio);
            if (ctx_audio->backend_is_npu() &&
                ctx_audio->model.aicas_w8a8_enabled &&
                ctx_audio->should_preload_aicas_w8a8_for_npu()) {
                ctx_audio->preload_aicas_w8a8_for_npu();
            }
            loader.alloc_compute_meta(*ctx_audio);
        }

    } catch (const std::exception & e) {
        LOG_ERR("%s: failed to load model '%s': %s\n", __func__, fname, e.what());
        if (ctx_vision) {
            delete ctx_vision;
        }
        if (ctx_audio) {
            delete ctx_audio;
        }
        return {nullptr, nullptr};
    }

    return {ctx_vision, ctx_audio};
}

struct clip_image_size * clip_image_size_init() {
    struct clip_image_size * load_image_size = new struct clip_image_size();
    load_image_size->width = 448;
    load_image_size->height = 448;
    return load_image_size;
}

struct clip_image_u8 * clip_image_u8_init() {
    return new clip_image_u8();
}

struct clip_image_f32 * clip_image_f32_init() {
    return new clip_image_f32();
}

struct clip_image_f32_batch * clip_image_f32_batch_init() {
    return new clip_image_f32_batch();
}

unsigned char * clip_image_u8_get_data(struct clip_image_u8 * img, uint32_t * nx, uint32_t * ny) {
    if (nx) *nx = img->nx;
    if (ny) *ny = img->ny;
    return img->buf.data();
}

void clip_image_size_free(struct clip_image_size * load_image_size) {
    if (load_image_size == nullptr) {
        return;
    }
    delete load_image_size;
}
void clip_image_u8_free(struct clip_image_u8  * img) { if (img) delete img; }
void clip_image_f32_free(struct clip_image_f32 * img) { if (img) delete img; }
void clip_image_u8_batch_free(struct clip_image_u8_batch * batch) { if (batch) delete batch; }
void clip_image_f32_batch_free(struct clip_image_f32_batch * batch) { if (batch) delete batch; }

size_t clip_image_f32_batch_n_images(const struct clip_image_f32_batch * batch) {
    return batch->entries.size();
}

size_t clip_image_f32_batch_nx(const struct clip_image_f32_batch * batch, int idx) {
    if (idx < 0 || idx >= (int)batch->entries.size()) {
        LOG_ERR("%s: invalid index %d\n", __func__, idx);
        return 0;
    }
    return batch->entries[idx]->nx;
}

size_t clip_image_f32_batch_ny(const struct clip_image_f32_batch * batch, int idx) {
    if (idx < 0 || idx >= (int)batch->entries.size()) {
        LOG_ERR("%s: invalid index %d\n", __func__, idx);
        return 0;
    }
    return batch->entries[idx]->ny;
}

clip_image_f32 * clip_image_f32_get_img(const struct clip_image_f32_batch * batch, int idx) {
    if (idx < 0 || idx >= (int)batch->entries.size()) {
        LOG_ERR("%s: invalid index %d\n", __func__, idx);
        return nullptr;
    }
    return batch->entries[idx].get();
}

void clip_build_img_from_pixels(const unsigned char * rgb_pixels, int nx, int ny, clip_image_u8 * img) {
    img->nx = nx;
    img->ny = ny;
    img->buf.resize(3 * nx * ny);
    memcpy(img->buf.data(), rgb_pixels, img->buf.size());
}

// Normalize image to float32 - careful with pytorch .to(model.device, dtype=torch.float16) - this sometimes reduces precision (32>16>32), sometimes not
static void normalize_image_u8_to_f32(const clip_image_u8 & src, clip_image_f32 & dst, const float mean[3], const float std[3]) {
    dst.nx = src.nx;
    dst.ny = src.ny;
    dst.buf.resize(src.buf.size());

    // TODO @ngxson : seems like this could be done more efficiently on cgraph
    for (size_t i = 0; i < src.buf.size(); ++i) {
        int c = i % 3; // rgb
        dst.buf[i] = (static_cast<float>(src.buf[i]) / 255.0f - mean[c]) / std[c];
    }
}

// set of tools to manupulate images
// in the future, we can have HW acceleration by allowing this struct to access 3rd party lib like imagick or opencv
struct image_manipulation {
    // Bilinear resize function
    static void bilinear_resize(const clip_image_u8& src, clip_image_u8& dst, int target_width, int target_height) {
        dst.nx = target_width;
        dst.ny = target_height;
        dst.buf.resize(3 * target_width * target_height);

        float x_ratio = static_cast<float>(src.nx - 1) / target_width;
        float y_ratio = static_cast<float>(src.ny - 1) / target_height;

        for (int y = 0; y < target_height; y++) {
            for (int x = 0; x < target_width; x++) {
                float px = x_ratio * x;
                float py = y_ratio * y;
                int x_floor = static_cast<int>(px);
                int y_floor = static_cast<int>(py);
                float x_lerp = px - x_floor;
                float y_lerp = py - y_floor;

                for (int c = 0; c < 3; c++) {
                    float top = lerp(
                        static_cast<float>(src.buf[3 * (y_floor * src.nx + x_floor) + c]),
                        static_cast<float>(src.buf[3 * (y_floor * src.nx + (x_floor + 1)) + c]),
                        x_lerp
                    );
                    float bottom = lerp(
                        static_cast<float>(src.buf[3 * ((y_floor + 1) * src.nx + x_floor) + c]),
                        static_cast<float>(src.buf[3 * ((y_floor + 1) * src.nx + (x_floor + 1)) + c]),
                        x_lerp
                    );
                    dst.buf[3 * (y * target_width + x) + c] = static_cast<uint8_t>(lerp(top, bottom, y_lerp));
                }
            }
        }
    }

    // Bicubic resize function
    // part of image will be cropped if the aspect ratio is different
    static bool bicubic_resize(const clip_image_u8 & img, clip_image_u8 & dst, int target_width, int target_height) {
        const int nx = img.nx;
        const int ny = img.ny;

        dst.nx = target_width;
        dst.ny = target_height;
        dst.buf.resize(3 * target_width * target_height);

        float Cc;
        float C[5] = {};
        float d0, d2, d3, a0, a1, a2, a3;
        int i, j, k, jj;
        int x, y;
        float dx, dy;
        float tx, ty;

        tx = (float)nx / (float)target_width;
        ty = (float)ny / (float)target_height;

        // Bicubic interpolation; adapted from ViT.cpp, inspired from :
        //    -> https://github.com/yglukhov/bicubic-interpolation-image-processing/blob/master/libimage.c#L36
        //    -> https://en.wikipedia.org/wiki/Bicubic_interpolation

        for (i = 0; i < target_height; i++) {
            for (j = 0; j < target_width; j++) {
                x = (int)(tx * j);
                y = (int)(ty * i);

                dx = tx * j - x;
                dy = ty * i - y;

                for (k = 0; k < 3; k++) {
                    for (jj = 0; jj <= 3; jj++) {
                        d0 = img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x - 1, 0, nx - 1)) * 3 + k] - img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x, 0, nx - 1)) * 3 + k];
                        d2 = img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x + 1, 0, nx - 1)) * 3 + k] - img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x, 0, nx - 1)) * 3 + k];
                        d3 = img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x + 2, 0, nx - 1)) * 3 + k] - img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x, 0, nx - 1)) * 3 + k];
                        a0 = img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x, 0, nx - 1)) * 3 + k];

                        a1 = -1.0 / 3 * d0 + d2 - 1.0 / 6 * d3;
                        a2 =  1.0 / 2 * d0 +      1.0 / 2 * d2;
                        a3 = -1.0 / 6 * d0 -      1.0 / 2 * d2 + 1.0 / 6 * d3;

                        C[jj] = a0 + a1 * dx + a2 * dx * dx + a3 * dx * dx * dx;

                        d0 = C[0] - C[1];
                        d2 = C[2] - C[1];
                        d3 = C[3] - C[1];
                        a0 = C[1];
                        a1 = -1.0 / 3 * d0 + d2 - 1.0 / 6 * d3;
                        a2 =  1.0 / 2 * d0 +      1.0 / 2 * d2;
                        a3 = -1.0 / 6 * d0 -      1.0 / 2 * d2 + 1.0 / 6 * d3;
                        Cc = a0 + a1 * dy + a2 * dy * dy + a3 * dy * dy * dy;

                        const uint8_t Cc2 = std::min(std::max(std::round(Cc), 0.0f), 255.0f);
                        dst.buf[(i * target_width + j) * 3 + k] = float(Cc2);
                    }
                }
            }
        }

        return true;
    }

    // llava-1.6 type of resize_and_pad
    // if the ratio is not 1:1, padding with pad_color will be applied
    // pad_color is single channel, default is 0 (black)
    static void resize_and_pad_image(const clip_image_u8 & image, clip_image_u8 & dst, const clip_image_size & target_resolution, std::array<uint8_t, 3> pad_color = {0, 0, 0}) {
        int target_width  = target_resolution.width;
        int target_height = target_resolution.height;

        float scale_w = static_cast<float>(target_width) / image.nx;
        float scale_h = static_cast<float>(target_height) / image.ny;

        int new_width, new_height;

        if (scale_w < scale_h) {
            new_width  = target_width;
            new_height = std::min(static_cast<int>(std::ceil(image.ny * scale_w)), target_height);
        } else {
            new_height = target_height;
            new_width  = std::min(static_cast<int>(std::ceil(image.nx * scale_h)), target_width);
        }

        clip_image_u8 resized_image;
        bicubic_resize(image, resized_image, new_width, new_height);

        clip_image_u8 padded_image;
        padded_image.nx = target_width;
        padded_image.ny = target_height;
        padded_image.buf.resize(3 * target_width * target_height);

        // Fill the padded image with the fill color
        for (size_t i = 0; i < padded_image.buf.size(); i += 3) {
            padded_image.buf[i]     = pad_color[0];
            padded_image.buf[i + 1] = pad_color[1];
            padded_image.buf[i + 2] = pad_color[2];
        }

        // Calculate padding offsets
        int pad_x = (target_width  - new_width)  / 2;
        int pad_y = (target_height - new_height) / 2;

        // Copy the resized image into the center of the padded buffer
        for (int y = 0; y < new_height; ++y) {
            for (int x = 0; x < new_width; ++x) {
                for (int c = 0; c < 3; ++c) {
                    padded_image.buf[3 * ((y + pad_y) * target_width + (x + pad_x)) + c] = resized_image.buf[3 * (y * new_width + x) + c];
                }
            }
        }
        dst = std::move(padded_image);
    }

    static void crop_image(const clip_image_u8 & image, clip_image_u8 & dst, int x, int y, int w, int h) {
        dst.nx = w;
        dst.ny = h;
        dst.buf.resize(3 * w * h);

        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                int src_idx = 3 * ((y + i)*image.nx + (x + j));
                int dst_idx = 3 * (i*w + j);
                dst.buf[dst_idx]     = image.buf[src_idx];
                dst.buf[dst_idx + 1] = image.buf[src_idx + 1];
                dst.buf[dst_idx + 2] = image.buf[src_idx + 2];
            }
        }
    }

    // calculate the size of the **resized** image, while preserving the aspect ratio
    // the calculated size will be aligned to the nearest multiple of align_size
    // if H or W size is larger than max_dimension, it will be resized to max_dimension
    static clip_image_size calc_size_preserved_ratio(const clip_image_size & inp_size, const int align_size, const int max_dimension) {
        if (inp_size.width <= 0 || inp_size.height <= 0 || align_size <= 0 || max_dimension <= 0) {
            return {0, 0};
        }

        float scale = std::min(1.0f, std::min(static_cast<float>(max_dimension) / inp_size.width,
                                              static_cast<float>(max_dimension) / inp_size.height));

        float target_width_f  = static_cast<float>(inp_size.width)  * scale;
        float target_height_f = static_cast<float>(inp_size.height) * scale;

        int aligned_width  = CLIP_ALIGN((int)target_width_f,  align_size);
        int aligned_height = CLIP_ALIGN((int)target_height_f, align_size);

        return {aligned_width, aligned_height};
    }

private:
    static inline int clip(int x, int lower, int upper) {
        return std::max(lower, std::min(x, upper));
    }

    // Linear interpolation between two points
    static inline float lerp(float s, float e, float t) {
        return s + (e - s) * t;
    }
};

/**
 * implementation of LLaVA-UHD:
 *  - https://arxiv.org/pdf/2403.11703
 *  - https://github.com/thunlp/LLaVA-UHD
 *  - https://github.com/thunlp/LLaVA-UHD/blob/302301bc2175f7e717fb8548516188e89f649753/llava_uhd/train/llava-uhd/slice_logic.py#L118
 *
 * overview:
 *   - an image always have a single overview (downscaled image)
 *   - an image can have 0 or multiple slices, depending on the image size
 *   - each slice can then be considered as a separate image
 *
 * for example:
 *
 * [overview] --> [slice 1] --> [slice 2]
 *           |                |
 *           +--> [slice 3] --> [slice 4]
 */
struct llava_uhd {
    struct slice_coordinates {
        int x;
        int y;
        clip_image_size size;
    };

    struct slice_instructions {
        clip_image_size overview_size; // size of downscaled image
        clip_image_size refined_size;  // size of image right before slicing (must be multiple of slice size)
        clip_image_size grid_size;     // grid_size.width * grid_size.height = number of slices
        std::vector<slice_coordinates> slices;
        bool padding_refined = false;  // if true, refine image will be padded to the grid size (e.g. llava-1.6)
    };

    static slice_instructions get_slice_instructions(struct clip_ctx * ctx, const clip_image_size & original_size) {
        slice_instructions res;
        const int patch_size      = clip_get_patch_size(ctx);
        const int slice_size      = clip_get_image_size(ctx);
        const int original_width  = original_size.width;
        const int original_height = original_size.height;

        const bool has_slices    = original_size.width > slice_size || original_size.height > slice_size;
        const bool has_pinpoints = !ctx->model.hparams.image_res_candidates.empty();

        if (!has_slices) {
            // skip slicing logic
            res.overview_size = clip_image_size{slice_size, slice_size};
            res.refined_size  = clip_image_size{0, 0};
            res.grid_size     = clip_image_size{0, 0};

            return res;
        }

        if (has_pinpoints) {
            // has pinpoints, use them to calculate the grid size (e.g. llava-1.6)
            auto refine_size = llava_uhd::select_best_resolution(
                original_size,
                ctx->model.hparams.image_res_candidates);
            res.overview_size   = clip_image_size{slice_size, slice_size};
            res.refined_size    = refine_size;
            res.grid_size       = clip_image_size{0, 0};
            res.padding_refined = true;

            LOG_DBG("%s: using pinpoints for slicing\n", __func__);
            LOG_DBG("%s: original size: %d x %d, overview size: %d x %d, refined size: %d x %d\n",
                    __func__, original_width, original_height,
                    res.overview_size.width, res.overview_size.height,
                    res.refined_size.width,  res.refined_size.height);

            for (int y = 0; y < refine_size.height; y += slice_size) {
                for (int x = 0; x < refine_size.width; x += slice_size) {
                    slice_coordinates slice;
                    slice.x = x;
                    slice.y = y;
                    slice.size.width  = std::min(slice_size, refine_size.width  - x);
                    slice.size.height = std::min(slice_size, refine_size.height - y);
                    res.slices.push_back(slice);
                    LOG_DBG("%s: slice %d: x=%d, y=%d, size=%dx%d\n",
                            __func__, (int)res.slices.size() - 1,
                            slice.x, slice.y, slice.size.width, slice.size.height);
                }
            }

            res.grid_size.height = refine_size.height / slice_size;
            res.grid_size.width  = refine_size.width  / slice_size;
            LOG_DBG("%s: grid size: %d x %d\n", __func__, res.grid_size.width, res.grid_size.height);

            return res;
        }

        // no pinpoints, dynamically calculate the grid size (e.g. minicpmv)

        auto best_size    = get_best_resize(original_size, slice_size, patch_size, !has_slices);
        res.overview_size = best_size;

        {
            const int max_slice_nums = 9; // TODO: this is only used by minicpmv, maybe remove it
            const float log_ratio = log((float)original_width / original_height);
            const float ratio = (float)original_width * original_height / (slice_size * slice_size);
            const int multiple = fmin(ceil(ratio), max_slice_nums);

            auto best_grid   = get_best_grid(max_slice_nums, multiple, log_ratio);
            auto refine_size = get_refine_size(original_size, best_grid, slice_size, patch_size, true);
            res.grid_size    = best_grid;
            res.refined_size = refine_size;

            LOG_DBG("%s: original size: %d x %d, overview size: %d x %d, refined size: %d x %d, grid size: %d x %d\n",
                    __func__, original_width, original_height,
                    res.overview_size.width, res.overview_size.height,
                    res.refined_size.width, res.refined_size.height,
                    res.grid_size.width, res.grid_size.height);

            int width  = refine_size.width;
            int height = refine_size.height;
            int grid_x = int(width  / best_grid.width);
            int grid_y = int(height / best_grid.height);
            for (int patches_y = 0,                    ic = 0;
                    patches_y < refine_size.height && ic < best_grid.height;
                    patches_y += grid_y,              ic += 1) {
                for (int patches_x = 0,                   jc = 0;
                        patches_x < refine_size.width && jc < best_grid.width;
                        patches_x += grid_x,             jc += 1) {
                    slice_coordinates slice;
                    slice.x = patches_x;
                    slice.y = patches_y;
                    slice.size.width  = grid_x;
                    slice.size.height = grid_y;
                    res.slices.push_back(slice);
                    LOG_DBG("%s: slice %d: x=%d, y=%d, size=%dx%d\n",
                            __func__, (int)res.slices.size() - 1,
                            slice.x, slice.y, slice.size.width, slice.size.height);
                }
            }
        }

        return res;
    }

    static std::vector<clip_image_u8_ptr> slice_image(const clip_image_u8 * img, const slice_instructions & inst) {
        std::vector<clip_image_u8_ptr> output;

        // resize to overview size
        clip_image_u8_ptr resized_img(clip_image_u8_init());
        image_manipulation::bicubic_resize(*img, *resized_img, inst.overview_size.width, inst.overview_size.height);
        output.push_back(std::move(resized_img));
        if (inst.slices.empty()) {
            // no slices, just return the resized image
            return output;
        }

        // resize to refined size
        clip_image_u8_ptr refined_img(clip_image_u8_init());
        if (inst.padding_refined) {
            image_manipulation::resize_and_pad_image(*img, *refined_img, inst.refined_size);
        } else {
            image_manipulation::bilinear_resize(*img, *refined_img, inst.refined_size.width, inst.refined_size.height);
        }

        // create slices
        for (const auto & slice : inst.slices) {
            int x = slice.x;
            int y = slice.y;
            int w = slice.size.width;
            int h = slice.size.height;

            clip_image_u8_ptr img_slice(clip_image_u8_init());
            image_manipulation::crop_image(*refined_img, *img_slice, x, y, w, h);
            output.push_back(std::move(img_slice));
        }

        return output;
    }

private:
    static clip_image_size get_best_resize(const clip_image_size & original_size, int scale_resolution, int patch_size, bool allow_upscale = false) {
        int width  = original_size.width;
        int height = original_size.height;
        if ((width * height > scale_resolution * scale_resolution) || allow_upscale) {
            float r = static_cast<float>(width) / height;
            height  = static_cast<int>(scale_resolution / std::sqrt(r));
            width   = static_cast<int>(height * r);
        }
        clip_image_size res;
        res.width  = ensure_divide(width,  patch_size);
        res.height = ensure_divide(height, patch_size);
        return res;
    }

    static clip_image_size resize_maintain_aspect_ratio(const clip_image_size & orig, const clip_image_size & target_max) {
        float scale_width  = static_cast<float>(target_max.width)  / orig.width;
        float scale_height = static_cast<float>(target_max.height) / orig.height;
        float scale = std::min(scale_width, scale_height);
        return clip_image_size{
            static_cast<int>(orig.width  * scale),
            static_cast<int>(orig.height * scale),
        };
    }

    /**
     * Selects the best resolution from a list of possible resolutions based on the original size.
     *
     * For example, when given a list of resolutions:
     *  - 100x100
     *  - 200x100
     *  - 100x200
     *  - 200x200
     *
     * And an input image of size 111x200, then 100x200 is the best fit (least wasted resolution).
     *
     * @param original_size The original size of the image
     * @param possible_resolutions A list of possible resolutions
     * @return The best fit resolution
     */
    static clip_image_size select_best_resolution(const clip_image_size & original_size, const std::vector<clip_image_size> & possible_resolutions) {
        clip_image_size best_fit;
        int min_wasted_area = std::numeric_limits<int>::max();
        int max_effective_resolution = 0;

        for (const clip_image_size & candidate : possible_resolutions) {
            auto target_size = resize_maintain_aspect_ratio(original_size, candidate);
            int effective_resolution = std::min(
                target_size.width * target_size.height,
                original_size.width * original_size.height);
            int wasted_area = (candidate.width * candidate.height) - effective_resolution;

            if (effective_resolution > max_effective_resolution || (effective_resolution == max_effective_resolution && wasted_area < min_wasted_area)) {
                max_effective_resolution = effective_resolution;
                min_wasted_area = wasted_area;
                best_fit = candidate;
            }

            LOG_DBG("%s: candidate: %d x %d, target: %d x %d, wasted: %d, effective: %d\n", __func__, candidate.width, candidate.height, target_size.width, target_size.height, wasted_area, effective_resolution);
        }

        return best_fit;
    }

    static int ensure_divide(int length, int patch_size) {
        return std::max(static_cast<int>(std::round(static_cast<float>(length) / patch_size) * patch_size), patch_size);
    }

    static clip_image_size get_refine_size(const clip_image_size & original_size, const clip_image_size & grid, int scale_resolution, int patch_size, bool allow_upscale = false) {
        int width  = original_size.width;
        int height = original_size.height;
        int grid_x = grid.width;
        int grid_y = grid.height;

        int refine_width  = ensure_divide(width, grid_x);
        int refine_height = ensure_divide(height, grid_y);

        clip_image_size grid_size;
        grid_size.width  = refine_width  / grid_x;
        grid_size.height = refine_height / grid_y;

        auto best_grid_size  = get_best_resize(grid_size, scale_resolution, patch_size, allow_upscale);
        int best_grid_width  = best_grid_size.width;
        int best_grid_height = best_grid_size.height;

        clip_image_size refine_size;
        refine_size.width  = best_grid_width  * grid_x;
        refine_size.height = best_grid_height * grid_y;
        return refine_size;
    }

    static clip_image_size get_best_grid(const int max_slice_nums, const int multiple, const float log_ratio) {
        std::vector<int> candidate_split_grids_nums;
        for (int i : {multiple - 1, multiple, multiple + 1}) {
            if (i == 1 || i > max_slice_nums) {
                continue;
            }
            candidate_split_grids_nums.push_back(i);
        }

        std::vector<clip_image_size> candidate_grids;
        for (int split_grids_nums : candidate_split_grids_nums) {
            int m = 1;
            while (m <= split_grids_nums) {
                if (split_grids_nums % m == 0) {
                    candidate_grids.push_back(clip_image_size{m, split_grids_nums / m});
                }
                ++m;
            }
        }

        clip_image_size best_grid{1, 1};
        float min_error = std::numeric_limits<float>::infinity();
        for (const auto& grid : candidate_grids) {
            float error = std::abs(log_ratio - std::log(1.0 * grid.width / grid.height));
            if (error < min_error) {
                best_grid = grid;
                min_error = error;
            }
        }
        return best_grid;
    }
};

// returns the normalized float tensor for llava-1.5, for spatial_unpad with anyres processing for llava-1.6 it returns the normalized image patch tensors as a vector
// res_imgs memory is being allocated here, previous allocations will be freed if found
bool clip_image_preprocess(struct clip_ctx * ctx, const clip_image_u8 * img, struct clip_image_f32_batch * res_imgs) {
    clip_image_size original_size{img->nx, img->ny};
    bool pad_to_square = true;
    auto & params = ctx->model.hparams;
    // The model config actually contains all we need to decide on how to preprocess, here we automatically switch to the new llava-1.6 preprocessing
    if (params.mm_patch_merge_type == PATCH_MERGE_SPATIAL_UNPAD) {
        pad_to_square = false;
    }

    if (clip_is_minicpmv(ctx)) {
        auto const inst = llava_uhd::get_slice_instructions(ctx, original_size);
        std::vector<clip_image_u8_ptr> imgs = llava_uhd::slice_image(img, inst);

        for (size_t i = 0; i < imgs.size(); ++i) {
            // clip_image_save_to_bmp(*imgs[i], "slice_" + std::to_string(i) + ".bmp");
            clip_image_f32_ptr res(clip_image_f32_init());
            normalize_image_u8_to_f32(*imgs[i], *res, params.image_mean, params.image_std);
            res_imgs->entries.push_back(std::move(res));
        }

        res_imgs->grid_x = inst.grid_size.width;
        res_imgs->grid_y = inst.grid_size.height;
        return true;

    } else if (ctx->proj_type() == PROJECTOR_TYPE_QWEN2VL || ctx->proj_type() == PROJECTOR_TYPE_QWEN25VL) {
        clip_image_u8 resized;
        auto patch_size = params.patch_size * 2;
        auto new_size = image_manipulation::calc_size_preserved_ratio(original_size, patch_size, params.image_size);
        image_manipulation::bicubic_resize(*img, resized, new_size.width, new_size.height);

        clip_image_f32_ptr img_f32(clip_image_f32_init());
        // clip_image_f32_ptr res(clip_image_f32_init());
        normalize_image_u8_to_f32(resized, *img_f32, params.image_mean, params.image_std);
        // res_imgs->data[0] = *res;
        res_imgs->entries.push_back(std::move(img_f32));
        return true;
    } else if (ctx->proj_type() == PROJECTOR_TYPE_IDEFICS3) {
        // The refined size has two steps:
        // 1. Resize w/ aspect-ratio preserving such that the longer side is
        //      the preprocessor longest size
        // 2. Resize w/out preserving aspect ratio such that both sides are
        //      multiples of image_size (always rounding up)
        //
        // CITE: https://github.com/huggingface/transformers/blob/main/src/transformers/models/idefics3/image_processing_idefics3.py#L737
        const clip_image_size refined_size = image_manipulation::calc_size_preserved_ratio(
            original_size, params.image_size, params.preproc_image_size);

        llava_uhd::slice_instructions instructions;
        instructions.overview_size = clip_image_size{params.image_size, params.image_size};
        instructions.refined_size = refined_size;
        instructions.grid_size = clip_image_size{
            static_cast<int>(std::ceil(static_cast<float>(refined_size.width) / params.image_size)),
            static_cast<int>(std::ceil(static_cast<float>(refined_size.height) / params.image_size)),
        };
        for (int y = 0; y < refined_size.height; y += params.image_size) {
            for (int x = 0; x < refined_size.width; x += params.image_size) {
                instructions.slices.push_back(llava_uhd::slice_coordinates{
                    /* x    */x,
                    /* y    */y,
                    /* size */clip_image_size{
                        std::min(params.image_size, refined_size.width - x),
                        std::min(params.image_size, refined_size.height - y)
                    }
                });
            }
        }
        auto imgs = llava_uhd::slice_image(img, instructions);

        // cast and normalize to f32
        for (size_t i = 0; i < imgs.size(); ++i) {
            // clip_image_save_to_bmp(*imgs[i], "slice_" + std::to_string(i) + ".bmp");
            clip_image_f32_ptr res(clip_image_f32_init());
            normalize_image_u8_to_f32(*imgs[i], *res, params.image_mean, params.image_std);
            res_imgs->entries.push_back(std::move(res));
        }

        res_imgs->grid_x = instructions.grid_size.width;
        res_imgs->grid_y = instructions.grid_size.height;
        return true;
    } else if (ctx->proj_type() == PROJECTOR_TYPE_GLM_EDGE
            || ctx->proj_type() == PROJECTOR_TYPE_GEMMA3
            || ctx->proj_type() == PROJECTOR_TYPE_INTERNVL // TODO @ngxson : support dynamic resolution
    ) {
        clip_image_u8 resized_image;
        int sz = params.image_size;
        image_manipulation::resize_and_pad_image(*img, resized_image, {sz, sz});
        clip_image_f32_ptr img_f32(clip_image_f32_init());
        //clip_image_save_to_bmp(resized_image, "resized.bmp");
        normalize_image_u8_to_f32(resized_image, *img_f32, params.image_mean, params.image_std);
        res_imgs->entries.push_back(std::move(img_f32));
        return true;

    } else if (ctx->proj_type() == PROJECTOR_TYPE_PIXTRAL) {
        clip_image_u8 resized_image;
        auto new_size = image_manipulation::calc_size_preserved_ratio(original_size, params.patch_size, params.image_size);
        image_manipulation::bilinear_resize(*img, resized_image, new_size.width, new_size.height);
        clip_image_f32_ptr img_f32(clip_image_f32_init());
        normalize_image_u8_to_f32(resized_image, *img_f32, params.image_mean, params.image_std);
        res_imgs->entries.push_back(std::move(img_f32));
        return true;

    } else if (ctx->proj_type() == PROJECTOR_TYPE_LLAMA4) {
        GGML_ASSERT(!params.image_res_candidates.empty());
        auto const inst = llava_uhd::get_slice_instructions(ctx, original_size);
        std::vector<clip_image_u8_ptr> imgs = llava_uhd::slice_image(img, inst);

        for (size_t i = 0; i < imgs.size(); ++i) {
            clip_image_f32_ptr res(clip_image_f32_init());
            normalize_image_u8_to_f32(*imgs[i], *res, params.image_mean, params.image_std);
            res_imgs->entries.push_back(std::move(res));
        }

        res_imgs->grid_x = inst.grid_size.width;
        res_imgs->grid_y = inst.grid_size.height;
        return true;

    } else if ( ctx->proj_type() == PROJECTOR_TYPE_LFM2
             || ctx->proj_type() == PROJECTOR_TYPE_KIMIVL
    ) {
        GGML_ASSERT(params.proj_scale_factor);

        // smart resize
        const int width = img->nx;
        const int height = img->ny;
        const int total_factor = params.patch_size * params.proj_scale_factor;
        constexpr int min_image_tokens = 64;
        constexpr int max_image_tokens = 1024;
        const float min_pixels = min_image_tokens * total_factor * total_factor;
        const float max_pixels = max_image_tokens * total_factor * total_factor;

        auto round_by_factor = [f = total_factor](float x) { return static_cast<int>(std::nearbyintf(x / static_cast<float>(f))) * f; };
        auto ceil_by_factor  = [f = total_factor](float x) { return static_cast<int>(std::ceil(x / static_cast<float>(f))) * f; };
        auto floor_by_factor = [f = total_factor](float x) { return static_cast<int>(std::floor(x / static_cast<float>(f))) * f; };

        int h_bar = std::max(total_factor, round_by_factor(height));
        int w_bar = std::max(total_factor, round_by_factor(width));

        if (h_bar * w_bar > max_pixels) {
            const auto beta = std::sqrt((height * width) / max_pixels);
            h_bar = std::max(total_factor, floor_by_factor(height / beta));
            w_bar = std::max(total_factor, floor_by_factor(width / beta));
        } else if (h_bar * w_bar < min_pixels) {
            const auto beta = std::sqrt(min_pixels / (height * width));
            h_bar = ceil_by_factor(height * beta);
            w_bar = ceil_by_factor(width * beta);
        }

        const std::array<uint8_t, 3> pad_color = {122, 116, 104};

        clip_image_u8 resized_img;
        image_manipulation::resize_and_pad_image(*img, resized_img, clip_image_size{w_bar, h_bar}, pad_color);
        clip_image_f32_ptr res(clip_image_f32_init());
        normalize_image_u8_to_f32(resized_img, *res, params.image_mean, params.image_std);
        res_imgs->entries.push_back(std::move(res));
        return true;
    }

    // the logic below is to pad the shorter side to the longer side with a background color: rgb(122, 116, 104)
    // see https://github.com/haotian-liu/LLaVA/blob/e854a2bf85118c504f6f16bf5c3c7c92f8fa8c6b/llava/conversation.py#L113-L156

    clip_image_u8_ptr temp(clip_image_u8_init()); // we will keep the input image data here temporarily

    if (pad_to_square) {
        // for llava-1.5, we resize image to a square, and pad the shorter side with a background color
        // see https://github.com/haotian-liu/LLaVA/blob/e854a2bf85118c504f6f16bf5c3c7c92f8fa8c6b/llava/conversation.py#L113-L156
        const int longer_side = std::max(img->nx, img->ny);
        temp->nx = longer_side;
        temp->ny = longer_side;
        temp->buf.resize(3 * longer_side * longer_side);

        // background color in RGB from LLaVA (this is the mean rgb color * 255)
        const std::array<uint8_t, 3> pad_color = {122, 116, 104};

        // resize the image to the target_size
        image_manipulation::resize_and_pad_image(*img, *temp, clip_image_size{params.image_size, params.image_size}, pad_color);

        clip_image_f32_ptr res(clip_image_f32_init());
        normalize_image_u8_to_f32(*temp, *res, params.image_mean, params.image_std);
        res_imgs->entries.push_back(std::move(res));
        return true;

    } else if (!params.image_res_candidates.empty()) {
        // "spatial_unpad" with "anyres" processing for llava-1.6
        auto const inst = llava_uhd::get_slice_instructions(ctx, original_size);
        std::vector<clip_image_u8_ptr> imgs = llava_uhd::slice_image(img, inst);

        for (size_t i = 0; i < imgs.size(); ++i) {
            // clip_image_save_to_bmp(*imgs[i], "slice_" + std::to_string(i) + ".bmp");
            clip_image_f32_ptr res(clip_image_f32_init());
            normalize_image_u8_to_f32(*imgs[i], *res, params.image_mean, params.image_std);
            res_imgs->entries.push_back(std::move(res));
        }

        return true;
    } else {
        GGML_ABORT("Unknown image preprocessing type");
    }

}

ggml_tensor * clip_get_newline_tensor(const struct clip_ctx * ctx) {
    return ctx->model.image_newline;
}

void clip_free(clip_ctx * ctx) {
    if (ctx == nullptr) {
        return;
    }
    delete ctx;
}

// deprecated
size_t clip_embd_nbytes(const struct clip_ctx * ctx) {
    const int32_t nx = ctx->model.hparams.image_size;
    const int32_t ny = ctx->model.hparams.image_size;
    return clip_embd_nbytes_by_img(ctx, nx, ny);
}

size_t clip_embd_nbytes_by_img(const struct clip_ctx * ctx, int img_w, int img_h) {
    clip_image_f32 img;
    img.nx = img_w;
    img.ny = img_h;
    return clip_n_output_tokens(ctx, &img) * clip_n_mmproj_embd(ctx) * sizeof(float);
}

int32_t clip_get_image_size(const struct clip_ctx * ctx) {
    return ctx->model.hparams.image_size;
}

int32_t clip_get_patch_size(const struct clip_ctx * ctx) {
    return ctx->model.hparams.patch_size;
}

int32_t clip_get_hidden_size(const struct clip_ctx * ctx) {
    return ctx->model.hparams.n_embd;
}

const char * clip_patch_merge_type(const struct clip_ctx * ctx) {
    return ctx->model.hparams.mm_patch_merge_type == PATCH_MERGE_SPATIAL_UNPAD ? "spatial_unpad" : "flat";
}

int clip_n_output_tokens_x(const struct clip_ctx * ctx, struct clip_image_f32 * img) {
    const auto & params = ctx->model.hparams;
    const int n_total = clip_n_output_tokens(ctx, img);
    if (ctx->proj_type() == PROJECTOR_TYPE_QWEN2VL || ctx->proj_type() == PROJECTOR_TYPE_QWEN25VL) {
        return img->nx / (params.patch_size * 2) + (int)(img->nx % params.patch_size > 0);
    }
    return n_total;
}

int clip_n_output_tokens_y(const struct clip_ctx * ctx, struct clip_image_f32 * img) {
    const auto & params = ctx->model.hparams;
    if (ctx->proj_type() == PROJECTOR_TYPE_QWEN2VL || ctx->proj_type() == PROJECTOR_TYPE_QWEN25VL) {
        return img->ny / (params.patch_size * 2) + (int)(img->ny % params.patch_size > 0);
    }
    return 1;
}

int clip_n_output_tokens(const struct clip_ctx * ctx, struct clip_image_f32 * img) {
    const auto & params = ctx->model.hparams;

    // for models with fixed size image, the input image is already pre-processed and resized to square
    int patch_size = params.patch_size;
    int n_patches = (img->nx / patch_size) * (img->ny / patch_size);

    projector_type proj = ctx->proj_type();

    switch (proj) {
        case PROJECTOR_TYPE_MLP:
        case PROJECTOR_TYPE_MLP_NORM:
            {
                // do nothing
            } break;
        case PROJECTOR_TYPE_LDP:
        case PROJECTOR_TYPE_LDPV2:
        case PROJECTOR_TYPE_GLM_EDGE:
            {
                n_patches /= 4;
                if (ctx->model.mm_glm_tok_boi) {
                    n_patches += 2; // for BOI and EOI token embeddings
                }
            } break;
        case PROJECTOR_TYPE_MINICPMV:
            {
                // Use actual config value if available, otherwise fall back to hardcoded values
                if (params.minicpmv_query_num > 0) {
                    n_patches = params.minicpmv_query_num;
                } else {
                    // Fallback to hardcoded values for legacy models
                    if (params.minicpmv_version == 2) {
                        n_patches = 96;
                    } else if (params.minicpmv_version == 3) {
                        n_patches = 64;
                    } else if (params.minicpmv_version == 4) {
                        n_patches = 64;
                    } else if (params.minicpmv_version == 5) {
                        // MiniCPM-V 4.0
                        n_patches = 64;
                    } else if (params.minicpmv_version == 6) {
                        // MiniCPM-V 4.5
                        n_patches = 64;
                    } else {
                        GGML_ABORT("Unknown minicpmv version");
                    }
                }
            } break;
        case PROJECTOR_TYPE_QWEN2VL:
        case PROJECTOR_TYPE_QWEN25VL:
            {
                // dynamic size (2 conv, so double patch size)
                int patch_size = params.patch_size * 2;
                int x_patch = img->nx / patch_size + (int)(img->nx % patch_size > 0);
                int y_patch = img->ny / patch_size + (int)(img->ny % patch_size > 0);
                n_patches = x_patch * y_patch;
            } break;
        case PROJECTOR_TYPE_GEMMA3:
        case PROJECTOR_TYPE_IDEFICS3:
        case PROJECTOR_TYPE_INTERNVL:
        case PROJECTOR_TYPE_LLAMA4:
            {
                // both X and Y are downscaled by the scale factor
                int scale_factor = ctx->model.hparams.proj_scale_factor;
                n_patches /= (scale_factor * scale_factor);
            } break;
        case PROJECTOR_TYPE_LFM2:
        case PROJECTOR_TYPE_KIMIVL:
            {
                // dynamic size
                int scale_factor = ctx->model.hparams.proj_scale_factor;
                int out_patch_size = params.patch_size * scale_factor;
                int x_patch = CLIP_ALIGN(img->nx, out_patch_size) / out_patch_size;
                int y_patch = CLIP_ALIGN(img->ny, out_patch_size) / out_patch_size;
                n_patches = x_patch * y_patch;
            } break;
        case PROJECTOR_TYPE_PIXTRAL:
            {
                // dynamic size
                int n_merge = params.spatial_merge_size;
                int n_patches_x = img->nx / patch_size / (n_merge > 0 ? n_merge : 1);
                int n_patches_y = img->ny / patch_size / (n_merge > 0 ? n_merge : 1);
                n_patches = n_patches_y * n_patches_x + n_patches_y - 1; // + one [IMG_BREAK] per row, except the last row
            } break;
        case PROJECTOR_TYPE_VOXTRAL:
        case PROJECTOR_TYPE_ULTRAVOX:
        case PROJECTOR_TYPE_QWEN2A:
            {
                n_patches = img->nx;

                const int proj_stack_factor = ctx->model.hparams.proj_stack_factor;
                if (ctx->model.audio_has_stack_frames()) {
                    GGML_ASSERT(proj_stack_factor > 0);
                    const int n_len = CLIP_ALIGN(n_patches, proj_stack_factor);
                    n_patches = n_len / proj_stack_factor;
                }

                // whisper downscales input token by half after conv1d
                n_patches /= 2;

                if (ctx->model.audio_has_avgpool()) {
                    // divide by 2 because of nn.AvgPool1d(2, stride=2)
                    n_patches /= 2;
                }
            } break;
        default:
            GGML_ABORT("unsupported projector type");
    }

    return n_patches;
}

static std::vector<std::vector<std::vector<float>>> get_1d_sincos_pos_embed_from_grid_new(int embed_dim, const std::vector<std::vector<float>> & pos) {
    assert(embed_dim % 2 == 0);
    int H = pos.size();
    int W = pos[0].size();

    std::vector<float> omega(embed_dim / 2);
    for (int i = 0; i < embed_dim / 2; ++i) {
        omega[i] = 1.0 / pow(10000.0, static_cast<float>(i) / (embed_dim / 2));
    }

    std::vector<std::vector<std::vector<float>>> emb(H, std::vector<std::vector<float>>(W, std::vector<float>(embed_dim)));
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            for (int d = 0; d < embed_dim / 2; ++d) {
                float out_value = pos[h][w] * omega[d];
                emb[h][w][d] = sin(out_value);
                emb[h][w][d + embed_dim / 2] = cos(out_value);
            }
        }
    }

    return emb;
}

static std::vector<std::vector<std::vector<float>>> get_2d_sincos_pos_embed_from_grid(int embed_dim, const std::vector<std::vector<std::vector<float>>> & grid) {
    assert(embed_dim % 2 == 0);
    std::vector<std::vector<std::vector<float>>> emb_h = get_1d_sincos_pos_embed_from_grid_new(embed_dim / 2, grid[0]); // (H, W, D/2)
    std::vector<std::vector<std::vector<float>>> emb_w = get_1d_sincos_pos_embed_from_grid_new(embed_dim / 2, grid[1]); // (H, W, D/2)

    int H = emb_h.size();
    int W = emb_h[0].size();
    std::vector<std::vector<std::vector<float>>> emb(H, std::vector<std::vector<float>>(W, std::vector<float>(embed_dim)));

    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            for (int d = 0; d < embed_dim / 2; ++d) {
                emb[h][w][d] = emb_h[h][w][d];
                emb[h][w][d + embed_dim / 2] = emb_w[h][w][d];
            }
        }
    }
    return emb;
}

static std::vector<std::vector<float>> get_2d_sincos_pos_embed(int embed_dim, const std::pair<int, int> image_size) {
    int grid_h_size = image_size.first;
    int grid_w_size = image_size.second;

    std::vector<float> grid_h(grid_h_size);
    std::vector<float> grid_w(grid_w_size);

    for (int i = 0; i < grid_h_size; ++i) {
        grid_h[i] = static_cast<float>(i);
    }
    for (int i = 0; i < grid_w_size; ++i) {
        grid_w[i] = static_cast<float>(i);
    }

    std::vector<std::vector<float>> grid(grid_h_size, std::vector<float>(grid_w_size));
    for (int h = 0; h < grid_h_size; ++h) {
        for (int w = 0; w < grid_w_size; ++w) {
            grid[h][w] = grid_w[w];
        }
    }
    std::vector<std::vector<std::vector<float>>> grid_2d = {grid, grid};
    for (int h = 0; h < grid_h_size; ++h) {
        for (int w = 0; w < grid_w_size; ++w) {
            grid_2d[0][h][w] = grid_h[h];
            grid_2d[1][h][w] = grid_w[w];
        }
    }

    std::vector<std::vector<std::vector<float>>> pos_embed_3d = get_2d_sincos_pos_embed_from_grid(embed_dim, grid_2d);

    int H = image_size.first;
    int W = image_size.second;
    std::vector<std::vector<float>> pos_embed_2d(H * W, std::vector<float>(embed_dim));
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            pos_embed_2d[w * H + h] = pos_embed_3d[h][w];
        }
    }

    return pos_embed_2d;
}

bool clip_image_encode(struct clip_ctx * ctx, const int n_threads, clip_image_f32 * img, float * vec) {
    clip_image_f32_batch imgs;
    clip_image_f32_ptr img_copy(clip_image_f32_init());
    *img_copy = *img;
    imgs.entries.push_back(std::move(img_copy));

    return clip_image_batch_encode(ctx, n_threads, &imgs, vec);
}

bool clip_image_batch_encode(clip_ctx * ctx, const int n_threads, const clip_image_f32_batch * imgs_c_ptr, float * vec) {
    const clip_image_f32_batch & imgs = *imgs_c_ptr;
    int batch_size = imgs.entries.size();
    ctx->last_mmproj_summary_json.clear();
    const bool collect_light_summary = clip_mmproj_light_summary_enabled();
    const int64_t summary_start_us = collect_light_summary ? ggml_time_us() : 0;
    int64_t build_graph_us = 0;
    int64_t alloc_graph_us = 0;
    int64_t graph_summary_us = 0;
    int64_t set_inputs_us = 0;
    int64_t graph_compute_us = 0;
    int64_t output_readback_us = 0;
    json graph_summary = nullptr;

    // TODO @ngxson : implement batch size > 1 as a loop
    //                we don't need true batching support because the cgraph will gonna be big anyway
    if (batch_size != 1) {
        return false; // only support batch size of 1
    }

    // build the inference graph
    ctx->debug_print_tensors.clear();
    ctx->debug_dump_w8a8_tensors.clear();
    ctx->profiler.reset();
    ggml_backend_sched_reset(ctx->sched.get());
    const int64_t build_graph_start_us = collect_light_summary ? ggml_time_us() : 0;
    ggml_cgraph * gf = clip_image_build_graph(ctx, imgs);
    if (collect_light_summary) {
        build_graph_us = ggml_time_us() - build_graph_start_us;
    }
    if (ctx->aicas_w8a8_debug) {
        LOG_INF("%s: built mmproj graph\n", __func__);
    }
    if (!ctx->debug_dump_dot_done && !ctx->debug_dump_dot_path.empty()) {
        ggml_graph_dump_dot(gf, nullptr, ctx->debug_dump_dot_path.c_str());
        LOG_INF("%s: dumped mmproj graph to %s\n", __func__, ctx->debug_dump_dot_path.c_str());
        ctx->debug_dump_dot_done = true;
    }
    ggml_backend_sched_set_eval_callback(
        ctx->sched.get(),
        (ctx->profiler.enabled || !ctx->debug_dump_w8a8_tensors_dir.empty()) ? clip_eval_callback : nullptr,
        (ctx->profiler.enabled || !ctx->debug_dump_w8a8_tensors_dir.empty()) ? ctx : nullptr);
    if (ctx->aicas_w8a8_debug) {
        LOG_INF("%s: allocating mmproj graph\n", __func__);
    }
    const int64_t alloc_graph_start_us = collect_light_summary ? ggml_time_us() : 0;
    if (!ggml_backend_sched_alloc_graph(ctx->sched.get(), gf)) {
        LOG_ERR("%s: ggml_backend_sched_alloc_graph failed\n", __func__);
        return false;
    }
    if (collect_light_summary) {
        alloc_graph_us = ggml_time_us() - alloc_graph_start_us;
        const int64_t graph_summary_start_us = ggml_time_us();
        graph_summary = clip_build_mmproj_graph_summary(ctx, gf);
        graph_summary_us = ggml_time_us() - graph_summary_start_us;
    }
    if (ctx->aicas_w8a8_debug) {
        LOG_INF("%s: allocated mmproj graph\n", __func__);
    }

    // set inputs
    const int64_t set_inputs_start_us = collect_light_summary ? ggml_time_us() : 0;
    const auto & model   = ctx->model;
    const auto & hparams = model.hparams;

    const int image_size_width  = imgs.entries[0]->nx;
    const int image_size_height = imgs.entries[0]->ny;

    const int patch_size    = hparams.patch_size;
    const int num_patches   = ((image_size_width / patch_size) * (image_size_height / patch_size));
    const int n_pos = num_patches + (model.class_embedding ? 1 : 0);
    const int pos_w = image_size_width  / patch_size;
    const int pos_h = image_size_height / patch_size;

    const bool use_window_attn = hparams.n_wa_pattern > 0; // for qwen2.5vl

    auto get_inp_tensor = [&gf](const char * name) {
        ggml_tensor * inp = ggml_graph_get_tensor(gf, name);
        if (inp == nullptr) {
            GGML_ABORT("Failed to get tensor %s", name);
        }
        if (!(inp->flags & GGML_TENSOR_FLAG_INPUT)) {
            GGML_ABORT("Tensor %s is not an input tensor", name);
        }
        return inp;
    };

    auto set_input_f32 = [&get_inp_tensor](const char * name, std::vector<float> & values) {
        ggml_tensor * cur = get_inp_tensor(name);
        GGML_ASSERT(cur->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_nelements(cur) == (int64_t)values.size());
        ggml_backend_tensor_set(cur, values.data(), 0, ggml_nbytes(cur));
    };

    auto set_input_i32 = [&get_inp_tensor](const char * name, std::vector<int32_t> & values) {
        ggml_tensor * cur = get_inp_tensor(name);
        GGML_ASSERT(cur->type == GGML_TYPE_I32);
        GGML_ASSERT(ggml_nelements(cur) == (int64_t)values.size());
        ggml_backend_tensor_set(cur, values.data(), 0, ggml_nbytes(cur));
    };

    // set input pixel values
    if (!imgs.is_audio) {
        size_t nelem = 0;
        for (const auto & img : imgs.entries) {
            nelem += img->nx * img->ny * 3;
        }
        std::vector<float> inp_raw(nelem);

        // layout of data (note: the channel dim is unrolled to better visualize the layout):
        //
        // ┌──W──┐
        // │     H │  channel = R
        // ├─────┤ │
        // │     H │  channel = G
        // ├─────┤ │
        // │     H │  channel = B
        // └─────┘ │
        //   ──────┘ x B

        for (size_t i = 0; i < imgs.entries.size(); i++) {
            const int nx = imgs.entries[i]->nx;
            const int ny = imgs.entries[i]->ny;
            const int n = nx * ny;

            for (int b = 0; b < batch_size; b++) {
                float * batch_entry = inp_raw.data() + b * (3*n);
                for (int y = 0; y < ny; y++) {
                    for (int x = 0; x < nx; x++) {
                        size_t base_src = 3*(y * nx + x); // idx of the first channel
                        size_t base_dst =    y * nx + x;  // idx of the first channel
                        batch_entry[      base_dst] = imgs.entries[b]->buf[base_src    ];
                        batch_entry[1*n + base_dst] = imgs.entries[b]->buf[base_src + 1];
                        batch_entry[2*n + base_dst] = imgs.entries[b]->buf[base_src + 2];
                    }
                }
            }
        }
        set_input_f32("inp_raw", inp_raw);

    } else {
        // audio input
        GGML_ASSERT(imgs.entries.size() == 1);
        const auto & mel_inp = imgs.entries[0];
        const int n_step = mel_inp->nx;
        const int n_mel  = mel_inp->ny;
        std::vector<float> inp_raw(n_step * n_mel);
        std::memcpy(inp_raw.data(), mel_inp->buf.data(), n_step * n_mel * sizeof(float));
        set_input_f32("inp_raw", inp_raw);
    }

    // set input per projector
    switch (ctx->model.proj_type) {
        case PROJECTOR_TYPE_MINICPMV:
            {
                // inspired from siglip:
                //    -> https://huggingface.co/HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit
                //    -> https://huggingface.co/HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit/blob/d66538faeba44480d0bfaa42145eef26f9423199/modeling_siglip.py#L316
                std::vector<int32_t> positions(pos_h * pos_w);
                int bucket_coords_h[1024];
                int bucket_coords_w[1024];
                for (int i = 0; i < pos_h; i++){
                    bucket_coords_h[i] = std::floor(70.0*i/pos_h);
                }
                for (int i = 0; i < pos_w; i++){
                    bucket_coords_w[i] = std::floor(70.0*i/pos_w);
                }
                for (int i = 0, id = 0; i < pos_h; i++){
                    for (int j = 0; j < pos_w; j++){
                        positions[id++] = bucket_coords_h[i]*70 + bucket_coords_w[j];
                    }
                }
                set_input_i32("positions", positions);

                // inspired from resampler of Qwen-VL:
                //    -> https://huggingface.co/Qwen/Qwen-VL/tree/main
                //    -> https://huggingface.co/Qwen/Qwen-VL/blob/0547ed36a86561e2e42fecec8fd0c4f6953e33c4/visual.py#L23
                int embed_dim = clip_n_mmproj_embd(ctx);

                // TODO @ngxson : this is very inefficient, can we do this using ggml_sin and ggml_cos?
                auto pos_embed_t = get_2d_sincos_pos_embed(embed_dim, std::make_pair(pos_w, pos_h));

                std::vector<float> pos_embed(embed_dim * pos_w * pos_h);
                for(int i = 0; i < pos_w * pos_h; ++i){
                    for(int j = 0; j < embed_dim; ++j){
                        pos_embed[i * embed_dim + j] = pos_embed_t[i][j];
                    }
                }

                set_input_f32("pos_embed", pos_embed);
            } break;
        case PROJECTOR_TYPE_QWEN2VL:
            {
                const int merge_ratio = 2;
                const int pw = image_size_width  / patch_size;
                const int ph = image_size_height / patch_size;
                std::vector<int> positions(n_pos * 4);
                int ptr = 0;
                for (int y = 0; y < ph; y += merge_ratio) {
                    for (int x = 0; x < pw; x += merge_ratio) {
                        for (int dy = 0; dy < 2; dy++) {
                            for (int dx = 0; dx < 2; dx++) {
                                positions[                  ptr] = y + dy;
                                positions[    num_patches + ptr] = x + dx;
                                positions[2 * num_patches + ptr] = y + dy;
                                positions[3 * num_patches + ptr] = x + dx;
                                ptr++;
                            }
                        }
                    }
                }

                set_input_i32("positions", positions);
            } break;
        case PROJECTOR_TYPE_QWEN25VL:
            {
                // pw * ph = number of tokens output by ViT after apply patch merger
                // ipw * ipw = number of vision token been processed inside ViT
                const int merge_ratio = 2;
                const int pw  = image_size_width  / patch_size / merge_ratio;
                const int ph  = image_size_height / patch_size / merge_ratio;
                const int ipw = image_size_width  / patch_size;
                const int iph = image_size_height / patch_size;

                std::vector<int> idx    (ph * pw);
                std::vector<int> inv_idx(ph * pw);

                if (use_window_attn) {
                    const int attn_window_size = 112;
                    const int grid_window = attn_window_size / patch_size / merge_ratio;
                    int dst = 0;
                    // [num_vision_tokens, num_vision_tokens] attention mask tensor
                    std::vector<float> mask(pow(ipw * iph, 2), std::numeric_limits<float>::lowest());
                    int mask_row = 0;

                    for (int y = 0; y < ph; y += grid_window) {
                        for (int x = 0; x < pw; x += grid_window) {
                            const int win_h = std::min(grid_window, ph - y);
                            const int win_w = std::min(grid_window, pw - x);
                            const int dst_0 = dst;
                            // group all tokens belong to the same window togather (to a continue range)
                            for (int dy = 0; dy < win_h; dy++) {
                                for (int dx = 0; dx < win_w; dx++) {
                                    const int src = (y + dy) * pw + (x + dx);
                                    GGML_ASSERT(src < (int)idx.size());
                                    GGML_ASSERT(dst < (int)inv_idx.size());
                                    idx    [src] = dst;
                                    inv_idx[dst] = src;
                                    dst++;
                                }
                            }

                            for (int r=0; r < win_h * win_w * merge_ratio * merge_ratio; r++) {
                                int row_offset = mask_row * (ipw * iph);
                                std::fill(
                                    mask.begin() + row_offset + (dst_0 * merge_ratio * merge_ratio),
                                    mask.begin() + row_offset + (dst   * merge_ratio * merge_ratio),
                                    0.0);
                                mask_row++;
                            }
                        }
                    }

                    set_input_i32("window_idx",     idx);
                    set_input_i32("inv_window_idx", inv_idx);
                    set_input_f32("window_mask",    mask);
                } else {
                    for (int i = 0; i < ph * pw; i++) {
                        idx[i] = i;
                    }
                }

                const int mpow = merge_ratio * merge_ratio;
                std::vector<int> positions(n_pos * 4);

                int ptr = 0;
                for (int y = 0; y < iph; y += merge_ratio) {
                    for (int x = 0; x < ipw; x += merge_ratio) {
                        for (int dy = 0; dy < 2; dy++) {
                            for (int dx = 0; dx < 2; dx++) {
                                auto remap = idx[ptr / mpow];
                                remap = (remap * mpow) + (ptr % mpow);

                                positions[                  remap] = y + dy;
                                positions[    num_patches + remap] = x + dx;
                                positions[2 * num_patches + remap] = y + dy;
                                positions[3 * num_patches + remap] = x + dx;
                                ptr++;
                            }
                        }
                    }
                }

                set_input_i32("positions", positions);
            } break;
        case PROJECTOR_TYPE_PIXTRAL:
        case PROJECTOR_TYPE_KIMIVL:
            {
                // set the 2D positions
                int n_patches_per_col = image_size_width / patch_size;
                std::vector<int> pos_data(n_pos);
                // dimension H
                for (int i = 0; i < n_pos; i++) {
                    pos_data[i] = i / n_patches_per_col;
                }
                set_input_i32("pos_h", pos_data);
                // dimension W
                for (int i = 0; i < n_pos; i++) {
                    pos_data[i] = i % n_patches_per_col;
                }
                set_input_i32("pos_w", pos_data);
            } break;
        case PROJECTOR_TYPE_GLM_EDGE:
        {
            // llava and other models
            std::vector<int32_t> positions(n_pos);
            for (int i = 0; i < n_pos; i++) {
                positions[i] = i;
            }
            set_input_i32("positions", positions);
        } break;
        case PROJECTOR_TYPE_MLP:
        case PROJECTOR_TYPE_MLP_NORM:
        case PROJECTOR_TYPE_LDP:
        case PROJECTOR_TYPE_LDPV2:
            {
                // llava and other models
                std::vector<int32_t> positions(n_pos);
                for (int i = 0; i < n_pos; i++) {
                    positions[i] = i;
                }
                set_input_i32("positions", positions);

                // The patches vector is used to get rows to index into the embeds with;
                // we should skip dim 0 only if we have CLS to avoid going out of bounds
                // when retrieving the rows.
                int patch_offset = model.class_embedding ? 1 : 0;
                std::vector<int32_t> patches(num_patches);
                for (int i = 0; i < num_patches; i++) {
                    patches[i] = i + patch_offset;
                }
                set_input_i32("patches", patches);
            } break;
        case PROJECTOR_TYPE_GEMMA3:
        case PROJECTOR_TYPE_IDEFICS3:
        case PROJECTOR_TYPE_INTERNVL:
        case PROJECTOR_TYPE_QWEN2A:
        case PROJECTOR_TYPE_ULTRAVOX:
        case PROJECTOR_TYPE_LFM2:
        case PROJECTOR_TYPE_VOXTRAL:
            {
                // do nothing
            } break;
        case PROJECTOR_TYPE_LLAMA4:
            {
                // set the 2D positions
                int n_patches_per_col = image_size_width / patch_size;
                std::vector<int> pos_data(num_patches + 1, 0); // +1 for the [CLS] token
                // last pos is always kept 0, it's for CLS
                // dimension H
                for (int i = 0; i < num_patches; i++) {
                    pos_data[i] = (i / n_patches_per_col) + 1;
                }
                set_input_i32("pos_h", pos_data);
                // dimension W
                for (int i = 0; i < num_patches; i++) {
                    pos_data[i] = (i % n_patches_per_col) + 1;
                }
                set_input_i32("pos_w", pos_data);
            } break;
        default:
            GGML_ABORT("Unknown projector type");
    }
    if (collect_light_summary) {
        set_inputs_us = ggml_time_us() - set_inputs_start_us;
    }

    // ggml_backend_cpu_set_n_threads(ctx->backend_cpu, n_threads);
    ggml_backend_dev_t dev = ggml_backend_get_device(ctx->backend_cpu);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    if (reg) {
        auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (ggml_backend_set_n_threads_fn) {
            ggml_backend_set_n_threads_fn(ctx->backend_cpu, n_threads);
        }
    }

    if (ctx->aicas_w8a8_debug) {
        LOG_INF("%s: computing mmproj graph\n", __func__);
    }
    const int64_t graph_compute_start_us = collect_light_summary ? ggml_time_us() : 0;
    auto status = ggml_backend_sched_graph_compute(ctx->sched.get(), gf);
    if (collect_light_summary) {
        graph_compute_us = ggml_time_us() - graph_compute_start_us;
    }
    if (status != GGML_STATUS_SUCCESS) {
        LOG_ERR("%s: ggml_backend_sched_graph_compute failed with error %d\n", __func__, status);
        return false;
    }
    if (ctx->aicas_w8a8_debug) {
        LOG_INF("%s: computed mmproj graph\n", __func__);
    }

    ctx->profiler.write_json();

    // print debug nodes
    if (ctx->debug_graph) {
        LOG_INF("\n\n---\n\n");
        LOG_INF("\n\nDebug graph:\n\n");
        for (ggml_tensor * t : ctx->debug_print_tensors) {
            std::vector<uint8_t> data(ggml_nbytes(t));
            ggml_backend_tensor_get(t, data.data(), 0, ggml_nbytes(t));
            print_tensor_shape(t);
            print_tensor_data(t, data.data(), 3);
        }
    }

    // the last node is the embedding tensor
    ggml_tensor * embeddings = ggml_graph_node(gf, -1);

    // sanity check (only support batch size of 1 for now)
    const int n_tokens_out = embeddings->ne[1];
    const int expected_n_tokens_out = clip_n_output_tokens(ctx, imgs.entries[0].get());
    if (n_tokens_out != expected_n_tokens_out) {
        LOG_ERR("%s: expected output %d tokens, got %d\n", __func__, expected_n_tokens_out, n_tokens_out);
        GGML_ABORT("Invalid number of output tokens");
    }

    // copy the embeddings to the location passed by the user
    const int64_t output_readback_start_us = collect_light_summary ? ggml_time_us() : 0;
    ggml_backend_tensor_get(embeddings, vec, 0, ggml_nbytes(embeddings));
    clip_dump_embeddings_if_requested(embeddings, vec);
    if (collect_light_summary) {
        output_readback_us = ggml_time_us() - output_readback_start_us;
        const int64_t total_observed_us = ggml_time_us() - summary_start_us;
        json summary = {
            {"profile_kind", "mmproj_encode_light_summary"},
            {"timing_unit", "us"},
            {"instrumentation", "phase_wall_time_plus_graph_assignment"},
            {"note", "Phase timings use wall clock around coarse mmproj steps. Graph categories are scheduled-node counts and bytes, not exact CPU operator runtime."},
            {"known_profile_overhead_us", graph_summary_us},
            {"phases", {
                {"build_graph_us", build_graph_us},
                {"alloc_graph_us", alloc_graph_us},
                {"set_inputs_us", set_inputs_us},
                {"graph_compute_us", graph_compute_us},
                {"output_readback_us", output_readback_us},
                {"total_observed_us", total_observed_us},
            }},
            {"graph", graph_summary},
        };
        ctx->last_mmproj_summary_json = summary.dump();
    }

    return true;
}

int clip_n_mmproj_embd(const struct clip_ctx * ctx) {
    switch (ctx->model.proj_type) {
        case PROJECTOR_TYPE_LDP:
            return ctx->model.mm_model_block_1_block_2_1_b->ne[0];
        case PROJECTOR_TYPE_LDPV2:
            return ctx->model.mm_model_peg_0_b->ne[0];
        case PROJECTOR_TYPE_MLP:
        case PROJECTOR_TYPE_PIXTRAL:
            return ctx->model.mm_2_w->ne[1];
        case PROJECTOR_TYPE_MLP_NORM:
            return ctx->model.mm_3_b->ne[0];
        case PROJECTOR_TYPE_MINICPMV:
            return ctx->model.mm_model_proj->ne[0];
        case PROJECTOR_TYPE_GLM_EDGE:
            return ctx->model.mm_model_mlp_3_w->ne[1];
        case PROJECTOR_TYPE_QWEN2VL:
        case PROJECTOR_TYPE_QWEN25VL:
            return ctx->model.mm_1_b->ne[0];
        case PROJECTOR_TYPE_GEMMA3:
            return ctx->model.mm_input_proj_w->ne[0];
        case PROJECTOR_TYPE_IDEFICS3:
            return ctx->model.projection->ne[1];
        case PROJECTOR_TYPE_ULTRAVOX:
        case PROJECTOR_TYPE_VOXTRAL:
            return ctx->model.mm_2_w->ne[1];
        case PROJECTOR_TYPE_INTERNVL:
            return ctx->model.mm_3_w->ne[1];
        case PROJECTOR_TYPE_LLAMA4:
            return ctx->model.mm_model_proj->ne[1];
        case PROJECTOR_TYPE_QWEN2A:
            return ctx->model.mm_fc_w->ne[1];
        case PROJECTOR_TYPE_LFM2:
        case PROJECTOR_TYPE_KIMIVL:
            return ctx->model.mm_2_w->ne[1];
        default:
            GGML_ABORT("Unknown projector type");
    }
}

int clip_is_minicpmv(const struct clip_ctx * ctx) {
    if (ctx->proj_type() == PROJECTOR_TYPE_MINICPMV) {
        return ctx->model.hparams.minicpmv_version;
    }
    return 0;
}

bool clip_is_glm(const struct clip_ctx * ctx) {
    return ctx->proj_type() == PROJECTOR_TYPE_GLM_EDGE;
}

bool clip_is_qwen2vl(const struct clip_ctx * ctx) {
    return ctx->proj_type() == PROJECTOR_TYPE_QWEN2VL
        || ctx->proj_type() == PROJECTOR_TYPE_QWEN25VL;
}

bool clip_is_llava(const struct clip_ctx * ctx) {
    return ctx->model.hparams.has_llava_projector;
}

bool clip_is_gemma3(const struct clip_ctx * ctx) {
    return ctx->proj_type() == PROJECTOR_TYPE_GEMMA3;
}

bool clip_has_vision_encoder(const struct clip_ctx * ctx) {
    return ctx->model.modality == CLIP_MODALITY_VISION;
}

bool clip_has_audio_encoder(const struct clip_ctx * ctx) {
    return ctx->model.modality == CLIP_MODALITY_AUDIO;
}

bool clip_has_whisper_encoder(const struct clip_ctx * ctx) {
    return ctx->proj_type() == PROJECTOR_TYPE_ULTRAVOX
        || ctx->proj_type() == PROJECTOR_TYPE_QWEN2A
        || ctx->proj_type() == PROJECTOR_TYPE_VOXTRAL;
}

const char * clip_last_mmproj_summary_json(const struct clip_ctx * ctx) {
    if (ctx == nullptr || ctx->last_mmproj_summary_json.empty()) {
        return nullptr;
    }
    return ctx->last_mmproj_summary_json.c_str();
}

bool clip_encode_float_image (struct clip_ctx * ctx, int n_threads, float * img, int h, int w, float * vec) {
    clip_image_f32 clip_img;
    clip_img.buf.resize(h * w * 3);
    for (int i = 0; i < h*w*3; i++)
    {
        clip_img.buf[i] = img[i];
    }
    clip_img.nx = w;
    clip_img.ny = h;
    clip_image_encode(ctx, n_threads, &clip_img, vec);
    return true;
}

//
// API used internally with mtmd
//

projector_type clip_get_projector_type(const struct clip_ctx * ctx) {
    return ctx->proj_type();
}

void clip_image_f32_batch_add_mel(struct clip_image_f32_batch * batch, int n_mel, int n_frames, float * mel) {
    clip_image_f32 * audio = new clip_image_f32;
    audio->nx = n_frames;
    audio->ny = n_mel;
    audio->buf.resize(n_frames * n_mel);
    std::memcpy(audio->buf.data(), mel, n_frames * n_mel * sizeof(float));

    batch->entries.push_back(clip_image_f32_ptr(audio));
    batch->is_audio = true;
}
