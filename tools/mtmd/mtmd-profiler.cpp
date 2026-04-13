#include "arg.h"
#include "chat.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "sampling.h"
#include "ggml.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <climits>
#include <cinttypes>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using json = nlohmann::ordered_json;

static constexpr const char * DEFAULT_THROUGHPUT_PROMPT =
    "Please analyze this image in detail.\n"
    "1.  First, please perform a full OCR (Optical Character Recognition), extract all visible text\n"
    "    in the image, and list it in order from top-to-bottom, left-to-right.\n"
    "2.  Second, please describe the main visual elements in the image, including but not limited to\n"
    "    objects, people, scenery, and atmosphere.\n"
    "3.  Finally, based on the extracted text and visual elements, summarize the theme\n"
    "    and possible context of this image.\n";

enum class profile_phase {
    none,
    prefill,
    decode,
};

struct profiler_cli_args {
    std::string profile_output = "llama_mtmd_profile.json";
    std::string profile_nodes_output;
    std::string profile_mmproj_output;
    std::string npu_runtime_profile_output;
    std::string npu_node_trace_output;
    std::string dump_prefill_dot;
    std::string dump_decode_dot;
    std::string dump_mmproj_dot;
    bool mmproj_only = false;
    std::vector<char *> forwarded_argv;
};

struct tensor_info {
    bool present = false;
    std::string name;
    std::string type;
    bool is_quantized = false;
    std::array<int64_t, GGML_MAX_DIMS> ne = {};
};

struct node_timing {
    std::string phase;
    std::string node_name;
    std::string op_name;
    std::string tensor_type;
    bool tensor_is_quantized = false;
    std::array<int64_t, GGML_MAX_DIMS> ne = {};
    std::array<tensor_info, GGML_MAX_SRC> srcs = {};
    double duration_us = 0.0;
    int64_t event_index = 0;
};

struct op_aggregate {
    double duration_us = 0.0;
    int64_t node_count = 0;
};

struct mul_mat_signature_aggregate {
    tensor_info src0;
    tensor_info src1;
    tensor_info dst;
    double duration_us = 0.0;
    int64_t node_count = 0;
    std::vector<std::string> example_node_names;
};

static tensor_info capture_tensor_info(const ggml_tensor * t);

struct operator_profiler {
    profile_phase phase = profile_phase::none;
    const ggml_tensor * pending_tensor = nullptr;
    int64_t pending_start_us = 0;
    int64_t next_event_index = 0;
    std::vector<node_timing> prefill_nodes;
    std::vector<node_timing> decode_nodes;

    static const char * phase_name(profile_phase phase) {
        switch (phase) {
            case profile_phase::prefill: return "prefill";
            case profile_phase::decode:  return "decode";
            case profile_phase::none:    return "none";
        }
        return "unknown";
    }

    void begin(profile_phase next_phase) {
        phase = next_phase;
    }

    void end(profile_phase current_phase) {
        GGML_ASSERT(phase == current_phase);
        GGML_ASSERT(pending_tensor == nullptr);
        phase = profile_phase::none;
    }

    std::vector<node_timing> & nodes_for_phase(profile_phase current_phase) {
        return current_phase == profile_phase::prefill ? prefill_nodes : decode_nodes;
    }

    static bool eval_callback(struct ggml_tensor * t, bool ask, void * user_data) {
        auto * profiler = static_cast<operator_profiler *>(user_data);

        if (profiler->phase == profile_phase::none) {
            return false;
        }

        if (ask) {
            profiler->pending_tensor = t;
            profiler->pending_start_us = ggml_time_us();
            return true;
        }

        const double duration_us = static_cast<double>(ggml_time_us() - profiler->pending_start_us);

        node_timing timing;
        timing.phase = phase_name(profiler->phase);
        timing.node_name = t->name;
        timing.op_name = ggml_op_desc(t);
        timing.tensor_type = ggml_type_name(t->type);
        timing.tensor_is_quantized = ggml_is_quantized(t->type);
        timing.duration_us = duration_us;
        timing.event_index = profiler->next_event_index++;
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            timing.ne[i] = t->ne[i];
        }
        for (int i = 0; i < GGML_MAX_SRC; ++i) {
            timing.srcs[i] = capture_tensor_info(t->src[i]);
        }

        profiler->nodes_for_phase(profiler->phase).push_back(std::move(timing));
        profiler->pending_tensor = nullptr;
        profiler->pending_start_us = 0;

        return true;
    }
};

struct generation_result {
    int status = 0;
    int generated_token_count = 0;
    int decode_call_count = 0;
    std::string text;
};

struct mtmd_profile_context {
    mtmd::context_ptr ctx_vision;
    common_init_result llama_init;

    llama_model       * model = nullptr;
    llama_context     * lctx = nullptr;
    const llama_vocab * vocab = nullptr;
    common_sampler    * smpl = nullptr;
    llama_batch         batch = {};
    int                 n_batch = 0;

    mtmd::bitmaps bitmaps;
    common_chat_templates_ptr tmpls;

    int n_threads = 1;
    llama_pos n_past = 0;

    explicit mtmd_profile_context(common_params & params, bool require_chat_template = true) : llama_init(common_init_from_params(params)) {
        model = llama_init.model.get();
        lctx = llama_init.context.get();
        vocab = llama_model_get_vocab(model);
        smpl = common_sampler_init(model, params.sampling);
        n_threads = params.cpuparams.n_threads;
        batch = llama_batch_init(1, 0, 1);
        n_batch = params.n_batch;

        if (model == nullptr || lctx == nullptr) {
            LOG_ERR("failed to initialize multimodal profiling context\n");
            std::exit(1);
        }

        if (require_chat_template && !llama_model_chat_template(model, nullptr) && params.chat_template.empty()) {
            LOG_ERR("Model does not have chat template. Set --chat-template explicitly if needed.\n");
            std::exit(1);
        }

        if (require_chat_template) {
            tmpls = common_chat_templates_init(model, params.chat_template);
        }
        init_vision_context(params);
    }

    ~mtmd_profile_context() {
        llama_batch_free(batch);
        common_sampler_free(smpl);
    }

    void init_vision_context(common_params & params) {
        mtmd_context_params mparams = mtmd_context_params_default();
        mparams.use_gpu = params.mmproj_use_gpu;
        mparams.print_timings = true;
        mparams.n_threads = params.cpuparams.n_threads;
        mparams.verbosity = params.verbosity > 0 ? GGML_LOG_LEVEL_DEBUG : GGML_LOG_LEVEL_INFO;

        ctx_vision.reset(mtmd_init_from_file(params.mmproj.path.c_str(), model, mparams));
        if (!ctx_vision.get()) {
            LOG_ERR("Failed to load vision model from %s\n", params.mmproj.path.c_str());
            std::exit(1);
        }
    }

    bool load_media(const std::string & fname) {
        mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(ctx_vision.get(), fname.c_str()));
        if (!bmp.ptr) {
            return false;
        }
        bitmaps.entries.push_back(std::move(bmp));
        return true;
    }
};

static void show_additional_info(int /*argc*/, char ** argv) {
    LOG(
        "Profile ggml operator latency for a single multimodal turn.\n\n"
        "Usage: %s [standard mtmd options] --profile-output <json>\n\n"
        "Custom options:\n"
        "  --mmproj-only               only run image encode / mmproj, skip text prefill+decode\n"
        "  --profile-output <path>        path to the aggregated operator profile JSON\n"
        "  --profile-nodes-output <path>  optional path to dump per-node timing samples\n\n"
        "  --profile-mmproj-output <path> path to dump mmproj/clip operator profile JSON\n"
        "  --npu-runtime-profile-output <path>\n"
        "                               path to dump NPU runtime stage profile JSON\n"
        "  --npu-node-trace-output <path> path to dump ggml-npu node/tile trace JSON\n"
        "  --dump-prefill-dot <path>      dump the first batched libllama ggml graph to a .dot file\n"
        "  --dump-decode-dot <path>       dump the first single-token libllama ggml graph to a .dot file\n"
        "  --dump-mmproj-dot <path>       dump the mmproj/clip image-encode ggml graph to a .dot file\n\n"
        "Relevant defaults matching KV260 throughput eval:\n"
        "  prompt      = built-in LONG_PROMPT from throughput_eval.py\n"
        "  temperature = 0.0\n"
        "  n_predict   = 4096\n"
        "  mode        = single-turn only\n",
        argv[0]
    );
}

static bool parse_profiler_cli_args(int argc, char ** argv, profiler_cli_args & args) {
    args.forwarded_argv.clear();
    args.forwarded_argv.push_back(argv[0]);

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--mmproj-only") {
            args.mmproj_only = true;
            continue;
        }

        if (arg == "--profile-output") {
            if (i + 1 >= argc) {
                LOG_ERR("missing value for %s\n", arg.c_str());
                return false;
            }
            args.profile_output = argv[++i];
            continue;
        }

        if (arg == "--profile-nodes-output") {
            if (i + 1 >= argc) {
                LOG_ERR("missing value for %s\n", arg.c_str());
                return false;
            }
            args.profile_nodes_output = argv[++i];
            continue;
        }

        if (arg == "--profile-mmproj-output") {
            if (i + 1 >= argc) {
                LOG_ERR("missing value for %s\n", arg.c_str());
                return false;
            }
            args.profile_mmproj_output = argv[++i];
            continue;
        }

        if (arg == "--npu-runtime-profile-output") {
            if (i + 1 >= argc) {
                LOG_ERR("missing value for %s\n", arg.c_str());
                return false;
            }
            args.npu_runtime_profile_output = argv[++i];
            continue;
        }

        if (arg == "--npu-node-trace-output") {
            if (i + 1 >= argc) {
                LOG_ERR("missing value for %s\n", arg.c_str());
                return false;
            }
            args.npu_node_trace_output = argv[++i];
            continue;
        }

        if (arg == "--dump-prefill-dot") {
            if (i + 1 >= argc) {
                LOG_ERR("missing value for %s\n", arg.c_str());
                return false;
            }
            args.dump_prefill_dot = argv[++i];
            continue;
        }

        if (arg == "--dump-decode-dot") {
            if (i + 1 >= argc) {
                LOG_ERR("missing value for %s\n", arg.c_str());
                return false;
            }
            args.dump_decode_dot = argv[++i];
            continue;
        }

        if (arg == "--dump-mmproj-dot") {
            if (i + 1 >= argc) {
                LOG_ERR("missing value for %s\n", arg.c_str());
                return false;
            }
            args.dump_mmproj_dot = argv[++i];
            continue;
        }

        args.forwarded_argv.push_back(argv[i]);
    }

    return true;
}

static void set_debug_env_if_needed(const char * key, const std::string & value) {
    if (value.empty()) {
        return;
    }

#if defined(_WIN32)
    _putenv_s(key, value.c_str());
#else
    setenv(key, value.c_str(), 1);
#endif
}

static std::string derive_manifest_path(const std::string & output_path) {
    if (output_path.empty()) {
        return "";
    }

    if (output_path.size() >= 5 && output_path.substr(output_path.size() - 5) == ".json") {
        return output_path.substr(0, output_path.size() - 5) + "_manifest.json";
    }

    return output_path + ".manifest.json";
}

static double pct(double numerator, double denominator) {
    if (denominator <= 0.0) {
        return 0.0;
    }
    return numerator / denominator * 100.0;
}

static tensor_info capture_tensor_info(const ggml_tensor * t) {
    tensor_info info;
    if (t == nullptr) {
        return info;
    }

    info.present = true;
    info.name = t->name;
    info.type = ggml_type_name(t->type);
    info.is_quantized = ggml_is_quantized(t->type);
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        info.ne[i] = t->ne[i];
    }

    return info;
}

static json shape_json(const std::array<int64_t, GGML_MAX_DIMS> & ne) {
    return {ne[0], ne[1], ne[2], ne[3]};
}

static std::string shape_key(const std::array<int64_t, GGML_MAX_DIMS> & ne) {
    return std::to_string(ne[0]) + "x" + std::to_string(ne[1]) + "x" + std::to_string(ne[2]) + "x" + std::to_string(ne[3]);
}

static json make_tensor_json(const tensor_info & info) {
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

static std::string mul_mat_signature_key(const node_timing & node) {
    GGML_ASSERT(node.srcs[0].present);
    GGML_ASSERT(node.srcs[1].present);

    return node.srcs[0].type + "|" + shape_key(node.srcs[0].ne) + "|" +
           node.srcs[1].type + "|" + shape_key(node.srcs[1].ne) + "|" +
           node.tensor_type + "|" + shape_key(node.ne);
}

static json summarize_mul_mat_signatures(const std::vector<node_timing> & nodes, double total_us) {
    std::unordered_map<std::string, mul_mat_signature_aggregate> signatures;

    for (const auto & node : nodes) {
        if (node.op_name != "MUL_MAT") {
            continue;
        }
        if (!node.srcs[0].present || !node.srcs[1].present) {
            continue;
        }

        auto & agg = signatures[mul_mat_signature_key(node)];
        if (agg.node_count == 0) {
            agg.src0 = node.srcs[0];
            agg.src1 = node.srcs[1];
            agg.dst = capture_tensor_info(nullptr);
            agg.dst.present = true;
            agg.dst.name = node.node_name;
            agg.dst.type = node.tensor_type;
            agg.dst.is_quantized = node.tensor_is_quantized;
            agg.dst.ne = node.ne;
        }

        agg.duration_us += node.duration_us;
        agg.node_count += 1;
        if (agg.example_node_names.size() < 3) {
            if (std::find(agg.example_node_names.begin(), agg.example_node_names.end(), node.node_name) == agg.example_node_names.end()) {
                agg.example_node_names.push_back(node.node_name);
            }
        }
    }

    std::vector<std::pair<std::string, mul_mat_signature_aggregate>> sorted(signatures.begin(), signatures.end());
    std::sort(sorted.begin(), sorted.end(), [](const auto & lhs, const auto & rhs) {
        if (lhs.second.duration_us != rhs.second.duration_us) {
            return lhs.second.duration_us > rhs.second.duration_us;
        }
        return lhs.first < rhs.first;
    });

    json out = json::array();
    for (const auto & [_, agg] : sorted) {
        const bool quantized_input_present = agg.src0.is_quantized || agg.src1.is_quantized;
        out.push_back({
            {"src0", make_tensor_json(agg.src0)},
            {"src1", make_tensor_json(agg.src1)},
            {"dst", make_tensor_json(agg.dst)},
            {"quantized_input_present", quantized_input_present},
            {"likely_weight_quantized", agg.src0.is_quantized},
            {"compute_path_hint", quantized_input_present
                ? "At least one MUL_MAT input is quantized. In llama.cpp this usually means quantized matmul with implicit dequantization inside the kernel, not a separate ggml DEQUANTIZE node."
                : "Inputs are floating-point, so this is a non-quantized MUL_MAT path."},
            {"duration_us", agg.duration_us},
            {"duration_ms", agg.duration_us / 1000.0},
            {"share_of_phase_time_pct", pct(agg.duration_us, total_us)},
            {"node_event_count", agg.node_count},
            {"average_duration_per_node_us", agg.node_count > 0 ? agg.duration_us / agg.node_count : 0.0},
            {"average_duration_per_node_ms", agg.node_count > 0 ? (agg.duration_us / 1000.0) / agg.node_count : 0.0},
            {"example_node_names", agg.example_node_names},
        });
    }

    return out;
}

static json make_node_json(const node_timing & node, double total_us) {
    json inputs = json::array();
    for (const auto & src : node.srcs) {
        if (src.present) {
            inputs.push_back(make_tensor_json(src));
        }
    }

    return {
        {"event_index", node.event_index},
        {"phase", node.phase},
        {"node_name", node.node_name},
        {"operator_name", node.op_name},
        {"output", {
            {"name", node.node_name},
            {"type", node.tensor_type},
            {"is_quantized", node.tensor_is_quantized},
            {"shape", shape_json(node.ne)},
        }},
        {"inputs", inputs},
        {"duration_us", node.duration_us},
        {"duration_ms", node.duration_us / 1000.0},
        {"share_of_phase_time_pct", pct(node.duration_us, total_us)},
    };
}

static json summarize_nodes(const std::vector<node_timing> & nodes, int top_k = 20) {
    std::unordered_map<std::string, op_aggregate> ops;
    double total_us = 0.0;
    for (const auto & node : nodes) {
        total_us += node.duration_us;
        auto & agg = ops[node.op_name];
        agg.duration_us += node.duration_us;
        agg.node_count += 1;
    }

    std::vector<std::pair<std::string, op_aggregate>> sorted_ops(ops.begin(), ops.end());
    std::sort(sorted_ops.begin(), sorted_ops.end(), [](const auto & lhs, const auto & rhs) {
        if (lhs.second.duration_us != rhs.second.duration_us) {
            return lhs.second.duration_us > rhs.second.duration_us;
        }
        return lhs.first < rhs.first;
    });

    std::vector<node_timing> sorted_nodes = nodes;
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
            {"share_of_phase_time_pct", pct(agg.duration_us, total_us)},
            {"node_event_count", agg.node_count},
            {"average_duration_per_node_us", agg.node_count > 0 ? agg.duration_us / agg.node_count : 0.0},
            {"average_duration_per_node_ms", agg.node_count > 0 ? (agg.duration_us / 1000.0) / agg.node_count : 0.0},
        });
    }

    json top_nodes_json = json::array();
    for (int i = 0; i < static_cast<int>(sorted_nodes.size()) && i < top_k; ++i) {
        top_nodes_json.push_back(make_node_json(sorted_nodes[i], total_us));
    }

    return {
        {"node_event_count", nodes.size()},
        {"total_us", total_us},
        {"total_ms", total_us / 1000.0},
        {"operators", operators},
        {"mul_mat_signatures", summarize_mul_mat_signatures(nodes, total_us)},
        {"top_nodes", top_nodes_json},
    };
}

static json build_profile_report(
        const operator_profiler & profiler,
        int generated_token_count,
        int decode_call_count,
        const common_params & params) {
    std::vector<node_timing> combined_nodes = profiler.prefill_nodes;
    combined_nodes.insert(combined_nodes.end(), profiler.decode_nodes.begin(), profiler.decode_nodes.end());

    json prefill = summarize_nodes(profiler.prefill_nodes);
    json decode = summarize_nodes(profiler.decode_nodes);
    json combined = summarize_nodes(combined_nodes);

    const double decode_total_us = decode.value("total_us", 0.0);
    decode["decode_step_count"] = decode_call_count;
    decode["average_phase_time_per_step_us"] = decode_call_count > 0 ? decode_total_us / decode_call_count : 0.0;
    decode["average_phase_time_per_step_ms"] = decode_call_count > 0 ? (decode_total_us / 1000.0) / decode_call_count : 0.0;

    return {
        {"profile_kind", "ggml_operator_latency_share"},
        {"timing_unit", "us"},
        {"generated_token_count", generated_token_count},
        {"decode_step_count", decode_call_count},
        {"note",
            "Per-node wall-clock timings captured through ggml eval callback. "
            "Profiling forces node-by-node synchronized execution, so totals are for profiled replay "
            "and should not be compared directly to end-to-end throughput numbers."},
        {"model", params.model.path},
        {"mmproj", params.mmproj.path},
        {"images", params.image},
        {"prompt", params.prompt},
        {"threads", params.cpuparams.n_threads},
        {"n_predict", params.n_predict},
        {"temperature", params.sampling.temp},
        {"prefill", prefill},
        {"decode", decode},
        {"combined", combined},
    };
}

static const char * chunk_type_name(enum mtmd_input_chunk_type type) {
    switch (type) {
        case MTMD_INPUT_CHUNK_TYPE_TEXT:  return "text";
        case MTMD_INPUT_CHUNK_TYPE_IMAGE: return "image";
        case MTMD_INPUT_CHUNK_TYPE_AUDIO: return "audio";
    }

    return "unknown";
}

static std::string make_mmproj_only_prompt(size_t n_media) {
    std::string prompt;
    for (size_t i = 0; i < n_media; ++i) {
        if (!prompt.empty()) {
            prompt += '\n';
        }
        prompt += mtmd_default_marker();
    }
    return prompt;
}

static bool run_mmproj_only(
        mtmd_profile_context & ctx,
        const common_params & params,
        json * out_summary) {
    std::string prompt = make_mmproj_only_prompt(ctx.bitmaps.entries.size());
    mtmd_input_text text = {
        prompt.c_str(),
        false,
        true,
    };

    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    auto bitmaps_c_ptr = ctx.bitmaps.c_ptr();
    const int32_t tokenize_res = mtmd_tokenize(
        ctx.ctx_vision.get(),
        chunks.ptr.get(),
        &text,
        bitmaps_c_ptr.data(),
        bitmaps_c_ptr.size());
    if (tokenize_res != 0) {
        LOG_ERR("Unable to tokenize mmproj-only prompt, res = %d\n", tokenize_res);
        return false;
    }

    ctx.bitmaps.entries.clear();

    json chunk_reports = json::array();
    size_t image_chunk_count = 0;
    size_t audio_chunk_count = 0;
    size_t total_image_tokens = 0;
    size_t total_audio_tokens = 0;
    size_t total_positions = 0;

    for (size_t i = 0; i < mtmd_input_chunks_size(chunks.ptr.get()); ++i) {
        const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks.ptr.get(), i);
        const enum mtmd_input_chunk_type type = mtmd_input_chunk_get_type(chunk);
        const size_t n_tokens = mtmd_input_chunk_get_n_tokens(chunk);
        const llama_pos n_pos = mtmd_input_chunk_get_n_pos(chunk);
        const char * chunk_id = mtmd_input_chunk_get_id(chunk);

        json chunk_report = {
            {"chunk_index", i},
            {"chunk_type", chunk_type_name(type)},
            {"chunk_id", chunk_id ? json(chunk_id) : json(nullptr)},
            {"n_tokens", n_tokens},
            {"n_positions", n_pos},
        };

        if (type == MTMD_INPUT_CHUNK_TYPE_IMAGE || type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
            if (mtmd_encode_chunk(ctx.ctx_vision.get(), chunk) != 0) {
                LOG_ERR("Unable to encode chunk %zu in mmproj-only mode\n", i);
                return false;
            }
        }

        if (type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            image_chunk_count += 1;
            total_image_tokens += n_tokens;
            total_positions += static_cast<size_t>(n_pos);
        } else if (type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
            audio_chunk_count += 1;
            total_audio_tokens += n_tokens;
            total_positions += static_cast<size_t>(n_pos);
        }

        chunk_reports.push_back(std::move(chunk_report));
    }

    *out_summary = {
        {"profile_kind", "mmproj_only_run"},
        {"mode", "mmproj_only"},
        {"model", params.model.path},
        {"mmproj", params.mmproj.path},
        {"images", params.image},
        {"threads", params.cpuparams.n_threads},
        {"image_chunk_count", image_chunk_count},
        {"audio_chunk_count", audio_chunk_count},
        {"total_image_tokens", total_image_tokens},
        {"total_audio_tokens", total_audio_tokens},
        {"total_positions", total_positions},
        {"chunks", chunk_reports},
    };
    return true;
}

static int eval_message(mtmd_profile_context & ctx, const std::string & prompt, bool add_bos) {
    common_chat_msg msg;
    msg.role = "user";
    msg.content = prompt;

    common_chat_templates_inputs tmpl_inputs;
    tmpl_inputs.messages = {msg};
    tmpl_inputs.add_generation_prompt = true;
    tmpl_inputs.use_jinja = false;

    const auto formatted_chat = common_chat_templates_apply(ctx.tmpls.get(), tmpl_inputs);

    mtmd_input_text text;
    text.text = formatted_chat.prompt.c_str();
    text.add_special = add_bos;
    text.parse_special = true;

    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    auto bitmaps_c_ptr = ctx.bitmaps.c_ptr();
    const int32_t res = mtmd_tokenize(
        ctx.ctx_vision.get(),
        chunks.ptr.get(),
        &text,
        bitmaps_c_ptr.data(),
        bitmaps_c_ptr.size());
    if (res != 0) {
        LOG_ERR("Unable to tokenize prompt, res = %d\n", res);
        return 1;
    }

    ctx.bitmaps.entries.clear();

    llama_pos new_n_past = 0;
    if (mtmd_helper_eval_chunks(
            ctx.ctx_vision.get(),
            ctx.lctx,
            chunks.ptr.get(),
            ctx.n_past,
            0,
            ctx.n_batch,
            true,
            &new_n_past)) {
        LOG_ERR("Unable to eval prompt\n");
        return 1;
    }

    ctx.n_past = new_n_past;
    return 0;
}

static generation_result generate_response(mtmd_profile_context & ctx, int n_predict) {
    generation_result result;
    std::vector<llama_token> generated_tokens;

    for (int i = 0; i < n_predict; ++i) {
        const llama_token token_id = common_sampler_sample(ctx.smpl, ctx.lctx, -1);
        generated_tokens.push_back(token_id);
        common_sampler_accept(ctx.smpl, token_id, true);

        if (llama_vocab_is_eog(ctx.vocab, token_id)) {
            break;
        }

        result.text += common_token_to_piece(ctx.lctx, token_id);

        common_batch_clear(ctx.batch);
        common_batch_add(ctx.batch, token_id, ctx.n_past++, {0}, true);

        if (llama_decode(ctx.lctx, ctx.batch)) {
            LOG_ERR("failed to decode token\n");
            result.status = 1;
            result.generated_token_count = generated_tokens.size();
            result.decode_call_count = result.decode_call_count;
            return result;
        }

        result.decode_call_count += 1;
    }

    result.generated_token_count = generated_tokens.size();
    return result;
}

static bool write_json_file(const std::string & path, const json & data) {
    std::ofstream out(path);
    if (!out.is_open()) {
        LOG_ERR("failed to open %s for writing\n", path.c_str());
        return false;
    }
    out << data.dump(2) << '\n';
    return true;
}

int main(int argc, char ** argv) {
    ggml_time_init();

    profiler_cli_args cli_args;
    if (!parse_profiler_cli_args(argc, argv, cli_args)) {
        return 1;
    }

    common_params params;
    params.sampling.temp = 0.0f;
    params.n_predict = 4096;

    set_debug_env_if_needed("LLAMA_DUMP_PREFILL_DOT", cli_args.dump_prefill_dot);
    set_debug_env_if_needed("LLAMA_DUMP_DECODE_DOT", cli_args.dump_decode_dot);
    set_debug_env_if_needed("MTMD_DUMP_DOT", cli_args.dump_mmproj_dot);
    set_debug_env_if_needed("MTMD_PROFILE_MMPROJ_JSON", cli_args.profile_mmproj_output);
    set_debug_env_if_needed("NPU_PROFILE_OUT", cli_args.npu_runtime_profile_output);
    set_debug_env_if_needed("GGML_NPU_PROFILE_JSON", cli_args.npu_node_trace_output);
    if (!cli_args.npu_node_trace_output.empty()) {
        set_debug_env_if_needed("NPU_PROFILE_MANIFEST", derive_manifest_path(cli_args.npu_node_trace_output));
    }

    int filtered_argc = static_cast<int>(cli_args.forwarded_argv.size());
    if (!common_params_parse(filtered_argc, cli_args.forwarded_argv.data(), params, LLAMA_EXAMPLE_MTMD, show_additional_info)) {
        return 1;
    }

    common_init();

    if (params.mmproj.path.empty()) {
        LOG_ERR("ERR: Missing --mmproj argument\n");
        return 1;
    }

    if (params.image.empty()) {
        LOG_ERR("ERR: At least one --image is required for multimodal profiling\n");
        return 1;
    }

    if (!cli_args.mmproj_only) {
        if (params.prompt.empty()) {
            params.prompt = DEFAULT_THROUGHPUT_PROMPT;
        }

        if (params.prompt.find(mtmd_default_marker()) == std::string::npos) {
            for (size_t i = 0; i < params.image.size(); ++i) {
                params.prompt += mtmd_default_marker();
            }
        }
    }

    operator_profiler profiler;
    if (!cli_args.mmproj_only) {
        params.cb_eval = operator_profiler::eval_callback;
        params.cb_eval_user_data = &profiler;
    }
    params.warmup = false;

    mtmd_profile_context ctx(params, !cli_args.mmproj_only);
    LOG("%s: loading model: %s\n", __func__, params.model.path.c_str());

    for (const auto & image : params.image) {
        if (!ctx.load_media(image)) {
            return 1;
        }
    }

    if (cli_args.mmproj_only) {
        json report;
        if (!run_mmproj_only(ctx, params, &report)) {
            return 1;
        }

        report["raw_artifacts"] = {
            {"mmproj_operator_profile", cli_args.profile_mmproj_output.empty() ? json(nullptr) : json(cli_args.profile_mmproj_output)},
            {"npu_runtime_profile", cli_args.npu_runtime_profile_output.empty() ? json(nullptr) : json(cli_args.npu_runtime_profile_output)},
            {"npu_node_trace", cli_args.npu_node_trace_output.empty() ? json(nullptr) : json(cli_args.npu_node_trace_output)},
            {"npu_manifest", cli_args.npu_node_trace_output.empty() ? json(nullptr) : json(derive_manifest_path(cli_args.npu_node_trace_output))},
        };
        report["note"] =
            "This run only executes image encode / mmproj. "
            "Detailed operator and NPU stage breakdowns are written to the raw artifact JSON files.";

        if (!write_json_file(cli_args.profile_output, report)) {
            return 1;
        }

        LOG("\nSaved mmproj-only run metadata to %s\n", cli_args.profile_output.c_str());
        if (!cli_args.profile_mmproj_output.empty()) {
            LOG("Saved mmproj operator profile to %s\n", cli_args.profile_mmproj_output.c_str());
        }
        if (!cli_args.npu_runtime_profile_output.empty()) {
            LOG("Saved NPU runtime profile to %s\n", cli_args.npu_runtime_profile_output.c_str());
        }
        if (!cli_args.npu_node_trace_output.empty()) {
            LOG("Saved ggml-npu node trace to %s\n", cli_args.npu_node_trace_output.c_str());
        }
        return 0;
    }

    profiler.begin(profile_phase::prefill);
    if (eval_message(ctx, params.prompt, true)) {
        return 1;
    }
    profiler.end(profile_phase::prefill);

    const int n_predict = params.n_predict < 0 ? INT_MAX : params.n_predict;
    profiler.begin(profile_phase::decode);
    const generation_result generation = generate_response(ctx, n_predict);
    profiler.end(profile_phase::decode);
    if (generation.status != 0) {
        return generation.status;
    }

    const json report = build_profile_report(
        profiler,
        generation.generated_token_count,
        generation.decode_call_count,
        params);

    if (!write_json_file(cli_args.profile_output, report)) {
        return 1;
    }

    if (!cli_args.profile_nodes_output.empty()) {
        json nodes = json::array();
        for (const auto & node : profiler.prefill_nodes) {
            nodes.push_back(make_node_json(node, report["prefill"]["total_us"]));
        }
        for (const auto & node : profiler.decode_nodes) {
            nodes.push_back(make_node_json(node, report["decode"]["total_us"]));
        }
        if (!write_json_file(cli_args.profile_nodes_output, nodes)) {
            return 1;
        }
    }

    LOG("%s\n", generation.text.c_str());
    LOG("\nSaved operator profile to %s\n", cli_args.profile_output.c_str());
    if (!cli_args.profile_nodes_output.empty()) {
        LOG("Saved per-node samples to %s\n", cli_args.profile_nodes_output.c_str());
    }
    LOG("\n");
    llama_perf_context_print(ctx.lctx);

    return 0;
}
