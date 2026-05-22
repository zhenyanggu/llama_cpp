#pragma once

#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml-cpu.h"
#include "arg.h" // common_remote_get_content
#include "base64.hpp"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "chat.h"
#ifdef GGML_USE_NPU
#include "ggml/src/ggml-npu/ggml-npu.h"
#endif

// increase max payload length to allow use of larger context size
#define CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH 1048576
// increase backlog size to avoid connection resets for >> 1 slots
#define CPPHTTPLIB_LISTEN_BACKLOG 512
// disable Nagle's algorithm
#define CPPHTTPLIB_TCP_NODELAY true
#include <cpp-httplib/httplib.h>

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

#include <random>
#include <algorithm>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <cinttypes>
#include <fstream>
#include <cstdlib>
#include <cstring>

#define DEFAULT_OAICOMPAT_MODEL "gpt-3.5-turbo"

using json = nlohmann::ordered_json;

#define SLT_INF(slot, fmt, ...) LOG_INF("slot %12.*s: id %2d | task %d | " fmt, 12, __func__, (slot).id, ((slot).task ? (slot).task->id : -1), __VA_ARGS__)
#define SLT_WRN(slot, fmt, ...) LOG_WRN("slot %12.*s: id %2d | task %d | " fmt, 12, __func__, (slot).id, ((slot).task ? (slot).task->id : -1), __VA_ARGS__)
#define SLT_ERR(slot, fmt, ...) LOG_ERR("slot %12.*s: id %2d | task %d | " fmt, 12, __func__, (slot).id, ((slot).task ? (slot).task->id : -1), __VA_ARGS__)
#define SLT_DBG(slot, fmt, ...) LOG_DBG("slot %12.*s: id %2d | task %d | " fmt, 12, __func__, (slot).id, ((slot).task ? (slot).task->id : -1), __VA_ARGS__)

#define SRV_INF(fmt, ...) LOG_INF("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SRV_WRN(fmt, ...) LOG_WRN("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SRV_ERR(fmt, ...) LOG_ERR("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SRV_DBG(fmt, ...) LOG_DBG("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)

#define QUE_INF(fmt, ...) LOG_INF("que  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define QUE_WRN(fmt, ...) LOG_WRN("que  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define QUE_ERR(fmt, ...) LOG_ERR("que  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define QUE_DBG(fmt, ...) LOG_DBG("que  %12.*s: " fmt, 12, __func__, __VA_ARGS__)

using raw_buffer = std::vector<uint8_t>;

static std::string server_mtmd_profile_output_path() {
    const char * path = std::getenv("LLAMA_MTMD_PREFILL_SUMMARY_JSON");
    return path ? path : "";
}

static std::string server_npu_overlay_profile_output_path() {
    const char * path = std::getenv("AICAS_NPU_OVERLAY_PROFILE_JSONL");
    return path ? path : "";
}

static void server_npu_overlay_profile_write(
        const char * phase,
        const char * env_name,
        int rc,
        int64_t cleanup_us,
        int64_t switch_us,
        int64_t total_us) {
    const std::string path = server_npu_overlay_profile_output_path();
    if (path.empty()) {
        return;
    }

    static std::mutex mutex;
    json payload;
    payload["profile_kind"] = "aicas_npu_overlay_switch";
    payload["phase"] = phase != nullptr ? phase : "";
    payload["env_name"] = env_name != nullptr ? env_name : "";
    payload["rc"] = rc;
    payload["time_us"] = {
        {"cleanup_runtime", cleanup_us},
        {"xmutil_switch", switch_us},
        {"total", total_us},
    };

    std::lock_guard<std::mutex> lock(mutex);
    std::ofstream out(path, std::ios::app);
    if (out.good()) {
        out << payload.dump() << '\n';
    }
}

static bool server_run_npu_overlay_switch_cmd(const char * env_name, const char * phase) {
#ifdef GGML_USE_NPU
	    const char * cmd = std::getenv(env_name);
	    if (cmd == nullptr || cmd[0] == '\0') {
	        return true;
	    }

	    static std::mutex overlay_phase_mutex;
	    static std::string active_phase;
	    const std::string requested_phase = phase != nullptr ? phase : "";
	    {
	        std::lock_guard<std::mutex> lock(overlay_phase_mutex);
	        if (active_phase == requested_phase) {
	            return true;
	        }
	    }

	    SRV_INF("switching NPU overlay for %s\n", phase);
	    const int64_t total_start_us = ggml_time_us();
	    const int64_t cleanup_start_us = total_start_us;
    ggml_backend_npu_w8a8_preload_clear();
    ggml_backend_npu_runtime_shutdown();
    ggml_backend_npu_decode_runtime_shutdown();
    if (std::strcmp(phase, "prefill") == 0) {
        ggml_backend_npu_decode_overlay_mark_inactive();
    }
    const int64_t cleanup_us = ggml_time_us() - cleanup_start_us;
    const int64_t switch_start_us = ggml_time_us();
    const int rc = std::system(cmd);
    const int64_t switch_us = ggml_time_us() - switch_start_us;
    const int64_t total_us = ggml_time_us() - total_start_us;
    server_npu_overlay_profile_write(phase, env_name, rc, cleanup_us, switch_us, total_us);
    if (rc != 0) {
        SRV_ERR("NPU overlay switch command failed for %s, rc = %d\n", phase, rc);
        return false;
    }
	    if (std::strcmp(phase, "decode") == 0) {
	        ggml_backend_npu_decode_overlay_mark_active();
	    }
	    {
	        std::lock_guard<std::mutex> lock(overlay_phase_mutex);
	        active_phase = requested_phase;
	    }
	    return true;
#else
    GGML_UNUSED(env_name);
    GGML_UNUSED(phase);
    return true;
#endif
}

static bool server_mtmd_profile_enabled() {
    const std::string path = server_mtmd_profile_output_path();
    return !path.empty();
}

static bool server_mtmd_merge_prefill_enabled() {
    const char * value = std::getenv("LLAMA_MTMD_MERGE_PREFILL");
    return value && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

static bool server_mtmd_cpu_op_profile_enabled() {
    const char * env = std::getenv("LLAMA_MTMD_CPU_OP_PROFILE");
    return env != nullptr && env[0] != '\0' && std::string(env) != "0";
}

static std::string server_text_cpu_profile_output_path() {
    const char * path = std::getenv("LLAMA_TEXT_CPU_PROFILE_JSON");
    return path ? path : "";
}

static bool server_text_cpu_profile_enabled() {
    const std::string path = server_text_cpu_profile_output_path();
    return !path.empty();
}

static void server_text_cpu_profile_write(
        const char * phase,
        int32_t n_tokens,
        int32_t seq_id,
        int64_t wall_us,
        const char * cpu_backend_profile_json) {
    const std::string path = server_text_cpu_profile_output_path();
    if (path.empty() || cpu_backend_profile_json == nullptr || cpu_backend_profile_json[0] == '\0') {
        return;
    }

    json profile = json::parse(cpu_backend_profile_json, nullptr, false);
    if (profile.is_discarded()) {
        return;
    }

    json record = {
        {"profile_kind", "llama_server_text_cpu_profile_record"},
        {"timing_unit", "us"},
        {"phase", phase ? phase : "unknown"},
        {"n_tokens", n_tokens},
        {"seq_id", seq_id},
        {"wall_us", wall_us},
        {"cpu_backend_profile", std::move(profile)},
    };

    std::ofstream fout(path, std::ios::app | std::ios::binary);
    if (!fout.is_open()) {
        return;
    }
    fout << record.dump() << "\n";
}

struct server_mtmd_prefill_profile {
    struct mmproj_category_aggregate {
        int64_t node_count = 0;
        int64_t duration_us = 0;
        int64_t elements = 0;
        int64_t bytes = 0;
    };

    bool enabled = false;
    bool has_media = false;
    int64_t media_chunk_count = 0;
    int64_t image_chunk_count = 0;
    int64_t audio_chunk_count = 0;
    int64_t media_tokens = 0;
    int64_t media_positions = 0;
    int64_t mmproj_encode_us = 0;
    int64_t media_decode_us = 0;
    int64_t media_process_us = 0;
    bool merged_prefill = false;
    int64_t merged_tokens = 0;
    int64_t text_tokens = 0;
    int64_t merged_decode_us = 0;
    std::vector<json> mmproj_encode_details;
    std::vector<json> cpu_backend_profile_details;
#ifdef GGML_USE_NPU
    ggml_npu_profile_summary npu = {};
    ggml_npu_profile_summary text_prefill_npu = {};
#endif

    void reset(bool enable) {
        enabled = enable;
        has_media = false;
        media_chunk_count = 0;
        image_chunk_count = 0;
        audio_chunk_count = 0;
        media_tokens = 0;
        media_positions = 0;
        mmproj_encode_us = 0;
        media_decode_us = 0;
        media_process_us = 0;
        merged_prefill = false;
        merged_tokens = 0;
        text_tokens = 0;
        merged_decode_us = 0;
        mmproj_encode_details.clear();
        cpu_backend_profile_details.clear();
#ifdef GGML_USE_NPU
        npu = {};
        text_prefill_npu = {};
#endif
    }

#ifdef GGML_USE_NPU
    static void add_npu_summary(ggml_npu_profile_summary & dst, const ggml_npu_profile_summary & src) {
        dst.used_npu |= src.used_npu;
        dst.node_count += src.node_count;
        dst.exec_tile_count += src.exec_tile_count;
        dst.weight_pack_count += src.weight_pack_count;
        dst.bias_pack_count += src.bias_pack_count;
        dst.activation_pack_calls += src.activation_pack_calls;
        dst.host_copy_activation_calls += src.host_copy_activation_calls;
        dst.host_copy_weight_calls += src.host_copy_weight_calls;
        dst.bias_prepare_calls += src.bias_prepare_calls;
        dst.dma_in_activation_calls += src.dma_in_activation_calls;
        dst.dma_in_weight_calls += src.dma_in_weight_calls;
        dst.dma_in_bias_calls += src.dma_in_bias_calls;
        dst.dma_in_pair_calls += src.dma_in_pair_calls;
        dst.spm_activation_reuse_hits += src.spm_activation_reuse_hits;
        dst.spm_weight_reuse_hits += src.spm_weight_reuse_hits;
        dst.gemm_calls += src.gemm_calls;
        dst.gemm_plan_calls += src.gemm_plan_calls;
        dst.dma_out_calls += src.dma_out_calls;
        dst.postprocess_calls += src.postprocess_calls;
        dst.raw_acc_mvout_nodes += src.raw_acc_mvout_nodes;
        dst.raw_acc_mvout_tiles += src.raw_acc_mvout_tiles;
        dst.w_prefetch_calls += src.w_prefetch_calls;
        dst.w_prefetch_hits += src.w_prefetch_hits;
        dst.w_prefetch_conflicts += src.w_prefetch_conflicts;
        dst.packed_activation_bytes_total += src.packed_activation_bytes_total;
        dst.copied_weight_bytes_total += src.copied_weight_bytes_total;
        dst.dma_in_activation_bytes_total += src.dma_in_activation_bytes_total;
        dst.dma_in_weight_bytes_total += src.dma_in_weight_bytes_total;
        dst.bias_bytes_total += src.bias_bytes_total;
        dst.acc_readback_bytes_total += src.acc_readback_bytes_total;
        dst.output_write_bytes_total += src.output_write_bytes_total;
        dst.total_node_us += src.total_node_us;
        dst.activation_pack_us_total += src.activation_pack_us_total;
        dst.host_copy_activation_us_total += src.host_copy_activation_us_total;
        dst.host_copy_weight_us_total += src.host_copy_weight_us_total;
        dst.bias_prepare_us_total += src.bias_prepare_us_total;
        dst.dma_in_activation_us_total += src.dma_in_activation_us_total;
        dst.dma_in_weight_us_total += src.dma_in_weight_us_total;
        dst.dma_in_bias_us_total += src.dma_in_bias_us_total;
        dst.dma_in_pair_us_total += src.dma_in_pair_us_total;
        dst.w_prefetch_wait_us_total += src.w_prefetch_wait_us_total;
        dst.w_prefetch_hidden_candidate_us_total += src.w_prefetch_hidden_candidate_us_total;
        dst.gemm_us_total += src.gemm_us_total;
        dst.dma_out_us_total += src.dma_out_us_total;
        dst.postprocess_us_total += src.postprocess_us_total;
        dst.accounted_us_total += src.accounted_us_total;
        dst.unaccounted_us_total += src.unaccounted_us_total;
        dst.runtime_total_us += src.runtime_total_us;
        dst.runtime_dma_in_us += src.runtime_dma_in_us;
        dst.runtime_compute_us += src.runtime_compute_us;
        dst.runtime_compute_exclusive_us += src.runtime_compute_exclusive_us;
        dst.runtime_dma_out_us += src.runtime_dma_out_us;
        dst.runtime_layout_us += src.runtime_layout_us;
        dst.runtime_layout_exclusive_us += src.runtime_layout_exclusive_us;
        dst.runtime_wait_irq_us += src.runtime_wait_irq_us;
        dst.runtime_mvin_calls += src.runtime_mvin_calls;
        dst.runtime_compute_calls += src.runtime_compute_calls;
        dst.runtime_gemm_plan_calls += src.runtime_gemm_plan_calls;
        dst.runtime_mvout_calls += src.runtime_mvout_calls;
        dst.runtime_layout_calls += src.runtime_layout_calls;
    }

    static json npu_summary_to_json(const ggml_npu_profile_summary & summary) {
        return {
            {"used_npu", summary.used_npu != 0},
            {"node_count", summary.node_count},
            {"exec_tile_count", summary.exec_tile_count},
            {"weight_pack_count", summary.weight_pack_count},
            {"bias_pack_count", summary.bias_pack_count},
            {"gemm_calls", summary.gemm_calls},
            {"gemm_plan_calls", summary.gemm_plan_calls},
            {"dma_out_calls", summary.dma_out_calls},
            {"postprocess_calls", summary.postprocess_calls},
            {"dma_in_activation_calls", summary.dma_in_activation_calls},
            {"dma_in_weight_calls", summary.dma_in_weight_calls},
            {"dma_in_bias_calls", summary.dma_in_bias_calls},
            {"dma_in_pair_calls", summary.dma_in_pair_calls},
            {"spm_activation_reuse_hits", summary.spm_activation_reuse_hits},
            {"spm_weight_reuse_hits", summary.spm_weight_reuse_hits},
            {"dma_in_activation_bytes", summary.dma_in_activation_bytes_total},
            {"dma_in_weight_bytes", summary.dma_in_weight_bytes_total},
            {"acc_readback_bytes", summary.acc_readback_bytes_total},
            {"output_write_bytes", summary.output_write_bytes_total},
            {"w_pingpong_enabled", summary.w_prefetch_calls > 0 || summary.w_prefetch_hits > 0},
            {"w_prefetch_calls", summary.w_prefetch_calls},
            {"w_prefetch_hits", summary.w_prefetch_hits},
            {"w_prefetch_conflicts", summary.w_prefetch_conflicts},
            {"w_prefetch_wait_us", summary.w_prefetch_wait_us_total},
            {"w_prefetch_hidden_candidate_us", summary.w_prefetch_hidden_candidate_us_total},
            {"raw_acc_mvout_nodes", summary.raw_acc_mvout_nodes},
            {"raw_acc_mvout_tiles", summary.raw_acc_mvout_tiles},
            {"total_node_us", summary.total_node_us},
            {"accounted_us", summary.accounted_us_total},
            {"unaccounted_us", summary.unaccounted_us_total},
            {"accounted_share_pct", summary.total_node_us == 0 ? 0.0 :
                static_cast<double>(summary.accounted_us_total) / static_cast<double>(summary.total_node_us) * 100.0},
            {"activation_pack_us", summary.activation_pack_us_total},
            {"host_copy_activation_us", summary.host_copy_activation_us_total},
            {"host_copy_weight_us", summary.host_copy_weight_us_total},
            {"bias_prepare_us", summary.bias_prepare_us_total},
            {"postprocess_us", summary.postprocess_us_total},
            {"runtime_total_us", summary.runtime_total_us},
            {"runtime_dma_in_us", summary.runtime_dma_in_us},
            {"runtime_compute_us", summary.runtime_compute_us},
            {"runtime_compute_exclusive_us", summary.runtime_compute_exclusive_us},
            {"runtime_dma_out_us", summary.runtime_dma_out_us},
            {"runtime_layout_us", summary.runtime_layout_us},
            {"runtime_layout_exclusive_us", summary.runtime_layout_exclusive_us},
            {"runtime_wait_irq_us", summary.runtime_wait_irq_us},
            {"runtime_compute_calls", summary.runtime_compute_calls},
            {"runtime_gemm_plan_calls", summary.runtime_gemm_plan_calls},
            {"runtime_mvin_calls", summary.runtime_mvin_calls},
            {"runtime_mvout_calls", summary.runtime_mvout_calls},
            {"dma_in_activation_us", summary.dma_in_activation_us_total},
            {"dma_in_weight_us", summary.dma_in_weight_us_total},
            {"dma_in_bias_us", summary.dma_in_bias_us_total},
            {"dma_in_pair_us", summary.dma_in_pair_us_total},
        };
    }
#endif

    void add_chunk(
            const mtmd_input_chunk * chunk,
            int64_t encode_us,
            int64_t decode_us,
            int64_t total_us,
            const char * mmproj_summary_json,
            const char * cpu_backend_profile_json
#ifdef GGML_USE_NPU
            , const ggml_npu_profile_summary & npu_summary
#endif
            ) {
        if (!enabled || chunk == nullptr) {
            return;
        }

        has_media = true;
        media_chunk_count += 1;
        media_tokens += static_cast<int64_t>(mtmd_input_chunk_get_n_tokens(chunk));
        media_positions += static_cast<int64_t>(mtmd_input_chunk_get_n_pos(chunk));
        mmproj_encode_us += encode_us;
        media_decode_us += decode_us;
        media_process_us += total_us;

        const auto type = mtmd_input_chunk_get_type(chunk);
        if (type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            image_chunk_count += 1;
        } else if (type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
            audio_chunk_count += 1;
        }

        if (mmproj_summary_json != nullptr && mmproj_summary_json[0] != '\0') {
            json detail = json::parse(mmproj_summary_json, nullptr, false);
            if (!detail.is_discarded()) {
                mmproj_encode_details.push_back(std::move(detail));
            }
        }
        if (cpu_backend_profile_json != nullptr && cpu_backend_profile_json[0] != '\0') {
            json detail = json::parse(cpu_backend_profile_json, nullptr, false);
            if (!detail.is_discarded()) {
                cpu_backend_profile_details.push_back(std::move(detail));
            }
        }

#ifdef GGML_USE_NPU
        add_npu_summary(npu, npu_summary);
#endif
    }

    void set_merged_prefill(
            int64_t n_merged_tokens,
            int64_t n_text_tokens,
            int64_t decode_us
#ifdef GGML_USE_NPU
            , const ggml_npu_profile_summary * npu_summary = nullptr
#endif
            ) {
        if (!enabled) {
            return;
        }

        merged_prefill = true;
        merged_tokens = n_merged_tokens;
        text_tokens = n_text_tokens;
        merged_decode_us = decode_us;
#ifdef GGML_USE_NPU
        if (npu_summary != nullptr) {
            add_npu_summary(text_prefill_npu, *npu_summary);
        }
#endif
    }

    json aggregate_mmproj_phase_us() const {
        std::map<std::string, int64_t> phases;
        for (const json & detail : mmproj_encode_details) {
            if (!detail.contains("phases") || !detail["phases"].is_object()) {
                continue;
            }
            for (auto it = detail["phases"].begin(); it != detail["phases"].end(); ++it) {
                if (it.value().is_number_integer()) {
                    phases[it.key()] += it.value().get<int64_t>();
                }
            }
        }

        json out = json::object();
        for (const auto & kv : phases) {
            out[kv.first] = kv.second;
        }
        return out;
    }

    int64_t aggregate_mmproj_known_profile_overhead_us() const {
        int64_t overhead_us = 0;
        for (const json & detail : mmproj_encode_details) {
            if (detail.contains("known_profile_overhead_us") && detail["known_profile_overhead_us"].is_number_integer()) {
                overhead_us += detail["known_profile_overhead_us"].get<int64_t>();
            }
        }
        return overhead_us;
    }

    json aggregate_mmproj_cpu_operator_categories() const {
        std::map<std::string, mmproj_category_aggregate> categories;
        for (const json & detail : mmproj_encode_details) {
            if (!detail.contains("graph") ||
                    !detail["graph"].contains("cpu_operator_categories") ||
                    !detail["graph"]["cpu_operator_categories"].is_array()) {
                continue;
            }

            for (const json & item : detail["graph"]["cpu_operator_categories"]) {
                if (!item.contains("name") || !item["name"].is_string()) {
                    continue;
                }
                const std::string name = item["name"].get<std::string>();
                auto & dst = categories[name];
                dst.node_count += item.value("node_count", 0);
                dst.duration_us += item.value("duration_us", 0);
                dst.elements += item.value("elements", 0);
                dst.bytes += item.value("bytes", 0);
            }
        }

        std::vector<std::pair<std::string, mmproj_category_aggregate>> sorted(categories.begin(), categories.end());
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

    json aggregate_cpu_backend_operator_categories() const {
        std::map<std::string, mmproj_category_aggregate> categories;
        for (const json & detail : cpu_backend_profile_details) {
            if (!detail.contains("operator_categories") || !detail["operator_categories"].is_array()) {
                continue;
            }

            for (const json & item : detail["operator_categories"]) {
                if (!item.contains("name") || !item["name"].is_string()) {
                    continue;
                }
                const std::string name = item["name"].get<std::string>();
                auto & dst = categories[name];
                dst.node_count += item.value("node_count", 0);
                dst.duration_us += item.value("duration_us", 0);
                dst.elements += item.value("elements", 0);
                dst.bytes += item.value("bytes", 0);
            }
        }

        std::vector<std::pair<std::string, mmproj_category_aggregate>> sorted(categories.begin(), categories.end());
        std::sort(sorted.begin(), sorted.end(), [](const auto & a, const auto & b) {
            if (a.second.duration_us != b.second.duration_us) {
                return a.second.duration_us > b.second.duration_us;
            }
            return a.first < b.first;
        });

        json out = json::array();
        for (const auto & kv : sorted) {
            out.push_back({
                {"name", kv.first},
                {"node_count", kv.second.node_count},
                {"duration_us", kv.second.duration_us},
                {"duration_ms", static_cast<double>(kv.second.duration_us) / 1000.0},
                {"elements", kv.second.elements},
                {"bytes", kv.second.bytes},
            });
        }
        return out;
    }

    json to_json(
            double prefill_total_us,
            int32_t prompt_tokens_processed,
            int slot_id,
            int task_id) const {
        json npu_json = {
            {"used_npu", false},
            {"node_count", 0},
            {"exec_tile_count", 0},
            {"gemm_calls", 0},
            {"gemm_plan_calls", 0},
            {"total_node_us", 0},
            {"accounted_us", 0},
            {"unaccounted_us", 0},
            {"accounted_share_pct", 0.0},
            {"activation_pack_us", 0},
            {"host_copy_activation_us", 0},
            {"host_copy_weight_us", 0},
            {"bias_prepare_us", 0},
            {"postprocess_us", 0},
            {"runtime_total_us", 0},
            {"runtime_dma_in_us", 0},
            {"runtime_compute_us", 0},
            {"runtime_compute_exclusive_us", 0},
            {"runtime_dma_out_us", 0},
            {"runtime_layout_us", 0},
            {"runtime_layout_exclusive_us", 0},
            {"runtime_wait_irq_us", 0},
            {"runtime_compute_calls", 0},
            {"runtime_gemm_plan_calls", 0},
        };
        json text_npu_json = npu_json;
        double npu_host_us = 0.0;
        double npu_total_node_us = 0.0;
        double text_npu_host_us = 0.0;
        double text_npu_total_node_us = 0.0;

#ifdef GGML_USE_NPU
        npu_host_us =
            static_cast<double>(npu.activation_pack_us_total +
                                npu.host_copy_activation_us_total +
                                npu.host_copy_weight_us_total +
                                npu.bias_prepare_us_total +
                                npu.postprocess_us_total);
        npu_total_node_us = static_cast<double>(npu.total_node_us);
        text_npu_host_us =
            static_cast<double>(text_prefill_npu.activation_pack_us_total +
                                text_prefill_npu.host_copy_activation_us_total +
                                text_prefill_npu.host_copy_weight_us_total +
                                text_prefill_npu.bias_prepare_us_total +
                                text_prefill_npu.postprocess_us_total);
        text_npu_total_node_us = static_cast<double>(text_prefill_npu.total_node_us);
        npu_json = npu_summary_to_json(npu);
        text_npu_json = npu_summary_to_json(text_prefill_npu);
#endif

        const double residual_cpu_us = std::max(0.0, static_cast<double>(mmproj_encode_us) - npu_total_node_us);
        const double text_residual_us = std::max(0.0, static_cast<double>(merged_decode_us) - text_npu_total_node_us);

        return {
            {"profile_kind", "llama_server_mtmd_prefill_summary"},
            {"timing_unit", "us"},
            {"slot_id", slot_id},
            {"task_id", task_id},
            {"prompt_tokens_processed", prompt_tokens_processed},
            {"prefill_total_us", static_cast<int64_t>(prefill_total_us)},
            {"mmproj", {
                {"has_media", has_media},
                {"merged_prefill", merged_prefill},
                {"merged_tokens", merged_tokens},
                {"text_tokens", text_tokens},
                {"media_chunk_count", media_chunk_count},
                {"image_chunk_count", image_chunk_count},
                {"audio_chunk_count", audio_chunk_count},
                {"media_tokens", media_tokens},
                {"media_positions", media_positions},
                {"encode_us", mmproj_encode_us},
                {"decode_us", media_decode_us},
                {"merged_decode_us", merged_decode_us},
                {"total_chunk_us", media_process_us},
                {"encode_details", mmproj_encode_details},
                {"cpu_backend_profile_details", cpu_backend_profile_details},
            }},
            {"npu", npu_json},
            {"text_prefill", {
                {"decode_us", merged_decode_us},
                {"tokens", merged_tokens},
                {"text_tokens", text_tokens},
                {"media_tokens", media_tokens},
                {"npu", text_npu_json},
                {"derived", {
                    {"npu_host_us", static_cast<int64_t>(text_npu_host_us)},
                    {"npu_total_node_us", static_cast<int64_t>(text_npu_total_node_us)},
                    {"residual_cpu_or_other_us", static_cast<int64_t>(text_residual_us)},
                }},
            }},
            {"derived", {
                {"npu_host_us", static_cast<int64_t>(npu_host_us)},
                {"npu_total_node_us", static_cast<int64_t>(npu_total_node_us)},
                {"mmproj_residual_cpu_us", static_cast<int64_t>(residual_cpu_us)},
                {"mmproj_residual_cpu_detail", {
                    {"kind", "non_intrusive_phase_timing_and_graph_categories"},
                    {"note", "This splits the residual by coarse mmproj phases and scheduled CPU graph categories. CPU operator categories are node/byte counts, not exact per-operator runtime."},
                    {"known_profile_overhead_us", aggregate_mmproj_known_profile_overhead_us()},
                    {"phase_wall_time_us", aggregate_mmproj_phase_us()},
                    {"cpu_operator_categories", aggregate_mmproj_cpu_operator_categories()},
                    {"cpu_backend_operator_categories", aggregate_cpu_backend_operator_categories()},
                }},
                {"prefill_minus_mmproj_us", static_cast<int64_t>(std::max(0.0, prefill_total_us - static_cast<double>(mmproj_encode_us)))},
            }},
        };
    }
};

template <typename T>
static T json_value(const json & body, const std::string & key, const T & default_value) {
    // Fallback null to default value
    if (body.contains(key) && !body.at(key).is_null()) {
        try {
            return body.at(key);
        } catch (NLOHMANN_JSON_NAMESPACE::detail::type_error const & err) {
            LOG_WRN("Wrong type supplied for parameter '%s'. Expected '%s', using default value: %s\n", key.c_str(), json(default_value).type_name(), err.what());
            return default_value;
        }
    } else {
        return default_value;
    }
}

const static std::string build_info("b" + std::to_string(LLAMA_BUILD_NUMBER) + "-" + LLAMA_COMMIT);

// thin wrapper around common_grammar_trigger with (de)serialization functions
struct server_grammar_trigger {
    common_grammar_trigger value;

    server_grammar_trigger() = default;
    server_grammar_trigger(const common_grammar_trigger & value) : value(value) {}
    server_grammar_trigger(const json & in) {
        value.type = (common_grammar_trigger_type) in.at("type").get<int>();
        value.value = in.at("value").get<std::string>();
        if (value.type == COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN) {
            value.token = (llama_token) in.at("token").get<int>();
        }
    }

    json to_json() const {
        json out {
            {"type", (int) value.type},
            {"value", value.value},
        };
        if (value.type == COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN) {
            out["token"] = (int) value.token;
        }
        return out;
    }
};

//
// tokenizer and input processing utils
//

static bool json_is_array_of_numbers(const json & data) {
    if (data.is_array()) {
        for (const auto & e : data) {
            if (!e.is_number_integer()) {
                return false;
            }
        }
        return true;
    }
    return false;
}

// is array having BOTH numbers & strings?
static bool json_is_array_of_mixed_numbers_strings(const json & data) {
    bool seen_string = false;
    bool seen_number = false;
    if (data.is_array()) {
        for (const auto & e : data) {
            seen_string |= e.is_string();
            seen_number |= e.is_number_integer();
            if (seen_number && seen_string) {
                return true;
            }
        }
    }
    return false;
}

// does array have any individual integers/tokens?
static bool json_is_array_and_contains_numbers(const json & data) {
    if (data.is_array()) {
        for (const auto & e : data) {
            if (e.is_number_integer()) {
                return true;
            }
        }
        return false;
    }
    return false;
}

// get value by path(key1 / key2)
static json json_get_nested_values(const std::vector<std::string> & paths, const json & js) {
    json result = json::object();

    for (const std::string & path : paths) {
        json current = js;
        const auto keys = string_split<std::string>(path, /*separator*/ '/');
        bool valid_path = true;
        for (const std::string & k : keys) {
            if (valid_path && current.is_object() && current.contains(k)) {
                current = current[k];
            } else {
                valid_path = false;
            }
        }
        if (valid_path) {
            result[path] = current;
        }
    }
    return result;
}

/**
 * this handles 2 cases:
 * - only string, example: "string"
 * - mixed string and tokens, example: [12, 34, "string", 56, 78]
 */
static llama_tokens tokenize_mixed(const llama_vocab * vocab, const json & json_prompt, bool add_special, bool parse_special) {
    // If `add_bos` is true, we only add BOS, when json_prompt is a string,
    // or the first element of the json_prompt array is a string.
    llama_tokens prompt_tokens;

    if (json_prompt.is_array()) {
        bool first = true;
        for (const auto & p : json_prompt) {
            if (p.is_string()) {
                auto s = p.template get<std::string>();

                llama_tokens p;
                if (first) {
                    p = common_tokenize(vocab, s, add_special, parse_special);
                    first = false;
                } else {
                    p = common_tokenize(vocab, s, false, parse_special);
                }

                prompt_tokens.insert(prompt_tokens.end(), p.begin(), p.end());
            } else {
                if (first) {
                    first = false;
                }

                prompt_tokens.push_back(p.template get<llama_token>());
            }
        }
    } else {
        auto s = json_prompt.template get<std::string>();
        prompt_tokens = common_tokenize(vocab, s, add_special, parse_special);
    }

    return prompt_tokens;
}

// return the last index of character that can form a valid string
// if the last character is potentially cut in half, return the index before the cut
// if validate_utf8(text) == text.size(), then the whole text is valid utf8
static size_t validate_utf8(const std::string& text) {
    size_t len = text.size();
    if (len == 0) return 0;

    // Check the last few bytes to see if a multi-byte character is cut off
    for (size_t i = 1; i <= 4 && i <= len; ++i) {
        unsigned char c = text[len - i];
        // Check for start of a multi-byte sequence from the end
        if ((c & 0xE0) == 0xC0) {
            // 2-byte character start: 110xxxxx
            // Needs at least 2 bytes
            if (i < 2) return len - i;
        } else if ((c & 0xF0) == 0xE0) {
            // 3-byte character start: 1110xxxx
            // Needs at least 3 bytes
            if (i < 3) return len - i;
        } else if ((c & 0xF8) == 0xF0) {
            // 4-byte character start: 11110xxx
            // Needs at least 4 bytes
            if (i < 4) return len - i;
        }
    }

    // If no cut-off multi-byte character is found, return full length
    return len;
}

//
// template utils
//

// format infill task
static llama_tokens format_infill(
        const llama_vocab * vocab,
        const json & input_prefix,
        const json & input_suffix,
        const json & input_extra,
        const int n_batch,
        const int n_predict,
        const int n_ctx,
        const bool spm_infill,
        const llama_tokens & tokens_prompt
    ) {
    // TODO: optimize this block by reducing memory allocations and movement

    // use FIM repo-level pattern:
    // ref: https://arxiv.org/pdf/2409.12186
    //
    // [FIM_REP]myproject
    // [FIM_SEP]filename0
    // extra chunk 0
    // [FIM_SEP]filename1
    // extra chunk 1
    // ...
    // [FIM_SEP]filename
    // [FIM_PRE]prefix[FIM_SUF]suffix[FIM_MID]prompt
    //
    llama_tokens extra_tokens;
    extra_tokens.reserve(n_ctx);

    auto tokens_prefix = tokenize_mixed(vocab, input_prefix, false, false);
    auto tokens_suffix = tokenize_mixed(vocab, input_suffix, false, false);

    if (llama_vocab_fim_rep(vocab) != LLAMA_TOKEN_NULL) {
        // TODO: make project name an input
        static const auto k_fim_repo = common_tokenize(vocab, "myproject\n", false, false);

        extra_tokens.push_back(llama_vocab_fim_rep(vocab));
        extra_tokens.insert(extra_tokens.end(), k_fim_repo.begin(), k_fim_repo.end());
    }
    for (const auto & chunk : input_extra) {
        // { "text": string, "filename": string }
        const std::string text     = json_value(chunk, "text",     std::string());
        const std::string filename = json_value(chunk, "filename", std::string("tmp"));

        if (llama_vocab_fim_sep(vocab) != LLAMA_TOKEN_NULL) {
            const auto k_fim_file = common_tokenize(vocab, filename + "\n", false, false);

            extra_tokens.insert(extra_tokens.end(), llama_vocab_fim_sep(vocab));
            extra_tokens.insert(extra_tokens.end(), k_fim_file.begin(), k_fim_file.end());
        } else {
            // chunk separator in binary form to avoid confusing the AI
            static const char k_chunk_prefix_str[] = {0x0a, 0x0a, 0x2d, 0x2d, 0x2d, 0x20, 0x73, 0x6e, 0x69, 0x70, 0x70, 0x65, 0x74, 0x20, 0x2d, 0x2d, 0x2d, 0x0a, 0x0a, 0x00};
            static const auto k_chunk_prefix_tokens = common_tokenize(vocab, k_chunk_prefix_str, false, false);

            extra_tokens.insert(extra_tokens.end(), k_chunk_prefix_tokens.begin(), k_chunk_prefix_tokens.end());
        }

        const auto chunk_tokens = common_tokenize(vocab, text, false, false);
        extra_tokens.insert(extra_tokens.end(), chunk_tokens.begin(), chunk_tokens.end());
    }

    if (llama_vocab_fim_sep(vocab) != LLAMA_TOKEN_NULL) {
        // TODO: current filename
        static const auto k_fim_file = common_tokenize(vocab, "filename\n", false, false);

        extra_tokens.insert(extra_tokens.end(), llama_vocab_fim_sep(vocab));
        extra_tokens.insert(extra_tokens.end(), k_fim_file.begin(), k_fim_file.end());
    }

    // for now pick FIM context to fit in a batch (ratio prefix:suffix = 3:1, TODO: configurable?)
    const int n_prefix_take = std::min<int>(tokens_prefix.size(),                3*(n_batch/4));
    const int n_suffix_take = std::min<int>(tokens_suffix.size(), std::max<int>(0, (n_batch/4) - (2 + tokens_prompt.size())));

    SRV_DBG("n_prefix_take = %d, n_suffix_take = %d, total = %d\n", n_prefix_take, n_suffix_take, (n_prefix_take + n_suffix_take));

    // fill the rest of the context with extra chunks
    const int n_extra_take = std::min<int>(std::max<int>(0, n_ctx - (n_batch) - 2*n_predict), extra_tokens.size());

    tokens_prefix.erase(tokens_prefix.begin(), tokens_prefix.begin() + tokens_prefix.size() - n_prefix_take);
    tokens_suffix.resize(n_suffix_take);

    tokens_prefix.insert(tokens_prefix.begin(), llama_vocab_fim_pre(vocab));
    tokens_prefix.insert(tokens_prefix.end(),   tokens_prompt.begin(), tokens_prompt.end());
    tokens_suffix.insert(tokens_suffix.begin(), llama_vocab_fim_suf(vocab));

    auto embd_inp = spm_infill ? tokens_suffix : tokens_prefix;
    auto embd_end = spm_infill ? tokens_prefix : tokens_suffix;

    if (llama_vocab_get_add_bos(vocab)) {
        embd_inp.insert(embd_inp.begin(), llama_vocab_bos(vocab));
    }

    SRV_DBG("extra: n_ctx = %d, n_extra_take = %d, n_extra = %d\n", n_ctx, n_extra_take, (int) extra_tokens.size());

    // put the extra context before the FIM prefix
    embd_inp.insert(embd_inp.begin(), extra_tokens.end() - n_extra_take, extra_tokens.end());

    embd_inp.insert(embd_inp.end(), embd_end.begin(), embd_end.end());
    embd_inp.push_back(llama_vocab_fim_mid(vocab));

    return embd_inp;
}

//
// base64 utils (TODO: move to common in the future)
//

static const std::string base64_chars =
             "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
             "abcdefghijklmnopqrstuvwxyz"
             "0123456789+/";

static inline bool is_base64(uint8_t c) {
    return (isalnum(c) || (c == '+') || (c == '/'));
}

static inline raw_buffer base64_decode(const std::string & encoded_string) {
    int i = 0;
    int j = 0;
    int in_ = 0;

    int in_len = encoded_string.size();

    uint8_t char_array_4[4];
    uint8_t char_array_3[3];

    raw_buffer ret;

    while (in_len-- && (encoded_string[in_] != '=') && is_base64(encoded_string[in_])) {
        char_array_4[i++] = encoded_string[in_]; in_++;
        if (i == 4) {
            for (i = 0; i < 4; i++) {
                char_array_4[i] = base64_chars.find(char_array_4[i]);
            }

            char_array_3[0] = ((char_array_4[0]      ) << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) +   char_array_4[3];

            for (i = 0; (i < 3); i++) {
                ret.push_back(char_array_3[i]);
            }

            i = 0;
        }
    }

    if (i) {
        for (j = i; j < 4; j++) {
            char_array_4[j] = 0;
        }

        for (j = 0; j < 4; j++) {
            char_array_4[j] = base64_chars.find(char_array_4[j]);
        }

        char_array_3[0] = ((char_array_4[0]      ) << 2) + ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
        char_array_3[2] = ((char_array_4[2] & 0x3) << 6) +   char_array_4[3];

        for (j = 0; j < i - 1; j++) {
            ret.push_back(char_array_3[j]);
        }
    }

    return ret;
}

//
// random string / id
//

static std::string random_string() {
    static const std::string str("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz");

    std::random_device rd;
    std::mt19937 generator(rd());

    std::string result(32, ' ');

    for (int i = 0; i < 32; ++i) {
        result[i] = str[generator() % str.size()];
    }

    return result;
}

static std::string gen_chatcmplid() {
    return "chatcmpl-" + random_string();
}

static std::string gen_tool_call_id() {
    return random_string();
}

//
// other common utils
//

// TODO: reuse llama_detokenize
template <class Iter>
static std::string tokens_to_str(llama_context * ctx, Iter begin, Iter end) {
    std::string ret;
    for (; begin != end; ++begin) {
        ret += common_token_to_piece(ctx, *begin);
    }

    return ret;
}

// format incomplete utf-8 multibyte character for output
static std::string tokens_to_output_formatted_string(const llama_context * ctx, const llama_token token) {
    std::string out = token == LLAMA_TOKEN_NULL ? "" : common_token_to_piece(ctx, token);

    // if the size is 1 and first bit is 1, meaning it's a partial character
    //   (size > 1 meaning it's already a known token)
    if (out.size() == 1 && (out[0] & 0x80) == 0x80) {
        std::stringstream ss;
        ss << std::hex << (out[0] & 0xff);
        std::string res(ss.str());
        out = "byte: \\x" + res;
    }

    return out;
}

static bool server_sent_event(httplib::DataSink & sink, const json & data) {
    const std::string str =
        "data: " +
        data.dump(-1, ' ', false, json::error_handler_t::replace) +
        "\n\n"; // required by RFC 8895 - A message is terminated by a blank line (two line terminators in a row).

    LOG_DBG("data stream, to_send: %s", str.c_str());

    return sink.write(str.c_str(), str.size());
}

//
// OAI utils
//

// used by /completions endpoint
static json oaicompat_completion_params_parse(const json & body) {
    json llama_params;

    if (!body.contains("prompt")) {
        throw std::runtime_error("\"prompt\" is required");
    }

    // Handle "stop" field
    if (body.contains("stop") && body.at("stop").is_string()) {
        llama_params["stop"] = json::array({body.at("stop").get<std::string>()});
    } else {
        llama_params["stop"] = json_value(body, "stop", json::array());
    }

    // Handle "n" field
    int n_choices = json_value(body, "n", 1);
    if (n_choices != 1) {
        throw std::runtime_error("Only one completion choice is allowed");
    }

    // Handle "echo" field
    if (json_value(body, "echo", false)) {
        throw std::runtime_error("Only no echo is supported");
    }

    // Params supported by OAI but unsupported by llama.cpp
    static const std::vector<std::string> unsupported_params { "best_of", "suffix" };
    for (const auto & param : unsupported_params) {
        if (body.contains(param)) {
            throw std::runtime_error("Unsupported param: " + param);
        }
    }

    // Copy remaining properties to llama_params
    for (const auto & item : body.items()) {
        // Exception: if "n_predict" is present, we overwrite the value specified earlier by "max_tokens"
        if (!llama_params.contains(item.key()) || item.key() == "n_predict") {
            llama_params[item.key()] = item.value();
        }
    }

    return llama_params;
}

struct oaicompat_parser_options {
    bool use_jinja;
    bool prefill_assistant;
    common_reasoning_format reasoning_format;
    std::map<std::string,std::string> chat_template_kwargs;
    common_chat_templates * tmpls;
    bool allow_image;
    bool allow_audio;
    bool enable_thinking = true;
};

// used by /chat/completions endpoint
static json oaicompat_chat_params_parse(
    json & body, /* openai api json semantics */
    const oaicompat_parser_options & opt,
    std::vector<raw_buffer> & out_files)
{
    json llama_params;

    auto tools = json_value(body, "tools", json());
    auto has_tools = tools.is_array() && !tools.empty();
    auto stream = json_value(body, "stream", false);
    auto tool_choice = json_value(body, "tool_choice", std::string("auto"));

    if (!opt.use_jinja) {
        if (has_tools) {
            throw std::runtime_error("tools param requires --jinja flag");
        }
        if (tool_choice != "auto") {
            throw std::runtime_error("tool_choice param requires --jinja flag");
        }
    }

    // Handle "stop" field
    if (body.contains("stop") && body.at("stop").is_string()) {
        llama_params["stop"] = json::array({body.at("stop").get<std::string>()});
    } else {
        llama_params["stop"] = json_value(body, "stop", json::array());
    }

    auto json_schema = json_value(body, "json_schema", json());
    auto grammar = json_value(body, "grammar", std::string());
    if (!json_schema.is_null() && !grammar.empty()) {
        throw std::runtime_error("Cannot use both json_schema and grammar");
    }

    // Handle "response_format" field
    if (body.contains("response_format")) {
        json response_format      = json_value(body, "response_format", json::object());
        std::string response_type = json_value(response_format, "type", std::string());
        if (response_type == "json_object") {
            json_schema = json_value(response_format, "schema", json::object());
        } else if (response_type == "json_schema") {
            auto schema_wrapper = json_value(response_format, "json_schema", json::object());
            json_schema = json_value(schema_wrapper, "schema", json::object());
        } else if (!response_type.empty() && response_type != "text") {
            throw std::runtime_error("response_format type must be one of \"text\" or \"json_object\", but got: " + response_type);
        }
    }

    // get input files
    if (!body.contains("messages")) {
        throw std::runtime_error("'messages' is required");
    }
    json & messages = body.at("messages");
    if (!messages.is_array()) {
        throw std::runtime_error("Expected 'messages' to be an array");
    }
    for (auto & msg : messages) {
        std::string role = json_value(msg, "role", std::string());
        if (role != "assistant" && !msg.contains("content")) {
            throw std::runtime_error("All non-assistant messages must contain 'content'");
        }
        if (role == "assistant") {
            if (!msg.contains("content") && !msg.contains("tool_calls")) {
                throw std::runtime_error("Assistant message must contain either 'content' or 'tool_calls'!");
            }
            if (!msg.contains("content")) {
                continue; // avoid errors with no content
            }
        }
        json & content = msg.at("content");
        if (content.is_string() || content.is_null()) {
            continue;
        }

        if (!content.is_array()) {
            throw std::runtime_error("Expected 'content' to be a string or an array");
        }

        for (auto & p : content) {
            std::string type      = json_value(p, "type", std::string());
            if (type == "image_url") {
                if (!opt.allow_image) {
                    throw std::runtime_error("image input is not supported - hint: if this is unexpected, you may need to provide the mmproj");
                }

                json image_url  = json_value(p, "image_url", json::object());
                std::string url = json_value(image_url, "url", std::string());
                if (string_starts_with(url, "http")) {
                    // download remote image
                    // TODO @ngxson : maybe make these params configurable
                    common_remote_params params;
                    params.headers.push_back("User-Agent: llama.cpp/" + build_info);
                    params.max_size = 1024 * 1024 * 10; // 10MB
                    params.timeout  = 10; // seconds
                    SRV_INF("downloading image from '%s'\n", url.c_str());
                    auto res = common_remote_get_content(url, params);
                    if (200 <= res.first && res.first < 300) {
                        SRV_INF("downloaded %ld bytes\n", res.second.size());
                        raw_buffer data;
                        data.insert(data.end(), res.second.begin(), res.second.end());
                        out_files.push_back(data);
                    } else {
                        throw std::runtime_error("Failed to download image");
                    }

                } else {
                    // try to decode base64 image
                    std::vector<std::string> parts = string_split<std::string>(url, /*separator*/ ',');
                    if (parts.size() != 2) {
                        throw std::runtime_error("Invalid image_url.url value");
                    } else if (!string_starts_with(parts[0], "data:image/")) {
                        throw std::runtime_error("Invalid image_url.url format: " + parts[0]);
                    } else if (!string_ends_with(parts[0], "base64")) {
                        throw std::runtime_error("image_url.url must be base64 encoded");
                    } else {
                        auto base64_data = parts[1];
                        auto decoded_data = base64_decode(base64_data);
                        out_files.push_back(decoded_data);
                    }
                }

                // replace this chunk with a marker
                p["type"] = "text";
                p["text"] = mtmd_default_marker();
                p.erase("image_url");

            } else if (type == "input_audio") {
                if (!opt.allow_audio) {
                    throw std::runtime_error("audio input is not supported - hint: if this is unexpected, you may need to provide the mmproj");
                }

                json input_audio   = json_value(p, "input_audio", json::object());
                std::string data   = json_value(input_audio, "data", std::string());
                std::string format = json_value(input_audio, "format", std::string());
                // while we also support flac, we don't allow it here so we matches the OAI spec
                if (format != "wav" && format != "mp3") {
                    throw std::runtime_error("input_audio.format must be either 'wav' or 'mp3'");
                }
                auto decoded_data = base64_decode(data); // expected to be base64 encoded
                out_files.push_back(decoded_data);

                // replace this chunk with a marker
                p["type"] = "text";
                p["text"] = mtmd_default_marker();
                p.erase("input_audio");

            } else if (type != "text") {
                throw std::runtime_error("unsupported content[].type");
            }
        }
    }

    common_chat_templates_inputs inputs;
    inputs.messages              = common_chat_msgs_parse_oaicompat(messages);
    inputs.tools                 = common_chat_tools_parse_oaicompat(tools);
    inputs.tool_choice           = common_chat_tool_choice_parse_oaicompat(tool_choice);
    inputs.json_schema           = json_schema.is_null() ? "" : json_schema.dump();
    inputs.grammar               = grammar;
    inputs.use_jinja             = opt.use_jinja;
    inputs.parallel_tool_calls   = json_value(body, "parallel_tool_calls", false);
    inputs.add_generation_prompt = json_value(body, "add_generation_prompt", true);
    inputs.reasoning_format      = opt.reasoning_format;
    inputs.enable_thinking       = opt.enable_thinking;
    if (!inputs.tools.empty() && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
        if (body.contains("grammar")) {
            throw std::runtime_error("Cannot use custom grammar constraints with tools.");
        }
        llama_params["parse_tool_calls"] = true;
    }

    // merge the template args provided from command line with the args provided in the user request
    auto chat_template_kwargs_object = json_value(body, "chat_template_kwargs", json::object());
    inputs.chat_template_kwargs = opt.chat_template_kwargs;
    for (const auto & item : chat_template_kwargs_object.items()) {
        inputs.chat_template_kwargs[item.key()] = item.value().dump();
    }

    // parse the "enable_thinking" kwarg to override the default value
    auto enable_thinking_kwarg = json_value(inputs.chat_template_kwargs, "enable_thinking", std::string(""));
    if (enable_thinking_kwarg == "true") {
        inputs.enable_thinking = true;
    } else if (enable_thinking_kwarg == "false") {
        inputs.enable_thinking = false;
    } else if (!enable_thinking_kwarg.empty() && enable_thinking_kwarg[0] == '"') {
        throw std::runtime_error("invalid type for \"enable_thinking\" (expected boolean, got string)");
    }

    // if the assistant message appears at the end of list, we do not add end-of-turn token
    // for ex. this can be useful to modify the reasoning process in reasoning models
    bool prefill_assistant_message = !inputs.messages.empty() && inputs.messages.back().role == "assistant" && opt.prefill_assistant;
    common_chat_msg last_message;
    if (prefill_assistant_message) {
        last_message = inputs.messages.back();
        inputs.messages.pop_back();

        /* sanity check, max one assistant message at the end of the list */
        if (!inputs.messages.empty() && inputs.messages.back().role == "assistant"){
            throw std::runtime_error("Cannot have 2 or more assistant messages at the end of the list.");
        }

        /* TODO: test this properly */
        inputs.reasoning_format = COMMON_REASONING_FORMAT_NONE;

        if ( inputs.enable_thinking ) {
            throw std::runtime_error("Assistant response prefill is incompatible with enable_thinking.");
        }

        inputs.add_generation_prompt = true;
    }

    // Apply chat template to the list of messages
    auto chat_params = common_chat_templates_apply(opt.tmpls, inputs);

    /* Append assistant prefilled message */
    if (prefill_assistant_message) {
        if (!last_message.content_parts.empty()) {
            for (auto & p : last_message.content_parts) {
                chat_params.prompt += p.text;
            }
        } else {
            chat_params.prompt += last_message.content;
        }
    }

    llama_params["chat_format"]      = static_cast<int>(chat_params.format);
    llama_params["prompt"]           = chat_params.prompt;
    if (!chat_params.grammar.empty()) {
        llama_params["grammar"] = chat_params.grammar;
    }
    llama_params["grammar_lazy"]     = chat_params.grammar_lazy;
    auto grammar_triggers = json::array();
    for (const auto & trigger : chat_params.grammar_triggers) {
        server_grammar_trigger ct(trigger);
        grammar_triggers.push_back(ct.to_json());
    }
    llama_params["grammar_triggers"] = grammar_triggers;
    llama_params["preserved_tokens"] = chat_params.preserved_tokens;
    llama_params["thinking_forced_open"]     = chat_params.thinking_forced_open;
    for (const auto & stop : chat_params.additional_stops) {
        llama_params["stop"].push_back(stop);
    }

    // Handle "n" field
    int n_choices = json_value(body, "n", 1);
    if (n_choices != 1) {
        throw std::runtime_error("Only one completion choice is allowed");
    }

    // Handle "logprobs" field
    // TODO: The response format of this option is not yet OAI-compatible, but seems like no one really using it; We may need to fix it in the future
    if (json_value(body, "logprobs", false)) {
        if (has_tools && stream) {
            throw std::runtime_error("logprobs is not supported with tools + stream");
        }
        llama_params["n_probs"] = json_value(body, "top_logprobs", 20);
    } else if (body.contains("top_logprobs") && !body.at("top_logprobs").is_null()) {
        throw std::runtime_error("top_logprobs requires logprobs to be set to true");
    }

    // Copy remaining properties to llama_params
    // This allows user to use llama.cpp-specific params like "mirostat", ... via OAI endpoint.
    // See "launch_slot_with_task()" for a complete list of params supported by llama.cpp
    for (const auto & item : body.items()) {
        // Exception: if "n_predict" is present, we overwrite the value specified earlier by "max_tokens"
        if (!llama_params.contains(item.key()) || item.key() == "n_predict") {
            llama_params[item.key()] = item.value();
        }
    }

    return llama_params;
}

static json format_embeddings_response_oaicompat(const json & request, const json & embeddings, bool use_base64 = false) {
    json data = json::array();
    int32_t n_tokens = 0;
    int i = 0;
    for (const auto & elem : embeddings) {
        json embedding_obj;

        if (use_base64) {
            const auto& vec = json_value(elem, "embedding", json::array()).get<std::vector<float>>();
            const char* data_ptr = reinterpret_cast<const char*>(vec.data());
            size_t data_size = vec.size() * sizeof(float);
            embedding_obj = {
                {"embedding", base64::encode(data_ptr, data_size)},
                {"index", i++},
                {"object", "embedding"},
                {"encoding_format", "base64"}
            };
        } else {
            embedding_obj = {
                {"embedding", json_value(elem, "embedding", json::array())},
                {"index", i++},
                {"object", "embedding"}
            };
        }
        data.push_back(embedding_obj);

        n_tokens += json_value(elem, "tokens_evaluated", 0);
    }

    json res = json {
        {"model", json_value(request, "model", std::string(DEFAULT_OAICOMPAT_MODEL))},
        {"object", "list"},
        {"usage", json {
            {"prompt_tokens", n_tokens},
            {"total_tokens", n_tokens}
        }},
        {"data", data}
    };

    return res;
}

static json format_response_rerank(
        const json & request,
        const json & ranks,
        bool is_tei_format,
        std::vector<std::string> & texts,
        int top_n) {
    int32_t n_tokens = 0;
    bool return_text = is_tei_format && json_value(request, "return_text", false);
    std::vector<json> elements; // Temporary vector to hold unsorted elements
    std::string score_label = is_tei_format ? "score" : "relevance_score";
    for (const auto & rank : ranks) {
        int index = json_value(rank, "index", 0);
        json elem = json{
            {"index", index},
            {score_label, json_value(rank, "score", 0.0)},
        };
        n_tokens += json_value(rank, "tokens_evaluated", 0);
        if (return_text) {
            elem["text"] = std::move(texts[index]);
        }
        elements.push_back(elem);
    }

    std::sort(elements.begin(), elements.end(), [score_label](const json& a, const json& b) {
        return json_value(a, score_label, 0.0) > json_value(b, score_label, 0.0);
    });

    elements.resize(std::min(top_n, (int)elements.size()));
    json results = elements;

    if (is_tei_format) return results;

    json res = json{
        {"model", json_value(request, "model", std::string(DEFAULT_OAICOMPAT_MODEL))},
        {"object", "list"},
        {"usage", json{
            {"prompt_tokens", n_tokens},
            {"total_tokens", n_tokens}
        }},
        {"results", results}
    };

    return res;
}

static bool is_valid_utf8(const std::string & str) {
    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(str.data());
    const unsigned char* end = bytes + str.length();

    while (bytes < end) {
        if (*bytes <= 0x7F) {
            // 1-byte sequence (0xxxxxxx)
            bytes++;
        } else if ((*bytes & 0xE0) == 0xC0) {
            // 2-byte sequence (110xxxxx 10xxxxxx)
            if (end - bytes < 2 || (bytes[1] & 0xC0) != 0x80)
                return false;
            bytes += 2;
        } else if ((*bytes & 0xF0) == 0xE0) {
            // 3-byte sequence (1110xxxx 10xxxxxx 10xxxxxx)
            if (end - bytes < 3 || (bytes[1] & 0xC0) != 0x80 || (bytes[2] & 0xC0) != 0x80)
                return false;
            bytes += 3;
        } else if ((*bytes & 0xF8) == 0xF0) {
            // 4-byte sequence (11110xxx 10xxxxxx 10xxxxxx 10xxxxxx)
            if (end - bytes < 4 || (bytes[1] & 0xC0) != 0x80 ||
                (bytes[2] & 0xC0) != 0x80 || (bytes[3] & 0xC0) != 0x80)
                return false;
            bytes += 4;
        } else {
            // Invalid UTF-8 lead byte
            return false;
        }
    }

    return true;
}

static json format_tokenizer_response(const json & tokens) {
    return json {
        {"tokens", tokens}
    };
}

static json format_detokenized_response(const std::string & content) {
    return json {
        {"content", content}
    };
}

static json format_logit_bias(const std::vector<llama_logit_bias> & logit_bias) {
    json data = json::array();
    for (const auto & lb : logit_bias) {
        data.push_back(json{
            {"bias", lb.bias},
            {"token", lb.token},
        });
    }
    return data;
}

static std::string safe_json_to_str(const json & data) {
    return data.dump(-1, ' ', false, json::error_handler_t::replace);
}

static std::vector<llama_token_data> get_token_probabilities(llama_context * ctx, int idx) {
    std::vector<llama_token_data> cur;
    const auto * logits = llama_get_logits_ith(ctx, idx);

    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const int n_vocab = llama_vocab_n_tokens(vocab);

    cur.resize(n_vocab);
    for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
        cur[token_id] = llama_token_data{token_id, logits[token_id], 0.0f};
    }

    // sort tokens by logits
    std::sort(cur.begin(), cur.end(), [](const llama_token_data & a, const llama_token_data & b) {
        return a.logit > b.logit;
    });

    // apply softmax
    float max_l = cur[0].logit;
    float cum_sum = 0.0f;
    for (size_t i = 0; i < cur.size(); ++i) {
        float p = expf(cur[i].logit - max_l);
        cur[i].p = p;
        cum_sum += p;
    }
    for (size_t i = 0; i < cur.size(); ++i) {
        cur[i].p /= cum_sum;
    }

    return cur;
}

static bool are_lora_equal(
        const std::vector<common_adapter_lora_info> & l1,
        const std::vector<common_adapter_lora_info> & l2) {
    if (l1.size() != l2.size()) {
        return false;
    }
    for (size_t i = 0; i < l1.size(); ++i) {
        // we don't check lora.path to reduce the time complexity
        if (l1[i].scale != l2[i].scale || l1[i].ptr != l2[i].ptr) {
            return false;
        }
    }
    return true;
}

// get the ids of all enabled loras
static std::vector<size_t> lora_get_enabled_ids(const std::vector<common_adapter_lora_info> & loras) {
    std::vector<size_t> enabled_ids;
    for (size_t i = 0; i < loras.size(); ++i) {
        if (loras[i].scale > 0) {
            enabled_ids.push_back(i);
        }
    }
    return enabled_ids;
}

// check whether the given lora set has only aloras activated (empty => false)
static bool lora_all_alora(const std::vector<common_adapter_lora_info> & loras) {
    bool found_alora = false;
    for (const auto & lora : loras) {
        if (lora.scale != 0) {
            if (llama_adapter_get_alora_n_invocation_tokens(lora.ptr) == 0) {
                return false;
            }
            found_alora = true;
        }
    }
    return found_alora;
}

// if the two sets of loras are different, they require a cache clear unless the
// change is only from aloras to aloras.
static bool lora_should_clear_cache(
        const std::vector<common_adapter_lora_info> & current,
        const std::vector<common_adapter_lora_info> & next) {

    // This should always be called after determining that the two sets are
    // _not_ equal. This assert is therefore some slightly wasted work and
    // should be safe to remove as long as this method is called correctly.
    GGML_ASSERT(!are_lora_equal(current, next));

    return (
        !(lora_get_enabled_ids(current).empty() || lora_all_alora(current)) ||
        !lora_all_alora(next));
}

// parse lora config from JSON request, returned a copy of lora_base with updated scale
static std::vector<common_adapter_lora_info> parse_lora_request(
        const std::vector<common_adapter_lora_info> & lora_base,
        const json & data) {
    std::vector<common_adapter_lora_info> lora(lora_base);
    int max_idx = lora.size();

    // clear existing value
    for (auto & entry : lora) {
        entry.scale = 0.0f;
    }

    // set value
    for (const auto & entry : data) {
        int id      = json_value(entry, "id", -1);
        float scale = json_value(entry, "scale", 0.0f);
        if (0 <= id && id < max_idx) {
            lora[id].scale = scale;
        } else {
            throw std::runtime_error("invalid adapter id");
        }
    }

    return lora;
}

//
// utils for interacting with libmtmd
// (may need to refactor in near future)
//

/**
 * server_tokens is a helper to manage the input tokens and image for the server.
 * it is made this way to simplify the logic of KV cache management.
 */
struct server_tokens {
    bool has_mtmd = false;

private: // disallow accessing these members directly, risking out-of-sync

    // map a **start** position in tokens to the image chunk
    std::unordered_map<llama_pos, mtmd::input_chunk_ptr> map_pos_to_media;

    // list of tokens
    // it can include LLAMA_TOKEN_NULL, which is used to indicate a token that is not a text token
    // a mtmd_input_chunk can occupy multiple tokens, one llama_token per **position**
    // important: for models using mrope, an image can contain multiple tokens but will use only one **position**
    llama_tokens tokens;

    // for ex. with input of 5 text tokens and 2 images:
    //      [0] [1] [2] [3] [4] [img0] [img0] [img0] [img1] [img1]
    // pos  0   1   2   3   4   5      6      7      8      9
    // map_pos_to_media will contain: {5, img0}, {8, img1}

public:
    server_tokens() = default;
    ~server_tokens() = default;

    // Prevent copying
    // TODO: server_tokens should be copyable - remove this:
    server_tokens(const server_tokens&) = delete;
    server_tokens& operator=(const server_tokens&) = delete;

    // Allow moving (usually implicitly generated if members are movable)
    server_tokens(server_tokens&&) = default;
    server_tokens& operator=(server_tokens&&) = default;

    // Allow accessing elements using [] operator
    llama_token operator[](size_t index) { return tokens[index]; }
    const llama_token& operator[](size_t index) const { return tokens[index]; }

    server_tokens(mtmd::input_chunks & mtmd_chunks, bool has_mtmd) : has_mtmd(has_mtmd) {
        for (size_t i = 0; i < mtmd_chunks.size(); ++i) {
            push_back(mtmd_chunks[i]);
        }
    }

    server_tokens(const llama_tokens & tokens, bool has_mtmd) : has_mtmd(has_mtmd), tokens(tokens) {}

    // for debugging
    std::string str() const {
        std::ostringstream oss;
        oss << "tokens: ";
        for (const auto & t : tokens) {
            if (t == LLAMA_TOKEN_NULL) {
                oss << "<embd> ";
            } else {
                oss << t << " ";
            }
        }
        oss << "\n";
        oss << "image pos: ";
        for (const auto & it : map_pos_to_media) {
            oss << it.first << ", ";
        }
        return oss.str();
    }

    const mtmd::input_chunk_ptr & find_chunk(llama_pos pos) const {
        auto it = map_pos_to_media.find(pos);
        if (it != map_pos_to_media.end()) {
            return it->second;
        }
        throw std::runtime_error("Chunk not found");
    }

    void push_back(llama_token tok) {
        if (tok == LLAMA_TOKEN_NULL) {
            throw std::runtime_error("Invalid token");
        }
        tokens.emplace_back(tok);
    }

    // will create a copy of the chunk if it contains non-text data
    void push_back(const mtmd_input_chunk * chunk) {
        auto type = mtmd_input_chunk_get_type(chunk);
        if (type == MTMD_INPUT_CHUNK_TYPE_IMAGE || type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
            GGML_ASSERT(has_mtmd);
            const int n_pos = mtmd_input_chunk_get_n_pos(chunk);
            llama_pos start_pos = tokens.size();
            for (int i = 0; i < n_pos; ++i) {
                tokens.emplace_back(LLAMA_TOKEN_NULL);
            }
            mtmd::input_chunk_ptr new_chunk(mtmd_input_chunk_copy(chunk));
            map_pos_to_media[start_pos] = std::move(new_chunk);
        } else if (type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t n_tokens;
            const auto * text_tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_tokens);
            for (size_t i = 0; i < n_tokens; ++i) {
                push_back(text_tokens[i]);
            }
        } else {
            GGML_ABORT("Invalid chunk type");
        }
    }

    // appends server tokens, updates the media map. copies media chunks.
    void push_back(server_tokens & tokens) {
        size_t start_pos = size();
        for (size_t i = 0; i < tokens.size(); i++) {
            push_back(tokens[i]);
        }
        if (tokens.has_mtmd) {
            // Assert if we are copying MTMD chunks to a server_tokens that does not have mtmd.
            // We could also just check, but this will prevent silently dropping MTMD data.
            GGML_ASSERT(has_mtmd);
            for (auto it = tokens.map_pos_to_media.begin(); it != tokens.map_pos_to_media.end(); ) {
                auto * chunk = tokens.map_pos_to_media[it->first].get();
                mtmd::input_chunk_ptr new_chunk(mtmd_input_chunk_copy(chunk));
                map_pos_to_media[start_pos+it->first] = std::move(new_chunk);
            }
        }
    }

    void push_back_all_to(server_tokens & dst) const {
        GGML_ASSERT(!has_mtmd || dst.has_mtmd);
        for (size_t i = 0; i < tokens.size(); ++i) {
            const llama_token tok = tokens[i];
            if (tok != LLAMA_TOKEN_NULL) {
                dst.push_back(tok);
                continue;
            }

            const auto & chunk = find_chunk(i);
            dst.push_back(chunk.get());
            i += mtmd_input_chunk_get_n_pos(chunk.get()) - 1;
        }
    }

    // for compatibility with context shift and prompt truncation
    void insert(const llama_tokens & inp_tokens) {
        GGML_ASSERT(!has_mtmd); // only allow this if mtmd is disabled
        tokens.insert(tokens.end(), inp_tokens.begin(), inp_tokens.end());
    }

    // for compatibility with speculative decoding, ctx shift, slot save/load
    const llama_tokens & get_text_tokens() const {
        GGML_ASSERT(!has_mtmd); // only allow this if mtmd is disabled
        return tokens;
    }

    // for compatibility with speculative decoding
    void set_token(llama_pos pos, llama_token id) {
        GGML_ASSERT(!has_mtmd); // only allow this if mtmd is disabled
        tokens[pos] = id;
    }

    size_t size() const {
        return tokens.size();
    }

    bool has_media() const {
        return has_mtmd && !map_pos_to_media.empty();
    }

    bool empty() const {
        return tokens.empty();
    }

    void clear() {
        tokens.clear();
    }

    void keep_first(size_t n) {
        GGML_ASSERT(n <= tokens.size());
        if (has_mtmd) {
            if (n == tokens.size()) {
                return; // nothing to do
            }
            // we throw an error if we try to remove a token in the middle of an image
            // for ex. with input of 5 text tokens and 2 images:
            //    [0] [1] [2] [3] [4] [img0] [img0] [img0] [img1] [img1]
            // n  1   2   3   4   5   6      7      8      9      10
            // allowed to resize      ^                    ^
            // disallowed to resize          ^      ^             ^
            if (n > 0) {
                // make sure we never remove tokens in the middle of an image
                // note that the case where we keep a full image at the end is allowed:
                //   tokens[n - 1] == LLAMA_TOKEN_NULL && tokens[n] != LLAMA_TOKEN_NULL
                if (tokens[n - 1] == LLAMA_TOKEN_NULL && tokens[n] == LLAMA_TOKEN_NULL) {
                    find_chunk(n - 1); // will throw an error if the token is not begin-of-chunk
                }
            }
            // remove all image chunks that are not used anymore
            for (auto it = map_pos_to_media.begin(); it != map_pos_to_media.end(); ) {
                llama_pos pos = it->first;
                if (pos >= (llama_pos)n) {
                    it = map_pos_to_media.erase(it);
                } else {
                    ++it;
                }
            }
        }
        tokens.resize(n);
    }

    std::string detokenize(const llama_context * ctx, bool special) const {
        llama_tokens text_tokens;
        text_tokens.reserve(tokens.size());
        for (const auto & t : tokens) {
            if (t != LLAMA_TOKEN_NULL) {
                text_tokens.push_back(t);
            }
        }
        return common_detokenize(ctx, text_tokens, special);
    }

    size_t get_common_prefix(const server_tokens & b) const {
        const size_t max_idx = std::min(tokens.size(), b.tokens.size());

        if (!has_mtmd) {
            for (size_t i = 0; i < max_idx; ++i) {
                if (tokens[i] == b.tokens[i]) {
                    continue;
                }

                return i;
            }

            return max_idx;
        }

        for (size_t i = 0; i < max_idx; ++i) {
            const llama_token ai =   tokens[i];
            const llama_token bi = b.tokens[i];

            if (ai == LLAMA_TOKEN_NULL && bi == LLAMA_TOKEN_NULL) {
                const auto & a_chunk =   find_chunk(i);
                const auto & b_chunk = b.find_chunk(i);

                GGML_ASSERT(a_chunk && b_chunk);

                const std::string id_ai = mtmd_input_chunk_get_id(a_chunk.get());
                const std::string id_bi = mtmd_input_chunk_get_id(b_chunk.get());

                const size_t pos_a = mtmd_input_chunk_get_n_pos(a_chunk.get());
                const size_t pos_b = mtmd_input_chunk_get_n_pos(b_chunk.get());

                if (id_ai == id_bi && pos_a == pos_b) {
                    GGML_ASSERT(pos_a > 0 && "Invalid media chunk"); // should never happen
                    i += pos_a - 1; // will be +1 by the for loop
                    continue;
                }

                return i;
            }

            if (ai == bi) {
                continue;
            }

            return i;
        }

        return max_idx; // all tokens are equal
    }

    // make sure all text tokens are within the vocab range
    bool validate(const struct llama_context * ctx) const {
        const llama_model * model = llama_get_model(ctx);
        const llama_vocab * vocab = llama_model_get_vocab(model);
        const int32_t n_vocab = llama_vocab_n_tokens(vocab);

        for (size_t i = 0; i < tokens.size(); ++i) {
            const auto & t = tokens[i];
            if (t == LLAMA_TOKEN_NULL) {
                try {
                    const auto & chunk = find_chunk(i);
                    size_t n_pos = mtmd_input_chunk_get_n_pos(chunk.get());
                    i += n_pos - 1; // will be +1 by the for loop
                } catch (const std::exception & e) {
                    return false;
                }
            } else if (t < 0 || t >= n_vocab) {
                return false;
            }
        }
        return true;
    }

    // encode and decode the image chunk
    int32_t process_chunk(
                llama_context * ctx,
                mtmd_context * mctx,
                llama_pos n_past,
                int32_t seq_id,
                llama_pos & n_pos_out,
                server_mtmd_prefill_profile * profile = nullptr) const {
        const auto & chunk = find_chunk(n_past);
        const char * name = mtmd_input_chunk_get_type(chunk.get()) == MTMD_INPUT_CHUNK_TYPE_IMAGE
                            ? "image" : "audio";
        SRV_INF("processing %s...\n", name);
        int32_t n_batch = llama_n_batch(ctx);
        const int64_t process_start_ms = ggml_time_ms();
        const bool collect_profile = profile != nullptr && profile->enabled;
        const int64_t process_start_us = collect_profile ? ggml_time_us() : 0;

        int32_t result = 0;
        llama_pos new_n_past = n_past;

	        if (mtmd_input_chunk_get_type(chunk.get()) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
	            if (!server_run_npu_overlay_switch_cmd("AICAS_NPU_PREFILL_SWITCH_CMD", "prefill")) {
	                n_pos_out = n_past;
	                return -1;
	            }
	            result = mtmd_helper_eval_chunk_single(mctx, ctx,
	                chunk.get(),
	                n_past,
                seq_id,
                n_batch,
                true,
	                &new_n_past);
	        } else {
	            if (!server_run_npu_overlay_switch_cmd("AICAS_NPU_PREFILL_SWITCH_CMD", "prefill")) {
	                n_pos_out = n_past;
	                return -1;
	            }
	#ifdef GGML_USE_NPU
	            const bool collect_npu_summary = profile != nullptr && profile->enabled;
            ggml_npu_profile_summary npu_summary = {};
            if (collect_npu_summary) {
                ggml_backend_npu_profile_summary_start();
            }
#endif

            const int64_t encode_start_us = collect_profile ? ggml_time_us() : 0;
            const bool collect_cpu_backend_profile = collect_profile && server_mtmd_cpu_op_profile_enabled();
            if (collect_cpu_backend_profile) {
                ggml_backend_cpu_profile_start();
            }
            result = mtmd_encode_chunk(mctx, chunk.get());
            const int64_t encode_us = collect_profile ? (ggml_time_us() - encode_start_us) : 0;
            const char * cpu_backend_profile_json =
                collect_cpu_backend_profile ? ggml_backend_cpu_profile_stop_json() : nullptr;
            const char * mmproj_summary_json = collect_profile ? mtmd_get_last_mmproj_summary_json(mctx) : nullptr;

#ifdef GGML_USE_NPU
            if (profile != nullptr && profile->enabled) {
                ggml_backend_npu_profile_summary_stop(&npu_summary);
            }
#endif

            if (result == 0) {
                float * embd = mtmd_get_output_embd(mctx);
                const int64_t decode_start_us = collect_profile ? ggml_time_us() : 0;
                result = mtmd_helper_decode_image_chunk(
                    mctx,
                    ctx,
                    chunk.get(),
                    embd,
                    n_past,
                    seq_id,
                    n_batch,
                    &new_n_past);
                const int64_t decode_us = collect_profile ? (ggml_time_us() - decode_start_us) : 0;

                if (result == 0 && collect_profile) {
                    profile->add_chunk(
                        chunk.get(),
                        encode_us,
                        decode_us,
                        ggml_time_us() - process_start_us,
                        mmproj_summary_json,
                        cpu_backend_profile_json
#ifdef GGML_USE_NPU
                        , npu_summary
#endif
                        );
                }
            }
        }

        SRV_INF("%s processed in %" PRId64 " ms\n", name, ggml_time_ms() - process_start_ms);
        if (result != 0) {
            LOG_ERR("mtmd_helper_eval failed with status %d", result);
            n_pos_out = n_past;
            return result;
        }

        n_pos_out = new_n_past;
        return 0;
    }

    bool process_merged_prefill(
                llama_context * ctx,
                mtmd_context * mctx,
                int32_t seq_id,
                llama_pos & n_pos_out,
                int32_t & result,
                server_mtmd_prefill_profile * profile = nullptr) const {
        if (!server_mtmd_merge_prefill_enabled() || !has_media() || mctx == nullptr) {
            return false;
        }
        if (mtmd_decode_use_mrope(mctx) || mtmd_decode_use_non_causal(mctx)) {
            return false;
        }
        if (tokens.empty() || tokens.size() > (size_t) llama_n_batch(ctx)) {
            return false;
        }
        if (!server_run_npu_overlay_switch_cmd("AICAS_NPU_PREFILL_SWITCH_CMD", "prefill")) {
            result = -1;
            return true;
        }

        const llama_model * model = llama_get_model(ctx);
        const int32_t n_embd = llama_model_n_embd(model);
        std::vector<float> embd_all(tokens.size() * (size_t) n_embd);

        size_t cursor = 0;
        int64_t text_token_count = 0;
        for (size_t i = 0; i < tokens.size(); ++i) {
            const llama_token tok = tokens[i];
            if (tok != LLAMA_TOKEN_NULL) {
                float * out = embd_all.data() + cursor * (size_t) n_embd;
                if (!llama_model_get_token_embedding(model, tok, out, n_embd)) {
                    SRV_INF("%s", "merged prefill disabled: token embedding lookup is unsupported for this model\n");
                    return false;
                }
                ++cursor;
                ++text_token_count;
                continue;
            }

            const auto & chunk = find_chunk(i);
            const mtmd_input_chunk_type type = mtmd_input_chunk_get_type(chunk.get());
            if (type != MTMD_INPUT_CHUNK_TYPE_IMAGE && type != MTMD_INPUT_CHUNK_TYPE_AUDIO) {
                return false;
            }

            const int32_t n_pos = mtmd_input_chunk_get_n_pos(chunk.get());
            const int32_t n_tokens = mtmd_input_chunk_get_n_tokens(chunk.get());
            if (n_pos != n_tokens || i + (size_t) n_pos > tokens.size()) {
                return false;
            }

            const char * name = type == MTMD_INPUT_CHUNK_TYPE_IMAGE ? "image" : "audio";
            const int64_t t0 = ggml_time_ms();
            SRV_INF("encoding %s slice for merged prefill...\n", name);
            const bool collect_profile = profile != nullptr && profile->enabled;
#ifdef GGML_USE_NPU
            const bool collect_npu_summary = collect_profile;
            ggml_npu_profile_summary npu_summary = {};
            if (collect_npu_summary) {
                ggml_backend_npu_profile_summary_start();
            }
#endif
            const int64_t encode_start_us = collect_profile ? ggml_time_us() : 0;
            const bool collect_cpu_backend_profile = collect_profile && server_mtmd_cpu_op_profile_enabled();
            if (collect_cpu_backend_profile) {
                ggml_backend_cpu_profile_start();
            }
            result = mtmd_encode_chunk(mctx, chunk.get());
            const int64_t encode_us = collect_profile ? (ggml_time_us() - encode_start_us) : 0;
            const char * cpu_backend_profile_json =
                collect_cpu_backend_profile ? ggml_backend_cpu_profile_stop_json() : nullptr;
            const char * mmproj_summary_json = collect_profile ? mtmd_get_last_mmproj_summary_json(mctx) : nullptr;
#ifdef GGML_USE_NPU
            if (collect_npu_summary) {
                ggml_backend_npu_profile_summary_stop(&npu_summary);
            }
#endif
            if (result != 0) {
                SRV_ERR("failed to encode %s slice for merged prefill\n", name);
                return true;
            }
            SRV_INF("%s slice encoded for merged prefill in %" PRId64 " ms\n", name, ggml_time_ms() - t0);

            if (profile != nullptr && profile->enabled) {
                profile->add_chunk(
                        chunk.get(),
                        encode_us,
                        0,
                        encode_us,
                        mmproj_summary_json,
                        cpu_backend_profile_json
#ifdef GGML_USE_NPU
                        , npu_summary
#endif
                        );
            }

            float * embd = mtmd_get_output_embd(mctx);
            if (!embd) {
                SRV_ERR("failed to get %s embeddings for merged prefill\n", name);
                result = -1;
                return true;
            }

            std::copy(embd, embd + (size_t) n_tokens * n_embd, embd_all.begin() + cursor * (size_t) n_embd);
            cursor += n_tokens;
            i += (size_t) n_pos - 1;
        }

        if (cursor != tokens.size()) {
            return false;
        }

        std::vector<llama_pos> pos(tokens.size());
        std::vector<int32_t> n_seq_id(tokens.size(), 1);
        std::vector<llama_seq_id> seq_id_0(1, seq_id);
        std::vector<llama_seq_id *> seq_ids(tokens.size() + 1);
        std::vector<int8_t> logits(tokens.size(), false);

        for (size_t i = 0; i < tokens.size(); ++i) {
            pos[i] = (llama_pos) i;
            seq_ids[i] = seq_id_0.data();
        }
        seq_ids[tokens.size()] = nullptr;
        logits.back() = true;

        llama_batch batch = {
            /*n_tokens       =*/ (int32_t) tokens.size(),
            /*tokens         =*/ nullptr,
            /*embd           =*/ embd_all.data(),
            /*pos            =*/ pos.data(),
            /*n_seq_id       =*/ n_seq_id.data(),
            /*seq_id         =*/ seq_ids.data(),
            /*logits         =*/ logits.data(),
        };

        SRV_INF("decoding merged multimodal prefill, n_tokens = %zu\n", tokens.size());
        const int64_t t1 = ggml_time_ms();
        const int64_t decode_start_us = profile != nullptr && profile->enabled ? ggml_time_us() : 0;
#ifdef GGML_USE_NPU
        const bool collect_text_npu_summary = profile != nullptr && profile->enabled;
        ggml_npu_profile_summary text_npu_summary = {};
        if (collect_text_npu_summary) {
            ggml_backend_npu_profile_summary_start();
        }
#endif
        const bool collect_text_cpu_profile = server_text_cpu_profile_enabled();
        const int64_t text_profile_start_us = collect_text_cpu_profile ? ggml_time_us() : 0;
        if (collect_text_cpu_profile) {
            ggml_backend_cpu_profile_start();
        }
        result = llama_decode(ctx, batch);
        const char * text_cpu_profile_json =
            collect_text_cpu_profile ? ggml_backend_cpu_profile_stop_json() : nullptr;
#ifdef GGML_USE_NPU
        if (collect_text_npu_summary) {
            ggml_backend_npu_profile_summary_stop(&text_npu_summary);
        }
#endif
        if (collect_text_cpu_profile) {
            server_text_cpu_profile_write(
                    "merged_prefill",
                    (int32_t) tokens.size(),
                    seq_id,
                    ggml_time_us() - text_profile_start_us,
                    text_cpu_profile_json);
        }
        const int64_t decode_us = profile != nullptr && profile->enabled ? (ggml_time_us() - decode_start_us) : 0;
        if (result != 0) {
            SRV_ERR("failed to decode merged multimodal prefill, res = %d\n", result);
            return true;
        }

        SRV_INF("merged multimodal prefill decoded in %" PRId64 " ms\n", ggml_time_ms() - t1);
        if (!server_run_npu_overlay_switch_cmd("AICAS_NPU_DECODE_SWITCH_CMD", "decode")) {
            SRV_ERR("%s", "failed to switch NPU overlay for decode\n");
            result = -1;
            return true;
        }
        if (profile != nullptr && profile->enabled) {
            profile->set_merged_prefill(
                    (int64_t) tokens.size(),
                    text_token_count,
                    decode_us
#ifdef GGML_USE_NPU
                    , &text_npu_summary
#endif
                    );
        }
        n_pos_out = (llama_pos) tokens.size();
        return true;
    }
};

// Computes FNV-1a hash of the data
static std::string fnv_hash(const uint8_t * data, size_t len) {
    const uint64_t fnv_prime = 0x100000001b3ULL;
    uint64_t hash = 0xcbf29ce484222325ULL;

    for (size_t i = 0; i < len; ++i) {
        hash ^= data[i];
        hash *= fnv_prime;
    }
    return std::to_string(hash);
}

static server_tokens process_mtmd_prompt(mtmd_context * mctx, std::string prompt, std::vector<raw_buffer> files) {
    mtmd::bitmaps bitmaps;
    for (auto & file : files) {
        mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_buf(mctx, file.data(), file.size()));
        if (!bmp.ptr) {
            throw std::runtime_error("Failed to load image or audio file");
        }
        // calculate bitmap hash (for KV caching)
        std::string hash = fnv_hash(bmp.data(), bmp.n_bytes());
        bmp.set_id(hash.c_str());
        bitmaps.entries.push_back(std::move(bmp));
    }
    // process prompt
    std::vector<server_tokens> inputs;
    // multimodal
    mtmd_input_text inp_txt = {
        prompt.c_str(),
        /* add_special */   true,
        /* parse_special */ true,
    };
    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    auto bitmaps_c_ptr = bitmaps.c_ptr();
    int32_t tokenized = mtmd_tokenize(mctx,
                                      chunks.ptr.get(),
                                      &inp_txt,
                                      bitmaps_c_ptr.data(),
                                      bitmaps_c_ptr.size());
    if (tokenized != 0) {
        throw std::runtime_error("Failed to tokenize prompt");
    }
    auto result = server_tokens(chunks, true);
    return result;
}

/**
 * break the input "prompt" object into multiple prompt if needed, then tokenize them
 * use tokenize_input_prompts() if the input could be an array.
 * this supports these cases:
 * - "prompt": "string"
 * - "prompt": [12, 34, 56]
 * - "prompt": [12, 34, "string", 56, 78]
 * - "prompt": { "prompt_string": "string", "multimodal_data": [ "base64" ] }
 */
static server_tokens tokenize_input_subprompt(const llama_vocab * vocab, mtmd_context * mctx, const json & json_prompt, bool add_special, bool parse_special) {
    constexpr char JSON_STRING_PROMPT_KEY[] = "prompt_string";
    constexpr char JSON_MTMD_DATA_KEY[] = "multimodal_data";
    const bool has_mtmd = mctx != nullptr;
    if (json_prompt.is_string() || json_is_array_of_mixed_numbers_strings(json_prompt)) {
        // string or mixed
        llama_tokens tmp = tokenize_mixed(vocab, json_prompt, add_special, parse_special);
        return server_tokens(tmp, false);
    } else if (json_is_array_of_numbers(json_prompt)) {
        // array of tokens
        llama_tokens tmp = json_prompt.get<llama_tokens>();
        return server_tokens(tmp, false);
    } else if (json_prompt.contains(JSON_STRING_PROMPT_KEY)) {
        // JSON object with prompt key.
        if (json_prompt.contains(JSON_MTMD_DATA_KEY)) {
            if (!has_mtmd)
                throw std::runtime_error("Multimodal data provided, but model does not support multimodal requests.");

            // JSON object with prompt and multimodal key.
            std::vector<raw_buffer> files;
            for (const auto & entry : json_prompt.at(JSON_MTMD_DATA_KEY)) {
                files.push_back(base64_decode(entry));
            }
            return process_mtmd_prompt(mctx, json_prompt.at(JSON_STRING_PROMPT_KEY), files);
        } else {
            // Not multimodal, but contains a subobject.
            llama_tokens tmp = tokenize_mixed(vocab, json_prompt.at(JSON_STRING_PROMPT_KEY), add_special, parse_special);
            return server_tokens(tmp, false);
        }
   } else {
       throw std::runtime_error("\"prompt\" elements must be a string, a list of tokens, a JSON object containing a prompt string, or a list of mixed strings & tokens.");
   }
}

/**
 * break the input "prompt" object into multiple prompt if needed, then tokenize them
 * this supports these cases:
 * - "prompt": "string"
 * - "prompt": [12, 34, 56]
 * - "prompt": [12, 34, "string", 56, 78]
 * - "prompt": { "prompt_string": "string", "multimodal_data": [ "base64" ] }
 * and multiple prompts (multi-tasks):
 * - "prompt": ["string1", "string2"]
 * - "prompt": ["string1", [12, 34, 56]]
 * - "prompt": [[12, 34, 56], [78, 90, 12]]
 * - "prompt": [[12, 34, "string", 56, 78], [12, 34, 56], { "prompt_string": "string", "multimodal_data": [ "base64" ]}]
 */
static std::vector<server_tokens> tokenize_input_prompts(const llama_vocab * vocab, mtmd_context * mctx, const json & json_prompt, bool add_special, bool parse_special) {
    std::vector<server_tokens> result;
    if (json_prompt.is_array() && !json_is_array_and_contains_numbers(json_prompt)) {
        result.reserve(json_prompt.size());
        for (const auto & p : json_prompt) {
            result.push_back(tokenize_input_subprompt(vocab, mctx, p,add_special, parse_special));
        }
    } else {
        result.push_back(tokenize_input_subprompt(vocab, mctx, json_prompt, add_special, parse_special));
    }
    if (result.empty()) {
        throw std::runtime_error("\"prompt\" must not be empty");
    }
    return result;
}

// format rerank task: [BOS]query[EOS][SEP]doc[EOS].
static server_tokens format_rerank(const struct llama_model * model, const struct llama_vocab * vocab, mtmd_context * mctx, const std::string & query, const std::string & doc) {
    server_tokens result = {};

    const char * rerank_prompt = llama_model_chat_template(model, "rerank");

    if (rerank_prompt != nullptr) {
        std::string prompt = rerank_prompt;
        string_replace_all(prompt, "{query}"   , query);
        string_replace_all(prompt, "{document}", doc  );
        server_tokens tokens = tokenize_input_subprompt(vocab, mctx, prompt, false, true);
        result.push_back(tokens);
    } else {
        // Get EOS token - use SEP token as fallback if EOS is not available
        server_tokens query_tokens = tokenize_input_subprompt(vocab, mctx, query, false, false);
        server_tokens doc_tokens   = tokenize_input_subprompt(vocab, mctx, doc,   false, false);
        llama_token eos_token = llama_vocab_eos(vocab);
        if (eos_token == LLAMA_TOKEN_NULL) {
            eos_token = llama_vocab_sep(vocab);
        }

        if (llama_vocab_get_add_bos(vocab)) {
            result.push_back(llama_vocab_bos(vocab));
        }
        result.push_back(query_tokens);
        if (llama_vocab_get_add_eos(vocab)) {
            result.push_back(eos_token);
        }
        if (llama_vocab_get_add_sep(vocab)) {
            result.push_back(llama_vocab_sep(vocab));
        }
        result.push_back(doc_tokens);
        if (llama_vocab_get_add_eos(vocab)) {
            result.push_back(eos_token);
        }
    }

    return result;
}
