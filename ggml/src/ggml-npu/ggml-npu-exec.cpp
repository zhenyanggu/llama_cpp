#include "ggml-npu-exec.h"

#include "ggml-npu-plan.h"
#include "ggml-npu-profile.h"
#include "ggml-npu-quant.h"

#include "ggml-impl.h"
#include "npu_runtime.h"

#include <algorithm>
#include <cinttypes>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <limits>
#include <unordered_map>
#include <vector>

namespace ggml_npu {

static bool npu_debug_log_enabled() {
    return std::getenv("GGML_NPU_DEBUG_LOG") != nullptr || std::getenv("AICAS_MMPROJ_W8A8_DEBUG") != nullptr;
}

static bool npu_runtime_profile_requested() {
    const char * path = std::getenv("NPU_PROFILE_OUT");
    return path != nullptr && path[0] != '\0';
}

static std::string npu_stage_name(npu_loop_stage stage);

static const char * npu_tile_dump_dir() {
    const char * dir = std::getenv("GGML_NPU_DUMP_TILE_DIR");
    return (dir != nullptr && dir[0] != '\0') ? dir : nullptr;
}

static int64_t npu_dump_tile_layer_id() {
    const char * v = std::getenv("GGML_NPU_DUMP_TILE_LAYER_ID");
    return (v != nullptr && v[0] != '\0') ? std::strtoll(v, nullptr, 10) : -1;
}

static int64_t npu_dump_tile_index() {
    const char * v = std::getenv("GGML_NPU_DUMP_TILE_INDEX");
    return (v != nullptr && v[0] != '\0') ? std::strtoll(v, nullptr, 10) : -1;
}

static bool npu_should_dump_tile(int64_t layer_id, int64_t tile_index) {
    const char * dir = npu_tile_dump_dir();
    if (dir == nullptr) {
        return false;
    }
    const int64_t match_layer = npu_dump_tile_layer_id();
    const int64_t match_tile = npu_dump_tile_index();
    return (match_layer < 0 || match_layer == layer_id) &&
        (match_tile < 0 || match_tile == tile_index);
}

static void npu_dump_tile_blob(
        const char * dir,
        int64_t layer_id,
        int64_t tile_index,
        const char * suffix,
        const void * data,
        size_t size) {
    if (dir == nullptr || data == nullptr) {
        return;
    }
    char path[512];
    std::snprintf(path, sizeof(path), "%s/layer%" PRId64 "_tile%" PRId64 "_%s.bin",
        dir, layer_id, tile_index, suffix);
    FILE * fp = std::fopen(path, "wb");
    if (fp == nullptr) {
        return;
    }
    std::fwrite(data, 1, size, fp);
    std::fclose(fp);
}

static void npu_dump_tile_meta(
        const char * dir,
        int64_t layer_id,
        int64_t tile_index,
        const npu_exec_tile & exec_tile,
        float activation_scale,
        int32_t act_zp_i8,
        int32_t act_zp_u8,
        bool raw_acc_mvout,
        bool mvout_per_channel,
        bool apply_compensation) {
    if (dir == nullptr) {
        return;
    }
    char path[512];
    std::snprintf(path, sizeof(path), "%s/layer%" PRId64 "_tile%" PRId64 "_meta.txt",
        dir, layer_id, tile_index);
    FILE * fp = std::fopen(path, "w");
    if (fp == nullptr) {
        return;
    }
    std::fprintf(fp,
        "m0=%" PRId64 "\n"
        "n0=%" PRId64 "\n"
        "k0=%" PRId64 "\n"
        "m=%" PRId64 "\n"
        "n=%" PRId64 "\n"
        "k=%" PRId64 "\n"
        "activation_scale=%.9g\n"
        "act_zp_i8=%d\n"
        "act_zp_u8=%d\n"
        "stage=%s\n"
        "needs_bias=%d\n"
        "writes_output=%d\n"
        "raw_acc_mvout=%d\n"
        "mvout_per_channel=%d\n"
        "apply_compensation=%d\n",
        exec_tile.m0, exec_tile.n0, exec_tile.k0,
        exec_tile.m, exec_tile.n, exec_tile.k,
        activation_scale,
        act_zp_i8,
        act_zp_u8,
        npu_stage_name(exec_tile.stage).c_str(),
        exec_tile.needs_bias ? 1 : 0,
        exec_tile.writes_output ? 1 : 0,
        raw_acc_mvout ? 1 : 0,
        mvout_per_channel ? 1 : 0,
        apply_compensation ? 1 : 0);
    std::fclose(fp);
}

struct npu_exec_summary {
    npu_profile_summary_delta delta;
};

static std::string npu_stage_name(npu_loop_stage stage) {
    switch (stage) {
        case npu_loop_stage::single: return "single";
        case npu_loop_stage::head:   return "head";
        case npu_loop_stage::body:   return "body";
        case npu_loop_stage::tail:   return "tail";
    }

    return "unknown";
}

static uint32_t npu_float_to_q8_24_u32(float scale) {
    const double scaled = std::nearbyint(static_cast<double>(scale) * static_cast<double>(1u << 24));
    if (scaled > static_cast<double>(INT32_MAX)) {
        return static_cast<uint32_t>(INT32_MAX);
    }
    if (scaled < static_cast<double>(INT32_MIN)) {
        return static_cast<uint32_t>(INT32_MIN);
    }
    return static_cast<uint32_t>(static_cast<int32_t>(scaled));
}

static float npu_read_bias_value_f32_exec(
        const struct ggml_tensor * bias,
        int64_t m,
        int64_t n) {
    if (bias == nullptr) {
        return 0.0f;
    }

    const int64_t i0 = bias->ne[0] == 1 ? 0 : m;
    const int64_t i1 = bias->ne[1] == 1 ? 0 : n;
    const char * base = static_cast<const char *>(bias->data) + i1 * bias->nb[1] + i0 * bias->nb[0];

    switch (bias->type) {
        case GGML_TYPE_F32:
            return *reinterpret_cast<const float *>(base);
        case GGML_TYPE_F16:
            return ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t *>(base));
        default:
            return 0.0f;
    }
}

static bool npu_dst_is_dense_f32(const struct ggml_tensor * dst) {
    return dst != nullptr &&
        dst->type == GGML_TYPE_F32 &&
        dst->nb[0] == static_cast<int64_t>(sizeof(float));
}

static void npu_prepare_bias_tile_f32_exec(
        const struct ggml_tensor * bias,
        int64_t m0,
        int64_t n0,
        int64_t m_count,
        int64_t n_count,
        std::vector<float> & bias_tile) {
    bias_tile.clear();
    if (bias == nullptr || m_count <= 0 || n_count <= 0) {
        return;
    }

    const size_t tile_elems = static_cast<size_t>(m_count * n_count);
    bias_tile.resize(tile_elems);

    if (bias->type == GGML_TYPE_F32 && bias->nb[0] == static_cast<int64_t>(sizeof(float))) {
        if (bias->ne[1] == 1) {
            const int64_t src_m0 = bias->ne[0] == 1 ? 0 : m0;
            const float * src = reinterpret_cast<const float *>(static_cast<const char *>(bias->data) + src_m0 * bias->nb[0]);
            for (int64_t n = 0; n < n_count; ++n) {
                std::memcpy(
                    bias_tile.data() + static_cast<size_t>(n * m_count),
                    src,
                    static_cast<size_t>(m_count * sizeof(float)));
            }
            return;
        }

        if (bias->ne[0] == 1) {
            for (int64_t n = 0; n < n_count; ++n) {
                const float value = *reinterpret_cast<const float *>(static_cast<const char *>(bias->data) + (bias->ne[1] == 1 ? 0 : (n0 + n) * bias->nb[1]));
                std::fill_n(
                    bias_tile.data() + static_cast<size_t>(n * m_count),
                    static_cast<size_t>(m_count),
                    value);
            }
            return;
        }

        if (bias->nb[1] == bias->ne[0] * bias->nb[0]) {
            for (int64_t n = 0; n < n_count; ++n) {
                const float * src = reinterpret_cast<const float *>(
                    static_cast<const char *>(bias->data) +
                    (n0 + n) * bias->nb[1] +
                    m0 * bias->nb[0]);
                std::memcpy(
                    bias_tile.data() + static_cast<size_t>(n * m_count),
                    src,
                    static_cast<size_t>(m_count * sizeof(float)));
            }
            return;
        }
    }

    for (int64_t n = 0; n < n_count; ++n) {
        for (int64_t m = 0; m < m_count; ++m) {
            bias_tile[static_cast<size_t>(n * m_count + m)] =
                npu_read_bias_value_f32_exec(bias, m0 + m, n0 + n);
        }
    }
}

static int32_t npu_effective_activation_q(const npu_node_plan & plan, int8_t raw) {
    if (plan.activation_quant.valid && !plan.activation_quant.symmetric) {
        return static_cast<int32_t>(static_cast<uint8_t>(raw)) - 128;
    }
    return static_cast<int32_t>(raw);
}

static const char * npu_bias_mode_name(npu_bias_mode mode) {
    switch (mode) {
        case npu_bias_mode::auto_select: return "auto";
        case npu_bias_mode::precomp:     return "precomp";
        case npu_bias_mode::raw:         return "raw";
    }
    return "auto";
}

static npu_bias_mode npu_bias_mode_from_env() {
    const char * v = std::getenv("GGML_NPU_AICAS_BIAS_MODE");
    if (v == nullptr || v[0] == '\0') {
        return npu_bias_mode::auto_select;
    }
    if (std::strcmp(v, "precomp") == 0) {
        return npu_bias_mode::precomp;
    }
    if (std::strcmp(v, "raw") == 0) {
        return npu_bias_mode::raw;
    }
    return npu_bias_mode::auto_select;
}

static npu_bias_mode npu_resolve_bias_mode(const npu_node_plan & plan) {
    const npu_bias_mode env_mode = npu_bias_mode_from_env();
    if (env_mode != npu_bias_mode::auto_select) {
        return env_mode;
    }
    if (plan.bias_mode != npu_bias_mode::auto_select) {
        return plan.bias_mode;
    }
    return npu_bias_mode::precomp;
}

static bool npu_can_fold_output_reconstruction(const npu_node_plan & plan) {
    return plan.aicas_w8a8.valid && !plan.aicas_w8a8.weight_scale.empty();
}

static bool npu_force_raw_acc_mvout() {
    const char * raw_mvout = std::getenv("GGML_NPU_FORCE_RAW_ACC_MVOUT");
    return raw_mvout != nullptr && raw_mvout[0] != '\0' && std::strcmp(raw_mvout, "0") != 0;
}

struct npu_tile_align_debug_cfg {
    bool enabled = false;
    int64_t layer_id = -1;
    int64_t tile_index = -1;
    int64_t max_logs = 1;
    float abs_tol = 1e-3f;
};

static int64_t npu_env_i64(const char * name, int64_t default_v) {
    const char * v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') {
        return default_v;
    }
    char * end = nullptr;
    const long long parsed = std::strtoll(v, &end, 10);
    if (end == v) {
        return default_v;
    }
    return static_cast<int64_t>(parsed);
}

static float npu_env_f32(const char * name, float default_v) {
    const char * v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') {
        return default_v;
    }
    char * end = nullptr;
    const float parsed = std::strtof(v, &end);
    if (end == v) {
        return default_v;
    }
    return parsed;
}

static npu_tile_align_debug_cfg npu_get_tile_align_debug_cfg() {
    npu_tile_align_debug_cfg cfg;
    cfg.enabled = std::getenv("GGML_NPU_TILE_ALIGN_DEBUG") != nullptr;
    if (!cfg.enabled) {
        return cfg;
    }
    cfg.layer_id = npu_env_i64("GGML_NPU_TILE_ALIGN_LAYER_ID", -1);
    cfg.tile_index = npu_env_i64("GGML_NPU_TILE_ALIGN_TILE_INDEX", -1);
    cfg.max_logs = std::max<int64_t>(1, npu_env_i64("GGML_NPU_TILE_ALIGN_MAX_LOGS", 1));
    cfg.abs_tol = std::max(0.0f, npu_env_f32("GGML_NPU_TILE_ALIGN_ABS_TOL", 1e-3f));
    return cfg;
}

static bool npu_should_check_tile_align(
        const npu_tile_align_debug_cfg & cfg,
        int64_t layer_id,
        int64_t tile_index) {
    if (!cfg.enabled) {
        return false;
    }
    if (cfg.layer_id >= 0 && cfg.layer_id != layer_id) {
        return false;
    }
    if (cfg.tile_index >= 0 && cfg.tile_index != tile_index) {
        return false;
    }
    return true;
}

static bool npu_ensure_runtime(std::string * error) {
    if (npu_init() == 0) {
        return true;
    }

    if (error) {
        *error = "npu_init() 失败";
    }
    return false;
}

static bool npu_allocate_runtime_buffer(size_t bytes, void ** ptr, std::string * error) {
    *ptr = npu_mem_alloc(bytes);
    if (*ptr != nullptr) {
        return true;
    }

    if (error) {
        *error = "npu_mem_alloc() 失败";
    }
    return false;
}

static bool npu_ensure_weight_pack_cma(
        const npu_prepacked_weight & packed_weight,
        std::string * error,
        bool * copied_now) {
    if (copied_now != nullptr) {
        *copied_now = false;
    }

    if (packed_weight.packed.empty()) {
        if (error) {
            *error = "packed weight is empty";
        }
        return false;
    }

    if (packed_weight.cma_packed != nullptr) {
        return true;
    }

    void * cma_ptr = npu_mem_alloc(packed_weight.packed.size());
    if (cma_ptr == nullptr) {
        if (error) {
            *error = "npu_mem_alloc() 失败";
        }
        return false;
    }

    std::memcpy(cma_ptr, packed_weight.packed.data(), packed_weight.packed.size());
    packed_weight.cma_packed = cma_ptr;
    packed_weight.cma_bytes = packed_weight.packed.size();
    packed_weight.cma_persistent = false;
    if (copied_now != nullptr) {
        *copied_now = true;
    }
    return true;
}

struct npu_activation_tile_key {
    int64_t n0 = 0;
    int64_t n = 0;
    int64_t k0 = 0;
    int64_t k = 0;

    bool operator==(const npu_activation_tile_key & other) const {
        return n0 == other.n0 &&
               n == other.n &&
               k0 == other.k0 &&
               k == other.k;
    }
};

struct npu_activation_tile_key_hash {
    size_t operator()(const npu_activation_tile_key & key) const {
        size_t h = std::hash<int64_t>{}(key.n0);
        h ^= std::hash<int64_t>{}(key.n)  + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int64_t>{}(key.k0) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int64_t>{}(key.k)  + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

enum ggml_status npu_compute_node(const npu_node_plan & plan, int64_t layer_id, std::string * error) {
    const bool collect_detailed_profile = npu_profile_enabled();
    const bool collect_summary = npu_summary_active();
    const bool collect_stage_profile = collect_detailed_profile || collect_summary;
    const bool collect_runtime_profile = npu_runtime_profile_requested() || collect_summary;
    const char * profile_level_env = std::getenv("GGML_NPU_PROFILE_LEVEL");
    const std::string profile_level = profile_level_env ? profile_level_env : "";
    const bool collect_tile_profile = collect_detailed_profile &&
        profile_level != "diagnostic" && profile_level != "compact";

    npu_profile_node_record profile_record;
    if (collect_detailed_profile) {
        profile_record = npu_profile_init_node_record(layer_id, plan);
    }

    npu_exec_summary exec_summary;
    exec_summary.delta.node_count = 1;
    exec_summary.delta.exec_tile_count = static_cast<int64_t>(plan.exec_tiles.size());
    exec_summary.delta.weight_pack_count = static_cast<int64_t>(plan.weight_packs.size());
    exec_summary.delta.bias_pack_count = static_cast<int64_t>(plan.bias_packs.size());

    const int64_t node_start_us = ggml_time_us();
    bool runtime_profile_started = false;
    const npu_tile_align_debug_cfg tile_align_debug_cfg = npu_get_tile_align_debug_cfg();
    int64_t tile_align_logs = 0;

    auto cleanup_buffers = [](
            void * activation_buf,
            void * weight_buf,
            void * acc_buf,
            void * bias_buf,
            void * bias_cache_buf,
            void * scale_cache_buf) {
        if (activation_buf) {
            npu_mem_free(activation_buf);
        }
        if (weight_buf) {
            npu_mem_free(weight_buf);
        }
        if (acc_buf) {
            npu_mem_free(acc_buf);
        }
        if (bias_buf) {
            npu_mem_free(bias_buf);
        }
        if (bias_cache_buf) {
            npu_mem_free(bias_cache_buf);
        }
        if (scale_cache_buf) {
            npu_mem_free(scale_cache_buf);
        }
    };

    auto finalize_status = [&](enum ggml_status status) {
        if (runtime_profile_started) {
            npu_profile_end(layer_id);
            runtime_profile_started = false;
        }

        if (collect_stage_profile) {
            exec_summary.delta.total_node_us = ggml_time_us() - node_start_us;
        }

        if (collect_summary) {
            npu_summary_add_delta(exec_summary.delta);
        }

        if (collect_detailed_profile) {
            profile_record.activation_pack_calls = exec_summary.delta.activation_pack_calls;
            profile_record.host_copy_activation_calls = exec_summary.delta.host_copy_activation_calls;
            profile_record.host_copy_weight_calls = exec_summary.delta.host_copy_weight_calls;
            profile_record.bias_prepare_calls = exec_summary.delta.bias_prepare_calls;
            profile_record.dma_in_activation_calls = exec_summary.delta.dma_in_activation_calls;
            profile_record.dma_in_weight_calls = exec_summary.delta.dma_in_weight_calls;
            profile_record.dma_in_bias_calls = exec_summary.delta.dma_in_bias_calls;
            profile_record.gemm_calls = exec_summary.delta.gemm_calls;
            profile_record.dma_out_calls = exec_summary.delta.dma_out_calls;
            profile_record.postprocess_calls = exec_summary.delta.postprocess_calls;
            profile_record.packed_activation_bytes_total = exec_summary.delta.packed_activation_bytes_total;
            profile_record.copied_weight_bytes_total = exec_summary.delta.copied_weight_bytes_total;
            profile_record.bias_bytes_total = exec_summary.delta.bias_bytes_total;
            profile_record.acc_readback_bytes_total = exec_summary.delta.acc_readback_bytes_total;
            profile_record.output_write_bytes_total = exec_summary.delta.output_write_bytes_total;
            profile_record.activation_pack_us_total = exec_summary.delta.activation_pack_us_total;
            profile_record.host_copy_activation_us_total = exec_summary.delta.host_copy_activation_us_total;
            profile_record.host_copy_weight_us_total = exec_summary.delta.host_copy_weight_us_total;
            profile_record.bias_prepare_us_total = exec_summary.delta.bias_prepare_us_total;
            profile_record.dma_in_activation_us_total = exec_summary.delta.dma_in_activation_us_total;
            profile_record.dma_in_weight_us_total = exec_summary.delta.dma_in_weight_us_total;
            profile_record.dma_in_bias_us_total = exec_summary.delta.dma_in_bias_us_total;
            profile_record.dma_in_pair_calls = exec_summary.delta.dma_in_pair_calls;
            profile_record.dma_in_pair_us_total = exec_summary.delta.dma_in_pair_us_total;
            profile_record.gemm_us_total = exec_summary.delta.gemm_us_total;
            profile_record.dma_out_us_total = exec_summary.delta.dma_out_us_total;
            profile_record.postprocess_us_total = exec_summary.delta.postprocess_us_total;
            profile_record.total_node_us = static_cast<double>(exec_summary.delta.total_node_us);
            profile_record.status = status == GGML_STATUS_SUCCESS ? "success" : "failed";
            if (status != GGML_STATUS_SUCCESS && error != nullptr && !error->empty()) {
                profile_record.error = *error;
            }
            npu_profile_add_node_record(std::move(profile_record));
        }
        return status;
    };
    if (npu_debug_log_enabled()) {
        const char * root_name = plan.root && plan.root->name[0] != '\0' ? plan.root->name : "(unnamed)";
        GGML_LOG_INFO("%s: enter root=%s exec_tiles=%zu bias=%s summary=%s\n",
                __func__,
                root_name,
                plan.exec_tiles.size(),
                plan.bias ? "yes" : "no",
                plan.summary.c_str());
    }
    if (!npu_ensure_runtime(error)) {
        return finalize_status(GGML_STATUS_FAILED);
    }

    std::string aot_reason;
    if (!npu_plan_is_aot_stable(plan, &aot_reason)) {
        if (error) {
            *error = "AOT plan invalid: " + aot_reason;
        }
        return finalize_status(GGML_STATUS_FAILED);
    }

    const int64_t max_n = plan.config.sa_rows;
    const int64_t max_m = plan.config.sa_cols;
    const int64_t max_k = std::min<int64_t>(plan.config.k_block, plan.config.stage2_k_block);

    void * activation_buf = nullptr;
    void * acc_buf = nullptr;
    void * bias_buf = nullptr;
    void * bias_cache_buf = nullptr;
    void * scale_cache_buf = nullptr;

    if (!npu_allocate_runtime_buffer(static_cast<size_t>(max_n * max_k), &activation_buf, error) ||
        !npu_allocate_runtime_buffer(static_cast<size_t>(max_n * max_m * sizeof(int32_t)), &acc_buf, error) ||
        !npu_allocate_runtime_buffer(static_cast<size_t>(max_n * max_m * sizeof(int32_t)), &bias_buf, error)) {
        cleanup_buffers(activation_buf, nullptr, acc_buf, bias_buf, nullptr, nullptr);
        return finalize_status(GGML_STATUS_ALLOC_FAILED);
    }

    std::vector<int8_t> packed_activation;
    std::unordered_map<npu_activation_tile_key, std::vector<int8_t>, npu_activation_tile_key_hash> activation_tile_cache;
    const float activation_scale = plan.activation_quant.scale;
    const bool use_aicas_w8a8 = plan.aicas_w8a8.valid;
    const npu_bias_mode bias_mode = npu_resolve_bias_mode(plan);
    const bool hardware_asymmetric_activations =
        plan.activation_quant.valid && !plan.activation_quant.symmetric;
    const bool fold_output_reconstruction = npu_can_fold_output_reconstruction(plan);
    const bool apply_compensation = use_aicas_w8a8 && bias_mode == npu_bias_mode::raw;
    const bool raw_acc_mvout = npu_force_raw_acc_mvout();
    const bool mvout_per_channel = fold_output_reconstruction && plan.aicas_w8a8.weight_scale.size() > 1;
    const float per_tensor_weight_scale = fold_output_reconstruction ? plan.aicas_w8a8.weight_scale[0] : 1.0f;
    const bool use_bias_cache =
        use_aicas_w8a8 &&
        plan.bias != nullptr &&
        plan.config.layout.bias_cache.bytes >= static_cast<uint32_t>(plan.m * sizeof(int32_t));
    const bool use_scale_cache =
        !raw_acc_mvout &&
        mvout_per_channel &&
        plan.config.layout.scale_cache.bytes >=
            static_cast<uint32_t>(plan.config.sa_rows * plan.m * sizeof(uint32_t));
    std::vector<float> acc_scaled_values;
    std::vector<int32_t> acc_raw_values_host;
    std::vector<float> bias_tile_values;
    std::vector<int32_t> bias_values;

    if (use_bias_cache &&
        !npu_allocate_runtime_buffer(static_cast<size_t>(plan.config.layout.bias_cache.bytes), &bias_cache_buf, error)) {
        cleanup_buffers(activation_buf, nullptr, acc_buf, bias_buf, nullptr, nullptr);
        return finalize_status(GGML_STATUS_ALLOC_FAILED);
    }
    if (use_scale_cache &&
        !npu_allocate_runtime_buffer(static_cast<size_t>(plan.config.layout.scale_cache.bytes), &scale_cache_buf, error)) {
        cleanup_buffers(activation_buf, nullptr, acc_buf, bias_buf, bias_cache_buf, nullptr);
        return finalize_status(GGML_STATUS_ALLOC_FAILED);
    }

    activation_tile_cache.reserve(plan.exec_tiles.size());

    int64_t active_m0 = -1;
    int64_t active_n0 = -1;
    int64_t active_m = 0;
    int64_t active_n = 0;
    int64_t loaded_bias_cache_n0 = -1;
    int64_t loaded_scale_cache_n0 = -1;
    int64_t loaded_scale_cache_n = -1;
    npu_activation_tile_key loaded_activation_key;
    bool loaded_activation_valid = false;
    int32_t loaded_weight_pack_index = -1;
    if (collect_runtime_profile) {
        npu_profile_begin(layer_id);
        runtime_profile_started = true;
    }

    for (size_t exec_tile_idx = 0; exec_tile_idx < plan.exec_tiles.size(); ++exec_tile_idx) {
        const npu_exec_tile & exec_tile = plan.exec_tiles[exec_tile_idx];
        npu_profile_tile_record tile_record;
        const int64_t tile_start_us = collect_tile_profile ? ggml_time_us() : 0;
        if (collect_tile_profile) {
            tile_record.tile_index = static_cast<int64_t>(profile_record.tiles.size());
            tile_record.m0 = exec_tile.m0;
            tile_record.n0 = exec_tile.n0;
            tile_record.k0 = exec_tile.k0;
            tile_record.m = exec_tile.m;
            tile_record.n = exec_tile.n;
            tile_record.k = exec_tile.k;
            tile_record.stage = npu_stage_name(exec_tile.stage);
            tile_record.needs_bias = exec_tile.needs_bias;
            tile_record.writes_output = exec_tile.writes_output;
            tile_record.weight_pack_index = exec_tile.weight_pack_index;
            tile_record.bias_pack_index = exec_tile.bias_pack_index;
        }

        if (active_m0 != exec_tile.m0 || active_n0 != exec_tile.n0) {
            active_m0 = exec_tile.m0;
            active_n0 = exec_tile.n0;
            active_m = exec_tile.m;
            active_n = exec_tile.n;
            acc_scaled_values.assign(static_cast<size_t>(active_n * active_m), 0.0f);
            bias_values.assign(static_cast<size_t>(active_m), 0);
        }

        const npu_activation_tile_key activation_key {
            exec_tile.n0,
            exec_tile.n,
            exec_tile.k0,
            exec_tile.k,
        };

        const std::vector<int8_t> * packed_activation_view = nullptr;
        const auto cache_it = activation_tile_cache.find(activation_key);
        if (cache_it != activation_tile_cache.end()) {
            packed_activation_view = &cache_it->second;
            if (collect_tile_profile) {
                tile_record.activation_pack_us = 0.0;
            }
        } else {
            const int64_t activation_pack_start_us = collect_stage_profile ? ggml_time_us() : 0;
            if (!npu_pack_activation_tile_static_asym_i8(
                        plan.src1,
                        exec_tile.n0,
                        exec_tile.n,
                        exec_tile.k0,
                        exec_tile.k,
                        activation_scale,
                        plan.activation_quant.zero_point_u8,
                        plan.aicas_w8a8.smooth_scale.empty() ? nullptr : &plan.aicas_w8a8.smooth_scale,
                        &packed_activation,
                        error)) {
                const int64_t activation_pack_us = collect_stage_profile ? (ggml_time_us() - activation_pack_start_us) : 0;
                if (collect_stage_profile) {
                    exec_summary.delta.activation_pack_calls += 1;
                    exec_summary.delta.activation_pack_us_total += activation_pack_us;
                }
                if (collect_tile_profile) {
                    tile_record.activation_pack_us = static_cast<double>(activation_pack_us);
                    profile_record.tiles.push_back(std::move(tile_record));
                }
                cleanup_buffers(activation_buf, nullptr, acc_buf, bias_buf, bias_cache_buf, scale_cache_buf);
                return finalize_status(GGML_STATUS_FAILED);
            }
            const auto inserted = activation_tile_cache.emplace(activation_key, packed_activation);
            packed_activation_view = &inserted.first->second;
            if (collect_stage_profile) {
                const int64_t activation_pack_us = ggml_time_us() - activation_pack_start_us;
                exec_summary.delta.activation_pack_calls += 1;
                exec_summary.delta.activation_pack_us_total += activation_pack_us;
                exec_summary.delta.packed_activation_bytes_total += static_cast<int64_t>(packed_activation_view->size());
                if (collect_tile_profile) {
                    tile_record.activation_pack_us = static_cast<double>(activation_pack_us);
                }
            }
        }
        if (collect_tile_profile) {
            tile_record.activation_bytes = static_cast<int64_t>(packed_activation_view->size());
        }

        if (exec_tile.weight_pack_index < 0 ||
            static_cast<size_t>(exec_tile.weight_pack_index) >= plan.weight_packs.size()) {
            if (error) {
                *error = "weight_pack_index 非法";
            }
            if (collect_tile_profile) {
                profile_record.tiles.push_back(std::move(tile_record));
            }
            cleanup_buffers(activation_buf, nullptr, acc_buf, bias_buf, bias_cache_buf, scale_cache_buf);
            return finalize_status(GGML_STATUS_FAILED);
        }

        const npu_prepacked_weight & packed_weight = plan.weight_packs[static_cast<size_t>(exec_tile.weight_pack_index)];
        if (collect_tile_profile) {
            tile_record.weight_bytes = static_cast<int64_t>(packed_weight.packed.size());
        }

        const bool activation_already_in_spm =
            loaded_activation_valid && loaded_activation_key == activation_key;
        const bool weight_already_in_spm =
            loaded_weight_pack_index == exec_tile.weight_pack_index;

        const int64_t activation_copy_start_us = collect_stage_profile ? ggml_time_us() : 0;
        if (!activation_already_in_spm) {
            std::memcpy(activation_buf, packed_activation_view->data(), packed_activation_view->size());
        }
        if (collect_stage_profile && !activation_already_in_spm) {
            const int64_t activation_copy_us = ggml_time_us() - activation_copy_start_us;
            exec_summary.delta.host_copy_activation_calls += 1;
            exec_summary.delta.host_copy_activation_us_total += activation_copy_us;
            if (collect_tile_profile) {
                tile_record.host_copy_activation_us = static_cast<double>(activation_copy_us);
            }
        }

        bool weight_copied_now = false;
        const int64_t weight_copy_start_us = collect_stage_profile ? ggml_time_us() : 0;
        if (!weight_already_in_spm &&
                !npu_ensure_weight_pack_cma(packed_weight, error, &weight_copied_now)) {
            if (collect_tile_profile) {
                profile_record.tiles.push_back(std::move(tile_record));
            }
            cleanup_buffers(activation_buf, nullptr, acc_buf, bias_buf, bias_cache_buf, scale_cache_buf);
            return finalize_status(GGML_STATUS_FAILED);
        }
        if (collect_stage_profile && weight_copied_now) {
            const int64_t weight_copy_us = ggml_time_us() - weight_copy_start_us;
            exec_summary.delta.host_copy_weight_calls += 1;
            exec_summary.delta.host_copy_weight_us_total += weight_copy_us;
            exec_summary.delta.copied_weight_bytes_total += static_cast<int64_t>(packed_weight.packed.size());
            if (collect_tile_profile) {
                tile_record.host_copy_weight_us = static_cast<double>(weight_copy_us);
            }
        } else if (collect_tile_profile) {
            tile_record.host_copy_weight_us = 0.0;
        }

        const MvinConfig activation_mvin_cfg {
            activation_buf,
            plan.config.layout.activation.offset,
            static_cast<uint32_t>(exec_tile.n * exec_tile.k - 1),
            0,
            0,
            0,
            1,
            0,
            false,
            false,
            false,
            0,
            0,
            0,
        };

        if (npu_should_dump_tile(layer_id, static_cast<int64_t>(exec_tile_idx))) {
            const char * dump_dir = npu_tile_dump_dir();
            npu_dump_tile_meta(
                dump_dir,
                layer_id,
                static_cast<int64_t>(exec_tile_idx),
                exec_tile,
                activation_scale,
                plan.activation_quant.zero_point,
                plan.activation_quant.zero_point_u8,
                raw_acc_mvout,
                mvout_per_channel,
                apply_compensation);
            npu_dump_tile_blob(
                dump_dir,
                layer_id,
                static_cast<int64_t>(exec_tile_idx),
                "act_q",
                packed_activation_view->data(),
                static_cast<size_t>(exec_tile.n * exec_tile.k));
            npu_dump_tile_blob(
                dump_dir,
                layer_id,
                static_cast<int64_t>(exec_tile_idx),
                "wgt_q",
                packed_weight.packed.data(),
                static_cast<size_t>(exec_tile.k * exec_tile.m));
            npu_dump_tile_blob(
                dump_dir,
                layer_id,
                static_cast<int64_t>(exec_tile_idx),
                "wgt_scale_f32",
                packed_weight.scales.data(),
                packed_weight.scales.size() * sizeof(float));
        }

        const MvinConfig weight_mvin_cfg {
            packed_weight.cma_packed,
            plan.config.layout.weight.offset,
            static_cast<uint32_t>(exec_tile.k * exec_tile.m - 1),
            0,
            0,
            0,
            1,
            1,
            false,
            false,
            false,
            0,
            0,
            0,
        };

        const int64_t dma_in_pair_start_us = collect_stage_profile ? ggml_time_us() : 0;
        if (npu_debug_log_enabled()) {
            GGML_LOG_INFO("%s: tile m0=%" PRId64 " n0=%" PRId64 " k0=%" PRId64 " stage=%s MVIN_ACT_DMA0 col=%" PRIu32 " row=%" PRIu32 " sram=0x%08x\n",
                    __func__,
                    exec_tile.m0, exec_tile.n0, exec_tile.k0,
                    npu_stage_name(exec_tile.stage).c_str(),
                    static_cast<uint32_t>(exec_tile.n * exec_tile.k - 1),
                    0u,
                    plan.config.layout.activation.offset);
            GGML_LOG_INFO("%s: tile m0=%" PRId64 " n0=%" PRId64 " k0=%" PRId64 " stage=%s MVIN_WGT_DMA1 col=%" PRIu32 " row=%" PRIu32 " sram=0x%08x\n",
                    __func__,
                    exec_tile.m0, exec_tile.n0, exec_tile.k0,
                    npu_stage_name(exec_tile.stage).c_str(),
                    static_cast<uint32_t>(exec_tile.k * exec_tile.m - 1),
                    0u,
                    plan.config.layout.weight.offset);
        }
        uint32_t mvin_mask = 0;
        if (!activation_already_in_spm) {
            npu_dma_mvin_async(0, &activation_mvin_cfg);
            mvin_mask |= (1u << 0);
        }
        if (!weight_already_in_spm) {
            npu_dma_mvin_async(1, &weight_mvin_cfg);
            mvin_mask |= (1u << 1);
        }
        if (mvin_mask != 0) {
            npu_dma_wait_mvin(mvin_mask);
        }
        if (!activation_already_in_spm) {
            loaded_activation_key = activation_key;
            loaded_activation_valid = true;
        }
        if (!weight_already_in_spm) {
            loaded_weight_pack_index = exec_tile.weight_pack_index;
        }
        if (collect_stage_profile && mvin_mask != 0) {
            const int64_t dma_in_pair_us = ggml_time_us() - dma_in_pair_start_us;
            exec_summary.delta.dma_in_pair_calls += 1;
            exec_summary.delta.dma_in_pair_us_total += dma_in_pair_us;
            if (!activation_already_in_spm) {
                exec_summary.delta.dma_in_activation_calls += 1;
            }
            if (!weight_already_in_spm) {
                exec_summary.delta.dma_in_weight_calls += 1;
            }
            if (collect_tile_profile) {
                tile_record.dma_in_pair_us = static_cast<double>(dma_in_pair_us);
                tile_record.dma_in_activation_us = 0.0;
                tile_record.dma_in_weight_us = 0.0;
            }
        }

        const bool tile_uses_bias = exec_tile.needs_bias && exec_tile.bias_pack_index >= 0;
        if (tile_uses_bias) {
            if (exec_tile.bias_pack_index < 0 ||
                static_cast<size_t>(exec_tile.bias_pack_index) >= plan.bias_packs.size()) {
                if (error) {
                    *error = "bias_pack_index 非法";
                }
                if (collect_tile_profile) {
                    profile_record.tiles.push_back(std::move(tile_record));
                }
                cleanup_buffers(activation_buf, nullptr, acc_buf, bias_buf, bias_cache_buf, scale_cache_buf);
                return finalize_status(GGML_STATUS_FAILED);
            }
            if (use_bias_cache) {
                if (loaded_bias_cache_n0 != exec_tile.n0) {
                    const int64_t bias_prepare_start_us = collect_stage_profile ? ggml_time_us() : 0;
                    int32_t * bias_cache_words = reinterpret_cast<int32_t *>(bias_cache_buf);
                    for (int64_t global_m = 0; global_m < plan.m; ++global_m) {
                        const float bias_f32 = npu_read_bias_value_f32_exec(plan.bias, global_m, exec_tile.n0);
                        const float w_scale =
                            static_cast<size_t>(global_m) < plan.aicas_w8a8.weight_scale.size()
                            ? plan.aicas_w8a8.weight_scale[static_cast<size_t>(global_m)]
                            : per_tensor_weight_scale;
                        const float denom = activation_scale * w_scale;
                        float bias_q = denom != 0.0f ? (bias_f32 / denom) : 0.0f;
                        if (apply_compensation &&
                            static_cast<size_t>(global_m) < plan.weight_column_sum_q.size()) {
                            bias_q -= static_cast<float>(
                                plan.activation_quant.zero_point *
                                plan.weight_column_sum_q[static_cast<size_t>(global_m)]);
                        }
                        bias_cache_words[static_cast<size_t>(global_m)] = static_cast<int32_t>(std::lrint(bias_q));
                    }
                    if (collect_stage_profile) {
                        const int64_t bias_prepare_us = ggml_time_us() - bias_prepare_start_us;
                        exec_summary.delta.bias_prepare_calls += 1;
                        exec_summary.delta.bias_prepare_us_total += bias_prepare_us;
                        exec_summary.delta.bias_bytes_total += static_cast<int64_t>(plan.m * sizeof(int32_t));
                        if (collect_tile_profile) {
                            tile_record.bias_prepare_us = static_cast<double>(bias_prepare_us);
                        }
                    }
                    const int64_t dma_in_bias_start_us = collect_stage_profile ? ggml_time_us() : 0;
                    npu_dma_mvin(
                        bias_cache_buf,
                        plan.config.layout.bias_cache.offset,
                        static_cast<uint32_t>(plan.m - 1),
                        0,
                        0,
                        0,
                        1,
                        2,
                        true,
                        true,
                        false,
                        0,
                        0,
                        0);
                    loaded_bias_cache_n0 = exec_tile.n0;
                    if (collect_stage_profile) {
                        const int64_t dma_in_bias_us = ggml_time_us() - dma_in_bias_start_us;
                        exec_summary.delta.dma_in_bias_calls += 1;
                        exec_summary.delta.dma_in_bias_us_total += dma_in_bias_us;
                        if (collect_tile_profile) {
                            tile_record.dma_in_bias_us = static_cast<double>(dma_in_bias_us);
                        }
                    }
                }
                if (collect_tile_profile) {
                    tile_record.bias_bytes = static_cast<int64_t>(exec_tile.m * sizeof(int32_t));
                }
            } else {
                const npu_prepacked_bias & bias_pack =
                    plan.bias_packs[static_cast<size_t>(exec_tile.bias_pack_index)];
                if (bias_pack.values.size() < static_cast<size_t>(exec_tile.m)) {
                    if (error) {
                        *error = "bias pack size 不足";
                    }
                    if (collect_tile_profile) {
                        profile_record.tiles.push_back(std::move(tile_record));
                    }
                    cleanup_buffers(activation_buf, nullptr, acc_buf, bias_buf, bias_cache_buf, scale_cache_buf);
                    return finalize_status(GGML_STATUS_FAILED);
                }

                const int64_t bias_prepare_start_us = collect_stage_profile ? ggml_time_us() : 0;
                std::copy_n(
                    bias_pack.values.begin(),
                    static_cast<size_t>(exec_tile.m),
                    bias_values.begin());
                if (use_aicas_w8a8 && apply_compensation) {
                    for (int64_t m = 0; m < exec_tile.m; ++m) {
                        const int64_t global_m = exec_tile.m0 + m;
                        if (global_m >= 0 &&
                                static_cast<size_t>(global_m) < plan.weight_column_sum_q.size()) {
                            bias_values[static_cast<size_t>(m)] -=
                                plan.activation_quant.zero_point *
                                plan.weight_column_sum_q[static_cast<size_t>(global_m)];
                        }
                    }
                }

                std::memcpy(
                    bias_buf,
                    bias_values.data(),
                    static_cast<size_t>(exec_tile.m * sizeof(int32_t)));
                if (npu_should_dump_tile(layer_id, static_cast<int64_t>(exec_tile_idx))) {
                    npu_dump_tile_blob(
                        npu_tile_dump_dir(),
                        layer_id,
                        static_cast<int64_t>(exec_tile_idx),
                        "bias_i32",
                        bias_values.data(),
                        static_cast<size_t>(exec_tile.m * sizeof(int32_t)));
                }
                if (collect_stage_profile) {
                    const int64_t bias_prepare_us = ggml_time_us() - bias_prepare_start_us;
                    exec_summary.delta.bias_prepare_calls += 1;
                    exec_summary.delta.bias_prepare_us_total += bias_prepare_us;
                    exec_summary.delta.bias_bytes_total += static_cast<int64_t>(exec_tile.m * sizeof(int32_t));
                    if (collect_tile_profile) {
                        tile_record.bias_prepare_us = static_cast<double>(bias_prepare_us);
                    }
                }
                if (collect_tile_profile) {
                    tile_record.bias_bytes = static_cast<int64_t>(exec_tile.m * sizeof(int32_t));
                }

                const int64_t dma_in_bias_start_us = collect_stage_profile ? ggml_time_us() : 0;
                npu_dma_mvin(
                    bias_buf,
                    plan.config.layout.bias_accumulator.offset,
                    static_cast<uint32_t>(exec_tile.m - 1),
                    0,
                    0,
                    0,
                    1,
                    2,
                    true,
                    true,
                    false,
                    0,
                    0,
                    0);
                if (collect_stage_profile) {
                    const int64_t dma_in_bias_us = ggml_time_us() - dma_in_bias_start_us;
                    exec_summary.delta.dma_in_bias_calls += 1;
                    exec_summary.delta.dma_in_bias_us_total += dma_in_bias_us;
                    if (collect_tile_profile) {
                        tile_record.dma_in_bias_us = static_cast<double>(dma_in_bias_us);
                    }
                }
            }
        }

        const int64_t gemm_start_us = collect_stage_profile ? ggml_time_us() : 0;
        if (npu_debug_log_enabled()) {
                GGML_LOG_INFO("%s: tile m0=%" PRId64 " n0=%" PRId64 " k0=%" PRId64 " GEMM n=%" PRId64 " m=%" PRId64 " k=%" PRId64 " bias_acc=0x%08x out_acc=0x%08x\n",
                    __func__,
                    exec_tile.m0, exec_tile.n0, exec_tile.k0,
                    exec_tile.n, exec_tile.m, exec_tile.k,
                    plan.config.layout.bias_accumulator.offset,
                    plan.config.layout.output_accumulator.offset);
        }
        const uint32_t gemm_biaspsum_addr = tile_uses_bias
            ? (use_bias_cache
                ? plan.config.layout.bias_cache.offset + static_cast<uint32_t>(exec_tile.m0 * sizeof(int32_t))
                : plan.config.layout.bias_accumulator.offset)
            : (!exec_tile.needs_bias ? plan.config.layout.output_accumulator.offset : plan.config.layout.bias_accumulator.offset);
        npu_gemm_run(
            /*dataflow=*/true,
            /*int_type=*/0,
            /*optype=*/0,
            /*accout_dest=*/true,
            /*input_a_zeropoint=*/0,
            /*input_b_zeropoint=*/0,
            /*output_zeropoint=*/0,
            /*output_scale=*/1,
            /*output_scaleshift=*/0,
            /*biaspsum_addr=*/gemm_biaspsum_addr,
            /*biaspsum_stride=*/static_cast<uint16_t>(exec_tile.m),
            /*biaspsum_width=*/static_cast<uint8_t>(exec_tile.m),
            /*biaspsum_height=*/static_cast<uint8_t>(exec_tile.n),
            /*output_addr=*/plan.config.layout.output_accumulator.offset,
            /*output_stride=*/static_cast<uint16_t>(exec_tile.m),
            /*isaccu=*/!exec_tile.needs_bias || tile_uses_bias,
            /*relu=*/false,
            /*relu_type=*/0,
            /*is_bias=*/tile_uses_bias,
            /*input_a_addr=*/plan.config.layout.activation.offset,
            /*input_a_col_num=*/static_cast<uint16_t>(exec_tile.k - 1),
            /*input_a_row_num=*/static_cast<uint8_t>(exec_tile.n - 1),
            /*input_a_stride=*/static_cast<uint16_t>(exec_tile.k),
            /*input_b_addr=*/plan.config.layout.weight.offset,
            /*input_b_col_num=*/static_cast<uint8_t>(exec_tile.m - 1),
            /*input_b_row_num=*/static_cast<uint16_t>(exec_tile.k - 1),
            /*input_b_stride=*/static_cast<uint16_t>(exec_tile.m),
            /*asymmetric_activations=*/hardware_asymmetric_activations);
        if (collect_stage_profile) {
            const int64_t gemm_us = ggml_time_us() - gemm_start_us;
            exec_summary.delta.gemm_calls += 1;
            exec_summary.delta.gemm_us_total += gemm_us;
            if (collect_tile_profile) {
                tile_record.gemm_us = static_cast<double>(gemm_us);
            }
        }

        if (exec_tile.writes_output) {
            const uint32_t output_tile_elems = static_cast<uint32_t>(exec_tile.n * exec_tile.m);
            if (collect_tile_profile) {
                tile_record.acc_readback_bytes = static_cast<int64_t>(exec_tile.n * exec_tile.m * sizeof(int32_t));
            }

            const int64_t dma_out_start_us = collect_stage_profile ? ggml_time_us() : 0;
            if (npu_debug_log_enabled()) {
                GGML_LOG_INFO("%s: tile m0=%" PRId64 " n0=%" PRId64 " k0=%" PRId64 " MVOUT_ACC_%s col=%" PRIu32 " row=0 sram=0x%08x f32_scale=0x%08x\n",
                        __func__,
                        exec_tile.m0, exec_tile.n0, exec_tile.k0,
                        raw_acc_mvout ? "I32" : "F32",
                        static_cast<uint32_t>(exec_tile.n * exec_tile.m - 1),
                        plan.config.layout.output_accumulator.offset,
                        raw_acc_mvout ? 0
                                      : (use_scale_cache
                                            ? plan.config.layout.scale_cache.offset +
                                                static_cast<uint32_t>(exec_tile.n * exec_tile.m0 * sizeof(uint32_t))
                                            : 0x00070000));
            }
            if (use_scale_cache &&
                (loaded_scale_cache_n0 != exec_tile.n0 || loaded_scale_cache_n != exec_tile.n)) {
                uint32_t * scale_cache_words = reinterpret_cast<uint32_t *>(scale_cache_buf);
                for (int64_t stripe_m0 = 0; stripe_m0 < plan.m; stripe_m0 += plan.config.sa_cols) {
                    const int64_t stripe_m = std::min<int64_t>(plan.config.sa_cols, plan.m - stripe_m0);
                    const size_t tile_base = static_cast<size_t>(exec_tile.n * stripe_m0);
                    for (int64_t n = 0; n < exec_tile.n; ++n) {
                        for (int64_t local_m = 0; local_m < stripe_m; ++local_m) {
                            const int64_t global_m = stripe_m0 + local_m;
                            const float w_scale =
                                static_cast<size_t>(global_m) < plan.aicas_w8a8.weight_scale.size()
                                ? plan.aicas_w8a8.weight_scale[static_cast<size_t>(global_m)]
                                : per_tensor_weight_scale;
                            scale_cache_words[tile_base + static_cast<size_t>(n * stripe_m + local_m)] =
                                npu_float_to_q8_24_u32(activation_scale * w_scale);
                        }
                    }
                }
                npu_dma_mvin(
                    scale_cache_buf,
                    plan.config.layout.scale_cache.offset,
                    static_cast<uint32_t>(exec_tile.n * plan.m - 1),
                    0,
                    0,
                    0,
                    1,
                    2,
                    true,
                    false,
                    false,
                    0,
                    0,
                    0);
                loaded_scale_cache_n0 = exec_tile.n0;
                loaded_scale_cache_n = exec_tile.n;
            } else if (!raw_acc_mvout) {
                uint32_t * scale_words = reinterpret_cast<uint32_t *>(bias_buf);
                const int64_t scale_count = mvout_per_channel ? exec_tile.n * exec_tile.m : 1;
                for (int64_t idx = 0; idx < scale_count; ++idx) {
                    const int64_t m = mvout_per_channel ? (idx % exec_tile.m) : 0;
                    const int64_t global_m = exec_tile.m0 + m;
                    const float w_scale = mvout_per_channel && global_m >= 0 &&
                            static_cast<size_t>(global_m) < plan.aicas_w8a8.weight_scale.size()
                        ? plan.aicas_w8a8.weight_scale[static_cast<size_t>(global_m)]
                        : per_tensor_weight_scale;
                    scale_words[static_cast<size_t>(idx)] = npu_float_to_q8_24_u32(activation_scale * w_scale);
                }
                npu_dma_mvin(
                    scale_words,
                    0x00070000,
                    static_cast<uint32_t>(scale_count - 1),
                    0,
                    0,
                    0,
                    1,
                    2,
                    true,
                    false,
                    false,
                    0,
                    0,
                    0);
            }
            npu_dma_mvout_ex(
                acc_buf,
                plan.config.layout.output_accumulator.offset,
                output_tile_elems - 1,
                0,
                static_cast<uint16_t>(output_tile_elems),
                output_tile_elems,
                raw_acc_mvout ? 1 : 3,
                1,
                true,
                !raw_acc_mvout,
                0,
                raw_acc_mvout ? 0
                              : (use_scale_cache
                                    ? plan.config.layout.scale_cache.offset +
                                        static_cast<uint32_t>(exec_tile.n * exec_tile.m0 * sizeof(uint32_t))
                                    : 0x00070000),
                !raw_acc_mvout && mvout_per_channel);
            if (collect_stage_profile) {
                const int64_t dma_out_us = ggml_time_us() - dma_out_start_us;
                exec_summary.delta.dma_out_calls += 1;
                exec_summary.delta.dma_out_us_total += dma_out_us;
                exec_summary.delta.acc_readback_bytes_total += static_cast<int64_t>(exec_tile.n * exec_tile.m * sizeof(int32_t));
                if (collect_tile_profile) {
                    tile_record.dma_out_us = static_cast<double>(dma_out_us);
                }
            }

            if (fold_output_reconstruction) {
                const int64_t postprocess_start_us = collect_stage_profile ? ggml_time_us() : 0;
                const size_t tile_elems = static_cast<size_t>(exec_tile.n * exec_tile.m);
                if (raw_acc_mvout) {
                    acc_raw_values_host.resize(tile_elems);
                    std::memcpy(acc_raw_values_host.data(), acc_buf, tile_elems * sizeof(int32_t));
                    if (npu_should_dump_tile(layer_id, static_cast<int64_t>(exec_tile_idx))) {
                        npu_dump_tile_blob(
                            npu_tile_dump_dir(),
                            layer_id,
                            static_cast<int64_t>(exec_tile_idx),
                            "acc_raw_i32",
                            acc_raw_values_host.data(),
                            tile_elems * sizeof(int32_t));
                    }
                } else {
                    acc_scaled_values.resize(tile_elems);
                    std::memcpy(acc_scaled_values.data(), acc_buf, tile_elems * sizeof(float));
                }
                npu_prepare_bias_tile_f32_exec(
                    nullptr,
                    exec_tile.m0,
                    exec_tile.n0,
                    exec_tile.m,
                    exec_tile.n,
                    bias_tile_values);

                if (npu_dst_is_dense_f32(plan.dst)) {
                    for (int64_t n = 0; n < exec_tile.n; ++n) {
                        float * dst_row = reinterpret_cast<float *>(
                            static_cast<char *>(plan.dst->data) +
                            (exec_tile.n0 + n) * plan.dst->nb[1] +
                            exec_tile.m0 * sizeof(float));
                        const float * acc_row = raw_acc_mvout
                            ? nullptr
                            : acc_scaled_values.data() + static_cast<size_t>(n * exec_tile.m);
                        const int32_t * acc_raw_row = raw_acc_mvout
                            ? acc_raw_values_host.data() + static_cast<size_t>(n * exec_tile.m)
                            : nullptr;
                        const float * bias_row = bias_tile_values.empty()
                            ? nullptr
                            : bias_tile_values.data() + static_cast<size_t>(n * exec_tile.m);
                        for (int64_t m = 0; m < exec_tile.m; ++m) {
                            const size_t idx = static_cast<size_t>(m);
                            const int64_t global_m = exec_tile.m0 + m;
                            const float wgt_scale = mvout_per_channel &&
                                    global_m >= 0 &&
                                    static_cast<size_t>(global_m) < plan.aicas_w8a8.weight_scale.size()
                                ? plan.aicas_w8a8.weight_scale[static_cast<size_t>(global_m)]
                                : per_tensor_weight_scale;
                            const float dequant_scale = activation_scale * wgt_scale;
                            float value = raw_acc_mvout
                                ? static_cast<float>(acc_raw_row[idx]) * dequant_scale
                                : acc_row[idx];
                            if (apply_compensation) {
                                if (global_m >= 0 &&
                                    static_cast<size_t>(global_m) < plan.weight_column_sum_q.size()) {
                                    value -= static_cast<float>(
                                        plan.activation_quant.zero_point *
                                        plan.weight_column_sum_q[static_cast<size_t>(global_m)]) *
                                        dequant_scale;
                                }
                            }
                            if (bias_row != nullptr && !tile_uses_bias) {
                                value += bias_row[idx];
                            }
                            dst_row[idx] = value;
                        }
                    }
                } else {
                    for (int64_t n = 0; n < exec_tile.n; ++n) {
                        const float * acc_row = raw_acc_mvout
                            ? nullptr
                            : acc_scaled_values.data() + static_cast<size_t>(n * exec_tile.m);
                        const int32_t * acc_raw_row = raw_acc_mvout
                            ? acc_raw_values_host.data() + static_cast<size_t>(n * exec_tile.m)
                            : nullptr;
                        const float * bias_row = bias_tile_values.empty()
                            ? nullptr
                            : bias_tile_values.data() + static_cast<size_t>(n * exec_tile.m);
                        char * dst_row = static_cast<char *>(plan.dst->data) +
                            (exec_tile.n0 + n) * plan.dst->nb[1] +
                            exec_tile.m0 * plan.dst->nb[0];
                        for (int64_t m = 0; m < exec_tile.m; ++m) {
                            const int64_t global_m = exec_tile.m0 + m;
                            const float wgt_scale = mvout_per_channel &&
                                    global_m >= 0 &&
                                    static_cast<size_t>(global_m) < plan.aicas_w8a8.weight_scale.size()
                                ? plan.aicas_w8a8.weight_scale[static_cast<size_t>(global_m)]
                                : per_tensor_weight_scale;
                            const float dequant_scale = activation_scale * wgt_scale;
                            float value = raw_acc_mvout
                                ? static_cast<float>(acc_raw_row[static_cast<size_t>(m)]) * dequant_scale
                                : acc_row[static_cast<size_t>(m)];
                            if (apply_compensation) {
                                if (global_m >= 0 &&
                                    static_cast<size_t>(global_m) < plan.weight_column_sum_q.size()) {
                                    value -= static_cast<float>(
                                        plan.activation_quant.zero_point *
                                        plan.weight_column_sum_q[static_cast<size_t>(global_m)]) *
                                        dequant_scale;
                                }
                            }
                            if (bias_row != nullptr && !tile_uses_bias) {
                                value += bias_row[static_cast<size_t>(m)];
                            }
                            *reinterpret_cast<float *>(dst_row + m * plan.dst->nb[0]) = value;
                        }
                    }
                }
                if (collect_stage_profile) {
                    const int64_t postprocess_us = ggml_time_us() - postprocess_start_us;
                    exec_summary.delta.postprocess_calls += 1;
                    exec_summary.delta.postprocess_us_total += postprocess_us;
                    exec_summary.delta.output_write_bytes_total +=
                        static_cast<int64_t>(exec_tile.n * exec_tile.m * sizeof(float));
                    if (collect_tile_profile) {
                        tile_record.postprocess_us = static_cast<double>(postprocess_us);
                    }
                }
                if (collect_tile_profile) {
                    tile_record.output_write_bytes =
                        static_cast<int64_t>(exec_tile.n * exec_tile.m * sizeof(float));
                }
            } else {
                const int64_t postprocess_start_us = collect_stage_profile ? ggml_time_us() : 0;
                const size_t tile_elems = static_cast<size_t>(exec_tile.n * exec_tile.m);
                if (raw_acc_mvout) {
                    acc_raw_values_host.resize(tile_elems);
                    std::memcpy(acc_raw_values_host.data(), acc_buf, tile_elems * sizeof(int32_t));
                } else {
                    acc_scaled_values.resize(tile_elems);
                    std::memcpy(acc_scaled_values.data(), acc_buf, tile_elems * sizeof(float));
                }
                npu_prepare_bias_tile_f32_exec(
                    nullptr,
                    exec_tile.m0,
                    exec_tile.n0,
                    exec_tile.m,
                    exec_tile.n,
                    bias_tile_values);

                if (npu_dst_is_dense_f32(plan.dst)) {
                    for (int64_t n = 0; n < exec_tile.n; ++n) {
                        float * dst_row = reinterpret_cast<float *>(
                            static_cast<char *>(plan.dst->data) +
                            (exec_tile.n0 + n) * plan.dst->nb[1] +
                            exec_tile.m0 * sizeof(float));
                        const float * acc_row = raw_acc_mvout
                            ? nullptr
                            : acc_scaled_values.data() + static_cast<size_t>(n * exec_tile.m);
                        const int32_t * acc_raw_row = raw_acc_mvout
                            ? acc_raw_values_host.data() + static_cast<size_t>(n * exec_tile.m)
                            : nullptr;
                        const float * bias_row = bias_tile_values.empty()
                            ? nullptr
                            : bias_tile_values.data() + static_cast<size_t>(n * exec_tile.m);
                        for (int64_t m = 0; m < exec_tile.m; ++m) {
                            const size_t idx = static_cast<size_t>(m);
                            const float wgt_scale = packed_weight.scales[idx];
                            const int64_t global_m = exec_tile.m0 + m;
                            const float acc_scaled = raw_acc_mvout
                                ? static_cast<float>(acc_raw_row[idx]) * activation_scale
                                : acc_row[idx];
                            float value = acc_scaled * wgt_scale;
                            if (apply_compensation &&
                                global_m >= 0 &&
                                static_cast<size_t>(global_m) < plan.weight_column_sum_q.size()) {
                                value -= static_cast<float>(
                                    plan.activation_quant.zero_point *
                                    plan.weight_column_sum_q[static_cast<size_t>(global_m)]) *
                                    activation_scale * wgt_scale;
                            }
                            if (bias_row != nullptr && !tile_uses_bias) {
                                value += bias_row[idx];
                            }
                            dst_row[idx] = value;
                        }
                    }
                } else {
                    for (int64_t n = 0; n < exec_tile.n; ++n) {
                        const float * acc_row = raw_acc_mvout
                            ? nullptr
                            : acc_scaled_values.data() + static_cast<size_t>(n * exec_tile.m);
                        const int32_t * acc_raw_row = raw_acc_mvout
                            ? acc_raw_values_host.data() + static_cast<size_t>(n * exec_tile.m)
                            : nullptr;
                        const float * bias_row = bias_tile_values.empty()
                            ? nullptr
                            : bias_tile_values.data() + static_cast<size_t>(n * exec_tile.m);
                        char * dst_row = static_cast<char *>(plan.dst->data) +
                            (exec_tile.n0 + n) * plan.dst->nb[1] +
                            exec_tile.m0 * plan.dst->nb[0];
                        for (int64_t m = 0; m < exec_tile.m; ++m) {
                            const size_t idx = static_cast<size_t>(m);
                            const float wgt_scale = packed_weight.scales[idx];
                            const int64_t global_m = exec_tile.m0 + m;
                            const float acc_scaled = raw_acc_mvout
                                ? static_cast<float>(acc_raw_row[idx]) * activation_scale
                                : acc_row[idx];
                            float value = acc_scaled * wgt_scale;
                            if (apply_compensation &&
                                global_m >= 0 &&
                                static_cast<size_t>(global_m) < plan.weight_column_sum_q.size()) {
                                value -= static_cast<float>(
                                    plan.activation_quant.zero_point *
                                    plan.weight_column_sum_q[static_cast<size_t>(global_m)]) *
                                    activation_scale * wgt_scale;
                            }
                            if (bias_row != nullptr && !tile_uses_bias) {
                                value += bias_row[idx];
                            }
                            *reinterpret_cast<float *>(dst_row + m * plan.dst->nb[0]) = value;
                        }
                    }
                }
                if (collect_stage_profile) {
                    const int64_t postprocess_us = ggml_time_us() - postprocess_start_us;
                    exec_summary.delta.postprocess_calls += 1;
                    exec_summary.delta.postprocess_us_total += postprocess_us;
                    exec_summary.delta.output_write_bytes_total += static_cast<int64_t>(exec_tile.n * exec_tile.m * sizeof(float));
                    if (collect_tile_profile) {
                        tile_record.postprocess_us = static_cast<double>(postprocess_us);
                    }
                }
                if (collect_tile_profile) {
                    tile_record.output_write_bytes = static_cast<int64_t>(exec_tile.n * exec_tile.m * sizeof(float));
                }
            }

            if (tile_align_logs < tile_align_debug_cfg.max_logs &&
                npu_should_check_tile_align(tile_align_debug_cfg, layer_id, static_cast<int64_t>(exec_tile_idx))) {
                float max_abs_diff = 0.0f;
                float mean_abs_diff = 0.0f;
                float mean_abs_diff_alt_layout = 0.0f;
                int64_t max_diff_n = -1;
                int64_t max_diff_m = -1;
                float ref_at_max = 0.0f;
                float npu_at_max = 0.0f;
                int64_t compared = 0;
                const int8_t * act_q = packed_activation_view->data();
                const int8_t * w_q = packed_weight.packed.data();
                const npu_prepacked_bias * bias_pack = tile_uses_bias
                    ? &plan.bias_packs[static_cast<size_t>(exec_tile.bias_pack_index)]
                    : nullptr;

                for (int64_t n = 0; n < exec_tile.n; ++n) {
                    for (int64_t m = 0; m < exec_tile.m; ++m) {
                        int32_t acc_ref = 0;
                        int32_t acc_ref_alt_layout = 0;
                        for (int64_t k = 0; k < exec_tile.k; ++k) {
                            const int32_t qa = npu_effective_activation_q(
                                plan,
                                act_q[static_cast<size_t>(n * exec_tile.k + k)]);
                            const int32_t qw = static_cast<int32_t>(w_q[static_cast<size_t>(k * exec_tile.m + m)]);
                            acc_ref += qa * qw;
                            acc_ref_alt_layout += qa * static_cast<int32_t>(
                                w_q[static_cast<size_t>(m * exec_tile.k + k)]);
                        }
                        const int64_t global_m = exec_tile.m0 + m;
                        if (bias_pack != nullptr) {
                            acc_ref += bias_pack->values[static_cast<size_t>(m)];
                            acc_ref_alt_layout += bias_pack->values[static_cast<size_t>(m)];
                        }
                        const float wgt_scale = packed_weight.scales[static_cast<size_t>(m)];
                        float out_ref = static_cast<float>(acc_ref) * activation_scale * wgt_scale;
                        float out_ref_alt_layout =
                            static_cast<float>(acc_ref_alt_layout) * activation_scale * wgt_scale;
                        if (apply_compensation &&
                            global_m >= 0 &&
                            static_cast<size_t>(global_m) < plan.weight_column_sum_q.size()) {
                            out_ref -= static_cast<float>(
                                plan.activation_quant.zero_point *
                                plan.weight_column_sum_q[static_cast<size_t>(global_m)]) *
                                activation_scale * wgt_scale;
                            out_ref_alt_layout -= static_cast<float>(
                                plan.activation_quant.zero_point *
                                plan.weight_column_sum_q[static_cast<size_t>(global_m)]) *
                                activation_scale * wgt_scale;
                        }
                        if (use_aicas_w8a8 && plan.bias != nullptr && bias_pack == nullptr) {
                            out_ref += npu_read_bias_value_f32_exec(plan.bias, global_m, exec_tile.n0 + n);
                            out_ref_alt_layout +=
                                npu_read_bias_value_f32_exec(plan.bias, global_m, exec_tile.n0 + n);
                        }

                        const char * dst_ptr = static_cast<const char *>(plan.dst->data) +
                            (exec_tile.n0 + n) * plan.dst->nb[1] +
                            (exec_tile.m0 + m) * plan.dst->nb[0];
                        const float out_npu = *reinterpret_cast<const float *>(dst_ptr);
                        const float abs_diff = std::fabs(out_npu - out_ref);
                        const float abs_diff_alt_layout = std::fabs(out_npu - out_ref_alt_layout);
                        mean_abs_diff += abs_diff;
                        mean_abs_diff_alt_layout += abs_diff_alt_layout;
                        compared += 1;
                        if (abs_diff > max_abs_diff) {
                            max_abs_diff = abs_diff;
                            max_diff_n = n;
                            max_diff_m = m;
                            ref_at_max = out_ref;
                            npu_at_max = out_npu;
                        }
                    }
                }

                if (compared > 0) {
                    mean_abs_diff /= static_cast<float>(compared);
                    mean_abs_diff_alt_layout /= static_cast<float>(compared);
                }

                if (max_abs_diff > tile_align_debug_cfg.abs_tol || npu_debug_log_enabled()) {
                    const char * root_name = plan.root && plan.root->name[0] != '\0' ? plan.root->name : "(unnamed)";
                    GGML_LOG_INFO(
                        "%s: tile_align layer=%" PRId64 " tile=%zu root=%s m0=%" PRId64 " n0=%" PRId64 " k0=%" PRId64
                        " max_abs=%.8f mean_abs=%.8f mean_abs_alt_layout=%.8f tol=%.8f at(n=%" PRId64 ",m=%" PRId64 ") npu=%.8f ref=%.8f"
                        " act_scale=%.8f act_zp_i8=%d act_zp_u8=%d bias_mode=%s asym=%d fold=%d raw_mvout=%d bias_pack=%d\n",
                        __func__,
                        layer_id,
                        exec_tile_idx,
                        root_name,
                        exec_tile.m0, exec_tile.n0, exec_tile.k0,
                        max_abs_diff,
                        mean_abs_diff,
                        mean_abs_diff_alt_layout,
                        tile_align_debug_cfg.abs_tol,
                        max_diff_n, max_diff_m,
                        npu_at_max,
                        ref_at_max,
                        activation_scale,
                        plan.activation_quant.zero_point,
                        plan.activation_quant.zero_point_u8,
                        npu_bias_mode_name(bias_mode),
                        hardware_asymmetric_activations ? 1 : 0,
                        fold_output_reconstruction ? 1 : 0,
                        raw_acc_mvout ? 1 : 0,
                        tile_uses_bias ? 1 : 0);
                }
                tile_align_logs += 1;
            }
        }

        if (collect_tile_profile) {
            tile_record.total_us = static_cast<double>(ggml_time_us() - tile_start_us);
            profile_record.tiles.push_back(std::move(tile_record));
        }
    }

    cleanup_buffers(activation_buf, nullptr, acc_buf, bias_buf, bias_cache_buf, scale_cache_buf);
    if (npu_debug_log_enabled()) {
        const char * root_name = plan.root && plan.root->name[0] != '\0' ? plan.root->name : "(unnamed)";
        GGML_LOG_INFO("%s: leave root=%s bias_mode=%s\n",
                __func__,
                root_name,
                npu_bias_mode_name(bias_mode));
    }
    return finalize_status(GGML_STATUS_SUCCESS);
}

} // namespace ggml_npu
