#include "ggml-npu-exec.h"

#include "ggml-npu-plan.h"
#include "ggml-npu-profile.h"
#include "ggml-npu-quant.h"

#include "ggml-impl.h"
#include "npu_runtime.h"

#include <algorithm>
#include <cinttypes>
#include <condition_variable>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <limits>
#include <unordered_map>
#include <unordered_set>
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

static int64_t npu_hardware_n_rows(int64_t n_rows) {
#if defined(GGML_NPU_VERSA_P_RUNTIME)
    return npu_align_up_i64(n_rows, 16);
#else
    return n_rows;
#endif
}

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

static bool npu_dst_f32_tile_is_contiguous(
        const struct ggml_tensor * dst,
        int64_t m0,
        int64_t m,
        int64_t n) {
    return npu_dst_is_dense_f32(dst) &&
        m0 == 0 &&
        m == dst->ne[0] &&
        dst->nb[1] == static_cast<size_t>(dst->ne[0]) * sizeof(float) &&
        n > 0;
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

static int8_t npu_packed_weight_at(
        const npu_prepacked_weight & packed_weight,
        int64_t k,
        int64_t m) {
    const int64_t group = m / 32;
    const int64_t lane = m % 32;
    return packed_weight.packed[static_cast<size_t>((group * packed_weight.k + k) * 32 + lane)];
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
    if (raw_mvout != nullptr && raw_mvout[0] != '\0') {
        return std::strcmp(raw_mvout, "0") != 0;
    }
#if defined(GGML_NPU_VERSA_P_RUNTIME)
    const char * fp32_mvout = std::getenv("GGML_NPU_FORCE_FP32_MVOUT");
    if (fp32_mvout != nullptr && fp32_mvout[0] != '\0' && std::strcmp(fp32_mvout, "0") != 0) {
        return false;
    }
    return false;
#else
    return false;
#endif
}

static bool npu_w_pingpong_enabled() {
#if defined(GGML_NPU_VERSA_P_RUNTIME)
    const char * value = std::getenv("GGML_NPU_W_PINGPONG");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
#else
    return false;
#endif
}

static size_t npu_cma_read_chunk_bytes() {
    static bool initialized = false;
    static size_t chunk_bytes = 0;
    if (!initialized) {
        initialized = true;
        const char * value = std::getenv("GGML_NPU_CMA_READ_CHUNK_BYTES");
        if (value != nullptr && value[0] != '\0') {
            const long parsed = std::strtol(value, nullptr, 10);
            if (parsed > 0) {
                chunk_bytes = static_cast<size_t>(parsed);
            }
        }
    }
    return chunk_bytes;
}

static bool npu_cma_read_dma_enabled() {
#if defined(GGML_NPU_VERSA_P_RUNTIME)
    const char * value = std::getenv("GGML_NPU_CMA_READ_DMA");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
#else
    return false;
#endif
}

static bool npu_full_output_cma_enabled() {
#if defined(GGML_NPU_VERSA_P_RUNTIME)
    const char * value = std::getenv("GGML_NPU_FULL_OUTPUT_CMA");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
#else
    return false;
#endif
}

static uint32_t npu_cma_read_dma_chunk_bytes() {
    static bool initialized = false;
    static uint32_t chunk_bytes = 1024u * 1024u;
    if (!initialized) {
        initialized = true;
        const char * value = std::getenv("GGML_NPU_CMA_READ_DMA_CHUNK_BYTES");
        if (value != nullptr && value[0] != '\0') {
            const long parsed = std::strtol(value, nullptr, 10);
            if (parsed > 0) {
                chunk_bytes = static_cast<uint32_t>(parsed);
            }
        }
    }
    return chunk_bytes;
}

static size_t npu_cma_read_dma_min_bytes() {
    static bool initialized = false;
    static size_t min_bytes = 1024u * 1024u;
    if (!initialized) {
        initialized = true;
        const char * value = std::getenv("GGML_NPU_CMA_READ_DMA_MIN_BYTES");
        if (value != nullptr && value[0] != '\0') {
            const long parsed = std::strtol(value, nullptr, 10);
            if (parsed >= 0) {
                min_bytes = static_cast<size_t>(parsed);
            }
        }
    }
    return min_bytes;
}

static void npu_copy_from_cma(void * dst, const void * src, size_t bytes) {
#if defined(GGML_NPU_VERSA_P_RUNTIME)
    if (npu_cma_read_dma_enabled() && bytes >= npu_cma_read_dma_min_bytes()) {
        const int rc = npu_dma_copy_from_cma(src, dst, bytes, npu_cma_read_dma_chunk_bytes());
        if (rc == 0) {
            return;
        }
        static bool warned = false;
        if (!warned) {
            warned = true;
            std::fprintf(stderr,
                "npu_copy_from_cma: driver DMA copy failed rc=%d; falling back to CPU memcpy\n",
                rc);
        }
    }
#endif
    const size_t chunk_bytes = npu_cma_read_chunk_bytes();
    if (chunk_bytes == 0 || chunk_bytes >= bytes) {
        std::memcpy(dst, src, bytes);
        return;
    }
    size_t offset = 0;
    while (offset < bytes) {
        const size_t n = std::min(chunk_bytes, bytes - offset);
        std::memcpy(
            static_cast<char *>(dst) + offset,
            static_cast<const char *>(src) + offset,
            n);
        offset += n;
    }
}

struct npu_full_output_span {
    int64_t n0 = 0;
    int64_t n = 0;
    int64_t hw_n = 0;
    int64_t padded_row0 = 0;
};

static std::vector<npu_full_output_span> npu_build_full_output_spans(const npu_node_plan & plan) {
    std::map<int64_t, npu_full_output_span> by_n0;
    int64_t padded_row = 0;
    for (const npu_exec_tile & tile : plan.exec_tiles) {
        if (!tile.writes_output || by_n0.find(tile.n0) != by_n0.end()) {
            continue;
        }
        const int64_t hw_n = npu_hardware_n_rows(tile.n);
        npu_full_output_span span;
        span.n0 = tile.n0;
        span.n = tile.n;
        span.hw_n = hw_n;
        span.padded_row0 = padded_row;
        by_n0.emplace(tile.n0, span);
        padded_row += hw_n;
    }

    std::vector<npu_full_output_span> spans;
    spans.reserve(by_n0.size());
    for (const auto & entry : by_n0) {
        spans.push_back(entry.second);
    }
    return spans;
}

static const npu_full_output_span * npu_find_full_output_span(
        const std::vector<npu_full_output_span> & spans,
        int64_t n0) {
    for (const npu_full_output_span & span : spans) {
        if (span.n0 == n0) {
            return &span;
        }
    }
    return nullptr;
}

struct npu_tile_align_debug_cfg {
    bool enabled = false;
    int64_t layer_id = -1;
    int64_t layer_begin = -1;
    int64_t layer_end = -1;
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

static bool npu_env_enabled_default(const char * name, bool default_value) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }
    return std::strcmp(value, "0") != 0;
}

static npu_tile_align_debug_cfg npu_get_tile_align_debug_cfg() {
    npu_tile_align_debug_cfg cfg;
    cfg.enabled = std::getenv("GGML_NPU_TILE_ALIGN_DEBUG") != nullptr;
    if (!cfg.enabled) {
        return cfg;
    }
    cfg.layer_id = npu_env_i64("GGML_NPU_TILE_ALIGN_LAYER_ID", -1);
    cfg.layer_begin = npu_env_i64("GGML_NPU_TILE_ALIGN_LAYER_BEGIN", -1);
    cfg.layer_end = npu_env_i64("GGML_NPU_TILE_ALIGN_LAYER_END", -1);
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
    if (cfg.layer_id < 0) {
        if (cfg.layer_begin >= 0 && layer_id < cfg.layer_begin) {
            return false;
        }
        if (cfg.layer_end >= 0 && layer_id > cfg.layer_end) {
            return false;
        }
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

struct npu_activation_pack_async_stats {
    int64_t jobs = 0;
    int64_t async_us = 0;
    int64_t bytes = 0;
};

struct npu_activation_pack_async_entry {
    npu_activation_tile_key key;
    std::vector<int8_t> data;
    std::string error;
    bool ready = false;
    bool failed = false;
    std::mutex mutex;
    std::condition_variable cv;
};

class npu_activation_pack_scheduler {
public:
    npu_activation_pack_scheduler(
            const npu_node_plan & plan,
            float activation_scale,
            int64_t window)
        : plan_(plan),
          activation_scale_(activation_scale),
          window_(std::max<int64_t>(1, window)) {
        worker_ = std::thread([this]() { this->worker_loop(); });
    }

    ~npu_activation_pack_scheduler() {
        stop();
    }

    npu_activation_pack_scheduler(const npu_activation_pack_scheduler &) = delete;
    npu_activation_pack_scheduler & operator=(const npu_activation_pack_scheduler &) = delete;

    void schedule_window(size_t start_idx) {
        int64_t scheduled = 0;
        for (size_t idx = start_idx; idx < plan_.exec_tiles.size() && scheduled < window_; ++idx) {
            const npu_exec_tile & tile = plan_.exec_tiles[idx];
            const npu_activation_tile_key key {
                tile.n0,
                tile.n,
                tile.k0,
                tile.k,
            };
            if (schedule_one(key)) {
                scheduled += 1;
            }
        }
    }

    std::shared_ptr<npu_activation_pack_async_entry> wait_ready(
            const npu_activation_tile_key & key,
            std::string * error,
            bool * ready_before_wait) {
        std::shared_ptr<npu_activation_pack_async_entry> entry;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = entries_.find(key);
            if (it != entries_.end()) {
                entry = it->second;
            }
        }
        if (entry == nullptr) {
            schedule_one(key);
            std::lock_guard<std::mutex> lock(mutex_);
            entry = entries_[key];
        }

        std::unique_lock<std::mutex> entry_lock(entry->mutex);
        if (ready_before_wait != nullptr) {
            *ready_before_wait = entry->ready;
        }
        entry->cv.wait(entry_lock, [&entry]() { return entry->ready; });
        if (entry->failed) {
            if (error != nullptr) {
                *error = entry->error;
            }
            return nullptr;
        }
        return entry;
    }

    npu_activation_pack_async_stats stats() const {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        return stats_;
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stop_) {
                return;
            }
            stop_ = true;
            queue_.clear();
        }
        cv_.notify_all();
        if (worker_.joinable()) {
            worker_.join();
        }
    }

private:
    bool schedule_one(const npu_activation_tile_key & key) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (entries_.find(key) != entries_.end()) {
            return false;
        }
        auto entry = std::make_shared<npu_activation_pack_async_entry>();
        entry->key = key;
        entries_.emplace(key, entry);
        queue_.push_back(entry);
        cv_.notify_one();
        return true;
    }

    void worker_loop() {
        for (;;) {
            std::shared_ptr<npu_activation_pack_async_entry> entry;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this]() { return stop_ || !queue_.empty(); });
                if (stop_ && queue_.empty()) {
                    return;
                }
                entry = queue_.front();
                queue_.pop_front();
            }

            std::vector<int8_t> data;
            std::string local_error;
            const int64_t start_us = ggml_time_us();
            const bool ok = npu_pack_activation_tile_static_asym_i8(
                plan_.src1,
                entry->key.n0,
                entry->key.n,
                entry->key.k0,
                entry->key.k,
                npu_align_up_i64(entry->key.k, NPU_GEMM_PLAN_STRIDE_ALIGNMENT),
                activation_scale_,
                plan_.activation_quant.zero_point_u8,
                plan_.aicas_w8a8.smooth_scale.empty() ? nullptr : &plan_.aicas_w8a8.smooth_scale,
                &data,
                &local_error);
            const int64_t pack_us = ggml_time_us() - start_us;
            const int64_t bytes = static_cast<int64_t>(data.size());

            {
                std::lock_guard<std::mutex> lock(entry->mutex);
                entry->failed = !ok;
                entry->error = local_error;
                if (ok) {
                    entry->data = std::move(data);
                }
                entry->ready = true;
            }
            {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                stats_.jobs += 1;
                stats_.async_us += pack_us;
                if (ok) {
                    stats_.bytes += bytes;
                }
            }
            entry->cv.notify_all();
        }
    }

    const npu_node_plan & plan_;
    float activation_scale_ = 1.0f;
    int64_t window_ = 1;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_ = false;
    std::deque<std::shared_ptr<npu_activation_pack_async_entry>> queue_;
    std::unordered_map<npu_activation_tile_key, std::shared_ptr<npu_activation_pack_async_entry>, npu_activation_tile_key_hash> entries_;
    mutable std::mutex stats_mutex_;
    npu_activation_pack_async_stats stats_;
    std::thread worker_;
};

struct npu_output_tile_key {
    int64_t m0;
    int64_t n0;
    int64_t m;
    int64_t n;

    bool operator==(const npu_output_tile_key & other) const {
        return m0 == other.m0 &&
               n0 == other.n0 &&
               m == other.m &&
               n == other.n;
    }
};

struct npu_output_tile_key_hash {
    size_t operator()(const npu_output_tile_key & key) const {
        size_t h = std::hash<int64_t>{}(key.m0);
        h ^= std::hash<int64_t>{}(key.n0) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int64_t>{}(key.m)  + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int64_t>{}(key.n)  + 0x9e3779b9 + (h << 6) + (h >> 2);
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
    const bool force_reload_activations = std::getenv("GGML_NPU_FORCE_RELOAD_ACTIVATIONS") != nullptr;
    const bool force_reload_weights = std::getenv("GGML_NPU_FORCE_RELOAD_WEIGHTS") != nullptr;
    int64_t tile_align_logs = 0;
    std::unordered_set<npu_output_tile_key, npu_output_tile_key_hash> tile_align_collect_keys;
    if (tile_align_debug_cfg.enabled) {
        for (size_t i = 0; i < plan.exec_tiles.size(); ++i) {
            if (!npu_should_check_tile_align(tile_align_debug_cfg, layer_id, static_cast<int64_t>(i))) {
                continue;
            }
            const npu_exec_tile & t = plan.exec_tiles[i];
            tile_align_collect_keys.insert({ t.m0, t.n0, t.m, t.n });
        }
    }
    std::unordered_map<npu_output_tile_key, std::vector<int32_t>, npu_output_tile_key_hash> tile_align_acc_ref;
    std::unordered_map<npu_output_tile_key, std::vector<int32_t>, npu_output_tile_key_hash> tile_align_acc_ref_alt_layout;
    std::unordered_map<npu_output_tile_key, bool, npu_output_tile_key_hash> tile_align_bias_in_accumulator;
    std::unordered_map<npu_output_tile_key, bool, npu_output_tile_key_hash> output_bias_in_accumulator;
    void * full_output_buf = nullptr;
    std::unique_ptr<npu_activation_pack_scheduler> activation_scheduler;
    bool activation_scheduler_stats_merged = false;

    auto merge_activation_scheduler_stats = [&]() {
        if (activation_scheduler == nullptr || activation_scheduler_stats_merged) {
            return;
        }
        activation_scheduler->stop();
        if (collect_stage_profile) {
            const npu_activation_pack_async_stats stats = activation_scheduler->stats();
            exec_summary.delta.activation_pack_calls += stats.jobs;
            exec_summary.delta.activation_pack_async_jobs += stats.jobs;
            exec_summary.delta.activation_pack_async_us_total += stats.async_us;
            exec_summary.delta.packed_activation_bytes_total += stats.bytes;
        }
        activation_scheduler_stats_merged = true;
    };

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
        if (full_output_buf) {
            npu_mem_free(full_output_buf);
            full_output_buf = nullptr;
        }
        if (runtime_profile_started) {
            npu_profile_end(layer_id);
            runtime_profile_started = false;
        }
        merge_activation_scheduler_stats();

        if (collect_stage_profile) {
            exec_summary.delta.total_node_us = ggml_time_us() - node_start_us;
            exec_summary.delta.accounted_us_total =
                exec_summary.delta.setup_runtime_us_total +
                exec_summary.delta.setup_validate_us_total +
                exec_summary.delta.setup_buffer_alloc_us_total +
                exec_summary.delta.setup_cache_alloc_us_total +
                exec_summary.delta.setup_profile_begin_us_total +
                exec_summary.delta.activation_pack_us_total +
                exec_summary.delta.host_copy_activation_us_total +
                exec_summary.delta.host_copy_weight_us_total +
                exec_summary.delta.bias_prepare_us_total +
                exec_summary.delta.dma_in_pair_us_total +
                exec_summary.delta.dma_in_bias_us_total +
                exec_summary.delta.w_prefetch_wait_us_total +
                exec_summary.delta.gemm_us_total +
                exec_summary.delta.dma_out_us_total +
                exec_summary.delta.postprocess_us_total;
            exec_summary.delta.unaccounted_us_total =
                std::max<int64_t>(0, exec_summary.delta.total_node_us - exec_summary.delta.accounted_us_total);
        }

        if (collect_summary) {
            npu_summary_add_delta(exec_summary.delta);
        }

        if (collect_detailed_profile) {
            profile_record.activation_pack_calls = exec_summary.delta.activation_pack_calls;
            profile_record.activation_pack_async_jobs = exec_summary.delta.activation_pack_async_jobs;
            profile_record.activation_pack_async_hits = exec_summary.delta.activation_pack_async_hits;
            profile_record.host_copy_activation_calls = exec_summary.delta.host_copy_activation_calls;
            profile_record.host_copy_weight_calls = exec_summary.delta.host_copy_weight_calls;
            profile_record.bias_prepare_calls = exec_summary.delta.bias_prepare_calls;
            profile_record.dma_in_activation_calls = exec_summary.delta.dma_in_activation_calls;
            profile_record.dma_in_weight_calls = exec_summary.delta.dma_in_weight_calls;
            profile_record.dma_in_bias_calls = exec_summary.delta.dma_in_bias_calls;
            profile_record.spm_activation_reuse_hits = exec_summary.delta.spm_activation_reuse_hits;
            profile_record.spm_weight_reuse_hits = exec_summary.delta.spm_weight_reuse_hits;
            profile_record.gemm_calls = exec_summary.delta.gemm_calls;
            profile_record.gemm_plan_calls = exec_summary.delta.gemm_plan_calls;
            profile_record.dma_out_calls = exec_summary.delta.dma_out_calls;
            profile_record.postprocess_calls = exec_summary.delta.postprocess_calls;
            profile_record.raw_acc_mvout_nodes = exec_summary.delta.raw_acc_mvout_nodes;
            profile_record.raw_acc_mvout_tiles = exec_summary.delta.raw_acc_mvout_tiles;
            profile_record.w_prefetch_calls = exec_summary.delta.w_prefetch_calls;
            profile_record.w_prefetch_hits = exec_summary.delta.w_prefetch_hits;
            profile_record.w_prefetch_conflicts = exec_summary.delta.w_prefetch_conflicts;
            profile_record.packed_activation_bytes_total = exec_summary.delta.packed_activation_bytes_total;
            profile_record.copied_weight_bytes_total = exec_summary.delta.copied_weight_bytes_total;
            profile_record.dma_in_activation_bytes_total = exec_summary.delta.dma_in_activation_bytes_total;
            profile_record.dma_in_weight_bytes_total = exec_summary.delta.dma_in_weight_bytes_total;
            profile_record.bias_bytes_total = exec_summary.delta.bias_bytes_total;
            profile_record.acc_readback_bytes_total = exec_summary.delta.acc_readback_bytes_total;
            profile_record.output_write_bytes_total = exec_summary.delta.output_write_bytes_total;
            profile_record.setup_runtime_us_total = exec_summary.delta.setup_runtime_us_total;
            profile_record.setup_validate_us_total = exec_summary.delta.setup_validate_us_total;
            profile_record.setup_buffer_alloc_us_total = exec_summary.delta.setup_buffer_alloc_us_total;
            profile_record.setup_cache_alloc_us_total = exec_summary.delta.setup_cache_alloc_us_total;
            profile_record.setup_profile_begin_us_total = exec_summary.delta.setup_profile_begin_us_total;
            profile_record.activation_pack_us_total = exec_summary.delta.activation_pack_us_total;
            profile_record.activation_pack_async_us_total = exec_summary.delta.activation_pack_async_us_total;
            profile_record.activation_pack_wait_us_total = exec_summary.delta.activation_pack_wait_us_total;
            profile_record.host_copy_activation_us_total = exec_summary.delta.host_copy_activation_us_total;
            profile_record.host_copy_weight_us_total = exec_summary.delta.host_copy_weight_us_total;
            profile_record.bias_prepare_us_total = exec_summary.delta.bias_prepare_us_total;
            profile_record.dma_in_activation_us_total = exec_summary.delta.dma_in_activation_us_total;
            profile_record.dma_in_weight_us_total = exec_summary.delta.dma_in_weight_us_total;
            profile_record.dma_in_bias_us_total = exec_summary.delta.dma_in_bias_us_total;
            profile_record.dma_in_pair_calls = exec_summary.delta.dma_in_pair_calls;
            profile_record.dma_in_pair_us_total = exec_summary.delta.dma_in_pair_us_total;
            profile_record.w_prefetch_wait_us_total = exec_summary.delta.w_prefetch_wait_us_total;
            profile_record.w_prefetch_hidden_candidate_us_total = exec_summary.delta.w_prefetch_hidden_candidate_us_total;
            profile_record.gemm_us_total = exec_summary.delta.gemm_us_total;
            profile_record.dma_out_us_total = exec_summary.delta.dma_out_us_total;
            profile_record.postprocess_us_total = exec_summary.delta.postprocess_us_total;
            profile_record.accounted_us_total = static_cast<double>(exec_summary.delta.accounted_us_total);
            profile_record.unaccounted_us_total = static_cast<double>(exec_summary.delta.unaccounted_us_total);
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
    const int64_t setup_runtime_start_us = collect_stage_profile ? ggml_time_us() : 0;
    if (!npu_ensure_runtime(error)) {
        if (collect_stage_profile) {
            exec_summary.delta.setup_runtime_us_total += ggml_time_us() - setup_runtime_start_us;
        }
        return finalize_status(GGML_STATUS_FAILED);
    }
    if (collect_stage_profile) {
        exec_summary.delta.setup_runtime_us_total += ggml_time_us() - setup_runtime_start_us;
    }

    std::string aot_reason;
    const int64_t setup_validate_start_us = collect_stage_profile ? ggml_time_us() : 0;
    if (!npu_plan_is_aot_stable(plan, &aot_reason)) {
        if (collect_stage_profile) {
            exec_summary.delta.setup_validate_us_total += ggml_time_us() - setup_validate_start_us;
        }
        if (error) {
            *error = "AOT plan invalid: " + aot_reason;
        }
        return finalize_status(GGML_STATUS_FAILED);
    }
    if (collect_stage_profile) {
        exec_summary.delta.setup_validate_us_total += ggml_time_us() - setup_validate_start_us;
    }

    const int64_t max_n_logical = plan.use_gemm_plan ? plan.first_stage_tn : plan.config.sa_rows;
    const int64_t max_n = npu_hardware_n_rows(max_n_logical);
    const int64_t max_m = plan.use_gemm_plan ? plan.first_stage_tm : plan.config.sa_cols;
    const int64_t max_k = plan.use_gemm_plan
        ? plan.config.k_block
        : std::min<int64_t>(plan.config.k_block, plan.config.stage2_k_block);
    const int64_t max_k_stride = npu_align_up_i64(max_k, NPU_GEMM_PLAN_STRIDE_ALIGNMENT);

    void * activation_buf = nullptr;
    void * acc_buf = nullptr;
    void * bias_buf = nullptr;
    void * bias_cache_buf = nullptr;
    void * scale_cache_buf = nullptr;

    const int64_t setup_buffer_alloc_start_us = collect_stage_profile ? ggml_time_us() : 0;
    if (!npu_allocate_runtime_buffer(static_cast<size_t>(max_n * max_k_stride), &activation_buf, error) ||
        !npu_allocate_runtime_buffer(static_cast<size_t>(max_n * max_m * sizeof(int32_t)), &acc_buf, error) ||
        !npu_allocate_runtime_buffer(static_cast<size_t>(max_n * max_m * sizeof(int32_t)), &bias_buf, error)) {
        if (collect_stage_profile) {
            exec_summary.delta.setup_buffer_alloc_us_total += ggml_time_us() - setup_buffer_alloc_start_us;
        }
        cleanup_buffers(activation_buf, nullptr, acc_buf, bias_buf, nullptr, nullptr);
        return finalize_status(GGML_STATUS_ALLOC_FAILED);
    }
    if (collect_stage_profile) {
        exec_summary.delta.setup_buffer_alloc_us_total += ggml_time_us() - setup_buffer_alloc_start_us;
    }

    std::unordered_map<npu_activation_tile_key, std::vector<int8_t>, npu_activation_tile_key_hash> activation_tile_cache;
    const float activation_scale = plan.activation_quant.scale;
    const bool use_aicas_w8a8 = plan.aicas_w8a8.valid;
    const npu_bias_mode bias_mode = npu_resolve_bias_mode(plan);
    const bool hardware_asymmetric_activations =
        plan.activation_quant.valid && !plan.activation_quant.symmetric;
    const bool fold_output_reconstruction = npu_can_fold_output_reconstruction(plan);
    // GEMM_v6_app runtime now supports first-K hardware bias/compensation
    // correctly, so gemm_plan should use the ACC initializer path instead of
    // the older software-compensation fallback.
    const bool use_software_bias_for_gemm_plan = false;
    // AICAS W8A8 asymmetric activations need zero-point compensation. In the
    // default precomp mode, clip.cpp has already folded this into model bias
    // tensors, so only bias-free GEMMs still need runtime compensation.
    const bool apply_compensation =
        use_aicas_w8a8 && (bias_mode == npu_bias_mode::raw || plan.bias == nullptr);
    const bool fold_compensation_into_accumulator =
        apply_compensation && !use_software_bias_for_gemm_plan;
    const bool fold_compensation_into_init_bias = fold_compensation_into_accumulator;
    const bool apply_output_compensation =
        apply_compensation && !fold_compensation_into_accumulator;
    const bool raw_acc_mvout = npu_force_raw_acc_mvout();
    if (raw_acc_mvout) {
        static bool raw_acc_mvout_warned = false;
        if (!raw_acc_mvout_warned) {
            const char * root_name = plan.root && plan.root->name[0] != '\0' ? plan.root->name : "(unnamed)";
            GGML_LOG_WARN("%s: raw int32 accumulator MVOUT is active for root=%s; postprocess will dequantize on CPU\n",
                    __func__, root_name);
            raw_acc_mvout_warned = true;
        }
        exec_summary.delta.raw_acc_mvout_nodes = 1;
    }
    const bool mvout_per_channel = fold_output_reconstruction && plan.aicas_w8a8.weight_scale.size() > 1;
    const float per_tensor_weight_scale = fold_output_reconstruction ? plan.aicas_w8a8.weight_scale[0] : 1.0f;
    const bool use_bias_cache =
        npu_env_enabled_default("GGML_NPU_BIAS_CACHE", true) &&
        plan.config.layout.bias_cache.bytes >= static_cast<uint32_t>(plan.m * sizeof(int32_t)) &&
        (plan.bias != nullptr || fold_compensation_into_init_bias);
    const bool use_scale_cache = false;
    std::vector<float> acc_scaled_values;
    std::vector<int32_t> acc_raw_values_host;
    std::vector<float> bias_tile_values;
    std::vector<int32_t> bias_values;
    std::vector<npu_full_output_span> full_output_spans;
    bool full_output_cma_active = false;

    if (npu_full_output_cma_enabled() &&
        fold_output_reconstruction &&
        !raw_acc_mvout &&
        !apply_output_compensation &&
        !tile_align_debug_cfg.enabled &&
        npu_dst_is_dense_f32(plan.dst) &&
        plan.m > 0) {
        std::unordered_map<npu_output_tile_key, bool, npu_output_tile_key_hash> prescan_bias_in_accumulator;
        bool all_outputs_are_final_f32 = true;
        for (const npu_exec_tile & tile : plan.exec_tiles) {
            const npu_output_tile_key key{ tile.m0, tile.n0, tile.m, tile.n };
            const bool tile_has_model_bias =
                plan.bias != nullptr && tile.needs_bias && tile.bias_pack_index >= 0;
            const bool tile_uses_init_bias =
                tile.needs_bias && fold_compensation_into_init_bias;
            const bool tile_uses_bias = tile_has_model_bias || tile_uses_init_bias;
            if (tile_has_model_bias) {
                prescan_bias_in_accumulator[key] = true;
            }
            const auto bias_it = prescan_bias_in_accumulator.find(key);
            const bool output_has_bias_in_accumulator =
                bias_it != prescan_bias_in_accumulator.end() && bias_it->second;
            if (tile.writes_output &&
                !(tile_uses_bias || output_has_bias_in_accumulator || plan.bias == nullptr)) {
                all_outputs_are_final_f32 = false;
                break;
            }
        }

        if (all_outputs_are_final_f32) {
            full_output_spans = npu_build_full_output_spans(plan);
            int64_t padded_rows = 0;
            for (const npu_full_output_span & span : full_output_spans) {
                padded_rows = std::max(padded_rows, span.padded_row0 + span.hw_n);
            }
            const uint64_t full_output_bytes =
                static_cast<uint64_t>(padded_rows) *
                static_cast<uint64_t>(plan.m) *
                sizeof(float);
            if (full_output_bytes != 0 &&
                full_output_bytes <= static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
                full_output_buf = npu_mem_alloc(static_cast<size_t>(full_output_bytes));
                full_output_cma_active = full_output_buf != nullptr;
                if (!full_output_cma_active && npu_debug_log_enabled()) {
                    GGML_LOG_WARN("%s: GGML_NPU_FULL_OUTPUT_CMA requested but full output allocation failed, bytes=%" PRIu64 "\n",
                        __func__, full_output_bytes);
                }
            }
        }
    }

    const int64_t setup_cache_alloc_start_us = collect_stage_profile ? ggml_time_us() : 0;
    if (use_bias_cache &&
        !npu_allocate_runtime_buffer(static_cast<size_t>(plan.config.layout.bias_cache.bytes), &bias_cache_buf, error)) {
        if (collect_stage_profile) {
            exec_summary.delta.setup_cache_alloc_us_total += ggml_time_us() - setup_cache_alloc_start_us;
        }
        cleanup_buffers(activation_buf, nullptr, acc_buf, bias_buf, nullptr, nullptr);
        return finalize_status(GGML_STATUS_ALLOC_FAILED);
    }
    if (use_scale_cache &&
        !npu_allocate_runtime_buffer(static_cast<size_t>(plan.config.layout.scale_cache.bytes), &scale_cache_buf, error)) {
        if (collect_stage_profile) {
            exec_summary.delta.setup_cache_alloc_us_total += ggml_time_us() - setup_cache_alloc_start_us;
        }
        cleanup_buffers(activation_buf, nullptr, acc_buf, bias_buf, bias_cache_buf, nullptr);
        return finalize_status(GGML_STATUS_ALLOC_FAILED);
    }
    if (collect_stage_profile) {
        exec_summary.delta.setup_cache_alloc_us_total += ggml_time_us() - setup_cache_alloc_start_us;
    }

    activation_tile_cache.reserve(plan.exec_tiles.size());
    if (npu_env_enabled_default("GGML_NPU_ACT_PACK_ASYNC", true) &&
            !plan.exec_tiles.empty() &&
            !tile_align_debug_cfg.enabled) {
        const int64_t async_window =
            std::max<int64_t>(1, npu_env_i64("GGML_NPU_ACT_PACK_ASYNC_WINDOW", 8));
        activation_scheduler = std::make_unique<npu_activation_pack_scheduler>(
            plan,
            activation_scale,
            async_window);
    }

    int64_t active_m0 = -1;
    int64_t active_n0 = -1;
    int64_t active_m = 0;
    int64_t active_n = 0;
    int64_t loaded_bias_cache_n0 = -1;
    int64_t loaded_scale_cache_m0 = -1;
    int64_t loaded_scale_cache_m = -1;
    npu_activation_tile_key loaded_activation_key;
    bool loaded_activation_valid = false;
    int32_t loaded_weight_pack_index = -1;
    const bool w_pingpong_enabled =
        npu_w_pingpong_enabled() && plan.use_gemm_plan && !force_reload_weights;
    bool resident_weight_valid[2] = { false, false };
    bool resident_weight_prefetched[2] = { false, false };
    int32_t resident_weight_pack_index[2] = { -1, -1 };
    uint8_t last_gemm_w_bank = 1;
    auto find_resident_weight_bank = [&](int32_t weight_pack_index) -> int {
        for (int bank = 0; bank < 2; ++bank) {
            if (resident_weight_valid[bank] && resident_weight_pack_index[bank] == weight_pack_index) {
                return bank;
            }
        }
        return -1;
    };
    auto mark_weight_bank_loaded = [&](uint8_t bank, int32_t weight_pack_index, bool prefetched) {
        resident_weight_valid[bank] = true;
        resident_weight_pack_index[bank] = weight_pack_index;
        resident_weight_prefetched[bank] = prefetched;
    };
    if (collect_runtime_profile) {
        const int64_t setup_profile_begin_start_us = collect_stage_profile ? ggml_time_us() : 0;
        npu_profile_begin(layer_id);
        runtime_profile_started = true;
        if (collect_stage_profile) {
            exec_summary.delta.setup_profile_begin_us_total += ggml_time_us() - setup_profile_begin_start_us;
        }
    }

    for (size_t exec_tile_idx = 0; exec_tile_idx < plan.exec_tiles.size(); ++exec_tile_idx) {
        const npu_exec_tile & exec_tile = plan.exec_tiles[exec_tile_idx];
        const int64_t hw_n = npu_hardware_n_rows(exec_tile.n);
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
        std::shared_ptr<npu_activation_pack_async_entry> async_activation_entry;
        if (activation_scheduler != nullptr) {
            activation_scheduler->schedule_window(exec_tile_idx);
            bool ready_before_wait = false;
            const int64_t activation_wait_start_us = collect_stage_profile ? ggml_time_us() : 0;
            async_activation_entry = activation_scheduler->wait_ready(
                activation_key,
                error,
                &ready_before_wait);
            const int64_t activation_wait_us =
                collect_stage_profile ? (ggml_time_us() - activation_wait_start_us) : 0;
            if (async_activation_entry == nullptr) {
                if (collect_stage_profile) {
                    exec_summary.delta.activation_pack_us_total += activation_wait_us;
                    exec_summary.delta.activation_pack_wait_us_total += activation_wait_us;
                }
                if (collect_tile_profile) {
                    tile_record.activation_pack_us = static_cast<double>(activation_wait_us);
                    profile_record.tiles.push_back(std::move(tile_record));
                }
                cleanup_buffers(activation_buf, nullptr, acc_buf, bias_buf, bias_cache_buf, scale_cache_buf);
                return finalize_status(GGML_STATUS_FAILED);
            }
            packed_activation_view = &async_activation_entry->data;
            if (collect_stage_profile) {
                exec_summary.delta.activation_pack_us_total += activation_wait_us;
                exec_summary.delta.activation_pack_wait_us_total += activation_wait_us;
                if (ready_before_wait) {
                    exec_summary.delta.activation_pack_async_hits += 1;
                }
                if (collect_tile_profile) {
                    tile_record.activation_pack_us = static_cast<double>(activation_wait_us);
                }
            }
        } else {
            const auto cache_it = activation_tile_cache.find(activation_key);
            if (cache_it != activation_tile_cache.end()) {
                packed_activation_view = &cache_it->second;
                if (collect_tile_profile) {
                    tile_record.activation_pack_us = 0.0;
                }
            } else {
                const int64_t activation_pack_start_us = collect_stage_profile ? ggml_time_us() : 0;
                auto inserted = activation_tile_cache.try_emplace(activation_key);
                std::vector<int8_t> & cached_activation = inserted.first->second;
                if (!npu_pack_activation_tile_static_asym_i8(
                            plan.src1,
                            exec_tile.n0,
                            exec_tile.n,
                            exec_tile.k0,
                            exec_tile.k,
                            exec_tile.a_stride,
                            activation_scale,
                            plan.activation_quant.zero_point_u8,
                            plan.aicas_w8a8.smooth_scale.empty() ? nullptr : &plan.aicas_w8a8.smooth_scale,
                            &cached_activation,
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
                    activation_tile_cache.erase(inserted.first);
                    cleanup_buffers(activation_buf, nullptr, acc_buf, bias_buf, bias_cache_buf, scale_cache_buf);
                    return finalize_status(GGML_STATUS_FAILED);
                }
                packed_activation_view = &cached_activation;
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

        int current_w_bank = -1;
        const int resident_weight_bank = w_pingpong_enabled
            ? find_resident_weight_bank(exec_tile.weight_pack_index)
            : -1;
        const bool activation_already_in_spm =
            !force_reload_activations && loaded_activation_valid && loaded_activation_key == activation_key;
        const bool weight_already_in_spm = w_pingpong_enabled
            ? resident_weight_bank >= 0
            : (!force_reload_weights && loaded_weight_pack_index == exec_tile.weight_pack_index);
        if (weight_already_in_spm && resident_weight_bank >= 0) {
            current_w_bank = resident_weight_bank;
            if (resident_weight_prefetched[resident_weight_bank]) {
                exec_summary.delta.w_prefetch_hits += 1;
                resident_weight_prefetched[resident_weight_bank] = false;
                if (collect_tile_profile) {
                    tile_record.w_prefetch_hit = true;
                }
            }
        }
        if (collect_stage_profile) {
            if (activation_already_in_spm) {
                exec_summary.delta.spm_activation_reuse_hits += 1;
            }
            if (weight_already_in_spm) {
                exec_summary.delta.spm_weight_reuse_hits += 1;
            }
        }
        if (collect_tile_profile) {
            tile_record.activation_already_in_spm = activation_already_in_spm;
            tile_record.weight_already_in_spm = weight_already_in_spm;
            tile_record.w_bank = current_w_bank;
        }

        const int64_t activation_copy_start_us = collect_stage_profile ? ggml_time_us() : 0;
        if (!activation_already_in_spm) {
            if (hw_n != exec_tile.n) {
                std::memset(
                    activation_buf,
                    0,
                    static_cast<size_t>(hw_n * exec_tile.a_stride));
            }
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
            static_cast<uint32_t>(exec_tile.k),
            static_cast<uint32_t>(hw_n),
            static_cast<uint16_t>(exec_tile.a_stride),
            static_cast<uint32_t>(exec_tile.a_stride),
            1,
            0,
            false,
            false,
            hardware_asymmetric_activations,
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
                packed_activation_view->size());
            npu_dump_tile_blob(
                dump_dir,
                layer_id,
                static_cast<int64_t>(exec_tile_idx),
                "wgt_q",
                packed_weight.packed.data(),
                packed_weight.packed.size());
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
            static_cast<uint32_t>(exec_tile.m),
            static_cast<uint32_t>(exec_tile.k),
            static_cast<uint16_t>(packed_weight.stride_m),
            static_cast<uint32_t>(packed_weight.stride_m),
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
        uint32_t waited_mvin_mask = 0;
        bool weight_mvin_issued = false;
        uint8_t loaded_w_bank_now = 0;
        if (!activation_already_in_spm) {
            npu_dma_mvin_async(0, &activation_mvin_cfg);
            mvin_mask |= (1u << 0);
            waited_mvin_mask |= (1u << 0);
        }
        if (!weight_already_in_spm) {
#if defined(GGML_NPU_VERSA_P_RUNTIME)
            if (w_pingpong_enabled) {
                loaded_w_bank_now = static_cast<uint8_t>(last_gemm_w_bank ^ 1u);
                resident_weight_valid[loaded_w_bank_now] = false;
                resident_weight_pack_index[loaded_w_bank_now] = -1;
                resident_weight_prefetched[loaded_w_bank_now] = false;
                npu_dma_mvin_w_async_bank(loaded_w_bank_now, &weight_mvin_cfg);
                weight_mvin_issued = true;
                current_w_bank = loaded_w_bank_now;
                mvin_mask |= (1u << 1);
            } else
#endif
            {
                npu_dma_mvin_async(1, &weight_mvin_cfg);
                mvin_mask |= (1u << 1);
                waited_mvin_mask |= (1u << 1);
            }
        }
        if (waited_mvin_mask != 0) {
            npu_dma_wait_mvin(waited_mvin_mask);
        }
#if defined(GGML_NPU_VERSA_P_RUNTIME)
        if (weight_mvin_issued) {
            npu_dma_wait_w_bank(loaded_w_bank_now);
        }
#endif
        if (!activation_already_in_spm) {
            loaded_activation_key = activation_key;
            loaded_activation_valid = true;
        }
        if (!weight_already_in_spm) {
            if (w_pingpong_enabled) {
                mark_weight_bank_loaded(loaded_w_bank_now, exec_tile.weight_pack_index, false);
            } else {
                loaded_weight_pack_index = exec_tile.weight_pack_index;
            }
        }
        if (collect_tile_profile) {
            tile_record.mvin_mask = mvin_mask;
            tile_record.w_bank = current_w_bank;
        }
        if (collect_stage_profile && mvin_mask != 0) {
            const int64_t dma_in_pair_us = ggml_time_us() - dma_in_pair_start_us;
            exec_summary.delta.dma_in_pair_calls += 1;
            exec_summary.delta.dma_in_pair_us_total += dma_in_pair_us;
            if (!activation_already_in_spm) {
                exec_summary.delta.dma_in_activation_calls += 1;
                exec_summary.delta.dma_in_activation_bytes_total += static_cast<int64_t>(packed_activation_view->size());
            }
            if (!weight_already_in_spm) {
                exec_summary.delta.dma_in_weight_calls += 1;
                exec_summary.delta.dma_in_weight_bytes_total += static_cast<int64_t>(packed_weight.packed.size());
            }
            if (collect_tile_profile) {
                tile_record.dma_in_pair_us = static_cast<double>(dma_in_pair_us);
                tile_record.dma_in_activation_us = 0.0;
                tile_record.dma_in_weight_us = 0.0;
            }
        }

        const npu_output_tile_key tile_align_key {
            exec_tile.m0,
            exec_tile.n0,
            exec_tile.m,
            exec_tile.n,
        };
        const bool tile_has_model_bias =
            !use_software_bias_for_gemm_plan && exec_tile.needs_bias && exec_tile.bias_pack_index >= 0;
        const bool tile_uses_init_bias =
            exec_tile.needs_bias && fold_compensation_into_init_bias;
        const bool tile_uses_bias = tile_has_model_bias || tile_uses_init_bias;
        if (tile_has_model_bias) {
            output_bias_in_accumulator[tile_align_key] = true;
        }
        const auto output_bias_in_acc_it = output_bias_in_accumulator.find(tile_align_key);
        const bool output_has_bias_in_accumulator =
            output_bias_in_acc_it != output_bias_in_accumulator.end() && output_bias_in_acc_it->second;
        if (tile_uses_bias) {
            if (tile_has_model_bias &&
                (exec_tile.bias_pack_index < 0 ||
                 static_cast<size_t>(exec_tile.bias_pack_index) >= plan.bias_packs.size())) {
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
                        const float bias_f32 = plan.bias != nullptr
                            ? npu_read_bias_value_f32_exec(plan.bias, global_m, exec_tile.n0)
                            : 0.0f;
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
                        static_cast<uint32_t>(plan.m),
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
                const npu_prepacked_bias * bias_pack = tile_has_model_bias
                    ? &plan.bias_packs[static_cast<size_t>(exec_tile.bias_pack_index)]
                    : nullptr;
                if (bias_pack != nullptr && bias_pack->values.size() < static_cast<size_t>(exec_tile.m)) {
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
                if (bias_pack != nullptr) {
                    std::copy_n(
                        bias_pack->values.begin(),
                        static_cast<size_t>(exec_tile.m),
                        bias_values.begin());
                } else {
                    std::fill_n(bias_values.begin(), static_cast<size_t>(exec_tile.m), 0);
                }
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
                    static_cast<uint32_t>(exec_tile.m),
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
        int64_t gemm_us = 0;
        int64_t w_prefetch_wait_us = 0;
        if (npu_debug_log_enabled()) {
                GGML_LOG_INFO("%s: tile m0=%" PRId64 " n0=%" PRId64 " k0=%" PRId64 " GEMM n=%" PRId64 " m=%" PRId64 " k=%" PRId64 " bias_acc=0x%08x out_acc=0x%08x\n",
                    __func__,
                    exec_tile.m0, exec_tile.n0, exec_tile.k0,
                    hw_n, exec_tile.m, exec_tile.k,
                    plan.config.layout.bias_accumulator.offset,
                    plan.config.layout.output_accumulator.offset);
        }
        const bool use_hardware_bias = tile_uses_bias;
        const bool use_accumulate = !exec_tile.needs_bias || use_hardware_bias;
        const uint32_t gemm_biaspsum_addr = use_hardware_bias
            ? (use_bias_cache
                ? plan.config.layout.bias_cache.offset + static_cast<uint32_t>(exec_tile.m0 * sizeof(int32_t))
                : plan.config.layout.bias_accumulator.offset)
            : (!exec_tile.needs_bias ? plan.config.layout.output_accumulator.offset : plan.config.layout.bias_accumulator.offset);
#if defined(GGML_NPU_VERSA_P_RUNTIME)
        if (plan.use_gemm_plan && w_pingpong_enabled) {
            if (current_w_bank < 0) {
                if (error) {
                    *error = "W pingpong selected but no weight bank is resident";
                }
                if (collect_tile_profile) {
                    profile_record.tiles.push_back(std::move(tile_record));
                }
                cleanup_buffers(activation_buf, nullptr, acc_buf, bias_buf, bias_cache_buf, scale_cache_buf);
                return finalize_status(GGML_STATUS_FAILED);
            }
            npu_gemm_plan_start_ex_bank(
                static_cast<uint8_t>(current_w_bank),
                /*a_addr=*/plan.config.layout.activation.offset,
                /*b_addr=*/plan.config.layout.weight.offset,
                /*out_addr=*/plan.config.layout.output_accumulator.offset,
                /*scratch_addr=*/plan.config.layout.scratch_accumulator.offset,
                /*bias_addr=*/use_hardware_bias ? gemm_biaspsum_addr : 0,
                /*block_m=*/static_cast<uint16_t>(hw_n),
                /*block_n=*/static_cast<uint16_t>(exec_tile.m),
                /*block_k=*/static_cast<uint16_t>(exec_tile.k),
                /*a_stride=*/static_cast<uint16_t>(exec_tile.a_stride),
                /*b_stride=*/static_cast<uint16_t>(exec_tile.b_stride),
                /*out_stride=*/static_cast<uint16_t>(exec_tile.out_stride),
                /*bias_stride=*/static_cast<uint16_t>(exec_tile.bias_stride),
                /*have_bias=*/use_hardware_bias,
                /*is_accumulate=*/use_accumulate,
                /*asymmetric_activations=*/hardware_asymmetric_activations);

            bool prefetch_issued = false;
            uint8_t prefetch_w_bank = static_cast<uint8_t>(current_w_bank ^ 1);
            int32_t prefetch_weight_pack_index = -1;
            for (size_t next_idx = exec_tile_idx + 1; next_idx < plan.exec_tiles.size(); ++next_idx) {
                const npu_exec_tile & next_tile = plan.exec_tiles[next_idx];
                if (next_tile.weight_pack_index < 0 ||
                    static_cast<size_t>(next_tile.weight_pack_index) >= plan.weight_packs.size()) {
                    continue;
                }
                if (find_resident_weight_bank(next_tile.weight_pack_index) >= 0) {
                    continue;
                }
                prefetch_weight_pack_index = next_tile.weight_pack_index;
                const npu_prepacked_weight & next_weight =
                    plan.weight_packs[static_cast<size_t>(next_tile.weight_pack_index)];
                bool prefetch_weight_copied = false;
                const int64_t prefetch_copy_start_us = collect_stage_profile ? ggml_time_us() : 0;
                if (!npu_ensure_weight_pack_cma(next_weight, error, &prefetch_weight_copied)) {
                    npu_gemm_plan_wait();
                    if (collect_tile_profile) {
                        profile_record.tiles.push_back(std::move(tile_record));
                    }
                    cleanup_buffers(activation_buf, nullptr, acc_buf, bias_buf, bias_cache_buf, scale_cache_buf);
                    return finalize_status(GGML_STATUS_FAILED);
                }
                if (collect_stage_profile && prefetch_weight_copied) {
                    const int64_t prefetch_copy_us = ggml_time_us() - prefetch_copy_start_us;
                    exec_summary.delta.host_copy_weight_calls += 1;
                    exec_summary.delta.host_copy_weight_us_total += prefetch_copy_us;
                    exec_summary.delta.copied_weight_bytes_total += static_cast<int64_t>(next_weight.packed.size());
                }
                const MvinConfig prefetch_weight_mvin_cfg {
                    next_weight.cma_packed,
                    plan.config.layout.weight.offset,
                    static_cast<uint32_t>(next_tile.m),
                    static_cast<uint32_t>(next_tile.k),
                    static_cast<uint16_t>(next_weight.stride_m),
                    static_cast<uint32_t>(next_weight.stride_m),
                    1,
                    1,
                    false,
                    false,
                    false,
                    0,
                    0,
                    0,
                };
                try {
                    npu_dma_mvin_w_async_bank(prefetch_w_bank, &prefetch_weight_mvin_cfg);
                    resident_weight_valid[prefetch_w_bank] = false;
                    resident_weight_pack_index[prefetch_w_bank] = -1;
                    resident_weight_prefetched[prefetch_w_bank] = false;
                    prefetch_issued = true;
                    if (collect_stage_profile) {
                        exec_summary.delta.w_prefetch_calls += 1;
                        exec_summary.delta.dma_in_weight_calls += 1;
                        exec_summary.delta.dma_in_weight_bytes_total += static_cast<int64_t>(next_weight.packed.size());
                    }
                    if (collect_tile_profile) {
                        tile_record.w_prefetch_issued = true;
                    }
                } catch (const std::exception & ex) {
                    if (collect_stage_profile) {
                        exec_summary.delta.w_prefetch_conflicts += 1;
                    }
                    if (npu_debug_log_enabled()) {
                        GGML_LOG_INFO("%s: W prefetch skipped bank=%u weight_pack=%" PRId32 ": %s\n",
                                __func__, static_cast<unsigned>(prefetch_w_bank), prefetch_weight_pack_index, ex.what());
                    }
                }
                break;
            }

            npu_gemm_plan_wait();
            if (collect_stage_profile) {
                gemm_us = ggml_time_us() - gemm_start_us;
            }
            if (prefetch_issued) {
                const int64_t prefetch_wait_start_us = collect_stage_profile ? ggml_time_us() : 0;
                npu_dma_wait_w_bank(prefetch_w_bank);
                if (collect_stage_profile) {
                    w_prefetch_wait_us = ggml_time_us() - prefetch_wait_start_us;
                    exec_summary.delta.w_prefetch_wait_us_total += w_prefetch_wait_us;
                    exec_summary.delta.w_prefetch_hidden_candidate_us_total +=
                        std::max<int64_t>(0, gemm_us - w_prefetch_wait_us);
                }
                mark_weight_bank_loaded(prefetch_w_bank, prefetch_weight_pack_index, true);
            }
            last_gemm_w_bank = static_cast<uint8_t>(current_w_bank);
        } else
#endif
        if (plan.use_gemm_plan) {
            npu_gemm_plan_run_ex(
                /*a_addr=*/plan.config.layout.activation.offset,
                /*b_addr=*/plan.config.layout.weight.offset,
                /*out_addr=*/plan.config.layout.output_accumulator.offset,
                /*scratch_addr=*/plan.config.layout.scratch_accumulator.offset,
                /*bias_addr=*/use_hardware_bias ? gemm_biaspsum_addr : 0,
                /*block_m=*/static_cast<uint16_t>(hw_n),
                /*block_n=*/static_cast<uint16_t>(exec_tile.m),
                /*block_k=*/static_cast<uint16_t>(exec_tile.k),
                /*a_stride=*/static_cast<uint16_t>(exec_tile.a_stride),
                /*b_stride=*/static_cast<uint16_t>(exec_tile.b_stride),
                /*out_stride=*/static_cast<uint16_t>(exec_tile.out_stride),
                /*bias_stride=*/static_cast<uint16_t>(exec_tile.bias_stride),
                /*have_bias=*/use_hardware_bias,
                /*is_accumulate=*/use_accumulate,
                /*asymmetric_activations=*/hardware_asymmetric_activations);
        } else {
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
                /*isaccu=*/use_accumulate,
                /*relu=*/false,
                /*relu_type=*/0,
                /*is_bias=*/use_hardware_bias,
                /*input_a_addr=*/plan.config.layout.activation.offset,
                /*input_a_col_num=*/static_cast<uint16_t>(exec_tile.k - 1),
                /*input_a_row_num=*/static_cast<uint8_t>(hw_n - 1),
                /*input_a_stride=*/static_cast<uint16_t>(exec_tile.k),
                /*input_b_addr=*/plan.config.layout.weight.offset,
                /*input_b_col_num=*/static_cast<uint8_t>(exec_tile.m - 1),
                /*input_b_row_num=*/static_cast<uint16_t>(exec_tile.k - 1),
                /*input_b_stride=*/static_cast<uint16_t>(exec_tile.m),
                /*asymmetric_activations=*/hardware_asymmetric_activations);
        }
        if (collect_stage_profile) {
            if (gemm_us == 0) {
                gemm_us = ggml_time_us() - gemm_start_us;
            }
            exec_summary.delta.gemm_calls += 1;
            if (plan.use_gemm_plan) {
                exec_summary.delta.gemm_plan_calls += 1;
            }
            exec_summary.delta.gemm_us_total += gemm_us;
            if (collect_tile_profile) {
                tile_record.gemm_us = static_cast<double>(gemm_us);
                tile_record.w_prefetch_wait_us = static_cast<double>(w_prefetch_wait_us);
            }
        }

        const bool tile_align_collect_current =
            tile_align_collect_keys.find(tile_align_key) != tile_align_collect_keys.end();
        if (tile_align_collect_current) {
            const size_t tile_elems = static_cast<size_t>(exec_tile.n * exec_tile.m);
            auto & acc_ref_tile = tile_align_acc_ref[tile_align_key];
            auto & acc_ref_alt_tile = tile_align_acc_ref_alt_layout[tile_align_key];
            if (exec_tile.needs_bias ||
                acc_ref_tile.size() != tile_elems ||
                acc_ref_alt_tile.size() != tile_elems) {
                acc_ref_tile.assign(tile_elems, 0);
                acc_ref_alt_tile.assign(tile_elems, 0);

                const npu_prepacked_bias * bias_pack = tile_has_model_bias
                    ? &plan.bias_packs[static_cast<size_t>(exec_tile.bias_pack_index)]
                    : nullptr;
                const bool model_bias_in_acc = bias_pack != nullptr || output_has_bias_in_accumulator;
                tile_align_bias_in_accumulator[tile_align_key] = model_bias_in_acc;
                for (int64_t n = 0; n < exec_tile.n; ++n) {
                    for (int64_t m = 0; m < exec_tile.m; ++m) {
                        const size_t idx = static_cast<size_t>(n * exec_tile.m + m);
                        const int64_t global_m = exec_tile.m0 + m;
                        int32_t init = bias_pack != nullptr
                            ? bias_pack->values[static_cast<size_t>(m)]
                            : 0;
                        if (fold_compensation_into_accumulator &&
                            global_m >= 0 &&
                            static_cast<size_t>(global_m) < plan.weight_column_sum_q.size()) {
                            init -= plan.activation_quant.zero_point *
                                plan.weight_column_sum_q[static_cast<size_t>(global_m)];
                        }
                        acc_ref_tile[idx] = init;
                        acc_ref_alt_tile[idx] = init;
                    }
                }
            }

            const int8_t * act_q = packed_activation_view->data();
            const int8_t * w_q = packed_weight.packed.data();
            for (int64_t n = 0; n < exec_tile.n; ++n) {
                for (int64_t m = 0; m < exec_tile.m; ++m) {
                    int32_t partial = 0;
                    int32_t partial_alt_layout = 0;
                    for (int64_t k = 0; k < exec_tile.k; ++k) {
                        const int32_t qa = npu_effective_activation_q(
                            plan,
                            act_q[static_cast<size_t>(n * exec_tile.a_stride + k)]);
                        partial += qa * static_cast<int32_t>(
                            npu_packed_weight_at(packed_weight, k, m));
                        partial_alt_layout += qa * static_cast<int32_t>(
                            w_q[static_cast<size_t>(m * exec_tile.k + k)]);
                    }
                    const size_t idx = static_cast<size_t>(n * exec_tile.m + m);
                    acc_ref_tile[idx] += partial;
                    acc_ref_alt_tile[idx] += partial_alt_layout;
                }
            }
        }

        if (exec_tile.writes_output) {
            if (collect_stage_profile && raw_acc_mvout) {
                exec_summary.delta.raw_acc_mvout_tiles += 1;
            }
            if (collect_tile_profile) {
                tile_record.acc_readback_bytes = static_cast<int64_t>(hw_n * exec_tile.m * sizeof(float));
            }

            const int64_t dma_out_start_us = collect_stage_profile ? ggml_time_us() : 0;
            if (npu_debug_log_enabled()) {
                GGML_LOG_INFO("%s: tile m0=%" PRId64 " n0=%" PRId64 " k0=%" PRId64 " MVOUT_ACC_%s col=%" PRIu32 " row=%" PRIu32 " sram=0x%08x f32_scale=0x%08x\n",
                        __func__,
                        exec_tile.m0, exec_tile.n0, exec_tile.k0,
                        raw_acc_mvout ? "I32" : "F32",
                        static_cast<uint32_t>(exec_tile.m - 1),
                        static_cast<uint32_t>(exec_tile.n - 1),
                        plan.config.layout.output_accumulator.offset,
                        raw_acc_mvout ? 0
                                      : (use_scale_cache
                                            ? plan.config.layout.scale_cache.offset
                                            : 0x00070000));
            }
            if (use_scale_cache &&
                (loaded_scale_cache_m0 != exec_tile.m0 || loaded_scale_cache_m != exec_tile.m)) {
                uint32_t * scale_cache_words = reinterpret_cast<uint32_t *>(scale_cache_buf);
                for (int64_t local_m = 0; local_m < exec_tile.m; ++local_m) {
                    const int64_t global_m = exec_tile.m0 + local_m;
                    const float w_scale =
                        static_cast<size_t>(global_m) < plan.aicas_w8a8.weight_scale.size()
                        ? plan.aicas_w8a8.weight_scale[static_cast<size_t>(global_m)]
                        : per_tensor_weight_scale;
                    scale_cache_words[static_cast<size_t>(local_m)] =
                        npu_float_to_q8_24_u32(activation_scale * w_scale);
                }
                npu_dma_mvin(
                    scale_cache_buf,
                    plan.config.layout.scale_cache.offset,
                    static_cast<uint32_t>(exec_tile.m),
                    0,
                    static_cast<uint16_t>(exec_tile.m),
                    static_cast<uint32_t>(exec_tile.m),
                    1,
                    2,
                    true,
                    false,
                    false,
                    0,
                    0,
                    0);
                loaded_scale_cache_m0 = exec_tile.m0;
                loaded_scale_cache_m = exec_tile.m;
            } else if (!raw_acc_mvout) {
                uint32_t * scale_words = reinterpret_cast<uint32_t *>(bias_buf);
                const int64_t scale_count = mvout_per_channel ? exec_tile.m : 1;
                for (int64_t idx = 0; idx < scale_count; ++idx) {
                    const int64_t m = mvout_per_channel ? idx : 0;
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
                    static_cast<uint32_t>(scale_count),
                    0,
                    static_cast<uint16_t>(scale_count),
                    static_cast<uint32_t>(scale_count),
                    1,
                    2,
                    true,
                    false,
                    false,
                    0,
                    0,
                    0);
            }
            void * mvout_buf = acc_buf;
            uint32_t mvout_dram_stride = static_cast<uint32_t>(exec_tile.m);
            if (full_output_cma_active) {
                const npu_full_output_span * span =
                    npu_find_full_output_span(full_output_spans, exec_tile.n0);
                if (span != nullptr) {
                    mvout_buf = static_cast<char *>(full_output_buf) +
                        static_cast<size_t>(span->padded_row0 * plan.m + exec_tile.m0) * sizeof(float);
                    mvout_dram_stride = static_cast<uint32_t>(plan.m);
                }
            }
            npu_dma_mvout_ex(
                mvout_buf,
                plan.config.layout.output_accumulator.offset,
                static_cast<uint32_t>(exec_tile.m),
                static_cast<uint32_t>(hw_n),
                static_cast<uint16_t>(exec_tile.out_stride),
                mvout_dram_stride,
                1,
                1,
                true,
                !raw_acc_mvout,
                0,
                raw_acc_mvout ? 0
                              : (use_scale_cache
                                    ? plan.config.layout.scale_cache.offset
                                    : 0x00070000),
                !raw_acc_mvout && mvout_per_channel);
            if (collect_stage_profile) {
                const int64_t dma_out_us = ggml_time_us() - dma_out_start_us;
                exec_summary.delta.dma_out_calls += 1;
                exec_summary.delta.dma_out_us_total += dma_out_us;
                exec_summary.delta.acc_readback_bytes_total += static_cast<int64_t>(hw_n * exec_tile.m * sizeof(float));
                if (collect_tile_profile) {
                    tile_record.dma_out_us = static_cast<double>(dma_out_us);
                }
            }

            if (full_output_cma_active) {
                if (collect_tile_profile) {
                    tile_record.output_write_bytes = 0;
                }
            } else if (fold_output_reconstruction) {
                const int64_t postprocess_start_us = collect_stage_profile ? ggml_time_us() : 0;
                const size_t tile_elems = static_cast<size_t>(exec_tile.n * exec_tile.m);
                const bool output_is_final_f32 =
                    !raw_acc_mvout &&
                    !apply_output_compensation &&
                    (tile_uses_bias || output_has_bias_in_accumulator || plan.bias == nullptr);
                const bool fast_output_copy = output_is_final_f32 && npu_dst_is_dense_f32(plan.dst);
                if (fast_output_copy) {
                    const float * acc_f32 = reinterpret_cast<const float *>(acc_buf);
                    if (npu_dst_f32_tile_is_contiguous(plan.dst, exec_tile.m0, exec_tile.m, exec_tile.n)) {
                        float * dst_tile = reinterpret_cast<float *>(
                            static_cast<char *>(plan.dst->data) +
                            exec_tile.n0 * plan.dst->nb[1]);
                        npu_copy_from_cma(
                            dst_tile,
                            acc_f32,
                            static_cast<size_t>(exec_tile.n * exec_tile.m * sizeof(float)));
                    } else {
                        for (int64_t n = 0; n < exec_tile.n; ++n) {
                            float * dst_row = reinterpret_cast<float *>(
                                static_cast<char *>(plan.dst->data) +
                                (exec_tile.n0 + n) * plan.dst->nb[1] +
                                exec_tile.m0 * sizeof(float));
                            const float * acc_row = acc_f32 + static_cast<size_t>(n * exec_tile.m);
                            npu_copy_from_cma(dst_row, acc_row, static_cast<size_t>(exec_tile.m * sizeof(float)));
                        }
                    }
                } else if (raw_acc_mvout) {
                    acc_raw_values_host.resize(tile_elems);
                    npu_copy_from_cma(acc_raw_values_host.data(), acc_buf, tile_elems * sizeof(int32_t));
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
                    npu_copy_from_cma(acc_scaled_values.data(), acc_buf, tile_elems * sizeof(float));
                }
                if (!fast_output_copy) {
                    npu_prepare_bias_tile_f32_exec(
                        (tile_uses_bias || output_has_bias_in_accumulator) ? nullptr : plan.bias,
                        exec_tile.m0,
                        exec_tile.n0,
                        exec_tile.m,
                        exec_tile.n,
                        bias_tile_values);
                }

                if (!fast_output_copy && npu_dst_is_dense_f32(plan.dst)) {
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
                            if (apply_output_compensation) {
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
                } else if (!fast_output_copy) {
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
                            if (apply_output_compensation) {
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
                    npu_copy_from_cma(acc_raw_values_host.data(), acc_buf, tile_elems * sizeof(int32_t));
                } else {
                    acc_scaled_values.resize(tile_elems);
                    npu_copy_from_cma(acc_scaled_values.data(), acc_buf, tile_elems * sizeof(float));
                }
                npu_prepare_bias_tile_f32_exec(
                    (tile_uses_bias || output_has_bias_in_accumulator) ? nullptr : plan.bias,
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
                            if (apply_output_compensation &&
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
                            if (apply_output_compensation &&
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
                int32_t ref_acc_at_max = 0;
                int32_t ref_acc_alt_at_max = 0;
                int32_t npu_acc_raw_at_max = 0;
                float dequant_scale_at_max = 0.0f;
                bool npu_acc_raw_valid_at_max = false;
                int64_t compared = 0;
                const auto acc_ref_it = tile_align_acc_ref.find(tile_align_key);
                const auto acc_ref_alt_it = tile_align_acc_ref_alt_layout.find(tile_align_key);
                const auto bias_in_acc_it = tile_align_bias_in_accumulator.find(tile_align_key);
                const bool model_bias_in_acc =
                    bias_in_acc_it != tile_align_bias_in_accumulator.end() && bias_in_acc_it->second;

                if (acc_ref_it != tile_align_acc_ref.end() &&
                    acc_ref_alt_it != tile_align_acc_ref_alt_layout.end()) {
                    for (int64_t n = 0; n < exec_tile.n; ++n) {
                        for (int64_t m = 0; m < exec_tile.m; ++m) {
                            const size_t idx = static_cast<size_t>(n * exec_tile.m + m);
                            int32_t acc_ref = acc_ref_it->second[idx];
                            int32_t acc_ref_alt_layout = acc_ref_alt_it->second[idx];
                            const int64_t global_m = exec_tile.m0 + m;
                            const float wgt_scale = fold_output_reconstruction
                                ? (mvout_per_channel &&
                                        global_m >= 0 &&
                                        static_cast<size_t>(global_m) < plan.aicas_w8a8.weight_scale.size()
                                    ? plan.aicas_w8a8.weight_scale[static_cast<size_t>(global_m)]
                                    : per_tensor_weight_scale)
                                : packed_weight.scales[static_cast<size_t>(m)];
                            const float dequant_scale = activation_scale * wgt_scale;
                            float out_ref = static_cast<float>(acc_ref) * dequant_scale;
                            float out_ref_alt_layout =
                                static_cast<float>(acc_ref_alt_layout) * dequant_scale;
                            if (apply_output_compensation &&
                                global_m >= 0 &&
                                static_cast<size_t>(global_m) < plan.weight_column_sum_q.size()) {
                                out_ref -= static_cast<float>(
                                    plan.activation_quant.zero_point *
                                    plan.weight_column_sum_q[static_cast<size_t>(global_m)]) *
                                    dequant_scale;
                                out_ref_alt_layout -= static_cast<float>(
                                    plan.activation_quant.zero_point *
                                    plan.weight_column_sum_q[static_cast<size_t>(global_m)]) *
                                    dequant_scale;
                            }
                            if (use_aicas_w8a8 && plan.bias != nullptr && !model_bias_in_acc) {
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
                            ref_acc_at_max = acc_ref;
                            ref_acc_alt_at_max = acc_ref_alt_layout;
                            dequant_scale_at_max = dequant_scale;
                            npu_acc_raw_valid_at_max =
                                raw_acc_mvout && idx < acc_raw_values_host.size();
                            npu_acc_raw_at_max = npu_acc_raw_valid_at_max
                                ? acc_raw_values_host[idx]
                                : 0;
                        }
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
                        " raw_acc_valid=%d raw_acc_npu=%" PRId32 " raw_acc_ref=%" PRId32 " raw_acc_ref_alt=%" PRId32 " dequant_scale=%.10g"
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
                        npu_acc_raw_valid_at_max ? 1 : 0,
                        npu_acc_raw_at_max,
                        ref_acc_at_max,
                        ref_acc_alt_at_max,
                        dequant_scale_at_max,
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
            if (tile_align_collect_current) {
                tile_align_acc_ref.erase(tile_align_key);
                tile_align_acc_ref_alt_layout.erase(tile_align_key);
                tile_align_bias_in_accumulator.erase(tile_align_key);
            }
        }

        if (collect_tile_profile) {
            tile_record.total_us = static_cast<double>(ggml_time_us() - tile_start_us);
            tile_record.accounted_us =
                tile_record.activation_pack_us +
                tile_record.host_copy_activation_us +
                tile_record.host_copy_weight_us +
                tile_record.bias_prepare_us +
                tile_record.dma_in_pair_us +
                tile_record.dma_in_bias_us +
                tile_record.w_prefetch_wait_us +
                tile_record.gemm_us +
                tile_record.dma_out_us +
                tile_record.postprocess_us;
            tile_record.unaccounted_us =
                std::max(0.0, tile_record.total_us - tile_record.accounted_us);
            profile_record.tiles.push_back(std::move(tile_record));
        }
    }

    if (full_output_cma_active) {
        const int64_t postprocess_start_us = collect_stage_profile ? ggml_time_us() : 0;
        int64_t copied_bytes = 0;
        for (const npu_full_output_span & span : full_output_spans) {
            if (span.n <= 0) {
                continue;
            }
            const char * src_base = static_cast<const char *>(full_output_buf) +
                static_cast<size_t>(span.padded_row0 * plan.m) * sizeof(float);
            char * dst_base = static_cast<char *>(plan.dst->data) +
                span.n0 * plan.dst->nb[1];
            const size_t row_bytes = static_cast<size_t>(plan.m) * sizeof(float);
            if (static_cast<size_t>(plan.dst->nb[1]) == row_bytes) {
                const size_t bytes = static_cast<size_t>(span.n) * row_bytes;
                npu_copy_from_cma(dst_base, src_base, bytes);
                copied_bytes += static_cast<int64_t>(bytes);
            } else {
                for (int64_t n = 0; n < span.n; ++n) {
                    npu_copy_from_cma(
                        dst_base + n * plan.dst->nb[1],
                        src_base + static_cast<size_t>(n) * row_bytes,
                        row_bytes);
                    copied_bytes += static_cast<int64_t>(row_bytes);
                }
            }
        }
        if (collect_stage_profile) {
            const int64_t postprocess_us = ggml_time_us() - postprocess_start_us;
            exec_summary.delta.postprocess_calls += static_cast<int64_t>(full_output_spans.size());
            exec_summary.delta.postprocess_us_total += postprocess_us;
            exec_summary.delta.output_write_bytes_total += copied_bytes;
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
