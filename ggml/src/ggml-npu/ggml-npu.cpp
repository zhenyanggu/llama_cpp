#include "ggml-npu.h"

#include "ggml-npu-common.h"
#include "ggml-npu-exec.h"
#include "ggml-npu-plan.h"
#include "ggml-npu-profile.h"

#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "npu_runtime.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cinttypes>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <mutex>
#include <new>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

extern "C" int npu_decode_init();
extern "C" void npu_decode_destroy();
extern "C" void npu_decode_reset();
extern "C" void * npu_decode_mem_alloc(size_t size);
extern "C" void npu_decode_mem_free(void * ptr);
extern "C" void npu_decode_dma_mvin(
        void * host_ptr,
        uint32_t sram_addr,
        uint32_t col_num,
        uint32_t row_num,
        uint16_t sram_stride,
        uint32_t dram_stride,
        uint8_t precision,
        uint8_t input_type,
        bool dest,
        bool is_bias,
        bool is_quant,
        uint32_t quant_zero,
        uint16_t quant_scale,
        uint16_t quant_shift);
extern "C" void npu_decode_dma_mvout(
        void * host_ptr,
        uint32_t sram_addr,
        uint32_t col_num,
        uint32_t row_num,
        uint16_t sram_stride,
        uint32_t dram_stride,
        uint8_t precision,
        uint8_t output_type,
        bool source,
        bool is_quant,
        uint32_t quant_zero,
        uint32_t scale_or_addr);
extern "C" void npu_decode_gemv_block_run(uint16_t m, uint16_t n, uint32_t output_base_bytes);
extern "C" void npu_decode_gemv_pingpong_run(
        void * act_ptr,
        void * scale_ptr,
        void * weight_ptr,
        void * output_ptr,
        uint16_t m,
        uint16_t n);
extern "C" void * npu_decode_memory_base();
extern "C" uint32_t npu_decode_memory_size();

namespace ggml_npu {

static bool npu_debug_log_enabled() {
    return std::getenv("GGML_NPU_DEBUG_LOG") != nullptr || std::getenv("AICAS_MMPROJ_W8A8_DEBUG") != nullptr;
}

static bool npu_runtime_profile_requested() {
    const char * path = std::getenv("NPU_PROFILE_OUT");
    return path != nullptr && path[0] != '\0';
}

static bool npu_eager_init_enabled() {
    const char * v = std::getenv("GGML_NPU_EAGER_INIT");
    return v != nullptr && v[0] != '\0' && std::strcmp(v, "0") != 0;
}

static bool npu_raw_i8_gemm_debug_log_enabled() {
    return std::getenv("GGML_NPU_DEBUG_LOG") != nullptr || std::getenv("AICAS_MMPROJ_BFP8M_NPU_DEBUG") != nullptr;
}

static bool npu_fits_u16(int64_t value) {
    return value >= 0 && value <= std::numeric_limits<uint16_t>::max();
}

static bool npu_raw_i8_gemm_validate(
        const char * op_name,
        const int8_t * weight_kxm,
        int64_t m,
        int64_t k,
        int64_t weight_stride_m,
        const int8_t * act_nxk,
        int64_t n,
        int64_t act_stride_k,
        const int32_t * out_nxm,
        int64_t out_stride_m,
        std::string * error) {
    auto fail = [op_name, error](const char * reason) {
        if (error != nullptr) {
            *error = reason;
        }
        if (npu_raw_i8_gemm_debug_log_enabled()) {
            GGML_LOG_WARN("%s: reject %s: %s\n", __func__, op_name != nullptr ? op_name : "(unnamed)", reason);
        }
        return false;
    };

    if (weight_kxm == nullptr || act_nxk == nullptr || out_nxm == nullptr) {
        return fail("null pointer");
    }
    if (m <= 0 || n <= 0 || k <= 0) {
        return fail("non-positive dimension");
    }
    if (weight_stride_m < m || act_stride_k < k || out_stride_m < m) {
        return fail("stride smaller than dimension");
    }
    if (!npu_fits_u16(m) || !npu_fits_u16(n) || !npu_fits_u16(k) ||
            !npu_fits_u16(weight_stride_m) || !npu_fits_u16(act_stride_k) || !npu_fits_u16(out_stride_m)) {
        return fail("dimension or stride exceeds uint16 range");
    }
    if (!npu_is_aligned_i64(weight_stride_m, NPU_GEMM_PLAN_STRIDE_ALIGNMENT) ||
            !npu_is_aligned_i64(act_stride_k, NPU_GEMM_PLAN_STRIDE_ALIGNMENT) ||
            !npu_is_aligned_i64(out_stride_m, NPU_GEMM_PLAN_STRIDE_ALIGNMENT)) {
        return fail("stride is not NPU GEMM-plan aligned");
    }
    const uint64_t act_bytes = static_cast<uint64_t>(n) * static_cast<uint64_t>(act_stride_k);
    const uint64_t weight_bytes = static_cast<uint64_t>(k) * static_cast<uint64_t>(weight_stride_m);
    const uint64_t out_bytes = static_cast<uint64_t>(n) * static_cast<uint64_t>(out_stride_m) * sizeof(int32_t);
    if (act_bytes > std::numeric_limits<size_t>::max() ||
            weight_bytes > std::numeric_limits<size_t>::max() ||
            out_bytes > std::numeric_limits<size_t>::max()) {
        return fail("host buffer size overflow");
    }

    return true;
}

static bool npu_decode_w4a16_debug_log_enabled() {
    return std::getenv("GGML_NPU_DEBUG_LOG") != nullptr || std::getenv("AICAS_TEXT_DECODE_AWQ_NPU_DEBUG") != nullptr;
}

static bool npu_decode_require_active_overlay() {
    const char * value = std::getenv("AICAS_TEXT_DECODE_AWQ_NPU_REQUIRE_ACTIVE");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

static std::mutex & npu_decode_overlay_mutex() {
    static std::mutex mutex;
    return mutex;
}

static bool & npu_decode_overlay_active() {
    static bool active = false;
    return active;
}

static bool npu_decode_ensure_overlay_active(const char * op_name) {
    const char * cmd = std::getenv("AICAS_NPU_DECODE_SWITCH_CMD");
    if (cmd == nullptr || cmd[0] == '\0') {
        return true;
    }

    std::lock_guard<std::mutex> lock(npu_decode_overlay_mutex());
    if (npu_decode_overlay_active()) {
        return true;
    }

    if (npu_decode_require_active_overlay()) {
        return false;
    }

    npu_clear_preloaded_weight_cache();
    npu_destroy();
    npu_decode_destroy();
    const int rc = std::system(cmd);
    if (rc != 0) {
        if (npu_decode_w4a16_debug_log_enabled()) {
            GGML_LOG_WARN("%s: decode overlay switch failed for %s, rc=%d\n",
                    __func__, op_name != nullptr ? op_name : "(unnamed)", rc);
        }
        return false;
    }
    npu_decode_overlay_active() = true;
    return true;
}

constexpr uint32_t NPU_DECODE_GEMV_K_TILE = 128;
constexpr uint32_t NPU_DECODE_GEMV_M_TILE = 32;
constexpr uint32_t NPU_DECODE_GEMV_LINE_BYTES = 64;
constexpr uint32_t NPU_DECODE_GEMV_DMA_ALIGN = 256;
constexpr uint32_t NPU_DECODE_GEMV_ACT_BUFFER_BYTES = 2560 * 2;
constexpr uint32_t NPU_DECODE_GEMV_ACT_TILE_BYTES = NPU_DECODE_GEMV_K_TILE * 2;
constexpr uint32_t NPU_DECODE_GEMV_SCALE_TILE_BYTES = NPU_DECODE_GEMV_M_TILE * 2;
constexpr uint32_t NPU_DECODE_GEMV_WEIGHT_ROW_BYTES = (NPU_DECODE_GEMV_K_TILE * 4) / 8;
constexpr uint32_t NPU_DECODE_GEMV_WEIGHT_TILE_BYTES = NPU_DECODE_GEMV_M_TILE * NPU_DECODE_GEMV_WEIGHT_ROW_BYTES;
constexpr uint8_t NPU_DECODE_GEMV_INPUT_TYPE_DATA = 0;
constexpr uint8_t NPU_DECODE_GEMV_INPUT_TYPE_WEIGHT = 1;
constexpr uint8_t NPU_DECODE_GEMV_INPUT_TYPE_ACT = 3;
constexpr uint32_t NPU_DECODE_GEMV_PING_WEIGHT_BASE = 0x10000;
constexpr uint32_t NPU_DECODE_GEMV_PONG_WEIGHT_BASE = 0x40000;

struct npu_decode_gemv_layout {
    uint32_t row_tiles = 0;
    uint32_t col_tiles = 0;
    uint32_t act_bytes = 0;
    uint32_t scale_bytes = 0;
    uint32_t weight_base = 0;
    uint32_t weight_bytes = 0;
    uint32_t packed_bytes = 0;
    uint32_t output_bytes = 0;
};

struct npu_decode_preloaded_block {
    int64_t row_base = 0;
    uint32_t block_rows = 0;
    npu_decode_gemv_layout layout = {};
    uint32_t cma_offset = 0;
    uint32_t cma_bytes = 0;
    std::vector<uint8_t> packed;
};

struct npu_decode_preloaded_tensor {
    int64_t packed_k = 0;
    int64_t out_channels = 0;
    int64_t q4_nb1 = 0;
    int64_t k = 0;
    uint32_t max_block_rows = 0;
    npu_decode_gemv_layout max_layout = {};
    npu_decode_gemv_layout pingpong_layout = {};
    bool pingpong_available = false;
    uint32_t pingpong_scale_cma_offset = 0;
    uint32_t pingpong_scale_cma_bytes = 0;
    uint32_t pingpong_weight_cma_offset = 0;
    uint32_t pingpong_weight_cma_bytes = 0;
    bool zero_validated = false;
    std::vector<float> inv_smooth_scale;
    std::vector<uint8_t> pingpong_scale;
    std::vector<uint8_t> pingpong_weight;
    std::vector<npu_decode_preloaded_block> blocks;
};

struct npu_decode_preload_cache {
    std::mutex mutex;
    std::unordered_map<std::string, npu_decode_preloaded_tensor> entries;
    uint32_t cma_base_offset = 0;
    uint32_t cma_limit_bytes = 0;
    uint32_t cma_cursor = 0;
    uint32_t transient_offset = 0;
    uint32_t transient_size = 0;
};

struct npu_decode_gemv_profile_record {
    std::string op_name;
    int dst_type = GGML_TYPE_F32;
    int64_t k = 0;
    int64_t n_cols = 0;
    int64_t out_channels = 0;
    uint32_t max_block_rows = 0;
    uint32_t row_blocks_per_col = 0;
    bool preloaded_host = false;
    bool preloaded_cma = false;
    bool pingpong = false;
    uint64_t activation_bytes = 0;
    uint64_t weight_bytes = 0;
    uint64_t output_bytes = 0;
    uint64_t total_us = 0;
    uint64_t setup_us = 0;
    uint64_t validation_us = 0;
    uint64_t layout_us = 0;
    uint64_t preload_lookup_us = 0;
    uint64_t overlay_ensure_us = 0;
    uint64_t heap_update_us = 0;
    uint64_t runtime_init_us = 0;
    uint64_t memory_base_us = 0;
    uint64_t alloc_us = 0;
    uint64_t activation_preprocess_us = 0;
    uint64_t host_copy_activation_us = 0;
    uint64_t reset_us = 0;
    uint64_t mvin_activation_us = 0;
    uint64_t weight_preprocess_us = 0;
    uint64_t host_copy_weight_us = 0;
    uint64_t output_clear_us = 0;
    uint64_t mvin_weight_us = 0;
    uint64_t gemv_us = 0;
    uint64_t pingpong_us = 0;
    uint64_t mvout_us = 0;
    uint64_t postprocess_us = 0;
    uint64_t cleanup_us = 0;
    uint64_t activation_mvin_calls = 0;
    uint64_t weight_mvin_calls = 0;
    uint64_t gemv_calls = 0;
    uint64_t pingpong_calls = 0;
    uint64_t mvout_calls = 0;
};

struct npu_decode_gemv_profile_cache {
    std::mutex mutex;
    std::vector<npu_decode_gemv_profile_record> records;
    bool atexit_registered = false;
};

static npu_decode_preload_cache & npu_decode_get_preload_cache() {
    static npu_decode_preload_cache cache;
    return cache;
}

static npu_decode_gemv_profile_cache & npu_decode_profile_cache() {
    static auto * cache = new npu_decode_gemv_profile_cache();
    return *cache;
}

static const char * npu_decode_profile_path() {
    const char * path = std::getenv("GGML_NPU_DECODE_PROFILE_JSONL");
    return path != nullptr && path[0] != '\0' ? path : nullptr;
}

static bool npu_decode_profile_enabled() {
    return npu_decode_profile_path() != nullptr;
}

static bool npu_decode_pingpong_enabled() {
    const char * value = std::getenv("AICAS_TEXT_DECODE_AWQ_PINGPONG");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

static nlohmann::ordered_json npu_decode_profile_to_json(const npu_decode_gemv_profile_record & record) {
    nlohmann::ordered_json payload;
    payload["profile_kind"] = "aicas_decode_w4a16_gemv";
    payload["op_name"] = record.op_name;
    payload["dst_type"] = ggml_type_name(static_cast<ggml_type>(record.dst_type));
    payload["f16_output"] = record.dst_type == GGML_TYPE_F16;
    payload["dims"] = {
        {"m", record.out_channels},
        {"k", record.k},
        {"n_cols", record.n_cols},
        {"max_block_rows", record.max_block_rows},
        {"row_blocks_per_col", record.row_blocks_per_col},
    };
    payload["preload"] = {
        {"host_cache", record.preloaded_host},
        {"cma_resident", record.preloaded_cma},
        {"pingpong", record.pingpong},
    };
    payload["bytes"] = {
        {"activation", record.activation_bytes},
        {"weight_mvin", record.weight_bytes},
        {"output", record.output_bytes},
    };
    payload["calls"] = {
        {"activation_mvin", record.activation_mvin_calls},
        {"weight_mvin", record.weight_mvin_calls},
        {"gemv", record.gemv_calls},
        {"pingpong", record.pingpong_calls},
        {"mvout", record.mvout_calls},
    };
    payload["time_us"] = {
        {"total", record.total_us},
        {"setup", record.setup_us},
        {"validation", record.validation_us},
        {"layout", record.layout_us},
        {"preload_lookup", record.preload_lookup_us},
        {"overlay_ensure", record.overlay_ensure_us},
        {"heap_update", record.heap_update_us},
        {"runtime_init", record.runtime_init_us},
        {"memory_base", record.memory_base_us},
        {"alloc", record.alloc_us},
        {"activation_preprocess", record.activation_preprocess_us},
        {"host_copy_activation", record.host_copy_activation_us},
        {"reset", record.reset_us},
        {"mvin_activation", record.mvin_activation_us},
        {"weight_preprocess", record.weight_preprocess_us},
        {"host_copy_weight", record.host_copy_weight_us},
        {"output_clear", record.output_clear_us},
        {"mvin_weight", record.mvin_weight_us},
        {"gemv", record.gemv_us},
        {"pingpong", record.pingpong_us},
        {"mvout", record.mvout_us},
        {"postprocess", record.postprocess_us},
        {"cleanup", record.cleanup_us},
    };

    return payload;
}

static void npu_decode_profile_flush() {
    const char * path = npu_decode_profile_path();
    npu_decode_gemv_profile_cache & cache = npu_decode_profile_cache();
    std::vector<npu_decode_gemv_profile_record> snapshot;
    {
        std::lock_guard<std::mutex> lock(cache.mutex);
        snapshot.swap(cache.records);
    }

    if (path == nullptr || snapshot.empty()) {
        return;
    }

    std::ofstream out(path, std::ios::app);
    if (!out.good()) {
        return;
    }
    for (const npu_decode_gemv_profile_record & record : snapshot) {
        out << npu_decode_profile_to_json(record).dump() << '\n';
    }
}

static void npu_decode_profile_write(const npu_decode_gemv_profile_record & record) {
    npu_decode_gemv_profile_cache & cache = npu_decode_profile_cache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    if (!cache.atexit_registered) {
        std::atexit(npu_decode_profile_flush);
        cache.atexit_registered = true;
    }
    cache.records.push_back(record);
}

static uint32_t npu_decode_parse_size_env(const char * name, uint32_t fallback) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }

    char * end = nullptr;
    unsigned long long parsed = std::strtoull(value, &end, 0);
    if (end == value || parsed == 0) {
        return fallback;
    }
    while (*end && std::isspace(static_cast<unsigned char>(*end))) {
        ++end;
    }

    unsigned long long multiplier = 1;
    if (*end) {
        const char suffix = static_cast<char>(std::tolower(static_cast<unsigned char>(*end)));
        if (suffix == 'k') {
            multiplier = 1024ull;
            ++end;
        } else if (suffix == 'm') {
            multiplier = 1024ull * 1024ull;
            ++end;
        } else if (suffix == 'g') {
            multiplier = 1024ull * 1024ull * 1024ull;
            ++end;
        } else {
            return fallback;
        }
        if (*end == 'i' || *end == 'I') {
            ++end;
        }
        if (*end == 'b' || *end == 'B') {
            ++end;
        }
        while (*end && std::isspace(static_cast<unsigned char>(*end))) {
            ++end;
        }
        if (*end) {
            return fallback;
        }
    }

    if (parsed > std::numeric_limits<uint32_t>::max() / multiplier) {
        return fallback;
    }
    return static_cast<uint32_t>(parsed * multiplier);
}

static uint32_t npu_decode_align_up_bytes(uint32_t value, uint32_t alignment) {
    return (value + alignment - 1u) & ~(alignment - 1u);
}

static void npu_decode_update_runtime_heap_env() {
    npu_decode_preload_cache & cache = npu_decode_get_preload_cache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    if (cache.cma_limit_bytes == 0) {
        return;
    }

    const uint32_t reserved = npu_decode_align_up_bytes(cache.cma_cursor, NPU_DECODE_GEMV_DMA_ALIGN);
    if (reserved >= cache.cma_limit_bytes) {
        return;
    }

    const uint32_t transient_offset = cache.cma_base_offset + reserved;
    const uint32_t transient_size = cache.cma_limit_bytes - reserved;
    if (cache.transient_offset == transient_offset && cache.transient_size == transient_size) {
        return;
    }
    const std::string offset = std::to_string(transient_offset);
    const std::string size = std::to_string(transient_size);
    setenv("NPU_CMA_HEAP_OFFSET", offset.c_str(), 1);
    setenv("NPU_CMA_HEAP_SIZE", size.c_str(), 1);
    cache.transient_offset = transient_offset;
    cache.transient_size = transient_size;
}

static uint32_t npu_decode_ceil_div_u32(uint32_t value, uint32_t divisor) {
    return (value + divisor - 1u) / divisor;
}

static uint32_t npu_decode_align_up_u32(uint32_t value, uint32_t alignment) {
    return (value + alignment - 1u) & ~(alignment - 1u);
}

static bool npu_decode_make_gemv_layout(uint32_t m, uint32_t k, npu_decode_gemv_layout * layout) {
    if (m == 0 || k == 0 || layout == nullptr) {
        return false;
    }

    layout->row_tiles = npu_decode_ceil_div_u32(m, NPU_DECODE_GEMV_M_TILE);
    layout->col_tiles = npu_decode_ceil_div_u32(k, NPU_DECODE_GEMV_K_TILE);
    layout->act_bytes = layout->col_tiles * NPU_DECODE_GEMV_ACT_TILE_BYTES;
    layout->scale_bytes = layout->row_tiles * layout->col_tiles * NPU_DECODE_GEMV_SCALE_TILE_BYTES;
    layout->weight_base = npu_decode_align_up_u32(layout->scale_bytes, NPU_DECODE_GEMV_LINE_BYTES);
    layout->weight_bytes = layout->row_tiles * layout->col_tiles * NPU_DECODE_GEMV_WEIGHT_TILE_BYTES;
    layout->packed_bytes = npu_decode_align_up_u32(layout->weight_base + layout->weight_bytes, NPU_DECODE_GEMV_DMA_ALIGN);
    layout->output_bytes = npu_decode_align_up_u32(m * 2, NPU_DECODE_GEMV_LINE_BYTES);

    return layout->act_bytes <= NPU_DECODE_GEMV_ACT_BUFFER_BYTES &&
           layout->scale_bytes <= NPU_DEFAULT_SPM_BYTES &&
           layout->weight_base + layout->weight_bytes <= NPU_DEFAULT_SPM_BYTES &&
           layout->output_bytes <= NPU_DEFAULT_SPM_BYTES &&
           (layout->act_bytes % NPU_DECODE_GEMV_DMA_ALIGN) == 0 &&
           (layout->packed_bytes % NPU_DECODE_GEMV_DMA_ALIGN) == 0;
}

static bool npu_decode_make_pingpong_layout(uint32_t m, uint32_t k, npu_decode_gemv_layout * layout) {
    if (m == 0 || k == 0 || layout == nullptr) {
        return false;
    }

    layout->row_tiles = npu_decode_ceil_div_u32(m, NPU_DECODE_GEMV_M_TILE);
    layout->col_tiles = npu_decode_ceil_div_u32(k, NPU_DECODE_GEMV_K_TILE);
    layout->act_bytes = layout->col_tiles * NPU_DECODE_GEMV_ACT_TILE_BYTES;
    layout->scale_bytes = layout->row_tiles * layout->col_tiles * NPU_DECODE_GEMV_SCALE_TILE_BYTES;
    layout->weight_base = 0;
    layout->weight_bytes = layout->row_tiles * layout->col_tiles * NPU_DECODE_GEMV_WEIGHT_TILE_BYTES;
    layout->packed_bytes = 0;
    layout->output_bytes = m * sizeof(ggml_fp16_t);

    const uint32_t bytes_per_row_tile = layout->col_tiles * NPU_DECODE_GEMV_WEIGHT_TILE_BYTES;
    const uint32_t ping_capacity = NPU_DECODE_GEMV_PONG_WEIGHT_BASE - NPU_DECODE_GEMV_PING_WEIGHT_BASE;
    const uint32_t pong_capacity = NPU_DEFAULT_SPM_BYTES - NPU_DECODE_GEMV_PONG_WEIGHT_BASE;
    const uint32_t weight_capacity = std::min(ping_capacity, pong_capacity);
    return layout->act_bytes <= NPU_DECODE_GEMV_ACT_BUFFER_BYTES &&
           layout->scale_bytes <= NPU_DECODE_GEMV_PING_WEIGHT_BASE &&
           layout->output_bytes <= NPU_DEFAULT_SPM_BYTES &&
           layout->output_bytes <= std::numeric_limits<uint16_t>::max() + 1u &&
           bytes_per_row_tile > 0 &&
           bytes_per_row_tile <= weight_capacity &&
           (layout->act_bytes % NPU_DECODE_GEMV_DMA_ALIGN) == 0 &&
           (layout->scale_bytes % NPU_DECODE_GEMV_DMA_ALIGN) == 0;
}

static float npu_decode_read_typed_f32(const void * base, int type, int64_t byte_offset) {
    const char * ptr = static_cast<const char *>(base) + byte_offset;
    switch (type) {
        case GGML_TYPE_F32:
            return *reinterpret_cast<const float *>(ptr);
        case GGML_TYPE_F16:
            return ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t *>(ptr));
        default:
            return std::numeric_limits<float>::quiet_NaN();
    }
}

static bool npu_decode_validate_symmetric_zero(
        const void * zero_data,
        int zero_type,
        int64_t zero_nb0,
        int64_t zero_nb1,
        int64_t out_channels,
        int64_t groups,
        const char * op_name) {
    for (int64_t row = 0; row < out_channels; ++row) {
        for (int64_t group = 0; group < groups; ++group) {
            const float zero = npu_decode_read_typed_f32(zero_data, zero_type, row * zero_nb1 + group * zero_nb0);
            if (!(std::fabs(zero - 8.0f) <= 1.0e-3f)) {
                if (npu_decode_w4a16_debug_log_enabled()) {
                    GGML_LOG_WARN("%s: reject %s: non-symmetric AWQ zero row=%" PRId64 " group=%" PRId64 " zero=%g\n",
                            __func__, op_name != nullptr ? op_name : "(unnamed)", row, group, static_cast<double>(zero));
                }
                return false;
            }
        }
    }
    return true;
}

static uint8_t npu_decode_q4_unsigned_at(const uint8_t * row, int64_t k_idx) {
    const uint8_t packed = row[k_idx / 2];
    return (k_idx & 1) == 0 ? (packed & 0x0fu) : ((packed >> 4) & 0x0fu);
}

static void npu_decode_pack_activation(
        void * act_dst,
        size_t act_bytes,
        const npu_decode_gemv_layout & layout,
        const void * act_data,
        int act_type,
        int64_t act_nb0,
        int64_t act_nb1,
        const float * inv_smooth_scale,
        int64_t k,
        int64_t col) {
    std::memset(act_dst, 0, act_bytes);
    uint8_t * act = static_cast<uint8_t *>(act_dst);
    for (uint32_t col_tile = 0; col_tile < layout.col_tiles; ++col_tile) {
        for (uint32_t lane = 0; lane < NPU_DECODE_GEMV_K_TILE; ++lane) {
            const int64_t in_idx = static_cast<int64_t>(col_tile) * NPU_DECODE_GEMV_K_TILE + lane;
            if (in_idx >= k) {
                continue;
            }
            const float inv_smooth = inv_smooth_scale != nullptr ? inv_smooth_scale[in_idx] : 1.0f;
            const float value = npu_decode_read_typed_f32(act_data, act_type, col * act_nb1 + in_idx * act_nb0) * inv_smooth;
            const ggml_fp16_t fp16_value = ggml_fp32_to_fp16(value);
            std::memcpy(act + col_tile * NPU_DECODE_GEMV_ACT_TILE_BYTES + lane * 2, &fp16_value, sizeof(fp16_value));
        }
    }
}

static void npu_decode_convert_output_fp16_to_f32(
        const ggml_fp16_t * src,
        float * dst,
        uint32_t rows) {
    ggml_fp16_to_fp32_row(src, dst, rows);
}

static void npu_decode_pack_scale_weight_block(
        std::vector<uint8_t> & packed,
        const npu_decode_gemv_layout & layout,
        const void * q4_data,
        int64_t packed_k,
        int64_t q4_nb1,
        const void * scale_data,
        int scale_type,
        int64_t scale_nb0,
        int64_t scale_nb1,
        int64_t row_base,
        int64_t block_rows,
        int64_t k) {
    std::fill(packed.begin(), packed.end(), 0);

    for (uint32_t row_tile = 0; row_tile < layout.row_tiles; ++row_tile) {
        for (uint32_t col_tile = 0; col_tile < layout.col_tiles; ++col_tile) {
            const size_t scale_line = static_cast<size_t>(row_tile * layout.col_tiles + col_tile) * NPU_DECODE_GEMV_LINE_BYTES;
            for (uint32_t lane = 0; lane < NPU_DECODE_GEMV_M_TILE; ++lane) {
                const int64_t row = static_cast<int64_t>(row_tile) * NPU_DECODE_GEMV_M_TILE + lane;
                if (row >= block_rows) {
                    continue;
                }
                const float scale = npu_decode_read_typed_f32(
                        scale_data,
                        scale_type,
                        (row_base + row) * scale_nb1 + static_cast<int64_t>(col_tile) * scale_nb0);
                const ggml_fp16_t scale_fp16 = ggml_fp32_to_fp16(scale);
                std::memcpy(packed.data() + scale_line + lane * 2, &scale_fp16, sizeof(scale_fp16));
            }
        }
    }

    for (uint32_t row_tile = 0; row_tile < layout.row_tiles; ++row_tile) {
        for (uint32_t col_tile = 0; col_tile < layout.col_tiles; ++col_tile) {
            const uint32_t tile_index = row_tile * layout.col_tiles + col_tile;
            for (uint32_t row_in_tile = 0; row_in_tile < NPU_DECODE_GEMV_M_TILE; ++row_in_tile) {
                const int64_t row = static_cast<int64_t>(row_tile) * NPU_DECODE_GEMV_M_TILE + row_in_tile;
                if (row >= block_rows) {
                    continue;
                }
                const uint8_t * q4_row = static_cast<const uint8_t *>(q4_data) + (row_base + row) * q4_nb1;
                uint8_t * dst = packed.data() + layout.weight_base +
                    (static_cast<size_t>(tile_index) * NPU_DECODE_GEMV_M_TILE + row_in_tile) * NPU_DECODE_GEMV_WEIGHT_ROW_BYTES;
                for (uint32_t lane = 0; lane < NPU_DECODE_GEMV_K_TILE; ++lane) {
                    const int64_t in_idx = static_cast<int64_t>(col_tile) * NPU_DECODE_GEMV_K_TILE + lane;
                    if (in_idx >= k || in_idx / 2 >= packed_k) {
                        continue;
                    }
                    const int8_t signed_q4 = static_cast<int8_t>(npu_decode_q4_unsigned_at(q4_row, in_idx)) - 8;
                    const uint8_t nibble = static_cast<uint8_t>(signed_q4) & 0x0fu;
                    if ((lane & 1u) == 0) {
                        dst[lane / 2] = static_cast<uint8_t>((dst[lane / 2] & 0xf0u) | nibble);
                    } else {
                        dst[lane / 2] = static_cast<uint8_t>((dst[lane / 2] & 0x0fu) | (nibble << 4));
                    }
                }
            }
        }
    }
}

static void npu_decode_pack_pingpong_scale_weight(
        std::vector<uint8_t> & scale_packed,
        std::vector<uint8_t> & weight_packed,
        const npu_decode_gemv_layout & layout,
        const void * q4_data,
        int64_t packed_k,
        int64_t q4_nb1,
        const void * scale_data,
        int scale_type,
        int64_t scale_nb0,
        int64_t scale_nb1,
        int64_t out_channels,
        int64_t k) {
    std::fill(scale_packed.begin(), scale_packed.end(), 0);
    std::fill(weight_packed.begin(), weight_packed.end(), 0);

    for (uint32_t row_tile = 0; row_tile < layout.row_tiles; ++row_tile) {
        for (uint32_t col_tile = 0; col_tile < layout.col_tiles; ++col_tile) {
            const size_t scale_line =
                static_cast<size_t>(row_tile * layout.col_tiles + col_tile) * NPU_DECODE_GEMV_LINE_BYTES;
            for (uint32_t lane = 0; lane < NPU_DECODE_GEMV_M_TILE; ++lane) {
                const int64_t row = static_cast<int64_t>(row_tile) * NPU_DECODE_GEMV_M_TILE + lane;
                if (row >= out_channels) {
                    continue;
                }
                const float scale = npu_decode_read_typed_f32(
                        scale_data,
                        scale_type,
                        row * scale_nb1 + static_cast<int64_t>(col_tile) * scale_nb0);
                const ggml_fp16_t scale_fp16 = ggml_fp32_to_fp16(scale);
                std::memcpy(scale_packed.data() + scale_line + lane * 2, &scale_fp16, sizeof(scale_fp16));
            }
        }
    }

    for (uint32_t row_tile = 0; row_tile < layout.row_tiles; ++row_tile) {
        for (uint32_t col_tile = 0; col_tile < layout.col_tiles; ++col_tile) {
            const uint32_t tile_index = row_tile * layout.col_tiles + col_tile;
            for (uint32_t row_in_tile = 0; row_in_tile < NPU_DECODE_GEMV_M_TILE; ++row_in_tile) {
                const int64_t row = static_cast<int64_t>(row_tile) * NPU_DECODE_GEMV_M_TILE + row_in_tile;
                if (row >= out_channels) {
                    continue;
                }
                const uint8_t * q4_row = static_cast<const uint8_t *>(q4_data) + row * q4_nb1;
                uint8_t * dst = weight_packed.data() +
                    (static_cast<size_t>(tile_index) * NPU_DECODE_GEMV_M_TILE + row_in_tile) *
                    NPU_DECODE_GEMV_WEIGHT_ROW_BYTES;
                for (uint32_t lane = 0; lane < NPU_DECODE_GEMV_K_TILE; ++lane) {
                    const int64_t in_idx = static_cast<int64_t>(col_tile) * NPU_DECODE_GEMV_K_TILE + lane;
                    if (in_idx >= k || in_idx / 2 >= packed_k) {
                        continue;
                    }
                    const int8_t signed_q4 = static_cast<int8_t>(npu_decode_q4_unsigned_at(q4_row, in_idx)) - 8;
                    const uint8_t nibble = static_cast<uint8_t>(signed_q4) & 0x0fu;
                    if ((lane & 1u) == 0) {
                        dst[lane / 2] = static_cast<uint8_t>((dst[lane / 2] & 0xf0u) | nibble);
                    } else {
                        dst[lane / 2] = static_cast<uint8_t>((dst[lane / 2] & 0x0fu) | (nibble << 4));
                    }
                }
            }
        }
    }
}

static const npu_decode_preloaded_tensor * npu_decode_lookup_preloaded_tensor(
        const char * weight_name,
        int64_t packed_k,
        int64_t out_channels,
        int64_t q4_nb1,
        int64_t k) {
    if (weight_name == nullptr || weight_name[0] == '\0') {
        return nullptr;
    }

    npu_decode_preload_cache & cache = npu_decode_get_preload_cache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    auto it = cache.entries.find(weight_name);
    if (it == cache.entries.end()) {
        return nullptr;
    }
    const npu_decode_preloaded_tensor & entry = it->second;
    if (entry.packed_k != packed_k ||
            entry.out_channels != out_channels ||
            entry.q4_nb1 != q4_nb1 ||
            entry.k != k) {
        return nullptr;
    }
    return &it->second;
}

static int64_t & npu_profile_next_layer_id() {
    static int64_t next_layer_id = 0;
    return next_layer_id;
}

static bool & npu_profile_session_started() {
    static bool started = false;
    return started;
}

struct npu_backend_context {
    npu_tiling_config config;
};

struct npu_buffer_context {
    void * ptr = nullptr;
    bool own = false;
};

static void npu_buffer_free(ggml_backend_buffer_t buffer) {
    npu_buffer_context * ctx = static_cast<npu_buffer_context *>(buffer->context);
    if (ctx != nullptr && ctx->own) {
        ggml_aligned_free(ctx->ptr, buffer->size);
    }
    delete ctx;
}

static void * npu_buffer_get_base(ggml_backend_buffer_t buffer) {
    npu_buffer_context * ctx = static_cast<npu_buffer_context *>(buffer->context);
    return ctx != nullptr ? ctx->ptr : nullptr;
}

static void npu_buffer_memset_tensor(
        ggml_backend_buffer_t buffer,
        struct ggml_tensor * tensor,
        uint8_t value,
        size_t offset,
        size_t size) {
    GGML_UNUSED(buffer);
    std::memset(static_cast<char *>(tensor->data) + offset, value, size);
}

static void npu_buffer_set_tensor(
        ggml_backend_buffer_t buffer,
        struct ggml_tensor * tensor,
        const void * data,
        size_t offset,
        size_t size) {
    GGML_UNUSED(buffer);
    std::memcpy(static_cast<char *>(tensor->data) + offset, data, size);
}

static void npu_buffer_get_tensor(
        ggml_backend_buffer_t buffer,
        const struct ggml_tensor * tensor,
        void * data,
        size_t offset,
        size_t size) {
    GGML_UNUSED(buffer);
    std::memcpy(data, static_cast<const char *>(tensor->data) + offset, size);
}

static bool npu_buffer_cpy_tensor(
        ggml_backend_buffer_t buffer,
        const struct ggml_tensor * src,
        struct ggml_tensor * dst) {
    GGML_UNUSED(buffer);
    ggml_backend_tensor_copy(const_cast<struct ggml_tensor *>(src), dst);
    return true;
}

static void npu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    npu_buffer_context * ctx = static_cast<npu_buffer_context *>(buffer->context);
    if (ctx != nullptr && ctx->ptr != nullptr) {
        std::memset(ctx->ptr, value, buffer->size);
    }
}

static const ggml_backend_buffer_i npu_buffer_iface = {
    /* .free_buffer   = */ npu_buffer_free,
    /* .get_base      = */ npu_buffer_get_base,
    /* .init_tensor   = */ nullptr,
    /* .memset_tensor = */ npu_buffer_memset_tensor,
    /* .set_tensor    = */ npu_buffer_set_tensor,
    /* .get_tensor    = */ npu_buffer_get_tensor,
    /* .cpy_tensor    = */ npu_buffer_cpy_tensor,
    /* .clear         = */ npu_buffer_clear,
    /* .reset         = */ nullptr,
};

static ggml_backend_buffer_t npu_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    npu_buffer_context * ctx = new (std::nothrow) npu_buffer_context;
    if (ctx == nullptr) {
        return nullptr;
    }

    ctx->ptr = ggml_aligned_malloc(size);
    ctx->own = true;

    if (ctx->ptr == nullptr) {
        delete ctx;
        return nullptr;
    }

    return ggml_backend_buffer_init(buft, npu_buffer_iface, ctx, size);
}

static size_t npu_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    GGML_UNUSED(buft);
    return ggml_nbytes(tensor);
}

static const char * npu_buffer_type_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return "NPU_Host";
}

static size_t npu_buffer_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return 64;
}

static size_t npu_buffer_max_size(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return SIZE_MAX;
}

static bool npu_buffer_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return true;
}

static const ggml_backend_buffer_type_i npu_buffer_type_iface = {
    /* .get_name       = */ npu_buffer_type_name,
    /* .alloc_buffer   = */ npu_alloc_buffer,
    /* .get_alignment  = */ npu_buffer_alignment,
    /* .get_max_size   = */ npu_buffer_max_size,
    /* .get_alloc_size = */ npu_get_alloc_size,
    /* .is_host        = */ npu_buffer_is_host,
};

static ggml_guid_t ggml_backend_npu_guid(void) {
    static ggml_guid guid = {
        0x79, 0x13, 0xa2, 0x5d, 0x4a, 0x5f, 0x4e, 0x38,
        0x9a, 0x9f, 0x42, 0x61, 0x43, 0x10, 0x61, 0x01
    };
    return &guid;
}

static const char * npu_backend_name(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return "NPU";
}

static void npu_backend_free(ggml_backend_t backend) {
    delete static_cast<npu_backend_context *>(backend->context);
    delete backend;
}

static ggml_backend_graph_plan_t npu_backend_graph_plan_create(
        ggml_backend_t backend,
        const struct ggml_cgraph * cgraph) {
    GGML_UNUSED(backend);

    auto * graph_plan = new (std::nothrow) npu_graph_plan;
    if (graph_plan == nullptr) {
        return nullptr;
    }

    std::unordered_set<const struct ggml_tensor *> fused_mul_mat_roots;
    fused_mul_mat_roots.reserve(static_cast<size_t>(cgraph->n_nodes));
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        struct ggml_tensor * node = cgraph->nodes[i];
        if (node == nullptr || node->op != GGML_OP_ADD) {
            continue;
        }
        if (!npu_is_fusable_bias_add(node, nullptr)) {
            continue;
        }
        if (node->src[0] != nullptr && node->src[0]->op == GGML_OP_MUL_MAT) {
            fused_mul_mat_roots.insert(node->src[0]);
        } else if (node->src[1] != nullptr && node->src[1]->op == GGML_OP_MUL_MAT) {
            fused_mul_mat_roots.insert(node->src[1]);
        }
    }

    const npu_backend_context * ctx = static_cast<const npu_backend_context *>(backend->context);
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        struct ggml_tensor * node = cgraph->nodes[i];
        if (node == nullptr) {
            continue;
        }
        if (node->op == GGML_OP_MUL_MAT && fused_mul_mat_roots.find(node) != fused_mul_mat_roots.end()) {
            if (npu_debug_log_enabled()) {
                const char * node_name = node->name[0] != '\0' ? node->name : "(unnamed)";
                GGML_LOG_INFO("%s: skip node=%s op=%s reason=%s\n",
                        __func__,
                        node_name,
                        ggml_op_name(node->op),
                        "covered by fused ADD");
            }
            continue;
        }
        std::string reason;
        if (!npu_can_handle_mul_mat(node, &reason)) {
            if (npu_debug_log_enabled()) {
                const char * node_name = node && node->name[0] != '\0' ? node->name : "(unnamed)";
                GGML_LOG_INFO("%s: skip node=%s op=%s reason=%s\n",
                        __func__,
                        node_name,
                        node ? ggml_op_name(node->op) : "(null)",
                        reason.c_str());
            }
            continue;
        }
        graph_plan->nodes.push_back(npu_create_mul_mat_plan(node, ctx->config));
    }

    if (npu_debug_log_enabled()) {
        GGML_LOG_INFO("%s: created graph plan with %zu NPU nodes out of %d graph nodes\n",
                __func__, graph_plan->nodes.size(), cgraph->n_nodes);
        for (size_t i = 0; i < graph_plan->nodes.size(); ++i) {
            const npu_node_plan & node_plan = graph_plan->nodes[i];
            const char * root_name = node_plan.root && node_plan.root->name[0] != '\0' ? node_plan.root->name : "(unnamed)";
            GGML_LOG_INFO("%s: node[%zu] root=%s op=%s summary=%s\n",
                    __func__,
                    i,
                    root_name,
                    node_plan.root ? ggml_op_name(node_plan.root->op) : "(null)",
                    node_plan.summary.c_str());
        }
    }

    return graph_plan;
}

static void npu_backend_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan_ptr) {
    GGML_UNUSED(backend);
    npu_graph_plan * graph_plan = static_cast<npu_graph_plan *>(plan_ptr);
    if (graph_plan != nullptr) {
        for (npu_node_plan & node_plan : graph_plan->nodes) {
            for (npu_prepacked_weight & weight_pack : node_plan.weight_packs) {
                if (weight_pack.cma_packed != nullptr && !weight_pack.cma_persistent) {
                    npu_mem_free(weight_pack.cma_packed);
                    weight_pack.cma_packed = nullptr;
                    weight_pack.cma_bytes = 0;
                    weight_pack.cma_persistent = false;
                }
            }
        }
    }
    delete graph_plan;
}

static enum ggml_status npu_backend_graph_plan_compute(
        ggml_backend_t backend,
        ggml_backend_graph_plan_t plan_ptr) {
    GGML_UNUSED(backend);

    npu_graph_plan * graph_plan = static_cast<npu_graph_plan *>(plan_ptr);
    if (graph_plan == nullptr) {
        return GGML_STATUS_FAILED;
    }

    if ((npu_profile_enabled() || npu_runtime_profile_requested() || npu_summary_active()) && !npu_profile_session_started()) {
        npu_profile_reset();
        npu_profile_next_layer_id() = 0;
        npu_profile_session_started() = true;
    }

    for (size_t i = 0; i < graph_plan->nodes.size(); ++i) {
        const npu_node_plan & node_plan = graph_plan->nodes[i];
        const int64_t layer_id = npu_profile_next_layer_id()++;
        if (npu_debug_log_enabled()) {
            const char * root_name = node_plan.root && node_plan.root->name[0] != '\0' ? node_plan.root->name : "(unnamed)";
            GGML_LOG_INFO("%s: start root=%s op=%s summary=%s\n",
                    __func__,
                    root_name,
                    node_plan.root ? ggml_op_name(node_plan.root->op) : "(null)",
                    node_plan.summary.c_str());
        }
        std::string error;
        const enum ggml_status status = npu_compute_node(node_plan, layer_id, &error);
        if (status != GGML_STATUS_SUCCESS) {
            npu_profile_flush();
            if (npu_debug_log_enabled()) {
                const char * root_name = node_plan.root && node_plan.root->name[0] != '\0' ? node_plan.root->name : "(unnamed)";
                GGML_LOG_ERROR("%s: failed root=%s status=%d error=%s\n",
                        __func__,
                        root_name,
                        status,
                        error.c_str());
            }
            return status;
        }
        if (npu_debug_log_enabled()) {
            const char * root_name = node_plan.root && node_plan.root->name[0] != '\0' ? node_plan.root->name : "(unnamed)";
            GGML_LOG_INFO("%s: done root=%s\n", __func__, root_name);
        }
    }

    npu_profile_flush();
    return GGML_STATUS_SUCCESS;
}

static enum ggml_status npu_backend_graph_compute(
        ggml_backend_t backend,
        struct ggml_cgraph * cgraph) {
    ggml_backend_graph_plan_t plan = npu_backend_graph_plan_create(backend, cgraph);
    if (plan == nullptr) {
        return GGML_STATUS_FAILED;
    }

    const enum ggml_status status = npu_backend_graph_plan_compute(backend, plan);
    npu_backend_graph_plan_free(backend, plan);
    return status;
}

static const ggml_backend_i npu_backend_iface = {
    /* .get_name           = */ npu_backend_name,
    /* .free               = */ npu_backend_free,
    /* .set_tensor_async   = */ nullptr,
    /* .get_tensor_async   = */ nullptr,
    /* .cpy_tensor_async   = */ nullptr,
    /* .synchronize        = */ nullptr,
    /* .graph_plan_create  = */ npu_backend_graph_plan_create,
    /* .graph_plan_free    = */ npu_backend_graph_plan_free,
    /* .graph_plan_update  = */ nullptr,
    /* .graph_plan_compute = */ npu_backend_graph_plan_compute,
    /* .graph_compute      = */ npu_backend_graph_compute,
    /* .event_record       = */ nullptr,
    /* .event_wait         = */ nullptr,
    /* .graph_optimize     = */ nullptr,
};

static const char * npu_device_name(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return "NPU0";
}

static const char * npu_device_description(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return "AICAS NPU INT8 matmul offload prototype";
}

static void npu_device_memory(ggml_backend_dev_t dev, size_t * free_mem, size_t * total_mem) {
    GGML_UNUSED(dev);
    if (free_mem) {
        *free_mem = 0;
    }
    if (total_mem) {
        *total_mem = NPU_DEFAULT_SPM_BYTES;
    }
}

static enum ggml_backend_dev_type npu_device_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

static void npu_device_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name = npu_device_name(dev);
    props->description = npu_device_description(dev);
    props->memory_free = 0;
    props->memory_total = NPU_DEFAULT_SPM_BYTES;
    props->type = GGML_BACKEND_DEVICE_TYPE_ACCEL;
    props->device_id = "aicas-npu0";
    props->caps = {
        /* .async            = */ false,
        /* .host_buffer      = */ true,
        /* .buffer_from_host_ptr = */ true,
        /* .events           = */ false,
    };
}

static ggml_backend_buffer_type_t npu_device_get_buffer_type(ggml_backend_dev_t dev);

static ggml_backend_t npu_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);

    auto * ctx = new (std::nothrow) npu_backend_context;
    if (ctx == nullptr) {
        return nullptr;
    }

    ctx->config = npu_default_tiling_config();

    ggml_backend_t backend = new (std::nothrow) ggml_backend {
        /* .guid    = */ ggml_backend_npu_guid(),
        /* .iface   = */ npu_backend_iface,
        /* .device  = */ dev,
        /* .context = */ ctx,
    };

    if (backend == nullptr) {
        delete ctx;
        return nullptr;
    }

    if (npu_eager_init_enabled() && npu_init() != 0) {
        delete backend;
        delete ctx;
        return nullptr;
    }

    return backend;
}

static ggml_backend_buffer_t npu_device_buffer_from_host_ptr(
        ggml_backend_dev_t dev,
        void * ptr,
        size_t size,
        size_t max_tensor_size) {
    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);

    auto * ctx = new (std::nothrow) npu_buffer_context;
    if (ctx == nullptr) {
        return nullptr;
    }

    ctx->ptr = ptr;
    ctx->own = false;

    return ggml_backend_buffer_init(
        ggml_backend_npu_buffer_type(),
        npu_buffer_iface,
        ctx,
        size);
}

static bool npu_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    GGML_UNUSED(dev);
    return npu_can_handle_mul_mat(op, nullptr);
}

static bool npu_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(dev);
    return ggml_backend_buft_is_host(buft) || buft == ggml_backend_npu_buffer_type();
}

static const ggml_backend_device_i npu_device_iface = {
    /* .get_name             = */ npu_device_name,
    /* .get_description      = */ npu_device_description,
    /* .get_memory           = */ npu_device_memory,
    /* .get_type             = */ npu_device_type,
    /* .get_props            = */ npu_device_props,
    /* .init_backend         = */ npu_device_init_backend,
    /* .get_buffer_type      = */ npu_device_get_buffer_type,
    /* .get_host_buffer_type = */ npu_device_get_buffer_type,
    /* .buffer_from_host_ptr = */ npu_device_buffer_from_host_ptr,
    /* .supports_op          = */ npu_device_supports_op,
    /* .supports_buft        = */ npu_device_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

struct npu_reg_context {
    ggml_backend_device device;
    ggml_backend_buffer_type buffer_type;
};

static const char * npu_reg_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return "NPU";
}

static size_t npu_reg_device_count(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return 1;
}

static ggml_backend_dev_t npu_reg_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);
    npu_reg_context * ctx = static_cast<npu_reg_context *>(reg->context);
    return &ctx->device;
}

static void * npu_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);

    if (std::strcmp(name, "ggml_backend_npu_init") == 0) {
        return reinterpret_cast<void *>(ggml_backend_npu_init);
    }
    if (std::strcmp(name, "ggml_backend_is_npu") == 0) {
        return reinterpret_cast<void *>(ggml_backend_is_npu);
    }
    if (std::strcmp(name, "ggml_backend_npu_w8a8_register") == 0) {
        return reinterpret_cast<void *>(ggml_backend_npu_w8a8_register);
    }
    if (std::strcmp(name, "ggml_backend_npu_w8a8_preload") == 0) {
        return reinterpret_cast<void *>(ggml_backend_npu_w8a8_preload);
    }
    if (std::strcmp(name, "ggml_backend_npu_w8a8_preload_clear") == 0) {
        return reinterpret_cast<void *>(ggml_backend_npu_w8a8_preload_clear);
    }
    if (std::strcmp(name, "ggml_backend_npu_w8a8_clear") == 0) {
        return reinterpret_cast<void *>(ggml_backend_npu_w8a8_clear);
    }
    if (std::strcmp(name, "ggml_backend_npu_i8_gemm_raw_packed") == 0) {
        return reinterpret_cast<void *>(ggml_backend_npu_i8_gemm_raw_packed);
    }
    if (std::strcmp(name, "ggml_backend_npu_i8_gemm_raw_cma") == 0) {
        return reinterpret_cast<void *>(ggml_backend_npu_i8_gemm_raw_cma);
    }
    if (std::strcmp(name, "ggml_backend_npu_mem_alloc") == 0) {
        return reinterpret_cast<void *>(ggml_backend_npu_mem_alloc);
    }
    if (std::strcmp(name, "ggml_backend_npu_mem_free") == 0) {
        return reinterpret_cast<void *>(ggml_backend_npu_mem_free);
    }
    if (std::strcmp(name, "ggml_backend_npu_runtime_shutdown") == 0) {
        return reinterpret_cast<void *>(ggml_backend_npu_runtime_shutdown);
    }
    if (std::strcmp(name, "ggml_backend_npu_decode_runtime_shutdown") == 0) {
        return reinterpret_cast<void *>(ggml_backend_npu_decode_runtime_shutdown);
    }
    if (std::strcmp(name, "ggml_backend_npu_decode_overlay_mark_inactive") == 0) {
        return reinterpret_cast<void *>(ggml_backend_npu_decode_overlay_mark_inactive);
    }
    if (std::strcmp(name, "ggml_backend_npu_decode_overlay_mark_active") == 0) {
        return reinterpret_cast<void *>(ggml_backend_npu_decode_overlay_mark_active);
    }
    if (std::strcmp(name, "ggml_backend_npu_decode_w4a16_gemv") == 0) {
        return reinterpret_cast<void *>(ggml_backend_npu_decode_w4a16_gemv);
    }
    if (std::strcmp(name, "ggml_backend_npu_decode_w4a16_gemv_ex") == 0) {
        return reinterpret_cast<void *>(ggml_backend_npu_decode_w4a16_gemv_ex);
    }
    if (std::strcmp(name, "ggml_backend_npu_decode_w4a16_preload") == 0) {
        return reinterpret_cast<void *>(ggml_backend_npu_decode_w4a16_preload);
    }
    return nullptr;
}

static const ggml_backend_reg_i npu_reg_iface = {
    /* .get_name         = */ npu_reg_name,
    /* .get_device_count = */ npu_reg_device_count,
    /* .get_device       = */ npu_reg_device,
    /* .get_proc_address = */ npu_reg_get_proc_address,
};

static ggml_backend_buffer_type_t npu_device_get_buffer_type(ggml_backend_dev_t dev) {
    npu_reg_context * ctx = static_cast<npu_reg_context *>(dev->reg->context);
    return &ctx->buffer_type;
}

} // namespace ggml_npu

ggml_backend_buffer_type_t ggml_backend_npu_buffer_type(void) {
    return ggml_backend_dev_buffer_type(ggml_backend_reg_dev_get(ggml_backend_npu_reg(), 0));
}

ggml_backend_reg_t ggml_backend_npu_reg(void) {
    using namespace ggml_npu;

    static npu_reg_context ctx;
    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ npu_reg_iface,
        /* .context     = */ &ctx,
    };
    static bool initialized = false;

    if (!initialized) {
        ctx.device = ggml_backend_device {
            /* .iface   = */ npu_device_iface,
            /* .reg     = */ &reg,
            /* .context = */ nullptr,
        };

        ctx.buffer_type = ggml_backend_buffer_type {
            /* .iface   = */ npu_buffer_type_iface,
            /* .device  = */ &ctx.device,
            /* .context = */ nullptr,
        };

        initialized = true;
    }

    return &reg;
}

ggml_backend_t ggml_backend_npu_init(void) {
    return ggml_backend_dev_init(ggml_backend_reg_dev_get(ggml_backend_npu_reg(), 0), nullptr);
}

bool ggml_backend_is_npu(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, ggml_npu::ggml_backend_npu_guid());
}

void ggml_backend_npu_profile_summary_start(void) {
    ggml_npu::npu_summary_session_start();
}

void ggml_backend_npu_profile_summary_stop(ggml_npu_profile_summary * out) {
    ggml_npu::npu_summary_session_stop(out);
}

void ggml_backend_npu_w8a8_clear(void) {
    ggml_npu::npu_clear_aicas_w8a8_table();
}

bool ggml_backend_npu_w8a8_register(
        const char * weight_name,
        float act_scale,
        int32_t act_scale_q8_24,
        int32_t act_zero_point_u8,
        const float * weight_scale,
        size_t weight_scale_len,
        const int32_t * sum_w,
        size_t sum_w_len,
        const float * smooth_scale,
        size_t smooth_scale_len) {
    return ggml_npu::npu_register_aicas_w8a8(
        weight_name,
        act_scale,
        act_scale_q8_24,
        act_zero_point_u8,
        weight_scale,
        weight_scale_len,
        sum_w,
        sum_w_len,
        smooth_scale,
        smooth_scale_len);
}

bool ggml_backend_npu_w8a8_preload(const struct ggml_tensor * weight_tensor) {
    std::string error;
    const bool ok = ggml_npu::npu_preload_aicas_w8a8_tensor(weight_tensor, &error);
    if (!ok && ggml_npu::npu_debug_log_enabled()) {
        GGML_LOG_WARN("%s: preload failed for %s: %s\n",
                __func__,
                weight_tensor && weight_tensor->name[0] ? weight_tensor->name : "(unnamed)",
                error.c_str());
    }
    return ok;
}

void ggml_backend_npu_w8a8_preload_clear(void) {
    ggml_npu::npu_clear_preloaded_weight_cache();
}

void * ggml_backend_npu_mem_alloc(size_t size) {
    if (npu_init() != 0) {
        return nullptr;
    }
    return npu_mem_alloc(size);
}

void ggml_backend_npu_mem_free(void * ptr) {
    npu_mem_free(ptr);
}

void ggml_backend_npu_runtime_shutdown(void) {
    ggml_npu::npu_clear_preloaded_weight_cache();
    npu_destroy();
}

void ggml_backend_npu_decode_runtime_shutdown(void) {
    ggml_npu::npu_decode_profile_flush();
    npu_decode_destroy();
}

void ggml_backend_npu_decode_overlay_mark_inactive(void) {
    using namespace ggml_npu;
    std::lock_guard<std::mutex> lock(npu_decode_overlay_mutex());
    npu_decode_overlay_active() = false;
}

void ggml_backend_npu_decode_overlay_mark_active(void) {
    using namespace ggml_npu;
    std::lock_guard<std::mutex> lock(npu_decode_overlay_mutex());
    npu_decode_overlay_active() = true;
}

bool ggml_backend_npu_i8_gemm_raw_cma(
        const char * op_name,
        const int8_t * weight_cma_kxm,
        int64_t m,
        int64_t k,
        int64_t weight_stride_m,
        const int8_t * act_cma_nxk,
        int64_t n,
        int64_t act_stride_k,
        int32_t * out_nxm,
        int64_t out_stride_m) {
    using namespace ggml_npu;

    std::string error;
    if (!npu_raw_i8_gemm_validate(
            op_name,
            weight_cma_kxm,
            m,
            k,
            weight_stride_m,
            act_cma_nxk,
            n,
            act_stride_k,
            out_nxm,
            out_stride_m,
            &error)) {
        return false;
    }

    int64_t tile_m = std::min<int64_t>(m, 240);
    int64_t tile_n = std::min<int64_t>(n, 64);
    int64_t tile_m_stride = npu_align_up_i64(tile_m, NPU_GEMM_PLAN_STRIDE_ALIGNMENT);
    bool tile_fits = false;
    while (tile_m > 0 && tile_n > 0) {
        tile_m_stride = npu_align_up_i64(tile_m, NPU_GEMM_PLAN_STRIDE_ALIGNMENT);
        const uint64_t max_act_bytes = static_cast<uint64_t>(tile_n) * static_cast<uint64_t>(act_stride_k);
        const uint64_t max_weight_bytes = static_cast<uint64_t>(k) * static_cast<uint64_t>(tile_m_stride);
        const uint64_t max_out_bytes = static_cast<uint64_t>(tile_n) * static_cast<uint64_t>(tile_m_stride) * sizeof(int32_t);
        const uint64_t spm_weight_addr = static_cast<uint64_t>(npu_align_up_i64(static_cast<int64_t>(max_act_bytes), NPU_SPM_ALIGNMENT));
        const uint64_t spm_total = spm_weight_addr + max_weight_bytes;
        const uint64_t acc_scratch_addr = static_cast<uint64_t>(npu_align_up_i64(static_cast<int64_t>(max_out_bytes), NPU_GEMM_PLAN_ADDR_ALIGNMENT));
        const uint64_t acc_total = acc_scratch_addr + max_out_bytes;
        if (spm_total + NPU_DEFAULT_GUARD_BYTES <= NPU_DEFAULT_SPM_BYTES &&
                acc_total + NPU_DEFAULT_GUARD_BYTES <= NPU_DEFAULT_ACC_BYTES &&
                npu_fits_u16(tile_m_stride)) {
            tile_fits = true;
            break;
        }
        if (tile_n > 16) {
            tile_n /= 2;
        } else {
            tile_m /= 2;
        }
    }

    if (!tile_fits) {
        if (npu_raw_i8_gemm_debug_log_enabled()) {
            GGML_LOG_WARN("%s: reject %s: no raw GEMM tile fits SPM/ACC capacity\n",
                    __func__, op_name != nullptr ? op_name : "(unnamed)");
        }
        return false;
    }

    tile_m_stride = npu_align_up_i64(tile_m, NPU_GEMM_PLAN_STRIDE_ALIGNMENT);
    const size_t max_act_bytes = static_cast<size_t>(tile_n * act_stride_k);
    const size_t max_acc_out_bytes = static_cast<size_t>(tile_n * tile_m_stride * sizeof(int32_t));
    const size_t max_host_out_bytes = static_cast<size_t>(tile_n * tile_m * sizeof(int32_t));
    const uint32_t spm_act_addr = 0;
    const uint32_t spm_weight_addr = static_cast<uint32_t>(npu_align_up_i64(static_cast<int64_t>(max_act_bytes), NPU_SPM_ALIGNMENT));
    const uint32_t acc_out_addr = 0;
    const uint32_t acc_scratch_addr = static_cast<uint32_t>(npu_align_up_i64(static_cast<int64_t>(max_acc_out_bytes), NPU_GEMM_PLAN_ADDR_ALIGNMENT));

    if (npu_init() != 0) {
        if (npu_raw_i8_gemm_debug_log_enabled()) {
            GGML_LOG_WARN("%s: npu_init failed for %s\n", __func__, op_name != nullptr ? op_name : "(unnamed)");
        }
        return false;
    }

    void * out_cma = npu_mem_alloc(max_host_out_bytes);
    if (out_cma == nullptr) {
        if (npu_raw_i8_gemm_debug_log_enabled()) {
            GGML_LOG_WARN("%s: output CMA allocation failed for %s\n", __func__, op_name != nullptr ? op_name : "(unnamed)");
        }
        return false;
    }

    for (int64_t n0 = 0; n0 < n; n0 += tile_n) {
        const int64_t cur_n = std::min(tile_n, n - n0);
        for (int64_t m0 = 0; m0 < m; m0 += tile_m) {
            const int64_t cur_m = std::min(tile_m, m - m0);
            const MvinConfig act_mvin_cfg = {
                const_cast<int8_t *>(act_cma_nxk + n0 * act_stride_k),
                spm_act_addr,
                static_cast<uint32_t>(k - 1),
                static_cast<uint32_t>(cur_n - 1),
                static_cast<uint16_t>(act_stride_k),
                static_cast<uint32_t>(act_stride_k),
                1,
                0,
                false,
                false,
                false,
                0,
                0,
                0,
            };
            const MvinConfig weight_mvin_cfg = {
                const_cast<int8_t *>(weight_cma_kxm + m0),
                spm_weight_addr,
                static_cast<uint32_t>(cur_m - 1),
                static_cast<uint32_t>(k - 1),
                static_cast<uint16_t>(tile_m_stride),
                static_cast<uint32_t>(weight_stride_m),
                1,
                1,
                false,
                false,
                false,
                0,
                0,
                0,
            };
            npu_dma_mvin_async(0, &act_mvin_cfg);
            npu_dma_wait_mvin(1u << 0);
            npu_dma_mvin_async(1, &weight_mvin_cfg);
            npu_dma_wait_mvin(1u << 1);

            npu_gemm_plan_run_ex(
                    spm_act_addr,
                    spm_weight_addr,
                    acc_out_addr,
                    acc_scratch_addr,
                    0,
                    static_cast<uint16_t>(cur_n),
                    static_cast<uint16_t>(cur_m),
                    static_cast<uint16_t>(k),
                    static_cast<uint16_t>(act_stride_k),
                    static_cast<uint16_t>(tile_m_stride),
                    static_cast<uint16_t>(tile_m_stride),
                    0,
                    false,
                    false,
                    false);

            npu_dma_mvout_ex(
                    out_cma,
                    acc_out_addr,
                    static_cast<uint32_t>(cur_m - 1),
                    static_cast<uint32_t>(cur_n - 1),
                    static_cast<uint16_t>(tile_m_stride),
                    static_cast<uint32_t>(cur_m),
                    1,
                    1,
                    true,
                    false,
                    0,
                    0,
                    false);

            const int32_t * tile_out = static_cast<const int32_t *>(out_cma);
            for (int64_t row = 0; row < cur_n; ++row) {
                std::memcpy(
                        out_nxm + (n0 + row) * out_stride_m + m0,
                        tile_out + row * cur_m,
                        static_cast<size_t>(cur_m) * sizeof(int32_t));
            }
        }
    }

    npu_mem_free(out_cma);
    return true;
}

bool ggml_backend_npu_i8_gemm_raw_packed(
        const char * op_name,
        const int8_t * weight_kxm,
        int64_t m,
        int64_t k,
        int64_t weight_stride_m,
        const int8_t * act_nxk,
        int64_t n,
        int64_t act_stride_k,
        int32_t * out_nxm,
        int64_t out_stride_m) {
    using namespace ggml_npu;

    std::string error;
    if (!npu_raw_i8_gemm_validate(
            op_name,
            weight_kxm,
            m,
            k,
            weight_stride_m,
            act_nxk,
            n,
            act_stride_k,
            out_nxm,
            out_stride_m,
            &error)) {
        return false;
    }

    int64_t tile_m = std::min<int64_t>(m, 240);
    int64_t tile_n = std::min<int64_t>(n, 128);
    int64_t tile_m_stride = npu_align_up_i64(tile_m, NPU_GEMM_PLAN_STRIDE_ALIGNMENT);
    bool tile_fits = false;
    while (tile_m > 0 && tile_n > 0) {
        tile_m_stride = npu_align_up_i64(tile_m, NPU_GEMM_PLAN_STRIDE_ALIGNMENT);
        const uint64_t max_act_bytes = static_cast<uint64_t>(tile_n) * static_cast<uint64_t>(act_stride_k);
        const uint64_t max_weight_bytes = static_cast<uint64_t>(k) * static_cast<uint64_t>(tile_m_stride);
        const uint64_t max_out_bytes = static_cast<uint64_t>(tile_n) * static_cast<uint64_t>(tile_m_stride) * sizeof(int32_t);
        const uint64_t spm_weight_addr = static_cast<uint64_t>(npu_align_up_i64(static_cast<int64_t>(max_act_bytes), NPU_SPM_ALIGNMENT));
        const uint64_t spm_total = spm_weight_addr + max_weight_bytes;
        const uint64_t acc_scratch_addr = static_cast<uint64_t>(npu_align_up_i64(static_cast<int64_t>(max_out_bytes), NPU_GEMM_PLAN_ADDR_ALIGNMENT));
        const uint64_t acc_total = acc_scratch_addr + max_out_bytes;
        if (spm_total + NPU_DEFAULT_GUARD_BYTES <= NPU_DEFAULT_SPM_BYTES &&
                acc_total + NPU_DEFAULT_GUARD_BYTES <= NPU_DEFAULT_ACC_BYTES &&
                npu_fits_u16(tile_m_stride)) {
            tile_fits = true;
            break;
        }
        if (tile_n > 16) {
            tile_n /= 2;
        } else {
            tile_m /= 2;
        }
    }

    if (!tile_fits) {
        if (npu_raw_i8_gemm_debug_log_enabled()) {
            GGML_LOG_WARN("%s: reject %s: no raw GEMM tile fits SPM/ACC capacity\n",
                    __func__, op_name != nullptr ? op_name : "(unnamed)");
        }
        return false;
    }

    tile_m_stride = npu_align_up_i64(tile_m, NPU_GEMM_PLAN_STRIDE_ALIGNMENT);
    const size_t max_act_bytes = static_cast<size_t>(tile_n * act_stride_k);
    const size_t max_weight_bytes = static_cast<size_t>(k * tile_m_stride);
    const size_t max_acc_out_bytes = static_cast<size_t>(tile_n * tile_m_stride * sizeof(int32_t));
    const size_t max_host_out_bytes = static_cast<size_t>(tile_n * tile_m * sizeof(int32_t));
    const uint32_t spm_act_addr = 0;
    const uint32_t spm_weight_addr = static_cast<uint32_t>(npu_align_up_i64(static_cast<int64_t>(max_act_bytes), NPU_SPM_ALIGNMENT));
    const uint32_t acc_out_addr = 0;
    const uint32_t acc_scratch_addr = static_cast<uint32_t>(npu_align_up_i64(static_cast<int64_t>(max_acc_out_bytes), NPU_GEMM_PLAN_ADDR_ALIGNMENT));

    if (npu_init() != 0) {
        if (npu_raw_i8_gemm_debug_log_enabled()) {
            GGML_LOG_WARN("%s: npu_init failed for %s\n", __func__, op_name != nullptr ? op_name : "(unnamed)");
        }
        return false;
    }

    void * act_cma = npu_mem_alloc(max_act_bytes);
    void * weight_cma = npu_mem_alloc(max_weight_bytes);
    void * out_cma = npu_mem_alloc(max_host_out_bytes);
    if (act_cma == nullptr || weight_cma == nullptr || out_cma == nullptr) {
        npu_mem_free(act_cma);
        npu_mem_free(weight_cma);
        npu_mem_free(out_cma);
        if (npu_raw_i8_gemm_debug_log_enabled()) {
            GGML_LOG_WARN("%s: CMA allocation failed for %s\n", __func__, op_name != nullptr ? op_name : "(unnamed)");
        }
        return false;
    }

    std::vector<int8_t> act_tile(max_act_bytes);
    std::vector<int8_t> weight_tile(max_weight_bytes);

    for (int64_t n0 = 0; n0 < n; n0 += tile_n) {
        const int64_t cur_n = std::min(tile_n, n - n0);
        for (int64_t m0 = 0; m0 < m; m0 += tile_m) {
            const int64_t cur_m = std::min(tile_m, m - m0);
            if (npu_raw_i8_gemm_debug_log_enabled()) {
                GGML_LOG_INFO("%s: %s tile n0=%" PRId64 " m0=%" PRId64 " n=%" PRId64 " m=%" PRId64 " k=%" PRId64 " a_stride=%" PRId64 " b_stride=%" PRId64 "\n",
                        __func__,
                        op_name != nullptr ? op_name : "(unnamed)",
                        n0,
                        m0,
                        cur_n,
                        cur_m,
                        k,
                        act_stride_k,
                        tile_m_stride);
            }
            std::fill(act_tile.begin(), act_tile.end(), 0);
            std::fill(weight_tile.begin(), weight_tile.end(), 0);

            for (int64_t row = 0; row < cur_n; ++row) {
                std::memcpy(
                        act_tile.data() + row * act_stride_k,
                        act_nxk + (n0 + row) * act_stride_k,
                        static_cast<size_t>(k));
            }
            for (int64_t kk = 0; kk < k; ++kk) {
                std::memcpy(
                        weight_tile.data() + kk * tile_m_stride,
                        weight_kxm + kk * weight_stride_m + m0,
                        static_cast<size_t>(cur_m));
            }

            const size_t cur_act_bytes = static_cast<size_t>(cur_n * act_stride_k);
            const size_t cur_weight_bytes = static_cast<size_t>(k * tile_m_stride);
            std::memcpy(act_cma, act_tile.data(), cur_act_bytes);
            std::memcpy(weight_cma, weight_tile.data(), cur_weight_bytes);

            const MvinConfig act_mvin_cfg = {
                act_cma,
                spm_act_addr,
                static_cast<uint32_t>(k - 1),
                static_cast<uint32_t>(cur_n - 1),
                static_cast<uint16_t>(act_stride_k),
                static_cast<uint32_t>(act_stride_k),
                1,
                0,
                false,
                false,
                false,
                0,
                0,
                0,
            };
            const MvinConfig weight_mvin_cfg = {
                weight_cma,
                spm_weight_addr,
                static_cast<uint32_t>(cur_m - 1),
                static_cast<uint32_t>(k - 1),
                static_cast<uint16_t>(tile_m_stride),
                static_cast<uint32_t>(tile_m_stride),
                1,
                1,
                false,
                false,
                false,
                0,
                0,
                0,
            };
            npu_dma_mvin_async(0, &act_mvin_cfg);
            npu_dma_mvin_async(1, &weight_mvin_cfg);
            npu_dma_wait_mvin((1u << 0) | (1u << 1));

            npu_gemm_plan_run_ex(
                    spm_act_addr,
                    spm_weight_addr,
                    acc_out_addr,
                    acc_scratch_addr,
                    0,
                    static_cast<uint16_t>(cur_n),
                    static_cast<uint16_t>(cur_m),
                    static_cast<uint16_t>(k),
                    static_cast<uint16_t>(act_stride_k),
                    static_cast<uint16_t>(tile_m_stride),
                    static_cast<uint16_t>(tile_m_stride),
                    0,
                    false,
                    false,
                    false);

            npu_dma_mvout_ex(
                    out_cma,
                    acc_out_addr,
                    static_cast<uint32_t>(cur_m - 1),
                    static_cast<uint32_t>(cur_n - 1),
                    static_cast<uint16_t>(tile_m_stride),
                    static_cast<uint32_t>(cur_m),
                    1,
                    1,
                    true,
                    false,
                    0,
                    0,
                    false);

            const int32_t * tile_out = static_cast<const int32_t *>(out_cma);
            for (int64_t row = 0; row < cur_n; ++row) {
                std::memcpy(
                        out_nxm + (n0 + row) * out_stride_m + m0,
                        tile_out + row * cur_m,
                        static_cast<size_t>(cur_m) * sizeof(int32_t));
            }
        }
    }

    npu_mem_free(act_cma);
    npu_mem_free(weight_cma);
    npu_mem_free(out_cma);
    return true;
}

bool ggml_backend_npu_decode_w4a16_preload(
        const char * weight_name,
        const void * q4_data,
        int64_t packed_k,
        int64_t out_channels,
        int64_t q4_nb1,
        const void * scale_data,
        int scale_type,
        int64_t scale_nb0,
        int64_t scale_nb1,
        const void * zero_data,
        int zero_type,
        int64_t zero_nb0,
        int64_t zero_nb1,
        int64_t k,
        const float * smooth_scale,
        size_t smooth_scale_len) {
    using namespace ggml_npu;

    auto fail = [weight_name](const char * reason) {
        if (npu_decode_w4a16_debug_log_enabled()) {
            GGML_LOG_WARN("%s: reject %s: %s\n", __func__, weight_name != nullptr ? weight_name : "(unnamed)", reason);
        }
        return false;
    };

    if (weight_name == nullptr || weight_name[0] == '\0' ||
            q4_data == nullptr || scale_data == nullptr || zero_data == nullptr) {
        return fail("null pointer");
    }
    if (k <= 0 || out_channels <= 0 || packed_k != (k + 1) / 2) {
        return fail("invalid dimensions");
    }
    if (k > std::numeric_limits<uint16_t>::max() || out_channels > std::numeric_limits<uint16_t>::max()) {
        return fail("dimension exceeds uint16 range");
    }
    if (scale_type != GGML_TYPE_F16 && scale_type != GGML_TYPE_F32) {
        return fail("scale tensor must be F16 or F32");
    }
    if (zero_type != GGML_TYPE_F16 && zero_type != GGML_TYPE_F32) {
        return fail("zero tensor must be F16 or F32");
    }
    if (smooth_scale == nullptr || smooth_scale_len != static_cast<size_t>(k)) {
        return fail("invalid smooth scale");
    }

    const int64_t groups = (k + NPU_DECODE_GEMV_K_TILE - 1) / NPU_DECODE_GEMV_K_TILE;
    if (!npu_decode_validate_symmetric_zero(
            zero_data,
            zero_type,
            zero_nb0,
            zero_nb1,
            out_channels,
            groups,
            weight_name)) {
        return false;
    }

    const uint32_t total_row_tiles = npu_decode_ceil_div_u32(static_cast<uint32_t>(out_channels), NPU_DECODE_GEMV_M_TILE);
    uint32_t max_block_row_tiles = 0;
    for (uint32_t row_tiles = total_row_tiles; row_tiles >= 1; --row_tiles) {
        const uint32_t candidate_rows = std::min<uint32_t>(row_tiles * NPU_DECODE_GEMV_M_TILE, static_cast<uint32_t>(out_channels));
        npu_decode_gemv_layout candidate = {};
        if (npu_decode_make_gemv_layout(candidate_rows, static_cast<uint32_t>(k), &candidate)) {
            max_block_row_tiles = row_tiles;
            break;
        }
        if (row_tiles == 1) {
            break;
        }
    }
    if (max_block_row_tiles == 0) {
        return fail("no M block fits decode GEMV SPM capacity");
    }

    npu_decode_preloaded_tensor entry;
    entry.packed_k = packed_k;
    entry.out_channels = out_channels;
    entry.q4_nb1 = q4_nb1;
    entry.k = k;
    entry.max_block_rows = max_block_row_tiles * NPU_DECODE_GEMV_M_TILE;
    entry.zero_validated = true;
    entry.inv_smooth_scale.resize(static_cast<size_t>(k));
    for (int64_t i = 0; i < k; ++i) {
        const float smooth = smooth_scale[i];
        entry.inv_smooth_scale[static_cast<size_t>(i)] =
            smooth != 0.0f && std::isfinite(smooth) ? 1.0f / smooth : 1.0f;
    }
    if (!npu_decode_make_gemv_layout(entry.max_block_rows, static_cast<uint32_t>(k), &entry.max_layout)) {
        return fail("max block layout failed");
    }
    entry.pingpong_available = npu_decode_make_pingpong_layout(
            static_cast<uint32_t>(out_channels),
            static_cast<uint32_t>(k),
            &entry.pingpong_layout);
    if (entry.pingpong_available) {
        entry.pingpong_scale.resize(entry.pingpong_layout.scale_bytes);
        entry.pingpong_weight.resize(entry.pingpong_layout.weight_bytes);
        npu_decode_pack_pingpong_scale_weight(
                entry.pingpong_scale,
                entry.pingpong_weight,
                entry.pingpong_layout,
                q4_data,
                packed_k,
                q4_nb1,
                scale_data,
                scale_type,
                scale_nb0,
                scale_nb1,
                out_channels,
                k);
    }

    void * cma_base = nullptr;
    uint32_t cma_map_size = 0;
    uint32_t cma_base_offset = npu_decode_parse_size_env("NPU_CMA_HEAP_OFFSET", 0);
    uint32_t cma_limit_bytes = npu_decode_parse_size_env("NPU_CMA_HEAP_SIZE", 0);
    uint32_t cma_cursor = 0;
    {
        npu_decode_preload_cache & cache = npu_decode_get_preload_cache();
        std::lock_guard<std::mutex> lock(cache.mutex);
        if (cache.cma_limit_bytes != 0) {
            cma_base_offset = cache.cma_base_offset;
            cma_limit_bytes = cache.cma_limit_bytes;
            cma_cursor = cache.cma_cursor;
        }
    }
    if (npu_decode_init() == 0) {
        cma_base = npu_decode_memory_base();
        cma_map_size = npu_decode_memory_size();
        if (cma_limit_bytes == 0 && cma_base_offset < cma_map_size) {
            cma_limit_bytes = cma_map_size - cma_base_offset;
        }
    }
    if (cma_base_offset >= cma_map_size || cma_limit_bytes > cma_map_size - cma_base_offset) {
        cma_base = nullptr;
        cma_limit_bytes = 0;
        cma_cursor = 0;
    }

    if (entry.pingpong_available && cma_base != nullptr && cma_limit_bytes != 0) {
        const uint32_t pingpong_cma_start = cma_cursor;
        const uint32_t scale_cursor = npu_decode_align_up_bytes(cma_cursor, NPU_DECODE_GEMV_DMA_ALIGN);
        if (entry.pingpong_scale.size() <= static_cast<size_t>(std::numeric_limits<uint32_t>::max()) &&
                scale_cursor <= cma_limit_bytes &&
                static_cast<uint32_t>(entry.pingpong_scale.size()) <= cma_limit_bytes - scale_cursor) {
            entry.pingpong_scale_cma_offset = cma_base_offset + scale_cursor;
            entry.pingpong_scale_cma_bytes = static_cast<uint32_t>(entry.pingpong_scale.size());
            std::memcpy(
                    static_cast<uint8_t *>(cma_base) + entry.pingpong_scale_cma_offset,
                    entry.pingpong_scale.data(),
                    entry.pingpong_scale.size());
            cma_cursor = scale_cursor + entry.pingpong_scale_cma_bytes;
        }

        const uint32_t weight_cursor = npu_decode_align_up_bytes(cma_cursor, NPU_DECODE_GEMV_DMA_ALIGN);
        if (entry.pingpong_weight.size() <= static_cast<size_t>(std::numeric_limits<uint32_t>::max()) &&
                weight_cursor <= cma_limit_bytes &&
                static_cast<uint32_t>(entry.pingpong_weight.size()) <= cma_limit_bytes - weight_cursor) {
            entry.pingpong_weight_cma_offset = cma_base_offset + weight_cursor;
            entry.pingpong_weight_cma_bytes = static_cast<uint32_t>(entry.pingpong_weight.size());
            std::memcpy(
                    static_cast<uint8_t *>(cma_base) + entry.pingpong_weight_cma_offset,
                    entry.pingpong_weight.data(),
                    entry.pingpong_weight.size());
            cma_cursor = weight_cursor + entry.pingpong_weight_cma_bytes;
        }

        if (entry.pingpong_scale_cma_bytes == 0 || entry.pingpong_weight_cma_bytes == 0) {
            entry.pingpong_scale_cma_offset = 0;
            entry.pingpong_scale_cma_bytes = 0;
            entry.pingpong_weight_cma_offset = 0;
            entry.pingpong_weight_cma_bytes = 0;
            cma_cursor = pingpong_cma_start;
        }
    }

    const bool reserve_cma_for_pingpong =
        entry.pingpong_scale_cma_bytes != 0 && entry.pingpong_weight_cma_bytes != 0;

    for (int64_t row_base = 0; row_base < out_channels; row_base += entry.max_block_rows) {
        const uint32_t block_rows = static_cast<uint32_t>(std::min<int64_t>(entry.max_block_rows, out_channels - row_base));
        npu_decode_preloaded_block block;
        block.row_base = row_base;
        block.block_rows = block_rows;
        if (!npu_decode_make_gemv_layout(block_rows, static_cast<uint32_t>(k), &block.layout)) {
            return fail("block layout failed");
        }
        block.packed.resize(block.layout.packed_bytes);
        npu_decode_pack_scale_weight_block(
                block.packed,
                block.layout,
                q4_data,
                packed_k,
                q4_nb1,
                scale_data,
                scale_type,
                scale_nb0,
                scale_nb1,
                row_base,
                block_rows,
                k);
        if (!reserve_cma_for_pingpong && cma_base != nullptr && cma_limit_bytes != 0) {
            const uint32_t aligned_cursor = npu_decode_align_up_bytes(cma_cursor, NPU_DECODE_GEMV_DMA_ALIGN);
            if (block.packed.size() <= static_cast<size_t>(std::numeric_limits<uint32_t>::max()) &&
                    aligned_cursor <= cma_limit_bytes &&
                    static_cast<uint32_t>(block.packed.size()) <= cma_limit_bytes - aligned_cursor) {
                block.cma_offset = cma_base_offset + aligned_cursor;
                block.cma_bytes = static_cast<uint32_t>(block.packed.size());
                std::memcpy(static_cast<uint8_t *>(cma_base) + block.cma_offset, block.packed.data(), block.packed.size());
                cma_cursor = aligned_cursor + block.cma_bytes;
            }
        }
        entry.blocks.push_back(std::move(block));
    }

    npu_decode_preload_cache & cache = npu_decode_get_preload_cache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    if (cma_base != nullptr && cma_limit_bytes != 0) {
        cache.cma_base_offset = cma_base_offset;
        cache.cma_limit_bytes = cma_limit_bytes;
        cache.cma_cursor = cma_cursor;
    }
    cache.entries[weight_name] = std::move(entry);
    return true;
}

bool ggml_backend_npu_decode_w4a16_gemv_ex(
        const char * op_name,
        const void * q4_data,
        int64_t packed_k,
        int64_t out_channels,
        int64_t q4_nb1,
        const void * scale_data,
        int scale_type,
        int64_t scale_nb0,
        int64_t scale_nb1,
        const void * zero_data,
        int zero_type,
        int64_t zero_nb0,
        int64_t zero_nb1,
        const void * act_data,
        int act_type,
        int64_t act_nb0,
        int64_t act_nb1,
        const float * smooth_scale,
        int64_t k,
        int64_t n_cols,
        void * dst_data,
        int dst_type,
        int64_t dst_nb1) {
    using namespace ggml_npu;

    const bool collect_profile = npu_decode_profile_enabled();
    const int64_t total_start_us = collect_profile ? ggml_time_us() : 0;
    const int64_t setup_start_us = collect_profile ? total_start_us : 0;
    int64_t setup_stage_start_us = setup_start_us;
    npu_decode_gemv_profile_record profile = {};
    profile.op_name = op_name != nullptr ? op_name : "";
    profile.dst_type = dst_type;
    profile.k = k;
    profile.n_cols = n_cols;
    profile.out_channels = out_channels;

    auto fail = [op_name](const char * reason) {
        if (npu_decode_w4a16_debug_log_enabled()) {
            GGML_LOG_WARN("%s: reject %s: %s\n", __func__, op_name != nullptr ? op_name : "(unnamed)", reason);
        }
        return false;
    };

    GGML_UNUSED(scale_nb0);
    GGML_UNUSED(scale_nb1);
    GGML_UNUSED(zero_nb0);
    GGML_UNUSED(zero_nb1);
    GGML_UNUSED(smooth_scale);

    if (q4_data == nullptr || scale_data == nullptr || zero_data == nullptr || act_data == nullptr ||
            smooth_scale == nullptr || dst_data == nullptr) {
        return fail("null pointer");
    }
    if (k <= 0 || n_cols <= 0 || out_channels <= 0 || packed_k != (k + 1) / 2) {
        return fail("invalid dimensions");
    }
    if (k > std::numeric_limits<uint16_t>::max() || out_channels > std::numeric_limits<uint16_t>::max()) {
        return fail("dimension exceeds uint16 range");
    }
    if (scale_type != GGML_TYPE_F16 && scale_type != GGML_TYPE_F32) {
        return fail("scale tensor must be F16 or F32");
    }
    if (zero_type != GGML_TYPE_F16 && zero_type != GGML_TYPE_F32) {
        return fail("zero tensor must be F16 or F32");
    }
    if (act_type != GGML_TYPE_F16 && act_type != GGML_TYPE_F32) {
        return fail("activation tensor must be F16 or F32");
    }
    if (dst_type != GGML_TYPE_F16 && dst_type != GGML_TYPE_F32) {
        return fail("destination tensor must be F16 or F32");
    }

    if (collect_profile) {
        profile.validation_us = 0;
        profile.layout_us = 0;
        setup_stage_start_us = ggml_time_us();
    }

    const npu_decode_preloaded_tensor * preloaded = npu_decode_lookup_preloaded_tensor(
            op_name,
            packed_k,
            out_channels,
            q4_nb1,
            k);
    if (preloaded == nullptr || !preloaded->zero_validated || preloaded->blocks.empty()) {
        return fail("decode tensor is not preloaded");
    }
    const uint32_t max_block_rows = preloaded->max_block_rows;
    const npu_decode_gemv_layout first_layout = preloaded->max_layout;
    const uint32_t total_output_bytes = npu_decode_align_up_u32(
            static_cast<uint32_t>(out_channels) * static_cast<uint32_t>(sizeof(ggml_fp16_t)),
            NPU_DECODE_GEMV_LINE_BYTES);
    if (total_output_bytes > NPU_DEFAULT_SPM_BYTES) {
        return fail("decode output exceeds output SPM capacity");
    }
    bool preloaded_cma = preloaded != nullptr && !preloaded->blocks.empty();
    if (preloaded_cma) {
        for (const npu_decode_preloaded_block & block : preloaded->blocks) {
            if (block.cma_bytes == 0 || block.cma_bytes != block.packed.size()) {
                preloaded_cma = false;
                break;
            }
        }
    }
    const bool pingpong_requested =
        npu_decode_pingpong_enabled() &&
        preloaded->pingpong_available &&
        preloaded->pingpong_scale.size() == preloaded->pingpong_layout.scale_bytes &&
        preloaded->pingpong_weight.size() == preloaded->pingpong_layout.weight_bytes;
    bool pingpong_cma =
        pingpong_requested &&
        preloaded->pingpong_scale_cma_bytes == preloaded->pingpong_layout.scale_bytes &&
        preloaded->pingpong_weight_cma_bytes == preloaded->pingpong_layout.weight_bytes;
    if (collect_profile) {
        profile.preload_lookup_us = static_cast<uint64_t>(ggml_time_us() - setup_stage_start_us);
        setup_stage_start_us = ggml_time_us();
    }
    if (!npu_decode_ensure_overlay_active(op_name)) {
        return fail("decode overlay switch failed");
    }
    if (collect_profile) {
        profile.overlay_ensure_us = static_cast<uint64_t>(ggml_time_us() - setup_stage_start_us);
        setup_stage_start_us = ggml_time_us();
    }
    if (preloaded_cma || pingpong_cma) {
        npu_decode_update_runtime_heap_env();
    }
    if (collect_profile) {
        profile.heap_update_us = static_cast<uint64_t>(ggml_time_us() - setup_stage_start_us);
        setup_stage_start_us = ggml_time_us();
    }
    if (npu_decode_init() != 0) {
        return fail("decode runtime init failed");
    }
    if (collect_profile) {
        profile.runtime_init_us = static_cast<uint64_t>(ggml_time_us() - setup_stage_start_us);
        setup_stage_start_us = ggml_time_us();
    }
    void * decode_cma_base = (preloaded_cma || pingpong_cma) ? npu_decode_memory_base() : nullptr;
    if (preloaded_cma && decode_cma_base == nullptr) {
        preloaded_cma = false;
    }
    if (pingpong_cma && decode_cma_base == nullptr) {
        pingpong_cma = false;
    }
    if (collect_profile) {
        profile.memory_base_us = static_cast<uint64_t>(ggml_time_us() - setup_stage_start_us);
    }

    if (collect_profile) {
        profile.max_block_rows = max_block_rows;
        profile.row_blocks_per_col = npu_decode_ceil_div_u32(static_cast<uint32_t>(out_channels), max_block_rows);
        profile.preloaded_host = preloaded != nullptr;
        profile.preloaded_cma = preloaded_cma || pingpong_cma;
        profile.pingpong = pingpong_requested;
        profile.activation_bytes = static_cast<uint64_t>(first_layout.act_bytes) * static_cast<uint64_t>(n_cols);
        profile.setup_us = static_cast<uint64_t>(ggml_time_us() - setup_start_us);
    }

    const int64_t alloc_start_us = collect_profile ? ggml_time_us() : 0;
    void * act_cma = npu_decode_mem_alloc(first_layout.act_bytes);
    void * packed_cma = preloaded_cma ? nullptr : npu_decode_mem_alloc(first_layout.packed_bytes);
    void * out_cma = npu_decode_mem_alloc(total_output_bytes);
    void * pingpong_scale_cma = nullptr;
    void * pingpong_weight_cma = nullptr;
    if (pingpong_requested && !pingpong_cma) {
        pingpong_scale_cma = npu_decode_mem_alloc(preloaded->pingpong_layout.scale_bytes);
        pingpong_weight_cma = npu_decode_mem_alloc(preloaded->pingpong_layout.weight_bytes);
    }
    if (act_cma == nullptr || (!preloaded_cma && packed_cma == nullptr) || out_cma == nullptr ||
            (pingpong_requested && !pingpong_cma && (pingpong_scale_cma == nullptr || pingpong_weight_cma == nullptr))) {
        npu_decode_mem_free(act_cma);
        npu_decode_mem_free(packed_cma);
        npu_decode_mem_free(out_cma);
        npu_decode_mem_free(pingpong_scale_cma);
        npu_decode_mem_free(pingpong_weight_cma);
        return fail("decode CMA allocation failed");
    }
    if (pingpong_requested && !pingpong_cma) {
        std::memcpy(pingpong_scale_cma, preloaded->pingpong_scale.data(), preloaded->pingpong_layout.scale_bytes);
        std::memcpy(pingpong_weight_cma, preloaded->pingpong_weight.data(), preloaded->pingpong_layout.weight_bytes);
    }

    if (collect_profile) {
        profile.alloc_us = static_cast<uint64_t>(ggml_time_us() - alloc_start_us);
    }

    if (pingpong_requested) {
        void * pingpong_scale_ptr = pingpong_cma ?
            static_cast<uint8_t *>(decode_cma_base) + preloaded->pingpong_scale_cma_offset :
            pingpong_scale_cma;
        void * pingpong_weight_ptr = pingpong_cma ?
            static_cast<uint8_t *>(decode_cma_base) + preloaded->pingpong_weight_cma_offset :
            pingpong_weight_cma;

        bool pingpong_ok = true;
        for (int64_t col = 0; col < n_cols && pingpong_ok; ++col) {
            int64_t stage_start_us = collect_profile ? ggml_time_us() : 0;
            npu_decode_pack_activation(
                    act_cma,
                    preloaded->pingpong_layout.act_bytes,
                    preloaded->pingpong_layout,
                    act_data,
                    act_type,
                    act_nb0,
                    act_nb1,
                    preloaded->inv_smooth_scale.empty() ? nullptr : preloaded->inv_smooth_scale.data(),
                    k,
                    col);
            if (collect_profile) {
                profile.activation_preprocess_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
                stage_start_us = ggml_time_us();
            }

            npu_decode_reset();
            if (collect_profile) {
                profile.reset_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
                stage_start_us = ggml_time_us();
            }

            try {
                npu_decode_gemv_pingpong_run(
                        act_cma,
                        pingpong_scale_ptr,
                        pingpong_weight_ptr,
                        out_cma,
                        static_cast<uint16_t>(out_channels),
                        static_cast<uint16_t>(k));
            } catch (const std::exception & e) {
                if (npu_decode_w4a16_debug_log_enabled()) {
                    GGML_LOG_WARN("%s: pingpong failed for %s: %s\n",
                            __func__, op_name != nullptr ? op_name : "(unnamed)", e.what());
                }
                pingpong_ok = false;
                break;
            }
            if (collect_profile) {
                profile.pingpong_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
                profile.pingpong_calls += 1;
                profile.output_bytes += preloaded->pingpong_layout.output_bytes;
                profile.weight_bytes += preloaded->pingpong_layout.weight_bytes;
                stage_start_us = ggml_time_us();
            }

            const ggml_fp16_t * out_fp16 = static_cast<const ggml_fp16_t *>(out_cma);
            char * dst_col = reinterpret_cast<char *>(dst_data) + col * dst_nb1;
            if (dst_type == GGML_TYPE_F16) {
                std::memcpy(dst_col, out_fp16, static_cast<size_t>(out_channels) * sizeof(ggml_fp16_t));
            } else {
                npu_decode_convert_output_fp16_to_f32(
                        out_fp16,
                        reinterpret_cast<float *>(dst_col),
                        static_cast<uint32_t>(out_channels));
            }
            if (collect_profile) {
                profile.postprocess_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            }
        }

        if (pingpong_ok) {
            const int64_t cleanup_start_us = collect_profile ? ggml_time_us() : 0;
            npu_decode_mem_free(act_cma);
            npu_decode_mem_free(packed_cma);
            npu_decode_mem_free(out_cma);
            npu_decode_mem_free(pingpong_scale_cma);
            npu_decode_mem_free(pingpong_weight_cma);
            if (collect_profile) {
                profile.cleanup_us = static_cast<uint64_t>(ggml_time_us() - cleanup_start_us);
                profile.total_us = static_cast<uint64_t>(ggml_time_us() - total_start_us);
                npu_decode_profile_write(profile);
            }
            return true;
        }

        npu_decode_mem_free(pingpong_scale_cma);
        npu_decode_mem_free(pingpong_weight_cma);
        pingpong_scale_cma = nullptr;
        pingpong_weight_cma = nullptr;
    }

    for (int64_t col = 0; col < n_cols; ++col) {
        int64_t stage_start_us = collect_profile ? ggml_time_us() : 0;
        npu_decode_pack_activation(
                act_cma,
                first_layout.act_bytes,
                first_layout,
                act_data,
                act_type,
                act_nb0,
                act_nb1,
                preloaded->inv_smooth_scale.empty() ? nullptr : preloaded->inv_smooth_scale.data(),
                k,
                col);
        if (collect_profile) {
            profile.activation_preprocess_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            stage_start_us = ggml_time_us();
        }

        npu_decode_reset();
        if (collect_profile) {
            profile.reset_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            stage_start_us = ggml_time_us();
        }
        npu_decode_dma_mvin(
                act_cma,
                0,
                first_layout.act_bytes - 1u,
                0,
                0,
                0,
                2,
                NPU_DECODE_GEMV_INPUT_TYPE_ACT,
                false,
                false,
                false,
                0,
                0,
                0);
        if (collect_profile) {
            profile.mvin_activation_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            profile.activation_mvin_calls += 1;
        }

        size_t block_index = 0;
        for (const npu_decode_preloaded_block & block : preloaded->blocks) {
            const int64_t row_base = block.row_base;
            const uint32_t block_rows = block.block_rows;
            const npu_decode_gemv_layout & layout = block.layout;
            if (layout.packed_bytes > first_layout.packed_bytes || layout.output_bytes > first_layout.output_bytes) {
                npu_decode_mem_free(act_cma);
                npu_decode_mem_free(packed_cma);
                npu_decode_mem_free(out_cma);
                return fail("block buffer exceeds allocation");
            }

            void * weight_mvin_ptr = packed_cma;
            if (preloaded_cma &&
                    block_index < preloaded->blocks.size() &&
                    block.cma_bytes == layout.packed_bytes) {
                weight_mvin_ptr = static_cast<uint8_t *>(decode_cma_base) + block.cma_offset;
            } else if (!preloaded_cma &&
                    preloaded != nullptr &&
                    block_index < preloaded->blocks.size() &&
                    block.packed.size() == layout.packed_bytes) {
                stage_start_us = collect_profile ? ggml_time_us() : 0;
                std::memcpy(packed_cma, block.packed.data(), layout.packed_bytes);
                if (collect_profile) {
                    profile.host_copy_weight_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
                }
            } else {
                npu_decode_mem_free(act_cma);
                npu_decode_mem_free(out_cma);
                return fail("preloaded CMA block mismatch");
            }
            if (collect_profile) {
                profile.weight_bytes += layout.packed_bytes;
                stage_start_us = ggml_time_us();
            }

            npu_decode_dma_mvin(
                    weight_mvin_ptr,
                    0,
                    layout.packed_bytes - 1u,
                    0,
                    0,
                    0,
                    1,
                    NPU_DECODE_GEMV_INPUT_TYPE_WEIGHT,
                    false,
                    false,
                    false,
                    0,
                    0,
                    0);
            if (collect_profile) {
                profile.mvin_weight_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
                profile.weight_mvin_calls += 1;
                stage_start_us = ggml_time_us();
            }

            npu_decode_gemv_block_run(
                    static_cast<uint16_t>(block_rows),
                    static_cast<uint16_t>(k),
                    static_cast<uint32_t>(row_base) * static_cast<uint32_t>(sizeof(ggml_fp16_t)));
            if (collect_profile) {
                profile.gemv_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
                profile.gemv_calls += 1;
                stage_start_us = ggml_time_us();
            }
            ++block_index;
        }

        stage_start_us = collect_profile ? ggml_time_us() : 0;
        npu_decode_dma_mvout(
                out_cma,
                0,
                0,
                static_cast<uint32_t>(out_channels - 1),
                1,
                1,
                1,
                1,
                false,
                false,
                0,
                0);
        if (collect_profile) {
            profile.mvout_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            profile.mvout_calls += 1;
            profile.output_bytes += total_output_bytes;
            stage_start_us = ggml_time_us();
        }

        const ggml_fp16_t * out_fp16 = static_cast<const ggml_fp16_t *>(out_cma);
        char * dst_col = reinterpret_cast<char *>(dst_data) + col * dst_nb1;
        if (dst_type == GGML_TYPE_F16) {
            std::memcpy(dst_col, out_fp16, static_cast<size_t>(out_channels) * sizeof(ggml_fp16_t));
        } else {
            npu_decode_convert_output_fp16_to_f32(
                    out_fp16,
                    reinterpret_cast<float *>(dst_col),
                    static_cast<uint32_t>(out_channels));
        }
        if (collect_profile) {
            profile.postprocess_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
        }
    }

    const int64_t cleanup_start_us = collect_profile ? ggml_time_us() : 0;
    npu_decode_mem_free(act_cma);
    npu_decode_mem_free(packed_cma);
    npu_decode_mem_free(out_cma);
    npu_decode_mem_free(pingpong_scale_cma);
    npu_decode_mem_free(pingpong_weight_cma);
    if (collect_profile) {
        profile.cleanup_us = static_cast<uint64_t>(ggml_time_us() - cleanup_start_us);
        profile.total_us = static_cast<uint64_t>(ggml_time_us() - total_start_us);
        npu_decode_profile_write(profile);
    }
    return true;
}

bool ggml_backend_npu_decode_w4a16_gemv(
        const char * op_name,
        const void * q4_data,
        int64_t packed_k,
        int64_t out_channels,
        int64_t q4_nb1,
        const void * scale_data,
        int scale_type,
        int64_t scale_nb0,
        int64_t scale_nb1,
        const void * zero_data,
        int zero_type,
        int64_t zero_nb0,
        int64_t zero_nb1,
        const void * act_data,
        int act_type,
        int64_t act_nb0,
        int64_t act_nb1,
        const float * smooth_scale,
        int64_t k,
        int64_t n_cols,
        float * dst_data,
        int64_t dst_nb1) {
    return ggml_backend_npu_decode_w4a16_gemv_ex(
            op_name,
            q4_data,
            packed_k,
            out_channels,
            q4_nb1,
            scale_data,
            scale_type,
            scale_nb0,
            scale_nb1,
            zero_data,
            zero_type,
            zero_nb0,
            zero_nb1,
            act_data,
            act_type,
            act_nb0,
            act_nb1,
            smooth_scale,
            k,
            n_cols,
            dst_data,
            GGML_TYPE_F32,
            dst_nb1);
}

GGML_BACKEND_DL_IMPL(ggml_backend_npu_reg)
