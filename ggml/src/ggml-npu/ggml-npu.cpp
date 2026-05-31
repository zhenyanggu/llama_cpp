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
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <mutex>
#include <new>
#include <sstream>
#include <stdexcept>
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
extern "C" void npu_decode_gemv_pingpong_run(
        void * act_ptr,
        void * scale_ptr,
        void * weight_ptr,
        void * output_ptr,
        uint16_t m,
        uint16_t n);
extern "C" void npu_decode_dma_mvin_async(uint32_t dma_id, const MvinConfig * cfg);
extern "C" void npu_decode_dma_wait_mvin(uint32_t dma_mask);
extern "C" void npu_decode_matvec_run(
        uint32_t mat_addr,
        uint32_t vec_addr,
        uint16_t mat_width,
        uint16_t mat_height,
        uint16_t output_addr,
        uint16_t scale_addr);
extern "C" void npu_decode_matvec_mode_run(
        uint32_t mat_addr,
        uint32_t vec_addr,
        uint16_t mat_width,
        uint16_t mat_height,
        uint16_t output_addr,
        uint16_t scale_addr,
        uint8_t gemv_mode);
extern "C" void npu_decode_matvec_silu_run(
        uint32_t mat_addr,
        uint32_t vec_addr,
        uint16_t mat_width,
        uint16_t mat_height,
        uint16_t output_addr,
        uint16_t scale_addr);
extern "C" uint64_t npu_decode_flow_make(
        uint8_t src0,
        uint8_t src1,
        uint8_t unary_op,
        uint8_t binary_op,
        uint8_t reduce_op,
        uint8_t dst,
        uint8_t src_buffer_id,
        uint8_t dst_buffer_id,
        uint16_t elem_count,
        uint16_t position);
extern "C" void npu_decode_matvec_decode_flow_run(
        uint32_t mat_addr,
        uint32_t vec_addr,
        uint16_t mat_width,
        uint16_t mat_height,
        uint16_t output_addr,
        uint16_t scale_addr,
        uint8_t gemv_mode,
        uint64_t decode_flow);
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

static bool npu_decode_matvec_trace_enabled() {
    const char * value = std::getenv("AICAS_TEXT_DECODE_AWQ_MATVEC_TRACE");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

static void npu_decode_trace_matvec_flow(
        const char * tag,
        const char * op_name,
        uint32_t block_idx,
        uint32_t row_base,
        uint32_t rows,
        uint32_t k,
        uint8_t gemv_mode,
        uint32_t mat_addr,
        uint32_t vec_addr,
        uint32_t output_addr,
        uint32_t scale_addr,
        uint64_t flow) {
    if (!npu_decode_matvec_trace_enabled()) {
        return;
    }
    std::fprintf(
            stderr,
            "[NPU][MATVEC_TRACE] tag=%s op=%s block=%u row_base=%u rows=%u k=%u mode=%u"
            " mat=0x%05X vec=0x%05X out=0x%05X scale=0x%05X flow=0x%016" PRIx64 "\n",
            tag != nullptr ? tag : "(null)",
            op_name != nullptr ? op_name : "(unnamed)",
            static_cast<unsigned>(block_idx),
            static_cast<unsigned>(row_base),
            static_cast<unsigned>(rows),
            static_cast<unsigned>(k),
            static_cast<unsigned>(gemv_mode),
            static_cast<unsigned>(mat_addr),
            static_cast<unsigned>(vec_addr),
            static_cast<unsigned>(output_addr),
            static_cast<unsigned>(scale_addr),
            static_cast<uint64_t>(flow));
    std::fflush(stderr);
}

static bool npu_decode_cma_retry_enabled() {
    const char * value = std::getenv("AICAS_TEXT_DECODE_AWQ_CMA_RETRY");
    return value == nullptr || value[0] == '\0' || std::strcmp(value, "0") != 0;
}

static bool npu_decode_preload_cma_enabled() {
    const char * value = std::getenv("AICAS_TEXT_DECODE_AWQ_NPU_PRELOAD_CMA");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

static bool npu_decode_w16a16_cma_enabled() {
    const char * value = std::getenv("AICAS_TEXT_LM_HEAD_W16A16_CMA");
    return value == nullptr || value[0] == '\0' || std::strcmp(value, "0") != 0;
}

static bool npu_decode_w16a16_pingpong_enabled() {
    const char * value = std::getenv("AICAS_TEXT_LM_HEAD_W16A16_PINGPONG");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

static void npu_decode_clear_w16a16_weight_cache();
static void npu_decode_release_w16a16_runtime_cma();

static bool npu_decode_require_active_overlay() {
    const char * value = std::getenv("AICAS_TEXT_DECODE_AWQ_NPU_REQUIRE_ACTIVE");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

static bool npu_decode_hw_f32_mvout_enabled() {
    const char * value = std::getenv("AICAS_TEXT_DECODE_AWQ_HW_F32_MVOUT");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

static bool npu_decode_transient_scratch_enabled() {
    const char * value = std::getenv("AICAS_TEXT_DECODE_AWQ_TRANSIENT_SCRATCH");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

static bool npu_decode_zero_out_before_mvout_enabled() {
    const char * value = std::getenv("AICAS_TEXT_DECODE_AWQ_ZERO_OUT_BEFORE_MVOUT");
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
    npu_decode_release_w16a16_runtime_cma();
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

static int npu_decode_init_with_cma_retry(const char * op_name, bool release_prefill_runtime) {
    if (npu_decode_init() == 0) {
        return 0;
    }
    if (!npu_decode_cma_retry_enabled()) {
        return -1;
    }

    if (npu_decode_w4a16_debug_log_enabled()) {
        GGML_LOG_WARN("%s: decode runtime init failed for %s; releasing NPU runtime state and retrying\n",
                __func__, op_name != nullptr ? op_name : "(unnamed)");
    }
    if (release_prefill_runtime) {
        npu_clear_preloaded_weight_cache();
        npu_decode_release_w16a16_runtime_cma();
        npu_destroy();
    }
    npu_decode_destroy();
    return npu_decode_init();
}

constexpr uint32_t NPU_DECODE_GEMV_K_TILE = 128;
constexpr uint32_t NPU_DECODE_GEMV_M_TILE = 32;
constexpr uint32_t NPU_DECODE_GEMV_LINE_BYTES = 64;
constexpr uint32_t NPU_DECODE_GEMV_DMA_ALIGN = 256;
constexpr uint32_t NPU_DECODE_GEMV_ACT_BUFFER_BYTES = 256u * 64u;
constexpr uint32_t NPU_DECODE_GEMV_ACT_TILE_BYTES = NPU_DECODE_GEMV_K_TILE * 2;
constexpr uint32_t NPU_DECODE_GEMV_SCALE_TILE_BYTES = NPU_DECODE_GEMV_M_TILE * 2;
constexpr uint32_t NPU_DECODE_GEMV_WEIGHT_ROW_BYTES = (NPU_DECODE_GEMV_K_TILE * 4) / 8;
constexpr uint32_t NPU_DECODE_GEMV_WEIGHT_DATA_TILE_BYTES = NPU_DECODE_GEMV_M_TILE * NPU_DECODE_GEMV_WEIGHT_ROW_BYTES;
constexpr uint32_t NPU_DECODE_GEMV_WEIGHT_TILE_BYTES =
    NPU_DECODE_GEMV_SCALE_TILE_BYTES + NPU_DECODE_GEMV_WEIGHT_DATA_TILE_BYTES;
constexpr uint32_t NPU_DECODE_GEMV_W16_WEIGHT_ROW_BYTES = NPU_DECODE_GEMV_K_TILE * 2;
constexpr uint32_t NPU_DECODE_GEMV_W16_WEIGHT_TILE_BYTES =
    NPU_DECODE_GEMV_M_TILE * NPU_DECODE_GEMV_W16_WEIGHT_ROW_BYTES;
constexpr uint8_t NPU_DECODE_GEMV_INPUT_TYPE_DATA = 0;
constexpr uint8_t NPU_DECODE_GEMV_INPUT_TYPE_WEIGHT = 1;
constexpr uint8_t NPU_DECODE_GEMV_INPUT_TYPE_ACT = 3;
constexpr uint8_t NPU_DECODE_GEMV_MODE_W4A16 = 0;
constexpr uint8_t NPU_DECODE_GEMV_MODE_W8A16 = 1;
constexpr uint8_t NPU_DECODE_GEMV_MODE_W16A16 = 2;
constexpr uint8_t NPU_DECODE_SRC_GEMV_STREAM = 1;
constexpr uint8_t NPU_DECODE_SRC_STREAM_BUFFER = 2;
constexpr uint8_t NPU_DECODE_UNARY_BYPASS = 0;
constexpr uint8_t NPU_DECODE_BINARY_BYPASS = 0;
constexpr uint8_t NPU_DECODE_BINARY_FP16_MUL = 1;
constexpr uint8_t NPU_DECODE_REDUCE_BYPASS = 0;
constexpr uint8_t NPU_DECODE_DST_OUTPUT_SPM = 1;
constexpr uint8_t NPU_DECODE_DST_ACT_BUFFER = 2;
constexpr uint8_t NPU_DECODE_DST_STREAM_BUFFER = 3;
constexpr uint32_t NPU_DECODE_GEMV_SPM_BYTES = 512u * 1024u;
constexpr uint32_t NPU_DECODE_GEMV_SCALE_BASE = 0x00000;
constexpr uint32_t NPU_DECODE_GEMV_ACT_BASE = 0x00000;
constexpr uint32_t NPU_DECODE_GEMV_OUTPUT_BASE = 0x00000;
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
    bool pingpong_cma_reserve_attempted = false;
    bool zero_validated = false;
    bool prefer_block_cma = false;
    std::vector<float> inv_smooth_scale;
    std::vector<uint8_t> act_scale;
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
    bool block_cma_reserve_attempted = false;
};

struct npu_decode_w16a16_cached_weight {
    std::string weight_name;
    const void * weight_data = nullptr;
    int64_t k = 0;
    int64_t out_channels = 0;
    int64_t weight_nb0 = 0;
    int64_t weight_nb1 = 0;
    uint32_t row_tiles = 0;
    uint32_t col_tiles = 0;
    uint32_t max_block_rows = 0;
    uint32_t pingpong_max_block_rows = 0;
    uint32_t bytes_per_row_tile = 0;
    void * cma_packed = nullptr;
    uint32_t cma_packed_offset = 0;
    uint32_t cma_bytes = 0;
    bool cma_packed_owned = false;
    void * cma_scale_ones = nullptr;
    uint32_t cma_scale_ones_offset = 0;
    uint32_t cma_scale_ones_bytes = 0;
    bool cma_scale_ones_owned = false;
    std::vector<uint8_t> packed_weight;
    std::vector<uint8_t> scale_ones;
};

struct npu_decode_w16a16_cache {
    std::mutex mutex;
    npu_decode_w16a16_cached_weight entry;
};

struct npu_decode_gemv_profile_record {
    std::string profile_kind = "aicas_decode_w4a16_gemv";
    std::string gemv_mode = "w4a16";
    std::string op_role = "decode_gemv";
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
    uint64_t scale_bytes = 0;
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
    uint64_t scale_mvin_us = 0;
    uint64_t mvin_weight_us = 0;
    uint64_t gemv_us = 0;
    uint64_t pingpong_us = 0;
    uint64_t pingpong_initial_mvin_us = 0;
    uint64_t pingpong_prefetch_issue_us = 0;
    uint64_t pingpong_prefetch_window_us = 0;
    uint64_t pingpong_prefetch_wait_us = 0;
    uint64_t pingpong_gemv_us = 0;
    uint64_t pingpong_mvout_us = 0;
    uint64_t mvout_us = 0;
    uint64_t postprocess_us = 0;
    uint64_t cleanup_us = 0;
    uint64_t activation_mvin_calls = 0;
    uint64_t scale_mvin_calls = 0;
    uint64_t weight_mvin_calls = 0;
    uint64_t gemv_calls = 0;
    uint64_t pingpong_calls = 0;
    uint64_t pingpong_block_calls = 0;
    uint64_t pingpong_initial_mvin_calls = 0;
    uint64_t pingpong_async_mvin_calls = 0;
    uint64_t mvout_calls = 0;
    std::string fallback_reason;
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

static npu_decode_w16a16_cache & npu_decode_w16a16_weight_cache() {
    static npu_decode_w16a16_cache cache;
    return cache;
}

static void npu_decode_clear_w16a16_weight_cache() {
    npu_decode_w16a16_cache & cache = npu_decode_w16a16_weight_cache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    if (cache.entry.cma_packed != nullptr && cache.entry.cma_packed_owned) {
        npu_decode_mem_free(cache.entry.cma_packed);
    }
    if (cache.entry.cma_scale_ones != nullptr && cache.entry.cma_scale_ones_owned) {
        npu_decode_mem_free(cache.entry.cma_scale_ones);
    }
    cache.entry = {};
}

static void npu_decode_release_w16a16_runtime_cma() {
    npu_decode_w16a16_cache & cache = npu_decode_w16a16_weight_cache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    if (cache.entry.cma_packed != nullptr && cache.entry.cma_packed_owned) {
        npu_decode_mem_free(cache.entry.cma_packed);
    }
    if (cache.entry.cma_scale_ones != nullptr && cache.entry.cma_scale_ones_owned) {
        npu_decode_mem_free(cache.entry.cma_scale_ones);
    }
    cache.entry.cma_packed = nullptr;
    cache.entry.cma_packed_owned = false;
    cache.entry.cma_scale_ones = nullptr;
    cache.entry.cma_scale_ones_owned = false;
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

static size_t npu_decode_profile_flush_records() {
    const char * value = std::getenv("GGML_NPU_DECODE_PROFILE_FLUSH_RECORDS");
    if (value == nullptr || value[0] == '\0') {
        return 4096;
    }

    char * end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 0);
    if (end == value || parsed == 0) {
        return 4096;
    }
    return static_cast<size_t>(parsed);
}

static bool npu_decode_pingpong_enabled() {
    const char * value = std::getenv("AICAS_TEXT_DECODE_AWQ_PINGPONG");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

static bool npu_decode_fused_ffn_cma_enabled() {
    const char * value = std::getenv("AICAS_TEXT_DECODE_AWQ_FUSED_FFN_CMA");
    return value == nullptr || value[0] == '\0' || std::strcmp(value, "0") != 0;
}

static bool npu_decode_fused_ffn_pingpong_enabled() {
    const char * value = std::getenv("AICAS_TEXT_DECODE_AWQ_FUSED_FFN_PINGPONG");
    return npu_decode_pingpong_enabled() &&
           (value == nullptr || value[0] == '\0' || std::strcmp(value, "0") != 0);
}

static bool npu_decode_weight_name_is_ffn(const char * weight_name) {
    return weight_name != nullptr &&
           (std::strstr(weight_name, "ffn_gate") != nullptr ||
            std::strstr(weight_name, "ffn_up") != nullptr ||
            std::strstr(weight_name, "ffn_down") != nullptr);
}

static bool npu_decode_pingpong_block_mvout_enabled() {
    const char * value = std::getenv("AICAS_TEXT_DECODE_AWQ_BLOCK_MVOUT");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

static const char * npu_decode_repro_dump_dir() {
    const char * value = std::getenv("AICAS_TEXT_DECODE_AWQ_REPRO_DUMP_DIR");
    return value != nullptr && value[0] != '\0' ? value : nullptr;
}

static bool npu_decode_repro_dump_match(const char * op_name) {
    const char * dir = npu_decode_repro_dump_dir();
    if (dir == nullptr) {
        return false;
    }
    const char * filter = std::getenv("AICAS_TEXT_DECODE_AWQ_REPRO_DUMP_OP");
    if (filter == nullptr || filter[0] == '\0') {
        return true;
    }
    return op_name != nullptr && std::strstr(op_name, filter) != nullptr;
}

static int npu_decode_repro_dump_limit() {
    const char * value = std::getenv("AICAS_TEXT_DECODE_AWQ_REPRO_DUMP_LIMIT");
    if (value == nullptr || value[0] == '\0') {
        return 1;
    }
    char * end = nullptr;
    const long parsed = std::strtol(value, &end, 0);
    return end != value && parsed > 0 ? static_cast<int>(parsed) : 1;
}

static std::string npu_decode_repro_sanitize(const char * name) {
    std::string out = name != nullptr && name[0] != '\0' ? name : "unnamed";
    for (char & ch : out) {
        if (!std::isalnum(static_cast<unsigned char>(ch)) && ch != '-' && ch != '_') {
            ch = '_';
        }
    }
    if (out.size() > 96) {
        out.resize(96);
    }
    return out;
}

static bool npu_decode_write_binary_file(const std::string & path, const void * data, size_t bytes) {
    std::ofstream out(path, std::ios::binary);
    if (!out.good()) {
        return false;
    }
    out.write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(bytes));
    return out.good();
}

static nlohmann::ordered_json npu_decode_profile_to_json(const npu_decode_gemv_profile_record & record) {
    nlohmann::ordered_json payload;
    payload["profile_kind"] = record.profile_kind.empty() ? "aicas_decode_w4a16_gemv" : record.profile_kind;
    payload["gemv_mode"] = record.gemv_mode.empty() ? "w4a16" : record.gemv_mode;
    payload["op_role"] = record.op_role.empty() ? "decode_gemv" : record.op_role;
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
        {"scale_mvin", record.scale_bytes},
        {"weight_mvin", record.weight_bytes},
        {"output", record.output_bytes},
    };
    payload["calls"] = {
        {"activation_mvin", record.activation_mvin_calls},
        {"scale_mvin", record.scale_mvin_calls},
        {"weight_mvin", record.weight_mvin_calls},
        {"gemv", record.gemv_calls},
        {"pingpong", record.pingpong_calls},
        {"pingpong_blocks", record.pingpong_block_calls},
        {"pingpong_initial_mvin", record.pingpong_initial_mvin_calls},
        {"pingpong_async_mvin", record.pingpong_async_mvin_calls},
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
        {"scale_mvin", record.scale_mvin_us},
        {"mvin_weight", record.mvin_weight_us},
        {"gemv", record.gemv_us},
        {"pingpong", record.pingpong_us},
        {"pingpong_initial_mvin", record.pingpong_initial_mvin_us},
        {"pingpong_prefetch_issue", record.pingpong_prefetch_issue_us},
        {"pingpong_prefetch_window", record.pingpong_prefetch_window_us},
        {"pingpong_prefetch_wait", record.pingpong_prefetch_wait_us},
        {"pingpong_gemv", record.pingpong_gemv_us},
        {"pingpong_mvout", record.pingpong_mvout_us},
        {"mvout", record.mvout_us},
        {"postprocess", record.postprocess_us},
        {"cleanup", record.cleanup_us},
    };
    if (!record.fallback_reason.empty()) {
        payload["fallback_reason"] = record.fallback_reason;
    }

    return payload;
}

static void npu_decode_profile_append(
        const char * path,
        const std::vector<npu_decode_gemv_profile_record> & snapshot) {
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

    npu_decode_profile_append(path, snapshot);
}

static void npu_decode_profile_write(const npu_decode_gemv_profile_record & record) {
    npu_decode_gemv_profile_cache & cache = npu_decode_profile_cache();
    const char * path = npu_decode_profile_path();
    std::vector<npu_decode_gemv_profile_record> snapshot;
    {
        std::lock_guard<std::mutex> lock(cache.mutex);
        if (!cache.atexit_registered) {
            std::atexit(npu_decode_profile_flush);
            cache.atexit_registered = true;
        }
        cache.records.push_back(record);
        if (path != nullptr && cache.records.size() >= npu_decode_profile_flush_records()) {
            snapshot.swap(cache.records);
        }
    }
    npu_decode_profile_append(path, snapshot);
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

static bool npu_decode_get_transient_scratch(uint32_t * offset, uint32_t * size) {
    if (offset == nullptr || size == nullptr) {
        return false;
    }
    npu_decode_preload_cache & cache = npu_decode_get_preload_cache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    if (cache.transient_size == 0) {
        return false;
    }
    *offset = cache.transient_offset;
    *size = cache.transient_size;
    return true;
}

static bool npu_decode_reserve_preloaded_ffn_pingpong_cma_after_init(
        std::initializer_list<npu_decode_preloaded_tensor *> tensors) {
    using namespace ggml_npu;

    if (!npu_decode_preload_cma_enabled()) {
        return false;
    }

    void * cma_base = npu_decode_memory_base();
    const uint32_t cma_map_size = npu_decode_memory_size();
    if (cma_base == nullptr || cma_map_size == 0) {
        return false;
    }

    npu_decode_preload_cache & cache = npu_decode_get_preload_cache();
    std::lock_guard<std::mutex> lock(cache.mutex);

    uint32_t arena_bytes = 0;
    for (const npu_decode_preloaded_tensor * tensor : tensors) {
        if (tensor == nullptr || !tensor->pingpong_available ||
                tensor->pingpong_cma_reserve_attempted ||
                tensor->pingpong_weight_cma_bytes == tensor->pingpong_layout.weight_bytes) {
            continue;
        }
        arena_bytes = npu_decode_align_up_bytes(arena_bytes, NPU_DECODE_GEMV_DMA_ALIGN);
        if (tensor->pingpong_layout.scale_bytes > std::numeric_limits<uint32_t>::max() - arena_bytes) {
            return false;
        }
        arena_bytes += tensor->pingpong_layout.scale_bytes;
        arena_bytes = npu_decode_align_up_bytes(arena_bytes, NPU_DECODE_GEMV_DMA_ALIGN);
        if (tensor->pingpong_layout.weight_bytes > std::numeric_limits<uint32_t>::max() - arena_bytes) {
            return false;
        }
        arena_bytes += tensor->pingpong_layout.weight_bytes;
    }
    if (arena_bytes == 0) {
        return false;
    }

    void * arena = npu_decode_mem_alloc(arena_bytes);
    if (arena == nullptr) {
        for (npu_decode_preloaded_tensor * tensor : tensors) {
            if (tensor != nullptr) {
                tensor->pingpong_cma_reserve_attempted = true;
            }
        }
        return false;
    }
    const uintptr_t base_addr = reinterpret_cast<uintptr_t>(cma_base);
    const uintptr_t arena_addr = reinterpret_cast<uintptr_t>(arena);
    if (arena_addr < base_addr || arena_addr - base_addr > cma_map_size) {
        return false;
    }
    const uintptr_t arena_offset = arena_addr - base_addr;
    if (arena_offset > std::numeric_limits<uint32_t>::max() ||
            arena_offset + arena_bytes > cma_map_size) {
        return false;
    }

    uint32_t cursor = 0;
    for (npu_decode_preloaded_tensor * tensor : tensors) {
        if (tensor == nullptr || !tensor->pingpong_available ||
                tensor->pingpong_cma_reserve_attempted ||
                tensor->pingpong_weight_cma_bytes == tensor->pingpong_layout.weight_bytes) {
            continue;
        }
        npu_decode_preloaded_tensor & entry = *tensor;
        cursor = npu_decode_align_up_bytes(cursor, NPU_DECODE_GEMV_DMA_ALIGN);
        if (entry.pingpong_layout.scale_bytes > 0) {
            entry.pingpong_scale_cma_offset = static_cast<uint32_t>(arena_offset) + cursor;
            entry.pingpong_scale_cma_bytes = entry.pingpong_layout.scale_bytes;
            std::memcpy(static_cast<uint8_t *>(arena) + cursor, entry.pingpong_scale.data(), entry.pingpong_scale.size());
            cursor += entry.pingpong_layout.scale_bytes;
        }
        cursor = npu_decode_align_up_bytes(cursor, NPU_DECODE_GEMV_DMA_ALIGN);
        entry.pingpong_weight_cma_offset = static_cast<uint32_t>(arena_offset) + cursor;
        entry.pingpong_weight_cma_bytes = entry.pingpong_layout.weight_bytes;
        std::memcpy(static_cast<uint8_t *>(arena) + cursor, entry.pingpong_weight.data(), entry.pingpong_weight.size());
        cursor += entry.pingpong_layout.weight_bytes;
        entry.pingpong_cma_reserve_attempted = true;
    }
    return true;
}

static uint32_t npu_decode_ceil_div_u32(uint32_t value, uint32_t divisor) {
    return (value + divisor - 1u) / divisor;
}

static uint32_t npu_decode_align_up_u32(uint32_t value, uint32_t alignment) {
    return (value + alignment - 1u) & ~(alignment - 1u);
}

static uint32_t npu_decode_act_payload_bytes(const npu_decode_gemv_layout & layout) {
    return layout.act_bytes * 2u;
}

static uint8_t npu_decode_mvout_precision(bool hw_f32_mvout) {
    return hw_f32_mvout ? 3 : 1;
}

static uint32_t npu_decode_host_output_elem_bytes(bool hw_f32_mvout) {
    return hw_f32_mvout ? sizeof(float) : sizeof(ggml_fp16_t);
}

static uint32_t npu_decode_host_output_bytes(uint32_t rows, bool hw_f32_mvout) {
    return npu_decode_align_up_u32(rows * npu_decode_host_output_elem_bytes(hw_f32_mvout), NPU_DECODE_GEMV_LINE_BYTES);
}

static bool npu_decode_make_gemv_layout(uint32_t m, uint32_t k, npu_decode_gemv_layout * layout) {
    if (m == 0 || k == 0 || layout == nullptr) {
        return false;
    }

    layout->row_tiles = npu_decode_ceil_div_u32(m, NPU_DECODE_GEMV_M_TILE);
    layout->col_tiles = npu_decode_ceil_div_u32(k, NPU_DECODE_GEMV_K_TILE);
    layout->act_bytes = layout->col_tiles * NPU_DECODE_GEMV_ACT_TILE_BYTES;
    layout->scale_bytes = 0;
    layout->weight_base = 0;
    layout->weight_bytes = layout->row_tiles * layout->col_tiles * NPU_DECODE_GEMV_WEIGHT_TILE_BYTES;
    layout->packed_bytes = npu_decode_align_up_u32(layout->weight_base + layout->weight_bytes, NPU_DECODE_GEMV_DMA_ALIGN);
    layout->output_bytes = npu_decode_align_up_u32(m * 2, NPU_DECODE_GEMV_LINE_BYTES);

    return npu_decode_act_payload_bytes(*layout) <= NPU_DECODE_GEMV_ACT_BUFFER_BYTES &&
           layout->scale_bytes <= NPU_DECODE_GEMV_SPM_BYTES &&
           layout->weight_base + layout->weight_bytes <= NPU_DECODE_GEMV_SPM_BYTES &&
           layout->output_bytes <= NPU_DECODE_GEMV_SPM_BYTES &&
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
    layout->scale_bytes = 0;
    layout->weight_base = 0;
    layout->weight_bytes = layout->row_tiles * layout->col_tiles * NPU_DECODE_GEMV_WEIGHT_TILE_BYTES;
    layout->packed_bytes = 0;
    layout->output_bytes = m * sizeof(ggml_fp16_t);

    const uint32_t bytes_per_row_tile = layout->col_tiles * NPU_DECODE_GEMV_WEIGHT_TILE_BYTES;
    const uint32_t ping_capacity = NPU_DECODE_GEMV_PONG_WEIGHT_BASE - NPU_DECODE_GEMV_PING_WEIGHT_BASE;
    const uint32_t pong_capacity = NPU_DECODE_GEMV_SPM_BYTES - NPU_DECODE_GEMV_PONG_WEIGHT_BASE;
    const uint32_t weight_capacity = std::min(ping_capacity, pong_capacity);
    return npu_decode_act_payload_bytes(*layout) <= NPU_DECODE_GEMV_ACT_BUFFER_BYTES &&
           layout->scale_bytes <= NPU_DECODE_GEMV_PING_WEIGHT_BASE &&
           layout->output_bytes <= NPU_DECODE_GEMV_SPM_BYTES &&
           layout->output_bytes <= std::numeric_limits<uint16_t>::max() + 1u &&
           bytes_per_row_tile > 0 &&
           bytes_per_row_tile <= weight_capacity &&
           (layout->act_bytes % NPU_DECODE_GEMV_DMA_ALIGN) == 0 &&
           (layout->weight_bytes % NPU_DECODE_GEMV_DMA_ALIGN) == 0;
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

static int8_t npu_decode_q4_signed_at(const uint8_t * row, int64_t k_idx) {
    const uint8_t q = npu_decode_q4_unsigned_at(row, k_idx);
    return static_cast<int8_t>(q >= 8 ? static_cast<int>(q) - 16 : static_cast<int>(q));
}

static float npu_decode_read_packed_fp16(const uint8_t * ptr) {
    ggml_fp16_t value;
    std::memcpy(&value, ptr, sizeof(value));
    return ggml_fp16_to_fp32(value);
}

static void npu_decode_pack_act_scale(
        void * scale_dst,
        size_t scale_bytes,
        const npu_decode_gemv_layout & layout,
        const float * inv_smooth_scale,
        int64_t k) {
    std::memset(scale_dst, 0, scale_bytes);
    uint8_t * scale = static_cast<uint8_t *>(scale_dst);
    const ggml_fp16_t fp16_one = ggml_fp32_to_fp16(1.0f);
    for (uint32_t col_tile = 0; col_tile < layout.col_tiles; ++col_tile) {
        for (uint32_t lane = 0; lane < NPU_DECODE_GEMV_K_TILE; ++lane) {
            const int64_t in_idx = static_cast<int64_t>(col_tile) * NPU_DECODE_GEMV_K_TILE + lane;
            if (in_idx >= k) {
                continue;
            }
            ggml_fp16_t fp16_scale = fp16_one;
            if (inv_smooth_scale != nullptr) {
                fp16_scale = ggml_fp32_to_fp16(inv_smooth_scale[in_idx]);
            }
            std::memcpy(
                    scale + col_tile * NPU_DECODE_GEMV_ACT_TILE_BYTES + lane * sizeof(ggml_fp16_t),
                    &fp16_scale,
                    sizeof(fp16_scale));
        }
    }
}

static void npu_decode_pack_activation(
        void * act_dst,
        size_t act_payload_bytes,
        const npu_decode_gemv_layout & layout,
        const void * act_data,
        int act_type,
        int64_t act_nb0,
        int64_t act_nb1,
        const uint8_t * act_scale_packed,
        int64_t k,
        int64_t col) {
    std::memset(act_dst, 0, act_payload_bytes);
    uint8_t * act = static_cast<uint8_t *>(act_dst);
    if (act_type == GGML_TYPE_F16 && act_nb0 == static_cast<int64_t>(sizeof(ggml_fp16_t))) {
        const uint8_t * src_col = static_cast<const uint8_t *>(act_data) + col * act_nb1;
        for (uint32_t col_tile = 0; col_tile < layout.col_tiles; ++col_tile) {
            const int64_t in_idx = static_cast<int64_t>(col_tile) * NPU_DECODE_GEMV_K_TILE;
            if (in_idx >= k) {
                continue;
            }
            const size_t elems = static_cast<size_t>(
                    std::min<int64_t>(NPU_DECODE_GEMV_K_TILE, k - in_idx));
            std::memcpy(
                    act + col_tile * NPU_DECODE_GEMV_ACT_TILE_BYTES,
                    src_col + in_idx * act_nb0,
                    elems * sizeof(ggml_fp16_t));
        }
        if (act_scale_packed != nullptr) {
            std::memcpy(act + layout.act_bytes, act_scale_packed, layout.act_bytes);
        } else if (act_payload_bytes >= npu_decode_act_payload_bytes(layout)) {
            npu_decode_pack_act_scale(act + layout.act_bytes, layout.act_bytes, layout, nullptr, k);
        }
        return;
    }
    for (uint32_t col_tile = 0; col_tile < layout.col_tiles; ++col_tile) {
        for (uint32_t lane = 0; lane < NPU_DECODE_GEMV_K_TILE; ++lane) {
            const int64_t in_idx = static_cast<int64_t>(col_tile) * NPU_DECODE_GEMV_K_TILE + lane;
            if (in_idx >= k) {
                continue;
            }
            const float value = npu_decode_read_typed_f32(act_data, act_type, col * act_nb1 + in_idx * act_nb0);
            const ggml_fp16_t fp16_value = ggml_fp32_to_fp16(value);
            std::memcpy(act + col_tile * NPU_DECODE_GEMV_ACT_TILE_BYTES + lane * 2, &fp16_value, sizeof(fp16_value));
        }
    }
    if (act_scale_packed != nullptr) {
        std::memcpy(act + layout.act_bytes, act_scale_packed, layout.act_bytes);
    } else if (act_payload_bytes >= npu_decode_act_payload_bytes(layout)) {
        npu_decode_pack_act_scale(act + layout.act_bytes, layout.act_bytes, layout, nullptr, k);
    }
}

static void npu_decode_store_sim_output(
        float value,
        void * dst_data,
        int dst_type,
        int64_t dst_nb1,
        int64_t col,
        int64_t row) {
    char * dst_col = static_cast<char *>(dst_data) + col * dst_nb1;
    const ggml_fp16_t fp16_value = ggml_fp32_to_fp16(value);
    if (dst_type == GGML_TYPE_F16) {
        std::memcpy(dst_col + row * sizeof(ggml_fp16_t), &fp16_value, sizeof(fp16_value));
    } else {
        reinterpret_cast<float *>(dst_col)[row] = ggml_fp16_to_fp32(fp16_value);
    }
}

static void npu_decode_simulate_packed_block(
        const uint8_t * act_packed,
        const uint8_t * packed,
        const npu_decode_gemv_layout & layout,
        int64_t row_base,
        int64_t block_rows,
        int64_t k,
        void * dst_data,
        int dst_type,
        int64_t dst_nb1,
        int64_t col) {
    for (int64_t row = 0; row < block_rows; ++row) {
        float acc = 0.0f;
        const uint32_t row_tile = static_cast<uint32_t>(row) / NPU_DECODE_GEMV_M_TILE;
        const uint32_t row_in_tile = static_cast<uint32_t>(row) % NPU_DECODE_GEMV_M_TILE;
        for (uint32_t col_tile = 0; col_tile < layout.col_tiles; ++col_tile) {
            const uint32_t tile_index = row_tile * layout.col_tiles + col_tile;
            const uint8_t * scale_line =
                packed + static_cast<size_t>(tile_index) * NPU_DECODE_GEMV_WEIGHT_TILE_BYTES;
            const float scale =
                npu_decode_read_packed_fp16(scale_line + row_in_tile * sizeof(ggml_fp16_t));
            const uint8_t * weight_row =
                packed + layout.weight_base +
                static_cast<size_t>(tile_index) * NPU_DECODE_GEMV_WEIGHT_TILE_BYTES +
                NPU_DECODE_GEMV_SCALE_TILE_BYTES +
                row_in_tile * NPU_DECODE_GEMV_WEIGHT_ROW_BYTES;
            const uint8_t * act_tile =
                act_packed + static_cast<size_t>(col_tile) * NPU_DECODE_GEMV_ACT_TILE_BYTES;
            const uint8_t * act_scale_tile =
                act_packed + layout.act_bytes + static_cast<size_t>(col_tile) * NPU_DECODE_GEMV_ACT_TILE_BYTES;
            for (uint32_t lane = 0; lane < NPU_DECODE_GEMV_K_TILE; ++lane) {
                const int64_t in_idx = static_cast<int64_t>(col_tile) * NPU_DECODE_GEMV_K_TILE + lane;
                if (in_idx >= k) {
                    continue;
                }
                const float act = npu_decode_read_packed_fp16(act_tile + lane * sizeof(ggml_fp16_t));
                const float act_scale = npu_decode_read_packed_fp16(act_scale_tile + lane * sizeof(ggml_fp16_t));
                const float w = static_cast<float>(npu_decode_q4_signed_at(weight_row, lane)) * scale;
                acc += act * act_scale * w;
            }
        }
        npu_decode_store_sim_output(acc, dst_data, dst_type, dst_nb1, col, row_base + row);
    }
}

static void npu_decode_simulate_pingpong(
        const uint8_t * act_packed,
        const uint8_t * scale_packed,
        const uint8_t * weight_packed,
        const npu_decode_gemv_layout & layout,
        int64_t out_channels,
        int64_t k,
        void * dst_data,
        int dst_type,
        int64_t dst_nb1,
        int64_t col) {
    for (int64_t row = 0; row < out_channels; ++row) {
        float acc = 0.0f;
        const uint32_t row_tile = static_cast<uint32_t>(row) / NPU_DECODE_GEMV_M_TILE;
        const uint32_t row_in_tile = static_cast<uint32_t>(row) % NPU_DECODE_GEMV_M_TILE;
        for (uint32_t col_tile = 0; col_tile < layout.col_tiles; ++col_tile) {
            const uint32_t tile_index = row_tile * layout.col_tiles + col_tile;
            const uint8_t * scale_line =
                weight_packed + static_cast<size_t>(tile_index) * NPU_DECODE_GEMV_WEIGHT_TILE_BYTES;
            const float scale =
                npu_decode_read_packed_fp16(scale_line + row_in_tile * sizeof(ggml_fp16_t));
            const uint8_t * weight_row =
                weight_packed +
                static_cast<size_t>(tile_index) * NPU_DECODE_GEMV_WEIGHT_TILE_BYTES +
                NPU_DECODE_GEMV_SCALE_TILE_BYTES +
                row_in_tile * NPU_DECODE_GEMV_WEIGHT_ROW_BYTES;
            const uint8_t * act_tile =
                act_packed + static_cast<size_t>(col_tile) * NPU_DECODE_GEMV_ACT_TILE_BYTES;
            const uint8_t * act_scale_tile =
                act_packed + layout.act_bytes + static_cast<size_t>(col_tile) * NPU_DECODE_GEMV_ACT_TILE_BYTES;
            for (uint32_t lane = 0; lane < NPU_DECODE_GEMV_K_TILE; ++lane) {
                const int64_t in_idx = static_cast<int64_t>(col_tile) * NPU_DECODE_GEMV_K_TILE + lane;
                if (in_idx >= k) {
                    continue;
                }
                const float act = npu_decode_read_packed_fp16(act_tile + lane * sizeof(ggml_fp16_t));
                const float act_scale = npu_decode_read_packed_fp16(act_scale_tile + lane * sizeof(ggml_fp16_t));
                const float w = static_cast<float>(npu_decode_q4_signed_at(weight_row, lane)) * scale;
                acc += act * act_scale * w;
            }
        }
        npu_decode_store_sim_output(acc, dst_data, dst_type, dst_nb1, col, row);
    }
}

static MvinConfig npu_decode_make_mvin_cfg(
        void * host_ptr,
        uint32_t sram_addr,
        uint32_t bytes,
        uint8_t input_type,
        uint8_t precision) {
    MvinConfig cfg = {};
    cfg.host_ptr = host_ptr;
    cfg.sram_addr = sram_addr;
    cfg.col_num = bytes - 1u;
    cfg.row_num = 0;
    cfg.sram_stride = 0;
    cfg.dram_stride = 0;
    cfg.precision = precision;
    cfg.input_type = input_type;
    cfg.dest = false;
    cfg.is_bias = false;
    cfg.is_quant = false;
    cfg.quant_zero = 0;
    cfg.quant_scale = 0;
    cfg.quant_shift = 0;
    return cfg;
}

static uint64_t npu_decode_make_bypass_output_flow(uint16_t elem_count) {
    return npu_decode_flow_make(
            NPU_DECODE_SRC_GEMV_STREAM,
            0,
            NPU_DECODE_UNARY_BYPASS,
            NPU_DECODE_BINARY_BYPASS,
            NPU_DECODE_REDUCE_BYPASS,
            NPU_DECODE_DST_OUTPUT_SPM,
            0,
            0,
            elem_count,
            0);
}

static void npu_decode_scale_packed_row_scales(
        uint8_t * packed,
        const npu_decode_gemv_layout & layout,
        int64_t row_base,
        const std::vector<float> & row_scale) {
    for (uint32_t row_tile = 0; row_tile < layout.row_tiles; ++row_tile) {
        for (uint32_t col_tile = 0; col_tile < layout.col_tiles; ++col_tile) {
            const size_t scale_line =
                static_cast<size_t>(row_tile * layout.col_tiles + col_tile) *
                NPU_DECODE_GEMV_WEIGHT_TILE_BYTES;
            for (uint32_t lane = 0; lane < NPU_DECODE_GEMV_M_TILE; ++lane) {
                const int64_t row = static_cast<int64_t>(row_tile) * NPU_DECODE_GEMV_M_TILE + lane;
                const int64_t global_row = row_base + row;
                if (global_row < 0 || static_cast<size_t>(global_row) >= row_scale.size()) {
                    continue;
                }
                uint8_t * ptr = packed + scale_line + lane * sizeof(ggml_fp16_t);
                const float scale = npu_decode_read_packed_fp16(ptr) * row_scale[static_cast<size_t>(global_row)];
                const ggml_fp16_t scale_fp16 = ggml_fp32_to_fp16(scale);
                std::memcpy(ptr, &scale_fp16, sizeof(scale_fp16));
            }
        }
    }
}

static uint32_t npu_decode_pingpong_max_block_rows(const npu_decode_gemv_layout & layout) {
    const uint32_t bytes_per_row_tile = layout.col_tiles * NPU_DECODE_GEMV_WEIGHT_TILE_BYTES;
    const uint32_t ping_capacity = NPU_DECODE_GEMV_PONG_WEIGHT_BASE - NPU_DECODE_GEMV_PING_WEIGHT_BASE;
    const uint32_t pong_capacity = NPU_DECODE_GEMV_SPM_BYTES - NPU_DECODE_GEMV_PONG_WEIGHT_BASE;
    const uint32_t weight_capacity = std::min(ping_capacity, pong_capacity);
    const uint32_t max_row_tiles = bytes_per_row_tile == 0 ? 0 : weight_capacity / bytes_per_row_tile;
    return max_row_tiles * NPU_DECODE_GEMV_M_TILE;
}

static uint32_t npu_decode_w16a16_pingpong_max_block_rows(uint32_t col_tiles) {
    const uint32_t bytes_per_row_tile = col_tiles * NPU_DECODE_GEMV_W16_WEIGHT_TILE_BYTES;
    const uint32_t ping_capacity = NPU_DECODE_GEMV_PONG_WEIGHT_BASE - NPU_DECODE_GEMV_PING_WEIGHT_BASE;
    const uint32_t pong_capacity = NPU_DECODE_GEMV_SPM_BYTES - NPU_DECODE_GEMV_PONG_WEIGHT_BASE;
    const uint32_t weight_capacity = std::min(ping_capacity, pong_capacity);
    const uint32_t max_row_tiles = bytes_per_row_tile == 0 ? 0 : weight_capacity / bytes_per_row_tile;
    return max_row_tiles * NPU_DECODE_GEMV_M_TILE;
}

static void npu_decode_gemv_pingpong_manual_run(
        const char * op_name,
        void * act_ptr,
        void * scale_ptr,
        void * weight_ptr,
        void * output_ptr,
        const npu_decode_gemv_layout & layout,
        uint16_t m,
        uint16_t k,
        uint8_t output_precision,
        npu_decode_gemv_profile_record * profile) {
    const uint32_t max_block_rows = npu_decode_pingpong_max_block_rows(layout);
    if (max_block_rows == 0) {
        throw std::runtime_error("manual GEMV pingpong: no row block fits");
    }
    if (output_precision != 1 && output_precision != 3) {
        throw std::runtime_error("manual GEMV pingpong: output precision must be FP16 or FP32");
    }
    const bool block_mvout = npu_decode_pingpong_block_mvout_enabled();
    const uint32_t output_elem_bytes = output_precision == 3 ? sizeof(float) : sizeof(ggml_fp16_t);
    if (npu_decode_w4a16_debug_log_enabled()) {
        GGML_LOG_WARN("%s: manual pingpong M=%u K=%u blocks<=%u block_mvout=%d out_precision=%u"
                " act=0x%05X scale=0x%05X out=0x%05X pingW=0x%05X pongW=0x%05X\n",
                __func__,
                static_cast<unsigned>(m),
                static_cast<unsigned>(k),
                static_cast<unsigned>((static_cast<uint32_t>(m) + max_block_rows - 1u) / max_block_rows),
                block_mvout ? 1 : 0,
                static_cast<unsigned>(output_precision),
                NPU_DECODE_GEMV_ACT_BASE,
                NPU_DECODE_GEMV_SCALE_BASE,
                NPU_DECODE_GEMV_OUTPUT_BASE,
                NPU_DECODE_GEMV_PING_WEIGHT_BASE,
                NPU_DECODE_GEMV_PONG_WEIGHT_BASE);
    }
    const uint32_t bytes_per_row_tile = layout.col_tiles * NPU_DECODE_GEMV_WEIGHT_TILE_BYTES;
    auto block_rows_for = [&](uint32_t row_base) {
        return std::min<uint32_t>(max_block_rows, static_cast<uint32_t>(m) - row_base);
    };
    auto weight_offset_for = [&](uint32_t row_base) {
        return (row_base / NPU_DECODE_GEMV_M_TILE) * bytes_per_row_tile;
    };
    auto scale_addr_for = [&](uint32_t row_base) {
        GGML_UNUSED(row_base);
        if (layout.scale_bytes == 0) {
            return 0u;
        }
        return NPU_DECODE_GEMV_SCALE_BASE +
               (row_base / NPU_DECODE_GEMV_M_TILE) * layout.col_tiles * NPU_DECODE_GEMV_SCALE_TILE_BYTES;
    };

    npu_decode_dma_mvin(
            act_ptr,
            NPU_DECODE_GEMV_ACT_BASE,
            npu_decode_act_payload_bytes(layout) - 1u,
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
    if (layout.scale_bytes > 0) {
        npu_decode_dma_mvin(
                scale_ptr,
                NPU_DECODE_GEMV_SCALE_BASE,
                layout.scale_bytes - 1u,
                0,
                0,
                0,
                2,
                NPU_DECODE_GEMV_INPUT_TYPE_DATA,
                false,
                false,
                false,
                0,
                0,
                0);
    }

    uint8_t * weight_base = static_cast<uint8_t *>(weight_ptr);
    const uint32_t first_rows = block_rows_for(0);
    const uint32_t first_weight_bytes =
        npu_decode_ceil_div_u32(first_rows, NPU_DECODE_GEMV_M_TILE) * bytes_per_row_tile;
    int64_t stage_start_us = profile != nullptr ? ggml_time_us() : 0;
    npu_decode_dma_mvin(
            weight_base,
            NPU_DECODE_GEMV_PING_WEIGHT_BASE,
            first_weight_bytes - 1u,
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
    if (profile != nullptr) {
        profile->pingpong_initial_mvin_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
        profile->pingpong_initial_mvin_calls += 1;
        profile->weight_bytes += first_weight_bytes;
    }

    uint32_t block_idx = 0;
    for (uint32_t row_base = 0; row_base < static_cast<uint32_t>(m);
            row_base += block_rows_for(row_base), ++block_idx) {
        const uint32_t rows = block_rows_for(row_base);
        const uint32_t cur_weight_spm =
            (block_idx & 1u) ? NPU_DECODE_GEMV_PONG_WEIGHT_BASE : NPU_DECODE_GEMV_PING_WEIGHT_BASE;

        bool prefetch_started = false;
        int64_t prefetch_issue_done_us = 0;
        const uint32_t next_row_base = row_base + rows;
        if (next_row_base < static_cast<uint32_t>(m)) {
            const uint32_t next_rows = block_rows_for(next_row_base);
            const uint32_t next_weight_bytes =
                npu_decode_ceil_div_u32(next_rows, NPU_DECODE_GEMV_M_TILE) * bytes_per_row_tile;
            const uint32_t next_weight_spm =
                ((block_idx + 1u) & 1u) ? NPU_DECODE_GEMV_PONG_WEIGHT_BASE : NPU_DECODE_GEMV_PING_WEIGHT_BASE;
            MvinConfig next_mvin = npu_decode_make_mvin_cfg(
                    weight_base + weight_offset_for(next_row_base),
                    next_weight_spm,
                    next_weight_bytes,
                    NPU_DECODE_GEMV_INPUT_TYPE_WEIGHT,
                    1);
            stage_start_us = profile != nullptr ? ggml_time_us() : 0;
            npu_decode_dma_mvin_async(0, &next_mvin);
            if (profile != nullptr) {
                prefetch_issue_done_us = ggml_time_us();
                profile->pingpong_prefetch_issue_us += static_cast<uint64_t>(prefetch_issue_done_us - stage_start_us);
                profile->pingpong_async_mvin_calls += 1;
                profile->weight_bytes += next_weight_bytes;
            }
            prefetch_started = true;
        }

        const uint16_t output_addr = static_cast<uint16_t>(
                NPU_DECODE_GEMV_OUTPUT_BASE + (block_mvout ? 0u : row_base * sizeof(ggml_fp16_t)));
        const uint16_t scale_addr = static_cast<uint16_t>(
                layout.scale_bytes == 0 ? cur_weight_spm : scale_addr_for(row_base));
        if (npu_decode_w4a16_debug_log_enabled()) {
            GGML_LOG_WARN("%s: block=%u row_base=%u rows=%u W=0x%05X Wbytes=%u"
                    " O=0x%04X S=0x%04X K=%u prefetch=%d\n",
                    __func__,
                    static_cast<unsigned>(block_idx),
                    static_cast<unsigned>(row_base),
                    static_cast<unsigned>(rows),
                    static_cast<unsigned>(cur_weight_spm),
                    static_cast<unsigned>(npu_decode_ceil_div_u32(rows, NPU_DECODE_GEMV_M_TILE) * bytes_per_row_tile),
                    static_cast<unsigned>(output_addr),
                    static_cast<unsigned>(scale_addr),
                    static_cast<unsigned>(k),
                    prefetch_started ? 1 : 0);
        }

        const uint64_t flow = npu_decode_make_bypass_output_flow(static_cast<uint16_t>(rows));
        npu_decode_trace_matvec_flow(
                "manual_pingpong",
                op_name,
                block_idx,
                row_base,
                rows,
                k,
                NPU_DECODE_GEMV_MODE_W4A16,
                cur_weight_spm,
                NPU_DECODE_GEMV_ACT_BASE,
                output_addr,
                scale_addr,
                flow);
        stage_start_us = profile != nullptr ? ggml_time_us() : 0;
        npu_decode_matvec_decode_flow_run(
                cur_weight_spm,
                NPU_DECODE_GEMV_ACT_BASE,
                k,
                static_cast<uint16_t>(rows),
                output_addr,
                scale_addr,
                NPU_DECODE_GEMV_MODE_W4A16,
                flow);
        if (profile != nullptr) {
            profile->pingpong_gemv_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            profile->pingpong_block_calls += 1;
        }

        if (prefetch_started) {
            if (profile != nullptr) {
                profile->pingpong_prefetch_window_us +=
                    static_cast<uint64_t>(ggml_time_us() - prefetch_issue_done_us);
                stage_start_us = ggml_time_us();
            }
            npu_decode_dma_wait_mvin(1u << 0);
            if (profile != nullptr) {
                profile->pingpong_prefetch_wait_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            }
        }

        if (block_mvout) {
            stage_start_us = profile != nullptr ? ggml_time_us() : 0;
            npu_decode_dma_mvout(
                    static_cast<uint8_t *>(output_ptr) + row_base * output_elem_bytes,
                    NPU_DECODE_GEMV_OUTPUT_BASE,
                    0,
                    rows - 1u,
                    1,
                    1,
                    output_precision,
                    1,
                    false,
                    false,
                    0,
                    0);
            if (profile != nullptr) {
                profile->pingpong_mvout_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
                profile->mvout_calls += 1;
            }
        }
    }

    if (!block_mvout) {
        stage_start_us = profile != nullptr ? ggml_time_us() : 0;
        npu_decode_dma_mvout(
                output_ptr,
                NPU_DECODE_GEMV_OUTPUT_BASE,
                0,
                static_cast<uint32_t>(m) - 1u,
                1,
                1,
                output_precision,
                1,
                false,
                false,
                0,
                0);
        if (profile != nullptr) {
            profile->pingpong_mvout_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            profile->mvout_calls += 1;
        }
    }
}

static void npu_decode_w16a16_pingpong_manual_run(
        const char * op_name,
        void * act_ptr,
        void * scale_ptr,
        void * weight_ptr,
        void * output_ptr,
        uint32_t col_tiles,
        uint32_t bytes_per_row_tile,
        uint32_t max_block_rows,
        uint16_t m,
        uint16_t k,
        uint8_t output_precision,
        npu_decode_gemv_profile_record * profile) {
    if (max_block_rows == 0) {
        throw std::runtime_error("manual W16A16 pingpong: no row block fits");
    }
    if (output_precision != 1 && output_precision != 3) {
        throw std::runtime_error("manual W16A16 pingpong: output precision must be FP16 or FP32");
    }
    const uint32_t act_bytes = col_tiles * NPU_DECODE_GEMV_ACT_TILE_BYTES;
    const uint32_t scale_bytes = npu_decode_ceil_div_u32(max_block_rows, NPU_DECODE_GEMV_M_TILE) *
        col_tiles * NPU_DECODE_GEMV_SCALE_TILE_BYTES;
    const uint32_t output_elem_bytes = output_precision == 3 ? sizeof(float) : sizeof(ggml_fp16_t);
    auto block_rows_for = [&](uint32_t row_base) {
        return std::min<uint32_t>(max_block_rows, static_cast<uint32_t>(m) - row_base);
    };
    auto weight_offset_for = [&](uint32_t row_base) {
        return (row_base / NPU_DECODE_GEMV_M_TILE) * bytes_per_row_tile;
    };

    if (npu_decode_w4a16_debug_log_enabled()) {
        GGML_LOG_WARN("%s: manual W16 pingpong M=%u K=%u blocks<=%u out_precision=%u"
                " act=0x%05X scale=0x%05X out=0x%05X pingW=0x%05X pongW=0x%05X\n",
                __func__,
                static_cast<unsigned>(m),
                static_cast<unsigned>(k),
                static_cast<unsigned>((static_cast<uint32_t>(m) + max_block_rows - 1u) / max_block_rows),
                static_cast<unsigned>(output_precision),
                NPU_DECODE_GEMV_ACT_BASE,
                NPU_DECODE_GEMV_SCALE_BASE,
                NPU_DECODE_GEMV_OUTPUT_BASE,
                NPU_DECODE_GEMV_PING_WEIGHT_BASE,
                NPU_DECODE_GEMV_PONG_WEIGHT_BASE);
    }

    npu_decode_dma_mvin(
            act_ptr,
            NPU_DECODE_GEMV_ACT_BASE,
            act_bytes - 1u,
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
    npu_decode_dma_mvin(
            scale_ptr,
            NPU_DECODE_GEMV_SCALE_BASE,
            scale_bytes - 1u,
            0,
            0,
            0,
            2,
            NPU_DECODE_GEMV_INPUT_TYPE_DATA,
            false,
            false,
            false,
            0,
            0,
            0);

    uint8_t * weight_base = static_cast<uint8_t *>(weight_ptr);
    const uint32_t first_rows = block_rows_for(0);
    const uint32_t first_weight_bytes =
        npu_decode_ceil_div_u32(first_rows, NPU_DECODE_GEMV_M_TILE) * bytes_per_row_tile;
    int64_t stage_start_us = profile != nullptr ? ggml_time_us() : 0;
    npu_decode_dma_mvin(
            weight_base,
            NPU_DECODE_GEMV_PING_WEIGHT_BASE,
            first_weight_bytes - 1u,
            0,
            0,
            0,
            2,
            NPU_DECODE_GEMV_INPUT_TYPE_WEIGHT,
            false,
            false,
            false,
            0,
            0,
            0);
    if (profile != nullptr) {
        profile->pingpong_initial_mvin_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
        profile->pingpong_initial_mvin_calls += 1;
        profile->weight_bytes += first_weight_bytes;
    }

    uint32_t block_idx = 0;
    for (uint32_t row_base = 0; row_base < static_cast<uint32_t>(m);
            row_base += block_rows_for(row_base), ++block_idx) {
        const uint32_t rows = block_rows_for(row_base);
        const uint32_t cur_weight_spm =
            (block_idx & 1u) ? NPU_DECODE_GEMV_PONG_WEIGHT_BASE : NPU_DECODE_GEMV_PING_WEIGHT_BASE;

        bool prefetch_started = false;
        int64_t prefetch_issue_done_us = 0;
        const uint32_t next_row_base = row_base + rows;
        if (next_row_base < static_cast<uint32_t>(m)) {
            const uint32_t next_rows = block_rows_for(next_row_base);
            const uint32_t next_weight_bytes =
                npu_decode_ceil_div_u32(next_rows, NPU_DECODE_GEMV_M_TILE) * bytes_per_row_tile;
            const uint32_t next_weight_spm =
                ((block_idx + 1u) & 1u) ? NPU_DECODE_GEMV_PONG_WEIGHT_BASE : NPU_DECODE_GEMV_PING_WEIGHT_BASE;
            MvinConfig next_mvin = npu_decode_make_mvin_cfg(
                    weight_base + weight_offset_for(next_row_base),
                    next_weight_spm,
                    next_weight_bytes,
                    NPU_DECODE_GEMV_INPUT_TYPE_WEIGHT,
                    2);
            stage_start_us = profile != nullptr ? ggml_time_us() : 0;
            npu_decode_dma_mvin_async(0, &next_mvin);
            if (profile != nullptr) {
                prefetch_issue_done_us = ggml_time_us();
                profile->pingpong_prefetch_issue_us += static_cast<uint64_t>(prefetch_issue_done_us - stage_start_us);
                profile->pingpong_async_mvin_calls += 1;
                profile->weight_bytes += next_weight_bytes;
            }
            prefetch_started = true;
        }

        if (npu_decode_w4a16_debug_log_enabled() && (block_idx == 0 || row_base + rows >= static_cast<uint32_t>(m))) {
            GGML_LOG_WARN("%s: block %s idx=%u row_base=%u rows=%u W=0x%05X Wbytes=%u K=%u prefetch=%d\n",
                    __func__,
                    op_name != nullptr ? op_name : "(unnamed)",
                    block_idx,
                    row_base,
                    rows,
                    cur_weight_spm,
                    npu_decode_ceil_div_u32(rows, NPU_DECODE_GEMV_M_TILE) * bytes_per_row_tile,
                    k,
                    prefetch_started ? 1 : 0);
        }

        const uint64_t flow = npu_decode_make_bypass_output_flow(static_cast<uint16_t>(rows));
        npu_decode_trace_matvec_flow(
                "lm_head_w16_pingpong",
                op_name,
                block_idx,
                row_base,
                rows,
                k,
                NPU_DECODE_GEMV_MODE_W16A16,
                cur_weight_spm,
                NPU_DECODE_GEMV_ACT_BASE,
                NPU_DECODE_GEMV_OUTPUT_BASE,
                NPU_DECODE_GEMV_SCALE_BASE,
                flow);
        stage_start_us = profile != nullptr ? ggml_time_us() : 0;
        npu_decode_matvec_decode_flow_run(
                cur_weight_spm,
                NPU_DECODE_GEMV_ACT_BASE,
                k,
                static_cast<uint16_t>(rows),
                NPU_DECODE_GEMV_OUTPUT_BASE,
                NPU_DECODE_GEMV_SCALE_BASE,
                NPU_DECODE_GEMV_MODE_W16A16,
                flow);
        if (profile != nullptr) {
            profile->pingpong_gemv_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            profile->pingpong_block_calls += 1;
        }

        if (prefetch_started) {
            if (profile != nullptr) {
                profile->pingpong_prefetch_window_us +=
                    static_cast<uint64_t>(ggml_time_us() - prefetch_issue_done_us);
                stage_start_us = ggml_time_us();
            }
            npu_decode_dma_wait_mvin(1u << 0);
            if (profile != nullptr) {
                profile->pingpong_prefetch_wait_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            }
        }

        stage_start_us = profile != nullptr ? ggml_time_us() : 0;
        npu_decode_dma_mvout(
                static_cast<uint8_t *>(output_ptr) + static_cast<size_t>(row_base) * output_elem_bytes,
                NPU_DECODE_GEMV_OUTPUT_BASE,
                0,
                rows - 1u,
                1,
                1,
                output_precision,
                1,
                false,
                false,
                0,
                0);
        if (profile != nullptr) {
            profile->pingpong_mvout_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            profile->mvout_calls += 1;
            profile->output_bytes += npu_decode_host_output_bytes(rows, output_precision == 3);
        }
    }
}

static void npu_decode_convert_output_fp16_to_f32(
        const ggml_fp16_t * src,
        float * dst,
        uint32_t rows) {
    ggml_fp16_to_fp32_row(src, dst, rows);
}

static void npu_decode_repro_dump_pingpong(
        const char * op_name,
        int64_t col,
        const npu_decode_gemv_layout & layout,
        uint16_t m,
        uint16_t k,
        uint8_t output_precision,
        bool block_mvout,
        const void * act_ptr,
        const void * scale_ptr,
        const void * weight_ptr,
        const void * output_ptr,
        uint32_t host_output_bytes,
        const void * cma_base,
        uint32_t cma_size) {
    if (!npu_decode_repro_dump_match(op_name)) {
        return;
    }

    static std::mutex mutex;
    static int dumped = 0;
    std::lock_guard<std::mutex> lock(mutex);
    if (dumped >= npu_decode_repro_dump_limit()) {
        return;
    }

    const char * dump_dir = npu_decode_repro_dump_dir();
    if (dump_dir == nullptr) {
        return;
    }
    const int dump_idx = dumped++;
    std::ostringstream prefix;
    prefix << dump_dir << "/repro_" << dump_idx << "_" << npu_decode_repro_sanitize(op_name);

    const std::string base = prefix.str();
    const std::string meta_path = base + ".meta.txt";
    const std::string act_path = base + ".act.bin";
    const std::string scale_path = base + ".scale.bin";
    const std::string weight_path = base + ".weight.bin";
    const std::string output_path = base + ".output.bin";

    const uint32_t act_payload_bytes = npu_decode_act_payload_bytes(layout);
    auto cma_offset_of = [cma_base, cma_size](const void * ptr) -> uint64_t {
        if (cma_base == nullptr || ptr == nullptr || cma_size == 0) {
            return UINT64_MAX;
        }
        const uintptr_t base = reinterpret_cast<uintptr_t>(cma_base);
        const uintptr_t cur = reinterpret_cast<uintptr_t>(ptr);
        if (cur < base || cur - base >= static_cast<uintptr_t>(cma_size)) {
            return UINT64_MAX;
        }
        return static_cast<uint64_t>(cur - base);
    };
    std::ofstream meta(meta_path);
    if (!meta.good()) {
        GGML_LOG_WARN("%s: failed to open repro meta path %s\n", __func__, meta_path.c_str());
        return;
    }
    meta << "op_name=" << (op_name != nullptr ? op_name : "") << "\n";
    meta << "col=" << col << "\n";
    meta << "m=" << m << "\n";
    meta << "k=" << k << "\n";
    meta << "row_tiles=" << layout.row_tiles << "\n";
    meta << "col_tiles=" << layout.col_tiles << "\n";
    meta << "act_bytes=" << layout.act_bytes << "\n";
    meta << "act_payload_bytes=" << act_payload_bytes << "\n";
    meta << "scale_bytes=" << layout.scale_bytes << "\n";
    meta << "weight_bytes=" << layout.weight_bytes << "\n";
    meta << "output_bytes=" << layout.output_bytes << "\n";
    meta << "host_output_bytes=" << host_output_bytes << "\n";
    meta << "max_block_rows=" << npu_decode_pingpong_max_block_rows(layout) << "\n";
    meta << "output_precision=" << static_cast<unsigned>(output_precision) << "\n";
    meta << "block_mvout=" << (block_mvout ? 1 : 0) << "\n";
    meta << "cma_size=" << cma_size << "\n";
    meta << "act_cma_offset=" << cma_offset_of(act_ptr) << "\n";
    meta << "scale_cma_offset=" << cma_offset_of(scale_ptr) << "\n";
    meta << "weight_cma_offset=" << cma_offset_of(weight_ptr) << "\n";
    meta << "output_cma_offset=" << cma_offset_of(output_ptr) << "\n";
    meta << "act_file=" << act_path << "\n";
    meta << "scale_file=" << scale_path << "\n";
    meta << "weight_file=" << weight_path << "\n";
    meta << "output_file=" << output_path << "\n";
    meta.close();

    const bool ok =
        npu_decode_write_binary_file(act_path, act_ptr, act_payload_bytes) &&
        npu_decode_write_binary_file(scale_path, scale_ptr, layout.scale_bytes) &&
        npu_decode_write_binary_file(weight_path, weight_ptr, layout.weight_bytes) &&
        npu_decode_write_binary_file(output_path, output_ptr, host_output_bytes);
    if (!ok) {
        GGML_LOG_WARN("%s: failed to write one or more repro dump files under %s\n",
                __func__, dump_dir);
    } else if (npu_decode_w4a16_debug_log_enabled()) {
        GGML_LOG_WARN("%s: wrote repro dump %s.*\n", __func__, base.c_str());
    }
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
            const size_t scale_line =
                static_cast<size_t>(row_tile * layout.col_tiles + col_tile) *
                NPU_DECODE_GEMV_WEIGHT_TILE_BYTES;
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
                    static_cast<size_t>(tile_index) * NPU_DECODE_GEMV_WEIGHT_TILE_BYTES +
                    NPU_DECODE_GEMV_SCALE_TILE_BYTES +
                    row_in_tile * NPU_DECODE_GEMV_WEIGHT_ROW_BYTES;
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
                static_cast<size_t>(row_tile * layout.col_tiles + col_tile) *
                NPU_DECODE_GEMV_WEIGHT_TILE_BYTES;
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
                std::memcpy(weight_packed.data() + scale_line + lane * 2, &scale_fp16, sizeof(scale_fp16));
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
                    static_cast<size_t>(tile_index) * NPU_DECODE_GEMV_WEIGHT_TILE_BYTES +
                    NPU_DECODE_GEMV_SCALE_TILE_BYTES +
                    row_in_tile * NPU_DECODE_GEMV_WEIGHT_ROW_BYTES;
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
    if (std::strcmp(name, "ggml_backend_npu_decode_w4a16_simulate_ex") == 0) {
        return reinterpret_cast<void *>(ggml_backend_npu_decode_w4a16_simulate_ex);
    }
    if (std::strcmp(name, "ggml_backend_npu_decode_w4a16_preload") == 0) {
        return reinterpret_cast<void *>(ggml_backend_npu_decode_w4a16_preload);
    }
    if (std::strcmp(name, "ggml_backend_npu_decode_w16a16_preload") == 0) {
        return reinterpret_cast<void *>(ggml_backend_npu_decode_w16a16_preload);
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
    ggml_npu::npu_decode_release_w16a16_runtime_cma();
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
    entry.prefer_block_cma = npu_decode_fused_ffn_cma_enabled() && npu_decode_weight_name_is_ffn(weight_name);
    entry.inv_smooth_scale.resize(static_cast<size_t>(k));
    for (int64_t i = 0; i < k; ++i) {
        const float smooth = smooth_scale[i];
        entry.inv_smooth_scale[static_cast<size_t>(i)] =
            smooth != 0.0f && std::isfinite(smooth) ? 1.0f / smooth : 1.0f;
    }
    if (!npu_decode_make_gemv_layout(entry.max_block_rows, static_cast<uint32_t>(k), &entry.max_layout)) {
        return fail("max block layout failed");
    }
    entry.act_scale.resize(entry.max_layout.act_bytes);
    npu_decode_pack_act_scale(
            entry.act_scale.data(),
            entry.act_scale.size(),
            entry.max_layout,
            entry.inv_smooth_scale.data(),
            k);
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
    if (npu_decode_preload_cma_enabled() &&
            npu_decode_init_with_cma_retry(weight_name, false) == 0) {
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

    auto reserve_pingpong_cma = [&]() {
        if (!entry.pingpong_available || cma_base == nullptr || cma_limit_bytes == 0 ||
                entry.pingpong_scale_cma_bytes != 0 || entry.pingpong_weight_cma_bytes != 0) {
            return;
        }
        const uint32_t pingpong_cma_start = cma_cursor;
        if (!entry.pingpong_scale.empty()) {
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

        if ((!entry.pingpong_scale.empty() && entry.pingpong_scale_cma_bytes == 0) ||
                entry.pingpong_weight_cma_bytes == 0) {
            entry.pingpong_scale_cma_offset = 0;
            entry.pingpong_scale_cma_bytes = 0;
            entry.pingpong_weight_cma_offset = 0;
            entry.pingpong_weight_cma_bytes = 0;
            cma_cursor = pingpong_cma_start;
        }
    };

    if (!entry.prefer_block_cma) {
        reserve_pingpong_cma();
    }

    const bool reserve_cma_for_pingpong =
        !entry.prefer_block_cma &&
        (entry.pingpong_layout.scale_bytes == 0 || entry.pingpong_scale_cma_bytes != 0) &&
        entry.pingpong_weight_cma_bytes != 0;

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

    if (entry.prefer_block_cma) {
        reserve_pingpong_cma();
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

bool ggml_backend_npu_decode_w4a16_simulate_ex(
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

    auto fail = [op_name](const char * reason) {
        if (npu_decode_w4a16_debug_log_enabled()) {
            GGML_LOG_WARN("%s: reject %s: %s\n", __func__, op_name != nullptr ? op_name : "(unnamed)", reason);
        }
        return false;
    };

    if (q4_data == nullptr || scale_data == nullptr || zero_data == nullptr || act_data == nullptr ||
            smooth_scale == nullptr || dst_data == nullptr) {
        return fail("null pointer");
    }
    if (k <= 0 || n_cols <= 0 || out_channels <= 0 || packed_k != (k + 1) / 2) {
        return fail("invalid dimensions");
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

    const int64_t groups = (k + NPU_DECODE_GEMV_K_TILE - 1) / NPU_DECODE_GEMV_K_TILE;
    if (!npu_decode_validate_symmetric_zero(
            zero_data,
            zero_type,
            zero_nb0,
            zero_nb1,
            out_channels,
            groups,
            op_name)) {
        return false;
    }

    std::vector<float> inv_smooth_scale(static_cast<size_t>(k));
    for (int64_t i = 0; i < k; ++i) {
        const float smooth = smooth_scale[i];
        inv_smooth_scale[static_cast<size_t>(i)] =
            smooth != 0.0f && std::isfinite(smooth) ? 1.0f / smooth : 1.0f;
    }

    if (npu_decode_pingpong_enabled()) {
        npu_decode_gemv_layout layout = {};
        if (!npu_decode_make_pingpong_layout(
                static_cast<uint32_t>(out_channels),
                static_cast<uint32_t>(k),
                &layout)) {
            return fail("pingpong layout failed");
        }
        std::vector<uint8_t> scale_packed(layout.scale_bytes);
        std::vector<uint8_t> weight_packed(layout.weight_bytes);
        npu_decode_pack_pingpong_scale_weight(
                scale_packed,
                weight_packed,
                layout,
                q4_data,
                packed_k,
                q4_nb1,
                scale_data,
                scale_type,
                scale_nb0,
                scale_nb1,
                out_channels,
                k);
        std::vector<uint8_t> act_scale(layout.act_bytes);
        npu_decode_pack_act_scale(
                act_scale.data(),
                act_scale.size(),
                layout,
                inv_smooth_scale.data(),
                k);
        std::vector<uint8_t> act_packed(npu_decode_act_payload_bytes(layout));
        for (int64_t col = 0; col < n_cols; ++col) {
            npu_decode_pack_activation(
                    act_packed.data(),
                    act_packed.size(),
                    layout,
                    act_data,
                    act_type,
                    act_nb0,
                    act_nb1,
                    act_scale.data(),
                    k,
                    col);
            npu_decode_simulate_pingpong(
                    act_packed.data(),
                    scale_packed.data(),
                    weight_packed.data(),
                    layout,
                    out_channels,
                    k,
                    dst_data,
                    dst_type,
                    dst_nb1,
                    col);
        }
        return true;
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

    const uint32_t max_block_rows = max_block_row_tiles * NPU_DECODE_GEMV_M_TILE;
    npu_decode_gemv_layout first_layout = {};
    if (!npu_decode_make_gemv_layout(max_block_rows, static_cast<uint32_t>(k), &first_layout)) {
        return fail("max block layout failed");
    }
    std::vector<uint8_t> act_scale(first_layout.act_bytes);
    npu_decode_pack_act_scale(
            act_scale.data(),
            act_scale.size(),
            first_layout,
            inv_smooth_scale.data(),
            k);
    std::vector<uint8_t> act_packed(npu_decode_act_payload_bytes(first_layout));
    for (int64_t col = 0; col < n_cols; ++col) {
        npu_decode_pack_activation(
                act_packed.data(),
                act_packed.size(),
                first_layout,
                act_data,
                act_type,
                act_nb0,
                act_nb1,
                act_scale.data(),
                k,
                col);
        for (int64_t row_base = 0; row_base < out_channels; row_base += max_block_rows) {
            const uint32_t block_rows = static_cast<uint32_t>(std::min<int64_t>(max_block_rows, out_channels - row_base));
            npu_decode_gemv_layout layout = {};
            if (!npu_decode_make_gemv_layout(block_rows, static_cast<uint32_t>(k), &layout)) {
                return fail("block layout failed");
            }
            std::vector<uint8_t> packed(layout.packed_bytes);
            npu_decode_pack_scale_weight_block(
                    packed,
                    layout,
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
            npu_decode_simulate_packed_block(
                    act_packed.data(),
                    packed.data(),
                    layout,
                    row_base,
                    block_rows,
                    k,
                    dst_data,
                    dst_type,
                    dst_nb1,
                    col);
        }
    }
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
    const bool hw_f32_mvout = dst_type == GGML_TYPE_F32 && npu_decode_hw_f32_mvout_enabled();
    const uint8_t output_precision = npu_decode_mvout_precision(hw_f32_mvout);

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
    const uint32_t output_spm_bytes = npu_decode_align_up_u32(
            static_cast<uint32_t>(out_channels) * static_cast<uint32_t>(sizeof(ggml_fp16_t)),
            NPU_DECODE_GEMV_LINE_BYTES);
    const uint32_t host_output_bytes = npu_decode_host_output_bytes(static_cast<uint32_t>(out_channels), hw_f32_mvout);
    if (output_spm_bytes > NPU_DECODE_GEMV_SPM_BYTES) {
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
        (preloaded->pingpong_layout.scale_bytes == 0 ||
         preloaded->pingpong_scale_cma_bytes == preloaded->pingpong_layout.scale_bytes) &&
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
    if (npu_decode_init_with_cma_retry(op_name, true) != 0) {
        return fail("decode runtime init failed");
    }
    (void) npu_decode_reserve_preloaded_ffn_pingpong_cma_after_init({
        const_cast<npu_decode_preloaded_tensor *>(preloaded),
    });
    preloaded_cma = preloaded != nullptr && !preloaded->blocks.empty();
    if (preloaded_cma) {
        for (const npu_decode_preloaded_block & block : preloaded->blocks) {
            if (block.cma_bytes == 0 || block.cma_bytes != block.packed.size()) {
                preloaded_cma = false;
                break;
            }
        }
    }
    pingpong_cma =
        pingpong_requested &&
        (preloaded->pingpong_layout.scale_bytes == 0 ||
         preloaded->pingpong_scale_cma_bytes == preloaded->pingpong_layout.scale_bytes) &&
        preloaded->pingpong_weight_cma_bytes == preloaded->pingpong_layout.weight_bytes;
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
        profile.activation_bytes = static_cast<uint64_t>(npu_decode_act_payload_bytes(first_layout)) * static_cast<uint64_t>(n_cols);
        profile.setup_us = static_cast<uint64_t>(ggml_time_us() - setup_start_us);
    }

    const int64_t alloc_start_us = collect_profile ? ggml_time_us() : 0;
    void * act_cma = npu_decode_mem_alloc(npu_decode_act_payload_bytes(first_layout));
    void * packed_cma = preloaded_cma ? nullptr : npu_decode_mem_alloc(first_layout.packed_bytes);
    void * out_cma = npu_decode_mem_alloc(host_output_bytes);
    void * pingpong_scale_cma = nullptr;
    void * pingpong_weight_cma = nullptr;
    if (pingpong_requested && !pingpong_cma) {
        if (preloaded->pingpong_layout.scale_bytes > 0) {
            pingpong_scale_cma = npu_decode_mem_alloc(preloaded->pingpong_layout.scale_bytes);
        }
        pingpong_weight_cma = npu_decode_mem_alloc(preloaded->pingpong_layout.weight_bytes);
    }
    if (act_cma == nullptr || (!preloaded_cma && packed_cma == nullptr) || out_cma == nullptr ||
            (pingpong_requested && !pingpong_cma &&
             ((preloaded->pingpong_layout.scale_bytes > 0 && pingpong_scale_cma == nullptr) ||
              pingpong_weight_cma == nullptr))) {
        npu_decode_mem_free(act_cma);
        npu_decode_mem_free(packed_cma);
        npu_decode_mem_free(out_cma);
        npu_decode_mem_free(pingpong_scale_cma);
        npu_decode_mem_free(pingpong_weight_cma);
        return fail("decode CMA allocation failed");
    }
    if (pingpong_requested && !pingpong_cma) {
        if (preloaded->pingpong_layout.scale_bytes > 0) {
            std::memcpy(pingpong_scale_cma, preloaded->pingpong_scale.data(), preloaded->pingpong_layout.scale_bytes);
        }
        std::memcpy(pingpong_weight_cma, preloaded->pingpong_weight.data(), preloaded->pingpong_layout.weight_bytes);
    }

    if (collect_profile) {
        profile.alloc_us = static_cast<uint64_t>(ggml_time_us() - alloc_start_us);
    }

    bool transient_scratch = false;
    uint32_t transient_cursor = 0;
    uint32_t transient_limit = 0;
    auto transient_alloc = [&](size_t bytes) -> void * {
        if (decode_cma_base == nullptr || bytes == 0 ||
                bytes > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
            return nullptr;
        }
        uint32_t offset = 0;
        uint32_t size = 0;
        if (!npu_decode_get_transient_scratch(&offset, &size)) {
            return nullptr;
        }
        if (!transient_scratch) {
            transient_cursor = offset;
            transient_limit = offset + size;
            transient_scratch = true;
        }
        const uint32_t aligned = npu_decode_align_up_bytes(transient_cursor, NPU_DECODE_GEMV_DMA_ALIGN);
        const uint32_t nbytes = static_cast<uint32_t>(bytes);
        if (aligned < offset || aligned > transient_limit || nbytes > transient_limit - aligned) {
            return nullptr;
        }
        transient_cursor = aligned + nbytes;
        return static_cast<uint8_t *>(decode_cma_base) + aligned;
    };

    bool act_transient = false;
    bool packed_transient = false;
    bool out_transient = false;
    bool pingpong_scale_transient = false;
    bool pingpong_weight_transient = false;
    auto free_decode_alloc = [](void * ptr, bool transient) {
        if (ptr != nullptr && !transient) {
            npu_decode_mem_free(ptr);
        }
    };
    if (npu_decode_transient_scratch_enabled() && (preloaded_cma || pingpong_cma) && decode_cma_base != nullptr) {
        uint32_t transient_base_offset = 0;
        uint32_t transient_size = 0;
        const bool have_transient = npu_decode_get_transient_scratch(&transient_base_offset, &transient_size);
        void * transient_act = transient_alloc(npu_decode_act_payload_bytes(first_layout));
        void * transient_packed = preloaded_cma ? nullptr : transient_alloc(first_layout.packed_bytes);
        void * transient_out = transient_alloc(host_output_bytes);
        void * transient_pingpong_scale = nullptr;
        void * transient_pingpong_weight = nullptr;
        if (pingpong_requested && !pingpong_cma) {
            if (preloaded->pingpong_layout.scale_bytes > 0) {
                transient_pingpong_scale = transient_alloc(preloaded->pingpong_layout.scale_bytes);
            }
            transient_pingpong_weight = transient_alloc(preloaded->pingpong_layout.weight_bytes);
        }
        if (transient_act != nullptr && (preloaded_cma || transient_packed != nullptr) && transient_out != nullptr &&
                (!pingpong_requested || pingpong_cma ||
                 ((preloaded->pingpong_layout.scale_bytes == 0 || transient_pingpong_scale != nullptr) &&
                  transient_pingpong_weight != nullptr))) {
            npu_decode_mem_free(act_cma);
            npu_decode_mem_free(packed_cma);
            npu_decode_mem_free(out_cma);
            npu_decode_mem_free(pingpong_scale_cma);
            npu_decode_mem_free(pingpong_weight_cma);
            act_cma = transient_act;
            packed_cma = transient_packed;
            out_cma = transient_out;
            pingpong_scale_cma = transient_pingpong_scale;
            pingpong_weight_cma = transient_pingpong_weight;
            act_transient = true;
            packed_transient = transient_packed != nullptr;
            out_transient = true;
            pingpong_scale_transient = transient_pingpong_scale != nullptr;
            pingpong_weight_transient = transient_pingpong_weight != nullptr;
            if (pingpong_requested && !pingpong_cma) {
                if (preloaded->pingpong_layout.scale_bytes > 0) {
                    std::memcpy(pingpong_scale_cma, preloaded->pingpong_scale.data(), preloaded->pingpong_layout.scale_bytes);
                }
                std::memcpy(pingpong_weight_cma, preloaded->pingpong_weight.data(), preloaded->pingpong_layout.weight_bytes);
            }
            if (npu_decode_w4a16_debug_log_enabled()) {
                GGML_LOG_WARN("%s: using decode transient scratch for %s offset=0x%X used=%u limit=0x%X\n",
                        __func__,
                        op_name != nullptr ? op_name : "(unnamed)",
                        have_transient ? static_cast<unsigned>(transient_base_offset) : 0u,
                        have_transient ? static_cast<unsigned>(transient_cursor - transient_base_offset) : 0u,
                        static_cast<unsigned>(transient_limit));
            }
        }
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
                    npu_decode_act_payload_bytes(preloaded->pingpong_layout),
                    preloaded->pingpong_layout,
                    act_data,
                    act_type,
                    act_nb0,
                    act_nb1,
                    preloaded->act_scale.empty() ? nullptr : preloaded->act_scale.data(),
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

            if (npu_decode_zero_out_before_mvout_enabled()) {
                std::memset(out_cma, 0, host_output_bytes);
            }
            try {
                npu_decode_gemv_pingpong_manual_run(
                        op_name,
                        act_cma,
                        pingpong_scale_ptr,
                        pingpong_weight_ptr,
                        out_cma,
                        preloaded->pingpong_layout,
                        static_cast<uint16_t>(out_channels),
                        static_cast<uint16_t>(k),
                        output_precision,
                        collect_profile ? &profile : nullptr);
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
                profile.output_bytes += host_output_bytes;
                stage_start_us = ggml_time_us();
            }
            npu_decode_repro_dump_pingpong(
                    op_name,
                    col,
                    preloaded->pingpong_layout,
                    static_cast<uint16_t>(out_channels),
                    static_cast<uint16_t>(k),
                    output_precision,
                    npu_decode_pingpong_block_mvout_enabled(),
                    act_cma,
                    pingpong_scale_ptr,
                    pingpong_weight_ptr,
                    out_cma,
                    host_output_bytes,
                    decode_cma_base,
                    npu_decode_memory_size());

            char * dst_col = reinterpret_cast<char *>(dst_data) + col * dst_nb1;
            if (dst_type == GGML_TYPE_F16) {
                std::memcpy(dst_col, out_cma, static_cast<size_t>(out_channels) * sizeof(ggml_fp16_t));
            } else if (hw_f32_mvout) {
                std::memcpy(dst_col, out_cma, static_cast<size_t>(out_channels) * sizeof(float));
            } else {
                npu_decode_convert_output_fp16_to_f32(
                        static_cast<const ggml_fp16_t *>(out_cma),
                        reinterpret_cast<float *>(dst_col),
                        static_cast<uint32_t>(out_channels));
            }
            if (collect_profile) {
                profile.postprocess_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            }
        }

        if (pingpong_ok) {
            const int64_t cleanup_start_us = collect_profile ? ggml_time_us() : 0;
            free_decode_alloc(act_cma, act_transient);
            free_decode_alloc(packed_cma, packed_transient);
            free_decode_alloc(out_cma, out_transient);
            free_decode_alloc(pingpong_scale_cma, pingpong_scale_transient);
            free_decode_alloc(pingpong_weight_cma, pingpong_weight_transient);
            if (collect_profile) {
                profile.cleanup_us = static_cast<uint64_t>(ggml_time_us() - cleanup_start_us);
                profile.total_us = static_cast<uint64_t>(ggml_time_us() - total_start_us);
                npu_decode_profile_write(profile);
            }
            return true;
        }

        free_decode_alloc(pingpong_scale_cma, pingpong_scale_transient);
        free_decode_alloc(pingpong_weight_cma, pingpong_weight_transient);
        pingpong_scale_cma = nullptr;
        pingpong_weight_cma = nullptr;
        pingpong_scale_transient = false;
        pingpong_weight_transient = false;
    }

    for (int64_t col = 0; col < n_cols; ++col) {
        int64_t stage_start_us = collect_profile ? ggml_time_us() : 0;
        npu_decode_pack_activation(
                act_cma,
                npu_decode_act_payload_bytes(first_layout),
                first_layout,
                act_data,
                act_type,
                act_nb0,
                act_nb1,
                preloaded->act_scale.empty() ? nullptr : preloaded->act_scale.data(),
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
                npu_decode_act_payload_bytes(first_layout) - 1u,
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
                free_decode_alloc(act_cma, act_transient);
                free_decode_alloc(packed_cma, packed_transient);
                free_decode_alloc(out_cma, out_transient);
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
                free_decode_alloc(act_cma, act_transient);
                free_decode_alloc(out_cma, out_transient);
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

            const uint64_t flow = npu_decode_make_bypass_output_flow(static_cast<uint16_t>(block_rows));
            npu_decode_trace_matvec_flow(
                    "w4_preloaded_block",
                    op_name,
                    static_cast<uint32_t>(block_index),
                    static_cast<uint32_t>(row_base),
                    static_cast<uint32_t>(block_rows),
                    static_cast<uint32_t>(k),
                    NPU_DECODE_GEMV_MODE_W4A16,
                    layout.weight_base,
                    NPU_DECODE_GEMV_ACT_BASE,
                    static_cast<uint16_t>(row_base * sizeof(ggml_fp16_t)),
                    layout.weight_base,
                    flow);
            npu_decode_matvec_decode_flow_run(
                    layout.weight_base,
                    NPU_DECODE_GEMV_ACT_BASE,
                    static_cast<uint16_t>(k),
                    static_cast<uint16_t>(block_rows),
                    static_cast<uint16_t>(row_base * sizeof(ggml_fp16_t)),
                    static_cast<uint16_t>(layout.weight_base),
                    NPU_DECODE_GEMV_MODE_W4A16,
                    flow);
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
                output_precision,
                1,
                false,
                false,
                0,
                0);
        if (collect_profile) {
            profile.mvout_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            profile.mvout_calls += 1;
            profile.output_bytes += host_output_bytes;
            stage_start_us = ggml_time_us();
        }

        char * dst_col = reinterpret_cast<char *>(dst_data) + col * dst_nb1;
        if (dst_type == GGML_TYPE_F16) {
            std::memcpy(dst_col, out_cma, static_cast<size_t>(out_channels) * sizeof(ggml_fp16_t));
        } else if (hw_f32_mvout) {
            std::memcpy(dst_col, out_cma, static_cast<size_t>(out_channels) * sizeof(float));
        } else {
            npu_decode_convert_output_fp16_to_f32(
                    static_cast<const ggml_fp16_t *>(out_cma),
                    reinterpret_cast<float *>(dst_col),
                    static_cast<uint32_t>(out_channels));
        }
        if (collect_profile) {
            profile.postprocess_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
        }
    }

    const int64_t cleanup_start_us = collect_profile ? ggml_time_us() : 0;
    free_decode_alloc(act_cma, act_transient);
    free_decode_alloc(packed_cma, packed_transient);
    free_decode_alloc(out_cma, out_transient);
    free_decode_alloc(pingpong_scale_cma, pingpong_scale_transient);
    free_decode_alloc(pingpong_weight_cma, pingpong_weight_transient);
    if (collect_profile) {
        profile.cleanup_us = static_cast<uint64_t>(ggml_time_us() - cleanup_start_us);
        profile.total_us = static_cast<uint64_t>(ggml_time_us() - total_start_us);
        npu_decode_profile_write(profile);
    }
    return true;
}

static bool npu_decode_w16a16_prepare_weight_cache(
        ggml_npu::npu_decode_w16a16_cached_weight & entry,
        const char * weight_name,
        const void * weight_data,
        int64_t k,
        int64_t out_channels,
        int64_t weight_nb0,
        int64_t weight_nb1,
        bool allocate_heap_cma) {
    using namespace ggml_npu;

    const uint32_t row_tiles = npu_decode_ceil_div_u32(static_cast<uint32_t>(out_channels), NPU_DECODE_GEMV_M_TILE);
    const uint32_t col_tiles = npu_decode_ceil_div_u32(static_cast<uint32_t>(k), NPU_DECODE_GEMV_K_TILE);
    const uint32_t bytes_per_row_tile = col_tiles * NPU_DECODE_GEMV_W16_WEIGHT_TILE_BYTES;
    const uint32_t weight_capacity = NPU_DECODE_GEMV_SPM_BYTES - NPU_DECODE_GEMV_PING_WEIGHT_BASE;
    const uint32_t max_row_tiles = bytes_per_row_tile == 0 ? 0 :
        weight_capacity / bytes_per_row_tile;
    if (row_tiles == 0 || col_tiles == 0 || max_row_tiles == 0) {
        return false;
    }

    const bool same_weight =
        entry.weight_data == weight_data ||
        (weight_name != nullptr &&
         weight_name[0] != '\0' &&
         !entry.weight_name.empty() &&
         entry.weight_name == weight_name);
    if (same_weight &&
            entry.k == k &&
            entry.out_channels == out_channels &&
            entry.weight_nb0 == weight_nb0 &&
            entry.weight_nb1 == weight_nb1 &&
            entry.row_tiles == row_tiles &&
            entry.col_tiles == col_tiles &&
            !entry.packed_weight.empty()) {
        entry.weight_data = weight_data;
        if (weight_name != nullptr && weight_name[0] != '\0') {
            entry.weight_name = weight_name;
        }
        if (allocate_heap_cma &&
                npu_decode_w16a16_cma_enabled() &&
                entry.cma_packed == nullptr &&
                entry.cma_packed_offset == 0 &&
                entry.packed_weight.size() <= static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
            entry.cma_packed = npu_decode_mem_alloc(entry.packed_weight.size());
            if (entry.cma_packed != nullptr) {
                entry.cma_bytes = static_cast<uint32_t>(entry.packed_weight.size());
                entry.cma_packed_owned = true;
                std::memcpy(entry.cma_packed, entry.packed_weight.data(), entry.packed_weight.size());
            }
        }
        return true;
    }

    const uint64_t packed_bytes_64 = static_cast<uint64_t>(row_tiles) * bytes_per_row_tile;
    if (packed_bytes_64 > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        return false;
    }

    npu_decode_w16a16_cached_weight next = {};
    next.weight_name = weight_name != nullptr ? weight_name : "";
    next.weight_data = weight_data;
    next.k = k;
    next.out_channels = out_channels;
    next.weight_nb0 = weight_nb0;
    next.weight_nb1 = weight_nb1;
    next.row_tiles = row_tiles;
    next.col_tiles = col_tiles;
    next.max_block_rows = max_row_tiles * NPU_DECODE_GEMV_M_TILE;
    next.pingpong_max_block_rows = npu_decode_w16a16_pingpong_max_block_rows(col_tiles);
    next.bytes_per_row_tile = bytes_per_row_tile;
    next.packed_weight.assign(static_cast<size_t>(packed_bytes_64), 0);

    const auto * weight = static_cast<const uint8_t *>(weight_data);
    for (uint32_t row_tile = 0; row_tile < row_tiles; ++row_tile) {
        for (uint32_t col_tile = 0; col_tile < col_tiles; ++col_tile) {
            const size_t tile_base = static_cast<size_t>(row_tile) * bytes_per_row_tile +
                static_cast<size_t>(col_tile) * NPU_DECODE_GEMV_W16_WEIGHT_TILE_BYTES;
            for (uint32_t row_lane = 0; row_lane < NPU_DECODE_GEMV_M_TILE; ++row_lane) {
                const int64_t row = static_cast<int64_t>(row_tile) * NPU_DECODE_GEMV_M_TILE + row_lane;
                if (row >= out_channels) {
                    continue;
                }
                uint8_t * dst = next.packed_weight.data() + tile_base +
                    static_cast<size_t>(row_lane) * NPU_DECODE_GEMV_W16_WEIGHT_ROW_BYTES;
                for (uint32_t col_lane = 0; col_lane < NPU_DECODE_GEMV_K_TILE; ++col_lane) {
                    const int64_t col = static_cast<int64_t>(col_tile) * NPU_DECODE_GEMV_K_TILE + col_lane;
                    if (col >= k) {
                        continue;
                    }
                    std::memcpy(
                            dst + col_lane * sizeof(ggml_fp16_t),
                            weight + row * weight_nb1 + col * weight_nb0,
                            sizeof(ggml_fp16_t));
                }
            }
        }
    }

    if (allocate_heap_cma &&
            npu_decode_w16a16_cma_enabled() &&
            packed_bytes_64 <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        next.cma_packed = npu_decode_mem_alloc(static_cast<size_t>(packed_bytes_64));
        if (next.cma_packed != nullptr) {
            next.cma_bytes = static_cast<uint32_t>(packed_bytes_64);
            next.cma_packed_owned = true;
            std::memcpy(next.cma_packed, next.packed_weight.data(), next.packed_weight.size());
        }
    }

    if (entry.cma_packed != nullptr && entry.cma_packed_owned) {
        npu_decode_mem_free(entry.cma_packed);
    }
    if (entry.cma_scale_ones != nullptr && entry.cma_scale_ones_owned) {
        npu_decode_mem_free(entry.cma_scale_ones);
    }
    entry = std::move(next);
    return true;
}

static void * npu_decode_w16a16_ensure_scale_ones_cma(
        ggml_npu::npu_decode_w16a16_cached_weight & entry,
        uint32_t max_scale_bytes) {
    using namespace ggml_npu;

    if (max_scale_bytes == 0) {
        return nullptr;
    }
    if (entry.cma_scale_ones != nullptr && entry.cma_scale_ones_bytes >= max_scale_bytes) {
        return entry.cma_scale_ones;
    }

    if (entry.cma_scale_ones != nullptr && entry.cma_scale_ones_owned) {
        npu_decode_mem_free(entry.cma_scale_ones);
    }
    entry.cma_scale_ones = nullptr;
    entry.cma_scale_ones_bytes = 0;
    entry.cma_scale_ones_owned = false;
    entry.scale_ones.assign(max_scale_bytes, 0);
    const ggml_fp16_t fp16_one = ggml_fp32_to_fp16(1.0f);
    for (uint32_t i = 0; i + sizeof(ggml_fp16_t) <= max_scale_bytes; i += sizeof(ggml_fp16_t)) {
        std::memcpy(entry.scale_ones.data() + i, &fp16_one, sizeof(fp16_one));
    }

    void * scale_cma = npu_decode_mem_alloc(max_scale_bytes);
    if (scale_cma == nullptr) {
        return nullptr;
    }
    std::memcpy(scale_cma, entry.scale_ones.data(), entry.scale_ones.size());
    entry.cma_scale_ones = scale_cma;
    entry.cma_scale_ones_bytes = max_scale_bytes;
    entry.cma_scale_ones_owned = true;
    return entry.cma_scale_ones;
}

static bool npu_decode_w16a16_reserve_static_weight_cma(
        ggml_npu::npu_decode_w16a16_cached_weight & entry,
        const char * weight_name) {
    using namespace ggml_npu;

    if (!npu_decode_w16a16_cma_enabled() ||
            entry.packed_weight.empty() ||
            entry.packed_weight.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        return false;
    }
    if (entry.cma_packed != nullptr && entry.cma_bytes >= entry.packed_weight.size()) {
        return true;
    }
    if (entry.cma_packed != nullptr && entry.cma_packed_owned) {
        npu_decode_mem_free(entry.cma_packed);
    }
    entry.cma_packed = nullptr;
    entry.cma_bytes = 0;
    entry.cma_packed_owned = false;

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

    if (npu_decode_init_with_cma_retry(weight_name, false) == 0) {
        cma_base = npu_decode_memory_base();
        cma_map_size = npu_decode_memory_size();
        if (cma_limit_bytes == 0 && cma_base_offset < cma_map_size) {
            cma_limit_bytes = cma_map_size - cma_base_offset;
        }
    }
    if (cma_base == nullptr ||
            cma_base_offset >= cma_map_size ||
            cma_limit_bytes > cma_map_size - cma_base_offset) {
        return false;
    }

    const uint32_t aligned_cursor = npu_decode_align_up_bytes(cma_cursor, NPU_DECODE_GEMV_DMA_ALIGN);
    const uint32_t packed_bytes = static_cast<uint32_t>(entry.packed_weight.size());
    if (aligned_cursor > cma_limit_bytes || packed_bytes > cma_limit_bytes - aligned_cursor) {
        return false;
    }

    entry.cma_packed = static_cast<uint8_t *>(cma_base) + cma_base_offset + aligned_cursor;
    entry.cma_packed_offset = cma_base_offset + aligned_cursor;
    entry.cma_bytes = packed_bytes;
    entry.cma_packed_owned = false;
    std::memcpy(entry.cma_packed, entry.packed_weight.data(), entry.packed_weight.size());

    {
        npu_decode_preload_cache & cache = npu_decode_get_preload_cache();
        std::lock_guard<std::mutex> lock(cache.mutex);
        cache.cma_base_offset = cma_base_offset;
        cache.cma_limit_bytes = cma_limit_bytes;
        cache.cma_cursor = aligned_cursor + packed_bytes;
        cache.transient_offset = 0;
        cache.transient_size = 0;
    }
    npu_decode_update_runtime_heap_env();
    return true;
}

static bool npu_decode_w16a16_reserve_static_scale_cma(
        ggml_npu::npu_decode_w16a16_cached_weight & entry,
        const char * weight_name,
        uint32_t max_scale_bytes) {
    using namespace ggml_npu;

    if (max_scale_bytes == 0) {
        return false;
    }
    if (entry.cma_scale_ones != nullptr && entry.cma_scale_ones_bytes >= max_scale_bytes) {
        return true;
    }
    if (entry.cma_scale_ones != nullptr && entry.cma_scale_ones_owned) {
        npu_decode_mem_free(entry.cma_scale_ones);
    }
    entry.cma_scale_ones = nullptr;
    entry.cma_scale_ones_bytes = 0;
    entry.cma_scale_ones_owned = false;

    entry.scale_ones.assign(max_scale_bytes, 0);
    const ggml_fp16_t fp16_one = ggml_fp32_to_fp16(1.0f);
    for (uint32_t i = 0; i + sizeof(ggml_fp16_t) <= max_scale_bytes; i += sizeof(ggml_fp16_t)) {
        std::memcpy(entry.scale_ones.data() + i, &fp16_one, sizeof(fp16_one));
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

    if (npu_decode_init_with_cma_retry(weight_name, false) == 0) {
        cma_base = npu_decode_memory_base();
        cma_map_size = npu_decode_memory_size();
        if (cma_limit_bytes == 0 && cma_base_offset < cma_map_size) {
            cma_limit_bytes = cma_map_size - cma_base_offset;
        }
    }
    if (cma_base == nullptr ||
            cma_base_offset >= cma_map_size ||
            cma_limit_bytes > cma_map_size - cma_base_offset) {
        return false;
    }

    const uint32_t aligned_cursor = npu_decode_align_up_bytes(cma_cursor, NPU_DECODE_GEMV_DMA_ALIGN);
    if (aligned_cursor > cma_limit_bytes || max_scale_bytes > cma_limit_bytes - aligned_cursor) {
        return false;
    }
    entry.cma_scale_ones = static_cast<uint8_t *>(cma_base) + cma_base_offset + aligned_cursor;
    entry.cma_scale_ones_offset = cma_base_offset + aligned_cursor;
    entry.cma_scale_ones_bytes = max_scale_bytes;
    entry.cma_scale_ones_owned = false;
    std::memcpy(entry.cma_scale_ones, entry.scale_ones.data(), entry.scale_ones.size());

    {
        npu_decode_preload_cache & cache = npu_decode_get_preload_cache();
        std::lock_guard<std::mutex> lock(cache.mutex);
        cache.cma_base_offset = cma_base_offset;
        cache.cma_limit_bytes = cma_limit_bytes;
        cache.cma_cursor = aligned_cursor + max_scale_bytes;
        cache.transient_offset = 0;
        cache.transient_size = 0;
    }
    npu_decode_update_runtime_heap_env();
    return true;
}

static void npu_decode_w16a16_rebind_static_cma(
        ggml_npu::npu_decode_w16a16_cached_weight & entry) {
    using namespace ggml_npu;

    void * cma_base = npu_decode_memory_base();
    const uint32_t cma_size = npu_decode_memory_size();
    if (cma_base == nullptr || cma_size == 0) {
        return;
    }
    if (entry.cma_packed == nullptr &&
            entry.cma_packed_offset != 0 &&
            entry.cma_bytes != 0 &&
            entry.cma_packed_offset < cma_size &&
            entry.cma_bytes <= cma_size - entry.cma_packed_offset) {
        entry.cma_packed = static_cast<uint8_t *>(cma_base) + entry.cma_packed_offset;
        entry.cma_packed_owned = false;
    }
    if (entry.cma_scale_ones == nullptr &&
            entry.cma_scale_ones_offset != 0 &&
            entry.cma_scale_ones_bytes != 0 &&
            entry.cma_scale_ones_offset < cma_size &&
            entry.cma_scale_ones_bytes <= cma_size - entry.cma_scale_ones_offset) {
        entry.cma_scale_ones = static_cast<uint8_t *>(cma_base) + entry.cma_scale_ones_offset;
        entry.cma_scale_ones_owned = false;
    }
}

bool ggml_backend_npu_decode_w16a16_preload(
        const char * weight_name,
        const void * weight_data,
        int64_t k,
        int64_t out_channels,
        int64_t weight_nb0,
        int64_t weight_nb1) {
    using namespace ggml_npu;

    auto fail = [weight_name](const char * reason) {
        if (npu_decode_w4a16_debug_log_enabled()) {
            GGML_LOG_WARN("%s: reject %s: %s\n", __func__, weight_name != nullptr ? weight_name : "(unnamed)", reason);
        }
        return false;
    };

    if (weight_name == nullptr || weight_name[0] == '\0' || weight_data == nullptr) {
        return fail("null pointer");
    }
    if (k <= 0 || out_channels <= 0 ||
            k > std::numeric_limits<uint16_t>::max() ||
            out_channels > std::numeric_limits<uint16_t>::max()) {
        return fail("invalid dimensions");
    }
    if (weight_nb0 != static_cast<int64_t>(sizeof(ggml_fp16_t))) {
        return fail("weight tensor must be contiguous FP16 along K");
    }

    npu_decode_update_runtime_heap_env();
    if (npu_decode_init_with_cma_retry(weight_name, false) != 0) {
        return fail("decode runtime init failed");
    }

    npu_decode_w16a16_cache & cache = npu_decode_w16a16_weight_cache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    if (!npu_decode_w16a16_prepare_weight_cache(
                cache.entry,
                weight_name,
                weight_data,
                k,
                out_channels,
                weight_nb0,
                weight_nb1,
                false)) {
        return fail("weight cache preparation failed");
    }
    (void) npu_decode_w16a16_reserve_static_weight_cma(cache.entry, weight_name);

    const npu_decode_w16a16_cached_weight & cached_const = cache.entry;
    const bool cma_resident = cached_const.cma_packed != nullptr &&
        cached_const.cma_bytes >= cached_const.packed_weight.size();
    const bool use_pingpong = cma_resident &&
        npu_decode_w16a16_pingpong_enabled() &&
        cached_const.pingpong_max_block_rows > 0;
    const uint32_t max_block_rows = use_pingpong ? cached_const.pingpong_max_block_rows : cached_const.max_block_rows;
    const uint32_t max_row_tiles = npu_decode_ceil_div_u32(max_block_rows, NPU_DECODE_GEMV_M_TILE);
    const uint32_t max_scale_bytes = max_row_tiles * cached_const.col_tiles * NPU_DECODE_GEMV_SCALE_TILE_BYTES;
    npu_decode_w16a16_cached_weight & cached = cache.entry;
    (void) npu_decode_w16a16_reserve_static_scale_cma(cached, weight_name, max_scale_bytes);

    if (npu_decode_w4a16_debug_log_enabled()) {
        GGML_LOG_WARN("%s: preloaded %s M=%" PRId64 " K=%" PRId64 " packed=%zu cma=%d scale_cma=%u\n",
                __func__,
                weight_name,
                out_channels,
                k,
                cached.packed_weight.size(),
                cma_resident ? 1 : 0,
                cached.cma_scale_ones_bytes);
    }
    return !cached.packed_weight.empty();
}

bool ggml_backend_npu_decode_w16a16_gemv_ex(
        const char * op_name,
        const void * weight_data,
        int64_t k,
        int64_t out_channels,
        int64_t weight_nb0,
        int64_t weight_nb1,
        const void * act_data,
        int act_type,
        int64_t act_nb0,
        int64_t act_nb1,
        int64_t n_cols,
        void * dst_data,
        int dst_type,
        int64_t dst_nb1) {
    using namespace ggml_npu;

    auto fail = [op_name](const char * reason) {
        if (npu_decode_w4a16_debug_log_enabled()) {
            GGML_LOG_WARN("%s: reject %s: %s\n", __func__, op_name != nullptr ? op_name : "(unnamed)", reason);
        }
        return false;
    };

    if (weight_data == nullptr || act_data == nullptr || dst_data == nullptr) {
        return fail("null pointer");
    }
    if (k <= 0 || out_channels <= 0 || n_cols <= 0 ||
            k > std::numeric_limits<uint16_t>::max() ||
            out_channels > std::numeric_limits<uint16_t>::max()) {
        return fail("invalid dimensions");
    }
    if (weight_nb0 != static_cast<int64_t>(sizeof(ggml_fp16_t))) {
        return fail("weight tensor must be contiguous FP16 along K");
    }
    if (act_type != GGML_TYPE_F16 && act_type != GGML_TYPE_F32) {
        return fail("activation tensor must be F16 or F32");
    }
    if (dst_type != GGML_TYPE_F16 && dst_type != GGML_TYPE_F32) {
        return fail("destination tensor must be F16 or F32");
    }
    if (npu_decode_w4a16_debug_log_enabled()) {
        GGML_LOG_WARN("%s: start %s M=%" PRId64 " K=%" PRId64 " n_cols=%" PRId64
                " act_type=%s dst_type=%s nb0=%" PRId64 " nb1=%" PRId64 "\n",
                __func__,
                op_name != nullptr ? op_name : "(unnamed)",
                out_channels,
                k,
                n_cols,
                ggml_type_name(static_cast<ggml_type>(act_type)),
                ggml_type_name(static_cast<ggml_type>(dst_type)),
                weight_nb0,
                weight_nb1);
    }

    const bool collect_profile = npu_decode_profile_enabled();
    const int64_t total_start_us = collect_profile ? ggml_time_us() : 0;
    const int64_t setup_start_us = collect_profile ? total_start_us : 0;
    int64_t setup_stage_start_us = setup_start_us;
    npu_decode_gemv_profile_record profile = {};
    profile.profile_kind = "aicas_decode_w16a16_lm_head";
    profile.gemv_mode = "w16a16";
    profile.op_role = "lm_head";
    profile.op_name = op_name != nullptr ? op_name : "";
    profile.dst_type = dst_type;
    profile.k = k;
    profile.n_cols = n_cols;
    profile.out_channels = out_channels;
    profile.pingpong = false;

    if (collect_profile) {
        setup_stage_start_us = ggml_time_us();
    }
    if (!npu_decode_ensure_overlay_active(op_name)) {
        return fail("decode overlay switch failed");
    }
    if (collect_profile) {
        profile.overlay_ensure_us = static_cast<uint64_t>(ggml_time_us() - setup_stage_start_us);
        setup_stage_start_us = ggml_time_us();
    }
    npu_decode_update_runtime_heap_env();
    if (collect_profile) {
        profile.heap_update_us = static_cast<uint64_t>(ggml_time_us() - setup_stage_start_us);
        setup_stage_start_us = ggml_time_us();
    }
    if (npu_decode_init_with_cma_retry(op_name, true) != 0) {
        return fail("decode runtime init failed");
    }
    if (collect_profile) {
        profile.runtime_init_us = static_cast<uint64_t>(ggml_time_us() - setup_stage_start_us);
        setup_stage_start_us = ggml_time_us();
    }

    npu_decode_w16a16_cache & cache = npu_decode_w16a16_weight_cache();
    std::unique_lock<std::mutex> cache_lock(cache.mutex);
    if (!npu_decode_w16a16_prepare_weight_cache(
                cache.entry,
                op_name,
                weight_data,
                k,
                out_channels,
                weight_nb0,
                weight_nb1,
                true)) {
        return fail("weight cache preparation failed");
    }
    npu_decode_w16a16_cached_weight & cached = cache.entry;
    if (collect_profile) {
        profile.preload_lookup_us = static_cast<uint64_t>(ggml_time_us() - setup_stage_start_us);
        setup_stage_start_us = ggml_time_us();
    }
    npu_decode_w16a16_rebind_static_cma(cached);
    const bool cma_resident = cached.cma_packed != nullptr &&
        cached.cma_bytes >= cached.packed_weight.size();
    const bool use_pingpong = cma_resident &&
        npu_decode_w16a16_pingpong_enabled() &&
        cached.pingpong_max_block_rows > 0;
    if (npu_decode_w4a16_debug_log_enabled()) {
        GGML_LOG_WARN("%s: cache ready %s row_tiles=%u col_tiles=%u max_block_rows=%u pingpong_rows=%u packed=%zu cma=%d pingpong=%d\n",
                __func__,
                op_name != nullptr ? op_name : "(unnamed)",
                cached.row_tiles,
                cached.col_tiles,
                cached.max_block_rows,
                cached.pingpong_max_block_rows,
                cached.packed_weight.size(),
                cma_resident ? 1 : 0,
                use_pingpong ? 1 : 0);
    }

    npu_decode_gemv_layout act_layout = {};
    act_layout.col_tiles = cached.col_tiles;
    act_layout.act_bytes = cached.col_tiles * NPU_DECODE_GEMV_ACT_TILE_BYTES;
    if (act_layout.act_bytes > NPU_DECODE_GEMV_ACT_BUFFER_BYTES ||
            (act_layout.act_bytes % NPU_DECODE_GEMV_DMA_ALIGN) != 0) {
        return fail("activation layout exceeds act buffer");
    }
    const uint32_t max_block_rows = use_pingpong ? cached.pingpong_max_block_rows : cached.max_block_rows;
    const uint32_t max_row_tiles = npu_decode_ceil_div_u32(max_block_rows, NPU_DECODE_GEMV_M_TILE);
    const uint32_t max_weight_bytes = max_row_tiles * cached.bytes_per_row_tile;
    const uint32_t max_scale_bytes = max_row_tiles * cached.col_tiles * NPU_DECODE_GEMV_SCALE_TILE_BYTES;
    const bool hw_f32_mvout = dst_type == GGML_TYPE_F32 && npu_decode_hw_f32_mvout_enabled();
    const uint8_t output_precision = npu_decode_mvout_precision(hw_f32_mvout);
    const uint32_t block_output_bytes = npu_decode_host_output_bytes(max_block_rows, hw_f32_mvout);
    const uint32_t full_output_bytes = npu_decode_host_output_bytes(static_cast<uint32_t>(out_channels), hw_f32_mvout);
    const uint32_t output_cma_bytes = use_pingpong ? full_output_bytes : block_output_bytes;

    const int64_t alloc_start_us = collect_profile ? ggml_time_us() : 0;
    void * act_cma = npu_decode_mem_alloc(act_layout.act_bytes);
    void * weight_cma = cma_resident ? nullptr : npu_decode_mem_alloc(max_weight_bytes);
    void * scale_cma = npu_decode_w16a16_ensure_scale_ones_cma(cached, max_scale_bytes);
    bool scale_cma_owned = false;
    if (scale_cma == nullptr) {
        scale_cma = npu_decode_mem_alloc(max_scale_bytes);
        scale_cma_owned = scale_cma != nullptr;
    }
    void * out_cma = npu_decode_mem_alloc(output_cma_bytes);
    if (act_cma == nullptr || (!cma_resident && weight_cma == nullptr) ||
            scale_cma == nullptr || out_cma == nullptr) {
        npu_decode_mem_free(act_cma);
        npu_decode_mem_free(weight_cma);
        if (scale_cma_owned) {
            npu_decode_mem_free(scale_cma);
        }
        npu_decode_mem_free(out_cma);
        return fail("decode CMA allocation failed");
    }
    if (collect_profile) {
        profile.alloc_us = static_cast<uint64_t>(ggml_time_us() - alloc_start_us);
    }
    if (npu_decode_w4a16_debug_log_enabled()) {
        GGML_LOG_WARN("%s: cma allocated %s act=%u weight=%u scale=%u out=%u hw_f32=%d\n",
                __func__,
                op_name != nullptr ? op_name : "(unnamed)",
                act_layout.act_bytes,
                max_weight_bytes,
                max_scale_bytes,
                output_cma_bytes,
                hw_f32_mvout ? 1 : 0);
    }

    if (scale_cma_owned) {
        std::vector<uint8_t> scale_ones(max_scale_bytes, 0);
        const ggml_fp16_t fp16_one = ggml_fp32_to_fp16(1.0f);
        for (uint32_t i = 0; i + sizeof(ggml_fp16_t) <= max_scale_bytes; i += sizeof(ggml_fp16_t)) {
            std::memcpy(scale_ones.data() + i, &fp16_one, sizeof(fp16_one));
        }
        std::memcpy(scale_cma, scale_ones.data(), scale_ones.size());
    }

    if (collect_profile) {
        profile.max_block_rows = max_block_rows;
        profile.row_blocks_per_col =
            npu_decode_ceil_div_u32(static_cast<uint32_t>(out_channels), max_block_rows);
        profile.preloaded_host = true;
        profile.preloaded_cma = cma_resident;
        profile.pingpong = use_pingpong;
        profile.activation_bytes = static_cast<uint64_t>(act_layout.act_bytes) * static_cast<uint64_t>(n_cols);
        profile.scale_bytes = static_cast<uint64_t>(max_scale_bytes) * static_cast<uint64_t>(n_cols);
        profile.setup_us = static_cast<uint64_t>(ggml_time_us() - setup_start_us);
    }

    auto block_rows_for = [&](uint32_t row_base) {
        return std::min<uint32_t>(max_block_rows, static_cast<uint32_t>(out_channels) - row_base);
    };
    auto weight_offset_for = [&](uint32_t row_base) {
        return static_cast<size_t>(row_base / NPU_DECODE_GEMV_M_TILE) * cached.bytes_per_row_tile;
    };

    bool w16_pingpong_failed = false;
    for (int64_t col = 0; col < n_cols; ++col) {
        int64_t stage_start_us = collect_profile ? ggml_time_us() : 0;
        npu_decode_pack_activation(
                act_cma,
                act_layout.act_bytes,
                act_layout,
                act_data,
                act_type,
                act_nb0,
                act_nb1,
                nullptr,
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
        if (use_pingpong && !w16_pingpong_failed) {
            try {
                npu_decode_w16a16_pingpong_manual_run(
                        op_name,
                        act_cma,
                        scale_cma,
                        cached.cma_packed,
                        out_cma,
                        cached.col_tiles,
                        cached.bytes_per_row_tile,
                        max_block_rows,
                        static_cast<uint16_t>(out_channels),
                        static_cast<uint16_t>(k),
                        output_precision,
                        collect_profile ? &profile : nullptr);
            } catch (const std::exception & e) {
                if (npu_decode_w4a16_debug_log_enabled()) {
                    GGML_LOG_WARN("%s: W16 pingpong failed for %s: %s\n",
                            __func__, op_name != nullptr ? op_name : "(unnamed)", e.what());
                }
                w16_pingpong_failed = true;
                npu_decode_reset();
                stage_start_us = collect_profile ? ggml_time_us() : 0;
            }
            if (!w16_pingpong_failed) {
                if (collect_profile) {
                    profile.pingpong_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
                    profile.pingpong_calls += 1;
                    stage_start_us = ggml_time_us();
                }

                char * dst_col = reinterpret_cast<char *>(dst_data) + col * dst_nb1;
                if (dst_type == GGML_TYPE_F16) {
                    std::memcpy(dst_col, out_cma, static_cast<size_t>(out_channels) * sizeof(ggml_fp16_t));
                } else if (hw_f32_mvout) {
                    std::memcpy(reinterpret_cast<float *>(dst_col), out_cma, static_cast<size_t>(out_channels) * sizeof(float));
                } else {
                    npu_decode_convert_output_fp16_to_f32(
                            static_cast<const ggml_fp16_t *>(out_cma),
                            reinterpret_cast<float *>(dst_col),
                            static_cast<uint32_t>(out_channels));
                }
                if (collect_profile) {
                    profile.postprocess_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
                }
                continue;
            }
        }
        npu_decode_dma_mvin(
                act_cma,
                NPU_DECODE_GEMV_ACT_BASE,
                act_layout.act_bytes - 1u,
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
        npu_decode_dma_mvin(
                scale_cma,
                NPU_DECODE_GEMV_SCALE_BASE,
                max_scale_bytes - 1u,
                0,
                0,
                0,
                2,
                NPU_DECODE_GEMV_INPUT_TYPE_DATA,
                false,
                false,
                false,
                0,
                0,
                0);
        if (collect_profile) {
            profile.mvin_activation_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            profile.activation_mvin_calls += 1;
            stage_start_us = ggml_time_us();
        }

        uint32_t block_idx = 0;
        for (uint32_t row_base = 0; row_base < static_cast<uint32_t>(out_channels);
                row_base += block_rows_for(row_base), ++block_idx) {
            const uint32_t rows = block_rows_for(row_base);
            const uint32_t cur_weight_spm = NPU_DECODE_GEMV_PING_WEIGHT_BASE;
            const uint32_t weight_bytes =
                npu_decode_ceil_div_u32(rows, NPU_DECODE_GEMV_M_TILE) * cached.bytes_per_row_tile;
            const size_t weight_offset = weight_offset_for(row_base);
            stage_start_us = collect_profile ? ggml_time_us() : 0;
            void * weight_mvin_ptr = nullptr;
            if (cma_resident) {
                weight_mvin_ptr = static_cast<uint8_t *>(cached.cma_packed) + weight_offset;
            } else {
                std::memcpy(weight_cma, cached.packed_weight.data() + weight_offset, weight_bytes);
                weight_mvin_ptr = weight_cma;
                if (collect_profile) {
                    profile.host_copy_weight_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
                    stage_start_us = ggml_time_us();
                }
            }
            npu_decode_dma_mvin(
                    weight_mvin_ptr,
                    cur_weight_spm,
                    weight_bytes - 1u,
                    0,
                    0,
                    0,
                    2,
                    NPU_DECODE_GEMV_INPUT_TYPE_WEIGHT,
                    false,
                    false,
                    false,
                    0,
                    0,
                    0);
            if (npu_decode_w4a16_debug_log_enabled() && (block_idx == 0 || row_base + rows >= static_cast<uint32_t>(out_channels))) {
                GGML_LOG_WARN("%s: block %s idx=%u row_base=%u rows=%u weight_bytes=%u\n",
                        __func__,
                        op_name != nullptr ? op_name : "(unnamed)",
                        block_idx,
                        row_base,
                        rows,
                        weight_bytes);
            }
            if (collect_profile) {
                profile.mvin_weight_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
                profile.weight_mvin_calls += 1;
            }

            stage_start_us = collect_profile ? ggml_time_us() : 0;
            const uint64_t flow = npu_decode_make_bypass_output_flow(static_cast<uint16_t>(rows));
            npu_decode_trace_matvec_flow(
                    "lm_head_w16",
                    op_name,
                    block_idx,
                    row_base,
                    rows,
                    static_cast<uint32_t>(k),
                    NPU_DECODE_GEMV_MODE_W16A16,
                    cur_weight_spm,
                    NPU_DECODE_GEMV_ACT_BASE,
                    NPU_DECODE_GEMV_OUTPUT_BASE,
                    NPU_DECODE_GEMV_SCALE_BASE,
                    flow);
            npu_decode_matvec_decode_flow_run(
                    cur_weight_spm,
                    NPU_DECODE_GEMV_ACT_BASE,
                    static_cast<uint16_t>(k),
                    static_cast<uint16_t>(rows),
                    NPU_DECODE_GEMV_OUTPUT_BASE,
                    NPU_DECODE_GEMV_SCALE_BASE,
                    NPU_DECODE_GEMV_MODE_W16A16,
                    flow);
            if (collect_profile) {
                profile.gemv_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
                profile.gemv_calls += 1;
                profile.weight_bytes += weight_bytes;
                stage_start_us = ggml_time_us();
            }

            npu_decode_dma_mvout(
                    out_cma,
                    NPU_DECODE_GEMV_OUTPUT_BASE,
                    0,
                    rows - 1u,
                    1,
                    1,
                    output_precision,
                    1,
                    false,
                    false,
                    0,
                    0);
            if (collect_profile) {
                profile.mvout_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
                profile.mvout_calls += 1;
                profile.output_bytes += npu_decode_host_output_bytes(rows, hw_f32_mvout);
                stage_start_us = ggml_time_us();
            }

            char * dst_col = reinterpret_cast<char *>(dst_data) + col * dst_nb1;
            if (dst_type == GGML_TYPE_F16) {
                std::memcpy(
                        dst_col + static_cast<size_t>(row_base) * sizeof(ggml_fp16_t),
                        out_cma,
                        static_cast<size_t>(rows) * sizeof(ggml_fp16_t));
            } else if (hw_f32_mvout) {
                std::memcpy(
                        reinterpret_cast<float *>(dst_col) + row_base,
                        out_cma,
                        static_cast<size_t>(rows) * sizeof(float));
            } else {
                npu_decode_convert_output_fp16_to_f32(
                        static_cast<const ggml_fp16_t *>(out_cma),
                        reinterpret_cast<float *>(dst_col) + row_base,
                        rows);
            }
            if (collect_profile) {
                profile.postprocess_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            }
        }
    }

    const int64_t cleanup_start_us = collect_profile ? ggml_time_us() : 0;
    npu_decode_mem_free(act_cma);
    npu_decode_mem_free(weight_cma);
    if (scale_cma_owned) {
        npu_decode_mem_free(scale_cma);
    }
    npu_decode_mem_free(out_cma);
    if (collect_profile) {
        profile.cleanup_us = static_cast<uint64_t>(ggml_time_us() - cleanup_start_us);
        profile.total_us = static_cast<uint64_t>(ggml_time_us() - total_start_us);
        npu_decode_profile_write(profile);
    }
    return true;
}

bool ggml_backend_npu_decode_swiglu_ffn_w4a16_ex(
        const char * op_name,
        const struct ggml_npu_decode_awq_view * gate,
        const struct ggml_npu_decode_awq_view * up,
        const struct ggml_npu_decode_awq_view * down,
        const void * act_data,
        int act_type,
        int64_t act_nb0,
        int64_t act_nb1,
        void * dst_data,
        int dst_type,
        int64_t dst_nb1) {
    using namespace ggml_npu;

    auto fail = [op_name](const char * reason) {
        if (npu_decode_w4a16_debug_log_enabled()) {
            GGML_LOG_WARN("%s: reject %s: %s\n", __func__, op_name != nullptr ? op_name : "(unnamed)", reason);
        }
        return false;
    };

    if (gate == nullptr || up == nullptr || down == nullptr ||
            act_data == nullptr || dst_data == nullptr) {
        return fail("null pointer");
    }
    if (act_type != GGML_TYPE_F16 && act_type != GGML_TYPE_F32) {
        return fail("activation tensor must be F16 or F32");
    }
    if (dst_type != GGML_TYPE_F16 && dst_type != GGML_TYPE_F32) {
        return fail("destination tensor must be F16 or F32");
    }
    const bool collect_profile = npu_decode_profile_enabled();
    const int64_t total_start_us = collect_profile ? ggml_time_us() : 0;
    const int64_t setup_start_us = collect_profile ? total_start_us : 0;
    int64_t stage_start_us = collect_profile ? total_start_us : 0;
    npu_decode_gemv_profile_record profile = {};
    profile.profile_kind = "aicas_decode_w4a16_swiglu_ffn";
    profile.gemv_mode = "w4a16";
    profile.op_role = "swiglu_ffn";
    profile.op_name = op_name != nullptr ? op_name : "";
    profile.dst_type = dst_type;
    profile.k = gate != nullptr ? gate->k : 0;
    profile.n_cols = 1;
    profile.out_channels = down != nullptr ? down->out_channels : 0;

    const bool hw_f32_mvout = dst_type == GGML_TYPE_F32 && npu_decode_hw_f32_mvout_enabled();
    const uint8_t output_precision = npu_decode_mvout_precision(hw_f32_mvout);
    if (gate->k <= 0 || up->k <= 0 || down->k <= 0 ||
            gate->out_channels <= 0 || up->out_channels <= 0 || down->out_channels <= 0) {
        return fail("invalid dimensions");
    }
    if (gate->k != up->k || gate->out_channels != up->out_channels ||
            down->k != gate->out_channels) {
        return fail("incompatible SwiGLU dimensions");
    }
    if (gate->packed_k != (gate->k + 1) / 2 ||
            up->packed_k != (up->k + 1) / 2 ||
            down->packed_k != (down->k + 1) / 2) {
        return fail("invalid packed dimensions");
    }
    if (gate->k > std::numeric_limits<uint16_t>::max() ||
            gate->out_channels > std::numeric_limits<uint16_t>::max() ||
            down->out_channels > std::numeric_limits<uint16_t>::max()) {
        return fail("dimension exceeds uint16 range");
    }
    if (gate->out_channels > 4096) {
        return fail("SwiGLU intermediate exceeds stream buffer");
    }
    if (gate->smooth_scale == nullptr || up->smooth_scale == nullptr || down->smooth_scale == nullptr ||
            gate->smooth_scale_len != static_cast<size_t>(gate->k) ||
            up->smooth_scale_len != static_cast<size_t>(up->k) ||
            down->smooth_scale_len != static_cast<size_t>(down->k)) {
        return fail("invalid smooth scale");
    }
    const npu_decode_preloaded_tensor * gate_preloaded = npu_decode_lookup_preloaded_tensor(
            gate->weight_name, gate->packed_k, gate->out_channels, gate->q4_nb1, gate->k);
    const npu_decode_preloaded_tensor * up_preloaded = npu_decode_lookup_preloaded_tensor(
            up->weight_name, up->packed_k, up->out_channels, up->q4_nb1, up->k);
    const npu_decode_preloaded_tensor * down_preloaded = npu_decode_lookup_preloaded_tensor(
            down->weight_name, down->packed_k, down->out_channels, down->q4_nb1, down->k);
    if (gate_preloaded == nullptr || up_preloaded == nullptr || down_preloaded == nullptr ||
            gate_preloaded->blocks.empty() || up_preloaded->blocks.empty() || down_preloaded->blocks.empty()) {
        return fail("decode tensors are not preloaded");
    }
    if (!gate_preloaded->zero_validated || !up_preloaded->zero_validated || !down_preloaded->zero_validated) {
        return fail("decode tensor zero points are not validated");
    }
    if (gate_preloaded->act_scale.size() != gate_preloaded->max_layout.act_bytes ||
            up_preloaded->act_scale.size() != up_preloaded->max_layout.act_bytes ||
            down_preloaded->act_scale.size() != down_preloaded->max_layout.act_bytes) {
        return fail("preloaded act scale size mismatch");
    }

    const npu_decode_gemv_layout gate_layout = gate_preloaded->max_layout;
    const npu_decode_gemv_layout up_layout = up_preloaded->max_layout;
    const npu_decode_gemv_layout down_layout = down_preloaded->max_layout;
    const uint32_t act_base = NPU_DECODE_GEMV_ACT_BASE;
    const uint32_t input_act_payload_bytes = std::max(
            npu_decode_act_payload_bytes(gate_layout),
            npu_decode_act_payload_bytes(up_layout));
    const uint32_t swiglu_act_base = npu_decode_align_up_u32(input_act_payload_bytes, NPU_DECODE_GEMV_LINE_BYTES);
    if (swiglu_act_base + npu_decode_act_payload_bytes(down_layout) > NPU_DECODE_GEMV_ACT_BUFFER_BYTES) {
        return fail("SwiGLU activation does not fit act buffer");
    }
    const uint32_t output_spm_bytes = npu_decode_align_up_u32(
            static_cast<uint32_t>(down->out_channels) * static_cast<uint32_t>(sizeof(ggml_fp16_t)),
            NPU_DECODE_GEMV_LINE_BYTES);
    const uint32_t host_output_bytes = npu_decode_host_output_bytes(static_cast<uint32_t>(down->out_channels), hw_f32_mvout);
    if (output_spm_bytes > NPU_DECODE_GEMV_SPM_BYTES) {
        return fail("decode output exceeds output SPM capacity");
    }
    bool preloaded_cma = false;

    if (!npu_decode_ensure_overlay_active(op_name)) {
        return fail("decode overlay switch failed");
    }
    auto tensor_pingpong_cma_resident = [](const npu_decode_preloaded_tensor * tensor) {
        return tensor != nullptr &&
               tensor->pingpong_available &&
               tensor->pingpong_scale_cma_bytes == tensor->pingpong_layout.scale_bytes &&
               tensor->pingpong_weight_cma_bytes == tensor->pingpong_layout.weight_bytes;
    };
    bool ffn_pingpong_cma =
        npu_decode_fused_ffn_pingpong_enabled() &&
        tensor_pingpong_cma_resident(gate_preloaded) &&
        tensor_pingpong_cma_resident(up_preloaded) &&
        tensor_pingpong_cma_resident(down_preloaded);
    if (preloaded_cma || ffn_pingpong_cma) {
        npu_decode_update_runtime_heap_env();
    }
    if (npu_decode_init_with_cma_retry(op_name, true) != 0) {
        return fail("decode runtime init failed");
    }
    (void) npu_decode_reserve_preloaded_ffn_pingpong_cma_after_init({
        const_cast<npu_decode_preloaded_tensor *>(gate_preloaded),
        const_cast<npu_decode_preloaded_tensor *>(up_preloaded),
        const_cast<npu_decode_preloaded_tensor *>(down_preloaded),
    });
    preloaded_cma = true;
    for (const npu_decode_preloaded_tensor * tensor : { gate_preloaded, up_preloaded, down_preloaded }) {
        for (const npu_decode_preloaded_block & block : tensor->blocks) {
            if (block.cma_bytes == 0 || block.cma_bytes != block.packed.size()) {
                preloaded_cma = false;
                break;
            }
        }
        if (!preloaded_cma) {
            break;
        }
    }
    ffn_pingpong_cma =
        npu_decode_fused_ffn_pingpong_enabled() &&
        tensor_pingpong_cma_resident(gate_preloaded) &&
        tensor_pingpong_cma_resident(up_preloaded) &&
        tensor_pingpong_cma_resident(down_preloaded);
    void * decode_cma_base = (preloaded_cma || ffn_pingpong_cma) ? npu_decode_memory_base() : nullptr;
    if (decode_cma_base == nullptr) {
        preloaded_cma = false;
        ffn_pingpong_cma = false;
    }

    uint32_t max_packed_bytes = 0;
    for (const npu_decode_preloaded_tensor * tensor : { gate_preloaded, up_preloaded, down_preloaded }) {
        for (const npu_decode_preloaded_block & block : tensor->blocks) {
            max_packed_bytes = std::max(max_packed_bytes, block.layout.packed_bytes);
        }
    }

    const uint32_t act_cma_bytes = std::max(input_act_payload_bytes, down_layout.act_bytes);
    void * act_cma = npu_decode_mem_alloc(act_cma_bytes);
    void * packed_cma = (preloaded_cma || ffn_pingpong_cma) ? nullptr : npu_decode_mem_alloc(max_packed_bytes);
    void * out_cma = npu_decode_mem_alloc(host_output_bytes);
    if (act_cma == nullptr || (!preloaded_cma && !ffn_pingpong_cma && packed_cma == nullptr) || out_cma == nullptr) {
        npu_decode_mem_free(act_cma);
        npu_decode_mem_free(packed_cma);
        npu_decode_mem_free(out_cma);
        return fail("decode CMA allocation failed");
    }
    if (collect_profile) {
        profile.max_block_rows = std::max(gate_preloaded->max_block_rows, down_preloaded->max_block_rows);
        profile.row_blocks_per_col =
            static_cast<uint32_t>(gate_preloaded->blocks.size() + down_preloaded->blocks.size());
        profile.preloaded_host = true;
        profile.preloaded_cma = preloaded_cma || ffn_pingpong_cma;
        profile.pingpong = ffn_pingpong_cma;
        if (npu_decode_fused_ffn_pingpong_enabled() && !ffn_pingpong_cma) {
            profile.fallback_reason = "ffn pingpong CMA not resident";
        }
        profile.activation_bytes =
            npu_decode_act_payload_bytes(gate_layout) +
            npu_decode_act_payload_bytes(up_layout) +
            npu_decode_act_payload_bytes(down_layout);
        profile.setup_us = static_cast<uint64_t>(ggml_time_us() - setup_start_us);
    }

    auto cleanup = [&]() {
        npu_decode_mem_free(act_cma);
        if (!preloaded_cma && !ffn_pingpong_cma) {
            npu_decode_mem_free(packed_cma);
        }
        npu_decode_mem_free(out_cma);
    };

    stage_start_us = collect_profile ? ggml_time_us() : 0;
    npu_decode_reset();
    if (collect_profile) {
        profile.reset_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
    }

    if (gate_preloaded->blocks.size() != up_preloaded->blocks.size()) {
        cleanup();
        return fail("gate/up block count mismatch");
    }

    auto pingpong_scale_bytes_for =
        [](const npu_decode_preloaded_tensor * tensor, uint32_t block_rows) {
            if (tensor->pingpong_layout.scale_bytes == 0) {
                return 0u;
            }
            return npu_decode_ceil_div_u32(block_rows, NPU_DECODE_GEMV_M_TILE) *
                tensor->pingpong_layout.col_tiles * NPU_DECODE_GEMV_SCALE_TILE_BYTES;
        };
    auto pingpong_weight_bytes_for =
        [](const npu_decode_preloaded_tensor * tensor, uint32_t block_rows) {
            const uint32_t bytes_per_row_tile =
                tensor->pingpong_layout.col_tiles * NPU_DECODE_GEMV_WEIGHT_TILE_BYTES;
            return npu_decode_ceil_div_u32(block_rows, NPU_DECODE_GEMV_M_TILE) * bytes_per_row_tile;
        };
    auto pingpong_weight_ptr_for =
        [decode_cma_base](const npu_decode_preloaded_tensor * tensor, uint32_t row_base) {
            const uint32_t bytes_per_row_tile =
                tensor->pingpong_layout.col_tiles * NPU_DECODE_GEMV_WEIGHT_TILE_BYTES;
            const uint32_t offset =
                (row_base / NPU_DECODE_GEMV_M_TILE) * bytes_per_row_tile;
            return static_cast<uint8_t *>(decode_cma_base) + tensor->pingpong_weight_cma_offset + offset;
        };
    auto pingpong_scale_ptr_for =
        [decode_cma_base](const npu_decode_preloaded_tensor * tensor, uint32_t row_base) {
            const uint32_t offset =
                (row_base / NPU_DECODE_GEMV_M_TILE) *
                tensor->pingpong_layout.col_tiles * NPU_DECODE_GEMV_SCALE_TILE_BYTES;
            return static_cast<uint8_t *>(decode_cma_base) + tensor->pingpong_scale_cma_offset + offset;
        };
    auto mvin_pingpong_scale =
        [&](const npu_decode_preloaded_tensor * tensor, uint32_t row_base, uint32_t block_rows) {
            const uint32_t scale_bytes = pingpong_scale_bytes_for(tensor, block_rows);
            if (scale_bytes == 0) {
                return;
            }
            stage_start_us = collect_profile ? ggml_time_us() : 0;
            npu_decode_dma_mvin(
                    pingpong_scale_ptr_for(tensor, row_base),
                    NPU_DECODE_GEMV_SCALE_BASE,
                    scale_bytes - 1u,
                    0,
                    0,
                    0,
                    2,
                    NPU_DECODE_GEMV_INPUT_TYPE_DATA,
                    false,
                    false,
                    false,
                    0,
                    0,
                    0);
            if (collect_profile) {
                profile.scale_mvin_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
                profile.scale_mvin_calls += 1;
                profile.scale_bytes += scale_bytes;
            }
        };
    auto pingpong_scale_addr_for =
        [](const npu_decode_preloaded_tensor * tensor, uint32_t weight_spm) {
            return tensor->pingpong_layout.scale_bytes == 0 ?
                weight_spm : static_cast<uint32_t>(NPU_DECODE_GEMV_SCALE_BASE);
        };
    auto mvin_pingpong_weight_sync =
        [&](const npu_decode_preloaded_tensor * tensor, uint32_t row_base, uint32_t block_rows, uint32_t weight_spm) {
            const uint32_t weight_bytes = pingpong_weight_bytes_for(tensor, block_rows);
            stage_start_us = collect_profile ? ggml_time_us() : 0;
            npu_decode_dma_mvin(
                    pingpong_weight_ptr_for(tensor, row_base),
                    weight_spm,
                    weight_bytes - 1u,
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
                profile.weight_bytes += weight_bytes;
            }
        };
    auto issue_pingpong_weight_prefetch =
        [&](const npu_decode_preloaded_tensor * tensor,
                uint32_t row_base,
                uint32_t block_rows,
                uint32_t weight_spm,
                int64_t * issue_done_us) {
            const uint32_t weight_bytes = pingpong_weight_bytes_for(tensor, block_rows);
            MvinConfig next_mvin = npu_decode_make_mvin_cfg(
                    pingpong_weight_ptr_for(tensor, row_base),
                    weight_spm,
                    weight_bytes,
                    NPU_DECODE_GEMV_INPUT_TYPE_WEIGHT,
                    1);
            stage_start_us = collect_profile ? ggml_time_us() : 0;
            npu_decode_dma_mvin_async(0, &next_mvin);
            if (collect_profile) {
                *issue_done_us = ggml_time_us();
                profile.pingpong_prefetch_issue_us += static_cast<uint64_t>(*issue_done_us - stage_start_us);
                profile.pingpong_async_mvin_calls += 1;
                profile.weight_bytes += weight_bytes;
            }
        };
    auto wait_pingpong_prefetch = [&](int64_t issue_done_us) {
        if (collect_profile) {
            profile.pingpong_prefetch_window_us += static_cast<uint64_t>(ggml_time_us() - issue_done_us);
            stage_start_us = ggml_time_us();
        }
        npu_decode_dma_wait_mvin(1u << 0);
        if (collect_profile) {
            profile.pingpong_prefetch_wait_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
        }
    };

    if (ffn_pingpong_cma) {
        const uint32_t gate_spm = NPU_DECODE_GEMV_PING_WEIGHT_BASE;
        const uint32_t up_spm = NPU_DECODE_GEMV_PONG_WEIGHT_BASE;
        const uint32_t hidden_rows = static_cast<uint32_t>(gate->out_channels);
        const uint32_t gate_up_max_rows = std::min(
                npu_decode_pingpong_max_block_rows(gate_preloaded->pingpong_layout),
                npu_decode_pingpong_max_block_rows(up_preloaded->pingpong_layout));
        const uint32_t down_rows_total = static_cast<uint32_t>(down->out_channels);
        const uint32_t down_max_rows = npu_decode_pingpong_max_block_rows(down_preloaded->pingpong_layout);
        if (gate_up_max_rows == 0 || down_max_rows == 0) {
            cleanup();
            return fail("SwiGLU pingpong row block does not fit");
        }
        const uint32_t up_act_base = npu_decode_align_up_u32(
                swiglu_act_base + hidden_rows * static_cast<uint32_t>(sizeof(ggml_fp16_t)),
                NPU_DECODE_GEMV_LINE_BYTES);
        if (up_act_base + npu_decode_act_payload_bytes(up_layout) > NPU_DECODE_GEMV_ACT_BUFFER_BYTES) {
            cleanup();
            return fail("SwiGLU pingpong activation layout does not fit");
        }
        if (collect_profile) {
            profile.row_blocks_per_col =
                npu_decode_ceil_div_u32(hidden_rows, gate_up_max_rows) +
                npu_decode_ceil_div_u32(down_rows_total, down_max_rows);
        }

        stage_start_us = collect_profile ? ggml_time_us() : 0;
        npu_decode_pack_activation(
                act_cma,
                npu_decode_act_payload_bytes(gate_layout),
                gate_layout,
                act_data,
                act_type,
                act_nb0,
                act_nb1,
                gate_preloaded->act_scale.empty() ? nullptr : gate_preloaded->act_scale.data(),
                gate->k,
                0);
        if (collect_profile) {
            profile.activation_preprocess_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            stage_start_us = ggml_time_us();
        }
        npu_decode_dma_mvin(
                act_cma,
                act_base,
                npu_decode_act_payload_bytes(gate_layout) - 1u,
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

        stage_start_us = collect_profile ? ggml_time_us() : 0;
        npu_decode_pack_activation(
                act_cma,
                npu_decode_act_payload_bytes(up_layout),
                up_layout,
                act_data,
                act_type,
                act_nb0,
                act_nb1,
                up_preloaded->act_scale.empty() ? nullptr : up_preloaded->act_scale.data(),
                up->k,
                0);
        if (collect_profile) {
            profile.activation_preprocess_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            stage_start_us = ggml_time_us();
        }
        npu_decode_dma_mvin(
                act_cma,
                up_act_base,
                npu_decode_act_payload_bytes(up_layout) - 1u,
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

        const uint32_t first_gate_rows = std::min(gate_up_max_rows, hidden_rows);
        mvin_pingpong_weight_sync(gate_preloaded, 0, first_gate_rows, gate_spm);

        uint32_t gate_up_block_idx = 0;
        for (uint32_t row_base = 0; row_base < hidden_rows; row_base += std::min(gate_up_max_rows, hidden_rows - row_base), ++gate_up_block_idx) {
            const uint32_t block_rows = std::min(gate_up_max_rows, hidden_rows - row_base);
            if (block_rows > 4096) {
                cleanup();
                return fail("SwiGLU block exceeds stream buffer");
            }

            mvin_pingpong_scale(gate_preloaded, row_base, block_rows);

            int64_t up_prefetch_issue_done_us = 0;
            issue_pingpong_weight_prefetch(up_preloaded, row_base, block_rows, up_spm, &up_prefetch_issue_done_us);
            stage_start_us = collect_profile ? ggml_time_us() : 0;
            npu_decode_matvec_silu_run(
                    gate_spm,
                    act_base,
                    static_cast<uint16_t>(gate->k),
                    static_cast<uint16_t>(block_rows),
                    static_cast<uint16_t>(NPU_DECODE_GEMV_OUTPUT_BASE),
                    static_cast<uint16_t>(pingpong_scale_addr_for(gate_preloaded, gate_spm)));
            if (collect_profile) {
                profile.gemv_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
                profile.gemv_calls += 1;
                profile.pingpong_gemv_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
                profile.pingpong_block_calls += 1;
            }
            wait_pingpong_prefetch(up_prefetch_issue_done_us);

            mvin_pingpong_scale(up_preloaded, row_base, block_rows);

            bool gate_prefetch_started = false;
            int64_t gate_prefetch_issue_done_us = 0;
            const uint32_t next_row_base = row_base + block_rows;
            if (next_row_base < hidden_rows) {
                const uint32_t next_block_rows = std::min(gate_up_max_rows, hidden_rows - next_row_base);
                issue_pingpong_weight_prefetch(
                        gate_preloaded,
                        next_row_base,
                        next_block_rows,
                        gate_spm,
                        &gate_prefetch_issue_done_us);
                gate_prefetch_started = true;
            }
            const uint64_t flow = npu_decode_flow_make(
                    NPU_DECODE_SRC_GEMV_STREAM,
                    NPU_DECODE_SRC_STREAM_BUFFER,
                    NPU_DECODE_UNARY_BYPASS,
                    NPU_DECODE_BINARY_FP16_MUL,
                    NPU_DECODE_REDUCE_BYPASS,
                    NPU_DECODE_DST_ACT_BUFFER,
                    0,
                    0,
                    static_cast<uint16_t>(block_rows),
                    0);
            npu_decode_trace_matvec_flow(
                    "fused_ffn_up_pingpong",
                    op_name,
                    gate_up_block_idx,
                    row_base,
                    block_rows,
                    static_cast<uint32_t>(up->k),
                    NPU_DECODE_GEMV_MODE_W4A16,
                    up_spm,
                    up_act_base,
                    static_cast<uint16_t>(swiglu_act_base + row_base * sizeof(ggml_fp16_t)),
                    pingpong_scale_addr_for(up_preloaded, up_spm),
                    flow);
            stage_start_us = collect_profile ? ggml_time_us() : 0;
            npu_decode_matvec_decode_flow_run(
                    up_spm,
                    up_act_base,
                    static_cast<uint16_t>(up->k),
                    static_cast<uint16_t>(block_rows),
                    static_cast<uint16_t>(swiglu_act_base + row_base * sizeof(ggml_fp16_t)),
                    static_cast<uint16_t>(pingpong_scale_addr_for(up_preloaded, up_spm)),
                    NPU_DECODE_GEMV_MODE_W4A16,
                    flow);
            if (collect_profile) {
                profile.gemv_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
                profile.gemv_calls += 1;
                profile.pingpong_gemv_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
                profile.pingpong_block_calls += 1;
            }
            if (gate_prefetch_started) {
                wait_pingpong_prefetch(gate_prefetch_issue_done_us);
            }
        }

        if (down_preloaded->act_scale.size() != down_layout.act_bytes) {
            cleanup();
            return fail("down act scale size mismatch");
        }
        stage_start_us = collect_profile ? ggml_time_us() : 0;
        std::memcpy(act_cma, down_preloaded->act_scale.data(), down_layout.act_bytes);
        if (collect_profile) {
            profile.activation_preprocess_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            stage_start_us = ggml_time_us();
        }
        npu_decode_dma_mvin(
                act_cma,
                swiglu_act_base + down_layout.act_bytes,
                down_layout.act_bytes - 1u,
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

        const uint32_t down_ping_spm = NPU_DECODE_GEMV_PING_WEIGHT_BASE;
        const uint32_t down_pong_spm = NPU_DECODE_GEMV_PONG_WEIGHT_BASE;
        const uint32_t first_down_rows = std::min(down_max_rows, down_rows_total);
        mvin_pingpong_weight_sync(down_preloaded, 0, first_down_rows, down_ping_spm);
        uint32_t down_block_idx = 0;
        for (uint32_t row_base = 0; row_base < down_rows_total; row_base += std::min(down_max_rows, down_rows_total - row_base), ++down_block_idx) {
            const uint32_t block_rows = std::min(down_max_rows, down_rows_total - row_base);
            const uint32_t cur_spm = (down_block_idx & 1u) ? down_pong_spm : down_ping_spm;
            const uint32_t next_spm = (down_block_idx & 1u) ? down_ping_spm : down_pong_spm;
            mvin_pingpong_scale(down_preloaded, row_base, block_rows);

            bool prefetch_started = false;
            int64_t prefetch_issue_done_us = 0;
            const uint32_t next_row_base = row_base + block_rows;
            if (next_row_base < down_rows_total) {
                const uint32_t next_block_rows = std::min(down_max_rows, down_rows_total - next_row_base);
                issue_pingpong_weight_prefetch(
                        down_preloaded,
                        next_row_base,
                        next_block_rows,
                        next_spm,
                        &prefetch_issue_done_us);
                prefetch_started = true;
            }

            const uint64_t flow = npu_decode_make_bypass_output_flow(static_cast<uint16_t>(block_rows));
            npu_decode_trace_matvec_flow(
                    "fused_ffn_down_pingpong",
                    op_name,
                    down_block_idx,
                    row_base,
                    block_rows,
                    static_cast<uint32_t>(down->k),
                    NPU_DECODE_GEMV_MODE_W4A16,
                    cur_spm,
                    swiglu_act_base,
                    static_cast<uint16_t>(row_base * sizeof(ggml_fp16_t)),
                    pingpong_scale_addr_for(down_preloaded, cur_spm),
                    flow);
            stage_start_us = collect_profile ? ggml_time_us() : 0;
            npu_decode_matvec_decode_flow_run(
                    cur_spm,
                    swiglu_act_base,
                    static_cast<uint16_t>(down->k),
                    static_cast<uint16_t>(block_rows),
                    static_cast<uint16_t>(row_base * sizeof(ggml_fp16_t)),
                    static_cast<uint16_t>(pingpong_scale_addr_for(down_preloaded, cur_spm)),
                    NPU_DECODE_GEMV_MODE_W4A16,
                    flow);
            if (collect_profile) {
                profile.gemv_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
                profile.gemv_calls += 1;
                profile.pingpong_gemv_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
                profile.pingpong_block_calls += 1;
            }
            if (prefetch_started) {
                wait_pingpong_prefetch(prefetch_issue_done_us);
            }
        }
        if (collect_profile) {
            profile.pingpong_calls += 1;
            profile.pingpong_us =
                profile.pingpong_initial_mvin_us +
                profile.pingpong_prefetch_issue_us +
                profile.pingpong_prefetch_wait_us +
                profile.pingpong_gemv_us +
                profile.pingpong_mvout_us;
        }
    } else {
    for (size_t i = 0; i < gate_preloaded->blocks.size(); ++i) {
        const npu_decode_preloaded_block & gate_block = gate_preloaded->blocks[i];
        const npu_decode_preloaded_block & up_block = up_preloaded->blocks[i];
        if (gate_block.row_base != up_block.row_base || gate_block.block_rows != up_block.block_rows) {
            cleanup();
            return fail("gate/up block layout mismatch");
        }
        if (gate_block.block_rows > 4096) {
            cleanup();
            return fail("SwiGLU block exceeds stream buffer");
        }
        if (gate_block.packed.size() != gate_block.layout.packed_bytes ||
                up_block.packed.size() != up_block.layout.packed_bytes ||
                gate_block.layout.packed_bytes > max_packed_bytes ||
                up_block.layout.packed_bytes > max_packed_bytes) {
                cleanup();
                return fail("preloaded block size mismatch");
        }

        stage_start_us = collect_profile ? ggml_time_us() : 0;
        npu_decode_pack_activation(
                act_cma,
                npu_decode_act_payload_bytes(gate_layout),
                gate_layout,
                act_data,
                act_type,
                act_nb0,
                act_nb1,
                gate_preloaded->act_scale.empty() ? nullptr : gate_preloaded->act_scale.data(),
                gate->k,
                0);
        if (collect_profile) {
            profile.activation_preprocess_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            stage_start_us = ggml_time_us();
        }
        npu_decode_dma_mvin(
                act_cma,
                act_base,
                npu_decode_act_payload_bytes(gate_layout) - 1u,
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

        void * gate_weight_mvin_ptr = packed_cma;
        if (preloaded_cma && gate_block.cma_bytes == gate_block.layout.packed_bytes) {
            gate_weight_mvin_ptr = static_cast<uint8_t *>(decode_cma_base) + gate_block.cma_offset;
        } else {
            stage_start_us = collect_profile ? ggml_time_us() : 0;
            std::memcpy(packed_cma, gate_block.packed.data(), gate_block.layout.packed_bytes);
            if (collect_profile) {
                profile.host_copy_weight_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            }
        }
        stage_start_us = collect_profile ? ggml_time_us() : 0;
        npu_decode_dma_mvin(
                gate_weight_mvin_ptr,
                0,
                gate_block.layout.packed_bytes - 1u,
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
            profile.weight_bytes += gate_block.layout.packed_bytes;
            stage_start_us = ggml_time_us();
        }
        npu_decode_matvec_silu_run(
                gate_block.layout.weight_base,
                act_base,
                static_cast<uint16_t>(gate->k),
                static_cast<uint16_t>(gate_block.block_rows),
                static_cast<uint16_t>(NPU_DECODE_GEMV_OUTPUT_BASE),
                static_cast<uint16_t>(gate_block.layout.weight_base));
        if (collect_profile) {
            profile.gemv_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            profile.gemv_calls += 1;
        }

        stage_start_us = collect_profile ? ggml_time_us() : 0;
        npu_decode_pack_activation(
                act_cma,
                npu_decode_act_payload_bytes(up_layout),
                up_layout,
                act_data,
                act_type,
                act_nb0,
                act_nb1,
                up_preloaded->act_scale.empty() ? nullptr : up_preloaded->act_scale.data(),
                up->k,
                0);
        if (collect_profile) {
            profile.activation_preprocess_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            stage_start_us = ggml_time_us();
        }
        npu_decode_dma_mvin(
                act_cma,
                act_base,
                npu_decode_act_payload_bytes(up_layout) - 1u,
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

        void * up_weight_mvin_ptr = packed_cma;
        if (preloaded_cma && up_block.cma_bytes == up_block.layout.packed_bytes) {
            up_weight_mvin_ptr = static_cast<uint8_t *>(decode_cma_base) + up_block.cma_offset;
        } else {
            stage_start_us = collect_profile ? ggml_time_us() : 0;
            std::memcpy(packed_cma, up_block.packed.data(), up_block.layout.packed_bytes);
            if (collect_profile) {
                profile.host_copy_weight_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            }
        }
        stage_start_us = collect_profile ? ggml_time_us() : 0;
        npu_decode_dma_mvin(
                up_weight_mvin_ptr,
                0,
                up_block.layout.packed_bytes - 1u,
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
            profile.weight_bytes += up_block.layout.packed_bytes;
        }
        const uint64_t flow = npu_decode_flow_make(
                NPU_DECODE_SRC_GEMV_STREAM,
                NPU_DECODE_SRC_STREAM_BUFFER,
                NPU_DECODE_UNARY_BYPASS,
                NPU_DECODE_BINARY_FP16_MUL,
                NPU_DECODE_REDUCE_BYPASS,
                NPU_DECODE_DST_ACT_BUFFER,
                0,
                0,
                static_cast<uint16_t>(up_block.block_rows),
                0);
        npu_decode_trace_matvec_flow(
                "fused_ffn_up",
                op_name,
                0,
                static_cast<uint32_t>(up_block.row_base),
                static_cast<uint32_t>(up_block.block_rows),
                static_cast<uint32_t>(up->k),
                NPU_DECODE_GEMV_MODE_W4A16,
                up_block.layout.weight_base,
                act_base,
                static_cast<uint16_t>(swiglu_act_base + up_block.row_base * sizeof(ggml_fp16_t)),
                up_block.layout.weight_base,
                flow);
        stage_start_us = collect_profile ? ggml_time_us() : 0;
        npu_decode_matvec_decode_flow_run(
                up_block.layout.weight_base,
                act_base,
                static_cast<uint16_t>(up->k),
                static_cast<uint16_t>(up_block.block_rows),
                static_cast<uint16_t>(swiglu_act_base + up_block.row_base * sizeof(ggml_fp16_t)),
                static_cast<uint16_t>(up_block.layout.weight_base),
                NPU_DECODE_GEMV_MODE_W4A16,
                flow);
        if (collect_profile) {
            profile.gemv_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            profile.gemv_calls += 1;
        }
    }

    if (down_preloaded->act_scale.size() != down_layout.act_bytes) {
        cleanup();
        return fail("down act scale size mismatch");
    }
    stage_start_us = collect_profile ? ggml_time_us() : 0;
    std::memcpy(act_cma, down_preloaded->act_scale.data(), down_layout.act_bytes);
    if (collect_profile) {
        profile.activation_preprocess_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
        stage_start_us = ggml_time_us();
    }
    npu_decode_dma_mvin(
            act_cma,
            swiglu_act_base + down_layout.act_bytes,
            down_layout.act_bytes - 1u,
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

    for (const npu_decode_preloaded_block & block : down_preloaded->blocks) {
        if (block.packed.size() != block.layout.packed_bytes || block.layout.packed_bytes > max_packed_bytes) {
            cleanup();
            return fail("down preloaded block size mismatch");
        }
        void * down_weight_mvin_ptr = packed_cma;
        if (preloaded_cma && block.cma_bytes == block.layout.packed_bytes) {
            down_weight_mvin_ptr = static_cast<uint8_t *>(decode_cma_base) + block.cma_offset;
        } else {
            stage_start_us = collect_profile ? ggml_time_us() : 0;
            std::memcpy(packed_cma, block.packed.data(), block.layout.packed_bytes);
            if (collect_profile) {
                profile.host_copy_weight_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            }
        }
        stage_start_us = collect_profile ? ggml_time_us() : 0;
        npu_decode_dma_mvin(
                down_weight_mvin_ptr,
                0,
                block.layout.packed_bytes - 1u,
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
            profile.weight_bytes += block.layout.packed_bytes;
        }
        const uint64_t flow = npu_decode_make_bypass_output_flow(static_cast<uint16_t>(block.block_rows));
        npu_decode_trace_matvec_flow(
                "fused_ffn_down",
                op_name,
                static_cast<uint32_t>(&block - down_preloaded->blocks.data()),
                static_cast<uint32_t>(block.row_base),
                static_cast<uint32_t>(block.block_rows),
                static_cast<uint32_t>(down->k),
                NPU_DECODE_GEMV_MODE_W4A16,
                block.layout.weight_base,
                swiglu_act_base,
                static_cast<uint16_t>(block.row_base * sizeof(ggml_fp16_t)),
                block.layout.weight_base,
                flow);
        stage_start_us = collect_profile ? ggml_time_us() : 0;
        npu_decode_matvec_decode_flow_run(
                block.layout.weight_base,
                swiglu_act_base,
                static_cast<uint16_t>(down->k),
                static_cast<uint16_t>(block.block_rows),
                static_cast<uint16_t>(block.row_base * sizeof(ggml_fp16_t)),
                static_cast<uint16_t>(block.layout.weight_base),
                NPU_DECODE_GEMV_MODE_W4A16,
                flow);
        if (collect_profile) {
            profile.gemv_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            profile.gemv_calls += 1;
        }
    }
    }

    stage_start_us = collect_profile ? ggml_time_us() : 0;
    npu_decode_dma_mvout(
            out_cma,
            0,
            0,
            static_cast<uint32_t>(down->out_channels - 1),
            1,
            1,
            output_precision,
            1,
            false,
            false,
            0,
            0);
    if (collect_profile) {
        profile.mvout_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
        profile.mvout_calls += 1;
        profile.output_bytes += host_output_bytes;
        stage_start_us = ggml_time_us();
    }

    if (dst_type == GGML_TYPE_F16) {
        std::memcpy(dst_data, out_cma, static_cast<size_t>(down->out_channels) * sizeof(ggml_fp16_t));
    } else if (hw_f32_mvout) {
        std::memcpy(dst_data, out_cma, static_cast<size_t>(down->out_channels) * sizeof(float));
    } else {
        npu_decode_convert_output_fp16_to_f32(
                static_cast<const ggml_fp16_t *>(out_cma),
                reinterpret_cast<float *>(dst_data),
                static_cast<uint32_t>(down->out_channels));
    }
    if (collect_profile) {
        profile.postprocess_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
    }
    GGML_UNUSED(dst_nb1);

    const int64_t cleanup_start_us = collect_profile ? ggml_time_us() : 0;
    cleanup();
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
