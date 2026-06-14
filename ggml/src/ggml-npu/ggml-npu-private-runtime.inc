
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

static bool npu_decode_stream_trace_enabled() {
    const char * value = std::getenv("AICAS_TEXT_DECODE_AWQ_STREAM_TRACE");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

static void npu_decode_trace_stream_flow(
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
    if (!npu_decode_stream_trace_enabled()) {
        return;
    }
    std::fprintf(
            stderr,
            "[NPU][STREAM_TRACE] tag=%s op=%s block=%u row_base=%u rows=%u k=%u mode=%u"
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
    return value == nullptr || value[0] == '\0' || std::strcmp(value, "0") != 0;
}

static bool npu_decode_w16a16_cma_enabled() {
    const char * value = std::getenv("AICAS_TEXT_LM_HEAD_W16A16_CMA");
    return value == nullptr || value[0] == '\0' || std::strcmp(value, "0") != 0;
}

static bool npu_decode_w8a16_cma_enabled() {
    const char * value = std::getenv("AICAS_TEXT_LM_HEAD_W8A16_CMA");
    return value == nullptr || value[0] == '\0' || std::strcmp(value, "0") != 0;
}

static bool npu_decode_w16a16_pingpong_enabled() {
    const char * value = std::getenv("AICAS_TEXT_LM_HEAD_W16A16_PINGPONG");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

static bool npu_decode_w8a16_pingpong_enabled() {
    const char * value = std::getenv("AICAS_TEXT_LM_HEAD_W8A16_PINGPONG");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

static void npu_decode_clear_w16a16_weight_cache();
static void npu_decode_release_w16a16_runtime_cma();
static void npu_decode_attention_clear_cache();
static void npu_decode_attention_zero_cache();

static bool npu_decode_require_active_overlay() {
    const char * value = std::getenv("AICAS_TEXT_DECODE_AWQ_NPU_REQUIRE_ACTIVE");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

static bool npu_decode_hw_f32_mvout_enabled() {
    const char * value = std::getenv("AICAS_TEXT_DECODE_AWQ_HW_F32_MVOUT");
    return value == nullptr || value[0] == '\0' || std::strcmp(value, "0") != 0;
}

static bool npu_decode_transient_scratch_enabled() {
    const char * value = std::getenv("AICAS_TEXT_DECODE_AWQ_TRANSIENT_SCRATCH");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

static bool npu_decode_zero_out_before_mvout_enabled() {
    const char * value = std::getenv("AICAS_TEXT_DECODE_AWQ_ZERO_OUT_BEFORE_MVOUT");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

static const char * npu_decode_device_path() {
    const char * value = std::getenv("AICAS_NPU_DECODE_DEVICE");
    return value != nullptr && value[0] != '\0' ? value : "/dev/npu_kv260";
}

static std::mutex & npu_decode_stream_device_mutex() {
    static std::mutex mutex;
    return mutex;
}

static npu_device *& npu_decode_stream_device() {
    static npu_device * dev = nullptr;
    return dev;
}

static void npu_decode_stream_close_device() {
    std::lock_guard<std::mutex> lock(npu_decode_stream_device_mutex());
    npu_device *& dev = npu_decode_stream_device();
    if (dev != nullptr) {
        npu_close(dev);
        dev = nullptr;
    }
}

static npu_device * npu_decode_stream_ensure_device(const char * op_name) {
    std::lock_guard<std::mutex> lock(npu_decode_stream_device_mutex());
    npu_device *& dev = npu_decode_stream_device();
    if (dev != nullptr) {
        return dev;
    }
    const int rc = npu_open(&dev, npu_decode_device_path());
    if (rc != 0) {
        if (npu_decode_w4a16_debug_log_enabled()) {
            GGML_LOG_WARN("%s: npu_open failed for %s dev=%s rc=%d\n",
                    __func__, op_name != nullptr ? op_name : "(unnamed)", npu_decode_device_path(), rc);
        }
        dev = nullptr;
        return nullptr;
    }
    return dev;
}

static std::mutex & npu_decode_overlay_mutex() {
    static std::mutex mutex;
    return mutex;
}

static bool & npu_decode_overlay_active() {
    static bool active = false;
    return active;
}

static std::mutex & npu_decode_runtime_mutex() {
    static std::mutex mutex;
    return mutex;
}

static bool & npu_decode_runtime_ready() {
    static bool ready = false;
    return ready;
}

static void npu_decode_mark_runtime_not_ready() {
    std::lock_guard<std::mutex> lock(npu_decode_runtime_mutex());
    npu_decode_runtime_ready() = false;
}

static bool npu_decode_ensure_overlay_active(const char * op_name) {
    const char * cmd = std::getenv("AICAS_NPU_DECODE_SWITCH_CMD");
    std::lock_guard<std::mutex> lock(npu_decode_overlay_mutex());
    if (npu_decode_overlay_active()) {
        return true;
    }
    if (cmd == nullptr || cmd[0] == '\0') {
        if (npu_decode_require_active_overlay()) {
            return false;
        }
        return true;
    }
    npu_clear_preloaded_weight_cache();
    npu_decode_release_w16a16_runtime_cma();
    npu_decode_stream_close_device();
    npu_destroy();
    npu_decode_mark_runtime_not_ready();
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
    {
        std::lock_guard<std::mutex> lock(npu_decode_runtime_mutex());
        if (npu_decode_runtime_ready()) {
            return 0;
        }
        if (npu_decode_init() == 0) {
            npu_decode_runtime_ready() = true;
            return 0;
        }
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
        npu_decode_stream_close_device();
        npu_destroy();
    }
    npu_decode_stream_close_device();
    npu_decode_mark_runtime_not_ready();
    npu_decode_destroy();
    {
        std::lock_guard<std::mutex> lock(npu_decode_runtime_mutex());
        if (npu_decode_init() == 0) {
            npu_decode_runtime_ready() = true;
            return 0;
        }
    }
    return -1;
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
constexpr uint32_t NPU_DECODE_GEMV_WEIGHT_TILE_BYTES = NPU_DECODE_GEMV_WEIGHT_DATA_TILE_BYTES;
constexpr uint32_t NPU_DECODE_GEMV_W16_WEIGHT_ROW_BYTES = NPU_DECODE_GEMV_K_TILE * 2;
constexpr uint32_t NPU_DECODE_GEMV_W16_WEIGHT_TILE_BYTES =
    NPU_DECODE_GEMV_M_TILE * NPU_DECODE_GEMV_W16_WEIGHT_ROW_BYTES;
constexpr uint32_t NPU_DECODE_GEMV_W8_K_TILE = NPU_DECODE_GEMV_K_TILE / 2;
constexpr uint32_t NPU_DECODE_GEMV_W8_ACT_TILE_BYTES = NPU_DECODE_GEMV_W8_K_TILE * 2;
constexpr uint32_t NPU_DECODE_GEMV_W8_WEIGHT_ROW_BYTES = NPU_DECODE_GEMV_W8_K_TILE;
constexpr uint32_t NPU_DECODE_GEMV_W8_WEIGHT_TILE_BYTES =
    NPU_DECODE_GEMV_M_TILE * NPU_DECODE_GEMV_W8_WEIGHT_ROW_BYTES;
constexpr uint8_t NPU_DECODE_GEMV_INPUT_TYPE_DATA = 0;
constexpr uint8_t NPU_DECODE_GEMV_INPUT_TYPE_WEIGHT = 1;
constexpr uint8_t NPU_DECODE_GEMV_INPUT_TYPE_ACT = 3;
constexpr uint8_t NPU_DECODE_GEMV_MODE_W4A16 = 0;
constexpr uint8_t NPU_DECODE_GEMV_MODE_W8A16 = 1;
constexpr uint8_t NPU_DECODE_GEMV_MODE_W16A16 = 2;
constexpr uint8_t NPU_DECODE_SRC_GEMV_STREAM = 1;
constexpr uint8_t NPU_DECODE_SRC_STREAM_BUFFER = 2;
constexpr uint8_t NPU_DECODE_UNARY_BYPASS = 0;
constexpr uint8_t NPU_DECODE_UNARY_SILU = 1;
constexpr uint8_t NPU_DECODE_UNARY_ROPE = 2;
constexpr uint8_t NPU_DECODE_BINARY_BYPASS = 0;
constexpr uint8_t NPU_DECODE_BINARY_FP16_MUL = 1;
constexpr uint8_t NPU_DECODE_REDUCE_BYPASS = 0;
constexpr uint8_t NPU_DECODE_REDUCE_SOFTMAX = 1;
constexpr uint8_t NPU_DECODE_DST_OUTPUT_SPM = 1;
constexpr uint8_t NPU_DECODE_DST_ACT_BUFFER = 2;
constexpr uint8_t NPU_DECODE_DST_STREAM_BUFFER = 3;
constexpr uint64_t NPU_DECODE_FLAG_KV_QUANT = 1ull << 24;
constexpr uint64_t NPU_DECODE_FLAG_KV_COL_SCALE = 1ull << 27;
constexpr uint64_t NPU_DECODE_FLAG_KV_V_SEPARATED = 1ull << 28;
constexpr uint32_t NPU_DECODE_GEMV_SPM_BYTES = 512u * 1024u;
constexpr uint32_t NPU_DECODE_GEMV_SCALE_BASE = 0x00000;
constexpr uint32_t NPU_DECODE_GEMV_ACT_BASE = 0x00000;
constexpr uint32_t NPU_DECODE_GEMV_OUTPUT_BASE = 0x00000;
constexpr uint32_t NPU_DECODE_GEMV_PING_WEIGHT_BASE = 0x10000;
constexpr uint32_t NPU_DECODE_GEMV_PONG_WEIGHT_BASE = 0x40000;
constexpr uint16_t NPU_DECODE_GEMV_ACT_CACHE_INVALIDATE = 0x8000;

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
    uint32_t act_scale_cma_offset = 0;
    uint32_t act_scale_cma_bytes = 0;
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
    int64_t group = 0;
    int64_t group_count = 0;
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
    void * cma_scale = nullptr;
    uint32_t cma_scale_offset = 0;
    uint32_t cma_scale_bytes = 0;
    bool cma_scale_owned = false;
    std::vector<uint8_t> packed_weight;
    std::vector<uint8_t> packed_scale;
    std::vector<uint8_t> scale_ones;
};

struct npu_decode_w16a16_cache {
    std::mutex mutex;
    npu_decode_w16a16_cached_weight entry;
};

struct npu_decode_w8a16_cache {
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
    nlohmann::ordered_json extra_time_us = nlohmann::ordered_json::object();
    nlohmann::ordered_json extra_calls = nlohmann::ordered_json::object();
};

struct npu_decode_profile_role_aggregate {
    uint64_t records = 0;
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
};

struct npu_decode_gemv_profile_cache {
    std::mutex mutex;
    std::vector<npu_decode_gemv_profile_record> records;
    std::map<std::string, npu_decode_profile_role_aggregate> role_aggregates;
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

static npu_decode_w8a16_cache & npu_decode_w8a16_weight_cache() {
    static npu_decode_w8a16_cache cache;
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
    if (cache.entry.cma_scale != nullptr && cache.entry.cma_scale_owned) {
        npu_decode_mem_free(cache.entry.cma_scale);
    }
    cache.entry = {};
}

static void npu_decode_clear_w8a16_weight_cache() {
    npu_decode_w8a16_cache & cache = npu_decode_w8a16_weight_cache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    if (cache.entry.cma_packed != nullptr && cache.entry.cma_packed_owned) {
        npu_decode_mem_free(cache.entry.cma_packed);
    }
    if (cache.entry.cma_scale != nullptr && cache.entry.cma_scale_owned) {
        npu_decode_mem_free(cache.entry.cma_scale);
    }
    cache.entry = {};
}

static void npu_decode_release_w16a16_runtime_cma() {
    npu_decode_attention_clear_cache();
    {
        npu_decode_w16a16_cache & cache = npu_decode_w16a16_weight_cache();
        std::lock_guard<std::mutex> lock(cache.mutex);
        if (cache.entry.cma_packed != nullptr && cache.entry.cma_packed_owned) {
            npu_decode_mem_free(cache.entry.cma_packed);
        }
        if (cache.entry.cma_scale_ones != nullptr && cache.entry.cma_scale_ones_owned) {
            npu_decode_mem_free(cache.entry.cma_scale_ones);
        }
        if (cache.entry.cma_scale != nullptr && cache.entry.cma_scale_owned) {
            npu_decode_mem_free(cache.entry.cma_scale);
        }
        cache.entry.cma_packed = nullptr;
        cache.entry.cma_packed_owned = false;
        cache.entry.cma_scale_ones = nullptr;
        cache.entry.cma_scale_ones_owned = false;
        cache.entry.cma_scale = nullptr;
        cache.entry.cma_scale_owned = false;
    }
    {
        npu_decode_w8a16_cache & cache = npu_decode_w8a16_weight_cache();
        std::lock_guard<std::mutex> lock(cache.mutex);
        if (cache.entry.cma_packed != nullptr && cache.entry.cma_packed_owned) {
            npu_decode_mem_free(cache.entry.cma_packed);
        }
        if (cache.entry.cma_scale != nullptr && cache.entry.cma_scale_owned) {
            npu_decode_mem_free(cache.entry.cma_scale);
        }
        cache.entry.cma_packed = nullptr;
        cache.entry.cma_packed_owned = false;
        cache.entry.cma_scale = nullptr;
        cache.entry.cma_scale_owned = false;
    }
}

static npu_decode_gemv_profile_cache & npu_decode_profile_cache() {
    static auto * cache = new npu_decode_gemv_profile_cache();
    return *cache;
}

static const char * npu_decode_profile_path() {
    const char * path = std::getenv("GGML_NPU_DECODE_PROFILE_JSONL");
    return path != nullptr && path[0] != '\0' ? path : nullptr;
}

static const char * npu_decode_profile_aggregate_path() {
    const char * path = std::getenv("GGML_NPU_DECODE_PROFILE_AGG_JSON");
    return path != nullptr && path[0] != '\0' ? path : nullptr;
}

static bool npu_decode_profile_enabled() {
    return npu_decode_profile_path() != nullptr || npu_decode_profile_aggregate_path() != nullptr;
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
    return value == nullptr || value[0] == '\0' || std::strcmp(value, "0") != 0;
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
    if (!record.extra_time_us.empty()) {
        payload["extra_time_us"] = record.extra_time_us;
    }
    if (!record.extra_calls.empty()) {
        payload["extra_calls"] = record.extra_calls;
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

static void npu_decode_profile_accumulate_role(
        std::map<std::string, npu_decode_profile_role_aggregate> & aggregates,
        const npu_decode_gemv_profile_record & record) {
    const std::string role = record.op_role.empty() ? "decode_gemv" : record.op_role;
    npu_decode_profile_role_aggregate & dst = aggregates[role];
    dst.records += 1;
    dst.total_us += record.total_us;
    dst.setup_us += record.setup_us;
    dst.validation_us += record.validation_us;
    dst.layout_us += record.layout_us;
    dst.preload_lookup_us += record.preload_lookup_us;
    dst.overlay_ensure_us += record.overlay_ensure_us;
    dst.heap_update_us += record.heap_update_us;
    dst.runtime_init_us += record.runtime_init_us;
    dst.memory_base_us += record.memory_base_us;
    dst.alloc_us += record.alloc_us;
    dst.activation_preprocess_us += record.activation_preprocess_us;
    dst.host_copy_activation_us += record.host_copy_activation_us;
    dst.reset_us += record.reset_us;
    dst.mvin_activation_us += record.mvin_activation_us;
    dst.weight_preprocess_us += record.weight_preprocess_us;
    dst.host_copy_weight_us += record.host_copy_weight_us;
    dst.output_clear_us += record.output_clear_us;
    dst.scale_mvin_us += record.scale_mvin_us;
    dst.mvin_weight_us += record.mvin_weight_us;
    dst.gemv_us += record.gemv_us;
    dst.pingpong_us += record.pingpong_us;
    dst.pingpong_initial_mvin_us += record.pingpong_initial_mvin_us;
    dst.pingpong_prefetch_issue_us += record.pingpong_prefetch_issue_us;
    dst.pingpong_prefetch_window_us += record.pingpong_prefetch_window_us;
    dst.pingpong_prefetch_wait_us += record.pingpong_prefetch_wait_us;
    dst.pingpong_gemv_us += record.pingpong_gemv_us;
    dst.pingpong_mvout_us += record.pingpong_mvout_us;
    dst.mvout_us += record.mvout_us;
    dst.postprocess_us += record.postprocess_us;
    dst.cleanup_us += record.cleanup_us;
}

static void npu_decode_profile_write_aggregate(
        const char * path,
        const std::map<std::string, npu_decode_profile_role_aggregate> & aggregates) {
    if (path == nullptr || aggregates.empty()) {
        return;
    }

    nlohmann::ordered_json roles = nlohmann::ordered_json::array();
    uint64_t total_records = 0;
    uint64_t total_us = 0;
    for (const auto & kv : aggregates) {
        const npu_decode_profile_role_aggregate & a = kv.second;
        total_records += a.records;
        total_us += a.total_us;
        roles.push_back({
            {"op_role", kv.first},
            {"records", a.records},
            {"time_us", {
                {"total", a.total_us},
                {"setup", a.setup_us},
                {"validation", a.validation_us},
                {"layout", a.layout_us},
                {"preload_lookup", a.preload_lookup_us},
                {"overlay_ensure", a.overlay_ensure_us},
                {"heap_update", a.heap_update_us},
                {"runtime_init", a.runtime_init_us},
                {"memory_base", a.memory_base_us},
                {"alloc", a.alloc_us},
                {"activation_preprocess", a.activation_preprocess_us},
                {"host_copy_activation", a.host_copy_activation_us},
                {"reset", a.reset_us},
                {"mvin_activation", a.mvin_activation_us},
                {"weight_preprocess", a.weight_preprocess_us},
                {"host_copy_weight", a.host_copy_weight_us},
                {"output_clear", a.output_clear_us},
                {"scale_mvin", a.scale_mvin_us},
                {"mvin_weight", a.mvin_weight_us},
                {"gemv", a.gemv_us},
                {"pingpong", a.pingpong_us},
                {"pingpong_initial_mvin", a.pingpong_initial_mvin_us},
                {"pingpong_prefetch_issue", a.pingpong_prefetch_issue_us},
                {"pingpong_prefetch_window", a.pingpong_prefetch_window_us},
                {"pingpong_prefetch_wait", a.pingpong_prefetch_wait_us},
                {"pingpong_gemv", a.pingpong_gemv_us},
                {"pingpong_mvout", a.pingpong_mvout_us},
                {"mvout", a.mvout_us},
                {"postprocess", a.postprocess_us},
                {"cleanup", a.cleanup_us},
            }},
        });
    }

    nlohmann::ordered_json payload = {
        {"profile_kind", "aicas_decode_role_aggregate_profile"},
        {"timing_unit", "us"},
        {"note", "Low-overhead decode profile aggregated by op_role; per-op JSONL records are omitted unless GGML_NPU_DECODE_PROFILE_JSONL is also set."},
        {"total_records", total_records},
        {"total_us", total_us},
        {"roles", std::move(roles)},
    };

    std::ofstream out(path, std::ios::trunc);
    if (out.good()) {
        out << payload.dump(2) << '\n';
    }
}

static void npu_decode_profile_flush() {
    const char * path = npu_decode_profile_path();
    const char * aggregate_path = npu_decode_profile_aggregate_path();
    npu_decode_gemv_profile_cache & cache = npu_decode_profile_cache();
    std::vector<npu_decode_gemv_profile_record> snapshot;
    std::map<std::string, npu_decode_profile_role_aggregate> aggregate_snapshot;
    {
        std::lock_guard<std::mutex> lock(cache.mutex);
        snapshot.swap(cache.records);
        aggregate_snapshot = cache.role_aggregates;
    }

    if (path != nullptr && !snapshot.empty()) {
        npu_decode_profile_append(path, snapshot);
    }
    npu_decode_profile_write_aggregate(aggregate_path, aggregate_snapshot);
}

static void npu_decode_profile_write(const npu_decode_gemv_profile_record & record) {
    npu_decode_gemv_profile_cache & cache = npu_decode_profile_cache();
    const char * path = npu_decode_profile_path();
    const char * aggregate_path = npu_decode_profile_aggregate_path();
    std::vector<npu_decode_gemv_profile_record> snapshot;
    {
        std::lock_guard<std::mutex> lock(cache.mutex);
        if (!cache.atexit_registered) {
            std::atexit(npu_decode_profile_flush);
            cache.atexit_registered = true;
        }
        if (aggregate_path != nullptr) {
            npu_decode_profile_accumulate_role(cache.role_aggregates, record);
        }
        if (path != nullptr) {
            cache.records.push_back(record);
        }
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
                tensor->pingpong_cma_reserve_attempted) {
            continue;
        }
        if (tensor->act_scale_cma_bytes == tensor->pingpong_layout.act_bytes &&
                tensor->pingpong_scale_cma_bytes == tensor->pingpong_layout.scale_bytes &&
                tensor->pingpong_weight_cma_bytes == tensor->pingpong_layout.weight_bytes) {
            continue;
        }
        if (tensor->act_scale_cma_bytes != tensor->pingpong_layout.act_bytes) {
            arena_bytes = npu_decode_align_up_bytes(arena_bytes, NPU_DECODE_GEMV_DMA_ALIGN);
            if (tensor->pingpong_layout.act_bytes > std::numeric_limits<uint32_t>::max() - arena_bytes) {
                return false;
            }
            arena_bytes += tensor->pingpong_layout.act_bytes;
        }
        arena_bytes = npu_decode_align_up_bytes(arena_bytes, NPU_DECODE_GEMV_DMA_ALIGN);
        if (tensor->pingpong_scale_cma_bytes != tensor->pingpong_layout.scale_bytes) {
            if (tensor->pingpong_layout.scale_bytes > std::numeric_limits<uint32_t>::max() - arena_bytes) {
                return false;
            }
            arena_bytes += tensor->pingpong_layout.scale_bytes;
        }
        arena_bytes = npu_decode_align_up_bytes(arena_bytes, NPU_DECODE_GEMV_DMA_ALIGN);
        if (tensor->pingpong_weight_cma_bytes != tensor->pingpong_layout.weight_bytes) {
            if (tensor->pingpong_layout.weight_bytes > std::numeric_limits<uint32_t>::max() - arena_bytes) {
                return false;
            }
            arena_bytes += tensor->pingpong_layout.weight_bytes;
        }
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
                tensor->pingpong_cma_reserve_attempted) {
            continue;
        }
        npu_decode_preloaded_tensor & entry = *tensor;
        if (entry.act_scale_cma_bytes == entry.pingpong_layout.act_bytes &&
                entry.pingpong_scale_cma_bytes == entry.pingpong_layout.scale_bytes &&
                entry.pingpong_weight_cma_bytes == entry.pingpong_layout.weight_bytes) {
            continue;
        }
        if (entry.act_scale_cma_bytes != entry.pingpong_layout.act_bytes) {
            cursor = npu_decode_align_up_bytes(cursor, NPU_DECODE_GEMV_DMA_ALIGN);
            entry.act_scale_cma_offset = static_cast<uint32_t>(arena_offset) + cursor;
            entry.act_scale_cma_bytes = entry.pingpong_layout.act_bytes;
            if (!entry.act_scale.empty()) {
                std::memcpy(static_cast<uint8_t *>(arena) + cursor, entry.act_scale.data(), entry.act_scale.size());
            }
            cursor += entry.pingpong_layout.act_bytes;
        }
        cursor = npu_decode_align_up_bytes(cursor, NPU_DECODE_GEMV_DMA_ALIGN);
        if (entry.pingpong_scale_cma_bytes != entry.pingpong_layout.scale_bytes && entry.pingpong_layout.scale_bytes > 0) {
            entry.pingpong_scale_cma_offset = static_cast<uint32_t>(arena_offset) + cursor;
            entry.pingpong_scale_cma_bytes = entry.pingpong_layout.scale_bytes;
            std::memcpy(static_cast<uint8_t *>(arena) + cursor, entry.pingpong_scale.data(), entry.pingpong_scale.size());
            cursor += entry.pingpong_layout.scale_bytes;
        }
        cursor = npu_decode_align_up_bytes(cursor, NPU_DECODE_GEMV_DMA_ALIGN);
        if (entry.pingpong_weight_cma_bytes != entry.pingpong_layout.weight_bytes) {
            entry.pingpong_weight_cma_offset = static_cast<uint32_t>(arena_offset) + cursor;
            entry.pingpong_weight_cma_bytes = entry.pingpong_layout.weight_bytes;
            std::memcpy(static_cast<uint8_t *>(arena) + cursor, entry.pingpong_weight.data(), entry.pingpong_weight.size());
            cursor += entry.pingpong_layout.weight_bytes;
        }
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

static bool npu_decode_stream_api_enabled() {
    // Current decode overlay/runtime contract only supports the stream API.
    // Keep the helper for call-site readability, but do not allow old
    // split/legacy NPU paths to be selected by environment.
    return true;
}

static kv260_stream_output_precision npu_decode_stream_output_precision(bool hw_f32_mvout) {
    return hw_f32_mvout ? KV260_STREAM_OUTPUT_FP32 : KV260_STREAM_OUTPUT_FP16;
}

static bool npu_decode_stream_w4_cma_resident(const npu_decode_preloaded_tensor * tensor) {
    return tensor != nullptr &&
           tensor->pingpong_available &&
           tensor->act_scale_cma_bytes == tensor->pingpong_layout.act_bytes &&
           tensor->pingpong_scale_cma_bytes == tensor->pingpong_layout.scale_bytes &&
           tensor->pingpong_weight_cma_bytes == tensor->pingpong_layout.weight_bytes;
}

static void * npu_decode_stream_cma_ptr(void * decode_cma_base, uint32_t offset) {
    return static_cast<uint8_t *>(decode_cma_base) + offset;
}

static int npu_decode_stream_gemv(
        const char * op_name,
        const kv260_stream_gemv_desc & desc) {
    npu_device * dev = npu_decode_stream_ensure_device(op_name);
    if (dev == nullptr) {
        return -1;
    }
    return npu_stream_gemv_run(dev, &desc, 0);
}

static int npu_decode_stream_kv_scale_read(
        const char * op_name,
        bool is_v,
        kv260_stream_kv_scale_result * out) {
    if (npu_stream_kv_scale_read == nullptr) {
        return -1;
    }
    npu_device * dev = npu_decode_stream_ensure_device(op_name);
    if (dev == nullptr) {
        return -1;
    }
    return npu_stream_kv_scale_read(dev, is_v ? 1 : 0, out);
}

template<typename... Args>
static void npu_decode_stream_unsupported_void(Args&&...) {
    throw std::runtime_error("decode legacy/split runtime path is disabled; use npu_stream_gemv_run");
}

template<typename... Args>
static uint64_t npu_decode_stream_unsupported_flow(Args&&...) {
    throw std::runtime_error("decode legacy/split runtime flow is disabled; use npu_stream_gemv_run");
}
