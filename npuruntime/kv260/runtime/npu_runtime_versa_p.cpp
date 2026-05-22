#include "npu_runtime.h"

#include "versa_p_runtime.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace {

constexpr uint32_t kDefaultTimeoutMs = 10000;
constexpr uint32_t kBiasMetaOffset = 0;
constexpr uint32_t kScaleMetaOffset = 2048;

struct BufferRecord {
    versa_p_buffer buffer = {};
};

struct RuntimeProfile {
    npu_profile_runtime_summary summary = {};
    std::chrono::steady_clock::time_point layer_start;
    bool layer_active = false;
};

struct LoadedWeightBank {
    bool valid = false;
    uint16_t k = 0;
    uint16_t n = 0;
};

struct VersaPRuntimeState {
    versa_p_device * dev = nullptr;
    std::unordered_map<void *, BufferRecord> buffers;
    std::mutex mutex;
    RuntimeProfile profile;

    versa_p_api inflight_api[3] = {
        VERSA_P_API_MVIN_A,
        VERSA_P_API_MVIN_W,
        VERSA_P_API_MVIN_META,
    };
    bool inflight_valid[3] = {false, false, false};
    uint8_t active_w_bank = 1;
    uint8_t loaded_w_bank = 0;
    bool loaded_w_valid = false;
    uint16_t loaded_w_k = 0;
    uint16_t loaded_w_n = 0;
    LoadedWeightBank w_bank[2];
    uint8_t pending_w_bank = 0;
    uint16_t pending_w_k = 0;
    uint16_t pending_w_n = 0;
    bool gemm_inflight = false;
    std::chrono::steady_clock::time_point gemm_start;
};

VersaPRuntimeState & state() {
    static VersaPRuntimeState s;
    return s;
}

uint64_t elapsed_ns(std::chrono::steady_clock::time_point start) {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start).count());
}

bool trace_enabled() {
    const char * env = std::getenv("NPU_VERSA_P_TRACE");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

void add_profile_ns(versa_p_api api, uint64_t ns) {
    RuntimeProfile & p = state().profile;
    p.summary.total_ns += ns;
    switch (api) {
        case VERSA_P_API_MVIN_A:
        case VERSA_P_API_MVIN_W:
        case VERSA_P_API_MVIN_META:
            p.summary.dma_in_ns += ns;
            p.summary.mvin_calls += 1;
            break;
        case VERSA_P_API_GEMM_I8:
            p.summary.compute_ns += ns;
            p.summary.compute_calls += 1;
            p.summary.gemm_plan_calls += 1;
            break;
        case VERSA_P_API_MVOUT:
            p.summary.dma_out_ns += ns;
            p.summary.mvout_calls += 1;
            break;
    }
}

void clear_weight_state(VersaPRuntimeState & s) {
    s.active_w_bank = 1;
    s.loaded_w_bank = 0;
    s.loaded_w_valid = false;
    s.loaded_w_k = 0;
    s.loaded_w_n = 0;
    s.w_bank[0] = {};
    s.w_bank[1] = {};
    s.pending_w_bank = 0;
    s.pending_w_k = 0;
    s.pending_w_n = 0;
    s.gemm_inflight = false;
}

[[noreturn]] void fail_runtime(const char * call, int rc) {
    char buf[256];
    std::snprintf(buf, sizeof(buf), "%s failed: %s (%d)", call, versa_p_status_string(rc), rc);
    throw std::runtime_error(buf);
}

versa_p_device * require_dev() {
    VersaPRuntimeState & s = state();
    if (s.dev == nullptr) {
        const int rc = versa_p_init(&s.dev, nullptr);
        if (rc != VERSA_P_OK) {
            fail_runtime("versa_p_init", rc);
        }
        clear_weight_state(s);
    }
    return s.dev;
}

uint32_t dma_addr_for(void * ptr) {
    if (ptr == nullptr) {
        return 0;
    }
    return versa_p_dma_addr(require_dev(), ptr);
}

uint32_t actual_count(uint32_t value) {
    return value == 0 ? 1 : value;
}

uint16_t checked_u16(uint32_t value, const char * field) {
    if (value > 0xffffu) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "%s exceeds uint16: %" PRIu32, field, value);
        throw std::runtime_error(buf);
    }
    return static_cast<uint16_t>(value);
}

void wait_api(versa_p_api api, uint32_t timeout_ms = kDefaultTimeoutMs) {
    const auto start = std::chrono::steady_clock::now();
    const int rc = versa_p_wait(require_dev(), api, timeout_ms);
    add_profile_ns(api, elapsed_ns(start));
    if (rc != VERSA_P_OK) {
        fail_runtime("versa_p_wait", rc);
    }
    if (api == VERSA_P_API_MVIN_W) {
        VersaPRuntimeState & s = state();
        s.w_bank[s.pending_w_bank & 1u].valid = true;
        s.w_bank[s.pending_w_bank & 1u].k = s.pending_w_k;
        s.w_bank[s.pending_w_bank & 1u].n = s.pending_w_n;
        s.loaded_w_bank = s.pending_w_bank & 1u;
        s.loaded_w_k = s.pending_w_k;
        s.loaded_w_n = s.pending_w_n;
        s.loaded_w_valid = true;
        if (trace_enabled()) {
            std::fprintf(stderr, "[VERSA_P_TRACE] wait_mvin_w loaded bank=%u k=%u n=%u\n",
                         s.loaded_w_bank, s.loaded_w_k, s.loaded_w_n);
        }
    }
}

void start_mvin_w_bank(const MvinConfig & cfg, uint8_t w_bank, uint32_t dma_id) {
    VersaPRuntimeState & s = state();
    if (dma_id >= 3) {
        throw std::runtime_error("npu_dma_mvin_w_async_bank: dma_id out of range");
    }
    if (w_bank > 1) {
        throw std::runtime_error("npu_dma_mvin_w_async_bank: w_bank out of range");
    }
    if (cfg.input_type != 1) {
        throw std::runtime_error("npu_dma_mvin_w_async_bank: cfg is not weight MVIN");
    }
    if (s.inflight_valid[0]) {
        wait_api(s.inflight_api[0]);
        s.inflight_valid[0] = false;
    }
    if (s.inflight_valid[dma_id]) {
        wait_api(s.inflight_api[dma_id]);
        s.inflight_valid[dma_id] = false;
    }

    const uint16_t k = checked_u16(actual_count(cfg.row_num), "mvin_w.k");
    const uint16_t n = checked_u16(actual_count(cfg.col_num), "mvin_w.n");
    versa_p_mvin_w_desc desc = {};
    desc.dram_base = dma_addr_for(cfg.host_ptr);
    desc.k = k;
    desc.n = n;
    desc.w_bank = w_bank;
    const int rc = versa_p_start_mvin_w(require_dev(), &desc);
    if (rc != VERSA_P_OK) {
        fail_runtime("versa_p_start_mvin_w", rc);
    }
    s.w_bank[w_bank].valid = false;
    if (s.loaded_w_bank == w_bank) {
        s.loaded_w_valid = false;
    }
    s.pending_w_bank = w_bank;
    s.pending_w_k = k;
    s.pending_w_n = n;
    if (trace_enabled()) {
        std::fprintf(stderr, "[VERSA_P_TRACE] start_mvin_w bank=%u k=%u n=%u dram=0x%08x\n",
                     w_bank, k, n, desc.dram_base);
    }
    s.inflight_api[dma_id] = VERSA_P_API_MVIN_W;
    s.inflight_valid[dma_id] = true;
}

void start_mvin(const MvinConfig & cfg, uint32_t dma_id) {
    VersaPRuntimeState & s = state();
    if (dma_id >= 3) {
        throw std::runtime_error("npu_dma_mvin_async: dma_id out of range");
    }

    if (cfg.input_type == 0) {
        if (s.inflight_valid[1]) {
            wait_api(s.inflight_api[1]);
            s.inflight_valid[1] = false;
        }
        versa_p_mvin_a_desc desc = {};
        desc.dram_base = dma_addr_for(cfg.host_ptr);
        desc.dram_row_stride_bytes = cfg.dram_stride ? cfg.dram_stride : actual_count(cfg.col_num);
        desc.m = checked_u16(actual_count(cfg.row_num), "mvin_a.m");
        desc.k = checked_u16(actual_count(cfg.col_num), "mvin_a.k");
        desc.u8_minus_128 = cfg.is_quant ? 1 : 0;
        const int rc = versa_p_start_mvin_a(require_dev(), &desc);
        if (rc != VERSA_P_OK) {
            fail_runtime("versa_p_start_mvin_a", rc);
        }
        s.inflight_api[dma_id] = VERSA_P_API_MVIN_A;
        s.inflight_valid[dma_id] = true;
        return;
    }

    if (cfg.input_type == 1) {
        if (s.inflight_valid[0]) {
            wait_api(s.inflight_api[0]);
            s.inflight_valid[0] = false;
        }
        const uint16_t k = checked_u16(actual_count(cfg.row_num), "mvin_w.k");
        const uint16_t n = checked_u16(actual_count(cfg.col_num), "mvin_w.n");
        uint8_t bank = s.active_w_bank ^ 1u;
        if (s.loaded_w_valid && s.loaded_w_k == k && s.loaded_w_n == n) {
            bank = s.loaded_w_bank == s.active_w_bank ? static_cast<uint8_t>(s.active_w_bank ^ 1u) : s.loaded_w_bank;
        }
        start_mvin_w_bank(cfg, bank, dma_id);
        return;
    }

    if (cfg.input_type == 2) {
        const bool is_scale = !cfg.is_bias;
        const uint32_t meta_offset = is_scale ? kScaleMetaOffset : kBiasMetaOffset;
        const uint32_t words = actual_count(cfg.col_num);
        versa_p_mvin_meta_desc desc = {};
        desc.dram_base = dma_addr_for(cfg.host_ptr);
        desc.meta_offset_bytes = meta_offset;
        desc.byte_count = words * static_cast<uint32_t>(sizeof(int32_t));
        desc.meta_type = is_scale ? VERSA_P_META_SCALE : VERSA_P_META_BIAS;
        const int rc = versa_p_start_mvin_meta(require_dev(), &desc);
        if (rc != VERSA_P_OK) {
            fail_runtime("versa_p_start_mvin_meta", rc);
        }
        s.inflight_api[dma_id] = VERSA_P_API_MVIN_META;
        s.inflight_valid[dma_id] = true;
        return;
    }

    throw std::runtime_error("npu_dma_mvin_async: unsupported input_type");
}

void start_mvout(const MvoutConfig & cfg, uint32_t dma_id) {
    VersaPRuntimeState & s = state();
    if (dma_id >= 3) {
        throw std::runtime_error("npu_dma_mvout_async: dma_id out of range");
    }
    versa_p_mvout_desc desc = {};
    desc.dram_base = dma_addr_for(cfg.host_ptr);
    desc.m = checked_u16(actual_count(cfg.row_num), "mvout.m");
    desc.n = checked_u16(actual_count(cfg.col_num), "mvout.n");
    desc.output_stride_n = checked_u16(cfg.dram_stride ? cfg.dram_stride : actual_count(cfg.col_num), "mvout.output_stride_n");
    if (cfg.output_type == 1 && cfg.source && !cfg.is_quant) {
        desc.mode = VERSA_P_MVOUT_RAW_I32;
        desc.scale_param = 0;
    } else if (cfg.output_type == 1 && cfg.source && cfg.is_quant && cfg.per_channel) {
        desc.mode = VERSA_P_MVOUT_FP32_PER_CHANNEL_Q8_24;
        desc.scale_param = kScaleMetaOffset;
    } else if (cfg.output_type == 1 && cfg.source && cfg.is_quant) {
        desc.mode = VERSA_P_MVOUT_FP32_TENSOR_Q8_24;
        desc.scale_param = cfg.scale_or_addr;
    } else {
        throw std::runtime_error("npu_dma_mvout_async: unsupported mvout mode");
    }
    const int rc = versa_p_start_mvout(require_dev(), &desc);
    if (rc != VERSA_P_OK) {
        fail_runtime("versa_p_start_mvout", rc);
    }
    s.inflight_api[dma_id] = VERSA_P_API_MVOUT;
    s.inflight_valid[dma_id] = true;
}

} // namespace

extern "C" {

int npu_init() {
    try {
        (void)require_dev();
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "npu_init: %s\n", e.what());
        return -1;
    }
}

void npu_destroy() {
    VersaPRuntimeState & s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (s.dev != nullptr) {
        versa_p_destroy(s.dev);
        s.dev = nullptr;
    }
    s.buffers.clear();
    clear_weight_state(s);
}

void npu_reset() {
    VersaPRuntimeState & s = state();
    if (s.dev != nullptr) {
        const int rc = versa_p_reset(s.dev);
        if (rc != VERSA_P_OK) {
            fail_runtime("versa_p_reset", rc);
        }
    }
    clear_weight_state(s);
}

void npu_profile_begin(int64_t layer_id) {
    RuntimeProfile & p = state().profile;
    p.layer_active = true;
    p.layer_start = std::chrono::steady_clock::now();
    p.summary.layer_count += 1;
    p.summary.layer_invocations += 1;
    (void)layer_id;
}

void npu_profile_end(int64_t layer_id) {
    RuntimeProfile & p = state().profile;
    if (p.layer_active) {
        p.summary.layout_ns += elapsed_ns(p.layer_start);
        p.layer_active = false;
    }
    (void)layer_id;
}

void npu_profile_dump(const char * path) {
    (void)path;
}

void npu_profile_reset_summary() {
    state().profile.summary = {};
    state().profile.layer_active = false;
}

void npu_profile_get_summary(struct npu_profile_runtime_summary * out) {
    if (out != nullptr) {
        *out = state().profile.summary;
    }
}

void * npu_mem_alloc(size_t size) {
    if (size == 0) {
        return nullptr;
    }
    VersaPRuntimeState & s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    versa_p_buffer buffer = {};
    const int rc = versa_p_mem_alloc(require_dev(), static_cast<uint32_t>(size), VERSA_P_ALIGN_BYTES, &buffer);
    if (rc != VERSA_P_OK) {
        std::fprintf(stderr, "npu_mem_alloc(%zu): %s (%d)\n", size, versa_p_status_string(rc), rc);
        return nullptr;
    }
    s.buffers[buffer.vaddr] = BufferRecord{buffer};
    return buffer.vaddr;
}

void npu_mem_free(void * ptr) {
    if (ptr == nullptr) {
        return;
    }
    VersaPRuntimeState & s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    auto it = s.buffers.find(ptr);
    if (it == s.buffers.end()) {
        return;
    }
    versa_p_mem_free(require_dev(), &it->second.buffer);
    s.buffers.erase(it);
}

int npu_dma_copy_from_cma(const void * cma_ptr, void * dst, size_t bytes, uint32_t chunk_bytes) {
    if (cma_ptr == nullptr || dst == nullptr || bytes == 0) {
        return -EINVAL;
    }
    const int rc = versa_p_dma_copy_from_cma(require_dev(), cma_ptr, dst, bytes, chunk_bytes);
    if (rc == VERSA_P_OK) {
        return 0;
    }
    if (rc == VERSA_P_ERR_UNSUPPORTED_MODE) {
        return -EOPNOTSUPP;
    }
    if (rc == VERSA_P_ERR_INVAL) {
        return -EINVAL;
    }
    return -EIO;
}

void npu_dma_mvin_async(uint32_t dma_id, const MvinConfig * cfg) {
    if (cfg == nullptr) {
        throw std::runtime_error("npu_dma_mvin_async: null cfg");
    }
    start_mvin(*cfg, dma_id);
}

void npu_dma_mvout_async(uint32_t dma_id, const MvoutConfig * cfg) {
    if (cfg == nullptr) {
        throw std::runtime_error("npu_dma_mvout_async: null cfg");
    }
    start_mvout(*cfg, dma_id);
}

void npu_dma_wait_mvin(uint32_t dma_mask) {
    VersaPRuntimeState & s = state();
    for (uint32_t dma_id = 0; dma_id < 3; ++dma_id) {
        if ((dma_mask & (1u << dma_id)) == 0 || !s.inflight_valid[dma_id]) {
            continue;
        }
        wait_api(s.inflight_api[dma_id]);
        s.inflight_valid[dma_id] = false;
    }
}

void npu_dma_wait_mvout(uint32_t dma_mask) {
    npu_dma_wait_mvin(dma_mask);
}

void npu_dma_mvin_w_async_bank(uint8_t w_bank, const MvinConfig * cfg) {
    if (cfg == nullptr) {
        throw std::runtime_error("npu_dma_mvin_w_async_bank: null cfg");
    }
    start_mvin_w_bank(*cfg, w_bank, 1);
}

void npu_dma_wait_w_bank(uint8_t w_bank) {
    if (w_bank > 1) {
        throw std::runtime_error("npu_dma_wait_w_bank: w_bank out of range");
    }
    VersaPRuntimeState & s = state();
    if (s.inflight_valid[1] && s.pending_w_bank == w_bank) {
        wait_api(s.inflight_api[1]);
        s.inflight_valid[1] = false;
    }
}

void npu_dma_mvin(
        void * host_ptr, uint32_t sram_addr, uint32_t col_num, uint32_t row_num,
        uint16_t sram_stride, uint32_t dram_stride, uint8_t precision, uint8_t input_type,
        bool dest, bool is_bias, bool is_quant, uint32_t quant_zero, uint16_t quant_scale,
        uint16_t quant_shift) {
    MvinConfig cfg{host_ptr, sram_addr, col_num, row_num, sram_stride, dram_stride,
        precision, input_type, dest, is_bias, is_quant, quant_zero, quant_scale, quant_shift};
    npu_dma_mvin_async(input_type == 1 ? 1u : 2u, &cfg);
    npu_dma_wait_mvin(input_type == 1 ? (1u << 1) : (1u << 2));
}

void npu_dma_mvout(
        void * host_ptr, uint32_t sram_addr, uint32_t col_num, uint32_t row_num,
        uint16_t sram_stride, uint32_t dram_stride, uint8_t precision, uint8_t output_type,
        bool source, bool is_quant, uint32_t quant_zero, uint32_t scale_or_addr) {
    npu_dma_mvout_ex(host_ptr, sram_addr, col_num, row_num, sram_stride, dram_stride,
        precision, output_type, source, is_quant, quant_zero, scale_or_addr, false);
}

void npu_dma_mvout_ex(
        void * host_ptr, uint32_t sram_addr, uint32_t col_num, uint32_t row_num,
        uint16_t sram_stride, uint32_t dram_stride, uint8_t precision, uint8_t output_type,
        bool source, bool is_quant, uint32_t quant_zero, uint32_t scale_or_addr,
        bool per_channel) {
    MvoutConfig cfg{host_ptr, sram_addr, col_num, row_num, sram_stride, dram_stride,
        precision, output_type, source, is_quant, quant_zero, scale_or_addr, per_channel};
    npu_dma_mvout_async(2, &cfg);
    npu_dma_wait_mvout(1u << 2);
}

void npu_dma_double_mvin(const MvinConfig * dma0_cfg, const MvinConfig * dma1_cfg) {
    npu_dma_mvin_async(0, dma0_cfg);
    npu_dma_mvin_async(1, dma1_cfg);
    npu_dma_wait_mvin((1u << 0) | (1u << 1));
}

void npu_gemm_plan_start_ex_bank(
        uint8_t w_bank,
        uint32_t a_addr, uint32_t b_addr, uint32_t out_addr, uint32_t scratch_addr,
        uint32_t bias_addr, uint16_t block_m, uint16_t block_n, uint16_t block_k,
        uint16_t a_stride, uint16_t b_stride, uint16_t out_stride, uint16_t bias_stride,
        bool have_bias, bool is_accumulate, bool asymmetric_activations) {
    (void)a_addr; (void)b_addr; (void)out_addr; (void)scratch_addr;
    (void)bias_addr; (void)a_stride; (void)b_stride; (void)out_stride;
    (void)bias_stride; (void)asymmetric_activations;
    VersaPRuntimeState & s = state();
    if (w_bank > 1) {
        throw std::runtime_error("npu_gemm_plan_start_ex_bank: w_bank out of range");
    }
    if (!s.w_bank[w_bank].valid || s.w_bank[w_bank].k != block_k || s.w_bank[w_bank].n < block_n) {
        if (trace_enabled()) {
            std::fprintf(stderr,
                         "[VERSA_P_TRACE] gemm_plan missing_w bank=%u loaded=%u k=%u n=%u need_m=%u need_n=%u need_k=%u\n",
                         w_bank, s.w_bank[w_bank].valid ? 1u : 0u, s.w_bank[w_bank].k, s.w_bank[w_bank].n,
                         block_m, block_n, block_k);
        }
        throw std::runtime_error("npu_gemm_plan_start_ex_bank: W bank not loaded");
    }
    versa_p_gemm_i8_desc desc = {};
    desc.m = block_m;
    desc.n = block_n;
    desc.k = block_k;
    desc.w_bank = w_bank;
    desc.accumulate_en = (is_accumulate && !have_bias) ? 1 : 0;
    desc.add_bias_en = have_bias ? 1 : 0;
    desc.bias_offset_bytes = kBiasMetaOffset;
    const int rc = versa_p_start_gemm_i8(require_dev(), &desc);
    if (rc != VERSA_P_OK) {
        fail_runtime("versa_p_start_gemm_i8", rc);
    }
    s.active_w_bank = w_bank;
    s.loaded_w_bank = w_bank;
    s.loaded_w_valid = true;
    s.loaded_w_k = s.w_bank[w_bank].k;
    s.loaded_w_n = s.w_bank[w_bank].n;
    s.gemm_start = std::chrono::steady_clock::now();
    s.gemm_inflight = true;
}

void npu_gemm_plan_wait() {
    VersaPRuntimeState & s = state();
    if (!s.gemm_inflight) {
        return;
    }
    const int rc = versa_p_wait(require_dev(), VERSA_P_API_GEMM_I8, kDefaultTimeoutMs);
    add_profile_ns(VERSA_P_API_GEMM_I8, elapsed_ns(s.gemm_start));
    s.gemm_inflight = false;
    if (rc != VERSA_P_OK) {
        fail_runtime("versa_p_wait(GEMM_I8)", rc);
    }
}

void npu_gemm_plan_run_ex_bank(
        uint8_t w_bank,
        uint32_t a_addr, uint32_t b_addr, uint32_t out_addr, uint32_t scratch_addr,
        uint32_t bias_addr, uint16_t block_m, uint16_t block_n, uint16_t block_k,
        uint16_t a_stride, uint16_t b_stride, uint16_t out_stride, uint16_t bias_stride,
        bool have_bias, bool is_accumulate, bool asymmetric_activations) {
    npu_gemm_plan_start_ex_bank(
        w_bank, a_addr, b_addr, out_addr, scratch_addr, bias_addr,
        block_m, block_n, block_k, a_stride, b_stride, out_stride, bias_stride,
        have_bias, is_accumulate, asymmetric_activations);
    npu_gemm_plan_wait();
}

void npu_gemm_plan_run_ex(
        uint32_t a_addr, uint32_t b_addr, uint32_t out_addr, uint32_t scratch_addr,
        uint32_t bias_addr, uint16_t block_m, uint16_t block_n, uint16_t block_k,
        uint16_t a_stride, uint16_t b_stride, uint16_t out_stride, uint16_t bias_stride,
        bool have_bias, bool is_accumulate, bool asymmetric_activations) {
    VersaPRuntimeState & s = state();
    if (!s.loaded_w_valid || s.loaded_w_k != block_k || s.loaded_w_n < block_n) {
        if (trace_enabled()) {
            std::fprintf(stderr,
                         "[VERSA_P_TRACE] gemm_plan missing_w loaded=%u k=%u n=%u need_m=%u need_n=%u need_k=%u\n",
                         s.loaded_w_valid ? 1u : 0u, s.loaded_w_k, s.loaded_w_n,
                         block_m, block_n, block_k);
        }
        throw std::runtime_error("npu_gemm_plan_run_ex: W bank not loaded");
    }
    npu_gemm_plan_run_ex_bank(
        s.loaded_w_bank, a_addr, b_addr, out_addr, scratch_addr, bias_addr,
        block_m, block_n, block_k, a_stride, b_stride, out_stride, bias_stride,
        have_bias, is_accumulate, asymmetric_activations);
}

void npu_gemm_plan_run(
        uint32_t a_addr, uint32_t b_addr, uint32_t out_addr, uint32_t scratch_addr,
        uint32_t bias_addr, uint16_t block_m, uint16_t block_n, uint16_t block_k,
        uint16_t a_stride, uint16_t b_stride, uint16_t out_stride, uint16_t bias_stride,
        bool have_bias) {
    npu_gemm_plan_run_ex(a_addr, b_addr, out_addr, scratch_addr, bias_addr,
        block_m, block_n, block_k, a_stride, b_stride, out_stride, bias_stride,
        have_bias, !have_bias, false);
}

void npu_gemm_run(
        bool dataflow, uint8_t int_type, uint8_t optype, bool accout_dest,
        uint16_t input_a_zeropoint, uint16_t input_b_zeropoint,
        uint32_t output_zeropoint, uint16_t output_scale, uint16_t output_scaleshift,
        uint32_t biaspsum_addr, uint16_t biaspsum_stride, uint8_t biaspsum_width,
        uint8_t biaspsum_height, uint32_t output_addr, uint16_t output_stride,
        bool isaccu, bool relu, uint8_t relu_type, bool is_bias,
        uint32_t input_a_addr, uint16_t input_a_col_num, uint8_t input_a_row_num,
        uint16_t input_a_stride, uint32_t input_b_addr, uint8_t input_b_col_num,
        uint16_t input_b_row_num, uint16_t input_b_stride, bool asymmetric_activations) {
    (void)dataflow; (void)int_type; (void)optype; (void)accout_dest;
    (void)input_a_zeropoint; (void)input_b_zeropoint; (void)output_zeropoint;
    (void)output_scale; (void)output_scaleshift; (void)relu; (void)relu_type;
    (void)input_a_addr; (void)input_b_addr; (void)input_a_stride; (void)input_b_stride;
    (void)input_b_row_num;
    npu_gemm_plan_run_ex(input_a_addr, input_b_addr, output_addr, 0, biaspsum_addr,
        static_cast<uint16_t>(input_a_row_num + 1), static_cast<uint16_t>(input_b_col_num + 1),
        static_cast<uint16_t>(input_a_col_num + 1), input_a_stride, input_b_stride,
        output_stride, biaspsum_stride, is_bias, isaccu, asymmetric_activations);
    (void)biaspsum_width; (void)biaspsum_height;
}

}
