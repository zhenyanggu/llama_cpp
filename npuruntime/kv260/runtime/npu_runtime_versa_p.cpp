#include "npu_runtime.h"

#include "versa_p_runtime.h"
#include "versa_p_internal.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

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

struct LoadedActivationBank {
    bool valid = false;
    uint16_t m = 0;
    uint16_t k = 0;
};

struct AttentionWorkspace {
    versa_p_buffer q = {};
    versa_p_buffer k = {};
    versa_p_buffer v = {};
    versa_p_buffer logp = {};
    versa_p_buffer out = {};
    uint32_t q_bytes = 0;
    uint32_t k_bytes = 0;
    uint32_t v_bytes = 0;
    uint32_t logp_bytes = 0;
    uint32_t out_bytes = 0;
    bool q_valid = false;
    bool k_valid = false;
    bool v_valid = false;
    bool logp_valid = false;
    bool out_valid = false;
};

struct VersaPRuntimeState {
    versa_p_device * dev = nullptr;
    std::unordered_map<void *, BufferRecord> buffers;
    std::mutex mutex;
    RuntimeProfile profile;
    AttentionWorkspace attention_ws;

    versa_p_api inflight_api[3] = {
        VERSA_P_API_MVIN_A,
        VERSA_P_API_MVIN_W,
        VERSA_P_API_MVIN_META,
    };
    bool inflight_valid[3] = {false, false, false};
    uint8_t active_a_bank = 0;
    uint8_t loaded_a_bank = 0;
    bool loaded_a_valid = false;
    uint16_t loaded_a_m = 0;
    uint16_t loaded_a_k = 0;
    LoadedActivationBank a_bank[2];
    uint8_t pending_a_bank = 0;
    uint16_t pending_a_m = 0;
    uint16_t pending_a_k = 0;
    uint8_t active_w_bank = 1;
    uint8_t loaded_w_bank = 0;
    bool loaded_w_valid = false;
    uint16_t loaded_w_k = 0;
    uint16_t loaded_w_n = 0;
    LoadedWeightBank w_bank[2];
    uint8_t pending_w_bank = 0;
    uint16_t pending_w_k = 0;
    uint16_t pending_w_n = 0;
    uint8_t pending_o_bank = 0;
    bool o_bank_inflight[2] = {false, false};
    uint8_t gemm_o_bank = 0;
    bool gemm_inflight = false;
    std::chrono::steady_clock::time_point gemm_start;
    bool bias_meta_base_valid = false;
    uint32_t bias_meta_base_addr = 0;
    uint32_t bias_meta_bytes = 0;
    bool scale_meta_base_valid = false;
    uint32_t scale_meta_base_addr = 0;
    uint32_t scale_meta_offset_bytes = kScaleMetaOffset;
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

void clear_bank_state(VersaPRuntimeState & s) {
    s.active_a_bank = 0;
    s.loaded_a_bank = 0;
    s.loaded_a_valid = false;
    s.loaded_a_m = 0;
    s.loaded_a_k = 0;
    s.a_bank[0] = {};
    s.a_bank[1] = {};
    s.pending_a_bank = 0;
    s.pending_a_m = 0;
    s.pending_a_k = 0;
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
    s.pending_o_bank = 0;
    s.o_bank_inflight[0] = false;
    s.o_bank_inflight[1] = false;
    s.gemm_o_bank = 0;
    s.gemm_inflight = false;
    s.bias_meta_base_valid = false;
    s.bias_meta_base_addr = 0;
    s.bias_meta_bytes = 0;
    s.scale_meta_base_valid = false;
    s.scale_meta_base_addr = 0;
    s.scale_meta_offset_bytes = kScaleMetaOffset;
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
        clear_bank_state(s);
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

uint32_t align_u32(uint32_t value, uint32_t alignment) {
    return (value + alignment - 1u) & ~(alignment - 1u);
}

uint16_t checked_u16(uint32_t value, const char * field) {
    if (value > 0xffffu) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "%s exceeds uint16: %" PRIu32, field, value);
        throw std::runtime_error(buf);
    }
    return static_cast<uint16_t>(value);
}

uint32_t packed_w_bytes_u32(uint32_t k, uint32_t n) {
    return k * ((n + 31u) / 32u) * 32u;
}

void pack_w_versa_i8(const std::vector<int8_t> & src, uint32_t k, uint32_t n, std::vector<int8_t> & dst) {
    dst.assign(packed_w_bytes_u32(k, n), 0);
    const uint32_t n_groups = (n + 31u) / 32u;
    for (uint32_t group = 0; group < n_groups; ++group) {
        const uint32_t base_col = group * 32u;
        const uint32_t group_cols = std::min(32u, n - base_col);
        for (uint32_t kk = 0; kk < k; ++kk) {
            for (uint32_t lane = 0; lane < group_cols; ++lane) {
                dst[((size_t) group * k + kk) * 32u + lane] =
                    src[(size_t) kk * n + base_col + lane];
            }
        }
    }
}

std::vector<int8_t> transpose_token_dim_i8(const int8_t * src, uint32_t tokens) {
    std::vector<int8_t> dst((size_t) 64u * tokens);
    for (uint32_t token = 0; token < tokens; ++token) {
        for (uint32_t dim = 0; dim < 64u; ++dim) {
            dst[(size_t) dim * tokens + token] = src[(size_t) token * 64u + dim];
        }
    }
    return dst;
}

bool ensure_attention_buffer(
        versa_p_device * dev,
        versa_p_buffer & buffer,
        uint32_t & capacity,
        bool & valid,
        uint32_t required_bytes) {
    required_bytes = align_u32(required_bytes, VERSA_P_ALIGN_BYTES);
    if (valid && capacity >= required_bytes) {
        return true;
    }
    if (valid) {
        versa_p_mem_free(dev, &buffer);
        buffer = {};
        capacity = 0;
        valid = false;
    }
    const int rc = versa_p_mem_alloc(dev, required_bytes, VERSA_P_ALIGN_BYTES, &buffer);
    if (rc != VERSA_P_OK) {
        return false;
    }
    capacity = required_bytes;
    valid = true;
    return true;
}

void release_attention_workspace(versa_p_device * dev, AttentionWorkspace & ws) {
    if (ws.out_valid) {
        versa_p_mem_free(dev, &ws.out);
    }
    if (ws.logp_valid) {
        versa_p_mem_free(dev, &ws.logp);
    }
    if (ws.v_valid) {
        versa_p_mem_free(dev, &ws.v);
    }
    if (ws.k_valid) {
        versa_p_mem_free(dev, &ws.k);
    }
    if (ws.q_valid) {
        versa_p_mem_free(dev, &ws.q);
    }
    ws = {};
}

bool attention_workspace_enabled() {
    const char * env = std::getenv("AICAS_LOG8PV_ATTENTION_WORKSPACE");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

void wait_api(versa_p_api api, uint32_t timeout_ms = kDefaultTimeoutMs) {
    const auto start = std::chrono::steady_clock::now();
    const int rc = versa_p_wait(require_dev(), api, timeout_ms);
    add_profile_ns(api, elapsed_ns(start));
    if (rc != VERSA_P_OK) {
        fail_runtime("versa_p_wait", rc);
    }
    if (api == VERSA_P_API_MVIN_A) {
        VersaPRuntimeState & s = state();
        s.a_bank[s.pending_a_bank & 1u].valid = true;
        s.a_bank[s.pending_a_bank & 1u].m = s.pending_a_m;
        s.a_bank[s.pending_a_bank & 1u].k = s.pending_a_k;
        s.loaded_a_bank = s.pending_a_bank & 1u;
        s.loaded_a_m = s.pending_a_m;
        s.loaded_a_k = s.pending_a_k;
        s.loaded_a_valid = true;
        if (trace_enabled()) {
            std::fprintf(stderr, "[VERSA_P_TRACE] wait_mvin_a loaded bank=%u m=%u k=%u\n",
                         s.loaded_a_bank, s.loaded_a_m, s.loaded_a_k);
        }
    } else if (api == VERSA_P_API_MVIN_W) {
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
    } else if (api == VERSA_P_API_MVOUT) {
        VersaPRuntimeState & s = state();
        s.o_bank_inflight[s.pending_o_bank & 1u] = false;
        if (trace_enabled()) {
            std::fprintf(stderr, "[VERSA_P_TRACE] wait_mvout o_bank=%u\n",
                         s.pending_o_bank & 1u);
        }
    }
}

void start_mvin_a_bank(const MvinConfig & cfg, uint8_t a_bank, uint32_t dma_id) {
    VersaPRuntimeState & s = state();
    if (dma_id >= 3) {
        throw std::runtime_error("npu_dma_mvin_a_async_bank: dma_id out of range");
    }
    if (a_bank > 1) {
        throw std::runtime_error("npu_dma_mvin_a_async_bank: a_bank out of range");
    }
    if (cfg.input_type != 0) {
        throw std::runtime_error("npu_dma_mvin_a_async_bank: cfg is not activation MVIN");
    }
    if (s.inflight_valid[dma_id]) {
        wait_api(s.inflight_api[dma_id]);
        s.inflight_valid[dma_id] = false;
    }

    const uint16_t m = checked_u16(actual_count(cfg.row_num), "mvin_a.m");
    const uint16_t k = checked_u16(actual_count(cfg.col_num), "mvin_a.k");
    versa_p_mvin_a_desc desc = {};
    desc.dram_base = dma_addr_for(cfg.host_ptr);
    desc.dram_row_stride_bytes = cfg.dram_stride ? cfg.dram_stride : actual_count(cfg.col_num);
    desc.m = m;
    desc.k = k;
    desc.a_bank = a_bank;
    desc.u8_minus_128 = cfg.is_quant ? 1 : 0;
    const int rc = versa_p_start_mvin_a(require_dev(), &desc);
    if (rc != VERSA_P_OK) {
        char detail[256];
        std::snprintf(detail, sizeof(detail),
                      "versa_p_start_mvin_a bank=%u m=%u k=%u stride=%u dram=0x%08x",
                      static_cast<unsigned>(desc.a_bank),
                      static_cast<unsigned>(desc.m),
                      static_cast<unsigned>(desc.k),
                      static_cast<unsigned>(desc.dram_row_stride_bytes),
                      desc.dram_base);
        if (rc == VERSA_P_ERR_ILLEGAL_SHAPE) {
            throw std::runtime_error(std::string(detail) + " failed: illegal shape (-7)");
        }
        fail_runtime("versa_p_start_mvin_a", rc);
    }
    s.a_bank[a_bank].valid = false;
    if (s.loaded_a_bank == a_bank) {
        s.loaded_a_valid = false;
    }
    s.pending_a_bank = a_bank;
    s.pending_a_m = m;
    s.pending_a_k = k;
    if (trace_enabled()) {
        std::fprintf(stderr, "[VERSA_P_TRACE] start_mvin_a bank=%u m=%u k=%u dram=0x%08x stride=%u\n",
                     a_bank, m, k, desc.dram_base, desc.dram_row_stride_bytes);
    }
    s.inflight_api[dma_id] = VERSA_P_API_MVIN_A;
    s.inflight_valid[dma_id] = true;
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
        start_mvin_a_bank(cfg, s.active_a_bank, dma_id);
        return;
    }

    if (cfg.input_type == 1) {
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
        if (s.inflight_valid[dma_id]) {
            wait_api(s.inflight_api[dma_id]);
            s.inflight_valid[dma_id] = false;
        }
        const bool is_scale = !cfg.is_bias;
        const uint32_t words = actual_count(cfg.col_num);
        const uint32_t byte_count = words * static_cast<uint32_t>(sizeof(int32_t));
        uint32_t meta_offset = kBiasMetaOffset;
        if (is_scale) {
            meta_offset = cfg.sram_addr;
        }
        if (meta_offset + byte_count > VERSA_P_META_BYTES) {
            throw std::runtime_error("npu_dma_mvin_async: META bank overflow");
        }
        versa_p_mvin_meta_desc desc = {};
        desc.dram_base = dma_addr_for(cfg.host_ptr);
        desc.meta_offset_bytes = meta_offset;
        desc.byte_count = byte_count;
        desc.meta_type = is_scale ? VERSA_P_META_SCALE : VERSA_P_META_BIAS;
        const int rc = versa_p_start_mvin_meta(require_dev(), &desc);
        if (rc != VERSA_P_OK) {
            fail_runtime("versa_p_start_mvin_meta", rc);
        }
        if (!is_scale) {
            s.bias_meta_base_valid = true;
            s.bias_meta_base_addr = cfg.sram_addr;
            s.bias_meta_bytes = byte_count;
        } else {
            s.scale_meta_base_valid = true;
            s.scale_meta_base_addr = cfg.sram_addr;
            s.scale_meta_offset_bytes = meta_offset;
        }
        if (trace_enabled()) {
            std::fprintf(stderr,
                         "[VERSA_P_TRACE] start_mvin_meta type=%s meta_offset=0x%08x logical_sram=0x%08x bytes=%u\n",
                         is_scale ? "scale" : "bias",
                         desc.meta_offset_bytes,
                         cfg.sram_addr,
                         desc.byte_count);
        }
        s.inflight_api[dma_id] = VERSA_P_API_MVIN_META;
        s.inflight_valid[dma_id] = true;
        return;
    }

    throw std::runtime_error("npu_dma_mvin_async: unsupported input_type");
}

void start_mvout_bank(const MvoutConfig & cfg, uint8_t o_bank, uint32_t dma_id) {
    VersaPRuntimeState & s = state();
    if (dma_id >= 3) {
        throw std::runtime_error("npu_dma_mvout_async_bank: dma_id out of range");
    }
    if (o_bank > 1) {
        throw std::runtime_error("npu_dma_mvout_async_bank: o_bank out of range");
    }
    if (s.inflight_valid[dma_id]) {
        wait_api(s.inflight_api[dma_id]);
        s.inflight_valid[dma_id] = false;
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
        if (s.scale_meta_base_valid && cfg.scale_or_addr >= s.scale_meta_base_addr) {
            desc.scale_param = s.scale_meta_offset_bytes + (cfg.scale_or_addr - s.scale_meta_base_addr);
        } else if (cfg.scale_or_addr < VERSA_P_META_BYTES) {
            desc.scale_param = cfg.scale_or_addr;
        } else {
            desc.scale_param = kScaleMetaOffset;
        }
    } else if (cfg.output_type == 1 && cfg.source && cfg.is_quant) {
        desc.mode = VERSA_P_MVOUT_FP32_TENSOR_Q8_24;
        desc.scale_param = cfg.scale_or_addr;
    } else {
        throw std::runtime_error("npu_dma_mvout_async: unsupported mvout mode");
    }
    desc.o_bank = o_bank;
    const int rc = versa_p_start_mvout(require_dev(), &desc);
    if (rc != VERSA_P_OK) {
        fail_runtime("versa_p_start_mvout", rc);
    }
    s.pending_o_bank = o_bank;
    s.o_bank_inflight[o_bank] = true;
    if (trace_enabled()) {
        std::fprintf(stderr, "[VERSA_P_TRACE] start_mvout o_bank=%u m=%u n=%u dram=0x%08x stride=%u\n",
                     o_bank, desc.m, desc.n, desc.dram_base, desc.output_stride_n);
    }
    s.inflight_api[dma_id] = VERSA_P_API_MVOUT;
    s.inflight_valid[dma_id] = true;
}

void start_mvout(const MvoutConfig & cfg, uint32_t dma_id) {
    start_mvout_bank(cfg, 0, dma_id);
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
    clear_bank_state(s);
}

void npu_reset() {
    VersaPRuntimeState & s = state();
    if (s.dev != nullptr) {
        const int rc = versa_p_reset(s.dev);
        if (rc != VERSA_P_OK) {
            fail_runtime("versa_p_reset", rc);
        }
    }
    clear_bank_state(s);
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

int npu_dma_copy_to_cma(void * cma_ptr, const void * src, size_t bytes, uint32_t chunk_bytes) {
    if (cma_ptr == nullptr || src == nullptr || bytes == 0) {
        return -EINVAL;
    }
    const int rc = versa_p_dma_copy_to_cma(require_dev(), cma_ptr, src, bytes, chunk_bytes);
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

void npu_dma_mvin_a_async_bank(uint8_t a_bank, const MvinConfig * cfg) {
    if (cfg == nullptr) {
        throw std::runtime_error("npu_dma_mvin_a_async_bank: null cfg");
    }
    start_mvin_a_bank(*cfg, a_bank, 0);
}

void npu_dma_wait_a_bank(uint8_t a_bank) {
    if (a_bank > 1) {
        throw std::runtime_error("npu_dma_wait_a_bank: a_bank out of range");
    }
    VersaPRuntimeState & s = state();
    if (s.inflight_valid[0] && s.pending_a_bank == a_bank) {
        wait_api(s.inflight_api[0]);
        s.inflight_valid[0] = false;
    }
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

void npu_dma_mvout_async_bank(uint8_t o_bank, uint32_t dma_id, const MvoutConfig * cfg) {
    if (cfg == nullptr) {
        throw std::runtime_error("npu_dma_mvout_async_bank: null cfg");
    }
    start_mvout_bank(*cfg, o_bank, dma_id);
}

void npu_dma_wait_o_bank(uint8_t o_bank) {
    if (o_bank > 1) {
        throw std::runtime_error("npu_dma_wait_o_bank: o_bank out of range");
    }
    VersaPRuntimeState & s = state();
    for (uint32_t dma_id = 0; dma_id < 3; ++dma_id) {
        if (s.inflight_valid[dma_id] &&
                s.inflight_api[dma_id] == VERSA_P_API_MVOUT &&
                s.pending_o_bank == o_bank) {
            wait_api(s.inflight_api[dma_id]);
            s.inflight_valid[dma_id] = false;
            return;
        }
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
    npu_gemm_plan_start_ex_banks(
        state().loaded_a_valid ? state().loaded_a_bank : state().active_a_bank,
        w_bank,
        0,
        a_addr, b_addr, out_addr, scratch_addr, bias_addr,
        block_m, block_n, block_k, a_stride, b_stride, out_stride, bias_stride,
        have_bias, is_accumulate, asymmetric_activations);
}

void npu_gemm_plan_start_ex_banks(
        uint8_t a_bank, uint8_t w_bank, uint8_t o_bank,
        uint32_t a_addr, uint32_t b_addr, uint32_t out_addr, uint32_t scratch_addr,
        uint32_t bias_addr, uint16_t block_m, uint16_t block_n, uint16_t block_k,
        uint16_t a_stride, uint16_t b_stride, uint16_t out_stride, uint16_t bias_stride,
        bool have_bias, bool is_accumulate, bool asymmetric_activations) {
    (void)a_addr; (void)b_addr; (void)out_addr; (void)scratch_addr;
    (void)bias_addr; (void)a_stride; (void)b_stride; (void)out_stride;
    (void)bias_stride; (void)asymmetric_activations;
    VersaPRuntimeState & s = state();
    if (a_bank > 1) {
        throw std::runtime_error("npu_gemm_plan_start_ex_banks: a_bank out of range");
    }
    if (w_bank > 1) {
        throw std::runtime_error("npu_gemm_plan_start_ex_banks: w_bank out of range");
    }
    if (o_bank > 1) {
        throw std::runtime_error("npu_gemm_plan_start_ex_banks: o_bank out of range");
    }
    if (!s.a_bank[a_bank].valid || s.a_bank[a_bank].m != block_m || s.a_bank[a_bank].k != block_k) {
        if (trace_enabled()) {
            std::fprintf(stderr,
                         "[VERSA_P_TRACE] gemm_plan missing_a bank=%u loaded=%u m=%u k=%u need_m=%u need_n=%u need_k=%u\n",
                         a_bank, s.a_bank[a_bank].valid ? 1u : 0u, s.a_bank[a_bank].m, s.a_bank[a_bank].k,
                         block_m, block_n, block_k);
        }
        throw std::runtime_error("npu_gemm_plan_start_ex_banks: A bank not loaded");
    }
    if (!s.w_bank[w_bank].valid || s.w_bank[w_bank].k != block_k || s.w_bank[w_bank].n < block_n) {
        if (trace_enabled()) {
            std::fprintf(stderr,
                         "[VERSA_P_TRACE] gemm_plan missing_w bank=%u loaded=%u k=%u n=%u need_m=%u need_n=%u need_k=%u\n",
                         w_bank, s.w_bank[w_bank].valid ? 1u : 0u, s.w_bank[w_bank].k, s.w_bank[w_bank].n,
                         block_m, block_n, block_k);
        }
        throw std::runtime_error("npu_gemm_plan_start_ex_banks: W bank not loaded");
    }
    versa_p_gemm_i8_desc desc = {};
    desc.m = block_m;
    desc.n = block_n;
    desc.k = block_k;
    desc.a_bank = a_bank;
    desc.w_bank = w_bank;
    desc.accumulate_en = (is_accumulate && !have_bias) ? 1 : 0;
    desc.add_bias_en = have_bias ? 1 : 0;
    uint32_t bias_offset_bytes = kBiasMetaOffset;
    if (have_bias) {
        if (s.bias_meta_base_valid && bias_addr >= s.bias_meta_base_addr) {
            bias_offset_bytes = kBiasMetaOffset + (bias_addr - s.bias_meta_base_addr);
        } else if (bias_addr < VERSA_P_META_BYTES) {
            bias_offset_bytes = bias_addr;
        }
    }
    desc.bias_offset_bytes = bias_offset_bytes;
    desc.o_bank = o_bank;
    if (trace_enabled()) {
        std::fprintf(stderr,
                     "[VERSA_P_TRACE] start_gemm a=%u w=%u o=%u m=%u n=%u k=%u have_bias=%u logical_bias=0x%08x bias_offset=0x%08x\n",
                     a_bank, w_bank, o_bank,
                     static_cast<unsigned>(desc.m),
                     static_cast<unsigned>(desc.n),
                     static_cast<unsigned>(desc.k),
                     have_bias ? 1u : 0u,
                     bias_addr,
                     desc.bias_offset_bytes);
    }
    const int rc = versa_p_start_gemm_i8(require_dev(), &desc);
    if (rc != VERSA_P_OK) {
        fail_runtime("versa_p_start_gemm_i8", rc);
    }
    s.active_a_bank = a_bank;
    s.active_w_bank = w_bank;
    s.o_bank_inflight[o_bank] = true;
    s.gemm_o_bank = o_bank;
    s.loaded_w_bank = w_bank;
    s.loaded_w_valid = true;
    s.loaded_w_k = s.w_bank[w_bank].k;
    s.loaded_w_n = s.w_bank[w_bank].n;
    s.loaded_a_bank = a_bank;
    s.loaded_a_valid = true;
    s.loaded_a_m = s.a_bank[a_bank].m;
    s.loaded_a_k = s.a_bank[a_bank].k;
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
    s.o_bank_inflight[s.gemm_o_bank & 1u] = false;
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

bool npu_attention_log8pv_run_group(
        const int8_t * q_group,
        uint32_t group_size,
        uint32_t q_rows,
        const int8_t * k,
        const int8_t * v,
        uint32_t kv_tokens,
        uint32_t q_row_start,
        bool causal_mask,
        const uint32_t * gamma16_fix_group,
        int32_t * output_group,
        uint32_t output_stride_elems,
        uint32_t timeout_ms,
        npu_log8pv_attention_profile * profile_group) {
    const auto group_total_start = std::chrono::steady_clock::now();
    if (profile_group != nullptr) {
        for (uint32_t g = 0; g < group_size; ++g) {
            profile_group[g] = {};
        }
    }

    const uint32_t padded_tokens = align_u32(kv_tokens, VERSA_P_HW_ALIGN_ELEMS);
    const uint32_t exec_start = q_row_start & ~(VERSA_P_HW_ALIGN_ELEMS - 1u);
    const uint32_t exec_end = align_u32(q_row_start + q_rows, VERSA_P_HW_ALIGN_ELEMS);
    const uint32_t exec_rows = exec_end - exec_start;
    constexpr uint32_t chunk_rows_max = 256u;

    if (q_group == nullptr || k == nullptr || v == nullptr || gamma16_fix_group == nullptr ||
            output_group == nullptr || group_size == 0 || group_size > 8 ||
            q_rows == 0 || kv_tokens == 0 || padded_tokens > 1024u ||
            q_row_start > kv_tokens || q_rows > kv_tokens - q_row_start ||
            exec_rows == 0 || exec_end > padded_tokens ||
            (!causal_mask && (padded_tokens != kv_tokens || q_row_start != 0)) ||
            output_stride_elems < 64u) {
        return false;
    }
    for (uint32_t g = 0; g < group_size; ++g) {
        if (gamma16_fix_group[g] == 0) {
            return false;
        }
    }

    if (profile_group != nullptr) {
        for (uint32_t g = 0; g < group_size; ++g) {
            profile_group[g].kv_tokens = kv_tokens;
            profile_group[g].q_rows = q_rows;
            profile_group[g].q_row_start = q_row_start;
            profile_group[g].exec_rows = exec_rows;
            profile_group[g].group_size = group_size;
            profile_group[g].kv_reuse_hit = group_size > 1 ? 1u : 0u;
        }
    }

    auto add_profile = [&](uint32_t g, uint64_t npu_log8pv_attention_profile::* field, uint64_t us) {
        if (profile_group != nullptr) {
            profile_group[g].*field += us;
        }
    };
    auto timed_call = [&](uint64_t * elapsed_us, auto && fn) -> int {
        const auto start = std::chrono::steady_clock::now();
        const int timed_rc = fn();
        if (elapsed_us != nullptr) {
            *elapsed_us = elapsed_ns(start) / 1000u;
        }
        return timed_rc;
    };

    auto timed_start_wait = [&](uint64_t * elapsed_us, versa_p_api api, auto && start_fn) -> int {
        const auto start = std::chrono::steady_clock::now();
        int timed_rc = start_fn();
        if (timed_rc == VERSA_P_OK) {
            timed_rc = versa_p_wait(require_dev(), api, timeout_ms);
        }
        if (elapsed_us != nullptr) {
            *elapsed_us = elapsed_ns(start) / 1000u;
        }
        return timed_rc;
    };

    auto wait_timed = [&](uint64_t * elapsed_us, versa_p_api api) -> int {
        const auto start = std::chrono::steady_clock::now();
        const int timed_rc = versa_p_wait(require_dev(), api, timeout_ms);
        if (elapsed_us != nullptr) {
            *elapsed_us = elapsed_ns(start) / 1000u;
        }
        return timed_rc;
    };

    auto start_qk = [&](versa_p_device * dev, uint32_t chunk_start, uint32_t cur_rows,
                       uint8_t q_bank, uint8_t k_bank, uint8_t o_bank,
                       uint32_t gamma16_fix) -> int {
        versa_p_attention_qk_desc qk_desc = {};
        qk_desc.token_count = checked_u16(padded_tokens, "padded_tokens");
        qk_desc.q_row_start = checked_u16(chunk_start, "q_row_start");
        qk_desc.q_rows = checked_u16(cur_rows, "q_rows");
        qk_desc.gamma16_fix = gamma16_fix;
        qk_desc.q_bank = q_bank;
        qk_desc.k_bank = k_bank;
        qk_desc.o_bank = o_bank;
        qk_desc.causal_mask = causal_mask ? 1 : 0;
        return versa_p_start_attention_qk_logp(dev, &qk_desc);
    };

    auto start_logp_mvout = [&](versa_p_device * dev, uint32_t dram_offset,
                               uint32_t chunk_start, uint32_t cur_rows,
                               uint8_t o_bank) -> int {
        versa_p_attention_logp_mvout_desc logp_mvout = {};
        logp_mvout.dram_base = dram_offset;
        logp_mvout.token_count = checked_u16(padded_tokens, "padded_tokens");
        logp_mvout.q_row_start = checked_u16(chunk_start, "q_row_start");
        logp_mvout.q_rows = checked_u16(cur_rows, "q_rows");
        logp_mvout.output_stride_bytes = checked_u16(padded_tokens, "padded_tokens");
        logp_mvout.o_bank = o_bank;
        logp_mvout.causal_mask = causal_mask ? 1 : 0;
        return versa_p_start_mvout_attention_logp(dev, &logp_mvout);
    };

    auto start_mvin_p = [&](versa_p_device * dev, uint32_t dram_offset,
                           uint32_t cur_rows, uint8_t a_bank) -> int {
        versa_p_mvin_a_desc p_desc = {};
        p_desc.dram_base = dram_offset;
        p_desc.dram_row_stride_bytes = padded_tokens;
        p_desc.m = checked_u16(cur_rows, "q_rows");
        p_desc.k = checked_u16(padded_tokens, "padded_tokens");
        p_desc.a_bank = a_bank;
        return versa_p_start_mvin_a(dev, &p_desc);
    };

    auto start_pv = [&](versa_p_device * dev, uint32_t cur_rows,
                       uint8_t p_bank, uint8_t v_bank, uint8_t o_bank) -> int {
        versa_p_pv_log8_desc pv_desc = {};
        pv_desc.m = checked_u16(cur_rows, "q_rows");
        pv_desc.n = 64;
        pv_desc.k = checked_u16(padded_tokens, "padded_tokens");
        pv_desc.p_bank = p_bank;
        pv_desc.v_bank = v_bank;
        pv_desc.o_bank = o_bank;
        return versa_p_start_gemm_pv_log8(dev, &pv_desc);
    };

    auto start_pv_mvout = [&](versa_p_device * dev, versa_p_buffer & out_buf,
                             uint32_t cur_rows, uint8_t o_bank) -> int {
        versa_p_mvout_desc out_desc = {};
        out_desc.dram_base = out_buf.dma_addr;
        out_desc.m = checked_u16(cur_rows, "q_rows");
        out_desc.n = 64;
        out_desc.output_stride_n = 64;
        out_desc.mode = VERSA_P_MVOUT_RAW_I32;
        out_desc.o_bank = o_bank;
        return versa_p_start_mvout(dev, &out_desc);
    };

    auto copy_chunk_output = [&](versa_p_device * dev, versa_p_buffer & out_buf,
                                uint32_t chunk_start, uint32_t cur_rows,
                                int32_t * output) {
        (void) versa_p_sync_for_cpu(dev, out_buf.vaddr, (size_t) chunk_rows_max * 64u * sizeof(int32_t));
        const int32_t * got = static_cast<const int32_t *>(out_buf.vaddr);
        for (uint32_t local = 0; local < cur_rows; ++local) {
            const uint32_t global_row = chunk_start + local;
            if (global_row < q_row_start || global_row >= q_row_start + q_rows) {
                continue;
            }
            const uint32_t out_row = global_row - q_row_start;
            std::memcpy(output + (size_t) out_row * output_stride_elems,
                        got + (size_t) local * 64u,
                        64u * sizeof(int32_t));
        }
    };

    if (profile_group != nullptr) {
        const uint32_t chunks = (exec_rows + chunk_rows_max - 1u) / chunk_rows_max;
        for (uint32_t g = 0; g < group_size; ++g) {
            profile_group[g].chunks = chunks;
        }
    }

    VersaPRuntimeState & s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    AttentionWorkspace call_ws = {};
    const bool use_persistent_workspace = attention_workspace_enabled();

    auto cleanup = [&]() {
        versa_p_device * dev = s.dev;
        if (dev == nullptr) {
            return;
        }
        s.active_a_bank = dev->active_a_bank;
        s.active_w_bank = dev->active_w_bank;
        s.loaded_w_valid = false;
        if (!use_persistent_workspace) {
            release_attention_workspace(dev, call_ws);
        }
    };

    try {
        versa_p_device * dev = require_dev();
        clear_bank_state(s);
        s.active_a_bank = dev->active_a_bank;
        s.active_w_bank = dev->active_w_bank;

        std::vector<int8_t> q_padded((size_t) padded_tokens * 64u, 0);
        std::vector<int8_t> k_padded((size_t) padded_tokens * 64u, 0);
        std::vector<int8_t> v_padded((size_t) padded_tokens * 64u, 0);
        for (uint32_t row = 0; row < kv_tokens; ++row) {
            std::memcpy(k_padded.data() + (size_t) row * 64u, k + (size_t) row * 64u, 64u);
            std::memcpy(v_padded.data() + (size_t) row * 64u, v + (size_t) row * 64u, 64u);
        }

        std::vector<int8_t> kt = transpose_token_dim_i8(k_padded.data(), padded_tokens);
        std::vector<int8_t> packed_k;
        std::vector<int8_t> packed_v;
        pack_w_versa_i8(kt, 64u, padded_tokens, packed_k);
        pack_w_versa_i8(v_padded, padded_tokens, 64u, packed_v);

        const uint32_t q_bytes = padded_tokens * 64u;
        const uint32_t logp_bytes_one = exec_rows * padded_tokens;
        const uint32_t logp_bytes = logp_bytes_one * group_size;
        const uint32_t out_bytes = chunk_rows_max * 64u * sizeof(int32_t);

        AttentionWorkspace & ws = use_persistent_workspace ? s.attention_ws : call_ws;
        const bool reused_before =
            use_persistent_workspace &&
            ws.q_valid && ws.q_bytes >= align_u32(q_bytes, VERSA_P_ALIGN_BYTES) &&
            ws.k_valid && ws.k_bytes >= align_u32((uint32_t) packed_k.size(), VERSA_P_ALIGN_BYTES) &&
            ws.v_valid && ws.v_bytes >= align_u32((uint32_t) packed_v.size(), VERSA_P_ALIGN_BYTES) &&
            ws.logp_valid && ws.logp_bytes >= align_u32(logp_bytes, VERSA_P_ALIGN_BYTES) &&
            ws.out_valid && ws.out_bytes >= align_u32(out_bytes, VERSA_P_ALIGN_BYTES);
        if (!ensure_attention_buffer(dev, ws.q, ws.q_bytes, ws.q_valid, q_bytes) ||
                !ensure_attention_buffer(dev, ws.k, ws.k_bytes, ws.k_valid, (uint32_t) packed_k.size()) ||
                !ensure_attention_buffer(dev, ws.v, ws.v_bytes, ws.v_valid, (uint32_t) packed_v.size()) ||
                !ensure_attention_buffer(dev, ws.logp, ws.logp_bytes, ws.logp_valid, logp_bytes) ||
                !ensure_attention_buffer(dev, ws.out, ws.out_bytes, ws.out_valid, out_bytes)) {
            cleanup();
            return false;
        }
        if (profile_group != nullptr && reused_before) {
            for (uint32_t g = 0; g < group_size; ++g) {
                profile_group[g].workspace_reuse = 1;
            }
        }

        versa_p_buffer & q_buf = ws.q;
        versa_p_buffer & k_buf = ws.k;
        versa_p_buffer & v_buf = ws.v;
        versa_p_buffer & logp_buf = ws.logp;
        versa_p_buffer & out_buf = ws.out;

        std::memcpy(k_buf.vaddr, packed_k.data(), packed_k.size());
        std::memcpy(v_buf.vaddr, packed_v.data(), packed_v.size());
        (void) versa_p_sync_for_device(dev, k_buf.vaddr, packed_k.size());
        (void) versa_p_sync_for_device(dev, v_buf.vaddr, packed_v.size());

        const uint8_t attention_w_bank = static_cast<uint8_t>(dev->active_w_bank ^ 1u);

        versa_p_mvin_w_desc k_desc = {};
        k_desc.dram_base = k_buf.dma_addr;
        k_desc.k = 64;
        k_desc.n = checked_u16(padded_tokens, "padded_tokens");
        k_desc.w_bank = attention_w_bank;
        int rc = VERSA_P_OK;
        uint64_t elapsed = 0;
        rc = timed_call(&elapsed, [&]() {
            return versa_p_mvin_w(dev, &k_desc, timeout_ms);
        });
        if (rc != VERSA_P_OK) {
            cleanup();
            return false;
        }
        add_profile(0, &npu_log8pv_attention_profile::mvin_k_us, elapsed);

        for (uint32_t g = 0; g < group_size; ++g) {
            std::fill(q_padded.begin(), q_padded.end(), 0);
            const int8_t * q_src = q_group + (size_t) g * q_rows * 64u;
            for (uint32_t row = 0; row < q_rows; ++row) {
                std::memcpy(q_padded.data() + (size_t) (q_row_start + row) * 64u,
                            q_src + (size_t) row * 64u,
                            64u);
            }
            std::memcpy(q_buf.vaddr, q_padded.data(), q_bytes);
            (void) versa_p_sync_for_device(dev, q_buf.vaddr, q_bytes);

            versa_p_mvin_a_desc q_desc = {};
            q_desc.dram_base = q_buf.dma_addr;
            q_desc.dram_row_stride_bytes = 64u;
            q_desc.m = checked_u16(padded_tokens, "padded_tokens");
            q_desc.k = 64;
            q_desc.a_bank = 0;
            elapsed = 0;
            rc = timed_call(&elapsed, [&]() {
                return versa_p_mvin_a(dev, &q_desc, timeout_ms);
            });
            if (rc != VERSA_P_OK) {
                cleanup();
                return false;
            }
            add_profile(g, &npu_log8pv_attention_profile::mvin_q_us, elapsed);

            bool qk_started = false;
            uint32_t prev_chunk_start = 0;
            uint32_t prev_rows = 0;
            uint8_t prev_o_bank = 0;
            for (uint32_t chunk_start = exec_start, chunk_idx = 0;
                    chunk_start < exec_end;
                    chunk_start += chunk_rows_max, ++chunk_idx) {
                const uint32_t cur_rows = std::min(chunk_rows_max, exec_end - chunk_start);
                const uint8_t o_bank = static_cast<uint8_t>(chunk_idx & 1u);
                if (!qk_started) {
                    rc = start_qk(dev, chunk_start, cur_rows, 0, attention_w_bank, o_bank,
                            gamma16_fix_group[g]);
                } else {
                    elapsed = 0;
                    rc = wait_timed(&elapsed, VERSA_P_API_GEMM_I8);
                    const uint32_t prev_logp_offset =
                        g * logp_bytes_one + (prev_chunk_start - exec_start) * padded_tokens;
                    const auto mvout_start = std::chrono::steady_clock::now();
                    rc = rc == VERSA_P_OK ? start_logp_mvout(
                            dev,
                            logp_buf.dma_addr + prev_logp_offset,
                            prev_chunk_start,
                            prev_rows,
                            prev_o_bank) : rc;
                    rc = rc == VERSA_P_OK ? start_qk(dev, chunk_start, cur_rows, 0, attention_w_bank, o_bank,
                            gamma16_fix_group[g]) : rc;
                    uint64_t mvout_elapsed = 0;
                    rc = rc == VERSA_P_OK ? wait_timed(&mvout_elapsed, VERSA_P_API_MVOUT) : rc;
                    mvout_elapsed = elapsed_ns(mvout_start) / 1000u;
                    add_profile(g, &npu_log8pv_attention_profile::logp_mvout_us, mvout_elapsed);
                    add_profile(g, &npu_log8pv_attention_profile::qk_overlap_us,
                            std::min(elapsed, mvout_elapsed));
                    (void) versa_p_sync_for_device(
                            dev,
                            static_cast<uint8_t *>(logp_buf.vaddr) + prev_logp_offset,
                            (size_t) prev_rows * padded_tokens);
                    add_profile(g, &npu_log8pv_attention_profile::qk_us, elapsed);
                }
                if (rc != VERSA_P_OK) {
                    cleanup();
                    return false;
                }
                qk_started = true;
                prev_chunk_start = chunk_start;
                prev_rows = cur_rows;
                prev_o_bank = o_bank;
            }
            if (qk_started) {
                elapsed = 0;
                rc = wait_timed(&elapsed, VERSA_P_API_GEMM_I8);
                add_profile(g, &npu_log8pv_attention_profile::qk_us, elapsed);
                const uint32_t prev_logp_offset =
                    g * logp_bytes_one + (prev_chunk_start - exec_start) * padded_tokens;
                uint64_t mvout_elapsed = 0;
                rc = rc == VERSA_P_OK ? timed_start_wait(&mvout_elapsed, VERSA_P_API_MVOUT, [&]() {
                    return start_logp_mvout(
                            dev,
                            logp_buf.dma_addr + prev_logp_offset,
                            prev_chunk_start,
                            prev_rows,
                            prev_o_bank);
                }) : rc;
                if (rc != VERSA_P_OK) {
                    cleanup();
                    return false;
                }
                add_profile(g, &npu_log8pv_attention_profile::logp_mvout_us, mvout_elapsed);
                (void) versa_p_sync_for_device(
                        dev,
                        static_cast<uint8_t *>(logp_buf.vaddr) + prev_logp_offset,
                        (size_t) prev_rows * padded_tokens);
            }
        }

        versa_p_mvin_w_desc v_desc = {};
        v_desc.dram_base = v_buf.dma_addr;
        v_desc.k = checked_u16(padded_tokens, "padded_tokens");
        v_desc.n = 64;
        v_desc.w_bank = attention_w_bank;
        elapsed = 0;
        rc = timed_call(&elapsed, [&]() {
            return versa_p_mvin_w(dev, &v_desc, timeout_ms);
        });
        if (rc != VERSA_P_OK) {
            cleanup();
            return false;
        }
        add_profile(0, &npu_log8pv_attention_profile::mvin_v_us, elapsed);

        for (uint32_t g = 0; g < group_size; ++g) {
            int32_t * output = output_group + (size_t) g * q_rows * output_stride_elems;
            bool have_pv = false;
            uint32_t cur_chunk_start = 0;
            uint32_t cur_rows = 0;
            uint8_t cur_a_bank = 0;
            uint8_t cur_o_bank = 0;
            for (uint32_t chunk_start = exec_start, chunk_idx = 0;
                    chunk_start < exec_end;
                    chunk_start += chunk_rows_max, ++chunk_idx) {
                const uint32_t rows = std::min(chunk_rows_max, exec_end - chunk_start);
                const uint32_t logp_offset =
                    g * logp_bytes_one + (chunk_start - exec_start) * padded_tokens;
                const uint8_t a_bank = static_cast<uint8_t>(chunk_idx & 1u);
                const uint8_t o_bank = static_cast<uint8_t>(chunk_idx & 1u);

                uint64_t mvin_elapsed = 0;
                if (!have_pv) {
                    rc = timed_start_wait(&mvin_elapsed, VERSA_P_API_MVIN_A, [&]() {
                        return start_mvin_p(dev, logp_buf.dma_addr + logp_offset, rows, a_bank);
                    });
                    if (rc != VERSA_P_OK) {
                        cleanup();
                        return false;
                    }
                    add_profile(g, &npu_log8pv_attention_profile::mvin_p_us, mvin_elapsed);
                    rc = start_pv(dev, rows, a_bank, attention_w_bank, o_bank);
                    if (rc != VERSA_P_OK) {
                        cleanup();
                        return false;
                    }
                    cur_chunk_start = chunk_start;
                    cur_rows = rows;
                    cur_a_bank = a_bank;
                    cur_o_bank = o_bank;
                    have_pv = true;
                    continue;
                }

                const auto mvin_start = std::chrono::steady_clock::now();
                rc = start_mvin_p(dev, logp_buf.dma_addr + logp_offset, rows, a_bank);
                if (rc != VERSA_P_OK) {
                    cleanup();
                    return false;
                }
                elapsed = 0;
                rc = wait_timed(&elapsed, VERSA_P_API_GEMM_I8);
                add_profile(g, &npu_log8pv_attention_profile::pv_us, elapsed);
                const auto mvout_start = std::chrono::steady_clock::now();
                rc = rc == VERSA_P_OK ? start_pv_mvout(dev, out_buf, cur_rows, cur_o_bank) : rc;
                if (rc != VERSA_P_OK) {
                    cleanup();
                    return false;
                }

                uint64_t wait_mvin_elapsed = 0;
                rc = wait_timed(&wait_mvin_elapsed, VERSA_P_API_MVIN_A);
                if (rc != VERSA_P_OK) {
                    cleanup();
                    return false;
                }
                mvin_elapsed = elapsed_ns(mvin_start) / 1000u;
                add_profile(g, &npu_log8pv_attention_profile::mvin_p_us, mvin_elapsed);

                rc = start_pv(dev, rows, a_bank, attention_w_bank, o_bank);
                if (rc != VERSA_P_OK) {
                    cleanup();
                    return false;
                }
                uint64_t mvout_elapsed = 0;
                rc = wait_timed(&mvout_elapsed, VERSA_P_API_MVOUT);
                if (rc != VERSA_P_OK) {
                    cleanup();
                    return false;
                }
                mvout_elapsed = elapsed_ns(mvout_start) / 1000u;
                add_profile(g, &npu_log8pv_attention_profile::pv_mvout_us, mvout_elapsed);
                add_profile(g, &npu_log8pv_attention_profile::pv_overlap_us,
                        std::min(mvin_elapsed, elapsed + mvout_elapsed));
                copy_chunk_output(dev, out_buf, cur_chunk_start, cur_rows, output);
                cur_chunk_start = chunk_start;
                cur_rows = rows;
                cur_a_bank = a_bank;
                cur_o_bank = o_bank;
            }
            if (have_pv) {
                elapsed = 0;
                rc = wait_timed(&elapsed, VERSA_P_API_GEMM_I8);
                add_profile(g, &npu_log8pv_attention_profile::pv_us, elapsed);
                uint64_t mvout_elapsed = 0;
                rc = rc == VERSA_P_OK ? timed_start_wait(&mvout_elapsed, VERSA_P_API_MVOUT, [&]() {
                    return start_pv_mvout(dev, out_buf, cur_rows, cur_o_bank);
                }) : rc;
                if (rc != VERSA_P_OK) {
                    cleanup();
                    return false;
                }
                add_profile(g, &npu_log8pv_attention_profile::pv_mvout_us, mvout_elapsed);
                copy_chunk_output(dev, out_buf, cur_chunk_start, cur_rows, output);
            }
            (void) cur_a_bank;
        }

        const uint64_t group_total_us = elapsed_ns(group_total_start) / 1000u;
        if (profile_group != nullptr) {
            for (uint32_t g = 0; g < group_size; ++g) {
                npu_log8pv_attention_profile & p = profile_group[g];
                p.total_us = p.mvin_q_us + p.mvin_k_us + p.mvin_v_us + p.qk_us +
                    p.logp_mvout_us + p.mvin_p_us + p.pv_us + p.pv_mvout_us;
                if (group_size == 1) {
                    p.total_us = group_total_us;
                }
            }
        }
        cleanup();
        return true;
    } catch (...) {
        cleanup();
        return false;
    }
}

bool npu_attention_log8pv_run_ex(
        const int8_t * q,
        uint32_t q_rows,
        const int8_t * k,
        const int8_t * v,
        uint32_t kv_tokens,
        uint32_t q_row_start,
        bool causal_mask,
        uint32_t gamma16_fix,
        int32_t * output,
        uint32_t output_stride_elems,
        uint32_t timeout_ms,
        npu_log8pv_attention_profile * profile) {
    return npu_attention_log8pv_run_group(
            q,
            1,
            q_rows,
            k,
            v,
            kv_tokens,
            q_row_start,
            causal_mask,
            &gamma16_fix,
            output,
            output_stride_elems,
            timeout_ms,
            profile);
}

bool npu_attention_log8pv_run(
        const int8_t * q,
        const int8_t * k,
        const int8_t * v,
        uint32_t tokens,
        bool causal_mask,
        uint32_t gamma16_fix,
        int32_t * output,
        uint32_t output_stride_elems,
        uint32_t timeout_ms) {
    return npu_attention_log8pv_run_ex(
            q,
            tokens,
            k,
            v,
            tokens,
            0,
            causal_mask,
            gamma16_fix,
            output,
            output_stride_elems,
            timeout_ms,
            nullptr);
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
