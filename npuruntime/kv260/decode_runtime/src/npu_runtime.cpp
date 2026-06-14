#include "npu_runtime.h"

#include "npu_kv260_uapi.h"
#include "npu_regs_compat.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <fcntl.h>
#include <iterator>
#include <limits>
#include <map>
#include <new>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

struct npu_device {
    uint32_t magic;
};

namespace {

constexpr uint32_t kDefaultCmaMapSize = 256u * 1024u * 1024u;
constexpr uint32_t kRegMapSize = NPU_KV260_REG_MMAP_SIZE;
constexpr off_t kRegMapOffset = NPU_KV260_MMAP_REGS_OFFSET;
constexpr size_t kAlignment = 256;

// GEMV SPM is a single 512 KiB address space in RTL, but stream GEMV reserves
// fixed regions for one active cached weight block and the next staged block.
// Weight MVINs must fit in one region at a time; do not treat
// 0x10000..SPM_END as one contiguous block.  The supported per-block weight
// capacity is min(region0, region1):
//   region0: 0x10000..0x3ffff = 192 KiB
//   region1: 0x40000..0x7ffff = 256 KiB
// so the safe weight block limit is 192 KiB.  For example, W8A16 with K=64
// uses 2048 bytes per 32-row tile, hence max 96 row tiles / 3072 rows.
constexpr uint32_t kGemvStreamScaleSpmBase = 0x00000;
constexpr uint32_t kGemvPingWeightSpmBase = 0x10000;
constexpr uint32_t kGemvPongWeightSpmBase = 0x40000;
constexpr uint32_t kGemvScaleTileBytes = NPU_GEMV_LINE_BYTES;
constexpr uint32_t kGemvWeightW4RowBytes = NPU_GEMV_TILE_ELEMS / 2;
constexpr uint32_t kGemvWeightW8TileElems = NPU_GEMV_TILE_ELEMS / 2;
constexpr uint32_t kGemvWeightW8RowBytes = kGemvWeightW8TileElems;
constexpr uint32_t kGemvWeightW16RowBytes = NPU_GEMV_TILE_ELEMS * NPU_GEMV_FP16_BYTES;
constexpr uint32_t kRopeLutWindowBytes = NPU_ROPE_LUT_WINDOW_BYTES;
constexpr uint64_t kApiStatusBusy = 1ull << 0;
constexpr uint64_t kApiStatusDone = 1ull << 1;
constexpr uint64_t kApiStatusError = 1ull << 2;
constexpr uint32_t kApiStatusErrorCodeShift = 4;
constexpr uint32_t kNpuDeviceMagic = 0x4E505544u; // "NPUD"
constexpr uint16_t kDecodeAttentionHeadDim = 64;
constexpr uint16_t kDecodeAttentionKvHeads = 5;
constexpr uint16_t kDecodeAttentionKvDim =
    kDecodeAttentionHeadDim * kDecodeAttentionKvHeads;
constexpr uint16_t kGemvStreamCacheInvalidateTag = 0x8000;
#ifndef NPU_POLL_SPIN_COUNT
#define NPU_POLL_SPIN_COUNT 10000
#endif

#ifndef NPU_POLL_YIELD_COUNT
#define NPU_POLL_YIELD_COUNT 100
#endif

#ifndef NPU_POLL_YIELD_US
#define NPU_POLL_YIELD_US 1
#endif

#ifdef NPU_DEBUG
#define NPU_LOG(fmt, ...) std::fprintf(stdout, "[NPU] " fmt "\n", ##__VA_ARGS__)
#else
#define NPU_LOG(fmt, ...) do { } while (0)
#endif

#define NPU_ERR(fmt, ...) std::fprintf(stderr, "[NPU][ERR] " fmt "\n", ##__VA_ARGS__)

struct GemvPingPongConfig {
    void* act_ptr;
    void* act_scale_ptr;
    void* act_scale2_ptr;
    void* scale_ptr;
    void* weight_ptr;
    void* output_ptr;
    void* rope_lut_ptr;
    uint16_t m;
    uint16_t n;
    uint8_t gemv_mode;
    uint8_t output_precision;
    bool enable_act_scale;
    bool enable_act_scale2;
    uint64_t decode_flow;
    uint16_t cache_cell_idx;
    uint32_t act_group_stride_bytes;
    bool cached_group_enable;
};

uint32_t ceil_div_u32(uint32_t a, uint32_t b) {
    return (a + b - 1u) / b;
}

uint32_t align_up_u32(uint32_t value, uint32_t alignment) {
    return (value + alignment - 1u) & ~(alignment - 1u);
}

uint32_t gemv_tile_elems(uint8_t gemv_mode) {
    return (gemv_mode == NPU_GEMV_MODE_W8A16)
        ? static_cast<uint32_t>(kGemvWeightW8TileElems)
        : static_cast<uint32_t>(NPU_GEMV_TILE_ELEMS);
}

uint32_t gemv_weight_tile_bytes(uint8_t gemv_mode) {
    uint32_t row_bytes = kGemvWeightW4RowBytes;
    if (gemv_mode == NPU_GEMV_MODE_W8A16) {
        row_bytes = kGemvWeightW8RowBytes;
    } else if (gemv_mode == NPU_GEMV_MODE_W16A16) {
        row_bytes = kGemvWeightW16RowBytes;
    }
    const uint32_t data_bytes = NPU_GEMV_ROW_TILE_ELEMS * row_bytes;
    return data_bytes;
}

bool gemv_mode_uses_weight_scale(uint8_t gemv_mode) {
    return gemv_mode == NPU_GEMV_MODE_W4A16 || gemv_mode == NPU_GEMV_MODE_W8A16;
}

bool decode_flow_unit_weight_scale(uint64_t decode_flow) {
    return (decode_flow & NPU_DECODE_FLAG_UNIT_WEIGHT_SCALE) != 0;
}

bool gemv_request_uses_weight_scale(uint8_t gemv_mode, uint64_t decode_flow) {
    return gemv_mode_uses_weight_scale(gemv_mode) &&
           !decode_flow_unit_weight_scale(decode_flow);
}

bool decode_flow_uses_rope_lut(uint64_t decode_flow) {
    return ((decode_flow >> 8) & 0x3u) == NPU_DECODE_UNARY_ROPE;
}

bool decode_flow_kv_quant_scratch(uint64_t decode_flow) {
    return (decode_flow & NPU_DECODE_FLAG_KV_QUANT) != 0 &&
           (decode_flow & NPU_DECODE_FLAG_KV_QUANT_SCRATCH) != 0;
}

uint32_t act_scale_bytes_for(uint16_t mat_width, uint8_t gemv_mode,
                             uint64_t decode_flow, bool enable_act_scale) {
    if (!enable_act_scale) {
        return 0;
    }
    const uint32_t col_tiles = ceil_div_u32(mat_width, gemv_tile_elems(gemv_mode));
    const uint32_t act_tile_bytes = gemv_tile_elems(gemv_mode) * NPU_GEMV_FP16_BYTES;
    (void)decode_flow;
    return col_tiles * act_tile_bytes;
}

uint32_t group_count_from_decode_flow(uint64_t decode_flow) {
    return static_cast<uint32_t>((decode_flow >> 25) & 0x3u) + 1u;
}

uint32_t grouped_act_span_bytes(uint32_t act_bytes, uint32_t group_count,
                                uint32_t act_group_stride_bytes) {
    if (group_count <= 1u) {
        return act_bytes;
    }
    const uint32_t stride =
        act_group_stride_bytes != 0 ? act_group_stride_bytes : act_bytes;
    return (group_count - 1u) * stride + act_bytes;
}

uint64_t make_decode_flow_value(uint8_t src0, uint8_t src1,
                                uint8_t unary_op, uint8_t binary_op,
                                uint8_t reduce_op, uint8_t dst,
                                uint8_t src_buffer_id, uint8_t dst_buffer_id,
                                uint16_t elem_count, uint16_t position) {
    return (uint64_t(src0 & 0xfu) << 0) |
           (uint64_t(src1 & 0xfu) << 4) |
           (uint64_t(unary_op & 0x3u) << 8) |
           (uint64_t(binary_op & 0x1u) << 10) |
           (uint64_t(reduce_op & 0x3u) << 11) |
           (uint64_t(dst & 0x7u) << 13) |
           (uint64_t(src_buffer_id & 0xfu) << 16) |
           (uint64_t(dst_buffer_id & 0xfu) << 20) |
           (uint64_t(elem_count) << 32) |
           (uint64_t(position) << 48);
}

uint64_t parse_size_with_suffix(const char* value, uint64_t fallback) {
    if (!value || value[0] == '\0') {
        return fallback;
    }

    errno = 0;
    char* end = nullptr;
    unsigned long long parsed = std::strtoull(value, &end, 0);
    if (errno || end == value) {
        return fallback;
    }

    while (*end && std::isspace(static_cast<unsigned char>(*end))) {
        ++end;
    }

    uint64_t multiplier = 1;
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
    return parsed * multiplier;
}

uint32_t resolve_cma_request_size() {
    uint64_t size = parse_size_with_suffix(std::getenv("NPU_CMA_SIZE"), kDefaultCmaMapSize);
    if (size == 0 || size > std::numeric_limits<uint32_t>::max()) {
        size = kDefaultCmaMapSize;
    }
    return static_cast<uint32_t>(size);
}

enum class NpuWaitStrategy {
    Hybrid,
    Spin,
    Irq,
};

struct PreparedWaitContext {
    NpuWaitStrategy strategy = NpuWaitStrategy::Hybrid;
    std::string op;
    std::string shape_key;
    uint64_t bytes = 0;
    bool prepared = false;
};

thread_local PreparedWaitContext g_prepared_wait;

struct GemvBlockShadow {
    uint64_t desc0 = 0;
    uint64_t desc1 = 0;
    uint64_t desc2 = 0;
    uint64_t desc3 = 0;
    uint64_t desc4 = 0;
    uint64_t desc5 = 0;
    uint64_t desc6 = 0;
    uint64_t desc7 = 0;
    const char* tag = "unset";
};

GemvBlockShadow g_last_gemv_block;
uint32_t g_device_refcount = 0;
bool g_runtime_owned_by_device_api = false;

bool is_default_dev_path(const char* dev_path) {
    return dev_path == nullptr || dev_path[0] == '\0' ||
           std::strcmp(dev_path, NPU_KV260_DEV_PATH) == 0;
}

bool valid_device(const npu_device* dev) {
    return dev && dev->magic == kNpuDeviceMagic;
}

bool valid_output_precision(decode_output_precision precision) {
    return precision == DECODE_OUTPUT_FP16 || precision == DECODE_OUTPUT_FP32;
}

uint8_t output_precision_to_hw(decode_output_precision precision) {
    return precision == DECODE_OUTPUT_FP32 ? 3u : 1u;
}

bool valid_decode_gemv_mode(decode_gemv_mode mode) {
    return mode == DECODE_GEMV_W4A16 || mode == DECODE_GEMV_W8A16;
}

int validate_timeout_default_only(uint32_t timeout_ms) {
    return timeout_ms == 0 ? 0 : -ENOTSUP;
}

const char* wait_strategy_name(NpuWaitStrategy strategy) {
    switch (strategy) {
    case NpuWaitStrategy::Spin: return "spin";
    case NpuWaitStrategy::Irq: return "irq";
    case NpuWaitStrategy::Hybrid:
    default: return "hybrid";
    }
}

NpuWaitStrategy parse_wait_strategy(const std::string& value, NpuWaitStrategy fallback) {
    std::string lower;
    lower.reserve(value.size());
    for (char c : value) {
        lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    if (lower == "spin") return NpuWaitStrategy::Spin;
    if (lower == "irq" || lower == "kernel") return NpuWaitStrategy::Irq;
    if (lower == "hybrid") return NpuWaitStrategy::Hybrid;
    return fallback;
}

NpuWaitStrategy env_wait_policy() {
    const char* env = std::getenv("NPU_WAIT_POLICY");
    if (env == nullptr || env[0] == '\0' || std::strcmp(env, "adaptive") == 0) {
        return NpuWaitStrategy::Hybrid;
    }
    return parse_wait_strategy(env, NpuWaitStrategy::Hybrid);
}

bool adaptive_wait_enabled() {
    const char* env = std::getenv("NPU_WAIT_POLICY");
    return env != nullptr && std::strcmp(env, "adaptive") == 0;
}

std::optional<std::string> extract_json_string_field(const std::string& object, const char* key) {
    const std::string needle = std::string("\"") + key + "\"";
    size_t pos = object.find(needle);
    if (pos == std::string::npos) {
        return std::nullopt;
    }
    pos = object.find(':', pos + needle.size());
    if (pos == std::string::npos) {
        return std::nullopt;
    }
    pos = object.find('"', pos + 1);
    if (pos == std::string::npos) {
        return std::nullopt;
    }
    std::string out;
    bool escape = false;
    for (size_t i = pos + 1; i < object.size(); ++i) {
        const char c = object[i];
        if (escape) {
            out.push_back(c);
            escape = false;
            continue;
        }
        if (c == '\\') {
            escape = true;
            continue;
        }
        if (c == '"') {
            return out;
        }
        out.push_back(c);
    }
    return std::nullopt;
}

struct WaitPolicyTable {
    bool loaded = false;
    NpuWaitStrategy default_strategy = NpuWaitStrategy::Hybrid;
    std::map<std::string, NpuWaitStrategy> entries;
};

WaitPolicyTable& get_wait_policy_table() {
    static WaitPolicyTable table;
    if (table.loaded) {
        return table;
    }
    table.loaded = true;

    const char* path = std::getenv("NPU_WAIT_POLICY_JSON");
    if (path == nullptr || path[0] == '\0') {
        return table;
    }

    std::ifstream in(path);
    if (!in.is_open()) {
        std::fprintf(stderr, "[NPU_WAIT] failed to open policy JSON: %s\n", path);
        return table;
    }

    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (auto default_strategy = extract_json_string_field(content, "default_strategy")) {
        table.default_strategy = parse_wait_strategy(*default_strategy, table.default_strategy);
    }

    size_t pos = 0;
    while ((pos = content.find('{', pos)) != std::string::npos) {
        size_t end = content.find('}', pos + 1);
        if (end == std::string::npos) {
            break;
        }
        const std::string object = content.substr(pos, end - pos + 1);
        auto op = extract_json_string_field(object, "op");
        auto shape = extract_json_string_field(object, "shape_key");
        auto strategy = extract_json_string_field(object, "strategy");
        if (op && shape && strategy) {
            table.entries[*op + "\t" + *shape] =
                parse_wait_strategy(*strategy, table.default_strategy);
        }
        pos = end + 1;
    }

    return table;
}

NpuWaitStrategy select_wait_strategy(const std::string& op, const std::string& shape_key) {
    if (!adaptive_wait_enabled()) {
        return env_wait_policy();
    }
    WaitPolicyTable& table = get_wait_policy_table();
    const auto it = table.entries.find(op + "\t" + shape_key);
    return it == table.entries.end() ? table.default_strategy : it->second;
}

std::string json_escape(const std::string& value) {
    std::string out;
    out.reserve(value.size() + 8);
    for (char c : value) {
        switch (c) {
        case '\\': out += "\\\\"; break;
        case '"': out += "\\\""; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default: out.push_back(c); break;
        }
    }
    return out;
}

uint64_t elapsed_us(std::chrono::steady_clock::time_point start) {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - start).count());
}

void write_wait_trace(const std::string& op,
                      const std::string& shape_key,
                      uint64_t bytes,
                      uint64_t wait_us,
                      NpuWaitStrategy strategy,
                      const char* path_kind,
                      int spin_iters,
                      int yield_iters,
                      bool kernel_wait,
                      const char* status,
                      uint64_t full_us = 0) {
    const char* trace_path = std::getenv("NPU_WAIT_TRACE_JSONL");
    if (trace_path == nullptr || trace_path[0] == '\0') {
        return;
    }

    std::ofstream out(trace_path, std::ios::app);
    if (!out.is_open()) {
        return;
    }

    out << "{"
        << "\"op\":\"" << json_escape(op) << "\","
        << "\"shape_key\":\"" << json_escape(shape_key) << "\","
        << "\"bytes\":" << bytes << ","
        << "\"wait_us\":" << wait_us << ","
        << "\"strategy\":\"" << wait_strategy_name(strategy) << "\","
        << "\"path\":\"" << json_escape(path_kind ? path_kind : "") << "\","
        << "\"spin_iters\":" << spin_iters << ","
        << "\"yield_iters\":" << yield_iters << ","
        << "\"kernel_wait\":" << (kernel_wait ? "true" : "false") << ","
        << "\"status\":\"" << json_escape(status ? status : "") << "\","
        << "\"full_us\":" << full_us
        << "}\n";
}

class NpuRuntime {
public:
    NpuRuntime() = default;
    ~NpuRuntime();

    bool init(bool require_decode = true);
    bool reinit();
    void reset();
    bool hardware_ready() const { return hardware_ready_; }
    bool activate_decode();

    void* memory_base() const { return data_virt_base_; }
    uint32_t memory_size() const { return data_map_size_; }

    void* alloc(size_t size);
    void free_mem(void* ptr);

    void run_gemv_stream(const GemvPingPongConfig& cfg,
                         uint32_t weight_row_tile_stride_bytes = 0,
                         uint16_t weight_capacity_tokens = 0);
    void read_kv_scale(bool is_v, npu_stream_kv_scale_result* out) const;

private:
    struct alignas(256) BlockHeader {
        size_t size;
        bool is_free;
        BlockHeader* next;
        BlockHeader* prev;
    };

    struct ShadowRegs {
        uint64_t mvin_cfg = ~0ull;
        uint64_t mvout_cfg = ~0ull;
        uint64_t compute_cfg = ~0ull;
    };

    int fd_ = -1;
    void* regs_virt_base_ = nullptr;
    void* data_virt_base_ = nullptr;
    uint32_t data_phy_base_ = 0;
    uint32_t data_map_size_ = 0;
    bool hardware_ready_ = false;
    BlockHeader* free_list_head_ = nullptr;
    BlockHeader* alloc_cursor_ = nullptr;
    ShadowRegs shadow_;

    void init_allocator();
    void coalesce(BlockHeader* block);
    uint32_t virt_to_phys(void* ptr) const;

    void reg_write(uint32_t offset, uint32_t val);
    void reg_write64(uint32_t offset, uint64_t val);
    uint32_t reg_read(uint32_t offset) const;
    uint64_t reg_read64(uint32_t offset) const;
    void ack_irq(uint32_t mask);
    void dump_irq_regs();
    void dump_gemv_block_regs(const char* tag);
    void prepare_wait_irq_before_start(const std::string& op, const std::string& shape_key,
                                       uint64_t bytes, uint32_t expected_mask);
    void wait_irq(uint32_t expected_mask);
};

NpuRuntime* g_runtime = nullptr;

NpuRuntime& runtime() {
    if (!g_runtime) {
        if (npu_init() != 0 || !g_runtime) {
            throw std::runtime_error("NPU runtime is not initialized");
        }
    }
    return *g_runtime;
}

int dma_offset_for_ptr(const void* ptr, uint32_t* out_offset) {
    if (!ptr || !out_offset) {
        return -EINVAL;
    }
    void* base_ptr = runtime().memory_base();
    const uint32_t map_size = runtime().memory_size();
    if (!base_ptr || map_size == 0) {
        return -EIO;
    }

    const uintptr_t base = reinterpret_cast<uintptr_t>(base_ptr);
    const uintptr_t value = reinterpret_cast<uintptr_t>(ptr);
    if (value < base || value >= base + map_size) {
        return -ERANGE;
    }
    const uint32_t offset = static_cast<uint32_t>(value - base);
    if ((offset % kAlignment) != 0) {
        return -EINVAL;
    }
    *out_offset = offset;
    return 0;
}

uint16_t stream_group_count_or_default(const npu_stream_gemv_desc& desc) {
    return desc.group_count == 0 ? 1 : desc.group_count;
}

bool valid_stream_role(npu_stream_gemv_role role) {
    return role == NPU_STREAM_GEMV_ROLE_LINEAR ||
           role == NPU_STREAM_GEMV_ROLE_QK ||
           role == NPU_STREAM_GEMV_ROLE_PV ||
           role == NPU_STREAM_GEMV_ROLE_KV_PROJ ||
           role == NPU_STREAM_GEMV_ROLE_MLP;
}

bool valid_stream_dst(npu_stream_gemv_dst dst) {
    return dst == NPU_STREAM_GEMV_DST_OUTPUT ||
           dst == NPU_STREAM_GEMV_DST_ACT ||
           dst == NPU_STREAM_GEMV_DST_POST;
}

bool valid_stream_post_op(npu_stream_post_op post_op) {
    return post_op == NPU_STREAM_POST_BYPASS ||
           post_op == NPU_STREAM_POST_ROPE ||
           post_op == NPU_STREAM_POST_SILU ||
           post_op == NPU_STREAM_POST_SOFTMAX;
}

uint8_t stream_dst_to_decode_dst(npu_stream_gemv_dst dst) {
    switch (dst) {
    case NPU_STREAM_GEMV_DST_OUTPUT:
        return NPU_DECODE_DST_OUTPUT_SPM;
    case NPU_STREAM_GEMV_DST_ACT:
        return NPU_DECODE_DST_ACT_BUFFER;
    case NPU_STREAM_GEMV_DST_POST:
        return NPU_DECODE_DST_POST_BUFFER;
    }
    return 0;
}

uint8_t stream_post_to_unary(npu_stream_post_op post_op) {
    switch (post_op) {
    case NPU_STREAM_POST_ROPE:
        return NPU_DECODE_UNARY_ROPE;
    case NPU_STREAM_POST_SILU:
        return NPU_DECODE_UNARY_SILU;
    case NPU_STREAM_POST_BYPASS:
    case NPU_STREAM_POST_SOFTMAX:
        return NPU_DECODE_UNARY_BYPASS;
    }
    return NPU_DECODE_UNARY_BYPASS;
}

uint8_t stream_post_to_reduce(npu_stream_post_op post_op) {
    return post_op == NPU_STREAM_POST_SOFTMAX
        ? NPU_DECODE_REDUCE_SOFTMAX
        : NPU_DECODE_REDUCE_BYPASS;
}

uint64_t stream_flags_to_decode_flow(uint32_t flags) {
    uint64_t flow = 0;
    if (flags & NPU_STREAM_GEMV_F_KV_QUANT) {
        flow |= NPU_DECODE_FLAG_KV_QUANT |
                NPU_DECODE_FLAG_KV_QUANT_SCRATCH;
        if (flags & NPU_STREAM_GEMV_F_KV_IS_V) {
            flow |= NPU_DECODE_FLAG_KV_V_SEPARATED;
        }
    }
    if (flags & NPU_STREAM_GEMV_F_KV_COL_SCALE) {
        flow |= NPU_DECODE_FLAG_KV_COL_SCALE;
    }
    if (flags & NPU_STREAM_GEMV_F_UNIT_WEIGHT_SCALE) {
        flow |= NPU_DECODE_FLAG_UNIT_WEIGHT_SCALE;
    }
    return flow;
}

uint64_t stream_decode_flow_from_desc(const npu_stream_gemv_desc& desc) {
    const uint16_t elem_count = desc.elem_count != 0 ? desc.elem_count : desc.m;
    const uint16_t group_count = stream_group_count_or_default(desc);
    const bool needs_post_flow =
        desc.post_op != NPU_STREAM_POST_BYPASS ||
        desc.dst != NPU_STREAM_GEMV_DST_OUTPUT ||
        desc.elem_count != 0 ||
        desc.position != 0 ||
        desc.flags != 0 ||
        group_count != 1;
    uint64_t flow = needs_post_flow
        ? make_decode_flow_value(NPU_DECODE_SRC_GEMV_STREAM,
                                 0,
                                 stream_post_to_unary(desc.post_op),
                                 NPU_DECODE_BINARY_BYPASS,
                                 stream_post_to_reduce(desc.post_op),
                                 stream_dst_to_decode_dst(desc.dst),
                                 0,
                                 0,
                                 elem_count,
                                 desc.position)
        : 0;

    flow |= uint64_t(group_count - 1u) << 25;
    flow |= stream_flags_to_decode_flow(desc.flags);
    if (desc.role == NPU_STREAM_GEMV_ROLE_PV) {
        flow |= NPU_DECODE_FLAG_PV_UQ24;
    }
    return flow;
}

int validate_stream_ptr(const void* ptr) {
    uint32_t offset = 0;
    return dma_offset_for_ptr(ptr, &offset);
}

int validate_stream_gemv_desc(npu_device* dev,
                              const npu_stream_gemv_desc* desc) {
    constexpr uint32_t kKnownFlags =
        NPU_STREAM_GEMV_F_ENABLE_ACT_SCALE |
        NPU_STREAM_GEMV_F_KV_COL_SCALE |
        NPU_STREAM_GEMV_F_UNIT_WEIGHT_SCALE |
        NPU_STREAM_GEMV_F_KV_QUANT |
        NPU_STREAM_GEMV_F_KV_IS_V |
        NPU_STREAM_GEMV_F_ENABLE_ACT_SCALE2;
    constexpr uint32_t kPvRequiredFlags =
        NPU_STREAM_GEMV_F_ENABLE_ACT_SCALE |
        NPU_STREAM_GEMV_F_KV_COL_SCALE |
        NPU_STREAM_GEMV_F_UNIT_WEIGHT_SCALE;
    constexpr uint32_t kPvValueRowStrideTileBytes = 2048;

    if (!valid_device(dev) || !desc || !desc->act_ptr ||
        !desc->weight_payload_ptr || !desc->output_ptr ||
        desc->m == 0 || desc->n == 0 ||
        !valid_decode_gemv_mode(desc->mode) ||
        !valid_output_precision(desc->output_precision) ||
        !valid_stream_role(desc->role) ||
        !valid_stream_dst(desc->dst) ||
        !valid_stream_post_op(desc->post_op) ||
        (desc->flags & ~kKnownFlags) != 0) {
        return -EINVAL;
    }

    const uint16_t group_count = stream_group_count_or_default(*desc);
    const bool kv_quant = (desc->flags & NPU_STREAM_GEMV_F_KV_QUANT) != 0;
    const bool kv_is_v = (desc->flags & NPU_STREAM_GEMV_F_KV_IS_V) != 0;
    if (kv_is_v && !kv_quant) {
        return -EINVAL;
    }
    if (group_count > 4 ||
        (desc->act_group_stride_bytes != 0 &&
         (desc->act_group_stride_bytes % NPU_GEMV_LINE_BYTES) != 0)) {
        return -EINVAL;
    }
    if (group_count <= 1 && desc->act_group_stride_bytes != 0) {
        return -EINVAL;
    }
    if (group_count > 1 &&
        desc->role != NPU_STREAM_GEMV_ROLE_QK &&
        desc->role != NPU_STREAM_GEMV_ROLE_PV) {
        return -EINVAL;
    }
    if (group_count > 1 &&
        (desc->role == NPU_STREAM_GEMV_ROLE_QK ||
         desc->role == NPU_STREAM_GEMV_ROLE_PV) &&
        desc->act_group_stride_bytes != 0 &&
        (desc->act_group_stride_bytes % NPU_GEMV_MVIN_ALIGN_BYTES) != 0) {
        return -EINVAL;
    }

    if (validate_stream_ptr(desc->act_ptr) != 0 ||
        validate_stream_ptr(desc->weight_payload_ptr) != 0 ||
        validate_stream_ptr(desc->output_ptr) != 0) {
        return -EINVAL;
    }

    const bool enable_act_scale =
        (desc->flags & NPU_STREAM_GEMV_F_ENABLE_ACT_SCALE) != 0;
    const bool enable_act_scale2 =
        (desc->flags & NPU_STREAM_GEMV_F_ENABLE_ACT_SCALE2) != 0;
    if (enable_act_scale2 && !enable_act_scale) {
        return -EINVAL;
    }
    if (enable_act_scale2 &&
        (desc->mode != DECODE_GEMV_W4A16 ||
         desc->role != NPU_STREAM_GEMV_ROLE_LINEAR ||
         group_count != 1)) {
        return -EINVAL;
    }
    const uint64_t flow = stream_flags_to_decode_flow(desc->flags);
    const bool uses_weight_scale =
        gemv_request_uses_weight_scale(static_cast<uint8_t>(desc->mode), flow);

    const uint64_t full_flow = stream_decode_flow_from_desc(*desc);
    const bool uses_rope_lut = decode_flow_uses_rope_lut(full_flow);

    if ((enable_act_scale && !desc->act_scale_ptr) ||
        (enable_act_scale2 && !desc->act_scale2_ptr) ||
        (uses_rope_lut && !desc->rope_lut_ptr) ||
        (uses_weight_scale && !desc->weight_scale_ptr)) {
        return -EINVAL;
    }
    if (enable_act_scale && validate_stream_ptr(desc->act_scale_ptr) != 0) {
        return -EINVAL;
    }
    if (enable_act_scale2 && validate_stream_ptr(desc->act_scale2_ptr) != 0) {
        return -EINVAL;
    }
    if (uses_rope_lut && validate_stream_ptr(desc->rope_lut_ptr) != 0) {
        return -EINVAL;
    }
    if (uses_weight_scale && validate_stream_ptr(desc->weight_scale_ptr) != 0) {
        return -EINVAL;
    }

    if (desc->role == NPU_STREAM_GEMV_ROLE_PV) {
        if (desc->m != kDecodeAttentionHeadDim ||
            desc->mode != DECODE_GEMV_W8A16 ||
            !desc->act_scale_ptr ||
            desc->weight_scale_ptr != nullptr ||
            (desc->flags & kPvRequiredFlags) != kPvRequiredFlags ||
            desc->weight_capacity_tokens == 0 ||
            desc->n > desc->weight_capacity_tokens ||
            desc->weight_row_tile_stride_bytes == 0 ||
            (desc->weight_row_tile_stride_bytes % NPU_GEMV_MVIN_ALIGN_BYTES) != 0) {
            return -EINVAL;
        }
        const uint32_t min_stride =
            ceil_div_u32(desc->weight_capacity_tokens, 64) *
            kPvValueRowStrideTileBytes;
        if (desc->weight_row_tile_stride_bytes < min_stride) {
            return -EINVAL;
        }
    } else if (desc->weight_row_tile_stride_bytes != 0) {
        return -EINVAL;
    }

    if (desc->role == NPU_STREAM_GEMV_ROLE_QK &&
        (desc->mode != DECODE_GEMV_W8A16 ||
         desc->n != kDecodeAttentionHeadDim ||
         !desc->weight_scale_ptr)) {
        return -EINVAL;
    }

    if (kv_quant) {
        const uint16_t elem_count = desc->elem_count != 0 ? desc->elem_count : desc->m;
        if (desc->role != NPU_STREAM_GEMV_ROLE_KV_PROJ ||
            desc->mode != DECODE_GEMV_W4A16 ||
            desc->m != kDecodeAttentionKvDim ||
            elem_count != kDecodeAttentionKvDim ||
            desc->dst != NPU_STREAM_GEMV_DST_OUTPUT ||
            desc->output_precision != DECODE_OUTPUT_FP16 ||
            group_count != 1 ||
            desc->weight_row_tile_stride_bytes != 0 ||
            desc->weight_capacity_tokens != 0 ||
            (desc->flags & (NPU_STREAM_GEMV_F_KV_COL_SCALE |
                            NPU_STREAM_GEMV_F_UNIT_WEIGHT_SCALE)) != 0) {
            return -EINVAL;
        }
        if (kv_is_v) {
            if (desc->post_op != NPU_STREAM_POST_BYPASS) {
                return -EINVAL;
            }
        } else if (desc->post_op != NPU_STREAM_POST_ROPE) {
            return -EINVAL;
        }
    } else if (desc->role == NPU_STREAM_GEMV_ROLE_KV_PROJ) {
        return -EINVAL;
    }

    return 0;
}

void report_exception(const char* api, const std::exception& ex) {
    NPU_ERR("%s failed: %s", api, ex.what());
}

NpuRuntime::~NpuRuntime() {
    if (regs_virt_base_) {
        munmap(regs_virt_base_, kRegMapSize);
    }
    if (data_virt_base_) {
        munmap(data_virt_base_, data_map_size_);
    }
    if (fd_ >= 0) {
        ioctl(fd_, NPU_KV260_IOC_FREE_BUFFER);
        close(fd_);
    }
}

bool NpuRuntime::init(bool require_decode) {
    fd_ = open(NPU_KV260_DEV_PATH, O_RDWR);
    if (fd_ < 0) {
        perror("open " NPU_KV260_DEV_PATH);
        return false;
    }

    npu_kv260_buffer_request req = {};
    req.size = resolve_cma_request_size();
    if (ioctl(fd_, NPU_KV260_IOC_ALLOC_BUFFER, &req) < 0) {
        perror("NPU_KV260_IOC_ALLOC_BUFFER");
        close(fd_);
        fd_ = -1;
        return false;
    }

    if (req.dma_addr == 0 || req.size == 0 ||
        req.dma_addr > std::numeric_limits<uint32_t>::max() ||
        req.size > std::numeric_limits<uint32_t>::max()) {
        NPU_ERR("invalid CMA buffer dma=0x%llx size=0x%llx",
                static_cast<unsigned long long>(req.dma_addr),
                static_cast<unsigned long long>(req.size));
        ioctl(fd_, NPU_KV260_IOC_FREE_BUFFER);
        close(fd_);
        fd_ = -1;
        return false;
    }

    data_phy_base_ = static_cast<uint32_t>(req.dma_addr);
    data_map_size_ = static_cast<uint32_t>(req.size);

    regs_virt_base_ = mmap(nullptr, kRegMapSize, PROT_READ | PROT_WRITE,
                           MAP_SHARED, fd_, kRegMapOffset);
    if (regs_virt_base_ == MAP_FAILED) {
        perror("mmap regs");
        regs_virt_base_ = nullptr;
        ioctl(fd_, NPU_KV260_IOC_FREE_BUFFER);
        close(fd_);
        fd_ = -1;
        return false;
    }

    data_virt_base_ = mmap(nullptr, data_map_size_, PROT_READ | PROT_WRITE,
                           MAP_SHARED, fd_, NPU_KV260_MMAP_BUFFER_OFFSET);
    if (data_virt_base_ == MAP_FAILED) {
        perror("mmap CMA");
        data_virt_base_ = nullptr;
        munmap(regs_virt_base_, kRegMapSize);
        regs_virt_base_ = nullptr;
        ioctl(fd_, NPU_KV260_IOC_FREE_BUFFER);
        close(fd_);
        fd_ = -1;
        return false;
    }

    init_allocator();
    if (require_decode) {
        if (!reinit()) {
            munmap(data_virt_base_, data_map_size_);
            data_virt_base_ = nullptr;
            munmap(regs_virt_base_, kRegMapSize);
            regs_virt_base_ = nullptr;
            ioctl(fd_, NPU_KV260_IOC_FREE_BUFFER);
            close(fd_);
            fd_ = -1;
            return false;
        }

        ioctl(fd_, NPU_KV260_IOC_SET_IRQ_MODE, NPU_KV260_IRQ_MODE_USERSPACE);
        reset();
        ioctl(fd_, NPU_KV260_IOC_SET_IRQ_MODE, NPU_KV260_IRQ_MODE_USERSPACE);
    }
    return true;
}

bool NpuRuntime::reinit() {
    if (fd_ < 0) {
        return false;
    }

    npu_kv260_hw_state state = {};
    if (ioctl(fd_, NPU_KV260_IOC_REINIT, &state) < 0) {
        NPU_ERR("NPU generic reinit failed: errno=%d", errno);
        return false;
    }
    if (state.magic != NPU_KV260_HW_MAGIC ||
        state.abi_version != NPU_KV260_HW_ABI_VERSION ||
        state.mode_id != NPU_KV260_MODE_DECODE) {
        NPU_ERR("unexpected NPU hardware state: magic=0x%08X abi=%u mode=%u caps=0x%08X",
                state.magic, state.abi_version, state.mode_id, state.caps);
        return false;
    }

    shadow_ = ShadowRegs{};
    hardware_ready_ = true;
    NPU_LOG("NPU generic reinit OK: mode=decode caps=0x%08X status=0x%08X error=0x%08X",
            state.caps, state.status, state.error);
    return true;
}

bool NpuRuntime::activate_decode() {
    if (!reinit()) {
        return false;
    }
    ioctl(fd_, NPU_KV260_IOC_SET_IRQ_MODE, NPU_KV260_IRQ_MODE_USERSPACE);
    reset();
    ioctl(fd_, NPU_KV260_IOC_SET_IRQ_MODE, NPU_KV260_IRQ_MODE_USERSPACE);
    return hardware_ready_;
}

void NpuRuntime::reset() {
    if (fd_ >= 0 && ioctl(fd_, NPU_KV260_IOC_RESET_DEV) < 0) {
        NPU_ERR("hardware reset ioctl failed: errno=%d", errno);
    } else {
        reinit();
    }
    shadow_ = ShadowRegs{};
    ack_irq(NPU_REGS__IAR__ACK_bm);
    reg_write(RegOffset::MER, 0x1);
    reg_write(RegOffset::IER, 0x0);
}

void NpuRuntime::init_allocator() {
    uint32_t heap_offset =
        static_cast<uint32_t>(parse_size_with_suffix(std::getenv("NPU_CMA_HEAP_OFFSET"), 0));
    if (heap_offset > data_map_size_) {
        heap_offset = 0;
    }
    heap_offset = static_cast<uint32_t>((heap_offset + kAlignment - 1u) & ~(kAlignment - 1u));

    uint32_t heap_size = data_map_size_ - heap_offset;
    uint64_t requested_heap_size =
        parse_size_with_suffix(std::getenv("NPU_CMA_HEAP_SIZE"), heap_size);
    if (requested_heap_size > 0 && requested_heap_size <= heap_size) {
        heap_size = static_cast<uint32_t>(requested_heap_size);
    }
    heap_size &= ~(kAlignment - 1u);

    if (heap_size <= sizeof(BlockHeader)) {
        free_list_head_ = nullptr;
        return;
    }

    free_list_head_ = reinterpret_cast<BlockHeader*>(
        static_cast<uint8_t*>(data_virt_base_) + heap_offset);
    free_list_head_->size = heap_size - sizeof(BlockHeader);
    free_list_head_->is_free = true;
    free_list_head_->next = nullptr;
    free_list_head_->prev = nullptr;
    alloc_cursor_ = free_list_head_;
}

void* NpuRuntime::alloc(size_t size) {
    if (size == 0 || !free_list_head_) {
        return nullptr;
    }
    const size_t aligned_size = (size + kAlignment - 1u) & ~(kAlignment - 1u);

    BlockHeader* start = alloc_cursor_ ? alloc_cursor_ : free_list_head_;
    BlockHeader* curr = start;
    bool wrapped = false;
    while (curr) {
        if (!curr->is_free || curr->size < aligned_size) {
            curr = curr->next;
            if (!curr && !wrapped) {
                curr = free_list_head_;
                wrapped = true;
            }
            if (wrapped && curr == start) {
                break;
            }
            continue;
        }

        if (curr->size >= aligned_size + sizeof(BlockHeader) + kAlignment) {
            auto* next = reinterpret_cast<BlockHeader*>(
                reinterpret_cast<uint8_t*>(curr) + sizeof(BlockHeader) + aligned_size);
            next->size = curr->size - aligned_size - sizeof(BlockHeader);
            next->is_free = true;
            next->next = curr->next;
            next->prev = curr;
            if (curr->next) {
                curr->next->prev = next;
            }
            curr->next = next;
            curr->size = aligned_size;
        }

        curr->is_free = false;
        alloc_cursor_ = curr->next ? curr->next : free_list_head_;
        return reinterpret_cast<uint8_t*>(curr) + sizeof(BlockHeader);
    }

    return nullptr;
}

void NpuRuntime::free_mem(void* ptr) {
    if (!ptr) {
        return;
    }
    auto* block = reinterpret_cast<BlockHeader*>(
        static_cast<uint8_t*>(ptr) - sizeof(BlockHeader));
    if (block->is_free) {
        return;
    }
    block->is_free = true;
    BlockHeader* cursor = (block->prev && block->prev->is_free) ? block->prev : block;
    coalesce(block);
    alloc_cursor_ = cursor;
}

void NpuRuntime::coalesce(BlockHeader* block) {
    if (block->next && block->next->is_free) {
        block->size += sizeof(BlockHeader) + block->next->size;
        block->next = block->next->next;
        if (block->next) {
            block->next->prev = block;
        }
    }
    if (block->prev && block->prev->is_free) {
        block->prev->size += sizeof(BlockHeader) + block->size;
        block->prev->next = block->next;
        if (block->next) {
            block->next->prev = block->prev;
        }
    }
}

uint32_t NpuRuntime::virt_to_phys(void* ptr) const {
    const uintptr_t virt = reinterpret_cast<uintptr_t>(ptr);
    const uintptr_t base = reinterpret_cast<uintptr_t>(data_virt_base_);
    if (virt < base || virt >= base + data_map_size_) {
        throw std::runtime_error("pointer is outside NPU CMA range");
    }
    return data_phy_base_ + static_cast<uint32_t>(virt - base);
}

void NpuRuntime::reg_write(uint32_t offset, uint32_t val) {
    *reinterpret_cast<volatile uint32_t*>(static_cast<uint8_t*>(regs_virt_base_) + offset) = val;
}

void NpuRuntime::reg_write64(uint32_t offset, uint64_t val) {
#if NPU_CPU_WIDTH == 32
    reg_write(offset, static_cast<uint32_t>(val));
    reg_write(offset + 4, static_cast<uint32_t>(val >> 32));
#else
    *reinterpret_cast<volatile uint64_t*>(static_cast<uint8_t*>(regs_virt_base_) + offset) = val;
#endif
}

uint32_t NpuRuntime::reg_read(uint32_t offset) const {
    return *reinterpret_cast<volatile uint32_t*>(
        static_cast<uint8_t*>(regs_virt_base_) + offset);
}

uint64_t NpuRuntime::reg_read64(uint32_t offset) const {
#if NPU_CPU_WIDTH == 32
    const uint32_t lo = reg_read(offset);
    const uint32_t hi = reg_read(offset + 4);
    return (uint64_t(hi) << 32) | lo;
#else
    return *reinterpret_cast<volatile uint64_t*>(
        static_cast<uint8_t*>(regs_virt_base_) + offset);
#endif
}

void NpuRuntime::read_kv_scale(bool is_v, npu_stream_kv_scale_result* out) const {
    if (!out) {
        throw std::invalid_argument("null KV scale output");
    }
    const uint64_t scale0 = reg_read64(is_v ? RegOffset::KV_V_SCALE0 : RegOffset::KV_K_SCALE0);
    const uint64_t scale1 = reg_read64(is_v ? RegOffset::KV_V_SCALE1 : RegOffset::KV_K_SCALE1);
    out->values[0] = static_cast<uint16_t>(scale0);
    out->values[1] = static_cast<uint16_t>(scale0 >> 16);
    out->values[2] = static_cast<uint16_t>(scale0 >> 32);
    out->values[3] = static_cast<uint16_t>(scale0 >> 48);
    out->values[4] = static_cast<uint16_t>(scale1);
    out->count = static_cast<uint8_t>(scale1 >> 16);
    out->seq = static_cast<uint8_t>(scale1 >> 24);
    out->valid = static_cast<uint8_t>((scale1 >> 32) & 0x1u);
    out->reserved = 0;
}

void NpuRuntime::ack_irq(uint32_t mask) {
    reg_write(RegOffset::IAR, mask & NPU_REGS__IAR__ACK_bm);
}

void NpuRuntime::dump_irq_regs() {
    const uint32_t mer = reg_read(RegOffset::MER);
    const uint32_t ier = reg_read(RegOffset::IER);
    const uint32_t isr = reg_read(RegOffset::ISR);
    const uint32_t ipr = reg_read(RegOffset::IPR);
    NPU_ERR("IRQ/API status MER=0x%08x IER=0x%08x ISR=0x%08x IPR=0x%08x",
            mer, ier, isr, ipr);
    dump_gemv_block_regs("irq_dump");
}

void NpuRuntime::dump_gemv_block_regs(const char* tag) {
    const uint64_t status = reg_read64(RegOffset::GEMV_BLOCK_STATUS);
    NPU_ERR("%s GEMV_BLOCK shadow_tag=%s desc0=0x%016llx desc1=0x%016llx desc2=0x%016llx desc3=0x%016llx desc7=0x%016llx status=0x%016llx",
            tag,
            g_last_gemv_block.tag,
            static_cast<unsigned long long>(g_last_gemv_block.desc0),
            static_cast<unsigned long long>(g_last_gemv_block.desc1),
            static_cast<unsigned long long>(g_last_gemv_block.desc2),
            static_cast<unsigned long long>(g_last_gemv_block.desc3),
            static_cast<unsigned long long>(g_last_gemv_block.desc7),
            static_cast<unsigned long long>(status));
}

void NpuRuntime::prepare_wait_irq_before_start(const std::string& op,
                                               const std::string& shape_key,
                                               uint64_t bytes,
                                               uint32_t expected_mask) {
    expected_mask &= NPU_REGS__IAR__ACK_bm;
    if (expected_mask == 0) {
        expected_mask = NPU_REGS__IAR__ACK_bm;
    }

    g_prepared_wait.strategy = select_wait_strategy(op, shape_key);
    g_prepared_wait.op = op;
    g_prepared_wait.shape_key = shape_key;
    g_prepared_wait.bytes = bytes;
    g_prepared_wait.prepared = true;

    ack_irq(expected_mask);
    if (g_prepared_wait.strategy == NpuWaitStrategy::Irq) {
        ioctl(fd_, NPU_KV260_IOC_SET_IRQ_MODE, NPU_KV260_IRQ_MODE_KERNEL);
    }
}

void NpuRuntime::wait_irq(uint32_t expected_mask) {
    expected_mask &= NPU_REGS__IAR__ACK_bm;
    if (expected_mask == 0) {
        expected_mask = NPU_REGS__IAR__ACK_bm;
    }

    PreparedWaitContext wait_ctx = g_prepared_wait;
    if (!wait_ctx.prepared) {
        wait_ctx.op = "irq";
        wait_ctx.shape_key = "mask=0x" + std::to_string(expected_mask);
        wait_ctx.bytes = 0;
        wait_ctx.strategy = env_wait_policy();
    }
    g_prepared_wait = PreparedWaitContext {};
    const auto wait_start = std::chrono::steady_clock::now();

    volatile uint32_t* isr_ptr = reinterpret_cast<volatile uint32_t*>(
        static_cast<uint8_t*>(regs_virt_base_) + RegOffset::ISR);
    auto consume_irq = [&]() -> bool {
        const uint32_t pending = *isr_ptr & NPU_REGS__IAR__ACK_bm;
        if (pending == 0) {
            return false;
        }
        if (pending & expected_mask) {
            ack_irq(pending);
            return true;
        }
        ack_irq(pending & ~expected_mask);
        return false;
    };

    if (wait_ctx.strategy == NpuWaitStrategy::Irq) {
        if (consume_irq()) {
            ioctl(fd_, NPU_KV260_IOC_SET_IRQ_MODE, NPU_KV260_IRQ_MODE_USERSPACE);
            write_wait_trace(wait_ctx.op, wait_ctx.shape_key, wait_ctx.bytes, elapsed_us(wait_start),
                             wait_ctx.strategy, "irq_prearmed_pending", 0, 0, true, "ok");
            return;
        }

        uint32_t status = 0;
        const int ret = ioctl(fd_, NPU_KV260_IOC_WAIT_IRQ, &status);
        ioctl(fd_, NPU_KV260_IOC_SET_IRQ_MODE, NPU_KV260_IRQ_MODE_USERSPACE);
        if (ret < 0) {
            write_wait_trace(wait_ctx.op, wait_ctx.shape_key, wait_ctx.bytes, elapsed_us(wait_start),
                             wait_ctx.strategy, "irq_prearmed", 0, 0, true, "wait_failed");
            dump_irq_regs();
            throw std::runtime_error("IRQ wait failed");
        }
        if ((status & expected_mask) == 0) {
            NPU_ERR("received unrelated IRQ status=0x%08x expected=0x%08x",
                    status, expected_mask);
        }
        write_wait_trace(wait_ctx.op, wait_ctx.shape_key, wait_ctx.bytes, elapsed_us(wait_start),
                         wait_ctx.strategy, "irq_prearmed", 0, 0, true,
                         (status & expected_mask) ? "ok" : "unrelated");
        return;
    }

    int spin_iters = 0;
    for (int i = 0; i < NPU_POLL_SPIN_COUNT; ++i) {
        spin_iters = i + 1;
        if (consume_irq()) {
            write_wait_trace(wait_ctx.op, wait_ctx.shape_key, wait_ctx.bytes, elapsed_us(wait_start),
                             wait_ctx.strategy, "spin", spin_iters, 0, false, "ok");
            return;
        }
    }

    if (wait_ctx.strategy == NpuWaitStrategy::Spin) {
        const auto spin_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (true) {
            ++spin_iters;
            if (consume_irq()) {
                write_wait_trace(wait_ctx.op, wait_ctx.shape_key, wait_ctx.bytes, elapsed_us(wait_start),
                                 wait_ctx.strategy, "spin", spin_iters, 0, false, "ok");
                return;
            }
            if (std::chrono::steady_clock::now() >= spin_deadline) {
                write_wait_trace(wait_ctx.op, wait_ctx.shape_key, wait_ctx.bytes, elapsed_us(wait_start),
                                 wait_ctx.strategy, "spin", spin_iters, 0, false, "timeout");
                dump_irq_regs();
                throw std::runtime_error("IRQ spin wait timeout");
            }
            if ((spin_iters % NPU_POLL_SPIN_COUNT) == 0) {
                usleep(NPU_POLL_YIELD_US);
            }
        }
    }

    int yield_iters = 0;
    for (int i = 0; i < NPU_POLL_YIELD_COUNT; ++i) {
        yield_iters = i + 1;
        if (consume_irq()) {
            write_wait_trace(wait_ctx.op, wait_ctx.shape_key, wait_ctx.bytes, elapsed_us(wait_start),
                             wait_ctx.strategy, "yield", spin_iters, yield_iters, false, "ok");
            return;
        }
        usleep(NPU_POLL_YIELD_US);
    }

    if (consume_irq()) {
        write_wait_trace(wait_ctx.op, wait_ctx.shape_key, wait_ctx.bytes, elapsed_us(wait_start),
                         wait_ctx.strategy, "yield", spin_iters, yield_iters, false, "ok");
        return;
    }

    ioctl(fd_, NPU_KV260_IOC_SET_IRQ_MODE, NPU_KV260_IRQ_MODE_KERNEL);
    uint32_t status = 0;
    const int ret = ioctl(fd_, NPU_KV260_IOC_WAIT_IRQ, &status);
    ioctl(fd_, NPU_KV260_IOC_SET_IRQ_MODE, NPU_KV260_IRQ_MODE_USERSPACE);

    if (ret < 0) {
        write_wait_trace(wait_ctx.op, wait_ctx.shape_key, wait_ctx.bytes, elapsed_us(wait_start),
                         wait_ctx.strategy, "hybrid_irq", spin_iters, yield_iters, true, "wait_failed");
        dump_irq_regs();
        throw std::runtime_error("IRQ wait failed");
    }
    if ((status & expected_mask) == 0) {
        NPU_ERR("received unrelated IRQ status=0x%08x expected=0x%08x",
                status, expected_mask);
    }
    write_wait_trace(wait_ctx.op, wait_ctx.shape_key, wait_ctx.bytes, elapsed_us(wait_start),
                     wait_ctx.strategy, "hybrid_irq", spin_iters, yield_iters, true,
                     (status & expected_mask) ? "ok" : "unrelated");
}

void NpuRuntime::run_gemv_stream(const GemvPingPongConfig& cfg,
                                 uint32_t weight_row_tile_stride_bytes,
                                 uint16_t weight_capacity_tokens) {
    const bool uses_weight_scale =
        gemv_request_uses_weight_scale(cfg.gemv_mode, cfg.decode_flow);
    const bool uses_rope_lut = decode_flow_uses_rope_lut(cfg.decode_flow);
    const bool fixed_stride_enable = weight_row_tile_stride_bytes != 0;
    const bool cached_group_enable = cfg.cached_group_enable;
    if (!cfg.act_ptr || !cfg.weight_ptr || !cfg.output_ptr ||
        (cfg.enable_act_scale && !cfg.act_scale_ptr) ||
        (cfg.enable_act_scale2 && !cfg.act_scale2_ptr) ||
        (uses_rope_lut && !cfg.rope_lut_ptr) ||
        (uses_weight_scale && !cfg.scale_ptr)) {
        throw std::runtime_error("GEMV stream buffer pointer is null");
    }
    if (cfg.enable_act_scale2 && !cfg.enable_act_scale) {
        throw std::runtime_error("GEMV stream act_scale2 requires act_scale");
    }
    if (cfg.m == 0 || cfg.n == 0) {
        throw std::runtime_error("GEMV stream M/N must be non-zero");
    }
    if (cfg.gemv_mode > NPU_GEMV_MODE_W8A16) {
        throw std::runtime_error("GEMV stream gemv_mode must be 0(W4A16) or 1(W8A16) on this RTL");
    }
    if (cfg.output_precision != 1 && cfg.output_precision != 3) {
        throw std::runtime_error("GEMV stream output_precision must be 1(FP16) or 3(FP32)");
    }
    if (fixed_stride_enable &&
        (cfg.gemv_mode != NPU_GEMV_MODE_W8A16 ||
         weight_capacity_tokens == 0 ||
         cfg.n > weight_capacity_tokens ||
         (weight_row_tile_stride_bytes % NPU_GEMV_MVIN_ALIGN_BYTES) != 0)) {
        throw std::runtime_error("GEMV stream fixed-stride descriptor is invalid");
    }

    const uint32_t group_count = group_count_from_decode_flow(cfg.decode_flow);
    if (cfg.enable_act_scale2 &&
        (cfg.gemv_mode != NPU_GEMV_MODE_W4A16 ||
         fixed_stride_enable ||
         cached_group_enable ||
         group_count != 1)) {
        throw std::runtime_error("GEMV stream act_scale2 is only supported for single-group W4A16 linear streams");
    }
    if (group_count > 1 && !fixed_stride_enable && !cached_group_enable) {
        throw std::runtime_error("GEMV stream grouped mode requires fixed-stride weight payload");
    }
    if (cached_group_enable && group_count <= 1) {
        throw std::runtime_error("GEMV cached-group stream requires grouped decode_flow");
    }
    if (cached_group_enable && cfg.act_group_stride_bytes != 0 &&
        (cfg.act_group_stride_bytes % NPU_GEMV_MVIN_ALIGN_BYTES) != 0) {
        throw std::runtime_error("GEMV cached-group act_group_stride must be 256B aligned");
    }

    const uint32_t col_tiles = ceil_div_u32(cfg.n, gemv_tile_elems(cfg.gemv_mode));
    const uint32_t total_row_tiles = ceil_div_u32(cfg.m, NPU_GEMV_ROW_TILE_ELEMS);
    const uint32_t weight_tile_bytes = gemv_weight_tile_bytes(cfg.gemv_mode);
    const uint32_t act_tile_bytes = gemv_tile_elems(cfg.gemv_mode) * NPU_GEMV_FP16_BYTES;
    const uint32_t act_bytes = col_tiles * act_tile_bytes;
    const uint32_t act_group_span_bytes =
        grouped_act_span_bytes(act_bytes, group_count, cfg.act_group_stride_bytes);
    const uint32_t act_scale_bytes =
        act_scale_bytes_for(cfg.n, cfg.gemv_mode, cfg.decode_flow,
                            cfg.enable_act_scale);
    const uint32_t act_mvin_bytes = cached_group_enable
        ? align_up_u32(act_bytes, NPU_GEMV_MVIN_ALIGN_BYTES)
        : align_up_u32(act_group_span_bytes, NPU_GEMV_MVIN_ALIGN_BYTES);
    const uint32_t act_scale_mvin_bytes =
        cfg.enable_act_scale ?
            align_up_u32(act_scale_bytes, NPU_GEMV_MVIN_ALIGN_BYTES) : 0u;
    const uint32_t act_scale2_mvin_bytes =
        cfg.enable_act_scale2 ?
            align_up_u32(act_scale_bytes, NPU_GEMV_MVIN_ALIGN_BYTES) : 0u;
    const uint32_t weight_bytes = total_row_tiles * col_tiles * weight_tile_bytes;
    const uint32_t stream_weight_bytes = cached_group_enable ?
        weight_bytes : (fixed_stride_enable ? weight_bytes * group_count : weight_bytes);
    const uint32_t scale_bytes = total_row_tiles * col_tiles * kGemvScaleTileBytes;
    const uint32_t scale_mvin_bytes =
        uses_weight_scale ? align_up_u32(scale_bytes, NPU_GEMV_MVIN_ALIGN_BYTES) : 0u;
    const uint32_t output_elems = static_cast<uint32_t>(cfg.m) * group_count;
    const bool kv_quant_scratch = decode_flow_kv_quant_scratch(cfg.decode_flow);
    const uint32_t output_bytes =
        output_elems * (kv_quant_scratch ? 1u : ((cfg.output_precision == 3) ? 4u : 2u));

    if (act_mvin_bytes > NPU_GEMV_ACT_PAYLOAD_BYTES) {
        throw std::runtime_error("GEMV stream activation vector overlaps reserved RoPE LUT window");
    }
    if (cfg.enable_act_scale &&
        (act_scale_bytes == 0 ||
         ((cached_group_enable ? (2u * act_mvin_bytes) : act_mvin_bytes) +
          act_scale_mvin_bytes + act_scale2_mvin_bytes >
          NPU_GEMV_ACT_PAYLOAD_BYTES))) {
        throw std::runtime_error("GEMV stream activation scale exceeds act bank payload window");
    }
    if (weight_bytes == 0 || weight_bytes > 4194240u ||
        (weight_bytes % NPU_GEMV_MVIN_ALIGN_BYTES) != 0) {
        throw std::runtime_error("GEMV stream weight payload does not fit one stream MVIN descriptor");
    }
    if (uses_weight_scale &&
        kGemvStreamScaleSpmBase + scale_mvin_bytes > 256u * 1024u) {
        throw std::runtime_error("GEMV stream weight scale exceeds dedicated scale bank");
    }
    if (cached_group_enable &&
        kGemvPingWeightSpmBase + weight_bytes > kGemvPongWeightSpmBase) {
        throw std::runtime_error("GEMV cached-group weight does not fit cached SPM window");
    }
    const uint32_t output_spm_elem_bytes =
        kv_quant_scratch ? 1u : static_cast<uint32_t>(NPU_GEMV_FP16_BYTES);
    if (output_elems * output_spm_elem_bytes > NPU_GEMV_SPM_BYTES) {
        throw std::runtime_error("GEMV stream output exceeds output SPM capacity");
    }

    auto checked_phys = [&](void* ptr, const char* name) {
        const uint32_t phys = virt_to_phys(ptr);
        if ((phys % NPU_GEMV_MVIN_ALIGN_BYTES) != 0) {
            throw std::runtime_error(std::string("GEMV stream ") + name +
                                     " physical address must be 256B aligned");
        }
        return phys;
    };

    const uint32_t act_phys = checked_phys(cfg.act_ptr, "act_ptr");
    const uint32_t act_scale_phys =
        cfg.enable_act_scale ? checked_phys(cfg.act_scale_ptr, "act_scale_ptr") : 0u;
    const uint32_t act_scale2_phys =
        cfg.enable_act_scale2 ? checked_phys(cfg.act_scale2_ptr, "act_scale2_ptr") : 0u;
    const uint32_t weight_phys = checked_phys(cfg.weight_ptr, "weight_ptr");
    const uint32_t output_phys = checked_phys(cfg.output_ptr, "output_ptr");
    const uint32_t scale_phys =
        uses_weight_scale ? checked_phys(cfg.scale_ptr, "scale_ptr") : 0u;
    const uint32_t rope_lut_phys =
        uses_rope_lut ? checked_phys(cfg.rope_lut_ptr, "rope_lut_ptr") : 0u;

    const uint8_t legacy_output_precision = (cfg.output_precision == 3) ? 1u : 0u;
    const uint64_t desc0 = uint64_t(act_phys) | (uint64_t(weight_phys) << 32);
    const uint64_t desc1 = uint64_t(act_scale_phys) | (uint64_t(output_phys) << 32);
    const uint64_t desc2 =
        uint64_t(cfg.n) |
        (uint64_t(cfg.m) << 16) |
        (uint64_t(cfg.cache_cell_idx) << 32) |
        (uint64_t(cfg.gemv_mode) << 48) |
        (uint64_t(legacy_output_precision) << 56) |
        (uint64_t(cfg.enable_act_scale ? 1u : 0u) << 57) |
        (uint64_t(fixed_stride_enable ? 1u : 0u) << 58) |
        (uint64_t(cached_group_enable ? 1u : 0u) << 59) |
        (uint64_t(cfg.enable_act_scale2 ? 1u : 0u) << 60);
    const uint64_t desc3 = cfg.decode_flow;
    const uint64_t desc4 = fixed_stride_enable
        ? (uint64_t(weight_row_tile_stride_bytes) |
           (uint64_t(weight_capacity_tokens) << 32))
        : 0;
    const uint64_t desc5 = uint64_t(cfg.act_group_stride_bytes);
    const uint64_t desc6 = uint64_t(scale_phys) | (uint64_t(rope_lut_phys) << 32);
    const uint64_t desc7 = uint64_t(act_scale2_phys);

    const std::string shape_key =
        "m=" + std::to_string(cfg.m) +
        ",n=" + std::to_string(cfg.n) +
        ",mode=" + std::to_string(cfg.gemv_mode) +
        ",groups=" + std::to_string(group_count) +
        ",stream=1,act_scale=" + std::to_string(cfg.enable_act_scale ? 1 : 0) +
        ",act_scale2=" + std::to_string(cfg.enable_act_scale2 ? 1 : 0) +
        ",flow=" + std::to_string(cfg.decode_flow ? 1 : 0);
    NPU_LOG("gemv_stream %s cached=%u act_dma=%u act_scale_dma=%u act_scale2_dma=%u rope_lut_dma=%u scale_dma=%u weight=%u output=%u",
            shape_key.c_str(), cached_group_enable ? 1u : 0u,
            act_mvin_bytes, act_scale_mvin_bytes, act_scale2_mvin_bytes,
            uses_rope_lut ? kRopeLutWindowBytes : 0u,
            scale_mvin_bytes, stream_weight_bytes, output_bytes);

    g_last_gemv_block.desc0 = desc0;
    g_last_gemv_block.desc1 = desc1;
    g_last_gemv_block.desc2 = desc2;
    g_last_gemv_block.desc3 = desc3;
    g_last_gemv_block.desc4 = desc4;
    g_last_gemv_block.desc5 = desc5;
    g_last_gemv_block.desc6 = desc6;
    g_last_gemv_block.desc7 = desc7;
    g_last_gemv_block.tag = "gemv_stream";

    reg_write(RegOffset::GEMV_BLOCK_CTRL, 0u);
    reg_write(RegOffset::GEMV_BLOCK_STATUS, kApiStatusDone | kApiStatusError);
    ack_irq(NPU_IRQ_GEMV_BLOCK);

    reg_write64(RegOffset::GEMV_BLOCK_DESC0, desc0);
    reg_write64(RegOffset::GEMV_BLOCK_DESC1, desc1);
    reg_write64(RegOffset::GEMV_BLOCK_DESC2, desc2);
    reg_write64(RegOffset::GEMV_BLOCK_DESC3, desc3);
    reg_write64(RegOffset::GEMV_BLOCK_DESC4, desc4);
    reg_write64(RegOffset::GEMV_BLOCK_DESC5, desc5);
    reg_write64(RegOffset::GEMV_BLOCK_DESC6, desc6);
    reg_write64(RegOffset::GEMV_BLOCK_DESC7, desc7);
    prepare_wait_irq_before_start("gemv_stream", shape_key,
                                  uint64_t(act_mvin_bytes) +
                                      act_scale_mvin_bytes + act_scale2_mvin_bytes +
                                      (uses_rope_lut ? kRopeLutWindowBytes : 0u) +
                                      scale_mvin_bytes + stream_weight_bytes + output_bytes,
                                  NPU_IRQ_GEMV_BLOCK);
    reg_write(RegOffset::GEMV_BLOCK_CTRL, 1u);
    wait_irq(NPU_IRQ_GEMV_BLOCK);
    const uint64_t status = reg_read64(RegOffset::GEMV_BLOCK_STATUS);
    if (status & kApiStatusError) {
        throw std::runtime_error("GEMV_BLOCK stream failed, code=" +
                                 std::to_string((status >> kApiStatusErrorCodeShift) & 0xfu));
    }
    reg_write(RegOffset::GEMV_BLOCK_CTRL, 0u);
    reg_write(RegOffset::GEMV_BLOCK_STATUS, kApiStatusDone | kApiStatusError);
}

} // namespace

extern "C" {

int npu_init(void) {
    if (g_runtime) {
        if (!g_runtime->hardware_ready()) {
            if (!g_runtime->activate_decode()) {
                return -1;
            }
        }
        return 0;
    }
    auto* rt = new NpuRuntime();
    if (!rt->init(true)) {
        delete rt;
        return -1;
    }
    g_runtime = rt;
    return 0;
}

int npu_decode_cma_init(void) {
    if (g_runtime) {
        return 0;
    }
    auto* rt = new NpuRuntime();
    if (!rt->init(false)) {
        delete rt;
        return -1;
    }
    g_runtime = rt;
    return 0;
}

void npu_destroy(void) {
    delete g_runtime;
    g_runtime = nullptr;
}

void npu_reset(void) {
    try {
        runtime().reset();
    } catch (const std::exception& ex) {
        report_exception("npu_reset", ex);
    }
}

int npu_reinit(void) {
    try {
        return runtime().reinit() ? 0 : -1;
    } catch (const std::exception& ex) {
        report_exception("npu_reinit", ex);
        return -1;
    }
}

int npu_open(npu_device** out_dev, const char* dev_path) {
    if (!out_dev) {
        return -EINVAL;
    }
    *out_dev = nullptr;
    if (!is_default_dev_path(dev_path)) {
        return -ENOTSUP;
    }

    const bool need_init = (g_runtime == nullptr);
    if (need_init && npu_init() != 0) {
        return -EIO;
    }

    auto* dev = new (std::nothrow) npu_device{};
    if (!dev) {
        if (need_init) {
            npu_destroy();
        }
        return -ENOMEM;
    }

    dev->magic = kNpuDeviceMagic;
    ++g_device_refcount;
    if (need_init) {
        g_runtime_owned_by_device_api = true;
    }
    *out_dev = dev;
    return 0;
}

void npu_close(npu_device* dev) {
    if (!valid_device(dev)) {
        return;
    }
    dev->magic = 0;
    delete dev;

    if (g_device_refcount > 0) {
        --g_device_refcount;
    }
    if (g_device_refcount == 0 && g_runtime_owned_by_device_api) {
        g_runtime_owned_by_device_api = false;
        npu_destroy();
    }
}

void* npu_mem_alloc(size_t size) {
    try {
        return runtime().alloc(size);
    } catch (const std::exception& ex) {
        report_exception("npu_mem_alloc", ex);
        return nullptr;
    }
}

void npu_mem_free(void* ptr) {
    try {
        runtime().free_mem(ptr);
    } catch (const std::exception& ex) {
        report_exception("npu_mem_free", ex);
    }
}

void* npu_decode_memory_base(void) {
    try {
        return runtime().memory_base();
    } catch (const std::exception& ex) {
        report_exception("npu_decode_memory_base", ex);
        return nullptr;
    }
}

uint32_t npu_decode_memory_size(void) {
    try {
        return runtime().memory_size();
    } catch (const std::exception& ex) {
        report_exception("npu_decode_memory_size", ex);
        return 0;
    }
}

int npu_mem_dma_offset(const void* ptr, uint32_t* out_offset) {
    try {
        return dma_offset_for_ptr(ptr, out_offset);
    } catch (const std::exception& ex) {
        report_exception("npu_mem_dma_offset", ex);
        return -EIO;
    }
}

int npu_stream_gemv_run(npu_device* dev, const npu_stream_gemv_desc* desc,
                        uint32_t timeout_ms) {
    const int timeout_rc = validate_timeout_default_only(timeout_ms);
    if (timeout_rc != 0) {
        return timeout_rc;
    }
    const int desc_rc = validate_stream_gemv_desc(dev, desc);
    if (desc_rc != 0) {
        return desc_rc;
    }
    try {
        GemvPingPongConfig cfg = {};
        cfg.act_ptr = const_cast<void*>(desc->act_ptr);
        cfg.act_scale_ptr = const_cast<void*>(desc->act_scale_ptr);
        cfg.act_scale2_ptr = const_cast<void*>(desc->act_scale2_ptr);
        cfg.scale_ptr = const_cast<void*>(desc->weight_scale_ptr);
        cfg.weight_ptr = const_cast<void*>(desc->weight_payload_ptr);
        cfg.output_ptr = desc->output_ptr;
        cfg.rope_lut_ptr = const_cast<void*>(desc->rope_lut_ptr);
        cfg.m = desc->m;
        cfg.n = desc->n;
        cfg.gemv_mode = static_cast<uint8_t>(desc->mode);
        cfg.output_precision = output_precision_to_hw(desc->output_precision);
        cfg.enable_act_scale =
            (desc->flags & NPU_STREAM_GEMV_F_ENABLE_ACT_SCALE) != 0;
        cfg.enable_act_scale2 =
            (desc->flags & NPU_STREAM_GEMV_F_ENABLE_ACT_SCALE2) != 0;
        cfg.decode_flow = stream_decode_flow_from_desc(*desc);
        cfg.cache_cell_idx = kGemvStreamCacheInvalidateTag;
        cfg.act_group_stride_bytes = desc->act_group_stride_bytes;
        const uint16_t group_count = stream_group_count_or_default(*desc);
        cfg.cached_group_enable =
            group_count > 1 &&
            (desc->role == NPU_STREAM_GEMV_ROLE_QK ||
             desc->role == NPU_STREAM_GEMV_ROLE_PV);
        runtime().run_gemv_stream(cfg,
                                  desc->weight_row_tile_stride_bytes,
                                  desc->weight_capacity_tokens);
        return 0;
    } catch (const std::exception& ex) {
        report_exception("npu_stream_gemv_run", ex);
        return -EIO;
    }
}

int npu_stream_kv_scale_read(npu_device* dev, int is_v,
                             npu_stream_kv_scale_result* out) {
    if (!valid_device(dev) || !out) {
        return -EINVAL;
    }
    try {
        runtime().read_kv_scale(is_v != 0, out);
        return 0;
    } catch (const std::exception& ex) {
        report_exception("npu_stream_kv_scale_read", ex);
        return -EIO;
    }
}

} // extern "C"
