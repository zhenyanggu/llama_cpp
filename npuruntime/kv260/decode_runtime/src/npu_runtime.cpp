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
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace {

constexpr uint32_t kDmaChannelCount = 1;
constexpr uint32_t kDefaultCmaMapSize = 256u * 1024u * 1024u;
constexpr uint32_t kRegMapSize = NPU_KV260_REG_MMAP_SIZE;
constexpr off_t kRegMapOffset = NPU_KV260_MMAP_REGS_OFFSET;
constexpr size_t kAlignment = 64;

constexpr uint32_t kGemvActSpmBase = 0x00000;
constexpr uint32_t kGemvScaleSpmBase = 0x02000;
constexpr uint32_t kGemvPingWeightSpmBase = 0x10000;
constexpr uint32_t kGemvPongWeightSpmBase = 0x40000;
constexpr uint32_t kGemvOutputSpmBase = 0x00000;
constexpr uint32_t kGemvActTileBytes = NPU_GEMV_TILE_ELEMS * NPU_GEMV_FP16_BYTES;
constexpr uint32_t kGemvScaleTileBytes = NPU_GEMV_LINE_BYTES;
constexpr uint32_t kGemvWeightW4RowBytes = NPU_GEMV_TILE_ELEMS / 2;
constexpr uint32_t kGemvWeightW8RowBytes = NPU_GEMV_TILE_ELEMS;
constexpr uint32_t kGemvWeightW16RowBytes = NPU_GEMV_TILE_ELEMS * NPU_GEMV_FP16_BYTES;
constexpr uint32_t kRopeLutActSpmBase = NPU_ROPE_LUT_ACT_SPM_BASE;
constexpr uint32_t kRopeLutWindowBytes = NPU_ROPE_LUT_WINDOW_BYTES;

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

uint32_t ceil_div_u32(uint32_t a, uint32_t b) {
    return (a + b - 1u) / b;
}

uint32_t gemv_weight_tile_bytes(uint8_t gemv_mode) {
    uint32_t row_bytes = kGemvWeightW4RowBytes;
    if (gemv_mode == NPU_GEMV_MODE_W8A16) {
        row_bytes = kGemvWeightW8RowBytes;
    } else if (gemv_mode == NPU_GEMV_MODE_W16A16) {
        row_bytes = kGemvWeightW16RowBytes;
    }
    const uint32_t data_bytes = NPU_GEMV_ROW_TILE_ELEMS * row_bytes;
    return (gemv_mode == NPU_GEMV_MODE_W4A16 || gemv_mode == NPU_GEMV_MODE_W8A16)
        ? kGemvScaleTileBytes + data_bytes
        : data_bytes;
}

bool gemv_uses_embedded_scale(uint8_t gemv_mode) {
    return gemv_mode == NPU_GEMV_MODE_W4A16 || gemv_mode == NPU_GEMV_MODE_W8A16;
}

uint32_t default_act_scale_addr(uint32_t vec_addr, uint16_t mat_width, uint8_t gemv_mode) {
    if (gemv_mode != NPU_GEMV_MODE_W4A16) {
        return 0;
    }
    return vec_addr + ceil_div_u32(mat_width, NPU_GEMV_TILE_ELEMS) * kGemvActTileBytes;
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

uint64_t make_matvec_bypass_output_flow(uint16_t elem_count) {
    return make_decode_flow_value(
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

uint64_t make_matvec_silu_buffer_flow(uint16_t elem_count) {
    return make_decode_flow_value(
        NPU_DECODE_SRC_GEMV_STREAM,
        0,
        NPU_DECODE_UNARY_SILU,
        NPU_DECODE_BINARY_BYPASS,
        NPU_DECODE_REDUCE_BYPASS,
        NPU_DECODE_DST_POST_BUFFER,
        0,
        0,
        elem_count,
        0);
}

MvinConfig make_rope_lut_mvin_config(void* rope_table_base, uint16_t position) {
    if (!rope_table_base) {
        throw std::runtime_error("RoPE LUT table pointer is null");
    }

    const uint32_t even_position = uint32_t(position) & ~1u;
    const uint32_t window_index = even_position >> 1;

    MvinConfig cfg = {};
    cfg.host_ptr = static_cast<uint8_t*>(rope_table_base) +
                   window_index * kRopeLutWindowBytes;
    cfg.sram_addr = kRopeLutActSpmBase;
    cfg.col_num = kRopeLutWindowBytes - 1u;
    cfg.row_num = 0;
    cfg.precision = 1;
    cfg.input_type = NPU_GEMV_MVIN_ACT;
    return cfg;
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

struct DmaWaitContext {
    std::string shape_key;
    std::chrono::steady_clock::time_point start;
    uint64_t bytes = 0;
    bool valid = false;
};

thread_local PreparedWaitContext g_prepared_wait;

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

std::string mvin_shape_key(uint32_t dma_id, const MvinConfig& cfg, uint64_t bytes) {
    std::ostringstream oss;
    oss << "dma=" << dma_id
        << ",row=" << cfg.row_num
        << ",col=" << cfg.col_num
        << ",precision=" << static_cast<unsigned>(cfg.precision)
        << ",input_type=" << static_cast<unsigned>(cfg.input_type)
        << ",bytes=" << bytes;
    return oss.str();
}

std::string mvout_shape_key(uint32_t dma_id, const MvoutConfig& cfg, uint64_t bytes) {
    std::ostringstream oss;
    oss << "dma=" << dma_id
        << ",row=" << cfg.row_num
        << ",col=" << cfg.col_num
        << ",precision=" << static_cast<unsigned>(cfg.precision)
        << ",output_type=" << static_cast<unsigned>(cfg.output_type)
        << ",bytes=" << bytes;
    return oss.str();
}

std::string matvec_shape_key(const MatvecConfig& cfg) {
    std::ostringstream oss;
    oss << "mode=" << static_cast<unsigned>(cfg.gemv_mode)
        << ",m=" << cfg.mat_height
        << ",n=" << cfg.mat_width
        << ",flow=" << (cfg.decode_flow ? 1 : 0);
    return oss.str();
}

uint64_t mvout_transfer_bytes(const MvoutConfig& cfg) {
    const uint64_t elem_bytes = cfg.precision == 3 ? 4ull : 2ull;
    return (static_cast<uint64_t>(cfg.row_num) + 1ull) *
           (static_cast<uint64_t>(cfg.col_num) + 1ull) * elem_bytes;
}

class NpuRuntime {
public:
    NpuRuntime() = default;
    ~NpuRuntime();

    bool init();
    bool reinit();
    void reset();

    void* memory_base() const { return data_virt_base_; }
    uint32_t memory_size() const { return data_map_size_; }

    void* alloc(size_t size);
    void free_mem(void* ptr);

    void run_mvin(const MvinConfig& cfg);
    void run_mvout(const MvoutConfig& cfg);
    void run_mvin_async(uint32_t dma_id, const MvinConfig& cfg);
    void run_mvout_async(uint32_t dma_id, const MvoutConfig& cfg);
    void wait_mvin(uint32_t dma_mask);
    void wait_mvout(uint32_t dma_mask);
    void run_matvec(const MatvecConfig& cfg);
    void run_gemv_pingpong(const GemvPingPongConfig& cfg);

private:
    struct alignas(64) BlockHeader {
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
    void* pending_mvin_staging_[kDmaChannelCount] = {};
    DmaWaitContext pending_mvin_ctx_[kDmaChannelCount] = {};
    DmaWaitContext pending_mvout_ctx_[kDmaChannelCount] = {};
    BlockHeader* free_list_head_ = nullptr;
    BlockHeader* alloc_cursor_ = nullptr;
    ShadowRegs shadow_;

    void init_allocator();
    void coalesce(BlockHeader* block);
    uint32_t virt_to_phys(void* ptr) const;

    void validate_dma_id(uint32_t dma_id) const;
    void validate_dma_mask(uint32_t dma_mask) const;
    void validate_mvin(uint32_t dma_id, const MvinConfig& cfg) const;
    void validate_mvout(uint32_t dma_id, const MvoutConfig& cfg) const;
    void validate_matvec(const MatvecConfig& cfg) const;

    uint32_t read_dma_busy_mask(bool is_mvin);
    void wait_dma_idle(bool is_mvin, uint32_t dma_mask);
    void release_mvin_staging(uint32_t dma_mask);
    size_t mvin_transfer_bytes(const MvinConfig& cfg) const;

    void reg_write(uint32_t offset, uint32_t val);
    void reg_write64(uint32_t offset, uint64_t val);
    uint32_t reg_read(uint32_t offset) const;
    void reg_write64_cached(uint32_t offset, uint64_t val, uint64_t* cache);
    void ack_irq(uint32_t mask);
    void dump_irq_regs();
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

void report_exception(const char* api, const std::exception& ex) {
    NPU_ERR("%s failed: %s", api, ex.what());
}

NpuRuntime::~NpuRuntime() {
    release_mvin_staging((1u << kDmaChannelCount) - 1u);
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

bool NpuRuntime::init() {
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

    init_allocator();
    ioctl(fd_, NPU_KV260_IOC_SET_IRQ_MODE, NPU_KV260_IRQ_MODE_USERSPACE);
    reset();
    ioctl(fd_, NPU_KV260_IOC_SET_IRQ_MODE, NPU_KV260_IRQ_MODE_USERSPACE);
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

    release_mvin_staging((1u << kDmaChannelCount) - 1u);
    shadow_ = ShadowRegs{};
    NPU_LOG("NPU generic reinit OK: mode=decode caps=0x%08X status=0x%08X error=0x%08X",
            state.caps, state.status, state.error);
    return true;
}

void NpuRuntime::reset() {
    release_mvin_staging((1u << kDmaChannelCount) - 1u);
    if (fd_ >= 0 && ioctl(fd_, NPU_KV260_IOC_RESET_DEV) < 0) {
        NPU_ERR("hardware reset ioctl failed: errno=%d", errno);
    } else {
        reinit();
    }
    shadow_ = ShadowRegs{};
    ack_irq(NPU_REGS__IAR__ACK_bm);
    reg_write(RegOffset::MER, 0x3);
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

void NpuRuntime::validate_dma_id(uint32_t dma_id) const {
    if (dma_id >= kDmaChannelCount) {
        throw std::runtime_error("only DMA0 is present in the current GEMV top");
    }
}

void NpuRuntime::validate_dma_mask(uint32_t dma_mask) const {
    const uint32_t valid = (1u << kDmaChannelCount) - 1u;
    if ((dma_mask & ~valid) != 0) {
        throw std::runtime_error("DMA mask references a non-existent channel");
    }
}

void NpuRuntime::validate_mvin(uint32_t dma_id, const MvinConfig& cfg) const {
    validate_dma_id(dma_id);
    if (!cfg.host_ptr) {
        throw std::runtime_error("MVIN host_ptr is null");
    }
    if (cfg.row_num != 0) {
        throw std::runtime_error("GEMV MVIN supports one 4AXI row only");
    }
    const uint64_t bytes = static_cast<uint64_t>(cfg.col_num) + 1ull;
    if (bytes == 0 || (bytes % NPU_GEMV_MVIN_ALIGN_BYTES) != 0) {
        throw std::runtime_error("GEMV MVIN byte count must be a 256B multiple");
    }
    if ((cfg.sram_addr % NPU_GEMV_LINE_BYTES) != 0) {
        throw std::runtime_error("GEMV MVIN sram_addr must be 64B aligned");
    }
    if (cfg.dest || cfg.is_bias || cfg.is_quant || cfg.quant_zero ||
        cfg.quant_scale || cfg.quant_shift) {
        throw std::runtime_error("GEMV runtime does not support ACC/bias/quant MVIN");
    }
    if (cfg.input_type != NPU_GEMV_MVIN_INPUT_SPM &&
        cfg.input_type != NPU_GEMV_MVIN_WEIGHT &&
        cfg.input_type != NPU_GEMV_MVIN_ACT) {
        throw std::runtime_error("GEMV MVIN input_type must be input SPM, weight, or act");
    }
    if (cfg.precision > 3) {
        throw std::runtime_error("GEMV MVIN precision must fit the 2-bit hardware field");
    }
}

void NpuRuntime::validate_mvout(uint32_t dma_id, const MvoutConfig& cfg) const {
    validate_dma_id(dma_id);
    if (!cfg.host_ptr) {
        throw std::runtime_error("MVOUT host_ptr is null");
    }
    if ((cfg.sram_addr % NPU_GEMV_LINE_BYTES) != 0) {
        throw std::runtime_error("GEMV MVOUT sram_addr must be 64B aligned");
    }
    if (cfg.source || cfg.is_quant || cfg.per_channel || cfg.quant_zero || cfg.scale_or_addr) {
        throw std::runtime_error("GEMV runtime only supports raw output-SPM MVOUT");
    }
    if (cfg.precision != 1 && cfg.precision != 3) {
        throw std::runtime_error("GEMV MVOUT precision must be 1(FP16) or 3(FP32)");
    }
}

void NpuRuntime::validate_matvec(const MatvecConfig& cfg) const {
    if (cfg.mat_width == 0 || cfg.mat_height == 0) {
        throw std::runtime_error("MATVEC width/height must be non-zero");
    }
    if (cfg.gemv_mode > NPU_GEMV_MODE_W16A16) {
        throw std::runtime_error("MATVEC gemv_mode must be 0(W4A16), 1(W8A16), or 2(W16A16)");
    }
    if ((cfg.mat_addr % NPU_GEMV_LINE_BYTES) != 0 ||
        (cfg.vec_addr % NPU_GEMV_LINE_BYTES) != 0 ||
        (cfg.output_addr % NPU_GEMV_FP16_BYTES) != 0 ||
        (cfg.scale_addr % NPU_GEMV_LINE_BYTES) != 0 ||
        (cfg.gemv_mode == NPU_GEMV_MODE_W4A16 &&
         (cfg.act_scale_addr % NPU_GEMV_LINE_BYTES) != 0)) {
        throw std::runtime_error("MATVEC addresses do not meet GEMV alignment constraints");
    }
}

uint32_t NpuRuntime::read_dma_busy_mask(bool is_mvin) {
    const uint32_t raw = reg_read(RegOffset::DMA_STATUS);
    return is_mvin
        ? static_cast<uint32_t>(REG_GET_FIELD(DMA_STATUS, MVIN_BUSY, raw))
        : static_cast<uint32_t>(REG_GET_FIELD(DMA_STATUS, MVOUT_BUSY, raw));
}

void NpuRuntime::wait_dma_idle(bool is_mvin, uint32_t dma_mask) {
    if (dma_mask == 0) {
        return;
    }
    validate_dma_mask(dma_mask);

    const auto start = std::chrono::steady_clock::now();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    int iter = 0;
    while (true) {
        const uint32_t busy = read_dma_busy_mask(is_mvin) & dma_mask;
        if (busy == 0) {
            std::string shape_key = "dma_mask=" + std::to_string(dma_mask);
            uint64_t bytes = 0;
            for (uint32_t dma_id = 0; dma_id < kDmaChannelCount; ++dma_id) {
                if ((dma_mask & (1u << dma_id)) == 0) {
                    continue;
                }
                DmaWaitContext& ctx = is_mvin ? pending_mvin_ctx_[dma_id] : pending_mvout_ctx_[dma_id];
                if (ctx.valid) {
                    shape_key = ctx.shape_key;
                    bytes = ctx.bytes;
                    const uint64_t full_us = elapsed_us(ctx.start);
                    ctx.valid = false;
                    write_wait_trace(is_mvin ? "mvin" : "mvout", shape_key, bytes, elapsed_us(start),
                                     NpuWaitStrategy::Spin, "dma_idle", iter, 0, false, "ok", full_us);
                    return;
                }
            }
            write_wait_trace(is_mvin ? "mvin" : "mvout", shape_key, bytes, elapsed_us(start),
                             NpuWaitStrategy::Spin, "dma_idle", iter, 0, false, "ok");
            return;
        }
        if (++iter > NPU_POLL_SPIN_COUNT) {
            if (std::chrono::steady_clock::now() >= deadline) {
                write_wait_trace(is_mvin ? "mvin" : "mvout", "dma_mask=" + std::to_string(dma_mask),
                                 0, elapsed_us(start), NpuWaitStrategy::Spin, "dma_idle",
                                 iter, 0, false, "timeout");
                dump_irq_regs();
                throw std::runtime_error(is_mvin ? "MVIN timeout" : "MVOUT timeout");
            }
            usleep(NPU_POLL_YIELD_US);
        }
    }
}

void NpuRuntime::release_mvin_staging(uint32_t dma_mask) {
    if (dma_mask == 0) {
        return;
    }
    validate_dma_mask(dma_mask);
    for (uint32_t dma_id = 0; dma_id < kDmaChannelCount; ++dma_id) {
        if ((dma_mask & (1u << dma_id)) && pending_mvin_staging_[dma_id]) {
            free_mem(pending_mvin_staging_[dma_id]);
            pending_mvin_staging_[dma_id] = nullptr;
        }
    }
}

size_t NpuRuntime::mvin_transfer_bytes(const MvinConfig& cfg) const {
    return static_cast<size_t>(static_cast<uint64_t>(cfg.col_num) + 1ull);
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

void NpuRuntime::reg_write64_cached(uint32_t offset, uint64_t val, uint64_t* cache) {
    if (*cache != val) {
        reg_write64(offset, val);
        *cache = val;
    }
}

void NpuRuntime::ack_irq(uint32_t mask) {
    reg_write(RegOffset::IAR, mask & NPU_REGS__IAR__ACK_bm);
}

void NpuRuntime::dump_irq_regs() {
    const uint32_t mer = reg_read(RegOffset::MER);
    const uint32_t ier = reg_read(RegOffset::IER);
    const uint32_t isr = reg_read(RegOffset::ISR);
    const uint32_t ipr = reg_read(RegOffset::IPR);
    const uint32_t dma_status = reg_read(RegOffset::DMA_STATUS);
    NPU_ERR("IRQ/DMA status MER=0x%08x IER=0x%08x ISR=0x%08x IPR=0x%08x DMA_STATUS=0x%08x",
            mer, ier, isr, ipr, dma_status);
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

void NpuRuntime::run_mvin(const MvinConfig& cfg) {
    run_mvin_async(0, cfg);
    wait_mvin(1u);
}

void NpuRuntime::run_mvin_async(uint32_t dma_id, const MvinConfig& cfg) {
    validate_mvin(dma_id, cfg);
    if (read_dma_busy_mask(true) & (1u << dma_id)) {
        wait_mvin(1u << dma_id);
    }
    if (pending_mvin_staging_[dma_id]) {
        throw std::runtime_error("MVIN staging buffer is still pending");
    }

    void* dma_src = cfg.host_ptr;
    uint32_t phys = 0;
    try {
        phys = virt_to_phys(dma_src);
    } catch (const std::runtime_error&) {
        const size_t bytes = mvin_transfer_bytes(cfg);
        void* staging = alloc(bytes);
        if (!staging) {
            throw std::runtime_error("MVIN staging allocation failed");
        }
        std::memcpy(staging, cfg.host_ptr, bytes);
        pending_mvin_staging_[dma_id] = staging;
        dma_src = staging;
        phys = virt_to_phys(dma_src);
    }

    const uint64_t val_dram =
        REG_FIELD(MVIN_CTRL0, DRAM_ADDR, phys) |
        REG_FIELD(MVIN_CTRL0, ROW_NUM, cfg.row_num);
    const uint64_t val_sram =
        REG_FIELD(MVIN_CTRL1, SRAM_ADDR, cfg.sram_addr) |
        REG_FIELD(MVIN_CTRL1, COL_NUM, cfg.col_num);
    const uint64_t val_cfg =
        REG_FIELD(CFG_MVIN0, INPUT_TYPE, cfg.input_type) |
        REG_FIELD(CFG_MVIN0, INPUT_PRECISION, cfg.precision) |
        REG_FIELD(CFG_MVIN0, IS_QUANT, 0) |
        REG_FIELD(CFG_MVIN0, DEST, 0) |
        REG_FIELD(CFG_MVIN0, IS_BIAS, 0) |
        REG_FIELD(CFG_MVIN0, SRAM_STRIDE, cfg.sram_stride) |
        REG_FIELD(CFG_MVIN0, DRAM_STRIDE, cfg.dram_stride);

    reg_write64(RegOffset::MVIN_DRAM_ADDR, val_dram);
    reg_write64(RegOffset::MVIN_SRAM_ADDR, val_sram);
    reg_write64_cached(RegOffset::MVIN_CFG, val_cfg, &shadow_.mvin_cfg);
    const uint64_t bytes = mvin_transfer_bytes(cfg);
    pending_mvin_ctx_[dma_id].shape_key = mvin_shape_key(dma_id, cfg, bytes);
    pending_mvin_ctx_[dma_id].start = std::chrono::steady_clock::now();
    pending_mvin_ctx_[dma_id].bytes = bytes;
    pending_mvin_ctx_[dma_id].valid = true;
    reg_write(RegOffset::START,
              BIT_START_DMA_MVIN |
              static_cast<uint32_t>(REG_FIELD(START_REG, MVIN_DMA_SEL, dma_id)));
}

void NpuRuntime::run_mvout(const MvoutConfig& cfg) {
    run_mvout_async(0, cfg);
    wait_mvout(1u);
}

void NpuRuntime::run_mvout_async(uint32_t dma_id, const MvoutConfig& cfg) {
    validate_mvout(dma_id, cfg);
    if (read_dma_busy_mask(false) & (1u << dma_id)) {
        wait_mvout(1u << dma_id);
    }

    const uint32_t phys = virt_to_phys(cfg.host_ptr);
    const uint64_t val_dram =
        REG_FIELD(MVOUT_CTRL0, DRAM_ADDR, phys) |
        REG_FIELD(MVOUT_CTRL0, ROW_NUM, cfg.row_num);
    const uint64_t val_sram =
        REG_FIELD(MVOUT_CTRL1, SRAM_ADDR, cfg.sram_addr) |
        REG_FIELD(MVOUT_CTRL1, COL_NUM, cfg.col_num);
    const uint64_t val_cfg =
        REG_FIELD(CFG_MVOUT0, OUTPUT_TYPE, cfg.output_type) |
        REG_FIELD(CFG_MVOUT0, OUTPUT_PRECISION, cfg.precision) |
        REG_FIELD(CFG_MVOUT0, IS_QUANT, 0) |
        REG_FIELD(CFG_MVOUT0, SOURCE, 0) |
        REG_FIELD(CFG_MVOUT0, PER_CHANNEL, 0) |
        REG_FIELD(CFG_MVOUT0, SRAM_STRIDE, cfg.sram_stride) |
        REG_FIELD(CFG_MVOUT0, DRAM_STRIDE, cfg.dram_stride);

    reg_write64(RegOffset::MVOUT_DRAM_ADDR, val_dram);
    reg_write64(RegOffset::MVOUT_SRAM_ADDR, val_sram);
    reg_write64_cached(RegOffset::MVOUT_CFG, val_cfg, &shadow_.mvout_cfg);
    const uint64_t bytes = mvout_transfer_bytes(cfg);
    pending_mvout_ctx_[dma_id].shape_key = mvout_shape_key(dma_id, cfg, bytes);
    pending_mvout_ctx_[dma_id].start = std::chrono::steady_clock::now();
    pending_mvout_ctx_[dma_id].bytes = bytes;
    pending_mvout_ctx_[dma_id].valid = true;
    reg_write(RegOffset::START,
              BIT_START_DMA_MVOUT |
              static_cast<uint32_t>(REG_FIELD(START_REG, MVOUT_DMA_SEL, dma_id)));
}

void NpuRuntime::wait_mvin(uint32_t dma_mask) {
    wait_dma_idle(true, dma_mask);
    release_mvin_staging(dma_mask);
    ack_irq(BIT_START_DMA_MVIN);
}

void NpuRuntime::wait_mvout(uint32_t dma_mask) {
    wait_dma_idle(false, dma_mask);
    ack_irq(BIT_START_DMA_MVOUT);
}

void NpuRuntime::run_matvec(const MatvecConfig& cfg) {
    validate_matvec(cfg);

    const uint64_t decode_flow =
        cfg.decode_flow ? cfg.decode_flow : make_matvec_bypass_output_flow(cfg.mat_height);
    const uint64_t val_cfg =
        REG_FIELD(CFG_COMPUTE0, OPTYPE, 2) |
        REG_FIELD(CFG_COMPUTE0, GEMV_MODE, cfg.gemv_mode);
    const uint64_t val_ctrl0 =
        REG_FIELD(MATVEC_CTRL0, MAT_ADDR, cfg.mat_addr) |
        REG_FIELD(MATVEC_CTRL0, VEC_ADDR, cfg.vec_addr);
    const uint64_t val_ctrl1 =
        REG_FIELD(MATVEC_CTRL1, MAT_W, cfg.mat_width) |
        REG_FIELD(MATVEC_CTRL1, MAT_H, cfg.mat_height) |
        REG_FIELD(MATVEC_CTRL1, MAT_S, cfg.output_addr) |
        REG_FIELD(MATVEC_CTRL1, VEC_S, cfg.scale_addr);

    reg_write64_cached(RegOffset::CFG_COMPUTE_1, val_cfg, &shadow_.compute_cfg);
    reg_write64(RegOffset::CFG_COMPUTE_2, decode_flow);
    reg_write64(RegOffset::MATVEC_CTRL_0, val_ctrl0);
    reg_write64(RegOffset::MATVEC_CTRL_1, val_ctrl1);
    reg_write64(RegOffset::MATVEC_ACT_SCALE,
                REG_FIELD(MATVEC_ACT_SCALE_CTRL, ACT_SCALE_ADDR, cfg.act_scale_addr));
    prepare_wait_irq_before_start("matvec", matvec_shape_key(cfg),
                                  static_cast<uint64_t>(cfg.mat_height) * cfg.mat_width,
                                  BIT_START_MATVEC);
    reg_write(RegOffset::START, BIT_START_MATVEC);
    wait_irq(BIT_START_MATVEC);
}

void NpuRuntime::run_gemv_pingpong(const GemvPingPongConfig& cfg) {
    const bool embedded_scale = gemv_uses_embedded_scale(cfg.gemv_mode);
    if (!cfg.act_ptr || !cfg.weight_ptr || !cfg.output_ptr ||
        (!embedded_scale && !cfg.scale_ptr)) {
        throw std::runtime_error("GEMV ping-pong buffer pointer is null");
    }
    if (cfg.m == 0 || cfg.n == 0) {
        throw std::runtime_error("GEMV ping-pong M/N must be non-zero");
    }
    if (cfg.gemv_mode > NPU_GEMV_MODE_W16A16) {
        throw std::runtime_error("GEMV ping-pong gemv_mode must be 0(W4A16), 1(W8A16), or 2(W16A16)");
    }
    if (cfg.output_precision != 1 && cfg.output_precision != 3) {
        throw std::runtime_error("GEMV ping-pong output_precision must be 1(FP16) or 3(FP32)");
    }

    const uint32_t row_tiles = ceil_div_u32(cfg.m, NPU_GEMV_ROW_TILE_ELEMS);
    const uint32_t col_tiles = ceil_div_u32(cfg.n, NPU_GEMV_TILE_ELEMS);
    const uint32_t weight_tile_bytes = gemv_weight_tile_bytes(cfg.gemv_mode);
    const uint32_t act_bytes = col_tiles * kGemvActTileBytes;
    const uint32_t act_payload_bytes =
        cfg.gemv_mode == NPU_GEMV_MODE_W4A16 ? act_bytes * 2u : act_bytes;
    const uint32_t scale_bytes = row_tiles * col_tiles * kGemvScaleTileBytes;
    const uint32_t output_bytes = static_cast<uint32_t>(cfg.m) * NPU_GEMV_FP16_BYTES;
    const uint32_t bytes_per_row_tile = col_tiles * weight_tile_bytes;
    const uint32_t ping_capacity = kGemvPongWeightSpmBase - kGemvPingWeightSpmBase;
    const uint32_t pong_capacity = NPU_GEMV_SPM_BYTES - kGemvPongWeightSpmBase;
    const uint32_t weight_capacity = std::min(ping_capacity, pong_capacity);
    const uint32_t max_row_tiles = bytes_per_row_tile ? weight_capacity / bytes_per_row_tile : 0;

    if (act_payload_bytes > NPU_GEMV_ACT_PAYLOAD_BYTES) {
        throw std::runtime_error("GEMV activation vector overlaps reserved RoPE LUT window");
    }
    if (!embedded_scale &&
        kGemvScaleSpmBase + scale_bytes > kGemvPingWeightSpmBase) {
        throw std::runtime_error("GEMV scale layout overlaps weight ping buffer");
    }
    if (kGemvOutputSpmBase + output_bytes > NPU_GEMV_SPM_BYTES ||
        kGemvOutputSpmBase + output_bytes > std::numeric_limits<uint16_t>::max() + 1u) {
        throw std::runtime_error("GEMV output exceeds output SPM/MATVEC address capacity");
    }
    if (max_row_tiles == 0) {
        throw std::runtime_error("GEMV row tile cannot fit ping-pong weight buffers");
    }

    auto make_mvin_cfg = [](void* host_ptr, uint32_t spm_addr, uint32_t bytes,
                            uint8_t input_type, uint8_t precision) {
        MvinConfig mvin = {};
        mvin.host_ptr = host_ptr;
        mvin.sram_addr = spm_addr;
        mvin.col_num = bytes - 1u;
        mvin.row_num = 0;
        mvin.precision = precision;
        mvin.input_type = input_type;
        return mvin;
    };

    auto make_mvout_cfg = [](void* host_ptr, uint32_t spm_addr, uint32_t rows,
                             uint8_t precision) {
        MvoutConfig mvout = {};
        mvout.host_ptr = host_ptr;
        mvout.sram_addr = spm_addr;
        mvout.col_num = 0;
        mvout.row_num = rows - 1u;
        mvout.sram_stride = 1;
        mvout.dram_stride = 1;
        mvout.precision = precision;
        mvout.output_type = 0;
        return mvout;
    };

    auto block_rows_for = [&](uint32_t row_base) {
        const uint32_t remaining = static_cast<uint32_t>(cfg.m) - row_base;
        return std::min(max_row_tiles * NPU_GEMV_ROW_TILE_ELEMS, remaining);
    };
    auto weight_offset_for = [&](uint32_t row_base) {
        return (row_base / NPU_GEMV_ROW_TILE_ELEMS) * bytes_per_row_tile;
    };
    auto scale_addr_for = [&](uint32_t row_base) {
        return kGemvScaleSpmBase +
               (row_base / NPU_GEMV_ROW_TILE_ELEMS) * col_tiles * kGemvScaleTileBytes;
    };

    auto* weight_base = static_cast<uint8_t*>(cfg.weight_ptr);
    run_mvin(make_mvin_cfg(cfg.act_ptr, kGemvActSpmBase, act_payload_bytes,
                           NPU_GEMV_MVIN_ACT, 2));
    if (!embedded_scale) {
        run_mvin(make_mvin_cfg(cfg.scale_ptr, kGemvScaleSpmBase, scale_bytes,
                               NPU_GEMV_MVIN_INPUT_SPM, 2));
    }

    const uint32_t first_rows = block_rows_for(0);
    const uint32_t first_weight_bytes =
        ceil_div_u32(first_rows, NPU_GEMV_ROW_TILE_ELEMS) * bytes_per_row_tile;
    run_mvin(make_mvin_cfg(weight_base, kGemvPingWeightSpmBase, first_weight_bytes,
                           NPU_GEMV_MVIN_WEIGHT, 1));

    uint32_t block_idx = 0;
    for (uint32_t row_base = 0; row_base < cfg.m; row_base += block_rows_for(row_base), ++block_idx) {
        const uint32_t rows = block_rows_for(row_base);
        const uint32_t cur_weight_spm =
            (block_idx & 1u) ? kGemvPongWeightSpmBase : kGemvPingWeightSpmBase;

        bool prefetch_started = false;
        const uint32_t next_row_base = row_base + rows;
        if (next_row_base < cfg.m) {
            const uint32_t next_rows = block_rows_for(next_row_base);
            const uint32_t next_weight_bytes =
                ceil_div_u32(next_rows, NPU_GEMV_ROW_TILE_ELEMS) * bytes_per_row_tile;
            const uint32_t next_weight_spm =
                ((block_idx + 1u) & 1u) ? kGemvPongWeightSpmBase : kGemvPingWeightSpmBase;
            run_mvin_async(
                0,
                make_mvin_cfg(
                    weight_base + weight_offset_for(next_row_base),
                    next_weight_spm,
                    next_weight_bytes,
                    NPU_GEMV_MVIN_WEIGHT,
                    1));
            prefetch_started = true;
        }

        MatvecConfig matvec = {};
        matvec.mat_addr = cur_weight_spm;
        matvec.vec_addr = kGemvActSpmBase;
        matvec.mat_width = cfg.n;
        matvec.mat_height = static_cast<uint16_t>(rows);
        matvec.output_addr =
            static_cast<uint16_t>(kGemvOutputSpmBase + row_base * NPU_GEMV_FP16_BYTES);
        matvec.scale_addr = static_cast<uint16_t>(
            embedded_scale ? cur_weight_spm : scale_addr_for(row_base));
        matvec.act_scale_addr = kGemvActSpmBase + act_bytes;
        matvec.gemv_mode = cfg.gemv_mode;
        run_matvec(matvec);

        if (prefetch_started) {
            wait_mvin(1u);
        }
    }

    run_mvout(make_mvout_cfg(cfg.output_ptr, kGemvOutputSpmBase, cfg.m,
                             cfg.output_precision));
}

} // namespace

extern "C" {

int npu_init(void) {
    if (g_runtime) {
        return 0;
    }
    auto* rt = new NpuRuntime();
    if (!rt->init()) {
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

void npu_dma_mvin(void* host_ptr, uint32_t sram_addr, uint32_t col_num,
                  uint32_t row_num, uint16_t sram_stride, uint32_t dram_stride,
                  uint8_t precision, uint8_t input_type, bool dest, bool is_bias,
                  bool is_quant, uint32_t quant_zero, uint16_t quant_scale,
                  uint16_t quant_shift) {
    try {
        MvinConfig cfg = {host_ptr, sram_addr, col_num, row_num, sram_stride,
                          dram_stride, precision, input_type, dest, is_bias,
                          is_quant, quant_zero, quant_scale, quant_shift};
        runtime().run_mvin(cfg);
    } catch (const std::exception& ex) {
        report_exception("npu_dma_mvin", ex);
    }
}

void npu_dma_mvout(void* host_ptr, uint32_t sram_addr, uint32_t col_num,
                   uint32_t row_num, uint16_t sram_stride, uint32_t dram_stride,
                   uint8_t precision, uint8_t output_type, bool source,
                   bool is_quant, uint32_t quant_zero, uint32_t scale_or_addr) {
    try {
        MvoutConfig cfg = {host_ptr, sram_addr, col_num, row_num, sram_stride,
                           dram_stride, precision, output_type, source, is_quant,
                           quant_zero, scale_or_addr, false};
        runtime().run_mvout(cfg);
    } catch (const std::exception& ex) {
        report_exception("npu_dma_mvout", ex);
    }
}

void npu_dma_mvin_async(uint32_t dma_id, const MvinConfig* cfg) {
    try {
        if (!cfg) {
            throw std::runtime_error("MvinConfig pointer is null");
        }
        runtime().run_mvin_async(dma_id, *cfg);
    } catch (const std::exception& ex) {
        report_exception("npu_dma_mvin_async", ex);
    }
}

void npu_dma_mvout_async(uint32_t dma_id, const MvoutConfig* cfg) {
    try {
        if (!cfg) {
            throw std::runtime_error("MvoutConfig pointer is null");
        }
        runtime().run_mvout_async(dma_id, *cfg);
    } catch (const std::exception& ex) {
        report_exception("npu_dma_mvout_async", ex);
    }
}

void npu_dma_wait_mvin(uint32_t dma_mask) {
    try {
        runtime().wait_mvin(dma_mask);
    } catch (const std::exception& ex) {
        report_exception("npu_dma_wait_mvin", ex);
    }
}

void npu_dma_wait_mvout(uint32_t dma_mask) {
    try {
        runtime().wait_mvout(dma_mask);
    } catch (const std::exception& ex) {
        report_exception("npu_dma_wait_mvout", ex);
    }
}

void npu_rope_lut_mvin(void* rope_table_base, uint16_t position) {
    try {
        runtime().run_mvin(make_rope_lut_mvin_config(rope_table_base, position));
    } catch (const std::exception& ex) {
        report_exception("npu_rope_lut_mvin", ex);
    }
}

void npu_rope_lut_mvin_async(uint32_t dma_id, void* rope_table_base, uint16_t position) {
    try {
        runtime().run_mvin_async(dma_id, make_rope_lut_mvin_config(rope_table_base, position));
    } catch (const std::exception& ex) {
        report_exception("npu_rope_lut_mvin_async", ex);
    }
}

void npu_matvec_run(uint32_t mat_addr, uint32_t vec_addr, uint16_t mat_width,
                    uint16_t mat_height, uint16_t output_addr,
                    uint16_t scale_addr) {
    npu_matvec_mode_run(mat_addr, vec_addr, mat_width, mat_height, output_addr,
                        scale_addr, NPU_GEMV_MODE_W4A16);
}

void npu_matvec_mode_run(uint32_t mat_addr, uint32_t vec_addr, uint16_t mat_width,
                         uint16_t mat_height, uint16_t output_addr,
                         uint16_t scale_addr, uint8_t gemv_mode) {
    try {
        MatvecConfig cfg = {};
        cfg.mat_addr = mat_addr;
        cfg.vec_addr = vec_addr;
        cfg.mat_width = mat_width;
        cfg.mat_height = mat_height;
        cfg.output_addr = output_addr;
        cfg.scale_addr = scale_addr;
        cfg.gemv_mode = gemv_mode;
        cfg.act_scale_addr = default_act_scale_addr(vec_addr, mat_width, gemv_mode);
        runtime().run_matvec(cfg);
    } catch (const std::exception& ex) {
        report_exception("npu_matvec_mode_run", ex);
    }
}

void npu_matvec_silu_run(uint32_t mat_addr, uint32_t vec_addr, uint16_t mat_width,
                         uint16_t mat_height, uint16_t output_addr,
                         uint16_t scale_addr) {
    try {
        MatvecConfig cfg = {};
        cfg.mat_addr = mat_addr;
        cfg.vec_addr = vec_addr;
        cfg.mat_width = mat_width;
        cfg.mat_height = mat_height;
        cfg.output_addr = output_addr;
        cfg.scale_addr = scale_addr;
        cfg.gemv_mode = NPU_GEMV_MODE_W4A16;
        cfg.act_scale_addr = default_act_scale_addr(vec_addr, mat_width, cfg.gemv_mode);
        cfg.decode_flow = make_matvec_silu_buffer_flow(mat_height);
        runtime().run_matvec(cfg);
    } catch (const std::exception& ex) {
        report_exception("npu_matvec_silu_run", ex);
    }
}

uint64_t npu_decode_flow_make(uint8_t src0, uint8_t src1,
                              uint8_t unary_op, uint8_t binary_op,
                              uint8_t reduce_op, uint8_t dst,
                              uint8_t src_buffer_id, uint8_t dst_buffer_id,
                              uint16_t elem_count, uint16_t position) {
    return make_decode_flow_value(src0, src1, unary_op, binary_op, reduce_op,
                                  dst, src_buffer_id, dst_buffer_id,
                                  elem_count, position);
}

void npu_matvec_decode_flow_run(uint32_t mat_addr, uint32_t vec_addr,
                                uint16_t mat_width, uint16_t mat_height,
                                uint16_t output_addr, uint16_t scale_addr,
                                uint8_t gemv_mode, uint64_t decode_flow) {
    try {
        MatvecConfig cfg = {};
        cfg.mat_addr = mat_addr;
        cfg.vec_addr = vec_addr;
        cfg.mat_width = mat_width;
        cfg.mat_height = mat_height;
        cfg.output_addr = output_addr;
        cfg.scale_addr = scale_addr;
        cfg.gemv_mode = gemv_mode;
        cfg.act_scale_addr = default_act_scale_addr(vec_addr, mat_width, gemv_mode);
        cfg.decode_flow = decode_flow;
        runtime().run_matvec(cfg);
    } catch (const std::exception& ex) {
        report_exception("npu_matvec_decode_flow_run", ex);
    }
}

void npu_gemv_pingpong_run(void* act_ptr, void* scale_ptr, void* weight_ptr,
                           void* output_ptr, uint16_t m, uint16_t n) {
    npu_gemv_pingpong_mode_precision_run(act_ptr, scale_ptr, weight_ptr,
                                         output_ptr, m, n,
                                         NPU_GEMV_MODE_W4A16, 1);
}

void npu_gemv_pingpong_mode_run(void* act_ptr, void* scale_ptr, void* weight_ptr,
                                void* output_ptr, uint16_t m, uint16_t n,
                                uint8_t gemv_mode) {
    npu_gemv_pingpong_mode_precision_run(act_ptr, scale_ptr, weight_ptr,
                                         output_ptr, m, n, gemv_mode, 1);
}

void npu_gemv_pingpong_mode_precision_run(void* act_ptr, void* scale_ptr,
                                          void* weight_ptr, void* output_ptr,
                                          uint16_t m, uint16_t n,
                                          uint8_t gemv_mode,
                                          uint8_t output_precision) {
    try {
        GemvPingPongConfig cfg = {};
        cfg.act_ptr = act_ptr;
        cfg.scale_ptr = scale_ptr;
        cfg.weight_ptr = weight_ptr;
        cfg.output_ptr = output_ptr;
        cfg.m = m;
        cfg.n = n;
        cfg.gemv_mode = gemv_mode;
        cfg.output_precision = output_precision;
        runtime().run_gemv_pingpong(cfg);
    } catch (const std::exception& ex) {
        report_exception("npu_gemv_pingpong_mode_precision_run", ex);
    }
}

} // extern "C"
