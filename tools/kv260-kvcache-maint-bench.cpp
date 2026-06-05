#include "npu_runtime.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kLayers = 24;
constexpr uint32_t kKvHeads = 5;
constexpr uint32_t kHeadDim = 64;
constexpr uint32_t kQ8Block = 32;
constexpr uint32_t kQ8RowBytes = 68;
constexpr uint32_t kRowTile = 32;
constexpr uint32_t kDmaAlign = 256;
constexpr uint32_t kLineBytes = 64;
constexpr uint32_t kCurrentHeadStride = 65536;
constexpr uint32_t kCurrentTileBytes = kLineBytes + kRowTile * kHeadDim;
constexpr uint32_t kCurrentWindowTiles = kCurrentHeadStride / kCurrentTileBytes;
constexpr uint32_t kCurrentWindowTokens = kCurrentWindowTiles * kRowTile;
constexpr uint32_t kCurrentScaleRegion = ((kCurrentWindowTokens * 2 + kDmaAlign - 1) / kDmaAlign) * kDmaAlign;
constexpr uint32_t kCurrentQuantTileBytes = kRowTile * kHeadDim;

static_assert(kCurrentWindowTokens == 992, "current layout expectation changed");

struct q8_0_block {
    uint16_t d;
    int8_t qs[kQ8Block];
};
static_assert(sizeof(q8_0_block) == kQ8RowBytes / 2, "unexpected q8 block");

struct buffer {
    void * ptr = nullptr;
    size_t bytes = 0;
    bool cma = false;

    buffer() = default;
    buffer(size_t size, bool use_cma) : bytes(size), cma(use_cma) {
        if (use_cma) {
            ptr = npu_mem_alloc(size);
        } else {
            void * p = nullptr;
            if (posix_memalign(&p, 256, size) == 0) {
                ptr = p;
            }
        }
        if (ptr != nullptr) {
            std::memset(ptr, 0, size);
        }
    }

    buffer(const buffer &) = delete;
    buffer & operator=(const buffer &) = delete;

    buffer(buffer && other) noexcept {
        ptr = other.ptr;
        bytes = other.bytes;
        cma = other.cma;
        other.ptr = nullptr;
    }

    buffer & operator=(buffer && other) noexcept {
        if (this != &other) {
            release();
            ptr = other.ptr;
            bytes = other.bytes;
            cma = other.cma;
            other.ptr = nullptr;
        }
        return *this;
    }

    ~buffer() {
        release();
    }

    void release() {
        if (ptr != nullptr) {
            if (cma) {
                npu_mem_free(ptr);
            } else {
                free(ptr);
            }
            ptr = nullptr;
        }
    }
};

uint64_t now_us() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration_cast<std::chrono::microseconds>(
            clock::now().time_since_epoch()).count();
}

uint32_t current_scale_addr(uint32_t head, uint32_t token) {
    const uint32_t cell = token % kCurrentWindowTokens;
    return head * kCurrentHeadStride + cell * sizeof(uint16_t);
}

uint32_t current_quant_addr(uint32_t head, uint32_t token, uint32_t dim) {
    const uint32_t cell = token % kCurrentWindowTokens;
    const uint32_t tile = cell / kRowTile;
    const uint32_t lane = cell % kRowTile;
    return head * kCurrentHeadStride + kCurrentScaleRegion +
        tile * kCurrentQuantTileBytes + lane * kHeadDim + dim;
}

uint32_t trans_scale_region(uint32_t max_tokens) {
    return ((max_tokens * sizeof(uint16_t) + kDmaAlign - 1) / kDmaAlign) * kDmaAlign;
}

uint32_t trans_dense_head_stride(uint32_t max_tokens) {
    return trans_scale_region(max_tokens) + kHeadDim * max_tokens;
}

uint32_t trans_tiled_head_stride(uint32_t max_tokens) {
    const uint32_t tiles = (max_tokens + kRowTile - 1) / kRowTile;
    return trans_scale_region(max_tokens) + tiles * kHeadDim * kRowTile;
}

uint32_t natural_scale_region(uint32_t max_tokens) {
    return trans_scale_region(max_tokens);
}

uint32_t natural_head_stride(uint32_t max_tokens) {
    const uint32_t tiles = (max_tokens + kRowTile - 1) / kRowTile;
    return natural_scale_region(max_tokens) + tiles * kCurrentQuantTileBytes;
}

uint32_t natural_scale_addr(uint32_t max_tokens, uint32_t head, uint32_t token) {
    return head * natural_head_stride(max_tokens) + token * sizeof(uint16_t);
}

uint32_t natural_quant_addr(uint32_t max_tokens, uint32_t head, uint32_t token, uint32_t dim) {
    const uint32_t tile = token / kRowTile;
    const uint32_t lane = token % kRowTile;
    return head * natural_head_stride(max_tokens) + natural_scale_region(max_tokens) +
        tile * kCurrentQuantTileBytes + lane * kHeadDim + dim;
}

uint32_t trans_dense_quant_addr(uint32_t max_tokens, uint32_t head, uint32_t token, uint32_t dim) {
    return head * trans_dense_head_stride(max_tokens) +
        trans_scale_region(max_tokens) + dim * max_tokens + token;
}

uint32_t trans_tiled_quant_addr(uint32_t max_tokens, uint32_t head, uint32_t token, uint32_t dim) {
    const uint32_t tile = token / kRowTile;
    const uint32_t lane = token % kRowTile;
    return head * trans_tiled_head_stride(max_tokens) +
        trans_scale_region(max_tokens) + tile * kHeadDim * kRowTile + dim * kRowTile + lane;
}

uint32_t trans_scale_addr(uint32_t max_tokens, uint32_t head, uint32_t token, bool tiled) {
    return head * (tiled ? trans_tiled_head_stride(max_tokens) : trans_dense_head_stride(max_tokens)) +
        token * sizeof(uint16_t);
}

void fill_source(uint8_t * src, size_t bytes) {
    uint32_t x = 0x12345678u;
    for (size_t i = 0; i < bytes; ++i) {
        x = x * 1664525u + 1013904223u;
        src[i] = static_cast<uint8_t>(x >> 24);
    }
    const size_t rows = bytes / kQ8RowBytes;
    for (size_t r = 0; r < rows; ++r) {
        q8_0_block * b = reinterpret_cast<q8_0_block *>(src + r * kQ8RowBytes);
        b[0].d = static_cast<uint16_t>(0x3400u + (r & 0x1fu));
        b[1].d = b[0].d;
    }
}

void convert_one_current(const uint8_t * row, uint32_t token, uint32_t head, uint8_t * dst) {
    const q8_0_block * blocks = reinterpret_cast<const q8_0_block *>(row);
    std::memcpy(dst + current_scale_addr(head, token), &blocks[0].d, sizeof(uint16_t));
    uint8_t * qdst = dst + current_quant_addr(head, token, 0);
    std::memcpy(qdst, blocks[0].qs, kQ8Block);
    std::memcpy(qdst + kQ8Block, blocks[1].qs, kQ8Block);
}

void convert_prefix_current(const uint8_t * src, uint32_t tokens, uint8_t * dst) {
    const uint32_t token_stride = kKvHeads * kQ8RowBytes;
    for (uint32_t token = 0; token < tokens; ++token) {
        for (uint32_t head = 0; head < kKvHeads; ++head) {
            convert_one_current(src + token * token_stride + head * kQ8RowBytes, token, head, dst);
        }
    }
}

void convert_one_natural(
        const uint8_t * row,
        uint32_t max_tokens,
        uint32_t token,
        uint32_t head,
        uint8_t * dst) {
    const q8_0_block * blocks = reinterpret_cast<const q8_0_block *>(row);
    std::memcpy(dst + natural_scale_addr(max_tokens, head, token), &blocks[0].d, sizeof(uint16_t));
    uint8_t * qdst = dst + natural_quant_addr(max_tokens, head, token, 0);
    std::memcpy(qdst, blocks[0].qs, kQ8Block);
    std::memcpy(qdst + kQ8Block, blocks[1].qs, kQ8Block);
}

void convert_prefix_natural(const uint8_t * src, uint32_t max_tokens, uint32_t tokens, uint8_t * dst) {
    const uint32_t token_stride = kKvHeads * kQ8RowBytes;
    for (uint32_t token = 0; token < tokens; ++token) {
        for (uint32_t head = 0; head < kKvHeads; ++head) {
            convert_one_natural(src + token * token_stride + head * kQ8RowBytes, max_tokens, token, head, dst);
        }
    }
}

void convert_one_transposed(
        const uint8_t * row,
        uint32_t max_tokens,
        uint32_t token,
        uint32_t head,
        uint8_t * dst,
        bool tiled) {
    const q8_0_block * blocks = reinterpret_cast<const q8_0_block *>(row);
    std::memcpy(dst + trans_scale_addr(max_tokens, head, token, tiled), &blocks[0].d, sizeof(uint16_t));
    for (uint32_t dim = 0; dim < kQ8Block; ++dim) {
        const uint32_t addr = tiled ?
            trans_tiled_quant_addr(max_tokens, head, token, dim) :
            trans_dense_quant_addr(max_tokens, head, token, dim);
        dst[addr] = static_cast<uint8_t>(blocks[0].qs[dim]);
    }
    for (uint32_t dim = 0; dim < kQ8Block; ++dim) {
        const uint32_t d = dim + kQ8Block;
        const uint32_t addr = tiled ?
            trans_tiled_quant_addr(max_tokens, head, token, d) :
            trans_dense_quant_addr(max_tokens, head, token, d);
        dst[addr] = static_cast<uint8_t>(blocks[1].qs[dim]);
    }
}

void convert_prefix_transposed(
        const uint8_t * src,
        uint32_t max_tokens,
        uint32_t tokens,
        uint8_t * dst,
        bool tiled) {
    const uint32_t token_stride = kKvHeads * kQ8RowBytes;
    for (uint32_t token = 0; token < tokens; ++token) {
        for (uint32_t head = 0; head < kKvHeads; ++head) {
            convert_one_transposed(src + token * token_stride + head * kQ8RowBytes,
                    max_tokens, token, head, dst, tiled);
        }
    }
}

void append_current(const uint8_t * output, uint32_t token, const uint16_t * regs, uint8_t * dst) {
    const uint32_t src_cell = token % kCurrentWindowTokens;
    for (uint32_t head = 0; head < kKvHeads; ++head) {
        std::memcpy(dst + current_scale_addr(head, token), &regs[head], sizeof(uint16_t));
        const uint32_t src_addr = head * kCurrentHeadStride + kCurrentScaleRegion +
            (src_cell / kRowTile) * kCurrentQuantTileBytes +
            (src_cell % kRowTile) * kHeadDim;
        std::memcpy(dst + current_quant_addr(head, token, 0), output + src_addr, kHeadDim);
    }
}

void append_natural(const uint8_t * output_rows, uint32_t max_tokens, uint32_t token, const uint16_t * regs, uint8_t * dst) {
    for (uint32_t head = 0; head < kKvHeads; ++head) {
        std::memcpy(dst + natural_scale_addr(max_tokens, head, token), &regs[head], sizeof(uint16_t));
        std::memcpy(dst + natural_quant_addr(max_tokens, head, token, 0), output_rows + head * kHeadDim, kHeadDim);
    }
}

void append_transposed(
        const uint8_t * quant_rows,
        uint32_t max_tokens,
        uint32_t token,
        const uint16_t * regs,
        uint8_t * dst,
        bool tiled) {
    for (uint32_t head = 0; head < kKvHeads; ++head) {
        std::memcpy(dst + trans_scale_addr(max_tokens, head, token, tiled), &regs[head], sizeof(uint16_t));
        const uint8_t * row = quant_rows + head * kHeadDim;
        for (uint32_t dim = 0; dim < kHeadDim; ++dim) {
            const uint32_t addr = tiled ?
                trans_tiled_quant_addr(max_tokens, head, token, dim) :
                trans_dense_quant_addr(max_tokens, head, token, dim);
            dst[addr] = row[dim];
        }
    }
}

template <typename Fn>
double bench_us(Fn fn, int repeat) {
    std::vector<double> samples;
    samples.reserve(repeat);
    for (int i = 0; i < repeat; ++i) {
        const uint64_t t0 = now_us();
        fn();
        const uint64_t t1 = now_us();
        samples.push_back(static_cast<double>(t1 - t0));
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

uint64_t checksum(const uint8_t * ptr, size_t bytes) {
    uint64_t acc = 1469598103934665603ull;
    for (size_t i = 0; i < bytes; i += 4096) {
        acc ^= ptr[i];
        acc *= 1099511628211ull;
    }
    if (bytes) {
        acc ^= ptr[bytes - 1];
    }
    return acc;
}

void run_case(uint32_t tokens, uint32_t max_tokens, uint32_t layers, bool use_cma) {
    const size_t src_one_bytes = static_cast<size_t>(tokens) * kKvHeads * kQ8RowBytes;
    const size_t current_cache_bytes = static_cast<size_t>(kKvHeads) * kCurrentHeadStride;
    const size_t new_k_bytes = static_cast<size_t>(kKvHeads) * natural_head_stride(max_tokens);
    const size_t trans_dense_bytes = static_cast<size_t>(kKvHeads) * trans_dense_head_stride(max_tokens);
    const size_t trans_tiled_bytes = static_cast<size_t>(kKvHeads) * trans_tiled_head_stride(max_tokens);
    const size_t output_bytes = current_cache_bytes;

    std::vector<uint8_t> src_k(src_one_bytes);
    std::vector<uint8_t> src_v(src_one_bytes);
    fill_source(src_k.data(), src_k.size());
    fill_source(src_v.data(), src_v.size());

    buffer cur_k(current_cache_bytes, use_cma);
    buffer cur_v(current_cache_bytes, use_cma);
    buffer new_k(new_k_bytes, use_cma);
    buffer new_v_dense(trans_dense_bytes, use_cma);
    buffer new_v_tiled(trans_tiled_bytes, use_cma);
    buffer output(output_bytes, use_cma);
    if (!cur_k.ptr || !cur_v.ptr || !new_k.ptr || !new_v_dense.ptr || !new_v_tiled.ptr || !output.ptr) {
        std::printf("case tokens=%u max_tokens=%u mem=%s failed allocation\n",
                tokens, max_tokens, use_cma ? "cma" : "malloc");
        std::exit(2);
    }
    fill_source(static_cast<uint8_t *>(output.ptr), output_bytes);

    uint16_t regs[kKvHeads] = {};
    for (uint32_t i = 0; i < kKvHeads; ++i) {
        regs[i] = static_cast<uint16_t>(0x3c00u + i);
    }

    const int prefix_repeat = tokens >= 2048 ? 3 : 7;
    const int append_repeat = 20000;
    const uint32_t append_token = tokens ? tokens - 1 : 0;
    uint8_t quant_rows[kKvHeads * kHeadDim] = {};
    for (uint32_t i = 0; i < sizeof(quant_rows); ++i) {
        quant_rows[i] = static_cast<uint8_t>(i * 13u + 7u);
    }

    const bool current_supported = tokens <= kCurrentWindowTokens;
    const double current_prefix_one = current_supported ? bench_us([&]() {
        convert_prefix_current(src_k.data(), tokens, static_cast<uint8_t *>(cur_k.ptr));
        convert_prefix_current(src_v.data(), tokens, static_cast<uint8_t *>(cur_v.ptr));
    }, prefix_repeat) : 0.0;
    const double current_prefix_all_layers = current_prefix_one * layers;

    const double new_prefix_one_dense = bench_us([&]() {
        convert_prefix_natural(src_k.data(), max_tokens, tokens, static_cast<uint8_t *>(new_k.ptr));
        convert_prefix_transposed(src_v.data(), max_tokens, tokens, static_cast<uint8_t *>(new_v_dense.ptr), false);
    }, prefix_repeat);
    const double new_prefix_all_layers_dense = new_prefix_one_dense * layers;

    const double new_prefix_one_tiled = bench_us([&]() {
        convert_prefix_natural(src_k.data(), max_tokens, tokens, static_cast<uint8_t *>(new_k.ptr));
        convert_prefix_transposed(src_v.data(), max_tokens, tokens, static_cast<uint8_t *>(new_v_tiled.ptr), true);
    }, prefix_repeat);
    const double new_prefix_all_layers_tiled = new_prefix_one_tiled * layers;

    const double current_append_one = current_supported ? bench_us([&]() {
        append_current(static_cast<const uint8_t *>(output.ptr), append_token, regs, static_cast<uint8_t *>(cur_k.ptr));
        append_current(static_cast<const uint8_t *>(output.ptr), append_token, regs, static_cast<uint8_t *>(cur_v.ptr));
    }, append_repeat) : 0.0;
    const double current_append_all_layers = current_append_one * layers;

    const double current_append_clear_one = current_supported ? bench_us([&]() {
        std::memset(output.ptr, 0, output_bytes);
        append_current(static_cast<const uint8_t *>(output.ptr), append_token, regs, static_cast<uint8_t *>(cur_k.ptr));
        std::memset(output.ptr, 0, output_bytes);
        append_current(static_cast<const uint8_t *>(output.ptr), append_token, regs, static_cast<uint8_t *>(cur_v.ptr));
    }, 200) : 0.0;
    const double current_append_clear_all_layers = current_append_clear_one * layers;

    const double new_append_one_dense = bench_us([&]() {
        append_natural(quant_rows, max_tokens, append_token, regs, static_cast<uint8_t *>(new_k.ptr));
        append_transposed(quant_rows, max_tokens, append_token, regs, static_cast<uint8_t *>(new_v_dense.ptr), false);
    }, append_repeat);
    const double new_append_all_layers_dense = new_append_one_dense * layers;

    const double new_append_one_tiled = bench_us([&]() {
        append_natural(quant_rows, max_tokens, append_token, regs, static_cast<uint8_t *>(new_k.ptr));
        append_transposed(quant_rows, max_tokens, append_token, regs, static_cast<uint8_t *>(new_v_tiled.ptr), true);
    }, append_repeat);
    const double new_append_all_layers_tiled = new_append_one_tiled * layers;

    const uint64_t guard =
        checksum(static_cast<const uint8_t *>(cur_k.ptr), current_cache_bytes) ^
        checksum(static_cast<const uint8_t *>(cur_v.ptr), current_cache_bytes) ^
        checksum(static_cast<const uint8_t *>(new_k.ptr), new_k_bytes) ^
        checksum(static_cast<const uint8_t *>(new_v_dense.ptr), trans_dense_bytes) ^
        checksum(static_cast<const uint8_t *>(new_v_tiled.ptr), trans_tiled_bytes);

    std::printf(
            "RESULT mem=%s tokens=%u max_tokens=%u layers=%u "
            "current_supported=%u "
            "current_prefix_one_layer_us=%.2f current_prefix_all_layers_ms=%.3f "
            "new_dense_prefix_one_layer_us=%.2f new_dense_prefix_all_layers_ms=%.3f "
            "new_tiled_prefix_one_layer_us=%.2f new_tiled_prefix_all_layers_ms=%.3f "
            "current_append_one_layer_us=%.4f current_append_all_layers_us=%.2f "
            "current_append_with_clear_one_layer_us=%.2f current_append_with_clear_all_layers_ms=%.3f "
            "new_dense_append_one_layer_us=%.4f new_dense_append_all_layers_us=%.2f "
            "new_tiled_append_one_layer_us=%.4f new_tiled_append_all_layers_us=%.2f "
            "bytes_current_cache_per_layer=%zu bytes_new_dense_v_per_layer=%zu bytes_new_tiled_v_per_layer=%zu guard=%llu\n",
            use_cma ? "cma" : "malloc",
            tokens,
            max_tokens,
            layers,
            current_supported ? 1u : 0u,
            current_prefix_one,
            current_prefix_all_layers / 1000.0,
            new_prefix_one_dense,
            new_prefix_all_layers_dense / 1000.0,
            new_prefix_one_tiled,
            new_prefix_all_layers_tiled / 1000.0,
            current_append_one,
            current_append_all_layers,
            current_append_clear_one,
            current_append_clear_all_layers / 1000.0,
            new_append_one_dense,
            new_append_all_layers_dense,
            new_append_one_tiled,
            new_append_all_layers_tiled,
            current_cache_bytes * 2,
            new_k_bytes + trans_dense_bytes,
            new_k_bytes + trans_tiled_bytes,
            static_cast<unsigned long long>(guard));
}

uint32_t arg_u32(char ** argv, int argc, const char * name, uint32_t fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == name) {
            return static_cast<uint32_t>(std::strtoul(argv[i + 1], nullptr, 0));
        }
    }
    return fallback;
}

bool has_arg(char ** argv, int argc, const char * name) {
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == name) {
            return true;
        }
    }
    return false;
}

} // namespace

int main(int argc, char ** argv) {
    const uint32_t layers = arg_u32(argv, argc, "--layers", kLayers);
    const bool no_cma = has_arg(argv, argc, "--no-cma");

    npu_device * dev = nullptr;
    bool cma_available = false;
    if (!no_cma) {
        const int rc = npu_open(&dev, nullptr);
        if (rc == 0) {
            cma_available = true;
        } else {
            std::printf("WARN npu_open failed rc=%d; only malloc measurements will run\n", rc);
        }
    }

    std::printf("kv260_kvcache_maint_bench current_window_tokens=%u current_scale_region=%u current_head_stride=%u layers=%u\n",
            kCurrentWindowTokens, kCurrentScaleRegion, kCurrentHeadStride, layers);

    for (bool use_cma : {false, true}) {
        if (use_cma && !cma_available) {
            continue;
        }
        run_case(128, 992, layers, use_cma);
        run_case(512, 992, layers, use_cma);
        run_case(991, 992, layers, use_cma);
        run_case(4096, 4096, layers, use_cma);
    }

    if (dev != nullptr) {
        npu_close(dev);
    }
    return 0;
}
