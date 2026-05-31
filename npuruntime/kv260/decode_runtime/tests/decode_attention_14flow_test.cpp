#include "npu_regs_compat.h"
#include "npu_runtime.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

namespace {

constexpr uint32_t kTileElems = 128;
constexpr uint32_t kRowTileElems = 32;
constexpr uint32_t kLineBytes = 64;
constexpr uint32_t kMvinAlign = 256;
constexpr uint32_t kFp16Bytes = 2;

constexpr uint32_t kHiddenDim = 960;
constexpr uint32_t kHeadDim = 64;
constexpr uint32_t kQHeads = 15;
constexpr uint32_t kKvHeads = 5;
constexpr uint32_t kGroupSize = kQHeads / kKvHeads;
constexpr uint32_t kKvDim = kKvHeads * kHeadDim;
constexpr uint32_t kKvHeadStrideBytes = 65536;
constexpr uint32_t kWeightSpmPingBase = 0x00000;
constexpr uint32_t kWeightSpmPongBase = 0x20000;

constexpr uint32_t kInputSpmBase = 0x00000;
constexpr uint32_t kOutputSpmBase = 0x00000;
constexpr uint32_t kHiddenActBase = 0x0000;
constexpr uint32_t kAttentionActBase = 0x0000;
constexpr uint32_t kQGroupActBase = 0x5000;
constexpr uint32_t kScoreActBase = 0x6000;

constexpr uint16_t kFp16Zero = 0x0000;
constexpr uint16_t kFp16One = 0x3c00;

struct NpuBuffer {
    void* ptr = nullptr;
    size_t bytes = 0;

    explicit NpuBuffer(size_t size) : ptr(npu_mem_alloc(size)), bytes(size) {}
    ~NpuBuffer() {
        if (ptr) {
            npu_mem_free(ptr);
        }
    }

    NpuBuffer(const NpuBuffer&) = delete;
    NpuBuffer& operator=(const NpuBuffer&) = delete;

    uint8_t* data() { return static_cast<uint8_t*>(ptr); }
    const uint8_t* data() const { return static_cast<const uint8_t*>(ptr); }
};

uint32_t align_up(uint32_t value, uint32_t alignment) {
    return (value + alignment - 1u) & ~(alignment - 1u);
}

uint32_t ceil_div(uint32_t a, uint32_t b) {
    return (a + b - 1u) / b;
}

void store_u16_le(uint8_t* dst, uint16_t value) {
    dst[0] = static_cast<uint8_t>(value & 0xffu);
    dst[1] = static_cast<uint8_t>(value >> 8);
}

uint16_t load_u16_le(const uint8_t* src) {
    return static_cast<uint16_t>(src[0]) |
           static_cast<uint16_t>(static_cast<uint16_t>(src[1]) << 8);
}

uint32_t fp32_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

uint16_t fp32_to_fp16_bits(float value) {
    const uint32_t bits = fp32_bits(value);
    const uint32_t sign = (bits >> 31) & 1u;
    const uint32_t exp = (bits >> 23) & 0xffu;
    const uint32_t frac = bits & 0x7fffffu;
    const int exp16 = static_cast<int>(exp) - 127 + 15;
    if (exp == 0 || exp16 <= 0) {
        return static_cast<uint16_t>(sign << 15);
    }
    if (exp == 0xffu || exp16 >= 31) {
        return static_cast<uint16_t>((sign << 15) | 0x7c00u);
    }
    const uint32_t mant = (frac | 0x800000u) + 0x1000u;
    return static_cast<uint16_t>((sign << 15) |
                                 (static_cast<uint32_t>(exp16) << 10) |
                                 ((mant >> 13) & 0x3ffu));
}

float fp16_to_float(uint16_t h) {
    const uint32_t sign = (h >> 15) & 1u;
    const uint32_t exp = (h >> 10) & 0x1fu;
    const uint32_t frac = h & 0x3ffu;
    uint32_t bits = sign << 31;
    if (exp == 0) {
        if (frac == 0) {
            float value = 0.0f;
            std::memcpy(&value, &bits, sizeof(value));
            return value;
        }
        float value = std::ldexp(static_cast<float>(frac), -24);
        return sign ? -value : value;
    }
    if (exp == 0x1fu) {
        bits |= 0xffu << 23;
        bits |= frac ? 0x400000u : 0u;
    } else {
        bits |= (exp - 15u + 127u) << 23;
        bits |= frac << 13;
    }
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

uint16_t fp16_abs_bits(uint16_t value) {
    return static_cast<uint16_t>(value & 0x7fffu);
}

uint16_t fp16_div_pow2(uint16_t value, uint32_t shift) {
    uint16_t exp = static_cast<uint16_t>((value >> 10) & 0x1fu);
    uint16_t mant = static_cast<uint16_t>(0x400u | (value & 0x3ffu));
    if ((value & 0x7fffu) == 0) {
        return 0;
    }
    if (exp == 0x1fu) {
        return 0x7c00u;
    }
    if (exp <= shift) {
        const uint32_t sub_shift = shift - exp + 1u;
        if (sub_shift >= 11u) {
            return 0;
        }
        const uint16_t rounded =
            static_cast<uint16_t>((mant + (uint16_t(1) << (sub_shift - 1u))) >>
                                  sub_shift);
        return (rounded & 0x400u) ? 0x0400u : rounded;
    }
    return static_cast<uint16_t>((value & 0x03ffu) | ((exp - shift) << 10));
}

int8_t kv_quantize_ref(uint16_t value, uint16_t max_abs) {
    const bool sign = (value & 0x8000u) != 0;
    const uint16_t value_exp = static_cast<uint16_t>((value >> 10) & 0x1fu);
    const uint16_t max_exp = static_cast<uint16_t>((max_abs >> 10) & 0x1fu);
    if ((max_abs & 0x7fffu) == 0 || (value & 0x7fffu) == 0 ||
        value_exp == 0 || max_exp == 0) {
        return 0;
    }
    const uint64_t value_mant = 0x400ull | (value & 0x3ffu);
    const uint64_t max_mant = 0x400ull | (max_abs & 0x3ffu);
    int exp_diff = static_cast<int>(value_exp) - static_cast<int>(max_exp);
    uint64_t numerator = value_mant << 7;
    if (exp_diff > 0) {
        numerator <<= static_cast<uint32_t>(exp_diff);
    } else if (exp_diff < 0) {
        const uint32_t shift_amount = static_cast<uint32_t>(-exp_diff);
        numerator = (shift_amount >= 64u)
            ? 0
            : ((numerator + (1ull << (shift_amount - 1u))) >> shift_amount);
    }
    uint64_t q_abs = (numerator + (max_mant >> 1u)) / max_mant;
    if (q_abs > 127u) {
        q_abs = 127u;
    }
    const int signed_q = sign ? -static_cast<int>(q_abs) : static_cast<int>(q_abs);
    return static_cast<int8_t>(signed_q);
}

uint32_t rng_next(uint32_t* state) {
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

uint64_t decode_flow(uint8_t src0, uint8_t src1, uint8_t unary, uint8_t binary,
                     uint8_t reduce, uint8_t dst, uint16_t elem_count,
                     uint16_t position, uint8_t group_count = 1,
                     bool kv_quant = false) {
    uint64_t flow = npu_decode_flow_make(src0, src1, unary, binary, reduce, dst,
                                         0, 0, elem_count, position);
    if (kv_quant) {
        flow |= NPU_DECODE_FLAG_KV_QUANT;
    }
    if (group_count == 0 || group_count > 4) {
        std::fprintf(stderr, "invalid decode group_count=%u\n", group_count);
        std::abort();
    }
    flow |= uint64_t(group_count - 1u) << 25;
    return flow;
}

uint32_t w4_tile_bytes() {
    return kLineBytes + kRowTileElems * (kTileElems / 2u);
}

uint32_t w8_tile_bytes() {
    return kLineBytes + kRowTileElems * kTileElems;
}

uint32_t w8_dense_tile_bytes(uint32_t valid_cols) {
    return kLineBytes + kRowTileElems * valid_cols;
}

uint32_t w4_weight_bytes(uint32_t rows, uint32_t cols) {
    return ceil_div(rows, kRowTileElems) * ceil_div(cols, kTileElems) *
           w4_tile_bytes();
}

uint32_t w8_weight_bytes(uint32_t rows, uint32_t cols) {
    return ceil_div(rows, kRowTileElems) * ceil_div(cols, kTileElems) *
           w8_tile_bytes();
}

uint32_t w8_dense_weight_bytes(uint32_t rows, uint32_t cols) {
    const uint32_t row_tiles = ceil_div(rows, kRowTileElems);
    const uint32_t col_tiles = ceil_div(cols, kTileElems);
    uint32_t bytes = 0;
    for (uint32_t col_tile = 0; col_tile < col_tiles; ++col_tile) {
        const uint32_t valid_cols =
            std::min<uint32_t>(kTileElems, cols - col_tile * kTileElems);
        bytes += row_tiles * w8_dense_tile_bytes(valid_cols);
    }
    return bytes;
}

uint32_t w8_dense_tile_base(uint32_t row_tile, uint32_t col_tile,
                            uint32_t cols) {
    uint32_t base = 0;
    const uint32_t col_tiles = ceil_div(cols, kTileElems);
    for (uint32_t rt = 0; rt < row_tile; ++rt) {
        for (uint32_t ct = 0; ct < col_tiles; ++ct) {
            const uint32_t valid_cols =
                std::min<uint32_t>(kTileElems, cols - ct * kTileElems);
            base += w8_dense_tile_bytes(valid_cols);
        }
    }
    for (uint32_t ct = 0; ct < col_tile; ++ct) {
        const uint32_t valid_cols =
            std::min<uint32_t>(kTileElems, cols - ct * kTileElems);
        base += w8_dense_tile_bytes(valid_cols);
    }
    return base;
}

uint32_t act_payload_bytes(uint32_t cols, uint32_t groups, bool w4_mode) {
    const uint32_t col_tiles = ceil_div(cols, kTileElems);
    const uint32_t act_bytes = groups * col_tiles * kTileElems * kFp16Bytes;
    return w4_mode ? act_bytes * 2u : act_bytes;
}

bool env_flag(const char* name) {
    const char* value = std::getenv(name);
    if (!value || !value[0]) {
        return false;
    }
    return std::strcmp(value, "0") != 0 &&
           std::strcmp(value, "false") != 0 &&
           std::strcmp(value, "FALSE") != 0;
}

bool fp16_close(uint16_t got, uint16_t expected, uint32_t ulp = 2) {
    if ((got & 0x7fffu) == 0 && (expected & 0x7fffu) == 0) {
        return true;
    }
    if ((got >> 15) != (expected >> 15)) {
        return false;
    }
    const int got_mag = static_cast<int>(got & 0x7fffu);
    const int exp_mag = static_cast<int>(expected & 0x7fffu);
    return std::abs(got_mag - exp_mag) <= static_cast<int>(ulp);
}

void fill_w4_weight_from_values(uint8_t* dst, uint32_t rows, uint32_t cols,
                                const std::vector<int8_t>& values) {
    const uint32_t row_tiles = ceil_div(rows, kRowTileElems);
    const uint32_t col_tiles = ceil_div(cols, kTileElems);
    const uint32_t tile_bytes = w4_tile_bytes();
    std::memset(dst, 0, row_tiles * col_tiles * tile_bytes);
    for (uint32_t row_tile = 0; row_tile < row_tiles; ++row_tile) {
        for (uint32_t col_tile = 0; col_tile < col_tiles; ++col_tile) {
            const uint32_t tile_base = (row_tile * col_tiles + col_tile) * tile_bytes;
            for (uint32_t lane = 0; lane < kRowTileElems; ++lane) {
                const uint32_t row = row_tile * kRowTileElems + lane;
                store_u16_le(dst + tile_base + lane * kFp16Bytes,
                             row < rows ? kFp16One : kFp16Zero);
            }
            for (uint32_t row_lane = 0; row_lane < kRowTileElems; ++row_lane) {
                const uint32_t row = row_tile * kRowTileElems + row_lane;
                const uint32_t row_base = tile_base + kLineBytes +
                                          row_lane * (kTileElems / 2u);
                for (uint32_t col_lane = 0; col_lane < kTileElems; ++col_lane) {
                    const uint32_t col = col_tile * kTileElems + col_lane;
                    const uint8_t packed =
                        (row < rows && col < cols)
                            ? static_cast<uint8_t>(values[row * cols + col]) & 0x0fu
                            : 0u;
                    uint8_t& byte = dst[row_base + col_lane / 2u];
                    if (col_lane & 1u) {
                        byte = static_cast<uint8_t>((byte & 0x0fu) | (packed << 4));
                    } else {
                        byte = static_cast<uint8_t>((byte & 0xf0u) | packed);
                    }
                }
            }
        }
    }
}

void fill_w4_zero_weight(uint8_t* dst, uint32_t rows, uint32_t cols) {
    const uint32_t row_tiles = ceil_div(rows, kRowTileElems);
    const uint32_t col_tiles = ceil_div(cols, kTileElems);
    const uint32_t tile_bytes = w4_tile_bytes();
    std::memset(dst, 0, row_tiles * col_tiles * tile_bytes);
    for (uint32_t row_tile = 0; row_tile < row_tiles; ++row_tile) {
        for (uint32_t col_tile = 0; col_tile < col_tiles; ++col_tile) {
            const uint32_t tile_base = (row_tile * col_tiles + col_tile) * tile_bytes;
            for (uint32_t lane = 0; lane < kRowTileElems; ++lane) {
                const uint32_t row = row_tile * kRowTileElems + lane;
                if (row < rows) {
                    store_u16_le(dst + tile_base + lane * kFp16Bytes, kFp16One);
                }
            }
        }
    }
}

void fill_w8_zero_weight(uint8_t* dst, uint32_t rows, uint32_t cols) {
    const uint32_t row_tiles = ceil_div(rows, kRowTileElems);
    const uint32_t col_tiles = ceil_div(cols, kTileElems);
    const uint32_t tile_bytes = w8_tile_bytes();
    std::memset(dst, 0, row_tiles * col_tiles * tile_bytes);
    for (uint32_t row_tile = 0; row_tile < row_tiles; ++row_tile) {
        for (uint32_t col_tile = 0; col_tile < col_tiles; ++col_tile) {
            const uint32_t tile_base = (row_tile * col_tiles + col_tile) * tile_bytes;
            for (uint32_t lane = 0; lane < kRowTileElems; ++lane) {
                const uint32_t row = row_tile * kRowTileElems + lane;
                if (row < rows) {
                    store_u16_le(dst + tile_base + lane * kFp16Bytes, kFp16One);
                }
            }
        }
    }
}

void fill_fp16_activation_with_scale(uint8_t* dst, uint32_t elems) {
    const uint32_t col_tiles = ceil_div(elems, kTileElems);
    const uint32_t act_bytes = col_tiles * kTileElems * kFp16Bytes;
    std::memset(dst, 0, act_bytes * 2u);
    for (uint32_t i = 0; i < elems; ++i) {
        store_u16_le(dst + i * kFp16Bytes, kFp16One);
        store_u16_le(dst + act_bytes + i * kFp16Bytes, kFp16One);
    }
}

void fill_random_sparse_activation(uint8_t* dst, uint32_t elems,
                                   uint32_t active_elems,
                                   std::vector<int16_t>* act_values,
                                   uint32_t* rng) {
    const uint32_t col_tiles = ceil_div(elems, kTileElems);
    const uint32_t act_bytes = col_tiles * kTileElems * kFp16Bytes;
    std::memset(dst, 0, act_bytes * 2u);
    act_values->assign(elems, 0);
    for (uint32_t i = 0; i < elems; ++i) {
        int16_t value = 0;
        if (i < active_elems) {
            value = (rng_next(rng) & 1u) ? 1 : -1;
        }
        (*act_values)[i] = value;
        store_u16_le(dst + i * kFp16Bytes, fp32_to_fp16_bits(float(value)));
        store_u16_le(dst + act_bytes + i * kFp16Bytes, kFp16One);
    }
}

void fill_w4_random_sparse_weight(uint8_t* dst, uint32_t rows, uint32_t cols,
                                  uint32_t active_cols, uint32_t* rng,
                                  const std::vector<int16_t>* act_values,
                                  std::vector<uint16_t>* expected) {
    std::vector<int8_t> values(size_t(rows) * cols, 0);
    if (expected) {
        expected->assign(rows, kFp16Zero);
    }
    for (uint32_t row = 0; row < rows; ++row) {
        int32_t sum = 0;
        for (uint32_t col = 0; col < std::min(active_cols, cols); ++col) {
            int8_t w = static_cast<int8_t>(int(rng_next(rng) % 5u) - 2);
            if (w == 0) {
                w = (rng_next(rng) & 1u) ? 1 : -1;
            }
            values[row * cols + col] = w;
            if (act_values) {
                sum += int32_t(w) * int32_t((*act_values)[col]);
            }
        }
        if (expected) {
            (*expected)[row] = fp32_to_fp16_bits(float(sum));
        }
    }
    fill_w4_weight_from_values(dst, rows, cols, values);
}

void fill_w4_random_single_projection(uint8_t* dst, uint32_t rows, uint32_t cols,
                                      uint32_t active_cols, uint32_t* rng,
                                      const std::vector<int16_t>& act_values,
                                      std::vector<uint16_t>* expected) {
    std::vector<int8_t> values(size_t(rows) * cols, 0);
    expected->assign(rows, kFp16Zero);
    for (uint32_t row = 0; row < rows; ++row) {
        const uint32_t col = rng_next(rng) % std::min(active_cols, cols);
        int8_t w = (rng_next(rng) & 1u) ? 1 : -1;
        if (rng_next(rng) & 1u) {
            w = static_cast<int8_t>(w * 2);
        }
        values[row * cols + col] = w;
        const int32_t product = int32_t(w) * int32_t(act_values[col]);
        (*expected)[row] = fp32_to_fp16_bits(float(product));
    }
    fill_w4_weight_from_values(dst, rows, cols, values);
}

void fill_w4_identity_weight(uint8_t* dst, uint32_t rows, uint32_t cols) {
    std::vector<int8_t> values(size_t(rows) * cols, 0);
    for (uint32_t row = 0; row < rows; ++row) {
        if (row < cols) {
            values[row * cols + row] = 1;
        }
    }
    fill_w4_weight_from_values(dst, rows, cols, values);
}

void fill_fp16_scale_tiles(uint8_t* dst, uint32_t elems, uint16_t value) {
    const uint32_t col_tiles = ceil_div(elems, kTileElems);
    const uint32_t act_bytes = col_tiles * kTileElems * kFp16Bytes;
    std::memset(dst, 0, act_bytes);
    for (uint32_t i = 0; i < elems; ++i) {
        store_u16_le(dst + i * kFp16Bytes, value);
    }
}

void fill_grouped_fp16_activation(uint8_t* dst, uint32_t elem_count,
                                  uint32_t groups, uint16_t value) {
    const uint32_t group_stride = ceil_div(elem_count, kTileElems) *
                                  kTileElems * kFp16Bytes;
    std::memset(dst, 0, group_stride * groups);
    for (uint32_t group = 0; group < groups; ++group) {
        for (uint32_t i = 0; i < elem_count; ++i) {
            store_u16_le(dst + group * group_stride + i * kFp16Bytes, value);
        }
    }
}

void fill_identity_rope_lut(uint8_t* dst, uint32_t positions) {
    constexpr uint32_t kWindowBytes = NPU_ROPE_LUT_WINDOW_BYTES;
    std::memset(dst, 0, positions * kWindowBytes);
    for (uint32_t pos = 0; pos < positions; ++pos) {
        uint8_t* row = dst + pos * kWindowBytes;
        for (uint32_t i = 0; i < kHeadDim / 2; ++i) {
            store_u16_le(row + i * 4u + 0u, kFp16One);
            store_u16_le(row + i * 4u + 2u, kFp16Zero);
        }
    }
}

void mvin(void* ptr, uint32_t spm_addr, uint32_t bytes, uint8_t input_type,
          uint8_t precision) {
    const uint32_t aligned = align_up(bytes, kMvinAlign);
    npu_dma_mvin(ptr, spm_addr, aligned - 1u, 0, 0, 0, precision, input_type,
                 false, false, false, 0, 0, 0);
}

MvinConfig make_mvin_cfg(void* ptr, uint32_t spm_addr, uint32_t bytes,
                         uint8_t input_type, uint8_t precision) {
    MvinConfig cfg = {};
    cfg.host_ptr = ptr;
    cfg.sram_addr = spm_addr;
    cfg.col_num = align_up(bytes, kMvinAlign) - 1u;
    cfg.row_num = 0;
    cfg.precision = precision;
    cfg.input_type = input_type;
    return cfg;
}

void mvin_async(void* ptr, uint32_t spm_addr, uint32_t bytes,
                uint8_t input_type, uint8_t precision) {
    const MvinConfig cfg = make_mvin_cfg(ptr, spm_addr, bytes, input_type, precision);
    npu_dma_mvin_async(0, &cfg);
}

void run_flow(const char* name, uint32_t mat_addr, uint32_t vec_addr,
              uint16_t mat_width, uint16_t mat_height, uint16_t output_addr,
              uint8_t mode, uint64_t flow) {
    const auto start = std::chrono::steady_clock::now();
    npu_matvec_decode_flow_run(mat_addr, vec_addr, mat_width, mat_height,
                               output_addr, 0, mode, flow);
    const auto end = std::chrono::steady_clock::now();
    const double ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    std::printf("[flow] %-18s mode=%u width=%u height=%u out=0x%04x %.3f ms\n",
                name, mode, mat_width, mat_height, output_addr, ms);
}

uint32_t env_u32(const char* name, uint32_t fallback) {
    const char* value = std::getenv(name);
    if (!value || !value[0]) {
        return fallback;
    }
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 0);
    return (end && *end == '\0') ? static_cast<uint32_t>(parsed) : fallback;
}

bool final_output_is_zero(uint32_t elems) {
    const uint32_t bytes = align_up(elems * kFp16Bytes, kLineBytes);
    NpuBuffer out(bytes);
    if (!out.ptr) {
        std::fprintf(stderr, "final mvout allocation failed\n");
        return false;
    }
    std::memset(out.data(), 0xa5, bytes);
    npu_dma_mvout(out.ptr, kOutputSpmBase, 0, elems - 1u, 1, 1, 1, 0,
                  false, false, 0, 0);

    uint32_t mismatches = 0;
    for (uint32_t i = 0; i < elems; ++i) {
        const uint16_t got = load_u16_le(out.data() + i * kFp16Bytes);
        if (got != kFp16Zero) {
            if (mismatches < 8) {
                std::fprintf(stderr, "final[%u] expected=0x0000 got=0x%04x\n",
                             i, got);
            }
            ++mismatches;
        }
    }
    std::printf("final O projection checked=%u mismatches=%u\n", elems, mismatches);
    return mismatches == 0;
}

void mvout_raw_output_spm(void* ptr, uint32_t spm_addr, uint32_t bytes) {
    const uint32_t aligned = align_up(bytes, kLineBytes);
    npu_dma_mvout(ptr, spm_addr, 0, (aligned / kFp16Bytes) - 1u, 1, 1, 1, 0,
                  false, false, 0, 0);
}

uint16_t cache_scale_for_token(const uint8_t* cache, uint32_t head,
                               uint32_t position) {
    const uint32_t tile = position / kRowTileElems;
    const uint32_t lane = position % kRowTileElems;
    return load_u16_le(cache + head * kKvHeadStrideBytes +
                       tile * w8_tile_bytes() + lane * kFp16Bytes);
}

uint16_t cache_scale_for_128_block(const uint8_t* cache, uint32_t block,
                                   uint32_t position) {
    const uint32_t scale_head = (block * kTileElems) / kHeadDim;
    return cache_scale_for_token(cache, scale_head, position);
}

uint16_t cache_scale_for_head(const uint8_t* cache, uint32_t head,
                              uint32_t position) {
    const uint32_t block = (head * kHeadDim) / kTileElems;
    return cache_scale_for_128_block(cache, block, position);
}

int8_t cache_quant_for_token_dim(const uint8_t* cache, uint32_t head,
                                 uint32_t position, uint32_t dim) {
    const uint32_t tile = position / kRowTileElems;
    const uint32_t lane = position % kRowTileElems;
    const uint32_t beat = dim / kLineBytes;
    const uint32_t offset = dim % kLineBytes;
    const uint32_t addr = head * kKvHeadStrideBytes +
                          tile * w8_tile_bytes() + kLineBytes +
                          lane * kTileElems + beat * kLineBytes + offset;
    return static_cast<int8_t>(cache[addr]);
}

uint32_t kv_writer_window_tiles() {
    return std::max<uint32_t>(1, kKvHeadStrideBytes / w8_tile_bytes());
}

uint32_t kv_writer_window_tokens() {
    return kv_writer_window_tiles() * kRowTileElems;
}

uint32_t kv_writer_local_position(uint32_t position) {
    return ((position / kRowTileElems) % kv_writer_window_tiles()) *
           kRowTileElems + (position % kRowTileElems);
}

uint16_t cache_scale_for_token_stride(const uint8_t* cache, uint32_t head,
                                      uint32_t head_stride, uint32_t position) {
    const uint32_t tile = position / kRowTileElems;
    const uint32_t lane = position % kRowTileElems;
    return load_u16_le(cache + head * head_stride +
                       tile * w8_tile_bytes() + lane * kFp16Bytes);
}

uint16_t cache_scale_for_head_stride(const uint8_t* cache, uint32_t head_stride,
                                     uint32_t head, uint32_t position) {
    const uint32_t block = (head * kHeadDim) / kTileElems;
    const uint32_t scale_head = (block * kTileElems) / kHeadDim;
    return cache_scale_for_token_stride(cache, scale_head, head_stride, position);
}

int8_t cache_quant_for_token_dim_stride(const uint8_t* cache, uint32_t head,
                                        uint32_t head_stride,
                                        uint32_t position, uint32_t dim) {
    const uint32_t tile = position / kRowTileElems;
    const uint32_t lane = position % kRowTileElems;
    const uint32_t beat = dim / kLineBytes;
    const uint32_t offset = dim % kLineBytes;
    const uint32_t addr = head * head_stride +
                          tile * w8_tile_bytes() + kLineBytes +
                          lane * kTileElems + beat * kLineBytes + offset;
    return static_cast<int8_t>(cache[addr]);
}

void copy_cache_token(uint8_t* dst, uint32_t dst_stride,
                      const uint8_t* src, uint32_t src_stride,
                      uint32_t head, uint32_t dst_position,
                      uint32_t src_position) {
    const uint32_t dst_tile = dst_position / kRowTileElems;
    const uint32_t dst_lane = dst_position % kRowTileElems;
    const uint32_t src_tile = src_position / kRowTileElems;
    const uint32_t src_lane = src_position % kRowTileElems;

    uint8_t* dst_scale = dst + head * dst_stride +
                         dst_tile * w8_tile_bytes() + dst_lane * kFp16Bytes;
    const uint8_t* src_scale = src + head * src_stride +
                               src_tile * w8_tile_bytes() + src_lane * kFp16Bytes;
    dst_scale[0] = src_scale[0];
    dst_scale[1] = src_scale[1];

    uint8_t* dst_row = dst + head * dst_stride +
                       dst_tile * w8_tile_bytes() + kLineBytes +
                       dst_lane * kTileElems;
    const uint8_t* src_row = src + head * src_stride +
                             src_tile * w8_tile_bytes() + kLineBytes +
                             src_lane * kTileElems;
    std::memcpy(dst_row, src_row, kHeadDim);
}

void fill_score_weight_dense_from_k_cache(uint8_t* dst, uint32_t bytes,
                                          const uint8_t* k_cache,
                                          uint32_t head_stride,
                                          uint32_t head,
                                          uint32_t seq_len) {
    std::memset(dst, 0, bytes);
    const uint32_t rows = seq_len;
    const uint32_t cols = kHeadDim;
    const uint32_t row_tiles = ceil_div(rows, kRowTileElems);
    const uint32_t col_tiles = ceil_div(cols, kTileElems);
    for (uint32_t row_tile = 0; row_tile < row_tiles; ++row_tile) {
        for (uint32_t col_tile = 0; col_tile < col_tiles; ++col_tile) {
            const uint32_t valid_cols =
                std::min<uint32_t>(kTileElems, cols - col_tile * kTileElems);
            const uint32_t tile_base =
                w8_dense_tile_base(row_tile, col_tile, cols);
            for (uint32_t lane = 0; lane < kRowTileElems; ++lane) {
                const uint32_t token = row_tile * kRowTileElems + lane;
                if (token >= rows) {
                    continue;
                }
                const uint16_t scale =
                    cache_scale_for_head_stride(k_cache, head_stride, head, token);
                store_u16_le(dst + tile_base + lane * kFp16Bytes, scale);
                uint8_t* row = dst + tile_base + kLineBytes + lane * valid_cols;
                for (uint32_t col_lane = 0; col_lane < valid_cols; ++col_lane) {
                    const uint32_t dim = col_tile * kTileElems + col_lane;
                    row[col_lane] = static_cast<uint8_t>(
                        cache_quant_for_token_dim_stride(k_cache, head,
                                                         head_stride, token, dim));
                }
            }
        }
    }
}

void fill_value_weight_dense_from_v_cache(uint8_t* dst, uint32_t bytes,
                                          const uint8_t* v_cache,
                                          uint32_t head_stride,
                                          uint32_t head,
                                          uint32_t seq_len) {
    std::memset(dst, 0, bytes);
    const uint32_t rows = kHeadDim;
    const uint32_t cols = seq_len;
    const uint32_t row_tiles = ceil_div(rows, kRowTileElems);
    const uint32_t col_tiles = ceil_div(cols, kTileElems);
    for (uint32_t row_tile = 0; row_tile < row_tiles; ++row_tile) {
        for (uint32_t col_tile = 0; col_tile < col_tiles; ++col_tile) {
            const uint32_t valid_cols =
                std::min<uint32_t>(kTileElems, cols - col_tile * kTileElems);
            const uint32_t tile_base =
                w8_dense_tile_base(row_tile, col_tile, cols);
            const uint32_t scale_token = col_tile * kTileElems;
            for (uint32_t lane = 0; lane < kRowTileElems; ++lane) {
                const uint32_t dim = row_tile * kRowTileElems + lane;
                if (dim >= rows) {
                    continue;
                }
                const uint16_t scale =
                    cache_scale_for_token_stride(v_cache, head, head_stride,
                                                 scale_token);
                store_u16_le(dst + tile_base + lane * kFp16Bytes, scale);
                uint8_t* row = dst + tile_base + kLineBytes + lane * valid_cols;
                for (uint32_t col_lane = 0; col_lane < valid_cols; ++col_lane) {
                    const uint32_t token = col_tile * kTileElems + col_lane;
                    row[col_lane] = static_cast<uint8_t>(
                        cache_quant_for_token_dim_stride(v_cache, head,
                                                         head_stride, token, dim));
                }
            }
        }
    }
}

void fill_score_weight_from_k_cache(uint8_t* dst, uint32_t bytes,
                                    const uint8_t* k_cache, uint32_t head,
                                    uint32_t position) {
    std::memset(dst, 0, bytes);
    const uint16_t scale = cache_scale_for_head(k_cache, head, position);
    store_u16_le(dst, scale);
    for (uint32_t dim = 0; dim < kHeadDim; ++dim) {
        dst[kLineBytes + dim] =
            static_cast<uint8_t>(cache_quant_for_token_dim(k_cache, head,
                                                           position, dim));
    }
}

void fill_value_weight_from_v_cache(uint8_t* dst, uint32_t bytes,
                                    const uint8_t* v_cache, uint32_t head,
                                    uint32_t position) {
    std::memset(dst, 0, bytes);
    const uint16_t scale = cache_scale_for_head(v_cache, head, position);
    for (uint32_t row = 0; row < kHeadDim; ++row) {
        const uint32_t row_tile = row / kRowTileElems;
        const uint32_t lane = row % kRowTileElems;
        const uint32_t tile_base = row_tile * w8_tile_bytes();
        store_u16_le(dst + tile_base + lane * kFp16Bytes, scale);
        dst[tile_base + kLineBytes + lane * kTileElems] =
            static_cast<uint8_t>(cache_quant_for_token_dim(v_cache, head,
                                                           position, row));
    }
}

void fill_value_weight_block_from_v_cache(uint8_t* dst, uint32_t bytes,
                                          const uint8_t* v_cache,
                                          uint32_t head_stride,
                                          uint32_t head,
                                          uint32_t token_base,
                                          uint32_t block_tokens) {
    std::memset(dst, 0, bytes);
    const uint32_t row_tiles = ceil_div(kHeadDim, kRowTileElems);
    const uint32_t col_tiles = ceil_div(block_tokens, kTileElems);
    const uint32_t tile_bytes = w8_tile_bytes();
    for (uint32_t row_tile = 0; row_tile < row_tiles; ++row_tile) {
        for (uint32_t col_tile = 0; col_tile < col_tiles; ++col_tile) {
            const uint32_t tile_base = (row_tile * col_tiles + col_tile) * tile_bytes;
            const uint32_t scale_token =
                token_base + std::min<uint32_t>(col_tile * kTileElems, block_tokens - 1u);
            for (uint32_t lane = 0; lane < kRowTileElems; ++lane) {
                const uint32_t dim = row_tile * kRowTileElems + lane;
                if (dim >= kHeadDim) {
                    continue;
                }
                const uint16_t scale =
                    cache_scale_for_token_stride(v_cache, head, head_stride, scale_token);
                store_u16_le(dst + tile_base + lane * kFp16Bytes, scale);
                uint8_t* row = dst + tile_base + kLineBytes + lane * kTileElems;
                for (uint32_t col_lane = 0; col_lane < kTileElems; ++col_lane) {
                    const uint32_t token = token_base + col_tile * kTileElems + col_lane;
                    if (token < token_base + block_tokens) {
                        row[col_lane] = static_cast<uint8_t>(
                            cache_quant_for_token_dim_stride(v_cache, head,
                                                             head_stride,
                                                             token, dim));
                    }
                }
            }
        }
    }
}

bool verify_v_cache_quant(const uint8_t* v_cache,
                          const std::vector<uint16_t>& v_expected,
                          uint32_t position,
                          std::vector<uint16_t>* dequant_expected) {
    dequant_expected->assign(kKvDim, kFp16Zero);
    uint32_t mismatches = 0;
    const uint32_t blocks = ceil_div(kKvDim, kTileElems);
    std::vector<uint16_t> block_scale(blocks, kFp16Zero);
    std::vector<uint16_t> block_max(blocks, kFp16Zero);
    for (uint32_t block = 0; block < blocks; ++block) {
        uint16_t max_abs = kFp16Zero;
        const uint32_t begin = block * kTileElems;
        const uint32_t end = std::min<uint32_t>(begin + kTileElems, kKvDim);
        for (uint32_t idx = begin; idx < end; ++idx) {
            const uint16_t mag = fp16_abs_bits(v_expected[idx]);
            if (mag > max_abs) {
                max_abs = mag;
            }
        }
        const uint16_t expected_scale = fp16_div_pow2(max_abs, 7);
        const uint16_t got_scale = cache_scale_for_128_block(v_cache, block, position);
        block_scale[block] = expected_scale;
        block_max[block] = max_abs;
        if (got_scale != expected_scale) {
            if (mismatches < 8) {
                std::fprintf(stderr,
                             "v_cache scale block%u expected=0x%04x got=0x%04x\n",
                             block, expected_scale, got_scale);
            }
            ++mismatches;
        }
    }
    for (uint32_t head = 0; head < kKvHeads; ++head) {
        const uint32_t block = (head * kHeadDim) / kTileElems;
        for (uint32_t dim = 0; dim < kHeadDim; ++dim) {
            const uint16_t value = v_expected[head * kHeadDim + dim];
            const int8_t expected_q = kv_quantize_ref(value, block_max[block]);
            const int8_t got_q = cache_quant_for_token_dim(v_cache, head, position, dim);
            if (got_q != expected_q) {
                if (mismatches < 8) {
                    std::fprintf(stderr,
                                 "v_cache q h%u d%u expected=%d got=%d value=0x%04x\n",
                                 head, dim, int(expected_q), int(got_q), value);
                }
                ++mismatches;
            }
            (*dequant_expected)[head * kHeadDim + dim] =
                fp32_to_fp16_bits(float(expected_q) * fp16_to_float(block_scale[block]));
        }
    }
    std::printf("strict V cache quant checked=%u mismatches=%u\n",
                kKvDim + kKvHeads, mismatches);
    return mismatches == 0;
}

bool final_output_matches(const std::vector<uint16_t>& expected) {
    const uint32_t elems = static_cast<uint32_t>(expected.size());
    const uint32_t bytes = align_up(elems * kFp16Bytes, kLineBytes);
    NpuBuffer out(bytes);
    if (!out.ptr) {
        std::fprintf(stderr, "final mvout allocation failed\n");
        return false;
    }
    std::memset(out.data(), 0xa5, bytes);
    npu_dma_mvout(out.ptr, kOutputSpmBase, 0, elems - 1u, 1, 1, 1, 0,
                  false, false, 0, 0);

    uint32_t mismatches = 0;
    for (uint32_t i = 0; i < elems; ++i) {
        const uint16_t got = load_u16_le(out.data() + i * kFp16Bytes);
        if (!fp16_close(got, expected[i], 2)) {
            if (mismatches < 12) {
                std::fprintf(stderr,
                             "strict final[%u] expected=0x%04x(%g) got=0x%04x(%g)\n",
                             i, expected[i], fp16_to_float(expected[i]),
                             got, fp16_to_float(got));
            }
            ++mismatches;
        }
    }
    std::printf("strict final checked=%u mismatches=%u\n", elems, mismatches);
    return mismatches == 0;
}

bool run_attention_14flow_strict_random() {
    constexpr uint32_t kSeqLen = 1;
    constexpr uint16_t kPosition = 0;
    constexpr uint32_t kActiveHidden = 32;

    uint32_t rng = env_u32("NPU_ATTN_RANDOM_SEED", 0x20260529u);
    const uint32_t qkv_act_bytes = align_up(
        act_payload_bytes(kHiddenDim, 1, true), kMvinAlign);
    const uint32_t q_weight_bytes = align_up(
        w4_weight_bytes(kHiddenDim, kHiddenDim), kMvinAlign);
    const uint32_t kv_weight_bytes = align_up(
        w4_weight_bytes(kKvDim, kHiddenDim), kMvinAlign);
    const uint32_t score_weight_bytes = align_up(
        w8_weight_bytes(kSeqLen, kHeadDim), kMvinAlign);
    const uint32_t value_weight_bytes = align_up(
        w8_weight_bytes(kHeadDim, kSeqLen), kMvinAlign);
    const uint32_t kv_cache_head_bytes = w8_tile_bytes();
    const uint32_t kv_cache_dump_bytes =
        (kKvHeads - 1u) * kKvHeadStrideBytes + kv_cache_head_bytes;
    const uint32_t q_group_act_bytes = align_up(
        act_payload_bytes(kHeadDim, kGroupSize, false), kMvinAlign);
    const uint32_t o_scale_bytes = align_up(
        ceil_div(kHiddenDim, kTileElems) * kTileElems * kFp16Bytes, kMvinAlign);
    const uint32_t rope_lut_bytes = NPU_ROPE_LUT_WINDOW_BYTES;

    NpuBuffer act(qkv_act_bytes);
    NpuBuffer w_qo(q_weight_bytes);
    NpuBuffer w_kv(kv_weight_bytes);
    NpuBuffer value_weight(value_weight_bytes);
    NpuBuffer k_cache(align_up(kv_cache_dump_bytes, kLineBytes));
    NpuBuffer v_cache(align_up(kv_cache_dump_bytes, kLineBytes));
    NpuBuffer q_group(q_group_act_bytes);
    NpuBuffer o_scale(o_scale_bytes);
    NpuBuffer rope(rope_lut_bytes);

    if (!act.ptr || !w_qo.ptr || !w_kv.ptr || !value_weight.ptr ||
        !k_cache.ptr || !v_cache.ptr || !q_group.ptr || !o_scale.ptr ||
        !rope.ptr) {
        std::fprintf(stderr, "npu_mem_alloc failed for strict attention buffers\n");
        return false;
    }

    std::printf("kv260_decode_attention_14flow_test strict_random: "
                "hidden=%u q=%u kv=%u heads=%u/%u group=%u head_dim=%u "
                "seq_len=%u seed=0x%08x\n",
                kHiddenDim, kHiddenDim, kKvDim, kQHeads, kKvHeads, kGroupSize,
                kHeadDim, kSeqLen, rng);

    std::vector<int16_t> hidden_values;
    std::vector<uint16_t> v_expected;
    std::vector<uint16_t> v_dequant_expected;
    const auto cpu_begin = std::chrono::steady_clock::now();
    fill_random_sparse_activation(act.data(), kHiddenDim, kActiveHidden,
                                  &hidden_values, &rng);
    fill_w4_random_sparse_weight(w_kv.data(), kKvDim, kHiddenDim,
                                 kActiveHidden, &rng, nullptr, nullptr);
    fill_grouped_fp16_activation(q_group.data(), kHeadDim, kGroupSize, kFp16One);
    fill_fp16_scale_tiles(o_scale.data(), kHiddenDim, kFp16One);
    fill_identity_rope_lut(rope.data(), 1);
    const auto cpu_end = std::chrono::steady_clock::now();
    std::printf("[cpu] strict random setup %.3f ms\n",
                std::chrono::duration<double, std::milli>(cpu_end - cpu_begin).count());

    npu_reset();
    mvin(act.ptr, kHiddenActBase, qkv_act_bytes, NPU_GEMV_MVIN_ACT, 2);
    npu_rope_lut_mvin(rope.ptr, kPosition);

    mvin(w_kv.ptr, kInputSpmBase, kv_weight_bytes, NPU_GEMV_MVIN_WEIGHT, 1);
    run_flow("K_proj_rope_kv", kInputSpmBase, kHiddenActBase, kHiddenDim, kKvDim,
             kOutputSpmBase, NPU_GEMV_MODE_W4A16,
             decode_flow(NPU_DECODE_SRC_GEMV_STREAM, 0, NPU_DECODE_UNARY_ROPE,
                         NPU_DECODE_BINARY_BYPASS, NPU_DECODE_REDUCE_BYPASS,
                         NPU_DECODE_DST_OUTPUT_SPM, kKvDim, kPosition, 1, true));
    mvout_raw_output_spm(k_cache.ptr, kOutputSpmBase, kv_cache_dump_bytes);

    const auto v_cpu_begin = std::chrono::steady_clock::now();
    fill_w4_random_single_projection(w_kv.data(), kKvDim, kHiddenDim,
                                     kActiveHidden, &rng, hidden_values,
                                     &v_expected);
    const auto v_cpu_end = std::chrono::steady_clock::now();
    std::printf("[cpu] V projection golden %.3f ms\n",
                std::chrono::duration<double, std::milli>(v_cpu_end - v_cpu_begin).count());
    mvin(w_kv.ptr, kInputSpmBase, kv_weight_bytes, NPU_GEMV_MVIN_WEIGHT, 1);
    run_flow("V_proj_kv", kInputSpmBase, kHiddenActBase, kHiddenDim, kKvDim,
             kOutputSpmBase, NPU_GEMV_MODE_W4A16,
             decode_flow(NPU_DECODE_SRC_GEMV_STREAM, 0, NPU_DECODE_UNARY_BYPASS,
                         NPU_DECODE_BINARY_BYPASS, NPU_DECODE_REDUCE_BYPASS,
                         NPU_DECODE_DST_OUTPUT_SPM, kKvDim, kPosition, 1, true));
    mvout_raw_output_spm(v_cache.ptr, kOutputSpmBase, kv_cache_dump_bytes);

    const bool cache_ok = verify_v_cache_quant(v_cache.data(), v_expected,
                                               kPosition, &v_dequant_expected);

    fill_w4_random_sparse_weight(w_qo.data(), kHiddenDim, kHiddenDim,
                                 kActiveHidden, &rng, nullptr, nullptr);
    mvin(w_qo.ptr, kInputSpmBase, q_weight_bytes, NPU_GEMV_MVIN_WEIGHT, 1);
    run_flow("Q_proj_rope", kInputSpmBase, kHiddenActBase, kHiddenDim, kHiddenDim,
             kQGroupActBase, NPU_GEMV_MODE_W4A16,
             decode_flow(NPU_DECODE_SRC_GEMV_STREAM, 0, NPU_DECODE_UNARY_ROPE,
                         NPU_DECODE_BINARY_BYPASS, NPU_DECODE_REDUCE_BYPASS,
                         NPU_DECODE_DST_ACT_BUFFER, kHiddenDim, kPosition));

    uint32_t flow_count = 3;
    for (uint32_t kv_head = 0; kv_head < kKvHeads; ++kv_head) {
        mvin(q_group.ptr, kQGroupActBase, q_group_act_bytes, NPU_GEMV_MVIN_ACT, 2);
        fill_score_weight_from_k_cache(value_weight.data(), value_weight_bytes,
                                       k_cache.data(), kv_head, kPosition);
        mvin(value_weight.ptr, kInputSpmBase, score_weight_bytes,
             NPU_GEMV_MVIN_WEIGHT, 1);
        char score_name[32];
        std::snprintf(score_name, sizeof(score_name), "score_softmax_h%u", kv_head);
        run_flow(score_name, kInputSpmBase, kQGroupActBase, kHeadDim,
                 kSeqLen, kScoreActBase, NPU_GEMV_MODE_W8A16,
                 decode_flow(NPU_DECODE_SRC_GEMV_STREAM, 0,
                             NPU_DECODE_UNARY_BYPASS,
                             NPU_DECODE_BINARY_BYPASS,
                             NPU_DECODE_REDUCE_SOFTMAX,
                             NPU_DECODE_DST_ACT_BUFFER,
                             kSeqLen, kPosition, kGroupSize));
        ++flow_count;

        fill_value_weight_from_v_cache(value_weight.data(), value_weight_bytes,
                                       v_cache.data(), kv_head, kPosition);
        mvin(value_weight.ptr, kInputSpmBase, value_weight_bytes,
             NPU_GEMV_MVIN_WEIGHT, 1);
        char value_name[32];
        std::snprintf(value_name, sizeof(value_name), "value_gemv_h%u", kv_head);
        const uint16_t out_addr = static_cast<uint16_t>(
            kAttentionActBase + kv_head * kGroupSize * kHeadDim * kFp16Bytes);
        run_flow(value_name, kInputSpmBase, kScoreActBase,
                 kSeqLen, kHeadDim, out_addr, NPU_GEMV_MODE_W8A16,
                 decode_flow(NPU_DECODE_SRC_GEMV_STREAM, 0,
                             NPU_DECODE_UNARY_BYPASS,
                             NPU_DECODE_BINARY_BYPASS,
                             NPU_DECODE_REDUCE_BYPASS,
                             NPU_DECODE_DST_ACT_BUFFER,
                             kHeadDim, kPosition, kGroupSize));
        ++flow_count;
    }

    const uint32_t o_act_payload_bytes =
        ceil_div(kHiddenDim, kTileElems) * kTileElems * kFp16Bytes;
    mvin(o_scale.ptr, kAttentionActBase + o_act_payload_bytes, o_scale_bytes,
         NPU_GEMV_MVIN_ACT, 2);
    fill_w4_identity_weight(w_qo.data(), kHiddenDim, kHiddenDim);
    mvin(w_qo.ptr, kInputSpmBase, q_weight_bytes, NPU_GEMV_MVIN_WEIGHT, 1);
    run_flow("O_proj", kInputSpmBase, kAttentionActBase, kHiddenDim, kHiddenDim,
             kOutputSpmBase, NPU_GEMV_MODE_W4A16,
             decode_flow(NPU_DECODE_SRC_GEMV_STREAM, 0, NPU_DECODE_UNARY_BYPASS,
                         NPU_DECODE_BINARY_BYPASS, NPU_DECODE_REDUCE_BYPASS,
                         NPU_DECODE_DST_OUTPUT_SPM, kHiddenDim, kPosition));
    ++flow_count;

    std::vector<uint16_t> final_expected(kHiddenDim, kFp16Zero);
    for (uint32_t head = 0; head < kKvHeads; ++head) {
        for (uint32_t group = 0; group < kGroupSize; ++group) {
            for (uint32_t dim = 0; dim < kHeadDim; ++dim) {
                const uint32_t out_idx =
                    head * kGroupSize * kHeadDim + group * kHeadDim + dim;
                final_expected[out_idx] = v_dequant_expected[head * kHeadDim + dim];
            }
        }
    }

    const bool final_ok = final_output_matches(final_expected);
    std::printf("decode_attention strict flow_count=%u expected=14\n", flow_count);
    return flow_count == 14 && cache_ok && final_ok;
}

bool run_attention_14flow() {
    const uint32_t seq_len = std::max<uint32_t>(
        1, std::min<uint32_t>(env_u32("NPU_ATTN_SEQ_LEN", 4096), 4096));
    const uint16_t position = static_cast<uint16_t>(
        std::min<uint32_t>(env_u32("NPU_ATTN_POSITION", seq_len - 1u), 4095));

    const uint32_t qkv_act_bytes = align_up(
        act_payload_bytes(kHiddenDim, 1, true), kMvinAlign);
    const uint32_t q_weight_bytes = align_up(
        w4_weight_bytes(kHiddenDim, kHiddenDim), kMvinAlign);
    const uint32_t kv_weight_bytes = align_up(
        w4_weight_bytes(kKvDim, kHiddenDim), kMvinAlign);
    const uint32_t score_weight_bytes = align_up(
        w8_dense_weight_bytes(seq_len, kHeadDim), kMvinAlign);
    const uint32_t value_weight_bytes = align_up(
        w8_dense_weight_bytes(kHeadDim, seq_len), kMvinAlign);
    const uint32_t kv_cache_head_bytes =
        ceil_div(seq_len, kRowTileElems) * w8_tile_bytes();
    const uint32_t kv_cache_dump_head_bytes =
        kv_writer_window_tiles() * w8_tile_bytes();
    const uint32_t kv_cache_dump_bytes =
        (kKvHeads - 1u) * kKvHeadStrideBytes + kv_cache_dump_head_bytes;
    const uint32_t kv_external_stride = kv_cache_head_bytes;
    const uint32_t kv_external_bytes = kKvHeads * kv_external_stride;
    const uint32_t o_act_payload_bytes =
        ceil_div(kHiddenDim, kTileElems) * kTileElems * kFp16Bytes;
    const uint32_t o_scale_bytes = align_up(o_act_payload_bytes, kMvinAlign);
    const uint32_t rope_windows = ((uint32_t(position) & ~1u) >> 1) + 1u;
    const uint32_t rope_lut_bytes = rope_windows * NPU_ROPE_LUT_WINDOW_BYTES;

    if (score_weight_bytes > NPU_GEMV_SPM_BYTES ||
        value_weight_bytes > NPU_GEMV_SPM_BYTES) {
        std::fprintf(stderr,
                     "unsupported seq_len=%u: score_stage=%u value_stage=%u "
                     "spm_capacity=%u\n",
                     seq_len, score_weight_bytes, value_weight_bytes,
                     NPU_GEMV_SPM_BYTES);
        return false;
    }

    NpuBuffer act(qkv_act_bytes);
    NpuBuffer w_q(q_weight_bytes);
    NpuBuffer w_kv(kv_weight_bytes);
    NpuBuffer kv_window(align_up(kv_cache_dump_bytes, kLineBytes));
    NpuBuffer k_cache(align_up(kv_external_bytes, kLineBytes));
    NpuBuffer v_cache(align_up(kv_external_bytes, kLineBytes));
    NpuBuffer score_stage(score_weight_bytes);
    NpuBuffer value_stage(value_weight_bytes);
    NpuBuffer o_scale(o_scale_bytes);
    NpuBuffer rope(rope_lut_bytes);

    if (!act.ptr || !w_q.ptr || !w_kv.ptr ||
        !kv_window.ptr || !k_cache.ptr || !v_cache.ptr ||
        !score_stage.ptr || !value_stage.ptr || !o_scale.ptr || !rope.ptr) {
        std::fprintf(stderr, "npu_mem_alloc failed for attention buffers\n");
        return false;
    }

    std::printf("kv260_decode_attention_14flow_test: hidden=%u q=%u kv=%u "
                "heads=%u/%u group=%u head_dim=%u seq_len=%u position=%u "
                "score_weight=%u value_weight=%u window_tokens=%u\n",
                kHiddenDim, kHiddenDim, kKvDim, kQHeads, kKvHeads, kGroupSize,
                kHeadDim, seq_len, position, score_weight_bytes,
                value_weight_bytes, kv_writer_window_tokens());

    fill_fp16_activation_with_scale(act.data(), kHiddenDim);
    fill_w4_zero_weight(w_q.data(), kHiddenDim, kHiddenDim);
    fill_w4_zero_weight(w_kv.data(), kKvDim, kHiddenDim);
    fill_fp16_scale_tiles(o_scale.data(), kHiddenDim, kFp16One);
    fill_identity_rope_lut(rope.data(), rope_windows);
    std::memset(k_cache.data(), 0, k_cache.bytes);
    std::memset(v_cache.data(), 0, v_cache.bytes);

    npu_reset();
    mvin(act.ptr, kHiddenActBase, qkv_act_bytes, NPU_GEMV_MVIN_ACT, 2);
    npu_rope_lut_mvin(rope.ptr, position);

    mvin(w_kv.ptr, kInputSpmBase, kv_weight_bytes, NPU_GEMV_MVIN_WEIGHT, 1);
    run_flow("K_proj_rope_kv", kInputSpmBase, kHiddenActBase, kHiddenDim, kKvDim,
             kOutputSpmBase, NPU_GEMV_MODE_W4A16,
             decode_flow(NPU_DECODE_SRC_GEMV_STREAM, 0, NPU_DECODE_UNARY_ROPE,
                         NPU_DECODE_BINARY_BYPASS, NPU_DECODE_REDUCE_BYPASS,
                         NPU_DECODE_DST_OUTPUT_SPM, kKvDim, position, 1, true));
    std::memset(kv_window.data(), 0, kv_window.bytes);
    mvout_raw_output_spm(kv_window.ptr, kOutputSpmBase, kv_cache_dump_bytes);
    for (uint32_t head = 0; head < kKvHeads; ++head) {
        copy_cache_token(k_cache.data(), kv_external_stride,
                         kv_window.data(), kKvHeadStrideBytes,
                         head, position, kv_writer_local_position(position));
    }

    mvin(w_kv.ptr, kInputSpmBase, kv_weight_bytes, NPU_GEMV_MVIN_WEIGHT, 1);
    run_flow("V_proj_kv", kInputSpmBase, kHiddenActBase, kHiddenDim, kKvDim,
             kOutputSpmBase, NPU_GEMV_MODE_W4A16,
             decode_flow(NPU_DECODE_SRC_GEMV_STREAM, 0, NPU_DECODE_UNARY_BYPASS,
                         NPU_DECODE_BINARY_BYPASS, NPU_DECODE_REDUCE_BYPASS,
                         NPU_DECODE_DST_OUTPUT_SPM, kKvDim, position, 1, true));
    std::memset(kv_window.data(), 0, kv_window.bytes);
    mvout_raw_output_spm(kv_window.ptr, kOutputSpmBase, kv_cache_dump_bytes);
    for (uint32_t head = 0; head < kKvHeads; ++head) {
        copy_cache_token(v_cache.data(), kv_external_stride,
                         kv_window.data(), kKvHeadStrideBytes,
                         head, position, kv_writer_local_position(position));
    }

    mvin(w_q.ptr, kInputSpmBase, q_weight_bytes, NPU_GEMV_MVIN_WEIGHT, 1);
    run_flow("Q_proj_rope", kInputSpmBase, kHiddenActBase, kHiddenDim, kHiddenDim,
             kQGroupActBase, NPU_GEMV_MODE_W4A16,
             decode_flow(NPU_DECODE_SRC_GEMV_STREAM, 0, NPU_DECODE_UNARY_ROPE,
                         NPU_DECODE_BINARY_BYPASS, NPU_DECODE_REDUCE_BYPASS,
                         NPU_DECODE_DST_ACT_BUFFER, kHiddenDim, position));

    uint32_t logical_flow_count = 3;
    uint32_t matvec_cmd_count = 3;
    for (uint32_t kv_head = 0; kv_head < kKvHeads; ++kv_head) {
        fill_score_weight_dense_from_k_cache(score_stage.data(), score_weight_bytes,
                                             k_cache.data(), kv_external_stride,
                                             kv_head, seq_len);
        mvin(score_stage.ptr, kInputSpmBase, score_weight_bytes,
             NPU_GEMV_MVIN_WEIGHT, 1);

        char score_name[32];
        std::snprintf(score_name, sizeof(score_name), "score_softmax_h%u", kv_head);
        const uint32_t q_vec_addr =
            kQGroupActBase + kv_head * kGroupSize * kHeadDim * kFp16Bytes;
        run_flow(score_name, kInputSpmBase, q_vec_addr, kHeadDim,
                 static_cast<uint16_t>(seq_len), kScoreActBase,
                 NPU_GEMV_MODE_W8A16,
                 decode_flow(NPU_DECODE_SRC_GEMV_STREAM, 0,
                             NPU_DECODE_UNARY_BYPASS,
                             NPU_DECODE_BINARY_BYPASS,
                             NPU_DECODE_REDUCE_SOFTMAX,
                             NPU_DECODE_DST_ACT_BUFFER,
                             static_cast<uint16_t>(seq_len), position,
                             kGroupSize));
        ++matvec_cmd_count;

        fill_value_weight_dense_from_v_cache(value_stage.data(), value_weight_bytes,
                                             v_cache.data(), kv_external_stride,
                                             kv_head, seq_len);
        mvin(value_stage.ptr, kInputSpmBase, value_weight_bytes,
             NPU_GEMV_MVIN_WEIGHT, 1);

        char value_name[32];
        std::snprintf(value_name, sizeof(value_name), "value_gemv_h%u", kv_head);
        const uint16_t out_addr = static_cast<uint16_t>(
            kAttentionActBase + kv_head * kGroupSize * kHeadDim * kFp16Bytes);
        run_flow(value_name, kInputSpmBase, kScoreActBase,
                 static_cast<uint16_t>(seq_len), kHeadDim, out_addr,
                 NPU_GEMV_MODE_W8A16,
                 decode_flow(NPU_DECODE_SRC_GEMV_STREAM, 0,
                             NPU_DECODE_UNARY_BYPASS,
                             NPU_DECODE_BINARY_BYPASS,
                             NPU_DECODE_REDUCE_BYPASS,
                             NPU_DECODE_DST_ACT_BUFFER,
                             kHeadDim, position, kGroupSize));
        ++matvec_cmd_count;
        logical_flow_count += 2;
    }

    mvin(o_scale.ptr, kAttentionActBase + o_act_payload_bytes, o_scale_bytes,
         NPU_GEMV_MVIN_ACT, 2);
    mvin(w_q.ptr, kInputSpmBase, q_weight_bytes, NPU_GEMV_MVIN_WEIGHT, 1);
    run_flow("O_proj", kInputSpmBase, kAttentionActBase, kHiddenDim, kHiddenDim,
             kOutputSpmBase, NPU_GEMV_MODE_W4A16,
             decode_flow(NPU_DECODE_SRC_GEMV_STREAM, 0, NPU_DECODE_UNARY_BYPASS,
                         NPU_DECODE_BINARY_BYPASS, NPU_DECODE_REDUCE_BYPASS,
                         NPU_DECODE_DST_OUTPUT_SPM, kHiddenDim, position));
    ++logical_flow_count;
    ++matvec_cmd_count;

    std::printf("decode_attention logical_flow_count=%u expected=14 "
                "matvec_cmd_count=%u seq_len=%u\n",
                logical_flow_count, matvec_cmd_count, seq_len);
    return logical_flow_count == 14 && final_output_is_zero(kHiddenDim);
}

} // namespace

int main() {
    if (npu_init() != 0) {
        std::perror("npu_init");
        return 1;
    }

    const bool ok = env_flag("NPU_ATTN_STRICT_RANDOM")
        ? run_attention_14flow_strict_random()
        : run_attention_14flow();
    npu_destroy();
    std::puts(ok ? "kv260_decode_attention_14flow_test=ok"
                 : "kv260_decode_attention_14flow_test=fail");
    return ok ? 0 : 2;
}
