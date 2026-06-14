#include "npu_runtime.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kFp16Bytes = NPU_GEMV_FP16_BYTES;
constexpr uint32_t kMvinAlign = NPU_GEMV_MVIN_ALIGN_BYTES;
constexpr uint32_t kRowTileElems = NPU_GEMV_ROW_TILE_ELEMS;
constexpr uint32_t kTileElems = NPU_GEMV_TILE_ELEMS;
constexpr uint32_t kW8TileElems = NPU_GEMV_TILE_ELEMS / 2u;
constexpr uint32_t kLineBytes = NPU_GEMV_LINE_BYTES;
constexpr uint32_t kDecodeHeadDim = 64;
constexpr uint32_t kDecodeKvDim = 320;
constexpr uint16_t kFp16Zero = 0x0000;
constexpr uint16_t kFp16One = 0x3c00;

uint32_t ceil_div(uint32_t a, uint32_t b) {
    return (a + b - 1u) / b;
}

uint32_t align_up(uint32_t value, uint32_t alignment) {
    return (value + alignment - 1u) & ~(alignment - 1u);
}

void store_u16_le(uint8_t* p, uint16_t v) {
    p[0] = static_cast<uint8_t>(v & 0xffu);
    p[1] = static_cast<uint8_t>(v >> 8);
}

uint16_t load_u16_le(const uint8_t* p) {
    return static_cast<uint16_t>(p[0]) |
           static_cast<uint16_t>(static_cast<uint16_t>(p[1]) << 8);
}

uint32_t load_u32_le(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) |
           (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

uint16_t fp32_to_fp16_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;

    if (exp <= 0) {
        if (exp < -10) {
            return static_cast<uint16_t>(sign);
        }
        mant |= 0x800000u;
        const uint32_t shift = static_cast<uint32_t>(14 - exp);
        uint32_t half_mant = mant >> shift;
        if ((mant >> (shift - 1u)) & 1u) {
            ++half_mant;
        }
        return static_cast<uint16_t>(sign | half_mant);
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00u);
    }

    uint32_t half = sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13);
    if (mant & 0x00001000u) {
        ++half;
    }
    return static_cast<uint16_t>(half);
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
        const float value = std::ldexp(static_cast<float>(frac), -24);
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

uint32_t w4_weight_bytes(uint32_t rows, uint32_t cols) {
    return ceil_div(rows, kRowTileElems) *
           ceil_div(cols, kTileElems) *
           kRowTileElems * (kTileElems / 2u);
}

uint32_t w8_weight_bytes(uint32_t rows, uint32_t cols) {
    return ceil_div(rows, kRowTileElems) *
           ceil_div(cols, kW8TileElems) *
           kRowTileElems * kW8TileElems;
}

uint32_t weight_scale_bytes(uint32_t rows, uint32_t cols, bool w8_mode) {
    const uint32_t elems_per_tile = w8_mode ? kW8TileElems : kTileElems;
    return ceil_div(rows, kRowTileElems) *
           ceil_div(cols, elems_per_tile) *
           kLineBytes;
}

uint32_t act_payload_bytes(uint32_t cols, bool w8_mode) {
    const uint32_t elems_per_tile = w8_mode ? kW8TileElems : kTileElems;
    return ceil_div(cols, elems_per_tile) * elems_per_tile * kFp16Bytes;
}

struct DeterministicRng {
    uint32_t state;

    explicit DeterministicRng(uint32_t seed) : state(seed ? seed : 0x6d2b79f5u) {}

    uint32_t next() {
        uint32_t x = state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        state = x ? x : 0x6d2b79f5u;
        return state;
    }

    uint32_t range(uint32_t limit) {
        return limit ? (next() % limit) : 0u;
    }
};

int8_t sign_extend_i4(uint8_t nibble) {
    nibble &= 0x0fu;
    return static_cast<int8_t>((nibble & 0x08u) ?
                               static_cast<int>(nibble) - 16 :
                               static_cast<int>(nibble));
}

uint8_t pack_i4(int8_t value) {
    return static_cast<uint8_t>(value) & 0x0fu;
}

float random_activation_value(DeterministicRng& rng) {
    static const float values[] = {
        -1.0f, -0.75f, -0.5f, -0.25f,
         0.25f, 0.5f, 0.75f, 1.0f,
    };
    return values[rng.range(sizeof(values) / sizeof(values[0]))];
}

float random_act_scale_value(DeterministicRng& rng) {
    static const float values[] = {
        0.5f, 0.625f, 0.75f, 0.875f, 1.0f, 1.125f,
    };
    return values[rng.range(sizeof(values) / sizeof(values[0]))];
}

float random_weight_scale_value(DeterministicRng& rng) {
    static const float values[] = {
        0.125f, 0.1875f, 0.25f, 0.3125f, 0.375f, 0.5f,
    };
    return values[rng.range(sizeof(values) / sizeof(values[0]))];
}

int8_t random_w4_weight_value(DeterministicRng& rng) {
    return static_cast<int8_t>(static_cast<int>(rng.range(9u)) - 4);
}

int8_t random_w8_weight_value(DeterministicRng& rng) {
    static const int8_t values[] = {
        -5, -4, -3, -2, -1, 1, 2, 3, 4, 5,
    };
    return values[rng.range(sizeof(values) / sizeof(values[0]))];
}

float random_prob_value(DeterministicRng& rng) {
    static const float values[] = {
        0.0625f, 0.125f, 0.1875f, 0.25f,
        0.3125f, 0.375f, 0.4375f, 0.5f,
    };
    return values[rng.range(sizeof(values) / sizeof(values[0]))];
}

void fill_w4_random_activation(uint8_t* dst, uint32_t cols, uint32_t seed) {
    std::memset(dst, 0, align_up(act_payload_bytes(cols, false), kMvinAlign));
    DeterministicRng rng(seed);
    for (uint32_t col = 0; col < cols; ++col) {
        store_u16_le(dst + col * kFp16Bytes,
                     fp32_to_fp16_bits(random_activation_value(rng)));
    }
}

void fill_w4_random_act_scale(uint8_t* dst, uint32_t cols, uint32_t seed) {
    std::memset(dst, 0, align_up(act_payload_bytes(cols, false), kMvinAlign));
    DeterministicRng rng(seed);
    for (uint32_t col = 0; col < cols; ++col) {
        store_u16_le(dst + col * kFp16Bytes,
                     fp32_to_fp16_bits(random_act_scale_value(rng)));
    }
}

void fill_w4_random_weight(uint8_t* dst, uint32_t rows, uint32_t cols,
                           uint32_t seed) {
    const uint32_t row_tiles = ceil_div(rows, kRowTileElems);
    const uint32_t col_tiles = ceil_div(cols, kTileElems);
    const uint32_t row_bytes = kTileElems / 2u;
    const uint32_t tile_bytes = kRowTileElems * row_bytes;
    std::memset(dst, 0, row_tiles * col_tiles * tile_bytes);

    DeterministicRng rng(seed);
    for (uint32_t row = 0; row < rows; ++row) {
        const uint32_t row_tile = row / kRowTileElems;
        const uint32_t row_lane = row % kRowTileElems;
        for (uint32_t col = 0; col < cols; ++col) {
            const uint32_t col_tile = col / kTileElems;
            const uint32_t col_lane = col % kTileElems;
            const uint32_t tile_base =
                (row_tile * col_tiles + col_tile) * tile_bytes;
            uint8_t& packed = dst[tile_base + row_lane * row_bytes + col_lane / 2u];
            const uint8_t nibble = pack_i4(random_w4_weight_value(rng));
            if (col_lane & 1u) {
                packed = static_cast<uint8_t>((packed & 0x0fu) | (nibble << 4));
            } else {
                packed = static_cast<uint8_t>((packed & 0xf0u) | nibble);
            }
        }
    }
}

void fill_w4_random_weight_scale(uint8_t* dst, uint32_t rows, uint32_t cols,
                                 uint32_t seed) {
    const uint32_t row_tiles = ceil_div(rows, kRowTileElems);
    const uint32_t col_tiles = ceil_div(cols, kTileElems);
    std::memset(dst, 0, weight_scale_bytes(rows, cols, false));
    DeterministicRng rng(seed);
    for (uint32_t row_tile = 0; row_tile < row_tiles; ++row_tile) {
        for (uint32_t col_tile = 0; col_tile < col_tiles; ++col_tile) {
            const uint32_t tile_base =
                (row_tile * col_tiles + col_tile) * kLineBytes;
            for (uint32_t lane = 0; lane < kRowTileElems; ++lane) {
                if (row_tile * kRowTileElems + lane < rows) {
                    const uint16_t scale_bits =
                        fp32_to_fp16_bits(random_weight_scale_value(rng));
                    store_u16_le(dst + tile_base + lane * kFp16Bytes, scale_bits);
                }
            }
        }
    }
}

void fill_w8_random_activation(uint8_t* dst, uint32_t cols, uint32_t seed,
                               bool positive_only) {
    std::memset(dst, 0, align_up(act_payload_bytes(cols, true), kMvinAlign));
    DeterministicRng rng(seed);
    for (uint32_t col = 0; col < cols; ++col) {
        const float value = positive_only ?
            random_prob_value(rng) : random_activation_value(rng);
        store_u16_le(dst + col * kFp16Bytes, fp32_to_fp16_bits(value));
    }
}

void fill_w8_random_group_activation(uint8_t* dst, uint32_t cols,
                                     uint32_t group_count,
                                     uint32_t act_group_stride_bytes,
                                     uint32_t seed, bool positive_only) {
    const uint32_t act_bytes = act_payload_bytes(cols, true);
    const uint32_t span =
        (group_count <= 1u) ? act_bytes :
        (group_count - 1u) * act_group_stride_bytes + act_bytes;
    std::memset(dst, 0, align_up(span, kMvinAlign));
    for (uint32_t group = 0; group < group_count; ++group) {
        fill_w8_random_activation(dst + group * act_group_stride_bytes, cols,
                                  seed ^ (0x9e3779b9u * (group + 1u)),
                                  positive_only);
    }
}

void fill_w8_random_weight(uint8_t* dst, uint32_t rows, uint32_t cols,
                           uint32_t seed) {
    const uint32_t col_tiles = ceil_div(cols, kW8TileElems);
    const uint32_t tile_bytes = kRowTileElems * kW8TileElems;
    std::memset(dst, 0, w8_weight_bytes(rows, cols));

    DeterministicRng rng(seed);
    for (uint32_t row = 0; row < rows; ++row) {
        const uint32_t row_tile = row / kRowTileElems;
        const uint32_t row_lane = row % kRowTileElems;
        for (uint32_t col = 0; col < cols; ++col) {
            const uint32_t col_tile = col / kW8TileElems;
            const uint32_t col_lane = col % kW8TileElems;
            const uint32_t tile_base =
                (row_tile * col_tiles + col_tile) * tile_bytes;
            dst[tile_base + row_lane * kW8TileElems + col_lane] =
                static_cast<uint8_t>(random_w8_weight_value(rng));
        }
    }
}

void fill_w8_random_fixed_stride_weight(uint8_t* dst, uint32_t rows,
                                        uint32_t capacity,
                                        uint32_t row_tile_stride_bytes,
                                        uint32_t seed) {
    const uint32_t row_tiles = ceil_div(rows, kRowTileElems);
    const uint32_t col_tiles = ceil_div(capacity, kW8TileElems);
    const uint32_t tile_bytes = kRowTileElems * kW8TileElems;
    std::memset(dst, 0, row_tiles * row_tile_stride_bytes);

    DeterministicRng rng(seed);
    for (uint32_t row_tile = 0; row_tile < row_tiles; ++row_tile) {
        for (uint32_t col_tile = 0; col_tile < col_tiles; ++col_tile) {
            const uint32_t tile_base =
                row_tile * row_tile_stride_bytes + col_tile * tile_bytes;
            for (uint32_t row_lane = 0; row_lane < kRowTileElems; ++row_lane) {
                const uint32_t row = row_tile * kRowTileElems + row_lane;
                if (row >= rows) {
                    continue;
                }
                uint8_t* row_ptr = dst + tile_base + row_lane * kW8TileElems;
                for (uint32_t col_lane = 0; col_lane < kW8TileElems; ++col_lane) {
                    const uint32_t token = col_tile * kW8TileElems + col_lane;
                    if (token < capacity) {
                        row_ptr[col_lane] =
                            static_cast<uint8_t>(random_w8_weight_value(rng));
                    }
                }
            }
        }
    }
}

void fill_w8_random_weight_scale(uint8_t* dst, uint32_t rows, uint32_t cols,
                                 uint32_t seed) {
    const uint32_t row_tiles = ceil_div(rows, kRowTileElems);
    const uint32_t col_tiles = ceil_div(cols, kW8TileElems);
    std::memset(dst, 0, weight_scale_bytes(rows, cols, true));
    DeterministicRng rng(seed);
    for (uint32_t row_tile = 0; row_tile < row_tiles; ++row_tile) {
        for (uint32_t col_tile = 0; col_tile < col_tiles; ++col_tile) {
            const uint32_t tile_base =
                (row_tile * col_tiles + col_tile) * kLineBytes;
            for (uint32_t lane = 0; lane < kRowTileElems; ++lane) {
                if (row_tile * kRowTileElems + lane < rows) {
                    const uint16_t scale_bits =
                        fp32_to_fp16_bits(random_weight_scale_value(rng));
                    store_u16_le(dst + tile_base + lane * kFp16Bytes, scale_bits);
                }
            }
        }
    }
}

float load_fp16_value(const uint8_t* base, uint32_t elem) {
    return fp16_to_float(load_u16_le(base + elem * kFp16Bytes));
}

float load_w4_weight_value(const uint8_t* weight, uint32_t row, uint32_t col,
                           uint32_t cols) {
    const uint32_t col_tiles = ceil_div(cols, kTileElems);
    const uint32_t row_bytes = kTileElems / 2u;
    const uint32_t tile_bytes = kRowTileElems * row_bytes;
    const uint32_t row_tile = row / kRowTileElems;
    const uint32_t row_lane = row % kRowTileElems;
    const uint32_t col_tile = col / kTileElems;
    const uint32_t col_lane = col % kTileElems;
    const uint32_t tile_base =
        (row_tile * col_tiles + col_tile) * tile_bytes;
    const uint8_t packed =
        weight[tile_base + row_lane * row_bytes + col_lane / 2u];
    const uint8_t nibble = (col_lane & 1u) ?
        static_cast<uint8_t>(packed >> 4) :
        static_cast<uint8_t>(packed & 0x0fu);
    return static_cast<float>(sign_extend_i4(nibble));
}

float load_w4_weight_scale_value(const uint8_t* scale, uint32_t row,
                                 uint32_t col_tile, uint32_t cols) {
    const uint32_t col_tiles = ceil_div(cols, kTileElems);
    const uint32_t row_tile = row / kRowTileElems;
    const uint32_t row_lane = row % kRowTileElems;
    const uint32_t tile_base =
        (row_tile * col_tiles + col_tile) * kLineBytes;
    return fp16_to_float(load_u16_le(scale + tile_base +
                                    row_lane * kFp16Bytes));
}

float load_w8_weight_value(const uint8_t* weight, uint32_t row, uint32_t col,
                           uint32_t cols) {
    const uint32_t col_tiles = ceil_div(cols, kW8TileElems);
    const uint32_t tile_bytes = kRowTileElems * kW8TileElems;
    const uint32_t row_tile = row / kRowTileElems;
    const uint32_t row_lane = row % kRowTileElems;
    const uint32_t col_tile = col / kW8TileElems;
    const uint32_t col_lane = col % kW8TileElems;
    const uint32_t tile_base =
        (row_tile * col_tiles + col_tile) * tile_bytes;
    return static_cast<float>(static_cast<int8_t>(
        weight[tile_base + row_lane * kW8TileElems + col_lane]));
}

float load_w8_fixed_stride_value(const uint8_t* weight, uint32_t row,
                                 uint32_t token,
                                 uint32_t row_tile_stride_bytes) {
    const uint32_t row_tile = row / kRowTileElems;
    const uint32_t row_lane = row % kRowTileElems;
    const uint32_t col_tile = token / kW8TileElems;
    const uint32_t col_lane = token % kW8TileElems;
    const uint32_t tile_bytes = kRowTileElems * kW8TileElems;
    const uint32_t tile_base =
        row_tile * row_tile_stride_bytes + col_tile * tile_bytes;
    return static_cast<float>(static_cast<int8_t>(
        weight[tile_base + row_lane * kW8TileElems + col_lane]));
}

float load_w8_weight_scale_value(const uint8_t* scale, uint32_t row,
                                 uint32_t col_tile, uint32_t cols) {
    const uint32_t col_tiles = ceil_div(cols, kW8TileElems);
    const uint32_t row_tile = row / kRowTileElems;
    const uint32_t row_lane = row % kRowTileElems;
    const uint32_t tile_base =
        (row_tile * col_tiles + col_tile) * kLineBytes;
    return fp16_to_float(load_u16_le(scale + tile_base +
                                    row_lane * kFp16Bytes));
}

float expected_w4_dense_linear_row(const uint8_t* act, const uint8_t* act_scale,
                                   const uint8_t* weight, const uint8_t* scale,
                                   uint32_t row, uint32_t cols) {
    const uint32_t col_tiles = ceil_div(cols, kTileElems);
    float total = 0.0f;
    for (uint32_t col_tile = 0; col_tile < col_tiles; ++col_tile) {
        const uint32_t col_begin = col_tile * kTileElems;
        const uint32_t col_end = std::min(cols, col_begin + kTileElems);
        float partial = 0.0f;
        for (uint32_t col = col_begin; col < col_end; ++col) {
            partial += load_fp16_value(act, col) *
                       load_fp16_value(act_scale, col) *
                       load_w4_weight_value(weight, row, col, cols);
        }
        total += partial * load_w4_weight_scale_value(scale, row, col_tile, cols);
    }
    return total;
}

float expected_w8_dense_row(const uint8_t* act, const uint8_t* weight,
                            const uint8_t* scale, uint32_t row,
                            uint32_t cols) {
    const uint32_t col_tiles = ceil_div(cols, kW8TileElems);
    float total = 0.0f;
    for (uint32_t col_tile = 0; col_tile < col_tiles; ++col_tile) {
        const uint32_t col_begin = col_tile * kW8TileElems;
        const uint32_t col_end = std::min(cols, col_begin + kW8TileElems);
        float partial = 0.0f;
        for (uint32_t col = col_begin; col < col_end; ++col) {
            partial += load_fp16_value(act, col) *
                       load_w8_weight_value(weight, row, col, cols);
        }
        total += partial * load_w8_weight_scale_value(scale, row, col_tile, cols);
    }
    return total;
}

float expected_pv_fixed_stride_row(const uint8_t* prob, const uint8_t* v_scale,
                                   const uint8_t* v_payload, uint32_t row,
                                   uint32_t tokens,
                                   uint32_t row_tile_stride_bytes) {
    float total = 0.0f;
    for (uint32_t token = 0; token < tokens; ++token) {
        total += load_fp16_value(prob, token) *
                 load_fp16_value(v_scale, token) *
                 load_w8_fixed_stride_value(v_payload, row, token,
                                            row_tile_stride_bytes);
    }
    return total;
}

uint16_t fp16_div_pow2_runtime(uint16_t value, int shift) {
    const uint16_t exp = static_cast<uint16_t>((value >> 10) & 0x1fu);
    const uint16_t mant = static_cast<uint16_t>(0x400u | (value & 0x03ffu));
    if ((value & 0x7fffu) == 0) {
        return 0;
    }
    if (exp == 0x1fu) {
        return 0x7c00u;
    }
    if (static_cast<int>(exp) <= shift) {
        const int sub_shift = shift - static_cast<int>(exp) + 1;
        if (sub_shift >= 11) {
            return 0;
        }
        const uint16_t rounded =
            static_cast<uint16_t>((mant + (uint16_t{1} << (sub_shift - 1))) >>
                                  sub_shift);
        return (rounded & 0x0400u) ? 0x0400u : rounded;
    }
    const uint16_t exp_minus = static_cast<uint16_t>(static_cast<int>(exp) - shift);
    return static_cast<uint16_t>((exp_minus << 10) | (value & 0x03ffu));
}

int8_t kv_quantize_ref(uint16_t value, uint16_t max_abs) {
    const bool sign = (value & 0x8000u) != 0;
    const uint16_t value_exp = static_cast<uint16_t>((value >> 10) & 0x1fu);
    const uint16_t max_exp = static_cast<uint16_t>((max_abs >> 10) & 0x1fu);
    if ((max_abs & 0x7fffu) == 0 || (value & 0x7fffu) == 0 ||
        value_exp == 0 || max_exp == 0) {
        return 0;
    }

    const uint64_t value_mant = 0x400ull | (value & 0x03ffu);
    const uint64_t max_mant = 0x400ull | (max_abs & 0x03ffu);
    uint64_t numerator = value_mant << 7;
    const int exp_diff = static_cast<int>(value_exp) - static_cast<int>(max_exp);
    if (exp_diff > 0) {
        numerator <<= static_cast<unsigned>(exp_diff);
    } else if (exp_diff < 0) {
        const int rshift = -exp_diff;
        numerator = (rshift >= 64) ? 0 :
            ((numerator + (1ull << (rshift - 1))) >> rshift);
    }
    uint64_t q_abs = (numerator + (max_mant >> 1)) / max_mant;
    q_abs = std::min<uint64_t>(q_abs, 127u);
    const int q = static_cast<int>(q_abs);
    return static_cast<int8_t>(sign && q != 0 ? -q : q);
}

int32_t fp16_to_q6_14(uint16_t fp16) {
    constexpr int kDataWidth = 20;
    constexpr int kDataFrac = 14;
    const bool sign = (fp16 & 0x8000u) != 0;
    const uint16_t exp = static_cast<uint16_t>((fp16 >> 10) & 0x1fu);
    const uint16_t frac = static_cast<uint16_t>(fp16 & 0x03ffu);
    const uint16_t mant = exp == 0 ? frac : static_cast<uint16_t>(0x400u | frac);
    uint32_t mag = 0;

    if (exp == 0x1fu) {
        mag = 1u << (kDataWidth - 1);
    } else if (exp == 0) {
        if (frac != 0) {
            mag = (static_cast<uint32_t>(frac) + 512u) >> 10;
        }
    } else {
        const int shift = static_cast<int>(exp) - 15 + kDataFrac - 10;
        if (shift >= 0) {
            mag = static_cast<uint32_t>(mant) << shift;
        } else {
            mag = (static_cast<uint32_t>(mant) + (1u << ((-shift) - 1))) >>
                  (-shift);
        }
    }

    if (sign) {
        if (mag >= (1u << (kDataWidth - 1))) {
            return -(1 << (kDataWidth - 1));
        }
        return -static_cast<int32_t>(mag);
    }
    if (mag >= ((1u << (kDataWidth - 1)) - 1u)) {
        return (1 << (kDataWidth - 1)) - 1;
    }
    return static_cast<int32_t>(mag);
}

uint16_t q6_14_to_fp16(int32_t q) {
    constexpr int kDataFrac = 14;
    const bool sign = q < 0;
    uint32_t abs_q = sign ? static_cast<uint32_t>(-q) : static_cast<uint32_t>(q);
    if (abs_q == 0) {
        return static_cast<uint16_t>(sign ? 0x8000u : 0u);
    }

    int p = 0;
    for (int bit = 20; bit >= 0; --bit) {
        if (abs_q & (1u << bit)) {
            p = bit;
            break;
        }
    }
    const int exp_i = p - kDataFrac + 15;
    if (exp_i <= 0) {
        return static_cast<uint16_t>(sign ? 0x8000u : 0u);
    }
    uint16_t exp16 = static_cast<uint16_t>(exp_i & 0x1f);
    uint16_t frac16 = 0;
    uint16_t mant_round = 0;
    if (p >= 10) {
        const int shift = p - 10;
        mant_round = shift == 0 ?
            static_cast<uint16_t>(abs_q) :
            static_cast<uint16_t>((abs_q + (1u << (shift - 1))) >> shift);
    } else {
        mant_round = static_cast<uint16_t>(abs_q << (10 - p));
    }
    if (mant_round & 0x0800u) {
        ++exp16;
        frac16 = 0;
    } else {
        frac16 = static_cast<uint16_t>(mant_round & 0x03ffu);
    }
    if (exp16 >= 0x1fu) {
        return static_cast<uint16_t>((sign ? 0x8000u : 0u) | 0x7c00u);
    }
    return static_cast<uint16_t>((sign ? 0x8000u : 0u) |
                                 (static_cast<uint16_t>(exp16) << 10) |
                                 frac16);
}

int32_t rope_sat20(int64_t value) {
    if (value > 524287) {
        return 524287;
    }
    if (value < -524288) {
        return -524288;
    }
    return static_cast<int32_t>(value);
}

int16_t rope_lut_word_q1_15(uint32_t slot, uint32_t pair, bool is_sin) {
    static constexpr int16_t kCos[8] = {
        0x7e27, 0x7662, 0x5f61, 0x38ea,
        0x001a, static_cast<int16_t>(0xc745),
        static_cast<int16_t>(0x9bcd), static_cast<int16_t>(0x8409),
    };
    static constexpr int16_t kSin[8] = {
        0x15a8, 0x30aa, 0x555b, 0x72a5,
        0x7fff, 0x72bd, 0x4fa5, 0x1fdd,
    };
    const uint32_t sel = (pair * 5u + slot * 3u + 1u) & 7u;
    return is_sin ? kSin[sel] : kCos[sel];
}

void fill_random_rope_lut(uint8_t* dst) {
    std::memset(dst, 0, NPU_ROPE_LUT_WINDOW_BYTES);
    for (uint32_t line = 0; line < 4; ++line) {
        const uint32_t slot = line / 2u;
        const bool is_sin = (line & 1u) != 0;
        for (uint32_t pair = 0; pair < kDecodeHeadDim / 2u; ++pair) {
            const int16_t word = rope_lut_word_q1_15(slot, pair, is_sin);
            store_u16_le(dst + line * kLineBytes + pair * kFp16Bytes,
                         static_cast<uint16_t>(word));
        }
    }
}

uint16_t rope_expected_bits(uint16_t even_bits, uint16_t odd_bits,
                            uint32_t row, uint32_t position) {
    const uint32_t pair = (row % kDecodeHeadDim) / 2u;
    const uint32_t slot = position & 1u;
    const int32_t xe = fp16_to_q6_14(even_bits);
    const int32_t xo = fp16_to_q6_14(odd_bits);
    const int32_t cos_q = static_cast<int16_t>(rope_lut_word_q1_15(slot, pair, false));
    const int32_t sin_q = static_cast<int16_t>(rope_lut_word_q1_15(slot, pair, true));
    const int64_t even_acc = static_cast<int64_t>(xe) * cos_q -
                             static_cast<int64_t>(xo) * sin_q;
    const int64_t odd_acc = static_cast<int64_t>(xe) * sin_q +
                            static_cast<int64_t>(xo) * cos_q;
    const int64_t shifted = (row & 1u) ? (odd_acc >> 15) : (even_acc >> 15);
    return q6_14_to_fp16(rope_sat20(shifted));
}

uint32_t w8_fixed_stride_bytes(uint32_t rows, uint32_t row_tile_stride) {
    return ceil_div(rows, kRowTileElems) * row_tile_stride;
}

void fill_pv_scale_with_group_poison(uint8_t* dst, uint32_t tokens,
                                     uint32_t alloc_bytes) {
    const uint32_t act_bytes = act_payload_bytes(tokens, true);
    std::memset(dst, 0x7e, alloc_bytes);
    std::memset(dst, 0, align_up(act_bytes, kMvinAlign));
    if (align_up(act_bytes, kMvinAlign) < alloc_bytes) {
        std::memset(dst + align_up(act_bytes, kMvinAlign), 0x7e,
                    alloc_bytes - align_up(act_bytes, kMvinAlign));
    }
}

bool compare_fp16_output_tol(const char* label, const uint8_t* got,
                             const std::vector<float>& expected,
                             float abs_tol, float rel_tol) {
    uint32_t mismatches = 0;
    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    for (uint32_t i = 0; i < expected.size(); ++i) {
        const uint16_t got_bits = load_u16_le(got + i * kFp16Bytes);
        const float got_f = fp16_to_float(got_bits);
        const float abs_err = std::fabs(got_f - expected[i]);
        const float rel_err = abs_err / std::max(1.0f, std::fabs(expected[i]));
        const float tol = abs_tol + rel_tol * std::fabs(expected[i]);
        max_abs_err = std::max(max_abs_err, abs_err);
        max_rel_err = std::max(max_rel_err, rel_err);
        if (abs_err > tol) {
            if (mismatches < 8) {
                std::fprintf(stderr,
                             "%s[%u] expected=%g got=%g bits=0x%04x "
                             "abs_err=%g tol=%g\n",
                             label, i, expected[i], got_f, got_bits,
                             abs_err, tol);
            }
            ++mismatches;
        }
    }
    std::printf("%s checked=%zu mismatches=%u max_abs_err=%g max_rel_err=%g\n",
                label, expected.size(), mismatches, max_abs_err, max_rel_err);
    return mismatches == 0;
}

bool compare_fp32_output_tol(const char* label, const uint8_t* got,
                             const std::vector<float>& expected,
                             float abs_tol, float rel_tol) {
    uint32_t mismatches = 0;
    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    for (uint32_t i = 0; i < expected.size(); ++i) {
        const uint32_t got_bits = load_u32_le(got + i * sizeof(float));
        float got_f = 0.0f;
        std::memcpy(&got_f, &got_bits, sizeof(got_f));
        const float abs_err = std::fabs(got_f - expected[i]);
        const float rel_err = abs_err / std::max(1.0f, std::fabs(expected[i]));
        const float tol = abs_tol + rel_tol * std::fabs(expected[i]);
        max_abs_err = std::max(max_abs_err, abs_err);
        max_rel_err = std::max(max_rel_err, rel_err);
        if (abs_err > tol) {
            if (mismatches < 8) {
                std::fprintf(stderr,
                             "%s[%u] expected=%g got=%g bits=0x%08x "
                             "abs_err=%g tol=%g\n",
                             label, i, expected[i], got_f, got_bits,
                             abs_err, tol);
            }
            ++mismatches;
        }
    }
    std::printf("%s checked=%zu mismatches=%u max_abs_err=%g max_rel_err=%g\n",
                label, expected.size(), mismatches, max_abs_err, max_rel_err);
    return mismatches == 0;
}

bool compare_i8_output(const char* label, const uint8_t* got,
                       const std::vector<int8_t>& expected) {
    uint32_t mismatches = 0;
    for (uint32_t i = 0; i < expected.size(); ++i) {
        const int8_t got_i8 = static_cast<int8_t>(got[i]);
        if (got_i8 != expected[i]) {
            if (mismatches < 8) {
                std::fprintf(stderr,
                             "%s[%u] expected=%d got=%d raw=0x%02x\n",
                             label, i, static_cast<int>(expected[i]),
                             static_cast<int>(got_i8),
                             static_cast<unsigned>(got[i]));
            }
            ++mismatches;
        }
    }
    std::printf("%s checked=%zu mismatches=%u\n",
                label, expected.size(), mismatches);
    return mismatches == 0;
}

void print_stream_desc_progress(const char* phase, const char* name,
                                const npu_stream_gemv_desc& desc,
                                uint32_t act_bytes,
                                uint32_t act_scale_bytes,
                                uint32_t weight_bytes,
                                uint32_t scale_bytes,
                                uint32_t output_bytes) {
    std::printf(
        "stream_gemv_api:%s %s m=%u n=%u mode=%u out_prec=%u role=%u "
        "dst=%u post=%u flags=0x%08x groups=%u elem_count=%u "
        "row_tile_stride=%u capacity=%u act_group_stride=%u "
        "bytes act=%u act_scale=%u weight=%u scale=%u output=%u\n",
        phase, name,
        static_cast<unsigned>(desc.m),
        static_cast<unsigned>(desc.n),
        static_cast<unsigned>(desc.mode),
        static_cast<unsigned>(desc.output_precision),
        static_cast<unsigned>(desc.role),
        static_cast<unsigned>(desc.dst),
        static_cast<unsigned>(desc.post_op),
        static_cast<unsigned>(desc.flags),
        static_cast<unsigned>(desc.group_count),
        static_cast<unsigned>(desc.elem_count),
        static_cast<unsigned>(desc.weight_row_tile_stride_bytes),
        static_cast<unsigned>(desc.weight_capacity_tokens),
        static_cast<unsigned>(desc.act_group_stride_bytes),
        static_cast<unsigned>(act_bytes),
        static_cast<unsigned>(act_scale_bytes),
        static_cast<unsigned>(weight_bytes),
        static_cast<unsigned>(scale_bytes),
        static_cast<unsigned>(output_bytes));
    std::fflush(stdout);
}

bool run_linear_case(npu_device* dev) {
    constexpr uint32_t m = 32;
    constexpr uint32_t n = 128;
    const uint32_t act_bytes = align_up(act_payload_bytes(n, false), kMvinAlign);
    const uint32_t act_scale_bytes =
        align_up(act_payload_bytes(n, false), kMvinAlign);
    const uint32_t weight_bytes = align_up(w4_weight_bytes(m, n), kMvinAlign);
    const uint32_t scale_bytes =
        align_up(weight_scale_bytes(m, n, false), kMvinAlign);
    const uint32_t output_bytes = align_up(m * kFp16Bytes, kMvinAlign);

    NpuBuffer act(act_bytes);
    NpuBuffer act_scale(act_scale_bytes);
    NpuBuffer weight(weight_bytes);
    NpuBuffer scale(scale_bytes);
    NpuBuffer output(output_bytes);
    if (!act.ptr || !act_scale.ptr || !weight.ptr || !scale.ptr || !output.ptr) {
        std::fprintf(stderr, "stream_gemv_api_linear: npu_mem_alloc failed\n");
        return false;
    }

    fill_w4_random_activation(act.data(), n, 0x4c494e31u);
    fill_w4_random_act_scale(act_scale.data(), n, 0x4c494e32u);
    fill_w4_random_weight(weight.data(), m, n, 0x4c494e33u);
    fill_w4_random_weight_scale(scale.data(), m, n, 0x4c494e34u);
    std::memset(output.data(), 0xa5, output.bytes);

    npu_stream_gemv_desc desc = {};
    desc.act_ptr = act.ptr;
    desc.act_scale_ptr = act_scale.ptr;
    desc.weight_payload_ptr = weight.ptr;
    desc.weight_scale_ptr = scale.ptr;
    desc.output_ptr = output.ptr;
    desc.m = m;
    desc.n = n;
    desc.mode = DECODE_GEMV_W4A16;
    desc.output_precision = DECODE_OUTPUT_FP16;
    desc.role = NPU_STREAM_GEMV_ROLE_LINEAR;
    desc.dst = NPU_STREAM_GEMV_DST_OUTPUT;
    desc.post_op = NPU_STREAM_POST_BYPASS;
    desc.flags = NPU_STREAM_GEMV_F_ENABLE_ACT_SCALE;

    npu_reset();
    print_stream_desc_progress("BEGIN", "linear", desc, act_bytes,
                               act_scale_bytes, weight_bytes, scale_bytes,
                               output_bytes);
    const int rc = npu_stream_gemv_run(dev, &desc, 0);
    std::printf("stream_gemv_api:END linear rc=%d\n", rc);
    std::fflush(stdout);
    if (rc != 0) {
        std::fprintf(stderr, "stream_gemv_api_linear: npu_stream_gemv_run rc=%d\n", rc);
        return false;
    }

    std::vector<float> expected(m);
    for (uint32_t row = 0; row < m; ++row) {
        expected[row] = expected_w4_dense_linear_row(
            act.data(), act_scale.data(), weight.data(), scale.data(), row, n);
    }
    if (!compare_fp16_output_tol("stream_gemv_api_linear", output.data(),
                                 expected, 0.35f, 0.03f)) {
        return false;
    }
    std::puts("PASS: stream_gemv_api_linear");
    return true;
}

bool run_linear_fp32_mvout_case(npu_device* dev) {
    constexpr uint32_t m = 32;
    constexpr uint32_t n = 128;
    const uint32_t act_bytes = align_up(act_payload_bytes(n, false), kMvinAlign);
    const uint32_t act_scale_bytes =
        align_up(act_payload_bytes(n, false), kMvinAlign);
    const uint32_t weight_bytes = align_up(w4_weight_bytes(m, n), kMvinAlign);
    const uint32_t scale_bytes =
        align_up(weight_scale_bytes(m, n, false), kMvinAlign);
    const uint32_t output_bytes = align_up(m * sizeof(float), kMvinAlign);

    NpuBuffer act(act_bytes);
    NpuBuffer act_scale(act_scale_bytes);
    NpuBuffer weight(weight_bytes);
    NpuBuffer scale(scale_bytes);
    NpuBuffer output(output_bytes);
    if (!act.ptr || !act_scale.ptr || !weight.ptr || !scale.ptr || !output.ptr) {
        std::fprintf(stderr, "stream_gemv_api_linear_fp32_mvout: npu_mem_alloc failed\n");
        return false;
    }

    fill_w4_random_activation(act.data(), n, 0x46503331u);
    fill_w4_random_act_scale(act_scale.data(), n, 0x46503332u);
    fill_w4_random_weight(weight.data(), m, n, 0x46503333u);
    fill_w4_random_weight_scale(scale.data(), m, n, 0x46503334u);
    std::memset(output.data(), 0xa5, output.bytes);

    npu_stream_gemv_desc desc = {};
    desc.act_ptr = act.ptr;
    desc.act_scale_ptr = act_scale.ptr;
    desc.weight_payload_ptr = weight.ptr;
    desc.weight_scale_ptr = scale.ptr;
    desc.output_ptr = output.ptr;
    desc.m = m;
    desc.n = n;
    desc.mode = DECODE_GEMV_W4A16;
    desc.output_precision = DECODE_OUTPUT_FP32;
    desc.role = NPU_STREAM_GEMV_ROLE_LINEAR;
    desc.dst = NPU_STREAM_GEMV_DST_OUTPUT;
    desc.post_op = NPU_STREAM_POST_BYPASS;
    desc.flags = NPU_STREAM_GEMV_F_ENABLE_ACT_SCALE;

    npu_reset();
    print_stream_desc_progress("BEGIN", "linear_fp32_mvout", desc,
                               act_bytes, act_scale_bytes, weight_bytes,
                               scale_bytes, output_bytes);
    const int rc = npu_stream_gemv_run(dev, &desc, 0);
    std::printf("stream_gemv_api:END linear_fp32_mvout rc=%d\n", rc);
    std::fflush(stdout);
    if (rc != 0) {
        std::fprintf(stderr,
                     "stream_gemv_api_linear_fp32_mvout: npu_stream_gemv_run rc=%d\n",
                     rc);
        return false;
    }

    std::vector<float> expected(m);
    for (uint32_t row = 0; row < m; ++row) {
        expected[row] = expected_w4_dense_linear_row(
            act.data(), act_scale.data(), weight.data(), scale.data(), row, n);
    }
    if (!compare_fp32_output_tol("stream_gemv_api_linear_fp32_mvout",
                                 output.data(), expected, 0.35f, 0.03f)) {
        return false;
    }
    std::puts("PASS: stream_gemv_api_linear_fp32_mvout");
    return true;
}

bool run_llama_like_w4_sequence_case(npu_device* dev) {
    struct StreamOp {
        const char* name;
        uint32_t rows;
        uint32_t cols;
        uint32_t seed;
    };

    const StreamOp ops[] = {
        {"attn_q_960x960", 960, 960, 0x11a2c3d4u},
        {"attn_k_320x960", 320, 960, 0x22b3d4e5u},
        {"attn_v_320x960", 320, 960, 0x33c4e5f6u},
        {"attn_o_960x960", 960, 960, 0x44d5f607u},
        {"ffn_gate_2560x960", 2560, 960, 0x55e60718u},
        {"ffn_up_2560x960", 2560, 960, 0x66f71829u},
        {"ffn_down_960x2560", 960, 2560, 0x7708293au},
    };

    struct StreamBuffers {
        const StreamOp* op = nullptr;
        NpuBuffer act;
        NpuBuffer act_scale;
        NpuBuffer weight;
        NpuBuffer scale;
        NpuBuffer output;

        explicit StreamBuffers(const StreamOp& op_in)
            : op(&op_in),
              act(align_up(act_payload_bytes(op_in.cols, false), kMvinAlign)),
              act_scale(align_up(act_payload_bytes(op_in.cols, false), kMvinAlign)),
              weight(align_up(w4_weight_bytes(op_in.rows, op_in.cols), kMvinAlign)),
              scale(align_up(weight_scale_bytes(op_in.rows, op_in.cols, false), kMvinAlign)),
              output(align_up(op_in.rows * sizeof(float), kMvinAlign)) {}

        StreamBuffers(const StreamBuffers&) = delete;
        StreamBuffers& operator=(const StreamBuffers&) = delete;
        StreamBuffers(StreamBuffers&&) noexcept = default;
        StreamBuffers& operator=(StreamBuffers&&) noexcept = default;
    };

    std::vector<std::unique_ptr<StreamBuffers>> buffers;
    buffers.reserve(sizeof(ops) / sizeof(ops[0]));
    for (const StreamOp& op : ops) {
        buffers.emplace_back(new StreamBuffers(op));
        StreamBuffers& cur = *buffers.back();
        if (!cur.act.ptr || !cur.act_scale.ptr || !cur.weight.ptr ||
            !cur.scale.ptr || !cur.output.ptr) {
            std::fprintf(stderr,
                         "stream_gemv_api_llama_like_w4_sequence: npu_mem_alloc failed at %s\n",
                         op.name);
            return false;
        }
        fill_w4_random_activation(cur.act.data(), op.cols, op.seed ^ 0x13579bdfu);
        fill_w4_random_act_scale(cur.act_scale.data(), op.cols,
                                 op.seed ^ 0x2468ace0u);
        fill_w4_random_weight(cur.weight.data(), op.rows, op.cols,
                              op.seed ^ 0x9e3779b9u);
        fill_w4_random_weight_scale(cur.scale.data(), op.rows, op.cols,
                                    op.seed ^ 0x7f4a7c15u);
        std::memset(cur.output.data(), 0xa5, cur.output.bytes);
    }

    bool ok = true;
    npu_reset();
    for (const std::unique_ptr<StreamBuffers>& cur_ptr : buffers) {
        const StreamBuffers& cur = *cur_ptr;
        npu_stream_gemv_desc desc = {};
        desc.act_ptr = cur.act.ptr;
        desc.act_scale_ptr = cur.act_scale.ptr;
        desc.weight_payload_ptr = cur.weight.ptr;
        desc.weight_scale_ptr = cur.scale.ptr;
        desc.output_ptr = cur.output.ptr;
        desc.m = static_cast<uint16_t>(cur.op->rows);
        desc.n = static_cast<uint16_t>(cur.op->cols);
        desc.mode = DECODE_GEMV_W4A16;
        desc.output_precision = DECODE_OUTPUT_FP32;
        desc.role = NPU_STREAM_GEMV_ROLE_LINEAR;
        desc.dst = NPU_STREAM_GEMV_DST_OUTPUT;
        desc.post_op = NPU_STREAM_POST_BYPASS;
        desc.flags = NPU_STREAM_GEMV_F_ENABLE_ACT_SCALE;

        print_stream_desc_progress("BEGIN", cur.op->name, desc,
                                   cur.act.bytes, cur.act_scale.bytes,
                                   cur.weight.bytes, cur.scale.bytes,
                                   cur.output.bytes);
        const int rc = npu_stream_gemv_run(dev, &desc, 0);
        std::printf("stream_gemv_api:END %s rc=%d\n", cur.op->name, rc);
        std::fflush(stdout);
        if (rc != 0) {
            std::fprintf(stderr,
                         "stream_gemv_api_llama_like_w4_sequence: %s rc=%d\n",
                         cur.op->name, rc);
            return false;
        }
        std::vector<float> expected(cur.op->rows);
        for (uint32_t row = 0; row < cur.op->rows; ++row) {
            expected[row] = expected_w4_dense_linear_row(
                cur.act.data(), cur.act_scale.data(), cur.weight.data(),
                cur.scale.data(), row, cur.op->cols);
        }
        const std::string label =
            std::string("stream_gemv_api_llama_like_w4_sequence_") + cur.op->name;
        if (!compare_fp32_output_tol(label.c_str(), cur.output.data(), expected,
                                     0.35f, 0.03f)) {
            ok = false;
        }
    }

    std::puts(ok ? "PASS: stream_gemv_api_llama_like_w4_sequence"
                 : "FAIL: stream_gemv_api_llama_like_w4_sequence");
    return ok;
}

bool run_pv_fixed_stride_tail_case(npu_device* dev) {
    constexpr uint32_t m = 64;
    constexpr uint32_t n = 65;
    constexpr uint32_t capacity = 128;
    constexpr uint32_t row_tile_stride = 4096;

    const uint32_t act_bytes = align_up(act_payload_bytes(n, true), kMvinAlign);
    const uint32_t act_scale_bytes = align_up(act_payload_bytes(n, true), kMvinAlign);
    const uint32_t weight_bytes =
        align_up(w8_fixed_stride_bytes(m, row_tile_stride), kMvinAlign);
    const uint32_t output_bytes = align_up(m * kFp16Bytes, kMvinAlign);

    NpuBuffer prob(act_bytes);
    NpuBuffer v_scale(act_scale_bytes);
    NpuBuffer v_payload(weight_bytes);
    NpuBuffer output(output_bytes);
    if (!prob.ptr || !v_scale.ptr || !v_payload.ptr || !output.ptr) {
        std::fprintf(stderr,
                     "stream_gemv_api_pv_fixed_stride_tail: npu_mem_alloc failed\n");
        return false;
    }

    fill_w8_random_activation(prob.data(), n, 0x50565431u, true);
    fill_w8_random_activation(v_scale.data(), n, 0x50565432u, true);
    fill_w8_random_fixed_stride_weight(v_payload.data(), m, capacity,
                                       row_tile_stride, 0x50565433u);
    std::memset(output.data(), 0xa5, output.bytes);

    npu_stream_gemv_desc desc = {};
    desc.act_ptr = prob.ptr;
    desc.act_scale_ptr = v_scale.ptr;
    desc.weight_payload_ptr = v_payload.ptr;
    desc.weight_scale_ptr = nullptr;
    desc.output_ptr = output.ptr;
    desc.weight_row_tile_stride_bytes = row_tile_stride;
    desc.weight_capacity_tokens = capacity;
    desc.m = m;
    desc.n = n;
    desc.mode = DECODE_GEMV_W8A16;
    desc.output_precision = DECODE_OUTPUT_FP16;
    desc.role = NPU_STREAM_GEMV_ROLE_PV;
    desc.dst = NPU_STREAM_GEMV_DST_OUTPUT;
    desc.post_op = NPU_STREAM_POST_BYPASS;
    desc.flags = NPU_STREAM_GEMV_F_ENABLE_ACT_SCALE |
                 NPU_STREAM_GEMV_F_KV_COL_SCALE |
                 NPU_STREAM_GEMV_F_UNIT_WEIGHT_SCALE;

    npu_reset();
    print_stream_desc_progress("BEGIN", "pv_fixed_stride_tail", desc,
                               act_bytes, act_scale_bytes, weight_bytes,
                               0, output_bytes);
    const int rc = npu_stream_gemv_run(dev, &desc, 0);
    std::printf("stream_gemv_api:END pv_fixed_stride_tail rc=%d\n", rc);
    std::fflush(stdout);
    if (rc != 0) {
        std::fprintf(stderr,
                     "stream_gemv_api_pv_fixed_stride_tail: npu_stream_gemv_run rc=%d\n",
                     rc);
        return false;
    }

    std::vector<float> expected(m);
    for (uint32_t row = 0; row < m; ++row) {
        expected[row] = expected_pv_fixed_stride_row(
            prob.data(), v_scale.data(), v_payload.data(), row, n,
            row_tile_stride);
    }
    if (!compare_fp16_output_tol("stream_gemv_api_pv_fixed_stride_tail",
                                 output.data(), expected, 0.6f, 0.06f)) {
        return false;
    }
    std::puts("PASS: stream_gemv_api_pv_fixed_stride_tail");
    return true;
}

bool run_qk_cached_group_case(npu_device* dev) {
    constexpr uint32_t m = 96;
    constexpr uint32_t n = 64;
    constexpr uint32_t group_count = 3;
    const uint32_t act_bytes_raw = act_payload_bytes(n, true);
    const uint32_t act_group_stride = align_up(act_bytes_raw, kMvinAlign);
    const uint32_t act_span =
        (group_count - 1u) * act_group_stride + act_bytes_raw;
    const uint32_t act_bytes = align_up(act_span, kMvinAlign);
    const uint32_t weight_bytes = align_up(w8_weight_bytes(m, n), kMvinAlign);
    const uint32_t scale_bytes = align_up(weight_scale_bytes(m, n, true), kMvinAlign);
    const uint32_t output_bytes = align_up(group_count * m * kFp16Bytes, kMvinAlign);

    NpuBuffer act(act_bytes);
    NpuBuffer weight(weight_bytes);
    NpuBuffer scale(scale_bytes);
    NpuBuffer output(output_bytes);
    if (!act.ptr || !weight.ptr || !scale.ptr || !output.ptr) {
        std::fprintf(stderr,
                     "stream_gemv_api_qk_cached_group: npu_mem_alloc failed\n");
        return false;
    }

    fill_w8_random_group_activation(act.data(), n, group_count,
                                    act_group_stride, 0x514b4731u, false);
    fill_w8_random_weight(weight.data(), m, n, 0x514b4732u);
    fill_w8_random_weight_scale(scale.data(), m, n, 0x514b4733u);
    std::memset(output.data(), 0xa5, output.bytes);

    npu_stream_gemv_desc desc = {};
    desc.act_ptr = act.ptr;
    desc.weight_payload_ptr = weight.ptr;
    desc.weight_scale_ptr = scale.ptr;
    desc.output_ptr = output.ptr;
    desc.m = m;
    desc.n = n;
    desc.mode = DECODE_GEMV_W8A16;
    desc.output_precision = DECODE_OUTPUT_FP16;
    desc.role = NPU_STREAM_GEMV_ROLE_QK;
    desc.dst = NPU_STREAM_GEMV_DST_OUTPUT;
    desc.post_op = NPU_STREAM_POST_BYPASS;
    desc.elem_count = m;
    desc.group_count = group_count;
    desc.act_group_stride_bytes = act_group_stride;

    print_stream_desc_progress("BEGIN", "qk_cached_group", desc,
                               act_bytes, 0, weight_bytes, scale_bytes,
                               output_bytes);
    const int rc = npu_stream_gemv_run(dev, &desc, 0);
    std::printf("stream_gemv_api:END qk_cached_group rc=%d\n", rc);
    std::fflush(stdout);
    if (rc != 0) {
        std::fprintf(stderr,
                     "stream_gemv_api_qk_cached_group: npu_stream_gemv_run rc=%d\n",
                     rc);
        return false;
    }

    std::vector<float> expected(group_count * m);
    for (uint32_t group = 0; group < group_count; ++group) {
        const uint8_t* group_act = act.data() + group * act_group_stride;
        for (uint32_t row = 0; row < m; ++row) {
            expected[group * m + row] =
                expected_w8_dense_row(group_act, weight.data(), scale.data(),
                                      row, n);
        }
    }
    if (!compare_fp16_output_tol("stream_gemv_api_qk_cached_group",
                                 output.data(), expected, 0.6f, 0.06f)) {
        return false;
    }

    std::puts("PASS: stream_gemv_api_qk_cached_group");
    return true;
}

bool run_qk_softmax_group_case(npu_device* dev, uint32_t m,
                               const char* label) {
    constexpr uint32_t n = 64;
    constexpr uint32_t group_count = 3;
    constexpr float kQkSoftmaxScale = 0.125f;
    const uint32_t act_bytes_raw = act_payload_bytes(n, true);
    const uint32_t act_group_stride = align_up(act_bytes_raw, kMvinAlign);
    const uint32_t act_span =
        (group_count - 1u) * act_group_stride + act_bytes_raw;
    const uint32_t act_bytes = align_up(act_span, kMvinAlign);
    const uint32_t weight_bytes = align_up(w8_weight_bytes(m, n), kMvinAlign);
    const uint32_t scale_bytes = align_up(weight_scale_bytes(m, n, true), kMvinAlign);
    const uint32_t output_bytes = align_up(group_count * m * kFp16Bytes, kMvinAlign);

    NpuBuffer act(act_bytes);
    NpuBuffer weight(weight_bytes);
    NpuBuffer scale(scale_bytes);
    NpuBuffer output(output_bytes);
    if (!act.ptr || !weight.ptr || !scale.ptr || !output.ptr) {
        std::fprintf(stderr, "%s: npu_mem_alloc failed\n", label);
        return false;
    }

    fill_w8_random_group_activation(act.data(), n, group_count,
                                    act_group_stride, 0x534d5801u ^ m, false);
    fill_w8_random_weight(weight.data(), m, n, 0x534d5802u ^ (m << 1));
    fill_w8_random_weight_scale(scale.data(), m, n, 0x534d5803u ^ (m << 2));
    std::memset(output.data(), 0xa5, output.bytes);

    npu_stream_gemv_desc desc = {};
    desc.act_ptr = act.ptr;
    desc.weight_payload_ptr = weight.ptr;
    desc.weight_scale_ptr = scale.ptr;
    desc.output_ptr = output.ptr;
    desc.m = static_cast<uint16_t>(m);
    desc.n = n;
    desc.mode = DECODE_GEMV_W8A16;
    desc.output_precision = DECODE_OUTPUT_FP16;
    desc.role = NPU_STREAM_GEMV_ROLE_QK;
    desc.dst = NPU_STREAM_GEMV_DST_OUTPUT;
    desc.post_op = NPU_STREAM_POST_SOFTMAX;
    desc.elem_count = static_cast<uint16_t>(m);
    desc.group_count = group_count;
    desc.act_group_stride_bytes = act_group_stride;

    print_stream_desc_progress("BEGIN", label, desc,
                               act_bytes, 0, weight_bytes, scale_bytes,
                               output_bytes);
    const int rc = npu_stream_gemv_run(dev, &desc, 0);
    std::printf("stream_gemv_api:END %s rc=%d\n", label, rc);
    std::fflush(stdout);
    if (rc != 0) {
        std::fprintf(stderr, "%s: npu_stream_gemv_run rc=%d\n", label, rc);
        return false;
    }

    std::vector<float> expected(group_count * m);
    bool ok = true;
    for (uint32_t group = 0; group < group_count; ++group) {
        const uint8_t* group_act = act.data() + group * act_group_stride;
        std::vector<float> logits(m);
        float max_logit = -INFINITY;
        for (uint32_t row = 0; row < m; ++row) {
            logits[row] = expected_w8_dense_row(group_act, weight.data(),
                                                scale.data(), row, n) *
                          kQkSoftmaxScale;
            max_logit = std::max(max_logit, logits[row]);
        }
        double denom = 0.0;
        for (uint32_t row = 0; row < m; ++row) {
            denom += std::exp(static_cast<double>(logits[row] - max_logit));
        }
        double got_sum = 0.0;
        uint32_t bad_range = 0;
        for (uint32_t row = 0; row < m; ++row) {
            expected[group * m + row] =
                static_cast<float>(std::exp(static_cast<double>(logits[row] - max_logit)) / denom);
            const float got = load_fp16_value(output.data(), group * m + row);
            got_sum += static_cast<double>(got);
            if (got < -0.001f || got > 1.001f) {
                if (bad_range < 8) {
                    std::fprintf(stderr,
                                 "%s group=%u row=%u got probability out of range %g\n",
                                 label, group, row, got);
                }
                ++bad_range;
            }
        }
        const double sum_err = std::fabs(got_sum - 1.0);
        if (bad_range != 0 || sum_err > 0.08) {
            std::fprintf(stderr,
                         "%s group=%u probability check failed sum=%g sum_err=%g bad_range=%u\n",
                         label, group, got_sum, sum_err, bad_range);
            ok = false;
        } else {
            std::printf("%s group=%u probability sum=%g bad_range=0\n",
                        label, group, got_sum);
        }
    }

    if (!compare_fp16_output_tol(label, output.data(), expected, 0.08f, 0.08f)) {
        ok = false;
    }
    if (ok) {
        std::printf("PASS: %s\n", label);
    }
    return ok;
}

bool run_pv_fixed_stride_group_case(npu_device* dev) {
    constexpr uint32_t m = 64;
    constexpr uint32_t n = 65;
    constexpr uint32_t capacity = 128;
    constexpr uint32_t group_count = 3;
    constexpr uint32_t row_tile_stride = 4096;
    const uint32_t act_bytes_raw = act_payload_bytes(n, true);
    const uint32_t act_group_stride = align_up(act_bytes_raw, kMvinAlign);
    const uint32_t act_span =
        (group_count - 1u) * act_group_stride + act_bytes_raw;

    const uint32_t act_bytes = align_up(act_span, kMvinAlign);
    const uint32_t act_scale_alloc_bytes =
        align_up(group_count * act_bytes_raw, kMvinAlign);
    const uint32_t weight_bytes =
        align_up(w8_fixed_stride_bytes(m, row_tile_stride), kMvinAlign);
    const uint32_t output_bytes =
        align_up(group_count * m * kFp16Bytes, kMvinAlign);

    NpuBuffer prob(act_bytes);
    NpuBuffer v_scale(act_scale_alloc_bytes);
    NpuBuffer v_payload(weight_bytes);
    NpuBuffer output(output_bytes);
    if (!prob.ptr || !v_scale.ptr || !v_payload.ptr || !output.ptr) {
        std::fprintf(stderr,
                     "stream_gemv_api_pv_fixed_stride_group: npu_mem_alloc failed\n");
        return false;
    }

    fill_w8_random_group_activation(prob.data(), n, group_count,
                                    act_group_stride, 0x50564731u, true);
    fill_pv_scale_with_group_poison(v_scale.data(), n, v_scale.bytes);
    fill_w8_random_activation(v_scale.data(), n, 0x50564732u, true);
    if (align_up(act_payload_bytes(n, true), kMvinAlign) < v_scale.bytes) {
        std::memset(v_scale.data() + align_up(act_payload_bytes(n, true), kMvinAlign),
                    0x7e,
                    v_scale.bytes - align_up(act_payload_bytes(n, true), kMvinAlign));
    }
    fill_w8_random_fixed_stride_weight(v_payload.data(), m, capacity,
                                       row_tile_stride, 0x50564733u);
    std::memset(output.data(), 0xa5, output.bytes);

    npu_stream_gemv_desc desc = {};
    desc.act_ptr = prob.ptr;
    desc.act_scale_ptr = v_scale.ptr;
    desc.weight_payload_ptr = v_payload.ptr;
    desc.weight_scale_ptr = nullptr;
    desc.output_ptr = output.ptr;
    desc.weight_row_tile_stride_bytes = row_tile_stride;
    desc.weight_capacity_tokens = capacity;
    desc.m = m;
    desc.n = n;
    desc.mode = DECODE_GEMV_W8A16;
    desc.output_precision = DECODE_OUTPUT_FP16;
    desc.role = NPU_STREAM_GEMV_ROLE_PV;
    desc.dst = NPU_STREAM_GEMV_DST_OUTPUT;
    desc.post_op = NPU_STREAM_POST_BYPASS;
    desc.flags = NPU_STREAM_GEMV_F_ENABLE_ACT_SCALE |
                 NPU_STREAM_GEMV_F_KV_COL_SCALE |
                 NPU_STREAM_GEMV_F_UNIT_WEIGHT_SCALE;
    desc.group_count = group_count;
    desc.act_group_stride_bytes = act_group_stride;

    npu_reset();
    print_stream_desc_progress("BEGIN", "pv_fixed_stride_group", desc,
                               act_bytes, act_scale_alloc_bytes,
                               weight_bytes, 0, output_bytes);
    const int rc = npu_stream_gemv_run(dev, &desc, 0);
    std::printf("stream_gemv_api:END pv_fixed_stride_group rc=%d\n", rc);
    std::fflush(stdout);
    if (rc != 0) {
        std::fprintf(stderr,
                     "stream_gemv_api_pv_fixed_stride_group: npu_stream_gemv_run rc=%d\n",
                     rc);
        return false;
    }

    std::vector<float> expected(group_count * m);
    for (uint32_t group = 0; group < group_count; ++group) {
        const uint8_t* group_prob = prob.data() + group * act_group_stride;
        for (uint32_t row = 0; row < m; ++row) {
            expected[group * m + row] = expected_pv_fixed_stride_row(
                group_prob, v_scale.data(), v_payload.data(), row, n,
                row_tile_stride);
        }
    }
    if (!compare_fp16_output_tol("stream_gemv_api_pv_fixed_stride_group",
                                 output.data(), expected, 0.6f, 0.06f)) {
        return false;
    }
    std::puts("PASS: stream_gemv_api_pv_fixed_stride_group");
    return true;
}

bool run_pv_fixed_stride_group_llama88_case(npu_device* dev, uint16_t position,
                                            const char* label) {
    constexpr uint32_t m = 64;
    constexpr uint32_t n = 88;
    constexpr uint32_t capacity = 4096;
    constexpr uint32_t group_count = 3;
    constexpr uint32_t row_tile_stride = 131072;
    const uint32_t act_bytes_raw = act_payload_bytes(n, true);
    const uint32_t act_group_stride = align_up(act_bytes_raw, kMvinAlign);
    const uint32_t act_span =
        (group_count - 1u) * act_group_stride + act_bytes_raw;
    const uint32_t act_bytes = align_up(act_span, kMvinAlign);
    const uint32_t act_scale_alloc_bytes = align_up(act_bytes_raw, kMvinAlign);
    const uint32_t weight_bytes =
        align_up(w8_fixed_stride_bytes(m, row_tile_stride), kMvinAlign);
    const uint32_t output_bytes =
        align_up(group_count * m * kFp16Bytes, kMvinAlign);

    NpuBuffer prob(act_bytes);
    NpuBuffer v_scale(act_scale_alloc_bytes);
    NpuBuffer v_payload(weight_bytes);
    NpuBuffer output(output_bytes);
    if (!prob.ptr || !v_scale.ptr || !v_payload.ptr || !output.ptr) {
        std::fprintf(stderr, "%s: npu_mem_alloc failed\n", label);
        return false;
    }

    fill_w8_random_group_activation(prob.data(), n, group_count,
                                    act_group_stride, 0x50564c31u, true);
    fill_w8_random_activation(v_scale.data(), n, 0x50564c32u, true);
    fill_w8_random_fixed_stride_weight(v_payload.data(), m, capacity,
                                       row_tile_stride, 0x50564c33u);
    std::memset(output.data(), 0xa5, output.bytes);

    npu_stream_gemv_desc desc = {};
    desc.act_ptr = prob.ptr;
    desc.act_scale_ptr = v_scale.ptr;
    desc.weight_payload_ptr = v_payload.ptr;
    desc.weight_scale_ptr = nullptr;
    desc.output_ptr = output.ptr;
    desc.weight_row_tile_stride_bytes = row_tile_stride;
    desc.weight_capacity_tokens = capacity;
    desc.m = m;
    desc.n = n;
    desc.mode = DECODE_GEMV_W8A16;
    desc.output_precision = DECODE_OUTPUT_FP16;
    desc.role = NPU_STREAM_GEMV_ROLE_PV;
    desc.dst = NPU_STREAM_GEMV_DST_OUTPUT;
    desc.post_op = NPU_STREAM_POST_BYPASS;
    desc.flags = NPU_STREAM_GEMV_F_ENABLE_ACT_SCALE |
                 NPU_STREAM_GEMV_F_KV_COL_SCALE |
                 NPU_STREAM_GEMV_F_UNIT_WEIGHT_SCALE;
    desc.group_count = group_count;
    desc.act_group_stride_bytes = act_group_stride;
    desc.elem_count = m;
    desc.position = position;

    npu_reset();
    print_stream_desc_progress("BEGIN", label, desc,
                               act_bytes, act_scale_alloc_bytes,
                               weight_bytes, 0, output_bytes);
    const int rc = npu_stream_gemv_run(dev, &desc, 0);
    std::printf("stream_gemv_api:END %s rc=%d\n", label, rc);
    std::fflush(stdout);
    if (rc != 0) {
        std::fprintf(stderr, "%s: npu_stream_gemv_run rc=%d\n", label, rc);
        return false;
    }

    std::vector<float> expected(group_count * m);
    for (uint32_t group = 0; group < group_count; ++group) {
        const uint8_t* group_prob = prob.data() + group * act_group_stride;
        for (uint32_t row = 0; row < m; ++row) {
            expected[group * m + row] = expected_pv_fixed_stride_row(
                group_prob, v_scale.data(), v_payload.data(), row, n,
                row_tile_stride);
        }
    }
    if (!compare_fp16_output_tol(label, output.data(), expected, 0.6f, 0.06f)) {
        return false;
    }
    std::printf("PASS: %s\n", label);
    return true;
}

bool run_rope_postprocess_case(npu_device* dev) {
    constexpr uint32_t m = 64;
    constexpr uint32_t n = 128;
    constexpr uint32_t position = 17;
    const uint32_t act_bytes = align_up(act_payload_bytes(n, false), kMvinAlign);
    const uint32_t act_scale_bytes = align_up(act_payload_bytes(n, false), kMvinAlign);
    const uint32_t weight_bytes = align_up(w4_weight_bytes(m, n), kMvinAlign);
    const uint32_t scale_bytes = align_up(weight_scale_bytes(m, n, false), kMvinAlign);
    const uint32_t rope_lut_bytes = align_up(NPU_ROPE_LUT_WINDOW_BYTES, kMvinAlign);
    const uint32_t output_bytes = align_up(m * kFp16Bytes, kMvinAlign);

    NpuBuffer act(act_bytes);
    NpuBuffer act_scale(act_scale_bytes);
    NpuBuffer weight(weight_bytes);
    NpuBuffer scale(scale_bytes);
    NpuBuffer rope_lut(rope_lut_bytes);
    NpuBuffer output(output_bytes);
    if (!act.ptr || !act_scale.ptr || !weight.ptr || !scale.ptr ||
        !rope_lut.ptr || !output.ptr) {
        std::fprintf(stderr, "stream_gemv_api_rope_postprocess: npu_mem_alloc failed\n");
        return false;
    }

    fill_w4_random_activation(act.data(), n, 0x524f5031u);
    fill_w4_random_act_scale(act_scale.data(), n, 0x524f5032u);
    fill_w4_random_weight(weight.data(), m, n, 0x524f5033u);
    fill_w4_random_weight_scale(scale.data(), m, n, 0x524f5034u);
    fill_random_rope_lut(rope_lut.data());
    std::memset(output.data(), 0xa5, output.bytes);

    npu_stream_gemv_desc desc = {};
    desc.act_ptr = act.ptr;
    desc.act_scale_ptr = act_scale.ptr;
    desc.weight_payload_ptr = weight.ptr;
    desc.weight_scale_ptr = scale.ptr;
    desc.output_ptr = output.ptr;
    desc.rope_lut_ptr = rope_lut.ptr;
    desc.m = m;
    desc.n = n;
    desc.mode = DECODE_GEMV_W4A16;
    desc.output_precision = DECODE_OUTPUT_FP16;
    desc.role = NPU_STREAM_GEMV_ROLE_LINEAR;
    desc.dst = NPU_STREAM_GEMV_DST_OUTPUT;
    desc.post_op = NPU_STREAM_POST_ROPE;
    desc.flags = NPU_STREAM_GEMV_F_ENABLE_ACT_SCALE;
    desc.elem_count = m;
    desc.position = position;

    npu_reset();
    print_stream_desc_progress("BEGIN", "rope_postprocess", desc,
                               act_bytes, act_scale_bytes, weight_bytes,
                               scale_bytes, output_bytes);
    const int rc = npu_stream_gemv_run(dev, &desc, 0);
    std::printf("stream_gemv_api:END rope_postprocess rc=%d\n", rc);
    std::fflush(stdout);
    if (rc != 0) {
        std::fprintf(stderr,
                     "stream_gemv_api_rope_postprocess: npu_stream_gemv_run rc=%d\n",
                     rc);
        return false;
    }

    std::vector<float> expected(m);
    std::vector<uint16_t> gemv_bits(m);
    for (uint32_t row = 0; row < m; ++row) {
        gemv_bits[row] = fp32_to_fp16_bits(expected_w4_dense_linear_row(
            act.data(), act_scale.data(), weight.data(), scale.data(), row, n));
    }
    for (uint32_t row = 0; row < m; ++row) {
        const uint16_t even_bits = gemv_bits[row & ~1u];
        const uint16_t odd_bits = gemv_bits[row | 1u];
        expected[row] = fp16_to_float(rope_expected_bits(even_bits, odd_bits,
                                                         row, position));
    }
    if (!compare_fp16_output_tol("stream_gemv_api_rope_postprocess",
                                 output.data(), expected, 0.45f, 0.04f)) {
        return false;
    }
    std::puts("PASS: stream_gemv_api_rope_postprocess");
    return true;
}

bool run_kv_quant_case(npu_device* dev, bool is_v) {
    constexpr uint32_t m = kDecodeKvDim;
    constexpr uint32_t n = 128;
    constexpr uint32_t position = 17;
    const uint32_t act_bytes = align_up(act_payload_bytes(n, false), kMvinAlign);
    const uint32_t act_scale_bytes = align_up(act_payload_bytes(n, false), kMvinAlign);
    const uint32_t weight_bytes = align_up(w4_weight_bytes(m, n), kMvinAlign);
    const uint32_t scale_bytes = align_up(weight_scale_bytes(m, n, false), kMvinAlign);
    const uint32_t rope_lut_bytes = align_up(NPU_ROPE_LUT_WINDOW_BYTES, kMvinAlign);
    const uint32_t output_bytes = align_up(m, kMvinAlign);

    NpuBuffer act(act_bytes);
    NpuBuffer act_scale(act_scale_bytes);
    NpuBuffer weight(weight_bytes);
    NpuBuffer scale(scale_bytes);
    NpuBuffer rope_lut(rope_lut_bytes);
    NpuBuffer output(output_bytes);
    if (!act.ptr || !act_scale.ptr || !weight.ptr || !scale.ptr ||
        !rope_lut.ptr || !output.ptr) {
        std::fprintf(stderr, "stream_gemv_api_kv_quant_%s: npu_mem_alloc failed\n",
                     is_v ? "v" : "k");
        return false;
    }

    fill_w4_random_activation(act.data(), n, is_v ? 0x56514e31u : 0x4b514e31u);
    fill_w4_random_act_scale(act_scale.data(), n, is_v ? 0x56514e32u : 0x4b514e32u);
    fill_w4_random_weight(weight.data(), m, n, is_v ? 0x56514e33u : 0x4b514e33u);
    fill_w4_random_weight_scale(scale.data(), m, n, is_v ? 0x56514e34u : 0x4b514e34u);
    fill_random_rope_lut(rope_lut.data());
    std::memset(output.data(), 0xa5, output.bytes);

    npu_stream_gemv_desc desc = {};
    desc.act_ptr = act.ptr;
    desc.act_scale_ptr = act_scale.ptr;
    desc.weight_payload_ptr = weight.ptr;
    desc.weight_scale_ptr = scale.ptr;
    desc.output_ptr = output.ptr;
    desc.rope_lut_ptr = is_v ? nullptr : rope_lut.ptr;
    desc.m = m;
    desc.n = n;
    desc.mode = DECODE_GEMV_W4A16;
    desc.output_precision = DECODE_OUTPUT_FP16;
    desc.role = NPU_STREAM_GEMV_ROLE_KV_PROJ;
    desc.dst = NPU_STREAM_GEMV_DST_OUTPUT;
    desc.post_op = is_v ? NPU_STREAM_POST_BYPASS : NPU_STREAM_POST_ROPE;
    desc.flags = NPU_STREAM_GEMV_F_ENABLE_ACT_SCALE |
                 NPU_STREAM_GEMV_F_KV_QUANT |
                 (is_v ? static_cast<uint32_t>(NPU_STREAM_GEMV_F_KV_IS_V) : 0u);
    desc.elem_count = m;
    desc.position = position;

    const char* label = is_v ? "kv_quant_v_bypass" : "kv_quant_k_rope";
    npu_reset();
    print_stream_desc_progress("BEGIN", label, desc,
                               act_bytes, act_scale_bytes, weight_bytes,
                               scale_bytes, output_bytes);
    const int rc = npu_stream_gemv_run(dev, &desc, 0);
    std::printf("stream_gemv_api:END %s rc=%d\n", label, rc);
    std::fflush(stdout);
    if (rc != 0) {
        std::fprintf(stderr, "stream_gemv_api_%s: npu_stream_gemv_run rc=%d\n",
                     label, rc);
        return false;
    }

    std::vector<uint16_t> source_bits(m);
    std::vector<uint16_t> gemv_bits(m);
    for (uint32_t row = 0; row < m; ++row) {
        gemv_bits[row] = fp32_to_fp16_bits(expected_w4_dense_linear_row(
            act.data(), act_scale.data(), weight.data(), scale.data(), row, n));
    }
    for (uint32_t row = 0; row < m; ++row) {
        if (is_v) {
            source_bits[row] = gemv_bits[row];
        } else {
            source_bits[row] = rope_expected_bits(gemv_bits[row & ~1u],
                                                  gemv_bits[row | 1u],
                                                  row, position);
        }
    }

    npu_stream_kv_scale_result scale_result = {};
    const int scale_rc = npu_stream_kv_scale_read(dev, is_v ? 1 : 0, &scale_result);
    if (scale_rc != 0 || !scale_result.valid || scale_result.count != 5) {
        std::fprintf(stderr,
                     "stream_gemv_api_%s: scale read rc=%d valid=%u count=%u\n",
                     label, scale_rc, scale_result.valid, scale_result.count);
        return false;
    }

    std::vector<int8_t> expected_i8(m);
    bool ok = true;
    for (uint32_t head = 0; head < m / kDecodeHeadDim; ++head) {
        uint16_t max_abs = 0;
        for (uint32_t dim = 0; dim < kDecodeHeadDim; ++dim) {
            const uint16_t abs_bits =
                static_cast<uint16_t>(source_bits[head * kDecodeHeadDim + dim] &
                                      0x7fffu);
            max_abs = std::max(max_abs, abs_bits);
        }
        const uint16_t expected_scale = fp16_div_pow2_runtime(max_abs, 7);
        if (scale_result.values[head] != expected_scale) {
            std::fprintf(stderr,
                         "stream_gemv_api_%s_scale[%u] expected=0x%04x got=0x%04x max_abs=0x%04x\n",
                         label, head, expected_scale, scale_result.values[head],
                         max_abs);
            ok = false;
        }
        for (uint32_t dim = 0; dim < kDecodeHeadDim; ++dim) {
            const uint32_t elem = head * kDecodeHeadDim + dim;
            expected_i8[elem] = kv_quantize_ref(source_bits[elem], max_abs);
        }
    }

    ok = compare_i8_output((std::string("stream_gemv_api_") + label).c_str(),
                           output.data(), expected_i8) && ok;
    std::puts(ok ? (is_v ? "PASS: stream_gemv_api_kv_quant_v_bypass"
                         : "PASS: stream_gemv_api_kv_quant_k_rope")
                 : (is_v ? "FAIL: stream_gemv_api_kv_quant_v_bypass"
                         : "FAIL: stream_gemv_api_kv_quant_k_rope"));
    return ok;
}

bool run_negative_descriptor_cases(npu_device* dev) {
    constexpr uint32_t m = 32;
    constexpr uint32_t n = 64;
    const uint32_t act_bytes = align_up(act_payload_bytes(n, false), kMvinAlign);
    const uint32_t weight_bytes =
        align_up(ceil_div(m, kRowTileElems) *
                     ceil_div(n, kTileElems) * 4096u,
                 kMvinAlign);
    const uint32_t scale_bytes = align_up(ceil_div(m, kRowTileElems) *
                                              ceil_div(n, kTileElems) * kLineBytes,
                                          kMvinAlign);
    const uint32_t output_bytes = align_up(m * kFp16Bytes, kMvinAlign);

    NpuBuffer act(act_bytes);
    NpuBuffer weight(weight_bytes);
    NpuBuffer scale(scale_bytes);
    NpuBuffer output(output_bytes);
    if (!act.ptr || !weight.ptr || !scale.ptr || !output.ptr) {
        std::fprintf(stderr,
                     "stream_gemv_api_negative_descriptors: npu_mem_alloc failed\n");
        return false;
    }

    npu_stream_gemv_desc desc = {};
    desc.act_ptr = act.ptr;
    desc.weight_payload_ptr = weight.ptr;
    desc.weight_scale_ptr = scale.ptr;
    desc.output_ptr = output.ptr;
    desc.m = m;
    desc.n = n;
    desc.mode = DECODE_GEMV_W4A16;
    desc.output_precision = DECODE_OUTPUT_FP16;
    desc.role = NPU_STREAM_GEMV_ROLE_LINEAR;
    desc.dst = NPU_STREAM_GEMV_DST_OUTPUT;
    desc.post_op = NPU_STREAM_POST_BYPASS;
    desc.group_count = 2;
    print_stream_desc_progress("BEGIN", "negative_grouped_linear", desc,
                               act_bytes, 0, weight_bytes, scale_bytes,
                               output_bytes);
    int rc = npu_stream_gemv_run(dev, &desc, 0);
    std::printf("stream_gemv_api:END negative_grouped_linear rc=%d\n", rc);
    std::fflush(stdout);
    if (rc != -EINVAL) {
        std::fprintf(stderr,
                     "stream_gemv_api_negative_descriptors: grouped LINEAR rc=%d\n",
                     rc);
        return false;
    }

    desc.group_count = 1;
    desc.mode = DECODE_GEMV_W8A16;
    desc.role = NPU_STREAM_GEMV_ROLE_PV;
    desc.flags = NPU_STREAM_GEMV_F_ENABLE_ACT_SCALE |
                 NPU_STREAM_GEMV_F_KV_COL_SCALE |
                 NPU_STREAM_GEMV_F_UNIT_WEIGHT_SCALE;
    desc.act_scale_ptr = act.ptr;
    desc.weight_scale_ptr = nullptr;
    desc.m = 64;
    desc.n = n;
    desc.group_count = 2;
    desc.act_group_stride_bytes = align_up(act_payload_bytes(n, true), kMvinAlign);
    desc.weight_capacity_tokens = n;
    desc.weight_row_tile_stride_bytes = 0;
    print_stream_desc_progress("BEGIN", "negative_grouped_pv_compact", desc,
                               act_bytes, act_bytes, weight_bytes, 0,
                               output_bytes);
    rc = npu_stream_gemv_run(dev, &desc, 0);
    std::printf("stream_gemv_api:END negative_grouped_pv_compact rc=%d\n", rc);
    std::fflush(stdout);
    if (rc != -EINVAL) {
        std::fprintf(stderr,
                     "stream_gemv_api_negative_descriptors: grouped PV compact rc=%d\n",
                     rc);
        return false;
    }

    std::puts("PASS: stream_gemv_api_negative_descriptors");
    return true;
}

} // namespace

int main() {
    std::puts("kv260_stream_gemv_api_test: npu_stream_gemv_run positive API test");
    std::fflush(stdout);

    npu_device* dev = nullptr;
    const int open_rc = npu_open(&dev, nullptr);
    if (open_rc != 0 || !dev) {
        std::fprintf(stderr,
                     "kv260_stream_gemv_api_test: npu_open failed rc=%d "
                     "(is /dev/npu_kv260 present?)\n",
                     open_rc);
        return 1;
    }

    bool ok = true;
    ok = run_linear_case(dev) && ok;
    ok = run_linear_fp32_mvout_case(dev) && ok;
    ok = run_llama_like_w4_sequence_case(dev) && ok;
    ok = run_rope_postprocess_case(dev) && ok;
    ok = run_kv_quant_case(dev, false) && ok;
    ok = run_kv_quant_case(dev, true) && ok;
    ok = run_qk_cached_group_case(dev) && ok;
    ok = run_qk_softmax_group_case(dev, 88, "stream_gemv_api_qk_softmax_group_m88") && ok;
    ok = run_qk_softmax_group_case(dev, 502, "stream_gemv_api_qk_softmax_group_m502") && ok;
    ok = run_qk_softmax_group_case(dev, 1024, "stream_gemv_api_qk_softmax_group_m1024") && ok;
    ok = run_pv_fixed_stride_tail_case(dev) && ok;
    ok = run_pv_fixed_stride_group_case(dev) && ok;
    ok = run_pv_fixed_stride_group_llama88_case(
        dev, 0, "stream_gemv_api_pv_fixed_stride_group_llama88_pos0") && ok;
    ok = run_pv_fixed_stride_group_llama88_case(
        dev, 87, "stream_gemv_api_pv_fixed_stride_group_llama88_pos87") && ok;
    ok = run_negative_descriptor_cases(dev) && ok;

    npu_close(dev);

    std::puts(ok ? "kv260_stream_gemv_api_test=ok"
                 : "kv260_stream_gemv_api_test=fail");
    return ok ? 0 : 2;
}
