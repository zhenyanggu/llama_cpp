#include "npu_runtime.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kTileElems = 128;
constexpr uint32_t kRowTileElems = 32;
constexpr uint32_t kSpmBytes = 512 * 1024;
constexpr uint32_t kSpmLineBytes = 64;
constexpr uint32_t kMvinTransferAlign = 256;
constexpr uint32_t kFp16Bytes = 2;
constexpr uint32_t kActBufferBytes = NPU_GEMV_ACT_BUFFER_BYTES;
constexpr uint32_t kActTileBytes = kTileElems * kFp16Bytes;
constexpr uint32_t kScaleTileBytes = kRowTileElems * kFp16Bytes;
constexpr uint32_t kWeightRowBytes = (kTileElems * 4) / 8;
constexpr uint32_t kWeightTileBytes = kRowTileElems * kWeightRowBytes;
constexpr uint32_t kPackedWeightTileBytes = kScaleTileBytes + kWeightTileBytes;
constexpr uint8_t kInputTypeData = 0;
constexpr uint8_t kInputTypeWeight = 1;
constexpr uint8_t kInputTypeAct = 3;

constexpr uint32_t kScaleSpmBase = 0x0000;
constexpr uint32_t kOutputSpmBase = 0x0000;
constexpr uint32_t kWeightSpmBase = 0x10000;
constexpr uint32_t kPingWeightSpmBase = 0x10000;
constexpr uint32_t kPongWeightSpmBase = 0x40000;
constexpr uint32_t kPingActSpmBase = 0x0000;
constexpr double kDefaultDdrPeakGBps = 19.2;

constexpr uint16_t kFp16Zero = 0x0000;
constexpr uint16_t kFp16Half = 0x3800;
constexpr uint16_t kFp16One = 0x3c00;
constexpr uint16_t kFp16Two = 0x4000;

using Clock = std::chrono::steady_clock;

struct GemvCase {
    const char* name;
    uint16_t m;
    uint16_t k;
    bool varied_scale;
    bool varied_act_scale;
    bool explicit_bases;
    uint32_t weight_spm_base;
    uint32_t act_spm_base;
    uint32_t scale_spm_base;
    uint32_t output_spm_base;
};

struct Layout {
    uint32_t weight_bytes;
    uint32_t act_bytes;
    uint32_t scale_bytes;
    uint32_t output_dma_bytes;
    uint32_t weight_spm_base;
    uint32_t act_spm_base;
    uint32_t scale_spm_base;
    uint32_t output_spm_base;
};

struct FixedBlockLayout {
    uint32_t weight_bytes;
    uint32_t act_bytes;
    uint32_t scale_bytes;
    uint32_t output_dma_bytes;
    uint32_t weight_spm_base;
    uint32_t act_spm_base;
    uint32_t scale_spm_base;
    uint32_t output_spm_base;
};

struct NpuBuffer {
    void* ptr = nullptr;
    size_t bytes = 0;

    NpuBuffer() = default;
    explicit NpuBuffer(size_t size) : ptr(npu_mem_alloc(size)), bytes(size) {}
    ~NpuBuffer() {
        if (ptr) npu_mem_free(ptr);
    }

    NpuBuffer(const NpuBuffer&) = delete;
    NpuBuffer& operator=(const NpuBuffer&) = delete;
    NpuBuffer(NpuBuffer&& other) noexcept : ptr(other.ptr), bytes(other.bytes) {
        other.ptr = nullptr;
        other.bytes = 0;
    }
    NpuBuffer& operator=(NpuBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr) npu_mem_free(ptr);
            ptr = other.ptr;
            bytes = other.bytes;
            other.ptr = nullptr;
            other.bytes = 0;
        }
        return *this;
    }

    uint8_t* data() { return static_cast<uint8_t*>(ptr); }
};

struct BlockPlan {
    uint32_t row_base;
    uint32_t rows;
    uint32_t row_tiles;
    uint32_t weight_bytes;
    uint32_t weight_spm_base;
    NpuBuffer weight;

    BlockPlan(uint32_t row_base_in, uint32_t rows_in, uint32_t row_tiles_in,
              uint32_t weight_bytes_in, uint32_t weight_spm_base_in)
        : row_base(row_base_in),
          rows(rows_in),
          row_tiles(row_tiles_in),
          weight_bytes(weight_bytes_in),
          weight_spm_base(weight_spm_base_in),
          weight(weight_bytes_in) {}

    BlockPlan(const BlockPlan&) = delete;
    BlockPlan& operator=(const BlockPlan&) = delete;
    BlockPlan(BlockPlan&&) noexcept = default;
    BlockPlan& operator=(BlockPlan&&) noexcept = default;
};

uint32_t align_up(uint32_t value, uint32_t alignment) {
    return (value + alignment - 1u) & ~(alignment - 1u);
}

uint32_t ceil_div(uint32_t value, uint32_t divisor) {
    return (value + divisor - 1u) / divisor;
}

uint64_t elapsed_ns(Clock::time_point begin, Clock::time_point end) {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
}

double ns_to_ms(uint64_t ns) {
    return static_cast<double>(ns) / 1000000.0;
}

double bytes_per_ns_to_gbps(uint64_t bytes, uint64_t ns) {
    if (ns == 0) {
        return 0.0;
    }
    return static_cast<double>(bytes) / static_cast<double>(ns);
}

int get_env_int(const char* name, int fallback) {
    const char* value = std::getenv(name);
    if (!value || value[0] == '\0') {
        return fallback;
    }
    const int parsed = std::atoi(value);
    return parsed > 0 ? parsed : fallback;
}

double get_env_double(const char* name, double fallback) {
    const char* value = std::getenv(name);
    if (!value || value[0] == '\0') {
        return fallback;
    }
    const double parsed = std::atof(value);
    return parsed > 0.0 ? parsed : fallback;
}

bool get_env_flag(const char* name) {
    const char* value = std::getenv(name);
    if (!value || value[0] == '\0') {
        return false;
    }
    return std::strcmp(value, "0") != 0 &&
           std::strcmp(value, "false") != 0 &&
           std::strcmp(value, "FALSE") != 0;
}

uint32_t parse_size_env(const char* name, uint32_t fallback) {
    const char* value = std::getenv(name);
    if (!value || value[0] == '\0') {
        return fallback;
    }
    char* end = nullptr;
    unsigned long long parsed = std::strtoull(value, &end, 0);
    if (end == value) {
        return fallback;
    }
    unsigned long long multiplier = 1;
    if (*end == 'K' || *end == 'k') {
        multiplier = 1024ull;
    } else if (*end == 'M' || *end == 'm') {
        multiplier = 1024ull * 1024ull;
    } else if (*end == 'G' || *end == 'g') {
        multiplier = 1024ull * 1024ull * 1024ull;
    } else if (*end != '\0') {
        return fallback;
    }
    parsed *= multiplier;
    if (parsed > 0xffffffffull) {
        return fallback;
    }
    return static_cast<uint32_t>(parsed);
}

bool env_mode_is(const char* mode, const char* expected) {
    return mode && std::strcmp(mode, expected) == 0;
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
    static_assert(sizeof(bits) == sizeof(value), "float must be 32-bit");
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

uint16_t fp32_to_fp16_bits(float value) {
    const uint32_t bits = fp32_bits(value);
    const uint32_t sign = (bits >> 31) & 0x1u;
    const uint32_t exp = (bits >> 23) & 0xffu;
    const uint32_t frac = bits & 0x7fffffu;
    const int exp16 = static_cast<int>(exp) - 127 + 15;

    if (exp == 0) {
        return static_cast<uint16_t>(sign << 15);
    }
    if (exp == 0xffu) {
        return static_cast<uint16_t>((sign << 15) | 0x7c00u | (frac ? 0x0200u : 0u));
    }
    if (exp16 <= 0) {
        return static_cast<uint16_t>(sign << 15);
    }
    if (exp16 >= 31) {
        return static_cast<uint16_t>((sign << 15) | 0x7c00u);
    }

    const uint32_t mant = frac | 0x800000u;
    uint32_t rounded = mant + 0x1000u;
    if (rounded & 0x1000000u) {
        rounded >>= 1;
        const int bumped_exp = exp16 + 1;
        if (bumped_exp >= 31) {
            return static_cast<uint16_t>((sign << 15) | 0x7c00u);
        }
        return static_cast<uint16_t>((sign << 15) |
                                     (static_cast<uint32_t>(bumped_exp) << 10));
    }

    return static_cast<uint16_t>((sign << 15) |
                                 (static_cast<uint32_t>(exp16) << 10) |
                                 ((rounded >> 13) & 0x3ffu));
}

int8_t weight_value(uint32_t row, uint32_t col, int weight_bias = 0) {
    int base = 0;
    switch (row & 0x3u) {
    case 0:
        base = 1;
        break;
    case 1:
        base = -1;
        break;
    case 2:
        base = (col & 0x1u) ? -1 : 1;
        break;
    default:
        base = 2;
        break;
    }
    return static_cast<int8_t>(base + weight_bias);
}

uint16_t scale_bits(uint32_t row, uint32_t k_group, bool varied_scale) {
    if (!varied_scale) {
        return kFp16One;
    }
    switch ((row + k_group) & 0x3u) {
    case 0:
        return kFp16One;
    case 1:
        return kFp16Half;
    case 2:
        return kFp16Two;
    default:
        return kFp16One;
    }
}

float scale_value(uint32_t row, uint32_t k_group, bool varied_scale) {
    if (!varied_scale) {
        return 1.0f;
    }
    switch ((row + k_group) & 0x3u) {
    case 0:
        return 1.0f;
    case 1:
        return 0.5f;
    case 2:
        return 2.0f;
    default:
        return 1.0f;
    }
}

float arbitrary_scale_value(uint32_t row, uint32_t k_group) {
    static const float values[] = {
        0.001953125f, 0.0029296875f, 0.00390625f, 0.005859375f,
        0.0078125f, 0.01171875f, 0.015625f, 0.0234375f,
        0.03125f, 0.046875f, 0.0625f, 0.09375f,
    };
    return values[(row * 7u + k_group * 11u) % (sizeof(values) / sizeof(values[0]))];
}

uint16_t arbitrary_scale_bits(uint32_t row, uint32_t k_group) {
    return fp32_to_fp16_bits(arbitrary_scale_value(row, k_group));
}

uint16_t act_scale_bits(uint32_t col, bool varied_act_scale) {
    if (!varied_act_scale) {
        return kFp16One;
    }
    switch ((col >> 5) & 0x3u) {
    case 0:
        return kFp16Half;
    case 1:
        return kFp16One;
    case 2:
        return kFp16Two;
    default:
        return kFp16One;
    }
}

float act_scale_value(uint32_t col, bool varied_act_scale) {
    if (!varied_act_scale) {
        return 1.0f;
    }
    switch ((col >> 5) & 0x3u) {
    case 0:
        return 0.5f;
    case 1:
        return 1.0f;
    case 2:
        return 2.0f;
    default:
        return 1.0f;
    }
}

uint16_t act_value_bits(uint32_t col, bool varied_act_scale) {
    if (!varied_act_scale) {
        return kFp16One;
    }
    switch ((col >> 4) & 0x3u) {
    case 0:
        return kFp16One;
    case 1:
        return kFp16Half;
    case 2:
        return kFp16Two;
    default:
        return kFp16One;
    }
}

float act_value(uint32_t col, bool varied_act_scale) {
    if (!varied_act_scale) {
        return 1.0f;
    }
    switch ((col >> 4) & 0x3u) {
    case 0:
        return 1.0f;
    case 1:
        return 0.5f;
    case 2:
        return 2.0f;
    default:
        return 1.0f;
    }
}

bool fp16_close(uint16_t got, uint16_t expected) {
    if ((got & 0x7fffu) == 0 && (expected & 0x7fffu) == 0) {
        return true;
    }
    if ((got >> 15) != (expected >> 15)) {
        return false;
    }
    const int got_mag = static_cast<int>(got & 0x7fffu);
    const int exp_mag = static_cast<int>(expected & 0x7fffu);
    return std::abs(got_mag - exp_mag) <= 1;
}

float fp16_to_fp32(uint16_t bits) {
    const uint32_t sign = (bits >> 15) & 0x1u;
    const uint32_t exp = (bits >> 10) & 0x1fu;
    const uint32_t frac = bits & 0x03ffu;

    float value = 0.0f;
    if (exp == 0) {
        value = frac == 0 ? 0.0f : std::ldexp(static_cast<float>(frac), -24);
    } else if (exp == 0x1fu) {
        value = frac == 0 ? INFINITY : NAN;
    } else {
        value = std::ldexp(1.0f + static_cast<float>(frac) / 1024.0f,
                           static_cast<int>(exp) - 15);
    }
    return sign ? -value : value;
}

struct ErrorStats {
    uint32_t checked = 0;
    uint32_t mismatches = 0;
    double sum_abs = 0.0;
    double max_abs = 0.0;
    uint32_t max_abs_row = 0;
    uint32_t max_fp16_delta = 0;
    uint32_t max_fp16_delta_row = 0;
};

uint32_t fp16_delta(uint16_t got, uint16_t expected) {
    return static_cast<uint32_t>(std::abs(static_cast<int>(got) -
                                         static_cast<int>(expected)));
}

ErrorStats compare_output_to_golden(const char* name, const char* label,
                                    const uint8_t* out,
                                    const std::vector<uint16_t>& golden,
                                    uint32_t count,
                                    uint32_t mismatch_print_limit) {
    ErrorStats stats;
    stats.checked = count;
    for (uint32_t row = 0; row < count; ++row) {
        const uint16_t got = load_u16_le(out + row * kFp16Bytes);
        const uint16_t exp = golden[row];
        const double abs_err = std::fabs(static_cast<double>(fp16_to_fp32(got)) -
                                         static_cast<double>(fp16_to_fp32(exp)));
        const uint32_t bit_delta = fp16_delta(got, exp);

        stats.sum_abs += abs_err;
        if (abs_err > stats.max_abs) {
            stats.max_abs = abs_err;
            stats.max_abs_row = row;
        }
        if (bit_delta > stats.max_fp16_delta) {
            stats.max_fp16_delta = bit_delta;
            stats.max_fp16_delta_row = row;
        }

        if (!fp16_close(got, exp)) {
            if (stats.mismatches < mismatch_print_limit) {
                std::fprintf(stderr,
                             "%s%s mismatch row=%u expected=0x%04x got=0x%04x abs_err=%.6g fp16_delta=%u\n",
                             name, label, row, exp, got, abs_err, bit_delta);
            }
            ++stats.mismatches;
        }
    }
    return stats;
}

ErrorStats compare_fp32_output_to_golden(const char* name, const char* label,
                                         const uint8_t* out,
                                         const std::vector<uint16_t>& golden,
                                         uint32_t count,
                                         uint32_t mismatch_print_limit) {
    ErrorStats stats;
    stats.checked = count;
    const float* out_f32 = reinterpret_cast<const float*>(out);
    for (uint32_t row = 0; row < count; ++row) {
        const float got = out_f32[row];
        const float exp = fp16_to_fp32(golden[row]);
        const double abs_err = std::fabs(static_cast<double>(got) -
                                         static_cast<double>(exp));

        stats.sum_abs += abs_err;
        if (abs_err > stats.max_abs) {
            stats.max_abs = abs_err;
            stats.max_abs_row = row;
        }

        if (!(abs_err <= 0.0)) {
            if (stats.mismatches < mismatch_print_limit) {
                std::fprintf(stderr,
                             "%s%s fp32 mismatch row=%u expected=%g got=%g abs_err=%.6g\n",
                             name, label, row, static_cast<double>(exp),
                             static_cast<double>(got), abs_err);
            }
            ++stats.mismatches;
        }
    }
    return stats;
}

void print_error_stats(const char* name, const char* label, const ErrorStats& stats) {
    const double mean_abs =
        stats.checked == 0 ? 0.0 : stats.sum_abs / static_cast<double>(stats.checked);
    std::printf("%s%s error: checked=%u mismatches=%u max_abs=%.6g(row=%u)"
                " mean_abs=%.6g max_fp16_delta=%u(row=%u)\n",
                name,
                label,
                stats.checked,
                stats.mismatches,
                stats.max_abs,
                stats.max_abs_row,
                mean_abs,
                stats.max_fp16_delta,
                stats.max_fp16_delta_row);
}

std::vector<uint8_t> read_binary_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.good()) {
        return {};
    }
    in.seekg(0, std::ios::end);
    const std::streamoff size = in.tellg();
    if (size <= 0) {
        return {};
    }
    in.seekg(0, std::ios::beg);
    std::vector<uint8_t> data(static_cast<size_t>(size));
    in.read(reinterpret_cast<char*>(data.data()), size);
    if (!in.good()) {
        return {};
    }
    return data;
}

std::map<std::string, std::string> read_meta_file(const std::string& path) {
    std::ifstream in(path);
    std::map<std::string, std::string> meta;
    std::string line;
    while (std::getline(in, line)) {
        const size_t pos = line.find('=');
        if (pos == std::string::npos) {
            continue;
        }
        meta[line.substr(0, pos)] = line.substr(pos + 1);
    }
    return meta;
}

uint32_t meta_u32(const std::map<std::string, std::string>& meta,
                  const char* key, uint32_t fallback = 0) {
    const auto it = meta.find(key);
    if (it == meta.end()) {
        return fallback;
    }
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(it->second.c_str(), &end, 0);
    return end != it->second.c_str() ? static_cast<uint32_t>(parsed) : fallback;
}

std::string meta_string(const std::map<std::string, std::string>& meta,
                        const char* key) {
    const auto it = meta.find(key);
    return it == meta.end() ? std::string() : it->second;
}

int8_t load_q4_signed(const uint8_t* row, uint32_t lane) {
    const uint8_t byte = row[lane / 2u];
    const uint8_t nibble = (lane & 1u) ? static_cast<uint8_t>(byte >> 4) :
                                         static_cast<uint8_t>(byte & 0x0fu);
    return static_cast<int8_t>(nibble >= 8u ? static_cast<int>(nibble) - 16 : nibble);
}

std::vector<float> make_repro_reference_f32(const uint8_t* act,
                                            const uint8_t* scale,
                                            const uint8_t* weight,
                                            uint32_t m,
                                            uint32_t k,
                                            uint32_t col_tiles) {
    std::vector<float> expected(m, 0.0f);
    const uint32_t act_bytes = col_tiles * kActTileBytes;
    for (uint32_t row = 0; row < m; ++row) {
        float accum = 0.0f;
        const uint32_t row_tile = row / kRowTileElems;
        const uint32_t row_lane = row % kRowTileElems;
        for (uint32_t col_tile = 0; col_tile < col_tiles; ++col_tile) {
            const uint32_t scale_off = (row_tile * col_tiles + col_tile) *
                                       kScaleTileBytes + row_lane * kFp16Bytes;
            const float w_scale = fp16_to_fp32(load_u16_le(scale + scale_off));
            const uint8_t* weight_row =
                weight + (row_tile * col_tiles + col_tile) * kWeightTileBytes +
                row_lane * kWeightRowBytes;
            const uint8_t* act_tile = act + col_tile * kActTileBytes;
            const uint8_t* act_scale_tile = act + act_bytes + col_tile * kActTileBytes;

            float dot = 0.0f;
            const uint32_t col_begin = col_tile * kTileElems;
            const uint32_t cols = std::min<uint32_t>(kTileElems, k - col_begin);
            for (uint32_t lane = 0; lane < cols; ++lane) {
                const float a = fp16_to_fp32(load_u16_le(act_tile + lane * kFp16Bytes));
                const float as = fp16_to_fp32(load_u16_le(act_scale_tile + lane * kFp16Bytes));
                dot += static_cast<float>(load_q4_signed(weight_row, lane)) * a * as;
            }
            accum += dot * w_scale;
        }
        expected[row] = accum;
    }
    return expected;
}

ErrorStats compare_fp32_to_reference(const char* name, const char* label,
                                     const uint8_t* out,
                                     const std::vector<float>& ref,
                                     uint32_t count,
                                     double atol,
                                     double rtol,
                                     uint32_t mismatch_print_limit) {
    ErrorStats stats;
    stats.checked = count;
    const float* got_f32 = reinterpret_cast<const float*>(out);
    for (uint32_t row = 0; row < count; ++row) {
        const double got = got_f32[row];
        const double exp = ref[row];
        const double abs_err = std::fabs(got - exp);
        stats.sum_abs += abs_err;
        if (abs_err > stats.max_abs) {
            stats.max_abs = abs_err;
            stats.max_abs_row = row;
        }
        if (abs_err > atol + rtol * std::fabs(exp)) {
            if (stats.mismatches < mismatch_print_limit) {
                std::fprintf(stderr,
                             "%s%s fp32 mismatch row=%u expected=%g got=%g abs_err=%.6g\n",
                             name, label, row, exp, got, abs_err);
            }
            ++stats.mismatches;
        }
    }
    return stats;
}

ErrorStats compare_fp32_buffers(const char* name, const char* label,
                                const uint8_t* got_buf,
                                const uint8_t* ref_buf,
                                uint32_t count,
                                double atol,
                                double rtol,
                                uint32_t mismatch_print_limit) {
    ErrorStats stats;
    stats.checked = count;
    const float* got_f32 = reinterpret_cast<const float*>(got_buf);
    const float* ref_f32 = reinterpret_cast<const float*>(ref_buf);
    for (uint32_t row = 0; row < count; ++row) {
        const double got = got_f32[row];
        const double exp = ref_f32[row];
        const double abs_err = std::fabs(got - exp);
        stats.sum_abs += abs_err;
        if (abs_err > stats.max_abs) {
            stats.max_abs = abs_err;
            stats.max_abs_row = row;
        }
        if (abs_err > atol + rtol * std::fabs(exp)) {
            if (stats.mismatches < mismatch_print_limit) {
                std::fprintf(stderr,
                             "%s%s fp32 buffer mismatch row=%u expected=%g got=%g abs_err=%.6g\n",
                             name, label, row, exp, got, abs_err);
            }
            ++stats.mismatches;
        }
    }
    return stats;
}

bool make_layout(const GemvCase& tc, Layout* layout) {
    const uint32_t row_tiles = ceil_div(tc.m, kRowTileElems);
    const uint32_t col_tiles = ceil_div(tc.k, kTileElems);

    layout->weight_bytes = row_tiles * col_tiles * kPackedWeightTileBytes;
    layout->act_bytes = col_tiles * kActTileBytes * 2u;
    layout->scale_bytes = row_tiles * col_tiles * kScaleTileBytes;
    layout->output_dma_bytes = align_up(static_cast<uint32_t>(tc.m) * kFp16Bytes, kSpmLineBytes);
    layout->weight_spm_base = tc.explicit_bases ? tc.weight_spm_base : kWeightSpmBase;
    layout->scale_spm_base = tc.explicit_bases ? tc.scale_spm_base : kScaleSpmBase;
    layout->output_spm_base = tc.explicit_bases ? tc.output_spm_base : kOutputSpmBase;
    layout->act_spm_base = tc.explicit_bases ? tc.act_spm_base : 0;

    if (layout->weight_spm_base + layout->weight_bytes > kSpmBytes ||
        layout->act_spm_base + layout->act_bytes > kActBufferBytes ||
        layout->scale_spm_base + layout->scale_bytes > kSpmBytes ||
        layout->output_spm_base + layout->output_dma_bytes > kSpmBytes) {
        return false;
    }
    if (!tc.explicit_bases) {
        if (layout->scale_spm_base + layout->scale_bytes > layout->weight_spm_base) {
            return false;
        }
    }
    if ((layout->weight_bytes % kMvinTransferAlign) ||
        (layout->act_bytes % kMvinTransferAlign) ||
        (layout->scale_bytes % kMvinTransferAlign)) {
        return false;
    }
    return true;
}

bool make_fixed_block_layout(uint16_t m, uint16_t k, FixedBlockLayout* layout) {
    const uint32_t row_tiles = ceil_div(m, kRowTileElems);
    const uint32_t col_tiles = ceil_div(k, kTileElems);

    layout->act_spm_base = 0;
    layout->act_bytes = col_tiles * kActTileBytes * 2u;
    layout->scale_spm_base = 0;
    layout->scale_bytes = row_tiles * col_tiles * kScaleTileBytes;
    layout->weight_spm_base = align_up(layout->scale_spm_base + layout->scale_bytes,
                                       kSpmLineBytes);
    layout->weight_bytes = row_tiles * col_tiles * kPackedWeightTileBytes;
    layout->output_spm_base = 0;
    layout->output_dma_bytes = align_up(static_cast<uint32_t>(m) * kFp16Bytes,
                                        kSpmLineBytes);

    if (layout->act_spm_base + layout->act_bytes > kActBufferBytes ||
        layout->scale_spm_base + layout->scale_bytes > kSpmBytes ||
        layout->weight_spm_base + layout->weight_bytes > kSpmBytes ||
        layout->output_spm_base + layout->output_dma_bytes > kSpmBytes) {
        return false;
    }
    const uint32_t packed_scale_weight_bytes =
        align_up(layout->weight_spm_base + layout->weight_bytes, kMvinTransferAlign);
    if ((layout->act_bytes % kMvinTransferAlign) ||
        (packed_scale_weight_bytes % kMvinTransferAlign)) {
        return false;
    }
    return true;
}

void fill_activation(uint8_t* dst, uint16_t k, bool varied_act_scale = false) {
    const uint32_t col_tiles = ceil_div(k, kTileElems);
    const uint32_t act_bytes = col_tiles * kActTileBytes;
    std::memset(dst, 0, act_bytes * 2u);

    for (uint32_t col_tile = 0; col_tile < col_tiles; ++col_tile) {
        for (uint32_t lane = 0; lane < kTileElems; ++lane) {
            const uint32_t col = col_tile * kTileElems + lane;
            const uint16_t bits = (col < k) ? act_value_bits(col, varied_act_scale) : kFp16Zero;
            store_u16_le(dst + col_tile * kActTileBytes + lane * kFp16Bytes, bits);
            store_u16_le(dst + act_bytes + col_tile * kActTileBytes + lane * kFp16Bytes,
                         (col < k) ? act_scale_bits(col, varied_act_scale) : kFp16Zero);
        }
    }
}

void fill_scale(uint8_t* dst, uint16_t m, uint16_t k, bool varied_scale) {
    const uint32_t row_tiles = ceil_div(m, kRowTileElems);
    const uint32_t col_tiles = ceil_div(k, kTileElems);
    std::memset(dst, 0, row_tiles * col_tiles * kScaleTileBytes);

    for (uint32_t row_tile = 0; row_tile < row_tiles; ++row_tile) {
        for (uint32_t col_tile = 0; col_tile < col_tiles; ++col_tile) {
            const uint32_t tile_base = (row_tile * col_tiles + col_tile) * kScaleTileBytes;
            for (uint32_t lane = 0; lane < kRowTileElems; ++lane) {
                const uint32_t row = row_tile * kRowTileElems + lane;
                const uint16_t bits =
                    (row < m) ? scale_bits(row, col_tile, varied_scale) : kFp16Zero;
                store_u16_le(dst + tile_base + lane * kFp16Bytes, bits);
            }
        }
    }
}

void fill_scale_arbitrary(uint8_t* dst, uint16_t m, uint16_t k) {
    const uint32_t row_tiles = ceil_div(m, kRowTileElems);
    const uint32_t col_tiles = ceil_div(k, kTileElems);
    std::memset(dst, 0, row_tiles * col_tiles * kScaleTileBytes);

    for (uint32_t row_tile = 0; row_tile < row_tiles; ++row_tile) {
        for (uint32_t col_tile = 0; col_tile < col_tiles; ++col_tile) {
            const uint32_t tile_base = (row_tile * col_tiles + col_tile) * kScaleTileBytes;
            for (uint32_t lane = 0; lane < kRowTileElems; ++lane) {
                const uint32_t row = row_tile * kRowTileElems + lane;
                const uint16_t bits = (row < m) ? arbitrary_scale_bits(row, col_tile) : kFp16Zero;
                store_u16_le(dst + tile_base + lane * kFp16Bytes, bits);
            }
        }
    }
}

void fill_scale_range(uint8_t* dst, uint32_t row_base, uint16_t rows, uint16_t k,
                      bool varied_scale) {
    const uint32_t row_tiles = ceil_div(rows, kRowTileElems);
    const uint32_t col_tiles = ceil_div(k, kTileElems);
    std::memset(dst, 0, row_tiles * col_tiles * kScaleTileBytes);

    for (uint32_t row_tile = 0; row_tile < row_tiles; ++row_tile) {
        for (uint32_t col_tile = 0; col_tile < col_tiles; ++col_tile) {
            const uint32_t tile_base = (row_tile * col_tiles + col_tile) * kScaleTileBytes;
            for (uint32_t lane = 0; lane < kRowTileElems; ++lane) {
                const uint32_t local_row = row_tile * kRowTileElems + lane;
                const uint32_t global_row = row_base + local_row;
                const uint16_t bits =
                    (local_row < rows) ? scale_bits(global_row, col_tile, varied_scale) : kFp16Zero;
                store_u16_le(dst + tile_base + lane * kFp16Bytes, bits);
            }
        }
    }
}

void fill_weight(uint8_t* dst, uint16_t m, uint16_t k,
                 bool varied_scale = false, int weight_bias = 0) {
    const uint32_t row_tiles = ceil_div(m, kRowTileElems);
    const uint32_t col_tiles = ceil_div(k, kTileElems);
    std::memset(dst, 0, row_tiles * col_tiles * kPackedWeightTileBytes);

    for (uint32_t row_tile = 0; row_tile < row_tiles; ++row_tile) {
        for (uint32_t col_tile = 0; col_tile < col_tiles; ++col_tile) {
            const uint32_t tile_idx = row_tile * col_tiles + col_tile;
            const uint32_t tile_base = tile_idx * kPackedWeightTileBytes;
            for (uint32_t lane = 0; lane < kRowTileElems; ++lane) {
                const uint32_t row = row_tile * kRowTileElems + lane;
                const uint16_t bits =
                    (row < m) ? scale_bits(row, col_tile, varied_scale) : kFp16Zero;
                store_u16_le(dst + tile_base + lane * kFp16Bytes, bits);
            }
            for (uint32_t row_lane = 0; row_lane < kRowTileElems; ++row_lane) {
                const uint32_t row = row_tile * kRowTileElems + row_lane;
                const uint32_t row_base = tile_base + kScaleTileBytes +
                                          row_lane * kWeightRowBytes;
                for (uint32_t col_lane = 0; col_lane < kTileElems; ++col_lane) {
                    const uint32_t col = col_tile * kTileElems + col_lane;
                    const uint8_t packed =
                        (row < m && col < k)
                            ? static_cast<uint8_t>(weight_value(row, col, weight_bias)) & 0x0fu
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

void fill_weight_range(uint8_t* dst, uint32_t row_base_global,
                       uint16_t block_m, uint16_t k,
                       bool varied_scale = false, int weight_bias = 0) {
    const uint32_t row_tiles = ceil_div(block_m, kRowTileElems);
    const uint32_t col_tiles = ceil_div(k, kTileElems);
    std::memset(dst, 0, row_tiles * col_tiles * kPackedWeightTileBytes);

    for (uint32_t row_tile = 0; row_tile < row_tiles; ++row_tile) {
        for (uint32_t col_tile = 0; col_tile < col_tiles; ++col_tile) {
            const uint32_t tile_idx = row_tile * col_tiles + col_tile;
            const uint32_t tile_base = tile_idx * kPackedWeightTileBytes;
            for (uint32_t lane = 0; lane < kRowTileElems; ++lane) {
                const uint32_t local_row = row_tile * kRowTileElems + lane;
                const uint32_t global_row = row_base_global + local_row;
                const uint16_t bits =
                    (local_row < block_m) ? scale_bits(global_row, col_tile, varied_scale) :
                    kFp16Zero;
                store_u16_le(dst + tile_base + lane * kFp16Bytes, bits);
            }
            for (uint32_t row_lane = 0; row_lane < kRowTileElems; ++row_lane) {
                const uint32_t local_row = row_tile * kRowTileElems + row_lane;
                const uint32_t global_row = row_base_global + local_row;
                const uint32_t row_offset = tile_base + kScaleTileBytes +
                                            row_lane * kWeightRowBytes;
                for (uint32_t col_lane = 0; col_lane < kTileElems; ++col_lane) {
                    const uint32_t col = col_tile * kTileElems + col_lane;
                    const uint8_t packed =
                        (local_row < block_m && col < k)
                            ? static_cast<uint8_t>(weight_value(global_row, col, weight_bias)) & 0x0fu
                            : 0u;
                    uint8_t& byte = dst[row_offset + col_lane / 2u];
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

void make_expected(std::vector<uint16_t>* expected, uint16_t m, uint16_t k,
                   bool varied_scale, int weight_bias = 0,
                   bool varied_act_scale = false) {
    expected->resize(m);
    for (uint32_t row = 0; row < m; ++row) {
        float accum = 0.0f;
        const uint32_t col_tiles = ceil_div(k, kTileElems);
        for (uint32_t col_tile = 0; col_tile < col_tiles; ++col_tile) {
            float sum = 0.0f;
            const uint32_t col_begin = col_tile * kTileElems;
            const uint32_t col_end = std::min<uint32_t>(col_begin + kTileElems, k);
            for (uint32_t col = col_begin; col < col_end; ++col) {
                sum += static_cast<float>(weight_value(row, col, weight_bias)) *
                       act_value(col, varied_act_scale) *
                       act_scale_value(col, varied_act_scale);
            }
            accum += sum * scale_value(row, col_tile, varied_scale);
        }
        (*expected)[row] = fp32_to_fp16_bits(accum);
    }
}

void make_expected_range(std::vector<uint16_t>* expected, uint16_t m, uint16_t k,
                         bool varied_scale, uint32_t row_base, uint32_t rows,
                         int weight_bias = 0, bool varied_act_scale = false) {
    if (expected->size() < m) {
        expected->resize(m);
    }
    for (uint32_t local_row = 0; local_row < rows; ++local_row) {
        const uint32_t row = row_base + local_row;
        float accum = 0.0f;
        const uint32_t col_tiles = ceil_div(k, kTileElems);
        for (uint32_t col_tile = 0; col_tile < col_tiles; ++col_tile) {
            float sum = 0.0f;
            const uint32_t col_begin = col_tile * kTileElems;
            const uint32_t col_end = std::min<uint32_t>(col_begin + kTileElems, k);
            for (uint32_t col = col_begin; col < col_end; ++col) {
                sum += static_cast<float>(weight_value(row, col, weight_bias)) *
                       act_value(col, varied_act_scale) *
                       act_scale_value(col, varied_act_scale);
            }
            accum += sum * scale_value(row, col_tile, varied_scale);
        }
        (*expected)[row] = fp32_to_fp16_bits(accum);
    }
}

void make_expected_range_arbitrary_scale(std::vector<uint16_t>* expected,
                                         uint16_t m, uint16_t k,
                                         uint32_t row_base, uint32_t rows,
                                         int weight_bias = 0,
                                         bool varied_act_scale = false) {
    if (expected->size() < m) {
        expected->resize(m);
    }
    for (uint32_t local_row = 0; local_row < rows; ++local_row) {
        const uint32_t row = row_base + local_row;
        float accum = 0.0f;
        const uint32_t col_tiles = ceil_div(k, kTileElems);
        for (uint32_t col_tile = 0; col_tile < col_tiles; ++col_tile) {
            float sum = 0.0f;
            const uint32_t col_begin = col_tile * kTileElems;
            const uint32_t col_end = std::min<uint32_t>(col_begin + kTileElems, k);
            for (uint32_t col = col_begin; col < col_end; ++col) {
                sum += static_cast<float>(weight_value(row, col, weight_bias)) *
                       act_value(col, varied_act_scale) *
                       act_scale_value(col, varied_act_scale);
            }
            accum += sum * arbitrary_scale_value(row, col_tile);
        }
        (*expected)[row] = fp32_to_fp16_bits(accum);
    }
}

MvinConfig make_mvin_config(void* host_ptr, uint32_t spm_addr, uint32_t bytes,
                            uint8_t input_type, uint8_t precision) {
    MvinConfig cfg = {};
    cfg.host_ptr = host_ptr;
    cfg.sram_addr = spm_addr;
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

void mvin_to_spm(void* host_ptr, uint32_t spm_addr, uint32_t bytes,
                 uint8_t input_type, uint8_t precision) {
    npu_dma_mvin(host_ptr, spm_addr, bytes - 1u, 0,
                 0, 0, precision, input_type,
                 false, false, false, 0, 0, 0);
}

void mvin_to_spm_async(uint32_t dma_id, void* host_ptr, uint32_t spm_addr,
                       uint32_t bytes, uint8_t input_type, uint8_t precision) {
    MvinConfig cfg = make_mvin_config(host_ptr, spm_addr, bytes, input_type, precision);
    npu_dma_mvin_async(dma_id, &cfg);
}

bool run_case(const GemvCase& tc) {
    Layout layout = {};
    if (!make_layout(tc, &layout)) {
        std::printf("%s: skipped, M=%u K=%u does not fit current 512KiB SPM layout\n",
                    tc.name, tc.m, tc.k);
        return true;
    }

    NpuBuffer weight(layout.weight_bytes);
    NpuBuffer act(layout.act_bytes);
    NpuBuffer scale(layout.scale_bytes);
    NpuBuffer out(layout.output_dma_bytes);
    if (!weight.ptr || !act.ptr || !scale.ptr || !out.ptr) {
        std::fprintf(stderr, "%s: npu_mem_alloc failed\n", tc.name);
        return false;
    }

    fill_weight(weight.data(), tc.m, tc.k, tc.varied_scale);
    fill_activation(act.data(), tc.k, tc.varied_act_scale);
    fill_scale(scale.data(), tc.m, tc.k, tc.varied_scale);
    std::memset(out.data(), 0xa5, layout.output_dma_bytes);

    std::vector<uint16_t> expected;
    make_expected(&expected, tc.m, tc.k, tc.varied_scale, 0, tc.varied_act_scale);

    npu_reset();
    mvin_to_spm(weight.ptr, layout.weight_spm_base, layout.weight_bytes, kInputTypeWeight, 1);
    mvin_to_spm(act.ptr, layout.act_spm_base, layout.act_bytes, kInputTypeAct, 2);
    mvin_to_spm(scale.ptr, layout.scale_spm_base, layout.scale_bytes, kInputTypeData, 2);

    npu_matvec_run(layout.weight_spm_base,
                   layout.act_spm_base,
                   tc.k,
                   tc.m,
                   static_cast<uint16_t>(layout.output_spm_base),
                   static_cast<uint16_t>(layout.weight_spm_base));

    npu_dma_mvout(out.ptr, layout.output_spm_base, 0, tc.m - 1u,
                  1, 1, 1, 1, false, false, 0, 0);

    const ErrorStats stats =
        compare_output_to_golden(tc.name, "", out.data(), expected, tc.m, 16);
    print_error_stats(tc.name, "", stats);
    if (stats.mismatches != 0) {
        std::fprintf(stderr, "%s: failed mismatches=%u/%u\n",
                     tc.name, stats.mismatches, tc.m);
        return false;
    }

    std::printf("%s: ok\n", tc.name);
    return true;
}

bool run_fixed_block_cmd_case(const char* name, uint16_t m, uint16_t k,
                              bool varied_scale, bool varied_act_scale = false) {
    const uint32_t total_row_tiles = ceil_div(m, kRowTileElems);
    uint32_t max_block_row_tiles = 0;

    for (uint32_t row_tiles = total_row_tiles; row_tiles >= 1; --row_tiles) {
        FixedBlockLayout candidate = {};
        const uint32_t candidate_rows =
            std::min<uint32_t>(row_tiles * kRowTileElems, m);
        if (make_fixed_block_layout(static_cast<uint16_t>(candidate_rows), k,
                                    &candidate)) {
            max_block_row_tiles = row_tiles;
            break;
        }
        if (row_tiles == 1) {
            break;
        }
    }

    if (max_block_row_tiles == 0) {
        std::printf("%s: skipped fixed block command, M=%u K=%u cannot fit one row tile\n",
                    name, m, k);
        return true;
    }

    const uint32_t max_block_rows = max_block_row_tiles * kRowTileElems;
    FixedBlockLayout first_layout = {};
    make_fixed_block_layout(
        static_cast<uint16_t>(std::min<uint32_t>(max_block_rows, m)),
        k,
        &first_layout);

    const uint32_t output_bytes =
        align_up(static_cast<uint32_t>(m) * kFp16Bytes, kSpmLineBytes);
    NpuBuffer act(first_layout.act_bytes);
    NpuBuffer out(output_bytes);
    if (!act.ptr || !out.ptr) {
        std::fprintf(stderr, "%s: npu_mem_alloc failed\n", name);
        return false;
    }

    fill_activation(act.data(), k, varied_act_scale);
    std::memset(out.data(), 0xa5, output_bytes);

    std::vector<uint16_t> expected(m, kFp16Zero);

    std::printf("%s: fixed block MATVEC M=%u K=%u max_block_rows=%u\n",
                name, m, k, max_block_rows);

    npu_reset();
    mvin_to_spm(act.ptr, first_layout.act_spm_base, first_layout.act_bytes, kInputTypeAct, 2);

    for (uint32_t row_base = 0; row_base < m; row_base += max_block_rows) {
        const uint32_t block_rows_u32 =
            std::min<uint32_t>(max_block_rows, static_cast<uint32_t>(m) - row_base);
        const uint16_t block_rows = static_cast<uint16_t>(block_rows_u32);
        FixedBlockLayout layout = {};
        if (!make_fixed_block_layout(block_rows, k, &layout)) {
            std::fprintf(stderr,
                         "%s: fixed block layout failed row_base=%u rows=%u\n",
                         name, row_base, block_rows_u32);
            return false;
        }

        const uint32_t packed_bytes =
            align_up(layout.weight_spm_base + layout.weight_bytes, kMvinTransferAlign);
        NpuBuffer packed_scale_weight(packed_bytes);
        if (!packed_scale_weight.ptr) {
            std::fprintf(stderr, "%s: npu_mem_alloc failed for block\n", name);
            return false;
        }

        std::memset(packed_scale_weight.data(), 0, packed_bytes);
        fill_scale_range(packed_scale_weight.data() + layout.scale_spm_base,
                         row_base, block_rows, k, varied_scale);
        fill_weight_range(packed_scale_weight.data() + layout.weight_spm_base,
                          row_base, block_rows, k, varied_scale);
        make_expected_range(&expected, m, k, varied_scale, row_base, block_rows,
                            0, varied_act_scale);

        mvin_to_spm(packed_scale_weight.ptr, 0, packed_bytes, kInputTypeWeight, 1);

        npu_matvec_run(layout.weight_spm_base,
                       layout.act_spm_base,
                       k,
                       block_rows,
                       static_cast<uint16_t>(row_base * kFp16Bytes),
                       static_cast<uint16_t>(layout.weight_spm_base));
    }

    npu_dma_mvout(out.data(),
                  first_layout.output_spm_base,
                  0,
                  m - 1u,
                  1,
                  1,
                  1,
                  1,
                  false,
                  false,
                  0,
                  0);

    const ErrorStats stats =
        compare_output_to_golden(name, " fixed", out.data(), expected, m, 16);
    print_error_stats(name, " fixed", stats);
    if (stats.mismatches != 0) {
        std::fprintf(stderr, "%s: fixed block MATVEC failed mismatches=%u/%u\n",
                     name, stats.mismatches, m);
        return false;
    }

    std::printf("%s: fixed block MATVEC ok\n", name);
    return true;
}

uint64_t measure_cpu_reference_ns(uint16_t m, uint16_t k, bool varied_scale, int iterations,
                                  bool varied_act_scale = false) {
    std::vector<uint16_t> expected;
    uint32_t checksum = 0;
    const auto begin = Clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        make_expected(&expected, m, k, varied_scale, iter & 1, varied_act_scale);
        for (uint16_t value : expected) {
            checksum = (checksum << 1) ^ value;
        }
    }
    const auto end = Clock::now();
    if (checksum == 0xFFFFFFFFu) {
        std::printf("cpu checksum guard=%u\n", checksum);
    }
    return elapsed_ns(begin, end);
}

bool make_pingpong_layout(uint16_t m, uint16_t k, Layout* layout) {
    const uint32_t row_tiles = ceil_div(m, kRowTileElems);
    const uint32_t col_tiles = ceil_div(k, kTileElems);

    layout->weight_bytes = row_tiles * col_tiles * kPackedWeightTileBytes;
    layout->act_bytes = col_tiles * kActTileBytes * 2u;
    layout->scale_bytes = row_tiles * col_tiles * kScaleTileBytes;
    layout->output_dma_bytes = align_up(static_cast<uint32_t>(m) * kFp16Bytes, kSpmLineBytes);
    layout->weight_spm_base = kPingWeightSpmBase;
    layout->act_spm_base = kPingActSpmBase;
    layout->scale_spm_base = kScaleSpmBase;
    layout->output_spm_base = kOutputSpmBase;

    if (layout->act_spm_base + layout->act_bytes > kActBufferBytes ||
        layout->scale_spm_base + layout->scale_bytes > kPingWeightSpmBase ||
        layout->output_spm_base + layout->output_dma_bytes > kSpmBytes ||
        kPingWeightSpmBase + layout->weight_bytes > kPongWeightSpmBase ||
        kPongWeightSpmBase + layout->weight_bytes > kSpmBytes) {
        return false;
    }
    if ((layout->weight_bytes % kMvinTransferAlign) ||
        (layout->act_bytes % kMvinTransferAlign) ||
        (layout->scale_bytes % kMvinTransferAlign)) {
        return false;
    }
    return true;
}

bool run_api_pingpong_case(const char* name, uint16_t m, uint16_t k,
                           bool varied_act_scale = false) {
    Layout layout = {};
    if (!make_pingpong_layout(m, k, &layout)) {
        std::printf("%s: skipped API pingpong, M=%u K=%u cannot fit ping/pong SPM layout\n",
                    name, m, k);
        return true;
    }

    NpuBuffer weight(layout.weight_bytes);
    NpuBuffer act(layout.act_bytes);
    NpuBuffer scale(layout.scale_bytes);
    NpuBuffer out(layout.output_dma_bytes);
    if (!weight.ptr || !act.ptr || !scale.ptr || !out.ptr) {
        std::fprintf(stderr, "%s: npu_mem_alloc failed\n", name);
        return false;
    }

    fill_weight(weight.data(), m, k, true);
    fill_activation(act.data(), k, varied_act_scale);
    fill_scale(scale.data(), m, k, true);
    std::memset(out.data(), 0xa5, layout.output_dma_bytes);

    std::vector<uint16_t> expected;
    make_expected(&expected, m, k, true, 0, varied_act_scale);

    npu_reset();
    npu_gemv_pingpong_run(act.ptr, scale.ptr, weight.ptr, out.ptr, m, k);

    const ErrorStats stats =
        compare_output_to_golden(name, " API pingpong", out.data(), expected, m, 16);
    print_error_stats(name, " API pingpong", stats);
    if (stats.mismatches != 0) {
        std::fprintf(stderr, "%s: API pingpong failed mismatches=%u/%u\n",
                     name, stats.mismatches, m);
        return false;
    }

    std::printf("%s: API pingpong ok\n", name);
    return true;
}

bool run_api_pingpong_fp32_mvout_case(const char* name, uint16_t m, uint16_t k,
                                      bool varied_act_scale = false) {
    Layout layout = {};
    if (!make_pingpong_layout(m, k, &layout)) {
        std::printf("%s: skipped API pingpong FP32 MVOUT, M=%u K=%u cannot fit ping/pong SPM layout\n",
                    name, m, k);
        return true;
    }

    const uint32_t output_bytes =
        align_up(static_cast<uint32_t>(m) * static_cast<uint32_t>(sizeof(float)),
                 kSpmLineBytes);
    NpuBuffer weight(layout.weight_bytes);
    NpuBuffer act(layout.act_bytes);
    NpuBuffer scale(layout.scale_bytes);
    NpuBuffer out(output_bytes);
    if (!weight.ptr || !act.ptr || !scale.ptr || !out.ptr) {
        std::fprintf(stderr, "%s: npu_mem_alloc failed\n", name);
        return false;
    }

    fill_weight(weight.data(), m, k, true);
    fill_activation(act.data(), k, varied_act_scale);
    fill_scale(scale.data(), m, k, true);
    std::memset(out.data(), 0xa5, output_bytes);

    std::vector<uint16_t> expected;
    make_expected(&expected, m, k, true, 0, varied_act_scale);

    npu_reset();
    npu_gemv_pingpong_mode_precision_run(
        act.ptr, scale.ptr, weight.ptr, out.ptr, m, k, 0, 3);

    const ErrorStats stats =
        compare_fp32_output_to_golden(name, " API pingpong fp32-mvout",
                                      out.data(), expected, m, 16);
    print_error_stats(name, " API pingpong fp32-mvout", stats);
    if (stats.mismatches != 0) {
        std::fprintf(stderr, "%s: API pingpong FP32 MVOUT failed mismatches=%u/%u\n",
                     name, stats.mismatches, m);
        return false;
    }

    std::printf("%s: API pingpong FP32 MVOUT ok\n", name);
    return true;
}

bool run_raw_fp16_to_fp32_mvout_case(const char* name, uint16_t count) {
    const uint32_t spm_bytes =
        align_up(static_cast<uint32_t>(count) * kFp16Bytes, kSpmLineBytes);
    const uint32_t out_bytes =
        align_up(static_cast<uint32_t>(count) * static_cast<uint32_t>(sizeof(float)),
                 kSpmLineBytes);
    NpuBuffer input(spm_bytes);
    NpuBuffer out(out_bytes);
    if (!input.ptr || !out.ptr) {
        std::fprintf(stderr, "%s: npu_mem_alloc failed\n", name);
        return false;
    }

    static const float seeds[] = {
        0.0f, 0.0332218f, 0.061526f, 0.0809402f, 0.0952361f,
        -0.0442104f, -0.230058f, -0.365023f, -0.699097f,
        -1.00533f, -1.90323f, 1.25977f, 3.29924f, 5.31098f,
    };
    std::memset(input.data(), 0, spm_bytes);
    std::memset(out.data(), 0xa5, out_bytes);
    for (uint32_t row = 0; row < count; ++row) {
        const float base = seeds[row % (sizeof(seeds) / sizeof(seeds[0]))];
        const float value =
            base + static_cast<float>(static_cast<int>(row % 17) - 8) * 0.001953125f;
        const uint16_t bits = fp32_to_fp16_bits(value);
        store_u16_le(input.data() + row * kFp16Bytes, bits);
    }

    npu_reset();
    mvin_to_spm(input.ptr, 0, spm_bytes, kInputTypeData, 2);
    npu_dma_mvout(out.data(),
                  0,
                  0,
                  count - 1u,
                  1,
                  1,
                  3,
                  1,
                  false,
                  false,
                  0,
                  0);

    ErrorStats stats;
    stats.checked = count;
    const float* out_f32 = reinterpret_cast<const float*>(out.data());
    for (uint32_t row = 0; row < count; ++row) {
        const uint16_t bits = load_u16_le(input.data() + row * kFp16Bytes);
        const float exp = fp16_to_fp32(bits);
        const float got = out_f32[row];
        const double abs_err = std::fabs(static_cast<double>(got) -
                                         static_cast<double>(exp));
        stats.sum_abs += abs_err;
        if (abs_err > stats.max_abs) {
            stats.max_abs = abs_err;
            stats.max_abs_row = row;
        }
        if (!(abs_err <= 0.0)) {
            if (stats.mismatches < 16) {
                std::fprintf(stderr,
                             "%s raw fp32-mvout mismatch row=%u bits=0x%04x expected=%g got=%g abs_err=%.6g\n",
                             name, row, bits, static_cast<double>(exp),
                             static_cast<double>(got), abs_err);
            }
            ++stats.mismatches;
        }
    }
    print_error_stats(name, " raw fp16->fp32-mvout", stats);
    if (stats.mismatches != 0) {
        std::fprintf(stderr, "%s: raw FP16->FP32 MVOUT failed mismatches=%u/%u\n",
                     name, stats.mismatches, count);
        return false;
    }

    std::printf("%s: raw FP16->FP32 MVOUT ok\n", name);
    return true;
}

bool run_pingpong_perf_case(const char* name, uint16_t m, uint16_t k, int iterations,
                            bool varied_act_scale = false) {
    Layout layout = {};
    if (!make_pingpong_layout(m, k, &layout)) {
        std::printf("%s: skipped pingpong perf, M=%u K=%u cannot fit ping/pong SPM layout\n",
                    name, m, k);
        return true;
    }

    NpuBuffer weight_a(layout.weight_bytes);
    NpuBuffer weight_b(layout.weight_bytes);
    NpuBuffer act(layout.act_bytes);
    NpuBuffer scale(layout.scale_bytes);
    NpuBuffer out(layout.output_dma_bytes);
    if (!weight_a.ptr || !weight_b.ptr || !act.ptr || !scale.ptr || !out.ptr) {
        std::fprintf(stderr, "%s: npu_mem_alloc failed\n", name);
        return false;
    }

    fill_weight(weight_a.data(), m, k, true, 0);
    fill_weight(weight_b.data(), m, k, true, 1);
    fill_activation(act.data(), k, varied_act_scale);
    fill_scale(scale.data(), m, k, true);
    std::memset(out.data(), 0xa5, layout.output_dma_bytes);

    std::printf("%s: pingpong perf M=%u K=%u iters=%d pingW=0x%05x pongW=0x%05x"
                " A=0x%04x S=0x%04x O=0x%04x Wbytes=%u\n",
                name,
                m,
                k,
                iterations,
                kPingWeightSpmBase,
                kPongWeightSpmBase,
                layout.act_spm_base,
                layout.scale_spm_base,
                layout.output_spm_base,
                layout.weight_bytes);

    npu_reset();
    uint64_t preload_ns = 0;
    auto t0 = Clock::now();
    mvin_to_spm(act.ptr, layout.act_spm_base, layout.act_bytes, kInputTypeAct, 2);
    mvin_to_spm(scale.ptr, layout.scale_spm_base, layout.scale_bytes, kInputTypeData, 2);
    mvin_to_spm(weight_a.ptr, kPingWeightSpmBase, layout.weight_bytes, kInputTypeWeight, 1);
    preload_ns = elapsed_ns(t0, Clock::now());

    void* weight_ptrs[2] = {weight_a.ptr, weight_b.ptr};
    const uint32_t weight_spm[2] = {kPingWeightSpmBase, kPongWeightSpmBase};

    uint64_t gemv_ns = 0;
    uint64_t prefetch_window_ns = 0;
    uint64_t post_compute_wait_ns = 0;
    uint64_t async_mvin_bytes = 0;
    const uint32_t async_dma_id = static_cast<uint32_t>(get_env_int("GEMV_ASYNC_DMA_ID", 0) & 1);
    const bool disable_async_prefetch = get_env_flag("GEMV_DISABLE_ASYNC_PREFETCH");
    std::printf("  async_dma_id=%u disable_async_prefetch=%d\n",
                async_dma_id, disable_async_prefetch ? 1 : 0);

    const auto hw_begin = Clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        const int cur = iter & 1;
        const int next = (iter + 1) & 1;
        Clock::time_point prefetch_begin{};
        bool prefetch_started = false;

        if (iter + 1 < iterations) {
            prefetch_begin = Clock::now();
            if (disable_async_prefetch) {
                mvin_to_spm(weight_ptrs[next], weight_spm[next],
                            layout.weight_bytes, kInputTypeWeight, 1);
            } else {
                mvin_to_spm_async(async_dma_id, weight_ptrs[next], weight_spm[next],
                                  layout.weight_bytes, kInputTypeWeight, 1);
                prefetch_started = true;
                async_mvin_bytes += layout.weight_bytes;
            }
        }

        const auto gemv_begin = Clock::now();
        npu_matvec_run(weight_spm[cur],
                       layout.act_spm_base,
                       k,
                       m,
                       static_cast<uint16_t>(layout.output_spm_base),
                       static_cast<uint16_t>(layout.weight_spm_base));
        const auto gemv_end = Clock::now();
        gemv_ns += elapsed_ns(gemv_begin, gemv_end);

        if (prefetch_started) {
            const auto wait_begin = Clock::now();
            npu_dma_wait_mvin(1u << async_dma_id);
            const auto wait_end = Clock::now();
            post_compute_wait_ns += elapsed_ns(wait_begin, wait_end);
            prefetch_window_ns += elapsed_ns(prefetch_begin, wait_end);
        }
    }

    const auto mvout_begin = Clock::now();
    npu_dma_mvout(out.ptr, layout.output_spm_base, 0, m - 1u,
                  1, 1, 1, 1, false, false, 0, 0);
    const uint64_t mvout_ns = elapsed_ns(mvout_begin, Clock::now());
    const uint64_t hw_total_ns = elapsed_ns(hw_begin, Clock::now());

    std::vector<uint16_t> expected;
    make_expected(&expected, m, k, true, (iterations - 1) & 1, varied_act_scale);
    const ErrorStats stats =
        compare_output_to_golden(name, " pingpong", out.data(), expected, m, 8);
    print_error_stats(name, " pingpong", stats);
    if (stats.mismatches != 0) {
        std::fprintf(stderr, "%s: pingpong perf failed mismatches=%u/%u\n",
                     name, stats.mismatches, m);
        return false;
    }

    const uint64_t cpu_ns = measure_cpu_reference_ns(m, k, true, iterations, varied_act_scale);
    const uint64_t preload_bytes = layout.act_bytes + layout.scale_bytes + layout.weight_bytes;
    const uint64_t mvout_bytes = layout.output_dma_bytes;
    const uint64_t total_ddr_bytes = preload_bytes + async_mvin_bytes + mvout_bytes;
    const double ddr_peak_gbps = get_env_double("KV260_DDR_PEAK_GBPS", kDefaultDdrPeakGBps);
    const double async_bw_gbps = bytes_per_ns_to_gbps(async_mvin_bytes, prefetch_window_ns);
    const double total_bw_gbps = bytes_per_ns_to_gbps(total_ddr_bytes, hw_total_ns);

    std::printf("%s perf:\n", name);
    std::printf("  gemv_compute_total_ms=%.3f avg_ms=%.3f\n",
                ns_to_ms(gemv_ns), ns_to_ms(gemv_ns) / iterations);
    std::printf("  cpu_compute_total_ms=%.3f avg_ms=%.3f speedup=%.2fx\n",
                ns_to_ms(cpu_ns), ns_to_ms(cpu_ns) / iterations,
                gemv_ns ? static_cast<double>(cpu_ns) / static_cast<double>(gemv_ns) : 0.0);
    std::printf("  ddr_preload_ms=%.3f mvout_ms=%.3f async_prefetch_window_ms=%.3f"
                " post_compute_wait_ms=%.3f\n",
                ns_to_ms(preload_ns), ns_to_ms(mvout_ns),
                ns_to_ms(prefetch_window_ns), ns_to_ms(post_compute_wait_ns));
    std::printf("  ddr_async_mvin_bytes=%llu observed_bw=%.3fGB/s util=%.2f%%"
                " peak=%.3fGB/s\n",
                static_cast<unsigned long long>(async_mvin_bytes),
                async_bw_gbps,
                ddr_peak_gbps > 0.0 ? async_bw_gbps * 100.0 / ddr_peak_gbps : 0.0,
                ddr_peak_gbps);
    std::printf("  ddr_total_bytes=%llu observed_end_to_end_bw=%.3fGB/s util=%.2f%%\n",
                static_cast<unsigned long long>(total_ddr_bytes),
                total_bw_gbps,
                ddr_peak_gbps > 0.0 ? total_bw_gbps * 100.0 / ddr_peak_gbps : 0.0);
    std::printf("%s: pingpong perf ok\n", name);
    return true;
}

bool make_blocked_layout(uint16_t m, uint16_t k, Layout* layout,
                         uint32_t* max_block_rows) {
    const uint32_t col_tiles = ceil_div(k, kTileElems);
    const uint32_t total_row_tiles = ceil_div(m, kRowTileElems);
    const uint32_t ping_capacity = kPongWeightSpmBase - kPingWeightSpmBase;
    const uint32_t pong_capacity = kSpmBytes - kPongWeightSpmBase;
    const uint32_t weight_capacity = std::min(ping_capacity, pong_capacity);
    const uint32_t bytes_per_row_tile = col_tiles * kPackedWeightTileBytes;
    const uint32_t max_row_tiles =
        bytes_per_row_tile == 0 ? 0 : std::min(total_row_tiles, weight_capacity / bytes_per_row_tile);

    if (max_row_tiles == 0) {
        return false;
    }

    layout->weight_bytes = max_row_tiles * bytes_per_row_tile;
    layout->act_bytes = col_tiles * kActTileBytes * 2u;
    layout->scale_bytes = total_row_tiles * col_tiles * kScaleTileBytes;
    layout->output_dma_bytes = align_up(max_row_tiles * kRowTileElems * kFp16Bytes, kSpmLineBytes);
    layout->weight_spm_base = kPingWeightSpmBase;
    layout->act_spm_base = kPingActSpmBase;
    layout->scale_spm_base = kScaleSpmBase;
    layout->output_spm_base = kOutputSpmBase;

    if (layout->act_spm_base + layout->act_bytes > kActBufferBytes ||
        layout->scale_spm_base + layout->scale_bytes > kPingWeightSpmBase ||
        layout->output_spm_base + layout->output_dma_bytes > kSpmBytes) {
        return false;
    }
    if ((layout->act_bytes % kMvinTransferAlign) ||
        (layout->scale_bytes % kMvinTransferAlign)) {
        return false;
    }

    *max_block_rows = max_row_tiles * kRowTileElems;
    return true;
}

bool run_m_blocked_pingpong_case(const char* name, uint16_t m, uint16_t k,
                                 bool varied_act_scale = false) {
    Layout layout = {};
    uint32_t max_block_rows = 0;
    if (!make_blocked_layout(m, k, &layout, &max_block_rows)) {
        std::printf("%s: skipped M-blocked pingpong, M=%u K=%u cannot fit SPM layout\n",
                    name, m, k);
        return true;
    }

    const uint32_t col_tiles = ceil_div(k, kTileElems);
    const uint32_t bytes_per_row_tile = col_tiles * kPackedWeightTileBytes;
    std::vector<BlockPlan> blocks;
    for (uint32_t row_base = 0; row_base < m; row_base += max_block_rows) {
        const uint32_t rows = std::min<uint32_t>(max_block_rows, m - row_base);
        const uint32_t row_tiles = ceil_div(rows, kRowTileElems);
        const uint32_t weight_bytes = row_tiles * bytes_per_row_tile;
        const uint32_t spm_base = (blocks.size() & 1u) ? kPongWeightSpmBase : kPingWeightSpmBase;
        blocks.emplace_back(row_base, rows, row_tiles, weight_bytes, spm_base);
        if (!blocks.back().weight.ptr) {
            std::fprintf(stderr, "%s: npu_mem_alloc failed for block weight\n", name);
            return false;
        }
        fill_weight_range(blocks.back().weight.data(), row_base,
                          static_cast<uint16_t>(rows), k, true, 0);
    }

    NpuBuffer act(layout.act_bytes);
    NpuBuffer scale(layout.scale_bytes);
    NpuBuffer out(align_up(static_cast<uint32_t>(m) * kFp16Bytes, kSpmLineBytes));
    if (!act.ptr || !scale.ptr || !out.ptr) {
        std::fprintf(stderr, "%s: npu_mem_alloc failed\n", name);
        return false;
    }

    fill_activation(act.data(), k, varied_act_scale);
    fill_scale(scale.data(), m, k, true);
    std::memset(out.data(), 0xa5, out.bytes);

    std::vector<uint16_t> expected;
    expected.resize(m);
    for (const BlockPlan& block : blocks) {
        make_expected_range(&expected, m, k, true, block.row_base, block.rows, 0,
                            varied_act_scale);
    }

    std::printf("%s: M-blocked pingpong M=%u K=%u blocks=%zu max_block_rows=%u"
                " act=0x%04x/%u scale=0x%04x/%u out=0x%04x pingW=0x%05x pongW=0x%05x\n",
                name,
                m,
                k,
                blocks.size(),
                max_block_rows,
                layout.act_spm_base,
                layout.act_bytes,
                layout.scale_spm_base,
                layout.scale_bytes,
                layout.output_spm_base,
                kPingWeightSpmBase,
                kPongWeightSpmBase);

    npu_reset();
    const auto preload_begin = Clock::now();
    mvin_to_spm(act.ptr, layout.act_spm_base, layout.act_bytes, kInputTypeAct, 2);
    mvin_to_spm(scale.ptr, layout.scale_spm_base, layout.scale_bytes, kInputTypeData, 2);
    mvin_to_spm(blocks[0].weight.ptr, blocks[0].weight_spm_base,
                blocks[0].weight_bytes, kInputTypeWeight, 1);
    const uint64_t preload_ns = elapsed_ns(preload_begin, Clock::now());

    uint64_t gemv_ns = 0;
    uint64_t mvout_ns = 0;
    uint64_t prefetch_window_ns = 0;
    uint64_t post_compute_wait_ns = 0;
    uint64_t hidden_prefetch_ns = 0;
    uint64_t async_mvin_bytes = 0;
    const uint32_t async_dma_id = static_cast<uint32_t>(get_env_int("GEMV_ASYNC_DMA_ID", 0) & 1);
    const bool disable_async_prefetch = get_env_flag("GEMV_DISABLE_ASYNC_PREFETCH");
    std::printf("  async_dma_id=%u disable_async_prefetch=%d\n",
                async_dma_id, disable_async_prefetch ? 1 : 0);

    const auto hw_begin = Clock::now();
    for (size_t i = 0; i < blocks.size(); ++i) {
        BlockPlan& cur = blocks[i];
        bool prefetch_started = false;
        Clock::time_point prefetch_begin{};
        if (i + 1 < blocks.size()) {
            BlockPlan& next = blocks[i + 1];
            prefetch_begin = Clock::now();
            if (disable_async_prefetch) {
                mvin_to_spm(next.weight.ptr, next.weight_spm_base,
                            next.weight_bytes, kInputTypeWeight, 1);
            } else {
                mvin_to_spm_async(async_dma_id, next.weight.ptr, next.weight_spm_base,
                                  next.weight_bytes, kInputTypeWeight, 1);
                async_mvin_bytes += next.weight_bytes;
                prefetch_started = true;
            }
        }

        const uint16_t scale_addr = static_cast<uint16_t>(cur.weight_spm_base);
        const auto gemv_begin = Clock::now();
        npu_matvec_run(cur.weight_spm_base,
                       layout.act_spm_base,
                       k,
                       static_cast<uint16_t>(cur.rows),
                       static_cast<uint16_t>(layout.output_spm_base),
                       scale_addr);
        const auto gemv_end = Clock::now();
        const uint64_t this_gemv_ns = elapsed_ns(gemv_begin, gemv_end);
        gemv_ns += this_gemv_ns;

        if (prefetch_started) {
            const auto wait_begin = Clock::now();
            npu_dma_wait_mvin(1u << async_dma_id);
            const auto wait_end = Clock::now();
            const uint64_t wait_ns = elapsed_ns(wait_begin, wait_end);
            const uint64_t total_prefetch_ns = elapsed_ns(prefetch_begin, wait_end);
            post_compute_wait_ns += wait_ns;
            prefetch_window_ns += total_prefetch_ns;
            hidden_prefetch_ns +=
                (total_prefetch_ns > wait_ns) ? (total_prefetch_ns - wait_ns) : 0;
        }

        const auto mvout_begin = Clock::now();
        npu_dma_mvout(out.data() + cur.row_base * kFp16Bytes,
                      layout.output_spm_base,
                      0,
                      cur.rows - 1u,
                      1,
                      1,
                      1,
                      1,
                      false,
                      false,
                      0,
                      0);
        mvout_ns += elapsed_ns(mvout_begin, Clock::now());
    }
    const uint64_t hw_total_ns = elapsed_ns(hw_begin, Clock::now());

    const ErrorStats stats =
        compare_output_to_golden(name, " M-block", out.data(), expected, m, 16);
    print_error_stats(name, " M-block", stats);
    if (stats.mismatches != 0) {
        std::fprintf(stderr, "%s: M-blocked pingpong failed mismatches=%u/%u\n",
                     name, stats.mismatches, m);
        return false;
    }

    const uint64_t cpu_ns = measure_cpu_reference_ns(m, k, true, 1, varied_act_scale);
    uint64_t total_weight_bytes = 0;
    for (const BlockPlan& block : blocks) {
        total_weight_bytes += block.weight_bytes;
    }
    const uint64_t total_ddr_bytes =
        layout.act_bytes + layout.scale_bytes + total_weight_bytes +
        align_up(static_cast<uint32_t>(m) * kFp16Bytes, kSpmLineBytes);
    const double ddr_peak_gbps = get_env_double("KV260_DDR_PEAK_GBPS", kDefaultDdrPeakGBps);
    const double async_bw_gbps = bytes_per_ns_to_gbps(async_mvin_bytes, prefetch_window_ns);
    const double total_bw_gbps = bytes_per_ns_to_gbps(total_ddr_bytes, hw_total_ns);

    std::printf("%s perf:\n", name);
    std::printf("  gemv_compute_total_ms=%.3f avg_block_ms=%.3f blocks=%zu\n",
                ns_to_ms(gemv_ns), ns_to_ms(gemv_ns) / blocks.size(), blocks.size());
    std::printf("  cpu_compute_total_ms=%.3f speedup=%.2fx\n",
                ns_to_ms(cpu_ns),
                gemv_ns ? static_cast<double>(cpu_ns) / static_cast<double>(gemv_ns) : 0.0);
    std::printf("  ddr_preload_ms=%.3f mvout_total_ms=%.3f async_prefetch_window_ms=%.3f"
                " post_compute_wait_ms=%.3f hidden_prefetch_ms=%.3f\n",
                ns_to_ms(preload_ns),
                ns_to_ms(mvout_ns),
                ns_to_ms(prefetch_window_ns),
                ns_to_ms(post_compute_wait_ns),
                ns_to_ms(hidden_prefetch_ns));
    std::printf("  ddr_async_mvin_bytes=%llu observed_bw=%.3fGB/s util=%.2f%%"
                " peak=%.3fGB/s\n",
                static_cast<unsigned long long>(async_mvin_bytes),
                async_bw_gbps,
                ddr_peak_gbps > 0.0 ? async_bw_gbps * 100.0 / ddr_peak_gbps : 0.0,
                ddr_peak_gbps);
    std::printf("  ddr_total_bytes=%llu observed_end_to_end_bw=%.3fGB/s util=%.2f%%\n",
                static_cast<unsigned long long>(total_ddr_bytes),
                total_bw_gbps,
                ddr_peak_gbps > 0.0 ? total_bw_gbps * 100.0 / ddr_peak_gbps : 0.0);
    std::printf("%s: M-blocked pingpong ok\n", name);
    return true;
}

bool run_m_blocked_pingpong_offset_output_case(const char* name, uint16_t m, uint16_t k,
                                               bool varied_act_scale = false,
                                               uint8_t final_output_precision = 1,
                                               bool arbitrary_scale = false) {
    if (final_output_precision != 1 && final_output_precision != 3) {
        std::fprintf(stderr, "%s: unsupported final MVOUT precision=%u\n",
                     name, final_output_precision);
        return false;
    }
    Layout layout = {};
    uint32_t max_block_rows = 0;
    if (!make_blocked_layout(m, k, &layout, &max_block_rows)) {
        std::printf("%s: skipped offset-output pingpong, M=%u K=%u cannot fit SPM layout\n",
                    name, m, k);
        return true;
    }

    const uint32_t full_output_spm_bytes =
        align_up(static_cast<uint32_t>(m) * kFp16Bytes, kSpmLineBytes);
    const uint32_t host_elem_bytes = (final_output_precision == 3) ? sizeof(float) : kFp16Bytes;
    const uint32_t full_output_host_bytes =
        align_up(static_cast<uint32_t>(m) * host_elem_bytes, kSpmLineBytes);
    if (layout.output_spm_base + full_output_spm_bytes > kSpmBytes) {
        std::printf("%s: skipped offset-output pingpong, full output exceeds SPM"
                    " out_base=0x%04x bytes=%u\n",
                    name, layout.output_spm_base, full_output_spm_bytes);
        return true;
    }

    const uint32_t col_tiles = ceil_div(k, kTileElems);
    const uint32_t bytes_per_row_tile = col_tiles * kPackedWeightTileBytes;
    std::vector<BlockPlan> blocks;
    for (uint32_t row_base = 0; row_base < m; row_base += max_block_rows) {
        const uint32_t rows = std::min<uint32_t>(max_block_rows, m - row_base);
        const uint32_t row_tiles = ceil_div(rows, kRowTileElems);
        const uint32_t weight_bytes = row_tiles * bytes_per_row_tile;
        const uint32_t spm_base = (blocks.size() & 1u) ? kPongWeightSpmBase : kPingWeightSpmBase;
        blocks.emplace_back(row_base, rows, row_tiles, weight_bytes, spm_base);
        if (!blocks.back().weight.ptr) {
            std::fprintf(stderr, "%s: npu_mem_alloc failed for block weight\n", name);
            return false;
        }
        fill_weight_range(blocks.back().weight.data(), row_base,
                          static_cast<uint16_t>(rows), k, true, 0);
    }

    NpuBuffer act(layout.act_bytes);
    NpuBuffer scale(layout.scale_bytes);
    NpuBuffer out(full_output_host_bytes);
    if (!act.ptr || !scale.ptr || !out.ptr) {
        std::fprintf(stderr, "%s: npu_mem_alloc failed\n", name);
        return false;
    }

    fill_activation(act.data(), k, varied_act_scale);
    if (arbitrary_scale) {
        fill_scale_arbitrary(scale.data(), m, k);
    } else {
        fill_scale(scale.data(), m, k, true);
    }
    std::memset(out.data(), 0xa5, out.bytes);

    std::vector<uint16_t> expected;
    expected.resize(m);
    for (const BlockPlan& block : blocks) {
        if (arbitrary_scale) {
            make_expected_range_arbitrary_scale(&expected, m, k, block.row_base,
                                                block.rows, 0, varied_act_scale);
        } else {
            make_expected_range(&expected, m, k, true, block.row_base, block.rows, 0,
                                varied_act_scale);
        }
    }

    std::printf("%s: offset-output M-blocked pingpong M=%u K=%u blocks=%zu"
                " max_block_rows=%u act=0x%04x/%u scale=0x%04x/%u"
                " out=0x%04x/%u host_out=%u final_mvout=%u arbitrary_scale=%d"
                " pingW=0x%05x pongW=0x%05x\n",
                name,
                m,
                k,
                blocks.size(),
                max_block_rows,
                layout.act_spm_base,
                layout.act_bytes,
                layout.scale_spm_base,
                layout.scale_bytes,
                layout.output_spm_base,
                full_output_spm_bytes,
                full_output_host_bytes,
                final_output_precision,
                arbitrary_scale ? 1 : 0,
                kPingWeightSpmBase,
                kPongWeightSpmBase);

    npu_reset();
    const auto preload_begin = Clock::now();
    mvin_to_spm(act.ptr, layout.act_spm_base, layout.act_bytes, kInputTypeAct, 2);
    mvin_to_spm(scale.ptr, layout.scale_spm_base, layout.scale_bytes, kInputTypeData, 2);
    mvin_to_spm(blocks[0].weight.ptr, blocks[0].weight_spm_base,
                blocks[0].weight_bytes, kInputTypeWeight, 1);
    const uint64_t preload_ns = elapsed_ns(preload_begin, Clock::now());

    uint64_t gemv_ns = 0;
    uint64_t prefetch_window_ns = 0;
    uint64_t post_compute_wait_ns = 0;
    uint64_t hidden_prefetch_ns = 0;
    uint64_t async_mvin_bytes = 0;
    const uint32_t async_dma_id = static_cast<uint32_t>(get_env_int("GEMV_ASYNC_DMA_ID", 0) & 1);
    const bool disable_async_prefetch = get_env_flag("GEMV_DISABLE_ASYNC_PREFETCH");
    std::printf("  async_dma_id=%u disable_async_prefetch=%d final_mvout=%u\n",
                async_dma_id, disable_async_prefetch ? 1 : 0, final_output_precision);

    const auto hw_begin = Clock::now();
    for (size_t i = 0; i < blocks.size(); ++i) {
        BlockPlan& cur = blocks[i];
        bool prefetch_started = false;
        Clock::time_point prefetch_begin{};
        if (i + 1 < blocks.size()) {
            BlockPlan& next = blocks[i + 1];
            prefetch_begin = Clock::now();
            if (disable_async_prefetch) {
                mvin_to_spm(next.weight.ptr, next.weight_spm_base,
                            next.weight_bytes, kInputTypeWeight, 1);
            } else {
                mvin_to_spm_async(async_dma_id, next.weight.ptr, next.weight_spm_base,
                                  next.weight_bytes, kInputTypeWeight, 1);
                async_mvin_bytes += next.weight_bytes;
                prefetch_started = true;
            }
        }

        const uint16_t output_addr =
            static_cast<uint16_t>(layout.output_spm_base + cur.row_base * kFp16Bytes);
        const uint16_t scale_addr = static_cast<uint16_t>(cur.weight_spm_base);
        std::printf("  block[%zu/%zu]: row_base=%u rows=%u W=0x%05x Wbytes=%u"
                    " O=0x%04x S=0x%04x K=%u prefetch=%d\n",
                    i,
                    blocks.size(),
                    cur.row_base,
                    cur.rows,
                    cur.weight_spm_base,
                    cur.weight_bytes,
                    output_addr,
                    scale_addr,
                    k,
                    prefetch_started ? 1 : 0);
        std::fflush(stdout);

        const auto gemv_begin = Clock::now();
        npu_matvec_run(cur.weight_spm_base,
                       layout.act_spm_base,
                       k,
                       static_cast<uint16_t>(cur.rows),
                       output_addr,
                       scale_addr);
        const auto gemv_end = Clock::now();
        gemv_ns += elapsed_ns(gemv_begin, gemv_end);

        if (prefetch_started) {
            const auto wait_begin = Clock::now();
            npu_dma_wait_mvin(1u << async_dma_id);
            const auto wait_end = Clock::now();
            const uint64_t wait_ns = elapsed_ns(wait_begin, wait_end);
            const uint64_t total_prefetch_ns = elapsed_ns(prefetch_begin, wait_end);
            post_compute_wait_ns += wait_ns;
            prefetch_window_ns += total_prefetch_ns;
            hidden_prefetch_ns +=
                (total_prefetch_ns > wait_ns) ? (total_prefetch_ns - wait_ns) : 0;
        }
    }

    const auto mvout_begin = Clock::now();
    npu_dma_mvout(out.data(),
                  layout.output_spm_base,
                  0,
                  m - 1u,
                  1,
                  1,
                  final_output_precision,
                  1,
                  false,
                  false,
                  0,
                  0);
    const uint64_t mvout_ns = elapsed_ns(mvout_begin, Clock::now());
    const uint64_t hw_total_ns = elapsed_ns(hw_begin, Clock::now());

    const ErrorStats stats =
        final_output_precision == 3
            ? compare_fp32_output_to_golden(name, " offset-output fp32-mvout",
                                            out.data(), expected, m, 16)
            : compare_output_to_golden(name, " offset-output", out.data(), expected, m, 16);
    print_error_stats(name, " offset-output", stats);
    if (stats.mismatches != 0) {
        std::fprintf(stderr, "%s: offset-output pingpong failed mismatches=%u/%u\n",
                     name, stats.mismatches, m);
        return false;
    }

    const uint64_t cpu_ns = measure_cpu_reference_ns(m, k, true, 1, varied_act_scale);
    std::printf("%s perf:\n", name);
    std::printf("  gemv_compute_total_ms=%.3f avg_block_ms=%.3f blocks=%zu\n",
                ns_to_ms(gemv_ns), ns_to_ms(gemv_ns) / blocks.size(), blocks.size());
    std::printf("  cpu_compute_total_ms=%.3f speedup=%.2fx\n",
                ns_to_ms(cpu_ns),
                gemv_ns ? static_cast<double>(cpu_ns) / static_cast<double>(gemv_ns) : 0.0);
    std::printf("  ddr_preload_ms=%.3f final_mvout_ms=%.3f async_prefetch_window_ms=%.3f"
                " post_compute_wait_ms=%.3f hidden_prefetch_ms=%.3f async_bytes=%llu\n",
                ns_to_ms(preload_ns),
                ns_to_ms(mvout_ns),
                ns_to_ms(prefetch_window_ns),
                ns_to_ms(post_compute_wait_ns),
                ns_to_ms(hidden_prefetch_ns),
                static_cast<unsigned long long>(async_mvin_bytes));
    std::printf("  hw_total_ms=%.3f\n", ns_to_ms(hw_total_ns));
    std::printf("%s: offset-output M-blocked pingpong ok\n", name);
    return true;
}

bool run_repro_dump_case(const char* name) {
    const char* meta_env = std::getenv("GEMV_REPRO_META");
    if (!meta_env || meta_env[0] == '\0') {
        std::printf("%s: skipped, GEMV_REPRO_META is not set\n", name);
        return true;
    }

    const std::map<std::string, std::string> meta = read_meta_file(meta_env);
    const uint32_t m = meta_u32(meta, "m");
    const uint32_t k = meta_u32(meta, "k");
    const uint32_t col_tiles = meta_u32(meta, "col_tiles");
    const uint32_t act_payload_bytes = meta_u32(meta, "act_payload_bytes");
    const uint32_t scale_bytes = meta_u32(meta, "scale_bytes");
    const uint32_t weight_bytes = meta_u32(meta, "weight_bytes");
    const uint32_t host_output_bytes = meta_u32(meta, "host_output_bytes");
    const uint32_t max_block_rows = meta_u32(meta, "max_block_rows");
    const uint8_t output_precision = static_cast<uint8_t>(meta_u32(meta, "output_precision", 3));
    const bool block_mvout = meta_u32(meta, "block_mvout", 0) != 0;
    if (m == 0 || k == 0 || col_tiles == 0 || max_block_rows == 0 ||
        act_payload_bytes == 0 || scale_bytes == 0 || weight_bytes == 0 ||
        host_output_bytes == 0 || output_precision != 3) {
        std::fprintf(stderr, "%s: invalid repro meta: %s\n", name, meta_env);
        return false;
    }

    const std::vector<uint8_t> act_file = read_binary_file(meta_string(meta, "act_file"));
    const std::vector<uint8_t> scale_file = read_binary_file(meta_string(meta, "scale_file"));
    const std::vector<uint8_t> weight_file = read_binary_file(meta_string(meta, "weight_file"));
    const std::vector<uint8_t> dumped_output = read_binary_file(meta_string(meta, "output_file"));
    if (act_file.size() != act_payload_bytes ||
        scale_file.size() != scale_bytes ||
        weight_file.size() != weight_bytes ||
        dumped_output.size() != host_output_bytes) {
        std::fprintf(stderr,
                     "%s: repro file size mismatch act=%zu/%u scale=%zu/%u weight=%zu/%u out=%zu/%u\n",
                     name,
                     act_file.size(), act_payload_bytes,
                     scale_file.size(), scale_bytes,
                     weight_file.size(), weight_bytes,
                     dumped_output.size(), host_output_bytes);
        return false;
    }

    NpuBuffer act;
    NpuBuffer scale;
    NpuBuffer weight;
    NpuBuffer out;
    std::vector<NpuBuffer> offset_pads;
    if (get_env_flag("GEMV_REPRO_USE_META_OFFSETS")) {
        void* cma_base = npu_decode_memory_base();
        const uintptr_t base = reinterpret_cast<uintptr_t>(cma_base);
        uint32_t cursor = parse_size_env("NPU_CMA_HEAP_OFFSET", 0);
        auto offset_of = [base](void* ptr) -> uint32_t {
            if (ptr == nullptr || base == 0) {
                return 0xffffffffu;
            }
            return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(ptr) - base);
        };
        auto alloc_at = [&](const char* label, uint32_t desired, uint32_t bytes) -> NpuBuffer {
            if (desired != 0xffffffffu && desired > cursor) {
                offset_pads.emplace_back(desired - cursor);
                if (!offset_pads.back().ptr) {
                    std::fprintf(stderr, "%s: failed to allocate offset pad for %s bytes=%u\n",
                                 name, label, desired - cursor);
                    return NpuBuffer();
                }
                cursor = offset_of(offset_pads.back().ptr) +
                         align_up(static_cast<uint32_t>(offset_pads.back().bytes), kSpmLineBytes);
            }
            NpuBuffer buf(bytes);
            const uint32_t got = offset_of(buf.ptr);
            if (buf.ptr) {
                cursor = got + align_up(static_cast<uint32_t>(bytes), kSpmLineBytes);
            }
            std::printf("  cma_alloc %s desired=0x%08x got=0x%08x bytes=%u\n",
                        label, desired, got, bytes);
            return buf;
        };
        scale = alloc_at("scale", meta_u32(meta, "scale_cma_offset", 0xffffffffu), scale_bytes);
        weight = alloc_at("weight", meta_u32(meta, "weight_cma_offset", 0xffffffffu), weight_bytes);
        act = alloc_at("act", meta_u32(meta, "act_cma_offset", 0xffffffffu), act_payload_bytes);
        out = alloc_at("out", meta_u32(meta, "output_cma_offset", 0xffffffffu), host_output_bytes);
    } else {
        act = NpuBuffer(act_payload_bytes);
        scale = NpuBuffer(scale_bytes);
        weight = NpuBuffer(weight_bytes);
        out = NpuBuffer(host_output_bytes);
    }
    if (!act.ptr || !scale.ptr || !weight.ptr || !out.ptr) {
        std::fprintf(stderr, "%s: npu_mem_alloc failed\n", name);
        return false;
    }
    std::memcpy(act.data(), act_file.data(), act_payload_bytes);
    std::memcpy(scale.data(), scale_file.data(), scale_bytes);
    std::memcpy(weight.data(), weight_file.data(), weight_bytes);
    std::memset(out.data(), 0xa5, host_output_bytes);

    std::printf("%s: replay %s M=%u K=%u col_tiles=%u max_block_rows=%u"
                " block_mvout=%d output_precision=%u act=%u scale=%u weight=%u out=%u\n",
                name,
                meta_string(meta, "op_name").c_str(),
                m,
                k,
                col_tiles,
                max_block_rows,
                block_mvout ? 1 : 0,
                output_precision,
                act_payload_bytes,
                scale_bytes,
                weight_bytes,
                host_output_bytes);

    npu_reset();
    mvin_to_spm(act.ptr, kPingActSpmBase, act_payload_bytes, kInputTypeAct, 2);
    mvin_to_spm(scale.ptr, kScaleSpmBase, scale_bytes, kInputTypeData, 2);

    const uint32_t bytes_per_row_tile = col_tiles * kWeightTileBytes;
    auto block_rows_for = [&](uint32_t row_base) {
        return std::min<uint32_t>(max_block_rows, m - row_base);
    };
    auto weight_offset_for = [&](uint32_t row_base) {
        return (row_base / kRowTileElems) * bytes_per_row_tile;
    };

    const uint32_t first_rows = block_rows_for(0);
    const uint32_t first_weight_bytes = ceil_div(first_rows, kRowTileElems) * bytes_per_row_tile;
    mvin_to_spm(weight.data(), kPingWeightSpmBase, first_weight_bytes, kInputTypeWeight, 1);

    const uint32_t async_dma_id = static_cast<uint32_t>(get_env_int("GEMV_ASYNC_DMA_ID", 0) & 1);
    uint32_t block_idx = 0;
    for (uint32_t row_base = 0; row_base < m;
         row_base += block_rows_for(row_base), ++block_idx) {
        const uint32_t rows = block_rows_for(row_base);
        const uint32_t cur_weight_spm =
            (block_idx & 1u) ? kPongWeightSpmBase : kPingWeightSpmBase;

        bool prefetch_started = false;
        const uint32_t next_row_base = row_base + rows;
        if (next_row_base < m) {
            const uint32_t next_rows = block_rows_for(next_row_base);
            const uint32_t next_weight_bytes =
                ceil_div(next_rows, kRowTileElems) * bytes_per_row_tile;
            const uint32_t next_weight_spm =
                ((block_idx + 1u) & 1u) ? kPongWeightSpmBase : kPingWeightSpmBase;
            mvin_to_spm_async(async_dma_id,
                              weight.data() + weight_offset_for(next_row_base),
                              next_weight_spm,
                              next_weight_bytes,
                              kInputTypeWeight,
                              1);
            prefetch_started = true;
        }

        const uint16_t output_addr = static_cast<uint16_t>(
            kOutputSpmBase + (block_mvout ? 0u : row_base * kFp16Bytes));
        const uint16_t scale_addr = static_cast<uint16_t>(
            kScaleSpmBase + (row_base / kRowTileElems) * col_tiles * kScaleTileBytes);
        std::printf("  replay block=%u row_base=%u rows=%u W=0x%05x O=0x%04x S=0x%04x prefetch=%d\n",
                    block_idx, row_base, rows, cur_weight_spm, output_addr, scale_addr,
                    prefetch_started ? 1 : 0);
        const uint64_t flow = npu_decode_flow_make(
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            static_cast<uint16_t>(rows),
            0);
        npu_matvec_decode_flow_run(cur_weight_spm,
                                   kPingActSpmBase,
                                   static_cast<uint16_t>(k),
                                   static_cast<uint16_t>(rows),
                                   output_addr,
                                   scale_addr,
                                   0,
                                   flow);

        if (prefetch_started) {
            npu_dma_wait_mvin(1u << async_dma_id);
        }

        if (block_mvout) {
            npu_dma_mvout(out.data() + row_base * sizeof(float),
                          kOutputSpmBase,
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
        }
    }

    if (!block_mvout) {
        npu_dma_mvout(out.data(),
                      kOutputSpmBase,
                      0,
                      m - 1u,
                      1,
                      1,
                      output_precision,
                      1,
                      false,
                      false,
                      0,
                      0);
    }

    const std::vector<float> reference =
        make_repro_reference_f32(act_file.data(), scale_file.data(), weight_file.data(),
                                 m, k, col_tiles);
    const ErrorStats vs_ref =
        compare_fp32_to_reference(name, " replay-vs-ref", out.data(), reference,
                                  m, 3.0e-2, 3.0e-2, 16);
    print_error_stats(name, " replay-vs-ref", vs_ref);

    const ErrorStats vs_dump =
        compare_fp32_buffers(name, " replay-vs-llama-dump", out.data(),
                             dumped_output.data(), m, 0.0, 0.0, 16);
    print_error_stats(name, " replay-vs-llama-dump", vs_dump);

    const ErrorStats dump_vs_ref =
        compare_fp32_to_reference(name, " llama-dump-vs-ref",
                                  dumped_output.data(), reference,
                                  m, 3.0e-2, 3.0e-2, 16);
    print_error_stats(name, " llama-dump-vs-ref", dump_vs_ref);

    if (vs_ref.mismatches != 0) {
        std::fprintf(stderr, "%s: replay reproduced mismatch bad=%u/%u\n",
                     name, vs_ref.mismatches, m);
        return false;
    }
    std::printf("%s: replay ok\n", name);
    return true;
}

} // namespace

int main() {
    std::puts("kv260_gemv_w4a16_test: MATVEC W4A16 functional test");
    const char* mode = std::getenv("GEMV_TEST_MODE");
    if (!mode || mode[0] == '\0') {
        mode = "all";
    }
    const bool run_direct = env_mode_is(mode, "all") || env_mode_is(mode, "direct");
    const bool run_fixed = env_mode_is(mode, "all") || env_mode_is(mode, "fixed");
    const bool run_perf = env_mode_is(mode, "all") || env_mode_is(mode, "perf");
    const bool run_act_scale = env_mode_is(mode, "all") || env_mode_is(mode, "act_scale");
    const bool run_fp32_mvout = env_mode_is(mode, "all") || env_mode_is(mode, "fp32_mvout");
    const bool run_repro_dump = env_mode_is(mode, "repro_dump");
    const bool run_offset = env_mode_is(mode, "offset");

    if (npu_init() != 0) {
        std::perror("npu_init");
        return 1;
    }

    const GemvCase cases[] = {
        {"tb_system_64x64_unit", 64, 64, false, false, false, 0, 0, 0, 0},
        {"tb_layout_64x64_unit", 64, 64, false, false, true, 0x0000, 0x1000, 0x1100, 0x2000},
        {"tb_layout_64x64_out_0x2040", 64, 64, false, false, true, 0x0000, 0x1000, 0x1100, 0x2040},
        {"tb_layout_64x64_out_0x2080", 64, 64, false, false, true, 0x0000, 0x1000, 0x1100, 0x2080},
        {"tb_layout_64x64_out_0x4000", 64, 64, false, false, true, 0x0000, 0x1000, 0x1100, 0x4000},
        {"edge_63x1_scale", 63, 1, true, false, false, 0, 0, 0, 0},
        {"edge_64x1_scale", 64, 1, true, false, false, 0, 0, 0, 0},
        {"edge_65x1_scale", 65, 1, true, false, false, 0, 0, 0, 0},
        {"tile_128x1_scale", 128, 1, true, false, false, 0, 0, 0, 0},
        {"tile_128x128_scale", 128, 128, true, false, false, 0, 0, 0, 0},
        {"smol_kv_320x960_scale", 320, 960, true, false, false, 0, 0, 0, 0},
        {"smol_qo_960x960_from_standalone_tb", 960, 960, true, false, false, 0, 0, 0, 0},
        {"smol_mlp_up_2560x960_from_standalone_tb", 2560, 960, true, false, false, 0, 0, 0, 0},
        {"smol_mlp_down_960x2560_from_standalone_tb", 960, 2560, true, false, false, 0, 0, 0, 0},
    };

    const GemvCase act_scale_cases[] = {
        {"act_scale_edge_63x1", 63, 1, true, true, false, 0, 0, 0, 0},
        {"act_scale_tile_128x128", 128, 128, true, true, false, 0, 0, 0, 0},
        {"act_scale_smol_kv_320x960", 320, 960, true, true, false, 0, 0, 0, 0},
    };

    bool ok = true;
    if (run_direct) {
        for (const auto& tc : cases) {
            ok = run_case(tc) && ok;
        }
    }

    if (run_fixed) {
        ok = run_fixed_block_cmd_case("fixed_cmd_smol_attn_320x960", 320, 960, true) && ok;
        ok = run_fixed_block_cmd_case("fixed_cmd_smol_qo_960x960", 960, 960, true) && ok;
        ok = run_fixed_block_cmd_case("fixed_cmd_smol_mlp_up_2560x960", 2560, 960, true) && ok;
        ok = run_fixed_block_cmd_case("fixed_cmd_smol_mlp_down_960x2560", 960, 2560, true) && ok;
    }

    if (run_perf) {
        ok = run_api_pingpong_case("api_pingpong_320x512", 320, 512) && ok;

        const int perf_iterations = get_env_int("GEMV_PERF_ITERS", 8);
        ok = run_pingpong_perf_case("perf_pingpong_320x512", 320, 512, perf_iterations) && ok;
        ok = run_m_blocked_pingpong_case("blocked_pingpong_smol_qo_960x960", 960, 960) && ok;
        ok = run_m_blocked_pingpong_case("blocked_pingpong_smol_mlp_up_2560x960", 2560, 960) && ok;
        ok = run_m_blocked_pingpong_case("blocked_pingpong_smol_mlp_down_960x2560", 960, 2560) && ok;
    }

    if (run_act_scale) {
        for (const auto& tc : act_scale_cases) {
            ok = run_case(tc) && ok;
        }

        ok = run_fixed_block_cmd_case("fixed_cmd_act_scale_smol_attn_320x960",
                                      320, 960, true, true) && ok;
        ok = run_fixed_block_cmd_case("fixed_cmd_act_scale_smol_qo_960x960",
                                      960, 960, true, true) && ok;
        ok = run_fixed_block_cmd_case("fixed_cmd_act_scale_smol_mlp_up_2560x960",
                                      2560, 960, true, true) && ok;
        ok = run_fixed_block_cmd_case("fixed_cmd_act_scale_smol_mlp_down_960x2560",
                                      960, 2560, true, true) && ok;

        ok = run_api_pingpong_case("api_pingpong_act_scale_320x512",
                                   320, 512, true) && ok;

        const int perf_iterations = get_env_int("GEMV_PERF_ITERS", 8);
        ok = run_pingpong_perf_case("perf_pingpong_act_scale_320x512",
                                    320, 512, perf_iterations, true) && ok;
        ok = run_m_blocked_pingpong_case("blocked_pingpong_act_scale_smol_qo_960x960",
                                         960, 960, true) && ok;
        ok = run_m_blocked_pingpong_case("blocked_pingpong_act_scale_smol_mlp_up_2560x960",
                                         2560, 960, true) && ok;
        ok = run_m_blocked_pingpong_case("blocked_pingpong_act_scale_smol_mlp_down_960x2560",
                                         960, 2560, true) && ok;
        ok = run_m_blocked_pingpong_offset_output_case(
                 "offset_pingpong_act_scale_smol_qo_960x960", 960, 960, true) && ok;
        ok = run_m_blocked_pingpong_offset_output_case(
                 "offset_pingpong_act_scale_smol_mlp_up_2560x960", 2560, 960, true) && ok;
        ok = run_m_blocked_pingpong_offset_output_case(
                 "offset_pingpong_act_scale_smol_mlp_down_960x2560", 960, 2560, true) && ok;
    }

    if (run_fp32_mvout) {
        if (get_env_flag("GEMV_RUN_RAW_FP32_MVOUT")) {
            ok = run_raw_fp16_to_fp32_mvout_case(
                     "raw_fp16_to_fp32_mvout_2560", 2560) && ok;
        }
        ok = run_api_pingpong_fp32_mvout_case(
                 "api_pingpong_fp32_mvout_act_scale_smol_kv_320x960",
                 320, 960, true) && ok;
        ok = run_api_pingpong_fp32_mvout_case(
                 "api_pingpong_fp32_mvout_act_scale_smol_qo_960x960",
                 960, 960, true) && ok;
        ok = run_api_pingpong_fp32_mvout_case(
                 "api_pingpong_fp32_mvout_act_scale_smol_mlp_up_2560x960",
                 2560, 960, true) && ok;
        ok = run_api_pingpong_fp32_mvout_case(
                 "api_pingpong_fp32_mvout_act_scale_smol_mlp_down_960x2560",
                 960, 2560, true) && ok;
        ok = run_m_blocked_pingpong_offset_output_case(
                 "offset_pingpong_fp32_mvout_act_scale_smol_qo_960x960",
                 960, 960, true, 3) && ok;
        ok = run_m_blocked_pingpong_offset_output_case(
                 "offset_pingpong_fp32_mvout_act_scale_smol_mlp_up_2560x960",
                 2560, 960, true, 3) && ok;
        ok = run_m_blocked_pingpong_offset_output_case(
                 "offset_pingpong_fp32_mvout_act_scale_smol_mlp_down_960x2560",
                 960, 2560, true, 3) && ok;
        if (get_env_flag("GEMV_RUN_ARBITRARY_SCALE_FP32_MVOUT")) {
            ok = run_m_blocked_pingpong_offset_output_case(
                     "offset_pingpong_fp32_mvout_arbitrary_scale_smol_qo_960x960",
                     960, 960, true, 3, true) && ok;
            ok = run_m_blocked_pingpong_offset_output_case(
                     "offset_pingpong_fp32_mvout_arbitrary_scale_smol_mlp_up_2560x960",
                     2560, 960, true, 3, true) && ok;
        }
    }

    if (run_repro_dump) {
        ok = run_repro_dump_case("repro_dump") && ok;
    }

    if (run_offset) {
        const char* only = std::getenv("GEMV_OFFSET_ONLY");
        auto run_offset_case = [&](const char* name, uint16_t m, uint16_t k) {
            if (only && only[0] != '\0' && std::strstr(name, only) == nullptr) {
                return;
            }
            ok = run_m_blocked_pingpong_offset_output_case(name, m, k) && ok;
        };
        run_offset_case("offset_pingpong_smol_qo_960x960", 960, 960);
        run_offset_case("offset_pingpong_smol_mlp_up_2560x960", 2560, 960);
        run_offset_case("offset_pingpong_smol_mlp_down_960x2560", 960, 2560);
    }

    npu_destroy();
    std::puts(ok ? "kv260_gemv_w4a16_test=ok" : "kv260_gemv_w4a16_test=fail");
    return ok ? 0 : 2;
}
