#include "npu_regs_compat.h"
#include "npu_runtime.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr uint32_t kTileElems = 128;
constexpr uint32_t kRowTileElems = 32;
constexpr uint32_t kFp16Bytes = 2;
constexpr uint32_t kLineBytes = 64;
constexpr uint32_t kMvinAlign = 256;
constexpr uint32_t kActSpmBase = 0x0000;
constexpr uint32_t kScaleSpmBase = 0x0000;
constexpr uint32_t kWeightSpmBase = 0x10000;
constexpr uint32_t kOutputSpmBase = 0x0000;

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
};

uint32_t align_up(uint32_t value, uint32_t alignment) {
    return (value + alignment - 1u) & ~(alignment - 1u);
}

uint32_t ceil_div(uint32_t value, uint32_t divisor) {
    return (value + divisor - 1u) / divisor;
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

int8_t weight_value(uint32_t row, uint32_t col) {
    return static_cast<int8_t>(static_cast<int>((row * 3u + col * 5u) % 5u) - 2);
}

uint32_t weight_row_bytes(uint8_t mode) {
    if (mode == NPU_GEMV_MODE_W8A16) {
        return kTileElems;
    }
    if (mode == NPU_GEMV_MODE_W16A16) {
        return kTileElems * kFp16Bytes;
    }
    return kTileElems / 2u;
}

uint8_t weight_precision(uint8_t mode) {
    return mode == NPU_GEMV_MODE_W16A16 ? 2 : 1;
}

const char* mode_name(uint8_t mode) {
    if (mode == NPU_GEMV_MODE_W8A16) {
        return "W8A16";
    }
    if (mode == NPU_GEMV_MODE_W16A16) {
        return "W16A16";
    }
    return "W4A16";
}

bool uses_embedded_scale(uint8_t mode) {
    return mode == NPU_GEMV_MODE_W4A16 || mode == NPU_GEMV_MODE_W8A16;
}

void fill_activation(uint8_t* dst, uint32_t k, uint8_t mode) {
    const uint32_t col_tiles = ceil_div(k, kTileElems);
    const uint32_t act_bytes = col_tiles * kTileElems * kFp16Bytes;
    const uint32_t payload_bytes =
        mode == NPU_GEMV_MODE_W4A16 ? act_bytes * 2u : act_bytes;
    std::memset(dst, 0, payload_bytes);
    for (uint32_t col = 0; col < k; ++col) {
        store_u16_le(dst + col * kFp16Bytes, 0x3c00);
    }
    if (mode == NPU_GEMV_MODE_W4A16) {
        for (uint32_t col = 0; col < k; ++col) {
            store_u16_le(dst + act_bytes + col * kFp16Bytes, 0x3c00);
        }
    }
}

void fill_scale(uint8_t* dst, uint32_t m, uint32_t k) {
    const uint32_t row_tiles = ceil_div(m, kRowTileElems);
    const uint32_t col_tiles = ceil_div(k, kTileElems);
    const uint32_t scale_tile_bytes = kRowTileElems * kFp16Bytes;
    std::memset(dst, 0, row_tiles * col_tiles * scale_tile_bytes);
    for (uint32_t row_tile = 0; row_tile < row_tiles; ++row_tile) {
        for (uint32_t col_tile = 0; col_tile < col_tiles; ++col_tile) {
            const uint32_t tile_base = (row_tile * col_tiles + col_tile) * scale_tile_bytes;
            for (uint32_t lane = 0; lane < kRowTileElems; ++lane) {
                const uint32_t row = row_tile * kRowTileElems + lane;
                if (row < m) {
                    store_u16_le(dst + tile_base + lane * kFp16Bytes, 0x3c00);
                }
            }
        }
    }
}

void fill_weight(uint8_t* dst, uint32_t m, uint32_t k, uint8_t mode) {
    const uint32_t row_tiles = ceil_div(m, kRowTileElems);
    const uint32_t col_tiles = ceil_div(k, kTileElems);
    const uint32_t row_bytes = weight_row_bytes(mode);
    const uint32_t data_tile_bytes = kRowTileElems * row_bytes;
    const uint32_t tile_bytes =
        uses_embedded_scale(mode) ? kLineBytes + data_tile_bytes : data_tile_bytes;
    std::memset(dst, 0, row_tiles * col_tiles * tile_bytes);

    for (uint32_t row_tile = 0; row_tile < row_tiles; ++row_tile) {
        for (uint32_t col_tile = 0; col_tile < col_tiles; ++col_tile) {
            const uint32_t tile_base = (row_tile * col_tiles + col_tile) * tile_bytes;
            if (uses_embedded_scale(mode)) {
                for (uint32_t lane = 0; lane < kRowTileElems; ++lane) {
                    const uint32_t row = row_tile * kRowTileElems + lane;
                    if (row < m) {
                        store_u16_le(dst + tile_base + lane * kFp16Bytes, 0x3c00);
                    }
                }
            }
            for (uint32_t row_lane = 0; row_lane < kRowTileElems; ++row_lane) {
                const uint32_t row = row_tile * kRowTileElems + row_lane;
                const uint32_t row_base = tile_base +
                                          (uses_embedded_scale(mode) ? kLineBytes : 0u) +
                                          row_lane * row_bytes;
                for (uint32_t col_lane = 0; col_lane < kTileElems; ++col_lane) {
                    const uint32_t col = col_tile * kTileElems + col_lane;
                    if (row >= m || col >= k) {
                        continue;
                    }
                    const int8_t value = weight_value(row, col);
                    if (mode == NPU_GEMV_MODE_W8A16) {
                        dst[row_base + col_lane] = static_cast<uint8_t>(value);
                    } else if (mode == NPU_GEMV_MODE_W16A16) {
                        store_u16_le(dst + row_base + col_lane * kFp16Bytes,
                                     fp32_to_fp16_bits(static_cast<float>(value)));
                    } else {
                        uint8_t& packed = dst[row_base + col_lane / 2u];
                        const uint8_t nibble = static_cast<uint8_t>(value) & 0x0fu;
                        if (col_lane & 1u) {
                            packed = static_cast<uint8_t>((packed & 0x0fu) | (nibble << 4));
                        } else {
                            packed = static_cast<uint8_t>((packed & 0xf0u) | nibble);
                        }
                    }
                }
            }
        }
    }
}

std::vector<uint16_t> make_expected(uint32_t m, uint32_t k) {
    std::vector<uint16_t> expected(m);
    for (uint32_t row = 0; row < m; ++row) {
        int sum = 0;
        for (uint32_t col = 0; col < k; ++col) {
            sum += weight_value(row, col);
        }
        expected[row] = fp32_to_fp16_bits(static_cast<float>(sum));
    }
    return expected;
}

bool fp16_close(uint16_t got, uint16_t expected) {
    if ((got & 0x7fffu) == 0 && (expected & 0x7fffu) == 0) {
        return true;
    }
    if ((got >> 15) != (expected >> 15)) {
        return false;
    }
    return std::abs(static_cast<int>(got & 0x7fffu) -
                    static_cast<int>(expected & 0x7fffu)) <= 1;
}

bool run_mode_case(uint8_t mode) {
    constexpr uint32_t m = 128;
    constexpr uint32_t k = 128;
    const uint32_t row_tiles = ceil_div(m, kRowTileElems);
    const uint32_t col_tiles = ceil_div(k, kTileElems);
    const uint32_t act_bytes = col_tiles * kTileElems * kFp16Bytes;
    const uint32_t act_payload_bytes =
        mode == NPU_GEMV_MODE_W4A16 ? act_bytes * 2u : act_bytes;
    const uint32_t scale_bytes = row_tiles * col_tiles * kRowTileElems * kFp16Bytes;
    const uint32_t weight_data_tile_bytes = kRowTileElems * weight_row_bytes(mode);
    const uint32_t weight_tile_bytes =
        uses_embedded_scale(mode) ? kLineBytes + weight_data_tile_bytes : weight_data_tile_bytes;
    const uint32_t weight_bytes = row_tiles * col_tiles * weight_tile_bytes;
    const uint32_t output_bytes = align_up(m * kFp16Bytes, kLineBytes);

    static_assert((m * kFp16Bytes) % kLineBytes == 0, "output is line aligned");
    if ((act_payload_bytes % kMvinAlign) || (scale_bytes % kMvinAlign) ||
        (weight_bytes % kMvinAlign)) {
        std::fprintf(stderr, "%s layout is not MVIN aligned\n", mode_name(mode));
        return false;
    }

    NpuBuffer act(act_payload_bytes);
    NpuBuffer scale(scale_bytes);
    NpuBuffer weight(weight_bytes);
    NpuBuffer out(output_bytes);
    if (!act.ptr || !scale.ptr || !weight.ptr || !out.ptr) {
        std::fprintf(stderr, "%s: npu_mem_alloc failed\n", mode_name(mode));
        return false;
    }

    fill_activation(act.data(), k, mode);
    fill_scale(scale.data(), m, k);
    fill_weight(weight.data(), m, k, mode);
    std::memset(out.data(), 0xa5, output_bytes);
    const std::vector<uint16_t> expected = make_expected(m, k);

    npu_reset();
    npu_dma_mvin(act.ptr, kActSpmBase, act_payload_bytes - 1u, 0, 0, 0, 2,
                 NPU_GEMV_MVIN_ACT, false, false, false, 0, 0, 0);
    if (!uses_embedded_scale(mode)) {
        npu_dma_mvin(scale.ptr, kScaleSpmBase, scale_bytes - 1u, 0, 0, 0, 2,
                     NPU_GEMV_MVIN_INPUT_SPM, false, false, false, 0, 0, 0);
    }
    npu_dma_mvin(weight.ptr, kWeightSpmBase, weight_bytes - 1u, 0, 0, 0,
                 weight_precision(mode), NPU_GEMV_MVIN_WEIGHT, false, false,
                 false, 0, 0, 0);
    npu_matvec_mode_run(kWeightSpmBase, kActSpmBase, k, m, kOutputSpmBase,
                        uses_embedded_scale(mode) ? 0 : kScaleSpmBase,
                        mode);
    npu_dma_mvout(out.ptr, kOutputSpmBase, 0, m - 1u, 1, 1, 1, 1,
                  false, false, 0, 0);

    uint32_t mismatches = 0;
    for (uint32_t row = 0; row < m; ++row) {
        const uint16_t got = load_u16_le(out.data() + row * kFp16Bytes);
        const uint16_t exp = expected[row];
        if (!fp16_close(got, exp)) {
            if (mismatches < 8) {
                std::fprintf(stderr, "%s row=%u expected=0x%04x got=0x%04x\n",
                             mode_name(mode), row, exp, got);
            }
            ++mismatches;
        }
    }

    std::printf("%s checked=%u mismatches=%u\n", mode_name(mode), m, mismatches);
    return mismatches == 0;
}

} // namespace

int main() {
    std::puts("kv260_gemv_modes_test: W4A16/W8A16/W16A16 MATVEC modes");
    if (npu_init() != 0) {
        std::perror("npu_init");
        return 1;
    }

    bool ok = true;
    ok = run_mode_case(NPU_GEMV_MODE_W4A16) && ok;
    ok = run_mode_case(NPU_GEMV_MODE_W8A16) && ok;
    ok = run_mode_case(NPU_GEMV_MODE_W16A16) && ok;

    npu_destroy();
    std::puts(ok ? "kv260_gemv_modes_test=ok" : "kv260_gemv_modes_test=fail");
    return ok ? 0 : 2;
}
