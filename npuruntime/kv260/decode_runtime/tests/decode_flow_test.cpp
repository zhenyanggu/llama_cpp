#include "npu_regs_compat.h"
#include "npu_runtime.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

namespace {

constexpr uint32_t kTileElems = 128;
constexpr uint32_t kRowTileElems = 32;
constexpr uint32_t kFp16Bytes = 2;
constexpr uint32_t kLineBytes = 64;
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

constexpr uint32_t align_up(uint32_t value, uint32_t alignment) {
    return (value + alignment - 1u) & ~(alignment - 1u);
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

int8_t weight_value(uint32_t row, uint32_t col, int bias) {
    return static_cast<int8_t>(static_cast<int>((row * 7u + col * 3u) % 5u) - 2 + bias);
}

void fill_activation(uint8_t* dst, uint32_t k) {
    const uint32_t act_bytes = kTileElems * kFp16Bytes;
    std::memset(dst, 0, act_bytes * 2u);
    for (uint32_t col = 0; col < k; ++col) {
        store_u16_le(dst + col * kFp16Bytes, 0x3c00);
        store_u16_le(dst + act_bytes + col * kFp16Bytes, 0x3c00);
    }
}

void fill_scale(uint8_t* dst, uint32_t m) {
    std::memset(dst, 0, kRowTileElems * 4u * kFp16Bytes);
    for (uint32_t row = 0; row < m; ++row) {
        store_u16_le(dst + row * kFp16Bytes, 0x3c00);
    }
}

void fill_weight_w4_scaled(uint8_t* dst, uint32_t m, uint32_t k, int bias, uint16_t scale_bits) {
    const uint32_t row_tiles = (m + kRowTileElems - 1u) / kRowTileElems;
    const uint32_t row_bytes = kTileElems / 2u;
    const uint32_t tile_bytes = kLineBytes + kRowTileElems * row_bytes;
    std::memset(dst, 0, row_tiles * tile_bytes);
    for (uint32_t row_tile = 0; row_tile < row_tiles; ++row_tile) {
        const uint32_t tile_base = row_tile * tile_bytes;
        for (uint32_t lane = 0; lane < kRowTileElems; ++lane) {
            const uint32_t row = row_tile * kRowTileElems + lane;
            if (row < m) {
                store_u16_le(dst + tile_base + lane * kFp16Bytes, scale_bits);
            }
        }
        for (uint32_t row_lane = 0; row_lane < kRowTileElems; ++row_lane) {
            const uint32_t row = row_tile * kRowTileElems + row_lane;
            const uint32_t row_base = tile_base + kLineBytes + row_lane * row_bytes;
            for (uint32_t col = 0; col < k; ++col) {
                if (row >= m) {
                    continue;
                }
                uint8_t& packed = dst[row_base + col / 2u];
                const uint8_t nibble = static_cast<uint8_t>(weight_value(row, col, bias)) & 0x0fu;
                if (col & 1u) {
                    packed = static_cast<uint8_t>((packed & 0x0fu) | (nibble << 4));
                } else {
                    packed = static_cast<uint8_t>((packed & 0xf0u) | nibble);
                }
            }
        }
    }
}

void fill_weight_w4(uint8_t* dst, uint32_t m, uint32_t k, int bias) {
    fill_weight_w4_scaled(dst, m, k, bias, 0x3c00);
}

std::vector<uint16_t> expected_w4_scaled(uint32_t m, uint32_t k, int bias, float scale) {
    std::vector<uint16_t> expected(m);
    for (uint32_t row = 0; row < m; ++row) {
        float sum = 0.0f;
        for (uint32_t col = 0; col < k; ++col) {
            sum += static_cast<float>(weight_value(row, col, bias)) * scale;
        }
        expected[row] = fp32_to_fp16_bits(sum);
    }
    return expected;
}

std::vector<uint16_t> expected_w4(uint32_t m, uint32_t k, int bias) {
    return expected_w4_scaled(m, k, bias, 1.0f);
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
            static_cast<uint16_t>((mant + (uint16_t(1) << (sub_shift - 1u))) >> sub_shift);
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

bool compare_output(const uint8_t* out, const std::vector<uint16_t>& expected) {
    uint32_t mismatches = 0;
    for (uint32_t row = 0; row < expected.size(); ++row) {
        const uint16_t got = load_u16_le(out + row * kFp16Bytes);
        const uint16_t exp = expected[row];
        if (got != exp) {
            if (mismatches < 8) {
                std::fprintf(stderr, "decode_flow bypass row=%u expected=0x%04x got=0x%04x\n",
                             row, exp, got);
            }
            ++mismatches;
        }
    }
    std::printf("decode_flow bypass checked=%zu mismatches=%u\n",
                expected.size(), mismatches);
    return mismatches == 0;
}

bool compare_output_f32(const char* label, const uint8_t* out, const std::vector<float>& expected) {
    uint32_t mismatches = 0;
    float max_abs = 0.0f;
    uint32_t max_row = 0;
    for (uint32_t row = 0; row < expected.size(); ++row) {
        const float got = fp16_to_float(load_u16_le(out + row * kFp16Bytes));
        const float exp = expected[row];
        const float abs_err = std::fabs(got - exp);
        const float tol = 0.75f + 0.08f * std::fabs(exp);
        if (abs_err > max_abs) {
            max_abs = abs_err;
            max_row = row;
        }
        if (abs_err > tol) {
            if (mismatches < 8) {
                std::fprintf(stderr, "%s row=%u expected=%f got=%f abs=%f tol=%f\n",
                             label, row, exp, got, abs_err, tol);
            }
            ++mismatches;
        }
    }
    std::printf("%s checked=%zu mismatches=%u max_abs=%f(row=%u)\n",
                label, expected.size(), mismatches, max_abs, max_row);
    return mismatches == 0;
}

std::vector<float> expected_swiglu_down(uint32_t m, uint32_t k, float scale) {
    std::vector<float> swiglu(k, 0.0f);
    for (uint32_t row = 0; row < k; ++row) {
        float gate = 0.0f;
        float up = 0.0f;
        for (uint32_t col = 0; col < k; ++col) {
            gate += static_cast<float>(weight_value(row, col, 0)) * scale;
            up += static_cast<float>(weight_value(row, col, 1)) * scale;
        }
        const float silu = gate / (1.0f + std::exp(-gate));
        swiglu[row] = silu * up;
    }

    std::vector<float> expected(m, 0.0f);
    for (uint32_t row = 0; row < m; ++row) {
        float acc = 0.0f;
        for (uint32_t col = 0; col < k; ++col) {
            acc += swiglu[col] * static_cast<float>(weight_value(row, col, -1)) * scale;
        }
        expected[row] = acc;
    }
    return expected;
}

bool compare_kv_cache_block(const uint8_t* out, const std::vector<uint16_t>& src) {
    uint16_t max_abs = 0;
    uint32_t max_idx = 0;
    for (uint32_t idx = 0; idx < src.size(); ++idx) {
        const uint16_t value = src[idx];
        const uint16_t abs_value = fp16_abs_bits(value);
        if (abs_value > max_abs) {
            max_abs = abs_value;
            max_idx = idx;
        }
    }
    const uint16_t expected_scale = fp16_div_pow2(max_abs, 7);
    const bool debug = std::getenv("NPU_TEST_KV_DEBUG") != nullptr;

    uint32_t mismatches = 0;
    const uint16_t got_scale = load_u16_le(out);
    if (got_scale != expected_scale) {
        std::fprintf(stderr, "decode_flow kv_quant scale expected=0x%04x got=0x%04x\n",
                     expected_scale, got_scale);
        ++mismatches;
    }
    if (debug) {
        std::printf("decode_flow kv_quant debug max_idx=%u max_abs=0x%04x got_scale=0x%04x expected_scale=0x%04x\n",
                    max_idx, max_abs, got_scale, expected_scale);
    }
    for (uint32_t idx = 0; idx < src.size(); ++idx) {
        const int8_t got = static_cast<int8_t>(out[kLineBytes + idx]);
        const int8_t expected = kv_quantize_ref(src[idx], max_abs);
        if (debug) {
            std::printf("  kv[%02u] src=0x%04x got=%4d expected=%4d raw=0x%02x\n",
                        idx, src[idx], got, expected, out[kLineBytes + idx]);
        }
        if (got != expected) {
            if (mismatches < 8) {
                std::fprintf(stderr,
                             "decode_flow kv_quant elem=%u src=0x%04x expected=%d got=%d\n",
                             idx, src[idx], expected, got);
            }
            ++mismatches;
        }
    }
    std::printf("decode_flow kv_quant checked=%zu scale=0x%04x expected_scale=0x%04x mismatches=%u\n",
                src.size(), got_scale, expected_scale, mismatches);
    return mismatches == 0;
}

bool run_decode_flow_cases() {
    constexpr uint32_t m = 128;
    constexpr uint32_t k = 128;
    constexpr uint32_t act_bytes = kTileElems * kFp16Bytes * 2u;
    constexpr uint32_t scale_bytes = kRowTileElems * 4u * kFp16Bytes;
    constexpr uint32_t weight_bytes =
        4u * (kLineBytes + kRowTileElems * (kTileElems / 2u));
    constexpr uint32_t output_bytes = align_up(m * kFp16Bytes, kLineBytes);

    NpuBuffer act(act_bytes);
    NpuBuffer scale(scale_bytes);
    NpuBuffer weight(weight_bytes);
    NpuBuffer out(output_bytes);
    if (!act.ptr || !scale.ptr || !weight.ptr || !out.ptr) {
        std::fprintf(stderr, "decode_flow: npu_mem_alloc failed\n");
        return false;
    }

    fill_activation(act.data(), k);
    fill_scale(scale.data(), m);
    std::memset(out.data(), 0xa5, output_bytes);

    npu_reset();
    npu_dma_mvin(act.ptr, kActSpmBase, act_bytes - 1u, 0, 0, 0, 2,
                 NPU_GEMV_MVIN_ACT, false, false, false, 0, 0, 0);

    fill_weight_w4(weight.data(), m, k, 0);
    npu_dma_mvin(weight.ptr, kWeightSpmBase, weight_bytes - 1u, 0, 0, 0, 1,
                 NPU_GEMV_MVIN_WEIGHT, false, false, false, 0, 0, 0);
    const uint64_t bypass_flow = npu_decode_flow_make(
        NPU_DECODE_SRC_GEMV_STREAM,
        0,
        NPU_DECODE_UNARY_BYPASS,
        NPU_DECODE_BINARY_BYPASS,
        NPU_DECODE_REDUCE_BYPASS,
        NPU_DECODE_DST_OUTPUT_SPM,
        0,
        0,
        m,
        0);
    npu_matvec_decode_flow_run(kWeightSpmBase, kActSpmBase, k, m, kOutputSpmBase,
                               0, NPU_GEMV_MODE_W4A16, bypass_flow);
    npu_dma_mvout(out.ptr, kOutputSpmBase, 0, m - 1u, 1, 1, 1, 1,
                  false, false, 0, 0);
    bool ok = compare_output(out.data(), expected_w4(m, k, 0));

    constexpr float swiglu_scale = 1.0f / 64.0f;
    const uint16_t swiglu_scale_bits = fp32_to_fp16_bits(swiglu_scale);

    fill_weight_w4_scaled(weight.data(), m, k, 0, swiglu_scale_bits);
    npu_dma_mvin(weight.ptr, kWeightSpmBase, weight_bytes - 1u, 0, 0, 0, 1,
                 NPU_GEMV_MVIN_WEIGHT, false, false, false, 0, 0, 0);
    npu_matvec_silu_run(kWeightSpmBase, kActSpmBase, k, m, kOutputSpmBase, 0);
    std::puts("decode_flow silu_to_stream_buffer=completed");

    fill_weight_w4_scaled(weight.data(), m, k, 1, swiglu_scale_bits);
    npu_dma_mvin(weight.ptr, kWeightSpmBase, weight_bytes - 1u, 0, 0, 0, 1,
                 NPU_GEMV_MVIN_WEIGHT, false, false, false, 0, 0, 0);
    const uint64_t swiglu_flow = npu_decode_flow_make(
        NPU_DECODE_SRC_GEMV_STREAM,
        NPU_DECODE_SRC_POST_BUFFER,
        NPU_DECODE_UNARY_BYPASS,
        NPU_DECODE_BINARY_FP16_MUL,
        NPU_DECODE_REDUCE_BYPASS,
        NPU_DECODE_DST_ACT_BUFFER,
        0,
        0,
        m,
        0);
    npu_matvec_decode_flow_run(kWeightSpmBase, kActSpmBase, k, m, kOutputSpmBase,
                               0, NPU_GEMV_MODE_W4A16, swiglu_flow);
    std::puts("decode_flow swiglu_mul_to_act=completed");

    fill_weight_w4_scaled(weight.data(), m, k, -1, swiglu_scale_bits);
    npu_dma_mvin(weight.ptr, kWeightSpmBase, weight_bytes - 1u, 0, 0, 0, 1,
                 NPU_GEMV_MVIN_WEIGHT, false, false, false, 0, 0, 0);
    const uint64_t down_flow = npu_decode_flow_make(
        NPU_DECODE_SRC_GEMV_STREAM,
        0,
        NPU_DECODE_UNARY_BYPASS,
        NPU_DECODE_BINARY_BYPASS,
        NPU_DECODE_REDUCE_BYPASS,
        NPU_DECODE_DST_OUTPUT_SPM,
        0,
        0,
        m,
        0);
    npu_matvec_decode_flow_run(kWeightSpmBase, kActSpmBase, k, m, kOutputSpmBase,
                               0, NPU_GEMV_MODE_W4A16, down_flow);
    npu_dma_mvout(out.ptr, kOutputSpmBase, 0, m - 1u, 1, 1, 1, 1,
                  false, false, 0, 0);
    ok = compare_output_f32("decode_flow swiglu_down", out.data(), expected_swiglu_down(m, k, swiglu_scale)) && ok;

    if (std::getenv("NPU_TEST_KV_QUANT")) {
        uint32_t kv_elems = 64;
        if (const char* env = std::getenv("NPU_TEST_KV_ELEMS")) {
            const uint32_t parsed = static_cast<uint32_t>(std::strtoul(env, nullptr, 0));
            if (parsed > 0 && parsed <= m) {
                kv_elems = parsed;
            }
        }
        constexpr uint32_t kv_bytes = 2u * kLineBytes;
        std::memset(out.data(), 0xa5, output_bytes);
        fill_activation(act.data(), k);
        npu_dma_mvin(act.ptr, kActSpmBase, act_bytes - 1u, 0, 0, 0, 2,
                     NPU_GEMV_MVIN_ACT, false, false, false, 0, 0, 0);
        fill_weight_w4(weight.data(), kv_elems, k, 1);
        npu_dma_mvin(weight.ptr, kWeightSpmBase, weight_bytes - 1u, 0, 0, 0, 1,
                     NPU_GEMV_MVIN_WEIGHT, false, false, false, 0, 0, 0);

        const uint64_t kv_probe_flow = npu_decode_flow_make(
            NPU_DECODE_SRC_GEMV_STREAM,
            0,
            NPU_DECODE_UNARY_BYPASS,
            NPU_DECODE_BINARY_BYPASS,
            NPU_DECODE_REDUCE_BYPASS,
            NPU_DECODE_DST_OUTPUT_SPM,
            0,
            0,
            kv_elems,
            0);
        npu_matvec_decode_flow_run(kWeightSpmBase, kActSpmBase, k, kv_elems, kOutputSpmBase,
                                   kScaleSpmBase, NPU_GEMV_MODE_W4A16, kv_probe_flow);
        npu_dma_mvout(out.ptr, kOutputSpmBase, 0, kv_elems - 1u, 1, 1, 1, 1,
                      false, false, 0, 0);
        std::vector<uint16_t> kv_src(kv_elems);
        for (uint32_t idx = 0; idx < kv_elems; ++idx) {
            kv_src[idx] = load_u16_le(out.data() + idx * kFp16Bytes);
        }

        const uint64_t kv_flow = npu_decode_flow_make(
            NPU_DECODE_SRC_GEMV_STREAM,
            0,
            NPU_DECODE_UNARY_BYPASS,
            NPU_DECODE_BINARY_BYPASS,
            NPU_DECODE_REDUCE_BYPASS,
            NPU_DECODE_DST_OUTPUT_SPM,
            0,
            0,
            kv_elems,
            0) | NPU_DECODE_FLAG_KV_QUANT;
        npu_matvec_decode_flow_run(kWeightSpmBase, kActSpmBase, k, kv_elems, kOutputSpmBase,
                                   kScaleSpmBase, NPU_GEMV_MODE_W4A16, kv_flow);
        npu_dma_mvout(out.ptr, kOutputSpmBase, 0, (kv_bytes / kFp16Bytes) - 1u,
                      1, 1, 1, 1, false, false, 0, 0);
        ok = compare_kv_cache_block(out.data(), kv_src) && ok;
    } else {
        std::puts("decode_flow kv_quant=skipped enable_with_NPU_TEST_KV_QUANT=1");
    }
    return ok;
}

} // namespace

int main() {
    std::puts("kv260_decode_flow_test: explicit decode-flow control");
    if (npu_init() != 0) {
        std::perror("npu_init");
        return 1;
    }

    const bool ok = run_decode_flow_cases();
    npu_destroy();
    std::puts(ok ? "kv260_decode_flow_test=ok" : "kv260_decode_flow_test=fail");
    return ok ? 0 : 2;
}
