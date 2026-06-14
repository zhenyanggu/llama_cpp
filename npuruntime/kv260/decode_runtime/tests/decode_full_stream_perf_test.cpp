#include "npu_runtime.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr uint32_t kFp16Bytes = NPU_GEMV_FP16_BYTES;
constexpr uint32_t kMvinAlign = NPU_GEMV_MVIN_ALIGN_BYTES;
constexpr uint32_t kRowTileElems = NPU_GEMV_ROW_TILE_ELEMS;
constexpr uint32_t kW4TileElems = NPU_GEMV_TILE_ELEMS;
constexpr uint32_t kW8TileElems = NPU_GEMV_TILE_ELEMS / 2u;
constexpr uint32_t kLineBytes = NPU_GEMV_LINE_BYTES;

constexpr uint32_t kHidden = 960;
constexpr uint32_t kFfn = 2560;
constexpr uint32_t kQHeads = 15;
constexpr uint32_t kKvHeads = 5;
constexpr uint32_t kGroup = 3;
constexpr uint32_t kHeadDim = 64;
constexpr uint32_t kAttentionLen = 992;
constexpr uint32_t kVCapacity = 1024;
constexpr uint32_t kVocab = 49152;
constexpr uint32_t kLlamaLmHeadRows = 49280;
constexpr uint32_t kDefaultLmHeadBlockRows = 4096;
constexpr uint32_t kDecoderLayers = 32;

struct Timing {
    double q_proj_ms = 0.0;
    double k_proj_ms = 0.0;
    double v_proj_ms = 0.0;
    double qk_ms = 0.0;
    double pv_ms = 0.0;
    double o_proj_ms = 0.0;
    double ffn_gate_ms = 0.0;
    double ffn_up_ms = 0.0;
    double ffn_down_ms = 0.0;
    double lm_head_ms = 0.0;
};

uint32_t ceil_div(uint32_t a, uint32_t b) {
    return (a + b - 1u) / b;
}

uint32_t align_up(uint32_t value, uint32_t alignment) {
    return (value + alignment - 1u) & ~(alignment - 1u);
}

uint32_t gcd_u32(uint32_t a, uint32_t b) {
    while (b != 0) {
        const uint32_t t = a % b;
        a = b;
        b = t;
    }
    return a;
}

uint64_t elapsed_ns(Clock::time_point begin, Clock::time_point end) {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
}

void store_u16_le(uint8_t* p, uint16_t v) {
    p[0] = static_cast<uint8_t>(v & 0xffu);
    p[1] = static_cast<uint8_t>(v >> 8);
}

uint16_t load_u16_le(const uint8_t* p) {
    return static_cast<uint16_t>(p[0]) |
           static_cast<uint16_t>(static_cast<uint16_t>(p[1]) << 8);
}

float load_f32_le(const uint8_t* p) {
    float v = 0.0f;
    std::memcpy(&v, p, sizeof(v));
    return v;
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

float exact_value(uint32_t idx, uint32_t salt) {
    static constexpr float kValues[] = {
        -1.5f, -1.0f, -0.75f, -0.5f, 0.5f, 0.75f, 1.0f, 1.5f
    };
    return kValues[(idx * 17u + salt * 13u) & 7u];
}

float positive_scale(uint32_t idx, uint32_t salt) {
    static constexpr float kValues[] = {0.25f, 0.5f, 0.75f, 1.0f};
    return kValues[(idx * 11u + salt * 5u) & 3u];
}

int8_t signed_quant(uint32_t idx, uint32_t salt) {
    static constexpr int8_t kValues[] = {-3, -2, -1, 1, 2, 3};
    return kValues[(idx * 19u + salt * 7u) % 6u];
}

uint32_t sparse_col(uint32_t row, uint32_t i, uint32_t cols, uint32_t salt) {
    const uint32_t stride = cols <= 64 ? 7u : 127u;
    return (row * 37u + salt * 13u + i * stride) % cols;
}

uint32_t act_payload_bytes(uint32_t cols, bool w8_mode) {
    const uint32_t elems_per_tile = w8_mode ? kW8TileElems : kW4TileElems;
    return ceil_div(cols, elems_per_tile) * elems_per_tile * kFp16Bytes;
}

uint32_t weight_tile_bytes(bool w8_mode) {
    return kRowTileElems * (w8_mode ? kW8TileElems : (kW4TileElems / 2u));
}

uint32_t weight_bytes(uint32_t rows, uint32_t cols, bool w8_mode) {
    const uint32_t elems_per_tile = w8_mode ? kW8TileElems : kW4TileElems;
    return ceil_div(rows, kRowTileElems) *
           ceil_div(cols, elems_per_tile) *
           weight_tile_bytes(w8_mode);
}

uint32_t scale_bytes(uint32_t rows, uint32_t cols, bool w8_mode) {
    const uint32_t elems_per_tile = w8_mode ? kW8TileElems : kW4TileElems;
    return ceil_div(rows, kRowTileElems) *
           ceil_div(cols, elems_per_tile) *
           kLineBytes;
}

void fill_vector(NpuBuffer& buf, uint32_t elems, bool w8_mode, uint32_t salt) {
    std::memset(buf.data(), 0, buf.bytes);
    (void)w8_mode;
    for (uint32_t i = 0; i < elems; ++i) {
        store_u16_le(buf.data() + i * kFp16Bytes,
                     fp32_to_fp16_bits(exact_value(i, salt)));
    }
}

void fill_scales(NpuBuffer& buf, uint32_t rows, uint32_t cols, bool w8_mode,
                 uint32_t salt) {
    const uint32_t elems_per_tile = w8_mode ? kW8TileElems : kW4TileElems;
    const uint32_t row_tiles = ceil_div(rows, kRowTileElems);
    const uint32_t col_tiles = ceil_div(cols, elems_per_tile);
    std::memset(buf.data(), 0, buf.bytes);
    for (uint32_t rt = 0; rt < row_tiles; ++rt) {
        for (uint32_t ct = 0; ct < col_tiles; ++ct) {
            const uint32_t base = (rt * col_tiles + ct) * kLineBytes;
            for (uint32_t lane = 0; lane < kRowTileElems; ++lane) {
                store_u16_le(buf.data() + base + lane * kFp16Bytes,
                             fp32_to_fp16_bits(positive_scale(
                                 (rt * col_tiles + ct) * kRowTileElems + lane,
                                 salt)));
            }
        }
    }
}

float scale_at(uint32_t row, uint32_t col, uint32_t cols, bool w8_mode,
               uint32_t salt) {
    const uint32_t elems_per_tile = w8_mode ? kW8TileElems : kW4TileElems;
    const uint32_t col_tiles = ceil_div(cols, elems_per_tile);
    const uint32_t rt = row / kRowTileElems;
    const uint32_t lane = row % kRowTileElems;
    const uint32_t ct = col / elems_per_tile;
    return positive_scale((rt * col_tiles + ct) * kRowTileElems + lane, salt);
}

void fill_sparse_w4(NpuBuffer& weight, uint32_t rows, uint32_t cols,
                    uint32_t nnz_per_row, uint32_t salt) {
    const uint32_t col_tiles = ceil_div(cols, kW4TileElems);
    const uint32_t row_bytes = kW4TileElems / 2u;
    const uint32_t tile_bytes = kRowTileElems * row_bytes;
    std::memset(weight.data(), 0, weight.bytes);
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t i = 0; i < nnz_per_row; ++i) {
            const uint32_t col = sparse_col(row, i, cols, salt);
            const int8_t value = signed_quant(row * 31u + i, salt);
            const uint32_t rt = row / kRowTileElems;
            const uint32_t lane = row % kRowTileElems;
            const uint32_t ct = col / kW4TileElems;
            const uint32_t cl = col % kW4TileElems;
            const uint32_t base = (rt * col_tiles + ct) * tile_bytes;
            uint8_t& byte = weight.data()[base + lane * row_bytes + cl / 2u];
            const uint8_t nibble = static_cast<uint8_t>(value) & 0x0fu;
            if (cl & 1u) {
                byte = static_cast<uint8_t>((byte & 0x0fu) | (nibble << 4));
            } else {
                byte = static_cast<uint8_t>((byte & 0xf0u) | nibble);
            }
        }
    }
}

void fill_sparse_w8(NpuBuffer& weight, uint32_t rows, uint32_t cols,
                    uint32_t nnz_per_row, uint32_t salt) {
    const uint32_t col_tiles = ceil_div(cols, kW8TileElems);
    const uint32_t tile_bytes = kRowTileElems * kW8TileElems;
    std::memset(weight.data(), 0, weight.bytes);
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t i = 0; i < nnz_per_row; ++i) {
            const uint32_t col = sparse_col(row, i, cols, salt);
            const int8_t value = signed_quant(row * 29u + i, salt);
            const uint32_t rt = row / kRowTileElems;
            const uint32_t lane = row % kRowTileElems;
            const uint32_t ct = col / kW8TileElems;
            const uint32_t cl = col % kW8TileElems;
            const uint32_t base = (rt * col_tiles + ct) * tile_bytes;
            weight.data()[base + lane * kW8TileElems + cl] =
                static_cast<uint8_t>(value);
        }
    }
}

float expected_sparse(uint32_t row, uint32_t cols, bool w8_mode,
                      uint32_t nnz_per_row, uint32_t weight_salt,
                      uint32_t scale_salt, const NpuBuffer& act,
                      const NpuBuffer* act_scale,
                      const NpuBuffer* act_scale2) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < nnz_per_row; ++i) {
        const uint32_t col = sparse_col(row, i, cols, weight_salt);
        const int8_t q = w8_mode ? signed_quant(row * 29u + i, weight_salt)
                                 : signed_quant(row * 31u + i, weight_salt);
        const float a = fp16_to_float(load_u16_le(act.data() + col * kFp16Bytes));
        const float as = act_scale
            ? fp16_to_float(load_u16_le(act_scale->data() + col * kFp16Bytes))
            : 1.0f;
        const float as2 = act_scale2
            ? fp16_to_float(load_u16_le(act_scale2->data() + col * kFp16Bytes))
            : 1.0f;
        sum += static_cast<float>(q) *
               scale_at(row, col, cols, w8_mode, scale_salt) * a * as * as2;
    }
    return sum;
}

bool compare_fp16_vector(const char* label, const NpuBuffer& output,
                         const std::vector<float>& expected) {
    uint32_t mismatches = 0;
    float max_abs = 0.0f;
    for (uint32_t i = 0; i < expected.size(); ++i) {
        const float got = fp16_to_float(load_u16_le(output.data() + i * kFp16Bytes));
        const float abs_err = std::fabs(got - expected[i]);
        max_abs = std::max(max_abs, abs_err);
        const float tol = 0.10f + 0.03f * std::fabs(expected[i]);
        if (abs_err > tol) {
            if (mismatches < 6) {
                std::fprintf(stderr,
                             "%s[%u] expected=%g got=%g abs_err=%g tol=%g\n",
                             label, i, expected[i], got, abs_err, tol);
            }
            ++mismatches;
        }
    }
    std::printf("%s checked=%zu mismatches=%u max_abs=%g\n",
                label, expected.size(), mismatches, max_abs);
    return mismatches == 0;
}

bool compare_fp32_vector(const char* label, const NpuBuffer& output,
                         const std::vector<float>& expected) {
    uint32_t mismatches = 0;
    float max_abs = 0.0f;
    for (uint32_t i = 0; i < expected.size(); ++i) {
        const float got = load_f32_le(output.data() + i * sizeof(float));
        const float abs_err = std::fabs(got - expected[i]);
        max_abs = std::max(max_abs, abs_err);
        const float tol = 0.35f + 0.03f * std::fabs(expected[i]);
        if (abs_err > tol) {
            if (mismatches < 6) {
                std::fprintf(stderr,
                             "%s[%u] expected=%g got=%g abs_err=%g tol=%g\n",
                             label, i, expected[i], got, abs_err, tol);
            }
            ++mismatches;
        }
    }
    std::printf("%s checked=%zu mismatches=%u max_abs=%g\n",
                label, expected.size(), mismatches, max_abs);
    return mismatches == 0;
}

double run_stream(npu_device* dev, const npu_stream_gemv_desc& desc,
                  uint32_t warmup, uint32_t repeats) {
    npu_reset();
    for (uint32_t i = 0; i < warmup; ++i) {
        const int rc = npu_stream_gemv_run(dev, &desc, 0);
        if (rc != 0) {
            std::fprintf(stderr, "warmup npu_stream_gemv_run rc=%d\n", rc);
            return -1.0;
        }
    }
    uint64_t total_ns = 0;
    for (uint32_t i = 0; i < repeats; ++i) {
        const auto begin = Clock::now();
        const int rc = npu_stream_gemv_run(dev, &desc, 0);
        const auto end = Clock::now();
        if (rc != 0) {
            std::fprintf(stderr, "timed npu_stream_gemv_run rc=%d iter=%u\n", rc, i);
            return -1.0;
        }
        total_ns += elapsed_ns(begin, end);
    }
    return static_cast<double>(total_ns) / repeats / 1000000.0;
}

bool run_w4_case(npu_device* dev, const char* label, uint32_t rows,
                 uint32_t cols, NpuBuffer& act, const NpuBuffer* act_scale,
                 const NpuBuffer* act_scale2, NpuBuffer& output,
                 uint32_t salt, uint32_t nnz_per_row, double* avg_ms) {
    const uint32_t weight_b = align_up(weight_bytes(rows, cols, false), kMvinAlign);
    const uint32_t scale_b = align_up(scale_bytes(rows, cols, false), kMvinAlign);
    NpuBuffer weight(weight_b);
    NpuBuffer scale(scale_b);
    if (!weight.ptr || !scale.ptr) {
        std::fprintf(stderr, "%s: npu_mem_alloc failed\n", label);
        return false;
    }
    fill_sparse_w4(weight, rows, cols, nnz_per_row, salt);
    fill_scales(scale, rows, cols, false, salt + 101u);
    std::memset(output.data(), 0xa5, output.bytes);

    npu_stream_gemv_desc desc = {};
    desc.act_ptr = act.ptr;
    desc.act_scale_ptr = act_scale ? act_scale->ptr : nullptr;
    desc.act_scale2_ptr = act_scale2 ? act_scale2->ptr : nullptr;
    desc.weight_payload_ptr = weight.ptr;
    desc.weight_scale_ptr = scale.ptr;
    desc.output_ptr = output.ptr;
    desc.m = static_cast<uint16_t>(rows);
    desc.n = static_cast<uint16_t>(cols);
    desc.mode = DECODE_GEMV_W4A16;
    desc.output_precision = DECODE_OUTPUT_FP16;
    desc.role = NPU_STREAM_GEMV_ROLE_MLP;
    desc.dst = NPU_STREAM_GEMV_DST_OUTPUT;
    desc.post_op = NPU_STREAM_POST_BYPASS;
    desc.flags = act_scale ? NPU_STREAM_GEMV_F_ENABLE_ACT_SCALE : 0;
    if (act_scale2) {
        desc.flags |= NPU_STREAM_GEMV_F_ENABLE_ACT_SCALE2;
    }

    const double ms = run_stream(dev, desc, 1, 3);
    if (ms < 0.0) {
        return false;
    }
    *avg_ms = ms;

    std::vector<float> expected(rows);
    for (uint32_t row = 0; row < rows; ++row) {
        expected[row] = expected_sparse(row, cols, false, nnz_per_row,
                                        salt, salt + 101u, act, act_scale,
                                        act_scale2);
    }
    const bool ok = compare_fp16_vector(label, output, expected);
    const uint64_t ops = 2ull * rows * cols;
    const uint64_t payload = static_cast<uint64_t>(act.bytes) +
                             (act_scale ? act_scale->bytes : 0u) +
                             (act_scale2 ? act_scale2->bytes : 0u) +
                             weight_b + scale_b + output.bytes;
    std::printf("decode_full_perf,%s,m=%u,n=%u,avg_ms=%.6f,effective_gops=%.6f,payload_gbps=%.6f\n",
                label, rows, cols, ms, static_cast<double>(ops) / (ms * 1000000.0),
                static_cast<double>(payload) / (ms * 1000000.0));
    return ok;
}

bool run_w8_fp32_case(npu_device* dev, const char* label, uint32_t rows,
                      uint32_t cols, NpuBuffer& act, uint32_t salt,
                      uint32_t nnz_per_row, double* avg_ms) {
    const uint32_t weight_b = align_up(weight_bytes(rows, cols, true), kMvinAlign);
    const uint32_t scale_b = align_up(scale_bytes(rows, cols, true), kMvinAlign);
    const uint32_t output_b = align_up(rows * static_cast<uint32_t>(sizeof(float)),
                                       kMvinAlign);
    NpuBuffer weight(weight_b);
    NpuBuffer scale(scale_b);
    NpuBuffer output(output_b);
    if (!weight.ptr || !scale.ptr || !output.ptr) {
        std::fprintf(stderr, "%s: npu_mem_alloc failed\n", label);
        return false;
    }

    fill_sparse_w8(weight, rows, cols, nnz_per_row, salt);
    fill_scales(scale, rows, cols, true, salt + 101u);
    std::memset(output.data(), 0xa5, output.bytes);

    npu_stream_gemv_desc desc = {};
    desc.act_ptr = act.ptr;
    desc.weight_payload_ptr = weight.ptr;
    desc.weight_scale_ptr = scale.ptr;
    desc.output_ptr = output.ptr;
    desc.m = static_cast<uint16_t>(rows);
    desc.n = static_cast<uint16_t>(cols);
    desc.mode = DECODE_GEMV_W8A16;
    desc.output_precision = DECODE_OUTPUT_FP32;
    desc.role = NPU_STREAM_GEMV_ROLE_LINEAR;
    desc.dst = NPU_STREAM_GEMV_DST_OUTPUT;
    desc.post_op = NPU_STREAM_POST_BYPASS;

    const double ms = run_stream(dev, desc, 1, 3);
    if (ms < 0.0) {
        return false;
    }
    *avg_ms = ms;

    std::vector<float> expected(rows);
    for (uint32_t row = 0; row < rows; ++row) {
        expected[row] = expected_sparse(row, cols, true, nnz_per_row,
                                        salt, salt + 101u, act, nullptr, nullptr);
    }
    const bool ok = compare_fp32_vector(label, output, expected);
    const uint64_t ops = 2ull * rows * cols;
    const uint64_t payload = static_cast<uint64_t>(act.bytes) + weight_b +
                             scale_b + output.bytes;
    std::printf("decode_full_perf,%s,m=%u,n=%u,out_precision=fp32,avg_ms=%.6f,"
                "effective_gops=%.6f,payload_gbps=%.6f\n",
                label, rows, cols, ms,
                static_cast<double>(ops) / (ms * 1000000.0),
                static_cast<double>(payload) / (ms * 1000000.0));
    return ok;
}

bool run_qk_group_case(npu_device* dev, uint32_t kv_head, double* avg_ms) {
    const uint32_t act_raw = act_payload_bytes(kHeadDim, true);
    const uint32_t act_stride = align_up(act_raw, kMvinAlign);
    const uint32_t act_span = (kGroup - 1u) * act_stride + act_raw;
    const uint32_t act_b = align_up(act_span, kMvinAlign);
    const uint32_t weight_b = align_up(weight_bytes(kAttentionLen, kHeadDim, true), kMvinAlign);
    const uint32_t scale_b = align_up(scale_bytes(kAttentionLen, kHeadDim, true), kMvinAlign);
    const uint32_t out_b = align_up(kGroup * kAttentionLen * kFp16Bytes, kMvinAlign);
    NpuBuffer q(act_b);
    NpuBuffer k_weight(weight_b);
    NpuBuffer k_scale(scale_b);
    NpuBuffer score(out_b);
    if (!q.ptr || !k_weight.ptr || !k_scale.ptr || !score.ptr) {
        std::fprintf(stderr, "attention_qk_group_%u: npu_mem_alloc failed\n", kv_head);
        return false;
    }
    std::memset(q.data(), 0, q.bytes);
    for (uint32_t group = 0; group < kGroup; ++group) {
        uint8_t* group_base = q.data() + group * act_stride;
        std::memset(group_base, 0, act_raw);
        const uint32_t salt = 200u + kv_head * kGroup + group;
        for (uint32_t i = 0; i < kHeadDim; ++i) {
            store_u16_le(group_base + i * kFp16Bytes,
                         fp32_to_fp16_bits(exact_value(i, salt)));
        }
    }
    fill_sparse_w8(k_weight, kAttentionLen, kHeadDim, 4, 300u + kv_head);
    fill_scales(k_scale, kAttentionLen, kHeadDim, true, 400u + kv_head);

    npu_stream_gemv_desc desc = {};
    desc.act_ptr = q.ptr;
    desc.weight_payload_ptr = k_weight.ptr;
    desc.weight_scale_ptr = k_scale.ptr;
    desc.output_ptr = score.ptr;
    desc.m = kAttentionLen;
    desc.n = kHeadDim;
    desc.mode = DECODE_GEMV_W8A16;
    desc.output_precision = DECODE_OUTPUT_FP16;
    desc.role = NPU_STREAM_GEMV_ROLE_QK;
    desc.dst = NPU_STREAM_GEMV_DST_OUTPUT;
    desc.post_op = NPU_STREAM_POST_BYPASS;
    desc.elem_count = kAttentionLen;
    desc.group_count = kGroup;
    desc.act_group_stride_bytes = act_stride;

    std::memset(score.data(), 0xa5, score.bytes);
    const double total_ms = run_stream(dev, desc, 0, 1);
    if (total_ms < 0.0) {
        return false;
    }
    std::vector<float> expected(kGroup * kAttentionLen, 0.0f);
    for (uint32_t group = 0; group < kGroup; ++group) {
        for (uint32_t row = 0; row < kAttentionLen; ++row) {
            float sum = 0.0f;
            for (uint32_t i = 0; i < 4; ++i) {
                const uint32_t col = sparse_col(row, i, kHeadDim, 300u + kv_head);
                const int8_t w = signed_quant(row * 29u + i, 300u + kv_head);
                const float a = fp16_to_float(load_u16_le(
                    q.data() + group * act_stride + col * kFp16Bytes));
                sum += static_cast<float>(w) *
                       scale_at(row, col, kHeadDim, true, 400u + kv_head) * a;
            }
            expected[group * kAttentionLen + row] = sum;
        }
    }
    const std::string check_label = "attention_qk_group_kvhead" + std::to_string(kv_head);
    if (!compare_fp16_vector(check_label.c_str(), score, expected)) {
        return false;
    }
    const std::string label = "attention_qk_group_kvhead" + std::to_string(kv_head);
    *avg_ms = total_ms;
    std::printf("decode_full_perf,%s,commands=1,m=%u,n=%u,groups=%u,total_ms=%.6f,effective_gops=%.6f\n",
                label.c_str(), kAttentionLen, kHeadDim, kGroup, total_ms,
                static_cast<double>(2ull * kGroup * kAttentionLen * kHeadDim) /
                    (total_ms * 1000000.0));
    return true;
}

void fill_sparse_pv_weight(NpuBuffer& weight, uint32_t salt,
                           uint32_t row_tile_stride) {
    const uint32_t row_tiles = ceil_div(kHeadDim, kRowTileElems);
    const uint32_t tile_bytes = kRowTileElems * kW8TileElems;
    std::memset(weight.data(), 0, weight.bytes);
    for (uint32_t row = 0; row < kHeadDim; ++row) {
        for (uint32_t i = 0; i < 6; ++i) {
            const uint32_t col = sparse_col(row, i, kAttentionLen, salt);
            const int8_t value = signed_quant(row * 23u + i, salt);
            const uint32_t rt = row / kRowTileElems;
            const uint32_t lane = row % kRowTileElems;
            const uint32_t ct = col / kW8TileElems;
            const uint32_t cl = col % kW8TileElems;
            const uint32_t base = rt * row_tile_stride + ct * tile_bytes;
            weight.data()[base + lane * kW8TileElems + cl] =
                static_cast<uint8_t>(value);
        }
    }
    (void)row_tiles;
}

bool run_pv_case(npu_device* dev, uint32_t kv_head, double* avg_ms) {
    const uint32_t prob_raw = act_payload_bytes(kAttentionLen, true);
    const uint32_t prob_stride = align_up(prob_raw, kMvinAlign);
    const uint32_t prob_span = (kGroup - 1u) * prob_stride + prob_raw;
    const uint32_t prob_b = align_up(prob_span, kMvinAlign);
    const uint32_t scale_b = align_up(act_payload_bytes(kAttentionLen, true), kMvinAlign);
    const uint32_t row_tile_stride =
        ceil_div(kVCapacity, kW8TileElems) * kRowTileElems * kW8TileElems;
    const uint32_t weight_b =
        align_up(ceil_div(kHeadDim, kRowTileElems) * row_tile_stride, kMvinAlign);
    const uint32_t out_b = align_up(kGroup * kHeadDim * kFp16Bytes, kMvinAlign);

    NpuBuffer prob(prob_b);
    NpuBuffer v_scale(scale_b);
    NpuBuffer v_weight(weight_b);
    NpuBuffer out(out_b);
    if (!prob.ptr || !v_scale.ptr || !v_weight.ptr || !out.ptr) {
        std::fprintf(stderr, "attention_pv_%u: npu_mem_alloc failed\n", kv_head);
        return false;
    }
    std::memset(prob.data(), 0, prob.bytes);
    for (uint32_t group = 0; group < kGroup; ++group) {
        for (uint32_t token = 0; token < kAttentionLen; ++token) {
            store_u16_le(prob.data() + group * prob_stride + token * kFp16Bytes,
                         fp32_to_fp16_bits(positive_scale(token, 500u + group + kv_head)));
        }
    }
    std::memset(v_scale.data(), 0, v_scale.bytes);
    for (uint32_t token = 0; token < kAttentionLen; ++token) {
        store_u16_le(v_scale.data() + token * kFp16Bytes,
                     fp32_to_fp16_bits(positive_scale(token, 600u + kv_head)));
    }
    fill_sparse_pv_weight(v_weight, 700u + kv_head, row_tile_stride);
    std::memset(out.data(), 0xa5, out.bytes);

    npu_stream_gemv_desc desc = {};
    desc.act_ptr = prob.ptr;
    desc.act_scale_ptr = v_scale.ptr;
    desc.weight_payload_ptr = v_weight.ptr;
    desc.output_ptr = out.ptr;
    desc.weight_row_tile_stride_bytes = row_tile_stride;
    desc.weight_capacity_tokens = kVCapacity;
    desc.m = kHeadDim;
    desc.n = kAttentionLen;
    desc.mode = DECODE_GEMV_W8A16;
    desc.output_precision = DECODE_OUTPUT_FP16;
    desc.role = NPU_STREAM_GEMV_ROLE_PV;
    desc.dst = NPU_STREAM_GEMV_DST_OUTPUT;
    desc.post_op = NPU_STREAM_POST_BYPASS;
    desc.flags = NPU_STREAM_GEMV_F_ENABLE_ACT_SCALE |
                 NPU_STREAM_GEMV_F_KV_COL_SCALE |
                 NPU_STREAM_GEMV_F_UNIT_WEIGHT_SCALE;
    desc.group_count = kGroup;
    desc.act_group_stride_bytes = prob_stride;

    const double ms = run_stream(dev, desc, 1, 3);
    if (ms < 0.0) {
        return false;
    }
    *avg_ms = ms;

    std::vector<float> expected(kGroup * kHeadDim, 0.0f);
    for (uint32_t group = 0; group < kGroup; ++group) {
        for (uint32_t row = 0; row < kHeadDim; ++row) {
            float sum = 0.0f;
            for (uint32_t i = 0; i < 6; ++i) {
                const uint32_t token = sparse_col(row, i, kAttentionLen,
                                                  700u + kv_head);
                const int8_t q = signed_quant(row * 23u + i, 700u + kv_head);
                const float p = fp16_to_float(load_u16_le(
                    prob.data() + group * prob_stride + token * kFp16Bytes));
                const float s = fp16_to_float(load_u16_le(
                    v_scale.data() + token * kFp16Bytes));
                sum += static_cast<float>(q) * p * s;
            }
            expected[group * kHeadDim + row] = sum;
        }
    }
    const std::string label = "attention_pv_kvhead" + std::to_string(kv_head);
    const bool ok = compare_fp16_vector(label.c_str(), out, expected);
    std::printf("decode_full_perf,%s,m=%u,n=%u,groups=%u,avg_ms=%.6f,effective_gops=%.6f\n",
                label.c_str(), kHeadDim, kAttentionLen, kGroup, ms,
                static_cast<double>(2ull * kHeadDim * kAttentionLen * kGroup) /
                    (ms * 1000000.0));
    return ok;
}

void make_silu_act_scale(const NpuBuffer& gate, NpuBuffer& gate_silu,
                         uint32_t elems) {
    std::memset(gate_silu.data(), 0, gate_silu.bytes);
    for (uint32_t i = 0; i < elems; ++i) {
        const float x = fp16_to_float(load_u16_le(gate.data() + i * kFp16Bytes));
        const float y = x / (1.0f + std::exp(-x));
        store_u16_le(gate_silu.data() + i * kFp16Bytes, fp32_to_fp16_bits(y));
    }
}

bool run_lm_head(npu_device* dev, NpuBuffer& hidden, double* total_ms) {
    uint32_t lm_head_block_rows = kDefaultLmHeadBlockRows;
    if (const char* env = std::getenv("NPU_LM_HEAD_BLOCK_ROWS")) {
        const unsigned long parsed = std::strtoul(env, nullptr, 0);
        if (parsed > 0 && parsed <= kVocab) {
            lm_head_block_rows =
                static_cast<uint32_t>(std::min<unsigned long>(parsed,
                                                              kDefaultLmHeadBlockRows));
        }
    }
    double sum_ms = 0.0;
    uint32_t row_base = 0;
    uint32_t block_idx = 0;
    while (row_base < kVocab) {
        const uint32_t rows = std::min<uint32_t>(lm_head_block_rows,
                                                 kVocab - row_base);
        const uint32_t out_b = align_up(rows * kFp16Bytes, kMvinAlign);
        NpuBuffer out(out_b);
        if (!out.ptr) {
            std::fprintf(stderr, "lm_head block alloc failed\n");
            return false;
        }
        double ms = 0.0;
        const std::string label = "lm_head_block" + std::to_string(block_idx);
        if (!run_w4_case(dev, label.c_str(), rows, kHidden, hidden, nullptr, nullptr,
                         out, 900u + block_idx, 4, &ms)) {
            return false;
        }
        sum_ms += ms;
        row_base += rows;
        ++block_idx;
    }
    *total_ms = sum_ms;
    std::printf("decode_full_perf,lm_head_total,vocab=%u,blocks=%u,total_ms=%.6f\n",
                kVocab, block_idx, sum_ms);
    return true;
}

bool run_lm_head_w8_balanced_49280(npu_device* dev, NpuBuffer& hidden,
                                   double* total_ms) {
    const uint32_t col_tiles = ceil_div(kHidden, kW8TileElems);
    const uint32_t scale_bytes_per_row_tile = col_tiles * kLineBytes;
    const uint32_t row_tile_align =
        kMvinAlign / gcd_u32(scale_bytes_per_row_tile, kMvinAlign);
    const uint32_t block_align_rows = row_tile_align * kRowTileElems;
    if (block_align_rows == 0 ||
        (kDefaultLmHeadBlockRows % block_align_rows) != 0 ||
        (kLlamaLmHeadRows % block_align_rows) != 0) {
        std::fprintf(stderr,
                     "lm_head_w8_balanced_49280: unsupported alignment rows=%u\n",
                     block_align_rows);
        return false;
    }

    const uint32_t total_units = kLlamaLmHeadRows / block_align_rows;
    const uint32_t max_units = kDefaultLmHeadBlockRows / block_align_rows;
    const uint32_t blocks = ceil_div(total_units, max_units);
    const uint32_t base_units = total_units / blocks;
    const uint32_t extra_blocks = total_units % blocks;

    std::printf("lm_head_w8_balanced_49280_plan,total_rows=%u,k=%u,"
                "block_align_rows=%u,max_rows=%u,blocks=%u,base_units=%u,"
                "extra_blocks=%u\n",
                kLlamaLmHeadRows, kHidden, block_align_rows,
                kDefaultLmHeadBlockRows, blocks, base_units, extra_blocks);

    double sum_ms = 0.0;
    uint32_t row_base = 0;
    for (uint32_t block_idx = 0; block_idx < blocks; ++block_idx) {
        const uint32_t units = base_units + (block_idx < extra_blocks ? 1u : 0u);
        const uint32_t rows = units * block_align_rows;
        if (rows == 0 || rows > kDefaultLmHeadBlockRows ||
            row_base + rows > kLlamaLmHeadRows) {
            std::fprintf(stderr,
                         "lm_head_w8_balanced_49280: bad block idx=%u rows=%u row_base=%u\n",
                         block_idx, rows, row_base);
            return false;
        }

        double ms = 0.0;
        const std::string label =
            "lm_head_w8_balanced_block" + std::to_string(block_idx);
        if (!run_w8_fp32_case(dev, label.c_str(), rows, kHidden, hidden,
                              1200u + block_idx, 4, &ms)) {
            return false;
        }
        std::printf("lm_head_w8_balanced_49280_block,idx=%u,row_base=%u,rows=%u,"
                    "avg_ms=%.6f\n",
                    block_idx, row_base, rows, ms);
        sum_ms += ms;
        row_base += rows;
    }
    if (row_base != kLlamaLmHeadRows) {
        std::fprintf(stderr,
                     "lm_head_w8_balanced_49280: ended at row_base=%u expected=%u\n",
                     row_base, kLlamaLmHeadRows);
        return false;
    }

    *total_ms = sum_ms;
    std::printf("lm_head_w8_balanced_49280_summary,total_rows=%u,blocks=%u,"
                "total_ms=%.6f\n",
                kLlamaLmHeadRows, blocks, sum_ms);
    return true;
}

bool run_lm_head_w8_fixed4096_49280(npu_device* dev, NpuBuffer& hidden,
                                    double* total_ms) {
    double sum_ms = 0.0;
    uint32_t row_base = 0;
    uint32_t block_idx = 0;
    std::printf("lm_head_w8_fixed4096_49280_plan,total_rows=%u,k=%u,"
                "max_rows=%u\n",
                kLlamaLmHeadRows, kHidden, kDefaultLmHeadBlockRows);
    while (row_base < kLlamaLmHeadRows) {
        const uint32_t rows =
            std::min<uint32_t>(kDefaultLmHeadBlockRows,
                               kLlamaLmHeadRows - row_base);
        double ms = 0.0;
        const std::string label =
            "lm_head_w8_fixed4096_block" + std::to_string(block_idx);
        if (!run_w8_fp32_case(dev, label.c_str(), rows, kHidden, hidden,
                              1200u + block_idx, 4, &ms)) {
            return false;
        }
        std::printf("lm_head_w8_fixed4096_49280_block,idx=%u,row_base=%u,"
                    "rows=%u,avg_ms=%.6f\n",
                    block_idx, row_base, rows, ms);
        sum_ms += ms;
        row_base += rows;
        ++block_idx;
    }

    *total_ms = sum_ms;
    std::printf("lm_head_w8_fixed4096_49280_summary,total_rows=%u,blocks=%u,"
                "total_ms=%.6f\n",
                kLlamaLmHeadRows, block_idx, sum_ms);
    return true;
}

} // namespace

int main() {
    std::puts("kv260_decode_full_stream_perf_test: one-layer decode plus lm_head");
    std::fflush(stdout);

    npu_device* dev = nullptr;
    const int open_rc = npu_open(&dev, nullptr);
    if (open_rc != 0 || !dev) {
        std::fprintf(stderr, "npu_open failed rc=%d\n", open_rc);
        return 1;
    }

    bool ok = true;
    Timing t = {};

    NpuBuffer hidden(align_up(act_payload_bytes(kHidden, false), kMvinAlign));
    NpuBuffer q_out(align_up(kHidden * kFp16Bytes, kMvinAlign));
    NpuBuffer k_out(align_up(kKvHeads * kHeadDim * kFp16Bytes, kMvinAlign));
    NpuBuffer v_out(align_up(kKvHeads * kHeadDim * kFp16Bytes, kMvinAlign));
    NpuBuffer o_out(align_up(kHidden * kFp16Bytes, kMvinAlign));
    NpuBuffer gate_out(align_up(kFfn * kFp16Bytes, kMvinAlign));
    NpuBuffer up_out(align_up(kFfn * kFp16Bytes, kMvinAlign));
    NpuBuffer gate_silu(align_up(act_payload_bytes(kFfn, false), kMvinAlign));
    NpuBuffer down_smooth(align_up(act_payload_bytes(kFfn, false), kMvinAlign));
    NpuBuffer ffn_out(align_up(kHidden * kFp16Bytes, kMvinAlign));
    NpuBuffer lm_hidden(align_up(act_payload_bytes(kHidden, false), kMvinAlign));
    if (!hidden.ptr || !q_out.ptr || !k_out.ptr || !v_out.ptr || !o_out.ptr ||
        !gate_out.ptr || !up_out.ptr || !gate_silu.ptr || !down_smooth.ptr ||
        !ffn_out.ptr || !lm_hidden.ptr) {
        std::fprintf(stderr, "top-level npu_mem_alloc failed\n");
        npu_close(dev);
        return 2;
    }

    fill_vector(hidden, kHidden, false, 1u);
    fill_vector(lm_hidden, kHidden, false, 70u);

    const bool lm_head_w8_balanced =
        std::getenv("NPU_DECODE_FULL_LMHEAD_W8_BALANCED_49280") != nullptr;
    if (lm_head_w8_balanced) {
        ok = run_lm_head_w8_balanced_49280(dev, lm_hidden, &t.lm_head_ms) && ok;
        npu_close(dev);
        std::puts(ok ? "kv260_decode_full_stream_perf_test=ok"
                     : "kv260_decode_full_stream_perf_test=fail");
        return ok ? 0 : 2;
    }

    const bool lm_head_w8_fixed4096 =
        std::getenv("NPU_DECODE_FULL_LMHEAD_W8_FIXED4096_49280") != nullptr;
    if (lm_head_w8_fixed4096) {
        ok = run_lm_head_w8_fixed4096_49280(dev, lm_hidden, &t.lm_head_ms) && ok;
        npu_close(dev);
        std::puts(ok ? "kv260_decode_full_stream_perf_test=ok"
                     : "kv260_decode_full_stream_perf_test=fail");
        return ok ? 0 : 2;
    }

    const bool only_lm_head = std::getenv("NPU_DECODE_FULL_ONLY_LMHEAD") != nullptr;
    if (only_lm_head) {
        ok = run_lm_head(dev, lm_hidden, &t.lm_head_ms) && ok;
        std::printf("decode_full_summary,attention_len=%u,attn_ms=0.000000,"
                    "ffn_ms=0.000000,layer_ms=0.000000,lm_head_ms=%.6f,"
                    "decoder_layers=%u,decode_layers_plus_lm_ms=%.6f\n",
                    kAttentionLen, t.lm_head_ms, kDecoderLayers, t.lm_head_ms);
        std::printf("decode_full_breakdown,q_proj=0.000000,k_proj=0.000000,"
                    "v_proj=0.000000,qk_total=0.000000,pv_total=0.000000,"
                    "o_proj=0.000000,ffn_gate=0.000000,ffn_up=0.000000,"
                    "ffn_down=0.000000,lm_head=%.6f\n",
                    t.lm_head_ms);
        npu_close(dev);
        std::puts(ok ? "kv260_decode_full_stream_perf_test=ok"
                     : "kv260_decode_full_stream_perf_test=fail");
        return ok ? 0 : 2;
    }

    ok = run_w4_case(dev, "attn_q_proj", kHidden, kHidden, hidden, nullptr, nullptr,
                     q_out, 10u, 8, &t.q_proj_ms) && ok;
    ok = run_w4_case(dev, "attn_k_proj", kKvHeads * kHeadDim, kHidden,
                     hidden, nullptr, nullptr, k_out, 11u, 8, &t.k_proj_ms) && ok;
    ok = run_w4_case(dev, "attn_v_proj", kKvHeads * kHeadDim, kHidden,
                     hidden, nullptr, nullptr, v_out, 12u, 8, &t.v_proj_ms) && ok;

    const bool skip_qk = std::getenv("NPU_DECODE_FULL_SKIP_QK") != nullptr;
    if (skip_qk) {
        std::puts("decode_full_qk=skipped_by_NPU_DECODE_FULL_SKIP_QK");
    } else {
        for (uint32_t kv = 0; kv < kKvHeads; ++kv) {
            double ms = 0.0;
            ok = run_qk_group_case(dev, kv, &ms) && ok;
            t.qk_ms += ms;
        }
    }
    for (uint32_t kv = 0; kv < kKvHeads; ++kv) {
        double ms = 0.0;
        ok = run_pv_case(dev, kv, &ms) && ok;
        t.pv_ms += ms;
    }

    NpuBuffer pv_all(align_up(act_payload_bytes(kHidden, false), kMvinAlign));
    if (!pv_all.ptr) {
        std::fprintf(stderr, "pv_all alloc failed\n");
        ok = false;
    } else {
        fill_vector(pv_all, kHidden, false, 30u);
        ok = run_w4_case(dev, "attn_o_proj", kHidden, kHidden, pv_all, nullptr, nullptr,
                         o_out, 13u, 8, &t.o_proj_ms) && ok;
    }

    ok = run_w4_case(dev, "ffn_gate_proj", kFfn, kHidden, hidden, nullptr, nullptr,
                     gate_out, 20u, 8, &t.ffn_gate_ms) && ok;
    ok = run_w4_case(dev, "ffn_up_proj", kFfn, kHidden, hidden, nullptr, nullptr,
                     up_out, 21u, 8, &t.ffn_up_ms) && ok;
    make_silu_act_scale(gate_out, gate_silu, kFfn);
    fill_vector(down_smooth, kFfn, true, 23u);
    ok = run_w4_case(dev, "ffn_down_swiglu_actscale", kHidden, kFfn,
                     up_out, &gate_silu, &down_smooth, ffn_out, 22u, 8,
                     &t.ffn_down_ms) && ok;

    ok = run_lm_head(dev, lm_hidden, &t.lm_head_ms) && ok;

    const double attn_ms = t.q_proj_ms + t.k_proj_ms + t.v_proj_ms +
                           t.qk_ms + t.pv_ms + t.o_proj_ms;
    const double ffn_ms = t.ffn_gate_ms + t.ffn_up_ms + t.ffn_down_ms;
    const double layer_ms = attn_ms + ffn_ms;
    const double decode_layers_ms =
        layer_ms * static_cast<double>(kDecoderLayers) + t.lm_head_ms;

    std::printf("decode_full_summary,attention_len=%u,attn_ms=%.6f,ffn_ms=%.6f,"
                "layer_ms=%.6f,lm_head_ms=%.6f,decoder_layers=%u,"
                "decode_layers_plus_lm_ms=%.6f\n",
                kAttentionLen, attn_ms, ffn_ms, layer_ms, t.lm_head_ms,
                kDecoderLayers, decode_layers_ms);
    std::printf("decode_full_breakdown,q_proj=%.6f,k_proj=%.6f,v_proj=%.6f,"
                "qk_total=%.6f,pv_total=%.6f,o_proj=%.6f,ffn_gate=%.6f,"
                "ffn_up=%.6f,ffn_down=%.6f,lm_head=%.6f\n",
                t.q_proj_ms, t.k_proj_ms, t.v_proj_ms, t.qk_ms, t.pv_ms,
                t.o_proj_ms, t.ffn_gate_ms, t.ffn_up_ms, t.ffn_down_ms,
                t.lm_head_ms);

    npu_close(dev);
    std::puts(ok ? "kv260_decode_full_stream_perf_test=ok"
                 : "kv260_decode_full_stream_perf_test=fail");
    return ok ? 0 : 2;
}
