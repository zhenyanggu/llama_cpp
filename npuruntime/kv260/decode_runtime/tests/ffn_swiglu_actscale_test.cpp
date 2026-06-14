#include "npu_runtime.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr uint32_t kFp16Bytes = NPU_GEMV_FP16_BYTES;
constexpr uint32_t kMvinAlign = NPU_GEMV_MVIN_ALIGN_BYTES;
constexpr uint32_t kRowTileElems = NPU_GEMV_ROW_TILE_ELEMS;
constexpr uint32_t kTileElems = NPU_GEMV_TILE_ELEMS;
constexpr uint32_t kLineBytes = NPU_GEMV_LINE_BYTES;

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

float load_f32_le(const uint8_t* p) {
    float value = 0.0f;
    std::memcpy(&value, p, sizeof(value));
    return value;
}

struct Lcg {
    uint32_t state = 0xdec0de05u;

    uint32_t next() {
        state = state * 1664525u + 1013904223u;
        return state;
    }
};

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

uint32_t weight_scale_bytes(uint32_t rows, uint32_t cols) {
    return ceil_div(rows, kRowTileElems) *
           ceil_div(cols, kTileElems) *
           kLineBytes;
}

uint32_t act_payload_bytes(uint32_t cols) {
    return ceil_div(cols, kTileElems) * kTileElems * kFp16Bytes;
}

float random_exact_scale(Lcg& rng) {
    static constexpr float kValues[] = {
        0.25f, 0.5f, 0.75f, 1.0f, 1.25f, 1.5f
    };
    return kValues[rng.next() % (sizeof(kValues) / sizeof(kValues[0]))];
}

int8_t random_w4_nonzero(Lcg& rng) {
    static constexpr int8_t kValues[] = {-3, -2, -1, 1, 2, 3};
    return kValues[rng.next() % (sizeof(kValues) / sizeof(kValues[0]))];
}

void pack_w4(uint8_t* packed, uint32_t rows, uint32_t cols,
             const std::vector<int8_t>& dense) {
    const uint32_t col_tiles = ceil_div(cols, kTileElems);
    const uint32_t row_bytes = kTileElems / 2u;
    const uint32_t tile_bytes = kRowTileElems * row_bytes;
    std::memset(packed, 0, w4_weight_bytes(rows, cols));

    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t col = 0; col < cols; ++col) {
            const int8_t value = dense[row * cols + col];
            if (value == 0) {
                continue;
            }
            const uint32_t row_tile = row / kRowTileElems;
            const uint32_t row_lane = row % kRowTileElems;
            const uint32_t col_tile = col / kTileElems;
            const uint32_t col_lane = col % kTileElems;
            const uint32_t tile_base =
                (row_tile * col_tiles + col_tile) * tile_bytes;
            uint8_t& byte = packed[tile_base + row_lane * row_bytes + col_lane / 2u];
            const uint8_t nibble = static_cast<uint8_t>(value) & 0x0fu;
            if (col_lane & 1u) {
                byte = static_cast<uint8_t>((byte & 0x0fu) | (nibble << 4));
            } else {
                byte = static_cast<uint8_t>((byte & 0xf0u) | nibble);
            }
        }
    }
}

bool compare_output_fp16(const char* label, const uint8_t* got,
                         const std::vector<float>& expected) {
    uint32_t mismatches = 0;
    float max_abs = 0.0f;
    for (uint32_t row = 0; row < expected.size(); ++row) {
        const float got_f = fp16_to_float(load_u16_le(got + row * kFp16Bytes));
        const float abs_err = std::fabs(got_f - expected[row]);
        max_abs = std::max(max_abs, abs_err);
        const float tol = 0.08f + 0.02f * std::fabs(expected[row]);
        if (abs_err > tol) {
            if (mismatches < 8) {
                std::fprintf(stderr,
                             "%s row=%u expected=%g got=%g abs_err=%g tol=%g\n",
                             label, row, expected[row], got_f, abs_err, tol);
            }
            ++mismatches;
        }
    }
    std::printf("%s checked=%zu mismatches=%u max_abs=%g\n",
                label, expected.size(), mismatches, max_abs);
    return mismatches == 0;
}

bool compare_output_fp32(const char* label, const uint8_t* got,
                         const std::vector<float>& expected) {
    uint32_t mismatches = 0;
    float max_abs = 0.0f;
    for (uint32_t row = 0; row < expected.size(); ++row) {
        const float got_f = load_f32_le(got + row * sizeof(float));
        const float abs_err = std::fabs(got_f - expected[row]);
        max_abs = std::max(max_abs, abs_err);
        const float tol = 0.35f + 0.03f * std::fabs(expected[row]);
        if (abs_err > tol) {
            if (mismatches < 8) {
                std::fprintf(stderr,
                             "%s row=%u expected=%g got=%g abs_err=%g tol=%g\n",
                             label, row, expected[row], got_f, abs_err, tol);
            }
            ++mismatches;
        }
    }
    std::printf("%s checked=%zu mismatches=%u max_abs=%g\n",
                label, expected.size(), mismatches, max_abs);
    return mismatches == 0;
}

uint64_t elapsed_ns(Clock::time_point begin, Clock::time_point end) {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
}

bool run_case(npu_device* dev, decode_output_precision output_precision,
              const char* label, bool combine_extra_act_scale = false) {
    constexpr uint32_t m = 960;
    constexpr uint32_t n = 2560;
    constexpr uint32_t nnz_per_row = 8;
    constexpr uint32_t warmup = 1;
    constexpr uint32_t repeats = 8;

    const uint32_t act_bytes = align_up(act_payload_bytes(n), kMvinAlign);
    const uint32_t act_scale_bytes = align_up(act_payload_bytes(n), kMvinAlign);
    const uint32_t weight_bytes = align_up(w4_weight_bytes(m, n), kMvinAlign);
    const uint32_t scale_bytes = align_up(weight_scale_bytes(m, n), kMvinAlign);
    const uint32_t output_elem_bytes =
        output_precision == DECODE_OUTPUT_FP32 ? sizeof(float) : kFp16Bytes;
    const uint32_t output_bytes = align_up(m * output_elem_bytes, kMvinAlign);

    NpuBuffer up(act_bytes);
    NpuBuffer gate_silu(act_scale_bytes);
    NpuBuffer extra_act_scale(act_scale_bytes);
    NpuBuffer down_weight(weight_bytes);
    NpuBuffer down_scale(scale_bytes);
    NpuBuffer output(output_bytes);
    if (!up.ptr || !gate_silu.ptr || !extra_act_scale.ptr || !down_weight.ptr ||
        !down_scale.ptr || !output.ptr) {
        std::fprintf(stderr, "%s: npu_mem_alloc failed\n", label);
        return false;
    }

    Lcg rng;
    std::vector<float> up_f(n, 0.0f);
    std::vector<float> gate_silu_f(n, 0.0f);
    std::vector<float> extra_act_scale_f(n, 1.0f);
    std::memset(up.data(), 0, up.bytes);
    std::memset(gate_silu.data(), 0, gate_silu.bytes);
    std::memset(extra_act_scale.data(), 0, extra_act_scale.bytes);
    for (uint32_t col = 0; col < n; ++col) {
        up_f[col] = random_exact_scale(rng);
        gate_silu_f[col] = random_exact_scale(rng);
        extra_act_scale_f[col] = combine_extra_act_scale ? random_exact_scale(rng) : 1.0f;
        store_u16_le(up.data() + col * kFp16Bytes,
                     fp32_to_fp16_bits(up_f[col]));
        store_u16_le(gate_silu.data() + col * kFp16Bytes,
                     fp32_to_fp16_bits(gate_silu_f[col]));
        store_u16_le(extra_act_scale.data() + col * kFp16Bytes,
                     fp32_to_fp16_bits(extra_act_scale_f[col]));
    }

    const uint32_t row_tiles = ceil_div(m, kRowTileElems);
    const uint32_t col_tiles = ceil_div(n, kTileElems);
    std::vector<int8_t> dense_weight(m * n, 0);
    std::vector<float> scale_f(row_tiles * col_tiles * kRowTileElems, 0.0f);
    std::memset(down_scale.data(), 0, down_scale.bytes);
    for (uint32_t rt = 0; rt < row_tiles; ++rt) {
        for (uint32_t ct = 0; ct < col_tiles; ++ct) {
            const uint32_t scale_base = (rt * col_tiles + ct) * kLineBytes;
            for (uint32_t lane = 0; lane < kRowTileElems; ++lane) {
                const float scale = random_exact_scale(rng);
                scale_f[(rt * col_tiles + ct) * kRowTileElems + lane] = scale;
                store_u16_le(down_scale.data() + scale_base + lane * kFp16Bytes,
                             fp32_to_fp16_bits(scale));
            }
        }
    }

    for (uint32_t row = 0; row < m; ++row) {
        for (uint32_t i = 0; i < nnz_per_row; ++i) {
            const uint32_t col = rng.next() % n;
            dense_weight[row * n + col] = random_w4_nonzero(rng);
        }
    }
    pack_w4(down_weight.data(), m, n, dense_weight);
    std::memset(output.data(), 0xa5, output.bytes);

    std::vector<float> expected(m, 0.0f);
    for (uint32_t row = 0; row < m; ++row) {
        float sum = 0.0f;
        const uint32_t rt = row / kRowTileElems;
        const uint32_t lane = row % kRowTileElems;
        for (uint32_t col = 0; col < n; ++col) {
            const int8_t w = dense_weight[row * n + col];
            if (w == 0) {
                continue;
            }
            const uint32_t ct = col / kTileElems;
            const float row_scale =
                scale_f[(rt * col_tiles + ct) * kRowTileElems + lane];
            sum += static_cast<float>(w) * row_scale *
                   up_f[col] * gate_silu_f[col] * extra_act_scale_f[col];
        }
        expected[row] = sum;
    }

    npu_stream_gemv_desc desc = {};
    desc.act_ptr = up.ptr;
    desc.act_scale_ptr = gate_silu.ptr;
    desc.act_scale2_ptr = combine_extra_act_scale ? extra_act_scale.ptr : nullptr;
    desc.weight_payload_ptr = down_weight.ptr;
    desc.weight_scale_ptr = down_scale.ptr;
    desc.output_ptr = output.ptr;
    desc.m = m;
    desc.n = n;
    desc.mode = DECODE_GEMV_W4A16;
    desc.output_precision = output_precision;
    desc.role = NPU_STREAM_GEMV_ROLE_MLP;
    desc.dst = NPU_STREAM_GEMV_DST_OUTPUT;
    desc.post_op = NPU_STREAM_POST_BYPASS;
    desc.flags = NPU_STREAM_GEMV_F_ENABLE_ACT_SCALE |
                 (combine_extra_act_scale ?
                  static_cast<uint32_t>(NPU_STREAM_GEMV_F_ENABLE_ACT_SCALE2) : 0u);

    npu_reset();
    for (uint32_t i = 0; i < warmup; ++i) {
        const int rc = npu_stream_gemv_run(dev, &desc, 0);
        if (rc != 0) {
            std::fprintf(stderr, "%s warmup rc=%d\n", label, rc);
            return false;
        }
    }

    uint64_t total_ns = 0;
    for (uint32_t i = 0; i < repeats; ++i) {
        const auto begin = Clock::now();
        const int rc = npu_stream_gemv_run(dev, &desc, 0);
        const auto end = Clock::now();
        if (rc != 0) {
            std::fprintf(stderr, "%s timed rc=%d iter=%u\n", label, rc, i);
            return false;
        }
        total_ns += elapsed_ns(begin, end);
    }

    const double avg_ns = repeats ? static_cast<double>(total_ns) / repeats : 0.0;
    const uint64_t dense_ops = 2ull * m * n;
    const uint64_t actual_ops = 2ull * m * nnz_per_row;
    const uint64_t payload_bytes =
        static_cast<uint64_t>(act_bytes) + act_scale_bytes + weight_bytes +
        scale_bytes + output_bytes;

    std::printf("%s_perf,m=%u,n=%u,out_precision=%u,nnz_per_row=%u,repeats=%u,"
                "avg_ms=%.6f,dense_effective_gops=%.6f,actual_sparse_gops=%.6f,"
                "payload_gbps=%.6f,payload_bytes=%llu\n",
                label, m, n, static_cast<unsigned>(output_precision),
                nnz_per_row, repeats, avg_ns / 1000000.0,
                avg_ns > 0.0 ? static_cast<double>(dense_ops) / avg_ns : 0.0,
                avg_ns > 0.0 ? static_cast<double>(actual_ops) / avg_ns : 0.0,
                avg_ns > 0.0 ? static_cast<double>(payload_bytes) / avg_ns : 0.0,
                static_cast<unsigned long long>(payload_bytes));

    if (output_precision == DECODE_OUTPUT_FP32) {
        return compare_output_fp32(label, output.data(), expected);
    }
    return compare_output_fp16(label, output.data(), expected);
}

} // namespace

int main() {
    std::puts("kv260_ffn_swiglu_actscale_test: stream API SwiGLU via act_scale");
    std::fflush(stdout);

    npu_device* dev = nullptr;
    const int open_rc = npu_open(&dev, nullptr);
    if (open_rc != 0 || !dev) {
        std::fprintf(stderr, "kv260_ffn_swiglu_actscale_test: npu_open failed rc=%d\n",
                     open_rc);
        return 1;
    }

    bool ok = true;
    ok = run_case(dev, DECODE_OUTPUT_FP16, "ffn_swiglu_actscale_fp16") && ok;
    ok = run_case(dev, DECODE_OUTPUT_FP32, "ffn_swiglu_actscale_fp32") && ok;
    ok = run_case(dev, DECODE_OUTPUT_FP32, "ffn_swiglu_actscale_fp32_separated_scale", true) && ok;
    npu_close(dev);

    std::puts(ok ? "kv260_ffn_swiglu_actscale_test=ok"
                 : "kv260_ffn_swiglu_actscale_test=fail");
    return ok ? 0 : 2;
}
