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
constexpr uint32_t kW8TileElems = NPU_GEMV_TILE_ELEMS / 2u;
constexpr uint32_t kLineBytes = NPU_GEMV_LINE_BYTES;
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

void fill_fp16_ones(uint8_t* dst, uint32_t elems, bool w8_mode) {
    const uint32_t bytes = align_up(act_payload_bytes(elems, w8_mode), kMvinAlign);
    std::memset(dst, 0, bytes);
    for (uint32_t i = 0; i < elems; ++i) {
        store_u16_le(dst + i * kFp16Bytes, kFp16One);
    }
}

void fill_linear_activation(uint8_t* dst, uint32_t cols) {
    fill_fp16_ones(dst, cols, false);
}

void fill_w4_linear_weight(uint8_t* dst, uint32_t rows, uint32_t cols) {
    const uint32_t row_tiles = ceil_div(rows, kRowTileElems);
    const uint32_t col_tiles = ceil_div(cols, kTileElems);
    const uint32_t row_bytes = kTileElems / 2u;
    const uint32_t tile_bytes = kRowTileElems * row_bytes;
    std::memset(dst, 0, row_tiles * col_tiles * tile_bytes);

    for (uint32_t row = 0; row < rows; ++row) {
        const uint32_t cols_to_write[2] = {row % cols, (row + 32u) % cols};
        const int8_t values[2] = {1, 2};
        for (uint32_t i = 0; i < 2; ++i) {
            const uint32_t col = cols_to_write[i];
            const uint32_t row_tile = row / kRowTileElems;
            const uint32_t row_lane = row % kRowTileElems;
            const uint32_t col_tile = col / kTileElems;
            const uint32_t col_lane = col % kTileElems;
            const uint32_t tile_base =
                (row_tile * col_tiles + col_tile) * tile_bytes;
            uint8_t& packed = dst[tile_base + row_lane * row_bytes + col_lane / 2u];
            const uint8_t nibble = static_cast<uint8_t>(values[i]) & 0x0fu;
            if (col_lane & 1u) {
                packed = static_cast<uint8_t>((packed & 0x0fu) | (nibble << 4));
            } else {
                packed = static_cast<uint8_t>((packed & 0xf0u) | nibble);
            }
        }
    }
}

void fill_weight_scale(uint8_t* dst, uint32_t rows, uint32_t cols,
                       bool w8_mode, uint16_t value_bits) {
    const uint32_t elems_per_tile = w8_mode ? kW8TileElems : kTileElems;
    const uint32_t row_tiles = ceil_div(rows, kRowTileElems);
    const uint32_t col_tiles = ceil_div(cols, elems_per_tile);
    std::memset(dst, 0, weight_scale_bytes(rows, cols, w8_mode));
    for (uint32_t row_tile = 0; row_tile < row_tiles; ++row_tile) {
        for (uint32_t col_tile = 0; col_tile < col_tiles; ++col_tile) {
            const uint32_t tile_base =
                (row_tile * col_tiles + col_tile) * kLineBytes;
            for (uint32_t lane = 0; lane < kRowTileElems; ++lane) {
                if (row_tile * kRowTileElems + lane < rows) {
                    store_u16_le(dst + tile_base + lane * kFp16Bytes, value_bits);
                }
            }
        }
    }
}

int8_t pv_value(uint32_t row, uint32_t token) {
    return static_cast<int8_t>(1u + ((row * 7u + token * 5u) % 7u));
}

void fill_pv_group_prob(uint8_t* dst, uint32_t tokens, uint32_t group_count,
                        uint32_t act_group_stride_bytes) {
    const uint32_t act_bytes = act_payload_bytes(tokens, true);
    const uint32_t span =
        (group_count <= 1u) ? act_bytes :
        (group_count - 1u) * act_group_stride_bytes + act_bytes;
    std::memset(dst, 0, align_up(span, kMvinAlign));
    for (uint32_t group = 0; group < group_count; ++group) {
        uint8_t* group_base = dst + group * act_group_stride_bytes;
        store_u16_le(group_base + group * kFp16Bytes, kFp16One);
        store_u16_le(group_base + 64 * kFp16Bytes, kFp16One);
    }
}

void fill_pv_scale(uint8_t* dst, uint32_t tokens) {
    fill_fp16_ones(dst, tokens, true);
}

void fill_pv_fixed_stride_weight(uint8_t* dst, uint32_t rows, uint32_t capacity,
                                 uint32_t row_tile_stride) {
    const uint32_t row_tiles = ceil_div(rows, kRowTileElems);
    const uint32_t col_tiles = ceil_div(capacity, kW8TileElems);
    const uint32_t tile_bytes = kRowTileElems * kW8TileElems;
    std::memset(dst, 0, row_tiles * row_tile_stride);

    for (uint32_t row_tile = 0; row_tile < row_tiles; ++row_tile) {
        for (uint32_t col_tile = 0; col_tile < col_tiles; ++col_tile) {
            const uint32_t tile_base =
                row_tile * row_tile_stride + col_tile * tile_bytes;
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
                            static_cast<uint8_t>(pv_value(row, token));
                    }
                }
            }
        }
    }
}

uint64_t elapsed_ns(Clock::time_point begin, Clock::time_point end) {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
}

double ns_to_ms(double ns) {
    return ns / 1000000.0;
}

double ops_to_gops(uint64_t ops, double avg_ns) {
    return avg_ns > 0.0 ? static_cast<double>(ops) / avg_ns : 0.0;
}

double bytes_to_gbps(uint64_t bytes, double avg_ns) {
    return avg_ns > 0.0 ? static_cast<double>(bytes) / avg_ns : 0.0;
}

bool check_linear_output(const NpuBuffer& output, uint32_t rows) {
    const uint16_t expected = fp32_to_fp16_bits(3.0f);
    uint32_t mismatches = 0;
    for (uint32_t row = 0; row < rows; ++row) {
        const uint16_t got = load_u16_le(output.data() + row * kFp16Bytes);
        if (!fp16_close(got, expected)) {
            if (mismatches < 4) {
                std::fprintf(stderr,
                             "stream_perf_linear_check[%u] expected=0x%04x got=0x%04x\n",
                             row, expected, got);
            }
            ++mismatches;
        }
    }
    std::printf("stream_perf_linear_check checked=%u mismatches=%u\n",
                rows, mismatches);
    return mismatches == 0;
}

bool check_pv_output(const NpuBuffer& output, uint32_t rows,
                     uint32_t group_count) {
    uint32_t mismatches = 0;
    for (uint32_t group = 0; group < group_count; ++group) {
        for (uint32_t row = 0; row < rows; ++row) {
            const int sum = static_cast<int>(pv_value(row, group)) +
                            static_cast<int>(pv_value(row, 64));
            const uint16_t expected = fp32_to_fp16_bits(static_cast<float>(sum));
            const uint32_t out_idx = group * rows + row;
            const uint16_t got = load_u16_le(output.data() + out_idx * kFp16Bytes);
            if (!fp16_close(got, expected)) {
                if (mismatches < 4) {
                    std::fprintf(stderr,
                                 "stream_perf_pv_check[%u,%u] expected=0x%04x got=0x%04x\n",
                                 group, row, expected, got);
                }
                ++mismatches;
            }
        }
    }
    std::printf("stream_perf_pv_group_check checked=%u mismatches=%u\n",
                rows * group_count, mismatches);
    return mismatches == 0;
}

bool check_qk_softmax_output(const NpuBuffer& output, uint32_t rows,
                             uint32_t group_count) {
    const float expected_f = 1.0f / static_cast<float>(rows);
    const uint16_t expected = fp32_to_fp16_bits(expected_f);
    uint32_t mismatches = 0;
    for (uint32_t group = 0; group < group_count; ++group) {
        for (uint32_t row = 0; row < rows; ++row) {
            const uint32_t out_idx = group * rows + row;
            const uint16_t got = load_u16_le(output.data() + out_idx * kFp16Bytes);
            if (!fp16_close(got, expected, 4)) {
                if (mismatches < 4) {
                    std::fprintf(stderr,
                                 "stream_perf_qk_softmax_check[%u,%u] expected=0x%04x got=0x%04x\n",
                                 group, row, expected, got);
                }
                ++mismatches;
            }
        }
    }
    std::printf("stream_perf_qk_softmax_check checked=%u mismatches=%u expected_fp16=0x%04x\n",
                rows * group_count, mismatches, expected);
    return mismatches == 0;
}

bool run_timed(npu_device* dev, const npu_stream_gemv_desc& desc,
               uint32_t warmup, uint32_t repeats, double* avg_ns) {
    npu_reset();
    for (uint32_t i = 0; i < warmup; ++i) {
        const int rc = npu_stream_gemv_run(dev, &desc, 0);
        if (rc != 0) {
            std::fprintf(stderr, "warmup npu_stream_gemv_run rc=%d\n", rc);
            return false;
        }
    }

    uint64_t total_ns = 0;
    for (uint32_t i = 0; i < repeats; ++i) {
        const auto begin = Clock::now();
        const int rc = npu_stream_gemv_run(dev, &desc, 0);
        const auto end = Clock::now();
        if (rc != 0) {
            std::fprintf(stderr, "timed npu_stream_gemv_run rc=%d iter=%u\n", rc, i);
            return false;
        }
        total_ns += elapsed_ns(begin, end);
    }
    *avg_ns = repeats ? static_cast<double>(total_ns) / repeats : 0.0;
    return true;
}

bool run_qk_softmax_group_perf(npu_device* dev, uint32_t m) {
    constexpr uint32_t n = 64;
    constexpr uint32_t group_count = 3;
    constexpr uint32_t warmup = 2;
    constexpr uint32_t repeats = 50;

    const uint32_t act_bytes_raw = act_payload_bytes(n, true);
    const uint32_t act_group_stride = align_up(act_bytes_raw, kMvinAlign);
    const uint32_t act_span =
        (group_count - 1u) * act_group_stride + act_bytes_raw;
    const uint32_t act_bytes = align_up(act_span, kMvinAlign);
    const uint32_t weight_bytes =
        align_up(ceil_div(m, kRowTileElems) *
                 ceil_div(n, kW8TileElems) *
                 kRowTileElems * kW8TileElems,
                 kMvinAlign);
    const uint32_t scale_bytes =
        align_up(weight_scale_bytes(m, n, true), kMvinAlign);
    const uint32_t output_bytes =
        align_up(group_count * m * kFp16Bytes, kMvinAlign);

    NpuBuffer q(act_bytes);
    NpuBuffer k_payload(weight_bytes);
    NpuBuffer k_scale(scale_bytes);
    NpuBuffer output(output_bytes);
    if (!q.ptr || !k_payload.ptr || !k_scale.ptr || !output.ptr) {
        std::fprintf(stderr, "stream_perf_qk_softmax_group: npu_mem_alloc failed\n");
        return false;
    }

    std::memset(q.data(), 0, q.bytes);
    for (uint32_t group = 0; group < group_count; ++group) {
        fill_fp16_ones(q.data() + group * act_group_stride, n, true);
    }
    std::memset(k_payload.data(), 0, k_payload.bytes);
    fill_weight_scale(k_scale.data(), m, n, true, kFp16One);
    std::memset(output.data(), 0xa5, output.bytes);

    npu_stream_gemv_desc desc = {};
    desc.act_ptr = q.ptr;
    desc.weight_payload_ptr = k_payload.ptr;
    desc.weight_scale_ptr = k_scale.ptr;
    desc.output_ptr = output.ptr;
    desc.m = m;
    desc.n = n;
    desc.mode = DECODE_GEMV_W8A16;
    desc.output_precision = DECODE_OUTPUT_FP16;
    desc.role = NPU_STREAM_GEMV_ROLE_QK;
    desc.dst = NPU_STREAM_GEMV_DST_OUTPUT;
    desc.post_op = NPU_STREAM_POST_SOFTMAX;
    desc.elem_count = m;
    desc.group_count = group_count;
    desc.act_group_stride_bytes = act_group_stride;

    double avg_ns = 0.0;
    if (!run_timed(dev, desc, warmup, repeats, &avg_ns)) {
        return false;
    }
    if (!check_qk_softmax_output(output, m, group_count)) {
        return false;
    }

    const uint64_t ops = 2ull * group_count * m * n;
    const uint64_t payload_bytes =
        static_cast<uint64_t>(act_bytes) + weight_bytes +
        scale_bytes + output_bytes;
    std::printf("stream_qk_w8a16_group_softmax,m=%u,n=%u,groups=%u,"
                "repeats=%u,avg_ms=%.6f,effective_gops=%.6f,"
                "payload_gbps=%.6f,payload_bytes=%llu,act_group_stride=%u\n",
                m, n, group_count, repeats, ns_to_ms(avg_ns),
                ops_to_gops(ops, avg_ns), bytes_to_gbps(payload_bytes, avg_ns),
                static_cast<unsigned long long>(payload_bytes), act_group_stride);
    return true;
}

bool run_linear_perf(npu_device* dev) {
    constexpr uint32_t m = 960;
    constexpr uint32_t n = 960;
    constexpr uint32_t warmup = 2;
    constexpr uint32_t repeats = 20;

    const uint32_t act_bytes = align_up(act_payload_bytes(n, false), kMvinAlign);
    const uint32_t act_scale_bytes = align_up(act_payload_bytes(n, false), kMvinAlign);
    const uint32_t weight_bytes = align_up(w4_weight_bytes(m, n), kMvinAlign);
    const uint32_t scale_bytes = align_up(weight_scale_bytes(m, n, false), kMvinAlign);
    const uint32_t output_bytes = align_up(m * kFp16Bytes, kMvinAlign);

    NpuBuffer act(act_bytes);
    NpuBuffer act_scale(act_scale_bytes);
    NpuBuffer weight(weight_bytes);
    NpuBuffer scale(scale_bytes);
    NpuBuffer output(output_bytes);
    if (!act.ptr || !act_scale.ptr || !weight.ptr || !scale.ptr || !output.ptr) {
        std::fprintf(stderr, "stream_perf_linear: npu_mem_alloc failed\n");
        return false;
    }

    fill_linear_activation(act.data(), n);
    fill_fp16_ones(act_scale.data(), n, false);
    fill_w4_linear_weight(weight.data(), m, n);
    fill_weight_scale(scale.data(), m, n, false, kFp16One);
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

    double avg_ns = 0.0;
    if (!run_timed(dev, desc, warmup, repeats, &avg_ns)) {
        return false;
    }
    if (!check_linear_output(output, m)) {
        return false;
    }

    const uint64_t ops = 2ull * m * n;
    const uint64_t payload_bytes =
        static_cast<uint64_t>(act_bytes) + act_scale_bytes + weight_bytes +
        scale_bytes + output_bytes;
    std::printf("stream_linear_w4a16,m=%u,n=%u,repeats=%u,avg_ms=%.6f,"
                "effective_gops=%.6f,payload_gbps=%.6f,payload_bytes=%llu\n",
                m, n, repeats, ns_to_ms(avg_ns), ops_to_gops(ops, avg_ns),
                bytes_to_gbps(payload_bytes, avg_ns),
                static_cast<unsigned long long>(payload_bytes));
    return true;
}

bool run_pv_group_perf(npu_device* dev, uint32_t n) {
    constexpr uint32_t m = 64;
    constexpr uint32_t capacity = 1024;
    constexpr uint32_t group_count = 3;
    constexpr uint32_t warmup = 2;
    constexpr uint32_t repeats = 50;
    constexpr uint32_t tile_bytes = kRowTileElems * kW8TileElems;
    constexpr uint32_t row_tile_stride =
        ((capacity + kW8TileElems - 1u) / kW8TileElems) * tile_bytes;

    const uint32_t act_bytes_raw = act_payload_bytes(n, true);
    const uint32_t act_group_stride = align_up(act_bytes_raw, kLineBytes);
    const uint32_t act_span =
        (group_count - 1u) * act_group_stride + act_bytes_raw;
    const uint32_t act_bytes = align_up(act_span, kMvinAlign);
    const uint32_t act_scale_bytes = align_up(act_payload_bytes(n, true), kMvinAlign);
    const uint32_t weight_bytes =
        align_up(ceil_div(m, kRowTileElems) * row_tile_stride, kMvinAlign);
    const uint32_t output_bytes = align_up(group_count * m * kFp16Bytes, kMvinAlign);

    NpuBuffer prob(act_bytes);
    NpuBuffer v_scale(act_scale_bytes);
    NpuBuffer v_payload(weight_bytes);
    NpuBuffer output(output_bytes);
    if (!prob.ptr || !v_scale.ptr || !v_payload.ptr || !output.ptr) {
        std::fprintf(stderr, "stream_perf_pv_group: npu_mem_alloc failed\n");
        return false;
    }

    fill_pv_group_prob(prob.data(), n, group_count, act_group_stride);
    fill_pv_scale(v_scale.data(), n);
    fill_pv_fixed_stride_weight(v_payload.data(), m, capacity, row_tile_stride);
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

    double avg_ns = 0.0;
    if (!run_timed(dev, desc, warmup, repeats, &avg_ns)) {
        return false;
    }
    if (!check_pv_output(output, m, group_count)) {
        return false;
    }

    const uint64_t ops = 2ull * group_count * m * n;
    const uint32_t active_col_tiles = ceil_div(n, kW8TileElems);
    const uint64_t streamed_weight_bytes =
        static_cast<uint64_t>(ceil_div(m, kRowTileElems)) *
        active_col_tiles * tile_bytes;
    const uint64_t payload_bytes =
        static_cast<uint64_t>(act_bytes) + act_scale_bytes +
        streamed_weight_bytes + output_bytes;
    std::printf("stream_pv_w8a16_group,m=%u,n=%u,groups=%u,capacity=%u,"
                "repeats=%u,avg_ms=%.6f,effective_gops=%.6f,"
                "payload_gbps=%.6f,payload_bytes=%llu,row_tile_stride=%u\n",
                m, n, group_count, capacity, repeats, ns_to_ms(avg_ns),
                ops_to_gops(ops, avg_ns), bytes_to_gbps(payload_bytes, avg_ns),
                static_cast<unsigned long long>(payload_bytes), row_tile_stride);
    return true;
}

} // namespace

int main() {
    std::puts("kv260_stream_gemv_perf_test: current stream API performance");
    std::fflush(stdout);

    npu_device* dev = nullptr;
    const int open_rc = npu_open(&dev, nullptr);
    if (open_rc != 0 || !dev) {
        std::fprintf(stderr, "kv260_stream_gemv_perf_test: npu_open failed rc=%d\n",
                     open_rc);
        return 1;
    }

    bool ok = true;
    ok = run_linear_perf(dev) && ok;
    for (const uint32_t tokens : {512u, 640u, 992u}) {
        ok = run_qk_softmax_group_perf(dev, tokens) && ok;
        ok = run_pv_group_perf(dev, tokens) && ok;
    }

    npu_close(dev);

    std::puts(ok ? "kv260_stream_gemv_perf_test=ok"
                 : "kv260_stream_gemv_perf_test=fail");
    return ok ? 0 : 2;
}
