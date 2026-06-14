#include "npu_runtime.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kFp16Bytes = NPU_GEMV_FP16_BYTES;
constexpr uint32_t kMvinAlign = NPU_GEMV_MVIN_ALIGN_BYTES;
constexpr uint32_t kRowTileElems = NPU_GEMV_ROW_TILE_ELEMS;
constexpr uint32_t kW8TileElems = NPU_GEMV_TILE_ELEMS / 2u;

struct StreamDumpHeader {
    char magic[16];
    uint32_t version;
    uint32_t layer;
    uint32_t position;
    uint32_t seq_len;
    uint32_t hidden_dim;
    uint32_t q_heads;
    uint32_t kv_heads;
    uint32_t group_size;
    uint32_t head_dim;
    uint32_t q_head_stride;
    uint32_t score_prob_raw_bytes;
    uint32_t score_prob_stride_bytes;
    uint32_t q_head_bytes;
    uint32_t qk_compact_per_kv_head_bytes;
    uint32_t qk_compact_all_bytes;
    uint32_t prob_padded_bytes;
    uint32_t pv_heads_bytes;
    uint32_t cache_bytes;
    uint32_t kv_head_stride;
    uint32_t cache_scale_region;
    uint32_t v_row_tile_stride_bytes;
    uint32_t cache_capacity_tokens;
    uint32_t visible_window_tokens;
    uint32_t flags;
};

struct StreamDump {
    StreamDumpHeader hdr = {};
    std::vector<uint8_t> q;
    std::vector<uint8_t> k;
    std::vector<uint8_t> v;
    std::vector<uint8_t> qk;
    std::vector<uint8_t> prob;
    std::vector<uint8_t> pv;
};

struct NpuBuffer {
    void * ptr = nullptr;
    size_t bytes = 0;
    explicit NpuBuffer(size_t size) : ptr(npu_mem_alloc(size)), bytes(size) {}
    ~NpuBuffer() {
        if (ptr != nullptr) {
            npu_mem_free(ptr);
        }
    }
    NpuBuffer(const NpuBuffer &) = delete;
    NpuBuffer & operator=(const NpuBuffer &) = delete;
    uint8_t * data() { return static_cast<uint8_t *>(ptr); }
};

uint32_t align_up(uint32_t value, uint32_t alignment) {
    return (value + alignment - 1u) & ~(alignment - 1u);
}

uint16_t load_u16_le(const uint8_t * p) {
    return static_cast<uint16_t>(p[0]) |
           static_cast<uint16_t>(static_cast<uint16_t>(p[1]) << 8);
}

float fp16_to_float(uint16_t h) {
    const uint32_t sign = (h >> 15) & 1u;
    const uint32_t exp = (h >> 10) & 0x1fu;
    const uint32_t frac = h & 0x3ffu;
    if (exp == 0) {
        const float value = std::ldexp(static_cast<float>(frac), -24);
        return sign ? -value : value;
    }
    if (exp == 0x1fu) {
        return frac != 0 ? NAN : (sign ? -INFINITY : INFINITY);
    }
    const float value = std::ldexp(static_cast<float>(1024u + frac),
                                   static_cast<int>(exp) - 25);
    return sign ? -value : value;
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

float fp16_round(float value) {
    return fp16_to_float(fp32_to_fp16_bits(value));
}

int32_t quant_int24(double value, int frac_bits) {
    constexpr int32_t q_min = -(1 << 23);
    constexpr int32_t q_max =  (1 << 23) - 1;
    if (!std::isfinite(value)) {
        return value < 0.0 ? q_min : q_max;
    }
    const double scaled = std::ldexp(value, frac_bits);
    const long long q = std::llround(scaled);
    return static_cast<int32_t>(std::max<long long>(q_min, std::min<long long>(q_max, q)));
}

bool read_exact(std::ifstream & in, std::vector<uint8_t> & dst, uint32_t bytes) {
    dst.resize(bytes);
    if (bytes == 0) {
        return true;
    }
    in.read(reinterpret_cast<char *>(dst.data()), bytes);
    return static_cast<bool>(in);
}

bool load_dump(const char * path, StreamDump * dump) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::perror(path);
        return false;
    }
    in.read(reinterpret_cast<char *>(&dump->hdr), sizeof(dump->hdr));
    if (!in ||
        std::memcmp(dump->hdr.magic, "AICAS_ATTNSTRM1", 15) != 0 ||
        dump->hdr.version != 1 ||
        dump->hdr.head_dim != 64 ||
        dump->hdr.group_size == 0 ||
        dump->hdr.kv_heads == 0 ||
        dump->hdr.seq_len == 0 ||
        dump->hdr.seq_len > dump->hdr.cache_capacity_tokens ||
        dump->hdr.score_prob_stride_bytes % kMvinAlign != 0 ||
        dump->hdr.kv_head_stride % kMvinAlign != 0 ||
        dump->hdr.cache_scale_region % kMvinAlign != 0 ||
        dump->hdr.v_row_tile_stride_bytes % kMvinAlign != 0) {
        std::fprintf(stderr, "bad AICAS_ATTNSTRM1 header\n");
        return false;
    }
    return read_exact(in, dump->q, dump->hdr.q_head_bytes) &&
           read_exact(in, dump->k, dump->hdr.cache_bytes) &&
           read_exact(in, dump->v, dump->hdr.cache_bytes) &&
           read_exact(in, dump->qk, dump->hdr.qk_compact_all_bytes) &&
           read_exact(in, dump->prob, dump->hdr.prob_padded_bytes) &&
           read_exact(in, dump->pv, dump->hdr.pv_heads_bytes);
}

uint32_t v_quant_offset(const StreamDumpHeader & h, uint32_t kv_head,
                        uint32_t token, uint32_t dim) {
    const uint32_t row_tile = dim / kRowTileElems;
    const uint32_t row_lane = dim % kRowTileElems;
    const uint32_t col_tile = token / kW8TileElems;
    const uint32_t col_lane = token % kW8TileElems;
    return kv_head * h.kv_head_stride +
           h.cache_scale_region +
           row_tile * h.v_row_tile_stride_bytes +
           col_tile * (kRowTileElems * kW8TileElems) +
           row_lane * kW8TileElems +
           col_lane;
}

float expected_pv_float(const StreamDump & dump, uint32_t kv_head,
                        uint32_t group, uint32_t dim) {
    const StreamDumpHeader & h = dump.hdr;
    const uint32_t prob_group =
        (kv_head * h.group_size + group) * h.score_prob_stride_bytes;
    float sum = 0.0f;
    for (uint32_t token = 0; token < h.seq_len; ++token) {
        const float prob = fp16_to_float(load_u16_le(
            dump.prob.data() + prob_group + token * kFp16Bytes));
        const float scale = fp16_to_float(load_u16_le(
            dump.v.data() + kv_head * h.kv_head_stride + token * kFp16Bytes));
        const int8_t value = static_cast<int8_t>(
            dump.v[v_quant_offset(h, kv_head, token, dim)]);
        sum += prob * scale * static_cast<float>(value);
    }
    return fp16_round(sum);
}

float expected_pv_rtl_int24(const StreamDump & dump, uint32_t kv_head,
                            uint32_t group, uint32_t dim) {
    const StreamDumpHeader & h = dump.hdr;
    const uint32_t prob_group =
        (kv_head * h.group_size + group) * h.score_prob_stride_bytes;
    int64_t acc = 0;
    for (uint32_t token = 0; token < h.seq_len; ++token) {
        const float prob = fp16_to_float(load_u16_le(
            dump.prob.data() + prob_group + token * kFp16Bytes));
        const float scale = fp16_to_float(load_u16_le(
            dump.v.data() + kv_head * h.kv_head_stride + token * kFp16Bytes));
        const float scaled_fp16 = fp16_round(prob * scale);
        const int32_t q24 = quant_int24(static_cast<double>(scaled_fp16), 16);
        const int8_t value = static_cast<int8_t>(
            dump.v[v_quant_offset(h, kv_head, token, dim)]);
        acc += static_cast<int64_t>(q24) * static_cast<int64_t>(value);
    }
    return fp16_round(static_cast<float>(std::ldexp(static_cast<double>(acc), -16)));
}

bool compare_output(const char * label, const StreamDump & dump,
                    const uint8_t * got, uint32_t kv_head) {
    const StreamDumpHeader & h = dump.hdr;
    uint32_t float_bad = 0;
    uint32_t rtl_bad = 0;
    float float_max = 0.0f;
    float rtl_max = 0.0f;
    for (uint32_t group = 0; group < h.group_size; ++group) {
        for (uint32_t dim = 0; dim < h.head_dim; ++dim) {
            const uint32_t idx = group * h.head_dim + dim;
            const float got_f = fp16_to_float(load_u16_le(got + idx * kFp16Bytes));
            const float exp_float = expected_pv_float(dump, kv_head, group, dim);
            const float exp_rtl = expected_pv_rtl_int24(dump, kv_head, group, dim);
            const float float_err = std::fabs(got_f - exp_float);
            const float rtl_err = std::fabs(got_f - exp_rtl);
            float_max = std::max(float_max, float_err);
            rtl_max = std::max(rtl_max, rtl_err);
            if (float_err > 0.08f) {
                ++float_bad;
            }
            if (rtl_err > 0.08f) {
                if (rtl_bad < 8) {
                    std::printf(
                        "%s kv_head=%u idx=%u group=%u dim=%u got=%g rtl=%g float=%g rtl_err=%g\n",
                        label, kv_head, idx, group, dim, got_f, exp_rtl,
                        exp_float, rtl_err);
                }
                ++rtl_bad;
            }
        }
    }
    std::printf(
        "%s kv_head=%u elems=%u float_bad=%u float_max=%g rtl_bad=%u rtl_max=%g\n",
        label, kv_head, h.group_size * h.head_dim, float_bad, float_max,
        rtl_bad, rtl_max);
    return rtl_bad == 0;
}

bool run() {
    const char * path = std::getenv("NPU_ATTN_STREAM_DUMP");
    if (path == nullptr || path[0] == '\0') {
        std::fprintf(stderr, "set NPU_ATTN_STREAM_DUMP=/path/to/AICAS_ATTNSTRM1.bin\n");
        return false;
    }
    StreamDump dump;
    if (!load_dump(path, &dump)) {
        return false;
    }
    const StreamDumpHeader & h = dump.hdr;
    std::printf(
        "stream_attn_pv_dump_repro layer=%u pos=%u seq_len=%u kv_heads=%u group=%u stride=%u v_stride=%u flags=0x%x\n",
        h.layer, h.position, h.seq_len, h.kv_heads, h.group_size,
        h.kv_head_stride, h.v_row_tile_stride_bytes, h.flags);

    NpuBuffer prob(align_up(h.prob_padded_bytes, kMvinAlign));
    NpuBuffer v_cache(align_up(h.cache_bytes, kMvinAlign));
    NpuBuffer out(align_up(h.group_size * h.head_dim * kFp16Bytes, kMvinAlign));
    if (!prob.ptr || !v_cache.ptr || !out.ptr) {
        std::fprintf(stderr, "npu_mem_alloc failed\n");
        return false;
    }
    std::memset(prob.data(), 0, prob.bytes);
    std::memcpy(prob.data(), dump.prob.data(), dump.prob.size());
    std::memset(v_cache.data(), 0, v_cache.bytes);
    std::memcpy(v_cache.data(), dump.v.data(), dump.v.size());

    npu_device * dev = nullptr;
    if (npu_open(&dev, "/dev/npu_kv260") != 0 || dev == nullptr) {
        std::fprintf(stderr, "npu_open failed\n");
        return false;
    }

    bool ok = true;
    for (uint32_t kv_head = 0; kv_head < h.kv_heads; ++kv_head) {
        std::memset(out.data(), 0, out.bytes);
        npu_stream_gemv_desc desc = {};
        desc.act_ptr =
            prob.data() + kv_head * h.group_size * h.score_prob_stride_bytes;
        desc.act_scale_ptr = v_cache.data() + kv_head * h.kv_head_stride;
        desc.weight_payload_ptr =
            v_cache.data() + kv_head * h.kv_head_stride + h.cache_scale_region;
        desc.weight_scale_ptr = nullptr;
        desc.output_ptr = out.ptr;
        desc.weight_row_tile_stride_bytes = h.v_row_tile_stride_bytes;
        desc.weight_capacity_tokens = static_cast<uint16_t>(h.cache_capacity_tokens);
        desc.m = static_cast<uint16_t>(h.head_dim);
        desc.n = static_cast<uint16_t>(h.seq_len);
        desc.mode = DECODE_GEMV_W8A16;
        desc.output_precision = DECODE_OUTPUT_FP16;
        desc.role = NPU_STREAM_GEMV_ROLE_PV;
        desc.dst = NPU_STREAM_GEMV_DST_OUTPUT;
        desc.post_op = NPU_STREAM_POST_BYPASS;
        desc.flags = NPU_STREAM_GEMV_F_ENABLE_ACT_SCALE |
                     NPU_STREAM_GEMV_F_KV_COL_SCALE |
                     NPU_STREAM_GEMV_F_UNIT_WEIGHT_SCALE;
        desc.elem_count = static_cast<uint16_t>(h.head_dim);
        desc.position = static_cast<uint16_t>(h.position);
        desc.group_count = static_cast<uint16_t>(h.group_size);
        desc.act_group_stride_bytes = h.score_prob_stride_bytes;

        npu_reset();
        const int rc = npu_stream_gemv_run(dev, &desc, 0);
        std::printf("stream_attn_pv_dump_repro kv_head=%u rc=%d\n", kv_head, rc);
        if (rc != 0) {
            ok = false;
            continue;
        }
        ok = compare_output("stream_attn_pv_dump_repro", dump, out.data(), kv_head) && ok;
    }
    npu_close(dev);
    std::puts(ok ? "kv260_stream_attn_pv_dump_repro_test=ok"
                 : "kv260_stream_attn_pv_dump_repro_test=fail");
    return ok;
}

}  // namespace

int main() {
    return run() ? 0 : 2;
}
