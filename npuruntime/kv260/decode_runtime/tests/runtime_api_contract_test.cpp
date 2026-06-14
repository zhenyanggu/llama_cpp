#include "npu_runtime.h"

#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <cstring>

namespace {

bool expect_rc(const char* label, int got, int expected) {
    if (got != expected) {
        std::fprintf(stderr, "%s: expected rc=%d got=%d\n", label, expected, got);
        return false;
    }
    return true;
}

} // namespace

int main() {
    std::puts("kv260_runtime_api_contract_test: stream GEMV API contract smoke");

    bool ok = true;
    ok = expect_rc("npu_stream_gemv_null_dev",
                   npu_stream_gemv_run(nullptr, nullptr, 0), -EINVAL) && ok;

    if (NPU_STREAM_GEMV_ROLE_LINEAR != 0 ||
        NPU_STREAM_GEMV_ROLE_QK != 1 ||
        NPU_STREAM_GEMV_ROLE_PV != 2 ||
        NPU_STREAM_GEMV_ROLE_KV_PROJ != 3 ||
        NPU_STREAM_GEMV_ROLE_MLP != 4 ||
        NPU_STREAM_GEMV_DST_OUTPUT != 0 ||
        NPU_STREAM_GEMV_DST_ACT != 1 ||
        NPU_STREAM_GEMV_DST_POST != 2 ||
        NPU_STREAM_POST_BYPASS != 0 ||
        NPU_STREAM_POST_ROPE != 1 ||
        NPU_STREAM_POST_SILU != 2 ||
        NPU_STREAM_POST_SOFTMAX != 3) {
        std::fprintf(stderr, "unexpected stream GEMV enum values\n");
        ok = false;
    }
    if ((NPU_STREAM_GEMV_F_ENABLE_ACT_SCALE |
         NPU_STREAM_GEMV_F_KV_COL_SCALE |
         NPU_STREAM_GEMV_F_UNIT_WEIGHT_SCALE |
         NPU_STREAM_GEMV_F_KV_QUANT |
         NPU_STREAM_GEMV_F_KV_IS_V |
         NPU_STREAM_GEMV_F_ENABLE_ACT_SCALE2) != 0x3fu) {
        std::fprintf(stderr, "unexpected stream GEMV flag values\n");
        ok = false;
    }
    npu_stream_kv_scale_result scale_result = {};
    ok = expect_rc("npu_stream_kv_scale_null_dev",
                   npu_stream_kv_scale_read(nullptr, 0, &scale_result), -EINVAL) && ok;
    if (NPU_ROPE_LUT_ACT_SPM_BASE != 0x3f00 ||
        NPU_ROPE_LUT_WINDOW_BYTES != 256 ||
        NPU_GEMV_MVIN_ALIGN_BYTES != 256) {
        std::fprintf(stderr, "unexpected stream GEMV alignment/RoPE constants\n");
        ok = false;
    }
    if (sizeof(npu_stream_gemv_desc) != 104) {
        std::fprintf(stderr, "unexpected stream GEMV descriptor size\n");
        ok = false;
    }
    if (offsetof(npu_stream_gemv_desc, act_ptr) != 0 ||
        offsetof(npu_stream_gemv_desc, act_scale_ptr) != 8 ||
        offsetof(npu_stream_gemv_desc, weight_payload_ptr) != 16 ||
        offsetof(npu_stream_gemv_desc, weight_scale_ptr) != 24 ||
        offsetof(npu_stream_gemv_desc, output_ptr) != 32 ||
        offsetof(npu_stream_gemv_desc, rope_lut_ptr) != 40 ||
        offsetof(npu_stream_gemv_desc, weight_row_tile_stride_bytes) != 48 ||
        offsetof(npu_stream_gemv_desc, weight_capacity_tokens) != 52 ||
        offsetof(npu_stream_gemv_desc, m) != 54 ||
        offsetof(npu_stream_gemv_desc, n) != 56 ||
        offsetof(npu_stream_gemv_desc, mode) != 60 ||
        offsetof(npu_stream_gemv_desc, output_precision) != 64 ||
        offsetof(npu_stream_gemv_desc, role) != 68 ||
        offsetof(npu_stream_gemv_desc, dst) != 72 ||
        offsetof(npu_stream_gemv_desc, post_op) != 76 ||
        offsetof(npu_stream_gemv_desc, flags) != 80 ||
        offsetof(npu_stream_gemv_desc, elem_count) != 84 ||
        offsetof(npu_stream_gemv_desc, position) != 86 ||
        offsetof(npu_stream_gemv_desc, group_count) != 88 ||
        offsetof(npu_stream_gemv_desc, act_group_stride_bytes) != 92 ||
        offsetof(npu_stream_gemv_desc, act_scale2_ptr) != 96) {
        std::fprintf(stderr, "unexpected stream GEMV descriptor layout\n");
        ok = false;
    }
    npu_device* dev = nullptr;
    int rc = npu_open(&dev, nullptr);
    if (rc != 0 || !dev) {
        std::fprintf(stderr, "npu_open failed rc=%d\n", rc);
        return 1;
    }

    void* buf = npu_mem_alloc(64 * 1024);
    if (!buf) {
        std::fprintf(stderr, "npu_mem_alloc failed\n");
        npu_close(dev);
        return 1;
    }

    uint32_t base = 0;
    ok = expect_rc("npu_mem_dma_offset", npu_mem_dma_offset(buf, &base), 0) && ok;
    uint32_t misaligned = 0;
    ok = expect_rc("npu_mem_dma_offset_misaligned",
                   npu_mem_dma_offset(static_cast<unsigned char*>(buf) + 1, &misaligned),
                   -EINVAL) && ok;

    npu_stream_gemv_desc stream = {};
    stream.act_ptr = buf;
    stream.weight_payload_ptr = buf;
    stream.weight_scale_ptr = buf;
    stream.output_ptr = buf;
    stream.m = 32;
    stream.n = 128;
    stream.mode = DECODE_GEMV_W4A16;
    stream.output_precision = DECODE_OUTPUT_FP16;
    stream.role = NPU_STREAM_GEMV_ROLE_LINEAR;
    stream.dst = NPU_STREAM_GEMV_DST_OUTPUT;
    stream.post_op = NPU_STREAM_POST_BYPASS;
    ok = expect_rc("npu_stream_gemv_timeout_reserved",
                   npu_stream_gemv_run(dev, &stream, 1), -ENOTSUP) && ok;
    stream.m = 0;
    ok = expect_rc("npu_stream_gemv_bad_shape",
                   npu_stream_gemv_run(dev, &stream, 0), -EINVAL) && ok;
    stream.m = 32;
    stream.weight_row_tile_stride_bytes = NPU_GEMV_MVIN_ALIGN_BYTES;
    ok = expect_rc("npu_stream_gemv_compact_bad_stride",
                   npu_stream_gemv_run(dev, &stream, 0), -EINVAL) && ok;
    stream.weight_row_tile_stride_bytes = 0;
    stream.post_op = NPU_STREAM_POST_ROPE;
    stream.dst = NPU_STREAM_GEMV_DST_ACT;
    ok = expect_rc("npu_stream_gemv_rope_missing_lut",
                   npu_stream_gemv_run(dev, &stream, 0), -EINVAL) && ok;
    stream.rope_lut_ptr = static_cast<unsigned char*>(buf) + 1;
    ok = expect_rc("npu_stream_gemv_rope_bad_lut_alignment",
                   npu_stream_gemv_run(dev, &stream, 0), -EINVAL) && ok;
    stream.rope_lut_ptr = nullptr;
    stream.post_op = NPU_STREAM_POST_BYPASS;
    stream.dst = NPU_STREAM_GEMV_DST_OUTPUT;
    stream.flags = NPU_STREAM_GEMV_F_ENABLE_ACT_SCALE2;
    stream.act_scale2_ptr = buf;
    ok = expect_rc("npu_stream_gemv_act_scale2_without_act_scale",
                   npu_stream_gemv_run(dev, &stream, 0), -EINVAL) && ok;
    stream.flags = NPU_STREAM_GEMV_F_ENABLE_ACT_SCALE |
                   NPU_STREAM_GEMV_F_ENABLE_ACT_SCALE2;
    stream.act_scale_ptr = buf;
    stream.act_scale2_ptr = static_cast<unsigned char*>(buf) + 1;
    ok = expect_rc("npu_stream_gemv_bad_act_scale2_alignment",
                   npu_stream_gemv_run(dev, &stream, 0), -EINVAL) && ok;
    stream.act_scale2_ptr = buf;
    stream.mode = DECODE_GEMV_W8A16;
    ok = expect_rc("npu_stream_gemv_act_scale2_w8_rejected",
                   npu_stream_gemv_run(dev, &stream, 0), -EINVAL) && ok;
    stream.mode = DECODE_GEMV_W4A16;
    stream.flags = 0;
    stream.act_scale_ptr = nullptr;
    stream.act_scale2_ptr = nullptr;
    stream.group_count = 2;
    ok = expect_rc("npu_stream_gemv_grouped_linear_rejected",
                   npu_stream_gemv_run(dev, &stream, 0), -EINVAL) && ok;
    stream.group_count = 1;
    stream.act_group_stride_bytes = 64;
    ok = expect_rc("npu_stream_gemv_bad_act_group_stride",
                   npu_stream_gemv_run(dev, &stream, 0), -EINVAL) && ok;
    stream.act_group_stride_bytes = 0;
    stream.post_op = static_cast<npu_stream_post_op>(99);
    ok = expect_rc("npu_stream_gemv_bad_post_op",
                   npu_stream_gemv_run(dev, &stream, 0), -EINVAL) && ok;
    stream.post_op = NPU_STREAM_POST_BYPASS;
    stream.dst = static_cast<npu_stream_gemv_dst>(99);
    ok = expect_rc("npu_stream_gemv_bad_dst",
                   npu_stream_gemv_run(dev, &stream, 0), -EINVAL) && ok;
    stream.dst = NPU_STREAM_GEMV_DST_OUTPUT;
    ok = expect_rc("npu_stream_gemv_timeout_reserved_again",
                   npu_stream_gemv_run(dev, &stream, 1), -ENOTSUP) && ok;

    npu_stream_gemv_desc pv = {};
    pv.act_ptr = buf;
    pv.act_scale_ptr = buf;
    pv.weight_payload_ptr = buf;
    pv.output_ptr = buf;
    pv.weight_capacity_tokens = 128;
    pv.weight_row_tile_stride_bytes = 4096;
    pv.m = 64;
    pv.n = 65;
    pv.mode = DECODE_GEMV_W8A16;
    pv.output_precision = DECODE_OUTPUT_FP16;
    pv.role = NPU_STREAM_GEMV_ROLE_PV;
    pv.dst = NPU_STREAM_GEMV_DST_OUTPUT;
    pv.post_op = NPU_STREAM_POST_BYPASS;
    pv.flags = NPU_STREAM_GEMV_F_ENABLE_ACT_SCALE |
               NPU_STREAM_GEMV_F_KV_COL_SCALE |
               NPU_STREAM_GEMV_F_UNIT_WEIGHT_SCALE;
    ok = expect_rc("npu_stream_gemv_pv_fixed_stride_timeout_reserved",
                   npu_stream_gemv_run(dev, &pv, 1), -ENOTSUP) && ok;
    pv.m = 32;
    ok = expect_rc("npu_stream_gemv_pv_bad_m",
                   npu_stream_gemv_run(dev, &pv, 0), -EINVAL) && ok;
    pv.m = 64;
    pv.weight_row_tile_stride_bytes = 2048;
    ok = expect_rc("npu_stream_gemv_pv_bad_stride",
                   npu_stream_gemv_run(dev, &pv, 0), -EINVAL) && ok;
    pv.weight_row_tile_stride_bytes = 4096;
    pv.weight_capacity_tokens = 64;
    ok = expect_rc("npu_stream_gemv_pv_bad_capacity",
                   npu_stream_gemv_run(dev, &pv, 0), -EINVAL) && ok;
    pv.weight_capacity_tokens = 128;
    pv.group_count = 5;
    ok = expect_rc("npu_stream_gemv_too_many_groups",
                   npu_stream_gemv_run(dev, &pv, 0), -EINVAL) && ok;
    pv.group_count = 3;
    pv.act_group_stride_bytes = 64;
    ok = expect_rc("npu_stream_gemv_grouped_pv_bad_stride",
                   npu_stream_gemv_run(dev, &pv, 0), -EINVAL) && ok;
    pv.act_group_stride_bytes = 256;
    pv.act_scale2_ptr = buf;
    pv.flags |= NPU_STREAM_GEMV_F_ENABLE_ACT_SCALE2;
    ok = expect_rc("npu_stream_gemv_grouped_pv_act_scale2_rejected",
                   npu_stream_gemv_run(dev, &pv, 0), -EINVAL) && ok;

    npu_stream_gemv_desc qk = {};
    qk.act_ptr = buf;
    qk.weight_payload_ptr = buf;
    qk.weight_scale_ptr = buf;
    qk.output_ptr = buf;
    qk.m = 96;
    qk.n = 64;
    qk.mode = DECODE_GEMV_W8A16;
    qk.output_precision = DECODE_OUTPUT_FP16;
    qk.role = NPU_STREAM_GEMV_ROLE_QK;
    qk.dst = NPU_STREAM_GEMV_DST_OUTPUT;
    qk.post_op = NPU_STREAM_POST_SOFTMAX;
    qk.elem_count = qk.m;
    qk.group_count = 3;
    qk.act_group_stride_bytes = 64;
    ok = expect_rc("npu_stream_gemv_grouped_qk_bad_stride",
                   npu_stream_gemv_run(dev, &qk, 0), -EINVAL) && ok;
    qk.act_group_stride_bytes = 256;
    qk.act_scale_ptr = buf;
    qk.act_scale2_ptr = buf;
    qk.flags = NPU_STREAM_GEMV_F_ENABLE_ACT_SCALE |
               NPU_STREAM_GEMV_F_ENABLE_ACT_SCALE2;
    ok = expect_rc("npu_stream_gemv_grouped_qk_act_scale2_rejected",
                   npu_stream_gemv_run(dev, &qk, 0), -EINVAL) && ok;

    npu_stream_gemv_desc kv = {};
    kv.act_ptr = buf;
    kv.act_scale_ptr = buf;
    kv.weight_payload_ptr = buf;
    kv.weight_scale_ptr = buf;
    kv.output_ptr = buf;
    kv.rope_lut_ptr = buf;
    kv.m = 320;
    kv.n = 128;
    kv.mode = DECODE_GEMV_W4A16;
    kv.output_precision = DECODE_OUTPUT_FP16;
    kv.role = NPU_STREAM_GEMV_ROLE_KV_PROJ;
    kv.dst = NPU_STREAM_GEMV_DST_OUTPUT;
    kv.post_op = NPU_STREAM_POST_BYPASS;
    kv.flags = NPU_STREAM_GEMV_F_ENABLE_ACT_SCALE |
               NPU_STREAM_GEMV_F_KV_IS_V;
    kv.elem_count = 320;
    ok = expect_rc("npu_stream_gemv_kv_is_v_without_quant",
                   npu_stream_gemv_run(dev, &kv, 0), -EINVAL) && ok;
    kv.flags = NPU_STREAM_GEMV_F_ENABLE_ACT_SCALE |
               NPU_STREAM_GEMV_F_KV_QUANT;
    ok = expect_rc("npu_stream_gemv_k_quant_requires_rope",
                   npu_stream_gemv_run(dev, &kv, 0), -EINVAL) && ok;
    kv.flags = NPU_STREAM_GEMV_F_ENABLE_ACT_SCALE |
               NPU_STREAM_GEMV_F_KV_QUANT |
               NPU_STREAM_GEMV_F_KV_IS_V;
    kv.post_op = NPU_STREAM_POST_ROPE;
    ok = expect_rc("npu_stream_gemv_v_quant_rejects_rope",
                   npu_stream_gemv_run(dev, &kv, 0), -EINVAL) && ok;

    npu_mem_free(buf);
    npu_close(dev);

    std::puts(ok ? "kv260_runtime_api_contract_test=ok" :
                   "kv260_runtime_api_contract_test=fail");
    return ok ? 0 : 2;
}
