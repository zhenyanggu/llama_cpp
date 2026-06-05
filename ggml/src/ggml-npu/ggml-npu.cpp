#include "ggml-npu.h"

#include "ggml-npu-common.h"
#include "ggml-npu-exec.h"
#include "ggml-npu-plan.h"
#include "ggml-npu-profile.h"

#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "npu_runtime.h"
#include "npu_regs.h"
#include "../ggml-cpu/aicas-rtl-fp16.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cinttypes>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fcntl.h>
#include <fstream>
#include <limits>
#include <mutex>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <sys/mman.h>
#include <unistd.h>

#include <nlohmann/json.hpp>

extern "C" int npu_decode_init();
extern "C" int npu_decode_cma_init();
extern "C" void npu_decode_destroy();
extern "C" void npu_decode_reset();
extern "C" void * npu_decode_mem_alloc(size_t size);
extern "C" void npu_decode_mem_free(void * ptr);
extern "C" void * npu_decode_memory_base();
extern "C" uint32_t npu_decode_memory_size();

struct npu_device;


typedef enum kv260_stream_decode_gemv_mode {
    KV260_STREAM_DECODE_GEMV_W4A16 = 0,
    KV260_STREAM_DECODE_GEMV_W8A16 = 1,
    KV260_STREAM_DECODE_GEMV_W16A16 = 2,
} kv260_stream_decode_gemv_mode;

typedef enum kv260_stream_output_precision {
    KV260_STREAM_OUTPUT_FP16 = 0,
    KV260_STREAM_OUTPUT_FP32 = 1,
} kv260_stream_output_precision;

typedef enum kv260_stream_gemv_role {
    KV260_STREAM_GEMV_ROLE_LINEAR = 0,
    KV260_STREAM_GEMV_ROLE_QK = 1,
    KV260_STREAM_GEMV_ROLE_PV = 2,
    KV260_STREAM_GEMV_ROLE_KV_PROJ = 3,
    KV260_STREAM_GEMV_ROLE_MLP = 4,
} kv260_stream_gemv_role;

typedef enum kv260_stream_gemv_dst {
    KV260_STREAM_GEMV_DST_OUTPUT = 0,
    KV260_STREAM_GEMV_DST_ACT = 1,
    KV260_STREAM_GEMV_DST_POST = 2,
} kv260_stream_gemv_dst;

typedef enum kv260_stream_post_op {
    KV260_STREAM_POST_BYPASS = 0,
    KV260_STREAM_POST_ROPE = 1,
    KV260_STREAM_POST_SILU = 2,
    KV260_STREAM_POST_SOFTMAX = 3,
} kv260_stream_post_op;

enum {
    KV260_STREAM_GEMV_F_ENABLE_ACT_SCALE = 1u << 0,
    KV260_STREAM_GEMV_F_KV_COL_SCALE = 1u << 1,
    KV260_STREAM_GEMV_F_UNIT_WEIGHT_SCALE = 1u << 2,
};

typedef struct kv260_stream_gemv_desc {
    const void * act_ptr;
    const void * act_scale_ptr;
    const void * weight_payload_ptr;
    const void * weight_scale_ptr;
    void * output_ptr;
    const void * rope_lut_ptr;
    uint32_t weight_row_tile_stride_bytes;
    uint16_t weight_capacity_tokens;
    uint16_t m;
    uint16_t n;
    kv260_stream_decode_gemv_mode mode;
    kv260_stream_output_precision output_precision;
    kv260_stream_gemv_role role;
    kv260_stream_gemv_dst dst;
    kv260_stream_post_op post_op;
    uint32_t flags;
    uint16_t elem_count;
    uint16_t position;
    uint16_t group_count;
    uint32_t act_group_stride_bytes;
} kv260_stream_gemv_desc;

static_assert(offsetof(kv260_stream_gemv_desc, rope_lut_ptr) == 40, "decode stream ABI drift");
static_assert(offsetof(kv260_stream_gemv_desc, weight_row_tile_stride_bytes) == 48, "decode stream ABI drift");
static_assert(offsetof(kv260_stream_gemv_desc, weight_capacity_tokens) == 52, "decode stream ABI drift");
static_assert(offsetof(kv260_stream_gemv_desc, output_precision) == 64, "decode stream ABI drift");
static_assert(offsetof(kv260_stream_gemv_desc, act_group_stride_bytes) == 92, "decode stream ABI drift");
static_assert(sizeof(kv260_stream_gemv_desc) == 96, "decode stream ABI drift");

extern "C" int npu_open(npu_device ** out_dev, const char * dev_path);
extern "C" void npu_close(npu_device * dev);
extern "C" int npu_stream_gemv_run(npu_device * dev, const kv260_stream_gemv_desc * desc, uint32_t timeout_ms);

extern "C" bool ggml_backend_npu_text_log8pv_attention(
        const int8_t * q,
        const int8_t * k,
        const int8_t * v,
        uint32_t tokens,
        bool causal_mask,
        uint32_t gamma16_fix,
        int32_t * output,
        uint32_t output_stride_elems,
        uint32_t timeout_ms);

struct ggml_backend_npu_log8pv_attention_profile {
    uint32_t kv_tokens;
    uint32_t q_rows;
    uint32_t q_row_start;
    uint32_t exec_rows;
    uint32_t chunks;
    uint32_t group_size;
    uint32_t kv_reuse_hit;
    uint32_t workspace_reuse;
    uint64_t total_us;
    uint64_t mvin_q_us;
    uint64_t mvin_k_us;
    uint64_t mvin_v_us;
    uint64_t qk_us;
    uint64_t logp_mvout_us;
    uint64_t mvin_p_us;
    uint64_t pv_us;
    uint64_t pv_mvout_us;
    uint64_t qk_overlap_us;
    uint64_t pv_overlap_us;
};

extern "C" bool ggml_backend_npu_log8pv_attention_ex(
        const int8_t * q,
        uint32_t q_rows,
        const int8_t * k,
        const int8_t * v,
        uint32_t kv_tokens,
        uint32_t q_row_start,
        bool causal_mask,
        uint32_t gamma16_fix,
        int32_t * output,
        uint32_t output_stride_elems,
        uint32_t timeout_ms,
        ggml_backend_npu_log8pv_attention_profile * profile);

extern "C" bool ggml_backend_npu_log8pv_attention_group(
        const int8_t * q_group,
        uint32_t group_size,
        uint32_t q_rows,
        const int8_t * k,
        const int8_t * v,
        uint32_t kv_tokens,
        uint32_t q_row_start,
        bool causal_mask,
        const uint32_t * gamma16_fix_group,
        int32_t * output_group,
        uint32_t output_stride_elems,
        uint32_t timeout_ms,
        ggml_backend_npu_log8pv_attention_profile * profile_group);


namespace ggml_npu {

#include "ggml-npu-private-runtime.inc"
#include "ggml-npu-private-decode-helpers.inc"
#include "ggml-npu-private-backend.inc"

} // namespace ggml_npu

#include "ggml-npu-api-backend.inc"
#include "ggml-npu-api-prefill.inc"
#include "ggml-npu-api-decode-linear.inc"
#include "ggml-npu-api-decode-ffn-attn.inc"

GGML_BACKEND_DL_IMPL(ggml_backend_npu_reg)
