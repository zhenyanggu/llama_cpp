#ifndef NPU_RUNTIME_H
#define NPU_RUNTIME_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    NPU_GEMV_TILE_ELEMS = 128,
    NPU_GEMV_ROW_TILE_ELEMS = 32,
    NPU_GEMV_LINE_BYTES = 64,
    /*
     * GEMV/decode MVIN hardware issues 256B AXI bursts from a 256B-aligned
     * physical base. This contract keeps every burst inside one 4KB AXI
     * boundary; callers must not pass an unaligned CMA sub-buffer.
     */
    NPU_GEMV_MVIN_ALIGN_BYTES = 256,
    NPU_GEMV_ACT_BUFFER_BYTES = 256 * 64,
    NPU_GEMV_ACT_PAYLOAD_BYTES = 0x3f00,
    NPU_GEMV_SPM_BYTES = 512 * 1024,
    NPU_GEMV_FP16_BYTES = 2,
    NPU_ROPE_LUT_ACT_SPM_BASE = 0x3f00,
    NPU_ROPE_LUT_WINDOW_BYTES = 256,
};

typedef struct npu_device npu_device;

typedef enum decode_gemv_mode {
    DECODE_GEMV_W4A16 = 0,
    DECODE_GEMV_W8A16 = 1,
    /* ABI-reserved. Current decode runtime split-GEMV path supports W4A16/W8A16 only. */
    DECODE_GEMV_W16A16 = 2,
} decode_gemv_mode;

typedef enum decode_output_precision {
    DECODE_OUTPUT_FP16 = 0,
    DECODE_OUTPUT_FP32 = 1,
} decode_output_precision;

typedef enum npu_stream_gemv_role {
    NPU_STREAM_GEMV_ROLE_LINEAR = 0,
    NPU_STREAM_GEMV_ROLE_QK     = 1,
    NPU_STREAM_GEMV_ROLE_PV     = 2,
    NPU_STREAM_GEMV_ROLE_KV_PROJ = 3,
    NPU_STREAM_GEMV_ROLE_MLP    = 4,
} npu_stream_gemv_role;

typedef enum npu_stream_gemv_dst {
    NPU_STREAM_GEMV_DST_OUTPUT = 0,
    NPU_STREAM_GEMV_DST_ACT    = 1,
    NPU_STREAM_GEMV_DST_POST   = 2,
} npu_stream_gemv_dst;

typedef enum npu_stream_post_op {
    NPU_STREAM_POST_BYPASS  = 0,
    NPU_STREAM_POST_ROPE    = 1,
    NPU_STREAM_POST_SILU    = 2,
    NPU_STREAM_POST_SOFTMAX = 3,
} npu_stream_post_op;

enum {
    NPU_STREAM_GEMV_F_ENABLE_ACT_SCALE  = 1u << 0,
    NPU_STREAM_GEMV_F_KV_COL_SCALE      = 1u << 1,
    NPU_STREAM_GEMV_F_UNIT_WEIGHT_SCALE = 1u << 2,
    NPU_STREAM_GEMV_F_KV_QUANT          = 1u << 3,
    NPU_STREAM_GEMV_F_KV_IS_V           = 1u << 4,
    NPU_STREAM_GEMV_F_ENABLE_ACT_SCALE2 = 1u << 5,
};

typedef struct npu_stream_kv_scale_result {
    uint16_t values[5];
    uint8_t count;
    uint8_t seq;
    uint8_t valid;
    uint8_t reserved;
} npu_stream_kv_scale_result;

typedef struct npu_stream_gemv_desc {
    /*
     * Production public GEMV API. All pointers must resolve to 256B-aligned
     * NPU/CMA physical addresses; the runtime rejects non-CMA pointers.
     *
     * Stream output supports FP16 or FP32 host writeback. The GEMV output SPM
     * stores FP16; FP32 output expands those values during MVOUT.
     * The logical output is m * group_count elements, but callers should keep
     * at least NPU_GEMV_LINE_BYTES-rounded host space available after the
     * logical output because the output-SPM MVOUT path is line-granular.
     *
     * QK score GEMV uses role=NPU_STREAM_GEMV_ROLE_QK, W8A16 mode, n=64, and
     * a non-null weight_scale_ptr. With post_op=NPU_STREAM_POST_SOFTMAX, the
     * hardware applies the SmolVLM2 head_dim=64 score scale before softmax:
     * probability = softmax(dot / 8). Grouped QK output is compact
     * group-major: group0 probs, then group1 probs, with no 64B padding
     * between groups.
     *
     * PV/value GEMV uses role=NPU_STREAM_GEMV_ROLE_PV, W8A16 mode,
     * fixed-stride pre-transposed V payload,
     * NPU_STREAM_GEMV_F_UNIT_WEIGHT_SCALE, and
     * NPU_STREAM_GEMV_F_KV_COL_SCALE. act_scale_ptr points to one shared
     * V-scale vector for the active attention window; do not duplicate it per
     * group. Grouped PV output is group-major: group0 dims, then group1 dims.
     */
    const void* act_ptr;
    const void* act_scale_ptr;
    const void* weight_payload_ptr;
    const void* weight_scale_ptr;
    void* output_ptr;
    const void* rope_lut_ptr; /* required when post_op == NPU_STREAM_POST_ROPE */

    uint32_t weight_row_tile_stride_bytes;
    uint16_t weight_capacity_tokens;

    uint16_t m;
    uint16_t n;
    decode_gemv_mode mode;
    decode_output_precision output_precision;

    npu_stream_gemv_role role;
    npu_stream_gemv_dst dst;
    npu_stream_post_op post_op;
    uint32_t flags;

    uint16_t elem_count;
    uint16_t position;
    uint16_t group_count;
    uint32_t act_group_stride_bytes;
    /*
     * Optional second activation scale vector. Intended for FFN down/SwiGLU
     * when silu(gate) and the down-projection smooth scale must remain
     * separate in memory. Requires both ENABLE_ACT_SCALE and
     * ENABLE_ACT_SCALE2; hardware serializes the two scale passes through the
     * same FP16 multiplier.
     */
    const void* act_scale2_ptr;
} npu_stream_gemv_desc;

int npu_init(void);
int npu_decode_cma_init(void);
void npu_destroy(void);
void npu_reset(void);
int npu_reinit(void);

int npu_open(npu_device** out_dev, const char* dev_path);
void npu_close(npu_device* dev);

void* npu_mem_alloc(size_t size);
void npu_mem_free(void* ptr);
void* npu_decode_memory_base(void);
uint32_t npu_decode_memory_size(void);
int npu_mem_dma_offset(const void* ptr, uint32_t* out_offset);

int npu_stream_gemv_run(npu_device* dev, const npu_stream_gemv_desc* desc,
                        uint32_t timeout_ms);
int npu_stream_kv_scale_read(npu_device* dev, int is_v,
                             npu_stream_kv_scale_result* out);

#ifdef __cplusplus
}
#endif

#endif
