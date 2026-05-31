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
    NPU_GEMV_MVIN_ALIGN_BYTES = 256,
    NPU_GEMV_ACT_BUFFER_BYTES = 256 * 64,
    NPU_GEMV_ACT_PAYLOAD_BYTES = 0x3f00,
    NPU_GEMV_SPM_BYTES = 512 * 1024,
    NPU_GEMV_FP16_BYTES = 2,
    NPU_ROPE_LUT_ACT_SPM_BASE = 0x3f00,
    NPU_ROPE_LUT_WINDOW_BYTES = 256,
};

enum {
    NPU_GEMV_MVIN_INPUT_SPM = 0,
    NPU_GEMV_MVIN_WEIGHT = 1,
    NPU_GEMV_MVIN_ACT = 3,
};

struct MvinConfig {
    void* host_ptr;
    uint32_t sram_addr;
    uint32_t col_num;
    uint32_t row_num;
    uint16_t sram_stride;
    uint32_t dram_stride;
    uint8_t  precision;
    uint8_t  input_type;
    bool     dest;
    bool     is_bias;
    bool     is_quant;
    uint32_t quant_zero;
    uint16_t quant_scale;
    uint16_t quant_shift;
};

struct MvoutConfig {
    void* host_ptr;
    uint32_t sram_addr;
    uint32_t col_num;
    uint32_t row_num;
    uint16_t sram_stride;
    uint32_t dram_stride;
    uint8_t  precision;
    uint8_t  output_type;
    bool     source;
    bool     is_quant;
    uint32_t quant_zero;
    uint32_t scale_or_addr;
    bool     per_channel;
};

struct MatvecConfig {
    uint32_t mat_addr;
    uint32_t vec_addr;
    uint16_t mat_width;
    uint16_t mat_height;
    uint16_t output_addr;
    uint16_t scale_addr;
    uint32_t act_scale_addr;
    uint8_t  gemv_mode;
    uint64_t decode_flow;
};

struct GemvPingPongConfig {
    void* act_ptr;
    void* scale_ptr;
    void* weight_ptr;
    void* output_ptr;
    uint16_t m;
    uint16_t n;
    uint8_t gemv_mode;
    uint8_t output_precision;
};

int npu_init(void);
void npu_destroy(void);
void npu_reset(void);
int npu_reinit(void);

void* npu_mem_alloc(size_t size);
void npu_mem_free(void* ptr);
void* npu_decode_memory_base(void);
uint32_t npu_decode_memory_size(void);

void npu_dma_mvin(
    void* host_ptr,
    uint32_t sram_addr,
    uint32_t col_num,
    uint32_t row_num,
    uint16_t sram_stride,
    uint32_t dram_stride,
    uint8_t  precision,
    uint8_t  input_type,
    bool     dest,
    bool     is_bias,
    bool     is_quant,
    uint32_t quant_zero,
    uint16_t quant_scale,
    uint16_t quant_shift
);

void npu_dma_mvout(
    void* host_ptr,
    uint32_t sram_addr,
    uint32_t col_num,
    uint32_t row_num,
    uint16_t sram_stride,
    uint32_t dram_stride,
    uint8_t  precision,
    uint8_t  output_type,
    bool     source,
    bool     is_quant,
    uint32_t quant_zero,
    uint32_t scale_or_addr
);

void npu_dma_mvin_async(uint32_t dma_id, const struct MvinConfig* cfg);
void npu_dma_mvout_async(uint32_t dma_id, const struct MvoutConfig* cfg);
void npu_dma_wait_mvin(uint32_t dma_mask);
void npu_dma_wait_mvout(uint32_t dma_mask);

void npu_rope_lut_mvin(void* rope_table_base, uint16_t position);
void npu_rope_lut_mvin_async(uint32_t dma_id, void* rope_table_base, uint16_t position);

void npu_matvec_run(
    uint32_t mat_addr,
    uint32_t vec_addr,
    uint16_t mat_width,
    uint16_t mat_height,
    uint16_t output_addr,
    uint16_t scale_addr
);

void npu_matvec_mode_run(
    uint32_t mat_addr,
    uint32_t vec_addr,
    uint16_t mat_width,
    uint16_t mat_height,
    uint16_t output_addr,
    uint16_t scale_addr,
    uint8_t  gemv_mode
);

void npu_matvec_silu_run(
    uint32_t mat_addr,
    uint32_t vec_addr,
    uint16_t mat_width,
    uint16_t mat_height,
    uint16_t output_addr,
    uint16_t scale_addr
);

uint64_t npu_decode_flow_make(
    uint8_t src0,
    uint8_t src1,
    uint8_t unary_op,
    uint8_t binary_op,
    uint8_t reduce_op,
    uint8_t dst,
    uint8_t src_buffer_id,
    uint8_t dst_buffer_id,
    uint16_t elem_count,
    uint16_t position
);

void npu_matvec_decode_flow_run(
    uint32_t mat_addr,
    uint32_t vec_addr,
    uint16_t mat_width,
    uint16_t mat_height,
    uint16_t output_addr,
    uint16_t scale_addr,
    uint8_t  gemv_mode,
    uint64_t decode_flow
);

void npu_gemv_pingpong_run(
    void* act_ptr,
    void* scale_ptr,
    void* weight_ptr,
    void* output_ptr,
    uint16_t m,
    uint16_t n
);

void npu_gemv_pingpong_mode_run(
    void* act_ptr,
    void* scale_ptr,
    void* weight_ptr,
    void* output_ptr,
    uint16_t m,
    uint16_t n,
    uint8_t gemv_mode
);

void npu_gemv_pingpong_mode_precision_run(
    void* act_ptr,
    void* scale_ptr,
    void* weight_ptr,
    void* output_ptr,
    uint16_t m,
    uint16_t n,
    uint8_t gemv_mode,
    uint8_t output_precision
);

#ifdef __cplusplus
}
#endif

#endif
