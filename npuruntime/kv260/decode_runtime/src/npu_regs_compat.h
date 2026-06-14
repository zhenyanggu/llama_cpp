#ifndef NPU_REGS_COMPAT_H
#define NPU_REGS_COMPAT_H

/**
 * @file npu_regs_compat.h
 * @brief 寄存器兼容层 - 桥接 RDL 自动生成的头文件与 Runtime
 * 
 * 本文件提供从 PeakRDL 生成的 npu_regs.h 到 Runtime 使用格式的映射。
 * ABI v2 公共偏移从 npu_regs.h 的显式 offset 常量派生，避免 runtime 与 RTL 分叉。
 * 
 * 使用方法:
 *   1. 修改 doc/rdl/npu_regs.rdl
 *   2. 运行 make (在 doc/rdl 目录)
 *   3. 重新编译 runtime
 */

#include <cstddef>  // for offsetof
#include "npu_regs.h"

// ==========================================
// 寄存器偏移量定义 (decode inst ctrl ABI v2)
// ==========================================

namespace RegOffset {
    constexpr uint32_t IAR            = NPU_REGS__IAR_offset;
    constexpr uint32_t MER            = NPU_REGS__MER_offset;
    constexpr uint32_t IER            = NPU_REGS__IER_offset;
    constexpr uint32_t ISR            = NPU_REGS__ISR_offset;
    constexpr uint32_t IPR            = NPU_REGS__IPR_offset;
    constexpr uint32_t GLOBAL_CLEAR   = NPU_REGS__GLOBAL_CLEAR_offset;
    constexpr uint32_t PROFILE0       = NPU_REGS__PROFILE0_offset;

    constexpr uint32_t GEMV_BLOCK_DESC0  = NPU_REGS__GEMV_BLOCK_DESC0_offset;
    constexpr uint32_t GEMV_BLOCK_DESC1  = NPU_REGS__GEMV_BLOCK_DESC1_offset;
    constexpr uint32_t GEMV_BLOCK_DESC2  = NPU_REGS__GEMV_BLOCK_DESC2_offset;
    constexpr uint32_t GEMV_BLOCK_DESC3  = NPU_REGS__GEMV_BLOCK_DESC3_offset;
    constexpr uint32_t GEMV_BLOCK_DESC4  = NPU_REGS__GEMV_BLOCK_DESC4_offset;
    constexpr uint32_t GEMV_BLOCK_DESC5  = NPU_REGS__GEMV_BLOCK_DESC5_offset;
    constexpr uint32_t GEMV_BLOCK_DESC6  = NPU_REGS__GEMV_BLOCK_DESC6_offset;
    constexpr uint32_t GEMV_BLOCK_DESC7  = NPU_REGS__GEMV_BLOCK_DESC7_offset;
    constexpr uint32_t GEMV_BLOCK_CTRL   = NPU_REGS__GEMV_BLOCK_CTRL_offset;
    constexpr uint32_t GEMV_BLOCK_STATUS = NPU_REGS__GEMV_BLOCK_STATUS_offset;

    constexpr uint32_t MLP_DESC0      = NPU_REGS__MLP_DESC0_offset;
    constexpr uint32_t MLP_DESC1      = NPU_REGS__MLP_DESC1_offset;
    constexpr uint32_t MLP_DESC2      = NPU_REGS__MLP_DESC2_offset;
    constexpr uint32_t MLP_DESC3      = NPU_REGS__MLP_DESC3_offset;
    constexpr uint32_t MLP_CTRL       = NPU_REGS__MLP_CTRL_offset;
    constexpr uint32_t MLP_STATUS     = NPU_REGS__MLP_STATUS_offset;

    constexpr uint32_t ATTN_DESC0     = NPU_REGS__ATTN_DESC0_offset;
    constexpr uint32_t ATTN_DESC1     = NPU_REGS__ATTN_DESC1_offset;
    constexpr uint32_t ATTN_DESC2     = NPU_REGS__ATTN_DESC2_offset;
    constexpr uint32_t ATTN_DESC3     = NPU_REGS__ATTN_DESC3_offset;
    constexpr uint32_t ATTN_DESC4     = NPU_REGS__ATTN_DESC4_offset;
    constexpr uint32_t ATTN_DESC5     = NPU_REGS__ATTN_DESC5_offset;
    constexpr uint32_t ATTN_DESC6     = NPU_REGS__ATTN_DESC6_offset;
    constexpr uint32_t ATTN_CTRL      = NPU_REGS__ATTN_CTRL_offset;
    constexpr uint32_t ATTN_STATUS    = NPU_REGS__ATTN_STATUS_offset;

    constexpr uint32_t KV_K_SCALE0      = NPU_REGS__KV_K_SCALE0_offset;
    constexpr uint32_t KV_K_SCALE1      = NPU_REGS__KV_K_SCALE1_offset;
    constexpr uint32_t KV_V_SCALE0      = NPU_REGS__KV_V_SCALE0_offset;
    constexpr uint32_t KV_V_SCALE1      = NPU_REGS__KV_V_SCALE1_offset;

    constexpr uint32_t GENERIC_MAGIC   = NPU_REGS__GENERIC_MAGIC_offset;
    constexpr uint32_t GENERIC_VERSION = NPU_REGS__GENERIC_VERSION_offset;
    constexpr uint32_t GENERIC_MODE    = NPU_REGS__GENERIC_MODE_offset;
    constexpr uint32_t GENERIC_CAPS    = NPU_REGS__GENERIC_CAPS_offset;
    constexpr uint32_t GENERIC_CONTROL = NPU_REGS__GENERIC_CONTROL_offset;
    constexpr uint32_t GENERIC_STATUS  = NPU_REGS__GENERIC_STATUS_offset;
    constexpr uint32_t GENERIC_ERROR   = NPU_REGS__GENERIC_ERROR_offset;
}

// ==========================================
// 启动位定义 (来自 npu_regs.h 的宏)
// ==========================================

#undef NPU_REGS__IAR__ACK_bm
#undef NPU_REGS__IAR__ACK_bp
#undef NPU_REGS__IAR__ACK_bw
#define NPU_REGS__IAR__ACK_bm 0xffu
#define NPU_REGS__IAR__ACK_bp 0
#define NPU_REGS__IAR__ACK_bw 8

#define NPU_IRQ_GEMV_BLOCK  (1u << 0)
#define NPU_IRQ_MLP         (1u << 1)
#define NPU_IRQ_ATTENTION   (1u << 2)
#define NPU_IRQ_ERROR       (1u << 7)

#define NPU_REGS__CFG_COMPUTE0__GEMV_MODE_bm NPU_REGS__CFG_COMPUTE0__INT_TYPE_bm
#define NPU_REGS__CFG_COMPUTE0__GEMV_MODE_bp NPU_REGS__CFG_COMPUTE0__INT_TYPE_bp
#define NPU_REGS__CFG_COMPUTE0__GEMV_MODE_bw NPU_REGS__CFG_COMPUTE0__INT_TYPE_bw

#define NPU_GEMV_MODE_W4A16  0
#define NPU_GEMV_MODE_W8A16  1
/* ABI-reserved. Current KV260 decode RTL rejects debug GEMV mode 2. */
#define NPU_GEMV_MODE_W16A16 2

#define NPU_DECODE_SRC_GEMV_STREAM   1
#define NPU_DECODE_SRC_POST_BUFFER   2
#define NPU_DECODE_SRC_STREAM_BUFFER NPU_DECODE_SRC_POST_BUFFER

#define NPU_DECODE_UNARY_BYPASS 0
#define NPU_DECODE_UNARY_SILU   1
#define NPU_DECODE_UNARY_ROPE   2

#define NPU_DECODE_BINARY_BYPASS   0
#define NPU_DECODE_BINARY_FP16_MUL 1

#define NPU_DECODE_REDUCE_BYPASS  0
#define NPU_DECODE_REDUCE_SOFTMAX 1

#define NPU_DECODE_DST_OUTPUT_SPM    1
#define NPU_DECODE_DST_ACT_BUFFER    2
#define NPU_DECODE_DST_POST_BUFFER   3
#define NPU_DECODE_DST_STREAM_BUFFER NPU_DECODE_DST_POST_BUFFER

#define NPU_DECODE_FLAG_KV_QUANT     (1ull << 24)
#define NPU_DECODE_FLAG_KV_COL_SCALE (1ull << 27)
#define NPU_DECODE_FLAG_KV_V_SEPARATED (1ull << 28)
/* For attention PV/value GEMV: use FP16 1.0 for weight scale internally. */
#define NPU_DECODE_FLAG_UNIT_WEIGHT_SCALE (1ull << 29)
#define NPU_DECODE_FLAG_KV_QUANT_SCRATCH (1ull << 30)
#define NPU_DECODE_FLAG_PV_UQ24          (1ull << 31)

// ==========================================
// SFU 操作码
// ==========================================

#define SFU_OP_SOFTMAX          0
#define SFU_OP_GELU             1
#define SFU_OP_LAYERNORM        2
#define SFU_OP_DOWNSAMPLE_MAX   3
#define SFU_OP_DOWNSAMPLE_AVG   4
#define SFU_OP_UPSAMPLE_NEAREST 5
#define SFU_OP_TRANSPOSE        8

// Resample 类型定义 (对应 resample_type 信号)
#define RESAMPLE_TYPE_DOWNSAMPLE    0   // 下采样 (2x2 -> 1x1)
#define RESAMPLE_TYPE_UPSAMPLE      1   // 上采样 (1x1 -> 2x2)
#define RESAMPLE_TYPE_POOLING       2   // 池化

// Resample 操作定义 (对应 resample_op 信号)
#define RESAMPLE_OP_MAX             0   // 最大值 (下采样/池化) 或 最近邻 (上采样)
#define RESAMPLE_OP_AVG             1   // 平均值 (下采样/池化) 或 双线性 (上采样, 暂不支持)

// ==========================================
// 位域辅助宏 - 用于构建寄存器值
// ==========================================

// 通用位域构建宏: 将 value 按照 REG__FIELD_bp 和 FIELD_bm 放置
#define REG_FIELD(reg, field, value) \
    (((uint64_t)(value) << NPU_REGS__##reg##__##field##_bp) & NPU_REGS__##reg##__##field##_bm)

// 示例用法:
// uint64_t cfg = REG_FIELD(CFG_MVIN0, INPUT_TYPE, 1) |
//                REG_FIELD(CFG_MVIN0, INPUT_PRECISION, 0) |
//                REG_FIELD(CFG_MVIN0, IS_QUANT, 1);

// ==========================================
// 位域提取宏 - 用于解析寄存器值
// ==========================================

#define REG_GET_FIELD(reg, field, value) \
    (((value) & NPU_REGS__##reg##__##field##_bm) >> NPU_REGS__##reg##__##field##_bp)

#endif // NPU_REGS_COMPAT_H
