#ifndef NPU_REGS_COMPAT_H
#define NPU_REGS_COMPAT_H

/**
 * @file npu_regs_compat.h
 * @brief 寄存器兼容层 - 桥接 RDL 自动生成的头文件与 Runtime
 * 
 * 本文件提供从 PeakRDL 生成的 npu_regs.h 到 Runtime 使用格式的映射。
 * 当 RDL 定义变化时，只需重新生成 npu_regs.h，本文件会自动适配。
 * 
 * 使用方法:
 *   1. 修改 doc/rdl/npu_regs.rdl
 *   2. 运行 make (在 doc/rdl 目录)
 *   3. 重新编译 runtime
 */

#include <cstddef>  // for offsetof
#include "npu_regs.h"

// ==========================================
// 寄存器偏移量定义 (基于 npu_regs_t 结构体)
// ==========================================

namespace RegOffset {
    // 使用 offsetof 从结构体自动计算偏移量
    constexpr uint32_t MVIN_DRAM_ADDR = offsetof(npu_regs_t, MVIN_CTRL0);   // 0x00
    constexpr uint32_t MVIN_SRAM_ADDR = offsetof(npu_regs_t, MVIN_CTRL1);   // 0x08
    constexpr uint32_t MVOUT_DRAM_ADDR = offsetof(npu_regs_t, MVOUT_CTRL0); // 0x10
    constexpr uint32_t MVOUT_SRAM_ADDR = offsetof(npu_regs_t, MVOUT_CTRL1); // 0x18
    constexpr uint32_t MVIN_CFG       = offsetof(npu_regs_t, CFG_MVIN0);    // 0x20
    constexpr uint32_t MVIN_QUANT     = offsetof(npu_regs_t, CFG_MVIN1);    // 0x28
    constexpr uint32_t MVOUT_CFG      = offsetof(npu_regs_t, CFG_MVOUT0);   // 0x30
    constexpr uint32_t MVOUT_QUANT    = offsetof(npu_regs_t, CFG_MVOUT1);   // 0x38
    
    constexpr uint32_t CFG_COMPUTE_1  = offsetof(npu_regs_t, CFG_COMPUTE0); // 0x40
    constexpr uint32_t CFG_COMPUTE_2  = offsetof(npu_regs_t, CFG_COMPUTE1); // 0x48
    constexpr uint32_t CFG_SFU_1      = offsetof(npu_regs_t, CFG_SFU0);     // 0x50
    constexpr uint32_t CFG_SFU_2      = offsetof(npu_regs_t, CFG_SFU1);     // 0x58
    constexpr uint32_t CFG_ACCU_1     = offsetof(npu_regs_t, CFG_ACCU0);    // 0x60
    constexpr uint32_t CFG_ACCU_2     = offsetof(npu_regs_t, CFG_ACCU1);    // 0x68
    constexpr uint32_t SA_INPUT_A     = offsetof(npu_regs_t, SA_IN_A);      // 0x70
    constexpr uint32_t SA_INPUT_B     = offsetof(npu_regs_t, SA_IN_B);      // 0x78
    
    constexpr uint32_t MATADD_CTRL_0  = offsetof(npu_regs_t, MATADD_CTRL0); // 0x80
    constexpr uint32_t MATADD_CTRL_1  = offsetof(npu_regs_t, MATADD_CTRL1); // 0x88
    constexpr uint32_t MATVEC_CTRL_0  = offsetof(npu_regs_t, MATVEC_CTRL0); // 0x90
    constexpr uint32_t MATVEC_CTRL_1  = offsetof(npu_regs_t, MATVEC_CTRL1); // 0x98
    
    constexpr uint32_t SFU_INPUT      = offsetof(npu_regs_t, SFU_EXE0);     // 0xA0
    constexpr uint32_t SFU_OUTPUT     = offsetof(npu_regs_t, SFU_EXE1);     // 0xA8
    constexpr uint32_t START          = offsetof(npu_regs_t, START_REG);    // 0xB0
    constexpr uint32_t IAR            = offsetof(npu_regs_t, IAR);          // 0xB8
    constexpr uint32_t MER            = offsetof(npu_regs_t, MER);          // 0xC0
    constexpr uint32_t IER            = offsetof(npu_regs_t, IER);          // 0xC8
    constexpr uint32_t ISR            = offsetof(npu_regs_t, ISR);          // 0xD0
    constexpr uint32_t IPR            = offsetof(npu_regs_t, IPR);          // 0xD8
}

// ==========================================
// 启动位定义 (来自 npu_regs.h 的宏)
// ==========================================

#define BIT_START_DMA_MVIN  NPU_REGS__START_REG__MVIN_bm
#define BIT_START_DMA_MVOUT NPU_REGS__START_REG__MVOUT_bm
#define BIT_START_SA        NPU_REGS__START_REG__SA_bm
#define BIT_START_MATADD    NPU_REGS__START_REG__MATADD_bm
#define BIT_START_MATVEC    NPU_REGS__START_REG__MATVEC_bm
#define BIT_START_SFU       NPU_REGS__START_REG__SFU_bm

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
