#ifndef VERSA_P_REGS_H
#define VERSA_P_REGS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VERSA_P_REG_MVIN_A_DESC0       0x000u
#define VERSA_P_REG_MVIN_A_DESC1       0x008u
#define VERSA_P_REG_MVIN_A_STATUS      0x010u
#define VERSA_P_REG_MVIN_W_DESC0       0x020u
#define VERSA_P_REG_MVIN_W_DESC1       0x028u
#define VERSA_P_REG_MVIN_W_STATUS      0x030u
#define VERSA_P_REG_MVIN_META_DESC0    0x040u
#define VERSA_P_REG_MVIN_META_DESC1    0x048u
#define VERSA_P_REG_MVIN_META_STATUS   0x050u
#define VERSA_P_REG_GEMM_DESC0         0x060u
#define VERSA_P_REG_GEMM_DESC1         0x068u
#define VERSA_P_REG_GEMM_STATUS        0x070u
#define VERSA_P_REG_MVOUT_DESC0        0x080u
#define VERSA_P_REG_MVOUT_DESC1        0x088u
#define VERSA_P_REG_MVOUT_STATUS       0x090u
#define VERSA_P_REG_IAR                0x100u
#define VERSA_P_REG_MER                0x108u
#define VERSA_P_REG_IER                0x110u
#define VERSA_P_REG_ISR                0x118u
#define VERSA_P_REG_IPR                0x120u
#define VERSA_P_REG_GLOBAL_CLEAR       0x128u
#define VERSA_P_REG_PROFILE_CTRL       0x130u
#define VERSA_P_REG_PROFILE_GLOBAL     0x138u
#define VERSA_P_REG_PROFILE_A_BUSY     0x140u
#define VERSA_P_REG_PROFILE_W_BUSY     0x148u
#define VERSA_P_REG_PROFILE_META_BUSY  0x150u
#define VERSA_P_REG_PROFILE_GEMM_BUSY  0x158u
#define VERSA_P_REG_PROFILE_MVOUT_BUSY 0x160u
#define VERSA_P_REG_PROFILE_BUSY_ANY   0x168u
#define VERSA_P_REG_PROFILE_BUSY_MULTI 0x170u
#define VERSA_P_REG_PROFILE_AXI0_R_BEATS 0x178u
#define VERSA_P_REG_PROFILE_AXI1_R_BEATS 0x180u
#define VERSA_P_REG_PROFILE_AXI2_R_BEATS 0x188u
#define VERSA_P_REG_PROFILE_AXI3_R_BEATS 0x190u
#define VERSA_P_REG_PROFILE_AXI0_W_BEATS 0x198u
#define VERSA_P_REG_PROFILE_AXI1_W_BEATS 0x1a0u
#define VERSA_P_REG_PROFILE_AXI2_W_BEATS 0x1a8u
#define VERSA_P_REG_PROFILE_AXI3_W_BEATS 0x1b0u
#define VERSA_P_REG_PROFILE_AXI0_AR_STALL 0x1b8u
#define VERSA_P_REG_PROFILE_AXI1_AR_STALL 0x1c0u
#define VERSA_P_REG_PROFILE_AXI2_AR_STALL 0x1c8u
#define VERSA_P_REG_PROFILE_AXI3_AR_STALL 0x1d0u
#define VERSA_P_REG_PROFILE_AXI0_R_STALL 0x1d8u
#define VERSA_P_REG_PROFILE_AXI1_R_STALL 0x1e0u
#define VERSA_P_REG_PROFILE_AXI2_R_STALL 0x1e8u
#define VERSA_P_REG_PROFILE_AXI3_R_STALL 0x1f0u
#define VERSA_P_REG_PROFILE_AXI0_AW_STALL 0x1f8u
#define VERSA_P_REG_PROFILE_AXI1_AW_STALL 0x200u
#define VERSA_P_REG_PROFILE_AXI2_AW_STALL 0x208u
#define VERSA_P_REG_PROFILE_AXI3_AW_STALL 0x210u
#define VERSA_P_REG_PROFILE_AXI0_W_STALL 0x218u
#define VERSA_P_REG_PROFILE_AXI1_W_STALL 0x220u
#define VERSA_P_REG_PROFILE_AXI2_W_STALL 0x228u
#define VERSA_P_REG_PROFILE_AXI3_W_STALL 0x230u

#define VERSA_P_STATUS_BUSY_MASK       (1ull << 0)
#define VERSA_P_STATUS_DONE_MASK       (1ull << 1)
#define VERSA_P_STATUS_ERROR_MASK      (1ull << 2)
#define VERSA_P_STATUS_ERROR_SHIFT     4
#define VERSA_P_STATUS_ERROR_MASK4     0xfu
#define VERSA_P_STATUS_ACCEPTED_SHIFT  32

#define VERSA_P_START_MVIN_A_BIT       33
#define VERSA_P_MVIN_A_BANK_BIT        34
#define VERSA_P_START_MVIN_W_BIT       33
#define VERSA_P_START_MVIN_META_BIT    34
#define VERSA_P_START_GEMM_BIT         51
#define VERSA_P_GEMM_A_BANK_BIT        52
#define VERSA_P_START_MVOUT_BIT        50

typedef enum versa_p_api {
    VERSA_P_API_MVIN_A = 0,
    VERSA_P_API_MVIN_W = 1,
    VERSA_P_API_MVIN_META = 2,
    VERSA_P_API_GEMM_I8 = 3,
    VERSA_P_API_MVOUT = 4
} versa_p_api;

typedef enum versa_p_hw_error {
    VERSA_P_HW_ERR_NONE = 0,
    VERSA_P_HW_ERR_ILLEGAL_SHAPE = 1,
    VERSA_P_HW_ERR_START_WHILE_BUSY = 2,
    VERSA_P_HW_ERR_RESOURCE_CONFLICT = 3,
    VERSA_P_HW_ERR_BANK_CONFLICT = 4,
    VERSA_P_HW_ERR_BANK_NOT_VALID = 5,
    VERSA_P_HW_ERR_SHAPE_MISMATCH = 6,
    VERSA_P_HW_ERR_ALIGNMENT = 7,
    VERSA_P_HW_ERR_UNSUPPORTED_MODE = 8,
    VERSA_P_HW_ERR_INTERNAL_ASSERT = 10,
    VERSA_P_HW_ERR_ILLEGAL_FLAGS = 11
} versa_p_hw_error;

typedef enum versa_p_mvout_mode {
    VERSA_P_MVOUT_RAW_I32 = 0,
    VERSA_P_MVOUT_FP32_TENSOR_Q8_24 = 1,
    VERSA_P_MVOUT_FP32_PER_CHANNEL_Q8_24 = 2
} versa_p_mvout_mode;

typedef enum versa_p_meta_type {
    VERSA_P_META_BIAS = 0,
    VERSA_P_META_SCALE = 1
} versa_p_meta_type;

typedef struct versa_p_reg_status {
    uint8_t busy;
    uint8_t done;
    uint8_t error;
    uint8_t error_code;
    uint32_t accepted_count;
    uint64_t raw;
} versa_p_reg_status;

static inline versa_p_reg_status versa_p_decode_status(uint64_t raw)
{
    versa_p_reg_status status;
    status.busy = (uint8_t)((raw & VERSA_P_STATUS_BUSY_MASK) != 0);
    status.done = (uint8_t)((raw & VERSA_P_STATUS_DONE_MASK) != 0);
    status.error = (uint8_t)((raw & VERSA_P_STATUS_ERROR_MASK) != 0);
    status.error_code = (uint8_t)((raw >> VERSA_P_STATUS_ERROR_SHIFT) &
                                  VERSA_P_STATUS_ERROR_MASK4);
    status.accepted_count = (uint32_t)(raw >> VERSA_P_STATUS_ACCEPTED_SHIFT);
    status.raw = raw;
    return status;
}

#ifdef __cplusplus
}
#endif

#endif
