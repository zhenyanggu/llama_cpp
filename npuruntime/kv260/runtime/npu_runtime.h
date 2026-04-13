#ifndef NPU_RUNTIME_H
#define NPU_RUNTIME_H

#include <stdint.h>
#include <stddef.h>
#include <string>
#include <vector>

// ==========================================
// 寄存器与硬件定义 (来自 RDL 自动生成)
// ==========================================
#include "npu_regs_compat.h"

// ==========================================
// 配置结构体定义
// ==========================================

struct MvinConfig {
    void* host_ptr;        // 虚拟地址指针
    uint32_t sram_addr;    
    uint32_t col_num;
    uint32_t row_num;
    uint16_t sram_stride;//row num=0是随便配置
    uint32_t dram_stride;//同上
    uint8_t  precision;    // 2-bit: 数据精度，01是int8，int32也是int8，所以全部配1
    uint8_t  input_type;   // 2-bit: 输入类型，00=IFM, 01=WEIGHT（IFM还是weight不区分）都是int8输入 | 10=BIAS int32 到acc
    bool     dest;         // 1-bit: 0=SPM, 1=ACC
    bool     is_bias;      // 1-bit: 0=No, 1=Bias（mvin到acc的bias寄存器时为1，一次32个数据）
    bool     is_quant;     // 1-bit: 是否量化，是则输入int8后量化到int32存到acc
    uint32_t quant_zero;
    uint16_t quant_scale;
    uint16_t quant_shift;
};

struct MvoutConfig {
    void* host_ptr;      // 虚拟地址指针
    uint32_t sram_addr;
    uint32_t col_num;
    uint32_t row_num;
    uint16_t sram_stride;
    uint32_t dram_stride;
    uint8_t  precision;    // 2-bit: 数据精度：仍然全1
    uint8_t  output_type;  // 2-bit: 输出类型：00代表int8（从spm），01代表int32（从acc）
    bool     source;       // 1-bit: 0=SPM, 1=ACC
    bool     is_quant;     // 1-bit: 是否量化（没有作用，全0）
    uint32_t quant_zero;
    uint16_t quant_scale;
    uint16_t quant_shift;
};

struct SfuConfig {
    uint8_t  op_type;            // SFU Opcode
    uint8_t  int_type;           // Data Type (0=int8, 1=int16, etc.)
    bool     is_quant;           // Is Quantized

    // Input/Output Config
    uint32_t input_sram_addr;
    uint16_t input_col_num;      // Width - 1
    uint16_t input_row_num;      // Height - 1
    uint32_t output_sram_addr;

    // Quant Parameters
    uint32_t input_zeropoint;
    uint16_t output_zeropoint;
    uint16_t input_scale;
    uint16_t input_scale_shift;
    uint16_t output_scale;
    uint16_t output_scale_shift;
};

struct ConvConfig {
    // Padding
    uint8_t pad_top;
    uint8_t pad_bottom;
    uint8_t pad_left;
    uint8_t pad_right;
    uint8_t pad_mode; // 0=Zero

    // Weight
    uint8_t weight_shape_m1;    // 4-bit: kernel size - 1 (e.g., 2 for 3x3)
    uint8_t weight_stride_m1;   // 2-bit: stride - 1
    uint8_t weight_dilation_m1; // 5-bit: dilation - 1
    bool    is_group_conv;      // 1-bit: 是否分组卷积

    // Compute
    uint8_t int_type;           // 2-bit: 0=int8, 1=int16, ...
    uint8_t op_type;            // 2-bit: 0=GEMM, 1=Conv, 2=GEMV
    bool    dataflow_mode;      // 1-bit: 0=im2col & OS, 1=OS only
    bool    accout_dest;        // 1-bit: 0=SPM, 1=ACC

    // Zeropoints
    uint16_t input_a_zeropoint;
    uint16_t input_b_zeropoint;

    // Input A (IFM)
    uint32_t input_a_addr;
    uint16_t input_a_col_num_m1;
    uint8_t input_a_row_num_m1;
    uint16_t input_a_stride;

    // Input B (Weights)
    uint32_t input_b_addr;
    uint8_t input_b_col_num_m1;
    uint16_t input_b_row_num_m1;
    uint16_t input_b_stride;

    // Accumulator
    uint8_t biaspsum_width;
    uint8_t biaspsum_height;
    uint32_t biaspsum_addr;
    uint16_t biaspsum_stride;

    // Output
    uint32_t output_addr;
    uint16_t output_stride;
    bool    is_accumulate;      // 1-bit: 0=No accumulate, 1=Accumulate with previous psum
    bool    relu_enable;        // 1-bit: 0=Disabled, 1=Enabled
    uint8_t relu_type;          // 3-bit: 0=relu, 1=relu6, 2=leaky(0.1), 3=leaky(0.2), 4=leaky(0.01)
    bool    is_bias;            // 1-bit: 0=accumulate psum, 1=accumulate bias

    // Quantization
    uint32_t output_zeropoint;
    uint16_t quant_scale;
    uint16_t quant_scaleshift;
};

// ==========================================
// Micro-Tiling Conv Tile API
// ==========================================
// 本函数只负责单个 cout block 的 micro-tile 计算（j_h × j_w × j_cin 三层循环）。
// IFM / Weight / Bias 的 MVIN，OFM 的 MVOUT，以及 j_cout 循环均由调用者完成。
// Padding 由编译器在 IFM 数据上预处理完毕，硬件侧 padding 全部为 0。
// 对称量化：所有 zeropoint 均为 0，不在此结构体中暴露。
//
// Layout:
//   IFM/OFM : NCHWC32 — [C/32][H][W][32]
//   Weight  : blocked — (Cout/32)*(Cin/32)*Kh*Kw*32*32
//   c_in / t_cin / t_cout 必须是 32 的整数倍（需提前 pad）
struct NpuConvTileConfig {
    // ---- SPM / ACC 基地址（由调用者预先 MVIN 完毕） ----
    uint32_t sram_addr_ifm;     // IFM 在 SPM 中的基地址，布局 [cin_blk][h][w][32]
    uint32_t sram_addr_weight;  // Weight 在 SPM 中的基地址，布局 [cin_blk]*Kh*Kw*32*32（单个 cout block）
    uint32_t sram_addr_ofm;     // OFM 在 SPM 中的基地址（最后一轮 cin 时写出到此处）
    uint32_t acc_addr_psum;     // Partial sum 在 ACC 中的基地址（中间 cin 轮次累加用）
                                // Bias 由调用者 MVIN 到 ACC bias 寄存器，硬件 is_bias=1 时自动读取

    // ---- 全局张量维度 ----
    int32_t c_in;               // 完整 IFM 输入通道数（padded to 32），用于判断 is_last_cin_global

    // ---- 卷积参数 ----
    int32_t k_h;                // 卷积核高度，写入 weight_shape_m1 并计算 IFM 感受野
    int32_t k_w;                // 卷积核宽度，计算 IFM 感受野和 weight_block_size
    int32_t stride;             // 卷积步长，写入 weight_stride_m1 并计算 IFM 感受野
    int32_t dilation;           // 膨胀系数，写入 weight_dilation_m1 并计算 IFM 感受野

    // ---- 宏 tile 索引与尺寸 ----
    int32_t i_cin;              // 当前 tile 在 cin 维度的全局起始位置（必须是 32 的倍数）
                                // 用于判断 is_first/last_cin_global → 控制 bias 加法 / psum 累加 / relu / 输出去向
    int32_t t_cout;             // 当前 tile 的 cout 大小（仅用于校验，通常等于 SA_SIZE=32）
    int32_t t_h_out;            // 当前 tile 的输出高度，作为 j_h 循环的上界
    int32_t t_w_out;            // 当前 tile 的输出宽度，作为 j_w 循环的上界及 stride/offset 计算
    int32_t t_cin;              // 当前 tile 的 cin 大小，作为 j_cin 循环的上界

    // ---- 量化 / 激活（仅在最后一轮 cin 输出到 SPM 时生效） ----
    uint16_t quant_scale;       // ACC→SPM 反量化 scale
    uint16_t quant_scaleshift;  // ACC→SPM 反量化 shift
    bool     relu_enable;       // 是否启用 ReLU
    uint8_t  relu_type;         // ReLU 类型: 0=relu, 1=relu6, 2=leaky(0.1), 3=leaky(0.2), 4=leaky(0.01)
    bool     bias_enable;       // 是否启用 bias（仅在首轮 cin 生效）

    // ---- 分组卷积（预留） ----
    bool is_group_conv;         // 是否分组卷积，直接传给 ConvConfig
};

struct GemmConfig {
    // config_compute
    bool     dataflow;            // 1-bit: 0=im2col & OS, 1=OS only，只支持os，填1（卷积时填0）
    uint8_t  int_type;            // 2-bit: 0=int8, 1=int16, ...全部填00，代表int8
    uint8_t  optype;              // 2-bit: 0=GEMM, 1=Conv, （2=GEMV）
    bool     accout_dest;         // 1-bit: 0=SPM（量化到spm 变成int8）, 1=ACC（int32存回acc）
    bool     asymmetric_activations; // 1-bit: GEMM only, treat input A as uint8 and subtract fixed 128 before SA
    uint16_t input_a_zeropoint;//对称量化全0
    uint16_t input_b_zeropoint;

    uint32_t output_zeropoint;    // ACC to SPM zeropoint
    uint16_t output_scale;        // ACC to SPM scale，只需要配置out的scale，=scale_a*scaleb/scale_out
    uint16_t output_scaleshift;   // ACC to SPM scaleshift
    
    // config_accumulate
    uint32_t biaspsum_addr;         //acc中的部分和地址，如果是加bias请使用bias寄存器
    uint16_t biaspsum_stride;       
    uint8_t  biaspsum_width;        //不累加也要配置，也就是输出矩阵的大小，不需要减1
    uint8_t  biaspsum_height;       //不累加也要配置

    uint32_t output_addr;         // Output matrix address (To ACC/SPM)，先不要原位写
    uint16_t output_stride;       // Output matrix stride  (To ACC/SPM)
    bool     isaccu;              // 1-bit: 0=No accumulate, 1=Accumulate with previous psum，
    bool     relu;                // 1-bit: 0=Disabled, 1=Enabled
    uint8_t  relu_type;           // 3-bit: 0=relu, 1=relu6, 2=leaky(0.1), 3=leaky(0.2), 4=leaky(0.01)
    bool     is_bias;             // 1-bit: 0=accumulate psum, 1=accumulate bias，bias现在都在acc中的bias寄存器里，如果这位是1则加寄存器中的值

    // compute_sa
    uint32_t input_a_addr;
    uint16_t input_a_col_num;
    uint8_t  input_a_row_num;
    uint16_t input_a_stride;

    uint32_t input_b_addr;
    uint8_t  input_b_col_num;
    uint16_t input_b_row_num;
    uint16_t input_b_stride;
};

struct MataddConfig {
    // Input/Output addresses in ACC/SPM
    uint32_t input_a_addr;     // Matrix A address in ACC
    uint32_t input_b_addr;     // Matrix B address in ACC
    uint32_t output_addr;      // Output address in SPM
    
    // Dimensions
    uint8_t  col_num_m1;       // Column number - 1
    uint8_t  row_num_m1;       // Row number - 1
    
    // Output Quantization (int32 -> int8)
    uint32_t output_zeropoint;
    uint16_t output_scale;
    uint16_t output_scaleshift;
};

struct TransposeConfig {
    // Transpose: 通过 SFU 实现矩阵转置
    uint32_t input_sram_addr;   // 输入矩阵在 SPM 中的地址
    uint32_t output_sram_addr;  // 输出矩阵在 SPM 中的地址
    uint16_t col_num;           // 列数 (Width - 1)
    uint16_t row_num;           // 行数 (Height - 1)
    bool     out_padding_row;   // 输出行方向是否补零
    bool     out_padding_col;   // 输出列方向是否补零
};

struct ResampleConfig {
    // Resample: 2x 下采样/上采样操作 (通过 SFU 实现)
    uint8_t  resample_type;     // 0=downsample, 1=upsample, 2=pooling
    uint8_t  resample_op;       // 0=max/nearest, 1=avg/bilinear
    uint32_t input_sram_addr;   // 输入数据在 SPM 中的地址
    uint32_t output_sram_addr;  // 输出数据在 SPM 中的地址
    uint16_t input_col_num;     // 输入列数 (Width - 1)
    uint16_t input_row_num;     // 输入行数 (Height - 1)
};

struct LayoutConvertConfig {
    uint32_t sram_addr;      // 输入地址 (SPM)
    uint32_t output_addr;    // 输出地址 (SPM)
    uint16_t n;              // batch size
    uint16_t c;              // channel 数
    uint16_t h;              // height
    uint16_t w;              // width
};

// ==========================================
// NpuRuntime 类定义
// ==========================================

class NpuRuntime {
public:
    NpuRuntime();
    ~NpuRuntime();

    // 初始化：打开驱动，进行 mmap
    bool init();

    // 复位：软件控制硬件复位 + 清除影子寄存器
    void reset(); // <--- [新增]

    // 内存管理基础接口
    void* get_memory_base(); 
    uint32_t get_memory_size();

    // 执行指令接口
    void run_mvin(const MvinConfig& cfg);
    void run_mvout(const MvoutConfig& cfg);
    void run_sfu(const SfuConfig& cfg);
    void run_conv(const ConvConfig& cfg);
    void run_gemm(const GemmConfig& cfg);
    void run_matadd(const MataddConfig& cfg);
    void run_transpose(const TransposeConfig& cfg);
    void run_resample(const ResampleConfig& cfg);
    void run_nchw_to_nchwc32(const LayoutConvertConfig& cfg);
    void run_nchwc32_to_nchw(const LayoutConvertConfig& cfg);
    int run_conv_tile(const NpuConvTileConfig& cfg);

    // --- Memory Allocator (Heap) ---
    void* alloc(size_t size);
    void free(void* ptr);

private:
    int fd;
    void* regs_virt_base;   // 寄存器空间的虚拟基地址
    void* data_virt_base;   // CMA 数据空间的虚拟基地址
    uint32_t data_phy_base; // CMA 数据空间的 DMA/物理基地址
    uint32_t data_map_size; // CMA 数据空间的实际映射大小

    // Allocator State
    static const size_t ALIGNMENT = 64; 
    
    struct alignas(64) BlockHeader {
        size_t size;       
        bool is_free;
        BlockHeader* next;
        BlockHeader* prev;
    };

    BlockHeader* free_list_head;
    void init_allocator();
    void coalesce(BlockHeader* block);

    // 辅助函数：虚拟地址转物理地址
    uint32_t virt_to_phys(void* ptr);
    
    // 基础寄存器读写
    void reg_write(uint32_t offset, uint32_t val);
    void reg_write64(uint32_t offset, uint64_t val);
    uint32_t reg_read(uint32_t offset);
    
    // ----------------------------------------------------
    // 【优化】影子寄存器逻辑
    // ----------------------------------------------------
    
    // 仅缓存配置类寄存器 (CFG, QUANT)
    struct ShadowRegs {
        uint64_t mvin_cfg;
        uint64_t mvin_quant;
        uint64_t mvout_cfg;
        uint64_t mvout_quant;
        uint64_t sfu_cfg1;
        uint64_t sfu_cfg2;
        uint64_t compute_cfg1;
        uint64_t compute_cfg2;
        uint64_t accu_cfg1;
        uint64_t accu_cfg2;
    } shadow;

    // 初始化/重置影子寄存器
    void reset_shadows();
    
    // 带缓存检查的写操作：只有值变化时才写硬件
    void reg_write64_cached(uint32_t offset, uint64_t val, uint64_t* cache_ptr);

    // 混合轮询+中断等待
    void wait_irq();
    bool check_irq_pending();  // 检查中断是否挂起，返回挂起的中断位
    void ack_irq();  // 清除/应答全部中断位
    void dump_irq_regs(); // 轮询超时后打印中断相关寄存器
};

// ==========================================
// C Interface (API)
// ==========================================

extern "C" {
    // ---------------------------------------------------------------------
    // Lifecycle
    // ---------------------------------------------------------------------

    // 初始化 Runtime，打开驱动并完成寄存器/DDR 映射。
    // 返回值：0=成功，-1=失败。
    int npu_init();

    // 释放 Runtime 资源（mmap / fd / allocator）。
    void npu_destroy();

    // 触发硬件复位，同时清空 Runtime 影子寄存器缓存。
    void npu_reset(); // <--- [新增]

    // ---------------------------------------------------------------------
    // Profiling
    // ---------------------------------------------------------------------

    // 标记逻辑层开始/结束，并在需要时手动落盘 profiling 报告。
    struct npu_profile_runtime_summary {
        uint64_t layer_count;
        uint64_t layer_invocations;
        uint64_t total_ns;
        uint64_t dma_in_ns;
        uint64_t compute_ns;
        uint64_t dma_out_ns;
        uint64_t layout_ns;
        uint64_t wait_irq_ns;
        uint64_t mvin_calls;
        uint64_t compute_calls;
        uint64_t mvout_calls;
        uint64_t layout_calls;
    };

    void npu_profile_begin(int64_t layer_id);
    void npu_profile_end(int64_t layer_id);
    void npu_profile_dump(const char* path);
    void npu_profile_reset_summary();
    void npu_profile_get_summary(struct npu_profile_runtime_summary * out);
    
    // ---------------------------------------------------------------------
    // Memory
    // ---------------------------------------------------------------------

    // 从 Runtime 管理的 DDR 映射区申请内存。
    // size: 申请字节数，建议按 64B 对齐（内部最小对齐 64B）。
    // 返回：可直接传给 DMA API 的虚拟地址；失败返回 nullptr。
    void* npu_mem_alloc(size_t size);

    // 释放 npu_mem_alloc() 申请的内存。
    // ptr: 必须是 npu_mem_alloc 返回值；传 nullptr 时无操作。
    void npu_mem_free(void* ptr);

    // ---------------------------------------------------------------------
    // DMA Operations
    // ---------------------------------------------------------------------

    /**
     * @brief DRAM -> SPM/ACC（MVIN）
     *
     * @param host_ptr        DDR 映射区内虚拟地址（32-bit 物理地址会在内部换算）。
     * @param sram_addr       目标 SPM/ACC 地址，硬件位宽 32-bit。
     * @param col_num         传输列数，硬件位宽 32-bit（按硬件协议填写，当前实现直接透传）。
     * @param row_num         传输行数，硬件位宽 32-bit（按硬件协议填写，当前实现直接透传）。
     * @param sram_stride     SPM 步长，硬件位宽 16-bit。
     * @param dram_stride     DRAM 步长，硬件位宽 32-bit。
     * @param precision       输入精度，寄存器位宽 2-bit；当前 Runtime 固定写 1（该入参暂不生效）。
     * @param input_type      输入类型，寄存器位宽 2-bit：0=IFM, 1=WEIGHT, 2=BIAS。
     * @param dest            目的地，寄存器位宽 1-bit：0=SPM, 1=ACC。
     * @param is_bias         是否走 bias 路径，寄存器位宽 1-bit：0/1。
     * @param is_quant        是否量化，寄存器位宽 1-bit：0/1。
     * @param quant_zero      量化零点，寄存器位宽 32-bit。
     * @param quant_scale     量化 scale，寄存器位宽 16-bit。
     * @param quant_shift     量化 shift，寄存器位宽 16-bit。
     */
    void npu_dma_mvin(
        void* host_ptr,
        uint32_t sram_addr,
        uint32_t col_num,
        uint32_t row_num,
        uint16_t sram_stride,
        uint32_t dram_stride,
        uint8_t  precision,    // 2-bit
        uint8_t  input_type,   // 2-bit
        bool     dest,         // 1-bit: 0=SPM, 1=ACC
        bool     is_bias,      // 1-bit
        bool     is_quant,     // 1-bit
        uint32_t quant_zero,
        uint16_t quant_scale,
        uint16_t quant_shift
    );

    /**
     * @brief SPM/ACC -> DRAM（MVOUT）
     *
     * @param host_ptr        DDR 映射区内虚拟地址（32-bit 物理地址会在内部换算）。
     * @param sram_addr       源 SPM/ACC 地址，硬件位宽 32-bit。
     * @param col_num         传输列数，硬件位宽 32-bit（按硬件协议填写，当前实现直接透传）。
     * @param row_num         传输行数，硬件位宽 32-bit（按硬件协议填写，当前实现直接透传）。
     * @param sram_stride     SPM 步长，硬件位宽 16-bit。
     * @param dram_stride     DRAM 步长，硬件位宽 32-bit。
     * @param precision       输出精度，寄存器位宽 2-bit；当前 Runtime 固定写 1（该入参暂不生效）。
     * @param output_type     输出类型，寄存器位宽 2-bit：0=int8(SPM), 1=int32(ACC)。
     * @param source          数据源，寄存器位宽 1-bit：0=SPM, 1=ACC。
     * @param is_quant        是否量化，寄存器位宽 1-bit：0/1。
     * @param quant_zero      量化零点，寄存器位宽 32-bit。
     * @param quant_scale     量化 scale，寄存器位宽 16-bit。
     * @param quant_shift     量化 shift，寄存器位宽 16-bit。
     */
    void npu_dma_mvout(
        void* host_ptr,
        uint32_t sram_addr,
        uint32_t col_num,
        uint32_t row_num,
        uint16_t sram_stride,
        uint32_t dram_stride,
        uint8_t  precision,    // 2-bit
        uint8_t  output_type,  // 2-bit
        bool     source,       // 1-bit: 0=SPM, 1=ACC
        bool     is_quant,     // 1-bit
        uint32_t quant_zero,
        uint16_t quant_scale,
        uint16_t quant_shift
    );

    // ---------------------------------------------------------------------
    // SFU Operations
    // ---------------------------------------------------------------------

    /**
     * @brief 通用 SFU 计算（softmax/gelu/layernorm/transpose/resample 等）
     *
     * @param op_type             操作码，硬件位宽 6-bit。
     * @param int_type            数据类型，硬件位宽 2-bit（常用 0=int8）。
     * @param is_quant            是否量化，硬件位宽 1-bit。
     * @param input_sram_addr     输入地址，硬件位宽 32-bit。
     * @param input_col_num       输入列数，硬件位宽 16-bit（多数场景填写 width-1）。
     * @param input_row_num       输入行数，硬件位宽 16-bit（多数场景填写 height-1）。
     * @param output_sram_addr    输出地址，硬件位宽 32-bit。
     * @param input_zeropoint     输入零点，硬件位宽 32-bit。
     * @param output_zeropoint    输出零点，硬件位宽 16-bit。
     * @param input_scale         输入 scale，硬件位宽 16-bit。
     * @param input_scale_shift   输入 shift，硬件位宽 16-bit。
     * @param output_scale        输出 scale，硬件位宽 16-bit。
     * @param output_scale_shift  输出 shift，硬件位宽 16-bit。
     */
    void npu_sfu_run(
        uint8_t  op_type,
        uint8_t  int_type,
        bool     is_quant,
        uint32_t input_sram_addr,
        uint16_t input_col_num,
        uint16_t input_row_num,
        uint32_t output_sram_addr,
        uint32_t input_zeropoint,
        uint16_t output_zeropoint,
        uint16_t input_scale,
        uint16_t input_scale_shift,
        uint16_t output_scale,
        uint16_t output_scale_shift
    );

    // ---------------------------------------------------------------------
    // Convolution (SA)
    // ---------------------------------------------------------------------

    /**
     * @brief 卷积/SA 执行接口（直接对应硬件寄存器）
     *
     * Padding / Weight / Compute
     * @param pad_top             上 padding，硬件位宽 2-bit，范围 0~3。
     * @param pad_bottom          下 padding，硬件位宽 2-bit，范围 0~3。
     * @param pad_left            左 padding，硬件位宽 2-bit，范围 0~3。
     * @param pad_right           右 padding，硬件位宽 2-bit，范围 0~3。
     * @param pad_mode            padding 模式，硬件位宽 2-bit（常用 0=zero）。
     * @param weight_shape_m1     卷积核尺寸-1，硬件位宽 4-bit（如 3x3 填 2）。
     * @param weight_stride_m1    步长-1，硬件位宽 2-bit（如 stride=1 填 0）。
     * @param weight_dilation_m1  膨胀-1，硬件位宽 5-bit（如 dilation=1 填 0）。
     * @param is_group_conv       分组卷积使能，硬件位宽 1-bit。
     * @param int_type            数据类型，硬件位宽 2-bit（常用 0=int8）。
     * @param op_type             操作类型，硬件位宽 2-bit（0=GEMM,1=Conv,2=GEMV）。
     * @param dataflow_mode       数据流，硬件位宽 1-bit（0=im2col+OS,1=OS）。
     * @param accout_dest         输出去向，硬件位宽 1-bit（0=SPM,1=ACC）。
     * @param input_a_zeropoint   输入 A 零点，硬件位宽 16-bit。
     * @param input_b_zeropoint   输入 B 零点，硬件位宽 16-bit。
     *
     * SA 输入
     * @param input_a_addr        输入 A 地址，硬件位宽 32-bit。
     * @param input_a_col_num_m1  输入 A 列数-1，硬件位宽 11-bit。
     * @param input_a_row_num_m1  输入 A 行数-1，硬件位宽 5-bit。
     * @param input_a_stride      输入 A 步长，硬件位宽 16-bit。
     * @param input_b_addr        输入 B 地址，硬件位宽 32-bit。
     * @param input_b_col_num_m1  输入 B 列数-1，硬件位宽 5-bit。
     * @param input_b_row_num_m1  输入 B 行数-1，硬件位宽 11-bit。
     * @param input_b_stride      输入 B 步长，硬件位宽 16-bit。
     *
     * Accumulator / 输出
     * @param biaspsum_width      输出宽度，硬件位宽 8-bit（建议填写实际值，不减 1）。
     * @param biaspsum_height     输出高度，硬件位宽 8-bit（建议填写实际值，不减 1）。
     * @param biaspsum_addr       bias/psum 地址，硬件位宽 32-bit。
     * @param biaspsum_stride     bias/psum 步长，硬件位宽 16-bit。
     * @param output_addr         输出地址，硬件位宽 32-bit。
     * @param output_stride       输出步长，硬件位宽 16-bit。
     * @param is_accumulate       是否累加 psum，硬件位宽 1-bit。
     * @param relu_enable         是否启用激活，硬件位宽 1-bit。
     * @param relu_type           激活类型，硬件位宽 3-bit（0=relu,1=relu6,2/3/4=leaky）。
     * @param is_bias             是否累加 bias 寄存器，硬件位宽 1-bit。
     * @param output_zeropoint    输出零点，硬件位宽 32-bit。
     * @param quant_scale         输出量化 scale，硬件位宽 16-bit。
     * @param quant_scaleshift    输出量化 shift，硬件位宽 16-bit。
     */
    void npu_conv_run(
        uint8_t pad_top,           // 2-bit
        uint8_t pad_bottom,        // 2-bit
        uint8_t pad_left,          // 2-bit
        uint8_t pad_right,         // 2-bit
        uint8_t pad_mode,          // 2-bit
        uint8_t weight_shape_m1,   // 4-bit
        uint8_t weight_stride_m1,  // 2-bit
        uint8_t weight_dilation_m1,// 5-bit
        bool    is_group_conv,     // 1-bit
        uint8_t int_type,          // 2-bit
        uint8_t op_type,           // 2-bit
        bool    dataflow_mode,     // 1-bit
        bool    accout_dest,       // 1-bit
        uint16_t input_a_zeropoint,
        uint16_t input_b_zeropoint,
        uint32_t input_a_addr,
        uint16_t input_a_col_num_m1,
        uint8_t input_a_row_num_m1,
        uint16_t input_a_stride,
        uint32_t input_b_addr,
        uint8_t input_b_col_num_m1,
        uint16_t input_b_row_num_m1,
        uint16_t input_b_stride,
        uint8_t biaspsum_width,
        uint8_t biaspsum_height,
        uint32_t biaspsum_addr,
        uint16_t biaspsum_stride,
        uint32_t output_addr,
        uint16_t output_stride,
        bool    is_accumulate,     // 1-bit
        bool    relu_enable,       // 1-bit
        uint8_t relu_type,         // 3-bit
        bool    is_bias,           // 1-bit
        uint32_t output_zeropoint,
        uint16_t quant_scale,
        uint16_t quant_scaleshift
    );


    // ---------------------------------------------------------------------
    // Micro-tiling conv tile
    // ---------------------------------------------------------------------

    /**
     * @brief 单个 cout block 的 micro-tile 卷积内核
     *
     * 该接口主要是软件调度参数（`int32_t` 维度为软件位宽，不等于寄存器位宽），
     * 内部会拆分并调用 `npu_conv_run()`。
     *
     * @param sram_addr_ifm      IFM 基地址（SPM, 32-bit）。
     * @param sram_addr_weight   Weight 基地址（SPM, 32-bit）。
     * @param sram_addr_ofm      OFM 基地址（SPM, 32-bit）。
     * @param acc_addr_psum      中间 psum 基地址（ACC, 32-bit）。
     * @param c_in               全局输入通道（建议 pad 到 32 的倍数）。
     * @param k_h                卷积核高（建议 >=1）。
     * @param k_w                卷积核宽（建议 >=1）。
     * @param stride             步长（建议 >=1）。
     * @param dilation           膨胀（建议 >=1）。
     * @param i_cin              当前 cin 起始索引（建议 32 对齐）。
     * @param t_cout             当前 tile cout（硬件 SA_SIZE=32，建议 <=32 且 32 对齐）。
     * @param t_h_out            当前 tile 输出高（建议 >0）。
     * @param t_w_out            当前 tile 输出宽（建议 >0）。
     * @param t_cin              当前 tile cin（建议 32 对齐）。
     * @param quant_scale        最终输出量化 scale（16-bit）。
     * @param quant_scaleshift   最终输出量化 shift（16-bit）。
     * @param relu_enable        是否启用 ReLU（1-bit）。
     * @param relu_type          ReLU 类型（3-bit，定义同 `npu_conv_run`）。
     * @param bias_enable        是否启用 bias（1-bit，仅首轮 cin 有效）。
     * @param is_group_conv      是否分组卷积（1-bit）。
     */
    void npu_conv_tile_run(
        uint32_t sram_addr_ifm,
        uint32_t sram_addr_weight,
        uint32_t sram_addr_ofm,
        uint32_t acc_addr_psum,
        int32_t  c_in,
        int32_t  k_h,
        int32_t  k_w,
        int32_t  stride,
        int32_t  dilation,
        int32_t  i_cin,
        int32_t  t_cout,
        int32_t  t_h_out,
        int32_t  t_w_out,
        int32_t  t_cin,
        uint16_t quant_scale,
        uint16_t quant_scaleshift,
        bool     relu_enable,
        uint8_t  relu_type,
        bool     bias_enable,
        bool     is_group_conv
    );

    /**
     * @brief GEMM 接口（矩阵乘）
     *
     * @param dataflow           数据流，硬件位宽 1-bit（0=im2col+OS,1=OS）。
     * @param int_type           数据类型，硬件位宽 2-bit（常用 0=int8）。
     * @param optype             操作类型，硬件位宽 2-bit（通常 GEMM 填 0）。
     * @param accout_dest        输出去向，硬件位宽 1-bit（0=SPM,1=ACC）。
     * @param input_a_zeropoint  输入 A 零点，硬件位宽 16-bit。
     * @param input_b_zeropoint  输入 B 零点，硬件位宽 16-bit。
     * @param output_zeropoint   输出零点，硬件位宽 32-bit。
     * @param output_scale       输出量化 scale，硬件位宽 16-bit。
     * @param output_scaleshift  输出量化 shift，硬件位宽 16-bit。
     * @param biaspsum_addr      bias/psum 地址，硬件位宽 32-bit。
     * @param biaspsum_stride    bias/psum 步长，硬件位宽 16-bit。
     * @param biaspsum_width     输出宽度，硬件位宽 8-bit（建议实际值）。
     * @param biaspsum_height    输出高度，硬件位宽 8-bit（建议实际值）。
     * @param output_addr        输出地址，硬件位宽 32-bit。
     * @param output_stride      输出步长，硬件位宽 16-bit。
     * @param isaccu             是否累加，硬件位宽 1-bit。
     * @param relu               是否启用 ReLU，硬件位宽 1-bit。
     * @param relu_type          ReLU 类型，硬件位宽 3-bit。
     * @param is_bias            是否加 bias，硬件位宽 1-bit。
     * @param input_a_addr       输入 A 地址，硬件位宽 32-bit。
     * @param input_a_col_num    输入 A 列数，硬件位宽 11-bit（当前实现透传）。
     * @param input_a_row_num    输入 A 行数，硬件位宽 5-bit（当前实现透传）。
     * @param input_a_stride     输入 A 步长，硬件位宽 16-bit。
     * @param input_b_addr       输入 B 地址，硬件位宽 32-bit。
     * @param input_b_col_num    输入 B 列数，硬件位宽 5-bit（当前实现透传）。
     * @param input_b_row_num    输入 B 行数，硬件位宽 11-bit（当前实现透传）。
     * @param input_b_stride     输入 B 步长，硬件位宽 16-bit。
     * @param asymmetric_activations
     *                           GEMM 专用开关；true 时将输入 A 视作 uint8 编码并在进入 SA 前固定减 128。
     */
    void npu_gemm_run(
        bool     dataflow,         // 1-bit: 0=im2col & OS, 1=OS only
        uint8_t  int_type,         // 2-bit
        uint8_t  optype,           // 2-bit
        bool     accout_dest,      // 1-bit: 0=SPM, 1=ACC
        uint16_t input_a_zeropoint,
        uint16_t input_b_zeropoint,
        uint32_t output_zeropoint,
        uint16_t output_scale,
        uint16_t output_scaleshift,
        uint32_t biaspsum_addr,
        uint16_t biaspsum_stride,
        uint8_t  biaspsum_width,
        uint8_t  biaspsum_height,
        uint32_t output_addr,
        uint16_t output_stride,
        bool     isaccu,           // 1-bit
        bool     relu,             // 1-bit
        uint8_t  relu_type,        // 3-bit
        bool     is_bias,          // 1-bit
        uint32_t input_a_addr,
        uint16_t input_a_col_num,
        uint8_t  input_a_row_num,
        uint16_t input_a_stride,
        uint32_t input_b_addr,
        uint8_t  input_b_col_num,
        uint16_t input_b_row_num,
        uint16_t input_b_stride,
        bool     asymmetric_activations = false
    );

    // ---------------------------------------------------------------------
    // MATADD Operations
    // ---------------------------------------------------------------------

    /**
     * @brief 两个矩阵相加（通常在 ACC）并量化输出到 SPM
     *
     * @param input_a_addr       输入 A 地址，硬件位宽 32-bit。
     * @param input_b_addr       输入 B 地址，硬件位宽 32-bit。
     * @param output_addr        输出地址，硬件位宽 32-bit。
     * @param col_num            列数-1，硬件位宽 8-bit（实际列数范围 1~256）。
     * @param row_num            行数-1，硬件位宽 8-bit（实际行数范围 1~256）。
     * @param output_zeropoint   输出零点，硬件位宽 32-bit。
     * @param output_scale       输出量化 scale，硬件位宽 16-bit。
     * @param output_scaleshift  输出量化 shift，硬件位宽 16-bit。
     */
    void npu_matadd_run(
        uint32_t input_a_addr,
        uint32_t input_b_addr,
        uint32_t output_addr,
        uint8_t  col_num_m1,
        uint8_t  row_num_m1,
        uint32_t output_zeropoint,
        uint16_t output_scale,
        uint16_t output_scaleshift
    );

    // ---------------------------------------------------------------------
    // Transpose Operation (via SFU)
    // ---------------------------------------------------------------------

    /**
     * @brief 矩阵转置
     *
     * @param input_sram_addr    输入地址，硬件位宽 32-bit。
     * @param output_sram_addr   输出地址，硬件位宽 32-bit。
     * @param col_num            输入列数-1，硬件位宽 16-bit。
     * @param row_num            输入行数-1，硬件位宽 16-bit。
     * @param out_padding_row    输出行方向补零，硬件位宽 1-bit（0/1）。
     * @param out_padding_col    输出列方向补零，硬件位宽 1-bit（0/1）。
     */
    void npu_transpose_run(
        uint32_t input_sram_addr,   // 输入矩阵在 SPM 中的地址
        uint32_t output_sram_addr,  // 输出矩阵在 SPM 中的地址
        uint16_t col_num,           // 列数 (Width - 1)
        uint16_t row_num,           // 行数 (Height - 1)
        bool     out_padding_row,   // 输出行方向是否补零
        bool     out_padding_col    // 输出列方向是否补零
    );

    // ---------------------------------------------------------------------
    // Resample Operation (via SFU)
    // ---------------------------------------------------------------------

    /**
     * @brief 2x 重采样（下采样/上采样/池化）
     *
     * @param resample_type      采样类型：0=downsample, 1=upsample, 2=pooling。
     *                           软件入参位宽 8-bit，内部映射到 SFU OP(6-bit)。
     * @param resample_op        操作：0=max/nearest, 1=avg/bilinear（上采样双线性暂不支持）。
     *                           软件入参位宽 8-bit。
     * @param input_sram_addr    输入地址，硬件位宽 32-bit。
     * @param output_sram_addr   输出地址，硬件位宽 32-bit。
     * @param input_col_num      输入列数-1，硬件位宽 16-bit。
     * @param input_row_num      输入行数-1，硬件位宽 16-bit。
     */
    void npu_resample_run(
        uint8_t  resample_type,     // 采样类型
        uint8_t  resample_op,       // 采样操作
        uint32_t input_sram_addr,   // 输入数据在 SPM 中的地址
        uint32_t output_sram_addr,  // 输出数据在 SPM 中的地址
        uint16_t input_col_num,     // 输入列数 (Width - 1)
        uint16_t input_row_num      // 输入行数 (Height - 1)
    );

    // ---------------------------------------------------------------------
    // Layout Convert (NCHW <-> NCHWC32 / NHWC)
    // ---------------------------------------------------------------------

    /**
     * @brief NCHW -> NCHWC32
     * @param sram_addr          输入地址（SPM，32-bit）。
     * @param output_addr        输出地址（SPM，32-bit）。
     * @param n                  batch，软件入参 16-bit（建议 >=1）。
     * @param c                  channel，软件入参 16-bit（建议 >=1）。
     * @param h                  height，软件入参 16-bit（建议 >=1）。
     * @param w                  width，软件入参 16-bit（建议 >=1）。
     */
    void npu_layout_nchw_to_nchwc32(
        uint32_t sram_addr,
        uint32_t output_addr,
        uint16_t n,
        uint16_t c,
        uint16_t h,
        uint16_t w
    );

    /**
     * @brief NCHWC32 -> NCHW
     * @param sram_addr          输入地址（SPM，32-bit）。
     * @param output_addr        输出地址（SPM，32-bit）。
     * @param n                  batch，软件入参 16-bit（建议 >=1）。
     * @param c                  channel，软件入参 16-bit（建议 >=1）。
     * @param h                  height，软件入参 16-bit（建议 >=1）。
     * @param w                  width，软件入参 16-bit（建议 >=1）。
     */
    void npu_layout_nchwc32_to_nchw(
        uint32_t sram_addr,
        uint32_t output_addr,
        uint16_t n,
        uint16_t c,
        uint16_t h,
        uint16_t w
    );

    // ---------------------------------------------------------------------
    // Test Interface
    // ---------------------------------------------------------------------

    // 测试接口：参数定义、推荐取值、位宽与 npu_dma_mvin() 完全一致。
    void npu_dma_mvin_test(
        void* host_ptr,
        uint32_t sram_addr,
        uint32_t col_num,
        uint32_t row_num,
        uint16_t sram_stride,
        uint32_t dram_stride,
        uint8_t  precision,    // 2-bit
        uint8_t  input_type,   // 2-bit
        bool     dest,         // 1-bit: 0=SPM, 1=ACC
        bool     is_bias,      // 1-bit
        bool     is_quant,     // 1-bit
        uint32_t quant_zero,
        uint16_t quant_scale,
        uint16_t quant_shift
    );

    // 测试接口：参数定义、推荐取值、位宽与 npu_dma_mvout() 完全一致。
    void npu_dma_mvout_test(
        void* host_ptr,
        uint32_t sram_addr,
        uint32_t col_num,
        uint32_t row_num,
        uint16_t sram_stride,
        uint32_t dram_stride,
        uint8_t  precision,    // 2-bit
        uint8_t  output_type,  // 2-bit
        bool     source,       // 1-bit: 0=SPM, 1=ACC
        bool     is_quant,     // 1-bit
        uint32_t quant_zero,
        uint16_t quant_scale,
        uint16_t quant_shift
    );
}

#endif // NPU_RUNTIME_H
