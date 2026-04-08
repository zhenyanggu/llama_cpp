/**
 * conv_test.cpp: Basic NPU Convolution Test (using npu_conv_run API)
 * 
 * Test Strategy:
 * 1. Basic case: 3x3 conv, single 32-channel block
 * 2. Use fixed data for easier debugging
 * 3. IFM layout: NCHWC32 [C/32][H][W][32]
 * 4. Weight layout: blocked [Kh][Kw][Cin][Cout] (single 32x32 block)
 */

#include "../npu_runtime.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <set>
#include <tuple>
#include <vector>

// ==========================================
// Test Case Configuration
// ==========================================

// Base test dimensions
#define H_IN        4       // Input height
#define W_IN        4       // Input width
#define C_IN        32      // Input channels (must be 32 for single block)
#define C_OUT       32      // Output channels (must be 32 for single block)
#define K_H         3       // Kernel height (3x3 convolution)
#define K_W         3       // Kernel width
#define STRIDE      1       // Stride
#define DILATION    1       // Dilation

// No padding for simplest case
#define P_TOP       0
#define P_BOTTOM    0
#define P_LEFT      0
#define P_RIGHT     0

// Hardware constants
#define SA_SIZE     32

// Calculated output dimensions
#define H_OUT       ((H_IN + P_TOP + P_BOTTOM - DILATION * (K_H - 1) - 1) / STRIDE + 1)
#define W_OUT       ((W_IN + P_LEFT + P_RIGHT - DILATION * (K_W - 1) - 1) / STRIDE + 1)

// Buffer sizes  
#define IFM_SIZE    (H_IN * W_IN * C_IN)
#define WEIGHT_SIZE (K_H * K_W * C_IN * C_OUT)
#define OFM_SIZE    (H_OUT * W_OUT * C_OUT)

// SPM/ACC Addresses
// SPM address is assigned dynamically based on IFM/Weight/OFM sizes.
static constexpr uint32_t SPM_ADDR_LIMIT = 0x80000;
static constexpr uint32_t ACC_ADDR_LIMIT = 0x40000;
static uint32_t SRAM_ADDR_IFM = 0;
static uint32_t SRAM_ADDR_WEIGHT = 0;
static uint32_t SRAM_ADDR_OFM = 0;
static constexpr uint32_t ACC_ADDR_PSUM = 0x0000;   // ACC address for int32 readback debug

static inline uint32_t align_up_u32(uint32_t x, uint32_t align) {
    return (x + align - 1u) & ~(align - 1u);
}

static bool assign_spm_layout(uint32_t ifm_size, uint32_t weight_size, uint32_t ofm_size,
                              const char* tag) {
    // Simple bump allocator with 32-byte alignment.
    uint32_t cursor = 0;
    SRAM_ADDR_IFM = align_up_u32(cursor, 32u);
    cursor = SRAM_ADDR_IFM + ifm_size;

    SRAM_ADDR_WEIGHT = align_up_u32(cursor, 32u);
    cursor = SRAM_ADDR_WEIGHT + weight_size;

    SRAM_ADDR_OFM = align_up_u32(cursor, 32u);
    cursor = SRAM_ADDR_OFM + ofm_size;

    if (cursor > SPM_ADDR_LIMIT) {
        std::cerr << "[SPM] " << tag << " layout overflow: need 0x"
                  << std::hex << cursor << " > limit 0x" << SPM_ADDR_LIMIT << std::dec << std::endl;
        return false;
    }

    std::cout << "[SPM] " << tag
              << " IFM=0x" << std::hex << SRAM_ADDR_IFM
              << " WGT=0x" << SRAM_ADDR_WEIGHT
              << " OFM=0x" << SRAM_ADDR_OFM
              << " END=0x" << cursor
              << " (limit=0x" << SPM_ADDR_LIMIT << ")"
              << std::dec << std::endl;
    return true;
}

static bool check_acc_capacity(uint32_t ofm_elems, const char* tag) {
    uint64_t need_bytes = static_cast<uint64_t>(ofm_elems) * sizeof(int32_t);
    if (need_bytes > ACC_ADDR_LIMIT) {
        std::cerr << "[ACC] " << tag << " overflow: need 0x"
                  << std::hex << need_bytes << " > limit 0x" << ACC_ADDR_LIMIT
                  << std::dec << " (ofm_elems=" << ofm_elems << ")" << std::endl;
        return false;
    }

    std::cout << "[ACC] " << tag
              << " need=0x" << std::hex << need_bytes
              << " (limit=0x" << ACC_ADDR_LIMIT << ")"
              << std::dec << std::endl;
    return true;
}

// ==========================================
// Debug Flags (set to 1 to enable)
// ==========================================
#define DBG_MVIN_READBACK   0   // MVIN->MVOUT loopback verify IFM & Weight
#define DBG_ACC_READBACK    0  // Run conv to ACC(int32), MVOUT raw accumulators
#define DBG_HEX_DUMP        0   // Print hex dumps of raw memory
#define DBG_CHANNEL_DUMP    0   // Per-position channel-by-channel dump on mismatch
#define DBG_SPATIAL_SWEEP   0   // Run multiple H/W combinations to find failure pattern
#define DBG_CONV_TILE_TEST  1   // Run npu_conv_tile_run tile-shape tests

// ==========================================
// Quantization Helpers (from gemm_test)
// ==========================================

struct MulShift {
    int16_t scale;
    int16_t shift;
};

static MulShift pick_output_scale_shift(float M) {
    MulShift r{0, 0};
    if (M <= 0.0f) return r;

    int bestN = -1;
    int32_t bestScale = 0;

    for (int N = 0; N <= 30; N++) {
        double s = static_cast<double>(M) * static_cast<double>(1u << N);
        int32_t sc = static_cast<int32_t>(std::llround(s));
        if (sc <= 0) continue;
        if (sc > 32767) break;
        bestN = N;
        bestScale = sc;
    }
    if (bestN < 0) {
        r.scale = 1;
        r.shift = -30;
        return r;
    }
    r.scale = static_cast<int16_t>(bestScale);
    r.shift = static_cast<int16_t>(-bestN);
    return r;
}

static int8_t quant_i32_to_i8(int32_t x, int16_t scale, int16_t shift) {
    int64_t v = static_cast<int64_t>(x) * static_cast<int64_t>(scale);
    if (shift < 0) {
        int s = -shift;
        if (s > 0) {
            int64_t round = 1LL << (s - 1);
            v = (v >= 0) ? (v + round) : (v - round);
        }
        v >>= s;
    } else if (shift > 0) {
        v <<= shift;
    }

    if (v > 127) v = 127;
    if (v < -128) v = -128;
    return static_cast<int8_t>(v);
}

// ==========================================
// Data Layout Helpers
// ==========================================

/**
 * @brief Calculate IFM index in NCHWC32 format
 * Layout: [C/32][H][W][32]
 * For single 32-channel block: [H][W][32]
 */
static inline int ifm_index_nchwc32(int c, int h, int w) {
    int c_block = c / SA_SIZE;
    int c_inner = c % SA_SIZE;
    return ((c_block * H_IN + h) * W_IN + w) * SA_SIZE + c_inner;
}

/**
 * @brief Calculate OFM index in NCHWC32 format
 * Layout: [C/32][H][W][32]
 */
static inline int ofm_index_nchwc32(int c, int h, int w) {
    int c_block = c / SA_SIZE;
    int c_inner = c % SA_SIZE;
    return ((c_block * H_OUT + h) * W_OUT + w) * SA_SIZE + c_inner;
}

/**
 * @brief Calculate Weight index in blocked format
 * Layout: [Cout/32][Cin/32][Kh][Kw][Cin32][Cout32]
 */
static inline int weight_index_blocked(int cout, int cin, int kh, int kw) {
    int cout_blk = cout / SA_SIZE;
    int cout_inner = cout % SA_SIZE;
    int cin_blk = cin / SA_SIZE;
    int cin_inner = cin % SA_SIZE;
    int cin_blks = C_IN / SA_SIZE;
    return (((((cout_blk * cin_blks + cin_blk) * K_H + kh) * K_W + kw)
             * SA_SIZE + cin_inner) * SA_SIZE + cout_inner);
}

// Dynamic (runtime-sized) layout helpers for conv_tile tests
static inline int ifm_index_nchwc32_dyn(int c, int h, int w, int H, int W) {
    int c_block = c / SA_SIZE;
    int c_inner = c % SA_SIZE;
    return ((c_block * H + h) * W + w) * SA_SIZE + c_inner;
}

static inline int ifm_index_dyn(int c, int h, int w, int H, int W, int C) {
    if (C < SA_SIZE) {
        return (h * W + w) * C + c;  // NHWC
    }
    int c_block = c / SA_SIZE;
    int c_inner = c % SA_SIZE;
    return ((c_block * H + h) * W + w) * SA_SIZE + c_inner;  // NCHWC32
}

static inline int ofm_index_nchwc32_dyn(int c, int h, int w, int H, int W) {
    int c_block = c / SA_SIZE;
    int c_inner = c % SA_SIZE;
    return ((c_block * H + h) * W + w) * SA_SIZE + c_inner;
}

static inline int weight_index_blocked_dyn(int cout, int cin, int kh, int kw,
                                            int K_H_d, int K_W_d, int C_IN_d, int C_OUT_d) {
    int cout_blk = cout / SA_SIZE;
    int cout_inner = cout % SA_SIZE;

    // cin < 32: [Cout/32][Kh][Kw][Cin][Cout32]
    if (C_IN_d < SA_SIZE) {
        (void)C_OUT_d;
        return ((((cout_blk * K_H_d + kh) * K_W_d + kw) * C_IN_d + cin) * SA_SIZE + cout_inner);
    }

    // cin >= 32: [Cout/32][Cin/32][Kh][Kw][Cin32][Cout32]
    int cin_blk = cin / SA_SIZE;
    int cin_inner = cin % SA_SIZE;
    int cin_blks = C_IN_d / SA_SIZE;
    (void)C_OUT_d;
    return (((((cout_blk * cin_blks + cin_blk) * K_H_d + kh) * K_W_d + kw)
             * SA_SIZE + cin_inner) * SA_SIZE + cout_inner);
}

// ==========================================
// Debug Utilities
// ==========================================

/**
 * @brief Print raw hex dump of a buffer
 */
static void hex_dump(const char* label, const void* data, int len, int bytes_per_line = 32) {
#if DBG_HEX_DUMP
    const uint8_t* p = static_cast<const uint8_t*>(data);
    std::cout << "[HexDump] " << label << " (" << len << " bytes):" << std::endl;
    for (int i = 0; i < len && i < 256; i += bytes_per_line) {
        std::cout << "  " << std::hex << std::setfill('0') << std::setw(4) << i << ": ";
        for (int j = 0; j < bytes_per_line && (i + j) < len && (i + j) < 256; j++) {
            std::cout << std::setw(2) << static_cast<int>(p[i + j]) << " ";
        }
        std::cout << std::dec << std::setfill(' ') << std::endl;
    }
    if (len > 256) std::cout << "  ... (" << len - 256 << " more bytes)" << std::endl;
#endif
}

/**
 * @brief Print int8 matrix in readable form
 */
static void print_matrix_i8(const char* name, const int8_t* data, int rows, int cols,
                            int max_rows = 8, int max_cols = 16) {
    std::cout << name << " (" << rows << "x" << cols << "):" << std::endl;
    for (int i = 0; i < rows && i < max_rows; ++i) {
        std::cout << "  [";
        for (int j = 0; j < cols && j < max_cols; ++j) {
            std::cout << std::setw(4) << static_cast<int>(data[i * cols + j]);
            if (j < cols - 1 && j < max_cols - 1) std::cout << ",";
        }
        if (cols > max_cols) std::cout << ", ...";
        std::cout << "]" << std::endl;
    }
    if (rows > max_rows) std::cout << "  ... (" << rows - max_rows << " more rows)" << std::endl;
}

/**
 * @brief Print int32 matrix in readable form
 */
static void print_matrix_i32(const char* name, const int32_t* data, int rows, int cols,
                             int max_rows = 8, int max_cols = 8) {
    std::cout << name << " (" << rows << "x" << cols << "):" << std::endl;
    for (int i = 0; i < rows && i < max_rows; ++i) {
        std::cout << "  [";
        for (int j = 0; j < cols && j < max_cols; ++j) {
            std::cout << std::setw(8) << data[i * cols + j];
            if (j < cols - 1 && j < max_cols - 1) std::cout << ",";
        }
        if (cols > max_cols) std::cout << ", ...";
        std::cout << "]" << std::endl;
    }
    if (rows > max_rows) std::cout << "  ... (" << rows - max_rows << " more rows)" << std::endl;
}

/**
 * @brief Print NCHWC32 buffer in per-position channel view
 *   For each (h,w), print all channels
 */
static void dump_nchwc32(const char* label, const int8_t* buf,
                         int C, int H, int W, int max_ch = 32) {
    std::cout << "[NCHWC32 Dump] " << label << " C=" << C << " H=" << H << " W=" << W << std::endl;
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            std::cout << "  (" << h << "," << w << ") ch[0.." << std::min(max_ch, C) - 1 << "]: ";
            for (int c = 0; c < C && c < max_ch; c++) {
                int c_block = c / SA_SIZE;
                int c_inner = c % SA_SIZE;
                int idx = ((c_block * H + h) * W + w) * SA_SIZE + c_inner;
                std::cout << std::setw(4) << static_cast<int>(buf[idx]);
            }
            std::cout << std::endl;
        }
    }
}

/**
 * @brief DMA readback verification: MVIN data -> MVOUT to readback buffer -> compare
 * @return number of mismatches (0 = DMA is correct)
 */
static int verify_mvin_readback(const char* label,
                                const int8_t* original, int size,
                                uint32_t sram_addr) {
    // Allocate readback buffer
    int8_t* readback = static_cast<int8_t*>(npu_mem_alloc(size));
    if (!readback) {
        std::cerr << "[READBACK] Failed to alloc readback buffer for " << label << std::endl;
        return -1;
    }
    std::memset(readback, 0xAA, size);  // Fill with sentinel

    // MVOUT from SPM back to readback buffer.
    // NOTE:
    //   DMA col_num/row_num are uint16 (minus-one encoded), so a single 1D
    //   transfer cannot exceed 65536 bytes. For large buffers, split into
    //   2D chunks (rows of 65536 bytes) to avoid truncation/wrap-around.
    constexpr uint32_t kMaxElemsPerRow = 0x10000u;   // col_num max + 1
    constexpr uint32_t kMaxRowsPerCall = 0x10000u;   // row_num max + 1

    uint32_t remain = static_cast<uint32_t>(size);
    uint32_t host_off = 0;
    uint32_t sram_off = 0;

    while (remain > 0) {
        if (remain <= kMaxElemsPerRow) {
            // Tail (<= 65536B): 1D transfer
            npu_dma_mvout(
                /*host_ptr=*/readback + host_off,
                /*sram_addr=*/sram_addr + sram_off,
                /*col_num=*/static_cast<uint16_t>(remain - 1),
                /*row_num=*/0,
                /*sram_stride=*/0,
                /*dram_stride=*/0,
                /*precision=*/1,
                /*output_type=*/0,  // int8
                /*source=*/0,       // SPM
                /*is_quant=*/false,
                /*quant_zero=*/0,
                /*quant_scale=*/0,
                /*quant_shift=*/0
            );
            break;
        }

        // Full-row 2D chunk, each row is 65536 bytes.
        uint32_t full_rows = remain / kMaxElemsPerRow;
        uint32_t rows_this_call = std::min(full_rows, kMaxRowsPerCall);
        uint32_t bytes_this_call = rows_this_call * kMaxElemsPerRow;

        npu_dma_mvout(
            /*host_ptr=*/readback + host_off,
            /*sram_addr=*/sram_addr + sram_off,
            /*col_num=*/static_cast<uint16_t>(kMaxElemsPerRow - 1),
            /*row_num=*/static_cast<uint16_t>(rows_this_call - 1),
            /*sram_stride=*/static_cast<uint16_t>(kMaxElemsPerRow),
            /*dram_stride=*/kMaxElemsPerRow,
            /*precision=*/1,
            /*output_type=*/0,  // int8
            /*source=*/0,       // SPM
            /*is_quant=*/false,
            /*quant_zero=*/0,
            /*quant_scale=*/0,
            /*quant_shift=*/0
        );

        remain -= bytes_this_call;
        host_off += bytes_this_call;
        sram_off += bytes_this_call;
    }

    // Compare byte-by-byte
    int mismatches = 0;
    int first_mismatch = -1;
    for (int i = 0; i < size; i++) {
        if (readback[i] != original[i]) {
            if (mismatches < 10) {
                std::cout << "  [READBACK MISMATCH] " << label
                          << " idx=" << i
                          << " original=" << static_cast<int>(original[i])
                          << " readback=" << static_cast<int>(readback[i])
                          << std::endl;
            }
            if (first_mismatch < 0) first_mismatch = i;
            mismatches++;
        }
    }

    if (mismatches == 0) {
        std::cout << "  [READBACK] " << label << " OK - all " << size << " bytes match" << std::endl;
    } else {
        std::cout << "  [READBACK] " << label << " FAIL - " << mismatches << "/" << size
                  << " mismatches, first at idx=" << first_mismatch << std::endl;
        // Print hex comparison around first mismatch
        int start = std::max(0, first_mismatch - 8);
        int end = std::min(size, first_mismatch + 24);
        std::cout << "  Original  [" << start << ".." << end-1 << "]: ";
        for (int i = start; i < end; i++)
            std::cout << std::hex << std::setw(2) << std::setfill('0') << (static_cast<int>(original[i]) & 0xFF) << " ";
        std::cout << std::dec << std::setfill(' ') << std::endl;
        std::cout << "  Readback  [" << start << ".." << end-1 << "]: ";
        for (int i = start; i < end; i++)
            std::cout << std::hex << std::setw(2) << std::setfill('0') << (static_cast<int>(readback[i]) & 0xFF) << " ";
        std::cout << std::dec << std::setfill(' ') << std::endl;
    }

    npu_mem_free(readback);
    return mismatches;
}

/**
 * @brief Per-position channel dump for mismatch analysis
 */
static void channel_dump_at(const char* label, int oh, int ow,
                            const int8_t* hw_out, const int8_t* ref_out,
                            const int32_t* ref_acc, const int32_t* hw_acc,
                            int16_t scale, int16_t shift) {
    std::cout << "[ChannelDump] " << label << " at (" << oh << "," << ow << "):" << std::endl;
    std::cout << "  ch |   hw_i8 |  ref_i8 | diff_i8 |   hw_i32 |  ref_i32 | diff_i32 |  re-quant" << std::endl;
    std::cout << "  ---+---------+---------+---------+----------+----------+----------+----------" << std::endl;
    for (int c = 0; c < C_OUT; c++) {
        int idx = ofm_index_nchwc32(c, oh, ow);
        int hw8  = static_cast<int>(hw_out[idx]);
        int ref8 = static_cast<int>(ref_out[idx]);
        int32_t hw32  = hw_acc ? hw_acc[idx] : 0;
        int32_t ref32 = ref_acc[idx];
        int rq = static_cast<int>(quant_i32_to_i8(ref32, scale, shift));
        std::cout << "  " << std::setw(3) << c
                  << " | " << std::setw(7) << hw8
                  << " | " << std::setw(7) << ref8
                  << " | " << std::setw(7) << (hw8 - ref8)
                  << " | " << std::setw(8) << hw32
                  << " | " << std::setw(8) << ref32
                  << " | " << std::setw(8) << (hw32 - ref32)
                  << " | " << std::setw(8) << rq
                  << std::endl;
    }
}

// ==========================================
// Test Data Generation
// ==========================================

static void generate_fixed_data(int8_t* ifm, int8_t* weight) {
    // Generate IFM in NCHWC32 layout with simple fixed pattern
    for (int c = 0; c < C_IN; c++) {
        for (int h = 0; h < H_IN; h++) {
            for (int w = 0; w < W_IN; w++) {
                int idx = ifm_index_nchwc32(c, h, w);
                // Simple pattern: small values to avoid overflow
                int val = (h + w + c) % 5 + 1;  // Values 1-5
                ifm[idx] = static_cast<int8_t>(val);
            }
        }
    }

    // Generate Weight in blocked layout [Cout/32][Cin/32][Kh][Kw][Cin32][Cout32]
    for (int cout = 0; cout < C_OUT; cout++) {
        for (int cin = 0; cin < C_IN; cin++) {
            for (int kh = 0; kh < K_H; kh++) {
                for (int kw = 0; kw < K_W; kw++) {
                    int idx = weight_index_blocked(cout, cin, kh, kw);
                    // Simple pattern: small values
                    int val = (cout + cin + kh + kw) % 5 + 1;  // Values 1-5
                    weight[idx] = static_cast<int8_t>(val);
                }
            }
        }
    }
}

static void generate_fixed_data_dyn(int8_t* ifm, int8_t* weight,
                                    int H_IN_d, int W_IN_d, int C_IN_d, int C_OUT_d,
                                    int K_H_d, int K_W_d) {
    for (int c = 0; c < C_IN_d; c++) {
        for (int h = 0; h < H_IN_d; h++) {
            for (int w = 0; w < W_IN_d; w++) {
                int idx = ifm_index_dyn(c, h, w, H_IN_d, W_IN_d, C_IN_d);
                int val = (h + w + c) % 5 + 1;
                ifm[idx] = static_cast<int8_t>(val);
            }
        }
    }

    for (int cout = 0; cout < C_OUT_d; cout++) {
        for (int cin = 0; cin < C_IN_d; cin++) {
            for (int kh = 0; kh < K_H_d; kh++) {
                for (int kw = 0; kw < K_W_d; kw++) {
                    int idx = weight_index_blocked_dyn(cout, cin, kh, kw, K_H_d, K_W_d, C_IN_d, C_OUT_d);
                    int val = (cout + cin + kh + kw) % 5 + 1;
                    weight[idx] = static_cast<int8_t>(val);
                }
            }
        }
    }
}

// ==========================================
// Reference Convolution
// ==========================================

static void compute_reference_conv(const int8_t* ifm, const int8_t* weight, 
                                   int32_t* ref_acc, int8_t* ref_out,
                                   int16_t scale, int16_t shift) {
    // Compute convolution in int32
    for (int oh = 0; oh < H_OUT; oh++) {
        for (int ow = 0; ow < W_OUT; ow++) {
            for (int oc = 0; oc < C_OUT; oc++) {
                int32_t sum = 0;
                
                for (int kh = 0; kh < K_H; kh++) {
                    for (int kw = 0; kw < K_W; kw++) {
                        int ih = oh * STRIDE + kh * DILATION - P_TOP;
                        int iw = ow * STRIDE + kw * DILATION - P_LEFT;
                        
                        // Skip if in padding region
                        if (ih < 0 || ih >= H_IN || iw < 0 || iw >= W_IN) {
                            continue;
                        }
                        
                        for (int ic = 0; ic < C_IN; ic++) {
                            int ifm_idx = ifm_index_nchwc32(ic, ih, iw);
                            int wgt_idx = weight_index_blocked(oc, ic, kh, kw);
                            
                            sum += static_cast<int32_t>(ifm[ifm_idx]) * 
                                   static_cast<int32_t>(weight[wgt_idx]);
                        }
                    }
                }
                
                // Store int32 accumulator
                int out_idx = ofm_index_nchwc32(oc, oh, ow);
                ref_acc[out_idx] = sum;
                
                // Quantize to int8
                ref_out[out_idx] = quant_i32_to_i8(sum, scale, shift);
            }
        }
    }
}

static void compute_reference_conv_dyn(const int8_t* ifm, const int8_t* weight,
                                       int H_IN_d, int W_IN_d, int C_IN_d, int C_OUT_d,
                                       int K_H_d, int K_W_d, int STRIDE_d, int DILATION_d,
                                       int P_TOP_d, int P_LEFT_d,
                                       int H_OUT_d, int W_OUT_d,
                                       int32_t* ref_acc, int8_t* ref_out,
                                       int16_t scale, int16_t shift) {
    for (int oh = 0; oh < H_OUT_d; oh++) {
        for (int ow = 0; ow < W_OUT_d; ow++) {
            for (int oc = 0; oc < C_OUT_d; oc++) {
                int32_t sum = 0;
                for (int kh = 0; kh < K_H_d; kh++) {
                    for (int kw = 0; kw < K_W_d; kw++) {
                        int ih = oh * STRIDE_d + kh * DILATION_d - P_TOP_d;
                        int iw = ow * STRIDE_d + kw * DILATION_d - P_LEFT_d;
                        if (ih < 0 || ih >= H_IN_d || iw < 0 || iw >= W_IN_d) {
                            continue;
                        }
                        for (int ic = 0; ic < C_IN_d; ic++) {
                            int ifm_idx = ifm_index_dyn(ic, ih, iw, H_IN_d, W_IN_d, C_IN_d);
                            int wgt_idx = weight_index_blocked_dyn(oc, ic, kh, kw,
                                                                    K_H_d, K_W_d,
                                                                    C_IN_d, C_OUT_d);
                            sum += static_cast<int32_t>(ifm[ifm_idx]) *
                                   static_cast<int32_t>(weight[wgt_idx]);
                        }
                    }
                }
                int out_idx = ofm_index_nchwc32_dyn(oc, oh, ow, H_OUT_d, W_OUT_d);
                ref_acc[out_idx] = sum;
                ref_out[out_idx] = quant_i32_to_i8(sum, scale, shift);
            }
        }
    }
}

// ==========================================
// Test Case
// ==========================================

struct CaseStats {
    size_t total = 0;
    size_t mismatch = 0;
    int max_diff = 0;
    bool skipped = false;
};

static CaseStats run_conv_test(int8_t* host_ifm, int8_t* host_weight, int8_t* host_out) {
    CaseStats stats;

    if (!assign_spm_layout(IFM_SIZE, WEIGHT_SIZE, OFM_SIZE, "conv_test")) {
        std::cout << "[CASE] Skip conv_test (SPM overflow)" << std::endl;
        stats.skipped = true;
        return stats;
    }
    if (!check_acc_capacity(OFM_SIZE, "conv_test")) {
        std::cout << "[CASE] Skip conv_test (ACC overflow)" << std::endl;
        stats.skipped = true;
        return stats;
    }
    
    std::cout << "======================================" << std::endl;
    std::cout << "      NPU Conv Test (npu_conv_run)    " << std::endl;
    std::cout << "======================================" << std::endl;
    std::cout << "IFM: " << H_IN << "x" << W_IN << "x" << C_IN << std::endl;
    std::cout << "Weight: " << K_H << "x" << K_W << "x" << C_IN << "x" << C_OUT << std::endl;
    std::cout << "OFM: " << H_OUT << "x" << W_OUT << "x" << C_OUT << std::endl;
    std::cout << "Stride=" << STRIDE << ", Dilation=" << DILATION << std::endl;
    std::cout << "======================================" << std::endl;

    // 1. Generate fixed test data
    generate_fixed_data(host_ifm, host_weight);
    std::memset(host_out, 0, OFM_SIZE);
    
    // 2. Compute quantization parameters
    // For simplicity, use fixed scale/shift that works for small values
    // Since we use values 1-5, max accumulator ~ 5 * 5 * 32 = 800
    // We want this to map to ~127, so scale ~ 127/800 ~ 0.16
    float Mscale = 0.16f;
    MulShift ms = pick_output_scale_shift(Mscale);
    
    std::cout << "[Quant] scale=" << ms.scale << " shift=" << ms.shift << std::endl;
    
    // 3. Compute reference
    std::vector<int32_t> ref_acc(OFM_SIZE);
    std::vector<int8_t> ref_out(OFM_SIZE);
    compute_reference_conv(host_ifm, host_weight, ref_acc.data(), ref_out.data(),
                           ms.scale, ms.shift);
    
    // Print minimal reference (first position, first channel only)
    std::cout << "[Ref] Pos(0,0) ch0: acc=" << ref_acc[ofm_index_nchwc32(0, 0, 0)] 
              << " out=" << static_cast<int>(ref_out[ofm_index_nchwc32(0, 0, 0)]) << std::endl;
    
    // 4. MVIN IFM to SPM
    std::cout << "\n[MVIN] Loading IFM to SPM (addr=0x" << std::hex << SRAM_ADDR_IFM 
              << ", size=" << std::dec << IFM_SIZE << ")..." << std::endl;
    npu_dma_mvin(
        /*host_ptr=*/host_ifm,
        /*sram_addr=*/SRAM_ADDR_IFM,
        /*col_num=*/static_cast<uint16_t>(IFM_SIZE - 1),
        /*row_num=*/0,
        /*sram_stride=*/0,
        /*dram_stride=*/0,
        /*precision=*/1,
        /*input_type=*/0,   // IFM
        /*dest=*/0,         // SPM
        /*is_bias=*/false,
        /*is_quant=*/false,
        /*quant_zero=*/0,
        /*quant_scale=*/0,
        /*quant_shift=*/0
    );
    
    // 4b. Readback IFM from SPM to verify MVIN
#if DBG_MVIN_READBACK
    std::cout << "\n[DEBUG] Verifying IFM MVIN readback..." << std::endl;
    int ifm_rb_err = verify_mvin_readback("IFM", host_ifm, IFM_SIZE, SRAM_ADDR_IFM);
    if (ifm_rb_err != 0) {
        std::cout << "[DEBUG] *** IFM DMA broken! Fix DMA before debugging conv. ***" << std::endl;
    }
#endif
    
    // 5. MVIN Weight to SPM
    std::cout << "\n[MVIN] Loading Weight to SPM (addr=0x" << std::hex << SRAM_ADDR_WEIGHT 
              << ", size=" << std::dec << WEIGHT_SIZE << ")..." << std::endl;
    npu_dma_mvin(
        /*host_ptr=*/host_weight,
        /*sram_addr=*/SRAM_ADDR_WEIGHT,
        /*col_num=*/static_cast<uint16_t>(WEIGHT_SIZE - 1),
        /*row_num=*/0,
        /*sram_stride=*/0,
        /*dram_stride=*/0,
        /*precision=*/1,
        /*input_type=*/1,   // Weight
        /*dest=*/0,         // SPM
        /*is_bias=*/false,
        /*is_quant=*/false,
        /*quant_zero=*/0,
        /*quant_scale=*/0,
        /*quant_shift=*/0
    );
    
    // 5b. Readback Weight from SPM to verify MVIN
#if DBG_MVIN_READBACK
    std::cout << "\n[DEBUG] Verifying Weight MVIN readback..." << std::endl;
    int wgt_rb_err = verify_mvin_readback("Weight", host_weight, WEIGHT_SIZE, SRAM_ADDR_WEIGHT);
    if (wgt_rb_err != 0) {
        std::cout << "[DEBUG] *** Weight DMA broken! Fix DMA before debugging conv. ***" << std::endl;
    }
    
    if (ifm_rb_err != 0 || wgt_rb_err != 0) {
        std::cout << "\n[DEBUG] ========================================" << std::endl;
        std::cout << "[DEBUG]  DMA READBACK FAILED - skipping conv   " << std::endl;
        std::cout << "[DEBUG]  Fix DMA first before proceeding!       " << std::endl;
        std::cout << "[DEBUG] ========================================" << std::endl;
        // Still continue so we can see conv output too, but flag it
    }
    
    // Re-MVIN IFM since readback MVOUT may have triggered state changes
    npu_dma_mvin(
        /*host_ptr=*/host_ifm,
        /*sram_addr=*/SRAM_ADDR_IFM,
        /*col_num=*/static_cast<uint16_t>(IFM_SIZE - 1),
        /*row_num=*/0,
        /*sram_stride=*/0,
        /*dram_stride=*/0,
        /*precision=*/1,
        /*input_type=*/0,  // IFM
        /*dest=*/0,        // SPM
        /*is_bias=*/false,
        /*is_quant=*/false,
        /*quant_zero=*/0,
        /*quant_scale=*/0,
        /*quant_shift=*/0
    );
    npu_dma_mvin(
        /*host_ptr=*/host_weight,
        /*sram_addr=*/SRAM_ADDR_WEIGHT,
        /*col_num=*/static_cast<uint16_t>(WEIGHT_SIZE - 1),
        /*row_num=*/0,
        /*sram_stride=*/0,
        /*dram_stride=*/0,
        /*precision=*/1,
        /*input_type=*/1,  // Weight
        /*dest=*/0,        // SPM
        /*is_bias=*/false,
        /*is_quant=*/false,
        /*quant_zero=*/0,
        /*quant_scale=*/0,
        /*quant_shift=*/0
    );
#endif

    // 6a. [DEBUG] Run conv to ACC first to get raw int32 accumulators
    //     Use is_accumulate=1, is_bias=0 to match npu_conv_tile_run pattern.
    //     accout_dest=1 + is_accumulate=0 may be an unsupported combination.
#if DBG_ACC_READBACK
    std::cout << "\n[DEBUG] Running conv -> ACC (int32 readback)..." << std::endl;
    npu_conv_run(
        /*pad_top=*/static_cast<uint8_t>(P_TOP),
        /*pad_bottom=*/static_cast<uint8_t>(P_BOTTOM),
        /*pad_left=*/static_cast<uint8_t>(P_LEFT),
        /*pad_right=*/static_cast<uint8_t>(P_RIGHT),
        /*pad_mode=*/0,
        /*weight_shape_m1=*/static_cast<uint8_t>(K_H - 1),
        /*weight_stride_m1=*/static_cast<uint8_t>(STRIDE - 1),
        /*weight_dilation_m1=*/static_cast<uint8_t>(DILATION - 1),
        /*is_group_conv=*/false,
        /*int_type=*/0,
        /*op_type=*/1,
        /*dataflow_mode=*/0,
        /*accout_dest=*/1,      // *** Output to ACC (int32) ***
        /*input_a_zeropoint=*/0,
        /*input_b_zeropoint=*/0,
        /*input_a_addr=*/SRAM_ADDR_IFM,
        /*input_a_col_num_m1=*/static_cast<uint16_t>(W_IN - 1),
        /*input_a_row_num_m1=*/static_cast<uint8_t>(H_IN - 1),
        /*input_a_stride=*/static_cast<uint16_t>(W_IN),
        /*input_b_addr=*/SRAM_ADDR_WEIGHT,
        /*input_b_col_num_m1=*/static_cast<uint8_t>(C_OUT - 1),
        /*input_b_row_num_m1=*/static_cast<uint16_t>(C_IN - 1),
        /*input_b_stride=*/static_cast<uint16_t>(C_OUT),
        /*biaspsum_width=*/static_cast<uint8_t>(W_OUT),
        /*biaspsum_height=*/static_cast<uint8_t>(H_OUT),
        /*biaspsum_addr=*/ACC_ADDR_PSUM,
        /*biaspsum_stride=*/static_cast<uint16_t>(W_OUT),
        /*output_addr=*/ACC_ADDR_PSUM,
        /*output_stride=*/static_cast<uint16_t>(W_OUT),
        /*is_accumulate=*/true,   // Must be true for ACC output path
        /*relu_enable=*/false,
        /*relu_type=*/0,
        /*is_bias=*/false,        // Not bias, just direct write (psum=0 first time)
        /*output_zeropoint=*/0,
        /*quant_scale=*/1,
        /*quant_scaleshift=*/0  // identity, unused for ACC output
    );

    // MVOUT int32 from ACC (must use npu_mem_alloc, not std::vector)
    int32_t* hw_acc_buf = static_cast<int32_t*>(npu_mem_alloc(OFM_SIZE * sizeof(int32_t)));
    if (!hw_acc_buf) {
        std::cerr << "[DEBUG] Failed to alloc ACC readback buffer!" << std::endl;
    } else {
        std::memset(hw_acc_buf, 0xEE, OFM_SIZE * sizeof(int32_t));  // sentinel
        npu_dma_mvout(
            /*host_ptr=*/hw_acc_buf,
            /*sram_addr=*/ACC_ADDR_PSUM,
            /*col_num=*/static_cast<uint16_t>(OFM_SIZE - 1),
            /*row_num=*/0,
            /*sram_stride=*/0,
            /*dram_stride=*/0,
            /*precision=*/1,
            /*output_type=*/1,  // int32
            /*source=*/true,    // ACC
            /*is_quant=*/false,
            /*quant_zero=*/0,
            /*quant_scale=*/0,
            /*quant_shift=*/0
        );
    }
    // Copy to std::vector for later use in channel_dump
    std::vector<int32_t> hw_acc(OFM_SIZE);
    if (hw_acc_buf) {
        std::memcpy(hw_acc.data(), hw_acc_buf, OFM_SIZE * sizeof(int32_t));
        npu_mem_free(hw_acc_buf);
    }

    // Compare int32 accumulators
    std::cout << "[DEBUG] ACC int32 comparison (ref vs hw):" << std::endl;
    int acc_mismatches = 0;
    for (int oh = 0; oh < H_OUT; oh++) {
        for (int ow = 0; ow < W_OUT; ow++) {
            std::cout << "  pos(" << oh << "," << ow << ") ch[0..7]: ";
            for (int c = 0; c < std::min(8, C_OUT); c++) {
                int idx = ofm_index_nchwc32(c, oh, ow);
                std::cout << "hw=" << hw_acc[idx] << "/ref=" << ref_acc[idx];
                if (hw_acc[idx] != ref_acc[idx]) {
                    std::cout << "*";
                    acc_mismatches++;
                }
                std::cout << "  ";
            }
            std::cout << std::endl;
        }
    }
    if (acc_mismatches == 0) {
        std::cout << "  [DEBUG] ACC int32 values MATCH reference - compute is correct!" << std::endl;
        std::cout << "  [DEBUG] If final int8 mismatches, problem is in quantization." << std::endl;
    } else {
        std::cout << "  [DEBUG] ACC int32 MISMATCH count: " << acc_mismatches << std::endl;
        std::cout << "  [DEBUG] Problem is in the compute path (im2col/SA), not quantization." << std::endl;
    }
    
    // Re-MVIN data for the actual int8 conv run
    npu_dma_mvin(
        /*host_ptr=*/host_ifm,
        /*sram_addr=*/SRAM_ADDR_IFM,
        /*col_num=*/static_cast<uint16_t>(IFM_SIZE - 1),
        /*row_num=*/0,
        /*sram_stride=*/0,
        /*dram_stride=*/0,
        /*precision=*/1,
        /*input_type=*/0,  // IFM
        /*dest=*/0,        // SPM
        /*is_bias=*/false,
        /*is_quant=*/false,
        /*quant_zero=*/0,
        /*quant_scale=*/0,
        /*quant_shift=*/0
    );
    npu_dma_mvin(
        /*host_ptr=*/host_weight,
        /*sram_addr=*/SRAM_ADDR_WEIGHT,
        /*col_num=*/static_cast<uint16_t>(WEIGHT_SIZE - 1),
        /*row_num=*/0,
        /*sram_stride=*/0,
        /*dram_stride=*/0,
        /*precision=*/1,
        /*input_type=*/1,  // Weight
        /*dest=*/0,        // SPM
        /*is_bias=*/false,
        /*is_quant=*/false,
        /*quant_zero=*/0,
        /*quant_scale=*/0,
        /*quant_shift=*/0
    );
#else
    std::vector<int32_t> hw_acc;  // empty, not used
#endif

    // 6. Run Convolution using npu_conv_run
    std::cout << "[CONV] Running convolution..." << std::endl;
    npu_conv_run(
        /*pad_top=*/static_cast<uint8_t>(P_TOP),
        /*pad_bottom=*/static_cast<uint8_t>(P_BOTTOM),
        /*pad_left=*/static_cast<uint8_t>(P_LEFT),
        /*pad_right=*/static_cast<uint8_t>(P_RIGHT),
        /*pad_mode=*/0,
        /*weight_shape_m1=*/static_cast<uint8_t>(K_H - 1),
        /*weight_stride_m1=*/static_cast<uint8_t>(STRIDE - 1),
        /*weight_dilation_m1=*/static_cast<uint8_t>(DILATION - 1),
        /*is_group_conv=*/false,
        /*int_type=*/0,     // int8
        /*op_type=*/1,      // Conv
        /*dataflow_mode=*/0,// im2col & OS
        /*accout_dest=*/0,  // Output to SPM (int8)
        /*input_a_zeropoint=*/0,
        /*input_b_zeropoint=*/0,
        /*input_a_addr=*/SRAM_ADDR_IFM,
        /*input_a_col_num_m1=*/static_cast<uint16_t>(W_IN - 1),
        /*input_a_row_num_m1=*/static_cast<uint8_t>(H_IN - 1),
        /*input_a_stride=*/static_cast<uint16_t>(W_IN),
        /*input_b_addr=*/SRAM_ADDR_WEIGHT,
        /*input_b_col_num_m1=*/static_cast<uint8_t>(C_OUT - 1),
        /*input_b_row_num_m1=*/static_cast<uint16_t>(C_IN - 1),
        /*input_b_stride=*/static_cast<uint16_t>(C_OUT),
        /*biaspsum_width=*/static_cast<uint8_t>(W_OUT),
        /*biaspsum_height=*/static_cast<uint8_t>(H_OUT),
        /*biaspsum_addr=*/0,
        /*biaspsum_stride=*/0,
        /*output_addr=*/SRAM_ADDR_OFM,
        /*output_stride=*/static_cast<uint16_t>(W_OUT),
        /*is_accumulate=*/false,  // No accumulation (first and only pass)
        /*relu_enable=*/false,
        /*relu_type=*/0,
        /*is_bias=*/false,        // No bias
        /*output_zeropoint=*/0,
        /*quant_scale=*/static_cast<uint16_t>(ms.scale),
        /*quant_scaleshift=*/static_cast<uint16_t>(ms.shift)
    );
    
    // 7. MVOUT OFM from SPM (2D transfer matching metal test pattern)
    // OFM layout in SPM: [H][W][32], so each row is W_OUT * 32 bytes
    std::cout << "[MVOUT] Reading OFM from SPM..." << std::endl;
    npu_dma_mvout(
        /*host_ptr=*/host_out,
        /*sram_addr=*/SRAM_ADDR_OFM,
        /*col_num=*/static_cast<uint16_t>(W_OUT * SA_SIZE - 1),  // Each row: W * 32 elements
        /*row_num=*/static_cast<uint16_t>(H_OUT - 1),            // H rows
        /*sram_stride=*/static_cast<uint16_t>(W_OUT * SA_SIZE),  // SPM row stride
        /*dram_stride=*/static_cast<uint32_t>(W_OUT * SA_SIZE),  // DRAM row stride
        /*precision=*/1,
        /*output_type=*/0,  // int8
        /*source=*/0,       // SPM
        /*is_quant=*/false,
        /*quant_zero=*/0,
        /*quant_scale=*/0,
        /*quant_shift=*/0
    );
    
    // 8. Verify results
    std::cout << "\n[Verify] Checking results..." << std::endl;
    
    stats.total = OFM_SIZE;
    stats.mismatch = 0;
    stats.max_diff = 0;
    int first_mismatch_h = -1, first_mismatch_w = -1;
    
    // Distribution histogram: how many elements have each diff value
    std::vector<int> diff_histogram(256, 0);
    
    // Per-position summary
    std::cout << "\n[Per-Position Summary]" << std::endl;
    for (int oh = 0; oh < H_OUT; oh++) {
        for (int ow = 0; ow < W_OUT; ow++) {
            int pos_mismatch = 0;
            int pos_max_diff = 0;
            for (int oc = 0; oc < C_OUT; oc++) {
                int idx = ofm_index_nchwc32(oc, oh, ow);
                int diff = std::abs(static_cast<int>(host_out[idx]) - static_cast<int>(ref_out[idx]));
                if (diff < 256) diff_histogram[diff]++;
                if (diff > 2) {
                    pos_mismatch++;
                    stats.mismatch++;
                    if (first_mismatch_h < 0) {
                        first_mismatch_h = oh;
                        first_mismatch_w = ow;
                    }
                }
                if (diff > pos_max_diff) pos_max_diff = diff;
                if (diff > stats.max_diff) stats.max_diff = diff;
            }
            std::cout << "  pos(" << oh << "," << ow << "): "
                      << pos_mismatch << "/" << C_OUT << " mismatches, max_diff=" << pos_max_diff
                      << std::endl;
        }
    }
    
    // Print diff histogram
    std::cout << "\n[Diff Histogram]" << std::endl;
    for (int d = 0; d < 256 && d <= stats.max_diff + 2; d++) {
        if (diff_histogram[d] > 0) {
            std::cout << "  diff=" << d << ": " << diff_histogram[d] << " elements";
            if (d <= 2) std::cout << " (OK)";
            else std::cout << " (MISMATCH)";
            std::cout << std::endl;
        }
    }

    // Channel dump at first mismatch position (and pos 0,0 always)
#if DBG_CHANNEL_DUMP
    {
        int32_t* hw_acc_ptr = hw_acc.empty() ? nullptr : hw_acc.data();
        channel_dump_at("Position (0,0)", 0, 0, host_out, ref_out.data(),
                        ref_acc.data(), hw_acc_ptr, ms.scale, ms.shift);
        if (first_mismatch_h >= 0 && (first_mismatch_h != 0 || first_mismatch_w != 0)) {
            channel_dump_at("First mismatch position", first_mismatch_h, first_mismatch_w,
                           host_out, ref_out.data(), ref_acc.data(), hw_acc_ptr,
                           ms.scale, ms.shift);
        }
    }
#endif
    
    // Summary
    std::cout << "\n======================================" << std::endl;
    if (stats.mismatch == 0) {
        std::cout << "  PASS: All " << stats.total << " outputs match!" << std::endl;
    } else {
        std::cout << "  FAIL: " << stats.mismatch << "/" << stats.total 
                  << " mismatches, max_diff=" << stats.max_diff << std::endl;
#if DBG_ACC_READBACK
        std::cout << "  [Hint] Check ACC int32 results above to isolate compute vs quant." << std::endl;
#endif
#if DBG_MVIN_READBACK
        std::cout << "  [Hint] Check MVIN readback results above to isolate DMA issues." << std::endl;
#endif
    }
    std::cout << "======================================" << std::endl;
    
    return stats;
}

// ==========================================
// Conv Tile Tests (npu_conv_tile_run)
// ==========================================

static int normalize_blk_to_channels(int blk) {
    if (blk <= 0) return 0;
    if (blk < SA_SIZE) return blk;            // interpret as channels (support cin<32)
    if (blk % SA_SIZE == 0) return blk;       // interpret as channels
    return blk * SA_SIZE;                     // fallback to block count
}

struct TileCaseResult {
    bool skipped = false;
    size_t mismatch = 0;
    size_t total = 0;
    int max_diff = 0;
};

static TileCaseResult run_conv_tile_case(int oh, int ow, int ic_blk, int oc_blk) {
    TileCaseResult result;
    const int K_H_d = 3, K_W_d = 3, STRIDE_d = 1, DILATION_d = 1;
    const int P_TOP_d = 0, P_LEFT_d = 0, P_BOTTOM_d = 0, P_RIGHT_d = 0;

    int C_IN_d = normalize_blk_to_channels(ic_blk);
    int C_OUT_d = normalize_blk_to_channels(oc_blk);
    int H_OUT_d = oh;
    int W_OUT_d = ow;
    // Reverse infer input shape from output shape:
    // O = floor((I + P_top + P_bottom - D*(K-1) - 1) / S + 1)
    // For exact tiling here (no floor loss), use:
    // I = (O - 1) * S - P_top - P_bottom + D*(K-1) + 1
    int H_IN_d = (H_OUT_d - 1) * STRIDE_d - P_TOP_d - P_BOTTOM_d + DILATION_d * (K_H_d - 1) + 1;
    int W_IN_d = (W_OUT_d - 1) * STRIDE_d - P_LEFT_d - P_RIGHT_d + DILATION_d * (K_W_d - 1) + 1;

    bool cin_ok = (C_IN_d < SA_SIZE) || (C_IN_d % SA_SIZE == 0);
    if (!cin_ok || C_OUT_d % SA_SIZE != 0) {
        std::cout << "[TILE] Skip OH=" << oh << " OW=" << ow
                  << " IC_blk=" << ic_blk << " OC_blk=" << oc_blk
                  << " (invalid channel alignment)" << std::endl;
        result.skipped = true;
        return result;
    }

    const int IFM_SIZE_d = H_IN_d * W_IN_d * C_IN_d;
    const int OFM_SIZE_d = H_OUT_d * W_OUT_d * C_OUT_d;
    const int cout_blocks = C_OUT_d / SA_SIZE;
    const uint32_t weight_block_size = static_cast<uint32_t>(K_H_d * K_W_d * C_IN_d * SA_SIZE);

    if (!assign_spm_layout(static_cast<uint32_t>(IFM_SIZE_d),
                           weight_block_size,
                           static_cast<uint32_t>(OFM_SIZE_d),
                           "conv_tile_case")) {
        std::cout << "[TILE] Skip OH=" << oh << " OW=" << ow
                  << " IC_blk=" << ic_blk << " OC_blk=" << oc_blk
                  << " (SPM overflow)" << std::endl;
        result.skipped = true;
        return result;
    }
    if (!check_acc_capacity(static_cast<uint32_t>(OFM_SIZE_d), "conv_tile_case")) {
        std::cout << "[TILE] Skip OH=" << oh << " OW=" << ow
                  << " IC_blk=" << ic_blk << " OC_blk=" << oc_blk
                  << " (ACC overflow)" << std::endl;
        result.skipped = true;
        return result;
    }

    int8_t* host_ifm = static_cast<int8_t*>(npu_mem_alloc(IFM_SIZE_d));
    int8_t* host_weight = static_cast<int8_t*>(npu_mem_alloc(K_H_d * K_W_d * C_IN_d * C_OUT_d));
    int8_t* host_out = static_cast<int8_t*>(npu_mem_alloc(OFM_SIZE_d));
    if (!host_ifm || !host_weight || !host_out) {
        std::cerr << "[TILE] Alloc failed OH=" << oh << " OW=" << ow
                  << " C_IN=" << C_IN_d << " C_OUT=" << C_OUT_d << std::endl;
        if (host_ifm) npu_mem_free(host_ifm);
        if (host_weight) npu_mem_free(host_weight);
        if (host_out) npu_mem_free(host_out);
        result.skipped = true;
        return result;
    }

    generate_fixed_data_dyn(host_ifm, host_weight,
                            H_IN_d, W_IN_d, C_IN_d, C_OUT_d, K_H_d, K_W_d);
    std::memset(host_out, 0, OFM_SIZE_d);

    float max_acc = 5.0f * 5.0f * static_cast<float>(C_IN_d) * static_cast<float>(K_H_d) * static_cast<float>(K_W_d);
    float Mscale = (max_acc > 0.0f) ? (127.0f / max_acc) : 0.0f;
    MulShift ms = pick_output_scale_shift(Mscale);

    std::vector<int32_t> ref_acc(OFM_SIZE_d);
    std::vector<int8_t> ref_out(OFM_SIZE_d);
    compute_reference_conv_dyn(host_ifm, host_weight,
                               H_IN_d, W_IN_d, C_IN_d, C_OUT_d,
                               K_H_d, K_W_d, STRIDE_d, DILATION_d,
                               P_TOP_d, P_LEFT_d,
                               H_OUT_d, W_OUT_d,
                               ref_acc.data(), ref_out.data(),
                               ms.scale, ms.shift);

    // MVIN IFM once
    // cin >= 32: NCHWC32 rows=(C_IN/32)*H, cols=W*32
    // cin <  32: NHWC    rows=H,          cols=W*C_IN
    uint16_t ifm_col_num = 0;
    uint16_t ifm_row_num = 0;
    uint16_t ifm_stride = 0;
    if (C_IN_d < SA_SIZE) {
        ifm_col_num = static_cast<uint16_t>(W_IN_d * C_IN_d - 1);
        ifm_row_num = static_cast<uint16_t>(H_IN_d - 1);
        ifm_stride = static_cast<uint16_t>(W_IN_d * C_IN_d);
    } else {
        int c_blocks = C_IN_d / SA_SIZE;
        ifm_col_num = static_cast<uint16_t>(W_IN_d * SA_SIZE - 1);
        ifm_row_num = static_cast<uint16_t>(c_blocks * H_IN_d - 1);
        ifm_stride = static_cast<uint16_t>(W_IN_d * SA_SIZE);
    }
    npu_dma_mvin(
        /*host_ptr=*/host_ifm,
        /*sram_addr=*/SRAM_ADDR_IFM,
        /*col_num=*/ifm_col_num,
        /*row_num=*/ifm_row_num,
        /*sram_stride=*/ifm_stride,
        /*dram_stride=*/ifm_stride,
        /*precision=*/1,
        /*input_type=*/0,  // IFM
        /*dest=*/0,        // SPM
        /*is_bias=*/false,
        /*is_quant=*/false,
        /*quant_zero=*/0,
        /*quant_scale=*/0,
        /*quant_shift=*/0
    );
    // MVOUT IFM回host验证MVIN正确性，避免后续conv计算错误不好debug
    {
        int ifm_rb_err = verify_mvin_readback("TILE-IFM", host_ifm, IFM_SIZE_d, SRAM_ADDR_IFM);
        if (ifm_rb_err != 0) {
            std::cout << "[TILE] WARNING: IFM MVIN readback mismatch=" << ifm_rb_err
                      << ", test result may be affected." << std::endl;
        }
    }

    //这里先不使用bias，认为bias是0
    // Bias (zeros) for bias_enable path (must be NPU-alloc memory)
    int32_t* bias_zero = static_cast<int32_t*>(npu_mem_alloc(SA_SIZE * sizeof(int32_t)));
    if (!bias_zero) {
        std::cerr << "[TILE] Bias alloc failed" << std::endl;
        npu_mem_free(host_ifm);
        npu_mem_free(host_weight);
        npu_mem_free(host_out);
        result.skipped = true;
        return result;
    }
    std::memset(bias_zero, 0, SA_SIZE * sizeof(int32_t));

    for (int b = 0; b < cout_blocks; b++) {

        // 每次搬运一个 cout=32 的权重块（包含该块全部 cin）到固定 SRAM_ADDR_WEIGHT
        npu_dma_mvin(
            /*host_ptr=*/host_weight + static_cast<size_t>(b) * weight_block_size,
            /*sram_addr=*/SRAM_ADDR_WEIGHT,
            /*col_num=*/static_cast<uint16_t>(weight_block_size - 1),
            /*row_num=*/0,
            /*sram_stride=*/0,
            /*dram_stride=*/0,
            /*precision=*/1,
            /*input_type=*/1,   // Weight
            /*dest=*/0,         // SPM
            /*is_bias=*/false,
            /*is_quant=*/false,
            /*quant_zero=*/0,
            /*quant_scale=*/0,
            /*quant_shift=*/0
        );
        
        // MVIN zero bias to ACC bias register
        npu_dma_mvin(
            /*host_ptr=*/bias_zero,
            /*sram_addr=*/ACC_ADDR_PSUM,
            /*col_num=*/static_cast<uint16_t>(SA_SIZE - 1),
            /*row_num=*/0,
            /*sram_stride=*/0,
            /*dram_stride=*/0,
            /*precision=*/1,
            /*input_type=*/2,  // BIAS(int32)
            /*dest=*/true,     // ACC
            /*is_bias=*/true,
            /*is_quant=*/false,
            /*quant_zero=*/0,
            /*quant_scale=*/0,
            /*quant_shift=*/0
        );

        uint32_t ofm_block_offset = static_cast<uint32_t>(b * H_OUT_d * W_OUT_d * SA_SIZE);
        npu_conv_tile_run(
            /*sram_addr_ifm=*/SRAM_ADDR_IFM,
            /*sram_addr_weight=*/SRAM_ADDR_WEIGHT,
            /*sram_addr_ofm=*/SRAM_ADDR_OFM + ofm_block_offset,
            /*acc_addr_psum=*/ACC_ADDR_PSUM,
            /*c_in=*/C_IN_d,
            /*k_h=*/K_H_d,
            /*k_w=*/K_W_d,
            /*stride=*/STRIDE_d,
            /*dilation=*/DILATION_d,
            /*i_cin=*/0,
            /*t_cout=*/SA_SIZE,
            /*t_h_out=*/H_OUT_d,
            /*t_w_out=*/W_OUT_d,
            /*t_cin=*/C_IN_d,
            /*quant_scale=*/static_cast<uint16_t>(ms.scale),
            /*quant_scaleshift=*/static_cast<uint16_t>(ms.shift),
            /*relu_enable=*/false,
            /*relu_type=*/0,
            /*bias_enable=*/true,
            /*is_group_conv=*/false
        );

        // MVOUT this cout block
        npu_dma_mvout(
            /*host_ptr=*/host_out + ofm_block_offset,
            /*sram_addr=*/SRAM_ADDR_OFM + ofm_block_offset,
            /*col_num=*/static_cast<uint16_t>(W_OUT_d * SA_SIZE - 1),
            /*row_num=*/static_cast<uint16_t>(H_OUT_d - 1),
            /*sram_stride=*/static_cast<uint16_t>(W_OUT_d * SA_SIZE),
            /*dram_stride=*/static_cast<uint32_t>(W_OUT_d * SA_SIZE),
            /*precision=*/1,
            /*output_type=*/0,  // int8
            /*source=*/false,   // SPM
            /*is_quant=*/false,
            /*quant_zero=*/0,
            /*quant_scale=*/0,
            /*quant_shift=*/0
        );
    }

    // Verify per cout block
    size_t total_mismatch = 0;
    int total_max_diff = 0;

    for (int b = 0; b < cout_blocks; b++) {
        size_t blk_mismatch = 0;
        int blk_max_diff = 0;
        int blk_offset = b * H_OUT_d * W_OUT_d * SA_SIZE;
        int blk_size = H_OUT_d * W_OUT_d * SA_SIZE;
        
        for (int i = 0; i < blk_size; i++) {
            int diff = std::abs(static_cast<int>(host_out[blk_offset + i]) - 
                               static_cast<int>(ref_out[blk_offset + i]));
            if (diff > 2) blk_mismatch++;
            if (diff > blk_max_diff) blk_max_diff = diff;
        }
        
        total_mismatch += blk_mismatch;
        if (blk_max_diff > total_max_diff) total_max_diff = blk_max_diff;
    }

    std::cout << "[TILE] OH=" << oh << " OW=" << ow
              << " IC_blk=" << ic_blk << " OC_blk=" << oc_blk
              << " mismatch=" << total_mismatch << "/" << OFM_SIZE_d
              << " max_diff=" << total_max_diff
              << ((total_mismatch == 0 && total_max_diff <= 2) ? " PASS" : " FAIL")
              << std::endl;

    result.mismatch = total_mismatch;
    result.total = static_cast<size_t>(OFM_SIZE_d);
    result.max_diff = total_max_diff;

    npu_mem_free(bias_zero);
    npu_mem_free(host_ifm);
    npu_mem_free(host_weight);
    npu_mem_free(host_out);

    return result;
}

// ==========================================
// Spatial Sweep Test - explore which H/W combos fail
// ==========================================

#if DBG_SPATIAL_SWEEP
/**
 * @brief Run a single conv test with given spatial dimensions, return mismatch info.
 *        Uses npu_mem_alloc'd buffers, 1x1 kernel, 32 cin/cout, no padding.
 *        Prints which output positions are all-zero or mismatch.
 */
static void run_spatial_probe(int h_in, int w_in) {
    const int sa = SA_SIZE;
    const int c_in = 32, c_out = 32, kh = 1, kw = 1;
    const int h_out = h_in, w_out = w_in;  // 1x1 conv, stride 1, no pad
    const int ifm_size = h_in * w_in * c_in;
    const int wgt_size = kh * kw * c_in * c_out;
    const int ofm_size = h_out * w_out * c_out;

    if (!assign_spm_layout(static_cast<uint32_t>(ifm_size),
                           static_cast<uint32_t>(wgt_size),
                           static_cast<uint32_t>(ofm_size),
                           "spatial_probe")) {
        std::cout << "  [SWEEP " << h_in << "x" << w_in << "] Skip (SPM overflow)" << std::endl;
        return;
    }
    if (!check_acc_capacity(static_cast<uint32_t>(ofm_size), "spatial_probe")) {
        std::cout << "  [SWEEP " << h_in << "x" << w_in << "] Skip (ACC overflow)" << std::endl;
        return;
    }

    int8_t* ifm = static_cast<int8_t*>(npu_mem_alloc(ifm_size));
    int8_t* wgt = static_cast<int8_t*>(npu_mem_alloc(wgt_size));
    int8_t* ofm = static_cast<int8_t*>(npu_mem_alloc(ofm_size));
    if (!ifm || !wgt || !ofm) {
        std::cerr << "[SWEEP] Alloc failed for " << h_in << "x" << w_in << std::endl;
        if (ifm) npu_mem_free(ifm);
        if (wgt) npu_mem_free(wgt);
        if (ofm) npu_mem_free(ofm);
        return;
    }

    // Fill IFM: NCHWC32 layout [H][W][32], value = (h+w+c)%5+1
    for (int c = 0; c < c_in; c++)
        for (int h = 0; h < h_in; h++)
            for (int w = 0; w < w_in; w++)
                ifm[(h * w_in + w) * sa + c] = static_cast<int8_t>((h + w + c) % 5 + 1);

    // Fill Weight: [Cin][Cout], value = (cout+cin)%5+1
    for (int cin = 0; cin < c_in; cin++)
        for (int cout = 0; cout < c_out; cout++)
            wgt[cin * c_out + cout] = static_cast<int8_t>((cout + cin) % 5 + 1);

    std::memset(ofm, 0xCC, ofm_size);  // sentinel

    // Software reference (int32)
    std::vector<int32_t> ref_acc(ofm_size);
    for (int oh = 0; oh < h_out; oh++) {
        for (int ow = 0; ow < w_out; ow++) {
            for (int oc = 0; oc < c_out; oc++) {
                int32_t sum = 0;
                for (int ic = 0; ic < c_in; ic++) {
                    sum += static_cast<int32_t>(ifm[(oh * w_in + ow) * sa + ic]) *
                           static_cast<int32_t>(wgt[ic * c_out + oc]);
                }
                ref_acc[(oh * w_out + ow) * sa + oc] = sum;
            }
        }
    }

    // Quantize reference
    float Mscale = 127.0f / 800.0f;
    MulShift ms = pick_output_scale_shift(Mscale);
    std::vector<int8_t> ref_out(ofm_size);
    for (int i = 0; i < ofm_size; i++)
        ref_out[i] = quant_i32_to_i8(ref_acc[i], ms.scale, ms.shift);

    // MVIN
    npu_dma_mvin(
        /*host_ptr=*/ifm,
        /*sram_addr=*/SRAM_ADDR_IFM,
        /*col_num=*/static_cast<uint16_t>(ifm_size - 1),
        /*row_num=*/0,
        /*sram_stride=*/0,
        /*dram_stride=*/0,
        /*precision=*/1,
        /*input_type=*/0,  // IFM
        /*dest=*/0,        // SPM
        /*is_bias=*/false,
        /*is_quant=*/false,
        /*quant_zero=*/0,
        /*quant_scale=*/0,
        /*quant_shift=*/0
    );
    npu_dma_mvin(
        /*host_ptr=*/wgt,
        /*sram_addr=*/SRAM_ADDR_WEIGHT,
        /*col_num=*/static_cast<uint16_t>(wgt_size - 1),
        /*row_num=*/0,
        /*sram_stride=*/0,
        /*dram_stride=*/0,
        /*precision=*/1,
        /*input_type=*/1,  // Weight
        /*dest=*/0,        // SPM
        /*is_bias=*/false,
        /*is_quant=*/false,
        /*quant_zero=*/0,
        /*quant_scale=*/0,
        /*quant_shift=*/0
    );

    // Conv (int8 output)
    npu_conv_run(
        /*pad_top=*/0,
        /*pad_bottom=*/0,
        /*pad_left=*/0,
        /*pad_right=*/0,
        /*pad_mode=*/0,
        /*weight_shape_m1=*/static_cast<uint8_t>(kh - 1),
        /*weight_stride_m1=*/0,
        /*weight_dilation_m1=*/0,
        /*is_group_conv=*/false,
        /*int_type=*/0,
        /*op_type=*/1,
        /*dataflow_mode=*/0,
        /*accout_dest=*/0,                           // SPM output (int8)
        /*input_a_zeropoint=*/0,
        /*input_b_zeropoint=*/0,
        /*input_a_addr=*/SRAM_ADDR_IFM,
        /*input_a_col_num_m1=*/static_cast<uint16_t>(w_in - 1),
        /*input_a_row_num_m1=*/static_cast<uint8_t>(h_in - 1),
        /*input_a_stride=*/static_cast<uint16_t>(w_in),
        /*input_b_addr=*/SRAM_ADDR_WEIGHT,
        /*input_b_col_num_m1=*/static_cast<uint8_t>(c_out - 1),
        /*input_b_row_num_m1=*/static_cast<uint16_t>(c_in - 1),
        /*input_b_stride=*/static_cast<uint16_t>(c_out),
        /*biaspsum_width=*/static_cast<uint8_t>(w_out),
        /*biaspsum_height=*/static_cast<uint8_t>(h_out),
        /*biaspsum_addr=*/0,
        /*biaspsum_stride=*/0,
        /*output_addr=*/SRAM_ADDR_OFM,
        /*output_stride=*/static_cast<uint16_t>(w_out),
        /*is_accumulate=*/false,
        /*relu_enable=*/false,
        /*relu_type=*/0,
        /*is_bias=*/false,
        /*output_zeropoint=*/0,
        /*quant_scale=*/static_cast<uint16_t>(ms.scale),
        /*quant_scaleshift=*/static_cast<uint16_t>(ms.shift)
    );

    // MVOUT
    npu_dma_mvout(
        /*host_ptr=*/ofm,
        /*sram_addr=*/SRAM_ADDR_OFM,
        /*col_num=*/static_cast<uint16_t>(w_out * sa - 1),
        /*row_num=*/static_cast<uint16_t>(h_out - 1),
        /*sram_stride=*/static_cast<uint16_t>(w_out * sa),
        /*dram_stride=*/static_cast<uint32_t>(w_out * sa),
        /*precision=*/1,
        /*output_type=*/0,  // int8
        /*source=*/false,   // SPM
        /*is_quant=*/false,
        /*quant_zero=*/0,
        /*quant_scale=*/0,
        /*quant_shift=*/0
    );

    // Check per-position
    int total_pos = h_out * w_out;
    int bad_pos = 0;
    int zero_pos = 0;
    std::cout << "  [SWEEP " << h_in << "x" << w_in << "] ";

    for (int oh = 0; oh < h_out; oh++) {
        for (int ow = 0; ow < w_out; ow++) {
            int pos_mismatch = 0;
            bool all_zero = true;
            for (int oc = 0; oc < c_out; oc++) {
                int idx = (oh * w_out + ow) * sa + oc;
                if (ofm[idx] != 0) all_zero = false;
                int diff = std::abs(static_cast<int>(ofm[idx]) - static_cast<int>(ref_out[idx]));
                if (diff > 2) pos_mismatch++;
            }
            if (all_zero && ref_acc[(oh * w_out + ow) * sa] != 0) {
                std::cout << "(" << oh << "," << ow << ")=ZERO! ";
                zero_pos++;
                bad_pos++;
            } else if (pos_mismatch > 0) {
                std::cout << "(" << oh << "," << ow << ")=ERR" << pos_mismatch << " ";
                bad_pos++;
            }
        }
    }
    if (bad_pos == 0) {
        std::cout << "ALL " << total_pos << " positions PASS";
    } else {
        std::cout << " [" << bad_pos << "/" << total_pos << " bad, " << zero_pos << " zeros]";
    }
    std::cout << std::endl;

    npu_mem_free(ifm);
    npu_mem_free(wgt);
    npu_mem_free(ofm);
}

static void run_spatial_sweep() {
    std::cout << "\n==========================================" << std::endl;
    std::cout << "  Spatial Sweep: 1x1 conv, varying H x W  " << std::endl;
    std::cout << "==========================================" << std::endl;

    // Test single-dimension variations
    struct { int h; int w; } cases[] = {
        {1, 1},  // scalar - simplest
        {1, 2},  // single row, 2 cols
        {2, 1},  // 2 rows, single col
        {1, 3},  // single row, 3 cols
        {3, 1},  // 3 rows, single col
        {2, 2},  // the failing case
        {2, 3},  // wider
        {3, 2},  // taller
        {3, 3},  // square
        {4, 4},  // larger
        {1, 4},  // single row, 4 cols
        {4, 1},  // 4 rows, single col
    };
    int num_cases = sizeof(cases) / sizeof(cases[0]);

    for (int i = 0; i < num_cases; i++) {
        // Ensure total positions fit in SA_SIZE (32) to avoid micro-tiling
        if (cases[i].h * cases[i].w > SA_SIZE) {
            std::cout << "  [SWEEP " << cases[i].h << "x" << cases[i].w 
                      << "] Skipped (exceeds SA_SIZE)" << std::endl;
            continue;
        }
        run_spatial_probe(cases[i].h, cases[i].w);
    }

    std::cout << "==========================================" << std::endl;
}
#endif

// ==========================================
// Tile Shape Tests (requested list)
// ==========================================
#if DBG_CONV_TILE_TEST
static void run_tile_shape_tests() {
    std::cout << "\n==========================================" << std::endl;
    std::cout << "  Tile Shape Tests: npu_conv_tile_run       " << std::endl;
    std::cout << "==========================================" << std::endl;

    struct TileShape { int oh; int ow; int ic_blk; int oc_blk; } cases[] = {
        {32,32, 32, 32},
        {32,8, 64, 64},
        {8,32, 64, 64},
        {8, 4, 3, 32},
        {8, 4, 3, 64},
        {16, 16, 3, 32},
        {16, 16, 3, 64},
        {14, 14, 32, 32},
        {14, 14, 32, 64},
        {14, 14, 64, 32},
        {14, 14, 64, 64},
        {8, 4, 32, 32},
        {8, 8, 32, 32},
        {8, 4, 64, 32},
        {8, 8, 32, 64},
        {8, 8, 64, 64},
        {16, 16, 64, 64},
        {8, 28, 64, 64},
        {8, 8, 64, 256},
        {8, 4, 64, 128},
        {8, 4, 64, 256},
        {8, 28, 32, 128},
        {8, 28, 64, 128},
        {16, 28, 64, 64},
        {8, 28, 64, 256},
        {14, 14, 128, 64},
        {28, 28, 128, 64},
        {32, 32, 3, 64},
        {32, 32, 64, 64},
        {16, 56, 64, 64},
        // {8, 56, 64, 128}
    };

    std::set<std::tuple<int, int, int, int>> unique_case_set;
    std::vector<TileShape> unique_cases;
    int num_cases = sizeof(cases) / sizeof(cases[0]);
    for (int i = 0; i < num_cases; i++) {
        auto key = std::make_tuple(cases[i].oh, cases[i].ow, cases[i].ic_blk, cases[i].oc_blk);
        if (unique_case_set.insert(key).second) {
            unique_cases.push_back(cases[i]);
        }
    }

    size_t total_mismatch_all_cases = 0;
    size_t total_elements_all_cases = 0;
    int executed_cases = 0;
    int skipped_cases = 0;

    for (const auto& tc : unique_cases) {
        TileCaseResult r = run_conv_tile_case(tc.oh, tc.ow, tc.ic_blk, tc.oc_blk);
        if (r.skipped) {
            skipped_cases++;
            continue;
        }
        executed_cases++;
        total_mismatch_all_cases += r.mismatch;
        total_elements_all_cases += r.total;
    }

    std::cout << "[TILE] Summary: unique_cases=" << unique_cases.size()
              << " executed=" << executed_cases
              << " skipped=" << skipped_cases
              << " total_mismatch=" << total_mismatch_all_cases
              << "/" << total_elements_all_cases
              << std::endl;

    std::cout << "==========================================" << std::endl;
}
#endif

// ==========================================
// Main
// ==========================================

int main() {
    std::cout << "===== NPU Conv Test (Runtime API) =====" << std::endl;
    
    if (npu_init() != 0) {
        std::cerr << "Error: NPU init failed. Check kernel driver." << std::endl;
        return -1;
    }
    
    npu_reset();
    
    // Allocate buffers
    int8_t* host_ifm = static_cast<int8_t*>(npu_mem_alloc(IFM_SIZE));
    int8_t* host_weight = static_cast<int8_t*>(npu_mem_alloc(WEIGHT_SIZE));
    int8_t* host_out = static_cast<int8_t*>(npu_mem_alloc(OFM_SIZE));
    
    if (!host_ifm || !host_weight || !host_out) {
        std::cerr << "Error: NPU memory alloc failed." << std::endl;
        npu_destroy();
        return -1;
    }
    
    // Run test
    CaseStats stats = run_conv_test(host_ifm, host_weight, host_out);
    
    // Run spatial sweep to find failure pattern
#if DBG_SPATIAL_SWEEP
    run_spatial_sweep();
#endif

#if DBG_CONV_TILE_TEST
    run_tile_shape_tests();
#endif
    
    // Cleanup
    npu_mem_free(host_ifm);
    npu_mem_free(host_weight);
    npu_mem_free(host_out);
    npu_destroy();
    
    return (stats.skipped || stats.mismatch == 0) ? 0 : 1;
}
