#include "../npu_runtime.h"
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <cstdint>

// 测试配置
#define TEST_SIZE (8 * 1024)      // 超过4K页面大小，用于测试边界问题
#define SRAM_ADDR 0x0             // SRAM中的起始地址
#define NUM_ROWS 64*1             // 64行
#define NUM_COLS 64               // 64列 (512 bytes per row if int8)

int main() {
    std::cout << "===== NPU Runtime Test (4K Boundary Test) =====" << std::endl;
    
    // 初始化NPU运行时
    std::cout << "1. Initializing NPU runtime..." << std::endl;
    npu_init();

    std::cout << "   NPU initialized successfully." << std::endl;

    // 分配大于4K的内存（用于测试跨页面的DMA）
    std::cout << "\n2. Allocating " << TEST_SIZE << " bytes of memory..." << std::endl;
    void* host_buffer = npu_mem_alloc(TEST_SIZE);
    if (!host_buffer) {
        std::cerr << "Failed to allocate memory!" << std::endl;
        npu_destroy();
        return -1;
    }
    std::cout << "   Memory allocated at: " << host_buffer << std::endl;

    // // 初始化源数据（MVIN测试）
    // std::cout << "\n3. Initializing test data..." << std::endl;
    // uint8_t* data = static_cast<uint8_t*>(host_buffer);
    // for (int i = 0; i < TEST_SIZE; i++) {
    //     data[i] = (uint8_t)(i & 0xFF);  // 填充0-255循环的字节
    // }
    // std::cout << "   Test data initialized (first 16 bytes: ";
    // for (int i = 0; i < 16; i++) {
    //     std::cout << std::hex << (int)data[i] << " ";
    // }
    // std::cout << std::dec << ")" << std::endl;

    // // 执行MVIN操作（主机内存 -> SRAM）
    // std::cout << "\n4. Executing MVIN (Host -> SRAM)..." << std::endl;
    // std::cout << "   Config:" << std::endl;
    // std::cout << "   - Host buffer: " << host_buffer << std::endl;
    // std::cout << "   - SRAM address: 0x" << std::hex << SRAM_ADDR << std::dec << std::endl;
    // std::cout << "   - Rows: " << NUM_ROWS << ", Cols: " << NUM_COLS << std::endl;
    // std::cout << "   - Total size: " << TEST_SIZE << " bytes" << std::endl;
    
    // npu_dma_mvin(
    //     /*host_ptr=*/host_buffer,
    //     /*sram_addr=*/SRAM_ADDR,
    //     /*col_num=*/(NUM_COLS - 1),
    //     /*row_num=*/(NUM_ROWS - 1),
    //     /*sram_stride=*/NUM_COLS,
    //     /*dram_stride=*/NUM_COLS,
    //     /*precision=*/1,
    //     /*input_type=*/0,
    //     /*dest=*/0,
    //     /*is_bias=*/false,
    //     /*is_quant=*/false,
    //     /*quant_zero=*/0,
    //     /*quant_scale=*/0,
    //     /*quant_shift=*/0
    // );
    // std::cout << "   MVIN completed." << std::endl;

    // // 等待一下，确保MVIN完成
    // std::cout << "\n5. Waiting for MVIN to complete..." << std::endl;
    

    // // 清空原始缓冲区用于MVOUT测试
    // std::cout << "\n6. Clearing host buffer for MVOUT verification..." << std::endl;
    // memset(host_buffer, 0, TEST_SIZE);
    // std::cout << "   Buffer cleared." << std::endl;

    // // 执行MVOUT操作（SRAM -> 主机内存）
    // std::cout << "\n7. Executing MVOUT (SRAM -> Host)..." << std::endl;
    // std::cout << "   Config:" << std::endl;
    // std::cout << "   - Host buffer: " << host_buffer << std::endl;
    // std::cout << "   - SRAM address: 0x" << std::hex << SRAM_ADDR << std::dec << std::endl;
    // std::cout << "   - Rows: " << NUM_ROWS << ", Cols: " << NUM_COLS << std::endl;
    // std::cout << "   - Total size: " << TEST_SIZE << " bytes" << std::endl;
    
    // npu_dma_mvout(
    //     /*host_ptr=*/host_buffer,
    //     /*sram_addr=*/SRAM_ADDR,
    //     /*col_num=*/(NUM_COLS - 1),
    //     /*row_num=*/(NUM_ROWS - 1),
    //     /*sram_stride=*/NUM_COLS,
    //     /*dram_stride=*/NUM_COLS,
    //     /*precision=*/1,
    //     /*output_type=*/0,
    //     /*source=*/0,
    //     /*is_quant=*/false,
    //     /*quant_zero=*/0,
    //     /*quant_scale=*/0,
    //     /*quant_shift=*/0
    // );
    // std::cout << "   MVOUT completed." << std::endl;

    // // 等待MVOUT完成
    // std::cout << "\n8. Waiting for MVOUT to complete..." << std::endl;
    

    // // 验证数据完整性
    // std::cout << "\n9. Verifying data integrity..." << std::endl;
    // bool data_valid = true;
    // for (int i = 0; i < TEST_SIZE; i++) {
    //     uint8_t expected = (uint8_t)(i & 0xFF);
    //     if (data[i] != expected) {
    //         std::cerr << "   Data mismatch at offset " << i 
    //                   << ": expected 0x" << std::hex << (int)expected 
    //                   << ", got 0x" << (int)data[i] << std::dec << std::endl;
    //         data_valid = false;
    //     }
    // }
    
    // if (data_valid) {
    //     std::cout << "   ✓ Data verification PASSED" << std::endl;
    // } else {
    //     std::cerr << "   ✗ Data verification FAILED" << std::endl;
    // }

    // // 显示验证数据（前16字节和最后16字节）
    // std::cout << "\n10. Data samples:" << std::endl;
    // std::cout << "    First 16 bytes: ";
    // for (int i = 0; i < 16; i++) {
    //     std::cout << std::hex << (int)data[i] << " ";
    // }
    // std::cout << std::dec << std::endl;
    
    // std::cout << "    Last 16 bytes: ";
    // for (int i = TEST_SIZE - 16; i < TEST_SIZE; i++) {
    //     std::cout << std::hex << (int)data[i] << " ";
    // }
    // std::cout << std::dec << std::endl;

    // // 小数据搬运测试：每次MVIN/MVOUT一个int8，MVOUT的DRAM地址连续递增
    // std::cout << "\n11. Byte-by-byte DMA test (int8, sequential DRAM addresses)..." << std::endl;
    // const int SMALL_TEST_SIZE = 128;
    // uint8_t* small_src = static_cast<uint8_t*>(npu_mem_alloc(SMALL_TEST_SIZE));
    // uint8_t* small_dst = static_cast<uint8_t*>(npu_mem_alloc(SMALL_TEST_SIZE));
    // if (!small_src || !small_dst) {
    //     std::cerr << "Failed to allocate small test buffers!" << std::endl;
    //     if (small_src) npu_mem_free(small_src);
    //     if (small_dst) npu_mem_free(small_dst);
    //     npu_mem_free(host_buffer);
    //     npu_destroy();
    //     return -1;
    // }

    // for (int i = 0; i < SMALL_TEST_SIZE; i++) {
    //     small_src[i] = static_cast<uint8_t>((i) & 0xFF);
    //     small_dst[i] = 0;
    // }

    // for (int i = 0; i < SMALL_TEST_SIZE; i++) {
    //     // 每次搬运1字节（row=1, col=1）
    //     npu_dma_mvin(
    //         /*host_ptr=*/&small_src[i],
    //         /*sram_addr=*/SRAM_ADDR,
    //         /*col_num=*/0,
    //         /*row_num=*/0,
    //         /*sram_stride=*/1,
    //         /*dram_stride=*/1,
    //         /*precision=*/1,
    //         /*input_type=*/0,
    //         /*dest=*/0,
    //         /*is_bias=*/false,
    //         /*is_quant=*/false,
    //         /*quant_zero=*/0,
    //         /*quant_scale=*/0,
    //         /*quant_shift=*/0
    //     );

    //     npu_dma_mvout(
    //         /*host_ptr=*/&small_dst[i],
    //         /*sram_addr=*/SRAM_ADDR,
    //         /*col_num=*/0,
    //         /*row_num=*/0,
    //         /*sram_stride=*/1,
    //         /*dram_stride=*/1,
    //         /*precision=*/1,
    //         /*output_type=*/0,
    //         /*source=*/0,
    //         /*is_quant=*/false,
    //         /*quant_zero=*/0,
    //         /*quant_scale=*/0,
    //         /*quant_shift=*/0
    //     );
    //     std::cout << "   Transferred byte " << i << ": 0x" 
    //               << std::hex << (int)small_src[i] <<" got 0x" << (int)small_dst[i] << std::dec << std::endl;
    // }

    // bool small_valid = true;
    // for (int i = 0; i < SMALL_TEST_SIZE; i++) {
    //     if (small_dst[i] != small_src[i]) {
    //         std::cerr << "   Byte test mismatch at offset " << i
    //                   << ": expected 0x" << std::hex << (int)small_src[i]
    //                   << ", got 0x" << (int)small_dst[i] << std::dec << std::endl;
    //         small_valid = false;
    //     }
    // }

    // if (small_valid) {
    //     std::cout << "   ✓ Byte-by-byte test PASSED" << std::endl;
    // } else {
    //     std::cerr << "   ✗ Byte-by-byte test FAILED" << std::endl;
    // }

    // npu_mem_free(small_src);
    // npu_mem_free(small_dst);

    // 线性访存模式测试：复现Level2读回模式
    std::cout << "\n12. Linear DMA test (match Level2 M/N/K + stride)..." << std::endl;
    const int M = 4;
    const int N = 1;
    const int K = 1;
    const int A_SIZE = M * K;
    const int B_SIZE = K * N;
    const uint32_t SRAM_ADDR_A = SRAM_ADDR;
    const uint32_t SRAM_ADDR_B = SRAM_ADDR + 0x2000;

    uint8_t* linear_a_src = static_cast<uint8_t*>(npu_mem_alloc(A_SIZE));
    uint8_t* linear_a_dst = static_cast<uint8_t*>(npu_mem_alloc(A_SIZE));
    uint8_t* linear_b_src = static_cast<uint8_t*>(npu_mem_alloc(B_SIZE));
    uint8_t* linear_b_dst = static_cast<uint8_t*>(npu_mem_alloc(B_SIZE));

    if (!linear_a_src || !linear_a_dst || !linear_b_src || !linear_b_dst) {
        std::cerr << "Failed to allocate linear test buffers!" << std::endl;
        if (linear_a_src) npu_mem_free(linear_a_src);
        if (linear_a_dst) npu_mem_free(linear_a_dst);
        if (linear_b_src) npu_mem_free(linear_b_src);
        if (linear_b_dst) npu_mem_free(linear_b_dst);
        npu_mem_free(host_buffer);
        npu_destroy();
        return -1;
    }

    for (int i = 0; i < A_SIZE; i++) {
        linear_a_src[i] = static_cast<uint8_t>((i * 3) & 0xFF);
        linear_a_dst[i] = 0;
    }
    for (int i = 0; i < B_SIZE; i++) {
        linear_b_src[i] = static_cast<uint8_t>(0x5A + i);
        linear_b_dst[i] = 0;
    }

    // MVIN A: col=K-1, row=M-1, sram_stride=K, dram_stride=0
    npu_dma_mvin(
        /*host_ptr=*/linear_a_src,
        /*sram_addr=*/SRAM_ADDR_A,
        /*col_num=*/(K - 1),
        /*row_num=*/(M - 1),
        /*sram_stride=*/K,
        /*dram_stride=*/K,
        /*precision=*/1,
        /*input_type=*/0,
        /*dest=*/0,
        /*is_bias=*/false,
        /*is_quant=*/false,
        /*quant_zero=*/0,
        /*quant_scale=*/0,
        /*quant_shift=*/0
    );

    // MVIN B: col=N-1, row=K-1, sram_stride=N, dram_stride=0
    npu_dma_mvin(
        /*host_ptr=*/linear_b_src,
        /*sram_addr=*/SRAM_ADDR_B,
        /*col_num=*/(N - 1),
        /*row_num=*/(K - 1),
        /*sram_stride=*/N,
        /*dram_stride=*/0,
        /*precision=*/1,
        /*input_type=*/0,
        /*dest=*/0,
        /*is_bias=*/false,
        /*is_quant=*/false,
        /*quant_zero=*/0,
        /*quant_scale=*/0,
        /*quant_shift=*/0
    );

    // MVOUT A (readback): col=A_SIZE-1, row=0
    npu_dma_mvout(
        /*host_ptr=*/linear_a_dst,
        /*sram_addr=*/SRAM_ADDR_A,
        /*col_num=*/(A_SIZE - 1),
        /*row_num=*/0,
        /*sram_stride=*/0,
        /*dram_stride=*/0,
        /*precision=*/1,
        /*output_type=*/0,
        /*source=*/0,
        /*is_quant=*/false,
        /*quant_zero=*/0,
        /*quant_scale=*/0,
        /*quant_shift=*/0
    );

    // MVOUT B (readback): col=B_SIZE-1, row=0
    npu_dma_mvout(
        /*host_ptr=*/linear_b_dst,
        /*sram_addr=*/SRAM_ADDR_B,
        /*col_num=*/(B_SIZE - 1),
        /*row_num=*/0,
        /*sram_stride=*/0,
        /*dram_stride=*/0,
        /*precision=*/1,
        /*output_type=*/0,
        /*source=*/0,
        /*is_quant=*/false,
        /*quant_zero=*/0,
        /*quant_scale=*/0,
        /*quant_shift=*/0
    );

    bool linear_valid = true;
    for (int i = 0; i < A_SIZE; i++) {
        std::cout << "   A byte " << i << ": 0x" 
                  << std::hex << (int)linear_a_src[i] <<" got 0x" << (int)linear_a_dst[i] << std::dec << std::endl;
        if (linear_a_dst[i] != linear_a_src[i]) {
            std::cerr << "   Linear A mismatch at offset " << i
                      << ": expected 0x" << std::hex << (int)linear_a_src[i]
                      << ", got 0x" << (int)linear_a_dst[i] << std::dec << std::endl;
            linear_valid = false;
            break;
        }
    }
    for (int i = 0; i < B_SIZE; i++) {
        std::cout << "   B byte " << i << ": 0x" << std::hex << (int)linear_b_src[i]
                  << " got 0x" << (int)linear_b_dst[i] << std::dec << std::endl;
        if (linear_b_dst[i] != linear_b_src[i]) {
            std::cerr << "   Linear B mismatch at offset " << i
                      << ": expected 0x" << std::hex << (int)linear_b_src[i]
                      << ", got 0x" << (int)linear_b_dst[i] << std::dec << std::endl;
            linear_valid = false;
            break;
        }
    }

    if (linear_valid) {
        std::cout << "   ✓ Linear DMA test PASSED" << std::endl;
    } else {
        std::cerr << "   ✗ Linear DMA test FAILED" << std::endl;
    }

    npu_mem_free(linear_a_src);
    npu_mem_free(linear_a_dst);
    npu_mem_free(linear_b_src);
    npu_mem_free(linear_b_dst);

    // SPM / ACC 容量探测测试：通过地址别名(alias)定位最小回绕边界
    std::cout << "\n13. Capacity probe test for SPM/ACC..." << std::endl;
    uint8_t* probe_wr = static_cast<uint8_t*>(npu_mem_alloc(64));
    uint8_t* probe_rd = static_cast<uint8_t*>(npu_mem_alloc(64));
    uint8_t* probe_acc_zero = static_cast<uint8_t*>(npu_mem_alloc(64));
    uint8_t* probe_acc_out = static_cast<uint8_t*>(npu_mem_alloc(64));
    if (!probe_wr || !probe_rd || !probe_acc_zero || !probe_acc_out) {
        std::cerr << "   Failed to allocate probe buffers!" << std::endl;
        if (probe_wr) npu_mem_free(probe_wr);
        if (probe_rd) npu_mem_free(probe_rd);
        if (probe_acc_zero) npu_mem_free(probe_acc_zero);
        if (probe_acc_out) npu_mem_free(probe_acc_out);
        npu_mem_free(host_buffer);
        npu_destroy();
        return -1;
    }

    auto write_spm_u8 = [&](uint32_t addr, uint8_t value) {
        probe_wr[0] = value;
        npu_dma_mvin(
            /*host_ptr=*/probe_wr,
            /*sram_addr=*/addr,
            /*col_num=*/0,
            /*row_num=*/0,
            /*sram_stride=*/1,
            /*dram_stride=*/1,
            /*precision=*/1,
            /*input_type=*/0,
            /*dest=*/0,
            /*is_bias=*/false,
            /*is_quant=*/false,
            /*quant_zero=*/0,
            /*quant_scale=*/0,
            /*quant_shift=*/0
        );
    };

    auto read_spm_u8 = [&](uint32_t addr) -> uint8_t {
        probe_rd[0] = 0;
        npu_dma_mvout(
            /*host_ptr=*/probe_rd,
            /*sram_addr=*/addr,
            /*col_num=*/0,
            /*row_num=*/0,
            /*sram_stride=*/1,
            /*dram_stride=*/1,
            /*precision=*/1,
            /*output_type=*/0,
            /*source=*/0,
            /*is_quant=*/false,
            /*quant_zero=*/0,
            /*quant_scale=*/0,
            /*quant_shift=*/0
        );
        return probe_rd[0];
    };

    auto write_acc_i8 = [&](uint32_t addr, int8_t value) {
        probe_wr[0] = static_cast<uint8_t>(value);
        npu_dma_mvin(
            /*host_ptr=*/probe_wr,
            /*sram_addr=*/addr,
            /*col_num=*/0,
            /*row_num=*/0,
            /*sram_stride=*/0,
            /*dram_stride=*/0,
            /*precision=*/1,
            /*input_type=*/0,
            /*dest=*/1,
            /*is_bias=*/false,
            /*is_quant=*/true,
            /*quant_zero=*/0,
            /*quant_scale=*/1,
            /*quant_shift=*/0
        );
    };

    // 参考 matadd_test：通过 MATADD 读取 ACC（A + 0 -> SPM），再从 SPM MVOUT
    // 这样可避开 ACC 直接 MVOUT 路径的歧义，更贴近真实计算路径。
    constexpr uint32_t ACC_ZERO_ADDR = 0x2000;
    constexpr uint32_t SPM_MATADD_OUT_ADDR = 0x3000;
    auto read_acc_i8_via_matadd = [&](uint32_t addr) -> int8_t {
        probe_acc_zero[0] = 0;
        probe_acc_out[0] = 0;

        // 把 0 写入 ACC_ZERO_ADDR
        npu_dma_mvin(
            /*host_ptr=*/probe_acc_zero,
            /*sram_addr=*/ACC_ZERO_ADDR,
            /*col_num=*/0,
            /*row_num=*/0,
            /*sram_stride=*/0,
            /*dram_stride=*/0,
            /*precision=*/1,
            /*input_type=*/0,
            /*dest=*/1,
            /*is_bias=*/false,
            /*is_quant=*/true,
            /*quant_zero=*/0,
            /*quant_scale=*/1,
            /*quant_shift=*/0
        );

        // ACC(addr) + ACC(0) -> SPM，1x1
        npu_matadd_run(
            /*input_a_addr=*/addr,
            /*input_b_addr=*/ACC_ZERO_ADDR,
            /*output_addr=*/SPM_MATADD_OUT_ADDR,
            /*col_num=*/1,
            /*row_num=*/1,
            /*output_zeropoint=*/0,
            /*output_scale=*/1,
            /*output_scaleshift=*/0
        );

        // 从 SPM 读回 1byte
        npu_dma_mvout(
            /*host_ptr=*/probe_acc_out,
            /*sram_addr=*/SPM_MATADD_OUT_ADDR,
            /*col_num=*/0,
            /*row_num=*/0,
            /*sram_stride=*/0,
            /*dram_stride=*/0,
            /*precision=*/1,
            /*output_type=*/0,
            /*source=*/0,
            /*is_quant=*/false,
            /*quant_zero=*/0,
            /*quant_scale=*/0,
            /*quant_shift=*/0
        );
        return static_cast<int8_t>(probe_acc_out[0]);
    };

    constexpr uint32_t PROBE_LIMIT = (1u << 24); // 最多探测到 16MB
    uint32_t spm_capacity = 0;
    uint32_t acc_capacity = 0;

    for (uint32_t off = 1; off <= PROBE_LIMIT; off <<= 1) {
        const uint8_t base_mark = 0x11;
        const uint8_t probe_mark = 0xA5;
        write_spm_u8(0, base_mark);
        write_spm_u8(off, probe_mark);
        if (read_spm_u8(0) == probe_mark) {
            spm_capacity = off;
            break;
        }
    }

    // ACC 为 INT32 存储，地址按 4-byte 对齐探测，且使用 MATADD 路径读回
    for (uint32_t off = 4; off <= PROBE_LIMIT; off <<= 1) {
        const int8_t base_mark = 17;
        const int8_t probe_mark = 93;
        write_acc_i8(0, base_mark);
        write_acc_i8(off, probe_mark);
        if (read_acc_i8_via_matadd(0) == probe_mark) {
            acc_capacity = off;
            break;
        }
    }

    bool capacity_valid = true;

    if (spm_capacity == 0) {
        std::cerr << "   ✗ SPM capacity probe FAILED (no alias within " << PROBE_LIMIT << " bytes)" << std::endl;
        capacity_valid = false;
    } else {
        std::cout << "   SPM detected capacity: " << spm_capacity
                  << " bytes (" << (spm_capacity / 1024) << " KB)" << std::endl;

        const uint32_t spm_last_addr = spm_capacity - 1;
        write_spm_u8(spm_last_addr, 0x3C);
        uint8_t last_val = read_spm_u8(spm_last_addr);
        write_spm_u8(0, 0x12);
        write_spm_u8(spm_capacity, 0x7E); // 越界1字节，预期回绕到0地址
        uint8_t wrapped_val = read_spm_u8(0);

        if (last_val == 0x3C && wrapped_val == 0x7E) {
            std::cout << "   ✓ SPM boundary check PASSED" << std::endl;
        } else {
            std::cerr << "   ✗ SPM boundary check FAILED"
                      << " (last=0x" << std::hex << static_cast<int>(last_val)
                      << ", wrap0=0x" << static_cast<int>(wrapped_val) << std::dec << ")" << std::endl;
            capacity_valid = false;
        }
    }

    if (acc_capacity == 0) {
        std::cerr << "   ✗ ACC capacity probe FAILED (no alias within " << PROBE_LIMIT << " bytes)" << std::endl;
        capacity_valid = false;
    } else {
        std::cout << "   ACC detected capacity: " << acc_capacity
                  << " bytes (" << (acc_capacity / 1024) << " KB)" << std::endl;

        const uint32_t acc_last_aligned_addr = (acc_capacity >= 4) ? (acc_capacity - 4) : 0;
        write_acc_i8(acc_last_aligned_addr, 31);
        int8_t acc_last_val = read_acc_i8_via_matadd(acc_last_aligned_addr);
        write_acc_i8(0, 22);
        write_acc_i8(acc_capacity, 77); // 越界，预期回绕到0地址
        int8_t acc_wrapped_val = read_acc_i8_via_matadd(0);

        if (acc_last_val == 31 && acc_wrapped_val == 77) {
            std::cout << "   ✓ ACC boundary check PASSED" << std::endl;
        } else {
            std::cerr << "   ✗ ACC boundary check FAILED"
                      << " (last=" << acc_last_val
                      << ", wrap0=" << acc_wrapped_val << ")" << std::endl;
            capacity_valid = false;
        }
    }

    // 进一步按“期望配置”做定点验证
    constexpr uint32_t EXPECT_SPM_BYTES = 524288; // 512KB
    constexpr uint32_t EXPECT_ACC_BYTES = 262144; // 256KB (byte address)

    std::cout << "   --- Expected-capacity spot checks ---" << std::endl;

    // 一次性清零 SPM，再一次性读回后在 host 侧比较
    uint8_t* spm_zero_buf = static_cast<uint8_t*>(npu_mem_alloc(EXPECT_SPM_BYTES));
    uint8_t* spm_readback_buf = static_cast<uint8_t*>(npu_mem_alloc(EXPECT_SPM_BYTES));
    if (!spm_zero_buf || !spm_readback_buf) {
        std::cerr << "   ✗ Failed to allocate expected-capacity SPM buffers" << std::endl;
        if (spm_zero_buf) npu_mem_free(spm_zero_buf);
        if (spm_readback_buf) npu_mem_free(spm_readback_buf);
        capacity_valid = false;
    } else {
        std::memset(spm_zero_buf, 0, EXPECT_SPM_BYTES);
        std::memset(spm_readback_buf, 0, EXPECT_SPM_BYTES);

        constexpr uint32_t BULK_COLS = 256;
        constexpr uint32_t BULK_ROWS = EXPECT_SPM_BYTES / BULK_COLS;

        // Host -> SPM: 512KB 全量写 0
        npu_dma_mvin(
            /*host_ptr=*/spm_zero_buf,
            /*sram_addr=*/0,
            /*col_num=*/(BULK_COLS - 1),
            /*row_num=*/(BULK_ROWS - 1),
            /*sram_stride=*/BULK_COLS,
            /*dram_stride=*/BULK_COLS,
            /*precision=*/1,
            /*input_type=*/0,
            /*dest=*/0,
            /*is_bias=*/false,
            /*is_quant=*/false,
            /*quant_zero=*/0,
            /*quant_scale=*/0,
            /*quant_shift=*/0
        );

        // 写几个关键点（包含越界回绕点）
        write_spm_u8(EXPECT_SPM_BYTES - 1, 0x66);
        write_spm_u8(EXPECT_SPM_BYTES, 0x77); // 预期回绕到 0x0
        write_spm_u8(EXPECT_SPM_BYTES / 2 - 1, 0x23);
        write_spm_u8(EXPECT_SPM_BYTES / 2, 0x45);

        // SPM -> Host: 512KB 一次性读回
        npu_dma_mvout(
            /*host_ptr=*/spm_readback_buf,
            /*sram_addr=*/0,
            /*col_num=*/(BULK_COLS - 1),
            /*row_num=*/(BULK_ROWS - 1),
            /*sram_stride=*/BULK_COLS,
            /*dram_stride=*/BULK_COLS,
            /*precision=*/1,
            /*output_type=*/0,
            /*source=*/0,
            /*is_quant=*/false,
            /*quant_zero=*/0,
            /*quant_scale=*/0,
            /*quant_shift=*/0
        );

        // 生成期望并比较
        spm_zero_buf[0] = 0x77;                          // EXPECT_SPM_BYTES 回绕到 0
        spm_zero_buf[EXPECT_SPM_BYTES - 1] = 0x66;
        spm_zero_buf[EXPECT_SPM_BYTES / 2 - 1] = 0x23;
        spm_zero_buf[EXPECT_SPM_BYTES / 2] = 0x45;

        uint32_t mismatch_cnt = 0;
        for (uint32_t i = 0; i < EXPECT_SPM_BYTES; i++) {
            if (spm_readback_buf[i] != spm_zero_buf[i]) {
                if (mismatch_cnt < 16) {
                    std::cout << "   SPM mismatch @0x" << std::hex << i
                              << ": expect 0x" << static_cast<int>(spm_zero_buf[i])
                              << " got 0x" << static_cast<int>(spm_readback_buf[i])
                              << std::dec << std::endl;
                }
                mismatch_cnt++;
            }
        }

        if (mismatch_cnt == 0) {
            std::cout << "   ✓ SPM bulk zero/write/readback check PASSED" << std::endl;
        } else {
            std::cerr << "   ✗ SPM bulk zero/write/readback check FAILED"
                      << " (mismatch=" << mismatch_cnt << ")" << std::endl;
            capacity_valid = false;
        }

        npu_mem_free(spm_zero_buf);
        npu_mem_free(spm_readback_buf);
    }


    // SPM 期望边界：0x7FFFF 有效，0x80000 回绕到 
    // write_spm_u8(EXPECT_SPM_BYTES - 1, 0x67);
    // uint8_t spm_expected_last = read_spm_u8(EXPECT_SPM_BYTES - 1);
    // printf("   SPM expected last byte check at 0x%X: wrote 0x67, read back 0x%X\n", EXPECT_SPM_BYTES - 1, spm_expected_last);
    // write_spm_u8(0, 0x23);
    // uint8_t spm_expected_0 = read_spm_u8(0);
    // printf("   SPM expected wrap byte check at 0x%X: wrote 0x23, read back 0x%X\n", EXPECT_SPM_BYTES, spm_expected_0);
    // write_spm_u8(EXPECT_SPM_BYTES, 0x78);
    // uint8_t spm_expected_wrap = read_spm_u8(EXPECT_SPM_BYTES);
    // uint8_t spm_expected_wrap_val = read_spm_u8(0);
    // printf("   SPM expected wrap byte check at 0x%X: wrote 0x78,read back at 0x%X got 0x%X\n", EXPECT_SPM_BYTES, EXPECT_SPM_BYTES, spm_expected_wrap);
    // printf("   SPM expected wrap byte check at 0x%X: wrote 0x78, read back at 0x%X got 0x%X\n", EXPECT_SPM_BYTES, 0, spm_expected_wrap_val);

    // write_spm_u8(EXPECT_SPM_BYTES +1, 0x23);
    // uint8_t spm_expected_wrap_plus = read_spm_u8(EXPECT_SPM_BYTES +1);
    // printf("   SPM expected wrap byte check at 0x%X: wrote 0x23, read back at 0x%X got 0x%X\n", EXPECT_SPM_BYTES +1, EXPECT_SPM_BYTES +1, spm_expected_wrap_plus);
    // // write_spm_u8(EXPECT_SPM_BYTES - 1, 0x66);
    //


    // uint8_t spm_expected_last = read_spm_u8(EXPECT_SPM_BYTES - 1);
    // write_spm_u8(0, 0x21);
    // write_spm_u8(EXPECT_SPM_BYTES, 0x77);
    // uint8_t spm_expected_wrap = read_spm_u8(0);
    // if (spm_expected_last == 0x66 && spm_expected_wrap == 0x77) {
    //     std::cout << "   ✓ SPM expected boundary (512KB) PASSED" << std::endl;
    // } else {
    //     std::cerr << "   ✗ SPM expected boundary (512KB) FAILED"
    //               << " (last=0x" << std::hex << static_cast<int>(spm_expected_last)
    //               << ", wrap0=0x" << static_cast<int>(spm_expected_wrap) << std::dec << ")" << std::endl;
    //     capacity_valid = false;
    // }

    // // 关键反例检查：若真实容量更小（如 256KB），0x40000 会提前回绕到 0，必须判失败
    // write_spm_u8(0, 0x31);
    // write_spm_u8(EXPECT_SPM_BYTES / 2, 0x32); // 256KB
    // uint8_t spm_half_alias = read_spm_u8(0);
    // if (spm_half_alias == 0x31) {
    //     std::cout << "   ✓ SPM anti-alias check PASSED (0x40000 does not wrap to 0)" << std::endl;
    // } else {
    //     std::cerr << "   ✗ SPM anti-alias check FAILED (0x40000 already aliases 0)" << std::endl;
    //     capacity_valid = false;
    // }

    // // ACC 期望边界：最后一个 int32 起始地址 0x3FFFC，有效；0x40000 回绕到 0
    // const uint32_t acc_expected_last = EXPECT_ACC_BYTES - 4;
    // write_acc_i8(acc_expected_last, 44);
    // int8_t acc_expected_last_val = read_acc_i8_via_matadd(acc_expected_last);
    // write_acc_i8(0, 12);
    // write_acc_i8(EXPECT_ACC_BYTES, 55);
    // int8_t acc_expected_wrap_val = read_acc_i8_via_matadd(0);
    // if (acc_expected_last_val == 44 && acc_expected_wrap_val == 55) {
    //     std::cout << "   ✓ ACC expected boundary (256KB) PASSED" << std::endl;
    // } else {
    //     std::cerr << "   ✗ ACC expected boundary (256KB) FAILED"
    //               << " (last=" << static_cast<int>(acc_expected_last_val)
    //               << ", wrap0=" << static_cast<int>(acc_expected_wrap_val) << ")" << std::endl;
    //     capacity_valid = false;
    // }

    // // ACC 反例检查：若真实容量更小（如 64KB），0x10000 会提前回绕到 0
    // write_acc_i8(0, 21);
    // write_acc_i8(EXPECT_ACC_BYTES / 4, 22); // 64KB
    // int8_t acc_quarter_alias = read_acc_i8_via_matadd(0);
    // if (acc_quarter_alias == 21) {
    //     std::cout << "   ✓ ACC anti-alias check PASSED (0x10000 does not wrap to 0)" << std::endl;
    // } else {
    //     std::cerr << "   ✗ ACC anti-alias check FAILED (0x10000 already aliases 0)" << std::endl;
    //     capacity_valid = false;
    // }

    // if (capacity_valid) {
    //     std::cout << "   ✓ Capacity probe PASSED" << std::endl;
    // } else {
    //     std::cerr << "   ✗ Capacity probe FAILED" << std::endl;
    // }

    npu_mem_free(probe_wr);
    npu_mem_free(probe_rd);
    npu_mem_free(probe_acc_zero);
    npu_mem_free(probe_acc_out);

    // 清理资源
    std::cout << "\n14. Cleanup..." << std::endl;
    npu_mem_free(host_buffer);
    npu_destroy();
    
    // std::cout << "\n===== Test Completed =====" << std::endl;
    // std::cout << ((data_valid && small_valid) ? "RESULT: PASS" : "RESULT: FAIL") << std::endl;
    
    return 0;
}
