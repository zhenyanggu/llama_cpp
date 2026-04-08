#include "../npu_runtime.h"
#include <cstdint>
#include <cstring>
#include <iostream>

int main() {
    std::cout << "===== DMA 32-bit row/col Regression Test =====" << std::endl;
    npu_init();
    constexpr uint32_t SPM_LIMIT_BYTES = 512u * 1024u; // 512KB
    constexpr uint32_t DIM32_ELEMS = 70001u;           // > 65535

    bool all_passed = true;

        // Case A: large col_num, single row
        {
            const uint32_t bytes = DIM32_ELEMS;
            if (bytes > SPM_LIMIT_BYTES) {
                std::cerr << "Case A skipped: payload exceeds 512KB" << std::endl;
                all_passed = false;
            } else {
                uint8_t* src = static_cast<uint8_t*>(npu_mem_alloc(bytes));
                uint8_t* dst = static_cast<uint8_t*>(npu_mem_alloc(bytes));

                if (!src || !dst) {
                    std::cerr << "Case A alloc failed" << std::endl;
                    if (src) npu_mem_free(src);
                    if (dst) npu_mem_free(dst);
                    npu_destroy();
                    return -1;
                }

                for (uint32_t i = 0; i < bytes; ++i) {
                    src[i] = static_cast<uint8_t>((i * 13u + 7u) & 0xFFu);
                }
                std::memset(dst, 0, bytes);

                npu_dma_mvin(
                    /*host_ptr=*/src,
                    /*sram_addr=*/0,
                    /*col_num=*/DIM32_ELEMS - 1,
                    /*row_num=*/0,
                    /*sram_stride=*/0,
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

                npu_dma_mvout(
                    /*host_ptr=*/dst,
                    /*sram_addr=*/0,
                    /*col_num=*/DIM32_ELEMS - 1,
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

                uint32_t mismatch = 0;
                for (uint32_t i = 0; i < bytes; ++i) {
                    if (dst[i] != src[i]) {
                        ++mismatch;
                    }
                }

                if (mismatch == 0) {
                    std::cout << "Case A PASSED (col_num=" << (DIM32_ELEMS - 1) << ", row_num=0)" << std::endl;
                } else {
                    std::cerr << "Case A FAILED, mismatch=" << mismatch << std::endl;
                    all_passed = false;
                }

                npu_mem_free(src);
                npu_mem_free(dst);
            }
        }

        // Case B: large row_num, single column
        {
            const uint32_t rows = DIM32_ELEMS;
            const uint32_t bytes = rows;
            if (bytes > SPM_LIMIT_BYTES) {
                std::cerr << "Case B skipped: payload exceeds 512KB" << std::endl;
                all_passed = false;
            } else {
                uint8_t* src = static_cast<uint8_t*>(npu_mem_alloc(bytes));
                uint8_t* dst = static_cast<uint8_t*>(npu_mem_alloc(bytes));

                if (!src || !dst) {
                    std::cerr << "Case B alloc failed" << std::endl;
                    if (src) npu_mem_free(src);
                    if (dst) npu_mem_free(dst);
                    npu_destroy();
                    return -1;
                }

                for (uint32_t i = 0; i < bytes; ++i) {
                    src[i] = static_cast<uint8_t>((i * 5u + 3u) & 0xFFu);
                }
                std::memset(dst, 0, bytes);

                npu_dma_mvin(
                    /*host_ptr=*/src,
                    /*sram_addr=*/0,
                    /*col_num=*/0,
                    /*row_num=*/rows - 1,
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

                npu_dma_mvout(
                    /*host_ptr=*/dst,
                    /*sram_addr=*/0,
                    /*col_num=*/0,
                    /*row_num=*/rows - 1,
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

                uint32_t mismatch = 0;
                for (uint32_t i = 0; i < bytes; ++i) {
                    if (dst[i] != src[i]) {
                        ++mismatch;
                    }
                }

                if (mismatch == 0) {
                    std::cout << "Case B PASSED (col_num=0, row_num=" << (rows - 1) << ")" << std::endl;
                } else {
                    std::cerr << "Case B FAILED, mismatch=" << mismatch << std::endl;
                    all_passed = false;
                }

                npu_mem_free(src);
                npu_mem_free(dst);
            }
        }

        npu_destroy();

        std::cout << (all_passed ? "RESULT: PASS" : "RESULT: FAIL") << std::endl;
        return all_passed ? 0 : -1;
    }


