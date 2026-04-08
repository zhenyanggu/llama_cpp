#include "../npu_runtime.h"

#include <cstdint>
#include <cstring>
#include <iostream>

namespace {

void fill_pattern(uint8_t* p, size_t n, uint8_t seed) {
    for (size_t i = 0; i < n; ++i) {
        p[i] = static_cast<uint8_t>((seed + (i * 13u)) & 0xFFu);
    }
}

void do_mvin(void* host_ptr,
             uint32_t sram_addr,
             uint32_t col_num,
             uint32_t row_num,
             uint16_t sram_stride,
             uint32_t dram_stride,
             uint8_t precision,
             uint8_t input_type,
             bool dest,
             bool is_bias,
             bool is_quant,
             const char* tag) {
    std::cout << "[RUN] " << tag
              << " host_ptr=" << host_ptr
              << " sram_addr=0x" << std::hex << sram_addr << std::dec
              << " col_num=" << col_num
              << " row_num=" << row_num
              << " sram_stride=" << sram_stride
              << " dram_stride=" << dram_stride
              << " precision=" << static_cast<int>(precision)
              << " input_type=" << static_cast<int>(input_type)
              << " dest=" << static_cast<int>(dest)
              << " is_bias=" << static_cast<int>(is_bias)
              << std::endl;

    npu_dma_mvin(host_ptr,
                 sram_addr,
                 col_num,
                 row_num,
                 sram_stride,
                 dram_stride,
                 precision,
                 input_type,
                 dest,
                 is_bias,
                 is_quant,
                 /*quant_zero=*/0,
                 /*quant_scale=*/0,
                 /*quant_shift=*/0);
}

} // namespace

int main() {
    std::cout << "===== MVIN Repro Test (PCIe Slave Error) =====" << std::endl;

    if (npu_init() != 0) {
        std::cerr << "npu_init failed." << std::endl;
        return -1;
    }

    // 覆盖日志中出现的最大访问：row_num=7, col_num=8191, dram_stride=16384
    // 按常见 (row_num+1) 行模型，最后一行末尾地址约为 7*16384 + 8192。
    constexpr size_t kMainBufBytes = 131072; // 128 KiB，留足余量
    constexpr size_t kBiasBufBytes = 256;    // bias path 32B * 多次

    auto* main_buf = static_cast<uint8_t*>(npu_mem_alloc(kMainBufBytes));
    auto* bias_buf = static_cast<uint8_t*>(npu_mem_alloc(kBiasBufBytes));

    if (!main_buf || !bias_buf) {
        std::cerr << "npu_mem_alloc failed." << std::endl;
        if (main_buf) npu_mem_free(main_buf);
        if (bias_buf) npu_mem_free(bias_buf);
        npu_destroy();
        return -1;
    }

    fill_pattern(main_buf, kMainBufBytes, 0x11);
    fill_pattern(bias_buf, kBiasBufBytes, 0x5A);

    // 让每轮都从确定状态开始，避免受前一轮影响。
    npu_reset();

    // 你提供日志中的核心参数序列：
    // 1) col_num=223,row_num=6,stride=224
    do_mvin(main_buf + 0x0000,
            /*sram_addr=*/0x000024C0,
            /*col_num=*/223,
            /*row_num=*/6,
            /*sram_stride=*/224,
            /*dram_stride=*/224,
            /*precision=*/1,
            /*input_type=*/0,
            /*dest=*/0,
            /*is_bias=*/0,
            /*is_quant=*/0,
            "ifm-224x7-A");

    // 2) 同规格第二块
    do_mvin(main_buf + 0x0620,
            /*sram_addr=*/0x00002AE0,
            /*col_num=*/223,
            /*row_num=*/6,
            /*sram_stride=*/224,
            /*dram_stride=*/224,
            /*precision=*/1,
            /*input_type=*/0,
            /*dest=*/0,
            /*is_bias=*/0,
            /*is_quant=*/0,
            "ifm-224x7-B");

    // 3) 大 stride 场景（日志中伴随 nwl-pcie Slave error）
    do_mvin(main_buf + 0x2000,
            /*sram_addr=*/0x00003100,
            /*col_num=*/8191,
            /*row_num=*/7,
            /*sram_stride=*/8192,
            /*dram_stride=*/16384,
            /*precision=*/1,
            /*input_type=*/0,
            /*dest=*/0,
            /*is_bias=*/0,
            /*is_quant=*/0,
            "wide-8192x8-stride16384");

    // 4) bias 路径：dest=1,input_type=2,is_bias=1,row_num=0,col_num=31,stride=0
    //    按日志模式重复几次，观察是否触发同样异常。
    for (int round = 0; round < 2; ++round) {
        do_mvin(bias_buf + 0x00,
                /*sram_addr=*/0x00000000,
                /*col_num=*/31,
                /*row_num=*/0,
                /*sram_stride=*/0,
                /*dram_stride=*/0,
                /*precision=*/0,
                /*input_type=*/2,
                /*dest=*/1,
                /*is_bias=*/1,
                /*is_quant=*/0,
                "bias-32B-p0");

        do_mvin(bias_buf + 0x20,
                /*sram_addr=*/0x00000000,
                /*col_num=*/31,
                /*row_num=*/0,
                /*sram_stride=*/0,
                /*dram_stride=*/0,
                /*precision=*/0,
                /*input_type=*/2,
                /*dest=*/1,
                /*is_bias=*/1,
                /*is_quant=*/0,
                "bias-32B-p1");

        do_mvin(bias_buf + 0x40,
                /*sram_addr=*/0x00000000,
                /*col_num=*/31,
                /*row_num=*/0,
                /*sram_stride=*/0,
                /*dram_stride=*/0,
                /*precision=*/0,
                /*input_type=*/2,
                /*dest=*/1,
                /*is_bias=*/1,
                /*is_quant=*/0,
                "bias-32B-p2");
    }

    std::cout << "[DONE] Sequence finished. Please check dmesg for nwl-pcie errors." << std::endl;

    npu_mem_free(main_buf);
    npu_mem_free(bias_buf);
    npu_destroy();
    return 0;
}
