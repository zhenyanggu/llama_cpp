#include "../npu_runtime.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>

namespace {

constexpr uint32_t ACC_ADDR_A   = 0x00000;
constexpr uint32_t ACC_ADDR_B   = 0x20000; // 256*128 int32 = 0x20000 bytes
constexpr uint32_t SPM_ADDR_OUT = 0x00000;

constexpr uint32_t ACC_ADDR_STRESS = 0x00000;

constexpr int COL = 256;
constexpr int ROW = 126;
constexpr int ELEM_COUNT = COL * ROW;

constexpr int ACC_STRESS_BYTES = 256 * 1024;
constexpr int ACC_STRESS_ELEMS = ACC_STRESS_BYTES / 4; // ACC按int32元素寻址

inline int8_t sat_add_i8(int8_t a, int8_t b) {
    int32_t s = static_cast<int32_t>(a) + static_cast<int32_t>(b);
    if (s > 127) s = 127;
    if (s < -128) s = -128;
    return static_cast<int8_t>(s);
}

} // namespace

int main() {
    std::cout << "[MATADD 256x128] start" << std::endl;

    if (npu_init() != 0) {
        std::cerr << "[ERROR] npu_init failed" << std::endl;
        return 1;
    }

    auto* host_a   = static_cast<int8_t*>(npu_mem_alloc(ELEM_COUNT));
    auto* host_b   = static_cast<int8_t*>(npu_mem_alloc(ELEM_COUNT));
    auto* host_out = static_cast<int8_t*>(npu_mem_alloc(ELEM_COUNT));
    auto* host_acc_in  = static_cast<int8_t*>(npu_mem_alloc(ACC_STRESS_ELEMS));
    auto* host_acc_out = static_cast<int32_t*>(npu_mem_alloc(ACC_STRESS_ELEMS * sizeof(int32_t)));

    if (!host_a || !host_b || !host_out || !host_acc_in || !host_acc_out) {
        std::cerr << "[ERROR] npu_mem_alloc failed" << std::endl;
        if (host_a) npu_mem_free(host_a);
        if (host_b) npu_mem_free(host_b);
        if (host_out) npu_mem_free(host_out);
        if (host_acc_in) npu_mem_free(host_acc_in);
        if (host_acc_out) npu_mem_free(host_acc_out);
        npu_destroy();
        return 1;
    }

    // ============================================================
    // ACC容量压测：写满 256KB（int32视角）再读回校验
    // 方式：int8 --(MVIN量化scale=1,shift=0)--> ACC int32，再MVOUT int32读回
    // ============================================================
    for (int i = 0; i < ACC_STRESS_ELEMS; ++i) {
        host_acc_in[i] = static_cast<int8_t>(((i * 13 + 17) & 0xFF) - 128);
    }
    std::memset(host_acc_out, 0, ACC_STRESS_ELEMS * sizeof(int32_t));

    npu_dma_mvin(
        host_acc_in,
        ACC_ADDR_STRESS,
        static_cast<uint32_t>(ACC_STRESS_ELEMS - 1),
        0,
        0,
        0,
        1,
        0,
        true,
        false,
        true,
        0,
        1,
        0);

    npu_dma_mvout(
        host_acc_out,
        ACC_ADDR_STRESS,
        static_cast<uint32_t>(ACC_STRESS_ELEMS - 1),
        0,
        0,
        0,
        1,
        1,
        true,
        false,
        0,
        0,
        0);

    int acc_mismatch = 0;
    for (int i = 0; i < ACC_STRESS_ELEMS; ++i) {
        int32_t expect = static_cast<int32_t>(host_acc_in[i]);
        if (host_acc_out[i] != expect) {
            if (acc_mismatch < 10) {
                std::cout << "[ACC-RW Mismatch] idx=" << i
                          << " hw=" << host_acc_out[i]
                          << " exp=" << expect << std::endl;
            }
            ++acc_mismatch;
        }
    }

    std::cout << "[ACC 256KB RW] mismatch=" << acc_mismatch
              << " elems=" << ACC_STRESS_ELEMS << std::endl;
    if (acc_mismatch != 0) {
        npu_mem_free(host_a);
        npu_mem_free(host_b);
        npu_mem_free(host_out);
        npu_mem_free(host_acc_in);
        npu_mem_free(host_acc_out);
        npu_destroy();
        std::cout << "[FAIL] ACC 256KB roundtrip failed" << std::endl;
        return 3;
    }

    int8_t ref[ELEM_COUNT];

    for (int i = 0; i < ELEM_COUNT; ++i) {
        // 覆盖正负值与饱和场景
        host_a[i] = static_cast<int8_t>((i % 256) - 128);
        host_b[i] = static_cast<int8_t>(((i * 7 + 13) % 256) - 128);
        ref[i] = sat_add_i8(host_a[i], host_b[i]);
    }
    std::memset(host_out, 0, ELEM_COUNT);

    // DRAM -> ACC: 按线性buffer搬运
    npu_dma_mvin(
        host_a,
        ACC_ADDR_A,
        static_cast<uint32_t>(ELEM_COUNT - 1),
        0,
        0,
        0,
        1,
        0,
        true,
        false,
        true,
        0,
        1,
        0);

    npu_dma_mvin(
        host_b,
        ACC_ADDR_B,
        static_cast<uint32_t>(ELEM_COUNT - 1),
        0,
        0,
        0,
        1,
        0,
        true,
        false,
        true,
        0,
        1,
        0);

    // 注意：MATADD API 的 col/row 现在是 -1 语义
    // 256x128 => col_num_m1=255, row_num_m1=127
    npu_matadd_run(
        ACC_ADDR_A,
        ACC_ADDR_B,
        SPM_ADDR_OUT,
        static_cast<uint8_t>(COL - 1),
        static_cast<uint8_t>(ROW - 1),
        0,
        1,
        0);

    // SPM -> DRAM
    npu_dma_mvout(
        host_out,
        SPM_ADDR_OUT,
        static_cast<uint32_t>(ELEM_COUNT - 1),
        0,
        0,
        0,
        1,
        0,
        false,
        false,
        0,
        0,
        0);

    int mismatch = 0;
    int max_diff = 0;
    for (int i = 0; i < ELEM_COUNT; ++i) {
        int diff = std::abs(static_cast<int>(host_out[i]) - static_cast<int>(ref[i]));
        if (diff > 0) {
            if (mismatch < 10) {
                int r = i / COL;
                int c = i % COL;
                std::cout << "[Mismatch] idx=" << i
                          << " (" << r << "," << c << ")"
                          << " hw=" << static_cast<int>(host_out[i])
                          << " ref=" << static_cast<int>(ref[i])
                          << " a=" << static_cast<int>(host_a[i])
                          << " b=" << static_cast<int>(host_b[i])
                          << std::endl;
            }
            ++mismatch;
        }
        max_diff = std::max(max_diff, diff);
    }

    std::cout << "[MATADD 256x128] done, mismatch=" << mismatch
              << ", max_diff=" << max_diff << std::endl;

    npu_mem_free(host_a);
    npu_mem_free(host_b);
    npu_mem_free(host_out);
    npu_mem_free(host_acc_in);
    npu_mem_free(host_acc_out);
    npu_destroy();

    if (mismatch == 0) {
        std::cout << "[PASS]" << std::endl;
        return 0;
    }

    std::cout << "[FAIL]" << std::endl;
    return 2;
}
