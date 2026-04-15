#include "npu_runtime.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>

namespace {

struct Config {
    uint32_t acc_addr = 0x0000;
    uint32_t col_num = 127;
    uint32_t row_num = 0;
    uint16_t sram_stride = 0;
    uint32_t dram_stride = 0;
    uint32_t zero_point = 0;
    float scale = 1.0f;
    int loops = 1;
    bool run_fp32 = true;
};

uint32_t pack_f32_scale(float scale) {
    return static_cast<uint32_t>(std::round(scale * 16777216.0f));
}

float dequant_theory(int32_t x, uint32_t zero_point, float scale) {
    const float centered = static_cast<float>(x) - static_cast<float>(zero_point);
    return centered * scale;
}

uint32_t parse_u32(const char * s, uint32_t fallback) {
    if (s == nullptr || s[0] == '\0') {
        return fallback;
    }
    return static_cast<uint32_t>(std::strtoull(s, nullptr, 0));
}

uint16_t parse_u16(const char * s, uint16_t fallback) {
    return static_cast<uint16_t>(parse_u32(s, fallback));
}

float parse_f32(const char * s, float fallback) {
    if (s == nullptr || s[0] == '\0') {
        return fallback;
    }
    return std::strtof(s, nullptr);
}

Config parse_args(int argc, char ** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--col") == 0 && i + 1 < argc) {
            cfg.col_num = parse_u32(argv[++i], cfg.col_num);
            continue;
        }
        if (std::strcmp(argv[i], "--row") == 0 && i + 1 < argc) {
            cfg.row_num = parse_u32(argv[++i], cfg.row_num);
            continue;
        }
        if (std::strcmp(argv[i], "--sram-stride") == 0 && i + 1 < argc) {
            cfg.sram_stride = parse_u16(argv[++i], cfg.sram_stride);
            continue;
        }
        if (std::strcmp(argv[i], "--dram-stride") == 0 && i + 1 < argc) {
            cfg.dram_stride = parse_u32(argv[++i], cfg.dram_stride);
            continue;
        }
        if (std::strcmp(argv[i], "--scale") == 0 && i + 1 < argc) {
            cfg.scale = parse_f32(argv[++i], cfg.scale);
            continue;
        }
        if (std::strcmp(argv[i], "--zero-point") == 0 && i + 1 < argc) {
            cfg.zero_point = parse_u32(argv[++i], cfg.zero_point);
            continue;
        }
        if (std::strcmp(argv[i], "--loops") == 0 && i + 1 < argc) {
            const int v = std::atoi(argv[++i]);
            cfg.loops = v > 0 ? v : cfg.loops;
            continue;
        }
        if (std::strcmp(argv[i], "--acc-addr") == 0 && i + 1 < argc) {
            cfg.acc_addr = parse_u32(argv[++i], cfg.acc_addr);
            continue;
        }
        if (std::strcmp(argv[i], "--no-fp32") == 0) {
            cfg.run_fp32 = false;
            continue;
        }
    }
    return cfg;
}

} // namespace

int main(int argc, char ** argv) {
    const Config cfg = parse_args(argc, argv);
    const size_t rows = static_cast<size_t>(cfg.row_num + 1);
    const size_t cols = static_cast<size_t>(cfg.col_num + 1);
    const size_t elem_count = rows * cols;
    const uint32_t f32_scale = pack_f32_scale(cfg.scale);

    std::printf(
        "kv260_dma_acc_int32_fp32_test: loops=%d col=%u row=%u elems=%zu sram_stride=%u dram_stride=%u scale=%.9g zp=%u fp32=%d\n",
        cfg.loops,
        cfg.col_num,
        cfg.row_num,
        elem_count,
        cfg.sram_stride,
        cfg.dram_stride,
        cfg.scale,
        cfg.zero_point,
        cfg.run_fp32 ? 1 : 0);

    if (npu_init() != 0) {
        std::perror("npu_init");
        return 1;
    }
    npu_reset();

    auto * host_src = static_cast<int32_t *>(npu_mem_alloc(elem_count * sizeof(int32_t)));
    auto * host_raw = static_cast<int32_t *>(npu_mem_alloc(elem_count * sizeof(int32_t)));
    auto * host_f32 = static_cast<float *>(npu_mem_alloc(elem_count * sizeof(float)));
    if (!host_src || !host_raw || !host_f32) {
        std::fprintf(stderr, "npu_mem_alloc failed\n");
        if (host_src) npu_mem_free(host_src);
        if (host_raw) npu_mem_free(host_raw);
        if (host_f32) npu_mem_free(host_f32);
        npu_destroy();
        return 2;
    }

    for (int it = 0; it < cfg.loops; ++it) {
        for (size_t i = 0; i < elem_count; ++i) {
            host_src[i] = static_cast<int32_t>((i + it * 17) % 10007 - 5003);
        }
        std::memset(host_raw, 0, elem_count * sizeof(int32_t));
        std::memset(host_f32, 0, elem_count * sizeof(float));

        // Step 1: write int32 to ACC.
        npu_dma_mvin(
            /*host_ptr=*/host_src,
            /*sram_addr=*/cfg.acc_addr,
            /*col_num=*/cfg.col_num,
            /*row_num=*/cfg.row_num,
            /*sram_stride=*/cfg.sram_stride,
            /*dram_stride=*/cfg.dram_stride,
            /*precision=*/1,
            /*input_type=*/2, // bias/int32 path
            /*dest=*/true,    // ACC
            /*is_bias=*/false,
            /*is_quant=*/false,
            /*quant_zero=*/0,
            /*quant_scale=*/0,
            /*quant_shift=*/0);

        // Step 2: raw int32 readback.
        npu_dma_mvout(
            /*host_ptr=*/host_raw,
            /*sram_addr=*/cfg.acc_addr,
            /*col_num=*/cfg.col_num,
            /*row_num=*/cfg.row_num,
            /*sram_stride=*/cfg.sram_stride,
            /*dram_stride=*/cfg.dram_stride,
            /*precision=*/1,
            /*output_type=*/1, // int32
            /*source=*/true,   // ACC
            /*is_quant=*/false,
            /*quant_zero=*/0,
            /*f32_scale=*/0);

        size_t raw_mismatch = 0;
        for (size_t i = 0; i < elem_count; ++i) {
            if (host_raw[i] != host_src[i]) {
                if (raw_mismatch < 8) {
                    std::fprintf(stderr,
                                 "iter=%d raw mismatch idx=%zu expect=%d got=%d\n",
                                 it, i, host_src[i], host_raw[i]);
                }
                ++raw_mismatch;
            }
        }
        if (raw_mismatch != 0) {
            std::fprintf(stderr, "iter=%d raw readback failed mismatches=%zu\n", it, raw_mismatch);
            npu_mem_free(host_src);
            npu_mem_free(host_raw);
            npu_mem_free(host_f32);
            npu_destroy();
            return 3;
        }

        if (cfg.run_fp32) {
            // Step 3: fp32 readback with dequant.
            npu_dma_mvout(
                /*host_ptr=*/host_f32,
                /*sram_addr=*/cfg.acc_addr,
                /*col_num=*/cfg.col_num,
                /*row_num=*/cfg.row_num,
                /*sram_stride=*/cfg.sram_stride,
                /*dram_stride=*/cfg.dram_stride,
                /*precision=*/3,   // fp32 output path
                /*output_type=*/1, // ACC int32 source
                /*source=*/true,
                /*is_quant=*/true,
                /*quant_zero=*/cfg.zero_point,
                /*f32_scale=*/f32_scale);

            size_t f32_mismatch = 0;
            const float abs_tol = std::max(1e-4f, std::fabs(cfg.scale) * 2e-4f);
            for (size_t i = 0; i < elem_count; ++i) {
                const float expect = dequant_theory(host_src[i], cfg.zero_point, cfg.scale);
                const float got = host_f32[i];
                if (std::fabs(expect - got) > abs_tol) {
                    if (f32_mismatch < 8) {
                        std::fprintf(stderr,
                                     "iter=%d f32 mismatch idx=%zu expect=%.6f got=%.6f src=%d\n",
                                     it, i, expect, got, host_src[i]);
                    }
                    ++f32_mismatch;
                }
            }
            if (f32_mismatch != 0) {
                std::fprintf(stderr, "iter=%d fp32 readback failed mismatches=%zu\n", it, f32_mismatch);
                npu_mem_free(host_src);
                npu_mem_free(host_raw);
                npu_mem_free(host_f32);
                npu_destroy();
                return 4;
            }
        }
    }

    std::puts("kv260_dma_acc_int32_fp32_test=ok");
    npu_mem_free(host_src);
    npu_mem_free(host_raw);
    npu_mem_free(host_f32);
    npu_destroy();
    return 0;
}

