#include "npu_runtime.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <csignal>

namespace {

volatile sig_atomic_t g_last_stage = 0;

enum StageId : sig_atomic_t {
    STAGE_NONE = 0,
    STAGE_INIT,
    STAGE_RESET,
    STAGE_ALLOC,
    STAGE_PREPARE_INPUT,
    STAGE_MVIN_ACC_INT32_BEGIN,
    STAGE_MVIN_ACC_INT32_END,
    STAGE_MVOUT_ACC_RAW_BEGIN,
    STAGE_MVOUT_ACC_RAW_END,
    STAGE_CHECK_RAW,
    STAGE_MVOUT_ACC_FP32_BEGIN,
    STAGE_MVOUT_ACC_FP32_END,
    STAGE_CHECK_FP32,
    STAGE_DONE,
};

const char * stage_name(sig_atomic_t stage) {
    switch (stage) {
        case STAGE_NONE: return "none";
        case STAGE_INIT: return "init";
        case STAGE_RESET: return "reset";
        case STAGE_ALLOC: return "alloc";
        case STAGE_PREPARE_INPUT: return "prepare_input";
        case STAGE_MVIN_ACC_INT32_BEGIN: return "mvin_acc_int32_begin";
        case STAGE_MVIN_ACC_INT32_END: return "mvin_acc_int32_end";
        case STAGE_MVOUT_ACC_RAW_BEGIN: return "mvout_acc_raw_begin";
        case STAGE_MVOUT_ACC_RAW_END: return "mvout_acc_raw_end";
        case STAGE_CHECK_RAW: return "check_raw";
        case STAGE_MVOUT_ACC_FP32_BEGIN: return "mvout_acc_fp32_begin";
        case STAGE_MVOUT_ACC_FP32_END: return "mvout_acc_fp32_end";
        case STAGE_CHECK_FP32: return "check_fp32";
        case STAGE_DONE: return "done";
        default: return "unknown";
    }
}

void set_stage(sig_atomic_t stage) {
    g_last_stage = stage;
}

void log_stage(const char * label) {
    std::fprintf(stderr, "[stage] %s\n", label);
    std::fflush(stderr);
}

void log_dma_call(
        const char * op,
        void * host_ptr,
        uint32_t sram_addr,
        uint32_t col_num,
        uint32_t row_num,
        uint16_t sram_stride,
        uint32_t dram_stride,
        uint8_t precision,
        uint8_t type,
        bool flag0,
        bool flag1,
        uint32_t arg0,
        uint32_t arg1) {
    std::fprintf(
        stderr,
        "[call] %s host_ptr=%p sram_addr=0x%08x col_num=%u row_num=%u sram_stride=%u dram_stride=%u precision=%u type=%u flag0=%d flag1=%d arg0=0x%08x arg1=0x%08x\n",
        op,
        host_ptr,
        sram_addr,
        col_num,
        row_num,
        static_cast<unsigned>(sram_stride),
        dram_stride,
        static_cast<unsigned>(precision),
        static_cast<unsigned>(type),
        flag0 ? 1 : 0,
        flag1 ? 1 : 0,
        arg0,
        arg1);
    std::fflush(stderr);
}

void signal_handler(int signo) {
    std::fprintf(
        stderr,
        "[signal] signo=%d last_stage=%d (%s)\n",
        signo,
        static_cast<int>(g_last_stage),
        stage_name(g_last_stage));
    std::fflush(stderr);
    std::_Exit(128 + signo);
}

void install_signal_handlers() {
    std::signal(SIGBUS, signal_handler);
    std::signal(SIGSEGV, signal_handler);
    std::signal(SIGABRT, signal_handler);
}

struct Config {
    uint32_t acc_addr = 0x0000;
    uint32_t col_num = 127;
    uint32_t row_num = 0;
    uint16_t mvout_sram_stride = 128;
    uint32_t mvout_dram_stride = 0;
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
            cfg.mvout_sram_stride = parse_u16(argv[++i], cfg.mvout_sram_stride);
            continue;
        }
        if (std::strcmp(argv[i], "--dram-stride") == 0 && i + 1 < argc) {
            cfg.mvout_dram_stride = parse_u32(argv[++i], cfg.mvout_dram_stride);
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

    if (cfg.mvout_sram_stride == 0) {
        cfg.mvout_sram_stride = static_cast<uint16_t>(cfg.col_num + 1);
    }
    return cfg;
}

} // namespace

int main(int argc, char ** argv) {
    install_signal_handlers();

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
        cfg.mvout_sram_stride,
        cfg.mvout_dram_stride,
        cfg.scale,
        cfg.zero_point,
        cfg.run_fp32 ? 1 : 0);

    set_stage(STAGE_INIT);
    log_stage("before npu_init");
    if (npu_init() != 0) {
        std::perror("npu_init");
        return 1;
    }
    set_stage(STAGE_RESET);
    log_stage("before npu_reset");
    npu_reset();

    set_stage(STAGE_ALLOC);
    log_stage("before npu_mem_alloc");
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
        set_stage(STAGE_PREPARE_INPUT);
        std::fprintf(stderr, "[iter] %d prepare_input elems=%zu\n", it, elem_count);
        std::fflush(stderr);
        for (size_t i = 0; i < elem_count; ++i) {
            host_src[i] = static_cast<int32_t>((i + it * 17) % 10007 - 5003);
        }
        std::memset(host_raw, 0, elem_count * sizeof(int32_t));
        std::memset(host_f32, 0, elem_count * sizeof(float));

        // Step 1: write int32 to ACC.
        set_stage(STAGE_MVIN_ACC_INT32_BEGIN);
        log_dma_call(
            "npu_dma_mvin(acc-int32)",
            host_src,
            cfg.acc_addr,
            cfg.col_num,
            cfg.row_num,
            0,
            0,
            1,
            2,
            true,
            false,
            0,
            0);
        npu_dma_mvin(
            /*host_ptr=*/host_src,
            /*sram_addr=*/cfg.acc_addr,
            /*col_num=*/cfg.col_num,
            /*row_num=*/cfg.row_num,
            /*sram_stride=*/0,
            /*dram_stride=*/0,
            /*precision=*/1,
            /*input_type=*/2, // bias/int32 path
            /*dest=*/true,    // ACC
            /*is_bias=*/false,
            /*is_quant=*/false,
            /*quant_zero=*/0,
            /*quant_scale=*/0,
            /*quant_shift=*/0);
        set_stage(STAGE_MVIN_ACC_INT32_END);
        log_stage("after npu_dma_mvin(acc-int32)");

        // Step 2: raw int32 readback.
        set_stage(STAGE_MVOUT_ACC_RAW_BEGIN);
        log_dma_call(
            "npu_dma_mvout(acc-raw-int32)",
            host_raw,
            cfg.acc_addr,
            cfg.col_num,
            cfg.row_num,
            cfg.mvout_sram_stride,
            cfg.mvout_dram_stride,
            1,
            1,
            true,
            false,
            0,
            0);
        npu_dma_mvout(
            /*host_ptr=*/host_raw,
            /*sram_addr=*/cfg.acc_addr,
            /*col_num=*/cfg.col_num,
            /*row_num=*/cfg.row_num,
            /*sram_stride=*/cfg.mvout_sram_stride,
            /*dram_stride=*/cfg.mvout_dram_stride,
            /*precision=*/1,
            /*output_type=*/1, // int32
            /*source=*/true,   // ACC
            /*is_quant=*/false,
            /*quant_zero=*/0,
            /*f32_scale=*/0);
        set_stage(STAGE_MVOUT_ACC_RAW_END);
        log_stage("after npu_dma_mvout(acc-raw-int32)");

        set_stage(STAGE_CHECK_RAW);
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
            set_stage(STAGE_MVOUT_ACC_FP32_BEGIN);
            log_dma_call(
                "npu_dma_mvout(acc-fp32-dequant)",
                host_f32,
                cfg.acc_addr,
                cfg.col_num,
                cfg.row_num,
                cfg.mvout_sram_stride,
                cfg.mvout_dram_stride,
                3,
                1,
                true,
                true,
                cfg.zero_point,
                f32_scale);
            npu_dma_mvout(
                /*host_ptr=*/host_f32,
                /*sram_addr=*/cfg.acc_addr,
                /*col_num=*/cfg.col_num,
                /*row_num=*/cfg.row_num,
                /*sram_stride=*/cfg.mvout_sram_stride,
                /*dram_stride=*/cfg.mvout_dram_stride,
                /*precision=*/3,   // fp32 output path
                /*output_type=*/1, // ACC int32 source
                /*source=*/true,
                /*is_quant=*/true,
                /*quant_zero=*/cfg.zero_point,
                /*f32_scale=*/f32_scale);
            set_stage(STAGE_MVOUT_ACC_FP32_END);
            log_stage("after npu_dma_mvout(acc-fp32-dequant)");

            set_stage(STAGE_CHECK_FP32);
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

    set_stage(STAGE_DONE);
    std::puts("kv260_dma_acc_int32_fp32_test=ok");
    npu_mem_free(host_src);
    npu_mem_free(host_raw);
    npu_mem_free(host_f32);
    npu_destroy();
    return 0;
}
