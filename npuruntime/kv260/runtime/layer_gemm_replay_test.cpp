#include "npu_runtime.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

namespace {
constexpr uint32_t kAccDmaId = 2;
}

struct Config {
    int m = 16;   // output channels (tile width in profile)
    int n = 16;   // token rows (tile height in profile)
    int k = 768;  // reduction dim
    int loops = 1;
    uint32_t sram_act = 0x00000000;
    uint32_t sram_wgt = 0x00010000;
    uint32_t bias_acc_addr = 0x00000000;
    uint32_t out_acc_addr = 0x00004000;
    bool use_double_mvin = true;
    bool enable_bias = true;
    bool bias_via_psum = false;
    bool zero_bias = false;
    bool bias_clear_then_load = false;
    bool bias_const_set = false;
    int32_t bias_const = 0;
    bool bias_second_const_set = false;
    int32_t bias_second_const = 0;
    bool asymmetric_activations = false;
    uint16_t input_a_zeropoint = 0;
    uint16_t input_b_zeropoint = 0;
    bool mvout_fp32 = false;
    float fp32_scale = 1.0f;
    uint32_t fp32_zero_point = 0;
    uint32_t seed = 20260416u;
    bool act_2d_mvin = false;
    bool wgt_2d_mvin = false;
};

static uint32_t parse_u32(const char * s, uint32_t fallback) {
    if (!s || !*s) {
        return fallback;
    }
    char * end = nullptr;
    const unsigned long v = std::strtoul(s, &end, 0);
    if (!end || *end != '\0') {
        return fallback;
    }
    return static_cast<uint32_t>(v);
}

static int parse_i32(const char * s, int fallback) {
    if (!s || !*s) {
        return fallback;
    }
    char * end = nullptr;
    const long v = std::strtol(s, &end, 0);
    if (!end || *end != '\0') {
        return fallback;
    }
    return static_cast<int>(v);
}

static void print_usage(const char * prog) {
    std::printf(
        "Usage: %s [options]\n"
        "  --m <int>             GEMM output channels (default: 16)\n"
        "  --n <int>             GEMM output rows/tokens (default: 16)\n"
        "  --k <int>             GEMM reduction dim (default: 768)\n"
        "  --loops <int>         Repeat loops (default: 1)\n"
        "  --sram-act <hex/int>  Activation SPM addr (default: 0x0)\n"
        "  --sram-wgt <hex/int>  Weight SPM addr (default: 0x10000)\n"
        "  --acc-addr <hex/int>  Legacy shortcut: set both bias and output ACC addr\n"
        "  --bias-acc-addr <hex/int>  ACC base addr for bias/psum input (default: 0x0)\n"
        "  --out-acc-addr <hex/int>   ACC base addr for GEMM output (default: 0x4000)\n"
        "  --seed <uint>         Pseudo-random seed (default: 20260416)\n"
        "  --double-mvin         Use npu_dma_double_mvin when possible (default)\n"
        "  --no-double-mvin      Disable double mvin\n"
        "  --act-2d-mvin         MVIN activation as 2D tensor [n,k]\n"
        "  --wgt-2d-mvin         MVIN weight as 2D tensor [k,m]\n"
        "  --both-2d-mvin        MVIN both activation and weight in 2D mode\n"
        "  --no-bias             Do not load/apply bias\n"
        "  --bias-psum           Bias through ACC psum buffer (is_bias=0, isaccu=1)\n"
        "  --zero-bias           Load bias path but fill zeros\n"
        "  --bias-clear-then-load  Load zero bias once, then load target bias once\n"
        "  --bias-const <int>    Fill bias vector with a constant int32 value\n"
        "  --bias-second-const <int>  Load a second constant bias vector before GEMM\n"
        "  --asym-act            Enable asymmetric activations in npu_gemm_run\n"
        "  --input-a-zp <uint>   INPUTA_ZP register (default: 0)\n"
        "  --input-b-zp <uint>   INPUTB_ZP register (default: 0)\n"
        "  --mvout-fp32          Read ACC as fp32 using per-tensor dequant\n"
        "  --fp32-scale <float>  Per-tensor FP32 dequant scale, packed as Q8.24 immediate (default: 1.0)\n"
        "  --fp32-zp <uint>      FP32 dequant zero point (default: 0)\n"
        "  -h, --help            Show this help\n",
        prog);
}

static Config parse_args(int argc, char ** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        if ((std::strcmp(argv[i], "-h") == 0) || (std::strcmp(argv[i], "--help") == 0)) {
            print_usage(argv[0]);
            std::exit(0);
        }
        if (std::strcmp(argv[i], "--m") == 0 && i + 1 < argc) {
            cfg.m = parse_i32(argv[++i], cfg.m);
            continue;
        }
        if (std::strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
            cfg.n = parse_i32(argv[++i], cfg.n);
            continue;
        }
        if (std::strcmp(argv[i], "--k") == 0 && i + 1 < argc) {
            cfg.k = parse_i32(argv[++i], cfg.k);
            continue;
        }
        if (std::strcmp(argv[i], "--loops") == 0 && i + 1 < argc) {
            cfg.loops = parse_i32(argv[++i], cfg.loops);
            continue;
        }
        if (std::strcmp(argv[i], "--sram-act") == 0 && i + 1 < argc) {
            cfg.sram_act = parse_u32(argv[++i], cfg.sram_act);
            continue;
        }
        if (std::strcmp(argv[i], "--sram-wgt") == 0 && i + 1 < argc) {
            cfg.sram_wgt = parse_u32(argv[++i], cfg.sram_wgt);
            continue;
        }
        if (std::strcmp(argv[i], "--acc-addr") == 0 && i + 1 < argc) {
            const uint32_t acc_addr = parse_u32(argv[++i], cfg.bias_acc_addr);
            cfg.bias_acc_addr = acc_addr;
            cfg.out_acc_addr = acc_addr;
            continue;
        }
        if (std::strcmp(argv[i], "--bias-acc-addr") == 0 && i + 1 < argc) {
            cfg.bias_acc_addr = parse_u32(argv[++i], cfg.bias_acc_addr);
            continue;
        }
        if (std::strcmp(argv[i], "--out-acc-addr") == 0 && i + 1 < argc) {
            cfg.out_acc_addr = parse_u32(argv[++i], cfg.out_acc_addr);
            continue;
        }
        if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            cfg.seed = parse_u32(argv[++i], cfg.seed);
            continue;
        }
        if (std::strcmp(argv[i], "--double-mvin") == 0) {
            cfg.use_double_mvin = true;
            continue;
        }
        if (std::strcmp(argv[i], "--no-double-mvin") == 0) {
            cfg.use_double_mvin = false;
            continue;
        }
        if (std::strcmp(argv[i], "--act-2d-mvin") == 0) {
            cfg.act_2d_mvin = true;
            continue;
        }
        if (std::strcmp(argv[i], "--wgt-2d-mvin") == 0) {
            cfg.wgt_2d_mvin = true;
            continue;
        }
        if (std::strcmp(argv[i], "--both-2d-mvin") == 0) {
            cfg.act_2d_mvin = true;
            cfg.wgt_2d_mvin = true;
            continue;
        }
        if (std::strcmp(argv[i], "--no-bias") == 0) {
            cfg.enable_bias = false;
            continue;
        }
        if (std::strcmp(argv[i], "--bias-psum") == 0) {
            cfg.bias_via_psum = true;
            continue;
        }
        if (std::strcmp(argv[i], "--zero-bias") == 0) {
            cfg.zero_bias = true;
            continue;
        }
        if (std::strcmp(argv[i], "--bias-clear-then-load") == 0) {
            cfg.bias_clear_then_load = true;
            continue;
        }
        if (std::strcmp(argv[i], "--bias-const") == 0 && i + 1 < argc) {
            cfg.bias_const = static_cast<int32_t>(parse_i32(argv[++i], cfg.bias_const));
            cfg.bias_const_set = true;
            continue;
        }
        if (std::strcmp(argv[i], "--bias-second-const") == 0 && i + 1 < argc) {
            cfg.bias_second_const = static_cast<int32_t>(parse_i32(argv[++i], cfg.bias_second_const));
            cfg.bias_second_const_set = true;
            continue;
        }
        if (std::strcmp(argv[i], "--asym-act") == 0) {
            cfg.asymmetric_activations = true;
            continue;
        }
        if (std::strcmp(argv[i], "--input-a-zp") == 0 && i + 1 < argc) {
            cfg.input_a_zeropoint = static_cast<uint16_t>(parse_u32(argv[++i], cfg.input_a_zeropoint));
            continue;
        }
        if (std::strcmp(argv[i], "--input-b-zp") == 0 && i + 1 < argc) {
            cfg.input_b_zeropoint = static_cast<uint16_t>(parse_u32(argv[++i], cfg.input_b_zeropoint));
            continue;
        }
        if (std::strcmp(argv[i], "--mvout-fp32") == 0) {
            cfg.mvout_fp32 = true;
            continue;
        }
        if (std::strcmp(argv[i], "--fp32-scale") == 0 && i + 1 < argc) {
            cfg.fp32_scale = std::strtof(argv[++i], nullptr);
            continue;
        }
        if (std::strcmp(argv[i], "--fp32-zp") == 0 && i + 1 < argc) {
            cfg.fp32_zero_point = parse_u32(argv[++i], cfg.fp32_zero_point);
            continue;
        }
        std::fprintf(stderr, "Unknown arg: %s\n", argv[i]);
        print_usage(argv[0]);
        std::exit(2);
    }

    if (cfg.m <= 0 || cfg.n <= 0 || cfg.k <= 0) {
        std::fprintf(stderr, "Invalid shape: m=%d n=%d k=%d\n", cfg.m, cfg.n, cfg.k);
        std::exit(2);
    }
    if (cfg.m > 255 || cfg.n > 255) {
        std::fprintf(stderr, "m and n must be <=255 because gemm bias width/height are uint8.\n");
        std::exit(2);
    }
    if (cfg.loops <= 0) {
        cfg.loops = 1;
    }

    return cfg;
}

static int32_t effective_a(int8_t raw, const Config & cfg) {
    if (cfg.asymmetric_activations) {
        return static_cast<int32_t>(static_cast<uint8_t>(raw)) - 128 - static_cast<int32_t>(cfg.input_a_zeropoint);
    }
    return static_cast<int32_t>(raw) - static_cast<int32_t>(cfg.input_a_zeropoint);
}

static int32_t effective_b(int8_t raw, const Config & cfg) {
    return static_cast<int32_t>(raw) - static_cast<int32_t>(cfg.input_b_zeropoint);
}

static uint32_t pack_f32_scale(float scale) {
    return static_cast<uint32_t>(std::round(scale * 16777216.0f));
}

static int8_t gen_i8(uint32_t seed, uint32_t idx, uint32_t salt) {
    uint32_t x = idx ^ (seed + salt * 0x9E3779B9u);
    x ^= x >> 16;
    x *= 0x7FEB352Du;
    x ^= x >> 15;
    x *= 0x846CA68Bu;
    x ^= x >> 16;
    return static_cast<int8_t>((x & 0xFFu) - 128);
}

static int32_t gen_i32(uint32_t seed, uint32_t idx, uint32_t salt) {
    uint32_t x = idx ^ (seed + salt * 0x85EBCA77u);
    x ^= x >> 16;
    x *= 0x7FEB352Du;
    x ^= x >> 15;
    x *= 0x846CA68Bu;
    x ^= x >> 16;
    return static_cast<int32_t>(x % 4096u) - 2048;
}

static void mvin_bias_vector(const int32_t * host_bias, const Config & cfg) {
    std::puts("runtime_call=npu_dma_mvin bias->ACC");
    const MvinConfig bias_mvin_cfg {
        const_cast<int32_t *>(host_bias),
        cfg.bias_acc_addr,
        static_cast<uint32_t>(cfg.m - 1),
        0,
        0,
        0,
        1,
        2,
        true,
        !cfg.bias_via_psum,
        false,
        0,
        0,
        0,
    };
    npu_dma_mvin_async(kAccDmaId, &bias_mvin_cfg);
    npu_dma_wait_mvin(1u << kAccDmaId);
}

int main(int argc, char ** argv) {
    const Config cfg = parse_args(argc, argv);

    const uint32_t a_bytes = static_cast<uint32_t>(cfg.n * cfg.k);         // A: [n, k], int8
    const uint32_t b_bytes = static_cast<uint32_t>(cfg.k * cfg.m);         // B: [k, m], int8
    const uint32_t bias_elems = static_cast<uint32_t>(cfg.m);              // bias: [m], int32
    const uint32_t out_elems = static_cast<uint32_t>(cfg.n * cfg.m);       // C: [n, m], int32
    const uint32_t out_bytes = out_elems * static_cast<uint32_t>(sizeof(int32_t));

    std::printf(
        "kv260_layer_gemm_replay_test: m=%d n=%d k=%d loops=%d seed=%u sram_act=0x%08x sram_wgt=0x%08x bias_acc=0x%08x out_acc=0x%08x double_mvin=%d bias=%d bias_via_psum=%d zero_bias=%d asym_act=%d input_a_zp=%u input_b_zp=%u mvout_fp32=%d fp32_scale=%.9g fp32_zp=%u\n",
        cfg.m, cfg.n, cfg.k, cfg.loops, cfg.seed, cfg.sram_act, cfg.sram_wgt, cfg.bias_acc_addr, cfg.out_acc_addr,
        cfg.use_double_mvin ? 1 : 0, cfg.enable_bias ? 1 : 0, cfg.bias_via_psum ? 1 : 0, cfg.zero_bias ? 1 : 0,
        cfg.asymmetric_activations ? 1 : 0,
        static_cast<unsigned>(cfg.input_a_zeropoint),
        static_cast<unsigned>(cfg.input_b_zeropoint),
        cfg.mvout_fp32 ? 1 : 0, cfg.fp32_scale, cfg.fp32_zero_point);
    std::printf("mvin_modes: act_2d=%d wgt_2d=%d\n",
        cfg.act_2d_mvin ? 1 : 0,
        cfg.wgt_2d_mvin ? 1 : 0);
    std::printf("tensor_bytes: activation=%u weight=%u bias=%u output=%u\n",
        a_bytes, b_bytes, bias_elems * static_cast<uint32_t>(sizeof(int32_t)), out_bytes);

    if (npu_init() != 0) {
        std::fprintf(stderr, "npu_init failed\n");
        return 1;
    }
    npu_reset();

    auto * host_a = static_cast<int8_t *>(npu_mem_alloc(a_bytes));
    auto * host_b = static_cast<int8_t *>(npu_mem_alloc(b_bytes));
    auto * host_bias = static_cast<int32_t *>(npu_mem_alloc(bias_elems * sizeof(int32_t)));
    auto * host_bias_zero = static_cast<int32_t *>(npu_mem_alloc(bias_elems * sizeof(int32_t)));
    auto * host_bias_second = static_cast<int32_t *>(npu_mem_alloc(bias_elems * sizeof(int32_t)));
    auto * host_out = static_cast<int32_t *>(npu_mem_alloc(out_bytes));
    auto * host_out_f32 = static_cast<float *>(cfg.mvout_fp32 ? npu_mem_alloc(out_elems * sizeof(float)) : nullptr);

    if (!host_a || !host_b || !host_bias || !host_bias_zero || !host_bias_second || !host_out || (cfg.mvout_fp32 && !host_out_f32)) {
        std::fprintf(stderr, "npu_mem_alloc failed\n");
        if (host_a) npu_mem_free(host_a);
        if (host_b) npu_mem_free(host_b);
        if (host_bias) npu_mem_free(host_bias);
        if (host_bias_zero) npu_mem_free(host_bias_zero);
        if (host_bias_second) npu_mem_free(host_bias_second);
        if (host_out) npu_mem_free(host_out);
        if (host_out_f32) npu_mem_free(host_out_f32);
        npu_destroy();
        return 2;
    }

    std::vector<int32_t> ref(static_cast<size_t>(out_elems), 0);
    uint64_t total_gemm_calls = 0;
    uint64_t total_mismatch = 0;

    for (int loop = 0; loop < cfg.loops; ++loop) {
        for (uint32_t i = 0; i < a_bytes; ++i) {
            host_a[i] = gen_i8(cfg.seed + static_cast<uint32_t>(loop), i, 17u);
        }
        for (uint32_t i = 0; i < b_bytes; ++i) {
            host_b[i] = gen_i8(cfg.seed + static_cast<uint32_t>(loop), i, 29u);
        }
        for (uint32_t i = 0; i < bias_elems; ++i) {
            if (cfg.bias_const_set) {
                host_bias[i] = cfg.bias_const;
            } else {
                host_bias[i] = cfg.zero_bias ? 0 : gen_i32(cfg.seed + static_cast<uint32_t>(loop), i, 53u);
            }
            host_bias_zero[i] = 0;
            host_bias_second[i] = cfg.bias_second_const_set ? cfg.bias_second_const : host_bias[i];
        }
        std::memset(host_out, 0, out_bytes);
        if (host_out_f32) {
            std::memset(host_out_f32, 0, out_elems * sizeof(float));
        }
        std::fill(ref.begin(), ref.end(), 0);
        const int32_t * ref_bias = cfg.bias_second_const_set ? host_bias_second : host_bias;

        for (int row = 0; row < cfg.n; ++row) {
            for (int col = 0; col < cfg.m; ++col) {
                int32_t acc = ref_bias[col];
                if (!cfg.enable_bias) {
                    acc = 0;
                }
                for (int kk = 0; kk < cfg.k; ++kk) {
                    const int8_t a = host_a[row * cfg.k + kk];
                    const int8_t b = host_b[kk * cfg.m + col];
                    acc += effective_a(a, cfg) * effective_b(b, cfg);
                }
                ref[static_cast<size_t>(row * cfg.m + col)] = acc;
            }
        }

        if (cfg.use_double_mvin && a_bytes == b_bytes && !cfg.act_2d_mvin && !cfg.wgt_2d_mvin) {
            MvinConfig mvin0{};
            mvin0.host_ptr = host_a;
            mvin0.sram_addr = cfg.sram_act;
            mvin0.col_num = a_bytes - 1;
            mvin0.row_num = 0;
            mvin0.sram_stride = 0;
            mvin0.dram_stride = 0;
            mvin0.precision = 1;
            mvin0.input_type = 0;
            mvin0.dest = false;
            mvin0.is_bias = false;
            mvin0.is_quant = false;
            mvin0.quant_zero = 0;
            mvin0.quant_scale = 0;
            mvin0.quant_shift = 0;

            MvinConfig mvin1 = mvin0;
            mvin1.host_ptr = host_b;
            mvin1.sram_addr = cfg.sram_wgt;
            mvin1.input_type = 1;

            std::puts("runtime_call=npu_dma_double_mvin (activation+weight)");
            npu_dma_double_mvin(&mvin0, &mvin1);
        } else {
            std::puts("runtime_call=npu_dma_mvin activation");
            npu_dma_mvin(
                host_a,
                cfg.sram_act,
                cfg.act_2d_mvin ? static_cast<uint32_t>(cfg.k - 1) : (a_bytes - 1),
                cfg.act_2d_mvin ? static_cast<uint32_t>(cfg.n - 1) : 0u,
                cfg.act_2d_mvin ? static_cast<uint16_t>(cfg.k) : 0u,
                cfg.act_2d_mvin ? static_cast<uint32_t>(cfg.k) : 0u,
                1,
                0,
                false,
                false,
                false,
                0,
                0,
                0);

            std::puts("runtime_call=npu_dma_mvin weight");
            npu_dma_mvin(
                host_b,
                cfg.sram_wgt,
                cfg.wgt_2d_mvin ? static_cast<uint32_t>(cfg.m - 1) : (b_bytes - 1),
                cfg.wgt_2d_mvin ? static_cast<uint32_t>(cfg.k - 1) : 0u,
                cfg.wgt_2d_mvin ? static_cast<uint16_t>(cfg.m) : 0u,
                cfg.wgt_2d_mvin ? static_cast<uint32_t>(cfg.m) : 0u,
                1,
                1,
                false,
                false,
                false,
                0,
                0,
                0);
        }

        if (cfg.enable_bias) {
            if (cfg.bias_clear_then_load) {
                std::puts("runtime_call=npu_dma_mvin bias->ACC clear-pass");
                mvin_bias_vector(host_bias_zero, cfg);
                std::puts("runtime_call=npu_dma_mvin bias->ACC target-pass");
                mvin_bias_vector(host_bias, cfg);
            } else {
                mvin_bias_vector(host_bias, cfg);
            }
            if (cfg.bias_second_const_set) {
                std::puts("runtime_call=npu_dma_mvin bias->ACC second-pass");
                mvin_bias_vector(host_bias_second, cfg);
            }
        }

        std::puts("runtime_call=npu_gemm_run accout_dest=ACC");
        npu_gemm_run(
            /*dataflow=*/true,
            /*int_type=*/0,
            /*optype=*/0,
            /*accout_dest=*/true,
            /*input_a_zeropoint=*/cfg.input_a_zeropoint,
            /*input_b_zeropoint=*/cfg.input_b_zeropoint,
            /*output_zeropoint=*/0,
            /*output_scale=*/1,
            /*output_scaleshift=*/0,
            /*biaspsum_addr=*/cfg.bias_acc_addr,
            /*biaspsum_stride=*/static_cast<uint16_t>(cfg.m),
            /*biaspsum_width=*/static_cast<uint8_t>(cfg.m),
            /*biaspsum_height=*/static_cast<uint8_t>(cfg.n),
            /*output_addr=*/cfg.out_acc_addr,
            /*output_stride=*/static_cast<uint16_t>(cfg.m),
            /*isaccu=*/cfg.enable_bias,
            /*relu=*/false,
            /*relu_type=*/0,
            /*is_bias=*/cfg.enable_bias ? !cfg.bias_via_psum : false,
            /*input_a_addr=*/cfg.sram_act,
            /*input_a_col_num=*/static_cast<uint16_t>(cfg.k - 1),
            /*input_a_row_num=*/static_cast<uint8_t>(cfg.n - 1),
            /*input_a_stride=*/static_cast<uint16_t>(cfg.k),
            /*input_b_addr=*/cfg.sram_wgt,
            /*input_b_col_num=*/static_cast<uint8_t>(cfg.m - 1),
            /*input_b_row_num=*/static_cast<uint16_t>(cfg.k - 1),
            /*input_b_stride=*/static_cast<uint16_t>(cfg.m),
            /*asymmetric_activations=*/cfg.asymmetric_activations);
        total_gemm_calls += 1;

        uint32_t mismatch = 0;
        float max_abs_diff = 0.0f;
        if (cfg.mvout_fp32) {
            std::puts("runtime_call=npu_dma_mvout ACC->fp32");
            // Prime the ACC->fp32 path once. On current overlay the first fp32 mvout
            // beat can intermittently come back as zero right after reload.
            const MvoutConfig acc_fp32_prime_cfg {
                host_out_f32,
                cfg.out_acc_addr,
                0,
                0,
                1,
                1,
                3,
                1,
                true,
                true,
                cfg.fp32_zero_point,
                pack_f32_scale(cfg.fp32_scale),
                false,
            };
            npu_dma_mvout_async(kAccDmaId, &acc_fp32_prime_cfg);
            npu_dma_wait_mvout(1u << kAccDmaId);

            const MvoutConfig acc_fp32_mvout_cfg {
                host_out_f32,
                cfg.out_acc_addr,
                out_elems - 1,
                0,
                static_cast<uint16_t>(out_elems),
                out_elems,
                3,
                1,
                true,
                true,
                cfg.fp32_zero_point,
                pack_f32_scale(cfg.fp32_scale),
                false,
            };
            npu_dma_mvout_async(kAccDmaId, &acc_fp32_mvout_cfg);
            npu_dma_wait_mvout(1u << kAccDmaId);
            for (uint32_t i = 0; i < out_elems; ++i) {
                const float got = host_out_f32[i];
                const float exp = static_cast<float>(ref[i] - static_cast<int32_t>(cfg.fp32_zero_point)) * cfg.fp32_scale;
                const float diff = std::fabs(got - exp);
                if (diff > 1e-3f) {
                    if (mismatch < 8) {
                        std::fprintf(stderr, "loop=%d mismatch idx=%u got=%.6f exp=%.6f diff=%.6f\n",
                            loop, i, got, exp, diff);
                    }
                    mismatch += 1;
                }
                if (diff > max_abs_diff) {
                    max_abs_diff = diff;
                }
            }
        } else {
            std::puts("runtime_call=npu_dma_mvout ACC->int32");
            const MvoutConfig acc_i32_mvout_cfg {
                host_out,
                cfg.out_acc_addr,
                out_elems - 1,
                0,
                static_cast<uint16_t>(out_elems),
                out_elems,
                1,
                1,
                true,
                false,
                0,
                0,
                false,
            };
            npu_dma_mvout_async(kAccDmaId, &acc_i32_mvout_cfg);
            npu_dma_wait_mvout(1u << kAccDmaId);
            for (uint32_t i = 0; i < out_elems; ++i) {
                const int32_t got = host_out[i];
                const int32_t exp = ref[i];
                const float diff = static_cast<float>(got > exp ? got - exp : exp - got);
                if (diff != 0.0f) {
                    if (mismatch < 8) {
                        std::fprintf(stderr, "loop=%d mismatch idx=%u got=%d exp=%d diff=%.0f\n",
                            loop, i, got, exp, diff);
                    }
                    mismatch += 1;
                }
                if (diff > max_abs_diff) {
                    max_abs_diff = diff;
                }
            }
        }
        total_mismatch += mismatch;

        std::printf("loop=%d result: mismatches=%u/%u max_abs_diff=%.6f\n",
            loop, mismatch, out_elems, max_abs_diff);
    }

    npu_mem_free(host_a);
    npu_mem_free(host_b);
    npu_mem_free(host_bias);
    npu_mem_free(host_bias_zero);
    npu_mem_free(host_bias_second);
    npu_mem_free(host_out);
    if (host_out_f32) npu_mem_free(host_out_f32);
    npu_destroy();

    std::printf("summary: gemm_calls=%llu total_mismatches=%llu\n",
        static_cast<unsigned long long>(total_gemm_calls),
        static_cast<unsigned long long>(total_mismatch));

    if (total_mismatch != 0) {
        std::puts("kv260_layer_gemm_replay_test=fail");
        return 3;
    }
    std::puts("kv260_layer_gemm_replay_test=ok");
    return 0;
}
