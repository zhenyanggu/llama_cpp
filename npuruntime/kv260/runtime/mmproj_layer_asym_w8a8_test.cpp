#include "npu_runtime.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct Config {
    int m = 96;   // output channels
    int n = 73;   // token rows
    int k = 768;  // reduction dimension
    int loops = 1;

    int tile_m = 16;
    int tile_n = 16;
    int tile_k = 768;

    uint32_t sram_act = 0x00000000;
    uint32_t sram_wgt = 0x00020000;
    uint32_t acc_addr = 0x00000000;

    uint32_t seed = 20260417u;
    uint8_t act_zp_u8 = 113;
    float act_scale = 0.0f; // <= 0 means auto from first loop activation.
    float tol = 5e-3f;
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

static float parse_f32(const char * s, float fallback) {
    if (!s || !*s) {
        return fallback;
    }
    char * end = nullptr;
    const float v = std::strtof(s, &end);
    if (!end || *end != '\0') {
        return fallback;
    }
    return v;
}

static void print_usage(const char * prog) {
    std::printf(
        "Usage: %s [options]\n"
        "  --m <int>             Output channels (default: 96)\n"
        "  --n <int>             Token rows (default: 73)\n"
        "  --k <int>             Reduction dimension (default: 768)\n"
        "  --loops <int>         Repeat loops (default: 1)\n"
        "  --tile-m <int>        Tile output channels, <=16 for SA16 (default: 16)\n"
        "  --tile-n <int>        Tile token rows, <=16 for SA16 (default: 16)\n"
        "  --tile-k <int>        Tile K (default: 768)\n"
        "  --sram-act <hex/int>  Activation SPM base (default: 0x0)\n"
        "  --sram-wgt <hex/int>  Weight SPM base (default: 0x20000)\n"
        "  --acc-addr <hex/int>  ACC base (default: 0x0)\n"
        "  --seed <uint>         Seed (default: 20260417)\n"
        "  --act-zp-u8 <uint>    Activation u8 zero-point (default: 113)\n"
        "  --act-scale <float>   Activation scale; <=0 means auto (default: auto)\n"
        "  --tol <float>         Abs tolerance (default: 0.005)\n"
        "  -h, --help            Show help\n",
        prog);
}

static Config parse_args(int argc, char ** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
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
        if (std::strcmp(argv[i], "--tile-m") == 0 && i + 1 < argc) {
            cfg.tile_m = parse_i32(argv[++i], cfg.tile_m);
            continue;
        }
        if (std::strcmp(argv[i], "--tile-n") == 0 && i + 1 < argc) {
            cfg.tile_n = parse_i32(argv[++i], cfg.tile_n);
            continue;
        }
        if (std::strcmp(argv[i], "--tile-k") == 0 && i + 1 < argc) {
            cfg.tile_k = parse_i32(argv[++i], cfg.tile_k);
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
            cfg.acc_addr = parse_u32(argv[++i], cfg.acc_addr);
            continue;
        }
        if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            cfg.seed = parse_u32(argv[++i], cfg.seed);
            continue;
        }
        if (std::strcmp(argv[i], "--act-zp-u8") == 0 && i + 1 < argc) {
            cfg.act_zp_u8 = static_cast<uint8_t>(parse_u32(argv[++i], cfg.act_zp_u8));
            continue;
        }
        if (std::strcmp(argv[i], "--act-scale") == 0 && i + 1 < argc) {
            cfg.act_scale = parse_f32(argv[++i], cfg.act_scale);
            continue;
        }
        if (std::strcmp(argv[i], "--tol") == 0 && i + 1 < argc) {
            cfg.tol = parse_f32(argv[++i], cfg.tol);
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
    if (cfg.tile_m <= 0 || cfg.tile_n <= 0 || cfg.tile_k <= 0) {
        std::fprintf(stderr, "Invalid tiles: tile_m=%d tile_n=%d tile_k=%d\n", cfg.tile_m, cfg.tile_n, cfg.tile_k);
        std::exit(2);
    }
    if (cfg.tile_m > 16 || cfg.tile_n > 16) {
        std::fprintf(stderr, "tile_m and tile_n must be <= 16 for SA16.\n");
        std::exit(2);
    }
    if (cfg.loops <= 0) {
        cfg.loops = 1;
    }
    if (cfg.tol < 0.0f) {
        cfg.tol = 0.0f;
    }
    return cfg;
}

static uint32_t mix32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7FEB352Du;
    x ^= x >> 15;
    x *= 0x846CA68Bu;
    x ^= x >> 16;
    return x;
}

static float gen_f32(uint32_t seed, uint32_t idx, uint32_t salt, float amplitude) {
    const uint32_t h = mix32(idx ^ (seed + salt * 0x9E3779B9u));
    const int32_t s = static_cast<int32_t>(h);
    const float u = static_cast<float>(s) / 2147483648.0f;
    return u * amplitude;
}

static uint32_t pack_f32_scale(float scale) {
    return static_cast<uint32_t>(std::llround(static_cast<double>(scale) * 16777216.0));
}

static float derive_activation_scale(
        const std::vector<float> & act,
        uint8_t zp_u8) {
    float min_v = act.empty() ? 0.0f : act[0];
    float max_v = act.empty() ? 0.0f : act[0];
    for (float v : act) {
        min_v = std::min(min_v, v);
        max_v = std::max(max_v, v);
    }

    const int zp = static_cast<int>(zp_u8);
    float pos_scale = 0.0f;
    float neg_scale = 0.0f;
    if (max_v > 0.0f && zp < 255) {
        pos_scale = max_v / static_cast<float>(255 - zp);
    }
    if (min_v < 0.0f && zp > 0) {
        neg_scale = (-min_v) / static_cast<float>(zp);
    }
    float s = std::max(pos_scale, neg_scale);
    if (!(s > 0.0f) || !std::isfinite(s)) {
        s = 1e-6f;
    }
    return s;
}

static void quantize_activation_to_u8(
        const std::vector<float> & act_f32,
        float act_scale,
        uint8_t zp_u8,
        std::vector<uint8_t> * act_q_u8) {
    act_q_u8->assign(act_f32.size(), 0);
    for (size_t i = 0; i < act_f32.size(); ++i) {
        int32_t q_u8 = static_cast<int32_t>(std::lrint(act_f32[i] / act_scale)) + static_cast<int32_t>(zp_u8);
        q_u8 = std::max(0, std::min(255, q_u8));
        (*act_q_u8)[i] = static_cast<uint8_t>(q_u8);
    }
}

static void quantize_weight_per_channel(
        const std::vector<float> & w_f32,
        int k,
        int m,
        std::vector<float> * w_scale,
        std::vector<int8_t> * w_q_i8,
        std::vector<int32_t> * sum_w) {
    w_scale->assign(static_cast<size_t>(m), 1.0f);
    w_q_i8->assign(static_cast<size_t>(k * m), 0);
    sum_w->assign(static_cast<size_t>(m), 0);

    for (int col = 0; col < m; ++col) {
        float max_abs = 0.0f;
        for (int kk = 0; kk < k; ++kk) {
            max_abs = std::max(max_abs, std::fabs(w_f32[static_cast<size_t>(kk * m + col)]));
        }
        const float s = max_abs > 0.0f ? (max_abs / 127.0f) : 1.0f;
        (*w_scale)[static_cast<size_t>(col)] = s;

        int32_t sum = 0;
        for (int kk = 0; kk < k; ++kk) {
            const float v = w_f32[static_cast<size_t>(kk * m + col)];
            int32_t q = static_cast<int32_t>(std::lrint(v / s));
            q = std::max(-127, std::min(127, q));
            (*w_q_i8)[static_cast<size_t>(kk * m + col)] = static_cast<int8_t>(q);
            sum += q;
        }
        (*sum_w)[static_cast<size_t>(col)] = sum;
    }
}

static void build_compensated_bias_f32(
        const std::vector<float> & model_bias_f32,
        const std::vector<float> & w_scale,
        const std::vector<int32_t> & sum_w,
        float act_scale,
        int32_t zp_i8,
        std::vector<float> * bias_comp_f32) {
    const int m = static_cast<int>(model_bias_f32.size());
    bias_comp_f32->assign(static_cast<size_t>(m), 0.0f);
    for (int col = 0; col < m; ++col) {
        const int32_t comp_i32 = -zp_i8 * sum_w[static_cast<size_t>(col)];
        const float comp_f32 = static_cast<float>(comp_i32) * act_scale * w_scale[static_cast<size_t>(col)];
        (*bias_comp_f32)[static_cast<size_t>(col)] = model_bias_f32[static_cast<size_t>(col)] + comp_f32;
    }
}

static void reference_mmproj_layer(
        const std::vector<uint8_t> & act_q_u8,
        const std::vector<int8_t> & w_q_i8,
        const std::vector<float> & w_scale,
        const std::vector<float> & bias_comp_f32,
        int n,
        int m,
        int k,
        float act_scale,
        std::vector<float> * out) {
    out->assign(static_cast<size_t>(n * m), 0.0f);
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < m; ++col) {
            int32_t acc = 0;
            for (int kk = 0; kk < k; ++kk) {
                const int32_t a = static_cast<int32_t>(act_q_u8[static_cast<size_t>(row * k + kk)]) - 128;
                const int32_t w = static_cast<int32_t>(w_q_i8[static_cast<size_t>(kk * m + col)]);
                acc += a * w;
            }
            const float v = static_cast<float>(acc) * act_scale * w_scale[static_cast<size_t>(col)] +
                bias_comp_f32[static_cast<size_t>(col)];
            (*out)[static_cast<size_t>(row * m + col)] = v;
        }
    }
}

static void pack_activation_tile(
        const std::vector<uint8_t> & act_q_u8,
        int n,
        int k,
        int n0,
        int tn,
        int k0,
        int tk,
        uint8_t * tile_buf) {
    for (int r = 0; r < tn; ++r) {
        const int src_row = n0 + r;
        const uint8_t * src = &act_q_u8[static_cast<size_t>(src_row * k + k0)];
        std::memcpy(&tile_buf[static_cast<size_t>(r * tk)], src, static_cast<size_t>(tk));
    }
    (void)n;
}

static void pack_weight_tile(
        const std::vector<int8_t> & w_q_i8,
        int m,
        int k0,
        int tk,
        int m0,
        int tm,
        int8_t * tile_buf) {
    for (int kk = 0; kk < tk; ++kk) {
        const int src_k = k0 + kk;
        for (int col = 0; col < tm; ++col) {
            tile_buf[static_cast<size_t>(kk * tm + col)] =
                w_q_i8[static_cast<size_t>(src_k * m + (m0 + col))];
        }
    }
}

int main(int argc, char ** argv) {
    const Config cfg = parse_args(argc, argv);

    if (npu_init() != 0) {
        std::fprintf(stderr, "npu_init failed\n");
        return 1;
    }
    npu_reset();

    const size_t max_act_tile_bytes = static_cast<size_t>(cfg.tile_n * cfg.tile_k);
    const size_t max_wgt_tile_bytes = static_cast<size_t>(cfg.tile_k * cfg.tile_m);
    const size_t max_out_tile_elems = static_cast<size_t>(cfg.tile_n * cfg.tile_m);

    auto * tile_act = static_cast<uint8_t *>(npu_mem_alloc(max_act_tile_bytes));
    auto * tile_wgt = static_cast<int8_t *>(npu_mem_alloc(max_wgt_tile_bytes));
    auto * tile_out_f32 = static_cast<float *>(npu_mem_alloc(max_out_tile_elems * sizeof(float)));
    if (!tile_act || !tile_wgt || !tile_out_f32) {
        std::fprintf(stderr, "npu_mem_alloc failed for tile buffers\n");
        if (tile_act) npu_mem_free(tile_act);
        if (tile_wgt) npu_mem_free(tile_wgt);
        if (tile_out_f32) npu_mem_free(tile_out_f32);
        npu_destroy();
        return 2;
    }

    std::vector<float> weight_f32(static_cast<size_t>(cfg.k * cfg.m), 0.0f);
    std::vector<float> model_bias_f32(static_cast<size_t>(cfg.m), 0.0f);
    for (size_t i = 0; i < weight_f32.size(); ++i) {
        weight_f32[i] = gen_f32(cfg.seed, static_cast<uint32_t>(i), 101u, 0.9f);
    }
    for (int col = 0; col < cfg.m; ++col) {
        model_bias_f32[static_cast<size_t>(col)] = gen_f32(cfg.seed, static_cast<uint32_t>(col), 211u, 0.25f);
    }

    std::vector<float> w_scale;
    std::vector<int8_t> w_q_i8;
    std::vector<int32_t> sum_w;
    quantize_weight_per_channel(weight_f32, cfg.k, cfg.m, &w_scale, &w_q_i8, &sum_w);

    std::vector<float> activation_f32(static_cast<size_t>(cfg.n * cfg.k), 0.0f);
    std::vector<uint8_t> act_q_u8(static_cast<size_t>(cfg.n * cfg.k), 0);
    std::vector<float> bias_comp_f32;
    std::vector<float> ref(static_cast<size_t>(cfg.n * cfg.m), 0.0f);
    std::vector<float> out_npu(static_cast<size_t>(cfg.n * cfg.m), 0.0f);

    const int32_t act_zp_i8 = static_cast<int32_t>(cfg.act_zp_u8) - 128;
    float act_scale = cfg.act_scale;
    const bool use_auto_scale = !(act_scale > 0.0f);

    uint64_t total_mismatch = 0;
    float global_max_abs_diff = 0.0f;

    std::printf(
        "kv260_mmproj_layer_asym_w8a8_test: m=%d n=%d k=%d loops=%d tile_m=%d tile_n=%d tile_k=%d seed=%u act_zp_u8=%u act_scale=%s tol=%.6f\n",
        cfg.m, cfg.n, cfg.k, cfg.loops, cfg.tile_m, cfg.tile_n, cfg.tile_k, cfg.seed,
        static_cast<unsigned>(cfg.act_zp_u8),
        use_auto_scale ? "auto" : "fixed",
        cfg.tol);

    for (int loop = 0; loop < cfg.loops; ++loop) {
        for (size_t i = 0; i < activation_f32.size(); ++i) {
            activation_f32[i] = gen_f32(cfg.seed + static_cast<uint32_t>(loop), static_cast<uint32_t>(i), 307u, 1.4f);
        }
        if (use_auto_scale && loop == 0) {
            act_scale = derive_activation_scale(activation_f32, cfg.act_zp_u8);
        }
        if (!(act_scale > 0.0f) || !std::isfinite(act_scale)) {
            std::fprintf(stderr, "Invalid activation scale: %.8g\n", act_scale);
            npu_mem_free(tile_act);
            npu_mem_free(tile_wgt);
            npu_mem_free(tile_out_f32);
            npu_destroy();
            return 2;
        }
        if (loop == 0) {
            build_compensated_bias_f32(model_bias_f32, w_scale, sum_w, act_scale, act_zp_i8, &bias_comp_f32);
        }

        quantize_activation_to_u8(activation_f32, act_scale, cfg.act_zp_u8, &act_q_u8);
        reference_mmproj_layer(act_q_u8, w_q_i8, w_scale, bias_comp_f32, cfg.n, cfg.m, cfg.k, act_scale, &ref);
        std::fill(out_npu.begin(), out_npu.end(), 0.0f);

        const uint32_t mvout_f32_scale = pack_f32_scale(act_scale);

        for (int n0 = 0; n0 < cfg.n; n0 += cfg.tile_n) {
            const int tn = std::min(cfg.tile_n, cfg.n - n0);
            for (int m0 = 0; m0 < cfg.m; m0 += cfg.tile_m) {
                const int tm = std::min(cfg.tile_m, cfg.m - m0);
                bool first_k = true;

                for (int k0 = 0; k0 < cfg.k; k0 += cfg.tile_k) {
                    const int tk = std::min(cfg.tile_k, cfg.k - k0);
                    pack_activation_tile(act_q_u8, cfg.n, cfg.k, n0, tn, k0, tk, tile_act);
                    pack_weight_tile(w_q_i8, cfg.m, k0, tk, m0, tm, tile_wgt);

                    const MvinConfig act_mvin_cfg {
                        tile_act,
                        cfg.sram_act,
                        static_cast<uint32_t>(tn * tk - 1),
                        0,
                        0,
                        0,
                        1,
                        0,
                        false,
                        false,
                        false,
                        0,
                        0,
                        0,
                    };
                    const MvinConfig wgt_mvin_cfg {
                        tile_wgt,
                        cfg.sram_wgt,
                        static_cast<uint32_t>(tk * tm - 1),
                        0,
                        0,
                        0,
                        1,
                        1,
                        false,
                        false,
                        false,
                        0,
                        0,
                        0,
                    };

                    npu_dma_mvin_async(0, &act_mvin_cfg);
                    npu_dma_mvin_async(1, &wgt_mvin_cfg);
                    npu_dma_wait_mvin((1u << 0) | (1u << 1));

                    npu_gemm_run(
                        /*dataflow=*/true,
                        /*int_type=*/0,
                        /*optype=*/0,
                        /*accout_dest=*/true,
                        /*input_a_zeropoint=*/0,
                        /*input_b_zeropoint=*/0,
                        /*output_zeropoint=*/0,
                        /*output_scale=*/1,
                        /*output_scaleshift=*/0,
                        /*biaspsum_addr=*/cfg.acc_addr,
                        /*biaspsum_stride=*/static_cast<uint16_t>(tm),
                        /*biaspsum_width=*/static_cast<uint8_t>(tm),
                        /*biaspsum_height=*/static_cast<uint8_t>(tn),
                        /*output_addr=*/cfg.acc_addr,
                        /*output_stride=*/static_cast<uint16_t>(tm),
                        /*isaccu=*/!first_k,
                        /*relu=*/false,
                        /*relu_type=*/0,
                        /*is_bias=*/false,
                        /*input_a_addr=*/cfg.sram_act,
                        /*input_a_col_num=*/static_cast<uint16_t>(tk - 1),
                        /*input_a_row_num=*/static_cast<uint8_t>(tn - 1),
                        /*input_a_stride=*/static_cast<uint16_t>(tk),
                        /*input_b_addr=*/cfg.sram_wgt,
                        /*input_b_col_num=*/static_cast<uint8_t>(tm - 1),
                        /*input_b_row_num=*/static_cast<uint16_t>(tk - 1),
                        /*input_b_stride=*/static_cast<uint16_t>(tm),
                        /*asymmetric_activations=*/true);
                    first_k = false;
                }

                const uint32_t out_tile_elems = static_cast<uint32_t>(tn * tm);
                npu_dma_mvout(
                    tile_out_f32,
                    cfg.acc_addr,
                    out_tile_elems - 1,
                    0,
                    out_tile_elems,
                    out_tile_elems,
                    /*precision=*/3,
                    /*output_type=*/1,
                    /*source=*/true,
                    /*is_quant=*/true,
                    /*quant_zero=*/0,
                    /*f32_scale=*/mvout_f32_scale);

                for (int r = 0; r < tn; ++r) {
                    for (int c = 0; c < tm; ++c) {
                        const size_t local_idx = static_cast<size_t>(r * tm + c);
                        const int global_row = n0 + r;
                        const int global_col = m0 + c;
                        const float v = tile_out_f32[local_idx] * w_scale[static_cast<size_t>(global_col)] +
                            bias_comp_f32[static_cast<size_t>(global_col)];
                        out_npu[static_cast<size_t>(global_row * cfg.m + global_col)] = v;
                    }
                }
            }
        }

        uint32_t mismatch = 0;
        float loop_max_abs = 0.0f;
        for (size_t i = 0; i < out_npu.size(); ++i) {
            const float got = out_npu[i];
            const float exp = ref[i];
            const float diff = std::fabs(got - exp);
            if (diff > cfg.tol) {
                if (mismatch < 8) {
                    const int row = static_cast<int>(i / static_cast<size_t>(cfg.m));
                    const int col = static_cast<int>(i % static_cast<size_t>(cfg.m));
                    std::fprintf(stderr,
                        "loop=%d mismatch row=%d col=%d got=%.8f exp=%.8f diff=%.8f\n",
                        loop, row, col, got, exp, diff);
                }
                mismatch += 1;
            }
            loop_max_abs = std::max(loop_max_abs, diff);
        }

        total_mismatch += mismatch;
        global_max_abs_diff = std::max(global_max_abs_diff, loop_max_abs);
        std::printf("loop=%d result: mismatches=%u/%zu max_abs_diff=%.8f act_scale=%.9g\n",
            loop, mismatch, out_npu.size(), loop_max_abs, act_scale);
    }

    npu_mem_free(tile_act);
    npu_mem_free(tile_wgt);
    npu_mem_free(tile_out_f32);
    npu_destroy();

    std::printf(
        "summary: total_mismatches=%llu max_abs_diff=%.8f act_scale=%.9g act_zp_i8=%d\n",
        static_cast<unsigned long long>(total_mismatch),
        global_max_abs_diff,
        act_scale,
        act_zp_i8);

    if (total_mismatch != 0) {
        std::puts("kv260_mmproj_layer_asym_w8a8_test=fail");
        return 3;
    }
    std::puts("kv260_mmproj_layer_asym_w8a8_test=ok");
    return 0;
}
