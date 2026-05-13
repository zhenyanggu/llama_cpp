#include "npu_runtime.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kSpmBytes = 512u * 1024u;
constexpr uint32_t kAccBytes = 512u * 1024u;
constexpr uint32_t kSaTile = 16;
constexpr uint32_t kPlanDimMax = 240;
constexpr uint32_t kPlanKMax = 4096;
constexpr uint32_t kSpmAct = 0x00000000;
constexpr uint32_t kAccOut = 0x00000000;

struct LayerShape {
    const char* name;
    uint32_t k;
    uint32_t m;
    uint32_t n;
};

struct TileShape {
    uint32_t m;
    uint32_t n;
    uint32_t k;
};

struct Timings {
    uint64_t weight_pre_ns = 0;
    uint64_t act_quant_ns = 0;
    uint64_t weight_pack_ns = 0;
    uint64_t bias_prepare_ns = 0;
    uint64_t scale_prepare_ns = 0;
    uint64_t dma_in_pair_ns = 0;
    uint64_t dma_in_bias_ns = 0;
    uint64_t gemm_ns = 0;
    uint64_t dma_in_scale_ns = 0;
    uint64_t dma_out_ns = 0;
    uint64_t total_ns = 0;
};

struct LayerResult {
    uint64_t gemm_calls = 0;
    uint64_t mn_tiles = 0;
    uint64_t k_tiles = 0;
    uint64_t mismatches = 0;
    uint64_t samples_checked = 0;
    double checksum = 0.0;
    Timings t;
};

uint64_t now_ns() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

uint32_t align_up(uint32_t value, uint32_t alignment) {
    return (value + alignment - 1u) & ~(alignment - 1u);
}

int32_t pack_q8_24(float scale) {
    const double scaled = std::nearbyint(static_cast<double>(scale) * static_cast<double>(1u << 24));
    if (scaled > static_cast<double>(std::numeric_limits<int32_t>::max())) {
        return std::numeric_limits<int32_t>::max();
    }
    if (scaled < static_cast<double>(std::numeric_limits<int32_t>::min())) {
        return std::numeric_limits<int32_t>::min();
    }
    return static_cast<int32_t>(scaled);
}

uint32_t env_u32(const char* name, uint32_t fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 0);
    if (end == value || *end != '\0') {
        return fallback;
    }
    return static_cast<uint32_t>(parsed);
}

TileShape env_forced_tile() {
    const uint32_t tm = env_u32("NPU_NETWORK_GEMM_FORCE_TILE_M", 0);
    const uint32_t tn = env_u32("NPU_NETWORK_GEMM_FORCE_TILE_N", 0);
    const uint32_t tk = env_u32("NPU_NETWORK_GEMM_FORCE_TILE_K", 0);
    return {tm, tn, tk};
}

int32_t env_i32(const char* name, int32_t fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0') {
        return fallback;
    }
    return static_cast<int32_t>(parsed);
}

float activation_value(uint32_t layer_idx, uint32_t k, uint32_t n) {
    const int base = static_cast<int>((k * 17u + n * 31u + layer_idx * 23u) % 257u) - 128;
    const int ripple = static_cast<int>((k + 3u * n + layer_idx) % 11u) - 5;
    return static_cast<float>(base) * 0.015625f + static_cast<float>(ripple) * 0.001953125f;
}

int8_t weight_q_value(uint32_t layer_idx, uint32_t k, uint32_t m) {
    const int value = static_cast<int>((k * 5u + m * 11u + layer_idx * 19u) % 255u) - 127;
    return static_cast<int8_t>(value);
}

float weight_scale_value(uint32_t layer_idx, uint32_t m) {
    return 0.0020f + static_cast<float>((m + layer_idx * 3u) % 17u) * 0.00005f;
}

void compute_activation_quant_params(const LayerShape& layer, uint32_t layer_idx, float* scale, uint8_t* zero_point) {
    float min_v = std::numeric_limits<float>::max();
    float max_v = -std::numeric_limits<float>::max();
    for (uint32_t n = 0; n < layer.n; ++n) {
        for (uint32_t k = 0; k < layer.k; ++k) {
            const float v = activation_value(layer_idx, k, n);
            min_v = std::min(min_v, v);
            max_v = std::max(max_v, v);
        }
    }

    float s = 1.0f;
    int32_t zp = 0;
    if (max_v > min_v) {
        s = (max_v - min_v) / 255.0f;
        if (s < 1e-8f) {
            s = 1e-8f;
        }
        zp = static_cast<int32_t>(std::lrint(-min_v / s));
        zp = std::max(0, std::min(255, zp));
    }
    *scale = s;
    *zero_point = static_cast<uint8_t>(zp);
}

TileShape choose_tile(const LayerShape& layer) {
    const TileShape forced = env_forced_tile();
    if (forced.m > 0 && forced.n > 0 && forced.k > 0) {
        return {
            std::min(forced.m, layer.m),
            std::min(forced.n, layer.n),
            std::min(forced.k, layer.k),
        };
    }

    const uint32_t tk = std::min(layer.k, kPlanKMax);
    TileShape best{16, 16, tk};
    uint64_t best_spm = 0;
    uint64_t best_area = 0;

    const uint32_t max_m = std::min(layer.m, kPlanDimMax);
    const uint32_t max_n = std::min(layer.n, kPlanDimMax);
    for (uint32_t tm = kSaTile; tm <= max_m; tm += kSaTile) {
        for (uint32_t tn = kSaTile; tn <= max_n; tn += kSaTile) {
            const uint64_t spm = static_cast<uint64_t>(tm + tn) * tk;
            const uint64_t out_bytes = static_cast<uint64_t>(tm) * tn * sizeof(int32_t);
            const uint64_t bias_bytes = static_cast<uint64_t>(tm) * sizeof(int32_t);
            const uint64_t acc_need = out_bytes * 3u + bias_bytes;
            if (spm > kSpmBytes || acc_need > kAccBytes) {
                continue;
            }
            const uint64_t area = static_cast<uint64_t>(tm) * tn;
            if (spm > best_spm || (spm == best_spm && area > best_area)) {
                best = {tm, tn, tk};
                best_spm = spm;
                best_area = area;
            }
        }
    }
    return best;
}

std::vector<std::pair<uint32_t, uint32_t>> make_samples(const LayerShape& layer, uint32_t count) {
    std::vector<std::pair<uint32_t, uint32_t>> samples;
    auto add = [&](uint32_t m, uint32_t n) {
        samples.emplace_back(std::min(m, layer.m - 1), std::min(n, layer.n - 1));
    };
    add(0, 0);
    add(layer.m - 1, 0);
    add(0, layer.n - 1);
    add(layer.m - 1, layer.n - 1);
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t m = (i * 97u + 13u) % layer.m;
        const uint32_t n = (i * 193u + 29u) % layer.n;
        add(m, n);
    }
    std::sort(samples.begin(), samples.end());
    samples.erase(std::unique(samples.begin(), samples.end()), samples.end());
    return samples;
}

float golden_value(
    const std::vector<int8_t>& weights,
    const std::vector<float>& weight_scales,
    const LayerShape& layer,
    uint32_t layer_idx,
    uint32_t m,
    uint32_t n,
    float act_scale,
    uint8_t act_zero_point) {
    int64_t acc = 0;
    for (uint32_t k = 0; k < layer.k; ++k) {
        const float v = activation_value(layer_idx, k, n);
        int32_t q = static_cast<int32_t>(std::lrint(v / act_scale)) + static_cast<int32_t>(act_zero_point);
        q = std::max(0, std::min(255, q));
        acc += static_cast<int64_t>(q - static_cast<int32_t>(act_zero_point)) *
               static_cast<int64_t>(weights[static_cast<size_t>(k) * layer.m + m]);
    }
    return static_cast<float>(acc) * act_scale * weight_scales[m];
}

bool alloc_or_fail(void** ptr, size_t bytes, const char* label) {
    *ptr = npu_mem_alloc(bytes);
    if (*ptr == nullptr) {
        std::fprintf(stderr, "npu_mem_alloc failed for %s bytes=%zu\n", label, bytes);
        return false;
    }
    return true;
}

LayerResult run_layer(const LayerShape& layer, uint32_t layer_idx, uint32_t verify_samples) {
    LayerResult result;
    const uint64_t total_start = now_ns();
    const TileShape tile = choose_tile(layer);
    const std::vector<std::pair<uint32_t, uint32_t>> samples = make_samples(layer, verify_samples);
    std::vector<uint8_t> sample_seen(samples.size(), 0);

    float act_scale = 1.0f;
    uint8_t act_zero_point = 128;
    compute_activation_quant_params(layer, layer_idx, &act_scale, &act_zero_point);
    const int32_t forced_act_zp = env_i32("NPU_NETWORK_GEMM_FORCE_ACT_ZP_U8", -1);
    if (forced_act_zp >= 0 && forced_act_zp <= 255) {
        act_zero_point = static_cast<uint8_t>(forced_act_zp);
    }
    const int32_t act_zp_i8 = static_cast<int32_t>(act_zero_point) - 128;
    const bool software_compensation = env_u32("NPU_NETWORK_GEMM_SOFTWARE_COMPENSATION", 0) != 0;

    const uint64_t weight_pre_start = now_ns();
    std::vector<int8_t> weights(static_cast<size_t>(layer.k) * layer.m);
    std::vector<int32_t> sum_w(layer.m, 0);
    std::vector<float> weight_scales(layer.m, 1.0f);
    for (uint32_t m = 0; m < layer.m; ++m) {
        weight_scales[m] = weight_scale_value(layer_idx, m);
    }
    for (uint32_t k = 0; k < layer.k; ++k) {
        for (uint32_t m = 0; m < layer.m; ++m) {
            const int8_t q = weight_q_value(layer_idx, k, m);
            weights[static_cast<size_t>(k) * layer.m + m] = q;
            sum_w[m] += static_cast<int32_t>(q);
        }
    }
    result.t.weight_pre_ns += now_ns() - weight_pre_start;

    const size_t max_act_bytes = static_cast<size_t>(tile.n) * tile.k;
    const size_t max_weight_bytes = static_cast<size_t>(tile.k) * tile.m;
    const size_t max_out_elems = static_cast<size_t>(tile.n) * tile.m;
    const size_t max_out_bytes = max_out_elems * sizeof(float);
    const size_t max_bias_bytes = static_cast<size_t>(tile.m) * sizeof(int32_t);
    const size_t max_scale_bytes = static_cast<size_t>(tile.m) * sizeof(int32_t);

    void* act_buf = nullptr;
    void* weight_buf = nullptr;
    void* out_buf = nullptr;
    void* bias_buf = nullptr;
    void* scale_buf = nullptr;
    if (!alloc_or_fail(&act_buf, max_act_bytes, "activation") ||
        !alloc_or_fail(&weight_buf, max_weight_bytes, "weight") ||
        !alloc_or_fail(&out_buf, max_out_bytes, "output") ||
        !alloc_or_fail(&bias_buf, std::max(max_bias_bytes, max_scale_bytes), "bias/scale") ||
        !alloc_or_fail(&scale_buf, max_scale_bytes, "scale")) {
        if (act_buf) npu_mem_free(act_buf);
        if (weight_buf) npu_mem_free(weight_buf);
        if (out_buf) npu_mem_free(out_buf);
        if (bias_buf) npu_mem_free(bias_buf);
        if (scale_buf) npu_mem_free(scale_buf);
        result.mismatches = 1;
        return result;
    }

    auto* act_bytes = static_cast<int8_t*>(act_buf);
    auto* weight_bytes = static_cast<int8_t*>(weight_buf);
    auto* out_f32 = static_cast<float*>(out_buf);
    auto* bias_i32 = static_cast<int32_t*>(bias_buf);
    auto* scale_i32 = static_cast<int32_t*>(scale_buf);

    for (uint32_t n0 = 0; n0 < layer.n; n0 += tile.n) {
        const uint32_t tn = std::min(tile.n, layer.n - n0);
        for (uint32_t m0 = 0; m0 < layer.m; m0 += tile.m) {
            const uint32_t tm = std::min(tile.m, layer.m - m0);
            ++result.mn_tiles;

            const size_t out_elems = static_cast<size_t>(tn) * tm;
            const uint32_t out_bytes_i32 = static_cast<uint32_t>(out_elems * sizeof(int32_t));
            const uint32_t acc_scratch = align_up(kAccOut + out_bytes_i32, 4);
            const uint32_t acc_bias = align_up(acc_scratch + out_bytes_i32, 4);
            const uint32_t acc_scale = align_up(acc_bias + tm * sizeof(int32_t), 4);
            if (acc_scale + tm * sizeof(int32_t) > kAccBytes) {
                std::fprintf(stderr, "[%s] ACC layout overflow tm=%u tn=%u\n", layer.name, tm, tn);
                ++result.mismatches;
                continue;
            }

            bool first_k = true;
            for (uint32_t k0 = 0; k0 < layer.k; k0 += tile.k) {
                const uint32_t tk = std::min(tile.k, layer.k - k0);
                ++result.k_tiles;
                ++result.gemm_calls;

                const size_t act_count = static_cast<size_t>(tn) * tk;
                const size_t weight_count = static_cast<size_t>(tk) * tm;
                const uint32_t spm_weight = align_up(static_cast<uint32_t>(act_count), 32);
                if (spm_weight + weight_count > kSpmBytes) {
                    std::fprintf(stderr, "[%s] SPM layout overflow tm=%u tn=%u tk=%u\n", layer.name, tm, tn, tk);
                    ++result.mismatches;
                    continue;
                }

                const uint64_t act_start = now_ns();
                for (uint32_t n = 0; n < tn; ++n) {
                    for (uint32_t k = 0; k < tk; ++k) {
                        const float v = activation_value(layer_idx, k0 + k, n0 + n);
                        int32_t q = static_cast<int32_t>(std::lrint(v / act_scale)) + static_cast<int32_t>(act_zero_point);
                        q = std::max(0, std::min(255, q));
                        act_bytes[static_cast<size_t>(n) * tk + k] = static_cast<int8_t>(q);
                    }
                }
                result.t.act_quant_ns += now_ns() - act_start;

                const uint64_t weight_pack_start = now_ns();
                for (uint32_t k = 0; k < tk; ++k) {
                    const int8_t* src = weights.data() + static_cast<size_t>(k0 + k) * layer.m + m0;
                    std::memcpy(weight_bytes + static_cast<size_t>(k) * tm, src, tm);
                }
                result.t.weight_pack_ns += now_ns() - weight_pack_start;

                MvinConfig act_cfg{};
                act_cfg.host_ptr = act_buf;
                act_cfg.sram_addr = kSpmAct;
                act_cfg.col_num = static_cast<uint32_t>(act_count - 1);
                act_cfg.row_num = 0;
                act_cfg.precision = 1;
                act_cfg.input_type = 0;

                MvinConfig weight_cfg{};
                weight_cfg.host_ptr = weight_buf;
                weight_cfg.sram_addr = spm_weight;
                weight_cfg.col_num = static_cast<uint32_t>(weight_count - 1);
                weight_cfg.row_num = 0;
                weight_cfg.precision = 1;
                weight_cfg.input_type = 1;

                const uint64_t dma_pair_start = now_ns();
                npu_dma_mvin_async(0, &act_cfg);
                npu_dma_mvin_async(1, &weight_cfg);
                npu_dma_wait_mvin((1u << 0) | (1u << 1));
                result.t.dma_in_pair_ns += now_ns() - dma_pair_start;

                if (first_k && !software_compensation) {
                    const uint64_t bias_prepare_start = now_ns();
                    for (uint32_t m = 0; m < tm; ++m) {
                        bias_i32[m] = -act_zp_i8 * sum_w[m0 + m];
                    }
                    result.t.bias_prepare_ns += now_ns() - bias_prepare_start;

                    const uint64_t bias_dma_start = now_ns();
                    npu_dma_mvin(
                        bias_buf, acc_bias, tm - 1, 0, 0, 0,
                        1, 2, true, true, false, 0, 0, 0);
                    result.t.dma_in_bias_ns += now_ns() - bias_dma_start;
                }

                const uint64_t gemm_start = now_ns();
                npu_gemm_plan_run_ex(
                    kSpmAct,
                    spm_weight,
                    kAccOut,
                    acc_scratch,
                    acc_bias,
                    static_cast<uint16_t>(tn),
                    static_cast<uint16_t>(tm),
                    static_cast<uint16_t>(tk),
                    static_cast<uint16_t>(tk),
                    static_cast<uint16_t>(tm),
                    static_cast<uint16_t>(tm),
                    static_cast<uint16_t>(tm),
                    first_k && !software_compensation,
                    !first_k,
                    true);
                result.t.gemm_ns += now_ns() - gemm_start;
                first_k = false;
            }

            const uint64_t scale_prepare_start = now_ns();
            for (uint32_t m = 0; m < tm; ++m) {
                scale_i32[m] = pack_q8_24(act_scale * weight_scales[m0 + m]);
            }
            result.t.scale_prepare_ns += now_ns() - scale_prepare_start;

            const uint64_t scale_dma_start = now_ns();
            npu_dma_mvin(
                scale_buf, acc_scale, static_cast<uint32_t>(tm - 1), 0, static_cast<uint16_t>(tm), static_cast<uint32_t>(tm),
                1, 2, true, false, false, 0, 0, 0);
            result.t.dma_in_scale_ns += now_ns() - scale_dma_start;

            std::memset(out_f32, 0, out_elems * sizeof(float));
            const uint64_t mvout_start = now_ns();
            npu_dma_mvout_ex(
                out_buf,
                kAccOut,
                static_cast<uint32_t>(tm - 1),
                static_cast<uint32_t>(tn - 1),
                static_cast<uint16_t>(tm),
                static_cast<uint32_t>(tm),
                3,
                1,
                true,
                true,
                0,
                acc_scale,
                true);
            result.t.dma_out_ns += now_ns() - mvout_start;

            if (software_compensation) {
                for (uint32_t n = 0; n < tn; ++n) {
                    for (uint32_t m = 0; m < tm; ++m) {
                        out_f32[static_cast<size_t>(n) * tm + m] +=
                            static_cast<float>(-act_zp_i8 * sum_w[m0 + m]) *
                            act_scale * weight_scales[m0 + m];
                    }
                }
            }

            for (size_t i = 0; i < out_elems; ++i) {
                result.checksum += static_cast<double>(out_f32[i]);
            }

            for (size_t i = 0; i < samples.size(); ++i) {
                const uint32_t sm = samples[i].first;
                const uint32_t sn = samples[i].second;
                if (sm < m0 || sm >= m0 + tm || sn < n0 || sn >= n0 + tn) {
                    continue;
                }
                sample_seen[i] = 1;
                ++result.samples_checked;
                const float got = out_f32[static_cast<size_t>(sn - n0) * tm + (sm - m0)];
                const float expect = golden_value(weights, weight_scales, layer, layer_idx, sm, sn, act_scale, act_zero_point);
                const float abs_diff = std::fabs(got - expect);
                const float tol = 0.05f + std::fabs(expect) * 0.002f;
                if (abs_diff > tol) {
                    if (result.mismatches < 16) {
                        std::fprintf(
                            stderr,
                            "[%s] sample mismatch m=%u n=%u expect=%.6f got=%.6f diff=%.6f tol=%.6f\n",
                            layer.name, sm, sn, expect, got, abs_diff, tol);
                    }
                    ++result.mismatches;
                }
            }
        }
    }

    for (uint8_t seen : sample_seen) {
        if (!seen) {
            ++result.mismatches;
        }
    }

    result.t.total_ns = now_ns() - total_start;

    std::printf(
        "layer_result name=%s M=%u N=%u K=%u tile_m=%u tile_n=%u tile_k=%u "
        "act_scale=%.9g act_zp_u8=%u act_zp_i8=%d mn_tiles=%llu k_tile_visits=%llu gemm_calls=%llu "
        "samples=%llu mismatches=%llu checksum=%.6f\n",
        layer.name,
        layer.m,
        layer.n,
        layer.k,
        tile.m,
        tile.n,
        tile.k,
        act_scale,
        static_cast<unsigned>(act_zero_point),
        act_zp_i8,
        static_cast<unsigned long long>(result.mn_tiles),
        static_cast<unsigned long long>(result.k_tiles),
        static_cast<unsigned long long>(result.gemm_calls),
        static_cast<unsigned long long>(result.samples_checked),
        static_cast<unsigned long long>(result.mismatches),
        result.checksum);

    const double ops = 2.0 * static_cast<double>(layer.m) * static_cast<double>(layer.n) * static_cast<double>(layer.k);
    const double total_s = static_cast<double>(result.t.total_ns) / 1.0e9;
    const double gemm_s = static_cast<double>(result.t.gemm_ns) / 1.0e9;
    std::printf(
        "layer_timing name=%s total_ms=%.3f gemm_ms=%.3f effective_gops_total=%.3f effective_gops_gemm=%.3f "
        "weight_pre_ms=%.3f act_quant_ms=%.3f weight_pack_ms=%.3f bias_prepare_ms=%.3f scale_prepare_ms=%.3f "
        "dma_in_pair_ms=%.3f dma_in_bias_ms=%.3f dma_in_scale_ms=%.3f dma_out_ms=%.3f\n",
        layer.name,
        total_s * 1000.0,
        gemm_s * 1000.0,
        total_s > 0.0 ? ops / total_s / 1.0e9 : 0.0,
        gemm_s > 0.0 ? ops / gemm_s / 1.0e9 : 0.0,
        static_cast<double>(result.t.weight_pre_ns) / 1.0e6,
        static_cast<double>(result.t.act_quant_ns) / 1.0e6,
        static_cast<double>(result.t.weight_pack_ns) / 1.0e6,
        static_cast<double>(result.t.bias_prepare_ns) / 1.0e6,
        static_cast<double>(result.t.scale_prepare_ns) / 1.0e6,
        static_cast<double>(result.t.dma_in_pair_ns) / 1.0e6,
        static_cast<double>(result.t.dma_in_bias_ns) / 1.0e6,
        static_cast<double>(result.t.dma_in_scale_ns) / 1.0e6,
        static_cast<double>(result.t.dma_out_ns) / 1.0e6);

    npu_mem_free(act_buf);
    npu_mem_free(weight_buf);
    npu_mem_free(out_buf);
    npu_mem_free(bias_buf);
    npu_mem_free(scale_buf);
    return result;
}

} // namespace

int main() {
    std::puts("kv260_network_gemm_layers_test: gemm_plan network-like W8A8 layers");

    if (npu_init() != 0) {
        std::fprintf(stderr, "npu_init failed\n");
        return 1;
    }
    npu_reset();

    const uint32_t verify_samples = env_u32("NPU_NETWORK_GEMM_VERIFY_SAMPLES", 64);
    const int32_t only_layer = env_i32("NPU_NETWORK_GEMM_ONLY_LAYER", -1);
    const uint32_t custom_k = env_u32("NPU_NETWORK_GEMM_CUSTOM_K", 0);
    const uint32_t custom_m = env_u32("NPU_NETWORK_GEMM_CUSTOM_M", 0);
    const uint32_t custom_n = env_u32("NPU_NETWORK_GEMM_CUSTOM_N", 0);
    const LayerShape layers[] = {
        {"q8w_3072x768_f32_3072x1024", 3072, 768, 1024},
        {"q8w_768x768_f32_768x1024", 768, 768, 1024},
        {"q8w_768x320_f32_768x1024", 768, 320, 1024},
        {"q8w_768x3072_f32_768x1024", 768, 3072, 1024},
        {"q8w_12288x960_f32_12288x64", 12288, 960, 64},
    };

    uint64_t total_mismatches = 0;
    uint64_t total_gemm_calls = 0;
    const uint64_t suite_start = now_ns();
    try {
        if (custom_k > 0 && custom_m > 0 && custom_n > 0) {
            const LayerShape custom_layer = {"custom_env_shape", custom_k, custom_m, custom_n};
            LayerResult result = run_layer(custom_layer, 0, verify_samples);
            total_mismatches += result.mismatches;
            total_gemm_calls += result.gemm_calls;
        } else {
            for (uint32_t i = 0; i < sizeof(layers) / sizeof(layers[0]); ++i) {
                if (only_layer >= 0 && static_cast<int32_t>(i) != only_layer) {
                    continue;
                }
                LayerResult result = run_layer(layers[i], i, verify_samples);
                total_mismatches += result.mismatches;
                total_gemm_calls += result.gemm_calls;
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "exception: %s\n", e.what());
        npu_destroy();
        return 2;
    }

    const uint64_t suite_ns = now_ns() - suite_start;
    std::printf(
        "suite_timing total_ms=%.3f total_gemm_calls=%llu total_mismatches=%llu\n",
        static_cast<double>(suite_ns) / 1.0e6,
        static_cast<unsigned long long>(total_gemm_calls),
        static_cast<unsigned long long>(total_mismatches));

    npu_destroy();
    if (total_mismatches != 0) {
        std::puts("kv260_network_gemm_layers_test=fail");
        return 3;
    }

    std::puts("kv260_network_gemm_layers_test=ok");
    return 0;
}
