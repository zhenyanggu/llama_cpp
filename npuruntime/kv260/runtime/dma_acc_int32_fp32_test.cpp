#include "npu_runtime.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

constexpr uint32_t kAccAddr = 0x0000;
constexpr uint32_t kScaleAddr = 0x00070000;
constexpr uint32_t kCols = 16;
constexpr uint32_t kRows = 16;
constexpr uint32_t kElemCount = kCols * kRows;

int32_t pack_q8_24(float scale) {
    return static_cast<int32_t>(std::round(scale * 16777216.0f));
}

float dequant_theory(int32_t x, uint32_t zero_point, float scale) {
    return (static_cast<float>(x) - static_cast<float>(zero_point)) * scale;
}

bool check_raw(const int32_t* got, const int32_t* expect) {
    uint32_t mismatches = 0;
    for (uint32_t i = 0; i < kElemCount; ++i) {
        if (got[i] != expect[i]) {
            if (mismatches < 16) {
                std::fprintf(stderr, "raw mismatch idx=%u expect=%d got=%d\n", i, expect[i], got[i]);
            }
            ++mismatches;
        }
    }
    if (mismatches != 0) {
        std::fprintf(stderr, "raw readback failed mismatches=%u\n", mismatches);
        return false;
    }
    return true;
}

bool check_fp32(const float* got, const int32_t* src, const float* scales, uint32_t zero_point, const char* tag) {
    uint32_t mismatches = 0;
    constexpr float kAbsTol = 1e-5f;
    for (uint32_t i = 0; i < kElemCount; ++i) {
        const float expect = dequant_theory(src[i], zero_point, scales[i]);
        if (std::fabs(expect - got[i]) > kAbsTol) {
            if (mismatches < 16) {
                std::fprintf(
                    stderr, "%s mismatch idx=%u expect=%.6f got=%.6f src=%d scale=%.6f\n",
                    tag, i, expect, got[i], src[i], scales[i]);
            }
            ++mismatches;
        }
    }
    if (mismatches != 0) {
        std::fprintf(stderr, "%s failed mismatches=%u\n", tag, mismatches);
        return false;
    }
    return true;
}

enum class Stage {
    Mvin,
    All,
    Raw,
    Tensor,
    Channel,
};

Stage parse_stage() {
    const char* env = std::getenv("NPU_DMA_ACC_STAGE");
    if (env == nullptr || *env == '\0' || std::strcmp(env, "all") == 0) {
        return Stage::All;
    }
    if (std::strcmp(env, "mvin") == 0) return Stage::Mvin;
    if (std::strcmp(env, "raw") == 0) return Stage::Raw;
    if (std::strcmp(env, "tensor") == 0) return Stage::Tensor;
    if (std::strcmp(env, "channel") == 0) return Stage::Channel;
    std::fprintf(stderr, "unknown NPU_DMA_ACC_STAGE=%s\n", env);
    std::exit(2);
}

} // namespace

int main() {
    std::puts("kv260_dma_acc_int32_fp32_test: acc int32 raw/per-tensor/per-channel fp32 mvout");

    if (npu_init() != 0) {
        std::perror("npu_init");
        return 1;
    }
    npu_reset();

    auto* host_src = static_cast<int32_t*>(npu_mem_alloc(kElemCount * sizeof(int32_t)));
    auto* host_raw = static_cast<int32_t*>(npu_mem_alloc(kElemCount * sizeof(int32_t)));
    auto* host_f32 = static_cast<float*>(npu_mem_alloc(kElemCount * sizeof(float)));
    auto* host_scales = static_cast<int32_t*>(npu_mem_alloc(kElemCount * sizeof(int32_t)));
    if (!host_src || !host_raw || !host_f32 || !host_scales) {
        std::fprintf(stderr, "npu_mem_alloc failed\n");
        return 2;
    }

    for (uint32_t i = 0; i < kElemCount; ++i) {
        host_src[i] = static_cast<int32_t>((static_cast<int>(i) - 128) * 17);
        host_scales[i] = pack_q8_24(1.0f);
    }
    std::memset(host_raw, 0, kElemCount * sizeof(int32_t));
    std::memset(host_f32, 0, kElemCount * sizeof(float));

    const Stage stage = parse_stage();
    const bool do_mvin_only = (stage == Stage::Mvin);
    const bool do_raw = (stage == Stage::All || stage == Stage::Raw);
    const bool do_tensor = (stage == Stage::All || stage == Stage::Tensor);
    const bool do_channel = (stage == Stage::All || stage == Stage::Channel);

    // Use 2D stride form for ACC int32 path to avoid 1D packing ambiguity.
    npu_dma_mvin(
        host_src, kAccAddr, kCols - 1, kRows - 1, static_cast<uint16_t>(kCols), kCols,
        1, 2, true, false, false, 0, 0, 0);

    if (do_mvin_only) {
        npu_mem_free(host_src);
        npu_mem_free(host_raw);
        npu_mem_free(host_f32);
        npu_mem_free(host_scales);
        npu_destroy();
        std::puts("mvin stage: ok");
        return 0;
    }

    if (do_raw) {
        npu_dma_mvout(
            host_raw, kAccAddr, kCols - 1, kRows - 1, static_cast<uint16_t>(kCols), kCols,
            1, 1, true, false, 0, 0);
        if (!check_raw(host_raw, host_src)) return 3;
        std::puts("raw readback: ok");
    }

    constexpr uint32_t kZeroPoint = 0;
    if (do_tensor || do_channel) {
        const int32_t tensor_scale = pack_q8_24(1.0f);
        npu_dma_mvout(
            host_f32, kAccAddr, kCols - 1, kRows - 1, static_cast<uint16_t>(kCols), kCols,
            3, 1, true, true, kZeroPoint, static_cast<uint32_t>(tensor_scale));

        float tensor_scales[kElemCount];
        for (uint32_t i = 0; i < kElemCount; ++i) tensor_scales[i] = 1.0f;
        if (!check_fp32(host_f32, host_src, tensor_scales, kZeroPoint, "per_tensor fp32")) return 4;
        std::puts("per_tensor fp32: ok");
    }

    if (do_channel) {
        float channel_scales[kElemCount];
        for (uint32_t i = 0; i < kElemCount; ++i) {
            channel_scales[i] = 0.25f * static_cast<float>((i % 4) + 1);
            host_scales[i] = pack_q8_24(channel_scales[i]);
        }
        npu_dma_mvin(
            host_scales, kScaleAddr, kCols - 1, kRows - 1, static_cast<uint16_t>(kCols), kCols,
            1, 2, true, false, false, 0, 0, 0);
        std::memset(host_f32, 0, kElemCount * sizeof(float));

        npu_dma_mvout_ex(
            host_f32, kAccAddr, kCols - 1, kRows - 1, static_cast<uint16_t>(kCols), kCols,
            3, 1, true, true, kZeroPoint, kScaleAddr, true);
        if (!check_fp32(host_f32, host_src, channel_scales, kZeroPoint, "per_channel fp32")) return 5;
        std::puts("per_channel fp32: ok");
    }

    npu_mem_free(host_src);
    npu_mem_free(host_raw);
    npu_mem_free(host_f32);
    npu_mem_free(host_scales);
    npu_destroy();
    std::puts("kv260_dma_acc_int32_fp32_test=ok");
    return 0;
}
