#include "../ggml/src/ggml-npu/ggml-npu-quant.h"

#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

static int8_t ref_quantize_u8_payload(float shifted) {
    int32_t q = static_cast<int32_t>(std::lrint(shifted));
    q = std::max(0, std::min(255, q));
    return static_cast<int8_t>(q);
}

static bool check_pack(
        const char * label,
        const std::vector<int8_t> & actual,
        const std::vector<int8_t> & expected) {
    if (actual.size() != expected.size() ||
            std::memcmp(actual.data(), expected.data(), actual.size()) != 0) {
        std::fprintf(stderr, "mismatch: %s\n", label);
        return false;
    }
    return true;
}

static std::vector<int8_t> reference_pack(
        const std::vector<float> & values,
        int64_t k_total,
        int64_t n0,
        int64_t n_rows,
        int64_t k0,
        int64_t k_cols,
        float scale,
        int32_t zero_point_u8,
        const std::vector<float> * smooth_scale) {
    std::vector<int8_t> ref(static_cast<size_t>(n_rows * k_cols));
    const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
    const float zp = static_cast<float>(zero_point_u8);
    for (int64_t n = 0; n < n_rows; ++n) {
        for (int64_t k = 0; k < k_cols; ++k) {
            const float v = values[static_cast<size_t>((n0 + n) * k_total + k0 + k)];
            const float mul = smooth_scale != nullptr ? (*smooth_scale)[static_cast<size_t>(k0 + k)] : inv_scale;
            ref[static_cast<size_t>(n * k_cols + k)] = ref_quantize_u8_payload(v * mul + zp);
        }
    }
    return ref;
}

static bool run_case(ggml_type type, const char * label) {
    constexpr int64_t k_total = 8;
    constexpr int64_t n_total = 5;
    constexpr int64_t n0 = 1;
    constexpr int64_t n_rows = 3;
    constexpr int64_t k0 = 2;
    constexpr int64_t k_cols = 5;
    constexpr float scale = 0.03125f;
    constexpr int32_t zero_point_u8 = 129;

    std::vector<float> values(static_cast<size_t>(k_total * n_total));
    for (int64_t n = 0; n < n_total; ++n) {
        for (int64_t k = 0; k < k_total; ++k) {
            values[static_cast<size_t>(n * k_total + k)] =
                0.125f * static_cast<float>((n - 2) * 7 + (k - 3));
        }
    }

    struct ggml_init_params params = {
        /*.mem_size   = */ 1024 * 1024,
        /*.mem_buffer = */ nullptr,
        /*.no_alloc   = */ false,
    };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), ggml_free);
    if (!ctx) {
        std::fprintf(stderr, "failed to init ggml context\n");
        return false;
    }

    ggml_tensor * tensor = ggml_new_tensor_2d(ctx.get(), type, k_total, n_total);
    if (type == GGML_TYPE_F32) {
        float * dst = static_cast<float *>(tensor->data);
        std::memcpy(dst, values.data(), values.size() * sizeof(float));
    } else if (type == GGML_TYPE_F16) {
        ggml_fp16_t * dst = static_cast<ggml_fp16_t *>(tensor->data);
        for (size_t i = 0; i < values.size(); ++i) {
            dst[i] = ggml_fp32_to_fp16(values[i]);
        }
    } else if (type == GGML_TYPE_I8) {
        int8_t * dst = static_cast<int8_t *>(tensor->data);
        for (size_t i = 0; i < values.size(); ++i) {
            dst[i] = static_cast<int8_t>(std::lrint(values[i]));
            values[i] = static_cast<float>(dst[i]);
        }
    }

    std::vector<int8_t> actual;
    std::string error;
    if (!ggml_npu::npu_pack_activation_tile_static_asym_i8(
                tensor, n0, n_rows, k0, k_cols, k_cols, scale, zero_point_u8, nullptr, &actual, &error)) {
        std::fprintf(stderr, "%s no-smooth pack failed: %s\n", label, error.c_str());
        return false;
    }
    std::vector<int8_t> expected =
        reference_pack(values, k_total, n0, n_rows, k0, k_cols, scale, zero_point_u8, nullptr);
    if (!check_pack(label, actual, expected)) {
        return false;
    }

    std::vector<float> smooth(static_cast<size_t>(k_total));
    for (int64_t k = 0; k < k_total; ++k) {
        smooth[static_cast<size_t>(k)] = 0.75f + 0.125f * static_cast<float>(k);
    }
    actual.clear();
    if (!ggml_npu::npu_pack_activation_tile_static_asym_i8(
                tensor, n0, n_rows, k0, k_cols, k_cols, scale, zero_point_u8, &smooth, &actual, &error)) {
        std::fprintf(stderr, "%s smooth pack failed: %s\n", label, error.c_str());
        return false;
    }
    expected = reference_pack(values, k_total, n0, n_rows, k0, k_cols, scale, zero_point_u8, &smooth);
    return check_pack(label, actual, expected);
}

int main() {
#ifndef GGML_USE_NPU
    std::fprintf(stderr, "GGML_USE_NPU is disabled, skipping test-npu-quant\n");
    return 0;
#else
    if (!run_case(GGML_TYPE_F32, "f32")) {
        return 1;
    }
    if (!run_case(GGML_TYPE_F16, "f16")) {
        return 1;
    }
    if (!run_case(GGML_TYPE_I8, "i8")) {
        return 1;
    }

    std::puts("test-npu-quant: ok");
    return 0;
#endif
}
