#include "../ggml/src/ggml-npu/ggml-npu.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

static bool check_bytes(
        const char * label,
        const void * actual,
        const void * expected,
        size_t size) {
    if (std::memcmp(actual, expected, size) != 0) {
        std::fprintf(stderr, "mismatch: %s\n", label);
        return false;
    }
    return true;
}

int main() {
#ifndef GGML_USE_NPU
    std::fprintf(stderr, "GGML_USE_NPU is disabled, skipping test-npu-buffer-io\n");
    return 0;
#else
    struct ggml_init_params params = {
        /*.mem_size   = */ 1024 * 1024,
        /*.mem_buffer = */ nullptr,
        /*.no_alloc   = */ true,
    };

    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), ggml_free);
    if (!ctx) {
        std::fprintf(stderr, "failed to init ggml context\n");
        return 1;
    }

    ggml_tensor * base = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 8);
    ggml_set_name(base, "base");
    ggml_tensor * tail = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 4);
    ggml_set_name(tail, "tail");
    ggml_tensor * view = ggml_view_1d(ctx.get(), base, 3, 2 * sizeof(int32_t));
    ggml_set_name(view, "view");

    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(
        ggml_backend_npu_init(), ggml_backend_free);
    if (!backend) {
        std::fprintf(stderr, "failed to init NPU backend\n");
        return 1;
    }

    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buf(
        ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()),
        ggml_backend_buffer_free);
    if (!buf) {
        std::fprintf(stderr, "failed to allocate NPU buffer\n");
        return 1;
    }

    const std::array<int32_t, 8> base_init = { 10, 11, 12, 13, 14, 15, 16, 17 };
    const std::array<int32_t, 4> tail_init = { 100, 101, 102, 103 };
    ggml_backend_tensor_set(base, base_init.data(), 0, sizeof(base_init));
    ggml_backend_tensor_set(tail, tail_init.data(), 0, sizeof(tail_init));

    const std::array<int32_t, 3> view_write = { 200, 201, 202 };
    ggml_backend_tensor_set(view, view_write.data(), 0, sizeof(view_write));

    std::array<int32_t, 8> base_after = {};
    std::array<int32_t, 4> tail_after = {};
    std::array<int32_t, 3> view_after = {};
    ggml_backend_tensor_get(base, base_after.data(), 0, sizeof(base_after));
    ggml_backend_tensor_get(tail, tail_after.data(), 0, sizeof(tail_after));
    ggml_backend_tensor_get(view, view_after.data(), 0, sizeof(view_after));

    const std::array<int32_t, 8> base_expected = { 10, 11, 200, 201, 202, 15, 16, 17 };
    if (!check_bytes("base after view write", base_after.data(), base_expected.data(), sizeof(base_expected))) {
        return 1;
    }
    if (!check_bytes("tail preserved", tail_after.data(), tail_init.data(), sizeof(tail_init))) {
        return 1;
    }
    if (!check_bytes("view readback", view_after.data(), view_write.data(), sizeof(view_write))) {
        return 1;
    }

    ggml_backend_tensor_memset(base, 0, 3 * sizeof(int32_t), 2 * sizeof(int32_t));
    ggml_backend_tensor_get(base, base_after.data(), 0, sizeof(base_after));
    const std::array<int32_t, 8> memset_expected = { 10, 11, 200, 0, 0, 15, 16, 17 };
    if (!check_bytes("base after partial memset", base_after.data(), memset_expected.data(), sizeof(memset_expected))) {
        return 1;
    }

    std::puts("test-npu-buffer-io: ok");
    return 0;
#endif
}
