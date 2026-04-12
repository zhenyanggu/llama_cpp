#include "ggml-npu-exec.h"

#include "ggml-npu-plan.h"
#include "ggml-npu-quant.h"

#include "ggml-impl.h"
#include "npu_runtime.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace ggml_npu {

static bool npu_debug_log_enabled() {
    return std::getenv("GGML_NPU_DEBUG_LOG") != nullptr || std::getenv("AICAS_MMPROJ_W8A8_DEBUG") != nullptr;
}

static float npu_read_bias_value_f32_exec(
        const struct ggml_tensor * bias,
        int64_t m,
        int64_t n) {
    if (bias == nullptr) {
        return 0.0f;
    }

    const int64_t i0 = bias->ne[0] == 1 ? 0 : m;
    const int64_t i1 = bias->ne[1] == 1 ? 0 : n;
    const char * base = static_cast<const char *>(bias->data) + i1 * bias->nb[1] + i0 * bias->nb[0];

    switch (bias->type) {
        case GGML_TYPE_F32:
            return *reinterpret_cast<const float *>(base);
        case GGML_TYPE_F16:
            return ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t *>(base));
        default:
            return 0.0f;
    }
}

static bool npu_ensure_runtime(std::string * error) {
    if (npu_init() == 0) {
        return true;
    }

    if (error) {
        *error = "npu_init() 失败";
    }
    return false;
}

static bool npu_allocate_runtime_buffer(size_t bytes, void ** ptr, std::string * error) {
    *ptr = npu_mem_alloc(bytes);
    if (*ptr != nullptr) {
        return true;
    }

    if (error) {
        *error = "npu_mem_alloc() 失败";
    }
    return false;
}

enum ggml_status npu_compute_node(const npu_node_plan & plan, std::string * error) {
    if (npu_debug_log_enabled()) {
        const char * root_name = plan.root && plan.root->name[0] != '\0' ? plan.root->name : "(unnamed)";
        GGML_LOG_INFO("%s: enter root=%s exec_tiles=%zu bias=%s summary=%s\n",
                __func__,
                root_name,
                plan.exec_tiles.size(),
                plan.bias ? "yes" : "no",
                plan.summary.c_str());
    }
    if (!npu_ensure_runtime(error)) {
        return GGML_STATUS_FAILED;
    }

    std::string aot_reason;
    if (!npu_plan_is_aot_stable(plan, &aot_reason)) {
        if (error) {
            *error = "AOT plan invalid: " + aot_reason;
        }
        return GGML_STATUS_FAILED;
    }

    const int64_t max_n = plan.config.sa_rows;
    const int64_t max_m = plan.config.sa_cols;
    const int64_t max_k = std::min<int64_t>(plan.config.k_block, plan.config.stage2_k_block);

    void * activation_buf = nullptr;
    void * weight_buf = nullptr;
    void * acc_buf = nullptr;
    void * bias_buf = nullptr;

    if (!npu_allocate_runtime_buffer(static_cast<size_t>(max_n * max_k), &activation_buf, error) ||
        !npu_allocate_runtime_buffer(static_cast<size_t>(max_k * max_m), &weight_buf, error) ||
        !npu_allocate_runtime_buffer(static_cast<size_t>(max_n * max_m * sizeof(int32_t)), &acc_buf, error) ||
        !npu_allocate_runtime_buffer(static_cast<size_t>(max_m * sizeof(int32_t)), &bias_buf, error)) {
        if (activation_buf) {
            npu_mem_free(activation_buf);
        }
        if (weight_buf) {
            npu_mem_free(weight_buf);
        }
        if (acc_buf) {
            npu_mem_free(acc_buf);
        }
        if (bias_buf) {
            npu_mem_free(bias_buf);
        }
        return GGML_STATUS_ALLOC_FAILED;
    }

    std::vector<int8_t> packed_activation;
    const float activation_scale = plan.activation_quant.scale;
    const bool use_aicas_w8a8 = plan.aicas_w8a8.valid;
    std::vector<int32_t> acc_values;
    std::vector<int32_t> bias_values;

    int64_t active_m0 = -1;
    int64_t active_n0 = -1;
    int64_t active_m = 0;
    int64_t active_n = 0;

    for (const npu_exec_tile & exec_tile : plan.exec_tiles) {
        if (active_m0 != exec_tile.m0 || active_n0 != exec_tile.n0) {
            active_m0 = exec_tile.m0;
            active_n0 = exec_tile.n0;
            active_m = exec_tile.m;
            active_n = exec_tile.n;
            acc_values.assign(static_cast<size_t>(active_n * active_m), 0);
            bias_values.assign(static_cast<size_t>(active_m), 0);
        }

        if (!npu_pack_activation_tile_static_asym_i8(
                    plan.src1,
                    exec_tile.n0,
                    exec_tile.n,
                    exec_tile.k0,
                    exec_tile.k,
                    activation_scale,
                    plan.activation_quant.zero_point,
                    &packed_activation,
                    error)) {
            npu_mem_free(activation_buf);
            npu_mem_free(weight_buf);
            npu_mem_free(acc_buf);
            npu_mem_free(bias_buf);
            return GGML_STATUS_FAILED;
        }

        if (exec_tile.weight_pack_index < 0 ||
            static_cast<size_t>(exec_tile.weight_pack_index) >= plan.weight_packs.size()) {
            if (error) {
                *error = "weight_pack_index 非法";
            }
            npu_mem_free(activation_buf);
            npu_mem_free(weight_buf);
            npu_mem_free(acc_buf);
            npu_mem_free(bias_buf);
            return GGML_STATUS_FAILED;
        }

        const npu_prepacked_weight & packed_weight = plan.weight_packs[static_cast<size_t>(exec_tile.weight_pack_index)];

        std::memcpy(activation_buf, packed_activation.data(), packed_activation.size());
        std::memcpy(weight_buf, packed_weight.packed.data(), packed_weight.packed.size());

        npu_dma_mvin(
            activation_buf,
            plan.config.layout.activation.offset,
            static_cast<uint32_t>(exec_tile.k - 1),
            static_cast<uint32_t>(exec_tile.n - 1),
            static_cast<uint16_t>(exec_tile.k),
            static_cast<uint32_t>(exec_tile.k),
            1,
            0,
            false,
            false,
            false,
            0,
            0,
            0);

        npu_dma_mvin(
            weight_buf,
            plan.config.layout.weight.offset,
            static_cast<uint32_t>(exec_tile.m - 1),
            static_cast<uint32_t>(exec_tile.k - 1),
            static_cast<uint16_t>(exec_tile.m),
            static_cast<uint32_t>(exec_tile.m),
            1,
            1,
            false,
            false,
            false,
            0,
            0,
            0);

        if (!use_aicas_w8a8 && plan.bias != nullptr && exec_tile.needs_bias) {
            if (exec_tile.bias_pack_index < 0 ||
                static_cast<size_t>(exec_tile.bias_pack_index) >= plan.bias_packs.size()) {
                if (error) {
                    *error = "bias_pack_index 非法";
                }
                npu_mem_free(activation_buf);
                npu_mem_free(weight_buf);
                npu_mem_free(acc_buf);
                npu_mem_free(bias_buf);
                return GGML_STATUS_FAILED;
            }
            const npu_prepacked_bias & bias_pack =
                plan.bias_packs[static_cast<size_t>(exec_tile.bias_pack_index)];
            if (bias_pack.values.size() < static_cast<size_t>(exec_tile.m)) {
                if (error) {
                    *error = "bias pack size 不足";
                }
                npu_mem_free(activation_buf);
                npu_mem_free(weight_buf);
                npu_mem_free(acc_buf);
                npu_mem_free(bias_buf);
                return GGML_STATUS_FAILED;
            }
            std::copy_n(
                bias_pack.values.begin(),
                static_cast<size_t>(exec_tile.m),
                bias_values.begin());

            std::memcpy(
                bias_buf,
                bias_values.data(),
                static_cast<size_t>(exec_tile.m * sizeof(int32_t)));

            // TODO: The exact bias register base may need a dedicated offset if
            // runtime later separates psum ACC and bias ACC regions.
            npu_dma_mvin(
                bias_buf,
                plan.config.layout.accumulator.offset,
                static_cast<uint32_t>(exec_tile.m - 1),
                0,
                0,
                0,
                1,
                2,
                true,
                true,
                false,
                0,
                0,
                0);
        }

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
            /*biaspsum_addr=*/plan.config.layout.accumulator.offset,
            /*biaspsum_stride=*/static_cast<uint16_t>(exec_tile.m),
            /*biaspsum_width=*/static_cast<uint8_t>(exec_tile.m),
            /*biaspsum_height=*/static_cast<uint8_t>(exec_tile.n),
            /*output_addr=*/plan.config.layout.accumulator.offset,
            /*output_stride=*/static_cast<uint16_t>(exec_tile.m),
            /*isaccu=*/!exec_tile.needs_bias,
            /*relu=*/false,
            /*relu_type=*/0,
            /*is_bias=*/!use_aicas_w8a8 && plan.bias != nullptr && exec_tile.needs_bias,
            /*input_a_addr=*/plan.config.layout.activation.offset,
            /*input_a_col_num=*/static_cast<uint16_t>(exec_tile.k - 1),
            /*input_a_row_num=*/static_cast<uint8_t>(exec_tile.n - 1),
            /*input_a_stride=*/static_cast<uint16_t>(exec_tile.k),
            /*input_b_addr=*/plan.config.layout.weight.offset,
            /*input_b_col_num=*/static_cast<uint8_t>(exec_tile.m - 1),
            /*input_b_row_num=*/static_cast<uint16_t>(exec_tile.k - 1),
            /*input_b_stride=*/static_cast<uint16_t>(exec_tile.m),
            /*asymmetric_activations=*/false);

        if (exec_tile.writes_output) {
            npu_dma_mvout(
                acc_buf,
                plan.config.layout.accumulator.offset,
                static_cast<uint32_t>(exec_tile.n * exec_tile.m - 1),
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

            std::memcpy(
                acc_values.data(),
                acc_buf,
                static_cast<size_t>(exec_tile.n * exec_tile.m * sizeof(int32_t)));

            for (int64_t n = 0; n < exec_tile.n; ++n) {
                for (int64_t m = 0; m < exec_tile.m; ++m) {
                    const float wgt_scale = packed_weight.scales[static_cast<size_t>(m)];
                    const size_t idx = static_cast<size_t>(n * exec_tile.m + m);
                    int32_t acc = acc_values[idx];
                    const int64_t global_m = exec_tile.m0 + m;

                    if (use_aicas_w8a8 &&
                        global_m >= 0 &&
                        static_cast<size_t>(global_m) < plan.weight_column_sum_q.size()) {
                        acc -= plan.activation_quant.zero_point * plan.weight_column_sum_q[static_cast<size_t>(global_m)];
                    }

                    float value = static_cast<float>(acc) * activation_scale * wgt_scale;
                    if (use_aicas_w8a8 && plan.bias != nullptr) {
                        value += npu_read_bias_value_f32_exec(plan.bias, global_m, exec_tile.n0 + n);
                    }
                    char * dst_ptr = static_cast<char *>(plan.dst->data) +
                        (exec_tile.n0 + n) * plan.dst->nb[1] +
                        (exec_tile.m0 + m) * plan.dst->nb[0];
                    *reinterpret_cast<float *>(dst_ptr) = value;
                }
            }
        }
    }

    npu_mem_free(activation_buf);
    npu_mem_free(weight_buf);
    npu_mem_free(acc_buf);
    npu_mem_free(bias_buf);
    if (npu_debug_log_enabled()) {
        const char * root_name = plan.root && plan.root->name[0] != '\0' ? plan.root->name : "(unnamed)";
        GGML_LOG_INFO("%s: leave root=%s\n", __func__, root_name);
    }
    return GGML_STATUS_SUCCESS;
}

} // namespace ggml_npu
