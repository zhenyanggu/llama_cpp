#include "ggml-npu-exec.h"

#include "ggml-npu-plan.h"
#include "ggml-npu-profile.h"
#include "ggml-npu-quant.h"

#include "ggml-impl.h"
#include "npu_runtime.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace ggml_npu {

static bool npu_debug_log_enabled() {
    return std::getenv("GGML_NPU_DEBUG_LOG") != nullptr || std::getenv("AICAS_MMPROJ_W8A8_DEBUG") != nullptr;
}

static bool npu_runtime_profile_requested() {
    const char * path = std::getenv("NPU_PROFILE_OUT");
    return path != nullptr && path[0] != '\0';
}

struct npu_exec_summary {
    npu_profile_summary_delta delta;
};

static std::string npu_stage_name(npu_loop_stage stage) {
    switch (stage) {
        case npu_loop_stage::single: return "single";
        case npu_loop_stage::head:   return "head";
        case npu_loop_stage::body:   return "body";
        case npu_loop_stage::tail:   return "tail";
    }

    return "unknown";
}

static uint32_t npu_float_to_q8_24_u32(float scale) {
    const double scaled = std::nearbyint(static_cast<double>(scale) * static_cast<double>(1u << 24));
    if (scaled > static_cast<double>(INT32_MAX)) {
        return static_cast<uint32_t>(INT32_MAX);
    }
    if (scaled < static_cast<double>(INT32_MIN)) {
        return static_cast<uint32_t>(INT32_MIN);
    }
    return static_cast<uint32_t>(static_cast<int32_t>(scaled));
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

struct npu_activation_tile_key {
    int64_t n0 = 0;
    int64_t n = 0;
    int64_t k0 = 0;
    int64_t k = 0;

    bool operator==(const npu_activation_tile_key & other) const {
        return n0 == other.n0 &&
               n == other.n &&
               k0 == other.k0 &&
               k == other.k;
    }
};

struct npu_activation_tile_key_hash {
    size_t operator()(const npu_activation_tile_key & key) const {
        size_t h = std::hash<int64_t>{}(key.n0);
        h ^= std::hash<int64_t>{}(key.n)  + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int64_t>{}(key.k0) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int64_t>{}(key.k)  + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

enum ggml_status npu_compute_node(const npu_node_plan & plan, int64_t layer_id, std::string * error) {
    const bool collect_detailed_profile = npu_profile_enabled();
    const bool collect_summary = npu_summary_active();
    const bool collect_stage_profile = collect_detailed_profile || collect_summary;
    const bool collect_runtime_profile = npu_runtime_profile_requested() || collect_summary;

    npu_profile_node_record profile_record;
    if (collect_detailed_profile) {
        profile_record = npu_profile_init_node_record(layer_id, plan);
    }

    npu_exec_summary exec_summary;
    exec_summary.delta.node_count = 1;
    exec_summary.delta.exec_tile_count = static_cast<int64_t>(plan.exec_tiles.size());
    exec_summary.delta.weight_pack_count = static_cast<int64_t>(plan.weight_packs.size());
    exec_summary.delta.bias_pack_count = static_cast<int64_t>(plan.bias_packs.size());

    const int64_t node_start_us = ggml_time_us();
    bool runtime_profile_started = false;

    auto cleanup_buffers = [](void * activation_buf, void * weight_buf, void * acc_buf, void * bias_buf) {
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
    };

    auto finalize_status = [&](enum ggml_status status) {
        if (runtime_profile_started) {
            npu_profile_end(layer_id);
            runtime_profile_started = false;
        }

        if (collect_stage_profile) {
            exec_summary.delta.total_node_us = ggml_time_us() - node_start_us;
        }

        if (collect_summary) {
            npu_summary_add_delta(exec_summary.delta);
        }

        if (collect_detailed_profile) {
            profile_record.activation_pack_calls = exec_summary.delta.activation_pack_calls;
            profile_record.host_copy_activation_calls = exec_summary.delta.host_copy_activation_calls;
            profile_record.host_copy_weight_calls = exec_summary.delta.host_copy_weight_calls;
            profile_record.bias_prepare_calls = exec_summary.delta.bias_prepare_calls;
            profile_record.dma_in_activation_calls = exec_summary.delta.dma_in_activation_calls;
            profile_record.dma_in_weight_calls = exec_summary.delta.dma_in_weight_calls;
            profile_record.dma_in_bias_calls = exec_summary.delta.dma_in_bias_calls;
            profile_record.gemm_calls = exec_summary.delta.gemm_calls;
            profile_record.dma_out_calls = exec_summary.delta.dma_out_calls;
            profile_record.postprocess_calls = exec_summary.delta.postprocess_calls;
            profile_record.packed_activation_bytes_total = exec_summary.delta.packed_activation_bytes_total;
            profile_record.copied_weight_bytes_total = exec_summary.delta.copied_weight_bytes_total;
            profile_record.bias_bytes_total = exec_summary.delta.bias_bytes_total;
            profile_record.acc_readback_bytes_total = exec_summary.delta.acc_readback_bytes_total;
            profile_record.output_write_bytes_total = exec_summary.delta.output_write_bytes_total;
            profile_record.activation_pack_us_total = exec_summary.delta.activation_pack_us_total;
            profile_record.host_copy_activation_us_total = exec_summary.delta.host_copy_activation_us_total;
            profile_record.host_copy_weight_us_total = exec_summary.delta.host_copy_weight_us_total;
            profile_record.bias_prepare_us_total = exec_summary.delta.bias_prepare_us_total;
            profile_record.dma_in_activation_us_total = exec_summary.delta.dma_in_activation_us_total;
            profile_record.dma_in_weight_us_total = exec_summary.delta.dma_in_weight_us_total;
            profile_record.dma_in_bias_us_total = exec_summary.delta.dma_in_bias_us_total;
            profile_record.gemm_us_total = exec_summary.delta.gemm_us_total;
            profile_record.dma_out_us_total = exec_summary.delta.dma_out_us_total;
            profile_record.postprocess_us_total = exec_summary.delta.postprocess_us_total;
            profile_record.total_node_us = static_cast<double>(exec_summary.delta.total_node_us);
            profile_record.status = status == GGML_STATUS_SUCCESS ? "success" : "failed";
            if (status != GGML_STATUS_SUCCESS && error != nullptr && !error->empty()) {
                profile_record.error = *error;
            }
            npu_profile_add_node_record(std::move(profile_record));
        }
        return status;
    };
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
        return finalize_status(GGML_STATUS_FAILED);
    }

    std::string aot_reason;
    if (!npu_plan_is_aot_stable(plan, &aot_reason)) {
        if (error) {
            *error = "AOT plan invalid: " + aot_reason;
        }
        return finalize_status(GGML_STATUS_FAILED);
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
        cleanup_buffers(activation_buf, weight_buf, acc_buf, bias_buf);
        return finalize_status(GGML_STATUS_ALLOC_FAILED);
    }

    std::vector<int8_t> packed_activation;
    std::unordered_map<npu_activation_tile_key, std::vector<int8_t>, npu_activation_tile_key_hash> activation_tile_cache;
    const float activation_scale = plan.activation_quant.scale;
    const bool use_aicas_w8a8 = plan.aicas_w8a8.valid;
    const uint32_t mvout_f32_scale = use_aicas_w8a8 && plan.aicas_w8a8.act_scale_q8_24 != 0
        ? static_cast<uint32_t>(plan.aicas_w8a8.act_scale_q8_24)
        : npu_float_to_q8_24_u32(activation_scale);
    std::vector<float> acc_scaled_values;
    std::vector<int32_t> bias_values;

    activation_tile_cache.reserve(plan.exec_tiles.size());

    int64_t active_m0 = -1;
    int64_t active_n0 = -1;
    int64_t active_m = 0;
    int64_t active_n = 0;

    if (collect_runtime_profile) {
        npu_profile_begin(layer_id);
        runtime_profile_started = true;
    }

    for (const npu_exec_tile & exec_tile : plan.exec_tiles) {
        npu_profile_tile_record tile_record;
        const int64_t tile_start_us = collect_detailed_profile ? ggml_time_us() : 0;
        if (collect_detailed_profile) {
            tile_record.tile_index = static_cast<int64_t>(profile_record.tiles.size());
            tile_record.m0 = exec_tile.m0;
            tile_record.n0 = exec_tile.n0;
            tile_record.k0 = exec_tile.k0;
            tile_record.m = exec_tile.m;
            tile_record.n = exec_tile.n;
            tile_record.k = exec_tile.k;
            tile_record.stage = npu_stage_name(exec_tile.stage);
            tile_record.needs_bias = exec_tile.needs_bias;
            tile_record.writes_output = exec_tile.writes_output;
            tile_record.weight_pack_index = exec_tile.weight_pack_index;
            tile_record.bias_pack_index = exec_tile.bias_pack_index;
        }

        if (active_m0 != exec_tile.m0 || active_n0 != exec_tile.n0) {
            active_m0 = exec_tile.m0;
            active_n0 = exec_tile.n0;
            active_m = exec_tile.m;
            active_n = exec_tile.n;
            acc_scaled_values.assign(static_cast<size_t>(active_n * active_m), 0.0f);
            bias_values.assign(static_cast<size_t>(active_m), 0);
        }

        const npu_activation_tile_key activation_key {
            exec_tile.n0,
            exec_tile.n,
            exec_tile.k0,
            exec_tile.k,
        };

        const std::vector<int8_t> * packed_activation_view = nullptr;
        const auto cache_it = activation_tile_cache.find(activation_key);
        if (cache_it != activation_tile_cache.end()) {
            packed_activation_view = &cache_it->second;
            if (collect_detailed_profile) {
                tile_record.activation_pack_us = 0.0;
            }
        } else {
            const int64_t activation_pack_start_us = collect_stage_profile ? ggml_time_us() : 0;
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
                const int64_t activation_pack_us = collect_stage_profile ? (ggml_time_us() - activation_pack_start_us) : 0;
                if (collect_stage_profile) {
                    exec_summary.delta.activation_pack_calls += 1;
                    exec_summary.delta.activation_pack_us_total += activation_pack_us;
                }
                if (collect_detailed_profile) {
                    tile_record.activation_pack_us = static_cast<double>(activation_pack_us);
                    profile_record.tiles.push_back(std::move(tile_record));
                }
                cleanup_buffers(activation_buf, weight_buf, acc_buf, bias_buf);
                return finalize_status(GGML_STATUS_FAILED);
            }
            const auto inserted = activation_tile_cache.emplace(activation_key, packed_activation);
            packed_activation_view = &inserted.first->second;
            if (collect_stage_profile) {
                const int64_t activation_pack_us = ggml_time_us() - activation_pack_start_us;
                exec_summary.delta.activation_pack_calls += 1;
                exec_summary.delta.activation_pack_us_total += activation_pack_us;
                exec_summary.delta.packed_activation_bytes_total += static_cast<int64_t>(packed_activation_view->size());
                if (collect_detailed_profile) {
                    tile_record.activation_pack_us = static_cast<double>(activation_pack_us);
                }
            }
        }
        if (collect_detailed_profile) {
            tile_record.activation_bytes = static_cast<int64_t>(packed_activation_view->size());
        }

        if (exec_tile.weight_pack_index < 0 ||
            static_cast<size_t>(exec_tile.weight_pack_index) >= plan.weight_packs.size()) {
            if (error) {
                *error = "weight_pack_index 非法";
            }
            if (collect_detailed_profile) {
                profile_record.tiles.push_back(std::move(tile_record));
            }
            cleanup_buffers(activation_buf, weight_buf, acc_buf, bias_buf);
            return finalize_status(GGML_STATUS_FAILED);
        }

        const npu_prepacked_weight & packed_weight = plan.weight_packs[static_cast<size_t>(exec_tile.weight_pack_index)];
        if (collect_detailed_profile) {
            tile_record.weight_bytes = static_cast<int64_t>(packed_weight.packed.size());
        }

        const int64_t activation_copy_start_us = collect_stage_profile ? ggml_time_us() : 0;
        std::memcpy(activation_buf, packed_activation_view->data(), packed_activation_view->size());
        if (collect_stage_profile) {
            const int64_t activation_copy_us = ggml_time_us() - activation_copy_start_us;
            exec_summary.delta.host_copy_activation_calls += 1;
            exec_summary.delta.host_copy_activation_us_total += activation_copy_us;
            if (collect_detailed_profile) {
                tile_record.host_copy_activation_us = static_cast<double>(activation_copy_us);
            }
        }

        const int64_t weight_copy_start_us = collect_stage_profile ? ggml_time_us() : 0;
        std::memcpy(weight_buf, packed_weight.packed.data(), packed_weight.packed.size());
        if (collect_stage_profile) {
            const int64_t weight_copy_us = ggml_time_us() - weight_copy_start_us;
            exec_summary.delta.host_copy_weight_calls += 1;
            exec_summary.delta.host_copy_weight_us_total += weight_copy_us;
            exec_summary.delta.copied_weight_bytes_total += static_cast<int64_t>(packed_weight.packed.size());
            if (collect_detailed_profile) {
                tile_record.host_copy_weight_us = static_cast<double>(weight_copy_us);
            }
        }

        const MvinConfig activation_mvin_cfg {
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
            0,
        };

        const MvinConfig weight_mvin_cfg {
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
            0,
        };

        const int64_t dma_in_pair_start_us = collect_stage_profile ? ggml_time_us() : 0;
        if (npu_debug_log_enabled()) {
            GGML_LOG_INFO("%s: tile m0=%" PRId64 " n0=%" PRId64 " k0=%" PRId64 " stage=%s MVIN_ACT_DMA0 col=%" PRIu32 " row=%" PRIu32 " sram=0x%08x\n",
                    __func__,
                    exec_tile.m0, exec_tile.n0, exec_tile.k0,
                    npu_stage_name(exec_tile.stage).c_str(),
                    static_cast<uint32_t>(exec_tile.k - 1),
                    static_cast<uint32_t>(exec_tile.n - 1),
                    plan.config.layout.activation.offset);
            GGML_LOG_INFO("%s: tile m0=%" PRId64 " n0=%" PRId64 " k0=%" PRId64 " stage=%s MVIN_WGT_DMA1 col=%" PRIu32 " row=%" PRIu32 " sram=0x%08x\n",
                    __func__,
                    exec_tile.m0, exec_tile.n0, exec_tile.k0,
                    npu_stage_name(exec_tile.stage).c_str(),
                    static_cast<uint32_t>(exec_tile.m - 1),
                    static_cast<uint32_t>(exec_tile.k - 1),
                    plan.config.layout.weight.offset);
        }
        npu_dma_double_mvin(&activation_mvin_cfg, &weight_mvin_cfg);
        if (collect_stage_profile) {
            const int64_t dma_in_pair_us = ggml_time_us() - dma_in_pair_start_us;
            const int64_t dma_in_activation_us = dma_in_pair_us / 2;
            const int64_t dma_in_weight_us = dma_in_pair_us - dma_in_activation_us;
            exec_summary.delta.dma_in_activation_calls += 1;
            exec_summary.delta.dma_in_weight_calls += 1;
            exec_summary.delta.dma_in_activation_us_total += dma_in_activation_us;
            exec_summary.delta.dma_in_weight_us_total += dma_in_weight_us;
            if (collect_detailed_profile) {
                tile_record.dma_in_activation_us = static_cast<double>(dma_in_activation_us);
                tile_record.dma_in_weight_us = static_cast<double>(dma_in_weight_us);
            }
        }

        if (plan.bias != nullptr && exec_tile.needs_bias) {
            if (exec_tile.bias_pack_index < 0 ||
                static_cast<size_t>(exec_tile.bias_pack_index) >= plan.bias_packs.size()) {
                if (error) {
                    *error = "bias_pack_index 非法";
                }
                if (collect_detailed_profile) {
                    profile_record.tiles.push_back(std::move(tile_record));
                }
                cleanup_buffers(activation_buf, weight_buf, acc_buf, bias_buf);
                return finalize_status(GGML_STATUS_FAILED);
            }
            const npu_prepacked_bias & bias_pack =
                plan.bias_packs[static_cast<size_t>(exec_tile.bias_pack_index)];
            if (bias_pack.values.size() < static_cast<size_t>(exec_tile.m)) {
                if (error) {
                    *error = "bias pack size 不足";
                }
                if (collect_detailed_profile) {
                    profile_record.tiles.push_back(std::move(tile_record));
                }
                cleanup_buffers(activation_buf, weight_buf, acc_buf, bias_buf);
                return finalize_status(GGML_STATUS_FAILED);
            }

            const int64_t bias_prepare_start_us = collect_stage_profile ? ggml_time_us() : 0;
            std::copy_n(
                bias_pack.values.begin(),
                static_cast<size_t>(exec_tile.m),
                bias_values.begin());

            std::memcpy(
                bias_buf,
                bias_values.data(),
                static_cast<size_t>(exec_tile.m * sizeof(int32_t)));
            if (collect_stage_profile) {
                const int64_t bias_prepare_us = ggml_time_us() - bias_prepare_start_us;
                exec_summary.delta.bias_prepare_calls += 1;
                exec_summary.delta.bias_prepare_us_total += bias_prepare_us;
                exec_summary.delta.bias_bytes_total += static_cast<int64_t>(exec_tile.m * sizeof(int32_t));
                if (collect_detailed_profile) {
                    tile_record.bias_prepare_us = static_cast<double>(bias_prepare_us);
                }
            }
            if (collect_detailed_profile) {
                tile_record.bias_bytes = static_cast<int64_t>(exec_tile.m * sizeof(int32_t));
            }

            // TODO: The exact bias register base may need a dedicated offset if
            // runtime later separates psum ACC and bias ACC regions.
            const int64_t dma_in_bias_start_us = collect_stage_profile ? ggml_time_us() : 0;
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
            if (collect_stage_profile) {
                const int64_t dma_in_bias_us = ggml_time_us() - dma_in_bias_start_us;
                exec_summary.delta.dma_in_bias_calls += 1;
                exec_summary.delta.dma_in_bias_us_total += dma_in_bias_us;
                if (collect_detailed_profile) {
                    tile_record.dma_in_bias_us = static_cast<double>(dma_in_bias_us);
                }
            }
        }

        const int64_t gemm_start_us = collect_stage_profile ? ggml_time_us() : 0;
        if (npu_debug_log_enabled()) {
            GGML_LOG_INFO("%s: tile m0=%" PRId64 " n0=%" PRId64 " k0=%" PRId64 " GEMM n=%" PRId64 " m=%" PRId64 " k=%" PRId64 " acc=0x%08x\n",
                    __func__,
                    exec_tile.m0, exec_tile.n0, exec_tile.k0,
                    exec_tile.n, exec_tile.m, exec_tile.k,
                    plan.config.layout.accumulator.offset);
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
            /*is_bias=*/plan.bias != nullptr && exec_tile.needs_bias,
            /*input_a_addr=*/plan.config.layout.activation.offset,
            /*input_a_col_num=*/static_cast<uint16_t>(exec_tile.k - 1),
            /*input_a_row_num=*/static_cast<uint8_t>(exec_tile.n - 1),
            /*input_a_stride=*/static_cast<uint16_t>(exec_tile.k),
            /*input_b_addr=*/plan.config.layout.weight.offset,
            /*input_b_col_num=*/static_cast<uint8_t>(exec_tile.m - 1),
            /*input_b_row_num=*/static_cast<uint16_t>(exec_tile.k - 1),
            /*input_b_stride=*/static_cast<uint16_t>(exec_tile.m),
            /*asymmetric_activations=*/false);
        if (collect_stage_profile) {
            const int64_t gemm_us = ggml_time_us() - gemm_start_us;
            exec_summary.delta.gemm_calls += 1;
            exec_summary.delta.gemm_us_total += gemm_us;
            if (collect_detailed_profile) {
                tile_record.gemm_us = static_cast<double>(gemm_us);
            }
        }

        if (exec_tile.writes_output) {
            if (collect_detailed_profile) {
                tile_record.acc_readback_bytes = static_cast<int64_t>(exec_tile.n * exec_tile.m * sizeof(int32_t));
            }

            const int64_t dma_out_start_us = collect_stage_profile ? ggml_time_us() : 0;
            if (npu_debug_log_enabled()) {
                GGML_LOG_INFO("%s: tile m0=%" PRId64 " n0=%" PRId64 " k0=%" PRId64 " MVOUT_ACC_F32 col=%" PRIu32 " row=0 sram=0x%08x f32_scale=0x%08x\n",
                        __func__,
                        exec_tile.m0, exec_tile.n0, exec_tile.k0,
                        static_cast<uint32_t>(exec_tile.n * exec_tile.m - 1),
                        plan.config.layout.accumulator.offset,
                        mvout_f32_scale);
            }
            npu_dma_mvout(
                acc_buf,
                plan.config.layout.accumulator.offset,
                static_cast<uint32_t>(exec_tile.n * exec_tile.m - 1),
                0,
                0,
                0,
                3,
                1,
                true,
                true,
                0,
                mvout_f32_scale);
            if (collect_stage_profile) {
                const int64_t dma_out_us = ggml_time_us() - dma_out_start_us;
                exec_summary.delta.dma_out_calls += 1;
                exec_summary.delta.dma_out_us_total += dma_out_us;
                exec_summary.delta.acc_readback_bytes_total += static_cast<int64_t>(exec_tile.n * exec_tile.m * sizeof(int32_t));
                if (collect_detailed_profile) {
                    tile_record.dma_out_us = static_cast<double>(dma_out_us);
                }
            }

            const int64_t postprocess_start_us = collect_stage_profile ? ggml_time_us() : 0;
            std::memcpy(
                acc_scaled_values.data(),
                acc_buf,
                static_cast<size_t>(exec_tile.n * exec_tile.m * sizeof(float)));

            for (int64_t n = 0; n < exec_tile.n; ++n) {
                for (int64_t m = 0; m < exec_tile.m; ++m) {
                    const float wgt_scale = packed_weight.scales[static_cast<size_t>(m)];
                    const size_t idx = static_cast<size_t>(n * exec_tile.m + m);
                    const float acc_scaled = acc_scaled_values[idx];
                    const int64_t global_m = exec_tile.m0 + m;

                    float value = acc_scaled * wgt_scale;
                    if (use_aicas_w8a8 &&
                        global_m >= 0 &&
                        static_cast<size_t>(global_m) < plan.weight_column_sum_q.size()) {
                        value -= static_cast<float>(
                            plan.activation_quant.zero_point *
                            plan.weight_column_sum_q[static_cast<size_t>(global_m)]) *
                            activation_scale * wgt_scale;
                    }
                    char * dst_ptr = static_cast<char *>(plan.dst->data) +
                        (exec_tile.n0 + n) * plan.dst->nb[1] +
                        (exec_tile.m0 + m) * plan.dst->nb[0];
                    *reinterpret_cast<float *>(dst_ptr) = value;
                }
            }
            if (collect_stage_profile) {
                const int64_t postprocess_us = ggml_time_us() - postprocess_start_us;
                exec_summary.delta.postprocess_calls += 1;
                exec_summary.delta.postprocess_us_total += postprocess_us;
                exec_summary.delta.output_write_bytes_total += static_cast<int64_t>(exec_tile.n * exec_tile.m * sizeof(float));
                if (collect_detailed_profile) {
                    tile_record.postprocess_us = static_cast<double>(postprocess_us);
                }
            }
            if (collect_detailed_profile) {
                tile_record.output_write_bytes = static_cast<int64_t>(exec_tile.n * exec_tile.m * sizeof(float));
            }
        }

        if (collect_detailed_profile) {
            tile_record.total_us = static_cast<double>(ggml_time_us() - tile_start_us);
            profile_record.tiles.push_back(std::move(tile_record));
        }
    }

    cleanup_buffers(activation_buf, weight_buf, acc_buf, bias_buf);
    if (npu_debug_log_enabled()) {
        const char * root_name = plan.root && plan.root->name[0] != '\0' ? plan.root->name : "(unnamed)";
        GGML_LOG_INFO("%s: leave root=%s\n", __func__, root_name);
    }
    return finalize_status(GGML_STATUS_SUCCESS);
}

} // namespace ggml_npu
