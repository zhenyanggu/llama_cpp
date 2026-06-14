bool ggml_backend_npu_decode_swiglu_ffn_w4a16_ex(
        const char * op_name,
        const struct ggml_npu_decode_awq_view * gate,
        const struct ggml_npu_decode_awq_view * up,
        const struct ggml_npu_decode_awq_view * down,
        const void * act_data,
        int act_type,
        int64_t act_nb0,
        int64_t act_nb1,
        void * dst_data,
        int dst_type,
        int64_t dst_nb1) {
    using namespace ggml_npu;

    auto fail = [op_name](const char * reason) {
        if (npu_decode_w4a16_debug_log_enabled()) {
            GGML_LOG_WARN("%s: reject %s: %s\n", __func__, op_name != nullptr ? op_name : "(unnamed)", reason);
        }
        return false;
    };

    if (gate == nullptr || up == nullptr || down == nullptr ||
            act_data == nullptr || dst_data == nullptr) {
        return fail("null pointer");
    }
    if (act_type != GGML_TYPE_F16 && act_type != GGML_TYPE_F32) {
        return fail("activation tensor must be F16 or F32");
    }
    if (dst_type != GGML_TYPE_F16 && dst_type != GGML_TYPE_F32) {
        return fail("destination tensor must be F16 or F32");
    }
    const bool collect_profile = npu_decode_profile_enabled();
    const int64_t total_start_us = collect_profile ? ggml_time_us() : 0;
    const int64_t setup_start_us = collect_profile ? total_start_us : 0;
    int64_t stage_start_us = collect_profile ? total_start_us : 0;
    npu_decode_gemv_profile_record profile = {};
    profile.profile_kind = "aicas_decode_w4a16_swiglu_ffn";
    profile.gemv_mode = "w4a16";
    profile.op_role = "swiglu_ffn";
    profile.op_name = op_name != nullptr ? op_name : "";
    profile.dst_type = dst_type;
    profile.k = gate != nullptr ? gate->k : 0;
    profile.n_cols = 1;
    profile.out_channels = down != nullptr ? down->out_channels : 0;

    const bool hw_f32_mvout = dst_type == GGML_TYPE_F32 && npu_decode_hw_f32_mvout_enabled();
    const uint8_t output_precision = npu_decode_mvout_precision(hw_f32_mvout);
    if (gate->k <= 0 || up->k <= 0 || down->k <= 0 ||
            gate->out_channels <= 0 || up->out_channels <= 0 || down->out_channels <= 0) {
        return fail("invalid dimensions");
    }
    if (gate->k != up->k || gate->out_channels != up->out_channels ||
            down->k != gate->out_channels) {
        return fail("incompatible SwiGLU dimensions");
    }
    if (gate->packed_k != (gate->k + 1) / 2 ||
            up->packed_k != (up->k + 1) / 2 ||
            down->packed_k != (down->k + 1) / 2) {
        return fail("invalid packed dimensions");
    }
    if (gate->k > std::numeric_limits<uint16_t>::max() ||
            gate->out_channels > std::numeric_limits<uint16_t>::max() ||
            down->out_channels > std::numeric_limits<uint16_t>::max()) {
        return fail("dimension exceeds uint16 range");
    }
    if (gate->out_channels > 4096) {
        return fail("SwiGLU intermediate exceeds stream buffer");
    }
    if (gate->smooth_scale == nullptr || up->smooth_scale == nullptr || down->smooth_scale == nullptr ||
            gate->smooth_scale_len != static_cast<size_t>(gate->k) ||
            up->smooth_scale_len != static_cast<size_t>(up->k) ||
            down->smooth_scale_len != static_cast<size_t>(down->k)) {
        return fail("invalid smooth scale");
    }
    if (collect_profile) {
        profile.validation_us = static_cast<uint64_t>(ggml_time_us() - stage_start_us);
        stage_start_us = ggml_time_us();
    }
    const npu_decode_preloaded_tensor * gate_preloaded = npu_decode_lookup_preloaded_tensor(
            gate->weight_name, gate->packed_k, gate->out_channels, gate->q4_nb1, gate->k);
    const npu_decode_preloaded_tensor * up_preloaded = npu_decode_lookup_preloaded_tensor(
            up->weight_name, up->packed_k, up->out_channels, up->q4_nb1, up->k);
    const npu_decode_preloaded_tensor * down_preloaded = npu_decode_lookup_preloaded_tensor(
            down->weight_name, down->packed_k, down->out_channels, down->q4_nb1, down->k);
    if (gate_preloaded == nullptr || up_preloaded == nullptr || down_preloaded == nullptr ||
            gate_preloaded->blocks.empty() || up_preloaded->blocks.empty() || down_preloaded->blocks.empty()) {
        return fail("decode tensors are not preloaded");
    }
    if (!gate_preloaded->zero_validated || !up_preloaded->zero_validated || !down_preloaded->zero_validated) {
        return fail("decode tensor zero points are not validated");
    }
    if (gate_preloaded->act_scale.size() != gate_preloaded->max_layout.act_bytes ||
            up_preloaded->act_scale.size() != up_preloaded->max_layout.act_bytes ||
            down_preloaded->act_scale.size() != down_preloaded->max_layout.act_bytes) {
        return fail("preloaded act scale size mismatch");
    }
    if (collect_profile) {
        profile.preload_lookup_us = static_cast<uint64_t>(ggml_time_us() - stage_start_us);
        stage_start_us = ggml_time_us();
    }

    const npu_decode_gemv_layout gate_layout = gate_preloaded->max_layout;
    const npu_decode_gemv_layout up_layout = up_preloaded->max_layout;
    const npu_decode_gemv_layout down_layout = down_preloaded->max_layout;
    const uint32_t act_base = NPU_DECODE_GEMV_ACT_BASE;
    const uint32_t input_act_payload_bytes = std::max(
            npu_decode_act_payload_bytes(gate_layout),
            npu_decode_act_payload_bytes(up_layout));
    const uint32_t swiglu_act_base = npu_decode_align_up_u32(input_act_payload_bytes, NPU_DECODE_GEMV_LINE_BYTES);
    if (swiglu_act_base + npu_decode_act_payload_bytes(down_layout) > NPU_DECODE_GEMV_ACT_BUFFER_BYTES) {
        return fail("SwiGLU activation does not fit act buffer");
    }
    const uint32_t output_spm_bytes = npu_decode_align_up_u32(
            static_cast<uint32_t>(down->out_channels) * static_cast<uint32_t>(sizeof(ggml_fp16_t)),
            NPU_DECODE_GEMV_LINE_BYTES);
    const uint32_t host_output_bytes = npu_decode_host_output_bytes(static_cast<uint32_t>(down->out_channels), hw_f32_mvout);
    if (output_spm_bytes > NPU_DECODE_GEMV_SPM_BYTES) {
        return fail("decode output exceeds output SPM capacity");
    }
    if (collect_profile) {
        profile.layout_us = static_cast<uint64_t>(ggml_time_us() - stage_start_us);
        stage_start_us = ggml_time_us();
    }
    bool preloaded_cma = false;

    if (!npu_decode_ensure_overlay_active(op_name)) {
        return fail("decode overlay switch failed");
    }
    if (collect_profile) {
        profile.overlay_ensure_us = static_cast<uint64_t>(ggml_time_us() - stage_start_us);
        stage_start_us = ggml_time_us();
    }
    auto tensor_pingpong_cma_resident = [](const npu_decode_preloaded_tensor * tensor) {
        return npu_decode_stream_w4_cma_resident(tensor);
    };
    bool ffn_stream_cma =
        npu_decode_stream_api_enabled() &&
        tensor_pingpong_cma_resident(gate_preloaded) &&
        tensor_pingpong_cma_resident(up_preloaded) &&
        tensor_pingpong_cma_resident(down_preloaded);
    bool ffn_pingpong_cma =
        npu_decode_fused_ffn_pingpong_enabled() &&
        tensor_pingpong_cma_resident(gate_preloaded) &&
        tensor_pingpong_cma_resident(up_preloaded) &&
        tensor_pingpong_cma_resident(down_preloaded);
    if (preloaded_cma || ffn_pingpong_cma || ffn_stream_cma) {
        npu_decode_update_runtime_heap_env();
    }
    if (collect_profile) {
        profile.heap_update_us = static_cast<uint64_t>(ggml_time_us() - stage_start_us);
        stage_start_us = ggml_time_us();
    }
    if (npu_decode_init_with_cma_retry(op_name, true) != 0) {
        return fail("decode runtime init failed");
    }
    if ((npu_decode_fused_ffn_pingpong_enabled() || npu_decode_stream_api_enabled()) &&
            !ffn_pingpong_cma && !ffn_stream_cma) {
        (void) npu_decode_reserve_preloaded_ffn_pingpong_cma_after_init({
            const_cast<npu_decode_preloaded_tensor *>(gate_preloaded),
            const_cast<npu_decode_preloaded_tensor *>(up_preloaded),
            const_cast<npu_decode_preloaded_tensor *>(down_preloaded),
        });
    }
    if (collect_profile) {
        profile.runtime_init_us = static_cast<uint64_t>(ggml_time_us() - stage_start_us);
        stage_start_us = ggml_time_us();
    }
    preloaded_cma = true;
    for (const npu_decode_preloaded_tensor * tensor : { gate_preloaded, up_preloaded, down_preloaded }) {
        for (const npu_decode_preloaded_block & block : tensor->blocks) {
            if (block.cma_bytes == 0 || block.cma_bytes != block.packed.size()) {
                preloaded_cma = false;
                break;
            }
        }
        if (!preloaded_cma) {
            break;
        }
    }
    ffn_stream_cma =
        npu_decode_stream_api_enabled() &&
        tensor_pingpong_cma_resident(gate_preloaded) &&
        tensor_pingpong_cma_resident(up_preloaded) &&
        tensor_pingpong_cma_resident(down_preloaded);
    ffn_pingpong_cma =
        npu_decode_fused_ffn_pingpong_enabled() &&
        tensor_pingpong_cma_resident(gate_preloaded) &&
        tensor_pingpong_cma_resident(up_preloaded) &&
        tensor_pingpong_cma_resident(down_preloaded);
    void * decode_cma_base = (preloaded_cma || ffn_pingpong_cma || ffn_stream_cma) ? npu_decode_memory_base() : nullptr;
    if (decode_cma_base == nullptr) {
        preloaded_cma = false;
        ffn_pingpong_cma = false;
        ffn_stream_cma = false;
    }
    if (collect_profile) {
        profile.memory_base_us = static_cast<uint64_t>(ggml_time_us() - stage_start_us);
        stage_start_us = ggml_time_us();
    }

    uint32_t max_packed_bytes = 0;
    for (const npu_decode_preloaded_tensor * tensor : { gate_preloaded, up_preloaded, down_preloaded }) {
        for (const npu_decode_preloaded_block & block : tensor->blocks) {
            max_packed_bytes = std::max(max_packed_bytes, block.layout.packed_bytes);
        }
    }

    const uint32_t act_cma_bytes = std::max(input_act_payload_bytes, down_layout.act_bytes);
    void * act_cma = npu_decode_mem_alloc(act_cma_bytes);
    void * packed_cma = (preloaded_cma || ffn_pingpong_cma || ffn_stream_cma) ? nullptr : npu_decode_mem_alloc(max_packed_bytes);
    void * out_cma = npu_decode_mem_alloc(host_output_bytes);
    if (act_cma == nullptr || (!preloaded_cma && !ffn_pingpong_cma && !ffn_stream_cma && packed_cma == nullptr) || out_cma == nullptr) {
        npu_decode_mem_free(act_cma);
        npu_decode_mem_free(packed_cma);
        npu_decode_mem_free(out_cma);
        return fail("decode CMA allocation failed");
    }
    if (collect_profile) {
        profile.alloc_us = static_cast<uint64_t>(ggml_time_us() - stage_start_us);
    }
    if (collect_profile) {
        profile.max_block_rows = std::max(gate_preloaded->max_block_rows, down_preloaded->max_block_rows);
        profile.row_blocks_per_col =
            static_cast<uint32_t>(gate_preloaded->blocks.size() + down_preloaded->blocks.size());
        profile.preloaded_host = true;
        profile.preloaded_cma = preloaded_cma || ffn_pingpong_cma;
        profile.pingpong = ffn_pingpong_cma;
        if (npu_decode_fused_ffn_pingpong_enabled() && !ffn_pingpong_cma) {
            profile.fallback_reason = "ffn pingpong CMA not resident";
        }
        profile.activation_bytes =
            npu_decode_act_payload_bytes(gate_layout) +
            npu_decode_act_payload_bytes(up_layout) +
            npu_decode_act_payload_bytes(down_layout);
        profile.setup_us = static_cast<uint64_t>(ggml_time_us() - setup_start_us);
    }

    auto cleanup = [&]() {
        npu_decode_mem_free(act_cma);
        if (!preloaded_cma && !ffn_pingpong_cma && !ffn_stream_cma) {
            npu_decode_mem_free(packed_cma);
        }
        npu_decode_mem_free(out_cma);
    };

    stage_start_us = collect_profile ? ggml_time_us() : 0;
    npu_decode_reset();
    if (collect_profile) {
        profile.reset_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
    }

    if (ffn_stream_cma) {
        const uint32_t hidden_rows = static_cast<uint32_t>(gate->out_channels);
        const uint32_t hidden_bytes = npu_decode_align_up_u32(
                hidden_rows * static_cast<uint32_t>(sizeof(ggml_fp16_t)),
                NPU_DECODE_GEMV_DMA_ALIGN);
        void * gate_cma = npu_decode_mem_alloc(hidden_bytes);
        void * up_cma = npu_decode_mem_alloc(hidden_bytes);
        if (gate_cma == nullptr || up_cma == nullptr) {
            npu_decode_mem_free(gate_cma);
            npu_decode_mem_free(up_cma);
            cleanup();
            return fail("decode FFN stream intermediate CMA allocation failed");
        }
        auto cleanup_stream_intermediates = [&]() {
            npu_decode_mem_free(gate_cma);
            npu_decode_mem_free(up_cma);
        };

        auto stream_weight_ptr = [decode_cma_base](const npu_decode_preloaded_tensor * tensor) {
            return npu_decode_stream_cma_ptr(decode_cma_base, tensor->pingpong_weight_cma_offset);
        };
        auto stream_weight_scale_ptr = [decode_cma_base](const npu_decode_preloaded_tensor * tensor) {
            return npu_decode_stream_cma_ptr(decode_cma_base, tensor->pingpong_scale_cma_offset);
        };
        auto stream_act_scale_ptr = [decode_cma_base](const npu_decode_preloaded_tensor * tensor) {
            return npu_decode_stream_cma_ptr(decode_cma_base, tensor->act_scale_cma_offset);
        };
        auto run_w4_stream = [&](const npu_decode_preloaded_tensor * tensor,
                                 const npu_decode_gemv_layout & layout,
                                 const char * stats_label,
                                 void * act_ptr,
                                 void * act_scale_ptr,
                                 void * act_scale2_ptr,
                                 void * output_ptr,
                                 uint16_t rows,
                                 uint16_t cols,
                                 kv260_stream_post_op post_op,
                                 kv260_stream_output_precision precision) {
            kv260_stream_gemv_desc desc = {};
            desc.act_ptr = act_ptr;
            desc.act_scale_ptr = act_scale_ptr;
            desc.act_scale2_ptr = act_scale2_ptr;
            desc.weight_payload_ptr = stream_weight_ptr(tensor);
            desc.weight_scale_ptr = stream_weight_scale_ptr(tensor);
            desc.output_ptr = output_ptr;
            desc.m = rows;
            desc.n = cols;
            desc.mode = KV260_STREAM_DECODE_GEMV_W4A16;
            desc.output_precision = precision;
            desc.role = act_scale2_ptr != nullptr ? KV260_STREAM_GEMV_ROLE_LINEAR : KV260_STREAM_GEMV_ROLE_MLP;
            desc.dst = KV260_STREAM_GEMV_DST_OUTPUT;
            desc.post_op = post_op;
            desc.flags = KV260_STREAM_GEMV_F_ENABLE_ACT_SCALE;
            if (act_scale2_ptr != nullptr) {
                desc.flags |= KV260_STREAM_GEMV_F_ENABLE_ACT_SCALE2;
            }
            npu_decode_dump_act_scale_stats_packed(
                    stats_label,
                    op_name,
                    npu_decode_parse_blk_layer_id(op_name),
                    -1,
                    layout,
                    cols,
                    act_ptr,
                    act_scale_ptr,
                    act_scale2_ptr);
            return npu_decode_stream_gemv(op_name, desc);
        };

        stage_start_us = collect_profile ? ggml_time_us() : 0;
        npu_decode_pack_activation_slice(
                act_cma,
                gate_layout.act_bytes,
                gate_layout,
                act_data,
                act_type,
                act_nb0,
                act_nb1,
                gate->k,
                0,
                gate->k,
                0);
        if (collect_profile) {
            profile.activation_preprocess_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            stage_start_us = ggml_time_us();
        }

        int rc = run_w4_stream(
                gate_preloaded,
                gate_layout,
                "ffn_gate_hidden_scale",
                act_cma,
                stream_act_scale_ptr(gate_preloaded),
                nullptr,
                gate_cma,
                static_cast<uint16_t>(gate->out_channels),
                static_cast<uint16_t>(gate->k),
                KV260_STREAM_POST_SILU,
                KV260_STREAM_OUTPUT_FP16);
        if (rc != 0) {
            cleanup_stream_intermediates();
            cleanup();
            return fail("decode FFN stream gate failed");
        }
        if (collect_profile) {
            profile.pingpong_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            profile.pingpong_calls += 1;
            profile.weight_bytes += gate_preloaded->pingpong_layout.weight_bytes;
            profile.output_bytes += hidden_bytes;
            stage_start_us = ggml_time_us();
        }

        rc = run_w4_stream(
                up_preloaded,
                up_layout,
                "ffn_up_hidden_scale",
                act_cma,
                stream_act_scale_ptr(up_preloaded),
                nullptr,
                up_cma,
                static_cast<uint16_t>(up->out_channels),
                static_cast<uint16_t>(up->k),
                KV260_STREAM_POST_BYPASS,
                KV260_STREAM_OUTPUT_FP16);
        if (rc != 0) {
            cleanup_stream_intermediates();
            cleanup();
            return fail("decode FFN stream up failed");
        }
        if (collect_profile) {
            profile.pingpong_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            profile.pingpong_calls += 1;
            profile.weight_bytes += up_preloaded->pingpong_layout.weight_bytes;
            profile.output_bytes += hidden_bytes;
            stage_start_us = ggml_time_us();
        }

        rc = run_w4_stream(
                down_preloaded,
                down_layout,
                "ffn_down_up_silu_scale",
                up_cma,
                gate_cma,
                stream_act_scale_ptr(down_preloaded),
                out_cma,
                static_cast<uint16_t>(down->out_channels),
                static_cast<uint16_t>(down->k),
                KV260_STREAM_POST_BYPASS,
                npu_decode_stream_output_precision(hw_f32_mvout));
        if (rc != 0) {
            cleanup_stream_intermediates();
            cleanup();
            return fail("decode FFN stream down failed");
        }
        if (collect_profile) {
            profile.pingpong_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            profile.pingpong_calls += 1;
            profile.weight_bytes += down_preloaded->pingpong_layout.weight_bytes;
            profile.output_bytes += host_output_bytes;
            stage_start_us = ggml_time_us();
        }

        char * dst_col = reinterpret_cast<char *>(dst_data);
        if (dst_type == GGML_TYPE_F16) {
            std::memcpy(dst_col, out_cma, static_cast<size_t>(down->out_channels) * sizeof(ggml_fp16_t));
        } else if (hw_f32_mvout) {
            std::memcpy(dst_col, out_cma, static_cast<size_t>(down->out_channels) * sizeof(float));
        } else {
            npu_decode_convert_output_fp16_to_f32(
                    static_cast<const ggml_fp16_t *>(out_cma),
                    reinterpret_cast<float *>(dst_col),
                    static_cast<uint32_t>(down->out_channels));
        }
        if (collect_profile) {
            profile.postprocess_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
        }

        cleanup_stream_intermediates();
        const int64_t cleanup_start_us = collect_profile ? ggml_time_us() : 0;
        cleanup();
        if (collect_profile) {
            profile.cleanup_us = static_cast<uint64_t>(ggml_time_us() - cleanup_start_us);
            profile.total_us = static_cast<uint64_t>(ggml_time_us() - total_start_us);
            npu_decode_profile_write(profile);
        }
        GGML_UNUSED(dst_nb1);
        return true;
    }

    if (npu_decode_stream_api_enabled()) {
        cleanup();
        return fail("SwiGLU FFN stream assets are not CMA resident; split/legacy NPU path disabled");
    }

    if (gate_preloaded->blocks.size() != up_preloaded->blocks.size()) {
        cleanup();
        return fail("gate/up block count mismatch");
    }

    auto pingpong_scale_bytes_for =
        [](const npu_decode_preloaded_tensor * tensor, uint32_t block_rows) {
            return npu_decode_ceil_div_u32(block_rows, NPU_DECODE_GEMV_M_TILE) *
                tensor->pingpong_layout.col_tiles * NPU_DECODE_GEMV_SCALE_TILE_BYTES;
        };
    auto pingpong_weight_bytes_for =
        [](const npu_decode_preloaded_tensor * tensor, uint32_t block_rows) {
            const uint32_t bytes_per_row_tile =
                tensor->pingpong_layout.col_tiles * NPU_DECODE_GEMV_WEIGHT_TILE_BYTES;
            return npu_decode_ceil_div_u32(block_rows, NPU_DECODE_GEMV_M_TILE) * bytes_per_row_tile;
        };
    auto pingpong_weight_ptr_for =
        [decode_cma_base](const npu_decode_preloaded_tensor * tensor, uint32_t row_base) {
            const uint32_t bytes_per_row_tile =
                tensor->pingpong_layout.col_tiles * NPU_DECODE_GEMV_WEIGHT_TILE_BYTES;
            const uint32_t offset =
                (row_base / NPU_DECODE_GEMV_M_TILE) * bytes_per_row_tile;
            return static_cast<uint8_t *>(decode_cma_base) + tensor->pingpong_weight_cma_offset + offset;
        };
    auto pingpong_scale_ptr_for =
        [decode_cma_base](const npu_decode_preloaded_tensor * tensor, uint32_t row_base) {
            const uint32_t offset =
                (row_base / NPU_DECODE_GEMV_M_TILE) *
                tensor->pingpong_layout.col_tiles * NPU_DECODE_GEMV_SCALE_TILE_BYTES;
            return static_cast<uint8_t *>(decode_cma_base) + tensor->pingpong_scale_cma_offset + offset;
        };
    auto mvin_pingpong_scale =
        [&](const npu_decode_preloaded_tensor * tensor, uint32_t row_base, uint32_t block_rows) {
            const uint32_t scale_bytes = pingpong_scale_bytes_for(tensor, block_rows);
            stage_start_us = collect_profile ? ggml_time_us() : 0;
            npu_decode_stream_unsupported_void(
                    op_name,
                    pingpong_scale_ptr_for(tensor, row_base),
                    NPU_DECODE_GEMV_SCALE_BASE,
                    scale_bytes,
                    2,
                    NPU_DECODE_GEMV_INPUT_TYPE_DATA,
                    0,
                    0);
            if (collect_profile) {
                profile.scale_mvin_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
                profile.scale_mvin_calls += 1;
                profile.scale_bytes += scale_bytes;
            }
        };
    auto pingpong_scale_addr_for =
        [](const npu_decode_preloaded_tensor * tensor, uint32_t row_base) {
            GGML_UNUSED(tensor);
            GGML_UNUSED(row_base);
            return NPU_DECODE_GEMV_SCALE_BASE;
        };
    auto mvin_pingpong_weight_sync =
        [&](const npu_decode_preloaded_tensor * tensor, uint32_t row_base, uint32_t block_rows, uint32_t weight_spm) {
            const uint32_t weight_bytes = pingpong_weight_bytes_for(tensor, block_rows);
            stage_start_us = collect_profile ? ggml_time_us() : 0;
            npu_decode_stream_unsupported_void(
                    op_name,
                    pingpong_weight_ptr_for(tensor, row_base),
                    weight_spm,
                    weight_bytes,
                    1,
                    NPU_DECODE_GEMV_INPUT_TYPE_WEIGHT);
            if (collect_profile) {
                profile.mvin_weight_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
                profile.weight_mvin_calls += 1;
                profile.weight_bytes += weight_bytes;
            }
        };
    auto issue_pingpong_weight_prefetch =
        [&](const npu_decode_preloaded_tensor * tensor,
                uint32_t row_base,
                uint32_t block_rows,
                uint32_t weight_spm,
                int64_t * issue_done_us) {
            const uint32_t weight_bytes = pingpong_weight_bytes_for(tensor, block_rows);
            MvinConfig next_mvin = npu_decode_make_mvin_cfg(
                    pingpong_weight_ptr_for(tensor, row_base),
                    weight_spm,
                    weight_bytes,
                    NPU_DECODE_GEMV_INPUT_TYPE_WEIGHT,
                    1);
            stage_start_us = collect_profile ? ggml_time_us() : 0;
            npu_decode_stream_unsupported_void(0, &next_mvin);
            if (collect_profile) {
                *issue_done_us = ggml_time_us();
                profile.pingpong_prefetch_issue_us += static_cast<uint64_t>(*issue_done_us - stage_start_us);
                profile.pingpong_async_mvin_calls += 1;
                profile.weight_bytes += weight_bytes;
            }
        };
    auto wait_pingpong_prefetch = [&](int64_t issue_done_us) {
        if (collect_profile) {
            profile.pingpong_prefetch_window_us += static_cast<uint64_t>(ggml_time_us() - issue_done_us);
            stage_start_us = ggml_time_us();
        }
        npu_decode_stream_unsupported_void(1u << 0);
        if (collect_profile) {
            profile.pingpong_prefetch_wait_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
        }
    };

    if (ffn_pingpong_cma) {
        const uint32_t gate_spm = NPU_DECODE_GEMV_PING_WEIGHT_BASE;
        const uint32_t up_spm = NPU_DECODE_GEMV_PONG_WEIGHT_BASE;
        const uint32_t hidden_rows = static_cast<uint32_t>(gate->out_channels);
        const uint32_t gate_up_max_rows = std::min(
                npu_decode_pingpong_max_block_rows(gate_preloaded->pingpong_layout),
                npu_decode_pingpong_max_block_rows(up_preloaded->pingpong_layout));
        const uint32_t down_rows_total = static_cast<uint32_t>(down->out_channels);
        const uint32_t down_max_rows = npu_decode_pingpong_max_block_rows(down_preloaded->pingpong_layout);
        if (gate_up_max_rows == 0 || down_max_rows == 0) {
            cleanup();
            return fail("SwiGLU pingpong row block does not fit");
        }
        const uint32_t up_act_base = npu_decode_align_up_u32(
                swiglu_act_base + hidden_rows * static_cast<uint32_t>(sizeof(ggml_fp16_t)),
                NPU_DECODE_GEMV_LINE_BYTES);
        if (up_act_base + npu_decode_act_payload_bytes(up_layout) > NPU_DECODE_GEMV_ACT_BUFFER_BYTES) {
            cleanup();
            return fail("SwiGLU pingpong activation layout does not fit");
        }
        if (collect_profile) {
            profile.row_blocks_per_col =
                npu_decode_ceil_div_u32(hidden_rows, gate_up_max_rows) +
                npu_decode_ceil_div_u32(down_rows_total, down_max_rows);
        }

        stage_start_us = collect_profile ? ggml_time_us() : 0;
        npu_decode_pack_activation(
                act_cma,
                npu_decode_act_payload_bytes(gate_layout),
                gate_layout,
                act_data,
                act_type,
                act_nb0,
                act_nb1,
                gate_preloaded->act_scale.empty() ? nullptr : gate_preloaded->act_scale.data(),
                gate->k,
                0);
        if (collect_profile) {
            profile.activation_preprocess_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            stage_start_us = ggml_time_us();
        }
        npu_decode_stream_unsupported_void(
                op_name,
                act_cma,
                act_base,
                npu_decode_act_payload_bytes(gate_layout),
                2,
                NPU_DECODE_GEMV_INPUT_TYPE_ACT);
        if (collect_profile) {
            profile.mvin_activation_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            profile.activation_mvin_calls += 1;
        }

        stage_start_us = collect_profile ? ggml_time_us() : 0;
        npu_decode_pack_activation(
                act_cma,
                npu_decode_act_payload_bytes(up_layout),
                up_layout,
                act_data,
                act_type,
                act_nb0,
                act_nb1,
                up_preloaded->act_scale.empty() ? nullptr : up_preloaded->act_scale.data(),
                up->k,
                0);
        if (collect_profile) {
            profile.activation_preprocess_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            stage_start_us = ggml_time_us();
        }
        npu_decode_stream_unsupported_void(
                op_name,
                act_cma,
                up_act_base,
                npu_decode_act_payload_bytes(up_layout),
                2,
                NPU_DECODE_GEMV_INPUT_TYPE_ACT);
        if (collect_profile) {
            profile.mvin_activation_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            profile.activation_mvin_calls += 1;
        }

        const uint32_t first_gate_rows = std::min(gate_up_max_rows, hidden_rows);
        mvin_pingpong_weight_sync(gate_preloaded, 0, first_gate_rows, gate_spm);

        uint32_t gate_up_block_idx = 0;
        for (uint32_t row_base = 0; row_base < hidden_rows; row_base += std::min(gate_up_max_rows, hidden_rows - row_base), ++gate_up_block_idx) {
            const uint32_t block_rows = std::min(gate_up_max_rows, hidden_rows - row_base);
            if (block_rows > 4096) {
                cleanup();
                return fail("SwiGLU block exceeds stream buffer");
            }

            mvin_pingpong_scale(gate_preloaded, row_base, block_rows);

            int64_t up_prefetch_issue_done_us = 0;
            issue_pingpong_weight_prefetch(up_preloaded, row_base, block_rows, up_spm, &up_prefetch_issue_done_us);
            stage_start_us = collect_profile ? ggml_time_us() : 0;
            const uint64_t gate_flow = npu_decode_stream_unsupported_flow(
                    NPU_DECODE_SRC_GEMV_STREAM,
                    0,
                    NPU_DECODE_UNARY_SILU,
                    NPU_DECODE_BINARY_BYPASS,
                    NPU_DECODE_REDUCE_BYPASS,
                    NPU_DECODE_DST_STREAM_BUFFER,
                    0,
                    0,
                    static_cast<uint16_t>(block_rows),
                    0);
            npu_decode_stream_unsupported_void(
                    op_name,
                    gate_spm,
                    act_base,
                    static_cast<uint16_t>(gate->k),
                    static_cast<uint16_t>(block_rows),
                    static_cast<uint16_t>(NPU_DECODE_GEMV_OUTPUT_BASE),
                    static_cast<uint16_t>(pingpong_scale_addr_for(gate_preloaded, row_base)),
                    NPU_DECODE_GEMV_MODE_W4A16,
                    gate_flow,
                    act_base + gate_layout.act_bytes);
            if (collect_profile) {
                profile.gemv_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
                profile.gemv_calls += 1;
                profile.pingpong_gemv_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
                profile.pingpong_block_calls += 1;
            }
            wait_pingpong_prefetch(up_prefetch_issue_done_us);

            mvin_pingpong_scale(up_preloaded, row_base, block_rows);

            bool gate_prefetch_started = false;
            int64_t gate_prefetch_issue_done_us = 0;
            const uint32_t next_row_base = row_base + block_rows;
            if (next_row_base < hidden_rows) {
                const uint32_t next_block_rows = std::min(gate_up_max_rows, hidden_rows - next_row_base);
                issue_pingpong_weight_prefetch(
                        gate_preloaded,
                        next_row_base,
                        next_block_rows,
                        gate_spm,
                        &gate_prefetch_issue_done_us);
                gate_prefetch_started = true;
            }
            const uint64_t flow = npu_decode_stream_unsupported_flow(
                    NPU_DECODE_SRC_GEMV_STREAM,
                    NPU_DECODE_SRC_STREAM_BUFFER,
                    NPU_DECODE_UNARY_BYPASS,
                    NPU_DECODE_BINARY_FP16_MUL,
                    NPU_DECODE_REDUCE_BYPASS,
                    NPU_DECODE_DST_ACT_BUFFER,
                    0,
                    0,
                    static_cast<uint16_t>(block_rows),
                    0);
            npu_decode_trace_stream_flow(
                    "fused_ffn_up_pingpong",
                    op_name,
                    gate_up_block_idx,
                    row_base,
                    block_rows,
                    static_cast<uint32_t>(up->k),
                    NPU_DECODE_GEMV_MODE_W4A16,
                    up_spm,
                    up_act_base,
                    static_cast<uint16_t>(swiglu_act_base + row_base * sizeof(ggml_fp16_t)),
                    pingpong_scale_addr_for(up_preloaded, row_base),
                    flow);
            stage_start_us = collect_profile ? ggml_time_us() : 0;
            npu_decode_stream_unsupported_void(
                    op_name,
                    up_spm,
                    up_act_base,
                    static_cast<uint16_t>(up->k),
                    static_cast<uint16_t>(block_rows),
                    static_cast<uint16_t>(swiglu_act_base + row_base * sizeof(ggml_fp16_t)),
                    static_cast<uint16_t>(pingpong_scale_addr_for(up_preloaded, row_base)),
                    NPU_DECODE_GEMV_MODE_W4A16,
                    flow,
                    up_act_base + up_layout.act_bytes);
            if (collect_profile) {
                profile.gemv_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
                profile.gemv_calls += 1;
                profile.pingpong_gemv_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
                profile.pingpong_block_calls += 1;
            }
            if (gate_prefetch_started) {
                wait_pingpong_prefetch(gate_prefetch_issue_done_us);
            }
        }

        if (down_preloaded->act_scale.size() != down_layout.act_bytes) {
            cleanup();
            return fail("down act scale size mismatch");
        }
        stage_start_us = collect_profile ? ggml_time_us() : 0;
        std::memcpy(act_cma, down_preloaded->act_scale.data(), down_layout.act_bytes);
        if (collect_profile) {
            profile.activation_preprocess_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            stage_start_us = ggml_time_us();
        }
        npu_decode_stream_unsupported_void(
                op_name,
                act_cma,
                swiglu_act_base + down_layout.act_bytes,
                down_layout.act_bytes,
                2,
                NPU_DECODE_GEMV_INPUT_TYPE_ACT);
        if (collect_profile) {
            profile.mvin_activation_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            profile.activation_mvin_calls += 1;
        }

        const uint32_t down_ping_spm = NPU_DECODE_GEMV_PING_WEIGHT_BASE;
        const uint32_t down_pong_spm = NPU_DECODE_GEMV_PONG_WEIGHT_BASE;
        const uint32_t first_down_rows = std::min(down_max_rows, down_rows_total);
        mvin_pingpong_weight_sync(down_preloaded, 0, first_down_rows, down_ping_spm);
        uint32_t down_block_idx = 0;
        for (uint32_t row_base = 0; row_base < down_rows_total; row_base += std::min(down_max_rows, down_rows_total - row_base), ++down_block_idx) {
            const uint32_t block_rows = std::min(down_max_rows, down_rows_total - row_base);
            const uint32_t cur_spm = (down_block_idx & 1u) ? down_pong_spm : down_ping_spm;
            const uint32_t next_spm = (down_block_idx & 1u) ? down_ping_spm : down_pong_spm;
            mvin_pingpong_scale(down_preloaded, row_base, block_rows);

            bool prefetch_started = false;
            int64_t prefetch_issue_done_us = 0;
            const uint32_t next_row_base = row_base + block_rows;
            if (next_row_base < down_rows_total) {
                const uint32_t next_block_rows = std::min(down_max_rows, down_rows_total - next_row_base);
                issue_pingpong_weight_prefetch(
                        down_preloaded,
                        next_row_base,
                        next_block_rows,
                        next_spm,
                        &prefetch_issue_done_us);
                prefetch_started = true;
            }

            const uint64_t flow = npu_decode_make_bypass_output_flow(static_cast<uint16_t>(block_rows));
            npu_decode_trace_stream_flow(
                    "fused_ffn_down_pingpong",
                    op_name,
                    down_block_idx,
                    row_base,
                    block_rows,
                    static_cast<uint32_t>(down->k),
                    NPU_DECODE_GEMV_MODE_W4A16,
                    cur_spm,
                    swiglu_act_base,
                    static_cast<uint16_t>(row_base * sizeof(ggml_fp16_t)),
                    pingpong_scale_addr_for(down_preloaded, row_base),
                    flow);
            stage_start_us = collect_profile ? ggml_time_us() : 0;
            npu_decode_stream_unsupported_void(
                    op_name,
                    cur_spm,
                    swiglu_act_base,
                    static_cast<uint16_t>(down->k),
                    static_cast<uint16_t>(block_rows),
                    static_cast<uint16_t>(row_base * sizeof(ggml_fp16_t)),
                    static_cast<uint16_t>(pingpong_scale_addr_for(down_preloaded, row_base)),
                    NPU_DECODE_GEMV_MODE_W4A16,
                    flow,
                    swiglu_act_base + down_layout.act_bytes);
            if (collect_profile) {
                profile.gemv_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
                profile.gemv_calls += 1;
                profile.pingpong_gemv_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
                profile.pingpong_block_calls += 1;
            }
            if (prefetch_started) {
                wait_pingpong_prefetch(prefetch_issue_done_us);
            }
        }
        if (collect_profile) {
            profile.pingpong_calls += 1;
            profile.pingpong_us =
                profile.pingpong_initial_mvin_us +
                profile.pingpong_prefetch_issue_us +
                profile.pingpong_prefetch_wait_us +
                profile.pingpong_gemv_us +
                profile.pingpong_mvout_us;
        }
    } else {
    for (size_t i = 0; i < gate_preloaded->blocks.size(); ++i) {
        const npu_decode_preloaded_block & gate_block = gate_preloaded->blocks[i];
        const npu_decode_preloaded_block & up_block = up_preloaded->blocks[i];
        if (gate_block.row_base != up_block.row_base || gate_block.block_rows != up_block.block_rows) {
            cleanup();
            return fail("gate/up block layout mismatch");
        }
        if (gate_block.block_rows > 4096) {
            cleanup();
            return fail("SwiGLU block exceeds stream buffer");
        }
        if (gate_block.packed.size() != gate_block.layout.packed_bytes ||
                up_block.packed.size() != up_block.layout.packed_bytes ||
                gate_block.layout.packed_bytes > max_packed_bytes ||
                up_block.layout.packed_bytes > max_packed_bytes) {
                cleanup();
                return fail("preloaded block size mismatch");
        }

        stage_start_us = collect_profile ? ggml_time_us() : 0;
        npu_decode_pack_activation(
                act_cma,
                npu_decode_act_payload_bytes(gate_layout),
                gate_layout,
                act_data,
                act_type,
                act_nb0,
                act_nb1,
                gate_preloaded->act_scale.empty() ? nullptr : gate_preloaded->act_scale.data(),
                gate->k,
                0);
        if (collect_profile) {
            profile.activation_preprocess_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            stage_start_us = ggml_time_us();
        }
        npu_decode_stream_unsupported_void(
                act_cma,
                act_base,
                npu_decode_act_payload_bytes(gate_layout) - 1u,
                0,
                0,
                0,
                2,
                NPU_DECODE_GEMV_INPUT_TYPE_ACT,
                false,
                false,
                false,
                0,
                0,
                0);
        if (collect_profile) {
            profile.mvin_activation_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            profile.activation_mvin_calls += 1;
        }

        void * gate_weight_mvin_ptr = packed_cma;
        if (preloaded_cma && gate_block.cma_bytes == gate_block.layout.packed_bytes) {
            gate_weight_mvin_ptr = static_cast<uint8_t *>(decode_cma_base) + gate_block.cma_offset;
        } else {
            stage_start_us = collect_profile ? ggml_time_us() : 0;
            std::memcpy(packed_cma, gate_block.packed.data(), gate_block.layout.packed_bytes);
            if (collect_profile) {
                profile.host_copy_weight_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            }
        }
        stage_start_us = collect_profile ? ggml_time_us() : 0;
        npu_decode_stream_unsupported_void(
                gate_weight_mvin_ptr,
                0,
                gate_block.layout.packed_bytes - 1u,
                0,
                0,
                0,
                1,
                NPU_DECODE_GEMV_INPUT_TYPE_WEIGHT,
                false,
                false,
                false,
                0,
                0,
                0);
        if (collect_profile) {
            profile.mvin_weight_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            profile.weight_mvin_calls += 1;
            profile.weight_bytes += gate_block.layout.packed_bytes;
            stage_start_us = ggml_time_us();
        }
        const uint64_t gate_flow = npu_decode_stream_unsupported_flow(
                NPU_DECODE_SRC_GEMV_STREAM,
                0,
                NPU_DECODE_UNARY_SILU,
                NPU_DECODE_BINARY_BYPASS,
                NPU_DECODE_REDUCE_BYPASS,
                NPU_DECODE_DST_STREAM_BUFFER,
                0,
                0,
                static_cast<uint16_t>(gate_block.block_rows),
                0);
        npu_decode_trace_stream_flow(
                "fused_ffn_gate",
                op_name,
                static_cast<uint32_t>(i),
                static_cast<uint32_t>(gate_block.row_base),
                static_cast<uint32_t>(gate_block.block_rows),
                static_cast<uint32_t>(gate->k),
                NPU_DECODE_GEMV_MODE_W4A16,
                gate_block.layout.weight_base,
                act_base,
                static_cast<uint16_t>(NPU_DECODE_GEMV_OUTPUT_BASE),
                NPU_DECODE_GEMV_SCALE_BASE,
                gate_flow);
        npu_decode_stream_unsupported_void(
                op_name,
                gate_block.layout.weight_base,
                act_base,
                static_cast<uint16_t>(gate->k),
                static_cast<uint16_t>(gate_block.block_rows),
                static_cast<uint16_t>(NPU_DECODE_GEMV_OUTPUT_BASE),
                NPU_DECODE_GEMV_SCALE_BASE,
                NPU_DECODE_GEMV_MODE_W4A16,
                gate_flow,
                act_base + gate_layout.act_bytes);
        if (collect_profile) {
            profile.gemv_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            profile.gemv_calls += 1;
        }

        stage_start_us = collect_profile ? ggml_time_us() : 0;
        npu_decode_pack_activation(
                act_cma,
                npu_decode_act_payload_bytes(up_layout),
                up_layout,
                act_data,
                act_type,
                act_nb0,
                act_nb1,
                up_preloaded->act_scale.empty() ? nullptr : up_preloaded->act_scale.data(),
                up->k,
                0);
        if (collect_profile) {
            profile.activation_preprocess_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            stage_start_us = ggml_time_us();
        }
        npu_decode_stream_unsupported_void(
                act_cma,
                act_base,
                npu_decode_act_payload_bytes(up_layout) - 1u,
                0,
                0,
                0,
                2,
                NPU_DECODE_GEMV_INPUT_TYPE_ACT,
                false,
                false,
                false,
                0,
                0,
                0);
        if (collect_profile) {
            profile.mvin_activation_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            profile.activation_mvin_calls += 1;
        }

        void * up_weight_mvin_ptr = packed_cma;
        if (preloaded_cma && up_block.cma_bytes == up_block.layout.packed_bytes) {
            up_weight_mvin_ptr = static_cast<uint8_t *>(decode_cma_base) + up_block.cma_offset;
        } else {
            stage_start_us = collect_profile ? ggml_time_us() : 0;
            std::memcpy(packed_cma, up_block.packed.data(), up_block.layout.packed_bytes);
            if (collect_profile) {
                profile.host_copy_weight_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            }
        }
        stage_start_us = collect_profile ? ggml_time_us() : 0;
        npu_decode_stream_unsupported_void(
                up_weight_mvin_ptr,
                0,
                up_block.layout.packed_bytes - 1u,
                0,
                0,
                0,
                1,
                NPU_DECODE_GEMV_INPUT_TYPE_WEIGHT,
                false,
                false,
                false,
                0,
                0,
                0);
        if (collect_profile) {
            profile.mvin_weight_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            profile.weight_mvin_calls += 1;
            profile.weight_bytes += up_block.layout.packed_bytes;
        }
        const uint64_t flow = npu_decode_stream_unsupported_flow(
                NPU_DECODE_SRC_GEMV_STREAM,
                NPU_DECODE_SRC_STREAM_BUFFER,
                NPU_DECODE_UNARY_BYPASS,
                NPU_DECODE_BINARY_FP16_MUL,
                NPU_DECODE_REDUCE_BYPASS,
                NPU_DECODE_DST_ACT_BUFFER,
                0,
                0,
                static_cast<uint16_t>(up_block.block_rows),
                0);
        npu_decode_trace_stream_flow(
                "fused_ffn_up",
                op_name,
                0,
                static_cast<uint32_t>(up_block.row_base),
                static_cast<uint32_t>(up_block.block_rows),
                static_cast<uint32_t>(up->k),
                NPU_DECODE_GEMV_MODE_W4A16,
                up_block.layout.weight_base,
                act_base,
                static_cast<uint16_t>(swiglu_act_base + up_block.row_base * sizeof(ggml_fp16_t)),
                NPU_DECODE_GEMV_SCALE_BASE,
                flow);
        stage_start_us = collect_profile ? ggml_time_us() : 0;
        npu_decode_stream_unsupported_void(
                op_name,
                up_block.layout.weight_base,
                act_base,
                static_cast<uint16_t>(up->k),
                static_cast<uint16_t>(up_block.block_rows),
                static_cast<uint16_t>(swiglu_act_base + up_block.row_base * sizeof(ggml_fp16_t)),
                NPU_DECODE_GEMV_SCALE_BASE,
                NPU_DECODE_GEMV_MODE_W4A16,
                flow);
        if (collect_profile) {
            profile.gemv_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            profile.gemv_calls += 1;
        }
    }

    if (down_preloaded->act_scale.size() != down_layout.act_bytes) {
        cleanup();
        return fail("down act scale size mismatch");
    }
    stage_start_us = collect_profile ? ggml_time_us() : 0;
    std::memcpy(act_cma, down_preloaded->act_scale.data(), down_layout.act_bytes);
    if (collect_profile) {
        profile.activation_preprocess_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
        stage_start_us = ggml_time_us();
    }
    npu_decode_stream_unsupported_void(
            act_cma,
            swiglu_act_base + down_layout.act_bytes,
            down_layout.act_bytes - 1u,
            0,
            0,
            0,
            2,
            NPU_DECODE_GEMV_INPUT_TYPE_ACT,
            false,
            false,
            false,
            0,
            0,
            0);
    if (collect_profile) {
        profile.mvin_activation_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
        profile.activation_mvin_calls += 1;
    }

    for (const npu_decode_preloaded_block & block : down_preloaded->blocks) {
        if (block.packed.size() != block.layout.packed_bytes || block.layout.packed_bytes > max_packed_bytes) {
            cleanup();
            return fail("down preloaded block size mismatch");
        }
        void * down_weight_mvin_ptr = packed_cma;
        if (preloaded_cma && block.cma_bytes == block.layout.packed_bytes) {
            down_weight_mvin_ptr = static_cast<uint8_t *>(decode_cma_base) + block.cma_offset;
        } else {
            stage_start_us = collect_profile ? ggml_time_us() : 0;
            std::memcpy(packed_cma, block.packed.data(), block.layout.packed_bytes);
            if (collect_profile) {
                profile.host_copy_weight_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            }
        }
        stage_start_us = collect_profile ? ggml_time_us() : 0;
        npu_decode_stream_unsupported_void(
                down_weight_mvin_ptr,
                0,
                block.layout.packed_bytes - 1u,
                0,
                0,
                0,
                1,
                NPU_DECODE_GEMV_INPUT_TYPE_WEIGHT,
                false,
                false,
                false,
                0,
                0,
                0);
        if (collect_profile) {
            profile.mvin_weight_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            profile.weight_mvin_calls += 1;
            profile.weight_bytes += block.layout.packed_bytes;
        }
        const uint64_t flow = npu_decode_make_bypass_output_flow(static_cast<uint16_t>(block.block_rows));
        npu_decode_trace_stream_flow(
                "fused_ffn_down",
                op_name,
                static_cast<uint32_t>(&block - down_preloaded->blocks.data()),
                static_cast<uint32_t>(block.row_base),
                static_cast<uint32_t>(block.block_rows),
                static_cast<uint32_t>(down->k),
                NPU_DECODE_GEMV_MODE_W4A16,
                block.layout.weight_base,
                swiglu_act_base,
                static_cast<uint16_t>(block.row_base * sizeof(ggml_fp16_t)),
                NPU_DECODE_GEMV_SCALE_BASE,
                flow);
        stage_start_us = collect_profile ? ggml_time_us() : 0;
        npu_decode_stream_unsupported_void(
                op_name,
                block.layout.weight_base,
                swiglu_act_base,
                static_cast<uint16_t>(down->k),
                static_cast<uint16_t>(block.block_rows),
                static_cast<uint16_t>(block.row_base * sizeof(ggml_fp16_t)),
                NPU_DECODE_GEMV_SCALE_BASE,
                NPU_DECODE_GEMV_MODE_W4A16,
                flow);
        if (collect_profile) {
            profile.gemv_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            profile.gemv_calls += 1;
        }
    }
    }

    stage_start_us = collect_profile ? ggml_time_us() : 0;
    npu_decode_stream_unsupported_void(
            op_name,
            out_cma,
            0,
            host_output_bytes,
            output_precision,
            0);
    if (collect_profile) {
        profile.mvout_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
        profile.mvout_calls += 1;
        profile.output_bytes += host_output_bytes;
        stage_start_us = ggml_time_us();
    }

    if (dst_type == GGML_TYPE_F16) {
        std::memcpy(dst_data, out_cma, static_cast<size_t>(down->out_channels) * sizeof(ggml_fp16_t));
    } else if (hw_f32_mvout) {
        std::memcpy(dst_data, out_cma, static_cast<size_t>(down->out_channels) * sizeof(float));
    } else {
        npu_decode_convert_output_fp16_to_f32(
                static_cast<const ggml_fp16_t *>(out_cma),
                reinterpret_cast<float *>(dst_data),
                static_cast<uint32_t>(down->out_channels));
    }
    if (collect_profile) {
        profile.postprocess_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
    }
    GGML_UNUSED(dst_nb1);

    const int64_t cleanup_start_us = collect_profile ? ggml_time_us() : 0;
    cleanup();
    if (collect_profile) {
        profile.cleanup_us = static_cast<uint64_t>(ggml_time_us() - cleanup_start_us);
        profile.total_us = static_cast<uint64_t>(ggml_time_us() - total_start_us);
        npu_decode_profile_write(profile);
    }
    return true;
}

bool ggml_backend_npu_decode_attention_w4a16_ex(
        const char * op_name,
        const struct ggml_npu_decode_awq_view * q,
        const struct ggml_npu_decode_awq_view * k,
        const struct ggml_npu_decode_awq_view * v,
        const struct ggml_npu_decode_awq_view * o,
        const void * act_data,
        int act_type,
        int64_t act_nb0,
        int64_t act_nb1,
        const struct ggml_tensor * llama_k_cache,
        const struct ggml_tensor * llama_v_cache,
        int32_t position,
        float kq_scale,
        float rope_freq_base,
        float rope_freq_scale,
        int32_t layer_id,
        void * dst_data,
        int dst_type,
        int64_t dst_nb1) {
    using namespace ggml_npu;

    auto fail = [op_name](const char * reason) {
        if (npu_decode_w4a16_debug_log_enabled()) {
            GGML_LOG_WARN("%s: reject %s: %s\n", __func__, op_name != nullptr ? op_name : "(unnamed)", reason);
        }
        return false;
    };

    if (q == nullptr || k == nullptr || v == nullptr || o == nullptr ||
            act_data == nullptr || dst_data == nullptr || llama_k_cache == nullptr || llama_v_cache == nullptr) {
        return fail("null pointer");
    }
    const bool attention_stream_v2 = npu_decode_stream_api_enabled();
    if (position < 0 ||
            static_cast<uint32_t>(position) >= NPU_DECODE_ATTN_CACHE_CAPACITY_TOKENS ||
            layer_id < 0) {
        return fail("unsupported position or layer");
    }
    if (act_type != GGML_TYPE_F16 && act_type != GGML_TYPE_F32) {
        return fail("activation tensor must be F16 or F32");
    }
    if (dst_type != GGML_TYPE_F16 && dst_type != GGML_TYPE_F32) {
        return fail("destination tensor must be F16 or F32");
    }
    const bool collect_profile = npu_decode_profile_enabled();
    const int64_t total_start_us = collect_profile ? ggml_time_us() : 0;
    int64_t stage_start_us = collect_profile ? total_start_us : 0;
    npu_decode_gemv_profile_record profile = {};
    profile.profile_kind = "aicas_decode_w4a16_attention";
    profile.gemv_mode = "mixed";
    profile.op_role = "attention";
    profile.op_name = op_name != nullptr ? op_name : "";
    profile.dst_type = dst_type;
    profile.k = NPU_DECODE_ATTN_HIDDEN_DIM;
    profile.n_cols = 1;
    profile.out_channels = NPU_DECODE_ATTN_HIDDEN_DIM;
    auto profile_add_extra_time = [&](const char * key, uint64_t value) {
        if (!collect_profile || value == 0) {
            return;
        }
        uint64_t current = 0;
        if (profile.extra_time_us.contains(key)) {
            current = profile.extra_time_us[key].get<uint64_t>();
        }
        profile.extra_time_us[key] = current + value;
    };
    auto profile_inc_extra_call = [&](const char * key) {
        if (!collect_profile) {
            return;
        }
        uint64_t current = 0;
        if (profile.extra_calls.contains(key)) {
            current = profile.extra_calls[key].get<uint64_t>();
        }
        profile.extra_calls[key] = current + 1u;
    };
    if (q->k != NPU_DECODE_ATTN_HIDDEN_DIM || k->k != NPU_DECODE_ATTN_HIDDEN_DIM ||
            v->k != NPU_DECODE_ATTN_HIDDEN_DIM || o->k != NPU_DECODE_ATTN_HIDDEN_DIM ||
            q->out_channels != NPU_DECODE_ATTN_HIDDEN_DIM ||
            k->out_channels != NPU_DECODE_ATTN_KV_DIM ||
            v->out_channels != NPU_DECODE_ATTN_KV_DIM ||
            o->out_channels != NPU_DECODE_ATTN_HIDDEN_DIM ||
            q->packed_k != (q->k + 1) / 2 ||
            k->packed_k != (k->k + 1) / 2 ||
            v->packed_k != (v->k + 1) / 2 ||
            o->packed_k != (o->k + 1) / 2) {
        return fail("unsupported attention dimensions");
    }
    if (collect_profile) {
        profile.validation_us = static_cast<uint64_t>(ggml_time_us() - stage_start_us);
        stage_start_us = ggml_time_us();
    }

    const npu_decode_preloaded_tensor * q_preloaded = nullptr;
    const npu_decode_preloaded_tensor * k_preloaded = nullptr;
    const npu_decode_preloaded_tensor * v_preloaded = nullptr;
    const npu_decode_preloaded_tensor * o_preloaded = nullptr;
    if (!npu_decode_attention_prepare_w4_projection(q, &q_preloaded) ||
            !npu_decode_attention_prepare_w4_projection(k, &k_preloaded) ||
            !npu_decode_attention_prepare_w4_projection(v, &v_preloaded) ||
            !npu_decode_attention_prepare_w4_projection(o, &o_preloaded)) {
        return fail("attention W4 projections are not preloaded");
    }
    if (collect_profile) {
        profile.preload_lookup_us = static_cast<uint64_t>(ggml_time_us() - stage_start_us);
        stage_start_us = ggml_time_us();
    }

    if (!npu_decode_ensure_overlay_active(op_name)) {
        return fail("decode overlay switch failed");
    }
    if (collect_profile) {
        profile.overlay_ensure_us = static_cast<uint64_t>(ggml_time_us() - stage_start_us);
        stage_start_us = ggml_time_us();
    }
    npu_decode_update_runtime_heap_env();
    if (collect_profile) {
        profile.heap_update_us = static_cast<uint64_t>(ggml_time_us() - stage_start_us);
        stage_start_us = ggml_time_us();
    }
    if (npu_decode_init_with_cma_retry(op_name, true) != 0) {
        return fail("decode runtime init failed");
    }
    if (collect_profile) {
        profile.runtime_init_us = static_cast<uint64_t>(ggml_time_us() - stage_start_us);
        stage_start_us = ggml_time_us();
    }
    if (attention_stream_v2) {
        (void) npu_decode_reserve_preloaded_ffn_pingpong_cma_after_init({
            const_cast<npu_decode_preloaded_tensor *>(q_preloaded),
            const_cast<npu_decode_preloaded_tensor *>(k_preloaded),
            const_cast<npu_decode_preloaded_tensor *>(v_preloaded),
            const_cast<npu_decode_preloaded_tensor *>(o_preloaded),
        });
    }

    npu_decode_attention_layer_cache * layer_cache = nullptr;
    {
        npu_decode_attention_cache & cache = npu_decode_attention_get_cache();
        std::lock_guard<std::mutex> lock(cache.mutex);
        if (cache.layers.size() <= static_cast<size_t>(layer_id)) {
            cache.layers.resize(static_cast<size_t>(layer_id) + 1);
        }
        layer_cache = &cache.layers[static_cast<size_t>(layer_id)];
        if (layer_cache->k_cache == nullptr) {
            layer_cache->k_cache = npu_decode_mem_alloc(
                    NPU_DECODE_ATTN_KV_HEADS * NPU_DECODE_ATTN_KV_HEAD_STRIDE_BYTES);
            if (layer_cache->k_cache != nullptr) {
                std::memset(
                        layer_cache->k_cache,
                        0,
                        NPU_DECODE_ATTN_KV_HEADS * NPU_DECODE_ATTN_KV_HEAD_STRIDE_BYTES);
            }
        }
        if (layer_cache->v_cache == nullptr) {
            layer_cache->v_cache = npu_decode_mem_alloc(
                    NPU_DECODE_ATTN_KV_HEADS * NPU_DECODE_ATTN_KV_HEAD_STRIDE_BYTES);
            if (layer_cache->v_cache != nullptr) {
                std::memset(
                        layer_cache->v_cache,
                        0,
                        NPU_DECODE_ATTN_KV_HEADS * NPU_DECODE_ATTN_KV_HEAD_STRIDE_BYTES);
            }
        }
        if (layer_cache->k_cache == nullptr || layer_cache->v_cache == nullptr) {
            return fail("decode attention KV CMA allocation failed");
        }
        const uint32_t cur_position = static_cast<uint32_t>(position);
        const bool position_rewound =
            layer_cache->has_last_position && cur_position <= layer_cache->last_position;
        const bool prefix_rewound = cur_position < layer_cache->converted_len;
        if (position_rewound || prefix_rewound) {
            if (npu_decode_w4a16_debug_log_enabled()) {
                GGML_LOG_INFO(
                        "%s: reset decode attention private KV cache layer=%d position=%u last_position=%u has_last=%d converted_len=%u reason=%s%s\n",
                        __func__,
                        layer_id,
                        cur_position,
                        layer_cache->last_position,
                        layer_cache->has_last_position ? 1 : 0,
                        layer_cache->converted_len,
                        position_rewound ? "position" : "",
                        prefix_rewound ? "+prefix" : "");
            }
            std::memset(layer_cache->k_cache, 0, NPU_DECODE_ATTN_KV_HEADS * NPU_DECODE_ATTN_KV_HEAD_STRIDE_BYTES);
            std::memset(layer_cache->v_cache, 0, NPU_DECODE_ATTN_KV_HEADS * NPU_DECODE_ATTN_KV_HEAD_STRIDE_BYTES);
            layer_cache->converted_len = 0;
            layer_cache->last_position = 0;
            layer_cache->has_last_position = false;
        }
        const uint32_t prefix_tokens = cur_position;
        if (layer_cache->converted_len < prefix_tokens) {
            if (!npu_decode_attention_convert_llama_cache_prefix(
                        llama_k_cache,
                        prefix_tokens,
                        static_cast<uint8_t *>(layer_cache->k_cache)) ||
                    !npu_decode_attention_convert_llama_cache_prefix(
                        llama_v_cache,
                        prefix_tokens,
                        static_cast<uint8_t *>(layer_cache->v_cache),
                        attention_stream_v2)) {
                return fail("failed to convert llama prefill KV cache");
            }
            layer_cache->converted_len = prefix_tokens;
        }
    }
    if (collect_profile) {
        const uint64_t elapsed = static_cast<uint64_t>(ggml_time_us() - stage_start_us);
        profile.activation_preprocess_us += elapsed;
        profile_add_extra_time("kv_cache_prepare", elapsed);
        stage_start_us = ggml_time_us();
    }

    const uint32_t total_tokens = static_cast<uint32_t>(position) + 1u;
    const uint32_t raw_attention_start =
        total_tokens > NPU_DECODE_ATTN_VISIBLE_WINDOW_TOKENS ?
            total_tokens - NPU_DECODE_ATTN_VISIBLE_WINDOW_TOKENS : 0u;
    const uint32_t attention_start =
        raw_attention_start == 0u ? 0u :
            npu_decode_align_up_u32(raw_attention_start, NPU_DECODE_ATTN_W8_COL_TILE_ELEMS);
    const uint32_t seq_len = total_tokens - attention_start;
    if (seq_len == 0 || seq_len > NPU_DECODE_ATTN_VISIBLE_WINDOW_TOKENS) {
        return fail("invalid attention sliding window");
    }
    if (!attention_stream_v2 && attention_start != 0) {
        return fail("sliding window decode attention requires stream API");
    }
    if (attention_start != 0) {
        static bool warned_sliding_window = false;
        if (!warned_sliding_window) {
            warned_sliding_window = true;
            GGML_LOG_WARN(
                    "%s: stream attention sliding window enabled visible_tokens=%u block_align=%u; "
                    "first window start=%u len=%u total_tokens=%u\n",
                    __func__,
                    NPU_DECODE_ATTN_VISIBLE_WINDOW_TOKENS,
                    NPU_DECODE_ATTN_W8_COL_TILE_ELEMS,
                    attention_start,
                    seq_len,
                    total_tokens);
        }
    }
    const uint32_t scale_region = npu_decode_attention_kv_cache_scale_region_bytes();
    const uint32_t kv_dump_head_bytes =
        npu_decode_attention_npu_kv_cache_scale_region_bytes() +
        (NPU_DECODE_ATTN_VISIBLE_WINDOW_TOKENS / NPU_DECODE_ATTN_ROW_TILE_ELEMS) *
            npu_decode_attention_kv_cache_quant_tile_bytes();
    const uint32_t kv_dump_bytes = npu_decode_align_up_u32(
            (NPU_DECODE_ATTN_KV_HEADS - 1u) * NPU_DECODE_ATTN_NPU_KV_HEAD_STRIDE_BYTES +
                kv_dump_head_bytes,
            NPU_DECODE_GEMV_DMA_ALIGN);
    const uint32_t raw_k_mvin_bytes = npu_decode_align_up_u32(
            npu_decode_ceil_div_u32(seq_len, NPU_DECODE_ATTN_ROW_TILE_ELEMS) *
                npu_decode_attention_kv_cache_quant_tile_bytes(),
            NPU_DECODE_GEMV_DMA_ALIGN);
    const uint32_t score_prob_stride_bytes =
        npu_decode_ceil_div_u32(seq_len, NPU_DECODE_GEMV_K_TILE) *
        NPU_DECODE_GEMV_K_TILE * sizeof(ggml_fp16_t);

    const npu_decode_gemv_layout & max_layout = q_preloaded->pingpong_layout;
    const uint32_t act_cma_bytes = std::max<uint32_t>(
            npu_decode_act_payload_bytes(max_layout),
            std::max<uint32_t>(k_preloaded->pingpong_layout.scale_bytes, q_preloaded->pingpong_layout.scale_bytes));
    const uint32_t max_scale_bytes = std::max<uint32_t>(
            std::max<uint32_t>(q_preloaded->pingpong_layout.scale_bytes, k_preloaded->pingpong_layout.scale_bytes),
            std::max<uint32_t>(v_preloaded->pingpong_layout.scale_bytes, o_preloaded->pingpong_layout.scale_bytes));
    const uint32_t max_weight_bytes = std::max<uint32_t>(
            std::max<uint32_t>(q_preloaded->pingpong_layout.weight_bytes, k_preloaded->pingpong_layout.weight_bytes),
            std::max<uint32_t>(v_preloaded->pingpong_layout.weight_bytes, o_preloaded->pingpong_layout.weight_bytes));
    const uint32_t k_score_scale_bytes = npu_decode_align_up_u32(
            npu_decode_ceil_div_u32(seq_len, NPU_DECODE_ATTN_ROW_TILE_ELEMS) * NPU_DECODE_GEMV_LINE_BYTES,
            NPU_DECODE_GEMV_DMA_ALIGN);
    const uint32_t host_output_bytes = npu_decode_host_output_bytes(
            NPU_DECODE_ATTN_HIDDEN_DIM,
            dst_type == GGML_TYPE_F32 && npu_decode_hw_f32_mvout_enabled());
    const bool pre_o_output = npu_decode_attention_pre_o_output_enabled();
    const bool cpu_qkpv = npu_decode_attention_cpu_qkpv_enabled();
    const bool cpu_qk = cpu_qkpv || npu_decode_attention_cpu_qk_enabled();
    const bool cpu_pv = cpu_qkpv || npu_decode_attention_cpu_pv_enabled();
    const bool cpu_pv_fast = cpu_pv && npu_decode_attention_cpu_pv_fast_enabled();
    const bool qk_split_group = npu_decode_attention_qk_split_group_enabled();
    const bool cpu_rope = npu_decode_attention_cpu_rope_enabled();
    const bool hw_kv_quant = !cpu_rope && npu_decode_attention_hw_kv_quant_enabled();
    const int pv_scale_boost_log2 = cpu_pv ? 0 : npu_decode_attention_pv_scale_boost_log2();
    const float pv_scale_boost = std::ldexp(1.0f, pv_scale_boost_log2);
    const float pv_scale_unboost = std::ldexp(1.0f, -pv_scale_boost_log2);
    const bool pv_scale_boost_fold_o =
        pv_scale_boost_log2 > 0 &&
        !pre_o_output &&
        npu_decode_attention_pv_scale_boost_fold_o_enabled();
    const bool rope_input_stats = npu_decode_attention_rope_input_stats_enabled();
    const bool rope_input_stats_only = npu_decode_attention_rope_input_stats_only_enabled();
    const bool compare_qkpv = npu_decode_attention_compare_qkpv_requested(layer_id, position);
    const bool dump_attention = npu_decode_attention_dump_requested(layer_id, position);

    void * act_cma = npu_decode_mem_alloc(std::max<uint32_t>(act_cma_bytes, o_preloaded->pingpong_layout.act_bytes));
    void * scale_cma = npu_decode_mem_alloc(std::max(max_scale_bytes, k_score_scale_bytes));
    void * weight_cma = npu_decode_mem_alloc(max_weight_bytes);
    void * kv_window_cma = npu_decode_mem_alloc(kv_dump_bytes);
    void * out_cma = npu_decode_mem_alloc(host_output_bytes);
    void * rope_lut_cma = nullptr;
    if (act_cma == nullptr || scale_cma == nullptr || weight_cma == nullptr || kv_window_cma == nullptr || out_cma == nullptr) {
        npu_decode_mem_free(act_cma);
        npu_decode_mem_free(scale_cma);
        npu_decode_mem_free(weight_cma);
        npu_decode_mem_free(kv_window_cma);
        npu_decode_mem_free(out_cma);
        return fail("decode attention transient CMA allocation failed");
    }
    if (collect_profile) {
        profile.alloc_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
        stage_start_us = ggml_time_us();
    }
    auto cleanup = [&]() {
        npu_decode_mem_free(act_cma);
        npu_decode_mem_free(scale_cma);
        npu_decode_mem_free(weight_cma);
        npu_decode_mem_free(kv_window_cma);
        npu_decode_mem_free(out_cma);
        npu_decode_mem_free(rope_lut_cma);
    };

    const uint32_t rope_windows =
        ((static_cast<uint32_t>(position) & ~1u) >> 1) + 1u;
    std::vector<uint8_t> rope_lut(
            static_cast<size_t>(rope_windows) * NPU_DECODE_ATTN_ROPE_LUT_WINDOW_BYTES,
            0);
    const uint32_t even_position = static_cast<uint32_t>(position) & ~1u;
    uint8_t * rope_window =
        rope_lut.data() +
        static_cast<size_t>(even_position >> 1) * NPU_DECODE_ATTN_ROPE_LUT_WINDOW_BYTES;
    auto rope_q1_15 = [](double value) {
        const long quant = std::lrint(value * 32767.0);
        return static_cast<int16_t>(std::max<long>(-32768, std::min<long>(32767, quant)));
    };
    for (uint32_t pair = 0; pair < NPU_DECODE_ATTN_HEAD_DIM / 2u; ++pair) {
        const double inv_freq = std::pow(
                static_cast<double>(rope_freq_base),
                -2.0 * static_cast<double>(pair) / static_cast<double>(NPU_DECODE_ATTN_HEAD_DIM));
        for (uint32_t slot = 0; slot < 2; ++slot) {
            const double theta =
                static_cast<double>(even_position + slot) *
                static_cast<double>(rope_freq_scale) *
                inv_freq;
            const uint16_t cos_q15 = static_cast<uint16_t>(rope_q1_15(std::cos(theta)));
            const uint16_t sin_q15 = static_cast<uint16_t>(rope_q1_15(std::sin(theta)));
            npu_decode_attention_store_u16_le(
                    rope_window + slot * 128u + pair * sizeof(uint16_t),
                    cos_q15);
            npu_decode_attention_store_u16_le(
                    rope_window + slot * 128u + NPU_DECODE_GEMV_LINE_BYTES + pair * sizeof(uint16_t),
                    sin_q15);
        }
    }

    npu_decode_reset();
    rope_lut_cma = npu_decode_mem_alloc(rope_lut.size());
    if (rope_lut_cma == nullptr) {
        cleanup();
        return fail("decode attention RoPE LUT CMA allocation failed");
    }
    std::memcpy(rope_lut_cma, rope_lut.data(), rope_lut.size());
    void * rope_lut_window_cma =
        static_cast<uint8_t *>(rope_lut_cma) +
        static_cast<size_t>(even_position >> 1) * NPU_DECODE_ATTN_ROPE_LUT_WINDOW_BYTES;
    if (collect_profile) {
        profile.host_copy_activation_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
        stage_start_us = ggml_time_us();
    }

    if (attention_stream_v2) {
        void * decode_cma_base = npu_decode_memory_base();
        if (decode_cma_base == nullptr ||
                !npu_decode_stream_w4_cma_resident(q_preloaded) ||
                !npu_decode_stream_w4_cma_resident(k_preloaded) ||
                !npu_decode_stream_w4_cma_resident(v_preloaded) ||
                !npu_decode_stream_w4_cma_resident(o_preloaded)) {
            cleanup();
            return fail("attention stream W4 assets are not CMA resident");
        }

        const uint32_t q_cma_bytes = npu_decode_host_output_bytes(NPU_DECODE_ATTN_HIDDEN_DIM, false);
        const uint32_t q_head_stride = NPU_DECODE_GEMV_DMA_ALIGN;
        const uint32_t q_head_cma_bytes = NPU_DECODE_ATTN_Q_HEADS * q_head_stride;
        const uint32_t score_prob_raw_bytes = seq_len * sizeof(ggml_fp16_t);
        const uint32_t qk_compact_cma_bytes = npu_decode_align_up_u32(
                NPU_DECODE_ATTN_GROUP_SIZE * score_prob_raw_bytes,
                NPU_DECODE_GEMV_DMA_ALIGN);
        const uint32_t prob_cma_bytes =
            NPU_DECODE_ATTN_KV_HEADS * NPU_DECODE_ATTN_GROUP_SIZE * score_prob_stride_bytes;
        const uint32_t pv_head_stride = npu_decode_align_up_u32(
                NPU_DECODE_ATTN_GROUP_SIZE * NPU_DECODE_ATTN_HEAD_DIM * sizeof(ggml_fp16_t),
                NPU_DECODE_GEMV_DMA_ALIGN);
        void * q_cma = npu_decode_mem_alloc(q_cma_bytes);
        void * q_head_cma = npu_decode_mem_alloc(q_head_cma_bytes);
        void * qk_compact_cma = npu_decode_mem_alloc(qk_compact_cma_bytes);
        void * prob_cma = npu_decode_mem_alloc(prob_cma_bytes);
        void * cpu_prob_cma = compare_qkpv ? npu_decode_mem_alloc(prob_cma_bytes) : nullptr;
        void * pv_head_cma = npu_decode_mem_alloc(pv_head_stride);
        void * pv_scale_cma =
            (pv_scale_boost_log2 > 0 || attention_start > 0) ?
                npu_decode_mem_alloc(scale_region) : nullptr;
        if (q_cma == nullptr || q_head_cma == nullptr || qk_compact_cma == nullptr ||
                prob_cma == nullptr || (compare_qkpv && cpu_prob_cma == nullptr) ||
                pv_head_cma == nullptr ||
                ((pv_scale_boost_log2 > 0 || attention_start > 0) && pv_scale_cma == nullptr)) {
            npu_decode_mem_free(q_cma);
            npu_decode_mem_free(q_head_cma);
            npu_decode_mem_free(qk_compact_cma);
            npu_decode_mem_free(prob_cma);
            npu_decode_mem_free(cpu_prob_cma);
            npu_decode_mem_free(pv_head_cma);
            npu_decode_mem_free(pv_scale_cma);
            cleanup();
            return fail("attention stream CMA allocation failed");
        }
        if (collect_profile) {
            profile.alloc_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            stage_start_us = ggml_time_us();
            profile.preloaded_host = true;
            profile.preloaded_cma = true;
            profile.pingpong = true;
            profile.activation_bytes =
                q_preloaded->pingpong_layout.act_bytes +
                q_head_cma_bytes +
                prob_cma_bytes +
                (pv_scale_boost_log2 > 0 ? scale_region : 0u);
            profile.scale_bytes =
                q_preloaded->pingpong_layout.scale_bytes +
                k_preloaded->pingpong_layout.scale_bytes +
                v_preloaded->pingpong_layout.scale_bytes +
                o_preloaded->pingpong_layout.scale_bytes;
            profile.weight_bytes =
                q_preloaded->pingpong_layout.weight_bytes +
                k_preloaded->pingpong_layout.weight_bytes +
                v_preloaded->pingpong_layout.weight_bytes +
                o_preloaded->pingpong_layout.weight_bytes;
            profile.output_bytes = host_output_bytes + pv_head_stride;
        }
        std::memset(q_cma, 0, q_cma_bytes);
        std::memset(q_head_cma, 0, q_head_cma_bytes);
        std::memset(qk_compact_cma, 0, qk_compact_cma_bytes);
        std::memset(prob_cma, 0, prob_cma_bytes);
        if (cpu_prob_cma != nullptr) {
            std::memset(cpu_prob_cma, 0, prob_cma_bytes);
        }
        std::memset(pv_head_cma, 0, pv_head_stride);
        if (collect_profile) {
            profile.output_clear_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            stage_start_us = ggml_time_us();
        }

        auto free_stream_attention = [&]() {
            npu_decode_mem_free(q_cma);
            npu_decode_mem_free(q_head_cma);
            npu_decode_mem_free(qk_compact_cma);
            npu_decode_mem_free(prob_cma);
            npu_decode_mem_free(cpu_prob_cma);
            npu_decode_mem_free(pv_head_cma);
            npu_decode_mem_free(pv_scale_cma);
        };
        if (pv_scale_boost_log2 > 0) {
            static bool warned_pv_scale_boost = false;
            if (!warned_pv_scale_boost) {
                warned_pv_scale_boost = true;
                GGML_LOG_WARN(
                        "%s: stream attention NPU PV scale boost enabled log2=%d factor=%g; "
                        "V scales are staged larger for existing FP16-premul hardware; fold_o=%d\n",
                        __func__,
                        pv_scale_boost_log2,
                        static_cast<double>(pv_scale_boost),
                        pv_scale_boost_fold_o ? 1 : 0);
            }
        }
        std::vector<uint8_t> qk_compact_dump;
        if (dump_attention) {
            qk_compact_dump.resize(NPU_DECODE_ATTN_KV_HEADS * qk_compact_cma_bytes);
        }
        auto stream_weight_ptr = [decode_cma_base](const npu_decode_preloaded_tensor * tensor) {
            return npu_decode_stream_cma_ptr(decode_cma_base, tensor->pingpong_weight_cma_offset);
        };
        auto stream_weight_scale_ptr = [decode_cma_base](const npu_decode_preloaded_tensor * tensor) {
            return npu_decode_stream_cma_ptr(decode_cma_base, tensor->pingpong_scale_cma_offset);
        };
        auto stream_act_scale_ptr = [decode_cma_base](const npu_decode_preloaded_tensor * tensor) {
            return npu_decode_stream_cma_ptr(decode_cma_base, tensor->act_scale_cma_offset);
        };
        auto stage_window_token_scales = [&](
                uint8_t * dst,
                uint32_t dst_bytes,
                const uint8_t * head_cache_base,
                uint32_t kv_head,
                bool apply_boost) {
            std::memset(dst, 0, dst_bytes);
            for (uint32_t local_token = 0; local_token < seq_len; ++local_token) {
                const uint32_t global_token = attention_start + local_token;
                const uint16_t scale_bits = npu_decode_attention_load_u16_le(
                        head_cache_base +
                        npu_decode_attention_cache_scale_addr(kv_head, global_token) -
                        kv_head * NPU_DECODE_ATTN_KV_HEAD_STRIDE_BYTES);
                ggml_fp16_t out_bits = scale_bits;
                if (apply_boost) {
                    out_bits = ggml_fp32_to_fp16(
                            std::min(65504.0f, ggml_fp16_to_fp32(scale_bits) * pv_scale_boost));
                }
                npu_decode_attention_store_u16_le(
                        dst + local_token * sizeof(ggml_fp16_t),
                        out_bits);
            }
        };
        auto get_folded_o_act_scale_ptr = [&]() -> const void * {
            if (!pv_scale_boost_fold_o) {
                return nullptr;
            }
            struct folded_o_scale_entry {
                const npu_decode_preloaded_tensor * tensor = nullptr;
                int boost_log2 = 0;
                void * ptr = nullptr;
            };
            static std::mutex mutex;
            static std::vector<folded_o_scale_entry> entries;
            std::lock_guard<std::mutex> lock(mutex);
            for (const folded_o_scale_entry & entry : entries) {
                if (entry.tensor == o_preloaded && entry.boost_log2 == pv_scale_boost_log2) {
                    return entry.ptr;
                }
            }

            const int64_t cache_build_start_us = collect_profile ? ggml_time_us() : 0;
            void * folded_cma = npu_decode_mem_alloc(o_preloaded->pingpong_layout.act_bytes);
            if (folded_cma == nullptr) {
                return nullptr;
            }
            std::memset(folded_cma, 0, o_preloaded->pingpong_layout.act_bytes);
            const uint8_t * src_scale =
                static_cast<const uint8_t *>(stream_act_scale_ptr(o_preloaded));
            uint8_t * dst_scale = static_cast<uint8_t *>(folded_cma);
            for (uint32_t col_tile = 0; col_tile < o_preloaded->pingpong_layout.col_tiles; ++col_tile) {
                for (uint32_t lane = 0; lane < NPU_DECODE_GEMV_K_TILE; ++lane) {
                    const uint32_t idx = col_tile * NPU_DECODE_GEMV_K_TILE + lane;
                    if (idx >= NPU_DECODE_ATTN_HIDDEN_DIM) {
                        continue;
                    }
                    const uint32_t off =
                        col_tile * NPU_DECODE_GEMV_ACT_TILE_BYTES + lane * sizeof(ggml_fp16_t);
                    const uint16_t scale_bits =
                        npu_decode_attention_load_u16_le(src_scale + off);
                    const ggml_fp16_t folded =
                        ggml_fp32_to_fp16(ggml_fp16_to_fp32(scale_bits) * pv_scale_unboost);
                    npu_decode_attention_store_u16_le(dst_scale + off, folded);
                }
            }
            entries.push_back({o_preloaded, pv_scale_boost_log2, folded_cma});
            if (collect_profile) {
                const uint64_t elapsed = static_cast<uint64_t>(ggml_time_us() - cache_build_start_us);
                profile.host_copy_activation_us += elapsed;
                profile_add_extra_time("o_scale_fold_cache_build", elapsed);
            }
            return folded_cma;
        };
        auto run_w4_projection = [&](const npu_decode_preloaded_tensor * tensor,
                                     const char * stats_label,
                                     void * output,
                                     uint16_t rows,
                                     kv260_stream_gemv_role role,
                                     kv260_stream_post_op post_op,
                                     kv260_stream_output_precision output_precision,
                                     uint32_t extra_flags = 0,
                                     const void * act_scale_override_ptr = nullptr) {
            const int64_t projection_start_us = collect_profile ? ggml_time_us() : 0;
            const void * act_scale_ptr =
                act_scale_override_ptr != nullptr ? act_scale_override_ptr : stream_act_scale_ptr(tensor);
            kv260_stream_gemv_desc desc = {};
            desc.act_ptr = act_cma;
            desc.act_scale_ptr = act_scale_ptr;
            desc.weight_payload_ptr = stream_weight_ptr(tensor);
            desc.weight_scale_ptr = stream_weight_scale_ptr(tensor);
            desc.output_ptr = output;
            desc.m = rows;
            desc.n = NPU_DECODE_ATTN_HIDDEN_DIM;
            desc.mode = KV260_STREAM_DECODE_GEMV_W4A16;
            desc.output_precision = output_precision;
            desc.role = role;
            desc.dst = KV260_STREAM_GEMV_DST_OUTPUT;
            desc.post_op = post_op;
            desc.flags = KV260_STREAM_GEMV_F_ENABLE_ACT_SCALE | extra_flags;
            desc.rope_lut_ptr = post_op == KV260_STREAM_POST_ROPE ? rope_lut_window_cma : nullptr;
            desc.elem_count = rows;
            desc.position = static_cast<uint16_t>(position);
            npu_decode_dump_act_scale_stats_packed(
                    stats_label,
                    op_name,
                    layer_id,
                    position,
                    tensor->pingpong_layout,
                    NPU_DECODE_ATTN_HIDDEN_DIM,
                    act_cma,
                    act_scale_ptr);
            const int projection_rc = npu_decode_stream_gemv(op_name, desc);
            if (collect_profile) {
                const uint64_t elapsed = static_cast<uint64_t>(ggml_time_us() - projection_start_us);
                profile.pingpong_us += elapsed;
                profile.pingpong_calls += 1;
                if (stats_label != nullptr) {
                    std::string key = "projection_";
                    key += stats_label;
                    profile_add_extra_time(key.c_str(), elapsed);
                }
            }
            if (projection_rc != 0) {
                GGML_LOG_WARN(
                        "%s: stream W4 projection failed rc=%d rows=%u role=%d post_op=%d output_precision=%d flags=0x%x\n",
                        __func__,
                        projection_rc,
                        static_cast<unsigned>(rows),
                        static_cast<int>(role),
                        static_cast<int>(post_op),
                        static_cast<int>(output_precision),
                        static_cast<unsigned>(desc.flags));
            }
            return projection_rc;
        };

        npu_decode_pack_activation_slice(
                act_cma,
                q_preloaded->pingpong_layout.act_bytes,
                q_preloaded->pingpong_layout,
                act_data,
                act_type,
                act_nb0,
                act_nb1,
                q->k,
                0,
                q->k,
                0);
        if (collect_profile) {
            profile.activation_preprocess_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            stage_start_us = ggml_time_us();
        }

        if (rope_input_stats) {
            std::memset(kv_window_cma, 0, kv_dump_bytes);
            int stats_rc = run_w4_projection(
                    k_preloaded,
                    "attn_k_raw_hidden_scale",
                    kv_window_cma,
                    NPU_DECODE_ATTN_KV_DIM,
                    KV260_STREAM_GEMV_ROLE_LINEAR,
                    KV260_STREAM_POST_BYPASS,
                    KV260_STREAM_OUTPUT_FP16);
            if (stats_rc != 0) {
                free_stream_attention();
                cleanup();
                return fail("attention raw K projection stats failed");
            }
            npu_decode_attention_log_rope_input_stats(
                    op_name,
                    "K",
                    layer_id,
                    position,
                    kv_window_cma,
                    NPU_DECODE_ATTN_KV_HEADS);

            std::memset(q_cma, 0, q_cma_bytes);
            stats_rc = run_w4_projection(
                    q_preloaded,
                    "attn_q_raw_hidden_scale",
                    q_cma,
                    NPU_DECODE_ATTN_HIDDEN_DIM,
                    KV260_STREAM_GEMV_ROLE_LINEAR,
                    KV260_STREAM_POST_BYPASS,
                    KV260_STREAM_OUTPUT_FP16);
            if (stats_rc != 0) {
                free_stream_attention();
                cleanup();
                return fail("attention raw Q projection stats failed");
            }
            npu_decode_attention_log_rope_input_stats(
                    op_name,
                    "Q",
                    layer_id,
                    position,
                    q_cma,
                    NPU_DECODE_ATTN_Q_HEADS);

            if (rope_input_stats_only) {
                if (dst_type == GGML_TYPE_F16) {
                    std::memset(dst_data, 0, NPU_DECODE_ATTN_HIDDEN_DIM * sizeof(ggml_fp16_t));
                } else {
                    std::memset(dst_data, 0, NPU_DECODE_ATTN_HIDDEN_DIM * sizeof(float));
                }
                GGML_UNUSED(kq_scale);
                GGML_UNUSED(dst_nb1);
                free_stream_attention();
                cleanup();
                return true;
            }
        }

        std::memset(kv_window_cma, 0, kv_dump_bytes);
        int rc = run_w4_projection(
                k_preloaded,
                "attn_k_hidden_scale",
                kv_window_cma,
                NPU_DECODE_ATTN_KV_DIM,
                hw_kv_quant ? KV260_STREAM_GEMV_ROLE_KV_PROJ : KV260_STREAM_GEMV_ROLE_LINEAR,
                cpu_rope ? KV260_STREAM_POST_BYPASS : KV260_STREAM_POST_ROPE,
                KV260_STREAM_OUTPUT_FP16,
                hw_kv_quant ? static_cast<uint32_t>(KV260_STREAM_GEMV_F_KV_QUANT) : 0u);
        if (rc != 0) {
            free_stream_attention();
            cleanup();
            return fail("attention stream K projection failed");
        }
        int64_t kv_append_start_us = collect_profile ? ggml_time_us() : 0;
        if (cpu_rope) {
            npu_decode_attention_apply_cpu_rope_fp16(
                    kv_window_cma,
                    NPU_DECODE_ATTN_KV_HEADS,
                    static_cast<uint32_t>(position),
                    rope_freq_base,
                    rope_freq_scale);
        }
        if (!hw_kv_quant) {
            static bool warned_attention_cpu_kv_quant = false;
            if (!warned_attention_cpu_kv_quant) {
                warned_attention_cpu_kv_quant = true;
                GGML_LOG_WARN(
                        "%s: stream attention uses CPU KV quant cache maintenance; "
                        "set AICAS_TEXT_DECODE_ATTN_NPU_HW_KV_QUANT=1 only after runtime KV quant passes\n",
                        __func__);
            }
            npu_decode_attention_append_fp16_projection_to_cache(
                    static_cast<uint8_t *>(layer_cache->k_cache),
                    static_cast<const uint8_t *>(kv_window_cma),
                    static_cast<uint32_t>(position),
                    false);
            if (!npu_decode_attention_write_fp16_projection_to_llama_cache(
                        llama_k_cache,
                        static_cast<const uint8_t *>(kv_window_cma),
                        static_cast<uint32_t>(position))) {
                free_stream_attention();
                cleanup();
                return fail("failed to append stream K projection to llama KV cache");
            }
        } else {
            kv260_stream_kv_scale_result k_scale_result = {};
            rc = npu_decode_stream_kv_scale_read(op_name, false, &k_scale_result);
            npu_decode_attention_kv_scale_regs k_regs;
            if (rc != 0 ||
                    !npu_decode_attention_stream_scale_result_to_regs(k_scale_result, &k_regs)) {
                free_stream_attention();
                cleanup();
                return fail("failed to read stream K quant scale");
            }
            npu_decode_attention_append_stream_quant_projection_to_cache(
                    static_cast<uint8_t *>(layer_cache->k_cache),
                    static_cast<const uint8_t *>(kv_window_cma),
                    static_cast<uint32_t>(position),
                    k_regs,
                    false);
            if (!npu_decode_attention_write_stream_quant_projection_to_llama_cache(
                        llama_k_cache,
                        static_cast<const uint8_t *>(kv_window_cma),
                        static_cast<uint32_t>(position),
                        k_regs)) {
                free_stream_attention();
                cleanup();
                return fail("failed to append stream K quant projection to llama KV cache");
            }
        }
        if (collect_profile) {
            const uint64_t elapsed = static_cast<uint64_t>(ggml_time_us() - kv_append_start_us);
            profile.activation_preprocess_us += elapsed;
            profile_add_extra_time("k_cache_append", elapsed);
        }

        std::memset(kv_window_cma, 0, kv_dump_bytes);
        rc = run_w4_projection(
                v_preloaded,
                "attn_v_hidden_scale",
                kv_window_cma,
                NPU_DECODE_ATTN_KV_DIM,
                hw_kv_quant ? KV260_STREAM_GEMV_ROLE_KV_PROJ : KV260_STREAM_GEMV_ROLE_LINEAR,
                KV260_STREAM_POST_BYPASS,
                KV260_STREAM_OUTPUT_FP16,
                hw_kv_quant ?
                    static_cast<uint32_t>(KV260_STREAM_GEMV_F_KV_QUANT | KV260_STREAM_GEMV_F_KV_IS_V) :
                    0u);
        if (rc != 0) {
            free_stream_attention();
            cleanup();
            return fail("attention stream V projection failed");
        }
        kv_append_start_us = collect_profile ? ggml_time_us() : 0;
        if (hw_kv_quant) {
            kv260_stream_kv_scale_result v_scale_result = {};
            rc = npu_decode_stream_kv_scale_read(op_name, true, &v_scale_result);
            npu_decode_attention_kv_scale_regs v_regs;
            if (rc != 0 ||
                    !npu_decode_attention_stream_scale_result_to_regs(v_scale_result, &v_regs)) {
                free_stream_attention();
                cleanup();
                return fail("failed to read stream V quant scale");
            }
            npu_decode_attention_append_stream_quant_projection_to_cache(
                    static_cast<uint8_t *>(layer_cache->v_cache),
                    static_cast<const uint8_t *>(kv_window_cma),
                    static_cast<uint32_t>(position),
                    v_regs,
                    true);
            if (!npu_decode_attention_write_stream_quant_projection_to_llama_cache(
                        llama_v_cache,
                        static_cast<const uint8_t *>(kv_window_cma),
                        static_cast<uint32_t>(position),
                        v_regs)) {
                free_stream_attention();
                cleanup();
                return fail("failed to append stream V quant projection to llama KV cache");
            }
        } else {
            npu_decode_attention_append_fp16_projection_to_cache(
                    static_cast<uint8_t *>(layer_cache->v_cache),
                    static_cast<const uint8_t *>(kv_window_cma),
                    static_cast<uint32_t>(position),
                    true);
            if (!npu_decode_attention_write_fp16_projection_to_llama_cache(
                        llama_v_cache,
                        static_cast<const uint8_t *>(kv_window_cma),
                        static_cast<uint32_t>(position))) {
                free_stream_attention();
                cleanup();
                return fail("failed to append stream V projection to llama KV cache");
            }
        }
        if (collect_profile) {
            const uint64_t elapsed = static_cast<uint64_t>(ggml_time_us() - kv_append_start_us);
            profile.activation_preprocess_us += elapsed;
            profile_add_extra_time("v_cache_append", elapsed);
        }

        rc = run_w4_projection(
                q_preloaded,
                "attn_q_hidden_scale",
                q_cma,
                NPU_DECODE_ATTN_HIDDEN_DIM,
                KV260_STREAM_GEMV_ROLE_LINEAR,
                cpu_rope ? KV260_STREAM_POST_BYPASS : KV260_STREAM_POST_ROPE,
                KV260_STREAM_OUTPUT_FP16);
        if (rc != 0) {
            free_stream_attention();
            cleanup();
            return fail("attention stream Q projection failed");
        }
        if (cpu_rope) {
            static bool warned_cpu_rope = false;
            if (!warned_cpu_rope) {
                warned_cpu_rope = true;
                GGML_LOG_WARN(
                        "%s: stream attention uses NPU raw Q/K GEMV and CPU RoPE for isolation\n",
                        __func__);
            }
            npu_decode_attention_apply_cpu_rope_fp16(
                    q_cma,
                    NPU_DECODE_ATTN_Q_HEADS,
                    static_cast<uint32_t>(position),
                    rope_freq_base,
                    rope_freq_scale);
        }

        const int64_t q_head_pack_start_us = collect_profile ? ggml_time_us() : 0;
        for (uint32_t head = 0; head < NPU_DECODE_ATTN_Q_HEADS; ++head) {
            std::memcpy(
                    static_cast<uint8_t *>(q_head_cma) + head * q_head_stride,
                    static_cast<uint8_t *>(q_cma) + head * NPU_DECODE_ATTN_HEAD_DIM * sizeof(ggml_fp16_t),
                    NPU_DECODE_ATTN_HEAD_DIM * sizeof(ggml_fp16_t));
        }
        if (collect_profile) {
            const uint64_t elapsed = static_cast<uint64_t>(ggml_time_us() - q_head_pack_start_us);
            profile.host_copy_activation_us += elapsed;
            profile_add_extra_time("q_head_pack", elapsed);
        }

        if (cpu_qk || cpu_pv) {
            static bool warned_cpu_qk_pv_split = false;
            if (!warned_cpu_qk_pv_split) {
                warned_cpu_qk_pv_split = true;
                GGML_LOG_WARN(
                        "%s: stream attention isolation mode cpu_qk=%d cpu_pv=%d cpu_pv_fast=%d; Q/K/V/O projections stay on NPU\n",
                        __func__,
                        cpu_qk ? 1 : 0,
                        cpu_pv ? 1 : 0,
                        cpu_pv_fast ? 1 : 0);
            }
        }

        auto compute_cpu_qk_to_prob = [&](uint32_t kv_head, uint8_t * prob_group_base) {
            std::vector<float> scores(seq_len);
            for (uint32_t group = 0; group < NPU_DECODE_ATTN_GROUP_SIZE; ++group) {
                const uint32_t q_head = kv_head * NPU_DECODE_ATTN_GROUP_SIZE + group;
                const uint8_t * q_base =
                    static_cast<const uint8_t *>(q_head_cma) + q_head * q_head_stride;
                float max_score = -std::numeric_limits<float>::infinity();
                for (uint32_t token = 0; token < seq_len; ++token) {
                    const uint32_t global_token = attention_start + token;
                    const uint16_t k_scale_bits = npu_decode_attention_load_u16_le(
                            static_cast<const uint8_t *>(layer_cache->k_cache) +
                            npu_decode_attention_cache_scale_addr(kv_head, global_token));
                    const float k_scale = ggml_fp16_to_fp32(k_scale_bits);
                    float dot = 0.0f;
                    for (uint32_t dim = 0; dim < NPU_DECODE_ATTN_HEAD_DIM; ++dim) {
                        ggml_fp16_t q_bits = 0;
                        std::memcpy(&q_bits, q_base + dim * sizeof(ggml_fp16_t), sizeof(q_bits));
                        const float q_value = ggml_fp16_to_fp32(q_bits);
                        const int8_t k_value = static_cast<int8_t>(
                                *(static_cast<const uint8_t *>(layer_cache->k_cache) +
                                  npu_decode_attention_cache_quant_addr(kv_head, global_token, dim)));
                        dot += q_value * static_cast<float>(k_value) * k_scale;
                    }
                    const float scaled = dot * kq_scale;
                    scores[token] = scaled;
                    max_score = std::max(max_score, scaled);
                }

                float denom = 0.0f;
                for (uint32_t token = 0; token < seq_len; ++token) {
                    const float p = std::exp(scores[token] - max_score);
                    scores[token] = p;
                    denom += p;
                }
                const float inv_denom = denom != 0.0f ? 1.0f / denom : 0.0f;
                uint8_t * prob_base = prob_group_base + group * score_prob_stride_bytes;
                std::memset(prob_base, 0, score_prob_stride_bytes);
                for (uint32_t token = 0; token < seq_len; ++token) {
                    const ggml_fp16_t prob_bits = ggml_fp32_to_fp16(scores[token] * inv_denom);
                    std::memcpy(prob_base + token * sizeof(ggml_fp16_t), &prob_bits, sizeof(prob_bits));
                }
            }
        };

        auto compute_cpu_pv_from_prob = [&](uint32_t kv_head, const uint8_t * prob_group_base, ggml_fp16_t * dst_heads) {
            for (uint32_t group = 0; group < NPU_DECODE_ATTN_GROUP_SIZE; ++group) {
                const uint32_t q_head = kv_head * NPU_DECODE_ATTN_GROUP_SIZE + group;
                const uint8_t * prob_base = prob_group_base + group * score_prob_stride_bytes;
                ggml_fp16_t * dst_head =
                    dst_heads +
                    q_head * NPU_DECODE_ATTN_HEAD_DIM;
                for (uint32_t dim = 0; dim < NPU_DECODE_ATTN_HEAD_DIM; ++dim) {
                    float sum = 0.0f;
                    for (uint32_t token = 0; token < seq_len; ++token) {
                        const uint32_t global_token = attention_start + token;
                        ggml_fp16_t prob_bits = 0;
                        std::memcpy(&prob_bits, prob_base + token * sizeof(ggml_fp16_t), sizeof(prob_bits));
                        const float prob = ggml_fp16_to_fp32(prob_bits);
                        const uint16_t v_scale_bits = npu_decode_attention_load_u16_le(
                                static_cast<const uint8_t *>(layer_cache->v_cache) +
                                npu_decode_attention_cache_scale_addr(kv_head, global_token));
                        const float v_scale = ggml_fp16_to_fp32(v_scale_bits);
                        const int8_t v_value = static_cast<int8_t>(
                                *(static_cast<const uint8_t *>(layer_cache->v_cache) +
                                  npu_decode_attention_v_cache_quant_addr(kv_head, global_token, dim)));
                        sum += prob * static_cast<float>(v_value) * v_scale;
                    }
                    dst_head[dim] = ggml_fp32_to_fp16(sum);
                }
            }
        };

        auto compute_cpu_pv_from_prob_fast = [&](uint32_t kv_head, const uint8_t * prob_group_base, ggml_fp16_t * dst_heads) {
            std::vector<float> scaled_prob(
                    static_cast<size_t>(NPU_DECODE_ATTN_GROUP_SIZE) * seq_len);
            const uint8_t * v_head_base =
                static_cast<const uint8_t *>(layer_cache->v_cache) +
                kv_head * NPU_DECODE_ATTN_KV_HEAD_STRIDE_BYTES;
            for (uint32_t token = 0; token < seq_len; ++token) {
                const uint32_t global_token = attention_start + token;
                const uint16_t v_scale_bits =
                    npu_decode_attention_load_u16_le(v_head_base + global_token * sizeof(ggml_fp16_t));
                const float v_scale = ggml_fp16_to_fp32(v_scale_bits);
                for (uint32_t group = 0; group < NPU_DECODE_ATTN_GROUP_SIZE; ++group) {
                    const uint8_t * prob_base =
                        prob_group_base + group * score_prob_stride_bytes;
                    ggml_fp16_t prob_bits = 0;
                    std::memcpy(&prob_bits, prob_base + token * sizeof(ggml_fp16_t), sizeof(prob_bits));
                    scaled_prob[static_cast<size_t>(group) * seq_len + token] =
                        ggml_fp16_to_fp32(prob_bits) * v_scale;
                }
            }

            const uint8_t * v_payload_base =
                v_head_base + npu_decode_attention_kv_cache_scale_region_bytes();
            const uint32_t window_col_tile_start =
                attention_start / NPU_DECODE_ATTN_W8_COL_TILE_ELEMS;
            for (uint32_t group = 0; group < NPU_DECODE_ATTN_GROUP_SIZE; ++group) {
                const uint32_t q_head = kv_head * NPU_DECODE_ATTN_GROUP_SIZE + group;
                const float * group_scaled_prob =
                    scaled_prob.data() + static_cast<size_t>(group) * seq_len;
                ggml_fp16_t * dst_head =
                    dst_heads +
                    q_head * NPU_DECODE_ATTN_HEAD_DIM;
                for (uint32_t dim = 0; dim < NPU_DECODE_ATTN_HEAD_DIM; ++dim) {
                    const uint32_t row_tile = dim / NPU_DECODE_GEMV_M_TILE;
                    const uint32_t row_lane = dim % NPU_DECODE_GEMV_M_TILE;
                    const uint8_t * row_base =
                        v_payload_base +
                        row_tile * NPU_DECODE_ATTN_V_ROW_TILE_STRIDE_BYTES +
                        window_col_tile_start * NPU_DECODE_GEMV_W8_WEIGHT_TILE_BYTES +
                        row_lane * NPU_DECODE_ATTN_W8_COL_TILE_ELEMS;
                    float sum = 0.0f;
                    uint32_t token = 0;
                    for (; token + 3u < seq_len; token += 4u) {
                        const uint32_t col_tile = token / NPU_DECODE_ATTN_W8_COL_TILE_ELEMS;
                        const uint32_t col_lane = token % NPU_DECODE_ATTN_W8_COL_TILE_ELEMS;
                        const uint8_t * v_ptr =
                            row_base +
                            col_tile * NPU_DECODE_GEMV_W8_WEIGHT_TILE_BYTES +
                            col_lane;
                        sum += group_scaled_prob[token + 0u] * static_cast<float>(static_cast<int8_t>(v_ptr[0]));
                        sum += group_scaled_prob[token + 1u] * static_cast<float>(static_cast<int8_t>(v_ptr[1]));
                        sum += group_scaled_prob[token + 2u] * static_cast<float>(static_cast<int8_t>(v_ptr[2]));
                        sum += group_scaled_prob[token + 3u] * static_cast<float>(static_cast<int8_t>(v_ptr[3]));
                    }
                    for (; token < seq_len; ++token) {
                        const uint32_t col_tile = token / NPU_DECODE_ATTN_W8_COL_TILE_ELEMS;
                        const uint32_t col_lane = token % NPU_DECODE_ATTN_W8_COL_TILE_ELEMS;
                        const uint8_t * v_ptr =
                            row_base +
                            col_tile * NPU_DECODE_GEMV_W8_WEIGHT_TILE_BYTES +
                            col_lane;
                        sum += group_scaled_prob[token] * static_cast<float>(static_cast<int8_t>(*v_ptr));
                    }
                    dst_head[dim] = ggml_fp32_to_fp16(sum);
                }
            }
        };

        auto compare_fp16_buffers = [&](
                const char * label,
                uint32_t kv_head,
                const uint8_t * got,
                const uint8_t * expected,
                uint32_t elems) {
            float max_abs = 0.0f;
            double sum_abs = 0.0;
            uint32_t mismatches = 0;
            for (uint32_t i = 0; i < elems; ++i) {
                ggml_fp16_t got_bits = 0;
                ggml_fp16_t exp_bits = 0;
                std::memcpy(&got_bits, got + i * sizeof(ggml_fp16_t), sizeof(got_bits));
                std::memcpy(&exp_bits, expected + i * sizeof(ggml_fp16_t), sizeof(exp_bits));
                const float got_f = ggml_fp16_to_fp32(got_bits);
                const float exp_f = ggml_fp16_to_fp32(exp_bits);
                const float abs_err = std::fabs(got_f - exp_f);
                max_abs = std::max(max_abs, abs_err);
                sum_abs += abs_err;
                const float tol = 0.015f + 0.08f * std::fabs(exp_f);
                if (abs_err > tol) {
                    if (mismatches < 6) {
                        GGML_LOG_WARN(
                                "%s: compare_qkpv %s layer=%d pos=%d kv_head=%u idx=%u expected=%g got=%g abs_err=%g tol=%g\n",
                                op_name,
                                label,
                                layer_id,
                                position,
                                kv_head,
                                i,
                                static_cast<double>(exp_f),
                                static_cast<double>(got_f),
                                static_cast<double>(abs_err),
                                static_cast<double>(tol));
                    }
                    ++mismatches;
                }
            }
            GGML_LOG_WARN(
                    "%s: compare_qkpv %s layer=%d pos=%d kv_head=%u elems=%u mismatches=%u max_abs=%g mean_abs=%g\n",
                    op_name,
                    label,
                    layer_id,
                    position,
                    kv_head,
                    elems,
                    mismatches,
                    static_cast<double>(max_abs),
                    elems != 0 ? sum_abs / static_cast<double>(elems) : 0.0);
        };

        auto compare_qk_prob_valid_tokens = [&](
                uint32_t kv_head,
                const uint8_t * got_group_base,
                const uint8_t * expected_group_base) {
            std::vector<ggml_fp16_t> got_compact(
                    NPU_DECODE_ATTN_GROUP_SIZE * seq_len);
            std::vector<ggml_fp16_t> expected_compact(
                    NPU_DECODE_ATTN_GROUP_SIZE * seq_len);
            for (uint32_t group = 0; group < NPU_DECODE_ATTN_GROUP_SIZE; ++group) {
                std::memcpy(
                        got_compact.data() + group * seq_len,
                        got_group_base + group * score_prob_stride_bytes,
                        seq_len * sizeof(ggml_fp16_t));
                std::memcpy(
                        expected_compact.data() + group * seq_len,
                        expected_group_base + group * score_prob_stride_bytes,
                        seq_len * sizeof(ggml_fp16_t));
            }
            compare_fp16_buffers(
                    "qk_prob_valid",
                    kv_head,
                    reinterpret_cast<const uint8_t *>(got_compact.data()),
                    reinterpret_cast<const uint8_t *>(expected_compact.data()),
                    NPU_DECODE_ATTN_GROUP_SIZE * seq_len);
        };

        for (uint32_t kv_head = 0; kv_head < NPU_DECODE_ATTN_KV_HEADS; ++kv_head) {
            uint8_t * k_head_base =
                static_cast<uint8_t *>(layer_cache->k_cache) +
                kv_head * NPU_DECODE_ATTN_KV_HEAD_STRIDE_BYTES;
            uint8_t * prob_group_base =
                static_cast<uint8_t *>(prob_cma) +
                kv_head * NPU_DECODE_ATTN_GROUP_SIZE * score_prob_stride_bytes;
            std::memset(
                    prob_group_base,
                    0,
                    NPU_DECODE_ATTN_GROUP_SIZE * score_prob_stride_bytes);
            const int64_t qk_start_us = collect_profile ? ggml_time_us() : 0;
            if (cpu_qk) {
                compute_cpu_qk_to_prob(kv_head, prob_group_base);
            } else {
                const uint8_t * direct_qk_scale_ptr =
                    k_head_base + attention_start * sizeof(ggml_fp16_t);
                const bool direct_qk_scale =
                    (reinterpret_cast<uintptr_t>(direct_qk_scale_ptr) &
                     (NPU_DECODE_GEMV_DMA_ALIGN - 1u)) == 0;
                const void * qk_scale_ptr = nullptr;
                if (direct_qk_scale) {
                    qk_scale_ptr = direct_qk_scale_ptr;
                    if (collect_profile) {
                        profile_inc_extra_call("qk_scale_direct");
                    }
                } else {
                    const int64_t qk_scale_stage_start_us = collect_profile ? ggml_time_us() : 0;
                    stage_window_token_scales(
                            static_cast<uint8_t *>(scale_cma),
                            k_score_scale_bytes,
                            k_head_base,
                            kv_head,
                            false);
                    qk_scale_ptr = scale_cma;
                    if (collect_profile) {
                        const uint64_t elapsed = static_cast<uint64_t>(ggml_time_us() - qk_scale_stage_start_us);
                        profile.host_copy_activation_us += elapsed;
                        profile_add_extra_time("qk_scale_stage", elapsed);
                    }
                }
                uint8_t * k_quant_base =
                    k_head_base +
                    scale_region +
                    (attention_start / NPU_DECODE_ATTN_ROW_TILE_ELEMS) *
                        npu_decode_attention_kv_cache_quant_tile_bytes();
                auto make_qk_desc = [&]() {
                    kv260_stream_gemv_desc desc = {};
                    desc.weight_payload_ptr = k_quant_base;
                    desc.weight_scale_ptr = qk_scale_ptr;
                    desc.m = static_cast<uint16_t>(seq_len);
                    desc.n = NPU_DECODE_ATTN_HEAD_DIM;
                    desc.mode = KV260_STREAM_DECODE_GEMV_W8A16;
                    desc.output_precision = KV260_STREAM_OUTPUT_FP16;
                    desc.role = KV260_STREAM_GEMV_ROLE_QK;
                    desc.dst = KV260_STREAM_GEMV_DST_OUTPUT;
                    desc.post_op = KV260_STREAM_POST_SOFTMAX;
                    desc.elem_count = static_cast<uint16_t>(seq_len);
                    desc.position = static_cast<uint16_t>(position);
                    return desc;
                };
                if (qk_split_group) {
                    for (uint32_t group = 0; group < NPU_DECODE_ATTN_GROUP_SIZE; ++group) {
                        kv260_stream_gemv_desc qk_desc = make_qk_desc();
                        qk_desc.act_ptr =
                            static_cast<uint8_t *>(q_head_cma) +
                            (kv_head * NPU_DECODE_ATTN_GROUP_SIZE + group) * q_head_stride;
                        qk_desc.output_ptr = prob_group_base + group * score_prob_stride_bytes;
                        qk_desc.group_count = 1;
                        qk_desc.act_group_stride_bytes = 0;
                        const int64_t qk_runtime_start_us = collect_profile ? ggml_time_us() : 0;
                        rc = npu_decode_stream_gemv(op_name, qk_desc);
                        if (collect_profile) {
                            const uint64_t elapsed = static_cast<uint64_t>(ggml_time_us() - qk_runtime_start_us);
                            profile_add_extra_time("qk_runtime_stream", elapsed);
                            profile_inc_extra_call("qk_split_group");
                        }
                        if (rc != 0) {
                            free_stream_attention();
                            cleanup();
                            return fail("attention stream QK split-group failed");
                        }
                    }
                } else {
                    kv260_stream_gemv_desc qk_desc = make_qk_desc();
                    qk_desc.act_ptr =
                        static_cast<uint8_t *>(q_head_cma) +
                        kv_head * NPU_DECODE_ATTN_GROUP_SIZE * q_head_stride;
                    qk_desc.output_ptr = qk_compact_cma;
                    qk_desc.group_count = NPU_DECODE_ATTN_GROUP_SIZE;
                    qk_desc.act_group_stride_bytes = q_head_stride;
                    const int64_t qk_clear_start_us = collect_profile ? ggml_time_us() : 0;
                    std::memset(qk_compact_cma, 0, qk_compact_cma_bytes);
                    if (collect_profile) {
                        const uint64_t elapsed = static_cast<uint64_t>(ggml_time_us() - qk_clear_start_us);
                        profile.output_clear_us += elapsed;
                        profile_add_extra_time("qk_compact_clear", elapsed);
                    }
                    const int64_t qk_runtime_start_us = collect_profile ? ggml_time_us() : 0;
                    rc = npu_decode_stream_gemv(op_name, qk_desc);
                    if (collect_profile) {
                        const uint64_t elapsed = static_cast<uint64_t>(ggml_time_us() - qk_runtime_start_us);
                        profile_add_extra_time("qk_runtime_stream", elapsed);
                    }
                    if (rc != 0) {
                        free_stream_attention();
                        cleanup();
                        return fail("attention stream QK failed");
                    }
                    const int64_t qk_prob_copy_start_us = collect_profile ? ggml_time_us() : 0;
                    for (uint32_t group = 0; group < NPU_DECODE_ATTN_GROUP_SIZE; ++group) {
                        std::memcpy(
                                prob_group_base + group * score_prob_stride_bytes,
                                static_cast<const uint8_t *>(qk_compact_cma) + group * score_prob_raw_bytes,
                                score_prob_raw_bytes);
                    }
                    if (collect_profile) {
                        const uint64_t elapsed = static_cast<uint64_t>(ggml_time_us() - qk_prob_copy_start_us);
                        profile.host_copy_activation_us += elapsed;
                        profile_add_extra_time("qk_prob_pad_copy", elapsed);
                    }
                    if (!qk_compact_dump.empty()) {
                        std::memcpy(
                                qk_compact_dump.data() + kv_head * qk_compact_cma_bytes,
                                qk_compact_cma,
                                qk_compact_cma_bytes);
                    }
                }
            }
            if (collect_profile) {
                const uint64_t elapsed = static_cast<uint64_t>(ggml_time_us() - qk_start_us);
                profile.gemv_us += elapsed;
                profile.gemv_calls += 1;
                profile_add_extra_time(cpu_qk ? "qk_cpu" : "qk_stream", elapsed);
                profile_inc_extra_call(cpu_qk ? "qk_cpu" : "qk_stream");
            }
            if (compare_qkpv && !cpu_qk && cpu_prob_cma != nullptr) {
                uint8_t * cpu_prob_group_base =
                    static_cast<uint8_t *>(cpu_prob_cma) +
                    kv_head * NPU_DECODE_ATTN_GROUP_SIZE * score_prob_stride_bytes;
                compute_cpu_qk_to_prob(kv_head, cpu_prob_group_base);
                compare_qk_prob_valid_tokens(
                        kv_head,
                        prob_group_base,
                        cpu_prob_group_base);
                compare_fp16_buffers(
                        "qk_prob_padded",
                        kv_head,
                        prob_group_base,
                        cpu_prob_group_base,
                        (NPU_DECODE_ATTN_GROUP_SIZE * score_prob_stride_bytes) / sizeof(ggml_fp16_t));
            }

            if (cpu_pv) {
                const int64_t pv_cpu_start_us = collect_profile ? ggml_time_us() : 0;
                if (cpu_pv_fast) {
                    compute_cpu_pv_from_prob_fast(kv_head, prob_group_base, static_cast<ggml_fp16_t *>(act_cma));
                } else {
                    compute_cpu_pv_from_prob(kv_head, prob_group_base, static_cast<ggml_fp16_t *>(act_cma));
                }
                if (collect_profile) {
                    const uint64_t elapsed = static_cast<uint64_t>(ggml_time_us() - pv_cpu_start_us);
                    profile.mvout_us += elapsed;
                    profile.mvout_calls += 1;
                    profile_add_extra_time(cpu_pv_fast ? "pv_cpu_fast" : "pv_cpu", elapsed);
                    profile_inc_extra_call(cpu_pv_fast ? "pv_cpu_fast" : "pv_cpu");
                }
            } else {
                uint8_t * v_head_base =
                    static_cast<uint8_t *>(layer_cache->v_cache) +
                    kv_head * NPU_DECODE_ATTN_KV_HEAD_STRIDE_BYTES;
                const void * pv_scale_ptr = v_head_base;
                if (pv_scale_boost_log2 > 0 || attention_start > 0) {
                    const int64_t pv_scale_stage_start_us = collect_profile ? ggml_time_us() : 0;
                    stage_window_token_scales(
                            static_cast<uint8_t *>(pv_scale_cma),
                            scale_region,
                            v_head_base,
                            kv_head,
                            pv_scale_boost_log2 > 0);
                    pv_scale_ptr = pv_scale_cma;
                    if (collect_profile) {
                        const uint64_t elapsed = static_cast<uint64_t>(ggml_time_us() - pv_scale_stage_start_us);
                        profile.host_copy_activation_us += elapsed;
                        profile_add_extra_time("pv_scale_stage", elapsed);
                    }
                }
                kv260_stream_gemv_desc pv_desc = {};
                pv_desc.act_ptr = prob_group_base;
                pv_desc.act_scale_ptr = pv_scale_ptr;
                pv_desc.weight_payload_ptr =
                    v_head_base +
                    scale_region +
                    (attention_start / NPU_DECODE_ATTN_W8_COL_TILE_ELEMS) *
                        NPU_DECODE_GEMV_W8_WEIGHT_TILE_BYTES;
                pv_desc.output_ptr = pv_head_cma;
                pv_desc.weight_row_tile_stride_bytes = NPU_DECODE_ATTN_V_ROW_TILE_STRIDE_BYTES;
                pv_desc.weight_capacity_tokens =
                    static_cast<uint16_t>(NPU_DECODE_ATTN_CACHE_CAPACITY_TOKENS - attention_start);
                pv_desc.m = NPU_DECODE_ATTN_HEAD_DIM;
                pv_desc.n = static_cast<uint16_t>(seq_len);
                pv_desc.mode = KV260_STREAM_DECODE_GEMV_W8A16;
                pv_desc.output_precision = KV260_STREAM_OUTPUT_FP16;
                pv_desc.role = KV260_STREAM_GEMV_ROLE_PV;
                pv_desc.dst = KV260_STREAM_GEMV_DST_OUTPUT;
                pv_desc.post_op = KV260_STREAM_POST_BYPASS;
                pv_desc.flags =
                    KV260_STREAM_GEMV_F_ENABLE_ACT_SCALE |
                    KV260_STREAM_GEMV_F_KV_COL_SCALE |
                    KV260_STREAM_GEMV_F_UNIT_WEIGHT_SCALE;
                pv_desc.elem_count = NPU_DECODE_ATTN_HEAD_DIM;
                pv_desc.position = static_cast<uint16_t>(position);
                pv_desc.group_count = NPU_DECODE_ATTN_GROUP_SIZE;
                pv_desc.act_group_stride_bytes = score_prob_stride_bytes;
                npu_decode_dump_grouped_act_scale_stats(
                        "attn_pv_prob_vscale",
                        op_name,
                        layer_id,
                        position,
                        prob_group_base,
                        NPU_DECODE_ATTN_GROUP_SIZE,
                        score_prob_stride_bytes,
                        static_cast<const uint8_t *>(pv_scale_ptr),
                        seq_len);
                const int64_t pv_stream_start_us = collect_profile ? ggml_time_us() : 0;
                rc = npu_decode_stream_gemv(op_name, pv_desc);
                if (collect_profile) {
                    const uint64_t elapsed = static_cast<uint64_t>(ggml_time_us() - pv_stream_start_us);
                    profile.mvout_us += elapsed;
                    profile.mvout_calls += 1;
                    profile_add_extra_time("pv_stream", elapsed);
                    profile_inc_extra_call("pv_stream");
                }
                if (rc != 0) {
                    free_stream_attention();
                    cleanup();
                    return fail("attention stream PV failed");
                }

                const int64_t pv_copy_start_us = collect_profile ? ggml_time_us() : 0;
                for (uint32_t group = 0; group < NPU_DECODE_ATTN_GROUP_SIZE; ++group) {
                    ggml_fp16_t * dst_head =
                        static_cast<ggml_fp16_t *>(act_cma) +
                        (kv_head * NPU_DECODE_ATTN_GROUP_SIZE + group) * NPU_DECODE_ATTN_HEAD_DIM;
                    const ggml_fp16_t * src_head =
                        reinterpret_cast<const ggml_fp16_t *>(
                                static_cast<const uint8_t *>(pv_head_cma) +
                                group * NPU_DECODE_ATTN_HEAD_DIM * sizeof(ggml_fp16_t));
                    if (pv_scale_boost_log2 > 0 && !pv_scale_boost_fold_o) {
                        for (uint32_t dim = 0; dim < NPU_DECODE_ATTN_HEAD_DIM; ++dim) {
                            dst_head[dim] = ggml_fp32_to_fp16(
                                    ggml_fp16_to_fp32(src_head[dim]) * pv_scale_unboost);
                        }
                    } else {
                        std::memcpy(
                                dst_head,
                                src_head,
                                NPU_DECODE_ATTN_HEAD_DIM * sizeof(ggml_fp16_t));
                    }
                }
                if (collect_profile) {
                    const uint64_t elapsed = static_cast<uint64_t>(ggml_time_us() - pv_copy_start_us);
                    profile.postprocess_us += elapsed;
                    profile_add_extra_time(
                            pv_scale_boost_log2 > 0 && !pv_scale_boost_fold_o ? "pv_unboost_copy" : "pv_copy",
                            elapsed);
                }
                if (compare_qkpv) {
                    std::vector<ggml_fp16_t> cpu_pv_ref(NPU_DECODE_ATTN_HIDDEN_DIM);
                    compute_cpu_pv_from_prob(kv_head, prob_group_base, cpu_pv_ref.data());
                    compare_fp16_buffers(
                            "pv_out",
                            kv_head,
                            reinterpret_cast<const uint8_t *>(
                                    static_cast<ggml_fp16_t *>(act_cma) +
                                    kv_head * NPU_DECODE_ATTN_GROUP_SIZE * NPU_DECODE_ATTN_HEAD_DIM),
                            reinterpret_cast<const uint8_t *>(
                                    cpu_pv_ref.data() +
                                    kv_head * NPU_DECODE_ATTN_GROUP_SIZE * NPU_DECODE_ATTN_HEAD_DIM),
                            NPU_DECODE_ATTN_GROUP_SIZE * NPU_DECODE_ATTN_HEAD_DIM);
                }
            }
        }

        if (dump_attention) {
            uint32_t dump_flags = 0;
            dump_flags |= cpu_qk ? 1u : 0u;
            dump_flags |= cpu_pv ? 2u : 0u;
            dump_flags |= cpu_rope ? 4u : 0u;
            dump_flags |= compare_qkpv ? 8u : 0u;
            npu_decode_attention_write_stream_qkpv_dump(
                    op_name,
                    layer_id,
                    position,
                    seq_len,
                    q_head_cma,
                    q_head_cma_bytes,
                    layer_cache->k_cache,
                    layer_cache->v_cache,
                    qk_compact_dump.empty() ? nullptr : qk_compact_dump.data(),
                    qk_compact_cma_bytes,
                    static_cast<uint32_t>(qk_compact_dump.size()),
                    prob_cma,
                    prob_cma_bytes,
                    act_cma,
                    NPU_DECODE_ATTN_HIDDEN_DIM * sizeof(ggml_fp16_t),
                    score_prob_raw_bytes,
                    score_prob_stride_bytes,
                    dump_flags);
        }

        if (pre_o_output) {
            const int64_t pre_o_copy_start_us = collect_profile ? ggml_time_us() : 0;
            if (dst_type == GGML_TYPE_F16) {
                std::memcpy(dst_data, act_cma, NPU_DECODE_ATTN_HIDDEN_DIM * sizeof(ggml_fp16_t));
            } else {
                npu_decode_convert_output_fp16_to_f32(
                        static_cast<const ggml_fp16_t *>(act_cma),
                        reinterpret_cast<float *>(dst_data),
                        NPU_DECODE_ATTN_HIDDEN_DIM);
            }
            if (collect_profile) {
                const uint64_t elapsed = static_cast<uint64_t>(ggml_time_us() - pre_o_copy_start_us);
                profile.postprocess_us += elapsed;
                profile_add_extra_time("pre_o_output_copy", elapsed);
            }
            GGML_UNUSED(kq_scale);
            GGML_UNUSED(dst_nb1);
            const int64_t cleanup_start_us = collect_profile ? ggml_time_us() : 0;
            free_stream_attention();
            cleanup();
            if (collect_profile) {
                profile.cleanup_us = static_cast<uint64_t>(ggml_time_us() - cleanup_start_us);
                profile.total_us = static_cast<uint64_t>(ggml_time_us() - total_start_us);
                profile.setup_us =
                    profile.validation_us +
                    profile.preload_lookup_us +
                    profile.overlay_ensure_us +
                    profile.heap_update_us +
                    profile.runtime_init_us +
                    profile.memory_base_us +
                    profile.alloc_us;
                npu_decode_profile_write(profile);
            }
            npu_decode_attention_cache & cache = npu_decode_attention_get_cache();
            std::lock_guard<std::mutex> lock(cache.mutex);
            if (cache.layers.size() > static_cast<size_t>(layer_id)) {
                npu_decode_attention_layer_cache & done_layer = cache.layers[static_cast<size_t>(layer_id)];
                done_layer.converted_len =
                    std::max<uint32_t>(
                            done_layer.converted_len,
                            static_cast<uint32_t>(position) + 1u);
                done_layer.last_position = static_cast<uint32_t>(position);
                done_layer.has_last_position = true;
            }
            return true;
        }

        const int64_t o_input_pack_start_us = collect_profile ? ggml_time_us() : 0;
        const uint32_t pv_bytes = NPU_DECODE_ATTN_HIDDEN_DIM * sizeof(ggml_fp16_t);
        if (o_preloaded->pingpong_layout.act_bytes > pv_bytes) {
            std::memset(
                    static_cast<uint8_t *>(act_cma) + pv_bytes,
                    0,
                    o_preloaded->pingpong_layout.act_bytes - pv_bytes);
        }
        if (collect_profile) {
            const uint64_t elapsed = static_cast<uint64_t>(ggml_time_us() - o_input_pack_start_us);
            profile.postprocess_us += elapsed;
            profile_add_extra_time("o_input_pack", elapsed);
        }

        const void * o_act_scale_ptr = nullptr;
        if (pv_scale_boost_fold_o) {
            o_act_scale_ptr = get_folded_o_act_scale_ptr();
            if (o_act_scale_ptr == nullptr) {
                const int64_t o_scale_stage_start_us = collect_profile ? ggml_time_us() : 0;
                std::memset(scale_cma, 0, o_preloaded->pingpong_layout.act_bytes);
                const uint8_t * src_scale =
                    static_cast<const uint8_t *>(stream_act_scale_ptr(o_preloaded));
                uint8_t * dst_scale = static_cast<uint8_t *>(scale_cma);
                for (uint32_t col_tile = 0; col_tile < o_preloaded->pingpong_layout.col_tiles; ++col_tile) {
                    for (uint32_t lane = 0; lane < NPU_DECODE_GEMV_K_TILE; ++lane) {
                        const uint32_t idx = col_tile * NPU_DECODE_GEMV_K_TILE + lane;
                        if (idx >= NPU_DECODE_ATTN_HIDDEN_DIM) {
                            continue;
                        }
                        const uint32_t off =
                            col_tile * NPU_DECODE_GEMV_ACT_TILE_BYTES + lane * sizeof(ggml_fp16_t);
                        const uint16_t scale_bits =
                            npu_decode_attention_load_u16_le(src_scale + off);
                        const ggml_fp16_t folded =
                            ggml_fp32_to_fp16(ggml_fp16_to_fp32(scale_bits) * pv_scale_unboost);
                        npu_decode_attention_store_u16_le(dst_scale + off, folded);
                    }
                }
                o_act_scale_ptr = scale_cma;
                if (collect_profile) {
                    const uint64_t elapsed = static_cast<uint64_t>(ggml_time_us() - o_scale_stage_start_us);
                    profile.host_copy_activation_us += elapsed;
                    profile_add_extra_time("o_scale_fold_stage", elapsed);
                }
            }
        }

        rc = run_w4_projection(
                o_preloaded,
                "attn_o_pv_scale",
                out_cma,
                NPU_DECODE_ATTN_HIDDEN_DIM,
                KV260_STREAM_GEMV_ROLE_LINEAR,
                KV260_STREAM_POST_BYPASS,
                npu_decode_stream_output_precision(dst_type == GGML_TYPE_F32 && npu_decode_hw_f32_mvout_enabled()),
                0,
                o_act_scale_ptr);
        if (rc != 0) {
            free_stream_attention();
            cleanup();
            return fail("attention stream O projection failed");
        }
        const bool hw_f32_mvout = dst_type == GGML_TYPE_F32 && npu_decode_hw_f32_mvout_enabled();
        const int64_t o_output_copy_start_us = collect_profile ? ggml_time_us() : 0;
        if (dst_type == GGML_TYPE_F16) {
            std::memcpy(dst_data, out_cma, NPU_DECODE_ATTN_HIDDEN_DIM * sizeof(ggml_fp16_t));
        } else if (hw_f32_mvout) {
            std::memcpy(dst_data, out_cma, NPU_DECODE_ATTN_HIDDEN_DIM * sizeof(float));
        } else {
            npu_decode_convert_output_fp16_to_f32(
                    static_cast<const ggml_fp16_t *>(out_cma),
                    reinterpret_cast<float *>(dst_data),
                    NPU_DECODE_ATTN_HIDDEN_DIM);
        }
        if (collect_profile) {
            const uint64_t elapsed = static_cast<uint64_t>(ggml_time_us() - o_output_copy_start_us);
            profile.postprocess_us += elapsed;
            profile_add_extra_time("o_output_copy", elapsed);
        }
        GGML_UNUSED(kq_scale);
        GGML_UNUSED(dst_nb1);
        const int64_t cleanup_start_us = collect_profile ? ggml_time_us() : 0;
        free_stream_attention();
        cleanup();
        if (collect_profile) {
            profile.cleanup_us = static_cast<uint64_t>(ggml_time_us() - cleanup_start_us);
            profile.total_us = static_cast<uint64_t>(ggml_time_us() - total_start_us);
            profile.setup_us =
                profile.validation_us +
                profile.preload_lookup_us +
                profile.overlay_ensure_us +
                profile.heap_update_us +
                profile.runtime_init_us +
                profile.memory_base_us +
                profile.alloc_us;
            npu_decode_profile_write(profile);
        }
        npu_decode_attention_cache & cache = npu_decode_attention_get_cache();
        std::lock_guard<std::mutex> lock(cache.mutex);
        if (cache.layers.size() > static_cast<size_t>(layer_id)) {
            npu_decode_attention_layer_cache & done_layer = cache.layers[static_cast<size_t>(layer_id)];
            done_layer.converted_len =
                std::max<uint32_t>(
                        done_layer.converted_len,
                        static_cast<uint32_t>(position) + 1u);
            done_layer.last_position = static_cast<uint32_t>(position);
            done_layer.has_last_position = true;
        }
        return true;
    }

    npu_decode_attention_run_w4_projection(
            op_name,
            k_preloaded,
            act_data,
            act_type,
            act_nb0,
            act_nb1,
            act_cma,
            scale_cma,
            weight_cma,
            NPU_DECODE_ATTN_OUTPUT_SPM_BASE,
            npu_decode_attention_flow(
                    NPU_DECODE_SRC_GEMV_STREAM,
                    0,
                    NPU_DECODE_UNARY_ROPE,
                    NPU_DECODE_BINARY_BYPASS,
                    NPU_DECODE_REDUCE_BYPASS,
                    NPU_DECODE_DST_OUTPUT_SPM,
                    NPU_DECODE_ATTN_KV_DIM,
                    static_cast<uint16_t>(position),
                    1,
                    true),
                npu_decode_attention_npu_cache_cell_from_position(static_cast<uint32_t>(position)));
    npu_decode_attention_kv_scale_regs k_regs;
    if (!npu_decode_attention_read_scale_regs(false, &k_regs)) {
        cleanup();
        return fail("failed to read K scale regs");
    }
    std::memset(kv_window_cma, 0, kv_dump_bytes);
    npu_decode_stream_unsupported_void(op_name, kv_window_cma, NPU_DECODE_ATTN_OUTPUT_SPM_BASE, kv_dump_bytes, 1, 0);
    npu_decode_attention_append_quant_from_output(
            static_cast<uint8_t *>(layer_cache->k_cache),
            static_cast<const uint8_t *>(kv_window_cma),
            static_cast<uint32_t>(position),
            k_regs);

    npu_decode_attention_run_w4_projection(
            op_name,
            v_preloaded,
            act_data,
            act_type,
            act_nb0,
            act_nb1,
            act_cma,
            scale_cma,
            weight_cma,
            NPU_DECODE_ATTN_OUTPUT_SPM_BASE,
            npu_decode_attention_flow(
                    NPU_DECODE_SRC_GEMV_STREAM,
                    0,
                    NPU_DECODE_UNARY_BYPASS,
                    NPU_DECODE_BINARY_BYPASS,
                    NPU_DECODE_REDUCE_BYPASS,
                    NPU_DECODE_DST_OUTPUT_SPM,
                    NPU_DECODE_ATTN_KV_DIM,
                    static_cast<uint16_t>(position),
                    1,
                    true,
                    false,
                    true),
            npu_decode_attention_npu_cache_cell_from_position(static_cast<uint32_t>(position)));
    npu_decode_attention_kv_scale_regs v_regs;
    if (!npu_decode_attention_read_scale_regs(true, &v_regs)) {
        cleanup();
        return fail("failed to read V scale regs");
    }
    std::memset(kv_window_cma, 0, kv_dump_bytes);
    npu_decode_stream_unsupported_void(op_name, kv_window_cma, NPU_DECODE_ATTN_OUTPUT_SPM_BASE, kv_dump_bytes, 1, 0);
    npu_decode_attention_append_quant_from_output(
            static_cast<uint8_t *>(layer_cache->v_cache),
            static_cast<const uint8_t *>(kv_window_cma),
            static_cast<uint32_t>(position),
            v_regs,
            attention_stream_v2);

    npu_decode_attention_run_w4_projection(
            op_name,
            q_preloaded,
            act_data,
            act_type,
            act_nb0,
            act_nb1,
            act_cma,
            scale_cma,
            weight_cma,
            NPU_DECODE_ATTN_Q_GROUP_ACT_BASE,
            npu_decode_attention_flow(
                    NPU_DECODE_SRC_GEMV_STREAM,
                    0,
                    NPU_DECODE_UNARY_ROPE,
                    NPU_DECODE_BINARY_BYPASS,
                    NPU_DECODE_REDUCE_BYPASS,
                    NPU_DECODE_DST_ACT_BUFFER,
                    NPU_DECODE_ATTN_HIDDEN_DIM,
                    static_cast<uint16_t>(position)));

    if (npu_decode_attention_dump_requested(layer_id, position)) {
        const uint32_t q_dump_bytes = npu_decode_align_up_u32(
                NPU_DECODE_ATTN_HIDDEN_DIM * sizeof(ggml_fp16_t),
                NPU_DECODE_GEMV_DMA_ALIGN);
        npu_decode_attention_run_w4_projection(
                op_name,
                q_preloaded,
                act_data,
                act_type,
                act_nb0,
                act_nb1,
                act_cma,
                scale_cma,
                weight_cma,
                NPU_DECODE_ATTN_OUTPUT_SPM_BASE,
                npu_decode_attention_flow(
                        NPU_DECODE_SRC_GEMV_STREAM,
                        0,
                        NPU_DECODE_UNARY_ROPE,
                        NPU_DECODE_BINARY_BYPASS,
                        NPU_DECODE_REDUCE_BYPASS,
                        NPU_DECODE_DST_OUTPUT_SPM,
                        NPU_DECODE_ATTN_HIDDEN_DIM,
                        static_cast<uint16_t>(position)));
        std::memset(out_cma, 0, host_output_bytes);
        npu_decode_stream_unsupported_void(
                op_name,
                out_cma,
                NPU_DECODE_ATTN_OUTPUT_SPM_BASE,
                q_dump_bytes,
                1,
                0);
        npu_decode_attention_write_debug_dump(
                op_name,
                layer_id,
                position,
                seq_len,
                out_cma,
                q_dump_bytes,
                layer_cache->k_cache,
                layer_cache->v_cache);
    }

    cleanup();
    return fail("legacy attention V-transpose path is disabled for 4K KV cache layout");

    for (uint32_t kv_head = 0; kv_head < NPU_DECODE_ATTN_KV_HEADS; ++kv_head) {
        uint8_t * raw_k_head =
            static_cast<uint8_t *>(layer_cache->k_cache) +
            kv_head * NPU_DECODE_ATTN_KV_HEAD_STRIDE_BYTES +
            npu_decode_attention_kv_cache_scale_region_bytes();
        npu_decode_attention_fill_k_score_scale(
                static_cast<uint8_t *>(scale_cma),
                k_score_scale_bytes,
                static_cast<const uint8_t *>(layer_cache->k_cache),
                kv_head,
                seq_len);
        npu_decode_stream_unsupported_void(
                op_name,
                scale_cma,
                NPU_DECODE_ATTN_WEIGHT_SCALE_SPM_BASE,
                k_score_scale_bytes,
                2,
                NPU_DECODE_GEMV_INPUT_TYPE_DATA,
                0,
                0);
        npu_decode_stream_unsupported_void(
                op_name,
                raw_k_head,
                NPU_DECODE_ATTN_INPUT_SPM_BASE,
                raw_k_mvin_bytes,
                1,
                NPU_DECODE_GEMV_INPUT_TYPE_WEIGHT);

        const uint32_t q_vec_addr =
            NPU_DECODE_ATTN_Q_GROUP_ACT_BASE +
            kv_head * NPU_DECODE_ATTN_GROUP_SIZE * NPU_DECODE_ATTN_HEAD_DIM * sizeof(ggml_fp16_t);
        for (uint32_t group = 0; group < NPU_DECODE_ATTN_GROUP_SIZE; ++group) {
            npu_decode_stream_unsupported_void(
                    op_name,
                    NPU_DECODE_ATTN_INPUT_SPM_BASE,
                    q_vec_addr + group * NPU_DECODE_ATTN_HEAD_DIM * sizeof(ggml_fp16_t),
                    NPU_DECODE_ATTN_HEAD_DIM,
                    static_cast<uint16_t>(seq_len),
                    NPU_DECODE_ATTN_SCORE_ACT_BASE + group * score_prob_stride_bytes,
                    NPU_DECODE_ATTN_WEIGHT_SCALE_SPM_BASE,
                    NPU_DECODE_GEMV_MODE_W8A16,
                    npu_decode_attention_flow(
                            NPU_DECODE_SRC_GEMV_STREAM,
                            0,
                            NPU_DECODE_UNARY_BYPASS,
                            NPU_DECODE_BINARY_BYPASS,
                            NPU_DECODE_REDUCE_SOFTMAX,
                            NPU_DECODE_DST_ACT_BUFFER,
                            static_cast<uint16_t>(seq_len),
                            static_cast<uint16_t>(position),
                            1),
                    std::numeric_limits<uint32_t>::max(),
                    0,
                    0);
        }

        const uint32_t value_act_scale_base =
            NPU_DECODE_ATTN_SCORE_ACT_BASE +
            NPU_DECODE_ATTN_GROUP_SIZE * score_prob_stride_bytes;
        npu_decode_stream_unsupported_void(
                static_cast<uint8_t *>(layer_cache->v_cache) +
                    kv_head * NPU_DECODE_ATTN_KV_HEAD_STRIDE_BYTES,
                NPU_DECODE_ATTN_INPUT_SPM_BASE,
                NPU_DECODE_ATTN_WEIGHT_SCALE_SPM_BASE,
                value_act_scale_base,
                static_cast<uint16_t>(seq_len));

        const uint32_t out_addr =
            (pre_o_output ? NPU_DECODE_ATTN_OUTPUT_SPM_BASE : NPU_DECODE_ATTN_ATTENTION_ACT_BASE) +
            kv_head * NPU_DECODE_ATTN_GROUP_SIZE * NPU_DECODE_ATTN_HEAD_DIM * sizeof(ggml_fp16_t);
        npu_decode_stream_unsupported_void(
                op_name,
                NPU_DECODE_ATTN_INPUT_SPM_BASE,
                NPU_DECODE_ATTN_SCORE_ACT_BASE,
                static_cast<uint16_t>(seq_len),
                NPU_DECODE_ATTN_HEAD_DIM,
                out_addr,
                NPU_DECODE_ATTN_WEIGHT_SCALE_SPM_BASE,
                NPU_DECODE_GEMV_MODE_W8A16,
                npu_decode_attention_flow(
                        NPU_DECODE_SRC_GEMV_STREAM,
                        0,
                        NPU_DECODE_UNARY_BYPASS,
                        NPU_DECODE_BINARY_BYPASS,
                        NPU_DECODE_REDUCE_BYPASS,
                        pre_o_output ? NPU_DECODE_DST_OUTPUT_SPM : NPU_DECODE_DST_ACT_BUFFER,
                        NPU_DECODE_ATTN_HEAD_DIM,
                        static_cast<uint16_t>(position),
                        NPU_DECODE_ATTN_GROUP_SIZE,
                        false,
                        true),
                std::numeric_limits<uint32_t>::max(),
                0,
                score_prob_stride_bytes);
    }

    if (pre_o_output) {
        const bool hw_f32_mvout = dst_type == GGML_TYPE_F32 && npu_decode_hw_f32_mvout_enabled();
        const uint8_t output_precision = npu_decode_mvout_precision(hw_f32_mvout);
        npu_decode_stream_unsupported_void(
                op_name,
                out_cma,
                NPU_DECODE_ATTN_OUTPUT_SPM_BASE,
                host_output_bytes,
                output_precision,
                0);
        if (dst_type == GGML_TYPE_F16) {
            std::memcpy(dst_data, out_cma, NPU_DECODE_ATTN_HIDDEN_DIM * sizeof(ggml_fp16_t));
        } else if (hw_f32_mvout) {
            std::memcpy(dst_data, out_cma, NPU_DECODE_ATTN_HIDDEN_DIM * sizeof(float));
        } else {
            npu_decode_convert_output_fp16_to_f32(
                    static_cast<const ggml_fp16_t *>(out_cma),
                    reinterpret_cast<float *>(dst_data),
                    NPU_DECODE_ATTN_HIDDEN_DIM);
        }
        GGML_UNUSED(kq_scale);
        GGML_UNUSED(dst_nb1);
        cleanup();
        npu_decode_attention_cache & cache = npu_decode_attention_get_cache();
        std::lock_guard<std::mutex> lock(cache.mutex);
        if (cache.layers.size() > static_cast<size_t>(layer_id)) {
            npu_decode_attention_layer_cache & done_layer = cache.layers[static_cast<size_t>(layer_id)];
            done_layer.converted_len =
                std::max<uint32_t>(
                        done_layer.converted_len,
                        static_cast<uint32_t>(position) + 1u);
            done_layer.last_position = static_cast<uint32_t>(position);
            done_layer.has_last_position = true;
        }
        return true;
    }

    const npu_decode_gemv_layout & o_layout = o_preloaded->pingpong_layout;
    if (o_preloaded->act_scale.size() != o_layout.act_bytes) {
        cleanup();
        return fail("O projection act scale size mismatch");
    }
    std::memcpy(act_cma, o_preloaded->act_scale.data(), o_layout.act_bytes);
    npu_decode_stream_unsupported_void(
            op_name,
            act_cma,
            NPU_DECODE_ATTN_ATTENTION_ACT_BASE + o_layout.act_bytes,
            o_layout.act_bytes,
            2,
            NPU_DECODE_GEMV_INPUT_TYPE_ACT);
    npu_decode_attention_run_w4_projection(
            op_name,
            o_preloaded,
            act_data,
            act_type,
            act_nb0,
            act_nb1,
            act_cma,
            scale_cma,
            weight_cma,
            NPU_DECODE_ATTN_OUTPUT_SPM_BASE,
            npu_decode_attention_flow(
                    NPU_DECODE_SRC_GEMV_STREAM,
                    0,
                    NPU_DECODE_UNARY_BYPASS,
                    NPU_DECODE_BINARY_BYPASS,
                    NPU_DECODE_REDUCE_BYPASS,
                    NPU_DECODE_DST_OUTPUT_SPM,
                    NPU_DECODE_ATTN_HIDDEN_DIM,
                    static_cast<uint16_t>(position)),
            0,
            0,
            false,
            NPU_DECODE_ATTN_ATTENTION_ACT_BASE);

    const bool hw_f32_mvout = dst_type == GGML_TYPE_F32 && npu_decode_hw_f32_mvout_enabled();
    const uint8_t output_precision = npu_decode_mvout_precision(hw_f32_mvout);
    npu_decode_stream_unsupported_void(
            op_name,
            out_cma,
            NPU_DECODE_ATTN_OUTPUT_SPM_BASE,
            host_output_bytes,
            output_precision,
            0);
    if (dst_type == GGML_TYPE_F16) {
        std::memcpy(dst_data, out_cma, NPU_DECODE_ATTN_HIDDEN_DIM * sizeof(ggml_fp16_t));
    } else if (hw_f32_mvout) {
        std::memcpy(dst_data, out_cma, NPU_DECODE_ATTN_HIDDEN_DIM * sizeof(float));
    } else {
        npu_decode_convert_output_fp16_to_f32(
                static_cast<const ggml_fp16_t *>(out_cma),
                reinterpret_cast<float *>(dst_data),
                NPU_DECODE_ATTN_HIDDEN_DIM);
    }
    GGML_UNUSED(kq_scale);
    GGML_UNUSED(dst_nb1);

    {
        npu_decode_attention_cache & cache = npu_decode_attention_get_cache();
        std::lock_guard<std::mutex> lock(cache.mutex);
        if (cache.layers.size() > static_cast<size_t>(layer_id)) {
            npu_decode_attention_layer_cache & done_layer = cache.layers[static_cast<size_t>(layer_id)];
            done_layer.converted_len =
                std::max<uint32_t>(
                        done_layer.converted_len,
                        static_cast<uint32_t>(position) + 1u);
            done_layer.last_position = static_cast<uint32_t>(position);
            done_layer.has_last_position = true;
        }
    }

    cleanup();
    return true;
}
