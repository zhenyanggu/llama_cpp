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
                                 void * act_ptr,
                                 void * act_scale_ptr,
                                 void * output_ptr,
                                 uint16_t rows,
                                 uint16_t cols,
                                 kv260_stream_post_op post_op,
                                 kv260_stream_output_precision precision) {
            kv260_stream_gemv_desc desc = {};
            desc.act_ptr = act_ptr;
            desc.act_scale_ptr = act_scale_ptr;
            desc.weight_payload_ptr = stream_weight_ptr(tensor);
            desc.weight_scale_ptr = stream_weight_scale_ptr(tensor);
            desc.output_ptr = output_ptr;
            desc.m = rows;
            desc.n = cols;
            desc.mode = KV260_STREAM_DECODE_GEMV_W4A16;
            desc.output_precision = precision;
            desc.role = KV260_STREAM_GEMV_ROLE_MLP;
            desc.dst = KV260_STREAM_GEMV_DST_OUTPUT;
            desc.post_op = post_op;
            desc.flags = KV260_STREAM_GEMV_F_ENABLE_ACT_SCALE;
            GGML_UNUSED(layout);
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
                act_cma,
                stream_act_scale_ptr(gate_preloaded),
                gate_cma,
                static_cast<uint16_t>(gate->out_channels),
                static_cast<uint16_t>(gate->k),
                KV260_STREAM_POST_SILU,
                KV260_STREAM_OUTPUT_FP16);
        if (rc != 0) {
            npu_decode_mem_free(gate_cma);
            npu_decode_mem_free(up_cma);
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
                act_cma,
                stream_act_scale_ptr(up_preloaded),
                up_cma,
                static_cast<uint16_t>(up->out_channels),
                static_cast<uint16_t>(up->k),
                KV260_STREAM_POST_BYPASS,
                KV260_STREAM_OUTPUT_FP16);
        if (rc != 0) {
            npu_decode_mem_free(gate_cma);
            npu_decode_mem_free(up_cma);
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
                up_cma,
                gate_cma,
                out_cma,
                static_cast<uint16_t>(down->out_channels),
                static_cast<uint16_t>(down->k),
                KV260_STREAM_POST_BYPASS,
                npu_decode_stream_output_precision(hw_f32_mvout));
        if (rc != 0) {
            npu_decode_mem_free(gate_cma);
            npu_decode_mem_free(up_cma);
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

        npu_decode_mem_free(gate_cma);
        npu_decode_mem_free(up_cma);
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
            static_cast<uint32_t>(position) >= NPU_DECODE_ATTN_VISIBLE_WINDOW_TOKENS ||
            layer_id < 0) {
        return fail("unsupported position or layer");
    }
    if (act_type != GGML_TYPE_F16 && act_type != GGML_TYPE_F32) {
        return fail("activation tensor must be F16 or F32");
    }
    if (dst_type != GGML_TYPE_F16 && dst_type != GGML_TYPE_F32) {
        return fail("destination tensor must be F16 or F32");
    }
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

    if (!npu_decode_ensure_overlay_active(op_name)) {
        return fail("decode overlay switch failed");
    }
    npu_decode_update_runtime_heap_env();
    if (npu_decode_init_with_cma_retry(op_name, true) != 0) {
        return fail("decode runtime init failed");
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
        }
        if (layer_cache->v_cache == nullptr) {
            layer_cache->v_cache = npu_decode_mem_alloc(
                    NPU_DECODE_ATTN_KV_HEADS * NPU_DECODE_ATTN_KV_HEAD_STRIDE_BYTES);
        }
        if (layer_cache->k_cache == nullptr || layer_cache->v_cache == nullptr) {
            return fail("decode attention KV CMA allocation failed");
        }
        if (static_cast<uint32_t>(position) < layer_cache->converted_len) {
            std::memset(layer_cache->k_cache, 0, NPU_DECODE_ATTN_KV_HEADS * NPU_DECODE_ATTN_KV_HEAD_STRIDE_BYTES);
            std::memset(layer_cache->v_cache, 0, NPU_DECODE_ATTN_KV_HEADS * NPU_DECODE_ATTN_KV_HEAD_STRIDE_BYTES);
            layer_cache->converted_len = 0;
        }
        const uint32_t prefix_tokens = static_cast<uint32_t>(position);
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
        layer_cache->last_position = static_cast<uint32_t>(position);
    }

    const uint32_t seq_len = static_cast<uint32_t>(position) + 1u;
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
    for (uint32_t pair = 0; pair < NPU_DECODE_ATTN_HEAD_DIM / 2u; ++pair) {
        const double inv_freq = std::pow(
                static_cast<double>(rope_freq_base),
                -2.0 * static_cast<double>(pair) / static_cast<double>(NPU_DECODE_ATTN_HEAD_DIM));
        for (uint32_t slot = 0; slot < 2; ++slot) {
            const double theta =
                static_cast<double>(even_position + slot) *
                static_cast<double>(rope_freq_scale) *
                inv_freq;
            const uint16_t cos_fp16 = ggml_fp32_to_fp16(static_cast<float>(std::cos(theta)));
            const uint16_t sin_fp16 = ggml_fp32_to_fp16(static_cast<float>(std::sin(theta)));
            npu_decode_attention_store_u16_le(
                    rope_window + slot * 128u + pair * sizeof(uint16_t),
                    cos_fp16);
            npu_decode_attention_store_u16_le(
                    rope_window + slot * 128u + NPU_DECODE_GEMV_LINE_BYTES + pair * sizeof(uint16_t),
                    sin_fp16);
        }
    }

    npu_decode_reset();
    rope_lut_cma = npu_decode_mem_alloc(rope_lut.size());
    if (rope_lut_cma == nullptr) {
        cleanup();
        return fail("decode attention RoPE LUT CMA allocation failed");
    }
    std::memcpy(rope_lut_cma, rope_lut.data(), rope_lut.size());

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

    if (npu_decode_attention_dump_requested(layer_id)) {
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

    if (attention_stream_v2) {
        void * decode_cma_base = npu_decode_memory_base();
        if (decode_cma_base == nullptr ||
                !npu_decode_stream_w4_cma_resident(q_preloaded) ||
                !npu_decode_stream_w4_cma_resident(o_preloaded)) {
            cleanup();
            return fail("attention stream W4 assets are not CMA resident");
        }

        const uint32_t q_cma_bytes = npu_decode_host_output_bytes(NPU_DECODE_ATTN_HIDDEN_DIM, false);
        const uint32_t q_head_stride = NPU_DECODE_GEMV_DMA_ALIGN;
        const uint32_t q_head_cma_bytes = NPU_DECODE_ATTN_Q_HEADS * q_head_stride;
        const uint32_t prob_cma_bytes =
            NPU_DECODE_ATTN_KV_HEADS * NPU_DECODE_ATTN_GROUP_SIZE * score_prob_stride_bytes;
        const uint32_t pv_head_stride = npu_decode_align_up_u32(
                NPU_DECODE_ATTN_GROUP_SIZE * NPU_DECODE_ATTN_HEAD_DIM * sizeof(ggml_fp16_t),
                NPU_DECODE_GEMV_DMA_ALIGN);
        void * q_cma = npu_decode_mem_alloc(q_cma_bytes);
        void * q_head_cma = npu_decode_mem_alloc(q_head_cma_bytes);
        void * prob_cma = npu_decode_mem_alloc(prob_cma_bytes);
        void * pv_head_cma = npu_decode_mem_alloc(pv_head_stride);
        if (q_cma == nullptr || q_head_cma == nullptr || prob_cma == nullptr || pv_head_cma == nullptr) {
            npu_decode_mem_free(q_cma);
            npu_decode_mem_free(q_head_cma);
            npu_decode_mem_free(prob_cma);
            npu_decode_mem_free(pv_head_cma);
            cleanup();
            return fail("attention stream CMA allocation failed");
        }
        std::memset(q_cma, 0, q_cma_bytes);
        std::memset(q_head_cma, 0, q_head_cma_bytes);
        std::memset(prob_cma, 0, prob_cma_bytes);
        std::memset(pv_head_cma, 0, pv_head_stride);

        auto free_stream_attention = [&]() {
            npu_decode_mem_free(q_cma);
            npu_decode_mem_free(q_head_cma);
            npu_decode_mem_free(prob_cma);
            npu_decode_mem_free(pv_head_cma);
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
        kv260_stream_gemv_desc q_desc = {};
        q_desc.act_ptr = act_cma;
        q_desc.act_scale_ptr = stream_act_scale_ptr(q_preloaded);
        q_desc.weight_payload_ptr = stream_weight_ptr(q_preloaded);
        q_desc.weight_scale_ptr = stream_weight_scale_ptr(q_preloaded);
        q_desc.output_ptr = q_cma;
        q_desc.m = NPU_DECODE_ATTN_HIDDEN_DIM;
        q_desc.n = NPU_DECODE_ATTN_HIDDEN_DIM;
        q_desc.mode = KV260_STREAM_DECODE_GEMV_W4A16;
        q_desc.output_precision = KV260_STREAM_OUTPUT_FP16;
        q_desc.role = KV260_STREAM_GEMV_ROLE_LINEAR;
        q_desc.dst = KV260_STREAM_GEMV_DST_OUTPUT;
        q_desc.post_op = KV260_STREAM_POST_ROPE;
        q_desc.flags = KV260_STREAM_GEMV_F_ENABLE_ACT_SCALE;
        q_desc.rope_lut_ptr = rope_lut_cma;
        q_desc.elem_count = NPU_DECODE_ATTN_HIDDEN_DIM;
        q_desc.position = static_cast<uint16_t>(position);
        int rc = npu_decode_stream_gemv(op_name, q_desc);
        if (rc != 0) {
            free_stream_attention();
            cleanup();
            return fail("attention stream Q projection failed");
        }

        for (uint32_t head = 0; head < NPU_DECODE_ATTN_Q_HEADS; ++head) {
            std::memcpy(
                    static_cast<uint8_t *>(q_head_cma) + head * q_head_stride,
                    static_cast<uint8_t *>(q_cma) + head * NPU_DECODE_ATTN_HEAD_DIM * sizeof(ggml_fp16_t),
                    NPU_DECODE_ATTN_HEAD_DIM * sizeof(ggml_fp16_t));
        }

        for (uint32_t kv_head = 0; kv_head < NPU_DECODE_ATTN_KV_HEADS; ++kv_head) {
            uint8_t * k_head_base =
                static_cast<uint8_t *>(layer_cache->k_cache) +
                kv_head * NPU_DECODE_ATTN_KV_HEAD_STRIDE_BYTES;
            uint8_t * k_quant_base = k_head_base + scale_region;
            uint8_t * prob_group_base =
                static_cast<uint8_t *>(prob_cma) +
                kv_head * NPU_DECODE_ATTN_GROUP_SIZE * score_prob_stride_bytes;
            for (uint32_t group = 0; group < NPU_DECODE_ATTN_GROUP_SIZE; ++group) {
                kv260_stream_gemv_desc qk_desc = {};
                qk_desc.act_ptr =
                    static_cast<uint8_t *>(q_head_cma) +
                    (kv_head * NPU_DECODE_ATTN_GROUP_SIZE + group) * q_head_stride;
                qk_desc.weight_payload_ptr = k_quant_base;
                qk_desc.weight_scale_ptr = k_head_base;
                qk_desc.output_ptr = prob_group_base + group * score_prob_stride_bytes;
                qk_desc.m = static_cast<uint16_t>(seq_len);
                qk_desc.n = NPU_DECODE_ATTN_HEAD_DIM;
                qk_desc.mode = KV260_STREAM_DECODE_GEMV_W8A16;
                qk_desc.output_precision = KV260_STREAM_OUTPUT_FP16;
                qk_desc.role = KV260_STREAM_GEMV_ROLE_QK;
                qk_desc.dst = KV260_STREAM_GEMV_DST_OUTPUT;
                qk_desc.post_op = KV260_STREAM_POST_SOFTMAX;
                qk_desc.elem_count = static_cast<uint16_t>(seq_len);
                qk_desc.position = static_cast<uint16_t>(position);
                qk_desc.group_count = 1;
                rc = npu_decode_stream_gemv(op_name, qk_desc);
                if (rc != 0) {
                    free_stream_attention();
                    cleanup();
                    return fail("attention stream QK failed");
                }
            }

            uint8_t * v_head_base =
                static_cast<uint8_t *>(layer_cache->v_cache) +
                kv_head * NPU_DECODE_ATTN_KV_HEAD_STRIDE_BYTES;
            kv260_stream_gemv_desc pv_desc = {};
            pv_desc.act_ptr = prob_group_base;
            pv_desc.act_scale_ptr = v_head_base;
            pv_desc.weight_payload_ptr = v_head_base + scale_region;
            pv_desc.output_ptr = pv_head_cma;
            pv_desc.weight_row_tile_stride_bytes = NPU_DECODE_ATTN_V_ROW_TILE_STRIDE_BYTES;
            pv_desc.weight_capacity_tokens = NPU_DECODE_ATTN_CACHE_CAPACITY_TOKENS;
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
            rc = npu_decode_stream_gemv(op_name, pv_desc);
            if (rc != 0) {
                free_stream_attention();
                cleanup();
                return fail("attention stream PV failed");
            }

            for (uint32_t group = 0; group < NPU_DECODE_ATTN_GROUP_SIZE; ++group) {
                std::memcpy(
                        static_cast<ggml_fp16_t *>(act_cma) +
                            (kv_head * NPU_DECODE_ATTN_GROUP_SIZE + group) * NPU_DECODE_ATTN_HEAD_DIM,
                        static_cast<uint8_t *>(pv_head_cma) +
                            group * NPU_DECODE_ATTN_HEAD_DIM * sizeof(ggml_fp16_t),
                        NPU_DECODE_ATTN_HEAD_DIM * sizeof(ggml_fp16_t));
            }
        }

        if (pre_o_output) {
            if (dst_type == GGML_TYPE_F16) {
                std::memcpy(dst_data, act_cma, NPU_DECODE_ATTN_HIDDEN_DIM * sizeof(ggml_fp16_t));
            } else {
                npu_decode_convert_output_fp16_to_f32(
                        static_cast<const ggml_fp16_t *>(act_cma),
                        reinterpret_cast<float *>(dst_data),
                        NPU_DECODE_ATTN_HIDDEN_DIM);
            }
            GGML_UNUSED(kq_scale);
            GGML_UNUSED(dst_nb1);
            free_stream_attention();
            cleanup();
            npu_decode_attention_cache & cache = npu_decode_attention_get_cache();
            std::lock_guard<std::mutex> lock(cache.mutex);
            if (cache.layers.size() > static_cast<size_t>(layer_id)) {
                cache.layers[static_cast<size_t>(layer_id)].converted_len =
                    std::max<uint32_t>(
                            cache.layers[static_cast<size_t>(layer_id)].converted_len,
                            static_cast<uint32_t>(position) + 1u);
            }
            return true;
        }

        std::vector<ggml_fp16_t> pv_vector(NPU_DECODE_ATTN_HIDDEN_DIM);
        std::memcpy(
                pv_vector.data(),
                act_cma,
                NPU_DECODE_ATTN_HIDDEN_DIM * sizeof(ggml_fp16_t));
        npu_decode_pack_activation_slice(
                act_cma,
                o_preloaded->pingpong_layout.act_bytes,
                o_preloaded->pingpong_layout,
                pv_vector.data(),
                GGML_TYPE_F16,
                sizeof(ggml_fp16_t),
                NPU_DECODE_ATTN_HIDDEN_DIM * sizeof(ggml_fp16_t),
                NPU_DECODE_ATTN_HIDDEN_DIM,
                0,
                NPU_DECODE_ATTN_HIDDEN_DIM,
                0);

        kv260_stream_gemv_desc o_desc = {};
        o_desc.act_ptr = act_cma;
        o_desc.act_scale_ptr = stream_act_scale_ptr(o_preloaded);
        o_desc.weight_payload_ptr = stream_weight_ptr(o_preloaded);
        o_desc.weight_scale_ptr = stream_weight_scale_ptr(o_preloaded);
        o_desc.output_ptr = out_cma;
        o_desc.m = NPU_DECODE_ATTN_HIDDEN_DIM;
        o_desc.n = NPU_DECODE_ATTN_HIDDEN_DIM;
        o_desc.mode = KV260_STREAM_DECODE_GEMV_W4A16;
        o_desc.output_precision = npu_decode_stream_output_precision(dst_type == GGML_TYPE_F32 && npu_decode_hw_f32_mvout_enabled());
        o_desc.role = KV260_STREAM_GEMV_ROLE_LINEAR;
        o_desc.dst = KV260_STREAM_GEMV_DST_OUTPUT;
        o_desc.post_op = KV260_STREAM_POST_BYPASS;
        o_desc.flags = KV260_STREAM_GEMV_F_ENABLE_ACT_SCALE;
        rc = npu_decode_stream_gemv(op_name, o_desc);
        if (rc != 0) {
            free_stream_attention();
            cleanup();
            return fail("attention stream O projection failed");
        }
        const bool hw_f32_mvout = dst_type == GGML_TYPE_F32 && npu_decode_hw_f32_mvout_enabled();
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
        free_stream_attention();
        cleanup();
        npu_decode_attention_cache & cache = npu_decode_attention_get_cache();
        std::lock_guard<std::mutex> lock(cache.mutex);
        if (cache.layers.size() > static_cast<size_t>(layer_id)) {
            cache.layers[static_cast<size_t>(layer_id)].converted_len =
                std::max<uint32_t>(
                        cache.layers[static_cast<size_t>(layer_id)].converted_len,
                        static_cast<uint32_t>(position) + 1u);
        }
        return true;
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
            cache.layers[static_cast<size_t>(layer_id)].converted_len =
                std::max<uint32_t>(
                        cache.layers[static_cast<size_t>(layer_id)].converted_len,
                        static_cast<uint32_t>(position) + 1u);
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
            cache.layers[static_cast<size_t>(layer_id)].converted_len =
                std::max<uint32_t>(
                        cache.layers[static_cast<size_t>(layer_id)].converted_len,
                        static_cast<uint32_t>(position) + 1u);
        }
    }

    cleanup();
    return true;
}
