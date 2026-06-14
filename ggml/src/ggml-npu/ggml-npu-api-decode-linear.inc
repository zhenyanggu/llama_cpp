bool ggml_backend_npu_decode_w4a16_preload(
        const char * weight_name,
        const void * q4_data,
        int64_t packed_k,
        int64_t out_channels,
        int64_t q4_nb1,
        const void * scale_data,
        int scale_type,
        int64_t scale_nb0,
        int64_t scale_nb1,
        const void * zero_data,
        int zero_type,
        int64_t zero_nb0,
        int64_t zero_nb1,
        int64_t k,
        const float * smooth_scale,
        size_t smooth_scale_len) {
    using namespace ggml_npu;

    auto fail = [weight_name](const char * reason) {
        if (npu_decode_w4a16_debug_log_enabled()) {
            GGML_LOG_WARN("%s: reject %s: %s\n", __func__, weight_name != nullptr ? weight_name : "(unnamed)", reason);
        }
        return false;
    };

    if (weight_name == nullptr || weight_name[0] == '\0' ||
            q4_data == nullptr || scale_data == nullptr || zero_data == nullptr) {
        return fail("null pointer");
    }
    if (k <= 0 || out_channels <= 0 || packed_k != (k + 1) / 2) {
        return fail("invalid dimensions");
    }
    if (k > std::numeric_limits<uint16_t>::max() || out_channels > std::numeric_limits<uint16_t>::max()) {
        return fail("dimension exceeds uint16 range");
    }
    if (scale_type != GGML_TYPE_F16 && scale_type != GGML_TYPE_F32) {
        return fail("scale tensor must be F16 or F32");
    }
    if (zero_type != GGML_TYPE_F16 && zero_type != GGML_TYPE_F32) {
        return fail("zero tensor must be F16 or F32");
    }
    if (smooth_scale == nullptr || smooth_scale_len != static_cast<size_t>(k)) {
        return fail("invalid smooth scale");
    }

    const int64_t groups = (k + NPU_DECODE_GEMV_K_TILE - 1) / NPU_DECODE_GEMV_K_TILE;
    if (!npu_decode_validate_symmetric_zero(
            zero_data,
            zero_type,
            zero_nb0,
            zero_nb1,
            out_channels,
            groups,
            weight_name)) {
        return false;
    }

    const uint32_t total_row_tiles = npu_decode_ceil_div_u32(static_cast<uint32_t>(out_channels), NPU_DECODE_GEMV_M_TILE);
    uint32_t max_block_row_tiles = 0;
    for (uint32_t row_tiles = total_row_tiles; row_tiles >= 1; --row_tiles) {
        const uint32_t candidate_rows = std::min<uint32_t>(row_tiles * NPU_DECODE_GEMV_M_TILE, static_cast<uint32_t>(out_channels));
        npu_decode_gemv_layout candidate = {};
        if (npu_decode_make_gemv_layout(candidate_rows, static_cast<uint32_t>(k), &candidate)) {
            max_block_row_tiles = row_tiles;
            break;
        }
        if (row_tiles == 1) {
            break;
        }
    }
    if (max_block_row_tiles == 0) {
        return fail("no M block fits decode GEMV SPM capacity");
    }

    npu_decode_preloaded_tensor entry;
    entry.packed_k = packed_k;
    entry.out_channels = out_channels;
    entry.q4_nb1 = q4_nb1;
    entry.k = k;
    entry.max_block_rows = max_block_row_tiles * NPU_DECODE_GEMV_M_TILE;
    entry.zero_validated = true;
    entry.prefer_block_cma = npu_decode_fused_ffn_cma_enabled() && npu_decode_weight_name_is_ffn(weight_name);
    entry.inv_smooth_scale.resize(static_cast<size_t>(k));
    for (int64_t i = 0; i < k; ++i) {
        const float smooth = smooth_scale[i];
        entry.inv_smooth_scale[static_cast<size_t>(i)] =
            smooth != 0.0f && std::isfinite(smooth) ? 1.0f / smooth : 1.0f;
    }
    if (!npu_decode_make_gemv_layout(entry.max_block_rows, static_cast<uint32_t>(k), &entry.max_layout)) {
        return fail("max block layout failed");
    }
    entry.act_scale.resize(entry.max_layout.act_bytes);
    npu_decode_pack_act_scale(
            entry.act_scale.data(),
            entry.act_scale.size(),
            entry.max_layout,
            entry.inv_smooth_scale.data(),
            k);
    entry.pingpong_available = npu_decode_make_pingpong_layout(
            static_cast<uint32_t>(out_channels),
            static_cast<uint32_t>(k),
            &entry.pingpong_layout);
    if (entry.pingpong_available) {
        entry.pingpong_scale.resize(entry.pingpong_layout.scale_bytes);
        entry.pingpong_weight.resize(entry.pingpong_layout.weight_bytes);
        npu_decode_pack_pingpong_scale_weight(
                entry.pingpong_scale,
                entry.pingpong_weight,
                entry.pingpong_layout,
                q4_data,
                packed_k,
                q4_nb1,
                scale_data,
                scale_type,
                scale_nb0,
                scale_nb1,
                out_channels,
                k);
    }

    void * cma_base = nullptr;
    uint32_t cma_map_size = 0;
    uint32_t cma_base_offset = npu_decode_parse_size_env("NPU_CMA_HEAP_OFFSET", 0);
    uint32_t cma_limit_bytes = npu_decode_parse_size_env("NPU_CMA_HEAP_SIZE", 0);
    uint32_t cma_cursor = 0;
    {
        npu_decode_preload_cache & cache = npu_decode_get_preload_cache();
        std::lock_guard<std::mutex> lock(cache.mutex);
        if (cache.cma_limit_bytes != 0) {
            cma_base_offset = cache.cma_base_offset;
            cma_limit_bytes = cache.cma_limit_bytes;
            cma_cursor = cache.cma_cursor;
        }
    }
    if (npu_decode_preload_cma_enabled() &&
            npu_decode_cma_init() == 0) {
        cma_base = npu_decode_memory_base();
        cma_map_size = npu_decode_memory_size();
        if (cma_limit_bytes == 0 && cma_base_offset < cma_map_size) {
            cma_limit_bytes = cma_map_size - cma_base_offset;
        }
    }
    if (cma_base_offset >= cma_map_size || cma_limit_bytes > cma_map_size - cma_base_offset) {
        cma_base = nullptr;
        cma_limit_bytes = 0;
        cma_cursor = 0;
    }

    auto reserve_pingpong_cma = [&]() {
        if (!entry.pingpong_available || cma_base == nullptr || cma_limit_bytes == 0) {
            return;
        }
        if (entry.act_scale_cma_bytes == entry.pingpong_layout.act_bytes &&
                entry.pingpong_scale_cma_bytes == entry.pingpong_layout.scale_bytes &&
                entry.pingpong_weight_cma_bytes == entry.pingpong_layout.weight_bytes) {
            return;
        }
        const uint32_t pingpong_cma_start = cma_cursor;
        if (!entry.act_scale.empty() && entry.act_scale_cma_bytes != entry.pingpong_layout.act_bytes) {
            const uint32_t act_scale_cursor = npu_decode_align_up_bytes(cma_cursor, NPU_DECODE_GEMV_DMA_ALIGN);
            if (entry.act_scale.size() <= static_cast<size_t>(std::numeric_limits<uint32_t>::max()) &&
                    act_scale_cursor <= cma_limit_bytes &&
                    static_cast<uint32_t>(entry.act_scale.size()) <= cma_limit_bytes - act_scale_cursor) {
                entry.act_scale_cma_offset = cma_base_offset + act_scale_cursor;
                entry.act_scale_cma_bytes = static_cast<uint32_t>(entry.act_scale.size());
                std::memcpy(
                        static_cast<uint8_t *>(cma_base) + entry.act_scale_cma_offset,
                        entry.act_scale.data(),
                        entry.act_scale.size());
                cma_cursor = act_scale_cursor + entry.act_scale_cma_bytes;
            }
        }
        if (!entry.pingpong_scale.empty() && entry.pingpong_scale_cma_bytes != entry.pingpong_layout.scale_bytes) {
            const uint32_t scale_cursor = npu_decode_align_up_bytes(cma_cursor, NPU_DECODE_GEMV_DMA_ALIGN);
            if (entry.pingpong_scale.size() <= static_cast<size_t>(std::numeric_limits<uint32_t>::max()) &&
                    scale_cursor <= cma_limit_bytes &&
                    static_cast<uint32_t>(entry.pingpong_scale.size()) <= cma_limit_bytes - scale_cursor) {
                entry.pingpong_scale_cma_offset = cma_base_offset + scale_cursor;
                entry.pingpong_scale_cma_bytes = static_cast<uint32_t>(entry.pingpong_scale.size());
                std::memcpy(
                        static_cast<uint8_t *>(cma_base) + entry.pingpong_scale_cma_offset,
                        entry.pingpong_scale.data(),
                        entry.pingpong_scale.size());
                cma_cursor = scale_cursor + entry.pingpong_scale_cma_bytes;
            }
        }

        if (entry.pingpong_weight_cma_bytes != entry.pingpong_layout.weight_bytes) {
            const uint32_t weight_cursor = npu_decode_align_up_bytes(cma_cursor, NPU_DECODE_GEMV_DMA_ALIGN);
            if (entry.pingpong_weight.size() <= static_cast<size_t>(std::numeric_limits<uint32_t>::max()) &&
                    weight_cursor <= cma_limit_bytes &&
                    static_cast<uint32_t>(entry.pingpong_weight.size()) <= cma_limit_bytes - weight_cursor) {
                entry.pingpong_weight_cma_offset = cma_base_offset + weight_cursor;
                entry.pingpong_weight_cma_bytes = static_cast<uint32_t>(entry.pingpong_weight.size());
                std::memcpy(
                        static_cast<uint8_t *>(cma_base) + entry.pingpong_weight_cma_offset,
                        entry.pingpong_weight.data(),
                        entry.pingpong_weight.size());
                cma_cursor = weight_cursor + entry.pingpong_weight_cma_bytes;
            }
        }

        if ((!entry.act_scale.empty() && entry.act_scale_cma_bytes == 0) ||
                (!entry.pingpong_scale.empty() && entry.pingpong_scale_cma_bytes == 0) ||
                entry.pingpong_weight_cma_bytes == 0) {
            entry.act_scale_cma_offset = 0;
            entry.act_scale_cma_bytes = 0;
            entry.pingpong_scale_cma_offset = 0;
            entry.pingpong_scale_cma_bytes = 0;
            entry.pingpong_weight_cma_offset = 0;
            entry.pingpong_weight_cma_bytes = 0;
            cma_cursor = pingpong_cma_start;
        }
    };

    if (!entry.prefer_block_cma ||
            (entry.pingpong_available && npu_decode_fused_ffn_pingpong_enabled())) {
        reserve_pingpong_cma();
    }

    const bool reserve_cma_for_pingpong =
        !entry.prefer_block_cma &&
        entry.act_scale_cma_bytes == entry.pingpong_layout.act_bytes &&
        entry.pingpong_scale_cma_bytes == entry.pingpong_layout.scale_bytes &&
        entry.pingpong_weight_cma_bytes != 0;

    for (int64_t row_base = 0; row_base < out_channels; row_base += entry.max_block_rows) {
        const uint32_t block_rows = static_cast<uint32_t>(std::min<int64_t>(entry.max_block_rows, out_channels - row_base));
        npu_decode_preloaded_block block;
        block.row_base = row_base;
        block.block_rows = block_rows;
        if (!npu_decode_make_gemv_layout(block_rows, static_cast<uint32_t>(k), &block.layout)) {
            return fail("block layout failed");
        }
        block.packed.resize(block.layout.packed_bytes);
        npu_decode_pack_scale_weight_block(
                block.packed,
                block.layout,
                q4_data,
                packed_k,
                q4_nb1,
                scale_data,
                scale_type,
                scale_nb0,
                scale_nb1,
                row_base,
                block_rows,
                k);
        if (!reserve_cma_for_pingpong && cma_base != nullptr && cma_limit_bytes != 0) {
            const uint32_t aligned_cursor = npu_decode_align_up_bytes(cma_cursor, NPU_DECODE_GEMV_DMA_ALIGN);
            if (block.packed.size() <= static_cast<size_t>(std::numeric_limits<uint32_t>::max()) &&
                    aligned_cursor <= cma_limit_bytes &&
                    static_cast<uint32_t>(block.packed.size()) <= cma_limit_bytes - aligned_cursor) {
                block.cma_offset = cma_base_offset + aligned_cursor;
                block.cma_bytes = static_cast<uint32_t>(block.packed.size());
                std::memcpy(static_cast<uint8_t *>(cma_base) + block.cma_offset, block.packed.data(), block.packed.size());
                cma_cursor = aligned_cursor + block.cma_bytes;
            }
        }
        entry.blocks.push_back(std::move(block));
    }

    if (entry.prefer_block_cma &&
            (entry.act_scale_cma_bytes != entry.pingpong_layout.act_bytes ||
             entry.pingpong_scale_cma_bytes != entry.pingpong_layout.scale_bytes ||
             entry.pingpong_weight_cma_bytes != entry.pingpong_layout.weight_bytes)) {
        reserve_pingpong_cma();
    }

    npu_decode_preload_cache & cache = npu_decode_get_preload_cache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    if (cma_base != nullptr && cma_limit_bytes != 0) {
        cache.cma_base_offset = cma_base_offset;
        cache.cma_limit_bytes = cma_limit_bytes;
        cache.cma_cursor = cma_cursor;
    }
    cache.entries[weight_name] = std::move(entry);
    return true;
}

bool ggml_backend_npu_decode_w4a16_gemv_ex(
        const char * op_name,
        const void * q4_data,
        int64_t packed_k,
        int64_t out_channels,
        int64_t q4_nb1,
        const void * scale_data,
        int scale_type,
        int64_t scale_nb0,
        int64_t scale_nb1,
        const void * zero_data,
        int zero_type,
        int64_t zero_nb0,
        int64_t zero_nb1,
        const void * act_data,
        int act_type,
        int64_t act_nb0,
        int64_t act_nb1,
        const float * smooth_scale,
        int64_t k,
        int64_t n_cols,
        void * dst_data,
        int dst_type,
        int64_t dst_nb1) {
    using namespace ggml_npu;

    const bool collect_profile = npu_decode_profile_enabled();
    const int64_t total_start_us = collect_profile ? ggml_time_us() : 0;
    const int64_t setup_start_us = collect_profile ? total_start_us : 0;
    int64_t setup_stage_start_us = setup_start_us;
    npu_decode_gemv_profile_record profile = {};
    profile.op_name = op_name != nullptr ? op_name : "";
    profile.dst_type = dst_type;
    profile.k = k;
    profile.n_cols = n_cols;
    profile.out_channels = out_channels;

    auto fail = [op_name](const char * reason) {
        if (npu_decode_w4a16_debug_log_enabled()) {
            GGML_LOG_WARN("%s: reject %s: %s\n", __func__, op_name != nullptr ? op_name : "(unnamed)", reason);
        }
        return false;
    };

    GGML_UNUSED(scale_nb0);
    GGML_UNUSED(scale_nb1);
    GGML_UNUSED(zero_nb0);
    GGML_UNUSED(zero_nb1);
    GGML_UNUSED(smooth_scale);

    if (q4_data == nullptr || scale_data == nullptr || zero_data == nullptr || act_data == nullptr ||
            smooth_scale == nullptr || dst_data == nullptr) {
        return fail("null pointer");
    }
    if (k <= 0 || n_cols <= 0 || out_channels <= 0 || packed_k != (k + 1) / 2) {
        return fail("invalid dimensions");
    }
    if (k > std::numeric_limits<uint16_t>::max() || out_channels > std::numeric_limits<uint16_t>::max()) {
        return fail("dimension exceeds uint16 range");
    }
    if (scale_type != GGML_TYPE_F16 && scale_type != GGML_TYPE_F32) {
        return fail("scale tensor must be F16 or F32");
    }
    if (zero_type != GGML_TYPE_F16 && zero_type != GGML_TYPE_F32) {
        return fail("zero tensor must be F16 or F32");
    }
    if (act_type != GGML_TYPE_F16 && act_type != GGML_TYPE_F32) {
        return fail("activation tensor must be F16 or F32");
    }
    if (dst_type != GGML_TYPE_F16 && dst_type != GGML_TYPE_F32) {
        return fail("destination tensor must be F16 or F32");
    }
    const bool hw_f32_mvout = dst_type == GGML_TYPE_F32 && npu_decode_hw_f32_mvout_enabled();
    const uint8_t output_precision = npu_decode_mvout_precision(hw_f32_mvout);

    if (collect_profile) {
        profile.validation_us = 0;
        profile.layout_us = 0;
        setup_stage_start_us = ggml_time_us();
    }

    const npu_decode_preloaded_tensor * preloaded = npu_decode_lookup_preloaded_tensor(
            op_name,
            packed_k,
            out_channels,
            q4_nb1,
            k);
    if (preloaded == nullptr || !preloaded->zero_validated || preloaded->blocks.empty()) {
        return fail("decode tensor is not preloaded");
    }
    const uint32_t max_block_rows = preloaded->max_block_rows;
    const npu_decode_gemv_layout first_layout = preloaded->max_layout;
    const uint32_t output_spm_bytes = npu_decode_align_up_u32(
            static_cast<uint32_t>(out_channels) * static_cast<uint32_t>(sizeof(ggml_fp16_t)),
            NPU_DECODE_GEMV_LINE_BYTES);
    const uint32_t host_output_bytes = npu_decode_host_output_bytes(static_cast<uint32_t>(out_channels), hw_f32_mvout);
    if (output_spm_bytes > NPU_DECODE_GEMV_SPM_BYTES) {
        return fail("decode output exceeds output SPM capacity");
    }
    bool preloaded_cma = preloaded != nullptr && !preloaded->blocks.empty();
    if (preloaded_cma) {
        for (const npu_decode_preloaded_block & block : preloaded->blocks) {
            if (block.cma_bytes == 0 || block.cma_bytes != block.packed.size()) {
                preloaded_cma = false;
                break;
            }
        }
    }
    const bool pingpong_requested =
        npu_decode_pingpong_enabled() &&
        preloaded->pingpong_available &&
        preloaded->pingpong_scale.size() == preloaded->pingpong_layout.scale_bytes &&
        preloaded->pingpong_weight.size() == preloaded->pingpong_layout.weight_bytes;
    bool pingpong_cma =
        pingpong_requested &&
        npu_decode_stream_w4_cma_resident(preloaded);
    if (collect_profile) {
        profile.preload_lookup_us = static_cast<uint64_t>(ggml_time_us() - setup_stage_start_us);
        setup_stage_start_us = ggml_time_us();
    }
    if (!npu_decode_ensure_overlay_active(op_name)) {
        return fail("decode overlay switch failed");
    }
    if (collect_profile) {
        profile.overlay_ensure_us = static_cast<uint64_t>(ggml_time_us() - setup_stage_start_us);
        setup_stage_start_us = ggml_time_us();
    }
    if (preloaded_cma || pingpong_cma) {
        npu_decode_update_runtime_heap_env();
    }
    if (collect_profile) {
        profile.heap_update_us = static_cast<uint64_t>(ggml_time_us() - setup_stage_start_us);
        setup_stage_start_us = ggml_time_us();
    }
    if (npu_decode_init_with_cma_retry(op_name, true) != 0) {
        return fail("decode runtime init failed");
    }
    if (pingpong_requested && !pingpong_cma) {
        (void) npu_decode_reserve_preloaded_ffn_pingpong_cma_after_init({
            const_cast<npu_decode_preloaded_tensor *>(preloaded),
        });
    }
    preloaded_cma = preloaded != nullptr && !preloaded->blocks.empty();
    if (preloaded_cma) {
        for (const npu_decode_preloaded_block & block : preloaded->blocks) {
            if (block.cma_bytes == 0 || block.cma_bytes != block.packed.size()) {
                preloaded_cma = false;
                break;
            }
        }
    }
    pingpong_cma =
        pingpong_requested &&
        npu_decode_stream_w4_cma_resident(preloaded);
    if (collect_profile) {
        profile.runtime_init_us = static_cast<uint64_t>(ggml_time_us() - setup_stage_start_us);
        setup_stage_start_us = ggml_time_us();
    }
    void * decode_cma_base = (preloaded_cma || pingpong_cma) ? npu_decode_memory_base() : nullptr;
    if (preloaded_cma && decode_cma_base == nullptr) {
        preloaded_cma = false;
    }
    if (pingpong_cma && decode_cma_base == nullptr) {
        pingpong_cma = false;
    }
    if (collect_profile) {
        profile.memory_base_us = static_cast<uint64_t>(ggml_time_us() - setup_stage_start_us);
    }

    if (collect_profile) {
        profile.max_block_rows = max_block_rows;
        profile.row_blocks_per_col = npu_decode_ceil_div_u32(static_cast<uint32_t>(out_channels), max_block_rows);
        profile.preloaded_host = preloaded != nullptr;
        profile.preloaded_cma = preloaded_cma || pingpong_cma;
        profile.pingpong = pingpong_requested;
        profile.activation_bytes = static_cast<uint64_t>(npu_decode_act_payload_bytes(first_layout)) * static_cast<uint64_t>(n_cols);
        profile.setup_us = static_cast<uint64_t>(ggml_time_us() - setup_start_us);
    }

    const int64_t alloc_start_us = collect_profile ? ggml_time_us() : 0;
    void * act_cma = npu_decode_mem_alloc(npu_decode_act_payload_bytes(first_layout));
    void * packed_cma = preloaded_cma ? nullptr : npu_decode_mem_alloc(first_layout.packed_bytes);
    void * out_cma = npu_decode_mem_alloc(host_output_bytes);
    void * pingpong_scale_cma = nullptr;
    void * pingpong_weight_cma = nullptr;
    if (pingpong_requested && !pingpong_cma) {
        if (preloaded->pingpong_layout.scale_bytes > 0) {
            pingpong_scale_cma = npu_decode_mem_alloc(preloaded->pingpong_layout.scale_bytes);
        }
        pingpong_weight_cma = npu_decode_mem_alloc(preloaded->pingpong_layout.weight_bytes);
    }
    if (act_cma == nullptr || (!preloaded_cma && packed_cma == nullptr) || out_cma == nullptr ||
            (pingpong_requested && !pingpong_cma &&
             ((preloaded->pingpong_layout.scale_bytes > 0 && pingpong_scale_cma == nullptr) ||
              pingpong_weight_cma == nullptr))) {
        npu_decode_mem_free(act_cma);
        npu_decode_mem_free(packed_cma);
        npu_decode_mem_free(out_cma);
        npu_decode_mem_free(pingpong_scale_cma);
        npu_decode_mem_free(pingpong_weight_cma);
        return fail("decode CMA allocation failed");
    }
    if (pingpong_requested && !pingpong_cma) {
        if (preloaded->pingpong_layout.scale_bytes > 0) {
            std::memcpy(pingpong_scale_cma, preloaded->pingpong_scale.data(), preloaded->pingpong_layout.scale_bytes);
        }
        std::memcpy(pingpong_weight_cma, preloaded->pingpong_weight.data(), preloaded->pingpong_layout.weight_bytes);
    }

    if (collect_profile) {
        profile.alloc_us = static_cast<uint64_t>(ggml_time_us() - alloc_start_us);
    }

    bool transient_scratch = false;
    uint32_t transient_cursor = 0;
    uint32_t transient_limit = 0;
    auto transient_alloc = [&](size_t bytes) -> void * {
        if (decode_cma_base == nullptr || bytes == 0 ||
                bytes > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
            return nullptr;
        }
        uint32_t offset = 0;
        uint32_t size = 0;
        if (!npu_decode_get_transient_scratch(&offset, &size)) {
            return nullptr;
        }
        if (!transient_scratch) {
            transient_cursor = offset;
            transient_limit = offset + size;
            transient_scratch = true;
        }
        const uint32_t aligned = npu_decode_align_up_bytes(transient_cursor, NPU_DECODE_GEMV_DMA_ALIGN);
        const uint32_t nbytes = static_cast<uint32_t>(bytes);
        if (aligned < offset || aligned > transient_limit || nbytes > transient_limit - aligned) {
            return nullptr;
        }
        transient_cursor = aligned + nbytes;
        return static_cast<uint8_t *>(decode_cma_base) + aligned;
    };

    bool act_transient = false;
    bool packed_transient = false;
    bool out_transient = false;
    bool pingpong_scale_transient = false;
    bool pingpong_weight_transient = false;
    auto free_decode_alloc = [](void * ptr, bool transient) {
        if (ptr != nullptr && !transient) {
            npu_decode_mem_free(ptr);
        }
    };
    if (npu_decode_transient_scratch_enabled() && (preloaded_cma || pingpong_cma) && decode_cma_base != nullptr) {
        uint32_t transient_base_offset = 0;
        uint32_t transient_size = 0;
        const bool have_transient = npu_decode_get_transient_scratch(&transient_base_offset, &transient_size);
        void * transient_act = transient_alloc(npu_decode_act_payload_bytes(first_layout));
        void * transient_packed = preloaded_cma ? nullptr : transient_alloc(first_layout.packed_bytes);
        void * transient_out = transient_alloc(host_output_bytes);
        void * transient_pingpong_scale = nullptr;
        void * transient_pingpong_weight = nullptr;
        if (pingpong_requested && !pingpong_cma) {
            if (preloaded->pingpong_layout.scale_bytes > 0) {
                transient_pingpong_scale = transient_alloc(preloaded->pingpong_layout.scale_bytes);
            }
            transient_pingpong_weight = transient_alloc(preloaded->pingpong_layout.weight_bytes);
        }
        if (transient_act != nullptr && (preloaded_cma || transient_packed != nullptr) && transient_out != nullptr &&
                (!pingpong_requested || pingpong_cma ||
                 (transient_pingpong_scale != nullptr &&
                  transient_pingpong_weight != nullptr))) {
            npu_decode_mem_free(act_cma);
            npu_decode_mem_free(packed_cma);
            npu_decode_mem_free(out_cma);
            npu_decode_mem_free(pingpong_scale_cma);
            npu_decode_mem_free(pingpong_weight_cma);
            act_cma = transient_act;
            packed_cma = transient_packed;
            out_cma = transient_out;
            pingpong_scale_cma = transient_pingpong_scale;
            pingpong_weight_cma = transient_pingpong_weight;
            act_transient = true;
            packed_transient = transient_packed != nullptr;
            out_transient = true;
            pingpong_scale_transient = transient_pingpong_scale != nullptr;
            pingpong_weight_transient = transient_pingpong_weight != nullptr;
            if (pingpong_requested && !pingpong_cma) {
                if (preloaded->pingpong_layout.scale_bytes > 0) {
                    std::memcpy(pingpong_scale_cma, preloaded->pingpong_scale.data(), preloaded->pingpong_layout.scale_bytes);
                }
                std::memcpy(pingpong_weight_cma, preloaded->pingpong_weight.data(), preloaded->pingpong_layout.weight_bytes);
            }
            if (npu_decode_w4a16_debug_log_enabled()) {
                GGML_LOG_WARN("%s: using decode transient scratch for %s offset=0x%X used=%u limit=0x%X\n",
                        __func__,
                        op_name != nullptr ? op_name : "(unnamed)",
                        have_transient ? static_cast<unsigned>(transient_base_offset) : 0u,
                        have_transient ? static_cast<unsigned>(transient_cursor - transient_base_offset) : 0u,
                        static_cast<unsigned>(transient_limit));
            }
        }
    }

    if (npu_decode_stream_api_enabled() && pingpong_cma && decode_cma_base != nullptr) {
        void * act_scale_ptr = npu_decode_stream_cma_ptr(decode_cma_base, preloaded->act_scale_cma_offset);
        void * weight_scale_ptr = npu_decode_stream_cma_ptr(decode_cma_base, preloaded->pingpong_scale_cma_offset);
        void * weight_payload_ptr = npu_decode_stream_cma_ptr(decode_cma_base, preloaded->pingpong_weight_cma_offset);

        bool stream_ok = true;
        for (int64_t col = 0; col < n_cols && stream_ok; ++col) {
            int64_t stage_start_us = collect_profile ? ggml_time_us() : 0;
            npu_decode_pack_activation_slice(
                    act_cma,
                    preloaded->pingpong_layout.act_bytes,
                    preloaded->pingpong_layout,
                    act_data,
                    act_type,
                    act_nb0,
                    act_nb1,
                    k,
                    0,
                    k,
                    col);
            if (collect_profile) {
                profile.activation_preprocess_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
                stage_start_us = ggml_time_us();
            }

            npu_decode_reset();
            if (collect_profile) {
                profile.reset_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
                stage_start_us = ggml_time_us();
            }
            if (npu_decode_zero_out_before_mvout_enabled()) {
                std::memset(out_cma, 0, host_output_bytes);
            }

            kv260_stream_gemv_desc desc = {};
            desc.act_ptr = act_cma;
            desc.act_scale_ptr = act_scale_ptr;
            desc.weight_payload_ptr = weight_payload_ptr;
            desc.weight_scale_ptr = weight_scale_ptr;
            desc.output_ptr = out_cma;
            desc.m = static_cast<uint16_t>(out_channels);
            desc.n = static_cast<uint16_t>(k);
            desc.mode = KV260_STREAM_DECODE_GEMV_W4A16;
            desc.output_precision = npu_decode_stream_output_precision(hw_f32_mvout);
            desc.role = KV260_STREAM_GEMV_ROLE_LINEAR;
            desc.dst = KV260_STREAM_GEMV_DST_OUTPUT;
            desc.post_op = KV260_STREAM_POST_BYPASS;
            desc.flags = KV260_STREAM_GEMV_F_ENABLE_ACT_SCALE;
            const int rc = npu_decode_stream_gemv(op_name, desc);
            if (rc != 0) {
                if (npu_decode_w4a16_debug_log_enabled()) {
                    GGML_LOG_WARN("%s: stream GEMV failed for %s rc=%d\n",
                            __func__, op_name != nullptr ? op_name : "(unnamed)", rc);
                }
                stream_ok = false;
                break;
            }
            if (collect_profile) {
                profile.pingpong_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
                profile.pingpong_calls += 1;
                profile.output_bytes += host_output_bytes;
                stage_start_us = ggml_time_us();
            }
            std::vector<uint8_t> repro_act_payload;
            const void * repro_act_ptr = act_cma;
            if (npu_decode_repro_dump_match(op_name)) {
                repro_act_payload.assign(npu_decode_act_payload_bytes(preloaded->pingpong_layout), 0);
                std::memcpy(
                        repro_act_payload.data(),
                        act_cma,
                        preloaded->pingpong_layout.act_bytes);
                std::memcpy(
                        repro_act_payload.data() + preloaded->pingpong_layout.act_bytes,
                        act_scale_ptr,
                        preloaded->pingpong_layout.act_bytes);
                repro_act_ptr = repro_act_payload.data();
            }
            npu_decode_repro_dump_pingpong(
                    op_name,
                    col,
                    preloaded->pingpong_layout,
                    static_cast<uint16_t>(out_channels),
                    static_cast<uint16_t>(k),
                    output_precision,
                    false,
                    repro_act_ptr,
                    weight_scale_ptr,
                    weight_payload_ptr,
                    out_cma,
                    host_output_bytes,
                    decode_cma_base,
                    npu_decode_memory_size());

            char * dst_col = reinterpret_cast<char *>(dst_data) + col * dst_nb1;
            if (dst_type == GGML_TYPE_F16) {
                std::memcpy(dst_col, out_cma, static_cast<size_t>(out_channels) * sizeof(ggml_fp16_t));
            } else if (hw_f32_mvout) {
                std::memcpy(dst_col, out_cma, static_cast<size_t>(out_channels) * sizeof(float));
            } else {
                npu_decode_convert_output_fp16_to_f32(
                        static_cast<const ggml_fp16_t *>(out_cma),
                        reinterpret_cast<float *>(dst_col),
                        static_cast<uint32_t>(out_channels));
            }
            if (collect_profile) {
                profile.postprocess_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            }
        }
        if (stream_ok) {
            const int64_t cleanup_start_us = collect_profile ? ggml_time_us() : 0;
            free_decode_alloc(act_cma, act_transient);
            free_decode_alloc(packed_cma, packed_transient);
            free_decode_alloc(out_cma, out_transient);
            free_decode_alloc(pingpong_scale_cma, pingpong_scale_transient);
            free_decode_alloc(pingpong_weight_cma, pingpong_weight_transient);
            if (collect_profile) {
                profile.cleanup_us = static_cast<uint64_t>(ggml_time_us() - cleanup_start_us);
                profile.total_us = static_cast<uint64_t>(ggml_time_us() - total_start_us);
                npu_decode_profile_write(profile);
            }
            return true;
        }
    }

    if (npu_decode_stream_api_enabled()) {
        free_decode_alloc(act_cma, act_transient);
        free_decode_alloc(packed_cma, packed_transient);
        free_decode_alloc(out_cma, out_transient);
        free_decode_alloc(pingpong_scale_cma, pingpong_scale_transient);
        free_decode_alloc(pingpong_weight_cma, pingpong_weight_transient);
        return fail("stream GEMV failed or assets are not CMA resident; split/legacy NPU path disabled");
    }

    if (pingpong_requested) {
        void * pingpong_scale_ptr = pingpong_cma ?
            static_cast<uint8_t *>(decode_cma_base) + preloaded->pingpong_scale_cma_offset :
            pingpong_scale_cma;
        void * pingpong_weight_ptr = pingpong_cma ?
            static_cast<uint8_t *>(decode_cma_base) + preloaded->pingpong_weight_cma_offset :
            pingpong_weight_cma;

        bool pingpong_ok = true;
        for (int64_t col = 0; col < n_cols && pingpong_ok; ++col) {
            int64_t stage_start_us = collect_profile ? ggml_time_us() : 0;
            npu_decode_pack_activation(
                    act_cma,
                    npu_decode_act_payload_bytes(preloaded->pingpong_layout),
                    preloaded->pingpong_layout,
                    act_data,
                    act_type,
                    act_nb0,
                    act_nb1,
                    preloaded->act_scale.empty() ? nullptr : preloaded->act_scale.data(),
                    k,
                    col);
            if (collect_profile) {
                profile.activation_preprocess_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
                stage_start_us = ggml_time_us();
            }

            npu_decode_reset();
            if (collect_profile) {
                profile.reset_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
                stage_start_us = ggml_time_us();
            }

            if (npu_decode_zero_out_before_mvout_enabled()) {
                std::memset(out_cma, 0, host_output_bytes);
            }
            try {
                npu_decode_gemv_pingpong_manual_run(
                        op_name,
                        act_cma,
                        pingpong_scale_ptr,
                        pingpong_weight_ptr,
                        out_cma,
                        preloaded->pingpong_layout,
                        static_cast<uint16_t>(out_channels),
                        static_cast<uint16_t>(k),
                        output_precision,
                        collect_profile ? &profile : nullptr);
            } catch (const std::exception & e) {
                if (npu_decode_w4a16_debug_log_enabled()) {
                    GGML_LOG_WARN("%s: pingpong failed for %s: %s\n",
                            __func__, op_name != nullptr ? op_name : "(unnamed)", e.what());
                }
                pingpong_ok = false;
                break;
            }
            if (collect_profile) {
                profile.pingpong_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
                profile.pingpong_calls += 1;
                profile.output_bytes += host_output_bytes;
                stage_start_us = ggml_time_us();
            }
            npu_decode_repro_dump_pingpong(
                    op_name,
                    col,
                    preloaded->pingpong_layout,
                    static_cast<uint16_t>(out_channels),
                    static_cast<uint16_t>(k),
                    output_precision,
                    npu_decode_pingpong_block_mvout_enabled(),
                    act_cma,
                    pingpong_scale_ptr,
                    pingpong_weight_ptr,
                    out_cma,
                    host_output_bytes,
                    decode_cma_base,
                    npu_decode_memory_size());

            char * dst_col = reinterpret_cast<char *>(dst_data) + col * dst_nb1;
            if (dst_type == GGML_TYPE_F16) {
                std::memcpy(dst_col, out_cma, static_cast<size_t>(out_channels) * sizeof(ggml_fp16_t));
            } else if (hw_f32_mvout) {
                std::memcpy(dst_col, out_cma, static_cast<size_t>(out_channels) * sizeof(float));
            } else {
                npu_decode_convert_output_fp16_to_f32(
                        static_cast<const ggml_fp16_t *>(out_cma),
                        reinterpret_cast<float *>(dst_col),
                        static_cast<uint32_t>(out_channels));
            }
            if (collect_profile) {
                profile.postprocess_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            }
        }

        if (pingpong_ok) {
            const int64_t cleanup_start_us = collect_profile ? ggml_time_us() : 0;
            free_decode_alloc(act_cma, act_transient);
            free_decode_alloc(packed_cma, packed_transient);
            free_decode_alloc(out_cma, out_transient);
            free_decode_alloc(pingpong_scale_cma, pingpong_scale_transient);
            free_decode_alloc(pingpong_weight_cma, pingpong_weight_transient);
            if (collect_profile) {
                profile.cleanup_us = static_cast<uint64_t>(ggml_time_us() - cleanup_start_us);
                profile.total_us = static_cast<uint64_t>(ggml_time_us() - total_start_us);
                npu_decode_profile_write(profile);
            }
            return true;
        }

        free_decode_alloc(pingpong_scale_cma, pingpong_scale_transient);
        free_decode_alloc(pingpong_weight_cma, pingpong_weight_transient);
        pingpong_scale_cma = nullptr;
        pingpong_weight_cma = nullptr;
        pingpong_scale_transient = false;
        pingpong_weight_transient = false;
    }

    for (int64_t col = 0; col < n_cols; ++col) {
        int64_t stage_start_us = collect_profile ? ggml_time_us() : 0;
        npu_decode_pack_activation(
                act_cma,
                npu_decode_act_payload_bytes(first_layout),
                first_layout,
                act_data,
                act_type,
                act_nb0,
                act_nb1,
                preloaded->act_scale.empty() ? nullptr : preloaded->act_scale.data(),
                k,
                col);
        if (collect_profile) {
            profile.activation_preprocess_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            stage_start_us = ggml_time_us();
        }

        npu_decode_reset();
        if (collect_profile) {
            profile.reset_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            stage_start_us = ggml_time_us();
        }
        npu_decode_stream_unsupported_void(
                act_cma,
                0,
                npu_decode_act_payload_bytes(first_layout) - 1u,
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

        size_t block_index = 0;
        for (const npu_decode_preloaded_block & block : preloaded->blocks) {
            const int64_t row_base = block.row_base;
            const uint32_t block_rows = block.block_rows;
            const npu_decode_gemv_layout & layout = block.layout;
            if (layout.packed_bytes > first_layout.packed_bytes || layout.output_bytes > first_layout.output_bytes) {
                free_decode_alloc(act_cma, act_transient);
                free_decode_alloc(packed_cma, packed_transient);
                free_decode_alloc(out_cma, out_transient);
                return fail("block buffer exceeds allocation");
            }

            void * weight_mvin_ptr = packed_cma;
            if (preloaded_cma &&
                    block_index < preloaded->blocks.size() &&
                    block.cma_bytes == layout.packed_bytes) {
                weight_mvin_ptr = static_cast<uint8_t *>(decode_cma_base) + block.cma_offset;
            } else if (!preloaded_cma &&
                    preloaded != nullptr &&
                    block_index < preloaded->blocks.size() &&
                    block.packed.size() == layout.packed_bytes) {
                stage_start_us = collect_profile ? ggml_time_us() : 0;
                std::memcpy(packed_cma, block.packed.data(), layout.packed_bytes);
                if (collect_profile) {
                    profile.host_copy_weight_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
                }
            } else {
                free_decode_alloc(act_cma, act_transient);
                free_decode_alloc(out_cma, out_transient);
                return fail("preloaded CMA block mismatch");
            }
            if (collect_profile) {
                profile.weight_bytes += layout.packed_bytes;
                stage_start_us = ggml_time_us();
            }

            npu_decode_stream_unsupported_void(
                    weight_mvin_ptr,
                    0,
                    layout.packed_bytes - 1u,
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
                stage_start_us = ggml_time_us();
            }

            const uint64_t flow = npu_decode_make_bypass_output_flow(static_cast<uint16_t>(block_rows));
            npu_decode_trace_stream_flow(
                    "w4_preloaded_block",
                    op_name,
                    static_cast<uint32_t>(block_index),
                    static_cast<uint32_t>(row_base),
                    static_cast<uint32_t>(block_rows),
                    static_cast<uint32_t>(k),
                    NPU_DECODE_GEMV_MODE_W4A16,
                    layout.weight_base,
                    NPU_DECODE_GEMV_ACT_BASE,
                    static_cast<uint16_t>(row_base * sizeof(ggml_fp16_t)),
                    NPU_DECODE_GEMV_SCALE_BASE,
                    flow);
            npu_decode_stream_unsupported_void(
                    op_name,
                    layout.weight_base,
                    NPU_DECODE_GEMV_ACT_BASE,
                    static_cast<uint16_t>(k),
                    static_cast<uint16_t>(block_rows),
                    static_cast<uint16_t>(row_base * sizeof(ggml_fp16_t)),
                    NPU_DECODE_GEMV_SCALE_BASE,
                    NPU_DECODE_GEMV_MODE_W4A16,
                    flow);
            if (collect_profile) {
                profile.gemv_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
                profile.gemv_calls += 1;
                stage_start_us = ggml_time_us();
            }
            ++block_index;
        }

        stage_start_us = collect_profile ? ggml_time_us() : 0;
        npu_decode_stream_unsupported_void(
                out_cma,
                0,
                0,
                static_cast<uint32_t>(out_channels - 1),
                1,
                1,
                output_precision,
                1,
                false,
                false,
                0,
                0);
        if (collect_profile) {
            profile.mvout_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            profile.mvout_calls += 1;
            profile.output_bytes += host_output_bytes;
            stage_start_us = ggml_time_us();
        }

        char * dst_col = reinterpret_cast<char *>(dst_data) + col * dst_nb1;
        if (dst_type == GGML_TYPE_F16) {
            std::memcpy(dst_col, out_cma, static_cast<size_t>(out_channels) * sizeof(ggml_fp16_t));
        } else if (hw_f32_mvout) {
            std::memcpy(dst_col, out_cma, static_cast<size_t>(out_channels) * sizeof(float));
        } else {
            npu_decode_convert_output_fp16_to_f32(
                    static_cast<const ggml_fp16_t *>(out_cma),
                    reinterpret_cast<float *>(dst_col),
                    static_cast<uint32_t>(out_channels));
        }
        if (collect_profile) {
            profile.postprocess_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
        }
    }

    const int64_t cleanup_start_us = collect_profile ? ggml_time_us() : 0;
    free_decode_alloc(act_cma, act_transient);
    free_decode_alloc(packed_cma, packed_transient);
    free_decode_alloc(out_cma, out_transient);
    free_decode_alloc(pingpong_scale_cma, pingpong_scale_transient);
    free_decode_alloc(pingpong_weight_cma, pingpong_weight_transient);
    if (collect_profile) {
        profile.cleanup_us = static_cast<uint64_t>(ggml_time_us() - cleanup_start_us);
        profile.total_us = static_cast<uint64_t>(ggml_time_us() - total_start_us);
        npu_decode_profile_write(profile);
    }
    return true;
}

static bool npu_decode_w16a16_prepare_weight_cache(
        ggml_npu::npu_decode_w16a16_cached_weight & entry,
        const char * weight_name,
        const void * weight_data,
        int64_t k,
        int64_t out_channels,
        int64_t weight_nb0,
        int64_t weight_nb1,
        bool allocate_heap_cma) {
    using namespace ggml_npu;

    const uint32_t row_tiles = npu_decode_ceil_div_u32(static_cast<uint32_t>(out_channels), NPU_DECODE_GEMV_M_TILE);
    const uint32_t col_tiles = npu_decode_ceil_div_u32(static_cast<uint32_t>(k), NPU_DECODE_GEMV_K_TILE);
    const uint32_t bytes_per_row_tile = col_tiles * NPU_DECODE_GEMV_W16_WEIGHT_TILE_BYTES;
    const uint32_t weight_capacity = NPU_DECODE_GEMV_SPM_BYTES - NPU_DECODE_GEMV_PING_WEIGHT_BASE;
    const uint32_t max_row_tiles = bytes_per_row_tile == 0 ? 0 :
        weight_capacity / bytes_per_row_tile;
    if (row_tiles == 0 || col_tiles == 0 || max_row_tiles == 0) {
        return false;
    }

    const bool same_weight =
        entry.weight_data == weight_data ||
        (weight_name != nullptr &&
         weight_name[0] != '\0' &&
         !entry.weight_name.empty() &&
         entry.weight_name == weight_name);
    if (same_weight &&
            entry.k == k &&
            entry.out_channels == out_channels &&
            entry.weight_nb0 == weight_nb0 &&
            entry.weight_nb1 == weight_nb1 &&
            entry.row_tiles == row_tiles &&
            entry.col_tiles == col_tiles &&
            !entry.packed_weight.empty()) {
        entry.weight_data = weight_data;
        if (weight_name != nullptr && weight_name[0] != '\0') {
            entry.weight_name = weight_name;
        }
        if (allocate_heap_cma &&
                npu_decode_w16a16_cma_enabled() &&
                entry.cma_packed == nullptr &&
                entry.cma_packed_offset == 0 &&
                entry.packed_weight.size() <= static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
            entry.cma_packed = npu_decode_mem_alloc(entry.packed_weight.size());
            if (entry.cma_packed != nullptr) {
                entry.cma_bytes = static_cast<uint32_t>(entry.packed_weight.size());
                entry.cma_packed_owned = true;
                std::memcpy(entry.cma_packed, entry.packed_weight.data(), entry.packed_weight.size());
            }
        }
        return true;
    }

    const uint64_t packed_bytes_64 = static_cast<uint64_t>(row_tiles) * bytes_per_row_tile;
    if (packed_bytes_64 > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        return false;
    }

    npu_decode_w16a16_cached_weight next = {};
    next.weight_name = weight_name != nullptr ? weight_name : "";
    next.weight_data = weight_data;
    next.k = k;
    next.out_channels = out_channels;
    next.weight_nb0 = weight_nb0;
    next.weight_nb1 = weight_nb1;
    next.row_tiles = row_tiles;
    next.col_tiles = col_tiles;
    next.max_block_rows = max_row_tiles * NPU_DECODE_GEMV_M_TILE;
    next.pingpong_max_block_rows = npu_decode_w16a16_pingpong_max_block_rows(col_tiles);
    next.bytes_per_row_tile = bytes_per_row_tile;
    next.packed_weight.assign(static_cast<size_t>(packed_bytes_64), 0);

    const auto * weight = static_cast<const uint8_t *>(weight_data);
    for (uint32_t row_tile = 0; row_tile < row_tiles; ++row_tile) {
        for (uint32_t col_tile = 0; col_tile < col_tiles; ++col_tile) {
            const size_t tile_base = static_cast<size_t>(row_tile) * bytes_per_row_tile +
                static_cast<size_t>(col_tile) * NPU_DECODE_GEMV_W16_WEIGHT_TILE_BYTES;
            for (uint32_t row_lane = 0; row_lane < NPU_DECODE_GEMV_M_TILE; ++row_lane) {
                const int64_t row = static_cast<int64_t>(row_tile) * NPU_DECODE_GEMV_M_TILE + row_lane;
                if (row >= out_channels) {
                    continue;
                }
                uint8_t * dst = next.packed_weight.data() + tile_base +
                    static_cast<size_t>(row_lane) * NPU_DECODE_GEMV_W16_WEIGHT_ROW_BYTES;
                for (uint32_t col_lane = 0; col_lane < NPU_DECODE_GEMV_K_TILE; ++col_lane) {
                    const int64_t col = static_cast<int64_t>(col_tile) * NPU_DECODE_GEMV_K_TILE + col_lane;
                    if (col >= k) {
                        continue;
                    }
                    std::memcpy(
                            dst + col_lane * sizeof(ggml_fp16_t),
                            weight + row * weight_nb1 + col * weight_nb0,
                            sizeof(ggml_fp16_t));
                }
            }
        }
    }

    if (allocate_heap_cma &&
            npu_decode_w16a16_cma_enabled() &&
            packed_bytes_64 <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        next.cma_packed = npu_decode_mem_alloc(static_cast<size_t>(packed_bytes_64));
        if (next.cma_packed != nullptr) {
            next.cma_bytes = static_cast<uint32_t>(packed_bytes_64);
            next.cma_packed_owned = true;
            std::memcpy(next.cma_packed, next.packed_weight.data(), next.packed_weight.size());
        }
    }

    if (entry.cma_packed != nullptr && entry.cma_packed_owned) {
        npu_decode_mem_free(entry.cma_packed);
    }
    if (entry.cma_scale_ones != nullptr && entry.cma_scale_ones_owned) {
        npu_decode_mem_free(entry.cma_scale_ones);
    }
    entry = std::move(next);
    return true;
}

static void * npu_decode_w16a16_ensure_scale_ones_cma(
        ggml_npu::npu_decode_w16a16_cached_weight & entry,
        uint32_t max_scale_bytes) {
    using namespace ggml_npu;

    if (max_scale_bytes == 0) {
        return nullptr;
    }
    if (entry.cma_scale_ones != nullptr && entry.cma_scale_ones_bytes >= max_scale_bytes) {
        return entry.cma_scale_ones;
    }

    if (entry.cma_scale_ones != nullptr && entry.cma_scale_ones_owned) {
        npu_decode_mem_free(entry.cma_scale_ones);
    }
    entry.cma_scale_ones = nullptr;
    entry.cma_scale_ones_bytes = 0;
    entry.cma_scale_ones_owned = false;
    entry.scale_ones.assign(max_scale_bytes, 0);
    const ggml_fp16_t fp16_one = ggml_fp32_to_fp16(1.0f);
    for (uint32_t i = 0; i + sizeof(ggml_fp16_t) <= max_scale_bytes; i += sizeof(ggml_fp16_t)) {
        std::memcpy(entry.scale_ones.data() + i, &fp16_one, sizeof(fp16_one));
    }

    void * scale_cma = npu_decode_mem_alloc(max_scale_bytes);
    if (scale_cma == nullptr) {
        return nullptr;
    }
    std::memcpy(scale_cma, entry.scale_ones.data(), entry.scale_ones.size());
    entry.cma_scale_ones = scale_cma;
    entry.cma_scale_ones_bytes = max_scale_bytes;
    entry.cma_scale_ones_owned = true;
    return entry.cma_scale_ones;
}

static bool npu_decode_w16a16_reserve_static_weight_cma(
        ggml_npu::npu_decode_w16a16_cached_weight & entry,
        const char * weight_name) {
    using namespace ggml_npu;
    GGML_UNUSED(weight_name);

    if (!npu_decode_w16a16_cma_enabled() ||
            entry.packed_weight.empty() ||
            entry.packed_weight.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        return false;
    }
    if (entry.cma_packed != nullptr && entry.cma_bytes >= entry.packed_weight.size()) {
        return true;
    }
    if (entry.cma_packed != nullptr && entry.cma_packed_owned) {
        npu_decode_mem_free(entry.cma_packed);
    }
    entry.cma_packed = nullptr;
    entry.cma_bytes = 0;
    entry.cma_packed_owned = false;

    void * cma_base = nullptr;
    uint32_t cma_map_size = 0;
    uint32_t cma_base_offset = npu_decode_parse_size_env("NPU_CMA_HEAP_OFFSET", 0);
    uint32_t cma_limit_bytes = npu_decode_parse_size_env("NPU_CMA_HEAP_SIZE", 0);
    uint32_t cma_cursor = 0;
    {
        npu_decode_preload_cache & cache = npu_decode_get_preload_cache();
        std::lock_guard<std::mutex> lock(cache.mutex);
        if (cache.cma_limit_bytes != 0) {
            cma_base_offset = cache.cma_base_offset;
            cma_limit_bytes = cache.cma_limit_bytes;
            cma_cursor = cache.cma_cursor;
        }
    }

    if (npu_decode_cma_init() == 0) {
        cma_base = npu_decode_memory_base();
        cma_map_size = npu_decode_memory_size();
        if (cma_limit_bytes == 0 && cma_base_offset < cma_map_size) {
            cma_limit_bytes = cma_map_size - cma_base_offset;
        }
    }
    if (cma_base == nullptr ||
            cma_base_offset >= cma_map_size ||
            cma_limit_bytes > cma_map_size - cma_base_offset) {
        return false;
    }

    const uint32_t aligned_cursor = npu_decode_align_up_bytes(cma_cursor, NPU_DECODE_GEMV_DMA_ALIGN);
    const uint32_t packed_bytes = static_cast<uint32_t>(entry.packed_weight.size());
    if (aligned_cursor > cma_limit_bytes || packed_bytes > cma_limit_bytes - aligned_cursor) {
        return false;
    }

    entry.cma_packed = static_cast<uint8_t *>(cma_base) + cma_base_offset + aligned_cursor;
    entry.cma_packed_offset = cma_base_offset + aligned_cursor;
    entry.cma_bytes = packed_bytes;
    entry.cma_packed_owned = false;
    std::memcpy(entry.cma_packed, entry.packed_weight.data(), entry.packed_weight.size());

    {
        npu_decode_preload_cache & cache = npu_decode_get_preload_cache();
        std::lock_guard<std::mutex> lock(cache.mutex);
        cache.cma_base_offset = cma_base_offset;
        cache.cma_limit_bytes = cma_limit_bytes;
        cache.cma_cursor = aligned_cursor + packed_bytes;
        cache.transient_offset = 0;
        cache.transient_size = 0;
    }
    npu_decode_update_runtime_heap_env();
    return true;
}

static bool npu_decode_w16a16_reserve_static_scale_cma(
        ggml_npu::npu_decode_w16a16_cached_weight & entry,
        const char * weight_name,
        uint32_t max_scale_bytes) {
    using namespace ggml_npu;
    GGML_UNUSED(weight_name);

    if (max_scale_bytes == 0) {
        return false;
    }
    if (entry.cma_scale_ones != nullptr && entry.cma_scale_ones_bytes >= max_scale_bytes) {
        return true;
    }
    if (entry.cma_scale_ones != nullptr && entry.cma_scale_ones_owned) {
        npu_decode_mem_free(entry.cma_scale_ones);
    }
    entry.cma_scale_ones = nullptr;
    entry.cma_scale_ones_bytes = 0;
    entry.cma_scale_ones_owned = false;

    entry.scale_ones.assign(max_scale_bytes, 0);
    const ggml_fp16_t fp16_one = ggml_fp32_to_fp16(1.0f);
    for (uint32_t i = 0; i + sizeof(ggml_fp16_t) <= max_scale_bytes; i += sizeof(ggml_fp16_t)) {
        std::memcpy(entry.scale_ones.data() + i, &fp16_one, sizeof(fp16_one));
    }

    void * cma_base = nullptr;
    uint32_t cma_map_size = 0;
    uint32_t cma_base_offset = npu_decode_parse_size_env("NPU_CMA_HEAP_OFFSET", 0);
    uint32_t cma_limit_bytes = npu_decode_parse_size_env("NPU_CMA_HEAP_SIZE", 0);
    uint32_t cma_cursor = 0;
    {
        npu_decode_preload_cache & cache = npu_decode_get_preload_cache();
        std::lock_guard<std::mutex> lock(cache.mutex);
        if (cache.cma_limit_bytes != 0) {
            cma_base_offset = cache.cma_base_offset;
            cma_limit_bytes = cache.cma_limit_bytes;
            cma_cursor = cache.cma_cursor;
        }
    }

    if (npu_decode_cma_init() == 0) {
        cma_base = npu_decode_memory_base();
        cma_map_size = npu_decode_memory_size();
        if (cma_limit_bytes == 0 && cma_base_offset < cma_map_size) {
            cma_limit_bytes = cma_map_size - cma_base_offset;
        }
    }
    if (cma_base == nullptr ||
            cma_base_offset >= cma_map_size ||
            cma_limit_bytes > cma_map_size - cma_base_offset) {
        return false;
    }

    const uint32_t aligned_cursor = npu_decode_align_up_bytes(cma_cursor, NPU_DECODE_GEMV_DMA_ALIGN);
    if (aligned_cursor > cma_limit_bytes || max_scale_bytes > cma_limit_bytes - aligned_cursor) {
        return false;
    }
    entry.cma_scale_ones = static_cast<uint8_t *>(cma_base) + cma_base_offset + aligned_cursor;
    entry.cma_scale_ones_offset = cma_base_offset + aligned_cursor;
    entry.cma_scale_ones_bytes = max_scale_bytes;
    entry.cma_scale_ones_owned = false;
    std::memcpy(entry.cma_scale_ones, entry.scale_ones.data(), entry.scale_ones.size());

    {
        npu_decode_preload_cache & cache = npu_decode_get_preload_cache();
        std::lock_guard<std::mutex> lock(cache.mutex);
        cache.cma_base_offset = cma_base_offset;
        cache.cma_limit_bytes = cma_limit_bytes;
        cache.cma_cursor = aligned_cursor + max_scale_bytes;
        cache.transient_offset = 0;
        cache.transient_size = 0;
    }
    npu_decode_update_runtime_heap_env();
    return true;
}

static bool npu_decode_w8a16_reserve_static_scale_cma(
        ggml_npu::npu_decode_w16a16_cached_weight & entry) {
    using namespace ggml_npu;

    if (!npu_decode_w8a16_cma_enabled() ||
            entry.packed_scale.empty() ||
            entry.packed_scale.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        return false;
    }
    if (entry.cma_scale != nullptr && entry.cma_scale_bytes >= entry.packed_scale.size()) {
        return true;
    }
    if (entry.cma_scale != nullptr && entry.cma_scale_owned) {
        npu_decode_mem_free(entry.cma_scale);
    }
    entry.cma_scale = nullptr;
    entry.cma_scale_offset = 0;
    entry.cma_scale_bytes = 0;
    entry.cma_scale_owned = false;

    void * cma_base = nullptr;
    uint32_t cma_map_size = 0;
    uint32_t cma_base_offset = npu_decode_parse_size_env("NPU_CMA_HEAP_OFFSET", 0);
    uint32_t cma_limit_bytes = npu_decode_parse_size_env("NPU_CMA_HEAP_SIZE", 0);
    uint32_t cma_cursor = 0;
    {
        npu_decode_preload_cache & cache = npu_decode_get_preload_cache();
        std::lock_guard<std::mutex> lock(cache.mutex);
        if (cache.cma_limit_bytes != 0) {
            cma_base_offset = cache.cma_base_offset;
            cma_limit_bytes = cache.cma_limit_bytes;
            cma_cursor = cache.cma_cursor;
        }
    }

    if (npu_decode_cma_init() == 0) {
        cma_base = npu_decode_memory_base();
        cma_map_size = npu_decode_memory_size();
        if (cma_limit_bytes == 0 && cma_base_offset < cma_map_size) {
            cma_limit_bytes = cma_map_size - cma_base_offset;
        }
    }
    if (cma_base == nullptr ||
            cma_base_offset >= cma_map_size ||
            cma_limit_bytes > cma_map_size - cma_base_offset) {
        return false;
    }

    const uint32_t aligned_cursor = npu_decode_align_up_bytes(cma_cursor, NPU_DECODE_GEMV_DMA_ALIGN);
    const uint32_t scale_bytes = static_cast<uint32_t>(entry.packed_scale.size());
    if (aligned_cursor > cma_limit_bytes || scale_bytes > cma_limit_bytes - aligned_cursor) {
        return false;
    }

    entry.cma_scale = static_cast<uint8_t *>(cma_base) + cma_base_offset + aligned_cursor;
    entry.cma_scale_offset = cma_base_offset + aligned_cursor;
    entry.cma_scale_bytes = scale_bytes;
    entry.cma_scale_owned = false;
    std::memcpy(entry.cma_scale, entry.packed_scale.data(), entry.packed_scale.size());

    {
        npu_decode_preload_cache & cache = npu_decode_get_preload_cache();
        std::lock_guard<std::mutex> lock(cache.mutex);
        cache.cma_base_offset = cma_base_offset;
        cache.cma_limit_bytes = cma_limit_bytes;
        cache.cma_cursor = aligned_cursor + scale_bytes;
        cache.transient_offset = 0;
        cache.transient_size = 0;
    }
    npu_decode_update_runtime_heap_env();
    return true;
}

static void npu_decode_w16a16_rebind_static_cma(
        ggml_npu::npu_decode_w16a16_cached_weight & entry) {
    using namespace ggml_npu;

    void * cma_base = npu_decode_memory_base();
    const uint32_t cma_size = npu_decode_memory_size();
    if (cma_base == nullptr || cma_size == 0) {
        return;
    }
    if (entry.cma_packed == nullptr &&
            entry.cma_packed_offset != 0 &&
            entry.cma_bytes != 0 &&
            entry.cma_packed_offset < cma_size &&
            entry.cma_bytes <= cma_size - entry.cma_packed_offset) {
        entry.cma_packed = static_cast<uint8_t *>(cma_base) + entry.cma_packed_offset;
        entry.cma_packed_owned = false;
    }
    if (entry.cma_scale == nullptr &&
            entry.cma_scale_offset != 0 &&
            entry.cma_scale_bytes != 0 &&
            entry.cma_scale_offset < cma_size &&
            entry.cma_scale_bytes <= cma_size - entry.cma_scale_offset) {
        entry.cma_scale = static_cast<uint8_t *>(cma_base) + entry.cma_scale_offset;
        entry.cma_scale_owned = false;
    }
    if (entry.cma_scale_ones == nullptr &&
            entry.cma_scale_ones_offset != 0 &&
            entry.cma_scale_ones_bytes != 0 &&
            entry.cma_scale_ones_offset < cma_size &&
            entry.cma_scale_ones_bytes <= cma_size - entry.cma_scale_ones_offset) {
        entry.cma_scale_ones = static_cast<uint8_t *>(cma_base) + entry.cma_scale_ones_offset;
        entry.cma_scale_ones_owned = false;
    }
}

static bool npu_decode_w8a16_prepare_weight_cache(
        ggml_npu::npu_decode_w16a16_cached_weight & entry,
        const char * weight_name,
        const void * weight_data,
        int64_t k,
        int64_t out_channels,
        int64_t weight_nb0,
        int64_t weight_nb1,
        int64_t group,
        bool allocate_heap_cma) {
    using namespace ggml_npu;

    group = std::max<int64_t>(1, std::min<int64_t>(NPU_DECODE_GEMV_K_TILE, group));
    const uint32_t row_tiles = npu_decode_ceil_div_u32(static_cast<uint32_t>(out_channels), NPU_DECODE_GEMV_M_TILE);
    const uint32_t group_count = 1;
    const uint32_t col_tiles = npu_decode_ceil_div_u32(static_cast<uint32_t>(k), NPU_DECODE_GEMV_W8_K_TILE);
    const uint32_t bytes_per_row_tile = col_tiles * NPU_DECODE_GEMV_W8_WEIGHT_TILE_BYTES;
    const uint32_t scale_bytes_per_row_tile = col_tiles * NPU_DECODE_GEMV_SCALE_TILE_BYTES;
    const uint32_t ping_capacity = NPU_DECODE_GEMV_PONG_WEIGHT_BASE - NPU_DECODE_GEMV_PING_WEIGHT_BASE;
    const uint32_t pong_capacity = NPU_DECODE_GEMV_SPM_BYTES - NPU_DECODE_GEMV_PONG_WEIGHT_BASE;
    const uint32_t weight_capacity = std::min(ping_capacity, pong_capacity);
    const uint32_t max_row_tiles = bytes_per_row_tile == 0 ? 0 : weight_capacity / bytes_per_row_tile;
    if (row_tiles == 0 || group_count == 0 || col_tiles == 0 || max_row_tiles == 0) {
        return false;
    }

    const bool same_weight =
        entry.weight_data == weight_data ||
        (weight_name != nullptr &&
         weight_name[0] != '\0' &&
         !entry.weight_name.empty() &&
         entry.weight_name == weight_name);
    if (same_weight &&
            entry.k == k &&
            entry.out_channels == out_channels &&
            entry.weight_nb0 == weight_nb0 &&
            entry.weight_nb1 == weight_nb1 &&
            entry.group == group &&
            entry.group_count == static_cast<int64_t>(group_count) &&
            entry.row_tiles == row_tiles &&
            entry.col_tiles == col_tiles &&
            !entry.packed_scale.empty() &&
            !entry.packed_weight.empty()) {
        entry.weight_data = weight_data;
        if (weight_name != nullptr && weight_name[0] != '\0') {
            entry.weight_name = weight_name;
        }
        if (allocate_heap_cma && npu_decode_w8a16_cma_enabled()) {
            npu_decode_w16a16_rebind_static_cma(entry);
            if (entry.cma_packed == nullptr &&
                    entry.cma_packed_offset == 0 &&
                    entry.packed_weight.size() <= static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
                entry.cma_packed = npu_decode_mem_alloc(entry.packed_weight.size());
                if (entry.cma_packed != nullptr) {
                    entry.cma_bytes = static_cast<uint32_t>(entry.packed_weight.size());
                    entry.cma_packed_owned = true;
                    std::memcpy(entry.cma_packed, entry.packed_weight.data(), entry.packed_weight.size());
                }
            }
            if (entry.cma_scale == nullptr && entry.cma_scale_offset == 0) {
                (void) npu_decode_w8a16_reserve_static_scale_cma(entry);
            }
        }
        return true;
    }

    const uint64_t packed_bytes_64 =
        static_cast<uint64_t>(group_count) * static_cast<uint64_t>(row_tiles) * bytes_per_row_tile;
    const uint64_t scale_bytes_64 =
        static_cast<uint64_t>(group_count) * static_cast<uint64_t>(row_tiles) * scale_bytes_per_row_tile;
    if (packed_bytes_64 > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
            scale_bytes_64 > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        return false;
    }

    npu_decode_w16a16_cached_weight next = {};
    next.weight_name = weight_name != nullptr ? weight_name : "";
    next.weight_data = weight_data;
    next.k = k;
    next.out_channels = out_channels;
    next.weight_nb0 = weight_nb0;
    next.weight_nb1 = weight_nb1;
    next.group = group;
    next.group_count = group_count;
    next.row_tiles = row_tiles;
    next.col_tiles = col_tiles;
    next.max_block_rows = max_row_tiles * NPU_DECODE_GEMV_M_TILE;
    next.pingpong_max_block_rows = 0;
    next.bytes_per_row_tile = bytes_per_row_tile;
    next.packed_weight.assign(static_cast<size_t>(packed_bytes_64), 0);
    next.packed_scale.assign(static_cast<size_t>(scale_bytes_64), 0);

    const auto * weight = static_cast<const uint8_t *>(weight_data);
    for (uint32_t row_tile = 0; row_tile < row_tiles; ++row_tile) {
        for (uint32_t col_tile = 0; col_tile < col_tiles; ++col_tile) {
            const size_t tile_base =
                static_cast<size_t>(row_tile) * bytes_per_row_tile +
                static_cast<size_t>(col_tile) * NPU_DECODE_GEMV_W8_WEIGHT_TILE_BYTES;
            const size_t scale_tile_base =
                static_cast<size_t>(row_tile) * scale_bytes_per_row_tile +
                static_cast<size_t>(col_tile) * NPU_DECODE_GEMV_SCALE_TILE_BYTES;
            for (uint32_t row_lane = 0; row_lane < NPU_DECODE_GEMV_M_TILE; ++row_lane) {
                const int64_t row = static_cast<int64_t>(row_tile) * NPU_DECODE_GEMV_M_TILE + row_lane;
                if (row >= out_channels) {
                    continue;
                }
                const auto * w_row = weight + row * weight_nb1;
                float amax = 0.0f;
                for (uint32_t lane = 0; lane < NPU_DECODE_GEMV_W8_K_TILE; ++lane) {
                    const int64_t col = static_cast<int64_t>(col_tile) * NPU_DECODE_GEMV_W8_K_TILE + lane;
                    if (col >= k) {
                        continue;
                    }
                    ggml_fp16_t fp16_value;
                    std::memcpy(&fp16_value, w_row + col * weight_nb0, sizeof(fp16_value));
                    const uint16_t bits = aicas_rtl_fp16_zero_if_non_normal(aicas_rtl_fp16_bits(fp16_value));
                    amax = std::max(amax, std::fabs(aicas_rtl_fp16_to_f32(bits)));
                }
                const float d = amax / 127.0f;
                const float id = d != 0.0f ? 1.0f / d : 0.0f;
                const ggml_fp16_t scale_fp16 = ggml_fp32_to_fp16(d);
                std::memcpy(next.packed_scale.data() + scale_tile_base + row_lane * sizeof(ggml_fp16_t),
                        &scale_fp16,
                        sizeof(scale_fp16));

                uint8_t * dst = next.packed_weight.data() + tile_base +
                    static_cast<size_t>(row_lane) * NPU_DECODE_GEMV_W8_WEIGHT_ROW_BYTES;
                for (uint32_t lane = 0; lane < NPU_DECODE_GEMV_W8_K_TILE; ++lane) {
                    const int64_t col = static_cast<int64_t>(col_tile) * NPU_DECODE_GEMV_W8_K_TILE + lane;
                    if (col >= k) {
                        continue;
                    }
                    ggml_fp16_t fp16_value;
                    std::memcpy(&fp16_value, w_row + col * weight_nb0, sizeof(fp16_value));
                    const uint16_t bits = aicas_rtl_fp16_zero_if_non_normal(aicas_rtl_fp16_bits(fp16_value));
                    const float value = aicas_rtl_fp16_to_f32(bits);
                    int q = static_cast<int>(std::round(value * id));
                    q = std::max(-127, std::min(127, q));
                    dst[lane] = static_cast<uint8_t>(static_cast<int8_t>(q));
                }
            }
        }
    }

    if (allocate_heap_cma &&
            npu_decode_w8a16_cma_enabled() &&
            packed_bytes_64 <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        next.cma_packed = npu_decode_mem_alloc(static_cast<size_t>(packed_bytes_64));
        if (next.cma_packed != nullptr) {
            next.cma_bytes = static_cast<uint32_t>(packed_bytes_64);
            next.cma_packed_owned = true;
            std::memcpy(next.cma_packed, next.packed_weight.data(), next.packed_weight.size());
        }
    }
    if (allocate_heap_cma &&
            npu_decode_w8a16_cma_enabled() &&
            scale_bytes_64 <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        next.cma_scale = npu_decode_mem_alloc(static_cast<size_t>(scale_bytes_64));
        if (next.cma_scale != nullptr) {
            next.cma_scale_bytes = static_cast<uint32_t>(scale_bytes_64);
            next.cma_scale_owned = true;
            std::memcpy(next.cma_scale, next.packed_scale.data(), next.packed_scale.size());
        }
    }

    if (entry.cma_packed != nullptr && entry.cma_packed_owned) {
        npu_decode_mem_free(entry.cma_packed);
    }
    if (entry.cma_scale != nullptr && entry.cma_scale_owned) {
        npu_decode_mem_free(entry.cma_scale);
    }
    entry = std::move(next);
    return true;
}

bool ggml_backend_npu_decode_w8a16_preload(
        const char * weight_name,
        const void * weight_data,
        int64_t k,
        int64_t out_channels,
        int64_t weight_nb0,
        int64_t weight_nb1,
        int64_t group) {
    using namespace ggml_npu;

    auto fail = [weight_name](const char * reason) {
        if (npu_decode_w4a16_debug_log_enabled()) {
            GGML_LOG_WARN("%s: reject %s: %s\n", __func__, weight_name != nullptr ? weight_name : "(unnamed)", reason);
        }
        return false;
    };

    if (weight_data == nullptr || k <= 0 || out_channels <= 0 ||
            k > std::numeric_limits<uint16_t>::max() ||
            out_channels > std::numeric_limits<uint16_t>::max() ||
            group <= 0 || group > static_cast<int64_t>(NPU_DECODE_GEMV_K_TILE)) {
        return fail("invalid dimensions");
    }
    if (weight_nb0 != static_cast<int64_t>(sizeof(ggml_fp16_t))) {
        return fail("weight tensor must be contiguous FP16 along K");
    }
    npu_decode_w8a16_cache & cache = npu_decode_w8a16_weight_cache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    const bool ok = npu_decode_w8a16_prepare_weight_cache(
            cache.entry,
            weight_name,
            weight_data,
            k,
            out_channels,
            weight_nb0,
            weight_nb1,
            group,
            false);
    bool cma_ok = false;
    if (ok) {
        cma_ok = npu_decode_w16a16_reserve_static_weight_cma(cache.entry, weight_name);
        (void) npu_decode_w8a16_reserve_static_scale_cma(cache.entry);
    }
    if (npu_decode_w4a16_debug_log_enabled() && ok) {
        GGML_LOG_WARN("%s: preloaded %s M=%" PRId64 " K=%" PRId64 " group=%" PRId64
                " packed=%zu scale=%zu cma=%d cma_offset=0x%08x scale_cma=%u\n",
                __func__,
                weight_name != nullptr ? weight_name : "(unnamed)",
                out_channels,
                k,
                group,
                cache.entry.packed_weight.size(),
                cache.entry.packed_scale.size(),
                cma_ok ? 1 : 0,
                cache.entry.cma_packed_offset,
                cache.entry.cma_scale_bytes);
    }
    return ok;
}

bool ggml_backend_npu_decode_w16a16_preload(
        const char * weight_name,
        const void * weight_data,
        int64_t k,
        int64_t out_channels,
        int64_t weight_nb0,
        int64_t weight_nb1) {
    // W16A16 is intentionally disabled for the current decode RTL/runtime.
    (void) weight_name;
    (void) weight_data;
    (void) k;
    (void) out_channels;
    (void) weight_nb0;
    (void) weight_nb1;
    return false;

    using namespace ggml_npu;

    auto fail = [weight_name](const char * reason) {
        if (npu_decode_w4a16_debug_log_enabled()) {
            GGML_LOG_WARN("%s: reject %s: %s\n", __func__, weight_name != nullptr ? weight_name : "(unnamed)", reason);
        }
        return false;
    };

    if (weight_name == nullptr || weight_name[0] == '\0' || weight_data == nullptr) {
        return fail("null pointer");
    }
    if (k <= 0 || out_channels <= 0 ||
            k > std::numeric_limits<uint16_t>::max() ||
            out_channels > std::numeric_limits<uint16_t>::max()) {
        return fail("invalid dimensions");
    }
    if (weight_nb0 != static_cast<int64_t>(sizeof(ggml_fp16_t))) {
        return fail("weight tensor must be contiguous FP16 along K");
    }

    npu_decode_update_runtime_heap_env();
    if (npu_decode_init_with_cma_retry(weight_name, false) != 0) {
        return fail("decode runtime init failed");
    }

    npu_decode_w16a16_cache & cache = npu_decode_w16a16_weight_cache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    if (!npu_decode_w16a16_prepare_weight_cache(
                cache.entry,
                weight_name,
                weight_data,
                k,
                out_channels,
                weight_nb0,
                weight_nb1,
                false)) {
        return fail("weight cache preparation failed");
    }
    (void) npu_decode_w16a16_reserve_static_weight_cma(cache.entry, weight_name);

    const npu_decode_w16a16_cached_weight & cached_const = cache.entry;
    const bool cma_resident = cached_const.cma_packed != nullptr &&
        cached_const.cma_bytes >= cached_const.packed_weight.size();
    const bool use_pingpong = cma_resident &&
        npu_decode_w16a16_pingpong_enabled() &&
        cached_const.pingpong_max_block_rows > 0;
    const uint32_t max_block_rows = use_pingpong ? cached_const.pingpong_max_block_rows : cached_const.max_block_rows;
    const uint32_t max_row_tiles = npu_decode_ceil_div_u32(max_block_rows, NPU_DECODE_GEMV_M_TILE);
    const uint32_t max_scale_bytes = max_row_tiles * cached_const.col_tiles * NPU_DECODE_GEMV_SCALE_TILE_BYTES;
    npu_decode_w16a16_cached_weight & cached = cache.entry;
    (void) npu_decode_w16a16_reserve_static_scale_cma(cached, weight_name, max_scale_bytes);

    if (npu_decode_w4a16_debug_log_enabled()) {
        GGML_LOG_WARN("%s: preloaded %s M=%" PRId64 " K=%" PRId64 " packed=%zu cma=%d scale_cma=%u\n",
                __func__,
                weight_name,
                out_channels,
                k,
                cached.packed_weight.size(),
                cma_resident ? 1 : 0,
                cached.cma_scale_ones_bytes);
    }
    return !cached.packed_weight.empty();
}

bool ggml_backend_npu_decode_w16a16_gemv_ex(
        const char * op_name,
        const void * weight_data,
        int64_t k,
        int64_t out_channels,
        int64_t weight_nb0,
        int64_t weight_nb1,
        const void * act_data,
        int act_type,
        int64_t act_nb0,
        int64_t act_nb1,
        int64_t n_cols,
        void * dst_data,
        int dst_type,
        int64_t dst_nb1) {
    // W16A16 is intentionally disabled for the current decode RTL/runtime.
    (void) op_name;
    (void) weight_data;
    (void) k;
    (void) out_channels;
    (void) weight_nb0;
    (void) weight_nb1;
    (void) act_data;
    (void) act_type;
    (void) act_nb0;
    (void) act_nb1;
    (void) n_cols;
    (void) dst_data;
    (void) dst_type;
    (void) dst_nb1;
    return false;

    using namespace ggml_npu;

    auto fail = [op_name](const char * reason) {
        if (npu_decode_w4a16_debug_log_enabled()) {
            GGML_LOG_WARN("%s: reject %s: %s\n", __func__, op_name != nullptr ? op_name : "(unnamed)", reason);
        }
        return false;
    };

    if (weight_data == nullptr || act_data == nullptr || dst_data == nullptr) {
        return fail("null pointer");
    }
    if (k <= 0 || out_channels <= 0 || n_cols <= 0 ||
            k > std::numeric_limits<uint16_t>::max() ||
            out_channels > std::numeric_limits<uint16_t>::max()) {
        return fail("invalid dimensions");
    }
    if (weight_nb0 != static_cast<int64_t>(sizeof(ggml_fp16_t))) {
        return fail("weight tensor must be contiguous FP16 along K");
    }
    if (act_type != GGML_TYPE_F16 && act_type != GGML_TYPE_F32) {
        return fail("activation tensor must be F16 or F32");
    }
    if (dst_type != GGML_TYPE_F16 && dst_type != GGML_TYPE_F32) {
        return fail("destination tensor must be F16 or F32");
    }
    if (npu_decode_w4a16_debug_log_enabled()) {
        GGML_LOG_WARN("%s: start %s M=%" PRId64 " K=%" PRId64 " n_cols=%" PRId64
                " act_type=%s dst_type=%s nb0=%" PRId64 " nb1=%" PRId64 "\n",
                __func__,
                op_name != nullptr ? op_name : "(unnamed)",
                out_channels,
                k,
                n_cols,
                ggml_type_name(static_cast<ggml_type>(act_type)),
                ggml_type_name(static_cast<ggml_type>(dst_type)),
                weight_nb0,
                weight_nb1);
    }

    const bool collect_profile = npu_decode_profile_enabled();
    const int64_t total_start_us = collect_profile ? ggml_time_us() : 0;
    const int64_t setup_start_us = collect_profile ? total_start_us : 0;
    int64_t setup_stage_start_us = setup_start_us;
    npu_decode_gemv_profile_record profile = {};
    profile.profile_kind = "aicas_decode_w16a16_lm_head";
    profile.gemv_mode = "w16a16";
    profile.op_role = "lm_head";
    profile.op_name = op_name != nullptr ? op_name : "";
    profile.dst_type = dst_type;
    profile.k = k;
    profile.n_cols = n_cols;
    profile.out_channels = out_channels;
    profile.pingpong = false;

    if (collect_profile) {
        setup_stage_start_us = ggml_time_us();
    }
    if (!npu_decode_ensure_overlay_active(op_name)) {
        return fail("decode overlay switch failed");
    }
    if (collect_profile) {
        profile.overlay_ensure_us = static_cast<uint64_t>(ggml_time_us() - setup_stage_start_us);
        setup_stage_start_us = ggml_time_us();
    }
    npu_decode_update_runtime_heap_env();
    if (collect_profile) {
        profile.heap_update_us = static_cast<uint64_t>(ggml_time_us() - setup_stage_start_us);
        setup_stage_start_us = ggml_time_us();
    }
    if (npu_decode_init_with_cma_retry(op_name, true) != 0) {
        return fail("decode runtime init failed");
    }
    if (collect_profile) {
        profile.runtime_init_us = static_cast<uint64_t>(ggml_time_us() - setup_stage_start_us);
        setup_stage_start_us = ggml_time_us();
    }

    npu_decode_w16a16_cache & cache = npu_decode_w16a16_weight_cache();
    std::unique_lock<std::mutex> cache_lock(cache.mutex);
    if (!npu_decode_w16a16_prepare_weight_cache(
                cache.entry,
                op_name,
                weight_data,
                k,
                out_channels,
                weight_nb0,
                weight_nb1,
                true)) {
        return fail("weight cache preparation failed");
    }
    npu_decode_w16a16_cached_weight & cached = cache.entry;
    if (collect_profile) {
        profile.preload_lookup_us = static_cast<uint64_t>(ggml_time_us() - setup_stage_start_us);
        setup_stage_start_us = ggml_time_us();
    }
    npu_decode_w16a16_rebind_static_cma(cached);
    const bool cma_resident = cached.cma_packed != nullptr &&
        cached.cma_bytes >= cached.packed_weight.size();
    const bool use_pingpong = cma_resident &&
        npu_decode_w16a16_pingpong_enabled() &&
        cached.pingpong_max_block_rows > 0;
    if (npu_decode_w4a16_debug_log_enabled()) {
        GGML_LOG_WARN("%s: cache ready %s row_tiles=%u col_tiles=%u max_block_rows=%u pingpong_rows=%u packed=%zu cma=%d pingpong=%d\n",
                __func__,
                op_name != nullptr ? op_name : "(unnamed)",
                cached.row_tiles,
                cached.col_tiles,
                cached.max_block_rows,
                cached.pingpong_max_block_rows,
                cached.packed_weight.size(),
                cma_resident ? 1 : 0,
                use_pingpong ? 1 : 0);
    }

    npu_decode_gemv_layout act_layout = {};
    act_layout.col_tiles = cached.col_tiles;
    act_layout.act_bytes = cached.col_tiles * NPU_DECODE_GEMV_ACT_TILE_BYTES;
    if (act_layout.act_bytes > NPU_DECODE_GEMV_ACT_BUFFER_BYTES ||
            (act_layout.act_bytes % NPU_DECODE_GEMV_DMA_ALIGN) != 0) {
        return fail("activation layout exceeds act buffer");
    }
    const uint32_t max_block_rows = use_pingpong ? cached.pingpong_max_block_rows : cached.max_block_rows;
    const uint32_t max_row_tiles = npu_decode_ceil_div_u32(max_block_rows, NPU_DECODE_GEMV_M_TILE);
    const uint32_t max_weight_bytes = max_row_tiles * cached.bytes_per_row_tile;
    const uint32_t max_scale_bytes = max_row_tiles * cached.col_tiles * NPU_DECODE_GEMV_SCALE_TILE_BYTES;
    const bool hw_f32_mvout = dst_type == GGML_TYPE_F32 && npu_decode_hw_f32_mvout_enabled();
    const uint8_t output_precision = npu_decode_mvout_precision(hw_f32_mvout);
    const uint32_t block_output_bytes = npu_decode_host_output_bytes(max_block_rows, hw_f32_mvout);
    const uint32_t full_output_bytes = npu_decode_host_output_bytes(static_cast<uint32_t>(out_channels), hw_f32_mvout);
    const uint32_t output_cma_bytes = use_pingpong ? full_output_bytes : block_output_bytes;

    const int64_t alloc_start_us = collect_profile ? ggml_time_us() : 0;
    void * act_cma = npu_decode_mem_alloc(act_layout.act_bytes);
    void * weight_cma = cma_resident ? nullptr : npu_decode_mem_alloc(max_weight_bytes);
    void * scale_cma = npu_decode_w16a16_ensure_scale_ones_cma(cached, max_scale_bytes);
    bool scale_cma_owned = false;
    if (scale_cma == nullptr) {
        scale_cma = npu_decode_mem_alloc(max_scale_bytes);
        scale_cma_owned = scale_cma != nullptr;
    }
    void * out_cma = npu_decode_mem_alloc(output_cma_bytes);
    if (act_cma == nullptr || (!cma_resident && weight_cma == nullptr) ||
            scale_cma == nullptr || out_cma == nullptr) {
        npu_decode_mem_free(act_cma);
        npu_decode_mem_free(weight_cma);
        if (scale_cma_owned) {
            npu_decode_mem_free(scale_cma);
        }
        npu_decode_mem_free(out_cma);
        return fail("decode CMA allocation failed");
    }
    if (collect_profile) {
        profile.alloc_us = static_cast<uint64_t>(ggml_time_us() - alloc_start_us);
    }
    if (npu_decode_w4a16_debug_log_enabled()) {
        GGML_LOG_WARN("%s: cma allocated %s act=%u weight=%u scale=%u out=%u hw_f32=%d\n",
                __func__,
                op_name != nullptr ? op_name : "(unnamed)",
                act_layout.act_bytes,
                max_weight_bytes,
                max_scale_bytes,
                output_cma_bytes,
                hw_f32_mvout ? 1 : 0);
    }

    if (scale_cma_owned) {
        std::vector<uint8_t> scale_ones(max_scale_bytes, 0);
        const ggml_fp16_t fp16_one = ggml_fp32_to_fp16(1.0f);
        for (uint32_t i = 0; i + sizeof(ggml_fp16_t) <= max_scale_bytes; i += sizeof(ggml_fp16_t)) {
            std::memcpy(scale_ones.data() + i, &fp16_one, sizeof(fp16_one));
        }
        std::memcpy(scale_cma, scale_ones.data(), scale_ones.size());
    }

    if (collect_profile) {
        profile.max_block_rows = max_block_rows;
        profile.row_blocks_per_col =
            npu_decode_ceil_div_u32(static_cast<uint32_t>(out_channels), max_block_rows);
        profile.preloaded_host = true;
        profile.preloaded_cma = cma_resident;
        profile.pingpong = use_pingpong;
        profile.activation_bytes = static_cast<uint64_t>(act_layout.act_bytes) * static_cast<uint64_t>(n_cols);
        profile.scale_bytes = static_cast<uint64_t>(max_scale_bytes) * static_cast<uint64_t>(n_cols);
        profile.setup_us = static_cast<uint64_t>(ggml_time_us() - setup_start_us);
    }

    auto block_rows_for = [&](uint32_t row_base) {
        return std::min<uint32_t>(max_block_rows, static_cast<uint32_t>(out_channels) - row_base);
    };
    auto weight_offset_for = [&](uint32_t row_base) {
        return static_cast<size_t>(row_base / NPU_DECODE_GEMV_M_TILE) * cached.bytes_per_row_tile;
    };

    bool w16_pingpong_failed = false;
    for (int64_t col = 0; col < n_cols; ++col) {
        int64_t stage_start_us = collect_profile ? ggml_time_us() : 0;
        npu_decode_pack_activation(
                act_cma,
                act_layout.act_bytes,
                act_layout,
                act_data,
                act_type,
                act_nb0,
                act_nb1,
                nullptr,
                k,
                col);
        if (collect_profile) {
            profile.activation_preprocess_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            stage_start_us = ggml_time_us();
        }

        npu_decode_reset();
        if (collect_profile) {
            profile.reset_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            stage_start_us = ggml_time_us();
        }
        if (use_pingpong && !w16_pingpong_failed) {
            try {
                npu_decode_w16a16_pingpong_manual_run(
                        op_name,
                        act_cma,
                        scale_cma,
                        cached.cma_packed,
                        out_cma,
                        cached.col_tiles,
                        cached.bytes_per_row_tile,
                        max_block_rows,
                        static_cast<uint16_t>(out_channels),
                        static_cast<uint16_t>(k),
                        output_precision,
                        collect_profile ? &profile : nullptr);
            } catch (const std::exception & e) {
                if (npu_decode_w4a16_debug_log_enabled()) {
                    GGML_LOG_WARN("%s: W16 pingpong failed for %s: %s\n",
                            __func__, op_name != nullptr ? op_name : "(unnamed)", e.what());
                }
                w16_pingpong_failed = true;
                npu_decode_reset();
                stage_start_us = collect_profile ? ggml_time_us() : 0;
            }
            if (!w16_pingpong_failed) {
                if (collect_profile) {
                    profile.pingpong_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
                    profile.pingpong_calls += 1;
                    stage_start_us = ggml_time_us();
                }

                char * dst_col = reinterpret_cast<char *>(dst_data) + col * dst_nb1;
                if (dst_type == GGML_TYPE_F16) {
                    std::memcpy(dst_col, out_cma, static_cast<size_t>(out_channels) * sizeof(ggml_fp16_t));
                } else if (hw_f32_mvout) {
                    std::memcpy(reinterpret_cast<float *>(dst_col), out_cma, static_cast<size_t>(out_channels) * sizeof(float));
                } else {
                    npu_decode_convert_output_fp16_to_f32(
                            static_cast<const ggml_fp16_t *>(out_cma),
                            reinterpret_cast<float *>(dst_col),
                            static_cast<uint32_t>(out_channels));
                }
                if (collect_profile) {
                    profile.postprocess_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
                }
                continue;
            }
        }
        npu_decode_stream_unsupported_void(
                act_cma,
                NPU_DECODE_GEMV_ACT_BASE,
                act_layout.act_bytes - 1u,
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
        npu_decode_stream_unsupported_void(
                scale_cma,
                NPU_DECODE_GEMV_SCALE_BASE,
                max_scale_bytes - 1u,
                0,
                0,
                0,
                2,
                NPU_DECODE_GEMV_INPUT_TYPE_DATA,
                false,
                false,
                false,
                0,
                0,
                0);
        if (collect_profile) {
            profile.mvin_activation_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            profile.activation_mvin_calls += 1;
            stage_start_us = ggml_time_us();
        }

        uint32_t block_idx = 0;
        for (uint32_t row_base = 0; row_base < static_cast<uint32_t>(out_channels);
                row_base += block_rows_for(row_base), ++block_idx) {
            const uint32_t rows = block_rows_for(row_base);
            const uint32_t cur_weight_spm = NPU_DECODE_GEMV_PING_WEIGHT_BASE;
            const uint32_t weight_bytes =
                npu_decode_ceil_div_u32(rows, NPU_DECODE_GEMV_M_TILE) * cached.bytes_per_row_tile;
            const size_t weight_offset = weight_offset_for(row_base);
            stage_start_us = collect_profile ? ggml_time_us() : 0;
            void * weight_mvin_ptr = nullptr;
            if (cma_resident) {
                weight_mvin_ptr = static_cast<uint8_t *>(cached.cma_packed) + weight_offset;
            } else {
                std::memcpy(weight_cma, cached.packed_weight.data() + weight_offset, weight_bytes);
                weight_mvin_ptr = weight_cma;
                if (collect_profile) {
                    profile.host_copy_weight_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
                    stage_start_us = ggml_time_us();
                }
            }
            npu_decode_stream_unsupported_void(
                    weight_mvin_ptr,
                    cur_weight_spm,
                    weight_bytes - 1u,
                    0,
                    0,
                    0,
                    2,
                    NPU_DECODE_GEMV_INPUT_TYPE_WEIGHT,
                    false,
                    false,
                    false,
                    0,
                    0,
                    0);
            if (npu_decode_w4a16_debug_log_enabled() && (block_idx == 0 || row_base + rows >= static_cast<uint32_t>(out_channels))) {
                GGML_LOG_WARN("%s: block %s idx=%u row_base=%u rows=%u weight_bytes=%u\n",
                        __func__,
                        op_name != nullptr ? op_name : "(unnamed)",
                        block_idx,
                        row_base,
                        rows,
                        weight_bytes);
            }
            if (collect_profile) {
                profile.mvin_weight_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
                profile.weight_mvin_calls += 1;
            }

            stage_start_us = collect_profile ? ggml_time_us() : 0;
            const uint64_t flow = npu_decode_make_bypass_output_flow(static_cast<uint16_t>(rows));
            npu_decode_trace_stream_flow(
                    "lm_head_w16",
                    op_name,
                    block_idx,
                    row_base,
                    rows,
                    static_cast<uint32_t>(k),
                    NPU_DECODE_GEMV_MODE_W16A16,
                    cur_weight_spm,
                    NPU_DECODE_GEMV_ACT_BASE,
                    NPU_DECODE_GEMV_OUTPUT_BASE,
                    NPU_DECODE_GEMV_SCALE_BASE,
                    flow);
            npu_decode_stream_unsupported_void(
                    op_name,
                    cur_weight_spm,
                    NPU_DECODE_GEMV_ACT_BASE,
                    static_cast<uint16_t>(k),
                    static_cast<uint16_t>(rows),
                    NPU_DECODE_GEMV_OUTPUT_BASE,
                    NPU_DECODE_GEMV_SCALE_BASE,
                    NPU_DECODE_GEMV_MODE_W16A16,
                    flow);
            if (collect_profile) {
                profile.gemv_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
                profile.gemv_calls += 1;
                profile.weight_bytes += weight_bytes;
                stage_start_us = ggml_time_us();
            }

            npu_decode_stream_unsupported_void(
                    out_cma,
                    NPU_DECODE_GEMV_OUTPUT_BASE,
                    0,
                    rows - 1u,
                    1,
                    1,
                    output_precision,
                    1,
                    false,
                    false,
                    0,
                    0);
            if (collect_profile) {
                profile.mvout_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
                profile.mvout_calls += 1;
                profile.output_bytes += npu_decode_host_output_bytes(rows, hw_f32_mvout);
                stage_start_us = ggml_time_us();
            }

            char * dst_col = reinterpret_cast<char *>(dst_data) + col * dst_nb1;
            if (dst_type == GGML_TYPE_F16) {
                std::memcpy(
                        dst_col + static_cast<size_t>(row_base) * sizeof(ggml_fp16_t),
                        out_cma,
                        static_cast<size_t>(rows) * sizeof(ggml_fp16_t));
            } else if (hw_f32_mvout) {
                std::memcpy(
                        reinterpret_cast<float *>(dst_col) + row_base,
                        out_cma,
                        static_cast<size_t>(rows) * sizeof(float));
            } else {
                npu_decode_convert_output_fp16_to_f32(
                        static_cast<const ggml_fp16_t *>(out_cma),
                        reinterpret_cast<float *>(dst_col) + row_base,
                        rows);
            }
            if (collect_profile) {
                profile.postprocess_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            }
        }
    }

    const int64_t cleanup_start_us = collect_profile ? ggml_time_us() : 0;
    npu_decode_mem_free(act_cma);
    npu_decode_mem_free(weight_cma);
    if (scale_cma_owned) {
        npu_decode_mem_free(scale_cma);
    }
    npu_decode_mem_free(out_cma);
    if (collect_profile) {
        profile.cleanup_us = static_cast<uint64_t>(ggml_time_us() - cleanup_start_us);
        profile.total_us = static_cast<uint64_t>(ggml_time_us() - total_start_us);
        npu_decode_profile_write(profile);
    }
    return true;
}

bool ggml_backend_npu_decode_w8a16_gemv_ex(
        const char * op_name,
        const void * weight_data,
        int64_t k,
        int64_t out_channels,
        int64_t weight_nb0,
        int64_t weight_nb1,
        int64_t group,
        const void * act_data,
        int act_type,
        int64_t act_nb0,
        int64_t act_nb1,
        int64_t n_cols,
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

    if (weight_data == nullptr || act_data == nullptr || dst_data == nullptr) {
        return fail("null pointer");
    }
    if (k <= 0 || out_channels <= 0 || n_cols <= 0 ||
            k > std::numeric_limits<uint16_t>::max() ||
            out_channels > std::numeric_limits<uint16_t>::max() ||
            group <= 0 || group > static_cast<int64_t>(NPU_DECODE_GEMV_K_TILE)) {
        return fail("invalid dimensions");
    }
    if (weight_nb0 != static_cast<int64_t>(sizeof(ggml_fp16_t))) {
        return fail("weight tensor must be contiguous FP16 along K");
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
    int64_t setup_stage_start_us = setup_start_us;
    npu_decode_gemv_profile_record profile = {};
    profile.profile_kind = "aicas_decode_w8a16_lm_head";
    profile.gemv_mode = "w8a16";
    profile.op_role = "lm_head";
    profile.op_name = op_name != nullptr ? op_name : "";
    profile.dst_type = dst_type;
    profile.k = k;
    profile.n_cols = n_cols;
    profile.out_channels = out_channels;
    profile.pingpong = false;

    if (collect_profile) {
        setup_stage_start_us = ggml_time_us();
    }
    if (!npu_decode_ensure_overlay_active(op_name)) {
        return fail("decode overlay switch failed");
    }
    if (collect_profile) {
        profile.overlay_ensure_us = static_cast<uint64_t>(ggml_time_us() - setup_stage_start_us);
        setup_stage_start_us = ggml_time_us();
    }
    npu_decode_update_runtime_heap_env();
    if (collect_profile) {
        profile.heap_update_us = static_cast<uint64_t>(ggml_time_us() - setup_stage_start_us);
        setup_stage_start_us = ggml_time_us();
    }
    if (npu_decode_init_with_cma_retry(op_name, true) != 0) {
        return fail("decode runtime init failed");
    }
    if (collect_profile) {
        profile.runtime_init_us = static_cast<uint64_t>(ggml_time_us() - setup_stage_start_us);
        setup_stage_start_us = ggml_time_us();
    }

    npu_decode_w8a16_cache & cache = npu_decode_w8a16_weight_cache();
    std::unique_lock<std::mutex> cache_lock(cache.mutex);
    if (!npu_decode_w8a16_prepare_weight_cache(
                cache.entry,
                op_name,
                weight_data,
                k,
                out_channels,
                weight_nb0,
                weight_nb1,
                group,
                true)) {
        return fail("weight cache preparation failed");
    }
    npu_decode_w16a16_cached_weight & cached = cache.entry;
    if (collect_profile) {
        profile.preload_lookup_us = static_cast<uint64_t>(ggml_time_us() - setup_stage_start_us);
        setup_stage_start_us = ggml_time_us();
    }
    npu_decode_w16a16_rebind_static_cma(cached);
    const bool cma_resident = cached.cma_packed != nullptr &&
        cached.cma_bytes >= cached.packed_weight.size();

    npu_decode_gemv_layout act_layout = {};
    act_layout.col_tiles = cached.col_tiles;
    act_layout.act_bytes = npu_decode_w8_act_mvin_bytes(cached.col_tiles);
    if (act_layout.act_bytes > NPU_DECODE_GEMV_ACT_BUFFER_BYTES ||
            (act_layout.act_bytes % NPU_DECODE_GEMV_DMA_ALIGN) != 0) {
        return fail("activation layout exceeds act buffer");
    }
    const uint32_t scale_bytes_per_row_tile = cached.col_tiles * NPU_DECODE_GEMV_SCALE_TILE_BYTES;
    const uint64_t expected_scale_bytes =
        static_cast<uint64_t>(cached.group_count) * static_cast<uint64_t>(cached.row_tiles) *
        static_cast<uint64_t>(scale_bytes_per_row_tile);
    if (cached.packed_scale.size() != static_cast<size_t>(expected_scale_bytes)) {
        return fail("W8 scale cache is not separated or has invalid size");
    }
    const bool scale_cma_resident = cached.cma_scale != nullptr &&
        cached.cma_scale_bytes >= cached.packed_scale.size();
    const bool stream_cma_path = npu_decode_stream_api_enabled() && cma_resident && scale_cma_resident;
    auto gcd_u32 = [](uint32_t a, uint32_t b) {
        while (b != 0) {
            const uint32_t t = a % b;
            a = b;
            b = t;
        }
        return a;
    };
    auto w8_stream_block_rows = [&]() {
        constexpr uint32_t stream_max_rows = 4096u;
        constexpr uint32_t stream_weight_max_bytes = 4194240u;
        uint32_t max_row_tiles = stream_max_rows / NPU_DECODE_GEMV_M_TILE;
        if (cached.bytes_per_row_tile == 0 || scale_bytes_per_row_tile == 0) {
            return 0u;
        }
        max_row_tiles = std::min<uint32_t>(
                max_row_tiles,
                stream_weight_max_bytes / cached.bytes_per_row_tile);
        max_row_tiles = std::min<uint32_t>(
                max_row_tiles,
                (256u * 1024u) / scale_bytes_per_row_tile);
        const uint32_t row_tile_align =
            NPU_DECODE_GEMV_DMA_ALIGN / gcd_u32(scale_bytes_per_row_tile, NPU_DECODE_GEMV_DMA_ALIGN);
        if (max_row_tiles < row_tile_align) {
            return 0u;
        }
        max_row_tiles = (max_row_tiles / row_tile_align) * row_tile_align;
        return max_row_tiles * NPU_DECODE_GEMV_M_TILE;
    };
    const uint32_t max_block_rows = stream_cma_path ? w8_stream_block_rows() : cached.max_block_rows;
    if (max_block_rows == 0) {
        return fail("invalid W8 stream block rows");
    }
    const uint32_t max_row_tiles = npu_decode_ceil_div_u32(max_block_rows, NPU_DECODE_GEMV_M_TILE);
    const uint32_t max_weight_bytes = max_row_tiles * cached.bytes_per_row_tile;
    const uint32_t max_scale_bytes = max_row_tiles * scale_bytes_per_row_tile;
    const bool use_pingpong = !stream_cma_path && cma_resident && scale_cma_resident && npu_decode_w8a16_pingpong_enabled();
    const bool hw_f32_mvout = dst_type == GGML_TYPE_F32 && npu_decode_hw_f32_mvout_enabled();
    const uint8_t output_precision = npu_decode_mvout_precision(hw_f32_mvout);
    const uint32_t block_output_bytes = npu_decode_host_output_bytes(max_block_rows, hw_f32_mvout);
    const uint32_t full_output_bytes = npu_decode_host_output_bytes(static_cast<uint32_t>(out_channels), hw_f32_mvout);
    const uint32_t output_cma_bytes = use_pingpong ? full_output_bytes : block_output_bytes;

    if (npu_decode_w4a16_debug_log_enabled()) {
        GGML_LOG_WARN("%s: %s M=%" PRId64 " K=%" PRId64 " group=%" PRId64
                " row_tiles=%u col_tiles=%u max_rows=%u weight_bytes=%zu scale_bytes=%zu"
                " cma=%d cma_off=0x%08x cma_bytes=%u scale_cma=%d scale_off=0x%08x scale_cma_bytes=%u"
                " pingpong=%d hw_f32=%d out_cma_bytes=%u\n",
                __func__,
                op_name != nullptr ? op_name : "(unnamed)",
                out_channels,
                k,
                group,
                cached.row_tiles,
                cached.col_tiles,
                max_block_rows,
                cached.packed_weight.size(),
                cached.packed_scale.size(),
                cma_resident ? 1 : 0,
                cached.cma_packed_offset,
                cached.cma_bytes,
                scale_cma_resident ? 1 : 0,
                cached.cma_scale_offset,
                cached.cma_scale_bytes,
                use_pingpong ? 1 : 0,
                hw_f32_mvout ? 1 : 0,
                output_cma_bytes);
        std::fflush(stderr);
    }

    const int64_t alloc_start_us = collect_profile ? ggml_time_us() : 0;
    void * act_cma = npu_decode_mem_alloc(act_layout.act_bytes);
    void * weight_cma = cma_resident ? nullptr : npu_decode_mem_alloc(max_weight_bytes);
    void * scale_cma = scale_cma_resident ? nullptr : npu_decode_mem_alloc(max_scale_bytes);
    void * out_cma = npu_decode_mem_alloc(output_cma_bytes);
    if (act_cma == nullptr || (!cma_resident && weight_cma == nullptr) ||
            (!scale_cma_resident && scale_cma == nullptr) || out_cma == nullptr) {
        npu_decode_mem_free(act_cma);
        npu_decode_mem_free(weight_cma);
        npu_decode_mem_free(scale_cma);
        npu_decode_mem_free(out_cma);
        return fail("decode CMA allocation failed");
    }
    if (collect_profile) {
        profile.alloc_us = static_cast<uint64_t>(ggml_time_us() - alloc_start_us);
        profile.max_block_rows = max_block_rows;
        profile.row_blocks_per_col =
            npu_decode_ceil_div_u32(static_cast<uint32_t>(out_channels), max_block_rows);
        profile.preloaded_host = true;
        profile.preloaded_cma = cma_resident && scale_cma_resident;
        profile.pingpong = use_pingpong;
        profile.activation_bytes = static_cast<uint64_t>(act_layout.act_bytes) *
            static_cast<uint64_t>(n_cols);
        profile.scale_bytes = static_cast<uint64_t>(max_scale_bytes) *
            static_cast<uint64_t>(n_cols);
        profile.setup_us = static_cast<uint64_t>(ggml_time_us() - setup_start_us);
    }

    auto block_rows_for = [&](uint32_t row_base) {
        return std::min<uint32_t>(max_block_rows, static_cast<uint32_t>(out_channels) - row_base);
    };
    auto weight_offset_for = [&](uint32_t row_base) {
        return static_cast<size_t>(row_base / NPU_DECODE_GEMV_M_TILE) * cached.bytes_per_row_tile;
    };
    auto scale_offset_for = [&](uint32_t row_base) {
        return static_cast<size_t>(row_base / NPU_DECODE_GEMV_M_TILE) * scale_bytes_per_row_tile;
    };

    for (int64_t col = 0; col < n_cols; ++col) {
        int64_t stage_start_us = collect_profile ? ggml_time_us() : 0;
        npu_decode_pack_w8_activation_slice(
                act_cma,
                act_layout.act_bytes,
                act_layout,
                act_data,
                act_type,
                act_nb0,
                act_nb1,
                k,
                0,
                k,
                col);
        if (collect_profile) {
            profile.activation_preprocess_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            stage_start_us = ggml_time_us();
        }

        npu_decode_reset();
        if (collect_profile) {
            profile.reset_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            stage_start_us = ggml_time_us();
        }

        if (npu_decode_stream_api_enabled() && cma_resident && scale_cma_resident) {
            bool stream_ok = true;
            for (uint32_t row_base = 0; row_base < static_cast<uint32_t>(out_channels);
                    row_base += block_rows_for(row_base)) {
                const uint32_t rows = block_rows_for(row_base);
                const size_t weight_offset = weight_offset_for(row_base);
                const size_t scale_offset = scale_offset_for(row_base);
                kv260_stream_gemv_desc desc = {};
                desc.act_ptr = act_cma;
                desc.weight_payload_ptr = static_cast<uint8_t *>(cached.cma_packed) + weight_offset;
                desc.weight_scale_ptr = static_cast<uint8_t *>(cached.cma_scale) + scale_offset;
                desc.output_ptr = out_cma;
                desc.m = static_cast<uint16_t>(rows);
                desc.n = static_cast<uint16_t>(k);
                desc.mode = KV260_STREAM_DECODE_GEMV_W8A16;
                desc.output_precision = npu_decode_stream_output_precision(hw_f32_mvout);
                desc.role = KV260_STREAM_GEMV_ROLE_LINEAR;
                desc.dst = KV260_STREAM_GEMV_DST_OUTPUT;
                desc.post_op = KV260_STREAM_POST_BYPASS;
                stage_start_us = collect_profile ? ggml_time_us() : 0;
                const int rc = npu_decode_stream_gemv(op_name, desc);
                if (rc != 0) {
                    if (npu_decode_w4a16_debug_log_enabled()) {
                        GGML_LOG_WARN("%s: stream W8 GEMV failed for %s row_base=%u rows=%u rc=%d\n",
                                __func__, op_name != nullptr ? op_name : "(unnamed)", row_base, rows, rc);
                    }
                    stream_ok = false;
                    break;
                }
                if (collect_profile) {
                    profile.pingpong_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
                    profile.pingpong_calls += 1;
                    profile.weight_bytes += npu_decode_ceil_div_u32(rows, NPU_DECODE_GEMV_M_TILE) * cached.bytes_per_row_tile;
                    profile.output_bytes += npu_decode_host_output_bytes(rows, hw_f32_mvout);
                    stage_start_us = ggml_time_us();
                }

                char * dst_col = reinterpret_cast<char *>(dst_data) + col * dst_nb1;
                if (dst_type == GGML_TYPE_F16) {
                    std::memcpy(
                            reinterpret_cast<ggml_fp16_t *>(dst_col) + row_base,
                            out_cma,
                            static_cast<size_t>(rows) * sizeof(ggml_fp16_t));
                } else if (hw_f32_mvout) {
                    std::memcpy(
                            reinterpret_cast<float *>(dst_col) + row_base,
                            out_cma,
                            static_cast<size_t>(rows) * sizeof(float));
                } else {
                    npu_decode_convert_output_fp16_to_f32(
                            static_cast<const ggml_fp16_t *>(out_cma),
                            reinterpret_cast<float *>(dst_col) + row_base,
                            rows);
                }
                if (collect_profile) {
                    profile.postprocess_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
                }
            }
            if (stream_ok) {
                continue;
            }
        }

        if (npu_decode_stream_api_enabled()) {
            npu_decode_mem_free(act_cma);
            npu_decode_mem_free(weight_cma);
            npu_decode_mem_free(scale_cma);
            npu_decode_mem_free(out_cma);
            return fail("stream W8 GEMV failed or assets are not CMA resident; split/legacy NPU path disabled");
        }

        if (use_pingpong) {
            try {
                npu_decode_w8a16_pingpong_manual_run(
                        op_name,
                        act_cma,
                        cached.cma_scale,
                        cached.cma_packed,
                        out_cma,
                        cached.col_tiles,
                        cached.bytes_per_row_tile,
                        max_block_rows,
                        static_cast<uint16_t>(out_channels),
                        static_cast<uint16_t>(k),
                        output_precision,
                        collect_profile ? &profile : nullptr);
            } catch (const std::exception & e) {
                return fail(e.what());
            }
            if (collect_profile) {
                profile.pingpong_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
                profile.pingpong_calls += 1;
                stage_start_us = ggml_time_us();
            }

            char * dst_col = reinterpret_cast<char *>(dst_data) + col * dst_nb1;
            if (dst_type == GGML_TYPE_F16) {
                std::memcpy(dst_col, out_cma, static_cast<size_t>(out_channels) * sizeof(ggml_fp16_t));
            } else if (hw_f32_mvout) {
                std::memcpy(dst_col, out_cma, static_cast<size_t>(out_channels) * sizeof(float));
            } else {
                npu_decode_convert_output_fp16_to_f32(
                        static_cast<const ggml_fp16_t *>(out_cma),
                        reinterpret_cast<float *>(dst_col),
                        static_cast<uint32_t>(out_channels));
            }
            if (collect_profile) {
                profile.postprocess_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            }
            continue;
        }

        npu_decode_stream_unsupported_void(
                op_name,
                act_cma,
                NPU_DECODE_GEMV_ACT_BASE,
                act_layout.act_bytes,
                2,
                NPU_DECODE_GEMV_INPUT_TYPE_ACT);
        if (collect_profile) {
            profile.mvin_activation_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            profile.activation_mvin_calls += 1;
            stage_start_us = ggml_time_us();
        }

        uint32_t block_idx = 0;
        for (uint32_t row_base = 0; row_base < static_cast<uint32_t>(out_channels);
                row_base += block_rows_for(row_base), ++block_idx) {
            const uint32_t rows = block_rows_for(row_base);
            const uint32_t cur_weight_spm = NPU_DECODE_GEMV_PING_WEIGHT_BASE;
            const uint32_t weight_bytes =
                npu_decode_ceil_div_u32(rows, NPU_DECODE_GEMV_M_TILE) * cached.bytes_per_row_tile;
            const uint32_t scale_bytes =
                npu_decode_ceil_div_u32(rows, NPU_DECODE_GEMV_M_TILE) * scale_bytes_per_row_tile;
            const size_t weight_offset = weight_offset_for(row_base);
            const size_t scale_offset = scale_offset_for(row_base);
            stage_start_us = collect_profile ? ggml_time_us() : 0;
            void * scale_mvin_ptr = nullptr;
            if (scale_cma_resident) {
                scale_mvin_ptr = static_cast<uint8_t *>(cached.cma_scale) + scale_offset;
            } else {
                std::memcpy(scale_cma, cached.packed_scale.data() + scale_offset, scale_bytes);
                scale_mvin_ptr = scale_cma;
                if (collect_profile) {
                    profile.host_copy_weight_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
                    stage_start_us = ggml_time_us();
                }
            }
            npu_decode_stream_unsupported_void(
                    op_name,
                    scale_mvin_ptr,
                    NPU_DECODE_GEMV_SCALE_BASE,
                    scale_bytes,
                    2,
                    NPU_DECODE_GEMV_INPUT_TYPE_DATA,
                    0,
                    0);
            if (collect_profile) {
                profile.scale_mvin_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
                profile.scale_mvin_calls += 1;
                stage_start_us = ggml_time_us();
            }
            void * weight_mvin_ptr = nullptr;
            if (cma_resident) {
                weight_mvin_ptr = static_cast<uint8_t *>(cached.cma_packed) + weight_offset;
            } else {
                std::memcpy(weight_cma, cached.packed_weight.data() + weight_offset, weight_bytes);
                weight_mvin_ptr = weight_cma;
                if (collect_profile) {
                    profile.host_copy_weight_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
                    stage_start_us = ggml_time_us();
                }
            }
            npu_decode_stream_unsupported_void(
                    op_name,
                    weight_mvin_ptr,
                    cur_weight_spm,
                    weight_bytes,
                    1,
                    NPU_DECODE_GEMV_INPUT_TYPE_WEIGHT);
            if (collect_profile) {
                profile.mvin_weight_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
                profile.weight_mvin_calls += 1;
                stage_start_us = ggml_time_us();
            }

            const uint64_t flow = npu_decode_make_bypass_output_flow(static_cast<uint16_t>(rows));
            npu_decode_trace_stream_flow(
                    "lm_head_w8",
                    op_name,
                    block_idx,
                    row_base,
                    rows,
                    static_cast<uint32_t>(k),
                    NPU_DECODE_GEMV_MODE_W8A16,
                    cur_weight_spm,
                    NPU_DECODE_GEMV_ACT_BASE,
                    NPU_DECODE_GEMV_OUTPUT_BASE,
                    NPU_DECODE_GEMV_SCALE_BASE,
                    flow);
            npu_decode_stream_unsupported_void(
                    op_name,
                    cur_weight_spm,
                    NPU_DECODE_GEMV_ACT_BASE,
                    static_cast<uint16_t>(k),
                    static_cast<uint16_t>(rows),
                    NPU_DECODE_GEMV_OUTPUT_BASE,
                    NPU_DECODE_GEMV_SCALE_BASE,
                    NPU_DECODE_GEMV_MODE_W8A16,
                    flow);
            if (collect_profile) {
                profile.gemv_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
                profile.gemv_calls += 1;
                profile.weight_bytes += weight_bytes;
                stage_start_us = ggml_time_us();
            }

            npu_decode_stream_unsupported_void(
                    op_name,
                    out_cma,
                    NPU_DECODE_GEMV_OUTPUT_BASE,
                    npu_decode_host_output_bytes(rows, hw_f32_mvout),
                    output_precision,
                    0);
            npu_decode_repro_dump_w8_block(
                    op_name,
                    col,
                    0,
                    row_base,
                    rows,
                    static_cast<uint32_t>(k),
                    cached.col_tiles,
                    npu_decode_ceil_div_u32(rows, NPU_DECODE_GEMV_M_TILE),
                    act_layout.act_bytes,
                    scale_bytes,
                    weight_bytes,
                    npu_decode_host_output_bytes(rows, hw_f32_mvout),
                    act_cma,
                    scale_mvin_ptr,
                    weight_mvin_ptr,
                    out_cma);
            if (collect_profile) {
                profile.mvout_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
                profile.mvout_calls += 1;
                profile.output_bytes += npu_decode_host_output_bytes(rows, hw_f32_mvout);
                stage_start_us = ggml_time_us();
            }

            char * dst_col = reinterpret_cast<char *>(dst_data) + col * dst_nb1;
            if (dst_type == GGML_TYPE_F16) {
                std::memcpy(
                        reinterpret_cast<ggml_fp16_t *>(dst_col) + row_base,
                        out_cma,
                        static_cast<size_t>(rows) * sizeof(ggml_fp16_t));
            } else if (hw_f32_mvout) {
                std::memcpy(
                        reinterpret_cast<float *>(dst_col) + row_base,
                        out_cma,
                        static_cast<size_t>(rows) * sizeof(float));
            } else {
                npu_decode_convert_output_fp16_to_f32(
                        static_cast<const ggml_fp16_t *>(out_cma),
                        reinterpret_cast<float *>(dst_col) + row_base,
                        rows);
            }
            if (collect_profile) {
                profile.postprocess_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            }
        }
    }

    const int64_t cleanup_start_us = collect_profile ? ggml_time_us() : 0;
    npu_decode_mem_free(act_cma);
    npu_decode_mem_free(weight_cma);
    npu_decode_mem_free(scale_cma);
    npu_decode_mem_free(out_cma);
    if (collect_profile) {
        profile.cleanup_us = static_cast<uint64_t>(ggml_time_us() - cleanup_start_us);
        profile.total_us = static_cast<uint64_t>(ggml_time_us() - total_start_us);
        npu_decode_profile_write(profile);
    }
    return true;
}
