
static bool npu_decode_make_gemv_layout(uint32_t m, uint32_t k, npu_decode_gemv_layout * layout) {
    if (m == 0 || k == 0 || layout == nullptr) {
        return false;
    }

    layout->row_tiles = npu_decode_ceil_div_u32(m, NPU_DECODE_GEMV_M_TILE);
    layout->col_tiles = npu_decode_ceil_div_u32(k, NPU_DECODE_GEMV_K_TILE);
    layout->act_bytes = layout->col_tiles * NPU_DECODE_GEMV_ACT_TILE_BYTES;
    layout->scale_bytes = layout->row_tiles * layout->col_tiles * NPU_DECODE_GEMV_SCALE_TILE_BYTES;
    layout->weight_base = NPU_DECODE_GEMV_PING_WEIGHT_BASE;
    layout->weight_bytes = layout->row_tiles * layout->col_tiles * NPU_DECODE_GEMV_WEIGHT_TILE_BYTES;
    layout->packed_bytes = npu_decode_align_up_u32(layout->weight_base + layout->weight_bytes, NPU_DECODE_GEMV_DMA_ALIGN);
    layout->output_bytes = npu_decode_align_up_u32(m * 2, NPU_DECODE_GEMV_LINE_BYTES);

    return npu_decode_act_payload_bytes(*layout) <= NPU_DECODE_GEMV_ACT_BUFFER_BYTES &&
           layout->scale_bytes <= NPU_DECODE_GEMV_SPM_BYTES &&
           layout->weight_base + layout->weight_bytes <= NPU_DECODE_GEMV_SPM_BYTES &&
           layout->output_bytes <= NPU_DECODE_GEMV_SPM_BYTES &&
           (layout->act_bytes % NPU_DECODE_GEMV_DMA_ALIGN) == 0 &&
           (layout->packed_bytes % NPU_DECODE_GEMV_DMA_ALIGN) == 0;
}

static bool npu_decode_make_pingpong_layout(uint32_t m, uint32_t k, npu_decode_gemv_layout * layout) {
    if (m == 0 || k == 0 || layout == nullptr) {
        return false;
    }

    layout->row_tiles = npu_decode_ceil_div_u32(m, NPU_DECODE_GEMV_M_TILE);
    layout->col_tiles = npu_decode_ceil_div_u32(k, NPU_DECODE_GEMV_K_TILE);
    layout->act_bytes = layout->col_tiles * NPU_DECODE_GEMV_ACT_TILE_BYTES;
    layout->scale_bytes = layout->row_tiles * layout->col_tiles * NPU_DECODE_GEMV_SCALE_TILE_BYTES;
    layout->weight_base = NPU_DECODE_GEMV_PING_WEIGHT_BASE;
    layout->weight_bytes = layout->row_tiles * layout->col_tiles * NPU_DECODE_GEMV_WEIGHT_TILE_BYTES;
    layout->packed_bytes = 0;
    layout->output_bytes = m * sizeof(ggml_fp16_t);

    const uint32_t bytes_per_row_tile = layout->col_tiles * NPU_DECODE_GEMV_WEIGHT_TILE_BYTES;
    const uint32_t ping_capacity = NPU_DECODE_GEMV_PONG_WEIGHT_BASE - NPU_DECODE_GEMV_PING_WEIGHT_BASE;
    const uint32_t pong_capacity = NPU_DECODE_GEMV_SPM_BYTES - NPU_DECODE_GEMV_PONG_WEIGHT_BASE;
    const uint32_t weight_capacity = std::min(ping_capacity, pong_capacity);
    return npu_decode_act_payload_bytes(*layout) <= NPU_DECODE_GEMV_ACT_BUFFER_BYTES &&
           layout->scale_bytes <= NPU_DECODE_GEMV_PING_WEIGHT_BASE &&
           layout->output_bytes <= NPU_DECODE_GEMV_SPM_BYTES &&
           layout->output_bytes <= std::numeric_limits<uint16_t>::max() + 1u &&
           bytes_per_row_tile > 0 &&
           bytes_per_row_tile <= weight_capacity &&
           (layout->act_bytes % NPU_DECODE_GEMV_DMA_ALIGN) == 0 &&
           (layout->weight_bytes % NPU_DECODE_GEMV_DMA_ALIGN) == 0;
}

static float npu_decode_read_typed_f32(const void * base, int type, int64_t byte_offset) {
    const char * ptr = static_cast<const char *>(base) + byte_offset;
    switch (type) {
        case GGML_TYPE_F32:
            return *reinterpret_cast<const float *>(ptr);
        case GGML_TYPE_F16:
            return ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t *>(ptr));
        default:
            return std::numeric_limits<float>::quiet_NaN();
    }
}

static bool npu_decode_validate_symmetric_zero(
        const void * zero_data,
        int zero_type,
        int64_t zero_nb0,
        int64_t zero_nb1,
        int64_t out_channels,
        int64_t groups,
        const char * op_name) {
    for (int64_t row = 0; row < out_channels; ++row) {
        for (int64_t group = 0; group < groups; ++group) {
            const float zero = npu_decode_read_typed_f32(zero_data, zero_type, row * zero_nb1 + group * zero_nb0);
            if (!(std::fabs(zero - 8.0f) <= 1.0e-3f)) {
                if (npu_decode_w4a16_debug_log_enabled()) {
                    GGML_LOG_WARN("%s: reject %s: non-symmetric AWQ zero row=%" PRId64 " group=%" PRId64 " zero=%g\n",
                            __func__, op_name != nullptr ? op_name : "(unnamed)", row, group, static_cast<double>(zero));
                }
                return false;
            }
        }
    }
    return true;
}

static uint8_t npu_decode_q4_unsigned_at(const uint8_t * row, int64_t k_idx) {
    const uint8_t packed = row[k_idx / 2];
    return (k_idx & 1) == 0 ? (packed & 0x0fu) : ((packed >> 4) & 0x0fu);
}

static int8_t npu_decode_q4_signed_at(const uint8_t * row, int64_t k_idx) {
    const uint8_t q = npu_decode_q4_unsigned_at(row, k_idx);
    return static_cast<int8_t>(q >= 8 ? static_cast<int>(q) - 16 : static_cast<int>(q));
}

static float npu_decode_read_packed_fp16(const uint8_t * ptr) {
    ggml_fp16_t value;
    std::memcpy(&value, ptr, sizeof(value));
    return ggml_fp16_to_fp32(value);
}

static void npu_decode_pack_act_scale(
        void * scale_dst,
        size_t scale_bytes,
        const npu_decode_gemv_layout & layout,
        const float * inv_smooth_scale,
        int64_t k) {
    std::memset(scale_dst, 0, scale_bytes);
    uint8_t * scale = static_cast<uint8_t *>(scale_dst);
    const ggml_fp16_t fp16_one = ggml_fp32_to_fp16(1.0f);
    for (uint32_t col_tile = 0; col_tile < layout.col_tiles; ++col_tile) {
        for (uint32_t lane = 0; lane < NPU_DECODE_GEMV_K_TILE; ++lane) {
            const int64_t in_idx = static_cast<int64_t>(col_tile) * NPU_DECODE_GEMV_K_TILE + lane;
            if (in_idx >= k) {
                continue;
            }
            ggml_fp16_t fp16_scale = fp16_one;
            if (inv_smooth_scale != nullptr) {
                fp16_scale = ggml_fp32_to_fp16(inv_smooth_scale[in_idx]);
            }
            std::memcpy(
                    scale + col_tile * NPU_DECODE_GEMV_ACT_TILE_BYTES + lane * sizeof(ggml_fp16_t),
                    &fp16_scale,
                    sizeof(fp16_scale));
        }
    }
}

static void npu_decode_pack_activation(
        void * act_dst,
        size_t act_payload_bytes,
        const npu_decode_gemv_layout & layout,
        const void * act_data,
        int act_type,
        int64_t act_nb0,
        int64_t act_nb1,
        const uint8_t * act_scale_packed,
        int64_t k,
        int64_t col) {
    std::memset(act_dst, 0, act_payload_bytes);
    uint8_t * act = static_cast<uint8_t *>(act_dst);
    if (act_type == GGML_TYPE_F16 && act_nb0 == static_cast<int64_t>(sizeof(ggml_fp16_t))) {
        const uint8_t * src_col = static_cast<const uint8_t *>(act_data) + col * act_nb1;
        for (uint32_t col_tile = 0; col_tile < layout.col_tiles; ++col_tile) {
            const int64_t in_idx = static_cast<int64_t>(col_tile) * NPU_DECODE_GEMV_K_TILE;
            if (in_idx >= k) {
                continue;
            }
            const size_t elems = static_cast<size_t>(
                    std::min<int64_t>(NPU_DECODE_GEMV_K_TILE, k - in_idx));
            std::memcpy(
                    act + col_tile * NPU_DECODE_GEMV_ACT_TILE_BYTES,
                    src_col + in_idx * act_nb0,
                    elems * sizeof(ggml_fp16_t));
        }
        if (act_scale_packed != nullptr) {
            std::memcpy(act + layout.act_bytes, act_scale_packed, layout.act_bytes);
        } else if (act_payload_bytes >= npu_decode_act_payload_bytes(layout)) {
            npu_decode_pack_act_scale(act + layout.act_bytes, layout.act_bytes, layout, nullptr, k);
        }
        return;
    }
    for (uint32_t col_tile = 0; col_tile < layout.col_tiles; ++col_tile) {
        for (uint32_t lane = 0; lane < NPU_DECODE_GEMV_K_TILE; ++lane) {
            const int64_t in_idx = static_cast<int64_t>(col_tile) * NPU_DECODE_GEMV_K_TILE + lane;
            if (in_idx >= k) {
                continue;
            }
            const float value = npu_decode_read_typed_f32(act_data, act_type, col * act_nb1 + in_idx * act_nb0);
            const ggml_fp16_t fp16_value = ggml_fp32_to_fp16(value);
            std::memcpy(act + col_tile * NPU_DECODE_GEMV_ACT_TILE_BYTES + lane * 2, &fp16_value, sizeof(fp16_value));
        }
    }
    if (act_scale_packed != nullptr) {
        std::memcpy(act + layout.act_bytes, act_scale_packed, layout.act_bytes);
    } else if (act_payload_bytes >= npu_decode_act_payload_bytes(layout)) {
        npu_decode_pack_act_scale(act + layout.act_bytes, layout.act_bytes, layout, nullptr, k);
    }
}

static void npu_decode_pack_activation_slice(
        void * act_dst,
        size_t act_payload_bytes,
        const npu_decode_gemv_layout & layout,
        const void * act_data,
        int act_type,
        int64_t act_nb0,
        int64_t act_nb1,
        int64_t k_total,
        int64_t k_offset,
        int64_t k_slice,
        int64_t col) {
    std::memset(act_dst, 0, act_payload_bytes);
    uint8_t * act = static_cast<uint8_t *>(act_dst);
    for (uint32_t col_tile = 0; col_tile < layout.col_tiles; ++col_tile) {
        for (uint32_t lane = 0; lane < NPU_DECODE_GEMV_K_TILE; ++lane) {
            const int64_t local_idx = static_cast<int64_t>(col_tile) * NPU_DECODE_GEMV_K_TILE + lane;
            const int64_t in_idx = k_offset + local_idx;
            if (local_idx >= k_slice || in_idx >= k_total) {
                continue;
            }
            ggml_fp16_t fp16_value;
            if (act_type == GGML_TYPE_F16 && act_nb0 == static_cast<int64_t>(sizeof(ggml_fp16_t))) {
                const uint8_t * src_col = static_cast<const uint8_t *>(act_data) + col * act_nb1;
                std::memcpy(&fp16_value, src_col + in_idx * act_nb0, sizeof(fp16_value));
            } else {
                const float value = npu_decode_read_typed_f32(act_data, act_type, col * act_nb1 + in_idx * act_nb0);
                fp16_value = ggml_fp32_to_fp16(value);
            }
            std::memcpy(act + col_tile * NPU_DECODE_GEMV_ACT_TILE_BYTES + lane * sizeof(ggml_fp16_t),
                    &fp16_value,
                    sizeof(fp16_value));
        }
    }
}

static uint32_t npu_decode_w8_act_mvin_bytes(uint32_t col_tiles) {
    const uint32_t payload_bytes = col_tiles * NPU_DECODE_GEMV_W8_ACT_TILE_BYTES;
    return npu_decode_align_up_u32(payload_bytes, NPU_DECODE_GEMV_DMA_ALIGN);
}

static void npu_decode_pack_w8_activation_slice(
        void * act_dst,
        size_t act_mvin_bytes,
        const npu_decode_gemv_layout & layout,
        const void * act_data,
        int act_type,
        int64_t act_nb0,
        int64_t act_nb1,
        int64_t k_total,
        int64_t k_offset,
        int64_t k_slice,
        int64_t col) {
    std::memset(act_dst, 0, act_mvin_bytes);
    uint8_t * act = static_cast<uint8_t *>(act_dst);
    for (uint32_t col_tile = 0; col_tile < layout.col_tiles; ++col_tile) {
        for (uint32_t lane = 0; lane < NPU_DECODE_GEMV_W8_K_TILE; ++lane) {
            const int64_t local_idx = static_cast<int64_t>(col_tile) * NPU_DECODE_GEMV_W8_K_TILE + lane;
            const int64_t in_idx = k_offset + local_idx;
            if (local_idx >= k_slice || in_idx >= k_total) {
                continue;
            }
            ggml_fp16_t fp16_value;
            if (act_type == GGML_TYPE_F16 && act_nb0 == static_cast<int64_t>(sizeof(ggml_fp16_t))) {
                const uint8_t * src_col = static_cast<const uint8_t *>(act_data) + col * act_nb1;
                std::memcpy(&fp16_value, src_col + in_idx * act_nb0, sizeof(fp16_value));
            } else {
                const float value = npu_decode_read_typed_f32(act_data, act_type, col * act_nb1 + in_idx * act_nb0);
                fp16_value = ggml_fp32_to_fp16(value);
            }
            std::memcpy(
                    act + col_tile * NPU_DECODE_GEMV_W8_ACT_TILE_BYTES + lane * sizeof(ggml_fp16_t),
                    &fp16_value,
                    sizeof(fp16_value));
        }
    }
}

static void npu_decode_store_sim_output(
        float value,
        void * dst_data,
        int dst_type,
        int64_t dst_nb1,
        int64_t col,
        int64_t row) {
    char * dst_col = static_cast<char *>(dst_data) + col * dst_nb1;
    const ggml_fp16_t fp16_value = ggml_fp32_to_fp16(value);
    if (dst_type == GGML_TYPE_F16) {
        std::memcpy(dst_col + row * sizeof(ggml_fp16_t), &fp16_value, sizeof(fp16_value));
    } else {
        reinterpret_cast<float *>(dst_col)[row] = ggml_fp16_to_fp32(fp16_value);
    }
}

static void npu_decode_simulate_packed_block(
        const uint8_t * act_packed,
        const uint8_t * packed,
        const npu_decode_gemv_layout & layout,
        int64_t row_base,
        int64_t block_rows,
        int64_t k,
        void * dst_data,
        int dst_type,
        int64_t dst_nb1,
        int64_t col) {
    for (int64_t row = 0; row < block_rows; ++row) {
        float acc = 0.0f;
        const uint32_t row_tile = static_cast<uint32_t>(row) / NPU_DECODE_GEMV_M_TILE;
        const uint32_t row_in_tile = static_cast<uint32_t>(row) % NPU_DECODE_GEMV_M_TILE;
        for (uint32_t col_tile = 0; col_tile < layout.col_tiles; ++col_tile) {
            const uint32_t tile_index = row_tile * layout.col_tiles + col_tile;
            const uint8_t * scale_line =
                packed + static_cast<size_t>(tile_index) * NPU_DECODE_GEMV_SCALE_TILE_BYTES;
            const float scale =
                npu_decode_read_packed_fp16(scale_line + row_in_tile * sizeof(ggml_fp16_t));
            const uint8_t * weight_row =
                packed + layout.weight_base +
                static_cast<size_t>(tile_index) * NPU_DECODE_GEMV_WEIGHT_TILE_BYTES +
                row_in_tile * NPU_DECODE_GEMV_WEIGHT_ROW_BYTES;
            const uint8_t * act_tile =
                act_packed + static_cast<size_t>(col_tile) * NPU_DECODE_GEMV_ACT_TILE_BYTES;
            const uint8_t * act_scale_tile =
                act_packed + layout.act_bytes + static_cast<size_t>(col_tile) * NPU_DECODE_GEMV_ACT_TILE_BYTES;
            for (uint32_t lane = 0; lane < NPU_DECODE_GEMV_K_TILE; ++lane) {
                const int64_t in_idx = static_cast<int64_t>(col_tile) * NPU_DECODE_GEMV_K_TILE + lane;
                if (in_idx >= k) {
                    continue;
                }
                const float act = npu_decode_read_packed_fp16(act_tile + lane * sizeof(ggml_fp16_t));
                const float act_scale = npu_decode_read_packed_fp16(act_scale_tile + lane * sizeof(ggml_fp16_t));
                const float w = static_cast<float>(npu_decode_q4_signed_at(weight_row, lane)) * scale;
                acc += act * act_scale * w;
            }
        }
        npu_decode_store_sim_output(acc, dst_data, dst_type, dst_nb1, col, row_base + row);
    }
}

static void npu_decode_simulate_pingpong(
        const uint8_t * act_packed,
        const uint8_t * scale_packed,
        const uint8_t * weight_packed,
        const npu_decode_gemv_layout & layout,
        int64_t out_channels,
        int64_t k,
        void * dst_data,
        int dst_type,
        int64_t dst_nb1,
        int64_t col) {
    for (int64_t row = 0; row < out_channels; ++row) {
        float acc = 0.0f;
        const uint32_t row_tile = static_cast<uint32_t>(row) / NPU_DECODE_GEMV_M_TILE;
        const uint32_t row_in_tile = static_cast<uint32_t>(row) % NPU_DECODE_GEMV_M_TILE;
        for (uint32_t col_tile = 0; col_tile < layout.col_tiles; ++col_tile) {
            const uint32_t tile_index = row_tile * layout.col_tiles + col_tile;
            const uint8_t * scale_line =
                scale_packed + static_cast<size_t>(tile_index) * NPU_DECODE_GEMV_SCALE_TILE_BYTES;
            const float scale =
                npu_decode_read_packed_fp16(scale_line + row_in_tile * sizeof(ggml_fp16_t));
            const uint8_t * weight_row =
                weight_packed +
                static_cast<size_t>(tile_index) * NPU_DECODE_GEMV_WEIGHT_TILE_BYTES +
                row_in_tile * NPU_DECODE_GEMV_WEIGHT_ROW_BYTES;
            const uint8_t * act_tile =
                act_packed + static_cast<size_t>(col_tile) * NPU_DECODE_GEMV_ACT_TILE_BYTES;
            const uint8_t * act_scale_tile =
                act_packed + layout.act_bytes + static_cast<size_t>(col_tile) * NPU_DECODE_GEMV_ACT_TILE_BYTES;
            for (uint32_t lane = 0; lane < NPU_DECODE_GEMV_K_TILE; ++lane) {
                const int64_t in_idx = static_cast<int64_t>(col_tile) * NPU_DECODE_GEMV_K_TILE + lane;
                if (in_idx >= k) {
                    continue;
                }
                const float act = npu_decode_read_packed_fp16(act_tile + lane * sizeof(ggml_fp16_t));
                const float act_scale = npu_decode_read_packed_fp16(act_scale_tile + lane * sizeof(ggml_fp16_t));
                const float w = static_cast<float>(npu_decode_q4_signed_at(weight_row, lane)) * scale;
                acc += act * act_scale * w;
            }
        }
        npu_decode_store_sim_output(acc, dst_data, dst_type, dst_nb1, col, row);
    }
}

static MvinConfig npu_decode_make_mvin_cfg(
        void * host_ptr,
        uint32_t sram_addr,
        uint32_t bytes,
        uint8_t input_type,
        uint8_t precision) {
    MvinConfig cfg = {};
    cfg.host_ptr = host_ptr;
    cfg.sram_addr = sram_addr;
    cfg.col_num = bytes - 1u;
    cfg.row_num = 0;
    cfg.sram_stride = 0;
    cfg.dram_stride = 0;
    cfg.precision = precision;
    cfg.input_type = input_type;
    cfg.dest = false;
    cfg.is_bias = false;
    cfg.is_quant = false;
    cfg.quant_zero = 0;
    cfg.quant_scale = 0;
    cfg.quant_shift = 0;
    return cfg;
}

static uint64_t npu_decode_make_bypass_output_flow(uint16_t elem_count) {
    return npu_decode_stream_unsupported_flow(
            NPU_DECODE_SRC_GEMV_STREAM,
            0,
            NPU_DECODE_UNARY_BYPASS,
            NPU_DECODE_BINARY_BYPASS,
            NPU_DECODE_REDUCE_BYPASS,
            NPU_DECODE_DST_OUTPUT_SPM,
            0,
            0,
            elem_count,
            0);
}

static void npu_decode_scale_packed_row_scales(
        uint8_t * scale_packed,
        const npu_decode_gemv_layout & layout,
        int64_t row_base,
        const std::vector<float> & row_scale) {
    for (uint32_t row_tile = 0; row_tile < layout.row_tiles; ++row_tile) {
        for (uint32_t col_tile = 0; col_tile < layout.col_tiles; ++col_tile) {
            const size_t scale_line =
                static_cast<size_t>(row_tile * layout.col_tiles + col_tile) *
                NPU_DECODE_GEMV_SCALE_TILE_BYTES;
            for (uint32_t lane = 0; lane < NPU_DECODE_GEMV_M_TILE; ++lane) {
                const int64_t row = static_cast<int64_t>(row_tile) * NPU_DECODE_GEMV_M_TILE + lane;
                const int64_t global_row = row_base + row;
                if (global_row < 0 || static_cast<size_t>(global_row) >= row_scale.size()) {
                    continue;
                }
                uint8_t * ptr = scale_packed + scale_line + lane * sizeof(ggml_fp16_t);
                const float scale = npu_decode_read_packed_fp16(ptr) * row_scale[static_cast<size_t>(global_row)];
                const ggml_fp16_t scale_fp16 = ggml_fp32_to_fp16(scale);
                std::memcpy(ptr, &scale_fp16, sizeof(scale_fp16));
            }
        }
    }
}

static uint32_t npu_decode_pingpong_max_block_rows(const npu_decode_gemv_layout & layout) {
    const uint32_t bytes_per_row_tile = layout.col_tiles * NPU_DECODE_GEMV_WEIGHT_TILE_BYTES;
    const uint32_t ping_capacity = NPU_DECODE_GEMV_PONG_WEIGHT_BASE - NPU_DECODE_GEMV_PING_WEIGHT_BASE;
    const uint32_t pong_capacity = NPU_DECODE_GEMV_SPM_BYTES - NPU_DECODE_GEMV_PONG_WEIGHT_BASE;
    const uint32_t weight_capacity = std::min(ping_capacity, pong_capacity);
    const uint32_t max_row_tiles = bytes_per_row_tile == 0 ? 0 : weight_capacity / bytes_per_row_tile;
    return max_row_tiles * NPU_DECODE_GEMV_M_TILE;
}

static uint32_t npu_decode_w16a16_pingpong_max_block_rows(uint32_t col_tiles) {
    const uint32_t bytes_per_row_tile = col_tiles * NPU_DECODE_GEMV_W16_WEIGHT_TILE_BYTES;
    const uint32_t ping_capacity = NPU_DECODE_GEMV_PONG_WEIGHT_BASE - NPU_DECODE_GEMV_PING_WEIGHT_BASE;
    const uint32_t pong_capacity = NPU_DECODE_GEMV_SPM_BYTES - NPU_DECODE_GEMV_PONG_WEIGHT_BASE;
    const uint32_t weight_capacity = std::min(ping_capacity, pong_capacity);
    const uint32_t max_row_tiles = bytes_per_row_tile == 0 ? 0 : weight_capacity / bytes_per_row_tile;
    return max_row_tiles * NPU_DECODE_GEMV_M_TILE;
}

static void npu_decode_gemv_pingpong_manual_run(
        const char * op_name,
        void * act_ptr,
        void * scale_ptr,
        void * weight_ptr,
        void * output_ptr,
        const npu_decode_gemv_layout & layout,
        uint16_t m,
        uint16_t k,
        uint8_t output_precision,
        npu_decode_gemv_profile_record * profile) {
    const uint32_t max_block_rows = npu_decode_pingpong_max_block_rows(layout);
    if (max_block_rows == 0) {
        throw std::runtime_error("manual GEMV pingpong: no row block fits");
    }
    if (output_precision != 1 && output_precision != 3) {
        throw std::runtime_error("manual GEMV pingpong: output precision must be FP16 or FP32");
    }
    const bool block_mvout = npu_decode_pingpong_block_mvout_enabled();
    const uint32_t output_elem_bytes = output_precision == 3 ? sizeof(float) : sizeof(ggml_fp16_t);
    if (npu_decode_w4a16_debug_log_enabled()) {
        GGML_LOG_WARN("%s: manual pingpong M=%u K=%u blocks<=%u block_mvout=%d out_precision=%u"
                " act=0x%05X scale=0x%05X out=0x%05X pingW=0x%05X pongW=0x%05X\n",
                __func__,
                static_cast<unsigned>(m),
                static_cast<unsigned>(k),
                static_cast<unsigned>((static_cast<uint32_t>(m) + max_block_rows - 1u) / max_block_rows),
                block_mvout ? 1 : 0,
                static_cast<unsigned>(output_precision),
                NPU_DECODE_GEMV_ACT_BASE,
                NPU_DECODE_GEMV_SCALE_BASE,
                NPU_DECODE_GEMV_OUTPUT_BASE,
                NPU_DECODE_GEMV_PING_WEIGHT_BASE,
                NPU_DECODE_GEMV_PONG_WEIGHT_BASE);
    }
    const uint32_t bytes_per_row_tile = layout.col_tiles * NPU_DECODE_GEMV_WEIGHT_TILE_BYTES;
    auto block_rows_for = [&](uint32_t row_base) {
        return std::min<uint32_t>(max_block_rows, static_cast<uint32_t>(m) - row_base);
    };
    auto weight_offset_for = [&](uint32_t row_base) {
        return (row_base / NPU_DECODE_GEMV_M_TILE) * bytes_per_row_tile;
    };
    auto scale_addr_for = [&](uint32_t row_base) {
        return NPU_DECODE_GEMV_SCALE_BASE +
               (row_base / NPU_DECODE_GEMV_M_TILE) * layout.col_tiles * NPU_DECODE_GEMV_SCALE_TILE_BYTES;
    };

    npu_decode_stream_unsupported_void(
            op_name,
            act_ptr,
            NPU_DECODE_GEMV_ACT_BASE,
            npu_decode_act_payload_bytes(layout),
            2,
            NPU_DECODE_GEMV_INPUT_TYPE_ACT);
    int64_t stage_start_us = profile != nullptr ? ggml_time_us() : 0;
    npu_decode_stream_unsupported_void(
            op_name,
            scale_ptr,
            NPU_DECODE_GEMV_SCALE_BASE,
            layout.scale_bytes,
            2,
            NPU_DECODE_GEMV_INPUT_TYPE_DATA,
            0,
            0);
    if (profile != nullptr) {
        profile->scale_mvin_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
        profile->scale_mvin_calls += 1;
        profile->scale_bytes += layout.scale_bytes;
    }

    uint8_t * weight_base = static_cast<uint8_t *>(weight_ptr);
    const uint32_t first_rows = block_rows_for(0);
    const uint32_t first_weight_bytes =
        npu_decode_ceil_div_u32(first_rows, NPU_DECODE_GEMV_M_TILE) * bytes_per_row_tile;
    stage_start_us = profile != nullptr ? ggml_time_us() : 0;
    npu_decode_stream_unsupported_void(
            op_name,
            weight_base,
            NPU_DECODE_GEMV_PING_WEIGHT_BASE,
            first_weight_bytes,
            1,
            NPU_DECODE_GEMV_INPUT_TYPE_WEIGHT);
    if (profile != nullptr) {
        profile->pingpong_initial_mvin_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
        profile->pingpong_initial_mvin_calls += 1;
        profile->weight_bytes += first_weight_bytes;
    }

    uint32_t block_idx = 0;
    for (uint32_t row_base = 0; row_base < static_cast<uint32_t>(m);
            row_base += block_rows_for(row_base), ++block_idx) {
        const uint32_t rows = block_rows_for(row_base);
        const uint32_t cur_weight_spm =
            (block_idx & 1u) ? NPU_DECODE_GEMV_PONG_WEIGHT_BASE : NPU_DECODE_GEMV_PING_WEIGHT_BASE;

        bool prefetch_started = false;
        int64_t prefetch_issue_done_us = 0;
        const uint32_t next_row_base = row_base + rows;
        if (next_row_base < static_cast<uint32_t>(m)) {
            const uint32_t next_rows = block_rows_for(next_row_base);
            const uint32_t next_weight_bytes =
                npu_decode_ceil_div_u32(next_rows, NPU_DECODE_GEMV_M_TILE) * bytes_per_row_tile;
            const uint32_t next_weight_spm =
                ((block_idx + 1u) & 1u) ? NPU_DECODE_GEMV_PONG_WEIGHT_BASE : NPU_DECODE_GEMV_PING_WEIGHT_BASE;
            MvinConfig next_mvin = npu_decode_make_mvin_cfg(
                    weight_base + weight_offset_for(next_row_base),
                    next_weight_spm,
                    next_weight_bytes,
                    NPU_DECODE_GEMV_INPUT_TYPE_WEIGHT,
                    1);
            stage_start_us = profile != nullptr ? ggml_time_us() : 0;
            npu_decode_stream_unsupported_void(0, &next_mvin);
            if (profile != nullptr) {
                prefetch_issue_done_us = ggml_time_us();
                profile->pingpong_prefetch_issue_us += static_cast<uint64_t>(prefetch_issue_done_us - stage_start_us);
                profile->pingpong_async_mvin_calls += 1;
                profile->weight_bytes += next_weight_bytes;
            }
            prefetch_started = true;
        }

        const uint16_t output_addr = static_cast<uint16_t>(
                NPU_DECODE_GEMV_OUTPUT_BASE + (block_mvout ? 0u : row_base * sizeof(ggml_fp16_t)));
        const uint16_t scale_addr = static_cast<uint16_t>(scale_addr_for(row_base));
        if (npu_decode_w4a16_debug_log_enabled()) {
            GGML_LOG_WARN("%s: block=%u row_base=%u rows=%u W=0x%05X Wbytes=%u"
                    " O=0x%04X S=0x%04X K=%u prefetch=%d\n",
                    __func__,
                    static_cast<unsigned>(block_idx),
                    static_cast<unsigned>(row_base),
                    static_cast<unsigned>(rows),
                    static_cast<unsigned>(cur_weight_spm),
                    static_cast<unsigned>(npu_decode_ceil_div_u32(rows, NPU_DECODE_GEMV_M_TILE) * bytes_per_row_tile),
                    static_cast<unsigned>(output_addr),
                    static_cast<unsigned>(scale_addr),
                    static_cast<unsigned>(k),
                    prefetch_started ? 1 : 0);
        }

        const uint64_t flow = npu_decode_make_bypass_output_flow(static_cast<uint16_t>(rows));
        npu_decode_trace_stream_flow(
                "manual_pingpong",
                op_name,
                block_idx,
                row_base,
                rows,
                k,
                NPU_DECODE_GEMV_MODE_W4A16,
                cur_weight_spm,
                NPU_DECODE_GEMV_ACT_BASE,
                output_addr,
                scale_addr,
                flow);
        stage_start_us = profile != nullptr ? ggml_time_us() : 0;
        npu_decode_stream_unsupported_void(
                op_name,
                cur_weight_spm,
                NPU_DECODE_GEMV_ACT_BASE,
                k,
                static_cast<uint16_t>(rows),
                output_addr,
                scale_addr,
                NPU_DECODE_GEMV_MODE_W4A16,
                flow,
                NPU_DECODE_GEMV_ACT_BASE + layout.act_bytes);
        if (profile != nullptr) {
            profile->pingpong_gemv_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            profile->pingpong_block_calls += 1;
        }

        if (prefetch_started) {
            if (profile != nullptr) {
                profile->pingpong_prefetch_window_us +=
                    static_cast<uint64_t>(ggml_time_us() - prefetch_issue_done_us);
                stage_start_us = ggml_time_us();
            }
            npu_decode_stream_unsupported_void(1u << 0);
            if (profile != nullptr) {
                profile->pingpong_prefetch_wait_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            }
        }

        if (block_mvout) {
            stage_start_us = profile != nullptr ? ggml_time_us() : 0;
            npu_decode_stream_unsupported_void(
                    op_name,
                    static_cast<uint8_t *>(output_ptr) + row_base * output_elem_bytes,
                    NPU_DECODE_GEMV_OUTPUT_BASE,
                    rows * output_elem_bytes,
                    output_precision,
                    0);
            if (profile != nullptr) {
                profile->pingpong_mvout_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
                profile->mvout_calls += 1;
            }
        }
    }

    if (!block_mvout) {
        stage_start_us = profile != nullptr ? ggml_time_us() : 0;
        npu_decode_stream_unsupported_void(
                op_name,
                output_ptr,
                NPU_DECODE_GEMV_OUTPUT_BASE,
                static_cast<uint32_t>(m) * output_elem_bytes,
                output_precision,
                0);
        if (profile != nullptr) {
            profile->pingpong_mvout_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            profile->mvout_calls += 1;
        }
    }
}

static void npu_decode_w16a16_pingpong_manual_run(
        const char * op_name,
        void * act_ptr,
        void * scale_ptr,
        void * weight_ptr,
        void * output_ptr,
        uint32_t col_tiles,
        uint32_t bytes_per_row_tile,
        uint32_t max_block_rows,
        uint16_t m,
        uint16_t k,
        uint8_t output_precision,
        npu_decode_gemv_profile_record * profile) {
    if (max_block_rows == 0) {
        throw std::runtime_error("manual W16A16 pingpong: no row block fits");
    }
    if (output_precision != 1 && output_precision != 3) {
        throw std::runtime_error("manual W16A16 pingpong: output precision must be FP16 or FP32");
    }
    const uint32_t act_bytes = col_tiles * NPU_DECODE_GEMV_ACT_TILE_BYTES;
    const uint32_t scale_bytes = npu_decode_ceil_div_u32(max_block_rows, NPU_DECODE_GEMV_M_TILE) *
        col_tiles * NPU_DECODE_GEMV_SCALE_TILE_BYTES;
    const uint32_t output_elem_bytes = output_precision == 3 ? sizeof(float) : sizeof(ggml_fp16_t);
    auto block_rows_for = [&](uint32_t row_base) {
        return std::min<uint32_t>(max_block_rows, static_cast<uint32_t>(m) - row_base);
    };
    auto weight_offset_for = [&](uint32_t row_base) {
        return (row_base / NPU_DECODE_GEMV_M_TILE) * bytes_per_row_tile;
    };

    if (npu_decode_w4a16_debug_log_enabled()) {
        GGML_LOG_WARN("%s: manual W16 pingpong M=%u K=%u blocks<=%u out_precision=%u"
                " act=0x%05X scale=0x%05X out=0x%05X pingW=0x%05X pongW=0x%05X\n",
                __func__,
                static_cast<unsigned>(m),
                static_cast<unsigned>(k),
                static_cast<unsigned>((static_cast<uint32_t>(m) + max_block_rows - 1u) / max_block_rows),
                static_cast<unsigned>(output_precision),
                NPU_DECODE_GEMV_ACT_BASE,
                NPU_DECODE_GEMV_SCALE_BASE,
                NPU_DECODE_GEMV_OUTPUT_BASE,
                NPU_DECODE_GEMV_PING_WEIGHT_BASE,
                NPU_DECODE_GEMV_PONG_WEIGHT_BASE);
    }

    npu_decode_stream_unsupported_void(
            act_ptr,
            NPU_DECODE_GEMV_ACT_BASE,
            act_bytes - 1u,
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
            scale_ptr,
            NPU_DECODE_GEMV_SCALE_BASE,
            scale_bytes - 1u,
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

    uint8_t * weight_base = static_cast<uint8_t *>(weight_ptr);
    const uint32_t first_rows = block_rows_for(0);
    const uint32_t first_weight_bytes =
        npu_decode_ceil_div_u32(first_rows, NPU_DECODE_GEMV_M_TILE) * bytes_per_row_tile;
    int64_t stage_start_us = profile != nullptr ? ggml_time_us() : 0;
    npu_decode_stream_unsupported_void(
            weight_base,
            NPU_DECODE_GEMV_PING_WEIGHT_BASE,
            first_weight_bytes - 1u,
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
    if (profile != nullptr) {
        profile->pingpong_initial_mvin_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
        profile->pingpong_initial_mvin_calls += 1;
        profile->weight_bytes += first_weight_bytes;
    }

    uint32_t block_idx = 0;
    for (uint32_t row_base = 0; row_base < static_cast<uint32_t>(m);
            row_base += block_rows_for(row_base), ++block_idx) {
        const uint32_t rows = block_rows_for(row_base);
        const uint32_t cur_weight_spm =
            (block_idx & 1u) ? NPU_DECODE_GEMV_PONG_WEIGHT_BASE : NPU_DECODE_GEMV_PING_WEIGHT_BASE;

        bool prefetch_started = false;
        int64_t prefetch_issue_done_us = 0;
        const uint32_t next_row_base = row_base + rows;
        if (next_row_base < static_cast<uint32_t>(m)) {
            const uint32_t next_rows = block_rows_for(next_row_base);
            const uint32_t next_weight_bytes =
                npu_decode_ceil_div_u32(next_rows, NPU_DECODE_GEMV_M_TILE) * bytes_per_row_tile;
            const uint32_t next_weight_spm =
                ((block_idx + 1u) & 1u) ? NPU_DECODE_GEMV_PONG_WEIGHT_BASE : NPU_DECODE_GEMV_PING_WEIGHT_BASE;
            MvinConfig next_mvin = npu_decode_make_mvin_cfg(
                    weight_base + weight_offset_for(next_row_base),
                    next_weight_spm,
                    next_weight_bytes,
                    NPU_DECODE_GEMV_INPUT_TYPE_WEIGHT,
                    2);
            stage_start_us = profile != nullptr ? ggml_time_us() : 0;
            npu_decode_stream_unsupported_void(0, &next_mvin);
            if (profile != nullptr) {
                prefetch_issue_done_us = ggml_time_us();
                profile->pingpong_prefetch_issue_us += static_cast<uint64_t>(prefetch_issue_done_us - stage_start_us);
                profile->pingpong_async_mvin_calls += 1;
                profile->weight_bytes += next_weight_bytes;
            }
            prefetch_started = true;
        }

        if (npu_decode_w4a16_debug_log_enabled() && (block_idx == 0 || row_base + rows >= static_cast<uint32_t>(m))) {
            GGML_LOG_WARN("%s: block %s idx=%u row_base=%u rows=%u W=0x%05X Wbytes=%u K=%u prefetch=%d\n",
                    __func__,
                    op_name != nullptr ? op_name : "(unnamed)",
                    block_idx,
                    row_base,
                    rows,
                    cur_weight_spm,
                    npu_decode_ceil_div_u32(rows, NPU_DECODE_GEMV_M_TILE) * bytes_per_row_tile,
                    k,
                    prefetch_started ? 1 : 0);
        }

        const uint64_t flow = npu_decode_make_bypass_output_flow(static_cast<uint16_t>(rows));
        npu_decode_trace_stream_flow(
                "lm_head_w16_pingpong",
                op_name,
                block_idx,
                row_base,
                rows,
                k,
                NPU_DECODE_GEMV_MODE_W16A16,
                cur_weight_spm,
                NPU_DECODE_GEMV_ACT_BASE,
                NPU_DECODE_GEMV_OUTPUT_BASE,
                NPU_DECODE_GEMV_SCALE_BASE,
                flow);
        stage_start_us = profile != nullptr ? ggml_time_us() : 0;
        npu_decode_stream_unsupported_void(
                op_name,
                cur_weight_spm,
                NPU_DECODE_GEMV_ACT_BASE,
                k,
                static_cast<uint16_t>(rows),
                NPU_DECODE_GEMV_OUTPUT_BASE,
                NPU_DECODE_GEMV_SCALE_BASE,
                NPU_DECODE_GEMV_MODE_W16A16,
                flow);
        if (profile != nullptr) {
            profile->pingpong_gemv_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            profile->pingpong_block_calls += 1;
        }

        if (prefetch_started) {
            if (profile != nullptr) {
                profile->pingpong_prefetch_window_us +=
                    static_cast<uint64_t>(ggml_time_us() - prefetch_issue_done_us);
                stage_start_us = ggml_time_us();
            }
            npu_decode_stream_unsupported_void(1u << 0);
            if (profile != nullptr) {
                profile->pingpong_prefetch_wait_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            }
        }

        stage_start_us = profile != nullptr ? ggml_time_us() : 0;
        npu_decode_stream_unsupported_void(
                static_cast<uint8_t *>(output_ptr) + static_cast<size_t>(row_base) * output_elem_bytes,
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
        if (profile != nullptr) {
            profile->pingpong_mvout_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            profile->mvout_calls += 1;
            profile->output_bytes += npu_decode_host_output_bytes(rows, output_precision == 3);
        }
    }
}

static void npu_decode_w8a16_pingpong_manual_run(
        const char * op_name,
        void * act_ptr,
        void * scale_ptr,
        void * weight_ptr,
        void * output_ptr,
        uint32_t col_tiles,
        uint32_t bytes_per_row_tile,
        uint32_t max_block_rows,
        uint16_t m,
        uint16_t k,
        uint8_t output_precision,
        npu_decode_gemv_profile_record * profile) {
    if (max_block_rows == 0) {
        throw std::runtime_error("manual W8A16 pingpong: no row block fits");
    }
    if (output_precision != 1 && output_precision != 3) {
        throw std::runtime_error("manual W8A16 pingpong: output precision must be FP16 or FP32");
    }

    const uint32_t act_bytes = npu_decode_w8_act_mvin_bytes(col_tiles);
    const uint32_t scale_bytes_per_row_tile = col_tiles * NPU_DECODE_GEMV_SCALE_TILE_BYTES;
    const uint32_t output_elem_bytes = output_precision == 3 ? sizeof(float) : sizeof(ggml_fp16_t);
    auto block_rows_for = [&](uint32_t row_base) {
        return std::min<uint32_t>(max_block_rows, static_cast<uint32_t>(m) - row_base);
    };
    auto weight_offset_for = [&](uint32_t row_base) {
        return (row_base / NPU_DECODE_GEMV_M_TILE) * bytes_per_row_tile;
    };
    auto scale_offset_for = [&](uint32_t row_base) {
        return (row_base / NPU_DECODE_GEMV_M_TILE) * scale_bytes_per_row_tile;
    };
    auto scale_bytes_for_rows = [&](uint32_t rows) {
        return npu_decode_ceil_div_u32(rows, NPU_DECODE_GEMV_M_TILE) * scale_bytes_per_row_tile;
    };

    if (npu_decode_w4a16_debug_log_enabled()) {
        GGML_LOG_WARN("%s: manual W8 pingpong M=%u K=%u blocks<=%u out_precision=%u"
                " act=0x%05X scale=0x%05X out=0x%05X pingW=0x%05X pongW=0x%05X\n",
                __func__,
                static_cast<unsigned>(m),
                static_cast<unsigned>(k),
                static_cast<unsigned>((static_cast<uint32_t>(m) + max_block_rows - 1u) / max_block_rows),
                static_cast<unsigned>(output_precision),
                NPU_DECODE_GEMV_ACT_BASE,
                NPU_DECODE_GEMV_SCALE_BASE,
                NPU_DECODE_GEMV_OUTPUT_BASE,
                NPU_DECODE_GEMV_PING_WEIGHT_BASE,
                NPU_DECODE_GEMV_PONG_WEIGHT_BASE);
    }

    npu_decode_stream_unsupported_void(
            op_name,
            act_ptr,
            NPU_DECODE_GEMV_ACT_BASE,
            act_bytes,
            2,
            NPU_DECODE_GEMV_INPUT_TYPE_ACT);

    uint8_t * weight_base = static_cast<uint8_t *>(weight_ptr);
    uint8_t * scale_base = static_cast<uint8_t *>(scale_ptr);
    const uint32_t first_rows = block_rows_for(0);
    const uint32_t first_weight_bytes =
        npu_decode_ceil_div_u32(first_rows, NPU_DECODE_GEMV_M_TILE) * bytes_per_row_tile;
    int64_t stage_start_us = profile != nullptr ? ggml_time_us() : 0;
    npu_decode_stream_unsupported_void(
            op_name,
            weight_base,
            NPU_DECODE_GEMV_PING_WEIGHT_BASE,
            first_weight_bytes,
            1,
            NPU_DECODE_GEMV_INPUT_TYPE_WEIGHT);
    if (profile != nullptr) {
        profile->pingpong_initial_mvin_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
        profile->pingpong_initial_mvin_calls += 1;
        profile->weight_bytes += first_weight_bytes;
    }

    uint32_t block_idx = 0;
    for (uint32_t row_base = 0; row_base < static_cast<uint32_t>(m);
            row_base += block_rows_for(row_base), ++block_idx) {
        const uint32_t rows = block_rows_for(row_base);
        const uint32_t cur_weight_spm =
            (block_idx & 1u) ? NPU_DECODE_GEMV_PONG_WEIGHT_BASE : NPU_DECODE_GEMV_PING_WEIGHT_BASE;
        const uint32_t scale_bytes = scale_bytes_for_rows(rows);
        stage_start_us = profile != nullptr ? ggml_time_us() : 0;
        npu_decode_stream_unsupported_void(
                op_name,
                scale_base + scale_offset_for(row_base),
                NPU_DECODE_GEMV_SCALE_BASE,
                scale_bytes,
                2,
                NPU_DECODE_GEMV_INPUT_TYPE_DATA,
                0,
                0);
        if (profile != nullptr) {
            profile->scale_mvin_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            profile->scale_mvin_calls += 1;
            profile->scale_bytes += scale_bytes;
        }
        const uint16_t scale_addr = static_cast<uint16_t>(NPU_DECODE_GEMV_SCALE_BASE);

        bool prefetch_started = false;
        int64_t prefetch_issue_done_us = 0;
        const uint32_t next_row_base = row_base + rows;
        if (next_row_base < static_cast<uint32_t>(m)) {
            const uint32_t next_rows = block_rows_for(next_row_base);
            const uint32_t next_weight_bytes =
                npu_decode_ceil_div_u32(next_rows, NPU_DECODE_GEMV_M_TILE) * bytes_per_row_tile;
            const uint32_t next_weight_spm =
                ((block_idx + 1u) & 1u) ? NPU_DECODE_GEMV_PONG_WEIGHT_BASE : NPU_DECODE_GEMV_PING_WEIGHT_BASE;
            MvinConfig next_mvin = npu_decode_make_mvin_cfg(
                    weight_base + weight_offset_for(next_row_base),
                    next_weight_spm,
                    next_weight_bytes,
                    NPU_DECODE_GEMV_INPUT_TYPE_WEIGHT,
                    1);
            stage_start_us = profile != nullptr ? ggml_time_us() : 0;
            npu_decode_stream_unsupported_void(0, &next_mvin);
            if (profile != nullptr) {
                prefetch_issue_done_us = ggml_time_us();
                profile->pingpong_prefetch_issue_us += static_cast<uint64_t>(prefetch_issue_done_us - stage_start_us);
                profile->pingpong_async_mvin_calls += 1;
                profile->weight_bytes += next_weight_bytes;
            }
            prefetch_started = true;
        }

        if (npu_decode_w4a16_debug_log_enabled() && (block_idx == 0 || row_base + rows >= static_cast<uint32_t>(m))) {
            GGML_LOG_WARN("%s: block %s idx=%u row_base=%u rows=%u W=0x%05X Wbytes=%u"
                    " S=0x%04X Sbytes=%u K=%u prefetch=%d\n",
                    __func__,
                    op_name != nullptr ? op_name : "(unnamed)",
                    block_idx,
                    row_base,
                    rows,
                    cur_weight_spm,
                    npu_decode_ceil_div_u32(rows, NPU_DECODE_GEMV_M_TILE) * bytes_per_row_tile,
                    scale_addr,
                    scale_bytes,
                    k,
                    prefetch_started ? 1 : 0);
        }

        const uint64_t flow = npu_decode_make_bypass_output_flow(static_cast<uint16_t>(rows));
        npu_decode_trace_stream_flow(
                "lm_head_w8_pingpong",
                op_name,
                block_idx,
                row_base,
                rows,
                k,
                NPU_DECODE_GEMV_MODE_W8A16,
                cur_weight_spm,
                NPU_DECODE_GEMV_ACT_BASE,
                NPU_DECODE_GEMV_OUTPUT_BASE,
                scale_addr,
                flow);
        stage_start_us = profile != nullptr ? ggml_time_us() : 0;
        npu_decode_stream_unsupported_void(
                op_name,
                cur_weight_spm,
                NPU_DECODE_GEMV_ACT_BASE,
                k,
                static_cast<uint16_t>(rows),
                NPU_DECODE_GEMV_OUTPUT_BASE,
                scale_addr,
                NPU_DECODE_GEMV_MODE_W8A16,
                flow);
        if (profile != nullptr) {
            const uint64_t gemv_us = static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            profile->gemv_us += gemv_us;
            profile->gemv_calls += 1;
            profile->pingpong_gemv_us += gemv_us;
            profile->pingpong_block_calls += 1;
        }

        if (prefetch_started) {
            if (profile != nullptr) {
                profile->pingpong_prefetch_window_us +=
                    static_cast<uint64_t>(ggml_time_us() - prefetch_issue_done_us);
                stage_start_us = ggml_time_us();
            }
            npu_decode_stream_unsupported_void(1u << 0);
            if (profile != nullptr) {
                profile->pingpong_prefetch_wait_us += static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            }
        }

        stage_start_us = profile != nullptr ? ggml_time_us() : 0;
        npu_decode_stream_unsupported_void(
                op_name,
                static_cast<uint8_t *>(output_ptr) + static_cast<size_t>(row_base) * output_elem_bytes,
                NPU_DECODE_GEMV_OUTPUT_BASE,
                rows * output_elem_bytes,
                output_precision,
                0);
        if (profile != nullptr) {
            const uint64_t mvout_us = static_cast<uint64_t>(ggml_time_us() - stage_start_us);
            profile->pingpong_mvout_us += mvout_us;
            profile->mvout_us += mvout_us;
            profile->mvout_calls += 1;
            profile->output_bytes += npu_decode_host_output_bytes(rows, output_precision == 3);
        }
    }
}

static void npu_decode_convert_output_fp16_to_f32(
        const ggml_fp16_t * src,
        float * dst,
        uint32_t rows) {
    ggml_fp16_to_fp32_row(src, dst, rows);
}

static void npu_decode_repro_dump_pingpong(
        const char * op_name,
        int64_t col,
        const npu_decode_gemv_layout & layout,
        uint16_t m,
        uint16_t k,
        uint8_t output_precision,
        bool block_mvout,
        const void * act_ptr,
        const void * scale_ptr,
        const void * weight_ptr,
        const void * output_ptr,
        uint32_t host_output_bytes,
        const void * cma_base,
        uint32_t cma_size) {
    if (!npu_decode_repro_dump_match(op_name)) {
        return;
    }

    static std::mutex mutex;
    static int dumped = 0;
    std::lock_guard<std::mutex> lock(mutex);
    if (dumped >= npu_decode_repro_dump_limit()) {
        return;
    }

    const char * dump_dir = npu_decode_repro_dump_dir();
    if (dump_dir == nullptr) {
        return;
    }
    const int dump_idx = dumped++;
    std::ostringstream prefix;
    prefix << dump_dir << "/repro_" << dump_idx << "_" << npu_decode_repro_sanitize(op_name);

    const std::string base = prefix.str();
    const std::string meta_path = base + ".meta.txt";
    const std::string act_path = base + ".act.bin";
    const std::string scale_path = base + ".scale.bin";
    const std::string weight_path = base + ".weight.bin";
    const std::string output_path = base + ".output.bin";

    const uint32_t act_payload_bytes = npu_decode_act_payload_bytes(layout);
    auto cma_offset_of = [cma_base, cma_size](const void * ptr) -> uint64_t {
        if (cma_base == nullptr || ptr == nullptr || cma_size == 0) {
            return UINT64_MAX;
        }
        const uintptr_t base = reinterpret_cast<uintptr_t>(cma_base);
        const uintptr_t cur = reinterpret_cast<uintptr_t>(ptr);
        if (cur < base || cur - base >= static_cast<uintptr_t>(cma_size)) {
            return UINT64_MAX;
        }
        return static_cast<uint64_t>(cur - base);
    };
    std::ofstream meta(meta_path);
    if (!meta.good()) {
        GGML_LOG_WARN("%s: failed to open repro meta path %s\n", __func__, meta_path.c_str());
        return;
    }
    meta << "op_name=" << (op_name != nullptr ? op_name : "") << "\n";
    meta << "col=" << col << "\n";
    meta << "m=" << m << "\n";
    meta << "k=" << k << "\n";
    meta << "row_tiles=" << layout.row_tiles << "\n";
    meta << "col_tiles=" << layout.col_tiles << "\n";
    meta << "act_bytes=" << layout.act_bytes << "\n";
    meta << "act_payload_bytes=" << act_payload_bytes << "\n";
    meta << "scale_bytes=" << layout.scale_bytes << "\n";
    meta << "weight_bytes=" << layout.weight_bytes << "\n";
    meta << "output_bytes=" << layout.output_bytes << "\n";
    meta << "host_output_bytes=" << host_output_bytes << "\n";
    meta << "max_block_rows=" << npu_decode_pingpong_max_block_rows(layout) << "\n";
    meta << "output_precision=" << static_cast<unsigned>(output_precision) << "\n";
    meta << "block_mvout=" << (block_mvout ? 1 : 0) << "\n";
    meta << "cma_size=" << cma_size << "\n";
    meta << "act_cma_offset=" << cma_offset_of(act_ptr) << "\n";
    meta << "scale_cma_offset=" << cma_offset_of(scale_ptr) << "\n";
    meta << "weight_cma_offset=" << cma_offset_of(weight_ptr) << "\n";
    meta << "output_cma_offset=" << cma_offset_of(output_ptr) << "\n";
    meta << "act_file=" << act_path << "\n";
    meta << "scale_file=" << scale_path << "\n";
    meta << "weight_file=" << weight_path << "\n";
    meta << "output_file=" << output_path << "\n";
    meta.close();

    const bool ok =
        npu_decode_write_binary_file(act_path, act_ptr, act_payload_bytes) &&
        npu_decode_write_binary_file(scale_path, scale_ptr, layout.scale_bytes) &&
        npu_decode_write_binary_file(weight_path, weight_ptr, layout.weight_bytes) &&
        npu_decode_write_binary_file(output_path, output_ptr, host_output_bytes);
    if (!ok) {
        GGML_LOG_WARN("%s: failed to write one or more repro dump files under %s\n",
                __func__, dump_dir);
    } else if (npu_decode_w4a16_debug_log_enabled()) {
        GGML_LOG_WARN("%s: wrote repro dump %s.*\n", __func__, base.c_str());
    }
}

static void npu_decode_repro_dump_w8_block(
        const char * op_name,
        int64_t col,
        uint32_t group_idx,
        uint32_t row_base,
        uint32_t rows,
        uint32_t group_k,
        uint32_t col_tiles,
        uint32_t row_tiles,
        uint32_t act_bytes,
        uint32_t scale_bytes,
        uint32_t weight_bytes,
        uint32_t host_output_bytes,
        const void * act_ptr,
        const void * scale_ptr,
        const void * weight_ptr,
        const void * output_ptr) {
    if (!npu_decode_repro_dump_match(op_name)) {
        return;
    }

    static std::mutex mutex;
    static int dumped = 0;
    std::lock_guard<std::mutex> lock(mutex);
    if (dumped >= npu_decode_repro_dump_limit()) {
        return;
    }

    const char * dump_dir = npu_decode_repro_dump_dir();
    if (dump_dir == nullptr) {
        return;
    }
    const int dump_idx = dumped++;
    std::ostringstream prefix;
    prefix << dump_dir << "/repro_w8_" << dump_idx << "_" << npu_decode_repro_sanitize(op_name);

    const std::string base = prefix.str();
    const std::string meta_path = base + ".meta.txt";
    const std::string act_path = base + ".act.bin";
    const std::string scale_path = base + ".scale.bin";
    const std::string weight_path = base + ".weight.bin";
    const std::string output_path = base + ".output.bin";

    std::ofstream meta(meta_path);
    if (!meta.good()) {
        GGML_LOG_WARN("%s: failed to open repro meta path %s\n", __func__, meta_path.c_str());
        return;
    }
    meta << "schema=aicas_decode_w8_block_repro_v1\n";
    meta << "op_name=" << (op_name != nullptr ? op_name : "") << "\n";
    meta << "col=" << col << "\n";
    meta << "group_idx=" << group_idx << "\n";
    meta << "row_base=" << row_base << "\n";
    meta << "rows=" << rows << "\n";
    meta << "group_k=" << group_k << "\n";
    meta << "row_tiles=" << row_tiles << "\n";
    meta << "col_tiles=" << col_tiles << "\n";
    meta << "act_bytes=" << act_bytes << "\n";
    meta << "scale_bytes=" << scale_bytes << "\n";
    meta << "weight_bytes=" << weight_bytes << "\n";
    meta << "host_output_bytes=" << host_output_bytes << "\n";
    meta << "act_file=" << act_path << "\n";
    meta << "scale_file=" << scale_path << "\n";
    meta << "weight_file=" << weight_path << "\n";
    meta << "output_file=" << output_path << "\n";
    meta.close();

    const bool ok =
        npu_decode_write_binary_file(act_path, act_ptr, act_bytes) &&
        npu_decode_write_binary_file(scale_path, scale_ptr, scale_bytes) &&
        npu_decode_write_binary_file(weight_path, weight_ptr, weight_bytes) &&
        npu_decode_write_binary_file(output_path, output_ptr, host_output_bytes);
    if (!ok) {
        GGML_LOG_WARN("%s: failed to write one or more W8 repro dump files under %s\n",
                __func__, dump_dir);
    } else if (npu_decode_w4a16_debug_log_enabled()) {
        GGML_LOG_WARN("%s: wrote W8 repro dump %s.*\n", __func__, base.c_str());
    }
}

static void npu_decode_pack_scale_weight_block(
        std::vector<uint8_t> & packed,
        const npu_decode_gemv_layout & layout,
        const void * q4_data,
        int64_t packed_k,
        int64_t q4_nb1,
        const void * scale_data,
        int scale_type,
        int64_t scale_nb0,
        int64_t scale_nb1,
        int64_t row_base,
        int64_t block_rows,
        int64_t k) {
    std::fill(packed.begin(), packed.end(), 0);

    for (uint32_t row_tile = 0; row_tile < layout.row_tiles; ++row_tile) {
        for (uint32_t col_tile = 0; col_tile < layout.col_tiles; ++col_tile) {
            const size_t scale_line =
                static_cast<size_t>(row_tile * layout.col_tiles + col_tile) *
                NPU_DECODE_GEMV_SCALE_TILE_BYTES;
            for (uint32_t lane = 0; lane < NPU_DECODE_GEMV_M_TILE; ++lane) {
                const int64_t row = static_cast<int64_t>(row_tile) * NPU_DECODE_GEMV_M_TILE + lane;
                if (row >= block_rows) {
                    continue;
                }
                const float scale = npu_decode_read_typed_f32(
                        scale_data,
                        scale_type,
                        (row_base + row) * scale_nb1 + static_cast<int64_t>(col_tile) * scale_nb0);
                const ggml_fp16_t scale_fp16 = ggml_fp32_to_fp16(scale);
                std::memcpy(packed.data() + scale_line + lane * 2, &scale_fp16, sizeof(scale_fp16));
            }
        }
    }

    for (uint32_t row_tile = 0; row_tile < layout.row_tiles; ++row_tile) {
        for (uint32_t col_tile = 0; col_tile < layout.col_tiles; ++col_tile) {
            const uint32_t tile_index = row_tile * layout.col_tiles + col_tile;
            for (uint32_t row_in_tile = 0; row_in_tile < NPU_DECODE_GEMV_M_TILE; ++row_in_tile) {
                const int64_t row = static_cast<int64_t>(row_tile) * NPU_DECODE_GEMV_M_TILE + row_in_tile;
                if (row >= block_rows) {
                    continue;
                }
                const uint8_t * q4_row = static_cast<const uint8_t *>(q4_data) + (row_base + row) * q4_nb1;
                uint8_t * dst = packed.data() + layout.weight_base +
                    static_cast<size_t>(tile_index) * NPU_DECODE_GEMV_WEIGHT_TILE_BYTES +
                    row_in_tile * NPU_DECODE_GEMV_WEIGHT_ROW_BYTES;
                for (uint32_t lane = 0; lane < NPU_DECODE_GEMV_K_TILE; ++lane) {
                    const int64_t in_idx = static_cast<int64_t>(col_tile) * NPU_DECODE_GEMV_K_TILE + lane;
                    if (in_idx >= k || in_idx / 2 >= packed_k) {
                        continue;
                    }
                    const int8_t signed_q4 = static_cast<int8_t>(npu_decode_q4_unsigned_at(q4_row, in_idx)) - 8;
                    const uint8_t nibble = static_cast<uint8_t>(signed_q4) & 0x0fu;
                    if ((lane & 1u) == 0) {
                        dst[lane / 2] = static_cast<uint8_t>((dst[lane / 2] & 0xf0u) | nibble);
                    } else {
                        dst[lane / 2] = static_cast<uint8_t>((dst[lane / 2] & 0x0fu) | (nibble << 4));
                    }
                }
            }
        }
    }
}

static void npu_decode_pack_pingpong_scale_weight(
        std::vector<uint8_t> & scale_packed,
        std::vector<uint8_t> & weight_packed,
        const npu_decode_gemv_layout & layout,
        const void * q4_data,
        int64_t packed_k,
        int64_t q4_nb1,
        const void * scale_data,
        int scale_type,
        int64_t scale_nb0,
        int64_t scale_nb1,
        int64_t out_channels,
        int64_t k) {
    std::fill(scale_packed.begin(), scale_packed.end(), 0);
    std::fill(weight_packed.begin(), weight_packed.end(), 0);

    for (uint32_t row_tile = 0; row_tile < layout.row_tiles; ++row_tile) {
        for (uint32_t col_tile = 0; col_tile < layout.col_tiles; ++col_tile) {
            const size_t scale_line =
                static_cast<size_t>(row_tile * layout.col_tiles + col_tile) *
                NPU_DECODE_GEMV_SCALE_TILE_BYTES;
            for (uint32_t lane = 0; lane < NPU_DECODE_GEMV_M_TILE; ++lane) {
                const int64_t row = static_cast<int64_t>(row_tile) * NPU_DECODE_GEMV_M_TILE + lane;
                if (row >= out_channels) {
                    continue;
                }
                const float scale = npu_decode_read_typed_f32(
                        scale_data,
                        scale_type,
                        row * scale_nb1 + static_cast<int64_t>(col_tile) * scale_nb0);
                const ggml_fp16_t scale_fp16 = ggml_fp32_to_fp16(scale);
                std::memcpy(scale_packed.data() + scale_line + lane * 2, &scale_fp16, sizeof(scale_fp16));
            }
        }
    }

    for (uint32_t row_tile = 0; row_tile < layout.row_tiles; ++row_tile) {
        for (uint32_t col_tile = 0; col_tile < layout.col_tiles; ++col_tile) {
            const uint32_t tile_index = row_tile * layout.col_tiles + col_tile;
            for (uint32_t row_in_tile = 0; row_in_tile < NPU_DECODE_GEMV_M_TILE; ++row_in_tile) {
                const int64_t row = static_cast<int64_t>(row_tile) * NPU_DECODE_GEMV_M_TILE + row_in_tile;
                if (row >= out_channels) {
                    continue;
                }
                const uint8_t * q4_row = static_cast<const uint8_t *>(q4_data) + row * q4_nb1;
                uint8_t * dst = weight_packed.data() +
                    static_cast<size_t>(tile_index) * NPU_DECODE_GEMV_WEIGHT_TILE_BYTES +
                    row_in_tile * NPU_DECODE_GEMV_WEIGHT_ROW_BYTES;
                for (uint32_t lane = 0; lane < NPU_DECODE_GEMV_K_TILE; ++lane) {
                    const int64_t in_idx = static_cast<int64_t>(col_tile) * NPU_DECODE_GEMV_K_TILE + lane;
                    if (in_idx >= k || in_idx / 2 >= packed_k) {
                        continue;
                    }
                    const int8_t signed_q4 = static_cast<int8_t>(npu_decode_q4_unsigned_at(q4_row, in_idx)) - 8;
                    const uint8_t nibble = static_cast<uint8_t>(signed_q4) & 0x0fu;
                    if ((lane & 1u) == 0) {
                        dst[lane / 2] = static_cast<uint8_t>((dst[lane / 2] & 0xf0u) | nibble);
                    } else {
                        dst[lane / 2] = static_cast<uint8_t>((dst[lane / 2] & 0x0fu) | (nibble << 4));
                    }
                }
            }
        }
    }
}

static const npu_decode_preloaded_tensor * npu_decode_lookup_preloaded_tensor(
        const char * weight_name,
        int64_t packed_k,
        int64_t out_channels,
        int64_t q4_nb1,
        int64_t k) {
    if (weight_name == nullptr || weight_name[0] == '\0') {
        return nullptr;
    }

    npu_decode_preload_cache & cache = npu_decode_get_preload_cache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    auto it = cache.entries.find(weight_name);
    if (it == cache.entries.end()) {
        return nullptr;
    }
    const npu_decode_preloaded_tensor & entry = it->second;
    if (entry.packed_k != packed_k ||
            entry.out_channels != out_channels ||
            entry.q4_nb1 != q4_nb1 ||
            entry.k != k) {
        return nullptr;
    }
    return &it->second;
}

constexpr uint32_t NPU_DECODE_ATTN_HIDDEN_DIM = 960;
constexpr uint32_t NPU_DECODE_ATTN_HEAD_DIM = 64;
constexpr uint32_t NPU_DECODE_ATTN_Q_HEADS = 15;
constexpr uint32_t NPU_DECODE_ATTN_KV_HEADS = 5;
constexpr uint32_t NPU_DECODE_ATTN_GROUP_SIZE = NPU_DECODE_ATTN_Q_HEADS / NPU_DECODE_ATTN_KV_HEADS;
constexpr uint32_t NPU_DECODE_ATTN_KV_DIM = NPU_DECODE_ATTN_KV_HEADS * NPU_DECODE_ATTN_HEAD_DIM;
constexpr uint32_t NPU_DECODE_ATTN_CACHE_CAPACITY_TOKENS = 4096;
constexpr uint32_t NPU_DECODE_ATTN_VISIBLE_WINDOW_TOKENS = 992;
constexpr uint32_t NPU_DECODE_ATTN_KV_HEAD_STRIDE_BYTES = 270336;
constexpr uint32_t NPU_DECODE_ATTN_NPU_KV_HEAD_STRIDE_BYTES = 65536;
constexpr uint32_t NPU_DECODE_ATTN_ROW_TILE_ELEMS = 32;
constexpr uint32_t NPU_DECODE_ATTN_W8_COL_TILE_ELEMS = 64;
constexpr uint32_t NPU_DECODE_ATTN_KV_CACHE_ROW_BYTES = NPU_DECODE_ATTN_HEAD_DIM;
constexpr uint32_t NPU_DECODE_ATTN_V_ROW_TILE_STRIDE_BYTES =
    (NPU_DECODE_ATTN_CACHE_CAPACITY_TOKENS / NPU_DECODE_ATTN_W8_COL_TILE_ELEMS) *
    NPU_DECODE_GEMV_W8_WEIGHT_TILE_BYTES;
constexpr uint32_t NPU_DECODE_ATTN_WEIGHT_SCALE_SPM_BASE = 0x00000;
constexpr uint32_t NPU_DECODE_ATTN_INPUT_SPM_BASE = 0x10000;
constexpr uint32_t NPU_DECODE_ATTN_OUTPUT_SPM_BASE = 0x00000;
constexpr uint32_t NPU_DECODE_ATTN_HIDDEN_ACT_BASE = 0x0000;
constexpr uint32_t NPU_DECODE_ATTN_ATTENTION_ACT_BASE = 0x0000;
constexpr uint32_t NPU_DECODE_ATTN_Q_GROUP_ACT_BASE = 0x5000;
constexpr uint32_t NPU_DECODE_ATTN_SCORE_ACT_BASE = 0x6000;
constexpr uint32_t NPU_DECODE_ATTN_ROPE_LUT_WINDOW_BYTES = 256;
constexpr off_t NPU_DECODE_ATTN_KV260_MMAP_REGS_OFFSET = 0x10000000;
constexpr uint32_t NPU_DECODE_ATTN_KV260_REG_MMAP_SIZE = 0x10000;
constexpr uint32_t NPU_DECODE_ATTN_KV_K_SCALE0_OFFSET = 0x4a0;
constexpr uint32_t NPU_DECODE_ATTN_KV_K_SCALE1_OFFSET = 0x4a8;
constexpr uint32_t NPU_DECODE_ATTN_KV_V_SCALE0_OFFSET = 0x4b0;
constexpr uint32_t NPU_DECODE_ATTN_KV_V_SCALE1_OFFSET = 0x4b8;

struct npu_decode_attention_block_q8_0 {
    ggml_fp16_t d;
    int8_t qs[32];
};
static_assert(sizeof(npu_decode_attention_block_q8_0) == sizeof(ggml_fp16_t) + 32, "unexpected q8_0 block layout");

struct npu_decode_attention_kv_scale_regs {
    uint16_t values[NPU_DECODE_ATTN_KV_HEADS] = {};
    uint8_t count = 0;
    uint8_t seq = 0;
    bool valid = false;
};

struct npu_decode_attention_layer_cache {
    void * k_cache = nullptr;
    void * v_cache = nullptr;
    uint32_t converted_len = 0;
    uint32_t last_position = 0;
};

struct npu_decode_attention_cache {
    std::mutex mutex;
    std::vector<npu_decode_attention_layer_cache> layers;
};

static npu_decode_attention_cache & npu_decode_attention_get_cache() {
    static npu_decode_attention_cache cache;
    return cache;
}

static void npu_decode_attention_clear_cache() {
    npu_decode_attention_cache & cache = npu_decode_attention_get_cache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    for (auto & layer : cache.layers) {
        if (layer.k_cache != nullptr) {
            npu_decode_mem_free(layer.k_cache);
        }
        if (layer.v_cache != nullptr) {
            npu_decode_mem_free(layer.v_cache);
        }
        layer = {};
    }
    cache.layers.clear();
}

static uint32_t npu_decode_attention_kv_cache_tile_bytes() {
    return NPU_DECODE_GEMV_LINE_BYTES +
        NPU_DECODE_ATTN_ROW_TILE_ELEMS * NPU_DECODE_ATTN_KV_CACHE_ROW_BYTES;
}

static uint32_t npu_decode_attention_kv_cache_window_tiles() {
    return NPU_DECODE_ATTN_CACHE_CAPACITY_TOKENS / NPU_DECODE_ATTN_ROW_TILE_ELEMS;
}

static uint32_t npu_decode_attention_kv_cache_window_tokens() {
    return NPU_DECODE_ATTN_CACHE_CAPACITY_TOKENS;
}

static uint32_t npu_decode_attention_kv_cache_scale_region_bytes() {
    return npu_decode_align_up_u32(
            npu_decode_attention_kv_cache_window_tokens() * sizeof(ggml_fp16_t),
            NPU_DECODE_GEMV_DMA_ALIGN);
}

static uint32_t npu_decode_attention_kv_cache_quant_tile_bytes() {
    return NPU_DECODE_ATTN_ROW_TILE_ELEMS * NPU_DECODE_ATTN_KV_CACHE_ROW_BYTES;
}

static uint32_t npu_decode_attention_cache_cell_from_position(uint32_t position) {
    const uint32_t window_tiles = npu_decode_attention_kv_cache_window_tiles();
    return ((position / NPU_DECODE_ATTN_ROW_TILE_ELEMS) % window_tiles) *
        NPU_DECODE_ATTN_ROW_TILE_ELEMS + (position % NPU_DECODE_ATTN_ROW_TILE_ELEMS);
}

static uint32_t npu_decode_attention_cache_scale_addr(uint32_t head, uint32_t token) {
    const uint32_t cell = token % npu_decode_attention_kv_cache_window_tokens();
    return head * NPU_DECODE_ATTN_KV_HEAD_STRIDE_BYTES + cell * sizeof(ggml_fp16_t);
}

static uint32_t npu_decode_attention_cache_quant_addr(uint32_t head, uint32_t token, uint32_t dim) {
    const uint32_t cell = token % npu_decode_attention_kv_cache_window_tokens();
    const uint32_t tile = cell / NPU_DECODE_ATTN_ROW_TILE_ELEMS;
    const uint32_t lane = cell % NPU_DECODE_ATTN_ROW_TILE_ELEMS;
    return head * NPU_DECODE_ATTN_KV_HEAD_STRIDE_BYTES +
        npu_decode_attention_kv_cache_scale_region_bytes() +
        tile * npu_decode_attention_kv_cache_quant_tile_bytes() +
        lane * NPU_DECODE_ATTN_KV_CACHE_ROW_BYTES + dim;
}

static uint32_t npu_decode_attention_v_cache_quant_addr(uint32_t head, uint32_t token, uint32_t dim) {
    const uint32_t cell = token % npu_decode_attention_kv_cache_window_tokens();
    const uint32_t row_tile = dim / NPU_DECODE_GEMV_M_TILE;
    const uint32_t row_lane = dim % NPU_DECODE_GEMV_M_TILE;
    const uint32_t col_tile = cell / NPU_DECODE_ATTN_W8_COL_TILE_ELEMS;
    const uint32_t col_lane = cell % NPU_DECODE_ATTN_W8_COL_TILE_ELEMS;
    return head * NPU_DECODE_ATTN_KV_HEAD_STRIDE_BYTES +
        npu_decode_attention_kv_cache_scale_region_bytes() +
        row_tile * NPU_DECODE_ATTN_V_ROW_TILE_STRIDE_BYTES +
        col_tile * NPU_DECODE_GEMV_W8_WEIGHT_TILE_BYTES +
        row_lane * NPU_DECODE_ATTN_W8_COL_TILE_ELEMS +
        col_lane;
}

static uint32_t npu_decode_attention_npu_kv_cache_scale_region_bytes() {
    return npu_decode_align_up_u32(
            NPU_DECODE_ATTN_VISIBLE_WINDOW_TOKENS * sizeof(ggml_fp16_t),
            NPU_DECODE_GEMV_DMA_ALIGN);
}

static uint32_t npu_decode_attention_npu_cache_cell_from_position(uint32_t position) {
    const uint32_t visible_tiles = NPU_DECODE_ATTN_VISIBLE_WINDOW_TOKENS / NPU_DECODE_ATTN_ROW_TILE_ELEMS;
    return ((position / NPU_DECODE_ATTN_ROW_TILE_ELEMS) % visible_tiles) *
        NPU_DECODE_ATTN_ROW_TILE_ELEMS + (position % NPU_DECODE_ATTN_ROW_TILE_ELEMS);
}

static void npu_decode_attention_store_u16_le(uint8_t * dst, uint16_t value) {
    dst[0] = static_cast<uint8_t>(value & 0xffu);
    dst[1] = static_cast<uint8_t>((value >> 8) & 0xffu);
}

static uint16_t npu_decode_attention_load_u16_le(const uint8_t * src) {
    return static_cast<uint16_t>(src[0]) |
        static_cast<uint16_t>(static_cast<uint16_t>(src[1]) << 8);
}

static uint64_t npu_decode_attention_flow(
        uint8_t src0,
        uint8_t src1,
        uint8_t unary,
        uint8_t binary,
        uint8_t reduce,
        uint8_t dst,
        uint16_t elem_count,
        uint16_t position,
        uint8_t group_count = 1,
        bool kv_quant = false,
        bool kv_col_scale = false,
        bool kv_v_separated = false) {
    uint64_t flow = npu_decode_stream_unsupported_flow(
            src0,
            src1,
            unary,
            binary,
            reduce,
            dst,
            0,
            0,
            elem_count,
            position);
    if (kv_quant) {
        flow |= NPU_DECODE_FLAG_KV_QUANT;
    }
    if (kv_col_scale) {
        flow |= NPU_DECODE_FLAG_KV_COL_SCALE;
    }
    if (kv_v_separated) {
        flow |= NPU_DECODE_FLAG_KV_V_SEPARATED;
    }
    flow |= uint64_t(group_count - 1u) << 25;
    return flow;
}

static bool npu_decode_attention_read_scale_regs(bool is_v, npu_decode_attention_kv_scale_regs * out) {
    if (out == nullptr) {
        return false;
    }
    const int fd = open(npu_decode_device_path(), O_RDWR | O_CLOEXEC);
    if (fd < 0) {
        return false;
    }
    void * mapped = mmap(
            nullptr,
            NPU_DECODE_ATTN_KV260_REG_MMAP_SIZE,
            PROT_READ | PROT_WRITE,
            MAP_SHARED,
            fd,
            NPU_DECODE_ATTN_KV260_MMAP_REGS_OFFSET);
    if (mapped == MAP_FAILED) {
        close(fd);
        return false;
    }
    volatile char * regs = reinterpret_cast<volatile char *>(mapped);
    const uint32_t scale0_off = is_v ? NPU_DECODE_ATTN_KV_V_SCALE0_OFFSET : NPU_DECODE_ATTN_KV_K_SCALE0_OFFSET;
    const uint32_t scale1_off = is_v ? NPU_DECODE_ATTN_KV_V_SCALE1_OFFSET : NPU_DECODE_ATTN_KV_K_SCALE1_OFFSET;
    const uint64_t scale0 = *reinterpret_cast<volatile uint64_t *>(regs + scale0_off);
    const uint64_t scale1 = *reinterpret_cast<volatile uint64_t *>(regs + scale1_off);
    munmap(mapped, NPU_DECODE_ATTN_KV260_REG_MMAP_SIZE);
    close(fd);

    out->values[0] = static_cast<uint16_t>(scale0 & 0xffffu);
    out->values[1] = static_cast<uint16_t>((scale0 >> 16) & 0xffffu);
    out->values[2] = static_cast<uint16_t>((scale0 >> 32) & 0xffffu);
    out->values[3] = static_cast<uint16_t>((scale0 >> 48) & 0xffffu);
    out->values[4] = static_cast<uint16_t>(scale1 & 0xffffu);
    out->count = static_cast<uint8_t>((scale1 >> 16) & 0xffu);
    out->seq = static_cast<uint8_t>((scale1 >> 24) & 0xffu);
    out->valid = ((scale1 >> 32) & 1u) != 0;
    return out->valid && out->count == NPU_DECODE_ATTN_KV_HEADS;
}

static bool npu_decode_attention_convert_one_llama_cache_row(
        const ggml_tensor * src,
        uint32_t token,
        uint32_t head,
        uint8_t * dst_cache,
        bool transposed_v_layout) {
    const char * row = static_cast<const char *>(src->data) +
        static_cast<size_t>(head) * src->nb[1] +
        static_cast<size_t>(token) * src->nb[2];
    uint16_t scale = ggml_fp32_to_fp16(1.0f);
    int8_t quant[NPU_DECODE_ATTN_HEAD_DIM] = {};
    if (src->type == GGML_TYPE_Q8_0) {
        constexpr uint32_t q8_block = 32;
        const npu_decode_attention_block_q8_0 * blocks =
            reinterpret_cast<const npu_decode_attention_block_q8_0 *>(row);
        scale = blocks[0].d;
        bool same_scale = true;
        for (uint32_t block = 1; block < NPU_DECODE_ATTN_HEAD_DIM / q8_block; ++block) {
            same_scale = same_scale && blocks[block].d == scale;
        }
        if (same_scale) {
            for (uint32_t block = 0; block < NPU_DECODE_ATTN_HEAD_DIM / q8_block; ++block) {
                std::memcpy(
                        quant + block * q8_block,
                        blocks[block].qs,
                        q8_block);
            }
        } else {
            float values[NPU_DECODE_ATTN_HEAD_DIM] = {};
            float amax = 0.0f;
            for (uint32_t block = 0; block < NPU_DECODE_ATTN_HEAD_DIM / q8_block; ++block) {
                const float d = ggml_fp16_to_fp32(blocks[block].d);
                for (uint32_t lane = 0; lane < q8_block; ++lane) {
                    const uint32_t dim = block * q8_block + lane;
                    values[dim] = d * static_cast<float>(blocks[block].qs[lane]);
                    amax = std::max(amax, std::fabs(values[dim]));
                }
            }
            const float d = amax / 128.0f;
            const float id = d != 0.0f ? 1.0f / d : 0.0f;
            scale = ggml_fp32_to_fp16(d);
            for (uint32_t dim = 0; dim < NPU_DECODE_ATTN_HEAD_DIM; ++dim) {
                int q = static_cast<int>(std::round(values[dim] * id));
                q = std::max(-127, std::min(127, q));
                quant[dim] = static_cast<int8_t>(q);
            }
        }
    } else if (src->type == GGML_TYPE_F16 || src->type == GGML_TYPE_F32) {
        float amax = 0.0f;
        float values[NPU_DECODE_ATTN_HEAD_DIM] = {};
        for (uint32_t dim = 0; dim < NPU_DECODE_ATTN_HEAD_DIM; ++dim) {
            values[dim] = npu_decode_read_typed_f32(
                    src->data,
                    src->type,
                    static_cast<int64_t>(head) * src->nb[1] + static_cast<int64_t>(token) * src->nb[2] +
                        static_cast<int64_t>(dim) * src->nb[0]);
            amax = std::max(amax, std::fabs(values[dim]));
        }
        const float d = amax / 128.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        scale = ggml_fp32_to_fp16(d);
        for (uint32_t dim = 0; dim < NPU_DECODE_ATTN_HEAD_DIM; ++dim) {
            int q = static_cast<int>(std::round(values[dim] * id));
            q = std::max(-127, std::min(127, q));
            quant[dim] = static_cast<int8_t>(q);
        }
    } else {
        return false;
    }

    npu_decode_attention_store_u16_le(
            dst_cache + npu_decode_attention_cache_scale_addr(head, token),
            scale);
    if (transposed_v_layout) {
        for (uint32_t dim = 0; dim < NPU_DECODE_ATTN_HEAD_DIM; ++dim) {
            dst_cache[npu_decode_attention_v_cache_quant_addr(head, token, dim)] =
                static_cast<uint8_t>(quant[dim]);
        }
    } else {
        std::memcpy(
                dst_cache + npu_decode_attention_cache_quant_addr(head, token, 0),
                quant,
                NPU_DECODE_ATTN_HEAD_DIM);
    }
    return true;
}

static bool npu_decode_attention_convert_llama_cache_prefix(
        const ggml_tensor * llama_cache,
        uint32_t token_count,
        uint8_t * dst_cache,
        bool transposed_v_layout = false) {
    if (llama_cache == nullptr || dst_cache == nullptr ||
            llama_cache->ne[0] != NPU_DECODE_ATTN_HEAD_DIM ||
            llama_cache->ne[1] != NPU_DECODE_ATTN_KV_HEADS ||
            llama_cache->ne[2] < token_count) {
        return false;
    }
    for (uint32_t token = 0; token < token_count; ++token) {
        for (uint32_t head = 0; head < NPU_DECODE_ATTN_KV_HEADS; ++head) {
            if (!npu_decode_attention_convert_one_llama_cache_row(
                        llama_cache, token, head, dst_cache, transposed_v_layout)) {
                return false;
            }
        }
    }
    return true;
}

static void npu_decode_attention_append_quant_from_output(
        uint8_t * dst_cache,
        const uint8_t * output,
        uint32_t token,
        const npu_decode_attention_kv_scale_regs & regs,
        bool transposed_v_layout = false) {
    const uint32_t src_cell = npu_decode_attention_npu_cache_cell_from_position(token);
    const uint32_t src_scale_region = npu_decode_attention_npu_kv_cache_scale_region_bytes();
    const uint32_t quant_tile_bytes = npu_decode_attention_kv_cache_quant_tile_bytes();
    for (uint32_t head = 0; head < NPU_DECODE_ATTN_KV_HEADS; ++head) {
        npu_decode_attention_store_u16_le(
                dst_cache + npu_decode_attention_cache_scale_addr(head, token),
                regs.values[head]);
        const uint32_t src_addr =
            head * NPU_DECODE_ATTN_NPU_KV_HEAD_STRIDE_BYTES +
            src_scale_region +
            (src_cell / NPU_DECODE_ATTN_ROW_TILE_ELEMS) * quant_tile_bytes +
            (src_cell % NPU_DECODE_ATTN_ROW_TILE_ELEMS) * NPU_DECODE_ATTN_KV_CACHE_ROW_BYTES;
        if (transposed_v_layout) {
            for (uint32_t dim = 0; dim < NPU_DECODE_ATTN_HEAD_DIM; ++dim) {
                dst_cache[npu_decode_attention_v_cache_quant_addr(head, token, dim)] = output[src_addr + dim];
            }
        } else {
            std::memcpy(
                    dst_cache + npu_decode_attention_cache_quant_addr(head, token, 0),
                    output + src_addr,
                    NPU_DECODE_ATTN_HEAD_DIM);
        }
    }
}

static bool npu_decode_attention_dump_requested(int32_t layer_id) {
    const char * path = std::getenv("AICAS_TEXT_DECODE_ATTN_NPU_DUMP");
    if (path == nullptr || path[0] == '\0') {
        return false;
    }
    const char * layer = std::getenv("AICAS_TEXT_DECODE_ATTN_NPU_DUMP_LAYER");
    if (layer == nullptr || layer[0] == '\0') {
        return layer_id == 0;
    }
    char * end = nullptr;
    const long parsed = std::strtol(layer, &end, 10);
    return end != layer && *end == '\0' && parsed == layer_id;
}

static bool npu_decode_attention_pre_o_output_enabled() {
    const char * value = std::getenv("AICAS_TEXT_DECODE_ATTN_NPU_PRE_O");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

static void npu_decode_attention_write_debug_dump(
        const char * op_name,
        int32_t layer_id,
        int32_t position,
        uint32_t seq_len,
        const void * q_fp16,
        uint32_t q_bytes,
        const void * k_cache,
        const void * v_cache) {
    const char * path = std::getenv("AICAS_TEXT_DECODE_ATTN_NPU_DUMP");
    if (path == nullptr || path[0] == '\0' || q_fp16 == nullptr || k_cache == nullptr || v_cache == nullptr) {
        return;
    }
    std::ostringstream filename;
    filename << path << ".layer" << layer_id << ".pos" << position << ".bin";
    std::ofstream out(filename.str(), std::ios::binary);
    if (!out.is_open()) {
        if (npu_decode_w4a16_debug_log_enabled()) {
            GGML_LOG_WARN("%s: failed to open attention dump %s\n", op_name != nullptr ? op_name : __func__, filename.str().c_str());
        }
        return;
    }
    struct header {
        char magic[16];
        uint32_t version;
        uint32_t layer;
        uint32_t position;
        uint32_t seq_len;
        uint32_t hidden_dim;
        uint32_t q_heads;
        uint32_t kv_heads;
        uint32_t head_dim;
        uint32_t kv_head_stride;
        uint32_t scale_region;
        uint32_t q_bytes;
        uint32_t cache_bytes;
    } hdr = {
        {'A','I','C','A','S','_','A','T','T','N','D','U','M','P','1','\0'},
        1u,
        static_cast<uint32_t>(layer_id),
        static_cast<uint32_t>(position),
        seq_len,
        NPU_DECODE_ATTN_HIDDEN_DIM,
        NPU_DECODE_ATTN_Q_HEADS,
        NPU_DECODE_ATTN_KV_HEADS,
        NPU_DECODE_ATTN_HEAD_DIM,
        NPU_DECODE_ATTN_KV_HEAD_STRIDE_BYTES,
        npu_decode_attention_kv_cache_scale_region_bytes(),
        q_bytes,
        NPU_DECODE_ATTN_KV_HEADS * NPU_DECODE_ATTN_KV_HEAD_STRIDE_BYTES,
    };
    out.write(reinterpret_cast<const char *>(&hdr), sizeof(hdr));
    out.write(static_cast<const char *>(q_fp16), q_bytes);
    out.write(static_cast<const char *>(k_cache), hdr.cache_bytes);
    out.write(static_cast<const char *>(v_cache), hdr.cache_bytes);
    if (npu_decode_w4a16_debug_log_enabled()) {
        GGML_LOG_INFO("%s: wrote attention dump %s seq_len=%u q_bytes=%u cache_bytes=%u\n",
                op_name != nullptr ? op_name : __func__,
                filename.str().c_str(),
                seq_len,
                q_bytes,
                hdr.cache_bytes);
    }
}

static void npu_decode_attention_fill_k_score_scale(
        uint8_t * dst,
        uint32_t dst_bytes,
        const uint8_t * k_cache,
        uint32_t kv_head,
        uint32_t seq_len) {
    std::memset(dst, 0, dst_bytes);
    for (uint32_t token = 0; token < seq_len; ++token) {
        const uint32_t row_tile = token / NPU_DECODE_ATTN_ROW_TILE_ELEMS;
        const uint32_t lane = token % NPU_DECODE_ATTN_ROW_TILE_ELEMS;
        const uint16_t scale = npu_decode_attention_load_u16_le(
                k_cache + npu_decode_attention_cache_scale_addr(kv_head, token));
        npu_decode_attention_store_u16_le(
                dst + row_tile * NPU_DECODE_GEMV_LINE_BYTES + lane * sizeof(ggml_fp16_t),
                scale);
    }
}

static bool npu_decode_attention_prepare_w4_projection(
        const ggml_npu_decode_awq_view * view,
        const npu_decode_preloaded_tensor ** out) {
    if (view == nullptr || out == nullptr ||
            view->weight_name == nullptr ||
            view->q4_data == nullptr ||
            view->scale_data == nullptr ||
            view->zero_data == nullptr ||
            view->smooth_scale == nullptr ||
            view->smooth_scale_len != static_cast<size_t>(view->k)) {
        return false;
    }
    const npu_decode_preloaded_tensor * preloaded = npu_decode_lookup_preloaded_tensor(
            view->weight_name,
            view->packed_k,
            view->out_channels,
            view->q4_nb1,
            view->k);
    if (preloaded == nullptr) {
        if (!ggml_backend_npu_decode_w4a16_preload(
                    view->weight_name,
                    view->q4_data,
                    view->packed_k,
                    view->out_channels,
                    view->q4_nb1,
                    view->scale_data,
                    view->scale_type,
                    view->scale_nb0,
                    view->scale_nb1,
                    view->zero_data,
                    view->zero_type,
                    view->zero_nb0,
                    view->zero_nb1,
                    view->k,
                    view->smooth_scale,
                    view->smooth_scale_len)) {
            return false;
        }
        preloaded = npu_decode_lookup_preloaded_tensor(
                view->weight_name,
                view->packed_k,
                view->out_channels,
                view->q4_nb1,
                view->k);
    }
    if (preloaded == nullptr ||
            !preloaded->pingpong_available ||
            preloaded->pingpong_scale.size() != preloaded->pingpong_layout.scale_bytes ||
            preloaded->pingpong_weight.size() != preloaded->pingpong_layout.weight_bytes) {
        return false;
    }
    *out = preloaded;
    return true;
}

static void npu_decode_attention_run_w4_projection(
        const char * op_name,
        const npu_decode_preloaded_tensor * preloaded,
        const void * act_data,
        int act_type,
        int64_t act_nb0,
        int64_t act_nb1,
        void * act_cma,
        void * scale_cma,
        void * weight_cma,
        uint32_t output_spm_addr,
        uint64_t flow,
        uint16_t cache_cell = 0,
        uint32_t act_group_stride_bytes = 0,
        bool mvin_activation = true,
        uint32_t vec_spm_addr = NPU_DECODE_ATTN_HIDDEN_ACT_BASE) {
    const npu_decode_gemv_layout & layout = preloaded->pingpong_layout;
    const uint32_t act_bytes = npu_decode_act_payload_bytes(layout);
    if (mvin_activation) {
        npu_decode_pack_activation(
                act_cma,
                act_bytes,
                layout,
                act_data,
                act_type,
                act_nb0,
                act_nb1,
                preloaded->act_scale.data(),
                preloaded->k,
                0);
        npu_decode_stream_unsupported_void(
                op_name,
                act_cma,
                vec_spm_addr,
                act_bytes,
                2,
                NPU_DECODE_GEMV_INPUT_TYPE_ACT);
    }
    std::memcpy(scale_cma, preloaded->pingpong_scale.data(), preloaded->pingpong_layout.scale_bytes);
    npu_decode_stream_unsupported_void(
            op_name,
            scale_cma,
            NPU_DECODE_ATTN_WEIGHT_SCALE_SPM_BASE,
            preloaded->pingpong_layout.scale_bytes,
            2,
            NPU_DECODE_GEMV_INPUT_TYPE_DATA,
            0,
            0);
    std::memcpy(weight_cma, preloaded->pingpong_weight.data(), preloaded->pingpong_layout.weight_bytes);
    npu_decode_stream_unsupported_void(
            op_name,
            weight_cma,
            NPU_DECODE_ATTN_INPUT_SPM_BASE,
            preloaded->pingpong_layout.weight_bytes,
            1,
            NPU_DECODE_GEMV_INPUT_TYPE_WEIGHT);
    npu_decode_stream_unsupported_void(
            op_name,
            NPU_DECODE_ATTN_INPUT_SPM_BASE,
            vec_spm_addr,
            static_cast<uint16_t>(preloaded->k),
            static_cast<uint16_t>(preloaded->out_channels),
            output_spm_addr,
            NPU_DECODE_ATTN_WEIGHT_SCALE_SPM_BASE,
            NPU_DECODE_GEMV_MODE_W4A16,
            flow,
            std::numeric_limits<uint32_t>::max(),
            cache_cell,
            act_group_stride_bytes);
}

static int64_t & npu_profile_next_layer_id() {
    static int64_t next_layer_id = 0;
    return next_layer_id;
}

static bool & npu_profile_session_started() {
    static bool started = false;
    return started;
}

struct npu_backend_context {
    npu_tiling_config config;
};

struct npu_buffer_context {
    void * ptr = nullptr;
    bool own = false;
};

static void npu_buffer_free(ggml_backend_buffer_t buffer) {
    npu_buffer_context * ctx = static_cast<npu_buffer_context *>(buffer->context);
    if (ctx != nullptr && ctx->own) {
        ggml_aligned_free(ctx->ptr, buffer->size);
    }
    delete ctx;
}

static void * npu_buffer_get_base(ggml_backend_buffer_t buffer) {
    npu_buffer_context * ctx = static_cast<npu_buffer_context *>(buffer->context);
    return ctx != nullptr ? ctx->ptr : nullptr;
}

static void npu_buffer_memset_tensor(
        ggml_backend_buffer_t buffer,
        struct ggml_tensor * tensor,
        uint8_t value,
        size_t offset,
        size_t size) {
    GGML_UNUSED(buffer);
    std::memset(static_cast<char *>(tensor->data) + offset, value, size);
}

static void npu_buffer_set_tensor(
        ggml_backend_buffer_t buffer,
        struct ggml_tensor * tensor,
        const void * data,
        size_t offset,
        size_t size) {
    GGML_UNUSED(buffer);
    std::memcpy(static_cast<char *>(tensor->data) + offset, data, size);
}

static void npu_buffer_get_tensor(
        ggml_backend_buffer_t buffer,
        const struct ggml_tensor * tensor,
        void * data,
        size_t offset,
        size_t size) {
    GGML_UNUSED(buffer);
    std::memcpy(data, static_cast<const char *>(tensor->data) + offset, size);
}

static bool npu_buffer_cpy_tensor(
        ggml_backend_buffer_t buffer,
        const struct ggml_tensor * src,
        struct ggml_tensor * dst) {
    GGML_UNUSED(buffer);
    ggml_backend_tensor_copy(const_cast<struct ggml_tensor *>(src), dst);
    return true;
}

static void npu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    npu_buffer_context * ctx = static_cast<npu_buffer_context *>(buffer->context);
    if (ctx != nullptr && ctx->ptr != nullptr) {
        std::memset(ctx->ptr, value, buffer->size);
    }
}

static const ggml_backend_buffer_i npu_buffer_iface = {
    /* .free_buffer   = */ npu_buffer_free,
    /* .get_base      = */ npu_buffer_get_base,
    /* .init_tensor   = */ nullptr,
    /* .memset_tensor = */ npu_buffer_memset_tensor,
    /* .set_tensor    = */ npu_buffer_set_tensor,
    /* .get_tensor    = */ npu_buffer_get_tensor,
    /* .cpy_tensor    = */ npu_buffer_cpy_tensor,
    /* .clear         = */ npu_buffer_clear,
    /* .reset         = */ nullptr,
};

