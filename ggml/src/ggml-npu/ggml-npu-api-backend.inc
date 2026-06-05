
ggml_backend_buffer_type_t ggml_backend_npu_buffer_type(void) {
    return ggml_backend_dev_buffer_type(ggml_backend_reg_dev_get(ggml_backend_npu_reg(), 0));
}

ggml_backend_reg_t ggml_backend_npu_reg(void) {
    using namespace ggml_npu;

    static npu_reg_context ctx;
    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ npu_reg_iface,
        /* .context     = */ &ctx,
    };
    static bool initialized = false;

    if (!initialized) {
        ctx.device = ggml_backend_device {
            /* .iface   = */ npu_device_iface,
            /* .reg     = */ &reg,
            /* .context = */ nullptr,
        };

        ctx.buffer_type = ggml_backend_buffer_type {
            /* .iface   = */ npu_buffer_type_iface,
            /* .device  = */ &ctx.device,
            /* .context = */ nullptr,
        };

        initialized = true;
    }

    return &reg;
}

ggml_backend_t ggml_backend_npu_init(void) {
    return ggml_backend_dev_init(ggml_backend_reg_dev_get(ggml_backend_npu_reg(), 0), nullptr);
}

bool ggml_backend_is_npu(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, ggml_npu::ggml_backend_npu_guid());
}

void ggml_backend_npu_profile_summary_start(void) {
    ggml_npu::npu_summary_session_start();
}

void ggml_backend_npu_profile_summary_stop(ggml_npu_profile_summary * out) {
    ggml_npu::npu_summary_session_stop(out);
}

void ggml_backend_npu_w8a8_clear(void) {
    ggml_npu::npu_clear_aicas_w8a8_table();
}

bool ggml_backend_npu_w8a8_register(
        const char * weight_name,
        float act_scale,
        int32_t act_scale_q8_24,
        int32_t act_zero_point_u8,
        const float * weight_scale,
        size_t weight_scale_len,
        const int32_t * sum_w,
        size_t sum_w_len,
        const float * smooth_scale,
        size_t smooth_scale_len) {
    return ggml_npu::npu_register_aicas_w8a8(
        weight_name,
        act_scale,
        act_scale_q8_24,
        act_zero_point_u8,
        weight_scale,
        weight_scale_len,
        sum_w,
        sum_w_len,
        smooth_scale,
        smooth_scale_len);
}

bool ggml_backend_npu_w8a8_preload(const struct ggml_tensor * weight_tensor) {
    std::string error;
    const bool ok = ggml_npu::npu_preload_aicas_w8a8_tensor(weight_tensor, &error);
    if (!ok && ggml_npu::npu_debug_log_enabled()) {
        GGML_LOG_WARN("%s: preload failed for %s: %s\n",
                __func__,
                weight_tensor && weight_tensor->name[0] ? weight_tensor->name : "(unnamed)",
                error.c_str());
    }
    return ok;
}

void ggml_backend_npu_w8a8_preload_clear(void) {
    ggml_npu::npu_clear_preloaded_weight_cache();
}

void * ggml_backend_npu_mem_alloc(size_t size) {
    if (npu_init() != 0) {
        return nullptr;
    }
    return npu_mem_alloc(size);
}

void ggml_backend_npu_mem_free(void * ptr) {
    npu_mem_free(ptr);
}

extern "C" bool ggml_backend_npu_log8pv_attention_ex(
        const int8_t * q,
        uint32_t q_rows,
        const int8_t * k,
        const int8_t * v,
        uint32_t kv_tokens,
        uint32_t q_row_start,
        bool causal_mask,
        uint32_t gamma16_fix,
        int32_t * output,
        uint32_t output_stride_elems,
        uint32_t timeout_ms,
        ggml_backend_npu_log8pv_attention_profile * profile) {
    if (npu_init() != 0) {
        return false;
    }

    npu_log8pv_attention_profile runtime_profile = {};
    const bool ok = npu_attention_log8pv_run_ex(
            q,
            q_rows,
            k,
            v,
            kv_tokens,
            q_row_start,
            causal_mask,
            gamma16_fix,
            output,
            output_stride_elems,
            timeout_ms,
            profile != nullptr ? &runtime_profile : nullptr);
    if (ok && profile != nullptr) {
        profile->kv_tokens = runtime_profile.kv_tokens;
        profile->q_rows = runtime_profile.q_rows;
        profile->q_row_start = runtime_profile.q_row_start;
        profile->exec_rows = runtime_profile.exec_rows;
        profile->chunks = runtime_profile.chunks;
        profile->group_size = runtime_profile.group_size;
        profile->kv_reuse_hit = runtime_profile.kv_reuse_hit;
        profile->workspace_reuse = runtime_profile.workspace_reuse;
        profile->total_us = runtime_profile.total_us;
        profile->mvin_q_us = runtime_profile.mvin_q_us;
        profile->mvin_k_us = runtime_profile.mvin_k_us;
        profile->mvin_v_us = runtime_profile.mvin_v_us;
        profile->qk_us = runtime_profile.qk_us;
        profile->logp_mvout_us = runtime_profile.logp_mvout_us;
        profile->mvin_p_us = runtime_profile.mvin_p_us;
        profile->pv_us = runtime_profile.pv_us;
        profile->pv_mvout_us = runtime_profile.pv_mvout_us;
        profile->qk_overlap_us = runtime_profile.qk_overlap_us;
        profile->pv_overlap_us = runtime_profile.pv_overlap_us;
    }
    return ok;
}

extern "C" bool ggml_backend_npu_log8pv_attention_group(
        const int8_t * q_group,
        uint32_t group_size,
        uint32_t q_rows,
        const int8_t * k,
        const int8_t * v,
        uint32_t kv_tokens,
        uint32_t q_row_start,
        bool causal_mask,
        const uint32_t * gamma16_fix_group,
        int32_t * output_group,
        uint32_t output_stride_elems,
        uint32_t timeout_ms,
        ggml_backend_npu_log8pv_attention_profile * profile_group) {
    if (npu_init() != 0) {
        return false;
    }

    std::vector<npu_log8pv_attention_profile> runtime_profiles(
            profile_group != nullptr ? group_size : 0);
    const bool ok = npu_attention_log8pv_run_group(
            q_group,
            group_size,
            q_rows,
            k,
            v,
            kv_tokens,
            q_row_start,
            causal_mask,
            gamma16_fix_group,
            output_group,
            output_stride_elems,
            timeout_ms,
            profile_group != nullptr ? runtime_profiles.data() : nullptr);
    if (ok && profile_group != nullptr) {
        for (uint32_t i = 0; i < group_size; ++i) {
            const auto & runtime_profile = runtime_profiles[i];
            auto & profile = profile_group[i];
            profile.kv_tokens = runtime_profile.kv_tokens;
            profile.q_rows = runtime_profile.q_rows;
            profile.q_row_start = runtime_profile.q_row_start;
            profile.exec_rows = runtime_profile.exec_rows;
            profile.chunks = runtime_profile.chunks;
            profile.group_size = runtime_profile.group_size;
            profile.kv_reuse_hit = runtime_profile.kv_reuse_hit;
            profile.workspace_reuse = runtime_profile.workspace_reuse;
            profile.total_us = runtime_profile.total_us;
            profile.mvin_q_us = runtime_profile.mvin_q_us;
            profile.mvin_k_us = runtime_profile.mvin_k_us;
            profile.mvin_v_us = runtime_profile.mvin_v_us;
            profile.qk_us = runtime_profile.qk_us;
            profile.logp_mvout_us = runtime_profile.logp_mvout_us;
            profile.mvin_p_us = runtime_profile.mvin_p_us;
            profile.pv_us = runtime_profile.pv_us;
            profile.pv_mvout_us = runtime_profile.pv_mvout_us;
            profile.qk_overlap_us = runtime_profile.qk_overlap_us;
            profile.pv_overlap_us = runtime_profile.pv_overlap_us;
        }
    }
    return ok;
}

extern "C" bool ggml_backend_npu_text_log8pv_attention(
        const int8_t * q,
        const int8_t * k,
        const int8_t * v,
        uint32_t tokens,
        bool causal_mask,
        uint32_t gamma16_fix,
        int32_t * output,
        uint32_t output_stride_elems,
        uint32_t timeout_ms) {
    if (npu_init() != 0) {
        return false;
    }
    return npu_attention_log8pv_run(
            q,
            k,
            v,
            tokens,
            causal_mask,
            gamma16_fix,
            output,
            output_stride_elems,
            timeout_ms);
}

void ggml_backend_npu_runtime_shutdown(void) {
    ggml_npu::npu_profile_flush();
    ggml_npu::npu_clear_preloaded_weight_cache();
    npu_destroy();
}

void ggml_backend_npu_decode_runtime_shutdown(void) {
    ggml_npu::npu_decode_profile_flush();
    ggml_npu::npu_decode_release_w16a16_runtime_cma();
    ggml_npu::npu_decode_stream_close_device();
    ggml_npu::npu_decode_mark_runtime_not_ready();
    npu_decode_destroy();
}

void ggml_backend_npu_decode_overlay_mark_inactive(void) {
    using namespace ggml_npu;
    std::lock_guard<std::mutex> lock(npu_decode_overlay_mutex());
    npu_decode_stream_close_device();
    npu_decode_mark_runtime_not_ready();
    npu_decode_overlay_active() = false;
}

void ggml_backend_npu_decode_overlay_mark_active(void) {
    using namespace ggml_npu;
    std::lock_guard<std::mutex> lock(npu_decode_overlay_mutex());
    npu_decode_overlay_active() = true;
}

