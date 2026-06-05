bool ggml_backend_npu_i8_gemm_raw_cma(
        const char * op_name,
        const int8_t * weight_cma_kxm,
        int64_t m,
        int64_t k,
        int64_t weight_stride_m,
        const int8_t * act_cma_nxk,
        int64_t n,
        int64_t act_stride_k,
        int32_t * out_nxm,
        int64_t out_stride_m) {
    using namespace ggml_npu;

    std::string error;
    if (!npu_raw_i8_gemm_validate(
            op_name,
            weight_cma_kxm,
            m,
            k,
            weight_stride_m,
            act_cma_nxk,
            n,
            act_stride_k,
            out_nxm,
            out_stride_m,
            &error)) {
        return false;
    }

    int64_t tile_m = std::min<int64_t>(m, 240);
    int64_t tile_n = std::min<int64_t>(n, 64);
    int64_t tile_m_stride = npu_align_up_i64(tile_m, NPU_GEMM_PLAN_STRIDE_ALIGNMENT);
    bool tile_fits = false;
    while (tile_m > 0 && tile_n > 0) {
        tile_m_stride = npu_align_up_i64(tile_m, NPU_GEMM_PLAN_STRIDE_ALIGNMENT);
        const uint64_t max_act_bytes = static_cast<uint64_t>(tile_n) * static_cast<uint64_t>(act_stride_k);
        const uint64_t max_weight_bytes = static_cast<uint64_t>(k) * static_cast<uint64_t>(tile_m_stride);
        const uint64_t max_out_bytes = static_cast<uint64_t>(tile_n) * static_cast<uint64_t>(tile_m_stride) * sizeof(int32_t);
        const uint64_t spm_weight_addr = static_cast<uint64_t>(npu_align_up_i64(static_cast<int64_t>(max_act_bytes), NPU_SPM_ALIGNMENT));
        const uint64_t spm_total = spm_weight_addr + max_weight_bytes;
        const uint64_t acc_scratch_addr = static_cast<uint64_t>(npu_align_up_i64(static_cast<int64_t>(max_out_bytes), NPU_GEMM_PLAN_ADDR_ALIGNMENT));
        const uint64_t acc_total = acc_scratch_addr + max_out_bytes;
        if (spm_total + NPU_DEFAULT_GUARD_BYTES <= NPU_DEFAULT_SPM_BYTES &&
                acc_total + NPU_DEFAULT_GUARD_BYTES <= NPU_DEFAULT_ACC_BYTES &&
                npu_fits_u16(tile_m_stride)) {
            tile_fits = true;
            break;
        }
        if (tile_n > 16) {
            tile_n /= 2;
        } else {
            tile_m /= 2;
        }
    }

    if (!tile_fits) {
        if (npu_raw_i8_gemm_debug_log_enabled()) {
            GGML_LOG_WARN("%s: reject %s: no raw GEMM tile fits SPM/ACC capacity\n",
                    __func__, op_name != nullptr ? op_name : "(unnamed)");
        }
        return false;
    }

    tile_m_stride = npu_align_up_i64(tile_m, NPU_GEMM_PLAN_STRIDE_ALIGNMENT);
    const size_t max_act_bytes = static_cast<size_t>(tile_n * act_stride_k);
    const size_t max_acc_out_bytes = static_cast<size_t>(tile_n * tile_m_stride * sizeof(int32_t));
    const size_t max_host_out_bytes = static_cast<size_t>(tile_n * tile_m * sizeof(int32_t));
    const uint32_t spm_act_addr = 0;
    const uint32_t spm_weight_addr = static_cast<uint32_t>(npu_align_up_i64(static_cast<int64_t>(max_act_bytes), NPU_SPM_ALIGNMENT));
    const uint32_t acc_out_addr = 0;
    const uint32_t acc_scratch_addr = static_cast<uint32_t>(npu_align_up_i64(static_cast<int64_t>(max_acc_out_bytes), NPU_GEMM_PLAN_ADDR_ALIGNMENT));

    if (npu_init() != 0) {
        if (npu_raw_i8_gemm_debug_log_enabled()) {
            GGML_LOG_WARN("%s: npu_init failed for %s\n", __func__, op_name != nullptr ? op_name : "(unnamed)");
        }
        return false;
    }

    void * out_cma = npu_mem_alloc(max_host_out_bytes);
    if (out_cma == nullptr) {
        if (npu_raw_i8_gemm_debug_log_enabled()) {
            GGML_LOG_WARN("%s: output CMA allocation failed for %s\n", __func__, op_name != nullptr ? op_name : "(unnamed)");
        }
        return false;
    }

    for (int64_t n0 = 0; n0 < n; n0 += tile_n) {
        const int64_t cur_n = std::min(tile_n, n - n0);
        for (int64_t m0 = 0; m0 < m; m0 += tile_m) {
            const int64_t cur_m = std::min(tile_m, m - m0);
            const MvinConfig act_mvin_cfg = {
                const_cast<int8_t *>(act_cma_nxk + n0 * act_stride_k),
                spm_act_addr,
                static_cast<uint32_t>(k - 1),
                static_cast<uint32_t>(cur_n - 1),
                static_cast<uint16_t>(act_stride_k),
                static_cast<uint32_t>(act_stride_k),
                1,
                0,
                false,
                false,
                false,
                0,
                0,
                0,
            };
            const MvinConfig weight_mvin_cfg = {
                const_cast<int8_t *>(weight_cma_kxm + m0),
                spm_weight_addr,
                static_cast<uint32_t>(cur_m - 1),
                static_cast<uint32_t>(k - 1),
                static_cast<uint16_t>(tile_m_stride),
                static_cast<uint32_t>(weight_stride_m),
                1,
                1,
                false,
                false,
                false,
                0,
                0,
                0,
            };
            npu_dma_mvin_async(0, &act_mvin_cfg);
            npu_dma_wait_mvin(1u << 0);
            npu_dma_mvin_async(1, &weight_mvin_cfg);
            npu_dma_wait_mvin(1u << 1);

            npu_gemm_plan_run_ex(
                    spm_act_addr,
                    spm_weight_addr,
                    acc_out_addr,
                    acc_scratch_addr,
                    0,
                    static_cast<uint16_t>(cur_n),
                    static_cast<uint16_t>(cur_m),
                    static_cast<uint16_t>(k),
                    static_cast<uint16_t>(act_stride_k),
                    static_cast<uint16_t>(tile_m_stride),
                    static_cast<uint16_t>(tile_m_stride),
                    0,
                    false,
                    false,
                    false);

            npu_dma_mvout_ex(
                    out_cma,
                    acc_out_addr,
                    static_cast<uint32_t>(cur_m - 1),
                    static_cast<uint32_t>(cur_n - 1),
                    static_cast<uint16_t>(tile_m_stride),
                    static_cast<uint32_t>(cur_m),
                    1,
                    1,
                    true,
                    false,
                    0,
                    0,
                    false);

            const int32_t * tile_out = static_cast<const int32_t *>(out_cma);
            for (int64_t row = 0; row < cur_n; ++row) {
                std::memcpy(
                        out_nxm + (n0 + row) * out_stride_m + m0,
                        tile_out + row * cur_m,
                        static_cast<size_t>(cur_m) * sizeof(int32_t));
            }
        }
    }

    npu_mem_free(out_cma);
    return true;
}

bool ggml_backend_npu_i8_gemm_raw_packed(
        const char * op_name,
        const int8_t * weight_kxm,
        int64_t m,
        int64_t k,
        int64_t weight_stride_m,
        const int8_t * act_nxk,
        int64_t n,
        int64_t act_stride_k,
        int32_t * out_nxm,
        int64_t out_stride_m) {
    using namespace ggml_npu;

    std::string error;
    if (!npu_raw_i8_gemm_validate(
            op_name,
            weight_kxm,
            m,
            k,
            weight_stride_m,
            act_nxk,
            n,
            act_stride_k,
            out_nxm,
            out_stride_m,
            &error)) {
        return false;
    }

    int64_t tile_m = std::min<int64_t>(m, 240);
    int64_t tile_n = std::min<int64_t>(n, 128);
    int64_t tile_m_stride = npu_align_up_i64(tile_m, NPU_GEMM_PLAN_STRIDE_ALIGNMENT);
    bool tile_fits = false;
    while (tile_m > 0 && tile_n > 0) {
        tile_m_stride = npu_align_up_i64(tile_m, NPU_GEMM_PLAN_STRIDE_ALIGNMENT);
        const uint64_t max_act_bytes = static_cast<uint64_t>(tile_n) * static_cast<uint64_t>(act_stride_k);
        const uint64_t max_weight_bytes = static_cast<uint64_t>(k) * static_cast<uint64_t>(tile_m_stride);
        const uint64_t max_out_bytes = static_cast<uint64_t>(tile_n) * static_cast<uint64_t>(tile_m_stride) * sizeof(int32_t);
        const uint64_t spm_weight_addr = static_cast<uint64_t>(npu_align_up_i64(static_cast<int64_t>(max_act_bytes), NPU_SPM_ALIGNMENT));
        const uint64_t spm_total = spm_weight_addr + max_weight_bytes;
        const uint64_t acc_scratch_addr = static_cast<uint64_t>(npu_align_up_i64(static_cast<int64_t>(max_out_bytes), NPU_GEMM_PLAN_ADDR_ALIGNMENT));
        const uint64_t acc_total = acc_scratch_addr + max_out_bytes;
        if (spm_total + NPU_DEFAULT_GUARD_BYTES <= NPU_DEFAULT_SPM_BYTES &&
                acc_total + NPU_DEFAULT_GUARD_BYTES <= NPU_DEFAULT_ACC_BYTES &&
                npu_fits_u16(tile_m_stride)) {
            tile_fits = true;
            break;
        }
        if (tile_n > 16) {
            tile_n /= 2;
        } else {
            tile_m /= 2;
        }
    }

    if (!tile_fits) {
        if (npu_raw_i8_gemm_debug_log_enabled()) {
            GGML_LOG_WARN("%s: reject %s: no raw GEMM tile fits SPM/ACC capacity\n",
                    __func__, op_name != nullptr ? op_name : "(unnamed)");
        }
        return false;
    }

    tile_m_stride = npu_align_up_i64(tile_m, NPU_GEMM_PLAN_STRIDE_ALIGNMENT);
    const size_t max_act_bytes = static_cast<size_t>(tile_n * act_stride_k);
    const size_t max_weight_bytes = static_cast<size_t>(k * tile_m_stride);
    const size_t max_acc_out_bytes = static_cast<size_t>(tile_n * tile_m_stride * sizeof(int32_t));
    const size_t max_host_out_bytes = static_cast<size_t>(tile_n * tile_m * sizeof(int32_t));
    const uint32_t spm_act_addr = 0;
    const uint32_t spm_weight_addr = static_cast<uint32_t>(npu_align_up_i64(static_cast<int64_t>(max_act_bytes), NPU_SPM_ALIGNMENT));
    const uint32_t acc_out_addr = 0;
    const uint32_t acc_scratch_addr = static_cast<uint32_t>(npu_align_up_i64(static_cast<int64_t>(max_acc_out_bytes), NPU_GEMM_PLAN_ADDR_ALIGNMENT));

    if (npu_init() != 0) {
        if (npu_raw_i8_gemm_debug_log_enabled()) {
            GGML_LOG_WARN("%s: npu_init failed for %s\n", __func__, op_name != nullptr ? op_name : "(unnamed)");
        }
        return false;
    }

    void * act_cma = npu_mem_alloc(max_act_bytes);
    void * weight_cma = npu_mem_alloc(max_weight_bytes);
    void * out_cma = npu_mem_alloc(max_host_out_bytes);
    if (act_cma == nullptr || weight_cma == nullptr || out_cma == nullptr) {
        npu_mem_free(act_cma);
        npu_mem_free(weight_cma);
        npu_mem_free(out_cma);
        if (npu_raw_i8_gemm_debug_log_enabled()) {
            GGML_LOG_WARN("%s: CMA allocation failed for %s\n", __func__, op_name != nullptr ? op_name : "(unnamed)");
        }
        return false;
    }

    std::vector<int8_t> act_tile(max_act_bytes);
    std::vector<int8_t> weight_tile(max_weight_bytes);

    for (int64_t n0 = 0; n0 < n; n0 += tile_n) {
        const int64_t cur_n = std::min(tile_n, n - n0);
        for (int64_t m0 = 0; m0 < m; m0 += tile_m) {
            const int64_t cur_m = std::min(tile_m, m - m0);
            if (npu_raw_i8_gemm_debug_log_enabled()) {
                GGML_LOG_INFO("%s: %s tile n0=%" PRId64 " m0=%" PRId64 " n=%" PRId64 " m=%" PRId64 " k=%" PRId64 " a_stride=%" PRId64 " b_stride=%" PRId64 "\n",
                        __func__,
                        op_name != nullptr ? op_name : "(unnamed)",
                        n0,
                        m0,
                        cur_n,
                        cur_m,
                        k,
                        act_stride_k,
                        tile_m_stride);
            }
            std::fill(act_tile.begin(), act_tile.end(), 0);
            std::fill(weight_tile.begin(), weight_tile.end(), 0);

            for (int64_t row = 0; row < cur_n; ++row) {
                std::memcpy(
                        act_tile.data() + row * act_stride_k,
                        act_nxk + (n0 + row) * act_stride_k,
                        static_cast<size_t>(k));
            }
            for (int64_t kk = 0; kk < k; ++kk) {
                std::memcpy(
                        weight_tile.data() + kk * tile_m_stride,
                        weight_kxm + kk * weight_stride_m + m0,
                        static_cast<size_t>(cur_m));
            }

            const size_t cur_act_bytes = static_cast<size_t>(cur_n * act_stride_k);
            const size_t cur_weight_bytes = static_cast<size_t>(k * tile_m_stride);
            std::memcpy(act_cma, act_tile.data(), cur_act_bytes);
            std::memcpy(weight_cma, weight_tile.data(), cur_weight_bytes);

            const MvinConfig act_mvin_cfg = {
                act_cma,
                spm_act_addr,
                static_cast<uint32_t>(k - 1),
                static_cast<uint32_t>(cur_n - 1),
                static_cast<uint16_t>(act_stride_k),
                static_cast<uint32_t>(act_stride_k),
                1,
                0,
                false,
                false,
                false,
                0,
                0,
                0,
            };
            const MvinConfig weight_mvin_cfg = {
                weight_cma,
                spm_weight_addr,
                static_cast<uint32_t>(cur_m - 1),
                static_cast<uint32_t>(k - 1),
                static_cast<uint16_t>(tile_m_stride),
                static_cast<uint32_t>(tile_m_stride),
                1,
                1,
                false,
                false,
                false,
                0,
                0,
                0,
            };
            npu_dma_mvin_async(0, &act_mvin_cfg);
            npu_dma_mvin_async(1, &weight_mvin_cfg);
            npu_dma_wait_mvin((1u << 0) | (1u << 1));

            npu_gemm_plan_run_ex(
                    spm_act_addr,
                    spm_weight_addr,
                    acc_out_addr,
                    acc_scratch_addr,
                    0,
                    static_cast<uint16_t>(cur_n),
                    static_cast<uint16_t>(cur_m),
                    static_cast<uint16_t>(k),
                    static_cast<uint16_t>(act_stride_k),
                    static_cast<uint16_t>(tile_m_stride),
                    static_cast<uint16_t>(tile_m_stride),
                    0,
                    false,
                    false,
                    false);

            npu_dma_mvout_ex(
                    out_cma,
                    acc_out_addr,
                    static_cast<uint32_t>(cur_m - 1),
                    static_cast<uint32_t>(cur_n - 1),
                    static_cast<uint16_t>(tile_m_stride),
                    static_cast<uint32_t>(cur_m),
                    1,
                    1,
                    true,
                    false,
                    0,
                    0,
                    false);

            const int32_t * tile_out = static_cast<const int32_t *>(out_cma);
            for (int64_t row = 0; row < cur_n; ++row) {
                std::memcpy(
                        out_nxm + (n0 + row) * out_stride_m + m0,
                        tile_out + row * cur_m,
                        static_cast<size_t>(cur_m) * sizeof(int32_t));
            }
        }
    }

    npu_mem_free(act_cma);
    npu_mem_free(weight_cma);
    npu_mem_free(out_cma);
    return true;
}
