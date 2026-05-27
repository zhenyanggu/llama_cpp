#include "ggml-npu-tiling.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>

namespace ggml_npu {

int64_t npu_ceil_div(int64_t a, int64_t b) {
    if (b <= 0) {
        return 0;
    }
    if (a <= 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

int64_t npu_align_down(int64_t x, int64_t align) {
    if (align <= 0) {
        return x;
    }
    if (x <= 0) {
        return 0;
    }
    return (x / align) * align;
}

static int64_t npu_align_up(int64_t x, int64_t align) {
    if (align <= 0) {
        return x;
    }
    if (x <= 0) {
        return 0;
    }
    return ((x + align - 1) / align) * align;
}

const char * npu_stationary_mode_name(npu_gemm_stationary_mode mode) {
    switch (mode) {
        case npu_gemm_stationary_mode::activation:
            return "activation_stationary";
        case npu_gemm_stationary_mode::weight:
            return "weight_stationary";
    }
    return "unknown";
}

static bool npu_force_mode(const char * env, npu_gemm_stationary_mode mode) {
    if (env == nullptr || env[0] == '\0' || std::strcmp(env, "auto") == 0) {
        return false;
    }
    if (mode == npu_gemm_stationary_mode::activation) {
        return std::strcmp(env, "activation") != 0 &&
               std::strcmp(env, "activation_stationary") != 0;
    }
    return std::strcmp(env, "weight") != 0 &&
           std::strcmp(env, "weight_stationary") != 0;
}

static void npu_update_best(
        npu_gemm_tiling_result candidate,
        npu_gemm_tiling_result * best) {
    if (!candidate.valid) {
        return;
    }
    if (!best->valid ||
        candidate.cost < best->cost ||
        (candidate.cost == best->cost && candidate.u * candidate.v > best->u * best->v)) {
        *best = candidate;
    }
}

static int64_t npu_fit_tk_to_banks(
        int64_t tk,
        int64_t tk_align,
        int64_t u,
        int64_t v_dma,
        int64_t spm_bytes,
        int64_t a_bank_bytes,
        int64_t w_bank_bytes) {
    while (tk >= tk_align) {
        const int64_t a_stride_k = npu_align_up(tk, 32);
        const int64_t a_need = u * a_stride_k;
        const int64_t w_need = v_dma * tk;
        const bool split_banks = a_bank_bytes > 0 && w_bank_bytes > 0;
        if ((split_banks && a_need <= a_bank_bytes && w_need <= w_bank_bytes) ||
                (!split_banks && a_need + w_need <= spm_bytes)) {
            return tk;
        }
        tk = npu_align_down(tk - 1, tk_align);
    }
    return 0;
}

npu_gemm_tiling_result npu_search_gemm_tiling(const npu_gemm_tiling_params & params) {
    npu_gemm_tiling_result best;
    if (params.n <= 0 || params.m <= 0 || params.k <= 0 ||
        params.spm_bytes <= 0 || params.acc_bytes <= 0 ||
        params.sa_rows <= 0 || params.sa_cols <= 0 ||
        params.tk_align <= 0 || params.acc_tile_buffers <= 0 ||
        params.spm_factor_a <= 0 ||
        params.spm_factor_b <= 0) {
        return best;
    }

    const char * forced_mode = std::getenv("GGML_NPU_TILING_MODE");
    const int64_t max_u = std::min(params.n, params.max_u > 0 ? params.max_u : params.n);
    const int64_t max_v = std::min(params.m, params.max_v > 0 ? params.max_v : params.m);
    int64_t max_tk = std::min(params.k, params.max_tk > 0 ? params.max_tk : params.k);
#if defined(GGML_NPU_VERSA_P_RUNTIME)
    max_tk = std::min<int64_t>(max_tk, 768);
#endif
    const int64_t o_bytes = params.o_bank_bytes > 0 ? params.o_bank_bytes : params.acc_bytes;
    const int64_t acc_words = o_bytes / 4;
    const int64_t meta_words = params.meta_bytes > 0 ? params.meta_bytes / 4 : 0;

    const int64_t first_u = max_u < params.sa_rows ? max_u : params.sa_rows;
    const int64_t first_v = max_v < params.sa_cols ? max_v : params.sa_cols;
    std::vector<int64_t> u_values;
    std::vector<int64_t> v_values;
    for (int64_t u = first_u; u <= max_u; u += params.sa_rows) {
        u_values.push_back(u);
    }
    if (u_values.empty() || u_values.back() != max_u) {
        u_values.push_back(max_u);
    }
    for (int64_t v = first_v; v <= max_v; v += params.sa_cols) {
        v_values.push_back(v);
    }
    if (v_values.empty() || v_values.back() != max_v) {
        v_values.push_back(max_v);
    }

    for (const int64_t u : u_values) {
        for (const int64_t v : v_values) {
            const int64_t u_dma = npu_align_up(u, 32);
            const int64_t v_dma = npu_align_up(v, 32);
            // Phase-1 Versa_P W/GEMM path is organized in 32-column groups.
            // Board replay shows raw accumulator corruption for tail-N groups
            // such as 208 or 240 columns, so keep output-channel tiles on full
            // groups and let model-level tails fall back to CPU if needed.
            if ((v % 32) != 0) {
                continue;
            }
            // Current Versa_P hardware has a dedicated metadata buffer for
            // bias/scale; the O bank only holds int32 accumulator/output words.
            if (v_dma * (params.acc_tile_buffers * u_dma) + params.fixed_acc_words > acc_words) {
                continue;
            }
            if (meta_words > 0 &&
                    params.fixed_meta_words + params.metadata_words_per_channel * v_dma > meta_words) {
                continue;
            }

            const int64_t n_tiles = npu_ceil_div(params.n, u);
            const int64_t m_tiles = npu_ceil_div(params.m, v);
            const uint64_t q_a = static_cast<uint64_t>(m_tiles) *
                static_cast<uint64_t>(params.n) * static_cast<uint64_t>(params.k);
            const uint64_t q_b = static_cast<uint64_t>(n_tiles) *
                static_cast<uint64_t>(params.k) * static_cast<uint64_t>(params.m);
            const uint64_t q_meta = static_cast<uint64_t>(n_tiles) *
                static_cast<uint64_t>(params.metadata_words_per_channel) *
                static_cast<uint64_t>(params.m) * 4ull;
            const uint64_t q_data = q_a + q_b + q_meta;

            if (!npu_force_mode(forced_mode, npu_gemm_stationary_mode::activation)) {
                const int64_t denom = std::max(
                    params.spm_factor_a * u_dma + params.spm_factor_b * v_dma,
                    u_dma + v_dma);
                int64_t tk = denom > 0 ? params.spm_bytes / denom : 0;
                tk = npu_align_down(std::min(tk, max_tk), params.tk_align);
                tk = npu_fit_tk_to_banks(
                    tk, params.tk_align, u_dma, v_dma, params.spm_bytes,
                    params.a_bank_bytes, params.w_bank_bytes);
                if (tk >= params.tk_align) {
                    const int64_t k_tiles = npu_ceil_div(params.k, tk);
                    const uint64_t dma_per_k = 1ull + static_cast<uint64_t>(npu_ceil_div(v, params.sa_cols));
                    const uint64_t num_dma = static_cast<uint64_t>(n_tiles) *
                        static_cast<uint64_t>(m_tiles) *
                        static_cast<uint64_t>(k_tiles) * dma_per_k;
                    npu_update_best({
                        true,
                        npu_gemm_stationary_mode::activation,
                        u,
                        v,
                        tk,
                        n_tiles,
                        m_tiles,
                        k_tiles,
                        q_a,
                        q_b,
                        q_meta,
                        q_data,
                        num_dma,
                        static_cast<double>(q_data) + params.lambda_dma * static_cast<double>(num_dma),
                    }, &best);
                }
            }

            if (!npu_force_mode(forced_mode, npu_gemm_stationary_mode::weight)) {
                const int64_t denom = std::max(
                    params.spm_factor_a * u_dma + params.spm_factor_b * v_dma,
                    u_dma + v_dma);
                int64_t tk = denom > 0 ? params.spm_bytes / denom : 0;
                tk = npu_align_down(std::min(tk, max_tk), params.tk_align);
                tk = npu_fit_tk_to_banks(
                    tk, params.tk_align, u_dma, v_dma, params.spm_bytes,
                    params.a_bank_bytes, params.w_bank_bytes);
                if (tk >= params.tk_align) {
                    const int64_t k_tiles = npu_ceil_div(params.k, tk);
                    const uint64_t dma_per_k = 1ull + static_cast<uint64_t>(npu_ceil_div(u, params.sa_rows));
                    const uint64_t num_dma = static_cast<uint64_t>(n_tiles) *
                        static_cast<uint64_t>(m_tiles) *
                        static_cast<uint64_t>(k_tiles) * dma_per_k;
                    npu_update_best({
                        true,
                        npu_gemm_stationary_mode::weight,
                        u,
                        v,
                        tk,
                        n_tiles,
                        m_tiles,
                        k_tiles,
                        q_a,
                        q_b,
                        q_meta,
                        q_data,
                        num_dma,
                        static_cast<double>(q_data) + params.lambda_dma * static_cast<double>(num_dma),
                    }, &best);
                }
            }
        }
    }

    return best;
}

} // namespace ggml_npu
