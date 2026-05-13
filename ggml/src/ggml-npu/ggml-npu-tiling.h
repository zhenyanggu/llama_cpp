#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace ggml_npu {

enum class npu_gemm_stationary_mode {
    activation,
    weight,
};

struct npu_gemm_tiling_params {
    int64_t n = 0;
    int64_t m = 0;
    int64_t k = 0;
    int64_t spm_bytes = 0;
    int64_t acc_bytes = 0;
    int64_t sa_rows = 16;
    int64_t sa_cols = 16;
    int64_t tk_align = 16;
    int64_t metadata_words_per_channel = 0;
    int64_t acc_tile_buffers = 3;
    int64_t spm_factor_a = 1;
    int64_t spm_factor_b = 1;
    int64_t max_u = 255;
    int64_t max_v = 255;
    int64_t max_tk = 4096;
    double lambda_dma = 0.0;
};

struct npu_gemm_tiling_result {
    bool valid = false;
    npu_gemm_stationary_mode mode = npu_gemm_stationary_mode::activation;
    int64_t u = 0;
    int64_t v = 0;
    int64_t tk = 0;
    int64_t n_tiles = 0;
    int64_t m_tiles = 0;
    int64_t k_tiles = 0;
    uint64_t q_a = 0;
    uint64_t q_b = 0;
    uint64_t q_meta = 0;
    uint64_t q_data = 0;
    uint64_t num_dma = 0;
    double cost = 0.0;
};

int64_t npu_ceil_div(int64_t a, int64_t b);
int64_t npu_align_down(int64_t x, int64_t align);
const char * npu_stationary_mode_name(npu_gemm_stationary_mode mode);

npu_gemm_tiling_result npu_search_gemm_tiling(const npu_gemm_tiling_params & params);

} // namespace ggml_npu
