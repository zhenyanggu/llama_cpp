#include "llama-graph.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-cparams.h"
#include "llama-model.h"

#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-recurrent.h"

#define GGML_COMMON_DECL_CPP
#include "../ggml/src/ggml-common.h"
#include "../ggml/src/ggml-cpu/aicas-rtl-fp16.h"

#include "../vendor/nlohmann/json.hpp"

#include <cassert>
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <cinttypes>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <future>
#include <iomanip>
#include <limits>
#include <map>
#include <mutex>
#include <regex>
#include <set>
#include <sstream>
#include <unordered_map>

using json = nlohmann::ordered_json;

namespace {

#ifdef GGML_USE_NPU
extern "C" bool ggml_backend_npu_w8a8_register(
        const char * weight_name,
        float act_scale,
        int32_t act_scale_q8_24,
        int32_t act_zero_point_u8,
        const float * weight_scale,
        size_t weight_scale_len,
        const int32_t * sum_w,
        size_t sum_w_len,
        const float * smooth_scale,
        size_t smooth_scale_len);
extern "C" bool ggml_backend_npu_text_log8pv_attention(
        const int8_t * q,
        const int8_t * k,
        const int8_t * v,
        uint32_t tokens,
        bool causal_mask,
        uint32_t gamma16_fix,
        int32_t * output,
        uint32_t output_stride_elems,
        uint32_t timeout_ms);
struct ggml_backend_npu_log8pv_attention_profile {
    uint32_t kv_tokens;
    uint32_t q_rows;
    uint32_t q_row_start;
    uint32_t exec_rows;
    uint32_t chunks;
    uint32_t group_size;
    uint32_t kv_reuse_hit;
    uint32_t workspace_reuse;
    uint64_t total_us;
    uint64_t mvin_q_us;
    uint64_t mvin_k_us;
    uint64_t mvin_v_us;
    uint64_t qk_us;
    uint64_t logp_mvout_us;
    uint64_t mvin_p_us;
    uint64_t pv_us;
    uint64_t pv_mvout_us;
    uint64_t qk_overlap_us;
    uint64_t pv_overlap_us;
};
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
        ggml_backend_npu_log8pv_attention_profile * profile);
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
        ggml_backend_npu_log8pv_attention_profile * profile_group);
extern "C" bool ggml_backend_npu_decode_w8a16_gemv_ex(
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
        int64_t dst_nb1);
extern "C" bool ggml_backend_npu_decode_overlay_ensure_active(const char * op_name);
extern "C" bool ggml_backend_npu_decode_w4a16_gemv_ex(
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
        int64_t dst_nb1);
struct ggml_npu_decode_awq_view {
        const char * weight_name;
        const void * q4_data;
        int64_t packed_k;
        int64_t out_channels;
        int64_t q4_nb1;
        const void * scale_data;
        int scale_type;
        int64_t scale_nb0;
        int64_t scale_nb1;
        const void * zero_data;
        int zero_type;
        int64_t zero_nb0;
        int64_t zero_nb1;
        const float * smooth_scale;
        size_t smooth_scale_len;
        int64_t k;
};
extern "C" bool ggml_backend_npu_decode_swiglu_ffn_w4a16_ex(
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
        int64_t dst_nb1);
extern "C" bool ggml_backend_npu_decode_attention_w4a16_ex(
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
        int64_t dst_nb1);
#endif

struct llama_text_activation_stats {
    uint64_t count = 0;
    int64_t in_channels = 0;
    float min = std::numeric_limits<float>::infinity();
    float max = -std::numeric_limits<float>::infinity();
    uint64_t seen_for_reservoir = 0;
    uint64_t reservoir_state = 0x9e3779b97f4a7c15ULL;
    size_t sample_limit = 0;
    std::vector<float> samples;
    std::vector<uint32_t> sample_channels;
    std::vector<float> per_channel_min;
    std::vector<float> per_channel_max;
    std::vector<float> per_channel_absmax;

    void ensure_channel_buffers(size_t channels) {
        if (channels == 0) {
            return;
        }
        if (in_channels == 0) {
            in_channels = static_cast<int64_t>(channels);
        }
        GGML_ASSERT(in_channels == static_cast<int64_t>(channels));
        if (per_channel_absmax.empty()) {
            per_channel_min.assign(channels, std::numeric_limits<float>::infinity());
            per_channel_max.assign(channels, -std::numeric_limits<float>::infinity());
            per_channel_absmax.assign(channels, 0.0f);
        }
    }

    void update(const float * data, size_t channels, size_t cols) {
        if (data == nullptr || channels == 0 || cols == 0) {
            return;
        }
        ensure_channel_buffers(channels);
        count += channels * cols;
        for (size_t col = 0; col < cols; ++col) {
            const float * col_ptr = data + col * channels;
            for (size_t ch = 0; ch < channels; ++ch) {
                const float v = col_ptr[ch];
                min = std::min(min, v);
                max = std::max(max, v);
                per_channel_min[ch] = std::min(per_channel_min[ch], v);
                per_channel_max[ch] = std::max(per_channel_max[ch], v);
                per_channel_absmax[ch] = std::max(per_channel_absmax[ch], std::fabs(v));

                if (sample_limit == 0) {
                    continue;
                }
                ++seen_for_reservoir;
                if (samples.size() < sample_limit) {
                    samples.push_back(v);
                    sample_channels.push_back(static_cast<uint32_t>(ch));
                    continue;
                }
                reservoir_state = reservoir_state * 6364136223846793005ULL + 1;
                const uint64_t slot = reservoir_state % seen_for_reservoir;
                if (slot < sample_limit) {
                    samples[(size_t) slot] = v;
                    sample_channels[(size_t) slot] = static_cast<uint32_t>(ch);
                }
            }
        }
    }
};

struct llama_text_activation_observer {
    std::string tensor_name;
};

static inline float llama_tensor_get_f32_4d(
        const struct ggml_tensor * tensor,
        int64_t i0,
        int64_t i1,
        int64_t i2,
        int64_t i3);

enum class llama_text_act_collect_mode {
    prefill,
    decode,
    both,
};

enum class llama_decode_awq_accum_mode {
    fp32,
    fp16,
};

struct llama_decode_awq_int24_sample {
    int64_t col = 0;
    int64_t row = 0;
    float ref = 0.0f;
    float int24 = 0.0f;
    float err = 0.0f;
};

struct llama_decode_awq_int24_thread_stats {
    static constexpr int hist_log2_min_exp = -32;
    static constexpr int hist_log2_max_exp = 16;
    static constexpr size_t hist_log2_bins = 2 + (hist_log2_max_exp - hist_log2_min_exp + 1) + 1;

    uint64_t count = 0;
    uint64_t scalar_count = 0;
    uint64_t clip_count = 0;
    uint64_t non_integer_zero_count = 0;
    double sum_abs_scaled_act = 0.0;
    double sum_sq_scaled_act = 0.0;
    double max_abs_scaled_act = 0.0;
    double max_abs_scaled_act_nonzero = 0.0;
    double sum_abs_scalar_err = 0.0;
    double sum_sq_scalar_err = 0.0;
    double max_abs_scalar_err = 0.0;
    double sum_abs_err = 0.0;
    double sum_sq_err = 0.0;
    double sum_ref_sq = 0.0;
    double max_abs_err = 0.0;
    double max_abs_ref = 0.0;
    double max_abs_int24 = 0.0;
    int32_t max_abs_q24 = 0;
    std::array<uint64_t, hist_log2_bins> scaled_act_abs_hist = {};
    std::array<uint64_t, hist_log2_bins> scalar_err_abs_hist = {};
    std::array<uint64_t, hist_log2_bins> output_err_abs_hist = {};
    std::array<uint64_t, hist_log2_bins> ref_abs_hist = {};
    std::vector<llama_decode_awq_int24_sample> samples;

    static void update_log2_abs_hist(std::array<uint64_t, hist_log2_bins> & hist, double value) {
        const double abs_value = std::fabs(value);
        if (abs_value == 0.0) {
            ++hist[0];
            return;
        }
        const int exp = (int) std::floor(std::log2(abs_value));
        if (exp < hist_log2_min_exp) {
            ++hist[1];
            return;
        }
        if (exp > hist_log2_max_exp) {
            ++hist[hist_log2_bins - 1];
            return;
        }
        ++hist[2 + (size_t) (exp - hist_log2_min_exp)];
    }

    void update_scalar(double scaled_act, int32_t q24, int frac_bits) {
        const double deq = std::ldexp((double) q24, -frac_bits);
        const double err = deq - scaled_act;
        const double abs_scaled_act = std::fabs(scaled_act);
        const double abs_err = std::fabs(err);
        ++scalar_count;
        sum_abs_scaled_act += abs_scaled_act;
        sum_sq_scaled_act += scaled_act * scaled_act;
        max_abs_scaled_act = std::max(max_abs_scaled_act, abs_scaled_act);
        if (abs_scaled_act > 0.0) {
            max_abs_scaled_act_nonzero = std::max(max_abs_scaled_act_nonzero, abs_scaled_act);
        }
        sum_abs_scalar_err += abs_err;
        sum_sq_scalar_err += err * err;
        max_abs_scalar_err = std::max(max_abs_scalar_err, abs_err);
        update_log2_abs_hist(scaled_act_abs_hist, scaled_act);
        update_log2_abs_hist(scalar_err_abs_hist, err);
    }

    void update(float ref, float int24) {
        const double err = (double) int24 - (double) ref;
        const double abs_err = std::fabs(err);
        ++count;
        sum_abs_err += abs_err;
        sum_sq_err += err * err;
        sum_ref_sq += (double) ref * (double) ref;
        max_abs_err = std::max(max_abs_err, abs_err);
        max_abs_ref = std::max(max_abs_ref, std::fabs((double) ref));
        max_abs_int24 = std::max(max_abs_int24, std::fabs((double) int24));
        update_log2_abs_hist(output_err_abs_hist, err);
        update_log2_abs_hist(ref_abs_hist, ref);
    }
};

static llama_text_act_collect_mode llama_text_act_collect_mode_from_env() {
    static const llama_text_act_collect_mode mode = []() {
        const char * value = std::getenv("AICAS_TEXT_ACT_COLLECT_MODE");
        if (value == nullptr || value[0] == '\0') {
            return llama_text_act_collect_mode::prefill;
        }
        if (std::strcmp(value, "decode") == 0) {
            return llama_text_act_collect_mode::decode;
        }
        if (std::strcmp(value, "both") == 0) {
            return llama_text_act_collect_mode::both;
        }
        return llama_text_act_collect_mode::prefill;
    }();
    return mode;
}

static llama_decode_awq_accum_mode llama_text_decode_awq_accum_mode_from_env() {
    static const llama_decode_awq_accum_mode mode = []() {
        const char * value = std::getenv("AICAS_TEXT_DECODE_AWQ_ACCUM");
        if (value != nullptr && std::strcmp(value, "fp32") == 0) {
            return llama_decode_awq_accum_mode::fp32;
        }
        return llama_decode_awq_accum_mode::fp16;
    }();
    return mode;
}

static bool llama_text_decode_awq_int24_enabled() {
    static const bool enabled = []() {
        const char * value = std::getenv("AICAS_TEXT_DECODE_AWQ_INT24");
        if (value == nullptr || value[0] == '\0') {
            return false;
        }
        return std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

static int llama_text_decode_awq_int24_frac_bits() {
    static const int bits = []() {
        const char * value = std::getenv("AICAS_TEXT_DECODE_AWQ_INT24_FRAC");
        if (value == nullptr || value[0] == '\0') {
            return 16;
        }
        const long parsed = std::strtol(value, nullptr, 10);
        if (parsed < 0 || parsed > 30) {
            return 16;
        }
        return (int) parsed;
    }();
    return bits;
}

static std::string llama_text_decode_awq_int24_diag_path() {
    const char * value = std::getenv("AICAS_TEXT_DECODE_AWQ_INT24_DIAG_FILE");
    return value != nullptr ? std::string(value) : std::string();
}

static size_t llama_text_decode_awq_int24_dump_limit() {
    static const size_t limit = []() {
        const char * value = std::getenv("AICAS_TEXT_DECODE_AWQ_INT24_DUMP_LIMIT");
        if (value == nullptr || value[0] == '\0') {
            return (size_t) 32;
        }
        char * end = nullptr;
        const unsigned long parsed = std::strtoul(value, &end, 10);
        if (end == value || *end != '\0') {
            return (size_t) 32;
        }
        return (size_t) parsed;
    }();
    return limit;
}

static std::string llama_text_lm_head_w8a16_int24_diag_path() {
    const char * value = std::getenv("AICAS_TEXT_LM_HEAD_W8A16_INT24_DIAG_FILE");
    return value != nullptr ? std::string(value) : std::string();
}

static bool llama_text_lm_head_w8a16_int24_enabled() {
    static const bool enabled = []() {
        const char * value = std::getenv("AICAS_TEXT_LM_HEAD_W8A16_INT24");
        if (value == nullptr || value[0] == '\0') {
            return false;
        }
        return std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

static int llama_text_lm_head_w8a16_int24_frac_bits() {
    static const int bits = []() {
        const char * value = std::getenv("AICAS_TEXT_LM_HEAD_W8A16_INT24_FRAC");
        if (value == nullptr || value[0] == '\0') {
            return 16;
        }
        const long parsed = std::strtol(value, nullptr, 10);
        if (parsed < 0 || parsed > 30) {
            return 16;
        }
        return (int) parsed;
    }();
    return bits;
}

static bool llama_text_flash_attn_accum_fp16() {
    static const bool enabled = []() {
        const char * value = std::getenv("AICAS_TEXT_FLASH_ATTN_ACCUM");
        return value != nullptr && std::strcmp(value, "fp16") == 0;
    }();
    return enabled;
}

static bool llama_text_lm_head_w16a16_dp128_enabled() {
    // W16A16 is no longer a supported decode path on the current RTL/runtime.
    return false;
}

static bool llama_text_lm_head_w8a16_dp128_enabled() {
    static const bool enabled = []() {
        const char * value = std::getenv("AICAS_TEXT_LM_HEAD_W8A16_DP128");
        if (value == nullptr || value[0] == '\0') {
            return false;
        }
        return std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

static bool llama_text_prefill_lm_head_default_enabled() {
    static const bool enabled = []() {
        const char * value = std::getenv("AICAS_TEXT_PREFILL_LM_HEAD_DEFAULT");
        if (value == nullptr || value[0] == '\0') {
            return true;
        }
        return std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

static bool llama_prefill_tail_decode_switch_enabled() {
#ifdef GGML_USE_NPU
    const char * value = std::getenv("AICAS_PREFILL_TAIL_DECODE_SWITCH");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
#else
    return false;
#endif
}

static bool llama_text_decode_awq_npu_gemv_only() {
    static const bool enabled = []() {
        const char * decode_npu = std::getenv("AICAS_TEXT_DECODE_AWQ_NPU");
        if (decode_npu == nullptr || decode_npu[0] == '\0' || std::strcmp(decode_npu, "0") == 0) {
            return false;
        }
        const char * value = std::getenv("AICAS_TEXT_DECODE_AWQ_NPU_GEMV_ONLY");
        if (value == nullptr || value[0] == '\0') {
            return false;
        }
        const bool gemv_only = std::strcmp(value, "0") != 0;
        if (gemv_only) {
            LLAMA_LOG_WARN(
                    "%s: WARNING: AICAS_TEXT_DECODE_AWQ_NPU_GEMV_ONLY is enabled; "
                    "attention NPU, fused SwiGLU FFN, and lm_head NPU decode paths will be disabled\n",
                    __func__);
        }
        return gemv_only;
    }();
    return enabled;
}

static bool llama_text_lm_head_w8a16_npu_enabled() {
    static const bool enabled = []() {
        if (llama_text_decode_awq_npu_gemv_only()) {
            return false;
        }
        const char * value = std::getenv("AICAS_TEXT_LM_HEAD_W8A16_NPU");
        if (value == nullptr || value[0] == '\0') {
            const char * decode_npu = std::getenv("AICAS_TEXT_DECODE_AWQ_NPU");
            return decode_npu != nullptr && decode_npu[0] != '\0' && std::strcmp(decode_npu, "0") != 0;
        }
        return std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

static bool llama_text_lm_head_w8a16_npu_compare_enabled() {
    static const bool enabled = []() {
        const char * value = std::getenv("AICAS_TEXT_LM_HEAD_W8A16_NPU_COMPARE");
        if (value == nullptr || value[0] == '\0') {
            return false;
        }
        return std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

static int llama_text_lm_head_w8a16_npu_compare_limit() {
    static const int limit = []() {
        const char * value = std::getenv("AICAS_TEXT_LM_HEAD_W8A16_NPU_COMPARE_LIMIT");
        if (value == nullptr || value[0] == '\0') {
            return 8;
        }
        char * end = nullptr;
        const long parsed = std::strtol(value, &end, 0);
        return end != value && parsed > 0 ? (int) parsed : 8;
    }();
    return limit;
}

static int64_t llama_text_lm_head_w8a16_group() {
    static const int64_t group = []() {
        const char * value = std::getenv("AICAS_TEXT_LM_HEAD_W8A16_GROUP");
        if (value == nullptr || value[0] == '\0') {
            return (int64_t) 128;
        }
        const long parsed = std::strtol(value, nullptr, 10);
        if (parsed <= 0 || parsed > 128) {
            return (int64_t) 128;
        }
        return (int64_t) parsed;
    }();
    return group;
}

static bool llama_text_token_embd_w8a16_enabled() {
    static const bool enabled = []() {
        const char * value = std::getenv("AICAS_TEXT_TOKEN_EMBD_W8A16");
        if (value == nullptr || value[0] == '\0') {
            return false;
        }
        return std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

static int64_t llama_text_token_embd_w8a16_group() {
    static const int64_t group = []() {
        const char * value = std::getenv("AICAS_TEXT_TOKEN_EMBD_W8A16_GROUP");
        if (value == nullptr || value[0] == '\0') {
            return (int64_t) 64;
        }
        const long parsed = std::strtol(value, nullptr, 10);
        if (parsed <= 0 || parsed > 128) {
            return (int64_t) 64;
        }
        return (int64_t) parsed;
    }();
    return group;
}

static bool llama_text_lm_head_w16a16_accum_fp16() {
    static const bool enabled = []() {
        const char * value = std::getenv("AICAS_TEXT_LM_HEAD_W16A16_ACCUM");
        return value != nullptr && std::strcmp(value, "fp16") == 0;
    }();
    return enabled;
}

static bool llama_text_sq_enable_decode_gemv() {
    static const bool enabled = []() {
        const char * value = std::getenv("AICAS_TEXT_SQ_ENABLE_DECODE_GEMV");
        if (value == nullptr || value[0] == '\0') {
            return false;
        }
        return std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

static bool llama_text_decode_awq_enabled() {
    static const bool enabled = []() {
        const char * value = std::getenv("AICAS_TEXT_DECODE_AWQ");
        if (value == nullptr || value[0] == '\0') {
            return false;
        }
        return std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

static bool llama_text_decode_awq_npu_enabled() {
    static const bool enabled = []() {
        const char * value = std::getenv("AICAS_TEXT_DECODE_AWQ_NPU");
        if (value == nullptr || value[0] == '\0') {
            return false;
        }
        return std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

static bool llama_text_decode_awq_fused_ffn_npu_enabled() {
    static const bool enabled = []() {
        if (llama_text_decode_awq_npu_gemv_only()) {
            return false;
        }
        const char * value = std::getenv("AICAS_TEXT_DECODE_AWQ_FUSED_FFN_NPU");
        if (value == nullptr || value[0] == '\0') {
            return llama_text_decode_awq_npu_enabled();
        }
        return std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

static bool llama_text_decode_awq_fused_ffn_npu_log_enabled() {
    static const bool enabled = []() {
        const char * value = std::getenv("AICAS_TEXT_DECODE_AWQ_FUSED_FFN_NPU_LOG");
        return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

static bool llama_text_decode_awq_npu_compare_enabled() {
    static const bool enabled = []() {
        const char * value = std::getenv("AICAS_TEXT_DECODE_AWQ_NPU_COMPARE");
        return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

static int llama_text_decode_awq_npu_compare_limit() {
    static const int limit = []() {
        const char * value = std::getenv("AICAS_TEXT_DECODE_AWQ_NPU_COMPARE_LIMIT");
        if (value == nullptr || value[0] == '\0') {
            return 16;
        }
        char * end = nullptr;
        const long parsed = std::strtol(value, &end, 0);
        return end != value && parsed > 0 ? (int) parsed : 16;
    }();
    return limit;
}

struct llama_npu_shape_key {
    std::string weight_name;
    int64_t m = 0;
    int64_t n = 0;
    int64_t k = 0;
    bool bias_present = false;

    std::string str() const {
        return weight_name + "|m=" + std::to_string(m) +
            "|n=" + std::to_string(n) +
            "|k=" + std::to_string(k) +
            "|bias=" + (bias_present ? "1" : "0");
    }
};

struct llama_npu_shape_table_state {
    std::once_flag load_once;
    std::mutex mutex;
    std::string table_path;
    std::string record_path;
    std::set<std::string> enabled_keys;
    std::set<std::string> recorded_keys;
    std::vector<std::string> recorded_entries;
};

static llama_npu_shape_table_state & llama_npu_shape_state() {
    static llama_npu_shape_table_state state;
    return state;
}

static bool llama_npu_json_get_string(const std::string & object, const char * key, std::string * out) {
    const std::regex re(std::string("\"") + key + "\"\\s*:\\s*\"([^\"]*)\"");
    std::smatch match;
    if (!std::regex_search(object, match, re)) {
        return false;
    }
    *out = match[1].str();
    return true;
}

static int64_t llama_npu_json_get_i64(const std::string & object, const char * key, int64_t fallback) {
    const std::regex re(std::string("\"") + key + "\"\\s*:\\s*(-?[0-9]+)");
    std::smatch match;
    if (!std::regex_search(object, match, re)) {
        return fallback;
    }
    return std::strtoll(match[1].str().c_str(), nullptr, 10);
}

static bool llama_npu_json_get_bool(const std::string & object, const char * key, bool fallback) {
    const std::regex re(std::string("\"") + key + "\"\\s*:\\s*(true|false)");
    std::smatch match;
    if (!std::regex_search(object, match, re)) {
        return fallback;
    }
    return match[1].str() == "true";
}

static std::string llama_npu_json_escape(const std::string & s) {
    std::ostringstream out;
    for (const char c : s) {
        switch (c) {
            case '\\': out << "\\\\"; break;
            case '"':  out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:   out << c; break;
        }
    }
    return out.str();
}

static void llama_npu_shape_load_once() {
    llama_npu_shape_table_state & state = llama_npu_shape_state();
    state.table_path = std::getenv("GGML_NPU_SHAPE_TABLE_JSON") ? std::getenv("GGML_NPU_SHAPE_TABLE_JSON") : "";
    state.record_path = std::getenv("GGML_NPU_SHAPE_RECORD_JSON") ? std::getenv("GGML_NPU_SHAPE_RECORD_JSON") : "";

    if (!state.table_path.empty()) {
        std::ifstream in(state.table_path);
        if (in) {
            try {
                std::ostringstream buffer;
                buffer << in.rdbuf();
                const std::string text = buffer.str();
                const std::regex object_re("\\{[^{}]*\\}");
                for (auto it = std::sregex_iterator(text.begin(), text.end(), object_re);
                        it != std::sregex_iterator(); ++it) {
                    const std::string object = it->str();
                    if (!llama_npu_json_get_bool(object, "enabled", true)) {
                        continue;
                    }
                    llama_npu_shape_key key;
                    if (!llama_npu_json_get_string(object, "weight_name", &key.weight_name)) {
                        continue;
                    }
                    key.m = llama_npu_json_get_i64(object, "m", 0);
                    key.n = llama_npu_json_get_i64(object, "n", 0);
                    key.k = llama_npu_json_get_i64(object, "k", 0);
                    key.bias_present = llama_npu_json_get_bool(object, "bias_present", false);
                    if (!key.weight_name.empty() && key.m > 0 && key.n > 0 && key.k > 0) {
                        state.enabled_keys.insert(key.str());
                    }
                }
            } catch (const std::exception & e) {
                LLAMA_LOG_WARN("%s: failed to parse GGML_NPU_SHAPE_TABLE_JSON=%s: %s\n",
                        __func__, state.table_path.c_str(), e.what());
            }
        } else {
            LLAMA_LOG_WARN("%s: failed to open GGML_NPU_SHAPE_TABLE_JSON=%s\n",
                    __func__, state.table_path.c_str());
        }
    }
}

static bool llama_npu_shape_table_hit(const llama_npu_shape_key & key) {
    llama_npu_shape_table_state & state = llama_npu_shape_state();
    std::call_once(state.load_once, llama_npu_shape_load_once);
    if (state.table_path.empty()) {
        return false;
    }
    std::lock_guard<std::mutex> lock(state.mutex);
    return state.enabled_keys.find(key.str()) != state.enabled_keys.end();
}

static bool llama_npu_text_prefill_dynamic_enabled() {
    const char * value = std::getenv("GGML_NPU_TEXT_PREFILL_DYNAMIC");
    return value == nullptr || value[0] == '\0' || std::strcmp(value, "0") != 0;
}

static void llama_npu_record_shape(
        const llama_npu_shape_key & key,
        const char * source,
        const llama_aicas_text_sq_tensor & cfg) {
    llama_npu_shape_table_state & state = llama_npu_shape_state();
    std::call_once(state.load_once, llama_npu_shape_load_once);
    if (state.record_path.empty()) {
        return;
    }

    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.recorded_keys.insert(key.str()).second) {
        return;
    }

    std::ostringstream entry;
    entry << "    {\n"
          << "      \"weight_name\": \"" << llama_npu_json_escape(key.weight_name) << "\",\n"
          << "      \"m\": " << key.m << ",\n"
          << "      \"n\": " << key.n << ",\n"
          << "      \"k\": " << key.k << ",\n"
          << "      \"bias_present\": " << (key.bias_present ? "true" : "false") << ",\n"
          << "      \"src_type\": \"I8\",\n"
          << "      \"dst_type\": \"F32\",\n"
          << "      \"quant_schema\": \"aicas_text_w8a8\",\n"
          << "      \"act_quant_mode\": \"" << llama_npu_json_escape(cfg.act_quant_mode) << "\",\n"
          << "      \"weight_scale_mode\": \"" << llama_npu_json_escape(cfg.weight_scale_mode) << "\",\n"
          << "      \"source\": \"" << llama_npu_json_escape(source ? source : "text_prefill_w8a8") << "\",\n"
          << "      \"enabled\": true,\n"
          << "      \"tm\": 0,\n"
          << "      \"tn\": 0,\n"
          << "      \"tk\": 0\n"
          << "    }";
    state.recorded_entries.push_back(entry.str());

    std::ofstream out(state.record_path);
    if (!out) {
        LLAMA_LOG_WARN("%s: failed to write GGML_NPU_SHAPE_RECORD_JSON=%s\n",
                __func__, state.record_path.c_str());
        return;
    }
    out << "{\n"
        << "  \"version\": 1,\n"
        << "  \"note\": \"Fill tm/tn/tk before using this file as GGML_NPU_SHAPE_TABLE_JSON.\",\n"
        << "  \"entries\": [\n";
    for (size_t i = 0; i < state.recorded_entries.size(); ++i) {
        out << state.recorded_entries[i];
        if (i + 1 < state.recorded_entries.size()) {
            out << ",";
        }
        out << "\n";
    }
    out << "  ]\n"
        << "}\n";
}

static bool llama_register_text_w8a8_for_npu(
        const llama_npu_shape_key & key,
        const llama_aicas_text_sq_tensor & cfg) {
#ifdef GGML_USE_NPU
    if (cfg.weight_scale.empty() || cfg.sum_w.empty()) {
        return false;
    }
    if (cfg.act_scale <= 0.0f || cfg.act_zero_point < 0 || cfg.act_zero_point > 255) {
        return false;
    }
    return ggml_backend_npu_w8a8_register(
            key.weight_name.c_str(),
            cfg.act_scale,
            0,
            cfg.act_zero_point,
            cfg.weight_scale.data(),
            cfg.weight_scale.size(),
            cfg.sum_w.data(),
            cfg.sum_w.size(),
            cfg.smooth_scale.empty() ? nullptr : cfg.smooth_scale.data(),
            cfg.smooth_scale.size());
#else
    GGML_UNUSED(key);
    GGML_UNUSED(cfg);
    return false;
#endif
}

static bool llama_text_prefill_attn_bfp16m_enabled() {
    static const bool enabled = []() {
        const char * value = std::getenv("AICAS_TEXT_PREFILL_ATTN_BFP16M");
        if (value == nullptr || value[0] == '\0') {
            return false;
        }
        return std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

static bool llama_text_prefill_attn_bfp8m_enabled() {
    static const bool enabled = []() {
        const char * value = std::getenv("AICAS_TEXT_PREFILL_ATTN_BFP8M");
        if (value == nullptr || value[0] == '\0') {
            return false;
        }
        return std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

static bool llama_text_prefill_attn_log8pv_enabled() {
    static const bool enabled = []() {
        const char * value = std::getenv("AICAS_TEXT_PREFILL_ATTN_LOG8PV");
        if (value == nullptr || value[0] == '\0') {
            return false;
        }
        return std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

static bool llama_text_prefill_log8pv_npu_enabled() {
    static const bool enabled = []() {
        const char * value = std::getenv("AICAS_TEXT_PREFILL_LOG8PV_NPU");
        if (value == nullptr || value[0] == '\0') {
            return false;
        }
        return std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

static bool llama_text_log8pv_cpu_npu_pipeline_enabled() {
    static const bool enabled = []() {
        const char * value = std::getenv("AICAS_TEXT_LOG8PV_CPU_NPU_PIPELINE");
        if (value == nullptr || value[0] == '\0') {
            return true;
        }
        return std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

static int64_t llama_text_log8pv_cpu_npu_pipeline_workers() {
    static const int64_t workers = []() {
        constexpr int64_t default_workers = 4;
        const char * value = std::getenv("AICAS_TEXT_LOG8PV_CPU_PREP_WORKERS");
        if (value == nullptr || value[0] == '\0') {
            return default_workers;
        }
        char * end = nullptr;
        const long parsed = std::strtol(value, &end, 10);
        if (end != value && *end == '\0' && parsed > 0 && parsed <= 4) {
            return (int64_t) parsed;
        }
        return default_workers;
    }();
    return workers;
}

static bool llama_text_prefill_log8pv_direct_kv_enabled() {
    static const bool enabled = []() {
        const char * value = std::getenv("AICAS_TEXT_PREFILL_LOG8PV_DIRECT_KV");
        if (value == nullptr || value[0] == '\0') {
            return false;
        }
        return std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

static int64_t llama_text_prefill_log8pv_tile() {
    static const int64_t tile = []() {
        constexpr int64_t default_tile = 64;
        const char * value = std::getenv("AICAS_TEXT_PREFILL_LOG8PV_TILE");
        if (value == nullptr || value[0] == '\0') {
            return default_tile;
        }
        char * end = nullptr;
        const long parsed = std::strtol(value, &end, 10);
        if (end != value && *end == '\0' && parsed > 0 && parsed <= std::numeric_limits<int16_t>::max()) {
            return (int64_t) parsed;
        }
        return default_tile;
    }();
    return tile;
}

static float llama_text_prefill_log8pv_p_log2_range() {
    static const float range = []() {
        constexpr float default_range = 16.0f;
        const char * value = std::getenv("AICAS_TEXT_PREFILL_LOG8PV_P_LOG2_RANGE");
        if (value == nullptr || value[0] == '\0') {
            return default_range;
        }
        char * end = nullptr;
        const float parsed = std::strtof(value, &end);
        if (end != value && *end == '\0' && std::isfinite(parsed) && parsed > 0.0f) {
            return parsed;
        }
        return default_range;
    }();
    return range;
}

static bool llama_text_prefill_log8pv_p_direct_u16() {
    static const bool enabled = []() {
        const char * value = std::getenv("AICAS_TEXT_PREFILL_LOG8PV_P_MODE");
        return value != nullptr &&
            (std::strcmp(value, "direct_u16") == 0 || std::strcmp(value, "direct-u16") == 0);
    }();
    return enabled;
}

static int llama_text_prefill_log8pv_p_bits() {
    static const int bits = []() {
        constexpr int default_bits = 15;
        const char * value = std::getenv("AICAS_TEXT_PREFILL_LOG8PV_P_BITS");
        if (value == nullptr || value[0] == '\0') {
            return default_bits;
        }
        char * end = nullptr;
        const long parsed = std::strtol(value, &end, 10);
        if (end != value && *end == '\0' && (parsed == 14 || parsed == 15 || parsed == 16)) {
            return (int) parsed;
        }
        return default_bits;
    }();
    return bits;
}

static bool llama_text_prefill_log8pv_bit_accurate_enabled() {
    static const bool enabled = []() {
        const char * value = std::getenv("AICAS_TEXT_PREFILL_LOG8PV_BIT_ACCURATE");
        if (value == nullptr || value[0] == '\0') {
            return false;
        }
        return std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

static int llama_text_prefill_log8pv_alpha_bits() {
    static const int bits = []() {
        constexpr int default_bits = 20;
        const char * value = std::getenv("AICAS_TEXT_PREFILL_LOG8PV_ALPHA_BITS");
        if (value == nullptr || value[0] == '\0') {
            return default_bits;
        }
        char * end = nullptr;
        const long parsed = std::strtol(value, &end, 10);
        if (end != value && *end == '\0' && (parsed == 20 || parsed == 24)) {
            return (int) parsed;
        }
        return default_bits;
    }();
    return bits;
}

static int llama_text_prefill_log8pv_v_bits() {
    static const int bits = []() {
        constexpr int default_bits = 8;
        const char * value = std::getenv("AICAS_TEXT_PREFILL_LOG8PV_V_BITS");
        if (value == nullptr || value[0] == '\0') {
            return default_bits;
        }
        char * end = nullptr;
        const long parsed = std::strtol(value, &end, 10);
        if (end != value && *end == '\0' && (parsed == 8 || parsed == 10)) {
            return (int) parsed;
        }
        return default_bits;
    }();
    return bits;
}

static bool llama_text_prefill_log8pv_v_centering_enabled() {
    static const bool enabled = []() {
        const char * value = std::getenv("AICAS_TEXT_PREFILL_LOG8PV_V_CENTERING");
        if (value == nullptr || value[0] == '\0') {
            return true;
        }
        return std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

static int64_t llama_text_prefill_bfp16m_k_block() {
    static const int64_t k_block = []() {
        constexpr int64_t default_k_block = 64;
        const char * value = std::getenv("AICAS_TEXT_PREFILL_BFP16M_K_BLOCK");
        if (value == nullptr || value[0] == '\0') {
            return default_k_block;
        }
        char * end = nullptr;
        const long parsed = std::strtol(value, &end, 10);
        if (end != value && *end == '\0' && parsed > 0 && parsed <= std::numeric_limits<int16_t>::max()) {
            return (int64_t) parsed;
        }
        return default_k_block;
    }();
    return k_block;
}

enum class llama_bfp16m_exp_mode {
    kblock,
    static_tensor,
};

static llama_bfp16m_exp_mode llama_text_prefill_bfp16m_exp_mode() {
    static const llama_bfp16m_exp_mode mode = []() {
        const char * value = std::getenv("AICAS_TEXT_PREFILL_BFP16M_EXP_MODE");
        if (value == nullptr || value[0] == '\0' || std::strcmp(value, "kblock") == 0) {
            return llama_bfp16m_exp_mode::kblock;
        }
        if (std::strcmp(value, "static_tensor") == 0 || std::strcmp(value, "static-tensor") == 0 ||
                std::strcmp(value, "static_per_tensor") == 0 || std::strcmp(value, "static-per-tensor") == 0 ||
                std::strcmp(value, "per_tensor") == 0 || std::strcmp(value, "per-tensor") == 0) {
            return llama_bfp16m_exp_mode::static_tensor;
        }
        return llama_bfp16m_exp_mode::kblock;
    }();
    return mode;
}

static int llama_text_prefill_bfp16m_static_exp(const char * env_name, int default_exp) {
    const char * value = std::getenv(env_name);
    if (value == nullptr || value[0] == '\0') {
        return default_exp;
    }
    char * end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end != value && *end == '\0' && parsed >= -128 && parsed <= 127) {
        return (int) parsed;
    }
    return default_exp;
}

static int64_t llama_text_prefill_bfp8m_k_block() {
    static const int64_t k_block = []() {
        const int64_t default_k_block = llama_text_prefill_bfp16m_k_block();
        const char * value = std::getenv("AICAS_TEXT_PREFILL_BFP8M_K_BLOCK");
        if (value == nullptr || value[0] == '\0') {
            return default_k_block;
        }
        char * end = nullptr;
        const long parsed = std::strtol(value, &end, 10);
        if (end != value && *end == '\0' && parsed > 0 && parsed <= std::numeric_limits<int16_t>::max()) {
            return (int64_t) parsed;
        }
        return default_k_block;
    }();
    return k_block;
}

enum class llama_bfp8m_scale_mode {
    block,
    tensor,
    tile,
};

static llama_bfp8m_scale_mode llama_text_prefill_bfp8m_scale_mode() {
    static const llama_bfp8m_scale_mode mode = []() {
        const char * value = std::getenv("AICAS_TEXT_PREFILL_BFP8M_SCALE_MODE");
        if (value == nullptr || value[0] == '\0' || std::strcmp(value, "block") == 0) {
            return llama_bfp8m_scale_mode::block;
        }
        if (std::strcmp(value, "tensor") == 0 || std::strcmp(value, "per_tensor") == 0 ||
                std::strcmp(value, "per-tensor") == 0 || std::strcmp(value, "static_tensor") == 0 ||
                std::strcmp(value, "static-per-tensor") == 0) {
            return llama_bfp8m_scale_mode::tensor;
        }
        if (std::strcmp(value, "tile") == 0 || std::strcmp(value, "tile32") == 0 ||
                std::strcmp(value, "systolic_tile") == 0 || std::strcmp(value, "systolic-tile") == 0) {
            return llama_bfp8m_scale_mode::tile;
        }
        return llama_bfp8m_scale_mode::block;
    }();
    return mode;
}

static int64_t llama_text_prefill_bfp8m_tile() {
    static const int64_t tile = []() {
        constexpr int64_t default_tile = 32;
        const char * value = std::getenv("AICAS_TEXT_PREFILL_BFP8M_TILE");
        if (value == nullptr || value[0] == '\0') {
            value = std::getenv("AICAS_BFP8M_TILE");
        }
        if (value == nullptr || value[0] == '\0') {
            return default_tile;
        }
        char * end = nullptr;
        const long parsed = std::strtol(value, &end, 10);
        if (end != value && *end == '\0' && parsed > 0 && parsed <= std::numeric_limits<int16_t>::max()) {
            return (int64_t) parsed;
        }
        return default_tile;
    }();
    return tile;
}

struct llama_text_bfp16m_userdata {
    int64_t k_block = 64;
    llama_bfp16m_exp_mode exp_mode = llama_bfp16m_exp_mode::kblock;
    bool pv_matmul = false;
    int static_exp_a = 0;
    int static_exp_b = 0;
};

static llama_text_bfp16m_userdata llama_make_text_bfp16m_userdata(bool pv_matmul) {
    llama_text_bfp16m_userdata cfg;
    cfg.k_block = llama_text_prefill_bfp16m_k_block();
    cfg.exp_mode = llama_text_prefill_bfp16m_exp_mode();
    cfg.pv_matmul = pv_matmul;
    if (cfg.exp_mode == llama_bfp16m_exp_mode::static_tensor) {
        const int q_exp = llama_text_prefill_bfp16m_static_exp("AICAS_TEXT_PREFILL_BFP16M_Q_EXP", -11);
        const int k_exp = llama_text_prefill_bfp16m_static_exp("AICAS_TEXT_PREFILL_BFP16M_K_EXP", -11);
        const int v_exp = llama_text_prefill_bfp16m_static_exp("AICAS_TEXT_PREFILL_BFP16M_V_EXP", -12);
        const int p_exp = llama_text_prefill_bfp16m_static_exp("AICAS_TEXT_PREFILL_BFP16M_P_EXP", -15);
        cfg.static_exp_a = pv_matmul ? v_exp : k_exp;
        cfg.static_exp_b = pv_matmul ? p_exp : q_exp;
    }
    return cfg;
}

static std::string llama_text_attn_dist_stats_path() {
    static const std::string path = []() {
        const char * value = std::getenv("AICAS_TEXT_PREFILL_ATTN_DIST_FILE");
        return value == nullptr ? std::string() : std::string(value);
    }();
    return path;
}

static int64_t llama_text_attn_dist_k_block() {
    static const int64_t k_block = []() {
        constexpr int64_t default_k_block = 64;
        const char * value = std::getenv("AICAS_ATTN_DIST_K_BLOCK");
        if (value == nullptr || value[0] == '\0') {
            return default_k_block;
        }
        char * end = nullptr;
        const long parsed = std::strtol(value, &end, 10);
        if (end != value && *end == '\0' && parsed > 0 && parsed <= std::numeric_limits<int16_t>::max()) {
            return (int64_t) parsed;
        }
        return default_k_block;
    }();
    return k_block;
}

static void llama_write_float_array_json(std::ostream & out, const std::vector<float> & values) {
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            out << ',';
        }
        out << values[i];
    }
    out << ']';
}

static void llama_record_attn_tensor_dist(
        const char * scope,
        const char * op,
        const char * tensor_role,
        const struct ggml_tensor * tensor,
        bool a_layout) {
    const std::string path = llama_text_attn_dist_stats_path();
    if (path.empty() || tensor == nullptr) {
        return;
    }

    float min_v = std::numeric_limits<float>::infinity();
    float max_v = -std::numeric_limits<float>::infinity();
    float absmax = 0.0f;
    double sum = 0.0;
    double sum_abs = 0.0;
    double sum_sq = 0.0;
    uint64_t count = 0;
    std::vector<float> abs_samples;
    std::vector<float> top_abs;
    const uint64_t total = (uint64_t) tensor->ne[0] * (uint64_t) tensor->ne[1] *
        (uint64_t) tensor->ne[2] * (uint64_t) tensor->ne[3];
    const uint64_t sample_stride = std::max<uint64_t>(1, total / 8192);

    for (int64_t i3 = 0; i3 < tensor->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < tensor->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < tensor->ne[1]; ++i1) {
                for (int64_t i0 = 0; i0 < tensor->ne[0]; ++i0) {
                    const float v = llama_tensor_get_f32_4d(tensor, i0, i1, i2, i3);
                    const float av = std::fabs(v);
                    min_v = std::min(min_v, v);
                    max_v = std::max(max_v, v);
                    absmax = std::max(absmax, av);
                    sum += v;
                    sum_abs += av;
                    sum_sq += (double) v * (double) v;
                    if (count % sample_stride == 0) {
                        abs_samples.push_back(av);
                    }
                    if (top_abs.size() < 16) {
                        top_abs.push_back(av);
                        if (top_abs.size() == 16) {
                            std::make_heap(top_abs.begin(), top_abs.end(), std::greater<float>());
                        }
                    } else if (av > top_abs.front()) {
                        std::pop_heap(top_abs.begin(), top_abs.end(), std::greater<float>());
                        top_abs.back() = av;
                        std::push_heap(top_abs.begin(), top_abs.end(), std::greater<float>());
                    }
                    ++count;
                }
            }
        }
    }
    if (top_abs.size() == 16) {
        std::sort_heap(top_abs.begin(), top_abs.end(), std::greater<float>());
    }
    std::sort(top_abs.begin(), top_abs.end(), std::greater<float>());

    uint64_t tensor_bfp8_zero = 0;
    uint64_t tensor_bfp8_sat = 0;
    const double tensor_scale = absmax > 0.0f ? (double) absmax / 127.0 : 0.0;
    if (tensor_scale > 0.0) {
        for (int64_t i3 = 0; i3 < tensor->ne[3]; ++i3) {
            for (int64_t i2 = 0; i2 < tensor->ne[2]; ++i2) {
                for (int64_t i1 = 0; i1 < tensor->ne[1]; ++i1) {
                    for (int64_t i0 = 0; i0 < tensor->ne[0]; ++i0) {
                        const double q_abs = std::fabs(std::lrint((double) llama_tensor_get_f32_4d(tensor, i0, i1, i2, i3) / tensor_scale));
                        tensor_bfp8_zero += q_abs == 0.0;
                        tensor_bfp8_sat += q_abs >= 127.0;
                    }
                }
            }
        }
    }

    const int64_t k_total = tensor->ne[0];
    const int64_t block = std::max<int64_t>(1, llama_text_attn_dist_k_block());
    std::vector<float> block_absmax;
    uint64_t block_bfp8_zero = 0;
    uint64_t block_bfp8_sat = 0;
    for (int64_t i3 = 0; i3 < tensor->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < tensor->ne[2]; ++i2) {
            for (int64_t row_col = 0; row_col < tensor->ne[1]; ++row_col) {
                for (int64_t k0 = 0; k0 < k_total; k0 += block) {
                    const int64_t k1 = std::min(k_total, k0 + block);
                    float bmax = 0.0f;
                    for (int64_t k = k0; k < k1; ++k) {
                        const float v = a_layout
                            ? llama_tensor_get_f32_4d(tensor, k, row_col, i2, i3)
                            : llama_tensor_get_f32_4d(tensor, k, row_col, i2, i3);
                        bmax = std::max(bmax, std::fabs(v));
                    }
                    block_absmax.push_back(bmax);
                    const double block_scale = bmax > 0.0f ? (double) bmax / 127.0 : 0.0;
                    if (block_scale == 0.0) {
                        continue;
                    }
                    for (int64_t k = k0; k < k1; ++k) {
                        const double q_abs = std::fabs(std::lrint((double) llama_tensor_get_f32_4d(tensor, k, row_col, i2, i3) / block_scale));
                        block_bfp8_zero += q_abs == 0.0;
                        block_bfp8_sat += q_abs >= 127.0;
                    }
                }
            }
        }
    }

    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    std::ofstream out(path, std::ios::app | std::ios::binary);
    if (!out.is_open()) {
        LLAMA_LOG_ERROR("%s: failed to open text attention dist file: %s\n", __func__, path.c_str());
        return;
    }
    out << std::setprecision(9)
        << "{\"schema\":\"aicas.attn_tensor_dist.v1\""
        << ",\"scope\":\"" << scope << "\""
        << ",\"op\":\"" << op << "\""
        << ",\"tensor\":\"" << tensor_role << "\""
        << ",\"shape\":[" << tensor->ne[0] << ',' << tensor->ne[1] << ',' << tensor->ne[2] << ',' << tensor->ne[3] << ']'
        << ",\"count\":" << count
        << ",\"min\":" << min_v
        << ",\"max\":" << max_v
        << ",\"absmax\":" << absmax
        << ",\"mean\":" << (count > 0 ? sum / (double) count : 0.0)
        << ",\"mean_abs\":" << (count > 0 ? sum_abs / (double) count : 0.0)
        << ",\"rms\":" << (count > 0 ? std::sqrt(sum_sq / (double) count) : 0.0)
        << ",\"bfp8_tensor_scale\":" << tensor_scale
        << ",\"bfp8_tensor_zero_count\":" << tensor_bfp8_zero
        << ",\"bfp8_tensor_sat_count\":" << tensor_bfp8_sat
        << ",\"bfp8_block_k\":" << block
        << ",\"bfp8_block_zero_count\":" << block_bfp8_zero
        << ",\"bfp8_block_sat_count\":" << block_bfp8_sat
        << ",\"abs_samples\":";
    llama_write_float_array_json(out, abs_samples);
    out << ",\"top_abs\":";
    llama_write_float_array_json(out, top_abs);
    out << ",\"block_absmax\":";
    llama_write_float_array_json(out, block_absmax);
    out << "}\n";
}

struct llama_text_bfp8m_userdata {
    int64_t k_block = 64;
    int64_t tile = 32;
    llama_bfp8m_scale_mode scale_mode = llama_bfp8m_scale_mode::block;
};

struct llama_text_log8pv_attn_userdata {
    const ggml_tensor * mask = nullptr;
    const struct llama_text_log8pv_static_layer * calib = nullptr;
    struct llama_text_log8pv_collect_layer * collect = nullptr;
    float kq_scale = 1.0f;
    int layer = 0;
    int64_t tile = 64;
    float p_log2_range = 16.0f;
    int alpha_bits = 20;
    int v_bits = 8;
    int p_bits = 15;
    bool p_direct_u16 = false;
    bool bit_accurate = false;
    int qk_scale_mode = 0; // 0=static, 1=dynamic per head, 2=dynamic per tile
    int64_t qk_scale_tile = 32;
    int v_scale_mode = 0; // 0=static, 1=dynamic per head/channel
    bool v_centering = true;
};

struct llama_text_log8pv_static_layer {
    int layer = 0;
    int64_t n_head = 0;
    int64_t n_kv_head = 0;
    int64_t d_head = 0;
    std::vector<float> q_threshold_head;
    std::vector<float> k_threshold_head;
    std::vector<float> qk_gain_head;
    std::vector<float> v_mean;
    std::vector<float> v_threshold;

    bool valid_for(int64_t heads, int64_t kv_heads, int64_t dim) const {
        return n_head == heads &&
            n_kv_head == kv_heads &&
            d_head == dim &&
            q_threshold_head.size() == (size_t) heads &&
            k_threshold_head.size() == (size_t) kv_heads &&
            (qk_gain_head.empty() || qk_gain_head.size() == (size_t) heads) &&
            v_mean.size() == (size_t) (kv_heads * dim) &&
            v_threshold.size() == (size_t) (kv_heads * dim);
    }
};

struct llama_text_log8pv_collect_layer {
    int layer = 0;
    int64_t n_head = 0;
    int64_t n_kv_head = 0;
    int64_t d_head = 0;
    uint64_t calls = 0;
    std::vector<float> q_absmax_head;
    std::vector<float> k_absmax_head;
    std::vector<double> v_sum;
    std::vector<uint64_t> v_count;
    std::vector<float> v_min;
    std::vector<float> v_max;

    void ensure(int in_layer, int64_t heads, int64_t kv_heads, int64_t dim) {
        if (n_head == 0 && n_kv_head == 0 && d_head == 0) {
            layer = in_layer;
            n_head = heads;
            n_kv_head = kv_heads;
            d_head = dim;
            q_absmax_head.assign((size_t) heads, 0.0f);
            k_absmax_head.assign((size_t) kv_heads, 0.0f);
            const size_t n = (size_t) (kv_heads * dim);
            v_sum.assign(n, 0.0);
            v_count.assign(n, 0);
            v_min.assign(n, std::numeric_limits<float>::infinity());
            v_max.assign(n, -std::numeric_limits<float>::infinity());
            return;
        }
        GGML_ASSERT(layer == in_layer);
        GGML_ASSERT(n_head == heads);
        GGML_ASSERT(n_kv_head == kv_heads);
        GGML_ASSERT(d_head == dim);
    }

    json to_json() const {
        return {
            {"layer", layer},
            {"n_head", n_head},
            {"n_kv_head", n_kv_head},
            {"d_head", d_head},
            {"calls", calls},
            {"q_absmax_head", q_absmax_head},
            {"k_absmax_head", k_absmax_head},
            {"v_sum", v_sum},
            {"v_count", v_count},
            {"v_min", v_min},
            {"v_max", v_max},
        };
    }
};

struct llama_text_log8pv_calib_registry {
    std::string calib_path;
    std::string collect_path;
    std::once_flag load_once;
    std::mutex collect_mutex;
    std::map<int, llama_text_log8pv_static_layer> layers;
    std::map<int, llama_text_log8pv_collect_layer> collect_layers;

    llama_text_log8pv_calib_registry() {
        const char * calib_env = std::getenv("AICAS_TEXT_PREFILL_LOG8PV_CALIB_FILE");
        calib_path = calib_env ? calib_env : "";
        const char * collect_env = std::getenv("AICAS_TEXT_PREFILL_LOG8PV_CALIB_COLLECT_FILE");
        collect_path = collect_env ? collect_env : "";
    }

    ~llama_text_log8pv_calib_registry() {
        flush_collect();
    }

    void load() {
        if (calib_path.empty()) {
            return;
        }
        std::ifstream fin(calib_path, std::ios::binary);
        if (!fin.is_open()) {
            LLAMA_LOG_ERROR("%s: failed to open AICAS_TEXT_PREFILL_LOG8PV_CALIB_FILE=%s\n",
                    __func__, calib_path.c_str());
            GGML_ABORT("missing text log8PV calibration file");
        }

        json doc;
        try {
            fin >> doc;
        } catch (const std::exception & e) {
            LLAMA_LOG_ERROR("%s: failed to parse text log8PV calibration file %s: %s\n",
                    __func__, calib_path.c_str(), e.what());
            GGML_ABORT("invalid text log8PV calibration file");
        }

        const std::string schema = doc.value("schema", "");
        if (schema != "aicas.text_prefill.log8pv_calib.v1") {
            LLAMA_LOG_ERROR("%s: unsupported text log8PV calibration schema '%s' in %s\n",
                    __func__, schema.c_str(), calib_path.c_str());
            GGML_ABORT("unsupported text log8PV calibration schema");
        }

        for (const auto & item : doc.value("layers", json::array())) {
            llama_text_log8pv_static_layer layer;
            layer.layer = item.value("layer", 0);
            layer.n_head = item.value("n_head", 0);
            layer.n_kv_head = item.value("n_kv_head", 0);
            layer.d_head = item.value("d_head", 0);
            layer.q_threshold_head = item.value("q_threshold_head", std::vector<float>{});
            layer.k_threshold_head = item.value("k_threshold_head", std::vector<float>{});
            layer.qk_gain_head = item.value("qk_gain_head", std::vector<float>{});
            layer.v_mean = item.value("v_mean", std::vector<float>{});
            layer.v_threshold = item.value("v_threshold", std::vector<float>{});
            if (layer.n_head <= 0 || layer.n_kv_head <= 0 || layer.d_head <= 0) {
                LLAMA_LOG_WARN("%s: ignoring invalid text log8PV calibration layer=%d\n", __func__, layer.layer);
                continue;
            }
            layers[layer.layer] = std::move(layer);
        }
        LLAMA_LOG_INFO("%s: loaded text log8PV static calibration layers=%zu from %s\n",
                __func__, layers.size(), calib_path.c_str());
    }

    const llama_text_log8pv_static_layer * get_layer(int layer) {
        std::call_once(load_once, [this]() { load(); });
        auto it = layers.find(layer);
        return it == layers.end() ? nullptr : &it->second;
    }

    llama_text_log8pv_collect_layer * get_collect_layer(int layer) {
        if (collect_path.empty()) {
            return nullptr;
        }
        std::lock_guard<std::mutex> lock(collect_mutex);
        return &collect_layers[layer];
    }

    void flush_collect() {
        if (collect_path.empty()) {
            return;
        }
        json out = {
            {"schema", "aicas.text_prefill.log8pv_calib_raw.v1"},
            {"layers", json::array()},
        };
        {
            std::lock_guard<std::mutex> lock(collect_mutex);
            for (const auto & kv : collect_layers) {
                out["layers"].push_back(kv.second.to_json());
            }
        }
        std::ofstream fout(collect_path, std::ios::binary);
        if (!fout.is_open()) {
            LLAMA_LOG_ERROR("%s: failed to write text log8PV calibration collect file: %s\n",
                    __func__, collect_path.c_str());
            return;
        }
        fout << out.dump(2);
    }
};

static llama_text_log8pv_calib_registry & llama_text_log8pv_calib_state() {
    static llama_text_log8pv_calib_registry state;
    return state;
}

struct llama_scale_shift32 {
    int32_t scale = 0;
    int32_t shift = 0;
};

static inline float llama_tensor_get_f32_4d(
        const struct ggml_tensor * tensor,
        int64_t i0,
        int64_t i1,
        int64_t i2,
        int64_t i3) {
    const char * ptr = (const char *) tensor->data +
        i0 * tensor->nb[0] + i1 * tensor->nb[1] + i2 * tensor->nb[2] + i3 * tensor->nb[3];
    switch (tensor->type) {
        case GGML_TYPE_F32:
            return *(const float *) ptr;
        case GGML_TYPE_F16:
            return ggml_fp16_to_fp32(*(const ggml_fp16_t *) ptr);
        case GGML_TYPE_Q8_0:
            {
                const char * row_ptr = (const char *) tensor->data + i1 * tensor->nb[1] + i2 * tensor->nb[2] + i3 * tensor->nb[3];
                const block_q8_0 * blocks = reinterpret_cast<const block_q8_0 *>(row_ptr);
                const int64_t block_idx = i0 / QK8_0;
                const int64_t block_off = i0 % QK8_0;
                return ggml_fp16_to_fp32(blocks[block_idx].d) * (float) blocks[block_idx].qs[block_off];
            }
        default:
            GGML_ABORT("unsupported AICAS tensor read type");
    }
}

static inline void llama_tensor_set_f32_4d(
        struct ggml_tensor * tensor,
        int64_t i0,
        int64_t i1,
        int64_t i2,
        int64_t i3,
        float value) {
    char * ptr = (char *) tensor->data +
        i0 * tensor->nb[0] + i1 * tensor->nb[1] + i2 * tensor->nb[2] + i3 * tensor->nb[3];
    GGML_ASSERT(tensor->type == GGML_TYPE_F32);
    *(float *) ptr = value;
}

static llama_scale_shift32 llama_scale_to_scale_shift32(float scale) {
    llama_scale_shift32 out;
    if (!std::isfinite(scale) || scale == 0.0f) {
        return out;
    }

    int exponent = 0;
    const double abs_scale = std::fabs(static_cast<double>(scale));
    const double mantissa = std::frexp(abs_scale, &exponent);
    int64_t scale_i64 = llrint(mantissa * static_cast<double>(1ULL << 31));
    if (scale_i64 >= (1LL << 31)) {
        scale_i64 >>= 1;
        exponent += 1;
    }

    if (scale_i64 > INT32_MAX) {
        scale_i64 = INT32_MAX;
    }
    if (std::signbit(scale)) {
        scale_i64 = -scale_i64;
    }

    out.scale = static_cast<int32_t>(scale_i64);
    out.shift = exponent - 31;
    return out;
}

static inline double llama_scale_shift32_to_double(llama_scale_shift32 scale) {
    return std::ldexp(static_cast<double>(scale.scale), scale.shift);
}

static inline int16_t llama_bfp16m_quant_value(float value, int exp) {
    const float scaled = std::ldexp(value, -exp);
    long q = lrintf(scaled);
    q = std::max<long>(-32768, std::min<long>(32767, q));
    return (int16_t) q;
}

static inline int8_t llama_bfp16m_exp_to_i8(int exp) {
    return (int8_t) std::max<int>(-128, std::min<int>(127, exp));
}

static inline int64_t llama_bfp16m_rshift_rne_i64(int64_t value, int shift) {
    if (shift <= 0) {
        return value;
    }
    if (shift >= 63) {
        return value < 0 ? -1 : 0;
    }

    const bool neg = value < 0;
    uint64_t abs_value = neg ? (uint64_t) (-(value + 1)) + 1 : (uint64_t) value;
    const uint64_t base = abs_value >> shift;
    const uint64_t rem_mask = (UINT64_C(1) << shift) - 1;
    const uint64_t rem = abs_value & rem_mask;
    const uint64_t half = UINT64_C(1) << (shift - 1);
    uint64_t rounded = base;
    if (rem > half || (rem == half && (base & 1))) {
        ++rounded;
    }

    if (!neg) {
        return rounded > (uint64_t) std::numeric_limits<int64_t>::max()
            ? std::numeric_limits<int64_t>::max()
            : (int64_t) rounded;
    }
    if (rounded >= (UINT64_C(1) << 63)) {
        return std::numeric_limits<int64_t>::min();
    }
    return -(int64_t) rounded;
}

static inline int64_t llama_bfp16m_saturating_add_i64(int64_t a, int64_t b) {
    if (b > 0 && a > std::numeric_limits<int64_t>::max() - b) {
        return std::numeric_limits<int64_t>::max();
    }
    if (b < 0 && a < std::numeric_limits<int64_t>::min() - b) {
        return std::numeric_limits<int64_t>::min();
    }
    return a + b;
}

static int llama_bfp16m_block_exp_for_a(
        const struct ggml_tensor * a,
        const llama_text_bfp16m_userdata * cfg,
        int64_t row,
        int64_t a_i2,
        int64_t a_i3,
        int64_t k0,
        int64_t k1) {
    if (cfg != nullptr && cfg->exp_mode == llama_bfp16m_exp_mode::static_tensor) {
        return cfg->static_exp_a;
    }
    float max_abs = 0.0f;
    for (int64_t k = k0; k < k1; ++k) {
        max_abs = std::max(max_abs, std::fabs(llama_tensor_get_f32_4d(a, k, row, a_i2, a_i3)));
    }
    return max_abs > 0.0f ? (int) std::ceil(std::log2((double) max_abs / 32767.0)) : 0;
}

static int llama_bfp16m_block_exp_for_b(
        const struct ggml_tensor * b,
        const llama_text_bfp16m_userdata * cfg,
        int64_t col,
        int64_t b_i2,
        int64_t b_i3,
        int64_t k0,
        int64_t k1) {
    if (cfg != nullptr && cfg->exp_mode == llama_bfp16m_exp_mode::static_tensor) {
        return cfg->static_exp_b;
    }
    float max_abs = 0.0f;
    for (int64_t k = k0; k < k1; ++k) {
        max_abs = std::max(max_abs, std::fabs(llama_tensor_get_f32_4d(b, k, col, b_i2, b_i3)));
    }
    return max_abs > 0.0f ? (int) std::ceil(std::log2((double) max_abs / 32767.0)) : 0;
}

static inline int8_t llama_bfp8m_quant_value(float value, llama_scale_shift32 scale) {
    const double scale_f = llama_scale_shift32_to_double(scale);
    if (scale_f == 0.0) {
        return 0;
    }
    long q = std::lrint(static_cast<double>(value) / scale_f);
    q = std::max<long>(-127, std::min<long>(127, q));
    return static_cast<int8_t>(q);
}

static inline int8_t llama_attn_log8pv_quant_i8(float value, float threshold) {
    if (!std::isfinite(value) || !std::isfinite(threshold) || threshold <= 0.0f) {
        return 0;
    }
    long q = std::lrint((double) value / (double) threshold * 127.0);
    q = std::max<long>(-127, std::min<long>(127, q));
    return (int8_t) q;
}

static inline int16_t llama_attn_log8pv_quant_v(float value, float mean, float threshold, int v_absmax) {
    if (!std::isfinite(value) || !std::isfinite(mean) || !std::isfinite(threshold) || threshold <= 0.0f) {
        return 0;
    }
    long v8 = std::lrint((double) (value - mean) / (double) threshold * 127.0);
    v8 = std::max<long>(-127, std::min<long>(127, v8));
    if (v_absmax == 127) {
        return (int16_t) v8;
    }
    long vq = std::lrint((double) v8 * (double) v_absmax / 127.0);
    vq = std::max<long>(-v_absmax, std::min<long>(v_absmax, vq));
    return (int16_t) vq;
}

static inline int8_t llama_attn_log8pv_quant_v_i8(float value, float mean, float threshold) {
    return (int8_t) llama_attn_log8pv_quant_v(value, mean, threshold, 127);
}

static uint16_t llama_attn_log8pv_p_max(int p_bits) {
    return (uint16_t) ((1u << p_bits) - 1u);
}

static std::array<uint16_t, 256> llama_attn_log8pv_make_base2_lut(int p_bits) {
    std::array<uint16_t, 256> out{};
    const uint16_t p_max = llama_attn_log8pv_p_max(p_bits);
    for (int d = 0; d < 255; ++d) {
        const long decoded = std::lrint((double) p_max * std::exp2(-(double) d / 16.0));
        out[(size_t) d] = (uint16_t) std::max<long>(0, std::min<long>((long) p_max, decoded));
    }
    out[255] = 0;
    return out;
}

static uint16_t llama_attn_log8pv_decode_u16(uint8_t code, float p_log2_range, int p_bits) {
    if (code == 0) {
        return 0;
    }
    const uint16_t p_max = llama_attn_log8pv_p_max(p_bits);
    const double exponent = -static_cast<double>(p_log2_range) *
        static_cast<double>(255 - code) / 254.0;
    const long decoded = std::lrint((double) p_max * std::exp2(exponent));
    return (uint16_t) std::max<long>(0, std::min<long>((long) p_max, decoded));
}

static uint16_t llama_attn_log8pv_encode_decode_u16(float distance, float p_log2_range, int p_bits) {
    if (!std::isfinite(distance) || distance < 0.0f) {
        return 0;
    }
    const double cutoff = static_cast<double>(p_log2_range) * std::log(2.0);
    if ((double) distance >= cutoff) {
        return 0;
    }
    long code = 255 - std::lrint((double) distance / cutoff * 254.0);
    code = std::max<long>(1, std::min<long>(255, code));
    return llama_attn_log8pv_decode_u16((uint8_t) code, p_log2_range, p_bits);
}

static uint16_t llama_attn_log8pv_encode_decode_u16_q8_12(int32_t distance_q8_12, float p_log2_range, int p_bits) {
    if (distance_q8_12 < 0) {
        return 0;
    }
    const int64_t cutoff_q8_12 = std::max<int64_t>(1,
            (int64_t) std::llround((double) p_log2_range * std::log(2.0) * 4096.0));
    if ((int64_t) distance_q8_12 >= cutoff_q8_12) {
        return 0;
    }
    int64_t code = 255 - ((int64_t) distance_q8_12 * 254 + cutoff_q8_12 / 2) / cutoff_q8_12;
    code = std::max<int64_t>(1, std::min<int64_t>(255, code));
    return llama_attn_log8pv_decode_u16((uint8_t) code, p_log2_range, p_bits);
}

static uint16_t llama_attn_p_direct_u16(float distance, int p_bits) {
    if (!std::isfinite(distance) || distance < 0.0f) {
        return 0;
    }
    const uint16_t p_max = llama_attn_log8pv_p_max(p_bits);
    const long decoded = std::lrint((double) p_max * std::exp(-(double) distance));
    return (uint16_t) std::max<long>(0, std::min<long>((long) p_max, decoded));
}

static uint16_t llama_attn_p_direct_u16_q8_12(int32_t distance_q8_12, int p_bits) {
    if (distance_q8_12 < 0) {
        return 0;
    }
    const uint16_t p_max = llama_attn_log8pv_p_max(p_bits);
    const long decoded = std::lrint((double) p_max * std::exp(-(double) distance_q8_12 / 4096.0));
    return (uint16_t) std::max<long>(0, std::min<long>((long) p_max, decoded));
}

static const std::array<uint16_t, 256> & llama_attn_log8pv_base2_lut(int p_bits) {
    static const std::array<uint16_t, 256> lut16 = llama_attn_log8pv_make_base2_lut(16);
    static const std::array<uint16_t, 256> lut15 = llama_attn_log8pv_make_base2_lut(15);
    static const std::array<uint16_t, 256> lut14 = llama_attn_log8pv_make_base2_lut(14);
    return p_bits == 14 ? lut14 : (p_bits == 15 ? lut15 : lut16);
}

static inline uint16_t llama_attn_log8pv_base2_p(uint32_t code, int p_bits) {
    return llama_attn_log8pv_base2_lut(p_bits)[std::min<uint32_t>(255, code)];
}

struct llama_log8pv_diag_stat {
    uint64_t count = 0;
    uint64_t zero_count = 0;
    uint64_t clip_count = 0;
    double sum_ref = 0.0;
    double sum_hat = 0.0;
    double sum_abs_err = 0.0;
    double sum_sq_err = 0.0;
    double max_abs_err = 0.0;
    double max_abs_ref = 0.0;

    void update(double ref, double hat) {
        const double err = hat - ref;
        const double abs_err = std::fabs(err);
        ++count;
        sum_ref += ref;
        sum_hat += hat;
        sum_abs_err += abs_err;
        sum_sq_err += err * err;
        max_abs_err = std::max(max_abs_err, abs_err);
        max_abs_ref = std::max(max_abs_ref, std::fabs(ref));
    }

    json to_json() const {
        return {
            {"count", count},
            {"zero_count", zero_count},
            {"zero_rate", count > 0 ? (double) zero_count / (double) count : 0.0},
            {"clip_count", clip_count},
            {"clip_rate", count > 0 ? (double) clip_count / (double) count : 0.0},
            {"mean_ref", count > 0 ? sum_ref / (double) count : 0.0},
            {"mean_hat", count > 0 ? sum_hat / (double) count : 0.0},
            {"bias", count > 0 ? (sum_hat - sum_ref) / (double) count : 0.0},
            {"mae", count > 0 ? sum_abs_err / (double) count : 0.0},
            {"rmse", count > 0 ? std::sqrt(sum_sq_err / (double) count) : 0.0},
            {"max_abs_err", max_abs_err},
            {"max_abs_ref", max_abs_ref},
        };
    }
};

struct llama_log8pv_diag_row_stat {
    uint64_t rows = 0;
    double sum_l_rel_err = 0.0;
    double max_l_rel_err = 0.0;
    double sum_p_zero_mass_frac = 0.0;
    double max_p_zero_mass_frac = 0.0;

    json to_json() const {
        return {
            {"rows", rows},
            {"mean_l_rel_err", rows > 0 ? sum_l_rel_err / (double) rows : 0.0},
            {"max_l_rel_err", max_l_rel_err},
            {"mean_p_zero_mass_frac", rows > 0 ? sum_p_zero_mass_frac / (double) rows : 0.0},
            {"max_p_zero_mass_frac", max_p_zero_mass_frac},
        };
    }
};

static std::string llama_text_log8pv_diag_path() {
    const char * value = std::getenv("AICAS_TEXT_PREFILL_LOG8PV_DIAG_FILE");
    return value == nullptr ? std::string() : std::string(value);
}

static std::string llama_text_log8pv_pv_diag_path() {
    const char * value = std::getenv("AICAS_TEXT_PREFILL_LOG8PV_PV_DIAG_FILE");
    return value == nullptr ? std::string() : std::string(value);
}

static const char * llama_text_log8pv_qk_scale_mode_name(int mode) {
    switch (mode) {
        case 1: return "dynamic_perhead";
        case 2: return "dynamic_pertile";
        default: return "static";
    }
}

static const char * llama_text_log8pv_v_scale_mode_name(int mode) {
    switch (mode) {
        case 1: return "dynamic_perhead";
        default: return "static";
    }
}

static int64_t llama_attn_log8pv_round_shift_signed(int64_t value, int bits) {
    const int64_t half = 1LL << (bits - 1);
    if (value >= 0) {
        return (value + half) >> bits;
    }
    return -(((-value) + half) >> bits);
}

static int64_t llama_attn_log8pv_saturate_int48(int64_t value) {
    constexpr int64_t min_value = -(1LL << 47);
    constexpr int64_t max_value =  (1LL << 47) - 1;
    return std::max<int64_t>(min_value, std::min<int64_t>(max_value, value));
}

static int32_t llama_attn_log8pv_saturate_q8_12(int64_t value) {
    constexpr int64_t min_value = -128LL * 4096LL;
    constexpr int64_t max_value =  128LL * 4096LL - 1;
    return (int32_t) std::max<int64_t>(min_value, std::min<int64_t>(max_value, value));
}

static float llama_attn_mask_get_f32(const ggml_tensor * mask, int64_t ik, int64_t iq) {
    if (mask == nullptr) {
        return 0.0f;
    }
    if (ik >= mask->ne[0] || iq >= mask->ne[1]) {
        return -INFINITY;
    }
    return llama_tensor_get_f32_4d(mask, ik, iq, 0, 0);
}

struct llama_log8pv_attention_host_profile {
    int64_t threshold_us = 0;
    int64_t q_quant_us = 0;
    int64_t kv_quant_us = 0;
    int64_t v8_cast_us = 0;
    int64_t gamma_us = 0;
    int64_t npu_call_wall_us = 0;
    int64_t dequant_store_us = 0;
    int64_t cpu_quant_overlap_us = 0;
    int64_t kv_quant_hidden_us = 0;
    int64_t q_quant_hidden_us = 0;
    int64_t npu_wait_after_quant_us = 0;
    int64_t post_hidden_us = 0;
    int64_t pipeline_group = -1;
    int64_t pipeline_enabled = 0;
    int64_t pipeline_workers = 0;
};

static bool llama_log8pv_npu_profile_enabled() {
    static const bool enabled = []() {
        const char * path = std::getenv("AICAS_LOG8PV_NPU_PROFILE_JSONL");
        return path != nullptr && path[0] != '\0';
    }();
    return enabled;
}

#ifdef GGML_USE_NPU
static bool llama_text_log8pv_infer_causal_q_start(
        const ggml_tensor * mask,
        int64_t n_q,
        int64_t n_k,
        uint32_t * q_row_start) {
    if (q_row_start == nullptr || n_q <= 0 || n_k <= 0 || n_q > n_k) {
        return false;
    }
    if (mask == nullptr) {
        *q_row_start = 0;
        return n_q == n_k;
    }

    int64_t inferred_start = -1;
    for (int64_t iq = 0; iq < n_q; ++iq) {
        int64_t max_allowed = -1;
        for (int64_t ik = 0; ik < n_k; ++ik) {
            const float mask_value = llama_attn_mask_get_f32(mask, ik, iq);
            if (std::isfinite(mask_value) && mask_value > -1.0e20f) {
                max_allowed = ik;
            }
        }
        if (max_allowed < 0) {
            return false;
        }
        const int64_t row_start = max_allowed - iq;
        if (row_start < 0 || row_start + n_q > n_k) {
            return false;
        }
        if (inferred_start < 0) {
            inferred_start = row_start;
        } else if (inferred_start != row_start) {
            return false;
        }
    }

    *q_row_start = (uint32_t) inferred_start;
    return true;
}

static void llama_log8pv_npu_profile_append(
        const char * scope,
        int layer,
        int64_t head,
        int64_t kv_head,
        int64_t n_q,
        int64_t n_k,
        uint32_t q_row_start,
        const ggml_backend_npu_log8pv_attention_profile & profile,
        const llama_log8pv_attention_host_profile & host_profile) {
    const char * path = std::getenv("AICAS_LOG8PV_NPU_PROFILE_JSONL");
    if (path == nullptr || path[0] == '\0') {
        return;
    }
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    json record = {
        {"schema", "aicas.log8pv_npu_attention_profile.v1"},
        {"scope", scope},
        {"layer", layer},
        {"head", head},
        {"kv_head", kv_head},
        {"n_q", n_q},
        {"n_k", n_k},
        {"q_row_start", q_row_start},
        {"kv_tokens", profile.kv_tokens},
        {"q_rows", profile.q_rows},
        {"exec_rows", profile.exec_rows},
        {"chunks", profile.chunks},
        {"group_size", profile.group_size},
        {"kv_reuse_hit", profile.kv_reuse_hit},
        {"workspace_reuse", profile.workspace_reuse},
        {"total_us", profile.total_us},
        {"mvin_q_us", profile.mvin_q_us},
        {"mvin_k_us", profile.mvin_k_us},
        {"mvin_v_us", profile.mvin_v_us},
        {"qk_us", profile.qk_us},
        {"logp_mvout_us", profile.logp_mvout_us},
        {"mvin_p_us", profile.mvin_p_us},
        {"pv_us", profile.pv_us},
        {"pv_mvout_us", profile.pv_mvout_us},
        {"qk_overlap_us", profile.qk_overlap_us},
        {"pv_overlap_us", profile.pv_overlap_us},
        {"threshold_us", host_profile.threshold_us},
        {"q_quant_us", host_profile.q_quant_us},
        {"kv_quant_us", host_profile.kv_quant_us},
        {"v8_cast_us", host_profile.v8_cast_us},
        {"gamma_us", host_profile.gamma_us},
        {"npu_call_wall_us", host_profile.npu_call_wall_us},
        {"dequant_store_us", host_profile.dequant_store_us},
        {"cpu_quant_overlap_us", host_profile.cpu_quant_overlap_us},
        {"kv_quant_hidden_us", host_profile.kv_quant_hidden_us},
        {"q_quant_hidden_us", host_profile.q_quant_hidden_us},
        {"npu_wait_after_quant_us", host_profile.npu_wait_after_quant_us},
        {"post_hidden_us", host_profile.post_hidden_us},
        {"pipeline_group", host_profile.pipeline_group},
        {"pipeline_enabled", host_profile.pipeline_enabled},
        {"pipeline_workers", host_profile.pipeline_workers},
    };
    std::ofstream fout(path, std::ios::app | std::ios::binary);
    if (fout.is_open()) {
        fout << record.dump() << "\n";
    }
}
#endif

static llama_scale_shift32 llama_bfp8m_block_scale_for_a(
        const struct ggml_tensor * a,
        int64_t row,
        int64_t a_i2,
        int64_t a_i3,
        int64_t k0,
        int64_t k1) {
    float max_abs = 0.0f;
    for (int64_t k = k0; k < k1; ++k) {
        max_abs = std::max(max_abs, std::fabs(llama_tensor_get_f32_4d(a, k, row, a_i2, a_i3)));
    }
    return max_abs > 0.0f ? llama_scale_to_scale_shift32(max_abs / 127.0f) : llama_scale_shift32{};
}

static llama_scale_shift32 llama_bfp8m_block_scale_for_b(
        const struct ggml_tensor * b,
        int64_t col,
        int64_t b_i2,
        int64_t b_i3,
        int64_t k0,
        int64_t k1) {
    float max_abs = 0.0f;
    for (int64_t k = k0; k < k1; ++k) {
        max_abs = std::max(max_abs, std::fabs(llama_tensor_get_f32_4d(b, k, col, b_i2, b_i3)));
    }
    return max_abs > 0.0f ? llama_scale_to_scale_shift32(max_abs / 127.0f) : llama_scale_shift32{};
}

static llama_scale_shift32 llama_bfp8m_tensor_scale(const struct ggml_tensor * tensor) {
    float max_abs = 0.0f;
    for (int64_t i3 = 0; i3 < tensor->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < tensor->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < tensor->ne[1]; ++i1) {
                for (int64_t i0 = 0; i0 < tensor->ne[0]; ++i0) {
                    max_abs = std::max(max_abs, std::fabs(llama_tensor_get_f32_4d(tensor, i0, i1, i2, i3)));
                }
            }
        }
    }
    return max_abs > 0.0f ? llama_scale_to_scale_shift32(max_abs / 127.0f) : llama_scale_shift32{};
}

static llama_scale_shift32 llama_bfp8m_tile_scale_for_a(
        const struct ggml_tensor * a,
        int64_t row,
        int64_t a_i2,
        int64_t a_i3,
        int64_t tile) {
    const int64_t row0 = (row / tile) * tile;
    const int64_t row1 = std::min<int64_t>(a->ne[1], row0 + tile);
    float max_abs = 0.0f;
    for (int64_t r = row0; r < row1; ++r) {
        for (int64_t k = 0; k < a->ne[0]; ++k) {
            max_abs = std::max(max_abs, std::fabs(llama_tensor_get_f32_4d(a, k, r, a_i2, a_i3)));
        }
    }
    return max_abs > 0.0f ? llama_scale_to_scale_shift32(max_abs / 127.0f) : llama_scale_shift32{};
}

static llama_scale_shift32 llama_bfp8m_tile_scale_for_b(
        const struct ggml_tensor * b,
        int64_t col,
        int64_t b_i2,
        int64_t b_i3,
        int64_t tile) {
    const int64_t col0 = (col / tile) * tile;
    const int64_t col1 = std::min<int64_t>(b->ne[1], col0 + tile);
    float max_abs = 0.0f;
    for (int64_t c = col0; c < col1; ++c) {
        for (int64_t k = 0; k < b->ne[0]; ++k) {
            max_abs = std::max(max_abs, std::fabs(llama_tensor_get_f32_4d(b, k, c, b_i2, b_i3)));
        }
    }
    return max_abs > 0.0f ? llama_scale_to_scale_shift32(max_abs / 127.0f) : llama_scale_shift32{};
}

static void llama_compute_text_bfp16m_mul_mat(
        struct ggml_tensor * dst,
        const struct ggml_tensor * out_template,
        const struct ggml_tensor * a,
        const struct ggml_tensor * b,
        int ith,
        int nth,
        void * userdata) {
    GGML_UNUSED(out_template);
    GGML_UNUSED(nth);

    const auto * cfg = static_cast<const llama_text_bfp16m_userdata *>(userdata);
    GGML_ASSERT(cfg != nullptr);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(a->type == GGML_TYPE_F32 || a->type == GGML_TYPE_F16);
    GGML_ASSERT(b->type == GGML_TYPE_F32 || b->type == GGML_TYPE_F16);
    GGML_ASSERT(a->ne[0] == b->ne[0]);
    GGML_ASSERT(dst->ne[0] == a->ne[1]);
    GGML_ASSERT(dst->ne[1] == b->ne[1]);
    GGML_ASSERT(dst->ne[2] == b->ne[2]);
    GGML_ASSERT(dst->ne[3] == b->ne[3]);

    if (ith != 0) {
        return;
    }

    llama_record_attn_tensor_dist("text_prefill", cfg->pv_matmul ? "pv" : "qk", cfg->pv_matmul ? "v" : "k", a, true);
    llama_record_attn_tensor_dist("text_prefill", cfg->pv_matmul ? "pv" : "qk", cfg->pv_matmul ? "p" : "q", b, false);

    const int64_t k_total = a->ne[0];
    const int64_t exp_block = std::max<int64_t>(1, cfg->k_block);
    const int64_t n_kb = (k_total + exp_block - 1) / exp_block;
    const int64_t a_e_stride_row = n_kb;
    const int64_t a_e_stride_i2 = a->ne[1] * a_e_stride_row;
    const int64_t a_e_stride_i3 = a->ne[2] * a_e_stride_i2;
    const int64_t a_q_stride_row = k_total;
    const int64_t a_q_stride_i2 = a->ne[1] * a_q_stride_row;
    const int64_t a_q_stride_i3 = a->ne[2] * a_q_stride_i2;

    const int64_t b_e_stride_col = n_kb;
    const int64_t b_e_stride_i2 = b->ne[1] * b_e_stride_col;
    const int64_t b_e_stride_i3 = b->ne[2] * b_e_stride_i2;
    const int64_t b_q_stride_col = k_total;
    const int64_t b_q_stride_i2 = b->ne[1] * b_q_stride_col;
    const int64_t b_q_stride_i3 = b->ne[2] * b_q_stride_i2;

    std::vector<int8_t> a_exp((size_t) (a->ne[3] * a_e_stride_i3));
    std::vector<int16_t> a_q((size_t) (a->ne[3] * a_q_stride_i3));
    std::vector<int8_t> b_exp((size_t) (b->ne[3] * b_e_stride_i3));
    std::vector<int16_t> b_q((size_t) (b->ne[3] * b_q_stride_i3));

    for (int64_t i3 = 0; i3 < a->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < a->ne[2]; ++i2) {
            for (int64_t row = 0; row < a->ne[1]; ++row) {
                for (int64_t kb = 0; kb < n_kb; ++kb) {
                    const int64_t k0 = kb * exp_block;
                    const int64_t k1 = std::min(k_total, k0 + exp_block);
                    const int exp = llama_bfp16m_block_exp_for_a(a, cfg, row, i2, i3, k0, k1);
                    const int8_t exp_i8 = llama_bfp16m_exp_to_i8(exp);
                    a_exp[(size_t) (i3 * a_e_stride_i3 + i2 * a_e_stride_i2 + row * a_e_stride_row + kb)] = exp_i8;
                    for (int64_t k = k0; k < k1; ++k) {
                        a_q[(size_t) (i3 * a_q_stride_i3 + i2 * a_q_stride_i2 + row * a_q_stride_row + k)] =
                            llama_bfp16m_quant_value(llama_tensor_get_f32_4d(a, k, row, i2, i3), (int) exp_i8);
                    }
                }
            }
        }
    }

    for (int64_t i3 = 0; i3 < b->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < b->ne[2]; ++i2) {
            for (int64_t col = 0; col < b->ne[1]; ++col) {
                for (int64_t kb = 0; kb < n_kb; ++kb) {
                    const int64_t k0 = kb * exp_block;
                    const int64_t k1 = std::min(k_total, k0 + exp_block);
                    const int exp = llama_bfp16m_block_exp_for_b(b, cfg, col, i2, i3, k0, k1);
                    const int8_t exp_i8 = llama_bfp16m_exp_to_i8(exp);
                    b_exp[(size_t) (i3 * b_e_stride_i3 + i2 * b_e_stride_i2 + col * b_e_stride_col + kb)] = exp_i8;
                    for (int64_t k = k0; k < k1; ++k) {
                        b_q[(size_t) (i3 * b_q_stride_i3 + i2 * b_q_stride_i2 + col * b_q_stride_col + k)] =
                            llama_bfp16m_quant_value(llama_tensor_get_f32_4d(b, k, col, i2, i3), (int) exp_i8);
                    }
                }
            }
        }
    }

    const int64_t total = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
    for (int64_t index = 0; index < total; ++index) {
        int64_t rem = index;
        const int64_t row = rem % dst->ne[0];
        rem /= dst->ne[0];
        const int64_t col = rem % dst->ne[1];
        rem /= dst->ne[1];
        const int64_t i2 = rem % dst->ne[2];
        rem /= dst->ne[2];
        const int64_t i3 = rem;
        const int64_t a_i2 = i2 % a->ne[2];
        const int64_t a_i3 = i3 % a->ne[3];

        bool have_acc = false;
        int acc_exp = 0;
        int64_t acc = 0;
        for (int64_t kb = 0; kb < n_kb; ++kb) {
            const int64_t k0 = kb * exp_block;
            const int64_t k1 = std::min(k_total, k0 + exp_block);
            const int e_a = a_exp[(size_t) (a_i3 * a_e_stride_i3 + a_i2 * a_e_stride_i2 + row * a_e_stride_row + kb)];
            const int e_b = b_exp[(size_t) (i3 * b_e_stride_i3 + i2 * b_e_stride_i2 + col * b_e_stride_col + kb)];
            const int partial_exp = e_a + e_b;

            int64_t partial = 0;
            for (int64_t k = k0; k < k1; ++k) {
                const int16_t qa = a_q[(size_t) (a_i3 * a_q_stride_i3 + a_i2 * a_q_stride_i2 + row * a_q_stride_row + k)];
                const int16_t qb = b_q[(size_t) (i3 * b_q_stride_i3 + i2 * b_q_stride_i2 + col * b_q_stride_col + k)];
                partial += (int32_t) qa * (int32_t) qb;
            }

            if (!have_acc) {
                acc = partial;
                acc_exp = partial_exp;
                have_acc = true;
                continue;
            }

            if (partial_exp > acc_exp) {
                acc = llama_bfp16m_rshift_rne_i64(acc, partial_exp - acc_exp);
                acc_exp = partial_exp;
            }
            acc = llama_bfp16m_saturating_add_i64(acc, llama_bfp16m_rshift_rne_i64(partial, acc_exp - partial_exp));
        }

        llama_tensor_set_f32_4d(dst, row, col, i2, i3, (float) std::ldexp((double) acc, acc_exp));
    }
}

static ggml_tensor * llama_build_text_bfp16m_mul_mat(
        ggml_context * ctx0,
        ggml_tensor * a,
        ggml_tensor * b,
        bool pv_matmul) {
    GGML_ASSERT(a->ne[0] == b->ne[0]);
    GGML_ASSERT(b->ne[2] % a->ne[2] == 0);
    GGML_ASSERT(b->ne[3] % a->ne[3] == 0);

    ggml_tensor * out_template = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, a->ne[1], b->ne[1], b->ne[2], b->ne[3]);
    static llama_text_bfp16m_userdata qk_cfg = llama_make_text_bfp16m_userdata(false);
    static llama_text_bfp16m_userdata pv_cfg = llama_make_text_bfp16m_userdata(true);
    llama_text_bfp16m_userdata * cfg = pv_matmul ? &pv_cfg : &qk_cfg;
    return ggml_map_custom3(
            ctx0,
            out_template,
            a,
            b,
            llama_compute_text_bfp16m_mul_mat,
            1,
            cfg);
}

static void llama_compute_text_bfp8m_mul_mat(
        struct ggml_tensor * dst,
        const struct ggml_tensor * out_template,
        const struct ggml_tensor * a,
        const struct ggml_tensor * b,
        int ith,
        int nth,
        void * userdata) {
    GGML_UNUSED(out_template);
    GGML_UNUSED(nth);

    const auto * cfg = static_cast<const llama_text_bfp8m_userdata *>(userdata);
    GGML_ASSERT(cfg != nullptr);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(a->type == GGML_TYPE_F32 || a->type == GGML_TYPE_F16);
    GGML_ASSERT(b->type == GGML_TYPE_F32 || b->type == GGML_TYPE_F16);
    GGML_ASSERT(a->ne[0] == b->ne[0]);
    GGML_ASSERT(dst->ne[0] == a->ne[1]);
    GGML_ASSERT(dst->ne[1] == b->ne[1]);
    GGML_ASSERT(dst->ne[2] == b->ne[2]);
    GGML_ASSERT(dst->ne[3] == b->ne[3]);

    if (ith != 0) {
        return;
    }

    const int64_t k_total = a->ne[0];
    const bool per_tensor_scale = cfg->scale_mode == llama_bfp8m_scale_mode::tensor;
    const bool per_tile_scale = cfg->scale_mode == llama_bfp8m_scale_mode::tile;
    const int64_t scale_block = (per_tensor_scale || per_tile_scale) ? k_total : std::max<int64_t>(1, cfg->k_block);
    const int64_t n_kb = (k_total + scale_block - 1) / scale_block;
    const int64_t a_s_stride_row = n_kb;
    const int64_t a_s_stride_i2 = a->ne[1] * a_s_stride_row;
    const int64_t a_s_stride_i3 = a->ne[2] * a_s_stride_i2;
    const int64_t a_q_stride_row = k_total;
    const int64_t a_q_stride_i2 = a->ne[1] * a_q_stride_row;
    const int64_t a_q_stride_i3 = a->ne[2] * a_q_stride_i2;

    const int64_t b_s_stride_col = n_kb;
    const int64_t b_s_stride_i2 = b->ne[1] * b_s_stride_col;
    const int64_t b_s_stride_i3 = b->ne[2] * b_s_stride_i2;
    const int64_t b_q_stride_col = k_total;
    const int64_t b_q_stride_i2 = b->ne[1] * b_q_stride_col;
    const int64_t b_q_stride_i3 = b->ne[2] * b_q_stride_i2;

    std::vector<llama_scale_shift32> a_scale((size_t) (a->ne[3] * a_s_stride_i3));
    std::vector<int8_t> a_q((size_t) (a->ne[3] * a_q_stride_i3));
    std::vector<llama_scale_shift32> b_scale((size_t) (b->ne[3] * b_s_stride_i3));
    std::vector<int8_t> b_q((size_t) (b->ne[3] * b_q_stride_i3));
    const llama_scale_shift32 tensor_scale_a = per_tensor_scale ? llama_bfp8m_tensor_scale(a) : llama_scale_shift32{};
    const llama_scale_shift32 tensor_scale_b = per_tensor_scale ? llama_bfp8m_tensor_scale(b) : llama_scale_shift32{};
    const int64_t tile = std::max<int64_t>(1, cfg->tile);

    for (int64_t i3 = 0; i3 < a->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < a->ne[2]; ++i2) {
            for (int64_t row = 0; row < a->ne[1]; ++row) {
                for (int64_t kb = 0; kb < n_kb; ++kb) {
                    const int64_t k0 = kb * scale_block;
                    const int64_t k1 = std::min(k_total, k0 + scale_block);
                    const llama_scale_shift32 scale = per_tensor_scale
                        ? tensor_scale_a
                        : (per_tile_scale
                            ? llama_bfp8m_tile_scale_for_a(a, row, i2, i3, tile)
                            : llama_bfp8m_block_scale_for_a(a, row, i2, i3, k0, k1));
                    a_scale[(size_t) (i3 * a_s_stride_i3 + i2 * a_s_stride_i2 + row * a_s_stride_row + kb)] = scale;
                    for (int64_t k = k0; k < k1; ++k) {
                        a_q[(size_t) (i3 * a_q_stride_i3 + i2 * a_q_stride_i2 + row * a_q_stride_row + k)] =
                            llama_bfp8m_quant_value(llama_tensor_get_f32_4d(a, k, row, i2, i3), scale);
                    }
                }
            }
        }
    }

    for (int64_t i3 = 0; i3 < b->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < b->ne[2]; ++i2) {
            for (int64_t col = 0; col < b->ne[1]; ++col) {
                for (int64_t kb = 0; kb < n_kb; ++kb) {
                    const int64_t k0 = kb * scale_block;
                    const int64_t k1 = std::min(k_total, k0 + scale_block);
                    const llama_scale_shift32 scale = per_tensor_scale
                        ? tensor_scale_b
                        : (per_tile_scale
                            ? llama_bfp8m_tile_scale_for_b(b, col, i2, i3, tile)
                            : llama_bfp8m_block_scale_for_b(b, col, i2, i3, k0, k1));
                    b_scale[(size_t) (i3 * b_s_stride_i3 + i2 * b_s_stride_i2 + col * b_s_stride_col + kb)] = scale;
                    for (int64_t k = k0; k < k1; ++k) {
                        b_q[(size_t) (i3 * b_q_stride_i3 + i2 * b_q_stride_i2 + col * b_q_stride_col + k)] =
                            llama_bfp8m_quant_value(llama_tensor_get_f32_4d(b, k, col, i2, i3), scale);
                    }
                }
            }
        }
    }

    const int64_t total = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
    for (int64_t index = 0; index < total; ++index) {
        int64_t rem = index;
        const int64_t row = rem % dst->ne[0];
        rem /= dst->ne[0];
        const int64_t col = rem % dst->ne[1];
        rem /= dst->ne[1];
        const int64_t i2 = rem % dst->ne[2];
        rem /= dst->ne[2];
        const int64_t i3 = rem;
        const int64_t a_i2 = i2 % a->ne[2];
        const int64_t a_i3 = i3 % a->ne[3];

        double acc = 0.0;
        for (int64_t kb = 0; kb < n_kb; ++kb) {
            const int64_t k0 = kb * scale_block;
            const int64_t k1 = std::min(k_total, k0 + scale_block);
            int64_t partial = 0;
            for (int64_t k = k0; k < k1; ++k) {
                const int8_t qa = a_q[(size_t) (a_i3 * a_q_stride_i3 + a_i2 * a_q_stride_i2 + row * a_q_stride_row + k)];
                const int8_t qb = b_q[(size_t) (i3 * b_q_stride_i3 + i2 * b_q_stride_i2 + col * b_q_stride_col + k)];
                partial += (int32_t) qa * (int32_t) qb;
            }

            const llama_scale_shift32 scale_a =
                a_scale[(size_t) (a_i3 * a_s_stride_i3 + a_i2 * a_s_stride_i2 + row * a_s_stride_row + kb)];
            const llama_scale_shift32 scale_b =
                b_scale[(size_t) (i3 * b_s_stride_i3 + i2 * b_s_stride_i2 + col * b_s_stride_col + kb)];
            const double partial_scale = std::ldexp(
                    static_cast<double>(scale_a.scale) * static_cast<double>(scale_b.scale),
                    scale_a.shift + scale_b.shift);
            acc += static_cast<double>(partial) * partial_scale;
        }

        llama_tensor_set_f32_4d(dst, row, col, i2, i3, static_cast<float>(acc));
    }
}

static ggml_tensor * llama_build_text_bfp8m_mul_mat(
        ggml_context * ctx0,
        ggml_tensor * a,
        ggml_tensor * b) {
    GGML_ASSERT(a->ne[0] == b->ne[0]);
    GGML_ASSERT(b->ne[2] % a->ne[2] == 0);
    GGML_ASSERT(b->ne[3] % a->ne[3] == 0);

    ggml_tensor * out_template = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, a->ne[1], b->ne[1], b->ne[2], b->ne[3]);
    static llama_text_bfp8m_userdata cfg = {
        /*.k_block =*/ llama_text_prefill_bfp8m_k_block(),
        /*.tile =*/ llama_text_prefill_bfp8m_tile(),
        /*.scale_mode =*/ llama_text_prefill_bfp8m_scale_mode(),
    };
    return ggml_map_custom3(
            ctx0,
            out_template,
            a,
            b,
            llama_compute_text_bfp8m_mul_mat,
            1,
            &cfg);
}

static void llama_compute_text_log8pv_attn(
        struct ggml_tensor * dst,
        int ith,
        int nth,
        void * userdata) {
    GGML_UNUSED(nth);

    const auto * cfg = static_cast<const llama_text_log8pv_attn_userdata *>(userdata);
    const struct ggml_tensor * q = dst->src[0];
    const struct ggml_tensor * k = dst->src[1];
    const struct ggml_tensor * v = dst->src[2];
    const struct ggml_tensor * mask = dst->src[3] != nullptr ? dst->src[3] : cfg->mask;
    GGML_ASSERT(cfg != nullptr);
    GGML_ASSERT(q != nullptr);
    GGML_ASSERT(k != nullptr);
    GGML_ASSERT(v != nullptr);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(q->type == GGML_TYPE_F32 || q->type == GGML_TYPE_F16);
    GGML_ASSERT(k->type == GGML_TYPE_F32 || k->type == GGML_TYPE_F16 || k->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(v->type == GGML_TYPE_F32 || v->type == GGML_TYPE_F16 || v->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(q->ne[0] == k->ne[0]);
    GGML_ASSERT(q->ne[0] == v->ne[0]);
    GGML_ASSERT(dst->ne[0] == q->ne[0]);
    GGML_ASSERT(dst->ne[1] == q->ne[2]);
    GGML_ASSERT(dst->ne[2] == q->ne[1]);
    GGML_ASSERT(dst->ne[3] == q->ne[3]);

    if (ith != 0) {
        return;
    }

    const int64_t d_head = q->ne[0];
    const int64_t n_q = q->ne[1];
    const int64_t n_head = q->ne[2];
    const int64_t n_batch = q->ne[3];
    const int64_t n_k = k->ne[1];
    const int64_t n_kv_head = k->ne[2];
    const int64_t tile = std::max<int64_t>(1, cfg->tile);
    const uint16_t p_max = llama_attn_log8pv_p_max(cfg->p_bits);
    const bool dynamic_qk_head = cfg->qk_scale_mode == 1;
    const bool dynamic_qk_tile = cfg->qk_scale_mode == 2;
    const bool dynamic_v_head = cfg->v_scale_mode == 1;
    const int64_t dynamic_scale_tile = std::max<int64_t>(1, cfg->qk_scale_tile);
    const int64_t n_q_scale_tiles = (n_q + dynamic_scale_tile - 1) / dynamic_scale_tile;
    const int64_t n_k_scale_tiles = (n_k + dynamic_scale_tile - 1) / dynamic_scale_tile;
    GGML_ASSERT(d_head == 64);
    GGML_ASSERT(n_head % n_kv_head == 0);

    static bool logged = false;
    if (!logged) {
        logged = true;
        LLAMA_LOG_INFO(
                "llama_compute_text_log8pv_attn: AICAS text prefill log8PV CPU reference enabled, "
                "tile=%" PRId64 " qk_scale_mode=%s qk_scale_tile=%" PRId64 " v_scale_mode=%s"
                " v_centering=%d p_log2_range=%.3f p_mode=%s p_bits=%d bit_accurate=%d alpha_bits=%d v_bits=%d\n",
                tile, llama_text_log8pv_qk_scale_mode_name(cfg->qk_scale_mode), dynamic_scale_tile,
                llama_text_log8pv_v_scale_mode_name(cfg->v_scale_mode),
                cfg->v_centering ? 1 : 0,
                (double) cfg->p_log2_range, cfg->p_direct_u16 ? "direct_u16" : "log8",
                cfg->p_bits, cfg->bit_accurate ? 1 : 0, cfg->alpha_bits, cfg->v_bits);
    }

    const bool has_calib = cfg->calib != nullptr && cfg->calib->valid_for(n_head, n_kv_head, d_head);
    if (!has_calib && cfg->collect == nullptr && !dynamic_v_head) {
        LLAMA_LOG_ERROR("%s: AICAS_TEXT_PREFILL_LOG8PV_CALIB_FILE is required for static text log8PV attention\n",
                __func__);
        GGML_ABORT("missing static text log8PV calibration");
    }
    if (cfg->calib != nullptr && !has_calib) {
        LLAMA_LOG_ERROR("%s: text log8PV calibration layer=%d shape mismatch, got heads=%" PRId64
                " kv_heads=%" PRId64 " dim=%" PRId64 "\n",
                __func__, cfg->layer, n_head, n_kv_head, d_head);
        GGML_ABORT("invalid static text log8PV calibration shape");
    }
    if (cfg->p_direct_u16 || cfg->p_log2_range != 16.0f || cfg->alpha_bits != 20 || cfg->v_bits != 8) {
        LLAMA_LOG_WARN("%s: text log8PV hardware-equivalent path ignores legacy p/alpha/v controls; "
                "using base2 S_q, P%d LUT, and V8\n", __func__, cfg->p_bits);
    }

    constexpr int v_absmax = 127;
    constexpr int32_t invalid_score = std::numeric_limits<int32_t>::min();
    std::vector<int8_t> q8((size_t) (n_q * d_head));
    std::vector<int8_t> k8((size_t) (n_k * d_head));
    std::vector<int16_t> vq((size_t) (n_k * d_head));
    std::vector<int32_t> score_s16((size_t) n_k);
    std::vector<uint8_t> q_local((size_t) n_k);
    std::vector<int32_t> m_tile((size_t) ((n_k + tile - 1) / tile));
    std::vector<float> v_mean((size_t) d_head);
    std::vector<float> v_threshold((size_t) d_head);
    std::vector<float> q_threshold_dyn_tile(dynamic_qk_tile ? (size_t) n_q_scale_tiles : 0);
    std::vector<float> k_threshold_dyn_tile(dynamic_qk_tile ? (size_t) n_k_scale_tiles : 0);
    std::vector<int64_t> accum((size_t) d_head);
    const std::string diag_path = llama_text_log8pv_diag_path();
    const std::string pv_diag_path = llama_text_log8pv_pv_diag_path();
    const bool diag_enabled = !diag_path.empty();
    const bool pv_diag_enabled = diag_enabled || !pv_diag_path.empty();
    llama_log8pv_diag_stat diag_q;
    llama_log8pv_diag_stat diag_k;
    llama_log8pv_diag_stat diag_v;
    llama_log8pv_diag_stat diag_qk;
    llama_log8pv_diag_stat diag_p;
    llama_log8pv_diag_stat diag_o;
    llama_log8pv_diag_row_stat diag_rows;
    int64_t diag_pv_max_abs_product = 0;
    int64_t diag_pv_max_abs_accum = 0;
    int64_t diag_pv_max_abs_final_accum = 0;
    int64_t diag_pv_max_row_l = 0;
    uint64_t diag_pv_accum_over_i32 = 0;
    uint64_t diag_pv_final_over_i32 = 0;
    uint64_t diag_pv_accum_updates = 0;
    std::vector<double> ref_logits(diag_enabled ? (size_t) n_k : 0);
    std::vector<double> ref_out(diag_enabled ? (size_t) d_head : 0);

#ifdef GGML_USE_NPU
    const bool text_log8pv_npu_requested_fast = llama_text_prefill_log8pv_npu_enabled();
    uint32_t npu_fast_q_row_start = 0;
    const bool text_log8pv_npu_causal_ok_fast =
        llama_text_log8pv_infer_causal_q_start(mask, n_q, n_k, &npu_fast_q_row_start);
    const bool text_log8pv_npu_fast_shape_ok =
        text_log8pv_npu_requested_fast &&
        cfg->collect == nullptr &&
        !diag_enabled &&
        !pv_diag_enabled &&
        has_calib &&
        !dynamic_qk_head &&
        !dynamic_qk_tile &&
        !dynamic_v_head &&
        !cfg->p_direct_u16 &&
        cfg->p_log2_range == 16.0f &&
        cfg->p_bits == 15 &&
        cfg->alpha_bits == 20 &&
        cfg->v_bits == 8 &&
        d_head == 64 &&
        n_q > 0 &&
        n_k > 0 &&
        n_k <= 1024 &&
        n_head % n_kv_head == 0 &&
        (n_head / n_kv_head) <= 8 &&
        npu_fast_q_row_start + (uint32_t) n_q <= (uint32_t) n_k &&
        text_log8pv_npu_causal_ok_fast &&
        (mask != nullptr || (n_k % 32) == 0);

    if (text_log8pv_npu_requested_fast && text_log8pv_npu_fast_shape_ok) {
        const int64_t gqa_ratio = n_head / n_kv_head;
        const bool causal_mask = mask != nullptr;
        std::vector<int8_t> q_group((size_t) gqa_ratio * n_q * d_head);
        std::vector<int8_t> k8_group((size_t) n_k * d_head);
        std::vector<int8_t> v8_group((size_t) n_k * d_head);
        std::vector<int32_t> npu_out_group((size_t) gqa_ratio * n_q * d_head);
        std::vector<uint32_t> gamma_group((size_t) gqa_ratio);
        std::vector<ggml_backend_npu_log8pv_attention_profile> npu_profiles((size_t) gqa_ratio);
        std::vector<llama_log8pv_attention_host_profile> host_profiles((size_t) gqa_ratio);
        std::vector<float> v_mean_group((size_t) d_head);
        std::vector<float> v_threshold_group((size_t) d_head);

        bool npu_fast_ok = true;
        static std::atomic<bool> npu_group_logged{false};
        if (!npu_group_logged.exchange(true)) {
            LLAMA_LOG_INFO(
                    "llama_compute_text_log8pv_attn: trying Versa_P NPU grouped text log8PV attention, "
                    "n_q=%" PRId64 " n_k=%" PRId64 " q_row_start=%u gqa_ratio=%" PRId64 " causal=%d\n",
                    n_q, n_k, npu_fast_q_row_start, gqa_ratio, causal_mask ? 1 : 0);
        }

        struct prepared_group {
            int64_t ib = 0;
            int64_t ikh = 0;
            int64_t group_index = 0;
            bool ok = true;
            int64_t prepare_wall_us = 0;
            std::vector<int8_t> q;
            std::vector<int8_t> k;
            std::vector<int8_t> v;
            std::vector<int32_t> out;
            std::vector<uint32_t> gamma;
            std::vector<ggml_backend_npu_log8pv_attention_profile> npu_profiles;
            std::vector<llama_log8pv_attention_host_profile> host_profiles;
            std::vector<float> v_mean;
            std::vector<float> v_threshold;
        };

        const bool pipeline_enabled =
            llama_text_log8pv_cpu_npu_pipeline_enabled() &&
            (n_batch * n_kv_head) > 1;
        auto prepare_group = [&](int64_t group_index) {
            prepared_group work;
            work.ib = group_index / n_kv_head;
            work.ikh = group_index % n_kv_head;
            work.group_index = group_index;
            work.q.resize((size_t) gqa_ratio * n_q * d_head);
            work.k.resize((size_t) n_k * d_head);
            work.v.resize((size_t) n_k * d_head);
            work.out.resize((size_t) gqa_ratio * n_q * d_head);
            work.gamma.resize((size_t) gqa_ratio);
            work.npu_profiles.resize((size_t) gqa_ratio);
            work.host_profiles.resize((size_t) gqa_ratio);
            work.v_mean.resize((size_t) d_head);
            work.v_threshold.resize((size_t) d_head);
            const int64_t prepare_start_us = ggml_time_us();
            int64_t npu_profile_t0 = llama_log8pv_npu_profile_enabled() ? ggml_time_us() : 0;
            const float k_threshold = std::max(cfg->calib->k_threshold_head[(size_t) work.ikh], 1.0e-6f);
            for (int64_t id = 0; id < d_head; ++id) {
                const size_t offset = (size_t) (work.ikh * d_head + id);
                if (cfg->v_centering) {
                    work.v_mean[(size_t) id] = cfg->calib->v_mean[offset];
                    work.v_threshold[(size_t) id] = std::max(cfg->calib->v_threshold[offset], 1.0e-6f);
                } else {
                    work.v_mean[(size_t) id] = 0.0f;
                    work.v_threshold[(size_t) id] = std::max(
                            std::fabs(cfg->calib->v_mean[offset]) + cfg->calib->v_threshold[offset],
                            1.0e-6f);
                }
            }
            if (llama_log8pv_npu_profile_enabled()) {
                work.host_profiles[0].threshold_us += ggml_time_us() - npu_profile_t0;
            }

            npu_profile_t0 = llama_log8pv_npu_profile_enabled() ? ggml_time_us() : 0;
            for (int64_t ik = 0; ik < n_k; ++ik) {
                for (int64_t id = 0; id < d_head; ++id) {
                    work.k[(size_t) ik * d_head + id] = llama_attn_log8pv_quant_i8(
                            llama_tensor_get_f32_4d(k, id, ik, work.ikh, work.ib),
                            k_threshold);
                    work.v[(size_t) ik * d_head + id] = llama_attn_log8pv_quant_v_i8(
                            llama_tensor_get_f32_4d(v, id, ik, work.ikh, work.ib),
                            work.v_mean[(size_t) id],
                            work.v_threshold[(size_t) id]);
                }
            }
            if (llama_log8pv_npu_profile_enabled()) {
                work.host_profiles[0].kv_quant_us += ggml_time_us() - npu_profile_t0;
            }

            for (int64_t local = 0; local < gqa_ratio; ++local) {
                const int64_t ih = work.ikh * gqa_ratio + local;
                const float q_threshold = std::max(cfg->calib->q_threshold_head[(size_t) ih], 1.0e-6f);
                npu_profile_t0 = llama_log8pv_npu_profile_enabled() ? ggml_time_us() : 0;
                for (int64_t iq = 0; iq < n_q; ++iq) {
                    for (int64_t id = 0; id < d_head; ++id) {
                        work.q[(size_t) local * n_q * d_head + (size_t) iq * d_head + id] =
                            llama_attn_log8pv_quant_i8(
                                    llama_tensor_get_f32_4d(q, id, iq, ih, work.ib),
                                    q_threshold);
                    }
                }
                if (llama_log8pv_npu_profile_enabled()) {
                    work.host_profiles[(size_t) local].q_quant_us += ggml_time_us() - npu_profile_t0;
                }

                const double qk_gain = cfg->calib->qk_gain_head.size() == (size_t) n_head &&
                    std::isfinite(cfg->calib->qk_gain_head[(size_t) ih]) &&
                    cfg->calib->qk_gain_head[(size_t) ih] > 0.0f
                        ? (double) cfg->calib->qk_gain_head[(size_t) ih]
                        : 1.0;
                npu_profile_t0 = llama_log8pv_npu_profile_enabled() ? ggml_time_us() : 0;
                const double gamma2 =
                    (double) q_threshold / 127.0 *
                    (double) k_threshold / 127.0 *
                    (double) cfg->kq_scale *
                    qk_gain / std::log(2.0);
                const double gamma_fix_f = std::round(16.0 * gamma2 * (double) (UINT64_C(1) << 24));
                work.gamma[(size_t) local] =
                    gamma_fix_f > 1.0 && gamma_fix_f < (double) std::numeric_limits<uint32_t>::max()
                        ? (uint32_t) gamma_fix_f
                        : 0u;
                if (llama_log8pv_npu_profile_enabled()) {
                    work.host_profiles[(size_t) local].gamma_us += ggml_time_us() - npu_profile_t0;
                }
                if (work.gamma[(size_t) local] == 0) {
                    work.ok = false;
                    break;
                }
            }

            work.prepare_wall_us = ggml_time_us() - prepare_start_us;
            return work;
        };

        auto launch_prepare = [&](int64_t group_index) {
            return std::async(std::launch::async, prepare_group, group_index);
        };

        const int64_t group_count = n_batch * n_kv_head;
        const int64_t pipeline_workers = pipeline_enabled
            ? std::min<int64_t>(llama_text_log8pv_cpu_npu_pipeline_workers(), group_count)
            : 0;
        std::deque<std::pair<int64_t, std::future<prepared_group>>> pending;
        int64_t next_prepare_group = 0;
        auto fill_prepare_queue = [&]() {
            while (pipeline_workers > 0 &&
                    (int64_t) pending.size() < pipeline_workers &&
                    next_prepare_group < group_count) {
                pending.emplace_back(next_prepare_group, launch_prepare(next_prepare_group));
                ++next_prepare_group;
            }
        };
        if (pipeline_enabled) {
            fill_prepare_queue();
        }
        for (int64_t group_index = 0; group_index < group_count && npu_fast_ok; ++group_index) {
            int64_t prepare_wait_us = 0;
            prepared_group work;
            if (pipeline_enabled) {
                if (pending.empty() || pending.front().first != group_index) {
                    npu_fast_ok = false;
                    break;
                }
                const int64_t wait_start_us = ggml_time_us();
                work = pending.front().second.get();
                pending.pop_front();
                prepare_wait_us = ggml_time_us() - wait_start_us;
                fill_prepare_queue();
            } else {
                work = prepare_group(group_index);
            }

            const int64_t prepare_hidden_us = pipeline_enabled
                ? std::max<int64_t>(0, work.prepare_wall_us - prepare_wait_us)
                : 0;
            for (int64_t local = 0; local < gqa_ratio; ++local) {
                llama_log8pv_attention_host_profile & hp = work.host_profiles[(size_t) local];
                hp.pipeline_enabled = pipeline_enabled ? 1 : 0;
                hp.pipeline_group = work.group_index;
                hp.pipeline_workers = pipeline_workers;
                hp.cpu_quant_overlap_us += prepare_hidden_us / std::max<int64_t>(1, gqa_ratio);
                hp.npu_wait_after_quant_us += prepare_wait_us / std::max<int64_t>(1, gqa_ratio);
                hp.q_quant_hidden_us += std::min<int64_t>(hp.q_quant_us, prepare_hidden_us);
            }
            work.host_profiles[0].kv_quant_hidden_us += std::min<int64_t>(
                    work.host_profiles[0].kv_quant_us,
                    prepare_hidden_us);
            if (!work.ok) {
                npu_fast_ok = false;
                break;
            }

            int64_t npu_profile_t0 = llama_log8pv_npu_profile_enabled() ? ggml_time_us() : 0;
            const bool group_ok = ggml_backend_npu_log8pv_attention_group(
                    work.q.data(),
                    (uint32_t) gqa_ratio,
                    (uint32_t) n_q,
                    work.k.data(),
                    work.v.data(),
                    (uint32_t) n_k,
                    npu_fast_q_row_start,
                    causal_mask,
                    work.gamma.data(),
                    work.out.data(),
                    (uint32_t) d_head,
                    60000,
                    work.npu_profiles.data());
            if (llama_log8pv_npu_profile_enabled()) {
                work.host_profiles[0].npu_call_wall_us += ggml_time_us() - npu_profile_t0;
            }
            if (!group_ok) {
                npu_fast_ok = false;
                break;
            }

            for (int64_t local = 0; local < gqa_ratio; ++local) {
                const int64_t ih = work.ikh * gqa_ratio + local;
                npu_profile_t0 = llama_log8pv_npu_profile_enabled() ? ggml_time_us() : 0;
                const int32_t * npu_out = work.out.data() + (size_t) local * n_q * d_head;
                for (int64_t iq = 0; iq < n_q; ++iq) {
                    for (int64_t id = 0; id < d_head; ++id) {
                        const float out = (float) (
                                (double) npu_out[(size_t) iq * d_head + id] *
                                ((double) work.v_threshold[(size_t) id] / (double) v_absmax) +
                                (double) work.v_mean[(size_t) id]);
                        llama_tensor_set_f32_4d(dst, id, ih, iq, work.ib, out);
                    }
                }
                if (llama_log8pv_npu_profile_enabled()) {
                    const int64_t post_us = ggml_time_us() - npu_profile_t0;
                    work.host_profiles[(size_t) local].dequant_store_us += post_us;
                    work.host_profiles[(size_t) local].post_hidden_us += pipeline_enabled ? post_us : 0;
                }
                llama_log8pv_npu_profile_append(
                        "text", cfg->layer, ih, work.ikh, n_q, n_k, npu_fast_q_row_start,
                        work.npu_profiles[(size_t) local], work.host_profiles[(size_t) local]);
            }
        }
        if (npu_fast_ok) {
            return;
        }
        static std::atomic<bool> npu_group_fallback_logged{false};
        if (!npu_group_fallback_logged.exchange(true)) {
            LLAMA_LOG_WARN(
                    "llama_compute_text_log8pv_attn: grouped Versa_P NPU text log8PV attention failed; "
                    "falling back to per-head path\n");
        }
    }
#endif

    for (int64_t ib = 0; ib < n_batch; ++ib) {
        for (int64_t ih = 0; ih < n_head; ++ih) {
            const int64_t ikh = ih / (n_head / n_kv_head);
            llama_log8pv_attention_host_profile npu_host_profile = {};
            const bool npu_host_profile_enabled = llama_log8pv_npu_profile_enabled();
            if (cfg->collect != nullptr) {
                cfg->collect->ensure(cfg->layer, n_head, n_kv_head, d_head);
                cfg->collect->calls += 1;
                for (int64_t iq = 0; iq < n_q; ++iq) {
                    for (int64_t id = 0; id < d_head; ++id) {
                        const float qv = llama_tensor_get_f32_4d(q, id, iq, ih, ib);
                        cfg->collect->q_absmax_head[(size_t) ih] =
                            std::max(cfg->collect->q_absmax_head[(size_t) ih], std::fabs(qv));
                    }
                }
                for (int64_t ik = 0; ik < n_k; ++ik) {
                    for (int64_t id = 0; id < d_head; ++id) {
                        const size_t offset = (size_t) (ikh * d_head + id);
                        const float kv = llama_tensor_get_f32_4d(k, id, ik, ikh, ib);
                        const float vv = llama_tensor_get_f32_4d(v, id, ik, ikh, ib);
                        cfg->collect->k_absmax_head[(size_t) ikh] =
                            std::max(cfg->collect->k_absmax_head[(size_t) ikh], std::fabs(kv));
                        cfg->collect->v_sum[offset] += vv;
                        cfg->collect->v_count[offset] += 1;
                        cfg->collect->v_min[offset] = std::min(cfg->collect->v_min[offset], vv);
                        cfg->collect->v_max[offset] = std::max(cfg->collect->v_max[offset], vv);
                    }
                }
            }

            int64_t npu_profile_t0 = npu_host_profile_enabled ? ggml_time_us() : 0;
            float q_threshold = has_calib ? cfg->calib->q_threshold_head[(size_t) ih] : 0.0f;
            float k_threshold = has_calib ? cfg->calib->k_threshold_head[(size_t) ikh] : 0.0f;
            if (q_threshold <= 0.0f || !std::isfinite(q_threshold)) {
                for (int64_t iq = 0; iq < n_q; ++iq) {
                    for (int64_t id = 0; id < d_head; ++id) {
                        q_threshold = std::max(q_threshold, std::fabs(llama_tensor_get_f32_4d(q, id, iq, ih, ib)));
                    }
                }
            }
            if (k_threshold <= 0.0f || !std::isfinite(k_threshold)) {
                for (int64_t ik = 0; ik < n_k; ++ik) {
                    for (int64_t id = 0; id < d_head; ++id) {
                        k_threshold = std::max(k_threshold, std::fabs(llama_tensor_get_f32_4d(k, id, ik, ikh, ib)));
                    }
                }
            }
            if (q_threshold <= 0.0f || !std::isfinite(q_threshold)) {
                q_threshold = 1.0f;
            }
            if (k_threshold <= 0.0f || !std::isfinite(k_threshold)) {
                k_threshold = 1.0f;
            }
            if (dynamic_qk_head) {
                q_threshold = 0.0f;
                k_threshold = 0.0f;
                for (int64_t iq = 0; iq < n_q; ++iq) {
                    for (int64_t id = 0; id < d_head; ++id) {
                        q_threshold = std::max(q_threshold, std::fabs(llama_tensor_get_f32_4d(q, id, iq, ih, ib)));
                    }
                }
                for (int64_t ik = 0; ik < n_k; ++ik) {
                    for (int64_t id = 0; id < d_head; ++id) {
                        k_threshold = std::max(k_threshold, std::fabs(llama_tensor_get_f32_4d(k, id, ik, ikh, ib)));
                    }
                }
                q_threshold = std::max(q_threshold, 1.0e-6f);
                k_threshold = std::max(k_threshold, 1.0e-6f);
            } else if (dynamic_qk_tile) {
                std::fill(q_threshold_dyn_tile.begin(), q_threshold_dyn_tile.end(), 0.0f);
                std::fill(k_threshold_dyn_tile.begin(), k_threshold_dyn_tile.end(), 0.0f);
                for (int64_t iq = 0; iq < n_q; ++iq) {
                    const size_t qt = (size_t) std::min<int64_t>(n_q_scale_tiles - 1, iq / dynamic_scale_tile);
                    for (int64_t id = 0; id < d_head; ++id) {
                        q_threshold_dyn_tile[qt] =
                            std::max(q_threshold_dyn_tile[qt], std::fabs(llama_tensor_get_f32_4d(q, id, iq, ih, ib)));
                    }
                }
                for (int64_t ik = 0; ik < n_k; ++ik) {
                    const size_t kt = (size_t) std::min<int64_t>(n_k_scale_tiles - 1, ik / dynamic_scale_tile);
                    for (int64_t id = 0; id < d_head; ++id) {
                        k_threshold_dyn_tile[kt] =
                            std::max(k_threshold_dyn_tile[kt], std::fabs(llama_tensor_get_f32_4d(k, id, ik, ikh, ib)));
                    }
                }
                for (float & threshold : q_threshold_dyn_tile) {
                    threshold = std::max(threshold, 1.0e-6f);
                }
                for (float & threshold : k_threshold_dyn_tile) {
                    threshold = std::max(threshold, 1.0e-6f);
                }
            }
            const auto q_threshold_for_iq = [&](int64_t iq) -> float {
                if (dynamic_qk_tile) {
                    return q_threshold_dyn_tile[(size_t) std::min<int64_t>(n_q_scale_tiles - 1, iq / dynamic_scale_tile)];
                }
                return q_threshold;
            };
            const auto k_threshold_for_ik = [&](int64_t ik) -> float {
                if (dynamic_qk_tile) {
                    return k_threshold_dyn_tile[(size_t) std::min<int64_t>(n_k_scale_tiles - 1, ik / dynamic_scale_tile)];
                }
                return k_threshold;
            };

            if (has_calib && !dynamic_v_head) {
                for (int64_t id = 0; id < d_head; ++id) {
                    const size_t offset = (size_t) (ikh * d_head + id);
                    if (cfg->v_centering) {
                        v_mean[(size_t) id] = cfg->calib->v_mean[offset];
                        v_threshold[(size_t) id] = std::max(cfg->calib->v_threshold[offset], 1.0e-6f);
                    } else {
                        v_mean[(size_t) id] = 0.0f;
                        v_threshold[(size_t) id] = std::max(
                                std::fabs(cfg->calib->v_mean[offset]) + cfg->calib->v_threshold[offset],
                                1.0e-6f);
                    }
                }
            } else {
                std::fill(v_mean.begin(), v_mean.end(), 0.0f);
                if (cfg->v_centering) {
                    for (int64_t ik = 0; ik < n_k; ++ik) {
                        for (int64_t id = 0; id < d_head; ++id) {
                            v_mean[(size_t) id] += llama_tensor_get_f32_4d(v, id, ik, ikh, ib);
                        }
                    }
                    const float inv_n_k = n_k > 0 ? 1.0f / (float) n_k : 0.0f;
                    for (int64_t id = 0; id < d_head; ++id) {
                        v_mean[(size_t) id] *= inv_n_k;
                    }
                }
                std::fill(v_threshold.begin(), v_threshold.end(), 0.0f);
                for (int64_t ik = 0; ik < n_k; ++ik) {
                    for (int64_t id = 0; id < d_head; ++id) {
                        const float residual = cfg->v_centering
                            ? llama_tensor_get_f32_4d(v, id, ik, ikh, ib) - v_mean[(size_t) id]
                            : llama_tensor_get_f32_4d(v, id, ik, ikh, ib);
                        v_threshold[(size_t) id] = std::max(v_threshold[(size_t) id], std::fabs(residual));
                    }
                }
                for (int64_t id = 0; id < d_head; ++id) {
                    if (v_threshold[(size_t) id] <= 0.0f || !std::isfinite(v_threshold[(size_t) id])) {
                        v_threshold[(size_t) id] = 1.0f;
                    }
                }
            }
            if (npu_host_profile_enabled) {
                npu_host_profile.threshold_us += ggml_time_us() - npu_profile_t0;
            }

            npu_profile_t0 = npu_host_profile_enabled ? ggml_time_us() : 0;
            for (int64_t iq = 0; iq < n_q; ++iq) {
                const float q_threshold_iq = q_threshold_for_iq(iq);
                for (int64_t id = 0; id < d_head; ++id) {
                    const float qv = llama_tensor_get_f32_4d(q, id, iq, ih, ib);
                    q8[(size_t) (iq * d_head + id)] = llama_attn_log8pv_quant_i8(qv, q_threshold_iq);
                    if (diag_enabled) {
                        const int8_t code = q8[(size_t) (iq * d_head + id)];
                        const double qhat = (double) code * (double) q_threshold_iq / 127.0;
                        if (code == 0 && qv != 0.0f) {
                            ++diag_q.zero_count;
                        }
                        if (std::fabs(qv) > q_threshold_iq) {
                            ++diag_q.clip_count;
                        }
                        diag_q.update(qv, qhat);
                    }
                }
            }
            if (npu_host_profile_enabled) {
                npu_host_profile.q_quant_us += ggml_time_us() - npu_profile_t0;
            }
            npu_profile_t0 = npu_host_profile_enabled ? ggml_time_us() : 0;
            for (int64_t ik = 0; ik < n_k; ++ik) {
                const float k_threshold_ik = k_threshold_for_ik(ik);
                for (int64_t id = 0; id < d_head; ++id) {
                    const float kv = llama_tensor_get_f32_4d(k, id, ik, ikh, ib);
                    k8[(size_t) (ik * d_head + id)] = llama_attn_log8pv_quant_i8(kv, k_threshold_ik);
                    vq[(size_t) (ik * d_head + id)] = llama_attn_log8pv_quant_v(
                            llama_tensor_get_f32_4d(v, id, ik, ikh, ib),
                            v_mean[(size_t) id],
                            v_threshold[(size_t) id],
                            v_absmax);
                    if (diag_enabled) {
                        const int8_t kcode = k8[(size_t) (ik * d_head + id)];
                        const double khat = (double) kcode * (double) k_threshold_ik / 127.0;
                        if (kcode == 0 && kv != 0.0f) {
                            ++diag_k.zero_count;
                        }
                        if (std::fabs(kv) > k_threshold_ik) {
                            ++diag_k.clip_count;
                        }
                        diag_k.update(kv, khat);

                        const float vv = llama_tensor_get_f32_4d(v, id, ik, ikh, ib);
                        const double vref = (double) vv - (double) v_mean[(size_t) id];
                        const int16_t vcode = vq[(size_t) (ik * d_head + id)];
                        const double vhat = (double) vcode * (double) v_threshold[(size_t) id] / (double) v_absmax;
                        if (vcode == 0 && vref != 0.0) {
                            ++diag_v.zero_count;
                        }
                        if (std::fabs(vref) > v_threshold[(size_t) id]) {
                            ++diag_v.clip_count;
                        }
                        diag_v.update(vref, vhat);
                    }
                }
            }
            if (npu_host_profile_enabled) {
                npu_host_profile.kv_quant_us += ggml_time_us() - npu_profile_t0;
            }

            const double qk_gain = has_calib && cfg->calib->qk_gain_head.size() == (size_t) n_head &&
                std::isfinite(cfg->calib->qk_gain_head[(size_t) ih]) &&
                cfg->calib->qk_gain_head[(size_t) ih] > 0.0f
                    ? (double) cfg->calib->qk_gain_head[(size_t) ih]
                    : 1.0;

#ifdef GGML_USE_NPU
            const bool text_log8pv_npu_requested = llama_text_prefill_log8pv_npu_enabled();
            uint32_t npu_q_row_start = 0;
            const bool text_log8pv_npu_causal_ok =
                llama_text_log8pv_infer_causal_q_start(mask, n_q, n_k, &npu_q_row_start);
            const bool text_log8pv_npu_shape_ok =
                cfg->collect == nullptr &&
                !diag_enabled &&
                !pv_diag_enabled &&
                has_calib &&
                !dynamic_qk_head &&
                !dynamic_qk_tile &&
                !dynamic_v_head &&
                !cfg->p_direct_u16 &&
                cfg->p_log2_range == 16.0f &&
                cfg->p_bits == 15 &&
                cfg->alpha_bits == 20 &&
                cfg->v_bits == 8 &&
                n_q > 0 &&
                n_k > 0 &&
                n_k <= 1024 &&
                npu_q_row_start + (uint32_t) n_q <= (uint32_t) n_k &&
                text_log8pv_npu_causal_ok &&
                (mask != nullptr || (n_k % 32) == 0);
            if (text_log8pv_npu_requested && !text_log8pv_npu_shape_ok &&
                    std::getenv("AICAS_TEXT_PREFILL_LOG8PV_NPU_DEBUG") != nullptr) {
                LLAMA_LOG_WARN(
                        "llama_compute_text_log8pv_attn: NPU rejected layer=%d head=%" PRId64
                        " n_q=%" PRId64 " n_k=%" PRId64 " has_calib=%d collect=%d diag=%d pv_diag=%d"
                        " dyn_qk_head=%d dyn_qk_tile=%d dyn_v_head=%d p_direct=%d p_log2_range=%.3f"
                        " p_bits=%d alpha_bits=%d v_bits=%d mask=%d q_start_ok=%d q_start=%u\n",
                        cfg->layer, ih, n_q, n_k, has_calib ? 1 : 0, cfg->collect != nullptr ? 1 : 0,
                        diag_enabled ? 1 : 0, pv_diag_enabled ? 1 : 0,
                        dynamic_qk_head ? 1 : 0, dynamic_qk_tile ? 1 : 0, dynamic_v_head ? 1 : 0,
                        cfg->p_direct_u16 ? 1 : 0, (double) cfg->p_log2_range,
                        cfg->p_bits, cfg->alpha_bits, cfg->v_bits, mask != nullptr ? 1 : 0,
                        text_log8pv_npu_causal_ok ? 1 : 0, npu_q_row_start);
            }
            if (text_log8pv_npu_requested && text_log8pv_npu_shape_ok) {
                npu_profile_t0 = npu_host_profile_enabled ? ggml_time_us() : 0;
                std::vector<int8_t> v8((size_t) n_k * d_head);
                for (int64_t ik = 0; ik < n_k; ++ik) {
                    for (int64_t id = 0; id < d_head; ++id) {
                        const int16_t value = vq[(size_t) ik * d_head + id];
                        v8[(size_t) ik * d_head + id] =
                            (int8_t) std::max<int16_t>(-128, std::min<int16_t>(127, value));
                    }
                }
                if (npu_host_profile_enabled) {
                    npu_host_profile.v8_cast_us += ggml_time_us() - npu_profile_t0;
                }
                npu_profile_t0 = npu_host_profile_enabled ? ggml_time_us() : 0;
                const double gamma2 =
                    (double) q_threshold / 127.0 *
                    (double) k_threshold / 127.0 *
                    (double) cfg->kq_scale *
                    qk_gain / std::log(2.0);
                const double gamma_fix_f = std::round(16.0 * gamma2 * (double) (UINT64_C(1) << 24));
                const uint32_t gamma16_fix = gamma_fix_f > 1.0 && gamma_fix_f < (double) std::numeric_limits<uint32_t>::max()
                    ? (uint32_t) gamma_fix_f
                    : 0u;
                if (npu_host_profile_enabled) {
                    npu_host_profile.gamma_us += ggml_time_us() - npu_profile_t0;
                }
                if (gamma16_fix != 0) {
                    std::vector<int32_t> npu_out((size_t) n_q * d_head);
                    const bool causal_mask = mask != nullptr;
                    ggml_backend_npu_log8pv_attention_profile npu_profile = {};
                    static std::atomic<bool> npu_logged{false};
                    if (!npu_logged.exchange(true)) {
                        LLAMA_LOG_INFO(
                                "llama_compute_text_log8pv_attn: trying Versa_P NPU text log8PV attention, "
                                "n_q=%" PRId64 " n_k=%" PRId64 " q_row_start=%u causal=%d\n",
                                n_q, n_k, npu_q_row_start, causal_mask ? 1 : 0);
                    }
                    npu_profile_t0 = npu_host_profile_enabled ? ggml_time_us() : 0;
                    const bool npu_ok = ggml_backend_npu_log8pv_attention_ex(
                            q8.data(),
                            (uint32_t) n_q,
                            k8.data(),
                            v8.data(),
                            (uint32_t) n_k,
                            npu_q_row_start,
                            causal_mask,
                            gamma16_fix,
                            npu_out.data(),
                            (uint32_t) d_head,
                            60000,
                            &npu_profile);
                    if (npu_host_profile_enabled) {
                        npu_host_profile.npu_call_wall_us += ggml_time_us() - npu_profile_t0;
                    }
                    if (npu_ok) {
                        npu_profile_t0 = npu_host_profile_enabled ? ggml_time_us() : 0;
                        for (int64_t iq = 0; iq < n_q; ++iq) {
                            for (int64_t id = 0; id < d_head; ++id) {
                                const float out = (float) (
                                        (double) npu_out[(size_t) iq * d_head + id] *
                                        ((double) v_threshold[(size_t) id] / (double) v_absmax) +
                                        (double) v_mean[(size_t) id]);
                                llama_tensor_set_f32_4d(dst, id, ih, iq, ib, out);
                            }
                        }
                        if (npu_host_profile_enabled) {
                            npu_host_profile.dequant_store_us += ggml_time_us() - npu_profile_t0;
                        }
                        llama_log8pv_npu_profile_append(
                                "text", cfg->layer, ih, ikh, n_q, n_k, npu_q_row_start, npu_profile, npu_host_profile);
                        continue;
                    }
                    static std::atomic<bool> npu_fallback_logged{false};
                    if (!npu_fallback_logged.exchange(true)) {
                        LLAMA_LOG_WARN(
                                "llama_compute_text_log8pv_attn: Versa_P NPU text log8PV attention failed or rejected; "
                                "falling back to CPU reference\n");
                    }
                }
            }
#endif

            for (int64_t iq = 0; iq < n_q; ++iq) {
                int32_t m_global = invalid_score;
                double row_l_exact = 0.0;
                double row_exact_mass = 0.0;
                double row_zero_mass = 0.0;
                for (int64_t ik0 = 0; ik0 < n_k; ik0 += tile) {
                    const int64_t ik1 = std::min<int64_t>(n_k, ik0 + tile);
                    const size_t tile_idx = (size_t) (ik0 / tile);
                    int32_t mt = invalid_score;
                    for (int64_t ik = ik0; ik < ik1; ++ik) {
                        int64_t dot = 0;
                        double ref_dot = 0.0;
                        for (int64_t id = 0; id < d_head; ++id) {
                            dot += (int32_t) q8[(size_t) (iq * d_head + id)] *
                                (int32_t) k8[(size_t) (ik * d_head + id)];
                            if (diag_enabled) {
                                ref_dot += (double) llama_tensor_get_f32_4d(q, id, iq, ih, ib) *
                                    (double) llama_tensor_get_f32_4d(k, id, ik, ikh, ib);
                            }
                        }
                        const float mask_value = llama_attn_mask_get_f32(mask, ik, iq);
                        if (!std::isfinite(mask_value) || mask_value < -1.0e20f) {
                            score_s16[(size_t) ik] = invalid_score;
                            if (diag_enabled) {
                                ref_logits[(size_t) ik] = -std::numeric_limits<double>::infinity();
                            }
                            continue;
                        }
                        if (diag_enabled) {
                            ref_logits[(size_t) ik] = ref_dot * (double) cfg->kq_scale;
                        }
                        const double gamma2 =
                            (double) q_threshold_for_iq(iq) / 127.0 *
                            (double) k_threshold_for_ik(ik) / 127.0 *
                            (double) cfg->kq_scale *
                            qk_gain / std::log(2.0);
                        const int64_t sq = (int64_t) std::llround(16.0 * (double) dot * gamma2);
                        score_s16[(size_t) ik] = (int32_t) std::max<int64_t>(
                                std::numeric_limits<int32_t>::min() + 1,
                                std::min<int64_t>(std::numeric_limits<int32_t>::max(), sq));
                        mt = std::max(mt, score_s16[(size_t) ik]);
                        if (diag_enabled) {
                            diag_qk.update(ref_logits[(size_t) ik] / std::log(2.0),
                                    (double) score_s16[(size_t) ik] / 16.0);
                        }
                    }
                    m_tile[tile_idx] = mt;
                    m_global = std::max(m_global, mt);
                    for (int64_t ik = ik0; ik < ik1; ++ik) {
                        if (score_s16[(size_t) ik] == invalid_score || mt == invalid_score) {
                            q_local[(size_t) ik] = 255;
                            continue;
                        }
                        const int64_t d_local = (int64_t) mt - (int64_t) score_s16[(size_t) ik];
                        q_local[(size_t) ik] = d_local >= 255 ? 255 : (uint8_t) std::max<int64_t>(0, d_local);
                    }
                }

                std::fill(accum.begin(), accum.end(), 0);
                int64_t row_l = 0;
                for (int64_t ik0 = 0; ik0 < n_k; ik0 += tile) {
                    const int64_t ik1 = std::min<int64_t>(n_k, ik0 + tile);
                    const int32_t mt = m_tile[(size_t) (ik0 / tile)];
                    const int64_t delta = (m_global == invalid_score || mt == invalid_score)
                        ? 255
                        : (int64_t) m_global - (int64_t) mt;
                    for (int64_t ik = ik0; ik < ik1; ++ik) {
                        const int64_t qg = (int64_t) q_local[(size_t) ik] + delta;
                        const uint16_t p = q_local[(size_t) ik] != 255 && qg < 255
                            ? llama_attn_log8pv_base2_p((uint32_t) qg, cfg->p_bits)
                            : 0;
                        row_l += (int64_t) p;
                        if (diag_enabled && score_s16[(size_t) ik] != invalid_score && m_global != invalid_score) {
                            const double p_exact = std::exp2(((double) score_s16[(size_t) ik] - (double) m_global) / 16.0);
                            const double p_hat = (double) p / (double) p_max;
                            row_l_exact += p_exact * (double) p_max;
                            row_exact_mass += p_exact;
                            if (p == 0) {
                                row_zero_mass += p_exact;
                                ++diag_p.zero_count;
                            }
                            diag_p.update(p_exact, p_hat);
                        }
                        for (int64_t id = 0; id < d_head; ++id) {
                            const int64_t product = (int64_t) p * (int64_t) vq[(size_t) (ik * d_head + id)];
                            accum[(size_t) id] += product;
                            if (pv_diag_enabled) {
                                diag_pv_max_abs_product = std::max<int64_t>(diag_pv_max_abs_product, std::llabs(product));
                                diag_pv_max_abs_accum = std::max<int64_t>(diag_pv_max_abs_accum, std::llabs(accum[(size_t) id]));
                                diag_pv_accum_over_i32 += std::llabs(accum[(size_t) id]) > (int64_t) std::numeric_limits<int32_t>::max();
                                ++diag_pv_accum_updates;
                            }
                        }
                    }
                }
                if (pv_diag_enabled) {
                    diag_pv_max_row_l = std::max<int64_t>(diag_pv_max_row_l, row_l);
                    for (int64_t id = 0; id < d_head; ++id) {
                        diag_pv_max_abs_final_accum = std::max<int64_t>(diag_pv_max_abs_final_accum, std::llabs(accum[(size_t) id]));
                        diag_pv_final_over_i32 += std::llabs(accum[(size_t) id]) > (int64_t) std::numeric_limits<int32_t>::max();
                    }
                }
                if (diag_enabled) {
                    const double l_rel = row_l_exact > 0.0 ? std::fabs((double) row_l - row_l_exact) / row_l_exact : 0.0;
                    const double zero_mass_frac = row_exact_mass > 0.0 ? row_zero_mass / row_exact_mass : 0.0;
                    ++diag_rows.rows;
                    diag_rows.sum_l_rel_err += l_rel;
                    diag_rows.max_l_rel_err = std::max(diag_rows.max_l_rel_err, l_rel);
                    diag_rows.sum_p_zero_mass_frac += zero_mass_frac;
                    diag_rows.max_p_zero_mass_frac = std::max(diag_rows.max_p_zero_mass_frac, zero_mass_frac);

                    double ref_m = -std::numeric_limits<double>::infinity();
                    for (int64_t ik = 0; ik < n_k; ++ik) {
                        ref_m = std::max(ref_m, ref_logits[(size_t) ik]);
                    }
                    double ref_l = 0.0;
                    std::fill(ref_out.begin(), ref_out.end(), 0.0);
                    if (std::isfinite(ref_m)) {
                        for (int64_t ik = 0; ik < n_k; ++ik) {
                            if (!std::isfinite(ref_logits[(size_t) ik])) {
                                continue;
                            }
                            const double p = std::exp(ref_logits[(size_t) ik] - ref_m);
                            ref_l += p;
                            for (int64_t id = 0; id < d_head; ++id) {
                                ref_out[(size_t) id] += p * (double) llama_tensor_get_f32_4d(v, id, ik, ikh, ib);
                            }
                        }
                    }
                    if (ref_l > 0.0) {
                        for (int64_t id = 0; id < d_head; ++id) {
                            ref_out[(size_t) id] /= ref_l;
                        }
                    }
                }
                for (int64_t id = 0; id < d_head; ++id) {
                    const float out = row_l > 0
                        ? (float) (((double) accum[(size_t) id] / (double) row_l) *
                                ((double) v_threshold[(size_t) id] / (double) v_absmax) +
                                (double) v_mean[(size_t) id])
                        : 0.0f;
                    llama_tensor_set_f32_4d(dst, id, ih, iq, ib, out);
                    if (diag_enabled) {
                        diag_o.update(ref_out[(size_t) id], out);
                    }
                }
            }
        }
    }

    if (diag_enabled) {
        json out = {
            {"schema", "aicas.text_prefill.log8pv_diag.v1"},
            {"layer", cfg->layer},
            {"n_batch", n_batch},
            {"n_head", n_head},
            {"n_kv_head", n_kv_head},
            {"n_q", n_q},
            {"n_k", n_k},
            {"d_head", d_head},
            {"tile", tile},
            {"qk_scale_mode", llama_text_log8pv_qk_scale_mode_name(cfg->qk_scale_mode)},
            {"qk_scale_tile", dynamic_scale_tile},
            {"v_scale_mode", llama_text_log8pv_v_scale_mode_name(cfg->v_scale_mode)},
            {"has_calib", has_calib},
            {"q", diag_q.to_json()},
            {"k", diag_k.to_json()},
            {"v_centered", diag_v.to_json()},
            {"qk_logits", diag_qk.to_json()},
            {"p_log8", diag_p.to_json()},
            {"rows", diag_rows.to_json()},
            {"pv_accum", {
                {"max_abs_product", diag_pv_max_abs_product},
                {"max_abs_accum", diag_pv_max_abs_accum},
                {"max_abs_final_accum", diag_pv_max_abs_final_accum},
                {"max_row_l", diag_pv_max_row_l},
                {"accum_over_i32_count", diag_pv_accum_over_i32},
                {"final_over_i32_count", diag_pv_final_over_i32},
                {"accum_update_count", diag_pv_accum_updates},
            }},
            {"output", diag_o.to_json()},
        };
        std::ofstream fout(diag_path, std::ios::app | std::ios::binary);
        if (fout.is_open()) {
            fout << out.dump() << "\n";
        }
    }
    if (!pv_diag_path.empty()) {
        json out = {
            {"schema", "aicas.text_prefill.log8pv_pv_diag.v1"},
            {"layer", cfg->layer},
            {"n_batch", n_batch},
            {"n_head", n_head},
            {"n_kv_head", n_kv_head},
            {"n_q", n_q},
            {"n_k", n_k},
            {"d_head", d_head},
            {"tile", tile},
            {"qk_scale_mode", llama_text_log8pv_qk_scale_mode_name(cfg->qk_scale_mode)},
            {"qk_scale_tile", dynamic_scale_tile},
            {"v_scale_mode", llama_text_log8pv_v_scale_mode_name(cfg->v_scale_mode)},
            {"has_calib", has_calib},
            {"pv_accum", {
                {"max_abs_product", diag_pv_max_abs_product},
                {"max_abs_accum", diag_pv_max_abs_accum},
                {"max_abs_final_accum", diag_pv_max_abs_final_accum},
                {"max_row_l", diag_pv_max_row_l},
                {"accum_over_i32_count", diag_pv_accum_over_i32},
                {"final_over_i32_count", diag_pv_final_over_i32},
                {"accum_update_count", diag_pv_accum_updates},
            }},
        };
        std::ofstream fout(pv_diag_path, std::ios::app | std::ios::binary);
        if (fout.is_open()) {
            fout << out.dump() << "\n";
        }
    }
}

static ggml_tensor * llama_build_text_log8pv_attn(
        ggml_context * ctx0,
        ggml_tensor * q,
        ggml_tensor * k,
        ggml_tensor * v,
        const ggml_tensor * mask,
        float kq_scale,
        int il) {
    GGML_ASSERT(q->ne[0] == 64);
    GGML_ASSERT(k->ne[0] == q->ne[0]);
    GGML_ASSERT(v->ne[0] == q->ne[0]);
    GGML_ASSERT(q->ne[2] % k->ne[2] == 0);

    static std::mutex userdata_mutex;
    static std::vector<std::unique_ptr<llama_text_log8pv_attn_userdata>> userdata_store;
    auto node_cfg = std::make_unique<llama_text_log8pv_attn_userdata>();
    node_cfg->mask = mask;
    node_cfg->kq_scale = kq_scale;
    node_cfg->layer = il;
    node_cfg->calib = llama_text_log8pv_calib_state().get_layer(il);
    node_cfg->collect = llama_text_log8pv_calib_state().get_collect_layer(il);
    node_cfg->tile = llama_text_prefill_log8pv_tile();
    node_cfg->p_log2_range = llama_text_prefill_log8pv_p_log2_range();
    node_cfg->p_direct_u16 = llama_text_prefill_log8pv_p_direct_u16();
    node_cfg->p_bits = llama_text_prefill_log8pv_p_bits();
    node_cfg->alpha_bits = llama_text_prefill_log8pv_alpha_bits();
    node_cfg->v_bits = llama_text_prefill_log8pv_v_bits();
    node_cfg->bit_accurate = llama_text_prefill_log8pv_bit_accurate_enabled();
    node_cfg->v_centering = llama_text_prefill_log8pv_v_centering_enabled();
    if (const char * value = std::getenv("AICAS_TEXT_PREFILL_LOG8PV_QK_SCALE_MODE")) {
        if (std::strcmp(value, "dynamic_perhead") == 0 || std::strcmp(value, "dynamic-perhead") == 0 ||
                std::strcmp(value, "dynamic_head") == 0 || std::strcmp(value, "dynamic-head") == 0) {
            node_cfg->qk_scale_mode = 1;
        } else if (std::strcmp(value, "dynamic_pertile") == 0 || std::strcmp(value, "dynamic-pertile") == 0 ||
                std::strcmp(value, "dynamic_tile") == 0 || std::strcmp(value, "dynamic-tile") == 0) {
            node_cfg->qk_scale_mode = 2;
        } else {
            node_cfg->qk_scale_mode = 0;
        }
    }
    if (const char * value = std::getenv("AICAS_TEXT_PREFILL_LOG8PV_QK_SCALE_TILE")) {
        const long parsed = std::strtol(value, nullptr, 10);
        if (parsed > 0 && parsed <= std::numeric_limits<int16_t>::max()) {
            node_cfg->qk_scale_tile = parsed;
        }
    }
    if (const char * value = std::getenv("AICAS_TEXT_PREFILL_LOG8PV_V_SCALE_MODE")) {
        if (std::strcmp(value, "dynamic_perhead") == 0 || std::strcmp(value, "dynamic-perhead") == 0 ||
                std::strcmp(value, "dynamic_head") == 0 || std::strcmp(value, "dynamic-head") == 0) {
            node_cfg->v_scale_mode = 1;
        } else {
            node_cfg->v_scale_mode = 0;
        }
    }
    llama_text_log8pv_attn_userdata * ptr = node_cfg.get();
    {
        std::lock_guard<std::mutex> lock(userdata_mutex);
        userdata_store.push_back(std::move(node_cfg));
    }

    ggml_tensor * args[] = { q, k, v, const_cast<ggml_tensor *>(mask) };
    return ggml_custom_4d(
            ctx0,
            GGML_TYPE_F32,
            q->ne[0],
            q->ne[2],
            q->ne[1],
            q->ne[3],
            args,
            mask != nullptr ? 4 : 3,
            llama_compute_text_log8pv_attn,
            1,
            ptr);
}

static void llama_compute_text_w8a8_mul_mat(
        struct ggml_tensor * dst,
        const struct ggml_tensor * a,
        const struct ggml_tensor * b,
        const struct ggml_tensor * c,
        int ith,
        int nth,
        void * userdata) {
    GGML_UNUSED(a);

    const auto * cfg = static_cast<const llama_aicas_text_sq_tensor *>(userdata);
    GGML_ASSERT(cfg != nullptr);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(b->type == GGML_TYPE_F32);
    GGML_ASSERT(c->type == GGML_TYPE_I8);

    const int64_t k = c->ne[0];
    const int64_t out_channels = c->ne[1];
    const int64_t n_cols = b->ne[1];
    GGML_ASSERT(b->ne[0] == k);
    GGML_ASSERT(cfg->smooth_scale.size() == static_cast<size_t>(k));
    GGML_ASSERT(cfg->weight_scale.size() == cfg->expected_weight_scale_len(out_channels));

    const int64_t cols_per_thread = (n_cols + nth - 1) / nth;
    const int64_t col_begin = ith * cols_per_thread;
    const int64_t col_end = std::min(n_cols, col_begin + cols_per_thread);
    if (col_begin >= col_end) {
        return;
    }

    const float sa = cfg->act_scale;
    const int32_t za = cfg->act_zero_point;
    std::vector<int8_t> act_i8(k);

    for (int64_t col = col_begin; col < col_end; ++col) {
        const float * act_col = (const float *) ((const char *) b->data + col * b->nb[1]);
        float * out_col = (float *) ((char *) dst->data + col * dst->nb[1]);

        for (int64_t i = 0; i < k; ++i) {
            const float act_value = act_col[i] / cfg->smooth_scale[static_cast<size_t>(i)];
            int32_t q = (int32_t) lrintf(act_value / sa) + za;
            q = std::max(0, std::min(255, q));
            act_i8[i] = (int8_t) (q - 128);
        }

        for (int64_t j = 0; j < out_channels; ++j) {
            const int8_t * w_row = (const int8_t *) ((const char *) c->data + j * c->nb[1]);
            int32_t acc = 0;
            for (int64_t i = 0; i < k; ++i) {
                acc += (int32_t) act_i8[i] * (int32_t) w_row[i];
            }
            const int32_t correction = cfg->sum_w.empty()
                ? 0
                : (cfg->act_zero_point - 128) * cfg->sum_w[static_cast<size_t>(j)];
            const float sw = cfg->uses_per_tensor_weight_scale()
                ? cfg->weight_scale[0]
                : cfg->weight_scale[static_cast<size_t>(j)];
            out_col[j] = (float) (acc - correction) * (sa * sw);
        }
    }
}

static uint8_t llama_decode_awq_q4(
        const uint8_t packed,
        const int64_t idx_in_row) {
    return (idx_in_row & 1) == 0
        ? (packed & 0x0fu)
        : ((packed >> 4) & 0x0fu);
}

static float llama_decode_awq_read_scale(
        const struct ggml_tensor * scale_tensor,
        int64_t row,
        int64_t group) {
    const char * ptr = (const char *) scale_tensor->data + row * scale_tensor->nb[1] + group * scale_tensor->nb[0];
    switch (scale_tensor->type) {
        case GGML_TYPE_F32:
            return *(const float *) ptr;
        case GGML_TYPE_F16:
            return ggml_fp16_to_fp32(*(const ggml_fp16_t *) ptr);
        default:
            GGML_ABORT("unsupported AICAS text decode AWQ scale tensor type");
    }
}

static float llama_decode_awq_read_act(
        const struct ggml_tensor * act_tensor,
        const char * act_col,
        int64_t i) {
    switch (act_tensor->type) {
        case GGML_TYPE_F32:
            return ((const float *) act_col)[i];
        case GGML_TYPE_F16:
            return ggml_fp16_to_fp32(((const ggml_fp16_t *) act_col)[i]);
        default:
            GGML_ABORT("unsupported AICAS text decode AWQ activation tensor type");
    }
}

static int32_t llama_decode_awq_quant_int24(
        double value,
        int frac_bits,
        uint64_t & clip_count) {
    constexpr int32_t q_min = -(1 << 23);
    constexpr int32_t q_max =  (1 << 23) - 1;
    if (!std::isfinite(value)) {
        ++clip_count;
        return value < 0.0 ? q_min : q_max;
    }
    const double scaled = std::ldexp(value, frac_bits);
    long long q = std::llround(scaled);
    if (q < q_min) {
        ++clip_count;
        return q_min;
    }
    if (q > q_max) {
        ++clip_count;
        return q_max;
    }
    return (int32_t) q;
}

static void llama_decode_awq_int24_write_diag(
        const llama_aicas_text_decode_awq_tensor * cfg,
        int ith,
        int frac_bits,
        bool accum_fp16,
        bool output_int24,
        const llama_decode_awq_int24_thread_stats & stats) {
    const std::string path = llama_text_decode_awq_int24_diag_path();
    if (path.empty() || stats.count == 0) {
        return;
    }

    static std::mutex diag_mutex;
    static size_t sample_count = 0;

    std::lock_guard<std::mutex> lock(diag_mutex);
    std::ofstream fout(path, std::ios::app);
    if (!fout) {
        return;
    }

    const double rmse = std::sqrt(stats.sum_sq_err / (double) stats.count);
    const double mae = stats.sum_abs_err / (double) stats.count;
    const double nrmse = stats.sum_ref_sq > 0.0 ?
        std::sqrt(stats.sum_sq_err / stats.sum_ref_sq) : 0.0;
    json summary = {
        {"schema", "aicas.text_decode_awq_int24_diag.v1"},
        {"kind", "summary"},
        {"tensor", cfg != nullptr ? cfg->quant_tensor_name : std::string{}},
        {"quant_tensor", cfg != nullptr ? cfg->quant_tensor_name : std::string{}},
        {"thread", ith},
        {"frac_bits", frac_bits},
        {"accum", accum_fp16 ? "fp16" : "fp32"},
        {"output_int24", output_int24},
        {"count", stats.count},
        {"scalar_count", stats.scalar_count},
        {"sum_abs_scaled_act", stats.sum_abs_scaled_act},
        {"sum_sq_scaled_act", stats.sum_sq_scaled_act},
        {"scaled_act_abs_mean", stats.scalar_count > 0 ? stats.sum_abs_scaled_act / (double) stats.scalar_count : 0.0},
        {"scaled_act_rms", stats.scalar_count > 0 ? std::sqrt(stats.sum_sq_scaled_act / (double) stats.scalar_count) : 0.0},
        {"scaled_act_abs_max", stats.max_abs_scaled_act},
        {"sum_abs_scalar_err", stats.sum_abs_scalar_err},
        {"sum_sq_scalar_err", stats.sum_sq_scalar_err},
        {"scaled_act_q24_mae", stats.scalar_count > 0 ? stats.sum_abs_scalar_err / (double) stats.scalar_count : 0.0},
        {"scaled_act_q24_rmse", stats.scalar_count > 0 ? std::sqrt(stats.sum_sq_scalar_err / (double) stats.scalar_count) : 0.0},
        {"scaled_act_q24_max_abs_err", stats.max_abs_scalar_err},
        {"sum_abs_err", stats.sum_abs_err},
        {"sum_sq_err", stats.sum_sq_err},
        {"sum_ref_sq", stats.sum_ref_sq},
        {"mae", mae},
        {"rmse", rmse},
        {"nrmse", nrmse},
        {"max_abs_err", stats.max_abs_err},
        {"max_abs_ref", stats.max_abs_ref},
        {"max_abs_int24", stats.max_abs_int24},
        {"max_abs_q24", stats.max_abs_q24},
        {"clip_count", stats.clip_count},
        {"non_integer_zero_count", stats.non_integer_zero_count},
        {"hist_log2_abs_min_exp", llama_decode_awq_int24_thread_stats::hist_log2_min_exp},
        {"hist_log2_abs_max_exp", llama_decode_awq_int24_thread_stats::hist_log2_max_exp},
        {"hist_log2_abs_layout", "zero,lt_min_exp,floor_log2_min_to_max,gt_max_exp"},
        {"scaled_act_abs_hist", stats.scaled_act_abs_hist},
        {"scalar_err_abs_hist", stats.scalar_err_abs_hist},
        {"output_err_abs_hist", stats.output_err_abs_hist},
        {"ref_abs_hist", stats.ref_abs_hist},
    };
    fout << summary.dump() << "\n";

    const size_t dump_limit = llama_text_decode_awq_int24_dump_limit();
    for (const auto & sample : stats.samples) {
        if (sample_count >= dump_limit) {
            break;
        }
        json item = {
            {"schema", "aicas.text_decode_awq_int24_diag.v1"},
            {"kind", "sample"},
            {"tensor", cfg != nullptr ? cfg->quant_tensor_name : std::string{}},
            {"thread", ith},
            {"frac_bits", frac_bits},
            {"col", sample.col},
            {"row", sample.row},
            {"ref", sample.ref},
            {"int24", sample.int24},
            {"err", sample.err},
        };
        fout << item.dump() << "\n";
        ++sample_count;
    }
}

static void llama_lm_head_w8a16_int24_write_diag(
        int ith,
        int frac_bits,
        int64_t tile,
        const llama_decode_awq_int24_thread_stats & stats) {
    const std::string path = llama_text_lm_head_w8a16_int24_diag_path();
    if (path.empty() || stats.count == 0) {
        return;
    }

    static std::mutex diag_mutex;
    static size_t sample_count = 0;

    std::lock_guard<std::mutex> lock(diag_mutex);
    std::ofstream fout(path, std::ios::app);
    if (!fout) {
        return;
    }

    const double rmse = std::sqrt(stats.sum_sq_err / (double) stats.count);
    const double mae = stats.sum_abs_err / (double) stats.count;
    const double nrmse = stats.sum_ref_sq > 0.0 ?
        std::sqrt(stats.sum_sq_err / stats.sum_ref_sq) : 0.0;
    json summary = {
        {"schema", "aicas.text_lm_head_w8a16_int24_diag.v1"},
        {"kind", "summary"},
        {"tensor", "output.weight"},
        {"thread", ith},
        {"frac_bits", frac_bits},
        {"tile", tile},
        {"count", stats.count},
        {"scalar_count", stats.scalar_count},
        {"sum_abs_scaled_act", stats.sum_abs_scaled_act},
        {"sum_sq_scaled_act", stats.sum_sq_scaled_act},
        {"scaled_act_abs_mean", stats.scalar_count > 0 ? stats.sum_abs_scaled_act / (double) stats.scalar_count : 0.0},
        {"scaled_act_rms", stats.scalar_count > 0 ? std::sqrt(stats.sum_sq_scaled_act / (double) stats.scalar_count) : 0.0},
        {"scaled_act_abs_max", stats.max_abs_scaled_act},
        {"sum_abs_scalar_err", stats.sum_abs_scalar_err},
        {"sum_sq_scalar_err", stats.sum_sq_scalar_err},
        {"scaled_act_q24_mae", stats.scalar_count > 0 ? stats.sum_abs_scalar_err / (double) stats.scalar_count : 0.0},
        {"scaled_act_q24_rmse", stats.scalar_count > 0 ? std::sqrt(stats.sum_sq_scalar_err / (double) stats.scalar_count) : 0.0},
        {"scaled_act_q24_max_abs_err", stats.max_abs_scalar_err},
        {"sum_abs_err", stats.sum_abs_err},
        {"sum_sq_err", stats.sum_sq_err},
        {"sum_ref_sq", stats.sum_ref_sq},
        {"mae", mae},
        {"rmse", rmse},
        {"nrmse", nrmse},
        {"max_abs_err", stats.max_abs_err},
        {"max_abs_ref", stats.max_abs_ref},
        {"max_abs_int24", stats.max_abs_int24},
        {"max_abs_q24", stats.max_abs_q24},
        {"clip_count", stats.clip_count},
        {"hist_log2_abs_min_exp", llama_decode_awq_int24_thread_stats::hist_log2_min_exp},
        {"hist_log2_abs_max_exp", llama_decode_awq_int24_thread_stats::hist_log2_max_exp},
        {"hist_log2_abs_layout", "zero,lt_min_exp,floor_log2_min_to_max,gt_max_exp"},
        {"scaled_act_abs_hist", stats.scaled_act_abs_hist},
        {"scalar_err_abs_hist", stats.scalar_err_abs_hist},
        {"output_err_abs_hist", stats.output_err_abs_hist},
        {"ref_abs_hist", stats.ref_abs_hist},
    };
    fout << summary.dump() << "\n";

    const size_t dump_limit = llama_text_decode_awq_int24_dump_limit();
    for (const auto & sample : stats.samples) {
        if (sample_count >= dump_limit) {
            break;
        }
        json item = {
            {"schema", "aicas.text_lm_head_w8a16_int24_diag.v1"},
            {"kind", "sample"},
            {"thread", ith},
            {"frac_bits", frac_bits},
            {"col", sample.col},
            {"row", sample.row},
            {"ref", sample.ref},
            {"int24", sample.int24},
            {"err", sample.err},
        };
        fout << item.dump() << "\n";
        ++sample_count;
    }
}

struct llama_text_lm_head_w16a16_userdata {
    int64_t tile = 128;
};

struct llama_text_lm_head_w8a16_userdata {
    std::mutex mutex;
    const void * weight_data = nullptr;
    int64_t tile = 128;
    int64_t cached_tile = 0;
    int64_t k = 0;
    int64_t out_channels = 0;
    int64_t n_tiles = 0;
    std::vector<int8_t> qweight;
    std::vector<uint16_t> scales;
};

struct llama_text_token_embd_w8a16_userdata {
    std::mutex mutex;
    const void * weight_data = nullptr;
    int64_t group = 64;
    int64_t cached_group = 0;
    int64_t n_embd = 0;
    int64_t vocab = 0;
    int64_t n_groups = 0;
    std::vector<float> decoded_rows;
};

static llama_text_token_embd_w8a16_userdata & llama_text_token_embd_w8a16_cache() {
    static llama_text_token_embd_w8a16_userdata cache;
    return cache;
}

static void llama_compute_text_lm_head_w16a16_dp128_mul_mat(
        struct ggml_tensor * dst,
        const struct ggml_tensor * out_template,
        const struct ggml_tensor * act,
        const struct ggml_tensor * weight,
        int ith,
        int nth,
        void * userdata) {
    GGML_UNUSED(out_template);

    const auto * cfg = static_cast<const llama_text_lm_head_w16a16_userdata *>(userdata);
    GGML_ASSERT(cfg != nullptr);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(act->type == GGML_TYPE_F16);
    GGML_ASSERT(weight->type == GGML_TYPE_F16);
    GGML_ASSERT(weight->ne[0] == act->ne[0]);
    GGML_ASSERT(dst->ne[0] == weight->ne[1]);
    GGML_ASSERT(dst->ne[1] == act->ne[1]);

    const int64_t k = weight->ne[0];
    const int64_t out_channels = weight->ne[1];
    const int64_t n_cols = act->ne[1];
    const int64_t tile = std::max<int64_t>(1, cfg->tile);
    GGML_ASSERT(tile == 128);
    const int64_t total_tasks = n_cols * out_channels;
    const int64_t tasks_per_thread = (total_tasks + nth - 1) / nth;
    const int64_t task_begin = ith * tasks_per_thread;
    const int64_t task_end = std::min(total_tasks, task_begin + tasks_per_thread);
    if (task_begin >= task_end) {
        return;
    }

    for (int64_t task = task_begin; task < task_end; ++task) {
        const int64_t col = task / out_channels;
        const int64_t row = task % out_channels;
        const auto * act_col = (const ggml_fp16_t *) ((const char *) act->data + col * act->nb[1]);
        const auto * w_row = (const ggml_fp16_t *) ((const char *) weight->data + row * weight->nb[1]);
        float * out_col = (float *) ((char *) dst->data + col * dst->nb[1]);

        uint16_t acc = 0;
        bool have_acc = false;
        for (int64_t start = 0; start < k; start += tile) {
            const int64_t end = std::min(k, start + tile);
            const int valid = (int) (end - start);
            const uint16_t tile_sum = aicas_rtl_dp128_tile(
                    [&](int lane) {
                        return aicas_rtl_fp16_zero_if_non_normal(aicas_rtl_fp16_bits(w_row[start + lane]));
                    },
                    [&](int lane) {
                        return aicas_rtl_fp16_zero_if_non_normal(aicas_rtl_fp16_bits(act_col[start + lane]));
                    },
                    valid);
            acc = have_acc ? aicas_rtl_fp16_add(acc, tile_sum) : tile_sum;
            have_acc = true;
        }

        out_col[row] = aicas_rtl_fp16_to_f32(acc);
    }
}

static void llama_lm_head_w8a16_prepare_cache(
        llama_text_lm_head_w8a16_userdata * cfg,
        const struct ggml_tensor * weight) {
    GGML_ASSERT(cfg != nullptr);
    GGML_ASSERT(weight->type == GGML_TYPE_F16);

    const int64_t tile = std::max<int64_t>(1, std::min<int64_t>(128, cfg->tile));
    const int64_t k = weight->ne[0];
    const int64_t out_channels = weight->ne[1];
    const int64_t n_tiles = (k + tile - 1) / tile;

    std::lock_guard<std::mutex> lock(cfg->mutex);
    if (cfg->weight_data == weight->data &&
        cfg->cached_tile == tile &&
        cfg->k == k &&
        cfg->out_channels == out_channels &&
        cfg->n_tiles == n_tiles &&
        !cfg->qweight.empty()) {
        return;
    }

    cfg->weight_data = weight->data;
    cfg->cached_tile = tile;
    cfg->k = k;
    cfg->out_channels = out_channels;
    cfg->n_tiles = n_tiles;
    cfg->qweight.assign((size_t) out_channels * (size_t) n_tiles * (size_t) tile, 0);
    cfg->scales.assign((size_t) out_channels * (size_t) n_tiles, 0);

    for (int64_t row = 0; row < out_channels; ++row) {
        const auto * w_row = (const ggml_fp16_t *) ((const char *) weight->data + row * weight->nb[1]);
        for (int64_t t = 0; t < n_tiles; ++t) {
            const int64_t start = t * tile;
            const int64_t end = std::min<int64_t>(k, start + tile);
            float amax = 0.0f;
            for (int64_t i = start; i < end; ++i) {
                const uint16_t bits = aicas_rtl_fp16_zero_if_non_normal(aicas_rtl_fp16_bits(w_row[i]));
                amax = std::max(amax, std::fabs(aicas_rtl_fp16_to_f32(bits)));
            }

            const float d = amax / 127.0f;
            const float id = d != 0.0f ? 1.0f / d : 0.0f;
            cfg->scales[(size_t) row * (size_t) n_tiles + (size_t) t] =
                aicas_rtl_fp16_bits(ggml_fp32_to_fp16(d));

            int8_t * q_tile = cfg->qweight.data() +
                ((size_t) row * (size_t) n_tiles + (size_t) t) * (size_t) tile;
            for (int64_t lane = 0; lane < end - start; ++lane) {
                const uint16_t bits = aicas_rtl_fp16_zero_if_non_normal(aicas_rtl_fp16_bits(w_row[start + lane]));
                const float value = aicas_rtl_fp16_to_f32(bits);
                int q = (int) std::round(value * id);
                q = std::max(-127, std::min(127, q));
                q_tile[lane] = (int8_t) q;
            }
        }
    }
}

static void llama_compute_text_lm_head_w8a16_dp128_mul_mat(
        struct ggml_tensor * dst,
        const struct ggml_tensor * out_template,
        const struct ggml_tensor * act,
        const struct ggml_tensor * weight,
        int ith,
        int nth,
        void * userdata) {
    GGML_UNUSED(out_template);

    auto * cfg = static_cast<llama_text_lm_head_w8a16_userdata *>(userdata);
    GGML_ASSERT(cfg != nullptr);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(act->type == GGML_TYPE_F16);
    GGML_ASSERT(weight->type == GGML_TYPE_F16);
    GGML_ASSERT(weight->ne[0] == act->ne[0]);
    GGML_ASSERT(dst->ne[0] == weight->ne[1]);
    GGML_ASSERT(dst->ne[1] == act->ne[1]);

    llama_lm_head_w8a16_prepare_cache(cfg, weight);

    const int64_t k = weight->ne[0];
    const int64_t out_channels = weight->ne[1];
    const int64_t n_cols = act->ne[1];
    const int64_t tile = std::max<int64_t>(1, std::min<int64_t>(128, cfg->tile));
    const int64_t n_tiles = (k + tile - 1) / tile;
    GGML_ASSERT(cfg->cached_tile == tile);
    GGML_ASSERT(cfg->k == k);
    GGML_ASSERT(cfg->out_channels == out_channels);
    GGML_ASSERT(cfg->n_tiles == n_tiles);
    const bool output_int24 = llama_text_lm_head_w8a16_int24_enabled();
    const std::string int24_diag_path = llama_text_lm_head_w8a16_int24_diag_path();
    const bool diag_int24 = !int24_diag_path.empty();
    const bool compute_int24 = output_int24 || diag_int24;
    const int int24_frac_bits = llama_text_lm_head_w8a16_int24_frac_bits();
    const size_t sample_limit = llama_text_decode_awq_int24_dump_limit();
    llama_decode_awq_int24_thread_stats int24_stats;

    static bool int24_logged = false;
    if (compute_int24 && ith == 0 && !int24_logged) {
        LLAMA_LOG_INFO("%s: AICAS lm_head W8A16 int24 %s enabled; frac_bits=%d diag=%s\n",
                __func__, output_int24 ? "output" : "diagnostic", int24_frac_bits,
                int24_diag_path.empty() ? "<none>" : int24_diag_path.c_str());
        int24_logged = true;
    }

    const int64_t total_tasks = n_cols * out_channels;
    const int64_t tasks_per_thread = (total_tasks + nth - 1) / nth;
    const int64_t task_begin = ith * tasks_per_thread;
    const int64_t task_end = std::min(total_tasks, task_begin + tasks_per_thread);
    if (task_begin >= task_end) {
        return;
    }

    for (int64_t task = task_begin; task < task_end; ++task) {
        const int64_t col = task / out_channels;
        const int64_t row = task % out_channels;
        const auto * act_col = (const ggml_fp16_t *) ((const char *) act->data + col * act->nb[1]);
        float * out_col = (float *) ((char *) dst->data + col * dst->nb[1]);

        uint16_t acc = 0;
        bool have_acc = false;
        double acc_int24 = 0.0;
        for (int64_t t = 0; t < n_tiles; ++t) {
            const int64_t start = t * tile;
            const int64_t end = std::min<int64_t>(k, start + tile);
            const int valid = (int) (end - start);
            const int8_t * q_tile = cfg->qweight.data() +
                ((size_t) row * (size_t) n_tiles + (size_t) t) * (size_t) tile;
            const uint16_t scale = cfg->scales[(size_t) row * (size_t) n_tiles + (size_t) t];
            const uint16_t tile_sum = aicas_rtl_dp128_tile(
                    [&](int lane) {
                        return aicas_rtl_fp16_from_int9((int) q_tile[lane]);
                    },
                    [&](int lane) {
                        return aicas_rtl_fp16_zero_if_non_normal(aicas_rtl_fp16_bits(act_col[start + lane]));
                    },
                    valid);
            const uint16_t scaled = aicas_rtl_fp16_mul(tile_sum, scale);
            acc = have_acc ? aicas_rtl_fp16_add(acc, scaled) : scaled;
            have_acc = true;
            if (compute_int24) {
                const double scale_f = (double) aicas_rtl_fp16_to_f32(scale);
                for (int lane = 0; lane < valid; ++lane) {
                    const uint16_t act_bits = aicas_rtl_fp16_zero_if_non_normal(aicas_rtl_fp16_bits(act_col[start + lane]));
                    const double scaled_act = (double) aicas_rtl_fp16_to_f32(act_bits) * scale_f;
                    int32_t q24 = llama_decode_awq_quant_int24(scaled_act, int24_frac_bits, int24_stats.clip_count);
                    if (diag_int24) {
                        int24_stats.max_abs_q24 = std::max<int32_t>(int24_stats.max_abs_q24, std::abs(q24));
                        int24_stats.update_scalar(scaled_act, q24, int24_frac_bits);
                    }
                    acc_int24 += (double) q24 * (double) q_tile[lane];
                }
            }
        }

        const float ref_out = aicas_rtl_fp16_to_f32(acc);
        const float int24_out = compute_int24 ? (float) std::ldexp(acc_int24, -int24_frac_bits) : 0.0f;
        if (diag_int24) {
            int24_stats.update(ref_out, int24_out);
            if (int24_stats.samples.size() < sample_limit) {
                int24_stats.samples.push_back({
                    /*.col =*/ col,
                    /*.row =*/ row,
                    /*.ref =*/ ref_out,
                    /*.int24 =*/ int24_out,
                    /*.err =*/ int24_out - ref_out,
                });
            }
        }
        out_col[row] = output_int24 ? int24_out : ref_out;
    }

    if (diag_int24) {
        llama_lm_head_w8a16_int24_write_diag(ith, int24_frac_bits, tile, int24_stats);
    }
}

static void llama_compute_text_lm_head_w8a16_npu_mul_mat(
        struct ggml_tensor * dst,
        const struct ggml_tensor * out_template,
        const struct ggml_tensor * act,
        const struct ggml_tensor * weight,
        int ith,
        int nth,
        void * userdata) {
    GGML_UNUSED(out_template);
    GGML_UNUSED(nth);

    if (ith != 0) {
        return;
    }

    auto * cfg = static_cast<llama_text_lm_head_w8a16_userdata *>(userdata);
    GGML_ASSERT(cfg != nullptr);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(act->type == GGML_TYPE_F16);
    GGML_ASSERT(weight->type == GGML_TYPE_F16);
    GGML_ASSERT(weight->ne[0] == act->ne[0]);
    GGML_ASSERT(dst->ne[0] == weight->ne[1]);
    GGML_ASSERT(dst->ne[1] == act->ne[1]);

#ifdef GGML_USE_NPU
    const int64_t group = std::max<int64_t>(1, std::min<int64_t>(128, cfg->tile));
    const bool ok = ggml_backend_npu_decode_w8a16_gemv_ex(
            ggml_get_name(weight),
            weight->data,
            weight->ne[0],
            weight->ne[1],
            weight->nb[0],
            weight->nb[1],
            group,
            act->data,
            act->type,
            act->nb[0],
            act->nb[1],
            act->ne[1],
            dst->data,
            dst->type,
            dst->nb[1]);
    if (ok) {
        if (llama_text_lm_head_w8a16_npu_compare_enabled()) {
            static std::atomic<int> compare_reports{0};
            const int report_idx = compare_reports.fetch_add(1, std::memory_order_relaxed);
            if (report_idx < llama_text_lm_head_w8a16_npu_compare_limit()) {
                const int64_t out_channels = weight->ne[1];
                const int64_t n_cols = act->ne[1];
                const size_t ref_bytes = (size_t) (n_cols - 1) * (size_t) dst->nb[1] +
                    (size_t) out_channels * sizeof(float);
                std::vector<uint8_t> ref_storage(ref_bytes, 0);
                struct ggml_tensor ref_dst = *dst;
                ref_dst.data = ref_storage.data();
                llama_compute_text_lm_head_w8a16_dp128_mul_mat(
                        &ref_dst,
                        out_template,
                        act,
                        weight,
                        0,
                        1,
                        userdata);

                double sum_sq = 0.0;
                float max_abs = 0.0f;
                float max_rel = 0.0f;
                int64_t max_col = 0;
                int64_t max_row = 0;
                float max_npu = 0.0f;
                float max_cpu = 0.0f;
                int64_t top_npu_row = -1;
                int64_t top_cpu_row = -1;
                float top_npu = -std::numeric_limits<float>::infinity();
                float top_cpu = -std::numeric_limits<float>::infinity();

                for (int64_t col = 0; col < n_cols; ++col) {
                    const float * npu_col = (const float *) ((const char *) dst->data + col * dst->nb[1]);
                    const float * cpu_col = (const float *) ((const char *) ref_dst.data + col * ref_dst.nb[1]);
                    for (int64_t row = 0; row < out_channels; ++row) {
                        const float npu_v = npu_col[row];
                        const float cpu_v = cpu_col[row];
                        if (npu_v > top_npu) {
                            top_npu = npu_v;
                            top_npu_row = row;
                        }
                        if (cpu_v > top_cpu) {
                            top_cpu = cpu_v;
                            top_cpu_row = row;
                        }
                        const float abs_err = std::fabs(npu_v - cpu_v);
                        const float rel_err = abs_err / std::max(1.0e-8f, std::fabs(cpu_v));
                        sum_sq += (double) abs_err * (double) abs_err;
                        if (abs_err > max_abs) {
                            max_abs = abs_err;
                            max_rel = rel_err;
                            max_col = col;
                            max_row = row;
                            max_npu = npu_v;
                            max_cpu = cpu_v;
                        }
                    }
                }

                const double denom = std::max<int64_t>(1, n_cols * out_channels);
                LLAMA_LOG_WARN(
                        "%s: compare %s report=%d cols=%" PRId64 " rows=%" PRId64
                        " max_abs=%.8g max_rel=%.8g rms=%.8g at col=%" PRId64 " row=%" PRId64
                        " npu=%.8g cpu=%.8g top_npu=(%" PRId64 ",%.8g) top_cpu=(%" PRId64 ",%.8g)\n",
                        __func__,
                        ggml_get_name(weight),
                        report_idx,
                        n_cols,
                        out_channels,
                        (double) max_abs,
                        (double) max_rel,
                        std::sqrt(sum_sq / denom),
                        max_col,
                        max_row,
                        (double) max_npu,
                        (double) max_cpu,
                        top_npu_row,
                        (double) top_npu,
                        top_cpu_row,
                        (double) top_cpu);
            }
        }
        return;
    }
    static bool warned = false;
    if (!warned) {
        LLAMA_LOG_WARN("%s: AICAS lm_head W8A16 NPU path failed; falling back to CPU DP128 simulation\n",
                __func__);
        warned = true;
    }
#endif

    llama_compute_text_lm_head_w8a16_dp128_mul_mat(
            dst,
            out_template,
            act,
            weight,
            0,
            1,
            userdata);
}

static void llama_token_embd_w8a16_decode_row(
        float * dst,
        const struct ggml_tensor * weight,
        int64_t token,
        int64_t group) {
    const int64_t n_embd = weight->ne[0];
    const int64_t n_groups = (n_embd + group - 1) / group;
    const auto * row = (const ggml_fp16_t *) ((const char *) weight->data + token * weight->nb[1]);

    for (int64_t g = 0; g < n_groups; ++g) {
        const int64_t start = g * group;
        const int64_t end = std::min<int64_t>(n_embd, start + group);
        float amax = 0.0f;
        for (int64_t i = start; i < end; ++i) {
            const uint16_t bits = aicas_rtl_fp16_zero_if_non_normal(aicas_rtl_fp16_bits(row[i]));
            amax = std::max(amax, std::fabs(aicas_rtl_fp16_to_f32(bits)));
        }

        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        const uint16_t scale = aicas_rtl_fp16_bits(ggml_fp32_to_fp16(d));

        for (int64_t lane = 0; lane < end - start; ++lane) {
            const uint16_t bits = aicas_rtl_fp16_zero_if_non_normal(aicas_rtl_fp16_bits(row[start + lane]));
            const float value = aicas_rtl_fp16_to_f32(bits);
            int q = (int) std::round(value * id);
            q = std::max(-127, std::min(127, q));
            const uint16_t dequant = aicas_rtl_fp16_mul(
                    aicas_rtl_fp16_from_int9(q),
                    scale);
            dst[start + lane] = aicas_rtl_fp16_to_f32(dequant);
        }
    }
}

static bool llama_text_token_embd_w8a16_preload_impl(const struct ggml_tensor * tok_embd, int64_t group) {
    if (tok_embd == nullptr ||
            tok_embd->data == nullptr ||
            tok_embd->type != GGML_TYPE_F16 ||
            tok_embd->ne[0] <= 0 ||
            tok_embd->ne[1] <= 0 ||
            tok_embd->nb[0] != static_cast<int64_t>(sizeof(ggml_fp16_t))) {
        return false;
    }

    group = std::max<int64_t>(1, std::min<int64_t>(128, group));
    const int64_t n_embd = tok_embd->ne[0];
    const int64_t vocab = tok_embd->ne[1];
    const int64_t n_groups = (n_embd + group - 1) / group;
    llama_text_token_embd_w8a16_userdata & cfg = llama_text_token_embd_w8a16_cache();

    std::lock_guard<std::mutex> lock(cfg.mutex);
    if (cfg.weight_data == tok_embd->data &&
            cfg.cached_group == group &&
            cfg.n_embd == n_embd &&
            cfg.vocab == vocab &&
            cfg.n_groups == n_groups &&
            cfg.decoded_rows.size() == (size_t) vocab * (size_t) n_embd) {
        return true;
    }

    const int64_t t0 = ggml_time_us();
    std::vector<float> decoded((size_t) vocab * (size_t) n_embd);
    for (int64_t token = 0; token < vocab; ++token) {
        llama_token_embd_w8a16_decode_row(
                decoded.data() + (size_t) token * (size_t) n_embd,
                tok_embd,
                token,
                group);
    }

    cfg.weight_data = tok_embd->data;
    cfg.group = group;
    cfg.cached_group = group;
    cfg.n_embd = n_embd;
    cfg.vocab = vocab;
    cfg.n_groups = n_groups;
    cfg.decoded_rows = std::move(decoded);
    LLAMA_LOG_INFO("%s: predecoded token_embd.weight W8A16 rows=%" PRId64 " dim=%" PRId64
            " group=%" PRId64 " bytes=%.2f MiB time=%.3f ms\n",
            __func__, vocab, n_embd, group,
            (double) cfg.decoded_rows.size() * sizeof(float) / 1048576.0,
            (double) (ggml_time_us() - t0) / 1000.0);
    return true;
}

static void llama_compute_text_token_embd_w8a16_get_rows(
        struct ggml_tensor * dst,
        const struct ggml_tensor * out_template,
        const struct ggml_tensor * weight,
        const struct ggml_tensor * tokens,
        int ith,
        int nth,
        void * userdata) {
    GGML_UNUSED(out_template);

    auto * cfg = static_cast<llama_text_token_embd_w8a16_userdata *>(userdata);
    GGML_ASSERT(cfg != nullptr);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(weight->type == GGML_TYPE_F16);
    GGML_ASSERT(tokens->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->ne[0] == weight->ne[0]);
    GGML_ASSERT(dst->ne[1] == tokens->ne[0]);
    GGML_ASSERT(dst->nb[0] == static_cast<int64_t>(sizeof(float)));

    const int64_t n_embd = weight->ne[0];
    const int64_t vocab = weight->ne[1];
    const int64_t n_tokens = tokens->ne[0];
    const int64_t group = std::max<int64_t>(1, std::min<int64_t>(128, cfg->group));
    const int64_t n_groups = (n_embd + group - 1) / group;
    const int64_t rows_per_thread = (n_tokens + nth - 1) / nth;
    const int64_t row_begin = ith * rows_per_thread;
    const int64_t row_end = std::min<int64_t>(n_tokens, row_begin + rows_per_thread);
    if (row_begin >= row_end) {
        return;
    }

    const float * cached_rows = nullptr;
    {
        std::lock_guard<std::mutex> lock(cfg->mutex);
        if (cfg->weight_data == weight->data &&
                cfg->cached_group == group &&
                cfg->n_embd == n_embd &&
                cfg->vocab == vocab &&
                cfg->n_groups == n_groups &&
                cfg->decoded_rows.size() == (size_t) vocab * (size_t) n_embd) {
            cached_rows = cfg->decoded_rows.data();
        }
    }

    const int32_t * token_data = (const int32_t *) tokens->data;
    for (int64_t token_idx = row_begin; token_idx < row_end; ++token_idx) {
        const int64_t token = token_data[token_idx];
        GGML_ASSERT(token >= 0 && token < vocab);

        auto * out_col = (float *) ((char *) dst->data + token_idx * dst->nb[1]);
        if (cached_rows != nullptr) {
            std::memcpy(out_col,
                    cached_rows + (size_t) token * (size_t) n_embd,
                    (size_t) n_embd * sizeof(float));
        } else {
            llama_token_embd_w8a16_decode_row(out_col, weight, token, group);
        }
    }
}

static void llama_compute_text_decode_awq_mul_mat(
        struct ggml_tensor * dst,
        const struct ggml_tensor * a,
        const struct ggml_tensor * b,
        const struct ggml_tensor * c,
        int ith,
        int nth,
        void * userdata) {
    GGML_UNUSED(a);

    const auto * cfg = static_cast<const llama_aicas_text_decode_awq_tensor *>(userdata);
    GGML_ASSERT(cfg != nullptr);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(b->type == GGML_TYPE_F32 || b->type == GGML_TYPE_F16);
    GGML_ASSERT(c->type == GGML_TYPE_I8);
    GGML_ASSERT(cfg->scale_tensor != nullptr);
    GGML_ASSERT(cfg->zero_tensor != nullptr);
    GGML_ASSERT(cfg->scale_tensor->type == GGML_TYPE_F32 || cfg->scale_tensor->type == GGML_TYPE_F16);
    GGML_ASSERT(cfg->zero_tensor->type == GGML_TYPE_F32 || cfg->zero_tensor->type == GGML_TYPE_F16);

    const int64_t k = cfg->in_features;
    const int64_t packed_k = cfg->packed_in_features();
    const int64_t out_channels = c->ne[1];
    const int64_t n_cols = b->ne[1];

    GGML_ASSERT(k > 0);
    GGML_ASSERT(packed_k == c->ne[0]);
    GGML_ASSERT(b->ne[0] == k);
    GGML_ASSERT(cfg->has_valid_smooth_config());
    GGML_ASSERT(cfg->has_valid_group_params(out_channels));

    const float * zero_data = (const float *) cfg->zero_tensor->data;
    GGML_ASSERT(cfg->scale_tensor->type == GGML_TYPE_F32 || cfg->scale_tensor->type == GGML_TYPE_F16);
    GGML_ASSERT(cfg->zero_tensor->type == GGML_TYPE_F32);

    const int64_t total_tasks = n_cols * out_channels;
    const int64_t tasks_per_thread = (total_tasks + nth - 1) / nth;
    const int64_t task_begin = ith * tasks_per_thread;
    const int64_t task_end = std::min(total_tasks, task_begin + tasks_per_thread);
    if (task_begin >= task_end) {
        return;
    }

    const bool accum_fp16 = llama_text_decode_awq_accum_mode_from_env() == llama_decode_awq_accum_mode::fp16;
    const bool output_int24 = llama_text_decode_awq_int24_enabled();
    const std::string int24_diag_path = llama_text_decode_awq_int24_diag_path();
    const bool diag_int24 = !int24_diag_path.empty();
    const bool compute_int24 = output_int24 || diag_int24;
    const bool compute_ref = !output_int24 || diag_int24;
    const int int24_frac_bits = llama_text_decode_awq_int24_frac_bits();
    const size_t sample_limit = llama_text_decode_awq_int24_dump_limit();
    llama_decode_awq_int24_thread_stats int24_stats;

    static bool int24_logged = false;
    if (compute_int24 && ith == 0 && !int24_logged) {
        LLAMA_LOG_INFO("%s: AICAS decode AWQ int24 %s enabled; frac_bits=%d diag=%s\n",
                __func__, output_int24 ? "output" : "diagnostic", int24_frac_bits,
                int24_diag_path.empty() ? "<none>" : int24_diag_path.c_str());
        int24_logged = true;
    }

    int64_t cached_col = -1;
    std::vector<float> act_smooth(static_cast<size_t>(k));

    for (int64_t task = task_begin; task < task_end; ++task) {
        const int64_t col = task / out_channels;
        const int64_t j   = task % out_channels;
        const char * act_col = (const char *) b->data + col * b->nb[1];
        float * out_col = (float *) ((char *) dst->data + col * dst->nb[1]);

        if (cached_col != col) {
            for (int64_t i = 0; i < k; ++i) {
                const float act_value = llama_decode_awq_read_act(b, act_col, i);
                act_smooth[static_cast<size_t>(i)] = act_value / cfg->smooth_scale[static_cast<size_t>(i)];
            }
            cached_col = col;
        }

        const uint8_t * w_row = (const uint8_t *) ((const char *) c->data + j * c->nb[1]);
        const float * zero_row  = zero_data + j * cfg->zero_tensor->ne[0];

        float acc = 0.0f;
        ggml_fp16_t acc_fp16 = ggml_fp32_to_fp16(0.0f);
        double acc_int24 = 0.0;
        const int64_t groups = cfg->zero_tensor->ne[0];
        for (int64_t g = 0; g < groups; ++g) {
            const int64_t start = g * cfg->group_size;
            const int64_t end = std::min(k, start + cfg->group_size);
            const float group_scale = llama_decode_awq_read_scale(cfg->scale_tensor, j, g);
            const float group_zero = zero_row[g];
            if (diag_int24 && std::fabs(group_zero - std::round(group_zero)) > 1.0e-6f) {
                ++int24_stats.non_integer_zero_count;
            }
            for (int64_t i = start; i < end; ++i) {
                const uint8_t packed = w_row[i / 2];
                const float q = (float) llama_decode_awq_q4(packed, i);
                if (compute_ref) {
                    const float w = (q - group_zero) * group_scale;
                    const float product = act_smooth[static_cast<size_t>(i)] * w;
                    if (accum_fp16) {
                        acc_fp16 = ggml_fp32_to_fp16(ggml_fp16_to_fp32(acc_fp16) + product);
                    } else {
                        acc += product;
                    }
                }
                if (compute_int24) {
                    const double scaled_act = (double) act_smooth[static_cast<size_t>(i)] * (double) group_scale;
                    int32_t q24 = llama_decode_awq_quant_int24(scaled_act, int24_frac_bits, int24_stats.clip_count);
                    if (diag_int24) {
                        int24_stats.max_abs_q24 = std::max<int32_t>(int24_stats.max_abs_q24, std::abs(q24));
                        int24_stats.update_scalar(scaled_act, q24, int24_frac_bits);
                    }
                    acc_int24 += (double) q24 * ((double) q - (double) group_zero);
                }
            }
        }
        const float ref_out = compute_ref ? (accum_fp16 ? ggml_fp16_to_fp32(acc_fp16) : acc) : 0.0f;
        const float int24_out = compute_int24 ? (float) std::ldexp(acc_int24, -int24_frac_bits) : 0.0f;
        if (diag_int24) {
            int24_stats.update(ref_out, int24_out);
            if (int24_stats.samples.size() < sample_limit) {
                int24_stats.samples.push_back({
                    /*.col =*/ col,
                    /*.row =*/ j,
                    /*.ref =*/ ref_out,
                    /*.int24 =*/ int24_out,
                    /*.err =*/ int24_out - ref_out,
                });
            }
        }
        out_col[j] = output_int24 ? int24_out : ref_out;
    }

    if (diag_int24) {
        llama_decode_awq_int24_write_diag(cfg, ith, int24_frac_bits, accum_fp16, output_int24, int24_stats);
    }
}

#ifdef GGML_USE_NPU
static void llama_compute_text_decode_awq_npu_mul_mat(
        struct ggml_tensor * dst,
        const struct ggml_tensor * a,
        const struct ggml_tensor * b,
        const struct ggml_tensor * c,
        int ith,
        int nth,
        void * userdata) {
    GGML_UNUSED(a);
    GGML_UNUSED(nth);

    if (ith != 0) {
        return;
    }

    const auto * cfg = static_cast<const llama_aicas_text_decode_awq_tensor *>(userdata);
    GGML_ASSERT(cfg != nullptr);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(b->type == GGML_TYPE_F32 || b->type == GGML_TYPE_F16);
    GGML_ASSERT(c->type == GGML_TYPE_I8);
    GGML_ASSERT(cfg->quant_tensor == c);
    GGML_ASSERT(cfg->scale_tensor != nullptr);
    GGML_ASSERT(cfg->zero_tensor != nullptr);
    GGML_ASSERT(cfg->has_valid_smooth_config());

    const bool ok = ggml_backend_npu_decode_w4a16_gemv_ex(
            cfg->quant_tensor != nullptr ? ggml_get_name(cfg->quant_tensor) : nullptr,
            cfg->quant_tensor != nullptr ? cfg->quant_tensor->data : nullptr,
            cfg->packed_in_features(),
            cfg->quant_tensor != nullptr ? cfg->quant_tensor->ne[1] : 0,
            cfg->quant_tensor != nullptr ? cfg->quant_tensor->nb[1] : 0,
            cfg->scale_tensor->data,
            (int) cfg->scale_tensor->type,
            cfg->scale_tensor->nb[0],
            cfg->scale_tensor->nb[1],
            cfg->zero_tensor->data,
            (int) cfg->zero_tensor->type,
            cfg->zero_tensor->nb[0],
            cfg->zero_tensor->nb[1],
            b->data,
            (int) b->type,
            b->nb[0],
            b->nb[1],
            cfg->smooth_scale.data(),
            cfg->in_features,
            b->ne[1],
            dst->data,
            (int) dst->type,
            dst->nb[1]);
    if (ok) {
        if (llama_text_decode_awq_npu_compare_enabled()) {
            static std::atomic<int> compare_reports {0};
            const int report_idx = compare_reports.load(std::memory_order_relaxed);
            if (report_idx < llama_text_decode_awq_npu_compare_limit()) {
                const int64_t out_channels = dst->ne[0];
                const int64_t n_cols = dst->ne[1];
                const int64_t ref_stride_bytes = out_channels * (int64_t) sizeof(float);
                std::vector<float> ref((size_t) out_channels * (size_t) n_cols, 0.0f);
                struct ggml_tensor ref_dst = *dst;
                ref_dst.data = ref.data();
                ref_dst.nb[0] = sizeof(float);
                ref_dst.nb[1] = ref_stride_bytes;
                llama_compute_text_decode_awq_mul_mat(&ref_dst, a, b, c, 0, 1, userdata);

                double sum_sq = 0.0;
                double max_abs = 0.0;
                double max_rel = 0.0;
                int64_t max_col = 0;
                int64_t max_row = 0;
                float max_ref = 0.0f;
                float max_got = 0.0f;
                for (int64_t col = 0; col < n_cols; ++col) {
                    const float * got_col = (const float *) ((const char *) dst->data + col * dst->nb[1]);
                    const float * ref_col = ref.data() + (size_t) col * (size_t) out_channels;
                    for (int64_t row = 0; row < out_channels; ++row) {
                        const float got = got_col[row];
                        const float exp = ref_col[row];
                        const double abs_err = std::fabs((double) got - (double) exp);
                        const double rel_err = abs_err / std::max(1.0e-6, std::fabs((double) exp));
                        sum_sq += abs_err * abs_err;
                        if (abs_err > max_abs) {
                            max_abs = abs_err;
                            max_rel = rel_err;
                            max_col = col;
                            max_row = row;
                            max_ref = exp;
                            max_got = got;
                        }
                    }
                }

                if (compare_reports.fetch_add(1, std::memory_order_relaxed) < llama_text_decode_awq_npu_compare_limit()) {
                    const double count = (double) std::max<int64_t>(1, out_channels * n_cols);
                    LLAMA_LOG_WARN(
                            "%s: compare op=%s dims=(m=%" PRId64 ",k=%" PRId64 ",n=%" PRId64
                            ") max_abs=%.8g max_rel=%.8g rms=%.8g at(col=%" PRId64 ",row=%" PRId64
                            ") npu=%.8g cpu=%.8g\n",
                            __func__,
                            cfg->quant_tensor != nullptr ? ggml_get_name(cfg->quant_tensor) : "(unnamed)",
                            out_channels,
                            cfg->in_features,
                            n_cols,
                            max_abs,
                            max_rel,
                            std::sqrt(sum_sq / count),
                            max_col,
                            max_row,
                            (double) max_got,
                            (double) max_ref);
                }
            }
        }
        return;
    }

    static bool warned = false;
    if (!warned) {
        LLAMA_LOG_WARN("%s: NPU W4A16 decode GEMV rejected at least one op; falling back to CPU AWQ kernel\n", __func__);
        warned = true;
    }
    llama_compute_text_decode_awq_mul_mat(dst, a, b, c, 0, 1, userdata);
}

struct llama_text_decode_swiglu_ffn_npu_userdata {
    std::string op_name;
    ggml_npu_decode_awq_view gate = {};
    ggml_npu_decode_awq_view up   = {};
    ggml_npu_decode_awq_view down = {};
};

static ggml_npu_decode_awq_view llama_make_npu_decode_awq_view(const llama_aicas_text_decode_awq_tensor & cfg) {
    ggml_npu_decode_awq_view view = {};
    view.weight_name = cfg.quant_tensor != nullptr ? ggml_get_name(cfg.quant_tensor) : nullptr;
    view.q4_data = cfg.quant_tensor != nullptr ? cfg.quant_tensor->data : nullptr;
    view.packed_k = cfg.packed_in_features();
    view.out_channels = cfg.quant_tensor != nullptr ? cfg.quant_tensor->ne[1] : 0;
    view.q4_nb1 = cfg.quant_tensor != nullptr ? cfg.quant_tensor->nb[1] : 0;
    view.scale_data = cfg.scale_tensor != nullptr ? cfg.scale_tensor->data : nullptr;
    view.scale_type = cfg.scale_tensor != nullptr ? (int) cfg.scale_tensor->type : GGML_TYPE_COUNT;
    view.scale_nb0 = cfg.scale_tensor != nullptr ? cfg.scale_tensor->nb[0] : 0;
    view.scale_nb1 = cfg.scale_tensor != nullptr ? cfg.scale_tensor->nb[1] : 0;
    view.zero_data = cfg.zero_tensor != nullptr ? cfg.zero_tensor->data : nullptr;
    view.zero_type = cfg.zero_tensor != nullptr ? (int) cfg.zero_tensor->type : GGML_TYPE_COUNT;
    view.zero_nb0 = cfg.zero_tensor != nullptr ? cfg.zero_tensor->nb[0] : 0;
    view.zero_nb1 = cfg.zero_tensor != nullptr ? cfg.zero_tensor->nb[1] : 0;
    view.smooth_scale = cfg.smooth_scale.empty() ? nullptr : cfg.smooth_scale.data();
    view.smooth_scale_len = cfg.smooth_scale.size();
    view.k = cfg.in_features;
    return view;
}

static void llama_compute_text_decode_swiglu_ffn_npu(
        struct ggml_tensor * dst,
        int ith,
        int nth,
        void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }
    const struct ggml_tensor * a = dst->src[0];
    const auto * cfg = static_cast<const llama_text_decode_swiglu_ffn_npu_userdata *>(userdata);
    GGML_ASSERT(cfg != nullptr);
    GGML_ASSERT(a != nullptr);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(a->type == GGML_TYPE_F16 || a->type == GGML_TYPE_F32);
    GGML_ASSERT(a->ne[1] == 1);
    const bool ok = ggml_backend_npu_decode_swiglu_ffn_w4a16_ex(
            cfg->op_name.c_str(),
            &cfg->gate,
            &cfg->up,
            &cfg->down,
            a->data,
            (int) a->type,
            a->nb[0],
            a->nb[1],
            dst->data,
            (int) dst->type,
            dst->nb[1]);
    GGML_ASSERT(ok);
}

struct llama_text_decode_attention_npu_userdata {
    std::string op_name;
    ggml_npu_decode_awq_view q = {};
    ggml_npu_decode_awq_view k = {};
    ggml_npu_decode_awq_view v = {};
    ggml_npu_decode_awq_view o = {};
    float kq_scale = 1.0f;
    float rope_freq_base = 10000.0f;
    float rope_freq_scale = 1.0f;
    int32_t layer_id = -1;
};

static bool llama_text_decode_attention_npu_enabled() {
    if (llama_text_decode_awq_npu_gemv_only()) {
        return false;
    }
    const char * value = std::getenv("AICAS_TEXT_DECODE_ATTN_NPU");
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    return std::strcmp(value, "0") != 0;
}

static int llama_text_decode_attention_npu_max_layer() {
    const char * value = std::getenv("AICAS_TEXT_DECODE_ATTN_NPU_MAX_LAYER");
    if (value == nullptr || value[0] == '\0') {
        return std::numeric_limits<int>::max();
    }
    char * end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || parsed < 0) {
        return std::numeric_limits<int>::max();
    }
    return parsed > std::numeric_limits<int>::max() ? std::numeric_limits<int>::max() : static_cast<int>(parsed);
}

static void llama_compute_text_decode_attention_npu(
        struct ggml_tensor * dst,
        int ith,
        int nth,
        void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }
    auto * cfg = static_cast<llama_text_decode_attention_npu_userdata *>(userdata);
    GGML_ASSERT(cfg != nullptr);
    const ggml_tensor * act = dst->src[0];
    const ggml_tensor * pos = dst->src[1];
    const ggml_tensor * k_cache = dst->src[2];
    const ggml_tensor * v_cache = dst->src[3];
    GGML_ASSERT(act != nullptr);
    GGML_ASSERT(pos != nullptr);
    GGML_ASSERT(k_cache != nullptr);
    GGML_ASSERT(v_cache != nullptr);
    GGML_ASSERT(pos->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
    GGML_ASSERT(act->type == GGML_TYPE_F16 || act->type == GGML_TYPE_F32);

    const int32_t position = *(const int32_t *) pos->data;
    const bool ok = ggml_backend_npu_decode_attention_w4a16_ex(
            cfg->op_name.c_str(),
            &cfg->q,
            &cfg->k,
            &cfg->v,
            &cfg->o,
            act->data,
            (int) act->type,
            act->nb[0],
            act->nb[1],
            k_cache,
            v_cache,
            position,
            cfg->kq_scale,
            cfg->rope_freq_base,
            cfg->rope_freq_scale,
            cfg->layer_id,
            dst->data,
            (int) dst->type,
            dst->nb[1]);
    GGML_ASSERT(ok);
}
#endif

struct llama_text_activation_registry {
    std::string output_path;
    size_t sample_limit = 0;
    std::mutex mutex;
    std::unordered_map<std::string, llama_text_activation_stats> stats;
    std::unordered_map<std::string, llama_text_activation_observer> observers;

    llama_text_activation_registry() {
        const char * stats_path = std::getenv("AICAS_TEXT_ACT_STATS_FILE");
        output_path = stats_path ? stats_path : "";
        if (!output_path.empty()) {
            sample_limit = 4096;
            if (const char * samples_env = std::getenv("AICAS_TEXT_ACT_SAMPLES")) {
                const long parsed = strtol(samples_env, nullptr, 10);
                if (parsed > 0) {
                    sample_limit = (size_t) parsed;
                }
            }
        }
    }

    ~llama_text_activation_registry() {
        flush();
    }

    bool enabled() const {
        return !output_path.empty();
    }

    llama_text_activation_observer * get_observer(const std::string & tensor_name) {
        if (!enabled()) {
            return nullptr;
        }
        std::lock_guard<std::mutex> lock(mutex);
        auto stats_it = stats.find(tensor_name);
        if (stats_it == stats.end()) {
            llama_text_activation_stats item;
            item.sample_limit = sample_limit;
            stats_it = stats.emplace(tensor_name, std::move(item)).first;
        }
        auto obs_it = observers.find(tensor_name);
        if (obs_it == observers.end()) {
            obs_it = observers.emplace(tensor_name, llama_text_activation_observer{tensor_name}).first;
        }
        GGML_UNUSED(stats_it);
        return &obs_it->second;
    }

    void record(const std::string & tensor_name, const float * data, size_t channels, size_t cols) {
        if (!enabled()) {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex);
        auto & item = stats[tensor_name];
        if (item.sample_limit == 0) {
            item.sample_limit = sample_limit;
        }
        item.update(data, channels, cols);
    }

    void flush() {
        if (!enabled()) {
            return;
        }

        std::lock_guard<std::mutex> lock(mutex);
        std::vector<std::string> names;
        names.reserve(stats.size());
        for (const auto & kv : stats) {
            names.push_back(kv.first);
        }
        std::sort(names.begin(), names.end());

        std::ofstream fout(output_path, std::ios::binary);
        if (!fout.is_open()) {
            LLAMA_LOG_ERROR("%s: failed to open text activation stats file: %s\n", __func__, output_path.c_str());
            return;
        }

        fout << "{\n";
        fout << "  \"schema\": \"aicas.llama.text.act_stats.v1\",\n";
        fout << "  \"samples_per_tensor\": " << sample_limit << ",\n";
        fout << "  \"tensors\": [\n";

        bool first_tensor = true;
        for (const auto & name : names) {
            const auto & item = stats.at(name);
            if (item.count == 0) {
                continue;
            }

            if (!first_tensor) {
                fout << ",\n";
            }
            first_tensor = false;

            auto write_float_array = [&](const std::vector<float> & values, int indent) {
                fout << "[";
                for (size_t i = 0; i < values.size(); ++i) {
                    if (i > 0) {
                        fout << ", ";
                    }
                    fout << values[i];
                }
                fout << "]";
                GGML_UNUSED(indent);
            };

            auto write_u32_array = [&](const std::vector<uint32_t> & values) {
                fout << "[";
                for (size_t i = 0; i < values.size(); ++i) {
                    if (i > 0) {
                        fout << ", ";
                    }
                    fout << values[i];
                }
                fout << "]";
            };

            fout << "    {\n";
            fout << "      \"tensor_name\": \"" << name << "\",\n";
            fout << "      \"count\": " << item.count << ",\n";
            fout << "      \"in_channels\": " << item.in_channels << ",\n";
            fout << "      \"min\": " << item.min << ",\n";
            fout << "      \"max\": " << item.max << ",\n";
            fout << "      \"per_channel_min\": ";
            write_float_array(item.per_channel_min, 6);
            fout << ",\n";
            fout << "      \"per_channel_max\": ";
            write_float_array(item.per_channel_max, 6);
            fout << ",\n";
            fout << "      \"per_channel_absmax\": ";
            write_float_array(item.per_channel_absmax, 6);
            fout << ",\n";
            fout << "      \"samples\": ";
            write_float_array(item.samples, 6);
            fout << ",\n";
            fout << "      \"sample_channels\": ";
            write_u32_array(item.sample_channels);
            fout << "\n";
            fout << "    }";
        }

        fout << "\n  ]\n";
        fout << "}\n";
    }
};

static llama_text_activation_registry & llama_get_text_activation_registry() {
    static llama_text_activation_registry registry;
    return registry;
}

static void llama_collect_text_activation_f32_passthrough(
        struct ggml_tensor * dst,
        const struct ggml_tensor * a,
        int ith,
        int nth,
        void * userdata) {
    GGML_UNUSED(nth);
    auto * observer = static_cast<llama_text_activation_observer *>(userdata);
    GGML_ASSERT(observer != nullptr);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(a->type == GGML_TYPE_F32);

    if (ith != 0) {
        return;
    }

    const size_t nbytes = ggml_nbytes(a);
    GGML_ASSERT(nbytes == ggml_nbytes(dst));
    memcpy(dst->data, a->data, nbytes);
    llama_get_text_activation_registry().record(
        observer->tensor_name,
        (const float *) a->data,
        (size_t) a->ne[0],
        (size_t) ggml_nelements(a) / (size_t) a->ne[0]);
}

static ggml_tensor * llama_maybe_observe_text_activation(
        ggml_context * ctx0,
        ggml_tensor * act,
        const char * tensor_name,
        bool is_prefill_gemm_only) {
    if (act == nullptr || tensor_name == nullptr || tensor_name[0] == '\0') {
        return act;
    }
    const llama_text_act_collect_mode mode = llama_text_act_collect_mode_from_env();
    const bool should_collect = mode == llama_text_act_collect_mode::both ||
        (mode == llama_text_act_collect_mode::prefill && is_prefill_gemm_only) ||
        (mode == llama_text_act_collect_mode::decode && !is_prefill_gemm_only);
    if (!should_collect) {
        return act;
    }
    auto * observer = llama_get_text_activation_registry().get_observer(tensor_name);
    if (observer == nullptr) {
        return act;
    }
    return ggml_map_custom1(ctx0, act, llama_collect_text_activation_f32_passthrough, 1, observer);
}

static void llama_compute_prefill_tail_decode_switch_passthrough(
        struct ggml_tensor * dst,
        const struct ggml_tensor * a,
        int ith,
        int nth,
        void * userdata) {
    GGML_UNUSED(nth);
    GGML_UNUSED(userdata);

    if (ith != 0) {
        return;
    }

#ifdef GGML_USE_NPU
    static std::atomic<bool> logged{false};
    if (!logged.exchange(true, std::memory_order_relaxed)) {
        LLAMA_LOG_INFO("%s: switching to decode overlay before prefill-tail lm_head\n", __func__);
    }
    if (!ggml_backend_npu_decode_overlay_ensure_active("prefill_tail_lm_head")) {
        LLAMA_LOG_WARN("%s: decode overlay switch before prefill-tail lm_head failed\n", __func__);
    }
#endif

    GGML_ASSERT(ggml_nbytes(dst) == ggml_nbytes(a));
    memcpy(dst->data, a->data, ggml_nbytes(a));
}

} // namespace

bool llama_text_token_embd_w8a16_preload(const struct ggml_tensor * tok_embd, int64_t group) {
    return llama_text_token_embd_w8a16_preload_impl(tok_embd, group);
}

void llm_graph_input_embd::set_input(const llama_ubatch * ubatch) {
    if (ubatch->token) {
        const int64_t n_tokens = ubatch->n_tokens;

        ggml_backend_tensor_set(tokens, ubatch->token, 0, n_tokens*ggml_element_size(tokens));
    }

    if (ubatch->embd) {
        const int64_t n_embd   = embd->ne[0];
        const int64_t n_tokens = ubatch->n_tokens;

        ggml_backend_tensor_set(embd, ubatch->embd, 0, n_tokens*n_embd*ggml_element_size(embd));
    }
}

bool llm_graph_input_embd::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= (!tokens && !params.ubatch.token) || (tokens && tokens->ne[0] == params.ubatch.n_tokens);
    res &= (!embd   && !params.ubatch.embd)  || (embd   &&   embd->ne[0] == params.ubatch.n_tokens);

    return res;
}

void llm_graph_input_pos::set_input(const llama_ubatch * ubatch) {
    if (ubatch->pos && pos) {
        const int64_t n_tokens = ubatch->n_tokens;

        if (ubatch->token && n_pos_per_embd == 4) {
            // in case we're using M-RoPE with text tokens, convert the 1D positions to 4D
            // the 3 first dims are the same, and 4th dim is all 0
            std::vector<llama_pos> pos_data(n_tokens*n_pos_per_embd);
            // copy the first dimension
            for (int i = 0; i < n_tokens; ++i) {
                pos_data[               i] = ubatch->pos[i];
                pos_data[    n_tokens + i] = ubatch->pos[i];
                pos_data[2 * n_tokens + i] = ubatch->pos[i];
                pos_data[3 * n_tokens + i] = 0; // 4th dim is 0
            }
            ggml_backend_tensor_set(pos, pos_data.data(), 0, pos_data.size()*ggml_element_size(pos));
        } else {
            ggml_backend_tensor_set(pos, ubatch->pos, 0, n_tokens*n_pos_per_embd*ggml_element_size(pos));
        }
    }
}

bool llm_graph_input_pos::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= pos->ne[0] == params.ubatch.n_tokens;

    return res;
}

void llm_graph_input_attn_temp::set_input(const llama_ubatch * ubatch) {
    if (ubatch->pos && attn_scale) {
        const int64_t n_tokens = ubatch->n_tokens;

        std::vector<float> attn_scale_data(n_tokens, 0.0f);
        for (int i = 0; i < n_tokens; ++i) {
            const float pos = ubatch->pos[i];
            attn_scale_data[i] = std::log(
                std::floor((pos + 1.0f) / n_attn_temp_floor_scale) + 1.0
            ) * f_attn_temp_scale + 1.0;
        }

        ggml_backend_tensor_set(attn_scale, attn_scale_data.data(), 0, n_tokens*ggml_element_size(attn_scale));
    }
}

void llm_graph_input_pos_bucket::set_input(const llama_ubatch * ubatch) {
    if (pos_bucket) {
        const int64_t n_tokens = ubatch->n_tokens;

        GGML_ASSERT(ggml_backend_buffer_is_host(pos_bucket->buffer));
        GGML_ASSERT(!ubatch->equal_seqs()); // TODO: use ubatch->n_seqs instead of failing

        int32_t * data = (int32_t *) pos_bucket->data;

        for (int h = 0; h < 1; ++h) {
            for (int j = 0; j < n_tokens; ++j) {
                for (int i = 0; i < n_tokens; ++i) {
                    data[h*(n_tokens*n_tokens) + j*n_tokens + i] = llama_relative_position_bucket(ubatch->pos[i], ubatch->pos[j], hparams.n_rel_attn_bkts, true);
                }
            }
        }
    }
}

void llm_graph_input_pos_bucket_kv::set_input(const llama_ubatch * ubatch) {
    if (pos_bucket) {
        mctx->set_input_pos_bucket(pos_bucket, ubatch);
    }
}

void llm_graph_input_out_ids::set_input(const llama_ubatch * ubatch) {
    GGML_ASSERT(out_ids);

    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(ggml_backend_buffer_is_host(out_ids->buffer));
    int32_t * data = (int32_t *) out_ids->data;

    if (n_outputs == n_tokens) {
        for (int i = 0; i < n_tokens; ++i) {
            data[i] = i;
        }

        return;
    }

    GGML_ASSERT(ubatch->output);

    int n_outputs = 0;

    for (int i = 0; i < n_tokens; ++i) {
        if (ubatch->output[i]) {
            data[n_outputs++] = i;
        }
    }
}

bool llm_graph_input_out_ids::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= n_outputs == params.n_outputs;

    return res;
}

void llm_graph_input_mean::set_input(const llama_ubatch * ubatch) {
    if (cparams.embeddings && cparams.pooling_type == LLAMA_POOLING_TYPE_MEAN) {
        const int64_t n_tokens     = ubatch->n_tokens;
        const int64_t n_seq_tokens = ubatch->n_seq_tokens;
        const int64_t n_seqs_unq   = ubatch->n_seqs_unq;

        GGML_ASSERT(mean);
        GGML_ASSERT(ggml_backend_buffer_is_host(mean->buffer));

        float * data = (float *) mean->data;
        memset(mean->data, 0, n_tokens*n_seqs_unq*ggml_element_size(mean));

        std::vector<uint64_t> sums(n_seqs_unq, 0);
        for (int i = 0; i < n_tokens; i += n_seq_tokens) {
            for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                const llama_seq_id seq_id  = ubatch->seq_id[i][s];
                const int32_t      seq_idx = ubatch->seq_idx[seq_id];

                sums[seq_idx] += ubatch->n_seq_tokens;
            }
        }

        std::vector<float> div(n_seqs_unq, 0.0f);
        for (int s = 0; s < n_seqs_unq; ++s) {
            const uint64_t sum = sums[s];
            if (sum > 0) {
                div[s] = 1.0f/float(sum);
            }
        }

        for (int i = 0; i < n_tokens; i += n_seq_tokens) {
            for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                const llama_seq_id seq_id  = ubatch->seq_id[i][s];
                const int32_t      seq_idx = ubatch->seq_idx[seq_id];

                for (int j = 0; j < n_seq_tokens; ++j) {
                    data[seq_idx*n_tokens + i + j] = div[seq_idx];
                }
            }
        }
    }
}

void llm_graph_input_cls::set_input(const llama_ubatch * ubatch) {
    const int64_t n_tokens     = ubatch->n_tokens;
    const int64_t n_seqs_unq   = ubatch->n_seqs_unq;

    if (cparams.embeddings && (
        cparams.pooling_type == LLAMA_POOLING_TYPE_CLS  ||
        cparams.pooling_type == LLAMA_POOLING_TYPE_RANK ||
        cparams.pooling_type == LLAMA_POOLING_TYPE_LAST
    )) {
        GGML_ASSERT(cls);
        GGML_ASSERT(ggml_backend_buffer_is_host(cls->buffer));

        uint32_t * data = (uint32_t *) cls->data;
        memset(cls->data, 0, n_seqs_unq*ggml_element_size(cls));

        std::vector<int> target_pos(n_seqs_unq, -1);
        std::vector<int> target_row(n_seqs_unq, -1);

        const bool last = (
             cparams.pooling_type == LLAMA_POOLING_TYPE_LAST ||
            (cparams.pooling_type == LLAMA_POOLING_TYPE_RANK && arch == LLM_ARCH_QWEN3) // qwen3 reranking & embedding models use last token
        );

        for (int i = 0; i < n_tokens; ++i) {
            const llama_pos pos = ubatch->pos[i];

            for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                const llama_seq_id seq_id  = ubatch->seq_id[i][s];
                const int32_t      seq_idx = ubatch->seq_idx[seq_id];

                if (
                    (target_pos[seq_idx] == -1) ||
                    ( last && pos >= target_pos[seq_idx]) ||
                    (!last && pos <  target_pos[seq_idx])
                ) {
                    target_pos[seq_idx] = pos;
                    target_row[seq_idx] = i;
                }
            }
        }

        for (int s = 0; s < n_seqs_unq; ++s) {
            if (target_row[s] >= 0) {
                data[s] = target_row[s];
            }
        }
    }
}

void llm_graph_input_rs::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);

    const int64_t n_rs = mctx->get_n_rs();

    if (s_copy) {
        GGML_ASSERT(ggml_backend_buffer_is_host(s_copy->buffer));
        int32_t * data = (int32_t *) s_copy->data;

        // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
        for (uint32_t i = 0; i < n_rs; ++i) {
            data[i] = mctx->s_copy(i);
        }
    }
}

void llm_graph_input_cross_embd::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);

    if (cross_embd && !cross->v_embd.empty()) {
        assert(cross_embd->type == GGML_TYPE_F32);

        ggml_backend_tensor_set(cross_embd, cross->v_embd.data(), 0, ggml_nbytes(cross_embd));
    }
}

static void print_mask(const float * data, int64_t n_tokens, int64_t n_kv, int64_t n_swa, llama_swa_type swa_type) {
    LLAMA_LOG_DEBUG("%s: === Attention mask ===\n", __func__);
    const char * swa_type_str = "unknown";

    switch (swa_type) {
        case LLAMA_SWA_TYPE_NONE:      swa_type_str = "LLAMA_SWA_TYPE_NONE"; break;
        case LLAMA_SWA_TYPE_STANDARD:  swa_type_str = "LLAMA_SWA_TYPE_STANDARD"; break;
        case LLAMA_SWA_TYPE_CHUNKED:   swa_type_str = "LLAMA_SWA_TYPE_CHUNKED"; break;
        case LLAMA_SWA_TYPE_SYMMETRIC: swa_type_str = "LLAMA_SWA_TYPE_SYMMETRIC"; break;
    };

    LLAMA_LOG_DEBUG("%s: n_swa : %d, n_kv: %d, swq_type: %s\n", __func__, (int)n_swa, (int)n_kv, swa_type_str);
    LLAMA_LOG_DEBUG("%s: '0' = can attend, '∞' = masked\n", __func__);
    LLAMA_LOG_DEBUG("%s: Rows = query tokens, Columns = key/value tokens\n\n", __func__);

    LLAMA_LOG_DEBUG("    ");
    for (int j = 0; j < std::min((int64_t)20, n_kv); ++j) {
        LLAMA_LOG_DEBUG("%2d", j);
    }
    LLAMA_LOG_DEBUG("\n");

    for (int i = 0; i < std::min((int64_t)20, n_tokens); ++i) {
        LLAMA_LOG_DEBUG(" %2d ", i);
        for (int j = 0; j < std::min((int64_t)20, n_kv); ++j) {
            float val = data[i * n_kv + j];
            if (val == -INFINITY) {
                LLAMA_LOG_DEBUG(" ∞");
            } else {
                LLAMA_LOG_DEBUG(" 0");
            }
        }
        LLAMA_LOG_DEBUG("\n");
    }
}

void llm_graph_input_attn_no_cache::set_input(const llama_ubatch * ubatch) {
    const int64_t n_kv     = ubatch->n_tokens;
    const int64_t n_tokens = ubatch->n_tokens;

    const auto fill_mask = [&](float * data, int n_swa, llama_swa_type swa_type) {
        for (int h = 0; h < 1; ++h) {
            for (int i1 = 0; i1 < n_tokens; ++i1) {
                const llama_seq_id s1 = ubatch->seq_id[i1][0];
                const llama_pos    p1 = ubatch->pos[i1];

                const uint64_t idst = h*(n_kv*n_tokens) + i1*n_kv;

                for (int i0 = 0; i0 < n_tokens; ++i0) {
                    const llama_seq_id s0 = ubatch->seq_id[i0][0];
                    const llama_pos p0    = ubatch->pos[i0];

                    // mask different sequences
                    if (s0 != s1) {
                        continue;
                    }

                    // mask future tokens
                    if (cparams.causal_attn && p0 > p1) {
                        continue;
                    }

                    // apply SWA if any
                    if (llama_hparams::is_masked_swa(n_swa, swa_type, p0, p1)) {
                        continue;
                    }

                    data[idst + i0] = hparams.use_alibi ? -std::abs(p0 - p1) : 0.0f;
                }
            }
        }
    };

    {
        GGML_ASSERT(self_kq_mask);
        GGML_ASSERT(ggml_backend_buffer_is_host(self_kq_mask->buffer));

        float * data = (float *) self_kq_mask->data;

        std::fill(data, data + ggml_nelements(self_kq_mask), -INFINITY);

        fill_mask(data, 0, LLAMA_SWA_TYPE_NONE);

        if (debug) {
            print_mask(data, n_tokens, n_kv, 0, LLAMA_SWA_TYPE_NONE);
        }
    }

    if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
        GGML_ASSERT(self_kq_mask_swa);
        GGML_ASSERT(ggml_backend_buffer_is_host(self_kq_mask_swa->buffer));

        float * data = (float *) self_kq_mask_swa->data;

        std::fill(data, data + ggml_nelements(self_kq_mask_swa), -INFINITY);

        fill_mask(data, hparams.n_swa, hparams.swa_type);

        if (debug) {
            print_mask(data, n_tokens, n_kv, hparams.n_swa, hparams.swa_type);
        }
    }
}

void llm_graph_input_attn_kv::set_input(const llama_ubatch * ubatch) {
    if (self_k_idxs && self_k_idxs->buffer) {
        mctx->set_input_k_idxs(self_k_idxs, ubatch);
    }
    if (self_v_idxs && self_v_idxs->buffer) {
        mctx->set_input_v_idxs(self_v_idxs, ubatch);
    }

    if (self_kq_mask && self_kq_mask->buffer) {
        mctx->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);
    }
}

bool llm_graph_input_attn_kv::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;
  //res &= self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there

    res &= self_kq_mask->ne[0] == mctx->get_n_kv();
    res &= self_kq_mask->ne[1] == GGML_PAD(params.ubatch.n_tokens, GGML_KQ_MASK_PAD);

    return res;
}

void llm_graph_input_attn_kv_iswa::set_input(const llama_ubatch * ubatch) {
    mctx->get_base()->set_input_k_idxs(self_k_idxs, ubatch);
    mctx->get_base()->set_input_v_idxs(self_v_idxs, ubatch);

    mctx->get_base()->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);

    mctx->get_swa()->set_input_k_idxs(self_k_idxs_swa, ubatch);
    mctx->get_swa()->set_input_v_idxs(self_v_idxs_swa, ubatch);

    mctx->get_swa()->set_input_kq_mask(self_kq_mask_swa, ubatch, cparams.causal_attn);
}

bool llm_graph_input_attn_kv_iswa::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_iswa_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;
  //res &= self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there

    res &= self_k_idxs_swa->ne[0] == params.ubatch.n_tokens;
  //res &= self_v_idxs_swa->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there

    res &= self_kq_mask->ne[0] == mctx->get_base()->get_n_kv();
    res &= self_kq_mask->ne[1] == GGML_PAD(params.ubatch.n_tokens, GGML_KQ_MASK_PAD);

    res &= self_kq_mask_swa->ne[0] == mctx->get_swa()->get_n_kv();
    res &= self_kq_mask_swa->ne[1] == GGML_PAD(params.ubatch.n_tokens, GGML_KQ_MASK_PAD);

    return res;
}

void llm_graph_input_attn_cross::set_input(const llama_ubatch * ubatch) {
    GGML_ASSERT(cross_kq_mask);

    const int64_t n_enc    = cross_kq_mask->ne[0];
    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(ggml_backend_buffer_is_host(cross_kq_mask->buffer));
    GGML_ASSERT(!ubatch->equal_seqs()); // TODO: use ubatch->n_seqs instead of failing

    float * data = (float *) cross_kq_mask->data;

    for (int h = 0; h < 1; ++h) {
        for (int i = 0; i < n_tokens; ++i) {
            for (int j = 0; j < n_enc; ++j) {
                float f = -INFINITY;

                for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                    const llama_seq_id seq_id = ubatch->seq_id[i][s];

                    if (cross->seq_ids_enc[j].find(seq_id) != cross->seq_ids_enc[j].end()) {
                        f = 0.0f;
                    }
                }

                data[h*(n_enc*n_tokens) + i*n_enc + j] = f;
            }
        }

        for (int i = n_tokens; i < GGML_PAD(n_tokens, GGML_KQ_MASK_PAD); ++i) {
            for (int j = 0; j < n_enc; ++j) {
                data[h*(n_enc*n_tokens) + i*n_enc + j] = -INFINITY;
            }
        }
    }
}

void llm_graph_input_mem_hybrid::set_input(const llama_ubatch * ubatch) {
    inp_attn->set_input(ubatch);
    inp_rs->set_input(ubatch);
}

//
// llm_graph_result
//

llm_graph_result::llm_graph_result(int64_t max_nodes) : max_nodes(max_nodes) {
    reset();

    const char * LLAMA_GRAPH_RESULT_DEBUG = getenv("LLAMA_GRAPH_RESULT_DEBUG");
    debug = LLAMA_GRAPH_RESULT_DEBUG ? atoi(LLAMA_GRAPH_RESULT_DEBUG) : 0;
}

int64_t llm_graph_result::get_max_nodes() const {
    return max_nodes;
}

void llm_graph_result::reset() {
    t_tokens      = nullptr;
    t_logits      = nullptr;
    t_embd        = nullptr;
    t_embd_pooled = nullptr;

    params = {};

    inputs.clear();

    buf_compute_meta.resize(ggml_tensor_overhead()*max_nodes + ggml_graph_overhead_custom(max_nodes, false));

    ggml_init_params params = {
        /*.mem_size   =*/ buf_compute_meta.size(),
        /*.mem_buffer =*/ buf_compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    ctx_compute.reset(ggml_init(params));

    gf = ggml_new_graph_custom(ctx_compute.get(), max_nodes, false);
}

void llm_graph_result::set_inputs(const llama_ubatch * ubatch) {
    for (auto & input : inputs) {
        input->set_input(ubatch);
    }
}

bool llm_graph_result::can_reuse(const llm_graph_params & params) {
    if (!this->params.allow_reuse(params)) {
        if (debug > 1) {
            LLAMA_LOG_DEBUG("%s: cannot reuse graph due to incompatible graph parameters\n", __func__);
        }

        return false;
    }

    if (debug > 1) {
        LLAMA_LOG_DEBUG("%s: checking compatibility of %d inputs:\n", __func__, (int) inputs.size());
    }

    bool res = true;

    for (auto & input : inputs) {
        const bool cur = input->can_reuse(params);

        if (debug > 1) {
            LLAMA_LOG_DEBUG("%s: can_reuse = %d\n", "placeholder", cur);
        }

        res = res && cur;
    }

    if (debug > 0) {
        LLAMA_LOG_DEBUG("%s: can reuse graph = %d\n", __func__, res);
    }

    return res;
}

llm_graph_input_i * llm_graph_result::add_input(llm_graph_input_ptr input) {
    inputs.emplace_back(std::move(input));
    return inputs.back().get();
}

void llm_graph_result::set_params(const llm_graph_params & params) {
    this->params = params;
}

//
// llm_graph_context
//

llm_graph_context::llm_graph_context(const llm_graph_params & params) :
    model            (*params.model),
    arch             (params.arch),
    hparams          (params.hparams),
    cparams          (params.cparams),
    ubatch           (params.ubatch),
    n_embd           (hparams.n_embd),
    n_layer          (hparams.n_layer),
    n_rot            (hparams.n_rot),
    n_ctx            (cparams.n_ctx),
    n_head           (hparams.n_head()),
    n_head_kv        (hparams.n_head_kv()),
    n_embd_head_k    (hparams.n_embd_head_k),
    n_embd_k_gqa     (hparams.n_embd_k_gqa()),
    n_embd_head_v    (hparams.n_embd_head_v),
    n_embd_v_gqa     (hparams.n_embd_v_gqa()),
    n_expert         (hparams.n_expert),
    n_expert_used    (cparams.warmup ? hparams.n_expert : hparams.n_expert_used),
    freq_base        (cparams.rope_freq_base),
    freq_scale       (cparams.rope_freq_scale),
    ext_factor       (cparams.yarn_ext_factor),
    attn_factor      (cparams.yarn_attn_factor),
    beta_fast        (cparams.yarn_beta_fast),
    beta_slow        (cparams.yarn_beta_slow),
    norm_eps         (hparams.f_norm_eps),
    norm_rms_eps     (hparams.f_norm_rms_eps),
    n_tokens         (ubatch.n_tokens),
    n_outputs        (params.n_outputs),
    n_ctx_orig       (cparams.n_ctx_orig_yarn),
    pooling_type     (cparams.pooling_type),
    rope_type        (hparams.rope_type),
    sched            (params.sched),
    backend_cpu      (params.backend_cpu),
    cvec             (params.cvec),
    loras            (params.loras),
    mctx             (params.mctx),
    cross            (params.cross),
    cb_func          (params.cb),
    res              (params.res),
    ctx0             (res->get_ctx()),
    gf               (res->get_gf()) {
        res->set_params(params);
    }

void llm_graph_context::cb(ggml_tensor * cur, const char * name, int il) const {
    if (cb_func) {
        cb_func(ubatch, cur, name, il);
    }
}

ggml_tensor * llm_graph_context::build_cvec(
         ggml_tensor * cur,
                 int   il) const {
    return cvec->apply_to(ctx0, cur, il);
}

ggml_tensor * llm_graph_context::build_lora_mm(
          ggml_tensor * w,
          ggml_tensor * cur,
          decode_awq_output_type /*decode_awq_out*/) const {
    cur = llama_maybe_observe_text_activation(ctx0, cur, w->name, cur->ne[1] > 1);
    ggml_tensor * res = nullptr;

    const bool is_prefill_gemm = cur->ne[1] > 1;
    const bool is_decode_step = n_tokens == 1;
    const bool is_prefill_tail_lm_head =
        n_tokens > 1 &&
        n_outputs > 0 &&
        std::strcmp(w->name, "output.weight") == 0;
    const bool prefill_tail_decode_switch =
        is_prefill_tail_lm_head &&
        llama_prefill_tail_decode_switch_enabled();
    const bool use_decode_gemv_sq = !is_prefill_gemm && llama_text_sq_enable_decode_gemv();
    const bool allow_text_sq = is_prefill_gemm || use_decode_gemv_sq;
    auto it_sq = model.aicas_text_sq_tensors.find(w->name);
    if (model.aicas_text_sq_enabled &&
        allow_text_sq &&
        it_sq != model.aicas_text_sq_tensors.end()) {
        const auto & cfg = it_sq->second;
        if (cfg.enabled &&
            cfg.policy == "W8A8" &&
            cfg.quant_tensor != nullptr &&
            cfg.act_scale > 0.0f &&
            cur->type == GGML_TYPE_F32 &&
            cfg.quant_tensor->type == GGML_TYPE_I8 &&
            cur->ne[0] == cfg.quant_tensor->ne[0] &&
            cfg.smooth_scale.size() == static_cast<size_t>(cur->ne[0]) &&
            cfg.weight_scale.size() == cfg.expected_weight_scale_len(cfg.quant_tensor->ne[1])) {
            const llama_npu_shape_key npu_shape_key {
                ggml_get_name(cfg.quant_tensor),
                cfg.quant_tensor->ne[1],
                cur->ne[1],
                cur->ne[0],
                false,
            };
            if (is_prefill_gemm) {
                llama_npu_record_shape(npu_shape_key, "text_prefill_w8a8", cfg);
            }
            const bool npu_shape_hit = is_prefill_gemm && llama_npu_shape_table_hit(npu_shape_key);
            const bool npu_dynamic = is_prefill_gemm && !npu_shape_hit && llama_npu_text_prefill_dynamic_enabled();
            if ((npu_shape_hit || npu_dynamic) && llama_register_text_w8a8_for_npu(npu_shape_key, cfg)) {
                ggml_tensor * npu_cur = cur;
                if (!ggml_is_contiguous(npu_cur) || npu_cur->view_src != nullptr) {
                    npu_cur = ggml_cont(ctx0, npu_cur);
                    ggml_set_name(npu_cur, "text_w8a8_npu_cont");
                }
                res = ggml_mul_mat(ctx0, cfg.quant_tensor, npu_cur);
                ggml_set_name(res, npu_shape_hit ? "text_w8a8_mul_mat_npu" : "text_w8a8_mul_mat_npu_dynamic");
            } else {
                ggml_tensor * out_template = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, cfg.quant_tensor->ne[1], cur->ne[1]);
                res = ggml_map_custom3(
                    ctx0,
                    out_template,
                    cur,
                    cfg.quant_tensor,
                    llama_compute_text_w8a8_mul_mat,
                    GGML_N_TASKS_MAX,
                    const_cast<llama_aicas_text_sq_tensor *>(&cfg));
            }
        }
    }

    if (res == nullptr &&
        !is_prefill_gemm &&
        is_decode_step &&
        llama_text_decode_awq_enabled()) {
        auto it_awq = model.aicas_text_decode_awq_tensors.find(w->name);
        const bool is_lm_head_weight =
            std::strcmp(w->name, "output.weight") == 0 &&
            w->type == GGML_TYPE_F16;
        static bool lm_head_q4_disabled_logged = false;
        if (is_lm_head_weight &&
            it_awq != model.aicas_text_decode_awq_tensors.end() &&
            !lm_head_q4_disabled_logged) {
            LLAMA_LOG_INFO("%s: Q4_AWQ lm_head metadata found for output.weight but runtime path is disabled; using FP16/default or explicit W8/W16 hook\n",
                    __func__);
            lm_head_q4_disabled_logged = true;
        }
        if (!is_lm_head_weight &&
            model.aicas_text_decode_awq_enabled &&
            it_awq != model.aicas_text_decode_awq_tensors.end()) {
            const auto & cfg = it_awq->second;
            ggml_tensor * cur_awq = cur;
            // For decode GEMV, quantize activation to FP16 before entering AWQ kernel when source is FP32.
            if (cur_awq->type == GGML_TYPE_F32) {
                cur_awq = ggml_cast(ctx0, cur_awq, GGML_TYPE_F16);
            }
            if (cfg.enabled &&
                cfg.policy == "Q4_AWQ" &&
                cfg.quant_tensor != nullptr &&
                cfg.scale_tensor != nullptr &&
                cfg.zero_tensor != nullptr &&
                (cur_awq->type == GGML_TYPE_F16 || cur_awq->type == GGML_TYPE_F32) &&
                cfg.quant_tensor->type == GGML_TYPE_I8 &&
                cfg.in_features > 0 &&
                cur_awq->ne[0] == cfg.in_features &&
                cfg.packed_in_features() == cfg.quant_tensor->ne[0] &&
                cfg.has_valid_smooth_config() &&
                cfg.has_valid_group_params(cfg.quant_tensor->ne[1])) {
                ggml_tensor * out_template = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, cfg.quant_tensor->ne[1], cur_awq->ne[1]);
                bool decode_awq_npu = false;
#ifdef GGML_USE_NPU
                decode_awq_npu =
                    llama_text_decode_awq_npu_enabled() &&
                    !llama_text_decode_awq_int24_enabled() &&
                    llama_text_decode_awq_int24_diag_path().empty();
#endif
                res = ggml_map_custom3(
                    ctx0,
                    out_template,
                    cur_awq,
                    cfg.quant_tensor,
#ifdef GGML_USE_NPU
                    decode_awq_npu ? llama_compute_text_decode_awq_npu_mul_mat : llama_compute_text_decode_awq_mul_mat,
#else
                    llama_compute_text_decode_awq_mul_mat,
#endif
                    decode_awq_npu ? 1 : GGML_N_TASKS_MAX,
                    const_cast<llama_aicas_text_decode_awq_tensor *>(&cfg));
                ggml_set_name(res, decode_awq_npu ? "text_decode_awq_w4a16_npu" : "text_decode_awq_cpu");
            }
        }
    }

    if (res == nullptr &&
        !(is_prefill_tail_lm_head && llama_text_prefill_lm_head_default_enabled() && !prefill_tail_decode_switch) &&
        llama_text_lm_head_w8a16_dp128_enabled() &&
        std::strcmp(w->name, "output.weight") == 0 &&
        w->type == GGML_TYPE_F16 &&
        (cur->type == GGML_TYPE_F16 || cur->type == GGML_TYPE_F32) &&
        cur->ne[0] == w->ne[0]) {
        ggml_tensor * cur_lm_head = cur;
        if (cur_lm_head->type == GGML_TYPE_F32) {
            cur_lm_head = ggml_cast(ctx0, cur_lm_head, GGML_TYPE_F16);
        }
        static llama_text_lm_head_w8a16_userdata lm_head_w8a16_cfg;
        lm_head_w8a16_cfg.tile = llama_text_lm_head_w8a16_group();
        const bool lm_head_w8a16_npu =
            llama_text_lm_head_w8a16_npu_enabled() &&
            !llama_text_lm_head_w8a16_int24_enabled() &&
            llama_text_lm_head_w8a16_int24_diag_path().empty();
        if (prefill_tail_decode_switch && lm_head_w8a16_npu) {
            cur_lm_head = ggml_map_custom1(
                    ctx0,
                    cur_lm_head,
                    llama_compute_prefill_tail_decode_switch_passthrough,
                    1,
                    nullptr);
            ggml_set_name(cur_lm_head, "prefill_tail_decode_switch_barrier");
            ggml_backend_sched_set_tensor_backend(sched, cur_lm_head, backend_cpu);
        }
        ggml_tensor * out_template = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, w->ne[1], cur_lm_head->ne[1]);
        static bool lm_head_w8a16_logged = false;
        if (!lm_head_w8a16_logged) {
            LLAMA_LOG_INFO("%s: using AICAS lm_head W8A16 %s path for output.weight; group=%" PRId64 "; Q4 lm_head disabled\n",
                    __func__, lm_head_w8a16_npu ? "NPU" : "DP128", lm_head_w8a16_cfg.tile);
            lm_head_w8a16_logged = true;
        }
        res = ggml_map_custom3(
            ctx0,
            out_template,
            cur_lm_head,
            w,
            lm_head_w8a16_npu ?
                llama_compute_text_lm_head_w8a16_npu_mul_mat :
                llama_compute_text_lm_head_w8a16_dp128_mul_mat,
            lm_head_w8a16_npu ? 1 : GGML_N_TASKS_MAX,
            &lm_head_w8a16_cfg);
        ggml_set_name(res, lm_head_w8a16_npu ? "text_lm_head_w8a16_npu" : "text_lm_head_w8a16_dp128");
    }

    if (res == nullptr &&
        is_prefill_tail_lm_head &&
        llama_text_prefill_lm_head_default_enabled() &&
        !prefill_tail_decode_switch) {
        static bool logged = false;
        if (!logged) {
            LLAMA_LOG_INFO("%s: using default CPU FP lm_head for prefill-tail output.weight; AICAS W8A16 lm_head and decode overlay are not requested\n",
                    __func__);
            logged = true;
        }
        res = ggml_mul_mat(ctx0, w, cur);
        ggml_set_name(res, "text_prefill_lm_head_default_cpu");
        ggml_backend_sched_set_tensor_backend(sched, res, backend_cpu);
    }

    if (false && res == nullptr &&
        !is_prefill_gemm &&
        llama_text_lm_head_w16a16_dp128_enabled() &&
        std::strcmp(w->name, "output.weight") == 0 &&
        w->type == GGML_TYPE_F16 &&
        (cur->type == GGML_TYPE_F16 || cur->type == GGML_TYPE_F32) &&
        cur->ne[0] == w->ne[0]) {
        ggml_tensor * cur_lm_head = cur;
        if (cur_lm_head->type == GGML_TYPE_F32) {
            cur_lm_head = ggml_cast(ctx0, cur_lm_head, GGML_TYPE_F16);
        }
        ggml_tensor * out_template = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, w->ne[1], cur_lm_head->ne[1]);
        static llama_text_lm_head_w16a16_userdata lm_head_w16a16_cfg;
        static bool lm_head_accum_warned = false;
        static bool lm_head_w16a16_logged = false;
        if (!lm_head_w16a16_logged) {
            LLAMA_LOG_INFO("%s: using AICAS lm_head W16A16 DP128 path for output.weight; Q4 lm_head disabled\n",
                    __func__);
            lm_head_w16a16_logged = true;
        }
        if (llama_text_lm_head_w16a16_accum_fp16() && !lm_head_accum_warned) {
            LLAMA_LOG_WARN("%s: AICAS_TEXT_LM_HEAD_W16A16_ACCUM=fp16 is ignored by RTL DP128 lm_head simulation\n",
                    __func__);
            lm_head_accum_warned = true;
        }
        res = ggml_map_custom3(
            ctx0,
            out_template,
            cur_lm_head,
            w,
            llama_compute_text_lm_head_w16a16_dp128_mul_mat,
            GGML_N_TASKS_MAX,
            &lm_head_w16a16_cfg);
        ggml_set_name(res, "text_lm_head_w16a16_dp128");
    }

    if (res == nullptr) {
        res = ggml_mul_mat(ctx0, w, cur);
    }

    for (const auto & lora : *loras) {
        llama_adapter_lora_weight * lw = lora.first->get_weight(w);
        if (lw == nullptr) {
            continue;
        }

        const float adapter_scale = lora.second;
        const float scale = lw->get_scale(lora.first->alpha, adapter_scale);

        ggml_tensor * ab_cur = ggml_mul_mat(
                ctx0, lw->b,
                ggml_mul_mat(ctx0, lw->a, cur)
                );

        ab_cur = ggml_scale(ctx0, ab_cur, scale);
        res = ggml_add(ctx0, res, ab_cur);
    }

    return res;
}

ggml_tensor * llm_graph_context::build_lora_mm_id(
          ggml_tensor * w,   // ggml_tensor * as
          ggml_tensor * cur, // ggml_tensor * b
          ggml_tensor * ids) const {
    ggml_tensor * res = ggml_mul_mat_id(ctx0, w, cur, ids);
    for (const auto & lora : *loras) {
        llama_adapter_lora_weight * lw = lora.first->get_weight(w);
        if (lw == nullptr) {
            continue;
        }

        const float alpha = lora.first->alpha;
        const float rank  = (float) lw->b->ne[0];
        const float scale = alpha ? lora.second * alpha / rank : lora.second;

        ggml_tensor * ab_cur = ggml_mul_mat_id(
                ctx0, lw->b,
                ggml_mul_mat_id(ctx0, lw->a, cur, ids),
                ids
                );

        ab_cur = ggml_scale(ctx0, ab_cur, scale);
        res = ggml_add(ctx0, res, ab_cur);
    }

    return res;
}

ggml_tensor * llm_graph_context::build_norm(
         ggml_tensor * cur,
         ggml_tensor * mw,
         ggml_tensor * mb,
       llm_norm_type   type,
                 int   il) const {
    switch (type) {
        case LLM_NORM:       cur = ggml_norm    (ctx0, cur, hparams.f_norm_eps);     break;
        case LLM_NORM_RMS:   cur = ggml_rms_norm(ctx0, cur, hparams.f_norm_rms_eps); break;
        case LLM_NORM_GROUP:
            {
                cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], 1, cur->ne[1]);
                cur = ggml_group_norm(ctx0, cur, hparams.n_norm_groups, hparams.f_norm_group_eps);
                cur = ggml_reshape_2d(ctx0, cur, cur->ne[0],    cur->ne[2]);
            } break;
    }

    if (mw || mb) {
        cb(cur, "norm", il);
    }

    if (mw) {
        cur = ggml_mul(ctx0, cur, mw);
        if (mb) {
            cb(cur, "norm_w", il);
        }
    }

    if (mb) {
        cur = ggml_add(ctx0, cur, mb);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_ffn(
         ggml_tensor * cur,
         ggml_tensor * up,
         ggml_tensor * up_b,
         ggml_tensor * up_s,
         ggml_tensor * gate,
         ggml_tensor * gate_b,
         ggml_tensor * gate_s,
         ggml_tensor * down,
         ggml_tensor * down_b,
         ggml_tensor * down_s,
         ggml_tensor * act_scales,
     llm_ffn_op_type   type_op,
   llm_ffn_gate_type   type_gate,
                int   il) const {
#ifdef GGML_USE_NPU
    if (n_tokens == 1 &&
            cur->ne[1] == 1 &&
            type_op == LLM_FFN_SILU &&
            type_gate == LLM_FFN_PAR &&
            up != nullptr &&
            gate != nullptr &&
            down != nullptr &&
            up_b == nullptr &&
            up_s == nullptr &&
            gate_b == nullptr &&
            gate_s == nullptr &&
            down_b == nullptr &&
            down_s == nullptr &&
            act_scales == nullptr &&
            llama_text_decode_awq_enabled() &&
            llama_text_decode_awq_npu_enabled() &&
            llama_text_decode_awq_fused_ffn_npu_enabled()) {
        auto it_gate = model.aicas_text_decode_awq_tensors.find(gate->name);
        auto it_up   = model.aicas_text_decode_awq_tensors.find(up->name);
        auto it_down = model.aicas_text_decode_awq_tensors.find(down->name);
        if (it_gate != model.aicas_text_decode_awq_tensors.end() &&
                it_up != model.aicas_text_decode_awq_tensors.end() &&
                it_down != model.aicas_text_decode_awq_tensors.end()) {
            const llama_aicas_text_decode_awq_tensor & gate_cfg = it_gate->second;
            const llama_aicas_text_decode_awq_tensor & up_cfg   = it_up->second;
            const llama_aicas_text_decode_awq_tensor & down_cfg = it_down->second;
            const bool valid =
                gate_cfg.enabled && up_cfg.enabled && down_cfg.enabled &&
                gate_cfg.policy == "Q4_AWQ" && up_cfg.policy == "Q4_AWQ" && down_cfg.policy == "Q4_AWQ" &&
                gate_cfg.group_size == 128 && up_cfg.group_size == 128 && down_cfg.group_size == 128 &&
                gate_cfg.quant_tensor != nullptr && up_cfg.quant_tensor != nullptr && down_cfg.quant_tensor != nullptr &&
                gate_cfg.scale_tensor != nullptr && up_cfg.scale_tensor != nullptr && down_cfg.scale_tensor != nullptr &&
                gate_cfg.zero_tensor != nullptr && up_cfg.zero_tensor != nullptr && down_cfg.zero_tensor != nullptr &&
                gate_cfg.quant_tensor->type == GGML_TYPE_I8 &&
                up_cfg.quant_tensor->type == GGML_TYPE_I8 &&
                down_cfg.quant_tensor->type == GGML_TYPE_I8 &&
                gate_cfg.in_features > 0 && up_cfg.in_features > 0 && down_cfg.in_features > 0 &&
                gate_cfg.packed_in_features() == gate_cfg.quant_tensor->ne[0] &&
                up_cfg.packed_in_features() == up_cfg.quant_tensor->ne[0] &&
                down_cfg.packed_in_features() == down_cfg.quant_tensor->ne[0] &&
                gate_cfg.has_valid_smooth_config() &&
                up_cfg.has_valid_smooth_config() &&
                down_cfg.has_valid_smooth_config() &&
                gate_cfg.has_valid_group_params(gate_cfg.quant_tensor->ne[1]) &&
                up_cfg.has_valid_group_params(up_cfg.quant_tensor->ne[1]) &&
                down_cfg.has_valid_group_params(down_cfg.quant_tensor->ne[1]) &&
                gate_cfg.in_features == cur->ne[0] &&
                up_cfg.in_features == cur->ne[0] &&
                gate_cfg.quant_tensor->ne[1] == up_cfg.quant_tensor->ne[1] &&
                down_cfg.in_features == gate_cfg.quant_tensor->ne[1];
            if (valid) {
                if (llama_text_decode_awq_fused_ffn_npu_log_enabled()) {
                    LLAMA_LOG_INFO(
                            "%s: using AICAS text decode SwiGLU FFN NPU path layer=%d hidden=%" PRId64
                            " intermediate=%" PRId64 " gate=%s up=%s down=%s\n",
                            __func__, il, cur->ne[0], gate_cfg.quant_tensor->ne[1],
                            gate->name, up->name, down->name);
                }

                ggml_tensor * npu_cur = cur;
                if (!ggml_is_contiguous(npu_cur) || npu_cur->view_src != nullptr) {
                    npu_cur = ggml_cont(ctx0, npu_cur);
                    ggml_set_name(npu_cur, "ffn_swiglu_npu_cont");
                }

                static std::mutex userdata_mutex;
                static std::vector<std::unique_ptr<llama_text_decode_swiglu_ffn_npu_userdata>> userdata_store;
                auto node_cfg = std::make_unique<llama_text_decode_swiglu_ffn_npu_userdata>();
                node_cfg->op_name = "blk." + std::to_string(il) + ".ffn_swiglu_npu";
                node_cfg->gate = llama_make_npu_decode_awq_view(gate_cfg);
                node_cfg->up = llama_make_npu_decode_awq_view(up_cfg);
                node_cfg->down = llama_make_npu_decode_awq_view(down_cfg);
                llama_text_decode_swiglu_ffn_npu_userdata * ptr = node_cfg.get();
                {
                    std::lock_guard<std::mutex> lock(userdata_mutex);
                    userdata_store.push_back(std::move(node_cfg));
                }

                ggml_tensor * args[] = {
                    npu_cur,
                    gate_cfg.quant_tensor,
                    up_cfg.quant_tensor,
                    down_cfg.quant_tensor,
                };
                ggml_tensor * fused = ggml_custom_4d(
                        ctx0,
                        GGML_TYPE_F32,
                        down_cfg.quant_tensor->ne[1],
                        npu_cur->ne[1],
                        1,
                        1,
                        args,
                        4,
                        llama_compute_text_decode_swiglu_ffn_npu,
                        1,
                        ptr);
                ggml_set_name(fused, "text_decode_swiglu_ffn_npu");
                cb(fused, "ffn_swiglu_npu", il);
                return fused;
            }
        }
    }
#endif

    ggml_tensor * tmp = up ? build_lora_mm(up, cur) : cur;
    cb(tmp, "ffn_up", il);

    if (up_b) {
        tmp = ggml_add(ctx0, tmp, up_b);
        cb(tmp, "ffn_up_b", il);
    }

    if (up_s) {
        tmp = ggml_mul(ctx0, tmp, up_s);
        cb(tmp, "ffn_up_s", il);
    }

    if (gate) {
        switch (type_gate) {
            case LLM_FFN_SEQ:
                {
                    cur = build_lora_mm(gate, tmp);
                    cb(cur, "ffn_gate", il);
                } break;
            case LLM_FFN_PAR:
                {
                    cur = build_lora_mm(gate, cur);
                    cb(cur, "ffn_gate", il);
                } break;
        }

        if (gate_b) {
            cur = ggml_add(ctx0, cur, gate_b);
            cb(cur, "ffn_gate_b", il);
        }

        if (gate_s) {
            cur = ggml_mul(ctx0, cur, gate_s);
            cb(cur, "ffn_gate_s", il);
        }

    } else {
        cur = tmp;
    }

    switch (type_op) {
        case LLM_FFN_SILU:
            if (gate && type_gate == LLM_FFN_PAR) {
                const std::string swiglu_input_name = "blk." + std::to_string(il) + ".ffn_swiglu_silu_input";
                cur = llama_maybe_observe_text_activation(ctx0, cur, swiglu_input_name.c_str(), cur->ne[1] > 1);
                cur = ggml_swiglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_swiglu", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                const std::string silu_input_name = "blk." + std::to_string(il) + ".ffn_silu_input";
                cur = llama_maybe_observe_text_activation(ctx0, cur, silu_input_name.c_str(), cur->ne[1] > 1);
                cur = ggml_silu(ctx0, cur);
                cb(cur, "ffn_silu", il);
            } break;
        case LLM_FFN_GELU:
            if (gate && type_gate == LLM_FFN_PAR) {
                cur = ggml_geglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_geglu", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                cur = ggml_gelu(ctx0, cur);
                cb(cur, "ffn_gelu", il);
                if (act_scales != NULL) {
                    cur = ggml_div(ctx0, cur, act_scales);
                    cb(cur, "ffn_act", il);
                }
            } break;
        case LLM_FFN_RELU:
            if (gate && type_gate == LLM_FFN_PAR) {
                cur = ggml_reglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_reglu", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                cur = ggml_relu(ctx0, cur);
                cb(cur, "ffn_relu", il);
            } break;
        case LLM_FFN_RELU_SQR:
            {
                cur = ggml_relu(ctx0, cur);
                cb(cur, "ffn_relu", il);

                cur = ggml_sqr(ctx0, cur);
                cb(cur, "ffn_sqr(relu)", il);
            } break;
        case LLM_FFN_SWIGLU:
            {
                cur = ggml_swiglu(ctx0, cur);
                cb(cur, "ffn_swiglu", il);
            } break;
        case LLM_FFN_GEGLU:
            {
                cur = ggml_geglu(ctx0, cur);
                cb(cur, "ffn_geglu", il);
            } break;
        case LLM_FFN_REGLU:
            {
                cur = ggml_reglu(ctx0, cur);
                cb(cur, "ffn_reglu", il);
            } break;
        default:
            GGML_ABORT("fatal error");
    }

    if (gate && type_gate == LLM_FFN_PAR) {
        cur = ggml_mul(ctx0, cur, tmp);
        cb(cur, "ffn_gate_par", il);
    }

    if (down) {
        cur = build_lora_mm(down, cur);
        if (arch == LLM_ARCH_GLM4 || arch == LLM_ARCH_GLM4_MOE) {
            // GLM4 and GLM4_MOE seem to have numerical issues with half-precision accumulators
            ggml_mul_mat_set_prec(cur, GGML_PREC_F32);
        }
    }

    if (down_b) {
        cb(cur, "ffn_down", il);
    }

    if (down_b) {
        cur = ggml_add(ctx0, cur, down_b);
    }

    if (down_s) {
        cur = ggml_mul(ctx0, cur, down_s);
        cb(cur, "ffn_down_s", il);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_moe_ffn(
         ggml_tensor * cur,
         ggml_tensor * gate_inp,
         ggml_tensor * up_exps,
         ggml_tensor * gate_exps,
         ggml_tensor * down_exps,
         ggml_tensor * exp_probs_b,
             int64_t   n_expert,
             int64_t   n_expert_used,
     llm_ffn_op_type   type_op,
                bool   norm_w,
                bool   scale_w,
               float   w_scale,
         llama_expert_gating_func_type gating_op,
                 int   il,
         ggml_tensor * probs_in) const {
    return build_moe_ffn(
        cur,
        gate_inp,  /* gate_inp_b  */ nullptr,
        up_exps,   /* up_exps_b   */ nullptr,
        gate_exps, /* gate_exps_b */ nullptr,
        down_exps, /* down_exps_b */ nullptr,
        exp_probs_b,
        n_expert,
        n_expert_used,
        type_op,
        norm_w,
        scale_w,
        w_scale,
        gating_op,
        il,
        probs_in
    );
}

ggml_tensor * llm_graph_context::build_moe_ffn(
         ggml_tensor * cur,
         ggml_tensor * gate_inp,
         ggml_tensor * gate_inp_b,
         ggml_tensor * up_exps,
         ggml_tensor * up_exps_b,
         ggml_tensor * gate_exps,
         ggml_tensor * gate_exps_b,
         ggml_tensor * down_exps,
         ggml_tensor * down_exps_b,
         ggml_tensor * exp_probs_b,
             int64_t   n_expert,
             int64_t   n_expert_used,
     llm_ffn_op_type   type_op,
                bool   norm_w,
                bool   scale_w,
               float   w_scale,
        llama_expert_gating_func_type gating_op,
                 int   il,
         ggml_tensor * probs_in) const {
    const int64_t n_embd   = cur->ne[0];
    const int64_t n_tokens = cur->ne[1];
    const bool weight_before_ffn = arch == LLM_ARCH_LLAMA4; // for llama4, we apply the sigmoid-ed weights before the FFN

    ggml_tensor * logits = nullptr;

    if (probs_in == nullptr) {
        logits = build_lora_mm(gate_inp, cur); // [n_expert, n_tokens]
        cb(logits, "ffn_moe_logits", il);
    } else {
        logits = probs_in;
    }

    if (gate_inp_b) {
        logits = ggml_add(ctx0, logits, gate_inp_b);
        cb(logits, "ffn_moe_logits_biased", il);
    }

    ggml_tensor * probs = nullptr;
    switch (gating_op) {
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX:
            {
                probs = ggml_soft_max(ctx0, logits); // [n_expert, n_tokens]
            } break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID:
            {
                probs = ggml_sigmoid(ctx0, logits); // [n_expert, n_tokens]
            } break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT:
            {
                probs = logits; // [n_expert, n_tokens]
            } break;
        default:
            GGML_ABORT("fatal error");
    }
    cb(probs, "ffn_moe_probs", il);

    // add experts selection bias - introduced in DeepSeek V3
    // leave probs unbiased as it's later used to get expert weights
    ggml_tensor * selection_probs = probs;
    if (exp_probs_b != nullptr) {
        selection_probs = ggml_add(ctx0, probs, exp_probs_b);
        cb(selection_probs, "ffn_moe_probs_biased", il);
    }

    // llama4 doesn't have exp_probs_b, and sigmoid is only used after top_k
    // see: https://github.com/meta-llama/llama-models/blob/699a02993512fb36936b1b0741e13c06790bcf98/models/llama4/moe.py#L183-L198
    if (arch == LLM_ARCH_LLAMA4) {
        selection_probs = logits;
    }

    if (arch == LLM_ARCH_GROVEMOE) {
        selection_probs = ggml_sigmoid(ctx0, logits); // [n_expert, n_tokens]
        cb(selection_probs, "ffn_moe_probs_biased", il);
    }

    // select top n_group_used expert groups
    // https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/e815299b0bcbac849fa540c768ef21845365c9eb/modeling_deepseek.py#L440-L457
    if (hparams.n_expert_groups > 1 && n_tokens > 0) {
        const int64_t n_exp_per_group = n_expert / hparams.n_expert_groups;

        // organize experts into n_expert_groups
        ggml_tensor * selection_groups = ggml_reshape_3d(ctx0, selection_probs, n_exp_per_group, hparams.n_expert_groups, n_tokens); // [n_exp_per_group, n_expert_groups, n_tokens]

        ggml_tensor * group_scores = ggml_top_k(ctx0, selection_groups, 2); // [2, n_expert_groups, n_tokens]
        group_scores = ggml_get_rows(ctx0, ggml_reshape_4d(ctx0, selection_groups, 1, selection_groups->ne[0], selection_groups->ne[1], selection_groups->ne[2]), group_scores); // [1, 2, n_expert_groups, n_tokens]

        // get top n_group_used expert groups
        group_scores = ggml_sum_rows(ctx0, ggml_reshape_3d(ctx0, group_scores, group_scores->ne[1], group_scores->ne[2], group_scores->ne[3])); // [1, n_expert_groups, n_tokens]
        group_scores = ggml_reshape_2d(ctx0, group_scores, group_scores->ne[1], group_scores->ne[2]); // [n_expert_groups, n_tokens]

        ggml_tensor * expert_groups = ggml_top_k(ctx0, group_scores, hparams.n_group_used); // [n_group_used, n_tokens]
        cb(expert_groups, "ffn_moe_group_topk", il);

        // mask out the other groups
        selection_probs = ggml_get_rows(ctx0, selection_groups, expert_groups); // [n_exp_per_group, n_group_used, n_tokens]
        selection_probs = ggml_set_rows(ctx0, ggml_scale_bias(ctx0, selection_groups, 0.0f, -INFINITY), selection_probs, expert_groups); // [n_exp_per_group, n_expert_groups, n_tokens]
        selection_probs = ggml_reshape_2d(ctx0, selection_probs, n_expert, n_tokens); // [n_expert, n_tokens]
        cb(selection_probs, "ffn_moe_probs_masked", il);
    }

    // select experts
    ggml_tensor * selected_experts = ggml_top_k(ctx0, selection_probs, n_expert_used); // [n_expert_used, n_tokens]
    cb(selected_experts->src[0], "ffn_moe_argsort", il);
    cb(selected_experts, "ffn_moe_topk", il);

    if (arch == LLM_ARCH_GROVEMOE && n_expert != hparams.n_expert) {
        // TODO: Use scalar div instead when/if implemented
        ggml_tensor * f_sel = ggml_cast(ctx0, selected_experts, GGML_TYPE_F32);
        selected_experts = ggml_cast(ctx0, ggml_scale(ctx0, f_sel, 1.0f / float(hparams.n_group_experts)), GGML_TYPE_I32);
        probs = ggml_reshape_3d(ctx0, probs, 1, hparams.n_expert, n_tokens);
    } else {
        probs = ggml_reshape_3d(ctx0, probs, 1, n_expert, n_tokens);
    }

    ggml_tensor * weights = ggml_get_rows(ctx0, probs, selected_experts); // [1, n_expert_used, n_tokens]
    cb(weights, "ffn_moe_weights", il);


    if (gating_op == LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT) {
        weights = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens);
        weights = ggml_soft_max(ctx0, weights); // [n_expert_used, n_tokens]
        weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_tokens);
        cb(weights, "ffn_moe_weights_softmax", il);
    }

    if (norm_w) {
        weights = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens);

        ggml_tensor * weights_sum = ggml_sum_rows(ctx0, weights); // [1, n_tokens]
        cb(weights_sum, "ffn_moe_weights_sum", il);

        if (arch == LLM_ARCH_BAILINGMOE2) {
            weights_sum = ggml_scale_bias(ctx0, weights_sum, 1.0, 1e-20);
            cb(weights_sum, "ffn_moe_weights_sum_biased", il);
        }

        weights = ggml_div(ctx0, weights, weights_sum); // [n_expert_used, n_tokens]
        cb(weights, "ffn_moe_weights_norm", il);

        weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_tokens);
    }
    if (scale_w) {
        weights = ggml_scale(ctx0, weights, w_scale);
        cb(weights, "ffn_moe_weights_scaled", il);
    }

    //call early so that topk-moe can be used
    ggml_build_forward_expand(gf, weights);

    cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);

    if (weight_before_ffn) {
        // repeat cur to [n_embd, n_expert_used, n_tokens]
        ggml_tensor * repeated = ggml_repeat_4d(ctx0, cur, n_embd, n_expert_used, n_tokens, 1);
        cur = ggml_mul(ctx0, repeated, weights);
        cb(cur, "ffn_moe_weighted", il);
    }

    ggml_tensor * up = build_lora_mm_id(up_exps, cur, selected_experts); // [n_ff, n_expert_used, n_tokens]
    cb(up, "ffn_moe_up", il);

    if (up_exps_b) {
        up = ggml_add_id(ctx0, up, up_exps_b, selected_experts);
        cb(up, "ffn_moe_up_biased", il);
    }

    ggml_tensor * experts = nullptr;
    if (gate_exps) {
        cur = build_lora_mm_id(gate_exps, cur, selected_experts); // [n_ff, n_expert_used, n_tokens]
        cb(cur, "ffn_moe_gate", il);
    } else {
        cur = up;
    }

    if (gate_exps_b) {
        cur = ggml_add_id(ctx0, cur, gate_exps_b, selected_experts);
        cb(cur, "ffn_moe_gate_biased", il);
    }

    switch (type_op) {
        case LLM_FFN_SILU:
            if (gate_exps) {
                const std::string swiglu_input_name = "blk." + std::to_string(il) + ".ffn_moe_swiglu_silu_input";
                cur = llama_maybe_observe_text_activation(ctx0, cur, swiglu_input_name.c_str(), cur->ne[1] > 1);
                cur = ggml_swiglu_split(ctx0, cur, up);
                cb(cur, "ffn_moe_swiglu", il);
            } else {
                const std::string silu_input_name = "blk." + std::to_string(il) + ".ffn_moe_silu_input";
                cur = llama_maybe_observe_text_activation(ctx0, cur, silu_input_name.c_str(), cur->ne[1] > 1);
                cur = ggml_silu(ctx0, cur);
                cb(cur, "ffn_moe_silu", il);
            } break;
        case LLM_FFN_GELU:
            if (gate_exps) {
                cur = ggml_geglu_split(ctx0, cur, up);
                cb(cur, "ffn_moe_geglu", il);
            } else {
                cur = ggml_gelu(ctx0, cur);
                cb(cur, "ffn_moe_gelu", il);
            } break;
        case LLM_FFN_SWIGLU_OAI_MOE:
            {
                // TODO: move to hparams?
                constexpr float alpha = 1.702f;
                constexpr float limit = 7.0f;
                cur = ggml_swiglu_oai(ctx0, cur, up, alpha, limit);
                cb(cur, "ffn_moe_swiglu_oai", il);
            } break;
        case LLM_FFN_RELU:
            if (gate_exps) {
                cur = ggml_reglu_split(ctx0, cur, up);
                cb(cur, "ffn_moe_reglu", il);
            } else {
                cur = ggml_relu(ctx0, cur);
                cb(cur, "ffn_moe_relu", il);
            } break;
        default:
            GGML_ABORT("fatal error");
    }

    experts = build_lora_mm_id(down_exps, cur, selected_experts); // [n_embd, n_expert_used, n_tokens]
    cb(experts, "ffn_moe_down", il);

    if (down_exps_b) {
        experts = ggml_add_id(ctx0, experts, down_exps_b, selected_experts);
        cb(experts, "ffn_moe_down_biased", il);
    }

    if (!weight_before_ffn) {
        experts = ggml_mul(ctx0, experts, weights);
        cb(cur, "ffn_moe_weighted", il);
    }

    ggml_tensor * cur_experts[LLAMA_MAX_EXPERTS] = { nullptr };

    assert(n_expert_used > 0);

    // order the views before the adds
    for (uint32_t i = 0; i < hparams.n_expert_used; ++i) {
        cur_experts[i] = ggml_view_2d(ctx0, experts, n_embd, n_tokens, experts->nb[2], i*experts->nb[1]);

        ggml_build_forward_expand(gf, cur_experts[i]);
    }

    // aggregate experts
    // note: here we explicitly use hparams.n_expert_used instead of n_expert_used
    //       to avoid potentially a large number of add nodes during warmup
    //       ref: https://github.com/ggml-org/llama.cpp/pull/14753
    ggml_tensor * moe_out = cur_experts[0];

    for (uint32_t i = 1; i < hparams.n_expert_used; ++i) {
        moe_out = ggml_add(ctx0, moe_out, cur_experts[i]);
    }

    if (hparams.n_expert_used == 1) {
        // avoid returning a non-contiguous tensor
        moe_out = ggml_cont(ctx0, moe_out);
    }

    cb(moe_out, "ffn_moe_out", il);

    return moe_out;
}

// input embeddings with optional lora
ggml_tensor * llm_graph_context::build_inp_embd(ggml_tensor * tok_embd) const {
    const int64_t n_embd = hparams.n_embd;

    auto inp = std::make_unique<llm_graph_input_embd>();

    ggml_tensor * cur = nullptr;

    if (ubatch.token) {
        inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_tokens);
        //cb(inp->tokens, "inp_tokens", -1);
        ggml_set_input(inp->tokens);
        res->t_tokens = inp->tokens;

        if (llama_text_token_embd_w8a16_enabled() && tok_embd->type == GGML_TYPE_F16) {
            ggml_tensor * out_template = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, tok_embd->ne[0], ubatch.n_tokens);
            llama_text_token_embd_w8a16_userdata & token_embd_w8a16_cfg = llama_text_token_embd_w8a16_cache();
            token_embd_w8a16_cfg.group = llama_text_token_embd_w8a16_group();
            static bool token_embd_w8a16_logged = false;
            if (!token_embd_w8a16_logged) {
                LLAMA_LOG_INFO("%s: using AICAS token embedding W8A16 path for token_embd.weight; group=%" PRId64 "\n",
                        __func__, token_embd_w8a16_cfg.group);
                token_embd_w8a16_logged = true;
            }
            cur = ggml_map_custom3(
                ctx0,
                out_template,
                tok_embd,
                inp->tokens,
                llama_compute_text_token_embd_w8a16_get_rows,
                GGML_N_TASKS_MAX,
                &token_embd_w8a16_cfg);
            ggml_set_name(cur, "text_token_embd_w8a16");
        } else {
            cur = ggml_get_rows(ctx0, tok_embd, inp->tokens);
        }

        // apply lora for embedding tokens if needed
        for (const auto & lora : *loras) {
            llama_adapter_lora_weight * lw = lora.first->get_weight(tok_embd);
            if (lw == nullptr) {
                continue;
            }

            const float adapter_scale = lora.second;
            const float scale = lw->get_scale(lora.first->alpha, adapter_scale);

            ggml_tensor * inpL_delta = ggml_scale(ctx0, ggml_mul_mat(
                        ctx0, lw->b, // non-transposed lora_b
                        ggml_get_rows(ctx0, lw->a, inp->tokens)
                        ), scale);

            cur = ggml_add(ctx0, cur, inpL_delta);
        }
    } else {
        inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, ubatch.n_tokens);
        ggml_set_input(inp->embd);

        cur = inp->embd;
    }

    // For Granite architecture
    if (hparams.f_embedding_scale != 0.0f) {
        cur = ggml_scale(ctx0, cur, hparams.f_embedding_scale);
    }

    cb(cur, "inp_embd", -1);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_pos() const {
    auto inp = std::make_unique<llm_graph_input_pos>(hparams.n_pos_per_embd());

    auto & cur = inp->pos;

    cur = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, (int64_t)n_tokens*hparams.n_pos_per_embd());
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_attn_scale() const {
    auto inp = std::make_unique<llm_graph_input_attn_temp>(hparams.n_attn_temp_floor_scale, hparams.f_attn_temp_scale);

    auto & cur = inp->attn_scale;

    // this need to be 1x1xN for broadcasting
    cur = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, 1, n_tokens);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_out_ids() const {
    // note: when all tokens are output, we could skip this optimization to spare the ggml_get_rows() calls,
    //       but this would make the graph topology depend on the number of output tokens, which can interere with
    //       features that require constant topology such as pipline parallelism
    //       ref: https://github.com/ggml-org/llama.cpp/pull/14275#issuecomment-2987424471
    //if (n_outputs < n_tokens) {
    //    return nullptr;
    //}

    auto inp = std::make_unique<llm_graph_input_out_ids>(hparams, cparams, n_outputs);

    auto & cur = inp->out_ids;

    cur = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_outputs);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_mean() const {
    auto inp = std::make_unique<llm_graph_input_mean>(cparams);

    auto & cur = inp->mean;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_tokens, ubatch.n_seqs_unq);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_cls() const {
    auto inp = std::make_unique<llm_graph_input_cls>(cparams, arch);

    auto & cur = inp->cls;

    cur = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_seqs_unq);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_cross_embd() const {
    auto inp = std::make_unique<llm_graph_input_cross_embd>(cross);

    auto & cur = inp->cross_embd;

    // if we have the output embeddings from the encoder, use them directly
    // TODO: needs more work to be correct, for now just use the tensor shape
    //if (cross->t_embd) {
    //    cur = ggml_view_tensor(ctx0, cross->t_embd);

    //    return cur;
    //}

    const auto n_embd = !cross->v_embd.empty() ? cross->n_embd : hparams.n_embd;
    const auto n_enc  = !cross->v_embd.empty() ? cross->n_enc : hparams.n_ctx_train;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_enc);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_pos_bucket_enc() const {
    auto inp = std::make_unique<llm_graph_input_pos_bucket>(hparams);

    auto & cur = inp->pos_bucket;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_tokens, n_tokens);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_pos_bucket_dec() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_pos_bucket_kv>(hparams, mctx_cur);

    const auto n_kv = mctx_cur->get_n_kv();

    auto & cur = inp->pos_bucket;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_kv, n_tokens);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_pos_bias(ggml_tensor * pos_bucket, ggml_tensor * attn_rel_b) const {
    ggml_tensor * pos_bucket_1d = ggml_reshape_1d(ctx0, pos_bucket, pos_bucket->ne[0] * pos_bucket->ne[1]);
    cb(pos_bucket_1d, "pos_bucket_1d", -1);

    ggml_tensor * pos_bias = ggml_get_rows(ctx0, attn_rel_b, pos_bucket_1d);

    pos_bias = ggml_reshape_3d(ctx0, pos_bias, pos_bias->ne[0], pos_bucket->ne[0], pos_bucket->ne[1]);
    pos_bias = ggml_permute   (ctx0, pos_bias, 2, 0, 1, 3);
    pos_bias = ggml_cont      (ctx0, pos_bias);

    cb(pos_bias, "pos_bias", -1);

    return pos_bias;
}

ggml_tensor * llm_graph_context::build_attn_mha(
         ggml_tensor * q,
         ggml_tensor * k,
         ggml_tensor * v,
         ggml_tensor * kq_b,
         ggml_tensor * kq_mask,
         ggml_tensor * sinks,
         ggml_tensor * v_mla,
               float   kq_scale,
                 int   il) const {
    const bool v_trans = v->nb[1] > v->nb[2];

    // split the batch into streams if needed
    const auto n_stream = k->ne[3];

    q = ggml_view_4d(ctx0, q, q->ne[0], q->ne[1], q->ne[2]/n_stream, n_stream, q->nb[1], q->nb[2], q->nb[3]/n_stream, 0);

    ggml_tensor * cur;

    const bool can_use_text_prefill_log8pv_pre_permute =
        llama_text_prefill_attn_log8pv_enabled() &&
        n_tokens > 1 &&
        kq_b == nullptr &&
        sinks == nullptr &&
        v_mla == nullptr &&
        !v_trans &&
        q->ne[0] == 64 &&
        k->ne[0] == 64 &&
        v->ne[0] == 64 &&
        q->ne[2] == 15 &&
        k->ne[2] == 5 &&
        v->ne[2] == 5 &&
        q->ne[2] % k->ne[2] == 0 &&
        !hparams.attn_soft_cap &&
        hparams.f_max_alibi_bias == 0.0f &&
        (q->type == GGML_TYPE_F32 || q->type == GGML_TYPE_F16) &&
        (k->type == GGML_TYPE_F32 || k->type == GGML_TYPE_F16 || k->type == GGML_TYPE_Q8_0) &&
        (v->type == GGML_TYPE_F32 || v->type == GGML_TYPE_F16 || v->type == GGML_TYPE_Q8_0);

    if (can_use_text_prefill_log8pv_pre_permute) {
        cur = llama_build_text_log8pv_attn(ctx0, q, k, v, kq_mask, kq_scale, il);
        cb(cur, "fattn_log8pv", il);
        cur = ggml_reshape_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);
        ggml_build_forward_expand(gf, cur);
        return cur;
    }

    q = ggml_permute(ctx0, q, 0, 2, 1, 3);
    k = ggml_permute(ctx0, k, 0, 2, 1, 3);
    v = ggml_permute(ctx0, v, 0, 2, 1, 3);

    if (cparams.flash_attn && kq_b == nullptr) {
        GGML_ASSERT(kq_b == nullptr && "Flash attention does not support KQ bias yet");

        // this can happen when KV cache is not used (e.g. an embedding model with non-causal attn)
        if (k->type == GGML_TYPE_F32) {
            k = ggml_cast(ctx0, k, GGML_TYPE_F16);
        }

        if (v->type == GGML_TYPE_F32) {
            v = ggml_cast(ctx0, v, GGML_TYPE_F16);
        }

        const bool want_text_prefill_log8pv =
            llama_text_prefill_attn_log8pv_enabled() &&
            n_tokens > 1 &&
            sinks == nullptr &&
            v_mla == nullptr &&
            !v_trans &&
            q->ne[0] == 64 &&
            k->ne[0] == 64 &&
            v->ne[0] == 64 &&
            q->ne[2] == 15 &&
            k->ne[2] == 5 &&
            v->ne[2] == 5 &&
            q->ne[2] % k->ne[2] == 0 &&
            !hparams.attn_soft_cap &&
            hparams.f_max_alibi_bias == 0.0f;

        const bool can_use_text_prefill_log8pv =
            want_text_prefill_log8pv &&
            (q->type == GGML_TYPE_F32 || q->type == GGML_TYPE_F16) &&
            (k->type == GGML_TYPE_F32 || k->type == GGML_TYPE_F16 || k->type == GGML_TYPE_Q8_0) &&
            (v->type == GGML_TYPE_F32 || v->type == GGML_TYPE_F16 || v->type == GGML_TYPE_Q8_0);

        if (can_use_text_prefill_log8pv) {
            cur = llama_build_text_log8pv_attn(ctx0, q, k, v, kq_mask, kq_scale, il);
            cb(cur, "fattn_log8pv", il);
        } else {
            if (v_trans) {
                v = ggml_transpose(ctx0, v);
            }

            cur = ggml_flash_attn_ext(ctx0, q, k, v, kq_mask, kq_scale, hparams.f_max_alibi_bias,
                                      hparams.attn_soft_cap ? hparams.f_attn_logit_softcapping : 0.0f);
            cb(cur, LLAMA_TENSOR_NAME_FATTN, il);

            ggml_flash_attn_ext_add_sinks(cur, sinks);
            ggml_flash_attn_ext_set_prec (cur, llama_text_flash_attn_accum_fp16() ? GGML_PREC_DEFAULT : GGML_PREC_F32);
        }

        if (v_mla) {
#if 0
            // v_mla can be applied as a matrix-vector multiplication with broadcasting across dimension 3 == n_tokens.
            // However, the code is optimized for dimensions 0 and 1 being large, so this is ineffient.
            cur = ggml_reshape_4d(ctx0, cur, v_mla->ne[0], 1, n_head, n_tokens);
            cur = ggml_mul_mat(ctx0, v_mla, cur);
#else
            // It's preferable to do the calculation as a matrix-matrix multiplication with n_tokens in dimension 1.
            // The permutations are noops and only change how the tensor data is interpreted.
            cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
            cur = ggml_mul_mat(ctx0, v_mla, cur);
            cb(cur, "fattn_mla", il);
            cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
            cur = ggml_cont(ctx0, cur); // Needed because ggml_reshape_2d expects contiguous inputs.
#endif
        }

        cur = ggml_reshape_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);
    } else {
        const bool can_use_text_prefill_custom_attn =
            n_tokens > 1 &&
            kq_b == nullptr &&
            sinks == nullptr &&
            v_mla == nullptr;
        const bool use_text_prefill_bfp8m =
            can_use_text_prefill_custom_attn &&
            llama_text_prefill_attn_bfp8m_enabled();
        const bool use_text_prefill_bfp16m =
            can_use_text_prefill_custom_attn &&
            !use_text_prefill_bfp8m &&
            llama_text_prefill_attn_bfp16m_enabled();

        ggml_tensor * kq = nullptr;
        if (use_text_prefill_bfp8m) {
            kq = llama_build_text_bfp8m_mul_mat(ctx0, k, q);
        } else if (use_text_prefill_bfp16m) {
            kq = llama_build_text_bfp16m_mul_mat(ctx0, k, q, false);
        } else {
            kq = ggml_mul_mat(ctx0, k, q);
        }
        cb(kq, "kq", il);

        // note: this op tends to require high floating point range
        //       while for some models F16 is enough, for others it is not, so we default to F32 here
        if (!use_text_prefill_bfp8m && !use_text_prefill_bfp16m) {
            ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
        }

        if (arch == LLM_ARCH_GROK) {
            // need to do the following:
            // multiply by attn_output_multiplier
            // and then :
            // kq = 30 * tanh(kq / 30)
            // before the softmax below

            kq = ggml_tanh(ctx0, ggml_scale(ctx0, kq, hparams.f_attn_out_scale / hparams.f_attn_logit_softcapping));
            cb(kq, "kq_tanh", il);
            kq = ggml_scale(ctx0, kq, hparams.f_attn_logit_softcapping);
            cb(kq, "kq_scaled", il);
        }

        if (hparams.attn_soft_cap) {
            kq = ggml_scale(ctx0, kq, 1.0f / hparams.f_attn_logit_softcapping);
            cb(kq, "kq_scaled_1", il);
            kq = ggml_tanh (ctx0, kq);
            cb(kq, "kq_tanh", il);
            kq = ggml_scale(ctx0, kq, hparams.f_attn_logit_softcapping);
            cb(kq, "kq_scaled_2", il);
        }

        if (kq_b) {
            kq = ggml_add(ctx0, kq, kq_b);
            cb(kq, "kq_plus_kq_b", il);
        }

        kq = ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale, hparams.f_max_alibi_bias);
        ggml_soft_max_add_sinks(kq, sinks);
        cb(kq, "kq_soft_max", il);

        if (!v_trans) {
            // note: avoid this branch
            v = ggml_cont(ctx0, ggml_transpose(ctx0, v));
            cb(v, "v_cont", il);
        }

        ggml_tensor * kqv = nullptr;
        if (use_text_prefill_bfp8m) {
            kqv = llama_build_text_bfp8m_mul_mat(ctx0, v, kq);
        } else if (use_text_prefill_bfp16m) {
            kqv = llama_build_text_bfp16m_mul_mat(ctx0, v, kq, true);
        } else {
            kqv = ggml_mul_mat(ctx0, v, kq);
        }
        cb(kqv, "kqv", il);

        // for MLA with the absorption optimization, we need to "decompress" from MQA back to MHA
        if (v_mla) {
            kqv = ggml_mul_mat(ctx0, v_mla, kqv);
            cb(kqv, "kqv_mla", il);
        }

        cur = ggml_permute(ctx0, kqv, 0, 2, 1, 3);

        // recombine streams
        cur = ggml_cont_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);

        if (!cparams.offload_kqv) {
            // all nodes between the KV store and the attention output are run on the CPU
            ggml_backend_sched_set_tensor_backend(sched, cur, backend_cpu);
        }
    }

    ggml_build_forward_expand(gf, cur);

    return cur;
}

llm_graph_input_attn_no_cache * llm_graph_context::build_attn_inp_no_cache() const {
    auto inp = std::make_unique<llm_graph_input_attn_no_cache>(hparams, cparams);

    // note: there is no KV cache, so the number of KV values is equal to the number of tokens in the batch
    inp->self_kq_mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_tokens, GGML_PAD(n_tokens, GGML_KQ_MASK_PAD), 1, 1);
    ggml_set_input(inp->self_kq_mask);

    inp->self_kq_mask_cnv = cparams.flash_attn ? ggml_cast(ctx0, inp->self_kq_mask, GGML_TYPE_F16) : inp->self_kq_mask;

    if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
        inp->self_kq_mask_swa = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_tokens, GGML_PAD(n_tokens, GGML_KQ_MASK_PAD), 1, 1);
        ggml_set_input(inp->self_kq_mask_swa);

        inp->self_kq_mask_swa_cnv = cparams.flash_attn ? ggml_cast(ctx0, inp->self_kq_mask_swa, GGML_TYPE_F16) : inp->self_kq_mask_swa;
    } else {
        inp->self_kq_mask_swa     = nullptr;
        inp->self_kq_mask_swa_cnv = nullptr;
    }

    return (llm_graph_input_attn_no_cache *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_no_cache * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    GGML_UNUSED(n_tokens);

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, k_cur);
    ggml_build_forward_expand(gf, v_cur);

    const bool is_swa = hparams.is_swa(il);

    const auto & kq_mask = is_swa ? inp->get_kq_mask_swa() : inp->get_kq_mask();

    // [TAG_NO_CACHE_PAD]
    // TODO: if ubatch.equal_seqs() == true, we can split the three tensors below into ubatch.n_seqs_unq streams
    //       but it might not be worth it: https://github.com/ggml-org/llama.cpp/pull/15636
    //assert(!ubatch.equal_seqs() || (k_cur->ne[3] == 1 && k_cur->ne[3] == ubatch.n_seqs_unq));

    ggml_tensor * q = q_cur;
    ggml_tensor * k = k_cur;
    ggml_tensor * v = v_cur;

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_lora_mm(wo, cur);
    }

    if (wo_b) {
        //cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

static std::unique_ptr<llm_graph_input_attn_kv> build_attn_inp_kv_impl(
           ggml_context * ctx0,
     const llama_ubatch & ubatch,
    const llama_hparams & hparams,
    const llama_cparams & cparams,
    const llama_kv_cache_context * mctx_cur) {

    auto inp = std::make_unique<llm_graph_input_attn_kv>(hparams, cparams, mctx_cur);

    {
        GGML_ASSERT(hparams.swa_type == LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache_iswa for SWA");

        const auto n_kv     = mctx_cur->get_n_kv();
        const auto n_tokens = ubatch.n_tokens;
        const auto n_stream = cparams.kv_unified ? 1 : ubatch.n_seqs_unq;

        inp->self_k_idxs = mctx_cur->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs = mctx_cur->build_input_v_idxs(ctx0, ubatch);

        inp->self_kq_mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_kv, GGML_PAD(n_tokens/n_stream, GGML_KQ_MASK_PAD), 1, n_stream);
        ggml_set_input(inp->self_kq_mask);

        inp->self_kq_mask_cnv = cparams.flash_attn ? ggml_cast(ctx0, inp->self_kq_mask, GGML_TYPE_F16) : inp->self_kq_mask;
    }

    return inp;
}

llm_graph_input_attn_kv * llm_graph_context::build_attn_inp_kv() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_context *>(mctx);

    auto inp = build_attn_inp_kv_impl(ctx0, ubatch, hparams, cparams, mctx_cur);

    return (llm_graph_input_attn_kv *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn_decode_npu(
        llm_graph_input_attn_kv * inp,
        ggml_tensor * inp_pos,
        ggml_tensor * cur,
        ggml_tensor * wq,
        ggml_tensor * bq,
        ggml_tensor * wk,
        ggml_tensor * bk,
        ggml_tensor * wv,
        ggml_tensor * bv,
        ggml_tensor * wo,
        ggml_tensor * bo,
            float     kq_scale,
            int       il) const {
#ifndef GGML_USE_NPU
    GGML_UNUSED(inp);
    GGML_UNUSED(inp_pos);
    GGML_UNUSED(cur);
    GGML_UNUSED(wq);
    GGML_UNUSED(bq);
    GGML_UNUSED(wk);
    GGML_UNUSED(bk);
    GGML_UNUSED(wv);
    GGML_UNUSED(bv);
    GGML_UNUSED(wo);
    GGML_UNUSED(bo);
    GGML_UNUSED(kq_scale);
    GGML_UNUSED(il);
    return nullptr;
#else
    if (!llama_text_decode_attention_npu_enabled() ||
            il >= llama_text_decode_attention_npu_max_layer() ||
            n_tokens != 1 ||
            inp == nullptr ||
            inp_pos == nullptr ||
            cur == nullptr ||
            wq == nullptr || wk == nullptr || wv == nullptr || wo == nullptr ||
            bq != nullptr || bk != nullptr || bv != nullptr || bo != nullptr ||
            hparams.use_kq_norm ||
            hparams.swa_type != LLAMA_SWA_TYPE_NONE ||
            hparams.attn_soft_cap ||
            hparams.f_max_alibi_bias != 0.0f ||
            loras == nullptr || !loras->empty()) {
        return nullptr;
    }
    if ((cur->type != GGML_TYPE_F16 && cur->type != GGML_TYPE_F32) ||
            cur->ne[0] != 960 || cur->ne[1] != 1 ||
            n_embd_head_k != 64 || n_embd_head_v != 64 ||
            n_head != 15 || n_head_kv != 5 ||
            wq->ne[0] != 960 || wq->ne[1] != 960 ||
            wk->ne[0] != 960 || wk->ne[1] != 320 ||
            wv->ne[0] != 960 || wv->ne[1] != 320 ||
            wo->ne[0] != 960 || wo->ne[1] != 960) {
        return nullptr;
    }

    auto find_awq = [&](const ggml_tensor * w) -> const llama_aicas_text_decode_awq_tensor * {
        auto it = model.aicas_text_decode_awq_tensors.find(w->name);
        if (!model.aicas_text_decode_awq_enabled || it == model.aicas_text_decode_awq_tensors.end()) {
            return nullptr;
        }
        const auto & cfg = it->second;
        if (!cfg.enabled ||
                cfg.policy != "Q4_AWQ" ||
                cfg.group_size != 128 ||
                cfg.quant_tensor == nullptr ||
                cfg.scale_tensor == nullptr ||
                cfg.zero_tensor == nullptr ||
                cfg.quant_tensor->type != GGML_TYPE_I8 ||
                cfg.packed_in_features() != cfg.quant_tensor->ne[0] ||
                !cfg.has_valid_smooth_config() ||
                !cfg.has_valid_group_params(cfg.quant_tensor->ne[1])) {
            return nullptr;
        }
        return &cfg;
    };
    const llama_aicas_text_decode_awq_tensor * q_cfg = find_awq(wq);
    const llama_aicas_text_decode_awq_tensor * k_cfg = find_awq(wk);
    const llama_aicas_text_decode_awq_tensor * v_cfg = find_awq(wv);
    const llama_aicas_text_decode_awq_tensor * o_cfg = find_awq(wo);
    if (q_cfg == nullptr || k_cfg == nullptr || v_cfg == nullptr || o_cfg == nullptr) {
        return nullptr;
    }

    const auto * mctx_cur = inp->mctx;
    ggml_tensor * k_cache = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v_cache = mctx_cur->get_v(ctx0, il);
    if (k_cache == nullptr || v_cache == nullptr ||
            k_cache->ne[0] != 64 || k_cache->ne[1] != 5 ||
            v_cache->ne[0] != 64 || v_cache->ne[1] != 5 ||
            k_cache->ne[2] < 1 || v_cache->ne[2] < 1 ||
            (k_cache->type != GGML_TYPE_Q8_0 && k_cache->type != GGML_TYPE_F16 && k_cache->type != GGML_TYPE_F32) ||
            (v_cache->type != GGML_TYPE_Q8_0 && v_cache->type != GGML_TYPE_F16 && v_cache->type != GGML_TYPE_F32)) {
        return nullptr;
    }

    static std::atomic<bool> logged{false};
    if (!logged.exchange(true)) {
        LLAMA_LOG_INFO("%s: using AICAS text decode attention NPU path; KV cache converted to separated CMA layout on first decode token\n",
                __func__);
    }

    static std::mutex userdata_mutex;
    static std::vector<std::unique_ptr<llama_text_decode_attention_npu_userdata>> userdata_store;
    auto node_cfg = std::make_unique<llama_text_decode_attention_npu_userdata>();
    node_cfg->op_name = "blk." + std::to_string(il) + ".attn_npu";
    node_cfg->q = llama_make_npu_decode_awq_view(*q_cfg);
    node_cfg->k = llama_make_npu_decode_awq_view(*k_cfg);
    node_cfg->v = llama_make_npu_decode_awq_view(*v_cfg);
    node_cfg->o = llama_make_npu_decode_awq_view(*o_cfg);
    node_cfg->kq_scale = kq_scale;
    node_cfg->rope_freq_base = freq_base;
    node_cfg->rope_freq_scale = freq_scale;
    node_cfg->layer_id = il;
    llama_text_decode_attention_npu_userdata * ptr = node_cfg.get();
    {
        std::lock_guard<std::mutex> lock(userdata_mutex);
        userdata_store.push_back(std::move(node_cfg));
    }

    ggml_tensor * args[] = {
        cur,
        inp_pos,
        k_cache,
        v_cache,
    };
    ggml_tensor * out = ggml_custom_4d(
            ctx0,
            GGML_TYPE_F32,
            960,
            1,
            1,
            1,
            args,
            4,
            llama_compute_text_decode_attention_npu,
            1,
            ptr);
    ggml_set_name(out, "text_decode_attention_npu");
    cb(out, "attn_npu", il);
    return out;
#endif
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_kv * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, k_cur);
    ggml_build_forward_expand(gf, v_cur);

    const auto * mctx_cur = inp->mctx;

    // store to KV cache
    {
        const auto & k_idxs = inp->get_k_idxs();
        const auto & v_idxs = inp->get_v_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
    }

    const auto & kq_mask = inp->get_kq_mask();

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = mctx_cur->get_v(ctx0, il);

    bool direct_log8pv_kv_current_chunk_covers_context = ubatch.n_tokens == n_tokens;
    for (uint32_t i = 0; i < ubatch.n_tokens && direct_log8pv_kv_current_chunk_covers_context; ++i) {
        direct_log8pv_kv_current_chunk_covers_context = ubatch.pos[i] == (llama_pos) i;
    }

    const bool use_direct_log8pv_kv =
        llama_text_prefill_log8pv_direct_kv_enabled() &&
        llama_text_prefill_attn_log8pv_enabled() &&
        cparams.flash_attn &&
        n_tokens > 1 &&
        direct_log8pv_kv_current_chunk_covers_context &&
        kq_b == nullptr &&
        sinks == nullptr &&
        v_mla == nullptr &&
        !hparams.attn_soft_cap &&
        hparams.f_max_alibi_bias == 0.0f &&
        q_cur->ne[0] == 64 &&
        k_cur->ne[0] == 64 &&
        v_cur->ne[0] == 64 &&
        q_cur->ne[1] % k_cur->ne[1] == 0 &&
        k_cur->ne[1] == v_cur->ne[1] &&
        q_cur->ne[2] == k_cur->ne[2] &&
        q_cur->ne[2] == v_cur->ne[2];
    static std::atomic<bool> direct_log8pv_kv_logged{false};
    if (use_direct_log8pv_kv) {
        if (!direct_log8pv_kv_logged.exchange(true)) {
            LLAMA_LOG_INFO("%s: AICAS text prefill log8PV direct-KV path enabled; "
                    "prefill attention uses current k_cur/v_cur while cache writes still store persistent KV\n",
                    __func__);
        }
        k = k_cur;
        v = v_cur;
    }

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_lora_mm(wo, cur);
        if (arch == LLM_ARCH_GLM4 || arch == LLM_ARCH_GLM4_MOE) {
            // GLM4 and GLM4_MOE seem to have numerical issues with half-precision accumulators
            ggml_mul_mat_set_prec(cur, GGML_PREC_F32);
        }
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_kv_iswa * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);

    if (k_cur) {
        ggml_build_forward_expand(gf, k_cur);
    }

    if (v_cur) {
        ggml_build_forward_expand(gf, v_cur);
    }

    const auto * mctx_iswa = inp->mctx;

    const bool is_swa = hparams.is_swa(il);

    const auto * mctx_cur = is_swa ? mctx_iswa->get_swa() : mctx_iswa->get_base();

    // optionally store to KV cache
    if (k_cur) {
        const auto & k_idxs = is_swa ? inp->get_k_idxs_swa() : inp->get_k_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
    }

    if (v_cur) {
        const auto & v_idxs = is_swa ? inp->get_v_idxs_swa() : inp->get_v_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
    }

    const auto & kq_mask = is_swa ? inp->get_kq_mask_swa() : inp->get_kq_mask();

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = mctx_cur->get_v(ctx0, il);

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_lora_mm(wo, cur);
    }

    if (wo_b) {
        //cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

llm_graph_input_attn_cross * llm_graph_context::build_attn_inp_cross() const {
    auto inp = std::make_unique<llm_graph_input_attn_cross>(cross);

    const int32_t n_enc = !cross->v_embd.empty() ? cross->n_enc : hparams.n_ctx_train;

    inp->cross_kq_mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_enc, GGML_PAD(n_tokens, GGML_KQ_MASK_PAD), 1, 1);
    ggml_set_input(inp->cross_kq_mask);

    inp->cross_kq_mask_cnv = cparams.flash_attn ? ggml_cast(ctx0, inp->cross_kq_mask, GGML_TYPE_F16) : inp->cross_kq_mask;

    return (llm_graph_input_attn_cross *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_cross * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, k_cur);
    ggml_build_forward_expand(gf, v_cur);

    const auto & kq_mask = inp->get_kq_mask_cross();

    ggml_tensor * q = q_cur;
    ggml_tensor * k = k_cur;
    ggml_tensor * v = v_cur;

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_lora_mm(wo, cur);
    }

    if (wo_b) {
        //cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

// TODO: maybe separate the inner implementation into a separate function
//       like with the non-sliding window equivalent
//       once sliding-window hybrid caches are a thing.
llm_graph_input_attn_kv_iswa * llm_graph_context::build_attn_inp_kv_iswa() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_iswa_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_attn_kv_iswa>(hparams, cparams, mctx_cur);

    const auto n_stream = cparams.kv_unified ? 1 : ubatch.n_seqs_unq;

    {
        const auto n_kv = mctx_cur->get_base()->get_n_kv();

        inp->self_k_idxs = mctx_cur->get_base()->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs = mctx_cur->get_base()->build_input_v_idxs(ctx0, ubatch);

        inp->self_kq_mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_kv, GGML_PAD(n_tokens/n_stream, GGML_KQ_MASK_PAD), 1, n_stream);
        ggml_set_input(inp->self_kq_mask);

        inp->self_kq_mask_cnv = cparams.flash_attn ? ggml_cast(ctx0, inp->self_kq_mask, GGML_TYPE_F16) : inp->self_kq_mask;
    }

    {
        GGML_ASSERT(hparams.swa_type != LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache for non-SWA");

        const auto n_kv = mctx_cur->get_swa()->get_n_kv();

        inp->self_k_idxs_swa = mctx_cur->get_swa()->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs_swa = mctx_cur->get_swa()->build_input_v_idxs(ctx0, ubatch);

        inp->self_kq_mask_swa = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_kv, GGML_PAD(n_tokens/n_stream, GGML_KQ_MASK_PAD), 1, n_stream);
        ggml_set_input(inp->self_kq_mask_swa);

        inp->self_kq_mask_swa_cnv = cparams.flash_attn ? ggml_cast(ctx0, inp->self_kq_mask_swa, GGML_TYPE_F16) : inp->self_kq_mask_swa;
    }

    return (llm_graph_input_attn_kv_iswa *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_rs(
        ggml_tensor * s,
        ggml_tensor * state_copy_main,
        ggml_tensor * state_copy_extra,
            int32_t   state_size,
            int32_t   n_seqs,
           uint32_t   n_rs,
           uint32_t   rs_head,
           uint32_t   rs_size,
            int32_t   rs_zero,
        const llm_graph_get_rows_fn & get_state_rows) const {

    ggml_tensor * states = ggml_reshape_2d(ctx0, s, state_size, rs_size);

    // Clear a single state which will then be copied to the other cleared states.
    // Note that this is a no-op when the view is zero-sized.
    ggml_tensor * state_zero = ggml_view_1d(ctx0, states, state_size*(rs_zero >= 0), rs_zero*states->nb[1]*(rs_zero >= 0));
    ggml_build_forward_expand(gf, ggml_scale_inplace(ctx0, state_zero, 0));

    // copy states
    // NOTE: assuming the copy destinations are ALL contained between rs_head and rs_head + n_rs
    // {state_size, rs_size} -> {state_size, n_seqs}
    ggml_tensor * output_states = get_state_rows(ctx0, states, state_copy_main);
    ggml_build_forward_expand(gf, output_states);

    // copy extra states which won't be changed further (between n_seqs and n_rs)
    ggml_tensor * states_extra = ggml_get_rows(ctx0, states, state_copy_extra);
    ggml_build_forward_expand(gf,
        ggml_cpy(ctx0,
            states_extra,
            ggml_view_1d(ctx0, s, state_size*(n_rs - n_seqs), (rs_head + n_seqs)*state_size*ggml_element_size(s))));

    return output_states;
}

static std::unique_ptr<llm_graph_input_rs> build_rs_inp_impl(
           ggml_context * ctx0,
     const llama_ubatch & ubatch,
    const llama_memory_recurrent_context * mctx_cur) {

    auto inp = std::make_unique<llm_graph_input_rs>(mctx_cur);

    const int64_t n_rs   = mctx_cur->get_n_rs();
    const int64_t n_seqs = ubatch.n_seqs;

    inp->s_copy = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_rs);
    ggml_set_input(inp->s_copy);

    inp->s_copy_main  = ggml_view_1d(ctx0, inp->s_copy, n_seqs, 0);
    inp->s_copy_extra = ggml_view_1d(ctx0, inp->s_copy, n_rs - n_seqs, n_seqs * inp->s_copy->nb[0]);

    return inp;
}

llm_graph_input_rs * llm_graph_context::build_rs_inp() const {
    const auto * mctx_cur = static_cast<const llama_memory_recurrent_context *>(mctx);

    auto inp = build_rs_inp_impl(ctx0, ubatch, mctx_cur);

    return (llm_graph_input_rs *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_rs(
        llm_graph_input_rs * inp,
        ggml_tensor * s,
            int32_t   state_size,
            int32_t   n_seqs,
        const llm_graph_get_rows_fn & get_state_rows) const {
    const auto * kv_state = inp->mctx;

    return build_rs(s, inp->s_copy_main, inp->s_copy_extra, state_size, n_seqs,
                    kv_state->get_n_rs(), kv_state->get_head(), kv_state->get_size(), kv_state->get_rs_z(),
                    get_state_rows);
}

ggml_tensor * llm_graph_context::build_rwkv_token_shift_load(
    llm_graph_input_rs * inp,
    const llama_ubatch & ubatch,
                   int   il) const {
    const auto * mctx_cur = static_cast<const llama_memory_recurrent_context *>(mctx);

    const auto token_shift_count = hparams.token_shift_count;

    const int64_t n_seqs  = ubatch.n_seqs;

    ggml_tensor * token_shift_all = mctx_cur->get_r_l(il);

    ggml_tensor * token_shift = build_rs(
            inp, token_shift_all,
            hparams.n_embd_r(), n_seqs);

    token_shift = ggml_reshape_3d(ctx0, token_shift, hparams.n_embd, token_shift_count, n_seqs);

    return token_shift;
}

ggml_tensor * llm_graph_context::build_rwkv_token_shift_store(
         ggml_tensor * token_shift,
  const llama_ubatch & ubatch,
                 int   il) const {
    const auto * mctx_cur = static_cast<const llama_memory_recurrent_context *>(mctx);

    const auto token_shift_count = hparams.token_shift_count;
    const auto n_embd = hparams.n_embd;

    const int64_t n_seqs = ubatch.n_seqs;

    const auto kv_head = mctx_cur->get_head();

    return ggml_cpy(
        ctx0,
        ggml_view_1d(ctx0, token_shift, n_embd * n_seqs * token_shift_count, 0),
        ggml_view_1d(ctx0, mctx_cur->get_r_l(il), hparams.n_embd_r()*n_seqs, hparams.n_embd_r()*kv_head*ggml_element_size(mctx_cur->get_r_l(il)))
    );
}

llm_graph_input_mem_hybrid * llm_graph_context::build_inp_mem_hybrid() const {
    const auto * mctx_cur = static_cast<const llama_memory_hybrid_context *>(mctx);

    auto inp_rs   = build_rs_inp_impl(ctx0, ubatch, mctx_cur->get_recr());
    auto inp_attn = build_attn_inp_kv_impl(ctx0, ubatch, hparams, cparams, mctx_cur->get_attn());

    auto inp = std::make_unique<llm_graph_input_mem_hybrid>(std::move(inp_attn), std::move(inp_rs), mctx_cur);

    return (llm_graph_input_mem_hybrid *) res->add_input(std::move(inp));
}

void llm_graph_context::build_dense_out(
    ggml_tensor * dense_2,
    ggml_tensor * dense_3) const {
    if (!cparams.embeddings || dense_2 == nullptr || dense_3 == nullptr) {
        return;
    }
    ggml_tensor * cur = res->t_embd_pooled != nullptr ? res->t_embd_pooled : res->t_embd;
    GGML_ASSERT(cur != nullptr && "missing t_embd_pooled/t_embd");

    cur = ggml_mul_mat(ctx0, dense_2, cur);
    cur = ggml_mul_mat(ctx0, dense_3, cur);
    cb(cur, "result_embd_pooled", -1);
    res->t_embd_pooled = cur;
    ggml_build_forward_expand(gf, cur);
}


void llm_graph_context::build_pooling(
        ggml_tensor * cls,
        ggml_tensor * cls_b,
        ggml_tensor * cls_out,
        ggml_tensor * cls_out_b) const {
    if (!cparams.embeddings) {
        return;
    }

    ggml_tensor * inp = res->t_embd;

    //// find result_norm tensor for input
    //for (int i = ggml_graph_n_nodes(gf) - 1; i >= 0; --i) {
    //    inp = ggml_graph_node(gf, i);
    //    if (strcmp(inp->name, "result_norm") == 0 || strcmp(inp->name, "result_embd") == 0) {
    //        break;
    //    }

    //    inp = nullptr;
    //}

    GGML_ASSERT(inp != nullptr && "missing result_norm/result_embd tensor");

    ggml_tensor * cur;

    switch (pooling_type) {
        case LLAMA_POOLING_TYPE_NONE:
            {
                cur = inp;
            } break;
        case LLAMA_POOLING_TYPE_MEAN:
            {
                ggml_tensor * inp_mean = build_inp_mean();
                cur = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, inp)), inp_mean);
            } break;
        case LLAMA_POOLING_TYPE_CLS:
        case LLAMA_POOLING_TYPE_LAST:
            {
                ggml_tensor * inp_cls = build_inp_cls();
                cur = ggml_get_rows(ctx0, inp, inp_cls);
            } break;
        case LLAMA_POOLING_TYPE_RANK:
            {
                ggml_tensor * inp_cls = build_inp_cls();
                cur = ggml_get_rows(ctx0, inp, inp_cls);

                // classification head
                // https://github.com/huggingface/transformers/blob/5af7d41e49bbfc8319f462eb45253dcb3863dfb7/src/transformers/models/roberta/modeling_roberta.py#L1566
                if (cls) {
                    cur = ggml_mul_mat(ctx0, cls, cur);
                    if (cls_b) {
                        cur = ggml_add(ctx0, cur, cls_b);
                    }
                    cur = ggml_tanh(ctx0, cur);
                }

                // some models don't have `cls_out`, for example: https://huggingface.co/jinaai/jina-reranker-v1-tiny-en
                // https://huggingface.co/jinaai/jina-reranker-v1-tiny-en/blob/cb5347e43979c3084a890e3f99491952603ae1b7/modeling_bert.py#L884-L896
                // Single layer classification head (direct projection)
                // https://github.com/huggingface/transformers/blob/f4fc42216cd56ab6b68270bf80d811614d8d59e4/src/transformers/models/bert/modeling_bert.py#L1476
                if (cls_out) {
                    cur = ggml_mul_mat(ctx0, cls_out, cur);
                    if (cls_out_b) {
                        cur = ggml_add(ctx0, cur, cls_out_b);
                    }
                }

                // softmax for qwen3 reranker
                if (arch == LLM_ARCH_QWEN3) {
                    cur = ggml_soft_max(ctx0, cur);
                }
            } break;
        default:
            {
                GGML_ABORT("unknown pooling type");
            }
    }

    cb(cur, "result_embd_pooled", -1);
    res->t_embd_pooled = cur;

    ggml_build_forward_expand(gf, cur);
}

int32_t llama_relative_position_bucket(llama_pos x, llama_pos y, uint64_t n_buckets, bool bidirectional) {
    // TODO move to hparams if a T5 variant appears that uses a different value
    const int64_t max_distance = 128;

    if (bidirectional) {
        n_buckets >>= 1;
    }

    const int64_t max_exact = n_buckets >> 1;

    int32_t relative_position = x - y;
    int32_t relative_bucket = 0;

    if (bidirectional) {
        relative_bucket += (relative_position > 0) * n_buckets;
        relative_position = abs(relative_position);
    } else {
        relative_position = -std::min<int32_t>(relative_position, 0);
    }

    int32_t relative_position_if_large = floorf(max_exact + logf(1.0 * relative_position / max_exact) * (n_buckets - max_exact) / log(1.0 * max_distance / max_exact));
    relative_position_if_large = std::min<int32_t>(relative_position_if_large, n_buckets - 1);
    relative_bucket += (relative_position < max_exact ? relative_position : relative_position_if_large);

    return relative_bucket;
}
