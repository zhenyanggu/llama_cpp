#include "ggml-npu-plan.h"
#include "ggml-npu-quant.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace ggml_npu {

struct npu_static_quant_entry {
    std::string key;
    int64_t ne0 = -1;
    int64_t ne1 = -1;
    int64_t ne2 = -1;
    int64_t ne3 = -1;
    float scale = 1.0f;
    int32_t zero_point = 0;
};

struct npu_static_quant_table {
    bool loaded = false;
    std::vector<npu_static_quant_entry> entries;
};

struct npu_aicas_w8a8_table {
    std::mutex mutex;
    std::unordered_map<std::string, npu_aicas_w8a8_config> entries;
};

static npu_static_quant_table & npu_get_static_quant_table() {
    static npu_static_quant_table table;
    return table;
}

static npu_aicas_w8a8_table & npu_get_aicas_w8a8_table() {
    static npu_aicas_w8a8_table table;
    return table;
}

static int32_t npu_float_to_q8_24(float scale) {
    const double scaled = std::nearbyint(static_cast<double>(scale) * static_cast<double>(1u << 24));
    if (scaled > static_cast<double>(INT32_MAX)) {
        return INT32_MAX;
    }
    if (scaled < static_cast<double>(INT32_MIN)) {
        return INT32_MIN;
    }
    return static_cast<int32_t>(scaled);
}

void npu_clear_aicas_w8a8_table(void) {
    npu_aicas_w8a8_table & table = npu_get_aicas_w8a8_table();
    std::lock_guard<std::mutex> lock(table.mutex);
    table.entries.clear();
}

bool npu_register_aicas_w8a8(
        const char * weight_name,
        float act_scale,
    int32_t act_scale_q8_24,
        int32_t act_zero_point_u8,
        const float * weight_scale,
        size_t weight_scale_len,
        const int32_t * sum_w,
        size_t sum_w_len) {
    if (weight_name == nullptr || weight_name[0] == '\0') {
        return false;
    }
    if (!(act_scale > 0.0f)) {
        return false;
    }
    if (act_zero_point_u8 < 0 || act_zero_point_u8 > 255) {
        return false;
    }
    if (weight_scale_len == 0 || sum_w_len == 0 || weight_scale_len != sum_w_len) {
        return false;
    }
    if (weight_scale == nullptr || sum_w == nullptr) {
        return false;
    }

    npu_aicas_w8a8_config cfg;
    cfg.valid = true;
    cfg.act_scale = act_scale;
    cfg.act_scale_q8_24 = act_scale_q8_24 != 0 ? act_scale_q8_24 : npu_float_to_q8_24(act_scale);
    cfg.act_zero_point_i8 = act_zero_point_u8 - 128;
    cfg.weight_scale.assign(weight_scale, weight_scale + weight_scale_len);
    cfg.sum_w.assign(sum_w, sum_w + sum_w_len);

    npu_aicas_w8a8_table & table = npu_get_aicas_w8a8_table();
    std::lock_guard<std::mutex> lock(table.mutex);
    table.entries[weight_name] = std::move(cfg);
    return true;
}

static bool npu_lookup_aicas_w8a8(
        const struct ggml_tensor * src0,
        npu_aicas_w8a8_config * out) {
    if (out == nullptr || src0 == nullptr || src0->name[0] == '\0') {
        return false;
    }

    npu_aicas_w8a8_table & table = npu_get_aicas_w8a8_table();
    std::lock_guard<std::mutex> lock(table.mutex);
    auto it = table.entries.find(src0->name);
    if (it == table.entries.end()) {
        return false;
    }
    *out = it->second;
    return true;
}

static const char * npu_quant_param_key(
        const struct ggml_tensor * dst,
        const struct ggml_tensor * src1) {
    if (dst != nullptr && dst->name[0] != '\0') {
        return dst->name;
    }
    if (src1 != nullptr && src1->name[0] != '\0') {
        return src1->name;
    }
    return "ggml_npu_quant_sim_unnamed";
}

static void npu_load_static_quant_table_once() {
    npu_static_quant_table & table = npu_get_static_quant_table();
    if (table.loaded) {
        return;
    }
    table.loaded = true;

    const char * path = std::getenv("GGML_NPU_STATIC_ACT_TABLE");
    if (path == nullptr || path[0] == '\0') {
        // Keep compatibility with ggml-cpu static calibration input variable.
        path = std::getenv("GGML_NPU_QUANT_SIM_CALIB_IN");
    }
    if (path == nullptr || path[0] == '\0') {
        return;
    }

    FILE * fp = std::fopen(path, "r");
    if (fp == nullptr) {
        return;
    }

    char line[2048];
    while (std::fgets(line, sizeof(line), fp) != nullptr) {
        char key[GGML_MAX_NAME] = {0};
        int64_t ne0 = -1;
        int64_t ne1 = -1;
        int64_t ne2 = -1;
        int64_t ne3 = -1;
        float min_v = 0.0f;
        float max_v = 0.0f;
        float scale = 1.0f;
        int32_t zp = 0;

        // Full ggml-cpu dump format:
        // key ne0 ne1 ne2 ne3 min max scale zp
        if (std::sscanf(
                line,
                "%63[^\t]\t%" PRId64 "\t%" PRId64 "\t%" PRId64 "\t%" PRId64 "\t%f\t%f\t%f\t%d",
                key, &ne0, &ne1, &ne2, &ne3, &min_v, &max_v, &scale, &zp) == 9) {
            table.entries.push_back(npu_static_quant_entry {
                std::string(key), ne0, ne1, ne2, ne3, scale, zp,
            });
            continue;
        }

        // Compact fallback format: key scale zp
        if (std::sscanf(line, "%63[^\t]\t%f\t%d", key, &scale, &zp) == 3) {
            table.entries.push_back(npu_static_quant_entry {
                std::string(key), -1, -1, -1, -1, scale, zp,
            });
        }
    }
    std::fclose(fp);
}

static npu_activation_quant_config npu_lookup_static_quant_config(
        const struct ggml_tensor * dst,
        const struct ggml_tensor * src1) {
    npu_load_static_quant_table_once();

    npu_activation_quant_config cfg = {};
    cfg.dynamic = false;
    cfg.per_tensor = true;
    cfg.symmetric = false;

    const npu_static_quant_table & table = npu_get_static_quant_table();
    const std::string key = npu_quant_param_key(dst, src1);

    for (const npu_static_quant_entry & entry : table.entries) {
        if (entry.key != key) {
            continue;
        }
        if (entry.ne0 >= 0 &&
            (entry.ne0 != src1->ne[0] ||
             entry.ne1 != src1->ne[1] ||
             entry.ne2 != src1->ne[2] ||
             entry.ne3 != src1->ne[3])) {
            continue;
        }
        cfg.valid = true;
        cfg.scale = entry.scale;
        cfg.zero_point = entry.zero_point;
        return cfg;
    }

    for (const npu_static_quant_entry & entry : table.entries) {
        if (entry.key != key) {
            continue;
        }
        cfg.valid = true;
        cfg.scale = entry.scale;
        cfg.zero_point = entry.zero_point;
        return cfg;
    }

    return cfg;
}

bool npu_is_tensor_2d(const struct ggml_tensor * tensor) {
    return tensor != nullptr && tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

bool npu_is_tensor_plain_contiguous(const struct ggml_tensor * tensor) {
    return tensor != nullptr && ggml_is_contiguous(tensor) && !tensor->view_src;
}

bool npu_is_supported_weight_type(enum ggml_type type) {
    return type == GGML_TYPE_I8 || type == GGML_TYPE_Q8_0 || type == GGML_TYPE_F16 || type == GGML_TYPE_F32;
}

bool npu_is_supported_activation_type(enum ggml_type type) {
    return type == GGML_TYPE_Q8_0 || type == GGML_TYPE_F16 || type == GGML_TYPE_F32;
}

static bool npu_is_bias_tensor_compatible(
        const struct ggml_tensor * bias,
        const struct ggml_tensor * target) {
    if (bias == nullptr || target == nullptr) {
        return false;
    }

    if (bias->type != GGML_TYPE_F32 && bias->type != GGML_TYPE_F16) {
        return false;
    }
    if (bias->op != GGML_OP_NONE) {
        return false;
    }

    if (bias->ne[0] != target->ne[0]) {
        return false;
    }
    if (!(bias->ne[1] == 1 && bias->ne[2] == 1 && bias->ne[3] == 1)) {
        return false;
    }
    return ggml_can_repeat(bias, target);
}

bool npu_is_fusable_bias_add(const struct ggml_tensor * op, std::string * reason) {
    if (op == nullptr || op->op != GGML_OP_ADD) {
        if (reason) {
            *reason = "不是可融合的 ADD";
        }
        return false;
    }

    const struct ggml_tensor * lhs = op->src[0];
    const struct ggml_tensor * rhs = op->src[1];

    if (lhs != nullptr && lhs->op == GGML_OP_MUL_MAT && npu_is_bias_tensor_compatible(rhs, lhs)) {
        return true;
    }
    if (rhs != nullptr && rhs->op == GGML_OP_MUL_MAT && npu_is_bias_tensor_compatible(lhs, rhs)) {
        return true;
    }

    if (reason) {
        *reason = "ADD 不是 MUL_MAT + bias 的可融合模式";
    }
    return false;
}

std::string npu_shape_string(const struct ggml_tensor * tensor) {
    if (tensor == nullptr) {
        return "(null)";
    }

    std::ostringstream oss;
    oss << "[" << tensor->ne[0] << ", " << tensor->ne[1]
        << ", " << tensor->ne[2] << ", " << tensor->ne[3] << "]";
    return oss.str();
}

static bool npu_env_to_size(const char * name, size_t * out) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return false;
    }

    char * end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (end == value || *end != '\0' || parsed == 0) {
        return false;
    }

    *out = static_cast<size_t>(parsed);
    return true;
}

static int64_t npu_round_down_multiple(int64_t value, int64_t multiple) {
    return (value / multiple) * multiple;
}

static int64_t npu_valid_tile_size(int64_t value) {
    if (value < NPU_SA_TILE) {
        return value;
    }
    return npu_round_down_multiple(value, NPU_SA_TILE);
}

static uint32_t npu_align_u32(uint32_t value, uint32_t alignment) {
    if (alignment == 0) {
        return value;
    }
    return (value + alignment - 1u) & ~(alignment - 1u);
}

npu_tiling_config npu_default_tiling_config(void) {
    npu_tiling_config cfg;

    size_t spm_bytes = 0;
    if (npu_env_to_size("GGML_NPU_SPM_BYTES", &spm_bytes)) {
        cfg.spm_bytes = spm_bytes;
    }

    size_t guard_bytes = 0;
    if (npu_env_to_size("GGML_NPU_GUARD_BYTES", &guard_bytes)) {
        cfg.guard_bytes = guard_bytes;
    }

    size_t acc_bytes = 0;
    if (npu_env_to_size("GGML_NPU_ACC_BYTES", &acc_bytes)) {
        cfg.acc_bytes = acc_bytes;
    }

    size_t stage2_k = 0;
    if (npu_env_to_size("GGML_NPU_STAGE2_K_BYTES", &stage2_k)) {
        cfg.stage2_k_block = static_cast<int64_t>(stage2_k);
    }

    // TODO: Thread real SRAM/ACC capacity values from the target platform
    // configuration instead of relying on environment variables only.
    return cfg;
}

bool npu_can_handle_mul_mat(const struct ggml_tensor * op, std::string * reason) {
    if (op == nullptr) {
        if (reason) {
            *reason = "空节点";
        }
        return false;
    }

    const struct ggml_tensor * root = op;
    if (op->op == GGML_OP_ADD) {
        std::string add_reason;
        if (!npu_is_fusable_bias_add(op, &add_reason)) {
            if (reason) {
                *reason = add_reason;
            }
            return false;
        }
        root = (op->src[0] && op->src[0]->op == GGML_OP_MUL_MAT) ? op->src[0] : op->src[1];
    } else if (op->op != GGML_OP_MUL_MAT) {
        if (reason) {
            *reason = "只支持 GGML_OP_MUL_MAT 或其 bias 融合 ADD";
        }
        return false;
    }

    const struct ggml_tensor * src0 = root->src[0];
    const struct ggml_tensor * src1 = root->src[1];

    if (src0 == nullptr || src1 == nullptr) {
        if (reason) {
            *reason = "MUL_MAT 缺少输入";
        }
        return false;
    }

    if (!npu_is_tensor_2d(src0) || !npu_is_tensor_2d(src1) || !npu_is_tensor_2d(op)) {
        if (reason) {
            *reason = "当前仅支持 2D MUL_MAT";
        }
        return false;
    }

    if (!npu_is_tensor_plain_contiguous(src0)) {
        if (reason) {
            *reason = "当前仅支持 contiguous 的权重张量";
        }
        return false;
    }

    if (!npu_is_tensor_plain_contiguous(src1)) {
        if (reason) {
            *reason = "当前仅支持 contiguous 的激活张量";
        }
        return false;
    }

    if (!npu_is_supported_weight_type(src0->type)) {
        if (reason) {
            *reason = "当前仅支持 I8/Q8_0/F16/F32 权重";
        }
        return false;
    }

    if (!npu_is_supported_activation_type(src1->type)) {
        if (reason) {
            *reason = "当前仅支持 Q8_0/F16/F32 激活";
        }
        return false;
    }

    if (op->type != GGML_TYPE_F32) {
        if (reason) {
            *reason = "当前仅支持 F32 输出";
        }
        return false;
    }

    if (src0->ne[0] != src1->ne[0]) {
        if (reason) {
            *reason = "K 维不匹配";
        }
        return false;
    }

    if (src0->type == GGML_TYPE_I8) {
        npu_aicas_w8a8_config cfg;
        if (!npu_lookup_aicas_w8a8(src0, &cfg) || !cfg.valid) {
            if (reason) {
                *reason = "I8 权重缺少 AICAS W8A8 元数据";
            }
            return false;
        }
        if (!(cfg.act_scale > 0.0f)) {
            if (reason) {
                *reason = "AICAS W8A8 激活 scale 非法";
            }
            return false;
        }
        if (cfg.weight_scale.size() != static_cast<size_t>(src0->ne[1]) ||
            cfg.sum_w.size() != static_cast<size_t>(src0->ne[1])) {
            if (reason) {
                *reason = "AICAS W8A8 权重元数据长度与输出通道不匹配";
            }
            return false;
        }
    } else {
        const npu_activation_quant_config act_cfg = npu_lookup_static_quant_config(op, src1);
        if (!act_cfg.valid) {
            if (reason) {
                *reason = "缺少静态非对称量化参数";
            }
            return false;
        }
    }

    return true;
}

static void npu_calculate_auto_gemm_tile(
        int64_t M,
        int64_t N,
        int64_t K,
        int64_t spm_size,
        int64_t acc_size,
        bool requires_output_in_spm,
        int64_t * tm,
        int64_t * tn,
        int64_t * tk) {
    // TODO: If the npux-mlir tiling helper evolves, keep this formula aligned
    // with Gemm.cpp so the first-stage macro tiles stay behaviorally identical.
    const int64_t array_h = NPU_SA_TILE;
    const int64_t array_w = NPU_SA_TILE;
    const int64_t input_dtype_bytes = 1;
    const int64_t output_dtype_bytes = 1;
    const int64_t acc_dtype_bytes = 4;

    const int64_t m_aligned = npu_valid_tile_size(M);
    const int64_t n_aligned = npu_valid_tile_size(N);
    const int64_t min_tm = std::min<int64_t>(M, array_h);
    const int64_t min_tn = std::min<int64_t>(N, array_w);

    const int64_t base_out_spm = requires_output_in_spm
        ? (min_tm * min_tn * output_dtype_bytes)
        : 0;

    int64_t max_tk_spm = 1;
    if (spm_size > base_out_spm) {
        max_tk_spm = (spm_size - base_out_spm) /
            ((min_tm + min_tn) * input_dtype_bytes);
    }
    int64_t tile_k = std::max<int64_t>(1, std::min<int64_t>(K, max_tk_spm));

    const int64_t max_tm_acc = acc_size / (min_tn * acc_dtype_bytes);

    int64_t max_tm_spm = m_aligned;
    const int64_t spm_rem_for_m = spm_size - min_tn * tile_k * input_dtype_bytes;
    if (spm_rem_for_m > 0) {
        int64_t denominator = tile_k * input_dtype_bytes;
        if (requires_output_in_spm) {
            denominator += min_tn * output_dtype_bytes;
        }
        if (denominator > 0) {
            max_tm_spm = spm_rem_for_m / denominator;
        }
    }

    int64_t tile_m = std::min({m_aligned, max_tm_acc, max_tm_spm});
    tile_m = std::max<int64_t>(1, npu_valid_tile_size(tile_m));

    const int64_t max_tn_acc = acc_size / (tile_m * acc_dtype_bytes);

    int64_t max_tn_spm = n_aligned;
    const int64_t spm_rem_for_n = spm_size - tile_k * tile_m * input_dtype_bytes;
    if (spm_rem_for_n > 0) {
        int64_t denominator = tile_k * input_dtype_bytes;
        if (requires_output_in_spm) {
            denominator += tile_m * output_dtype_bytes;
        }
        if (denominator > 0) {
            max_tn_spm = spm_rem_for_n / denominator;
        }
    }

    int64_t tile_n = std::min({n_aligned, max_tn_acc, max_tn_spm});
    tile_n = std::max<int64_t>(1, npu_valid_tile_size(tile_n));

    *tm = tile_m;
    *tn = tile_n;
    *tk = tile_k;
}

static bool npu_assign_runtime_offsets(npu_node_plan * plan, std::string * reason) {
    const int64_t micro_n = std::min<int64_t>(plan->first_stage_tn, plan->config.sa_rows);
    const int64_t micro_m = std::min<int64_t>(plan->first_stage_tm, plan->config.sa_cols);
    const int64_t micro_k = std::min<int64_t>(plan->config.k_block, plan->config.stage2_k_block);

    const uint32_t act_bytes = static_cast<uint32_t>(micro_n * micro_k);
    const uint32_t act_offset = npu_align_u32(0, NPU_SPM_ALIGNMENT);

    const uint32_t weight_offset = npu_align_u32(act_offset + act_bytes, NPU_SPM_ALIGNMENT);
    const uint32_t weight_bytes = static_cast<uint32_t>(micro_m * micro_k);

    const uint32_t spm_end = weight_offset + weight_bytes;
    const uint32_t spm_limit = static_cast<uint32_t>(
        plan->config.spm_bytes > plan->config.guard_bytes
            ? plan->config.spm_bytes - plan->config.guard_bytes
            : plan->config.spm_bytes);

    if (spm_end > spm_limit) {
        if (reason) {
            *reason = "SPM offset allocation overflow";
        }
        return false;
    }

    const uint32_t acc_bytes = static_cast<uint32_t>(micro_n * micro_m * sizeof(int32_t));
    const uint32_t acc_offset = npu_align_u32(0, NPU_ACC_ALIGNMENT);
    const uint32_t acc_end = acc_offset + acc_bytes;
    if (acc_end > plan->config.acc_bytes) {
        if (reason) {
            *reason = "ACC offset allocation overflow";
        }
        return false;
    }

    plan->config.layout.activation = {
        npu_memory_space::spm,
        act_offset,
        act_bytes,
    };
    plan->config.layout.weight = {
        npu_memory_space::spm,
        weight_offset,
        weight_bytes,
    };
    plan->config.layout.accumulator = {
        npu_memory_space::acc,
        acc_offset,
        acc_bytes,
    };

    return true;
}

static npu_loop_stage npu_make_stage(bool first_k, bool last_k) {
    if (first_k && last_k) {
        return npu_loop_stage::single;
    }
    if (first_k) {
        return npu_loop_stage::head;
    }
    if (last_k) {
        return npu_loop_stage::tail;
    }
    return npu_loop_stage::body;
}

static int32_t npu_find_or_create_weight_pack(
        npu_node_plan * plan,
        int64_t m0,
        int64_t m,
        int64_t k0,
        int64_t k,
        std::string * error) {
    for (size_t i = 0; i < plan->weight_packs.size(); ++i) {
        const npu_prepacked_weight & existing = plan->weight_packs[i];
        if (existing.m0 == m0 && existing.m == m && existing.k0 == k0 && existing.k == k) {
            return static_cast<int32_t>(i);
        }
    }

    npu_prepacked_weight packed;
    packed.m0 = m0;
    packed.m = m;
    packed.k0 = k0;
    packed.k = k;
    if (plan->aicas_w8a8.valid) {
        if (m0 < 0 || m < 0 || static_cast<size_t>(m0 + m) > plan->aicas_w8a8.weight_scale.size()) {
            if (error) {
                *error = "AICAS W8A8 weight_scale 越界";
            }
            return -1;
        }
        packed.scales.assign(
            plan->aicas_w8a8.weight_scale.begin() + m0,
            plan->aicas_w8a8.weight_scale.begin() + m0 + m);
        if (!npu_pack_weight_tile_prequant_i8_transposed(
                    plan->src0,
                    m0,
                    m,
                    k0,
                    k,
                    &packed.packed,
                    error)) {
            return -1;
        }
    } else {
        packed.scales = npu_compute_weight_output_scales(plan->src0, m0, m);
        if (!npu_pack_weight_tile_fixed_i8_transposed(
                    plan->src0,
                    m0,
                    m,
                    k0,
                    k,
                    packed.scales.data(),
                    &packed.packed,
                    error)) {
            return -1;
        }
    }

    plan->weight_packs.push_back(std::move(packed));
    return static_cast<int32_t>(plan->weight_packs.size() - 1);
}

static float npu_read_bias_value_f32_plan(
        const struct ggml_tensor * bias,
        int64_t m,
        int64_t n) {
    if (bias == nullptr) {
        return 0.0f;
    }

    const int64_t i0 = bias->ne[0] == 1 ? 0 : m;
    const int64_t i1 = bias->ne[1] == 1 ? 0 : n;
    const char * base = static_cast<const char *>(bias->data) + i1 * bias->nb[1] + i0 * bias->nb[0];

    switch (bias->type) {
        case GGML_TYPE_F32:
            return *reinterpret_cast<const float *>(base);
        case GGML_TYPE_F16:
            return ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t *>(base));
        default:
            return 0.0f;
    }
}

static void npu_build_weight_column_sum_q(npu_node_plan * plan) {
    if (plan->aicas_w8a8.valid &&
        plan->aicas_w8a8.sum_w.size() == static_cast<size_t>(plan->m)) {
        plan->weight_column_sum_q = plan->aicas_w8a8.sum_w;
        return;
    }

    plan->weight_column_sum_q.assign(static_cast<size_t>(plan->m), 0);
    for (const npu_prepacked_weight & pack : plan->weight_packs) {
        for (int64_t local_m = 0; local_m < pack.m; ++local_m) {
            int32_t sum = 0;
            for (int64_t k = 0; k < pack.k; ++k) {
                sum += static_cast<int32_t>(pack.packed[static_cast<size_t>(k * pack.m + local_m)]);
            }
            plan->weight_column_sum_q[static_cast<size_t>(pack.m0 + local_m)] += sum;
        }
    }
}

static int32_t npu_find_or_create_bias_pack(
        npu_node_plan * plan,
        const npu_exec_tile & exec_tile,
        std::string * error) {
    for (size_t i = 0; i < plan->bias_packs.size(); ++i) {
        const npu_prepacked_bias & existing = plan->bias_packs[i];
        if (existing.m0 == exec_tile.m0 &&
            existing.n0 == exec_tile.n0 &&
            existing.m == exec_tile.m) {
            return static_cast<int32_t>(i);
        }
    }

    if (exec_tile.weight_pack_index < 0 ||
        static_cast<size_t>(exec_tile.weight_pack_index) >= plan->weight_packs.size()) {
        if (error) {
            *error = "bias pack requires valid weight_pack_index";
        }
        return -1;
    }

    const npu_prepacked_weight & wpack =
        plan->weight_packs[static_cast<size_t>(exec_tile.weight_pack_index)];

    npu_prepacked_bias packed;
    packed.m0 = exec_tile.m0;
    packed.n0 = exec_tile.n0;
    packed.m = exec_tile.m;
    packed.values.assign(static_cast<size_t>(exec_tile.m), 0);

    const float act_scale = plan->activation_quant.scale;
    const int32_t act_zp = plan->activation_quant.zero_point;

    for (int64_t m = 0; m < exec_tile.m; ++m) {
        const int64_t global_m = exec_tile.m0 + m;
        const float bias_f32 = npu_read_bias_value_f32_plan(plan->bias, global_m, exec_tile.n0);
        const float denom = act_scale * wpack.scales[static_cast<size_t>(m)];
        const float inv = denom != 0.0f ? (bias_f32 / denom) : 0.0f;
        int32_t sum_qw = 0;
        if (global_m >= 0 && static_cast<size_t>(global_m) < plan->weight_column_sum_q.size()) {
            sum_qw = plan->weight_column_sum_q[static_cast<size_t>(global_m)];
        }
        packed.values[static_cast<size_t>(m)] =
            static_cast<int32_t>(std::lrint(inv)) - act_zp * sum_qw;
    }

    plan->bias_packs.push_back(std::move(packed));
    return static_cast<int32_t>(plan->bias_packs.size() - 1);
}

npu_node_plan npu_create_mul_mat_plan(struct ggml_tensor * op, const npu_tiling_config & config) {
    struct ggml_tensor * compute_root = op;
    const struct ggml_tensor * bias = nullptr;
    if (op->op == GGML_OP_ADD) {
        if (op->src[0] != nullptr && op->src[0]->op == GGML_OP_MUL_MAT) {
            compute_root = op->src[0];
            bias = op->src[1];
        } else {
            compute_root = op->src[1];
            bias = op->src[0];
        }
    }

    npu_node_plan plan;
    plan.op = compute_root;
    plan.root = op;
    plan.src0 = compute_root->src[0];
    plan.src1 = compute_root->src[1];
    plan.bias = bias;
    plan.dst = op;
    plan.m = op->ne[0];
    plan.n = op->ne[1];
    plan.k = compute_root->src[0]->ne[0];
    if (plan.src0 != nullptr && plan.src0->type == GGML_TYPE_I8) {
        npu_aicas_w8a8_config cfg;
        if (npu_lookup_aicas_w8a8(plan.src0, &cfg) &&
            cfg.valid &&
            cfg.weight_scale.size() == static_cast<size_t>(plan.m) &&
            cfg.sum_w.size() == static_cast<size_t>(plan.m)) {
            plan.aicas_w8a8 = std::move(cfg);
            plan.activation_quant.dynamic = false;
            plan.activation_quant.per_tensor = true;
            plan.activation_quant.symmetric = false;
            plan.activation_quant.valid = true;
            plan.activation_quant.scale = plan.aicas_w8a8.act_scale;
            plan.activation_quant.zero_point = plan.aicas_w8a8.act_zero_point_i8;
        }
    }
    if (!plan.activation_quant.valid) {
        plan.activation_quant = npu_lookup_static_quant_config(plan.dst, plan.src1);
    }
    plan.config = config;

    // TODO: Preserve more npux-mlir metadata here later, especially loop-stage
    // semantics like head/body/tail and explicit DMA planning info.
    // TODO: Current ggml-npu planning is still per-node. NpuMemPlan.cpp does
    // module-level alloc/free/reuse with first-fit allocators, so if we later
    // plan multiple NPU ops together we should switch to a graph/global memory
    // planner instead of these fixed working-set slices.
    int64_t first_stage_tm = 0;
    int64_t first_stage_tn = 0;
    int64_t first_stage_tk = 0;
    npu_calculate_auto_gemm_tile(
        plan.m,
        plan.n,
        plan.k,
        static_cast<int64_t>(plan.config.spm_bytes > plan.config.guard_bytes
            ? plan.config.spm_bytes - plan.config.guard_bytes
            : plan.config.spm_bytes),
        static_cast<int64_t>(plan.config.acc_bytes),
        plan.config.output_in_spm,
        &first_stage_tm,
        &first_stage_tn,
        &first_stage_tk);

    plan.first_stage_tm = first_stage_tm;
    plan.first_stage_tn = first_stage_tn;
    plan.first_stage_tk = first_stage_tk;

    if (plan.config.k_block <= 0) {
        plan.config.k_block = first_stage_tk;
    }
    if (plan.config.stage2_k_block <= 0) {
        plan.config.stage2_k_block = NPU_STAGE2_K_TILE;
    }

    for (int64_t n0 = 0; n0 < plan.n; n0 += first_stage_tn) {
        const int64_t n_rows = std::min<int64_t>(first_stage_tn, plan.n - n0);
        for (int64_t m0 = 0; m0 < plan.m; m0 += first_stage_tm) {
            const int64_t m_cols = std::min<int64_t>(first_stage_tm, plan.m - m0);
            plan.mn_tiles.push_back({m0, n0, m_cols, n_rows});
        }
    }

    for (int64_t k0 = 0; k0 < plan.k; k0 += plan.config.k_block) {
        const int64_t k_cols = std::min<int64_t>(plan.config.k_block, plan.k - k0);
        plan.k_tiles.push_back({
            k0,
            k_cols,
            k0 == 0,
            k0 + k_cols >= plan.k,
            static_cast<size_t>(plan.config.sa_rows * k_cols),
            static_cast<size_t>(plan.config.sa_cols * k_cols),
        });
    }

    for (const npu_mn_tile & macro_tile : plan.mn_tiles) {
        for (int64_t micro_n0 = macro_tile.n0; micro_n0 < macro_tile.n0 + macro_tile.n; micro_n0 += plan.config.sa_rows) {
            const int64_t micro_n = std::min<int64_t>(plan.config.sa_rows, macro_tile.n0 + macro_tile.n - micro_n0);
            for (int64_t micro_m0 = macro_tile.m0; micro_m0 < macro_tile.m0 + macro_tile.m; micro_m0 += plan.config.sa_cols) {
                const int64_t micro_m = std::min<int64_t>(plan.config.sa_cols, macro_tile.m0 + macro_tile.m - micro_m0);
                for (const npu_k_tile & k_tile : plan.k_tiles) {
                    for (int64_t sub_k0 = k_tile.k0; sub_k0 < k_tile.k0 + k_tile.k; sub_k0 += plan.config.stage2_k_block) {
                        const int64_t sub_k = std::min<int64_t>(plan.config.stage2_k_block, k_tile.k0 + k_tile.k - sub_k0);
                        const bool first_k_global = (sub_k0 == 0);
                        const bool last_k_global = (sub_k0 + sub_k >= plan.k);
                        const npu_loop_stage stage = npu_make_stage(first_k_global, last_k_global);
                        std::string pack_error;
                        const int32_t weight_pack_index = npu_find_or_create_weight_pack(
                            &plan, micro_m0, micro_m, sub_k0, sub_k, &pack_error);
                        if (weight_pack_index < 0) {
                            plan.summary = std::string("weight pack failed: ") + pack_error;
                            return plan;
                        }
                        plan.exec_tiles.push_back({
                            micro_m0,
                            micro_n0,
                            micro_m,
                            micro_n,
                            sub_k0,
                            sub_k,
                            stage,
                            first_k_global,
                            last_k_global,
                            weight_pack_index,
                            -1,
                        });
                    }
                }
            }
        }
    }

    npu_build_weight_column_sum_q(&plan);

    if (plan.bias != nullptr && plan.activation_quant.valid) {
        for (npu_exec_tile & exec_tile : plan.exec_tiles) {
            if (!exec_tile.needs_bias) {
                continue;
            }
            std::string bias_error;
            const int32_t bias_pack_index = npu_find_or_create_bias_pack(&plan, exec_tile, &bias_error);
            if (bias_pack_index < 0) {
                plan.summary = std::string("bias pack failed: ") + bias_error;
                return plan;
            }
            exec_tile.bias_pack_index = bias_pack_index;
        }
    }

    std::string offset_reason;
    if (!npu_assign_runtime_offsets(&plan, &offset_reason)) {
        // TODO: Surface this failure through a structured diagnostic path instead
        // of keeping the invalid layout in the summary string only.
        plan.summary = std::string("offset allocation failed: ") + offset_reason;
        return plan;
    }

    std::ostringstream oss;
    oss << "M=" << plan.m
        << ", N=" << plan.n
        << ", K=" << plan.k
        << ", w8a8_source=" << (plan.aicas_w8a8.valid ? "aicas_metadata" : "static_act_table")
        << ", macro_tm=" << first_stage_tm
        << ", macro_tn=" << first_stage_tn
        << ", macro_tk=" << plan.config.k_block
        << ", mn_tiles=" << plan.mn_tiles.size()
        << ", k_tiles=" << plan.k_tiles.size()
        << ", weight_packs=" << plan.weight_packs.size()
        << ", bias_packs=" << plan.bias_packs.size()
        << ", exec_tiles=" << plan.exec_tiles.size()
        << ", stage2_mn=32x32"
        << ", stage2_k=" << plan.config.stage2_k_block
        << ", act_quant=static_asymmetric"
        << ", act_quant_valid=" << (plan.activation_quant.valid ? "true" : "false")
        << ", act_scale=" << plan.activation_quant.scale
        << ", act_zp=" << plan.activation_quant.zero_point
        << ", output_in_spm=" << (plan.config.output_in_spm ? "true" : "false")
        << ", spm_bytes=" << plan.config.spm_bytes
        << ", acc_bytes=" << plan.config.acc_bytes
        << ", act_off=" << plan.config.layout.activation.offset
        << ", wgt_off=" << plan.config.layout.weight.offset
        << ", acc_off=" << plan.config.layout.accumulator.offset;
    plan.summary = oss.str();

    return plan;
}

bool npu_plan_is_aot_stable(const npu_node_plan & plan, std::string * reason) {
    if (plan.m <= 0 || plan.n <= 0 || plan.k <= 0) {
        if (reason) {
            *reason = "M/N/K 未静态确定";
        }
        return false;
    }

    if (plan.config.layout.activation.bytes == 0 ||
        plan.config.layout.weight.bytes == 0 ||
        plan.config.layout.accumulator.bytes == 0) {
        if (reason) {
            *reason = "AOT 地址分配未完成";
        }
        return false;
    }

    if (!plan.activation_quant.valid) {
        if (reason) {
            *reason = "缺少静态非对称量化参数，请设置 GGML_NPU_STATIC_ACT_TABLE 或 GGML_NPU_QUANT_SIM_CALIB_IN";
        }
        return false;
    }
    if (!(plan.activation_quant.scale > 0.0f)) {
        if (reason) {
            *reason = "静态量化参数 scale 非法";
        }
        return false;
    }

    // AOT-stable here means all geometry, memory offsets and static activation
    // quant params are fixed before execution.
    return true;
}

} // namespace ggml_npu
