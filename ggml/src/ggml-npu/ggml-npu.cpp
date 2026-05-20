#include "ggml-npu.h"

#include "ggml-npu-common.h"
#include "ggml-npu-exec.h"
#include "ggml-npu-plan.h"
#include "ggml-npu-profile.h"

#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "npu_runtime.h"

#include <algorithm>
#include <cinttypes>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <string>
#include <unordered_set>

namespace ggml_npu {

static bool npu_debug_log_enabled() {
    return std::getenv("GGML_NPU_DEBUG_LOG") != nullptr || std::getenv("AICAS_MMPROJ_W8A8_DEBUG") != nullptr;
}

static bool npu_runtime_profile_requested() {
    const char * path = std::getenv("NPU_PROFILE_OUT");
    return path != nullptr && path[0] != '\0';
}

static bool npu_eager_init_enabled() {
    const char * v = std::getenv("GGML_NPU_EAGER_INIT");
    return v != nullptr && v[0] != '\0' && std::strcmp(v, "0") != 0;
}

static bool npu_raw_i8_gemm_debug_log_enabled() {
    return std::getenv("GGML_NPU_DEBUG_LOG") != nullptr || std::getenv("AICAS_MMPROJ_BFP8M_NPU_DEBUG") != nullptr;
}

static bool npu_fits_u16(int64_t value) {
    return value >= 0 && value <= std::numeric_limits<uint16_t>::max();
}

static bool npu_raw_i8_gemm_validate(
        const char * op_name,
        const int8_t * weight_kxm,
        int64_t m,
        int64_t k,
        int64_t weight_stride_m,
        const int8_t * act_nxk,
        int64_t n,
        int64_t act_stride_k,
        const int32_t * out_nxm,
        int64_t out_stride_m,
        std::string * error) {
    auto fail = [op_name, error](const char * reason) {
        if (error != nullptr) {
            *error = reason;
        }
        if (npu_raw_i8_gemm_debug_log_enabled()) {
            GGML_LOG_WARN("%s: reject %s: %s\n", __func__, op_name != nullptr ? op_name : "(unnamed)", reason);
        }
        return false;
    };

    if (weight_kxm == nullptr || act_nxk == nullptr || out_nxm == nullptr) {
        return fail("null pointer");
    }
    if (m <= 0 || n <= 0 || k <= 0) {
        return fail("non-positive dimension");
    }
    if (weight_stride_m < m || act_stride_k < k || out_stride_m < m) {
        return fail("stride smaller than dimension");
    }
    if (!npu_fits_u16(m) || !npu_fits_u16(n) || !npu_fits_u16(k) ||
            !npu_fits_u16(weight_stride_m) || !npu_fits_u16(act_stride_k) || !npu_fits_u16(out_stride_m)) {
        return fail("dimension or stride exceeds uint16 range");
    }
    if (!npu_is_aligned_i64(weight_stride_m, NPU_GEMM_PLAN_STRIDE_ALIGNMENT) ||
            !npu_is_aligned_i64(act_stride_k, NPU_GEMM_PLAN_STRIDE_ALIGNMENT) ||
            !npu_is_aligned_i64(out_stride_m, NPU_GEMM_PLAN_STRIDE_ALIGNMENT)) {
        return fail("stride is not NPU GEMM-plan aligned");
    }
    const uint64_t act_bytes = static_cast<uint64_t>(n) * static_cast<uint64_t>(act_stride_k);
    const uint64_t weight_bytes = static_cast<uint64_t>(k) * static_cast<uint64_t>(weight_stride_m);
    const uint64_t out_bytes = static_cast<uint64_t>(n) * static_cast<uint64_t>(out_stride_m) * sizeof(int32_t);
    if (act_bytes > std::numeric_limits<size_t>::max() ||
            weight_bytes > std::numeric_limits<size_t>::max() ||
            out_bytes > std::numeric_limits<size_t>::max()) {
        return fail("host buffer size overflow");
    }

    return true;
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

static ggml_backend_buffer_t npu_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    npu_buffer_context * ctx = new (std::nothrow) npu_buffer_context;
    if (ctx == nullptr) {
        return nullptr;
    }

    ctx->ptr = ggml_aligned_malloc(size);
    ctx->own = true;

    if (ctx->ptr == nullptr) {
        delete ctx;
        return nullptr;
    }

    return ggml_backend_buffer_init(buft, npu_buffer_iface, ctx, size);
}

static size_t npu_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    GGML_UNUSED(buft);
    return ggml_nbytes(tensor);
}

static const char * npu_buffer_type_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return "NPU_Host";
}

static size_t npu_buffer_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return 64;
}

static size_t npu_buffer_max_size(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return SIZE_MAX;
}

static bool npu_buffer_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return true;
}

static const ggml_backend_buffer_type_i npu_buffer_type_iface = {
    /* .get_name       = */ npu_buffer_type_name,
    /* .alloc_buffer   = */ npu_alloc_buffer,
    /* .get_alignment  = */ npu_buffer_alignment,
    /* .get_max_size   = */ npu_buffer_max_size,
    /* .get_alloc_size = */ npu_get_alloc_size,
    /* .is_host        = */ npu_buffer_is_host,
};

static ggml_guid_t ggml_backend_npu_guid(void) {
    static ggml_guid guid = {
        0x79, 0x13, 0xa2, 0x5d, 0x4a, 0x5f, 0x4e, 0x38,
        0x9a, 0x9f, 0x42, 0x61, 0x43, 0x10, 0x61, 0x01
    };
    return &guid;
}

static const char * npu_backend_name(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return "NPU";
}

static void npu_backend_free(ggml_backend_t backend) {
    delete static_cast<npu_backend_context *>(backend->context);
    delete backend;
}

static ggml_backend_graph_plan_t npu_backend_graph_plan_create(
        ggml_backend_t backend,
        const struct ggml_cgraph * cgraph) {
    GGML_UNUSED(backend);

    auto * graph_plan = new (std::nothrow) npu_graph_plan;
    if (graph_plan == nullptr) {
        return nullptr;
    }

    std::unordered_set<const struct ggml_tensor *> fused_mul_mat_roots;
    fused_mul_mat_roots.reserve(static_cast<size_t>(cgraph->n_nodes));
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        struct ggml_tensor * node = cgraph->nodes[i];
        if (node == nullptr || node->op != GGML_OP_ADD) {
            continue;
        }
        if (!npu_is_fusable_bias_add(node, nullptr)) {
            continue;
        }
        if (node->src[0] != nullptr && node->src[0]->op == GGML_OP_MUL_MAT) {
            fused_mul_mat_roots.insert(node->src[0]);
        } else if (node->src[1] != nullptr && node->src[1]->op == GGML_OP_MUL_MAT) {
            fused_mul_mat_roots.insert(node->src[1]);
        }
    }

    const npu_backend_context * ctx = static_cast<const npu_backend_context *>(backend->context);
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        struct ggml_tensor * node = cgraph->nodes[i];
        if (node == nullptr) {
            continue;
        }
        if (node->op == GGML_OP_MUL_MAT && fused_mul_mat_roots.find(node) != fused_mul_mat_roots.end()) {
            if (npu_debug_log_enabled()) {
                const char * node_name = node->name[0] != '\0' ? node->name : "(unnamed)";
                GGML_LOG_INFO("%s: skip node=%s op=%s reason=%s\n",
                        __func__,
                        node_name,
                        ggml_op_name(node->op),
                        "covered by fused ADD");
            }
            continue;
        }
        std::string reason;
        if (!npu_can_handle_mul_mat(node, &reason)) {
            if (npu_debug_log_enabled()) {
                const char * node_name = node && node->name[0] != '\0' ? node->name : "(unnamed)";
                GGML_LOG_INFO("%s: skip node=%s op=%s reason=%s\n",
                        __func__,
                        node_name,
                        node ? ggml_op_name(node->op) : "(null)",
                        reason.c_str());
            }
            continue;
        }
        graph_plan->nodes.push_back(npu_create_mul_mat_plan(node, ctx->config));
    }

    if (npu_debug_log_enabled()) {
        GGML_LOG_INFO("%s: created graph plan with %zu NPU nodes out of %d graph nodes\n",
                __func__, graph_plan->nodes.size(), cgraph->n_nodes);
        for (size_t i = 0; i < graph_plan->nodes.size(); ++i) {
            const npu_node_plan & node_plan = graph_plan->nodes[i];
            const char * root_name = node_plan.root && node_plan.root->name[0] != '\0' ? node_plan.root->name : "(unnamed)";
            GGML_LOG_INFO("%s: node[%zu] root=%s op=%s summary=%s\n",
                    __func__,
                    i,
                    root_name,
                    node_plan.root ? ggml_op_name(node_plan.root->op) : "(null)",
                    node_plan.summary.c_str());
        }
    }

    return graph_plan;
}

static void npu_backend_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan_ptr) {
    GGML_UNUSED(backend);
    npu_graph_plan * graph_plan = static_cast<npu_graph_plan *>(plan_ptr);
    if (graph_plan != nullptr) {
        for (npu_node_plan & node_plan : graph_plan->nodes) {
            for (npu_prepacked_weight & weight_pack : node_plan.weight_packs) {
                if (weight_pack.cma_packed != nullptr && !weight_pack.cma_persistent) {
                    npu_mem_free(weight_pack.cma_packed);
                    weight_pack.cma_packed = nullptr;
                    weight_pack.cma_bytes = 0;
                    weight_pack.cma_persistent = false;
                }
            }
        }
    }
    delete graph_plan;
}

static enum ggml_status npu_backend_graph_plan_compute(
        ggml_backend_t backend,
        ggml_backend_graph_plan_t plan_ptr) {
    GGML_UNUSED(backend);

    npu_graph_plan * graph_plan = static_cast<npu_graph_plan *>(plan_ptr);
    if (graph_plan == nullptr) {
        return GGML_STATUS_FAILED;
    }

    if ((npu_profile_enabled() || npu_runtime_profile_requested() || npu_summary_active()) && !npu_profile_session_started()) {
        npu_profile_reset();
        npu_profile_next_layer_id() = 0;
        npu_profile_session_started() = true;
    }

    for (size_t i = 0; i < graph_plan->nodes.size(); ++i) {
        const npu_node_plan & node_plan = graph_plan->nodes[i];
        const int64_t layer_id = npu_profile_next_layer_id()++;
        if (npu_debug_log_enabled()) {
            const char * root_name = node_plan.root && node_plan.root->name[0] != '\0' ? node_plan.root->name : "(unnamed)";
            GGML_LOG_INFO("%s: start root=%s op=%s summary=%s\n",
                    __func__,
                    root_name,
                    node_plan.root ? ggml_op_name(node_plan.root->op) : "(null)",
                    node_plan.summary.c_str());
        }
        std::string error;
        const enum ggml_status status = npu_compute_node(node_plan, layer_id, &error);
        if (status != GGML_STATUS_SUCCESS) {
            npu_profile_flush();
            if (npu_debug_log_enabled()) {
                const char * root_name = node_plan.root && node_plan.root->name[0] != '\0' ? node_plan.root->name : "(unnamed)";
                GGML_LOG_ERROR("%s: failed root=%s status=%d error=%s\n",
                        __func__,
                        root_name,
                        status,
                        error.c_str());
            }
            return status;
        }
        if (npu_debug_log_enabled()) {
            const char * root_name = node_plan.root && node_plan.root->name[0] != '\0' ? node_plan.root->name : "(unnamed)";
            GGML_LOG_INFO("%s: done root=%s\n", __func__, root_name);
        }
    }

    npu_profile_flush();
    return GGML_STATUS_SUCCESS;
}

static enum ggml_status npu_backend_graph_compute(
        ggml_backend_t backend,
        struct ggml_cgraph * cgraph) {
    ggml_backend_graph_plan_t plan = npu_backend_graph_plan_create(backend, cgraph);
    if (plan == nullptr) {
        return GGML_STATUS_FAILED;
    }

    const enum ggml_status status = npu_backend_graph_plan_compute(backend, plan);
    npu_backend_graph_plan_free(backend, plan);
    return status;
}

static const ggml_backend_i npu_backend_iface = {
    /* .get_name           = */ npu_backend_name,
    /* .free               = */ npu_backend_free,
    /* .set_tensor_async   = */ nullptr,
    /* .get_tensor_async   = */ nullptr,
    /* .cpy_tensor_async   = */ nullptr,
    /* .synchronize        = */ nullptr,
    /* .graph_plan_create  = */ npu_backend_graph_plan_create,
    /* .graph_plan_free    = */ npu_backend_graph_plan_free,
    /* .graph_plan_update  = */ nullptr,
    /* .graph_plan_compute = */ npu_backend_graph_plan_compute,
    /* .graph_compute      = */ npu_backend_graph_compute,
    /* .event_record       = */ nullptr,
    /* .event_wait         = */ nullptr,
    /* .graph_optimize     = */ nullptr,
};

static const char * npu_device_name(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return "NPU0";
}

static const char * npu_device_description(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return "AICAS NPU INT8 matmul offload prototype";
}

static void npu_device_memory(ggml_backend_dev_t dev, size_t * free_mem, size_t * total_mem) {
    GGML_UNUSED(dev);
    if (free_mem) {
        *free_mem = 0;
    }
    if (total_mem) {
        *total_mem = NPU_DEFAULT_SPM_BYTES;
    }
}

static enum ggml_backend_dev_type npu_device_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

static void npu_device_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name = npu_device_name(dev);
    props->description = npu_device_description(dev);
    props->memory_free = 0;
    props->memory_total = NPU_DEFAULT_SPM_BYTES;
    props->type = GGML_BACKEND_DEVICE_TYPE_ACCEL;
    props->device_id = "aicas-npu0";
    props->caps = {
        /* .async            = */ false,
        /* .host_buffer      = */ true,
        /* .buffer_from_host_ptr = */ true,
        /* .events           = */ false,
    };
}

static ggml_backend_buffer_type_t npu_device_get_buffer_type(ggml_backend_dev_t dev);

static ggml_backend_t npu_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);

    auto * ctx = new (std::nothrow) npu_backend_context;
    if (ctx == nullptr) {
        return nullptr;
    }

    ctx->config = npu_default_tiling_config();

    ggml_backend_t backend = new (std::nothrow) ggml_backend {
        /* .guid    = */ ggml_backend_npu_guid(),
        /* .iface   = */ npu_backend_iface,
        /* .device  = */ dev,
        /* .context = */ ctx,
    };

    if (backend == nullptr) {
        delete ctx;
        return nullptr;
    }

    if (npu_eager_init_enabled() && npu_init() != 0) {
        delete backend;
        delete ctx;
        return nullptr;
    }

    return backend;
}

static ggml_backend_buffer_t npu_device_buffer_from_host_ptr(
        ggml_backend_dev_t dev,
        void * ptr,
        size_t size,
        size_t max_tensor_size) {
    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);

    auto * ctx = new (std::nothrow) npu_buffer_context;
    if (ctx == nullptr) {
        return nullptr;
    }

    ctx->ptr = ptr;
    ctx->own = false;

    return ggml_backend_buffer_init(
        ggml_backend_npu_buffer_type(),
        npu_buffer_iface,
        ctx,
        size);
}

static bool npu_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    GGML_UNUSED(dev);
    return npu_can_handle_mul_mat(op, nullptr);
}

static bool npu_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(dev);
    return ggml_backend_buft_is_host(buft) || buft == ggml_backend_npu_buffer_type();
}

static const ggml_backend_device_i npu_device_iface = {
    /* .get_name             = */ npu_device_name,
    /* .get_description      = */ npu_device_description,
    /* .get_memory           = */ npu_device_memory,
    /* .get_type             = */ npu_device_type,
    /* .get_props            = */ npu_device_props,
    /* .init_backend         = */ npu_device_init_backend,
    /* .get_buffer_type      = */ npu_device_get_buffer_type,
    /* .get_host_buffer_type = */ npu_device_get_buffer_type,
    /* .buffer_from_host_ptr = */ npu_device_buffer_from_host_ptr,
    /* .supports_op          = */ npu_device_supports_op,
    /* .supports_buft        = */ npu_device_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

struct npu_reg_context {
    ggml_backend_device device;
    ggml_backend_buffer_type buffer_type;
};

static const char * npu_reg_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return "NPU";
}

static size_t npu_reg_device_count(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return 1;
}

static ggml_backend_dev_t npu_reg_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);
    npu_reg_context * ctx = static_cast<npu_reg_context *>(reg->context);
    return &ctx->device;
}

static void * npu_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);

    if (std::strcmp(name, "ggml_backend_npu_init") == 0) {
        return reinterpret_cast<void *>(ggml_backend_npu_init);
    }
    if (std::strcmp(name, "ggml_backend_is_npu") == 0) {
        return reinterpret_cast<void *>(ggml_backend_is_npu);
    }
    if (std::strcmp(name, "ggml_backend_npu_w8a8_register") == 0) {
        return reinterpret_cast<void *>(ggml_backend_npu_w8a8_register);
    }
    if (std::strcmp(name, "ggml_backend_npu_w8a8_preload") == 0) {
        return reinterpret_cast<void *>(ggml_backend_npu_w8a8_preload);
    }
    if (std::strcmp(name, "ggml_backend_npu_w8a8_preload_clear") == 0) {
        return reinterpret_cast<void *>(ggml_backend_npu_w8a8_preload_clear);
    }
    if (std::strcmp(name, "ggml_backend_npu_w8a8_clear") == 0) {
        return reinterpret_cast<void *>(ggml_backend_npu_w8a8_clear);
    }
    if (std::strcmp(name, "ggml_backend_npu_i8_gemm_raw_packed") == 0) {
        return reinterpret_cast<void *>(ggml_backend_npu_i8_gemm_raw_packed);
    }
    if (std::strcmp(name, "ggml_backend_npu_i8_gemm_raw_cma") == 0) {
        return reinterpret_cast<void *>(ggml_backend_npu_i8_gemm_raw_cma);
    }
    if (std::strcmp(name, "ggml_backend_npu_mem_alloc") == 0) {
        return reinterpret_cast<void *>(ggml_backend_npu_mem_alloc);
    }
    if (std::strcmp(name, "ggml_backend_npu_mem_free") == 0) {
        return reinterpret_cast<void *>(ggml_backend_npu_mem_free);
    }
    return nullptr;
}

static const ggml_backend_reg_i npu_reg_iface = {
    /* .get_name         = */ npu_reg_name,
    /* .get_device_count = */ npu_reg_device_count,
    /* .get_device       = */ npu_reg_device,
    /* .get_proc_address = */ npu_reg_get_proc_address,
};

static ggml_backend_buffer_type_t npu_device_get_buffer_type(ggml_backend_dev_t dev) {
    npu_reg_context * ctx = static_cast<npu_reg_context *>(dev->reg->context);
    return &ctx->buffer_type;
}

} // namespace ggml_npu

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
                static_cast<uint32_t>(cur_act_bytes - 1),
                0,
                0,
                0,
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
                static_cast<uint32_t>(cur_weight_bytes - 1),
                0,
                0,
                0,
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

GGML_BACKEND_DL_IMPL(ggml_backend_npu_reg)
