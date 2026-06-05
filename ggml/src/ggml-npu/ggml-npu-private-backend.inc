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
    npu_profile_flush();
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

    if (!npu_profile_aggregate_only()) {
        npu_profile_flush();
    }
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
    if (std::strcmp(name, "ggml_backend_npu_decode_w8a16_preload") == 0) {
        return reinterpret_cast<void *>(ggml_backend_npu_decode_w8a16_preload);
    }
    if (std::strcmp(name, "ggml_backend_npu_decode_w8a16_gemv_ex") == 0) {
        return reinterpret_cast<void *>(ggml_backend_npu_decode_w8a16_gemv_ex);
    }
    if (std::strcmp(name, "ggml_backend_npu_decode_attention_w4a16_ex") == 0) {
        return reinterpret_cast<void *>(ggml_backend_npu_decode_attention_w4a16_ex);
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
    if (std::strcmp(name, "ggml_backend_npu_text_log8pv_attention") == 0) {
        return reinterpret_cast<void *>(ggml_backend_npu_text_log8pv_attention);
    }
    if (std::strcmp(name, "ggml_backend_npu_log8pv_attention_ex") == 0) {
        return reinterpret_cast<void *>(ggml_backend_npu_log8pv_attention_ex);
    }
    if (std::strcmp(name, "ggml_backend_npu_log8pv_attention_group") == 0) {
        return reinterpret_cast<void *>(ggml_backend_npu_log8pv_attention_group);
    }
    if (std::strcmp(name, "ggml_backend_npu_runtime_shutdown") == 0) {
        return reinterpret_cast<void *>(ggml_backend_npu_runtime_shutdown);
    }
    if (std::strcmp(name, "ggml_backend_npu_decode_runtime_shutdown") == 0) {
        return reinterpret_cast<void *>(ggml_backend_npu_decode_runtime_shutdown);
    }
    if (std::strcmp(name, "ggml_backend_npu_decode_overlay_mark_inactive") == 0) {
        return reinterpret_cast<void *>(ggml_backend_npu_decode_overlay_mark_inactive);
    }
    if (std::strcmp(name, "ggml_backend_npu_decode_overlay_mark_active") == 0) {
        return reinterpret_cast<void *>(ggml_backend_npu_decode_overlay_mark_active);
    }
if (std::strcmp(name, "ggml_backend_npu_decode_w4a16_gemv_ex") == 0) {
        return reinterpret_cast<void *>(ggml_backend_npu_decode_w4a16_gemv_ex);
    }
if (std::strcmp(name, "ggml_backend_npu_decode_w4a16_preload") == 0) {
        return reinterpret_cast<void *>(ggml_backend_npu_decode_w4a16_preload);
    }
    if (std::strcmp(name, "ggml_backend_npu_decode_w16a16_preload") == 0) {
        return reinterpret_cast<void *>(ggml_backend_npu_decode_w16a16_preload);
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

