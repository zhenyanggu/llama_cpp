#pragma once

#include "ggml-npu-common.h"

#include "ggml-backend.h"

namespace ggml_npu {

enum ggml_status npu_compute_node(const npu_node_plan & plan, std::string * error);

} // namespace ggml_npu
