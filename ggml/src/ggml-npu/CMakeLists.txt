message(STATUS "Using NPU backend")

file(GLOB GGML_HEADERS_NPU "*.h")
file(GLOB GGML_SOURCES_NPU "*.cpp")

ggml_add_backend_library(ggml-npu ${GGML_HEADERS_NPU} ${GGML_SOURCES_NPU})

get_filename_component(LLAMA_CPP_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
target_include_directories(ggml-npu PRIVATE
    "${LLAMA_CPP_ROOT}/npuruntime"
    "${CMAKE_CURRENT_SOURCE_DIR}/..")

set(NPU_RUNTIME_SRC "${LLAMA_CPP_ROOT}/npuruntime/npu_runtime.cpp")
if (EXISTS "${NPU_RUNTIME_SRC}")
    target_sources(ggml-npu PRIVATE "${NPU_RUNTIME_SRC}")
else()
    message(FATAL_ERROR "NPU runtime source not found: ${NPU_RUNTIME_SRC}")
endif()
