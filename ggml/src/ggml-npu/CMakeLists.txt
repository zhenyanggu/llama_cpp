message(STATUS "Using NPU backend")

file(GLOB GGML_HEADERS_NPU "*.h")
file(GLOB GGML_SOURCES_NPU "*.cpp")

ggml_add_backend_library(ggml-npu ${GGML_HEADERS_NPU} ${GGML_SOURCES_NPU})
target_compile_definitions(ggml-npu PRIVATE GGML_NPU_VERSA_P_RUNTIME=1)

get_filename_component(LLAMA_CPP_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
set(NPU_RUNTIME_ROOT "${LLAMA_CPP_ROOT}/npuruntime/kv260/runtime")
set(GGML_NPU_VERSA_P_RUNTIME_ROOT "/mnt/c/vivado/VersaEdge-dsp-mac-per-lane/Versa_P/runtime" CACHE PATH "Versa_P Phase-1 runtime root")
set(GGML_NPU_VERSA_P_DRIVER_ROOT "/mnt/c/vivado/VersaEdge-dsp-mac-per-lane/sw/kv260/driver" CACHE PATH "Versa_P KV260 driver UAPI root")

target_include_directories(ggml-npu PRIVATE
    "${NPU_RUNTIME_ROOT}"
    "${GGML_NPU_VERSA_P_RUNTIME_ROOT}/include"
    "${GGML_NPU_VERSA_P_RUNTIME_ROOT}/src"
    "${GGML_NPU_VERSA_P_DRIVER_ROOT}"
    "${LLAMA_CPP_ROOT}/vendor"
    "${CMAKE_CURRENT_SOURCE_DIR}/..")

set(NPU_RUNTIME_SRC "${NPU_RUNTIME_ROOT}/npu_runtime_versa_p.cpp")
if (EXISTS "${NPU_RUNTIME_SRC}")
    target_sources(ggml-npu PRIVATE "${NPU_RUNTIME_SRC}")
else()
    message(FATAL_ERROR "NPU runtime source not found: ${NPU_RUNTIME_SRC}")
endif()

set(VERSA_P_RUNTIME_SOURCES
    "${GGML_NPU_VERSA_P_RUNTIME_ROOT}/src/device.cpp"
    "${GGML_NPU_VERSA_P_RUNTIME_ROOT}/src/mmio.cpp"
    "${GGML_NPU_VERSA_P_RUNTIME_ROOT}/src/commands.cpp")
foreach(VERSA_P_SRC IN LISTS VERSA_P_RUNTIME_SOURCES)
    if (EXISTS "${VERSA_P_SRC}")
        target_sources(ggml-npu PRIVATE "${VERSA_P_SRC}")
    else()
        message(FATAL_ERROR "Versa_P runtime source not found: ${VERSA_P_SRC}")
    endif()
endforeach()

if (GGML_NPU_RUNTIME_PROFILE)
    target_compile_definitions(ggml-npu PRIVATE NPU_PROFILE)
endif()
