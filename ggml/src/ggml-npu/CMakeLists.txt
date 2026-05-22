message(STATUS "Using NPU backend")

file(GLOB GGML_HEADERS_NPU "*.h")
file(GLOB GGML_SOURCES_NPU "*.cpp")

ggml_add_backend_library(ggml-npu ${GGML_HEADERS_NPU} ${GGML_SOURCES_NPU})
target_compile_definitions(ggml-npu PRIVATE GGML_NPU_VERSA_P_RUNTIME=1)

get_filename_component(LLAMA_CPP_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
set(NPU_RUNTIME_ROOT "${LLAMA_CPP_ROOT}/npuruntime/kv260/runtime")
set(GGML_NPU_VERSA_P_RUNTIME_ROOT "/mnt/c/vivado/VersaEdge-dsp-mac-per-lane/Versa_P/runtime" CACHE PATH "Versa_P Phase-1 runtime root")
set(GGML_NPU_VERSA_P_DRIVER_ROOT "/mnt/c/vivado/VersaEdge-dsp-mac-per-lane/sw/kv260/driver" CACHE PATH "Versa_P KV260 driver UAPI root")
set(GGML_NPU_DECODE_RUNTIME_ROOT "/mnt/c/vivado/versa_decode/sw/kv260/runtime" CACHE PATH "Versa decode GEMV runtime root")
set(GGML_NPU_DECODE_DRIVER_ROOT "/mnt/c/vivado/versa_decode/sw/kv260/driver" CACHE PATH "Versa decode KV260 driver UAPI root")

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

set(NPU_DECODE_RUNTIME_SRC "${GGML_NPU_DECODE_RUNTIME_ROOT}/src/npu_runtime.cpp")
if (EXISTS "${NPU_DECODE_RUNTIME_SRC}")
    set_source_files_properties("${NPU_DECODE_RUNTIME_SRC}" PROPERTIES
        INCLUDE_DIRECTORIES "${GGML_NPU_DECODE_RUNTIME_ROOT}/include;${GGML_NPU_DECODE_RUNTIME_ROOT}/src;${GGML_NPU_DECODE_DRIVER_ROOT}"
        COMPILE_OPTIONS "-UNPU_PROFILE"
        COMPILE_DEFINITIONS "npu_init=npu_decode_init;npu_destroy=npu_decode_destroy;npu_reset=npu_decode_reset;npu_profile_begin=npu_decode_profile_begin;npu_profile_end=npu_decode_profile_end;npu_profile_dump=npu_decode_profile_dump;npu_profile_reset_summary=npu_decode_profile_reset_summary;npu_profile_get_summary=npu_decode_profile_get_summary;npu_mem_alloc=npu_decode_mem_alloc;npu_mem_free=npu_decode_mem_free;npu_dma_mvin=npu_decode_dma_mvin;npu_dma_mvout=npu_decode_dma_mvout;npu_dma_mvout_ex=npu_decode_dma_mvout_ex;npu_dma_mvin_async=npu_decode_dma_mvin_async;npu_dma_mvout_async=npu_decode_dma_mvout_async;npu_dma_wait_mvin=npu_decode_dma_wait_mvin;npu_dma_wait_mvout=npu_decode_dma_wait_mvout;npu_dma_double_mvin=npu_decode_dma_double_mvin;npu_sfu_run=npu_decode_sfu_run;npu_conv_run=npu_decode_conv_run;npu_conv_tile_run=npu_decode_conv_tile_run;npu_gemm_run=npu_decode_gemm_run;npu_gemm_plan_run=npu_decode_gemm_plan_run;npu_matvec_run=npu_decode_matvec_run;npu_gemv_block_run=npu_decode_gemv_block_run;npu_gemv_pingpong_run=npu_decode_gemv_pingpong_run;npu_matadd_run=npu_decode_matadd_run;npu_transpose_run=npu_decode_transpose_run;npu_resample_run=npu_decode_resample_run;npu_layout_nchw_to_nchwc32=npu_decode_layout_nchw_to_nchwc32;npu_layout_nchwc32_to_nchw=npu_decode_layout_nchwc32_to_nchw;npu_dma_mvin_test=npu_decode_dma_mvin_test;npu_dma_mvout_test=npu_decode_dma_mvout_test")
    target_sources(ggml-npu PRIVATE "${NPU_DECODE_RUNTIME_SRC}")
else()
    message(FATAL_ERROR "Versa decode runtime source not found: ${NPU_DECODE_RUNTIME_SRC}")
endif()

if (GGML_NPU_RUNTIME_PROFILE)
    target_compile_definitions(ggml-npu PRIVATE NPU_PROFILE)
endif()
