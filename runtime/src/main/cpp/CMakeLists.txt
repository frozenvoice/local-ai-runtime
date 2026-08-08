cmake_minimum_required(VERSION 3.31.6)

project(local-ai-runtime VERSION 0.1.0 LANGUAGES C CXX)

if(NOT ANDROID)
    message(FATAL_ERROR "local-ai-runtime is an Android-only native library")
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(LLAMA_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../../third_party/llama.cpp")
set(KLEIDIAI_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../../third_party/kleidiai")

# Never download native dependencies as part of an Android app build. The
# exact source release is checked into this independent runtime repository.
set(FETCHCONTENT_SOURCE_DIR_KLEIDIAI_DOWNLOAD
    "${KLEIDIAI_SOURCE_DIR}"
    CACHE PATH "Pinned KleidiAI v1.24.0 source" FORCE)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE BOOL "" FORCE)
set(FETCHCONTENT_FULLY_DISCONNECTED ON CACHE BOOL "" FORCE)

if(ANDROID_ABI STREQUAL "arm64-v8a")
    set(GGML_SYSTEM_ARCH "ARM")
    set(GGML_CPU_KLEIDIAI ON CACHE BOOL "" FORCE)
    set(GGML_OPENMP ON CACHE BOOL "" FORCE)
elseif(ANDROID_ABI STREQUAL "x86_64")
    set(GGML_SYSTEM_ARCH "x86")
    set(GGML_CPU_KLEIDIAI OFF CACHE BOOL "" FORCE)
    set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
else()
    message(FATAL_ERROR "Unsupported ABI: ${ANDROID_ABI}")
endif()

add_subdirectory("${LLAMA_SOURCE_DIR}" build-llama)

add_library(local-ai-runtime SHARED local_ai_runtime.cpp)
target_compile_definitions(local-ai-runtime PRIVATE
    GGML_SYSTEM_ARCH=${GGML_SYSTEM_ARCH}
    GGML_CPU_KLEIDIAI=$<BOOL:${GGML_CPU_KLEIDIAI}>
    GGML_OPENMP=$<BOOL:${GGML_OPENMP}>
)
target_include_directories(local-ai-runtime PRIVATE
    "${LLAMA_SOURCE_DIR}"
    "${LLAMA_SOURCE_DIR}/common"
    "${LLAMA_SOURCE_DIR}/include"
    "${LLAMA_SOURCE_DIR}/ggml/include"
    "${LLAMA_SOURCE_DIR}/ggml/src"
)
target_link_libraries(local-ai-runtime PRIVATE
    llama
    llama-common
    android
    log
)
