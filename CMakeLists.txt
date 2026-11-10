cmake_minimum_required(VERSION 3.21)
project(rafid VERSION 0.8.5 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# -----------------------------------------------------------------------------
# llama.cpp / ggml: the math engine (CPU kernels, CUDA kernels, GGUF loader).
# Pinned to a release tag so every build gets the same API.
# Pass -DGGML_CUDA=ON on the command line for NVIDIA GPUs.
# -----------------------------------------------------------------------------
set(RAFID_LLAMA_TAG "b12167" CACHE STRING "llama.cpp release tag")
set(RAFID_LLAMA_DIR ""       CACHE PATH   "Use a local llama.cpp checkout instead of downloading it")

set(LLAMA_BUILD_COMMON   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP      OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL        OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS    OFF CACHE BOOL "" FORCE)

if (RAFID_LLAMA_DIR)
    add_subdirectory(${RAFID_LLAMA_DIR} ${CMAKE_BINARY_DIR}/llama.cpp EXCLUDE_FROM_ALL)
    set(RAFID_LLAMA_SRC ${RAFID_LLAMA_DIR})
else()
    include(FetchContent)
    set(RAFID_FETCH_ARGS
        GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
        GIT_TAG        ${RAFID_LLAMA_TAG}
        GIT_SHALLOW    TRUE)
    if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.28)
        list(APPEND RAFID_FETCH_ARGS EXCLUDE_FROM_ALL) # build only what rafid links
    endif()
    FetchContent_Declare(llama_cpp ${RAFID_FETCH_ARGS})
    FetchContent_MakeAvailable(llama_cpp)
    set(RAFID_LLAMA_SRC ${llama_cpp_SOURCE_DIR})
endif()

# -----------------------------------------------------------------------------
# HTTP server library (vendored inside llama.cpp)
# -----------------------------------------------------------------------------
find_package(Threads REQUIRED)
add_library(rafid_httplib STATIC ${RAFID_LLAMA_SRC}/vendor/cpp-httplib/httplib.cpp)
target_include_directories(rafid_httplib PUBLIC ${RAFID_LLAMA_SRC}/vendor/cpp-httplib)
target_link_libraries(rafid_httplib PUBLIC Threads::Threads)
target_compile_features(rafid_httplib PUBLIC cxx_std_17)
target_compile_definitions(rafid_httplib PUBLIC CPPHTTPLIB_TCP_NODELAY=1)
if (WIN32)
    target_compile_definitions(rafid_httplib PUBLIC _WIN32_WINNT=0x0A00 NOMINMAX WIN32_LEAN_AND_MEAN)
    target_link_libraries(rafid_httplib PUBLIC ws2_32)
endif()
if (MSVC)
    target_compile_options(rafid_httplib PRIVATE /w /utf-8)
else()
    target_compile_options(rafid_httplib PRIVATE -w)
endif()

# -----------------------------------------------------------------------------
# Web UI embedded into the executable (re-generated when index.html changes)
# -----------------------------------------------------------------------------
set(RAFID_WEB_SRC ${CMAKE_CURRENT_SOURCE_DIR}/web/index.html)
set(RAFID_GEN_DIR ${CMAKE_BINARY_DIR}/generated)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${RAFID_WEB_SRC})
file(READ ${RAFID_WEB_SRC} RAFID_WEB_HEX HEX)
string(REGEX REPLACE "(................................................................................................................................)" "\\1\n" RAFID_WEB_HEX "${RAFID_WEB_HEX}")
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," RAFID_WEB_ARR "${RAFID_WEB_HEX}")
file(WRITE ${RAFID_GEN_DIR}/web_index.h.tmp
    "// generated from web/index.html - do not edit\n#pragma once\n#include <cstddef>\nstatic const unsigned char RAFID_INDEX_HTML[] = {\n${RAFID_WEB_ARR}\n};\nstatic const size_t RAFID_INDEX_HTML_LEN = sizeof(RAFID_INDEX_HTML);\n")
configure_file(${RAFID_GEN_DIR}/web_index.h.tmp ${RAFID_GEN_DIR}/web_index.h COPYONLY)

# -----------------------------------------------------------------------------
# Rafid
# -----------------------------------------------------------------------------
add_executable(rafid
    src/main.cpp
    src/util.cpp
    src/hw_probe.cpp
    src/gguf_info.cpp
    src/planner.cpp
    src/profiler.cpp
    src/analyzer.cpp
    src/expert_store.cpp
    src/slim.cpp
    src/engine.cpp
    src/server.cpp)

# ggml-backend-impl.h (the buffer interface used by the expert store) lives in ggml/src
target_include_directories(rafid PRIVATE src ${RAFID_GEN_DIR} ${RAFID_LLAMA_SRC}/ggml/src)
target_link_libraries(rafid PRIVATE llama ggml rafid_httplib vendor::nlohmann Threads::Threads)
target_compile_definitions(rafid PRIVATE RAFID_VERSION="${PROJECT_VERSION}" RAFID_LLAMA_TAG="${RAFID_LLAMA_TAG}")

if (MSVC)
    target_compile_options(rafid PRIVATE /utf-8 /W3 /permissive-)
    target_compile_definitions(rafid PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(rafid PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()
if (WIN32)
    target_link_libraries(rafid PRIVATE shell32)
endif()
