cmake_minimum_required(VERSION 3.20)
project(bud_flow_lang
    VERSION 0.1.0
    DESCRIPTION "Python DSL for SIMD Programming with JIT Compilation"
    LANGUAGES C CXX
)

# ============================================================================
# Build Options
# ============================================================================
option(BUD_BUILD_TESTS "Build unit tests" ON)
option(BUD_BUILD_BENCHMARKS "Build benchmarks" ON)
option(BUD_BUILD_FUZZERS "Build fuzz targets" OFF)
option(BUD_BUILD_PYTHON "Build Python bindings" ON)
option(BUD_BUILD_DOCS "Build documentation" OFF)
option(BUD_ENABLE_SANITIZERS "Enable sanitizers in debug builds" ON)
option(BUD_ENABLE_COVERAGE "Enable code coverage" OFF)
option(BUD_USE_SIMD_BENCH "Use simd-bench for profiling" ON)
option(BUD_ENABLE_LTO "Enable Link Time Optimization" OFF)

# JIT Backend Options
option(BUD_JIT_ENABLE_MIR "Enable MIR JIT backend" ON)
option(BUD_JIT_ENABLE_LLVM "Enable LLVM JIT backend (Tier 2)" OFF)

# ============================================================================
# C++ Standard and Compiler Configuration
# ============================================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Use ccache if available
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    message(STATUS "Using ccache: ${CCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
endif()

# ============================================================================
# Compiler Flags
# ============================================================================
# Common warning flags (applied globally)
set(BUD_COMMON_WARNINGS
    -Wall -Wextra -Wpedantic
    -Werror=return-type
    -Werror=uninitialized
    -Wno-unused-parameter
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(${BUD_COMMON_WARNINGS})
    # Note: -fno-rtti is applied per-target for core library only
    # nanobind and Python bindings require RTTI

    # Release flags
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -march=native -DNDEBUG")

    # Debug flags
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g3 -fno-omit-frame-pointer")

    # SIMD optimization reporting (useful for development)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        add_compile_options(-fopt-info-vec-missed)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        add_compile_options(-Rpass=loop-vectorize -Rpass-missed=loop-vectorize)
    endif()
endif()

# ============================================================================
# Sanitizers (Debug/Development)
# ============================================================================
if(BUD_ENABLE_SANITIZERS AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Enabling sanitizers for debug build")
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
endif()

# ============================================================================
# Code Coverage
# ============================================================================
if(BUD_ENABLE_COVERAGE)
    message(STATUS "Enabling code coverage")
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        add_compile_options(-fprofile-instr-generate -fcoverage-mapping)
        add_link_options(-fprofile-instr-generate -fcoverage-mapping)
    else()
        add_compile_options(--coverage)
        add_link_options(--coverage)
    endif()
endif()

# ============================================================================
# LTO
# ============================================================================
if(BUD_ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT LTO_SUPPORTED OUTPUT LTO_ERROR)
    if(LTO_SUPPORTED)
        message(STATUS "LTO enabled")
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        message(WARNING "LTO not supported: ${LTO_ERROR}")
    endif()
endif()

# ============================================================================
# CPM Package Manager Setup
# ============================================================================
set(CPM_DOWNLOAD_VERSION 0.40.2)
set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")

if(NOT EXISTS ${CPM_DOWNLOAD_LOCATION})
    message(STATUS "Downloading CPM.cmake v${CPM_DOWNLOAD_VERSION}")
    file(DOWNLOAD
        https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
        ${CPM_DOWNLOAD_LOCATION}
    )
endif()
include(${CPM_DOWNLOAD_LOCATION})

# ============================================================================
# Dependencies
# ============================================================================

# Highway (SIMD library) - Core dependency
CPMAddPackage(
    NAME highway
    GITHUB_REPOSITORY google/highway
    GIT_TAG master
    OPTIONS
        "HWY_ENABLE_TESTS OFF"
        "HWY_ENABLE_EXAMPLES OFF"
        "HWY_ENABLE_CONTRIB ON"
)

# nlohmann/json - Config and IR serialization
CPMAddPackage(
    NAME nlohmann_json
    VERSION 3.11.3
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)

# fmt - Fast string formatting
CPMAddPackage(
    NAME fmt
    GITHUB_REPOSITORY fmtlib/fmt
    GIT_TAG 10.2.1
)

# spdlog - Logging
CPMAddPackage(
    NAME spdlog
    GITHUB_REPOSITORY gabime/spdlog
    VERSION 1.13.0
    OPTIONS
        "SPDLOG_FMT_EXTERNAL ON"
)

# ============================================================================
# Testing Dependencies
# ============================================================================
if(BUD_BUILD_TESTS)
    CPMAddPackage(
        NAME googletest
        GITHUB_REPOSITORY google/googletest
        VERSION 1.14.0
        OPTIONS
            "gtest_force_shared_crt ON"
            "INSTALL_GTEST OFF"
    )
    enable_testing()
    include(GoogleTest)
endif()

# ============================================================================
# Benchmark Dependencies
# ============================================================================
if(BUD_BUILD_BENCHMARKS)
    CPMAddPackage(
        NAME nanobench
        GITHUB_REPOSITORY martinus/nanobench
        VERSION 4.3.11
        OPTIONS
            "NANOBENCH_ENABLE_TESTS OFF"
    )
endif()

# ============================================================================
# Python Bindings (nanobind)
# ============================================================================
if(BUD_BUILD_PYTHON)
    find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)

    CPMAddPackage(
        NAME nanobind
        GITHUB_REPOSITORY wjakob/nanobind
        GIT_TAG v2.2.0
        OPTIONS
            "NB_TEST OFF"
    )
endif()

# ============================================================================
# simd-bench Integration
# ============================================================================
if(BUD_USE_SIMD_BENCH)
    set(SIMD_BENCH_PATH "${CMAKE_SOURCE_DIR}/../simd-bench")
    if(EXISTS "${SIMD_BENCH_PATH}/CMakeLists.txt")
        message(STATUS "Found simd-bench at: ${SIMD_BENCH_PATH}")
        add_subdirectory(${SIMD_BENCH_PATH} ${CMAKE_BINARY_DIR}/simd-bench EXCLUDE_FROM_ALL)
        set(SIMD_BENCH_FOUND TRUE)
    else()
        message(WARNING "simd-bench not found at ${SIMD_BENCH_PATH} - profiling features disabled")
        set(SIMD_BENCH_FOUND FALSE)
    endif()
endif()

# ============================================================================
# MIR JIT Backend (Optional)
# ============================================================================
if(BUD_JIT_ENABLE_MIR)
    CPMAddPackage(
        NAME mir
        GITHUB_REPOSITORY vnmakarov/mir
        GIT_TAG master
        DOWNLOAD_ONLY YES
    )

    if(mir_ADDED)
        message(STATUS "MIR JIT backend enabled")
        # MIR is a C library, we'll build it manually
        set(MIR_SOURCES
            ${mir_SOURCE_DIR}/mir.c
            ${mir_SOURCE_DIR}/mir-gen.c
        )
        add_library(mir_jit STATIC ${MIR_SOURCES})
        target_include_directories(mir_jit PUBLIC ${mir_SOURCE_DIR})
        target_compile_definitions(mir_jit PRIVATE MIR_NO_IO=1)
        # Disable warnings for third-party MIR library (C code with intentional patterns)
        target_compile_options(mir_jit PRIVATE
            -Wno-error=uninitialized
            -Wno-error=maybe-uninitialized
            -Wno-uninitialized
            -Wno-maybe-uninitialized
            -Wno-unused-but-set-variable
            -Wno-unused-variable
        )
        set(MIR_FOUND TRUE)
    endif()
endif()

# ============================================================================
# Include Directories
# ============================================================================
include_directories(${CMAKE_SOURCE_DIR}/include)

# ============================================================================
# Core Library
# ============================================================================
set(BUD_CORE_SOURCES
    src/core/error.cc
    src/core/arena.cc
    src/core/type_system.cc
    src/core/hardware_detect.cc
    src/ir/ir_node.cc
    src/ir/ir_builder.cc
    src/ir/ir_interpreter.cc
    src/ir/optimizer.cc
    src/ir/lazy_evaluator.cc
    src/jit/stencil_registry.cc
    src/jit/copy_patch_compiler.cc
    src/jit/code_cache.cc
    src/runtime/bunch.cc
    src/runtime/executor.cc
    src/runtime/memory_pool.cc
    src/codegen/hwy_ops.cc
    src/codegen/fused_kernel.cc
    # Memory optimization
    src/memory/cache_config.cc
    src/memory/prefetch.cc
    src/memory/numa_allocator.cc
    src/memory/tiled_executor.cc
    # Developer tier abstractions
    src/pattern/pattern.cc
    src/pipeline/pipeline.cc
    # Expert tier Highway types
    src/highway/highway_types.cc
)

add_library(bud_flow_core STATIC ${BUD_CORE_SOURCES})
target_include_directories(bud_flow_core PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/codegen  # For hwy_ops-inl.h
    ${CMAKE_SOURCE_DIR}/src/highway  # For highway_types-inl.h
    ${CMAKE_BINARY_DIR}/_deps/highway-src  # Highway includes
)
target_link_libraries(bud_flow_core PUBLIC
    hwy
    hwy_contrib
    fmt::fmt
    spdlog::spdlog
    nlohmann_json::nlohmann_json
)

# LLVM-style: disable RTTI for core library (not for Python bindings)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(bud_flow_core PRIVATE -fno-rtti)
endif()

# Conditional linking
if(SIMD_BENCH_FOUND)
    target_compile_definitions(bud_flow_core PUBLIC BUD_HAS_SIMD_BENCH)
    target_link_libraries(bud_flow_core PUBLIC simd_bench_core)
endif()

if(MIR_FOUND)
    target_compile_definitions(bud_flow_core PUBLIC BUD_HAS_MIR_JIT)
    target_link_libraries(bud_flow_core PUBLIC mir_jit)
endif()

# ============================================================================
# Python Module
# ============================================================================
if(BUD_BUILD_PYTHON)
    # Enable RTTI for nanobind (required for typeid)
    # Note: We re-enable RTTI specifically for Python bindings since nanobind needs it
    nanobind_add_module(bud_flow_lang_py
        NB_STATIC  # Static linking for standalone wheel
        src/python/module.cc
        src/python/bunch_bindings.cc
        src/python/flow_bindings.cc
        src/python/hint_bindings.cc
        src/python/tracing_context.cc
        # Developer & Expert Tier bindings
        src/python/highway_bindings.cc
        src/python/pattern_bindings.cc
        src/python/pipeline_bindings.cc
        src/python/memory_bindings.cc
    )
    target_link_libraries(bud_flow_lang_py PRIVATE bud_flow_core)
    target_include_directories(bud_flow_lang_py PRIVATE ${CMAKE_SOURCE_DIR}/src/python)

    # Override -fno-rtti for the Python module (nanobind requires RTTI)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(bud_flow_lang_py PRIVATE -frtti)
    endif()

    # Generate stubs for IDE support
    nanobind_add_stub(
        bud_flow_lang_py_stub
        MODULE bud_flow_lang_py
        OUTPUT bud_flow_lang.pyi
        PYTHON_PATH $<TARGET_FILE_DIR:bud_flow_lang_py>
        DEPENDS bud_flow_lang_py
    )
endif()

# ============================================================================
# Tests
# ============================================================================
if(BUD_BUILD_TESTS)
    add_executable(bud_tests
        tests/test_main.cc
        tests/core/arena_test.cc
        tests/core/type_system_test.cc
        tests/ir/ir_builder_test.cc
        tests/ir/optimizer_test.cc
        tests/jit/copy_patch_test.cc
        tests/runtime/bunch_test.cc
        tests/integration/e2e_test.cc
        tests/integration/test_simple.cc
        tests/codegen/hwy_ops_test.cc
        tests/codegen/optimized_reductions_test.cc
        tests/codegen/size_specialized_kernels_test.cc
        tests/codegen/kernel_fusion_test.cc
        tests/codegen/prefetch_test.cc
        tests/codegen/dynamic_threshold_test.cc
        tests/unit/test_memory_optimization.cc
        tests/stress/test_stress.cc
        tests/limits/test_limits.cc
    )
    target_link_libraries(bud_tests PRIVATE
        bud_flow_core
        GTest::gtest
        GTest::gtest_main
        GTest::gmock
    )
    # Match core library RTTI setting
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(bud_tests PRIVATE -fno-rtti)
    endif()
    gtest_discover_tests(bud_tests)

    # Add test target
    add_custom_target(check
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
        DEPENDS bud_tests
    )
endif()

# ============================================================================
# Benchmarks
# ============================================================================
if(BUD_BUILD_BENCHMARKS)
    add_executable(bud_benchmarks
        benchmarks/bench_main.cc
        benchmarks/arena_bench.cc
        benchmarks/jit_compile_bench.cc
        benchmarks/simd_ops_bench.cc
    )
    target_link_libraries(bud_benchmarks PRIVATE
        bud_flow_core
        nanobench
    )
    # Match core library RTTI setting
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(bud_benchmarks PRIVATE -fno-rtti)
    endif()
endif()

# ============================================================================
# Fuzz Targets
# ============================================================================
if(BUD_BUILD_FUZZERS)
    add_executable(fuzz_ir_parser fuzz/fuzz_ir_parser.cc)
    target_compile_options(fuzz_ir_parser PRIVATE -fsanitize=fuzzer,address -fno-rtti)
    target_link_options(fuzz_ir_parser PRIVATE -fsanitize=fuzzer,address)
    target_link_libraries(fuzz_ir_parser PRIVATE bud_flow_core)

    add_executable(fuzz_type_inference fuzz/fuzz_type_inference.cc)
    target_compile_options(fuzz_type_inference PRIVATE -fsanitize=fuzzer,address -fno-rtti)
    target_link_options(fuzz_type_inference PRIVATE -fsanitize=fuzzer,address)
    target_link_libraries(fuzz_type_inference PRIVATE bud_flow_core)
endif()

# ============================================================================
# Examples
# ============================================================================
add_executable(example_basic examples/basic_usage.cc)
target_link_libraries(example_basic PRIVATE bud_flow_core)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(example_basic PRIVATE -fno-rtti)
endif()

add_executable(example_vectorize examples/auto_vectorize.cc)
target_link_libraries(example_vectorize PRIVATE bud_flow_core)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(example_vectorize PRIVATE -fno-rtti)
endif()

# ============================================================================
# Code Quality Targets
# ============================================================================

# Format target
find_program(CLANG_FORMAT clang-format)
if(CLANG_FORMAT)
    file(GLOB_RECURSE ALL_SOURCE_FILES
        ${CMAKE_SOURCE_DIR}/src/*.cc
        ${CMAKE_SOURCE_DIR}/src/*.h
        ${CMAKE_SOURCE_DIR}/include/*.h
        ${CMAKE_SOURCE_DIR}/tests/*.cc
    )
    add_custom_target(format
        COMMAND ${CLANG_FORMAT} -i ${ALL_SOURCE_FILES}
        COMMENT "Running clang-format"
    )
endif()

# Tidy target
find_program(CLANG_TIDY clang-tidy)
if(CLANG_TIDY)
    add_custom_target(tidy
        COMMAND ${CLANG_TIDY} -p ${CMAKE_BINARY_DIR}
            ${CMAKE_SOURCE_DIR}/src/**/*.cc
        COMMENT "Running clang-tidy"
    )
endif()

# Coverage target
if(BUD_ENABLE_COVERAGE)
    add_custom_target(coverage
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
        COMMAND llvm-profdata merge -sparse default.profraw -o coverage.profdata || true
        COMMAND llvm-cov report ./bud_tests -instr-profile=coverage.profdata || gcov -r ${CMAKE_SOURCE_DIR}/src/*.cc
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        DEPENDS bud_tests
    )
endif()

# ============================================================================
# Installation
# ============================================================================
include(GNUInstallDirs)

install(TARGETS bud_flow_core
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY include/bud_flow_lang
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

if(BUD_BUILD_PYTHON)
    install(TARGETS bud_flow_lang_py
        LIBRARY DESTINATION ${Python_SITEARCH}
    )
endif()

# ============================================================================
# Configuration Summary
# ============================================================================
message(STATUS "")
message(STATUS "=== Bud Flow Lang Configuration ===")
message(STATUS "Version:          ${PROJECT_VERSION}")
message(STATUS "Build Type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler:         ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C++ Standard:     ${CMAKE_CXX_STANDARD}")
message(STATUS "")
message(STATUS "Features:")
message(STATUS "  Tests:          ${BUD_BUILD_TESTS}")
message(STATUS "  Benchmarks:     ${BUD_BUILD_BENCHMARKS}")
message(STATUS "  Fuzzers:        ${BUD_BUILD_FUZZERS}")
message(STATUS "  Python:         ${BUD_BUILD_PYTHON}")
message(STATUS "  simd-bench:     ${SIMD_BENCH_FOUND}")
message(STATUS "  MIR JIT:        ${MIR_FOUND}")
message(STATUS "  Sanitizers:     ${BUD_ENABLE_SANITIZERS}")
message(STATUS "  Coverage:       ${BUD_ENABLE_COVERAGE}")
message(STATUS "  LTO:            ${BUD_ENABLE_LTO}")
message(STATUS "  ccache:         ${CCACHE_PROGRAM}")
message(STATUS "")
