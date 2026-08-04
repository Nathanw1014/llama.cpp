# llama-bench-impl: benchmark logic, reusable by app

set(TARGET llama-bench-impl)

add_library(${TARGET} llama-bench.cpp)
set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC llama-common llama ${CMAKE_THREAD_LIBS_INIT})

option(LLAMA_BENCH_ROCTX "Add ROCTX ranges to llama-bench" OFF)
if(LLAMA_BENCH_ROCTX)
    find_path(ROCTX_INCLUDE_DIR rocprofiler-sdk-roctx/roctx.h HINTS ENV ROCM_PATH PATHS /opt/rocm PATH_SUFFIXES include REQUIRED)
    find_library(ROCTX_LIBRARY NAMES rocprofiler-sdk-roctx HINTS ENV ROCM_PATH PATHS /opt/rocm PATH_SUFFIXES lib lib64 REQUIRED)
    target_include_directories(${TARGET} PRIVATE ${ROCTX_INCLUDE_DIR})
    target_link_libraries(${TARGET} PRIVATE ${ROCTX_LIBRARY})
    target_compile_definitions(${TARGET} PRIVATE LLAMA_BENCH_ROCTX)
endif()

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} LIBRARY)
endif()

# llama-bench executable

set(TARGET llama-bench)

add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE llama-bench-impl)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
