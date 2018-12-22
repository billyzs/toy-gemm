cmake_minimum_required(VERSION 3.10)
project(toy_gemm)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    add_subdirectory(test)
endif()

add_library(toy_gemm INTERFACE)
target_sources(toy_gemm INTERFACE
       ${CMAKE_CURRENT_SOURCE_DIR}/include/toy-gemm/matrix.hpp)
target_include_directories(toy_gemm INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
target_compile_features(toy_gemm INTERFACE cxx_std_17)