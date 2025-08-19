# Copyright (c) 2025 AlertAvert.com.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Author: Marco Massenzio (marco@alertavert.com)

cmake_minimum_required(VERSION 3.18)

# Option to build CPU-only version
option(CPU_ONLY "Build CPU-only version" OFF)

include_directories(${CMAKE_SOURCE_DIR}/src/includes)
if(CPU_ONLY)
    project(cuda-learn LANGUAGES CXX)
    include(${CMAKE_SOURCE_DIR}/cmake/CpuBuild.cmake)
else()
    message(STATUS "Building with CUDA support")
    project(cuda-learn LANGUAGES CXX CUDA)
    include(${CMAKE_SOURCE_DIR}/cmake/Cuda.cmake)
endif()
