# Copyright (c) 2025 AlertAvert.com.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Author: Marco Massenzio (marco@alertavert.com)
#
# This CMakeLists.txt enables building CUDA (.cu) source files as pure C++ files,
# without requiring CUDA tools or libraries. This is useful for development and
# testing of code that will eventually use CUDA features, but currently contains
# only standard C++.
#
# The source file to be compiled can be changed by modifying the SOURCE_FILE
# variable. The resulting executable will be named after the source file with
# a '-cpu' suffix and placed in the bin/ directory.

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define the source file name here
# Replace with the actual source file you want to compile
set(SOURCE_FILE "id-list-map.cu")
set(SOURCE_PATH "${CMAKE_SOURCE_DIR}/src/${SOURCE_FILE}")

# Force clang++ if available
find_program(CLANGXX clang++)
if(CLANGXX)
    set(CMAKE_CXX_COMPILER ${CLANGXX})
endif()

# Treat the .cu file as C++ source and disable CUDA processing
set_source_files_properties(${SOURCE_PATH}
    PROPERTIES
    LANGUAGE CXX
    COMPILE_FLAGS "-x c++ -nocudainc -nocudalib"
)

# Create executable name from source file (without extension)
string(REGEX REPLACE "\\.cu$" "" TARGET_NAME ${SOURCE_FILE})
add_executable(${TARGET_NAME}-cpu ${SOURCE_PATH})
set_target_properties(${TARGET_NAME}-cpu PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
