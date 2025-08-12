# Copyright (c) 2025 AlertAvert.com.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Author: Marco Massenzio (marco@alertavert.com)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(TARGET_DIR "${CMAKE_BINARY_DIR}/bin")

# Find CUDA package
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Set CUDA flags (similar to the ones in the Makefile)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2 -diag-suppress 2361")

# Create build directory if it doesn't exist
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/build)

# Add executable targets
add_executable(gpu-props src/gpu-props.cu)
set_target_properties(gpu-props PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY ${TARGET_DIR}
)

add_executable(demo 
        src/demo.cpp
        src/mat_gen.cu
)
set_target_properties(demo PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY ${TARGET_DIR}
)


# Link against CUDA libraries
target_link_libraries(gpu-props ${CUDA_LIBRARIES})
target_link_libraries(demo ${CUDA_LIBRARIES} ${CUDA_curand_LIBRARY})

# Print information about the build
message(STATUS "CUDA version: ${CUDA_VERSION}")
message(STATUS "CUDA libraries: ${CUDA_LIBRARIES}")
message(STATUS "CUDA include directories: ${CUDA_INCLUDE_DIRS}")
message(STATUS "Binaries in: ${TARGET_DIR}")
