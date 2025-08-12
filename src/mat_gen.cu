// Copyright (c) 2025.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0
// http://www.apache.org/licenses/LICENSE-2.0
//
// Author: Marco Massenzio (marco@alertavert.com)
#include <curand.h>
#include <curand_kernel.h>

#include <iostream>
#include <memory>

#include "boundaries.h"
#include "fillmatrix.h"

using namespace std;

__global__ void fillMatrixKernel(float* mat, void* strategy, float mean,
                                 float stddev, unsigned long seed) {
  // Here we lose the flexibility of polymorphic behavior,
  // as we need to know in advance the type of strategy.
  // However, in kernel functions, we must know in advance the
  // arrangement of data anyway.
  SizeCheck checker{strategy, BoundaryType::Rectangular};
  if (checker()) {
    auto idx = checker.idx();
    curandState state;
    curand_init(seed, idx, 0, &state);
    mat[idx] = curand_normal(&state) * stddev + mean;
  }
}

void fill(float* mat, uint m, uint n, float mean, float stddev) {
  float* d_mat;
  size_t size = m * n * sizeof(float);

  // Allocate device memory
  cudaError_t err = cudaMalloc(&d_mat, size);
  if (err != cudaSuccess) {
    cerr << "Failed to allocate device memory: " << cudaGetErrorString(err)
         << endl;
    return;
  }

  // Configure kernel launch parameters
  // Each block will handle 16x16 threads
  uint threadsPerBlockDim = 16;
  // The grid will be sized to cover the entire matrix, rows (m)
  // in the y dimension and columns (n) in the x dimension.
  dim3 gridDim{static_cast<uint>(ceil(n / threadsPerBlockDim) + 1),
               static_cast<uint>(ceil(m / threadsPerBlockDim) + 1)};
  dim3 blockDim{threadsPerBlockDim, threadsPerBlockDim};
  printf("Grid dimensions: %d x %d, Block dimensions: %d x %d\n", gridDim.x,
         gridDim.y, blockDim.x, blockDim.y);

  RectangularCheckStrategy strategy(m, n);
  RectangularCheckStrategy* d_strategy;
  err = cudaMalloc(&d_strategy, sizeof(RectangularCheckStrategy));
  if (err != cudaSuccess) {
    cerr << "Failed to allocate device memory for strategy: "
         << cudaGetErrorString(err) << endl;
    cudaFree(d_mat);
    return;
  }
  err = cudaMemcpy(d_strategy, &strategy, sizeof(RectangularCheckStrategy),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    cerr << "Failed to copy strategy to device: " << cudaGetErrorString(err)
         << endl;
    cudaFree(d_mat);
    cudaFree(d_strategy);
    return;
  }

  // Launch kernel
  fillMatrixKernel<<<gridDim, blockDim>>>(d_mat, d_strategy, mean, stddev,
                                          time(nullptr));

  // Copy result back to host
  err = cudaMemcpy(mat, d_mat, size, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    cerr << "Failed to copy data from device: " << cudaGetErrorString(err)
         << endl;
    cudaFree(d_mat);
    return;
  }

  // Cleanup
  cudaFree(d_mat);
}
