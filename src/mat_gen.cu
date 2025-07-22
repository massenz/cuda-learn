// Copyright (c) 2025.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0
// http://www.apache.org/licenses/LICENSE-2.0
//
// Author: Marco Massenzio (marco@alertavert.com)

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <memory>

#include "boundaries.h"

__global__ 
void fillMatrixKernel(float* mat, void* strategy, float mean, float stddev, unsigned long seed) {

    // Here we lose the flexibility of polymorphic behavior,
    // as we need to know in advance the type of strategy.
    // However, in kernel functions, we must know in advance the
    // arrangement of data.
    SizeCheck checker { strategy, BoundaryType::Rectangular };
    if (checker()) {
        auto idx = checker.idx();
        curandState state;
        curand_init(seed, idx, 0, &state);
        mat[idx] = curand_normal(&state) * stddev + mean;
    }
}


using namespace std;

void fill(float* mat, uint m, uint n, float mean = 0.0f, float stddev = 1.0f) {
    float* d_mat;
    size_t size = m * n * sizeof(float);

    // Allocate device memory
    cudaError_t err = cudaMalloc(&d_mat, size);
    if (err != cudaSuccess) {
        cerr << "Failed to allocate device memory: " << cudaGetErrorString(err) << endl;
        return;
    }

    // Configure kernel launch parameters
    // Each block will handle 16x16 threads
    uint threadsPerBlockDim = 16;
    // The grid will be sized to cover the entire matrix, rows (m)
    // in the y dimension and columns (n) in the x dimension.
    dim3 gridDim { 
        static_cast<uint>(ceil(n / threadsPerBlockDim) + 1), 
        static_cast<uint>(ceil(m / threadsPerBlockDim) + 1)};
    dim3 blockDim { threadsPerBlockDim, threadsPerBlockDim };
    printf("Grid dimensions: %d x %d, Block dimensions: %d x %d\n", 
           gridDim.x, gridDim.y, blockDim.x, blockDim.y);

    RectangularCheckStrategy strategy(m, n);
    RectangularCheckStrategy* d_strategy;
    err = cudaMalloc(&d_strategy, sizeof(RectangularCheckStrategy));
    if (err != cudaSuccess) {
        cerr << "Failed to allocate device memory for strategy: " << cudaGetErrorString(err) << endl;
        cudaFree(d_mat);
        return;
    }
    err = cudaMemcpy(d_strategy, &strategy, sizeof(RectangularCheckStrategy), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cerr << "Failed to copy strategy to device: " << cudaGetErrorString(err) << endl;
        cudaFree(d_mat);
        cudaFree(d_strategy);
        return;
    }

    // Launch kernel
    fillMatrixKernel<<<gridDim, blockDim>>>(
        d_mat, 
        d_strategy,
        mean, 
        stddev, 
        time(nullptr));

    // Copy result back to host
    err = cudaMemcpy(mat, d_mat, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cerr << "Failed to copy data from device: " << cudaGetErrorString(err) << endl;
        cudaFree(d_mat);
        return;
    }

    // Cleanup
    cudaFree(d_mat);
}


int main(int argc, char* argv[]) {
    // Default values for matrix dimensions
    int M = 10;
    int N = 15;

    // Parse command line arguments if provided
    if (argc > 1) M = atoi(argv[1]);
    if (argc > 2) N = atoi(argv[2]);

    printf("Creating matrix of size %d x %d\n", M, N);
    float* matrix = new float[M * N];

    fill(matrix, M, N);  // Using default mean=0.0 and stddev=1.0

    // Print the matrix
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%8.4f", matrix[i * N + j]);
        }
        printf("\n");
    }

    delete[] matrix;
    return 0;
}
