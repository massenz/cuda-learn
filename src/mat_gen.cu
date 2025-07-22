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
void fillMatrixKernel(float* mat, BoundaryCheckStrategy* strategy, float mean, float stddev, unsigned long seed) {

    SizeCheck checker { strategy };
    auto idx = checker.idx();
    printf("Filling index: %d\n", idx);
    if (checker()) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        mat[idx] = curand_normal(&state) * stddev + mean;
    }
}


using namespace std;

void fill(float* mat, int m, int n, float mean = 0.0f, float stddev = 1.0f) {
    float* d_mat;
    size_t size = m * n * sizeof(float);

    // Allocate device memory
    cudaError_t err = cudaMalloc(&d_mat, size);
    if (err != cudaSuccess) {
        cerr << "Failed to allocate device memory: " << cudaGetErrorString(err) << endl;
        return;
    }

    // Configure kernel launch parameters
    float threadsPerBlock = 16.0f;
    dim3 gridDim { ceil(n / threadsPerBlock), ceil(m / threadsPerBlock), 1 };
    dim3 blockDim { n, m, 1 };

    // TODO: Use a dim3 to initialize the strategy
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


int main() {
    // Example usage of fill function
    const int M = 3;
    const int N = 4;
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
