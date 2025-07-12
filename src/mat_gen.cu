// Copyright (c) 2025 Marco Massenzio, All rights reserved.

#include <iostream>

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>

using namespace std;

__global__ void fillMatrixKernel(float* mat, int m, int n, float mean, float stddev, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = m * n;
    
    if (idx < total_size) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        mat[idx] = curand_normal(&state) * stddev + mean;
    }
}

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
    int threadsPerBlock = 256;
    int numBlocks = (m * n + threadsPerBlock - 1) / threadsPerBlock;
    
    // Launch kernel
    fillMatrixKernel<<<numBlocks, threadsPerBlock>>>(
        d_mat, m, n, mean, stddev, time(nullptr));
    
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
    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    // Print some basic info
    cout << "Device name: " << prop.name << endl;
    
    // Check memory
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    cout << "Free memory: " << free_mem / (1024 * 1024) << " MB" << endl;
    cout << "Total memory: " << total_mem / (1024 * 1024) << " MB" << endl;

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
