// Copyright (c) 2025.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0
// http://www.apache.org/licenses/LICENSE-2.0
//
// Author: Marco Massenzio (marco@alertavert.com)

#include <iostream>

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <memory>

using namespace std;

class BoundaryCheckStrategy {
public:
    __device__
    virtual bool check() const = 0;

    __device__
    virtual dim3 pos() const = 0;

    __device__
    virtual dim3 dims() const = 0;
};

/**
 * A linear check strategy that checks if the current index is within the bounds of a given size
 * along a one-dimensional array.
 */
class LinearCheckStragegy: public BoundaryCheckStrategy {
 public:
    __host__
    explicit LinearCheckStragegy(int len): size_{len} {}
    
    __device__
    bool check() const override {
        return (pos().x < size_);
    }

    __device__
    dim3 pos() const override {
        return dim3{ blockIdx.x * blockDim.x + threadIdx.x, 0, 0 };
    }

    __device__
    dim3 dims() const override{
        return dim3{ size_, 0, 0 };
    }

 private:
    int size_;
};

class RectangularCheckStrategy: public BoundaryCheckStrategy {
 public:
    __host__
    RectangularCheckStrategy(int m, int n): rows_{m}, cols_{n} {}

    __device__
    bool check() const override {
        auto p = pos();
        return (p.y < rows_ && p.x < cols_);
    }

    __device__
    dim3 pos() const override {
        return dim3 {(blockIdx.x * blockDim.x + threadIdx.x),
               (blockIdx.y * blockDim.y + threadIdx.y),
               0 };
    }

    __device__
    dim3 dims() const override {
        return dim3 { cols_, rows_, 0 };
    }

 private:
    int rows_;
    int cols_;
};

class CubeCheckStrategy: public BoundaryCheckStrategy {
 public:
    __host__
    CubeCheckStrategy(int rows, int cols, int depth): size_{ cols, rows, depth } {}

    __device__
    bool check() const override {
        auto curPos = pos();
        return (curPos.x < size_.x && curPos.y < size_.y && curPos.z < size_.z);
    }

    __device__
    dim3 pos() const override {
        return {
            (blockIdx.x * blockDim.x + threadIdx.x),
            (blockIdx.y * blockDim.y + threadIdx.y),
            (blockIdx.z * blockDim.z + threadIdx.z) };
    }

    __device__
    dim3 dims() const override {
        return size_;
    }

 private:
    dim3 size_;
};


class SizeCheck {
 public:
    __device__
    explicit SizeCheck(const BoundaryCheckStrategy* strategy): 
        strategy_{ strategy } {}

    __device__
    bool operator()() const {
        return strategy_->check();
    }

    __device__
    int idx() const {
        auto pos = strategy_->pos();
        auto dims = strategy_->dims();
        printf("pos: %d, %d, %d\n", pos.x , pos.y , pos.z);
        auto matSize = dims.x * dims.y;
        int idx = pos.z * matSize + 
                  pos.y * dims.y + 
                  pos.x;

        return idx;
    }

 private:
    const BoundaryCheckStrategy* strategy_;
};

struct something {
    int a;
    int b;
};

__global__ 
void fillMatrixKernel(float* mat, something* s, BoundaryCheckStrategy* strategy, float mean, float stddev, unsigned long seed) {
    printf("threadIdx: %d, %d, %d\n", threadIdx.x, threadIdx.y, threadIdx.z);
    printf("something: %d, %d\n", s->a, s->b);

    dim3 dims = strategy->dims();
    printf("dims: %d, %d, %d\n", dims.x, dims.y, dims.z);
    printf(">>>>>");

    SizeCheck checker { strategy };
    auto idx = checker.idx();
    printf("Filling index: %d\n", idx);
    if (checker()) {
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

    something s { 42, 84 };
    something* d_s;
    cudaMalloc(&d_s, sizeof(something));
    cudaMemcpy(d_s, &s, sizeof(something), cudaMemcpyHostToDevice);

    // Launch kernel
    fillMatrixKernel<<<gridDim, blockDim>>>(
        d_mat, 
        d_s,
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
