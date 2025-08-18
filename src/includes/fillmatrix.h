// Copyright (c) 2025.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0
// http://www.apache.org/licenses/LICENSE-2.0
//
// Author: Marco Massenzio (marco@alertavert.com)
#pragma once
#include <sys/types.h>
#include <cuda_runtime.h>


/**
 * @file mat_gen.cu
 * @brief CUDA kernel to fill a matrix with random values from a normal distribution.
 *
 * This file contains the implementation of a CUDA kernel that fills a matrix
 * with random values drawn from a normal distribution using the CURAND library.
 * The kernel is designed to work with matrices of arbitrary size and allows
 * for specifying the mean and standard deviation of the distribution.
 *
 * The strategy used here to check for the boundaries of the matrix is a `RectangularCheckStrategy`.
 *
 * @param mat Pointer to the matrix to fill, must have enough space allocated.
 * @param strategy Pointer to a strategy object that defines the boundary checks.
 * @param mean Mean of the normal distribution.
 * @param stddev Standard deviation of the normal distribution.
 * @param seed Seed for the random number generator.
 */
__global__
void fillMatrixKernel(float* mat, void* strategy, float mean, float stddev, unsigned long seed);

/**
 * Fills a matrix with random values drawn from a normal distribution.
 *
 * Example usage:
 *     float* matrix = new float[M * N];
 *     // Using default mean=0.0 and stddev=1.0
 *     fill(matrix, M, N);
 *
 * @param mat Pointer to the matrix to fill, must have enough space allocated.
 * @param m Number of rows in the matrix.
 * @param n Number of columns in the matrix.
 * @param mean Mean of the normal distribution.
 * @param stddev Standard deviation of the normal distribution.
 */
void fill(float* mat, uint m, uint n, float mean = 0.0f, float stddev = 1.0f);
