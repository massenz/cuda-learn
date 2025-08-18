// Copyright (c) 2025.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0
// http://www.apache.org/licenses/LICENSE-2.0
//
// Author: Marco Massenzio (marco@alertavert.com)

/**
 * Reads a list of IDs from a file and computes the pooling (SUM) of their
 * hashes.
 *
 * The file contains several IDs per line, each an int64 value that needs to be
 * hashed to a unique index.
 *
 * The code uses CUDA for parallel processing to handle large lists efficiently.
 * Due to the limiation of the memory model, and the fact that there are a
 * variable number of IDs per line, we cannot simply pass them 'as-is' but need
 * to pre-process them.
 *
 * Additionally, given that we do not know in advance how many IDs there are, we
 * will allocate (CPU) memory in blocks of a fixed size, and then copy the data
 * to the GPU in chunks.
 *
 * The data passed to the GPU will be a "coalesced" array, where all the IDs are
 * concatenated into a single array, and the offsets are stored in a separate
 * array (of fixed size: the number of samples in the batch).
 *
 * The CUDA kernel should return two arrays:
 *  - the list of tensors (conceptually, it is a mapping, but the array indices
 *    constitute the keys);
 *  - the sum-pooling for each sample in the batch, in the same order as they
 *    are passed.
 */
#include <cuda_runtime.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "boundaries.h"
#include "chunks.h"
#include "cudacheck.h"
#include "fillmatrix.h"


/**
 * CUDA kernel to pool embeddings for each sample.
 * This kernel sums the embeddings for each value in the sample
 * and stores the result in the pooled features array.
 *
 * It takes the ID list as two separate arrays of offsets and values,
 * an Embedding table, and returns the pooled features for each sample.
 * The `values` will be "hashed" (in practice, modulo `numTensors`) to
 * get the index in the embedding table.
 *
 * See `chunks.h` for more details on the {offsets, values} arrays.
 *
 * @param d_offsets Pointer to the device memory containing the offsets.
 * @param d_values Pointer to the device memory containing the values.
 * @param d_strategy The LinearStrategy for the kernel,
 *        contains the number of samples.
 * @param d_embeddings Pointer to the device memory containing
 *        the embedding table.
 * @param d_pooledFeatures Pointer to the device memory where pooled
 *        features will be stored.
 * @param numTensors The size (rows) of the embedding table.
 * @param nDim The number of dimensions (columns) in the embedding table.
 */
__global__ void poolEmbeddings(const size_t *d_offsets, const int64_t *d_values,
                               void *d_strategy, float *d_embeddings,
                               float *d_pooledFeatures, size_t numTensors,
                               size_t nDim) {
  SizeCheck checker{d_strategy, BoundaryType::Linear};
  if (checker()) {
    auto idx = checker.idx();
    // Get the start and end offsets for the current sample.
    size_t start = idx > 0 ? d_offsets[idx - 1] : 0;
    size_t end = d_offsets[idx];
    printf("Processing sample %d: offsets [%ld, %ld)\n", idx, start, end);
    // Initialize the pooled features for this sample.
    for (size_t i = 0; i < nDim; ++i) {
      d_pooledFeatures[idx * nDim + i] = 0.0f;
    }
    // Iterate over the values for this sample.
    for (size_t i = start; i < end; ++i) {
      // Hash the value to get the index in the embedding table.
      size_t tensorIdx = d_values[i] % numTensors;
      if (tensorIdx < 0) {
        tensorIdx = -tensorIdx;  // Ensure positive index.
      }
      // Add the embedding to the pooled features.
      for (size_t j = 0; j < nDim; ++j) {
        d_pooledFeatures[idx * nDim + j] +=
            d_embeddings[tensorIdx * nDim + j];
      }
    }
  }
}

/**
 * Prepares the buffers for the offsets and values arrays on the device.
 * Allocates memory on the device and copies the data from the host to the
 * device.
 *
 * @param d_offsets Pointer to the device memory for offsets.
 * @param d_values Pointer to the device memory for values.
 * @param chunks The shared pointer to the Chunks object containing the data.
 */
void prepareBuffers(size_t **d_offsets, int64_t **d_values,
                    const std::shared_ptr<Chunks> &chunks) {
  auto [sizeOffsets, sizeValues] = getTotSizeOffsets(*chunks);
  CUDA_CHECK(cudaMalloc(d_offsets, sizeOffsets * sizeof(size_t)))
      << "Failed to allocate device memory for offsets";
  CUDA_CHECK(cudaMalloc(d_values, sizeValues * sizeof(int64_t)))
      << "Failed to allocate device memory for values";

  size_t offsetsPos = 0, valuesPos = 0;
  for (const auto &chunk : *chunks) {
    CUDA_CHECK(cudaMemcpy(*d_offsets + offsetsPos, chunk.offsets,
                          chunk.size * sizeof(size_t), cudaMemcpyHostToDevice))
        << "Failed to copy offsets to device";
    CUDA_CHECK(cudaMemcpy(*d_values + valuesPos, chunk.values,
                          chunk.numValues() * sizeof(int64_t),
                          cudaMemcpyHostToDevice))
        << "Failed to copy values to device";
    offsetsPos += chunk.size;
    valuesPos += chunk.numValues();
  }
  printf("Prepared buffers: %zu offsets, %zu values\n", sizeOffsets,
         sizeValues);
}

/**
 * Preprocesses the input data file, reads the IDs, prepares the buffers,
 * and launches the CUDA kernel to pool embeddings.
 *
 * @param dataFile The path to the input data file containing IDs.
 * @param pooledFeatures Pointer to a pointer to host memory where
 *    pooled features will be returned, the caller is responsible for
 *    freeing the array.
 * @param numSamples Will contain the number of samples that have been read
 * @param nDim Tensors dimensions.
 */
void preproc(const std::string &dataFile, float** pooledFeatures,
  size_t& numSamples, size_t nDim) {
  try {
    auto data = readInputFile(dataFile);

    // TODO: keeping it here for now, but we should remove
    // this code before merging to main.
    // for (const auto &chunk: *data) {
    //   chunk.PrintMetadata();
    //   chunk.PrintValues();
    // }

    auto [samples, values] = getTotSizeOffsets(*data);
    numSamples = samples;
    std::cout << "Successfully read " << numSamples << " samples, containing "
              << values << " values in total.\n";

    std::cout << "Moving host data to CUDA Global Memory\n";
    size_t *d_offsets = nullptr;
    int64_t *d_values = nullptr;
    prepareBuffers(&d_offsets, &d_values, data);

    // Creating the lookup tables.
    // TODO: for now fixed size, but we should make it dynamic.
    const size_t numTensors = 512;  // The embedding table size.
    float *d_embeddings;
    CUDA_CHECK(cudaMalloc(&d_embeddings, numTensors * nDim * sizeof(float)))
        << "Failed to allocate device memory for embeddings";

    // The pooled features batch, returned as an array of
    // `samples` tensors, each of size `nDim`.
    float *d_pooledFeatures = nullptr;
    CUDA_CHECK(cudaMalloc(&d_pooledFeatures, numSamples * nDim * sizeof(float)))
        << "Failed to allocate device memory for pooled features";

    std::cout << "Creating the Embedding Lookup Table\n";
    fillMatrixLaunchKernel(d_embeddings, numTensors, nDim, 0.0f, 1.0f);

    std::cout << "Launching the CUDA kernel to pool embeddings\n";
    LinearCheckStragegy strategy(numSamples);
    LinearCheckStragegy *d_strategy;
    CUDA_CHECK(cudaMalloc(&d_strategy, sizeof(LinearCheckStragegy)))
        << "Failed to allocate device memory for strategy";
    CUDA_CHECK(cudaMemcpy(d_strategy, &strategy, sizeof(LinearCheckStragegy),
                         cudaMemcpyHostToDevice))
        << "Failed to copy strategy to device";
    uint threadsPerBlock = 256;  // Number of threads per block.
    uint blocks = (numSamples + threadsPerBlock - 1) / threadsPerBlock;
    printf("Launching kernel with %u blocks of %u threads each\n", blocks,
           threadsPerBlock);
    poolEmbeddings<<<blocks, threadsPerBlock>>>(
      d_offsets, d_values, d_strategy, d_embeddings,
      d_pooledFeatures, numTensors, nDim);

    std::cout << "Copying pooled features back to host ("
              << numSamples << " x " << nDim << " floats)\n";
    *pooledFeatures = new float[numSamples * nDim];
    CUDA_CHECK(cudaMemcpy(*pooledFeatures, d_pooledFeatures,
                          numSamples * nDim * sizeof(float),
                          cudaMemcpyDeviceToHost))
        << "Failed to copy pooled features to host";
    cudaFree(d_strategy);
    cudaFree(d_pooledFeatures);
    cudaFree(d_embeddings);
    cudaFree(d_values);
    cudaFree(d_offsets);
  } catch (const std::exception &e) {
    std::cerr << "Could not preproc data file " << dataFile << ": " << e.what()
              << "\n";
  }
}
