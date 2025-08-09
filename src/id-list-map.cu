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
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "boundaries.h"
#include "chunks.h"
#include "cudacheck.h"

void prepareBuffers(
  size_t** d_offsets,
  int64_t** d_values,
  const std::shared_ptr<Chunks>& chunks) {

  auto [sizeOffsets, sizeValues] = getTotSizeOffsets(*chunks);
  CUDA_CHECK(cudaMalloc(d_offsets, sizeOffsets * sizeof(size_t)))
      << "Failed to allocate device memory for offsets";
  CUDA_CHECK(cudaMalloc(d_values, sizeValues * sizeof(int64_t)))
      << "Failed to allocate device memory for values";

  size_t offsetsPos = 0, valuesPos = 0;
  for (const auto& chunk : *chunks) {
    CUDA_CHECK(cudaMemcpy(*d_offsets + offsetsPos, chunk.offsets,
                          chunk.size * sizeof(size_t), 
                          cudaMemcpyHostToDevice))
        << "Failed to copy offsets to device";
    CUDA_CHECK(cudaMemcpy(*d_values + valuesPos, chunk.values,
                          chunk.numValues() * sizeof(int64_t),
                          cudaMemcpyHostToDevice))
        << "Failed to copy values to device";
    offsetsPos += chunk.size;
    valuesPos += chunk.numValues();
  }
  printf("Prepared buffers: %zu offsets, %zu values\n", sizeOffsets, sizeValues);
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <input-file>\n";
    return 1;
  }

  try {
    auto data = readInputFile(argv[1]);

    // Print the data to verify contents
    for (const auto &chunk: *data) {
      chunk.PrintMetadata();
      chunk.PrintValues();
    }

    auto [samples, values] = getTotSizeOffsets(*data);
    std::cout << "Successfully read " << samples << " samples, containing "
              << values << " values in total.\n";

    std::cout << "Preparing buffers for CUDA...\n";
    size_t *d_offsets = nullptr;
    int64_t *d_values = nullptr;
    prepareBuffers(&d_offsets, &d_values, data);
    
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
