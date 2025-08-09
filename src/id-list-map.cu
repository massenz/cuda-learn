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

#include "chunks.h"


int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <input-file>\n";
    return 1;
  }

  try {
    auto data = readInputFile(argv[1]);
    int lines = 0;
    int values = 0;
    // Print the data to verify contents
    for (const auto &chunk: *data) {
      chunk.PrintMetadata();
      lines += chunk.size;
      values += chunk.numValues();
      chunk.PrintValues();
    }

    std::cout << "Successfully read " << lines << " lines, containing "
              << values << " values.\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
