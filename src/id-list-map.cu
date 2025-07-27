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
#include <list>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

constexpr u_int16_t CHUNK_SIZE =
    1024;  // Size of each chunk to read from the file

struct Chunk {
  int64_t values[CHUNK_SIZE];    // Array to hold the values
  uint16_t offsets[CHUNK_SIZE];  // Offsets for each value in the chunk
  uint16_t size = 0;             // Number of valid values in this chunk

  bool hasRoom(uint16_t elements) const { return size + elements < CHUNK_SIZE; }
  void print() const {
    std::cout << "Chunk size: " << size << ", values: ";
    for (uint16_t i = 0; i < size; ++i) {
        uint16_t start = offsets[i];
        uint16_t end = (i + 1 < size) ? offsets[i + 1] : CHUNK_SIZE;
        for (uint16_t j = start; j < end; ++j) {
            std::cout << values[j] << " ";
        }
      std::cout << std::endl;
    }
    std::cout << "\n";
  }
};

using ListInput = std::list<Chunk>;

/**
 * Reads a CSV file containing comma-separated int64 values and converts them
 * into a list of Chunks.
 *
 * @param filename Path to the CSV file to read
 * @return std::shared_ptr to a list of Chunks containing the parsed int64
 * values
 * @throws std::runtime_error if the file cannot be opened
 * @throws std::invalid_argument if string-to-int64 conversion fails
 */
std::shared_ptr<ListInput> readInputFile(const std::string& filename) {
  auto result = std::make_shared<ListInput>();
  std::ifstream file(filename);

  if (!file) {
    throw std::runtime_error("Could not open file: " + filename);
  }

  std::string line;
  Chunk currentChunk;
  uint16_t currentOffset = 0;

  while (std::getline(file, line)) {
    if (line.empty()) {
      continue;  // Skip empty lines
    }
    printf("Processing line: %s\n", line.c_str());
    std::vector<int64_t> row;
    std::istringstream lineStream(line);
    int64_t value;

    // As we don't know how many values there are in the line,
    // and we won't know if there is enough room in the current chunk,
    // we will first read them into a temporary vector.
    std::vector<int64_t> values;
    // Read space-separated integers until end of line
    while (lineStream >> value) {
      printf("Processing value: %ld\n", value);
      values.push_back(value);
    }
    if (values.empty()) {
      continue;  // Skip lines with no values
    }
    if (!currentChunk.hasRoom(values.size())) {
        std::cout << "Current chunk is full, creating a new one.\n";
      // If the current chunk is full, add it to the list and create a new one
      result->push_back(currentChunk);
      currentChunk = Chunk();
      currentOffset = 0;
    }
    currentChunk.offsets[currentChunk.size] = currentOffset;
    // Copy values to the current chunk
    for (size_t i = 0; i < values.size(); ++i) {
      currentChunk.values[currentOffset + i] = values[i];
    }
    currentChunk.size++;
    currentOffset += values.size();
  }
  // Adding last chunk if it has any values
    if (currentChunk.size > 0) {
        std::cout << "Adding last chunk with size: " << currentChunk.size << "\n";
    result->push_back(currentChunk);
    } else {
        std::cout << "No values in the last chunk, not adding it.\n";
    }
std::cout << "Found " << result->size() << " chunks in the file.\n";
  if (!result->empty()) {
    std::cout << "CHUNK: " << result->back().size << " values, "
              << result->back().offsets[0] << " offsets.\n";
  }


  return result;
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <input-file>\n";
    return 1;
  }

  try {
    auto data = readInputFile(argv[1]);

    // Print the data to verify contents
    for (const auto& chunk : *data) {
      chunk.print();
    }

    std::cout << "Successfully read " << data->size() << " lines\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
