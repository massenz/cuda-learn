// Copyright (c) 2025.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0
// http://www.apache.org/licenses/LICENSE-2.0
//
// Author: Marco Massenzio (marco@alertavert.com)
#include "chunks.h"

#include <fstream>
#include <sstream>
#include <vector>


void Chunk::PrintMetadata(std::ostream &out) const {
  out << "Chunk size: " << size << " samples (" << numValues() << " values)"
      << std::endl;
  out << "Offsets: ";
  for (uint16_t i = 0; i < size; ++i) {
    out << offsets[i] << " ";
  }
  out << std::endl;
}

void Chunk::PrintValues(std::ostream &out) const {
  uint16_t start = 0;
  for (uint16_t i = 0; i < size; ++i) {
    out << i + 1 << ": ";
    uint16_t end = offsets[i] - offsetAdjust;
    for (uint16_t j = start; j < end; ++j) {
      out << values[j] << " ";
    }
    start = end;
    out << std::endl;
  }
  out << "\n";
}

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
std::shared_ptr<Chunks> readInputFile(const std::string &filename) {
  auto result = std::make_shared<Chunks>();
  std::ifstream file(filename);

  if (!file) {
    throw std::runtime_error("Could not open file: " + filename);
  }

  std::string line;
  Chunk currentChunk;
  uint16_t currentOffset = 0;

  while (std::getline(file, line)) {
    if (line.empty()) {
      continue; // Skip empty lines
    }
    std::istringstream lineStream(line);
    int64_t value;

    // As we don't know how many values there are in the line,
    // and we won't know if there is enough room in the current chunk,
    // we will first read them into a temporary vector.
    std::vector<int64_t> values;
    // Read space-separated integers until end of line
    while (lineStream >> value) {
      values.push_back(value);
    }
    if (values.empty()) {
      continue; // Skip lines with no values
    }
    if (!currentChunk.hasRoom(values.size())) {
      result->push_back(currentChunk);
      // NOTE we don't reset the currentOffset here, as it is used to
      // calculate the offsets for the next chunk.
      // This will simplify moving the data to the GPU, as we can just
      // copy the values in the GPU memory without having to modify the offsets.
      currentChunk = Chunk();
      currentChunk.offsetAdjust = currentOffset;
    }
    currentChunk.offsets[currentChunk.size] = currentOffset + values.size();
    // Copy values to the current chunk
    for (size_t i = 0; i < values.size(); ++i) {
      currentChunk.values[currentOffset - currentChunk.offsetAdjust + i] = values[i];
    }
    currentChunk.size++;
    currentOffset += values.size();
  }
  // Adding last chunk if it has any values
  if (currentChunk.size > 0) {
    result->push_back(currentChunk);
  }
  return result;
}



// Return type is std::tuple<type1, type2>
std::tuple<size_t, size_t> getTotSizeOffsets(const Chunks& chunks) {
    size_t total_size = 0;
    size_t total_values = 0;
    
    for (const auto& chunk : chunks) {
        total_size += chunk.size;
        total_values += chunk.numValues();
    }
    return {total_size, total_values};  // C++17 allows this simple return
}