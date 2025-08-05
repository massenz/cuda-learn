//
// Created by Marco Massenzio on 8/4/25.
//

#pragma once

#include <iostream>
#include <list>

// Size of each chunk to read from the file
constexpr size_t CHUNK_SIZE = 1024;

/**
 * A chunk of data that holds a fixed number of values and their offsets.
 * The offsets indicate where each set of features for a given sample ends,
 * as the previous offset indicates where it starts.
 *
 * This takes advantage of the fact that the first offset is always 0, while we
 * do not need to store an extra value for the end of the last sample.
 *
 * See the image in the documentation for a visual representation of how this works.
 *
 * @param size the number of samples in the batch, each carrying a variable number of values.
 * @param values the values of the samples in the batch.
 * @param offsets the offsets of the values in the batch, indicating where each sample's values
 *                start and end within the `values` array.
 */
struct Chunk {
  int64_t values[CHUNK_SIZE]; // Array to hold the values
  size_t offsets[CHUNK_SIZE]; // Offsets for each value in the chunk
  size_t size = 0; // Number of valid values in this chunk

  /**
   * Checks if the chunk has enough room to add a specified number of elements.
   * @param elements the number of elements to check for room.
   * @return `true` if there is enough room, `false` otherwise.
   */
  [[nodiscard]] bool hasRoom(size_t elements) const { return numValues() + elements < CHUNK_SIZE; }
  [[nodiscard]] size_t numValues() const { return size > 0 ? offsets[size-1] : 0; }

  void PrintMetadata(std::ostream &out = std::cout) const {
    out << "Chunk size: " << size << " samples (" << numValues() << " values)" << std::endl;
    out << "Offsets: ";
    for (uint16_t i = 0; i < size; ++i) {
      out << offsets[i] << " ";
    }
    out << std::endl;
};

void PrintValues(std::ostream &out = std::cout) const {
    uint16_t start = 0;
    for (uint16_t i = 0; i < size; ++i) {
      out << i+1 << ": ";
      uint16_t end = offsets[i];
      for (uint16_t j = start; j < end; ++j) {
        out << values[j] << " ";
      }
      start = end;
      out << std::endl;
    }
    out << "\n";
  }
};

using ListInput = std::list<Chunk>;
