// Copyright (c) 2025.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0
// http://www.apache.org/licenses/LICENSE-2.0
//
// Author: Marco Massenzio (marco@alertavert.com)
#pragma once

#include <iostream>
#include <list>
#include <memory>
#include <tuple>


// Size of each chunk to read from the file
constexpr size_t CHUNK_SIZE = 15;

/**
 * A chunk of data that holds a fixed number of values and their offsets.
 * The offsets indicate where each set of features for a given sample ends,
 * as the previous offset indicates where it starts.
 *
 * This takes advantage of the fact that the first offset is always 0, while we
 * do not need to store an extra value for the end of the last sample.
 *
 * See the image in the documentation for a visual representation of how this
 * works.
 *
 * @param size the number of samples in the batch, each carrying a variable
 * number of values.
 * @param values the values of the samples in the batch.
 * @param offsets the offsets of the values in the batch, indicating where each
 * sample's values start and end within the `values` array.
 */
struct Chunk {
  // Array to hold the values
  int64_t values[CHUNK_SIZE];

  /**
   * Offsets for each sample's values in the chunk.
   * The first offset is always assumed to be 0, and is not stored.
   * 
   * For each sample, we store the _ending_ offset of the values in the
   * `values` array, so the first offset is always 0, and the last
   * offset is the total number of values in the chunk.
   */
  size_t offsets[CHUNK_SIZE];

  // The batch size, each sample in the batch having a variable number
  // of values.
  size_t size = 0;

  // Used to keep track of the initial offset of the chunk, when
  // this is not the first in the chain.
  size_t offsetAdjust = 0;

  /**
   * Checks if the chunk has enough room to add a specified number of elements.
   * @param elements the number of elements to check for room.
   * @return `true` if there is enough room, `false` otherwise.
   */
  [[nodiscard]] bool hasRoom(size_t elements) const {
    return numValues() + elements < CHUNK_SIZE;
  }

  /**
   * Returns the total number of values in the chunk; as we store
   * the _ending_ offset at each sample position, the last offset
   * indicates the total number of values in the chunk.
   * 
   * @return the total number of values.
   */
  [[nodiscard]] size_t numValues() const {
    return size > 0 ? offsets[size - 1] - offsetAdjust : 0;
  }

  void PrintMetadata(std::ostream &out = std::cout) const;

  void PrintValues(std::ostream &out = std::cout) const;
};

using Chunks = std::list<Chunk>;

/**
 * Reads a text file containing space-separated int64 values and converts them
 * into a list of Chunks.
 *
 * @param filename Path to the CSV file to read
 * @return std::shared_ptr to a list of Chunks containing the parsed int64
 * values
 * @throws std::runtime_error if the file cannot be opened
 * @throws std::invalid_argument if string-to-int64 conversion fails
 */
std::shared_ptr<Chunks> readInputFile(const std::string &filename);

/**
 * Calculates the total size and number of values in a list of Chunks.
 *
 * @param chunks The list of Chunks to analyze
 * @return A tuple containing the total number of samples (lines) read
 *         and total number of values, respectively.
 */
std::tuple<size_t, size_t> getTotSizeOffsets(const Chunks& chunks);
