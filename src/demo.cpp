#include <iostream>
#include <iomanip>
#include <string>

#include "fillmatrix.h"

// TODO: add the REGISTER macro and wrapper functionality for demo functions.
// See: https://bitbucket.org/marco/samples/src/develop/
void printUsage() {
    std::cout << "Usage: demo <command>\n";
    std::cout << "Available commands:\n";
    std::cout << "  fillmat    Fill a matrix with random values\n";
    std::cout << "  id-list    Demo ID List features preprocessing\n";
}

void printMatrix(const float* mat, uint m, uint n) {
    std::cout << std::fixed << std::setprecision(4);
    for (uint i = 0; i < m; ++i) {
        for (uint j = 0; j < n; ++j) {
            std::cout << std::setw(8) << mat[i * n + j] << " ";
        }
        std::cout << std::endl;
    }
}

void fillmat() {
    // Create a small matrix for demonstration
    const uint rows = 10;
    const uint cols = 15;
    const auto matrix = new float[rows * cols];

    // Fill the matrix with random values
    fill(matrix, rows, cols);

    // Print the matrix
    std::cout << "Generated " << rows << "x" << cols << " matrix:\n";
    printMatrix(matrix, rows, cols);

    // Clean up
    delete[] matrix;
}

// From id-list-map.cu
void preproc(const std::string &dataFile, float** pooledFeatures, 
  size_t& numSamples, size_t nDim);

void idList(const char* dataFile) {
    try {
        float* pooledFeatures = nullptr;
        size_t numSamples;
        size_t nDim = 128; // Example dimension size, adjust as needed
        preproc(dataFile, &pooledFeatures, numSamples, nDim);
        std::cout << "Read " << numSamples << " samples in batch.\n";
        std::cout << "Pooled features:\n";
        printMatrix(pooledFeatures, numSamples, nDim);
        delete[] pooledFeatures; // Free the allocated memory
    } catch (const std::exception &e) {
        std::cerr << "Error during ID List preprocessing: " << e.what() << "\n";
    }
}


int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage();
        return 1;
    }

    std::string command(argv[1]);

    if (command == "fillmat") {
        fillmat();
        return 0;
    } else if (command == "id-list") {
        if (argc < 3) {
            std::cerr << "Usage: demo id-list <data-file>\n";
            return 1;
        }
        idList(argv[2]);
        return 0;
    }

    std::cout << "Unknown command: " << command << std::endl;
    printUsage();
    return 1;
}
