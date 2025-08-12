#include <iostream>
#include <iomanip>
#include <string>
#include <cstdlib>

#include "fillmatrix.h"

void printUsage() {
    std::cout << "Usage: demo <command>\n";
    std::cout << "Available commands:\n";
    std::cout << "  fillmat    Fill a matrix with random values\n";
}

void printMatrix(float* mat, uint m, uint n) {
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
    float* matrix = new float[rows * cols];

    // Fill the matrix with random values
    fill(matrix, rows, cols);

    // Print the matrix
    std::cout << "Generated " << rows << "x" << cols << " matrix:\n";
    printMatrix(matrix, rows, cols);

    // Clean up
    delete[] matrix;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printUsage();
        return 1;
    }

    std::string command(argv[1]);
    
    if (command == "fillmat") {
        fillmat();
        return 0;
    }

    std::cout << "Unknown command: " << command << std::endl;
    printUsage();
    return 1;
}
