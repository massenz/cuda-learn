//
// Created by Marco Massenzio on 7/17/25.
//
#include <stdio.h>
#include <cuda_runtime.h>

int main() {
  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);
  for (int i = 0; i < deviceCount; ++i) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device %d: %s\n", i, prop.name);
    printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
    printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("  Max Threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("  Shared Memory per SM: %zu bytes\n", prop.sharedMemPerMultiprocessor);
    printf("  Total Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
  }
  return 0;
}
