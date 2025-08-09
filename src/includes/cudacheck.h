#pragma once

#include <cuda_runtime.h>
#include <sstream>
#include <stdexcept>

class CudaException : public std::runtime_error {
 public:
  explicit CudaException(const std::string& msg) : std::runtime_error(msg) {}
};

class CudaErrorBuilder {
 public:
  explicit CudaErrorBuilder(cudaError_t err, const char* expr, const char* file, int line)
      : err_(err), expr_(expr), file_(file), line_(line) {}
  
  ~CudaErrorBuilder() {
    if (err_ != cudaSuccess) {
      std::ostringstream oss;
      oss << file_ << ":" << line_ << ": CUDA error in '" << expr_ 
          << "': " << cudaGetErrorString(err_);
      if (!msg_.str().empty()) {
        oss << " - " << msg_.str();
      }
      throw CudaException(oss.str());
    }
  }
  
  template<typename T>
  CudaErrorBuilder& operator<<(const T& msg) {
    msg_ << msg;
    return *this;
  }
  
 private:
  cudaError_t err_;
  const char* expr_;
  const char* file_;
  int line_;
  std::ostringstream msg_;
};

#define CUDA_CHECK(expr) \
  CudaErrorBuilder((expr), #expr, __FILE__, __LINE__)
