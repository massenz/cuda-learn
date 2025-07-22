#include <assert.h>

/**
 * A linear check strategy that checks if the current index is within the bounds
 * of a given size along a one-dimensional array.
 */
class LinearCheckStragegy {
 public:
  __host__ explicit LinearCheckStragegy(int len) : size_{len} {}

  __device__ bool check() const { return (pos().x < size_); }

  __device__ dim3 pos() const {
    return dim3{blockIdx.x * blockDim.x + threadIdx.x, 0, 0};
  }

  __device__ dim3 dims() const { return dim3{size_, 0, 0}; }

 private:
  int size_;
};

class RectangularCheckStrategy {
 public:
  __host__ RectangularCheckStrategy(uint m, uint n) : rows_{m}, cols_{n} {}

  __device__ bool check() const {
    auto p = pos();
    return (p.y < rows_ && p.x < cols_);
  }

  __device__ dim3 pos() const {
    return dim3{(blockIdx.x * blockDim.x + threadIdx.x),
                (blockIdx.y * blockDim.y + threadIdx.y), 0};
  }

  __device__ dim3 dims() const { return dim3{cols_, rows_, 0}; }

 private:
  uint rows_;
  uint cols_;
};

class CubeCheckStrategy {
 public:
  __host__ CubeCheckStrategy(uint rows, uint cols, uint depth)
      : size_{cols, rows, depth} {}

  __device__ bool check() const {
    auto curPos = pos();
    return (curPos.x < size_.x && curPos.y < size_.y && curPos.z < size_.z);
  }

  __device__ dim3 pos() const {
    return {(blockIdx.x * blockDim.x + threadIdx.x),
            (blockIdx.y * blockDim.y + threadIdx.y),
            (blockIdx.z * blockDim.z + threadIdx.z)};
  }

  __device__ dim3 dims() const { return size_; }

 private:
  dim3 size_;
};

enum class BoundaryType { Linear, Rectangular, Cube };

class SizeCheck {
 public:
  __device__ explicit SizeCheck(const void *strategy, BoundaryType type) {
    switch (type) {
      case BoundaryType::Linear:
        linStrategy_ = static_cast<const LinearCheckStragegy *>(strategy);
        assert(linStrategy_ != nullptr);
        break;
      case BoundaryType::Rectangular:
        rectStrategy_ = static_cast<const RectangularCheckStrategy *>(strategy);
        assert(rectStrategy_ != nullptr);
        break;
      case BoundaryType::Cube:
        cubeStrategy_ = static_cast<const CubeCheckStrategy *>(strategy);
        assert(cubeStrategy_ != nullptr);
        break;
    }
  }

  __device__ bool operator()() const {
    if (linStrategy_) {
      return linStrategy_->check();
    } else if (rectStrategy_) {
      return rectStrategy_->check();
    } else if (cubeStrategy_) {
      return cubeStrategy_->check();
    }
    assert(false);
    return false;  // Should never reach here
  }

  __device__ int idx() const {
    assert(linStrategy_ || rectStrategy_ || cubeStrategy_);
    dim3 pos, dims;
    if (linStrategy_) {
      pos = linStrategy_->pos();
      dims = linStrategy_->dims();
    } else if (rectStrategy_) {
      pos = rectStrategy_->pos();
      dims = rectStrategy_->dims();
    } else if (cubeStrategy_) {
      pos = cubeStrategy_->pos();
      dims = cubeStrategy_->dims();
    }
    auto matSize = dims.x * dims.y;
    return pos.z * matSize + pos.y * dims.x + pos.x;
  }

 private:
  const LinearCheckStragegy *linStrategy_ = nullptr;
  const RectangularCheckStrategy *rectStrategy_ = nullptr;
  const CubeCheckStrategy *cubeStrategy_ = nullptr;
};
