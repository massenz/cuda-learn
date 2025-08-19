#include <assert.h>

/**
 * A linear check strategy that checks if the current index is within the bounds
 * of a given size along a one-dimensional array.
 */
class LinearCheckStragegy {
 public:
  __host__ explicit LinearCheckStragegy(int len) : size_{len} {}

  __device__ bool check() const { return (pos() < size_); }

  __device__ int pos() const {
    return blockIdx.x * blockDim.x + threadIdx.x;
  }

  __device__ dim3 dims() const { return dim3{size_, 0, 0}; }

 private:
  int size_;
};

class RectangularCheckStrategy {
 public:
  __host__ RectangularCheckStrategy(uint rows, uint cols) : 
    rows_{rows}, cols_{cols} {}

  __device__ bool check() const {
    auto p = pos();
    return (p.y < rows_ && p.x < cols_);
  }

  __device__ dim3 pos() const {
    return dim3{(blockIdx.x * blockDim.x + threadIdx.x),
                (blockIdx.y * blockDim.y + threadIdx.y)};
  }

  __device__ dim3 dims() const { return dim3{cols_, rows_}; }

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


  /**
   * Returns the linear index for the current position, based
   * on the strategy used.
   *
   * Matrices are stored in row-major order, where
   * rows are along the Y dim, and cols are along the X dim.
   *
   * Cubes are stored in depth-major order, where
   * depth is along the Z dim, rows are along the Y dim,
   * and cols are along the X dim.
   * The mental model is a stack of images, along the Z axis,
   * where each image is a matrix of size rows x cols.
   * 
   *               /_______ z =2
   *              /|  
   *             /_________ z = 1
   *            / |
   *      cols /_1_/_2_/_3__ z = 0
   *  rows 1  |___|___|___
   *       2  |___|___|___
   *    ...   |
   */
  __device__ int idx() const {
    assert(linStrategy_ || rectStrategy_ || cubeStrategy_);
    dim3 pos, dims;
    if (linStrategy_) {
      return linStrategy_->pos();
    } else if (rectStrategy_) {
      pos = rectStrategy_->pos();
      dims = rectStrategy_->dims();
      return pos.y * dims.x + pos.x;
    } else if (cubeStrategy_) {
      pos = cubeStrategy_->pos();
      dims = cubeStrategy_->dims();
      auto matSize = dims.x * dims.y;
      return pos.z * matSize + pos.y * dims.x + pos.x;
    }
    printf("No strategy set for SizeCheck, cannot compute index.\n");
    assert(false);  // Should never reach here
    return -1; // Silence compiler warning.
  }

 private:
  const LinearCheckStragegy *linStrategy_ = nullptr;
  const RectangularCheckStrategy *rectStrategy_ = nullptr;
  const CubeCheckStrategy *cubeStrategy_ = nullptr;
};
