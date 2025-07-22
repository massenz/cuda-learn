class BoundaryCheckStrategy {
public:
    __device__
    virtual bool check() const = 0;

    __device__
    virtual dim3 pos() const = 0;

    __device__
    virtual dim3 dims() const = 0;
};

/**
 * A linear check strategy that checks if the current index is within the bounds of a given size
 * along a one-dimensional array.
 */
class LinearCheckStragegy: public BoundaryCheckStrategy {
 public:
    __host__
    explicit LinearCheckStragegy(int len): size_{len} {}
    
    __device__
    bool check() const override {
        return (pos().x < size_);
    }

    __device__
    dim3 pos() const override {
        return dim3{ blockIdx.x * blockDim.x + threadIdx.x, 0, 0 };
    }

    __device__
    dim3 dims() const override{
        return dim3{ size_, 0, 0 };
    }

 private:
    int size_;
};

class RectangularCheckStrategy: public BoundaryCheckStrategy {
 public:
    __host__
    RectangularCheckStrategy(int m, int n): rows_{m}, cols_{n} {}

    __device__
    bool check() const override {
        auto p = pos();
        return (p.y < rows_ && p.x < cols_);
    }

    __device__
    dim3 pos() const override {
        return dim3 {(blockIdx.x * blockDim.x + threadIdx.x),
               (blockIdx.y * blockDim.y + threadIdx.y),
               0 };
    }

    __device__
    dim3 dims() const override {
        return dim3 { cols_, rows_, 0 };
    }

 private:
    int rows_;
    int cols_;
};

class CubeCheckStrategy: public BoundaryCheckStrategy {
 public:
    __host__
    CubeCheckStrategy(int rows, int cols, int depth): size_{ cols, rows, depth } {}

    __device__
    bool check() const override {
        auto curPos = pos();
        return (curPos.x < size_.x && curPos.y < size_.y && curPos.z < size_.z);
    }

    __device__
    dim3 pos() const override {
        return {
            (blockIdx.x * blockDim.x + threadIdx.x),
            (blockIdx.y * blockDim.y + threadIdx.y),
            (blockIdx.z * blockDim.z + threadIdx.z) };
    }

    __device__
    dim3 dims() const override {
        return size_;
    }

 private:
    dim3 size_;
};


class SizeCheck {
 public:
    __device__
    explicit SizeCheck(const BoundaryCheckStrategy* strategy): 
        strategy_{ strategy } {}

    __device__
    bool operator()() const {
        return strategy_->check();
    }

    __device__
    int idx() const {
        auto pos = strategy_->pos();
        auto dims = strategy_->dims();
        printf("pos: %d, %d, %d\n", pos.x , pos.y , pos.z);
        auto matSize = dims.x * dims.y;
        int idx = pos.z * matSize + 
                  pos.y * dims.y + 
                  pos.x;

        return idx;
    }

 private:
    const BoundaryCheckStrategy* strategy_;
};
