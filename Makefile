NVCC = nvcc
NVCC_FLAGS = -O2

TARGET = build/matrix_gen
SRC = src/mat_gen.cu

.PHONY: all build run clean

all: build run

$(TARGET): $(SRC)
	@mkdir -p build
	@echo "--- Compiling CUDA source files"
	$(NVCC) $(NVCC_FLAGS) $(SRC) -o $(TARGET)

build: $(TARGET)
	@echo "--- Build complete"

run: build
	@echo "--- Running the program"
	./$(TARGET)

clean:
	@echo "--- Cleaning build directory"
	@rm -rf build
