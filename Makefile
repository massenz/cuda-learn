NVCC = nvcc
NVCC_FLAGS = -O2

TARGET = build/matrix_gen
SRC = src/mat_gen.cu

.PHONY: all build run clean

all: build run

build: $(SRC)
	@mkdir -p build
	@echo "--- Compiling CUDA source files"
	$(NVCC) $(NVCC_FLAGS) $(SRC) -o $(TARGET)

run: build
	@echo "--- Running the program"
	./$(TARGET)

clean:
	@echo "--- Cleaning build directory"
	@rm -rf build
