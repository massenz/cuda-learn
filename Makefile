NVCC = nvcc
NVCC_FLAGS = -O2

TARGET = build/matrix_gen
SRC = src/mat_gen.cu
CLI_TARGET = build/cuda-learn
CLI_SRC = go-aws-cli/cmd

.PHONY: all build run clean cli vpc instance

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

# CLI tool targets
cli:
	@mkdir -p build
	@echo "--- Building CLI tool"
	cd go-aws-cli && go build -o ../$(CLI_TARGET) ./cmd

vpc: cli
	@echo "--- Setting up VPC infrastructure"
	./$(CLI_TARGET) vpc

instance: cli
	@echo "--- Setting up EC2 instance"
	./$(CLI_TARGET) instance

clean:
	@echo "--- Cleaning build directory"
	@rm -rf build
