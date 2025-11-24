# Compiler and flags
CXX := g++
CXXFLAGS := -std=c++11 -O2
TARGET := matrixMul
SRC := assignment.cpp

# Detect platform
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    # macOS
    LDFLAGS := -framework OpenCL
else
    # Linux/Windows (assuming OpenCL headers and libraries are in standard paths)
    LDFLAGS := -lOpenCL
endif

.PHONY: all build clean run

all: build

build: $(SRC)
	@echo "Building $(TARGET) for $(UNAME_S)..."
	$(CXX) $(CXXFLAGS) $^ -o $(TARGET) $(LDFLAGS)

clean:
	@echo "Cleaning..."
	rm -f $(TARGET)

run: build
	@echo "Running $(TARGET)..."
	./$(TARGET)
