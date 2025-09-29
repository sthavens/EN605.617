NUM_ITEMS ?= 
BLOCK_SIZE ?=

TARGET = assignment_$(NUM_ITEMS)_$(BLOCK_SIZE)

all: $(TARGET)

$(TARGET): assignment.cu

	nvcc $(if $(NUM_ITEMS),-DITEMS=$(NUM_ITEMS)) $(if $(BLOCK_SIZE),-DBLOCK_SIZE=$(BLOCK_SIZE)) assignment.cu -o assignment

clean:
	rm -f assignment