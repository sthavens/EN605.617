//Based on the work of Andrew Krepps
#include <stdio.h>
#include <unordered_map>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef BUFFER_SIZE
#define BUFFER_SIZE 500 * 1024 * 1024 // 500MB buffer by default
#endif

__device__ void count_characters(char* input, std::unordered_map<char, size_t>& character_counts, size_t length) {
	int threadId = threadIdx.x + blockIdx.x * blockDim.x;
	if (threadId < length) {
		char c = input[threadId];
		atomicAdd(&character_counts[c], 1);
	}
}

__host__ void process_file(FILE* input, char* output, size_t length) {
	cudaEvent_t start, counted, compressed;
	std::unordered_map<char, size_t> character_counts;
	cudaEventCreate(&start);

	void* buffer = malloc(sizeof(char) * BUFFER_SIZE);
	size_t read_bytes = fread(buffer, 1, BUFFER_SIZE, input);

	count_characters<<<1, 256>>>(input, character_counts, length);
	cudaEventCreate(&counted);



}

int main(int argc, char** argv)
{
	// read command line arguments
	int totalThreads = (1 << 20);
	int blockSize = 256;
	
	if (argc >= 2) {
		totalThreads = atoi(argv[1]);
	}
	if (argc >= 3) {
		blockSize = atoi(argv[2]);
	}

	int numBlocks = totalThreads/blockSize;

	// validate command line arguments
	if (totalThreads % blockSize != 0) {
		++numBlocks;
		totalThreads = numBlocks*blockSize;
		
		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("The total number of threads will be rounded up to %d\n", totalThreads);
	}
}
