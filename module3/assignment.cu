#include "assignment.h"

__global__ void divisible_by_3(int * block) {
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	block[thread_idx] = thread_idx % 3 == 0 ? 1 : 0;
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

	unsigned int cpu_block[totalThreads];
	int * gpu_block;

	// validate command line arguments
	if (totalThreads % blockSize != 0) {
		++numBlocks;
		totalThreads = numBlocks*blockSize;
		
		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("The total number of threads will be rounded up to %d\n", totalThreads);
	}

	cudaMalloc((void **) &gpu_block, totalThreads * sizeof(unsigned int));
	// cudaMemcpy(gpu_block, cpu_block, sizeof(unsigned int) * totalThreads, cudaMemcpyHostToDevice);

	divisible_by_3<<<numBlocks, blockSize>>>(gpu_block);

	cudaMemcpy(cpu_block, gpu_block, sizeof(unsigned int) * totalThreads, cudaMemcpyDeviceToHost);
	cudaFree(gpu_block);

	for (unsigned int i = 0; i < totalThreads; i++) {
		const char * conditional = cpu_block[i] == 0 ? "is not" : "is"; 
		printf("%d %s divisible by 3.\n", i, conditional);
	}
	return 0;
}
